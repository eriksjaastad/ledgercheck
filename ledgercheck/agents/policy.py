"""Policy stage: check an invoice against local vendor and tax-code records.

Corpus
------
The "RAG" store is a handful of JSON chunk files in
``ledgercheck/fixtures/policy/`` (packaged data, no vector DB, no embeddings).
Each chunk has a ``chunk_id``, a ``kind`` (``"vendor"`` or ``"tax"``), free
``text`` and structured fields: a vendor carries ``vendor_id``, ``name``,
``aliases``, ``currency``, ``tax_code`` and ``po_required``; a tax code carries
``code`` and ``rate`` (a decimal string or integer; JSON floats and non-finite
values are rejected). String fields must be non-blank strings, ``aliases`` a
list of them and ``po_required`` a boolean. A vendor's ``currency`` must follow
the same rule as ``Invoice.currency`` (three uppercase letters A-Z, no
surrounding whitespace), and a ``rate`` must be a fraction in [0, 1), so a
negative rate or a percent-style one such as ``8.25`` is rejected; either
error names the chunk. ``PolicyCorpus.load`` rejects a
malformed corpus (invalid JSON, wrong types, missing keys, duplicate chunk ids,
vendor ids or tax codes) with ``PolicyCorpusError`` rather than skipping chunks.
``PolicyCorpus`` itself also rejects a vendor name or alias that normalizes to
the empty string (e.g. ``"Inc."`` or ``"&"``).

Retrieval
---------
``PolicyCorpus.find_vendor`` looks a vendor up by exact ``vendor_id`` when the
document states one, and by normalized name against the name and aliases
only when it does not. ``normalize_name`` applies NFKC and case folding, reads
``&`` as ``and``, deletes periods and apostrophes inside a word (so
``L.L.C.`` is ``llc`` and ``O'Neil`` is ``oneil``), splits on every other
non-alphanumeric character, drops ``and`` and then strips trailing legal
suffixes (LLC/Ltd/Inc/Co/Corp/GmbH). This is a heuristic first pass, not
fuzzy matching: some spellings of one name normalize differently (spaced vs
joined initials, a period vs a hyphen between words, a dotted suffix with
inner spaces or glued to another suffix); ``normalize_name`` lists these
"Known limits", and such names need an alias. A stated id that is not on file is a
miss: it never falls back to the name, so a wrong id is never papered over. A blank or
whitespace-only ``vendor_id`` counts as absent, and a document name that
normalizes to the empty string never matches by name. Two vendors whose
normalized names or aliases collide are rejected with ``PolicyCorpusError``,
so a name lookup can never depend on file order. Fuzzy matches
never count as a vendor match: ``search`` (keyword overlap) only names the
closest records in an unknown-vendor hit, so a reviewer has somewhere to look.

Rules (``rule_id`` → severity)
------------------------------
- ``POL-VENDOR-UNKNOWN`` ERROR: no vendor record for the document's stated
  (non-blank) vendor id, or, when it states none, for its vendor name; a stated
  id not on file is a miss even if the name is on file. The message, ``field``
  and ``observed`` name the vendor id when the document has a non-blank one,
  otherwise the vendor name.
- ``POL-VENDOR-NAME`` ERROR: the vendor id is on file but the name is neither
  the record's name nor an alias.
- ``POL-VENDOR-NO-ID`` INFO: matched by name; the document has no (or a blank)
  vendor id.
- ``POL-VENDOR-ALIAS`` INFO: billed under an alias, i.e. a name that still
  differs from the record's name after ``normalize_name``. Names that are equal
  after ``normalize_name`` do not trigger it; see its "Known limits" for
  spellings that normalize differently.
- ``POL-CURRENCY`` ERROR: currency differs from the vendor's billing currency.
- ``POL-PO-REQUIRED`` WARNING: vendor requires a PO and none is stated; an
  empty or whitespace-only PO number counts as none (credit notes, i.e.
  negative totals, are exempt).
- ``POL-TAX-RATE`` ERROR: stated tax rate differs from the vendor's tax code.
- ``POL-TAX-ROUNDING`` WARNING: charged tax differs from subtotal × code rate
  (rounded half-up to the cent) by more than zero and at most one cent.
- ``POL-TAX-AMOUNT`` ERROR: charged tax differs by more than one cent.

A clean invoice from a known vendor yields no hits at all. Subtotal/line-item
arithmetic is not a policy rule; it stays with the approval stage.

Rerank (off)
------------
``LEDGERCHECK_POLICY_RERANK=1`` turns on reranking of the ``search``
candidates for unknown vendors. With no injected ``reranker`` that builds
``llm_client.LLMClient``, which still needs ``LEDGERCHECK_LLM=1`` and an API
key and raises ``LiveLLMDisabled`` otherwise; even then its ``rerank`` raises
``NotImplementedError`` without a request. Reranking only reorders the
"closest on file" hint; it can never turn a miss into a vendor match.

Recording on a run
------------------
``record_policy(store, run_id, result_or_hits)`` appends a ``PolicyResult``
as the run's policy step::

    extraction = IntakeAgent().extract("tax_mismatch")
    store = RunStore()
    record_intake(store, extraction)
    record_policy(store, extraction.run_id, PolicyAgent().check(extraction))
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field, replace
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol, Sequence

from ledgercheck.agents.llm_client import LLMClient
from ledgercheck.connections import env_flag
from ledgercheck.models import (
    CENT,
    CURRENCY_RE,
    ExtractionResult,
    Invoice,
    PolicyHit,
    Severity,
    _decimal,
    _str_tuple,
)
from ledgercheck.run_store import RunRecord, RunStore, Stage

POLICY_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "policy"
RERANK_FLAG = "LEDGERCHECK_POLICY_RERANK"

_SUFFIXES = frozenset({"llc", "ltd", "inc", "co", "corp", "gmbh"})
_STOPWORDS = frozenset({"and", "the", "of", "for", "in", "a"}) | _SUFFIXES
_VENDOR_KEYS = {"vendor_id", "name", "aliases", "currency", "tax_code", "po_required"}
_TAX_KEYS = {"code", "rate"}
_VENDOR_STRS = ("vendor_id", "name", "currency", "tax_code")


class PolicyCorpusError(ValueError):
    """The policy corpus is missing or malformed."""


_IN_WORD_MARK = re.compile(r"(?<=[^\W_])[.'’](?=[^\W_])")


def _tokens(text: str, *, join_in_word: bool = False) -> list[str]:
    """Split ``text`` into words, in this order:

    1. Unicode NFKC normalization, so composed and decomposed forms agree
       (``"Mu\\u0308ller"`` == ``"Müller"``).
    2. Case folding (any script).
    3. ``&`` becomes the word ``and``.
    4. Only with ``join_in_word``: a period or apostrophe (``'`` or ``’``) with
       a letter or digit on both sides is deleted, joining its neighbours
       (``"l.l.c."`` -> ``"llc."``, ``"o'neil"`` -> ``"oneil"``).
    5. Words are the runs of letters and digits; everything else (whitespace,
       hyphens, commas, ``_``, a word-final period, ...) separates them.
    """
    text = unicodedata.normalize("NFKC", text).casefold().replace("&", " and ")
    if join_in_word:
        text = _IN_WORD_MARK.sub("", text)
    return re.findall(r"[^\W_]+", text)


def normalize_name(name: str) -> str:
    """The name's words (see ``_tokens``), minus ``and`` and trailing legal suffixes.

    A deterministic, heuristic first pass, not fuzzy matching. Steps, in order:

    1. Unicode NFKC normalization.
    2. Case folding.
    3. ``&`` is rewritten as the word ``and``.
    4. A period or apostrophe (``'`` or ``’``) with a letter or digit on both
       sides is deleted, joining its neighbours.
    5. Every other non-alphanumeric character (whitespace, hyphen, comma,
       ``_``, a period not between two letters or digits, ...) separates words.
    6. Every ``and`` word is dropped.
    7. Legal suffixes (llc, ltd, inc, co, corp, gmbh) are stripped from the end
       only, several in a row if need be; the rest is joined with single spaces.

    So ``"Harbor & Pine Consulting L.L.C."`` -> ``"harbor pine consulting"``,
    ``"Kestrel Industrial G.m.b.H."`` -> ``"kestrel industrial"`` and
    ``"Acme Co. Ltd"`` -> ``"acme"``, while a suffix elsewhere stays
    (``"Co-op Foods"`` -> ``"co op foods"``). A name of nothing but suffixes,
    ``&``/``and`` or punctuation normalizes to ``""``.

    Known limits (such names need an alias in the corpus):

    - Spaced initials and joined initials normalize differently:
      ``"J. P. Morgan"`` -> ``"j p morgan"`` but ``"J.P. Morgan"`` ->
      ``"jp morgan"``.
    - A period joins words while a hyphen or comma separates them:
      ``"Tallow.Print"`` -> ``"tallowprint"`` but ``"Tallow-Print"`` ->
      ``"tallow print"``.
    - A dotted suffix written with inner spaces (``"Acme L. L. C."`` ->
      ``"acme l l c"``) or glued to another suffix (``"Acme Co.Ltd"`` ->
      ``"acme coltd"``) is not stripped.
    """
    words = [t for t in _tokens(name, join_in_word=True) if t != "and"]
    while words and words[-1] in _SUFFIXES:
        words.pop()
    return " ".join(words)


def _stated(value: str | None) -> str | None:
    """``value`` without leading or trailing whitespace, or ``None`` when it is
    missing, empty or whitespace only.

    Stripped so a padded PO number (``" PO-4500-1282 "``) matches its PO-book
    key exactly; inner whitespace and case are kept, so ``"PO 4500-1282"`` or
    ``"po-4500-1282"`` still miss the book.
    """
    if value is None:
        return None
    return value.strip() or None


def _freeze(value: Any) -> Any:
    """Return ``value`` with lists as tuples and mappings as read-only views, recursively."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _text_leaves(value: Any) -> Iterable[str]:
    """Yield every ``str`` leaf, and every ``Decimal`` leaf as a string, in a
    (frozen) structured value; booleans and other leaves are skipped."""
    if isinstance(value, Mapping):
        for v in value.values():
            yield from _text_leaves(v)
    elif isinstance(value, tuple):
        for v in value:
            yield from _text_leaves(v)
    elif isinstance(value, (str, Decimal)):
        yield str(value)


@dataclass(frozen=True, slots=True)
class PolicyChunk:
    """One retrievable record; ``data`` holds its structured fields.

    ``data`` is deep-frozen on construction: mappings become read-only views
    and lists become tuples (so ``aliases`` is a tuple), at every level.
    """

    chunk_id: str
    kind: str
    text: str
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", _freeze(self.data))


class PolicyCorpus:
    """In-memory vendor and tax-code chunks with exact and keyword lookup."""

    def __init__(self, chunks: Iterable[PolicyChunk]) -> None:
        self.chunks = tuple(chunks)
        self._by_id: dict[str, PolicyChunk] = {}
        self._vendors: dict[str, PolicyChunk] = {}
        self._tax: dict[str, PolicyChunk] = {}
        for c in self.chunks:
            if c.chunk_id in self._by_id:
                raise PolicyCorpusError(f"duplicate chunk_id {c.chunk_id!r}")
            self._by_id[c.chunk_id] = c
            if c.kind == "vendor":
                if c.data["vendor_id"] in self._vendors:
                    raise PolicyCorpusError(f"duplicate vendor_id {c.data['vendor_id']!r}")
                self._vendors[c.data["vendor_id"]] = c
            elif c.kind == "tax":
                if c.data["code"] in self._tax:
                    raise PolicyCorpusError(f"duplicate tax code {c.data['code']!r}")
                self._tax[c.data["code"]] = c
            else:
                raise PolicyCorpusError(f"{c.chunk_id}: unknown chunk kind {c.kind!r}")
        names: dict[str, PolicyChunk] = {}
        for v in self._vendors.values():
            if v.data["tax_code"] not in self._tax:
                raise PolicyCorpusError(f"{v.chunk_id}: unknown tax_code {v.data['tax_code']!r}")
            for raw in (v.data["name"], *v.data["aliases"]):
                if not normalize_name(raw):
                    raise PolicyCorpusError(
                        f"{v.chunk_id}: name or alias {raw!r} normalizes to an empty name"
                    )
            for name in _vendor_names(v):
                if (other := names.setdefault(name, v)) is not v:
                    raise PolicyCorpusError(
                        f"{v.chunk_id}: name or alias {name!r} also matches {other.chunk_id}"
                    )

    @classmethod
    def load(cls, root: Path = POLICY_DIR) -> PolicyCorpus:
        """Load every ``*.json`` chunk list directly in ``root`` (not recursive)."""
        files = sorted(Path(root).glob("*.json")) if Path(root).is_dir() else []
        if not files:
            raise PolicyCorpusError(f"no policy corpus files in {root}")
        chunks = []
        for path in files:
            try:
                raws = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise PolicyCorpusError(f"{path.name}: invalid JSON: {exc}") from exc
            if not isinstance(raws, list):
                raise PolicyCorpusError(f"{path.name}: expected a list of chunks")
            for raw in raws:
                chunks.append(_parse_chunk(raw, path.name))
        return cls(chunks)

    def get(self, chunk_id: str) -> PolicyChunk:
        return self._by_id[chunk_id]

    def tax_code(self, code: str) -> PolicyChunk:
        return self._tax[code]

    def find_vendor(self, vendor_id: str | None, name: str) -> tuple[PolicyChunk | None, str]:
        """Return ``(chunk, how)`` with ``how`` in ``"id"``, ``"name"``, ``""`` (miss).

        An id on the document is authoritative: if it is not on file this is a
        miss even when the name would match, so a wrong id is never papered over.
        A blank or whitespace-only id counts as no id. A name that normalizes to
        ``""`` is a miss.
        """
        if _stated(vendor_id) is not None:
            chunk = self._vendors.get(vendor_id)
            return (chunk, "id") if chunk is not None else (None, "")
        wanted = normalize_name(name)
        if not wanted:
            return None, ""
        for chunk in self._vendors.values():
            if wanted in _vendor_names(chunk):
                return chunk, "name"
        return None, ""

    def search(self, query: str, *, kind: str | None = None, k: int = 3) -> list[PolicyChunk]:
        """Top ``k`` chunks by share of query keywords found in the chunk.

        A chunk's keywords come from its ``text`` plus every string value in
        ``data``, at any depth (vendor id, name, aliases, currency and tax codes),
        and every ``Decimal`` value (a tax code's rate). Booleans such as
        ``po_required`` and other non-text values are not keywords. Query words
        that are stopwords (``and``, ``the``, legal suffixes, ...) are not
        keywords, so a query with no keywords returns ``[]``. Chunks sharing no
        keyword are left out; ties sort by ``chunk_id``. ``k`` must not be
        negative (``ValueError``); ``k=0`` returns ``[]``.
        """
        if k < 0:
            raise ValueError(f"k must not be negative, got {k}")
        words = {t for t in _tokens(query) if t not in _STOPWORDS}
        scored = []
        for c in self.chunks:
            if kind is not None and c.kind != kind:
                continue
            hay = set(_tokens(" ".join([c.text, *_text_leaves(c.data)])))
            score = len(words & hay) / len(words) if words else 0.0
            if score > 0:
                scored.append((-score, c.chunk_id, c))
        return [c for _, _, c in sorted(scored, key=lambda s: s[:2])[:k]]


def _vendor_names(chunk: PolicyChunk) -> set[str]:
    return {normalize_name(n) for n in (chunk.data["name"], *chunk.data["aliases"])}


def _parse_chunk(raw: Any, where: str) -> PolicyChunk:
    if not isinstance(raw, Mapping):
        raise PolicyCorpusError(f"{where}: chunk must be an object, got {raw!r}")
    kind = raw.get("kind")
    if kind == "vendor":
        needed = _VENDOR_KEYS
    elif kind == "tax":
        needed = _TAX_KEYS
    else:
        raise PolicyCorpusError(f"{where}: unknown chunk kind {kind!r}")
    data = {k: v for k, v in raw.items() if k not in ("chunk_id", "kind", "text")}
    if missing := (needed | {"chunk_id", "text"}) - set(raw):
        raise PolicyCorpusError(f"{where}: {raw.get('chunk_id')!r} missing {sorted(missing)}")
    _require_str(raw["chunk_id"], "chunk_id", where)
    where = f"{where}: {raw['chunk_id']}"
    _require_str(raw["text"], "text", where)
    if kind == "vendor":
        for key in _VENDOR_STRS:
            _require_str(data[key], key, where)
        aliases = data["aliases"]
        if not isinstance(aliases, list):
            raise PolicyCorpusError(f"{where}: aliases must be a list, got {aliases!r}")
        for alias in aliases:
            _require_str(alias, "aliases", where)
        if not CURRENCY_RE.fullmatch(data["currency"]):
            raise PolicyCorpusError(
                f"{where}: currency must be three uppercase letters A-Z, "
                f"got {data['currency']!r}"
            )
        if not isinstance(data["po_required"], bool):
            raise PolicyCorpusError(
                f"{where}: po_required must be a boolean, got {data['po_required']!r}"
            )
    else:
        _require_str(data["code"], "code", where)
        try:
            data["rate"] = _decimal(data["rate"], "rate")
        except (TypeError, ValueError) as exc:
            raise PolicyCorpusError(f"{where}: bad rate {data['rate']!r}") from exc
        if not Decimal(0) <= data["rate"] < 1:
            raise PolicyCorpusError(
                f"{where}: rate must be a fraction in [0, 1), got {data['rate']}"
            )
    return PolicyChunk(raw["chunk_id"], kind, raw["text"], data)


def _require_str(value: Any, key: str, where: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PolicyCorpusError(f"{where}: {key} must be a non-blank string, got {value!r}")


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: Sequence[str]) -> Sequence[str]: ...


@dataclass(frozen=True, slots=True)
class PolicyResult:
    """Policy stage output: hits plus the ids of the chunks retrieved for them.

    ``retrieved`` depends on the outcome. For a vendor found on file it is
    ``(vendor chunk id, tax chunk id)``, the records the rules were checked
    against. For an unknown vendor nothing was checked: it holds the keyword
    ``search`` candidates (at most three) behind the "closest on file" hint,
    best first (after reranking, if a rerank ran), and is empty when no vendor
    shares a keyword with the name. It must be a sequence of strings; a bare
    string or a non-string item raises ``TypeError``.

    ``run_id`` is ``None`` when ``check`` was given a bare ``Invoice`` without
    one; ``record_policy`` fills it in. ``reranker`` names the reranker that
    reordered ``retrieved``. It is ``None`` whenever no rerank ran: reranking
    is off, the vendor was found on file (nothing to rerank), or the unknown
    vendor's name shares no keyword with any vendor (no candidates).
    """

    run_id: str | None
    hits: tuple[PolicyHit, ...] = ()
    retrieved: tuple[str, ...] = ()
    reranker: str | None = None

    def __post_init__(self) -> None:
        hits = tuple(self.hits)
        for h in hits:
            if not isinstance(h, PolicyHit):
                raise TypeError(f"hits: expected PolicyHit, got {type(h).__name__}")
        object.__setattr__(self, "hits", hits)
        object.__setattr__(self, "retrieved", _str_tuple(self.retrieved, "retrieved"))

    @property
    def errors(self) -> tuple[PolicyHit, ...]:
        return tuple(h for h in self.hits if h.severity is Severity.ERROR)


class PolicyAgent:
    """Runs the policy rules (see module docstring) for one invoice.

    ``rerank`` defaults to the ``LEDGERCHECK_POLICY_RERANK`` env flag (on only
    when exactly ``"1"``); an explicit ``rerank`` must be a ``bool``
    (``TypeError`` otherwise, so ``"0"`` cannot switch it on). ``reranker``
    reorders the unknown-vendor candidates when reranking is on and there are
    any; leave it ``None`` to go through the ``LLMClient`` spend gate.
    """

    def __init__(
        self,
        corpus: PolicyCorpus | None = None,
        *,
        rerank: bool | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        if rerank is not None and not isinstance(rerank, bool):
            raise TypeError(f"rerank: expected a bool or None, got {rerank!r}")
        self.corpus = PolicyCorpus.load() if corpus is None else corpus
        self.rerank = env_flag(RERANK_FLAG) if rerank is None else rerank
        self._reranker = reranker

    def check(
        self, subject: Invoice | ExtractionResult, *, run_id: str | None = None
    ) -> PolicyResult:
        """Check ``subject``; an ``ExtractionResult`` supplies its own ``run_id``.

        Passing a ``run_id`` that differs from the extraction's raises
        ``ValueError``.
        """
        if isinstance(subject, ExtractionResult):
            if run_id is not None and run_id != subject.run_id:
                raise ValueError(f"run_id {run_id!r} does not match extraction {subject.run_id!r}")
            run_id, invoice = subject.run_id, subject.invoice
        else:
            invoice = subject
        vendor, how = self.corpus.find_vendor(invoice.vendor_id, invoice.vendor_name)
        if vendor is None:
            return self._unknown_vendor(invoice, run_id)
        tax = self.corpus.tax_code(vendor.data["tax_code"])
        hits = _vendor_hits(invoice, vendor, how) + _tax_hits(invoice, tax)
        return PolicyResult(run_id, hits, (vendor.chunk_id, tax.chunk_id))

    def _unknown_vendor(self, invoice: Invoice, run_id: str | None) -> PolicyResult:
        query = invoice.vendor_name
        ids = [c.chunk_id for c in self.corpus.search(query, kind="vendor")]
        reranker = None
        if self.rerank and ids:
            client = self._reranker if self._reranker is not None else LLMClient()
            reranked = list(client.rerank(query, ids))
            if sorted(reranked) != sorted(ids):
                raise ValueError(f"{client.name} rerank must return a permutation of {ids}")
            ids, reranker = reranked, client.name
        closest = f"; closest on file: {self.corpus.get(ids[0]).data['name']}" if ids else ""
        if _stated(invoice.vendor_id) is not None:
            key, observed, missing = "vendor_id", invoice.vendor_id, "vendor id"
        else:
            key, observed, missing = "vendor_name", invoice.vendor_name, "vendor"
        hit = PolicyHit(
            "POL-VENDOR-UNKNOWN",
            Severity.ERROR,
            f"{missing} {observed!r} is not on file{closest}",
            field=key,
            observed=observed,
        )
        return PolicyResult(run_id, (hit,), tuple(ids), reranker)


def _vendor_hits(invoice: Invoice, vendor: PolicyChunk, how: str) -> tuple[PolicyHit, ...]:
    v = vendor.data
    hits = []
    if normalize_name(invoice.vendor_name) not in _vendor_names(vendor):
        hits.append(PolicyHit(
            "POL-VENDOR-NAME", Severity.ERROR,
            f"{v['vendor_id']} is on file as {v['name']!r}, not {invoice.vendor_name!r}",
            field="vendor_name", expected=v["name"], observed=invoice.vendor_name,
        ))
    else:
        if how == "name":
            hits.append(PolicyHit(
                "POL-VENDOR-NO-ID", Severity.INFO,
                f"no vendor id on the document; matched {v['vendor_id']} by name",
                field="vendor_id", expected=v["vendor_id"],
            ))
        if normalize_name(invoice.vendor_name) != normalize_name(v["name"]):
            hits.append(PolicyHit(
                "POL-VENDOR-ALIAS", Severity.INFO,
                f"billed as {invoice.vendor_name!r}, an alias of {v['name']!r}",
                field="vendor_name", expected=v["name"], observed=invoice.vendor_name,
            ))
    if invoice.currency != v["currency"]:
        hits.append(PolicyHit(
            "POL-CURRENCY", Severity.ERROR,
            f"{v['name']} bills in {v['currency']}, invoice is in {invoice.currency}",
            field="currency", expected=v["currency"], observed=invoice.currency,
        ))
    if v["po_required"] and _stated(invoice.po_number) is None and invoice.total >= 0:
        hits.append(PolicyHit(
            "POL-PO-REQUIRED", Severity.WARNING,
            f"{v['name']} invoices must reference a purchase order",
            field="po_number",
        ))
    return tuple(hits)


def _tax_hits(invoice: Invoice, tax: PolicyChunk) -> tuple[PolicyHit, ...]:
    code, rate = tax.data["code"], tax.data["rate"]
    hits = []
    if invoice.tax_rate is not None and invoice.tax_rate != rate:
        hits.append(PolicyHit(
            "POL-TAX-RATE", Severity.ERROR,
            f"stated tax rate {invoice.tax_rate} differs from {code} rate {rate}",
            field="tax_rate", expected=str(rate), observed=str(invoice.tax_rate),
        ))
    expected = (invoice.subtotal * rate).quantize(CENT, ROUND_HALF_UP)
    gap = abs(invoice.tax_amount - expected)
    if gap:
        rounding = gap <= CENT
        hits.append(PolicyHit(
            "POL-TAX-ROUNDING" if rounding else "POL-TAX-AMOUNT",
            Severity.WARNING if rounding else Severity.ERROR,
            f"tax charged {invoice.tax_amount} but {code} at {rate} on "
            f"{invoice.subtotal} is {expected}",
            field="tax_amount", expected=str(expected), observed=str(invoice.tax_amount),
        ))
    return tuple(hits)


def record_policy(
    store: RunStore, run_id: str, result_or_hits: PolicyResult | Iterable[PolicyHit]
) -> RunRecord:
    """Append a ``PolicyResult`` (or bare hits) as the policy step of ``run_id``.

    Bare hits are wrapped in a ``PolicyResult`` with no retrieved chunks. A
    result whose ``run_id`` is ``None`` takes ``run_id``; one for a different
    run raises ``ValueError``. The run must exist and be awaiting the policy
    stage, otherwise ``RunStore`` raises (``RunNotFound`` / ``ValueError``).
    """
    if isinstance(result_or_hits, PolicyResult):
        result = result_or_hits
        if result.run_id is None:
            result = replace(result, run_id=run_id)
    else:
        result = PolicyResult(run_id, tuple(result_or_hits))
    if result.run_id != run_id:
        raise ValueError(f"policy result is for run {result.run_id!r}, not {run_id!r}")
    return store.append_step(run_id, Stage.POLICY, result)
