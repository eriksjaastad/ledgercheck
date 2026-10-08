"""Load the golden dataset in ``ledgercheck/fixtures/golden/``.

A golden case pairs one input invoice with what every pipeline stage should
produce for it, so the offline pipeline (and, later, a live extractor) can be
scored against known answers. Like the invoice fixtures in
``ledgercheck/fixtures/invoices/``, every invoice is invented and the files
ship as package data.

Case naming
-----------
The file is ``<case_id>.json``; ``case_id`` must equal the file stem and has
the form ``<primary_tag>__<variant>``: two lowercase ``snake_case`` parts
(letters and digits) joined by a double underscore
(``over_po__total_exceeds_book``, ``vendor_alias__dotted_name_known_limit``).
``<primary_tag>`` is the scenario the case exists for and must be one of its
``tags``; ``<variant>`` is a short free-form description of what this case
varies. The name says nothing about reuse: a case that reuses an invoice
fixture says so only in ``input.fixture`` (``rounding__per_line_print``
reuses ``rounding``, ``weird_formatting__german_decimal_comma_text`` reuses
``european_format_text``).

JSON keys (all required, no others allowed)
-------------------------------------------
``case_id``
    See naming above.
``description``
    One line: what makes this case tricky.
``tags``
    Non-empty list drawn from ``GOLDEN_TAGS`` (the fixture ``KNOWN_TAGS`` plus
    golden-only scenarios). Unlike fixture tags they are labels for coverage
    and filtering, not arithmetic promises.
``input``
    Exactly one of ``{"fixture": "<invoice fixture case id>"}`` (reuses that
    case's invoice and source text) or ``{"invoice": {...}}``
    (``Invoice.from_dict`` shape; a text source goes in a sibling
    ``<case_id>.txt``). Optionally ``"po_book"``:
    ``{"<PO number>": "<amount>"}`` handed to ``ApprovalAgent(po_amounts=...)``;
    without it the PO amount rule is off (``null`` is an error, not "no book").
``expected``
    ``extraction``: the complete invoice intake must return, in
    ``Invoice.from_dict`` shape: every required header field and every line
    item (``description``, ``quantity``, ``unit_price``, ``amount``, ``sku``).
    An optional field left out means ``None`` and is pinned as such. It is
    compared with ``==`` against the whole extracted ``Invoice``, so values
    compare as typed (``"1.50"`` equals ``"1.5"``). The input is the
    document, so this repeats the input invoice (the reused fixture's
    invoice for a ``fixture`` input); it is spelled out so the file states
    the answer an extractor reading the text source is scored against.
    ``policy``: ``{"retrieved": [chunk ids], "hits": [hit, ...]}``.
    ``approval``: ``{"outcome": "<Outcome value>", "hits": [hit, ...]}``.
    A hit is ``{"rule_id": ..., "severity": "<Severity value>", "field": ...,
    "expected": ..., "observed": ...}``, the last three a string or ``null``;
    lists are in the order the stages emit them. Approval hits are the
    approval rules' hits followed by the policy hits unchanged, so
    ``approval.hits`` must end with ``policy.hits``.

What is not pinned: hit ``message`` text and the decision's ``reasons``
(``"<rule_id>: <message>"`` lines), which are prose; the run id; and the
``ExtractionResult`` metadata (``extractor``, ``field_confidence``,
``warnings``), which describes the extractor rather than the document.
Everything else in the extraction, policy result (including ``reranker``,
``None`` since ``run_case`` never reranks) and decision (including
``invoice_number`` and ``decided_by``) is compared in full.

Validation goes through the domain models: the input and expected
extraction through ``Invoice``, hits through ``PolicyHit``, the policy block
through ``PolicyResult`` and the approval block through ``ApprovalDecision``
(so an ``approve`` with an ERROR hit is rejected). Anything malformed raises
``GoldenError``.

Running a case
--------------
``run_case(case)`` sends it through ``run_pipeline`` offline (fixture intake,
no reranking, the case's PO book). ``pinned(hits)`` blanks hit messages,
leaving what the expectations pin, and ``codes(hits)`` reduces hits to
``(rule_id, severity)`` pairs for quick checks.

Coverage matrix
---------------
The table lists how many cases carry each tag, expect each rule hit and
expect each outcome. Rule hits are read from ``expected.approval.hits``, which
include the policy hits, and are counted once per case. ``coverage(cases)``
derives the same counts. ``tests/test_golden.py`` fails when the table and the
cases disagree, when a row has no case, or when a rule id named in the agents'
docstrings is in neither the table nor ``UNREACHABLE_RULES``. So a new case
means updating the table::

    tag      blank_field                   5
    tag      clean                        10
    tag      credit_note                   8
    tag      currency_mismatch             4
    tag      duplicate_candidate           2
    tag      foreign_currency             11
    tag      missing_po                   13
    tag      multi_issue                   7
    tag      no_tax_rate                   3
    tag      over_po                       4
    tag      po_book                      12
    tag      rounding                      7
    tag      subtotal_mismatch             6
    tag      tax_mismatch                 11
    tag      tax_rate_mismatch             6
    tag      unchecked_gap                 4
    tag      unknown_po                    3
    tag      unknown_vendor                5
    tag      vendor_alias                  8
    tag      vendor_mismatch               3
    tag      weird_formatting              8
    rule     APR-PO-AMOUNT (error)         4
    rule     APR-PO-MISSING (info)         4
    rule     APR-PO-MISSING (warning)      9
    rule     APR-PO-UNKNOWN (warning)      3
    rule     APR-SUBTOTAL (error)          6
    rule     APR-TAX-AMOUNT (error)        6
    rule     APR-TAX-ROUNDING (warning)    6
    rule     POL-CURRENCY (error)          4
    rule     POL-PO-REQUIRED (warning)     9
    rule     POL-TAX-AMOUNT (error)       10
    rule     POL-TAX-RATE (error)          7
    rule     POL-TAX-ROUNDING (warning)    7
    rule     POL-VENDOR-NAME (error)       3
    rule     POL-VENDOR-NO-ID (info)       4
    rule     POL-VENDOR-UNKNOWN (error)    5
    outcome  approve                      31
    outcome  flag                         14
    outcome  needs_human                  29

No case can fire these with the shipped corpus (``UNREACHABLE_RULES``):
``POL-VENDOR-ALIAS``, because every corpus alias normalizes to its vendor's
record name. No rule produces ``reject``; that outcome is left to a person.
``unchecked_gap`` cases (line quantity x price, total vs subtotal + tax, dates,
duplicates) pin an approval that a new rule would have to change.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from ledgercheck.agents.approval import ApprovalAgent, PipelineResult, run_pipeline
from ledgercheck.agents.policy import PolicyAgent, PolicyResult
from ledgercheck.fixtures_loader import (
    FIXTURES_DIR,
    KNOWN_TAGS,
    FixtureCase,
    FixtureError,
    load_case,
)
from ledgercheck.models import (
    ApprovalDecision,
    Invoice,
    PolicyHit,
    Severity,
    _decimal,
)

GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "golden"

GOLDEN_TAGS = KNOWN_TAGS | frozenset(
    {
        "po_book",  # the case supplies a PO book, so the PO amount rule runs
        "over_po",  # total exceeds the PO's amount in the book
        "unknown_po",  # stated PO is not in the book
        "unknown_vendor",  # no vendor record for the stated id or name
        "vendor_mismatch",  # vendor id on file, but the name is not that vendor's
        "currency_mismatch",  # currency differs from the vendor's billing currency
        "tax_rate_mismatch",  # stated rate differs from the vendor's tax code
        "no_tax_rate",  # the document states no tax rate
        "blank_field",  # a field present but empty or whitespace only
        "multi_issue",  # several independent discrepancies at once
        "unchecked_gap",  # a real discrepancy no rule checks yet, so it approves
    }
)

# Rule ids no golden case can trigger with the shipped policy corpus, and why.
UNREACHABLE_RULES = MappingProxyType({
    "POL-VENDOR-ALIAS": "every corpus alias normalizes to its vendor's record name",
})

_CASE_ID = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*__[a-z0-9]+(?:_[a-z0-9]+)*")
_CASE_KEYS = {"case_id", "description", "tags", "input", "expected"}
_EXPECTED_KEYS = {"extraction", "policy", "approval"}
_HIT_KEYS = {"rule_id", "severity", "field", "expected", "observed"}


class GoldenError(FixtureError):
    """A golden case file does not match the documented layout."""


Code = tuple[str, Severity]


def codes(hits: Iterable[PolicyHit]) -> tuple[Code, ...]:
    """``(rule_id, severity)`` for each hit, in order."""
    return tuple((h.rule_id, h.severity) for h in hits)


def pinned(hits: Iterable[PolicyHit]) -> tuple[PolicyHit, ...]:
    """``hits`` with messages blanked: everything a golden hit pins."""
    return tuple(replace(h, message="") for h in hits)


def coverage(cases: Iterable[GoldenCase]) -> dict[str, Counter[str]]:
    """Case counts keyed ``"tag"``, ``"rule"`` and ``"outcome"`` (see the coverage matrix).

    A rule key is ``"<rule_id> (<severity>)"``, counted once per case that
    expects it among its approval hits.
    """
    cases = list(cases)
    return {
        "tag": Counter(t for c in cases for t in c.tags),
        "rule": Counter(f"{rule} ({sev.value})" for c in cases
                        for rule, sev in set(codes(c.expected_decision.hits))),
        "outcome": Counter(c.expected_decision.outcome.value for c in cases),
    }


@dataclass(frozen=True, slots=True)
class GoldenCase:
    case_id: str
    description: str
    tags: frozenset[str]
    fixture: FixtureCase  # the input; ``case_id`` is this golden case's id
    source_fixture: str | None  # the reused invoice fixture case id, if any
    po_book: Mapping[str, Decimal] | None
    expected_extraction: Invoice  # the complete invoice intake must return
    expected_policy: PolicyResult
    expected_decision: ApprovalDecision

    @property
    def path(self) -> Path:
        return self.fixture.path


def _require(cond: bool, where: str, msg: str) -> None:
    if not cond:
        raise GoldenError(f"{where}: {msg}")


def _keys(
    data: Any, required: set[str], where: str, optional: frozenset[str] | set[str] = frozenset()
) -> None:
    _require(isinstance(data, dict), where, f"expected an object, got {type(data).__name__}")
    _require(
        required <= set(data) <= required | optional,
        where,
        f"expected keys {sorted(required)} (optional {sorted(optional)}), got {sorted(data)}",
    )


def _hits(raw: Any, where: str) -> tuple[PolicyHit, ...]:
    _require(isinstance(raw, list), where, "hits must be a list")
    hits = []
    for i, item in enumerate(raw):
        _keys(item, _HIT_KEYS, f"{where}[{i}]")
        _require(isinstance(item["rule_id"], str) and item["rule_id"].strip(), f"{where}[{i}]",
                 "rule_id must be a non-blank string")
        for name in ("field", "expected", "observed"):
            _require(item[name] is None or isinstance(item[name], str), f"{where}[{i}]",
                     f"{name} must be a string or null")
        hits.append(PolicyHit(item["rule_id"], item["severity"], message="", field=item["field"],
                              expected=item["expected"], observed=item["observed"]))
    return tuple(hits)


def _line_items(raw: Mapping[str, Any], where: str) -> None:
    # Invoice.from_dict iterates line_items, so "" or {} would become no items.
    if "line_items" in raw:
        _require(isinstance(raw["line_items"], list), where, "line_items must be a list")


def _input(data: Any, path: Path, fixtures_root: Path) -> tuple[Invoice, str | None, str | None]:
    """The input invoice, source text and reused fixture id."""
    where = f"{path.name}: input"
    _keys(data, set(), where, {"fixture", "invoice", "po_book"})
    _require(("fixture" in data) != ("invoice" in data), where,
             "needs exactly one of 'fixture' or 'invoice'")
    text_path = path.with_suffix(".txt")
    if "fixture" in data:
        _require(not text_path.exists(), where, f"{text_path.name} with a fixture reference")
        ref = data["fixture"]
        _require(isinstance(ref, str) and re.fullmatch(r"[a-z0-9_]+", ref) is not None
                 and (fixtures_root / f"{ref}.json").is_file(), where,
                 f"no fixture case {ref!r}")
        source = load_case(fixtures_root / f"{ref}.json")
        return source.invoice, source.raw_text, ref
    if isinstance(data["invoice"], dict):
        _line_items(data["invoice"], f"{where}.invoice")
    invoice = Invoice.from_dict(data["invoice"])
    raw_text = text_path.read_text(encoding="utf-8") if text_path.exists() else None
    return invoice, raw_text, None


def _po_book(data: Mapping[str, Any], where: str) -> Mapping[str, Decimal] | None:
    if "po_book" not in data:
        return None
    raw = data["po_book"]
    _require(isinstance(raw, dict) and all(isinstance(k, str) for k in raw), where,
             "po_book must be an object of PO number -> amount")
    return MappingProxyType({po: _decimal(amount, f"po_book[{po!r}]") for po, amount in raw.items()})


def _extraction(raw: Any, where: str) -> Invoice:
    where = f"{where}.extraction"
    _require(isinstance(raw, dict), where, "extraction must be an object")
    _line_items(raw, where)
    try:
        return Invoice.from_dict(raw)
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        raise GoldenError(f"{where}: {exc}") from exc


def load_golden_case(path: Path, *, fixtures_root: Path = FIXTURES_DIR) -> GoldenCase:
    """Load and validate one ``<case_id>.json`` golden case."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:  # JSONDecodeError or UnicodeDecodeError
        raise GoldenError(f"{path.name}: invalid JSON: {exc}") from exc
    _keys(data, _CASE_KEYS, path.name)
    case_id = data["case_id"]
    _require(case_id == path.stem, path.name, f"case_id {case_id!r} != file stem")
    _require(bool(_CASE_ID.fullmatch(case_id)), path.name,
             f"case_id {case_id!r} is not <primary_tag>__<variant> snake_case")
    _require(isinstance(data["description"], str) and data["description"].strip(),
             path.name, "description must be a non-blank string")
    raw_tags = data["tags"]
    _require(isinstance(raw_tags, list) and all(isinstance(t, str) for t in raw_tags),
             path.name, f"tags must be a list of strings, got {raw_tags!r}")
    tags = frozenset(raw_tags)
    _require(bool(tags) and tags <= GOLDEN_TAGS, path.name,
             f"unknown or empty tags {sorted(tags - GOLDEN_TAGS)}")
    _require(case_id.split("__")[0] in tags, path.name,
             f"primary tag {case_id.split('__')[0]!r} is not in tags")
    expected = data["expected"]
    _keys(expected, _EXPECTED_KEYS, f"{path.name}: expected")
    policy, approval = expected["policy"], expected["approval"]
    _keys(policy, {"retrieved", "hits"}, f"{path.name}: expected.policy")
    _keys(approval, {"outcome", "hits"}, f"{path.name}: expected.approval")
    try:
        invoice, raw_text, ref = _input(data["input"], path, fixtures_root)
        po_book = _po_book(data["input"], f"{path.name}: input")
        extraction = _extraction(expected["extraction"], f"{path.name}: expected")
        _require(isinstance(policy["retrieved"], list), f"{path.name}: expected.policy.retrieved",
                 "retrieved must be a list")
        expected_policy = PolicyResult(
            None, _hits(policy["hits"], f"{path.name}: expected.policy.hits"), policy["retrieved"]
        )
        decision = ApprovalDecision(
            run_id=case_id,
            invoice_number=invoice.invoice_number,
            outcome=approval["outcome"],
            hits=_hits(approval["hits"], f"{path.name}: expected.approval.hits"),
        )
    except GoldenError:
        raise
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        raise GoldenError(f"{path.name}: {exc}") from exc
    n = len(expected_policy.hits)
    _require(
        decision.hits[len(decision.hits) - n:] == expected_policy.hits,
        path.name, "approval.hits must end with policy.hits",
    )
    return GoldenCase(
        case_id=case_id,
        description=data["description"],
        tags=tags,
        fixture=FixtureCase(case_id, data["description"], tags, invoice, path, raw_text),
        source_fixture=ref,
        po_book=po_book,
        expected_extraction=extraction,
        expected_policy=expected_policy,
        expected_decision=decision,
    )


def load_golden_cases(
    root: Path = GOLDEN_DIR, *, fixtures_root: Path = FIXTURES_DIR
) -> list[GoldenCase]:
    """Load every golden case under ``root``, sorted by case id.

    Raises ``GoldenError`` for a missing or empty ``root``, a malformed case,
    or a ``.txt`` with no matching ``.json``; never returns ``[]``.
    """
    if not root.is_dir():
        raise GoldenError(f"golden directory not found: {root}")
    if orphans := sorted(
        p.name for p in root.glob("*.txt") if not p.with_suffix(".json").exists()
    ):
        raise GoldenError(f"text sources without a .json case: {orphans}")
    cases = [load_golden_case(p, fixtures_root=fixtures_root) for p in sorted(root.glob("*.json"))]
    if not cases:
        raise GoldenError(f"no golden cases (*.json) in {root}")
    return cases


def run_case(case: GoldenCase, **kwargs: Any) -> PipelineResult:
    """Run ``case`` through ``run_pipeline`` offline with its PO book.

    Extra keyword arguments (``store``, ``run_id``, ...) pass through. A
    ``policy`` or ``approval`` that is absent or ``None`` defaults to
    ``PolicyAgent(rerank=False)`` or ``ApprovalAgent(po_amounts=case.po_book)``;
    the defaults are only built when needed, so a supplied agent never loads
    the default policy corpus.
    """
    if kwargs.get("policy") is None:
        kwargs["policy"] = PolicyAgent(rerank=False)
    if kwargs.get("approval") is None:
        kwargs["approval"] = ApprovalAgent(po_amounts=case.po_book)
    return run_pipeline(case.fixture, **kwargs)
