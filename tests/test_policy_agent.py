"""PolicyAgent: local corpus retrieval, policy rules, rerank gate, RunStore recording."""

import dataclasses
import json
import socket
from decimal import Decimal

import pytest

from ledgercheck.agents import IntakeAgent, PolicyAgent, PolicyResult, record_intake, record_policy
from ledgercheck.agents.llm_client import API_KEY_ENV, ENV_FLAG, LiveLLMDisabled
from ledgercheck.agents.policy import (
    POLICY_DIR,
    RERANK_FLAG,
    PolicyChunk,
    PolicyCorpus,
    PolicyCorpusError,
    normalize_name,
)
from ledgercheck.fixtures_loader import FIXTURES_DIR, load_case, load_cases
from ledgercheck.models import PolicyHit, Severity, to_jsonable
from ledgercheck.run_store import RunStatus, RunStore, Stage

# Every invoice fixture, with the rule ids the policy stage should raise for it.
EXPECTED_RULES = {
    "clean_baseline": [],
    "credit_note": [],
    "duplicate_resubmission": [],
    "european_format_text": [],
    "gbp_cloud_services": [],
    "missing_po": ["POL-PO-REQUIRED"],
    "ocr_noise_text": [],
    "rounding": ["POL-TAX-ROUNDING"],
    "subtotal_mismatch": [],
    "tax_mismatch": ["POL-TAX-AMOUNT"],
    # "Harbor and Pine Consulting LLC" normalizes to the record name, so no alias hit.
    "vendor_alias": ["POL-VENDOR-NO-ID"],
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.delenv(RERANK_FLAG, raising=False)
    return PolicyAgent()


def invoice(case_id, **changes):
    inv = load_case(FIXTURES_DIR / f"{case_id}.json").invoice
    return dataclasses.replace(inv, **changes)


def rules(result):
    return [h.rule_id for h in result.hits]


def test_expected_rules_cover_every_fixture():
    assert set(EXPECTED_RULES) == {c.case_id for c in load_cases()}


@pytest.mark.parametrize("case_id", sorted(EXPECTED_RULES))
def test_fixture_cases_raise_expected_rules(agent, case_id):
    result = agent.check(invoice(case_id))
    assert rules(result) == EXPECTED_RULES[case_id]
    assert all(isinstance(h, PolicyHit) for h in result.hits)
    assert len(result.retrieved) == 2


def test_clean_known_vendor_passes(agent):
    result = agent.check(invoice("clean_baseline"))
    assert result.hits == ()
    assert result.errors == ()
    assert result.retrieved == ("vendor:V-1001", "tax:US-TX-STD")
    assert result.reranker is None


def test_vendor_alias_matches_by_name_with_info_only(agent):
    result = agent.check(invoice("vendor_alias"))
    assert {h.severity for h in result.hits} == {Severity.INFO}
    assert result.retrieved[0] == "vendor:V-1003"
    assert result.hits[0].expected == "V-1003"


@pytest.mark.parametrize(
    "case_id, name",
    [
        ("rounding", "Tallow Print Co"),
        ("rounding", "tallow print co"),
        ("clean_baseline", "NORTHWIND OFFICE SUPPLY"),
        ("clean_baseline", "Northwind Office Supply, Inc."),
    ],
)
def test_normalized_spelling_of_record_name_is_not_an_alias(agent, case_id, name):
    result = agent.check(invoice(case_id, vendor_name=name))
    assert "POL-VENDOR-ALIAS" not in rules(result)
    assert "POL-VENDOR-NAME" not in rules(result)


def test_genuine_alias_is_flagged():
    chunks = [
        dataclasses.replace(c, data={**c.data, "aliases": ["NW Supply"]})
        if c.chunk_id == "vendor:V-1001" else c
        for c in PolicyCorpus.load().chunks
    ]
    agent = PolicyAgent(PolicyCorpus(chunks), rerank=False)
    for name in ("NW Supply", "nw supply inc."):
        [hit] = agent.check(invoice("clean_baseline", vendor_name=name)).hits
        assert (hit.rule_id, hit.severity) == ("POL-VENDOR-ALIAS", Severity.INFO)
        assert (hit.expected, hit.observed) == ("Northwind Office Supply", name)


def test_known_vendor_id_with_wrong_name_is_flagged(agent):
    result = agent.check(invoice("clean_baseline", vendor_name="Harbor & Pine Consulting"))
    [hit] = result.errors
    assert hit.rule_id == "POL-VENDOR-NAME"
    assert hit.expected == "Northwind Office Supply"
    assert hit.observed == "Harbor & Pine Consulting"


def test_currency_mismatch_is_flagged(agent):
    result = agent.check(invoice("clean_baseline", currency="EUR"))
    assert rules(result) == ["POL-CURRENCY"]
    assert result.hits[0].severity is Severity.ERROR


def test_tax_mismatch_fixture_flags_amount(agent):
    [hit] = agent.check(invoice("tax_mismatch")).hits
    assert (hit.rule_id, hit.severity) == ("POL-TAX-AMOUNT", Severity.ERROR)
    assert (hit.expected, hit.observed) == ("435.90", "465.96")


def test_rounding_uses_half_up_on_code_rate(agent):
    # 80.75 * 0.06 = 4.845: half-up gives 4.85, a one-cent gap from 4.86.
    [hit] = agent.check(invoice("rounding")).hits
    assert (hit.severity, hit.expected, hit.observed) == (Severity.WARNING, "4.85", "4.86")


@pytest.mark.parametrize(
    "tax_amount, rule, severity",
    [
        ("4.855", "POL-TAX-ROUNDING", Severity.WARNING),  # sub-cent gap
        ("4.84", "POL-TAX-ROUNDING", Severity.WARNING),  # exactly one cent under
        ("4.861", "POL-TAX-AMOUNT", Severity.ERROR),  # just over one cent
        ("4.87", "POL-TAX-AMOUNT", Severity.ERROR),
    ],
)
def test_tax_gap_of_at_most_one_cent_is_rounding(agent, tax_amount, rule, severity):
    [hit] = agent.check(invoice("rounding", tax_amount=Decimal(tax_amount))).hits
    assert (hit.rule_id, hit.severity, hit.expected) == (rule, severity, "4.85")


def test_stated_rate_differs_from_tax_code(agent):
    result = agent.check(invoice("gbp_cloud_services", tax_rate=Decimal("0.175")))
    assert rules(result) == ["POL-TAX-RATE"]
    assert result.hits[0].expected == "0.20"


def test_credit_note_is_exempt_from_po_rule(agent):
    assert agent.check(invoice("credit_note")).hits == ()
    flipped = invoice("credit_note", total=Decimal("95.00"), subtotal=Decimal("95.00"))
    assert "POL-PO-REQUIRED" in rules(agent.check(flipped))


def test_unknown_vendor_is_an_error_with_closest_hint(agent):
    # The name is on file, but the unknown id is what the message must name.
    result = agent.check(invoice("clean_baseline", vendor_id="V-9999"))
    [hit] = result.hits
    assert (hit.rule_id, hit.severity) == ("POL-VENDOR-UNKNOWN", Severity.ERROR)
    assert (hit.field, hit.observed) == ("vendor_id", "V-9999")
    assert result.retrieved[0] == "vendor:V-1001"
    assert hit.message == (
        "vendor id 'V-9999' is not on file; closest on file: Northwind Office Supply"
    )


def test_unknown_vendor_name_without_id_names_the_vendor(agent):
    name = "Northwind Paper Traders"
    result = agent.check(invoice("clean_baseline", vendor_id=None, vendor_name=name))
    [hit] = result.hits
    assert (hit.rule_id, hit.severity) == ("POL-VENDOR-UNKNOWN", Severity.ERROR)
    assert (hit.field, hit.observed) == ("vendor_name", name)
    assert result.retrieved[0] == "vendor:V-1001"
    assert hit.message == (
        "vendor 'Northwind Paper Traders' is not on file; "
        "closest on file: Northwind Office Supply"
    )


def test_wrong_vendor_id_is_not_rescued_by_name(agent):
    # Checked against V-1003's record, so its tax code disagrees too.
    result = agent.check(invoice("clean_baseline", vendor_id="V-1003"))
    assert rules(result)[0] == "POL-VENDOR-NAME"
    assert result.retrieved == ("vendor:V-1003", "tax:US-WA-SVC")


def test_unknown_vendor_with_no_keyword_overlap_has_no_hint(agent):
    result = agent.check(invoice("clean_baseline", vendor_id=None, vendor_name="Zzyzx"))
    assert result.retrieved == ()
    assert "closest" not in result.hits[0].message


def test_normalize_name_ignores_suffixes_and_ampersand():
    assert normalize_name("Harbor & Pine Consulting") == normalize_name(
        "HARBOR AND PINE CONSULTING, LLC"
    )


def test_search_ranks_by_keyword_overlap():
    corpus = PolicyCorpus.load()
    assert [c.chunk_id for c in corpus.search("hydraulic fittings", k=1)] == ["vendor:V-2001"]
    assert [c.chunk_id for c in corpus.search("UK VAT", kind="tax", k=1)] == ["tax:UK-VAT-STD"]
    assert corpus.search("and the of") == []


def test_check_takes_run_id_from_extraction(agent):
    extraction = IntakeAgent().extract("tax_mismatch", run_id="run-p")
    assert agent.check(extraction).run_id == "run-p"
    assert agent.check(invoice("tax_mismatch")).run_id is None
    with pytest.raises(ValueError, match="does not match"):
        agent.check(extraction, run_id="run-other")


def tax_chunk(chunk_id="tax:X", **changes):
    return {"chunk_id": chunk_id, "kind": "tax", "code": "X", "rate": "0.05", "text": "t",
            **changes}


def vendor_chunk(chunk_id="vendor:V", **changes):
    return {"chunk_id": chunk_id, "kind": "vendor", "vendor_id": "V", "name": "Acme",
            "aliases": [], "currency": "USD", "tax_code": "X", "po_required": False,
            "text": "t", **changes}


@pytest.mark.parametrize(
    "files, match",
    [
        ({}, "no policy corpus files"),
        ({"a.json": [{"chunk_id": "x", "kind": "memo", "text": ""}]}, "unknown chunk kind"),
        ({"a.json": [{"chunk_id": "tax:X", "kind": "tax", "text": ""}]}, "missing"),
        ({"a.json": [tax_chunk(rate="abc")]}, "bad rate"),
        ({"a.json": [tax_chunk(rate=0.06)]}, "bad rate"),  # JSON float
        ({"a.json": [tax_chunk(rate="NaN")]}, "bad rate"),
        ({"a.json": [tax_chunk(rate="Infinity")]}, "bad rate"),
        ({"a.json": [tax_chunk(rate=True)]}, "bad rate"),
        ({"a.json": [tax_chunk()] * 2}, "duplicate chunk_id"),
        ({"a.json": [tax_chunk(), tax_chunk("tax:Y")]}, "duplicate tax code"),
        ({"a.json": [tax_chunk(), vendor_chunk(), vendor_chunk("vendor:W")]},
         "duplicate vendor_id"),
        ({"a.json": [tax_chunk(), vendor_chunk(po_required="yes")]}, "po_required"),
        ({"a.json": [tax_chunk(), vendor_chunk(po_required=1)]}, "po_required"),
        ({"a.json": [tax_chunk(), vendor_chunk(aliases="Acme Co")]}, "aliases"),
        ({"a.json": [tax_chunk(), vendor_chunk(aliases=["Acme Co", 7])]}, "aliases"),
        ({"a.json": [tax_chunk(), vendor_chunk(aliases=[""])]}, "aliases"),
        ({"a.json": [tax_chunk(), vendor_chunk(),
                     vendor_chunk("vendor:W", vendor_id="W", name="Acme, Inc.")]},
         "name or alias 'acme' also matches vendor:V"),
        ({"a.json": [tax_chunk(), vendor_chunk(),
                     vendor_chunk("vendor:W", vendor_id="W", name="Bolt", aliases=["ACME LLC"])]},
         "name or alias 'acme' also matches vendor:V"),
        ({"a.json": [tax_chunk(), vendor_chunk(name="")]}, "name"),
        ({"a.json": [tax_chunk(), vendor_chunk(currency=None)]}, "currency"),
        ({"a.json": [tax_chunk(), vendor_chunk(vendor_id=1001)]}, "vendor_id"),
        ({"a.json": [tax_chunk(code="  ")]}, "code"),
        ({"a.json": [tax_chunk(chunk_id=3)]}, "chunk_id"),
        ({"a.json": [tax_chunk(text=None)]}, "text"),
        ({"a.json": {"chunk_id": "tax:X"}}, "list of chunks"),
    ],
)
def test_malformed_corpus_raises(tmp_path, files, match):
    for name, chunks in files.items():
        (tmp_path / name).write_text(json.dumps(chunks), encoding="utf-8")
    with pytest.raises(PolicyCorpusError, match=match):
        PolicyCorpus.load(tmp_path)


def test_own_alias_matching_own_name_is_not_a_collision(tmp_path):
    chunks = [tax_chunk(), vendor_chunk(aliases=["ACME LLC", "Acme Labs"]),
              vendor_chunk("vendor:W", vendor_id="W", name="Bolt")]
    (tmp_path / "a.json").write_text(json.dumps(chunks), encoding="utf-8")
    corpus = PolicyCorpus.load(tmp_path)
    assert corpus.find_vendor(None, "acme labs")[0].chunk_id == "vendor:V"
    assert corpus.find_vendor(None, "Bolt Inc")[0].chunk_id == "vendor:W"


def test_chunk_data_is_deeply_frozen():
    aliases = ["NW Supply"]
    chunk = PolicyChunk("vendor:X", "vendor", "t", {"aliases": aliases, "extra": {"tags": ["a"]}})
    aliases.append("Other")  # the caller's list is copied, not shared
    assert chunk.data["aliases"] == ("NW Supply",)
    assert chunk.data["extra"]["tags"] == ("a",)
    with pytest.raises(TypeError):
        chunk.data["aliases"] = ("Other",)  # type: ignore[index]
    with pytest.raises(AttributeError):
        chunk.data["aliases"].append("Other")
    with pytest.raises(TypeError):
        chunk.data["extra"]["tags"] = ()  # type: ignore[index]
    with pytest.raises(AttributeError):
        chunk.data["extra"]["tags"].append("b")


def test_loaded_aliases_cannot_be_mutated_to_change_lookup():
    corpus = PolicyCorpus.load()
    vendor = corpus.get("vendor:V-1001")
    assert isinstance(vendor.data["aliases"], tuple)
    with pytest.raises(AttributeError):
        vendor.data["aliases"].append("Zzyzx")
    assert corpus.find_vendor(None, "Zzyzx") == (None, "")


def test_search_matches_nested_structured_values():
    base = PolicyCorpus.load()
    tax = base.get("tax:US-TX-STD")
    vendor = base.get("vendor:V-1001")
    chunk = dataclasses.replace(vendor, data={**vendor.data, "extra": {"tags": ["hydroponics"]}})
    corpus = PolicyCorpus([tax, chunk])
    assert corpus.search("hydroponics") == [chunk]
    assert corpus.search("mappingproxy") == []


def test_search_ignores_boolean_fields(agent):
    # po_required is True on every vendor; its str() form must not be a keyword.
    assert agent.corpus.search("true") == []
    assert agent.corpus.search("false") == []
    name = "True Value Hardware"
    result = agent.check(invoice("clean_baseline", vendor_id=None, vendor_name=name))
    [hit] = result.hits
    assert hit.rule_id == "POL-VENDOR-UNKNOWN"
    assert "closest on file" not in hit.message
    assert result.retrieved == ()


def test_search_matches_decimal_rate_and_codes():
    corpus = PolicyCorpus.load()
    assert [c.chunk_id for c in corpus.search("0825", kind="tax")] == ["tax:US-TX-STD"]
    assert corpus.search("US-TX-STD", kind="vendor")[0].data["tax_code"] == "US-TX-STD"


def test_search_rejects_negative_k():
    corpus = PolicyCorpus.load()
    assert corpus.search("hydraulic fittings", k=0) == []
    with pytest.raises(ValueError, match="k must not be negative"):
        corpus.search("hydraulic fittings", k=-1)


@pytest.mark.parametrize(
    "chunks, match",
    [
        ([tax_chunk(), vendor_chunk(currency="usd")], r"vendor:V: currency .* got 'usd'"),
        ([tax_chunk(), vendor_chunk(currency=" USD")], r"vendor:V: currency .* got ' USD'"),
        ([tax_chunk(), vendor_chunk(currency="US")], r"vendor:V: currency"),
        ([tax_chunk(rate="8.25")], r"tax:X: rate must be a fraction in \[0, 1\), got 8.25"),
        ([tax_chunk(rate="-0.0725")], r"tax:X: rate must be a fraction .* got -0.0725"),
        ([tax_chunk(rate=1)], r"tax:X: rate must be a fraction"),
    ],
)
def test_corpus_rejects_values_no_invoice_could_match(tmp_path, chunks, match):
    (tmp_path / "a.json").write_text(json.dumps(chunks), encoding="utf-8")
    with pytest.raises(PolicyCorpusError, match=match):
        PolicyCorpus.load(tmp_path)


def test_shipped_corpus_passes_value_validation():
    corpus = PolicyCorpus.load()
    vendors = [c for c in corpus.chunks if c.kind == "vendor"]
    taxes = [c for c in corpus.chunks if c.kind == "tax"]
    assert vendors and taxes
    assert all(len(v.data["currency"]) == 3 for v in vendors)
    assert all(0 <= t.data["rate"] < 1 for t in taxes)


def test_corpus_accepts_integer_rate(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([tax_chunk(rate=0)]), encoding="utf-8")
    assert PolicyCorpus.load(tmp_path).tax_code("X").data["rate"] == Decimal("0")


def test_invalid_json_raises_corpus_error(tmp_path):
    (tmp_path / "a.json").write_text('[{"chunk_id": "tax:X",', encoding="utf-8")
    with pytest.raises(PolicyCorpusError, match="a.json: invalid JSON") as exc:
        PolicyCorpus.load(tmp_path)
    assert isinstance(exc.value.__cause__, json.JSONDecodeError)


def test_duplicate_vendor_id_rejected_in_constructor():
    corpus = PolicyCorpus.load()
    v = corpus.get("vendor:V-1001")
    with pytest.raises(PolicyCorpusError, match="duplicate vendor_id 'V-1001'"):
        PolicyCorpus([*corpus.chunks, dataclasses.replace(v, chunk_id="vendor:copy")])


def test_vendor_with_unknown_tax_code_raises(tmp_path):
    vendors = json.loads((POLICY_DIR / "vendors.json").read_text(encoding="utf-8"))
    (tmp_path / "vendors.json").write_text(json.dumps(vendors), encoding="utf-8")
    with pytest.raises(PolicyCorpusError, match="unknown tax_code"):
        PolicyCorpus.load(tmp_path)


# --- blank fields, empty names, trailing suffixes --------------------------


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_vendor_id_counts_as_absent_for_unknown_vendor(agent, blank):
    name = "Northwind Paper Traders"
    result = agent.check(invoice("clean_baseline", vendor_id=blank, vendor_name=name))
    [hit] = result.hits
    assert hit.rule_id == "POL-VENDOR-UNKNOWN"
    assert (hit.field, hit.observed) == ("vendor_name", name)
    assert hit.message.startswith(f"vendor {name!r} is not on file")


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_vendor_id_falls_back_to_name_lookup(agent, blank):
    result = agent.check(invoice("clean_baseline", vendor_id=blank))
    assert rules(result) == ["POL-VENDOR-NO-ID"]
    assert result.retrieved == ("vendor:V-1001", "tax:US-TX-STD")


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_blank_po_number_counts_as_not_stated(agent, blank):
    [hit] = agent.check(invoice("clean_baseline", po_number=blank)).hits
    assert (hit.rule_id, hit.field) == ("POL-PO-REQUIRED", "po_number")


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Co-op Foods", "co op foods"),
        ("Acme Co. Ltd", "acme"),
        ("Acme & Co., Inc.", "acme"),
        ("Inc Holdings", "inc holdings"),
        ("Tallow Print Co.", "tallow print"),
        ("Corp Co LLC", ""),
        ("&", ""),
        ("Müller GmbH", "müller"),
    ],
)
def test_normalize_name_strips_only_trailing_suffixes(name, expected):
    assert normalize_name(name) == expected


def test_co_op_does_not_collide_with_op(tmp_path):
    chunks = [tax_chunk(), vendor_chunk(name="Co-op Foods"),
              vendor_chunk("vendor:W", vendor_id="W", name="Op Foods")]
    (tmp_path / "a.json").write_text(json.dumps(chunks), encoding="utf-8")
    corpus = PolicyCorpus.load(tmp_path)
    assert corpus.find_vendor(None, "CO-OP FOODS LLC")[0].chunk_id == "vendor:V"
    assert corpus.find_vendor(None, "Op Foods")[0].chunk_id == "vendor:W"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Harbor & Pine Consulting L.L.C.", "harbor pine consulting"),
        ("Kestrel Industrial G.m.b.H.", "kestrel industrial"),
        ("Tallow Print Co.", "tallow print"),
        ("Acme L.L.C. Ltd.", "acme"),
        ("O'Neil Supply", "oneil supply"),
        ("O’Neil Supply", "oneil supply"),
        ("Co-op Foods L.L.C.", "co op foods"),
        ("Acme, Inc.", "acme"),
        ("L.L.C.", ""),
        ("Müller GmbH", "müller"),
    ],
)
def test_normalize_name_joins_in_word_marks_and_applies_nfkc(name, expected):
    assert normalize_name(name) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("J. P. Morgan", "j p morgan"),
        ("J.P. Morgan", "jp morgan"),
        ("Tallow.Print", "tallowprint"),
        ("Tallow-Print", "tallow print"),
        ("Acme L. L. C.", "acme l l c"),
        ("Acme Co.Ltd", "acme coltd"),
    ],
)
def test_normalize_name_known_limits_match_docstring(name, expected):
    # Pins the "Known limits" in normalize_name's docstring; if one of these
    # changes, update the docstring with it.
    assert normalize_name(name) == expected


@pytest.mark.parametrize(
    "case_id, name",
    [
        ("tax_mismatch", "Harbor & Pine Consulting L.L.C."),
        ("european_format_text", "Kestrel Industrial G.m.b.H."),
        ("rounding", "Tallow Print Co."),
    ],
)
def test_dotted_legal_suffix_is_not_a_name_error(agent, case_id, name):
    by_id = agent.check(invoice(case_id, vendor_name=name))
    assert rules(by_id) == EXPECTED_RULES[case_id]
    by_name = agent.check(invoice(case_id, vendor_id=None, vendor_name=name))
    assert rules(by_name) == ["POL-VENDOR-NO-ID", *EXPECTED_RULES[case_id]]


@pytest.mark.parametrize("case", load_cases(), ids=lambda c: c.case_id)
def test_every_fixture_vendor_name_finds_its_record(case):
    chunk, how = PolicyCorpus.load().find_vendor(None, case.invoice.vendor_name)
    assert how == "name"
    if case.invoice.vendor_id is not None:
        assert chunk.data["vendor_id"] == case.invoice.vendor_id


def test_decomposed_name_matches_composed_record(tmp_path):
    chunks = [tax_chunk(), vendor_chunk(name="Müller GmbH")]
    (tmp_path / "a.json").write_text(json.dumps(chunks), encoding="utf-8")
    corpus = PolicyCorpus.load(tmp_path)
    assert corpus.find_vendor(None, "Müller GmbH")[0].chunk_id == "vendor:V"


@pytest.mark.parametrize("empty", ["Inc.", "LLC", "&", "Co. Ltd", " - "])
@pytest.mark.parametrize("key", ["name", "aliases"])
def test_corpus_rejects_name_or_alias_that_normalizes_to_empty(tmp_path, key, empty):
    value = [empty] if key == "aliases" else empty
    (tmp_path / "a.json").write_text(
        json.dumps([tax_chunk(), vendor_chunk(**{key: value})]), encoding="utf-8"
    )
    with pytest.raises(PolicyCorpusError, match="normalizes to an empty name"):
        PolicyCorpus.load(tmp_path)


def test_constructor_rejects_empty_normalized_name():
    corpus = PolicyCorpus.load()
    v = corpus.get("vendor:V-1001")
    bad = dataclasses.replace(v, data={**v.data, "aliases": ["LLC"]})
    others = [c for c in corpus.chunks if c is not v]
    with pytest.raises(PolicyCorpusError, match="'LLC' normalizes to an empty name"):
        PolicyCorpus([*others, bad])


@pytest.mark.parametrize("name", ["Inc.", "LLC", "&", "--"])
def test_document_name_normalizing_to_empty_never_matches(agent, name):
    assert agent.corpus.find_vendor(None, name) == (None, "")
    [hit] = agent.check(invoice("clean_baseline", vendor_id=None, vendor_name=name)).hits
    assert (hit.rule_id, hit.field, hit.observed) == ("POL-VENDOR-UNKNOWN", "vendor_name", name)
    # With a known id, an empty-normalizing name is a name mismatch, not a match.
    assert rules(agent.check(invoice("clean_baseline", vendor_name=name))) == ["POL-VENDOR-NAME"]


def test_retrieved_holds_search_candidates_for_unknown_vendor(agent):
    name = "Northwind Harbor Supply"
    result = agent.check(invoice("clean_baseline", vendor_id=None, vendor_name=name))
    hint = [c.chunk_id for c in agent.corpus.search(name, kind="vendor")]
    assert len(hint) >= 2
    assert result.retrieved == tuple(hint)


# --- rerank gate -----------------------------------------------------------

UNKNOWN = {"vendor_id": None, "vendor_name": "Northwind Harbor Supply"}


class FakeReranker:
    name = "fake-rerank"

    def __init__(self, reorder=lambda ids: list(reversed(ids))):
        self.reorder = reorder
        self.calls = []

    def rerank(self, query, candidates):
        self.calls.append((query, list(candidates)))
        return list(self.reorder(candidates))


def test_rerank_off_by_default(monkeypatch):
    monkeypatch.delenv(RERANK_FLAG, raising=False)
    fake = FakeReranker()
    result = PolicyAgent(reranker=fake).check(invoice("clean_baseline", **UNKNOWN))
    assert fake.calls == []
    assert result.reranker is None


@pytest.mark.parametrize("value", ["", "0", "true", "yes"])
def test_rerank_flag_must_be_exactly_one(monkeypatch, value):
    monkeypatch.setenv(RERANK_FLAG, value)
    assert PolicyAgent().rerank is False


def test_rerank_flag_without_llm_gate_refuses(monkeypatch):
    monkeypatch.setenv(RERANK_FLAG, "1")
    monkeypatch.delenv(ENV_FLAG, raising=False)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    agent = PolicyAgent()
    assert agent.check(invoice("clean_baseline")).hits == ()  # known vendor: no rerank needed
    with pytest.raises(LiveLLMDisabled, match="live LLM calls are off") as exc:
        agent.check(invoice("clean_baseline", **UNKNOWN))
    assert "extraction" not in str(exc.value)


def test_rerank_with_full_gate_still_makes_no_request(monkeypatch):
    monkeypatch.setenv(RERANK_FLAG, "1")
    monkeypatch.setenv(ENV_FLAG, "1")
    monkeypatch.setenv(API_KEY_ENV, "sk-test-not-real")
    with pytest.raises(NotImplementedError):
        PolicyAgent().check(invoice("clean_baseline", **UNKNOWN))


@pytest.mark.parametrize("value", ["0", "1", 1, 0])
def test_explicit_rerank_must_be_bool(value):
    with pytest.raises(TypeError, match="rerank"):
        PolicyAgent(rerank=value)  # type: ignore[arg-type]


def test_reranker_is_none_when_rerank_on_but_unused():
    fake = FakeReranker()
    agent = PolicyAgent(rerank=True, reranker=fake)
    known = agent.check(invoice("clean_baseline"))
    no_candidates = agent.check(invoice("clean_baseline", vendor_id=None, vendor_name="Zzyzx"))
    assert known.reranker is None and no_candidates.reranker is None
    assert no_candidates.retrieved == ()
    assert fake.calls == []


def test_injected_reranker_reorders_hint_but_never_matches():
    fake = FakeReranker()
    plain = PolicyAgent(rerank=False).check(invoice("clean_baseline", **UNKNOWN))
    result = PolicyAgent(rerank=True, reranker=fake).check(invoice("clean_baseline", **UNKNOWN))
    assert len(plain.retrieved) >= 2
    assert result.retrieved == tuple(reversed(plain.retrieved))
    assert result.reranker == "fake-rerank"
    assert rules(result) == ["POL-VENDOR-UNKNOWN"]


def test_reranker_must_return_a_permutation():
    fake = FakeReranker(reorder=lambda _ids: ["vendor:V-9999"])
    with pytest.raises(ValueError, match="permutation"):
        PolicyAgent(rerank=True, reranker=fake).check(invoice("clean_baseline", **UNKNOWN))


# --- RunStore integration --------------------------------------------------


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "runs")


def test_record_policy_appends_step(agent, store):
    extraction = IntakeAgent().extract("tax_mismatch", run_id="run-tax")
    record_intake(store, extraction)
    result = agent.check(extraction)
    record = record_policy(store, "run-tax", result)
    assert record.next_stage is Stage.APPROVAL
    assert record.status is RunStatus.RUNNING
    step = record.steps[-1]
    assert step.stage is Stage.POLICY
    assert step.output == to_jsonable(result)
    [hit] = store.get_run("run-tax").steps[-1].output["hits"]
    assert hit["rule_id"] == "POL-TAX-AMOUNT" and hit["severity"] == "error"
    assert step.output["retrieved"] == ["vendor:V-1003", "tax:US-WA-SVC"]


def test_record_policy_accepts_bare_hits_and_fills_run_id(agent, store):
    for run_id, payload in [
        ("run-hits", [PolicyHit("POL-X", Severity.INFO, "note")]),
        ("run-none", agent.check(invoice("clean_baseline"))),
    ]:
        record_intake(store, IntakeAgent().extract("clean_baseline", run_id=run_id))
        output = record_policy(store, run_id, payload).steps[-1].output
        assert output["run_id"] == run_id
    assert store.get_run("run-hits").steps[-1].output["hits"][0]["rule_id"] == "POL-X"


def test_record_policy_rejects_other_run_and_wrong_stage(agent, store):
    record_intake(store, IntakeAgent().extract("clean_baseline", run_id="run-a"))
    other = PolicyResult("run-b", ())
    with pytest.raises(ValueError, match="run-b"):
        record_policy(store, "run-a", other)
    record_policy(store, "run-a", agent.check(invoice("clean_baseline")))
    with pytest.raises(ValueError, match="expected stage"):
        record_policy(store, "run-a", [])


def test_policy_result_rejects_non_hits():
    with pytest.raises(TypeError, match="PolicyHit"):
        PolicyResult("run-a", ({"rule_id": "x"},))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "retrieved, message",
    [("vendor:V-1001", "bare string"), (("vendor:V-1001", 7), "expected str, got int")],
)
def test_policy_result_rejects_bad_retrieved(retrieved, message):
    with pytest.raises(TypeError, match=message):
        PolicyResult("run-a", (), retrieved)  # type: ignore[arg-type]


def test_policy_result_snapshots_retrieved_list():
    ids = ["vendor:V-1001", "tax:US-TX-STD"]
    result = PolicyResult("run-a", (), ids)  # type: ignore[arg-type]
    ids.append("vendor:V-9999")
    assert result.retrieved == ("vendor:V-1001", "tax:US-TX-STD")
