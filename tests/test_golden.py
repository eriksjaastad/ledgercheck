"""Golden cases parse, validate, cover the tricky scenarios, and match the offline pipeline."""

import copy
import json
import re
import socket
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

import ledgercheck
from ledgercheck import golden
from ledgercheck.agents import approval, policy
from ledgercheck.agents.approval import ApprovalAgent
from ledgercheck.agents.intake import IntakeAgent
from ledgercheck.agents.policy import PolicyAgent
from ledgercheck.fixtures_loader import load_cases
from ledgercheck.golden import (
    GOLDEN_DIR,
    GOLDEN_TAGS,
    UNREACHABLE_RULES,
    GoldenError,
    codes,
    coverage,
    load_golden_case,
    load_golden_cases,
    pinned,
    run_case,
)
from ledgercheck.models import Outcome, Severity
from ledgercheck.run_store import RunStatus, RunStore

CASES = load_golden_cases()
BY_ID = {c.case_id: c for c in CASES}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def test_at_least_50_unique_cases() -> None:
    assert len(CASES) >= 50
    assert len(BY_ID) == len(CASES)


def test_scenarios_are_covered() -> None:
    covered = set().union(*(c.tags for c in CASES))
    assert covered == GOLDEN_TAGS  # every documented tag has at least one case
    assert {c.expected_decision.outcome for c in CASES} >= {
        Outcome.APPROVE, Outcome.FLAG, Outcome.NEEDS_HUMAN
    }
    assert any(c.fixture.raw_text is not None and c.source_fixture is None for c in CASES)
    assert any(c.po_book is not None for c in CASES)


MATRIX_ROW = re.compile(r"^    (tag|rule|outcome) +(\S.*?) +(\d+)$", re.MULTILINE)


def _documented_matrix() -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {"tag": {}, "rule": {}, "outcome": {}}
    for kind, key, count in MATRIX_ROW.findall(golden.__doc__):
        assert key not in matrix[kind], f"duplicate matrix row {kind} {key}"
        matrix[kind][key] = int(count)
    return matrix


def test_coverage_matrix_matches_the_cases() -> None:
    documented = _documented_matrix()
    empty = [f"{kind} {key}" for kind, rows in documented.items() for key, n in rows.items()
             if n == 0]
    assert not empty, f"matrix rows with no case: {empty}"
    assert documented == {kind: dict(c) for kind, c in coverage(CASES).items()}


def test_coverage_matrix_lists_every_tag_and_reachable_rule() -> None:
    documented = _documented_matrix()
    assert set(documented["tag"]) == GOLDEN_TAGS
    rules = {key.split(" (")[0] for key in documented["rule"]}
    named = set(re.findall(r"\b(?:APR|POL)-[A-Z]+(?:-[A-Z]+)*\b",
                           approval.__doc__ + policy.__doc__))
    assert not rules & set(UNREACHABLE_RULES)  # a rule that fires must leave UNREACHABLE_RULES
    assert rules | set(UNREACHABLE_RULES) == named
    assert all(f"``{rule}``" in golden.__doc__ for rule in UNREACHABLE_RULES)


def test_source_fixtures_are_invoice_fixtures() -> None:
    # Reuse is where sensible, so a new invoice fixture needs no golden case.
    reused = {c.source_fixture for c in CASES} - {None}
    assert reused
    assert reused <= {f.case_id for f in load_cases()}


def test_reused_fixture_inherits_invoice_and_text() -> None:
    case = BY_ID["weird_formatting__german_decimal_comma_text"]
    (source,) = [f for f in load_cases() if f.case_id == case.source_fixture]
    assert case.fixture.invoice == source.invoice
    assert case.fixture.raw_text == source.raw_text
    assert case.fixture.case_id == case.case_id


@pytest.mark.parametrize(
    "case", [c for c in CASES if c.fixture.raw_text is not None], ids=lambda c: c.case_id
)
def test_text_source_contains_pinned_identifiers(case) -> None:
    """A text extractor is scored on the expected extraction, so the text must say it.

    Identifiers and line descriptions are copied verbatim into every text
    source; amounts, dates and names are reformatted, so they are not checked.
    """
    inv = case.expected_extraction
    pinned_text = [inv.invoice_number, inv.vendor_id, inv.po_number]
    for li in inv.line_items:
        pinned_text += [li.description, li.sku]
    missing = [v for v in pinned_text if v is not None and v not in case.fixture.raw_text]
    assert not missing, f"{case.path.name}: not in its text source: {missing}"


def _assert_matches(case, result) -> None:
    # Everything but run ids, hit messages and the message-built reasons.
    assert result.extraction.invoice == case.expected_extraction
    assert result.extraction.source == case.case_id
    policy = replace(result.policy, run_id=None, hits=pinned(result.policy.hits))
    assert policy == case.expected_policy
    decision = replace(result.decision, run_id=case.case_id, reasons=(),
                       hits=pinned(result.decision.hits))
    assert decision == case.expected_decision


@pytest.mark.parametrize("case_id", sorted(BY_ID))
def test_pipeline_matches_golden(case_id) -> None:
    case = BY_ID[case_id]
    _assert_matches(case, run_case(case))


def test_expected_extraction_is_the_whole_input_invoice() -> None:
    for case in CASES:
        assert case.expected_extraction == case.fixture.invoice, case.case_id
        assert case.expected_extraction.line_items, case.case_id


def _altered_item(**changes):
    class AlteredIntake(IntakeAgent):
        def extract(self, case, *, run_id=None):
            result = super().extract(case, run_id=run_id)
            first, *rest = result.invoice.line_items
            invoice = replace(result.invoice, line_items=(replace(first, **changes), *rest))
            return replace(result, invoice=invoice)

    return AlteredIntake()


@pytest.mark.parametrize("changes", [{"description": "Something else"}, {"quantity": "999"},
                                     {"unit_price": "0.01"}, {"sku": "X-1"}])
def test_line_item_regression_fails_the_case(changes) -> None:
    # Totals and hits are unchanged, so only the full invoice comparison catches it.
    case = BY_ID["clean__northwind_baseline"]
    result = run_case(case, intake=_altered_item(**changes))
    assert codes(result.decision.hits) == codes(case.expected_decision.hits)
    with pytest.raises(AssertionError):
        _assert_matches(case, result)


def test_hit_detail_regression_fails_the_case() -> None:
    case = BY_ID["tax_mismatch__two_cent_gap"]
    result = run_case(case)
    first, *rest = result.decision.hits
    wrong = replace(result.decision, hits=(replace(first, observed="6.03"), *rest))
    with pytest.raises(AssertionError):
        _assert_matches(case, replace(result, decision=wrong))


def test_run_case_records_on_a_store(tmp_path) -> None:
    store = RunStore(tmp_path / "runs")
    over = run_case(BY_ID["over_po__total_exceeds_book"], store=store)
    assert over.record is not None and over.record.status is RunStatus.NEEDS_HUMAN
    clean = run_case(BY_ID["clean__northwind_baseline"], store=store, run_id="run-golden-1")
    assert clean.record is not None and clean.record.status is RunStatus.COMPLETED
    assert clean.decision.run_id == "run-golden-1"


def test_run_case_uses_supplied_agents_without_building_defaults(monkeypatch) -> None:
    case = BY_ID["over_po__total_exceeds_book"]
    used = []

    class SpyPolicy(PolicyAgent):
        def check(self, extraction):
            used.append(extraction.source)
            return super().check(extraction)

    policy = SpyPolicy(rerank=False)
    approval = ApprovalAgent(po_amounts=case.po_book)

    def refuse(*args, **kwargs):
        raise AssertionError("default agent constructed")

    monkeypatch.setattr(golden, "PolicyAgent", refuse)
    monkeypatch.setattr(golden, "ApprovalAgent", refuse)
    result = run_case(case, policy=policy, approval=approval)
    assert used == [case.case_id]
    assert result.decision.outcome is case.expected_decision.outcome


def test_po_book_is_what_turns_on_the_amount_rule() -> None:
    case = BY_ID["over_po__total_exceeds_book"]
    assert run_case(case).decision.outcome is Outcome.NEEDS_HUMAN
    assert run_case(case, approval=ApprovalAgent()).decision.outcome is Outcome.APPROVE


def test_package_data_ships_every_golden_file() -> None:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    globs = tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["setuptools"][
        "package-data"
    ]["ledgercheck"]
    package_dir = Path(ledgercheck.__file__).resolve().parent
    shipped = {p for g in globs for p in package_dir.glob(g)}
    present = {p for p in GOLDEN_DIR.iterdir() if p.is_file()}
    assert present and present <= shipped, sorted(p.name for p in present - shipped)


def test_files_are_canonical_json() -> None:
    # Stable formatting keeps diffs of hand-edited cases readable.
    for case in CASES:
        text = case.path.read_text(encoding="utf-8")
        assert text == json.dumps(json.loads(text), indent=2, ensure_ascii=False) + "\n", case.path


# --- loader rejects malformed cases -----------------------------------------------------------

INLINE = json.loads((GOLDEN_DIR / "over_po__total_exceeds_book.json").read_text(encoding="utf-8"))
REF = json.loads((GOLDEN_DIR / "missing_po__freight_no_po.json").read_text(encoding="utf-8"))


def _write(root: Path, data: dict, name: str | None = None) -> Path:
    root.mkdir(exist_ok=True)
    path = root / f"{name or data['case_id']}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _mutated(base: dict, edit) -> dict:
    data = copy.deepcopy(base)
    edit(data)
    return data


def _rename(data: dict, case_id: str) -> None:
    data["case_id"] = case_id


BAD = {
    "extra key": (INLINE, lambda d: d.update(notes="x"), "expected keys"),
    "id not stem-shaped": (INLINE, lambda d: _rename(d, "overpo"), "primary_tag>__<variant"),
    "primary tag missing": (INLINE, lambda d: _rename(d, "clean__x"), "primary tag 'clean'"),
    "unknown tag": (INLINE, lambda d: d["tags"].append("bogus"), "unknown or empty tags"),
    "blank description": (INLINE, lambda d: d.update(description=" "), "description"),
    "both inputs": (REF, lambda d: d["input"].update(invoice=INLINE["input"]["invoice"]),
                    "exactly one of"),
    "unknown fixture": (REF, lambda d: d["input"].update(fixture="nope"), "no fixture case"),
    "fixture path": (REF, lambda d: d["input"].update(fixture="../invoices/rounding"),
                     "no fixture case"),
    "float amount": (INLINE, lambda d: d["input"]["invoice"].update(total=920.13), "total"),
    "float po_book": (INLINE, lambda d: d["input"].update(po_book={"PO-4500-1240": 900.0}),
                      "po_book"),
    "not an Invoice field": (INLINE, lambda d: d["expected"]["extraction"].update(vendor="x"),
                             "unknown keys"),
    "bad extraction value": (INLINE, lambda d: d["expected"]["extraction"].update(total="abc"),
                             "not a decimal"),
    "extraction without line items": (
        REF, lambda d: d["expected"]["extraction"].pop("line_items"),
        r"missing keys \['line_items'\]"),
    "extraction line item partial": (
        INLINE, lambda d: d["expected"]["extraction"]["line_items"][0].pop("quantity"),
        r"line item: missing keys \['quantity'\]"),
    "bad severity": (INLINE, lambda d: d["expected"]["approval"]["hits"][0].update(
        severity="fatal"), "fatal"),
    "bad outcome": (INLINE, lambda d: d["expected"]["approval"].update(outcome="maybe"), "maybe"),
    "approve with error": (INLINE, lambda d: d["expected"]["approval"].update(outcome="approve"),
                           "cannot approve"),
    "approval drops policy hits": (REF, lambda d: d["expected"]["approval"]["hits"].pop(),
                                   "must end with policy.hits"),
    "retrieved not strings": (INLINE, lambda d: d["expected"]["policy"].update(retrieved=[1]),
                              "retrieved"),
    "hit missing severity": (INLINE, lambda d: d["expected"]["approval"]["hits"][0].pop(
        "severity"), "expected keys"),
}


@pytest.mark.parametrize("label", sorted(BAD))
def test_loader_rejects(tmp_path, label) -> None:
    base, edit, match = BAD[label]
    data = _mutated(base, edit)
    path = _write(tmp_path / "golden", data, name=data["case_id"])
    with pytest.raises(GoldenError, match=match):
        load_golden_case(path)


SHAPE = {
    "policy hits not a list": (lambda d: d["expected"]["policy"].update(hits={}),
                               "expected.policy.hits: hits must be a list"),
    "approval hit not an object": (lambda d: d["expected"]["approval"]["hits"].append("x"),
                                   "expected.approval.hits[1]: expected an object, got str"),
    "blank rule_id": (lambda d: d["expected"]["approval"]["hits"][0].update(rule_id=" "),
                      "expected.approval.hits[0]: rule_id must be a non-blank string"),
    "retrieved an object": (lambda d: d["expected"]["policy"].update(
        retrieved={"vendor:V-1001": 1, "tax:US-TX-STD": 2}),
        "expected.policy.retrieved: retrieved must be a list"),
    "po_book null": (lambda d: d["input"].update(po_book=None),
                     "input: po_book must be an object of PO number -> amount"),
    "input line_items an object": (lambda d: d["input"]["invoice"].update(line_items={}),
                                   "input.invoice: line_items must be a list"),
    "extraction line_items an object": (
        lambda d: d["expected"]["extraction"].update(line_items={}),
        "expected.extraction: line_items must be a list"),
    "extraction not an object": (lambda d: d["expected"].update(extraction=[]),
                                 "expected.extraction: extraction must be an object"),
    "extraction missing total": (lambda d: d["expected"]["extraction"].pop("total"),
                                 "expected.extraction: invoice: missing keys ['total']"),
    "hit expected a number": (lambda d: d["expected"]["approval"]["hits"][0].update(expected=900),
                              "expected.approval.hits[0]: expected must be a string or null"),
}


@pytest.mark.parametrize("label", sorted(SHAPE))
def test_shape_errors_name_the_file(tmp_path, label) -> None:
    edit, msg = SHAPE[label]
    data = _mutated(INLINE, edit)
    _write(tmp_path / "golden", data)
    with pytest.raises(GoldenError) as err:
        load_golden_cases(tmp_path / "golden")
    assert str(err.value) == f"{data['case_id']}.json: {msg}"


def test_loader_rejects_case_id_other_than_stem(tmp_path) -> None:
    path = _write(tmp_path / "golden", INLINE, name="over_po__renamed")
    with pytest.raises(GoldenError, match="file stem"):
        load_golden_case(path)


def test_loader_rejects_text_next_to_a_fixture_reference(tmp_path) -> None:
    path = _write(tmp_path / "golden", REF)
    path.with_suffix(".txt").write_text("text", encoding="utf-8")
    with pytest.raises(GoldenError, match="fixture reference"):
        load_golden_case(path)


def test_loader_reads_text_next_to_an_inline_invoice(tmp_path) -> None:
    path = _write(tmp_path / "golden", INLINE)
    path.with_suffix(".txt").write_text("INVOICE INV-NW-20502", encoding="utf-8")
    assert load_golden_case(path).fixture.raw_text == "INVOICE INV-NW-20502"


def test_loader_rejects_missing_empty_and_orphaned_roots(tmp_path) -> None:
    with pytest.raises(GoldenError, match="not found"):
        load_golden_cases(tmp_path / "absent")
    with pytest.raises(GoldenError, match="no golden cases"):
        load_golden_cases(tmp_path)
    (tmp_path / "stray.txt").write_text("text", encoding="utf-8")
    with pytest.raises(GoldenError, match="without a .json"):
        load_golden_cases(tmp_path)


def test_codes_pairs_rule_and_severity() -> None:
    hits = run_case(BY_ID["missing_po__freight_no_po"]).decision.hits
    assert codes(hits) == (
        ("APR-PO-MISSING", Severity.WARNING), ("POL-PO-REQUIRED", Severity.WARNING)
    )
