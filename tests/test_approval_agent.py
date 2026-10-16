"""ApprovalAgent: discrepancy rules, outcome rubric, RunStore final step, E2E pipeline, resume."""

import dataclasses
import json
import socket
from decimal import Decimal

import pytest

from ledgercheck.agents import (
    ApprovalAgent,
    IntakeAgent,
    PolicyAgent,
    PolicyResult,
    record_approval,
    record_intake,
    record_policy,
    resume_run,
    run_pipeline,
)
from ledgercheck.agents.approval import (
    decide_outcome,
    normalize_invoice_number,
    prior_invoices,
    rule_duplicate,
    rule_line_amounts,
    rule_missing_po,
    rule_po_amount,
    rule_policy_hits,
    rule_subtotal,
    rule_tax,
    rule_total,
)
from ledgercheck.agents.policy import RERANK_FLAG
from ledgercheck.fixtures_loader import FIXTURES_DIR, load_case, load_cases
from ledgercheck.models import Outcome, PolicyHit, Severity
from ledgercheck.observability import NullTracer
from ledgercheck.run_store import RunNotFound, RunStatus, RunStore, Stage

# Every invoice fixture: expected outcome and rule ids (approval rules, then policy hits).
EXPECTED = {
    "clean_baseline": (Outcome.APPROVE, []),
    "credit_note": (Outcome.APPROVE, ["APR-PO-MISSING"]),  # INFO for a credit note
    "duplicate_resubmission": (Outcome.APPROVE, []),  # no prior invoices, nothing to duplicate
    "european_format_text": (Outcome.APPROVE, []),
    "gbp_cloud_services": (Outcome.APPROVE, []),
    "missing_po": (Outcome.FLAG, ["APR-PO-MISSING", "POL-PO-REQUIRED"]),
    "ocr_noise_text": (Outcome.APPROVE, []),
    "rounding": (Outcome.FLAG, ["APR-TAX-ROUNDING", "POL-TAX-ROUNDING"]),
    "subtotal_mismatch": (Outcome.NEEDS_HUMAN, ["APR-SUBTOTAL"]),
    "tax_mismatch": (Outcome.NEEDS_HUMAN, ["APR-TAX-AMOUNT", "POL-TAX-AMOUNT"]),
    "vendor_alias": (Outcome.APPROVE, ["POL-VENDOR-NO-ID"]),
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
def store(tmp_path):
    return RunStore(tmp_path / "runs")


def invoice(case_id, **changes):
    inv = load_case(FIXTURES_DIR / f"{case_id}.json").invoice
    return dataclasses.replace(inv, **changes)


def ids(hits):
    return [h.rule_id for h in hits]


def test_expected_covers_every_fixture():
    assert set(EXPECTED) == {c.case_id for c in load_cases()}


@pytest.mark.parametrize("case_id", sorted(EXPECTED))
def test_pipeline_flags_each_fixture(case_id):
    outcome, rules = EXPECTED[case_id]
    result = run_pipeline(case_id)
    assert result.decision.outcome is outcome
    assert ids(result.decision.hits) == rules
    assert result.decision.run_id == result.extraction.run_id == result.policy.run_id
    assert result.decision.invoice_number == result.extraction.invoice.invoice_number
    assert result.record is None


def test_clean_invoice_approves_with_no_hits_or_reasons():
    decision = run_pipeline("clean_baseline").decision
    assert decision.outcome is Outcome.APPROVE
    assert decision.hits == () and decision.reasons == ()
    assert decision.decided_by == "rules"
    assert not decision.requires_review


def test_reasons_list_warning_and_error_hits_only():
    decision = run_pipeline("missing_po").decision
    assert decision.reasons == (
        "APR-PO-MISSING: no PO number on the invoice",
        "POL-PO-REQUIRED: Bluepeak Logistics invoices must reference a purchase order",
    )
    assert run_pipeline("vendor_alias").decision.reasons == ()


@pytest.mark.parametrize("po", [None, "", "   "])
def test_missing_po_warns(po):
    (hit,) = rule_missing_po(invoice("clean_baseline", po_number=po))
    assert (hit.rule_id, hit.severity, hit.field) == ("APR-PO-MISSING", Severity.WARNING, "po_number")


def test_missing_po_is_info_for_credit_note():
    (hit,) = rule_missing_po(invoice("credit_note"))
    assert hit.severity is Severity.INFO


def test_missing_po_silent_when_po_stated():
    assert rule_missing_po(invoice("clean_baseline")) == ()


BOOK = {"PO-4500-1182": Decimal("1095.99")}


def test_po_amount_within_po_passes():
    assert rule_po_amount(invoice("clean_baseline"), BOOK) == ()
    assert rule_po_amount(invoice("clean_baseline"), {"PO-4500-1182": Decimal("5000")}) == ()


def test_po_amount_over_po_is_error():
    (hit,) = rule_po_amount(invoice("clean_baseline", total=Decimal("1096.00")), BOOK)
    assert (hit.rule_id, hit.severity) == ("APR-PO-AMOUNT", Severity.ERROR)
    assert (hit.expected, hit.observed) == ("1095.99", "1096.00")


def test_po_not_in_book_warns():
    (hit,) = rule_po_amount(invoice("rounding"), BOOK)
    assert (hit.rule_id, hit.severity, hit.observed) == (
        "APR-PO-UNKNOWN", Severity.WARNING, "PO-4500-1201"
    )


@pytest.mark.parametrize("padded", [" PO-4500-1182 ", "\tPO-4500-1182\n", "PO-4500-1182  "])
def test_padded_po_matches_book(padded):
    inv = invoice("clean_baseline", po_number=padded)
    assert rule_po_amount(inv, BOOK) == ()
    assert rule_missing_po(inv) == ()
    (hit,) = rule_po_amount(dataclasses.replace(inv, total=Decimal("1096.00")), BOOK)
    assert (hit.rule_id, hit.expected) == ("APR-PO-AMOUNT", "1095.99")


@pytest.mark.parametrize("variant", ["PO 4500-1182", "po-4500-1182"])
def test_po_lookup_keeps_inner_spacing_and_case(variant):
    (hit,) = rule_po_amount(invoice("clean_baseline", po_number=f" {variant} "), BOOK)
    assert (hit.rule_id, hit.observed) == ("APR-PO-UNKNOWN", variant)


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_padded_blank_po_still_missing(blank):
    inv = invoice("clean_baseline", po_number=blank)
    assert ids(rule_missing_po(inv)) == ["APR-PO-MISSING"]
    assert rule_po_amount(inv, BOOK) == ()


@pytest.mark.parametrize(
    "inv, book",
    [
        (invoice("clean_baseline"), None),  # no book
        (invoice("missing_po"), BOOK),  # no PO: rule_missing_po's job
        (invoice("clean_baseline", po_number="  "), BOOK),
        (invoice("credit_note", po_number="PO-4500-1182"), {}),  # credit note
    ],
)
def test_po_amount_nothing_to_check(inv, book):
    assert rule_po_amount(inv, book) == ()


def test_subtotal_mismatch_is_error():
    (hit,) = rule_subtotal(invoice("subtotal_mismatch"))
    assert (hit.rule_id, hit.severity) == ("APR-SUBTOTAL", Severity.ERROR)
    assert (hit.expected, hit.observed) == ("833.00", "858.00")


def test_subtotal_agrees_on_clean_invoice():
    assert rule_subtotal(invoice("clean_baseline")) == ()


def test_line_amount_a_cent_off_is_error():
    first, *rest = invoice("clean_baseline").line_items
    off = dataclasses.replace(first, unit_price=Decimal("42.5000"), amount=Decimal("510.01"))
    (hit,) = rule_line_amounts(invoice("clean_baseline", line_items=(off, *rest)))
    assert (hit.rule_id, hit.severity) == ("APR-LINE-AMOUNT", Severity.ERROR)
    assert (hit.expected, hit.observed) == ("510.00", "510.01")
    assert hit.message == "line 1 amount 510.01 but 12 x 42.5000 is 510.00"


def test_line_amount_sub_cent_gap_is_rounding():
    first, *rest = invoice("clean_baseline").line_items
    third = dataclasses.replace(first, quantity=Decimal("3"), unit_price=Decimal("0.335"),
                                amount=Decimal("1.00"))
    assert rule_line_amounts(invoice("clean_baseline", line_items=(third, *rest))) == ()


def test_total_mismatch_is_error():
    inv = invoice("clean_baseline")
    (hit,) = rule_total(dataclasses.replace(inv, total=inv.total + Decimal("0.01")))
    assert (hit.rule_id, hit.severity) == ("APR-TOTAL", Severity.ERROR)
    assert (hit.expected, hit.observed) == (str(inv.total), str(inv.total + Decimal("0.01")))


def test_tax_mismatch_is_error():
    (hit,) = rule_tax(invoice("tax_mismatch"))
    assert (hit.rule_id, hit.severity) == ("APR-TAX-AMOUNT", Severity.ERROR)
    assert (hit.expected, hit.observed) == ("435.90", "465.96")


def test_tax_one_cent_off_is_rounding_warning():
    (hit,) = rule_tax(invoice("rounding"))
    assert (hit.rule_id, hit.severity) == ("APR-TAX-ROUNDING", Severity.WARNING)
    assert (hit.expected, hit.observed) == ("4.85", "4.86")


def test_tax_two_cents_off_is_error():
    (hit,) = rule_tax(invoice("rounding", tax_amount=Decimal("4.87")))
    assert hit.rule_id == "APR-TAX-AMOUNT"


def test_tax_without_stated_rate_is_not_checked():
    assert rule_tax(invoice("tax_mismatch", tax_rate=None)) == ()
    assert rule_tax(invoice("clean_baseline")) == ()


def test_invoice_number_ignores_punctuation_case_and_width():
    forms = ["INV-NW-20417", "inv nw 20417", "INVNW20417", "ＩＮＶ－ＮＷ－２０４１７"]
    assert {normalize_invoice_number(f) for f in forms} == {"invnw20417"}


def test_duplicate_matches_same_vendor_and_number():
    (hit,) = rule_duplicate(invoice("duplicate_resubmission"), [invoice("clean_baseline")])
    assert (hit.rule_id, hit.severity, hit.field, hit.observed) == (
        "APR-DUPLICATE", Severity.ERROR, "invoice_number", "INVNW20417"
    )


def test_duplicate_vendor_is_the_id_else_the_normalized_name():
    renamed = invoice("clean_baseline", vendor_name="Someone Else")
    assert ids(rule_duplicate(invoice("duplicate_resubmission"), [renamed])) == ["APR-DUPLICATE"]
    by_name = invoice("clean_baseline", vendor_id=None)
    variant = invoice("duplicate_resubmission", vendor_id=" ", vendor_name="NORTHWIND OFFICE SUPPLY LLC")
    assert ids(rule_duplicate(variant, [by_name])) == ["APR-DUPLICATE"]
    assert rule_duplicate(invoice("duplicate_resubmission"), [by_name]) == ()  # known limit


@pytest.mark.parametrize("changes", [{"vendor_id": "V-2002"}, {"invoice_number": "INV-NW-20418"}])
def test_duplicate_needs_both_vendor_and_number(changes):
    assert rule_duplicate(invoice("duplicate_resubmission"), [invoice("clean_baseline", **changes)]) == ()


@pytest.mark.parametrize("changes", [{"invoice_number": "--"}, {"vendor_id": None, "vendor_name": "LLC"}])
def test_duplicate_never_matches_an_empty_key(changes):
    blank = invoice("clean_baseline", **changes)
    assert rule_duplicate(blank, [blank]) == ()


def test_prior_invoices_keeps_only_earlier_runs_with_an_extraction(store):
    run_pipeline("clean_baseline", store=store, run_id="run-a")
    store.start_run("rounding", run_id="run-b")
    run_pipeline("missing_po", store=store, run_id="run-c")
    run_pipeline("clean_baseline", store=store, run_id="run-d")
    assert prior_invoices(store, before_run_id="run-d") == (
        invoice("clean_baseline"), invoice("missing_po"))
    assert prior_invoices(store, before_run_id="run-a") == ()
    assert len(prior_invoices(store)) == 3


def test_resume_of_an_earlier_run_ignores_a_later_duplicate(store):
    run_pipeline("clean_baseline", store=store, run_id="run-a")
    run_pipeline("duplicate_resubmission", store=store, run_id="run-b")
    store.apply_correction("run-a", "po_number", "PO-4500-1182", corrected_by="reviewer")
    result = resume_run(store, "run-a", prior=prior_invoices(store, before_run_id="run-a"))
    assert result.decision.hits == ()


def test_pipeline_flags_duplicate_of_a_stored_run(store):
    run_pipeline("clean_baseline", store=store, run_id="run-1")
    result = run_pipeline("duplicate_resubmission", store=store, run_id="run-2",
                          prior=prior_invoices(store))
    assert result.decision.outcome is Outcome.NEEDS_HUMAN
    assert ids(result.decision.hits) == ["APR-DUPLICATE"]
    assert store.get_run("run-2").status is RunStatus.NEEDS_HUMAN


def test_resume_with_store_history_does_not_match_itself(store):
    run_pipeline("duplicate_resubmission", store=store, run_id="run-d")
    store.apply_correction("run-d", "po_number", "PO-4500-1182", corrected_by="reviewer")
    result = resume_run(store, "run-d", prior=prior_invoices(store, before_run_id="run-d"))
    assert result.decision.outcome is Outcome.APPROVE and result.decision.hits == ()


def hit(severity, rule_id="POL-X"):
    return PolicyHit(rule_id, severity, "seeded")


@pytest.mark.parametrize(
    "severities, outcome",
    [
        ([], Outcome.APPROVE),
        ([Severity.INFO], Outcome.APPROVE),
        ([Severity.INFO, Severity.WARNING], Outcome.FLAG),
        ([Severity.WARNING, Severity.ERROR], Outcome.NEEDS_HUMAN),
    ],
)
def test_decide_outcome_rubric(severities, outcome):
    assert decide_outcome(tuple(hit(s) for s in severities)) is outcome


@pytest.mark.parametrize(
    "severity, outcome",
    [(Severity.INFO, Outcome.APPROVE), (Severity.WARNING, Outcome.FLAG),
     (Severity.ERROR, Outcome.NEEDS_HUMAN)],
)
def test_policy_hits_drive_outcome_on_clean_invoice(severity, outcome):
    policy = PolicyResult("run-1", (hit(severity),))
    assert rule_policy_hits(policy) == policy.hits
    decision = ApprovalAgent().decide(invoice("clean_baseline"), policy)
    assert decision.outcome is outcome
    assert ids(decision.hits) == ["POL-X"]
    assert decision.run_id == "run-1"


def test_decide_run_id_sources_must_agree():
    agent = ApprovalAgent()
    extraction = IntakeAgent().extract("clean_baseline", run_id="run-a")
    assert agent.decide(extraction, PolicyResult(None), run_id="run-a").run_id == "run-a"
    with pytest.raises(ValueError, match="exactly one run id"):
        agent.decide(extraction, PolicyResult("run-b"))
    with pytest.raises(ValueError, match="exactly one run id"):
        agent.decide(extraction, PolicyResult(None), run_id="run-b")
    with pytest.raises(ValueError, match="exactly one run id"):
        agent.decide(extraction.invoice, PolicyResult(None))


def test_po_book_values_are_decimals():
    agent = ApprovalAgent(po_amounts={"PO-1": "10.00"})
    assert agent.po_amounts["PO-1"] == Decimal("10.00")
    with pytest.raises(TypeError):
        ApprovalAgent(po_amounts={"PO-1": 10.0})
    with pytest.raises(ValueError):
        ApprovalAgent(po_amounts={"PO-1": "NaN"})


def test_pipeline_with_po_book_escalates_over_po():
    approval = ApprovalAgent(po_amounts={"PO-4500-1182": "1000.00"})
    decision = run_pipeline("clean_baseline", approval=approval).decision
    assert decision.outcome is Outcome.NEEDS_HUMAN
    assert ids(decision.hits) == ["APR-PO-AMOUNT"]


def test_pipeline_stays_offline_with_rerank_flag_and_unknown_vendor(tmp_path, monkeypatch):
    monkeypatch.setenv(RERANK_FLAG, "1")
    data = json.loads((FIXTURES_DIR / "clean_baseline.json").read_text(encoding="utf-8"))
    data["invoice"]["vendor_id"] = "V-9999"
    (tmp_path / "clean_baseline.json").write_text(json.dumps(data), encoding="utf-8")
    result = run_pipeline("clean_baseline", intake=IntakeAgent(fixtures_root=tmp_path))
    assert result.policy.reranker is None and result.policy.retrieved
    assert result.decision.outcome is Outcome.NEEDS_HUMAN
    assert ids(result.decision.hits) == ["POL-VENDOR-UNKNOWN"]


@pytest.mark.parametrize(
    "case_id, status",
    [
        ("clean_baseline", RunStatus.COMPLETED),
        ("missing_po", RunStatus.COMPLETED),
        ("tax_mismatch", RunStatus.NEEDS_HUMAN),
    ],
)
def test_pipeline_records_final_step(store, case_id, status):
    result = run_pipeline(case_id, store=store, run_id=f"run-{case_id}")
    record = store.get_run(f"run-{case_id}")
    assert result.record == record
    assert record.status is status
    assert [s.stage for s in record.steps] == [Stage.INTAKE, Stage.POLICY, Stage.APPROVAL]
    assert record.next_stage is None
    final = record.steps[-1].output
    assert final["outcome"] == EXPECTED[case_id][0].value
    assert [h["rule_id"] for h in final["hits"]] == EXPECTED[case_id][1]


def test_record_approval_after_manual_stages(store):
    extraction = IntakeAgent().extract("subtotal_mismatch", run_id="run-m")
    record_intake(store, extraction)
    policy = PolicyAgent(rerank=False).check(extraction)
    record_policy(store, "run-m", policy)
    record = record_approval(store, "run-m", ApprovalAgent().decide(extraction, policy))
    assert record.status is RunStatus.NEEDS_HUMAN
    assert record.steps[-1].stage is Stage.APPROVAL


def test_record_approval_rejects_wrong_run_and_wrong_stage(store):
    extraction = IntakeAgent().extract("clean_baseline", run_id="run-w")
    record_intake(store, extraction)
    decision = ApprovalAgent().decide(extraction, PolicyResult("run-w"))
    with pytest.raises(ValueError, match="expected stage policy"):
        record_approval(store, "run-w", decision)
    record_policy(store, "run-w", PolicyResult("run-w"))
    with pytest.raises(ValueError, match="not 'run-w'"):
        record_approval(store, "run-w", dataclasses.replace(decision, run_id="run-x"))
    assert store.get_run("run-w").status is RunStatus.RUNNING


def test_resume_from_policy_rechecks_corrected_invoice(store):
    run_pipeline("tax_mismatch", store=store, run_id="run-t")
    store.apply_correction("run-t", "tax_amount", "435.90", corrected_by="reviewer")
    store.apply_correction("run-t", "total", "6448.25", corrected_by="reviewer")
    result = resume_run(store, "run-t", tracer=NullTracer())
    assert result.extraction.invoice.tax_amount == Decimal("435.90")
    assert result.extraction.extractor == "fixture" and result.extraction.source == "tax_mismatch"
    assert result.policy.hits == () and result.decision.outcome is Outcome.APPROVE
    record = store.get_run("run-t")
    assert result.record == record and record.status is RunStatus.COMPLETED
    assert [s.stage for s in record.active_steps()] == [Stage.INTAKE, Stage.POLICY, Stage.APPROVAL]
    assert len(record.steps) == 5 and record.next_stage is None


def test_resume_from_approval_reuses_stored_policy_hits(store):
    run_pipeline("missing_po", store=store, run_id="run-p")
    store.apply_correction("run-p", "po_number", "PO-1", corrected_by="reviewer", resume_from="approval")
    result = resume_run(store, "run-p")
    assert ids(result.policy.hits) == ["POL-PO-REQUIRED"]  # stale on purpose: policy not re-run
    assert ids(result.decision.hits) == ["POL-PO-REQUIRED"]
    record = store.get_run("run-p")
    assert [s.stage for s in record.steps].count(Stage.POLICY) == 1
    assert record.steps[-1].output["outcome"] == Outcome.FLAG.value


def test_resume_refuses_finished_unknown_and_unstarted_runs(store):
    run_pipeline("clean_baseline", store=store, run_id="run-c")
    with pytest.raises(ValueError, match="nothing to resume"):
        resume_run(store, "run-c")
    store.start_run("clean_baseline", run_id="run-new")
    with pytest.raises(ValueError, match="nothing to resume"):
        resume_run(store, "run-new")
    with pytest.raises(RunNotFound):
        resume_run(store, "run-none")
