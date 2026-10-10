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
    rule_missing_po,
    rule_po_amount,
    rule_policy_hits,
    rule_subtotal,
    rule_tax,
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
    "duplicate_resubmission": (Outcome.APPROVE, []),  # duplicates not checked yet
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
