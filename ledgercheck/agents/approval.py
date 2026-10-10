"""Approval stage: turn an extraction plus its policy result into a decision.

Every check is a plain, deterministic function of the invoice (and, for the
PO amount, of a purchase-order book), returning ``PolicyHit`` objects. The
approval rules check the document against itself; the policy stage has
already checked it against the vendor and tax-code records, and its hits are
carried into the decision unchanged. So one discrepancy can show up twice,
once per viewpoint: ``tax_mismatch`` has both ``APR-TAX-AMOUNT`` (charged tax
vs the document's own stated rate) and ``POL-TAX-AMOUNT`` (vs the vendor's
tax code).

Rules (``rule_id`` → severity)
------------------------------
- ``APR-PO-MISSING`` WARNING: no PO number on the document (missing, empty or
  whitespace only). INFO instead for a credit note (negative total), which
  references the original invoice rather than a PO.
- ``APR-PO-UNKNOWN`` WARNING: a PO book was supplied and the stated PO is not
  in it. The PO is looked up with leading and trailing whitespace stripped;
  any other difference from the book key (case, inner spaces) is a miss.
- ``APR-PO-AMOUNT`` ERROR: the invoice total exceeds the PO's amount. Billing
  less than the PO is partial billing, not a discrepancy. Amounts compare as
  is, so the book must be in the invoice currency.
- ``APR-SUBTOTAL`` ERROR: the stated subtotal differs from the sum of the
  stated line-item amounts by any amount.
- ``APR-TAX-ROUNDING`` WARNING / ``APR-TAX-AMOUNT`` ERROR: the charged tax
  differs from subtotal × stated rate (half-up to the cent) by at most one
  cent / by more. No stated rate means nothing to check here.
- Policy hits: passed through with their own severity.

Outcome rubric
--------------
``decide_outcome``: any ERROR hit → ``needs_human`` (escalate, payment
blocked); else any WARNING → ``flag`` (payable, a reviewer sees the hits);
else (no hits, or INFO only) → ``approve``. The rules never ``reject``; that
is left to a person. ``reasons`` lists ``"<rule_id>: <message>"`` for every
WARNING and ERROR hit, so an approval has no reasons.

Not checked yet: total vs subtotal + tax, line amount vs quantity × unit
price, dates (a due date before the invoice date approves), and duplicate
resubmissions (``duplicate_resubmission`` approves); no current fixture needs
the first two, and duplicates need cross-run history.

Pipeline
--------
``run_pipeline(case_id)`` runs intake → policy → approval on the fixture path
(no LLM, reranking off), traces each step (``ledgercheck.observability``; a
no-op without Langfuse keys) and, given a ``RunStore``, records all three
steps::

    result = run_pipeline("missing_po", store=RunStore())
    result.decision.outcome   # Outcome.FLAG
    result.record.status      # RunStatus.COMPLETED

``resume_run(store, run_id)`` continues a stored run from its ``next_stage``
(after ``RunStore.apply_correction`` that is ``resume_from``), on the stored,
corrected extraction; intake is never re-run. Resuming at approval reuses the
latest active policy step's hits instead of re-checking policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping

from ledgercheck import connections
from ledgercheck.agents.intake import IntakeAgent, record_intake
from ledgercheck.agents.policy import PolicyAgent, PolicyResult, _stated, record_policy
from ledgercheck.fixtures_loader import FixtureCase
from ledgercheck.models import (
    CENT,
    ApprovalDecision,
    ExtractionResult,
    Invoice,
    Outcome,
    PolicyHit,
    Severity,
    _decimal,
)
from ledgercheck.observability import PIPELINE_TRACE, Tracer, trace_run
from ledgercheck.run_store import RunRecord, RunStore, Stage, StepRecord


def rule_missing_po(invoice: Invoice) -> tuple[PolicyHit, ...]:
    """``APR-PO-MISSING``: no PO number stated.

    WARNING for an invoice; INFO for a credit note (negative total). A blank
    PO number counts as missing. A stated PO yields nothing.
    """
    if _stated(invoice.po_number) is not None:
        return ()
    if invoice.total < 0:
        return (PolicyHit(
            "APR-PO-MISSING", Severity.INFO,
            "credit note has no PO number; it should reference the original invoice",
            field="po_number",
        ),)
    return (PolicyHit(
        "APR-PO-MISSING", Severity.WARNING, "no PO number on the invoice", field="po_number"
    ),)


def rule_po_amount(
    invoice: Invoice, po_amounts: Mapping[str, Decimal] | None
) -> tuple[PolicyHit, ...]:
    """``APR-PO-UNKNOWN`` / ``APR-PO-AMOUNT``: the total against the PO book.

    Nothing to check (no hits) without a book, without a stated PO (see
    ``rule_missing_po``) or for a credit note. A PO not in the book is a
    WARNING; a total above the PO amount is an ERROR; a total at or below it
    passes.
    """
    po = _stated(invoice.po_number)
    if po_amounts is None or po is None or invoice.total < 0:
        return ()
    if po not in po_amounts:
        return (PolicyHit(
            "APR-PO-UNKNOWN", Severity.WARNING, f"PO {po} is not in the PO book",
            field="po_number", observed=po,
        ),)
    limit = po_amounts[po]
    if invoice.total > limit:
        return (PolicyHit(
            "APR-PO-AMOUNT", Severity.ERROR,
            f"total {invoice.total} exceeds PO {po} amount {limit}",
            field="total", expected=str(limit), observed=str(invoice.total),
        ),)
    return ()


def rule_subtotal(invoice: Invoice) -> tuple[PolicyHit, ...]:
    """``APR-SUBTOTAL`` ERROR: stated subtotal != sum of line-item amounts.

    Exact decimal comparison: line amounts are stated to the cent, so any gap
    is a real discrepancy, never rounding.
    """
    lines = invoice.line_items_total()
    if invoice.subtotal == lines:
        return ()
    return (PolicyHit(
        "APR-SUBTOTAL", Severity.ERROR,
        f"subtotal {invoice.subtotal} but line items add up to {lines}",
        field="subtotal", expected=str(lines), observed=str(invoice.subtotal),
    ),)


def rule_tax(invoice: Invoice) -> tuple[PolicyHit, ...]:
    """``APR-TAX-ROUNDING`` / ``APR-TAX-AMOUNT``: charged tax vs the stated rate.

    Expected tax is subtotal × stated rate, half-up to the cent. A gap of at
    most one cent is a WARNING (per-line rounding), a larger one an ERROR. No
    stated rate yields nothing; the policy stage checks the vendor's rate.
    """
    expected = invoice.expected_tax()
    if expected is None or invoice.tax_amount == expected:
        return ()
    rounding = abs(invoice.tax_amount - expected) <= CENT
    return (PolicyHit(
        "APR-TAX-ROUNDING" if rounding else "APR-TAX-AMOUNT",
        Severity.WARNING if rounding else Severity.ERROR,
        f"tax charged {invoice.tax_amount} but {invoice.tax_rate} on "
        f"{invoice.subtotal} is {expected}",
        field="tax_amount", expected=str(expected), observed=str(invoice.tax_amount),
    ),)


def rule_policy_hits(policy: PolicyResult) -> tuple[PolicyHit, ...]:
    """The policy stage's hits, unchanged; an ERROR among them escalates."""
    return policy.hits


def decide_outcome(hits: tuple[PolicyHit, ...]) -> Outcome:
    """ERROR → ``needs_human``, else WARNING → ``flag``, else ``approve``."""
    severities = {h.severity for h in hits}
    if Severity.ERROR in severities:
        return Outcome.NEEDS_HUMAN
    if Severity.WARNING in severities:
        return Outcome.FLAG
    return Outcome.APPROVE


class ApprovalAgent:
    """Applies the approval rules (see module docstring) to one invoice.

    ``po_amounts`` is an optional PO book, PO number → amount (a ``Decimal``
    or decimal string, in the invoice currency). Without it the PO amount rule
    is skipped; no PO book ships with the fixtures.
    """

    def __init__(self, *, po_amounts: Mapping[str, Decimal | str] | None = None) -> None:
        self.po_amounts = None if po_amounts is None else MappingProxyType(
            {po: _decimal(amount, f"po_amounts[{po!r}]") for po, amount in po_amounts.items()}
        )

    def decide(
        self,
        subject: Invoice | ExtractionResult,
        policy: PolicyResult,
        *,
        run_id: str | None = None,
    ) -> ApprovalDecision:
        """Decide on ``subject`` given its ``policy`` result.

        The run id comes from ``run_id``, the extraction and ``policy.run_id``;
        those given must agree and at least one must be given, otherwise
        ``ValueError``.
        """
        invoice = subject.invoice if isinstance(subject, ExtractionResult) else subject
        extracted = subject.run_id if isinstance(subject, ExtractionResult) else None
        given = {r for r in (run_id, extracted, policy.run_id) if r is not None}
        if len(given) != 1:
            raise ValueError(f"need exactly one run id, got {sorted(given)}")
        hits = (
            rule_missing_po(invoice)
            + rule_po_amount(invoice, self.po_amounts)
            + rule_subtotal(invoice)
            + rule_tax(invoice)
            + rule_policy_hits(policy)
        )
        return ApprovalDecision(
            run_id=given.pop(),
            invoice_number=invoice.invoice_number,
            outcome=decide_outcome(hits),
            hits=hits,
            reasons=tuple(
                f"{h.rule_id}: {h.message}" for h in hits if h.severity is not Severity.INFO
            ),
        )


def record_approval(store: RunStore, run_id: str, decision: ApprovalDecision) -> RunRecord:
    """Append ``decision`` as the approval (final) step of ``run_id``.

    ``RunStore.append_step`` sets the run's status: ``needs_human`` for that
    outcome, ``completed`` for any other. The run must be awaiting approval
    and the decision must be for this run, otherwise it raises
    (``RunNotFound`` / ``ValueError``).
    """
    return store.append_step(run_id, Stage.APPROVAL, decision)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Each stage's output; ``record`` is the final run, or ``None`` without a store."""

    extraction: ExtractionResult
    policy: PolicyResult
    decision: ApprovalDecision
    record: RunRecord | None = None


def run_pipeline(
    case_id: FixtureCase | str,
    *,
    store: RunStore | None = None,
    run_id: str | None = None,
    intake: IntakeAgent | None = None,
    policy: PolicyAgent | None = None,
    approval: ApprovalAgent | None = None,
    tracer: Tracer | None = None,
) -> PipelineResult:
    """Run intake → policy → approval on fixture ``case_id``, offline.

    ``case_id`` is anything ``IntakeAgent.extract`` takes: a case id or an
    already loaded ``FixtureCase`` (the golden dataset passes the latter).

    Defaults: ``IntakeAgent()`` (fixture path), ``PolicyAgent(rerank=False)``
    (whatever the env flag says) and ``ApprovalAgent()`` (no PO book). With a
    ``store`` each step is recorded as soon as it is produced, so a failure
    leaves the run at the failed stage.

    ``tracer`` defaults to ``connections.tracer()``: ``NullTracer`` (nothing
    recorded or sent) unless Langfuse keys are set. The run is one trace with
    spans ``intake``, ``policy`` and ``approval`` (see
    ``ledgercheck.observability``); tracing never changes the result.
    """
    intake = IntakeAgent() if intake is None else intake
    policy = PolicyAgent(rerank=False) if policy is None else policy
    approval = ApprovalAgent() if approval is None else approval
    tracer = connections.tracer() if tracer is None else tracer
    case_name = case_id.case_id if isinstance(case_id, FixtureCase) else case_id
    with trace_run(tracer, PIPELINE_TRACE, {"case_id": case_name}) as trace:
        with trace.span(Stage.INTAKE.value):
            extraction = intake.extract(case_id, run_id=run_id)
            if store is not None:
                record_intake(store, extraction)
        trace.annotate(run_id=extraction.run_id)
        with trace.span(Stage.POLICY.value):
            checked = policy.check(extraction)
            if store is not None:
                record_policy(store, extraction.run_id, checked)
        with trace.span(Stage.APPROVAL.value):
            decision = approval.decide(extraction, checked)
            record = None if store is None else record_approval(store, extraction.run_id, decision)
        trace.annotate(outcome=decision.outcome.value)
    return PipelineResult(extraction, checked, decision, record)


def _latest(record: RunRecord, stage: Stage) -> StepRecord | None:
    steps = [s for s in record.active_steps() if s.stage is stage]
    return steps[-1] if steps else None


def resume_run(
    store: RunStore,
    run_id: str,
    *,
    policy: PolicyAgent | None = None,
    approval: ApprovalAgent | None = None,
    tracer: Tracer | None = None,
) -> PipelineResult:
    """Run the remaining stages of stored run ``run_id`` from its ``next_stage``.

    The invoice is the run's ``extracted`` fields (corrections applied); the
    returned ``extraction`` wraps it with the run's source and the recorded
    extractor. From ``policy``: re-check policy, then decide. From
    ``approval``: rebuild the ``PolicyResult`` from the latest active policy
    step, then decide. Each new step is recorded with the ``record_*``
    helpers. A run with nothing left to run, or still awaiting intake, raises
    ``ValueError``; an unknown run raises ``RunNotFound``. Defaults and
    tracing match ``run_pipeline``; the trace carries ``resumed_from``.
    """
    record = store.get_run(run_id)
    stage, invoice = record.next_stage, record.invoice()
    if stage is None or stage is Stage.INTAKE or invoice is None:
        raise ValueError(f"run {run_id}: nothing to resume (next stage: {stage})")
    intake_step = _latest(record, Stage.INTAKE)
    extractor = "fixture" if intake_step is None else intake_step.output.get("extractor", "fixture")
    extraction = ExtractionResult(run_id, invoice, record.source, extractor=extractor)
    approval = ApprovalAgent() if approval is None else approval
    tracer = connections.tracer() if tracer is None else tracer
    meta = {"case_id": record.source, "run_id": run_id, "resumed_from": stage.value}
    with trace_run(tracer, PIPELINE_TRACE, meta) as trace:
        if stage is Stage.POLICY:
            policy = PolicyAgent(rerank=False) if policy is None else policy
            with trace.span(Stage.POLICY.value):
                checked = policy.check(extraction)
                record_policy(store, run_id, checked)
        else:
            step = _latest(record, Stage.POLICY)
            if step is None:
                raise ValueError(f"run {run_id}: no active policy step to resume approval from")
            out = step.output
            checked = PolicyResult(
                run_id,
                tuple(PolicyHit(**h) for h in out.get("hits", ())),
                tuple(out.get("retrieved", ())),
                out.get("reranker"),
            )
        with trace.span(Stage.APPROVAL.value):
            decision = approval.decide(extraction, checked)
            final = record_approval(store, run_id, decision)
        trace.annotate(outcome=decision.outcome.value)
    return PipelineResult(extraction, checked, decision, final)
