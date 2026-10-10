"""Domain model invariants, arithmetic helpers and JSON round-trip."""

from datetime import date, datetime
from decimal import Decimal

import pytest

from ledgercheck.models import (
    ApprovalDecision,
    ExtractionResult,
    Invoice,
    LineItem,
    Outcome,
    PolicyHit,
    Severity,
    to_jsonable,
)


def _invoice_dict(**overrides):
    data = {
        "invoice_number": "T-1",
        "vendor_name": "Test Vendor",
        "invoice_date": "2026-09-01",
        "currency": "USD",
        "line_items": [
            {"description": "Widget", "quantity": "3", "unit_price": "10.25", "amount": "30.75"}
        ],
        "subtotal": "30.75",
        "tax_rate": "0.06",
        "tax_amount": "1.85",
        "total": "32.60",
    }
    data.update(overrides)
    return data


def test_from_dict_parses_decimals_and_dates() -> None:
    inv = Invoice.from_dict(_invoice_dict())
    assert inv.invoice_date == date(2026, 9, 1)
    assert inv.subtotal == Decimal("30.75")
    assert inv.po_number is None and inv.due_date is None and inv.vendor_id is None
    assert isinstance(inv.line_items, tuple)
    assert inv.line_items[0].computed_amount() == Decimal("30.75")


def test_arithmetic_helpers() -> None:
    inv = Invoice.from_dict(_invoice_dict())
    assert inv.line_items_total() == Decimal("30.75")
    assert inv.expected_tax() == Decimal("1.85")  # 1.845 rounds half-up
    assert inv.expected_total() == Decimal("32.60")
    assert Invoice.from_dict(_invoice_dict(tax_rate=None)).expected_tax() is None


def test_round_trip_through_jsonable() -> None:
    inv = Invoice.from_dict(_invoice_dict(po_number="PO-1", due_date="2026-10-01"))
    assert Invoice.from_dict(to_jsonable(inv)) == inv


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"subtotal": 30.75}, TypeError),  # floats are refused
        ({"extra": "x"}, ValueError),
        ({"currency": "usd"}, ValueError),
        ({"invoice_number": "  "}, ValueError),
        ({"total": "NaN"}, ValueError),
        ({"total": "12,50"}, ValueError),  # InvalidOperation surfaces as ValueError
        ({"invoice_number": 1001}, ValueError),
        ({"vendor_id": 1001}, ValueError),
        (
            {"line_items": [{"description": 3, "quantity": "1", "unit_price": "1", "amount": "1"}]},
            ValueError,
        ),
        ({"line_items": ["not an object"]}, ValueError),
    ],
)
def test_invoice_rejects_bad_input(overrides, error) -> None:
    with pytest.raises(error):
        Invoice.from_dict(_invoice_dict(**overrides))


def test_invoice_requires_core_fields() -> None:
    data = _invoice_dict()
    del data["total"]
    with pytest.raises(ValueError, match="missing keys"):
        Invoice.from_dict(data)


def test_extraction_result_validates_confidence() -> None:
    inv = Invoice.from_dict(_invoice_dict())
    ok = ExtractionResult(run_id="r1", invoice=inv, source="case", field_confidence={"total": 0.9})
    assert ok.extractor == "fixture" and ok.warnings == ()
    with pytest.raises(ValueError):
        ExtractionResult(run_id="r1", invoice=inv, source="case", field_confidence={"total": 1.5})
    with pytest.raises(ValueError):
        ExtractionResult(run_id="r1", invoice=inv, source="case", field_confidence={"nope": 0.5})
    with pytest.raises(ValueError):
        ExtractionResult(run_id="", invoice=inv, source="case")


def test_extraction_result_snapshots_confidence_and_warnings() -> None:
    inv = Invoice.from_dict(_invoice_dict())
    confidence = {"total": 0.9}
    warnings = ["blurry scan"]
    result = ExtractionResult(
        run_id="r1", invoice=inv, source="case", field_confidence=confidence, warnings=warnings
    )
    confidence["total"] = 5.0
    confidence["subtotal"] = 0.1
    warnings.append("late edit")
    assert dict(result.field_confidence) == {"total": 0.9}
    assert result.warnings == ("blurry scan",)
    with pytest.raises(TypeError):
        result.field_confidence["total"] = 0.1  # read-only view
    assert to_jsonable(result)["field_confidence"] == {"total": 0.9}
    from_decimal = ExtractionResult(
        run_id="r1", invoice=inv, source="case", field_confidence={"total": Decimal("0.5")}
    )
    assert from_decimal.field_confidence["total"] == 0.5
    kept = ExtractionResult(run_id="r1", invoice=inv, source="case", warnings=(w for w in ["w"]))
    assert kept.warnings == ("w",)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"field_confidence": {"total": float("nan")}}, ValueError),
        ({"field_confidence": {"total": Decimal("NaN")}}, ValueError),
        ({"field_confidence": {"total": -0.1}}, ValueError),
        ({"field_confidence": {"total": True}}, TypeError),
        ({"field_confidence": {"total": "0.5"}}, TypeError),
        ({"warnings": "blurry scan"}, TypeError),
        ({"warnings": ["ok", 3]}, TypeError),
    ],
)
def test_extraction_result_rejects_bad_confidence_and_warnings(kwargs, error) -> None:
    inv = Invoice.from_dict(_invoice_dict())
    with pytest.raises(error):
        ExtractionResult(run_id="r1", invoice=inv, source="case", **kwargs)


def _direct_invoice(**overrides):
    item = LineItem(
        description="Widget",
        quantity=Decimal("3"),
        unit_price=Decimal("10.25"),
        amount=Decimal("30.75"),
    )
    kwargs = {
        "invoice_number": "T-1",
        "vendor_name": "Test Vendor",
        "invoice_date": date(2026, 9, 1),
        "currency": "USD",
        "line_items": (item,),
        "subtotal": Decimal("30.75"),
        "tax_amount": Decimal("1.85"),
        "total": Decimal("32.60"),
    }
    kwargs.update(overrides)
    return Invoice(**kwargs)


def test_invoice_constructor_snapshots_line_items() -> None:
    item = _direct_invoice().line_items[0]
    items = [item]
    inv = _direct_invoice(line_items=items)
    items.append(item)
    assert inv.line_items == (item,)
    assert _direct_invoice(line_items=(li for li in [item])).line_items == (item,)
    assert inv == Invoice.from_dict(_invoice_dict(tax_rate=None))


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"subtotal": 30.75}, TypeError),
        ({"total": Decimal("NaN")}, ValueError),
        ({"tax_rate": 0.06}, TypeError),
        ({"line_items": [{"description": "Widget"}]}, TypeError),
        ({"line_items": "not items"}, TypeError),
    ],
)
def test_invoice_constructor_rejects_bad_money_and_items(overrides, error) -> None:
    with pytest.raises(error):
        _direct_invoice(**overrides)


def test_invoice_constructor_coerces_iso_date_strings() -> None:
    inv = _direct_invoice(invoice_date="2026-09-01", due_date="2026-10-01")
    assert type(inv.invoice_date) is date and inv.invoice_date == date(2026, 9, 1)
    assert type(inv.due_date) is date and inv.due_date == date(2026, 10, 1)
    assert _direct_invoice(due_date=None).due_date is None
    assert inv == Invoice.from_dict(_invoice_dict(tax_rate=None, due_date="2026-10-01"))


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"invoice_date": datetime(2026, 9, 1, 12, 30)}, TypeError),
        ({"invoice_date": "not a date"}, ValueError),
        ({"invoice_date": "2026-09-01T00:00:00"}, ValueError),
        ({"invoice_date": 20260901}, TypeError),
    ],
)
def test_invoice_constructor_rejects_bad_dates(overrides, error) -> None:
    with pytest.raises(error):
        _direct_invoice(**overrides)


@pytest.mark.parametrize(
    "field, value, error",
    [
        ("quantity", 3.0, TypeError),
        ("unit_price", Decimal("NaN"), ValueError),
        ("amount", Decimal("-Infinity"), ValueError),
        ("amount", True, TypeError),
    ],
)
def test_line_item_constructor_rejects_bad_money(field, value, error) -> None:
    kwargs = {
        "description": "Widget",
        "quantity": Decimal("3"),
        "unit_price": Decimal("10.25"),
        "amount": Decimal("30.75"),
    }
    kwargs[field] = value
    with pytest.raises(error):
        LineItem(**kwargs)


def test_approval_cannot_approve_over_error_hit() -> None:
    error = PolicyHit(rule_id="tax.rate", severity=Severity.ERROR, message="bad tax")
    warning = PolicyHit(rule_id="po.missing", severity=Severity.WARNING, message="no PO")
    with pytest.raises(ValueError):
        ApprovalDecision(run_id="r1", invoice_number="T-1", outcome=Outcome.APPROVE, hits=(error,))
    flagged = ApprovalDecision(
        run_id="r1", invoice_number="T-1", outcome=Outcome.FLAG, hits=(warning,)
    )
    assert flagged.requires_review
    assert to_jsonable(flagged)["hits"][0]["severity"] == "warning"


def test_approval_coerces_plain_string_enums() -> None:
    error = PolicyHit(rule_id="tax.rate", severity="error", message="bad tax")
    assert error.severity is Severity.ERROR
    with pytest.raises(ValueError):
        ApprovalDecision(run_id="r1", invoice_number="T-1", outcome="approve", hits=(error,))
    approved = ApprovalDecision(run_id="r1", invoice_number="T-1", outcome="approve")
    assert approved.outcome is Outcome.APPROVE
    assert not approved.requires_review
    with pytest.raises(ValueError):
        ApprovalDecision(run_id="r1", invoice_number="T-1", outcome="bogus")
    with pytest.raises(ValueError):
        PolicyHit(rule_id="x", severity="fatal", message="m")


def test_approval_snapshots_hits_list() -> None:
    warning = PolicyHit(rule_id="po.missing", severity=Severity.WARNING, message="no PO")
    error = PolicyHit(rule_id="tax.rate", severity=Severity.ERROR, message="bad tax")
    hits = [warning]
    reasons = ["po missing"]
    approved = ApprovalDecision(
        run_id="r1", invoice_number="T-1", outcome=Outcome.APPROVE, hits=hits, reasons=reasons
    )
    hits.append(error)
    reasons.append("late edit")
    assert approved.hits == (warning,)
    assert approved.reasons == ("po missing",)
    with pytest.raises(ValueError):
        ApprovalDecision(
            run_id="r1", invoice_number="T-1", outcome=Outcome.APPROVE, hits=[warning, error]
        )


def test_approval_rejects_duck_typed_hit() -> None:
    class FakeHit:
        severity = "error"

    with pytest.raises(TypeError):
        ApprovalDecision(
            run_id="r1", invoice_number="T-1", outcome=Outcome.APPROVE, hits=(FakeHit(),)
        )
    with pytest.raises(TypeError):
        ApprovalDecision(run_id="r1", invoice_number="T-1", outcome=Outcome.FLAG, reasons=(1,))
    with pytest.raises(TypeError):
        ApprovalDecision(run_id="r1", invoice_number="T-1", outcome=Outcome.FLAG, reasons="why")


def test_approval_generator_hits_checked_and_kept() -> None:
    error = PolicyHit(rule_id="tax.rate", severity=Severity.ERROR, message="bad tax")
    with pytest.raises(ValueError):
        ApprovalDecision(
            run_id="r1", invoice_number="T-1", outcome=Outcome.APPROVE, hits=(h for h in [error])
        )
    flagged = ApprovalDecision(
        run_id="r1",
        invoice_number="T-1",
        outcome=Outcome.FLAG,
        hits=(h for h in [error]),
        reasons=(r for r in ["bad tax"]),
    )
    assert flagged.hits == (error,)
    assert flagged.reasons == ("bad tax",)


def test_line_item_from_dict_optional_sku() -> None:
    item = LineItem.from_dict(
        {"description": "x", "quantity": "1", "unit_price": "2", "amount": "2", "sku": "S"}
    )
    assert item.sku == "S"
