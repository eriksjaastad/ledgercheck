"""Domain models shared by every Ledger Check pipeline stage.

Stdlib dataclasses only. Each stage produces one of these and the next stage
consumes it:

- ``Invoice`` / ``LineItem``  what an invoice document states.
- ``ExtractionResult``        intake output: an ``Invoice`` plus run id,
                              per-field confidence and extraction warnings.
- ``PolicyHit``               one rule the policy stage found triggered.
- ``ApprovalDecision``        the approval stage's outcome for one run.

Money
-----
Amounts, rates and quantities are ``decimal.Decimal``. In JSON (fixtures,
``to_jsonable`` output) they travel as strings such as ``"1234.50"`` so no
value ever passes through a float; ``Invoice.from_dict`` rejects floats
(``TypeError``) and unparseable or non-finite values such as ``"NaN"`` or
``"Infinity"`` (``ValueError``).

Stated vs. checked
------------------
An ``Invoice`` records what the document *says*, including arithmetic that
does not add up. ``line_items_total``, ``expected_tax`` and ``expected_total``
compute the checked values. Deciding whether a difference matters belongs to
the policy and approval stages, not to the model.
"""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import Any, Mapping

CENT = Decimal("0.01")
# Three uppercase ASCII letters (ISO 4217 shape); use with ``fullmatch``.
CURRENCY_RE = re.compile(r"[A-Z]{3}")


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Outcome(StrEnum):
    APPROVE = "approve"
    FLAG = "flag"  # payable, but a reviewer should see the hits
    NEEDS_HUMAN = "needs_human"  # blocked until a person corrects or confirms
    REJECT = "reject"


def _decimal(value: Any, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise TypeError(f"{name}: expected a decimal string, got {value!r}")
    try:
        d = Decimal(value)
    except InvalidOperation:
        raise ValueError(f"{name}: not a decimal number: {value!r}") from None
    if not d.is_finite():
        raise ValueError(f"{name}: must be finite, got {value!r}")
    return d


def _optional_decimal(value: Any, name: str) -> Decimal | None:
    return None if value is None else _decimal(value, name)


def _date(value: Any, name: str) -> date:
    # datetime subclasses date; reject it so a time component can't sneak in.
    if isinstance(value, str):
        try:
            value = date.fromisoformat(value)
        except ValueError:
            raise ValueError(f"{name}: not an ISO date: {value!r}") from None
    if type(value) is not date:
        raise TypeError(f"{name}: expected a date or ISO date string, got {value!r}")
    return value


def _optional_date(value: Any, name: str) -> date | None:
    return None if value is None else _date(value, name)


def _check_str(value: Any, name: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"{name}: expected a string, got {value!r}")


def _str_tuple(value: Any, name: str) -> tuple[str, ...]:
    # Snapshot so a caller's list can't change after validation; a bare string
    # would otherwise split into characters.
    if isinstance(value, str):
        raise TypeError(f"{name}: expected a sequence of strings, got a bare string")
    items = tuple(value)
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"{name}: expected str, got {type(item).__name__}")
    return items


def _check_keys(data: Mapping[str, Any], cls: type, what: str) -> None:
    if not isinstance(data, Mapping):
        raise ValueError(f"{what}: expected an object, got {type(data).__name__}")
    known = {f.name for f in dataclasses.fields(cls)}
    required = {
        f.name
        for f in dataclasses.fields(cls)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    }
    if unknown := set(data) - known:
        raise ValueError(f"{what}: unknown keys {sorted(unknown)}")
    if missing := required - set(data):
        raise ValueError(f"{what}: missing keys {sorted(missing)}")


def to_jsonable(obj: Any) -> Any:
    """Convert a model (or nested containers of them) to JSON-safe values.

    Decimals become strings, dates ISO strings, enums their values, and
    tuples lists. ``Invoice.from_dict`` accepts the result for invoices.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Mapping):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj


@dataclass(frozen=True, slots=True)
class LineItem:
    description: str
    quantity: Decimal
    unit_price: Decimal
    amount: Decimal  # as stated; may disagree with quantity * unit_price
    sku: str | None = None

    def __post_init__(self) -> None:
        _check_str(self.description, "description")
        _check_str(self.sku, "sku", optional=True)
        for name in ("quantity", "unit_price", "amount"):
            object.__setattr__(self, name, _decimal(getattr(self, name), name))

    def computed_amount(self) -> Decimal:
        return (self.quantity * self.unit_price).quantize(CENT, ROUND_HALF_UP)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LineItem:
        _check_keys(data, cls, "line item")
        return cls(
            description=data["description"],
            quantity=_decimal(data["quantity"], "quantity"),
            unit_price=_decimal(data["unit_price"], "unit_price"),
            amount=_decimal(data["amount"], "amount"),
            sku=data.get("sku"),
        )


@dataclass(frozen=True, slots=True)
class Invoice:
    """One invoice exactly as the document states it.

    ``po_number``, ``vendor_id``, ``tax_rate`` and ``due_date`` are optional
    because real invoices omit them; ``None`` means "not on the document".
    ``currency`` is three uppercase ASCII letters (ISO 4217 shape; not checked
    against the ISO list). Credit notes carry negative amounts.
    """

    invoice_number: str
    vendor_name: str
    invoice_date: date
    currency: str
    line_items: tuple[LineItem, ...]
    subtotal: Decimal
    tax_amount: Decimal
    total: Decimal
    vendor_id: str | None = None
    po_number: str | None = None
    tax_rate: Decimal | None = None
    due_date: date | None = None

    def __post_init__(self) -> None:
        for name in ("invoice_number", "vendor_name", "currency"):
            _check_str(getattr(self, name), name)
        for name in ("vendor_id", "po_number"):
            _check_str(getattr(self, name), name, optional=True)
        if not self.invoice_number.strip():
            raise ValueError("invoice_number must not be blank")
        if not self.vendor_name.strip():
            raise ValueError("vendor_name must not be blank")
        if not CURRENCY_RE.fullmatch(self.currency):
            raise ValueError(
                f"currency must be three uppercase letters A-Z, got {self.currency!r}"
            )
        for name in ("subtotal", "tax_amount", "total"):
            object.__setattr__(self, name, _decimal(getattr(self, name), name))
        object.__setattr__(self, "tax_rate", _optional_decimal(self.tax_rate, "tax_rate"))
        object.__setattr__(self, "invoice_date", _date(self.invoice_date, "invoice_date"))
        object.__setattr__(self, "due_date", _optional_date(self.due_date, "due_date"))
        line_items = tuple(self.line_items)
        for li in line_items:
            if not isinstance(li, LineItem):
                raise TypeError(f"line_items: expected LineItem, got {type(li).__name__}")
        object.__setattr__(self, "line_items", line_items)

    def line_items_total(self) -> Decimal:
        return sum((li.amount for li in self.line_items), Decimal("0"))

    def expected_tax(self) -> Decimal | None:
        """Tax on the stated subtotal at the stated rate, rounded half-up to cents."""
        if self.tax_rate is None:
            return None
        return (self.subtotal * self.tax_rate).quantize(CENT, ROUND_HALF_UP)

    def expected_total(self) -> Decimal:
        return self.subtotal + self.tax_amount

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Invoice:
        """Build from the JSON shape used by fixtures; unknown keys are an error."""
        _check_keys(data, cls, "invoice")
        due = data.get("due_date")
        return cls(
            invoice_number=data["invoice_number"],
            vendor_name=data["vendor_name"],
            invoice_date=date.fromisoformat(data["invoice_date"]),
            currency=data["currency"],
            line_items=tuple(LineItem.from_dict(li) for li in data["line_items"]),
            subtotal=_decimal(data["subtotal"], "subtotal"),
            tax_amount=_decimal(data["tax_amount"], "tax_amount"),
            total=_decimal(data["total"], "total"),
            vendor_id=data.get("vendor_id"),
            po_number=data.get("po_number"),
            tax_rate=_optional_decimal(data.get("tax_rate"), "tax_rate"),
            due_date=None if due is None else date.fromisoformat(due),
        )


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Intake stage output for one run.

    ``source`` names what was read (a fixture case id or file path);
    ``extractor`` names what read it (``"fixture"`` for the no-LLM path).
    ``field_confidence`` maps ``Invoice`` field names to 0.0–1.0 (stored as a
    read-only mapping of floats); absent fields mean the extractor did not
    report a confidence.
    """

    run_id: str
    invoice: Invoice
    source: str
    extractor: str = "fixture"
    field_confidence: Mapping[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        known = {f.name for f in dataclasses.fields(Invoice)}
        confidence: dict[str, float] = {}
        for name, score in dict(self.field_confidence).items():
            if name not in known:
                raise ValueError(f"field_confidence: {name!r} is not an Invoice field")
            if isinstance(score, bool) or not isinstance(score, (int, float, Decimal)):
                raise TypeError(f"field_confidence[{name!r}]: expected a number, got {score!r}")
            score = float(score)
            if not (math.isfinite(score) and 0.0 <= score <= 1.0):
                raise ValueError(f"field_confidence[{name!r}] must be within 0..1")
            confidence[name] = score
        object.__setattr__(self, "field_confidence", MappingProxyType(confidence))
        object.__setattr__(self, "warnings", _str_tuple(self.warnings, "warnings"))


@dataclass(frozen=True, slots=True)
class PolicyHit:
    """One triggered policy rule; ``expected``/``observed`` are display strings."""

    rule_id: str
    severity: Severity
    message: str
    field: str | None = None
    expected: str | None = None
    observed: str | None = None

    def __post_init__(self) -> None:
        # Plain strings equal enum values but are not the members; coerce so
        # identity checks downstream hold. Unknown values raise ValueError.
        object.__setattr__(self, "severity", Severity(self.severity))


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """Approval stage outcome. An ERROR-severity hit can never be approved."""

    run_id: str
    invoice_number: str
    outcome: Outcome
    hits: tuple[PolicyHit, ...] = ()
    reasons: tuple[str, ...] = ()
    decided_by: str = "rules"  # "rules", a model name, or "human:<who>"

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", Outcome(self.outcome))
        # Snapshot to tuples before checking: a list could be mutated after the
        # check and a generator would be consumed by it.
        hits = tuple(self.hits)
        for h in hits:
            if not isinstance(h, PolicyHit):
                raise TypeError(f"hits: expected PolicyHit, got {type(h).__name__}")
        object.__setattr__(self, "hits", hits)
        object.__setattr__(self, "reasons", _str_tuple(self.reasons, "reasons"))
        if self.outcome is Outcome.APPROVE and any(
            h.severity is Severity.ERROR for h in self.hits
        ):
            raise ValueError("cannot approve with ERROR-severity policy hits")

    @property
    def requires_review(self) -> bool:
        return self.outcome is not Outcome.APPROVE
