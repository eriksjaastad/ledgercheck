"""Load the synthetic invoice fixtures in ``ledgercheck/fixtures/invoices/``.

Every invoice is invented; vendor names, ids and PO numbers are fictional.
The fixtures ship inside the package (see ``package-data`` in
``pyproject.toml``), so the default root works from a source checkout and
from an installed wheel alike.

Layout
------
Each case is ``<case_id>.json``. A case whose source document is plain text
also has ``<case_id>.txt`` holding that text verbatim; its JSON ``invoice``
is then the extraction a correct intake stage should produce from the text.

JSON keys (all required, no others allowed):

``case_id``
    Must equal the file stem.
``description``
    One line: what makes this case tricky.
``tags``
    Non-empty list of strings drawn from ``KNOWN_TAGS``. A tag names a discrepancy the case is
    built to seed, so a case without ``tax_mismatch`` / ``subtotal_mismatch``
    / ``rounding`` must be arithmetically consistent; tests enforce this.
``invoice``
    The invoice as stated on the document, in ``Invoice.from_dict`` shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ledgercheck.models import Invoice

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "invoices"

KNOWN_TAGS = frozenset(
    {
        "clean",  # no discrepancy; the control case
        "missing_po",  # po_number absent from the document
        "tax_mismatch",  # tax_amount != subtotal * tax_rate
        "subtotal_mismatch",  # subtotal != sum of line item amounts
        "rounding",  # tax off by one cent from per-line rounding
        "weird_formatting",  # text source with odd number/date/label formats
        "foreign_currency",  # not USD
        "duplicate_candidate",  # likely resubmission of another case
        "credit_note",  # negative amounts
        "vendor_alias",  # vendor named differently / missing vendor_id
    }
)

_CASE_KEYS = {"case_id", "description", "tags", "invoice"}


class FixtureError(ValueError):
    """The fixture set is missing or a fixture file does not match the documented layout."""


@dataclass(frozen=True, slots=True)
class FixtureCase:
    case_id: str
    description: str
    tags: frozenset[str]
    invoice: Invoice
    path: Path
    raw_text: str | None = None

    @property
    def source_format(self) -> str:
        return "text" if self.raw_text is not None else "json"


def load_case(path: Path) -> FixtureCase:
    """Load one ``<case_id>.json`` (and its ``.txt`` sibling, if present)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:  # JSONDecodeError or UnicodeDecodeError
        raise FixtureError(f"{path.name}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict) or set(data) != _CASE_KEYS:
        got = sorted(data) if isinstance(data, dict) else type(data).__name__
        raise FixtureError(f"{path.name}: expected keys {sorted(_CASE_KEYS)}, got {got}")
    if data["case_id"] != path.stem:
        raise FixtureError(f"{path.name}: case_id {data['case_id']!r} != file stem")
    if not isinstance(data["description"], str):
        raise FixtureError(f"{path.name}: description must be a string")
    raw_tags = data["tags"]
    if not isinstance(raw_tags, list) or not all(isinstance(t, str) for t in raw_tags):
        raise FixtureError(f"{path.name}: tags must be a list of strings, got {raw_tags!r}")
    tags = frozenset(raw_tags)
    if not tags or tags - KNOWN_TAGS:
        raise FixtureError(f"{path.name}: unknown or empty tags {sorted(tags - KNOWN_TAGS)}")
    try:
        invoice = Invoice.from_dict(data["invoice"])
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        raise FixtureError(f"{path.name}: {exc}") from exc
    text_path = path.with_suffix(".txt")
    try:
        raw_text = text_path.read_text(encoding="utf-8") if text_path.exists() else None
    except UnicodeDecodeError as exc:
        raise FixtureError(f"{text_path.name}: not valid UTF-8: {exc}") from exc
    return FixtureCase(
        case_id=data["case_id"],
        description=data["description"],
        tags=tags,
        invoice=invoice,
        path=path,
        raw_text=raw_text,
    )


def load_cases(root: Path = FIXTURES_DIR) -> list[FixtureCase]:
    """Load every case under ``root``, sorted by case id.

    Raises ``FixtureError`` if ``root`` is not a directory or holds no
    cases, for a malformed case, or for a ``.txt`` with no matching ``.json``.
    An empty result would look like a dataset with nothing to check, so it is
    an error rather than ``[]``.
    """
    if not root.is_dir():
        raise FixtureError(f"fixtures directory not found: {root}")
    if orphans := sorted(
        p.name for p in root.glob("*.txt") if not p.with_suffix(".json").exists()
    ):
        raise FixtureError(f"text fixtures without a .json case: {orphans}")
    cases = [load_case(p) for p in sorted(root.glob("*.json"))]
    if not cases:
        raise FixtureError(f"no fixture cases (*.json) in {root}")
    return cases
