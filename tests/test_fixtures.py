"""Synthetic fixtures load, follow the documented layout, and seed what their tags claim."""

import json
import tomllib
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

import ledgercheck
from ledgercheck.fixtures_loader import (
    FIXTURES_DIR,
    KNOWN_TAGS,
    FixtureError,
    load_cases,
)

CASES = load_cases()
ARITHMETIC_SEEDS = {"tax_mismatch", "subtotal_mismatch", "rounding"}


def test_at_least_eight_cases() -> None:
    assert len(CASES) >= 8
    assert len({c.case_id for c in CASES}) == len(CASES)


def test_required_scenarios_are_covered() -> None:
    covered = set().union(*(c.tags for c in CASES))
    assert {"missing_po", "weird_formatting", "tax_mismatch", "clean"} <= covered
    assert covered == KNOWN_TAGS  # every documented tag has at least one case
    assert any(c.source_format == "text" for c in CASES)
    assert any(c.source_format == "json" for c in CASES)


def _assert_case_matches_tags(case) -> None:
    inv = case.invoice
    assert case.description.strip()
    assert inv.line_items
    assert all(li.amount == li.computed_amount() for li in inv.line_items)
    assert inv.total == inv.expected_total()
    assert (inv.po_number is None) == ("missing_po" in case.tags)
    assert (inv.currency != "USD") == ("foreign_currency" in case.tags)
    assert (inv.total < 0) == ("credit_note" in case.tags)

    subtotal_ok = inv.subtotal == inv.line_items_total()
    assert subtotal_ok != ("subtotal_mismatch" in case.tags)

    expected_tax = inv.expected_tax()
    if expected_tax is None:
        # No stated rate means no expected tax, so nothing to seed a tax discrepancy against.
        assert not {"rounding", "tax_mismatch"} & case.tags, "tax tags need a stated tax_rate"
    else:
        tax_gap = abs(inv.tax_amount - expected_tax)
        if "rounding" in case.tags:
            assert tax_gap == Decimal("0.01")
        elif "tax_mismatch" in case.tags:
            assert tax_gap > Decimal("0.01")
        else:
            assert tax_gap == 0

    if "clean" in case.tags:
        assert case.tags == {"clean"}


def _assert_text_has_ground_truth(case) -> None:
    text = case.raw_text
    assert "weird_formatting" in case.tags
    for value in (case.invoice.po_number, case.invoice.vendor_id):
        if value is not None:  # None means "not on the document", so nothing to find
            assert value in text


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.case_id)
def test_case_matches_its_tags(case) -> None:
    _assert_case_matches_tags(case)


@pytest.mark.parametrize(
    "case", [c for c in CASES if c.source_format == "text"], ids=lambda c: c.case_id
)
def test_text_source_contains_expected_values(case) -> None:
    """The ground-truth fields must be recoverable from the text, modulo formatting."""
    _assert_text_has_ground_truth(case)


def _case(case_id: str):
    return next(c for c in CASES if c.case_id == case_id)


def test_tag_checks_accept_omitted_optional_fields() -> None:
    """A case may omit tax_rate, po_number and vendor_id without the checks crashing."""
    base = _case("clean_baseline")
    no_rate = replace(base, invoice=replace(base.invoice, tax_rate=None))
    assert no_rate.invoice.expected_tax() is None
    _assert_case_matches_tags(no_rate)

    text = next(c for c in CASES if c.source_format == "text")
    bare = replace(
        text,
        tags=text.tags | {"missing_po", "vendor_alias"},
        invoice=replace(text.invoice, po_number=None, vendor_id=None, tax_rate=None),
    )
    _assert_case_matches_tags(bare)
    _assert_text_has_ground_truth(bare)


def test_tax_tag_without_rate_fails_clearly() -> None:
    base = _case("tax_mismatch")
    no_rate = replace(base, invoice=replace(base.invoice, tax_rate=None))
    with pytest.raises(AssertionError, match="tax tags need a stated tax_rate"):
        _assert_case_matches_tags(no_rate)


def test_duplicate_candidate_mirrors_another_case() -> None:
    for dup in (c for c in CASES if "duplicate_candidate" in c.tags):
        twins = [
            c
            for c in CASES
            if c is not dup
            and c.invoice.vendor_id == dup.invoice.vendor_id
            and c.invoice.total == dup.invoice.total
            and c.invoice.po_number == dup.invoice.po_number
        ]
        assert twins, dup.case_id
        assert all(t.invoice.invoice_number != dup.invoice.invoice_number for t in twins)


def test_default_root_is_inside_the_package() -> None:
    package_dir = Path(ledgercheck.__file__).resolve().parent
    assert FIXTURES_DIR.is_relative_to(package_dir)
    assert FIXTURES_DIR.is_dir()
    assert len(load_cases()) >= 8
    assert all(c.path.parent == FIXTURES_DIR for c in CASES)


def test_package_data_ships_every_fixture_file() -> None:
    """Each file in the fixtures dir matches a package-data glob, so a wheel carries it."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    globs = config["tool"]["setuptools"]["package-data"]["ledgercheck"]
    package_dir = Path(ledgercheck.__file__).resolve().parent
    shipped = {p for g in globs for p in package_dir.glob(g)}
    present = {p for p in FIXTURES_DIR.iterdir() if p.is_file()}
    assert present and present <= shipped, sorted(p.name for p in present - shipped)


def test_loader_rejects_missing_directory(tmp_path) -> None:
    with pytest.raises(FixtureError, match="not found"):
        load_cases(tmp_path / "absent")


def test_loader_rejects_a_file_as_root(tmp_path) -> None:
    root = tmp_path / "invoices"
    root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(FixtureError, match="not found"):
        load_cases(root)


def test_loader_rejects_empty_directory(tmp_path) -> None:
    with pytest.raises(FixtureError, match="no fixture cases"):
        load_cases(tmp_path)


GOOD = json.loads((FIXTURES_DIR / "clean_baseline.json").read_text(encoding="utf-8"))


def _good_with(**invoice_overrides) -> dict:
    return {"clean_baseline.json": {**GOOD, "invoice": {**GOOD["invoice"], **invoice_overrides}}}


@pytest.mark.parametrize(
    "files, error",
    [
        ({"renamed.json": GOOD}, "file stem"),
        ({"clean_baseline.json": {**GOOD, "tags": ["made_up"]}}, "tags"),
        (_good_with(total=1.0), "total"),
        ({"clean_baseline.json": GOOD, "orphan.txt": "no json twin"}, "orphan.txt"),
        ({"clean_baseline.json": {**GOOD, "tags": "clean"}}, "tags"),
        ({"clean_baseline.json": {**GOOD, "tags": [["clean"]]}}, "tags"),
        ({"clean_baseline.json": {**GOOD, "description": 7}}, "description"),
        ({"clean_baseline.json": {**GOOD, "invoice": ["invoice_number"]}}, "invoice"),
        (_good_with(total="NaN"), "total"),
        (_good_with(total="abc"), "total"),
        (_good_with(vendor_name=42), "vendor_name"),
        (
            _good_with(line_items=[{**GOOD["invoice"]["line_items"][0], "description": None}]),
            "description",
        ),
    ],
    ids=[
        "stem",
        "tags",
        "float",
        "orphan",
        "tags-str",
        "tags-unhashable",
        "description-type",
        "invoice-not-object",
        "nan",
        "not-a-number",
        "name-type",
        "line-description-type",
    ],
)
def test_loader_rejects_malformed_cases(tmp_path, files, error) -> None:
    for name, payload in files.items():
        text = payload if isinstance(payload, str) else json.dumps(payload)
        (tmp_path / name).write_text(text, encoding="utf-8")
    with pytest.raises(FixtureError, match=error):
        load_cases(tmp_path)
