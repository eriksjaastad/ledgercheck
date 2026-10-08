"""beta-notes: prints every template field, writes nothing, and the docstring matches FIELDS."""

import re

import pytest

from ledgercheck import beta_notes, cli
from ledgercheck.beta_notes import FIELDS

NAMES = [name for name, _ in FIELDS]


def test_beta_notes_prints_every_field_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["beta-notes"]) == 0
    out = capsys.readouterr().out
    printed = re.findall(r"^(\w+):", out, re.MULTILINE)
    assert printed == NAMES
    assert list(tmp_path.iterdir()) == []


def test_help_lists_every_field(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["beta-notes", "--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for name in NAMES:
        assert f"``{name}``" in text


def test_docstring_fields_section_matches_fields():
    section = beta_notes.__doc__.split("Fields\n------\n", 1)[1].split("\n\n", 1)[0]
    assert re.findall(r"^``(\w+)``$", section, re.MULTILINE) == NAMES


def test_cli_docstring_lists_beta_notes():
    assert "``beta-notes``" in (cli.__doc__ or "")
    assert "beta-notes" in cli.build_parser().format_help()
