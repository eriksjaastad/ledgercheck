"""Smoke: package imports, CLI help exits 0, and an unexpected error in any command exits 2
with one line and no traceback."""

import pytest

from ledgercheck import __version__, beta_notes, connections, web
from ledgercheck.cli import COMMANDS, main
from ledgercheck.eval import judge


def test_version_is_semver_stub() -> None:
    assert __version__ == "0.1.0"


def test_cli_help_exits_zero() -> None:
    assert main([]) == 0


def _fail(*args, **kwargs):
    raise RuntimeError("SECRET-in-message")


# one internal function per command, patched to fail as a bug or a broken dependency would
FAILURES = {"judge": (judge, "select_cases"), "serve": (web, "make_server"),
            "beta-notes": (beta_notes, "template"), "connections": (connections, "load_settings")}


@pytest.mark.parametrize("command", COMMANDS)  # a new command without a FAILURES entry fails
def test_unexpected_error_exits_2_without_a_traceback(command, monkeypatch, capsys) -> None:
    monkeypatch.setattr(*FAILURES[command], _fail)
    assert main([command]) == 2
    out, err = capsys.readouterr()
    assert "Traceback" not in out + err and "SECRET" not in out + err
    assert err.startswith(f"ledgercheck {command}: unexpected RuntimeError at ")
