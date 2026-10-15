"""Smoke: package imports, CLI help exits 0, an unexpected error in any command exits 2
with one line and no traceback, and known issues point at their docs."""

import ast
from pathlib import Path

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


# Each known issue is one short line naming the docstring or --help that holds its details.
ISSUE_DETAILS = {
    "ledgercheck.agents.intake": ("agents/intake.py", "No command reads an invoice of your own"),
    "ledgercheck connections --help": ("connections.py", "intake text extraction"),
    "ledgercheck.agents.approval": ("agents/approval.py", "only golden cases that carry a"),
    "ledgercheck serve --help": ("web.py", "There is no login"),
    "ledgercheck.observability": ("observability.py", "never a real Langfuse server"),
    "python3 scripts/deploy.py --help": ("../scripts/deploy.py", "``terraform validate``"),
}


def test_known_issues_point_to_the_docs_that_hold_the_details() -> None:
    package = Path(__file__).resolve().parents[1] / "ledgercheck"
    issues = (package.parent / "ISSUES.md").read_text(encoding="utf-8").splitlines()
    bullets = [line for line in issues if line.startswith("- ")]
    assert len(bullets) == len(ISSUE_DETAILS)
    for bullet, (where, (path, fact)) in zip(bullets, ISSUE_DETAILS.items()):
        assert f"`{where}`" in bullet and len(bullet) < 200
        tree = ast.parse((package / path).read_text(encoding="utf-8"))
        docs = (ast.get_docstring(node) or "" for node in ast.walk(tree)
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)))
        assert fact in " ".join(" ".join(docs).split()), where
