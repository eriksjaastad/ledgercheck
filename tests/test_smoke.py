"""Smoke: package imports, CLI help exits 0, known issues point at their docs."""

import ast
from pathlib import Path

from ledgercheck import __version__
from ledgercheck.cli import main


def test_version_is_semver_stub() -> None:
    assert __version__ == "0.1.0"


def test_cli_help_exits_zero() -> None:
    assert main([]) == 0


# Each known issue is one short line naming the docstring or --help that holds its details.
ISSUE_DETAILS = {
    "ledgercheck.agents.intake": ("agents/intake.py", "No command reads an invoice of your own"),
    "ledgercheck connections --help": ("connections.py", "Only the LLM judge"),
    "ledgercheck.agents.approval": ("agents/approval.py", "only golden cases that carry a"),
    "ledgercheck.agents.policy": ("agents/policy.py", "surrounding whitespace is a miss"),
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
