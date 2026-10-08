"""ledgercheck beta-notes — print the beta-feedback template for one document run.

A beta user or operator fills in one note per document run, so production-ish
problems are written down while they are fresh. The command only prints the
template to stdout. It writes no files, needs no service, and makes no network
call. Copy the output wherever you keep notes, and fill in one per run.

Fields
------
``date``
    Day of the run, YYYY-MM-DD.
``tester``
    Who ran it (beta user or operator).
``document``
    Fixture case id or document file name.
``run_id``
    Run-store id (shown by ``ledgercheck serve``), or none.
``expected``
    What the tester expected: outcome and key values.
``actual``
    Approval outcome and key extracted values.
``flags_shown``
    Rule ids and severities shown, or none.
``correction``
    Field, old -> new, and the reason, or none.
``latency``
    Seconds, if known.
``cost``
    USD, if known (mock runs cost 0).
``model``
    Router model(s) used, or ``mock``.
``rate_limit_errors``
    429s seen, retries, and fallback model used, or none.
``context_errors``
    Oversized or truncated input, escalation, or fell back to human review, or none.
``notes``
    Anything else worth retelling.

The rate-limit and context-window runbook these fields feed is in the
``ledgercheck.agents.routing`` module docstring.

Exit code 0.
"""

from __future__ import annotations

import argparse
import sys

FIELDS: tuple[tuple[str, str], ...] = (
    ("date", "YYYY-MM-DD"),
    ("tester", "beta user or operator"),
    ("document", "fixture case id or document file name"),
    ("run_id", "run-store id from ledgercheck serve, or none"),
    ("expected", "outcome and key values the tester expected"),
    ("actual", "approval outcome and key extracted values"),
    ("flags_shown", "rule ids and severities shown, or none"),
    ("correction", "field, old -> new, reason; or none"),
    ("latency", "seconds, if known"),
    ("cost", "USD, if known (mock runs cost 0)"),
    ("model", "router model(s) used, or mock"),
    ("rate_limit_errors", "429s seen, retries, fallback model used; or none"),
    ("context_errors", "oversized/truncated input, escalation, human review; or none"),
    ("notes", "anything else worth retelling"),
)


def template() -> str:
    """The note template: one ``field:`` line per field, with its hint as a comment."""
    width = max(len(name) for name, _ in FIELDS) + 2
    lines = ["# ledgercheck beta note: one per document run"]
    lines += [f"{name + ':':<{width}}# {hint}" for name, hint in FIELDS]
    return "\n".join(lines) + "\n"


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Set up the beta-notes parser (a new one by default); --help shows the docstring."""
    if parser is None:
        parser = argparse.ArgumentParser(prog="python -m ledgercheck.beta_notes")
    parser.description = __doc__
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    return parser


def run(args: argparse.Namespace) -> int:
    sys.stdout.write(template())
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
