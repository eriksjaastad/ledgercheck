"""ledgercheck — CLI entry point.

Commands
--------
``judge``
    Score the golden suite against the judge rubric, offline by default.
    Exits 0 on pass, 1 below the pass rule and 2 on error; ``ledgercheck judge
    --help`` documents the rubric, thresholds, ``--case`` and the opt-in
    ``--live`` LLM judge (see ``ledgercheck.eval.judge``).
``connections``
    Show the resolved connection settings (provider, models, spend cap,
    tracing) and where each came from; the API key only as set or missing.
    ``ledgercheck connections --help`` explains how to supply a key and the
    order settings are read in (see ``ledgercheck.connections``).
``serve``
    Local web UI: run a fixture case, see its flags, correct a field and
    resume. Loopback only (127.0.0.1 by default), offline (fixture path,
    ``NullTracer`` unless Langfuse keys are set), runs stored under
    ``--runs-dir``; ``ledgercheck serve --help`` documents launch, options and
    routes (see ``ledgercheck.web``).
``beta-notes``
    Print the beta-feedback template, one note per document run (date, tester,
    run id, expected vs actual, flags, correction, latency/cost, model,
    rate-limit and context errors). It only prints to stdout: no files, no
    service. ``ledgercheck beta-notes --help`` lists the fields (see
    ``ledgercheck.beta_notes``).

Model routing (which model tier each LLM task uses, and the rate-limit and
context-window runbook) is not a command; see ``ledgercheck.agents.routing``.

Running ``ledgercheck`` with no subcommand prints this help and exits 0.

Options
-------
``--version``
    Print the package version and exit.
``-h`` / ``--help``
    Show this help and exit.
"""

from __future__ import annotations

import argparse
import sys

from ledgercheck import __version__, beta_notes, connections, web
from ledgercheck.eval import judge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ledgercheck",
        description=(
            "Invoice reconciliation pipeline (intake, policy check, approval flags) with a "
            "golden-dataset eval gate. Runs offline on bundled sample invoices; an LLM judge "
            "is opt-in with your own key (ledgercheck connections --help)."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    judge.build_parser(commands.add_parser("judge", help="score the golden suite (rubric gate)"))
    web.build_parser(commands.add_parser("serve", help="local web UI (loopback only, offline)"))
    beta_notes.build_parser(
        commands.add_parser("beta-notes", help="print the beta-feedback note template")
    )
    connections.build_parser(
        commands.add_parser("connections", help="show connection settings and their sources")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    connections.install_masked_excepthook()  # an unexpected crash never prints a secret
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "judge":
        return judge.run(args)
    if args.command == "serve":
        return web.run(args)
    if args.command == "beta-notes":
        return beta_notes.run(args)
    if args.command == "connections":
        return connections.run(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
