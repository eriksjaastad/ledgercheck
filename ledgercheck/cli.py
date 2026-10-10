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

Running ``ledgercheck`` with no subcommand prints this help and exits 0. An
unexpected error in any command prints one line naming its type, the command and
where it was raised (never its message, which may quote a secret) and exits 2;
Ctrl-C exits 130. ``python -m ledgercheck.<module>`` runs the same commands.

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
import traceback
from pathlib import Path

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


COMMANDS = {"judge": judge.run, "serve": web.run, "beta-notes": beta_notes.run,
            "connections": connections.run}


def main(argv: list[str] | None = None) -> int:
    """Run one command with all output masked and exit 2, not a traceback, on any error."""
    connections.mask_output()  # nothing printed, not even an error line, shows a secret
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # its message may quote a secret: show its type and place only
        place = traceback.extract_tb(exc.__traceback__)[-1]
        print(f"ledgercheck {args.command}: unexpected {type(exc).__name__} at "
              f"{Path(place.filename).name}:{place.lineno}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
