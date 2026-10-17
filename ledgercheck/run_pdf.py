"""ledgercheck run-pdf — run one PDF invoice through intake, policy and approval.

What it reads
-------------
Only the PDF's text layer, read locally with pypdf (``ledgercheck.pdf_text``).
A PDF with any page that has no text layer (a scan) is refused: OCR is not
supported. An encrypted PDF is refused too; it is never decrypted.

``--text-only``
    Print the text layer to stdout and exit. Fully offline: no model, no spend
    gate, no run store.

Default: the full pipeline
--------------------------
Turning the text into invoice fields needs the live extraction model, so the
default run goes through the spend gate: ``LEDGERCHECK_LLM=1``, an
``OPENROUTER_API_KEY``, provider ``openrouter``, a small-tier model with prices
and a spend cap (see ``ledgercheck connections --help``). With the default
configuration the gate refuses, the command says so on stderr and nothing is
stored or spent; ``--text-only`` still works.

Past the gate the extracted invoice is policy-checked and decided by the same
rules as ``ledgercheck serve`` (no PO book), and the run is recorded as a JSON file under
``--runs-dir`` (default ``.scratch/runs``, the store ``ledgercheck serve``
uses); its source is the file name, never the directory. The duplicate check
compares the invoice with every run already in that store, so running the
same invoice twice flags ``APR-DUPLICATE``. The output is the run id and
outcome, then one ``rule_id severity: message`` line per hit.

Exit codes: 0 on a printed text layer or a decided run; 2 when the PDF has no
readable text layer, the spend gate refuses, or on any other error.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ledgercheck import connections
from ledgercheck.agents.approval import prior_invoices, run_pipeline
from ledgercheck.agents.intake import IntakeAgent, PdfIntake
from ledgercheck.agents.llm_client import InputTooLarge
from ledgercheck.connections import LiveLLMDisabled
from ledgercheck.pdf_text import PdfTextError, read_pdf_text
from ledgercheck.run_store import DEFAULT_ROOT


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Add the run-pdf options to ``parser`` (a new one by default); --help shows the docstring."""
    parser = parser or argparse.ArgumentParser()
    parser.description = __doc__
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.add_argument("file", metavar="FILE", help="PDF invoice with a text layer")
    parser.add_argument("--text-only", action="store_true",
                        help="print the text layer and exit (offline)")
    parser.add_argument("--runs-dir", default=str(DEFAULT_ROOT), metavar="DIR",
                        help="run-store directory")
    return parser


def run(args: argparse.Namespace, intake: IntakeAgent | None = None) -> int:
    """Print FILE's text layer, or run it through the pipeline and print the decision.

    ``intake`` replaces the default ``IntakeAgent()`` (tests inject a fake
    client). An unreadable PDF, text over the input budget or a refused gate prints one stderr line, exit 2.
    """
    connections.mask_output()  # also for Python callers
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # its recovery warnings quote file bytes
    path = Path(args.file)
    try:
        if args.text_only:
            print(read_pdf_text(path))
            return 0
        pdf_intake = PdfIntake(path.parent, IntakeAgent() if intake is None else intake)
        store = connections.run_store(args.runs_dir)
        prior = prior_invoices(store)  # read before this run is stored
        result = run_pipeline(path.name, store=store, intake=pdf_intake, prior=prior)
    except (PdfTextError, InputTooLarge) as exc:
        print(f"ledgercheck run-pdf: {exc}", file=sys.stderr)
        return 2
    except LiveLLMDisabled as exc:
        print(f"ledgercheck run-pdf: {exc}; --text-only prints the text layer offline",
              file=sys.stderr)
        return 2
    decision = result.decision
    print(f"run {decision.run_id}: {decision.outcome.value}")
    for hit in decision.hits:
        print(f"{hit.rule_id} {hit.severity.value}: {hit.message}")
    return 0


if __name__ == "__main__":
    from ledgercheck import cli
    sys.exit(cli.main(["run-pdf", *sys.argv[1:]]))
