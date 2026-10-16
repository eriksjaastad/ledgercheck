"""Intake stage: turn a source document into an ``ExtractionResult``.

Fixture path (the default, no LLM)
----------------------------------
``IntakeAgent.extract`` takes a ``FixtureCase`` or a case id and returns the
case's invoice as the extraction, with ``extractor="fixture"`` and
``source=<case_id>``. For a text-backed case (``<case_id>.txt``) the paired
JSON invoice *is* the expected extraction; the text is not parsed here.
``field_confidence`` and ``warnings`` are left empty: the fixture path reports
no confidence and finds nothing to warn about.

``ledgercheck serve`` and ``ledgercheck judge`` run the packaged fixture and
golden cases, picked by case id. From Python, ``IntakeAgent(fixtures_root=...)``
loads cases in the same format from another directory.

Live path (text only, never run live)
-------------------------------------
``IntakeAgent.extract_text`` sends raw text to an LLM client. Without an
injected client it builds ``llm_client.LLMClient``, which runs the spend gate
in ``ledgercheck.connections`` and raises ``LiveLLMDisabled`` unless live calls
are explicitly enabled, so the default configuration never spends tokens.
Past the gate it is built and tested offline against a fake server only.
``ledgercheck run-pdf`` calls it through ``extract_pdf``; from Python, with the
gate's env set (spends money, up to the spend cap)::

    agent = IntakeAgent(llm_client=LLMClient())
    case = load_case(FIXTURES_DIR / "european_format_text.json")
    result = agent.extract_text(case.raw_text, source=case.case_id)
    result.invoice == case.invoice  # the expected extraction

PDF input (text layer, then the live path)
------------------------------------------
``IntakeAgent.extract_pdf(path)`` reads the PDF's text layer locally
(``ledgercheck.pdf_text``: no OCR, scans and encrypted files are refused) and
hands it to ``extract_text`` with ``source=<file name>``, never the directory.
``PdfIntake(directory, agent)`` is the same step in ``run_pipeline``'s shape,
for a file name in ``directory``; ``ledgercheck run-pdf`` uses it.

Recording on a run
------------------
``record_intake(store, result)`` appends the result as the run's intake step,
creating the run first (``source=result.source``) if ``result.run_id`` does not
exist yet. The step output is the whole ``ExtractionResult`` in JSON shape, so
its ``invoice`` becomes the run's ``extracted`` fields::

    agent = IntakeAgent()
    result = agent.extract("clean_baseline")
    record = record_intake(RunStore(), result)
    record.invoice() == result.invoice  # True
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Protocol

from ledgercheck.agents.llm_client import LLMClient
from ledgercheck.fixtures_loader import FIXTURES_DIR, FixtureCase, load_case
from ledgercheck.models import ExtractionResult, Invoice
from ledgercheck.pdf_text import read_pdf_text
from ledgercheck.run_store import RunNotFound, RunRecord, RunStore, Stage


class ExtractionClient(Protocol):
    name: str

    def extract_invoice(self, text: str) -> Mapping[str, Any]: ...


class Intake(Protocol):
    """What ``run_pipeline`` needs from its intake stage."""

    def extract(self, case: Any, *, run_id: str | None = None) -> ExtractionResult: ...


def _is_bare_stem(case_id: str) -> bool:
    # A case id is a file stem; keep it from naming anything but a file in the root.
    if case_id in ("", ".", ".."):
        return False
    if any(sep and sep in case_id for sep in ("/", os.sep, os.altsep)):
        return False
    return Path(case_id).name == case_id


def _entry_names(root: Path) -> set[str]:
    # Exact-case names: on a case-insensitive filesystem ``is_file`` alone would
    # let "CLEAN_BASELINE" resolve to clean_baseline.json.
    return set(os.listdir(root)) if root.is_dir() else set()


def _new_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"


class IntakeAgent:
    """Produces ``ExtractionResult`` objects for the intake stage.

    ``fixtures_root`` is where case ids are looked up (the packaged fixtures by
    default). ``llm_client`` is used by ``extract_text`` (and so ``extract_pdf``)
    only; leave it ``None`` to go through the env-flag gate.
    """

    def __init__(
        self,
        *,
        fixtures_root: Path = FIXTURES_DIR,
        llm_client: ExtractionClient | None = None,
    ) -> None:
        self.fixtures_root = Path(fixtures_root)
        self._llm_client = llm_client

    def extract(self, case: FixtureCase | str, *, run_id: str | None = None) -> ExtractionResult:
        """Extract a fixture case; ``run_id`` defaults to a fresh unique id.

        A case id must be a bare file stem: an empty id, ``"."``, ``".."``, or
        an id containing a path separator (``"./nope"``, ``"a/b"``,
        ``"/abs"``) raises ``ValueError``. A well-formed id with no file named
        exactly ``<case_id>.json`` under ``fixtures_root`` raises
        ``FileNotFoundError``; the match is case-sensitive on every
        filesystem, so ``"CLEAN_BASELINE"`` does not find
        ``clean_baseline.json``. A malformed case raises ``FixtureError``.
        """
        if isinstance(case, str):
            if not _is_bare_stem(case):
                raise ValueError(f"not a fixture case id: {case!r}")
            path = self.fixtures_root / f"{case}.json"
            if not (path.name in _entry_names(self.fixtures_root) and path.is_file()):
                raise FileNotFoundError(f"no fixture case {case!r} in {self.fixtures_root}")
            case = load_case(path)
        return ExtractionResult(
            run_id=_new_run_id() if run_id is None else run_id,
            invoice=case.invoice,
            source=case.case_id,
            extractor="fixture",
        )

    def extract_text(
        self, text: str, *, source: str, run_id: str | None = None
    ) -> ExtractionResult:
        """Extract ``text`` with the LLM client (live path, off by default).

        Raises ``LiveLLMDisabled`` when no client was injected and live
        extraction is not enabled. Fields the client returns must pass
        ``Invoice.from_dict``; a live reply with no JSON object raises
        ``ValueError``.
        """
        client = self._llm_client if self._llm_client is not None else LLMClient()
        invoice = Invoice.from_dict(client.extract_invoice(text))
        return ExtractionResult(
            run_id=_new_run_id() if run_id is None else run_id,
            invoice=invoice,
            source=source,
            extractor=client.name,
        )

    def extract_pdf(
        self, path: str | os.PathLike[str], *, run_id: str | None = None
    ) -> ExtractionResult:
        """Extract the text layer of PDF ``path`` with ``extract_text``; ``source`` is its name.

        Raises ``PdfTextError`` for a file without a readable text layer,
        before the gate is reached; then as ``extract_text``.
        """
        pdf = Path(path)
        return self.extract_text(read_pdf_text(pdf), source=pdf.name, run_id=run_id)


class PdfIntake:
    """``run_pipeline``'s intake for PDF files in ``directory``, by file name."""

    def __init__(self, directory: str | os.PathLike[str], agent: IntakeAgent) -> None:
        self.directory = Path(directory)
        self.agent = agent

    def extract(self, case: str, *, run_id: str | None = None) -> ExtractionResult:
        """``agent.extract_pdf`` on the file named ``case`` in ``directory``."""
        return self.agent.extract_pdf(self.directory / case, run_id=run_id)


def record_intake(store: RunStore, result: ExtractionResult) -> RunRecord:
    """Append ``result`` as the intake step of run ``result.run_id``.

    Starts the run if it does not exist. An existing run must be awaiting
    intake, otherwise ``RunStore.append_step`` raises ``ValueError``.
    """
    try:
        store.get_run(result.run_id)
    except RunNotFound:
        store.start_run(result.source, run_id=result.run_id)
    return store.append_step(result.run_id, Stage.INTAKE, result)
