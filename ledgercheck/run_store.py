"""File-backed state for one pipeline run, with human corrections.

A run moves through three stages in order: ``intake`` → ``policy`` →
``approval``. Each finished stage is appended to the run as a ``StepRecord``.
Steps are append-only: nothing in the store edits or drops a recorded step.

Human correction
----------------
``RunStore.apply_correction`` changes one extracted invoice field, records
who changed it and why, and marks a stage to resume from (``policy`` by
default). The steps already recorded stay on disk unchanged. Steps at or after
the resume stage that were recorded before the correction now count as
*stale*: ``RunRecord.active_steps`` leaves them out, and ``is_stale`` reports
them. The pipeline then appends fresh steps from ``next_stage`` onward without
running intake again.

Storage
-------
One JSON file per run, ``<root>/<run_id>.json``. The default root is
``.scratch/runs/`` under the current directory; ``.scratch/`` is gitignored.
Each save writes a temporary file and ``os.replace``-s it over the old one, so
an interrupted process leaves either the previous or the new complete file.
``start_run`` also writes a temporary file first and hard-links it into place,
so an interrupted create leaves no partial run file and the same id can be
retried (a killed process may leave a stray ``.tmp`` file, which the store
ignores). These guarantees cover process interruption only: nothing is
fsynced, so this is not power-loss or OS-crash durable. The store is meant for
a local, single-writer demo; it does no locking, and concurrent writers can
lose updates. ``get_run`` checks that the file is a JSON object of the expected
shape, that the stored ``run_id`` matches the file name, that enum fields are
valid, and that ``extracted`` is a valid invoice. It does not check every
workflow invariant, so a hand-edited file can load while being inconsistent.

Returned records have frozen attributes, but the JSON mappings inside them
(``extracted``, step ``output``, correction values) are ordinary mutable dicts.
Treat a returned record as a read-only snapshot: every store method reloads
from disk, so edits to one are never persisted.

Step outputs are stored as JSON (see ``models.to_jsonable``). The intake
output must hold an ``invoice`` in ``Invoice.from_dict`` shape (an
``ExtractionResult`` does); that invoice becomes the run's ``extracted``
fields. The approval output must hold an ``outcome`` (an ``ApprovalDecision``
does): ``needs_human`` leaves the run in status ``needs_human``, any other
outcome completes it.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import re
import stat
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Mapping

from ledgercheck.models import Invoice, Outcome, to_jsonable

DEFAULT_ROOT = Path(".scratch") / "runs"

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class Stage(StrEnum):
    INTAKE = "intake"
    POLICY = "policy"
    APPROVAL = "approval"


STAGE_ORDER: tuple[Stage, ...] = (Stage.INTAKE, Stage.POLICY, Stage.APPROVAL)


class RunStatus(StrEnum):
    RUNNING = "running"  # more stages to run (including after a correction)
    NEEDS_HUMAN = "needs_human"  # approval asked for a person to correct or confirm
    COMPLETED = "completed"


class RunNotFound(KeyError):
    """No run with that id exists under the store root."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _check_run_id(run_id: Any) -> str:
    # The id becomes a file name, so no separators and no leading dot.
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError(
            f"run_id must be 1-128 chars of letters, digits, '_', '.', '-' "
            f"and start with a letter or digit, got {run_id!r}"
        )
    return run_id


def _json_object(value: Any, what: str) -> dict[str, Any]:
    """Convert a model or mapping to a JSON-safe dict, or raise TypeError."""
    is_model = dataclasses.is_dataclass(value) and not isinstance(value, type)
    if not (is_model or isinstance(value, Mapping)):
        raise TypeError(f"{what}: expected a mapping or model, got {type(value).__name__}")
    data = to_jsonable(value)
    json.dumps(data)  # fail now, not halfway through a save
    return data


def _dumps(record: RunRecord) -> str:
    return json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One finished stage. ``seq`` is the step's index in ``RunRecord.steps``."""

    seq: int
    stage: Stage
    recorded_at: str  # ISO 8601, UTC
    output: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", Stage(self.stage))


@dataclass(frozen=True, slots=True)
class Correction:
    """A human change to one extracted field.

    ``after_step`` is how many steps existed when the correction was made;
    steps with a lower ``seq`` at or after ``resume_from`` are stale.
    """

    field: str
    old_value: Any
    new_value: Any
    corrected_by: str
    reason: str
    corrected_at: str  # ISO 8601, UTC
    resume_from: Stage
    after_step: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "resume_from", Stage(self.resume_from))


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Everything persisted about one run (attributes frozen, nested JSON not).

    ``extracted`` holds the current invoice fields in JSON shape: the intake
    output with every correction applied. ``resume_from`` is set by a
    correction and cleared when that stage is appended again.
    """

    run_id: str
    source: str
    status: RunStatus
    created_at: str
    updated_at: str
    steps: tuple[StepRecord, ...] = ()
    corrections: tuple[Correction, ...] = ()
    extracted: Mapping[str, Any] | None = None
    resume_from: Stage | None = None

    def __post_init__(self) -> None:
        _check_run_id(self.run_id)
        object.__setattr__(self, "status", RunStatus(self.status))
        if self.resume_from is not None:
            object.__setattr__(self, "resume_from", Stage(self.resume_from))
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "corrections", tuple(self.corrections))
        for i, step in enumerate(self.steps):
            if step.seq != i:
                raise ValueError(f"run {self.run_id}: step {i} has seq {step.seq}")
        # Intake is always the first step and sets extracted, so any recorded
        # step means there must be an invoice to correct and resume from.
        if self.steps and self.extracted is None:
            raise ValueError(f"run {self.run_id}: has steps but no extracted invoice")
        if self.extracted is not None:
            # Check on load, so a corrupt record fails in get_run, not later.
            try:
                Invoice.from_dict(self.extracted)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"run {self.run_id}: extracted is not a valid invoice: {exc}") from exc

    @property
    def next_stage(self) -> Stage | None:
        """The stage the pipeline should run next, or ``None`` when done."""
        if self.resume_from is not None:
            return self.resume_from
        if not self.steps:
            return Stage.INTAKE
        i = STAGE_ORDER.index(self.steps[-1].stage)
        return STAGE_ORDER[i + 1] if i + 1 < len(STAGE_ORDER) else None

    def is_stale(self, step: StepRecord) -> bool:
        """True when a later correction invalidated this step's inputs."""
        rank = STAGE_ORDER.index(step.stage)
        return any(
            step.seq < c.after_step and rank >= STAGE_ORDER.index(c.resume_from)
            for c in self.corrections
        )

    def active_steps(self) -> tuple[StepRecord, ...]:
        """Steps still valid after corrections, in the order they were recorded."""
        return tuple(s for s in self.steps if not self.is_stale(s))

    def invoice(self) -> Invoice | None:
        """The extracted invoice with corrections applied, or ``None`` before intake."""
        return None if self.extracted is None else Invoice.from_dict(self.extracted)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunRecord:
        """Rebuild from ``to_dict`` output; raises ValueError/TypeError on bad shape."""
        data = dict(data)
        data["steps"] = tuple(StepRecord(**s) for s in data.get("steps", ()))
        data["corrections"] = tuple(Correction(**c) for c in data.get("corrections", ()))
        return cls(**data)


class RunStore:
    """File-backed store of ``RunRecord`` objects under ``root``.

    ``clock`` returns the current time (timezone-aware); tests pass a fixed
    one. Every mutating method saves before returning and returns the new
    record, a read-only-by-convention snapshot (see the module docstring).
    """

    def __init__(
        self,
        root: str | os.PathLike[str] = DEFAULT_ROOT,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.root = Path(root)
        self._clock = clock

    def _now(self) -> str:
        return self._clock().astimezone(timezone.utc).isoformat()

    def _path(self, run_id: str) -> Path:
        return self.root / f"{_check_run_id(run_id)}.json"

    def _root_exists(self) -> bool:
        """False if the root is absent; raise ``NotADirectoryError`` if it is unusable.

        A regular file or a dangling symlink is not a store, and must not be
        mistaken for an empty one. Other probe errors propagate.
        """
        try:
            os.lstat(self.root)  # unlike exists(), only FileNotFoundError means absent
            absent = False
        except FileNotFoundError:
            absent = True
        if absent:
            return False
        try:
            is_dir = stat.S_ISDIR(os.stat(self.root).st_mode)  # follows symlinks
        except FileNotFoundError:  # lstat saw it, so this is a dangling symlink
            is_dir = False
        if not is_dir:
            raise NotADirectoryError(f"run store root {self.root} is not a directory")
        return True

    def _save(self, record: RunRecord) -> RunRecord:
        path = self._path(record.run_id)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(_dumps(record), encoding="utf-8")
        os.replace(tmp, path)
        return record

    def start_run(self, source: str, *, run_id: str | None = None) -> RunRecord:
        """Create a run awaiting intake; ``source`` names what will be read.

        Without ``run_id`` a unique id is generated. An existing id raises
        ``FileExistsError``. A root that exists but is not a directory (a regular
        file or a dangling symlink) raises ``NotADirectoryError``.
        """
        if not isinstance(source, str) or not source:
            raise ValueError("source must be a non-empty string")
        now = self._now()
        if run_id is None:
            stamp = self._clock().astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"run-{stamp}-{uuid.uuid4().hex[:8]}"
        record = RunRecord(
            run_id=run_id, source=source, status=RunStatus.RUNNING, created_at=now, updated_at=now
        )
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except FileExistsError:  # exist_ok only lets a directory through
            raise NotADirectoryError(f"run store root {self.root} is not a directory") from None
        # Write the whole record to a temp file, then hard-link it into place:
        # the link claims the id atomically (FileExistsError if taken), and
        # the run file never exists half-written. The temp file is removed on
        # exit, so a failed create leaves nothing behind and a retry succeeds.
        path = self._path(run_id)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.root, prefix=f".{path.name}.", suffix=".tmp"
        ) as fh:
            fh.write(_dumps(record))
            fh.flush()
            os.link(fh.name, path)
        return record

    def get_run(self, run_id: str) -> RunRecord:
        """Load a run; raises ``RunNotFound`` if absent, ``ValueError`` if corrupt.

        A root that is not a directory raises ``NotADirectoryError``.
        """
        path = self._path(run_id)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            # A dangling-symlink root also lands here; that is a broken store.
            self._root_exists()
            raise RunNotFound(run_id) from None
        try:
            record = RunRecord.from_dict(json.loads(text))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: not a valid run record: {exc}") from exc
        # _save writes under record.run_id, so a mismatch would overwrite another run.
        if record.run_id != run_id:
            raise ValueError(f"{path}: holds run {record.run_id!r}, not {run_id!r}")
        return record

    def list_runs(self, *, status: RunStatus | str | None = None) -> list[RunRecord]:
        """All runs, oldest first (by ``created_at``, then ``run_id``).

        ``status`` keeps only runs in that status. A missing root yields ``[]``;
        a root that exists but is not a directory (a regular file or a
        dangling symlink) raises ``NotADirectoryError``,
        and other errors reading it (e.g. permissions) propagate.
        """
        if not self._root_exists():
            return []
        # Path.glob suppresses PermissionError on 3.11, so list explicitly.
        names = [n for n in os.listdir(self.root) if n.endswith(".json")]
        wanted = None if status is None else RunStatus(status)
        records = [self.get_run(Path(n).stem) for n in names]
        records = [r for r in records if wanted is None or r.status is wanted]
        return sorted(records, key=lambda r: (r.created_at, r.run_id))

    def append_step(self, run_id: str, stage: Stage | str, output: Any) -> RunRecord:
        """Record ``stage`` as finished with ``output`` (a model or mapping).

        ``stage`` must equal the run's ``next_stage``; anything else raises
        ``ValueError``. An output carrying a ``run_id`` must match this run.
        Appending the stage named by ``resume_from`` clears it.
        """
        record = self.get_run(run_id)
        stage = Stage(stage)
        if stage is not record.next_stage:
            raise ValueError(
                f"run {run_id}: expected stage {record.next_stage}, got {stage}"
            )
        data = _json_object(output, f"{stage} output")
        if data.get("run_id", run_id) != run_id:
            raise ValueError(f"{stage} output is for run {data['run_id']!r}, not {run_id!r}")
        changes: dict[str, Any] = {"status": RunStatus.RUNNING, "resume_from": None}
        if stage is Stage.INTAKE:
            if not isinstance(data.get("invoice"), Mapping):
                raise ValueError("intake output must contain an 'invoice' mapping")
            Invoice.from_dict(data["invoice"])
            # own copy, so mutating rec.extracted cannot reach the step output
            changes["extracted"] = to_jsonable(data["invoice"])
        elif stage is Stage.APPROVAL:
            outcome = Outcome(data.get("outcome"))
            changes["status"] = (
                RunStatus.NEEDS_HUMAN if outcome is Outcome.NEEDS_HUMAN else RunStatus.COMPLETED
            )
        now = self._now()
        step = StepRecord(seq=len(record.steps), stage=stage, recorded_at=now, output=data)
        return self._save(
            dataclasses.replace(record, steps=record.steps + (step,), updated_at=now, **changes)
        )

    def apply_correction(
        self,
        run_id: str,
        field: str,
        value: Any,
        *,
        corrected_by: str,
        reason: str = "",
        resume_from: Stage | str = Stage.POLICY,
    ) -> RunRecord:
        """Set extracted invoice ``field`` to ``value`` and mark the run to resume.

        ``field`` is a top-level ``Invoice`` field; ``value`` may be a model
        value (``Decimal``, ``date``) or its JSON form. The corrected invoice
        must still pass ``Invoice.from_dict``, otherwise ``ValueError`` and
        nothing is saved. ``resume_from`` must be ``policy`` or ``approval``:
        re-running intake would overwrite the correction. A ``resume_from``
        later than the run's ``next_stage`` is moved back to it, so no stage
        is skipped. Recorded steps are kept as-is; the run's status goes back
        to ``running``. The correction history gets its own deep copies of the
        old and new values.
        """
        record = self.get_run(run_id)
        resume = Stage(resume_from)
        if resume is Stage.INTAKE:
            raise ValueError("resume_from must be policy or approval; intake would undo it")
        if record.extracted is None:
            raise ValueError(f"run {run_id}: nothing extracted yet, so nothing to correct")
        # Never skip a stage that has not run (or is already due to re-run).
        if record.next_stage is not None and STAGE_ORDER.index(resume) > STAGE_ORDER.index(
            record.next_stage
        ):
            resume = record.next_stage
        if not corrected_by:
            raise ValueError("corrected_by must name who made the correction")
        if field not in {f.name for f in dataclasses.fields(Invoice)}:
            raise ValueError(f"{field!r} is not an Invoice field")
        # Deep copies: nested values (line_items) must not be shared with the
        # previous record, the correction history, or the caller's ``value``.
        new_value = copy.deepcopy(to_jsonable(value))
        extracted = copy.deepcopy(dict(record.extracted))
        extracted[field] = copy.deepcopy(new_value)
        try:
            Invoice.from_dict(extracted)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"correction to {field!r} is invalid: {exc}") from exc
        now = self._now()
        correction = Correction(
            field=field,
            old_value=copy.deepcopy(record.extracted.get(field)),
            new_value=new_value,
            corrected_by=corrected_by,
            reason=reason,
            corrected_at=now,
            resume_from=resume,
            after_step=len(record.steps),
        )
        return self._save(
            dataclasses.replace(
                record,
                extracted=extracted,
                corrections=record.corrections + (correction,),
                resume_from=resume,
                status=RunStatus.RUNNING,
                updated_at=now,
            )
        )
