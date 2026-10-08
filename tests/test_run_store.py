"""RunStore: create, persist, correct and resume runs under tmp_path."""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from ledgercheck.fixtures_loader import FIXTURES_DIR, load_case
from ledgercheck.models import ApprovalDecision, ExtractionResult, Outcome
from ledgercheck.run_store import (
    DEFAULT_ROOT,
    RunNotFound,
    RunStatus,
    RunStore,
    Stage,
)

RUN = "run-1"


class FakeClock:
    """Advances one second per call so timestamps are distinct and ordered."""

    def __init__(self) -> None:
        self.now = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "runs", clock=FakeClock())


def _extraction(run_id=RUN, case="clean_baseline"):
    invoice = load_case(FIXTURES_DIR / f"{case}.json").invoice
    return ExtractionResult(run_id=run_id, invoice=invoice, source=case)


def _approval(outcome, run_id=RUN):
    return ApprovalDecision(run_id=run_id, invoice_number="INV-NW-20417", outcome=outcome)


def _through_policy(store):
    store.start_run("clean_baseline", run_id=RUN)
    store.append_step(RUN, Stage.INTAKE, _extraction())
    return store.append_step(RUN, Stage.POLICY, {"hits": []})


def test_start_run_persists_a_running_run(store, tmp_path):
    record = store.start_run("clean_baseline", run_id=RUN)
    path = tmp_path / "runs" / f"{RUN}.json"
    assert json.loads(path.read_text())["status"] == "running"
    assert store.get_run(RUN) == record
    assert record.next_stage is Stage.INTAKE
    assert record.steps == () and record.extracted is None


def test_generated_run_ids_are_unique(store):
    a = store.start_run("x")
    b = store.start_run("x")
    assert a.run_id != b.run_id and a.run_id.startswith("run-20261006T")


def test_default_root_is_scratch_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    RunStore().start_run("x", run_id=RUN)
    assert DEFAULT_ROOT.as_posix() == ".scratch/runs"
    assert (tmp_path / ".scratch" / "runs" / f"{RUN}.json").is_file()


def test_duplicate_run_id_is_refused(store, tmp_path):
    store.start_run("x", run_id=RUN)
    path = tmp_path / "runs" / f"{RUN}.json"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        store.start_run("y", run_id=RUN)
    assert path.read_bytes() == before
    assert list((tmp_path / "runs").glob(f".{RUN}.json.*.tmp")) == []
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == [f"{RUN}.json"]


def test_run_file_copied_under_another_id_is_refused(store, tmp_path):
    _through_policy(store)
    root = tmp_path / "runs"
    original = (root / f"{RUN}.json").read_bytes()
    (root / "run-2.json").write_bytes(original)
    with pytest.raises(ValueError, match="'run-1'.*'run-2'"):
        store.get_run("run-2")
    with pytest.raises(ValueError, match="'run-1'.*'run-2'"):
        store.append_step("run-2", Stage.APPROVAL, _approval(Outcome.APPROVE, run_id="run-2"))
    with pytest.raises(ValueError, match="'run-1'.*'run-2'"):
        store.apply_correction("run-2", "vendor_name", "Other", corrected_by="reviewer")
    assert (root / f"{RUN}.json").read_bytes() == original
    assert (root / "run-2.json").read_bytes() == original


@pytest.mark.parametrize("bad", ["", "../evil", ".hidden", "a/b", "a b", 7])
def test_unsafe_run_ids_are_refused(store, bad):
    with pytest.raises(ValueError):
        store.start_run("x", run_id=bad)


def test_missing_run_raises_run_not_found(store):
    with pytest.raises(RunNotFound):
        store.get_run("nope")


def test_full_pipeline_survives_a_new_store_instance(store, tmp_path):
    _through_policy(store)
    done = store.append_step(RUN, Stage.APPROVAL, _approval(Outcome.APPROVE))
    assert done.status is RunStatus.COMPLETED and done.next_stage is None

    reloaded = RunStore(tmp_path / "runs").get_run(RUN)
    assert reloaded == done
    assert [s.stage for s in reloaded.steps] == [Stage.INTAKE, Stage.POLICY, Stage.APPROVAL]
    assert [s.seq for s in reloaded.steps] == [0, 1, 2]
    assert reloaded.invoice() == _extraction().invoice
    assert reloaded.steps[2].output["outcome"] == "approve"


def test_steps_must_run_in_order(store):
    store.start_run("x", run_id=RUN)
    with pytest.raises(ValueError, match="expected stage intake"):
        store.append_step(RUN, Stage.POLICY, {})
    store.append_step(RUN, Stage.INTAKE, _extraction())
    with pytest.raises(ValueError, match="expected stage policy"):
        store.append_step(RUN, Stage.INTAKE, _extraction())


def test_completed_run_takes_no_more_steps(store):
    _through_policy(store)
    store.append_step(RUN, Stage.APPROVAL, _approval(Outcome.FLAG))
    with pytest.raises(ValueError, match="expected stage None"):
        store.append_step(RUN, Stage.APPROVAL, _approval(Outcome.FLAG))


def test_intake_output_needs_a_valid_invoice(store):
    store.start_run("x", run_id=RUN)
    with pytest.raises(ValueError, match="'invoice' mapping"):
        store.append_step(RUN, Stage.INTAKE, {"fields": {}})
    with pytest.raises(TypeError):
        store.append_step(RUN, Stage.INTAKE, ["not", "a", "mapping"])
    assert store.get_run(RUN).steps == ()


def test_extracted_does_not_share_the_intake_step_output(store):
    store.start_run("x", run_id=RUN)
    record = store.append_step(RUN, Stage.INTAKE, _extraction())
    original = record.steps[0].output["invoice"]["po_number"]
    record.extracted["po_number"] = "PO-MUTATED"
    assert record.steps[0].output["invoice"]["po_number"] == original
    assert store.get_run(RUN).extracted["po_number"] == original


def test_step_output_for_another_run_is_refused(store):
    store.start_run("x", run_id=RUN)
    with pytest.raises(ValueError, match="not 'run-1'"):
        store.append_step(RUN, Stage.INTAKE, _extraction(run_id="run-2"))


def test_approval_needs_human_sets_status(store):
    _through_policy(store)
    record = store.append_step(RUN, Stage.APPROVAL, _approval(Outcome.NEEDS_HUMAN))
    assert record.status is RunStatus.NEEDS_HUMAN
    assert [r.run_id for r in store.list_runs(status="needs_human")] == [RUN]


def test_correction_changes_field_and_keeps_prior_steps(store):
    _through_policy(store)
    before = store.append_step(RUN, Stage.APPROVAL, _approval(Outcome.NEEDS_HUMAN))

    after = store.apply_correction(
        RUN, "po_number", "PO-9999", corrected_by="human:reviewer", reason="PO on cover email"
    )

    assert after.extracted["po_number"] == "PO-9999"
    assert after.invoice().po_number == "PO-9999"
    assert after.steps == before.steps  # nothing edited or dropped
    assert after.steps[0].output["invoice"]["po_number"] == "PO-4500-1182"
    assert after.status is RunStatus.RUNNING
    assert after.resume_from is Stage.POLICY and after.next_stage is Stage.POLICY
    (c,) = after.corrections
    assert (c.field, c.old_value, c.new_value) == ("po_number", "PO-4500-1182", "PO-9999")
    assert c.corrected_by == "human:reviewer" and c.after_step == 3
    assert [s.stage for s in after.active_steps()] == [Stage.INTAKE]
    assert store.get_run(RUN) == after


def test_resume_after_correction_appends_fresh_steps(store, tmp_path):
    _through_policy(store)
    store.append_step(RUN, Stage.APPROVAL, _approval(Outcome.NEEDS_HUMAN))
    store.apply_correction(RUN, "tax_amount", Decimal("83.53"), corrected_by="human:reviewer")

    # Resume from a fresh store, as a restarted process would.
    resumed = RunStore(tmp_path / "runs", clock=FakeClock())
    record = resumed.get_run(RUN)
    assert record.next_stage is Stage.POLICY
    resumed.append_step(RUN, record.next_stage, {"hits": []})
    final = resumed.append_step(RUN, Stage.APPROVAL, _approval(Outcome.APPROVE))

    assert final.status is RunStatus.COMPLETED and final.resume_from is None
    assert [s.stage for s in final.steps] == [
        Stage.INTAKE, Stage.POLICY, Stage.APPROVAL, Stage.POLICY, Stage.APPROVAL
    ]
    assert [s.seq for s in final.active_steps()] == [0, 3, 4]
    assert [final.is_stale(s) for s in final.steps] == [False, True, True, False, False]


def test_correction_resuming_at_approval_keeps_policy_active(store):
    _through_policy(store)
    store.append_step(RUN, Stage.APPROVAL, _approval(Outcome.NEEDS_HUMAN))
    record = store.apply_correction(
        RUN, "due_date", "2026-11-01", corrected_by="human:reviewer", resume_from="approval"
    )
    assert record.next_stage is Stage.APPROVAL
    assert [s.stage for s in record.active_steps()] == [Stage.INTAKE, Stage.POLICY]


def test_correction_never_skips_an_unrun_stage(store):
    store.start_run("x", run_id=RUN)
    store.append_step(RUN, Stage.INTAKE, _extraction())
    record = store.apply_correction(
        RUN, "po_number", None, corrected_by="human:reviewer", resume_from=Stage.APPROVAL
    )
    assert record.resume_from is Stage.POLICY


@pytest.mark.parametrize(
    "field, value, kwargs, match",
    [
        ("currency", "usd", {}, "invalid"),
        ("total", 1.5, {}, "invalid"),
        ("not_a_field", "x", {}, "not an Invoice field"),
        ("po_number", "PO-1", {"resume_from": "intake"}, "intake would undo it"),
        ("po_number", "PO-1", {"corrected_by": ""}, "corrected_by"),
    ],
)
def test_bad_corrections_are_refused_and_not_saved(store, field, value, kwargs, match):
    before = _through_policy(store)
    kwargs = {"corrected_by": "human:reviewer", **kwargs}
    with pytest.raises(ValueError, match=match):
        store.apply_correction(RUN, field, value, **kwargs)
    assert store.get_run(RUN) == before


def test_correction_before_intake_is_refused(store):
    store.start_run("x", run_id=RUN)
    with pytest.raises(ValueError, match="nothing extracted"):
        store.apply_correction(RUN, "po_number", "PO-1", corrected_by="human:reviewer")


def test_list_runs_is_oldest_first_and_filters_by_status(store, tmp_path):
    assert RunStore(tmp_path / "absent").list_runs() == []
    store.start_run("x", run_id="b-run")
    _through_policy(store)
    store.append_step(RUN, Stage.APPROVAL, _approval(Outcome.APPROVE))
    store.start_run("x", run_id="a-run")

    assert [r.run_id for r in store.list_runs()] == ["b-run", RUN, "a-run"]
    assert [r.run_id for r in store.list_runs(status=RunStatus.COMPLETED)] == [RUN]
    assert [r.run_id for r in store.list_runs(status="running")] == ["b-run", "a-run"]


def test_corrupt_run_file_raises_value_error(store, tmp_path):
    store.start_run("x", run_id=RUN)
    (tmp_path / "runs" / f"{RUN}.json").write_text('{"run_id": "run-1"}')
    with pytest.raises(ValueError, match="not a valid run record"):
        store.get_run(RUN)


def test_failed_create_leaves_no_run_and_retry_succeeds(store, tmp_path, monkeypatch):
    def fail_link(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr("ledgercheck.run_store.os.link", fail_link)
    with pytest.raises(OSError, match="disk full"):
        store.start_run("clean_baseline", run_id=RUN)
    assert list((tmp_path / "runs").iterdir()) == []  # temp file cleaned up too
    with pytest.raises(RunNotFound):
        store.get_run(RUN)

    monkeypatch.undo()
    record = store.start_run("clean_baseline", run_id=RUN)
    assert store.get_run(RUN) == record


def test_partial_temp_file_from_a_crashed_create_does_not_block_retry(store, tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / f".{RUN}.json.k3x9.tmp").write_text('{"run_id": "run-')  # killed mid-write
    record = store.start_run("clean_baseline", run_id=RUN)
    assert store.get_run(RUN) == record
    assert store.list_runs() == [record]


@pytest.mark.parametrize(
    "change",
    [
        lambda inv: inv.pop("total"),
        lambda inv: inv.update(invoice_date="not-a-date"),
        lambda inv: inv.update(line_items=[{"description": "x"}]),
        lambda inv: inv.update(subtotal="lots"),
    ],
    ids=["missing-key", "bad-date", "bad-line-item", "bad-money"],
)
def test_malformed_extracted_invoice_fails_on_load(store, tmp_path, change):
    store.start_run("clean_baseline", run_id=RUN)
    store.append_step(RUN, Stage.INTAKE, _extraction())
    path = tmp_path / "runs" / f"{RUN}.json"
    data = json.loads(path.read_text())
    change(data["extracted"])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="extracted is not a valid invoice"):
        store.get_run(RUN)


def test_non_mapping_extracted_fails_on_load(store, tmp_path):
    store.start_run("clean_baseline", run_id=RUN)
    path = tmp_path / "runs" / f"{RUN}.json"
    data = json.loads(path.read_text())
    data["extracted"] = ["not", "an", "invoice"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not a valid run record"):
        store.get_run(RUN)


def test_steps_without_extracted_fail_on_load(store, tmp_path):
    store.start_run("clean_baseline", run_id=RUN)
    store.append_step(RUN, Stage.INTAKE, _extraction())
    path = tmp_path / "runs" / f"{RUN}.json"
    data = json.loads(path.read_text())
    data["extracted"] = None
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="has steps but no extracted invoice"):
        store.get_run(RUN)


def test_list_runs_refuses_a_root_that_is_a_file(tmp_path):
    root = tmp_path / "runs"
    root.write_text("not a directory")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        RunStore(root).list_runs()


def test_list_runs_returns_empty_only_for_an_absent_root(tmp_path):
    assert RunStore(tmp_path / "absent").list_runs() == []


def test_list_runs_refuses_a_dangling_symlink_root(tmp_path):
    root = tmp_path / "runs"
    root.symlink_to(tmp_path / "gone")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        RunStore(root).list_runs()


def test_list_runs_reads_through_a_symlink_to_a_directory(tmp_path):
    real = tmp_path / "real"
    RunStore(real).start_run("clean_baseline", run_id=RUN)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    assert [r.run_id for r in RunStore(link).list_runs()] == [RUN]


def test_list_runs_propagates_a_directory_scan_failure(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "runs")
    store.start_run("clean_baseline", run_id=RUN)

    def denied(path):
        raise PermissionError(13, "denied", str(path))

    monkeypatch.setattr("os.listdir", denied)
    with pytest.raises(PermissionError):
        store.list_runs()


def test_list_runs_propagates_unexpected_root_probe_errors(tmp_path, monkeypatch):
    def denied(path, *args, **kwargs):
        raise PermissionError(13, "denied", str(path))

    monkeypatch.setattr("os.lstat", denied)
    with pytest.raises(PermissionError):
        RunStore(tmp_path / "runs").list_runs()


def test_list_runs_ignores_non_json_files_in_the_root(tmp_path):
    store = RunStore(tmp_path / "runs")
    store.start_run("clean_baseline", run_id=RUN)
    (store.root / "notes.txt").write_text("not a run")
    assert [r.run_id for r in store.list_runs()] == [RUN]


def test_start_run_refuses_a_root_that_is_a_file(tmp_path):
    root = tmp_path / "runs"
    root.write_text("not a directory")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        RunStore(root).start_run("clean_baseline", run_id=RUN)
    assert root.read_text() == "not a directory"


def test_start_run_refuses_a_dangling_symlink_root(tmp_path):
    root = tmp_path / "runs"
    root.symlink_to(tmp_path / "gone")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        RunStore(root).start_run("clean_baseline", run_id=RUN)
    assert not (tmp_path / "gone").exists()


def test_duplicate_id_still_raises_file_exists_error(store):
    store.start_run("clean_baseline", run_id=RUN)
    with pytest.raises(FileExistsError):
        store.start_run("clean_baseline", run_id=RUN)


def test_get_run_refuses_a_dangling_symlink_root(tmp_path):
    root = tmp_path / "runs"
    root.symlink_to(tmp_path / "gone")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        RunStore(root).get_run(RUN)


def test_get_run_refuses_a_root_that_is_a_file(tmp_path):
    root = tmp_path / "runs"
    root.write_text("not a directory")
    with pytest.raises(NotADirectoryError):
        RunStore(root).get_run(RUN)


def test_get_run_on_an_absent_root_or_missing_run_is_run_not_found(tmp_path):
    with pytest.raises(RunNotFound):
        RunStore(tmp_path / "absent").get_run(RUN)
    real = tmp_path / "real"
    RunStore(real).start_run("clean_baseline", run_id="other")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(RunNotFound):
        RunStore(link).get_run(RUN)


def test_correction_does_not_share_nested_values_between_records(store):
    before = _through_policy(store)
    new_items = [dict(li) for li in before.extracted["line_items"]]
    after = store.apply_correction(
        RUN, "line_items", new_items, corrected_by="human:reviewer", reason="fix items"
    )
    (c,) = after.corrections
    snapshot_before = json.dumps(before.extracted, sort_keys=True)
    snapshot_c_old = json.dumps(c.old_value, sort_keys=True)

    after.extracted["line_items"][0]["description"] = "MUTATED"
    assert c.new_value[0]["description"] != "MUTATED"  # not aliased to extracted
    assert json.dumps(before.extracted, sort_keys=True) == snapshot_before
    assert new_items[0]["description"] != "MUTATED"  # caller's value untouched

    c.new_value[0]["description"] = "HISTORY"
    assert after.extracted["line_items"][0]["description"] == "MUTATED"
    assert json.dumps(c.old_value, sort_keys=True) == snapshot_c_old


def test_untouched_nested_values_are_not_shared_with_previous_record(store):
    before = _through_policy(store)
    after = store.apply_correction(RUN, "po_number", "PO-9999", corrected_by="human:reviewer")
    snapshot = json.dumps(before.extracted, sort_keys=True)
    after.extracted["line_items"][0]["description"] = "MUTATED"
    assert json.dumps(before.extracted, sort_keys=True) == snapshot
    assert store.get_run(RUN).extracted["line_items"][0]["description"] != "MUTATED"


def test_correction_old_value_does_not_alias_previous_extracted(store):
    before = _through_policy(store)
    after = store.apply_correction(
        RUN, "line_items", [], corrected_by="human:reviewer", reason="clear"
    )
    (c,) = after.corrections
    c.old_value[0]["description"] = "MUTATED"
    assert before.extracted["line_items"][0]["description"] != "MUTATED"
