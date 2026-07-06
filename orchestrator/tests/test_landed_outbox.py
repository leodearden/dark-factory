"""Tests for LandedOutbox, LandedRow, and MergeProvenance (task 2153 / W1 α).

Covers:
  step-1:  record() + re-open (simulated restart) + lookup() round-trip;
           lookup() of an unknown task_id returns None; LandedRow is frozen.
  step-3:  all() returns every landed row; record() is last-write-wins by
           task_id (WA-2); the overwrite persists across a re-open.
  step-5:  consume() idempotently prunes a row; a pruned row does not
           resurrect after a re-open (durable prune).
  step-7:  record() fsyncs both the row file and its parent directory before
           returning (WA-1) — executable proxy for crash-durability.
  step-9:  MergeProvenance.lookup() is a fail-safe façade over a bound
           LandedOutbox; returns None when unbound.
  step-11: SpeculativeMergeWorker constructs and holds a LandedOutbox
           (None-safe when git_ops has no project_root) and binds it to
           MergeProvenance.

Amendment pass (review of task 2153):
  Suggestion 1: a schema-drifted on-disk row is dropped at load time (with
                a WARNING) instead of raising out of lookup()/all().
  Suggestion 3: executable coverage for the fail-open invariants asserted
                in the module docstring — a corrupt file yields an empty
                store, and a failed durable write does not raise.
  Suggestion 4: a failed save does not leave an orphaned '.json.tmp'.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import stat
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orchestrator.git_ops import GitOps
from orchestrator.landed_outbox import LandedOutbox, LandedRow
from orchestrator.merge_queue import SpeculativeMergeWorker

# ---------------------------------------------------------------------------
# step-1 — record() + lookup() round-trip across a simulated restart
# ---------------------------------------------------------------------------


class TestLandedOutboxRecordLookupRoundtrip:
    """record() persists a row that survives a simulated restart (re-open)."""

    def test_record_survives_reopen_and_lookup(self, tmp_path: Path) -> None:
        path = tmp_path / 'data' / 'orchestrator' / 'landed_outbox.json'
        outbox = LandedOutbox(path)
        row = LandedRow(task_id='42', branch_tip_sha='aaa', advanced_sha='bbb', landed_at=123.0)

        outbox.record(row)

        # Simulated restart: a SECOND LandedOutbox instance at the same path
        # must observe the row purely from disk.
        reopened = LandedOutbox(path)
        looked_up = reopened.lookup('42')
        assert looked_up == row, f'Expected {row!r} to round-trip; got {looked_up!r}'

    def test_lookup_unknown_task_id_returns_none(self, tmp_path: Path) -> None:
        outbox = LandedOutbox(tmp_path / 'landed_outbox.json')
        assert outbox.lookup('nope') is None

    def test_landed_row_is_frozen(self) -> None:
        row = LandedRow(task_id='1', branch_tip_sha='a', advanced_sha='b', landed_at=1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            row.task_id = '2'  # type: ignore[misc]


# ---------------------------------------------------------------------------
# step-3 — all() + WA-2 last-write-wins
# ---------------------------------------------------------------------------


class TestLandedOutboxAllAndLastWriteWins:
    """all() lists every landed row; re-recording a task_id overwrites in place."""

    def test_fresh_store_all_is_empty(self, tmp_path: Path) -> None:
        outbox = LandedOutbox(tmp_path / 'landed_outbox.json')
        assert outbox.all() == []

    def test_all_returns_distinct_rows(self, tmp_path: Path) -> None:
        outbox = LandedOutbox(tmp_path / 'landed_outbox.json')
        row_a = LandedRow(task_id='a', branch_tip_sha='sha-a', advanced_sha='adv-a', landed_at=1.0)
        row_b = LandedRow(task_id='b', branch_tip_sha='sha-b', advanced_sha='adv-b', landed_at=2.0)

        outbox.record(row_a)
        outbox.record(row_b)

        all_rows = outbox.all()
        assert len(all_rows) == 2
        assert {r.task_id for r in all_rows} == {'a', 'b'}
        assert row_a in all_rows
        assert row_b in all_rows

    def test_record_overwrites_by_task_id(self, tmp_path: Path) -> None:
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)
        first = LandedRow(task_id='x', branch_tip_sha='sha-1', advanced_sha='adv-1', landed_at=1.0)
        second = LandedRow(task_id='x', branch_tip_sha='sha-1', advanced_sha='adv-2', landed_at=2.0)

        outbox.record(first)
        outbox.record(second)

        all_rows = outbox.all()
        assert len(all_rows) == 1, f'Expected exactly one entry for task_id="x"; got {all_rows!r}'
        assert outbox.lookup('x') == second

        # The overwrite must persist across a re-open, not just in the cache.
        reopened = LandedOutbox(path)
        assert reopened.lookup('x') == second


# ---------------------------------------------------------------------------
# step-5 — consume() idempotent prune
# ---------------------------------------------------------------------------


class TestLandedOutboxConsume:
    """consume() removes a row; repeated/unknown consume is a no-op; prune is durable."""

    def test_consume_removes_row(self, tmp_path: Path) -> None:
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)
        row = LandedRow(task_id='7', branch_tip_sha='sha', advanced_sha='adv', landed_at=1.0)
        outbox.record(row)

        outbox.consume('7')

        assert outbox.lookup('7') is None
        assert outbox.all() == []

    def test_consume_is_idempotent(self, tmp_path: Path) -> None:
        outbox = LandedOutbox(tmp_path / 'landed_outbox.json')
        row = LandedRow(task_id='7', branch_tip_sha='sha', advanced_sha='adv', landed_at=1.0)
        outbox.record(row)

        outbox.consume('7')
        outbox.consume('7')  # second call: silent no-op, must not raise

        assert outbox.lookup('7') is None

    def test_consume_unknown_task_id_is_noop(self, tmp_path: Path) -> None:
        outbox = LandedOutbox(tmp_path / 'landed_outbox.json')
        outbox.consume('does-not-exist')  # must not raise
        assert outbox.all() == []

    def test_consumed_row_does_not_resurrect_after_reopen(self, tmp_path: Path) -> None:
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)
        row = LandedRow(task_id='9', branch_tip_sha='sha', advanced_sha='adv', landed_at=1.0)
        outbox.record(row)
        outbox.consume('9')

        reopened = LandedOutbox(path)
        assert reopened.lookup('9') is None
        assert reopened.all() == []


# ---------------------------------------------------------------------------
# step-7 — WA-1 fsync durability (executable proxy for crash-durability)
# ---------------------------------------------------------------------------


class TestLandedOutboxFsyncDurability:
    """record() fsyncs both the row's file and its parent directory (WA-1).

    A unit test cannot trigger real power-loss, so this spies on every
    os.fsync call record() makes and asserts it touched (at least) one
    regular file and one directory — the executable proxy for "survives a
    simulated crash".
    """

    def test_record_fsyncs_file_and_parent_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import os as os_module

        real_fsync = os_module.fsync
        captured_modes: list[int] = []

        def _spy_fsync(fd: int) -> None:
            # Capture the fd's mode AT CALL TIME — before the impl closes it —
            # then delegate to the real fsync so durability behaviour is
            # otherwise unchanged.
            captured_modes.append(os_module.fstat(fd).st_mode)
            real_fsync(fd)

        monkeypatch.setattr('orchestrator.landed_outbox.os.fsync', _spy_fsync)

        path = tmp_path / 'data' / 'orchestrator' / 'landed_outbox.json'
        outbox = LandedOutbox(path)
        row = LandedRow(task_id='1', branch_tip_sha='a', advanced_sha='b', landed_at=1.0)

        outbox.record(row)

        assert len(captured_modes) >= 2, (
            f'Expected os.fsync called at least twice (file + dir); '
            f'got {len(captured_modes)} call(s)'
        )
        assert any(stat.S_ISREG(m) for m in captured_modes), (
            f'Expected at least one fsync on a regular file; modes={captured_modes!r}'
        )
        assert any(stat.S_ISDIR(m) for m in captured_modes), (
            f'Expected at least one fsync on a directory; modes={captured_modes!r}'
        )


# ---------------------------------------------------------------------------
# step-9 — MergeProvenance façade
# ---------------------------------------------------------------------------


@pytest.fixture
def _restore_merge_provenance_binding():
    """Snapshot/restore MergeProvenance's process-global bound outbox.

    MergeProvenance.bind() mutates a class-level reference shared across the
    whole test process; without this fixture a bind() in one test would leak
    into whichever test runs next in the same xdist worker. Imported locally
    (not at module scope) so this fixture itself does not exist — and thus
    cannot break collection — until step-10 defines MergeProvenance.
    """
    from orchestrator.landed_outbox import MergeProvenance

    original = MergeProvenance._outbox
    yield
    MergeProvenance._outbox = original


class TestMergeProvenanceFacade:
    """MergeProvenance.lookup() is a fail-safe façade over a bound LandedOutbox."""

    def test_lookup_with_nothing_bound_returns_none(
        self, _restore_merge_provenance_binding: None,
    ) -> None:
        from orchestrator.landed_outbox import MergeProvenance

        MergeProvenance._outbox = None  # explicit: nothing bound
        assert MergeProvenance.lookup('42') is None

    def test_lookup_routes_to_bound_outbox(
        self, tmp_path: Path, _restore_merge_provenance_binding: None,
    ) -> None:
        from orchestrator.landed_outbox import MergeProvenance

        outbox = LandedOutbox(tmp_path / 'landed_outbox.json')
        row = LandedRow(task_id='42', branch_tip_sha='a', advanced_sha='b', landed_at=1.0)
        outbox.record(row)

        MergeProvenance.bind(outbox)

        assert MergeProvenance.lookup('42') == row
        assert MergeProvenance.lookup('unknown') is None


# ---------------------------------------------------------------------------
# step-11 — SpeculativeMergeWorker holds a LandedOutbox instance
# ---------------------------------------------------------------------------


class TestSpeculativeMergeWorkerHoldsLandedOutbox:
    """SpeculativeMergeWorker.__init__ constructs + binds a LandedOutbox
    (None-safe when git_ops has no project_root, mirroring _shadow_state_path)."""

    def test_worker_holds_and_binds_landed_outbox(
        self, tmp_path: Path, _restore_merge_provenance_binding: None,
    ) -> None:
        from orchestrator.landed_outbox import MergeProvenance

        git_ops = MagicMock(spec=GitOps)
        git_ops.project_root = tmp_path

        worker = SpeculativeMergeWorker(git_ops=git_ops, queue=asyncio.Queue())

        assert isinstance(worker._landed_outbox, LandedOutbox)

        row = LandedRow(task_id='1', branch_tip_sha='a', advanced_sha='b', landed_at=1.0)
        worker._landed_outbox.record(row)

        assert MergeProvenance.lookup('1') == row

    def test_worker_landed_outbox_none_when_no_project_root(
        self, _restore_merge_provenance_binding: None,
    ) -> None:
        # Intentionally bare: project_root is NOT set on this mock.
        git_ops = MagicMock(spec=GitOps)
        worker = SpeculativeMergeWorker(git_ops=git_ops, queue=asyncio.Queue())

        assert worker._landed_outbox is None


# ---------------------------------------------------------------------------
# amend: Suggestion 1 — schema-drifted on-disk rows are dropped, not raised
# ---------------------------------------------------------------------------


class TestLandedOutboxSchemaDrift:
    """A row that is valid JSON but schema-drifted (relative to LandedRow's
    fields) must not raise out of lookup()/all() — it is dropped at load
    time with a WARNING instead, so one bad row can't defeat the fail-open
    contract for readers like the scheduler's consult-before-dispatch gate.
    """

    def test_row_missing_a_field_is_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / 'landed_outbox.json'
        path.write_text(
            json.dumps({
                'good': {'branch_tip_sha': 'a', 'advanced_sha': 'b', 'landed_at': 1.0},
                'bad-missing-field': {'branch_tip_sha': 'a', 'advanced_sha': 'b'},
            }),
            encoding='utf-8',
        )

        outbox = LandedOutbox(path)

        assert outbox.lookup('bad-missing-field') is None
        assert outbox.lookup('good') == LandedRow(
            task_id='good', branch_tip_sha='a', advanced_sha='b', landed_at=1.0,
        )
        assert {r.task_id for r in outbox.all()} == {'good'}

    def test_row_with_an_extra_field_is_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / 'landed_outbox.json'
        path.write_text(
            json.dumps({
                'bad-extra-field': {
                    'branch_tip_sha': 'a',
                    'advanced_sha': 'b',
                    'landed_at': 1.0,
                    'unexpected_key': 'surprise',
                },
            }),
            encoding='utf-8',
        )

        outbox = LandedOutbox(path)

        assert outbox.lookup('bad-extra-field') is None
        assert outbox.all() == []

    def test_non_dict_row_is_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / 'landed_outbox.json'
        path.write_text(json.dumps({'bad-non-dict': 'not-a-dict'}), encoding='utf-8')

        outbox = LandedOutbox(path)

        assert outbox.lookup('bad-non-dict') is None
        assert outbox.all() == []


# ---------------------------------------------------------------------------
# amend: Suggestion 3 — executable coverage for fail-open branches
# ---------------------------------------------------------------------------


class TestLandedOutboxFailOpenBranches:
    """Executable coverage for the fail-open invariants asserted in the
    module docstring but previously untested: a corrupt file yields a
    silent empty store, and a failed durable write does not raise (the row
    stays cache-only until a later successful flush)."""

    def test_corrupt_json_file_yields_empty_store(self, tmp_path: Path) -> None:
        path = tmp_path / 'landed_outbox.json'
        path.write_text('not json {{{{', encoding='utf-8')

        outbox = LandedOutbox(path)

        assert outbox.all() == []
        assert outbox.lookup('anything') is None

    def test_save_failure_does_not_raise_and_row_stays_in_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)
        row = LandedRow(task_id='1', branch_tip_sha='a', advanced_sha='b', landed_at=1.0)

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise OSError('disk full simulated failure')

        monkeypatch.setattr('orchestrator.landed_outbox.os.replace', _boom)

        outbox.record(row)  # must not raise, despite the durable flush failing

        # Visible from the in-memory cache even though the flush failed —
        # record() must never block the merge pipeline on a disk error.
        assert outbox.lookup('1') == row
        assert outbox.save_failures == 1
        # ...but the failed write must genuinely not have landed on disk.
        assert not path.exists()

    def test_save_failure_counter_increments_per_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / 'landed_outbox.json'
        outbox = LandedOutbox(path)

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise OSError('disk full simulated failure')

        monkeypatch.setattr('orchestrator.landed_outbox.os.replace', _boom)

        outbox.record(LandedRow(task_id='1', branch_tip_sha='a', advanced_sha='b', landed_at=1.0))
        outbox.record(LandedRow(task_id='2', branch_tip_sha='c', advanced_sha='d', landed_at=2.0))

        assert outbox.save_failures == 2


# ---------------------------------------------------------------------------
# amend: Suggestion 4 — a failed save must not orphan a '.json.tmp' file
# ---------------------------------------------------------------------------


class TestLandedOutboxTmpFileCleanup:
    """A failed _save_raw must not leave a stale '.json.tmp' sibling behind."""

    def test_failed_save_cleans_up_tmp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / 'landed_outbox.json'
        tmp_sibling = path.with_suffix('.json.tmp')
        outbox = LandedOutbox(path)

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise OSError('disk full simulated failure')

        monkeypatch.setattr('orchestrator.landed_outbox.os.replace', _boom)

        outbox.record(LandedRow(task_id='1', branch_tip_sha='a', advanced_sha='b', landed_at=1.0))

        assert not tmp_sibling.exists(), (
            f'Expected no orphaned tmp file after a failed save; found {tmp_sibling}'
        )
