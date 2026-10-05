"""Contract tests for ``fused_memory.utils.target_store_preflight`` (task 4319).

Background — the measured failure this guard exists to prevent
--------------------------------------------------------------
Both substrates the ``fused-memory/scripts/`` mutators target SILENTLY
AUTO-CREATE themselves when pointed at a path that does not exist:

  * ``escalation/src/escalation/queue.py::EscalationQueue.__init__`` does
    ``mkdir(parents=True, exist_ok=True)``, after which ``get_pending()``
    returns ``[]`` — a manufactured empty queue indistinguishable from a
    genuinely quiet one;
  * ``fused_memory/backends/sqlite_task_backend.py::SqliteTaskBackend.get_tasks``
    auto-creates ``.taskmaster/tasks/tasks.db`` and returns ``{"tasks": []}``
    for ANY ``project_root``, never raising.

So the hazard is MIS-TARGETING, not denial: a script run from a task worktree
(where neither ``data/reconciliation/`` nor ``.taskmaster/`` exists) reports a
clean all-clear and exits 0. That is why this guard READS whether the target
already holds a record (task 5468: mere existence passed the empty residue a
previous mis-targeted run leaves behind) and is not a capability probe — in
the failing case the target directory is perfectly writable, so a probe would
pass exactly when the danger is present.

Hermetic by construction: every target lives under ``tmp_path``, and the
residue and populated shapes are produced by the REAL ``EscalationQueue`` and
``SqliteTaskBackend``, so the guard is pinned against what the auto-creates
actually leave behind. The only monkeypatching is ``chdir``, which pins how a
RELATIVE target resolves — the CWD is then the test's independent variable,
chosen rather than inherited from wherever pytest was invoked.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _fm_helpers import make_populated_task_store
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.utils.target_store_preflight import (
    TargetStoreMissing,
    assert_queue_dir_populated,
    assert_task_store_populated,
    task_store_path,
)


def _seed_one_escalation(queue_dir: Path) -> tuple[EscalationQueue, Escalation]:
    """Submit one pending record through the REAL queue, never hand-rolled JSON."""
    queue = EscalationQueue(queue_dir)
    escalation = Escalation(
        id=f'esc-1-{uuid.uuid4().hex[:8]}',
        task_id='1',
        agent_role='reconciler',
        severity='info',
        category='recon_integrity_issue',
        summary='Non-actionable integrity finding: test.',
        timestamp=datetime.now(UTC).isoformat(),
    )
    queue.submit(escalation)
    return queue, escalation


class TestQueueDirArm:
    """The queue arm refuses unless the directory HOLDS at least one entry.

    Existence alone cannot separate "wrong location" from "genuinely quiet":
    ``EscalationQueue.__init__`` mkdirs its target and ``get_pending()`` writes
    nothing, so the residue a mis-targeted run leaves is an EMPTY directory.
    Every shape here is built by the real queue, not approximated by hand.
    """

    def test_an_existing_empty_directory_is_refused_and_left_empty(self, tmp_path: Path):
        queue_dir = tmp_path / 'escalations'
        queue_dir.mkdir()

        with pytest.raises(TargetStoreMissing):
            assert_queue_dir_populated(queue_dir, operation='dismiss_recon_integrity_noise')

        assert queue_dir.is_dir()
        assert list(queue_dir.iterdir()) == []

    def test_the_residue_a_mis_targeted_run_leaves_is_refused(self, tmp_path: Path):
        """The measured shape: a real queue constructed and read on a fresh path."""
        queue_dir = tmp_path / 'data' / 'reconciliation' / 'escalations'
        assert EscalationQueue(queue_dir).get_pending() == []

        with pytest.raises(TargetStoreMissing):
            assert_queue_dir_populated(queue_dir, operation='backfill_recon_escalations')

    def test_a_queue_holding_a_pending_escalation_passes(self, tmp_path: Path):
        _seed_one_escalation(tmp_path)

        assert assert_queue_dir_populated(tmp_path, operation='op') is None

    def test_a_queue_whose_only_record_was_resolved_still_passes(self, tmp_path: Path):
        """A queue that has minted once is never empty again.

        Which entries survive the resolve is the queue's business, so it is
        deliberately not asserted; only that a genuinely quiet live queue is
        not mistaken for the residue of a mis-targeted run.
        """
        queue, escalation = _seed_one_escalation(tmp_path)
        queue.resolve(escalation.id, 'note', resolved_by='test')
        assert queue.get_pending() == []

        assert assert_queue_dir_populated(tmp_path, operation='op') is None

    def test_a_regular_file_at_the_queue_path_fails_open(self, tmp_path: Path):
        """An unreadable target is not refused: the queue's own mkdir fails loudly."""
        not_a_dir = tmp_path / 'escalations'
        not_a_dir.write_text('not a queue')

        assert assert_queue_dir_populated(not_a_dir, operation='op') is None

    def test_the_empty_refusal_names_operation_resolved_path_and_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        relative = Path('escalations')
        relative.mkdir()

        with pytest.raises(TargetStoreMissing) as excinfo:
            assert_queue_dir_populated(relative, operation='derive_orphaned_recon_escalations')

        message = str(excinfo.value)
        assert 'derive_orphaned_recon_escalations' in message
        assert str(tmp_path / relative) in message
        assert '--queue-dir' in message


def _zero_byte_task_store(project_root: Path) -> Path:
    db = task_store_path(project_root)
    db.parent.mkdir(parents=True)
    db.touch()
    return db


def _real_backend(project_root: Path) -> SqliteTaskBackend:
    return SqliteTaskBackend(TaskmasterConfig(project_root=str(project_root)))


class TestTaskStoreArm:
    """The task-store arm refuses unless tasks.db has a ``tasks`` table with a row.

    ``SqliteTaskBackend.get_tasks`` seeds the full schema into a fresh root and
    returns ``{"tasks": []}``, so a schema check alone would pass the residue a
    mis-targeted run leaves; only "holds a task" separates it from the live
    store. Unreadable targets fail OPEN: the backend's own open fails loudly.
    """

    def test_a_zero_byte_store_is_refused_and_left_untouched(self, tmp_path: Path):
        db = _zero_byte_task_store(tmp_path)

        with pytest.raises(TargetStoreMissing):
            assert_task_store_populated(tmp_path, operation='audit_duplicate_tasks')

        assert db.stat().st_size == 0
        assert list(db.parent.iterdir()) == [db]

    @pytest.mark.asyncio
    async def test_the_residue_a_mis_targeted_run_leaves_is_refused(self, tmp_path: Path):
        """The measured shape: a real backend that read a fresh root and held no task."""
        backend = _real_backend(tmp_path)
        await backend.start()
        try:
            assert await backend.get_tasks(project_root=str(tmp_path)) == {'tasks': []}
        finally:
            await backend.close()
        assert task_store_path(tmp_path).exists()

        with pytest.raises(TargetStoreMissing):
            assert_task_store_populated(tmp_path, operation='audit_duplicate_tasks')

    @pytest.mark.asyncio
    async def test_a_store_holding_a_real_task_passes(self, tmp_path: Path):
        backend = _real_backend(tmp_path)
        await backend.start()
        try:
            await backend.add_task(project_root=str(tmp_path), title='one')
        finally:
            await backend.close()

        assert assert_task_store_populated(tmp_path, operation='op') is None

    @pytest.mark.asyncio
    async def test_a_populated_store_the_backend_still_holds_open_passes(self, tmp_path: Path):
        """The live case: another process has the WAL-mode store open right now."""
        backend = _real_backend(tmp_path)
        await backend.start()
        try:
            await backend.add_task(project_root=str(tmp_path), title='one')

            assert assert_task_store_populated(tmp_path, operation='op') is None
        finally:
            await backend.close()

    def test_a_database_with_rows_but_no_tasks_table_is_refused(self, tmp_path: Path):
        """A different database at the task-store path is not the task store."""
        db = task_store_path(tmp_path)
        db.parent.mkdir(parents=True)
        connection = sqlite3.connect(db)
        try:
            connection.execute('CREATE TABLE notes (id INTEGER PRIMARY KEY)')
            connection.execute('INSERT INTO notes (id) VALUES (1)')
            connection.commit()
        finally:
            connection.close()

        with pytest.raises(TargetStoreMissing):
            assert_task_store_populated(tmp_path, operation='op')

    def test_non_sqlite_bytes_fail_open(self, tmp_path: Path):
        db = task_store_path(tmp_path)
        db.parent.mkdir(parents=True)
        db.write_bytes(b'this is not a sqlite database\n' * 64)

        assert assert_task_store_populated(tmp_path, operation='op') is None

    def test_a_directory_at_the_store_path_fails_open(self, tmp_path: Path):
        task_store_path(tmp_path).mkdir(parents=True)

        assert assert_task_store_populated(tmp_path, operation='op') is None

    def test_a_missing_store_is_refused_and_nothing_is_created(self, tmp_path: Path):
        with pytest.raises(TargetStoreMissing) as excinfo:
            assert_task_store_populated(tmp_path, operation='audit_duplicate_tasks')

        message = str(excinfo.value)
        assert 'audit_duplicate_tasks' in message
        assert str(task_store_path(tmp_path)) in message
        assert '--project-root' in message
        assert not (tmp_path / '.taskmaster').exists()

    def test_the_empty_refusal_names_operation_resolved_path_and_flag(self, tmp_path: Path):
        """An existing store is refused as EMPTY, never misreported as absent."""
        _zero_byte_task_store(tmp_path)

        with pytest.raises(TargetStoreMissing) as excinfo:
            assert_task_store_populated(tmp_path, operation='correct_found_on_main_backlog')

        message = str(excinfo.value)
        assert 'correct_found_on_main_backlog' in message
        assert str(task_store_path(tmp_path).resolve()) in message
        assert '--project-root' in message
        assert 'does not exist' not in message


class TestRefusesAMissingTarget:
    def test_missing_leaf_under_an_existing_parent_raises(self, tmp_path: Path):
        """The measured worktree case: ``data/`` absent, deep leaf requested.

        A worktree has no ``data/reconciliation/`` at all, and the script's
        default ``--queue-dir`` is the RELATIVE
        ``./data/reconciliation/escalations`` — so the whole subtree below an
        existing repo root is missing, not just the leaf.
        """
        (tmp_path / 'data').mkdir()
        target = tmp_path / 'data' / 'reconciliation' / 'escalations'

        with pytest.raises(TargetStoreMissing):
            assert_queue_dir_populated(target, operation='op')

    def test_refusal_does_not_create_the_path(self, tmp_path: Path):
        """The anti-litter property that distinguishes this from a probe.

        ``store_mutation_preflight`` creates and removes a scratch file inside
        live state; this guard writes nothing at all, in either direction.
        """
        target = tmp_path / 'data' / 'reconciliation' / 'escalations'

        with pytest.raises(TargetStoreMissing):
            assert_queue_dir_populated(target, operation='op')

        assert not target.exists()
        assert not (tmp_path / 'data').exists()


class TestRefusalMessage:
    """An operator must be able to tell WHICH run was refused, and WHY."""

    def test_message_names_operation_path_what_and_remedy_flag(self, tmp_path: Path):
        target = tmp_path / 'data' / 'reconciliation' / 'escalations'

        with pytest.raises(TargetStoreMissing) as excinfo:
            assert_queue_dir_populated(target, operation='dismiss_recon_integrity_noise')

        message = str(excinfo.value)
        assert 'dismiss_recon_integrity_noise' in message
        assert str(target) in message
        assert 'escalation queue directory' in message
        assert '--queue-dir' in message

    def test_message_reports_the_resolved_path_for_a_relative_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """A relative target is the measured trigger; report what it resolved TO.

        ``./data/reconciliation/escalations`` printed back verbatim tells an
        operator nothing about which directory was actually consulted.

        ``chdir(tmp_path)`` is what makes the assertion mean anything: the CWD
        a relative path resolves against is the variable under test, so the
        test must CHOOSE it rather than inherit wherever pytest was invoked.
        Reading ``Path.cwd()`` instead would both leave this file's only
        non-hermetic reach outside ``tmp_path`` and compare against a path
        nobody picked.
        """
        monkeypatch.chdir(tmp_path)
        relative = Path('data') / 'reconciliation' / 'escalations-4319-absent'

        with pytest.raises(TargetStoreMissing) as excinfo:
            assert_queue_dir_populated(relative, operation='op')

        assert str(tmp_path / relative) in str(excinfo.value)


class TestExceptionType:
    def test_is_a_runtime_error_subclass(self):
        """Catchable alongside ``StoreMutationUnavailable``, which is also one."""
        assert issubclass(TargetStoreMissing, RuntimeError)


class TestFamilyWrappers:
    """The two entry points the scripts actually call.

    ``assert_task_store_populated`` / ``assert_queue_dir_populated`` exist so
    that only ``operation`` varies between call sites: the target derivation
    and the family-constant ``what``/``remedy`` prose live in one place
    (task 4319 amendment pass). Pinned here: the shared test helper's store
    really passes the guard the four task-store suites rely on, and the two
    families keep their own remedy, since collapsing the call sites is only
    safe if nothing was lost with the copies.
    """

    def test_task_store_wrapper_passes_the_shared_populated_fixture(self, tmp_path: Path):
        make_populated_task_store(tmp_path)

        assert assert_task_store_populated(
            str(tmp_path), operation='audit_duplicate_tasks',
        ) is None

    def test_queue_dir_wrapper_passes_when_the_queue_holds_a_record(self, tmp_path: Path):
        queue_dir = tmp_path / 'escalations'
        _seed_one_escalation(queue_dir)

        assert assert_queue_dir_populated(
            queue_dir, operation='backfill_recon_escalations',
        ) is None

    def test_the_two_families_carry_different_remedies(self, tmp_path: Path):
        """A shared helper must not flatten the two into one generic sentence.

        The remedy is the line an operator acts on, and the two families need
        opposite actions: point ``--project-root`` at the main checkout, versus
        pass an absolute ``--queue-dir``.
        """
        with pytest.raises(TargetStoreMissing) as task_exc:
            assert_task_store_populated(tmp_path, operation='op')
        with pytest.raises(TargetStoreMissing) as queue_exc:
            assert_queue_dir_populated(tmp_path / 'absent', operation='op')

        assert '--queue-dir' not in str(task_exc.value)
        assert '--project-root' not in str(queue_exc.value)
