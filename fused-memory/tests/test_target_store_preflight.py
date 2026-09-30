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
clean all-clear and exits 0. That is why this guard is an EXISTENCE assertion
and not a capability probe — in the failing case the target directory is
perfectly writable, so a probe would pass exactly when the danger is present.

Hermetic by construction: every target lives under ``tmp_path``. Unlike
``store_mutation_preflight``'s guard, this one takes its target as a parameter
rather than resolving it from the environment, so the only monkeypatching here
is the one ``chdir`` that pins how a RELATIVE target resolves — the CWD is that
test's independent variable, so it is chosen rather than inherited from
wherever pytest was invoked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fused_memory.utils.target_store_preflight import (
    TargetStoreMissing,
    assert_queue_dir_exists,
    assert_target_store_exists,
    assert_task_store_exists,
    task_store_path,
)

_KWARGS = {
    'operation': 'dismiss_recon_integrity_noise',
    'what': 'the durable escalation queue directory',
    'remedy': 'pass an absolute --queue-dir, or run from the project root',
}


class TestPassesForAnExistingTarget:
    """The guard is a no-op whenever the target store is really there."""

    def test_existing_directory_passes(self, tmp_path: Path):
        queue_dir = tmp_path / 'escalations'
        queue_dir.mkdir()

        assert assert_target_store_exists(queue_dir, **_KWARGS) is None

    def test_existing_file_passes(self, tmp_path: Path):
        """One helper serves both substrates: tasks.db is a FILE, a queue is a DIR."""
        db = tmp_path / '.taskmaster' / 'tasks' / 'tasks.db'
        db.parent.mkdir(parents=True)
        db.touch()

        assert assert_target_store_exists(db, **_KWARGS) is None

    def test_accepts_a_string_path(self, tmp_path: Path):
        queue_dir = tmp_path / 'escalations'
        queue_dir.mkdir()

        assert assert_target_store_exists(str(queue_dir), **_KWARGS) is None


class TestRefusesAMissingTarget:
    def test_missing_path_raises(self, tmp_path: Path):
        with pytest.raises(TargetStoreMissing):
            assert_target_store_exists(tmp_path / 'escalations', **_KWARGS)

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
            assert_target_store_exists(target, **_KWARGS)

    def test_refusal_does_not_create_the_path(self, tmp_path: Path):
        """The anti-litter property that distinguishes this from a probe.

        ``store_mutation_preflight`` creates and removes a scratch file inside
        live state; this guard writes nothing at all, in either direction.
        """
        target = tmp_path / 'data' / 'reconciliation' / 'escalations'

        with pytest.raises(TargetStoreMissing):
            assert_target_store_exists(target, **_KWARGS)

        assert not target.exists()
        assert not (tmp_path / 'data').exists()


class TestRefusalMessage:
    """An operator must be able to tell WHICH run was refused, and WHY."""

    def test_message_names_operation_path_what_and_remedy(self, tmp_path: Path):
        target = tmp_path / 'data' / 'reconciliation' / 'escalations'

        with pytest.raises(TargetStoreMissing) as excinfo:
            assert_target_store_exists(
                target,
                operation='dismiss_recon_integrity_noise',
                what='the durable escalation queue directory',
                remedy='pass an absolute --queue-dir, or run from the project root',
            )

        message = str(excinfo.value)
        assert 'dismiss_recon_integrity_noise' in message
        assert str(target) in message
        assert 'the durable escalation queue directory' in message
        assert 'pass an absolute --queue-dir, or run from the project root' in message

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
            assert_target_store_exists(relative, **_KWARGS)

        assert str(tmp_path / relative) in str(excinfo.value)


class TestExceptionType:
    def test_is_a_runtime_error_subclass(self):
        """Catchable alongside ``StoreMutationUnavailable``, which is also one."""
        assert issubclass(TargetStoreMissing, RuntimeError)


class TestFamilyWrappers:
    """The two entry points the six scripts actually call.

    ``assert_task_store_exists`` / ``assert_queue_dir_exists`` exist so that
    only ``operation`` varies between call sites: the target derivation and the
    family-constant ``what``/``remedy`` prose live in one place instead of six
    (task 4319 amendment pass).  What is pinned here is that they are genuinely
    equivalent to the general helper -- same refusal type, same anti-litter
    property, same resolved path in the message -- and that the shared remedy
    text really does reach the operator, since collapsing the call sites is
    only safe if nothing was lost with the copies.
    """

    def test_task_store_wrapper_passes_when_the_db_exists(self, tmp_path: Path):
        db = task_store_path(tmp_path)
        db.parent.mkdir(parents=True)
        db.touch()

        assert assert_task_store_exists(tmp_path, operation='audit_duplicate_tasks') is None

    def test_task_store_wrapper_refuses_and_creates_nothing(self, tmp_path: Path):
        with pytest.raises(TargetStoreMissing) as excinfo:
            assert_task_store_exists(tmp_path, operation='audit_duplicate_tasks')

        message = str(excinfo.value)
        assert 'audit_duplicate_tasks' in message
        assert str(task_store_path(tmp_path)) in message
        assert '--project-root' in message
        assert not (tmp_path / '.taskmaster').exists()

    def test_queue_dir_wrapper_passes_when_the_directory_exists(self, tmp_path: Path):
        queue_dir = tmp_path / 'escalations'
        queue_dir.mkdir()

        assert assert_queue_dir_exists(queue_dir, operation='backfill_recon_escalations') is None

    def test_queue_dir_wrapper_refuses_and_creates_nothing(self, tmp_path: Path):
        queue_dir = tmp_path / 'data' / 'reconciliation' / 'escalations'

        with pytest.raises(TargetStoreMissing) as excinfo:
            assert_queue_dir_exists(queue_dir, operation='backfill_recon_escalations')

        message = str(excinfo.value)
        assert 'backfill_recon_escalations' in message
        assert str(queue_dir) in message
        assert '--queue-dir' in message
        assert not (tmp_path / 'data').exists()

    def test_the_two_families_carry_different_remedies(self, tmp_path: Path):
        """A shared helper must not flatten the two into one generic sentence.

        The remedy is the line an operator acts on, and the two families need
        opposite actions: point ``--project-root`` at the main checkout, versus
        pass an absolute ``--queue-dir``.
        """
        with pytest.raises(TargetStoreMissing) as task_exc:
            assert_task_store_exists(tmp_path, operation='op')
        with pytest.raises(TargetStoreMissing) as queue_exc:
            assert_queue_dir_exists(tmp_path / 'absent', operation='op')

        assert '--queue-dir' not in str(task_exc.value)
        assert '--project-root' not in str(queue_exc.value)
