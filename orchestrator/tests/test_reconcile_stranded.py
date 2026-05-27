"""Tests for Harness._reconcile_stranded_in_progress and the _pid_alive helper."""

import json
import logging
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.harness import Harness, _pid_alive

# ---------------------------------------------------------------------------
# _pid_alive helper tests
# ---------------------------------------------------------------------------

class TestPidAlive:
    def test_current_pid_is_alive(self):
        assert _pid_alive(os.getpid()) is True

    def test_impossible_pid_is_dead(self):
        # PID well beyond the Linux kernel max (2^22 on 64-bit, 2^15 on 32-bit).
        # 2**31-1 is always invalid on all Linux systems.
        assert _pid_alive(2**31 - 1) is False

    # ------------------------------------------------------------------
    # Branch-mocked tests — each covers exactly one code path in _pid_alive
    # ------------------------------------------------------------------

    def test_pid_zero_returns_false_without_calling_os_kill(self, monkeypatch):
        """pid=0 guard → returns False before os.kill is ever called."""
        calls: list[tuple[int, int]] = []
        monkeypatch.setattr(os, 'kill', lambda pid, sig: calls.append((pid, sig)))
        assert _pid_alive(0) is False
        assert calls == [], 'os.kill must not be called for pid=0'

    def test_negative_pid_returns_false_without_calling_os_kill(self, monkeypatch):
        """pid=-1 guard → returns False before os.kill is ever called."""
        calls: list[tuple[int, int]] = []
        monkeypatch.setattr(os, 'kill', lambda pid, sig: calls.append((pid, sig)))
        assert _pid_alive(-1) is False
        assert calls == [], 'os.kill must not be called for pid=-1'

    def test_process_lookup_error_returns_false(self, monkeypatch):
        """os.kill raises ProcessLookupError → process is dead → False."""
        def _raise(pid: int, sig: int) -> None:
            raise ProcessLookupError()
        monkeypatch.setattr(os, 'kill', _raise)
        assert _pid_alive(12345) is False

    def test_permission_error_returns_true(self, monkeypatch):
        """os.kill raises PermissionError → process exists (no permission to signal) → True."""
        def _raise(pid: int, sig: int) -> None:
            raise PermissionError()
        monkeypatch.setattr(os, 'kill', _raise)
        assert _pid_alive(12345) is True

    def test_generic_oserror_returns_false(self, monkeypatch):
        """os.kill raises generic OSError → treat as dead → False."""
        def _raise(pid: int, sig: int) -> None:
            raise OSError(5, 'io error')
        monkeypatch.setattr(os, 'kill', _raise)
        assert _pid_alive(12345) is False

    def test_successful_signal_returns_true(self, monkeypatch):
        """os.kill succeeds → process is alive → True."""
        monkeypatch.setattr(os, 'kill', lambda pid, sig: None)
        assert _pid_alive(12345) is True


# ---------------------------------------------------------------------------
# Harness fixture (mirrors test_crash_recovery.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def harness(tmp_path: Path, mock_orch_config):
    """Create a Harness with mocked internals for unit testing reconciliation."""
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    # Replace scheduler with async mocks
    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[])
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))
    h.scheduler.set_task_status = AsyncMock()

    # mark_done forwards to set_task_status so the existing assertions on
    # set_task_status.call_args_list / set_task_status.assert_awaited_once_with
    # still cover the recovery-done path after the harness was refactored to
    # use Scheduler.mark_done as a thin wrapper.
    async def _fake_mark_done(tid, *, kind, sha, note=None):
        provenance = {'kind': kind, 'commit': sha}
        if note is not None:
            provenance['note'] = note
        await h.scheduler.set_task_status(
            tid, 'done', done_provenance=provenance,
        )
    h.scheduler.mark_done = AsyncMock(side_effect=_fake_mark_done)

    # Keep worktree_base real (under tmp_path) so we can create fake worktrees
    h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()

    # Mock cleanup_worktree: side_effect actually removes the directory so that
    # existing assertions like `assert not lock_path.exists()` continue to hold
    # after the impl switches from lock_path.unlink() to cleanup_worktree().
    def _fake_cleanup(worktree_path, tid):
        shutil.rmtree(worktree_path, ignore_errors=True)

    h.git_ops.cleanup_worktree = AsyncMock(side_effect=_fake_cleanup)

    # Default: is_ancestor returns False so no guard fires for existing tests.
    # Individual tests may override with AsyncMock(return_value=True).
    h.git_ops.is_ancestor = AsyncMock(return_value=False)

    # Default: resolve_branch_sha returns a fixed SHA so tests that trigger the
    # is_ancestor guard get a consistent commit in done_provenance.
    # Individual tests may override with AsyncMock(return_value=None).
    h.git_ops.resolve_branch_sha = AsyncMock(return_value='deadbeef' + 'a' * 32)

    # Default: find_merge_marker returns None so no deleted-branch guard fires
    # for existing tests.  Individual tests may override with
    # AsyncMock(return_value='<sha>') to exercise the marker path.
    h.git_ops.find_merge_marker = AsyncMock(return_value=None)

    # Default: find_task_citation_commit returns a SHA matching the
    # resolve_branch_sha default so existing happy-path tests pass through the
    # new citation guard unchanged.  Individual tests may override with
    # AsyncMock(return_value=None) to exercise the missing-citation path or
    # with a different SHA to assert citation precedence over branch-tip.
    h.git_ops.find_task_citation_commit = AsyncMock(
        return_value='deadbeef' + 'a' * 32,
    )

    # Default: get_task returns None (no branch_base_sha metadata) so Guard 3
    # falls through and existing is_ancestor tests keep their prior semantics.
    # Individual tests may override with AsyncMock(return_value={'metadata': {...}})
    # to exercise the Guard 3 branch-advanced check.
    h.scheduler.get_task = AsyncMock(return_value=None)

    return h


# ---------------------------------------------------------------------------
# _reconcile_stranded_in_progress tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestReconcileStrandedInProgress:
    async def test_orphan_without_worktree_reverted(self, harness: Harness):
        """In-progress task with no worktree dir → reverted to pending (no-lock)."""
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {'5': 'in-progress', '6': 'pending'}, None
        )
        # No worktree directory for task 5 exists (worktree_base not even created)

        await harness._reconcile_stranded_in_progress()

        calls = harness.scheduler.set_task_status.call_args_list  # type: ignore[attr-defined]
        assert len(calls) == 1
        assert calls[0].args[0] == '5'
        assert calls[0].args[1] == 'pending'

    async def test_in_progress_with_live_owner_pid_left_alone(
        self, harness: Harness, tmp_path: Path, caplog
    ):
        """In-progress task with plan.lock pointing to live PID → untouched, no revert logged."""
        harness.scheduler.get_statuses.return_value = ({'7': 'in-progress'}, None)  # type: ignore[attr-defined]
        # Create worktree with a plan.lock containing our own (live) PID
        lock_dir = harness.git_ops.worktree_base / '7' / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': '7-abcd1234',
            'locked_at': datetime.now(UTC).isoformat(),
            'owner_pid': os.getpid(),
        }))

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
            await harness._reconcile_stranded_in_progress()

        # Must NOT revert
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        # Lock file must still exist
        assert lock_path.exists()
        # No revert must have been logged ('reverted task' matches the stable log format)
        assert not any('reverted task' in r.message for r in caplog.records)

    async def test_stale_plan_lock_cleared_and_reverted(
        self, harness: Harness, monkeypatch
    ):
        """In-progress task with stale plan.lock (dead PID) → lock cleared and task reverted."""
        harness.scheduler.get_statuses.return_value = ({'8': 'in-progress'}, None)  # type: ignore[attr-defined]
        # Use a synthetic owner_pid — _pid_alive is mocked to always return False,
        # so no real PID is needed and there is no kernel-recycle race.
        owner_pid = 99999
        monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)

        # Create worktree with plan.lock referencing the synthetic dead PID
        lock_dir = harness.git_ops.worktree_base / '8' / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': '8-dead0001',
            'locked_at': datetime.now(UTC).isoformat(),
            'owner_pid': owner_pid,
        }))

        await harness._reconcile_stranded_in_progress()

        # Task must be reverted to pending
        harness.scheduler.set_task_status.assert_called_once_with('8', 'pending')  # type: ignore[attr-defined]
        # Stale lock must be deleted
        assert not lock_path.exists()

    async def test_in_progress_with_open_l1_left_intact(
        self, harness: Harness, monkeypatch
    ):
        """In-progress task with an open L1 escalation → worktree NOT reaped.

        Reproduces the /unblock-session reap: an escalated task sits at
        'in-progress' with an open L1 while a human edits its worktree.  The
        worktree's plan.lock is stale (the agent exited), so WITHOUT the L1
        guard the stranded sweep would clear the lock, force-delete the
        worktree, and revert to pending.  The open L1 must veto all of that —
        return None, no cleanup_worktree, no status change, lock preserved.
        """
        from escalation.models import Escalation
        from escalation.queue import EscalationQueue

        harness.scheduler.get_statuses.return_value = ({'60': 'in-progress'}, None)  # type: ignore[attr-defined]

        # Stale lock (dead PID) — the would-be-reaped shape absent the guard.
        monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)
        lock_dir = harness.git_ops.worktree_base / '60' / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': '60-human',
            'locked_at': datetime.now(UTC).isoformat(),
            'owner_pid': 99999,
        }))

        # Real EscalationQueue with an open L1 for task 60 (the human handoff).
        queue_dir = harness.git_ops.worktree_base.parent / 'escalations'
        harness._escalation_queue = EscalationQueue(queue_dir)
        harness._escalation_queue.submit(Escalation(
            id=harness._escalation_queue.make_id('60'),
            task_id='60',
            agent_role='task-steward',
            severity='blocking',
            category='task_failure',
            summary='Escalated — human is unblocking in the worktree',
            level=1,
            status='pending',
        ))

        await harness._reconcile_stranded_in_progress()

        # The worktree must be left intact: no cleanup, no revert, lock present.
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        assert lock_path.exists()

    async def test_in_progress_without_l1_still_reaped(
        self, harness: Harness, monkeypatch
    ):
        """Inverse of the L1 guard: a stale-lock in-progress task with NO open
        L1 for *that* task is still reaped — pins the guard to the specific tid.

        An L1 exists for a *different* task (999); the target (61) has none, so
        has_open_l1('61') is False and the stale-lock recovery proceeds.
        """
        from escalation.models import Escalation
        from escalation.queue import EscalationQueue

        harness.scheduler.get_statuses.return_value = ({'61': 'in-progress'}, None)  # type: ignore[attr-defined]

        monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)
        lock_dir = harness.git_ops.worktree_base / '61' / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': '61-dead',
            'locked_at': datetime.now(UTC).isoformat(),
            'owner_pid': 99999,
        }))

        queue_dir = harness.git_ops.worktree_base.parent / 'escalations'
        harness._escalation_queue = EscalationQueue(queue_dir)
        # L1 belongs to an unrelated task — must NOT shield task 61.
        harness._escalation_queue.submit(Escalation(
            id=harness._escalation_queue.make_id('999'),
            task_id='999',
            agent_role='task-steward',
            severity='blocking',
            category='task_failure',
            summary='Unrelated escalation',
            level=1,
            status='pending',
        ))

        await harness._reconcile_stranded_in_progress()

        # No L1 for 61 → normal stale-lock recovery: reaped + reverted.
        harness.git_ops.cleanup_worktree.assert_awaited_once()  # type: ignore[attr-defined]
        harness.scheduler.set_task_status.assert_called_once_with('61', 'pending')  # type: ignore[attr-defined]
        assert not lock_path.exists()

    @pytest.mark.parametrize(
        'lock_contents,task_id,expect_reverted,expect_lock_exists,warn_pattern',
        [
            # (a) Corrupt JSON → revert + unlink (JSONDecodeError still caught)
            pytest.param(
                'not-valid-json', 9, True, False, None,
                id='corrupt-json',
            ),
            # (b) Missing owner_pid key → owner_pid=None via .get() → owner_alive=False
            #     → stale-lock path: cleanup_worktree called, task reverted, lock gone
            #     → WARNING emitted for observability (Gap 2)
            pytest.param(
                json.dumps({'session_id': 'test-10', 'locked_at': '2026-01-01T00:00:00+00:00'}),
                '10', True, False, r'no owner_pid; treating as stale',
                id='missing-owner-pid',
            ),
            # (b2) Explicit null owner_pid → owner_pid=None → owner_alive=False
            #      → stale-lock path: cleanup_worktree called, task reverted, lock gone
            #      → WARNING emitted for observability (Gap 2)
            pytest.param(
                json.dumps({'session_id': 'test-16', 'locked_at': '2026-01-01T00:00:00+00:00', 'owner_pid': None}),
                16, True, False, r'no owner_pid; treating as stale',
                id='null-owner-pid',
            ),
            # (b3) Non-numeric owner_pid → int('abc') raises ValueError
            #      → except (TypeError, ValueError) catches it → owner_alive=False
            #      → stale-lock path: cleanup_worktree called, task reverted, lock gone
            pytest.param(
                json.dumps({'session_id': 'test-42', 'locked_at': '2026-01-01T00:00:00+00:00', 'owner_pid': 'abc'}),
                42, True, False, None,
                id='non-numeric-owner-pid',
            ),
            # (e) Non-dict JSON (list) → treated as corruption → revert + unlink
            pytest.param(
                '["not", "an", "object"]', 14, True, False, None,
                id='non-dict-json',
            ),
            # (c) Numeric-string owner_pid of a live process → task NOT reverted
            #     Exercises the int(owner_pid) cast path with a string value
            pytest.param(
                'LIVE_PID', 11, False, True, None,
                id='live-pid-as-string',
            ),
            # (d1) No lock file, id as int → reverted via no-lock branch
            pytest.param(None, 12, True, False, None, id='no-lock-int-id'),
            # (d2) No lock file, id as str → reverted via no-lock branch
            pytest.param(None, '13', True, False, None, id='no-lock-str-id'),
        ],
    )
    async def test_reconcile_lock_format_variants(
        self,
        harness: Harness,
        caplog,
        lock_contents,
        task_id,
        expect_reverted: bool,
        expect_lock_exists: bool,
        warn_pattern,
    ):
        """Parametrized coverage of plan.lock format edge cases."""
        import logging

        harness.scheduler.get_statuses.return_value = ({str(task_id): 'in-progress'}, None)  # type: ignore[attr-defined]

        tid_str = str(task_id)
        lock_dir = harness.git_ops.worktree_base / tid_str / '.task'
        lock_path = lock_dir / 'plan.lock'

        if lock_contents is not None:
            # Resolve sentinel for live-PID case
            if lock_contents == 'LIVE_PID':
                lock_contents = json.dumps({
                    'session_id': f'{tid_str}-live',
                    'locked_at': '2026-01-01T00:00:00+00:00',
                    'owner_pid': str(os.getpid()),
                })
            lock_dir.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(lock_contents)

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await harness._reconcile_stranded_in_progress()

        calls = harness.scheduler.set_task_status.call_args_list  # type: ignore[attr-defined]
        if expect_reverted:
            assert len(calls) == 1, f'Expected 1 revert call, got: {calls}'
            assert calls[0].args[0] == tid_str, (
                f'Expected set_task_status called with id={tid_str!r}, got {calls[0].args[0]!r}'
            )
            assert calls[0].args[1] == 'pending'
        else:
            assert len(calls) == 0, f'Expected no calls (task untouched), got: {calls}'

        assert lock_path.exists() == expect_lock_exists, (
            f'Lock file existence mismatch: expected {expect_lock_exists}, '
            f'got {lock_path.exists()}'
        )

        if warn_pattern is not None:
            matching = [
                r for r in caplog.records
                if re.search(warn_pattern, r.message, re.IGNORECASE)
            ]
            assert len(matching) >= 1, (
                f'Expected WARNING matching {warn_pattern!r} in orchestrator.harness logs, '
                f'got: {[r.message for r in caplog.records]}'
            )
            assert matching[0].levelno == logging.WARNING, (
                f'Expected WARNING level, got {logging.getLevelName(matching[0].levelno)}'
            )

        # Verify cleanup_worktree call behavior.
        # When a worktree was created on disk (lock_contents is not None) and the
        # task was reverted, cleanup_worktree must have been called with the correct
        # args.  When no worktree exists (no-lock-*-id cases, lock_contents=None) or
        # the task was left alone (live-pid-as-string), cleanup_worktree must not fire.
        worktree_path = harness.git_ops.worktree_base / tid_str
        if expect_reverted and lock_contents is not None:
            harness.git_ops.cleanup_worktree.assert_called_once_with(  # type: ignore[attr-defined]
                worktree_path, tid_str
            )
        else:
            harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_no_lock_worktree_cleaned_when_not_recovered(
        self, harness: Harness, tmp_path: Path
    ):
        """Worktree dir exists but has no plan.lock and task is NOT in _recovered_plans
        → cleanup_worktree is called and task is reverted to pending."""
        tid = 30
        harness.scheduler.get_statuses.return_value = ({str(tid): 'in-progress'}, None)  # type: ignore[attr-defined]
        # Create the worktree directory (no .task/plan.lock inside)
        worktree_path = harness.git_ops.worktree_base / str(tid)
        worktree_path.mkdir(parents=True)
        # _recovered_plans is empty (default)

        await harness._reconcile_stranded_in_progress()

        # cleanup_worktree must have been called with the worktree path and tid
        harness.git_ops.cleanup_worktree.assert_called_once_with(worktree_path, str(tid))  # type: ignore[attr-defined]
        # Task must be reverted to pending
        harness.scheduler.set_task_status.assert_called_once_with(str(tid), 'pending')  # type: ignore[attr-defined]

    async def test_no_lock_worktree_preserved_when_recovered(
        self, harness: Harness, tmp_path: Path
    ):
        """Worktree dir exists but has no plan.lock and task IS in _recovered_plans
        → cleanup_worktree is NOT called (worktree preserved), task still reverted."""
        tid = 31
        harness.scheduler.get_statuses.return_value = ({str(tid): 'in-progress'}, None)  # type: ignore[attr-defined]
        # Create the worktree directory (no .task/plan.lock inside)
        worktree_path = harness.git_ops.worktree_base / str(tid)
        worktree_path.mkdir(parents=True)
        # Mark task as recovered — worktree must be preserved for resumption
        harness._recovered_plans[str(tid)] = {'task_id': str(tid), 'steps': []}

        await harness._reconcile_stranded_in_progress()

        # cleanup_worktree must NOT have been called
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        # Worktree directory must still exist
        assert worktree_path.exists()
        # Task must still be reverted to pending (recovery runs separately)
        harness.scheduler.set_task_status.assert_called_once_with(str(tid), 'pending')  # type: ignore[attr-defined]

    async def test_no_lock_worktree_preserved_when_in_preserved_set(
        self, harness: Harness, tmp_path: Path
    ):
        """Worktree dir exists but has no plan.lock and task IS in
        _preserved_worktrees → cleanup_worktree is NOT called (worktree
        preserved for revalidation), task still reverted to pending.

        Mirrors test_no_lock_worktree_preserved_when_recovered for the
        crash-recovery-stamped-but-not-pre-loaded case (the architect filed
        a stamped plan that was rejected by blast-radius lock conflict).
        """
        tid = '34'
        harness.scheduler.get_statuses.return_value = ({str(tid): 'in-progress'}, None)  # type: ignore[attr-defined]
        worktree_path = harness.git_ops.worktree_base / str(tid)
        worktree_path.mkdir(parents=True)
        # Mark task as preserved (architect ran, plan stamped, no done steps).
        harness._preserved_worktrees.add(str(tid))

        await harness._reconcile_stranded_in_progress()

        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        assert worktree_path.exists()
        harness.scheduler.set_task_status.assert_called_once_with(str(tid), 'pending')  # type: ignore[attr-defined]

    async def test_stale_lock_worktree_preserved_when_in_preserved_set(
        self, harness: Harness, monkeypatch
    ):
        """Stale plan.lock + task IS in _preserved_worktrees →
        cleanup_worktree NOT called, worktree kept, stale lock unlinked,
        task reverted.  Defensive: _recover_crashed_tasks already unlinks
        plan.lock when adding to _preserved_worktrees, so this combined
        state shouldn't appear in practice — locks the invariant against
        future drift, mirroring test_stale_lock_worktree_preserved_when_recovered.
        """
        tid = '35'
        harness.scheduler.get_statuses.return_value = ({str(tid): 'in-progress'}, None)  # type: ignore[attr-defined]
        monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)

        lock_dir = harness.git_ops.worktree_base / str(tid) / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': f'{tid}-dead',
            'locked_at': '2026-01-01T00:00:00+00:00',
            'owner_pid': 99999,
        }))
        worktree_path = harness.git_ops.worktree_base / str(tid)
        harness._preserved_worktrees.add(str(tid))

        await harness._reconcile_stranded_in_progress()

        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        assert worktree_path.exists()
        assert not lock_path.exists()
        harness.scheduler.set_task_status.assert_called_once_with(str(tid), 'pending')  # type: ignore[attr-defined]

    async def test_stale_lock_worktree_cleaned_when_not_recovered(
        self, harness: Harness, monkeypatch
    ):
        """In-progress task with stale plan.lock (dead PID), not in _recovered_plans
        → cleanup_worktree called (removing entire worktree dir), task reverted."""
        tid = 32
        harness.scheduler.get_statuses.return_value = ({str(tid): 'in-progress'}, None)  # type: ignore[attr-defined]
        monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)

        # Create worktree with a plan.lock referencing a synthetic dead PID
        lock_dir = harness.git_ops.worktree_base / str(tid) / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': f'{tid}-dead',
            'locked_at': '2026-01-01T00:00:00+00:00',
            'owner_pid': 99999,
        }))
        worktree_path = harness.git_ops.worktree_base / str(tid)
        # _recovered_plans is empty (default)

        await harness._reconcile_stranded_in_progress()

        # cleanup_worktree must have been called (rmtree removes entire worktree)
        harness.git_ops.cleanup_worktree.assert_called_once_with(worktree_path, str(tid))  # type: ignore[attr-defined]
        # Task must be reverted to pending
        harness.scheduler.set_task_status.assert_called_once_with(str(tid), 'pending')  # type: ignore[attr-defined]
        # The entire worktree dir is gone (side_effect rmtree'd it)
        assert not worktree_path.exists()

    async def test_stale_lock_worktree_preserved_when_recovered(
        self, harness: Harness, monkeypatch
    ):
        """In-progress task with stale plan.lock (dead PID), task IS in _recovered_plans
        → cleanup_worktree NOT called, worktree preserved, stale lock unlinked, task reverted.

        NOTE — defensive branch only: this combined state (recovered plan + stale lock still
        present) is unreachable in the normal startup flow.  _recover_crashed_tasks always
        unlinks plan.lock before adding a task to _recovered_plans (harness.py:864-868), so
        in practice a recovered task arrives at the no-lock branch, not the stale-lock branch.
        This test exists to lock the invariant against future drift in the recovery path.
        """
        tid = 33
        harness.scheduler.get_statuses.return_value = ({str(tid): 'in-progress'}, None)  # type: ignore[attr-defined]
        monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)

        # Create worktree with a plan.lock referencing a synthetic dead PID
        lock_dir = harness.git_ops.worktree_base / str(tid) / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': f'{tid}-dead',
            'locked_at': '2026-01-01T00:00:00+00:00',
            'owner_pid': 99999,
        }))
        worktree_path = harness.git_ops.worktree_base / str(tid)
        # Mark task as recovered — worktree must be preserved for resumption
        harness._recovered_plans[str(tid)] = {'task_id': str(tid), 'steps': []}

        await harness._reconcile_stranded_in_progress()

        # cleanup_worktree must NOT have been called
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        # Worktree directory must still exist
        assert worktree_path.exists()
        # Stale lock must be removed (so the resumed session doesn't immediately requeue)
        assert not lock_path.exists()
        # Task must be reverted to pending
        harness.scheduler.set_task_status.assert_called_once_with(str(tid), 'pending')  # type: ignore[attr-defined]

    async def test_stale_lock_unlinked_when_cleanup_worktree_raises(
        self, harness: Harness, monkeypatch, caplog
    ):
        """Stale lock must be unlinked even when cleanup_worktree raises (Gap 3).

        If cleanup_worktree fails (e.g., permission error on the worktree dir),
        the plan.lock file must still be removed before the task is reverted to
        pending.  Without the fix, a subsequent reconcile sweep would re-encounter
        the lock, find its owner dead again, and loop forever.
        """
        tid = 40
        harness.scheduler.get_statuses.return_value = ({str(tid): 'in-progress'}, None)  # type: ignore[attr-defined]
        monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)

        # Create worktree with a plan.lock referencing a synthetic dead PID
        lock_dir = harness.git_ops.worktree_base / str(tid) / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': f'{tid}-dead',
            'locked_at': '2026-01-01T00:00:00+00:00',
            'owner_pid': 99999,
        }))
        worktree_path = harness.git_ops.worktree_base / str(tid)

        # Make cleanup_worktree raise so we exercise the except-branch
        harness.git_ops.cleanup_worktree = AsyncMock(side_effect=OSError('boom'))  # type: ignore[attr-defined]

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await harness._reconcile_stranded_in_progress()

        # cleanup_worktree must have been attempted
        harness.git_ops.cleanup_worktree.assert_called_once_with(worktree_path, str(tid))  # type: ignore[attr-defined]
        # Task must still be reverted to pending despite cleanup failure
        harness.scheduler.set_task_status.assert_called_once_with(str(tid), 'pending')  # type: ignore[attr-defined]
        # Lock must be gone — the unlink must happen unconditionally after cleanup
        assert not lock_path.exists(), 'plan.lock must be unlinked even when cleanup_worktree raises'
        # Cleanup-failure WARNING must be present in logs
        matching = [
            r for r in caplog.records
            if re.search(r'cleanup_worktree failed.*40.*stale-lock', r.message, re.IGNORECASE)
        ]
        assert len(matching) >= 1, (
            f'Expected cleanup-failure WARNING in harness logs, got: {[r.message for r in caplog.records]}'
        )

    async def test_no_lock_branch_cleanup_worktree_raises_still_reverts(
        self, harness: Harness, caplog
    ):
        """Regression lockdown: task is reverted even when cleanup_worktree raises
        in the no-lock branch.  Covers the uncovered except Exception at harness.py:908-913.

        The no-lock branch has no lock to unlink; after cleanup failure it must
        still call set_task_status so the task escapes in-progress.
        """
        tid = 41
        harness.scheduler.get_statuses.return_value = ({str(tid): 'in-progress'}, None)  # type: ignore[attr-defined]
        # Create worktree dir with NO plan.lock inside
        worktree_path = harness.git_ops.worktree_base / str(tid)
        worktree_path.mkdir(parents=True)

        # Make cleanup_worktree raise so we exercise the except-branch
        harness.git_ops.cleanup_worktree = AsyncMock(side_effect=OSError('boom'))  # type: ignore[attr-defined]

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await harness._reconcile_stranded_in_progress()

        # cleanup_worktree must have been attempted
        harness.git_ops.cleanup_worktree.assert_called_once_with(worktree_path, str(tid))  # type: ignore[attr-defined]
        # Task must still be reverted to pending despite cleanup failure
        harness.scheduler.set_task_status.assert_called_once_with(str(tid), 'pending')  # type: ignore[attr-defined]
        # Cleanup-failure WARNING must be present in logs
        matching = [
            r for r in caplog.records
            if re.search(r'cleanup_worktree failed.*41.*no-lock', r.message, re.IGNORECASE)
        ]
        assert len(matching) >= 1, (
            f'Expected cleanup-failure WARNING in harness logs, got: {[r.message for r in caplog.records]}'
        )

    async def test_reconcile_uses_get_statuses_not_get_tasks(self, harness: Harness):
        """_reconcile_stranded_in_progress must use get_statuses, not get_tasks.

        RED against current code: harness still calls get_tasks.
        After the migration (step-14 impl), get_statuses is called and
        get_tasks is never called for the reconcile sweep.
        """
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {'5': 'in-progress', '6': 'pending'}, None
        )
        # No worktree for task 5 (orphan → will be reverted to pending)
        # Task 6 is pending → not touched

        await harness._reconcile_stranded_in_progress()

        harness.scheduler.get_statuses.assert_called_once()  # type: ignore[attr-defined]
        harness.scheduler.get_tasks.assert_not_called()  # type: ignore[attr-defined]
        calls = harness.scheduler.set_task_status.call_args_list  # type: ignore[attr-defined]
        assert len(calls) == 1
        assert calls[0].args[0] == '5'
        assert calls[0].args[1] == 'pending'

    async def test_unexpected_exception_propagates_out_of_reconcile(
        self, harness: Harness
    ):
        """TypeError from json.loads must propagate — not be silently swallowed.

        RED against current code: `except Exception:` catches TypeError and
        treats the lock as stale (task reverted, lock deleted, no exception).
        After the fix (narrow to OSError/JSONDecodeError/ValueError), TypeError
        propagates out, set_task_status is never called, and the lock survives.
        """
        from unittest.mock import patch as _patch

        harness.scheduler.get_statuses.return_value = ({'15': 'in-progress'}, None)  # type: ignore[attr-defined]
        lock_dir = harness.git_ops.worktree_base / '15' / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text('{"session_id": "15-xyz", "owner_pid": 1}')  # valid-looking

        with _patch('orchestrator.harness.json.loads', side_effect=TypeError('unexpected')), pytest.raises(TypeError, match='unexpected'):
            await harness._reconcile_stranded_in_progress()

        # No revert must have happened
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        # Lock file must not have been deleted
        assert lock_path.exists(), 'Lock file must survive when an unexpected exception propagates'

    async def test_already_merged_branch_marked_done_with_provenance(
        self, harness: Harness
    ):
        """Stranded in-progress task whose branch is already merged to main →
        marked done with provenance; no pending revert; no cleanup_worktree.

        RED state: the guard doesn't exist yet; reconcile takes the no-lock
        branch and calls set_task_status('50', 'pending'), never calling
        is_ancestor.
        """
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {'50': 'in-progress'}, None
        )
        # No worktree dir or plan.lock for task 50 — guard must fire before
        # any worktree analysis.

        await harness._reconcile_stranded_in_progress()

        # is_ancestor must have been invoked with the configured branch + main_branch
        harness.git_ops.is_ancestor.assert_awaited_once_with('task/50', 'main')  # type: ignore[attr-defined]

        # set_task_status must be called exactly once: ('50', 'done') with kind/commit/note
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '50', 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': 'deadbeef' + 'a' * 32,
                'note': 'reconcile: branch already on main when stranded in-progress',
            },
        )
        # The found_on_main SHA now comes from the citation-commit grep
        # rather than the bare branch tip (gated by the post-fix Guard 2).
        harness.git_ops.find_task_citation_commit.assert_awaited_once_with(  # type: ignore[attr-defined]
            '50', pattern_template=None,
        )

        # cleanup_worktree must NOT have been called
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_already_merged_skipped_when_l1_escalation_open(
        self, harness: Harness, tmp_path: Path
    ):
        """Open L1 escalation for the task → reconciler must NOT flip to done.

        Reproduces the reify-3399 false-positive: branch tip equals main HEAD
        because the architect was never given a chance to write any commits
        (the task was declared unactionable and _mark_blocked(escalate_to_human=True)
        fired before any commit landed).  The deliberate human-escalation
        disposition must take precedence over the degenerate is_ancestor==True
        observation.
        """
        from escalation.models import Escalation
        from escalation.queue import EscalationQueue

        # Wire up a real EscalationQueue and submit an L1 record for task 50.
        queue_dir = tmp_path / 'escalations'
        harness._escalation_queue = EscalationQueue(queue_dir)
        harness._escalation_queue.submit(Escalation(
            id=harness._escalation_queue.make_id('50'),
            task_id='50',
            agent_role='task-steward',
            severity='blocking',
            category='task_failure',
            summary='Task declared unactionable',
            level=1,
            status='pending',
        ))

        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {'50': 'blocked'}, None
        )

        await harness._reconcile_stranded_in_progress()

        # is_ancestor was invoked (proves we got into the branch).
        harness.git_ops.is_ancestor.assert_awaited_once_with('task/50', 'main')  # type: ignore[attr-defined]

        # But neither mark_done nor set_task_status('done', ...) fired —
        # the L1 guard bailed before the flip.
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    async def test_already_merged_skipped_when_main_lacks_task_citation(
        self, harness: Harness
    ):
        """is_ancestor=True but no commit on main cites the task id →
        reconciler must NOT flip to done.

        Defends the same reify-3399 false-positive shape against tasks without
        an L1 escalation: a zero-commit branch tip still sits on a main
        ancestor, but if no commit on main mentions ``task/50`` (or matches the
        citation pattern), there is no positive evidence of merge and the task
        must stay in its current status.
        """
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.git_ops.find_task_citation_commit = AsyncMock(  # type: ignore[attr-defined]
            return_value=None,
        )
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {'50': 'blocked'}, None
        )

        await harness._reconcile_stranded_in_progress()

        harness.git_ops.is_ancestor.assert_awaited_once_with('task/50', 'main')  # type: ignore[attr-defined]
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    async def test_already_merged_uses_citation_commit_when_available(
        self, harness: Harness
    ):
        """is_ancestor=True and a commit on main cites the task id →
        ``done_provenance.commit`` records the citation SHA, NOT the bare
        branch tip from ``resolve_branch_sha``.

        The matched commit is stronger evidence than ``is_ancestor`` alone:
        the citation tells us exactly *which* commit lands the task, so the
        provenance row points at that commit rather than at the (possibly
        post-merge) branch ref.
        """
        citation_sha = 'cafefeed' + 'b' * 32
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.git_ops.find_task_citation_commit = AsyncMock(  # type: ignore[attr-defined]
            return_value=citation_sha,
        )
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {'50': 'in-progress'}, None
        )

        await harness._reconcile_stranded_in_progress()

        # mark_done was called with the citation SHA, not the branch-tip SHA.
        harness.scheduler.mark_done.assert_awaited_once()  # type: ignore[attr-defined]
        kwargs = harness.scheduler.mark_done.call_args.kwargs  # type: ignore[attr-defined]
        assert kwargs['sha'] == citation_sha, (
            f"mark_done must record the citation SHA "
            f"({citation_sha!r}), got {kwargs['sha']!r}"
        )
        assert kwargs['kind'] == 'found_on_main'

    async def test_already_merged_drops_recovered_plan_and_cleans_worktree(
        self, harness: Harness
    ):
        """Regression: when is_ancestor=True and the task has a recovered plan,
        the stale _recovered_plans entry must be dropped and the orphaned
        worktree must be cleaned up — no entry should linger after the task
        transitions to a terminal 'done' state where resumption is impossible.
        """
        tid = '52'
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {tid: 'in-progress'}, None
        )
        # Seed a recovered plan — simulates _recover_crashed_tasks having run
        harness._recovered_plans[tid] = {'task_id': tid, 'steps': []}

        # Create the worktree dir on disk so the cleanup branch is reachable
        worktree_path = harness.git_ops.worktree_base / tid
        worktree_path.mkdir(parents=True)

        await harness._reconcile_stranded_in_progress()

        # (1) Stale recovered-plan entry must be dropped
        assert tid not in harness._recovered_plans, (
            '_recovered_plans entry must be popped when branch is already on main'
        )

        # (2) cleanup_worktree must be called exactly once (unconditional cleanup)
        harness.git_ops.cleanup_worktree.assert_awaited_once_with(  # type: ignore[attr-defined]
            worktree_path, tid
        )

        # (3) Task must be marked done with the expected provenance (kind + commit + note)
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': 'deadbeef' + 'a' * 32,
                'note': 'reconcile: branch already on main when stranded in-progress',
            },
        )

        # (4) Worktree dir is gone — proves cleanup_worktree's rmtree side_effect ran
        assert not worktree_path.exists(), (
            'worktree dir must be removed by cleanup_worktree'
        )

    async def test_already_merged_takes_precedence_over_stale_lock(
        self, harness: Harness, monkeypatch
    ):
        """Placement-precedence regression lock: is_ancestor guard fires BEFORE
        the stale-lock analysis.

        A task with a stale plan.lock AND is_ancestor=True must take the done
        path (no pending revert, stale-lock analysis bypassed). The guard also
        cleans up the stale worktree dir (amendment: prevents worktree cruft
        accumulation when orchestrator crashed after merge but before cleanup).
        This test would fail if a future refactor moved the guard below the
        lock analysis (set_task_status would be called with 'pending').
        """
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {'51': 'in-progress'}, None
        )
        monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)

        # Create a worktree with a stale plan.lock (dead PID)
        worktree_path = harness.git_ops.worktree_base / '51'
        lock_dir = worktree_path / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': '51-dead',
            'locked_at': '2026-01-01T00:00:00+00:00',
            'owner_pid': 99999,
        }))

        await harness._reconcile_stranded_in_progress()

        # Must be marked done, NOT reverted to pending
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '51', 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': 'deadbeef' + 'a' * 32,
                'note': 'reconcile: branch already on main when stranded in-progress',
            },
        )

        # cleanup_worktree IS called — the guard cleans up stale worktrees for
        # already-merged tasks to prevent worktree cruft from accumulating
        harness.git_ops.cleanup_worktree.assert_called_once_with(  # type: ignore[attr-defined]
            worktree_path, '51'
        )

        # plan.lock is gone — cleanup_worktree's side_effect rmtree'd the dir,
        # proving the stale-lock analysis branch was bypassed (which would have
        # also called set_task_status('51', 'pending') if it had run)
        assert not lock_path.exists(), (
            'plan.lock should be removed by cleanup_worktree in the is_ancestor guard'
        )

    # NB: an earlier test (``test_already_merged_skipped_when_branch_unresolved``)
    # covered the ``resolve_branch_sha returns None`` race during the
    # is_ancestor fast-path.  After the post-fix flow uses
    # ``find_task_citation_commit`` to source the done_provenance SHA, that
    # race is structurally impossible (the citation grep does not depend on
    # a live branch ref).  Skip-without-flip semantics on missing evidence
    # are covered by ``test_already_merged_skipped_when_main_lacks_task_citation``.

    # ------------------------------------------------------------------
    # Guard 3 — branch-advanced check (is_ancestor fast-path)
    # Guards 1 (open L1) and 2 (citation grep) already passed;
    # Guard 3 structurally rejects zero-commit branches sitting on main.
    # ------------------------------------------------------------------

    async def test_already_merged_skipped_when_branch_never_advanced(
        self, harness: Harness
    ):
        """Guard 3: branch tip == branch_base_sha → never advanced; veto flip.

        Guards 1+2 would both pass (no open L1, citation grep hits), but the
        branch base equals the current tip, proving no commits were ever pushed
        on this incarnation.  The reconciler must NOT flip to done.
        """
        branch_tip = 'deadbeef' + 'a' * 32
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)
        harness.git_ops.find_task_citation_commit = AsyncMock(
            return_value='cafefeed' + 'b' * 32
        )
        # Branch tip equals base → branch never advanced past creation point
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=branch_tip)
        harness.scheduler.get_task = AsyncMock(
            return_value={'id': '50', 'metadata': {'branch_base_sha': branch_tip}}
        )
        harness.scheduler.get_statuses.return_value = ({'50': 'blocked'}, None)  # type: ignore[attr-defined]

        await harness._reconcile_stranded_in_progress()

        # Guard 3 must veto even though Guards 1+2 passed
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    async def test_already_merged_flips_when_branch_advanced(
        self, harness: Harness
    ):
        """Guard 3 positive case: branch tip != branch_base_sha → flip proceeds.

        branch_base_sha records the creation-point; the current tip has advanced
        (commits were pushed), so Guard 3 must NOT block the flip.
        """
        branch_tip = 'deadbeef' + 'a' * 32
        branch_base = 'aaaa' * 10  # different SHA → branch advanced
        citation_sha = 'cafefeed' + 'b' * 32
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)
        harness.git_ops.find_task_citation_commit = AsyncMock(return_value=citation_sha)
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=branch_tip)
        harness.scheduler.get_task = AsyncMock(
            return_value={'id': '50', 'metadata': {'branch_base_sha': branch_base}}
        )
        harness.scheduler.get_statuses.return_value = ({'50': 'blocked'}, None)  # type: ignore[attr-defined]

        await harness._reconcile_stranded_in_progress()

        # Branch advanced past base → Guard 3 must NOT block; flip to done
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '50', 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': citation_sha,
                'note': 'reconcile: branch on main while task was blocked (out-of-band merge)',
            },
        )

    async def test_already_merged_falls_through_when_branch_base_sha_missing_or_malformed(
        self, harness: Harness
    ):
        """Guard 3 backward-compat: missing/malformed branch_base_sha → fall through.

        Tasks created before branch_base_sha was introduced (or whose metadata
        write failed) must still be flipped to done by Guards 1+2 alone.
        Both the missing-key case and the malformed-value case are covered.
        """
        citation_sha = 'cafefeed' + 'b' * 32
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)
        harness.git_ops.find_task_citation_commit = AsyncMock(return_value=citation_sha)
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value='deadbeef' + 'a' * 32)
        harness.scheduler.get_statuses.return_value = ({'50': 'blocked'}, None)  # type: ignore[attr-defined]

        for metadata_value in [
            {},                              # missing key
            {'branch_base_sha': 'short'},    # malformed — not 40 hex chars
        ]:
            harness.scheduler.get_task = AsyncMock(
                return_value={'id': '50', 'metadata': metadata_value}
            )
            harness.scheduler.set_task_status.reset_mock()  # type: ignore[attr-defined]
            harness.scheduler.mark_done.reset_mock()  # type: ignore[attr-defined]

            await harness._reconcile_stranded_in_progress()

            # Guard 3 must short-circuit and defer to Guards 1+2
            harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
                '50', 'done',
                done_provenance={
                    'kind': 'found_on_main',
                    'commit': citation_sha,
                    'note': 'reconcile: branch on main while task was blocked (out-of-band merge)',
                },
            )

    # ------------------------------------------------------------------
    # find_merge_marker guard tests (deleted-branch fast-path)
    # ------------------------------------------------------------------

    async def test_deleted_branch_with_merge_marker_marked_done(
        self, harness: Harness
    ):
        """Stranded in-progress task whose branch was deleted but whose merge
        marker is found on main → marked done with {commit, note} provenance.

        is_ancestor=False (branch doesn't exist, so is_ancestor can't resolve it),
        find_merge_marker returns a SHA → task must be marked done with the marker
        SHA in done_provenance['commit'] and cleanup_worktree must NOT be called
        (no worktree dir was created in this test).
        """
        tid = '70'
        marker_sha = 'abc123def' + 'a' * 31
        harness.git_ops.find_merge_marker = AsyncMock(  # type: ignore[attr-defined]
            return_value=marker_sha
        )
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {tid: 'in-progress'}, None
        )
        # No worktree dir for task 70

        await harness._reconcile_stranded_in_progress()

        # find_merge_marker must have been invoked with the full branch name
        harness.git_ops.find_merge_marker.assert_awaited_once_with(  # type: ignore[attr-defined]
            f'task/{tid}'
        )

        # Task must be marked done with kind + commit + note provenance
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': marker_sha,
                'note': 'reconcile: branch deleted but merge marker found on main',
            },
        )

        # No worktree to clean up
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_deleted_branch_no_merge_marker_falls_through_to_revert(
        self, harness: Harness
    ):
        """Stranded in-progress task whose branch is deleted and whose marker is
        absent → falls through to the existing revert-to-pending path.

        Proves the marker guard does NOT swallow the no-lock / no-marker case:
        the task must still be reverted to pending so it can be re-queued.
        """
        tid = '71'
        # Default: find_merge_marker returns None (already in fixture)
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {tid: 'in-progress'}, None
        )
        # No worktree, no lock

        await harness._reconcile_stranded_in_progress()

        # Must fall through to the revert path
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'pending'
        )

    async def test_marker_takes_precedence_over_stale_lock(
        self, harness: Harness, monkeypatch
    ):
        """Placement-precedence: find_merge_marker guard fires BEFORE the
        stale-lock analysis.

        A task with a stale plan.lock AND a merge marker must take the done
        path with marker provenance.  cleanup_worktree is called once (worktree
        dir existed), and the stale-lock branch is bypassed entirely.
        """
        tid = '72'
        marker_sha = 'deadc0de' + 'b' * 32
        harness.git_ops.find_merge_marker = AsyncMock(  # type: ignore[attr-defined]
            return_value=marker_sha
        )
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {tid: 'in-progress'}, None
        )
        monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)

        # Create a worktree with a stale plan.lock (dead PID)
        worktree_path = harness.git_ops.worktree_base / tid
        lock_dir = worktree_path / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': f'{tid}-dead',
            'locked_at': '2026-01-01T00:00:00+00:00',
            'owner_pid': 99999,
        }))

        await harness._reconcile_stranded_in_progress()

        # Must be marked done with marker provenance, NOT reverted to pending
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': marker_sha,
                'note': 'reconcile: branch deleted but merge marker found on main',
            },
        )

        # cleanup_worktree IS called — worktree dir existed
        harness.git_ops.cleanup_worktree.assert_called_once_with(  # type: ignore[attr-defined]
            worktree_path, tid
        )

        # plan.lock is gone — cleanup_worktree's rmtree side_effect ran,
        # proving the stale-lock branch was bypassed
        assert not lock_path.exists(), (
            'plan.lock should be removed by cleanup_worktree in the marker guard'
        )

    async def test_marker_drops_recovered_plan_and_cleans_worktree(
        self, harness: Harness
    ):
        """Regression: when find_merge_marker returns a SHA and the task has a
        recovered plan, the stale _recovered_plans entry must be dropped and the
        orphaned worktree must be cleaned up.

        Analog of test_already_merged_drops_recovered_plan_and_cleans_worktree
        for the marker path.
        """
        tid = '73'
        marker_sha = 'cafe1234' + 'c' * 32
        harness.git_ops.find_merge_marker = AsyncMock(  # type: ignore[attr-defined]
            return_value=marker_sha
        )
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {tid: 'in-progress'}, None
        )
        # Seed a recovered plan — simulates _recover_crashed_tasks having run
        harness._recovered_plans[tid] = {'task_id': tid, 'steps': []}

        # Create the worktree dir on disk so the cleanup branch is reachable
        worktree_path = harness.git_ops.worktree_base / tid
        worktree_path.mkdir(parents=True)

        await harness._reconcile_stranded_in_progress()

        # (1) Stale recovered-plan entry must be dropped
        assert tid not in harness._recovered_plans, (
            '_recovered_plans entry must be popped when marker is found on main'
        )

        # (2) cleanup_worktree must be called exactly once
        harness.git_ops.cleanup_worktree.assert_awaited_once_with(  # type: ignore[attr-defined]
            worktree_path, tid
        )

        # (3) Task must be marked done with marker provenance
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': marker_sha,
                'note': 'reconcile: branch deleted but merge marker found on main',
            },
        )

        # (4) Worktree dir is gone — cleanup_worktree's rmtree side_effect ran
        assert not worktree_path.exists(), (
            'worktree dir must be removed by cleanup_worktree'
        )

    # ------------------------------------------------------------------
    # Stale-marker check tests (find_merge_marker path)
    # A marker that pre-dates the current branch incarnation must be
    # rejected; a marker from this incarnation must flip the task.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        'status',
        [
            pytest.param('in-progress', id='in-progress'),
            pytest.param('blocked', id='blocked'),
        ],
    )
    async def test_marker_skipped_when_marker_predates_branch_base_sha_parametrized(
        self, harness: Harness, status: str,
    ):
        """Stale-marker veto is status-agnostic: predates branch_base_sha → must NOT flip.

        The veto at harness.py:1312-1321 fires on the second is_ancestor call's
        result (True = marker is ancestor of base = stale = prior incarnation) and
        short-circuits with `return None` BEFORE the status-aware note construction
        at harness.py:1323-1327 runs.  The blocked-flavor message
        'reconcile: merge marker found on main while task was blocked' (and the
        in-progress fixed message) must NOT fire for vetoed markers.
        """
        marker_sha = 'cafebabe' + 'd' * 32
        branch_base = 'beef0000' + '9' * 32
        tid = '80'
        # First is_ancestor: branch check → False (skip is_ancestor fast-path)
        # Second is_ancestor: stale-marker check → True (marker predates base)
        harness.git_ops.is_ancestor = AsyncMock(side_effect=[False, True])
        harness.git_ops.find_merge_marker = AsyncMock(return_value=marker_sha)
        harness.scheduler.get_task = AsyncMock(
            return_value={'id': tid, 'metadata': {'branch_base_sha': branch_base}}
        )
        harness.scheduler.get_statuses.return_value = ({tid: status}, None)  # type: ignore[attr-defined]

        await harness._reconcile_stranded_in_progress()

        # Stale marker from prior incarnation — must NOT flip to done regardless of status
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        'status,expected_note',
        [
            pytest.param(
                'in-progress',
                'reconcile: branch deleted but merge marker found on main',
                id='in-progress',
            ),
            pytest.param(
                'blocked',
                'reconcile: merge marker found on main while task was blocked',
                id='blocked',
            ),
        ],
    )
    async def test_marker_accepted_when_marker_postdates_branch_base_sha_parametrized(
        self, harness: Harness, status: str, expected_note: str,
    ):
        """Stale-marker accept path: marker postdates base → flip to done with status-aware note.

        Exercises the status-aware note conditional at harness.py:1323-1327:
          - status == 'blocked' → 'reconcile: merge marker found on main while task was blocked'
          - any other status (here 'in-progress') → 'reconcile: branch deleted but merge marker found on main'

        The expected_note for the blocked arm is the verbatim f-string output and is
        pinned to catch regressions in the conditional.  The in-progress arm pins
        the fixed else-branch string.
        """
        marker_sha = 'cafebabe' + 'd' * 32
        branch_base = 'beef0000' + '9' * 32
        tid = '81'
        # First is_ancestor: branch check → False (skip is_ancestor fast-path)
        # Second is_ancestor: stale-marker check → False (marker postdates base)
        harness.git_ops.is_ancestor = AsyncMock(side_effect=[False, False])
        harness.git_ops.find_merge_marker = AsyncMock(return_value=marker_sha)
        harness.scheduler.get_task = AsyncMock(
            return_value={'id': tid, 'metadata': {'branch_base_sha': branch_base}}
        )
        harness.scheduler.get_statuses.return_value = ({tid: status}, None)  # type: ignore[attr-defined]

        await harness._reconcile_stranded_in_progress()

        # Marker is from this incarnation — must flip to done with the status-appropriate note
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': marker_sha,
                'note': expected_note,
            },
        )

    async def test_marker_falls_through_when_branch_base_sha_missing(
        self, harness: Harness
    ):
        """Stale-marker backward-compat: missing branch_base_sha → accept marker.

        Tasks created before branch_base_sha was introduced must still be
        flipped to done by the existing marker guard.  The new stale-marker
        check must short-circuit on missing metadata and NOT run a bogus
        is_ancestor call against None.
        """
        marker_sha = 'cafebabe' + 'd' * 32
        # is_ancestor: first call → False (branch check); must NOT be called again
        is_ancestor_mock = AsyncMock(return_value=False)
        harness.git_ops.is_ancestor = is_ancestor_mock
        harness.git_ops.find_merge_marker = AsyncMock(return_value=marker_sha)
        # No branch_base_sha in metadata
        harness.scheduler.get_task = AsyncMock(
            return_value={'id': '82', 'metadata': {}}
        )
        harness.scheduler.get_statuses.return_value = ({'82': 'in-progress'}, None)  # type: ignore[attr-defined]

        await harness._reconcile_stranded_in_progress()

        # Existing marker-flip must happen
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '82', 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': marker_sha,
                'note': 'reconcile: branch deleted but merge marker found on main',
            },
        )

        # is_ancestor must only have been called once (branch check), NOT for
        # the stale-marker check (no valid SHA to compare against)
        assert is_ancestor_mock.await_count == 1, (
            'is_ancestor must not be called for stale-marker check when '
            'branch_base_sha is missing'
        )

    async def test_stale_marker_veto_leaves_inprogress_without_lock_revert(
        self, harness: Harness, tmp_path: Path
    ):
        """Stale-marker veto returns None early — lock-state revert path skipped.

        When status='in-progress', branch is gone, and the stale-marker check
        rejects the marker (it predates branch_base_sha), the reconciler must
        return None immediately.  The lock-state classification path that would
        normally revert a task with no lock to 'pending' must NOT fire — it is
        only reached when there is no on-main evidence at all.

        Pinning this behaviour is intentional (see reviewer suggestion #4): the
        task will be re-evaluated on the next sweep with the same stale-marker
        rejection; the no-lock revert is a separate concern for tasks that have
        no evidence either way.
        """
        tid = '83'
        marker_sha = 'deadc0de' + 'f' * 32
        branch_base = 'babe0000' + '9' * 32
        # is_ancestor: first call (branch check) → False; second call
        # (stale-marker check) → True (marker predates branch creation)
        harness.git_ops.is_ancestor = AsyncMock(side_effect=[False, True])  # type: ignore[attr-defined]
        harness.git_ops.find_merge_marker = AsyncMock(return_value=marker_sha)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(
            return_value={'id': tid, 'metadata': {'branch_base_sha': branch_base}}
        )
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]

        # No worktree directory → if the revert path fired it would call
        # set_task_status(tid, 'pending') for the no-lock orphan case.

        changed = await harness._reconcile_stranded_in_progress()

        # Stale-marker veto returns None early → changed == 0, no status
        # mutation of any kind.
        assert changed == 0
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    async def test_find_merge_marker_not_invoked_when_is_ancestor_true(
        self, harness: Harness
    ):
        """Efficiency lock: find_merge_marker is never called when is_ancestor
        returns True.

        The is_ancestor branch short-circuits via `continue` before the marker
        guard is reached.
        """
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {'74': 'in-progress'}, None
        )

        await harness._reconcile_stranded_in_progress()

        # is_ancestor fired → should NOT call find_merge_marker
        harness.git_ops.find_merge_marker.assert_not_called()  # type: ignore[attr-defined]
        # But the task must be marked done via the is_ancestor path
        harness.scheduler.set_task_status.assert_awaited_once()  # type: ignore[attr-defined]

    async def test_is_ancestor_not_invoked_for_terminal_or_pending_tasks(
        self, harness: Harness
    ):
        """Placement-efficiency regression lock: is_ancestor is never called
        for tasks the sweep does not consider stranded — i.e. anything outside
        ``{in-progress, blocked}``.

        Note: 'blocked' is now included in the sweep (R4) — out-of-band-merged
        blocked tasks need recovery — so the input set here is restricted to
        statuses the sweep does NOT touch.
        """
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {
                '60': 'pending',
                '61': 'done',
                '63': 'cancelled',
                '64': 'review',
                '65': 'deferred',
            },
            None,
        )

        await harness._reconcile_stranded_in_progress()

        # is_ancestor must never be called (no sweep-eligible tasks)
        harness.git_ops.is_ancestor.assert_not_called()  # type: ignore[attr-defined]
        # No status changes either
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    async def test_find_merge_marker_not_invoked_for_terminal_or_pending_tasks(
        self, harness: Harness
    ):
        """Placement-efficiency regression lock: find_merge_marker is never
        called for tasks the sweep does not consider stranded.

        Same reasoning as ``test_is_ancestor_not_invoked_for_terminal_or_pending_tasks``:
        only ``{in-progress, blocked}`` are swept after R4.
        """
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {
                '80': 'pending',
                '81': 'done',
                '83': 'cancelled',
                '84': 'review',
                '85': 'deferred',
            },
            None,
        )

        await harness._reconcile_stranded_in_progress()

        # find_merge_marker must never be called (no sweep-eligible tasks)
        harness.git_ops.find_merge_marker.assert_not_called()  # type: ignore[attr-defined]
        # No status changes either
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    # -----------------------------------------------------------------------
    # Done-branch side-effect suite (symmetry + cleanup-failure + INFO log)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        'scenario,is_ancestor_val,marker_sha_val,reason,expected_provenance,cleanup_raises',
        [
            pytest.param(
                'is_ancestor',
                True,
                None,
                'branch-already-on-main',
                {
                    'kind': 'found_on_main',
                    'commit': 'deadbeef' + 'a' * 32,
                    'note': 'reconcile: branch already on main when stranded in-progress',
                },
                False,
                id='is_ancestor-branch-success',
            ),
            pytest.param(
                'marker',
                False,
                'cafebabe' + 'd' * 32,
                'branch-deleted-marker-found',
                {
                    'kind': 'found_on_main',
                    'commit': 'cafebabe' + 'd' * 32,
                    'note': 'reconcile: branch deleted but merge marker found on main',
                },
                False,
                id='marker-branch-success',
            ),
            pytest.param(
                'is_ancestor',
                True,
                None,
                'branch-already-on-main',
                {
                    'kind': 'found_on_main',
                    'commit': 'deadbeef' + 'a' * 32,
                    'note': 'reconcile: branch already on main when stranded in-progress',
                },
                True,
                id='is_ancestor-branch-cleanup-fails',
            ),
            pytest.param(
                'marker',
                False,
                'cafebabe' + 'd' * 32,
                'branch-deleted-marker-found',
                {
                    'kind': 'found_on_main',
                    'commit': 'cafebabe' + 'd' * 32,
                    'note': 'reconcile: branch deleted but merge marker found on main',
                },
                True,
                id='marker-branch-cleanup-fails',
            ),
        ],
    )
    async def test_done_branch_side_effects(
        self,
        harness: Harness,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        scenario: str,
        is_ancestor_val: bool,
        marker_sha_val: str | None,
        reason: str,
        expected_provenance: dict,
        cleanup_raises: bool,
    ):
        """Symmetry regression + cleanup-failure contract for both done-branches.

        cleanup_raises=False: verifies pop / cleanup / set_task_status / worktree
        removal / INFO log are all produced identically for both branches.
        cleanup_raises=True: verifies that a cleanup failure is swallowed and a
        WARNING is emitted — set_task_status still fires for both branches.
        """
        tid = '95'
        harness.git_ops.is_ancestor = AsyncMock(return_value=is_ancestor_val)  # type: ignore[attr-defined]
        harness.git_ops.find_merge_marker = AsyncMock(return_value=marker_sha_val)  # type: ignore[attr-defined]
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]

        # Seed a recovered plan entry — helper must pop it.
        harness._recovered_plans[tid] = {'task_id': tid, 'steps': []}  # type: ignore[attr-defined]

        # Create worktree dir — helper must attempt cleanup.
        worktree_path = harness.git_ops.worktree_base / tid
        worktree_path.mkdir(parents=True)

        if cleanup_raises:
            harness.git_ops.cleanup_worktree = AsyncMock(  # type: ignore[attr-defined]
                side_effect=OSError('boom')
            )

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
            await harness._reconcile_stranded_in_progress()

        # Recovered plan must be gone regardless of cleanup outcome.
        assert tid not in harness._recovered_plans  # type: ignore[attr-defined]

        # set_task_status must always fire with 'done' and the expected provenance.
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done', done_provenance=expected_provenance
        )

        if cleanup_raises:
            # cleanup_worktree must have been called (before the OSError was swallowed).
            harness.git_ops.cleanup_worktree.assert_called_once_with(  # type: ignore[attr-defined]
                worktree_path, tid
            )
            # Exception swallowed; WARNING log must contain tid and reason.
            warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert any(
                tid in r.getMessage() and reason in r.getMessage()
                for r in warning_logs
            ), (
                f'Expected WARNING containing tid={tid!r} and reason={reason!r}; '
                f'got: {[r.getMessage() for r in warning_logs]}'
            )
        else:
            # cleanup_worktree must have been called exactly once.
            harness.git_ops.cleanup_worktree.assert_called_once_with(  # type: ignore[attr-defined]
                worktree_path, tid
            )
            # Worktree dir must have been removed by the cleanup side-effect.
            assert not worktree_path.exists()
            # INFO log must mention tid and reason.
            info_logs = [r for r in caplog.records if r.levelno == logging.INFO]
            assert any(
                tid in r.getMessage() and reason in r.getMessage()
                for r in info_logs
            ), (
                f'Expected INFO containing tid={tid!r} and reason={reason!r}; '
                f'got: {[r.getMessage() for r in info_logs]}'
            )

    @pytest.mark.parametrize(
        'scenario,is_ancestor_val,marker_sha_val,expected_commit',
        [
            pytest.param(
                'is_ancestor',
                True,
                None,
                'deadbeef' + 'a' * 32,
                id='is_ancestor-branch',
            ),
            pytest.param(
                'marker',
                False,
                'cafebabe' + 'd' * 32,
                'cafebabe' + 'd' * 32,
                id='marker-branch',
            ),
        ],
    )
    async def test_absent_worktree_dir_skips_cleanup(
        self,
        harness: Harness,
        scenario: str,
        is_ancestor_val: bool,
        marker_sha_val: str | None,
        expected_commit: str,
    ):
        """When the worktree directory does not exist, cleanup_worktree must
        NOT be called — the existence guard must hold for both done-branches.
        scheduler.mark_done (harness.py:1464) still fires with kind='found_on_main'
        and the expected SHA, and _recovered_plans is popped regardless of
        worktree absence.
        """
        tid = '97'
        harness.git_ops.is_ancestor = AsyncMock(return_value=is_ancestor_val)  # type: ignore[attr-defined]
        harness.git_ops.find_merge_marker = AsyncMock(return_value=marker_sha_val)  # type: ignore[attr-defined]
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]

        # Seed a recovered plan entry — helper must pop it even without a worktree dir.
        harness._recovered_plans[tid] = {'task_id': tid, 'steps': []}  # type: ignore[attr-defined]

        # Deliberately do NOT mkdir — the worktree dir is absent.

        await harness._reconcile_stranded_in_progress()

        # Recovered plan must be gone regardless of worktree absence.
        assert tid not in harness._recovered_plans  # type: ignore[attr-defined]

        # cleanup_worktree must NOT have been called.
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        # Task must still be marked done at the production boundary (harness.py:1464).
        # note=ANY: pinning the literal prose adds no regression-detection value
        # beyond assert_awaited_once + kind + sha.
        harness.scheduler.mark_done.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, kind='found_on_main', sha=expected_commit, note=ANY
        )

    @pytest.mark.parametrize(
        'is_ancestor_val, marker_sha_val',
        [
            pytest.param(True, None, id='is-ancestor-path'),
            pytest.param(False, 'cafebabe' + 'c' * 32, id='merge-marker-path'),
            pytest.param(False, None, id='neither-path'),
        ],
    )
    async def test_get_task_fetched_exactly_once_regardless_of_path(
        self,
        harness: Harness,
        is_ancestor_val: bool,
        marker_sha_val: str | None,
    ):
        """Invariant: scheduler.get_task is awaited exactly once per stranded task
        regardless of which branch wins — is_ancestor fast-path, merge-marker
        fast-path, or neither.  The hoisted fetch at the top of
        _reconcile_one_stranded must never be duplicated by a fast-path.
        """
        harness.scheduler.get_statuses.return_value = ({'90': 'in-progress'}, None)  # type: ignore[attr-defined]
        harness.git_ops.is_ancestor = AsyncMock(return_value=is_ancestor_val)
        harness.git_ops.find_merge_marker = AsyncMock(return_value=marker_sha_val)

        await harness._reconcile_stranded_in_progress()

        assert harness.scheduler.get_task.await_count == 1, (  # type: ignore[attr-defined]
            f'Expected get_task awaited once (path: is_ancestor={is_ancestor_val!r}, '
            f'marker={marker_sha_val!r}); '
            f'got {harness.scheduler.get_task.await_count}'  # type: ignore[attr-defined]
        )
        # For the neither-path case, confirm the lock-state revert still fires.
        if not is_ancestor_val and marker_sha_val is None:
            harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
                '90', 'pending'
            )

    @pytest.mark.parametrize(
        'is_ancestor_val, marker_sha_val, branch_base_sha, expected_commit',
        [
            pytest.param(
                True,
                None,
                'aabbccdd' + 'e' * 32,
                'deadbeef' + 'a' * 32,
                id='is-ancestor-path-with-metadata',
            ),
            pytest.param(
                False,
                'cafebabe' + 'c' * 32,
                'beef0000' + '9' * 32,
                'cafebabe' + 'c' * 32,
                id='merge-marker-path-with-metadata',
            ),
        ],
    )
    async def test_hoisted_metadata_is_consumed_by_each_guard(
        self,
        harness: Harness,
        is_ancestor_val: bool,
        marker_sha_val: str | None,
        branch_base_sha: str,
        expected_commit: str,
    ):
        """Regression lock: the hoisted ``metadata`` dict is CONSUMED by each
        downstream guard that reads ``metadata.get('branch_base_sha')``.

        ``is-ancestor-path-with-metadata``: Guard 3 (harness.py) calls
        ``resolve_branch_sha`` only when ``_is_valid_sha_40(branch_base_sha)``
        is True, proving the guard read the hoisted value.

        ``merge-marker-path-with-metadata``: the stale-marker check calls
        ``is_ancestor(marker_sha, branch_base_sha)`` only when
        ``_is_valid_sha_40(branch_base_sha)`` is True, proving the guard read
        the hoisted value.

        A future refactor that silently drops ``metadata.get('branch_base_sha')``
        from either guard would break the path-specific assertions below —
        making the regression visible even though ``get_task.await_count == 1``
        would still hold in the sibling test.

        Additionally pins the final disposition: both paths must reach
        ``scheduler.mark_done`` (the production call site at harness.py:1464)
        with the path-specific commit SHA, catching veto-inversion refactors
        that would otherwise satisfy the input-args asserts.
        """
        harness.scheduler.get_statuses.return_value = ({'90': 'in-progress'}, None)  # type: ignore[attr-defined]
        # Decouple the two is_ancestor call sites:
        #   - 1st call: is_ancestor(branch, main_branch) — fires on both branches.
        #   - 2nd call: is_ancestor(marker_sha, branch_base_sha) — fires only on
        #     the merge-marker path, and must return False here so the stale-
        #     marker veto does NOT fire (we want pass-through on this test).
        # Element [1]=False is never consumed on the is-ancestor-path branch
        # (only one call happens before _mark_in_progress_done is invoked);
        # it is load-bearing on the merge-marker-path branch.  Do not collapse
        # back to return_value= — that couples two semantically distinct call
        # sites to the same value (reviewer ref: esc-1276-3 #2).
        harness.git_ops.is_ancestor = AsyncMock(side_effect=[is_ancestor_val, False])
        harness.git_ops.find_merge_marker = AsyncMock(return_value=marker_sha_val)
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={'metadata': {'branch_base_sha': branch_base_sha}}
        )
        # Explicitly anchor resolve_branch_sha to a sentinel distinct from
        # branch_base_sha so Guard 3's branch-advanced veto cannot fire
        # regardless of future fixture changes, ensuring the guard passes
        # through to _mark_in_progress_done.
        _BRANCH_TIP = 'c0ffee11' + '0' * 32
        assert branch_base_sha != _BRANCH_TIP, (
            'Test setup error: _BRANCH_TIP collides with branch_base_sha — '
            'update _BRANCH_TIP to a distinct 40-hex value'
        )
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=_BRANCH_TIP)  # type: ignore[attr-defined]
        # Anchor find_task_citation_commit explicitly so the is-ancestor-path
        # expected_commit is self-contained and not silently coupled to the
        # fixture default (test_reconcile_stranded.py:136).  If the fixture
        # default changes, this test's expected behaviour is still clearly
        # described here (reviewer ref: esc-1276-3 amendment #2).
        _CITATION_SHA = 'deadbeef' + 'a' * 32
        harness.git_ops.find_task_citation_commit = AsyncMock(return_value=_CITATION_SHA)

        await harness._reconcile_stranded_in_progress()

        # Pass-through semantics pin: after both consumer guards verify the
        # hoisted metadata was read, the task MUST be marked done via the
        # production boundary at harness.py:1464.  Without this assertion, a
        # refactor that inverted a veto condition (e.g. `not await
        # is_ancestor(...)`) would still call is_ancestor with the correct
        # arguments yet route to veto-and-return None — the existing input-args
        # assertions below would not detect the regression (reviewer ref:
        # esc-1276-3 #1).  note=ANY: the literal note prose adds no regression-
        # detection value beyond the assert-awaited-once + kind + sha checks.
        harness.scheduler.mark_done.assert_awaited_once_with(  # type: ignore[attr-defined]
            '90', kind='found_on_main', sha=expected_commit, note=ANY,
        )

        # Guard 3 consumer assertion: resolve_branch_sha is only called when
        # _is_valid_sha_40(branch_base_sha) is True, proving the hoisted metadata
        # dict was consumed by the is_ancestor fast-path guard.
        if is_ancestor_val:
            harness.git_ops.resolve_branch_sha.assert_awaited_once_with('task/90')  # type: ignore[attr-defined]
        # Stale-marker consumer assertion: the second is_ancestor call (marker_sha,
        # branch_base_sha) only fires when _is_valid_sha_40(branch_base_sha) is True,
        # proving the hoisted metadata dict was consumed by the merge-marker guard.
        elif marker_sha_val is not None:
            assert harness.git_ops.is_ancestor.await_count == 2, (  # type: ignore[attr-defined]
                f'Expected is_ancestor awaited twice (first for branch->main, '
                f'second for marker->base); '
                f'got {harness.git_ops.is_ancestor.await_count}'  # type: ignore[attr-defined]
            )
            second_call_args = harness.git_ops.is_ancestor.call_args_list[1].args  # type: ignore[attr-defined]
            assert second_call_args == (marker_sha_val, branch_base_sha), (
                f'Expected second is_ancestor call args == '
                f'(marker_sha_val={marker_sha_val!r}, branch_base_sha={branch_base_sha!r}); '
                f'got {second_call_args!r}'
            )


# ---------------------------------------------------------------------------
# Harness.run() call-order test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_harness_run_invokes_reconcile_before_scheduler_loop(
    tmp_path: Path,
):
    """run() must call _recover_crashed_tasks → _reconcile_stranded_in_progress
    → scheduler.acquire_next in that order.
    """
    call_order: list[str] = []

    git_cfg = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
    )
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.git = git_cfg
    config.project_root = tmp_path
    config.usage_cap.enabled = False
    config.review.enabled = False
    config.sandbox.backend = 'auto'
    config.max_concurrent_tasks = 2
    config.fused_memory.project_id = 'test'
    config.sandbox.backend = 'auto'
    # Real Path so OverrideStore.from_config(config) can call .parent.mkdir()
    # and sqlite3.connect(str(...)) without leaking MagicMock-named files —
    # Harness.__init__ wires OverrideStore unconditionally (task 1313).
    config.overrides_db_path = tmp_path / 'overrides.db'

    with patch('orchestrator.harness.McpLifecycle') as mock_mcp_cls, \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.OverrideStore'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(config)

    # --- mock infrastructure methods so run() doesn't fail early ---
    h.git_ops = MagicMock()
    h.git_ops.has_dirty_working_tree = AsyncMock(return_value=None)
    h.git_ops.worktree_base = tmp_path / '.worktrees'

    mock_mcp = mock_mcp_cls.return_value
    mock_mcp.start = AsyncMock()
    mock_mcp.stop = AsyncMock()

    h._start_escalation_server = AsyncMock()
    h._start_merge_worker = AsyncMock()
    h._dismiss_stale_escalations = AsyncMock()
    h._start_orphan_l0_reaper = MagicMock()
    h._start_terminal_status_watcher = MagicMock()
    h._start_stranded_reconcile = MagicMock()
    h._tag_task_modules = AsyncMock()

    # Provide one pending task so the "no pending tasks" check passes.
    # get_statuses is used by the startup block (post-step-16);
    # get_tasks is retained for methods not yet migrated (e.g. _tag_prd_metadata).
    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[
        {'id': 1, 'status': 'pending', 'title': 'A task'},
    ])
    h.scheduler.get_statuses = AsyncMock(return_value=({'1': 'pending'}, None))
    h.scheduler.set_task_status = AsyncMock()

    # Track ordering: _recover_crashed_tasks
    async def _fake_recover():
        call_order.append('recover')
    h._recover_crashed_tasks = _fake_recover

    # Track ordering: _reconcile_stranded_in_progress
    async def _fake_reconcile(*, mid_run: bool = False) -> int:
        call_order.append('reconcile')
        return 0
    h._reconcile_stranded_in_progress = _fake_reconcile

    # Track ordering: acquire_next — append then raise to break the loop
    async def _fake_acquire():
        call_order.append('acquire')
        raise RuntimeError('stop the loop')
    h.scheduler.acquire_next = _fake_acquire

    with pytest.raises(RuntimeError, match='stop the loop'):
        await h.run(prd_path=None)

    # _recover_crashed_tasks then _reconcile_stranded_in_progress then acquire_next
    assert 'recover' in call_order, "_recover_crashed_tasks was not called"
    assert 'reconcile' in call_order, "_reconcile_stranded_in_progress was not called"
    assert 'acquire' in call_order, "scheduler.acquire_next was not called"
    recover_idx = call_order.index('recover')
    reconcile_idx = call_order.index('reconcile')
    acquire_idx = call_order.index('acquire')
    assert recover_idx < reconcile_idx, "_recover_crashed_tasks must precede _reconcile_stranded_in_progress"
    assert reconcile_idx < acquire_idx, "_reconcile_stranded_in_progress must precede scheduler.acquire_next"

    # prd_path=None means _tag_prd_metadata is never called, so get_tasks
    # (which _tag_prd_metadata uses for full task data) must not be called.
    # This assertion locks in the migration boundary: all startup-block status
    # checks have moved to get_statuses; get_tasks is only retained for the
    # prd_path code paths that need full task metadata.
    h.scheduler.get_tasks.assert_not_called()


# ---------------------------------------------------------------------------
# Sweep only touches {in-progress, blocked} (regression guard for non-goal)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_terminal_and_pending_statuses_ignored(harness: Harness):
    """The sweep only touches {in-progress, blocked} tasks.

    Blocked tasks are checked (R4: out-of-band-merge recovery) but are only
    flipped to done when on-main evidence is observed; without is_ancestor /
    find_merge_marker hits, they stay blocked. ``done`` / ``cancelled`` /
    ``deferred`` / ``pending`` / ``review`` are never written.
    """
    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {
            '20': 'pending',
            '21': 'done',
            '22': 'blocked',  # checked but no on-main evidence → untouched
            '23': 'cancelled',
            '24': 'review',
            '25': 'in-progress',  # orphan-revert candidate
            '26': 'deferred',
        },
        None,
    )
    # No worktree for task 25 (orphan)

    await harness._reconcile_stranded_in_progress()

    calls = harness.scheduler.set_task_status.call_args_list  # type: ignore[attr-defined]
    assert len(calls) == 1, f"Expected exactly 1 call, got: {calls}"
    assert calls[0].args[0] == '25'
    assert calls[0].args[1] == 'pending'


# ---------------------------------------------------------------------------
# Mid-run filter (Fix 4)
#
# When the sweep runs *during* a live orchestrator run (mid_run=True), tasks
# that the scheduler is actively dispatching (in ``_dispatched``) or holding
# locks for (``lock_table._held``) must be skipped — they are not stranded,
# they're being worked on right now, and reverting their status would race
# the running workflow.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mid_run_skips_dispatched_tasks(harness: Harness):
    """Task in scheduler._dispatched is not stranded — left untouched."""
    # Replace the scheduler MagicMock attrs with real containers so the
    # mid-run guard can inspect membership.
    harness.scheduler._dispatched = {'40'}  # type: ignore[attr-defined]
    harness.scheduler.lock_table = MagicMock()  # type: ignore[attr-defined]
    harness.scheduler.lock_table._held = {}  # type: ignore[attr-defined]

    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'40': 'in-progress'}, None,
    )
    # No worktree for task 40 — would normally trigger no-lock revert.

    changed = await harness._reconcile_stranded_in_progress(mid_run=True)

    assert changed == 0
    harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_mid_run_skips_lock_held_tasks(harness: Harness):
    """Task with active lock_table membership is not stranded — left untouched."""
    harness.scheduler._dispatched = set()  # type: ignore[attr-defined]
    harness.scheduler.lock_table = MagicMock()  # type: ignore[attr-defined]
    harness.scheduler.lock_table._held = {'41': {'mod_a'}}  # type: ignore[attr-defined]

    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'41': 'in-progress'}, None,
    )

    changed = await harness._reconcile_stranded_in_progress(mid_run=True)

    assert changed == 0
    harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_mid_run_reverts_genuine_strand(harness: Harness):
    """Task NOT in _dispatched / _held but in-progress → genuinely stranded."""
    harness.scheduler._dispatched = set()  # type: ignore[attr-defined]
    harness.scheduler.lock_table = MagicMock()  # type: ignore[attr-defined]
    harness.scheduler.lock_table._held = {}  # type: ignore[attr-defined]

    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'42': 'in-progress'}, None,
    )

    changed = await harness._reconcile_stranded_in_progress(mid_run=True)

    assert changed == 1
    harness.scheduler.set_task_status.assert_called_once_with('42', 'pending')  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_returns_change_count(harness: Harness):
    """Reconcile returns int count (revert + marked-done) for main-loop hook."""
    harness.scheduler._dispatched = set()  # type: ignore[attr-defined]
    harness.scheduler.lock_table = MagicMock()  # type: ignore[attr-defined]
    harness.scheduler.lock_table._held = {}  # type: ignore[attr-defined]

    # No in-progress tasks → nothing to do.
    harness.scheduler.get_statuses.return_value = ({'10': 'pending'}, None)  # type: ignore[attr-defined]
    assert await harness._reconcile_stranded_in_progress() == 0

    # One stranded task → returns 1.
    harness.scheduler.get_statuses.return_value = ({'11': 'in-progress'}, None)  # type: ignore[attr-defined]
    assert await harness._reconcile_stranded_in_progress() == 1


# ---------------------------------------------------------------------------
# Per-tid try/except + N-strikes escalation (Stage 1)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_one_task_rejection_does_not_kill_sweep(harness: Harness, caplog):
    """A SetTaskStatusRejected on one task must NOT abort the iteration over others."""
    from orchestrator.scheduler import DoneGateRejection

    harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'100': 'in-progress', '101': 'in-progress'}, None,
    )

    call_count = 0

    async def _flaky_mark_done(tid, *, kind, sha, note=None):
        nonlocal call_count
        call_count += 1
        if tid == '100':
            raise DoneGateRejection(
                task_id=tid, missing_files=['x.py'],
                raw='done_gate_missing_files — metadata.files lists missing paths',
            )

    harness.scheduler.mark_done = AsyncMock(side_effect=_flaky_mark_done)  # type: ignore[attr-defined]

    with caplog.at_level(logging.ERROR, logger='orchestrator.harness'):
        result = await harness._reconcile_stranded_in_progress()

    # Both tasks attempted; only one succeeded.
    assert call_count == 2
    assert result == 1  # only task 101 marked done
    # Failure counter incremented for the rejecting task.
    assert harness._reconcile_failure_counts.get('100') == 1
    assert '101' not in harness._reconcile_failure_counts
    # Honest log mentions the error_code, not "marked done".
    assert any(
        'failed to mark task 100 done' in r.getMessage()
        and 'done_gate_missing_files' in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_n_strikes_escalates_to_l1(harness: Harness):
    """After MAX_RECONCILE_FAILURES rejections, an L1 is filed.

    Subsequent strikes (already-escalated) reset the counter to avoid
    re-escalating on every sweep.
    """
    from orchestrator.harness import MAX_RECONCILE_FAILURES
    from orchestrator.scheduler import DoneGateRejection

    # Inject a stub escalation queue to capture submissions.
    submissions = []

    class _StubEscalationQueue:
        def make_id(self, task_id):
            return f'esc-{task_id}-{len(submissions)}'
        def submit(self, esc):
            submissions.append(esc)
        def has_open_l1(self, task_id):  # noqa: ARG002
            return False

    harness._escalation_queue = _StubEscalationQueue()  # type: ignore[assignment]

    harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'200': 'in-progress'}, None,
    )

    async def _always_reject(tid, *, kind, sha, note=None):
        raise DoneGateRejection(
            task_id=tid, missing_files=['x.py'],
            raw='done_gate_missing_files — metadata.files lists missing paths',
        )

    harness.scheduler.mark_done = AsyncMock(side_effect=_always_reject)  # type: ignore[attr-defined]

    # Run MAX_RECONCILE_FAILURES sweeps — escalation fires on the Nth.
    for _ in range(MAX_RECONCILE_FAILURES):
        await harness._reconcile_stranded_in_progress()

    assert len(submissions) == 1, (
        f'expected exactly one L1 submission after {MAX_RECONCILE_FAILURES} '
        f'consecutive rejections, got {len(submissions)}'
    )
    esc = submissions[0]
    assert esc.task_id == '200'
    assert esc.severity == 'blocking'
    assert esc.category == 'reconcile_persistent_rejection'
    # Counter reset after escalation so the next sweep starts a fresh count.
    assert '200' not in harness._reconcile_failure_counts


@pytest.mark.asyncio
async def test_reconcile_persistent_citation_miss_escalates_l1(harness: Harness):
    """N consecutive Guard 2 skips file exactly one L1 escalation, and the
    counter resets on a successful sweep.

    Locks two contracts:
      1. ``find_task_citation_commit`` returning None for
         ``MAX_RECONCILE_FAILURES`` sweeps in a row triggers one (and only
         one) L1 escalation with category ``reconcile_citation_missing``.
      2. A subsequent sweep where the citation IS found clears the per-tid
         skip counter so a future drought starts a fresh count.
    """
    from orchestrator.harness import MAX_RECONCILE_FAILURES

    submissions = []

    class _StubEscalationQueue:
        def make_id(self, task_id):
            return f'esc-{task_id}-{len(submissions)}'

        def submit(self, esc):
            submissions.append(esc)

        def has_open_l1(self, task_id):  # noqa: ARG002
            return False

    harness._escalation_queue = _StubEscalationQueue()  # type: ignore[assignment]
    harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    harness.git_ops.find_task_citation_commit = AsyncMock(  # type: ignore[attr-defined]
        return_value=None,
    )
    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'400': 'blocked'}, None,
    )

    for _ in range(MAX_RECONCILE_FAILURES):
        await harness._reconcile_stranded_in_progress()

    # Exactly one L1 was filed, with the expected category and severity.
    assert len(submissions) == 1, (
        f'expected exactly one L1 submission after {MAX_RECONCILE_FAILURES} '
        f'consecutive citation misses, got {len(submissions)}'
    )
    esc = submissions[0]
    assert esc.task_id == '400'
    assert esc.severity == 'blocking'
    assert esc.category == 'reconcile_citation_missing'
    assert esc.level == 1
    # Counter reset after escalation so the next sweep starts fresh.
    assert '400' not in harness._reconcile_skip_counts

    # And mark_done was never invoked: the skip path bails before the flip.
    harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]

    # Now the citation IS found on the next sweep — counter resets again
    # via the explicit pop on the success path (already done by escalation),
    # but verify a successful sweep starts and ends with no counter.
    harness.git_ops.find_task_citation_commit = AsyncMock(  # type: ignore[attr-defined]
        return_value='deadbeef' + 'a' * 32,
    )
    await harness._reconcile_stranded_in_progress()
    assert '400' not in harness._reconcile_skip_counts


@pytest.mark.asyncio
async def test_failure_counter_resets_on_success(harness: Harness):
    """A successful mark_done clears the per-tid failure counter."""
    from orchestrator.scheduler import DoneGateRejection

    harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'300': 'in-progress'}, None,
    )

    # First sweep: reject.
    async def _reject(tid, *, kind, sha, note=None):
        raise DoneGateRejection(
            task_id=tid, missing_files=['x.py'],
            raw='done_gate_missing_files',
        )
    harness.scheduler.mark_done = AsyncMock(side_effect=_reject)  # type: ignore[attr-defined]
    await harness._reconcile_stranded_in_progress()
    assert harness._reconcile_failure_counts.get('300') == 1

    # Second sweep: succeed.
    async def _succeed(tid, *, kind, sha, note=None):
        await harness.scheduler.set_task_status(
            tid, 'done', done_provenance={'kind': kind, 'commit': sha},
        )
    harness.scheduler.mark_done = AsyncMock(side_effect=_succeed)  # type: ignore[attr-defined]
    await harness._reconcile_stranded_in_progress()
    assert '300' not in harness._reconcile_failure_counts


# ---------------------------------------------------------------------------
# R3: mid_run alive owner_pid (this run's harness PID) → fall through (Stage 3)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mid_run_alive_owner_pid_not_in_dispatch_recovers(
    harness: Harness, tmp_path: Path,
):
    """R3 fix: in mid_run sweep, an alive owner_pid that isn't in the
    dispatch table represents a workflow that exited without releasing the
    lock — the sweep must fall through to recovery rather than skip.

    Pre-fix: ``if owner_alive: continue`` skipped these tasks forever
    because harness.pid (this PID) is alive throughout the run.
    """
    harness.scheduler._dispatched = set()  # type: ignore[attr-defined]
    harness.scheduler.lock_table = MagicMock()  # type: ignore[attr-defined]
    harness.scheduler.lock_table._held = {}  # type: ignore[attr-defined]

    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'400': 'in-progress'}, None,
    )

    # Create a worktree with plan.lock pointing to OUR pid (harness.pid).
    lock_dir = harness.git_ops.worktree_base / '400' / '.task'
    lock_dir.mkdir(parents=True)
    (lock_dir / 'plan.lock').write_text(json.dumps({
        'session_id': '400-x', 'locked_at': datetime.now(UTC).isoformat(),
        'owner_pid': os.getpid(),
    }))

    changed = await harness._reconcile_stranded_in_progress(mid_run=True)

    # Stage 3: must fall through to revert (was skipped pre-fix).
    assert changed == 1
    harness.scheduler.set_task_status.assert_called_once_with('400', 'pending')  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_startup_alive_owner_pid_left_alone(harness: Harness):
    """At startup (mid_run=False), an alive owner_pid still skips recovery.

    R3 only changes mid_run behaviour — the startup path retains the
    historical skip-on-alive contract.
    """
    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'401': 'in-progress'}, None,
    )

    lock_dir = harness.git_ops.worktree_base / '401' / '.task'
    lock_dir.mkdir(parents=True)
    (lock_dir / 'plan.lock').write_text(json.dumps({
        'session_id': '401-x', 'locked_at': datetime.now(UTC).isoformat(),
        'owner_pid': os.getpid(),
    }))

    changed = await harness._reconcile_stranded_in_progress(mid_run=False)

    assert changed == 0
    harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_mid_run_cancel_window_grace(harness: Harness, tmp_path: Path):
    """R3 race guard: a workflow whose cancel_event was set within the grace
    window is skipped — its finally: block may still be writing state.

    Outside the window the sweep proceeds.
    """
    harness.scheduler._dispatched = set()  # type: ignore[attr-defined]
    harness.scheduler.lock_table = MagicMock()  # type: ignore[attr-defined]
    harness.scheduler.lock_table._held = {}  # type: ignore[attr-defined]

    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'500': 'in-progress'}, None,
    )

    # Stamp cancel-time NOW — within grace window → skip.
    import time as _time
    harness._workflow_cancel_at['500'] = _time.monotonic()

    changed = await harness._reconcile_stranded_in_progress(mid_run=True)
    assert changed == 0
    harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    # Stamp cancel-time well in the past → proceed.
    harness._workflow_cancel_at['500'] = (
        _time.monotonic() - harness._RECONCILE_CANCEL_GRACE_S - 1
    )
    changed = await harness._reconcile_stranded_in_progress(mid_run=True)
    # Task has no worktree → orphan revert.
    assert changed == 1
    harness.scheduler.set_task_status.assert_called_once_with('500', 'pending')  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# R4: blocked-task pass (Stage 4)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_blocked_with_branch_on_main_marked_done(harness: Harness):
    """Blocked task whose branch is already on main → marked done.

    Out-of-band-merged-while-blocked recovery: a human merged the branch
    manually, leaving the row in 'blocked'. Next sweep should mark it done.
    """
    harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'600': 'blocked'}, None,
    )

    changed = await harness._reconcile_stranded_in_progress()

    assert changed == 1
    # Provenance note distinguishes blocked-merge from in-progress-merge.
    harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
        '600', 'done',
        done_provenance={
            'kind': 'found_on_main',
            'commit': 'deadbeef' + 'a' * 32,
            'note': 'reconcile: branch on main while task was blocked (out-of-band merge)',
        },
    )


@pytest.mark.asyncio
async def test_blocked_without_on_main_evidence_left_alone(harness: Harness):
    """Blocked task with no on-main evidence → untouched.

    'blocked' is a deliberate state; we only flip it on observed evidence.
    """
    # is_ancestor=False, find_merge_marker=None (defaults from fixture)
    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'601': 'blocked'}, None,
    )

    changed = await harness._reconcile_stranded_in_progress()

    assert changed == 0
    harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_cancelled_and_deferred_never_swept(harness: Harness):
    """'cancelled' (terminal-by-decision) and 'deferred' (human-deferred) are
    never touched by the sweep, even when their branch is on main."""
    harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'700': 'cancelled', '701': 'deferred'}, None,
    )

    changed = await harness._reconcile_stranded_in_progress()

    assert changed == 0
    harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
    harness.git_ops.is_ancestor.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_blocked_with_merge_marker_marked_done(harness: Harness):
    """Blocked task whose branch was deleted but a merge marker is on main."""
    marker_sha = 'cafe' + 'b' * 36
    harness.git_ops.find_merge_marker = AsyncMock(return_value=marker_sha)  # type: ignore[attr-defined]
    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'602': 'blocked'}, None,
    )

    changed = await harness._reconcile_stranded_in_progress()

    assert changed == 1
    harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
        '602', 'done',
        done_provenance={
            'kind': 'found_on_main',
            'commit': marker_sha,
            'note': 'reconcile: merge marker found on main while task was blocked',
        },
    )
