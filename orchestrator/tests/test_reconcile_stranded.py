"""Tests for Harness._reconcile_stranded_in_progress and the _pid_alive helper."""

import json
import logging
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig
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

        # set_task_status must be called exactly once: ('50', 'done') with commit + note
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '50', 'done',
            done_provenance={
                'commit': 'deadbeef' + 'a' * 32,
                'note': 'reconcile: branch already on main when stranded in-progress',
            },
        )
        harness.git_ops.resolve_branch_sha.assert_awaited_once_with('task/50')  # type: ignore[attr-defined]

        # cleanup_worktree must NOT have been called
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

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

        # (3) Task must be marked done with the expected provenance (commit + note)
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done',
            done_provenance={
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

    async def test_already_merged_provenance_omits_commit_when_branch_unresolved(
        self, harness: Harness, caplog
    ):
        """Fallback: when resolve_branch_sha returns None (branch ref vanished after
        is_ancestor check), done_provenance has only 'note' — no 'commit' key —
        and a WARNING log is emitted containing the branch name and 'rev-parse'.
        """
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=None)  # type: ignore[attr-defined]
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {'53': 'in-progress'}, None
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await harness._reconcile_stranded_in_progress()

        # Note-only provenance — NO 'commit' key
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '53', 'done',
            done_provenance={
                'note': 'reconcile: branch already on main when stranded in-progress',
            },
        )

        # WARNING log must mention the branch name and 'rev-parse'
        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any(
            'task/53' in msg and 'rev-parse' in msg
            for msg in warning_messages
        ), f'Expected WARNING with task/53 and rev-parse, got: {warning_messages}'

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

        # Task must be marked done with commit + note provenance
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done',
            done_provenance={
                'note': 'reconcile: branch deleted but merge marker found on main',
                'commit': marker_sha,
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
                'note': 'reconcile: branch deleted but merge marker found on main',
                'commit': marker_sha,
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
                'note': 'reconcile: branch deleted but merge marker found on main',
                'commit': marker_sha,
            },
        )

        # (4) Worktree dir is gone — cleanup_worktree's rmtree side_effect ran
        assert not worktree_path.exists(), (
            'worktree dir must be removed by cleanup_worktree'
        )

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

    async def test_is_ancestor_not_invoked_for_non_in_progress_tasks(
        self, harness: Harness
    ):
        """Placement-efficiency regression lock: is_ancestor is never called
        when there are no in-progress tasks.

        Proves the guard sits below the `if status != 'in-progress': continue`
        filter and does not waste git invocations on non-in-progress tasks.
        """
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {
                '60': 'pending',
                '61': 'done',
                '62': 'blocked',
                '63': 'cancelled',
                '64': 'review',
            },
            None,
        )

        await harness._reconcile_stranded_in_progress()

        # is_ancestor must never be called (no in-progress tasks)
        harness.git_ops.is_ancestor.assert_not_called()  # type: ignore[attr-defined]
        # No status changes either
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    async def test_find_merge_marker_not_invoked_for_non_in_progress_tasks(
        self, harness: Harness
    ):
        """Placement-efficiency regression lock: find_merge_marker is never
        called when there are no in-progress tasks.

        Proves the guard sits below the `if status != 'in-progress': continue`
        filter and does not waste git invocations on non-in-progress tasks.
        """
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {
                '80': 'pending',
                '81': 'done',
                '82': 'blocked',
                '83': 'cancelled',
                '84': 'review',
            },
            None,
        )

        await harness._reconcile_stranded_in_progress()

        # find_merge_marker must never be called (no in-progress tasks)
        harness.git_ops.find_merge_marker.assert_not_called()  # type: ignore[attr-defined]
        # No status changes either
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    # -----------------------------------------------------------------------
    # Spy test: both done-branches must delegate to _mark_in_progress_done
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        'scenario,is_ancestor,marker_sha,expected_reason,expected_provenance',
        [
            pytest.param(
                'is_ancestor', True, None, 'branch-already-on-main',
                {
                    'note': 'reconcile: branch already on main when stranded in-progress',
                    'commit': 'deadbeef' + 'a' * 32,
                },
                id='is_ancestor-branch',
            ),
            pytest.param(
                'marker', False, 'cafebabe' + 'd' * 32, 'branch-deleted-marker-found',
                {
                    'note': 'reconcile: branch deleted but merge marker found on main',
                    'commit': 'cafebabe' + 'd' * 32,
                },
                id='marker-branch',
            ),
        ],
    )
    async def test_both_done_branches_invoke_mark_in_progress_done_helper(
        self,
        harness: Harness,
        scenario: str,
        is_ancestor: bool,
        marker_sha: str | None,
        expected_reason: str,
        expected_provenance: dict,
    ):
        """Spy test: _reconcile_stranded_in_progress delegates to
        _mark_in_progress_done from BOTH the is_ancestor and the
        find_merge_marker branches.

        Initially RED because the inline code does not call the helper yet.
        """
        harness.git_ops.is_ancestor = AsyncMock(return_value=is_ancestor)  # type: ignore[attr-defined]
        harness.git_ops.find_merge_marker = AsyncMock(return_value=marker_sha)  # type: ignore[attr-defined]
        harness.scheduler.get_statuses.return_value = ({'90': 'in-progress'}, None)  # type: ignore[attr-defined]

        spy = AsyncMock(wraps=harness._mark_in_progress_done)  # type: ignore[attr-defined]
        harness._mark_in_progress_done = spy  # type: ignore[attr-defined]

        await harness._reconcile_stranded_in_progress()

        spy.assert_awaited_once_with('90', expected_provenance, expected_reason)


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
    config = MagicMock()
    config.git = git_cfg
    config.project_root = tmp_path
    config.usage_cap.enabled = False
    config.review.enabled = False
    config.sandbox.backend = 'auto'
    config.max_concurrent_tasks = 2
    config.fused_memory.project_id = 'test'
    config.sandbox.backend = 'auto'

    with patch('orchestrator.harness.McpLifecycle') as mock_mcp_cls, \
         patch('orchestrator.harness.Scheduler'), \
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
    async def _fake_reconcile():
        call_order.append('reconcile')
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
# Non-in-progress statuses are ignored (regression guard for non-goal)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_in_progress_statuses_ignored(harness: Harness):
    """The sweep only touches in-progress tasks; other statuses are untouched."""
    # Only the 'in-progress' task has no lock (worktree doesn't exist)
    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {
            '20': 'pending',
            '21': 'done',
            '22': 'blocked',
            '23': 'cancelled',
            '24': 'review',
            '25': 'in-progress',  # <-- only this one
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
# _mark_in_progress_done helper unit tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestMarkInProgressDoneHelper:
    """Unit tests for Harness._mark_in_progress_done.

    All cases start RED (AttributeError) until step-2 adds the helper.
    """

    async def test_pops_recovered_plans_entry(self, harness: Harness, tmp_path: Path):
        """Helper pops harness._recovered_plans[tid] even when key exists."""
        tid = '77'
        harness._recovered_plans[tid] = {'task_id': tid, 'steps': []}  # type: ignore[attr-defined]
        provenance = {'note': 'test', 'commit': 'abc123'}

        await harness._mark_in_progress_done(tid, provenance, 'branch-already-on-main')  # type: ignore[attr-defined]

        assert tid not in harness._recovered_plans  # type: ignore[attr-defined]

    async def test_calls_cleanup_worktree_when_worktree_dir_exists(
        self, harness: Harness
    ):
        """Helper calls cleanup_worktree(worktree_path, tid) when the dir exists."""
        tid = '78'
        worktree_path = harness.git_ops.worktree_base / tid
        worktree_path.mkdir(parents=True)
        provenance = {'note': 'test'}

        await harness._mark_in_progress_done(tid, provenance, 'branch-already-on-main')  # type: ignore[attr-defined]

        harness.git_ops.cleanup_worktree.assert_called_once_with(  # type: ignore[attr-defined]
            worktree_path, tid
        )

    async def test_skips_cleanup_worktree_when_worktree_dir_absent(
        self, harness: Harness
    ):
        """Helper does NOT call cleanup_worktree when the worktree dir is absent."""
        tid = '79'
        # No worktree dir created
        provenance = {'note': 'test'}

        await harness._mark_in_progress_done(tid, provenance, 'branch-already-on-main')  # type: ignore[attr-defined]

        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.parametrize('reason', [
        'branch-already-on-main',
        'branch-deleted-marker-found',
    ])
    async def test_swallows_cleanup_worktree_exception_and_logs_warning_with_reason(
        self,
        harness: Harness,
        caplog: pytest.LogCaptureFixture,
        reason: str,
    ):
        """Helper swallows cleanup_worktree errors, logs WARNING with reason,
        and still calls set_task_status('done', ...)."""
        tid = '80'
        worktree_path = harness.git_ops.worktree_base / tid
        worktree_path.mkdir(parents=True)
        harness.git_ops.cleanup_worktree = AsyncMock(  # type: ignore[attr-defined]
            side_effect=OSError('boom')
        )
        provenance = {'note': 'test', 'commit': 'deadbeef'}

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await harness._mark_in_progress_done(tid, provenance, reason)  # type: ignore[attr-defined]

        # Must NOT raise, AND set_task_status must still be called
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done', done_provenance=provenance
        )
        # WARNING log must mention both tid and reason
        warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            tid in r.getMessage() and reason in r.getMessage()
            for r in warning_logs
        ), f"Expected WARNING containing tid={tid!r} and reason={reason!r}; got: {[r.getMessage() for r in warning_logs]}"

    async def test_calls_set_task_status_with_provenance(self, harness: Harness):
        """Helper calls set_task_status(tid, 'done', done_provenance=provenance)."""
        tid = '81'
        provenance = {'note': 'reconcile: branch already on main when stranded in-progress', 'commit': 'abc123def456'}

        await harness._mark_in_progress_done(tid, provenance, 'branch-already-on-main')  # type: ignore[attr-defined]

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done', done_provenance=provenance
        )

    async def test_emits_info_log_with_reason(
        self,
        harness: Harness,
        caplog: pytest.LogCaptureFixture,
    ):
        """Helper emits INFO log matching 'Reconcile: marked task <tid> done (reason=<reason>)'."""
        tid = '82'
        reason = 'branch-already-on-main'
        provenance = {'note': 'test'}

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
            await harness._mark_in_progress_done(tid, provenance, reason)  # type: ignore[attr-defined]

        info_logs = [r for r in caplog.records if r.levelno == logging.INFO]
        pattern = rf'Reconcile: marked task {tid} done \(reason={reason}\)'
        assert any(
            re.search(pattern, r.getMessage()) for r in info_logs
        ), f"Expected INFO log matching {pattern!r}; got: {[r.getMessage() for r in info_logs]}"
