"""Tests for the per-task live-workflow detector (fused_memory.services.live_workflow_detector).

Step-1 covers the two git-derived signals (worktree_registered, recent_commit) in isolation.
Step-3 adds the orchestrator_live signal, OR-aggregation, and the convenience wrapper.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

# RED in step-1: module does not exist yet; import will fail until step-2.
import fused_memory.services.live_workflow_detector as detector_module
from fused_memory.services.live_workflow_detector import (
    DEFAULT_HEARTBEAT_TTL,
    WorkflowLiveness,
    _routing_decided_after_restart,
    _task_in_scheduler_holders_or_parks,
    corroboration_for_task,
    detect_live_workflow,
    has_live_workflow_corroboration,
    is_pure_gate_metadata,
    is_workflow_live_for_task,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TASK_ID = '4321'
_BRANCH = f'task/{_TASK_ID}'


def _worktree_porcelain_with_branch(branch: str, path: str = '/tmp/fake') -> str:
    """Return minimal `git worktree list --porcelain` output that includes *branch*."""
    return (
        f'worktree {path}\n'
        f'HEAD abc1234\n'
        f'branch refs/heads/{branch}\n'
        '\n'
    )


def _worktree_porcelain_no_branch() -> str:
    """Return `git worktree list --porcelain` output with NO task branch."""
    return (
        'worktree /home/leo/src/dark-factory\n'
        'HEAD abc1234\n'
        'branch refs/heads/main\n'
        '\n'
    )


def _worktree_porcelain_prunable(branch: str, path: str = '/tmp/gone') -> str:
    """Return `git worktree list --porcelain` output for a REAPED worktree.

    Its directory was removed but git still carries a stale registration for
    *branch*, marked `prunable` — reify#5245's shape.
    """
    return (
        f'worktree {path}\n'
        f'HEAD abc1234\n'
        f'branch refs/heads/{branch}\n'
        'prunable gitdir file points to non-existent location\n'
        '\n'
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_orchestrator_live_default(monkeypatch):
    """Isolate git signals by making orchestrator always report not-live by default.

    Monkeypatches the name as imported *inside* the detector module so the
    OR-aggregation tests in step-3 can flip it independently per-test.
    """
    monkeypatch.setattr(detector_module, 'is_orchestrator_live_for', lambda _pr: False)


# ---------------------------------------------------------------------------
# Signal (a): worktree_registered
# ---------------------------------------------------------------------------

class TestWorktreeSignal:
    """detect_live_workflow.worktree_registered reflects git worktree list."""

    def _make_run(self, stdout: str, returncode: int = 0):
        """Build a mock subprocess.CompletedProcess for worktree list."""
        result = subprocess.CompletedProcess(
            args=['git', 'worktree', 'list', '--porcelain'],
            returncode=returncode,
            stdout=stdout,
            stderr='',
        )
        return result

    def _make_log_run(self, stdout: str = '', returncode: int = 1):
        """Build a mock subprocess.CompletedProcess for git log (branch absent by default)."""
        return subprocess.CompletedProcess(
            args=['git', 'log', '-1', '--format=%cI', _BRANCH],
            returncode=returncode,
            stdout=stdout,
            stderr='',
        )

    def _run_side_effect(self, worktree_stdout: str, log_stdout: str = '', log_rc: int = 1):
        """Return a side_effect callable for subprocess.run."""
        worktree_result = self._make_run(worktree_stdout)
        log_result = self._make_log_run(log_stdout, log_rc)

        def side_effect(args, **kwargs):
            if '--porcelain' in args:
                return worktree_result
            # git log call
            return log_result

        return side_effect

    def test_worktree_registered_true_when_branch_present(self, tmp_path):
        """worktree_registered=True and is_live=True when branch task/<id> is listed."""
        side_effect = self._run_side_effect(
            _worktree_porcelain_with_branch(_BRANCH)
        )
        with patch('subprocess.run', side_effect=side_effect):
            result = detect_live_workflow(_TASK_ID, str(tmp_path))

        assert isinstance(result, WorkflowLiveness)
        assert result.worktree_registered is True
        assert result.is_live is True
        assert result.branch == _BRANCH

    def test_worktree_registered_false_when_branch_absent(self, tmp_path):
        """worktree_registered=False when no task/<id> branch in worktree list."""
        side_effect = self._run_side_effect(
            _worktree_porcelain_no_branch()
        )
        with patch('subprocess.run', side_effect=side_effect):
            result = detect_live_workflow(_TASK_ID, str(tmp_path))

        assert result.worktree_registered is False

    def test_worktree_signal_false_on_subprocess_error(self, tmp_path):
        """subprocess.run raising does not propagate — worktree_registered=False (fail-safe)."""
        def raise_oserror(args, **kwargs):
            raise OSError('git not found')

        with patch('subprocess.run', side_effect=raise_oserror):
            result = detect_live_workflow(_TASK_ID, str(tmp_path))

        assert result.worktree_registered is False
        assert result.recent_commit is False
        # is_live False because all three signals are suppressed
        assert result.is_live is False

    def test_worktree_signal_false_on_non_zero_returncode(self, tmp_path):
        """A non-zero returncode from worktree list => worktree_registered=False (fail-safe)."""
        # Override: make worktree list return non-zero
        def failing_side_effect(args, **kwargs):
            if '--porcelain' in args:
                return subprocess.CompletedProcess(args=args, returncode=1, stdout='', stderr='')
            return subprocess.CompletedProcess(args=args, returncode=1, stdout='', stderr='')

        with patch('subprocess.run', side_effect=failing_side_effect):
            result = detect_live_workflow(_TASK_ID, str(tmp_path))

        assert result.worktree_registered is False


# ---------------------------------------------------------------------------
# Signal (b): recent_commit
# ---------------------------------------------------------------------------

class TestRecentCommitSignal:
    """detect_live_workflow.recent_commit reflects git log on task/<id>."""

    _NOW = datetime(2026, 6, 5, 12, 0, 0, tzinfo=UTC)

    def _run_side_effect(self, commit_ts_str: str | None, log_rc: int = 0):
        """Return subprocess.run side_effect with canned git log output."""
        def side_effect(args, **kwargs):
            if '--porcelain' in args:
                # No worktree registered
                return subprocess.CompletedProcess(
                    args=args, returncode=0,
                    stdout=_worktree_porcelain_no_branch(), stderr=''
                )
            # git log call
            stdout = commit_ts_str or ''
            return subprocess.CompletedProcess(
                args=args, returncode=log_rc, stdout=stdout, stderr=''
            )
        return side_effect

    def test_recent_commit_true_when_commit_within_threshold(self, tmp_path):
        """recent_commit=True and is_live=True when tip commit is within max_commit_age_hours."""
        # Commit 1 hour ago — well within default threshold (6h)
        ts = (self._NOW - timedelta(hours=1)).isoformat()
        with patch('subprocess.run', side_effect=self._run_side_effect(ts)):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), now=self._NOW, max_commit_age_hours=6.0
            )

        assert result.recent_commit is True
        assert result.is_live is True
        assert result.last_commit_at is not None

    def test_recent_commit_false_when_commit_older_than_threshold(self, tmp_path):
        """recent_commit=False when tip commit is older than max_commit_age_hours."""
        # Commit 10 hours ago — outside the 6h threshold
        ts = (self._NOW - timedelta(hours=10)).isoformat()
        with patch('subprocess.run', side_effect=self._run_side_effect(ts)):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), now=self._NOW, max_commit_age_hours=6.0
            )

        assert result.recent_commit is False

    def test_recent_commit_false_when_branch_missing(self, tmp_path):
        """recent_commit=False when git log returns non-zero (branch does not exist)."""
        with patch('subprocess.run', side_effect=self._run_side_effect(None, log_rc=1)):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), now=self._NOW
            )

        assert result.recent_commit is False

    def test_recent_commit_false_on_empty_log_output(self, tmp_path):
        """recent_commit=False when git log returns empty string."""
        with patch('subprocess.run', side_effect=self._run_side_effect('', log_rc=0)):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), now=self._NOW
            )

        assert result.recent_commit is False

    def test_recent_commit_false_on_parse_error(self, tmp_path):
        """recent_commit=False when git log returns an unparseable timestamp."""
        with patch('subprocess.run', side_effect=self._run_side_effect('not-a-timestamp')):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), now=self._NOW
            )

        assert result.recent_commit is False

    def test_recent_commit_false_on_subprocess_exception(self, tmp_path):
        """recent_commit=False when subprocess raises during git log (fail-safe)."""
        call_count = [0]

        def side_effect(args, **kwargs):
            call_count[0] += 1
            if '--porcelain' in args:
                return subprocess.CompletedProcess(
                    args=args, returncode=0,
                    stdout=_worktree_porcelain_no_branch(), stderr=''
                )
            # git log call
            raise subprocess.TimeoutExpired(cmd=args, timeout=10)

        with patch('subprocess.run', side_effect=side_effect):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), now=self._NOW
            )

        assert result.recent_commit is False
        # No exception propagated


# ---------------------------------------------------------------------------
# Step-3: orchestrator_live signal and OR-aggregation
# ---------------------------------------------------------------------------


class TestOrchestratorLiveSignal:
    """detect_live_workflow.orchestrator_live is sourced from is_orchestrator_live_for."""

    def _all_git_signals_false(self) -> object:
        """subprocess.run side_effect that makes both git signals False."""
        def side_effect(args, **kwargs):
            return subprocess.CompletedProcess(
                args=args, returncode=0,
                stdout=_worktree_porcelain_no_branch(), stderr=''
            )
        return side_effect

    def test_orchestrator_live_true_when_is_orchestrator_live_for_returns_true(
        self, tmp_path, monkeypatch
    ):
        """orchestrator_live=True and is_live=True when is_orchestrator_live_for returns True.

        Both git signals are False so is_live being True demonstrates the orchestrator
        signal is the sole contributor.
        """
        # Override the default autouse=False patch (both git signals stay False via side_effect)
        monkeypatch.setattr(detector_module, 'is_orchestrator_live_for', lambda _pr: True)

        with patch('subprocess.run', side_effect=self._all_git_signals_false()):
            result = detect_live_workflow(_TASK_ID, str(tmp_path))

        assert result.orchestrator_live is True
        assert result.worktree_registered is False
        assert result.recent_commit is False
        assert result.is_live is True

    def test_orchestrator_live_false_when_is_orchestrator_live_for_returns_false(
        self, tmp_path
    ):
        """orchestrator_live=False (default autouse patch) and git signals False => is_live=False."""
        with patch('subprocess.run', side_effect=self._all_git_signals_false()):
            result = detect_live_workflow(_TASK_ID, str(tmp_path))

        assert result.orchestrator_live is False
        assert result.is_live is False

    def test_is_orchestrator_live_for_is_called_with_project_root(
        self, tmp_path, monkeypatch
    ):
        """is_orchestrator_live_for must be called with project_root as its argument."""
        captured_roots: list[str | object] = []

        def capturing_detector(pr):
            captured_roots.append(pr)
            return False

        monkeypatch.setattr(detector_module, 'is_orchestrator_live_for', capturing_detector)

        project_root = str(tmp_path)
        with patch('subprocess.run', side_effect=self._all_git_signals_false()):
            detect_live_workflow(_TASK_ID, project_root)

        assert len(captured_roots) == 1
        assert str(captured_roots[0]) == project_root


class TestAllSignalsFalse:
    """When all three signals are False, is_live is False — the genuine stranded case."""

    def test_all_false_yields_not_live(self, tmp_path):
        """is_live=False with all three signals False — genuine stranded work must escalate."""
        def all_signals_false(args, **kwargs):
            return subprocess.CompletedProcess(
                args=args, returncode=0,
                stdout=_worktree_porcelain_no_branch(), stderr=''
            )

        with patch('subprocess.run', side_effect=all_signals_false):
            result = detect_live_workflow(_TASK_ID, str(tmp_path))

        assert result.worktree_registered is False
        assert result.recent_commit is False
        assert result.orchestrator_live is False
        assert result.is_live is False


class TestConvenienceWrapper:
    """is_workflow_live_for_task returns the same boolean as detect_live_workflow.is_live."""

    def test_wrapper_returns_true_when_live(self, tmp_path, monkeypatch):
        """is_workflow_live_for_task returns True when is_live is True."""
        monkeypatch.setattr(detector_module, 'is_orchestrator_live_for', lambda _pr: True)

        def all_git_false(args, **kwargs):
            return subprocess.CompletedProcess(
                args=args, returncode=0,
                stdout=_worktree_porcelain_no_branch(), stderr=''
            )

        with patch('subprocess.run', side_effect=all_git_false):
            live = is_workflow_live_for_task(_TASK_ID, str(tmp_path))
            expected = detect_live_workflow(_TASK_ID, str(tmp_path)).is_live

        assert live is True
        assert live == expected

    def test_wrapper_returns_false_when_not_live(self, tmp_path):
        """is_workflow_live_for_task returns False when all signals are False."""
        def all_git_false(args, **kwargs):
            return subprocess.CompletedProcess(
                args=args, returncode=0,
                stdout=_worktree_porcelain_no_branch(), stderr=''
            )

        with patch('subprocess.run', side_effect=all_git_false):
            live = is_workflow_live_for_task(_TASK_ID, str(tmp_path))

        assert live is False

    def test_wrapper_accepts_same_kwargs_as_detect(self, tmp_path):
        """is_workflow_live_for_task forwards kwargs (e.g. now, max_commit_age_hours)."""
        now = datetime(2026, 6, 5, 12, 0, 0, tzinfo=UTC)
        # Recent commit (1h ago)
        ts = (now - timedelta(hours=1)).isoformat()

        def side_effect(args, **kwargs):
            if '--porcelain' in args:
                return subprocess.CompletedProcess(
                    args=args, returncode=0,
                    stdout=_worktree_porcelain_no_branch(), stderr=''
                )
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout=ts, stderr=''
            )

        with patch('subprocess.run', side_effect=side_effect):
            live = is_workflow_live_for_task(
                _TASK_ID, str(tmp_path), now=now, max_commit_age_hours=6.0
            )

        assert live is True


# ---------------------------------------------------------------------------
# Status-scoped orchestrator_live signal
# ---------------------------------------------------------------------------


class TestStatusScopedOrchestratorSignal:
    """detect_live_workflow(status=...) suppresses the project-wide orchestrator_live
    signal for statuses that are never actively dispatched (deferred/done/cancelled),
    while the per-task worktree_registered/recent_commit signals remain unaffected.
    """

    def _no_worktree_no_commit_side_effect(self):
        """subprocess.run side_effect driving both git signals False."""
        def side_effect(args, **kwargs):
            return subprocess.CompletedProcess(
                args=args, returncode=0,
                stdout=_worktree_porcelain_no_branch(), stderr=''
            )
        return side_effect

    @pytest.mark.parametrize('status', ['deferred', 'done', 'cancelled'])
    def test_ineligible_status_suppresses_orchestrator_signal(self, tmp_path, status):
        """Ineligible statuses force orchestrator_live False even when _orchestrator_live=True.

        Both git signals are False, so is_live=False proves the project-wide
        orchestrator lock is not being consulted at all for these statuses.
        """
        with patch('subprocess.run', side_effect=self._no_worktree_no_commit_side_effect()):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), status=status, _orchestrator_live=True
            )

        assert result.orchestrator_live is False
        assert result.worktree_registered is False
        assert result.recent_commit is False
        assert result.is_live is False

    @pytest.mark.parametrize(
        'status', ['pending', 'in-progress', 'review', 'merge-deferred']
    )
    def test_non_ineligible_status_preserves_orchestrator_signal(self, tmp_path, status):
        """Non-ineligible statuses leave the project-wide orchestrator_live signal intact.

        'blocked' is deliberately EXCLUDED from this parametrize list (unlike prior to
        task 2409): with no task_kind passed (defaults to None) and no git evidence,
        'blocked' now depends on rule 3 (see _orchestrator_signal_ineligible) rather
        than being unconditionally preserved. See
        TestBlockedNormalOrchestratorSuppression (bare case: suppressed) and
        TestBlockedDeterministicOrchestratorSuppression for 'blocked''s task_kind-aware
        coverage.
        """
        with patch('subprocess.run', side_effect=self._no_worktree_no_commit_side_effect()):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), status=status, _orchestrator_live=True
            )

        assert result.orchestrator_live is True
        assert result.is_live is True

    def test_status_none_preserves_orchestrator_signal_backward_compatible(self, tmp_path):
        """status=None (the default) is backward-compatible: signal is unaffected."""
        with patch('subprocess.run', side_effect=self._no_worktree_no_commit_side_effect()):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), _orchestrator_live=True
            )

        assert result.orchestrator_live is True
        assert result.is_live is True

    def test_ineligible_status_does_not_suppress_worktree_signal(self, tmp_path):
        """A deferred task with a REGISTERED worktree is still live — per-task evidence wins."""
        def side_effect(args, **kwargs):
            if '--porcelain' in args:
                return subprocess.CompletedProcess(
                    args=args, returncode=0,
                    stdout=_worktree_porcelain_with_branch(_BRANCH), stderr=''
                )
            return subprocess.CompletedProcess(
                args=args, returncode=1, stdout='', stderr=''
            )

        with patch('subprocess.run', side_effect=side_effect):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), status='deferred', _orchestrator_live=True
            )

        assert result.orchestrator_live is False
        assert result.worktree_registered is True
        assert result.is_live is True

    def test_is_workflow_live_for_task_forwards_status(self, tmp_path):
        """is_workflow_live_for_task forwards status through to detect_live_workflow."""
        with patch('subprocess.run', side_effect=self._no_worktree_no_commit_side_effect()):
            live = is_workflow_live_for_task(
                _TASK_ID, str(tmp_path), status='deferred', _orchestrator_live=True
            )

        assert live is False


# ---------------------------------------------------------------------------
# Blocked-deterministic orchestrator_live suppression (task 2067)
# ---------------------------------------------------------------------------


class TestBlockedDeterministicOrchestratorSuppression:
    """detect_live_workflow(status='blocked', task_kind='deterministic') suppresses the
    project-wide orchestrator_live signal for deterministic tasks that are blocked.

    A deterministic task (task_kind == 'deterministic') never acquires a worktree/branch
    (it is routed to DeterministicRunner instead), so the bare project-wide orchestrator
    lock is not task-specific evidence for it while blocked. Normal blocked tasks
    (task_kind != 'deterministic', including task_kind=None) keep the signal ONLY when
    they carry genuine per-task evidence (a registered worktree or a recent commit) —
    since task 2409, the bare-orchestrator-only case is suppressed too. See
    TestBlockedNormalOrchestratorSuppression below for that case (task 2409 closes the
    repeated re-deferral loop this caused for tasks 2335/2196).
    """

    def _no_worktree_no_commit_side_effect(self):
        """subprocess.run side_effect driving both git signals False."""
        def side_effect(args, **kwargs):
            return subprocess.CompletedProcess(
                args=args, returncode=0,
                stdout=_worktree_porcelain_no_branch(), stderr=''
            )
        return side_effect

    def test_blocked_deterministic_suppresses_orchestrator_signal(self, tmp_path):
        """status='blocked' + task_kind='deterministic' forces orchestrator_live False.

        Both git signals are False, so is_live=False proves the project-wide
        orchestrator lock is not being consulted at all for this combination.
        """
        with patch('subprocess.run', side_effect=self._no_worktree_no_commit_side_effect()):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path),
                status='blocked', task_kind='deterministic', _orchestrator_live=True,
            )

        assert result.orchestrator_live is False
        assert result.worktree_registered is False
        assert result.recent_commit is False
        assert result.is_live is False

    # test_blocked_non_deterministic_preserves_orchestrator_signal previously lived
    # here, asserting that a blocked normal/None-task_kind task with NO worktree and
    # NO recent commit still preserved the orchestrator_live signal. Task 2409
    # reverses that premise for the bare-evidence case (it caused a repeated
    # re-deferral loop for tasks 2335/2196) — see
    # TestBlockedNormalOrchestratorSuppression below, which folds in both the
    # inverted bare-suppression case and the with-worktree/with-recent-commit
    # preservation cases.

    def test_blocked_deterministic_does_not_suppress_worktree_signal(self, tmp_path):
        """A blocked deterministic task with a REGISTERED worktree is still live —
        per-task evidence wins even though the bare orchestrator signal is suppressed.
        """
        def side_effect(args, **kwargs):
            if '--porcelain' in args:
                return subprocess.CompletedProcess(
                    args=args, returncode=0,
                    stdout=_worktree_porcelain_with_branch(_BRANCH), stderr=''
                )
            return subprocess.CompletedProcess(
                args=args, returncode=1, stdout='', stderr=''
            )

        with patch('subprocess.run', side_effect=side_effect):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path),
                status='blocked', task_kind='deterministic', _orchestrator_live=True,
            )

        assert result.orchestrator_live is False
        assert result.worktree_registered is True
        assert result.is_live is True

    @pytest.mark.parametrize('status', ['pending', 'in-progress', 'review', 'merge-deferred'])
    def test_non_blocked_status_preserves_orchestrator_signal_even_when_deterministic(
        self, tmp_path, status
    ):
        """Only the blocked+deterministic combination triggers the new rule.

        Other live statuses (pending/in-progress/review/merge-deferred) with a
        deterministic task_kind leave the project-wide orchestrator_live signal intact.
        """
        with patch('subprocess.run', side_effect=self._no_worktree_no_commit_side_effect()):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path),
                status=status, task_kind='deterministic', _orchestrator_live=True,
            )

        assert result.orchestrator_live is True
        assert result.is_live is True

    def test_is_workflow_live_for_task_forwards_task_kind(self, tmp_path):
        """is_workflow_live_for_task forwards task_kind through to detect_live_workflow."""
        with patch('subprocess.run', side_effect=self._no_worktree_no_commit_side_effect()):
            live = is_workflow_live_for_task(
                _TASK_ID, str(tmp_path),
                status='blocked', task_kind='deterministic', _orchestrator_live=True,
            )

        assert live is False


# ---------------------------------------------------------------------------
# Blocked-normal bare-orchestrator suppression (task 2409)
# ---------------------------------------------------------------------------


class TestBlockedNormalOrchestratorSuppression:
    """detect_live_workflow(status='blocked', task_kind in (None, 'normal')) suppresses
    the project-wide orchestrator_live signal when the task carries NO per-task
    evidence (no registered worktree, no recent commit) — the bare-orchestrator false
    positive that caused tasks 2335/2196 to loop through repeated re-deferral: a
    blocked, normal-kind task with satisfied deps and no genuine live pipeline showed
    only 'orchestrator' in Live-Workflow Signals and was treated as owned/live.

    When genuine per-task evidence DOES exist (a registered worktree or a recent
    commit on the task's branch), the project-wide orchestrator lock remains real
    corroborating evidence and is preserved — a normal blocked task may legitimately
    be mid-pipeline and auto-unblock (task 2031's concern). Non-blocked statuses
    (pending/in-progress/review/merge-deferred) are entirely unaffected by this rule,
    regardless of task_kind or git signals — the status guard from task 2031
    preserves dispatch-raceable and in-pipeline tasks.
    """

    def _no_worktree_no_commit_side_effect(self):
        """subprocess.run side_effect driving both git signals False."""
        def side_effect(args, **kwargs):
            return subprocess.CompletedProcess(
                args=args, returncode=0,
                stdout=_worktree_porcelain_no_branch(), stderr=''
            )
        return side_effect

    @pytest.mark.parametrize('task_kind', ['normal', None])
    def test_blocked_normal_bare_orchestrator_signal_suppressed(self, tmp_path, task_kind):
        """status='blocked' + task_kind in ('normal', None) + no worktree/commit forces
        orchestrator_live False — the bare-orchestrator false positive (tasks 2335/2196).

        Both git signals are False, so is_live=False proves the project-wide
        orchestrator lock is not being consulted at all for this combination. This
        inverts the pre-2409 expectation (see the note left in
        TestBlockedDeterministicOrchestratorSuppression where the old
        test_blocked_non_deterministic_preserves_orchestrator_signal used to live).
        """
        with patch('subprocess.run', side_effect=self._no_worktree_no_commit_side_effect()):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path),
                status='blocked', task_kind=task_kind, _orchestrator_live=True,
            )

        assert result.orchestrator_live is False
        assert result.worktree_registered is False
        assert result.recent_commit is False
        assert result.is_live is False

    def test_blocked_normal_with_registered_worktree_preserves_signal(self, tmp_path):
        """status='blocked' + task_kind='normal' + a REGISTERED worktree keeps
        orchestrator_live True — genuine per-task evidence means the bare-evidence
        guard does not apply, and the project-wide lock is still reported honestly
        for the renderer's per-signal display.
        """
        def side_effect(args, **kwargs):
            if '--porcelain' in args:
                return subprocess.CompletedProcess(
                    args=args, returncode=0,
                    stdout=_worktree_porcelain_with_branch(_BRANCH), stderr=''
                )
            return subprocess.CompletedProcess(
                args=args, returncode=1, stdout='', stderr=''
            )

        with patch('subprocess.run', side_effect=side_effect):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path),
                status='blocked', task_kind='normal', _orchestrator_live=True,
            )

        assert result.orchestrator_live is True
        assert result.worktree_registered is True
        assert result.is_live is True

    def test_blocked_normal_with_recent_commit_preserves_signal(self, tmp_path):
        """status='blocked' + task_kind='normal' + a recent commit on task/<id> keeps
        orchestrator_live True — genuine per-task evidence means the bare-evidence
        guard does not apply.
        """
        now = datetime(2026, 1, 1, tzinfo=UTC)
        recent_ts = (now - timedelta(hours=1)).isoformat()

        def side_effect(args, **kwargs):
            if '--porcelain' in args:
                return subprocess.CompletedProcess(
                    args=args, returncode=0,
                    stdout=_worktree_porcelain_no_branch(), stderr=''
                )
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout=recent_ts, stderr=''
            )

        with patch('subprocess.run', side_effect=side_effect):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path),
                status='blocked', task_kind='normal', _orchestrator_live=True,
                now=now,
            )

        assert result.orchestrator_live is True
        assert result.recent_commit is True
        assert result.is_live is True

    @pytest.mark.parametrize('status', ['pending', 'in-progress', 'review', 'merge-deferred'])
    def test_non_blocked_status_preserves_orchestrator_signal_for_normal_task(
        self, tmp_path, status
    ):
        """Only the blocked+normal(-or-None) combination triggers rule 3. Other live
        statuses (pending/in-progress/review/merge-deferred) with a normal task_kind
        leave the project-wide orchestrator_live signal intact — the status guard
        preserves dispatch-raceable / in-pipeline tasks regardless of task_kind.
        """
        with patch('subprocess.run', side_effect=self._no_worktree_no_commit_side_effect()):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path),
                status=status, task_kind='normal', _orchestrator_live=True,
            )

        assert result.orchestrator_live is True
        assert result.is_live is True

    def test_is_workflow_live_for_task_suppresses_for_blocked_normal(self, tmp_path):
        """is_workflow_live_for_task returns False for a blocked normal task with no
        git signals and only the bare project-wide orchestrator lock as evidence.
        """
        with patch('subprocess.run', side_effect=self._no_worktree_no_commit_side_effect()):
            live = is_workflow_live_for_task(
                _TASK_ID, str(tmp_path),
                status='blocked', task_kind='normal', _orchestrator_live=True,
            )

        assert live is False


# ---------------------------------------------------------------------------
# Pure-gate metadata classifier (task 3751)
# ---------------------------------------------------------------------------


class TestIsPureGateMetadata:
    """is_pure_gate_metadata identifies the DeterministicRunner PURE-GATE shape.

    A pure gate is `always_escalates` truthy AND `before_done` absent/falsy — the
    same two metadata fields DeterministicRunner itself branches on. Its whole
    execution is "file one born-at-L2 escalation, stamp metadata.gate_escalated_at,
    set status blocked": no script, no systemd, no git_ops. A truthy `before_done`
    DISQUALIFIES the shape because that path runs a blocking deploy/predicate
    script while the task's status is still 'pending'.

    Fail-safe direction: only POSITIVE evidence of the shape returns True. Any
    non-Mapping input (None, a string, an int, a list) returns False, so an
    absent/unparseable metadata blob leaves the task live.
    """

    def test_always_escalates_alone_is_a_pure_gate(self):
        """`always_escalates` truthy with `before_done` absent entirely => True.

        This is dark_factory task 3845's exact metadata shape (verified by a
        direct get_task read): no `before_done` key at all.
        """
        assert is_pure_gate_metadata({'always_escalates': True}) is True

    def test_explicit_none_before_done_is_a_pure_gate(self):
        """An explicit `before_done: None` is equivalent to the key being absent."""
        assert is_pure_gate_metadata({'always_escalates': True, 'before_done': None}) is True

    def test_truthy_before_done_disqualifies(self):
        """A before_done deploy/predicate task is NOT a pure gate.

        `Harness._run_deterministic_slot` never flips the task to 'in-progress',
        so such a task stays 'pending' for the entire duration of a blocking
        script run — recon must not treat it as provably-idle.
        """
        metadata = {
            'always_escalates': True,
            'before_done': {'kind': 'predicate', 'script': 'x.sh'},
        }
        assert is_pure_gate_metadata(metadata) is False

    @pytest.mark.parametrize(
        'metadata',
        [
            {'always_escalates': False},
            {},
            {'task_kind': 'deterministic'},
        ],
        ids=['always_escalates_false', 'empty', 'task_kind_only'],
    )
    def test_missing_or_falsy_always_escalates_is_not_a_pure_gate(self, metadata):
        """Without truthy `always_escalates` there is no positive evidence of the shape."""
        assert is_pure_gate_metadata(metadata) is False

    @pytest.mark.parametrize(
        'metadata',
        [None, 'not-a-dict', 42, []],
        ids=['none', 'str', 'int', 'list'],
    )
    def test_non_mapping_input_is_not_a_pure_gate(self, metadata):
        """Fail-safe toward live: anything that is not a Mapping returns False."""
        assert is_pure_gate_metadata(metadata) is False

    def test_truthiness_not_identity(self):
        """A truthy-but-not-`True` value still qualifies.

        Task metadata round-trips through JSON, so the classifier must use
        truthiness rather than `is True`.
        """
        assert is_pure_gate_metadata({'always_escalates': 'true'}) is True


# ---------------------------------------------------------------------------
# Pending-deterministic pure-gate orchestrator_live suppression (task 3751)
# ---------------------------------------------------------------------------


class TestPendingDeterministicPureGateOrchestratorSuppression:
    """detect_live_workflow(status='pending', task_kind='deterministic', pure_gate=True)
    suppresses the project-wide orchestrator_live signal — rule 5, task 3751.

    A PURE GATE (`always_escalates` truthy, `before_done` absent) never acquires a
    worktree/branch and its whole DeterministicRunner run is "file escalation, stamp
    gate_escalated_at, set blocked" — no script, no systemd, no git_ops. So the bare
    project-wide orchestrator lock is not task-specific evidence for it while pending.

    This is the direct sibling of TestBlockedDeterministicOrchestratorSuppression
    (task 2067's blocked case) and is deliberately NARROWER than adding 'pending' to
    ORCH_LIVE_INELIGIBLE_STATUSES: a pending deterministic task WITH `before_done`
    may be mid-deploy inside DeterministicRunner (Harness._run_deterministic_slot
    never flips it to 'in-progress'), and it has no git evidence to reveal that.
    """

    def _no_worktree_no_commit_side_effect(self):
        """subprocess.run side_effect driving both git signals False."""
        def side_effect(args, **kwargs):
            return subprocess.CompletedProcess(
                args=args, returncode=0,
                stdout=_worktree_porcelain_no_branch(), stderr=''
            )
        return side_effect

    def test_pending_deterministic_pure_gate_suppresses_orchestrator_signal(self, tmp_path):
        """THE REGRESSION — status='pending' + deterministic + pure_gate forces
        orchestrator_live False.

        This is dark_factory task 3845's exact shape (verified first-hand:
        `always_escalates=True` with no `before_done`), which stalled 3+ consecutive
        reconciliation cycles showing ONLY the bare `orchestrator` signal. Both git
        signals are False, so is_live=False proves the project-wide lock is not being
        consulted at all for this combination.
        """
        with patch('subprocess.run', side_effect=self._no_worktree_no_commit_side_effect()):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path),
                status='pending', task_kind='deterministic', pure_gate=True,
                _orchestrator_live=True,
            )

        assert result.orchestrator_live is False
        assert result.worktree_registered is False
        assert result.recent_commit is False
        assert result.is_live is False

    def test_pending_deterministic_without_pure_gate_preserves_signal(self, tmp_path):
        """NARROWING — pure_gate=False keeps the orchestrator signal for a pending
        deterministic task.

        Such a task carries a `before_done` deploy/predicate and may be mid-run inside
        DeterministicRunner with zero git evidence to reveal it; recon must not race it.
        """
        with patch('subprocess.run', side_effect=self._no_worktree_no_commit_side_effect()):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path),
                status='pending', task_kind='deterministic', pure_gate=False,
                _orchestrator_live=True,
            )

        assert result.orchestrator_live is True
        assert result.is_live is True

    @pytest.mark.parametrize('task_kind', ['normal', None], ids=['normal', 'absent'])
    def test_non_deterministic_task_kind_preserves_signal(self, tmp_path, task_kind):
        """task_kind guard — rule 5 requires task_kind == 'deterministic'.

        An ordinary pending task is dispatch-eligible, so the project-wide lock stays
        real evidence for it regardless of any pure_gate value.
        """
        with patch('subprocess.run', side_effect=self._no_worktree_no_commit_side_effect()):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path),
                status='pending', task_kind=task_kind, pure_gate=True,
                _orchestrator_live=True,
            )

        assert result.orchestrator_live is True
        assert result.is_live is True

    @pytest.mark.parametrize('status', ['in-progress', 'review', 'merge-deferred'])
    def test_non_pending_status_preserves_signal(self, tmp_path, status):
        """status guard — only 'pending' joins 'blocked' for the deterministic rules."""
        with patch('subprocess.run', side_effect=self._no_worktree_no_commit_side_effect()):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path),
                status=status, task_kind='deterministic', pure_gate=True,
                _orchestrator_live=True,
            )

        assert result.orchestrator_live is True
        assert result.is_live is True

    def test_pure_gate_omitted_is_inert(self, tmp_path):
        """DEFAULT INERTNESS — omitting pure_gate entirely preserves today's behavior.

        Byte-for-byte backward compatibility for every existing caller that does not
        pass the new kwarg.
        """
        with patch('subprocess.run', side_effect=self._no_worktree_no_commit_side_effect()):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path),
                status='pending', task_kind='deterministic', _orchestrator_live=True,
            )

        assert result.orchestrator_live is True
        assert result.is_live is True

    def test_pure_gate_does_not_suppress_worktree_signal(self, tmp_path):
        """PER-TASK EVIDENCE WINS — a pending pure gate with a REGISTERED worktree is
        still live, even though the bare orchestrator signal is suppressed.

        Mirrors task 2067's test_blocked_deterministic_does_not_suppress_worktree_signal.
        This is also what satisfies the "never dispatched / no worktree ever created"
        framing without needing a separate branch-absence condition.
        """
        def side_effect(args, **kwargs):
            if '--porcelain' in args:
                return subprocess.CompletedProcess(
                    args=args, returncode=0,
                    stdout=_worktree_porcelain_with_branch(_BRANCH), stderr=''
                )
            return subprocess.CompletedProcess(
                args=args, returncode=1, stdout='', stderr=''
            )

        with patch('subprocess.run', side_effect=side_effect):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path),
                status='pending', task_kind='deterministic', pure_gate=True,
                _orchestrator_live=True,
            )

        assert result.orchestrator_live is False
        assert result.worktree_registered is True
        assert result.is_live is True

    def test_is_workflow_live_for_task_forwards_pure_gate(self, tmp_path):
        """is_workflow_live_for_task forwards pure_gate through **kwargs."""
        with patch('subprocess.run', side_effect=self._no_worktree_no_commit_side_effect()):
            live = is_workflow_live_for_task(
                _TASK_ID, str(tmp_path),
                status='pending', task_kind='deterministic', pure_gate=True,
                _orchestrator_live=True,
            )

        assert live is False


# ---------------------------------------------------------------------------
# reify#5245 hardening: reaped-worktree / bare-branch / stranded-orchestrator
# false-positive (task 2767)
# ---------------------------------------------------------------------------


class TestPrunableWorktreeSignal:
    """A worktree porcelain entry marked `prunable` (its directory was reaped) must
    not count as worktree_registered — a reaped worktree is not a live workspace,
    even though its branch line is still present in the porcelain stanza
    (reify#5245's shape).
    """

    def _make_run(self, stdout: str, returncode: int = 0):
        """Build a mock subprocess.CompletedProcess for worktree list."""
        return subprocess.CompletedProcess(
            args=['git', 'worktree', 'list', '--porcelain'],
            returncode=returncode,
            stdout=stdout,
            stderr='',
        )

    def _make_log_run(self, stdout: str = '', returncode: int = 1):
        """Build a mock subprocess.CompletedProcess for git log (branch absent by default)."""
        return subprocess.CompletedProcess(
            args=['git', 'log', '-1', '--format=%cI', _BRANCH],
            returncode=returncode,
            stdout=stdout,
            stderr='',
        )

    def _make_revlist_run(self, stdout: str = '0', returncode: int = 0):
        """Build a mock subprocess.CompletedProcess for the (forthcoming) rev-list call."""
        return subprocess.CompletedProcess(
            args=['git', 'rev-list', '--count', f'main..{_BRANCH}'],
            returncode=returncode,
            stdout=stdout,
            stderr='',
        )

    def _run_side_effect(
        self,
        worktree_stdout: str,
        log_stdout: str = '',
        log_rc: int = 1,
        revlist_stdout: str = '0',
        revlist_rc: int = 0,
    ):
        """Return a function-style subprocess.run side_effect dispatching on args.

        Matches the existing style in TestWorktreeSignal (NOT a list side_effect),
        so an unanticipated extra call — e.g. the rev-list call landing in a later
        step — degrades gracefully instead of raising StopIteration.
        """
        worktree_result = self._make_run(worktree_stdout)
        log_result = self._make_log_run(log_stdout, log_rc)
        revlist_result = self._make_revlist_run(revlist_stdout, revlist_rc)

        def side_effect(args, **kwargs):
            if '--porcelain' in args:
                return worktree_result
            if 'rev-list' in args:
                return revlist_result
            return log_result

        return side_effect

    def test_prunable_worktree_not_counted(self, tmp_path):
        """A prunable (reaped) worktree entry does not count as worktree_registered,
        even though its branch line is present in the same porcelain stanza.
        """
        side_effect = self._run_side_effect(
            _worktree_porcelain_prunable(_BRANCH),
            log_rc=1,
            revlist_stdout='0',
        )
        with patch('subprocess.run', side_effect=side_effect):
            result = detect_live_workflow(_TASK_ID, str(tmp_path))

        assert result.worktree_registered is False
        assert result.is_live is False

    def test_live_nonprunable_worktree_still_counts_even_when_bare(self, tmp_path):
        """A LIVE (non-prunable) worktree stays a live signal even when the branch
        itself is bare (zero own commits) — a just-started dispatch (branch created,
        no commits yet) must stay live.
        """
        side_effect = self._run_side_effect(
            _worktree_porcelain_with_branch(_BRANCH),
            log_rc=1,
            revlist_stdout='0',
        )
        with patch('subprocess.run', side_effect=side_effect):
            result = detect_live_workflow(_TASK_ID, str(tmp_path))

        assert result.worktree_registered is True
        assert result.is_live is True


class TestBareBranchRecentCommit:
    """A branch's tip timestamp does not count as `recent_commit` when the branch
    has zero commits of its own — its tip is only the base-branch commit, not
    task work (reify#5245's shape: the reflog held only a "Created from main"
    entry with HEAD == main).
    """

    _NOW = datetime(2026, 6, 5, 12, 0, 0, tzinfo=UTC)

    def _run_side_effect(
        self,
        commit_ts_str: str | None,
        revlist_stdout: str = '0',
        revlist_rc: int = 0,
        revlist_raises: bool = False,
    ):
        """Return a subprocess.run side_effect: no worktree registered, a given
        git-log tip timestamp, and a canned (or error/raising) rev-list count.
        """
        def side_effect(args, **kwargs):
            if '--porcelain' in args:
                return subprocess.CompletedProcess(
                    args=args, returncode=0,
                    stdout=_worktree_porcelain_no_branch(), stderr=''
                )
            if 'rev-list' in args:
                if revlist_raises:
                    raise subprocess.TimeoutExpired(cmd=args, timeout=10)
                return subprocess.CompletedProcess(
                    args=args, returncode=revlist_rc, stdout=revlist_stdout, stderr=''
                )
            # git log call
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout=commit_ts_str or '', stderr=''
            )
        return side_effect

    def test_bare_branch_suppresses_recent_commit(self, tmp_path):
        """A recent tip timestamp is suppressed when rev-list reports zero own
        commits — the recent tip is only the base commit, not task work.
        """
        ts = (self._NOW - timedelta(hours=1)).isoformat()
        side_effect = self._run_side_effect(ts, revlist_stdout='0', revlist_rc=0)
        with patch('subprocess.run', side_effect=side_effect):
            result = detect_live_workflow(_TASK_ID, str(tmp_path), now=self._NOW)

        assert result.recent_commit is False

    def test_recent_commit_preserved_when_branch_has_own_commits(self, tmp_path):
        """A recent tip timestamp is preserved when rev-list reports own commits."""
        ts = (self._NOW - timedelta(hours=1)).isoformat()
        side_effect = self._run_side_effect(ts, revlist_stdout='3', revlist_rc=0)
        with patch('subprocess.run', side_effect=side_effect):
            result = detect_live_workflow(_TASK_ID, str(tmp_path), now=self._NOW)

        assert result.recent_commit is True

    def test_revlist_error_treated_as_not_bare(self, tmp_path):
        """An unknown own-commit count (rev-list error or exception) fails safe:
        not bare => no suppression => recent_commit stays True.
        """
        ts = (self._NOW - timedelta(hours=1)).isoformat()

        side_effect_rc1 = self._run_side_effect(ts, revlist_rc=1)
        with patch('subprocess.run', side_effect=side_effect_rc1):
            result = detect_live_workflow(_TASK_ID, str(tmp_path), now=self._NOW)
        assert result.recent_commit is True

        side_effect_raises = self._run_side_effect(ts, revlist_raises=True)
        with patch('subprocess.run', side_effect=side_effect_raises):
            result = detect_live_workflow(_TASK_ID, str(tmp_path), now=self._NOW)
        assert result.recent_commit is True


class TestBareStrandedOrchestratorSuppression:
    """The reify#5245 reproduction: a task whose branch is bare (zero own
    commits) and whose worktree was reaped (prunable) must not be kept "live"
    by the bare project-wide orchestrator_live signal — that signal only
    reflects a running orchestrator process is dispatching SOME task in the
    project, not necessarily this one.  Status-agnostic (unlike rules 1-3):
    recon_write_policy Gate 2 calls the detector without task_kind and with
    whatever status the task currently holds, so the suppression must fire
    regardless of status (excluding the statuses/combinations already handled
    by rules 1-3).
    """

    def _side_effect(
        self,
        worktree_stdout: str,
        revlist_stdout: str = '0',
        revlist_rc: int = 0,
        log_rc: int = 1,
        log_stdout: str = '',
    ):
        """subprocess.run side_effect: canned worktree/rev-list/git-log results."""
        def side_effect(args, **kwargs):
            if '--porcelain' in args:
                return subprocess.CompletedProcess(
                    args=args, returncode=0, stdout=worktree_stdout, stderr=''
                )
            if 'rev-list' in args:
                return subprocess.CompletedProcess(
                    args=args, returncode=revlist_rc, stdout=revlist_stdout, stderr=''
                )
            # git log call
            return subprocess.CompletedProcess(
                args=args, returncode=log_rc, stdout=log_stdout, stderr=''
            )
        return side_effect

    @pytest.mark.parametrize('status', ['pending', 'in-progress', 'review', 'merge-deferred'])
    def test_bare_stranded_task_not_live_under_running_orchestrator(self, tmp_path, status):
        """A prunable worktree + a bare branch must suppress the bare project-wide
        orchestrator_live signal, across every status not already covered by
        rules 1-3 — the exact reify#5245 shape.  Also asserts through
        is_workflow_live_for_task, the precise call recon_write_policy Gate 2
        makes (status passed, task_kind not passed).
        """
        side_effect = self._side_effect(_worktree_porcelain_prunable(_BRANCH))
        with patch('subprocess.run', side_effect=side_effect):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), status=status, _orchestrator_live=True,
            )
            live = is_workflow_live_for_task(
                _TASK_ID, str(tmp_path), status=status, _orchestrator_live=True,
            )

        assert result.orchestrator_live is False
        assert result.worktree_registered is False
        assert result.recent_commit is False
        assert result.is_live is False
        assert live is False

    def test_bare_branch_with_live_worktree_stays_live(self, tmp_path):
        """A LIVE (non-prunable) worktree on a bare branch must NOT be suppressed —
        rule 4 must not fire when a live worktree exists, protecting a
        freshly-dispatched pipeline (branch just created, worktree present, no
        commits yet).
        """
        side_effect = self._side_effect(_worktree_porcelain_with_branch(_BRANCH))
        with patch('subprocess.run', side_effect=side_effect):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), status='pending', _orchestrator_live=True,
            )

        assert result.worktree_registered is True
        assert result.is_live is True

    def test_absent_branch_preserves_orchestrator_signal(self, tmp_path):
        """No branch at all (not-yet-dispatched task) => rev-list fails => count
        None => NOT bare => rule 4 stays inert => the deliberate task-2031
        dispatch-race protection is preserved (a live orchestrator elsewhere
        still marks a pending task potentially-about-to-be-dispatched as live).
        """
        side_effect = self._side_effect(
            _worktree_porcelain_no_branch(), revlist_rc=1, revlist_stdout='',
        )
        with patch('subprocess.run', side_effect=side_effect):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), status='pending', _orchestrator_live=True,
            )

        assert result.orchestrator_live is True
        assert result.is_live is True

    def test_revlist_error_keeps_orchestrator_signal(self, tmp_path):
        """A rev-list exception (not just a non-zero return) also yields an unknown
        count => NOT bare => the orchestrator signal is preserved (fail-safe
        toward live, matching TestBareBranchRecentCommit's rev-list-error case).
        """
        def side_effect(args, **kwargs):
            if '--porcelain' in args:
                return subprocess.CompletedProcess(
                    args=args, returncode=0,
                    stdout=_worktree_porcelain_no_branch(), stderr='',
                )
            if 'rev-list' in args:
                raise subprocess.TimeoutExpired(cmd=args, timeout=10)
            return subprocess.CompletedProcess(args=args, returncode=1, stdout='', stderr='')

        with patch('subprocess.run', side_effect=side_effect):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), status='pending', _orchestrator_live=True,
            )

        assert result.orchestrator_live is True
        assert result.is_live is True


# ---------------------------------------------------------------------------
# Corroboration helpers (task 2963)
# ---------------------------------------------------------------------------

_CORROB_NOW = datetime(2026, 7, 23, 12, 0, 0, tzinfo=UTC)
_CORROB_TASK_ID = '2763'


def _routing_metadata(decided_at: str | None) -> dict:
    """Build a task ``metadata`` dict with a valid ``routing.latest`` mirror.

    Supplies all RoutingDecisionMirror-required fields so ``.latest`` is
    non-None; ``decided_at`` is the only field the corroboration path reads.
    """
    return {
        'routing': {
            'latest': {
                'role': 'implementer',
                'model': 'claude',
                'effort': 'high',
                'budget_usd': 1.0,
                'max_turns': 10,
                'source_layer': 'default',
                'decided_at': decided_at,
            }
        }
    }


class TestTaskInSchedulerHoldersOrParks:
    """_task_in_scheduler_holders_or_parks: task_id in parks-keys OR holders-values."""

    def test_true_when_task_id_is_parks_key(self):
        state = {'parks': {_CORROB_TASK_ID: {'reason': 'x'}}, 'current_holders': {}}
        assert _task_in_scheduler_holders_or_parks(_CORROB_TASK_ID, state) is True

    def test_true_when_task_id_is_current_holders_value(self):
        # current_holders is {module: task_id}; the task_id is a VALUE, not a key.
        state = {'parks': {}, 'current_holders': {'reconciliation': _CORROB_TASK_ID}}
        assert _task_in_scheduler_holders_or_parks(_CORROB_TASK_ID, state) is True

    def test_false_when_in_neither(self):
        state = {'parks': {'9999': {}}, 'current_holders': {'mod': '9999'}}
        assert _task_in_scheduler_holders_or_parks(_CORROB_TASK_ID, state) is False

    def test_no_raise_when_state_none(self):
        assert _task_in_scheduler_holders_or_parks(_CORROB_TASK_ID, None) is False

    def test_no_raise_when_keys_missing(self):
        assert _task_in_scheduler_holders_or_parks(_CORROB_TASK_ID, {}) is False

    def test_int_task_id_coerced_to_str(self):
        # scheduler_state keys/values are strings (JSON); an int task_id still matches.
        state = {'parks': {'2763': {}}, 'current_holders': {}}
        assert _task_in_scheduler_holders_or_parks(2763, state) is True


class TestRoutingDecidedAfterRestart:
    """_routing_decided_after_restart: decided_at strictly newer than started."""

    _STARTED = datetime(2026, 7, 23, 10, 0, 0, tzinfo=UTC)

    def test_true_when_decided_after_started(self):
        assert _routing_decided_after_restart('2026-07-23T11:00:00Z', self._STARTED) is True

    def test_false_when_decided_before_started(self):
        assert _routing_decided_after_restart('2026-07-23T09:00:00Z', self._STARTED) is False

    def test_false_when_decided_equals_started(self):
        assert _routing_decided_after_restart('2026-07-23T10:00:00Z', self._STARTED) is False

    def test_false_when_decided_none(self):
        assert _routing_decided_after_restart(None, self._STARTED) is False

    def test_false_when_decided_unparseable(self):
        assert _routing_decided_after_restart('not-a-time', self._STARTED) is False

    def test_false_when_started_none(self):
        # Cannot prove post-restart without a restart boundary.
        assert _routing_decided_after_restart('2026-07-23T11:00:00Z', None) is False


class TestHasLiveWorkflowCorroboration:
    """has_live_workflow_corroboration: OR of the three per-task signals."""

    _STARTED = datetime(2026, 7, 23, 10, 0, 0, tzinfo=UTC)

    def test_true_when_claimant_live_short_circuits(self):
        # Even with empty scheduler state and no routing, a live claimant suffices.
        assert has_live_workflow_corroboration(
            _CORROB_TASK_ID,
            claimant_live=True,
            scheduler_state=None,
            routing_latest_decided_at=None,
            orchestrator_started_at=None,
        ) is True

    def test_true_via_scheduler_only(self):
        assert has_live_workflow_corroboration(
            _CORROB_TASK_ID,
            claimant_live=False,
            scheduler_state={'parks': {_CORROB_TASK_ID: {}}, 'current_holders': {}},
            routing_latest_decided_at=None,
            orchestrator_started_at=None,
        ) is True

    def test_true_via_routing_only(self):
        assert has_live_workflow_corroboration(
            _CORROB_TASK_ID,
            claimant_live=False,
            scheduler_state=None,
            routing_latest_decided_at='2026-07-23T11:00:00Z',
            orchestrator_started_at=self._STARTED,
        ) is True

    def test_false_when_all_absent(self):
        assert has_live_workflow_corroboration(
            _CORROB_TASK_ID,
            claimant_live=False,
            scheduler_state={'parks': {}, 'current_holders': {}},
            routing_latest_decided_at='2026-07-23T09:00:00Z',  # before started
            orchestrator_started_at=self._STARTED,
        ) is False


class TestCorroborationForTask:
    """corroboration_for_task: assembles the three signals from a task dict."""

    _STARTED = datetime(2026, 7, 23, 10, 0, 0, tzinfo=UTC)

    def test_default_heartbeat_ttl_is_ten_minutes(self):
        assert timedelta(minutes=10) == DEFAULT_HEARTBEAT_TTL

    def test_fresh_heartbeat_true(self):
        task = {
            'id': _CORROB_TASK_ID,
            'status': 'in-progress',
            'claimant_run_id': 'run-1/sess-1/pid=42',
            'heartbeat_at': (_CORROB_NOW - timedelta(minutes=1)).isoformat(),
        }
        assert corroboration_for_task(
            task, _CORROB_TASK_ID, now=_CORROB_NOW,
            scheduler_state=None, orchestrator_started_at=None,
        ) is True

    def test_stale_heartbeat_no_other_signal_false(self):
        task = {
            'id': _CORROB_TASK_ID,
            'status': 'in-progress',
            'claimant_run_id': 'run-1/sess-1/pid=42',
            # 30 min old > 10 min ttl → stale → not a live claimant.
            'heartbeat_at': (_CORROB_NOW - timedelta(minutes=30)).isoformat(),
        }
        assert corroboration_for_task(
            task, _CORROB_TASK_ID, now=_CORROB_NOW,
            scheduler_state={'parks': {}, 'current_holders': {}},
            orchestrator_started_at=self._STARTED,
        ) is False

    def test_routing_after_started_true(self):
        task = {
            'id': _CORROB_TASK_ID,
            'status': 'in-progress',
            # No fresh claimant.
            'claimant_run_id': None,
            'metadata': _routing_metadata('2026-07-23T11:00:00Z'),  # after started
        }
        assert corroboration_for_task(
            task, _CORROB_TASK_ID, now=_CORROB_NOW,
            scheduler_state=None, orchestrator_started_at=self._STARTED,
        ) is True

    def test_routing_before_started_false(self):
        task = {
            'id': _CORROB_TASK_ID,
            'status': 'in-progress',
            'claimant_run_id': None,
            'metadata': _routing_metadata('2026-07-23T09:00:00Z'),  # before started
        }
        assert corroboration_for_task(
            task, _CORROB_TASK_ID, now=_CORROB_NOW,
            scheduler_state=None, orchestrator_started_at=self._STARTED,
        ) is False

    def test_missing_metadata_no_raise_relies_on_other_signals(self):
        # No metadata, no claimant, but present in scheduler parks → True (no raise).
        task = {'id': _CORROB_TASK_ID, 'status': 'in-progress', 'claimant_run_id': None}
        assert corroboration_for_task(
            task, _CORROB_TASK_ID, now=_CORROB_NOW,
            scheduler_state={'parks': {_CORROB_TASK_ID: {}}, 'current_holders': {}},
            orchestrator_started_at=self._STARTED,
        ) is True

    def test_empty_metadata_no_raise_all_absent_false(self):
        task = {
            'id': _CORROB_TASK_ID, 'status': 'in-progress',
            'claimant_run_id': None, 'metadata': {},
        }
        assert corroboration_for_task(
            task, _CORROB_TASK_ID, now=_CORROB_NOW,
            scheduler_state=None, orchestrator_started_at=None,
        ) is False


class TestCorroborationGate:
    """detect_live_workflow's in-progress corroboration gate + the new
    WorkflowLiveness.indeterminate field (task 2963)."""

    _NOW = datetime(2026, 7, 23, 12, 0, 0, tzinfo=UTC)

    def _side_effect(
        self,
        *,
        worktree_branch: bool = True,
        log_rc: int = 1,
        log_stdout: str = '',
        revlist_stdout: str = '1',
        revlist_rc: int = 0,
    ):
        """Drive the three git subprocess signals independently.

        ``worktree_branch``: whether the porcelain output lists ``task/<id>``
        (a LIVE, non-prunable worktree). ``log_rc``/``log_stdout``: the git-log
        recent-commit probe. ``revlist_stdout``: the branch's own-commit count
        (``'1'`` = non-bare, keeps rule-4 inert so the orchestrator signal is
        reported honestly).
        """
        worktree_stdout = (
            _worktree_porcelain_with_branch(_BRANCH)
            if worktree_branch
            else _worktree_porcelain_no_branch()
        )

        def side_effect(args, **kwargs):
            if '--porcelain' in args:
                return subprocess.CompletedProcess(
                    args=args, returncode=0, stdout=worktree_stdout, stderr='',
                )
            if 'rev-list' in args:
                return subprocess.CompletedProcess(
                    args=args, returncode=revlist_rc, stdout=revlist_stdout, stderr='',
                )
            return subprocess.CompletedProcess(
                args=args, returncode=log_rc, stdout=log_stdout, stderr='',
            )

        return side_effect

    def test_gate_fires_worktree_only_uncorroborated(self, tmp_path):
        """in-progress, worktree-only (recent_commit False), corroborated=False
        → is_live False, indeterminate True, raw signals preserved."""
        side = self._side_effect(worktree_branch=True, log_rc=1)
        with patch('subprocess.run', side_effect=side):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), now=self._NOW,
                status='in-progress', corroborated=False,
            )
        assert result.worktree_registered is True  # raw detection preserved
        assert result.recent_commit is False
        assert result.is_live is False
        assert result.indeterminate is True

    def test_gate_fires_orchestrator_only_uncorroborated(self, tmp_path):
        """in-progress, orchestrator-lock-only (no worktree, non-bare branch),
        corroborated=False → is_live False, indeterminate True, orchestrator
        signal still reported honestly."""
        side = self._side_effect(worktree_branch=False, log_rc=1, revlist_stdout='1')
        with patch('subprocess.run', side_effect=side):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), now=self._NOW,
                status='in-progress', _orchestrator_live=True, corroborated=False,
            )
        assert result.worktree_registered is False
        assert result.orchestrator_live is True  # raw detection preserved
        assert result.recent_commit is False
        assert result.is_live is False
        assert result.indeterminate is True

    def test_gate_inert_when_corroborated_true(self, tmp_path):
        """corroborated=True keeps the task live (indeterminate False)."""
        side = self._side_effect(worktree_branch=True, log_rc=1)
        with patch('subprocess.run', side_effect=side):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), now=self._NOW,
                status='in-progress', corroborated=True,
            )
        assert result.is_live is True
        assert result.indeterminate is False

    def test_gate_inert_when_corroborated_none_default(self, tmp_path):
        """corroborated=None (default) is backward-compatible — existing behavior."""
        side = self._side_effect(worktree_branch=True, log_rc=1)
        with patch('subprocess.run', side_effect=side):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), now=self._NOW, status='in-progress',
            )
        assert result.is_live is True
        assert result.indeterminate is False

    def test_recent_commit_exempt_from_gate(self, tmp_path):
        """A recent commit is genuine per-task evidence — gate does not fire."""
        recent_ts = (self._NOW - timedelta(hours=1)).isoformat()
        side = self._side_effect(
            worktree_branch=False, log_rc=0, log_stdout=recent_ts, revlist_stdout='1',
        )
        with patch('subprocess.run', side_effect=side):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), now=self._NOW,
                status='in-progress', corroborated=False,
            )
        assert result.recent_commit is True
        assert result.is_live is True
        assert result.indeterminate is False

    @pytest.mark.parametrize('status', ['pending', 'review', 'merge-deferred'])
    def test_gate_is_in_progress_only(self, tmp_path, status):
        """Non-in-progress statuses are never gated, even worktree-only + corroborated=False."""
        side = self._side_effect(worktree_branch=True, log_rc=1)
        with patch('subprocess.run', side_effect=side):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), now=self._NOW,
                status=status, corroborated=False,
            )
        assert result.is_live is True
        assert result.indeterminate is False

    def test_genuine_not_live_is_not_indeterminate(self, tmp_path):
        """All signals False (nothing to downgrade) → is_live False AND indeterminate
        False — distinct from the gated indeterminate case."""
        side = self._side_effect(worktree_branch=False, log_rc=1, revlist_stdout='1')
        with patch('subprocess.run', side_effect=side):
            result = detect_live_workflow(
                _TASK_ID, str(tmp_path), now=self._NOW,
                status='in-progress', corroborated=False,
            )
        assert result.worktree_registered is False
        assert result.recent_commit is False
        assert result.orchestrator_live is False
        assert result.is_live is False
        assert result.indeterminate is False

    def test_is_workflow_live_for_task_forwards_corroborated(self, tmp_path):
        """The convenience wrapper threads corroborated through **kwargs."""
        side = self._side_effect(worktree_branch=True, log_rc=1)
        with patch('subprocess.run', side_effect=side):
            live = is_workflow_live_for_task(
                _TASK_ID, str(tmp_path), now=self._NOW,
                status='in-progress', corroborated=False,
            )
        assert live is False

    def test_indeterminate_defaults_false_on_normal_live(self, tmp_path):
        """A normal live result (no status/corroborated) has indeterminate False."""
        side = self._side_effect(worktree_branch=True, log_rc=1)
        with patch('subprocess.run', side_effect=side):
            result = detect_live_workflow(_TASK_ID, str(tmp_path), now=self._NOW)
        assert result.is_live is True
        assert result.indeterminate is False
