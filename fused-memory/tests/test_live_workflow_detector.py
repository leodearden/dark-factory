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
    WorkflowLiveness,
    detect_live_workflow,
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
