"""Tests for ``verify.run_main_tip_sweep`` — task 1832.

run_main_tip_sweep(config, git_ops) -> tuple[str, VerifyResult] | None

Runs a FULL unscoped verification (all subprojects) against a throwaway
detached worktree pinned at the current main SHA.  Returns (main_sha,
VerifyResult) on success, or None on any infrastructure failure (fail-safe).

Mirrors the mock strategy of test_verify_preexisting_main_break.py:
  - MagicMock git_ops with get_main_sha (AsyncMock) and worktree_base
  - monkeypatch orchestrator.git_ops._run to simulate worktree add/remove
  - monkeypatch orchestrator.verify.run_full_verification (AsyncMock)

Test coverage:
  step-3:  test_run_main_tip_sweep_happy_path
  step-5:  test_run_main_tip_sweep_failsafe_empty_sha
           test_run_main_tip_sweep_failsafe_worktree_add_fails
           test_run_main_tip_sweep_drift_passthrough
  task-1925 step-1 (Part B retry-on-flake):
           test_run_main_tip_sweep_retries_once_and_suppresses_transient_flake
           test_run_main_tip_sweep_both_passes_fail_is_drift
           test_run_main_tip_sweep_passes_first_time_no_retry
           test_run_main_tip_sweep_internalerror_on_retry_returns_none
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.verify import VerifyResult

# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

MAIN_SHA = 'a' * 40

PASSING_RESULT = VerifyResult(
    passed=True,
    test_output='',
    lint_output='',
    type_output='',
    summary='all checks passed',
)

FAILING_RESULT = VerifyResult(
    passed=False,
    test_output='FAILED test_x',
    lint_output='',
    type_output='',
    summary='test_failure',
    cause_hint='FAILED test_x',
    category='test_failure',
)

INTERNALERROR_RESULT = VerifyResult(
    passed=False,
    test_output=(
        'INTERNALERROR> Traceback (most recent call last):\n'
        'INTERNALERROR>   KeyError: <WorkerController gw3>\n'
    ),
    lint_output='',
    type_output='',
    summary='pytest_internalerror',
    cause_hint='INTERNALERROR> KeyError: <WorkerController gw3>',
    category='pytest_internalerror',
)


def _make_config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


def _make_git_ops(tmp_path: Path, main_sha: str = MAIN_SHA) -> MagicMock:
    """Build a minimal MagicMock git_ops."""
    mock = MagicMock()
    mock.get_main_sha = AsyncMock(return_value=main_sha)
    worktree_base = tmp_path / '.worktrees'
    worktree_base.mkdir(parents=True, exist_ok=True)
    mock.worktree_base = worktree_base
    return mock


# ---------------------------------------------------------------------------
# step-3: happy path
# ---------------------------------------------------------------------------


class TestRunMainTipSweepHappyPath:
    """step-3: run_main_tip_sweep returns (main_sha, VerifyResult) on success
    AND the worktree add + remove commands were both issued."""

    def test_run_main_tip_sweep_happy_path(self, tmp_path: Path) -> None:
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        run_calls: list = []

        async def _fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return (0, '', '')

        async def _fake_full_verify(project_root, cfg, **kwargs):
            return PASSING_RESULT

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(verify_module, 'run_full_verification', side_effect=_fake_full_verify),
        ):
            result = asyncio.run(
                verify_module.run_main_tip_sweep(config, git_ops)
            )

        assert result is not None, 'Expected a (sha, VerifyResult) tuple, got None'
        swept_sha, vr = result
        assert swept_sha == MAIN_SHA, f'Expected main_sha={MAIN_SHA!r}, got {swept_sha!r}'
        assert vr.passed is True

        # Verify git worktree add was called
        add_calls = [c for c in run_calls if 'worktree' in c and 'add' in c]
        assert add_calls, 'Expected at least one git worktree add call'
        add_cmd = add_calls[0]
        assert '--detach' in add_cmd, f'Expected --detach in worktree add cmd: {add_cmd}'
        assert MAIN_SHA in add_cmd, f'Expected main_sha in worktree add cmd: {add_cmd}'

        # Verify git worktree remove was called (cleanup ran)
        remove_calls = [c for c in run_calls if 'worktree' in c and 'remove' in c]
        assert remove_calls, 'Expected a git worktree remove --force call (cleanup)'
        remove_cmd = remove_calls[0]
        assert '--force' in remove_cmd, f'Expected --force in worktree remove cmd: {remove_cmd}'


# ---------------------------------------------------------------------------
# step-5: fail-safe paths + drift passthrough
# ---------------------------------------------------------------------------


class TestRunMainTipSweepFailSafes:
    """step-5: None sentinel returned on infra failures; drift passed through."""

    def test_run_main_tip_sweep_failsafe_empty_sha(self, tmp_path: Path) -> None:
        """When get_main_sha returns '', run_main_tip_sweep returns None and
        neither run_full_verification nor git worktree add is called."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path, main_sha='')

        full_verify_called = []

        async def _fake_run(cmd, **kwargs):
            full_verify_called.append(('_run', cmd))
            return (0, '', '')

        async def _fake_full_verify(*args, **kwargs):
            full_verify_called.append(('full_verify',))
            return PASSING_RESULT

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(verify_module, 'run_full_verification', side_effect=_fake_full_verify),
        ):
            result = asyncio.run(
                verify_module.run_main_tip_sweep(config, git_ops)
            )

        assert result is None, f'Expected None on empty SHA, got {result!r}'
        worktree_adds = [x for x in full_verify_called if x[0] == '_run' and 'add' in x[1]]
        assert not worktree_adds, 'git worktree add should NOT be called when sha is empty'
        full_verifies = [x for x in full_verify_called if x[0] == 'full_verify']
        assert not full_verifies, 'run_full_verification should NOT be called when sha is empty'

    def test_run_main_tip_sweep_failsafe_worktree_add_fails(self, tmp_path: Path) -> None:
        """When git worktree add fails for every retry, returns None and
        run_full_verification is NOT called."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        full_verify_called = []

        async def _fake_run(cmd, **kwargs):
            # Always fail worktree add; succeed on remove (shouldn't be called but guard)
            if 'add' in cmd:
                return (1, '', 'lock contention')
            return (0, '', '')

        async def _fake_full_verify(*args, **kwargs):
            full_verify_called.append(True)
            return PASSING_RESULT

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(verify_module, 'run_full_verification', side_effect=_fake_full_verify),
        ):
            result = asyncio.run(
                verify_module.run_main_tip_sweep(config, git_ops)
            )

        assert result is None, f'Expected None when worktree add fails, got {result!r}'
        assert not full_verify_called, 'run_full_verification should NOT be called on add failure'

    def test_run_main_tip_sweep_drift_passthrough(self, tmp_path: Path) -> None:
        """When run_full_verification returns a failing result, run_main_tip_sweep
        returns (main_sha, failing_result) AND git worktree remove still ran (cleanup-on-failure)."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        run_calls: list = []

        async def _fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return (0, '', '')

        async def _fake_full_verify(*args, **kwargs):
            return FAILING_RESULT

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(verify_module, 'run_full_verification', side_effect=_fake_full_verify),
        ):
            result = asyncio.run(
                verify_module.run_main_tip_sweep(config, git_ops)
            )

        assert result is not None, 'Expected (sha, VerifyResult) even on drift, got None'
        swept_sha, vr = result
        assert swept_sha == MAIN_SHA
        assert vr.passed is False
        assert vr.category == 'test_failure'

        # Cleanup must run even when verify fails
        remove_calls = [c for c in run_calls if 'worktree' in c and 'remove' in c]
        assert remove_calls, 'git worktree remove should run even when verify fails (cleanup-on-failure)'

    def test_run_main_tip_sweep_internalerror_returns_none(self, tmp_path: Path) -> None:
        """When run_full_verification returns category='pytest_internalerror',
        run_main_tip_sweep returns None (the infra sentinel) and the git worktree
        remove cleanup still ran.

        Rationale: a pytest INTERNALERROR means the xdist test infrastructure
        itself crashed (e.g. a worker process was killed by os._exit). Returning
        None routes the tick into the harness's ``outcome is None`` path — retry
        next tick, no L1 drift escalation filed, SHA not marked swept — which is
        the correct behaviour for an infra crash.

        RED today: run_main_tip_sweep returns (sha, result) for any non-None
        result regardless of category; no special-case for pytest_internalerror.
        """
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        run_calls: list = []

        async def _fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return (0, '', '')

        async def _fake_full_verify(*args, **kwargs):
            return INTERNALERROR_RESULT

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(verify_module, 'run_full_verification', side_effect=_fake_full_verify),
        ):
            result = asyncio.run(
                verify_module.run_main_tip_sweep(config, git_ops)
            )

        assert result is None, (
            f'Expected None (infra sentinel) when category=pytest_internalerror, '
            f'got {result!r}'
        )

        # Cleanup must still run even when we return None early (finally block)
        remove_calls = [c for c in run_calls if 'worktree' in c and 'remove' in c]
        assert remove_calls, (
            'git worktree remove --force must run even when returning None for '
            'pytest_internalerror (cleanup-in-finally guarantee)'
        )


# ---------------------------------------------------------------------------
# task-1925 step-1: retry-on-flake-once (Part B)
# ---------------------------------------------------------------------------


class TestRunMainTipSweepRetryOnFlake:
    """task-1925 step-1: run_main_tip_sweep retries once on first-pass failure
    to distinguish transient flakiness from deterministic drift.

    RED today: current code calls run_full_verification exactly once and returns
    the first result regardless of pass/fail (no retry logic).
    """

    def test_run_main_tip_sweep_retries_once_and_suppresses_transient_flake(
        self, tmp_path: Path
    ) -> None:
        """First-pass FAIL + retry PASS → returns passing result (flake suppressed).

        RED today: returns (sha, FAILING_RESULT) after exactly one call.
        GREEN after impl: returns (sha, PASSING_RESULT) after exactly two calls.
        """
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        async def _fake_run(cmd, **kwargs):
            return (0, '', '')

        rfv = AsyncMock(side_effect=[FAILING_RESULT, PASSING_RESULT])

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(verify_module, 'run_full_verification', rfv),
        ):
            result = asyncio.run(
                verify_module.run_main_tip_sweep(config, git_ops)
            )

        assert result is not None, 'Expected (sha, VerifyResult), got None'
        swept_sha, vr = result
        assert swept_sha == MAIN_SHA
        assert vr.passed is True, (
            f'Expected passing result after retry suppressed transient flake, '
            f'got passed={vr.passed!r}'
        )
        assert rfv.call_count == 2, (
            f'Expected exactly 2 calls to run_full_verification (initial + retry), '
            f'got {rfv.call_count}'
        )

    def test_run_main_tip_sweep_both_passes_fail_is_drift(
        self, tmp_path: Path
    ) -> None:
        """First-pass FAIL + retry FAIL → returns failing result (deterministic drift).

        Both passes fail → the failure is real drift, not a transient flake.
        Harness must still receive a failing result so it can file the L1 escalation.
        """
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        async def _fake_run(cmd, **kwargs):
            return (0, '', '')

        rfv = AsyncMock(side_effect=[FAILING_RESULT, FAILING_RESULT])

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(verify_module, 'run_full_verification', rfv),
        ):
            result = asyncio.run(
                verify_module.run_main_tip_sweep(config, git_ops)
            )

        assert result is not None, 'Expected (sha, VerifyResult) even on drift, got None'
        swept_sha, vr = result
        assert swept_sha == MAIN_SHA
        assert vr.passed is False, 'Expected failing result — deterministic drift still escalates'
        assert vr.category == 'test_failure'
        assert rfv.call_count == 2, (
            f'Expected exactly 2 calls to run_full_verification (initial + retry), '
            f'got {rfv.call_count}'
        )

    def test_run_main_tip_sweep_passes_first_time_no_retry(
        self, tmp_path: Path
    ) -> None:
        """First-pass PASS → returns passing result; run_full_verification called once.

        No needless retry when the first pass succeeds.
        """
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        async def _fake_run(cmd, **kwargs):
            return (0, '', '')

        rfv = AsyncMock(side_effect=[PASSING_RESULT])

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(verify_module, 'run_full_verification', rfv),
        ):
            result = asyncio.run(
                verify_module.run_main_tip_sweep(config, git_ops)
            )

        assert result is not None
        swept_sha, vr = result
        assert swept_sha == MAIN_SHA
        assert vr.passed is True
        assert rfv.call_count == 1, (
            f'Expected exactly 1 call (no needless retry on first-pass success), '
            f'got {rfv.call_count}'
        )

    def test_run_main_tip_sweep_internalerror_on_retry_returns_none(
        self, tmp_path: Path
    ) -> None:
        """First-pass FAIL + retry INTERNALERROR → returns None (infra sentinel).

        pytest INTERNALERROR on the retry means the xdist infrastructure crashed.
        Must return None (retry next tick, no false-positive drift escalation).
        The worktree cleanup must still run (finally block).

        RED today: returns (sha, FAILING_RESULT) after one call (no retry logic).
        """
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        run_calls: list = []

        async def _fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return (0, '', '')

        rfv = AsyncMock(side_effect=[FAILING_RESULT, INTERNALERROR_RESULT])

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(verify_module, 'run_full_verification', rfv),
        ):
            result = asyncio.run(
                verify_module.run_main_tip_sweep(config, git_ops)
            )

        assert result is None, (
            f'Expected None (infra sentinel) when retry returns pytest_internalerror, '
            f'got {result!r}'
        )
        # Cleanup must still run even when we return None (finally block guarantee)
        remove_calls = [c for c in run_calls if 'worktree' in c and 'remove' in c]
        assert remove_calls, (
            'git worktree remove --force must run even when retry returns '
            'pytest_internalerror (cleanup-in-finally guarantee)'
        )
