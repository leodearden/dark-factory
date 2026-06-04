"""Tests for verify_failure_is_preexisting_on_main — the broken-main contagion helper.

The helper detects whether a verify failure that occurred on a task branch
already exists on main (i.e. was inherited from the rebase target, not
introduced by this task).  Tests are organised as:

  1. Happy path: same signature on main -> True
  2. False cases: main passes / different signature -> False
  3. Cheapness refinement: only non-flaky categories reach the helper
  4. Lifecycle / cleanup: temp worktree create + remove in finally, no task-worktree mutation
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

MAIN_SHA = 'aabbcc1122334455'

FAILING_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='error TS2769: foo.tsx:12',
    summary='TS2769 compile_error',
    cause_hint='error TS2769: foo.tsx:12',
    category='compile_error',
)

SAME_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='error TS2769: foo.tsx:12',
    summary='TS2769 compile_error',
    cause_hint='error TS2769: foo.tsx:12',
    category='compile_error',
)

PASSING_RESULT = VerifyResult(
    passed=True,
    test_output='',
    lint_output='',
    type_output='',
    summary='all checks passed',
)

DIFFERENT_RESULT = VerifyResult(
    passed=False,
    test_output='FAILED test_bar',
    lint_output='',
    type_output='',
    summary='test_failure',
    cause_hint='AssertionError in test_bar',
    category='test_failure',
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


# ---------------------------------------------------------------------------
# Test 1 — happy path: same failure signature on main -> True
# ---------------------------------------------------------------------------


class TestVerifyFailureIsPreexistingOnMain:
    """Step-1 / test-expectation #1: same (category, cause_hint) on main -> True."""

    def test_returns_true_when_same_signature_on_main(self, tmp_path: Path) -> None:
        """When main probe reproduces the same (category, cause_hint) -> True."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        module_configs: list = []
        task_files = ['src/foo.tsx']

        # Mock git_ops: get_main_sha returns a SHA
        mock_git_ops = MagicMock()
        mock_git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)

        # run_scoped_verification on main -> same failure
        async def _fake_verify(*args, **kwargs) -> VerifyResult:
            return SAME_RESULT

        # _run (git) -> always succeed (rc=0)
        async def _fake_run(cmd, **kwargs):
            return (0, '', '')

        with (
            patch.object(verify_module, 'run_scoped_verification', side_effect=_fake_verify),
            patch(
                'orchestrator.git_ops._run',  # lazy-imported inside the helper
                side_effect=_fake_run,
            ),
        ):
            result = asyncio.run(
                verify_module.verify_failure_is_preexisting_on_main(
                    worktree, config, module_configs, task_files,
                    FAILING_RESULT, mock_git_ops,
                )
            )

        assert result == (True, MAIN_SHA), (
            f'Expected (True, {MAIN_SHA!r}) — same signature reproduced on main means inherited; got {result!r}'
        )


# ---------------------------------------------------------------------------
# Test 2 — false cases: main passes -> False; different signature -> False
# ---------------------------------------------------------------------------


class TestVerifyFailureIsPreexistingFalseCases:
    """Step-3 / test-expectation #2 + invariant a: non-inherited cases return False."""

    def _run_helper(
        self, tmp_path: Path, failing_result: VerifyResult, main_result: VerifyResult,
    ) -> tuple[bool, str]:
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()

        mock_git_ops = MagicMock()
        mock_git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)

        async def _fake_verify(*args, **kwargs) -> VerifyResult:
            return main_result

        async def _fake_run(cmd, **kwargs):
            return (0, '', '')

        with (
            patch.object(verify_module, 'run_scoped_verification', side_effect=_fake_verify),
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
        ):
            return asyncio.run(
                verify_module.verify_failure_is_preexisting_on_main(
                    worktree, config, [], ['src/foo.tsx'], failing_result, mock_git_ops,
                )
            )

    def test_returns_false_when_main_passes(self, tmp_path: Path) -> None:
        """Main probe passes -> break is task-own -> False.

        Invariant (a): a sibling's already-landed hotfix makes main clean,
        so the helper correctly reports 'not inherited' and lets the debugger run.
        """
        result = self._run_helper(tmp_path, FAILING_RESULT, PASSING_RESULT)
        is_preexisting, probe_sha = result
        assert not is_preexisting, (
            'Main passes after sibling hotfix — should return (False, ...) so debugger runs.'
        )

    def test_returns_false_when_different_signature_on_main(self, tmp_path: Path) -> None:
        """Main fails with a DIFFERENT (category, cause_hint) -> different break -> False."""
        result = self._run_helper(tmp_path, FAILING_RESULT, DIFFERENT_RESULT)
        is_preexisting, probe_sha = result
        assert not is_preexisting, (
            'Different failure signature on main — not the same inherited break, should return (False, ...).'
        )


# ---------------------------------------------------------------------------
# Test 3 — lifecycle: temp worktree created + removed in finally, no task-wt mutation
# ---------------------------------------------------------------------------


class TestVerifyFailureIsPreexistingLifecycle:
    """Step-5 / test-expectation #6: cleanup runs even when probe raises; task-wt untouched."""

    def test_worktree_add_and_remove_are_called(self, tmp_path: Path) -> None:
        """A ``git worktree add --detach <tmp> <sha>`` and matching remove are issued."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        mock_git_ops = MagicMock()
        mock_git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)

        run_calls: list[list[str]] = []

        async def _fake_run(cmd, **kwargs):
            run_calls.append(list(cmd))
            return (0, '', '')

        with (
            # AsyncMock is required because run_scoped_verification is awaited;
            # a plain return_value=SAME_RESULT would produce a TypeError that the
            # helper's except-swallows, exercising the error path instead of the
            # intended happy probe path.
            patch.object(verify_module, 'run_scoped_verification',
                         new=AsyncMock(return_value=SAME_RESULT)),
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
        ):
            asyncio.run(
                verify_module.verify_failure_is_preexisting_on_main(
                    worktree, config, [], ['src/foo.tsx'], FAILING_RESULT, mock_git_ops,
                )
            )

        add_calls = [c for c in run_calls if 'worktree' in c and 'add' in c]
        remove_calls = [c for c in run_calls if 'worktree' in c and 'remove' in c]
        assert len(add_calls) == 1, f'Expected 1 worktree add; got: {add_calls}'
        assert len(remove_calls) == 1, f'Expected 1 worktree remove; got: {remove_calls}'
        add_cmd = add_calls[0]
        assert '--detach' in add_cmd, f'worktree add must use --detach: {add_cmd}'
        assert MAIN_SHA in add_cmd, f'worktree add must target main_sha: {add_cmd}'
        # The task worktree path must NOT appear in any _run invocation
        for c in run_calls:
            assert str(worktree) not in c, (
                f'Task worktree path leaked into git command: {c}'
            )

    def test_cleanup_runs_even_when_probe_raises(self, tmp_path: Path) -> None:
        """Cleanup (worktree remove + rmtree) runs even when run_scoped_verification raises."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        mock_git_ops = MagicMock()
        mock_git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)

        run_calls: list[list[str]] = []

        async def _fake_run(cmd, **kwargs):
            run_calls.append(list(cmd))
            return (0, '', '')

        async def _raising_verify(*args, **kwargs):
            raise RuntimeError('simulated probe crash')

        with (
            patch.object(verify_module, 'run_scoped_verification', side_effect=_raising_verify),
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
        ):
            result = asyncio.run(
                verify_module.verify_failure_is_preexisting_on_main(
                    worktree, config, [], ['src/foo.tsx'], FAILING_RESULT, mock_git_ops,
                )
            )

        # Fail-safe: exception does not propagate; returns (False, '') tuple
        is_preexisting, probe_sha = result
        assert not is_preexisting, 'Probe exception must return (False, ...) (fail-safe), not propagate.'
        remove_calls = [c for c in run_calls if 'worktree' in c and 'remove' in c]
        assert len(remove_calls) == 1, (
            f'Cleanup must run even when probe raises; remove calls: {remove_calls}'
        )


# ---------------------------------------------------------------------------
# Test 4 — step-17: probe worktree is created under git_ops.worktree_base,
#           NOT the system temp dir (environment-parity invariant)
# ---------------------------------------------------------------------------


class TestVerifyFailureProbeWorktreePlacement:
    """Step-17 / review-fix: probe worktree must be under git_ops.worktree_base.

    A /tmp probe cannot resolve node_modules / repo-root shared installs by
    upward directory traversal, so an inherited TS/compile break would surface
    a DIFFERENT signature ('Cannot find module' / 'tsc not found') — branch_sig
    != main_sig — and the contagion guard would silently never fire.

    Placement under worktree_base restores identical upward resolution to task
    worktrees.  Prune-safety is achieved via the '_mainprobe-' prefix (the disk-
    pressure prune targets '_merge-*' only).
    """

    def test_probe_path_is_under_worktree_base_not_tmp(self, tmp_path: Path) -> None:
        """The 'git worktree add --detach' target must be a child of git_ops.worktree_base."""
        import tempfile
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()

        # worktree_base must be a real Path (not auto-MagicMock) so is_relative_to() works.
        worktree_base = config.project_root / '.worktrees'
        worktree_base.mkdir(parents=True, exist_ok=True)

        mock_git_ops = MagicMock()
        mock_git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)
        mock_git_ops.worktree_base = worktree_base  # real Path — pins the placement invariant

        probe_paths: list[str] = []

        async def _spy_run(cmd, **kwargs):
            if 'worktree' in cmd and 'add' in cmd and '--detach' in cmd:
                detach_idx = cmd.index('--detach')
                probe_paths.append(cmd[detach_idx + 1])
            return (0, '', '')

        with (
            patch.object(verify_module, 'run_scoped_verification',
                         new=AsyncMock(return_value=SAME_RESULT)),
            patch('orchestrator.git_ops._run', side_effect=_spy_run),
        ):
            asyncio.run(
                verify_module.verify_failure_is_preexisting_on_main(
                    worktree, config, [], ['src/foo.tsx'], FAILING_RESULT, mock_git_ops,
                )
            )

        assert len(probe_paths) == 1, f'Expected exactly 1 worktree add call; got: {probe_paths}'
        probe_path = Path(probe_paths[0])

        # MUST be under worktree_base — not in /tmp or any system temp dir
        assert probe_path.is_relative_to(worktree_base), (
            f'Probe path {probe_path!r} must be under worktree_base={worktree_base} '
            f'for environment parity (upward dependency resolution). '
            f'Current parent: {probe_path.parent}'
        )
        assert not probe_path.is_relative_to(Path(tempfile.gettempdir())), (
            f'Probe path {probe_path!r} must NOT be under system temp dir '
            f'{tempfile.gettempdir()} — /tmp probes fail to resolve node_modules upward'
        )
