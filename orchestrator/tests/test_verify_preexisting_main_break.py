"""Tests for verify_failure_is_preexisting_on_main — the broken-main contagion helper.

The helper detects whether a verify failure that occurred on a task branch
already exists on main (i.e. was inherited from the rebase target, not
introduced by this task).  Tests are organised as:

  1. Happy path: same signature on main -> True
  2. False cases: main passes / different signature -> False
  3. Cheapness refinement: only non-flaky categories reach the helper
  4. Lifecycle / cleanup: probe worktree create + remove in finally, no task-wt mutation
  5. Placement: probe is under git_ops.worktree_base, not the system temp dir
  6. Integration: real git repo + real GitOps; only run_scoped_verification is stubbed
"""
from __future__ import annotations

import asyncio
import subprocess
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

        # Mock git_ops: get_main_sha returns a SHA; worktree_base is a real Path
        # so the helper builds a valid probe path under it (env-parity invariant).
        mock_git_ops = MagicMock()
        mock_git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)
        worktree_base = tmp_path / '.worktrees'
        worktree_base.mkdir(parents=True, exist_ok=True)
        mock_git_ops.worktree_base = worktree_base

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
        worktree_base = tmp_path / '.worktrees'
        worktree_base.mkdir(parents=True, exist_ok=True)
        mock_git_ops.worktree_base = worktree_base

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
        worktree_base = tmp_path / '.worktrees'
        worktree_base.mkdir(parents=True, exist_ok=True)
        mock_git_ops.worktree_base = worktree_base

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
        worktree_base = tmp_path / '.worktrees'
        worktree_base.mkdir(parents=True, exist_ok=True)
        mock_git_ops.worktree_base = worktree_base

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

        # MUST be a direct child of worktree_base with the '_mainprobe-' prefix.
        # This pins the environment-parity invariant: task worktrees live at
        # worktree_base/<name> and resolve node_modules / repo-root installs by
        # upward traversal that reaches project_root.  A probe at an independent
        # mkdtemp path (old behaviour: /tmp/df-mainprobe-xxx/probe) cannot reach
        # project_root via upward traversal, so inherited TS/compile breaks surface
        # a different signature ('Cannot find module') and the guard silently never fires.
        assert probe_path.parent == worktree_base, (
            f'Probe path {probe_path!r} must be a direct child of '
            f'worktree_base={worktree_base} (not a sub-directory of some other dir). '
            f'Actual parent: {probe_path.parent}'
        )
        assert probe_path.name.startswith('_mainprobe-'), (
            f"Probe dir name {probe_path.name!r} must start with '_mainprobe-' "
            f"(distinct from '_merge-*' so the disk-pressure prune never reclaims it mid-run)"
        )
        # Belt check: the probe is NOT the task worktree itself or a sub-path of it
        assert probe_path != worktree, f'Probe must differ from task worktree {worktree}'


# ---------------------------------------------------------------------------
# Test 6 — step-19: integration test with a REAL git repo and a REAL GitOps
# ---------------------------------------------------------------------------


class TestVerifyFailureIsPreexistingIntegration:
    """Step-19: integration test — real git + real GitOps; only run_scoped_verification stubbed.

    This is the test that would have caught the /tmp placement bug: it exercises
    a real ``git worktree add`` (not mocked), so the probe path must physically
    live under git_ops.worktree_base for it to succeed.  A /tmp path would fail
    to resolve node_modules upward and surface a wrong signature; this test
    asserts the probe actually exists on disk inside worktree_base during verify.
    """

    def test_real_git_probe_lifecycle(self, tmp_path: Path) -> None:
        """Real git repo: probe created under worktree_base, deregistered after, task-wt intact."""
        from orchestrator import verify as verify_module
        from orchestrator.config import GitConfig, OrchestratorConfig
        from orchestrator.git_ops import GitOps

        # ── Set up a real git repo ──────────────────────────────────────────
        project_root = tmp_path / 'repo'
        project_root.mkdir()
        for cmd in [
            ['git', 'init', '-b', 'main'],
            ['git', 'config', 'user.email', 'test@test.com'],
            ['git', 'config', 'user.name', 'Test'],
        ]:
            subprocess.run(cmd, cwd=project_root, check=True, capture_output=True)
        (project_root / 'file.txt').write_text('hello')
        subprocess.run(['git', 'add', '.'], cwd=project_root, check=True, capture_output=True)
        subprocess.run(
            ['git', 'commit', '-m', 'init'], cwd=project_root, check=True, capture_output=True,
        )

        # Record the real main SHA before the probe runs
        main_sha_before = subprocess.run(
            ['git', 'rev-parse', 'main'], cwd=project_root, capture_output=True, text=True,
        ).stdout.strip()

        # ── Construct real GitOps ────────────────────────────────────────────
        config = OrchestratorConfig(
            project_root=project_root,
            max_concurrent_tasks=1,
            git=GitConfig(
                main_branch='main',
                branch_prefix='task/',
                remote='origin',
                worktree_dir='.worktrees',
            ),
        )
        git_ops = GitOps(config.git, project_root)

        # Create a task worktree directory (placeholder; helper reads its path, not content)
        task_wt = git_ops.worktree_base / 'task-123'
        task_wt.mkdir(parents=True, exist_ok=True)

        # ── Stub ONLY run_scoped_verification ────────────────────────────────
        # The stub asserts from within that the probe physically exists on disk
        # and is under worktree_base — exactly what the /tmp-path bug prevented.
        probe_worktrees_seen: list[Path] = []

        async def _stub_verify(worktree_arg: Path, *args, **kwargs) -> VerifyResult:
            # Probe must exist on disk (git worktree add created it)
            assert worktree_arg.exists(), (
                f'Probe worktree {worktree_arg} must physically exist during verify; '
                f'a /tmp path may fail git worktree add for the same git repo'
            )
            # Probe must be under worktree_base for env-parity
            assert worktree_arg.is_relative_to(git_ops.worktree_base), (
                f'Probe {worktree_arg} must be under worktree_base={git_ops.worktree_base}'
            )
            probe_worktrees_seen.append(worktree_arg)
            return SAME_RESULT

        with patch.object(verify_module, 'run_scoped_verification', side_effect=_stub_verify):
            result = asyncio.run(
                verify_module.verify_failure_is_preexisting_on_main(
                    task_wt, config, [], ['file.txt'], FAILING_RESULT, git_ops,
                )
            )

        # (a) Helper returns (True, main_sha_before)
        is_preexisting, probe_sha = result
        assert is_preexisting, f'Expected (True, ...) — same signature on main; got {result!r}'
        assert probe_sha == main_sha_before, (
            f'Returned sha {probe_sha!r} must match real main SHA {main_sha_before!r}'
        )

        # (b) Probe dir is gone after return; git worktree list shows no _mainprobe-* leftover
        assert len(probe_worktrees_seen) == 1, 'Exactly one probe verify call expected'
        probe_path = probe_worktrees_seen[0]
        assert not probe_path.exists(), (
            f'Probe dir {probe_path} must be removed by cleanup (git worktree remove + rmtree)'
        )
        wt_list = subprocess.run(
            ['git', 'worktree', 'list', '--porcelain'],
            cwd=project_root, capture_output=True, text=True,
        ).stdout
        assert '_mainprobe-' not in wt_list, (
            f'No _mainprobe-* worktrees should remain registered after cleanup:\n{wt_list}'
        )

        # (c) main SHA unchanged — helper never modified the repo state
        main_sha_after = subprocess.run(
            ['git', 'rev-parse', 'main'], cwd=project_root, capture_output=True, text=True,
        ).stdout.strip()
        assert main_sha_after == main_sha_before, (
            f'main SHA changed from {main_sha_before} to {main_sha_after}; '
            f'helper must never modify the repo or task worktree'
        )
