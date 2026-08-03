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
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, WorktreeKind
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

        # Real GitOps (behavior-preserving swap from MagicMock — the helper
        # only touches get_main_sha/worktree_base/module-level _run, all
        # present on a real instance) with get_main_sha patched.
        mock_git_ops = GitOps(config.git, config.project_root)
        mock_git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)  # type: ignore[method-assign]
        mock_git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

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

        mock_git_ops = GitOps(config.git, config.project_root)
        mock_git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)  # type: ignore[method-assign]
        mock_git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

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
        mock_git_ops = GitOps(config.git, config.project_root)
        mock_git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)  # type: ignore[method-assign]
        mock_git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

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
        mock_git_ops = GitOps(config.git, config.project_root)
        mock_git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)  # type: ignore[method-assign]
        mock_git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

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

        # Real GitOps so git_ops.worktree_base is the actual resolved value
        # (not a manually-built parallel path) — pins the placement invariant.
        mock_git_ops = GitOps(config.git, config.project_root)
        mock_git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)  # type: ignore[method-assign]
        mock_git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

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
        assert probe_path.parent == mock_git_ops.worktree_base, (
            f'Probe path {probe_path!r} must be a direct child of '
            f'worktree_base={mock_git_ops.worktree_base} (not a sub-directory of some other dir). '
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


# ---------------------------------------------------------------------------
# Test 7 — task 2567: probe warm-seeds its target/ via ephemeral_worktree
# ---------------------------------------------------------------------------


class TestVerifyFailureProbeWarmSeed:
    """task 2567: the probe's ephemeral_worktree call passes warm_seed=True
    (CoW-seeding target/ from the shared warm-lane base instead of a cold
    from-scratch build), stays in the warm verify-timeout tier (no .task/,
    is_merge_verify not forced True), and degrades fail-soft to cold on a
    seed fault without changing the verdict.

    RED today: verify.py's probe CM call passes no warm_seed kwarg, so the
    spy sees warm_seed absent (falls back to ephemeral_worktree's own
    warm_seed=False default) instead of True.
    """

    def test_probe_calls_ephemeral_worktree_with_warm_seed_true(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        git_ops = GitOps(config.git, config.project_root)
        git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)  # type: ignore[method-assign]
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

        async def _fake_run(cmd, **kwargs):
            return (0, '', '')

        with (
            patch.object(verify_module, 'run_scoped_verification',
                         new=AsyncMock(return_value=SAME_RESULT)),
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(
                git_ops, 'ephemeral_worktree', wraps=git_ops.ephemeral_worktree,
            ) as spied_cm,
        ):
            asyncio.run(
                verify_module.verify_failure_is_preexisting_on_main(
                    worktree, config, [], ['src/foo.tsx'], FAILING_RESULT, git_ops,
                )
            )

        assert spied_cm.call_count == 1, (
            f'expected exactly 1 ephemeral_worktree call; got {spied_cm.call_count}'
        )
        call_args = spied_cm.call_args.args
        call_kwargs = spied_cm.call_args.kwargs
        assert call_args and call_args[0] is WorktreeKind.MAIN_PROBE, (
            f'expected the probe to mint a MAIN_PROBE-kind worktree; got args={call_args!r}'
        )
        assert call_kwargs.get('warm_seed') is True, (
            f'expected verify_failure_is_preexisting_on_main to call '
            f'ephemeral_worktree(..., warm_seed=True); got kwargs={call_kwargs!r}'
        )

    def test_probe_stays_in_warm_timeout_tier(self, tmp_path: Path) -> None:
        """Warm-tier regression: no .task/ dir and is_merge_verify is not
        forced True, so the probe resolves the WARM verify-timeout budget
        (matching its now-warm-seeded build) rather than the cold one."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        git_ops = GitOps(config.git, config.project_root)
        git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)  # type: ignore[method-assign]
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

        captured: dict[str, object] = {}

        async def _fake_run(cmd, **kwargs):
            return (0, '', '')

        async def _capturing_verify(worktree_arg, *args, **kwargs) -> VerifyResult:
            captured['worktree'] = worktree_arg
            captured['kwargs'] = kwargs
            return SAME_RESULT

        with (
            patch.object(verify_module, 'run_scoped_verification', side_effect=_capturing_verify),
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
        ):
            asyncio.run(
                verify_module.verify_failure_is_preexisting_on_main(
                    worktree, config, [], ['src/foo.tsx'], FAILING_RESULT, git_ops,
                )
            )

        assert 'worktree' in captured, 'expected run_scoped_verification to have been called'
        captured_worktree = captured['worktree']
        assert isinstance(captured_worktree, Path)
        assert verify_module._is_verify_cold(captured_worktree, None) is False, (
            'expected the probe worktree (no .task/ dir) to resolve WARM in '
            '_is_verify_cold — a future edit must not flip this to cold'
        )
        captured_kwargs = captured['kwargs']
        assert isinstance(captured_kwargs, dict)
        assert captured_kwargs.get('is_merge_verify') is not True, (
            'expected the probe to NOT force is_merge_verify=True (that would '
            'push it onto the cold verify-timeout tier)'
        )

    def test_probe_verdict_unchanged_when_seed_fails_soft(self, tmp_path: Path) -> None:
        """End-to-end fail-soft: base resolvable OK on disk, but
        _seed_warm_lane itself faults (rc=75) — the probe must still
        degrade to cold and produce the SAME verdict as before this task
        (same-signature main -> (True, MAIN_SHA))."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        git_ops = GitOps(config.git, config.project_root)
        git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)  # type: ignore[method-assign]
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

        # A resolvable warm base on disk (OK), so the seed subprocess is
        # actually attempted rather than short-circuited by the base gate.
        base = git_ops.warm_lane_base_target_path
        base.mkdir(parents=True, exist_ok=True)
        (base / 'sentinel').write_text('base content')

        async def _fake_run(cmd, **kwargs):
            return (0, '', '')

        with (
            patch.object(verify_module, 'run_scoped_verification',
                         new=AsyncMock(return_value=SAME_RESULT)),
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(git_ops, '_seed_warm_lane', new=AsyncMock(return_value=75)) as mock_seed,
        ):
            result = asyncio.run(
                verify_module.verify_failure_is_preexisting_on_main(
                    worktree, config, [], ['src/foo.tsx'], FAILING_RESULT, git_ops,
                )
            )

        mock_seed.assert_awaited_once()
        assert result == (True, MAIN_SHA), (
            f'expected the verdict to be unchanged by a seed fault '
            f'(fail-soft to cold); got {result!r}'
        )


# ---------------------------------------------------------------------------
# Test 8 — step-11 (task μ, verify-scope-inversion-prd.md): pure baseline-diff
#          helpers, diff_new_failures / is_wholly_preexisting — the decision
#          core of B1.  Unlike the rest of this file these are plain
#          synchronous unit tests: no worktree, no GitOps, no async — the
#          helpers are pure set operations over failing-test-id iterables.
# ---------------------------------------------------------------------------


class TestBaselineDiffHelpers:
    """diff_new_failures(branch, baseline) is set difference (branch - baseline);
    is_wholly_preexisting(branch, baseline) is True iff branch is non-empty and
    every id in branch also appears in baseline."""

    def test_diff_new_failures_is_set_difference(self) -> None:
        from orchestrator.verify import diff_new_failures

        assert diff_new_failures(['x::1', 'y::2'], ['x::1']) == frozenset({'y::2'})

    def test_diff_new_failures_row4_shape(self) -> None:
        """row-4: branch={X,Y}, baseline={X} -> new={Y} (mixed: one new id, one preexisting)."""
        from orchestrator.verify import diff_new_failures

        assert diff_new_failures({'X', 'Y'}, {'X'}) == frozenset({'Y'})

    def test_diff_new_failures_row5_shape(self) -> None:
        """row-5: branch={X}, baseline={X,Z} -> new=set() (branch wholly covered by baseline)."""
        from orchestrator.verify import diff_new_failures

        assert diff_new_failures({'X'}, {'X', 'Z'}) == frozenset()

    def test_diff_new_failures_empty_branch_is_empty(self) -> None:
        from orchestrator.verify import diff_new_failures

        assert diff_new_failures([], ['X']) == frozenset()

    def test_is_wholly_preexisting_true_for_row5_shape(self) -> None:
        """row-5: branch subset-of baseline and branch non-empty -> True."""
        from orchestrator.verify import is_wholly_preexisting

        assert is_wholly_preexisting({'X'}, {'X', 'Z'}) is True

    def test_is_wholly_preexisting_true_when_baseline_exactly_matches(self) -> None:
        from orchestrator.verify import is_wholly_preexisting

        assert is_wholly_preexisting({'X'}, {'X'}) is True

    def test_is_wholly_preexisting_false_for_row4_shape(self) -> None:
        """row-4: branch has a new id (Y) absent from baseline -> False."""
        from orchestrator.verify import is_wholly_preexisting

        assert is_wholly_preexisting({'X', 'Y'}, {'X'}) is False

    def test_is_wholly_preexisting_false_when_branch_has_any_id_absent_from_baseline(self) -> None:
        from orchestrator.verify import is_wholly_preexisting

        assert is_wholly_preexisting({'a::1', 'b::2'}, {'a::1'}) is False

    def test_is_wholly_preexisting_false_for_empty_branch(self) -> None:
        """An empty branch (no failures at all) is trivially False -- nothing to attribute."""
        from orchestrator.verify import is_wholly_preexisting

        assert is_wholly_preexisting([], ['X', 'Y']) is False


# ---------------------------------------------------------------------------
# Test 9 — step-13 (task μ, verify-scope-inversion-prd.md): per-main-SHA
#          baseline cache + probe — seed_main_baseline / main_baseline_failing_ids.
# ---------------------------------------------------------------------------


async def _fake_git_run(cmd, **kwargs):
    return (0, '', '')


class TestMainBaselineFailingIds:
    """seed_main_baseline seeds the cache for free (B2, every successful gate
    run); main_baseline_failing_ids is cache-first and probes (ONE full-suite,
    merge-role run, no task_files scoping) only on a genuine miss."""

    @pytest.fixture(autouse=True)
    def _clear_baseline_cache(self):
        """Scoped to this class only — the cache doesn't exist before step-14
        lands, so an autouse fixture at module scope would error out every
        already-passing test in this file during this step's RED phase."""
        from orchestrator.verify import _BASELINE_FAILING_IDS_CACHE
        _BASELINE_FAILING_IDS_CACHE.clear()
        yield
        _BASELINE_FAILING_IDS_CACHE.clear()

    def test_seeded_baseline_is_returned_without_probing(self, tmp_path: Path) -> None:
        """seed_main_baseline(sha, ids) primes the cache; a subsequent
        main_baseline_failing_ids for that sha must be served from cache —
        run_scoped_verification/ephemeral_worktree must NOT be invoked."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = GitOps(config.git, config.project_root)
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

        verify_module.seed_main_baseline(MAIN_SHA, frozenset())

        async def _explode(*args, **kwargs):
            raise AssertionError('probe must not run on a cache hit')

        with (
            patch.object(verify_module, 'run_scoped_verification', side_effect=_explode),
            patch.object(git_ops, 'ephemeral_worktree', side_effect=_explode),
        ):
            result = asyncio.run(
                verify_module.main_baseline_failing_ids(config, [], git_ops, MAIN_SHA)
            )

        assert result == frozenset(), f'expected the seeded (empty) baseline; got {result!r}'

    def test_cache_miss_probes_once_full_suite_merge_role_and_caches(self, tmp_path: Path) -> None:
        """A cold sha invokes exactly ONE probe — full-suite (no task_files
        scoping) and merge-role — and the result is cached: a second call
        for the same sha does not re-probe."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = GitOps(config.git, config.project_root)
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

        probe_result = VerifyResult(
            passed=False, test_output='', lint_output='', type_output='',
            summary='Failures: tests failed', failing_test_ids=['m::1'],
        )
        probe_calls: list[dict] = []

        async def _fake_verify(*args, **kwargs) -> VerifyResult:
            probe_calls.append(kwargs)
            return probe_result

        with (
            patch.object(verify_module, 'run_scoped_verification', side_effect=_fake_verify),
            patch('orchestrator.git_ops._run', side_effect=_fake_git_run),
        ):
            result = asyncio.run(
                verify_module.main_baseline_failing_ids(config, [], git_ops, MAIN_SHA)
            )
            assert result == frozenset({'m::1'}), (
                f'expected the probe-derived id set; got {result!r}'
            )
            assert len(probe_calls) == 1, f'expected exactly 1 probe call; got {len(probe_calls)}'
            assert probe_calls[0].get('role') == 'merge', (
                f"expected a merge-role probe; got kwargs={probe_calls[0]!r}"
            )
            assert not probe_calls[0].get('task_files'), (
                f'expected no task_files scoping (full-suite); got kwargs={probe_calls[0]!r}'
            )

            # Second call for the SAME sha: served from cache, no re-probe.
            result2 = asyncio.run(
                verify_module.main_baseline_failing_ids(config, [], git_ops, MAIN_SHA)
            )
        assert result2 == frozenset({'m::1'})
        assert len(probe_calls) == 1, 'second call must be served from cache, no re-probe'

    def test_probe_yielding_no_failing_test_ids_degrades_to_none_and_is_not_cached(
        self, tmp_path: Path,
    ) -> None:
        """A probe result with failing_test_ids=None (OPAQUE / unreadable junit)
        degrades to None (B3) and must NOT be cached — a transient hiccup
        shouldn't pin a falsely-empty baseline for the TTL window."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = GitOps(config.git, config.project_root)
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

        opaque_result = VerifyResult(
            passed=False, test_output='', lint_output='', type_output='',
            summary='opaque failure', failing_test_ids=None,
        )
        probe_calls: list[dict] = []

        async def _fake_verify(*args, **kwargs) -> VerifyResult:
            probe_calls.append(kwargs)
            return opaque_result

        with (
            patch.object(verify_module, 'run_scoped_verification', side_effect=_fake_verify),
            patch('orchestrator.git_ops._run', side_effect=_fake_git_run),
        ):
            result = asyncio.run(
                verify_module.main_baseline_failing_ids(config, [], git_ops, 'anothersha')
            )
            assert result is None, (
                f'expected degrade (None) when the probe yields no junit ids; got {result!r}'
            )

            # Not cached: a second call for the same sha must probe again.
            asyncio.run(
                verify_module.main_baseline_failing_ids(config, [], git_ops, 'anothersha')
            )
        assert len(probe_calls) == 2, (
            f'a None probe result must not be cached — expected a re-probe; '
            f'got {len(probe_calls)} call(s)'
        )


# ---------------------------------------------------------------------------
# Test 10 — step-15 (task μ, verify-scope-inversion-prd.md): the baseline-diff
#           fork of verify_failure_is_preexisting_on_main.  When
#           failing_result.failing_test_ids is not None, the helper decides
#           via main_baseline_failing_ids + is_wholly_preexisting instead of
#           probing main and comparing (category, cause_hint).  A None
#           baseline (degrade, B3) falls through to the legacy probe path;
#           failing_test_ids=None always takes the legacy path unchanged.
# ---------------------------------------------------------------------------


class TestVerifyFailureIsPreexistingBaselineDiffFork:
    """Step-15: baseline-diff decision core wired into the shared classifier.

    Each test stubs out main_baseline_failing_ids AND run_scoped_verification
    (the legacy probe's engine) as separate AsyncMocks so assertions can pin
    exactly which path ran, rather than relying on the returned (bool, str)
    tuple alone — a swallowed-exception fallback could otherwise coincidentally
    reproduce the expected tuple for the wrong reason.
    """

    def _run(
        self, tmp_path: Path, failing_result: VerifyResult, baseline: object,
    ) -> tuple[tuple[bool, str], AsyncMock, AsyncMock]:
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        mock_git_ops = GitOps(config.git, config.project_root)
        mock_git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)  # type: ignore[method-assign]
        mock_git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

        baseline_mock = AsyncMock(return_value=baseline)
        # A legacy-probe stub that would produce a DEFINITE, DIFFERENT verdict
        # if wrongly invoked — so an accidental fallthrough to the legacy path
        # can't coincidentally reproduce the expected baseline-diff verdict.
        legacy_probe_mock = AsyncMock(return_value=DIFFERENT_RESULT)

        with (
            patch.object(verify_module, 'main_baseline_failing_ids', new=baseline_mock),
            patch.object(verify_module, 'run_scoped_verification', new=legacy_probe_mock),
        ):
            result = asyncio.run(
                verify_module.verify_failure_is_preexisting_on_main(
                    worktree, config, [], ['src/foo.tsx'], failing_result, mock_git_ops,
                )
            )
        return result, baseline_mock, legacy_probe_mock

    def test_branch_wholly_subset_of_baseline_returns_true_and_skips_legacy_probe(
        self, tmp_path: Path,
    ) -> None:
        """row-5 shape: branch={X} subset of baseline={X,Z} -> (True, main_sha);
        the legacy category+cause_hint probe must NOT run."""
        failing_result = VerifyResult(
            passed=False, test_output='', lint_output='', type_output='',
            summary='Failures: tests failed', failing_test_ids=['X'],
        )
        result, baseline_mock, legacy_probe_mock = self._run(
            tmp_path, failing_result, frozenset({'X', 'Z'}),
        )
        assert result == (True, MAIN_SHA), (
            f'branch wholly covered by baseline must route MAIN_HEALTH_RED -> '
            f'(True, main_sha); got {result!r}'
        )
        baseline_mock.assert_awaited_once()
        legacy_probe_mock.assert_not_awaited()

    def test_branch_with_new_id_returns_false_and_skips_legacy_probe(
        self, tmp_path: Path,
    ) -> None:
        """row-4 shape: branch={X,Y}, baseline={X} -> new id Y present -> (False, '');
        the legacy category+cause_hint probe must NOT run."""
        failing_result = VerifyResult(
            passed=False, test_output='', lint_output='', type_output='',
            summary='Failures: tests failed', failing_test_ids=['X', 'Y'],
        )
        result, baseline_mock, legacy_probe_mock = self._run(
            tmp_path, failing_result, frozenset({'X'}),
        )
        assert result == (False, ''), (
            f"branch with a new id not on main must blame the branch -> (False, ''); got {result!r}"
        )
        baseline_mock.assert_awaited_once()
        legacy_probe_mock.assert_not_awaited()

    def test_baseline_none_falls_back_to_legacy_probe_path(self, tmp_path: Path) -> None:
        """A None baseline (degrade signal, B3) must fall through to today's
        category+cause_hint probe — NOT short-circuit to (False, '')."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        mock_git_ops = GitOps(config.git, config.project_root)
        mock_git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)  # type: ignore[method-assign]
        mock_git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

        failing_result = VerifyResult(
            passed=False, test_output='', lint_output='',
            type_output='error TS2769: foo.tsx:12',
            summary='TS2769 compile_error', cause_hint='error TS2769: foo.tsx:12',
            category='compile_error', failing_test_ids=['X'],
        )

        baseline_mock = AsyncMock(return_value=None)
        legacy_probe_mock = AsyncMock(return_value=SAME_RESULT)

        async def _fake_run(cmd, **kwargs):
            return (0, '', '')

        with (
            patch.object(verify_module, 'main_baseline_failing_ids', new=baseline_mock),
            patch.object(verify_module, 'run_scoped_verification', new=legacy_probe_mock),
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
        ):
            result = asyncio.run(
                verify_module.verify_failure_is_preexisting_on_main(
                    worktree, config, [], ['src/foo.tsx'], failing_result, mock_git_ops,
                )
            )

        assert result == (True, MAIN_SHA), (
            f'a None baseline must degrade to the legacy probe path (same '
            f'signature reproduces on main -> True); got {result!r}'
        )
        baseline_mock.assert_awaited_once()
        legacy_probe_mock.assert_awaited_once()

    def test_failing_test_ids_none_takes_legacy_path_unchanged(self, tmp_path: Path) -> None:
        """failing_test_ids=None (e.g. task-verify at workflow.py:5848) must take
        the EXISTING category+cause_hint path byte-identically; the new
        baseline-diff helper must not even be called."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        mock_git_ops = GitOps(config.git, config.project_root)
        mock_git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)  # type: ignore[method-assign]
        mock_git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

        assert FAILING_RESULT.failing_test_ids is None  # sanity: legacy fixture shape

        baseline_mock = AsyncMock(return_value=frozenset())
        legacy_probe_mock = AsyncMock(return_value=SAME_RESULT)

        async def _fake_run(cmd, **kwargs):
            return (0, '', '')

        with (
            patch.object(verify_module, 'main_baseline_failing_ids', new=baseline_mock),
            patch.object(verify_module, 'run_scoped_verification', new=legacy_probe_mock),
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
        ):
            result = asyncio.run(
                verify_module.verify_failure_is_preexisting_on_main(
                    worktree, config, [], ['src/foo.tsx'], FAILING_RESULT, mock_git_ops,
                )
            )

        assert result == (True, MAIN_SHA), (
            f'failing_test_ids=None must take the legacy probe path unchanged; got {result!r}'
        )
        baseline_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# task 3597: _main_probe_failure_is_isolated_flake — main-probe confirm gate
#
# verify._main_probe_failure_is_isolated_flake(probe_worktree, config,
# module_configs, main_result) -> list[str] | None
#
# Before verify_failure_is_preexisting_on_main returns (True, main_sha), this
# gate re-runs JUST the node-ids named in the MAIN PROBE's own failing output,
# in isolation (scoped + forced-serial + generous-timeout), inside the
# ALREADY-OPEN probe worktree pinned at main. A returned list[str] means every
# named test demonstrably passed in isolation — the caller MAY downgrade to
# (False, ''). None means every other (fail-safe) path — the caller keeps
# today's (True, main_sha) verdict.
# ---------------------------------------------------------------------------

PROBE_NODE_ID = 'tests/test_concurrent_verify_boundary.py::test_concurrent_verify_boundary'

LOAD_FLAKE_MAIN_RESULT = VerifyResult(
    passed=False,
    test_output=f'FAILED {PROBE_NODE_ID}\n[gw14] node down: Not properly terminated\n',
    lint_output='',
    type_output='',
    summary='test_failure',
    cause_hint=f'FAILED {PROBE_NODE_ID}',
    category='test_failure',
)

# On-disk layout materialized directly under the probe worktree (no real git
# checkout involved for these unit tests) so _group_node_ids_by_subproject's
# Path.exists() probes resolve PROBE_NODE_ID to the 'orchestrator' subproject.
_PROBE_CONFIRM_PROJECT_LAYOUT = {
    'orchestrator/tests/test_concurrent_verify_boundary.py': (
        'def test_concurrent_verify_boundary():\n    pass\n'
    ),
}


def _write_probe_layout(worktree: Path, layout: dict[str, str]) -> None:
    for relpath, content in layout.items():
        p = worktree / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


class TestMainProbeIsolatedFlakeConfirmGate:
    """step-1/step-3: verify._main_probe_failure_is_isolated_flake."""

    # -- step-1: POSITIVE path ---------------------------------------------

    def test_returns_node_ids_when_isolated_rerun_passes(self, tmp_path: Path) -> None:
        """All named failing tests pass on isolated re-run -> the confirmed
        node-id list, run against the SAME probe worktree (no second `git
        worktree add`).

        RED today: _main_probe_failure_is_isolated_flake does not exist.
        """
        from orchestrator import verify as verify_module
        from orchestrator.config import ModuleConfig

        config = _make_config(tmp_path)
        probe_worktree = tmp_path / 'mainprobe-wt'
        probe_worktree.mkdir()
        _write_probe_layout(probe_worktree, _PROBE_CONFIRM_PROJECT_LAYOUT)

        module_configs = [
            ModuleConfig(
                prefix='orchestrator',
                test_command=(
                    'uv run --project orchestrator --directory orchestrator '
                    'pytest tests/ --tb=short -q'
                ),
            )
        ]

        rv = AsyncMock(return_value=PASSING_RESULT)

        def _fail_if_called(cmd, **kwargs):
            raise AssertionError(
                f'orchestrator.git_ops._run should not be called (same-tree '
                f'gate, no second worktree) — got {cmd!r}'
            )

        with (
            patch.object(verify_module, 'run_verification', rv),
            patch('orchestrator.git_ops._run', side_effect=_fail_if_called),
        ):
            result = asyncio.run(
                verify_module._main_probe_failure_is_isolated_flake(
                    probe_worktree, config, module_configs, LOAD_FLAKE_MAIN_RESULT,
                )
            )

        assert result == [PROBE_NODE_ID], f'Expected [{PROBE_NODE_ID!r}], got {result!r}'

        rv.assert_awaited_once()
        assert rv.call_args.args[0] == probe_worktree, (
            f'Expected run_verification against the SAME probe worktree '
            f'{probe_worktree}, got {rv.call_args.args[0]!r}'
        )
        called_mc = rv.call_args.args[2]
        assert '-p no:xdist' in called_mc.test_command, called_mc.test_command
        assert '-o addopts=' in called_mc.test_command, called_mc.test_command
        assert '--timeout 300' in called_mc.test_command, called_mc.test_command
        assert PROBE_NODE_ID in called_mc.test_command, called_mc.test_command
        assert called_mc.lint_command is None, called_mc.lint_command
        assert called_mc.type_check_command is None, called_mc.type_check_command
        legacy_probe_mock.assert_awaited_once()
