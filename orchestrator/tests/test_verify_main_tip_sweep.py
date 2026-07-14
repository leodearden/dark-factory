"""Tests for ``verify.run_main_tip_sweep`` — task 1832.

run_main_tip_sweep(config, git_ops) -> tuple[str, VerifyResult] | None

Runs a FULL unscoped verification (all subprojects) against a throwaway
detached worktree pinned at the current main SHA.  Returns (main_sha,
VerifyResult) on success, or None on any infrastructure failure (fail-safe).

Mirrors the mock strategy of test_verify_preexisting_main_break.py:
  - Real GitOps with get_main_sha patched (AsyncMock) and worktree_base
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
  task 2507 step-5 (SECONDARY backstop): TestRunMainTipSweepEnoentOwnWorktree
           — an ENOENT result whose cause_hint names THIS sweep's own
           tmp_path maps to the None sentinel (retry-next-tick, no drift),
           covered independently at BOTH the first-pass and the retry call
           site; a CONTROL test pins narrowness — an ENOENT naming a
           DIFFERENT path is still real drift and passes through unchanged.
  task-2370 step-1 (confirm-before-alarm gate — node-id extraction):
           TestExtractFailingTestIds
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
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


def _make_git_ops(tmp_path: Path, main_sha: str = MAIN_SHA) -> GitOps:
    """Build a real GitOps with get_main_sha patched.

    Behavior-preserving swap from a MagicMock: run_main_tip_sweep only
    touches git_ops.worktree_base, git_ops.get_main_sha(), and the
    module-level orchestrator.git_ops._run (patched separately by each
    test) — all present and correct on a real GitOps instance. A real
    instance is required once the probe consumes
    GitOps.ephemeral_worktree(), which a plain MagicMock cannot satisfy
    as an async context manager.
    """
    git_ops = GitOps(_make_config(tmp_path).git, tmp_path)
    git_ops.get_main_sha = AsyncMock(return_value=main_sha)  # type: ignore[method-assign]
    git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
    return git_ops


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
        GREEN after impl: returns (sha, PASSING_RESULT) after exactly two calls,
        and appends a structured record to verify._suppressed_flake_records.
        """
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        async def _fake_run(cmd, **kwargs):
            return (0, '', '')

        rfv = AsyncMock(side_effect=[FAILING_RESULT, PASSING_RESULT])

        # Capture registry length before run so we can assert exactly one entry
        # was appended (module-level list accumulates across tests in the suite).
        pre_run_registry_len = len(verify_module._suppressed_flake_records)

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

        # Verify the structured audit record was appended.
        new_records = verify_module._suppressed_flake_records[pre_run_registry_len:]
        assert len(new_records) == 1, (
            f'Expected exactly 1 new entry in _suppressed_flake_records, '
            f'got {len(new_records)}: {new_records!r}'
        )
        rec = new_records[0]
        assert rec['sha'] == MAIN_SHA, f'record sha mismatch: {rec!r}'
        assert rec['first_pass_category'] == FAILING_RESULT.category, (
            f'record first_pass_category mismatch: {rec!r}'
        )
        assert rec['first_pass_cause_hint'] == FAILING_RESULT.cause_hint, (
            f'record first_pass_cause_hint mismatch: {rec!r}'
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


class TestRunMainTipSweepRoleStamp:
    """task 2391 (PRD T3): run_main_tip_sweep stamps role='background' on
    both the first-pass and flake-retry run_full_verification calls.

    RED today: run_main_tip_sweep calls run_full_verification(tmp_path,
    config) with no role kwarg at either call site.
    """

    def test_run_main_tip_sweep_stamps_background_role_on_both_calls(
        self, tmp_path: Path
    ) -> None:
        """Both the first-pass and the flake-retry call must receive
        role='background'.

        Uses the FAILING -> PASSING side_effect (mirrors
        TestRunMainTipSweepRetryOnFlake) to exercise both call sites in one
        sweep invocation.
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
        assert vr.passed is True

        assert rfv.call_count == 2, (
            f'Expected exactly 2 calls to run_full_verification '
            f'(first-pass + flake-retry), got {rfv.call_count}'
        )
        for call_index, one_call in enumerate(rfv.call_args_list):
            assert one_call.kwargs.get('role') == 'background', (
                f'Expected call #{call_index} to run_full_verification to receive '
                f"role='background'; got kwargs={one_call.kwargs!r}"
            )

    def test_run_main_tip_sweep_role_propagates_through_real_run_full_verification(
        self, tmp_path: Path
    ) -> None:
        """Thin end-to-end seam: role propagates across BOTH hops.

        The other tests here (and TestRunFullVerificationRole in
        test_verify.py) each mock one hop of the chain
        ``run_main_tip_sweep -> run_full_verification -> run_verification`` —
        that's reasonable seam-based coverage, but no single test lets both
        hops run for real together. This test mocks only the innermost
        function (``run_verification``) and lets the REAL
        ``run_full_verification`` run in between, confirming the
        ``role='background'`` stamp actually threads all the way down rather
        than merely being asserted at each seam independently.

        The sweep's worktree path is never created on disk (``git worktree
        add`` is faked via ``orchestrator.git_ops._run``), so the real
        ``run_full_verification``'s discovery walk
        (``_discover_module_configs``) silently finds zero subprojects on the
        nonexistent path (``os.walk`` on a missing dir yields nothing) and
        takes the no-subproject global-fallback branch — exercising the same
        single ``run_verification`` call the production sweep hits on a
        monorepo with no ``orchestrator.yaml`` subprojects.
        """
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        async def _fake_run(cmd, **kwargs):
            return (0, '', '')

        run_verification_calls: list = []

        async def _fake_run_verification(project_root, cfg, module_config=None, **kwargs):
            run_verification_calls.append(kwargs)
            return PASSING_RESULT

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(verify_module, 'run_verification', side_effect=_fake_run_verification),
        ):
            result = asyncio.run(
                verify_module.run_main_tip_sweep(config, git_ops)
            )

        assert result is not None, 'Expected (sha, VerifyResult), got None'
        swept_sha, vr = result
        assert swept_sha == MAIN_SHA
        assert vr.passed is True

        assert len(run_verification_calls) == 1, (
            f'Expected exactly 1 run_verification call — PASSING_RESULT on '
            f'the first pass needs no flake-retry — got '
            f'{len(run_verification_calls)}'
        )
        assert run_verification_calls[0].get('role') == 'background', (
            "Expected the real run_full_verification to thread role='background' "
            f'down to run_verification; got kwargs={run_verification_calls[0]!r}'
        )


# ---------------------------------------------------------------------------
# task 2507 step-5: SECONDARY backstop — ENOENT naming this sweep's OWN
# tmp_path maps to the None sentinel instead of drift passthrough
# ---------------------------------------------------------------------------


class TestRunMainTipSweepEnoentOwnWorktree:
    """task 2507 step-5: narrow defense-in-depth behind the PRIMARY
    flock-liveness fix (GitOps.ephemeral_worktree, steps 1-4). If the
    sweep's own worktree still vanishes mid-run for some OTHER reason
    (disk fault, a manual ``rm``, anything not closed by the flock), the
    resulting ENOENT must not masquerade as real main-tip drift.

    Deliberately narrow: matches ONLY an ENOENT whose cause_hint names
    THIS sweep's own minted tmp_path — NOT a blanket addition of
    'unknown_test_failure' to INFRA_TRANSIENT_CATEGORIES, which would
    silently swallow genuine drift under that (broad, real-failure)
    category.

    RED today: run_main_tip_sweep returns ``(sha, result)`` for any
    non-infra-category failure regardless of ENOENT/own-path — there is
    no such backstop yet.
    """

    def test_enoent_naming_own_tmp_path_returns_none(self, tmp_path: Path) -> None:
        """An ENOENT cause_hint naming this sweep's own tmp_path is treated
        as an infra transient (worktree vanished mid-run) — return None
        (retry-next-tick, no drift escalation) — and cleanup still runs."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        run_calls: list = []
        captured: dict[str, str] = {}

        async def _fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            if 'worktree' in cmd and 'add' in cmd:
                detach_idx = cmd.index('--detach')
                captured['tmp_path'] = cmd[detach_idx + 1]
            return (0, '', '')

        async def _fake_full_verify(*args, **kwargs):
            return VerifyResult(
                passed=False,
                test_output='',
                lint_output='',
                type_output='',
                summary='unknown_test_failure',
                cause_hint=(
                    f"[Errno 2] No such file or directory: "
                    f"'{captured['tmp_path']}' | further traceback noise"
                ),
                category='unknown_test_failure',
            )

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(verify_module, 'run_full_verification', side_effect=_fake_full_verify),
        ):
            result = asyncio.run(
                verify_module.run_main_tip_sweep(config, git_ops)
            )

        assert result is None, (
            f'Expected None (own-worktree ENOENT backstop) when the ENOENT '
            f"cause_hint names this sweep's own tmp_path, got {result!r}"
        )
        assert captured.get('tmp_path'), 'expected the fake add to have captured a tmp_path'

        # Cleanup must still run even though we return None early.
        remove_calls = [c for c in run_calls if 'worktree' in c and 'remove' in c]
        assert remove_calls, (
            'git worktree remove --force must still run even when the '
            'own-worktree ENOENT backstop returns None (cleanup-in-finally guarantee)'
        )

    def test_enoent_naming_own_tmp_path_on_retry_returns_none(
        self, tmp_path: Path,
    ) -> None:
        """Same backstop, but exercised on the RETRY call site
        (verify.py:4217's ``_enoent_on_self(retry)``), not the first-pass
        call site (verify.py:4179's ``_enoent_on_self(result)``) the sibling
        test above covers.

        The first-pass result is an ORDINARY (non-ENOENT) failure, so it
        triggers the existing retry-on-flake path without ever tripping the
        first-pass ENOENT check; only the retry result is an ENOENT naming
        this sweep's own tmp_path. Without this test, a regression that
        dropped or broke the retry-path branch specifically would ship
        green, since the first-pass test short-circuits (returns None)
        before any retry runs.
        """
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        run_calls: list = []
        captured: dict[str, str] = {}

        async def _fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            if 'worktree' in cmd and 'add' in cmd:
                detach_idx = cmd.index('--detach')
                captured['tmp_path'] = cmd[detach_idx + 1]
            return (0, '', '')

        call_count = 0

        async def _fake_full_verify(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Ordinary (non-ENOENT) first-pass failure: not in
                # INFRA_TRANSIENT_CATEGORIES and _enoent_on_self(result) is
                # False, so run_main_tip_sweep proceeds to the retry.
                return FAILING_RESULT
            return VerifyResult(
                passed=False,
                test_output='',
                lint_output='',
                type_output='',
                summary='unknown_test_failure',
                cause_hint=(
                    f"[Errno 2] No such file or directory: "
                    f"'{captured['tmp_path']}' | further traceback noise"
                ),
                category='unknown_test_failure',
            )

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(verify_module, 'run_full_verification', side_effect=_fake_full_verify),
        ):
            result = asyncio.run(
                verify_module.run_main_tip_sweep(config, git_ops)
            )

        assert result is None, (
            f'Expected None (own-worktree ENOENT backstop on the RETRY path) '
            f"when the retry's ENOENT cause_hint names this sweep's own "
            f'tmp_path, got {result!r}'
        )
        assert call_count == 2, (
            f'Expected exactly 2 calls to run_full_verification (first-pass '
            f'+ retry) — this test is only meaningful if it actually '
            f'reaches the retry-path branch, got {call_count}'
        )
        assert captured.get('tmp_path'), 'expected the fake add to have captured a tmp_path'

        # Cleanup must still run even though we return None early.
        remove_calls = [c for c in run_calls if 'worktree' in c and 'remove' in c]
        assert remove_calls, (
            'git worktree remove --force must still run even when the '
            'retry-path own-worktree ENOENT backstop returns None '
            '(cleanup-in-finally guarantee)'
        )

    def test_enoent_naming_different_path_is_still_drift_passthrough(
        self, tmp_path: Path,
    ) -> None:
        """CONTROL (narrowness): an ENOENT result naming a DIFFERENT path —
        NOT this sweep's own tmp_path — is real drift and must still pass
        through as (main_sha, failing_result), not get swallowed."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        run_calls: list = []
        other_path = tmp_path / 'some' / 'unrelated' / 'path'

        async def _fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return (0, '', '')

        enoent_elsewhere = VerifyResult(
            passed=False,
            test_output='',
            lint_output='',
            type_output='',
            summary='unknown_test_failure',
            cause_hint=f"[Errno 2] No such file or directory: '{other_path}'",
            category='unknown_test_failure',
        )

        async def _fake_full_verify(*args, **kwargs):
            return enoent_elsewhere

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(verify_module, 'run_full_verification', side_effect=_fake_full_verify),
        ):
            result = asyncio.run(
                verify_module.run_main_tip_sweep(config, git_ops)
            )

        assert result is not None, (
            'Expected (sha, VerifyResult) drift-passthrough for an ENOENT '
            'naming a DIFFERENT path, got None'
        )
        swept_sha, vr = result
        assert swept_sha == MAIN_SHA
        assert vr.passed is False
        assert vr.category == 'unknown_test_failure'
        assert vr.cause_hint == enoent_elsewhere.cause_hint

        remove_calls = [c for c in run_calls if 'worktree' in c and 'remove' in c]
        assert remove_calls, 'git worktree remove should run even on drift passthrough'


# ---------------------------------------------------------------------------
# task-2370 step-1: _extract_failing_test_ids — pure node-id extraction
# ---------------------------------------------------------------------------


class TestExtractFailingTestIds:
    """task-2370 step-1: verify._extract_failing_test_ids(test_output) -> list[str].

    Pure helper (no I/O) that recovers pytest node-ids from raw test output
    for the confirm-before-alarm isolated re-run gate. RED today: the
    function does not exist yet.
    """

    def test_extracts_node_ids_from_failed_lines(self) -> None:
        """FAILED lines yield node-ids, including a parametrized case and a
        line with a trailing ' - AssertionError: ...' reason."""
        from orchestrator import verify as verify_module

        output = (
            'FAILED orchestrator/tests/test_x.py::test_y\n'
            "FAILED orchestrator/tests/test_x.py::test_z[case-1] - AssertionError: boom\n"
        )
        assert verify_module._extract_failing_test_ids(output) == [
            'orchestrator/tests/test_x.py::test_y',
            'orchestrator/tests/test_x.py::test_z[case-1]',
        ]

    def test_extracts_node_id_from_xdist_worker_crash_notice(self) -> None:
        """An explicit "crashed while running '<nodeid>'" notice yields the
        quoted node-id."""
        from orchestrator import verify as verify_module

        output = "worker 'gw3' crashed while running 'tests/test_a.py::test_b'\n"
        assert verify_module._extract_failing_test_ids(output) == [
            'tests/test_a.py::test_b',
        ]

    def test_extracts_node_id_preceding_node_down_marker(self) -> None:
        """When no explicit "crashed while running" phrasing is present, the
        in-progress node-id line immediately preceding a
        "[gwN] node down: Not properly terminated" marker is recovered."""
        from orchestrator import verify as verify_module

        output = (
            'tests/test_a.py::test_b\n'
            '[gw3] node down: Not properly terminated\n'
        )
        assert verify_module._extract_failing_test_ids(output) == [
            'tests/test_a.py::test_b',
        ]

    def test_deduplicates_preserving_first_seen_order(self) -> None:
        """Repeated node-ids (e.g. FAILED line + a later re-mention) collapse
        to one entry, in first-seen order."""
        from orchestrator import verify as verify_module

        output = (
            'FAILED orchestrator/tests/test_x.py::test_y\n'
            'FAILED orchestrator/tests/test_a.py::test_b\n'
            'FAILED orchestrator/tests/test_x.py::test_y\n'
        )
        assert verify_module._extract_failing_test_ids(output) == [
            'orchestrator/tests/test_x.py::test_y',
            'orchestrator/tests/test_a.py::test_b',
        ]

    def test_returns_empty_list_for_non_test_failure(self) -> None:
        """A pure lint (ruff) error block has no pytest node-ids."""
        from orchestrator import verify as verify_module

        ruff_output = (
            'orchestrator/src/orchestrator/foo.py:12:5: F401 unused import\n'
            'Found 1 error.\n'
        )
        assert verify_module._extract_failing_test_ids(ruff_output) == []

    def test_returns_empty_list_for_unparseable_worker_crash(self) -> None:
        """A node-down marker with no adjacent node-id-shaped line yields []
        rather than a guessed/garbage id."""
        from orchestrator import verify as verify_module

        opaque_crash_output = (
            'Something bad happened.\n'
            '[gw3] node down: Not properly terminated\n'
        )
        assert verify_module._extract_failing_test_ids(opaque_crash_output) == []

    def test_returns_empty_list_for_falsy_output(self) -> None:
        from orchestrator import verify as verify_module

        assert verify_module._extract_failing_test_ids('') == []
