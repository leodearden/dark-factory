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
  task-2370 step-3/step-5 (confirm-before-alarm gate — confirm fn):
           TestConfirmMainTipFailureIsReal
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

    def test_extracts_node_id_from_error_line(self) -> None:
        """A fixture/teardown ``ERROR <nodeid>`` short-summary line yields the
        node-id, same as a FAILED line (robustness-masking fix, task 2370
        amendment)."""
        from orchestrator import verify as verify_module

        output = 'ERROR orchestrator/tests/test_x.py::test_y - Exception: setup boom\n'
        assert verify_module._extract_failing_test_ids(output) == [
            'orchestrator/tests/test_x.py::test_y',
        ]

    def test_extracts_file_from_bare_error_collection_line(self) -> None:
        """A collection-level ``ERROR <file.py>`` line (whole module fails to
        collect, no ``::``) yields the bare file path as the isolation
        target."""
        from orchestrator import verify as verify_module

        output = 'ERROR orchestrator/tests/test_x.py - ImportError: boom\n'
        assert verify_module._extract_failing_test_ids(output) == [
            'orchestrator/tests/test_x.py',
        ]

    def test_extracts_both_failed_and_error_node_ids_in_first_seen_order(self) -> None:
        """A mixed short-test-summary block — a genuine ERROR alongside a
        load-induced FAILED flake — surfaces BOTH node-ids. This is the
        scenario behind the robustness-masking finding: extracting only the
        FAILED id would let the confirm gate re-run and pass on just that
        test, suppressing the alarm while never re-running (and thus
        masking) the real ERROR."""
        from orchestrator import verify as verify_module

        output = (
            'FAILED orchestrator/tests/test_x.py::test_flaky - AssertionError: boom\n'
            'ERROR orchestrator/tests/test_y.py::test_setup_broken - Exception: boom\n'
        )
        assert verify_module._extract_failing_test_ids(output) == [
            'orchestrator/tests/test_x.py::test_flaky',
            'orchestrator/tests/test_y.py::test_setup_broken',
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


# ---------------------------------------------------------------------------
# task-2370 step-3/step-5: confirm_main_tip_failure_is_real
#
# verify.confirm_main_tip_failure_is_real(config, git_ops, failing_result, *,
# main_sha) -> bool
#
# Confirm-before-alarm gate: re-runs the named failing test(s) in isolation
# (fresh probe worktree, serial + cleared addopts) before the harness files a
# red-main L1 escalation. True = confirmed real (file the alarm); False =
# demonstrated flake (suppress). Fail-safe = True for every path except a
# clean isolated pass.
# ---------------------------------------------------------------------------

CONFIRM_NODE_ID = (
    'orchestrator/tests/test_concurrent_verify_boundary.py::test_concurrent_verify_boundary'
)

CONFIRM_FAILING_RESULT = VerifyResult(
    passed=False,
    test_output=f'FAILED {CONFIRM_NODE_ID}',
    lint_output='',
    type_output='',
    summary='test_failure',
    cause_hint=f'FAILED {CONFIRM_NODE_ID}',
    category='test_failure',
)

# A failing_result with no parseable node-ids: a pure lint/compile category
# whose test_output carries no FAILED/worker-crash markers.
CONFIRM_NO_NODEID_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='orchestrator/src/orchestrator/foo.py:12:5: F401 unused import',
    type_output='',
    summary='lint_failure',
    cause_hint='F401 unused import',
    category='lint_failure',
)

_CONFIRM_PROJECT_LAYOUT = {
    'orchestrator/orchestrator.yaml': (
        'test_command: "uv run --project orchestrator --directory orchestrator '
        'pytest tests/ --tb=short -q"\n'
    ),
    'orchestrator/tests/test_concurrent_verify_boundary.py': (
        'def test_concurrent_verify_boundary():\n    pass\n'
    ),
}

# A discovered-subprojects layout that deliberately OMITS CONFIRM_NODE_ID's
# file — the "no owning subproject" fail-safe path (task 2370 amendment,
# suggestion #3).
_CONFIRM_PROJECT_LAYOUT_MISSING_FILE = {
    'orchestrator/orchestrator.yaml': (
        'test_command: "uv run --project orchestrator --directory orchestrator '
        'pytest tests/ --tb=short -q"\n'
    ),
}

# A bare, subproject-*relative* node-id (no baked-in prefix) that exists
# under TWO discovered subprojects — the ambiguous-mapping edge case (task
# 2370 amendment, suggestion #5).
CONFIRM_AMBIGUOUS_NODE_ID = 'tests/test_dup.py::test_dup'

CONFIRM_AMBIGUOUS_FAILING_RESULT = VerifyResult(
    passed=False,
    test_output=f'FAILED {CONFIRM_AMBIGUOUS_NODE_ID}',
    lint_output='',
    type_output='',
    summary='test_failure',
    cause_hint=f'FAILED {CONFIRM_AMBIGUOUS_NODE_ID}',
    category='test_failure',
)

_CONFIRM_PROJECT_LAYOUT_AMBIGUOUS = {
    'alpha/orchestrator.yaml': (
        'test_command: "uv run --project alpha --directory alpha pytest tests/ --tb=short -q"\n'
    ),
    'alpha/tests/test_dup.py': 'def test_dup():\n    pass\n',
    'beta/orchestrator.yaml': (
        'test_command: "uv run --project beta --directory beta pytest tests/ --tb=short -q"\n'
    ),
    'beta/tests/test_dup.py': 'def test_dup():\n    pass\n',
}


def _make_confirm_fake_run(run_calls: list, project_layout: dict[str, str]):
    """Fake orchestrator.git_ops._run: on a worktree add, materializes
    *project_layout* under the target path so module discovery and the
    node-id -> subproject existence mapping run against real files without a
    real git checkout (the real ``git worktree add`` subprocess is never
    invoked in these unit tests)."""

    async def _fake_run(cmd, **kwargs):
        run_calls.append(cmd)
        if 'worktree' in cmd and 'add' in cmd:
            target = Path(cmd[4])
            target.mkdir(parents=True, exist_ok=True)
            for relpath, content in project_layout.items():
                p = target / relpath
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
        return (0, '', '')

    return _fake_run


class TestConfirmMainTipFailureIsReal:
    """task-2370 step-3/step-5: verify.confirm_main_tip_failure_is_real."""

    # -- step-3: SUPPRESS path -------------------------------------------

    def test_confirm_suppresses_when_isolated_rerun_passes(self, tmp_path: Path) -> None:
        """All named failing tests pass on isolated re-run -> False (suppress),
        with the probe-worktree lifecycle, the isolated+scoped ModuleConfig,
        the suppressed-flake audit record, and the INFO log all verified.

        RED today: confirm_main_tip_failure_is_real does not exist.
        """
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        run_calls: list = []
        fake_run = _make_confirm_fake_run(run_calls, _CONFIRM_PROJECT_LAYOUT)

        rv = AsyncMock(return_value=PASSING_RESULT)
        pre_run_registry_len = len(verify_module._suppressed_flake_records)

        with (
            patch('orchestrator.git_ops._run', side_effect=fake_run),
            patch.object(verify_module, 'run_verification', rv),
            patch.object(verify_module, 'logger') as mock_logger,
        ):
            result = asyncio.run(
                verify_module.confirm_main_tip_failure_is_real(
                    config, git_ops, CONFIRM_FAILING_RESULT, main_sha=MAIN_SHA,
                )
            )

        # (a) flake -> suppress
        assert result is False, f'Expected False (suppress), got {result!r}'

        # (b) probe worktree lifecycle: add --detach ... <MAIN_SHA> AND remove --force
        add_calls = [c for c in run_calls if 'worktree' in c and 'add' in c]
        assert add_calls, 'Expected a git worktree add call'
        assert '--detach' in add_calls[0], f'Expected --detach in add cmd: {add_calls[0]}'
        assert MAIN_SHA in add_calls[0], f'Expected main_sha in add cmd: {add_calls[0]}'
        remove_calls = [c for c in run_calls if 'worktree' in c and 'remove' in c]
        assert remove_calls, 'Expected a git worktree remove --force call'
        assert '--force' in remove_calls[0], f'Expected --force in remove cmd: {remove_calls[0]}'

        # (c) isolated + scoped ModuleConfig passed to run_verification
        rv.assert_awaited()
        called_mc = rv.call_args.args[2]
        assert '-p no:xdist' in called_mc.test_command, called_mc.test_command
        assert '-o addopts=' in called_mc.test_command, called_mc.test_command
        assert CONFIRM_NODE_ID in called_mc.test_command, called_mc.test_command

        # (d) exactly one new suppressed-flake record, tagged isolated_rerun
        new_records = verify_module._suppressed_flake_records[pre_run_registry_len:]
        assert len(new_records) == 1, f'Expected 1 new record, got {new_records!r}'
        rec = new_records[0]
        assert rec['sha'] == MAIN_SHA, rec
        assert rec['node_ids'] == [CONFIRM_NODE_ID], rec
        assert rec['suppressed_via'] == 'isolated_rerun', rec

        # (e) INFO log containing the sha prefix and "suppress"
        sha_prefix = MAIN_SHA[:12]
        found = False
        for call in mock_logger.info.call_args_list:
            args = call.args
            msg = (args[0] % args[1:]) if len(args) > 1 else args[0]
            if sha_prefix in msg and 'suppress' in msg.lower():
                found = True
                break
        assert found, (
            f'Expected an INFO log containing sha prefix {sha_prefix!r} and '
            f'"suppress"; got calls={mock_logger.info.call_args_list!r}'
        )

    # -- step-5: ALARM / fail-safe paths ----------------------------------

    def test_confirm_alarms_when_isolated_rerun_still_fails(self, tmp_path: Path) -> None:
        """The isolated re-run fails on every attempt -> True (real drift,
        file the alarm), and NO suppressed-flake record is appended."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        run_calls: list = []
        fake_run = _make_confirm_fake_run(run_calls, _CONFIRM_PROJECT_LAYOUT)

        rv = AsyncMock(return_value=FAILING_RESULT)
        pre_run_registry_len = len(verify_module._suppressed_flake_records)

        with (
            patch('orchestrator.git_ops._run', side_effect=fake_run),
            patch.object(verify_module, 'run_verification', rv),
        ):
            result = asyncio.run(
                verify_module.confirm_main_tip_failure_is_real(
                    config, git_ops, CONFIRM_FAILING_RESULT, main_sha=MAIN_SHA,
                )
            )

        assert result is True, f'Expected True (still failing -> alarm), got {result!r}'
        assert rv.call_count == 2, (
            f'Expected exactly 2 isolated-rerun attempts (_SWEEP_CONFIRM_MAX_ATTEMPTS), '
            f'got {rv.call_count}'
        )
        new_records = verify_module._suppressed_flake_records[pre_run_registry_len:]
        assert new_records == [], f'Expected no new suppressed-flake record, got {new_records!r}'

    def test_confirm_alarms_without_worktree_add_when_no_node_ids(self, tmp_path: Path) -> None:
        """A failing_result with no parseable node-ids (lint/compile category)
        -> True, WITHOUT issuing any git worktree add (cheap early-out)."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        run_calls: list = []

        async def _fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return (0, '', '')

        rv = AsyncMock(return_value=PASSING_RESULT)

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(verify_module, 'run_verification', rv),
        ):
            result = asyncio.run(
                verify_module.confirm_main_tip_failure_is_real(
                    config, git_ops, CONFIRM_NO_NODEID_RESULT, main_sha=MAIN_SHA,
                )
            )

        assert result is True, f'Expected True (unconfirmable -> alarm), got {result!r}'
        assert not run_calls, (
            f'Expected NO git worktree add (or any _run call) when there are no '
            f'parseable node-ids, got {run_calls!r}'
        )
        rv.assert_not_called()

    def test_confirm_alarms_when_worktree_add_fails(self, tmp_path: Path) -> None:
        """git worktree add fails every retry -> True (fail-safe), and
        run_verification is never called."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        async def _fake_run(cmd, **kwargs):
            if 'worktree' in cmd and 'add' in cmd:
                return (1, '', 'lock contention')
            return (0, '', '')

        rv = AsyncMock(return_value=PASSING_RESULT)

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run),
            patch.object(verify_module, 'run_verification', rv),
        ):
            result = asyncio.run(
                verify_module.confirm_main_tip_failure_is_real(
                    config, git_ops, CONFIRM_FAILING_RESULT, main_sha=MAIN_SHA,
                )
            )

        assert result is True, f'Expected True when worktree add fails, got {result!r}'
        rv.assert_not_called()

    def test_confirm_alarms_when_isolated_rerun_raises(self, tmp_path: Path) -> None:
        """run_verification raising on every attempt -> True (never suppress
        on an unconfirmable result), and no suppression record is appended."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        run_calls: list = []
        fake_run = _make_confirm_fake_run(run_calls, _CONFIRM_PROJECT_LAYOUT)

        rv = AsyncMock(side_effect=RuntimeError('boom'))
        pre_run_registry_len = len(verify_module._suppressed_flake_records)

        with (
            patch('orchestrator.git_ops._run', side_effect=fake_run),
            patch.object(verify_module, 'run_verification', rv),
        ):
            result = asyncio.run(
                verify_module.confirm_main_tip_failure_is_real(
                    config, git_ops, CONFIRM_FAILING_RESULT, main_sha=MAIN_SHA,
                )
            )

        assert result is True, f'Expected True when isolated re-run raises, got {result!r}'
        new_records = verify_module._suppressed_flake_records[pre_run_registry_len:]
        assert new_records == [], f'Expected no new suppressed-flake record, got {new_records!r}'

    def test_confirm_alarms_when_isolated_rerun_is_internalerror(self, tmp_path: Path) -> None:
        """run_verification returning category='pytest_internalerror' on every
        attempt -> True (infra-sentinel is never trusted as confirmation),
        and no suppression record is appended."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        run_calls: list = []
        fake_run = _make_confirm_fake_run(run_calls, _CONFIRM_PROJECT_LAYOUT)

        rv = AsyncMock(return_value=INTERNALERROR_RESULT)
        pre_run_registry_len = len(verify_module._suppressed_flake_records)

        with (
            patch('orchestrator.git_ops._run', side_effect=fake_run),
            patch.object(verify_module, 'run_verification', rv),
        ):
            result = asyncio.run(
                verify_module.confirm_main_tip_failure_is_real(
                    config, git_ops, CONFIRM_FAILING_RESULT, main_sha=MAIN_SHA,
                )
            )

        assert result is True, (
            f'Expected True on pytest_internalerror (unconfirmable), got {result!r}'
        )
        new_records = verify_module._suppressed_flake_records[pre_run_registry_len:]
        assert new_records == [], f'Expected no new suppressed-flake record, got {new_records!r}'

    # -- amendment pass (task 2370): additional coverage -------------------

    def test_confirm_suppresses_when_isolated_rerun_fails_then_passes(
        self, tmp_path: Path
    ) -> None:
        """Isolated re-run FAILS on attempt 1 but PASSES on attempt 2 -> the
        group is a confirmed flake -> False (suppress), with exactly 2
        attempts made. Locks in the 'pass on ANY attempt within
        _SWEEP_CONFIRM_MAX_ATTEMPTS' retry-loop contract: a regression that
        returned after the first non-passing attempt would leave this red."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        run_calls: list = []
        fake_run = _make_confirm_fake_run(run_calls, _CONFIRM_PROJECT_LAYOUT)

        rv = AsyncMock(side_effect=[FAILING_RESULT, PASSING_RESULT])
        pre_run_registry_len = len(verify_module._suppressed_flake_records)

        with (
            patch('orchestrator.git_ops._run', side_effect=fake_run),
            patch.object(verify_module, 'run_verification', rv),
        ):
            result = asyncio.run(
                verify_module.confirm_main_tip_failure_is_real(
                    config, git_ops, CONFIRM_FAILING_RESULT, main_sha=MAIN_SHA,
                )
            )

        assert result is False, (
            f'Expected False (flake confirmed on 2nd attempt), got {result!r}'
        )
        assert rv.call_count == 2, (
            f'Expected exactly 2 isolated-rerun attempts (fail then pass), '
            f'got {rv.call_count}'
        )
        new_records = verify_module._suppressed_flake_records[pre_run_registry_len:]
        assert len(new_records) == 1, (
            f'Expected 1 new suppressed-flake record, got {new_records!r}'
        )

    def test_confirm_alarms_when_node_id_matches_no_subproject(self, tmp_path: Path) -> None:
        """The failing node-id's file does not exist under any discovered
        subproject in the probe worktree -> True (fail-safe alarm), and
        run_verification is never awaited — there is no group to run."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        run_calls: list = []
        fake_run = _make_confirm_fake_run(run_calls, _CONFIRM_PROJECT_LAYOUT_MISSING_FILE)

        rv = AsyncMock(return_value=PASSING_RESULT)

        with (
            patch('orchestrator.git_ops._run', side_effect=fake_run),
            patch.object(verify_module, 'run_verification', rv),
        ):
            result = asyncio.run(
                verify_module.confirm_main_tip_failure_is_real(
                    config, git_ops, CONFIRM_FAILING_RESULT, main_sha=MAIN_SHA,
                )
            )

        assert result is True, f'Expected True (unmapped node-id -> alarm), got {result!r}'
        rv.assert_not_called()

    def test_confirm_logs_warning_on_ambiguous_subproject_match(self, tmp_path: Path) -> None:
        """A bare subproject-relative node-id that exists under TWO
        discovered subprojects logs a WARNING (rather than silently
        mis-attributing) and still resolves deterministically to one of them
        for the re-run."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        run_calls: list = []
        fake_run = _make_confirm_fake_run(run_calls, _CONFIRM_PROJECT_LAYOUT_AMBIGUOUS)

        rv = AsyncMock(return_value=PASSING_RESULT)

        with (
            patch('orchestrator.git_ops._run', side_effect=fake_run),
            patch.object(verify_module, 'run_verification', rv),
            patch.object(verify_module, 'logger') as mock_logger,
        ):
            result = asyncio.run(
                verify_module.confirm_main_tip_failure_is_real(
                    config, git_ops, CONFIRM_AMBIGUOUS_FAILING_RESULT, main_sha=MAIN_SHA,
                )
            )

        assert result is False, f'Expected False (isolated re-run passed), got {result!r}'
        warned = False
        for call in mock_logger.warning.call_args_list:
            args = call.args
            msg = (args[0] % args[1:]) if len(args) > 1 else args[0]
            if CONFIRM_AMBIGUOUS_NODE_ID in msg:
                warned = True
                break
        assert warned, (
            f'Expected a WARNING log about the ambiguous node-id match; got '
            f'calls={mock_logger.warning.call_args_list!r}'
        )


# ---------------------------------------------------------------------------
# task-3095 step-1: verify._group_node_ids_by_subproject
#
# _group_node_ids_by_subproject(worktree, module_configs, node_ids, *,
# log_label) -> dict[str, list[str]] | None
#
# Pure helper extracted from confirm_main_tip_failure_is_real's node-id ->
# subproject existence-mapping block so the new sweep pre-filter reuses it
# instead of the tree gaining a THIRD copy (INV-5). Returns None for any
# unmapped node-id (the caller's fail-safe signal); WARNs and takes the first
# candidate on an ambiguous match.
# ---------------------------------------------------------------------------

# Two subprojects owning DISTINCT test files — the "group separately" case.
_GROUP_PROJECT_LAYOUT_TWO = {
    'alpha/orchestrator.yaml': (
        'test_command: "uv run --project alpha --directory alpha pytest tests/ --tb=short -q"\n'
    ),
    'alpha/tests/test_a.py': 'def test_a():\n    pass\n',
    'beta/orchestrator.yaml': (
        'test_command: "uv run --project beta --directory beta pytest tests/ --tb=short -q"\n'
    ),
    'beta/tests/test_b.py': 'def test_b():\n    pass\n',
}


def _materialize(root: Path, layout: dict[str, str]) -> None:
    """Write *layout* (relpath -> content) under *root*."""
    for relpath, content in layout.items():
        p = root / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


class TestGroupNodeIdsBySubproject:
    """task-3095 step-1: the extracted pure node-id -> subproject mapper.

    RED today: verify._group_node_ids_by_subproject does not exist.
    """

    @staticmethod
    def _discover(root: Path):
        from orchestrator.config import _discover_module_configs

        return _discover_module_configs(root)

    def test_subproject_relative_node_id_resolves_to_qualified_id(
        self, tmp_path: Path
    ) -> None:
        """(a) A bare subproject-relative node-id resolves to its owning
        prefix, and the returned id is PREFIX-QUALIFIED (worktree-root
        relative) so the scoped re-run command names a real path."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, _GROUP_PROJECT_LAYOUT_TWO)
        mcs = self._discover(tmp_path)

        groups = verify_module._group_node_ids_by_subproject(
            tmp_path, mcs, ['tests/test_a.py::test_a'], log_label='unit',
        )

        assert groups == {'alpha': ['alpha/tests/test_a.py::test_a']}, groups

    def test_prefix_qualified_node_id_passes_through_unchanged(
        self, tmp_path: Path
    ) -> None:
        """(b) An ALREADY prefix-qualified node-id maps to that prefix and is
        returned verbatim (never double-prefixed)."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, _GROUP_PROJECT_LAYOUT_TWO)
        mcs = self._discover(tmp_path)

        groups = verify_module._group_node_ids_by_subproject(
            tmp_path, mcs, ['beta/tests/test_b.py::test_b'], log_label='unit',
        )

        assert groups == {'beta': ['beta/tests/test_b.py::test_b']}, groups

    def test_node_ids_group_by_owning_subproject_preserving_order(
        self, tmp_path: Path
    ) -> None:
        """(c) Node-ids owned by different subprojects land in separate
        groups, with input order preserved WITHIN each group."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, _GROUP_PROJECT_LAYOUT_TWO)
        mcs = self._discover(tmp_path)

        groups = verify_module._group_node_ids_by_subproject(
            tmp_path,
            mcs,
            [
                'tests/test_a.py::test_one',
                'tests/test_b.py::test_two',
                'tests/test_a.py::test_three',
            ],
            log_label='unit',
        )

        assert groups == {
            'alpha': [
                'alpha/tests/test_a.py::test_one',
                'alpha/tests/test_a.py::test_three',
            ],
            'beta': ['beta/tests/test_b.py::test_two'],
        }, groups

    def test_unmapped_node_id_returns_none(self, tmp_path: Path) -> None:
        """(d) A node-id whose file exists under NO discovered subproject
        returns the None sentinel — the caller's fail-safe signal. One bad
        node-id poisons the whole batch (no partial guessing)."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, _GROUP_PROJECT_LAYOUT_TWO)
        mcs = self._discover(tmp_path)

        groups = verify_module._group_node_ids_by_subproject(
            tmp_path,
            mcs,
            ['tests/test_a.py::test_a', 'tests/test_nonexistent.py::test_ghost'],
            log_label='unit',
        )

        assert groups is None, f'Expected None for an unmapped node-id, got {groups!r}'

    def test_ambiguous_node_id_takes_first_and_warns(self, tmp_path: Path) -> None:
        """(e) A relpath existing under TWO subprojects resolves to the FIRST
        by module_configs iteration order and emits a WARNING naming both
        candidate prefixes AND the caller's *log_label* (so the log line
        attributes to the right call site)."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, _CONFIRM_PROJECT_LAYOUT_AMBIGUOUS)
        mcs = self._discover(tmp_path)
        assert list(mcs) == ['alpha', 'beta'], f'fixture precondition: {list(mcs)}'

        with patch.object(verify_module, 'logger') as mock_logger:
            groups = verify_module._group_node_ids_by_subproject(
                tmp_path, mcs, [CONFIRM_AMBIGUOUS_NODE_ID], log_label='my-call-site',
            )

        assert groups == {'alpha': [f'alpha/{CONFIRM_AMBIGUOUS_NODE_ID}']}, groups

        warned = False
        for call in mock_logger.warning.call_args_list:
            args = call.args
            msg = (args[0] % args[1:]) if len(args) > 1 else args[0]
            if 'alpha' in msg and 'beta' in msg and 'my-call-site' in msg:
                warned = True
                break
        assert warned, (
            'Expected a WARNING naming both candidate prefixes and the '
            f'log_label; got calls={mock_logger.warning.call_args_list!r}'
        )

    def test_empty_node_ids_returns_empty_dict_not_none(self, tmp_path: Path) -> None:
        """(f) An empty node-id list is NOT the unmapped sentinel — it returns
        an empty dict, so callers distinguish "nothing to run" from
        "unmappable" via their own cheap early-out."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, _GROUP_PROJECT_LAYOUT_TWO)
        mcs = self._discover(tmp_path)

        groups = verify_module._group_node_ids_by_subproject(
            tmp_path, mcs, [], log_label='unit',
        )

        assert groups == {}, f'Expected {{}} for an empty node-id list, got {groups!r}'


# ---------------------------------------------------------------------------
# task-3095 step-3: verify._sweep_failure_reproduces_in_isolation
#
# _sweep_failure_reproduces_in_isolation(worktree, config, failing_result)
#     -> bool | None
#
# COST pre-filter for run_main_tip_sweep's full-suite retry — never a
# suppression verdict. Tri-state:
#   True  = the named failing tests reproduce deterministically in isolation
#           -> the caller may skip the expensive full retry.
#   False = every named test passed in isolation (suspected contention flake)
#           -> the caller pays for the full retry as before, so a genuine FULL
#           green is still required for the harness's self-heal.
#   None  = UNCONFIRMABLE (no node-id, unmapped node-id, infra sentinel, or a
#           raise) -> the caller falls through to byte-identical legacy
#           behavior.
# ---------------------------------------------------------------------------

# A node-id that maps cleanly into the _GROUP_PROJECT_LAYOUT_TWO alpha tree.
PREFILTER_ALPHA_NODE_ID = 'tests/test_a.py::test_a'
PREFILTER_BETA_NODE_ID = 'tests/test_b.py::test_b'

PREFILTER_FAILING_RESULT = VerifyResult(
    passed=False,
    test_output=f'FAILED {PREFILTER_ALPHA_NODE_ID}',
    lint_output='',
    type_output='',
    summary='test_failure',
    cause_hint=f'FAILED {PREFILTER_ALPHA_NODE_ID}',
    category='test_failure',
)

# Two node-ids owned by DIFFERENT subprojects -> two scoped groups.
PREFILTER_TWO_GROUP_FAILING_RESULT = VerifyResult(
    passed=False,
    test_output=(
        f'FAILED {PREFILTER_ALPHA_NODE_ID}\nFAILED {PREFILTER_BETA_NODE_ID}\n'
    ),
    lint_output='',
    type_output='',
    summary='test_failure',
    cause_hint=f'FAILED {PREFILTER_ALPHA_NODE_ID}',
    category='test_failure',
)

# No recoverable node-id: a pure lint failure with empty test output.
PREFILTER_NO_NODEID_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='orchestrator/src/orchestrator/foo.py:12:5: F401 unused import',
    type_output='',
    summary='lint_failure',
    cause_hint='F401 unused import',
    category='lint_failure',
)

# A node-id whose file exists under NO discovered subproject.
PREFILTER_UNMAPPED_RESULT = VerifyResult(
    passed=False,
    test_output='FAILED tests/test_ghost.py::test_ghost',
    lint_output='',
    type_output='',
    summary='test_failure',
    cause_hint='FAILED tests/test_ghost.py::test_ghost',
    category='test_failure',
)

# An infra sentinel paired with passed=True — never trusted either way.
PREFILTER_INTERNALERROR_PASS = VerifyResult(
    passed=True,
    test_output='',
    lint_output='',
    type_output='',
    summary='pytest_internalerror',
    cause_hint='INTERNALERROR> KeyError: <WorkerController gw3>',
    category='pytest_internalerror',
)


class TestSweepFailureReproducesInIsolation:
    """task-3095 step-3: the tri-state isolated-rerun COST pre-filter.

    RED today: verify._sweep_failure_reproduces_in_isolation does not exist.
    """

    def test_all_groups_pass_returns_false(self, tmp_path: Path) -> None:
        """(a) Every scoped group PASSES in isolation -> False (did not
        reproduce) -> the caller still pays for the full retry, preserving the
        full-verify-PASS evidence bar for the harness's self-heal."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, _GROUP_PROJECT_LAYOUT_TWO)
        config = _make_config(tmp_path)

        rv = AsyncMock(return_value=PASSING_RESULT)
        with patch.object(verify_module, 'run_verification', rv):
            verdict = asyncio.run(
                verify_module._sweep_failure_reproduces_in_isolation(
                    tmp_path, config, PREFILTER_TWO_GROUP_FAILING_RESULT,
                )
            )

        assert verdict is False, f'Expected False (did not reproduce), got {verdict!r}'
        assert rv.call_count == 2, (
            f'Expected one isolated re-run per subproject group, got {rv.call_count}'
        )

    def test_any_group_fails_returns_true(self, tmp_path: Path) -> None:
        """(b) A group that still FAILS in isolation -> True (deterministic
        reproduction) -> the caller may skip the full retry."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, _GROUP_PROJECT_LAYOUT_TWO)
        config = _make_config(tmp_path)

        rv = AsyncMock(return_value=FAILING_RESULT)
        with patch.object(verify_module, 'run_verification', rv):
            verdict = asyncio.run(
                verify_module._sweep_failure_reproduces_in_isolation(
                    tmp_path, config, PREFILTER_FAILING_RESULT,
                )
            )

        assert verdict is True, f'Expected True (reproduces), got {verdict!r}'

    def test_no_recoverable_node_id_returns_none_without_running(
        self, tmp_path: Path
    ) -> None:
        """(c) No recoverable node-id -> None WITHOUT invoking
        run_verification at all (cheap early-out: a pre-filter that costs a
        subprocess to learn it has nothing to filter is worse than useless)."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, _GROUP_PROJECT_LAYOUT_TWO)
        config = _make_config(tmp_path)

        rv = AsyncMock(return_value=PASSING_RESULT)
        with patch.object(verify_module, 'run_verification', rv):
            verdict = asyncio.run(
                verify_module._sweep_failure_reproduces_in_isolation(
                    tmp_path, config, PREFILTER_NO_NODEID_RESULT,
                )
            )

        assert verdict is None, f'Expected None (unconfirmable), got {verdict!r}'
        rv.assert_not_called()

    def test_unmapped_node_id_returns_none(self, tmp_path: Path) -> None:
        """(d) A node-id owned by no discovered subproject -> None, and no
        re-run is attempted (nothing safe to scope to)."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, _GROUP_PROJECT_LAYOUT_TWO)
        config = _make_config(tmp_path)

        rv = AsyncMock(return_value=PASSING_RESULT)
        with patch.object(verify_module, 'run_verification', rv):
            verdict = asyncio.run(
                verify_module._sweep_failure_reproduces_in_isolation(
                    tmp_path, config, PREFILTER_UNMAPPED_RESULT,
                )
            )

        assert verdict is None, f'Expected None (unmapped node-id), got {verdict!r}'
        rv.assert_not_called()

    def test_infra_sentinel_rerun_returns_none_even_when_passed(
        self, tmp_path: Path
    ) -> None:
        """(e) An infra-sentinel re-run category is UNCONFIRMABLE regardless of
        the passed flag — the category check must be independent of it, or a
        crashed re-run would masquerade as 'did not reproduce'."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, _GROUP_PROJECT_LAYOUT_TWO)
        config = _make_config(tmp_path)

        rv = AsyncMock(return_value=PREFILTER_INTERNALERROR_PASS)
        with patch.object(verify_module, 'run_verification', rv):
            verdict = asyncio.run(
                verify_module._sweep_failure_reproduces_in_isolation(
                    tmp_path, config, PREFILTER_FAILING_RESULT,
                )
            )

        assert verdict is None, (
            f'Expected None on an infra-sentinel re-run category, got {verdict!r}'
        )

    def test_rerun_raising_returns_none_and_logs_warning(
        self, tmp_path: Path, caplog
    ) -> None:
        """(f) A raise inside the pre-filter degrades to None AND is logged at
        WARNING — loud-over-silent, and required by the silent-fallthrough
        ratchet (signature (b) flags `except Exception: return None` handlers
        with no WARN+ logger call)."""
        import logging

        from orchestrator import verify as verify_module

        _materialize(tmp_path, _GROUP_PROJECT_LAYOUT_TWO)
        config = _make_config(tmp_path)

        rv = AsyncMock(side_effect=RuntimeError('boom'))
        with (
            patch.object(verify_module, 'run_verification', rv),
            caplog.at_level(logging.WARNING, logger=verify_module.logger.name),
        ):
            verdict = asyncio.run(
                verify_module._sweep_failure_reproduces_in_isolation(
                    tmp_path, config, PREFILTER_FAILING_RESULT,
                )
            )

        assert verdict is None, f'Expected None when the re-run raises, got {verdict!r}'
        assert any(r.levelno >= logging.WARNING for r in caplog.records), (
            f'Expected a WARNING+ record; got {[(r.levelname, r.message) for r in caplog.records]!r}'
        )

    def test_scoped_command_shape(self, tmp_path: Path) -> None:
        """The ModuleConfig handed to run_verification is scoped to the failing
        node-ids, forced serial, addopts-cleared, carries a generous explicit
        --timeout (pyproject's `timeout=60` survives `-o addopts=` and would
        otherwise starve the isolated run into a false 'reproduces'), and runs
        NEITHER lint NOR typecheck (not the flake surface; pure cost)."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, _GROUP_PROJECT_LAYOUT_TWO)
        config = _make_config(tmp_path)

        rv = AsyncMock(return_value=PASSING_RESULT)
        with patch.object(verify_module, 'run_verification', rv):
            asyncio.run(
                verify_module._sweep_failure_reproduces_in_isolation(
                    tmp_path, config, PREFILTER_FAILING_RESULT,
                )
            )

        rv.assert_awaited()
        called_mc = rv.call_args.args[2]
        cmd = called_mc.test_command
        assert '-p no:xdist' in cmd, cmd
        assert '-o addopts=' in cmd, cmd
        assert ('--timeout 300' in cmd or '--timeout=300' in cmd), cmd
        assert f'alpha/{PREFILTER_ALPHA_NODE_ID}' in cmd, cmd
        assert called_mc.lint_command is None, called_mc.lint_command
        assert called_mc.type_check_command is None, called_mc.type_check_command


# ---------------------------------------------------------------------------
# task-3095 step-7: run_main_tip_sweep <-> pre-filter wiring
#
# The pre-filter gates the expensive full-suite retry. It NEVER becomes the
# sweep's verdict: on the reproduces path the sweep returns the FIRST-PASS
# FAILING result, so the harness's "full-verify PASS required to self-heal"
# precondition (harness.py's _close_superseded_main_sweep_escalations) can
# never be satisfied by subset-only evidence.
# ---------------------------------------------------------------------------


class TestRunMainTipSweepIsolatedPrefilter:
    """task-3095 step-7: the pre-filter's effect on run_main_tip_sweep.

    RED today: the pre-filter is not wired in, so (a) and (d) see 2 calls /
    a non-zero pre-filter call count.
    """

    @staticmethod
    def _fake_git_run():
        async def _fake_run(cmd, **kwargs):
            return (0, '', '')

        return _fake_run

    def test_reproduces_skips_full_retry_and_returns_first_pass(
        self, tmp_path: Path
    ) -> None:
        """(a) Pre-filter True -> the full retry is SKIPPED and the sweep
        returns the FIRST-PASS failing object itself (identity), with no
        suppression record. The harness then adjudicates via its unchanged
        fresh-worktree confirm gate."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        rfv = AsyncMock(side_effect=[FAILING_RESULT, PASSING_RESULT])
        prefilter = AsyncMock(return_value=True)
        pre_run_registry_len = len(verify_module._suppressed_flake_records)

        with (
            patch('orchestrator.git_ops._run', side_effect=self._fake_git_run()),
            patch.object(verify_module, 'run_full_verification', rfv),
            patch.object(
                verify_module, '_sweep_failure_reproduces_in_isolation', prefilter
            ),
        ):
            result = asyncio.run(verify_module.run_main_tip_sweep(config, git_ops))

        assert rfv.call_count == 1, (
            f'Expected the full-suite retry to be SKIPPED (1 call), got {rfv.call_count}'
        )
        assert result is not None, 'Expected (sha, VerifyResult), got None'
        swept_sha, vr = result
        assert swept_sha == MAIN_SHA
        assert vr is FAILING_RESULT, (
            f'Expected the FIRST-PASS result object itself, got {vr!r}'
        )
        # Self-heal invariant guard: the sweep must never hand back a passing
        # result derived from subset-only evidence.
        assert vr.passed is False, f'Expected passed=False, got {vr.passed!r}'
        new_records = verify_module._suppressed_flake_records[pre_run_registry_len:]
        assert new_records == [], (
            f'Expected NO suppression record on the reproduces path, got {new_records!r}'
        )

    def test_does_not_reproduce_runs_full_retry_unchanged(
        self, tmp_path: Path
    ) -> None:
        """(b) Pre-filter False -> the legacy flake-suppression path is
        byte-identical: full retry runs, a retry PASS is returned, and exactly
        one audit record carries the FIRST-PASS sha/category/cause_hint."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        rfv = AsyncMock(side_effect=[FAILING_RESULT, PASSING_RESULT])
        prefilter = AsyncMock(return_value=False)
        pre_run_registry_len = len(verify_module._suppressed_flake_records)

        with (
            patch('orchestrator.git_ops._run', side_effect=self._fake_git_run()),
            patch.object(verify_module, 'run_full_verification', rfv),
            patch.object(
                verify_module, '_sweep_failure_reproduces_in_isolation', prefilter
            ),
        ):
            result = asyncio.run(verify_module.run_main_tip_sweep(config, git_ops))

        assert rfv.call_count == 2, (
            f'Expected the full retry to still run (2 calls), got {rfv.call_count}'
        )
        assert result is not None
        _swept_sha, vr = result
        assert vr is PASSING_RESULT, f'Expected the retry result, got {vr!r}'
        new_records = verify_module._suppressed_flake_records[pre_run_registry_len:]
        assert len(new_records) == 1, f'Expected 1 record, got {new_records!r}'
        rec = new_records[0]
        assert rec['sha'] == MAIN_SHA, rec
        assert rec['first_pass_category'] == FAILING_RESULT.category, rec
        assert rec['first_pass_cause_hint'] == FAILING_RESULT.cause_hint, rec

    def test_unconfirmable_falls_through_to_legacy_full_retry(
        self, tmp_path: Path
    ) -> None:
        """(c) Pre-filter None (UNCONFIRMABLE) -> byte-identical pre-3095
        behavior: the full retry runs."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        rfv = AsyncMock(side_effect=[FAILING_RESULT, FAILING_RESULT])
        prefilter = AsyncMock(return_value=None)

        with (
            patch('orchestrator.git_ops._run', side_effect=self._fake_git_run()),
            patch.object(verify_module, 'run_full_verification', rfv),
            patch.object(
                verify_module, '_sweep_failure_reproduces_in_isolation', prefilter
            ),
        ):
            result = asyncio.run(verify_module.run_main_tip_sweep(config, git_ops))

        assert rfv.call_count == 2, (
            f'Expected legacy fall-through (2 calls), got {rfv.call_count}'
        )
        assert result is not None
        assert result[1].passed is False

    def test_kill_switch_off_never_invokes_prefilter(self, tmp_path: Path) -> None:
        """(d) main_tip_sweep_isolated_prefilter_enabled=False -> the
        pre-filter is never invoked at all and the full retry runs, i.e. the
        operator revert path is byte-identical pre-3095 behavior."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        config = config.model_copy(
            update={'main_tip_sweep_isolated_prefilter_enabled': False}
        )
        git_ops = _make_git_ops(tmp_path)

        rfv = AsyncMock(side_effect=[FAILING_RESULT, PASSING_RESULT])
        prefilter = AsyncMock(return_value=True)

        with (
            patch('orchestrator.git_ops._run', side_effect=self._fake_git_run()),
            patch.object(verify_module, 'run_full_verification', rfv),
            patch.object(
                verify_module, '_sweep_failure_reproduces_in_isolation', prefilter
            ),
        ):
            result = asyncio.run(verify_module.run_main_tip_sweep(config, git_ops))

        assert prefilter.call_count == 0, (
            f'Expected the pre-filter to be gated off entirely, got '
            f'{prefilter.call_count} call(s)'
        )
        assert rfv.call_count == 2, (
            f'Expected the legacy full retry (2 calls), got {rfv.call_count}'
        )
        assert result is not None
        assert result[1] is PASSING_RESULT

    def test_prefilter_receives_sweep_worktree_and_first_pass_result(
        self, tmp_path: Path
    ) -> None:
        """(e) The pre-filter is handed the sweep's OWN pinned worktree (no
        second worktree add) and the FIRST-PASS result — not the retry's."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        rfv = AsyncMock(side_effect=[FAILING_RESULT, PASSING_RESULT])
        prefilter = AsyncMock(return_value=False)

        with (
            patch('orchestrator.git_ops._run', side_effect=self._fake_git_run()),
            patch.object(verify_module, 'run_full_verification', rfv),
            patch.object(
                verify_module, '_sweep_failure_reproduces_in_isolation', prefilter
            ),
        ):
            asyncio.run(verify_module.run_main_tip_sweep(config, git_ops))

        prefilter.assert_awaited_once()
        pf_worktree, pf_config, pf_result = prefilter.call_args.args
        # Same worktree the full verification ran in — the sweep's own pin.
        sweep_worktree = rfv.call_args_list[0].args[0]
        assert pf_worktree == sweep_worktree, (
            f'Expected the sweep worktree {sweep_worktree!r}, got {pf_worktree!r}'
        )
        assert str(pf_worktree).startswith(str(git_ops.worktree_base)), pf_worktree
        assert pf_config is config
        assert pf_result is FAILING_RESULT, (
            f'Expected the FIRST-PASS result, got {pf_result!r}'
        )

    # -- step-9: ordering / fail-safe guards -------------------------------

    def test_infra_sentinel_short_circuits_before_prefilter(
        self, tmp_path: Path
    ) -> None:
        """(a) A first-pass INFRA_TRANSIENT_CATEGORIES result still returns the
        None sentinel WITHOUT invoking the pre-filter — the infra-sentinel
        branch keeps precedence, so an infra crash never buys a subprocess
        trying to reproduce test failures that were never really observed."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        rfv = AsyncMock(return_value=INTERNALERROR_RESULT)
        prefilter = AsyncMock(return_value=True)

        with (
            patch('orchestrator.git_ops._run', side_effect=self._fake_git_run()),
            patch.object(verify_module, 'run_full_verification', rfv),
            patch.object(
                verify_module, '_sweep_failure_reproduces_in_isolation', prefilter
            ),
        ):
            result = asyncio.run(verify_module.run_main_tip_sweep(config, git_ops))

        assert result is None, f'Expected the None infra sentinel, got {result!r}'
        assert prefilter.call_count == 0, (
            f'Expected the infra branch to short-circuit BEFORE the pre-filter, '
            f'got {prefilter.call_count} call(s)'
        )
        assert rfv.call_count == 1, rfv.call_count

    def test_enoent_on_self_short_circuits_before_prefilter(
        self, tmp_path: Path
    ) -> None:
        """(b) A first-pass result whose cause_hint names THIS sweep's own
        tmp_path with an ENOENT returns None without invoking the pre-filter —
        the task-2507 _enoent_on_self backstop keeps precedence (the worktree
        is gone, so an isolated re-run in it is meaningless)."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        captured_worktrees: list = []

        async def _rfv(worktree, *args, **kwargs):
            captured_worktrees.append(worktree)
            return VerifyResult(
                passed=False,
                test_output='',
                lint_output='',
                type_output='',
                summary='unknown_test_failure',
                cause_hint=(
                    f'[Errno 2] No such file or directory: {worktree}/pyproject.toml'
                ),
                category='unknown_test_failure',
            )

        rfv = AsyncMock(side_effect=_rfv)
        prefilter = AsyncMock(return_value=True)

        with (
            patch('orchestrator.git_ops._run', side_effect=self._fake_git_run()),
            patch.object(verify_module, 'run_full_verification', rfv),
            patch.object(
                verify_module, '_sweep_failure_reproduces_in_isolation', prefilter
            ),
        ):
            result = asyncio.run(verify_module.run_main_tip_sweep(config, git_ops))

        assert captured_worktrees, 'fixture precondition: the sweep ran a verification'
        assert result is None, (
            f'Expected the None sentinel for a vanished sweep worktree, got {result!r}'
        )
        assert prefilter.call_count == 0, (
            f'Expected the _enoent_on_self backstop to short-circuit BEFORE the '
            f'pre-filter, got {prefilter.call_count} call(s)'
        )

    def test_prefilter_raise_does_not_abort_the_sweep(self, tmp_path: Path) -> None:
        """(c) A pre-filter that RAISES must not propagate into
        run_main_tip_sweep's outer `except Exception`, which would silently
        turn a real sweep signal into the None sentinel. The sweep falls
        through to the full retry instead."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        rfv = AsyncMock(side_effect=[FAILING_RESULT, FAILING_RESULT])
        prefilter = AsyncMock(side_effect=RuntimeError('boom'))

        with (
            patch('orchestrator.git_ops._run', side_effect=self._fake_git_run()),
            patch.object(verify_module, 'run_full_verification', rfv),
            patch.object(
                verify_module, '_sweep_failure_reproduces_in_isolation', prefilter
            ),
        ):
            result = asyncio.run(verify_module.run_main_tip_sweep(config, git_ops))

        assert result is not None, (
            'A raising pre-filter must not collapse the sweep into the None '
            'sentinel — the real failing signal has to survive'
        )
        assert rfv.call_count == 2, (
            f'Expected fall-through to the full retry (2 calls), got {rfv.call_count}'
        )
        assert result[1].passed is False
