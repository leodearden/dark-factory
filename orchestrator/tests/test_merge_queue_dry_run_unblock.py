"""Tests for merge_queue's dry-run-unblock investigation spawn (task η / 2141).

Closes the AFK coverage gap: today metadata.dry_run_proposals[] is written
ONLY by workflow._spawn_dry_run_unblock at agent-block time — merge_queue's
post-merge-verify block path (_run_post_merge_verify) produces a
MergeOutcome('blocked') but never spawned a dry-run investigation, so
b3_gate.check_proposal returned ABORT 'no proposal to gate' for the entire
merge-verify-RED class.

This module pins the spawn wiring at the two MERGE_VERIFY_RED outcome sites
(generic task-fault and unscoped-typecheck-FAILED), the guards that skip pure
timeouts / disabled unblock_auto / duplicate in-flight investigations, and the
exclusion of the sibling blocked classes (flock-contention, persistent
ENOSPC, disk-guard-skip, main-health-red). Reuses the
test_merge_queue_main_health.py driver (_make_config/_make_git_ops/_make_req,
COMPILE_ERROR_RESULT) to reach _run_post_merge_verify's blocked outcomes.
"""
from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from test_dry_run_unblock import _init_git_repo, _make_agent_result, _RecordingScheduler
from test_merge_queue_main_health import (
    COMPILE_ERROR_RESULT,
    INFRA_TIMEOUT_RESULT,
    MAIN_SHA,
    _make_config,
    _make_git_ops,
    _make_req,
)

from orchestrator.b3_gate import ABORT, check_proposal
from orchestrator.event_store import EventType
from orchestrator.merge_queue import (
    MAIN_HEALTH_RED_REASON_PREFIX,
    TRANSIENT_INFRA_REASON_PREFIX,
    MergeOutcome,
    MergeRequest,
    RealMergeItem,
    SpeculativeMergeWorker,
    _DryRunInvestigationHandles,
    _run_post_merge_verify,
)
from orchestrator.unblock_types import BlockClass
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import (
    FLOCK_CONTENTION_CATEGORY,
    UNSCOPED_TYPECHECK_FAILED_CATEGORY,
    UNSCOPED_TYPECHECK_TIMEOUT_CATEGORY,
    HostLease,
)

PERSISTENT_ENOSPC_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='',
    summary='no space left on device',
    category='',
)

UNSCOPED_TYPECHECK_FAILED_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='error TS2322: some type error',
    summary='frontend',
    category=UNSCOPED_TYPECHECK_FAILED_CATEGORY,
)

UNSCOPED_TYPECHECK_TIMEOUT_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='',
    summary='frontend',
    category=UNSCOPED_TYPECHECK_TIMEOUT_CATEGORY,
)

FLOCK_CONTENTION_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='',
    summary='flock contention',
    category=FLOCK_CONTENTION_CATEGORY,
    contention={'host': 'laptop1', 'holder_pgid': 123, 'waiter_pgid': 456},
)


def _make_handles(
    *, scheduler: object | None = None, mcp: object | None = None,
) -> _DryRunInvestigationHandles:
    """Build a handles bundle with a live (non-None) scheduler by default.

    A live scheduler is required for _spawn_merge_verify_dry_run to proceed
    past its `handles is None or handles.scheduler is None` early return.
    """
    return _DryRunInvestigationHandles(
        scheduler=MagicMock() if scheduler is None else scheduler,
        mcp=MagicMock() if mcp is None else mcp,
    )


async def _drive_verify_with_handles(
    req: MergeRequest,
    merge_wt: Path,
    git_ops,
    *,
    dry_run_handles: _DryRunInvestigationHandles | None,
) -> MergeOutcome | None:
    """test_merge_queue_main_health._drive_verify + the dry_run_handles kwarg."""
    return await _run_post_merge_verify(
        git_ops, req, merge_wt,
        timeouts={},
        enospc_retries={},
        max_timeouts=3,
        max_enospc=1,
        dry_run_handles=dry_run_handles,
    )


class TestGenericMergeVerifyRedSpawnsDryRun:
    """Step-1 (RED): the generic task-fault block site must spawn
    run_dry_run_unblock with block_class=MERGE_VERIFY_RED when handles are
    live, and must NOT spawn it when dry_run_handles=None (solo/train path).
    """

    def test_generic_merge_verify_red_spawns_dry_run(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        handles = _make_handles()
        run_dry_run_mock = AsyncMock(return_value=None)

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=COMPILE_ERROR_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(False, '')),
                ),
                patch(
                    'orchestrator.merge_queue.run_dry_run_unblock',
                    new=run_dry_run_mock,
                ),
            ):
                outcome = await _drive_verify_with_handles(
                    req, merge_wt, git_ops, dry_run_handles=handles,
                )
                # Let the fire-and-forget create_task run to completion
                # (run_dry_run_mock resolves immediately) inside this SAME
                # loop, before asyncio.run() tears it down — draining
                # avoids a "coroutine was never awaited" / pending-task
                # teardown warning (filterwarnings turns these into errors).
                await asyncio.sleep(0)
                if handles.background_tasks:
                    await asyncio.gather(
                        *handles.background_tasks, return_exceptions=True,
                    )
                return outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.status == 'blocked'
        run_dry_run_mock.assert_awaited_once()
        assert run_dry_run_mock.await_args is not None
        kwargs = run_dry_run_mock.await_args.kwargs
        assert kwargs['block_class'] == BlockClass.MERGE_VERIFY_RED, (
            f'Expected block_class=MERGE_VERIFY_RED; got {kwargs.get("block_class")!r}'
        )
        assert kwargs['worktree'] == str(merge_wt), (
            f'Expected worktree={str(merge_wt)!r}; got {kwargs.get("worktree")!r}'
        )
        assert kwargs['task_id'] == req.task_id, (
            f'Expected task_id={req.task_id!r}; got {kwargs.get("task_id")!r}'
        )
        assert kwargs['config'] is req.config, 'Expected config to be req.config'
        assert kwargs['reason'].startswith('Post-merge verification failed'), (
            f'Expected reason to start with the generic prefix; '
            f'got {kwargs.get("reason")!r}'
        )

    def test_solo_train_path_does_not_spawn_without_handles(
        self, tmp_path: Path,
    ) -> None:
        """dry_run_handles=None (the solo-reverify/train module-level callers)
        must not attempt to spawn any investigation."""
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        run_dry_run_mock = AsyncMock(return_value=None)

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=COMPILE_ERROR_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(False, '')),
                ),
                patch(
                    'orchestrator.merge_queue.run_dry_run_unblock',
                    new=run_dry_run_mock,
                ),
            ):
                outcome = await _drive_verify_with_handles(
                    req, merge_wt, git_ops, dry_run_handles=None,
                )
                await asyncio.sleep(0)
                return outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.status == 'blocked'
        run_dry_run_mock.assert_not_awaited()


class TestTimeoutAndTransientInfraDoNotSpawn:
    """Step-3 (RED for the timeout case): a pure verify timeout is not a
    mechanically-fixable diff, so the generic task-fault site must NOT spawn
    an investigation for it.  Transient-infra outcomes (disk-guard-skip,
    persistent ENOSPC) already never reach the generic site by construction
    (separate early-return branches) — pinned here so a future refactor that
    moved the spawn call earlier would be caught.
    """

    def test_pure_timeout_generic_red_does_not_spawn(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        handles = _make_handles()
        run_dry_run_mock = AsyncMock(return_value=None)

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=INFRA_TIMEOUT_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.run_dry_run_unblock',
                    new=run_dry_run_mock,
                ),
            ):
                outcome = await _drive_verify_with_handles(
                    req, merge_wt, git_ops, dry_run_handles=handles,
                )
                await asyncio.sleep(0)
                if handles.background_tasks:
                    await asyncio.gather(
                        *handles.background_tasks, return_exceptions=True,
                    )
                return outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.status == 'blocked'
        run_dry_run_mock.assert_not_awaited()

    def test_disk_guard_skip_does_not_spawn(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        handles = _make_handles()
        run_dry_run_mock = AsyncMock(return_value=None)
        disk_reason = f'{TRANSIENT_INFRA_REASON_PREFIX}: pre-verify disk guard found only 0.10 GiB free'

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue._ensure_verify_disk_space',
                    new=AsyncMock(return_value=disk_reason),
                ),
                patch(
                    'orchestrator.merge_queue.run_dry_run_unblock',
                    new=run_dry_run_mock,
                ),
            ):
                outcome = await _drive_verify_with_handles(
                    req, merge_wt, git_ops, dry_run_handles=handles,
                )
                await asyncio.sleep(0)
                return outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.status == 'blocked'
        assert outcome.verify_skipped is True
        run_dry_run_mock.assert_not_awaited()

    def test_persistent_enospc_does_not_spawn(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        handles = _make_handles()
        run_dry_run_mock = AsyncMock(return_value=None)

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=PERSISTENT_ENOSPC_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.run_dry_run_unblock',
                    new=run_dry_run_mock,
                ),
            ):
                outcome = await _drive_verify_with_handles(
                    req, merge_wt, git_ops, dry_run_handles=handles,
                )
                await asyncio.sleep(0)
                return outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.status == 'blocked'
        assert outcome.reason.startswith(TRANSIENT_INFRA_REASON_PREFIX), (
            f'Expected reason to start with TRANSIENT_INFRA_REASON_PREFIX; '
            f'got {outcome.reason!r}'
        )
        run_dry_run_mock.assert_not_awaited()


class TestUnscopedTypecheckFailedSpawns:
    """Step-5 (RED): the unscoped-typecheck-FAILED sub-branch must spawn a
    dry-run investigation; the sibling TIMEOUT sub-branch, the
    flock-contention outcome, and the main-health-red outcome must NOT.
    """

    def test_unscoped_typecheck_failed_spawns(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        handles = _make_handles()
        run_dry_run_mock = AsyncMock(return_value=None)

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=UNSCOPED_TYPECHECK_FAILED_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.run_dry_run_unblock',
                    new=run_dry_run_mock,
                ),
            ):
                outcome = await _drive_verify_with_handles(
                    req, merge_wt, git_ops, dry_run_handles=handles,
                )
                await asyncio.sleep(0)
                if handles.background_tasks:
                    await asyncio.gather(
                        *handles.background_tasks, return_exceptions=True,
                    )
                return outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.status == 'blocked'
        run_dry_run_mock.assert_awaited_once()
        assert run_dry_run_mock.await_args is not None
        kwargs = run_dry_run_mock.await_args.kwargs
        assert kwargs['block_class'] == BlockClass.MERGE_VERIFY_RED, (
            f'Expected block_class=MERGE_VERIFY_RED; got {kwargs.get("block_class")!r}'
        )
        assert kwargs['reason'].startswith(
            'Post-merge verification failed: unscoped type-check failed'
        ), f'Expected unscoped-failed reason prefix; got {kwargs.get("reason")!r}'

    def test_unscoped_typecheck_timeout_does_not_spawn(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        handles = _make_handles()
        run_dry_run_mock = AsyncMock(return_value=None)

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=UNSCOPED_TYPECHECK_TIMEOUT_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.run_dry_run_unblock',
                    new=run_dry_run_mock,
                ),
            ):
                outcome = await _drive_verify_with_handles(
                    req, merge_wt, git_ops, dry_run_handles=handles,
                )
                await asyncio.sleep(0)
                return outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.status == 'blocked'
        run_dry_run_mock.assert_not_awaited()

    def test_flock_contention_does_not_spawn(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        handles = _make_handles()
        run_dry_run_mock = AsyncMock(return_value=None)

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=FLOCK_CONTENTION_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.run_dry_run_unblock',
                    new=run_dry_run_mock,
                ),
            ):
                outcome = await _drive_verify_with_handles(
                    req, merge_wt, git_ops, dry_run_handles=handles,
                )
                await asyncio.sleep(0)
                return outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.status == 'blocked'
        run_dry_run_mock.assert_not_awaited()

    def test_main_health_red_does_not_spawn(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        handles = _make_handles()
        run_dry_run_mock = AsyncMock(return_value=None)

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=COMPILE_ERROR_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(True, MAIN_SHA)),
                ),
                patch(
                    'orchestrator.merge_queue.run_dry_run_unblock',
                    new=run_dry_run_mock,
                ),
            ):
                outcome = await _drive_verify_with_handles(
                    req, merge_wt, git_ops, dry_run_handles=handles,
                )
                await asyncio.sleep(0)
                return outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.status == 'blocked'
        assert outcome.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'Expected main-health-red reason prefix; got {outcome.reason!r}'
        )
        run_dry_run_mock.assert_not_awaited()


class TestUnblockAutoDisabledSkipsSpawn:
    """Step-7 (RED): unblock_auto.enabled=False must suppress the spawn even
    when dry_run_handles carries a live scheduler."""

    def test_unblock_auto_disabled_skips_spawn(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        config.unblock_auto.enabled = False
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        handles = _make_handles()
        run_dry_run_mock = AsyncMock(return_value=None)

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=COMPILE_ERROR_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(False, '')),
                ),
                patch(
                    'orchestrator.merge_queue.run_dry_run_unblock',
                    new=run_dry_run_mock,
                ),
            ):
                outcome = await _drive_verify_with_handles(
                    req, merge_wt, git_ops, dry_run_handles=handles,
                )
                await asyncio.sleep(0)
                return outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.status == 'blocked'
        run_dry_run_mock.assert_not_awaited()


class TestInflightDedupSkipsDuplicate:
    """Step-9 (RED): a not-done investigation task already registered under
    the same 'unblock-auto-<task_id>' name must suppress a second spawn."""

    def test_inflight_dedup_skips_duplicate(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        handles = _make_handles()
        run_dry_run_mock = AsyncMock(return_value=None)

        async def _run() -> MergeOutcome | None:
            async def _hang_forever() -> None:
                await asyncio.Event().wait()

            dummy_task = asyncio.create_task(
                _hang_forever(), name=f'unblock-auto-{req.task_id}',
            )
            handles.background_tasks.add(dummy_task)
            try:
                with (
                    patch(
                        'orchestrator.merge_queue.run_scoped_verification',
                        new=AsyncMock(return_value=COMPILE_ERROR_RESULT),
                    ),
                    patch(
                        'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                        new=AsyncMock(return_value=(False, '')),
                    ),
                    patch(
                        'orchestrator.merge_queue.run_dry_run_unblock',
                        new=run_dry_run_mock,
                    ),
                ):
                    outcome = await _drive_verify_with_handles(
                        req, merge_wt, git_ops, dry_run_handles=handles,
                    )
                    await asyncio.sleep(0)
                    return outcome
            finally:
                dummy_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await dummy_task

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.status == 'blocked'
        run_dry_run_mock.assert_not_awaited()


class TestWorkerStoresAndThreadsHandles:
    """Step-11 (RED): SpeculativeMergeWorker accepts optional
    scheduler/mcp/usage_gate/cost_store handles (harness-owned; the worker
    itself holds none of them today), stores them, and bundles them into a
    single self._dry_run_handles so _run_post_merge_verify stays a pure git
    engine (mirrors the train_callback_factory opaque-injection pattern at
    harness.py:6468-6472 — the worker never imports the scheduler).
    """

    def test_worker_stores_handles_and_threads_them(self, tmp_path: Path) -> None:
        git_ops = _make_git_ops(tmp_path)
        scheduler = MagicMock()
        mcp = MagicMock()
        usage_gate = MagicMock()
        cost_store = MagicMock()

        worker = SpeculativeMergeWorker(
            git_ops, asyncio.Queue(),
            scheduler=scheduler, mcp=mcp,
            usage_gate=usage_gate, cost_store=cost_store,
        )

        assert worker._scheduler is scheduler
        assert worker._mcp is mcp
        assert worker._usage_gate is usage_gate
        assert worker._cost_store is cost_store
        assert isinstance(worker._background_tasks, set)
        assert worker._dry_run_handles.scheduler is scheduler
        assert worker._dry_run_handles.mcp is mcp
        assert worker._dry_run_handles.usage_gate is usage_gate
        assert worker._dry_run_handles.cost_store is cost_store
        assert worker._dry_run_handles.background_tasks is worker._background_tasks, (
            'handles.background_tasks must be the SAME set instance as '
            'worker._background_tasks — a spawned investigation task strong-ref '
            'must live exactly as long as the worker'
        )

    def test_minimal_construction_still_works(self, tmp_path: Path) -> None:
        """The existing git_ops+queue-only construction convention (used
        throughout the test suite) must stay green — the four new handle
        params default to None."""
        git_ops = _make_git_ops(tmp_path)
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())

        assert worker._scheduler is None
        assert worker._mcp is None
        assert worker._usage_gate is None
        assert worker._cost_store is None
        assert worker._dry_run_handles.scheduler is None
        assert worker._dry_run_handles.mcp is None


class TestRunInflightVerifyPassesHandles:
    """Step-11 (RED): the production SpeculativeMergeWorker._run_inflight_verify
    call site must pass dry_run_handles=self._dry_run_handles into
    _run_post_merge_verify (mirrors
    test_merge_queue_multihost_wiring.py's
    TestRunInflightVerifyThreadsEscalationQueue, which pins the same
    call-site-threading shape for self._escalation_queue).
    """

    def test_run_inflight_verify_passes_handles(self, tmp_path: Path) -> None:
        git_ops = _make_git_ops(tmp_path)
        config = _make_config(tmp_path)
        scheduler = MagicMock()
        worker = SpeculativeMergeWorker(
            git_ops, asyncio.Queue(), scheduler=scheduler,
        )

        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        merge_result = MagicMock()
        merge_result.merge_commit = 'abc123def456789abc1'
        item = RealMergeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_wt,
            base_sha='base123',
            speculative=False,
        )

        # REMOTE lease — bypasses the local warm-swap path, keeping this
        # test focused on the dry_run_handles kwarg (mirrors
        # test_merge_queue_multihost_wiring.py's REMOTE-lease builder).
        fake_runner = MagicMock()
        fake_runner.name = 'leo-laptop'
        fake_runner.is_local = False
        lease = HostLease(name='leo-laptop', runner=fake_runner, is_local=False)

        spy = AsyncMock(return_value=None)

        async def _run() -> None:
            with patch('orchestrator.merge_queue._run_post_merge_verify', new=spy):
                await worker._run_inflight_verify(item, lease)

        asyncio.run(_run())

        assert spy.await_args is not None, '_run_post_merge_verify was not called'
        assert spy.await_args.kwargs.get('dry_run_handles') is worker._dry_run_handles, (
            '_run_inflight_verify must pass dry_run_handles=self._dry_run_handles '
            'into _run_post_merge_verify — the worker holds the bundled handles '
            'but never threaded them through'
        )


class TestHarnessWiresDryRunHandlesIntoWorker:
    """Step-13 (RED): _start_merge_worker forwards harness.scheduler/mcp/
    usage_gate/cost_store into SpeculativeMergeWorker.

    Mirrors test_harness_train_callbacks.py's TestHarnessWiring, which pins
    the same _start_merge_worker construction-site-threading shape for
    train_callback_factory.
    """

    def test_harness_wires_dry_run_handles_into_worker(self, tmp_path: Path) -> None:
        import asyncio as _asyncio

        from orchestrator.config import OrchestratorConfig
        from orchestrator.event_store import EventStore
        from orchestrator.harness import Harness

        config = OrchestratorConfig(project_root=tmp_path)
        harness = Harness(config)
        harness.event_store = EventStore(tmp_path / 'events.db', 'run-wiring-0001')

        sentinel_scheduler = MagicMock()
        sentinel_mcp = MagicMock()
        sentinel_usage_gate = MagicMock()
        sentinel_cost_store = MagicMock()
        harness.scheduler = sentinel_scheduler
        harness.mcp = sentinel_mcp
        harness.usage_gate = sentinel_usage_gate
        harness.cost_store = sentinel_cost_store

        harness.git_ops = MagicMock()
        harness.git_ops.project_root = None

        captured: dict[str, object] = {}

        class CapturingWorker:
            def __init__(self, *args: object, **kwargs: object) -> None:
                captured.update(kwargs)

            async def run(self) -> None:
                await _asyncio.sleep(0)

            async def stop(self) -> None:
                pass

        async def _run() -> None:
            with (
                patch(
                    'orchestrator.merge_queue.SpeculativeMergeWorker',
                    CapturingWorker,
                ),
                patch('orchestrator.merge_queue.enforce_merge_liveness_margin'),
                patch('orchestrator.merge_queue.enforce_persistent_worktree_serial_lane'),
                patch.object(
                    harness,
                    '_build_service_restart_coordinator',
                    return_value=MagicMock(note_merge=AsyncMock()),
                ),
            ):
                await harness._start_merge_worker()
            await harness._stop_merge_worker()

        asyncio.run(_run())

        assert captured.get('scheduler') is sentinel_scheduler, (
            f'Expected scheduler=harness.scheduler; got captured={captured!r}'
        )
        assert captured.get('mcp') is sentinel_mcp, (
            f'Expected mcp=harness.mcp; got captured={captured!r}'
        )
        assert captured.get('usage_gate') is sentinel_usage_gate, (
            f'Expected usage_gate=harness.usage_gate; got captured={captured!r}'
        )
        assert captured.get('cost_store') is sentinel_cost_store, (
            f'Expected cost_store=harness.cost_store; got captured={captured!r}'
        )


class TestMergeVerifyRedProducesGateableProposal:
    """Step-15 (RED — event_store not yet forwarded): the PRD test-9 capstone.

    End-to-end: a merge-verify RED on a trivial scoped diff must, via the
    REAL (unpatched) run_dry_run_unblock, write a dry_run_proposals[] entry
    with block_class='merge_verify_red' that b3_gate.check_proposal accepts
    as non-ABORT — AND the investigation must emit the same
    invocation_end/'blocked' telemetry event the agent-block path emits
    (observability parity).  Today _spawn_merge_verify_dry_run receives
    event_store but discards it (`_ = event_store`), so the investigation's
    emit never fires through this MagicMock — RED on the event assertion.

    Reuses test_dry_run_unblock.py's real-git-repo/_make_agent_result/
    _RecordingScheduler e2e pattern (only orchestrator.dry_run_unblock.
    invoke_agent is mocked) so the proposal is genuinely produced and
    genuinely gated, not merely spawn-mocked like the tests above.
    """

    def test_merge_verify_red_produces_gateable_proposal(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        head_sha = _init_git_repo(merge_wt)
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        scheduler = _RecordingScheduler({'dry_run_proposals': []})
        handles = _DryRunInvestigationHandles(scheduler=scheduler)
        event_store = MagicMock()

        structured = {
            'proposal_text': 'Fix the scoped lint failure',
            'risk_label': 'low',
            'files_referenced': ['orchestrator/src/orchestrator/foo.py'],
        }
        agent_result = _make_agent_result(structured_output=structured)

        def _fake_run_git(args: list[str], cwd: str) -> tuple[int, str]:
            """HEAD always matches the recorded sha; footprint diff is empty."""
            if 'rev-parse' in args:
                return (0, head_sha)
            return (0, '')

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=COMPILE_ERROR_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(False, '')),
                ),
                patch(
                    'orchestrator.dry_run_unblock.invoke_agent',
                    new=AsyncMock(return_value=agent_result),
                ),
            ):
                outcome = await _run_post_merge_verify(
                    git_ops, req, merge_wt,
                    timeouts={},
                    enospc_retries={},
                    max_timeouts=3,
                    max_enospc=1,
                    dry_run_handles=handles,
                    event_store=event_store,
                )
                # Drain the fire-and-forget investigation (real run_dry_run_unblock,
                # real git subprocess calls against merge_wt, mocked invoke_agent)
                # inside this same loop before asyncio.run() tears it down.
                await asyncio.sleep(0)
                if handles.background_tasks:
                    await asyncio.gather(
                        *handles.background_tasks, return_exceptions=True,
                    )
                return outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.status == 'blocked'

        proposals = scheduler._meta.get('dry_run_proposals', [])
        assert proposals, 'Expected a dry_run_proposals entry to be written'
        entry = proposals[-1]
        assert entry['block_class'] == 'merge_verify_red', (
            f"Expected block_class='merge_verify_red'; got {entry.get('block_class')!r}"
        )
        assert entry['risk_label'] == 'low', (
            f"Expected risk_label='low'; got {entry.get('risk_label')!r}"
        )

        verdict = check_proposal(
            entry, worktree=str(merge_wt), category='task_failure',
            run_git=_fake_run_git,
        )
        assert verdict['verdict'] != ABORT, (
            f'Expected a non-ABORT (gateable) verdict; got {verdict!r}'
        )

        # Observability parity: the investigation must emit the same
        # invocation_end/'blocked' event the agent-block path emits (other
        # emit calls — e.g. EventType.merge_verify from the verify pool —
        # are expected and ignored here).
        invocation_end_calls = [
            c for c in event_store.emit.call_args_list
            if c.args and c.args[0] == EventType.invocation_end
        ]
        assert invocation_end_calls, (
            f'Expected an invocation_end event; got calls='
            f'{event_store.emit.call_args_list!r}'
        )
        assert invocation_end_calls[-1].kwargs.get('phase') == 'blocked'
        assert invocation_end_calls[-1].kwargs.get('role') == 'unblock_auto'
