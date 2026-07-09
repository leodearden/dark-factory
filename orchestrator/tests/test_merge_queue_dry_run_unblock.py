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
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from test_merge_queue_main_health import (
    COMPILE_ERROR_RESULT,
    _make_config,
    _make_git_ops,
    _make_req,
)

from orchestrator.merge_queue import (
    MergeOutcome,
    MergeRequest,
    _DryRunInvestigationHandles,
    _run_post_merge_verify,
)
from orchestrator.unblock_types import BlockClass


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
