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
import os
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _merge_lane_fakes import FakeVerifier, VerifyScript, make_lane
from _orch_helpers import wait_responsive
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
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps
from orchestrator.git_ops import _run as _run_git
from orchestrator.merge_queue import (
    MAIN_HEALTH_RED_REASON_PREFIX,
    PRODUCTION_CLOCK,
    PRODUCTION_VERIFIER,
    TRANSIENT_INFRA_REASON_PREFIX,
    MergeOutcome,
    MergeRequest,
    QueuedBranch,
    _DryRunInvestigationHandles,
    _run_post_merge_verify,
)
from orchestrator.unblock_types import BlockClass
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import (
    FLOCK_CONTENTION_CATEGORY,
    UNSCOPED_TYPECHECK_FAILED_CATEGORY,
    UNSCOPED_TYPECHECK_TIMEOUT_CATEGORY,
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


class _SpawningVerifier(FakeVerifier):
    """``FakeVerifier`` whose ``dry_run_unblock`` is the REAL investigation.

    The capstones below assert on what ``run_dry_run_unblock`` actually
    WRITES -- the scheduler's ``dry_run_proposals`` blob, the
    ``invocation_end`` event -- so the port must run the production
    investigation rather than record it. Everything else (the scoped verify
    result, the disk guard) stays scripted, and the spawn is still recorded
    in ``investigations``.
    """

    def dry_run_unblock(self, **investigation: Any):
        self.investigations.append(investigation)
        return PRODUCTION_VERIFIER.dry_run_unblock(**investigation)


class _SpawnIsdirVerifier(FakeVerifier):
    """``FakeVerifier`` that reads the spawned worktree's liveness AT SPAWN.

    ``test_investigation_uses_retained_task_worktree_surviving_real_cleanup``
    is about which worktree the investigation is handed while the ephemeral
    merge worktree is being removed under it, so the ``isdir`` probe has to
    happen when the port is called, not after the test has finished.
    """

    def __init__(self) -> None:
        super().__init__(default=VerifyScript(result=COMPILE_ERROR_RESULT))
        self.isdir_at_spawn: list[bool] = []

    def dry_run_unblock(self, **investigation: object):
        worktree = investigation.get('worktree')
        self.isdir_at_spawn.append(
            isinstance(worktree, str) and os.path.isdir(worktree),
        )
        return super().dry_run_unblock(**investigation)


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
    verifier: FakeVerifier,
    dry_run_handles: _DryRunInvestigationHandles | None,
) -> MergeOutcome | None:
    """test_merge_queue_main_health._drive_verify + the dry_run_handles kwarg.

    *verifier* is the injected ``VerifyPort``: it decides the scoped verify's
    result, answers the disk guard, and records (instead of running) the
    dry-run investigation the block path spawns.
    """
    return await _run_post_merge_verify(
        git_ops, req, merge_wt,
        timeouts={},
        enospc_retries={},
        max_timeouts=3,
        max_enospc=1,
        dry_run_handles=dry_run_handles,
        verifier=verifier,
    )


class TestGenericMergeVerifyRedSpawnsDryRun:
    """Step-1 (RED): the generic task-fault block site must spawn
    run_dry_run_unblock with block_class=MERGE_VERIFY_RED when handles are
    live, and must NOT spawn it when dry_run_handles=None (solo/train path).
    """

    def test_generic_merge_verify_red_spawns_dry_run(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, escalate_preexisting=False)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        handles = _make_handles()
        verifier = FakeVerifier(default=VerifyScript(result=COMPILE_ERROR_RESULT))

        async def _run() -> MergeOutcome | None:
            outcome = await _drive_verify_with_handles(
                req, merge_wt, git_ops, verifier=verifier, dry_run_handles=handles,
            )
            # Let the fire-and-forget create_task run to completion
            # (the injected port's investigation resolves immediately) inside this SAME
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
        assert len(verifier.investigations) == 1, verifier.investigations
        kwargs = verifier.investigations[0]
        assert kwargs['block_class'] == BlockClass.MERGE_VERIFY_RED, (
            f'Expected block_class=MERGE_VERIFY_RED; got {kwargs.get("block_class")!r}'
        )
        assert kwargs['worktree'] == str(req.worktree), (
            f'Expected worktree={str(req.worktree)!r} (the task\'s own '
            f'retained worktree, not the ephemeral merge_wt); '
            f'got {kwargs.get("worktree")!r}'
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
        config = _make_config(tmp_path, escalate_preexisting=False)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        verifier = FakeVerifier(default=VerifyScript(result=COMPILE_ERROR_RESULT))

        async def _run() -> MergeOutcome | None:
            outcome = await _drive_verify_with_handles(
                req, merge_wt, git_ops, verifier=verifier, dry_run_handles=None,
            )
            await asyncio.sleep(0)
            return outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.status == 'blocked'
        assert verifier.investigations == [], verifier.investigations


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
        verifier = FakeVerifier(default=VerifyScript(result=INFRA_TIMEOUT_RESULT))

        async def _run() -> MergeOutcome | None:
            outcome = await _drive_verify_with_handles(
                req, merge_wt, git_ops, verifier=verifier, dry_run_handles=handles,
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
        assert verifier.investigations == [], verifier.investigations

    def test_disk_guard_skip_does_not_spawn(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        handles = _make_handles()
        disk_reason = f'{TRANSIENT_INFRA_REASON_PREFIX}: pre-verify disk guard found only 0.10 GiB free'

        verifier = FakeVerifier(disk_reason=disk_reason)

        async def _run() -> MergeOutcome | None:
            outcome = await _drive_verify_with_handles(
                req, merge_wt, git_ops, verifier=verifier, dry_run_handles=handles,
            )
            await asyncio.sleep(0)
            return outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.status == 'blocked'
        assert outcome.verify_skipped is True
        assert verifier.investigations == [], verifier.investigations

    def test_persistent_enospc_does_not_spawn(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        handles = _make_handles()
        verifier = FakeVerifier(default=VerifyScript(result=PERSISTENT_ENOSPC_RESULT))

        async def _run() -> MergeOutcome | None:
            outcome = await _drive_verify_with_handles(
                req, merge_wt, git_ops, verifier=verifier, dry_run_handles=handles,
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
        assert verifier.investigations == [], verifier.investigations


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
        verifier = FakeVerifier(default=VerifyScript(result=UNSCOPED_TYPECHECK_FAILED_RESULT))

        async def _run() -> MergeOutcome | None:
            outcome = await _drive_verify_with_handles(
                req, merge_wt, git_ops, verifier=verifier, dry_run_handles=handles,
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
        assert len(verifier.investigations) == 1, verifier.investigations
        kwargs = verifier.investigations[0]
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
        verifier = FakeVerifier(default=VerifyScript(result=UNSCOPED_TYPECHECK_TIMEOUT_RESULT))

        async def _run() -> MergeOutcome | None:
            outcome = await _drive_verify_with_handles(
                req, merge_wt, git_ops, verifier=verifier, dry_run_handles=handles,
            )
            await asyncio.sleep(0)
            return outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.status == 'blocked'
        assert verifier.investigations == [], verifier.investigations

    def test_flock_contention_does_not_spawn(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        handles = _make_handles()
        verifier = FakeVerifier(default=VerifyScript(result=FLOCK_CONTENTION_RESULT))

        async def _run() -> MergeOutcome | None:
            outcome = await _drive_verify_with_handles(
                req, merge_wt, git_ops, verifier=verifier, dry_run_handles=handles,
            )
            await asyncio.sleep(0)
            return outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.status == 'blocked'
        assert verifier.investigations == [], verifier.investigations

    def test_main_health_red_does_not_spawn(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        handles = _make_handles()
        verifier = FakeVerifier(default=VerifyScript(result=COMPILE_ERROR_RESULT))

        async def _run() -> MergeOutcome | None:
            # The ONE patch this file still needs (PRD γ9 residual). Every
            # sibling arm produces its condition for real -- the RED verify
            # through the injected port, "not pre-existing" through
            # `escalate_preexisting_main_break=False`, the disk skip through
            # the port's disk guard -- but a genuinely RED main has no such
            # seam: `verify_failure_is_preexisting_on_main` lives in
            # `orchestrator.verify`, takes no port, and probes by running a
            # real scoped verification in a real `_mainprobe-` worktree.
            with (
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(True, MAIN_SHA)),
                ),
            ):
                outcome = await _drive_verify_with_handles(
                    req, merge_wt, git_ops, verifier=verifier, dry_run_handles=handles,
                )
                await asyncio.sleep(0)
                return outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.status == 'blocked'
        assert outcome.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'Expected main-health-red reason prefix; got {outcome.reason!r}'
        )
        assert verifier.investigations == [], verifier.investigations


class TestUnblockAutoDisabledSkipsSpawn:
    """Step-7 (RED): unblock_auto.enabled=False must suppress the spawn even
    when dry_run_handles carries a live scheduler."""

    def test_unblock_auto_disabled_skips_spawn(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, escalate_preexisting=False)
        config.unblock_auto.enabled = False
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        handles = _make_handles()
        verifier = FakeVerifier(default=VerifyScript(result=COMPILE_ERROR_RESULT))

        async def _run() -> MergeOutcome | None:
            outcome = await _drive_verify_with_handles(
                req, merge_wt, git_ops, verifier=verifier, dry_run_handles=handles,
            )
            await asyncio.sleep(0)
            return outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.status == 'blocked'
        assert verifier.investigations == [], verifier.investigations


class TestInflightDedupSkipsDuplicate:
    """Step-9 (RED): a not-done investigation task already registered under
    the same 'unblock-auto-<task_id>' name must suppress a second spawn."""

    def test_inflight_dedup_skips_duplicate(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, escalate_preexisting=False)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        handles = _make_handles()
        verifier = FakeVerifier(default=VerifyScript(result=COMPILE_ERROR_RESULT))

        async def _run() -> MergeOutcome | None:
            async def _hang_forever() -> None:
                await asyncio.Event().wait()

            dummy_task = asyncio.create_task(
                _hang_forever(), name=f'unblock-auto-{req.task_id}',
            )
            handles.background_tasks.add(dummy_task)
            try:
                outcome = await _drive_verify_with_handles(
                    req, merge_wt, git_ops, verifier=verifier, dry_run_handles=handles,
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
        assert verifier.investigations == [], verifier.investigations


async def _lane_repo(tmp_path: Path) -> tuple[GitOps, OrchestratorConfig]:
    """A real one-commit repo, its GitOps and a matching config.

    The handles contract below is about what the LANE hands the
    investigation, so it is driven through the lane's own public path (queue
    in, MergeOutcome out) over a real repository -- the idiom
    test_merge_lane_package.py established for the ports.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    await _run_git(['git', 'init', '-b', 'main'], cwd=repo)
    await _run_git(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run_git(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run_git(['git', 'add', '-A'], cwd=repo)
    await _run_git(['git', 'commit', '-m', 'Initial commit'], cwd=repo)

    git_config = GitConfig(
        main_branch='main', branch_prefix='task/', remote='origin',
        worktree_dir='.worktrees', push_after_advance=False,
    )
    git_ops = GitOps(git_config, repo)
    config = OrchestratorConfig(
        project_root=repo,
        git=git_config,
        # The main-health probe is out of scope here and would run a REAL
        # scoped verification against a probe worktree; the production guard
        # that skips it is this flag (_classify_main_health_red).
        escalate_preexisting_main_break=False,
    )
    return git_ops, config


async def _drive_red_merge_through_lane(
    git_ops: GitOps,
    config: OrchestratorConfig,
    verifier: FakeVerifier,
    **lane_handles: object,
) -> MergeOutcome:
    """Land one branch through a real lane whose verify is scripted RED."""
    worktree = (await git_ops.create_worktree('99')).path
    (worktree / 'work.py').write_text('x = 1\n')
    await git_ops.commit(worktree, 'Add work.py')

    queue: asyncio.Queue = asyncio.Queue()
    lane = make_lane(
        git_ops, queue, verifier=verifier, clock=PRODUCTION_CLOCK, **lane_handles,
    )
    lane_task = asyncio.create_task(lane.run())
    try:
        req = MergeRequest(
            task_id='99',
            branch=QueuedBranch.parse('task/99', config.git.branch_prefix),
            worktree=worktree,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=asyncio.get_running_loop().create_future(),
        )
        await queue.put(req)
        return await wait_responsive(req.result, label='merge outcome')
    finally:
        await lane.stop()
        await lane_task


@pytest.mark.asyncio
class TestLaneHandsItsHandlesToTheInvestigation:
    """The lane bundles the harness-owned scheduler/mcp/usage_gate/cost_store
    it was constructed with and hands them to the investigation it spawns.

    Replaces three construction-site tests that read
    ``worker._scheduler`` / ``._mcp`` / ``._usage_gate`` / ``._cost_store`` /
    ``._dry_run_handles`` / ``._background_tasks`` and spied on
    ``_run_post_merge_verify`` to watch ``dry_run_handles`` being threaded:
    the same contract, observed where it actually matters -- on the
    investigation the port is asked to run.
    """

    async def test_a_red_merge_spawns_the_investigation_with_the_lane_handles(
        self, tmp_path: Path,
    ) -> None:
        git_ops, config = await _lane_repo(tmp_path)
        scheduler, mcp = MagicMock(), MagicMock()
        usage_gate, cost_store = MagicMock(), MagicMock()
        verifier = FakeVerifier(
            default=VerifyScript(result=COMPILE_ERROR_RESULT),
        )

        outcome = await _drive_red_merge_through_lane(
            git_ops, config, verifier,
            scheduler=scheduler, mcp=mcp,
            usage_gate=usage_gate, cost_store=cost_store,
        )

        assert outcome.status == 'blocked', outcome
        assert len(verifier.investigations) == 1, verifier.investigations
        spawned = verifier.investigations[0]
        assert spawned['block_class'] == BlockClass.MERGE_VERIFY_RED
        assert spawned['task_id'] == '99'
        assert (
            spawned['scheduler'], spawned['mcp'],
            spawned['usage_gate'], spawned['cost_store'],
        ) == (scheduler, mcp, usage_gate, cost_store), (
            'the investigation must be spawned with the handles the lane was '
            f'constructed with; got {spawned!r}'
        )

    async def test_a_lane_built_without_handles_spawns_nothing(
        self, tmp_path: Path,
    ) -> None:
        """The git_ops+queue-only construction convention stays green: with no
        scheduler there is nobody to file a proposal with, so the same RED
        merge spawns no investigation."""
        git_ops, config = await _lane_repo(tmp_path)
        verifier = FakeVerifier(
            default=VerifyScript(result=COMPILE_ERROR_RESULT),
        )

        outcome = await _drive_red_merge_through_lane(git_ops, config, verifier)

        assert outcome.status == 'blocked', outcome
        assert verifier.investigations == [], verifier.investigations



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
            # The other γ9 residual: the contract IS the construction site, and
            # Harness offers no seam for the worker it builds (no factory
            # argument, no accessor) -- so substituting the class is the only
            # way to see the kwargs it is constructed with. Observing this
            # behaviourally would mean driving a real merge through a worker
            # whose verify port the harness gives us no way to inject.
            with (
                patch(
                    'orchestrator.merge_queue.SpeculativeMergeWorker',
                    CapturingWorker,
                ),
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
        config = _make_config(tmp_path, escalate_preexisting=False)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        # The investigation reads req.worktree (the task's OWN retained
        # worktree), not the ephemeral merge_wt — git init creates the
        # task-wt dir, so no separate mkdir is needed.
        head_sha = _init_git_repo(req.worktree)

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

        verifier = _SpawningVerifier(default=VerifyScript(result=COMPILE_ERROR_RESULT))

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.dry_run_unblock.invoke_agent',
                    new=AsyncMock(return_value=agent_result),
                ),
                # task 2633: run_dry_run_unblock clamps a low-risk
                # MERGE_VERIFY_RED proposal to 'human-review-required' unless
                # the run's event history proves merge-completion eligibility
                # ((b) a passing workflow_verify AND (c) a phase_enter(merge)).
                # This golden test models the ELIGIBLE happy path (verify+review
                # passed, landing jammed), so stub the predicate True and the
                # proposal stays a gateable low-risk one. The ineligible→clamp
                # path is pinned by TestMergeVerifyRedClampedWithoutCompletionEvidence.
                patch(
                    'orchestrator.dry_run_unblock.merge_completion_eligible',
                    return_value=True,
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
                    verifier=verifier,
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

        # What the investigation WROTE, read off the recording scheduler's own
        # call log rather than out of the double's internal blob.
        writes = [
            c for c in scheduler.update_task_calls
            if 'dry_run_proposals' in c['metadata']
        ]
        assert writes, 'Expected a dry_run_proposals entry to be written'
        entry = writes[-1]['metadata']['dry_run_proposals'][-1]
        assert entry['block_class'] == 'merge_verify_red', (
            f"Expected block_class='merge_verify_red'; got {entry.get('block_class')!r}"
        )
        assert entry['risk_label'] == 'low', (
            f"Expected risk_label='low'; got {entry.get('risk_label')!r}"
        )

        verdict = check_proposal(
            entry, worktree=str(req.worktree), category='task_failure',
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


class TestMergeVerifyRedClampedWithoutCompletionEvidence:
    """task 2633 clamp (a33b6e1e4d): the ineligible counterpart to the golden
    test above.

    Same end-to-end merge-verify-RED path and same agent-labelled
    ``risk_label='low'`` proposal, but the run's event history carries NO
    merge-completion evidence (a bare MagicMock event_store — the REAL
    ``merge_completion_eligible`` reads it and returns False). ``run_dry_run_
    unblock`` must therefore clamp the proposal to ``'human-review-required'``,
    and ``b3_gate.check_proposal`` must then ABORT it (risk_label != 'low'),
    routing the merge-verify-RED to a human /unblock instead of autonomous
    consumption. Pins the clamp the clamp commit itself left untested.
    """

    def test_merge_verify_red_without_completion_evidence_clamps_to_abort(
        self, tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path, escalate_preexisting=False)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        head_sha = _init_git_repo(req.worktree)

        scheduler = _RecordingScheduler({'dry_run_proposals': []})
        handles = _DryRunInvestigationHandles(scheduler=scheduler)
        # Bare event_store: no workflow_verify(passed) and no phase_enter(merge)
        # rows -> the real merge_completion_eligible returns False -> clamp fires.
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

        verifier = _SpawningVerifier(default=VerifyScript(result=COMPILE_ERROR_RESULT))

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.dry_run_unblock.invoke_agent',
                    new=AsyncMock(return_value=agent_result),
                ),
                # NOTE: merge_completion_eligible is deliberately NOT patched
                # here — the real predicate runs against the evidence-free
                # event_store and returns False, so the clamp fires.
            ):
                outcome = await _run_post_merge_verify(
                    git_ops, req, merge_wt,
                    timeouts={},
                    enospc_retries={},
                    max_timeouts=3,
                    max_enospc=1,
                    dry_run_handles=handles,
                    event_store=event_store,
                    verifier=verifier,
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

        # What the investigation WROTE, read off the recording scheduler's own
        # call log rather than out of the double's internal blob.
        writes = [
            c for c in scheduler.update_task_calls
            if 'dry_run_proposals' in c['metadata']
        ]
        assert writes, 'Expected a dry_run_proposals entry to be written'
        entry = writes[-1]['metadata']['dry_run_proposals'][-1]
        assert entry['block_class'] == 'merge_verify_red', (
            f"Expected block_class='merge_verify_red'; got {entry.get('block_class')!r}"
        )
        # The clamp downgraded the agent's 'low' -> 'human-review-required'
        # because (b)+(c) evidence is absent.
        assert entry['risk_label'] == 'human-review-required', (
            f"Expected clamp to 'human-review-required'; got {entry.get('risk_label')!r}"
        )

        # b3_gate must ABORT the clamped proposal (risk_label != 'low'), so it
        # is never autonomously consumed — it falls to a human /unblock.
        verdict = check_proposal(
            entry, worktree=str(req.worktree), category='task_failure',
            run_git=_fake_run_git,
        )
        assert verdict['verdict'] == ABORT, (
            f'Expected an ABORT (non-gateable) verdict for the clamped '
            f'human-review-required proposal; got {verdict!r}'
        )


class TestInvestigationWorktreeSurvivesCleanup:
    """Step-17 (RED): regression for the blocking review finding
    (robustness_worktree_lifecycle).

    The reviewer's mandated 'at minimum' test: with a REAL
    ``cleanup_merge_worktree`` (not the test suite's usual no-op AsyncMock,
    which is exactly what hid this bug), *merge_wt* is actually removed
    before the fire-and-forget investigation task runs. The investigation
    must be pointed at the task's OWN retained worktree (``req.worktree`` —
    which is never touched by ``_run_post_merge_verify`` and survives while
    the task stays blocked), not the ephemeral merge worktree that has
    already been deleted by the time the spawned task reads it.
    """

    def test_investigation_uses_retained_task_worktree_surviving_real_cleanup(
        self, tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path, escalate_preexisting=False)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        task_wt = tmp_path / 'task-wt'
        task_wt.mkdir()
        req = _make_req('99', task_wt, config)

        # Faithfully simulate the real ephemeral `git worktree remove
        # --force` at merge_queue.py:1156 — the no-op mock installed by
        # _make_git_ops is exactly what hid this bug (merge_wt silently
        # "survived" for the whole test even though production removes it).
        async def _real_cleanup(wt: Path) -> None:
            shutil.rmtree(wt)

        git_ops.cleanup_merge_worktree = AsyncMock(side_effect=_real_cleanup)

        handles = _make_handles()
        verifier = _SpawnIsdirVerifier()

        async def _run() -> MergeOutcome | None:
            outcome = await _drive_verify_with_handles(
                req, merge_wt, git_ops, verifier=verifier, dry_run_handles=handles,
            )
            # Let the fire-and-forget create_task run to completion
            # inside this SAME loop before asyncio.run() tears it down.
            await asyncio.sleep(0)
            if handles.background_tasks:
                await asyncio.gather(
                    *handles.background_tasks, return_exceptions=True,
                )
            return outcome

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.status == 'blocked'
        assert not merge_wt.exists(), (
            'Expected the real cleanup_merge_worktree to have removed merge_wt'
        )
        assert len(verifier.investigations) == 1, verifier.investigations
        spawned = verifier.investigations[0]
        assert spawned.get('worktree') == str(task_wt), (
            f"Expected worktree={str(task_wt)!r} (the task's own retained "
            f"worktree, req.worktree); got {spawned.get('worktree')!r}"
        )
        assert verifier.isdir_at_spawn == [True], (
            'Expected the path handed to the investigation to still exist '
            'on disk when the investigation read it'
        )
