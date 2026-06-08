"""Tests for the dry-run unblock hook wired into _mark_blocked.

Pins three contracts:
1. When unblock_auto.enabled=True, _mark_blocked schedules a background
   asyncio.create_task for run_dry_run_unblock and returns immediately
   (fire-and-forget — _mark_blocked does NOT await the dry-run).
2. merge_phase=True suppresses the hook entirely.
3. unblock_auto.enabled=False suppresses the hook but does NOT suppress the
   real 'blocked' status write.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow


class _Scheduler:
    """Minimal fake scheduler satisfying _SchedulerLike for workflow tests."""

    def __init__(self) -> None:
        self.statuses: dict[str, list[str]] = {}
        self.update_calls: list[dict] = []

    async def set_task_status(self, tid, status, **_kw):
        self.statuses.setdefault(tid, []).append(status)

    async def get_status(self, tid):
        hist = self.statuses.get(tid, [])
        return hist[-1] if hist else None

    async def update_task(self, task_id, metadata, *, append=False):
        self.update_calls.append({'task_id': task_id, 'metadata': metadata, 'append': append})
        return True

    async def get_task(self, tid):
        return None

    async def mark_done(self, task_id, /, *, kind, sha, note=None):
        pass

    async def handle_blast_radius_expansion(self, task_id, current, needed, /):
        return False

    async def dispatch_tool(self, name, arguments, *, timeout=30.0):
        return {}

    def release(self, tid):
        pass

    async def get_tasks(self) -> list[dict]:
        return []

    async def get_statuses(
        self, ids: list[str] | None = None,
    ) -> tuple[dict[str, str], Exception | None]:
        return ({}, None)

    async def tasks_by_train(self, train_id: str, /) -> list[dict]:
        return []

    def clear_requeue_count(self, task_id: str, /) -> None:
        pass


def _make_workflow(*, tmp_path: Path, task_id: str = '42',
                   enabled: bool = True) -> tuple[TaskWorkflow, _Scheduler]:
    """Minimal TaskWorkflow with controllable unblock_auto flag."""
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = tmp_path / 'proj'
    config.unblock_auto.enabled = enabled

    git_ops = MagicMock()

    scheduler = _Scheduler()

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    wf.artifacts = TaskArtifacts(worktree)
    wf.worktree = worktree
    wf.plan = {'files': []}
    wf._module_configs = []
    # No escalation queue so _mark_blocked doesn't try to submit escalations
    wf.escalation_queue = None
    wf.event_store = None
    return wf, scheduler


# ---------------------------------------------------------------------------
# step-17: fire-and-forget contract
# ---------------------------------------------------------------------------

class TestMarkBlockedSpawnsFireAndForget:
    @pytest.mark.asyncio
    async def test_mark_blocked_spawns_fire_and_forget_dry_run(self, tmp_path):
        wf, scheduler = _make_workflow(tmp_path=tmp_path, task_id='42', enabled=True)

        # Track invocation arguments, then hang briefly to prove fire-and-forget
        calls = []
        hang_event = asyncio.Event()

        async def _hanging_dry_run(**kwargs):
            calls.append(kwargs)
            await hang_event.wait()  # hang until we release it

        with patch('orchestrator.workflow.run_dry_run_unblock', new=_hanging_dry_run):
            # _mark_blocked should return quickly even though _hanging_dry_run hangs
            result = await asyncio.wait_for(
                wf._mark_blocked('verify exhausted', detail='All attempts failed'),
                timeout=2.0,
            )

        # Returned before the dry-run finished
        assert result is not None

        # Give the background task a tick to register its call
        await asyncio.sleep(0)

        # Capture any still-pending background tasks before releasing hang,
        # then await them so pytest-asyncio doesn't warn "Task destroyed while pending".
        pending = list(wf._background_tasks)
        hang_event.set()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        # run_dry_run_unblock was called with the right kwargs
        assert len(calls) == 1
        kwargs = calls[0]
        assert kwargs['task_id'] == '42'
        assert 'verify exhausted' in kwargs['reason']
        assert 'worktree' in kwargs
        assert 'scheduler' in kwargs
        assert 'config' in kwargs

        # FakeScheduler captured the 'blocked' status write
        assert 'blocked' in scheduler.statuses.get('42', [])


# ---------------------------------------------------------------------------
# step-19: merge_phase suppresses hook (and blocked write)
# ---------------------------------------------------------------------------

class TestMarkBlockedSkipsDryRunWhenMergePhase:
    @pytest.mark.asyncio
    async def test_mark_blocked_skips_dry_run_when_merge_phase(self, tmp_path):
        wf, scheduler = _make_workflow(tmp_path=tmp_path, task_id='43', enabled=True)

        dry_run_calls = []

        async def _spy_dry_run(**kwargs):
            dry_run_calls.append(kwargs)

        with patch('orchestrator.workflow.run_dry_run_unblock', new=_spy_dry_run):
            await wf._mark_blocked('merge conflict', merge_phase=True)

        await asyncio.sleep(0)  # let any tasks run

        # hook was NOT called
        assert len(dry_run_calls) == 0

        # blocked status was NOT written (merge_phase suppresses it)
        assert 'blocked' not in scheduler.statuses.get('43', [])


# ---------------------------------------------------------------------------
# step-21: enabled=False suppresses hook but NOT the blocked write
# ---------------------------------------------------------------------------

class TestMarkBlockedSkipsDryRunWhenFeatureDisabled:
    @pytest.mark.asyncio
    async def test_mark_blocked_skips_dry_run_when_feature_disabled(self, tmp_path):
        wf, scheduler = _make_workflow(tmp_path=tmp_path, task_id='44', enabled=False)

        dry_run_calls = []

        async def _spy_dry_run(**kwargs):
            dry_run_calls.append(kwargs)

        with patch('orchestrator.workflow.run_dry_run_unblock', new=_spy_dry_run):
            await wf._mark_blocked('verify exhausted')

        await asyncio.sleep(0)

        # hook was NOT called
        assert len(dry_run_calls) == 0

        # but the 'blocked' status write still happened
        assert 'blocked' in scheduler.statuses.get('44', [])


# ---------------------------------------------------------------------------
# Suggestion 7: worktree=None skips the hook
# ---------------------------------------------------------------------------

class TestMarkBlockedSkipsDryRunWhenNoWorktree:
    @pytest.mark.asyncio
    async def test_mark_blocked_skips_dry_run_when_no_worktree(self, tmp_path):
        wf, scheduler = _make_workflow(tmp_path=tmp_path, task_id='45', enabled=True)
        # Simulate a workflow where no worktree has been set yet
        wf.worktree = None

        dry_run_calls = []

        async def _spy_dry_run(**kwargs):
            dry_run_calls.append(kwargs)

        with patch('orchestrator.workflow.run_dry_run_unblock', new=_spy_dry_run):
            await wf._mark_blocked('verify exhausted')

        await asyncio.sleep(0)

        # hook was NOT called — no worktree means no meaningful investigation target
        assert len(dry_run_calls) == 0

        # but the 'blocked' status write still happened
        assert 'blocked' in scheduler.statuses.get('45', [])


# ---------------------------------------------------------------------------
# Suggestion 5: deduplication — second _mark_blocked does not spawn again
# ---------------------------------------------------------------------------

class TestMarkBlockedDeduplicatesDryRun:
    @pytest.mark.asyncio
    async def test_mark_blocked_skips_duplicate_dry_run_when_one_already_running(
        self, tmp_path
    ):
        wf, scheduler = _make_workflow(tmp_path=tmp_path, task_id='46', enabled=True)

        calls = []
        hang_event = asyncio.Event()

        async def _hanging_dry_run(**kwargs):
            calls.append(kwargs)
            await hang_event.wait()

        with patch('orchestrator.workflow.run_dry_run_unblock', new=_hanging_dry_run):
            # First call — spawns a background task that hangs
            await asyncio.wait_for(
                wf._mark_blocked('verify exhausted', detail='attempt 1'),
                timeout=2.0,
            )
            await asyncio.sleep(0)  # let the task register

            # Second call while the first investigation is still running
            await asyncio.wait_for(
                wf._mark_blocked('verify exhausted', detail='attempt 2'),
                timeout=2.0,
            )
            await asyncio.sleep(0)

        # Only one investigation spawned despite two _mark_blocked calls
        assert len(calls) == 1, (
            f'Expected 1 dry-run invocation, got {len(calls)}'
        )

        # Clean up: release the hanging task
        pending = list(wf._background_tasks)
        hang_event.set()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


# ---------------------------------------------------------------------------
# step-1: merge_phase=True WITH spawn_dry_run=True opts in to the hook
# ---------------------------------------------------------------------------

class TestMarkBlockedSpawnsDryRunWhenMergePhaseAndOptIn:
    @pytest.mark.asyncio
    async def test_mark_blocked_spawns_dry_run_when_merge_phase_and_opt_in(
        self, tmp_path
    ):
        """_mark_blocked(merge_phase=True, spawn_dry_run=True) must schedule
        run_dry_run_unblock even though merge_phase suppresses the status
        transition.

        This is the post-merge red-main class: main is already advanced when
        _mark_blocked runs, so the SHA capture inside run_dry_run_unblock
        naturally reflects post-merge reality.  The status write is still
        suppressed (merge_phase=True contract intact).
        """
        wf, scheduler = _make_workflow(tmp_path=tmp_path, task_id='47', enabled=True)

        calls = []
        hang_event = asyncio.Event()

        async def _hanging_dry_run(**kwargs):
            calls.append(kwargs)
            await hang_event.wait()

        with patch('orchestrator.workflow.run_dry_run_unblock', new=_hanging_dry_run):
            result = await asyncio.wait_for(
                wf._mark_blocked(
                    'post-merge red',
                    merge_phase=True,
                    spawn_dry_run=True,
                ),
                timeout=2.0,
            )

        assert result is not None

        # Give the background task a tick to register its call
        await asyncio.sleep(0)

        # Capture pending tasks and drain them cleanly
        pending = list(wf._background_tasks)
        hang_event.set()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        # run_dry_run_unblock was called once with the correct task worktree
        assert len(calls) == 1, f'Expected 1 dry-run invocation, got {len(calls)}'
        kwargs = calls[0]
        assert kwargs['task_id'] == '47'
        assert 'post-merge red' in kwargs['reason']
        assert 'worktree' in kwargs
        assert kwargs['worktree'] == str(wf.worktree)

        # merge_phase=True suppresses the 'blocked' status write — still
        assert 'blocked' not in scheduler.statuses.get('47', [])


# ---------------------------------------------------------------------------
# Protocol-conformance assertion (static-only; never executes at runtime).
# Mirrors the if TYPE_CHECKING / _SchedulerLike conformance block near the
# bottom of test_workflow_e2e.py.
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from orchestrator.workflow import _SchedulerLike

    _scheduler_conforms: _SchedulerLike = _Scheduler()
