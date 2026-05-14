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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.workflow import TaskWorkflow


def _make_workflow(*, tmp_path: Path, task_id: str = '42',
                   enabled: bool = True) -> TaskWorkflow:
    """Minimal TaskWorkflow with controllable unblock_auto flag."""
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = ['mod_a']

    config = MagicMock()
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = tmp_path / 'proj'
    config.unblock_auto.enabled = enabled

    git_ops = MagicMock()

    # FakeScheduler inline — track set_task_status calls
    class _Scheduler:
        def __init__(self):
            self.statuses: dict[str, list[str]] = {}
            self.update_calls: list[dict] = []

        async def set_task_status(self, tid, status, **_kw):
            self.statuses.setdefault(tid, []).append(status)

        async def get_status(self, tid):
            hist = self.statuses.get(tid, [])
            return hist[-1] if hist else None

        async def update_task(self, tid, metadata, *, append=False):
            self.update_calls.append({'task_id': tid, 'metadata': metadata, 'append': append})
            return True

        async def get_task(self, tid):
            return None

        def release(self, tid):
            pass

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
        hang_event.set()
        await asyncio.sleep(0)

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
