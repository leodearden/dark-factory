"""Tests for the dry-run unblock hook wired into _mark_blocked.

Pins four contracts:
1. When unblock_auto.enabled=True, _mark_blocked schedules a background
   asyncio.create_task for run_dry_run_unblock and returns immediately
   (fire-and-forget — _mark_blocked does NOT await the dry-run).
2. merge_phase=True suppresses the hook entirely.
3. unblock_auto.enabled=False suppresses the hook but does NOT suppress the
   real 'blocked' status write.
4. spawn_dry_run=True with merge_phase=True is only valid for
   POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX (ordering invariant: advance_main
   has already moved main before this path runs).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.merge_queue import POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX
from orchestrator.workflow import TaskWorkflow


class _Scheduler:
    """Minimal fake scheduler satisfying SchedulerFacade for workflow tests."""

    def __init__(self) -> None:
        self.statuses: dict[str, list[str]] = {}
        self.update_calls: list[dict] = []

    async def set_task_status(self, tid, status, **_kw):
        self.statuses.setdefault(tid, []).append(status)

    async def set_task_claimant(self, tid, **_kw):
        pass

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

    async def handle_blast_radius_expansion(
        self, task_id, current, needed, /, *, persist_files=None
    ):
        return False

    async def dispatch_tool(self, name, arguments, *, timeout=30.0):
        return {}

    def release(self, tid):
        pass

    async def get_tasks(self, *, statuses: Iterable[str] | None = None) -> list[dict]:
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

        _reason = (
            f'{POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX}: '
            f'post-merge unscoped type-check failed for shared on deadbeef. detail'
        )
        with patch('orchestrator.workflow.run_dry_run_unblock', new=_hanging_dry_run):
            result = await asyncio.wait_for(
                wf._mark_blocked(
                    _reason,
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
        assert POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX in kwargs['reason']
        assert 'worktree' in kwargs
        assert kwargs['worktree'] == str(wf.worktree)

        # merge_phase=True suppresses the 'blocked' status write — still
        assert 'blocked' not in scheduler.statuses.get('47', [])


# ---------------------------------------------------------------------------
# Amendment 1: ordering invariant — spawn_dry_run=True only valid for
# POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX (the post-advance class).
# Pins the runtime assertion added to _mark_blocked's elif spawn_dry_run:
# block so a future caller on a pre-advance path fails loudly rather than
# silently producing a stale-SHA dry-run proposal.
# ---------------------------------------------------------------------------

class TestMarkBlockedSpawnDryRunOrderingInvariant:
    @pytest.mark.asyncio
    async def test_spawn_dry_run_wrong_prefix_raises_assertion_error(
        self, tmp_path
    ):
        """_mark_blocked(merge_phase=True, spawn_dry_run=True) must raise
        AssertionError when the reason does NOT start with
        POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX.

        This pins the ordering invariant: spawn_dry_run=True is only safe
        when advance_main has already moved refs/heads/main, which is
        guaranteed only for the POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX
        class (_check_post_merge_pyright is called exclusively from
        _finalize_advanced_merge, which runs after a successful
        advance_main).  Any other prefix likely indicates a pre-advance
        path where SHA capture would reflect stale state, causing
        b3_gate.check_proposal to act on incorrect data.

        If this test fails because you added a second post-advance class,
        extend the guard in _mark_blocked rather than removing it.
        """
        wf, _ = _make_workflow(tmp_path=tmp_path, task_id='48', enabled=True)

        with pytest.raises(AssertionError, match='POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX'):
            await wf._mark_blocked(
                'some-other-blocked-class: unexpected failure detail',
                merge_phase=True,
                spawn_dry_run=True,
            )

    @pytest.mark.asyncio
    async def test_spawn_dry_run_correct_prefix_does_not_raise(
        self, tmp_path
    ):
        """_mark_blocked(merge_phase=True, spawn_dry_run=True) must NOT
        raise when the reason starts with POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX.

        This is the counterpart to the wrong-prefix test: confirms the
        guard is a targeted prefix check, not an unconditional block.
        """
        wf, _ = _make_workflow(tmp_path=tmp_path, task_id='49', enabled=True)

        _reason = (
            f'{POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX}: '
            f'post-merge unscoped type-check failed for shared on cafecafe.'
        )
        dry_run_calls = []

        async def _spy(**kwargs):
            dry_run_calls.append(kwargs)

        with patch('orchestrator.workflow.run_dry_run_unblock', new=_spy):
            # Must not raise
            await wf._mark_blocked(_reason, merge_phase=True, spawn_dry_run=True)

        await asyncio.sleep(0)
        pending = list(wf._background_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        assert len(dry_run_calls) == 1, (
            f'Correct prefix should spawn once, got {len(dry_run_calls)}'
        )


# ---------------------------------------------------------------------------
# task 2021 step-9: usage_gate/cost_store forwarded into run_dry_run_unblock
# ---------------------------------------------------------------------------

class TestMarkBlockedForwardsResilienceContext:
    """_spawn_dry_run_unblock must forward self.usage_gate/self.cost_store
    into run_dry_run_unblock so the background investigation's
    invoke_with_cap_retry call can do account failover/cap detection like
    every other orchestrator agent invocation.
    """

    @pytest.mark.asyncio
    async def test_mark_blocked_forwards_usage_gate_and_cost_store(self, tmp_path):
        wf, _ = _make_workflow(tmp_path=tmp_path, task_id='50', enabled=True)

        usage_gate_sentinel = object()
        cost_store_sentinel = object()
        wf.usage_gate = usage_gate_sentinel  # type: ignore[assignment]
        wf.cost_store = cost_store_sentinel  # type: ignore[assignment]

        calls = []

        async def _spy_dry_run(**kwargs):
            calls.append(kwargs)

        with patch('orchestrator.workflow.run_dry_run_unblock', new=_spy_dry_run):
            await wf._mark_blocked('verify exhausted')

        await asyncio.sleep(0)  # let the background task register its call
        pending = list(wf._background_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        assert len(calls) == 1, f'Expected 1 dry-run invocation, got {len(calls)}'
        kwargs = calls[0]
        assert kwargs.get('usage_gate') is usage_gate_sentinel
        assert kwargs.get('cost_store') is cost_store_sentinel


# ---------------------------------------------------------------------------
# task 2557 step-9: last_blocked_at stamp on confirmed block transition
# ---------------------------------------------------------------------------

class TestMarkBlockedStampsLastBlockedAt:
    """_mark_blocked must stamp metadata.last_blocked_at on a confirmed,
    non-merge_phase block transition (task 2557).

    This is the staleness reference BriefingAssembler._format_prior_proposal
    compares a persisted dry_run_proposals entry against: a re-block without
    a fresh investigation must move last_blocked_at past the old proposal so
    stale analysis is omitted, while a fresh investigation's later-stamped
    proposal stays included.
    """

    @pytest.mark.asyncio
    async def test_mark_blocked_stamps_last_blocked_at(self, tmp_path):
        wf, scheduler = _make_workflow(tmp_path=tmp_path, task_id='46', enabled=True)

        async def _spy_dry_run(**kwargs):
            pass

        with patch('orchestrator.workflow.run_dry_run_unblock', new=_spy_dry_run):
            await wf._mark_blocked('verify exhausted', detail='All attempts failed')

        await asyncio.sleep(0)  # let any background tasks register
        pending = list(wf._background_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        # Confirmed block transition: the status write succeeded first.
        assert 'blocked' in scheduler.statuses.get('46', [])

        # last_blocked_at was stamped via scheduler.update_task, called with
        # only (task_id, metadata) — the fake scheduler's update_task has no
        # metadata_mode kwarg at all, so passing one explicitly would raise
        # TypeError here, pinning that the stamp relies on the real
        # scheduler's default ('merge') rather than overriding it.
        stamp_calls = [
            c for c in scheduler.update_calls
            if isinstance(c['metadata'], dict) and 'last_blocked_at' in c['metadata']
        ]
        assert len(stamp_calls) == 1, f'Expected exactly 1 last_blocked_at stamp, got {stamp_calls}'
        stamped_value = stamp_calls[0]['metadata']['last_blocked_at']
        assert isinstance(stamped_value, str)
        datetime.fromisoformat(stamped_value)  # must be ISO-8601 parseable


# ---------------------------------------------------------------------------
# Amendment (task 2557 review): produce-side ordering invariant.
# TestMarkBlockedStampsLastBlockedAt above only proves the stamp eventually
# happens — it would not catch a regression that reordered the stamp to
# after _spawn_dry_run_unblock. Because run_dry_run_unblock is fire-and-
# forget (asyncio.create_task), the background task cannot run any of its
# own code — including its own proposal-append update_task call — until the
# CURRENT coroutine genuinely yields (e.g. the test's `asyncio.sleep(0)`).
# With the fake, non-yielding scheduler used here, that means a post-hoc
# check of relative order in scheduler.update_calls would show "stamp
# first" regardless of which statement _mark_blocked calls first — it
# can't distinguish correct from reordered source. So this test instead
# records the order in which the two *call sites* inside _mark_blocked
# actually fire: the stamp's scheduler.update_task(...) await vs. the
# self._spawn_dry_run_unblock(...) invocation. That directly pins the
# source-level sequencing described at workflow.py's `_mark_blocked`
# comment (search "Awaited synchronously here, BEFORE the fire-and-forget"),
# which is what guarantees a fresh investigation's proposal timestamp lands
# after last_blocked_at under a real (slow, potentially-concurrent)
# scheduler in production.
# ---------------------------------------------------------------------------

class TestMarkBlockedStampPrecedesDryRunSpawn:
    @pytest.mark.asyncio
    async def test_last_blocked_at_stamp_precedes_dry_run_spawn(self, tmp_path):
        wf, scheduler = _make_workflow(tmp_path=tmp_path, task_id='52', enabled=True)

        call_order: list[str] = []

        real_update_task = scheduler.update_task

        async def _tracking_update_task(task_id, metadata, *, append=False):
            if isinstance(metadata, dict) and 'last_blocked_at' in metadata:
                call_order.append('stamp')
            return await real_update_task(task_id, metadata, append=append)

        scheduler.update_task = _tracking_update_task  # type: ignore[method-assign]

        real_spawn = wf._spawn_dry_run_unblock

        def _tracking_spawn(*args, **kwargs):
            call_order.append('spawn')
            return real_spawn(*args, **kwargs)

        wf._spawn_dry_run_unblock = _tracking_spawn  # type: ignore[method-assign]

        async def _spy_dry_run(**kwargs):
            pass

        with patch('orchestrator.workflow.run_dry_run_unblock', new=_spy_dry_run):
            await wf._mark_blocked('verify exhausted', detail='All attempts failed')

        await asyncio.sleep(0)  # let any background tasks register/finish
        pending = list(wf._background_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        assert call_order == ['stamp', 'spawn'], (
            'Expected the last_blocked_at stamp to be awaited BEFORE '
            '_spawn_dry_run_unblock is invoked, so a freshly-produced '
            f'proposal always compares as non-stale; got {call_order}'
        )


# ---------------------------------------------------------------------------
# Protocol-conformance assertion (static-only; never executes at runtime).
# Mirrors the if TYPE_CHECKING / SchedulerFacade conformance block near the
# bottom of test_workflow_e2e.py.
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from orchestrator.scheduler import SchedulerFacade

    _scheduler_conforms: SchedulerFacade = _Scheduler()
