"""Continuity credit γ through the scheduler (task 3820; plans/scheduler-dispatch-scoring-and-lock-layer-prd.md C3, INV-4)."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig, ScoringConfig
from orchestrator.event_store import EventStore
from orchestrator.scheduler import Scheduler, TaskAssignment
from orchestrator.work_product import (
    FAILURE_STREAK_ESCALATION_THRESHOLD,
    PROBE_SENTINEL,
    BranchProbeError,
    BranchWork,
)

NOW = datetime(2026, 10, 12, 12, 0, 0, tzinfo=UTC)
SHARED_ANCHOR = '2026-10-10T08:00:00.000Z'
TTL = ScoringConfig().continuity_probe_ttl_secs


def _task(task_id: str, *, priority: str = 'medium', pending_since: str | None = SHARED_ANCHOR) -> dict:
    metadata: dict = {'files': [f'mod{task_id}']}
    if pending_since is not None:
        metadata['pending_since'] = pending_since
    return {'id': task_id, 'title': f'Task {task_id}', 'status': 'pending', 'priority': priority,
            'dependencies': [], 'metadata': metadata}


def _tasks(count: int, first_id: int = 1) -> list[dict]:
    return [_task(str(first_id + i)) for i in range(count)]


class _RecordingReader:
    """The one-git-child seam: counts reads and records the thread each ran on."""

    def __init__(self, branches: Mapping[str, BranchWork] | None = None) -> None:
        self.branches = dict(branches or {})
        self.threads: list[int] = []

    @property
    def calls(self) -> int:
        return len(self.threads)

    def __call__(self) -> dict[str, BranchWork]:
        self.threads.append(threading.get_ident())
        return dict(self.branches)


def _scheduler(reader, *, event_store: EventStore | None, clock: list[float]) -> Scheduler:
    scheduler = Scheduler(
        OrchestratorConfig(max_per_module=1, max_concurrent_tasks=500, scoring=ScoringConfig()),
        event_store=event_store,
        time_source=lambda: clock[0],
        wall_time_source=lambda: NOW,
        branch_work_reader=reader,
    )
    scheduler.finish_startup()
    return scheduler


async def _tick(scheduler: Scheduler, tasks: list[dict]) -> TaskAssignment | None:
    """One dispatch tick; what it dispatched is released, so the next tick sees the same candidates."""
    scheduler.get_tasks = AsyncMock(return_value=tasks)
    assignment = await scheduler.acquire_next()
    if assignment is not None:
        scheduler.release(assignment.task_id)
    return assignment


async def _tick_and_settle(scheduler: Scheduler, tasks: list[dict]) -> TaskAssignment | None:
    assignment = await _tick(scheduler, tasks)
    await scheduler.work_product.join()
    return assignment


@pytest.fixture
def store(tmp_path) -> EventStore:
    return EventStore(tmp_path / 'runs.db', run_id='this-run')


@pytest.fixture
def clock() -> list[float]:
    return [0.0]


@pytest.mark.asyncio
class TestBoundedProbeOccupancy:
    """C3: at most one git child per TTL window, forked off the loop thread, never awaited by a tick."""

    @pytest.mark.parametrize('candidates', [1, 300])
    async def test_one_read_per_ttl_window_whatever_the_candidate_count(self, store, clock, candidates):
        reader = _RecordingReader()
        scheduler = _scheduler(reader, event_store=store, clock=clock)
        tasks = _tasks(candidates)

        for _ in range(20):
            assert await _tick_and_settle(scheduler, tasks) is not None
        assert reader.calls == 1

        clock[0] += TTL - 1
        await _tick_and_settle(scheduler, tasks)
        assert reader.calls == 1

        clock[0] += 1
        await _tick_and_settle(scheduler, tasks)
        assert reader.calls == 2

    async def test_the_read_runs_off_the_loop_thread(self, store, clock):
        reader = _RecordingReader()
        scheduler = _scheduler(reader, event_store=store, clock=clock)

        await _tick_and_settle(scheduler, _tasks(3))

        assert reader.threads and reader.threads[0] != threading.get_ident()

    async def test_the_tick_never_waits_on_git(self, store, clock):
        release = threading.Event()
        blocked: list[int] = []

        def blocked_reader() -> dict[str, BranchWork]:
            blocked.append(threading.get_ident())
            release.wait(timeout=30)
            return {}

        scheduler = _scheduler(blocked_reader, event_store=store, clock=clock)
        scheduler.get_tasks = AsyncMock(return_value=_tasks(3))
        try:
            assignment = await asyncio.wait_for(scheduler.acquire_next(), timeout=5)
            assert isinstance(assignment, TaskAssignment)
            assert not release.is_set()
        finally:
            release.set()
        await scheduler.work_product.join()
        assert len(blocked) == 1

    async def test_nothing_is_probed_without_an_event_store(self, clock):
        reader = _RecordingReader()
        scheduler = _scheduler(reader, event_store=None, clock=clock)

        for _ in range(5):
            clock[0] += TTL
            await _tick_and_settle(scheduler, _tasks(3))

        assert reader.calls == 0

    async def test_a_hot_ttl_retune_shortens_the_window_on_the_next_tick(self, store, clock):
        reader = _RecordingReader()
        scheduler = _scheduler(reader, event_store=store, clock=clock)
        tasks = _tasks(3)
        await _tick_and_settle(scheduler, tasks)

        scheduler.config.scoring.continuity_probe_ttl_secs = 60.0
        clock[0] = 60.0
        await _tick_and_settle(scheduler, tasks)

        assert reader.calls == 2


def _probe_escalations(queue: EscalationQueue) -> list[Escalation]:
    return [e for e in queue.get_pending() if e.task_id == PROBE_SENTINEL]


@pytest.mark.asyncio
class TestProbeStormEscape:
    """INV-4: a persistently failing probe reaches a human once, as one deduped L1."""

    async def test_a_failure_streak_files_one_deduped_l1(self, tmp_path, store, clock):
        queue = EscalationQueue(tmp_path / 'esc')

        def failing_reader() -> dict[str, BranchWork]:
            raise BranchProbeError(
                reason='git for-each-ref exited 128', returncode=128,
                stderr="fatal: failed to find 'main'",
            )

        scheduler = _scheduler(failing_reader, event_store=store, clock=clock)
        scheduler.escalation_queue = queue
        tasks = _tasks(3)

        async def failed_refresh() -> None:
            await _tick_and_settle(scheduler, tasks)
            clock[0] += TTL

        for _ in range(FAILURE_STREAK_ESCALATION_THRESHOLD - 1):
            await failed_refresh()
        assert _probe_escalations(queue) == []

        await failed_refresh()
        [esc] = _probe_escalations(queue)
        assert (esc.level, esc.category, esc.agent_role) == (1, 'infra_issue', 'orchestrator-scheduler')
        for fact in (f'{FAILURE_STREAK_ESCALATION_THRESHOLD} consecutive', '128', 'failed to find'):
            assert fact in esc.detail, fact

        await failed_refresh()
        await failed_refresh()
        assert len(_probe_escalations(queue)) == 1
