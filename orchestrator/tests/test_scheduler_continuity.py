"""Continuity credit γ through the scheduler (task 3820; plans/scheduler-dispatch-scoring-and-lock-layer-prd.md C3, INV-4)."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _git_fixtures import README_SEED, _git, build_repo
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig, ScoringConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.scheduler import Scheduler, TaskAssignment, TickContext
from orchestrator.work_product import (
    FAILURE_STREAK_ESCALATION_THRESHOLD,
    PROBE_SENTINEL,
    BranchProbeError,
    BranchWork,
)

NOW = datetime(2026, 10, 12, 12, 0, 0, tzinfo=UTC)
SHARED_ANCHOR = '2026-10-10T08:00:00.000Z'
TTL = ScoringConfig().continuity_probe_ttl_secs
CREDIT = ScoringConfig().continuity_credit
MAX_AGE = ScoringConfig().continuity_credit_max_age_secs
NOW_TS = NOW.timestamp()
HOUR = 3600.0


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


def _ctx(tasks: list[dict]) -> TickContext:
    return TickContext(
        tasks=tasks,
        status_map={t['id']: 'pending' for t in tasks},
        tasks_by_id={t['id']: t for t in tasks},
        max_id=max(int(t['id']) for t in tasks),
        transitive_counts={t['id']: 0 for t in tasks},
        candidates=tasks,
    )


def _scores(scheduler: Scheduler, tasks: list[dict]) -> dict[str, float]:
    return {tid: score for score, tid, _task, _tier in scheduler._score_candidates(_ctx(tasks))}


async def _credit_delta(scheduler: Scheduler, with_work: str, without: str) -> float:
    """Settle one refresh, then score two otherwise identical candidates."""
    tasks = [_task(with_work), _task(without)]
    await _tick_and_settle(scheduler, tasks)
    scores = _scores(scheduler, tasks)
    return scores[with_work] - scores[without]


def _fresh(commits_ahead: int = 2, *, age_secs: float = HOUR) -> BranchWork:
    return BranchWork(commits_ahead=commits_ahead, tip_committed_at=NOW_TS - age_secs)


@pytest.mark.asyncio
class TestContinuityCredit:
    """C3: a task with fresh, ungated work product scores exactly continuity_credit higher."""

    async def test_work_product_adds_exactly_the_continuity_credit(self, store, clock):
        scheduler = _scheduler(_RecordingReader({'10': _fresh()}), event_store=store, clock=clock)

        assert await _credit_delta(scheduler, '10', '11') == pytest.approx(CREDIT, abs=1e-9)

    async def test_a_zero_progress_row_from_a_prior_run_withholds_the_credit(self, tmp_path, clock):
        db = tmp_path / 'runs.db'
        EventStore(db, run_id='earlier-run').emit(
            EventType.zero_progress_requeue, task_id='10', data={'consecutive': 3},
        )
        scheduler = _scheduler(
            _RecordingReader({'10': _fresh()}), event_store=EventStore(db, run_id='this-run'), clock=clock,
        )

        assert await _credit_delta(scheduler, '10', '11') == 0

    async def test_another_tasks_zero_progress_row_does_not_withhold_the_credit(self, store, clock):
        store.emit(EventType.zero_progress_requeue, task_id='99', data={'consecutive': 3})
        scheduler = _scheduler(_RecordingReader({'10': _fresh()}), event_store=store, clock=clock)

        assert await _credit_delta(scheduler, '10', '11') == pytest.approx(CREDIT, abs=1e-9)

    @pytest.mark.parametrize(('age_secs', 'credited'), [(MAX_AGE, True), (MAX_AGE + 1, False)])
    async def test_the_credit_lapses_after_the_max_age(self, store, clock, age_secs, credited):
        scheduler = _scheduler(_RecordingReader({'10': _fresh(age_secs=age_secs)}), event_store=store, clock=clock)

        delta = await _credit_delta(scheduler, '10', '11')

        assert delta == pytest.approx(CREDIT if credited else 0.0, abs=1e-9)

    async def test_a_branch_with_nothing_ahead_earns_nothing(self, store, clock):
        scheduler = _scheduler(_RecordingReader({'10': _fresh(commits_ahead=0)}), event_store=store, clock=clock)

        assert await _credit_delta(scheduler, '10', '11') == 0

    async def test_work_product_outranks_the_numeric_tie_break(self, store, clock):
        scheduler = _scheduler(_RecordingReader({'11': _fresh(1)}), event_store=store, clock=clock)
        tasks = [_task('10'), _task('11')]

        before = await _tick_and_settle(scheduler, tasks)
        after = await _tick(scheduler, tasks)

        assert before is not None and before.task_id == '10'
        assert after is not None and after.task_id == '11'

    async def test_an_unarmed_tick_scores_without_continuity(self, store, clock):
        scheduler = _scheduler(_RecordingReader({'10': _fresh()}), event_store=store, clock=clock)
        tasks = [_task('10', pending_since=None), _task('11', pending_since=None)]
        await _tick_and_settle(scheduler, tasks)
        ctx = _ctx(tasks)

        scores = {tid: score for score, tid, _task, _tier in scheduler._score_candidates(ctx)}

        assert scores == {
            tid: scheduler._compute_score_legacy('medium', scheduler._compute_age(tid, ctx.max_id), 0)
            for tid in ('10', '11')
        }

    async def test_no_event_store_means_no_credit(self, clock):
        reader = _RecordingReader({'10': _fresh()})
        scheduler = _scheduler(reader, event_store=None, clock=clock)

        assert await _credit_delta(scheduler, '10', '11') == 0
        assert reader.calls == 0


@pytest.mark.asyncio
async def test_the_default_reader_credits_a_real_task_branch(tmp_path, monkeypatch, store, clock):
    monkeypatch.setenv('GIT_COMMITTER_DATE', f'{int(NOW_TS - 2 * HOUR)} +0000')
    repo = build_repo(tmp_path / 'project', README_SEED)
    _git(repo, 'branch', 'task/8', 'main')
    monkeypatch.setenv('GIT_COMMITTER_DATE', f'{int(NOW_TS - HOUR)} +0000')
    _git(repo, 'checkout', '-q', '-b', 'task/7', 'main')
    _git(repo, 'commit', '-q', '--allow-empty', '-m', 'task 7 work')
    _git(repo, 'checkout', '-q', 'main')
    scheduler = Scheduler(
        OrchestratorConfig(project_root=Path(repo), max_per_module=1, scoring=ScoringConfig()),
        event_store=store,
        time_source=lambda: clock[0],
        wall_time_source=lambda: NOW,
    )
    scheduler.finish_startup()

    assert await _credit_delta(scheduler, '7', '8') == pytest.approx(CREDIT, abs=1e-9)
