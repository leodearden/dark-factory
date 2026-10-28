"""The starvation watchdog through the real Scheduler (tasks 1880, 3821).

Observed only at behaviour seams: ``starvation_detected`` events in the event
store, the warn/resolve callbacks, and the override store.  Both clocks are
injected: the WALL clock drives the durable eligibility age, the monotonic
clock drives cooldowns.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.psi import PsiSample

from orchestrator.config import OrchestratorConfig, PsiAdmissionConfig, StarvationWatchdogConfig
from orchestrator.eligibility_clock import EligibilityClockStore
from orchestrator.event_store import EventStore
from orchestrator.overrides import OverrideSource, OverrideStore
from orchestrator.scheduler import Scheduler, SchedulerCallbacks, TaskAssignment
from orchestrator.starvation_watchdog import AUTO_PIN_CAP_SENTINEL, STARVATION_WATCHDOG_ROLE

T0 = datetime(2026, 10, 1, tzinfo=UTC)
FLOOR = 259200.0
_IDLE_PSI = PsiSample(cpu_some10=0.0, mem_some10=0.0, mem_full10=0.0, io_some10=0.0, read_ok=True)
_SATURATED_PSI = PsiSample(cpu_some10=99.0, mem_some10=0.0, mem_full10=0.0, io_some10=0.0, read_ok=True)


def _task(tid: str, priority: str, *files: str, **metadata: Any) -> dict:
    return {
        'id': tid, 'title': f'task {tid}', 'status': 'pending', 'priority': priority,
        'dependencies': [], 'metadata': {'files': list(files), **metadata},
    }


def _seed() -> dict:
    """Dispatches first and then holds the hot modules for the rest of the scene."""
    return _task('0', 'critical', 'hot/a.py', 'hot/b.py')


@dataclass
class _Rig:
    scheduler: Scheduler
    tasks: list[dict]
    wall: list[datetime]
    mono: list[float]
    warn: AsyncMock
    resolve: AsyncMock
    events_db: Path
    overrides: OverrideStore
    live_status: dict[str, str]
    status_faults: list[Exception]
    _low_seq: list[int] = field(default_factory=lambda: [1000])

    def advance(self, secs: float) -> None:
        self.wall[0] += timedelta(seconds=secs)
        self.mono[0] += secs

    def set_tasks(self, tasks: list[dict]) -> None:
        self.tasks[:] = tasks

    def fresh_low(self, **metadata: Any) -> dict:
        self._low_seq[0] += 1
        n = self._low_seq[0]
        return _task(str(n), 'low', f'free/{n}.py', **metadata)

    async def tick(self) -> TaskAssignment | None:
        assignment = await self.scheduler.acquire_next()
        if assignment is not None:
            for task in self.tasks:
                if str(task['id']) == assignment.task_id:
                    task['status'] = 'in-progress'
        return assignment

    async def tick_id(self) -> str | None:
        assignment = await self.tick()
        return None if assignment is None else assignment.task_id

    async def tick_with_low_winner(self) -> None:
        """One scan in which a fresh low-priority task dispatches below everything else."""
        low = self.fresh_low()
        self.tasks.append(low)
        assert await self.tick_id() == low['id']

    def detected(self, task_id: str | None = None) -> list[dict]:
        conn = sqlite3.connect(str(self.events_db))
        try:
            rows = conn.execute(
                "SELECT task_id, data FROM events WHERE event_type='starvation_detected' "
                'ORDER BY id'
            ).fetchall()
        finally:
            conn.close()
        payloads = [{'task_id': tid, **json.loads(data)} for tid, data in rows]
        if task_id is None:
            return payloads
        return [p for p in payloads if p['task_id'] == task_id]

    def warned_ids(self) -> list[str]:
        return [str(c.args[0]) for c in self.warn.call_args_list]

    @property
    def root(self) -> str:
        return str(self.scheduler.config.project_root)

    def override_rows(self) -> dict:
        return self.overrides.get_overrides(self.root)

    def watchdog_pins(self) -> list[str]:
        return [
            tid for tid, row in self.overrides.get_pin_queue(self.root)
            if row.source is OverrideSource.STARVATION_WATCHDOG
        ]

    def set_status(self, task_id: str, status: str) -> None:
        for task in self.tasks:
            if str(task['id']) == task_id:
                task['status'] = status


def _rig(
    tmp_path: Path,
    *,
    skip_threshold: int = 3,
    idle_secs: float = FLOOR,
    idle_only_secs: float = FLOOR,
    enabled: bool = True,
    psi: PsiAdmissionConfig | None = None,
    tasks: list[dict] | None = None,
    wall: list[datetime] | None = None,
    **watchdog: Any,
) -> _Rig:
    config = OrchestratorConfig(
        project_root=tmp_path,
        max_per_module=1,
        starvation_watchdog=StarvationWatchdogConfig(
            enabled=enabled, skip_threshold=skip_threshold,
            idle_secs=idle_secs, idle_only_secs=idle_only_secs, **watchdog,
        ),
        psi_admission=psi if psi is not None else PsiAdmissionConfig(),
    )
    # Fairness parks would turn a top-skipped task into a parked one; these
    # scenes exercise the count gates, so parks stay out of the way.
    config.fairness.skip_threshold = 9999
    wall = wall if wall is not None else [T0]
    mono = [0.0]
    warn, resolve = AsyncMock(), AsyncMock()
    task_list: list[dict] = tasks if tasks is not None else []
    events_db = tmp_path / 'runs.db'
    overrides = OverrideStore(tmp_path / 'scheduler_overrides.db')
    scheduler = Scheduler(
        config,
        event_store=EventStore(events_db, run_id='run-starvation'),
        callbacks=SchedulerCallbacks(on_starvation_warn=warn, on_starvation_resolve=resolve),
        time_source=lambda: mono[0],
        wall_time_source=lambda: wall[0],
        override_store=overrides,
        eligibility_clock_store=EligibilityClockStore(tmp_path / 'scheduler_eligibility.db'),
    )
    scheduler._read_psi_sample = lambda: _IDLE_PSI

    async def get_tasks(*, statuses: Any = None, **_: Any) -> list[dict]:
        wanted = None if statuses is None else set(statuses)
        return [dict(t) for t in task_list if wanted is None or t['status'] in wanted]

    live_status: dict[str, str] = {}
    status_faults: list[Exception] = []

    async def get_statuses(
        ids: list[str] | None = None,
    ) -> tuple[dict[str, str], Exception | None]:
        """The task list's statuses, with ``live_status`` overlaid; a queued fault fails one read."""
        if status_faults:
            return {}, status_faults.pop(0)
        statuses = {
            str(t['id']): t['status'] for t in task_list if ids is None or str(t['id']) in ids
        }
        statuses.update({tid: st for tid, st in live_status.items() if tid in statuses})
        return statuses, None

    scheduler.get_tasks = AsyncMock(side_effect=get_tasks)
    scheduler.get_statuses = AsyncMock(side_effect=get_statuses)
    scheduler.finish_startup()
    return _Rig(
        scheduler, task_list, wall, mono, warn, resolve, events_db, overrides,
        live_status, status_faults,
    )


async def _contended(rig: _Rig, *extra: dict) -> None:
    """Seed dispatches and holds hot/; TOP (high) and S (medium) both want hot/."""
    rig.set_tasks([
        _seed(), _task('1', 'high', 'hot/a.py'), _task('2', 'medium', 'hot/b.py'), *extra,
    ])
    assert await rig.tick_id() == '0'


# ---------------------------------------------------------------------------
# The retuned predicate (task 3821)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deep_backlog_never_fires_on_age_alone(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    backlog = [_task(str(100 + i), 'low', f'free/l{i}.py') for i in range(300)]
    backlog += [_task(str(1000 + i), 'medium', f'free/m{i}.py') for i in range(200)]
    rig.set_tasks(backlog)

    await rig.tick()
    rig.advance(10 * 86400)
    for _ in range(5):
        assert await rig.tick() is not None

    assert rig.detected() == []
    rig.warn.assert_not_called()


@pytest.mark.asyncio
async def test_lock_refused_fires_for_a_task_passed_by_lower_ranked_dispatches(
    tmp_path: Path,
) -> None:
    rig = _rig(tmp_path)
    await _contended(rig)
    rig.advance(FLOOR)

    for _ in range(3):
        await rig.tick_with_low_winner()
    assert rig.detected('2') == []
    await rig.tick_with_low_winner()

    [event] = rig.detected('2')
    assert event['gate'] == 'lock_refused'
    assert event['lock_refusals'] >= 3
    assert event['eligible_age_secs'] >= FLOOR
    assert {'module': 'hot/b.py', 'owner': '0', 'kind': 'held'} in event['blockers']
    assert [e['task_id'] for e in rig.detected() if e['gate'] == 'lock_refused'] == ['2']
    assert rig.warned_ids().count('2') == 1


@pytest.mark.asyncio
async def test_no_winner_and_psi_deferred_scans_count_no_refusals(tmp_path: Path) -> None:
    rig = _rig(tmp_path, psi=PsiAdmissionConfig(cpu_some_avg10=85.0, min_inflight_floor=1))
    await _contended(rig)
    rig.advance(FLOOR)

    for _ in range(2):
        await rig.tick_with_low_winner()
    for _ in range(3):
        assert await rig.tick() is None
    rig.scheduler._read_psi_sample = lambda: _SATURATED_PSI
    for _ in range(3):
        gate_task = rig.fresh_low(task_kind='deterministic')
        rig.tasks.append(gate_task)
        assert await rig.tick_id() == gate_task['id']
    rig.scheduler._read_psi_sample = lambda: _IDLE_PSI
    await rig.tick()
    assert rig.detected('2') == []

    await rig.tick_with_low_winner()
    await rig.tick()

    [event] = rig.detected('2')
    assert event['gate'] == 'lock_refused'
    assert event['lock_refusals'] == 3


@pytest.mark.asyncio
async def test_restart_keeps_the_eligibility_age_and_forms_no_cohort(tmp_path: Path) -> None:
    wall = [T0]
    tasks: list[dict] = []
    first = _rig(tmp_path, tasks=tasks, wall=wall)
    await _contended(first, _task('9000', 'low', 'free/backlog.py'))

    wall[0] = T0 + timedelta(seconds=FLOOR)
    for task in tasks:
        task['status'] = 'pending'
    second = _rig(tmp_path, tasks=tasks, wall=wall)
    assert await second.tick_id() == '0'
    for _ in range(4):
        await second.tick_with_low_winner()

    [event] = second.detected('2')
    assert event['gate'] == 'lock_refused'
    assert event['eligible_age_secs'] >= FLOOR
    assert second.detected('9000') == []


@pytest.mark.asyncio
async def test_a_candidacy_gap_does_not_reset_the_eligibility_age(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    rig.set_tasks([_seed(), _task('2', 'medium', 'hot/b.py')])
    assert await rig.tick_id() == '0'
    for _ in range(3):
        assert await rig.tick() is None

    rig.scheduler.release('2', requeued=True)
    rig.wall[0] = T0 + timedelta(seconds=FLOOR / 2)
    await rig.tick()
    rig.mono[0] += rig.scheduler.config.requeue_cooldown_secs + 1
    rig.wall[0] = T0 + timedelta(seconds=FLOOR)
    await rig.tick()

    [event] = rig.detected('2')
    assert event['gate'] == 'dual'
    assert event['eligible_age_secs'] == FLOOR


PARK_FLOOR = 14400.0


async def _parked_behind_holder(rig: _Rig) -> None:
    """H dispatches holding park/m2.py; P then parks over park/m1.py and park/m2.py."""
    rig.set_tasks([_task('H', 'critical', 'park/m2.py')])
    assert await rig.tick_id() == 'H'
    rig.tasks.append(_task('P', 'medium', 'park/m1.py', 'park/m2.py'))


@pytest.mark.asyncio
async def test_park_blocked_fires_naming_the_live_holder(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    await _parked_behind_holder(rig)
    rig.scheduler.lock_table.install_parks('P', ['park/m1.py', 'park/m2.py'], 'medium')
    rig.advance(PARK_FLOOR)

    for _ in range(3):
        assert await rig.tick() is None

    [event] = rig.detected('P')
    assert event['gate'] == 'park_blocked'
    assert event['blocking_holder'] == 'H'
    assert event['blocking_module'] == 'park/m2.py'
    assert event['park_age_secs'] >= PARK_FLOOR
    assert event['auto_pinned'] is False
    assert event['disposition'] == 'escalated_l1'
    assert 'P' not in rig.override_rows()
    rig.warn.assert_called_once()
    assert rig.warn.call_args.args[0] == 'P'
    assert rig.warn.call_args.kwargs['level'] == 1


@pytest.mark.asyncio
async def test_a_young_park_suppresses_every_gate_even_when_dual_is_satisfied(
    tmp_path: Path,
) -> None:
    rig = _rig(tmp_path)
    await _parked_behind_holder(rig)
    for _ in range(4):
        assert await rig.tick() is None
    rig.advance(FLOOR)
    rig.scheduler.lock_table.install_parks('P', ['park/m1.py', 'park/m2.py'], 'medium')
    rig.advance(PARK_FLOOR - 60)

    await rig.tick()

    assert rig.detected() == []
    rig.warn.assert_not_called()


# ---------------------------------------------------------------------------
# Bounded, owned auto-pin (task 3821)
# ---------------------------------------------------------------------------


async def _refused_while_lows_dispatch(rig: _Rig, *extra: dict) -> None:
    """S ('2') is refused hot/b.py in three scans that a lower-ranked low task wins.

    A young high task ('1') stays the top candidate, so S accrues lock
    refusals but no top-skips.  The next tick's watchdog pass fires S on
    gate 'lock_refused'.
    """
    rig.set_tasks([_seed(), _task('2', 'medium', 'hot/b.py'), *extra])
    assert await rig.tick_id() == '0'
    rig.advance(FLOOR)
    rig.tasks.append(_task('1', 'high', 'hot/a.py'))
    for _ in range(3):
        await rig.tick_with_low_winner()
    assert rig.detected() == []


@pytest.mark.asyncio
async def test_a_fire_pins_the_task_after_the_operator_pins(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    for tid in ('B1', 'B2'):
        rig.overrides.set_override(rig.root, tid, pinned=True)
    blocked = [_task(tid, 'medium', f'free/{tid}.py') for tid in ('B1', 'B2')]
    for task in blocked:
        task['status'] = 'blocked'
    await _refused_while_lows_dispatch(rig, *blocked)

    await rig.tick_with_low_winner()

    queue = rig.overrides.get_pin_queue(rig.root)
    assert [(tid, row.pin_order, row.source) for tid, row in queue] == [
        ('B1', 1, OverrideSource.OPERATOR),
        ('B2', 2, OverrideSource.OPERATOR),
        ('2', 3, OverrideSource.STARVATION_WATCHDOG),
    ]
    ttl = rig.scheduler.config.starvation_watchdog.auto_pin_ttl_secs
    assert queue[2][1].ttl_until == rig.wall[0] + timedelta(seconds=ttl)
    [event] = rig.detected('2')
    assert event['gate'] == 'lock_refused'
    assert event['auto_pinned'] is True
    assert event['pin_order'] == 3
    assert event['source'] == 'starvation_watchdog'
    assert event['disposition'] == 'auto_pinned'
    assert rig.warned_ids() == ['2']
    assert rig.warn.call_args.kwargs['level'] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'override', [{'pinned': True}, {'boost_tier': 'high'}], ids=['pinned', 'boost-only'],
)
async def test_an_operator_row_is_never_touched(tmp_path: Path, override: dict) -> None:
    rig = _rig(tmp_path)
    await _refused_while_lows_dispatch(rig)
    rig.overrides.set_override(rig.root, '2', **override)
    before = rig.override_rows()['2']

    await rig.tick_with_low_winner()

    assert rig.override_rows()['2'] == before
    [event] = rig.detected('2')
    assert event['auto_pinned'] is False
    assert event['disposition'] == 'operator_override'
    assert event['source'] == 'operator'
    assert rig.warned_ids() == ['2']


@pytest.mark.asyncio
async def test_the_cap_bounds_watchdog_pins_and_files_one_cohort_l1(tmp_path: Path) -> None:
    rig = _rig(tmp_path, auto_pin_max_concurrent=2, auto_pin_ttl_secs=10 * FLOOR)
    queue = EscalationQueue(tmp_path / 'esc')
    rig.scheduler.escalation_queue = queue
    seed = _task('0', 'critical', 'hot/a.py', 'hot/b.py', 'hot/c.py', 'hot/d.py')
    rig.set_tasks([seed, _task('S1', 'medium', 'hot/b.py'), _task('S2', 'medium', 'hot/c.py')])
    assert await rig.tick_id() == '0'
    rig.advance(60)
    rig.tasks.append(_task('S3', 'medium', 'hot/d.py'))
    assert await rig.tick() is None
    rig.advance(FLOOR)
    rig.tasks.append(_task('1', 'high', 'hot/a.py'))
    rig.tasks.append(_task('S4', 'medium', 'hot/d.py'))
    for _ in range(4):
        await rig.tick_with_low_winner()

    assert sorted(rig.watchdog_pins()) == ['S1', 'S2']
    [capped] = rig.detected('S3')
    assert capped['disposition'] == 'cap_reached'
    assert capped['auto_pinned'] is False
    assert sorted(rig.warned_ids()) == ['S1', 'S2']
    [cap_l1] = queue.get_by_task(AUTO_PIN_CAP_SENTINEL, status='pending', level=1)
    assert 'task S3' in cap_l1.detail

    rig.advance(FLOOR)
    await rig.tick_with_low_winner()

    [later] = rig.detected('S4')
    assert later['disposition'] == 'cap_reached'
    assert len(queue.get_by_task(AUTO_PIN_CAP_SENTINEL, status='pending', level=1)) == 1
    assert sorted(rig.watchdog_pins()) == ['S1', 'S2']


@pytest.mark.asyncio
async def test_dispatch_releases_the_watchdog_pin_but_never_an_operator_pin(
    tmp_path: Path,
) -> None:
    rig = _rig(tmp_path)
    await _refused_while_lows_dispatch(rig)
    await rig.tick_with_low_winner()
    assert rig.watchdog_pins() == ['2']

    rig.scheduler.release('0')
    assert await rig.tick_id() == '2'
    assert '2' not in rig.override_rows()

    rig.tasks.append(_task('OP', 'low', 'free/op.py'))
    rig.overrides.set_override(rig.root, 'OP', pinned=True)
    assert await rig.tick_id() == 'OP'
    assert rig.override_rows()['OP'].source is OverrideSource.OPERATOR


@pytest.mark.asyncio
async def test_leaving_pending_releases_the_watchdog_pin(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    await _refused_while_lows_dispatch(rig)
    await rig.tick_with_low_winner()
    assert rig.watchdog_pins() == ['2']

    rig.set_status('2', 'blocked')
    await rig.tick()

    assert '2' not in rig.override_rows()


@pytest.mark.asyncio
async def test_a_previous_process_watchdog_pin_is_released_at_dispatch(tmp_path: Path) -> None:
    store = OverrideStore(tmp_path / 'scheduler_overrides.db')
    assert store.create_watchdog_pin(
        str(tmp_path), 'W', ttl_until=T0 + timedelta(days=3650),
    ) == 1
    rig = _rig(tmp_path, tasks=[_task('W', 'low', 'free/w.py')])

    assert await rig.tick_id() == 'W'

    assert 'W' not in rig.override_rows()


# ---------------------------------------------------------------------------
# Kept watchdog behaviour (task 1880), on the wall clock
# ---------------------------------------------------------------------------


async def _top_skipped(rig: _Rig, ticks: int) -> None:
    """S is the top candidate, refused by the seed's lock, with no other winner."""
    rig.set_tasks([_seed(), _task('2', 'medium', 'hot/b.py')])
    assert await rig.tick_id() == '0'
    for _ in range(ticks):
        assert await rig.tick() is None


@pytest.mark.asyncio
async def test_dual_gate_fires_once_after_skips_and_age(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    await _top_skipped(rig, 3)
    rig.warn.assert_not_called()

    rig.advance(FLOOR)
    for _ in range(5):
        await rig.tick()

    assert rig.warned_ids() == ['2']
    assert 'STARVATION_WATCHDOG' in rig.warn.call_args.kwargs['summary']
    [event] = rig.detected('2')
    assert event['gate'] == 'dual'
    assert event['skip_count'] >= 3


@pytest.mark.asyncio
async def test_below_skip_threshold_does_not_fire(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    await _top_skipped(rig, 1)
    rig.advance(10 * FLOOR)
    await rig.tick()

    rig.warn.assert_not_called()
    assert rig.detected() == []


@pytest.mark.asyncio
async def test_below_age_floor_does_not_fire(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    await _top_skipped(rig, 6)
    rig.advance(FLOOR - 1)
    await rig.tick()

    rig.warn.assert_not_called()


@pytest.mark.asyncio
async def test_never_top_scored_low_task_does_not_fire_without_a_stuck_signal(
    tmp_path: Path,
) -> None:
    """The reify-5166 shape no longer fires: zero skips and zero refusals is backlog depth."""
    rig = _rig(tmp_path)
    low = _task('9000', 'low', 'backend/main.py')
    for n in range(4):
        high = _task(str(10 + n), 'high', 'frontend/main.py')
        rig.set_tasks([low, high])
        assert await rig.tick_id() == high['id']
        rig.scheduler.release(high['id'])
        if n == 1:
            rig.advance(10 * FLOOR)

    rig.warn.assert_not_called()
    assert rig.detected() == []


@pytest.mark.asyncio
async def test_disabled_watchdog_never_fires(tmp_path: Path) -> None:
    rig = _rig(tmp_path, enabled=False)
    await _top_skipped(rig, 5)
    rig.advance(10 * FLOOR)
    await rig.tick()

    rig.warn.assert_not_called()
    assert rig.detected() == []


@pytest.mark.asyncio
async def test_dispatch_resolves_the_escalation_and_restarts_the_clock(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    await _top_skipped(rig, 3)
    rig.advance(FLOOR)
    await rig.tick()
    assert rig.warned_ids() == ['2']

    rig.scheduler.release('0')
    assert await rig.tick_id() == '2'
    assert [str(c.args[0]) for c in rig.resolve.call_args_list] == ['2']

    rig.scheduler.release('2', requeued=True)
    rig.tasks[1]['status'] = 'pending'
    rig.tasks[0]['status'] = 'pending'
    rig.mono[0] += rig.scheduler.config.requeue_cooldown_secs + 1
    assert await rig.tick_id() == '0'
    for _ in range(4):
        await rig.tick()
    assert rig.warned_ids() == ['2'], 'the age restarts at dispatch, so no re-fire'


# ---------------------------------------------------------------------------
# Corroborate before acting; restart-safe dedup (task 3821)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('live', ['blocked', 'done'])
async def test_a_finding_the_live_read_contradicts_is_not_delivered(
    tmp_path: Path, live: str,
) -> None:
    rig = _rig(tmp_path)
    await _top_skipped(rig, 3)
    rig.advance(FLOOR)
    rig.live_status['2'] = live

    for _ in range(2):
        await rig.tick()

    assert rig.detected() == []
    rig.warn.assert_not_called()
    assert rig.override_rows() == {}


@pytest.mark.asyncio
async def test_a_failed_live_read_delivers_nothing_and_retries_next_tick(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    await _top_skipped(rig, 3)
    rig.advance(FLOOR)
    rig.status_faults.append(RuntimeError('fused-memory unreachable'))

    await rig.tick()
    assert rig.detected() == []
    rig.warn.assert_not_called()

    await rig.tick()
    [event] = rig.detected('2')
    assert event['gate'] == 'dual'
    assert rig.warned_ids() == ['2']


def _previous_process_alarm(queue: EscalationQueue, task_id: str, level: int) -> Escalation:
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role=STARVATION_WATCHDOG_ROLE,
        severity='info',
        category='risk_identified',
        summary=f'STARVATION_WATCHDOG [dual]: task {task_id}',
        detail='filed before the restart',
        suggested_action='manual_investigation',
        level=level,
    )
    queue.submit(esc)
    return esc


@pytest.mark.asyncio
@pytest.mark.parametrize('level', [0, 1])
async def test_an_open_alarm_from_a_previous_process_is_not_refiled_and_is_gc_resolved(
    tmp_path: Path, level: int,
) -> None:
    queue = EscalationQueue(tmp_path / 'esc')
    _previous_process_alarm(queue, '2', level)
    rig = _rig(tmp_path)
    rig.scheduler.escalation_queue = queue
    await _top_skipped(rig, 3)
    rig.advance(FLOOR)

    for _ in range(2):
        await rig.tick()

    assert rig.detected('2') == []
    rig.warn.assert_not_called()

    rig.set_status('2', 'blocked')
    await rig.tick()

    assert [str(c.args[0]) for c in rig.resolve.call_args_list] == ['2']


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['done', 'blocked'])
async def test_leaving_pending_resolves_the_escalation(tmp_path: Path, status: str) -> None:
    rig = _rig(tmp_path)
    await _top_skipped(rig, 3)
    rig.advance(FLOOR)
    await rig.tick()
    assert rig.warned_ids() == ['2']

    rig.tasks[1]['status'] = status
    await rig.tick()

    assert [str(c.args[0]) for c in rig.resolve.call_args_list] == ['2']
