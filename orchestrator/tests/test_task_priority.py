"""A task's live tier and pin as a never-raising lookup (task 9908).

``Scheduler.task_priority`` answers from the scheduler's own memory;
``read_task_priority`` is the one guard every consumer reads it through.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generic, TypeVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from shared.project_layout import scheduler_state_json
from shared.scheduler_state import SchedulerStateSnapshot

from orchestrator.config import PRIORITY_TIERS, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.merge_verify_facts import ChainMembers, MergeVerifyDispatchFacts, RunnerReason
from orchestrator.overrides import OverrideRow, OverrideStore
from orchestrator.pause_policy import PauseClass
from orchestrator.scheduler import Scheduler
from orchestrator.task_priority import (
    GATE_RANK,
    PINNED,
    UNKNOWN_PRIORITY,
    TaskAge,
    TaskPriority,
    anchor_from_epoch,
    gate_rank,
    order_key,
    read_task_priority,
    task_age,
)

_ANCHOR = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
_NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
_PERIOD = 3600.0


def _pending(
    task_id: str, priority: str, *, deps: tuple[str, ...] = (), status: str = 'pending',
) -> dict:
    return {
        'id': task_id, 'title': f'Task {task_id}', 'status': status,
        'dependencies': [{'id': d} for d in deps],
        'metadata': {'files': [f'{task_id}/src']}, 'priority': priority,
    }


def _write_snapshot(path: Path, effective_priorities: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = SchedulerStateSnapshot(effective_priorities=effective_priorities)
    path.write_text(json.dumps(snapshot.to_json()))


def _scheduler(
    tmp_path: Path, *, with_store: bool, store_type: type[OverrideStore] = OverrideStore,
    time_source: Callable[[], float] | None = None, event_store: EventStore | None = None,
) -> tuple[Scheduler, OverrideStore | None]:
    config = OrchestratorConfig(max_per_module=1, project_root=tmp_path)
    store = store_type(tmp_path / 'overrides.db') if with_store else None
    scheduler = Scheduler(
        config, event_store, override_store=store, time_source=time_source,
        state_snapshot_path=tmp_path / 'scheduler_state.json',
    )
    scheduler.finish_startup()
    return scheduler, store


class _ThreadRecordingStore(OverrideStore):
    def __init__(self, db_path: Path) -> None:
        super().__init__(db_path)
        self.read_threads: list[threading.Thread] = []

    def get_overrides(self, project_root: str) -> dict[str, OverrideRow]:
        self.read_threads.append(threading.current_thread())
        return super().get_overrides(project_root)


class _BreakableStore(OverrideStore):
    broken = False

    def get_overrides(self, project_root: str) -> dict[str, OverrideRow]:
        if self.broken:
            raise OSError('store unreachable')
        return super().get_overrides(project_root)


_S = TypeVar('_S', bound=OverrideStore)


class _PausedScene(Generic[_S]):
    """A scheduler ticked once over task A (tier high), then paused by the operator."""

    def __init__(self, tmp_path: Path, store_type: type[_S]) -> None:
        self.tmp_path = tmp_path
        self.now = [1000.0]
        self.events = EventStore(tmp_path / 'events.db', run_id='run-9933')
        self.scheduler, store = _scheduler(
            tmp_path, with_store=True, store_type=store_type,
            time_source=lambda: self.now[0], event_store=self.events,
        )
        assert isinstance(store, store_type)
        self.store: _S = store
        self.scheduler.get_tasks = AsyncMock(return_value=[_pending('A', 'high')])

    async def tick_then_pause(self) -> None:
        await self.scheduler.acquire_next()
        self.scheduler.pause('test', PauseClass.OPERATOR)

    def pin_a(self) -> None:
        self.store.set_override(str(self.tmp_path), 'A', pinned=True)

    async def poll_after(self, secs: float) -> None:
        self.now[0] += secs
        assert await self.scheduler.acquire_next() is None


class TestSchedulerTaskPriority:

    def test_the_lookup_answers_from_the_last_snapshot_and_the_store_before_any_tick(
        self, tmp_path: Path,
    ) -> None:
        _write_snapshot(tmp_path / 'scheduler_state.json', {'P': 'low', 'A': 'high'})
        OverrideStore(tmp_path / 'overrides.db').set_override(
            str(tmp_path), 'P', pinned=True, pin_order=4,
        )

        scheduler, _ = _scheduler(tmp_path, with_store=True)

        assert scheduler.task_priority('P') == TaskPriority(
            tier=PINNED, pinned=True, base_tier='low', pin_order=4,
        )
        assert scheduler.task_priority('A') == TaskPriority(
            tier='high', pinned=False, base_tier='high',
        )
        assert scheduler.get_state_snapshot().effective_priorities == {'P': 'low', 'A': 'high'}

    def test_no_snapshot_file_leaves_tiers_unknown_but_pins_known(self, tmp_path: Path) -> None:
        OverrideStore(tmp_path / 'overrides.db').set_override(str(tmp_path), 'P', pinned=True)

        scheduler, _ = _scheduler(tmp_path, with_store=True)

        assert scheduler.task_priority('P') == TaskPriority(
            tier=PINNED, pinned=True, base_tier=None, pin_order=1,
        )
        assert scheduler.task_priority('A') == TaskPriority(tier=None, pinned=False)

    def test_the_snapshot_path_follows_the_project_root(self, tmp_path: Path) -> None:
        _write_snapshot(scheduler_state_json(tmp_path), {'A': 'high'})

        scheduler = Scheduler(OrchestratorConfig(project_root=tmp_path))

        assert scheduler.task_priority('A').base_tier == 'high'

    @pytest.mark.asyncio
    async def test_a_tick_supplies_the_effective_tier_and_the_pin(self, tmp_path: Path) -> None:
        scheduler, store = _scheduler(tmp_path, with_store=True)
        assert store is not None
        store.set_override(str(tmp_path), 'P', pinned=True)
        scheduler.get_tasks = AsyncMock(return_value=[
            _pending('A', 'medium'), _pending('B', 'high', deps=('A',)), _pending('P', 'low'),
        ])

        await scheduler.acquire_next()

        assert scheduler.task_priority('P') == TaskPriority(
            tier=PINNED, pinned=True, base_tier='low', pin_order=1,
        )
        assert scheduler.task_priority('A') == TaskPriority(
            tier='high', pinned=False, base_tier='high',
        )
        assert scheduler.task_priority('unranked') == TaskPriority(tier=None, pinned=False)
        effective = scheduler.get_state_snapshot().effective_priorities
        assert (effective['A'], effective['P']) == ('high', 'low')

    @pytest.mark.asyncio
    async def test_pin_order_is_the_rows_order(self, tmp_path: Path) -> None:
        scheduler, store = _scheduler(tmp_path, with_store=True)
        assert store is not None
        store.set_override(str(tmp_path), 'P', pinned=True, pin_order=2)
        store.set_override(str(tmp_path), 'Q', pinned=True, pin_order=5)
        scheduler.get_tasks = AsyncMock(return_value=[_pending('P', 'low'), _pending('Q', 'low')])

        await scheduler.acquire_next()

        assert (scheduler.task_priority('P').pin_order, scheduler.task_priority('Q').pin_order) == (
            2, 5,
        )

    @pytest.mark.asyncio
    async def test_no_override_store_leaves_the_pin_unknown(self, tmp_path: Path) -> None:
        scheduler, _ = _scheduler(tmp_path, with_store=False)
        scheduler.get_tasks = AsyncMock(return_value=[_pending('A', 'medium')])

        await scheduler.acquire_next()

        assert scheduler.task_priority('A') == TaskPriority(
            tier='medium', pinned=None, base_tier='medium',
        )

    @pytest.mark.asyncio
    async def test_a_pinned_in_progress_task_keeps_its_base_tier(self, tmp_path: Path) -> None:
        scheduler, store = _scheduler(tmp_path, with_store=True)
        assert store is not None
        store.set_override(str(tmp_path), 'R', pinned=True)
        scheduler.get_tasks = AsyncMock(return_value=[
            _pending('R', 'low', status='in-progress'), _pending('A', 'medium'),
        ])

        await scheduler.acquire_next()

        found = scheduler.task_priority('R')
        assert (found.tier, found.base_tier) == (PINNED, 'low')

    @pytest.mark.asyncio
    async def test_a_pin_set_while_paused_is_honoured_within_a_minute(self, tmp_path: Path) -> None:
        scene = _PausedScene(tmp_path, OverrideStore)
        await scene.tick_then_pause()
        scene.pin_a()

        await scene.poll_after(0.0)
        assert scene.scheduler.task_priority('A').tier == 'high'

        await scene.poll_after(60.0)
        assert scene.scheduler.task_priority('A') == TaskPriority(
            tier=PINNED, pinned=True, base_tier='high', pin_order=1,
        )

    @pytest.mark.asyncio
    async def test_the_paused_reread_leaves_the_ticks_override_snapshot_alone(
        self, tmp_path: Path,
    ) -> None:
        scene = _PausedScene(tmp_path, OverrideStore)
        await scene.tick_then_pause()
        scene.pin_a()
        await scene.poll_after(60.0)

        scene.scheduler.resume()
        await scene.scheduler.acquire_next()

        pinned = scene.events.fetch_events_by_type(EventType.task_pinned)
        assert [e['task_id'] for e in pinned] == ['A']

    @pytest.mark.asyncio
    async def test_the_paused_reread_runs_off_the_event_loop_thread(self, tmp_path: Path) -> None:
        scene = _PausedScene(tmp_path, _ThreadRecordingStore)
        await scene.tick_then_pause()
        scene.store.read_threads.clear()

        await scene.poll_after(60.0)

        assert scene.store.read_threads
        assert threading.current_thread() not in scene.store.read_threads

    @pytest.mark.asyncio
    async def test_a_failing_paused_reread_keeps_the_last_pins(self, tmp_path: Path) -> None:
        scene = _PausedScene(tmp_path, _BreakableStore)
        scene.pin_a()
        await scene.tick_then_pause()
        before = scene.scheduler.task_priority('A')
        scene.store.broken = True

        await scene.poll_after(60.0)

        assert before.tier == PINNED
        assert scene.scheduler.task_priority('A') == before


class TestReadTaskPriority:

    def test_no_lookup_is_unknown(self) -> None:
        assert read_task_priority(None, '7') is UNKNOWN_PRIORITY

    def test_a_raising_lookup_is_unknown(self) -> None:
        def broken(task_id: str) -> TaskPriority:
            raise KeyError(task_id)

        assert read_task_priority(broken, '7') is UNKNOWN_PRIORITY

    def test_an_answer_of_the_wrong_type_is_unknown(self) -> None:
        assert read_task_priority(MagicMock(), '7') is UNKNOWN_PRIORITY

    def test_a_field_of_the_wrong_type_is_unknown(self) -> None:
        def odd(task_id: str) -> TaskPriority:
            not_a_bool: Any = 'yes'
            return TaskPriority(tier=MagicMock(), pinned=not_a_bool)

        assert read_task_priority(odd, '7') == UNKNOWN_PRIORITY

    def test_the_task_id_reaches_the_lookup_as_a_string(self) -> None:
        asked: list[object] = []

        def recording(task_id: str) -> TaskPriority:
            asked.append(task_id)
            return TaskPriority(tier='high', pinned=True)

        assert read_task_priority(recording, 42) == TaskPriority(tier='high', pinned=True)
        assert asked == ['42']

    def test_the_new_fields_survive_the_guard(self) -> None:
        answer = TaskPriority(
            tier='pinned', base_tier='low', pinned=True, pin_order=3, age_anchor=_ANCHOR,
        )

        assert read_task_priority(lambda task_id: answer, '7') == answer

    @pytest.mark.parametrize('field, odd_value', [
        ('base_tier', MagicMock()),
        ('age_anchor', datetime(2026, 11, 1, 12, 0)),
        ('age_anchor', '2026-11-01T12:00:00+00:00'),
    ])
    def test_an_ill_typed_new_field_is_unknown(self, field: str, odd_value: Any) -> None:
        well_typed = TaskPriority(
            tier='pinned', base_tier='low', pinned=True, pin_order=3, age_anchor=_ANCHOR,
        )
        answer = dataclasses.replace(well_typed, **{field: odd_value})

        assert read_task_priority(lambda task_id: answer, '7') == dataclasses.replace(
            well_typed, **{field: None},
        )

    @pytest.mark.parametrize('odd_order', [None, True, '3'])
    def test_a_pinned_tier_without_a_plain_order_reads_as_order_zero(self, odd_order: Any) -> None:
        answer = TaskPriority(tier=PINNED, pinned=True, base_tier='low', pin_order=odd_order)

        assert read_task_priority(lambda task_id: answer, '7') == dataclasses.replace(
            answer, pin_order=0,
        )

    @pytest.mark.parametrize('tier', ['high', None, MagicMock()])
    def test_an_order_without_the_pinned_tier_is_dropped(self, tier: Any) -> None:
        answer = TaskPriority(tier=tier, pinned=False, base_tier='high', pin_order=3)

        assert read_task_priority(lambda task_id: answer, '7').pin_order is None

    def test_the_positional_shape_keeps_its_meaning(self) -> None:
        assert TaskPriority('high', False) == TaskPriority(
            tier='high', pinned=False, base_tier=None, pin_order=None, age_anchor=None,
        )

    def test_unknown_priority_has_every_field_unknown(self) -> None:
        assert all(
            getattr(UNKNOWN_PRIORITY, f.name) is None
            for f in dataclasses.fields(UNKNOWN_PRIORITY)
        )


def _key(
    p: TaskPriority, *, waited: float = 0.0, period: float | None = None,
    fallback: datetime | None = _ANCHOR,
) -> tuple[int, int, float]:
    return order_key(
        p, now=_NOW, waited_secs=waited, promote_after_secs=period, fallback_anchor=fallback,
    )


def _rank(tier: str | None, *, waited: float = 0.0, period: float | None = _PERIOD) -> int:
    return _key(TaskPriority(tier=tier), waited=waited, period=period)[0]


def test_task_priority_imports_nothing_from_the_orchestrator() -> None:
    probe = (
        'import json, sys, orchestrator.task_priority; '
        'print(json.dumps(sorted(m for m in sys.modules if m.startswith("orchestrator."))))'
    )

    ran = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, check=True)

    assert json.loads(ran.stdout) == ['orchestrator.task_priority']


class TestGateRank:

    def test_pinned_ranks_ahead_of_every_tier(self) -> None:
        assert PINNED == 'pinned'
        assert (PINNED, *PRIORITY_TIERS) == GATE_RANK

    def test_a_known_tier_ranks_by_its_place(self) -> None:
        assert gate_rank('pinned') == 0
        assert gate_rank('medium') == GATE_RANK.index('medium')

    @pytest.mark.parametrize('tier', [None, 'urgent'])
    def test_an_unknown_tier_has_no_rank(self, tier: str | None) -> None:
        assert gate_rank(tier) is None


class TestOrderKey:

    def test_tiers_sort_pinned_first_then_by_tier(self) -> None:
        waiters = [TaskPriority(tier=t) for t in reversed(PRIORITY_TIERS)]
        waiters.append(TaskPriority(tier=PINNED, pin_order=1))

        assert [p.tier for p in sorted(waiters, key=_key)] == [PINNED, *PRIORITY_TIERS]

    def test_pin_order_orders_inside_pinned(self) -> None:
        first, second = TaskPriority(tier=PINNED, pin_order=1), TaskPriority(tier=PINNED, pin_order=2)

        assert sorted([second, first], key=_key) == [first, second]

    def test_a_pin_with_no_order_keys_as_zero(self) -> None:
        assert _key(TaskPriority(tier=PINNED))[1] == 0

    def test_an_older_anchor_sorts_first_within_a_tier(self) -> None:
        older = TaskPriority(tier='low', age_anchor=datetime(2026, 11, 1, 0, 0, tzinfo=UTC))
        newer = TaskPriority(tier='low', age_anchor=datetime(2026, 11, 2, 0, 0, tzinfo=UTC))

        assert sorted([newer, older], key=_key) == [older, newer]

    def test_the_third_element_is_minus_the_age(self) -> None:
        assert _key(TaskPriority(tier='low', age_anchor=_ANCHOR))[2] == -86400.0

    def test_the_anchor_wins_over_the_fallback(self) -> None:
        anchored = TaskPriority(tier='low', age_anchor=datetime(2026, 11, 2, 11, 0, tzinfo=UTC))

        assert _key(anchored, fallback=_ANCHOR)[2] == -3600.0

    def test_the_fallback_is_used_without_an_anchor(self) -> None:
        assert _key(TaskPriority(tier='low'), fallback=_ANCHOR)[2] == -86400.0

    def test_with_neither_anchor_the_age_is_zero(self) -> None:
        assert _key(TaskPriority(tier='low'), fallback=None)[2] == 0.0

    @pytest.mark.parametrize('tier', [None, 'urgent'])
    def test_an_unknown_tier_keys_as_medium(self, tier: str | None) -> None:
        assert _rank(tier, period=None) == gate_rank('medium')

    @pytest.mark.parametrize('tier, waited_periods, promoted_to', [
        ('low', 0.99, 'low'),
        ('low', 1.0, 'medium'),
        ('low', 2.9, 'high'),
        ('polish', 10.0, 'high'),
        ('critical', 10.0, 'critical'),
        (PINNED, 10.0, PINNED),
    ])
    def test_a_long_wait_promotes_never_past_high(
        self, tier: str, waited_periods: float, promoted_to: str,
    ) -> None:
        assert _rank(tier, waited=waited_periods * _PERIOD) == gate_rank(promoted_to)

    def test_no_period_promotes_nothing(self) -> None:
        assert _rank('polish', waited=1e9, period=None) == gate_rank('polish')

    @pytest.mark.parametrize('waited', [float('nan'), float('inf'), -_PERIOD])
    def test_an_odd_wait_promotes_nothing(self, waited: float) -> None:
        assert _rank('low', waited=waited) == gate_rank('low')

    @pytest.mark.parametrize('period', [0.0, -_PERIOD, float('nan')])
    def test_an_odd_period_promotes_nothing(self, period: float) -> None:
        assert _rank('low', waited=10 * _PERIOD, period=period) == gate_rank('low')

    def test_a_naive_fallback_ages_nothing(self) -> None:
        naive = datetime(2026, 11, 1, 12, 0)

        assert _key(TaskPriority(tier='low'), fallback=naive) == (gate_rank('low'), 0, 0.0)


class TestTaskAge:

    def test_an_anchor_measures_from_the_anchor(self) -> None:
        anchored = TaskPriority(age_anchor=_ANCHOR)

        assert task_age(anchored, now=_NOW, fallback_anchor=None) == TaskAge(
            86400.0, from_anchor=True,
        )

    def test_without_an_anchor_the_fallback_is_measured(self) -> None:
        fallback = datetime(2026, 11, 2, 11, 0, tzinfo=UTC)

        assert task_age(TaskPriority(), now=_NOW, fallback_anchor=fallback) == TaskAge(
            3600.0, from_anchor=False,
        )


class TestAnchorFromEpoch:

    def test_a_representable_epoch_is_that_instant_in_utc(self) -> None:
        anchor = anchor_from_epoch(_NOW.timestamp())

        assert anchor == _NOW
        assert anchor is not None and anchor.tzinfo is UTC

    @pytest.mark.parametrize('epoch', [float('nan'), float('inf'), float('-inf'), 1.7e12, -1e20])
    def test_an_epoch_naming_no_instant_is_no_anchor(self, epoch: float) -> None:
        assert anchor_from_epoch(epoch) is None


class TestChainMembersOnTheVerifyRow:

    def test_members_add_both_lists_in_order(self) -> None:
        facts = MergeVerifyDispatchFacts(runner_reason=RunnerReason.REMOTE_BUSY).covering(
            ChainMembers(('mr-a', 'mr-b'), ('1', '2')),
        )

        assert facts.event_fields() == {
            'runner_reason': 'remote_busy', 'base_relation': None,
            'chain_depth_reason': None,
            'chain_request_ids': ['mr-a', 'mr-b'], 'chain_task_ids': ['1', '2'],
        }

    def test_a_retry_covers_the_same_members(self) -> None:
        members = ChainMembers(('mr-a',), ('1',))

        assert MergeVerifyDispatchFacts().covering(members).as_retry().members == members
