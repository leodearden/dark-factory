"""The successor relation at the Scheduler boundary (early start, PRD section 5.2).

``plans/early-start-on-predecessor-branch-prd.md`` boundary rows 2, 7 (the
acquire half) and 12: a dependent D taken as the successor of its predecessor P
shares P's single-holder keys, a third task X waits for both, a two-holder key
never gets a third holder, and P's plan-time widening onto a key D holds is not
refused.  Every lock_acquired row of a successor names its predecessor, and the
state snapshot publishes the relation.

The Scheduler runs on the code defaults: pydantic-settings would otherwise merge
the operational yaml's own ``module_overrides`` into the test's holder limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _orch_helpers import HermeticMcpSession
from _park_test_helpers import make_task

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.lock_table import UncontestedAcquisition
from orchestrator.pin_reservation import Blocker, BlockerKind
from orchestrator.scheduler import Scheduler

pytestmark = [pytest.mark.usefixtures('code_default_config'), pytest.mark.asyncio]

FIXED_DT = datetime(2026, 11, 22, 0, 0, 0, tzinfo=UTC)
P_FILES = ['a.py', 'b.py']
D_FILES = ['a.py', 'c.py']


@dataclass(frozen=True)
class _Rig:
    scheduler: Scheduler
    store: EventStore
    status: AsyncMock


def _rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Rig:
    config = OrchestratorConfig(max_per_module=1, lock_depth=12, module_overrides={'two.py': 2})
    store = EventStore(tmp_path / 'runs.db', 'run-1')
    scheduler = Scheduler(
        config,
        mcp_session=HermeticMcpSession(),
        wall_time_source=lambda: FIXED_DT,
        event_store=store,
    )
    scheduler.finish_startup()
    status = AsyncMock()
    monkeypatch.setattr(scheduler, 'set_task_status', status)
    return _Rig(scheduler=scheduler, store=store, status=status)


def _offer(scheduler: Scheduler, *tasks: dict) -> None:
    scheduler.get_tasks = AsyncMock(return_value=list(tasks))


def _acquired_rows(store: EventStore, task_id: str) -> list[dict]:
    return [
        event['data']
        for event in store.fetch_events_by_type(EventType.lock_acquired)
        if event['task_id'] == task_id
    ]


def _released_rows(store: EventStore, task_id: str) -> list[dict]:
    return [
        event['data']
        for event in store.fetch_events_by_type(EventType.lock_released)
        if event['task_id'] == task_id
    ]


def _held(module: str, *owners: str) -> list[Blocker]:
    return [Blocker(module, owner, BlockerKind.HELD) for owner in owners]


async def _dispatch_p_then_take_d(rig: _Rig) -> None:
    """P is dispatched by the ordinary pass on P_FILES; D is then taken as its successor on D_FILES."""
    _offer(rig.scheduler, make_task('P', 'medium', P_FILES))
    assignment = await rig.scheduler.acquire_next()
    assert assignment is not None and assignment.task_id == 'P'

    assert rig.scheduler.acquire_as_successor('D', D_FILES, predecessor='P', priority='medium')


def _offer_x_beside_the_pair(scheduler: Scheduler, x_files: list[str]) -> None:
    _offer(
        scheduler,
        make_task('P', 'medium', P_FILES, status='in-progress'),
        make_task('D', 'medium', D_FILES, status='in-progress'),
        make_task('X', 'high', x_files),
    )


class TestRow2AThirdTaskWaitsForBoth:
    async def test_x_waits_until_both_partners_have_released(self, tmp_path, monkeypatch):
        rig = _rig(tmp_path, monkeypatch)
        scheduler = rig.scheduler
        await _dispatch_p_then_take_d(rig)
        _offer_x_beside_the_pair(scheduler, ['a.py'])

        assert await scheduler.acquire_next() is None
        assert scheduler.lock_table.blockers('X', ['a.py']) == _held('a.py', 'D', 'P')

        scheduler.release('P')
        assert await scheduler.acquire_next() is None
        assert scheduler.lock_table.blockers('X', ['a.py']) == _held('a.py', 'D')

        scheduler.release('D')
        assignment = await scheduler.acquire_next()
        assert assignment is not None and assignment.task_id == 'X'


class TestSuccessorOfOnTheLockEvent:
    async def test_the_successors_acquire_names_its_predecessor(self, tmp_path, monkeypatch):
        rig = _rig(tmp_path, monkeypatch)
        await _dispatch_p_then_take_d(rig)

        assert _acquired_rows(rig.store, 'D') == [
            {'modules': D_FILES, 'priority': 'medium', 'successor_of': 'P'},
        ]

    async def test_ordinary_acquires_carry_no_successor_of(self, tmp_path, monkeypatch):
        rig = _rig(tmp_path, monkeypatch)
        scheduler = rig.scheduler
        await _dispatch_p_then_take_d(rig)
        scheduler.release('P')
        scheduler.release('D')
        _offer(scheduler, make_task('X', 'high', ['a.py']))
        assert await scheduler.acquire_next() is not None

        for task_id in ('P', 'X'):
            rows = _acquired_rows(rig.store, task_id)
            assert rows and all('successor_of' not in row for row in rows), task_id

    async def test_a_plan_refinement_widen_of_the_successor_names_its_predecessor(
        self, tmp_path, monkeypatch
    ):
        rig = _rig(tmp_path, monkeypatch)
        await _dispatch_p_then_take_d(rig)

        result = await rig.scheduler.handle_blast_radius_expansion(
            'D', D_FILES, [*D_FILES, 'e.py'], persist_files=[*D_FILES, 'e.py'],
        )

        assert result.applied
        assert _acquired_rows(rig.store, 'D')[-1] == {
            'modules': ['e.py'], 'reason': 'plan_refinement', 'successor_of': 'P',
        }


class TestRow7TwoHolderKey:
    async def test_the_successor_is_refused_when_two_already_hold(self, tmp_path, monkeypatch):
        rig = _rig(tmp_path, monkeypatch)
        scheduler = rig.scheduler
        assert scheduler.lock_table.try_acquire('P', ['two.py'])
        assert scheduler.lock_table.try_acquire('U', ['two.py'])

        assert not scheduler.acquire_as_successor('D', ['two.py'], predecessor='P', priority='medium')

        assert _acquired_rows(rig.store, 'D') == []
        assert scheduler.get_state_snapshot().successor_of == {}

    async def test_a_third_task_is_refused_once_the_pair_fills_the_key(self, tmp_path, monkeypatch):
        rig = _rig(tmp_path, monkeypatch)
        scheduler = rig.scheduler
        _offer(scheduler, make_task('P', 'medium', ['two.py']))
        assert await scheduler.acquire_next() is not None

        assert scheduler.acquire_as_successor('D', ['two.py'], predecessor='P', priority='medium')
        _offer(
            scheduler,
            make_task('P', 'medium', ['two.py'], status='in-progress'),
            make_task('D', 'medium', ['two.py'], status='in-progress'),
            make_task('X', 'high', ['two.py']),
        )

        assert await scheduler.acquire_next() is None
        assert scheduler.lock_table.blockers('X', ['two.py']) == _held('two.py', 'D', 'P')


class TestRow12ThePredecessorWidensOntoItsSuccessorsKey:
    def _related(self, tmp_path, monkeypatch) -> _Rig:
        """P holds a.py; D, its successor, co-holds a.py and alone holds c.py."""
        rig = _rig(tmp_path, monkeypatch)
        assert rig.scheduler.lock_table.try_acquire('P', ['a.py'])
        assert rig.scheduler.acquire_as_successor('D', D_FILES, predecessor='P', priority='medium')
        return rig

    async def test_the_plan_time_widen_is_applied(self, tmp_path, monkeypatch):
        rig = self._related(tmp_path, monkeypatch)

        result = await rig.scheduler.handle_blast_radius_expansion(
            'P', ['a.py'], D_FILES, persist_files=D_FILES,
        )

        assert result.applied
        assert rig.scheduler.lock_table.held_modules('P') == {'a.py', 'c.py'}
        rig.status.assert_not_awaited()
        assert _released_rows(rig.store, 'P') == []
        assert _acquired_rows(rig.store, 'P') == [
            {'modules': ['c.py'], 'reason': 'plan_refinement'},
        ]

    async def test_control_an_unrelated_holder_still_sends_the_widener_back(
        self, tmp_path, monkeypatch
    ):
        rig = _rig(tmp_path, monkeypatch)
        assert rig.scheduler.lock_table.try_acquire('P', ['a.py'])
        assert rig.scheduler.lock_table.try_acquire('D', ['c.py'])

        result = await rig.scheduler.handle_blast_radius_expansion(
            'P', ['a.py'], D_FILES, persist_files=D_FILES,
        )

        assert not result.applied
        rig.status.assert_awaited_once_with('P', 'pending')
        assert not rig.scheduler.lock_table.is_held('P')

    async def test_the_in_place_widen_takes_the_key_uncontested(self, tmp_path, monkeypatch):
        rig = self._related(tmp_path, monkeypatch)

        result = await rig.scheduler.acquire_uncontested(
            'P', ['c.py'], reason='implement_scope_expanded',
        )

        assert result == UncontestedAcquisition(acquired=('c.py',), contested=())
        assert rig.scheduler.lock_table.held_modules('P') == {'a.py', 'c.py'}


class TestStateSnapshot:
    async def test_the_snapshot_publishes_the_relation_until_a_release(
        self, tmp_path, monkeypatch
    ):
        rig = _rig(tmp_path, monkeypatch)
        scheduler = rig.scheduler
        await _dispatch_p_then_take_d(rig)

        snapshot = scheduler.get_state_snapshot()
        assert snapshot.successor_of == {'D': 'P'}
        assert snapshot.held_by_task['P'] == tuple(P_FILES)
        assert snapshot.held_by_task['D'] == tuple(D_FILES)

        scheduler.release('D')

        assert scheduler.get_state_snapshot().successor_of == {}
