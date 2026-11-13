"""Scheduler.acquire_uncontested: the live-slot widen primitive (task 3090).

Unlike handle_blast_radius_expansion, it never re-pends and never releases:
it takes what is free, emits lock_acquired through the single lock-event
writer, and names each contested key with its holder.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _orch_helpers import HermeticMcpSession
from _workflow_helpers import FakeScheduler

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.lock_table import UncontestedAcquisition
from orchestrator.pin_reservation import Blocker, BlockerKind
from orchestrator.scheduler import Scheduler, SchedulerFacade

FIXED_DT = datetime(2026, 8, 1, 0, 0, 0, tzinfo=UTC)
_REASON = 'implement_scope_expanded'


@dataclass(frozen=True)
class _Rig:
    scheduler: Scheduler
    store: EventStore
    status: AsyncMock


def _scheduler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Rig:
    config = OrchestratorConfig(max_per_module=1, lock_depth=12)
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


def _lock_events(store: EventStore, event_type: EventType) -> list[dict]:
    return store.fetch_events_by_type(event_type)


@pytest.mark.asyncio
async def test_takes_the_free_keys_and_names_the_held_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    rig = _scheduler(tmp_path, monkeypatch)
    scheduler, store = rig.scheduler, rig.store
    assert scheduler.lock_table.try_acquire('t1', ['a.py'])
    assert scheduler.lock_table.try_acquire('t2', ['c.py'])

    result = await scheduler.acquire_uncontested('t1', ['b.py', 'c.py'], reason=_REASON)

    assert result.acquired == ('b.py',)
    assert [blocker.owner for blocker in result.contested] == ['t2']
    acquired_events = [e for e in _lock_events(store, EventType.lock_acquired) if e['task_id'] == 't1']
    assert len(acquired_events) == 1
    assert acquired_events[0]['data']['modules'] == ['b.py']
    assert acquired_events[0]['data']['reason'] == _REASON
    assert scheduler.lock_table.snapshot_holders() == {
        'a.py': 't1',
        'b.py': 't1',
        'c.py': 't2',
    }
    rig.status.assert_not_awaited()
    assert _lock_events(store, EventType.lock_released) == []


@pytest.mark.asyncio
async def test_all_contested_emits_no_lock_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    rig = _scheduler(tmp_path, monkeypatch)
    scheduler, store = rig.scheduler, rig.store
    assert scheduler.lock_table.try_acquire('t1', ['a.py'])
    assert scheduler.lock_table.try_acquire('t2', ['b.py', 'c.py'])

    result = await scheduler.acquire_uncontested('t1', ['b.py', 'c.py'], reason=_REASON)

    assert result.acquired == ()
    assert _lock_events(store, EventType.lock_acquired) == []
    assert _lock_events(store, EventType.lock_released) == []
    rig.status.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_fake_scheduler_conforms_and_honours_its_contested_map():
    fake = FakeScheduler()
    assert isinstance(fake, SchedulerFacade)
    fake.contested_modules = {'c.py': 't2'}

    result = await fake.acquire_uncontested('t1', ['b.py', 'c.py'], reason=_REASON)

    assert result == UncontestedAcquisition(
        acquired=('b.py',),
        contested=(Blocker('c.py', 't2', BlockerKind.HELD),),
    )
    assert fake.acquire_uncontested_calls == [('t1', ['b.py', 'c.py'], _REASON)]
    assert fake.statuses == {}
