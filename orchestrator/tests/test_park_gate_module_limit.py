"""The park gate honours the module limit (task 7313).

A park on a key takes one of that key's ``_limit_for`` slots instead of
reserving it exclusively.  A rank-blocking park refuses a request only once
the occupants — the foreign holders plus the blocking park owners, each task
counted once — fill ``min(limit of the requested key, limit of the parked
key)``.  Wherever a limit of 1 appears that minimum is 1, so a limit-1 park
stays exclusive, hierarchical cases included.

Every assertion goes through the lock table's public seams.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _park_test_helpers import event_payloads, make_task, make_withheld_task
from _recording_event_store import _RecordingEventStore

from orchestrator.config import OrchestratorConfig
from orchestrator.pin_reservation import BlockerKind
from orchestrator.scheduler import ModuleLockTable, Scheduler

WIDE = 'pkg/tests/wide.json'
NARROW = 'pkg/tests/narrow.json'
PARENT = 'pkg/tests'
WIDE_LIMIT = 3


def _table() -> ModuleLockTable:
    """lock_depth 3: WIDE has 3 slots; NARROW, PARENT and every other key have 1."""
    return ModuleLockTable(OrchestratorConfig(
        max_per_module=1, lock_depth=3, module_overrides={WIDE: WIDE_LIMIT},
    ))


def _acquire_all(table: ModuleLockTable, holders: list[str], module: str) -> None:
    for holder in holders:
        assert table.try_acquire(holder, [module]), f'{holder} must take a free slot on {module}'


def _parked_owners(table: ModuleLockTable, task_id: str, module: str) -> set[str]:
    return {
        b.owner for b in table.blockers(task_id, [module]) if b.kind is BlockerKind.PARKED
    }


class TestAWidenedParkTakesOneSlot:
    def test_a_park_and_two_holders_fill_a_three_slot_key(self):
        table = _table()
        table.install_parks('P', [WIDE], 'medium')
        _acquire_all(table, ['C1', 'C2'], WIDE)

        assert table.park_owners_blocking('C3', [WIDE]) == ['P']
        assert not table.try_acquire('C3', [WIDE])

    def test_without_the_park_the_third_holder_fits(self):
        table = _table()
        _acquire_all(table, ['C1', 'C2', 'C3'], WIDE)

    def test_the_park_owner_still_takes_its_slot(self):
        table = _table()
        table.install_parks('P', [WIDE], 'medium')
        _acquire_all(table, ['C1', 'C2'], WIDE)

        assert table.try_acquire('P', [WIDE])


class TestALimitOneParkStaysExclusive:
    def test_exact_key(self):
        table = _table()
        table.install_parks('P', [NARROW], 'medium')

        assert table.park_owners_blocking('C', [NARROW]) == ['P']
        assert not table.try_acquire('C', [NARROW])

    def test_a_parent_park_blocks_a_widened_child(self):
        table = _table()
        table.install_parks('P', [PARENT], 'medium')

        assert not table.try_acquire('C', [WIDE])
        assert table.park_owners_blocking('C', [WIDE]) == ['P']

    def test_a_widened_child_park_blocks_a_limit_one_parent(self):
        table = _table()
        table.install_parks('P', [WIDE], 'medium')

        assert not table.try_acquire('C', [PARENT])
        assert table.park_owners_blocking('C', [PARENT]) == ['P']


class TestOccupantCounting:
    def test_a_buried_park_takes_no_slot(self):
        table = _table()
        table.install_parks('L', [WIDE], 'low')
        table.install_parks('T', [WIDE], 'high')
        _acquire_all(table, ['H'], WIDE)

        assert table.park_owners_blocking('C', [WIDE]) == []
        assert table.try_acquire('C', [WIDE])

    def test_a_holder_that_also_parks_counts_once(self):
        table = _table()
        _acquire_all(table, ['P'], WIDE)
        table.install_parks('P', [WIDE], 'medium')
        _acquire_all(table, ['H'], WIDE)

        assert table.try_acquire('C', [WIDE])


@pytest.mark.parametrize('k', [0, 1, 2])
def test_the_gate_and_its_naming_views_agree(k: int):
    table = _table()
    table.install_parks('P', [WIDE], 'medium')
    _acquire_all(table, [f'H{i}' for i in range(k)], WIDE)

    named = table.park_owners_blocking('C', [WIDE])
    parked = _parked_owners(table, 'C', WIDE)
    acquired = table.try_acquire('C', [WIDE])

    assert named == (['P'] if k == WIDE_LIMIT - 1 else [])
    assert parked == set(named)
    assert acquired == (named == [])


def test_an_admission_borrow_at_capacity_waives_the_park_only():
    table = _table()
    table.install_parks('P', [WIDE], 'medium')
    _acquire_all(table, ['H1', 'H2'], WIDE)
    stacks = table.snapshot_park_stacks()

    assert not table.try_acquire('C', [WIDE])
    admitted = frozenset(table.park_owners_blocking('C', [WIDE]))
    assert table.try_acquire('C', [WIDE], admitted_parks=admitted)
    assert table.snapshot_park_stacks() == stacks

    assert not table.try_acquire('D', [WIDE], admitted_parks=frozenset({'P'})), (
        'the held gate still caps the key at its limit'
    )


class TestLockExpansion:
    def test_expanding_onto_a_widened_parked_key(self):
        table = _table()
        _acquire_all(table, ['C'], 'pkg/other.py')
        table.install_parks('P', [WIDE], 'medium')

        assert table.try_acquire_additional('C', [WIDE])

    def test_expanding_onto_a_limit_one_parked_key(self):
        table = _table()
        _acquire_all(table, ['C'], 'pkg/other.py')
        table.install_parks('P', [NARROW], 'medium')

        assert not table.try_acquire_additional('C', [NARROW])


def test_the_rule_follows_max_per_module_without_any_override():
    table = ModuleLockTable(OrchestratorConfig(
        max_per_module=2, lock_depth=3, module_overrides={},
    ))
    table.install_parks('P', ['pkg/m.py'], 'medium')

    assert table.try_acquire('C', ['pkg/m.py'])
    assert table.park_owners_blocking('D', ['pkg/m.py']) == ['P']
    assert not table.try_acquire('D', ['pkg/m.py'])


def _scheduler(module_overrides: dict[str, int]) -> tuple[Scheduler, _RecordingEventStore]:
    """A started Scheduler where withheld F parks 'wide.json' and C declares it."""
    config = OrchestratorConfig(
        max_per_module=1, lock_depth=2, module_overrides=module_overrides,
    )
    store = _RecordingEventStore()
    scheduler = Scheduler(config, event_store=store)  # type: ignore[arg-type]
    scheduler.finish_startup()
    scheduler.lock_table.install_parks('F', ['wide.json'], 'high')
    scheduler.get_tasks = AsyncMock(return_value=[
        make_task('C', 'medium', ['wide.json']),
        make_withheld_task('F', 'high', ['wide.json']),
    ])
    return scheduler, store


@pytest.mark.asyncio
async def test_a_widened_park_admits_a_declarer_through_the_ordinary_gate():
    scheduler, store = _scheduler({'wide.json': WIDE_LIMIT})

    assignment = await scheduler.acquire_next()

    assert assignment is not None and assignment.task_id == 'C'
    assert event_payloads(store, 'park_backfill_granted') == [], (
        'C was admitted by the gate itself, not by a backfill borrow'
    )
    stacks = scheduler.lock_table.snapshot_park_stacks()
    assert [entry['owner'] for entry in stacks['wide.json']] == ['F']


@pytest.mark.asyncio
async def test_a_limit_one_park_still_withholds_the_declarer():
    scheduler, _store = _scheduler({})

    assert await scheduler.acquire_next() is None
