"""The verify-cache registry: recorded greens, consumptions and the policy row (task 5652)."""

from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from orchestrator.verify_cache import keys, registry
from orchestrator.verify_evidence import UnitTier

_NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


def _key(unit_id: str = 'module:alpha:test', tier: UnitTier = UnitTier.MODULE, tree: str = 'a') -> keys.UnitKey:
    return keys.UnitKey.of(unit_id, tier, 'spec-1', (('spec', 'spec-1'), ('tree:alpha', tree * 40)))


def _registry(path: Path) -> registry.Registry:
    return registry.Registry(path, clock=lambda: _NOW)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / 'verify_cache.db'


@pytest.fixture
def store(db_path: Path) -> registry.Registry:
    return _registry(db_path)


class TestEntries:
    @pytest.mark.asyncio
    async def test_a_recorded_green_round_trips(self, store):
        key = _key()

        entry = await store.record_green(key, evidence_event_id=7, run_id='r1', sha='abc')
        view = await store.view([key, _key(tree='b')])

        assert (entry.unit_id, entry.key, entry.tier, entry.spec_hash, entry.inputs) == (
            key.unit_id, key.key, key.tier, key.spec_hash, key.inputs,
        )
        assert (entry.evidence_event_id, entry.run_id, entry.sha, entry.recorded_at) == (7, 'r1', 'abc', _NOW)
        assert view.entries == {(key.unit_id, key.key): entry}

    @pytest.mark.asyncio
    async def test_recording_the_same_unit_and_key_refreshes_one_entry(self, store):
        first = await store.record_green(_key(), evidence_event_id=7, run_id='r1', sha='abc')

        second = await store.record_green(_key(), evidence_event_id=9, run_id='r2', sha='def')

        assert second.entry_id == first.entry_id
        assert (second.evidence_event_id, second.run_id, second.sha) == (9, 'r2', 'def')
        assert (await store.summary()).entries_by_tier[UnitTier.MODULE] == 1

    @pytest.mark.asyncio
    async def test_the_same_unit_under_another_key_is_another_entry(self, store):
        first = await store.record_green(_key(tree='a'), evidence_event_id=7, run_id='r1', sha='abc')

        second = await store.record_green(_key(tree='b'), evidence_event_id=8, run_id='r1', sha='abc')

        assert second.entry_id != first.entry_id
        assert (await store.summary()).entries_by_tier[UnitTier.MODULE] == 2

    @pytest.mark.asyncio
    async def test_evicting_a_key_removes_its_entries_idempotently(self, store):
        key = _key()
        await store.record_green(key, evidence_event_id=7, run_id='r1', sha='abc')

        assert await store.evict_key(key.key, 'audit leak') == 1
        assert await store.evict_key(key.key, 'audit leak') == 0
        assert await store.evict_key('f' * 64, 'unknown') == 0
        assert (await store.view([key])).entries == {}

    @pytest.mark.asyncio
    async def test_an_evicted_entrys_id_is_never_reused(self, store):
        evicted = await store.record_green(_key(tree='a'), evidence_event_id=7, run_id='r1', sha='abc')
        await store.evict_key(evicted.key, 'audit leak')

        recorded = await store.record_green(_key(tree='b'), evidence_event_id=8, run_id='r2', sha='def')

        assert recorded.entry_id != evicted.entry_id


class TestPrune:
    @pytest.mark.asyncio
    async def test_prune_drops_only_greens_and_consumptions_older_than_their_cutoffs(self, db_path):
        clock = [_NOW - timedelta(days=30)]
        store = registry.Registry(db_path, clock=lambda: clock[0])
        old = await store.record_green(_key(tree='a'), evidence_event_id=7, run_id='r1', sha='abc')
        await store.consume('land-old', [old])
        clock[0] = _NOW
        fresh = await store.record_green(_key(tree='b'), evidence_event_id=8, run_id='r2', sha='def')
        await store.consume('land-fresh', [fresh])

        pruned = await store.prune(
            entries_recorded_before=_NOW - timedelta(days=7), consumed_before=_NOW - timedelta(days=7),
        )

        assert pruned == registry.Pruned(entries=1, consumed=1)
        assert (await store.view([_key(tree='a'), _key(tree='b')])).entries == {(fresh.unit_id, fresh.key): fresh}
        assert await store.consumed_for('land-old') == ()
        assert await store.consumed_for('land-fresh') == (fresh,)

    @pytest.mark.asyncio
    async def test_prune_compares_instants_not_offsets(self, store):
        await store.record_green(_key(), evidence_event_id=7, run_id='r1', sha='abc')
        an_hour_later_elsewhere = (_NOW + timedelta(hours=1)).astimezone(timezone(timedelta(hours=-5)))

        pruned = await store.prune(
            entries_recorded_before=an_hour_later_elsewhere, consumed_before=an_hour_later_elsewhere,
        )

        assert pruned.entries == 1

    @pytest.mark.asyncio
    async def test_prune_needs_timezone_aware_cutoffs(self, store):
        with pytest.raises(ValueError, match='entries_recorded_before'):
            await store.prune(entries_recorded_before=datetime(2026, 11, 10), consumed_before=_NOW)


class TestConsumption:
    @pytest.mark.asyncio
    async def test_a_landing_records_the_entries_it_consumed_once(self, store):
        entry = await store.record_green(_key(), evidence_event_id=7, run_id='r1', sha='abc')

        await store.consume('land1', [entry])
        await store.consume('land1', [entry])

        assert await store.consumed_for('land1') == (entry,)
        assert await store.consumed_for('land2') == ()

    @pytest.mark.asyncio
    async def test_a_consumption_survives_the_eviction_of_its_entry(self, store):
        entry = await store.record_green(_key(), evidence_event_id=7, run_id='r1', sha='abc')
        await store.consume('land1', [entry])

        await store.evict_key(entry.key, 'audit leak')

        assert await store.consumed_for('land1') == (entry,)


class TestPolicyState:
    @pytest.mark.asyncio
    async def test_a_fresh_registry_is_enabled(self, store):
        state = await store.policy_state()

        assert (state.disabled_until, state.streak, state.leaks) == (None, 0, ())
        assert not state.is_disabled(_NOW)

    @pytest.mark.asyncio
    async def test_a_written_state_round_trips(self, store):
        state = registry.PolicyState(
            disabled_until=_NOW + timedelta(days=14), reason='2 leaks', streak=2, leaks=('l1', 'l2'),
        )

        await store.write_policy_state(state)

        assert await store.policy_state() == state

    def test_a_disable_is_in_force_until_its_end(self):
        state = registry.PolicyState(
            disabled_until=_NOW + timedelta(days=14), reason='2 leaks', streak=2, leaks=('l1', 'l2'),
        )

        assert state.is_disabled(_NOW)
        assert not state.is_disabled(_NOW + timedelta(days=15))

    @pytest.mark.parametrize(('fields', 'named'), [
        ({'disabled_until': _NOW, 'reason': '', 'leaks': ('l1', 'l2')}, 'reason'),
        ({'disabled_until': _NOW, 'reason': 'x', 'leaks': ('l1',)}, "'l1'"),
        ({'disabled_until': None, 'reason': None, 'streak': -1}, '-1'),
        ({'disabled_until': datetime(2026, 11, 10), 'reason': 'x', 'leaks': ('l1', 'l2')}, '2026-11-10'),
    ])
    def test_a_state_outside_i5_is_rejected_naming_the_value(self, fields, named):
        with pytest.raises(ValueError, match=named):
            registry.PolicyState(**fields)


class TestStoreFile:
    def test_the_registry_lives_in_the_orchestrator_data_dir(self, tmp_path):
        assert registry.registry_db_path(tmp_path) == tmp_path / 'data' / 'orchestrator' / 'verify_cache.db'

    @pytest.mark.asyncio
    async def test_a_missing_store_is_a_cold_start(self, tmp_path):
        path = tmp_path / 'absent' / 'dir' / 'verify_cache.db'

        view = await _registry(path).view([_key()])

        assert view.entries == {}
        assert path.exists()

    @pytest.mark.asyncio
    async def test_a_summary_of_a_missing_store_creates_nothing(self, db_path):
        summary = await _registry(db_path).summary()

        assert summary.exists is False
        assert summary.entries_by_tier == dict.fromkeys(UnitTier, 0)
        assert summary.consumed == 0
        assert not db_path.exists()

    @pytest.mark.asyncio
    async def test_a_summary_counts_entries_per_tier(self, store):
        await store.record_green(_key(), evidence_event_id=7, run_id='r1', sha='abc')
        tree_key = keys.UnitKey.of('tree', UnitTier.TREE, 'plan-1', (('spec', 'plan-1'), ('tree', 't' * 40)))
        await store.record_green(tree_key, evidence_event_id=8, run_id='r1', sha='abc')

        summary = await store.summary()

        assert summary.exists is True
        assert summary.entries_by_tier == {UnitTier.TREE: 1, UnitTier.MODULE: 1, UnitTier.TEST: 0}

    @pytest.mark.asyncio
    @pytest.mark.parametrize('operation', ['view', 'record_green', 'summary'])
    @pytest.mark.parametrize('damage', ['garbage', 'directory'])
    async def test_a_damaged_store_is_a_registry_error_naming_it(self, db_path, operation, damage):
        if damage == 'garbage':
            db_path.write_bytes(b'not a sqlite database, just bytes' * 64)
        else:
            db_path.mkdir()
        store = _registry(db_path)
        calls = {
            'view': lambda: store.view([_key()]),
            'record_green': lambda: store.record_green(_key(), evidence_event_id=7, run_id='r1', sha='abc'),
            'summary': store.summary,
        }

        with pytest.raises(registry.RegistryError, match=re.escape(str(db_path))):
            await calls[operation]()


def _write_raw(db_path: Path, sql: str, params: tuple) -> None:
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.execute(sql, params)
    finally:
        conn.close()


class TestCorruptRows:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('operation', ['policy_state', 'summary'])
    @pytest.mark.parametrize(('streak', 'leaks', 'named'), [
        (-1, '[]', 'policy_state.streak must be >= 0, got -1'),
        (0, '5', 'int'),
        ('x', '[]', 'str'),
    ])
    async def test_a_policy_row_breaking_its_invariants_is_a_registry_error_naming_why(
        self, store, db_path, operation, streak, leaks, named,
    ):
        await store.policy_state()
        _write_raw(
            db_path, 'INSERT INTO policy_state (id, streak, leaks) VALUES (1, ?, ?)', (streak, leaks),
        )

        with pytest.raises(registry.RegistryError, match=named):
            await getattr(store, operation)()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('column', 'value', 'named'), [
        ('spec_hash', 'other', "spec_hash 'other'"),
        ('inputs', '5', 'int'),
        ('tier', 'galaxy', 'galaxy'),
    ])
    async def test_an_entry_row_breaking_its_invariants_is_a_registry_error_naming_why(
        self, store, db_path, column, value, named,
    ):
        key = _key()
        await store.record_green(key, evidence_event_id=7, run_id='r1', sha='abc')
        _write_raw(db_path, f'UPDATE entries SET {column} = ?', (value,))

        with pytest.raises(registry.RegistryError, match=named):
            await store.view([key])
