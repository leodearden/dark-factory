"""Recording a verify's admitted greens, and consuming the entries it replayed (task 5653)."""

from __future__ import annotations

from pathlib import Path

import pytest
from _verify_cache_units import (
    MERGE_LOG,
    clean_leg,
    merge_replay,
    ran_unit,
    record_entry,
    recorded_entries,
    unit_decision,
    unit_key,
)

from orchestrator.verify_cache.keys import UnitKey
from orchestrator.verify_cache.policy import Decision
from orchestrator.verify_cache.recorder import (
    LegRun,
    VerifyEvidenceRef,
    consume_replayed,
    record_greens,
)
from orchestrator.verify_cache.registry import Registry
from orchestrator.verify_cache.replay import ReplayPlan
from orchestrator.verify_evidence import Leg, VerifyUnit


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / 'verify_cache.db')


def _all_admitted() -> tuple[ReplayPlan, tuple[VerifyUnit, ...], tuple[LegRun, ...]]:
    unit_ids = ('module:alpha:test', 'module:alpha:lint')
    replay = merge_replay(
        {unit_id: unit_decision(Decision.MISS, key=unit_key(unit_id)) for unit_id in unit_ids}, modes=MERGE_LOG,
    )
    legs = (LegRun(unit_ids[0], clean_leg(Leg.TEST)), LegRun(unit_ids[1], clean_leg(Leg.LINT)))
    return replay, tuple(ran_unit(unit_id) for unit_id in unit_ids), legs


def _evidence(event_id: int = 11, *, passed: bool = True) -> VerifyEvidenceRef:
    return VerifyEvidenceRef(event_id=event_id, run_id='r1', sha='merge-sha', passed=passed)


class TestRecordGreens:
    @pytest.mark.asyncio
    async def test_only_admitted_units_are_recorded(self, registry):
        unit_ids = ('module:alpha:test', 'module:alpha:lint', 'module:alpha:type', 'module:beta:test')
        keys_ = {unit_id: unit_key(unit_id) for unit_id in unit_ids}
        replay = merge_replay({unit_id: unit_decision(Decision.MISS, key=keys_[unit_id]) for unit_id in unit_ids})
        legs = (
            LegRun('module:alpha:test', clean_leg(Leg.TEST)),
            LegRun('module:alpha:lint', clean_leg(Leg.LINT, rc=1)),
            LegRun('module:alpha:type', clean_leg(Leg.TYPE, timed_out=True)),
            LegRun('module:beta:test', clean_leg(Leg.TEST, collected=3)),
        )
        ran = (ran_unit(unit_ids[0]), ran_unit(unit_ids[1], 1), ran_unit(unit_ids[2]), ran_unit(unit_ids[3]))
        _units, record = replay.settle(ran, legs)

        outcome = await record_greens(registry, record, flake_suppressed=False, evidence=_evidence())

        recorded = await recorded_entries(registry, *keys_.values())
        assert set(recorded) == {'module:alpha:test'}
        assert recorded['module:alpha:test'].evidence_event_id == 11
        assert (recorded['module:alpha:test'].run_id, recorded['module:alpha:test'].sha) == ('r1', 'merge-sha')
        assert outcome.reason

    @pytest.mark.asyncio
    async def test_a_never_cache_leg_is_not_recorded(self, registry):
        key = unit_key('module:alpha:test')
        replay = merge_replay({'module:alpha:test': unit_decision(Decision.MISS, key=key)})
        legs = (LegRun('module:alpha:test', clean_leg(Leg.TEST, never_cache=True)),)
        _units, record = replay.settle((ran_unit('module:alpha:test'),), legs)

        await record_greens(registry, record, flake_suppressed=False, evidence=_evidence())

        assert await recorded_entries(registry, key) == {}

    @pytest.mark.asyncio
    async def test_a_re_record_refreshes_the_one_row(self, registry):
        replay, ran, legs = _all_admitted()
        _units, record = replay.settle(ran, legs)

        await record_greens(registry, record, flake_suppressed=False, evidence=_evidence(11))
        await record_greens(registry, record, flake_suppressed=False, evidence=_evidence(12))

        key = replay.modules['module:alpha:test'].key
        assert key is not None
        view = await registry.view([key])
        assert [entry.evidence_event_id for entry in view.entries.values()] == [12]

    @pytest.mark.asyncio
    async def test_a_suppressed_flake_records_nothing(self, registry):
        replay, ran, legs = _all_admitted()
        _units, record = replay.settle(ran, legs)

        await record_greens(registry, record, flake_suppressed=True, evidence=_evidence())

        assert await recorded_entries(registry, *_keys_of(replay)) == {}

    @pytest.mark.asyncio
    async def test_a_replayed_unit_is_not_new_evidence(self, registry):
        key = unit_key('module:alpha:lint')
        entry = await record_entry(registry, key, evidence=5)
        replay = merge_replay({'module:alpha:lint': unit_decision(Decision.HIT, entry, key=key)})
        _units, record = replay.settle((), ())

        await record_greens(registry, record, flake_suppressed=False, evidence=_evidence(99))

        assert (await recorded_entries(registry, key))['module:alpha:lint'].evidence_event_id == 5

    @pytest.mark.asyncio
    @pytest.mark.parametrize('passed', [True, False])
    async def test_the_tree_is_recorded_only_over_a_passing_verdict(self, registry, passed):
        replay, ran, legs = _all_admitted()
        _units, record = replay.settle(ran, legs)

        await record_greens(registry, record, flake_suppressed=False, evidence=_evidence(passed=passed))

        assert replay.tree.key is not None
        recorded = await recorded_entries(registry, replay.tree.key, *_keys_of(replay))
        assert set(recorded) == ({'tree'} if passed else set()) | {'module:alpha:test', 'module:alpha:lint'}

    @pytest.mark.asyncio
    async def test_the_tree_is_not_recorded_beside_a_refused_unit(self, registry):
        replay, ran, legs = _all_admitted()
        refused = (legs[0], LegRun(legs[1].unit_id, clean_leg(Leg.LINT, timed_out=True)))
        _units, record = replay.settle(ran, refused)

        await record_greens(registry, record, flake_suppressed=False, evidence=_evidence())

        assert replay.tree.key is not None
        assert 'tree' not in await recorded_entries(registry, replay.tree.key)

    @pytest.mark.asyncio
    async def test_no_evidence_records_nothing_and_says_why(self, registry):
        replay, ran, legs = _all_admitted()
        _units, record = replay.settle(ran, legs)

        outcome = await record_greens(registry, record, flake_suppressed=False, evidence=None)

        assert outcome.recorded == ()
        assert 'evidence' in outcome.reason
        assert await recorded_entries(registry, *_keys_of(replay)) == {}

    @pytest.mark.asyncio
    async def test_an_unreadable_registry_is_an_outcome_not_an_exception(self, tmp_path):
        db_path = tmp_path / 'garbage.db'
        db_path.write_bytes(b'garbage, not sqlite' * 64)
        replay, ran, legs = _all_admitted()
        _units, record = replay.settle(ran, legs)

        outcome = await record_greens(Registry(db_path), record, flake_suppressed=False, evidence=_evidence())

        assert outcome.recorded == ()
        assert str(db_path) in outcome.reason


def _keys_of(replay: ReplayPlan) -> list[UnitKey]:
    return [decided.key for decided in replay.modules.values() if decided.key is not None]


class TestConsumeReplayed:
    @pytest.mark.asyncio
    async def test_the_live_entries_of_every_hit_are_consumed_for_the_landing(self, registry):
        keys_ = {unit_id: unit_key(unit_id) for unit_id in ('module:alpha:test', 'module:alpha:lint')}
        entries = {unit_id: await record_entry(registry, key) for unit_id, key in keys_.items()}
        replay = merge_replay({
            unit_id: unit_decision(Decision.HIT, entries[unit_id], key=key) for unit_id, key in keys_.items()
        })
        _units, record = replay.settle((), ())

        outcome = await consume_replayed(registry, record, landing_sha='land-1')

        assert outcome.backs_every_hit
        assert set(await registry.consumed_for('land-1')) == set(entries.values())

    @pytest.mark.asyncio
    async def test_an_evicted_hit_is_unbacked_and_nothing_is_consumed(self, registry):
        keys_ = {unit_id: unit_key(unit_id) for unit_id in ('module:alpha:test', 'module:alpha:lint')}
        entries = {unit_id: await record_entry(registry, key) for unit_id, key in keys_.items()}
        replay = merge_replay({
            unit_id: unit_decision(Decision.HIT, entries[unit_id], key=key) for unit_id, key in keys_.items()
        })
        _units, record = replay.settle((), ())
        await registry.evict_key(keys_['module:alpha:lint'].key, 'test eviction')

        outcome = await consume_replayed(registry, record, landing_sha='land-1')

        assert not outcome.backs_every_hit
        assert outcome.unbacked == ('module:alpha:lint',)
        assert await registry.consumed_for('land-1') == ()

    @pytest.mark.asyncio
    async def test_an_unreadable_registry_consumes_nothing(self, registry, tmp_path):
        key = unit_key('module:alpha:test')
        entry = await record_entry(registry, key)
        _units, record = merge_replay({'module:alpha:test': unit_decision(Decision.HIT, entry, key=key)}).settle((), ())
        db_path = tmp_path / 'garbage.db'
        db_path.write_bytes(b'garbage, not sqlite' * 64)

        outcome = await consume_replayed(Registry(db_path), record, landing_sha='land-1')

        assert not outcome.backs_every_hit
        assert str(db_path) in outcome.reason

    @pytest.mark.asyncio
    async def test_a_record_without_a_hit_is_a_no_op(self, tmp_path):
        replay, ran, legs = _all_admitted()
        _units, record = replay.settle(ran, legs)
        db_path = tmp_path / 'untouched.db'

        outcome = await consume_replayed(Registry(db_path), record, landing_sha='land-1')

        assert outcome.backs_every_hit
        assert outcome.consumed == ()
        assert not db_path.exists()
