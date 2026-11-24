"""Recording a verify's admitted greens, and consuming the entries it replayed (task 5653)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from _verify_cache_units import (
    MERGE_LOG,
    MERGE_ON,
    clean_leg,
    merge_replay,
    ran_unit,
    record_entry,
    recorded_entries,
    unit_decision,
    unit_key,
    verify_result,
)

from orchestrator.verify_cache.keys import UnitKey
from orchestrator.verify_cache.policy import Admission, Decision
from orchestrator.verify_cache.recorder import (
    CacheRecord,
    CacheReport,
    LegRun,
    ReportedUnit,
    VerifyEvidenceRef,
    consume_replayed,
    record_greens,
    report_of,
)
from orchestrator.verify_cache.registry import Registry
from orchestrator.verify_cache.replay import ReplayPlan
from orchestrator.verify_evidence import Leg, UnitOutcome, VerifyUnit
from orchestrator.verify_runner import (
    result_from_dict,
    result_from_json,
    result_to_dict,
    result_to_json,
)


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


_RAN, _REPLAYED, _REFUSED = 'module:alpha:test', 'module:alpha:lint', 'module:beta:test'


async def _runner_record(registry: Registry, *, refused_leg: bool = False) -> tuple[CacheRecord, dict[str, UnitKey]]:
    """A settled record as a runner leaves it: alpha's test ran clean, alpha's lint
    replayed a recorded green, and beta's test ran (timed out when *refused_leg*)."""
    keys_ = {unit_id: unit_key(unit_id) for unit_id in (_RAN, _REPLAYED, _REFUSED)}
    replayed = (await record_entry(registry, keys_[_REPLAYED], evidence=5)).candidate
    replay = merge_replay({
        _RAN: unit_decision(Decision.MISS, key=keys_[_RAN]),
        _REPLAYED: unit_decision(Decision.HIT, replayed, key=keys_[_REPLAYED]),
        _REFUSED: unit_decision(Decision.MISS, key=keys_[_REFUSED]),
    })
    legs = (
        LegRun(_RAN, clean_leg(Leg.TEST)),
        LegRun(_REFUSED, clean_leg(Leg.TEST, timed_out=refused_leg)),
    )
    _units, record = replay.settle((ran_unit(_RAN), ran_unit(_REFUSED)), legs)
    return record, keys_


def _reported(report: CacheReport) -> dict[str, ReportedUnit]:
    return {reported.unit.unit: reported for reported in report.units}


class TestReportOf:
    @pytest.mark.asyncio
    async def test_each_unit_and_the_tree_carry_their_key_admission_and_cited_entry(self, registry):
        record, keys_ = await _runner_record(registry)

        report = report_of(record, flake_suppressed=False)

        units = _reported(report)
        assert set(units) == {_RAN, _REPLAYED, _REFUSED}
        assert units[_RAN].key == keys_[_RAN]
        assert units[_RAN].admission is not None and units[_RAN].admission.admitted
        assert units[_RAN].entry is None
        assert units[_REPLAYED].unit.outcome is UnitOutcome.CACHE_HIT
        assert units[_REPLAYED].admission == Admission(False, 'a replay is not new evidence')
        assert units[_REPLAYED].entry is not None and units[_REPLAYED].entry.evidence_event_id == 5
        assert report.tree is not None and record.tree is not None
        assert (report.tree.unit, report.tree.key) == (record.tree.unit, record.tree.key)

    @pytest.mark.asyncio
    async def test_a_would_hit_unit_cites_the_green_it_would_have_replayed(self, registry):
        key = unit_key(_RAN)
        candidate = (await record_entry(registry, key, evidence=6)).candidate
        replay = merge_replay({_RAN: unit_decision(Decision.WOULD_HIT, candidate, key=key)}, modes=MERGE_LOG)
        _units, record = replay.settle((ran_unit(_RAN),), (LegRun(_RAN, clean_leg(Leg.TEST)),))

        reported = _reported(report_of(record, flake_suppressed=False))[_RAN]

        assert reported.unit.outcome is UnitOutcome.WOULD_HIT
        assert reported.entry == candidate
        assert reported.admission is not None and reported.admission.admitted

    @pytest.mark.asyncio
    async def test_a_suppressed_flake_refuses_every_leg(self, registry):
        record, _keys = await _runner_record(registry)

        report = report_of(record, flake_suppressed=True)

        assert not any(reported.admission is not None and reported.admission.admitted for reported in report.units)


class TestReportedRecord:
    @pytest.mark.asyncio
    async def test_the_rebuilt_record_carries_the_reported_admissions(self, registry):
        record, _keys = await _runner_record(registry)
        report = report_of(record, flake_suppressed=False)

        rebuilt = report.record('merge', MERGE_ON)

        assert (rebuilt.role, rebuilt.modes, rebuilt.legs) == ('merge', MERGE_ON, ())
        assert [unit.admission for unit in rebuilt.units] == [reported.admission for reported in report.units]
        assert rebuilt.tree is not None and report.tree is not None
        assert rebuilt.tree.unit == report.tree.unit

    @pytest.mark.asyncio
    async def test_the_dispatcher_records_exactly_what_the_runner_admitted(self, tmp_path):
        runner_registry = Registry(tmp_path / 'runner-view.db')
        dispatcher = Registry(tmp_path / 'verify_cache.db')
        record, keys_ = await _runner_record(runner_registry, refused_leg=True)
        rebuilt = report_of(record, flake_suppressed=False).record('merge', MERGE_ON)

        await record_greens(dispatcher, rebuilt, flake_suppressed=True, evidence=_evidence(21))

        recorded = await recorded_entries(dispatcher, *keys_.values())
        assert set(recorded) == {_RAN}
        assert recorded[_RAN].evidence_event_id == 21
        assert rebuilt.tree is not None and rebuilt.tree.key is not None
        assert await recorded_entries(dispatcher, rebuilt.tree.key) == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize('passed', [True, False])
    async def test_the_tree_is_recorded_over_a_passing_verdict_of_admitted_or_replayed_units(self, tmp_path, passed):
        dispatcher = Registry(tmp_path / 'verify_cache.db')
        record, _keys = await _runner_record(Registry(tmp_path / 'runner-view.db'))
        rebuilt = report_of(record, flake_suppressed=False).record('merge', MERGE_ON)

        await record_greens(dispatcher, rebuilt, flake_suppressed=False, evidence=_evidence(passed=passed))

        assert rebuilt.tree is not None and rebuilt.tree.key is not None
        assert set(await recorded_entries(dispatcher, rebuilt.tree.key)) == ({'tree'} if passed else set())


class TestReportOnTheWire:
    @pytest.mark.asyncio
    async def test_a_result_carries_its_report_across_the_wire(self, registry):
        record, _keys = await _runner_record(registry)
        report = report_of(record, flake_suppressed=False)

        decoded = result_from_json(result_to_json(verify_result(cache_report=report)))

        assert decoded.cache_report == report

    @pytest.mark.asyncio
    @pytest.mark.parametrize('damage', ['list', 'missing-key', 'bad-tier'])
    async def test_a_malformed_report_decodes_to_none_loudly(self, registry, caplog, damage):
        record, _keys = await _runner_record(registry)
        wire = json.loads(result_to_json(verify_result(cache_report=report_of(record, flake_suppressed=False))))
        damaged = {
            'list': lambda raw: [raw],
            'missing-key': lambda raw: {'units': raw['units']},
            'bad-tier': lambda raw: {**raw, 'tree': {**raw['tree'], 'unit': {**raw['tree']['unit'], 'tier': 'galaxy'}}},
        }[damage](wire['cache_report'])

        with caplog.at_level(logging.WARNING):
            decoded = result_from_dict({**wire, 'cache_report': damaged})

        assert decoded.cache_report is None
        assert any('cache_report' in entry.getMessage() for entry in caplog.records)

    @pytest.mark.asyncio
    async def test_equality_ignores_the_report(self, registry):
        record, _keys = await _runner_record(registry)

        assert verify_result(cache_report=report_of(record, flake_suppressed=False)) == verify_result()
        assert 'cache_report' in result_to_dict(verify_result())
