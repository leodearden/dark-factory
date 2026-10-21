"""Write triage's attach guard (task 7158): two rules that refuse an attach and store the write standalone.

The FLOOR refuses any attach whose target's cosine to the write is below
``write_triage.min_parent_cosine``; the HOIST GAP refuses, in the judge band
only, a hoist from a matched child to its parent when the parent's cosine is
more than ``write_triage.hoist_max_cosine_gap`` below the child's. Both are
hand-set, hot-reloadable operator knobs; ``None`` switches a rule off. The
evidence for the shipped values lives in the gitignored soak directory of the
main checkout (``data/write-triage-soak-2026-10-19/``).
"""

from __future__ import annotations

import itertools
import logging
import types
from enum import StrEnum
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import yaml
from _fm_helpers import load_script_module
from pydantic import ValidationError

from fused_memory.config.reload import RELOADABLE_FIELDS, apply_reload
from fused_memory.config.schema import FusedMemoryConfig, WriteTriageConfig
from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server import tools, write_triage
from fused_memory.server.grouped_read import AMENDMENT_KIND, PARENT_ID_KEY, SIGHTING_KIND
from fused_memory.server.tools import create_mcp_server
from fused_memory.server.write_triage import (
    CANONICAL_ID_KEY,
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    ROUTED_KEY,
    AttachGuard,
    AttachRefusal,
    BandDecision,
    TriageFailOpenCounter,
    TriageJudgeVerdict,
    decide_band,
    resolve_attach_guard,
    resolve_hoist_max_cosine_gap,
    resolve_min_parent_cosine,
    triage_write,
)
from fused_memory.services.memory_service import RRF_K, SearchResults
from server.write_triage_harness import configure_triage_config, search_candidate

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_SHIPPED_CONFIG = _PACKAGE_ROOT / 'config' / 'config.yaml'
_CALIBRATION_SCRIPT = _PACKAGE_ROOT / 'scripts' / 'calibrate_write_triage.py'

_GUARD_KNOBS = ('min_parent_cosine', 'hoist_max_cosine_gap')

_PARENT = 'parent-P'
_WRITE = 'the write under test'
_PROJECT = 'proj-7158'
_TRIAGE_LOGGER = 'fused_memory.server.write_triage'
_EVERY_BAND_AND_HOIST = list(itertools.product([True, False], repeat=2))


@pytest.fixture(autouse=True)
def _pin_config_path(tmp_path, monkeypatch):
    """Pin CONFIG_PATH at a missing file so ``FusedMemoryConfig()`` is pure schema defaults."""
    monkeypatch.setenv('CONFIG_PATH', str(tmp_path / 'missing.yaml'))


def _service_with(**write_triage_leaves) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        config=types.SimpleNamespace(write_triage=types.SimpleNamespace(**write_triage_leaves)),
    )


def _result(id_: str, cosine: float, *, extra_metadata: dict | None = None) -> MemoryResult:
    """A post-RRF search hit: the Mem0 cosine sits in ``metadata['store_score']``."""
    return MemoryResult(
        id=id_,
        content=f'record {id_}',
        category=MemoryCategory.procedural_knowledge,
        source_store=SourceStore.mem0,
        relevance_score=1.0 / (RRF_K + 1),
        metadata={'store_rank': 1, 'store_score': cosine, **(extra_metadata or {})},
    )


def _child_of(parent_id: str, kind: str = SIGHTING_KIND) -> dict:
    return {'kind': kind, PARENT_ID_KEY: parent_id}


def _child(id_: str, cosine: float, *, kind: str = SIGHTING_KIND) -> MemoryResult:
    return _result(id_, cosine, extra_metadata=_child_of(_PARENT, kind))


def _svc(
    results: list[MemoryResult],
    *,
    min_parent_cosine: float | None = 0.60,
    hoist_max_cosine_gap: float | None = 0.05,
    t_high: float = 0.90,
    t_low: float = 0.50,
    parent_cosines: dict[str, float] | None = None,
) -> types.SimpleNamespace:
    """A memory service whose search returns *results* and whose by-id lookup returns *parent_cosines*."""
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            write_triage=types.SimpleNamespace(
                enabled=True,
                candidate_k=20,
                t_high=t_high,
                t_low=t_low,
                min_parent_cosine=min_parent_cosine,
                hoist_max_cosine_gap=hoist_max_cosine_gap,
            ),
        ),
        search=AsyncMock(return_value=SearchResults(results)),
        cosine_to_memories=AsyncMock(return_value=parent_cosines or {}),
    )


def _counter() -> TriageFailOpenCounter:
    return TriageFailOpenCounter(time_provider=lambda: 1000.0)


def _judge(outcome: str, candidate_id: str | None = None) -> AsyncMock:
    return AsyncMock(return_value=TriageJudgeVerdict(outcome, candidate_id))


async def _triage(service, counter=None, judge=None) -> BandDecision:
    return await triage_write(
        service,
        content=_WRITE,
        project_id=_PROJECT,
        counter=counter or _counter(),
        judge=judge or _judge(OUTCOME_STORED),
    )


def _refusal_lines(caplog) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records
        if record.name == _TRIAGE_LOGGER
        and record.getMessage().startswith('write_triage attach refused')
    ]


def _fail_open_lines(caplog) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records
        if record.name == _TRIAGE_LOGGER
        and record.getMessage().startswith('write_triage fail-open')
    ]


class TestTheGuardKnobs:
    def test_the_schema_ships_the_soak_derived_defaults(self):
        config = WriteTriageConfig()

        assert config.min_parent_cosine == 0.60
        assert config.hoist_max_cosine_gap == 0.05

    @pytest.mark.parametrize('knob', _GUARD_KNOBS)
    def test_none_is_the_off_switch(self, knob):
        assert getattr(WriteTriageConfig.model_validate({knob: None}), knob) is None

    @pytest.mark.parametrize('knob', _GUARD_KNOBS)
    @pytest.mark.parametrize('value', [-0.1, 1.5])
    def test_a_value_outside_the_cosine_range_is_rejected(self, knob, value):
        with pytest.raises(ValidationError):
            WriteTriageConfig.model_validate({knob: value})

    @pytest.mark.parametrize('knob', _GUARD_KNOBS)
    def test_each_knob_is_green_tier(self, knob):
        assert f'write_triage.{knob}' in RELOADABLE_FIELDS

    @pytest.mark.parametrize(
        ('knob', 'resolver_name', 'new_value'),
        [
            ('min_parent_cosine', 'resolve_min_parent_cosine', 0.70),
            ('min_parent_cosine', 'resolve_min_parent_cosine', None),
            ('hoist_max_cosine_gap', 'resolve_hoist_max_cosine_gap', 0.10),
            ('hoist_max_cosine_gap', 'resolve_hoist_max_cosine_gap', None),
        ],
    )
    def test_an_applied_reload_is_observed_by_the_live_resolver(
        self, knob, resolver_name, new_value,
    ):
        resolver = getattr(write_triage, resolver_name)
        memory_service = types.SimpleNamespace(config=FusedMemoryConfig())
        assert resolver(memory_service) == getattr(WriteTriageConfig(), knob)

        fresh = FusedMemoryConfig()
        object.__setattr__(fresh.write_triage, knob, new_value)

        report = apply_reload(memory_service.config, fresh)

        assert f'write_triage.{knob}' in report['applied']
        assert resolver(memory_service) == new_value

    @pytest.mark.parametrize('resolver', [resolve_min_parent_cosine, resolve_hoist_max_cosine_gap])
    @pytest.mark.parametrize(
        'memory_service',
        [
            types.SimpleNamespace(config=types.SimpleNamespace()),
            _service_with(),
            _service_with(min_parent_cosine=True, hoist_max_cosine_gap=True),
            _service_with(min_parent_cosine='0.6', hoist_max_cosine_gap='0.05'),
            Mock(),
        ],
        ids=['no-section', 'no-leaf', 'bool', 'str', 'unspecced-mock'],
    )
    def test_an_unreadable_leaf_switches_the_rule_off(self, resolver, memory_service):
        assert resolver(memory_service) is None

    @pytest.mark.parametrize('resolver', [resolve_min_parent_cosine, resolve_hoist_max_cosine_gap])
    def test_an_int_reads_as_a_float(self, resolver):
        value = resolver(_service_with(min_parent_cosine=1, hoist_max_cosine_gap=1))

        assert value == 1.0
        assert isinstance(value, float)

    def test_the_shipped_config_sets_both_knobs_to_cosines(self):
        section = yaml.safe_load(_SHIPPED_CONFIG.read_text())['write_triage']

        for knob in _GUARD_KNOBS:
            value = section[knob]
            assert isinstance(value, int | float) and not isinstance(value, bool), knob
            assert 0.0 <= value <= 1.0, knob

    def test_a_recalibration_never_reverts_a_hand_set_knob(self):
        calibrate = load_script_module(_CALIBRATION_SCRIPT, mod_name='calibrate_write_triage')
        text = _SHIPPED_CONFIG.read_text()

        rewritten = calibrate.write_triage_config_block(
            text, t_high=0.91, t_low=0.51, report_path='calibration/x.json',
        )

        before = yaml.safe_load(text)['write_triage']
        after = yaml.safe_load(rewritten)['write_triage']
        for knob in _GUARD_KNOBS:
            assert after[knob] == before[knob], knob


class TestAttachGuardRules:
    """Binary-exact cosines throughout, so float rounding cannot decide a boundary."""

    @pytest.mark.parametrize(('hoisted', 'judged'), _EVERY_BAND_AND_HOIST)
    def test_the_floor_refuses_a_target_below_it_in_every_band(self, hoisted, judged):
        guard = AttachGuard(min_parent_cosine=0.625, hoist_max_cosine_gap=None)

        refusals = guard.refusals(
            source_cosine=0.875, target_cosine=0.5, hoisted=hoisted, judged=judged,
        )

        assert refusals == {AttachRefusal.BELOW_MIN_PARENT_COSINE}

    @pytest.mark.parametrize(('hoisted', 'judged'), _EVERY_BAND_AND_HOIST)
    def test_a_target_at_the_floor_clears_it(self, hoisted, judged):
        guard = AttachGuard(min_parent_cosine=0.625, hoist_max_cosine_gap=None)

        refusals = guard.refusals(
            source_cosine=0.875, target_cosine=0.625, hoisted=hoisted, judged=judged,
        )

        assert refusals == frozenset()

    def test_a_judged_hoist_more_than_the_gap_below_the_child_is_refused(self):
        guard = AttachGuard(min_parent_cosine=None, hoist_max_cosine_gap=0.125)

        refusals = guard.refusals(source_cosine=0.875, target_cosine=0.625, hoisted=True, judged=True)

        assert refusals == {AttachRefusal.HOIST_COSINE_GAP}

    def test_a_judged_hoist_exactly_the_gap_below_the_child_attaches(self):
        guard = AttachGuard(min_parent_cosine=None, hoist_max_cosine_gap=0.125)

        refusals = guard.refusals(source_cosine=0.875, target_cosine=0.75, hoisted=True, judged=True)

        assert refusals == frozenset()

    def test_the_gap_never_applies_outside_the_judge_band(self):
        guard = AttachGuard(min_parent_cosine=None, hoist_max_cosine_gap=0.125)

        assert not guard.needs_target_cosine(hoisted=True, judged=False)
        assert guard.refusals(
            source_cosine=0.875, target_cosine=0.25, hoisted=True, judged=False,
        ) == frozenset()

    def test_the_gap_never_applies_without_a_hoist(self):
        guard = AttachGuard(min_parent_cosine=None, hoist_max_cosine_gap=0.125)

        assert not guard.needs_target_cosine(hoisted=False, judged=True)
        assert guard.refusals(
            source_cosine=0.875, target_cosine=0.25, hoisted=False, judged=True,
        ) == frozenset()

    def test_every_tripped_rule_is_recorded(self):
        guard = AttachGuard(min_parent_cosine=0.625, hoist_max_cosine_gap=0.125)

        refusals = guard.refusals(source_cosine=0.875, target_cosine=0.5, hoisted=True, judged=True)

        assert refusals == {AttachRefusal.BELOW_MIN_PARENT_COSINE, AttachRefusal.HOIST_COSINE_GAP}

    @pytest.mark.parametrize(('hoisted', 'judged'), _EVERY_BAND_AND_HOIST)
    def test_a_none_floor_is_off(self, hoisted, judged):
        guard = AttachGuard(min_parent_cosine=None, hoist_max_cosine_gap=0.125)

        refusals = guard.refusals(
            source_cosine=0.25, target_cosine=0.25, hoisted=hoisted, judged=judged,
        )

        assert refusals == frozenset()

    def test_a_none_gap_is_off(self):
        guard = AttachGuard(min_parent_cosine=0.125, hoist_max_cosine_gap=None)

        refusals = guard.refusals(source_cosine=0.875, target_cosine=0.25, hoisted=True, judged=True)

        assert refusals == frozenset()

    @pytest.mark.parametrize(('hoisted', 'judged'), _EVERY_BAND_AND_HOIST)
    def test_with_both_rules_off_no_target_cosine_is_needed(self, hoisted, judged):
        guard = AttachGuard(min_parent_cosine=None, hoist_max_cosine_gap=None)

        assert not guard.needs_target_cosine(hoisted=hoisted, judged=judged)
        assert guard.refusals(
            source_cosine=None, target_cosine=None, hoisted=hoisted, judged=judged,
        ) == frozenset()

    @pytest.mark.parametrize(('hoisted', 'judged'), _EVERY_BAND_AND_HOIST)
    def test_the_floor_always_needs_the_target_cosine(self, hoisted, judged):
        guard = AttachGuard(min_parent_cosine=0.625, hoist_max_cosine_gap=None)

        assert guard.needs_target_cosine(hoisted=hoisted, judged=judged)

    @pytest.mark.parametrize(('hoisted', 'judged'), _EVERY_BAND_AND_HOIST)
    def test_the_gap_alone_needs_it_only_for_a_judged_hoist(self, hoisted, judged):
        guard = AttachGuard(min_parent_cosine=None, hoist_max_cosine_gap=0.125)

        assert guard.needs_target_cosine(hoisted=hoisted, judged=judged) == (hoisted and judged)

    @pytest.mark.parametrize(
        'guard',
        [
            AttachGuard(min_parent_cosine=0.625, hoist_max_cosine_gap=None),
            AttachGuard(min_parent_cosine=None, hoist_max_cosine_gap=0.125),
        ],
        ids=['floor', 'gap'],
    )
    def test_an_unscorable_target_is_refused_while_a_rule_is_in_force(self, guard):
        refusals = guard.refusals(source_cosine=0.875, target_cosine=None, hoisted=True, judged=True)

        assert refusals == {AttachRefusal.TARGET_COSINE_UNAVAILABLE}

    def test_the_guard_is_read_from_the_live_config(self):
        guard = resolve_attach_guard(_service_with(min_parent_cosine=0.6, hoist_max_cosine_gap=0.05))

        assert guard == AttachGuard(min_parent_cosine=0.6, hoist_max_cosine_gap=0.05)

    def test_an_unconfigured_guard_is_off(self):
        guard = resolve_attach_guard(_service_with())

        assert guard == AttachGuard(min_parent_cosine=None, hoist_max_cosine_gap=None)

    def test_the_refusal_vocabulary(self):
        assert issubclass(AttachRefusal, StrEnum)
        assert {reason.value for reason in AttachRefusal} == {
            'below_min_parent_cosine', 'hoist_cosine_gap', 'target_cosine_unavailable',
        }


class TestTheDecisionNamesItsSource:
    def test_a_restated_decision_names_its_winner(self):
        decision = decide_band([_result('w', 0.95), _result('x', 0.6)], t_high=0.9, t_low=0.5)

        assert decision.outcome == OUTCOME_RESTATED
        assert (decision.winner_id, decision.canonical_id) == ('w', 'w')

    @pytest.mark.parametrize(
        ('cosine', 'outcome'), [(0.95, OUTCOME_RESTATED), (0.7, OUTCOME_JUDGE)],
    )
    def test_a_hoisted_winner_is_named_as_the_child(self, cosine, outcome):
        child = _result('c', cosine, extra_metadata=_child_of(_PARENT, AMENDMENT_KIND))

        decision = decide_band([child], t_high=0.9, t_low=0.5)

        assert decision.outcome == outcome
        assert (decision.winner_id, decision.canonical_id) == ('c', _PARENT)

    def test_a_stored_decision_names_no_winner(self):
        decision = decide_band([_result('w', 0.4)], t_high=0.9, t_low=0.5)

        assert decision.outcome == OUTCOME_STORED
        assert decision.winner_id is None

    def test_the_attach_source_is_the_judged_candidate_when_there_is_one(self):
        decision = BandDecision(
            OUTCOME_AMENDED, _PARENT, 0.7, 0.9, 0.5, judged_candidate_id='j', winner_id='w',
        )

        assert decision.attach_source_id == 'j'

    def test_the_attach_source_is_the_winner_otherwise(self):
        decision = BandDecision(OUTCOME_AMENDED, _PARENT, 0.7, 0.9, 0.5, winner_id='w')

        assert decision.attach_source_id == 'w'

    def test_a_fresh_decision_has_no_refusals(self):
        decision = BandDecision(OUTCOME_STORED, None, None, 0.9, 0.5)

        assert decision.attach_refusals == frozenset()


class TestTheHoistGuardInTheJudgeBand:
    @pytest.mark.asyncio
    async def test_a_parent_far_below_the_named_child_is_refused(self):
        service = _svc([_child('c', 0.80), _result(_PARENT, 0.70)])
        counter = _counter()

        decision = await _triage(service, counter, _judge(OUTCOME_AMENDED, 'c'))

        assert (decision.outcome, decision.canonical_id) == (OUTCOME_STORED, None)
        assert decision.attach_refusals == {AttachRefusal.HOIST_COSINE_GAP}
        assert counter.live_count() == 0
        service.cosine_to_memories.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_parent_within_the_gap_is_attached(self):
        service = _svc([_child('c', 0.80), _result(_PARENT, 0.78)])

        decision = await _triage(service, judge=_judge(OUTCOME_AMENDED, 'c'))

        assert (decision.outcome, decision.canonical_id) == (OUTCOME_AMENDED, _PARENT)
        assert decision.attach_refusals == frozenset()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('parent_cosine', 'outcome', 'canonical_id', 'refusals'),
        [
            (0.70, OUTCOME_STORED, None, {AttachRefusal.HOIST_COSINE_GAP}),
            (0.79, OUTCOME_AMENDED, _PARENT, frozenset()),
        ],
    )
    async def test_an_unretrieved_parent_is_scored_by_one_lookup(
        self, parent_cosine, outcome, canonical_id, refusals,
    ):
        service = _svc([_child('c', 0.80)], parent_cosines={_PARENT: parent_cosine})

        decision = await _triage(service, judge=_judge(OUTCOME_AMENDED, 'c'))

        service.cosine_to_memories.assert_awaited_once_with(
            query=_WRITE, memory_ids=[_PARENT], project_id=_PROJECT,
        )
        assert (decision.outcome, decision.canonical_id) == (outcome, canonical_id)
        assert decision.attach_refusals == refusals

    @pytest.mark.asyncio
    async def test_a_bare_word_verdict_measures_the_gap_from_the_band_winner(self):
        service = _svc([_child('c', 0.80), _result(_PARENT, 0.70)])

        decision = await _triage(service, judge=AsyncMock(return_value=OUTCOME_AMENDED))

        assert (decision.outcome, decision.canonical_id) == (OUTCOME_STORED, None)
        assert decision.attach_refusals == {AttachRefusal.HOIST_COSINE_GAP}

    @pytest.mark.asyncio
    async def test_a_named_record_that_is_not_a_child_needs_no_lookup(self):
        service = _svc([_result('w', 0.80), _result('x', 0.65)])

        decision = await _triage(service, judge=_judge(OUTCOME_CONTESTED, 'x'))

        assert (decision.outcome, decision.canonical_id) == (OUTCOME_CONTESTED, 'x')
        service.cosine_to_memories.assert_not_awaited()


class TestTheFloorOnEveryAttach:
    @pytest.mark.asyncio
    async def test_a_restated_parent_below_the_floor_is_refused(self):
        service = _svc([_child('c', 0.95), _result(_PARENT, 0.55)])
        judge = _judge(OUTCOME_AMENDED, 'c')

        decision = await _triage(service, judge=judge)

        assert (decision.outcome, decision.canonical_id) == (OUTCOME_STORED, None)
        assert decision.attach_refusals == {AttachRefusal.BELOW_MIN_PARENT_COSINE}
        judge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_gap_does_not_apply_to_a_restated_hoist(self):
        service = _svc([_child('c', 0.95), _result(_PARENT, 0.70)])

        decision = await _triage(service)

        assert (decision.outcome, decision.canonical_id) == (OUTCOME_RESTATED, _PARENT)
        assert decision.attach_refusals == frozenset()

    @pytest.mark.asyncio
    async def test_a_judged_record_below_the_floor_is_refused(self):
        service = _svc([_result('w', 0.55)])

        decision = await _triage(service, judge=_judge(OUTCOME_AMENDED, 'w'))

        assert (decision.outcome, decision.canonical_id) == (OUTCOME_STORED, None)
        assert decision.attach_refusals == {AttachRefusal.BELOW_MIN_PARENT_COSINE}

    @pytest.mark.asyncio
    async def test_a_hoist_tripping_both_rules_records_both(self):
        service = _svc([_child('c', 0.80), _result(_PARENT, 0.55)])

        decision = await _triage(service, judge=_judge(OUTCOME_AMENDED, 'c'))

        assert decision.attach_refusals == {
            AttachRefusal.BELOW_MIN_PARENT_COSINE, AttachRefusal.HOIST_COSINE_GAP,
        }


class TestRefusalsArePlainStoredNotFailOpens:
    @pytest.mark.asyncio
    async def test_a_parent_missing_from_the_store_is_refused_uncounted(self):
        service = _svc([_child('c', 0.95)], parent_cosines={})
        counter = _counter()

        decision = await _triage(service, counter)

        assert (decision.outcome, decision.canonical_id) == (OUTCOME_STORED, None)
        assert decision.attach_refusals == {AttachRefusal.TARGET_COSINE_UNAVAILABLE}
        assert counter.live_count() == 0

    @pytest.mark.asyncio
    async def test_a_refusal_is_one_info_line_and_no_fail_open_line(self, caplog):
        service = _svc([_child('c', 0.80), _result(_PARENT, 0.55)])
        caplog.set_level(logging.INFO, logger=_TRIAGE_LOGGER)

        await _triage(service, judge=_judge(OUTCOME_AMENDED, 'c'))

        [line] = _refusal_lines(caplog)
        assert line.levelno == logging.INFO
        message = line.getMessage()
        for fragment in (
            AttachRefusal.BELOW_MIN_PARENT_COSINE.value,
            AttachRefusal.HOIST_COSINE_GAP.value,
            _PARENT,
            '0.55',
            '0.8',
        ):
            assert fragment in message, (fragment, message)
        assert _fail_open_lines(caplog) == []

    @pytest.mark.asyncio
    async def test_a_lookup_that_raises_is_a_counted_fail_open(self, caplog):
        service = _svc([_child('c', 0.95)])
        service.cosine_to_memories.side_effect = TimeoutError('Mem0 cosine_to_points blew 5s')
        counter = _counter()
        caplog.set_level(logging.INFO, logger=_TRIAGE_LOGGER)

        decision = await _triage(service, counter)

        assert (decision.outcome, decision.canonical_id) == (OUTCOME_STORED, None)
        assert decision.attach_refusals == frozenset()
        assert counter.live_count() == 1
        [line] = _fail_open_lines(caplog)
        assert line.levelno == logging.WARNING
        assert 'stage=attach_guard' in line.getMessage()

    @pytest.mark.asyncio
    async def test_with_both_rules_off_a_hoist_attaches_as_before(self):
        service = _svc([_child('c', 0.95)], min_parent_cosine=None, hoist_max_cosine_gap=None)

        decision = await _triage(service)

        assert (decision.outcome, decision.canonical_id) == (OUTCOME_RESTATED, _PARENT)
        service.cosine_to_memories.assert_not_awaited()


class TestTheToolAcksARefusalAsStored:
    @pytest.mark.asyncio
    async def test_a_refused_restatement_is_acked_and_persisted_as_a_plain_store(self, monkeypatch):
        counter = TriageFailOpenCounter(time_provider=lambda: 1000.0)
        monkeypatch.setattr(tools, 'TriageFailOpenCounter', lambda: counter)
        mock_service = AsyncMock()
        configure_triage_config(mock_service, min_parent_cosine=0.60, hoist_max_cosine_gap=0.05)
        stored = MagicMock()
        stored.model_dump.return_value = {
            'id': 'new-id', 'category': 'procedural_knowledge', 'stored_in': ['mem0'],
        }
        mock_service.add_memory.return_value = stored
        mock_service.search.return_value = [
            search_candidate('m1', 0.55, store_rank=2),
            search_candidate('m2', 0.97, extra_metadata={PARENT_ID_KEY: 'm1', 'kind': SIGHTING_KIND}),
        ]
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool('add_memory', {
            'content': _WRITE,
            'category': 'procedural_knowledge',
            'agent_id': 'claude-interactive',
            'project_id': _PROJECT,
        })

        assert result[ROUTED_KEY] == OUTCOME_STORED, result
        assert CANONICAL_ID_KEY not in result, result
        assert mock_service.add_memory.await_count == 1
        metadata = mock_service.add_memory.await_args.kwargs.get('metadata') or {}
        assert PARENT_ID_KEY not in metadata and 'kind' not in metadata, metadata
        assert counter.live_count() == 0
