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
import types
from enum import StrEnum
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from _fm_helpers import load_script_module
from pydantic import ValidationError

from fused_memory.config.reload import RELOADABLE_FIELDS, apply_reload
from fused_memory.config.schema import FusedMemoryConfig, WriteTriageConfig
from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server import write_triage
from fused_memory.server.grouped_read import AMENDMENT_KIND, PARENT_ID_KEY, SIGHTING_KIND
from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    AttachGuard,
    AttachRefusal,
    BandDecision,
    decide_band,
    resolve_attach_guard,
    resolve_hoist_max_cosine_gap,
    resolve_min_parent_cosine,
)
from fused_memory.services.memory_service import RRF_K

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_SHIPPED_CONFIG = _PACKAGE_ROOT / 'config' / 'config.yaml'
_CALIBRATION_SCRIPT = _PACKAGE_ROOT / 'scripts' / 'calibrate_write_triage.py'

_GUARD_KNOBS = ('min_parent_cosine', 'hoist_max_cosine_gap')

_PARENT = 'parent-P'
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


class TestTheGuardKnobs:
    def test_the_schema_ships_the_soak_derived_defaults(self):
        config = WriteTriageConfig()

        assert config.min_parent_cosine == 0.60
        assert config.hoist_max_cosine_gap == 0.05

    @pytest.mark.parametrize('knob', _GUARD_KNOBS)
    def test_none_is_the_off_switch(self, knob):
        assert getattr(WriteTriageConfig(**{knob: None}), knob) is None

    @pytest.mark.parametrize('knob', _GUARD_KNOBS)
    @pytest.mark.parametrize('value', [-0.1, 1.5])
    def test_a_value_outside_the_cosine_range_is_rejected(self, knob, value):
        with pytest.raises(ValidationError):
            WriteTriageConfig(**{knob: value})

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
