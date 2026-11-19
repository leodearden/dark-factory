"""Tests for the shared per-cycle flood-control policy (task 5240, PRD C4/D6).

``reconciliation/consolidation_flood_control.py::rank_and_cap`` is the one home
of the ranking and backlog policy used by the auto-consolidation executor and by
task 4916's detector-report submitter. Every test here runs ``rank_and_cap``
against a real ``ConsolidationAutoConfig`` and asserts on the returned plan.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from fused_memory.config.schema import ConsolidationAutoConfig
from fused_memory.reconciliation.consolidation_flood_control import (
    FloodCandidate,
    FloodPlan,
    rank_and_cap,
)

PROC = 'procedural_knowledge'
OBS = 'observations_and_summaries'


def _cand(key, size, category=PROC, canonical_absent=False):
    return FloodCandidate(
        key=key, size=size, category=category, canonical_absent=canonical_absent,
    )


def _keys(candidates):
    return tuple(c.key for c in candidates)


class TestScore:
    def test_canonical_absent_bonus_outranks_a_larger_consolidated_cluster(self):
        small_absent = _cand('small', 4, canonical_absent=True)
        large_present = _cand('large', 6)
        plan = rank_and_cap(
            [large_present, small_absent],
            cap=2,
            config=ConsolidationAutoConfig(canonical_absent_bonus=2.0),
        )
        assert _keys(plan.selected) == ('small', 'large')

    def test_order_flips_when_the_bonus_is_neutral(self):
        small_absent = _cand('small', 4, canonical_absent=True)
        large_present = _cand('large', 6)
        plan = rank_and_cap(
            [small_absent, large_present],
            cap=2,
            config=ConsolidationAutoConfig(canonical_absent_bonus=1.0),
        )
        assert _keys(plan.selected) == ('large', 'small')

    def test_observations_weight_down_ranks_an_equal_size_cluster(self):
        obs = _cand('obs', 5, category=OBS)
        proc = _cand('proc', 5, category=PROC)
        plan = rank_and_cap([obs, proc], cap=2, config=ConsolidationAutoConfig())
        assert _keys(plan.selected) == ('proc', 'obs')

    def test_ties_keep_input_order(self):
        cands = [_cand(k, 5) for k in ('a', 'b', 'c', 'd')]
        plan = rank_and_cap(cands, cap=4, config=ConsolidationAutoConfig())
        assert _keys(plan.selected) == ('a', 'b', 'c', 'd')

    def test_live_config_weights_change_the_ranking(self):
        obs = _cand('obs', 5, category=OBS)
        proc = _cand('proc', 5, category=PROC)
        default = rank_and_cap([obs, proc], cap=2, config=ConsolidationAutoConfig())
        flipped = rank_and_cap(
            [obs, proc],
            cap=2,
            config=ConsolidationAutoConfig(
                category_weights={PROC: 0.5, OBS: 1.0},
            ),
        )
        assert _keys(default.selected) == ('proc', 'obs')
        assert _keys(flipped.selected) == ('obs', 'proc')


class TestPartition:
    def test_selected_is_the_top_cap_and_deferred_the_ranked_remainder(self):
        cands = [_cand(f'k{n}', n) for n in range(1, 6)]
        plan = rank_and_cap(cands, cap=2, config=ConsolidationAutoConfig())
        assert _keys(plan.selected) == ('k5', 'k4')
        assert _keys(plan.deferred) == ('k3', 'k2', 'k1')
        assert plan.excluded == ()
        assert plan.cap == 2

    def test_nothing_is_dropped(self):
        cands = [
            _cand('a', 3),
            _cand('b', 7, category='temporal_facts'),
            _cand('c', 2, category=OBS),
            _cand('d', 9),
        ]
        plan = rank_and_cap(cands, cap=1, config=ConsolidationAutoConfig())
        assert len(plan.selected) + len(plan.deferred) + len(plan.excluded) == len(cands)

    def test_category_absent_from_weights_is_excluded_not_ranked(self):
        graphiti = _cand('g', 50, category='temporal_facts')
        proc = _cand('p', 2)
        plan = rank_and_cap([graphiti, proc], cap=5, config=ConsolidationAutoConfig())
        assert _keys(plan.excluded) == ('g',)
        assert _keys(plan.selected) == ('p',)
        assert plan.deferred == ()

    def test_plan_fields_are_tuples(self):
        plan = rank_and_cap([_cand('a', 2)], cap=1, config=ConsolidationAutoConfig())
        assert isinstance(plan, FloodPlan)
        assert isinstance(plan.selected, tuple)
        assert isinstance(plan.deferred, tuple)
        assert isinstance(plan.excluded, tuple)


class TestBacklog:
    def test_exactly_the_threshold_does_not_trip(self):
        config = ConsolidationAutoConfig(backlog_multiplier=2)
        cands = [_cand(f'k{n}', 2) for n in range(6)]
        plan = rank_and_cap(cands, cap=3, config=config)
        assert plan.backlog_threshold == 6
        assert plan.backlog_tripped is False
        assert len(plan.selected) == 3

    def test_one_above_the_threshold_trips_and_selects_nothing(self):
        config = ConsolidationAutoConfig(backlog_multiplier=2)
        cands = [_cand(f'k{n}', 2) for n in range(7)]
        plan = rank_and_cap(cands, cap=3, config=config)
        assert plan.backlog_tripped is True
        assert plan.selected == ()
        assert _keys(plan.deferred) == _keys(cands)

    def test_excluded_candidates_count_toward_the_trip(self):
        config = ConsolidationAutoConfig(backlog_multiplier=1)
        cands = [
            _cand('p1', 2),
            _cand('g1', 2, category='temporal_facts'),
            _cand('g2', 2, category='temporal_facts'),
        ]
        plan = rank_and_cap(cands, cap=2, config=config)
        assert plan.backlog_tripped is True
        assert plan.selected == ()
        assert _keys(plan.deferred) == ('p1',)
        assert _keys(plan.excluded) == ('g1', 'g2')

    def test_zero_cap_trips_on_any_candidate(self):
        plan = rank_and_cap([_cand('a', 2)], cap=0, config=ConsolidationAutoConfig())
        assert plan.backlog_threshold == 0
        assert plan.backlog_tripped is True
        assert plan.selected == ()

    def test_empty_input_never_trips(self):
        plan = rank_and_cap([], cap=0, config=ConsolidationAutoConfig())
        assert plan.backlog_tripped is False
        assert (plan.selected, plan.deferred, plan.excluded) == ((), (), ())


class TestValidation:
    def test_negative_cap_raises_naming_the_value(self):
        with pytest.raises(ValueError, match='-1'):
            rank_and_cap([_cand('a', 2)], cap=-1, config=ConsolidationAutoConfig())

    def test_negative_size_raises_naming_the_value(self):
        with pytest.raises(ValueError, match='-4'):
            rank_and_cap([_cand('a', -4)], cap=1, config=ConsolidationAutoConfig())


class TestFloodCandidateShape:
    def test_candidate_is_frozen(self):
        cand = _cand('a', 2)
        attribute = 'size'
        with pytest.raises(AttributeError):
            setattr(cand, attribute, 3)

    def test_key_may_be_any_hashable(self):
        cands = [_cand(('project', 'topic-a'), 3), _cand(('project', 'topic-b'), 4)]
        plan = rank_and_cap(cands, cap=2, config=ConsolidationAutoConfig())
        assert _keys(plan.selected) == (('project', 'topic-b'), ('project', 'topic-a'))


class TestImportLeaf:
    """The flood-control module must stay a stdlib-only leaf (the
    ``tests/test_consolidation_auto.py`` rule): importing it loads neither the
    config schema nor pydantic. Probed in a fresh interpreter because this
    process already imported everything at collection."""

    def test_import_is_stdlib_only(self):
        result = subprocess.run(
            [
                sys.executable,
                '-c',
                'import sys\n'
                'import fused_memory.reconciliation.consolidation_flood_control  # noqa: F401\n'
                "heavy = ['fused_memory.config.schema', 'pydantic']\n"
                'present = [m for m in heavy if m in sys.modules]\n'
                'assert not present, present\n',
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
