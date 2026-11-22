"""Tests for scripts/luna_eval_score.py (task 9945): the paired statistic and each site's pre-registered decision.

Every input is what the score is handed in production: run records as
luna_eval_sites writes them, gold and ratings as plain mappings, and
population_weighted blocks as replay_overreach_extraction.score_replay passes
them to its ``decide``.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

SCRIPTS = Path(__file__).parent.parent / 'scripts'
score = load_script_module(SCRIPTS / 'luna_eval_score.py', mod_name='luna_eval_score')
audit = load_script_module(SCRIPTS / 'audit_ruling_overreach.py', mod_name='audit_ruling_overreach')
replay = load_script_module(SCRIPTS / 'replay_overreach_extraction.py', mod_name='replay_overreach_extraction')

CONTROL, LUNA = 'control', 'luna-low'
CLIENTS = {
    CONTROL: {'model': 'gpt-4o-mini', 'reasoning_effort': None, 'endpoint': 'configured'},
    LUNA: {'model': 'gpt-6-luna', 'reasoning_effort': 'low', 'endpoint': 'plan_proxy'},
}


def _run(site: str, arm: str, fixture_id: str, **measured: Any) -> dict[str, Any]:
    return {
        'kind': 'replayed', 'site': site, 'arm': arm, **CLIENTS[arm], 'fixture_id': fixture_id, **measured,
        'tokens': {site: {'input_tokens': 100, 'output_tokens': 10, 'llm_calls': 1}},
    }


def _failed(site: str, arm: str, fixture_id: str) -> dict[str, Any]:
    return {'kind': 'failed', 'site': site, 'arm': arm, 'fixture_id': fixture_id,
            'error_type': 'APIConnectionError', 'error': 'connection reset'}


def _paired(b: int, c: int, n: int) -> tuple[float, list[float]]:
    """The pre-registered Wald interval, written out by hand: d=(b-c)/n, se=sqrt((b+c)-(b-c)²/n)/n."""
    d = (b - c) / n
    se = math.sqrt((b + c) - (b - c) ** 2 / n) / n
    return round(d, 4), [round(d - audit.WILSON_Z * se, 4), round(d + audit.WILSON_Z * se, 4)]


class TestTheBars:
    def test_every_bar_is_pre_registered_here(self) -> None:
        assert (score.EXTRACTION_OVERREACH_MARGIN, score.RESOLVE_ACCURACY_MARGIN, score.RESOLVE_FALSE_MERGE_MARGIN,
                score.SUMMARY_FIDELITY_MARGIN, score.ROUTING_LABEL_MARGIN, score.LUNA_FAILURE_SHARE_MAX) == (
            0.05, 0.05, 0.03, 0.10, 0.05, 0.02)

    def test_holding_retention_is_the_replays_own_bar(self) -> None:
        assert score.HOLDING_RETENTION_MIN is replay.HOLDING_RETENTION_MIN

    def test_the_interval_uses_the_audits_z(self) -> None:
        assert score.WILSON_Z is audit.WILSON_Z


class TestPairedNoninferiority:
    def test_no_discordant_pair_passes(self) -> None:
        result = score.paired_noninferiority(0, 0, 300, 0.05)
        assert (result['difference'], result['ci'], result['passes']) == (0.0, [0.0, 0.0], True)

    def test_known_values(self) -> None:
        result = score.paired_noninferiority(11, 19, 300, 0.05)
        assert (result['difference'], result['ci']) == (-0.0267, [-0.0623, 0.009])
        assert result['passes'] is False
        assert score.paired_noninferiority(11, 19, 300, 0.07)['passes'] is True

    @pytest.mark.parametrize(('b', 'c', 'n'), [(3, 9, 120), (0, 4, 200), (7, 2, 80), (5, 5, 10)])
    def test_the_pre_registered_formula(self, b: int, c: int, n: int) -> None:
        result = score.paired_noninferiority(b, c, n, 0.05)
        assert (result['difference'], result['ci']) == _paired(b, c, n)

    def test_it_reports_what_it_was_given(self) -> None:
        result = score.paired_noninferiority(11, 19, 300, 0.05)
        assert (result['candidate_only'], result['control_only'], result['n'], result['margin']) == (11, 19, 300, 0.05)

    def test_the_bound_is_inclusive(self) -> None:
        every_pair_lost = score.paired_noninferiority(0, 10, 10, 1.0)
        assert (every_pair_lost['ci'], every_pair_lost['passes']) == ([-1.0, -1.0], True)
        assert score.paired_noninferiority(0, 10, 10, 0.9999)['passes'] is False

    def test_no_pair_cannot_pass(self) -> None:
        result = score.paired_noninferiority(0, 0, 0, 0.05)
        assert (result['difference'], result['ci'], result['passes']) == (None, None, False)

    @pytest.mark.parametrize(('b', 'c', 'n'), [(6, 5, 10), (-1, 0, 10), (0, -1, 10), (0, 0, -1)])
    def test_impossible_counts_are_refused(self, b: int, c: int, n: int) -> None:
        with pytest.raises(ValueError):
            score.paired_noninferiority(b, c, n, 0.05)


def _weighted(rate: float | None, ci: list[float] | None, holding: float) -> dict[str, Any]:
    """A population_weighted block, as replay_overreach_extraction.arm_score builds it."""
    return {
        'overreach_rate_minted': {'rate': rate, 'ci': ci, 'effective_n': 120.0},
        'window_counts': {'minted': 500.0, 'overreach': 60.0, 'holding': holding},
    }


CONTROL_WEIGHTED = _weighted(0.12, [0.08, 0.17], 50.0)


class TestExtractionNoninferiority:
    def test_it_passes_within_both_bars(self) -> None:
        decision = score.extraction_noninferiority(CONTROL_WEIGHTED, _weighted(0.10, [0.06, 0.15], 45.0))
        assert decision['checks'] == {'E1_overreach_upper_within_margin': True, 'E2_retains_holdings': True}
        assert decision['passes'] is True

    @pytest.mark.parametrize(('upper', 'passes'), [(0.17, True), (0.1701, False)])
    def test_e1_is_inclusive_at_the_control_point_plus_the_margin(self, upper: float, passes: bool) -> None:
        decision = score.extraction_noninferiority(CONTROL_WEIGHTED, _weighted(0.13, [0.09, upper], 50.0))
        assert decision['checks']['E1_overreach_upper_within_margin'] is passes
        assert decision['passes'] is passes

    @pytest.mark.parametrize(('holding', 'passes'), [(40.0, True), (39.9, False)])
    def test_e2_is_inclusive_at_the_retention_bar(self, holding: float, passes: bool) -> None:
        decision = score.extraction_noninferiority(CONTROL_WEIGHTED, _weighted(0.10, [0.06, 0.15], holding))
        assert decision['checks']['E2_retains_holdings'] is passes
        assert decision['passes'] is passes

    def test_it_states_its_statistics_and_bars(self) -> None:
        decision = score.extraction_noninferiority(CONTROL_WEIGHTED, _weighted(0.10, [0.06, 0.15], 45.0))
        assert decision['statistics'] == {
            'control_overreach_rate': 0.12, 'candidate_overreach_upper': 0.15, 'holding_retention': 0.9,
        }
        assert decision['bars'] == {'overreach_margin': 0.05, 'holding_retention_min': 0.8}

    @pytest.mark.parametrize('candidate', [_weighted(None, None, 45.0), _weighted(0.1, [0.06, 0.15], 0.0)])
    def test_an_unmeasured_candidate_cannot_pass(self, candidate: dict[str, Any]) -> None:
        assert score.extraction_noninferiority(CONTROL_WEIGHTED, candidate)['passes'] is False

    def test_no_control_holding_cannot_pass(self) -> None:
        decision = score.extraction_noninferiority(_weighted(0.12, [0.08, 0.17], 0.0), _weighted(0.1, [0.06, 0.15], 9.0))
        assert decision['statistics']['holding_retention'] is None
        assert decision['checks']['E2_retains_holdings'] is False


# --------------------------------------------------------------------------- #
# resolve_edge: each arm's merge decision against gold labelled blind
# --------------------------------------------------------------------------- #

RESOLVE = 'resolve_edge'


def _resolved(arm: str, fixture_id: str, duplicates: list[str], invalidated: Sequence[str] = ()) -> dict[str, Any]:
    return _run(RESOLVE, arm, fixture_id, duplicate_uuids=list(duplicates),
                merged_into=duplicates[0] if duplicates else None, invalidated_uuids=list(invalidated))


def _gold(*duplicates: str, contradicted: tuple[str, ...] = ()) -> Any:
    return score.ResolveGold(duplicates=frozenset(duplicates), contradicted=frozenset(contradicted))


UNJUDGEABLE = score.ResolveGold(duplicates=frozenset(), contradicted=frozenset(), judgeable=False)


def _resolve_runs(control: dict[str, list[str]], luna: dict[str, list[str]]) -> list[dict[str, Any]]:
    return ([_resolved(CONTROL, fid, dups) for fid, dups in control.items()]
            + [_resolved(LUNA, fid, dups) for fid, dups in luna.items()])


class TestResolveEdgeDecision:
    @pytest.mark.parametrize(('luna', 'gold', 'correct'), [
        ([], (), True),
        (['s1'], ('s1',), True),
        (['s1', 's2'], ('s1',), True),
        (['s2', 's1'], ('s1',), False),
        ([], ('s1',), False),
        (['s1'], (), False),
    ])
    def test_correct_is_the_merge_decision_and_its_first_duplicate(
        self, luna: list[str], gold: tuple[str, ...], correct: bool,
    ) -> None:
        control = list(gold[:1])
        decision = score.resolve_edge_decision(_resolve_runs({'f': control}, {'f': luna}), {'f': _gold(*gold)})
        accuracy = decision['statistics']['merge_accuracy']
        assert (accuracy['candidate_only'], accuracy['control_only'], accuracy['n']) == (0, 0 if correct else 1, 1)

    @pytest.mark.parametrize(('luna', 'false_merge'), [([], False), (['s2'], True), (['s1'], False)])
    def test_a_false_merge_is_a_merge_into_a_non_duplicate(self, luna: list[str], false_merge: bool) -> None:
        decision = score.resolve_edge_decision(_resolve_runs({'f': ['s1']}, {'f': luna}), {'f': _gold('s1')})
        assert decision['statistics']['no_false_merge']['control_only'] == int(false_merge)

    def test_merging_where_gold_holds_no_duplicate_is_a_false_merge(self) -> None:
        decision = score.resolve_edge_decision(_resolve_runs({'f': []}, {'f': ['s1']}), {'f': _gold()})
        assert decision['statistics']['no_false_merge']['control_only'] == 1

    def test_an_unjudgeable_fixture_is_excluded_from_n(self) -> None:
        runs = _resolve_runs({'a': [], 'b': ['s1'], 'c': []}, {'a': [], 'b': [], 'c': ['s9']})
        decision = score.resolve_edge_decision(runs, {'a': _gold(), 'b': _gold('s1'), 'c': UNJUDGEABLE})
        assert decision['statistics']['merge_accuracy']['n'] == 2
        assert decision['statistics']['no_false_merge']['n'] == 2
        assert decision['statistics']['unjudgeable'] == 1

    def test_r1_and_r2_are_separate_bars(self) -> None:
        """Four false merges in 200 keep merge accuracy within 0.05 and break the 0.03 false-merge guard."""
        fixtures = {f'f{i:03d}': ['s1'] if i < 4 else [] for i in range(200)}
        runs = _resolve_runs({fid: [] for fid in fixtures}, fixtures)
        decision = score.resolve_edge_decision(runs, {fid: _gold() for fid in fixtures})
        assert decision['statistics']['merge_accuracy']['ci'] == _paired(0, 4, 200)[1]
        assert decision['checks'] == {
            'R1_merge_accuracy_noninferior': True, 'R2_no_false_merge_noninferior': False,
            'candidate_failures_within_share': True,
        }
        assert decision['passes'] is False
        assert decision['bars'] == {
            'merge_accuracy_margin': 0.05, 'no_false_merge_margin': 0.03, 'candidate_failure_share_max': 0.02,
        }

    def test_arms_that_agree_with_gold_pass(self) -> None:
        fixtures = {'a': ['s1'], 'b': []}
        decision = score.resolve_edge_decision(_resolve_runs(fixtures, fixtures), {'a': _gold('s1'), 'b': _gold()})
        assert decision['passes'] is True

    def test_contradiction_agreement_is_reported_not_gated(self) -> None:
        runs = [_resolved(CONTROL, 'f', [], ['old']), _resolved(LUNA, 'f', [], [])]
        decision = score.resolve_edge_decision(runs, {'f': _gold(contradicted=('old',))})
        assert decision['statistics']['contradiction_agreement'] == {CONTROL: 1.0, LUNA: 0.0}
        assert decision['passes'] is True

    def test_a_judged_fixture_without_gold_is_refused(self) -> None:
        with pytest.raises(score.MissingVerdicts) as caught:
            score.resolve_edge_decision(_resolve_runs({'a': [], 'b': []}, {'a': [], 'b': []}), {'a': _gold()})
        assert 'b' in str(caught.value)


# --------------------------------------------------------------------------- #
# summaries: each arm's summary per node, rated blind
# --------------------------------------------------------------------------- #

SUMMARIES = 'summaries'


def _summary_runs(items: dict[str, list[str]]) -> list[dict[str, Any]]:
    return [_run(SUMMARIES, arm, fid, summaries={node: f'{arm} on {node}' for node in nodes})
            for arm in (CONTROL, LUNA) for fid, nodes in items.items()]


def _ratings(control: list[str], luna: list[str]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, str]]]:
    """One fixture per rated pair, one node each."""
    items = {f'g/ep{i:03d}': ['n'] for i in range(len(control))}
    ratings = {(fid, 'n'): {CONTROL: c, LUNA: lu} for fid, c, lu in zip(items, control, luna, strict=True)}
    return _summary_runs(items), ratings


class TestSummaryDecision:
    def test_correct_is_rated_faithful(self) -> None:
        runs, ratings = _ratings(['faithful', 'unsupported', 'unsupported', 'faithful'],
                                 ['dropped', 'faithful', 'unsupported', 'faithful'])
        fidelity = score.summary_decision(runs, ratings)['statistics']['faithful_rate']
        assert (fidelity['candidate_only'], fidelity['control_only'], fidelity['n']) == (1, 1, 4)

    @pytest.mark.parametrize(('lost', 'passes'), [(3, True), (6, False)])
    def test_s1_margin_is_ten_points(self, lost: int, passes: bool) -> None:
        runs, ratings = _ratings(['faithful'] * 80, ['unsupported'] * lost + ['faithful'] * (80 - lost))
        decision = score.summary_decision(runs, ratings)
        assert decision['statistics']['faithful_rate']['ci'] == _paired(0, lost, 80)[1]
        assert decision['checks']['S1_faithful_rate_noninferior'] is passes
        assert decision['passes'] is passes
        assert decision['bars']['faithful_rate_margin'] == 0.10

    def test_every_node_of_a_paired_fixture_is_an_item(self) -> None:
        runs = _summary_runs({'g/ep': ['n1', 'n2']})
        ratings = {('g/ep', node): {CONTROL: 'faithful', LUNA: 'faithful'} for node in ('n1', 'n2')}
        assert score.summary_decision(runs, ratings)['statistics']['faithful_rate']['n'] == 2

    def test_an_unrated_item_is_refused(self) -> None:
        runs = _summary_runs({'g/ep': ['n1', 'n2']})
        with pytest.raises(score.MissingVerdicts) as caught:
            score.summary_decision(runs, {('g/ep', 'n1'): {CONTROL: 'faithful', LUNA: 'faithful'}})
        assert 'n2' in str(caught.value)

    def test_an_item_rated_under_one_arm_only_is_refused(self) -> None:
        runs = _summary_runs({'g/ep': ['n1']})
        with pytest.raises(score.MissingVerdicts):
            score.summary_decision(runs, {('g/ep', 'n1'): {CONTROL: 'faithful'}})


# --------------------------------------------------------------------------- #
# classifier and router: labels, adjudicated on discordant items only
# --------------------------------------------------------------------------- #

CLASSIFIER, ROUTER = 'classifier', 'router'


def _labelled(site: str, labels: dict[str, tuple[str, str]]) -> list[dict[str, Any]]:
    """fixture_id -> (control label, luna label)."""
    return [_run(site, arm, fid, label=pair[index]) for index, arm in enumerate((CONTROL, LUNA))
            for fid, pair in labels.items()]


class TestRoutingLabelDecision:
    def test_an_agreement_needs_no_gold_and_counts_as_equally_right_or_wrong(self) -> None:
        decision = score.routing_label_decision(_labelled(CLASSIFIER, {'1': ('temporal_facts', 'temporal_facts')}), {})
        accuracy = decision['statistics']['label_accuracy']
        assert (accuracy['candidate_only'], accuracy['control_only'], accuracy['n']) == (0, 0, 1)

    @pytest.mark.parametrize(('gold', 'candidate_only', 'control_only'), [
        ('procedural_knowledge', 1, 0),
        ('temporal_facts', 0, 1),
        ('observations_and_summaries', 0, 0),
    ])
    def test_gold_decides_a_discordant_item(self, gold: str, candidate_only: int, control_only: int) -> None:
        runs = _labelled(CLASSIFIER, {'1': ('temporal_facts', 'procedural_knowledge')})
        accuracy = score.routing_label_decision(runs, {'1': gold})['statistics']['label_accuracy']
        assert (accuracy['candidate_only'], accuracy['control_only'], accuracy['n']) == (
            candidate_only, control_only, 1)

    def test_discordance_is_reported(self) -> None:
        runs = _labelled(ROUTER, {'1': ('temporal', 'temporal'), '2': ('entity', 'procedural')})
        assert score.routing_label_decision(runs, {'2': 'entity'})['statistics']['discordant'] == 1

    def test_a_discordant_item_without_gold_is_refused(self) -> None:
        with pytest.raises(score.MissingVerdicts) as caught:
            score.routing_label_decision(_labelled(ROUTER, {'7': ('temporal', 'entity')}), {})
        assert '7' in str(caught.value)

    @pytest.mark.parametrize(('control_only', 'passes'), [(9, True), (19, False)])
    def test_the_label_margin_is_five_points(self, control_only: int, passes: bool) -> None:
        """At n=300, 9 items lost and none won stays within 0.05. 19 lost against 11 won does not."""
        won = 0 if control_only == 9 else 11
        labels = {f'{i:03d}': ('temporal', 'temporal') for i in range(300)}
        gold = {}
        for i in range(control_only):
            labels[f'{i:03d}'], gold[f'{i:03d}'] = ('temporal', 'entity'), 'temporal'
        for i in range(control_only, control_only + won):
            labels[f'{i:03d}'], gold[f'{i:03d}'] = ('temporal', 'entity'), 'entity'
        decision = score.routing_label_decision(_labelled(ROUTER, labels), gold)
        assert decision['statistics']['label_accuracy']['ci'] == _paired(won, control_only, 300)[1]
        assert decision['checks']['label_accuracy_noninferior'] is passes
        assert decision['bars']['label_accuracy_margin'] == 0.05


def _decision(passes: bool) -> dict[str, Any]:
    return {'statistics': {}, 'bars': {}, 'checks': {}, 'passes': passes}


class TestTheRoutingLayer:
    @pytest.mark.parametrize(('classifier', 'router'), [(True, True), (True, False), (False, True), (False, False)])
    def test_it_passes_only_if_both_sites_pass(self, classifier: bool, router: bool) -> None:
        layer = score.routing_layer_decision(_decision(classifier), _decision(router))
        assert layer['checks'] == {'C1_classifier_label_accuracy': classifier, 'Q1_router_label_accuracy': router}
        assert layer['passes'] is (classifier and router)


# --------------------------------------------------------------------------- #
# Failure lines: a candidate that failed too often fails, whatever its statistic
# --------------------------------------------------------------------------- #


def _agreeing_with_failures(site: str, luna_failures: int, control_failures: int = 0) -> list[dict[str, Any]]:
    """50 fixtures both arms label alike, with the first *luna_failures* failed under luna and the last
    *control_failures* failed under control."""
    ids = [f'{i:02d}' for i in range(50)]
    records = []
    for index, fid in enumerate(ids):
        records.append(_failed(site, CONTROL, fid) if index >= 50 - control_failures
                       else _run(site, CONTROL, fid, label='temporal'))
        records.append(_failed(site, LUNA, fid) if index < luna_failures else _run(site, LUNA, fid, label='temporal'))
    return records


class TestTheFailureShare:
    @pytest.mark.parametrize(('failures', 'passes'), [(0, True), (1, True), (2, False)])
    def test_more_than_two_percent_luna_failures_fail_the_site(self, failures: int, passes: bool) -> None:
        decision = score.routing_label_decision(_agreeing_with_failures(ROUTER, failures), {})
        assert decision['checks']['label_accuracy_noninferior'] is True
        assert decision['checks']['candidate_failures_within_share'] is passes
        assert decision['passes'] is passes
        assert decision['statistics']['failures'] == {
            'candidate_arm': LUNA, 'fixtures': 50, 'candidate_failed': failures, 'control_failed': 0,
            'candidate_failure_share': failures / 50,
        }
        assert decision['bars']['candidate_failure_share_max'] == 0.02

    def test_a_failed_fixture_leaves_the_pairs(self) -> None:
        decision = score.routing_label_decision(_agreeing_with_failures(ROUTER, 1, control_failures=3), {})
        assert decision['statistics']['label_accuracy']['n'] == 46
        assert decision['statistics']['failures']['control_failed'] == 3
        assert decision['checks']['candidate_failures_within_share'] is True

    def test_the_gate_applies_to_resolve_edge(self) -> None:
        runs = _resolve_runs({'a': [], 'b': []}, {'a': []}) + [_failed(RESOLVE, LUNA, 'b')]
        decision = score.resolve_edge_decision(runs, {'a': _gold(), 'b': _gold()})
        assert decision['checks']['candidate_failures_within_share'] is False
        assert decision['passes'] is False

    def test_the_gate_applies_to_summaries(self) -> None:
        runs = _summary_runs({'g/ep': ['n']})[:1] + [_failed(SUMMARIES, LUNA, 'g/ep')]
        decision = score.summary_decision(runs, {})
        assert decision['statistics']['faithful_rate']['n'] == 0
        assert decision['passes'] is False


class TestTheRunsFailClosed:
    def test_arms_covering_different_fixtures_are_refused(self) -> None:
        runs = _labelled(ROUTER, {'1': ('temporal', 'temporal')}) + [_run(ROUTER, CONTROL, '2', label='temporal')]
        with pytest.raises(score.RunFileError) as caught:
            score.routing_label_decision(runs, {})
        assert '2' in str(caught.value)

    def test_runs_of_two_sites_are_refused(self) -> None:
        runs = _labelled(ROUTER, {'1': ('temporal', 'temporal')}) + _labelled(CLASSIFIER, {'1': ('a', 'a')})
        with pytest.raises(score.RunFileError):
            score.routing_label_decision(runs, {})

    def test_a_third_arm_is_refused(self) -> None:
        runs = _labelled(ROUTER, {'1': ('temporal', 'temporal')})
        runs.append(_run(ROUTER, CONTROL, '1', label='temporal') | {'arm': 'o6-low'})
        with pytest.raises(score.RunFileError):
            score.routing_label_decision(runs, {})

    def test_runs_without_a_control_are_refused(self) -> None:
        with pytest.raises(score.RunFileError):
            score.routing_label_decision([_run(ROUTER, LUNA, '1', label='temporal')], {})

    def test_one_fixture_recorded_twice_under_an_arm_is_refused(self) -> None:
        runs = _labelled(ROUTER, {'1': ('temporal', 'temporal')})
        with pytest.raises(score.RunFileError):
            score.routing_label_decision([*runs, runs[0]], {})

    def test_no_runs_are_refused(self) -> None:
        with pytest.raises(score.RunFileError):
            score.routing_label_decision([], {})
