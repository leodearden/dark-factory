"""Tests for scripts/luna_eval_score.py (task 9945): the paired statistic and each site's pre-registered decision.

Every input is what the score is handed in production: run records as
luna_eval_sites writes them, gold and ratings as plain mappings, and
population_weighted blocks as replay_overreach_extraction.score_replay passes
them to its ``decide``.
"""
from __future__ import annotations

import hashlib
import json
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


# --------------------------------------------------------------------------- #
# Blind worksheets
# --------------------------------------------------------------------------- #

ARM_WORDS = ('control', 'luna', 'gpt', 'arm', 'model', 'plan_proxy', 'configured')


def _blind(rows: Any) -> bool:
    text = json.dumps(rows).lower()
    return not any(word in text for word in ARM_WORDS)


def _edge(uuid: str, fact: str) -> dict[str, Any]:
    return {'uuid': uuid, 'fact': fact, 'source_node_uuid': 'A', 'target_node_uuid': 'B',
            'created_at': '2026-10-01T00:00:00+00:00', 'valid_at': None, 'invalid_at': None, 'expired_at': None}


RESOLVE_FIXTURES = {
    'dark_factory/new': {
        'graph': 'dark_factory', 'edge_uuid': 'new', 'fact': 'Alpha adopts Beta for the queue',
        'source_node_uuid': 'A', 'target_node_uuid': 'B', 'created_at': '2026-10-06T00:00:00+00:00',
        'valid_at': None, 'invalid_at': None, 'expired_at': None,
        'episode_uuid': 'ep-1', 'episode_valid_at': '2026-10-06T00:00:00+00:00',
        'related': [_edge('sib', 'Alpha uses Beta for the queue')],
        'invalidation_candidates': [_edge('old', 'Alpha uses Xi for the queue')],
    },
}


class TestTheResolveGoldWorksheet:
    def test_it_carries_only_the_fixtures_facts(self) -> None:
        assert score.resolve_gold_worksheet(RESOLVE_FIXTURES) == [{
            'fixture_id': 'dark_factory/new', 'fact': 'Alpha adopts Beta for the queue',
            'related': [{'uuid': 'sib', 'fact': 'Alpha uses Beta for the queue'}],
            'invalidation_candidates': [{'uuid': 'old', 'fact': 'Alpha uses Xi for the queue'}],
        }]

    def test_it_names_no_arm_model_or_label(self) -> None:
        assert _blind(score.resolve_gold_worksheet(RESOLVE_FIXTURES))


EPISODE = 'dark_factory/ep-1'
SUMMARY_FIXTURES = {
    EPISODE: {
        'graph': 'dark_factory', 'episode_uuid': 'ep-1', 'episode_valid_at': '2026-10-06T00:00:00+00:00',
        'episode_content': 'Alpha now routes the queue through Beta.', 'previous': ['Alpha ran the queue alone.'],
        'nodes': [{'node_uuid': 'A', 'node_name': 'Alpha', 'summary': 'Alpha runs the queue.'},
                  {'node_uuid': 'B', 'node_name': 'Beta', 'summary': 'Beta is a broker.'}],
        'edges': [{'uuid': 'e1', 'source_node_uuid': 'A', 'target_node_uuid': 'B', 'fact': 'Alpha routes via Beta'},
                  {'uuid': 'e2', 'source_node_uuid': 'B', 'target_node_uuid': 'B', 'fact': 'Beta restarts nightly'}],
    },
}


def _summary_run(arm: str, summaries: dict[str, str]) -> dict[str, Any]:
    return _run(SUMMARIES, arm, EPISODE, summaries=summaries)


SUMMARY_RUNS = [
    _summary_run(CONTROL, {'A': 'Alpha routes the queue via Beta.', 'B': 'Beta brokers the queue.'}),
    _summary_run(LUNA, {'A': 'Alpha sends queue work to Beta.', 'B': ''}),
]


def _letters(fixture_id: str) -> dict[str, str]:
    """The pre-registered blinding: the arm whose sha256(fixture_id:arm) sorts first is A."""
    first, second = sorted((CONTROL, LUNA), key=lambda arm: hashlib.sha256(f'{fixture_id}:{arm}'.encode()).hexdigest())
    return {'A': first, 'B': second}


class TestTheSummaryRatingWorksheet:
    def test_each_node_is_one_item_with_its_inputs_and_both_summaries(self) -> None:
        rows, _ = score.summary_rating_worksheet(SUMMARY_FIXTURES, SUMMARY_RUNS)
        letters = _letters(EPISODE)
        by_arm = {record['arm']: record['summaries'] for record in SUMMARY_RUNS}
        assert rows[0] == {
            'fixture_id': EPISODE, 'node_uuid': 'A', 'node_name': 'Alpha', 'previous_summary': 'Alpha runs the queue.',
            'episode_content': 'Alpha now routes the queue through Beta.',
            'previous_episodes': ['Alpha ran the queue alone.'], 'facts': ['Alpha routes via Beta'],
            'summaries': {letter: by_arm[arm]['A'] for letter, arm in letters.items()},
        }
        assert [row['node_uuid'] for row in rows] == ['A', 'B']
        assert rows[1]['facts'] == ['Alpha routes via Beta', 'Beta restarts nightly']

    def test_an_empty_summary_is_shown_as_none(self) -> None:
        rows, _ = score.summary_rating_worksheet(SUMMARY_FIXTURES, SUMMARY_RUNS)
        luna_letter = next(letter for letter, arm in _letters(EPISODE).items() if arm == LUNA)
        assert rows[1]['summaries'][luna_letter] is None

    def test_the_key_maps_each_letter_to_its_arm_by_sha256(self) -> None:
        _, key = score.summary_rating_worksheet(SUMMARY_FIXTURES, SUMMARY_RUNS)
        assert key == [{'fixture_id': EPISODE, 'node_uuid': node, **_letters(EPISODE)} for node in ('A', 'B')]

    def test_the_worksheet_never_names_an_arm_or_model(self) -> None:
        rows, _ = score.summary_rating_worksheet(SUMMARY_FIXTURES, SUMMARY_RUNS)
        assert _blind(rows)

    def test_a_fixture_one_arm_failed_is_not_offered(self) -> None:
        runs = [SUMMARY_RUNS[0], _failed(SUMMARIES, LUNA, EPISODE)]
        assert score.summary_rating_worksheet(SUMMARY_FIXTURES, runs) == ([], [])


CATEGORIES = ('temporal_facts', 'entities_and_relations', 'procedural_knowledge')


def _routing_runs(site: str, labels: dict[str, tuple[str, str]]) -> list[dict[str, Any]]:
    return _labelled(site, labels)


class TestTheRoutingDiscordantWorksheet:
    def _rows(self) -> list[dict[str, Any]]:
        runs = _routing_runs(CLASSIFIER, {'1': ('temporal_facts', 'temporal_facts'),
                                          '2': ('temporal_facts', 'procedural_knowledge')})
        return score.routing_discordant_worksheet({'1': 'first text', '2': 'second text'}, runs, CATEGORIES)

    def test_only_discordant_items_are_offered(self) -> None:
        assert [row['fixture_id'] for row in self._rows()] == ['2']

    def test_the_labels_are_unattributed_in_sha256_order(self) -> None:
        (row,) = self._rows()
        assert row['labels'] == sorted(
            ['temporal_facts', 'procedural_knowledge'],
            key=lambda label: hashlib.sha256(f'2:{label}'.encode()).hexdigest(),
        )
        assert row == {'site': CLASSIFIER, 'fixture_id': '2', 'text': 'second text', 'labels': row['labels'],
                       'vocabulary': list(CATEGORIES)}

    def test_it_never_names_an_arm_or_model(self) -> None:
        assert _blind(self._rows())


# --------------------------------------------------------------------------- #
# Verdict files fail closed, naming the offender
# --------------------------------------------------------------------------- #


def _gold_file(**overrides: Any) -> dict[str, Any]:
    verdict = {'fixture_id': 'dark_factory/new', 'duplicates': ['sib'], 'contradicted': ['old'], 'judgeable': True,
               'rationale': '"uses Beta for the queue" restates "adopts Beta for the queue"'}
    return {'gold': [verdict | overrides]}


class TestTheResolveGoldLoader:
    def test_a_complete_file_loads(self) -> None:
        worksheet = score.resolve_gold_worksheet(RESOLVE_FIXTURES)
        assert score.load_resolve_gold(_gold_file(), worksheet) == {
            'dark_factory/new': score.ResolveGold(duplicates=frozenset({'sib'}), contradicted=frozenset({'old'})),
        }

    def test_an_unjudgeable_fixture_loads_as_such(self) -> None:
        worksheet = score.resolve_gold_worksheet(RESOLVE_FIXTURES)
        gold = score.load_resolve_gold(_gold_file(duplicates=[], contradicted=[], judgeable=False), worksheet)
        assert gold['dark_factory/new'].judgeable is False

    @pytest.mark.parametrize(('obj', 'error', 'offender'), [
        ({'gold': []}, 'MissingVerdicts', 'dark_factory/new'),
        ({'gold': _gold_file()['gold'] * 2}, 'DuplicateVerdict', 'dark_factory/new'),
        (_gold_file(duplicates=['old']), 'UnknownLabel', 'dark_factory/new'),
        (_gold_file(contradicted=['elsewhere']), 'UnknownLabel', 'dark_factory/new'),
        (_gold_file(rationale='  '), 'MissingRationale', 'dark_factory/new'),
        ({'gold': _gold_file()['gold'] + _gold_file(fixture_id='reify/other')['gold']}, 'OutOfSampleVerdict',
         'reify/other'),
        ({'gold': [{'fixture_id': 'dark_factory/new'}]}, 'MalformedVerdicts', 'dark_factory/new'),
        ({'verdicts': []}, 'MalformedVerdicts', ''),
    ])
    def test_a_bad_file_is_refused_naming_the_offender(self, obj: Any, error: str, offender: str) -> None:
        with pytest.raises(getattr(score, error)) as caught:
            score.load_resolve_gold(obj, score.resolve_gold_worksheet(RESOLVE_FIXTURES))
        assert offender in str(caught.value)
        assert isinstance(caught.value, score.VerdictError)


def _ratings_file(rows: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    return {'ratings': [
        {'fixture_id': row['fixture_id'], 'node_uuid': row['node_uuid'], 'summary': letter,
         'rating': 'dropped' if row['summaries'][letter] is None else 'faithful',
         'rationale': 'quotes "routes the queue through Beta"'} | overrides
        for row in rows for letter in ('A', 'B')
    ]}


class TestTheSummaryRatingLoader:
    def test_ratings_are_unblinded_through_the_key(self) -> None:
        rows, key = score.summary_rating_worksheet(SUMMARY_FIXTURES, SUMMARY_RUNS)
        ratings = score.load_summary_ratings(_ratings_file(rows), SUMMARY_RUNS, key)
        assert ratings[(EPISODE, 'A')] == {CONTROL: 'faithful', LUNA: 'faithful'}
        assert ratings[(EPISODE, 'B')] == {CONTROL: 'faithful', LUNA: 'dropped'}

    def test_the_loaded_ratings_drive_the_decision(self) -> None:
        rows, key = score.summary_rating_worksheet(SUMMARY_FIXTURES, SUMMARY_RUNS)
        decision = score.summary_decision(
            SUMMARY_RUNS, score.load_summary_ratings(_ratings_file(rows), SUMMARY_RUNS, key),
        )
        assert decision['statistics']['faithful_rate']['control_only'] == 1

    @pytest.mark.parametrize(('mutate', 'error', 'offender'), [
        (lambda ratings: ratings[1:], 'MissingVerdicts', 'ep-1/A'),
        (lambda ratings: ratings + ratings[:1], 'DuplicateVerdict', 'ep-1/A'),
        (lambda ratings: [ratings[0] | {'rating': 'great'}, *ratings[1:]], 'UnknownLabel', 'ep-1/A'),
        (lambda ratings: [ratings[0] | {'rationale': ''}, *ratings[1:]], 'MissingRationale', 'ep-1/A'),
        (lambda ratings: [*ratings, ratings[0] | {'node_uuid': 'Z'}], 'OutOfSampleVerdict', 'ep-1/Z'),
        (lambda ratings: [*ratings, ratings[0] | {'summary': 'C'}], 'OutOfSampleVerdict', 'ep-1/A'),
    ])
    def test_a_bad_file_is_refused_naming_the_offender(self, mutate: Any, error: str, offender: str) -> None:
        rows, key = score.summary_rating_worksheet(SUMMARY_FIXTURES, SUMMARY_RUNS)
        obj = {'ratings': mutate(_ratings_file(rows)['ratings'])}
        with pytest.raises(getattr(score, error)) as caught:
            score.load_summary_ratings(obj, SUMMARY_RUNS, key)
        assert offender in str(caught.value)

    @pytest.mark.parametrize(('mutate', 'offender'), [
        pytest.param(lambda key: key[:1], 'ep-1/B', id='uncovered'),
        pytest.param(lambda key: [key[0] | {'A': key[0]['B'], 'B': key[0]['A']}, *key[1:]], 'ep-1/A', id='swapped'),
        pytest.param(lambda key: [*key, key[1]], 'ep-1/B', id='repeated'),
        pytest.param(lambda key: [*key, key[0] | {'node_uuid': 'Z'}], 'ep-1/Z', id='unoffered'),
    ])
    def test_a_key_other_than_the_sha256_rule_is_refused_naming_the_item(self, mutate: Any, offender: str) -> None:
        rows, key = score.summary_rating_worksheet(SUMMARY_FIXTURES, SUMMARY_RUNS)
        with pytest.raises(score.MalformedVerdicts) as caught:
            score.load_summary_ratings(_ratings_file(rows), SUMMARY_RUNS, mutate(key))
        assert offender in str(caught.value)

    def test_a_key_that_is_not_a_list_of_entries_is_refused(self) -> None:
        rows, _ = score.summary_rating_worksheet(SUMMARY_FIXTURES, SUMMARY_RUNS)
        with pytest.raises(score.MalformedVerdicts):
            score.load_summary_ratings(_ratings_file(rows), SUMMARY_RUNS, {'A': CONTROL})


class TestTheRoutingVerdictLoader:
    def _worksheet(self) -> list[dict[str, Any]]:
        classifier = score.routing_discordant_worksheet(
            {'2': 'second text'}, _routing_runs(CLASSIFIER, {'2': ('temporal_facts', 'procedural_knowledge')}),
            CATEGORIES)
        router = score.routing_discordant_worksheet(
            {'9': 'a query'}, _routing_runs(ROUTER, {'9': ('temporal', 'entity')}), ('temporal', 'entity', 'procedural'))
        return classifier + router

    def _file(self, **overrides: Any) -> dict[str, Any]:
        return {'verdicts': [
            {'site': CLASSIFIER, 'fixture_id': '2', 'label': 'entities_and_relations', 'rationale': 'names two teams'},
            {'site': ROUTER, 'fixture_id': '9', 'label': 'entity', 'rationale': 'asks what Alpha is'} | overrides,
        ]}

    def test_gold_is_keyed_by_site_then_fixture(self) -> None:
        assert score.load_routing_verdicts(self._file(), self._worksheet()) == {
            CLASSIFIER: {'2': 'entities_and_relations'}, ROUTER: {'9': 'entity'},
        }

    def test_a_label_neither_arm_gave_is_accepted_from_the_vocabulary(self) -> None:
        gold = score.load_routing_verdicts(self._file(label='procedural'), self._worksheet())
        assert gold[ROUTER] == {'9': 'procedural'}

    @pytest.mark.parametrize(('overrides', 'error', 'offender'), [
        ({'label': 'entities_and_relations'}, 'UnknownLabel', 'router/9'),
        ({'rationale': ''}, 'MissingRationale', 'router/9'),
        ({'fixture_id': '10'}, 'OutOfSampleVerdict', 'router/10'),
    ])
    def test_a_bad_verdict_is_refused_naming_the_offender(
        self, overrides: dict[str, Any], error: str, offender: str,
    ) -> None:
        with pytest.raises(getattr(score, error)) as caught:
            score.load_routing_verdicts(self._file(**overrides), self._worksheet())
        assert offender in str(caught.value)

    def test_a_missing_verdict_is_refused(self) -> None:
        obj = {'verdicts': self._file()['verdicts'][:1]}
        with pytest.raises(score.MissingVerdicts) as caught:
            score.load_routing_verdicts(obj, self._worksheet())
        assert 'router/9' in str(caught.value)

    def test_a_duplicate_verdict_is_refused(self) -> None:
        verdicts = self._file()['verdicts']
        with pytest.raises(score.DuplicateVerdict) as caught:
            score.load_routing_verdicts({'verdicts': [*verdicts, verdicts[1]]}, self._worksheet())
        assert 'router/9' in str(caught.value)


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #

pricing = load_script_module(SCRIPTS / 'score_write_triage_pairs.py', mod_name='score_write_triage_pairs')


def _costed(site: str, arm: str, fixture_id: str, input_tokens: int, output_tokens: int) -> dict[str, Any]:
    return _run(site, arm, fixture_id, label='x') | {
        'tokens': {site: {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'llm_calls': 1}},
    }


COST_RUNS = [
    _costed(ROUTER, CONTROL, '1', 100, 10), _costed(ROUTER, CONTROL, '2', 300, 30),
    _costed(ROUTER, LUNA, '1', 100, 10), _costed(ROUTER, LUNA, '2', 300, 30),
    _costed(CLASSIFIER, CONTROL, '1', 1000, 0), _costed(CLASSIFIER, LUNA, '1', 2000, 0),
    _failed(ROUTER, LUNA, '3'),
]


class TestTheCostBlock:
    def test_the_prices_are_the_write_triage_scorers(self) -> None:
        assert score.LIST_PRICES is pricing.LIST_PRICES
        assert score.LIST_PRICES_AS_OF is pricing.LIST_PRICES_AS_OF

    def test_tokens_per_call_per_site_per_arm(self) -> None:
        cost = score.cost_block(COST_RUNS, {ROUTER: 1000.0}, (280.0, 470.0))
        router = cost['sites'][ROUTER]
        assert {arm: (block['calls'], block['input_tokens_per_call'], block['output_tokens_per_call'])
                for arm, block in router.items()} == {CONTROL: (2, 200.0, 20.0), LUNA: (2, 200.0, 20.0)}
        assert (router[CONTROL]['model'], router[LUNA]['model']) == ('gpt-4o-mini', 'gpt-6-luna')

    def test_each_arm_is_priced_at_its_models_list_price(self) -> None:
        router = score.cost_block(COST_RUNS, {ROUTER: 1000.0}, (280.0, 470.0))['sites'][ROUTER]
        assert router[CONTROL]['usd_per_call'] == pytest.approx((200 * 0.15 + 20 * 0.60) / 1e6)
        assert router[LUNA]['usd_per_call'] == pytest.approx((200 * 0.10 + 20 * 0.50) / 1e6)

    def test_daily_usd_is_per_call_usd_times_the_sites_calls_per_day(self) -> None:
        cost = score.cost_block(COST_RUNS, {ROUTER: 1000.0}, (280.0, 470.0))
        assert cost['sites'][ROUTER][LUNA]['daily_usd'] == pytest.approx(0.03)
        assert cost['sites'][ROUTER][CONTROL]['daily_usd'] == pytest.approx(0.042)
        assert cost['sites'][CLASSIFIER][LUNA]['daily_usd'] is None

    def test_a_model_without_a_list_price_prices_as_none(self) -> None:
        runs = [_costed(ROUTER, CONTROL, '1', 100, 10), _costed(ROUTER, LUNA, '1', 100, 10) | {'model': 'unpriced'}]
        block = score.cost_block(runs, {ROUTER: 10.0}, (280.0, 470.0))['sites'][ROUTER][LUNA]
        assert (block['usd_per_call'], block['daily_usd']) == (None, None)

    def test_the_plan_window_share_is_inferred_at_each_end_of_the_window(self) -> None:
        cost = score.cost_block(COST_RUNS, {ROUTER: 1000.0, CLASSIFIER: 500.0}, (280.0, 470.0))
        weekly = 7 * (0.03 + 500.0 * 2000 * 0.10 / 1e6)
        window = cost['plan_window']
        assert window['weekly_candidate_usd'] == pytest.approx(weekly)
        assert window['window_usd_per_week'] == [280.0, 470.0]
        assert window['share'] == [pytest.approx(weekly / 280.0), pytest.approx(weekly / 470.0)]
        assert window['tag'] == 'inferred'
        assert cost['prices']['as_of'] == pricing.LIST_PRICES_AS_OF


# --------------------------------------------------------------------------- #
# report.json
# --------------------------------------------------------------------------- #


class TestTheReport:
    def test_a_site_block_carries_each_arms_client_counts_n_failures_and_decision(self) -> None:
        runs = _agreeing_with_failures(ROUTER, 1)
        decision = score.routing_label_decision(runs, {})
        block = score.site_block(runs, decision)
        assert block['arms'][CONTROL] == {
            **CLIENTS[CONTROL], 'replayed': 50, 'failed': 0,
            'tokens': {ROUTER: {'input_tokens': 5000, 'output_tokens': 500, 'llm_calls': 50}},
        }
        assert (block['arms'][LUNA]['model'], block['arms'][LUNA]['replayed'], block['arms'][LUNA]['failed']) == (
            'gpt-6-luna', 49, 1)
        assert block['n'] == 49
        assert block['failures'] == decision['statistics']['failures']
        assert block['decision'] == decision

    def test_an_arm_on_two_models_is_refused(self) -> None:
        runs = _labelled(ROUTER, {'1': ('temporal', 'temporal'), '2': ('temporal', 'temporal')})
        runs[-1] = runs[-1] | {'model': 'gpt-6.1-sol'}
        with pytest.raises(score.RunFileError):
            score.site_block(runs, _decision(True))

    def test_the_report_echoes_the_frame_by_sha256_and_assembles_its_parts(self) -> None:
        frame_text = json.dumps(EXTRACTION_FRAME)
        runs = _labelled(ROUTER, {'1': ('temporal', 'temporal')})
        sites = {ROUTER: score.site_block(runs, score.routing_label_decision(runs, {}))}
        layer = score.routing_layer_decision(_decision(True), sites[ROUTER]['decision'])
        cost = score.cost_block(runs, {ROUTER: 10.0}, (280.0, 470.0))
        records, verdicts = _extraction_replay()
        report = score.build_report(frame_text=frame_text, extraction_replay=records, extraction_verdicts=verdicts,
                                    sites=sites, routing_layer=layer, cost=cost)
        assert report == {
            'sample': {'frame_sha256': hashlib.sha256(frame_text.encode()).hexdigest()},
            'extraction': score.score_extraction(records, verdicts, EXTRACTION_FRAME),
            'sites': sites, 'routing_layer': layer, 'cost': cost,
        }
        json.dumps(report)


TEMPORAL = 'temporal_facts'
EXTRACTION_DEFINITION = audit.SampleDefinition(
    window_start='2026-08-01T00:00:00Z', window_end='2026-08-11T00:00:00Z', cap=3, strata=(TEMPORAL,),
)
EXTRACTION_FRAME = {
    'sample': {'definition': EXTRACTION_DEFINITION.to_dict(), 'window_days': 10.0,
               'sizes': {'dark_factory': {TEMPORAL: {'minted': 0, 'corroborated': 0, 'episodes': 2}}}},
    'strata': {'dark_factory': {TEMPORAL: {'window_population': 20}}},
}
EXTRACTION_LABELS = {
    CONTROL: {'ep1': ('holding', 'overreach'), 'ep2': ('holding', 'context')},
    LUNA: {'ep1': ('holding',), 'ep2': ('holding', 'holding')},
}


def _extraction_replay() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Both arms replayed over a two-episode frame, and one verdict per minted edge."""
    records, verdicts = [], []
    for arm, by_episode in EXTRACTION_LABELS.items():
        for episode, labels in by_episode.items():
            minted = [{'edge_uuid': f'{arm}-{episode}-{i}', 'fact': f'fact {i}', 'source_name': 'S',
                       'target_name': 'T'} for i in range(len(labels))]
            records.append({
                'kind': 'replayed', 'arm': arm, **CLIENTS[arm], 'instructions_version': 'none',
                'graph': 'dark_factory', 'uuid': episode, 'stratum': TEMPORAL, 'previous_count': 0,
                'entities': ['S', 'T'], 'minted': minted,
                'tokens': {'extract_nodes': {'input_tokens': 10, 'output_tokens': 1, 'llm_calls': 1}},
            })
            verdicts.extend({'graph': 'dark_factory', 'episode_uuid': episode, 'edge_uuid': edge['edge_uuid'],
                             'fact': edge['fact'], 'label': label, 'rationale': 'quotes the span'}
                            for edge, label in zip(minted, labels, strict=True))
    return records, {'sample': EXTRACTION_DEFINITION.to_dict(), 'verdicts': verdicts}


class TestTheExtractionLeg:
    def test_each_candidate_is_judged_by_e1_and_e2_through_the_replays_own_score(self) -> None:
        records, verdicts = _extraction_replay()
        scored = score.score_extraction(records, verdicts, EXTRACTION_FRAME)
        arms = scored['arms']
        assert scored['comparison'] == {
            LUNA: score.extraction_noninferiority(
                arms[CONTROL]['population_weighted'], arms[LUNA]['population_weighted']),
        }
        assert set(scored['comparison'][LUNA]['checks']) == {
            'E1_overreach_upper_within_margin', 'E2_retains_holdings'}
        assert arms[LUNA]['rates'][TEMPORAL]['label_counts']['holding'] == 3
