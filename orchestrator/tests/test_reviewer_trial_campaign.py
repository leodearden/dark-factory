"""Tests for the campaign reducer — pure over JSON-shaped score records."""

from __future__ import annotations

import pytest

from orchestrator.evals.reviewer_trial import campaign


def _rec(variant: str, diff: str, trial: int, matched: list[tuple[str, str]], fps: list[str],
         f1: float, cost: float = 1.0) -> dict:
    """A score record; *matched* is (gt_id, reviewer_severity), *fps* reviewer severities."""
    return {
        'variant_name': variant, 'diff_id': diff, 'trial': trial,
        'matches': [
            {'reviewer_issue': {'severity': sev}, 'ground_truth_id': gt, 'match_confidence': 1.0, 'match_reasoning': ''}
            for gt, sev in matched
        ],
        'unmatched_gt': [], 'false_positives': [{'severity': s} for s in fps],
        'recall': f1, 'precision': f1, 'f1': f1, 'blocking_recall': f1,
        'cost_usd': cost, 'match_cost_usd': 0.0, 'wall_clock_ms': 0,
    }


BLOCKING_GT = {'d1': {'g1', 'g2'}, 'd2': {'g3'}}


def test_blocking_precision_counts_only_blocking_findings_matched_to_blocking_truth() -> None:
    rec = _rec('a', 'd1', 1, [('g1', 'blocking'), ('g2', 'suggestion'), ('gx', 'blocking')], ['blocking', 'suggestion'], 0.5)
    found, true = campaign.blocking_counts(rec, BLOCKING_GT['d1'])
    assert (found, true) == (3, 1)


def test_complement_is_what_the_arm_caught_and_the_incumbent_missed_on_the_same_pair() -> None:
    incumbent = [_rec('inc', 'd1', 1, [('g1', 'blocking')], [], 0.5), _rec('inc', 'd1', 2, [('g1', 'blocking'), ('g2', 'blocking')], [], 0.6)]
    arm = [_rec('arm', 'd1', 1, [('g1', 'blocking'), ('g2', 'blocking')], [], 0.7), _rec('arm', 'd1', 2, [('g2', 'blocking')], [], 0.4)]
    assert campaign.complement(arm, incumbent) == (1, 3)


def test_strict_improvement_needs_three_trials_all_strictly_better() -> None:
    inc = [_rec('inc', 'd1', t, [], [], 0.5) for t in (1, 2, 3)]
    better = [_rec('arm', 'd1', t, [], [], 0.6) for t in (1, 2, 3)]
    tied_once = [_rec('tie', 'd1', t, [], [], f) for t, f in ((1, 0.6), (2, 0.5), (3, 0.6))]
    two_trials = [_rec('two', 'd1', t, [], [], 0.9) for t in (1, 2)]
    arms, decisions = campaign.summarize_campaign(inc + better + tied_once + two_trials, 'inc', BLOCKING_GT)
    assert decisions.strict_f1_improvement == {'arm': True, 'tie': False, 'two': False}
    assert [a.name for a in arms][0] == 'inc'


def test_summary_costs_and_decisions_and_markdown() -> None:
    inc = [_rec('inc', d, t, [('g1', 'blocking')], [], 0.5, cost=2.0) for d in ('d1', 'd2') for t in (1, 2, 3)]
    arm = [_rec('arm', d, t, [('g1', 'blocking'), ('g2', 'blocking')], ['blocking'], 0.6, cost=4.0) for d in ('d1', 'd2') for t in (1, 2, 3)]
    arms, decisions = campaign.summarize_campaign(inc + arm, 'inc', BLOCKING_GT, complement_threshold=0.10)
    by = {a.name: a for a in arms}
    assert by['inc'].cost_per_review == pytest.approx(2.0)
    assert by['inc'].cost_per_blocking_finding == pytest.approx(12.0 / 3)  # g1 is blocking truth only on d1
    assert by['arm'].blocking_precision == pytest.approx(3 / 9, abs=1e-3)
    assert by['arm'].complement_count == 6 and by['arm'].incumbent_matched == 6
    assert by['arm'].complement_share == pytest.approx(1.0)
    assert decisions.complement_pass == {'arm': True}
    assert by['arm'].paired_vs_incumbent[0] == {'trial': 1, 'wins': 2, 'losses': 0, 'ties': 0, 'mean_delta': pytest.approx(0.1)}
    md = campaign.format_markdown(arms, decisions)
    assert '| inc |' in md and '**PASS**' in md
    assert campaign.to_json(arms, decisions)['decisions']['incumbent'] == 'inc'


def test_missing_incumbent_is_an_error() -> None:
    with pytest.raises(ValueError):
        campaign.summarize_campaign([_rec('arm', 'd1', 1, [], [], 0.1)], 'inc', BLOCKING_GT)
