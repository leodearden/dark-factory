"""Tests for the campaign reducer — pure over JSON-shaped score records."""

from __future__ import annotations

from dataclasses import replace

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


def _unmeasured_rec(variant: str, diff: str, trial: int, cost: float = 1.0,
                    match_cost: float = 0.31) -> dict:
    """The ``asdict`` shape of a ScoringResult whose matcher did not measure."""
    return _rec(variant, diff, trial, [], [], f1=0.0, cost=cost) | {
        'unmeasured': {'stage': 'matcher', 'cause': 'cap_hit', 'reason': 'r', 'cost_usd': match_cost},
        'match_cost_usd': match_cost,
    }


def _measured_campaign() -> list[dict]:
    inc = [_rec('inc', d, t, [('g1', 'blocking')], [], 0.5, cost=2.0) for d in ('d1', 'd2') for t in (1, 2, 3)]
    arm = [_rec('arm', d, t, [('g1', 'blocking'), ('g2', 'blocking')], ['blocking'], 0.6, cost=4.0)
           for d in ('d1', 'd2') for t in (1, 2, 3)]
    return inc + arm


def test_an_unmeasured_record_changes_no_arm_figure_but_its_counts() -> None:
    without_refusal, _ = campaign.summarize_campaign(_measured_campaign(), 'inc', BLOCKING_GT)
    with_refusal, _ = campaign.summarize_campaign(
        [*_measured_campaign(), _unmeasured_rec('arm', 'd1', 1)], 'inc', BLOCKING_GT,
    )

    for refused, clean in zip(with_refusal, without_refusal, strict=True):
        assert replace(refused, unmeasured=0, unmeasured_cost_usd=0.0) == clean
    by = {a.name: a for a in with_refusal}
    assert by['arm'].unmeasured == 1
    assert by['arm'].unmeasured_cost_usd == pytest.approx(1.31)
    assert by['inc'].unmeasured == 0


def test_an_unmeasured_incumbent_cell_drops_that_pair_from_the_paired_comparison_and_complement() -> None:
    records = [r for r in _measured_campaign() if not (r['variant_name'] == 'inc' and (r['trial'], r['diff_id']) == (1, 'd1'))]
    records.append(_unmeasured_rec('inc', 'd1', 1))

    arms, _ = campaign.summarize_campaign(records, 'inc', BLOCKING_GT)

    arm = next(a for a in arms if a.name == 'arm')
    assert arm.paired_vs_incumbent[0] == {'trial': 1, 'wins': 1, 'losses': 0, 'ties': 0, 'mean_delta': pytest.approx(0.1)}
    assert (arm.complement_count, arm.incumbent_matched) == (5, 5)


def test_an_arm_with_nothing_measured_reports_none_never_zero() -> None:
    inc = [_rec('inc', d, t, [('g1', 'blocking')], [], 0.5) for d in ('d1', 'd2') for t in (1, 2, 3)]
    refused = [_unmeasured_rec('arm', d, t) for d in ('d1', 'd2') for t in (1, 2, 3)]

    arms, decisions = campaign.summarize_campaign(inc + refused, 'inc', BLOCKING_GT)

    arm = next(a for a in arms if a.name == 'arm')
    assert arm.mean_f1 is None
    assert arm.mean_recall is None
    assert arm.mean_precision is None
    assert arm.mean_blocking_recall is None
    assert arm.blocking_precision is None
    assert arm.cost_per_review is None
    assert arm.reviews == 0
    assert arm.trials == 0
    assert arm.f1_by_trial == []
    assert arm.unmeasured == len(refused)
    assert decisions.strict_f1_improvement['arm'] is False
    assert 'n/a' in campaign.format_markdown(arms, decisions)


def test_a_record_without_the_unmeasured_key_is_measured() -> None:
    pre_gamma = [_rec('inc', 'd1', t, [], [], 0.5) for t in (1, 2, 3)]
    assert all('unmeasured' not in r for r in pre_gamma)

    arms, _ = campaign.summarize_campaign(pre_gamma, 'inc', BLOCKING_GT)

    assert arms[0].mean_f1 == pytest.approx(0.5)
    assert arms[0].reviews == 3
    assert arms[0].unmeasured == 0


def test_markdown_names_each_arms_unmeasured_count() -> None:
    arms, decisions = campaign.summarize_campaign(
        [*_measured_campaign(), _unmeasured_rec('arm', 'd1', 1)], 'inc', BLOCKING_GT,
    )

    md = campaign.format_markdown(arms, decisions)

    assert '`arm` unmeasured: 1' in md
    assert '`inc` unmeasured: 0' in md
    assert '| unmeasured |' in md.splitlines()[0]
