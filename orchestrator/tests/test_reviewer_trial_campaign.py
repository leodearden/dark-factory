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


def test_complement_restricted_to_blocking_truth_counts_only_those_ids_on_both_sides() -> None:
    inc = [_rec('inc', 'd1', 1, [('g1', 'blocking'), ('gx', 'blocking')], [], 0.5),
           _rec('inc', 'd2', 1, [('g3', 'blocking')], [], 0.5)]
    arm = [_rec('arm', 'd1', 1, [('g1', 'blocking'), ('g2', 'blocking'), ('gy', 'blocking')], [], 0.5),
           _rec('arm', 'd2', 1, [('g3', 'blocking'), ('gz', 'blocking')], [], 0.5)]

    assert campaign.complement(arm, inc) == (3, 3)
    assert campaign.complement(arm, inc, restrict_to=BLOCKING_GT) == (1, 2)
    assert campaign.complement(arm, inc, restrict_to={'d1': {'g1', 'g2'}}) == (1, 1)


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


def _pair(band: campaign.SelfComplementBand, trial: int, over_trial: int) -> dict:
    return next(p for p in band.pairs if (p['trial'], p['over_trial']) == (trial, over_trial))


class TestSelfComplementBand:
    """The incumbent's run-to-run complement: every ordered pair of its own trials."""

    def test_every_ordered_pair_of_trials_is_a_row_and_the_bounds_span_the_defined_shares(self) -> None:
        inc = [
            _rec('inc', 'd1', 1, [('g1', 'blocking')], [], 0.5),
            _rec('inc', 'd1', 2, [('g1', 'blocking'), ('g2', 'blocking')], [], 0.5),
            _rec('inc', 'd1', 3, [('g1', 'blocking'), ('gx', 'blocking')], [], 0.5),
        ]
        overall = {(1, 2): 0.0, (1, 3): 0.0, (2, 1): 1.0, (2, 3): 0.5, (3, 1): 1.0, (3, 2): 0.5}
        blocking = {(1, 2): 0.0, (1, 3): 0.0, (2, 1): 1.0, (2, 3): 1.0, (3, 1): 0.0, (3, 2): 0.0}

        band = campaign.self_complement_band(inc, BLOCKING_GT)

        assert {(p['trial'], p['over_trial']) for p in band.pairs} == set(overall)
        assert len(band.pairs) == 6
        assert _pair(band, 2, 1) == {
            'trial': 2, 'over_trial': 1,
            'complement': 1, 'incumbent_matched': 1, 'share': 1.0,
            'blocking_complement': 1, 'blocking_incumbent_matched': 1, 'blocking_share': 1.0,
        }
        assert (_pair(band, 1, 2)['complement'], _pair(band, 1, 2)['share']) == (0, 0.0)
        assert (_pair(band, 3, 1)['share'], _pair(band, 3, 1)['blocking_share']) == (1.0, 0.0)
        for key, share in overall.items():
            assert _pair(band, *key)['share'] == share, key
        for key, share in blocking.items():
            assert _pair(band, *key)['blocking_share'] == share, key
        assert (band.overall_min, band.overall_max) == (min(overall.values()), max(overall.values()))
        assert (band.blocking_min, band.blocking_max) == (min(blocking.values()), max(blocking.values()))

    def test_a_fixture_unmeasured_in_one_trial_drops_out_of_every_pair_touching_that_trial(self) -> None:
        inc = [
            *(_rec('inc', 'd1', t, [('g1', 'blocking')], [], 0.5) for t in (1, 2, 3)),
            _rec('inc', 'd2', 1, [('g3', 'blocking')], [], 0.5),
            _unmeasured_rec('inc', 'd2', 2),
            _rec('inc', 'd2', 3, [('g3', 'blocking')], [], 0.5),
        ]

        band = campaign.self_complement_band(inc, BLOCKING_GT)

        for key in ((1, 2), (2, 1), (2, 3), (3, 2)):
            row = _pair(band, *key)
            assert (row['complement'], row['incumbent_matched']) == (0, 1), key
            assert (row['blocking_complement'], row['blocking_incumbent_matched']) == (0, 1), key
        for key in ((1, 3), (3, 1)):
            assert _pair(band, *key)['incumbent_matched'] == 2, key
        assert band.overall_max == 0.0
        assert band.blocking_max == 0.0

    def test_a_pair_over_a_trial_with_no_blocking_match_has_no_blocking_share(self) -> None:
        inc = [
            _rec('inc', 'd1', 1, [('gx', 'blocking')], [], 0.5),
            _rec('inc', 'd1', 2, [('g1', 'blocking'), ('gx', 'blocking')], [], 0.5),
        ]

        band = campaign.self_complement_band(inc, BLOCKING_GT)

        undefined = _pair(band, 2, 1)
        assert (undefined['blocking_complement'], undefined['blocking_incumbent_matched']) == (1, 0)
        assert undefined['blocking_share'] is None
        assert undefined['share'] == 1.0
        assert _pair(band, 1, 2)['blocking_share'] == 0.0
        assert (band.blocking_min, band.blocking_max) == (0.0, 0.0)
        assert (band.overall_min, band.overall_max) == (0.0, 1.0)

    @pytest.mark.parametrize(
        'incumbent',
        [
            pytest.param([_rec('inc', 'd1', 1, [('g1', 'blocking')], [], 0.5)], id='one-trial'),
            pytest.param(
                [_rec('inc', 'd1', 1, [('g1', 'blocking')], [], 0.5), _unmeasured_rec('inc', 'd1', 2)],
                id='second-trial-unmeasured',
            ),
            pytest.param([], id='no-records'),
        ],
    )
    def test_fewer_than_two_measured_trials_is_no_band(self, incumbent: list[dict]) -> None:
        band = campaign.self_complement_band(incumbent, BLOCKING_GT)

        assert band.pairs == []
        assert (band.overall_min, band.overall_max, band.blocking_min, band.blocking_max) == (
            None, None, None, None,
        )


def _trial_row(trial: int, blocking_share: float | None, share: float = 0.5) -> dict:
    """A per-trial complement row out of 100 incumbent matches (none blocking when undefined)."""
    blocking_matched = 0 if blocking_share is None else 100
    return {
        'trial': trial,
        'complement': round(share * 100), 'incumbent_matched': 100, 'share': share,
        'blocking_complement': round((blocking_share or 0.03) * 100),
        'blocking_incumbent_matched': blocking_matched, 'blocking_share': blocking_share,
    }


def _band(blocking_max: float | None) -> campaign.SelfComplementBand:
    return campaign.SelfComplementBand(
        pairs=[], overall_min=0.107, overall_max=0.333,
        blocking_min=None if blocking_max is None else 0.11, blocking_max=blocking_max,
    )


class TestComplementClearsTheBand:
    """D7's complement rule: the candidate's blocking complement beats the incumbent's own, every trial."""

    def test_complement_by_trial_pairs_each_trial_with_the_same_incumbent_trial_on_measured_cells(self) -> None:
        inc = [
            _rec('inc', 'd1', 3, [('g1', 'blocking')], [], 0.5),
            _rec('inc', 'd1', 1, [('g1', 'blocking')], [], 0.5),
            _rec('inc', 'd1', 2, [('g1', 'blocking'), ('g2', 'blocking')], [], 0.5),
            _rec('inc', 'd2', 1, [('g3', 'blocking')], [], 0.5),
            _unmeasured_rec('inc', 'd2', 2),
            _rec('inc', 'd2', 3, [('g3', 'blocking')], [], 0.5),
        ]
        arm = [
            _rec('arm', 'd1', 3, [('g1', 'blocking'), ('g2', 'blocking')], [], 0.5),
            _rec('arm', 'd1', 1, [('g1', 'blocking'), ('g2', 'blocking'), ('gx', 'blocking')], [], 0.5),
            _rec('arm', 'd1', 2, [('g1', 'blocking'), ('g2', 'blocking')], [], 0.5),
            _rec('arm', 'd1', 4, [('g1', 'blocking'), ('g2', 'blocking')], [], 0.5),
            _unmeasured_rec('arm', 'd2', 1),
            _rec('arm', 'd2', 2, [('g3', 'blocking')], [], 0.5),
            _rec('arm', 'd2', 3, [('g3', 'blocking')], [], 0.5),
        ]

        rows = campaign.complement_by_trial(arm, inc, BLOCKING_GT)

        assert rows == [
            {'trial': 1, 'complement': 2, 'incumbent_matched': 1, 'share': 2.0,
             'blocking_complement': 1, 'blocking_incumbent_matched': 1, 'blocking_share': 1.0},
            {'trial': 2, 'complement': 0, 'incumbent_matched': 2, 'share': 0.0,
             'blocking_complement': 0, 'blocking_incumbent_matched': 2, 'blocking_share': 0.0},
            {'trial': 3, 'complement': 1, 'incumbent_matched': 2, 'share': 0.5,
             'blocking_complement': 1, 'blocking_incumbent_matched': 2, 'blocking_share': 0.5},
        ]

    def test_passes_when_three_trials_each_beat_the_bands_blocking_maximum(self) -> None:
        rows = [_trial_row(1, 0.3), _trial_row(2, 0.25), _trial_row(3, 0.21)]

        assert campaign.clears_self_complement_band(rows, _band(0.2)) is True

    @pytest.mark.parametrize(
        ('rows', 'blocking_max'),
        [
            pytest.param([_trial_row(1, 0.3), _trial_row(2, 0.2), _trial_row(3, 0.3)], 0.2, id='one-at-the-max'),
            pytest.param([_trial_row(1, 0.3), _trial_row(2, 0.1), _trial_row(3, 0.3)], 0.2, id='one-below'),
            pytest.param([_trial_row(1, 0.3), _trial_row(2, None), _trial_row(3, 0.3)], 0.2, id='undefined-share'),
            pytest.param([_trial_row(t, 0.9) for t in (1, 2, 3)], None, id='no-band'),
            pytest.param(
                [_trial_row(t, 0.15, share=0.5) for t in (1, 2, 3)], 0.2,
                id='high-overall-share-but-blocking-inside-the-band',
            ),
            pytest.param([], 0.2, id='no-trials'),
        ],
    )
    def test_fails_closed(self, rows: list[dict], blocking_max: float | None) -> None:
        assert campaign.clears_self_complement_band(rows, _band(blocking_max)) is False

    def test_fewer_than_three_trials_fails_unless_the_floor_is_lowered(self) -> None:
        rows = [_trial_row(1, 0.3), _trial_row(2, 0.3)]

        assert campaign.clears_self_complement_band(rows, _band(0.2)) is False
        assert campaign.clears_self_complement_band(rows, _band(0.2), min_trials=2) is True


def _scored(variant: str, diff: str, trial: int, *, br: float, f1: float) -> dict:
    """A measured score record whose blocking recall and F1 differ."""
    return _rec(variant, diff, trial, [], [], f1) | {'blocking_recall': br}


def _refused(variant: str, diff: str, trial: int) -> dict:
    return _unmeasured_rec(variant, diff, trial)


def _three_trials(variant: str, scores: dict[str, tuple[float, float]]) -> list[dict]:
    """Every trial 1..3 with the same per-fixture ``(blocking_recall, f1)``."""
    return [
        _scored(variant, diff, t, br=br, f1=f1)
        for t in (1, 2, 3) for diff, (br, f1) in scores.items()
    ]


class TestNonInferior:
    """The D2 gate: blocking recall not below and F1 not worse, per trial, paired, measured."""

    def test_passes_when_every_trial_holds_both_metrics_ties_included(self) -> None:
        inc = _three_trials('inc', {'d1': (0.5, 0.5), 'd2': (0.5, 0.4)})
        arm = _three_trials('arm', {'d1': (0.5, 0.5), 'd2': (0.6, 0.4)})

        assert campaign.non_inferior(arm, inc) is True

    @pytest.mark.parametrize(
        ('metric', 'trial3'),
        [
            pytest.param('blocking_recall', {'br': 0.4, 'f1': 0.9}, id='lower-blocking-recall'),
            pytest.param('f1', {'br': 0.9, 'f1': 0.4}, id='lower-f1'),
        ],
    )
    def test_fails_when_one_trial_is_lower_on_either_metric(
        self, metric: str, trial3: dict[str, float],
    ) -> None:
        inc = _three_trials('inc', {'d1': (0.5, 0.5)})
        arm = [
            _scored('arm', 'd1', 1, br=0.9, f1=0.9),
            _scored('arm', 'd1', 2, br=0.9, f1=0.9),
            _scored('arm', 'd1', 3, **trial3),
        ]

        assert campaign.non_inferior(arm, inc) is False, metric

    def test_fails_with_fewer_than_three_shared_trials(self) -> None:
        inc = _three_trials('inc', {'d1': (0.5, 0.5)})
        arm = [_scored('arm', 'd1', t, br=0.9, f1=0.9) for t in (1, 2)]

        assert campaign.non_inferior(arm, inc) is False
        assert campaign.non_inferior(arm, inc, min_trials=2) is True

    def test_an_arm_is_judged_on_the_fixtures_it_shares_with_the_incumbent(self) -> None:
        inc = _three_trials('inc', {'d1': (0.5, 0.5), 'd2': (0.9, 0.9)})
        skipped_d2 = _three_trials('arm', {'d1': (0.6, 0.6)})

        assert campaign.non_inferior(skipped_d2, inc) is True

    def test_a_poor_incumbent_score_on_a_fixture_the_arm_lacks_does_not_count(self) -> None:
        inc = _three_trials('inc', {'d1': (0.5, 0.5), 'd2': (0.0, 0.0)})
        skipped_d2 = _three_trials('arm', {'d1': (0.4, 0.4)})

        assert campaign.non_inferior(skipped_d2, inc) is False

    def test_a_refused_incumbent_cell_drops_its_pair_rather_than_reading_as_zero(self) -> None:
        inc = [
            r for r in _three_trials('inc', {'d1': (0.5, 0.5), 'd2': (0.5, 0.5)})
            if (r['trial'], r['diff_id']) != (1, 'd1')
        ] + [_refused('inc', 'd1', 1)]
        arm = [
            r for r in _three_trials('arm', {'d1': (0.5, 0.5), 'd2': (0.5, 0.5)})
            if r['trial'] != 1
        ] + [_scored('arm', 'd1', 1, br=0.9, f1=0.9), _scored('arm', 'd2', 1, br=0.4, f1=0.4)]

        assert campaign.non_inferior(arm, inc) is False

    def test_a_refused_arm_cell_drops_its_pair_rather_than_reading_as_zero(self) -> None:
        inc = _three_trials('inc', {'d1': (0.5, 0.5), 'd2': (0.5, 0.5)})
        arm = [
            r for r in _three_trials('arm', {'d1': (0.5, 0.5), 'd2': (0.6, 0.6)})
            if (r['trial'], r['diff_id']) != (1, 'd1')
        ] + [_refused('arm', 'd1', 1)]

        assert campaign.non_inferior(arm, inc) is True

    def test_an_arm_with_nothing_measured_fails(self) -> None:
        inc = _three_trials('inc', {'d1': (0.5, 0.5)})
        arm = [_refused('arm', 'd1', t) for t in (1, 2, 3)]

        assert campaign.non_inferior(arm, inc) is False
        assert campaign.non_inferior([], inc) is False

    def test_records_without_the_unmeasured_key_are_measured(self) -> None:
        inc = _three_trials('inc', {'d1': (0.5, 0.5)})
        arm = _three_trials('arm', {'d1': (0.5, 0.5)})
        assert not any('unmeasured' in r for r in inc + arm)

        assert campaign.non_inferior(arm, inc) is True

    def test_the_campaign_decides_and_reports_it_per_arm(self) -> None:
        inc = _three_trials('inc', {'d1': (0.5, 0.5)})
        holds = _three_trials('holds', {'d1': (0.5, 0.6)})
        slips = _three_trials('slips', {'d1': (0.4, 0.9)})

        arms, decisions = campaign.summarize_campaign(inc + holds + slips, 'inc', BLOCKING_GT)

        assert decisions.non_inferiority == {'holds': True, 'slips': False}
        lines = campaign.format_markdown(arms, decisions).splitlines()
        holds_line = next(line for line in lines if line.startswith('- `holds` non-inferior'))
        slips_line = next(line for line in lines if line.startswith('- `slips` non-inferior'))
        assert '**PASS**' in holds_line
        assert '**FAIL**' in slips_line
        assert not any(line.startswith('- `inc` non-inferior') for line in lines)
        assert campaign.to_json(arms, decisions)['decisions']['non_inferiority'] == {
            'holds': True, 'slips': False,
        }
