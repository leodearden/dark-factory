"""The link adjudicator's eval against blind majority verdicts (task 6184, PRD H3).

The pure core is tested through the script's public functions, loaded by path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _fm_helpers import load_script_module

from fused_memory.maintenance.link_adjudicator import AdjudicationFailure, LinkVerdict
from fused_memory.maintenance.link_heal import Verdict

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'eval_link_adjudicator.py'

ev = load_script_module(SCRIPT_PATH, mod_name='eval_link_adjudicator')

FIGURES = (
    'misfile_recall',
    'false_detach_rate',
    'corrects_recall',
    'false_corrects_rate',
    'parse_failure_rate',
)


def _item(key: str, truth, *, kind: str | None = None, verdict: Verdict | None = None):
    return ev.ScoredItem(
        key=key,
        child_text=f'child {key}',
        parent_text=f'parent {key}',
        truth=truth,
        kind_at_rating=kind,
        verdict=verdict,
    )


def _said(key: str, verdict: Verdict) -> LinkVerdict:
    return LinkVerdict(key, 'arm', verdict=verdict, reason='r')


def _failed(key: str) -> LinkVerdict:
    return LinkVerdict(key, 'arm', failure=AdjudicationFailure.PARSE_FAILURE, detail='d')


T = ev.TruthClass


class TestTruthClass:
    @pytest.mark.parametrize(
        ('verdict', 'truth'),
        [
            (Verdict.RELATED, T.MISFILE),
            (Verdict.UNRELATED, T.MISFILE),
            (Verdict.CORRECTS, T.CORRECTS),
            (Verdict.SAME, T.AGREEING),
            (Verdict.EXTENDS, T.AGREEING),
            (Verdict.SUBSUMED, T.AGREEING),
            (Verdict.UNCLEAR, T.UNCLEAR),
        ],
    )
    def test_every_verdict_word_has_one_truth_class(self, verdict, truth):
        assert ev.truth_class(verdict) is truth


class TestScoreArm:
    def _scored(self) -> dict:
        items = [
            _item('m1', T.MISFILE), _item('m2', T.MISFILE), _item('m3', T.MISFILE),
            _item('c1', T.CORRECTS), _item('c2', T.CORRECTS),
            _item('a1', T.AGREEING), _item('a2', T.AGREEING), _item('a3', T.AGREEING),
            _item('u1', T.UNCLEAR),
        ]
        verdicts = [
            _said('m1', Verdict.RELATED), _said('m2', Verdict.UNRELATED), _failed('m3'),
            _said('c1', Verdict.CORRECTS), _failed('c2'),
            _said('a1', Verdict.RELATED), _said('a2', Verdict.CORRECTS), _said('a3', Verdict.SAME),
            _said('u1', Verdict.SAME),
        ]
        return ev.score_arm(items, verdicts)

    def test_the_five_figures_are_confusion_counts_against_the_majority(self):
        scored = self._scored()

        assert (scored['misfile_recall_num'], scored['misfile_recall_den']) == (2, 3)
        assert (scored['false_detach_rate_num'], scored['false_detach_rate_den']) == (1, 5)
        assert (scored['corrects_recall_num'], scored['corrects_recall_den']) == (1, 2)
        assert (scored['false_corrects_rate_num'], scored['false_corrects_rate_den']) == (1, 3)
        assert (scored['parse_failure_rate_num'], scored['parse_failure_rate_den']) == (2, 9)
        assert scored['misfile_recall'] == pytest.approx(2 / 3)
        assert scored['parse_failure_rate'] == pytest.approx(2 / 9)

    def test_each_figure_carries_its_counts_and_a_wilson_interval(self):
        scored = self._scored()

        for name in FIGURES:
            assert isinstance(scored[name], float)
            assert type(scored[f'{name}_num']) is int
            assert type(scored[f'{name}_den']) is int
            low, high = scored[f'{name}_ci95']
            assert 0.0 <= low <= scored[name] <= high <= 1.0

    def test_a_zero_denominator_gives_none_not_zero(self):
        scored = ev.score_arm([_item('a1', T.AGREEING)], [_said('a1', Verdict.SAME)])

        assert scored['misfile_recall'] is None
        assert scored['misfile_recall_ci95'] is None
        assert scored['misfile_recall_den'] == 0
        assert scored['corrects_recall'] is None
        assert scored['false_detach_rate'] == 0.0


class TestWilson95:
    def test_nine_of_fifteen(self):
        low, high = ev.wilson95(9, 15)

        assert low == pytest.approx(0.357, abs=1e-3)
        assert high == pytest.approx(0.802, abs=1e-3)

    def test_no_trials_is_none(self):
        assert ev.wilson95(0, 0) is None


class TestKindAgreement:
    def test_an_arm_agrees_when_it_heals_the_kind_the_majority_heals(self):
        items = [
            _item('s1', T.AGREEING, kind='sighting', verdict=Verdict.EXTENDS),
            _item('s2', T.AGREEING, kind='sighting', verdict=Verdict.SAME),
            _item('h1', T.MISFILE, kind=None, verdict=Verdict.RELATED),
            _item('h2', T.AGREEING, kind='correction', verdict=Verdict.EXTENDS),
            _item('am', T.AGREEING, kind='amendment', verdict=Verdict.EXTENDS),
            _item('pe', T.AGREEING, kind='peer', verdict=Verdict.SAME),
        ]
        verdicts = [
            _said('s1', Verdict.EXTENDS),
            _said('s2', Verdict.EXTENDS),
            _said('h1', Verdict.UNRELATED),
            _failed('h2'),
            _said('am', Verdict.RELATED),
            _said('pe', Verdict.RELATED),
        ]

        agreement = ev.kind_agreement(items, verdicts)

        assert (agreement['kind_agreement_num'], agreement['kind_agreement_den']) == (2, 4)
        assert agreement['kind_agreement'] == pytest.approx(0.5)
        baseline = (
            agreement['kind_agreement_always_amendment_baseline_num'],
            agreement['kind_agreement_always_amendment_baseline_den'],
        )
        assert baseline == (2, 4)


class TestSelectArm:
    def _row(self, arm: str, recall, detach, corrects, cost: float) -> dict:
        return {
            'arm': arm,
            'misfile_recall': recall,
            'false_detach_rate': detach,
            'false_corrects_rate': corrects,
            'cost_usd': cost,
            'nested': {'arm': arm},
        }

    def test_the_highest_misfile_recall_wins(self):
        rows = [self._row('a', 0.5, 0.0, 0.0, 1.0), self._row('b', 0.8, 0.1, 0.1, 9.0)]

        assert ev.select_arm(rows)['arm'] == 'b'

    @pytest.mark.parametrize(
        ('rows', 'winner'),
        [
            ([('a', 0.8, 0.02, 0.0, 1.0), ('b', 0.8, 0.01, 0.5, 9.0)], 'b'),
            ([('a', 0.8, 0.01, 0.10, 1.0), ('b', 0.8, 0.01, 0.05, 9.0)], 'b'),
            ([('a', 0.8, 0.01, 0.05, 2.0), ('b', 0.8, 0.01, 0.05, 1.0)], 'b'),
            ([('a', None, 0.0, 0.0, 0.0), ('b', 0.1, 0.9, 0.9, 9.0)], 'b'),
            ([('a', 0.8, None, 0.0, 0.0), ('b', 0.8, 0.9, 0.9, 9.0)], 'b'),
        ],
    )
    def test_ties_break_on_detach_then_flag_then_cost_and_none_ranks_worst(self, rows, winner):
        assert ev.select_arm([self._row(*row) for row in rows])['arm'] == winner

    def test_the_selection_is_a_deep_copy_of_the_chosen_row(self):
        rows = [self._row('a', 0.5, 0.0, 0.0, 1.0)]

        selection = ev.select_arm(rows)

        assert selection == rows[0]
        selection['nested']['arm'] = 'changed'
        assert rows[0]['nested']['arm'] == 'a'


def _vote(entry: str, target: str, verdict: str, rater: str = 'r0') -> dict:
    return {'entry_id': entry, 'target_id': target, 'verdict': verdict, 'rater': rater, 'batch': 'b'}


def _pair(entry: str, target: str, entry_text: str | None = 'child', target_text: str | None = 'parent') -> dict:
    return {'entry_id': entry, 'entry_text': entry_text, 'target_id': target, 'target_text': target_text}


class TestTriageItems:
    def test_votes_resolve_to_truth_and_join_their_texts(self):
        votes = [
            _vote('e1', 't1', 'RELATED'),
            _vote('e2', 't2', 'RELATED'), _vote('e2', 't2', 'EXTENDS', rater='r1'),
            _vote('e3', 't3', 'SAME'),
            _vote('e4', 't4', 'EXTENDS'),
            _vote('e5', 't5', 'CORRECTS'),
            _vote('e6', 't6', 'UNCLEAR'),
        ]
        pairs = [
            _pair('e1', 't1', 'child one', 'parent one'),
            _pair('e2', 't2'),
            _pair('e4', 't4', target_text=None),
            _pair('e5', 't5'),
            _pair('e6', 't6'),
            _pair('e7', 't7'),
        ]

        population = ev.triage_items(votes, pairs)

        truths = {item.key: item.truth for item in population.items}
        assert truths == {'e1:t1': T.MISFILE, 'e5:t5': T.CORRECTS, 'e6:t6': T.UNCLEAR}
        first = next(item for item in population.items if item.key == 'e1:t1')
        assert (first.child_text, first.parent_text) == ('child one', 'parent one')
        assert dict(population.excluded) == {
            ev.Exclusion.TIED: 1,
            ev.Exclusion.NO_TEXTS: 2,
            ev.Exclusion.UNRATED: 1,
        }

    @pytest.mark.parametrize('missing', ['entry_text', 'target_text'])
    def test_a_pairs_row_without_a_text_key_is_refused(self, missing):
        row = _pair('e1', 't1')
        del row[missing]

        with pytest.raises(ValueError, match=missing):
            ev.triage_items([_vote('e1', 't1', 'SAME')], [row])


class TestBuildReport:
    def test_the_report_states_its_population_arms_selection_and_provenance(self):
        population = ev.Population(
            items=(
                _item('m', T.MISFILE), _item('c', T.CORRECTS),
                _item('a', T.AGREEING), _item('u', T.UNCLEAR),
            ),
            excluded={ev.Exclusion.TIED: 2, ev.Exclusion.UNRATED: 1},
        )
        arm = {'arm': 'fake', 'misfile_recall': 1.0, 'false_detach_rate': 0.0,
               'false_corrects_rate': 0.0, 'cost_usd': 0.0}

        report = ev.build_report(
            population, [arm],
            mode=ev.Mode.TRIAGE,
            corpus_sha256='c' * 64,
            pairs_sha256='p' * 64,
            brief_sha256='b' * 64,
            field_chars=4000,
            shard_size=40,
        )

        assert report['population'] == {
            'n_pairs': 4,
            'n_misfile': 1,
            'n_corrects': 1,
            'n_belongs': 2,
            'n_unclear': 1,
            'excluded': 3,
            'excluded_by_reason': {'tied': 2, 'unrated': 1},
        }
        assert report['arms'] == [arm]
        assert report['selection'] == arm
        provenance = report['provenance']
        assert provenance['mode'] == 'triage'
        assert provenance['corpus_sha256'] == 'c' * 64
        assert provenance['pairs_sha256'] == 'p' * 64
        assert provenance['brief_sha256'] == 'b' * 64
        assert (provenance['field_chars'], provenance['shard_size']) == (4000, 40)
        assert provenance['text_keys'] == {'child': 'entry_text', 'parent': 'target_text'}
        assert 'write_triage_pairs_to_rate.jsonl' in provenance['text_key_basis']
