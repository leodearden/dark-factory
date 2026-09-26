"""Tests for eval_write_triage_reranker.py — the ρ1 reranker measurement core.

Every arm here is a fake, and every expected number is hand-computable from the
literals in the test. Nothing touches the network, torch or a live store.
"""
from __future__ import annotations

import contextlib
import functools
import types
from pathlib import Path

import pytest
from _fm_helpers import load_script_module

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.grouped_read import PARENT_ID_KEY, SIGHTING_KIND

SCRIPTS = Path(__file__).parent.parent / 'scripts'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'eval_write_triage_reranker.py', 'eval_write_triage_reranker')


@functools.cache
def _arms() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'eval_write_triage_reranker_arms.py', 'eval_write_triage_reranker_arms',
    )


def _record(memory_id: str, cluster_id: str, label: str = 'duplicate') -> dict:
    return {
        'memory_id': memory_id, 'cluster_id': cluster_id, 'label': label,
        'content': f'the entry {memory_id}',
    }


def _row(memory_id: str, cosine: float | None, **metadata) -> MemoryResult:
    """A post-RRF search row: the cosine lives in ``metadata['store_score']``."""
    return MemoryResult(
        id=memory_id, content=f'the text of {memory_id}',
        category=MemoryCategory.procedural_knowledge, source_store=SourceStore.mem0,
        metadata={'store_score': cosine, **metadata},
    )


def _child(memory_id: str, cosine: float, parent_id: str) -> MemoryResult:
    return _row(memory_id, cosine, kind=SIGHTING_KIND, **{PARENT_ID_KEY: parent_id})


def _retrieval(
    rows: list[MemoryResult],
    *,
    canonical_present: bool = True,
    degraded: bool = False,
    self_retrieved: bool = False,
) -> dict:
    """One entry of ``prefetch_retrievals``' output."""
    return {
        'results': list(rows), 'canonical_present': canonical_present,
        'degraded': degraded, 'self_retrieved': self_retrieved,
    }


def _case(record: dict, rows: list[MemoryResult], **flags):
    [case] = _mod().build_cases([record], {record['memory_id']: _retrieval(rows, **flags)})
    return case


def _rank1(cases: list, scores: list, aliases: dict | None = None):
    return _mod().ranking_metrics(cases, scores, aliases=aliases).rank1


class TestBuildCases:
    def test_one_case_per_record_carrying_its_slate_in_retrieval_order(self) -> None:
        records = [_record('d1', 'c1'), _record('d2', 'c2', label='distinct')]
        retrievals = {
            'd1': _retrieval([_row('x', 0.4), _row('c1', 0.9)]),
            'd2': _retrieval([_row('c2', 0.7)]),
        }
        first, second = _mod().build_cases(records, retrievals)
        assert (first.memory_id, first.label, first.entry) == ('d1', 'duplicate', 'the entry d1')
        assert first.canonical_id == 'c1'
        assert first.candidate_ids == ('x', 'c1')
        assert first.candidate_texts == ('the text of x', 'the text of c1')
        assert first.candidate_cosines == (0.4, 0.9)
        assert (second.memory_id, second.label, second.canonical_id) == ('d2', 'distinct', 'c2')
        assert second.candidate_ids == ('c2',)

    def test_only_children_map_to_their_hoisted_parent(self) -> None:
        case = _case(_record('d1', 'c1'), [_row('x', 0.9), _child('s', 0.8, 'c1')])
        assert case.candidate_parents == {'s': 'c1'}

    def test_retrieval_flags_are_carried_through(self) -> None:
        case = _case(
            _record('d1', 'gone'), [],
            canonical_present=False, degraded=True, self_retrieved=True,
        )
        assert case.canonical_present is False
        assert case.degraded is True
        assert case.self_retrieved is True
        assert case.candidate_ids == ()


class TestRankingMetrics:
    def test_the_top_scored_candidate_is_rank_one_whatever_its_retrieval_position(self) -> None:
        case = _case(_record('d1', 'c1'), [_row('x', 0.9), _row('y', 0.8), _row('c1', 0.7)])
        rank1 = _rank1([case], [[0.1, 0.2, 0.9]])
        assert (rank1.hits, rank1.total, rank1.rate) == (1, 1, 1.0)

    def test_tied_scores_keep_retrieval_order(self) -> None:
        behind = _case(_record('d1', 'c1'), [_row('x', 0.9), _row('c1', 0.8)])
        ahead = _case(_record('d2', 'c1'), [_row('c1', 0.9), _row('x', 0.8)])
        assert _rank1([behind], [[0.5, 0.5]]).hits == 0
        assert _rank1([ahead], [[0.5, 0.5]]).hits == 1

    def test_a_sighting_child_of_the_canonical_reaches_it(self) -> None:
        case = _case(_record('d1', 'c1'), [_row('x', 0.9), _child('s', 0.8, 'c1')])
        assert _rank1([case], [[0.1, 0.9]]).hits == 1

    def test_the_canonicals_alias_reaches_it(self) -> None:
        case = _case(
            _record('d1', 'old'), [_row('x', 0.9), _row('new', 0.8)], canonical_present=False,
        )
        assert _rank1([case], [[0.1, 0.9]], aliases={'old': 'new'}).hits == 1

    def test_an_absent_unreached_canonical_is_a_miss_in_the_denominator(self) -> None:
        hit = _case(_record('d1', 'c1'), [_row('c1', 0.9)])
        absent = _case(_record('d2', 'gone'), [_row('x', 0.9)], canonical_present=False)
        rank1 = _rank1([hit, absent], [[0.9], [0.9]], aliases={'other': 'z'})
        assert (rank1.hits, rank1.total, rank1.rate) == (1, 2, 0.5)

    def test_rank_five_counts_a_hit_anywhere_in_the_top_five(self) -> None:
        ids = ['a', 'b', 'c', 'd', 'e', 'f', 'c1']
        case = _case(_record('d1', 'c1'), [_row(i, 0.5) for i in ids])
        fifth = _mod().ranking_metrics([case], [[7, 6, 5, 4, 2, 1, 3]], aliases=None)
        sixth = _mod().ranking_metrics([case], [[7, 6, 5, 4, 3, 2, 1]], aliases=None)
        assert (fifth.rank1.hits, fifth.rank5.hits) == (0, 1)
        assert (sixth.rank1.hits, sixth.rank5.hits) == (0, 0)

    def test_no_cases_is_an_unmeasured_rate_not_a_zero(self) -> None:
        metrics = _mod().ranking_metrics([], [], aliases={'a': 'b'})
        assert (metrics.rank1.hits, metrics.rank1.total, metrics.rank1.rate) == (0, 0, None)
        assert metrics.rank5.rate is None


class TestAucTrueVsHardNegative:
    @pytest.mark.parametrize(('positives', 'negatives', 'expected'), [
        ([0.9, 0.8], [0.1], 1.0),
        ([0.1], [0.9], 0.0),
        ([0.5], [0.5], 0.5),
        ([0.9, 0.2], [0.5], 0.5),
    ])
    def test_it_is_the_probability_a_positive_outscores_a_negative(
        self, positives: list, negatives: list, expected: float,
    ) -> None:
        assert _mod().auc_true_vs_hard_negative(positives, negatives) == expected

    @pytest.mark.parametrize(('positives', 'negatives'), [([], [0.5]), ([0.5], []), ([], [])])
    def test_an_empty_class_is_unmeasured_not_zero(self, positives: list, negatives: list) -> None:
        assert _mod().auc_true_vs_hard_negative(positives, negatives) is None


class TestRankingMetricsAuc:
    def test_pair_scores_split_by_label_and_an_unreached_canonical_is_unscored(self) -> None:
        reached = _case(_record('d1', 'c1'), [_row('x', 0.9), _row('c1', 0.8)])
        negative = _case(_record('n1', 'c2', label='distinct'), [_row('c2', 0.9), _row('y', 0.8)])
        unreached = _case(_record('d2', 'c3'), [_row('z', 0.9)])
        auc = _mod().ranking_metrics(
            [reached, negative, unreached], [[0.1, 0.9], [0.2, 0.8], [0.7]], aliases=None,
        ).auc
        assert (auc.value, auc.n_true, auc.n_hard_negative, auc.unscored) == (1.0, 1, 1, 1)

    def test_the_pair_score_is_the_first_reaching_candidate_in_the_arms_order(self) -> None:
        child_first = _case(
            _record('d1', 'c1'), [_row('c1', 0.9), _child('s', 0.8, 'c1')],
        )
        negative = _case(
            _record('n1', 'c2', label='pseudo_contradiction'), [_row('c2', 0.9)],
        )
        auc = _mod().ranking_metrics(
            [child_first, negative], [[0.2, 0.9], [0.5]], aliases=None,
        ).auc
        assert auc.value == 1.0

    def test_an_alias_reaches_the_canonical_for_the_pair_score(self) -> None:
        aliased = _case(
            _record('d1', 'old'), [_row('new', 0.9)], canonical_present=False,
        )
        negative = _case(_record('n1', 'c2', label='distinct'), [_row('c2', 0.9)])
        auc = _mod().ranking_metrics(
            [aliased, negative], [[0.9], [0.1]], aliases={'old': 'new'},
        ).auc
        assert (auc.value, auc.n_true, auc.unscored) == (1.0, 1, 0)

    def test_no_hard_negative_leaves_the_auc_unmeasured(self) -> None:
        case = _case(_record('d1', 'c1'), [_row('c1', 0.9)])
        auc = _mod().ranking_metrics([case], [[0.9]], aliases=None).auc
        assert (auc.value, auc.n_true, auc.n_hard_negative) == (None, 1, 0)


class TestBaselineScores:
    def test_the_baseline_score_is_each_candidates_store_score(self) -> None:
        case = _case(_record('d1', 'c1'), [_row('x', 0.4), _row('c1', 0.9)])
        assert tuple(_mod().baseline_scores(case)) == (0.4, 0.9)

    def test_the_baseline_ranks_by_cosine_not_retrieval_order(self) -> None:
        case = _case(_record('d1', 'c1'), [_row('x', 0.4), _row('c1', 0.9)])
        assert _rank1([case], [_mod().baseline_scores(case)]).hits == 1

    def test_a_missing_cosine_ranks_last(self) -> None:
        case = _case(_record('d1', 'c1'), [_row('c1', None), _row('x', 0.3)])
        metrics = _mod().ranking_metrics([case], [_mod().baseline_scores(case)], aliases=None)
        assert (metrics.rank1.hits, metrics.rank5.hits) == (0, 1)


class _FakeScorer:
    """Answers each entry from *answers*; ``calls`` lists the entries it was asked about."""

    def __init__(self, answers: dict, *, facts=None, fail_on_call: int | None = None) -> None:
        self.answers = answers
        self.calls: list[str] = []
        self.fail_on_call = fail_on_call
        self._facts = facts or _arms().ScorerFacts(device='cpu', vram_peak_mib=None, max_length=None)

    def score(self, entry: str, candidate_texts):
        self.calls.append(entry)
        if self.fail_on_call == len(self.calls):
            raise RuntimeError('boom')
        return self.answers[entry]

    def facts(self):
        return self._facts


def _spec(scorer: _FakeScorer | None = None, *, name: str = 'fake', unavailable=None, log=None):
    arms = _arms()

    @contextlib.contextmanager
    def open_(context):
        if unavailable is not None:
            raise unavailable
        try:
            yield scorer
        finally:
            if log is not None:
                log.append('closed')

    return arms.ArmSpec(
        name=name, arm_class=arms.ArmClass.local_cross_encoder, model='fake-model', open=open_,
    )


def _context():
    return _arms().ArmContext(
        device='cpu', local_batch_size=4, vram_cap_gib=8.0, pairwise_concurrency=20,
    )


def _slate(scores: tuple, cost: float | None = 0.0, over: int | None = 0):
    return _arms().SlateScores(scores=scores, cost_usd=cost, pairs_over_max_length=over)


def _measure(spec, cases: list, clock: list, *, max_spend_usd: float = 10.0, aliases=None):
    return _mod().measure_arm(
        spec, cases, aliases=aliases, context=_context(),
        clock=iter(clock).__next__, max_spend_usd=max_spend_usd,
    )


_METRIC_KEYS = (
    'rank1_rate', 'rank5_rate', 'rank1', 'rank5', 'auc', 'p50_seconds', 'p95_seconds',
    'latency', 'cost_per_write_usd', 'device', 'vram_peak_mib', 'max_length',
    'pairs_over_max_length',
)


class TestMeasureArm:
    @staticmethod
    def _cases() -> list:
        return [
            _case(_record('d3', 'c4'), []),
            _case(_record('d1', 'c1'), [_row('x', 0.9), _row('c1', 0.8)]),
            _case(
                _record('n1', 'c2', label='distinct'),
                [_row('c2', 0.9), _row('y', 0.8), _row('z', 0.7)],
            ),
            _case(_record('d2', 'c3'), [_row('w', 0.9)]),
        ]

    @staticmethod
    def _answers(*, b_cost: float | None = 0.5, b_over: int | None = 0) -> dict:
        return {
            'the entry d1': _slate((0.2, 0.9), cost=0.25, over=1),
            'the entry n1': _slate((0.1, 0.8, 0.3), cost=b_cost, over=b_over),
            'the entry d2': _slate((0.6,), cost=0.75, over=2),
        }

    _CLOCK = [0.0, 1.5, 10.0, 10.25, 20.0, 20.5, 30.0, 31.0]

    def test_a_measured_row_carries_ranking_latency_cost_and_facts(self) -> None:
        facts = _arms().ScorerFacts(device='cuda:0 fake', vram_peak_mib=512.0, max_length=8192)
        scorer = _FakeScorer(self._answers(), facts=facts)
        row = _measure(_spec(scorer), self._cases(), self._CLOCK).to_json()
        assert (row['arm'], row['arm_class'], row['model']) == (
            'fake', 'local_cross_encoder', 'fake-model',
        )
        assert (row['status'], row['skip_reason'], row['skip_detail']) == ('measured', None, None)
        assert (row['rank1_rate'], row['rank1']) == (0.25, {'hits': 1, 'total': 4})
        assert (row['rank5_rate'], row['rank5']) == (0.5, {'hits': 2, 'total': 4})
        assert row['auc'] == {'value': 1.0, 'n_true': 1, 'n_hard_negative': 1, 'unscored': 2}
        assert (row['p50_seconds'], row['p95_seconds']) == (0.5, 1.0)
        assert row['latency'] == {
            'slates_timed': 3, 'pairs_per_slate_min': 1, 'pairs_per_slate_max': 3,
            'load_seconds': 1.5, 'warmup_slates': 1,
        }
        assert row['cost_per_write_usd'] == pytest.approx(0.5)
        assert row['pairs_over_max_length'] == 3
        assert (row['device'], row['vram_peak_mib'], row['max_length']) == (
            'cuda:0 fake', 512.0, 8192,
        )

    def test_the_first_non_empty_slate_warms_up_and_an_empty_slate_is_never_sent(self) -> None:
        scorer = _FakeScorer(self._answers())
        _measure(_spec(scorer), self._cases(), self._CLOCK)
        assert scorer.calls == ['the entry d1', 'the entry d1', 'the entry n1', 'the entry d2']

    def test_an_unpriced_slate_leaves_the_cost_unmeasured(self) -> None:
        scorer = _FakeScorer(self._answers(b_cost=None))
        assert _measure(_spec(scorer), self._cases(), self._CLOCK).to_json()[
            'cost_per_write_usd'
        ] is None

    def test_an_unreported_truncation_count_stays_unmeasured(self) -> None:
        scorer = _FakeScorer(self._answers(b_over=None))
        assert _measure(_spec(scorer), self._cases(), self._CLOCK).to_json()[
            'pairs_over_max_length'
        ] is None

    def test_an_unavailable_arm_is_skipped_with_every_metric_unmeasured(self) -> None:
        arms = _arms()
        spec = _spec(unavailable=arms.ArmUnavailable(
            arms.SkipReason.no_credential, 'JINA_API_KEY unset',
        ))
        row = _measure(spec, self._cases(), [0.0]).to_json()
        assert (row['status'], row['skip_reason'], row['skip_detail']) == (
            'skipped', 'no_credential', 'JINA_API_KEY unset',
        )
        assert {key: row[key] for key in _METRIC_KEYS} == dict.fromkeys(_METRIC_KEYS)

    def test_a_scorer_failure_mid_run_is_skipped_as_an_error_with_no_partial_numbers(self) -> None:
        log: list[str] = []
        scorer = _FakeScorer(self._answers(), fail_on_call=2)
        row = _measure(_spec(scorer, log=log), self._cases(), self._CLOCK).to_json()
        assert (row['status'], row['skip_reason']) == ('skipped', 'error')
        assert row['skip_detail'] == 'RuntimeError: boom'
        assert {key: row[key] for key in _METRIC_KEYS} == dict.fromkeys(_METRIC_KEYS)
        assert log == ['closed']

    def test_spend_past_the_ceiling_stops_the_arm_as_over_budget(self) -> None:
        scorer = _FakeScorer(self._answers())
        row = _measure(_spec(scorer), self._cases(), self._CLOCK, max_spend_usd=0.4).to_json()
        assert (row['status'], row['skip_reason']) == ('skipped', 'over_budget')
        assert '0.5000' in row['skip_detail']
        assert scorer.calls == ['the entry d1', 'the entry d1']
        assert {key: row[key] for key in _METRIC_KEYS} == dict.fromkeys(_METRIC_KEYS)

    def test_a_score_list_not_matching_the_slate_is_an_error_never_padded(self) -> None:
        answers = {**self._answers(), 'the entry n1': _slate((0.1, 0.8))}
        row = _measure(_spec(_FakeScorer(answers)), self._cases(), self._CLOCK).to_json()
        assert (row['status'], row['skip_reason']) == ('skipped', 'error')
        assert row['skip_detail'].startswith('ValueError: ')


def _arm_json(arm: str, rank1: float | None, p95: float | None, status: str = 'measured') -> dict:
    return {'arm': arm, 'status': status, 'rank1_rate': rank1, 'p95_seconds': p95}


def _best(rows: list, ceiling: float = 3.0) -> dict:
    return _mod().choose_best(rows, p95_ceiling_seconds=ceiling)


class TestChooseBest:
    def test_the_most_accurate_arm_under_the_ceiling_wins(self) -> None:
        rows = [
            _arm_json('fast', 0.3, 1.0), _arm_json('accurate', 0.5, 2.0),
            _arm_json('slow', 0.9, 4.0),
        ]
        assert _best(rows) == {
            'arm': 'accurate', 'rank1_rate': 0.5, 'p95_seconds': 2.0,
            'qualified': True, 'p95_ceiling_seconds': 3.0,
        }

    def test_a_p95_exactly_at_the_ceiling_qualifies(self) -> None:
        rows = [_arm_json('fast', 0.3, 1.0), _arm_json('edge', 0.6, 3.0)]
        assert (_best(rows)['arm'], _best(rows)['qualified']) == ('edge', True)

    def test_a_rank1_tie_goes_to_the_lower_p95(self) -> None:
        rows = [_arm_json('slower', 0.5, 2.0), _arm_json('quicker', 0.5, 1.0)]
        assert _best(rows)['arm'] == 'quicker'

    def test_a_full_tie_goes_to_the_earlier_row(self) -> None:
        rows = [_arm_json('first', 0.5, 1.0), _arm_json('second', 0.5, 1.0)]
        assert _best(rows)['arm'] == 'first'

    def test_with_nothing_under_the_ceiling_the_fastest_measured_arm_is_unqualified(self) -> None:
        rows = [_arm_json('slow', 0.9, 5.0), _arm_json('less_slow', 0.2, 4.0)]
        assert _best(rows) == {
            'arm': 'less_slow', 'rank1_rate': 0.2, 'p95_seconds': 4.0,
            'qualified': False, 'p95_ceiling_seconds': 3.0,
        }

    def test_with_nothing_measured_every_value_is_null_never_zero(self) -> None:
        rows = [_arm_json('a', None, None, 'skipped'), _arm_json('b', None, None, 'skipped')]
        assert _best(rows, ceiling=2.5) == {
            'arm': None, 'rank1_rate': None, 'p95_seconds': None,
            'qualified': False, 'p95_ceiling_seconds': 2.5,
        }

    def test_a_skipped_row_is_never_selected_even_carrying_numbers(self) -> None:
        rows = [_arm_json('stale', 0.99, 0.1, 'skipped'), _arm_json('real', 0.3, 1.0)]
        assert _best(rows)['arm'] == 'real'
        assert _best([_arm_json('stale', 0.99, 0.1, 'skipped')])['arm'] is None
