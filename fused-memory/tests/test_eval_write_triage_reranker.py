"""Tests for eval_write_triage_reranker.py — the ρ1 reranker measurement core.

Every arm here is a fake, and every expected number is hand-computable from the
literals in the test. Nothing touches the network, torch or a live store.
"""
from __future__ import annotations

import functools
import types
from pathlib import Path

from _fm_helpers import load_script_module

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.grouped_read import PARENT_ID_KEY, SIGHTING_KIND

SCRIPTS = Path(__file__).parent.parent / 'scripts'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'eval_write_triage_reranker.py', 'eval_write_triage_reranker')


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
