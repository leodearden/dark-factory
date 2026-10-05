"""Retrieval math (known-item rank, recall@k, MRR) and the guarded provenance probe."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from arm_harness._fakes import FakeArmGraph, llm_spec
from fused_memory.arm_harness.arm_spec import LlmArmSpec
from fused_memory.arm_harness.replay import EpisodeOutcome, ReplayItem
from fused_memory.arm_harness.retrieval import (
    RETRIEVAL_UTILITY_K,
    known_item_rank,
    mrr_metric,
    probe_retrieval_utility,
    provenance_matcher,
    recall_at_k,
    recall_metric,
)
from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, ScratchGuardError

RANKS = [1, 3, None, 12]


def _edge(*episode_uuids: str) -> SimpleNamespace:
    return SimpleNamespace(episodes=list(episode_uuids))


def _item(episode_id: str) -> ReplayItem:
    return ReplayItem(
        episode_id=episode_id,
        name=episode_id,
        content=f'body of {episode_id}',
        source_description='lme corpus',
        reference_time=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _outcome(episode_id: str, *, ok: bool = True) -> EpisodeOutcome:
    return EpisodeOutcome(
        episode_id=episode_id,
        ok=ok,
        error_class=None if ok else 'RuntimeError',
        duration_ms=10.0,
        tokens=None,
        replay_episode_uuid=f'replay-{episode_id}' if ok else None,
        entity_names=(),
        edge_triples=(),
    )


def test_retrieval_utility_k_is_ten():
    assert RETRIEVAL_UTILITY_K == 10


def test_known_item_rank_is_the_first_match_one_based():
    results = ['a', 'b', 'target', 'target']

    assert known_item_rank(results, lambda r: r == 'target') == 3


def test_known_item_rank_scans_the_whole_list():
    results = ['x'] * 50 + ['target']

    assert known_item_rank(results, lambda r: r == 'target') == 51


def test_known_item_rank_is_none_without_a_match():
    assert known_item_rank(['a', 'b'], lambda r: r == 'target') is None
    assert known_item_rank([], lambda r: True) is None


def test_recall_at_k_counts_ranks_within_k():
    assert recall_at_k(RANKS, 5) == (2, 4)
    assert recall_at_k(RANKS, 10) == (2, 4)
    assert recall_at_k(RANKS, 12) == (3, 4)
    assert recall_at_k([], 5) == (0, 0)


@pytest.mark.parametrize('ranks', [[0], [-1, 2]])
def test_ranks_are_one_based(ranks):
    with pytest.raises(ValueError, match='rank'):
        recall_at_k(ranks, 5)
    with pytest.raises(ValueError, match='rank'):
        mrr_metric(ranks)


def test_recall_at_k_rejects_a_non_positive_k():
    with pytest.raises(ValueError, match='k'):
        recall_at_k(RANKS, 0)


def test_recall_metric_is_an_m1_proportion():
    metric = recall_metric('known-item-recall@5', RANKS, 5)

    assert metric is not None
    assert metric.metric_id == 'known-item-recall@5'
    assert metric.kind == 'proportion'
    assert metric.direction == 'lower_is_worse'
    assert (metric.value, metric.n, metric.denominator) == (0.5, 4, 4)


def test_recall_metric_is_absent_without_ranks():
    assert recall_metric('retrieval-utility', [], RETRIEVAL_UTILITY_K) is None


def test_mrr_counts_misses_as_zero():
    metric = mrr_metric(RANKS)

    assert metric is not None
    assert metric.metric_id == 'mrr'
    assert metric.kind == 'scalar'
    assert metric.n == 4
    assert metric.value == pytest.approx((1 + 1 / 3 + 0 + 1 / 12) / 4)


def test_mrr_is_absent_without_ranks():
    assert mrr_metric([]) is None


def test_provenance_matcher_matches_results_citing_the_replayed_episode():
    matches = provenance_matcher('replay-ep-1')

    assert matches(_edge('other', 'replay-ep-1')) is True
    assert matches(_edge('other')) is False
    assert matches(SimpleNamespace()) is False


@pytest.mark.asyncio
async def test_probe_ranks_each_ok_episode_by_its_own_content():
    def search_results(query: str) -> list:
        by_query = {
            'body of ep-0': [_edge('replay-ep-0')],
            'body of ep-2': [_edge('noise'), _edge('noise'), _edge('replay-ep-2')],
        }
        return by_query.get(query, [_edge('noise')])

    spec = llm_spec()
    graph = FakeArmGraph(search_results=search_results)
    outcomes = [_outcome('ep-0'), _outcome('ep-1', ok=False), _outcome('ep-2'), _outcome('ep-3')]
    items = [_item(f'ep-{i}') for i in range(4)]

    ranks = await probe_retrieval_utility(graph, spec, outcomes, items=items)

    assert ranks == (1, 3, None)
    assert [call['query'] for call in graph.search_calls] == [
        'body of ep-0',
        'body of ep-2',
        'body of ep-3',
    ]
    for call in graph.search_calls:
        assert call['group_ids'] == [spec.scratch_group_id]
        assert call['num_results'] == RETRIEVAL_UTILITY_K


@pytest.mark.asyncio
async def test_probe_refuses_a_validation_bypassed_spec_before_any_search():
    valid = llm_spec()
    bypassed = LlmArmSpec.model_construct(**(dict(valid) | {'scratch_group_id': 'dark_factory'}))
    graph = FakeArmGraph()

    with pytest.raises(ScratchGuardError) as caught:
        await probe_retrieval_utility(graph, bypassed, [_outcome('ep-0')], items=[_item('ep-0')])

    assert caught.value.checkpoint is GuardCheckpoint.REPLAY
    assert graph.search_calls == []


@pytest.mark.asyncio
async def test_probe_refuses_an_outcome_without_its_replay_item():
    graph = FakeArmGraph()

    with pytest.raises(ValueError, match='ep-9'):
        await probe_retrieval_utility(graph, llm_spec(), [_outcome('ep-9')], items=[_item('ep-0')])

    assert graph.search_calls == []
