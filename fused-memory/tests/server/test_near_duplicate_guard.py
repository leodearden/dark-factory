"""Unit tests for the pure near-duplicate selection helper (task 2467).

``find_near_duplicate_memory`` is a pure, synchronous function: given a list
of already-fetched ``MemoryResult`` search hits and a similarity threshold,
it selects the single best near-duplicate candidate (or ``None``). It does
no I/O and makes no assumptions about embedding accuracy — callers inject
explicit ``relevance_score`` values, so these tests exercise pure comparison
logic only.
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock

from fused_memory.config.schema import ProceduralTopicCluster
from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.near_duplicate_guard import (
    _DEFAULT_NEAR_DUP_GUARD_ENABLED,
    _DEFAULT_NEAR_DUP_THRESHOLD,
    _TOPIC_CLUSTER_DEFAULT_HINT,
    build_topic_cluster_block,
    find_matching_topic_cluster,
    find_near_duplicate_memory,
    resolve_near_dup_guard_enabled,
    resolve_near_dup_threshold,
    resolve_topic_guard_clusters,
)


def _result(
    id_: str,
    score: float,
    *,
    category: MemoryCategory | None = MemoryCategory.procedural_knowledge,
    source_store: SourceStore = SourceStore.mem0,
    content: str = 'some procedural content',
) -> MemoryResult:
    return MemoryResult(
        id=id_,
        content=content,
        category=category,
        source_store=source_store,
        relevance_score=score,
    )


class TestFindNearDuplicateMemory:
    """Pure selection logic for the write-time near-duplicate guard."""

    def test_returns_match_when_above_threshold(self):
        results = [_result('m1', 0.97)]
        match = find_near_duplicate_memory(results, 0.92)
        assert match is not None
        assert match.id == 'm1'

    def test_returns_none_when_all_below_threshold(self):
        results = [_result('m1', 0.50), _result('m2', 0.10)]
        match = find_near_duplicate_memory(results, 0.92)
        assert match is None

    def test_score_equal_to_threshold_is_a_match(self):
        results = [_result('m1', 0.92)]
        match = find_near_duplicate_memory(results, 0.92)
        assert match is not None
        assert match.id == 'm1'

    def test_empty_results_returns_none(self):
        match = find_near_duplicate_memory([], 0.92)
        assert match is None

    def test_ignores_wrong_category(self):
        results = [
            _result('m1', 0.99, category=MemoryCategory.observations_and_summaries),
        ]
        match = find_near_duplicate_memory(results, 0.92)
        assert match is None

    def test_ignores_wrong_source_store(self):
        results = [
            _result('m1', 0.99, source_store=SourceStore.graphiti),
        ]
        match = find_near_duplicate_memory(results, 0.92)
        assert match is None

    def test_picks_max_score_among_multiple_qualifying_results(self):
        results = [
            _result('m1', 0.93),
            _result('m2', 0.99),
            _result('m3', 0.95),
        ]
        match = find_near_duplicate_memory(results, 0.92)
        assert match is not None
        assert match.id == 'm2'

    def test_mixed_qualifying_and_disqualified_picks_best_qualifying(self):
        results = [
            _result('m1', 0.999, category=MemoryCategory.observations_and_summaries),
            _result('m2', 0.93),
            _result('m3', 0.999, source_store=SourceStore.graphiti),
        ]
        match = find_near_duplicate_memory(results, 0.92)
        assert match is not None
        assert match.id == 'm2'


def _memory_service_with_reconciliation(**fields) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        config=types.SimpleNamespace(reconciliation=types.SimpleNamespace(**fields))
    )


class TestResolveNearDupThreshold:
    """Defensive config resolver: real float from config, else module default."""

    def test_returns_config_value_when_present_and_valid(self):
        memory_service = _memory_service_with_reconciliation(
            procedural_knowledge_near_dup_threshold=0.80
        )
        assert resolve_near_dup_threshold(memory_service) == 0.80

    def test_falls_back_to_default_when_config_attr_missing(self):
        memory_service = types.SimpleNamespace()
        assert resolve_near_dup_threshold(memory_service) == _DEFAULT_NEAR_DUP_THRESHOLD

    def test_falls_back_to_default_when_reconciliation_is_none(self):
        memory_service = types.SimpleNamespace(config=types.SimpleNamespace(reconciliation=None))
        assert resolve_near_dup_threshold(memory_service) == _DEFAULT_NEAR_DUP_THRESHOLD

    def test_falls_back_to_default_when_value_is_non_numeric(self):
        # An unspecced AsyncMock's attribute chain yields Mock instances, not
        # floats — the resolver must not propagate one as a threshold.
        memory_service = AsyncMock()
        assert resolve_near_dup_threshold(memory_service) == _DEFAULT_NEAR_DUP_THRESHOLD

    def test_falls_back_to_default_when_value_is_bool(self):
        memory_service = _memory_service_with_reconciliation(
            procedural_knowledge_near_dup_threshold=True
        )
        assert resolve_near_dup_threshold(memory_service) == _DEFAULT_NEAR_DUP_THRESHOLD


class TestResolveNearDupGuardEnabled:
    """Defensive config resolver: real bool from config, else module default."""

    def test_returns_config_value_when_present_and_valid(self):
        memory_service = _memory_service_with_reconciliation(
            procedural_knowledge_near_dup_guard_enabled=False
        )
        assert resolve_near_dup_guard_enabled(memory_service) is False

    def test_falls_back_to_default_when_config_attr_missing(self):
        memory_service = types.SimpleNamespace()
        assert resolve_near_dup_guard_enabled(memory_service) == _DEFAULT_NEAR_DUP_GUARD_ENABLED

    def test_falls_back_to_default_when_reconciliation_is_none(self):
        memory_service = types.SimpleNamespace(config=types.SimpleNamespace(reconciliation=None))
        assert resolve_near_dup_guard_enabled(memory_service) == _DEFAULT_NEAR_DUP_GUARD_ENABLED

    def test_falls_back_to_default_when_value_is_non_bool(self):
        # An unspecced AsyncMock's attribute chain yields Mock instances, not
        # bools — the resolver must not propagate one as the enable flag.
        memory_service = AsyncMock()
        assert resolve_near_dup_guard_enabled(memory_service) == _DEFAULT_NEAR_DUP_GUARD_ENABLED


def _cluster(
    topic_id: str = 'topic-a',
    phrases: list[str] | None = None,
    min_phrase_hits: int = 2,
    sufficient_phrases: list[str] | None = None,
) -> ProceduralTopicCluster:
    return ProceduralTopicCluster(
        topic_id=topic_id,
        phrases=phrases if phrases is not None else ['alpha', 'beta', 'gamma'],
        min_phrase_hits=min_phrase_hits,
        sufficient_phrases=sufficient_phrases if sufficient_phrases is not None else [],
    )


class TestFindMatchingTopicCluster:
    """Pure deterministic substring matcher for the topic-keyed write guard (task 2845)."""

    def test_returns_cluster_and_phrases_when_min_hits_met(self):
        cluster = _cluster(phrases=['alpha', 'beta'], min_phrase_hits=2)
        match = find_matching_topic_cluster('some alpha and beta content', [cluster])
        assert match is not None
        matched_cluster, matched_phrases = match
        assert matched_cluster.topic_id == cluster.topic_id
        assert matched_phrases == ['alpha', 'beta']

    def test_returns_none_when_fewer_than_min_hits(self):
        cluster = _cluster(phrases=['alpha', 'beta'], min_phrase_hits=2)
        assert find_matching_topic_cluster('only alpha here', [cluster]) is None

    def test_empty_clusters_returns_none(self):
        assert find_matching_topic_cluster('alpha beta gamma', []) is None

    def test_matching_is_case_insensitive(self):
        cluster = _cluster(phrases=['alpha', 'beta'], min_phrase_hits=2)
        match = find_matching_topic_cluster('SOME ALPHA and BeTa', [cluster])
        assert match is not None
        _, matched_phrases = match
        assert matched_phrases == ['alpha', 'beta']

    def test_repeated_phrase_counts_once(self):
        # 'alpha' appears 3x but is ONE distinct phrase -> below min_phrase_hits=2.
        cluster = _cluster(phrases=['alpha', 'beta'], min_phrase_hits=2)
        assert find_matching_topic_cluster('alpha alpha alpha', [cluster]) is None

    def test_selects_correct_cluster_among_multiple(self):
        cluster_a = _cluster(topic_id='topic-a', phrases=['alpha', 'beta'], min_phrase_hits=2)
        cluster_b = _cluster(topic_id='topic-b', phrases=['delta', 'epsilon'], min_phrase_hits=2)
        match = find_matching_topic_cluster(
            'this content mentions delta and epsilon', [cluster_a, cluster_b]
        )
        assert match is not None
        matched_cluster, matched_phrases = match
        assert matched_cluster.topic_id == 'topic-b'
        assert matched_phrases == ['delta', 'epsilon']

    def test_empty_content_returns_none(self):
        cluster = _cluster(phrases=['alpha', 'beta'], min_phrase_hits=1)
        assert find_matching_topic_cluster('', [cluster]) is None

    def test_whitespace_content_returns_none(self):
        cluster = _cluster(phrases=['alpha', 'beta'], min_phrase_hits=1)
        assert find_matching_topic_cluster('   \n\t ', [cluster]) is None


class TestFindMatchingTopicClusterSufficientPhrases:
    """A single DECLARED-sufficient phrase qualifies its cluster alone (task 3054).

    Count-only matching cannot block a write that straddles several
    clusters at one hit each; a phrase distinctive enough that it cannot
    appear incidentally should not have to wait for a second one.
    """

    def test_single_sufficient_phrase_matches_below_min_phrase_hits(self):
        cluster = _cluster(
            phrases=['alpha', 'beta'],
            min_phrase_hits=2,
            sufficient_phrases=['alpha'],
        )
        match = find_matching_topic_cluster('only alpha here', [cluster])
        assert match is not None

    def test_sufficient_match_still_returns_the_two_tuple_shape(self):
        """The return shape is load-bearing: server/tools.py destructures a 2-tuple.

        ``matched_phrases`` is deliberately SHORTER than ``min_phrase_hits``
        here -- it reports exactly what fired, which is the one sufficient
        phrase.
        """
        cluster = _cluster(
            phrases=['alpha', 'beta'],
            min_phrase_hits=2,
            sufficient_phrases=['alpha'],
        )
        match = find_matching_topic_cluster('only alpha here', [cluster])
        assert match is not None
        matched_cluster, matched_phrases = match
        assert matched_cluster.topic_id == cluster.topic_id
        assert matched_phrases == ['alpha']
        assert len(matched_phrases) < cluster.min_phrase_hits

    def test_single_non_sufficient_phrase_still_returns_none(self):
        # 'beta' is a phrase but NOT declared sufficient -- one hit is still
        # below min_phrase_hits, so the count-only path must still apply.
        cluster = _cluster(
            phrases=['alpha', 'beta'],
            min_phrase_hits=2,
            sufficient_phrases=['alpha'],
        )
        assert find_matching_topic_cluster('only beta here', [cluster]) is None

    def test_sufficient_matching_is_case_insensitive_and_lowercases(self):
        # Same convention as test_matching_is_case_insensitive above.
        cluster = _cluster(
            phrases=['Alpha', 'beta'],
            min_phrase_hits=2,
            sufficient_phrases=['ALPHA'],
        )
        match = find_matching_topic_cluster('SOME AlPhA content', [cluster])
        assert match is not None
        _, matched_phrases = match
        assert matched_phrases == ['alpha']

    def test_empty_sufficient_phrases_behaves_exactly_as_today(self):
        cluster = _cluster(phrases=['alpha', 'beta'], min_phrase_hits=2, sufficient_phrases=[])
        assert find_matching_topic_cluster('only alpha here', [cluster]) is None
        match = find_matching_topic_cluster('alpha and beta', [cluster])
        assert match is not None
        _, matched_phrases = match
        assert matched_phrases == ['alpha', 'beta']

    def test_count_path_still_qualifies_when_no_sufficient_phrase_hits(self):
        # Reaching min_phrase_hits on NON-sufficient phrases must still match:
        # sufficiency is an additional way to qualify, not a replacement.
        cluster = _cluster(
            phrases=['alpha', 'beta', 'gamma'],
            min_phrase_hits=2,
            sufficient_phrases=['alpha'],
        )
        match = find_matching_topic_cluster('beta and gamma only', [cluster])
        assert match is not None
        _, matched_phrases = match
        assert matched_phrases == ['beta', 'gamma']

    def test_empty_content_returns_none_even_with_sufficient_phrases(self):
        cluster = _cluster(
            phrases=['alpha', 'beta'],
            min_phrase_hits=2,
            sufficient_phrases=['alpha'],
        )
        assert find_matching_topic_cluster('', [cluster]) is None

    def test_whitespace_content_returns_none_even_with_sufficient_phrases(self):
        cluster = _cluster(
            phrases=['alpha', 'beta'],
            min_phrase_hits=2,
            sufficient_phrases=['alpha'],
        )
        assert find_matching_topic_cluster('   \n\t ', [cluster]) is None

    def test_earlier_cluster_wins_on_a_sufficient_phrase(self):
        # First-qualifying-cluster ordering is unchanged by sufficiency.
        cluster_a = _cluster(
            topic_id='topic-a',
            phrases=['alpha', 'beta'],
            min_phrase_hits=2,
            sufficient_phrases=['alpha'],
        )
        cluster_b = _cluster(topic_id='topic-b', phrases=['alpha', 'delta'], min_phrase_hits=1)
        match = find_matching_topic_cluster('alpha only', [cluster_a, cluster_b])
        assert match is not None
        assert match[0].topic_id == 'topic-a'

    def test_earlier_count_match_wins_over_a_later_sufficient_match(self):
        """The converse precedence, pinned deliberately rather than left incidental.

        Sufficiency decides WHETHER a cluster qualifies, never WHICH
        qualifying cluster wins: the scan is a single pass in list order, so
        an earlier cluster reaching ``min_phrase_hits`` on ordinary phrases
        beats a later cluster whose declared-sufficient phrase also fired.
        Seed order stays the single priority knob.
        """
        cluster_a = _cluster(topic_id='topic-a', phrases=['alpha', 'beta'], min_phrase_hits=2)
        cluster_b = _cluster(
            topic_id='topic-b',
            phrases=['gamma'],
            min_phrase_hits=2,
            sufficient_phrases=['gamma'],
        )
        match = find_matching_topic_cluster('alpha beta gamma', [cluster_a, cluster_b])
        assert match is not None
        assert match[0].topic_id == 'topic-a'
        assert match[1] == ['alpha', 'beta']

    def test_reordering_the_same_clusters_hands_the_win_to_the_sufficient_one(self):
        """Confirms the previous test pins ORDER, not a count-beats-sufficiency rule."""
        cluster_a = _cluster(topic_id='topic-a', phrases=['alpha', 'beta'], min_phrase_hits=2)
        cluster_b = _cluster(
            topic_id='topic-b',
            phrases=['gamma'],
            min_phrase_hits=2,
            sufficient_phrases=['gamma'],
        )
        match = find_matching_topic_cluster('alpha beta gamma', [cluster_b, cluster_a])
        assert match is not None
        assert match[0].topic_id == 'topic-b'
        assert match[1] == ['gamma']


class TestResolveTopicGuardClusters:
    """Defensive config resolver: real list from config, else empty list."""

    def test_returns_config_list_when_present(self):
        clusters = [_cluster(topic_id='topic-a'), _cluster(topic_id='topic-b')]
        memory_service = _memory_service_with_reconciliation(
            procedural_knowledge_topic_guard_clusters=clusters
        )
        assert resolve_topic_guard_clusters(memory_service) == clusters

    def test_returns_empty_list_when_config_attr_missing(self):
        memory_service = types.SimpleNamespace()
        assert resolve_topic_guard_clusters(memory_service) == []

    def test_returns_empty_list_when_reconciliation_is_none(self):
        memory_service = types.SimpleNamespace(config=types.SimpleNamespace(reconciliation=None))
        assert resolve_topic_guard_clusters(memory_service) == []

    def test_returns_empty_list_when_value_is_non_list(self):
        # An unspecced AsyncMock's attribute chain yields Mock instances, not
        # lists — the resolver must not propagate one as a clusters list.
        memory_service = AsyncMock()
        assert resolve_topic_guard_clusters(memory_service) == []

    def test_returns_empty_list_when_config_default_is_empty(self):
        memory_service = _memory_service_with_reconciliation(
            procedural_knowledge_topic_guard_clusters=[]
        )
        assert resolve_topic_guard_clusters(memory_service) == []


class TestBuildTopicClusterBlock:
    """The structured soft-block dict returned by the add_memory topic gate (task 2845)."""

    def test_block_shape_and_fields(self):
        cluster = _cluster(topic_id='topic-a', phrases=['alpha', 'beta'])
        block = build_topic_cluster_block(
            'claude-interactive', 'some alpha beta content', cluster, ['alpha', 'beta']
        )
        assert block['error'] == 'procedural_knowledge_known_topic_cluster_write_blocked'
        assert block['error_type'] == 'ProceduralKnowledgeKnownTopicClusterWriteRejected'
        assert block['agent_id'] == 'claude-interactive'
        assert block['content_excerpt'] == 'some alpha beta content'
        assert block['topic_id'] == 'topic-a'
        assert block['matched_phrases'] == ['alpha', 'beta']
        assert block['hint']

    def test_content_excerpt_truncated_to_200(self):
        cluster = _cluster(topic_id='topic-a')
        long_content = 'x' * 500
        block = build_topic_cluster_block('agent', long_content, cluster, ['alpha'])
        assert block['content_excerpt'] == long_content[:200]
        assert len(block['content_excerpt']) == 200

    def test_uses_cluster_hint_when_set(self):
        cluster = ProceduralTopicCluster(
            topic_id='topic-a', phrases=['alpha', 'beta'], hint='route to gate 2841'
        )
        block = build_topic_cluster_block('agent', 'content', cluster, ['alpha', 'beta'])
        assert block['hint'] == 'route to gate 2841'

    def test_falls_back_to_default_hint_when_cluster_hint_empty(self):
        cluster = ProceduralTopicCluster(topic_id='topic-a', phrases=['alpha', 'beta'], hint='')
        block = build_topic_cluster_block('agent', 'content', cluster, ['alpha', 'beta'])
        assert block['hint'] == _TOPIC_CLUSTER_DEFAULT_HINT
        assert block['hint']

    def test_echoes_none_agent_id(self):
        cluster = _cluster(topic_id='topic-a')
        block = build_topic_cluster_block(None, 'content', cluster, ['alpha'])
        assert block['agent_id'] is None
