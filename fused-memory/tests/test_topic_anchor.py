"""Unit tests for the topic-anchored canonical recall transform (task 3111).

Covers the ``MemoryResult.topic_anchored`` retrieval-state field and the pure,
synchronous selectors in ``services/topic_anchor.py``.  The service-seam
integration lives in ``test_memory_service_topic_anchored_recall.py``; the MCP
composition proof lives in ``tests/server/test_topic_anchored_recall_mcp.py``.
"""

from __future__ import annotations

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.services.topic_anchor import extract_anchor_topics


def _result(
    id_: str = 'm1',
    *,
    topic: object = 'harness-layout-gate-decision-rule',
    source_store: SourceStore = SourceStore.mem0,
    category: MemoryCategory | None = MemoryCategory.procedural_knowledge,
    content: str = 'a sibling record on the consolidated topic',
) -> MemoryResult:
    """Build a real post-RRF MemoryResult, mirroring _near_duplicate_result.

    *topic* is placed in ``metadata['topic']`` verbatim — including sentinel
    ``...`` to mean "omit the key entirely" — so the selector's defensive
    type handling is exercised against real Pydantic objects, not dicts.
    """
    metadata: dict = {}
    if topic is not ...:
        metadata['topic'] = topic
    return MemoryResult(
        id=id_,
        content=content,
        category=category,
        source_store=source_store,
        metadata=metadata,
    )


class TestMemoryResultTopicAnchoredField:
    """Pin ``topic_anchored`` as a real, always-present field on MemoryResult."""

    def test_defaults_to_false_when_not_supplied(self):
        """An ordinary search result is not topic-anchored unless the pin says so."""
        result = MemoryResult(
            id='m1',
            content='an ordinary ranked result',
            category=MemoryCategory.procedural_knowledge,
            source_store=SourceStore.mem0,
        )

        assert result.topic_anchored is False

    def test_true_round_trips_through_model_dump(self):
        """The search MCP tool surfaces results via model_dump(), so the flag must survive it."""
        result = MemoryResult(
            id='m1',
            content='the canonical, promoted into the window by rule',
            category=MemoryCategory.procedural_knowledge,
            source_store=SourceStore.mem0,
            topic_anchored=True,
        )

        assert result.topic_anchored is True
        assert result.model_dump()['topic_anchored'] is True

    def test_key_present_in_model_dump_even_when_false(self):
        """Consumers may read the key unconditionally — it is never omitted."""
        result = MemoryResult(
            id='m1',
            content='an ordinary ranked result',
            category=MemoryCategory.procedural_knowledge,
            source_store=SourceStore.mem0,
        )

        dumped = result.model_dump()
        assert 'topic_anchored' in dumped
        assert dumped['topic_anchored'] is False


class TestExtractAnchorTopics:
    """The pure, synchronous topic selector that decides what to look up."""

    def test_returns_distinct_mem0_topics_in_rank_order(self):
        """First-seen (rank) order is the contract — results arrive already sorted."""
        results = [
            _result('m1', topic='topic-a'),
            _result('m2', topic='topic-b'),
            _result('m3', topic='topic-c'),
        ]

        assert extract_anchor_topics(results, max_topics=3) == ['topic-a', 'topic-b', 'topic-c']

    def test_ignores_graphiti_sourced_results(self):
        """metadata.topic is a Mem0 vocabulary key; a Graphiti row can never anchor."""
        results = [
            _result('g1', topic='graphiti-topic', source_store=SourceStore.graphiti),
            _result('m1', topic='mem0-topic'),
        ]

        assert extract_anchor_topics(results, max_topics=3) == ['mem0-topic']

    def test_ignores_missing_empty_and_non_str_topics(self):
        """Raw payload values are not schema-enforced at read time — degrade, never raise."""
        results = [
            _result('m1', topic=...),
            _result('m2', topic=''),
            _result('m3', topic=7),
            _result('m4', topic=['topic-a']),
            _result('m5', topic=None),
            _result('m6', topic=True),
            _result('m7', topic='real-topic'),
        ]

        assert extract_anchor_topics(results, max_topics=5) == ['real-topic']

    def test_deduplicates_repeats_preserving_first_seen_order(self):
        """Nine siblings on one topic must cost ONE lookup, not nine."""
        results = [
            _result('m1', topic='topic-a'),
            _result('m2', topic='topic-b'),
            _result('m3', topic='topic-a'),
            _result('m4', topic='topic-b'),
            _result('m5', topic='topic-c'),
        ]

        assert extract_anchor_topics(results, max_topics=5) == ['topic-a', 'topic-b', 'topic-c']

    def test_truncates_to_max_topics(self):
        """The cap is a blast-radius bound on backend round-trips per search."""
        results = [_result(f'm{i}', topic=f'topic-{i}') for i in range(10)]

        assert extract_anchor_topics(results, max_topics=3) == ['topic-0', 'topic-1', 'topic-2']

    def test_empty_input_returns_empty_list(self):
        """The zero-cost path: no results means no topics means no I/O."""
        assert extract_anchor_topics([], max_topics=3) == []

    def test_no_topics_returns_empty_list(self):
        """The common case on the live corpus today — must short-circuit to no I/O."""
        results = [_result('m1', topic=...), _result('m2', topic=...)]

        assert extract_anchor_topics(results, max_topics=3) == []
