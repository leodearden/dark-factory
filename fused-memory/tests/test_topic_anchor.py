"""Unit tests for the topic-anchored canonical recall transform (task 3111).

Covers the ``MemoryResult.topic_anchored`` retrieval-state field and the pure,
synchronous selectors in ``services/topic_anchor.py``.  The service-seam
integration lives in ``test_memory_service_topic_anchored_recall.py``; the MCP
composition proof lives in ``tests/server/test_topic_anchored_recall_mcp.py``.
"""

from __future__ import annotations

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult


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
