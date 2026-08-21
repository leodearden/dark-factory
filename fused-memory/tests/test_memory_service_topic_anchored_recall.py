"""Service-seam tests for the topic-anchored canonical pin (task 3111).

THE USER-OBSERVABLE SIGNAL. Consolidating a near-duplicate cluster into one
canonical makes that canonical the LEAST retrievable member of its own
cluster: it is long, general and multi-claim, so its embedding sits far from
any narrow query, while every surviving narrow sibling stays a tight cosine
match. At ``limit=5`` the window fills with siblings and the record that
actually answers the question never appears. These tests assert that the
canonical now arrives at index 0 — and that it arrives by ORDER, never by a
synthetic score (see ``TestTopicPinScoreContract``).

The promoting pin is the arm SELECTED BY MEASUREMENT in task 4004
(``plans/read-transform-selection-report.md``, ``recommendation.arm =
promoting_pin``), not chosen by argument here.

The pure selectors are unit-tested in ``tests/test_topic_anchor.py``; the MCP
composition proof lives in ``tests/server/test_topic_anchored_recall_mcp.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import install_identity_mocks

from fused_memory.models.enums import SourceStore
from fused_memory.services.memory_service import MemoryService

_PROJECT_ID = 'dark_factory'
_TOPIC = 'harness-layout-gate-decision-rule'
_CANONICAL_ID = 'canonical-0001'
_CANONICAL_BODY = (
    'The consolidated canonical for the harness layout gate decision rule, '
    'carrying every claim the narrow siblings each carry only one of.'
)
_QUERY = 'which harness layout does the gate decision rule accept'


@pytest.fixture
def service(mock_config):
    """MemoryService with both backends mocked — no real DB needed.

    Mirrors tests/test_memory_service.py's ``service`` fixture, trimmed to the
    surface this module exercises. The service is REAL: the point is to run the
    production ``search`` tail (sort -> category filter -> anchor -> truncate),
    not a re-implementation of it.
    """
    svc = MemoryService(mock_config)

    svc.graphiti = MagicMock()
    svc.graphiti.search = AsyncMock(return_value=[])
    svc.graphiti.search_nodes = AsyncMock(return_value=[])
    install_identity_mocks(svc.graphiti)

    svc.mem0 = MagicMock()
    svc.mem0.search = AsyncMock(return_value={'results': []})
    svc.mem0.scroll_by_metadata = AsyncMock(return_value=[])
    return svc


def _sibling(n: int, *, topic: str | None = _TOPIC, category: str = 'procedural_knowledge') -> dict:
    """One narrow sibling as Mem0Backend.search really returns it.

    ``score`` is the raw COSINE; ``_search_mem0`` moves it to
    ``metadata['store_score']`` and puts an ordinal RRF value in
    ``relevance_score`` (task 3658). Cosines here are high — these are tight
    matches — but all safely below the 0.92 near-dup threshold.
    """
    metadata: dict = {'category': category}
    if topic is not None:
        metadata['topic'] = topic
    return {
        'id': f'sibling-{n}',
        'memory': f'narrow sibling {n} on the harness layout gate',
        'score': round(0.89 - 0.01 * n, 4),
        'metadata': metadata,
    }


def _canonical_payload(
    id_: str = _CANONICAL_ID,
    *,
    topic: str = _TOPIC,
    category: str = 'procedural_knowledge',
    **extra: object,
) -> dict:
    """The raw scroll dict get_memories_by_metadata returns.

    NOTE the body sits under the raw payload key ``'data'``, NOT ``'memory'``:
    a scroll payload is a FULL raw Qdrant payload, not a Mem0 *search item*, so
    the ``_MEM0_CONTENT_KEYS`` fallback order ('data' -> 'memory' -> 'content')
    is what turns it into text. Crucially it carries NO ``store_score`` — see
    TestTopicPinScoreContract for why that absence is load-bearing.
    """
    metadata: dict = {
        'canonical': True,
        'topic': topic,
        'category': category,
        'data': _CANONICAL_BODY,
    }
    metadata.update(extra)
    return {
        'id': id_,
        'created_at': '2026-08-09T00:00:00+00:00',
        'metadata': metadata,
    }


def _seed(service, *, siblings: int = 9, scroll: list[dict] | None = None) -> None:
    """Nine tight siblings in the store, one canonical reachable only by scroll."""
    service.mem0.search = AsyncMock(
        return_value={'results': [_sibling(n) for n in range(1, siblings + 1)]}
    )
    service.mem0.scroll_by_metadata = AsyncMock(
        return_value=[_canonical_payload()] if scroll is None else scroll
    )


class TestPromotingTopicPin:
    """The headline: the canonical reaches index 0 of a limit=5 window."""

    @pytest.mark.asyncio
    async def test_canonical_is_promoted_to_index_zero(self, service):
        """FAILS BEFORE THIS TASK: at limit=5 the canonical is never returned at all."""
        _seed(service)

        results = await service.search(
            query=_QUERY,
            project_id=_PROJECT_ID,
            categories=['procedural_knowledge'],
            stores=['mem0'],
            limit=5,
        )

        assert results[0].id == _CANONICAL_ID
        assert results[0].topic_anchored is True
        assert results[0].source_store is SourceStore.mem0

    @pytest.mark.asyncio
    async def test_canonical_content_comes_from_the_raw_data_key(self, service):
        """A scroll payload is not a search item — item.get('memory') does not apply.

        The body is under ``'data'``; reading it requires the
        ``_MEM0_CONTENT_KEYS`` fallback order, so this pins that the pin uses
        the shared helper rather than guessing a key.
        """
        _seed(service)

        results = await service.search(
            query=_QUERY,
            project_id=_PROJECT_ID,
            categories=['procedural_knowledge'],
            stores=['mem0'],
            limit=5,
        )

        assert results[0].content == _CANONICAL_BODY

    @pytest.mark.asyncio
    async def test_window_stays_exactly_limit_and_displaces_the_last_sibling(self, service):
        """PROMOTING, not appending — the whole reason 4004 rejected the additive arm.

        At an already-full window an appended 6th result is truncated straight
        back off by ``final = results[:limit]``, so the additive shape delivers
        nothing. The promoting pin displaces the LOWEST-ranked sibling instead;
        that displacement is exactly the measured cost 4004 priced in and
        accepted (baseline retention 0.83, displacement 1.71 authored).
        """
        _seed(service)

        results = await service.search(
            query=_QUERY,
            project_id=_PROJECT_ID,
            categories=['procedural_knowledge'],
            stores=['mem0'],
            limit=5,
        )

        assert len(results) == 5
        # Four siblings survive alongside the pin, in their original rank order.
        assert [r.id for r in results[1:]] == ['sibling-1', 'sibling-2', 'sibling-3', 'sibling-4']

    @pytest.mark.asyncio
    async def test_canonical_appears_exactly_once(self, service):
        """Nine siblings share ONE topic; the canonical must not be pinned nine times."""
        _seed(service)

        results = await service.search(
            query=_QUERY,
            project_id=_PROJECT_ID,
            categories=['procedural_knowledge'],
            stores=['mem0'],
            limit=5,
        )

        assert [r.id for r in results].count(_CANONICAL_ID) == 1

    @pytest.mark.asyncio
    async def test_lookup_is_made_once_with_the_write_side_filter_shape(self, service):
        """Same two-key filter the write-side uniqueness check uses (memory_service.py:955).

        Nine siblings carry the SAME topic, so a per-result lookup would make
        nine round-trips; the de-duplication in ``extract_anchor_topics`` is
        what makes it one.
        """
        _seed(service)

        await service.search(
            query=_QUERY,
            project_id=_PROJECT_ID,
            categories=['procedural_knowledge'],
            stores=['mem0'],
            limit=5,
        )

        assert service.mem0.scroll_by_metadata.await_count == 1
        _scope, filters, _limit = service.mem0.scroll_by_metadata.await_args.args
        assert filters == {'topic': _TOPIC, 'canonical': True}
