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


class TestTopicPinDedupeAndMoveToFront:
    """The canonical is pinned ONCE, and never rebuilt when it is already present."""

    @staticmethod
    def _canonical_as_search_hit(n: int = 9) -> dict:
        """The canonical as it appears in the COSINE results — ranked LAST.

        This is the measured inversion itself: the canonical is a genuine
        semantic match, just the WORST one in its own cluster, because it is
        long and general where every sibling is narrow and tight.
        """
        return {
            'id': _CANONICAL_ID,
            'memory': _CANONICAL_BODY,
            'score': round(0.89 - 0.01 * n, 4),
            'metadata': {
                'category': 'procedural_knowledge',
                'topic': _TOPIC,
                'canonical': True,
            },
        }

    @pytest.mark.asyncio
    async def test_already_present_canonical_is_moved_not_duplicated(self, service):
        """RED before the dedupe branch: an unconditional insert duplicates it.

        Uses limit=10 deliberately.  At limit=5 a duplicate at rank 9 is
        truncated back off by ``results[:limit]``, so the count would read 1
        whether or not the dedupe branch exists — the assertion would pass for
        the wrong reason and pin nothing.
        """
        service.mem0.search = AsyncMock(return_value={'results': [
            *[_sibling(n) for n in range(1, 9)],
            self._canonical_as_search_hit(),
        ]})
        service.mem0.scroll_by_metadata = AsyncMock(return_value=[_canonical_payload()])

        results = await service.search(
            query=_QUERY, project_id=_PROJECT_ID,
            categories=['procedural_knowledge'], stores=['mem0'], limit=10,
        )

        assert [r.id for r in results].count(_CANONICAL_ID) == 1
        assert len(results) == 9, 'a duplicate insert would make this 10'
        assert results[0].id == _CANONICAL_ID
        assert results[0].topic_anchored is True

    @pytest.mark.asyncio
    async def test_moved_result_keeps_its_original_score_and_metadata(self, service):
        """Move, do not REBUILD: the record's own measured cosine is real data.

        Zeroing or replacing it would discard the honest per-store signal that
        the near-duplicate guard and every score-reading consumer depend on.
        """
        service.mem0.search = AsyncMock(return_value={'results': [
            *[_sibling(n) for n in range(1, 9)],
            self._canonical_as_search_hit(),
        ]})
        service.mem0.scroll_by_metadata = AsyncMock(return_value=[_canonical_payload()])

        results = await service.search(
            query=_QUERY, project_id=_PROJECT_ID,
            categories=['procedural_knowledge'], stores=['mem0'], limit=5,
        )

        pinned = results[0]
        # Its REAL cosine, stamped by _search_mem0 — not the scroll payload's
        # (absent) score, and not a zero.
        assert pinned.metadata['store_score'] == pytest.approx(0.80)
        assert pinned.metadata['store_rank'] == 9
        assert pinned.relevance_score > 0.0

    @pytest.mark.asyncio
    async def test_nothing_is_displaced_when_anchor_was_already_inside_the_window(self, service):
        """A move-to-front costs no sibling its slot — only a fresh insert does.

        The canonical ranks 3rd here, i.e. genuinely INSIDE the limit=5 window,
        so all four siblings survive alongside it.  Contrast
        ``test_window_stays_exactly_limit_and_displaces_the_last_sibling``,
        where the canonical is absent and the insert costs sibling-5 its slot.
        """
        hits = [
            _sibling(1), _sibling(2), self._canonical_as_search_hit(n=3),
            _sibling(4), _sibling(5),
        ]
        service.mem0.search = AsyncMock(return_value={'results': hits})
        service.mem0.scroll_by_metadata = AsyncMock(return_value=[_canonical_payload()])

        results = await service.search(
            query=_QUERY, project_id=_PROJECT_ID,
            categories=['procedural_knowledge'], stores=['mem0'], limit=5,
        )

        assert len(results) == 5
        assert [r.id for r in results] == [
            _CANONICAL_ID, 'sibling-1', 'sibling-2', 'sibling-4', 'sibling-5',
        ]

    @pytest.mark.asyncio
    async def test_non_pinned_remainder_keeps_relative_order(self, service):
        """A move-to-front must not otherwise perturb the ranking."""
        hits = [*[_sibling(n) for n in range(1, 4)], self._canonical_as_search_hit(n=4)]
        service.mem0.search = AsyncMock(return_value={'results': hits})
        service.mem0.scroll_by_metadata = AsyncMock(return_value=[_canonical_payload()])

        results = await service.search(
            query=_QUERY, project_id=_PROJECT_ID,
            categories=['procedural_knowledge'], stores=['mem0'], limit=10,
        )

        assert [r.id for r in results] == [
            _CANONICAL_ID, 'sibling-1', 'sibling-2', 'sibling-3',
        ]

    @pytest.mark.asyncio
    async def test_two_topics_resolving_to_one_canonical_pin_it_once(self, service):
        """Distinct topics may legitimately share a canonical; it is still ONE pin."""
        service.mem0.search = AsyncMock(return_value={'results': [
            _sibling(1, topic='topic-a'),
            _sibling(2, topic='topic-b'),
            _sibling(3, topic='topic-a'),
        ]})
        # Both topic lookups resolve to the SAME canonical record.
        service.mem0.scroll_by_metadata = AsyncMock(return_value=[_canonical_payload()])

        results = await service.search(
            query=_QUERY, project_id=_PROJECT_ID,
            categories=['procedural_knowledge'], stores=['mem0'], limit=10,
        )

        assert service.mem0.scroll_by_metadata.await_count == 2
        assert [r.id for r in results].count(_CANONICAL_ID) == 1
        assert results[0].id == _CANONICAL_ID

    @pytest.mark.asyncio
    async def test_two_topics_with_distinct_canonicals_pin_both(self, service):
        """The cap is on TOPICS, not on pins — two real canonicals both land."""
        service.mem0.search = AsyncMock(return_value={'results': [
            _sibling(1, topic='topic-a'),
            _sibling(2, topic='topic-b'),
        ]})
        service.mem0.scroll_by_metadata = AsyncMock(side_effect=[
            [_canonical_payload('canonical-a', topic='topic-a')],
            [_canonical_payload('canonical-b', topic='topic-b')],
        ])

        results = await service.search(
            query=_QUERY, project_id=_PROJECT_ID,
            categories=['procedural_knowledge'], stores=['mem0'], limit=10,
        )

        # TOPIC-RANK order: topic-a outranks topic-b in the result window, so
        # its canonical is pinned ahead of topic-b's.
        assert [r.id for r in results[:2]] == ['canonical-a', 'canonical-b']
        assert all(r.topic_anchored for r in results[:2])


class TestTopicPinScoreContract:
    """The pin must not create — or suppress — a near-duplicate write block.

    WHY THIS IS PINNED, in production terms. Since task 3658
    ``relevance_score`` is an ordinal RRF value (rank-1 ~ 0.0164), NOT a
    cosine; the honest per-store cosine lives in ``metadata['store_score']``,
    and ``near_duplicate_guard.find_near_duplicate_memory`` reads it from there
    and qualifies on ``>= threshold``. The near-dup guard's pre-check is
    exactly ``search(categories=['procedural_knowledge'], stores=['mem0'],
    limit=5)``, so EVERY procedural_knowledge write now runs through the
    anchoring block.

    An injected anchor that carried a synthetic ``store_score >= threshold``
    would therefore hard-block every ``procedural_knowledge`` write on a
    consolidated topic — turning a retrieval fix into a WRITE OUTAGE on
    precisely the topics it exists to help. These tests use the production
    selector as the oracle rather than asserting an implementation detail.
    """

    @staticmethod
    async def _anchored_results(service, *, hot_sibling: bool = False):
        """The step-9 scenario's result list, run through the real search tail."""
        hits = [_sibling(n) for n in range(1, 10)]
        if hot_sibling:
            # A sibling that IS a genuine near-duplicate of the pending write.
            hits[0] = {**hits[0], 'score': 0.97}
        service.mem0.search = AsyncMock(return_value={'results': hits})
        service.mem0.scroll_by_metadata = AsyncMock(return_value=[_canonical_payload()])
        return await service.search(
            query=_QUERY, project_id=_PROJECT_ID,
            categories=['procedural_knowledge'], stores=['mem0'], limit=5,
        )

    @pytest.mark.asyncio
    async def test_pin_does_not_create_a_near_duplicate_match(self, service):
        """Every sibling is safely below 0.92 — the guard must still find nothing."""
        from fused_memory.server.near_duplicate_guard import find_near_duplicate_memory

        results = await self._anchored_results(service)

        assert find_near_duplicate_memory(results, 0.92) is None

    @pytest.mark.asyncio
    async def test_pin_does_not_suppress_a_real_near_duplicate_match(self, service):
        """The paired positive: a genuinely-similar sibling is still selected."""
        from fused_memory.server.near_duplicate_guard import find_near_duplicate_memory

        results = await self._anchored_results(service, hot_sibling=True)

        match = find_near_duplicate_memory(results, 0.92)
        assert match is not None
        assert match.id == 'sibling-1'

    @pytest.mark.asyncio
    async def test_injected_anchor_carries_no_store_score_at_all(self, service):
        """ABSENT is not the same as 0.0 — and only absent is safe.

        ``near_duplicate_guard._cosine_of`` treats a missing cosine as "not
        comparable", which can never qualify at ANY threshold; a measured 0.0
        would still be a measurement, and a low threshold could clear it. This
        assertion is what stops a future refactor 'helpfully' copying a score
        onto the anchor.
        """
        results = await self._anchored_results(service)
        anchor = results[0]

        assert anchor.topic_anchored is True
        assert 'store_score' not in anchor.metadata
        assert 'store_rank' not in anchor.metadata

    @pytest.mark.asyncio
    async def test_injected_anchor_relevance_score_is_zero(self, service):
        """The pin is by ORDER ONLY — never by an inflated score."""
        results = await self._anchored_results(service)

        assert results[0].relevance_score == 0.0
