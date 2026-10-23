"""A superseded CHILD keeps its supersession mark wherever it is nested.

Grouping folds an amendment or sighting into its parent's ``grouped`` block, so
a mark that rode only on top-level hits would vanish for every grouped child,
and a reader would take the retired child as current.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from fused_memory.memory_metadata import SUPERSEDED_BY_METADATA_KEY
from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.grouped_read import (
    MATCHED_CHILDREN_KEY,
    build_grouped_document,
    group_memory_document,
    group_search_results,
)

PROJECT_ID = 'dark_factory'
P = '11111111-1111-4111-8111-111111111111'
A1 = '22222222-2222-4222-8222-222222222221'
S1 = '33333333-3333-4333-8333-333333333331'
NEW_ID = '0f6c2a4e-8b1d-4c3e-9a7f-5d2e1b0c9a8f'
MARK = {SUPERSEDED_BY_METADATA_KEY: NEW_ID}


def _row(memory_id: str, kind: str, text: str, **extra: Any) -> dict[str, Any]:
    """A scrolled child row: ``metadata`` is the full raw Qdrant payload."""
    return {
        'id': memory_id,
        'created_at': None,
        'metadata': {'data': text, 'parent_id': P, 'kind': kind, **extra},
    }


def _service(rows: list[dict[str, Any]]) -> AsyncMock:
    """Answers the three Mem0 read primitives from *rows*."""

    def _matching(filters: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            row for row in rows
            if all(row['metadata'].get(key) == value for key, value in filters.items())
        ]

    async def count(*, project_id: str, filters: dict[str, Any]) -> int:
        return len(_matching(filters))

    async def scroll(
        *, project_id: str, filters: dict[str, Any], limit: int = 1000,
    ) -> list[dict[str, Any]]:
        return _matching(filters)[:limit]

    async def get_by_id(*, project_id: str, memory_id: str) -> dict[str, Any] | None:
        return None

    service = AsyncMock()
    service.count_memories_by_metadata = AsyncMock(side_effect=count)
    service.get_memories_by_metadata = AsyncMock(side_effect=scroll)
    service.get_memory_by_id = AsyncMock(side_effect=get_by_id)
    return service


def _hit(memory_id: str, content: str, score: float, metadata: dict | None = None) -> MemoryResult:
    return MemoryResult(
        id=memory_id,
        content=content,
        source_store=SourceStore.mem0,
        relevance_score=score,
        metadata=metadata or {},
    )


def _entry_for(entries: list[dict[str, Any]], memory_id: str) -> dict[str, Any]:
    [entry] = [entry for entry in entries if entry.get('id') == memory_id]
    return entry


class TestNestedEntriesKeepTheSupersessionMark:
    @pytest.mark.asyncio
    async def test_a_digest_carries_the_mark(self) -> None:
        service = _service([_row(A1, 'amendment', 'old amendment', **MARK)])

        block = await build_grouped_document(service, PROJECT_ID, P)

        assert block is not None
        assert _entry_for(block['amendments'], A1)['metadata'] == MARK

    @pytest.mark.asyncio
    async def test_the_mark_rides_beside_an_origin_tag(self) -> None:
        service = _service([_row(A1, 'amendment', 'old amendment', src_project='reify', **MARK)])

        block = await build_grouped_document(service, PROJECT_ID, P)

        assert block is not None
        assert _entry_for(block['amendments'], A1)['metadata'] == {'src_project': 'reify', **MARK}

    @pytest.mark.asyncio
    async def test_a_swallowed_sighting_is_pinned_with_its_mark(self) -> None:
        service = _service([_row(S1, 'sighting', 'old sighting', **MARK)])
        hits = [
            _hit(P, 'the canonical claim', 0.9),
            _hit(S1, 'old sighting', 0.8, {'kind': 'sighting', 'parent_id': P, **MARK}),
        ]

        entries = await group_search_results(service, PROJECT_ID, hits)

        assert [entry['id'] for entry in entries] == [P]
        pinned = _entry_for(entries[0]['grouped'][MATCHED_CHILDREN_KEY], S1)
        assert pinned['metadata'] == MARK

    @pytest.mark.asyncio
    async def test_a_matched_in_cap_amendment_keeps_its_mark(self) -> None:
        service = _service([_row(A1, 'amendment', 'old amendment', **MARK)])
        hits = [
            _hit(P, 'the canonical claim', 0.9),
            _hit(A1, 'old amendment', 0.8, {'kind': 'amendment', 'parent_id': P, **MARK}),
        ]

        entries = await group_search_results(service, PROJECT_ID, hits)

        assert [entry['id'] for entry in entries] == [P]
        digest = _entry_for(entries[0]['grouped']['amendments'], A1)
        assert digest['matched'] is True
        assert digest['metadata'] == MARK

    @pytest.mark.asyncio
    async def test_the_parents_point_read_carries_the_mark(self) -> None:
        service = _service([_row(A1, 'amendment', 'old amendment', **MARK)])
        parent_record = {
            'id': P,
            'content': 'the canonical claim',
            'metadata': {'data': 'the canonical claim'},
        }

        block = await group_memory_document(service, PROJECT_ID, P, parent_record)

        assert block is not None
        assert block['amendments'][0]['metadata'] == MARK

    @pytest.mark.asyncio
    @pytest.mark.parametrize('mark', ['', '   ', 123], ids=['empty', 'blank', 'not-a-str'])
    async def test_an_unreadable_mark_emits_no_metadata(self, mark) -> None:
        service = _service([
            _row(A1, 'amendment', 'old amendment', **{SUPERSEDED_BY_METADATA_KEY: mark}),
        ])

        block = await build_grouped_document(service, PROJECT_ID, P)

        assert block is not None
        assert 'metadata' not in _entry_for(block['amendments'], A1)
