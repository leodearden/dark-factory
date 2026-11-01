"""A contesting child's text rides in its parent's grouped document in full, up to the judge's field cap, so a reader who matched only the parent still sees the correction."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from fused_memory.config.schema import WriteTriageConfig
from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.grouped_read import (
    AMENDMENT_KIND,
    CONTESTED_METADATA_KEY,
    build_grouped_document,
    group_memory_document,
    group_search_results,
)
from fused_memory.utils.clip_disclosure import elision_marker

PROJECT_ID = 'dark_factory'
P = '11111111-1111-4111-8111-111111111111'
PLAIN_ID = '22222222-2222-4222-8222-222222222221'
CONTESTING_ID = '22222222-2222-4222-8222-222222222222'

PARENT_BODY = 'The merge-lane verify timeout is 600 seconds.'
PLAIN_BODY = 'Applies to coalesce-train verifies as well.'
CONTESTING_BODY = (
    'Correction: the 600-second merge-lane verify timeout no longer holds. After '
    'the coalesce-train rollout every verify the lane runs is bounded by a '
    'ceiling read from git.merge_verify_timeout_seconds, and a verify that '
    'overruns it is parked as verify_timeout instead of being retried by the '
    'lane. Plan around 900 seconds, not 600.'
)


def _row(memory_id: str, text: str, created_at: str, **extra: Any) -> dict[str, Any]:
    """A scrolled child row of P: ``metadata`` is the full raw Qdrant payload."""
    return {
        'id': memory_id,
        'created_at': created_at,
        'metadata': {'data': text, 'parent_id': P, 'kind': AMENDMENT_KIND, **extra},
    }


def _contesting_row(text: str = CONTESTING_BODY) -> dict[str, Any]:
    return _row(CONTESTING_ID, text, '2026-09-20T00:00:00+00:00', **{CONTESTED_METADATA_KEY: True})


_PLAIN_ROW = _row(PLAIN_ID, PLAIN_BODY, '2026-08-10T00:00:00+00:00')
_PARENT_RECORD = {'id': P, 'content': PARENT_BODY, 'metadata': {'data': PARENT_BODY}}
_PLAIN_RECORD = {'id': PLAIN_ID, 'content': PLAIN_BODY, 'metadata': _PLAIN_ROW['metadata']}


def _service(
    rows: list[dict[str, Any]], records: dict[str, dict[str, Any]] | None = None,
) -> AsyncMock:
    """Answers the three Mem0 read primitives: rows by filter, records by id."""
    known = records or {}

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
        return known.get(memory_id)

    service = AsyncMock()
    service.count_memories_by_metadata = AsyncMock(side_effect=count)
    service.get_memories_by_metadata = AsyncMock(side_effect=scroll)
    service.get_memory_by_id = AsyncMock(side_effect=get_by_id)
    return service


def _parent_hit() -> MemoryResult:
    return MemoryResult(
        id=P, content=PARENT_BODY, source_store=SourceStore.mem0, relevance_score=0.9,
    )


def _entry_for(entries: list[dict[str, Any]], memory_id: str) -> dict[str, Any]:
    [entry] = [entry for entry in entries if entry.get('id') == memory_id]
    return entry


def _cut_at(cap: int) -> str:
    return CONTESTING_BODY[:cap] + elision_marker(len(CONTESTING_BODY) - cap)


async def _contested_entry(contested_text_chars: int) -> dict[str, Any]:
    service = _service([_PLAIN_ROW, _contesting_row()])

    block = await build_grouped_document(
        service, PROJECT_ID, P, contested_text_chars=contested_text_chars,
    )

    assert block is not None
    return _entry_for(block['amendments'], CONTESTING_ID)


class TestTheGroupedDocumentCarriesTheContestedText:
    @pytest.mark.asyncio
    async def test_a_contested_amendment_carries_its_whole_text_beside_its_digest(self) -> None:
        entry = await _contested_entry(len(CONTESTING_BODY) + 100)

        assert entry['content'] == CONTESTING_BODY
        assert entry['contested'] is True
        assert entry['digest'] != CONTESTING_BODY
        assert entry['digest'].endswith('…')

    @pytest.mark.asyncio
    async def test_text_exactly_at_the_cap_is_carried_unmarked(self) -> None:
        entry = await _contested_entry(len(CONTESTING_BODY))

        assert entry['content'] == CONTESTING_BODY

    @pytest.mark.asyncio
    async def test_text_over_the_cap_is_cut_at_it_and_counted(self) -> None:
        entry = await _contested_entry(50)

        assert entry['content'] == _cut_at(50)

    @pytest.mark.asyncio
    async def test_an_uncontested_amendment_carries_no_content(self) -> None:
        service = _service([_PLAIN_ROW, _contesting_row()])

        block = await build_grouped_document(service, PROJECT_ID, P, contested_text_chars=50)

        assert block is not None
        plain = _entry_for(block['amendments'], PLAIN_ID)
        assert 'content' not in plain
        assert plain['digest'] == PLAIN_BODY

    @pytest.mark.asyncio
    async def test_with_no_cap_named_the_judges_shipped_default_applies(self) -> None:
        default = WriteTriageConfig().judge_field_chars
        service = _service([_contesting_row('x' * (default + 10))])

        block = await build_grouped_document(service, PROJECT_ID, P)

        assert block is not None
        entry = _entry_for(block['amendments'], CONTESTING_ID)
        assert entry['content'] == 'x' * default + elision_marker(10)


class TestTheGroupedEntryPointsForwardTheCap:
    @pytest.mark.asyncio
    async def test_a_parent_only_search_carries_the_contested_text_at_the_named_cap(self) -> None:
        service = _service([_PLAIN_ROW, _contesting_row()])

        results = await group_search_results(
            service, PROJECT_ID, [_parent_hit()], contested_text_chars=50,
        )

        entry = _entry_for(results[0]['grouped']['amendments'], CONTESTING_ID)
        assert entry['content'] == _cut_at(50)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('memory_id', 'record', 'block_of'),
        [
            pytest.param(P, _PARENT_RECORD, lambda grouped: grouped, id='canonical-read'),
            pytest.param(
                PLAIN_ID, _PLAIN_RECORD, lambda grouped: grouped['parent'], id='child-read',
            ),
        ],
    )
    async def test_a_point_read_carries_the_contested_text_at_the_named_cap(
        self, memory_id: str, record: dict[str, Any], block_of: Any,
    ) -> None:
        service = _service(
            [_PLAIN_ROW, _contesting_row()], {P: _PARENT_RECORD, PLAIN_ID: _PLAIN_RECORD},
        )

        grouped = await group_memory_document(
            service, PROJECT_ID, memory_id, record, contested_text_chars=50,
        )

        assert grouped is not None
        entry = _entry_for(block_of(grouped)['amendments'], CONTESTING_ID)
        assert entry['content'] == _cut_at(50)
