"""Every amendment rides in its parent's grouped document as its text up to the judge's field cap, so a reader of the parent sees the detail the amendment adds."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from _pre_7518_digest import PRE_7518_DIGEST_CHARS

from fused_memory.config.schema import WriteTriageConfig
from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.grouped_read import (
    AMENDMENT_KIND,
    CONTESTED_METADATA_KEY,
    DEFAULT_AMENDMENT_TEXT_CHARS,
    amendment_text,
    build_grouped_document,
    group_memory_document,
    group_search_results,
)
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import SearchResults
from fused_memory.utils.clip_disclosure import elision_marker

PROJECT_ID = 'dark_factory'
P = '11111111-1111-4111-8111-111111111111'
AMENDING_ID = '22222222-2222-4222-8222-222222222221'
SHORT_ID = '22222222-2222-4222-8222-222222222222'
CONTESTING_ID = '22222222-2222-4222-8222-222222222223'

PARENT_BODY = 'The merge-lane verify timeout is 600 seconds.'
SHORT_BODY = 'Applies to coalesce-train verifies as well.'
LATE_DETAIL = 'A verify that overruns git.merge_verify_timeout_seconds is parked as verify_timeout.'
AMENDING_BODY = (
    'The 600-second timeout applies to coalesce-train verifies as well. '
    + 'A train re-verifies its rebased tip before it lands. ' * 4
    + LATE_DETAIL
)

CAP = 50


def _row(memory_id: str, text: str, created_at: str, **extra: Any) -> dict[str, Any]:
    """A scrolled child row of P: ``metadata`` is the full raw Qdrant payload."""
    return {
        'id': memory_id,
        'created_at': created_at,
        'metadata': {'data': text, 'parent_id': P, 'kind': AMENDMENT_KIND, **extra},
    }


_AMENDING_ROW = _row(AMENDING_ID, AMENDING_BODY, '2026-08-10T00:00:00+00:00')
_SHORT_ROW = _row(SHORT_ID, SHORT_BODY, '2026-08-11T00:00:00+00:00')
_PARENT_RECORD = {'id': P, 'content': PARENT_BODY, 'metadata': {'data': PARENT_BODY}}
_SHORT_RECORD = {'id': SHORT_ID, 'content': SHORT_BODY, 'metadata': _SHORT_ROW['metadata']}


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


def _cut_at(cap: int, text: str = AMENDING_BODY) -> str:
    return text[:cap] + elision_marker(len(text) - cap)


async def _amendment_entry(rows: list[dict[str, Any]], memory_id: str, **cap: int) -> dict[str, Any]:
    block = await build_grouped_document(_service(rows), PROJECT_ID, P, **cap)

    assert block is not None
    return _entry_for(block['amendments'], memory_id)


class TestAmendmentText:
    @pytest.mark.parametrize('length', [CAP - 1, CAP])
    def test_text_at_or_under_the_cap_is_returned_whole(self, length: int) -> None:
        assert amendment_text('x' * length, CAP) == 'x' * length

    def test_text_over_the_cap_is_cut_at_it_with_the_counted_marker(self) -> None:
        assert amendment_text(AMENDING_BODY, CAP) == _cut_at(CAP)

    def test_the_default_cap_is_the_judges_shipped_field_cap(self) -> None:
        assert WriteTriageConfig().judge_field_chars == DEFAULT_AMENDMENT_TEXT_CHARS

    def test_with_no_cap_named_the_default_applies(self) -> None:
        text = 'x' * (DEFAULT_AMENDMENT_TEXT_CHARS + 10)

        assert amendment_text(text) == 'x' * DEFAULT_AMENDMENT_TEXT_CHARS + elision_marker(10)


class TestEveryAmendmentCarriesItsText:
    @pytest.mark.asyncio
    async def test_an_uncontested_amendment_carries_its_text_and_no_digest(self) -> None:
        entry = await _amendment_entry([_AMENDING_ROW], AMENDING_ID, amendment_text_chars=CAP)

        assert entry['content'] == amendment_text(AMENDING_BODY, CAP)
        assert 'digest' not in entry

    @pytest.mark.asyncio
    async def test_a_detail_past_the_old_digest_cut_reaches_a_reader_of_the_parent(self) -> None:
        assert AMENDING_BODY.index(LATE_DETAIL) > PRE_7518_DIGEST_CHARS

        entry = await _amendment_entry([_AMENDING_ROW], AMENDING_ID)

        assert LATE_DETAIL in entry['content']
        assert entry['content'] == AMENDING_BODY

    @pytest.mark.asyncio
    async def test_a_body_over_the_cap_is_cut_at_it_with_the_counted_marker(self) -> None:
        entry = await _amendment_entry([_AMENDING_ROW], AMENDING_ID, amendment_text_chars=CAP)

        assert entry['content'] == _cut_at(CAP)

    @pytest.mark.asyncio
    async def test_with_no_cap_named_the_judges_shipped_default_applies(self) -> None:
        default = WriteTriageConfig().judge_field_chars
        long_row = _row(AMENDING_ID, 'x' * (default + 10), '2026-08-10T00:00:00+00:00')

        entry = await _amendment_entry([long_row], AMENDING_ID)

        assert entry['content'] == 'x' * default + elision_marker(10)

    @pytest.mark.asyncio
    async def test_a_contested_amendment_differs_from_an_uncontested_one_only_by_its_flag(
        self,
    ) -> None:
        contesting_row = _row(
            CONTESTING_ID, AMENDING_BODY, '2026-08-12T00:00:00+00:00',
            **{CONTESTED_METADATA_KEY: True},
        )
        block = await build_grouped_document(
            _service([_AMENDING_ROW, contesting_row]), PROJECT_ID, P, amendment_text_chars=CAP,
        )

        assert block is not None
        plain = _entry_for(block['amendments'], AMENDING_ID)
        contested = _entry_for(block['amendments'], CONTESTING_ID)
        assert plain['content'] == contested['content'] == _cut_at(CAP)
        assert contested['contested'] is True
        assert 'contested' not in plain


class TestTheGroupedEntryPointsForwardTheCap:
    @pytest.mark.asyncio
    async def test_a_parent_only_search_carries_each_amendment_at_the_named_cap(self) -> None:
        service = _service([_AMENDING_ROW, _SHORT_ROW])

        results = await group_search_results(
            service, PROJECT_ID, [_parent_hit()], amendment_text_chars=CAP,
        )

        entry = _entry_for(results[0]['grouped']['amendments'], AMENDING_ID)
        assert entry['content'] == _cut_at(CAP)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('memory_id', 'record', 'block_of'),
        [
            pytest.param(P, _PARENT_RECORD, lambda grouped: grouped, id='canonical-read'),
            pytest.param(
                SHORT_ID, _SHORT_RECORD, lambda grouped: grouped['parent'], id='child-read',
            ),
        ],
    )
    async def test_a_point_read_carries_each_amendment_at_the_named_cap(
        self, memory_id: str, record: dict[str, Any], block_of: Any,
    ) -> None:
        service = _service(
            [_AMENDING_ROW, _SHORT_ROW], {P: _PARENT_RECORD, SHORT_ID: _SHORT_RECORD},
        )

        grouped = await group_memory_document(
            service, PROJECT_ID, memory_id, record, amendment_text_chars=CAP,
        )

        assert grouped is not None
        entry = _entry_for(block_of(grouped)['amendments'], AMENDING_ID)
        assert entry['content'] == _cut_at(CAP)


_ENTRY_POINTS = [
    pytest.param(
        lambda service, cap: build_grouped_document(
            service, PROJECT_ID, P, amendment_text_chars=cap,
        ),
        id='build_grouped_document',
    ),
    pytest.param(
        lambda service, cap: group_search_results(
            service, PROJECT_ID, [_parent_hit()], amendment_text_chars=cap,
        ),
        id='group_search_results',
    ),
    pytest.param(
        lambda service, cap: group_memory_document(
            service, PROJECT_ID, SHORT_ID, _SHORT_RECORD, amendment_text_chars=cap,
        ),
        id='group_memory_document',
    ),
]


class TestABadCapIsTheCallersErrorNotABackendFault:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('read', _ENTRY_POINTS)
    @pytest.mark.parametrize('cap', [0, -1, True, 50.0])
    async def test_it_is_rejected_by_name_before_any_child_read(
        self, read: Any, cap: Any,
    ) -> None:
        service = _service([_AMENDING_ROW, _SHORT_ROW], {P: _PARENT_RECORD})

        with pytest.raises(ValueError, match=rf'amendment_text_chars.*{cap!r}'):
            await read(service, cap)

        service.count_memories_by_metadata.assert_not_awaited()
        service.get_memories_by_metadata.assert_not_awaited()
        service.get_memory_by_id.assert_not_awaited()


def _tool_service(judge_field_chars: int) -> AsyncMock:
    """A parent-only corpus of one plain amendment whose live config names *judge_field_chars*."""
    service = _service([_AMENDING_ROW], {P: _PARENT_RECORD})
    service.search = AsyncMock(return_value=SearchResults([_parent_hit()]))
    service.config.write_triage.judge_field_chars = judge_field_chars
    return service


async def _searched_amendments(server: Any) -> list[dict[str, Any]]:
    reply = await server._tool_manager.call_tool(
        'search', {'query': 'merge-lane verify timeout', 'project_id': PROJECT_ID},
    )
    return reply['results'][0]['grouped']['amendments']


async def _read_amendments(server: Any) -> list[dict[str, Any]]:
    reply = await server._tool_manager.call_tool(
        'get_memory_by_id', {'project_id': PROJECT_ID, 'memory_id': P},
    )
    return reply['grouped']['amendments']


class TestToolsCapEveryAmendmentAtTheLiveJudgeFieldCap:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'amendments_of',
        [
            pytest.param(_searched_amendments, id='search'),
            pytest.param(_read_amendments, id='get_memory_by_id'),
        ],
    )
    async def test_the_cap_is_the_live_judge_field_chars_read_per_call(
        self, amendments_of: Any,
    ) -> None:
        service = _tool_service(50)
        server = create_mcp_server(service)

        first = _entry_for(await amendments_of(server), AMENDING_ID)['content']
        service.config.write_triage.judge_field_chars = 80
        second = _entry_for(await amendments_of(server), AMENDING_ID)['content']

        assert (first, second) == (_cut_at(50), _cut_at(80))
