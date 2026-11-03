"""Pin the recorded contesting-child ``search`` replies to the real producer.

The orchestrator renders a contesting child from bytes recorded under
``orchestrator/tests/fixtures/grouped_search_contesting_child/``. Those bytes
are only worth testing against if this producer still emits them, so each
recording is re-produced here through the real ``search`` tool over one fixed
scenario and compared for equality. See that directory's ``PROVENANCE.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic_core import to_jsonable_python

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.grouped_read import CONTESTED_METADATA_KEY
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import SearchResults

_PROJECT_ID = 'dark_factory'

_FIXTURES = (
    Path(__file__).resolve().parents[3]
    / 'orchestrator'
    / 'tests'
    / 'fixtures'
    / 'grouped_search_contesting_child'
)

_PARENT_ID = '11111111-1111-4111-8111-111111111111'
_PARENT_BODY = 'The merge-lane verify timeout is 600 seconds.'
_PARENT_CREATED_AT = '2026-08-01T00:00:00+00:00'

_AMENDMENT_ID = '22222222-2222-4222-8222-222222222221'
_AMENDMENT_BODY = 'Applies to coalesce-train verifies as well.'
_AMENDMENT_CREATED_AT = '2026-08-10T00:00:00+00:00'

_CONTESTING_ID = '22222222-2222-4222-8222-222222222222'
_CONTESTING_BODY = (
    'Correction: the 600-second merge-lane verify timeout is outdated. Since '
    'the coalesce-train rollout the lane runs every verify under a 900-second '
    'ceiling, set as git.merge_verify_timeout_seconds in '
    'dark-factory-orchestrator.yaml, and a verify that overruns it is parked '
    'as verify_timeout rather than retried. Plan around 900 seconds, not 600.'
)
_CONTESTING_CREATED_AT = '2026-09-20T00:00:00+00:00'

_CHILD_COUNTS = {'': 2, 'amendment': 2, 'sighting': 0}


def _hit(memory_id: str, content: str, score: float, metadata: dict, created_at: str) -> MemoryResult:
    return MemoryResult(
        id=memory_id,
        content=content,
        category=MemoryCategory.procedural_knowledge,
        source_store=SourceStore.mem0,
        relevance_score=score,
        metadata=metadata,
        created_at=created_at,
    )


def _child_metadata(*, contested: bool) -> dict:
    metadata: dict = {'kind': 'amendment', 'parent_id': _PARENT_ID}
    if contested:
        metadata[CONTESTED_METADATA_KEY] = True
    return metadata


def _scrolled_row(memory_id: str, body: str, created_at: str, *, contested: bool) -> dict:
    return {
        'id': memory_id,
        'created_at': created_at,
        'metadata': {'data': body, **_child_metadata(contested=contested)},
    }


_PARENT_HIT = _hit(_PARENT_ID, _PARENT_BODY, 0.9, {}, _PARENT_CREATED_AT)
_CONTESTING_HIT = _hit(
    _CONTESTING_ID,
    _CONTESTING_BODY,
    0.8,
    _child_metadata(contested=True),
    _CONTESTING_CREATED_AT,
)

_SCENARIO_HITS = {
    'child-also-matched': [_PARENT_HIT, _CONTESTING_HIT],
    'only-parent-matched': [_PARENT_HIT],
}


def _service(hits: list[MemoryResult]) -> AsyncMock:
    """A service whose parent P carries one plain and one contesting amendment."""
    rows = [
        _scrolled_row(_AMENDMENT_ID, _AMENDMENT_BODY, _AMENDMENT_CREATED_AT, contested=False),
        _scrolled_row(_CONTESTING_ID, _CONTESTING_BODY, _CONTESTING_CREATED_AT, contested=True),
    ]

    def _count(*, project_id: str, filters: dict) -> int:
        if filters.get('parent_id') != _PARENT_ID:
            return 0
        return _CHILD_COUNTS[filters.get('kind', '')]

    def _scroll(*, project_id: str, filters: dict, limit: int = 1000) -> list[dict]:
        return rows[:limit] if filters.get('parent_id') == _PARENT_ID else []

    service = AsyncMock()
    service.search = AsyncMock(return_value=SearchResults(hits))
    service.count_memories_by_metadata = AsyncMock(side_effect=_count)
    service.get_memories_by_metadata = AsyncMock(side_effect=_scroll)
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize('name', sorted(_SCENARIO_HITS))
async def test_the_recorded_reply_is_what_search_produces(name: str):
    fixture = _FIXTURES / f'{name}.json'
    assert fixture.is_file(), f'recorded reply missing at {fixture}'
    server = create_mcp_server(_service(_SCENARIO_HITS[name]))

    result = await server._tool_manager.call_tool(
        'search',
        {'query': 'merge-lane verify timeout', 'project_id': _PROJECT_ID},
    )

    produced = to_jsonable_python(result)
    assert produced == json.loads(fixture.read_text(encoding='utf-8')), (
        f'{fixture} is a recording of this producer and no longer matches it. '
        'Re-record it from the output below and fix the CONSUMER '
        '(orchestrator/src/orchestrator/agents/memory_recall.py); never '
        'hand-edit the fixture.\n'
        + json.dumps(produced, indent=2, ensure_ascii=False)
    )
