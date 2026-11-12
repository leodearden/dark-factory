"""Pin the recorded worst-case amendment-bodies ``search`` reply to the real producer.

The orchestrator pins the briefing's size against bytes recorded under
``orchestrator/tests/fixtures/grouped_search_amendment_bodies/``: one parent
carrying as many amendments as a grouped document lists, each longer than the
amendment text cap. Those bytes are only worth testing against if this
producer still emits them, so the recording is re-produced here through the
real ``search`` tool and compared for equality. See that directory's
``PROVENANCE.md``.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _pre_7518_digest import PRE_7518_DIGEST_CHARS
from pydantic_core import to_jsonable_python
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.grouped_read import (
    AMENDMENT_KIND,
    AMENDMENT_LIST_CAP,
    DEFAULT_AMENDMENT_TEXT_CHARS,
)
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import SearchResults

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

PROJECT_ID = 'dark_factory'

_RECORDING = (
    Path(__file__).resolve().parents[3]
    / 'orchestrator'
    / 'tests'
    / 'fixtures'
    / 'grouped_search_amendment_bodies'
    / 'ten-amendments-over-the-cap.json'
)

_PARENT_ID = '11111111-1111-4111-8111-111111111111'
_PARENT_BODY = 'The merge-lane verify timeout is 600 seconds.'
_PARENT_CREATED_AT = '2026-08-01T00:00:00+00:00'

_OPENING = (
    'The 600-second merge-lane verify timeout recorded on the parent no longer '
    'covers every verify the lane runs, because the coalesce-train rollout changed '
    'how a train re-verifies its rebased tip before it lands, so this amendment '
    'records the one case a reader acting on the parent would otherwise miss, '
)
_FILLER = 'A train that overruns the ceiling is parked rather than retried by the lane. '
_BODY_CHARS = DEFAULT_AMENDMENT_TEXT_CHARS + 500


def _marker(index: int) -> str:
    return f'DETAIL-{index:02d}'


def _amendment_id(index: int) -> str:
    return f'22222222-2222-4222-8222-{index:012d}'


def _amendment_body(index: int) -> str:
    """An opening sentence whose marker lies past char 240, padded to past the cap."""
    opening = f'{_OPENING}named {_marker(index)}. '
    filler = ''.join(itertools.islice(itertools.cycle([_FILLER]), _BODY_CHARS // len(_FILLER) + 1))
    return (opening + filler)[:_BODY_CHARS]


_AMENDMENT_ROWS = [
    {
        'id': _amendment_id(index),
        'created_at': f'2026-08-{index + 10:02d}T00:00:00+00:00',
        'metadata': {
            'data': _amendment_body(index),
            'kind': AMENDMENT_KIND,
            'parent_id': _PARENT_ID,
        },
    }
    for index in range(AMENDMENT_LIST_CAP)
]

_CHILD_COUNTS = {'': len(_AMENDMENT_ROWS), AMENDMENT_KIND: len(_AMENDMENT_ROWS), 'sighting': 0}


def _recorded_scenario_service() -> AsyncMock:
    """Parent P, the only hit, carries the most amendments a grouped document lists."""

    def _count(*, project_id: str, filters: dict) -> int:
        if filters.get('parent_id') != _PARENT_ID:
            return 0
        return _CHILD_COUNTS[filters.get('kind', '')]

    def _scroll(*, project_id: str, filters: dict, limit: int = 1000) -> list[dict]:
        return _AMENDMENT_ROWS[:limit] if filters.get('parent_id') == _PARENT_ID else []

    service = AsyncMock()
    service.search = AsyncMock(return_value=SearchResults([
        MemoryResult(
            id=_PARENT_ID,
            content=_PARENT_BODY,
            category=MemoryCategory.procedural_knowledge,
            source_store=SourceStore.mem0,
            relevance_score=0.9,
            metadata={},
            created_at=_PARENT_CREATED_AT,
        ),
    ]))
    service.count_memories_by_metadata = AsyncMock(side_effect=_count)
    service.get_memories_by_metadata = AsyncMock(side_effect=_scroll)
    return service


@pytest.mark.parametrize('index', range(AMENDMENT_LIST_CAP))
def test_each_amendment_hides_its_marker_past_the_old_digest_and_runs_past_the_cap(index: int):
    body = _amendment_body(index)

    assert body.index(_marker(index)) > PRE_7518_DIGEST_CHARS
    assert len(body) == DEFAULT_AMENDMENT_TEXT_CHARS + 500


@pytest.mark.asyncio
async def test_the_recorded_reply_is_what_search_produces():
    assert _RECORDING.is_file(), f'recorded reply missing at {_RECORDING}'
    server = create_mcp_server(_recorded_scenario_service())

    result = await server._tool_manager.call_tool(
        'search', {'query': 'merge-lane verify timeout', 'project_id': PROJECT_ID},
    )

    produced = to_jsonable_python(result)
    assert produced == json.loads(_RECORDING.read_text(encoding='utf-8')), (
        f'{_RECORDING} is a recording of this producer and no longer matches it. '
        'Re-record it from the output below and fix the CONSUMER '
        '(orchestrator/src/orchestrator/agents/memory_recall.py); never '
        'hand-edit the fixture.\n'
        + json.dumps(produced, indent=2, ensure_ascii=False)
    )
