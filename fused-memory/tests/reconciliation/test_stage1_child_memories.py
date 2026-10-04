"""Stage 1 sees child memories (amendments/sightings) as children of their parent (task 6193)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.models.reconciliation import Watermark
from fused_memory.server.grouped_read import CHILD_KINDS, PARENT_ID_KEY
from reconciliation.consolidator_fixtures import make_consolidator

_PARENT_ID = '11111111-2222-4333-8444-555555555555'
_PARENT_BODY = 'Run uv sync before pytest in a fresh worktree.'
_CHILD_IDS = {
    kind: f'aaaaaaaa-bbbb-4ccc-8ddd-{index:012d}'
    for index, kind in enumerate(sorted(CHILD_KINDS))
}


def _fixture_memories() -> list[dict]:
    parent = {
        'id': _PARENT_ID,
        'memory': _PARENT_BODY,
        'metadata': {'category': 'procedural_knowledge'},
    }
    children = [
        {
            'id': child_id,
            'memory': f'{_PARENT_BODY} (seen again)',
            'metadata': {
                'category': 'procedural_knowledge',
                PARENT_ID_KEY: _PARENT_ID,
                'kind': kind,
            },
        }
        for kind, child_id in _CHILD_IDS.items()
    ]
    return [parent, *children]


async def _render_payload() -> str:
    stage = make_consolidator()
    stage.memory.mem0.get_all = AsyncMock(return_value={'results': _fixture_memories()})
    return await stage.assemble_payload(
        events=[], watermark=Watermark(project_id='test_project'), prior_reports=[]
    )


def _line_for(payload: str, memory_id: str) -> str:
    matches = [line for line in payload.splitlines() if f'[{memory_id}]' in line]
    assert len(matches) == 1, (
        f'Expected exactly one payload line for memory {memory_id}, got {matches!r}; '
        f'payload:\n{payload}'
    )
    return matches[0]


class TestStage1PayloadRendersChildLinks:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('kind', sorted(CHILD_KINDS))
    async def test_child_line_carries_parent_id_and_kind(self, kind: str):
        payload = await _render_payload()
        line = _line_for(payload, _CHILD_IDS[kind])
        assert _PARENT_ID in line, (
            f'Child ({kind}) line must name its parent id; line={line!r}; payload:\n{payload}'
        )
        assert kind in line, (
            f'Child line must name its kind {kind!r}; line={line!r}; payload:\n{payload}'
        )

    @pytest.mark.asyncio
    async def test_plain_memory_line_unchanged(self):
        payload = await _render_payload()
        line = _line_for(payload, _PARENT_ID)
        assert line == f'- [{_PARENT_ID}] (procedural_knowledge): {_PARENT_BODY}', (
            f'A memory without {PARENT_ID_KEY} must keep the pre-existing line format; '
            f'payload:\n{payload}'
        )
