"""Stage 1 sees child memories (amendments/sightings) as children of their parent.

Covers both Stage 1 payload paths: the time-windowed fetch (task 6193) and the
ContextAssembler backlog chunk (task 6294).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.memory import MemoryCategory, MemoryResult, SourceStore
from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
    Watermark,
)
from fused_memory.reconciliation.context_assembler import ContextAssembler
from fused_memory.reconciliation.prompts.stage1 import CHILD_MEMORY_AUTHORITY_RULE
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


async def _render_payload(memories: list[dict] | None = None) -> str:
    stage = make_consolidator()
    results = _fixture_memories() if memories is None else memories
    stage.memory.mem0.get_all = AsyncMock(return_value={'results': results})
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


_NOT_A_CHILD_ID = 'cccccccc-dddd-4eee-8fff-000000000000'
_NON_CHILD_KIND = 'cycle_summary'


class TestStage1PayloadRendersNoLinkForNonChildren:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('metadata', 'rendered_category'),
        [
            pytest.param(
                {'category': 'procedural_knowledge', PARENT_ID_KEY: _PARENT_ID,
                 'kind': _NON_CHILD_KIND},
                'procedural_knowledge',
                id='parent-id-with-non-child-kind',
            ),
            pytest.param(
                {'category': 'procedural_knowledge', PARENT_ID_KEY: _PARENT_ID},
                'procedural_knowledge',
                id='parent-id-without-kind',
            ),
            pytest.param(
                {'category': 'procedural_knowledge', PARENT_ID_KEY: 42,
                 'kind': min(CHILD_KINDS)},
                'procedural_knowledge',
                id='child-kind-with-non-string-parent-id',
            ),
            pytest.param(None, '?', id='no-metadata'),
        ],
    )
    async def test_line_keeps_plain_format(self, metadata, rendered_category: str):
        assert _NON_CHILD_KIND not in CHILD_KINDS
        memory = {'id': _NOT_A_CHILD_ID, 'memory': _PARENT_BODY, 'metadata': metadata}
        payload = await _render_payload([memory])
        line = _line_for(payload, _NOT_A_CHILD_ID)
        assert line == f'- [{_NOT_A_CHILD_ID}] ({rendered_category}): {_PARENT_BODY}', (
            'Only a grouped_read child (child kind + string parent id) gets a parent '
            f'link; line={line!r}; payload:\n{payload}'
        )


def _as_search_hit(memory: dict) -> MemoryResult:
    return MemoryResult(
        id=memory['id'],
        content=memory['memory'],
        category=MemoryCategory.procedural_knowledge,
        source_store=SourceStore.mem0,
        metadata=dict(memory['metadata'] or {}),
    )


async def _render_backlog_payload(memories: list[dict] | None = None) -> str:
    records = _fixture_memories() if memories is None else memories
    assembler = ContextAssembler(
        memory_service=AsyncMock(
            search=AsyncMock(return_value=[_as_search_hit(m) for m in records])
        ),
        taskmaster=None,
        config=ReconciliationConfig(),
        project_root='/tmp/test',
    )
    event = ReconciliationEvent(
        id='backlog-event',
        type=EventType.memory_added,
        source=EventSource.agent,
        project_id='test_project',
        timestamp=datetime.now(UTC),
        payload={'content_preview': 'uv sync before pytest', 'category': 'procedural_knowledge'},
    )
    ap = await assembler.assemble([event], Watermark(project_id='test_project'), 'test_project')
    stage = make_consolidator()
    stage.assembled_payload = ap
    return await stage.assemble_payload(
        events=ap.events, watermark=Watermark(project_id='test_project'), prior_reports=[]
    )


def _link_of(line: str, memory_id: str) -> str:
    after_id = line.split(f'[{memory_id}]', 1)[1]
    return after_id.split(' (', 1)[0]


class TestBacklogPayloadRendersChildLinks:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('kind', sorted(CHILD_KINDS))
    async def test_child_line_carries_parent_id_and_kind(self, kind: str):
        payload = await _render_backlog_payload()
        line = _line_for(payload, _CHILD_IDS[kind])
        assert _PARENT_ID in line, (
            f'Backlog child ({kind}) line must name its parent id; line={line!r}; '
            f'payload:\n{payload}'
        )
        assert kind in line, (
            f'Backlog child line must name its kind {kind!r}; line={line!r}; payload:\n{payload}'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('kind', sorted(CHILD_KINDS))
    async def test_child_link_matches_time_windowed_path(self, kind: str):
        child_id = _CHILD_IDS[kind]
        backlog_link = _link_of(_line_for(await _render_backlog_payload(), child_id), child_id)
        windowed_link = _link_of(_line_for(await _render_payload(), child_id), child_id)
        assert windowed_link, f'Time-windowed path must render a link for child {kind!r}'
        assert backlog_link == windowed_link, (
            'Both Stage 1 payload paths must render the same parent link for the same '
            f'record; backlog={backlog_link!r}, time-windowed={windowed_link!r}'
        )

    @pytest.mark.asyncio
    async def test_plain_hit_line_unchanged(self):
        payload = await _render_backlog_payload()
        line = _line_for(payload, _PARENT_ID)
        assert line == f'- [{_PARENT_ID}] (mem0/procedural_knowledge): {_PARENT_BODY}', (
            f'A hit without {PARENT_ID_KEY} must keep the pre-existing line format; '
            f'payload:\n{payload}'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'metadata',
        [
            pytest.param(
                {'category': 'procedural_knowledge', PARENT_ID_KEY: _PARENT_ID,
                 'kind': _NON_CHILD_KIND},
                id='parent-id-with-non-child-kind',
            ),
            pytest.param(
                {'category': 'procedural_knowledge', PARENT_ID_KEY: _PARENT_ID},
                id='parent-id-without-kind',
            ),
            pytest.param(
                {'category': 'procedural_knowledge', PARENT_ID_KEY: 42,
                 'kind': min(CHILD_KINDS)},
                id='child-kind-with-non-string-parent-id',
            ),
            pytest.param({}, id='empty-metadata'),
        ],
    )
    async def test_non_child_hit_keeps_plain_format(self, metadata: dict):
        memory = {'id': _NOT_A_CHILD_ID, 'memory': _PARENT_BODY, 'metadata': metadata}
        payload = await _render_backlog_payload([memory])
        line = _line_for(payload, _NOT_A_CHILD_ID)
        assert line == f'- [{_NOT_A_CHILD_ID}] (mem0/procedural_knowledge): {_PARENT_BODY}', (
            'Only a grouped_read child (child kind + string parent id) gets a parent '
            f'link; line={line!r}; payload:\n{payload}'
        )


class TestStage1PromptChildRule:
    def test_rendered_prompt_carries_child_rule(self):
        assert CHILD_MEMORY_AUTHORITY_RULE in make_consolidator().get_system_prompt()

    @pytest.mark.parametrize('kind', sorted(CHILD_KINDS))
    def test_child_rule_names_every_child_kind(self, kind: str):
        assert f'`{kind}`' in CHILD_MEMORY_AUTHORITY_RULE, (
            f'Child rule must name CHILD_KINDS member {kind!r}; rule={CHILD_MEMORY_AUTHORITY_RULE!r}'
        )

    def test_child_rule_names_the_parent_link_key(self):
        assert f'`{PARENT_ID_KEY}`' in CHILD_MEMORY_AUTHORITY_RULE, (
            f'Child rule must name {PARENT_ID_KEY!r}; rule={CHILD_MEMORY_AUTHORITY_RULE!r}'
        )
