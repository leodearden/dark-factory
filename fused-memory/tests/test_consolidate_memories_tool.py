"""The `consolidate_memories` MCP tool (task 3133).

Stage-1's write-then-delete choreography is a RATCHET: a guard-exempt
canonical write plus unordered deletes with no verification nets +1 entry
per failed pass (17 of the 89 entries the reify curator deleted were the
consolidator's own prior canonicals and scaffolding). The cure is not a
better prompt — it is one op whose closure is CORROBORATED by a live
re-read, never inferred from "the delete call returned ok".

Harness per `tests/test_delete_memory_citation_guard.py`:
`create_mcp_server(mock_service)` + `_tool_manager.call_tool` +
`_parse_result` unwrapping FastMCP `TextContent`. The mem0_update leaves
are REAL `Mem0UpdateConfig` values rather than bare Mocks — a bare
`AsyncMock` makes every leaf a Mock, which the fail-closed authz resolver
rejects, so every case here would otherwise pass for the wrong reason
(`Mem0UpdateNotAuthorized` instead of the property under test). That
failure mode is recorded in `tests/test_update_memory_tool.py`.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from fused_memory.config.schema import Mem0UpdateConfig
from fused_memory.models.memory import AddMemoryResponse
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import DescendantScan

PROJECT_ID = 'dark_factory'
# On the default allowlist for both mem0_update arms, so no case here can
# fail for an authorization reason and be misread as a result.
AGENT = 'recon-stage-memory_consolidator'
RUN_ID = 'run-abc'
TOPIC = 'memory-consolidation'
CONTENT = 'Consolidation folds a duplicate cluster into one canonical claim.'

# The cluster: three superseded duplicates, folded into one new canonical.
S1 = '11111111-1111-4111-8111-111111111111'
S2 = '22222222-2222-4222-8222-222222222222'
S3 = '33333333-3333-4333-8333-333333333333'
SUPERSEDES = [S1, S2, S3]
CANONICAL = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'

RETAIN_1 = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb'
RETAIN_2 = 'cccccccc-cccc-4ccc-8ccc-cccccccccccc'


def _parse_result(result):
    """Parse a FastMCP call_tool result (list of TextContent) into a dict."""
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


def _row(memory_id, **metadata):
    return {
        'id': memory_id,
        'content': f'record {memory_id}',
        'created_at': '2026-01-01T00:00:00+00:00',
        'metadata': {'topic': TOPIC, **metadata},
    }


def make_service(
    *,
    gone=SUPERSEDES,
    topic_members=None,
    children=None,
    delete_errors=None,
):
    """A MemoryService mock modelling one consolidation cluster.

    *gone* is the set the post-delete re-read finds ABSENT — the ids that
    are genuinely closed. Anything in `SUPERSEDES` but not in *gone* still
    resolves, i.e. it is a survivor of a delete that claimed success.

    *children* maps a supersede id to its direct children; *delete_errors*
    maps a supersede id to an exception `delete_memory` raises for it.
    """
    gone = set(gone)
    children = children or {}
    delete_errors = delete_errors or {}
    members = SUPERSEDES if topic_members is None else topic_members

    svc = AsyncMock()
    svc.config.mem0_update = Mem0UpdateConfig()

    svc.add_memory = AsyncMock(
        return_value=AddMemoryResponse(memory_ids=[CANONICAL], message='ok')
    )

    async def _delete(**kwargs):
        mid = kwargs.get('memory_id')
        if mid in delete_errors:
            raise delete_errors[mid]
        return {'status': 'deleted', 'store': 'mem0', 'id': mid}

    svc.delete_memory = AsyncMock(side_effect=_delete)

    async def _get(project_id=None, memory_id=None, **_):
        if memory_id in gone:
            return None
        return _row(memory_id)

    svc.get_memory_by_id = AsyncMock(side_effect=_get)

    async def _list_child_ids(memory_id, *, project_id):
        return DescendantScan(ids=list(children.get(memory_id, ())), truncated=False)

    svc.list_child_ids = AsyncMock(side_effect=_list_child_ids)
    svc.update_memory = AsyncMock(
        return_value={'status': 'updated', 'store': 'mem0', 'metadata_patched': True}
    )
    svc.get_memories_by_metadata = AsyncMock(
        return_value=[_row(m) for m in members]
    )
    svc.count_memories_by_metadata = AsyncMock(return_value=len(members))
    return svc


async def call_consolidate(svc, **overrides):
    server = create_mcp_server(svc)
    args = {
        'canonical_content': CONTENT,
        'topic': TOPIC,
        'project_id': PROJECT_ID,
        'supersedes': list(SUPERSEDES),
        'run_id': RUN_ID,
        'agent_id': AGENT,
    }
    args.update(overrides)
    return _parse_result(
        await server._tool_manager.call_tool('consolidate_memories', args)
    )


class TestToolRegistration:
    @pytest.mark.asyncio
    async def test_tool_is_registered(self):
        server = create_mcp_server(make_service())

        tools = await server._tool_manager.list_tools()

        assert 'consolidate_memories' in {t.name for t in tools}


class TestHappyPath:
    """One canonical written, three supersedes folded, closure corroborated."""

    @pytest.mark.asyncio
    async def test_canonical_is_written_once_with_the_consolidation_metadata(self):
        svc = make_service()

        await call_consolidate(svc)

        svc.add_memory.assert_awaited_once()
        meta = svc.add_memory.await_args.kwargs['metadata']
        assert meta['topic'] == TOPIC
        assert meta['canonical'] is True
        assert list(meta['supersedes']) == SUPERSEDES
        assert svc.add_memory.await_args.kwargs['content'] == CONTENT

    @pytest.mark.asyncio
    async def test_every_supersede_is_deleted_from_mem0(self):
        svc = make_service()

        await call_consolidate(svc)

        deleted = [c.kwargs['memory_id'] for c in svc.delete_memory.await_args_list]
        assert deleted == SUPERSEDES
        assert {c.kwargs['store'] for c in svc.delete_memory.await_args_list} == {'mem0'}

    @pytest.mark.asyncio
    async def test_result_envelope(self):
        svc = make_service()

        result = await call_consolidate(svc)

        assert result['status'] == 'consolidated'
        assert result['canonical_id'] == CANONICAL
        assert result['topic'] == TOPIC
        assert result['deleted'] == SUPERSEDES
        assert result['survivors'] == []
        assert result['failed_deletes'] == []

    @pytest.mark.asyncio
    async def test_topic_closure_is_listed_from_the_deterministic_scroll(self):
        svc = make_service(topic_members=[CANONICAL, RETAIN_1])

        result = await call_consolidate(svc)

        assert [m['id'] for m in result['topic_members']] == [CANONICAL, RETAIN_1]
        assert result['topic_members_truncated'] is False
        call = svc.get_memories_by_metadata.await_args
        assert call.kwargs['filters'] == {'topic': TOPIC}
        assert call.kwargs['project_id'] == PROJECT_ID

    @pytest.mark.asyncio
    async def test_the_closure_listing_never_goes_through_search(self):
        """THE pinned negative.

        Semantic search is what made the original incident unfixable: run
        live, the re-derive query returned only superseded cluster members,
        routing dispatch back into the contradictory advice consolidation
        existed to collapse. A consolidation that reports its own result via
        a top-N ranked read can silently omit the record it just wrote.
        """
        svc = make_service()

        await call_consolidate(svc)

        svc.search.assert_not_called()


class TestValidationIsRefusedBeforeAnyWrite:
    """The validator's refusals reach the wire, and cost zero writes."""

    @pytest.mark.asyncio
    async def test_malformed_supersede_is_refused_and_nothing_is_written(self):
        svc = make_service()

        result = await call_consolidate(svc, supersedes=[S1, '873889a1'])

        assert result['error_type'] == 'ValidationError'
        assert '873889a1' in result['error']
        svc.add_memory.assert_not_awaited()
        svc.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_hint_survives_the_tool_boundary(self):
        """Returned, never raised.

        `@mcp_tool_errors` flattens an exception to {'error', 'error_type'},
        which would drop the hint — the part that tells the caller what to
        do about it.
        """
        svc = make_service()

        result = await call_consolidate(svc, topic='memory_consolidation')

        assert 'fused_memory.topic_slug' in result.get('hint', '')

    @pytest.mark.asyncio
    async def test_delete_arm_without_run_id_is_refused_before_the_canonical(self):
        svc = make_service()

        result = await call_consolidate(svc, run_id=None)

        assert result['error_type'] == 'ValidationError'
        assert 'run_id' in result['error']
        svc.add_memory.assert_not_awaited()
