"""Tests for redact_episode_content across backend, service, and MCP tool.

Task 3083, the Graphiti half of the remediation. The Mem0/Qdrant sweep
(``fused-memory/scripts/sweep_toolcall_xml_leak.py``) structurally cannot
reach a leaked Graphiti episode: different store, different transport, no
Qdrant payload to scan. The known residual episode
``d12b0eb4-f027-4d0c-a26c-096ccd0e75c2`` therefore needs its own
content-preserving primitive.

The design commitment pinned here as behaviour, not prose: redact, NEVER
cascade-delete. ``delete_episode(cascade=True)`` on that residual would
destroy demonstrably-valid extracted collateral such as edge ``ea4072dc``
— deleting real knowledge to fix appearance. So this method sets ONE
property on ONE EpisodicNode and touches nothing else, and the tests below
assert the negative (no episode/edge removal) as hard as the positive.

Layered like test_reassign_edge.py:
- GraphitiBackend.redact_episode_content() — scoped SET, refusals, audit dict
- MemoryService.redact_episode_content()   — delegation + journaling
- MCP tool redact_episode_content in tools.py
- DISALLOW_MEMORY_WRITES list in cli_stage_runner.py

SENTINEL-LITERAL HAZARD (plan pre-1): every sentinel is spelled with the
``\\x3c`` escape for ``<``. Writing one verbatim would reproduce the bug
under repair inside this file's own authoring tool call.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import extract_cypher, extract_params

from fused_memory.backends.graphiti_client import GraphitiBackend, NodeNotFoundError

_CLOSE_CONTENT = '\x3c/content>'
_CLOSE_INVOKE = '\x3c/invoke>'
_CLOSE_DESCRIPTION = '\x3c/description>'
_OPEN_PRIORITY = '\x3cparameter name="priority">'

# The c759c53b Mem0/episode shape: body, stray closing tag, real newline,
# bare closing invoke tag.
_LEAK_TAIL = _CLOSE_CONTENT + '\n' + _CLOSE_INVOKE
_DIRTY_CONTENT = 'A recorded observation about the grace window.' + _LEAK_TAIL

# What an operator would write back: the body, with the fragment replaced by
# an auditable marker rather than silently dropped.
_CLEAN_CONTENT = (
    'A recorded observation about the grace window. '
    '[redacted: leaked tool-call XML fragment, task 3083]'
)

_EPISODE_UUID = 'd12b0eb4-f027-4d0c-a26c-096ccd0e75c2'
_GROUP_ID = 'dark_factory'


def _wire_graph(backend, *, rows):
    """Route the backend's direct-Cypher graph object.

    ``ro_query`` returns *rows* (the before-content read); ``query`` is a
    plain AsyncMock so the write can be asserted on — or asserted absent.
    """
    graph = MagicMock()
    ro_result = MagicMock()
    ro_result.result_set = rows
    graph.ro_query = AsyncMock(return_value=ro_result)
    graph.query = AsyncMock(return_value=MagicMock(result_set=[]))
    backend._driver._get_graph = MagicMock(return_value=graph)
    return graph


# ---------------------------------------------------------------------------
# GraphitiBackend.redact_episode_content — the scoped content SET
# ---------------------------------------------------------------------------


class TestRedactEpisodeContentBackendWrite:
    """The write targets ONE episode by uuid within group_id and sets ONLY
    the ``content`` property."""

    @pytest.mark.asyncio
    async def test_sets_only_content_scoped_by_uuid_and_group(
        self, mock_config, make_backend,
    ):
        backend = make_backend(mock_config)
        graph = _wire_graph(backend, rows=[[_DIRTY_CONTENT]])

        await backend.redact_episode_content(
            _EPISODE_UUID, group_id=_GROUP_ID, new_content=_CLEAN_CONTENT,
        )

        graph.query.assert_awaited_once()
        cypher = extract_cypher(graph.query.call_args)
        params = extract_params(graph.query.call_args)
        # Targets the episode by uuid, scoped to the project's group.
        assert 'Episodic' in cypher
        assert '$uuid' in cypher
        assert '$group_id' in cypher
        assert params['uuid'] == _EPISODE_UUID
        assert params['group_id'] == _GROUP_ID
        # Sets content and nothing else.
        assert 'SET' in cypher
        set_clause = cypher.split('SET', 1)[1]
        assert 'content' in set_clause
        for forbidden in ('name', 'valid_at', 'created_at', 'source', 'entity_edges'):
            assert forbidden not in set_clause, f'redaction must not touch {forbidden}'
        assert params['content'] == _CLEAN_CONTENT

    @pytest.mark.asyncio
    async def test_before_read_is_scoped_to_the_same_group(
        self, mock_config, make_backend,
    ):
        """The before-content read is uuid + group scoped too, so a uuid
        belonging to another project cannot be read or redacted from here."""
        backend = make_backend(mock_config)
        graph = _wire_graph(backend, rows=[[_DIRTY_CONTENT]])

        await backend.redact_episode_content(
            _EPISODE_UUID, group_id=_GROUP_ID, new_content=_CLEAN_CONTENT,
        )

        cypher = extract_cypher(graph.ro_query.call_args)
        params = extract_params(graph.ro_query.call_args)
        assert 'Episodic' in cypher
        assert '$uuid' in cypher
        assert '$group_id' in cypher
        assert params == {'uuid': _EPISODE_UUID, 'group_id': _GROUP_ID}

    @pytest.mark.asyncio
    async def test_returns_before_and_after_content(self, mock_config, make_backend):
        """The caller must be able to audit exactly what was neutralised."""
        backend = make_backend(mock_config)
        _wire_graph(backend, rows=[[_DIRTY_CONTENT]])

        result = await backend.redact_episode_content(
            _EPISODE_UUID, group_id=_GROUP_ID, new_content=_CLEAN_CONTENT,
        )

        assert result['uuid'] == _EPISODE_UUID
        assert result['group_id'] == _GROUP_ID
        assert result['old_content'] == _DIRTY_CONTENT
        assert result['new_content'] == _CLEAN_CONTENT

    @pytest.mark.asyncio
    async def test_canonicalizes_group_id(self, mock_config, make_backend):
        """Carries the @_canonicalize_group_args contract like its siblings."""
        backend = make_backend(mock_config)
        graph = _wire_graph(backend, rows=[[_DIRTY_CONTENT]])

        result = await backend.redact_episode_content(
            _EPISODE_UUID, group_id='dark-factory', new_content=_CLEAN_CONTENT,
        )

        assert result['group_id'] == _GROUP_ID
        assert extract_params(graph.query.call_args)['group_id'] == _GROUP_ID
        backend._driver._get_graph.assert_called_with(_GROUP_ID)

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)  # client/_driver are None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.redact_episode_content(
                _EPISODE_UUID, group_id=_GROUP_ID, new_content=_CLEAN_CONTENT,
            )


class TestRedactEpisodeContentPreservesCollateral:
    """The whole point: the extracted edges survive. A redaction must never
    reach for a delete path — ``delete_episode(cascade=True)`` on residual
    d12b0eb4 would destroy valid collateral such as edge ea4072dc."""

    @pytest.mark.asyncio
    async def test_never_calls_any_removal_path(self, mock_config, make_backend):
        backend = make_backend(mock_config)
        _wire_graph(backend, rows=[[_DIRTY_CONTENT]])
        backend.remove_episode = AsyncMock()
        backend.remove_edge = AsyncMock()
        backend.bulk_remove_edges = AsyncMock()

        await backend.redact_episode_content(
            _EPISODE_UUID, group_id=_GROUP_ID, new_content=_CLEAN_CONTENT,
        )

        backend.remove_episode.assert_not_awaited()
        backend.remove_edge.assert_not_awaited()
        backend.bulk_remove_edges.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_issues_no_delete_or_detach_cypher(self, mock_config, make_backend):
        """Belt and braces: no issued query may contain a destructive clause."""
        backend = make_backend(mock_config)
        graph = _wire_graph(backend, rows=[[_DIRTY_CONTENT]])

        await backend.redact_episode_content(
            _EPISODE_UUID, group_id=_GROUP_ID, new_content=_CLEAN_CONTENT,
        )

        issued = [
            extract_cypher(call) for call in graph.query.call_args_list
        ] + [
            extract_cypher(call) for call in graph.ro_query.call_args_list
        ]
        for cypher in issued:
            upper = cypher.upper()
            assert 'DELETE' not in upper
            assert 'DETACH' not in upper
            assert 'REMOVE' not in upper


class TestRedactEpisodeContentRefusals:
    """Every refusal is LOUD and issues no write — a typo or a bad redaction
    can never be mistaken for a successful one."""

    @pytest.mark.asyncio
    async def test_absent_episode_raises_and_issues_no_write(
        self, mock_config, make_backend,
    ):
        """An absent uuid raises rather than silently matching zero rows."""
        backend = make_backend(mock_config)
        graph = _wire_graph(backend, rows=[])

        with pytest.raises(NodeNotFoundError, match=_EPISODE_UUID):
            await backend.redact_episode_content(
                _EPISODE_UUID, group_id=_GROUP_ID, new_content=_CLEAN_CONTENT,
            )
        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_refuses_new_content_that_still_carries_a_leak(
        self, mock_config, make_backend,
    ):
        """A redaction that re-introduces the fragment is rejected — the
        shared detector is the sole oracle, here as everywhere."""
        backend = make_backend(mock_config)
        graph = _wire_graph(backend, rows=[[_DIRTY_CONTENT]])
        still_dirty = 'Partially cleaned.' + _CLOSE_DESCRIPTION + '\n' + _OPEN_PRIORITY + 'low'

        with pytest.raises(ValueError, match='leak'):
            await backend.redact_episode_content(
                _EPISODE_UUID, group_id=_GROUP_ID, new_content=still_dirty,
            )
        graph.query.assert_not_awaited()
        graph.ro_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_refuses_empty_new_content(self, mock_config, make_backend):
        """Redaction is content-PRESERVING: emptying the episode would be a
        second silent destruction layered on the first."""
        backend = make_backend(mock_config)
        graph = _wire_graph(backend, rows=[[_DIRTY_CONTENT]])

        with pytest.raises(ValueError, match='new_content'):
            await backend.redact_episode_content(
                _EPISODE_UUID, group_id=_GROUP_ID, new_content='   ',
            )
        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_prose_quoting_the_escaped_newline_is_accepted(
        self, mock_config, make_backend,
    ):
        """The tasks 2938/2939 false-positive shape is NOT a leak — a
        redaction may legitimately describe the fragment it removed."""
        backend = make_backend(mock_config)
        graph = _wire_graph(backend, rows=[[_DIRTY_CONTENT]])
        quoted = (
            'Redacted a fragment that looked like `'
            + _CLOSE_DESCRIPTION + '\\n' + _OPEN_PRIORITY + 'low`.'
        )

        result = await backend.redact_episode_content(
            _EPISODE_UUID, group_id=_GROUP_ID, new_content=quoted,
        )

        assert result['new_content'] == quoted
        graph.query.assert_awaited_once()


# ---------------------------------------------------------------------------
# MemoryService.redact_episode_content
# ---------------------------------------------------------------------------


class TestMemoryServiceRedactEpisodeContent:
    """Delegates to the graphiti backend with journaling, mirroring
    delete_episode/update_edge."""

    @pytest.fixture
    def service(self, mock_config):
        from fused_memory.services.memory_service import MemoryService
        svc = MemoryService(mock_config)
        svc.graphiti = MagicMock()
        svc.graphiti.redact_episode_content = AsyncMock(return_value={
            'uuid': _EPISODE_UUID,
            'group_id': _GROUP_ID,
            'old_content': _DIRTY_CONTENT,
            'new_content': _CLEAN_CONTENT,
        })
        svc.mem0 = MagicMock()
        return svc

    @pytest.mark.asyncio
    async def test_delegates_to_graphiti_backend(self, service):
        result = await service.redact_episode_content(
            episode_uuid=_EPISODE_UUID,
            new_content=_CLEAN_CONTENT,
            project_id=_GROUP_ID,
        )

        service.graphiti.redact_episode_content.assert_awaited_once_with(
            _EPISODE_UUID, group_id=_GROUP_ID, new_content=_CLEAN_CONTENT,
        )
        assert result['status'] == 'redacted'
        assert result['store'] == 'graphiti'
        assert result['old_content'] == _DIRTY_CONTENT
        assert result['new_content'] == _CLEAN_CONTENT

    @pytest.mark.asyncio
    async def test_does_not_delete_the_episode(self, service):
        """The service layer must not reach for delete_episode either."""
        service.graphiti.remove_episode = AsyncMock()
        await service.redact_episode_content(
            episode_uuid=_EPISODE_UUID,
            new_content=_CLEAN_CONTENT,
            project_id=_GROUP_ID,
        )
        service.graphiti.remove_episode.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_logs_via_write_journal_when_present(self, service):
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        mock_journal.log_backend_op = AsyncMock()
        service.set_write_journal(mock_journal)

        await service.redact_episode_content(
            episode_uuid=_EPISODE_UUID,
            new_content=_CLEAN_CONTENT,
            project_id=_GROUP_ID,
        )

        mock_journal.log_write_op.assert_awaited_once()
        kwargs = mock_journal.log_write_op.call_args[1]
        assert kwargs['operation'] == 'redact_episode_content'
        assert kwargs['project_id'] == _GROUP_ID
        assert kwargs['params']['episode_uuid'] == _EPISODE_UUID

    @pytest.mark.asyncio
    async def test_works_without_journal(self, service):
        result = await service.redact_episode_content(
            episode_uuid=_EPISODE_UUID,
            new_content=_CLEAN_CONTENT,
            project_id=_GROUP_ID,
        )
        assert result['status'] == 'redacted'

    @pytest.mark.asyncio
    async def test_backend_failure_propagates(self, service):
        service.graphiti.redact_episode_content = AsyncMock(
            side_effect=NodeNotFoundError('Episodic node not found: nope')
        )
        with pytest.raises(NodeNotFoundError):
            await service.redact_episode_content(
                episode_uuid='nope',
                new_content=_CLEAN_CONTENT,
                project_id=_GROUP_ID,
            )


# ---------------------------------------------------------------------------
# MCP tool redact_episode_content
# ---------------------------------------------------------------------------


def _parse_tool_result(result):
    """Normalize a call_tool result (list of content blocks or a dict) to a dict."""
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


class TestRedactEpisodeContentMcpTool:
    """Registered, canonicalization prologue applied, delegates to the service."""

    @pytest.fixture
    def mock_service(self):
        svc = AsyncMock()
        svc.redact_episode_content = AsyncMock(return_value={
            'status': 'redacted',
            'store': 'graphiti',
            'uuid': _EPISODE_UUID,
            'old_content': _DIRTY_CONTENT,
            'new_content': _CLEAN_CONTENT,
        })
        return svc

    @pytest.fixture
    def mcp_server(self, mock_service):
        from fused_memory.server.tools import create_mcp_server
        return create_mcp_server(mock_service)

    @pytest.mark.asyncio
    async def test_tool_is_registered(self, mcp_server):
        tool_names = [t.name for t in await mcp_server.list_tools()]
        assert 'redact_episode_content' in tool_names

    @pytest.mark.asyncio
    async def test_delegates_to_memory_service(self, mcp_server, mock_service):
        await mcp_server._tool_manager.call_tool(
            'redact_episode_content',
            {
                'episode_uuid': _EPISODE_UUID,
                'new_content': _CLEAN_CONTENT,
                'project_id': _GROUP_ID,
            },
        )
        mock_service.redact_episode_content.assert_awaited_once()
        kwargs = mock_service.redact_episode_content.call_args[1]
        assert kwargs.get('episode_uuid') == _EPISODE_UUID
        assert kwargs.get('new_content') == _CLEAN_CONTENT
        assert kwargs.get('project_id') == _GROUP_ID

    @pytest.mark.asyncio
    async def test_canonicalizes_project_id(self, mcp_server, mock_service):
        await mcp_server._tool_manager.call_tool(
            'redact_episode_content',
            {
                'episode_uuid': _EPISODE_UUID,
                'new_content': _CLEAN_CONTENT,
                'project_id': 'dark-factory',
            },
        )
        kwargs = mock_service.redact_episode_content.call_args[1]
        assert kwargs.get('project_id') == _GROUP_ID

    @pytest.mark.asyncio
    async def test_empty_project_id_returns_error(self, mcp_server, mock_service):
        result = await mcp_server._tool_manager.call_tool(
            'redact_episode_content',
            {'episode_uuid': _EPISODE_UUID, 'new_content': _CLEAN_CONTENT, 'project_id': ''},
        )
        parsed = _parse_tool_result(result)
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.redact_episode_content.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_episode_uuid_returns_error(self, mcp_server, mock_service):
        result = await mcp_server._tool_manager.call_tool(
            'redact_episode_content',
            {'episode_uuid': '  ', 'new_content': _CLEAN_CONTENT, 'project_id': _GROUP_ID},
        )
        parsed = _parse_tool_result(result)
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.redact_episode_content.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_new_content_returns_error(self, mcp_server, mock_service):
        result = await mcp_server._tool_manager.call_tool(
            'redact_episode_content',
            {'episode_uuid': _EPISODE_UUID, 'new_content': '   ', 'project_id': _GROUP_ID},
        )
        parsed = _parse_tool_result(result)
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.redact_episode_content.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exception_returns_error_dict(self, mcp_server, mock_service):
        mock_service.redact_episode_content = AsyncMock(
            side_effect=RuntimeError('FalkorDB connection failed')
        )
        result = await mcp_server._tool_manager.call_tool(
            'redact_episode_content',
            {
                'episode_uuid': _EPISODE_UUID,
                'new_content': _CLEAN_CONTENT,
                'project_id': _GROUP_ID,
            },
        )
        parsed = _parse_tool_result(result)
        assert 'error' in parsed
        assert 'FalkorDB connection failed' in parsed['error']


# ---------------------------------------------------------------------------
# Disallow-list stage membership
# ---------------------------------------------------------------------------


class TestDisallowListForRedactEpisodeContent:
    """redact_episode_content is a memory write: it must be in
    DISALLOW_MEMORY_WRITES (blocked in Stage 3 read-only) but NOT in
    STAGE1_DISALLOWED (Stage 1 consolidation may call it)."""

    def test_in_disallow_memory_writes(self):
        from fused_memory.reconciliation.cli_stage_runner import DISALLOW_MEMORY_WRITES
        assert 'mcp__fused-memory__redact_episode_content' in DISALLOW_MEMORY_WRITES

    def test_not_in_stage1_disallowed(self):
        from fused_memory.reconciliation.cli_stage_runner import STAGE1_DISALLOWED
        assert 'mcp__fused-memory__redact_episode_content' not in STAGE1_DISALLOWED

    def test_in_stage3_disallowed(self):
        from fused_memory.reconciliation.cli_stage_runner import STAGE3_DISALLOWED
        assert 'mcp__fused-memory__redact_episode_content' in STAGE3_DISALLOWED
