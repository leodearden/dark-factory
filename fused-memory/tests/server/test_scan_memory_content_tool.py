"""Behaviour tests for the scan_memory_content MCP tool (task 3083, WORK b).

A read-only LITERAL substring scan over Qdrant payload text — neither
semantic ranking (``search``) nor metadata equality
(``get_memories_by_metadata``). It is the capability whose absence made the
tool-call XML leak corpus unsweepable: a leaked fragment carries almost no
semantic signal, so a live 2026-07-26 semantic probe returned zero.

Mirrors test_get_memories_by_metadata_tool.py, plus the B1/B2 canonicalization
cases from test_mcp_boundary_canonicalization.py (same
``_canonicalize_project_id_arg`` + ``validate_project_id`` prologue) and the
house real-MemoryService + real-Mem0Backend timeout test — a fully-mocked
service would pass even against a swallow-to-[] regression, which is exactly
the invariant this tool must not break.

Sentinel literals use the ``\\x3c`` escape for ``<`` (task 3083 plan pre-1).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from fused_memory.backends.mem0_client import Mem0Backend
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import MemoryService

_PROJECT_ID = 'dark_factory'
_KNOWN_PROJECTS = {'dark_factory': '/home/leo/src/dark-factory'}

_CLOSE_CONTENT = '\x3c/content>'
_CLOSE_INVOKE = '\x3c/invoke>'
_LEAK = _CLOSE_CONTENT + '\n' + _CLOSE_INVOKE

_SAMPLE_RESULT = {
    'matches': [
        {
            'id': 'mem-1',
            'created_at': '2026-07-27T00:00:00Z',
            'matched_fragments': [_LEAK],
            'excerpt': 'A memory about the grace window.' + _LEAK,
            'metadata': {'category': 'observations_and_summaries'},
        }
    ],
    'scanned': 19321,
    'truncated': False,
}

_EMPTY_RESULT = {'matches': [], 'scanned': 19321, 'truncated': False}


class TestScanMemoryContentTool:
    @pytest.mark.asyncio
    async def test_happy_path_returns_matches_and_forwards_args(self):
        mock_service = AsyncMock()
        mock_service.scan_memory_content = AsyncMock(return_value=_SAMPLE_RESULT)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'scan_memory_content', {'project_id': _PROJECT_ID}
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' not in result, f'Unexpected error: {result!r}'
        assert result['matches'] == _SAMPLE_RESULT['matches']
        mock_service.scan_memory_content.assert_called_once_with(
            project_id=_PROJECT_ID,
            needles=None,
            filters=None,
            exhaustive=False,
            limit=None,
        )

    @pytest.mark.asyncio
    async def test_all_arguments_are_threaded_through(self):
        mock_service = AsyncMock()
        mock_service.scan_memory_content = AsyncMock(return_value=_EMPTY_RESULT)
        server = create_mcp_server(mock_service)

        await server._tool_manager.call_tool(
            'scan_memory_content',
            {
                'project_id': _PROJECT_ID,
                'needles': [_CLOSE_CONTENT],
                'filters': {'category': 'procedural_knowledge'},
                'exhaustive': True,
                'limit': 500,
            },
        )

        mock_service.scan_memory_content.assert_called_once_with(
            project_id=_PROJECT_ID,
            needles=[_CLOSE_CONTENT],
            filters={'category': 'procedural_knowledge'},
            exhaustive=True,
            limit=500,
        )

    @pytest.mark.asyncio
    async def test_empty_match_list_is_success_not_error(self):
        """A clean corpus is the GOOD outcome, and must be distinguishable from
        a failed scan — otherwise 'no leaks' and 'the scan broke' look alike."""
        mock_service = AsyncMock()
        mock_service.scan_memory_content = AsyncMock(return_value=_EMPTY_RESULT)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'scan_memory_content', {'project_id': _PROJECT_ID}
        )

        assert result.get('matches') == []
        assert result.get('scanned') == 19321
        assert 'error' not in result, f'Unexpected error: {result!r}'

    @pytest.mark.asyncio
    async def test_return_shape_carries_the_self_disclosure_fields(self):
        mock_service = AsyncMock()
        mock_service.scan_memory_content = AsyncMock(
            return_value={'matches': [], 'scanned': 10, 'truncated': True}
        )
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'scan_memory_content',
            {'project_id': _PROJECT_ID, 'limit': 10, 'exhaustive': True},
        )

        assert result['matches'] == []
        assert result['scanned'] == 10
        assert result['truncated'] is True
        assert result['exhaustive'] is True
        assert result['limit'] == 10
        assert result['project_id'] == _PROJECT_ID

    @pytest.mark.asyncio
    async def test_needles_none_defaults_at_the_service_layer(self):
        """The tool must not substitute its own default list — the sentinels
        live in one place (the shared detector module)."""
        mock_service = AsyncMock()
        mock_service.scan_memory_content = AsyncMock(return_value=_EMPTY_RESULT)
        server = create_mcp_server(mock_service)

        await server._tool_manager.call_tool('scan_memory_content', {'project_id': _PROJECT_ID})

        assert mock_service.scan_memory_content.call_args.kwargs['needles'] is None

    @pytest.mark.asyncio
    async def test_invalid_project_id_returns_validation_error_without_calling_service(self):
        mock_service = AsyncMock()
        mock_service.scan_memory_content = AsyncMock(return_value=_EMPTY_RESULT)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'scan_memory_content', {'project_id': 'bad id!'}
        )

        assert result.get('error_type') == 'ValidationError', f'got: {result!r}'
        mock_service.scan_memory_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_hyphen_spelling_is_canonicalized_before_the_service_call(self):
        """B1 from test_mcp_boundary_canonicalization: same prologue, same rule."""
        mock_service = AsyncMock()
        mock_service.scan_memory_content = AsyncMock(return_value=_EMPTY_RESULT)
        server = create_mcp_server(mock_service, known_projects=_KNOWN_PROJECTS)

        result = await server._tool_manager.call_tool(
            'scan_memory_content', {'project_id': 'dark-factory'}
        )

        assert 'error' not in result, f'Unexpected error: {result!r}'
        assert mock_service.scan_memory_content.call_args.kwargs['project_id'] == 'dark_factory'

    @pytest.mark.asyncio
    async def test_path_shaped_project_id_is_rejected(self):
        """B2 from test_mcp_boundary_canonicalization."""
        mock_service = AsyncMock()
        mock_service.scan_memory_content = AsyncMock(return_value=_EMPTY_RESULT)
        server = create_mcp_server(mock_service, known_projects=_KNOWN_PROJECTS)

        result = await server._tool_manager.call_tool(
            'scan_memory_content', {'project_id': '/home/leo/src/dark-factory'}
        )

        assert result.get('error_type') == 'PathShapedProjectIdError', f'got: {result!r}'
        mock_service.scan_memory_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_service_exception_becomes_an_error_dict(self):
        mock_service = AsyncMock()
        mock_service.scan_memory_content = AsyncMock(side_effect=RuntimeError('qdrant down'))
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'scan_memory_content', {'project_id': _PROJECT_ID}
        )

        assert 'error' in result, f'Expected an error dict, got: {result!r}'
        assert 'matches' not in result

    @pytest.mark.asyncio
    async def test_tool_registered(self):
        server = create_mcp_server(AsyncMock())

        tools = await server.list_tools()

        assert 'scan_memory_content' in {t.name for t in tools}


class TestScanMemoryContentTimeoutSurfaces:
    """Real MemoryService + real Mem0Backend, patching ONLY _get_async_qdrant.

    A fully-mocked service would pass even against a swallow-to-[] regression
    in the backend — and a timed-out scan reported as a clean corpus is the
    single worst failure this tool could have, since the whole point is
    establishing a true incidence rate.
    """

    @pytest.mark.asyncio
    async def test_timeout_surfaces_as_an_error_not_an_empty_match_list(self, mock_config):
        service = MemoryService(mock_config)
        service.mem0 = Mem0Backend(mock_config)
        server = create_mcp_server(service)

        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(side_effect=TimeoutError('too slow'))

        with patch.object(
            service.mem0, '_get_async_qdrant', AsyncMock(return_value=mock_client)
        ):
            result = await server._tool_manager.call_tool(
                'scan_memory_content', {'project_id': _PROJECT_ID}
            )

        assert 'error' in result, f'Expected an error dict, got: {result!r}'
        assert result.get('error_type') == 'TimeoutError', f'got: {result!r}'
        assert result.get('matches') != [], 'a timed-out scan must not look clean'
