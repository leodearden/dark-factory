"""CGL-β (task 2268, seam S3): every MCP memory-tool prologue canonicalizes
project_id via the shared α helper (canonicalize_project_id) BEFORE
validate_project_id / _known_project_gate / any downstream service call.

Harness mirrors tests/test_tools_validation.py's TestKnownProjectRegistryGate:
create_mcp_server(AsyncMock(), known_projects={'dark_factory': '/home/leo/src/dark-factory'})
+ server._tool_manager.call_tool(name, kwargs), asserting on
mock_service.<method>.call_args[1] / assert_not_called().

Two behaviors pinned throughout:
- B1 (accept-flip): a hyphen spelling of a known project_id (e.g. 'dark-factory')
  is normalized to the underscore-canonical registry key ('dark_factory') and
  the call reaches the service — a deliberate behavior flip from the pre-S3
  gate-rejected state (PRD decision 1).
- B2 (loud path-shape rejection): a path-shaped project_id (e.g.
  '-home-leo-src-x') is rejected with error_type == 'PathShapedProjectIdError'
  naming the offending value, and the downstream service is never called.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.server.tools import create_mcp_server

_KNOWN_PROJECTS = {'dark_factory': '/home/leo/src/dark-factory'}
_PATH_SHAPED = '-home-leo-src-x'


class TestGatedWriteCanonicalization:
    """Gated writes: add_episode, add_memory (also delete_memory, delete_episode,
    update_edge — covered by the step-7 completeness sweep).
    """

    @pytest.mark.asyncio
    async def test_add_memory_accepts_hyphen_spelling_of_known_project(self):
        """B1: add_memory(project_id='dark-factory') is ACCEPTED and normalized.

        Pre-S3 this was gate-rejected because 'dark-factory' != 'dark_factory'
        in the registry. Post-S3, canonicalization runs before the gate, so the
        hyphen spelling resolves to the registered project and the service sees
        the underscore-canonical form.
        """
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service, known_projects=_KNOWN_PROJECTS)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {'content': 'x', 'project_id': 'dark-factory', 'metadata': {}},
        )

        mock_service.add_memory.assert_called_once()
        assert mock_service.add_memory.call_args[1]['project_id'] == 'dark_factory', (
            f'Expected canonicalized project_id to reach the service; result={result!r}'
        )

    @pytest.mark.asyncio
    async def test_add_memory_rejects_path_shaped_project_id(self):
        """B2: add_memory(project_id='-home-leo-src-x') is rejected loudly.

        Must surface PathShapedProjectIdError (not the generic 'unknown project'
        ValidationError a raw registry-miss would produce) and must never reach
        the service.
        """
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service, known_projects=_KNOWN_PROJECTS)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {'content': 'x', 'project_id': _PATH_SHAPED, 'metadata': {}},
        )

        assert result.get('error_type') == 'PathShapedProjectIdError', (
            f'Expected PathShapedProjectIdError, got: {result!r}'
        )
        assert _PATH_SHAPED in result.get('error', ''), (
            f'Expected offending value {_PATH_SHAPED!r} named in error, got: {result!r}'
        )
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_episode_rejects_path_shaped_project_id(self):
        """B2: add_episode(project_id='-home-leo-src-x') is rejected loudly.

        Mirrors test_add_memory_rejects_path_shaped_project_id for the other
        gated-write ingestion entry point.
        """
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service, known_projects=_KNOWN_PROJECTS)

        result = await server._tool_manager.call_tool(
            'add_episode',
            {'content': 'x', 'project_id': _PATH_SHAPED},
        )

        assert result.get('error_type') == 'PathShapedProjectIdError', (
            f'Expected PathShapedProjectIdError, got: {result!r}'
        )
        assert _PATH_SHAPED in result.get('error', ''), (
            f'Expected offending value {_PATH_SHAPED!r} named in error, got: {result!r}'
        )
        mock_service.add_episode.assert_not_called()
