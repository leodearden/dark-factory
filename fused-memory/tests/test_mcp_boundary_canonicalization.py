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


class TestReadCanonicalization:
    """Reads: search, get_entity, get_queue_stats (also count_memories_by_metadata,
    get_memories_by_metadata, get_entity_by_uuid, get_episodes — covered by the
    step-7 completeness sweep).
    """

    @pytest.mark.asyncio
    async def test_search_routes_hyphen_and_underscore_spellings_identically(self):
        """B1: search reaches the service with the canonical project_id for
        EITHER spelling of a known project — 'dark-factory' and 'dark_factory'
        must both resolve to 'dark_factory'.
        """
        mock_service = AsyncMock()
        mock_service.search.return_value = []
        server = create_mcp_server(mock_service, known_projects=_KNOWN_PROJECTS)

        await server._tool_manager.call_tool(
            'search', {'query': 'q', 'project_id': 'dark-factory'}
        )
        await server._tool_manager.call_tool(
            'search', {'query': 'q', 'project_id': 'dark_factory'}
        )

        assert mock_service.search.call_count == 2
        for call in mock_service.search.call_args_list:
            assert call[1]['project_id'] == 'dark_factory', (
                f'Expected canonicalized project_id on every call; calls={mock_service.search.call_args_list!r}'
            )

    @pytest.mark.asyncio
    async def test_search_rejects_path_shaped_project_id(self):
        """B2: search(project_id='-home-leo-src-x') is rejected loudly, no service call."""
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service, known_projects=_KNOWN_PROJECTS)

        result = await server._tool_manager.call_tool(
            'search', {'query': 'q', 'project_id': _PATH_SHAPED}
        )

        assert result.get('error_type') == 'PathShapedProjectIdError', (
            f'Expected PathShapedProjectIdError, got: {result!r}'
        )
        mock_service.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_entity_routes_hyphen_spelling_canonicalized(self):
        """B1: get_entity(project_id='dark-factory') reaches the service as 'dark_factory'."""
        mock_service = AsyncMock()
        mock_service.get_entity.return_value = {'nodes': [], 'edges': []}
        server = create_mcp_server(mock_service, known_projects=_KNOWN_PROJECTS)

        await server._tool_manager.call_tool(
            'get_entity', {'name': 'e', 'project_id': 'dark-factory'}
        )

        mock_service.get_entity.assert_called_once()
        assert mock_service.get_entity.call_args[1]['project_id'] == 'dark_factory'

    @pytest.mark.asyncio
    async def test_get_queue_stats_none_project_id_passes_through(self):
        """get_queue_stats(project_id=None) keeps global scope: group_id=None reaches durable_queue."""
        mock_service = AsyncMock()
        mock_service.durable_queue.get_stats = AsyncMock(return_value={'counts': {}})
        server = create_mcp_server(mock_service, known_projects=_KNOWN_PROJECTS)

        await server._tool_manager.call_tool('get_queue_stats', {'project_id': None})

        mock_service.durable_queue.get_stats.assert_called_once_with(group_id=None)

    @pytest.mark.asyncio
    async def test_get_queue_stats_hyphen_spelling_canonicalized(self):
        """B1: get_queue_stats(project_id='dark-factory') forwards group_id='dark_factory'."""
        mock_service = AsyncMock()
        mock_service.durable_queue.get_stats = AsyncMock(return_value={'counts': {}})
        server = create_mcp_server(mock_service, known_projects=_KNOWN_PROJECTS)

        await server._tool_manager.call_tool('get_queue_stats', {'project_id': 'dark-factory'})

        mock_service.durable_queue.get_stats.assert_called_once_with(group_id='dark_factory')


class TestUngatedMutatorCanonicalization:
    """Ungated graph mutators: refresh_entity_summary (also set_entity_summary,
    rename_entity, merge_entities, delete_entity, rebuild_entity_summaries,
    replay_to_graphiti — covered by the step-7 completeness sweep).
    """

    @pytest.mark.asyncio
    async def test_refresh_entity_summary_accepts_hyphen_spelling(self):
        """B1: refresh_entity_summary(project_id='dark-factory') is ACCEPTED and normalized."""
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service, known_projects=_KNOWN_PROJECTS)

        await server._tool_manager.call_tool(
            'refresh_entity_summary',
            {'project_id': 'dark-factory', 'entity_name': 'Foo'},
        )

        mock_service.refresh_entity_summary.assert_called_once()
        assert mock_service.refresh_entity_summary.call_args[1]['project_id'] == 'dark_factory'

    @pytest.mark.asyncio
    async def test_refresh_entity_summary_rejects_path_shaped_project_id(self):
        """B2: refresh_entity_summary(project_id='-home-leo-src-x') is rejected loudly.

        Deliberately omits both entity_uuid and entity_name. refresh_entity_summary's
        own required-arg guard (``if not entity_uuid and not entity_name``) would
        fire and return a generic (non-PathShapedProjectIdError) error dict if
        canonicalization ran AFTER it. Asserting PathShapedProjectIdError here
        therefore proves canonicalization precedes BOTH validate_project_id and
        that tool-specific required-arg check — not just the former, which a call
        that also supplied entity_name would leave unproven.
        """
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service, known_projects=_KNOWN_PROJECTS)

        result = await server._tool_manager.call_tool(
            'refresh_entity_summary',
            {'project_id': _PATH_SHAPED},
        )

        assert result.get('error_type') == 'PathShapedProjectIdError', (
            f'Expected PathShapedProjectIdError, got: {result!r}'
        )
        assert _PATH_SHAPED in result.get('error', ''), (
            f'Expected offending value {_PATH_SHAPED!r} named in error, got: {result!r}'
        )
        mock_service.refresh_entity_summary.assert_not_called()


def _resolve_mock_attr(mock_service, dotted_path: str):
    """Resolve a dotted attribute path (e.g. 'durable_queue.get_stats') off mock_service."""
    obj = mock_service
    for part in dotted_path.split('.'):
        obj = getattr(obj, part)
    return obj


# (tool_name, extra kwargs supplying the tool's mandatory no-default args other
# than project_id, dotted mock_service attr path the tool ultimately calls).
# Covers all 19 in-scope memory tools: the 5 gated writes, 7 reads (including
# get_queue_stats), and 7 ungated mutators.
_ALL_MEMORY_TOOL_SWEEP_CASES = [
    # Gated writes
    ('add_episode', {'content': 'x'}, 'add_episode'),
    ('add_memory', {'content': 'x'}, 'add_memory'),
    ('delete_memory', {'memory_id': 'm1', 'store': 'mem0'}, 'delete_memory'),
    ('delete_episode', {'episode_id': 'e1'}, 'delete_episode'),
    ('update_edge', {'edge_uuid': 'e1'}, 'update_edge'),
    # Reads
    ('search', {'query': 'q'}, 'search'),
    ('count_memories_by_metadata', {'filters': {}}, 'count_memories_by_metadata'),
    ('get_memories_by_metadata', {'filters': {}}, 'get_memories_by_metadata'),
    ('get_entity', {'name': 'e'}, 'get_entity'),
    ('get_entity_by_uuid', {'entity_uuid': 'u1'}, 'get_entity_by_uuid'),
    ('get_episodes', {}, 'get_episodes'),
    ('get_queue_stats', {}, 'durable_queue.get_stats'),
    # Ungated mutators
    ('refresh_entity_summary', {'entity_name': 'Foo'}, 'refresh_entity_summary'),
    ('set_entity_summary', {'entity_name': 'Foo', 'summary': 's'}, 'set_entity_summary'),
    ('rename_entity', {'entity_name': 'Foo', 'new_name': 'Bar'}, 'rename_entity'),
    ('merge_entities', {'deprecated_uuid': 'd1', 'surviving_uuid': 's2'}, 'merge_entities'),
    ('delete_entity', {'entity_uuid': 'u1'}, 'delete_entity'),
    ('rebuild_entity_summaries', {}, 'rebuild_entity_summaries'),
    ('replay_to_graphiti', {}, 'replay_from_store'),
]

assert len(_ALL_MEMORY_TOOL_SWEEP_CASES) == 19, (
    'Sweep must cover exactly the 19 in-scope memory tools (PRD seam S3) — '
    'update this table if the in-scope tool count ever changes.'
)


class TestCompletenessSweepAllMemoryTools:
    """Behavioral sweep: EVERY memory tool rejects a path-shaped project_id
    before its downstream service/durable_queue method is ever called.

    Guards the tools not individually B1/B2-tested above, and any tool added
    in the future — canonicalization is the first project_id operation in
    every prologue, ahead of any tool-specific required-arg check, so a
    path-shaped project_id must short-circuit uniformly across all 19.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'tool_name, extra_kwargs, service_attr_path',
        _ALL_MEMORY_TOOL_SWEEP_CASES,
        ids=[case[0] for case in _ALL_MEMORY_TOOL_SWEEP_CASES],
    )
    async def test_path_shaped_project_id_rejected_before_service_call(
        self, tool_name, extra_kwargs, service_attr_path
    ):
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service, known_projects=_KNOWN_PROJECTS)

        result = await server._tool_manager.call_tool(
            tool_name,
            {**extra_kwargs, 'project_id': _PATH_SHAPED},
        )

        assert result.get('error_type') == 'PathShapedProjectIdError', (
            f'{tool_name}: expected PathShapedProjectIdError, got: {result!r}'
        )
        assert _PATH_SHAPED in result.get('error', ''), (
            f'{tool_name}: expected offending value {_PATH_SHAPED!r} named in error, got: {result!r}'
        )
        mock_attr = _resolve_mock_attr(mock_service, service_attr_path)
        mock_attr.assert_not_called()


class TestGetExternalStatusesCharacterization:
    """get_external_statuses (step-8 refactor target): characterizes its
    CURRENT project_id normalization behavior so replacing the inline
    ``project_id.lower().replace('-', '_')`` with the shared, raise-free
    ``_to_underscore_canonical`` primitive (task 2267/S1) is a verified
    no-op refactor rather than a behavior change (see plan design decisions —
    get_external_statuses must keep its sentinel-only contract and must
    never raise PathShapedProjectIdError).
    """

    @pytest.fixture(autouse=True)
    def _passthrough_main_checkout(self, monkeypatch):
        """Stub resolve_main_checkout so a synthetic known_projects root
        doesn't need to be a real git working tree (mirrors the load-bearing
        fixture in tests/test_get_external_statuses.py).
        """
        monkeypatch.setattr(
            'fused_memory.server.tools.resolve_main_checkout',
            lambda p: str(p),
        )

    @pytest.mark.asyncio
    async def test_hyphen_dep_resolves_against_canonical_registry_key(self):
        """'dark-factory:5' resolves against registry key 'dark_factory' — NOT 'unknown_project'."""
        mock_task_interceptor = AsyncMock()
        mock_task_interceptor.get_statuses = AsyncMock(return_value={'5': 'in-progress'})
        server = create_mcp_server(
            AsyncMock(),
            task_interceptor=mock_task_interceptor,
            known_projects=_KNOWN_PROJECTS,
        )

        result = await server._tool_manager.call_tool(
            'get_external_statuses', {'deps': ['dark-factory:5']}
        )

        assert result == {'dark-factory:5': 'in-progress'}, (
            "Expected the hyphen dep to resolve via the canonical registry key "
            f"(not 'unknown_project'); got: {result!r}"
        )

    @pytest.mark.asyncio
    async def test_path_shaped_dep_returns_sentinel_without_raising(self):
        """A path-shaped dep project_id returns a sentinel — no raise, no backend call."""
        mock_task_interceptor = AsyncMock()
        server = create_mcp_server(
            AsyncMock(),
            task_interceptor=mock_task_interceptor,
            known_projects=_KNOWN_PROJECTS,
        )
        dep = f'{_PATH_SHAPED}:5'

        result = await server._tool_manager.call_tool('get_external_statuses', {'deps': [dep]})

        assert result.get(dep) in {'unknown_project', 'malformed'}, (
            f'Expected a sentinel value (no raise) for a path-shaped dep; got: {result!r}'
        )
        mock_task_interceptor.get_statuses.assert_not_called()
