"""Integration tests for the write-time near-duplicate guard in add_memory (task 2467).

Mirrors test_add_memory_snapshot_gate.py's harness: an AsyncMock memory
service wired through create_mcp_server, invoked via
server._tool_manager.call_tool('add_memory', {...}).
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.tools import create_mcp_server

_PROJECT_ID = 'dark_factory'
_CONTENT = 'canonical .task gitignore gotcha'


def _near_duplicate_result(
    id_: str = 'm1',
    score: float = 0.97,
    content: str = 'canonical .task gitignore gotcha (existing entry)',
    category: MemoryCategory = MemoryCategory.procedural_knowledge,
) -> MemoryResult:
    return MemoryResult(
        id=id_,
        content=content,
        category=category,
        source_store=SourceStore.mem0,
        relevance_score=score,
    )


def _configure_reconciliation(mock_service: AsyncMock, **fields) -> None:
    """Stand in for memory_service.config.reconciliation with a plain namespace.

    A plain types.SimpleNamespace (rather than more AsyncMock/MagicMock
    attribute chaining) means the resolver sees real bool/float values for
    exactly the fields under test.
    """
    mock_service.config = types.SimpleNamespace(
        reconciliation=types.SimpleNamespace(**fields)
    )


def _configure_pass_through_add_memory(mock_service: AsyncMock) -> None:
    """Configure mock_service.add_memory to return a dict-dumpable result.

    Mirrors test_add_memory_snapshot_gate.py: an unspecced AsyncMock's default
    return value chains AsyncMock all the way down, so ``result.model_dump()``
    would itself be an unawaited coroutine unless the return value is an
    explicit MagicMock with model_dump configured.
    """
    mem_result = MagicMock()
    mem_result.model_dump.return_value = {'id': 'ok'}
    mock_service.add_memory.return_value = mem_result


class TestAddMemoryNearDuplicateGate:
    """Write-gate: high-similarity procedural_knowledge writes are soft-blocked."""

    @pytest.mark.asyncio
    async def test_rejects_near_duplicate_procedural_knowledge_write(self):
        """A high-similarity search match blocks the write with a structured dict.

        The gate must return the standard soft-block error dict and must NOT
        call memory_service.add_memory.
        """
        mock_service = AsyncMock()
        matched = _near_duplicate_result(id_='m1', score=0.97)
        mock_service.search.return_value = [matched]
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result.get('error') == 'procedural_knowledge_near_duplicate_write_blocked', (
            f'Expected near-duplicate block, got: {result!r}'
        )
        assert result.get('error_type') == 'ProceduralKnowledgeNearDuplicateWriteRejected', (
            f'Expected error_type=ProceduralKnowledgeNearDuplicateWriteRejected, got: {result!r}'
        )
        assert result.get('agent_id') == 'claude-interactive', (
            f'Expected agent_id echoed, got: {result!r}'
        )
        assert result.get('content_excerpt') == _CONTENT[:200], (
            f'Expected content_excerpt=content[:200], got: {result!r}'
        )
        assert result.get('matched_memory_id') == 'm1', (
            f'Expected matched_memory_id=m1, got: {result!r}'
        )
        assert isinstance(result.get('similarity'), int | float), (
            f'Expected a numeric similarity, got: {result!r}'
        )
        assert result.get('matched_excerpt') == matched.content[:200], (
            f'Expected matched_excerpt=matched content[:200], got: {result!r}'
        )
        assert result.get('threshold') == 0.92, (
            f'Expected default threshold 0.92 echoed, got: {result!r}'
        )
        assert result.get('hint'), f'Expected a non-empty hint, got: {result!r}'
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_write_when_top_match_below_threshold(self):
        """A search match scoring well below the threshold does not block the write."""
        mock_service = AsyncMock()
        mock_service.search.return_value = [_near_duplicate_result(id_='m1', score=0.50)]
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'procedural_knowledge_near_duplicate_write_blocked', (
            f'Gate must not fire below threshold; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_guard_does_not_run_for_non_procedural_category(self):
        """The guard is scoped to category='procedural_knowledge' only.

        Even a high-similarity match must not trigger a search (or a block)
        when the write's own category is something else.
        """
        mock_service = AsyncMock()
        mock_service.search.return_value = [_near_duplicate_result(id_='m1', score=0.97)]
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'observations_and_summaries',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'procedural_knowledge_near_duplicate_write_blocked', (
            f'Gate must not fire for a non-procedural_knowledge category; got: {result!r}'
        )
        mock_service.search.assert_not_called()
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_allow_near_duplicate_override_bypasses_guard(self):
        """metadata={'allow_near_duplicate': True} bypasses the guard entirely.

        Even a high-similarity match must not block the write when the caller
        has explicitly opted in to writing a near-duplicate.
        """
        mock_service = AsyncMock()
        mock_service.search.return_value = [_near_duplicate_result(id_='m1', score=0.97)]
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
                'metadata': {'allow_near_duplicate': True},
            },
        )

        assert result.get('error') != 'procedural_knowledge_near_duplicate_write_blocked', (
            f'Override must bypass the guard; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_fails_open_when_search_raises(self):
        """A search error must never block the write — fail-open, not fail-closed."""
        mock_service = AsyncMock()
        mock_service.search.side_effect = TimeoutError('search backend unavailable')
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'procedural_knowledge_near_duplicate_write_blocked', (
            f'A search failure must not block the write; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_wiring_error_from_search_is_not_silently_absorbed(self):
        """A TypeError from memory_service.search must surface, not fail open.

        Unlike a transient backend failure (TimeoutError, see
        test_fails_open_when_search_raises), a TypeError here signals a
        wiring/programming bug -- e.g. a future refactor renaming the
        search() stores/categories kwargs. The guard must not swallow this
        into fail-open (which would silently disable the guard behind
        nothing but a WARNING log); it must propagate to the standard
        @mcp_tool_errors() error-dict response instead, and the write must
        NOT proceed.
        """
        mock_service = AsyncMock()
        mock_service.search.side_effect = TypeError(
            "search() got an unexpected keyword argument 'stores'"
        )
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error_type') == 'TypeError', (
            f'A wiring-level TypeError must surface as an error, not be '
            f'silently absorbed into fail-open; got: {result!r}'
        )
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_write_when_search_returns_empty(self):
        """No candidate results at all → no match → write proceeds."""
        mock_service = AsyncMock()
        mock_service.search.return_value = []
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'procedural_knowledge_near_duplicate_write_blocked', (
            f'Empty search results must not block the write; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_override_flag_is_stripped_from_persisted_metadata(self):
        """allow_near_duplicate is a write-time-only control flag.

        It must never leak into the metadata handed to
        memory_service.add_memory -- otherwise it is stamped on the stored
        memory forever and shows up in future search results' metadata.
        """
        mock_service = AsyncMock()
        mock_service.search.return_value = [_near_duplicate_result(id_='m1', score=0.97)]
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
                'metadata': {'allow_near_duplicate': True, 'keep_me': 'yes'},
            },
        )

        mock_service.add_memory.assert_called_once()
        persisted_metadata = mock_service.add_memory.call_args.kwargs.get('metadata')
        assert persisted_metadata == {'keep_me': 'yes'}, (
            f'allow_near_duplicate must be stripped before persistence; got: '
            f'{persisted_metadata!r}'
        )

    @pytest.mark.asyncio
    async def test_guard_skipped_when_category_is_none(self):
        """The guard only covers explicit category='procedural_knowledge'.

        A write that omits category (auto-classified downstream by the
        service) is not covered by this write-time guard. This is
        best-effort scoping, mirroring the explicit-category pattern the
        existing recon-stage guards (count_snapshot / mixed_temporal_framing
        / conflicting_task_status) already use.
        """
        mock_service = AsyncMock()
        mock_service.search.return_value = [_near_duplicate_result(id_='m1', score=0.97)]
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'procedural_knowledge_near_duplicate_write_blocked', (
            f'Gate must not fire when category is omitted; got: {result!r}'
        )
        mock_service.search.assert_not_called()
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_guard_exempts_recon_stage_agents(self):
        """recon-stage-* agents are exempt from the near-duplicate guard.

        Stage-1 consolidation writes a merged/canonical procedural_knowledge
        entry through this same add_memory tool with an
        agent_id='recon-stage-*' (see reconciliation/prompts/stage1.py). That
        canonical entry is expected to closely resemble the duplicate
        entries it replaces, and there is no guarantee those duplicates are
        deleted before the canonical write lands -- so recon-stage writes
        must not be blocked by this guard, mirroring the trust boundary the
        other recon-stage-scoped guards in this file already use.
        """
        mock_service = AsyncMock()
        mock_service.search.return_value = [_near_duplicate_result(id_='m1', score=0.97)]
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'recon-stage-memory_consolidator',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'procedural_knowledge_near_duplicate_write_blocked', (
            f'Gate must not fire for recon-stage-* agents; got: {result!r}'
        )
        mock_service.search.assert_not_called()
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_pre_check_search_pins_to_mem0_store(self):
        """The guard's pre-check search must pin stores=['mem0'].

        procedural_knowledge lives only in Mem0, but memory_service.search
        defaults to text-based routing (see routing/router.py): content
        containing temporal keywords like "before"/"after" is classified
        QueryType.temporal and routed graphiti-only, so _search_mem0 never
        runs and the guard would silently fail open on exactly the gotcha
        phrasing it exists to catch. Passing stores=['mem0'] short-circuits
        that routing decision via router.route()'s stores_override branch,
        regardless of how the content is phrased.
        """
        mock_service = AsyncMock()
        mock_service.search.return_value = []
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)
        content = (
            'Before writing to write_queue.db verify the path; the merge '
            'worker consumes the stash after advance'
        )

        await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': content,
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        mock_service.search.assert_called_once()
        kwargs = mock_service.search.call_args.kwargs
        assert kwargs.get('stores') == ['mem0'], (
            f"Expected pre-check search pinned to stores=['mem0'], got call_args: "
            f'{mock_service.search.call_args!r}'
        )
        assert kwargs.get('query') == content
        assert kwargs.get('project_id') == _PROJECT_ID
        assert kwargs.get('categories') == ['procedural_knowledge']
        assert kwargs.get('limit') == 5


class TestAddMemoryNearDuplicateGateConfig:
    """The guard's enable flag and threshold are read from config at call time."""

    @pytest.mark.asyncio
    async def test_guard_disabled_via_config_allows_write(self):
        """procedural_knowledge_near_dup_guard_enabled=False skips the guard entirely."""
        mock_service = AsyncMock()
        _configure_reconciliation(
            mock_service,
            procedural_knowledge_near_dup_guard_enabled=False,
            procedural_knowledge_near_dup_threshold=0.92,
        )
        mock_service.search.return_value = [_near_duplicate_result(id_='m1', score=0.97)]
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'procedural_knowledge_near_duplicate_write_blocked', (
            f'Guard must be skipped when disabled via config; got: {result!r}'
        )
        mock_service.search.assert_not_called()
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_configured_threshold_above_match_score_allows_write(self):
        """A configured threshold above the top match score allows the write.

        Default threshold (0.92) would block a 0.97 match, so this only
        passes if the configured 0.99 threshold is actually being read.
        """
        mock_service = AsyncMock()
        _configure_reconciliation(
            mock_service,
            procedural_knowledge_near_dup_guard_enabled=True,
            procedural_knowledge_near_dup_threshold=0.99,
        )
        mock_service.search.return_value = [_near_duplicate_result(id_='m1', score=0.97)]
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'procedural_knowledge_near_duplicate_write_blocked', (
            f'Configured threshold above the match score must allow the write; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_configured_threshold_below_match_score_blocks_write(self):
        """Guard enabled with an explicit threshold still blocks a qualifying match."""
        mock_service = AsyncMock()
        _configure_reconciliation(
            mock_service,
            procedural_knowledge_near_dup_guard_enabled=True,
            procedural_knowledge_near_dup_threshold=0.90,
        )
        mock_service.search.return_value = [_near_duplicate_result(id_='m1', score=0.95)]
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') == 'procedural_knowledge_near_duplicate_write_blocked', (
            f'Expected near-duplicate block, got: {result!r}'
        )
        assert result.get('threshold') == 0.90, f'Expected configured threshold echoed, got: {result!r}'
        mock_service.add_memory.assert_not_called()
