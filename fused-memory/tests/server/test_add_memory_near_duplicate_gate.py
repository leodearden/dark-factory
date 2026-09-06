"""Integration tests for the write-time near-duplicate guard in add_memory (task 2467).

Mirrors test_add_memory_snapshot_gate.py's harness: an AsyncMock memory
service wired through create_mcp_server, invoked via
server._tool_manager.call_tool('add_memory', {...}).
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# Shared note fixtures, owned by the pure-matcher suite. Imported rather than
# copied so the two suites cannot drift: an edit to the straddling note (e.g.
# adding or removing a phrase hit) must move BOTH the matcher assertions there
# and the whole-tool assertions here, instead of leaving one silently
# exercising a different scenario (task 3054, reviewer: duplication).
#
# The bare import resolves because fused-memory/tests/conftest.py inserts the
# tests dir onto sys.path (the same mechanism that makes `from _fm_helpers
# import X` work), and pytest loads a conftest by PATH regardless of import
# mode. Do NOT reason about this from pytest's rootdir/prepend behaviour: the
# repo-root pyproject.toml sets `--import-mode=importlib`, under which a test
# file's own directory is NOT put on sys.path, so a root-bound run would fail
# at collection without that insert. Verified green under -n0, under xdist,
# standalone, and root-bound via `pytest -c pyproject.toml` from the repo root.
from test_config_schema import (
    MERGE_BASE_NEGATIVE_CONTROL_NOTE,
    STRADDLING_WRITE_FIXTURE,
)

from fused_memory.config.schema import ProceduralTopicCluster, ReconciliationConfig
from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.models.scope import Scope
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import RRF_K

_PROJECT_ID = 'dark_factory'
_CONTENT = 'canonical .task gitignore gotcha'

# Content matching the injected test cluster's phrases ('plan-tools' + 'create_plan').
_TOPIC_MATCH_CONTENT = 'run create_plan against the missing plan-tools MCP server'


def _topic_cluster(
    topic_id: str = 'test-topic',
    phrases: tuple[str, ...] = ('plan-tools', 'create_plan'),
    min_phrase_hits: int = 2,
    hint: str = '',
) -> ProceduralTopicCluster:
    return ProceduralTopicCluster(
        topic_id=topic_id,
        phrases=list(phrases),
        min_phrase_hits=min_phrase_hits,
        hint=hint,
    )


def _near_duplicate_result(
    id_: str = 'm1',
    score: float = 0.97,
    content: str = 'canonical .task gitignore gotcha (existing entry)',
    category: MemoryCategory = MemoryCategory.procedural_knowledge,
    store_rank: int = 1,
) -> MemoryResult:
    """Build the POST-RRF result shape the tool really receives from search().

    *score* is the Mem0 COSINE and lands in ``metadata['store_score']``;
    ``relevance_score`` carries the ordinal RRF value, deliberately unrelated
    to it.  Every gate test in this module therefore exercises the real
    post-task-3658 shape, with its threshold expectations unchanged.

    ``RRF_K`` is imported from production rather than restated as the literal
    60, so a retune of the constant carries this fixture with it instead of
    leaving it silently modelling a shape ``search()`` no longer emits.
    """
    return MemoryResult(
        id=id_,
        content=content,
        category=category,
        source_store=SourceStore.mem0,
        relevance_score=1.0 / (RRF_K + store_rank),
        metadata={'store_rank': store_rank, 'store_score': score},
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


def _configure_triage_and_reconciliation(mock_service: AsyncMock, **reconciliation_fields) -> None:
    """Stand in for memory_service.config with BOTH write_triage (enabled)
    and reconciliation namespaces, for exercising the triage-supersedes-gate
    path (task 3127 PRD D2: redirect supersedes reject).

    ``_configure_reconciliation`` alone cannot express this: it builds only
    the ``reconciliation`` section, which is exactly why ``write_triage.enabled``
    reads False (attribute absent -> resolver default) in every other test in
    this module. ``judge_enabled=False`` here is REQUIRED, not cosmetic:
    omitting the attribute makes the resolver default it True, and
    OPENAI_API_KEY is set in this environment, so a middle-band write would
    place a real billed LLM call. Mirrors — without importing, since that
    file sits outside this task's declared 2-file set —
    test_add_memory_write_triage_gate.py's ``_configure_config`` helper shape.
    """
    mock_service.config = types.SimpleNamespace(
        write_triage=types.SimpleNamespace(
            enabled=True,
            candidate_k=20,
            t_high=0.90,
            t_low=0.70,
            judge_enabled=False,
        ),
        reconciliation=types.SimpleNamespace(**reconciliation_fields),
    )


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
        # The reported similarity must be the COSINE, directly comparable to
        # the 'threshold' emitted alongside it (0.97 vs 0.92). Quoting the
        # ordinal RRF value (~0.0164) against a 0.92 threshold would read to
        # the blocked agent as a guard malfunction (task 3658).
        assert result.get('similarity') == pytest.approx(0.97), (
            f"Expected the Mem0 cosine as 'similarity', not the fused RRF "
            f'ordinal, got: {result!r}'
        )
        assert result['similarity'] > result['threshold'], (
            f"'similarity' must be comparable to 'threshold', got: {result!r}"
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

    @pytest.mark.asyncio
    async def test_blocks_on_results_built_by_the_real_search_path(self):
        """End-to-end seam regression for esc-3658-1.

        Every other test here hand-builds the result shape, so all of them
        would have stayed green while the guard went dark in production.  This
        one drives ``memory_service.search`` with results produced by the real
        ``MemoryService._search_mem0`` — the actual RRF stamping, not a fixture
        approximating it — and asserts the write is still soft-blocked.
        """
        from fused_memory.services.memory_service import MemoryService

        # Real _search_mem0, real stamping, stubbed only at the backend seam.
        svc = MemoryService.__new__(MemoryService)
        svc.mem0 = MagicMock()
        svc.mem0.search = AsyncMock(return_value={
            'results': [
                {
                    'id': 'm-real-1',
                    'memory': 'canonical .task gitignore gotcha (existing entry)',
                    'score': 0.97,
                    'metadata': {'category': 'procedural_knowledge'},
                },
            ]
        })
        real_results = await svc._search_mem0(
            _CONTENT,
            Scope(project_id=_PROJECT_ID),
            limit=5,
            categories=['procedural_knowledge'],
        )

        # Sanity-check the premise: this is genuinely the post-RRF shape.
        assert real_results[0].relevance_score < 0.02
        assert real_results[0].metadata['store_score'] == pytest.approx(0.97)

        mock_service = AsyncMock()
        mock_service.search.return_value = real_results
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

        assert result.get('error_type') == 'ProceduralKnowledgeNearDuplicateWriteRejected', (
            f'Real post-RRF search results must still soft-block, got: {result!r}'
        )
        assert result.get('matched_memory_id') == 'm-real-1'
        assert result.get('similarity') == pytest.approx(0.97)
        mock_service.add_memory.assert_not_called()


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


class TestAddMemoryTopicClusterGate:
    """Write-gate: known-contradictory topic clusters are soft-blocked BEFORE
    the cosine search.

    Parametrized over every category `_TOPIC_GUARD_GATED_CATEGORIES` covers
    (task 3430 widened the set from procedural_knowledge-only to also include
    preferences_and_norms) wherever the two categories' expected behaviour is
    identical, so a future change to the shared exemptions can't silently
    diverge between two near-duplicate test classes without a test noticing
    (reviewer: test-duplication). Where the categories' observable behaviour
    genuinely differs — the cosine-search fallthrough is procedural_knowledge
    -only, and an empty clusters list therefore still triggers `search` for
    procedural_knowledge but not for preferences_and_norms — those stay
    separate, explicitly-named cases instead of a same-body parametrize.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize('category', ['procedural_knowledge', 'preferences_and_norms'])
    async def test_blocks_matching_topic_before_cosine_search(self, category):
        """A write matching a cluster's phrases is blocked; search is NOT
        called. The topic pre-check is deterministic (no embedding
        round-trip) and must short-circuit before the cosine near-dup
        search. Also pins the `category` echo on the block dict for both
        gated categories.
        """
        mock_service = AsyncMock()
        _configure_reconciliation(
            mock_service,
            procedural_knowledge_near_dup_guard_enabled=True,
            procedural_knowledge_near_dup_threshold=0.92,
            procedural_knowledge_topic_guard_clusters=[_topic_cluster()],
        )
        mock_service.search.return_value = []
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _TOPIC_MATCH_CONTENT,
                'category': category,
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error_type') == 'ProceduralKnowledgeKnownTopicClusterWriteRejected', (
            f'category={category}: expected topic-cluster block, got: {result!r}'
        )
        assert result.get('error') == 'procedural_knowledge_known_topic_cluster_write_blocked', (
            f'category={category}: expected topic-cluster error key, got: {result!r}'
        )
        assert result.get('topic_id') == 'test-topic', (
            f'category={category}: expected topic_id echoed, got: {result!r}'
        )
        assert result.get('matched_phrases'), (
            f'category={category}: expected matched_phrases, got: {result!r}'
        )
        assert result.get('agent_id') == 'claude-interactive'
        assert result.get('content_excerpt') == _TOPIC_MATCH_CONTENT[:200]
        assert result.get('hint'), f'category={category}: expected a non-empty hint, got: {result!r}'
        assert result.get('category') == category, (
            f'category={category}: expected the category echo on the block dict, got: {result!r}'
        )
        mock_service.search.assert_not_called()
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('category', ['procedural_knowledge', 'preferences_and_norms'])
    async def test_allow_near_duplicate_override_bypasses_topic_gate(self, category):
        """metadata={'allow_near_duplicate': True} bypasses the topic gate
        for either gated category — the override is evaluated before the
        category branch and works for any agent/category."""
        mock_service = AsyncMock()
        _configure_reconciliation(
            mock_service,
            procedural_knowledge_near_dup_guard_enabled=True,
            procedural_knowledge_near_dup_threshold=0.92,
            procedural_knowledge_topic_guard_clusters=[_topic_cluster()],
        )
        mock_service.search.return_value = []
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _TOPIC_MATCH_CONTENT,
                'category': category,
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
                'metadata': {'allow_near_duplicate': True},
            },
        )

        assert result.get('error_type') != 'ProceduralKnowledgeKnownTopicClusterWriteRejected', (
            f'category={category}: override must bypass the topic gate; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('category', ['procedural_knowledge', 'preferences_and_norms'])
    async def test_recon_stage_agent_exempt_from_topic_gate(self, category):
        """recon-stage-* agents are exempt from the topic gate for either
        gated category: Stage-1 consolidation writes the canonical merged
        entry, which by construction contains the cluster's phrases."""
        mock_service = AsyncMock()
        _configure_reconciliation(
            mock_service,
            procedural_knowledge_near_dup_guard_enabled=True,
            procedural_knowledge_near_dup_threshold=0.92,
            procedural_knowledge_topic_guard_clusters=[_topic_cluster()],
        )
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _TOPIC_MATCH_CONTENT,
                'category': category,
                'agent_id': 'recon-stage-memory_consolidator',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error_type') != 'ProceduralKnowledgeKnownTopicClusterWriteRejected', (
            f'category={category}: recon-stage agents must be exempt from the topic gate; '
            f'got: {result!r}'
        )
        mock_service.search.assert_not_called()
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_matching_content_falls_through_to_cosine_search(self):
        """Content matching NO cluster falls through to the existing cosine
        path. procedural_knowledge-specific: the cosine guard stays scoped to
        this category alone, so this behaviour has no preferences_and_norms
        counterpart — see test_non_matching_content_does_not_trigger_cosine_search.
        """
        mock_service = AsyncMock()
        _configure_reconciliation(
            mock_service,
            procedural_knowledge_near_dup_guard_enabled=True,
            procedural_knowledge_near_dup_threshold=0.92,
            procedural_knowledge_topic_guard_clusters=[_topic_cluster()],
        )
        mock_service.search.return_value = []
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': 'a totally unrelated note about coffee brewing temperatures',
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error_type') != 'ProceduralKnowledgeKnownTopicClusterWriteRejected', (
            f'Non-matching content must not hit the topic gate; got: {result!r}'
        )
        mock_service.search.assert_called_once()
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_matching_content_does_not_trigger_cosine_search(self):
        """The discriminating regression test for task 3430's restructure.

        A naive fix that merely widens the SHARED condition (swapping
        `category == 'procedural_knowledge'` for `category in
        _TOPIC_GUARD_GATED_CATEGORIES` on one fused block) would fall through
        into the procedural-only cosine search for a preferences_and_norms
        write too. A correct two-block split must not: the cosine block
        (Block B) stays gated on `category == 'procedural_knowledge'` alone,
        so a non-matching preferences_and_norms write is written straight
        through with no `search` call at all.
        """
        mock_service = AsyncMock()
        _configure_reconciliation(
            mock_service,
            procedural_knowledge_near_dup_guard_enabled=True,
            procedural_knowledge_near_dup_threshold=0.92,
            procedural_knowledge_topic_guard_clusters=[_topic_cluster()],
        )
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': 'a totally unrelated note about coffee brewing temperatures',
                'category': 'preferences_and_norms',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error_type') != 'ProceduralKnowledgeKnownTopicClusterWriteRejected', (
            f'Non-matching content must not hit the topic gate; got: {result!r}'
        )
        mock_service.search.assert_not_called()
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_observations_and_summaries_stays_inert(self):
        """The topic gate is inert for a category outside
        _TOPIC_GUARD_GATED_CATEGORIES. Task 3430 widened that set to also
        include preferences_and_norms; observations_and_summaries stays
        outside it deliberately — extending there is sibling task 4729's
        call, with its own false-positive analysis this task has not done.
        """
        mock_service = AsyncMock()
        _configure_reconciliation(
            mock_service,
            procedural_knowledge_near_dup_guard_enabled=True,
            procedural_knowledge_near_dup_threshold=0.92,
            procedural_knowledge_topic_guard_clusters=[_topic_cluster()],
        )
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _TOPIC_MATCH_CONTENT,
                'category': 'observations_and_summaries',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error_type') != 'ProceduralKnowledgeKnownTopicClusterWriteRejected', (
            f'Topic gate must be inert for a category outside _TOPIC_GUARD_GATED_CATEGORIES; '
            f'got: {result!r}'
        )
        mock_service.search.assert_not_called()
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'category, expect_cosine_fallthrough',
        [
            ('procedural_knowledge', True),
            ('preferences_and_norms', False),
        ],
    )
    async def test_empty_clusters_list_leaves_topic_gate_inert(
        self, category, expect_cosine_fallthrough
    ):
        """An empty clusters list disables the topic guard for both gated
        categories — but only procedural_knowledge has a cosine path to fall
        through to afterwards. preferences_and_norms has none, so `search`
        must never be called for it: the two categories cannot share a single
        `search` assertion here despite sharing everything else, which is why
        this case parametrizes the expectation rather than just the category.
        """
        mock_service = AsyncMock()
        _configure_reconciliation(
            mock_service,
            procedural_knowledge_near_dup_guard_enabled=True,
            procedural_knowledge_near_dup_threshold=0.92,
            procedural_knowledge_topic_guard_clusters=[],
        )
        mock_service.search.return_value = []
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _TOPIC_MATCH_CONTENT,
                'category': category,
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error_type') != 'ProceduralKnowledgeKnownTopicClusterWriteRejected', (
            f'category={category}: empty clusters list must leave the topic gate inert; '
            f'got: {result!r}'
        )
        if expect_cosine_fallthrough:
            mock_service.search.assert_called_once()
        else:
            mock_service.search.assert_not_called()
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('category', ['procedural_knowledge', 'preferences_and_norms'])
    async def test_master_kill_switch_disables_topic_gate(self, category):
        """procedural_knowledge_near_dup_guard_enabled=False disables BOTH
        guards, for either gated category."""
        mock_service = AsyncMock()
        _configure_reconciliation(
            mock_service,
            procedural_knowledge_near_dup_guard_enabled=False,
            procedural_knowledge_near_dup_threshold=0.92,
            procedural_knowledge_topic_guard_clusters=[_topic_cluster()],
        )
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _TOPIC_MATCH_CONTENT,
                'category': category,
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error_type') != 'ProceduralKnowledgeKnownTopicClusterWriteRejected', (
            f'category={category}: master kill-switch must disable the topic gate; got: {result!r}'
        )
        mock_service.search.assert_not_called()
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('category', ['preferences_and_norms', 'procedural_knowledge'])
    async def test_triage_supersedes_topic_gate_for_both_gated_categories(self, category):
        """Write triage (task 3127, PRD D2: redirect supersedes reject)
        retires the topic-cluster soft-block for BOTH categories the gate
        now covers — exactly as it already does for procedural_knowledge
        alone. Mirrors test_add_memory_write_triage_gate.py::
        test_a_topic_cluster_match_lands_rather_than_bouncing.

        Parametrized (rather than a manual `for` loop over both categories,
        reviewer: test-quality) so a regression affecting only one category
        is reported as its own failure instead of being masked by — or
        masking — the other's, and each iteration gets its own fresh
        mock/server rather than sharing state across loop iterations.
        """
        mock_service = AsyncMock()
        _configure_triage_and_reconciliation(
            mock_service,
            procedural_knowledge_near_dup_guard_enabled=True,
            procedural_knowledge_near_dup_threshold=0.92,
            procedural_knowledge_topic_guard_clusters=[_topic_cluster()],
        )
        mock_service.search.return_value = []
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _TOPIC_MATCH_CONTENT,
                'category': category,
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error_type') != 'ProceduralKnowledgeKnownTopicClusterWriteRejected', (
            f'category={category}: triage must supersede the topic gate, got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()


class TestAddMemorySufficientPhraseGate:
    """End-to-end: a sufficient-phrase block reaches a real add_memory call (task 3054).

    Unlike every class above, this wires the REAL shipped cluster seed
    (``ReconciliationConfig().procedural_knowledge_topic_guard_clusters``)
    rather than the synthetic ``_topic_cluster()``, so the clusters agents
    actually hit in production are exercised through the whole tool path.

    The content is the reconstructed straddling-write fixture IMPORTED from
    ``tests/test_config_schema.py`` -- one distinct phrase in each of three
    clusters, which under count-only matching was blocked by none. Importing
    rather than copying keeps this end-to-end assertion and the pure-matcher
    assertions pinned to the same scenario.
    """

    STRADDLING_CONTENT = STRADDLING_WRITE_FIXTURE

    @staticmethod
    def _configure_real_clusters(mock_service: AsyncMock) -> None:
        _configure_reconciliation(
            mock_service,
            procedural_knowledge_near_dup_guard_enabled=True,
            procedural_knowledge_near_dup_threshold=0.92,
            procedural_knowledge_topic_guard_clusters=(
                ReconciliationConfig().procedural_knowledge_topic_guard_clusters
            ),
        )

    @pytest.mark.asyncio
    async def test_straddling_write_is_blocked_before_the_cosine_search(self):
        """The headline fix, end to end: blocked, routed, and no embedding round-trip."""
        mock_service = AsyncMock()
        self._configure_real_clusters(mock_service)
        mock_service.search.return_value = []
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': self.STRADDLING_CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error_type') == 'ProceduralKnowledgeKnownTopicClusterWriteRejected', (
            f'Expected the straddling write to be blocked, got: {result!r}'
        )
        assert (
            result.get('topic_id') == 'architect-report-task-already-done-main-reachability'
        ), f'Expected routing to the report_task_already_done gate, got: {result!r}'
        # A SINGLE-element list, deliberately shorter than the cluster's
        # min_phrase_hits=2: a sufficient-phrase block reports exactly what
        # fired, which is what makes the routing unambiguous.
        assert result.get('matched_phrases') == ['report_task_already_done'], (
            f'Expected only the sufficient phrase reported, got: {result!r}'
        )
        assert result.get('hint'), f'Expected a non-empty hint, got: {result!r}'
        # The topic guard is deterministic and must short-circuit BEFORE the
        # cosine round-trip, exactly as the count-only path already does.
        mock_service.search.assert_not_called()
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_allow_near_duplicate_override_still_bypasses_a_sufficient_block(self):
        """The new match arm must not bypass the existing escape hatch."""
        mock_service = AsyncMock()
        self._configure_real_clusters(mock_service)
        mock_service.search.return_value = []
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': self.STRADDLING_CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
                'metadata': {'allow_near_duplicate': True},
            },
        )

        assert result.get('error_type') != 'ProceduralKnowledgeKnownTopicClusterWriteRejected', (
            f'Override must bypass a sufficient-phrase block; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_recon_stage_agent_still_exempt_from_a_sufficient_block(self):
        """Stage-1 consolidation writes the canonical entry, which contains the phrase."""
        mock_service = AsyncMock()
        self._configure_real_clusters(mock_service)
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': self.STRADDLING_CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'recon-stage-memory_consolidator',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error_type') != 'ProceduralKnowledgeKnownTopicClusterWriteRejected', (
            f'recon-stage agents must stay exempt; got: {result!r}'
        )
        mock_service.search.assert_not_called()
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_unrelated_note_still_falls_through_to_the_cosine_path(self):
        """Negative control against the REAL seed: sufficiency must not over-fire."""
        mock_service = AsyncMock()
        self._configure_real_clusters(mock_service)
        mock_service.search.return_value = []
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': MERGE_BASE_NEGATIVE_CONTROL_NOTE,
                'category': 'procedural_knowledge',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error_type') != 'ProceduralKnowledgeKnownTopicClusterWriteRejected', (
            f'A plain git-ancestry note must not be blocked; got: {result!r}'
        )
        mock_service.search.assert_called_once()
        mock_service.add_memory.assert_called_once()
