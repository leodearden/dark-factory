"""Tool-level tests for add_memory write triage (task 3127, PRD leaf beta).

The headline signal for the whole leaf: with ``write_triage.enabled`` on, a
restatement of an existing memory is no longer REJECTED with the submitted
content thrown away — it is REDIRECTED, attached as a SIGHTING child of the
memory it restates. Contract C1 is absolute: never lose content, never block a
write, never edit a canonical.

Harness mirrors ``test_add_memory_near_duplicate_gate.py``: an ``AsyncMock``
memory service wired through ``create_mcp_server``, invoked via
``server._tool_manager.call_tool('add_memory', {...})``. Two shapes that bite
if copied carelessly, both inherited from that suite:

* ``mock_service.config`` is a plain ``types.SimpleNamespace``. An unspecced
  ``AsyncMock``'s attribute chain auto-generates a truthy ``Mock`` for every
  hop, so ``config.write_triage.enabled`` would read as a Mock rather than a
  bool — which is exactly the shape the resolvers are built to REFUSE, and the
  test would silently exercise the flag-off path while claiming otherwise.
* ``mock_service.add_memory`` returns an explicit ``MagicMock`` with
  ``model_dump`` configured; otherwise ``result.model_dump()`` is an unawaited
  coroutine.

(The MCP-markup boundary guard task 4458 added is installed in
``server/main.py``, NOT in ``create_mcp_server``, so this harness does not run
through it and the test content needs no markup-proofing.)
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.grouped_read import PARENT_ID_KEY, SIGHTING_KIND
from fused_memory.server.tools import create_mcp_server
from fused_memory.server.write_triage import (
    CANONICAL_ID_KEY,
    OUTCOME_RESTATED,
    ROUTED_KEY,
)
from fused_memory.services.memory_service import RRF_K

_PROJECT_ID = 'dark_factory'

#: The write under test — a near-verbatim restatement of the canonical below.
_CONTENT = (
    'Never run git stash in a dark-factory checkout: refs/stash is a single '
    'ref in the shared .git dir and is not per-worktree.'
)
_CANONICAL_CONTENT = (
    'Do not use git stash in any dark-factory worktree — refs/stash is shared '
    'across every checkout, not per-worktree.'
)

# Calibrated bands for the harness. Chosen so the fixtures below sit
# unambiguously inside one band each, rather than near a boundary where a
# rounding change would silently reclassify them.
_T_HIGH = 0.90
_T_LOW = 0.70


def _candidate(
    id_: str = 'm1',
    score: float = 0.97,
    content: str = _CANONICAL_CONTENT,
    category: MemoryCategory = MemoryCategory.procedural_knowledge,
    store_rank: int = 1,
) -> MemoryResult:
    """Build the POST-RRF result shape the tool really receives from search().

    *score* is the Mem0 COSINE and lands in ``metadata['store_score']``;
    ``relevance_score`` carries the ordinal RRF value, deliberately unrelated
    to it, so a regression that bands on the RRF ordinal fails here rather
    than silently disabling triage for every input (task 3658).

    ``RRF_K`` comes from production rather than the literal 60, so a retune
    carries this fixture with it.
    """
    return MemoryResult(
        id=id_,
        content=content,
        category=category,
        source_store=SourceStore.mem0,
        relevance_score=1.0 / (RRF_K + store_rank),
        metadata={'store_rank': store_rank, 'store_score': score},
    )


def _configure_config(
    mock_service: AsyncMock,
    *,
    enabled: bool = True,
    candidate_k: int = 20,
    t_high: float | None = _T_HIGH,
    t_low: float | None = _T_LOW,
    near_dup_guard_enabled: bool = True,
    near_dup_threshold: float = 0.90,
    topic_clusters: list | None = None,
) -> None:
    """Stand in for ``memory_service.config`` with plain namespaces.

    Carries BOTH sections: ``write_triage`` for the new path and
    ``reconciliation`` for the two reject guards it supersedes, so a single
    harness can assert the flag-on and flag-off behaviours against identical
    config in every other respect.
    """
    mock_service.config = types.SimpleNamespace(
        write_triage=types.SimpleNamespace(
            enabled=enabled,
            candidate_k=candidate_k,
            t_high=t_high,
            t_low=t_low,
        ),
        reconciliation=types.SimpleNamespace(
            procedural_knowledge_near_dup_guard_enabled=near_dup_guard_enabled,
            procedural_knowledge_near_dup_threshold=near_dup_threshold,
            procedural_knowledge_topic_guard_clusters=topic_clusters or [],
        ),
    )


def _configure_pass_through_add_memory(
    mock_service: AsyncMock, dump: dict | None = None,
) -> MagicMock:
    """Configure ``mock_service.add_memory`` to return a dict-dumpable result."""
    mem_result = MagicMock()
    mem_result.model_dump.return_value = dump if dump is not None else {
        'id': 'new-id',
        'category': 'procedural_knowledge',
        'stored_in': ['mem0'],
    }
    mock_service.add_memory.return_value = mem_result
    return mem_result


async def _call(server, **overrides) -> dict:
    args = {
        'content': _CONTENT,
        'category': 'procedural_knowledge',
        'agent_id': 'claude-interactive',
        'project_id': _PROJECT_ID,
    }
    args.update(overrides)
    return await server._tool_manager.call_tool('add_memory', args)


class TestRestatementIsRedirectedNotRejected:
    """The headline: a restatement becomes a sighting child, and nothing is lost.

    This is the whole point of the leaf. The retired guard answered this exact
    write with a soft-block — the tool returned an error dict and the submitted
    text was gone unless the agent re-submitted with an override. Triage
    answers it by attaching, so the text survives and the rediscovery is
    counted (D9).
    """

    @pytest.mark.asyncio
    async def test_a_high_cosine_write_acks_restated_naming_its_canonical(self) -> None:
        mock_service = AsyncMock()
        _configure_config(mock_service)
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.return_value = [_candidate('m1', 0.97)]
        server = create_mcp_server(mock_service)

        result = await _call(server)

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result.get(ROUTED_KEY) == OUTCOME_RESTATED, (
            f'a cosine at/above t_high={_T_HIGH} is the DETERMINISTIC band: {result!r}'
        )
        assert result.get(CANONICAL_ID_KEY) == 'm1', (
            f'the ack must name what the write was attached to: {result!r}'
        )

    @pytest.mark.asyncio
    async def test_no_standalone_entry_is_created_the_write_becomes_a_child(
        self,
    ) -> None:
        """ONE write, and it carries the parent link — not two entries, not zero.

        A standalone store alongside the attach would recreate the duplicate
        triage exists to prevent; no write at all would be the content loss C1
        forbids.
        """
        mock_service = AsyncMock()
        _configure_config(mock_service)
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.return_value = [_candidate('m1', 0.97)]
        server = create_mcp_server(mock_service)

        await _call(server)

        assert mock_service.add_memory.await_count == 1, (
            f'expected exactly one write, got {mock_service.add_memory.await_args_list!r}'
        )
        metadata = mock_service.add_memory.await_args.kwargs['metadata']
        assert metadata is not None, 'the child must carry metadata naming its parent'
        assert metadata[PARENT_ID_KEY] == 'm1', (
            f'grouping is strictly metadata.parent_id + child kind: {metadata!r}'
        )
        assert metadata['kind'] == SIGHTING_KIND, (
            f'a restatement is a SIGHTING, not an amendment: {metadata!r}'
        )

    @pytest.mark.asyncio
    async def test_the_full_submitted_content_is_preserved_on_the_child(self) -> None:
        """C1 may never lose content — not a digest, not an excerpt, the text.

        The retired guard's rejection block carried only a `content_excerpt`
        (200 chars); the write itself was discarded. An attach that truncated
        would be the same defect wearing a different outcome name.
        """
        mock_service = AsyncMock()
        _configure_config(mock_service)
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.return_value = [_candidate('m1', 0.97)]
        server = create_mcp_server(mock_service)

        await _call(server)

        assert mock_service.add_memory.await_args.kwargs['content'] == _CONTENT

    @pytest.mark.asyncio
    async def test_the_child_keeps_the_writes_own_category_and_agent(self) -> None:
        """An attach reroutes the write, it does not rewrite it."""
        mock_service = AsyncMock()
        _configure_config(mock_service)
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.return_value = [_candidate('m1', 0.97)]
        server = create_mcp_server(mock_service)

        await _call(server)

        kwargs = mock_service.add_memory.await_args.kwargs
        assert kwargs['category'] == 'procedural_knowledge'
        assert kwargs['agent_id'] == 'claude-interactive'
        assert kwargs['project_id'] == _PROJECT_ID

    @pytest.mark.asyncio
    async def test_the_writers_own_metadata_survives_the_attach(self) -> None:
        """The parent link is ADDED to the submitted metadata, not swapped for it."""
        mock_service = AsyncMock()
        _configure_config(mock_service)
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.return_value = [_candidate('m1', 0.97)]
        server = create_mcp_server(mock_service)

        await _call(server, metadata={'source': 'session-notes'})

        metadata = mock_service.add_memory.await_args.kwargs['metadata']
        assert metadata['source'] == 'session-notes', (
            f'a caller key was dropped by the attach: {metadata!r}'
        )
        assert metadata[PARENT_ID_KEY] == 'm1'

    @pytest.mark.asyncio
    async def test_the_ack_is_purely_additive_over_the_add_memory_response(
        self,
    ) -> None:
        """`routed`/`canonical_id` are ADDED to the normal ack, never replace it.

        Every existing caller reads the AddMemoryResponse fields; an ack that
        swapped them for a triage verdict would break all of them at once, and
        would be a far larger change than this leaf is entitled to make.
        """
        mock_service = AsyncMock()
        _configure_config(mock_service)
        _configure_pass_through_add_memory(mock_service, dump={
            'id': 'child-id',
            'category': 'procedural_knowledge',
            'stored_in': ['mem0'],
        })
        mock_service.search.return_value = [_candidate('m1', 0.97)]
        server = create_mcp_server(mock_service)

        result = await _call(server)

        assert result['id'] == 'child-id'
        assert result['category'] == 'procedural_knowledge'
        assert result['stored_in'] == ['mem0']
        assert result[ROUTED_KEY] == OUTCOME_RESTATED
        assert result[CANONICAL_ID_KEY] == 'm1'

    @pytest.mark.asyncio
    async def test_the_write_is_never_rejected_on_the_restate_path(self) -> None:
        """No error keys at all — a redirect is a SUCCESS, not a soft failure."""
        mock_service = AsyncMock()
        _configure_config(mock_service)
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.return_value = [_candidate('m1', 0.97)]
        server = create_mcp_server(mock_service)

        result = await _call(server)

        assert 'error' not in result, f'a redirect must not read as an error: {result!r}'
        assert 'error_type' not in result, f'{result!r}'

    @pytest.mark.asyncio
    async def test_the_canonical_is_never_mutated_by_an_attach(self) -> None:
        """C1: triage issues no update_memory and no delete_memory, ever.

        The canonical's text is not the write's to edit, and never touching it
        is what keeps a WRONG attach cheap — re-parenting a child is a metadata
        edit, whereas an overwritten canonical is unrecoverable (D4).
        """
        mock_service = AsyncMock()
        _configure_config(mock_service)
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.return_value = [_candidate('m1', 0.97)]
        server = create_mcp_server(mock_service)

        await _call(server)

        mock_service.update_memory.assert_not_awaited()
        mock_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_winner_is_the_max_cosine_across_the_candidates(self) -> None:
        """Retrieval returns a ranked list; the attach target is the best COSINE.

        The rank-1 hit here carries the LOWEST cosine, so a regression that
        attaches to `results[0]` — or to the max `relevance_score`, which is
        the same thing post-RRF — picks 'm-low' and fails.
        """
        mock_service = AsyncMock()
        _configure_config(mock_service)
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.return_value = [
            _candidate('m-low', 0.72, store_rank=1),
            _candidate('m-best', 0.98, store_rank=2),
        ]
        server = create_mcp_server(mock_service)

        result = await _call(server)

        assert result[CANONICAL_ID_KEY] == 'm-best', (
            f'attached to the wrong candidate: {result!r}'
        )

    @pytest.mark.asyncio
    async def test_a_cross_category_duplicate_is_still_caught(self) -> None:
        """The blind spot this leaf exists to fix.

        The retired guard filtered candidates to the write's OWN category
        (`near_duplicate_guard.py:117`), so a procedural_knowledge write
        restating an observations_and_summaries entry was invisible to it —
        measured on reify esc-5547 and esc-5560, both of which had
        cross-category duplicates. Triage retrieves across all three
        Mem0-primary categories, so the attach still happens.
        """
        mock_service = AsyncMock()
        _configure_config(mock_service)
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.return_value = [
            _candidate('m-other', 0.97, category=MemoryCategory.observations_and_summaries),
        ]
        server = create_mcp_server(mock_service)

        result = await _call(server)

        assert result[ROUTED_KEY] == OUTCOME_RESTATED
        assert result[CANONICAL_ID_KEY] == 'm-other'
