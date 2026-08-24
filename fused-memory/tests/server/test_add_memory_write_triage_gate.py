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

from fused_memory.config.schema import ProceduralTopicCluster
from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server import tools
from fused_memory.server.grouped_read import PARENT_ID_KEY, SIGHTING_KIND
from fused_memory.server.tools import create_mcp_server
from fused_memory.server.write_triage import (
    CANONICAL_ID_KEY,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    ROUTED_KEY,
    TRIAGE_OUTCOMES,
    TriageFailOpenCounter,
)
from fused_memory.services.memory_service import RRF_K, SearchResults

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
    extra_metadata: dict | None = None,
) -> MemoryResult:
    """Build the POST-RRF result shape the tool really receives from search().

    *score* is the Mem0 COSINE and lands in ``metadata['store_score']``;
    ``relevance_score`` carries the ordinal RRF value, deliberately unrelated
    to it, so a regression that bands on the RRF ordinal fails here rather
    than silently disabling triage for every input (task 3658).

    ``RRF_K`` comes from production rather than the literal 60, so a retune
    carries this fixture with it.

    ``extra_metadata`` is merged in last and is how a CHILD candidate's shape
    (``{'kind': ..., 'parent_id': ...}``) is built — children arrive from
    ``search`` as ordinary hits, because grouping is applied at the MCP
    boundary and never inside the service.
    """
    return MemoryResult(
        id=id_,
        content=content,
        category=category,
        source_store=SourceStore.mem0,
        relevance_score=1.0 / (RRF_K + store_rank),
        metadata={
            'store_rank': store_rank,
            'store_score': score,
            **(extra_metadata or {}),
        },
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
    async def test_attaching_to_a_child_winner_lands_under_the_grandparent(
        self,
    ) -> None:
        """No grandchildren: a child winner hoists to its canonical.

        A sighting child stores the restatement text VERBATIM, so on the
        SECOND restatement of the same fact it is the likeliest max-cosine
        winner — this is the common case, not a corner one. Attaching to it
        would write `{parent_id: <the child>}`, and `grouped_read` resolves
        exactly ONE level, so that grandchild would never fold under the true
        canonical: a child that exists but never groups, which reads as
        content loss without being one.

        Asserted end-to-end rather than only at `decide_band`, because the
        persisted `parent_id` and the ack are what an operator and the
        grouped read actually see.
        """
        mock_service = AsyncMock()
        _configure_config(mock_service)
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.return_value = [
            _candidate('m1', 0.93, store_rank=2),
            _candidate(
                'm2', 0.97,
                content=_CONTENT,
                extra_metadata={PARENT_ID_KEY: 'm1', 'kind': SIGHTING_KIND},
            ),
        ]
        server = create_mcp_server(mock_service)

        result = await _call(server)

        assert mock_service.add_memory.await_count == 1
        metadata = mock_service.add_memory.await_args.kwargs['metadata']
        assert metadata[PARENT_ID_KEY] == 'm1', (
            f'must attach to the GRANDPARENT, not the winning child m2: {metadata!r}'
        )
        assert result.get(CANONICAL_ID_KEY) == 'm1', (
            f'the ack must name the canonical, not the child: {result!r}'
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


#: Content matching the injected test cluster's phrases below.
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


class TestTheFlagOffPathIsUntouched:
    """The shipped default (D10 staged rollout): triage is OFF.

    Everything below is the behaviour that exists TODAY, re-asserted from the
    triage suite so a regression in the new branch is caught by the suite that
    introduced it rather than only by the guard's own file. The whole leaf is
    inert until task 3169 flips the flag, and "inert" has to mean byte-identical
    — including the ack, which must not grow a `routed` key nobody asked for.
    """

    @pytest.mark.asyncio
    async def test_a_near_duplicate_write_is_still_rejected(self) -> None:
        mock_service = AsyncMock()
        _configure_config(mock_service, enabled=False)
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.return_value = [_candidate('m1', 0.97)]
        server = create_mcp_server(mock_service)

        result = await _call(server)

        assert result.get('error_type') == 'ProceduralKnowledgeNearDuplicateWriteRejected', (
            f'the retired guard must still be live while the flag is off: {result!r}'
        )
        mock_service.add_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_topic_cluster_match_is_still_rejected(self) -> None:
        mock_service = AsyncMock()
        _configure_config(
            mock_service, enabled=False, topic_clusters=[_topic_cluster()],
        )
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.return_value = []
        server = create_mcp_server(mock_service)

        result = await _call(server, content=_TOPIC_MATCH_CONTENT)

        assert result.get('error_type') == (
            'ProceduralKnowledgeKnownTopicClusterWriteRejected'
        ), f'the topic guard must still be live while the flag is off: {result!r}'
        mock_service.add_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_clean_write_acks_with_no_routed_key_at_all(self) -> None:
        """Not `routed: None`, not `routed: 'stored'` — ABSENT.

        A key that appears whenever the code is deployed, regardless of the
        flag, would tell every caller that triage is live when it is not, and
        would make the rollout unobservable from the outside.
        """
        mock_service = AsyncMock()
        _configure_config(mock_service, enabled=False)
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.return_value = []
        server = create_mcp_server(mock_service)

        result = await _call(server)

        assert ROUTED_KEY not in result, f'the ack leaked a triage key: {result!r}'
        assert CANONICAL_ID_KEY not in result, f'{result!r}'
        mock_service.add_memory.assert_awaited_once()


class TestTheFlagOnPathRetiresBothRejectGuards:
    """D2: redirect SUPERSEDES reject. Not "runs after", not "runs alongside".

    With triage on, neither reject error_type is reachable for a triaged write.
    A write that the retired guards would have bounced now LANDS — as a child
    when it restates something, as a plain entry otherwise. That is the whole
    behavioural claim of the leaf, so it is asserted from the guards' own
    inputs rather than from a fresh fixture that might simply miss them.
    """

    @pytest.mark.asyncio
    async def test_the_near_duplicate_reject_is_unreachable(self) -> None:
        mock_service = AsyncMock()
        _configure_config(mock_service, enabled=True)
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.return_value = [_candidate('m1', 0.97)]
        server = create_mcp_server(mock_service)

        result = await _call(server)

        assert result.get('error_type') != 'ProceduralKnowledgeNearDuplicateWriteRejected', (
            f'the reject guard must be retired for a triaged write: {result!r}'
        )
        assert result[ROUTED_KEY] == OUTCOME_RESTATED
        mock_service.add_memory.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_topic_cluster_match_lands_rather_than_bouncing(self) -> None:
        """The topic soft-block retires with the cosine one.

        Below `t_high` a topic hit is a JUDGE question, not a rejection (C1) —
        and while the judge is a deliberate stub answering `stored`, the
        observable contract at this boundary is that the write LANDS. What the
        judge eventually does with the topic signal is leaf gamma's; what this
        pins is that the agent's text is never thrown away for matching a
        known-contradictory topic.
        """
        mock_service = AsyncMock()
        _configure_config(
            mock_service, enabled=True, topic_clusters=[_topic_cluster()],
        )
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.return_value = []
        server = create_mcp_server(mock_service)

        result = await _call(server, content=_TOPIC_MATCH_CONTENT)

        assert result.get('error_type') != (
            'ProceduralKnowledgeKnownTopicClusterWriteRejected'
        ), f'the topic guard must be retired for a triaged write: {result!r}'
        assert 'error' not in result, f'{result!r}'
        assert result[ROUTED_KEY] == OUTCOME_STORED
        assert mock_service.add_memory.await_args.kwargs['content'] == _TOPIC_MATCH_CONTENT

    @pytest.mark.asyncio
    async def test_no_reject_error_type_is_reachable_for_any_triaged_input(
        self,
    ) -> None:
        """Swept across the whole band, including exactly at each threshold."""
        for score in (0.0, 0.69, _T_LOW, 0.80, _T_HIGH, 0.97, 1.0):
            mock_service = AsyncMock()
            _configure_config(
                mock_service, enabled=True, topic_clusters=[_topic_cluster()],
            )
            _configure_pass_through_add_memory(mock_service)
            mock_service.search.return_value = [_candidate('m1', score)]
            server = create_mcp_server(mock_service)

            result = await _call(server, content=_TOPIC_MATCH_CONTENT)

            assert 'error_type' not in result, f'score={score}: {result!r}'
            assert result[ROUTED_KEY] in TRIAGE_OUTCOMES, f'score={score}: {result!r}'


class TestTheForceStoreArms:
    """Two inputs that skip triage entirely, even with the flag on."""

    @pytest.mark.asyncio
    async def test_allow_near_duplicate_forces_a_plain_store(self) -> None:
        """D2 reinterprets the old bypass flag as the force-store escape hatch.

        Under the retired guard it meant "do not reject me"; under triage it
        means "do not reroute me". Same intent — the writer has asserted the
        content is genuinely distinct — expressed against the mechanism that
        replaced the one it was built for.
        """
        mock_service = AsyncMock()
        _configure_config(mock_service, enabled=True)
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.return_value = [_candidate('m1', 0.99)]
        server = create_mcp_server(mock_service)

        result = await _call(server, metadata={'allow_near_duplicate': True})

        assert result[ROUTED_KEY] == OUTCOME_STORED, f'{result!r}'
        assert CANONICAL_ID_KEY not in result, f'nothing was attached: {result!r}'
        metadata = mock_service.add_memory.await_args.kwargs['metadata']
        assert PARENT_ID_KEY not in (metadata or {}), f'{metadata!r}'

    @pytest.mark.asyncio
    async def test_allow_near_duplicate_skips_the_retrieval_round_trip(self) -> None:
        """No search at all — not a search whose result is then discarded.

        Retrieval is an embedding + vector round-trip on every triaged write.
        A writer who has already declared the content distinct should not pay
        for a lookup whose answer cannot change the outcome.
        """
        mock_service = AsyncMock()
        _configure_config(mock_service, enabled=True)
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        await _call(server, metadata={'allow_near_duplicate': True})

        mock_service.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_bypass_flag_is_still_stripped_from_persistence(self) -> None:
        """A write-time control flag must never reach the corpus.

        The same discipline task 4458 pinned for `allow_mcp_markup`, whose
        surviving `strip_markup_override` call sits in this same tool body: a
        flag that steers a write is not a fact ABOUT the write.
        """
        mock_service = AsyncMock()
        _configure_config(mock_service, enabled=True)
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        await _call(
            server, metadata={'allow_near_duplicate': True, 'source': 'notes'},
        )

        metadata = mock_service.add_memory.await_args.kwargs['metadata']
        assert 'allow_near_duplicate' not in metadata, f'flag persisted: {metadata!r}'
        assert metadata['source'] == 'notes', f'{metadata!r}'

    @pytest.mark.asyncio
    async def test_a_recon_stage_agent_is_force_stored_and_never_attached(self) -> None:
        """The recon-stage exemption SURVIVES this leaf. Leaf iota retires it.

        Stage-1 consolidation writes a merged canonical that is EXPECTED to
        closely resemble the duplicates it replaces, with no ordering guarantee
        that those duplicates are deleted first. Attaching it as a sighting of
        one of them would invert consolidation — the merged entry would become
        a child of the very memory it was written to supersede.

        Removing this is leaf iota's explicit signal ("a recon-agent direct
        near-dup add_memory now triages like anyone else"), so it must still
        hold here; iota is the owner of its removal, not this leaf.
        """
        mock_service = AsyncMock()
        _configure_config(mock_service, enabled=True)
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.return_value = [_candidate('m1', 0.99)]
        server = create_mcp_server(mock_service)

        result = await _call(server, agent_id='recon-stage-1')

        assert result[ROUTED_KEY] == OUTCOME_STORED, f'{result!r}'
        assert CANONICAL_ID_KEY not in result, f'consolidation was inverted: {result!r}'
        metadata = mock_service.add_memory.await_args.kwargs['metadata']
        assert PARENT_ID_KEY not in (metadata or {}), f'{metadata!r}'
        mock_service.search.assert_not_awaited()


class TestC1HoldsEndToEnd:
    """Contract C1 at the tool boundary, on every path a dependency can break.

    Never lose content, never block a write, never edit a canonical. The unit
    suite pins these inside `write_triage`; what these pin is that the WIRING
    honours them too — a fail-open that `triage_write` handled perfectly is
    still a blocked write if the tool body then raises on the attach.
    """

    @staticmethod
    def _install_counter(monkeypatch) -> TriageFailOpenCounter:
        """Bind a readable counter into the server the next call builds.

        `create_mcp_server` constructs its counter closure-locally (so nothing
        bleeds between servers), which also means a test cannot reach it. The
        zero-arg construction is the seam: substituting the class with a
        factory returning OUR instance leaves the production wiring identical
        while making the count observable.
        """
        counter = TriageFailOpenCounter(time_provider=lambda: 1000.0)
        monkeypatch.setattr(tools, 'TriageFailOpenCounter', lambda: counter)
        return counter

    @pytest.mark.asyncio
    async def test_a_raising_search_still_stores_the_write(self, monkeypatch) -> None:
        """A retrieval outage degrades triage; it must not touch the write.

        The retired guard's call site RE-RAISED TypeError/AttributeError/
        NameError so a wiring bug surfaced loudly. Triage cannot: re-raising
        here is an errored write, i.e. a blocked write. The loudness is
        preserved as an ERROR log plus a counted fail-open instead.
        """
        counter = self._install_counter(monkeypatch)
        mock_service = AsyncMock()
        _configure_config(mock_service, enabled=True)
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.side_effect = RuntimeError('mem0 unreachable')
        server = create_mcp_server(mock_service)

        result = await _call(server)

        assert result[ROUTED_KEY] == OUTCOME_STORED, f'{result!r}'
        assert 'error' not in result, f'a fail-open must not error the write: {result!r}'
        assert 'error_type' not in result, f'{result!r}'
        mock_service.add_memory.assert_awaited_once()
        assert mock_service.add_memory.await_args.kwargs['content'] == _CONTENT
        assert counter.live_count() == 1, 'the degradation must be counted (INV-4)'

    @pytest.mark.asyncio
    async def test_a_degraded_search_still_stores_the_write_and_counts(
        self, monkeypatch,
    ) -> None:
        """The outage shape, end to end — and it does NOT raise.

        The sibling test above stubs `search` with `side_effect=RuntimeError`,
        which is NOT how a mem0 outage reaches this path: `MemoryService.search`
        catches the store exception (and cancels the store on timeout), logs
        `search.store_failed`, and returns an EMPTY `SearchResults` with
        `degraded=True`. Stubbing the REAL return shape is the whole point of
        this test — with a raising mock the suite passes while an actual
        outage stores every write untriaged, uncounted and unescalated.
        """
        counter = self._install_counter(monkeypatch)
        mock_service = AsyncMock()
        _configure_config(mock_service, enabled=True)
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.return_value = SearchResults(
            [], degraded=True, failed_stores=['mem0'],
        )
        server = create_mcp_server(mock_service)

        result = await _call(server)

        assert result[ROUTED_KEY] == OUTCOME_STORED, f'{result!r}'
        assert 'error' not in result, f'a fail-open must not error the write: {result!r}'
        assert 'error_type' not in result, f'{result!r}'
        mock_service.add_memory.assert_awaited_once()
        assert mock_service.add_memory.await_args.kwargs['content'] == _CONTENT
        assert counter.live_count() == 1, 'the outage must be counted (INV-4)'

    @pytest.mark.asyncio
    async def test_a_wiring_bug_class_also_stores_rather_than_erroring(
        self, monkeypatch,
    ) -> None:
        """A changed MemoryService.search signature is the concrete case."""
        counter = self._install_counter(monkeypatch)
        mock_service = AsyncMock()
        _configure_config(mock_service, enabled=True)
        _configure_pass_through_add_memory(mock_service)
        mock_service.search.side_effect = TypeError(
            "search() got an unexpected keyword argument 'categories'"
        )
        server = create_mcp_server(mock_service)

        result = await _call(server)

        assert result[ROUTED_KEY] == OUTCOME_STORED, f'{result!r}'
        assert 'error_type' not in result, f'{result!r}'
        assert counter.live_count() == 1

    @pytest.mark.asyncio
    async def test_an_attach_failure_falls_back_to_a_standalone_store(
        self, monkeypatch,
    ) -> None:
        """The sharpest C1 case: the redirect fails, so the write must NOT.

        If the child write raises and nothing catches it, triage has converted
        a write that would have succeeded before the leaf into a hard failure —
        content loss caused by the very mechanism built to prevent it. The
        fallback stores the same FULL content standalone, which is exactly the
        pre-triage outcome.
        """
        counter = self._install_counter(monkeypatch)
        mock_service = AsyncMock()
        _configure_config(mock_service, enabled=True)
        mem_result = MagicMock()
        mem_result.model_dump.return_value = {'id': 'fallback-id'}
        mock_service.add_memory.side_effect = [
            RuntimeError('parent_id rejected by the write seam'),
            mem_result,
        ]
        mock_service.search.return_value = [_candidate('m1', 0.97)]
        server = create_mcp_server(mock_service)

        result = await _call(server)

        assert 'error' not in result, f'the write was blocked: {result!r}'
        assert 'error_type' not in result, f'{result!r}'
        assert result[ROUTED_KEY] == OUTCOME_STORED, (
            f'the attach did not happen, so the ack must not claim it did: {result!r}'
        )
        assert CANONICAL_ID_KEY not in result, (
            f'nothing was attached, so nothing may be named: {result!r}'
        )
        assert result['id'] == 'fallback-id'
        assert mock_service.add_memory.await_count == 2, (
            f'expected attach then fallback: {mock_service.add_memory.await_args_list!r}'
        )
        fallback = mock_service.add_memory.await_args.kwargs
        assert fallback['content'] == _CONTENT, 'the fallback must carry the FULL text'
        assert PARENT_ID_KEY not in (fallback['metadata'] or {}), (
            f'the failed parent link leaked into the fallback: {fallback["metadata"]!r}'
        )
        assert counter.live_count() == 1, 'a failed attach is a fail-open (INV-4)'

    @pytest.mark.asyncio
    async def test_the_canonical_is_never_mutated_on_any_path(
        self, monkeypatch,
    ) -> None:
        """Swept across restate, store, judge-stub and every fail-open.

        Triage issues no update_memory and no delete_memory, ever. A canonical
        the write path can rewrite is a canonical a mis-scored write can
        destroy; leaving it read-only is what makes a wrong attach a cheap,
        reversible metadata edit (D4).
        """
        self._install_counter(monkeypatch)
        scenarios = [
            ('restate', {'search': AsyncMock(return_value=[_candidate('m1', 0.97)])}),
            ('judge band', {'search': AsyncMock(return_value=[_candidate('m1', 0.80)])}),
            ('store', {'search': AsyncMock(return_value=[_candidate('m1', 0.10)])}),
            ('no candidates', {'search': AsyncMock(return_value=[])}),
            ('fail-open', {'search': AsyncMock(side_effect=RuntimeError('down'))}),
        ]
        for label, wiring in scenarios:
            mock_service = AsyncMock()
            _configure_config(mock_service, enabled=True)
            _configure_pass_through_add_memory(mock_service)
            mock_service.search = wiring['search']
            server = create_mcp_server(mock_service)

            await _call(server)

            mock_service.update_memory.assert_not_awaited()
            mock_service.delete_memory.assert_not_awaited()
            assert mock_service.add_memory.await_count == 1, label

    @pytest.mark.asyncio
    async def test_the_ack_contract_holds_across_the_whole_band(
        self, monkeypatch,
    ) -> None:
        """`routed` is always a published outcome; `canonical_id` iff attached.

        ABSENT rather than None for a non-attach: an omitted key is an
        unambiguous signal, whereas a null is a value the reader then has to
        disambiguate. This is the "omit rather than emit a null" convention now
        documented in shared/mcp_markup_middleware.py (it used to live on
        `build_markup_block`, which task 4458 deleted).
        """
        self._install_counter(monkeypatch)
        for score in (0.0, 0.10, 0.69, _T_LOW, 0.80, _T_HIGH, 0.97, 1.0):
            mock_service = AsyncMock()
            _configure_config(mock_service, enabled=True)
            _configure_pass_through_add_memory(mock_service)
            mock_service.search.return_value = [_candidate('m1', score)]
            server = create_mcp_server(mock_service)

            result = await _call(server)

            routed = result[ROUTED_KEY]
            assert routed in TRIAGE_OUTCOMES, f'score={score}: {result!r}'
            attached = routed != OUTCOME_STORED
            assert (CANONICAL_ID_KEY in result) is attached, (
                f'score={score}: canonical_id must be present iff attached, and '
                f'ABSENT (not None) otherwise: {result!r}'
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('label', 'category'),
        [
            ('a Graphiti-primary category', 'temporal_facts'),
            ('an auto-classify write', None),
        ],
    )
    async def test_untriaged_scopes_are_not_triaged_at_all(
        self, monkeypatch, label, category,
    ) -> None:
        """Neither a search nor a `routed` key — the write bypasses triage.

        `category=None` auto-classifies inside MemoryService.add_memory, BELOW
        this seam, so triaging it here would mean running the classifier a
        second time (INV-5) — and the second copy would be the one that drifts.
        A Graphiti-primary category is out of scope for a leaf whose retrieval
        is a mem0 vector search.
        """
        self._install_counter(monkeypatch)
        mock_service = AsyncMock()
        _configure_config(mock_service, enabled=True)
        _configure_pass_through_add_memory(mock_service)
        server = create_mcp_server(mock_service)

        result = await _call(server, category=category)

        mock_service.search.assert_not_awaited()
        assert ROUTED_KEY not in result, f'{label} was triaged: {result!r}'
        assert CANONICAL_ID_KEY not in result, f'{label}: {result!r}'
        mock_service.add_memory.assert_awaited_once()
