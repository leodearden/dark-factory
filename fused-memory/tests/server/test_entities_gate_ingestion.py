"""The `entities` declaration at the WRITE-TOOL boundary (task 3669, leaf delta).

`tests/server/test_entities_gate.py` pins the pure gate; this pins the WIRING —
that the gate is actually in `add_memory`'s (and, from step-11, `add_episode`'s)
chain, that a rejection means the write DOES NOT HAPPEN, and that an accepted
declaration reaches the service.

"The write must not occur" is half the PRD's signal and the half a
return-value assertion cannot see, so every rejection case here pairs the
error-dict assertion with `assert_not_called()`.

Harness copied from `test_add_memory_snapshot_gate.py` / its siblings: an
`AsyncMock` memory service wired through `create_mcp_server`, invoked via
`server._tool_manager.call_tool(...)`. Two inherited shapes that bite if copied
carelessly — `mock_service.add_memory` must return an explicit `MagicMock` with
`model_dump` configured (otherwise `result.model_dump()` is an unawaited
coroutine), and an unspecced `AsyncMock`'s attribute chain auto-generates a
truthy `Mock` for every config hop.

(The MCP-markup boundary guard task 4458 added is installed in `server/main.py`,
NOT in `create_mcp_server`, so this harness does not run through it and the
content below needs no markup-proofing.)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.server.tools import create_mcp_server
from fused_memory.utils.referent_resolution import _DECLARED_REFERENT_HINT

# The C1 fallback harness, imported rather than re-derived: reaching
# `add_memory`'s SECOND service call site needs write triage enabled with a
# calibrated high-cosine candidate, and a private copy would drift the moment
# those bands are retuned. The bare `server.` import resolves because
# fused-memory/tests/conftest.py inserts the tests dir onto sys.path and
# tests/server/ is a package (see the same note in
# test_add_memory_near_duplicate_gate.py, which imports test_config_schema the
# same way).
from server.test_add_memory_write_triage_gate import _candidate, _configure_config

# The malformed-shape corpus, IMPORTED rather than re-spelled. The unit tests
# pin what the pure gate answers for each shape; this module pins what an agent
# actually receives for the same shape. Two copies of the list would let those
# two claims drift — a shape added to one and not the other is silently
# unchecked at the seam that matters — which is the whole point of the
# single-list amendment (see `_MALFORMED_SHAPES`' own comment).
from server.test_entities_gate import _MALFORMED_SHAPES

_PROJECT_ID = 'dark_factory'

#: Prose naming exactly one own-project referent, Task 3127.
_CONTENT = 'the fix for Task 3127 landed'

#: The PRD's headline declaration: an adjacent-number typo the prose refutes.
_CONFLICTING = [{'kind': 'task', 'id': 3129}]
#: The same declaration, corrected to match the prose.
_CORROBORATING = [{'kind': 'task', 'id': 3127}]
#: Structurally invalid — gamma raises InputValidationError on a non-digit id.
_MALFORMED = [{'id': 'abc'}]


def _pass_through(mock_service: AsyncMock, dump: dict | None = None) -> MagicMock:
    """Configure ``mock_service.add_memory`` to return a dict-dumpable result."""
    mem_result = MagicMock()
    mem_result.model_dump.return_value = dump if dump is not None else {'id': 'ok'}
    mock_service.add_memory.return_value = mem_result
    return mem_result


async def _call(server, **overrides) -> dict:
    args = {
        'content': _CONTENT,
        'category': 'decisions_and_rationale',
        'agent_id': 'claude-interactive',
        'project_id': _PROJECT_ID,
    }
    args.update(overrides)
    return await server._tool_manager.call_tool('add_memory', args)


class TestAddMemoryEntitiesGate:
    """The gate is wired into `add_memory`'s chain, and a rejection blocks."""

    @pytest.mark.asyncio
    async def test_a_conflicting_declaration_rejects_and_does_not_write(self):
        """The PRD's user-observable signal, end to end at the boundary."""
        mock_service = AsyncMock()
        _pass_through(mock_service)
        server = create_mcp_server(mock_service)

        result = await _call(server, entities=_CONFLICTING)

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result.get('error_type') == 'DeclaredReferentConflictRejected', f'{result!r}'
        assert result.get('error') == 'declared_referent_conflict', f'{result!r}'
        assert result.get('agent_id') == 'claude-interactive', f'{result!r}'
        assert result.get('conflicts') == ['Task 3129'], f'{result!r}'
        assert result.get('declared') == ['Task 3129'], f'{result!r}'
        assert result.get('content_referents') == ['Task 3127'], f'{result!r}'
        assert result.get('hint'), f'{result!r}'
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_same_call_without_entities_writes(self):
        """The control. Absence is never a rejection, so the ONLY difference
        between this and the case above is the declaration."""
        mock_service = AsyncMock()
        _pass_through(mock_service)
        server = create_mcp_server(mock_service)

        result = await _call(server)

        assert 'error' not in result, f'an undeclared write was blocked: {result!r}'
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_corroborating_declaration_writes(self):
        mock_service = AsyncMock()
        _pass_through(mock_service)
        server = create_mcp_server(mock_service)

        result = await _call(server, entities=_CORROBORATING)

        assert 'error' not in result, f'{result!r}'
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_the_empty_declaration_writes(self):
        """Tri-state arm 2 — "considered, none apply" — is not a rejection,
        even though the scan DID find a referent the declaration omits."""
        mock_service = AsyncMock()
        _pass_through(mock_service)
        server = create_mcp_server(mock_service)

        result = await _call(server, entities=[])

        assert 'error' not in result, f'{result!r}'
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_malformed_declaration_rejects_and_does_not_write(self):
        mock_service = AsyncMock()
        _pass_through(mock_service)
        server = create_mcp_server(mock_service)

        result = await _call(server, entities=_MALFORMED)

        assert result.get('error_type') == 'ValidationError', f'{result!r}'
        assert result.get('content_excerpt') == _CONTENT[:200], f'{result!r}'
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_gate_is_category_independent(self):
        """A Mem0-primary category that never routes to Graphiti STILL rejects.

        Scoping the gate to `write_graphiti` is the tempting optimization and
        it is unimplementable at this seam: `category=None` auto-classifies
        BELOW the tool boundary, so the tool structurally cannot know the
        destination store without running the classifier a second time (INV-5).
        It is also wrong on the merits — a declaration contradicting its own
        prose is a defective assertion whichever store it lands in, and
        accepting it for Mem0 writes would teach agents that `entities` is
        advisory.
        """
        for category in ('observations_and_summaries', 'preferences_and_norms', None):
            mock_service = AsyncMock()
            _pass_through(mock_service)
            server = create_mcp_server(mock_service)

            result = await _call(server, category=category, entities=_CONFLICTING)

            assert result.get('error_type') == 'DeclaredReferentConflictRejected', (
                f'category={category!r} escaped the gate: {result!r}'
            )
            mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_rejection_is_cheap_no_embedding_round_trip(self):
        """Gate ORDERING: a self-contradictory declaration must not pay for the
        near-duplicate guard's embedding search or write triage's judge call.

        The control below is load-bearing — without it this assertion would
        also pass if `search` were simply never reachable for this input, and
        the ordering claim would be vacuous.
        """
        mock_service = AsyncMock()
        _pass_through(mock_service)
        server = create_mcp_server(mock_service)

        await _call(server, category='procedural_knowledge', entities=_CONFLICTING)

        mock_service.search.assert_not_called()

        control = AsyncMock()
        _pass_through(control)
        control_server = create_mcp_server(control)

        await _call(control_server, category='procedural_knowledge')

        assert control.search.call_count, (
            'the control never reached the near-duplicate search, so the '
            'ordering assertion above proves nothing'
        )


class TestAddMemoryForwardsTheDeclaration:
    """An ACCEPTED declaration must reach the service, at BOTH call sites.

    The tool owns the gate; the service owns the resolve. So what crosses the
    hop is the caller's list VERBATIM, not a parsed `ReferentSet` — parsing
    twice would fork what `declared` means between the boundary and the
    producer that encodes it onto the durable-queue payload.
    """

    @pytest.mark.asyncio
    async def test_the_caller_list_is_forwarded_verbatim(self):
        mock_service = AsyncMock()
        _pass_through(mock_service)
        server = create_mcp_server(mock_service)

        await _call(server, entities=_CORROBORATING)

        kwargs = mock_service.add_memory.call_args.kwargs
        assert kwargs['declared_referents'] == [{'kind': 'task', 'id': 3127}], f'{kwargs!r}'

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('entities', 'expected'),
        [
            pytest.param(None, None, id='omitted-stays-None'),
            pytest.param([], [], id='empty-stays-empty'),
        ],
    )
    async def test_the_tri_state_survives_the_hop_intact(self, entities, expected):
        """Collapsing [] to None here would silently downgrade the write from
        source='declared' to 'derived'/'none' and corrupt the very counter
        leaf iota reads."""
        mock_service = AsyncMock()
        _pass_through(mock_service)
        server = create_mcp_server(mock_service)

        kwargs = {} if entities is None else {'entities': entities}
        await _call(server, **kwargs)

        forwarded = mock_service.add_memory.call_args.kwargs['declared_referents']
        assert forwarded == expected, f'{forwarded!r}'
        assert (forwarded is None) == (expected is None), (
            f'[] and None must stay distinguishable, got {forwarded!r}'
        )

    @pytest.mark.asyncio
    async def test_the_write_triage_fallback_retry_also_carries_it(self):
        """`add_memory` has a SECOND service call site: the standalone retry
        write triage falls back to when an attach raises (contract C1).

        A declaration dropped there is exactly the silent degradation this
        repo's loud-over-silent norm forbids — the write would land stamped
        `derived` while the agent believes it declared, and no error would say
        so. The fallback deliberately drops the failed parent link; it must
        NOT drop this, for the same reason it keeps the full content.

        Harness reused from `test_add_memory_write_triage_gate.py`'s C1 suite
        rather than re-derived, so a retune of its bands or its candidate shape
        carries this test with it.
        """
        mock_service = AsyncMock()
        _configure_config(mock_service, enabled=True)
        mem_result = MagicMock()
        mem_result.model_dump.return_value = {'id': 'fallback-id'}
        mock_service.add_memory.side_effect = [
            RuntimeError('parent_id rejected by the write seam'),
            mem_result,
        ]
        mock_service.search.return_value = [_candidate('m1', 0.97, content=_CONTENT)]
        server = create_mcp_server(mock_service)

        result = await _call(
            server, category='procedural_knowledge', entities=_CORROBORATING,
        )

        assert 'error' not in result, f'the write was blocked: {result!r}'
        assert mock_service.add_memory.await_count == 2, (
            f'expected attach then fallback: {mock_service.add_memory.await_args_list!r}'
        )
        for label, call in zip(
            ('attach', 'fallback'), mock_service.add_memory.await_args_list, strict=True,
        ):
            assert call.kwargs['declared_referents'] == [{'kind': 'task', 'id': 3127}], (
                f'the {label} call dropped the declaration: {call.kwargs!r}'
            )


# ---------------------------------------------------------------------------
# add_episode (step-11) — PRD Open Question 3, resolved in this leaf: BOTH
# write tools get `entities`, in one leaf, because the gate stack does not
# diverge. Both tools reach the gate with a canonical project_id and a
# FastMCP-enforced `str` content, and the gate reads nothing else off either.
#
# The one honest asymmetry is a tier BELOW this one: add_episode persists no
# metadata, so epsilon deliberately passes metadata=None at its producer and
# the metadata['task_id'] bridge is dead by construction there. That is a
# statement about tier 2, not about the tier-1 declaration these tests pin.
# ---------------------------------------------------------------------------

#: Recon-stage content that DOES reach the live task-status authority — the
#: byte-identical fixture test_recon_premature_completion_gate.py uses, so the
#: ordering control below is anchored to a claim we know fires that lookup.
_COMPLETION_CONTENT = 'task 5252 has landed and now enforces the manifest gate'
_KNOWN_PROJECTS = {'dark_factory': '/root'}


def _pass_through_episode(mock_service: AsyncMock, dump: dict | None = None) -> MagicMock:
    """Configure ``mock_service.add_episode`` to return a dict-dumpable result.

    Same shape trap as `_pass_through`: `add_episode` ends in
    `result.model_dump()`, so a bare AsyncMock return would make the tool's
    final expression an unawaited coroutine rather than a dict.
    """
    ep_result = MagicMock()
    ep_result.model_dump.return_value = dump if dump is not None else {'id': 'ep'}
    mock_service.add_episode.return_value = ep_result
    return ep_result


async def _call_episode(server, **overrides) -> dict:
    args = {
        'content': _CONTENT,
        'agent_id': 'claude-interactive',
        'project_id': _PROJECT_ID,
    }
    args.update(overrides)
    return await server._tool_manager.call_tool('add_episode', args)


class TestAddEpisodeEntitiesGate:
    """The same gate, wired into `add_episode`'s chain."""

    @pytest.mark.asyncio
    async def test_a_conflicting_declaration_rejects_and_does_not_write(self):
        mock_service = AsyncMock()
        _pass_through_episode(mock_service)
        server = create_mcp_server(mock_service)

        result = await _call_episode(server, entities=_CONFLICTING)

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result.get('error_type') == 'DeclaredReferentConflictRejected', f'{result!r}'
        assert result.get('error') == 'declared_referent_conflict', f'{result!r}'
        assert result.get('agent_id') == 'claude-interactive', f'{result!r}'
        # INV-2: both sides named, exactly as at add_memory's boundary.
        assert result.get('declared') == ['Task 3129'], f'{result!r}'
        assert result.get('conflicts') == ['Task 3129'], f'{result!r}'
        assert result.get('content_referents') == ['Task 3127'], f'{result!r}'
        assert result.get('hint'), f'{result!r}'
        mock_service.add_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_malformed_declaration_rejects_and_does_not_write(self):
        mock_service = AsyncMock()
        _pass_through_episode(mock_service)
        server = create_mcp_server(mock_service)

        result = await _call_episode(server, entities=_MALFORMED)

        assert result.get('error_type') == 'ValidationError', f'{result!r}'
        assert result.get('content_excerpt') == _CONTENT[:200], f'{result!r}'
        mock_service.add_episode.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'entities',
        [
            pytest.param(None, id='omitted'),
            pytest.param([], id='considered-none-apply'),
            pytest.param(_CORROBORATING, id='corroborating'),
        ],
    )
    async def test_absence_and_agreement_always_write(self, entities):
        mock_service = AsyncMock()
        _pass_through_episode(mock_service)
        server = create_mcp_server(mock_service)

        kwargs = {} if entities is None else {'entities': entities}
        result = await _call_episode(server, **kwargs)

        assert 'error' not in result, f'entities={entities!r} was blocked: {result!r}'
        mock_service.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_the_rejection_is_cheap_no_live_authority_round_trip(self):
        """Gate ORDERING: the declaration is checked ahead of every gate that
        does live authority I/O.

        `add_episode`'s chain is far more expensive than `add_memory`'s once a
        recon-stage agent is writing a completion claim: the 2824
        premature-completion gate reads live task statuses, and
        `_completion_claim_gate` reads statuses / tickets / git after it. A
        structurally self-contradictory declaration should pay for none of
        them.

        The control is load-bearing for the same reason as add_memory's: with
        no call that DOES reach `get_statuses`, "get_statuses was not called"
        would also pass if this input could never reach it, and the ordering
        claim would be vacuous.
        """
        task_interceptor = MagicMock()
        task_interceptor.get_statuses = AsyncMock(return_value={'5252': 'in-progress'})
        mock_service = AsyncMock()
        _pass_through_episode(mock_service)
        server = create_mcp_server(
            mock_service,
            task_interceptor=task_interceptor,
            known_projects=_KNOWN_PROJECTS,
        )

        result = await _call_episode(
            server,
            content=_COMPLETION_CONTENT,
            agent_id='recon-stage-task_knowledge_sync',
            entities=_CONFLICTING,
        )

        assert result.get('error_type') == 'DeclaredReferentConflictRejected', f'{result!r}'
        task_interceptor.get_statuses.assert_not_called()
        mock_service.add_episode.assert_not_called()

        control_interceptor = MagicMock()
        control_interceptor.get_statuses = AsyncMock(return_value={'5252': 'in-progress'})
        control_service = AsyncMock()
        _pass_through_episode(control_service)
        control_server = create_mcp_server(
            control_service,
            task_interceptor=control_interceptor,
            known_projects=_KNOWN_PROJECTS,
        )

        await _call_episode(
            control_server,
            content=_COMPLETION_CONTENT,
            agent_id='recon-stage-task_knowledge_sync',
        )

        assert control_interceptor.get_statuses.call_count, (
            'the control never reached the live task-status lookup, so the '
            'ordering assertion above proves nothing'
        )

    @pytest.mark.asyncio
    async def test_the_pre_existing_temporal_context_arm_is_undisturbed(self):
        """The new chain link must not shadow or reorder what was already there."""
        mock_service = AsyncMock()
        _pass_through_episode(mock_service)
        server = create_mcp_server(mock_service)

        result = await _call_episode(server, temporal_context='nonsense')

        assert result.get('error_type') == 'ValidationError', f'{result!r}'
        assert 'temporal_context' in str(result.get('error', '')), f'{result!r}'
        mock_service.add_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_valid_reference_time_still_parses(self):
        mock_service = AsyncMock()
        _pass_through_episode(mock_service)
        server = create_mcp_server(mock_service)

        result = await _call_episode(
            server,
            reference_time='2026-03-22T00:00:00+00:00',
            entities=_CORROBORATING,
        )

        assert 'error' not in result, f'{result!r}'
        mock_service.add_episode.assert_called_once()


class TestAddEpisodeForwardsTheDeclaration:
    """`add_episode` has exactly ONE service call site (unlike add_memory's
    two), and the accepted declaration must cross it verbatim."""

    @pytest.mark.asyncio
    async def test_the_caller_list_is_forwarded_verbatim(self):
        mock_service = AsyncMock()
        _pass_through_episode(mock_service)
        server = create_mcp_server(mock_service)

        await _call_episode(server, entities=_CORROBORATING)

        kwargs = mock_service.add_episode.call_args.kwargs
        assert kwargs['declared_referents'] == [{'kind': 'task', 'id': 3127}], f'{kwargs!r}'

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('entities', 'expected'),
        [
            pytest.param(None, None, id='omitted-stays-None'),
            pytest.param([], [], id='empty-stays-empty'),
        ],
    )
    async def test_the_tri_state_survives_the_hop_intact(self, entities, expected):
        mock_service = AsyncMock()
        _pass_through_episode(mock_service)
        server = create_mcp_server(mock_service)

        kwargs = {} if entities is None else {'entities': entities}
        await _call_episode(server, **kwargs)

        forwarded = mock_service.add_episode.call_args.kwargs['declared_referents']
        assert forwarded == expected, f'{forwarded!r}'
        assert (forwarded is None) == (expected is None), (
            f'[] and None must stay distinguishable, got {forwarded!r}'
        )


class TestEveryMalformedShapeReachesTheGate:
    """The gate — not pydantic — answers EVERY wrong shape, at both tools.

    `entities` is annotated `Any` on both tools precisely so this class can
    exist. A narrower `list[dict] | None` would put FastMCP/pydantic's
    validator ahead of the tool body, and the two mistakes an agent is
    likeliest to make — a bare `'task 3127'`, a single un-wrapped
    `{'kind': ..., 'id': ...}` — would come back as a raw ToolError
    ("Input should be a valid list") carrying NO remediation, while a
    structurally-valid-but-wrong entry like `[{'id': 'abc'}] `reached the gate
    and came back with the hint. The write is blocked either way, so this is
    not a correctness defect; it is a legibility one, and precisely the
    guessing behaviour `_DECLARED_REFERENT_HINT` exists to prevent.

    So every case here asserts the AGENT-VISIBLE outcome: a returned dict (not
    a raised ToolError), the house-shape ValidationError, and the accepted
    entry shape inside the message. Re-narrowing either annotation turns this
    class red rather than silently reverting the boundary to two answers.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize('declared', _MALFORMED_SHAPES)
    async def test_add_memory_answers_with_the_hinted_block(self, declared):
        mock_service = AsyncMock()
        _pass_through(mock_service)
        server = create_mcp_server(mock_service)

        result = await _call(server, entities=declared)

        assert isinstance(result, dict), f'expected a dict, got {type(result)}: {result!r}'
        assert result.get('error_type') == 'ValidationError', f'{result!r}'
        assert _DECLARED_REFERENT_HINT in result.get('error', ''), f'{result!r}'
        assert result.get('content_excerpt') == _CONTENT[:200], f'{result!r}'
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('declared', _MALFORMED_SHAPES)
    async def test_add_episode_answers_with_the_hinted_block(self, declared):
        mock_service = AsyncMock()
        _pass_through_episode(mock_service)
        server = create_mcp_server(mock_service)

        result = await _call_episode(server, entities=declared)

        assert isinstance(result, dict), f'expected a dict, got {type(result)}: {result!r}'
        assert result.get('error_type') == 'ValidationError', f'{result!r}'
        assert _DECLARED_REFERENT_HINT in result.get('error', ''), f'{result!r}'
        assert result.get('content_excerpt') == _CONTENT[:200], f'{result!r}'
        mock_service.add_episode.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('tool', ['add_memory', 'add_episode'])
    async def test_the_declared_shape_is_still_advertised_in_the_tool_schema(self, tool):
        """The one thing `Any` costs, pinned so it stays paid for.

        Widening drops the JSON-schema type constraint, so the accepted shape
        survives only in the docstring Args text FastMCP publishes as the tool
        description. That text is now the parameter's ONLY machine-readable
        shape hint, which makes deleting it a silent regression rather than a
        docs edit — this asserts it is there.
        """
        server = create_mcp_server(AsyncMock())
        (spec,) = [t for t in server._tool_manager.list_tools() if t.name == tool]

        assert 'entities' in spec.parameters['properties'], f'{spec.parameters!r}'
        description = spec.description or ''
        assert "{'kind': 'task', 'id': <digits>" in description, (
            f'{tool} no longer documents the accepted `entities` entry shape, '
            'which is the only shape hint left once the annotation is `Any`'
        )
