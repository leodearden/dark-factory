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

# The C1 fallback harness, imported rather than re-derived: reaching
# `add_memory`'s SECOND service call site needs write triage enabled with a
# calibrated high-cosine candidate, and a private copy would drift the moment
# those bands are retuned. The bare `server.` import resolves because
# fused-memory/tests/conftest.py inserts the tests dir onto sys.path and
# tests/server/ is a package (see the same note in
# test_add_memory_near_duplicate_gate.py, which imports test_config_schema the
# same way).
from server.test_add_memory_write_triage_gate import _candidate, _configure_config

from fused_memory.server.tools import create_mcp_server

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

        control.search.assert_called(), (
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
