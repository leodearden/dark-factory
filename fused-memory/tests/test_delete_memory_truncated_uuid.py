"""Regression tests pinning delete_memory's HARD ERROR on truncated UUIDs.

Task 3132 (PRD memory-write-path-convergence §9 leaf η) inverted this file's
original premise. It used to pin the *silent no-op*: an 8-char hex prefix
lifted out of a search-result snippet (e.g. '2531b4d8') produced a fully
confirming `{'status': 'deleted'}` envelope, a `success=True` write-journal
entry and a `memory_deleted` event, while nothing was deleted. DF 1144 tried
to fix that with prompt wording alone; a prompt cannot enforce an API
contract, so the failure recurred. The enforcement now lives in the API, in
two layers that this file pins in the order a call passes through them:

`TestMemoryServiceDeleteMemoryRejectsTruncatedUuid` — `MemoryService.
delete_memory` raises `InputValidationError`. It uses a real `MemoryService`
with mocked backends rather than an `AsyncMock` service, because an
`AsyncMock` would intercept `delete_memory` before the guard could run — and
it is the service-level guard specifically that makes the enforcement real
for the internal (non-MCP) reconciliation and script callers.

`TestDeleteMemoryToolRejectsTruncatedUuid` and
`TestTruncatedUuidNeverReachesCitationRepointGate` — the agent-facing MCP
boundary, which returns a structured `ValidationError` naming the offending
id plus a `hint`, and rejects before either the service or the
citation-repoint gate is reached. Both layers exist because they cover
different callers and because `mcp_tool_errors` flattens an exception to
`{'error', 'error_type'}` and cannot carry the hint through a raise.

`TestGraphitiBackendRemoveEdgeTruncatedUuid` is deliberately UNCHANGED and
still passing. `GraphitiBackend.remove_edge` still swallows
`EdgeNotFoundError` and returns `None` without calling `edge.delete`, and
that is still correct for a genuinely already-deleted edge — hardening it
would change idempotent-delete semantics for every caller. It is retained
because it documents the layer *behind* the new guard: anything that
bypassed the guard would still fail silently, which is precisely why the
guard has to sit above it.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.server.tools import create_mcp_server
from fused_memory.utils.validation import InputValidationError

# A search-result snippet prefix, per stage1.py's own example — 8 hex chars,
# nowhere near a valid 36-char UUID.
TRUNCATED_PREFIX = '2531b4d8'

# Invariant that makes TRUNCATED_PREFIX a reliable not-found trigger: a
# truncated hex prefix can never equal a full 36-char UUID primary key, so a
# real driver lookup keyed on it always misses and get_by_uuid always raises
# EdgeNotFoundError. The test below injects that error directly rather than
# exercising a real driver lookup — this assertion documents *why* the
# injected error is representative of what a truncated prefix actually
# causes in production, rather than pinning behavior for an arbitrary string.
assert len(TRUNCATED_PREFIX) != 36

# A well-formed id, for the non-regression assertions: an over-broad validator
# that also rejected real ids would break every legitimate delete.
CANONICAL_UUID = 'fccfa232-16fe-404a-880a-8eca32eb974e'
assert len(CANONICAL_UUID) == 36


@pytest.fixture
def service(mock_config):
    """Real MemoryService with mocked backends.

    Deliberately NOT an AsyncMock service: an AsyncMock would intercept
    `delete_memory` wholesale, so the guard under test would never run. Only
    the real method makes service-layer enforcement observable.
    """
    from fused_memory.services.memory_service import MemoryService

    svc = MemoryService(mock_config)
    svc.graphiti = MagicMock()
    svc.graphiti.remove_edge = AsyncMock()
    svc.mem0 = MagicMock()
    svc.mem0.delete = AsyncMock(return_value={'message': 'deleted'})
    return svc


class TestGraphitiBackendRemoveEdgeTruncatedUuid:
    """Companion to TestGraphitiBackendRemoveEdge in test_memory_service.py
    (currently :1659), which covers remove_edge's happy path (get_by_uuid
    finds an edge, edge.delete is called). This class covers the not-found
    branch. These would ideally be co-located as methods on one class —
    left separate here because test_memory_service.py is outside this
    task's locked file scope (task 2673 amendment pass); re-surface as a
    follow-up if this file is touched again.
    """

    @pytest.mark.asyncio
    async def test_remove_edge_silently_no_ops_on_truncated_prefix(self, mock_config):
        """remove_edge(truncated_prefix) must not raise, and must not call
        edge.delete — EdgeNotFoundError from the failed lookup is swallowed.

        This pins the generic EdgeNotFoundError -> return-None swallow
        branch by injecting the error directly into get_by_uuid, rather
        than exercising a real driver lookup keyed on TRUNCATED_PREFIX. The
        module-level length invariant above documents why a truncated
        prefix reliably drives this exact branch in production.
        """
        from graphiti_core.errors import EdgeNotFoundError

        from fused_memory.backends.graphiti_client import GraphitiBackend

        backend = GraphitiBackend(mock_config)
        mock_driver = MagicMock()
        mock_cloned_driver = MagicMock()
        mock_driver.clone = MagicMock(return_value=mock_cloned_driver)
        backend._driver = mock_driver
        backend.client = MagicMock()

        with patch(
            'fused_memory.backends.graphiti_client.EntityEdge'
        ) as MockEntityEdge:
            MockEntityEdge.get_by_uuid = AsyncMock(
                side_effect=EdgeNotFoundError(TRUNCATED_PREFIX)
            )

            # Must not raise.
            result = await backend.remove_edge(TRUNCATED_PREFIX, group_id='test')

            assert result is None
            MockEntityEdge.get_by_uuid.assert_called_once_with(
                mock_cloned_driver, TRUNCATED_PREFIX
            )


# NOTE (task 3132): a second class here once duplicated
# MemoryService.delete_memory(store='graphiti') passthrough coverage and was
# dropped in task 2673's amendment pass, on the grounds that delete_memory
# "has no awareness of whether the id it was given was truncated". That is no
# longer true — the class below is what makes it false, and it therefore pins
# genuinely new behaviour rather than restating test_memory_service.py's
# happy-path wrapper assertions with a different string literal.


class TestMemoryServiceDeleteMemoryRejectsTruncatedUuid:
    """MemoryService.delete_memory raises on any non-full-UUID id.

    This is the API-level enforcement DF 1144's prompt fix could not achieve:
    it binds the internal callers (reconciliation stages, the scripts/
    maintenance sweeps) as well as the MCP boundary.
    """

    @pytest.mark.asyncio
    async def test_graphiti_store_raises(self, service):
        with pytest.raises(InputValidationError) as exc_info:
            await service.delete_memory(
                memory_id=TRUNCATED_PREFIX, store='graphiti', project_id='test'
            )
        assert TRUNCATED_PREFIX in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_mem0_store_raises(self, service):
        """One guard covers BOTH store paths — not two copies that must agree."""
        with pytest.raises(InputValidationError) as exc_info:
            await service.delete_memory(
                memory_id=TRUNCATED_PREFIX, store='mem0', project_id='test'
            )
        assert TRUNCATED_PREFIX in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_graphiti_backend_not_reached(self, service):
        with pytest.raises(InputValidationError):
            await service.delete_memory(
                memory_id=TRUNCATED_PREFIX, store='graphiti', project_id='test'
            )
        service.graphiti.remove_edge.assert_not_called()

    @pytest.mark.asyncio
    async def test_mem0_backend_not_reached(self, service):
        with pytest.raises(InputValidationError):
            await service.delete_memory(
                memory_id=TRUNCATED_PREFIX, store='mem0', project_id='test'
            )
        service.mem0.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_write_journal_entry_on_rejection(self, service):
        """The false audit trail is a distinct harm of the old behaviour: a
        rejected delete must leave no success=True write-op record."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        mock_journal.log_backend_op = AsyncMock()
        service._write_journal = mock_journal

        with pytest.raises(InputValidationError):
            await service.delete_memory(
                memory_id=TRUNCATED_PREFIX, store='mem0', project_id='test'
            )

        mock_journal.log_write_op.assert_not_awaited()
        mock_journal.log_backend_op.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_memory_deleted_event_on_rejection(self, service):
        """A rejected delete must not announce a deletion to reconciliation."""
        buffer = MagicMock()
        buffer.push = AsyncMock()
        service._event_buffer = buffer

        with pytest.raises(InputValidationError):
            await service.delete_memory(
                memory_id=TRUNCATED_PREFIX, store='graphiti', project_id='test'
            )

        buffer.push.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_canonical_uuid_still_deletes(self, service):
        """Non-regression: guards against an over-broad validator."""
        result = await service.delete_memory(
            memory_id=CANONICAL_UUID, store='graphiti', project_id='test'
        )
        assert result['status'] == 'deleted'
        service.graphiti.remove_edge.assert_called_once_with(
            CANONICAL_UUID, group_id='test'
        )


# ── MCP tool boundary ────────────────────────────────────────────────────────
# Harness modelled on test_delete_memory_alias.py (the real delete_memory
# MCP-tool test): create_mcp_server + _tool_manager.call_tool + _parse_result
# unwrapping FastMCP TextContent.

# A well-formed survivor id for the citation-repoint ordering case below.
REPLACEMENT_UUID = '9f3ac071-3333-4eee-8fff-000000000003'
RECON_AGENT = 'recon-stage-memory_consolidator'
KNOWN_PROJECTS = {'dark_factory': '/tmp/root'}


def _parse_result(result):
    """Parse a FastMCP call_tool result (list of TextContent) into a dict."""
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


class TestDeleteMemoryToolRejectsTruncatedUuid:
    """The agent-facing MCP boundary rejects a truncated id structurally.

    The tool returns a dict rather than letting the service's exception
    surface through mcp_tool_errors, because that decorator emits only
    {'error', 'error_type'} and would drop the `hint` the signal requires.
    """

    @pytest.fixture
    def mock_service(self):
        svc = AsyncMock()
        svc.delete_memory = AsyncMock(
            return_value={'status': 'deleted', 'store': 'mem0', 'id': CANONICAL_UUID}
        )
        svc.get_memory_by_id = AsyncMock(return_value=None)
        return svc

    @pytest.fixture
    def mcp_server(self, mock_service):
        return create_mcp_server(mock_service)

    async def _call(self, mcp_server, **overrides):
        args = {
            'memory_id': TRUNCATED_PREFIX,
            'store': 'mem0',
            'project_id': 'dark_factory',
        }
        args.update(overrides)
        return _parse_result(
            await mcp_server._tool_manager.call_tool('delete_memory', args)
        )

    @pytest.mark.asyncio
    async def test_mem0_truncated_id_returns_validation_error(self, mcp_server):
        parsed = await self._call(mcp_server)
        assert parsed.get('error_type') == 'ValidationError'

    @pytest.mark.asyncio
    async def test_error_names_the_malformed_id(self, mcp_server):
        """The signal explicitly requires the malformed id be named."""
        parsed = await self._call(mcp_server)
        assert TRUNCATED_PREFIX in parsed['error']

    @pytest.mark.asyncio
    async def test_error_carries_a_hint(self, mcp_server):
        parsed = await self._call(mcp_server)
        assert parsed.get('hint')

    @pytest.mark.asyncio
    async def test_does_not_report_deleted(self, mcp_server):
        """Directly inverts the documented defect: the old behaviour returned a
        fully confirming {'status': 'deleted'} for exactly this input."""
        parsed = await self._call(mcp_server)
        assert parsed.get('status') != 'deleted'

    @pytest.mark.asyncio
    async def test_service_not_reached(self, mcp_server, mock_service):
        await self._call(mcp_server)
        mock_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_graphiti_truncated_id_returns_validation_error(
        self, mcp_server, mock_service
    ):
        parsed = await self._call(mcp_server, store='graphiti')
        assert parsed.get('error_type') == 'ValidationError'
        assert TRUNCATED_PREFIX in parsed['error']
        mock_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_id_alias_cannot_bypass_the_guard(self, mcp_server, mock_service):
        """The `id` alias resolves to the same value, so it is guarded too."""
        parsed = _parse_result(
            await mcp_server._tool_manager.call_tool(
                'delete_memory',
                {'id': TRUNCATED_PREFIX, 'store': 'mem0', 'project_id': 'dark_factory'},
            )
        )
        assert parsed.get('error_type') == 'ValidationError'
        assert TRUNCATED_PREFIX in parsed['error']
        mock_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_store_error_still_wins(self, mcp_server):
        """Ordering guard #1: the guard sits BELOW the _VALID_STORES check, which
        is what keeps test_health_endpoint's invalid-store assertions green."""
        parsed = await self._call(mcp_server, store='redis')
        assert 'redis' in parsed['error']
        assert TRUNCATED_PREFIX not in parsed['error']

    @pytest.mark.asyncio
    async def test_canonical_uuid_still_delegates(self, mcp_server, mock_service):
        """Non-regression: a well-formed id passes straight through."""
        await self._call(mcp_server, memory_id=CANONICAL_UUID)
        mock_service.delete_memory.assert_awaited_once()
        assert mock_service.delete_memory.call_args[1]['memory_id'] == CANONICAL_UUID


class TestTruncatedUuidNeverReachesCitationRepointGate:
    """Ordering guard #2: the repoint gate mutates live task metadata ahead of
    an irreversible delete, so a malformed id must be rejected before it can
    drive any repoint."""

    @pytest.fixture
    def mock_service(self):
        svc = AsyncMock()
        svc.delete_memory = AsyncMock(
            return_value={'status': 'deleted', 'store': 'mem0', 'id': CANONICAL_UUID}
        )
        svc.get_memory_by_id = AsyncMock(
            return_value={'id': REPLACEMENT_UUID, 'content': 'survivor', 'metadata': {}}
        )
        return svc

    @pytest.fixture
    def interceptor(self):
        itc = MagicMock()
        itc.get_tasks = AsyncMock(return_value={'tasks': [
            {
                'id': '101',
                'status': 'pending',
                'title': 'task 101',
                'metadata': {'mem0_canonical_entry': TRUNCATED_PREFIX},
            },
        ]})
        itc.update_task = AsyncMock(return_value={'success': True})
        return itc

    @pytest.fixture
    def mcp_server(self, mock_service, interceptor):
        return create_mcp_server(
            mock_service, task_interceptor=interceptor, known_projects=KNOWN_PROJECTS
        )

    @pytest.mark.asyncio
    async def test_truncated_id_with_valid_replacement_is_a_validation_error(
        self, mcp_server
    ):
        """Not a CitationReplacement* error — the gate is never entered."""
        parsed = _parse_result(
            await mcp_server._tool_manager.call_tool('delete_memory', {
                'memory_id': TRUNCATED_PREFIX,
                'store': 'mem0',
                'project_id': 'dark_factory',
                'agent_id': RECON_AGENT,
                'replacement_memory_id': REPLACEMENT_UUID,
            })
        )
        assert parsed.get('error_type') == 'ValidationError'
        assert TRUNCATED_PREFIX in parsed['error']

    @pytest.mark.asyncio
    async def test_gate_never_reads_or_mutates_tasks(self, mcp_server, interceptor):
        await mcp_server._tool_manager.call_tool('delete_memory', {
            'memory_id': TRUNCATED_PREFIX,
            'store': 'mem0',
            'project_id': 'dark_factory',
            'agent_id': RECON_AGENT,
            'replacement_memory_id': REPLACEMENT_UUID,
        })
        interceptor.get_tasks.assert_not_awaited()
        interceptor.update_task.assert_not_awaited()
