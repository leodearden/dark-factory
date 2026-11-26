"""Behaviour tests for the get_memory_by_id MCP tool (task 2765).

Direct Mem0/Qdrant point-id lookup (raw content + metadata), non-semantic —
bypasses BOTH semantic search ranking and metadata-equality filtering. Read-only,
Mem0-only, not in any DISALLOW_* list (auto-allowed in Stage 1/Stage 3).

Load-bearing no-silent-fail contract: a GENUINE not-found returns
``{'found': False}`` (a valid answer), while a Qdrant timeout/backend error
surfaces as ``{'error', 'error_type'}`` with ``found`` ABSENT — so the
reconciliation consumer can tell "memory genuinely absent/folded" apart from
"backend timed out".

Mirrors tests/server/test_get_memories_by_metadata_tool.py.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest

from fused_memory.backends.mem0_client import Mem0Backend
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import MemoryService

_PROJECT_ID = 'dark_factory'
_MEMORY_ID = '77a3f6bc-0000-0000-0000-000000000000'
_SAMPLE_METADATA = {
    'data': 'the raw memory content',
    'category': 'observations_and_summaries',
    'agent_id': 'x',
}


class TestGetMemoryByIdTool:
    """Behaviour tests for mcp__fused-memory__get_memory_by_id."""

    @pytest.mark.asyncio
    async def test_happy_path_found(self):
        """Found: tool returns {'found':True, ids, content, metadata} and calls
        the service exactly once with the right project_id and memory_id."""
        mock_service = AsyncMock()
        mock_service.get_memory_by_id = AsyncMock(
            return_value={'id': _MEMORY_ID, 'content': 'the raw memory content', 'metadata': _SAMPLE_METADATA}
        )
        # A bare AsyncMock returns an AsyncMock from the child-count probe the
        # tool now issues (task 3129); pin the real zero-child corpus so this
        # keeps asserting the UNGROUPED happy-path shape.
        mock_service.count_memories_by_metadata = AsyncMock(return_value=0)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_memory_by_id',
            {'project_id': _PROJECT_ID, 'memory_id': _MEMORY_ID},
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result == {
            'found': True,
            'memory_id': _MEMORY_ID,
            'project_id': _PROJECT_ID,
            'content': 'the raw memory content',
            'metadata': _SAMPLE_METADATA,
        }, f'Unexpected result: {result!r}'
        assert 'error' not in result, f'Unexpected error in result: {result!r}'

        mock_service.get_memory_by_id.assert_called_once_with(
            project_id=_PROJECT_ID,
            memory_id=_MEMORY_ID,
        )

    @pytest.mark.asyncio
    async def test_not_found_returns_found_false_not_error(self):
        """Genuine miss: service returns None → {'found':False, ...} and NOT an error."""
        mock_service = AsyncMock()
        mock_service.get_memory_by_id = AsyncMock(return_value=None)
        # Explicit: a bare AsyncMock() would auto-create this as a coroutine
        # returning a truthy mock, i.e. a phantom tombstone. The real service
        # returns None when no tombstone row exists (task 3041).
        mock_service.get_mem0_deletion_tombstone = AsyncMock(return_value=None)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_memory_by_id',
            {'project_id': _PROJECT_ID, 'memory_id': _MEMORY_ID},
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result == {
            'found': False,
            'memory_id': _MEMORY_ID,
            'project_id': _PROJECT_ID,
        }, f'Unexpected result: {result!r}'
        assert 'error' not in result, (
            f'A genuine not-found must not be an error dict: {result!r}'
        )

    @pytest.mark.asyncio
    async def test_invalid_project_id_returns_validation_error(self):
        """Invalid project_id (unsafe chars) returns a validation error dict and
        does NOT call the service."""
        mock_service = AsyncMock()
        mock_service.get_memory_by_id = AsyncMock(return_value=None)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_memory_by_id',
            {'project_id': 'bad id!', 'memory_id': _MEMORY_ID},
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' in result, f'Expected error key in result: {result!r}'
        assert result.get('error_type') == 'ValidationError', (
            f"Expected error_type='ValidationError', got: {result!r}"
        )
        mock_service.get_memory_by_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_service_exception_caught(self):
        """A service exception is caught and returned as {'error','error_type'}."""
        mock_service = AsyncMock()
        mock_service.get_memory_by_id = AsyncMock(side_effect=ValueError('boom'))
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_memory_by_id',
            {'project_id': _PROJECT_ID, 'memory_id': _MEMORY_ID},
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' in result, f'Expected error key in result: {result!r}'
        assert result.get('error_type') == 'ValueError', (
            f"Expected error_type='ValueError', got: {result!r}"
        )
        assert 'boom' in result.get('error', ''), (
            f'Expected original error message in result: {result!r}'
        )

    @pytest.mark.asyncio
    async def test_timeout_surfaces_as_error_not_found_false(self, mock_config):
        """A Qdrant retrieve timeout surfaces as {'error','error_type':'TimeoutError'}
        — it must NOT masquerade as a genuine {'found': False} miss (no-silent-fail
        invariant; the whole reason this tool distinguishes miss from timeout).

        Drives a REAL MemoryService + REAL Mem0Backend (only the underlying async
        Qdrant client is patched) so the full get_point_by_id → get_memory_by_id →
        @mcp_tool_errors chain is genuinely exercised.
        """
        svc = MemoryService(mock_config)
        svc.mem0 = Mem0Backend(mock_config)
        mock_client = AsyncMock()
        mock_client.retrieve = AsyncMock(side_effect=TimeoutError('qdrant retrieve timed out'))
        server = create_mcp_server(svc)

        with patch.object(svc.mem0, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            result = await server._tool_manager.call_tool(
                'get_memory_by_id',
                {'project_id': _PROJECT_ID, 'memory_id': _MEMORY_ID},
            )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' in result, f'Expected error key in result: {result!r}'
        assert result.get('error_type') == 'TimeoutError', (
            f"Expected error_type='TimeoutError', got: {result!r}"
        )
        assert result.get('found') is None, (
            f'A timeout must NOT masquerade as found:False: {result!r}'
        )

    @pytest.mark.asyncio
    async def test_not_found_includes_tombstone_when_one_exists(self):
        """A deliberately-reaped record now SELF-EXPLAINS on the miss branch.

        This is the closure for the recon-gate-165 / esc-165-1 audit dead-end:
        the auditor ran exactly this query, got {'found': False}, and had no
        reachable path from a memory uuid to "who deleted it and why". The
        answer now rides along with the miss — no new tool to discover, no
        extra call to know about.
        """
        tombstone = {
            'deleter': 'stage1_cycle_summary_trim',
            'deleting_run_id': 'run-deleter',
            'deleted_at': '2026-07-20T00:00:00+00:00',
            'kind': 'cycle_summary',
            'record_type': 'ledger_stamp',
            'recon_pool': 'stage1_cycle_summary',
            'run_id': '84eae9bd',
            'created_at': '2026-07-18T00:00:00+00:00',
            'tombstone_created_at': '2026-07-20T00:00:00+00:00',
            'tombstone_expires_at': '2026-08-19T00:00:00+00:00',
        }
        mock_service = AsyncMock()
        mock_service.get_memory_by_id = AsyncMock(return_value=None)
        mock_service.get_mem0_deletion_tombstone = AsyncMock(return_value=tombstone)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_memory_by_id',
            {'project_id': _PROJECT_ID, 'memory_id': _MEMORY_ID},
        )

        assert result == {
            'found': False,
            'memory_id': _MEMORY_ID,
            'project_id': _PROJECT_ID,
            'tombstone': tombstone,
        }, f'Unexpected result: {result!r}'
        assert 'error' not in result, (
            f'A tombstoned miss is still a normal answer, not an error: {result!r}'
        )
        mock_service.get_mem0_deletion_tombstone.assert_awaited_once_with(
            _PROJECT_ID, _MEMORY_ID
        )

    @pytest.mark.asyncio
    async def test_not_found_omits_tombstone_key_when_absent(self):
        """No tombstone → the key is OMITTED entirely, not present as None.

        Keeps the ordinary never-existed miss byte-identical to today's shape,
        so the presence of the key is itself the signal "this was deliberately
        reaped" rather than something a consumer must null-check.
        """
        mock_service = AsyncMock()
        mock_service.get_memory_by_id = AsyncMock(return_value=None)
        mock_service.get_mem0_deletion_tombstone = AsyncMock(return_value=None)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_memory_by_id',
            {'project_id': _PROJECT_ID, 'memory_id': _MEMORY_ID},
        )

        assert 'tombstone' not in result, (
            f'tombstone key must be omitted, not None, when absent: {result!r}'
        )
        assert result == {
            'found': False,
            'memory_id': _MEMORY_ID,
            'project_id': _PROJECT_ID,
        }, f'Unexpected result: {result!r}'

    @pytest.mark.asyncio
    async def test_found_does_not_look_up_a_tombstone(self):
        """The HIT branch is untouched: no tombstone lookup, no extra key.

        A living record cannot have been deleted, so the lookup would be pure
        cost on the common path.
        """
        mock_service = AsyncMock()
        mock_service.get_memory_by_id = AsyncMock(
            return_value={'id': _MEMORY_ID, 'content': 'c', 'metadata': _SAMPLE_METADATA}
        )
        mock_service.get_mem0_deletion_tombstone = AsyncMock(return_value={'deleter': 'x'})
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_memory_by_id',
            {'project_id': _PROJECT_ID, 'memory_id': _MEMORY_ID},
        )

        assert result['found'] is True
        assert 'tombstone' not in result, f'HIT branch must not carry a tombstone: {result!r}'
        mock_service.get_mem0_deletion_tombstone.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tombstone_lookup_failure_still_returns_clean_found_false(self, caplog):
        """A raising tombstone lookup must NOT convert a correct miss into an error.

        The tombstone is diagnostic garnish on an answer that is already
        right. Letting its failure surface as {'error'} would break the
        load-bearing miss-vs-backend-failure distinction this tool exists for
        — trading a real signal for a decorative one.

        It must still be LOUD in the log (task 3041 amendment pass).
        get_mem0_deletion_tombstone is internally fail-safe for every ordinary
        case (no ledger, no row, malformed payload), so anything that reaches
        this handler is an unexpected fault — a silent degrade would leave the
        auditor unable to tell "no tombstone exists" from "the tombstone store
        is broken", which is the very undiscoverability this task exists to
        fix (loud-over-silent / no-silent-fail-soft).
        """
        mock_service = AsyncMock()
        mock_service.get_memory_by_id = AsyncMock(return_value=None)
        mock_service.get_mem0_deletion_tombstone = AsyncMock(
            side_effect=RuntimeError('ledger exploded')
        )
        server = create_mcp_server(mock_service)

        with caplog.at_level(logging.WARNING, logger='fused_memory.server.tools'):
            result = await server._tool_manager.call_tool(
                'get_memory_by_id',
                {'project_id': _PROJECT_ID, 'memory_id': _MEMORY_ID},
            )

        assert result == {
            'found': False,
            'memory_id': _MEMORY_ID,
            'project_id': _PROJECT_ID,
        }, f'Unexpected result: {result!r}'
        assert 'error' not in result, (
            f'A tombstone failure must not become a tool error: {result!r}'
        )

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'fused_memory.server.tools'
        ]
        assert len(warnings) == 1, (
            f'a broken tombstone store must log exactly one WARNING, got {warnings!r}'
        )
        assert _MEMORY_ID in warnings[0].getMessage()
        assert warnings[0].exc_info is not None, (
            'the WARNING must carry exc_info — the fault type is the diagnostic'
        )

    @pytest.mark.asyncio
    async def test_absent_tombstone_is_silent(self, caplog):
        """The ORDINARY miss (no tombstone) must not log — only faults are loud.

        Guards the other half of the loud-over-silent split: if every plain
        never-deleted lookup warned, the WARNING above would be noise and stop
        meaning "the tombstone store is broken".
        """
        mock_service = AsyncMock()
        mock_service.get_memory_by_id = AsyncMock(return_value=None)
        mock_service.get_mem0_deletion_tombstone = AsyncMock(return_value=None)
        server = create_mcp_server(mock_service)

        with caplog.at_level(logging.WARNING, logger='fused_memory.server.tools'):
            result = await server._tool_manager.call_tool(
                'get_memory_by_id',
                {'project_id': _PROJECT_ID, 'memory_id': _MEMORY_ID},
            )

        assert result == {
            'found': False,
            'memory_id': _MEMORY_ID,
            'project_id': _PROJECT_ID,
        }
        assert not [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and r.name == 'fused_memory.server.tools'
        ], f'an ordinary absent tombstone must be silent: {caplog.text!r}'

    @pytest.mark.asyncio
    async def test_backend_failure_still_omits_found_with_a_tombstone_available(self):
        """No-silent-fail survives the addition: a backend error is still
        {'error','error_type'} with `found` ABSENT, even when the record has a
        tombstone that a naive implementation might have surfaced instead.

        A tombstone answers "why is this gone"; it must never be used to
        manufacture an answer when the backend never told us whether it IS gone.
        """
        mock_service = AsyncMock()
        mock_service.get_memory_by_id = AsyncMock(side_effect=TimeoutError('qdrant timed out'))
        mock_service.get_mem0_deletion_tombstone = AsyncMock(
            return_value={'deleter': 'stage1_cycle_summary_trim'}
        )
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_memory_by_id',
            {'project_id': _PROJECT_ID, 'memory_id': _MEMORY_ID},
        )

        assert result.get('error_type') == 'TimeoutError', f'Unexpected result: {result!r}'
        assert result.get('found') is None, (
            f'A backend failure must NOT masquerade as found:False: {result!r}'
        )
        assert 'tombstone' not in result, (
            f'A backend failure must not carry a tombstone answer: {result!r}'
        )

    @pytest.mark.asyncio
    async def test_tool_registered(self):
        """The get_memory_by_id tool is registered on the MCP server."""
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)

        tool_names = [t.name for t in await server.list_tools()]
        assert 'get_memory_by_id' in tool_names, (
            f'get_memory_by_id not registered; tools: {tool_names!r}'
        )


# ---------------------------------------------------------------------------
# Task 3129 (leaf δ) — grouped reads on get_memory_by_id are ADDITIVE ONLY.
#
# This tool is an exact-point-id reader whose contract is load-bearing for
# reconciliation/citation_verifier.py, recon stage 1 and
# server/recon_report.py::cite_memory.  Swapping a CHILD's body for its
# parent's would silently verify a citation against different text — worse
# than the ungrouped state.  So a child keeps its own record and merely GAINS
# a grouped.parent block; upward *replacement* happens only in `search`.
# ---------------------------------------------------------------------------

_CANONICAL_ID = '11111111-1111-4111-8111-111111111111'
_AMEND_ID = '22222222-2222-4222-8222-222222222221'


def _grouped_read_service(record: dict, *, total: int = 5) -> AsyncMock:
    rows = [
        {
            'id': _AMEND_ID,
            'created_at': '2026-08-01T00:00:00+00:00',
            'metadata': {'data': 'a correction', 'parent_id': _CANONICAL_ID, 'kind': 'amendment'},
        },
        {
            'id': '22222222-2222-4222-8222-222222222222',
            'created_at': '2026-08-02T00:00:00+00:00',
            'metadata': {'data': 'another correction', 'parent_id': _CANONICAL_ID, 'kind': 'amendment'},
        },
    ]

    def _count(*, project_id: str, filters: dict):
        if filters.get('parent_id') != _CANONICAL_ID:
            return 0
        return {'sighting': 3, 'amendment': 2}.get(filters.get('kind', ''), total)

    def _scroll(*, project_id: str, filters: dict, limit: int = 1000):
        return rows[:limit]

    service = AsyncMock()
    service.get_memory_by_id = AsyncMock(return_value=record)
    service.count_memories_by_metadata = AsyncMock(side_effect=_count)
    service.get_memories_by_metadata = AsyncMock(side_effect=_scroll)
    return service


class TestGetMemoryByIdGroupedReads:
    @pytest.mark.asyncio
    async def test_canonical_gains_a_grouped_block(self):
        """A parented canonical keeps its own record and GAINS the group."""
        record = {
            'id': _CANONICAL_ID,
            'content': 'the canonical claim',
            'metadata': {'data': 'the canonical claim', 'kind': 'canonical'},
        }
        server = create_mcp_server(_grouped_read_service(record))

        result = await server._tool_manager.call_tool(
            'get_memory_by_id',
            {'project_id': _PROJECT_ID, 'memory_id': _CANONICAL_ID},
        )

        assert result['found'] is True
        assert result['content'] == 'the canonical claim'
        assert result['metadata'] == record['metadata']
        grouped = result['grouped']
        assert len(grouped['amendments']) == 2, f'Expected 2 digests, got {grouped!r}'
        assert grouped['sighting_count'] == 3, (
            f'Expected sighting_count 3, got {grouped!r}. RED: the tool does not group yet.'
        )

    @pytest.mark.asyncio
    async def test_child_keeps_its_own_record_and_gains_grouped_parent(self):
        """A CHILD id must never be answered with its parent's text."""
        record = {
            'id': _AMEND_ID,
            'content': 'a correction',
            'metadata': {
                'data': 'a correction',
                'parent_id': _CANONICAL_ID,
                'kind': 'amendment',
            },
        }
        server = create_mcp_server(_grouped_read_service(record))

        result = await server._tool_manager.call_tool(
            'get_memory_by_id',
            {'project_id': _PROJECT_ID, 'memory_id': _AMEND_ID},
        )

        assert result['memory_id'] == _AMEND_ID, 'The child keeps its OWN id'
        assert result['content'] == 'a correction', (
            f"The child keeps its OWN content, never the parent's, got {result!r}"
        )
        assert result['metadata'] == record['metadata'], 'The child keeps its OWN metadata'
        parent = result['grouped']['parent']
        assert parent['id'] == _CANONICAL_ID, f'grouped.parent must name the parent, got {parent!r}'
        assert parent['sighting_count'] == 3
        assert len(parent['amendments']) == 2

    @pytest.mark.asyncio
    async def test_childless_canonical_response_is_unchanged(self):
        """Zero-child corpus: no grouped key at all — byte-identical to today."""
        record = {
            'id': _CANONICAL_ID,
            'content': 'a plain memory',
            'metadata': {'data': 'a plain memory'},
        }
        service = _grouped_read_service(record, total=0)
        service.count_memories_by_metadata = AsyncMock(return_value=0)
        server = create_mcp_server(service)

        result = await server._tool_manager.call_tool(
            'get_memory_by_id',
            {'project_id': _PROJECT_ID, 'memory_id': _CANONICAL_ID},
        )

        assert result == {
            'found': True,
            'memory_id': _CANONICAL_ID,
            'project_id': _PROJECT_ID,
            'content': 'a plain memory',
            'metadata': {'data': 'a plain memory'},
        }, f'A childless hit must be byte-identical to today, got {result!r}'

    @pytest.mark.asyncio
    async def test_grouping_never_runs_on_the_miss_branch(self):
        """found:False (and its tombstone path) stays untouched."""
        service = AsyncMock()
        service.get_memory_by_id = AsyncMock(return_value=None)
        service.get_mem0_deletion_tombstone = AsyncMock(return_value=None)
        service.count_memories_by_metadata = AsyncMock(return_value=5)
        server = create_mcp_server(service)

        result = await server._tool_manager.call_tool(
            'get_memory_by_id',
            {'project_id': _PROJECT_ID, 'memory_id': _CANONICAL_ID},
        )

        assert result == {
            'found': False,
            'memory_id': _CANONICAL_ID,
            'project_id': _PROJECT_ID,
        }, f'The genuine-miss shape must be unchanged, got {result!r}'
        assert service.count_memories_by_metadata.await_count == 0, (
            'Grouping must never run on the miss branch'
        )

    @pytest.mark.asyncio
    async def test_grouping_failure_cannot_turn_a_hit_into_an_error(self):
        """One-way information only: a grouping fault leaves found:True intact."""
        record = {
            'id': _CANONICAL_ID,
            'content': 'the canonical claim',
            'metadata': {'data': 'the canonical claim'},
        }
        service = AsyncMock()
        service.get_memory_by_id = AsyncMock(return_value=record)
        service.count_memories_by_metadata = AsyncMock(side_effect=RuntimeError('boom'))
        server = create_mcp_server(service)

        result = await server._tool_manager.call_tool(
            'get_memory_by_id',
            {'project_id': _PROJECT_ID, 'memory_id': _CANONICAL_ID},
        )

        assert result['found'] is True, (
            f'A grouping fault must never convert a correct hit into an error, got {result!r}'
        )
        assert result['content'] == 'the canonical claim'

    @pytest.mark.asyncio
    async def test_child_with_a_dangling_parent_is_marked_unresolved(self):
        """A parent_id nothing resolves must never be reported as a real parent.

        `search` already stamps parent_unresolved for exactly this case; the
        point-id surface would otherwise hand back grouped.parent.id as though
        the pointer were live, because a childless parent and a missing parent
        are both `None` from build_grouped_document alone.
        """
        record = {
            'id': _AMEND_ID,
            'content': 'a correction',
            'metadata': {
                'data': 'a correction',
                'parent_id': _CANONICAL_ID,
                'kind': 'amendment',
            },
        }
        service = _grouped_read_service(record)
        # The child's own read hits; the parent pointer resolves to nothing.
        service.get_memory_by_id = AsyncMock(
            side_effect=lambda *, project_id, memory_id: record if memory_id == _AMEND_ID else None
        )
        server = create_mcp_server(service)

        result = await server._tool_manager.call_tool(
            'get_memory_by_id',
            {'project_id': _PROJECT_ID, 'memory_id': _AMEND_ID},
        )

        assert result['found'] is True, f'The CHILD still resolves and is returned, got {result!r}'
        assert result['content'] == 'a correction', 'The child keeps its own body'
        assert result['grouped']['parent_unresolved'] is True, (
            'A dangling parent_id must be marked, not silently presented as a real '
            f'parent, got {result["grouped"]!r}'
        )
        assert result['grouped']['parent']['id'] == _CANONICAL_ID, (
            'The pointer is still reported — marked, not hidden'
        )
