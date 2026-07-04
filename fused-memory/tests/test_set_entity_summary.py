"""Tests for set_entity_summary across backend, service, and MCP tool layers.

set_entity_summary is a direct summary-overwrite escape hatch: it writes an
explicit string verbatim (including '' to clear), bypassing edge-derivation
entirely.  This is distinct from refresh_entity_summary, which regenerates
the summary from the current valid-edge set — see test_refresh_entity_summary.py.

Covers:
- GraphitiBackend.set_entity_summary()
- MemoryService.set_entity_summary()
- MCP tool set_entity_summary in tools.py
- DISALLOW_MEMORY_WRITES list in cli_stage_runner.py
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend, NodeNotFoundError

# ---------------------------------------------------------------------------
# step-1: GraphitiBackend.set_entity_summary
# ---------------------------------------------------------------------------


class TestGraphitiBackendSetEntitySummary:
    """GraphitiBackend.set_entity_summary(node_uuid, summary, *, group_id) overwrites
    an Entity node's summary verbatim, bypassing edge-derivation entirely."""

    @pytest.mark.asyncio
    async def test_returns_result_dict(self, mock_config, make_backend):
        """Returns dict with uuid, name, old_summary, new_summary; new_summary is the
        exact string passed in (not an edge-derived join)."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('Alice', 'stale narrative text'))
        backend.update_node_summary = AsyncMock()
        backend.get_valid_edges_for_node = AsyncMock()

        result = await backend.set_entity_summary('node-uuid-1', 'Corrected summary.', group_id='test')

        assert result == {
            'uuid': 'node-uuid-1',
            'name': 'Alice',
            'old_summary': 'stale narrative text',
            'new_summary': 'Corrected summary.',
        }

    @pytest.mark.asyncio
    async def test_calls_update_node_summary_with_explicit_text(self, mock_config, make_backend):
        """Calls update_node_summary(node_uuid, summary, group_id=...) exactly once with
        the explicit text — and never touches get_valid_edges_for_node (no edge-derived join)."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('Alice', 'old'))
        backend.update_node_summary = AsyncMock()
        backend.get_valid_edges_for_node = AsyncMock()

        await backend.set_entity_summary('node-uuid-1', 'Exact text.', group_id='test')

        backend.update_node_summary.assert_awaited_once_with(
            'node-uuid-1', 'Exact text.', group_id='test'
        )
        backend.get_valid_edges_for_node.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_string_summary_forwarded_verbatim(self, mock_config, make_backend):
        """An empty-string summary is accepted and forwarded verbatim — the clear case."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(return_value=('Alice', 'stale text to clear'))
        backend.update_node_summary = AsyncMock()
        backend.get_valid_edges_for_node = AsyncMock()

        result = await backend.set_entity_summary('node-uuid-1', '', group_id='test')

        assert result['new_summary'] == ''
        backend.update_node_summary.assert_awaited_once_with('node-uuid-1', '', group_id='test')
        backend.get_valid_edges_for_node.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_node_not_found_propagates(self, mock_config, make_backend):
        """NodeNotFoundError from get_node_text propagates — a missing node is not a
        silent no-op."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(side_effect=NodeNotFoundError('Entity node not found: missing-uuid'))
        backend.update_node_summary = AsyncMock()

        with pytest.raises(NodeNotFoundError):
            await backend.set_entity_summary('missing-uuid', 'text', group_id='test')

        backend.update_node_summary.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when backend has no driver (not initialized)."""
        backend = GraphitiBackend(mock_config)  # _driver is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.set_entity_summary('node-uuid-1', 'text', group_id='test')
