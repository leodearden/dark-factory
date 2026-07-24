"""Tests for reassign_edge across backend, service, and MCP tool.

reassign_edge re-points ONE existing RELATES_TO edge's endpoint (source or
target) to a different Entity node, LOSSLESSLY — preserving the edge uuid,
fact text, fact_embedding, valid_at/invalid_at/expired_at, created_at,
group_id, and episode links — then refreshes the two affected node summaries.
It is the single-edge, uuid-targeted sibling of redirect_node_edges (the
bulk merge-time endpoint mover).

Covers (layered like test_merge_entities.py):
- GraphitiBackend.reassign_edge()  — validation, endpoint-move Cypher, refresh
- A skipif-gated live-FalkorDB acceptance test (real topology move + preservation)
- MemoryService.reassign_edge()    — delegation, journaling, event emission
- MCP tool reassign_edge in tools.py
- DISALLOW_MEMORY_WRITES list in cli_stage_runner.py
"""
from __future__ import annotations

import contextlib
import os
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from _fm_helpers import extract_cypher, extract_params
from falkordb import FalkorDB as _SyncFalkorDB
from falkordb.asyncio import FalkorDB
from graphiti_core.errors import EdgeNotFoundError

from fused_memory.backends.graphiti_client import (
    GraphitiBackend,
    NodeNotFoundError,
    _MultiTenantFalkorDriver,
)


# ---------------------------------------------------------------------------
# step-1: GraphitiBackend.reassign_edge — validation slice
# ---------------------------------------------------------------------------

class TestReassignEdgeBackendValidation:
    """GraphitiBackend.reassign_edge validates which_end, the target edge, the
    new endpoint node, and the initialized state before performing any move."""

    @pytest.mark.asyncio
    async def test_invalid_which_end_raises_value_error(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """which_end must be 'source' or 'target' — anything else raises ValueError."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        with pytest.raises(ValueError, match='which_end'):
            await backend.reassign_edge(
                'edge-uuid', 'new-endpoint-uuid', which_end='sideways', group_id='test',
            )

    @pytest.mark.asyncio
    async def test_missing_edge_raises_edge_not_found(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """When the topology read returns no rows, the edge does not exist."""
        backend = make_backend(mock_config)
        # ro_query returns empty -> topology read finds no edge.
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        with pytest.raises(EdgeNotFoundError):
            await backend.reassign_edge(
                'missing-edge', 'new-endpoint-uuid', which_end='source', group_id='test',
            )

    @pytest.mark.asyncio
    async def test_missing_new_endpoint_raises_node_not_found(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """Edge exists (topology read returns rows) but the new endpoint node
        does not -> NodeNotFoundError."""
        backend = make_backend(mock_config)

        async def router(cypher, params=None):
            result = MagicMock()
            if 'RELATES_TO' in cypher:
                # Topology endpoint read — edge exists (source, target).
                result.result_set = [['src-uuid', 'tgt-uuid']]
            else:
                # New endpoint node existence read — absent.
                result.result_set = []
            return result

        graph = make_graph_mock([])
        graph.ro_query = AsyncMock(side_effect=router)
        backend._driver._get_graph = MagicMock(return_value=graph)
        with pytest.raises(NodeNotFoundError):
            await backend.reassign_edge(
                'edge-uuid', 'missing-endpoint', which_end='source', group_id='test',
            )

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when the backend is not initialized (driver is None)."""
        backend = GraphitiBackend(mock_config)  # client/_driver are None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.reassign_edge(
                'edge-uuid', 'new-endpoint-uuid', which_end='source', group_id='test',
            )
