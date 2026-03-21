"""Tests for reindex maintenance: GraphitiBackend stale-embedding queries and ReindexManager."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.backends.graphiti_client import (
    EdgeNotFoundError,
    GraphitiBackend,
    NodeNotFoundError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend(config) -> GraphitiBackend:
    """Build a GraphitiBackend with a mock client attached."""
    backend = GraphitiBackend(config)
    # inject a mock Graphiti client
    mock_client = MagicMock()
    backend.client = mock_client
    return backend


def _make_graph_mock(rows: list[list]) -> MagicMock:
    """Return a mock whose .query() is an AsyncMock returning rows."""
    result = MagicMock()
    result.result_set = rows
    graph_mock = MagicMock()
    graph_mock.query = AsyncMock(return_value=result)
    return graph_mock


# ---------------------------------------------------------------------------
# step-1: query_stale_node_embeddings / query_stale_edge_embeddings
# ---------------------------------------------------------------------------

class TestQueryStaleNodeEmbeddings:
    """GraphitiBackend.query_stale_node_embeddings(expected_dim) returns stale nodes."""

    @pytest.mark.asyncio
    async def test_returns_stale_nodes(self, mock_config):
        backend = _make_backend(mock_config)
        rows = [
            ['uuid-1', 'Node A', 1024],
            ['uuid-2', 'Node B', 512],
        ]
        graph = _make_graph_mock(rows)
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.query_stale_node_embeddings(expected_dim=1536)
        assert len(result) == 2
        assert result[0] == ('uuid-1', 'Node A', 1024)
        assert result[1] == ('uuid-2', 'Node B', 512)

    @pytest.mark.asyncio
    async def test_returns_empty_when_all_match(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.query_stale_node_embeddings(expected_dim=1536)
        assert result == []

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)  # client is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.query_stale_node_embeddings(expected_dim=1536)

    @pytest.mark.asyncio
    async def test_passes_dim_param_to_query(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.query_stale_node_embeddings(expected_dim=768)
        call_kwargs = graph.query.call_args
        assert call_kwargs is not None
        # check dim is passed as a parameter
        args, kwargs = call_kwargs
        cypher_args = args[1] if len(args) > 1 else kwargs.get('params', {})
        assert cypher_args.get('dim') == 768


class TestQueryStaleEdgeEmbeddings:
    """GraphitiBackend.query_stale_edge_embeddings(expected_dim) returns stale edges."""

    @pytest.mark.asyncio
    async def test_returns_stale_edges(self, mock_config):
        backend = _make_backend(mock_config)
        rows = [
            ['edge-uuid-1', 'edge name', 1024],
        ]
        graph = _make_graph_mock(rows)
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.query_stale_edge_embeddings(expected_dim=1536)
        assert len(result) == 1
        assert result[0] == ('edge-uuid-1', 'edge name', 1024)

    @pytest.mark.asyncio
    async def test_returns_empty_when_all_match(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.query_stale_edge_embeddings(expected_dim=1536)
        assert result == []

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.query_stale_edge_embeddings(expected_dim=1536)

    @pytest.mark.asyncio
    async def test_passes_dim_param_to_query(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.query_stale_edge_embeddings(expected_dim=768)
        call_kwargs = graph.query.call_args
        assert call_kwargs is not None
        args, kwargs = call_kwargs
        cypher_args = args[1] if len(args) > 1 else kwargs.get('params', {})
        assert cypher_args.get('dim') == 768


# ---------------------------------------------------------------------------
# step-3: get_node_text / get_edge_text
# ---------------------------------------------------------------------------

class TestGetNodeText:
    """GraphitiBackend.get_node_text(uuid) returns (name, summary) tuple."""

    @pytest.mark.asyncio
    async def test_returns_name_and_summary(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([['Alice', 'Alice is a person']])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            name, summary = await backend.get_node_text('uuid-1')
        assert name == 'Alice'
        assert summary == 'Alice is a person'

    @pytest.mark.asyncio
    async def test_raises_node_not_found(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])  # empty result
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            with pytest.raises(NodeNotFoundError):
                await backend.get_node_text('missing-uuid')

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.get_node_text('uuid-1')

    @pytest.mark.asyncio
    async def test_passes_uuid_to_query(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([['Node', 'Summary']])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.get_node_text('specific-uuid')
        call_kwargs = graph.query.call_args
        args, kwargs = call_kwargs
        params = args[1] if len(args) > 1 else kwargs.get('params', {})
        assert params.get('uuid') == 'specific-uuid'


class TestGetEdgeText:
    """GraphitiBackend.get_edge_text(uuid) returns (name, fact) tuple."""

    @pytest.mark.asyncio
    async def test_returns_name_and_fact(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([['edge-name', 'Some fact about the edge']])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            name, fact = await backend.get_edge_text('edge-uuid-1')
        assert name == 'edge-name'
        assert fact == 'Some fact about the edge'

    @pytest.mark.asyncio
    async def test_raises_edge_not_found(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            with pytest.raises(EdgeNotFoundError):
                await backend.get_edge_text('missing-edge-uuid')

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.get_edge_text('edge-uuid-1')

    @pytest.mark.asyncio
    async def test_passes_uuid_to_query(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([['name', 'fact']])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.get_edge_text('specific-edge-uuid')
        call_kwargs = graph.query.call_args
        args, kwargs = call_kwargs
        params = args[1] if len(args) > 1 else kwargs.get('params', {})
        assert params.get('uuid') == 'specific-edge-uuid'


# ---------------------------------------------------------------------------
# step-5: update_node_embedding / update_edge_embedding
# ---------------------------------------------------------------------------

class TestUpdateNodeEmbedding:
    """GraphitiBackend.update_node_embedding(uuid, embedding) stores new vector."""

    @pytest.mark.asyncio
    async def test_calls_set_with_embedding(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        embedding = [0.1] * 1536
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.update_node_embedding('uuid-1', embedding)
        assert graph.query.called

    @pytest.mark.asyncio
    async def test_passes_uuid_and_embedding_to_query(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        embedding = [0.5] * 128
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.update_node_embedding('my-uuid', embedding)
        call_args = graph.query.call_args
        args, kwargs = call_args
        params = args[1] if len(args) > 1 else kwargs.get('params', {})
        assert params.get('uuid') == 'my-uuid'
        assert params.get('embedding') == embedding

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.update_node_embedding('uuid-1', [0.1] * 10)


class TestUpdateEdgeEmbedding:
    """GraphitiBackend.update_edge_embedding(uuid, embedding) stores new vector."""

    @pytest.mark.asyncio
    async def test_calls_set_with_embedding(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        embedding = [0.2] * 1536
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.update_edge_embedding('edge-uuid', embedding)
        assert graph.query.called

    @pytest.mark.asyncio
    async def test_passes_uuid_and_embedding_to_query(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        embedding = [0.7] * 64
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.update_edge_embedding('edge-uuid-99', embedding)
        call_args = graph.query.call_args
        args, kwargs = call_args
        params = args[1] if len(args) > 1 else kwargs.get('params', {})
        assert params.get('uuid') == 'edge-uuid-99'
        assert params.get('embedding') == embedding

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.update_edge_embedding('edge-uuid', [0.1] * 10)


# ---------------------------------------------------------------------------
# step-7: list_indices / drop_index
# ---------------------------------------------------------------------------

class TestListIndices:
    """GraphitiBackend.list_indices() returns parsed index records."""

    @pytest.mark.asyncio
    async def test_returns_index_list(self, mock_config):
        backend = _make_backend(mock_config)
        # FalkorDB index records: [label, property, type, entity_type]
        rows = [
            ['Entity', 'name_embedding', 'VECTOR', 'NODE'],
            ['Entity', 'name', 'FULLTEXT', 'NODE'],
        ]
        graph = _make_graph_mock(rows)
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.list_indices()
        assert len(result) == 2
        assert result[0]['label'] == 'Entity'
        assert result[0]['field'] == 'name_embedding'
        assert result[0]['type'] == 'VECTOR'
        assert result[0]['entity_type'] == 'NODE'

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_indices(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.list_indices()
        assert result == []

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.list_indices()


class TestDropIndex:
    """GraphitiBackend.drop_index(label, field) generates correct DROP Cypher."""

    @pytest.mark.asyncio
    async def test_drop_index_calls_query(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.drop_index('Entity', 'name_embedding')
        assert graph.query.called

    @pytest.mark.asyncio
    async def test_drop_index_cypher_contains_label_and_field(self, mock_config):
        backend = _make_backend(mock_config)
        graph = _make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.drop_index('MyLabel', 'my_field')
        call_args = graph.query.call_args
        cypher = call_args[0][0]
        assert 'MyLabel' in cypher
        assert 'my_field' in cypher

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.drop_index('Entity', 'name_embedding')
