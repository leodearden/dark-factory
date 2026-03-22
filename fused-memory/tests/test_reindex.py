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
        with (
            patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target),
            pytest.raises(NodeNotFoundError),
        ):
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
        with (
            patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target),
            pytest.raises(EdgeNotFoundError),
        ):
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


# ---------------------------------------------------------------------------
# step-1 (task-52): GraphitiBackend.drop_vector_indices()
# ---------------------------------------------------------------------------

class TestDropVectorIndices:
    """GraphitiBackend.drop_vector_indices() drops only VECTOR-type indices."""

    @pytest.mark.asyncio
    async def test_drops_only_vector_type_indices(self, mock_config):
        """Calls drop_index for VECTOR indices only, not FULLTEXT/RANGE."""
        backend = _make_backend(mock_config)

        indices = [
            {'label': 'Entity', 'field': 'name_embedding', 'type': 'VECTOR', 'entity_type': 'NODE'},
            {'label': 'Entity', 'field': 'name', 'type': 'FULLTEXT', 'entity_type': 'NODE'},
            {'label': 'RELATES_TO', 'field': 'fact_embedding', 'type': 'VECTOR', 'entity_type': 'RELATIONSHIP'},
        ]
        backend.list_indices = AsyncMock(return_value=indices)
        backend.drop_index = AsyncMock()

        await backend.drop_vector_indices()

        assert backend.drop_index.call_count == 2
        calls = backend.drop_index.call_args_list
        called_pairs = [(c[0][0], c[0][1]) for c in calls]
        assert ('Entity', 'name_embedding') in called_pairs
        assert ('RELATES_TO', 'fact_embedding') in called_pairs

    @pytest.mark.asyncio
    async def test_returns_list_of_dropped_indices(self, mock_config):
        """Returns list of dicts with 'label' and 'field' for each dropped index."""
        backend = _make_backend(mock_config)

        indices = [
            {'label': 'Entity', 'field': 'name_embedding', 'type': 'VECTOR', 'entity_type': 'NODE'},
            {'label': 'Entity', 'field': 'name', 'type': 'FULLTEXT', 'entity_type': 'NODE'},
        ]
        backend.list_indices = AsyncMock(return_value=indices)
        backend.drop_index = AsyncMock()

        result = await backend.drop_vector_indices()

        assert len(result) == 1
        assert result[0] == {'label': 'Entity', 'field': 'name_embedding'}

    @pytest.mark.asyncio
    async def test_no_op_when_no_vector_indices(self, mock_config):
        """When no VECTOR indices exist, drop_index not called and returns []."""
        backend = _make_backend(mock_config)

        indices = [
            {'label': 'Entity', 'field': 'name', 'type': 'FULLTEXT', 'entity_type': 'NODE'},
            {'label': 'Entity', 'field': 'created_at', 'type': 'RANGE', 'entity_type': 'NODE'},
        ]
        backend.list_indices = AsyncMock(return_value=indices)
        backend.drop_index = AsyncMock()

        result = await backend.drop_vector_indices()

        backend.drop_index.assert_not_called()
        assert result == []


# ---------------------------------------------------------------------------
# step-9: ReindexManager
# ---------------------------------------------------------------------------

class TestReindexManager:
    """ReindexManager.reindex() orchestrates stale-embedding detection and re-embedding."""

    def _make_manager(self, backend, embedder, expected_dim=1536):
        from fused_memory.maintenance.reindex import ReindexManager
        return ReindexManager(backend=backend, embedder=embedder, expected_dim=expected_dim)

    @pytest.mark.asyncio
    async def test_reindex_processes_stale_nodes_and_edges(self):
        """Finds 2 stale nodes and 1 stale edge, re-embeds each, updates embeddings."""
        from fused_memory.maintenance.reindex import ReindexManager

        backend = MagicMock()
        backend.query_stale_node_embeddings = AsyncMock(return_value=[
            ('node-uuid-1', 'Alice', 1024),
            ('node-uuid-2', 'Bob', 1024),
        ])
        backend.query_stale_edge_embeddings = AsyncMock(return_value=[
            ('edge-uuid-1', 'knows', 1024),
        ])
        backend.get_node_text = AsyncMock(side_effect=[
            ('Alice', 'Alice is a person'),
            ('Bob', 'Bob is a developer'),
        ])
        backend.get_edge_text = AsyncMock(return_value=('knows', 'Alice knows Bob'))
        backend.update_node_embedding = AsyncMock()
        backend.update_edge_embedding = AsyncMock()

        embedder = MagicMock()
        new_embedding = [0.1] * 1536
        embedder.create = AsyncMock(return_value=new_embedding)

        manager = ReindexManager(backend=backend, embedder=embedder, expected_dim=1536)
        result = await manager.reindex()

        assert result.nodes_updated == 2
        assert result.edges_updated == 1
        assert result.errors == 0
        assert backend.get_node_text.call_count == 2
        assert backend.get_edge_text.call_count == 1
        assert backend.update_node_embedding.call_count == 2
        assert backend.update_edge_embedding.call_count == 1
        assert embedder.create.call_count == 3

    @pytest.mark.asyncio
    async def test_reindex_with_no_stale_items_returns_zeros(self):
        """reindex() with no stale items returns ReindexResult(0, 0, 0) immediately."""
        from fused_memory.maintenance.reindex import ReindexManager

        backend = MagicMock()
        backend.query_stale_node_embeddings = AsyncMock(return_value=[])
        backend.query_stale_edge_embeddings = AsyncMock(return_value=[])
        embedder = MagicMock()

        manager = ReindexManager(backend=backend, embedder=embedder, expected_dim=1536)
        result = await manager.reindex()

        assert result.nodes_updated == 0
        assert result.edges_updated == 0
        assert result.errors == 0
        embedder.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_reindex_passes_expected_dim_to_queries(self):
        """query_stale_node/edge_embeddings receive the expected_dim from config."""
        from fused_memory.maintenance.reindex import ReindexManager

        backend = MagicMock()
        backend.query_stale_node_embeddings = AsyncMock(return_value=[])
        backend.query_stale_edge_embeddings = AsyncMock(return_value=[])
        embedder = MagicMock()

        manager = ReindexManager(backend=backend, embedder=embedder, expected_dim=768)
        await manager.reindex()

        backend.query_stale_node_embeddings.assert_called_once_with(expected_dim=768)
        backend.query_stale_edge_embeddings.assert_called_once_with(expected_dim=768)

    @pytest.mark.asyncio
    async def test_reindex_node_text_combined_for_embedding(self):
        """Embedder receives name + space + summary as text."""
        from fused_memory.maintenance.reindex import ReindexManager

        backend = MagicMock()
        backend.query_stale_node_embeddings = AsyncMock(return_value=[
            ('node-uuid-1', 'Node', 1024),
        ])
        backend.query_stale_edge_embeddings = AsyncMock(return_value=[])
        backend.get_node_text = AsyncMock(return_value=('My Node', 'A summary'))
        backend.update_node_embedding = AsyncMock()

        embedder = MagicMock()
        embedder.create = AsyncMock(return_value=[0.1] * 1536)

        manager = ReindexManager(backend=backend, embedder=embedder, expected_dim=1536)
        await manager.reindex()

        embed_call_text = embedder.create.call_args[0][0]
        assert 'My Node' in embed_call_text
        assert 'A summary' in embed_call_text

    @pytest.mark.asyncio
    async def test_reindex_edge_text_combined_for_embedding(self):
        """Embedder receives name + space + fact as text for edges."""
        from fused_memory.maintenance.reindex import ReindexManager

        backend = MagicMock()
        backend.query_stale_node_embeddings = AsyncMock(return_value=[])
        backend.query_stale_edge_embeddings = AsyncMock(return_value=[
            ('edge-uuid-1', 'knows', 1024),
        ])
        backend.get_edge_text = AsyncMock(return_value=('knows', 'Alice knows Bob'))
        backend.update_edge_embedding = AsyncMock()

        embedder = MagicMock()
        embedder.create = AsyncMock(return_value=[0.1] * 1536)

        manager = ReindexManager(backend=backend, embedder=embedder, expected_dim=1536)
        await manager.reindex()

        embed_call_text = embedder.create.call_args[0][0]
        assert 'knows' in embed_call_text
        assert 'Alice knows Bob' in embed_call_text


# ---------------------------------------------------------------------------
# step-11: ReindexManager.reindex_and_replay()
# ---------------------------------------------------------------------------

class TestReindexAndReplay:
    """ReindexManager.reindex_and_replay() calls reindex() then replays the dead-letter queue."""

    @pytest.mark.asyncio
    async def test_reindex_called_before_replay(self):
        """reindex() is called before replay_dead()."""
        from fused_memory.maintenance.reindex import ReindexManager, ReindexResult

        call_order: list[str] = []

        backend = MagicMock()
        embedder = MagicMock()

        manager = ReindexManager(backend=backend, embedder=embedder, expected_dim=1536)

        async def fake_reindex():
            call_order.append('reindex')
            return ReindexResult(1, 0, 0)

        async def fake_replay(group_id=None):
            call_order.append('replay')
            return 3

        manager.reindex = fake_reindex

        mock_queue = MagicMock()
        mock_queue.replay_dead = AsyncMock(side_effect=fake_replay)

        await manager.reindex_and_replay(mock_queue)

        assert call_order == ['reindex', 'replay']

    @pytest.mark.asyncio
    async def test_returns_combined_result(self):
        """Returns dict with reindex_result and replay_count."""
        from fused_memory.maintenance.reindex import ReindexManager, ReindexResult

        backend = MagicMock()
        embedder = MagicMock()

        manager = ReindexManager(backend=backend, embedder=embedder, expected_dim=1536)
        reindex_result = ReindexResult(nodes_updated=3, edges_updated=2, errors=0)
        manager.reindex = AsyncMock(return_value=reindex_result)

        mock_queue = MagicMock()
        mock_queue.replay_dead = AsyncMock(return_value=5)

        result = await manager.reindex_and_replay(mock_queue)

        assert result['reindex_result'] is reindex_result
        assert result['replay_count'] == 5

    @pytest.mark.asyncio
    async def test_passes_group_id_to_replay(self):
        """group_id parameter is forwarded to durable_queue.replay_dead()."""
        from fused_memory.maintenance.reindex import ReindexManager, ReindexResult

        backend = MagicMock()
        embedder = MagicMock()

        manager = ReindexManager(backend=backend, embedder=embedder, expected_dim=1536)
        manager.reindex = AsyncMock(return_value=ReindexResult())

        mock_queue = MagicMock()
        mock_queue.replay_dead = AsyncMock(return_value=0)

        await manager.reindex_and_replay(mock_queue, group_id='test-group')

        mock_queue.replay_dead.assert_awaited_once_with(group_id='test-group')

    @pytest.mark.asyncio
    async def test_default_group_id_is_none(self):
        """Default group_id=None replays all dead-letter items."""
        from fused_memory.maintenance.reindex import ReindexManager, ReindexResult

        backend = MagicMock()
        embedder = MagicMock()

        manager = ReindexManager(backend=backend, embedder=embedder, expected_dim=1536)
        manager.reindex = AsyncMock(return_value=ReindexResult())

        mock_queue = MagicMock()
        mock_queue.replay_dead = AsyncMock(return_value=0)

        await manager.reindex_and_replay(mock_queue)

        mock_queue.replay_dead.assert_awaited_once_with(group_id=None)

    @pytest.mark.asyncio
    async def test_drops_indices_before_reindex_when_requested(self):
        """drop_vector_indices() called before reindex() when drop_indices=True."""
        from fused_memory.maintenance.reindex import ReindexManager, ReindexResult

        call_order: list[str] = []

        backend = MagicMock()
        embedder = MagicMock()

        manager = ReindexManager(backend=backend, embedder=embedder, expected_dim=1536)

        async def fake_drop():
            call_order.append('drop')
            return [{'label': 'Entity', 'field': 'name_embedding'}]

        async def fake_reindex():
            call_order.append('reindex')
            return ReindexResult(1, 0, 0)

        async def fake_replay(group_id=None):
            call_order.append('replay')
            return 3

        manager.backend.drop_vector_indices = fake_drop
        manager.reindex = fake_reindex

        mock_queue = MagicMock()
        mock_queue.replay_dead = AsyncMock(side_effect=fake_replay)

        await manager.reindex_and_replay(mock_queue, drop_indices=True)

        assert call_order == ['drop', 'reindex', 'replay']

    @pytest.mark.asyncio
    async def test_skips_drop_when_drop_indices_false(self):
        """drop_vector_indices() NOT called when drop_indices=False."""
        from fused_memory.maintenance.reindex import ReindexManager, ReindexResult

        backend = MagicMock()
        backend.drop_vector_indices = AsyncMock(return_value=[])
        embedder = MagicMock()

        manager = ReindexManager(backend=backend, embedder=embedder, expected_dim=1536)
        manager.reindex = AsyncMock(return_value=ReindexResult())

        mock_queue = MagicMock()
        mock_queue.replay_dead = AsyncMock(return_value=0)

        await manager.reindex_and_replay(mock_queue, drop_indices=False)

        backend.drop_vector_indices.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_drop_indices_is_false(self):
        """drop_indices defaults to False (backward compatible)."""
        from fused_memory.maintenance.reindex import ReindexManager, ReindexResult

        backend = MagicMock()
        backend.drop_vector_indices = AsyncMock(return_value=[])
        embedder = MagicMock()

        manager = ReindexManager(backend=backend, embedder=embedder, expected_dim=1536)
        manager.reindex = AsyncMock(return_value=ReindexResult())

        mock_queue = MagicMock()
        mock_queue.replay_dead = AsyncMock(return_value=0)

        await manager.reindex_and_replay(mock_queue)  # no drop_indices arg

        backend.drop_vector_indices.assert_not_called()

    @pytest.mark.asyncio
    async def test_result_includes_indices_dropped(self):
        """Return dict includes 'indices_dropped' list from drop_vector_indices."""
        from fused_memory.maintenance.reindex import ReindexManager, ReindexResult

        dropped = [
            {'label': 'Entity', 'field': 'name_embedding'},
            {'label': 'RELATES_TO', 'field': 'fact_embedding'},
        ]
        backend = MagicMock()
        backend.drop_vector_indices = AsyncMock(return_value=dropped)
        embedder = MagicMock()

        manager = ReindexManager(backend=backend, embedder=embedder, expected_dim=1536)
        manager.reindex = AsyncMock(return_value=ReindexResult())

        mock_queue = MagicMock()
        mock_queue.replay_dead = AsyncMock(return_value=0)

        result = await manager.reindex_and_replay(mock_queue, drop_indices=True)

        assert result['indices_dropped'] == dropped

    @pytest.mark.asyncio
    async def test_result_indices_dropped_empty_when_skipped(self):
        """Return dict has 'indices_dropped': [] when drop_indices=False."""
        from fused_memory.maintenance.reindex import ReindexManager, ReindexResult

        backend = MagicMock()
        backend.drop_vector_indices = AsyncMock(return_value=[])
        embedder = MagicMock()

        manager = ReindexManager(backend=backend, embedder=embedder, expected_dim=1536)
        manager.reindex = AsyncMock(return_value=ReindexResult())

        mock_queue = MagicMock()
        mock_queue.replay_dead = AsyncMock(return_value=0)

        result = await manager.reindex_and_replay(mock_queue, drop_indices=False)

        assert result['indices_dropped'] == []


# ---------------------------------------------------------------------------
# step-13: run_reindex() CLI entrypoint
# ---------------------------------------------------------------------------

class TestRunReindex:
    """run_reindex() loads config, initializes MemoryService, runs reindex_and_replay, closes."""

    @pytest.mark.asyncio
    async def test_run_reindex_initializes_service_and_runs(self):
        """run_reindex() initializes the MemoryService and calls reindex_and_replay."""
        from fused_memory.maintenance.reindex import ReindexResult, run_reindex

        mock_config = MagicMock()
        mock_config.embedder.dimensions = 1536
        mock_config.embedder.providers.openai.api_key = 'test-key'
        mock_config.embedder.model = 'text-embedding-3-small'

        mock_service = AsyncMock()
        mock_service.durable_queue = MagicMock()

        mock_result = {'reindex_result': ReindexResult(2, 1, 0), 'replay_count': 5}

        with (
            patch('fused_memory.maintenance.reindex.FusedMemoryConfig', return_value=mock_config),
            patch('fused_memory.maintenance.reindex.MemoryService', return_value=mock_service),
            patch('fused_memory.maintenance.reindex.OpenAIEmbedder'),
            patch('fused_memory.maintenance.reindex.ReindexManager') as mock_mgr_cls,
        ):
            mock_mgr = MagicMock()
            mock_mgr.reindex_and_replay = AsyncMock(return_value=mock_result)
            mock_mgr_cls.return_value = mock_mgr

            result = await run_reindex()

        mock_service.initialize.assert_awaited_once()
        mock_mgr.reindex_and_replay.assert_awaited_once_with(
            mock_service.durable_queue, drop_indices=False
        )
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_run_reindex_closes_service_on_success(self):
        """run_reindex() calls service.close() in finally block after success."""
        from fused_memory.maintenance.reindex import ReindexResult, run_reindex

        mock_config = MagicMock()
        mock_config.embedder.dimensions = 1536
        mock_config.embedder.providers.openai.api_key = 'test-key'
        mock_config.embedder.model = 'text-embedding-3-small'

        mock_service = AsyncMock()
        mock_service.durable_queue = MagicMock()

        mock_result = {'reindex_result': ReindexResult(), 'replay_count': 0}

        with (
            patch('fused_memory.maintenance.reindex.FusedMemoryConfig', return_value=mock_config),
            patch('fused_memory.maintenance.reindex.MemoryService', return_value=mock_service),
            patch('fused_memory.maintenance.reindex.OpenAIEmbedder'),
            patch('fused_memory.maintenance.reindex.ReindexManager') as mock_mgr_cls,
        ):
            mock_mgr = MagicMock()
            mock_mgr.reindex_and_replay = AsyncMock(return_value=mock_result)
            mock_mgr_cls.return_value = mock_mgr

            await run_reindex()

        mock_service.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_reindex_closes_service_on_error(self):
        """run_reindex() calls service.close() in finally block even when an error occurs."""
        from fused_memory.maintenance.reindex import run_reindex

        mock_config = MagicMock()
        mock_config.embedder.dimensions = 1536
        mock_config.embedder.providers.openai.api_key = 'test-key'
        mock_config.embedder.model = 'text-embedding-3-small'

        mock_service = AsyncMock()
        mock_service.durable_queue = MagicMock()

        with (
            patch('fused_memory.maintenance.reindex.FusedMemoryConfig', return_value=mock_config),
            patch('fused_memory.maintenance.reindex.MemoryService', return_value=mock_service),
            patch('fused_memory.maintenance.reindex.OpenAIEmbedder'),
            patch('fused_memory.maintenance.reindex.ReindexManager') as mock_mgr_cls,
        ):
            mock_mgr = MagicMock()
            mock_mgr.reindex_and_replay = AsyncMock(
                side_effect=RuntimeError('embedding service unavailable')
            )
            mock_mgr_cls.return_value = mock_mgr

            with pytest.raises(RuntimeError, match='embedding service unavailable'):
                await run_reindex()

        mock_service.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_reindex_passes_drop_indices_true(self):
        """run_reindex(drop_indices=True) passes drop_indices=True to reindex_and_replay."""
        from fused_memory.maintenance.reindex import ReindexResult, run_reindex

        mock_config = MagicMock()
        mock_config.embedder.dimensions = 1536
        mock_config.embedder.providers.openai.api_key = 'test-key'
        mock_config.embedder.model = 'text-embedding-3-small'

        mock_service = AsyncMock()
        mock_service.durable_queue = MagicMock()

        mock_result = {'reindex_result': ReindexResult(), 'replay_count': 0, 'indices_dropped': []}

        with (
            patch('fused_memory.maintenance.reindex.FusedMemoryConfig', return_value=mock_config),
            patch('fused_memory.maintenance.reindex.MemoryService', return_value=mock_service),
            patch('fused_memory.maintenance.reindex.OpenAIEmbedder'),
            patch('fused_memory.maintenance.reindex.ReindexManager') as mock_mgr_cls,
        ):
            mock_mgr = MagicMock()
            mock_mgr.reindex_and_replay = AsyncMock(return_value=mock_result)
            mock_mgr_cls.return_value = mock_mgr

            await run_reindex(drop_indices=True)

        mock_mgr.reindex_and_replay.assert_awaited_once()
        call_kwargs = mock_mgr.reindex_and_replay.call_args[1]
        assert call_kwargs.get('drop_indices') is True

    @pytest.mark.asyncio
    async def test_run_reindex_default_drop_indices_false(self):
        """run_reindex() with no drop_indices arg passes drop_indices=False."""
        from fused_memory.maintenance.reindex import ReindexResult, run_reindex

        mock_config = MagicMock()
        mock_config.embedder.dimensions = 1536
        mock_config.embedder.providers.openai.api_key = 'test-key'
        mock_config.embedder.model = 'text-embedding-3-small'

        mock_service = AsyncMock()
        mock_service.durable_queue = MagicMock()

        mock_result = {'reindex_result': ReindexResult(), 'replay_count': 0, 'indices_dropped': []}

        with (
            patch('fused_memory.maintenance.reindex.FusedMemoryConfig', return_value=mock_config),
            patch('fused_memory.maintenance.reindex.MemoryService', return_value=mock_service),
            patch('fused_memory.maintenance.reindex.OpenAIEmbedder'),
            patch('fused_memory.maintenance.reindex.ReindexManager') as mock_mgr_cls,
        ):
            mock_mgr = MagicMock()
            mock_mgr.reindex_and_replay = AsyncMock(return_value=mock_result)
            mock_mgr_cls.return_value = mock_mgr

            await run_reindex()  # no drop_indices argument

        mock_mgr.reindex_and_replay.assert_awaited_once()
        call_kwargs = mock_mgr.reindex_and_replay.call_args[1]
        assert call_kwargs.get('drop_indices') is False

    @pytest.mark.asyncio
    async def test_run_reindex_creates_embedder_with_config_dimensions(self):
        """run_reindex() passes config.embedder.dimensions to OpenAIEmbedderConfig."""
        from fused_memory.maintenance.reindex import ReindexResult, run_reindex

        mock_config = MagicMock()
        mock_config.embedder.dimensions = 768
        mock_config.embedder.providers.openai.api_key = 'custom-key'
        mock_config.embedder.model = 'text-embedding-ada-002'

        mock_service = AsyncMock()
        mock_service.durable_queue = MagicMock()

        mock_result = {'reindex_result': ReindexResult(), 'replay_count': 0}

        with (
            patch('fused_memory.maintenance.reindex.FusedMemoryConfig', return_value=mock_config),
            patch('fused_memory.maintenance.reindex.MemoryService', return_value=mock_service),
            patch('fused_memory.maintenance.reindex.OpenAIEmbedderConfig') as mock_cfg_cls,
            patch('fused_memory.maintenance.reindex.OpenAIEmbedder'),
            patch('fused_memory.maintenance.reindex.ReindexManager') as mock_mgr_cls,
        ):
            mock_mgr = MagicMock()
            mock_mgr.reindex_and_replay = AsyncMock(return_value=mock_result)
            mock_mgr_cls.return_value = mock_mgr

            await run_reindex()

        # Verify OpenAIEmbedderConfig was constructed with the right dimension
        mock_cfg_cls.assert_called_once()
        call_kwargs = mock_cfg_cls.call_args[1]
        assert call_kwargs.get('embedding_dim') == 768
