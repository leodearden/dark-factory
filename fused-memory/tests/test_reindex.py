"""Tests for reindex maintenance: GraphitiBackend stale-embedding queries and ReindexManager."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import assert_ro_query_only, extract_cypher, extract_params

from fused_memory.backends.graphiti_client import (
    EdgeNotFoundError,
    GraphitiBackend,
    NodeNotFoundError,
)
from fused_memory.maintenance.reindex import ReindexManager, ReindexResult, run_reindex

# ---------------------------------------------------------------------------
# step-1: query_stale_node_embeddings / query_stale_edge_embeddings
# ---------------------------------------------------------------------------

class TestQueryStaleNodeEmbeddings:
    """GraphitiBackend.query_stale_node_embeddings(expected_dim) returns stale nodes."""

    @pytest.mark.asyncio
    async def test_returns_stale_nodes(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        vec_3 = '<' + ', '.join(['0.1'] * 3) + '>'
        vec_5 = '<' + ', '.join(['0.1'] * 5) + '>'
        rows = [
            ['uuid-1', 'Node A', vec_3],
            ['uuid-2', 'Node B', vec_5],
        ]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.query_stale_node_embeddings(expected_dim=1536, group_id='test')
        assert len(result) == 2
        assert result[0] == ('uuid-1', 'Node A', 3)
        assert result[1] == ('uuid-2', 'Node B', 5)

    @pytest.mark.asyncio
    async def test_returns_empty_when_all_match(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.query_stale_node_embeddings(expected_dim=1536, group_id='test')
        assert result == []

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)  # client is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.query_stale_node_embeddings(expected_dim=1536, group_id='test')

    @pytest.mark.asyncio
    async def test_calls_ro_query(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.query_stale_node_embeddings(expected_dim=768, group_id='test')
        graph.ro_query.assert_called_once()
        cypher = extract_cypher(graph.ro_query.call_args)
        assert 'Entity' in cypher
        assert 'name_embedding' in cypher


class TestQueryStaleEdgeEmbeddings:
    """GraphitiBackend.query_stale_edge_embeddings(expected_dim) returns stale edges."""

    @pytest.mark.asyncio
    async def test_returns_stale_edges(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        vec_4 = '<' + ', '.join(['0.1'] * 4) + '>'
        rows = [
            ['edge-uuid-1', 'edge name', vec_4],
        ]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.query_stale_edge_embeddings(expected_dim=1536, group_id='test')
        assert len(result) == 1
        assert result[0] == ('edge-uuid-1', 'edge name', 4)

    @pytest.mark.asyncio
    async def test_returns_empty_when_all_match(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.query_stale_edge_embeddings(expected_dim=1536, group_id='test')
        assert result == []

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.query_stale_edge_embeddings(expected_dim=1536, group_id='test')

    @pytest.mark.asyncio
    async def test_calls_ro_query(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.query_stale_edge_embeddings(expected_dim=768, group_id='test')
        graph.ro_query.assert_called_once()
        cypher = extract_cypher(graph.ro_query.call_args)
        assert 'RELATES_TO' in cypher
        assert 'fact_embedding' in cypher


# ---------------------------------------------------------------------------
# step-3: get_node_text / get_edge_text
# ---------------------------------------------------------------------------

class TestGetNodeText:
    """GraphitiBackend.get_node_text(uuid) returns (name, summary) tuple."""

    @pytest.mark.asyncio
    async def test_returns_name_and_summary(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        graph = make_graph_mock([['Alice', 'Alice is a person']])
        backend._driver._get_graph = MagicMock(return_value=graph)
        name, summary = await backend.get_node_text('uuid-1', group_id='test')
        assert name == 'Alice'
        assert summary == 'Alice is a person'

    @pytest.mark.asyncio
    async def test_raises_node_not_found(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        graph = make_graph_mock([])  # empty result
        backend._driver._get_graph = MagicMock(return_value=graph)
        with pytest.raises(NodeNotFoundError):
            await backend.get_node_text('missing-uuid', group_id='test')

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.get_node_text('uuid-1', group_id='test')

    @pytest.mark.asyncio
    async def test_passes_uuid_to_query(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        graph = make_graph_mock([['Node', 'Summary']])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.get_node_text('specific-uuid', group_id='test')
        params = extract_params(graph.ro_query.call_args)
        assert params.get('uuid') == 'specific-uuid'

    @pytest.mark.asyncio
    async def test_uses_ro_query_not_query(self, mock_config, make_backend, make_graph_mock):
        """get_node_text uses ro_query (read-only path) and never calls graph.query."""
        backend = make_backend(mock_config)
        await assert_ro_query_only(backend, make_graph_mock, [['Node', 'Summary']], 'get_node_text', 'uuid-1', group_id='test')


class TestGetEdgeText:
    """GraphitiBackend.get_edge_text(uuid) returns (name, fact) tuple."""

    @pytest.mark.asyncio
    async def test_returns_name_and_fact(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        graph = make_graph_mock([['edge-name', 'Some fact about the edge']])
        backend._driver._get_graph = MagicMock(return_value=graph)
        name, fact = await backend.get_edge_text('edge-uuid-1', group_id='test')
        assert name == 'edge-name'
        assert fact == 'Some fact about the edge'

    @pytest.mark.asyncio
    async def test_raises_edge_not_found(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        with pytest.raises(EdgeNotFoundError):
            await backend.get_edge_text('missing-edge-uuid', group_id='test')

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.get_edge_text('edge-uuid-1', group_id='test')

    @pytest.mark.asyncio
    async def test_passes_uuid_to_query(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        graph = make_graph_mock([['name', 'fact']])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.get_edge_text('specific-edge-uuid', group_id='test')
        params = extract_params(graph.ro_query.call_args)
        assert params.get('uuid') == 'specific-edge-uuid'

    @pytest.mark.asyncio
    async def test_uses_ro_query_not_query(self, mock_config, make_backend, make_graph_mock):
        """get_edge_text uses ro_query (read-only path) and never calls graph.query."""
        backend = make_backend(mock_config)
        await assert_ro_query_only(backend, make_graph_mock, [['edge-name', 'Some fact']], 'get_edge_text', 'edge-uuid-1', group_id='test')


# ---------------------------------------------------------------------------
# step-5: update_node_embedding / update_edge_embedding
# ---------------------------------------------------------------------------

class TestUpdateNodeEmbedding:
    """GraphitiBackend.update_node_embedding(uuid, embedding) stores new vector."""

    @pytest.mark.asyncio
    async def test_calls_set_with_embedding(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        embedding = [0.1] * 1536
        await backend.update_node_embedding('uuid-1', embedding, group_id='test')
        assert graph.query.called

    @pytest.mark.asyncio
    async def test_passes_uuid_and_embedding_to_query(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        embedding = [0.5] * 128
        await backend.update_node_embedding('my-uuid', embedding, group_id='test')
        params = extract_params(graph.query.call_args)
        assert params.get('uuid') == 'my-uuid'
        assert params.get('embedding') == embedding

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.update_node_embedding('uuid-1', [0.1] * 10, group_id='test')


class TestUpdateEdgeEmbedding:
    """GraphitiBackend.update_edge_embedding(uuid, embedding) stores new vector."""

    @pytest.mark.asyncio
    async def test_calls_set_with_embedding(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        embedding = [0.2] * 1536
        await backend.update_edge_embedding('edge-uuid', embedding, group_id='test')
        assert graph.query.called

    @pytest.mark.asyncio
    async def test_passes_uuid_and_embedding_to_query(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        embedding = [0.7] * 64
        await backend.update_edge_embedding('edge-uuid-99', embedding, group_id='test')
        params = extract_params(graph.query.call_args)
        assert params.get('uuid') == 'edge-uuid-99'
        assert params.get('embedding') == embedding

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.update_edge_embedding('edge-uuid', [0.1] * 10, group_id='test')


# ---------------------------------------------------------------------------
# step-7: list_indices / drop_index
# ---------------------------------------------------------------------------

class TestListIndices:
    """GraphitiBackend.list_indices() returns parsed index records.

    These unit tests pin the Cypher string and ro_query usage with a mocked
    FalkorDB graph.  The empirical verification that FalkorDB actually accepts
    ``CALL db.indexes()`` on the ``GRAPH.RO_QUERY`` path lives in
    ``fused-memory/tests/test_list_indices_integration.py``
    (Task 530 / esc-486-49).
    """

    @pytest.mark.asyncio
    async def test_returns_index_list(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        # FalkorDB index records: [label, property, type, entity_type]
        rows = [
            ['Entity', 'name_embedding', 'VECTOR', 'NODE'],
            ['Entity', 'name', 'FULLTEXT', 'NODE'],
        ]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.list_indices(group_id='test')
        assert len(result) == 2
        assert result[0]['label'] == 'Entity'
        assert result[0]['field'] == 'name_embedding'
        assert result[0]['type'] == 'VECTOR'
        assert result[0]['entity_type'] == 'NODE'

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_indices(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.list_indices(group_id='test')
        assert result == []

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.list_indices(group_id='test')

    @pytest.mark.asyncio
    async def test_uses_ro_query_not_query(self, mock_config, make_backend, make_graph_mock):
        """list_indices uses ro_query (read-only path) and never calls graph.query."""
        backend = make_backend(mock_config)
        graph = await assert_ro_query_only(backend, make_graph_mock, [], 'list_indices', group_id='test')
        cypher = extract_cypher(graph.ro_query.call_args)
        assert 'db.indexes' in cypher



class TestDropIndex:
    """GraphitiBackend.drop_index(label, field) generates correct DROP Cypher."""

    @pytest.mark.asyncio
    async def test_drop_index_calls_query(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.drop_index('Entity', 'name_embedding', group_id='test')
        assert graph.query.called

    @pytest.mark.asyncio
    async def test_drop_index_cypher_contains_label_and_field(self, mock_config, make_backend, make_graph_mock):
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.drop_index('MyLabel', 'my_field', group_id='test')
        call_args = graph.query.call_args
        cypher = extract_cypher(call_args)
        assert 'MyLabel' in cypher
        assert 'my_field' in cypher

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.drop_index('Entity', 'name_embedding', group_id='test')


# ---------------------------------------------------------------------------
# step-1 (task-52): GraphitiBackend.drop_vector_indices()
# ---------------------------------------------------------------------------

class TestDropVectorIndices:
    """GraphitiBackend.drop_vector_indices() drops only VECTOR-type indices."""

    @pytest.mark.asyncio
    async def test_drops_only_vector_type_indices(self, mock_config, make_backend):
        """Calls drop_index for VECTOR indices only, not FULLTEXT/RANGE."""
        backend = make_backend(mock_config)

        indices = [
            {'label': 'Entity', 'field': 'name_embedding', 'type': 'VECTOR', 'entity_type': 'NODE'},
            {'label': 'Entity', 'field': 'name', 'type': 'FULLTEXT', 'entity_type': 'NODE'},
            {'label': 'RELATES_TO', 'field': 'fact_embedding', 'type': 'VECTOR', 'entity_type': 'RELATIONSHIP'},
        ]
        backend.list_indices = AsyncMock(return_value=indices)
        backend.drop_index = AsyncMock()

        await backend.drop_vector_indices(group_id='test')

        assert backend.drop_index.call_count == 2
        calls = backend.drop_index.call_args_list
        called_pairs = [(c[0][0], c[0][1]) for c in calls]
        assert ('Entity', 'name_embedding') in called_pairs
        assert ('RELATES_TO', 'fact_embedding') in called_pairs

    @pytest.mark.asyncio
    async def test_returns_list_of_dropped_indices(self, mock_config, make_backend):
        """Returns list of dicts with 'label' and 'field' for each dropped index."""
        backend = make_backend(mock_config)

        indices = [
            {'label': 'Entity', 'field': 'name_embedding', 'type': 'VECTOR', 'entity_type': 'NODE'},
            {'label': 'Entity', 'field': 'name', 'type': 'FULLTEXT', 'entity_type': 'NODE'},
        ]
        backend.list_indices = AsyncMock(return_value=indices)
        backend.drop_index = AsyncMock()

        result = await backend.drop_vector_indices(group_id='test')

        assert len(result) == 1
        assert result[0] == {'label': 'Entity', 'field': 'name_embedding'}

    @pytest.mark.asyncio
    async def test_no_op_when_no_vector_indices(self, mock_config, make_backend):
        """When no VECTOR indices exist, drop_index not called and returns []."""
        backend = make_backend(mock_config)

        indices = [
            {'label': 'Entity', 'field': 'name', 'type': 'FULLTEXT', 'entity_type': 'NODE'},
            {'label': 'Entity', 'field': 'created_at', 'type': 'RANGE', 'entity_type': 'NODE'},
        ]
        backend.list_indices = AsyncMock(return_value=indices)
        backend.drop_index = AsyncMock()

        result = await backend.drop_vector_indices(group_id='test')

        backend.drop_index.assert_not_called()
        assert result == []


# ---------------------------------------------------------------------------
# step-9: ReindexManager
# ---------------------------------------------------------------------------

class TestReindexManager:
    """ReindexManager.reindex() orchestrates stale-embedding detection and re-embedding."""

    def _make_manager(self, backend, embedder, expected_dim=1536):
        return ReindexManager(backend=backend, embedder=embedder, expected_dim=expected_dim)

    @pytest.mark.asyncio
    async def test_reindex_processes_stale_nodes_and_edges(self):
        """Finds 2 stale nodes and 1 stale edge, re-embeds each, updates embeddings."""

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

        backend = MagicMock()
        backend.query_stale_node_embeddings = AsyncMock(return_value=[])
        backend.query_stale_edge_embeddings = AsyncMock(return_value=[])
        embedder = MagicMock()

        manager = ReindexManager(backend=backend, embedder=embedder, expected_dim=768, group_id='test')
        await manager.reindex()

        backend.query_stale_node_embeddings.assert_called_once_with(expected_dim=768, group_id='test')
        backend.query_stale_edge_embeddings.assert_called_once_with(expected_dim=768, group_id='test')

    @pytest.mark.asyncio
    async def test_reindex_node_text_combined_for_embedding(self):
        """Embedder receives name + space + summary as text."""

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

        call_order: list[str] = []

        backend = MagicMock()
        embedder = MagicMock()

        manager = ReindexManager(backend=backend, embedder=embedder, expected_dim=1536)

        async def fake_drop(group_id=None):
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
    """run_reindex() delegates service lifecycle to maintenance_service(), runs reindex_and_replay.

    Lifecycle behaviour (initialize, close, CONFIG_PATH management) is fully
    covered by TestMaintenanceService in test_maintenance_utils.py.
    """

    @pytest.mark.asyncio
    async def test_calls_reindex_and_replay_with_correct_args(self, standard_mock_config, make_fake_maintenance_service):
        """run_reindex() constructs ReindexManager and calls reindex_and_replay."""
        mock_service = AsyncMock()
        mock_service.durable_queue = MagicMock()
        mock_service.graphiti = MagicMock()

        mock_result = {'reindex_result': ReindexResult(2, 1, 0), 'replay_count': 5}

        with (
            patch(
                'fused_memory.maintenance.reindex.maintenance_service',
                side_effect=make_fake_maintenance_service(standard_mock_config, mock_service),
            ),
            patch('fused_memory.maintenance.reindex.OpenAIEmbedder'),
            patch('fused_memory.maintenance.reindex.ReindexManager') as mock_mgr_cls,
        ):
            mock_mgr = MagicMock()
            mock_mgr.reindex_and_replay = AsyncMock(return_value=mock_result)
            mock_mgr_cls.return_value = mock_mgr

            result = await run_reindex()

        mock_mgr.reindex_and_replay.assert_awaited_once_with(
            mock_service.durable_queue, drop_indices=False
        )
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_passes_drop_indices_true(self, standard_mock_config, make_fake_maintenance_service):
        """run_reindex(drop_indices=True) passes drop_indices=True to reindex_and_replay."""
        mock_service = AsyncMock()
        mock_service.durable_queue = MagicMock()
        mock_service.graphiti = MagicMock()

        mock_result = {'reindex_result': ReindexResult(), 'replay_count': 0, 'indices_dropped': []}

        with (
            patch(
                'fused_memory.maintenance.reindex.maintenance_service',
                side_effect=make_fake_maintenance_service(standard_mock_config, mock_service),
            ),
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
    async def test_default_drop_indices_false(self, standard_mock_config, make_fake_maintenance_service):
        """run_reindex() with no drop_indices arg passes drop_indices=False."""
        mock_service = AsyncMock()
        mock_service.durable_queue = MagicMock()
        mock_service.graphiti = MagicMock()

        mock_result = {'reindex_result': ReindexResult(), 'replay_count': 0, 'indices_dropped': []}

        with (
            patch(
                'fused_memory.maintenance.reindex.maintenance_service',
                side_effect=make_fake_maintenance_service(standard_mock_config, mock_service),
            ),
            patch('fused_memory.maintenance.reindex.OpenAIEmbedder'),
            patch('fused_memory.maintenance.reindex.ReindexManager') as mock_mgr_cls,
        ):
            mock_mgr = MagicMock()
            mock_mgr.reindex_and_replay = AsyncMock(return_value=mock_result)
            mock_mgr_cls.return_value = mock_mgr

            await run_reindex()

        mock_mgr.reindex_and_replay.assert_awaited_once()
        call_kwargs = mock_mgr.reindex_and_replay.call_args[1]
        assert call_kwargs.get('drop_indices') is False

    @pytest.mark.asyncio
    async def test_creates_embedder_with_config_dimensions(self, standard_mock_config, make_fake_maintenance_service):
        """run_reindex() passes config.embedder.dimensions to OpenAIEmbedderConfig."""
        standard_mock_config.embedder.dimensions = 768
        standard_mock_config.embedder.model = 'text-embedding-ada-002'

        mock_service = AsyncMock()
        mock_service.durable_queue = MagicMock()
        mock_service.graphiti = MagicMock()

        mock_result = {'reindex_result': ReindexResult(), 'replay_count': 0}

        with (
            patch(
                'fused_memory.maintenance.reindex.maintenance_service',
                side_effect=make_fake_maintenance_service(standard_mock_config, mock_service),
            ),
            patch('fused_memory.maintenance.reindex.OpenAIEmbedderConfig') as mock_cfg_cls,
            patch('fused_memory.maintenance.reindex.OpenAIEmbedder'),
            patch('fused_memory.maintenance.reindex.ReindexManager') as mock_mgr_cls,
        ):
            mock_mgr = MagicMock()
            mock_mgr.reindex_and_replay = AsyncMock(return_value=mock_result)
            mock_mgr_cls.return_value = mock_mgr

            await run_reindex()

        mock_cfg_cls.assert_called_once()
        call_kwargs = mock_cfg_cls.call_args[1]
        assert call_kwargs.get('embedding_dim') == 768

    @pytest.mark.asyncio
    async def test_embedder_constructor_failure_propagates(self, standard_mock_config, make_fake_maintenance_service):
        """When OpenAIEmbedderConfig raises inside the context, the exception propagates."""
        mock_service = AsyncMock()
        mock_service.graphiti = MagicMock()

        with (
            patch(
                'fused_memory.maintenance.reindex.maintenance_service',
                side_effect=make_fake_maintenance_service(standard_mock_config, mock_service),
            ),
            patch(
                'fused_memory.maintenance.reindex.OpenAIEmbedderConfig',
                side_effect=ValueError('invalid embedder config'),
            ),
            pytest.raises(ValueError, match='invalid embedder config'),
        ):
            await run_reindex()

    @pytest.mark.asyncio
    async def test_manager_constructor_failure_propagates(self, standard_mock_config, make_fake_maintenance_service):
        """When ReindexManager raises inside the context, the exception propagates."""
        mock_service = AsyncMock()
        mock_service.graphiti = MagicMock()

        with (
            patch(
                'fused_memory.maintenance.reindex.maintenance_service',
                side_effect=make_fake_maintenance_service(standard_mock_config, mock_service),
            ),
            patch('fused_memory.maintenance.reindex.OpenAIEmbedderConfig'),
            patch('fused_memory.maintenance.reindex.OpenAIEmbedder'),
            patch(
                'fused_memory.maintenance.reindex.ReindexManager',
                side_effect=RuntimeError('manager init failed'),
            ),
            pytest.raises(RuntimeError, match='manager init failed'),
        ):
            await run_reindex()


# ---------------------------------------------------------------------------
# step-7: run_reindex delegates to maintenance_service
# ---------------------------------------------------------------------------

class TestRunReindexDelegation:
    """run_reindex() delegates service lifecycle to maintenance_service()."""

    @pytest.mark.asyncio
    async def test_delegates_to_maintenance_service(self, standard_mock_config, make_fake_maintenance_service):
        """run_reindex() calls maintenance_service(config_path) and uses yielded (config, service)."""
        mock_service = AsyncMock()
        mock_service.graphiti = MagicMock()
        mock_service.durable_queue = MagicMock()

        mock_result = {
            'reindex_result': ReindexResult(nodes_updated=1),
            'replay_count': 0,
            'indices_dropped': [],
        }

        with (
            patch(
                'fused_memory.maintenance.reindex.maintenance_service',
                side_effect=make_fake_maintenance_service(standard_mock_config, mock_service),
            ),
            patch('fused_memory.maintenance.reindex.OpenAIEmbedderConfig'),
            patch('fused_memory.maintenance.reindex.OpenAIEmbedder') as mock_embedder_cls,
            patch('fused_memory.maintenance.reindex.ReindexManager') as mock_mgr_cls,
        ):
            mock_mgr = MagicMock()
            mock_mgr.reindex_and_replay = AsyncMock(return_value=mock_result)
            mock_mgr_cls.return_value = mock_mgr

            result = await run_reindex(config_path='/tmp/config.yaml')

        mock_mgr_cls.assert_called_once_with(
            backend=mock_service.graphiti,
            embedder=mock_embedder_cls.return_value,
            expected_dim=1536,
        )
        assert result is mock_result


# ---------------------------------------------------------------------------
# step-{task-527}: GraphitiBackend.node_count
# ---------------------------------------------------------------------------

class TestNodeCount:
    """GraphitiBackend.node_count(graph_name) returns node count for the named graph."""

    @pytest.mark.asyncio
    async def test_returns_count(self, mock_config, make_backend, make_graph_mock):
        """Returns the integer count from result_set[0][0]."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([[42]])
        with patch.object(backend, '_graph_for', return_value=graph):
            result = await backend.node_count('my_graph')
        assert result == 42

    @pytest.mark.asyncio
    async def test_returns_zero_when_empty(self, mock_config, make_backend, make_graph_mock):
        """Returns 0 when result_set is empty."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        with patch.object(backend, '_graph_for', return_value=graph):
            result = await backend.node_count('empty_graph')
        assert result == 0

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when backend is not initialized (both client and _driver are None)."""
        backend = GraphitiBackend(mock_config)  # client is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.node_count('some_graph')

    @pytest.mark.asyncio
    async def test_raises_when_driver_explicitly_none(self, mock_config, make_backend):
        """Raises RuntimeError when _driver is explicitly None (client is set but driver is not)."""
        backend = make_backend(mock_config)  # client is set via make_backend
        backend._driver = None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.node_count('some_graph')

    @pytest.mark.asyncio
    async def test_uses_ro_query_not_query(self, mock_config, make_backend, make_graph_mock):
        """node_count uses ro_query (read-only path) and never calls graph.query."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([[7]])
        with patch.object(backend, '_graph_for', return_value=graph):
            await backend.node_count('test_graph')
        graph.ro_query.assert_awaited_once()
        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delegates_to_graph_for(self, mock_config, make_backend, make_graph_mock):
        """node_count delegates graph resolution to _graph_for, not _require_driver._get_graph."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([[42]])
        with patch.object(backend, '_graph_for', return_value=graph) as mock_graph_for:
            result = await backend.node_count('my_graph')
        mock_graph_for.assert_called_once_with('my_graph')
        assert result == 42


class TestListGraphs:
    """GraphitiBackend.list_graphs() returns non-system FalkorDB graph names."""

    @pytest.mark.asyncio
    async def test_returns_filtered(self, mock_config, make_backend):
        """Filters out 'default_db' and names ending in '_db' via _driver.client.list_graphs."""
        backend = make_backend(mock_config)
        backend._driver.client.list_graphs = AsyncMock(
            return_value=['proj_a', 'default_db', 'proj_b', 'internal_db']
        )
        result = await backend.list_graphs()
        assert result == ['proj_a', 'proj_b']

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when backend is not initialized (_driver is None)."""
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.list_graphs()
