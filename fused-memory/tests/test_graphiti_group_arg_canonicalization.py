"""CGL-γ (task 2269, seam S4): every public GraphitiBackend method taking
group_id/group_ids canonicalizes those ARGUMENTS at method entry via α's
canonicalize_project_id (fused_memory.utils.validation), so the FalkorDB
graph KEY (via _driver_for/_graph_for), the node/edge group_id PROPERTY
(via graphiti_core's client.add_episode), and any direct-Cypher $group_id
FILTER always agree (RCA §4: task-2116's leak flag is exactly
resolve_project_id(graph) != node.group_id).

This seam covers durable-queue replay: durable_queue persists rows with a
raw ``group_id TEXT`` and memory_service._execute_graphiti_write re-executes
them verbatim into GraphitiBackend methods — after this task those raw
hyphen spellings land canonically at the backend boundary.

Organised by pipeline layer / test focus:
  TestReconcilePathKeyFilterAgreement   — step-1/2 (get_nodes_by_exact_name,
                                           find_duplicate_entity_nodes)
  TestAddEpisodePropertyAgreement       — step-3/4 (add_episode)
  TestGroupIdsFilterAgreement           — step-5/6 (search, search_nodes,
                                           build_communities)
  TestCompletenessSweep                 — step-7/8 (every remaining method)
  TestIdentityLockKeyAgreement          — step-9/10 (_identity_lock_for)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import extract_params

from fused_memory.backends.graphiti_client import GraphitiBackend
from fused_memory.utils.validation import PathShapedProjectIdError

_PATH_SHAPED = '-home-leo-src-x'


class TestReconcilePathKeyFilterAgreement:
    """get_nodes_by_exact_name / find_duplicate_entity_nodes: graph-KEY
    (selected via ``_driver._get_graph``) and the Cypher ``$group_id``
    FILTER param must both carry the canonical form, and a path-shaped
    group_id must be rejected before any DB call.
    """

    @pytest.mark.asyncio
    async def test_get_nodes_by_exact_name_canonicalizes_key_and_filter(
        self, mock_config, make_backend, make_graph_mock
    ):
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)

        await backend.get_nodes_by_exact_name('Foo', group_id='know-live')

        backend._driver._get_graph.assert_called_once_with('know_live')
        assert extract_params(graph.ro_query.call_args)['group_id'] == 'know_live'

    @pytest.mark.asyncio
    async def test_get_nodes_by_exact_name_rejects_path_shaped_group_id(
        self, mock_config, make_backend, make_graph_mock
    ):
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)

        with pytest.raises(PathShapedProjectIdError):
            await backend.get_nodes_by_exact_name('Foo', group_id=_PATH_SHAPED)

        graph.ro_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_find_duplicate_entity_nodes_canonicalizes_key_and_filter(
        self, mock_config, make_backend, make_graph_mock
    ):
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)

        await backend.find_duplicate_entity_nodes('Foo', group_id='know-live')

        backend._driver._get_graph.assert_called_once_with('know_live')
        assert extract_params(graph.ro_query.call_args)['group_id'] == 'know_live'

    @pytest.mark.asyncio
    async def test_find_duplicate_entity_nodes_rejects_path_shaped_group_id(
        self, mock_config, make_backend, make_graph_mock
    ):
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)

        with pytest.raises(PathShapedProjectIdError):
            await backend.find_duplicate_entity_nodes('Foo', group_id=_PATH_SHAPED)

        graph.ro_query.assert_not_awaited()


class TestAddEpisodePropertyAgreement:
    """add_episode: the group_id forwarded to graphiti_core's client.add_episode
    (which becomes the node/edge group_id PROPERTY) must be canonical.
    """

    @pytest.fixture
    def backend(self, mock_config):
        b = GraphitiBackend(mock_config)
        mock_client = MagicMock()
        mock_client.add_episode = AsyncMock(return_value=None)
        b.client = mock_client
        # add_episode routes via _client_for(group_id) (per-group client cache,
        # task 2266). Point _client_for at the same mock so assertions on
        # backend.client.add_episode still observe the call — mirrors
        # tests/test_temporal_context.py's backend fixture.
        b._client_for = MagicMock(return_value=mock_client)
        return b

    @pytest.mark.asyncio
    async def test_hyphen_group_id_canonicalized_on_client_call(self, backend):
        await backend.add_episode(name='e', content='c', group_id='know-live')

        assert backend.client.add_episode.call_args[1]['group_id'] == 'know_live'

    @pytest.mark.asyncio
    async def test_already_canonical_group_id_is_a_no_op(self, backend):
        """Idempotency: already-canonical input passes through unchanged."""
        await backend.add_episode(name='e', content='c', group_id='know_live')

        assert backend.client.add_episode.call_args[1]['group_id'] == 'know_live'

    @pytest.mark.asyncio
    async def test_rejects_path_shaped_group_id(self, backend):
        with pytest.raises(PathShapedProjectIdError):
            await backend.add_episode(name='e', content='c', group_id=_PATH_SHAPED)

        backend.client.add_episode.assert_not_called()


class TestGroupIdsFilterAgreement:
    """search / search_nodes / build_communities: the group_ids forwarded to
    graphiti_core's client (client-param agreement) and, for the two
    methods that select a driver, the graph chosen via _driver_for
    (driver-selection agreement) must both carry the canonical form.
    """

    @pytest.mark.asyncio
    async def test_search_canonicalizes_client_param_and_driver_selection(
        self, mock_config, make_backend
    ):
        backend = make_backend(mock_config)
        backend.client.search = AsyncMock(return_value=[])

        await backend.search('q', group_ids=['know-live'])

        assert backend.client.search.call_args.kwargs['group_ids'] == ['know_live']
        assert backend._driver.clone.call_args.kwargs['database'] == 'know_live'

    @pytest.mark.asyncio
    async def test_search_group_ids_none_passthrough(self, mock_config, make_backend):
        backend = make_backend(mock_config)
        backend.client.search = AsyncMock(return_value=[])

        await backend.search('q', group_ids=None)

        assert backend.client.search.call_args.kwargs['group_ids'] == []
        backend._driver.clone.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_rejects_path_shaped_element(self, mock_config, make_backend):
        backend = make_backend(mock_config)
        backend.client.search = AsyncMock(return_value=[])

        with pytest.raises(PathShapedProjectIdError):
            await backend.search('q', group_ids=[_PATH_SHAPED])

        backend.client.search.assert_not_called()
        backend._driver.clone.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_nodes_canonicalizes_client_param_and_driver_selection(
        self, mock_config, make_backend
    ):
        backend = make_backend(mock_config)
        backend.client.search_ = AsyncMock(return_value=MagicMock(nodes=[]))

        await backend.search_nodes('q', group_ids=['know-live'])

        assert backend.client.search_.call_args.kwargs['group_ids'] == ['know_live']
        assert backend._driver.clone.call_args.kwargs['database'] == 'know_live'

    @pytest.mark.asyncio
    async def test_search_nodes_rejects_path_shaped_element(self, mock_config, make_backend):
        backend = make_backend(mock_config)
        backend.client.search_ = AsyncMock(return_value=MagicMock(nodes=[]))

        with pytest.raises(PathShapedProjectIdError):
            await backend.search_nodes('q', group_ids=[_PATH_SHAPED])

        backend.client.search_.assert_not_called()
        backend._driver.clone.assert_not_called()

    @pytest.mark.asyncio
    async def test_build_communities_canonicalizes_client_param(
        self, mock_config, make_backend
    ):
        backend = make_backend(mock_config)
        backend.client.build_communities = AsyncMock()

        await backend.build_communities(group_ids=['know-live'])

        assert backend.client.build_communities.call_args.kwargs['group_ids'] == ['know_live']

    @pytest.mark.asyncio
    async def test_build_communities_rejects_path_shaped_element(
        self, mock_config, make_backend
    ):
        backend = make_backend(mock_config)
        backend.client.build_communities = AsyncMock()

        with pytest.raises(PathShapedProjectIdError):
            await backend.build_communities(group_ids=[_PATH_SHAPED])

        backend.client.build_communities.assert_not_called()


# (method_name, positional args, kwargs — group_id/group_ids set to the
# path-shaped constant) for every public group-arg GraphitiBackend method
# NOT already covered by the step-1/3/5 positive-path tests
# (get_nodes_by_exact_name, find_duplicate_entity_nodes, add_episode,
# search, search_nodes, build_communities). Together these 36 + 6 = 42
# cover GraphitiBackend's full public group-arg surface (excluding
# _driver_for/_graph_for/_ensure_indices/_resolve_or_create_entity/
# node_count, which are deliberately undecorated, and _identity_lock_for,
# covered separately by step-9/10).
_ALL_GROUP_ARG_SWEEP_CASES = [
    ('get_episode_by_uuid', ('ep1',), {'group_id': _PATH_SHAPED}),
    ('remove_episode', ('ep1',), {'group_id': _PATH_SHAPED}),
    ('remove_edge', ('e1',), {'group_id': _PATH_SHAPED}),
    ('update_edge', ('e1',), {'fact': 'f', 'group_id': _PATH_SHAPED}),
    ('query_stale_node_embeddings', (10,), {'group_id': _PATH_SHAPED}),
    ('query_stale_edge_embeddings', (10,), {'group_id': _PATH_SHAPED}),
    ('query_edges_by_time_range', ('2020-01-01', '2020-01-02'), {'group_id': _PATH_SHAPED}),
    ('get_valid_edges_for_node', ('n1',), {'group_id': _PATH_SHAPED}),
    ('get_connected_entity_uuids', ('n1',), {'group_id': _PATH_SHAPED}),
    ('get_all_valid_edges', (), {'group_id': _PATH_SHAPED}),
    ('bulk_remove_edges', (['e1'],), {'group_id': _PATH_SHAPED}),
    ('dedup_valid_edges_for_node', ('n1',), {'group_id': _PATH_SHAPED}),
    ('redirect_node_edges', ('d1', 's1'), {'group_id': _PATH_SHAPED}),
    ('merge_entities', ('d1', 's1'), {'group_id': _PATH_SHAPED}),
    ('delete_entity', ('n1',), {'group_id': _PATH_SHAPED}),
    ('delete_entity_node', ('n1',), {'group_id': _PATH_SHAPED}),
    ('get_node_text', ('n1',), {'group_id': _PATH_SHAPED}),
    ('resolve_entity_by_name', ('Foo',), {'group_id': _PATH_SHAPED}),
    ('refresh_entity_summary', ('n1',), {'group_id': _PATH_SHAPED}),
    ('set_entity_summary', ('n1', 's'), {'group_id': _PATH_SHAPED}),
    ('rename_entity_node', ('n1', 'NewName'), {'group_id': _PATH_SHAPED}),
    ('list_entity_nodes', (), {'group_id': _PATH_SHAPED}),
    ('detect_stale_with_edges', (), {'group_id': _PATH_SHAPED}),
    ('detect_stale_dry_run', (), {'group_id': _PATH_SHAPED}),
    ('detect_stale_summaries', (), {'group_id': _PATH_SHAPED}),
    ('rebuild_entity_from_edges', ('n1', 'Foo', []), {'group_id': _PATH_SHAPED, 'old_summary': ''}),
    ('update_node_summary', ('n1', 's'), {'group_id': _PATH_SHAPED}),
    ('update_node_name', ('n1', 'Name'), {'group_id': _PATH_SHAPED}),
    ('get_edge_text', ('e1',), {'group_id': _PATH_SHAPED}),
    ('get_edge_invalid_at', ('e1',), {'group_id': _PATH_SHAPED}),
    ('update_node_embedding', ('n1', [0.1]), {'group_id': _PATH_SHAPED}),
    ('update_edge_embedding', ('e1', [0.1]), {'group_id': _PATH_SHAPED}),
    ('list_indices', (), {'group_id': _PATH_SHAPED}),
    ('drop_index', ('Entity', 'name'), {'group_id': _PATH_SHAPED}),
    ('drop_vector_indices', (), {'group_id': _PATH_SHAPED}),
    ('retrieve_episodes', (), {'group_ids': [_PATH_SHAPED]}),
]

assert len(_ALL_GROUP_ARG_SWEEP_CASES) == 36, (
    'Sweep must cover exactly the 36 public group-arg GraphitiBackend methods '
    'not already covered by the step-1/3/5 positive-path tests — update this '
    'table if the decorated surface ever changes.'
)


class TestCompletenessSweep:
    """Behavioral sweep: EVERY remaining public group-arg GraphitiBackend
    method rejects a path-shaped group_id/group_ids element before any DB
    call — rejection happens at the decorator, before any body logic runs.

    Guards every method not individually tested above, and any method
    added in the future — canonicalize-and-reject is the first group_id
    operation for every one of them.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'method_name, args, kwargs',
        _ALL_GROUP_ARG_SWEEP_CASES,
        ids=[case[0] for case in _ALL_GROUP_ARG_SWEEP_CASES],
    )
    async def test_path_shaped_rejected_before_any_db_call(
        self, mock_config, make_backend, method_name, args, kwargs
    ):
        backend = make_backend(mock_config)

        with pytest.raises(PathShapedProjectIdError) as exc_info:
            await getattr(backend, method_name)(*args, **kwargs)

        assert _PATH_SHAPED in str(exc_info.value), (
            f'{method_name}: expected offending value {_PATH_SHAPED!r} named in error, '
            f'got: {exc_info.value!r}'
        )
        assert backend._driver.method_calls == [], (
            f'{method_name}: expected no driver calls before rejection, '
            f'got: {backend._driver.method_calls!r}'
        )
        assert backend.client.method_calls == [], (
            f'{method_name}: expected no client calls before rejection, '
            f'got: {backend.client.method_calls!r}'
        )


class TestIdentityLockKeyAgreement:
    """_identity_lock_for: the write-time-identity LOCK KEY must canonicalize
    so a replayed raw-hyphen durable-queue write and a normal canonical
    write for the same project — both now landing in the same 'know_live'
    graph — serialize under the SAME asyncio.Lock instance, instead of
    racing on entity-name resolution under two different locks guarding one
    graph.
    """

    def test_hyphen_and_canonical_share_the_same_lock(self, mock_config, make_backend):
        backend = make_backend(mock_config)

        assert backend._identity_lock_for('know-live') is backend._identity_lock_for('know_live')

    def test_rejects_path_shaped_group_id(self, mock_config, make_backend):
        backend = make_backend(mock_config)

        with pytest.raises(PathShapedProjectIdError):
            backend._identity_lock_for(_PATH_SHAPED)
