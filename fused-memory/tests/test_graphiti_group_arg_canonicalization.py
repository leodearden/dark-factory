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
