"""Unit tests for FalkorEdgeSearch, the graphiti ``search_interface`` for FalkorDB (task 6238).

Once γ's indices exist (task 3708), FalkorDB plans graphiti-core's stock edge
legs as a full Entity label scan driving a per-row index scan.  FalkorEdgeSearch
replaces those legs; every other leg graphiti dispatches to it unconditionally is
handed back to graphiti's built-in Cypher.

No connection is opened here: every driver is an unconnected
``_MultiTenantFalkorDriver`` from ``_query_recorder``, so query assembly and
dispatch are graphiti's real code.  Plan shape and rows against a live FalkorDB
are pinned in ``test_falkor_edge_search_integration.py``.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import pytest
from _query_recorder import IssuedQuery, recording_driver
from graphiti_core.driver.search_interface.search_interface import SearchInterface
from graphiti_core.search import search_utils
from graphiti_core.search.search_filters import SearchFilters

from fused_memory.backends.falkor_edge_search import FalkorEdgeSearch
from fused_memory.backends.graphiti_client import _MultiTenantFalkorDriver

if TYPE_CHECKING:
    from falkordb.asyncio import FalkorDB


def hardened_driver(
    records: Sequence[dict[str, Any]] = (),
) -> tuple[_MultiTenantFalkorDriver, list[IssuedQuery]]:
    return recording_driver(_MultiTenantFalkorDriver, records)


def declared_legs(interface: type[SearchInterface]) -> list[str]:
    """The search legs *interface* itself defines; graphiti awaits every one."""
    return sorted(
        name for name, member in vars(interface).items() if inspect.iscoroutinefunction(member)
    )


OVERRIDDEN_LEGS = declared_legs(FalkorEdgeSearch)
INHERITED_LEGS = sorted(set(declared_legs(SearchInterface)) - set(OVERRIDDEN_LEGS))


class TestDriverWiring:
    """Every per-group read and write driver comes from ``clone()``, so it must carry the seam."""

    def test_driver_class_carries_falkor_edge_search(self) -> None:
        assert isinstance(_MultiTenantFalkorDriver.search_interface, FalkorEdgeSearch)

    def test_clone_carries_falkor_edge_search(self) -> None:
        # Upstream ``__init__`` uses a supplied ``falkor_db`` directly instead of
        # dialling out; ``cast`` is for pyright only.
        stub_client = cast('FalkorDB', object())
        driver = _MultiTenantFalkorDriver(falkor_db=stub_client, database='graph_a')

        cloned = driver.clone('graph_b')

        assert isinstance(cloned, _MultiTenantFalkorDriver)
        assert isinstance(cloned.search_interface, FalkorEdgeSearch)


SearchLeg = Callable[[_MultiTenantFalkorDriver], Awaitable[object]]

# The legs graphiti-core hands to ``search_interface`` with NO NotImplementedError
# fallback, which FalkorEdgeSearch returns to graphiti's built-in Cypher.
DELEGATED_LEGS: dict[str, SearchLeg] = {
    'node_fulltext_search': lambda driver: search_utils.node_fulltext_search(
        driver, 'alpha beta', SearchFilters(), ['g']
    ),
    'node_similarity_search': lambda driver: search_utils.node_similarity_search(
        driver, [0.1, 0.2, 0.3], SearchFilters(), ['g']
    ),
    'episode_fulltext_search': lambda driver: search_utils.episode_fulltext_search(
        driver, 'alpha beta', SearchFilters(), ['g']
    ),
}

# Every leg FalkorEdgeSearch inherits, driven through its graphiti-core caller.
# Each caller catches the inherited NotImplementedError and runs its built-in Cypher.
FALLBACK_LEGS: dict[str, SearchLeg] = {
    'community_fulltext_search': lambda driver: search_utils.community_fulltext_search(
        driver, 'alpha beta', ['g']
    ),
    'community_similarity_search': lambda driver: search_utils.community_similarity_search(
        driver, [0.1, 0.2, 0.3], ['g']
    ),
    'episode_mentions_reranker': lambda driver: search_utils.episode_mentions_reranker(
        driver, [['a', 'b']]
    ),
    'get_embeddings_for_communities': lambda driver: search_utils.get_embeddings_for_communities(
        driver, []
    ),
    'node_bfs_search': lambda driver: search_utils.node_bfs_search(
        driver, ['origin'], SearchFilters(), 2, ['g']
    ),
    'node_distance_reranker': lambda driver: search_utils.node_distance_reranker(
        driver, ['a', 'b'], 'center'
    ),
}


class TestDelegatedLegsReachBuiltinCypher:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('leg', sorted(DELEGATED_LEGS))
    async def test_delegated_leg_issues_one_builtin_query(self, leg: str) -> None:
        """No NotImplementedError, no recursion back into the seam, one query issued."""
        driver, issued = hardened_driver()
        assert isinstance(driver.search_interface, FalkorEdgeSearch)

        result = await DELEGATED_LEGS[leg](driver)

        assert result == []
        assert len(issued) == 1

    @pytest.mark.asyncio
    async def test_delegated_node_fulltext_keeps_the_hardened_query_builder(self) -> None:
        """Delegation must still route through the task-3334 ``build_fulltext_query``."""
        driver, issued = hardened_driver()

        await DELEGATED_LEGS['node_fulltext_search'](driver)

        assert issued[0].params['query'] == '(@group_id:"g") (alpha | beta)'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('leg', INHERITED_LEGS)
    async def test_inherited_leg_falls_back_to_builtin_cypher(self, leg: str) -> None:
        """An upstream caller that stops catching NotImplementedError fails here, not in production."""
        assert leg in FALLBACK_LEGS, f'graphiti-core added {leg}: drive it through its caller here'
        driver, issued = hardened_driver()

        await FALLBACK_LEGS[leg](driver)

        assert issued != []


# One row shaped like graphiti's FalkorDB edge projection (get_entity_edge_return_query).
CANNED_EDGE_RECORD: dict[str, Any] = {
    'uuid': 'edge-uuid',
    'source_node_uuid': 'source-uuid',
    'target_node_uuid': 'target-uuid',
    'group_id': 'g',
    'created_at': '2026-10-03T00:00:00+00:00',
    'name': 'RELATES_TO',
    'fact': 'alpha beta',
    'episodes': ['ep'],
    'expired_at': None,
    'valid_at': None,
    'invalid_at': None,
    'attributes': {},
}


class TestEdgeFulltextSearch:
    @staticmethod
    async def _search(
        search_filter: SearchFilters | None = None, query: str = 'alpha beta'
    ) -> tuple[list[Any], list[IssuedQuery]]:
        driver, issued = hardened_driver([CANNED_EDGE_RECORD])
        edges = await search_utils.edge_fulltext_search(
            driver, query, search_filter or SearchFilters(), ['g'], 20
        )
        return edges, issued

    @pytest.mark.asyncio
    async def test_hits_are_not_rejoined_by_a_match_after_the_fulltext_call(self) -> None:
        """Any MATCH after the YIELD re-joins each hit, which FalkorDB plans as a per-row scan."""
        _, issued = await self._search()

        (query,) = issued
        assert 'MATCH' not in query.cypher.upper()

    @pytest.mark.asyncio
    async def test_forwards_graphiti_params(self) -> None:
        _, issued = await self._search()

        params = issued[0].params
        assert params['query'] == '(@group_id:"g") (alpha | beta)'
        assert params['group_ids'] == ['g']
        assert params['limit'] == 20

    @pytest.mark.asyncio
    async def test_applies_graphiti_search_filter_predicates(self) -> None:
        _, issued = await self._search(SearchFilters(edge_uuids=['x']))

        assert issued[0].params['edge_uuids'] == ['x']
        assert '$edge_uuids' in issued[0].cypher

    @pytest.mark.asyncio
    async def test_parses_records_into_entity_edges(self) -> None:
        edges, _ = await self._search()

        (edge,) = edges
        assert (edge.uuid, edge.source_node_uuid, edge.target_node_uuid) == (
            'edge-uuid',
            'source-uuid',
            'target-uuid',
        )

    @pytest.mark.asyncio
    async def test_all_stopword_query_issues_nothing(self) -> None:
        """graphiti's empty-sentinel contract: no searchable term, no query, no rows."""
        edges, issued = await self._search(query='the and of')

        assert edges == []
        assert issued == []


class TestEdgeSimilaritySearch:
    @staticmethod
    async def _search(
        group_ids: list[str] | None,
        source_node_uuid: str | None = None,
        target_node_uuid: str | None = None,
    ) -> tuple[list[Any], list[IssuedQuery]]:
        driver, issued = hardened_driver([CANNED_EDGE_RECORD])
        edges = await search_utils.edge_similarity_search(
            driver,
            [0.1, 0.2, 0.3],
            source_node_uuid,
            target_node_uuid,
            SearchFilters(),
            group_ids,
            20,
            0.6,
        )
        return edges, issued

    @pytest.mark.asyncio
    async def test_match_pattern_has_no_labeled_endpoint_node(self) -> None:
        """A labeled endpoint drives a label scan with a per-row edge index scan beneath it."""
        _, issued = await self._search(['g'])

        (query,) = issued
        assert '(n:Entity)' not in query.cypher
        assert '(m:Entity)' not in query.cypher

    @pytest.mark.asyncio
    async def test_forwards_graphiti_params(self) -> None:
        _, issued = await self._search(['g'])

        params = issued[0].params
        assert params['search_vector'] == [0.1, 0.2, 0.3]
        assert params['min_score'] == 0.6
        assert params['limit'] == 20
        assert params['group_ids'] == ['g']

    @pytest.mark.asyncio
    async def test_endpoint_filters_apply_with_group_ids(self) -> None:
        _, issued = await self._search(['g'], 's', 't')

        assert issued[0].params['source_uuid'] == 's'
        assert issued[0].params['target_uuid'] == 't'

    @pytest.mark.asyncio
    async def test_endpoint_filters_are_dropped_without_group_ids(self) -> None:
        """Upstream parity: graphiti applies them only inside its ``group_ids`` branch."""
        _, issued = await self._search(None, 's', 't')

        assert 'source_uuid' not in issued[0].params
        assert 'target_uuid' not in issued[0].params

    @pytest.mark.asyncio
    async def test_parses_records_into_entity_edges(self) -> None:
        edges, _ = await self._search(['g'])

        (edge,) = edges
        assert (edge.uuid, edge.source_node_uuid, edge.target_node_uuid) == (
            'edge-uuid',
            'source-uuid',
            'target-uuid',
        )


class TestEdgeBfsSearch:
    @staticmethod
    async def _search(
        origins: Sequence[str] | None = ('origin-entity', 'origin-episode'),
        depth: int = 3,
        search_filter: SearchFilters | None = None,
        group_ids: Sequence[str] | None = ('g',),
    ) -> tuple[list[Any], list[IssuedQuery]]:
        driver, issued = hardened_driver([CANNED_EDGE_RECORD])
        edges = await search_utils.edge_bfs_search(
            driver,
            None if origins is None else list(origins),
            depth,
            search_filter or SearchFilters(),
            None if group_ids is None else list(group_ids),
            20,
        )
        return edges, issued

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('origins', 'depth'),
        [(None, 3), ([], 3), (['origin-entity'], 0)],
        ids=['no-origins', 'empty-origins', 'zero-depth'],
    )
    async def test_no_reachable_edges_issues_nothing(
        self, origins: list[str] | None, depth: int
    ) -> None:
        """graphiti's contract: node_bfs_search treats depth < 1 as no results, and a path of 1..0 edges holds none."""
        edges, issued = await self._search(origins, depth)

        assert edges == []
        assert issued == []

    @pytest.mark.asyncio
    async def test_forwards_graphiti_params(self) -> None:
        _, issued = await self._search()

        (query,) = issued
        assert query.params['bfs_origin_node_uuids'] == ['origin-entity', 'origin-episode']
        assert query.params['limit'] == 20
        assert query.params['group_ids'] == ['g']
        assert '*1..3' in query.cypher

    @pytest.mark.asyncio
    async def test_group_filter_is_omitted_without_group_ids(self) -> None:
        _, issued = await self._search(group_ids=None)

        assert 'group_ids' not in issued[0].params

    @pytest.mark.asyncio
    async def test_applies_graphiti_search_filter_predicates(self) -> None:
        _, issued = await self._search(search_filter=SearchFilters(edge_uuids=['x']))

        assert issued[0].params['edge_uuids'] == ['x']
        assert '$edge_uuids' in issued[0].cypher

    @pytest.mark.asyncio
    async def test_parses_records_into_entity_edges(self) -> None:
        edges, _ = await self._search()

        (edge,) = edges
        assert (edge.uuid, edge.source_node_uuid, edge.target_node_uuid) == (
            'edge-uuid',
            'source-uuid',
            'target-uuid',
        )


@pytest.mark.parametrize('method', OVERRIDDEN_LEGS)
def test_override_signature_matches_search_interface(method: str) -> None:
    """graphiti calls these positionally; an upstream reorder must fail here, not bind silently."""
    assert inspect.signature(getattr(FalkorEdgeSearch, method)) == inspect.signature(
        getattr(SearchInterface, method)
    )
