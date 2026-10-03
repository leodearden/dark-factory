"""Unit tests for FalkorEdgeSearch, the graphiti ``search_interface`` for FalkorDB (task 6238).

Once γ's indices exist (task 3708), FalkorDB plans graphiti-core's stock edge
legs as a full Entity label scan driving a per-row index scan.  FalkorEdgeSearch
replaces those legs; every other leg graphiti dispatches to it unconditionally is
handed back to graphiti's built-in Cypher.

No connection is opened here.  The recording driver is
``object.__new__(_MultiTenantFalkorDriver)`` with an instance ``execute_query``,
so query assembly and dispatch are graphiti's real code.  Plan shape and rows
against a live FalkorDB are pinned in ``test_falkor_edge_search_integration.py``.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pytest
from graphiti_core.driver.search_interface.search_interface import SearchInterface
from graphiti_core.search import search_utils
from graphiti_core.search.search_filters import SearchFilters

from fused_memory.backends.falkor_edge_search import FalkorEdgeSearch
from fused_memory.backends.graphiti_client import _MultiTenantFalkorDriver

if TYPE_CHECKING:
    from falkordb.asyncio import FalkorDB


@dataclass(frozen=True)
class IssuedQuery:
    cypher: str
    params: dict[str, Any]


def recording_driver(
    records: Sequence[dict[str, Any]] = (),
) -> tuple[_MultiTenantFalkorDriver, list[IssuedQuery]]:
    """A hardened driver whose ``execute_query`` records the call and returns *records*."""
    driver = object.__new__(_MultiTenantFalkorDriver)
    issued: list[IssuedQuery] = []

    async def record(cypher: str, **params: Any):
        issued.append(IssuedQuery(cypher, params))
        return copy.deepcopy(list(records)), [], None

    driver.execute_query = record  # pyright: ignore[reportAttributeAccessIssue]
    return driver, issued


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


SearchLeg = Callable[[_MultiTenantFalkorDriver], Awaitable[list[Any]]]

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


class TestDelegatedLegsReachBuiltinCypher:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('leg', sorted(DELEGATED_LEGS))
    async def test_delegated_leg_issues_one_builtin_query(self, leg: str) -> None:
        """No NotImplementedError, no recursion back into the seam, one query issued."""
        driver, issued = recording_driver()
        assert isinstance(driver.search_interface, FalkorEdgeSearch)

        result = await DELEGATED_LEGS[leg](driver)

        assert result == []
        assert len(issued) == 1

    @pytest.mark.asyncio
    async def test_delegated_node_fulltext_keeps_the_hardened_query_builder(self) -> None:
        """Delegation must still route through the task-3334 ``build_fulltext_query``."""
        driver, issued = recording_driver()

        await DELEGATED_LEGS['node_fulltext_search'](driver)

        assert issued[0].params['query'] == '(@group_id:"g") (alpha | beta)'

    @pytest.mark.asyncio
    async def test_unoverridden_leg_falls_back_to_builtin_cypher(self) -> None:
        """Legs FalkorEdgeSearch leaves alone reach graphiti's NotImplementedError fallback."""
        driver, issued = recording_driver()

        result = await search_utils.community_fulltext_search(driver, 'alpha beta', ['g'])

        assert result == []
        assert len(issued) == 1


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
        driver, issued = recording_driver([CANNED_EDGE_RECORD])
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


OVERRIDDEN_METHODS = (
    'edge_fulltext_search',
    'edge_similarity_search',
    'node_fulltext_search',
    'node_similarity_search',
    'episode_fulltext_search',
)


@pytest.mark.parametrize('method', OVERRIDDEN_METHODS)
def test_override_signature_matches_search_interface(method: str) -> None:
    """graphiti calls these positionally; an upstream reorder must fail here, not bind silently."""
    assert inspect.signature(getattr(FalkorEdgeSearch, method)) == inspect.signature(
        getattr(SearchInterface, method)
    )
