"""Read protected graphs' index definitions, so a live arm-harness run can prove it left them untouched."""

from typing import Any

from falkordb.asyncio import FalkorDB
from falkordb.asyncio.graph import AsyncGraph

from arm_harness._fakes import PROTECTED_GRAPHS
from fused_memory.backends.falkor_indices import INDEX_DEFINITION_QUERY

CONCURRENT_TEST_GRAPH_PREFIXES = ('_test_', '_probe', 'evalmem_test_')
"""Throwaway graphs other test runs create and delete on a shared FalkorDB host."""


async def index_definitions(graph: AsyncGraph) -> list[Any]:
    """*graph*'s index definitions; ``INDEX_DEFINITION_QUERY`` names the definition columns and omits ``info`` and ``status``."""
    return (await graph.ro_query(INDEX_DEFINITION_QUERY)).result_set


async def protected_index_definitions(
    client: FalkorDB, graphs: set[str]
) -> dict[str, list[Any]]:
    """The index DEFINITIONS, over GRAPH.RO_QUERY, of every protected graph present on the host."""
    present = sorted(graphs & set(PROTECTED_GRAPHS))
    return {name: await index_definitions(client.select_graph(name)) for name in present}
