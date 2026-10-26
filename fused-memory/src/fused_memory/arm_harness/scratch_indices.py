"""The embedding-only index state: every index on an arm's scratch graph dropped, the empty catalog verified.

Catalog rows are read through falkor_indices' by-name header resolution and record
normalisation, so no ``CALL db.indexes()`` column is parsed here. A VECTOR index is
refused before anything is dropped: no copy of the reference carries one, so its
presence means the graph is not the copy this run made. The with-indices direction
is GraphitiBackend.ensure_indices, reached through arm_backend.
"""

from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import Any, Protocol

from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, require_scratch_name
from fused_memory.backends.falkor_indices import (
    IndexSpec,
    normalize_index_record,
    resolve_header_positions,
    vector_index_properties,
)

VECTOR = 'VECTOR'

_CATALOG_COLUMNS = {
    'label': 'label',
    'field': 'properties',
    'type': 'types',
    'entity_type': 'entitytype',
}

Drop = Callable[[str, str], Awaitable[object]]


class IndexedGraph(Protocol):
    """The slice of falkordb's AsyncGraph an index drop uses."""

    async def list_indices(self) -> Any: ...

    async def drop_node_range_index(self, label: str, attribute: str) -> object: ...

    async def drop_node_fulltext_index(self, label: str, attribute: str) -> object: ...

    async def drop_edge_range_index(self, label: str, attribute: str) -> object: ...

    async def drop_edge_fulltext_index(self, label: str, attribute: str) -> object: ...


_DROPS: Mapping[tuple[str, str], Callable[[IndexedGraph], Drop]] = MappingProxyType({
    ('NODE', 'RANGE'): lambda graph: graph.drop_node_range_index,
    ('NODE', 'FULLTEXT'): lambda graph: graph.drop_node_fulltext_index,
    ('RELATIONSHIP', 'RANGE'): lambda graph: graph.drop_edge_range_index,
    ('RELATIONSHIP', 'FULLTEXT'): lambda graph: graph.drop_edge_fulltext_index,
})


class IndexDropError(RuntimeError):
    """A scratch graph's index catalog is not empty where embedding-only needs it to be."""

    def __init__(self, graph_name: str, indices: tuple[IndexSpec, ...], reason: str) -> None:
        self.graph_name = graph_name
        self.indices = indices
        listed = '; '.join(
            f'{index_type} {entity_type} {label}.{field}'
            for label, entity_type, field, index_type in indices
        )
        super().__init__(f'{graph_name!r}: {reason}: {listed}')


async def drop_all_indices(graph: IndexedGraph, name: str) -> tuple[IndexSpec, ...]:
    """Drop every index on the scratch graph ``name`` and return what was dropped."""
    require_scratch_name(name, checkpoint=GuardCheckpoint.INDEX_DROP)
    catalog = await _read_catalog(graph)
    vectors = tuple(spec for spec in catalog if spec[3] == VECTOR)
    if vectors:
        raise IndexDropError(
            name, vectors, 'carries VECTOR indices, which no reference copy holds; nothing dropped'
        )
    for label, entity_type, field, index_type in catalog:
        await _DROPS[(entity_type, index_type)](graph)(label, field)
    survivors = await _read_catalog(graph)
    if survivors:
        raise IndexDropError(name, survivors, 'indices survived the drop')
    return catalog


async def _read_catalog(graph: IndexedGraph) -> tuple[IndexSpec, ...]:
    response = await graph.list_indices()
    positions = resolve_header_positions(response.header, _CATALOG_COLUMNS)
    specs: list[IndexSpec] = []
    for row in response.result_set or []:
        record = {key: row[position] for key, position in positions.items()}
        specs.extend(normalize_index_record(record))
        specs.extend(
            (record['label'], record['entity_type'], prop, VECTOR)
            for prop in vector_index_properties(record)
        )
    return tuple(specs)
