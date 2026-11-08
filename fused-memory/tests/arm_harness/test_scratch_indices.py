"""Index state for the embedding-only configuration (arm_harness/scratch_indices.py)."""

from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import pytest

from arm_harness._fakes import PROTECTED_GRAPHS
from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, ScratchGuardError
from fused_memory.arm_harness.scratch_indices import IndexDropError, drop_all_indices
from fused_memory.backends.falkor_indices import IndexHeaderShapeError, IndexSpec

SCRATCH = 'evalmem_lme_emb_granite'

DB_INDEXES_HEADER = (
    (1, 'label'),
    (1, 'properties'),
    (1, 'types'),
    (1, 'options'),
    (1, 'language'),
    (1, 'stopwords'),
    (1, 'entitytype'),
    (1, 'status'),
    (1, 'info'),
)
"""``CALL db.indexes()``'s header on FalkorDB 4.18.0, read from the frozen reference."""

Catalog = dict[str, tuple[str, dict[str, list[str]]]]

REFERENCE_CATALOG: Catalog = {
    'RELATES_TO': ('RELATIONSHIP', {
        'created_at': ['RANGE'],
        'expired_at': ['RANGE'],
        'group_id': ['RANGE', 'FULLTEXT'],
        'invalid_at': ['RANGE'],
        'name': ['RANGE', 'FULLTEXT'],
        'uuid': ['RANGE'],
        'valid_at': ['RANGE'],
        'fact': ['FULLTEXT'],
    }),
    'NEXT_EPISODE': ('RELATIONSHIP', {'group_id': ['RANGE'], 'uuid': ['RANGE']}),
    'MENTIONS': ('RELATIONSHIP', {'group_id': ['RANGE'], 'uuid': ['RANGE']}),
    'HAS_MEMBER': ('RELATIONSHIP', {'uuid': ['RANGE']}),
    'HAS_EPISODE': ('RELATIONSHIP', {'group_id': ['RANGE'], 'uuid': ['RANGE']}),
    'Saga': ('NODE', {'group_id': ['RANGE'], 'name': ['RANGE'], 'uuid': ['RANGE']}),
    'Episodic': ('NODE', {
        'created_at': ['RANGE'],
        'group_id': ['RANGE', 'FULLTEXT'],
        'uuid': ['RANGE'],
        'valid_at': ['RANGE'],
        'content': ['FULLTEXT'],
        'source': ['FULLTEXT'],
        'source_description': ['FULLTEXT'],
    }),
    'Entity': ('NODE', {
        'created_at': ['RANGE'],
        'group_id': ['RANGE', 'FULLTEXT'],
        'name': ['RANGE', 'FULLTEXT'],
        'uuid': ['RANGE'],
        'summary': ['FULLTEXT'],
    }),
    'Community': ('NODE', {'uuid': ['RANGE'], 'name': ['FULLTEXT'], 'group_id': ['FULLTEXT']}),
}
"""evalmem_lme_ref_incumbent_a's catalog as measured read-only on 2026-10-09: 38 specs, no VECTOR."""

DROP_METHOD = {
    ('NODE', 'RANGE'): 'drop_node_range_index',
    ('NODE', 'FULLTEXT'): 'drop_node_fulltext_index',
    ('RELATIONSHIP', 'RANGE'): 'drop_edge_range_index',
    ('RELATIONSHIP', 'FULLTEXT'): 'drop_edge_fulltext_index',
}


def _specs(catalog: Catalog) -> set[IndexSpec]:
    return {
        (label, entity_type, prop, index_type)
        for label, (entity_type, types) in catalog.items()
        for prop, index_types in types.items()
        for index_type in index_types
    }


def _row(label: str, entity_type: str, types: Mapping[str, Sequence[str]]) -> list[object]:
    return [
        label,
        list(types),
        {prop: list(index_types) for prop, index_types in types.items()},
        {prop: {} for prop in types},
        'english',
        [],
        entity_type,
        'OPERATIONAL',
        {},
    ]


class FakeIndexedGraph:
    """A FalkorDB graph's index catalog: ``list_indices`` rows plus falkordb's four typed drops.

    Dropping an index that is not there raises, as FalkorDB does, so a double drop is loud.
    A ``sticky`` drop reports success and removes nothing.
    """

    def __init__(
        self,
        catalog: Catalog,
        *,
        sticky: frozenset[tuple[str, str, str]] = frozenset(),
        header: Sequence[Sequence[object]] = DB_INDEXES_HEADER,
    ) -> None:
        self.catalog = {
            label: (entity_type, {prop: list(ts) for prop, ts in types.items()})
            for label, (entity_type, types) in catalog.items()
        }
        self.sticky = sticky
        self.header = [list(entry) for entry in header]
        self.calls: list[tuple[str, ...]] = []

    @property
    def drops(self) -> list[tuple[str, ...]]:
        return [call for call in self.calls if call[0] != 'list_indices']

    async def list_indices(self) -> SimpleNamespace:
        self.calls.append(('list_indices',))
        rows = [_row(label, et, types) for label, (et, types) in self.catalog.items()]
        return SimpleNamespace(header=self.header, result_set=rows)

    def _drop(self, method: str, entity_type: str, index_type: str, label: str, prop: str):
        self.calls.append((method, label, prop))
        if (method, label, prop) in self.sticky:
            return SimpleNamespace(result_set=[])
        record_type, types = self.catalog.get(label, (None, {}))
        if record_type != entity_type or index_type not in types.get(prop, []):
            raise RuntimeError(f'no such index: {index_type} {entity_type} {label}.{prop}')
        types[prop].remove(index_type)
        if not types[prop]:
            del types[prop]
        if not types:
            del self.catalog[label]
        return SimpleNamespace(result_set=[])

    async def drop_node_range_index(self, label: str, attribute: str):
        return self._drop('drop_node_range_index', 'NODE', 'RANGE', label, attribute)

    async def drop_node_fulltext_index(self, label: str, attribute: str):
        return self._drop('drop_node_fulltext_index', 'NODE', 'FULLTEXT', label, attribute)

    async def drop_edge_range_index(self, label: str, attribute: str):
        return self._drop('drop_edge_range_index', 'RELATIONSHIP', 'RANGE', label, attribute)

    async def drop_edge_fulltext_index(self, label: str, attribute: str):
        return self._drop('drop_edge_fulltext_index', 'RELATIONSHIP', 'FULLTEXT', label, attribute)


@pytest.mark.asyncio
async def test_drops_every_reference_index_once_through_its_typed_drop():
    graph = FakeIndexedGraph(REFERENCE_CATALOG)

    await drop_all_indices(graph, SCRATCH)

    expected = {
        (DROP_METHOD[(entity_type, index_type)], label, prop)
        for label, entity_type, prop, index_type in _specs(REFERENCE_CATALOG)
    }
    assert len(graph.drops) == len(expected) == 38
    assert set(graph.drops) == expected
    assert graph.catalog == {}


@pytest.mark.asyncio
async def test_a_property_with_range_and_fulltext_gets_both_drops():
    graph = FakeIndexedGraph({'Entity': ('NODE', {'name': ['RANGE', 'FULLTEXT']})})

    await drop_all_indices(graph, SCRATCH)

    assert sorted(graph.drops) == [
        ('drop_node_fulltext_index', 'Entity', 'name'),
        ('drop_node_range_index', 'Entity', 'name'),
    ]


@pytest.mark.asyncio
async def test_returns_the_dropped_specs():
    graph = FakeIndexedGraph(REFERENCE_CATALOG)

    dropped = await drop_all_indices(graph, SCRATCH)

    assert len(dropped) == len(set(dropped))
    assert set(dropped) == _specs(REFERENCE_CATALOG)


@pytest.mark.asyncio
async def test_rereads_the_catalog_after_the_last_drop():
    graph = FakeIndexedGraph(REFERENCE_CATALOG)

    await drop_all_indices(graph, SCRATCH)

    assert graph.calls[0] == ('list_indices',)
    assert graph.calls[-1] == ('list_indices',)
    assert graph.calls.count(('list_indices',)) == 2


@pytest.mark.asyncio
async def test_an_index_free_graph_drops_nothing():
    graph = FakeIndexedGraph({})

    assert await drop_all_indices(graph, SCRATCH) == ()
    assert graph.drops == []


@pytest.mark.asyncio
async def test_a_surviving_index_raises_naming_it():
    survivor = ('RELATES_TO', 'RELATIONSHIP', 'fact', 'FULLTEXT')
    graph = FakeIndexedGraph(
        REFERENCE_CATALOG,
        sticky=frozenset({('drop_edge_fulltext_index', 'RELATES_TO', 'fact')}),
    )

    with pytest.raises(IndexDropError) as caught:
        await drop_all_indices(graph, SCRATCH)

    assert caught.value.graph_name == SCRATCH
    assert caught.value.indices == (survivor,)
    assert SCRATCH in str(caught.value)
    assert 'RELATES_TO' in str(caught.value) and 'fact' in str(caught.value)


@pytest.mark.asyncio
async def test_a_vector_index_raises_before_anything_is_dropped():
    catalog: Catalog = {
        **REFERENCE_CATALOG,
        'Entity': ('NODE', {'uuid': ['RANGE'], 'name_embedding': ['RANGE', 'VECTOR']}),
    }
    graph = FakeIndexedGraph(catalog)

    with pytest.raises(IndexDropError, match='VECTOR') as caught:
        await drop_all_indices(graph, SCRATCH)

    assert caught.value.indices == (('Entity', 'NODE', 'name_embedding', 'VECTOR'),)
    assert graph.drops == []


@pytest.mark.asyncio
async def test_a_catalog_header_without_entitytype_raises_rather_than_guessing():
    header = tuple(entry for entry in DB_INDEXES_HEADER if entry[1] != 'entitytype')
    graph = FakeIndexedGraph(REFERENCE_CATALOG, header=header)

    with pytest.raises(IndexHeaderShapeError):
        await drop_all_indices(graph, SCRATCH)

    assert graph.drops == []


@pytest.mark.asyncio
@pytest.mark.parametrize('protected', PROTECTED_GRAPHS)
async def test_a_non_scratch_name_is_refused_before_the_catalog_is_read(protected: str):
    graph = FakeIndexedGraph(REFERENCE_CATALOG)

    with pytest.raises(ScratchGuardError) as caught:
        await drop_all_indices(graph, protected)

    assert caught.value.checkpoint is GuardCheckpoint.INDEX_DROP
    assert graph.calls == []
