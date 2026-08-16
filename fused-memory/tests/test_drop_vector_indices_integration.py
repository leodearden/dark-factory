"""Integration tests for GraphitiBackend.drop_vector_indices() — requires a running FalkorDB.

Context: Task 3769.

``drop_vector_indices()`` was a PERMANENT NO-OP, and the whole defect class here
is "unit mocks pinned a fiction": its predicate compared the ``types`` COLUMN (a
dict) against the string ``'VECTOR'``, it handed ``drop_index()`` a LIST of
properties where a per-property string was required, and it used the old-style
``DROP INDEX`` verb, which FalkorDB applies to RANGE indices only.  No unit test
could have caught the third defect, because a mock accepts any statement string.

So this module locks the fix in END TO END against live FalkorDB: seed a graph
carrying a NODE vector index, a RELATIONSHIP vector index and a RANGE index,
call ``drop_vector_indices()``, and assert both vector indices are really gone
while the RANGE index SURVIVES.

Skip automatically when FalkorDB is not reachable.

HAZARD (task description; open escalation esc-3375-1): the backend here is built
the way ``TestBackendListIndicesLive`` builds it — construct ``GraphitiBackend``
and inject a ``_MultiTenantFalkorDriver`` directly, NEVER calling
``initialize()`` — because ``FalkorDriver.__init__`` fire-and-forgets
``build_indices_and_constraints()``.  Every graph named here is a throwaway from
``unique_graph_name``; no real project graph is ever touched.  The absence of
indices on real graphs is protected evidence for esc-3375-1.
"""

from __future__ import annotations

import contextlib

import pytest
import pytest_asyncio
from _fm_helpers import (
    FALKOR_HOST,
    FALKOR_PORT,
    await_index_operational,
    falkor_skipif,
    unique_graph_name,
)
from falkordb.asyncio import FalkorDB

from fused_memory.backends.graphiti_client import GraphitiBackend, _MultiTenantFalkorDriver

TEST_GRAPH: str = unique_graph_name('3769_drop_vector_indices')

pytestmark = [
    falkor_skipif(),
    pytest.mark.timeout(15),
    pytest.mark.integration,
]


def _vector_properties(records) -> set[tuple[str, str]]:
    """Every (label, property) in *records* whose type list contains 'VECTOR'.

    Deliberately hand-rolled rather than reusing
    ``falkor_indices.vector_index_properties``: the point of this module is to
    check the live index STATE independently of the helper under test, so a bug
    in that helper cannot make its own verification pass.
    """
    found: set[tuple[str, str]] = set()
    for record in records:
        types = record.get('type') or {}
        for prop, raw in types.items():
            if 'VECTOR' in ([raw] if isinstance(raw, str) else raw):
                found.add((record['label'], prop))
    return found


@pytest_asyncio.fixture
async def live_vector_graph():
    """Seed a throwaway graph with NODE + RELATIONSHIP vector indices and a RANGE index."""
    client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    # Best-effort delete any stale graph from a prior run.
    with contextlib.suppress(Exception):
        stale = client.select_graph(TEST_GRAPH)
        await stale.delete()

    graph = client.select_graph(TEST_GRAPH)
    await graph.query(
        'CREATE (a:Entity {name: $n, name_embedding: vecf32([1.0, 2.0, 3.0, 4.0])}) '
        '-[:RELATES_TO {fact_embedding: vecf32([1.0, 2.0, 3.0, 4.0])}]-> (a)',
        {'n': 'test'},
    )
    await graph.query(
        'CREATE VECTOR INDEX FOR (n:Entity) ON (n.name_embedding) '
        "OPTIONS {dimension: 4, similarityFunction: 'cosine'}"
    )
    # The survivor: a RANGE index that drop_vector_indices must NOT touch.
    await graph.query('CREATE INDEX FOR (n:Entity) ON (n.name)')
    await graph.query(
        'CREATE VECTOR INDEX FOR ()-[e:RELATES_TO]-() ON (e.fact_embedding) '
        "OPTIONS {dimension: 4, similarityFunction: 'cosine'}"
    )
    # MANDATORY (task 3377) — do NOT drop this when copying this fixture.
    # FalkorDB builds indices asynchronously, and querying an under-construction
    # index silently succeeds for queries the engine would otherwise reject, so
    # without this barrier these tests are a false-green generator.  On this
    # single-node graph the build finishes before the first poll, so it costs one
    # CALL db.indexes() round-trip in practice.  If this fixture is ever widened
    # to seed bulk data, raise the module-wide @pytest.mark.timeout(15) alongside
    # it — the barrier's own budget is 10s.
    await await_index_operational(graph)
    try:
        yield graph
    finally:
        with contextlib.suppress(Exception):
            await graph.delete()
        with contextlib.suppress(Exception):
            await client.aclose()


class TestDropVectorIndicesLive:
    """End-to-end: the vector indices really go away, and only they do.

    MEASURED 2026-08-16 on throwaway graph ``_impl3769_probe``, which is what
    these assertions encode:

    * ``DROP INDEX ON :Entity(emb)`` — the old-style form the pre-fix code issued
      via ``drop_index()`` — fails against a live VECTOR index with
      ``ERR Unable to drop index on :Entity(emb): no such index.``
    * ``DROP VECTOR INDEX FOR (n:Entity) ON (n.emb)`` reports ``Indices deleted: 1``.
    * The NODE form against a RELATIONSHIP vector index fails the same way as the
      first bullet, so ``entity_type`` is load-bearing.
    * ``DROP VECTOR INDEX`` on a property carrying ``['RANGE', 'VECTOR']`` leaves
      ``['RANGE']`` behind — it is surgical.
    """

    @pytest.mark.asyncio
    async def test_drops_both_vector_indices_and_spares_the_range_index(
        self, mock_config, live_vector_graph,
    ):
        backend = GraphitiBackend(mock_config)
        # HAZARD: inject the driver and never call initialize() —
        # FalkorDriver.__init__ fire-and-forgets build_indices_and_constraints(),
        # and the absence of indices on real graphs is esc-3375-1's protected
        # evidence.  This graph is a unique_graph_name throwaway either way.
        backend._driver = _MultiTenantFalkorDriver(host=FALKOR_HOST, port=FALKOR_PORT)
        try:
            before = await backend.list_indices(group_id=TEST_GRAPH)
            assert _vector_properties(before) == {
                ('Entity', 'name_embedding'),
                ('RELATES_TO', 'fact_embedding'),
            }, f'fixture did not seed the expected vector indices: {before!r}'

            dropped = await backend.drop_vector_indices(group_id=TEST_GRAPH)

            # Both entries come back, in the documented {'label', 'field'} shape
            # with `field` a per-property STRING (never the record's field LIST).
            assert {(d['label'], d['field']) for d in dropped} == {
                ('Entity', 'name_embedding'),
                ('RELATES_TO', 'fact_embedding'),
            }
            assert all(set(d) == {'label', 'field'} for d in dropped)
            assert all(isinstance(d['field'], str) for d in dropped)

            after = await backend.list_indices(group_id=TEST_GRAPH)
            # THE lock-in: no 'VECTOR' survives anywhere in any record's types.
            assert _vector_properties(after) == set(), (
                f'vector indices survived drop_vector_indices(): {after!r}'
            )
            # THE surgical-precision case: the RANGE index is untouched.
            assert ('Entity', 'name') in {
                (record['label'], prop)
                for record in after
                for prop, raw in (record.get('type') or {}).items()
                if 'RANGE' in ([raw] if isinstance(raw, str) else raw)
            }, f'the RANGE index on Entity.name was collateral damage: {after!r}'
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_is_idempotent_and_reports_zero_on_a_second_run(
        self, mock_config, live_vector_graph,
    ):
        """A second run finds nothing to drop and says so truthfully.

        Guards the inverse reading of the old defect: "Dropped 0 VECTOR
        index(es)" must mean the graph really has none, which it did not before
        task 3769.
        """
        backend = GraphitiBackend(mock_config)
        backend._driver = _MultiTenantFalkorDriver(host=FALKOR_HOST, port=FALKOR_PORT)
        try:
            first = await backend.drop_vector_indices(group_id=TEST_GRAPH)
            assert len(first) == 2

            second = await backend.drop_vector_indices(group_id=TEST_GRAPH)
            assert second == []
        finally:
            await backend.close()
