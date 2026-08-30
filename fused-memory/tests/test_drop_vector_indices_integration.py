"""Integration tests for GraphitiBackend.drop_vector_indices() — requires a running FalkorDB.

Context: Task 3769.

``drop_vector_indices()`` was a PERMANENT NO-OP (its docstring records the three
defects), and the whole defect class here is "unit mocks pinned a fiction".  One
of the three — the old-style ``DROP INDEX`` verb, which FalkorDB applies to RANGE
indices only — is one NO unit test could have caught, because a mock accepts any
statement string.  That is what this module exists for.

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

HAZARD (task 4748): the drop side has its own asynchrony, alongside the
already-documented index BUILD asynchrony (see ``live_vector_graph``'s
MANDATORY task-3377 barrier below).  ``DROP VECTOR INDEX`` against a label
whose merged index carries other surviving fields is NOT an in-place catalog
mutation — FalkorDB builds a REPLACEMENT index for those fields, and any
catalog read taken before that build finishes can still see the pre-drop row,
including the just-dropped VECTOR property.  Every post-drop
``list_indices()`` / ``drop_vector_indices()`` call in this module is
therefore barriered by ``await_index_operational``, guarding the production
contract documented on
``fused_memory/backends/graphiti_client.py::GraphitiBackend.drop_vector_indices``;
those barrier calls are load-bearing, not defensive.
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
    poll_until,
    unique_graph_name,
)
from falkordb.asyncio import FalkorDB

from fused_memory.backends.graphiti_client import GraphitiBackend, _MultiTenantFalkorDriver

TEST_GRAPH: str = unique_graph_name('3769_drop_vector_indices')

pytestmark = [
    falkor_skipif(),
    # 30, not 15 (task 4748): pytest-timeout runs with func_only=False here, so
    # this budget covers fixture setup too, and each test below can now hold TWO
    # 10s barrier budgets -- live_vector_graph's build-side barrier (task 3377)
    # plus the post-drop barrier added by task 4748 -- so 15 would make that
    # pathological worst case (10s + 10s) a CERTAIN timeout instead of a rare
    # flake. fused-memory/pyproject.toml sets timeout_method='thread', whose
    # handler ends in os._exit(1) and kills the whole xdist worker, which is why
    # this headroom is not optional. 30 stays well under the pyproject default
    # of 60. Expected real cost of the added barrier is ~0.2ms on this module's
    # 1-node fixture graph (one CALL db.indexes() round-trip) -- this is
    # headroom, not spend.
    pytest.mark.timeout(30),
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
    # to seed bulk data, raise the module-wide @pytest.mark.timeout(30) alongside
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

    MEASURED 2026-08-27 (task 4748), extending the above with the post-drop
    rebuild window that motivates every ``await_index_operational`` call in
    this class's tests:

    * After ``DROP VECTOR INDEX FOR (n:Entity) ON (n.name_embedding)`` on a
      graph whose Entity index carries both ``name_embedding`` VECTOR and
      ``name`` RANGE, one ``CALL db.indexes()`` can return two Entity rows —
      ``['name'] {'name': ['RANGE']}`` at ``'[Indexing] N/M: UNDER
      CONSTRUCTION'`` beside ``['name_embedding', 'name']
      {'name_embedding': ['VECTOR'], 'name': ['RANGE']}`` at ``'OPERATIONAL'``.
    * The window is ~4 ms on the 1-node fixture graph and 0.21-0.75s at 50,000
      nodes; it is NOT a read-path artifact (RO_QUERY and QUERY agree at every
      instant).
    * A relationship index with a single field opens no window — dropping it
      removes the whole index.
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

            # MANDATORY (task 4748) — do NOT delete this as redundant with the
            # fixture's barrier. DROP re-opens the window the fixture's barrier
            # closed: dropping a VECTOR field from a label whose merged index
            # carries other fields makes FalkorDB BUILD A REPLACEMENT index, and
            # until that build finishes CALL db.indexes() returns BOTH rows in
            # one result set — the new ['name'] row reading '[Indexing] N/M:
            # UNDER CONSTRUCTION' and the OLD row still advertising the dropped
            # name_embedding VECTOR as OPERATIONAL. Without this barrier `after`
            # can capture the stale row and the `_vector_properties(after) ==
            # set()` assertion below fails with "vector indices survived
            # drop_vector_indices()". Measured against FalkorDB v41800; not a
            # read-path artifact — RO_QUERY and QUERY agree at every instant.
            # (RELATES_TO is never the phantom: its index has only the one
            # field, so dropping it removes the whole index and opens no window.)
            await await_index_operational(live_vector_graph)

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

            # MANDATORY (task 4748) — do NOT delete this as redundant with the
            # fixture's barrier. DROP re-opens the window the fixture's barrier
            # closed: dropping a VECTOR field from a label whose merged index
            # carries other fields makes FalkorDB BUILD A REPLACEMENT index, and
            # until that build finishes CALL db.indexes() returns BOTH rows in
            # one result set — the new ['name'] row reading '[Indexing] N/M:
            # UNDER CONSTRUCTION' and the OLD row still advertising the dropped
            # name_embedding VECTOR as OPERATIONAL. drop_vector_indices() opens
            # with its own list_indices(), so without this barrier pass 2's
            # internal read can land in the window, see the stale
            # Entity{name_embedding: ['VECTOR']} row, and re-issue
            # `DROP VECTOR INDEX FOR (n:Entity) ON (n.name_embedding)` — which
            # FalkorDB then answers with:
            #   redis.exceptions.ResponseError: Unable to drop index on
            #   :Entity(name_embedding): no such index.
            # drop_vector_indices() deliberately does not absorb per-statement
            # failures, so that ERRORS the test rather than failing an
            # assertion. Measured against FalkorDB v41800; not a read-path
            # artifact — RO_QUERY and QUERY agree at every instant. (RELATES_TO
            # is never the phantom: its index has only the one field, so
            # dropping it removes the whole index and opens no window.)
            await await_index_operational(live_vector_graph)

            second = await backend.drop_vector_indices(group_id=TEST_GRAPH)
            assert second == []
        finally:
            await backend.close()


@pytest_asyncio.fixture
async def bulk_vector_graph():
    """Seed a throwaway 1,000-node graph with a mixed VECTOR+RANGE Entity index.

    Backs TestDropRebuildWindow below — see its docstring for the measurement
    table that justifies 1,000 as the seed size now that the class polls for
    the phantom instead of requiring it on a single un-polled read. The mixed
    index (``name_embedding`` VECTOR merged with ``name`` RANGE, both on
    ``:Entity``) is what makes the class's ``DROP VECTOR INDEX`` a REBUILD
    rather than a removal: FalkorDB keeps the label's index record alive to
    carry the surviving RANGE field, and that rebuild is what opens the
    window under test. A single-field vector index — like the module
    fixture's ``live_vector_graph`` RELATES_TO index — never opens this
    window, because dropping its only field removes the whole record
    outright.

    No stale-graph pre-delete here (unlike ``live_vector_graph`` above):
    ``graph_name`` is minted fresh from ``unique_graph_name`` on every call,
    so no prior run's graph can ever share this name to delete.
    """
    client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    graph_name = unique_graph_name('4748_drop_rebuild_window')
    graph = client.select_graph(graph_name)
    await graph.query(
        'UNWIND range(1, 1000) AS i '
        'CREATE (:Entity {name: "n"+i, name_embedding: vecf32([1.0, 2.0, 3.0, 4.0])})'
    )
    await graph.query(
        'CREATE VECTOR INDEX FOR (n:Entity) ON (n.name_embedding) '
        "OPTIONS {dimension: 4, similarityFunction: 'cosine'}"
    )
    # The RANGE half of the merged index. Without this, dropping the VECTOR
    # field below removes the Entity index record entirely instead of
    # rebuilding it, and the window this fixture exists to open never opens.
    await graph.query('CREATE INDEX FOR (n:Entity) ON (n.name)')
    # (task 4748) At this seed size the initial HNSW build measured 0.02-0.3s
    # at loadavg ~96 on 32 cores, well inside await_index_operational's
    # default 10s budget -- no override needed. If this fixture's seed size
    # is ever raised, re-measure the build time before assuming the default
    # still covers it (see TestDropRebuildWindow's measurement table; at
    # 50,000 nodes the same build measured 8.5-9.2s and needed timeout_s=60).
    await await_index_operational(graph)
    try:
        yield graph
    finally:
        with contextlib.suppress(Exception):
            await graph.delete()
        with contextlib.suppress(Exception):
            await client.aclose()


def _entity_index_rows(result) -> list[tuple[dict, str]]:
    """Return (types, status) for every 'Entity'-labeled row in a raw db.indexes() result.

    Resolves the label/types/status columns BY NAME from ``result.header``,
    deliberately independent of ``list_indices()`` / ``resolve_header_positions``
    — the code path this module's barriers protect — for the same reason
    ``_vector_properties`` above hand-rolls its own read: a bug in the helper
    under test must not be able to make its own verification pass.
    """
    header_names = [col[1] for col in result.header]
    label_idx = header_names.index('label')
    types_idx = header_names.index('types')
    status_idx = header_names.index('status')
    return [
        (row[types_idx], row[status_idx])
        for row in result.result_set
        if row[label_idx] == 'Entity'
    ]


def _row_has_vector(types: dict) -> bool:
    """True if any property in a raw db.indexes() 'types' dict carries VECTOR."""
    return any(
        'VECTOR' in ([raw] if isinstance(raw, str) else raw)
        for raw in types.values()
    )


@pytest.mark.timeout(30)
class TestDropRebuildWindow:
    """PREMISE PIN for the post-DROP rebuild window — task 4748.

    This is NOT a red-first TDD test; it is expected to PASS on introduction.
    A timing race cannot be reliably RED-tested: at small seed sizes the
    phantom this test pins only shows up in a fraction of single-shot reads
    (see the measurement table below), so a test built to fail without the
    fix would itself be a new flake — the exact defect this task removes.
    This mirrors test_list_indices_integration.py::TestCallDbIndexesOverRoQuery
    .test_db_indexes_result_shape_matches_the_barriers_assumptions (task
    3377's build-side pin), which rejects the same idea for the same reason.

    Its durable job: fail loudly if a future FalkorDB upgrade either (a)
    removes the drop-side rebuild transient that every barrier task 4748 added
    exists to close — making them dead weight — or (b) changes the column
    names or the exact 'OPERATIONAL' ready sentinel await_index_operational
    depends on.

    MEASURED (task 4748, nice -n 19, loadavg ~96 on 32 cores, FalkorDB module
    v41800): after ``DROP VECTOR INDEX FOR (n:Entity) ON (n.name_embedding)``
    on a graph whose Entity index merges ``name_embedding`` VECTOR with
    ``name`` RANGE, the very next ``CALL db.indexes()`` can return TWO Entity
    rows — a fresh ``['name']`` row still ``'[Indexing] N/M: UNDER
    CONSTRUCTION'`` beside the stale ``['name_embedding', 'name']`` row still
    reporting ``'OPERATIONAL'`` and still advertising the just-dropped VECTOR
    type. Phantom rate on the FIRST (single-shot, un-polled) post-drop read,
    by seed size:

        nodes      phantom rate              window
        1          ~1 in 40-70 (tight loop)   ~4 ms
        1,000      1/3 (3/3 when polled)      40-64 ms
        10,000     1/3 (3/3 when polled)      52-135 ms
        50,000     7/7                        0.21-0.75s
        200,000    3/3                        ~0.73s

    A single un-polled read is only deterministic at 50,000+ nodes. Polling
    for the phantom within a bounded deadline (below) is deterministic at
    1,000 nodes too — and ~50x cheaper to seed and build (0.02-0.3s vs
    8.5-9.2s) — which is why the test below polls rather than reading once,
    and why bulk_vector_graph seeds only 1,000 nodes. A stuck poll still
    raises loudly at its deadline; it does not silently pass.
    """

    @pytest.mark.asyncio
    async def test_drop_vector_index_opens_a_rebuild_window_the_barrier_closes(
        self, bulk_vector_graph,
    ):
        """DROP opens the phantom window; await_index_operational closes it."""
        graph = bulk_vector_graph
        await graph.query('DROP VECTOR INDEX FOR (n:Entity) ON (n.name_embedding)')

        last_rows: list[tuple[dict, str]] = []

        async def _phantom_rows():
            nonlocal last_rows
            result = await graph.query('CALL db.indexes()')
            rows = _entity_index_rows(result)
            last_rows = rows
            # (a) the phantom is real: MORE THAN ONE Entity row, one of them
            # still listing name_embedding VECTOR; (b) await_index_operational
            # has a signal to wait on: at least one row is not 'OPERATIONAL'.
            is_phantom = (
                len(rows) > 1
                and any(_row_has_vector(types) for types, _ in rows)
                and any(status != 'OPERATIONAL' for _, status in rows)
            )
            return rows if is_phantom else None

        # A single un-polled read is only deterministic at 50,000+ nodes (see
        # class docstring); polling within a bounded deadline is what lets
        # bulk_vector_graph's seed size be 1,000 instead. The 5ms interval is
        # deliberately tight against the measured 40-64ms window at this seed
        # size — a round trip costs ~0.2ms, so this buys several attempts
        # inside the window rather than one.
        try:
            await poll_until(_phantom_rows, timeout=5.0, interval=0.005)
        except AssertionError as exc:
            raise AssertionError(
                'expected the drop-side rebuild phantom (>1 Entity row, one '
                'still VECTOR-typed, one not yet OPERATIONAL) within 5s but '
                f'last saw {last_rows!r}. If FalkorDB has stopped rebuilding '
                'the merged index in place, every post-drop barrier this '
                'task (4748) added is dead weight -- see this class '
                'docstring before deleting them.'
            ) from exc

        # (c) The barrier is SUFFICIENT: once satisfied, the phantom is gone.
        await await_index_operational(graph)

        result = await graph.query('CALL db.indexes()')
        entity_rows = _entity_index_rows(result)
        assert len(entity_rows) == 1, (
            f'expected exactly one Entity row once the barrier is satisfied but '
            f'got {len(entity_rows)}: {entity_rows!r}'
        )
        assert not _row_has_vector(entity_rows[0][0]), (
            f'expected no VECTOR type to survive the barrier: {entity_rows[0]!r}'
        )
