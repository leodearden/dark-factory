"""Integration tests for CALL db.indexes() over GRAPH.RO_QUERY — requires a running FalkorDB.

Context: Task 530 / esc-486-49
During Task 486, list_indices() in graphiti_client.py was refactored to use
``await graph.ro_query('CALL db.indexes()')`` instead of ``graph.query(...)``.
Reviewer robustness flagged a compatibility risk: stored-procedure CALL statements
are sometimes classified as write-capable by the DB engine and may be rejected
over the read-only GRAPH.RO_QUERY command path.

These tests empirically pin that FalkorDB (module v41800 / 4.18.0) accepts
``CALL db.indexes()`` on the GRAPH.RO_QUERY path and that GraphitiBackend.list_indices()
correctly parses the result.

Skip automatically when FalkorDB is not reachable.
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

TEST_GRAPH: str = unique_graph_name('530_list_indices_integration')

pytestmark = [
    falkor_skipif(),
    pytest.mark.timeout(15),
    pytest.mark.integration,
]


@pytest_asyncio.fixture
async def live_test_graph():
    """Provision a throwaway FalkorDB graph with a known index, yield it, then clean up."""
    client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    # Best-effort delete any stale graph from a prior run
    with contextlib.suppress(Exception):
        stale = client.select_graph(TEST_GRAPH)
        await stale.delete()
    # Create a fresh graph with a node and an index
    graph = client.select_graph(TEST_GRAPH)
    await graph.query("CREATE (:Entity {name: $n})", {'n': 'test'})
    await graph.query("CREATE INDEX FOR (n:Entity) ON (n.name)")
    # MANDATORY (task 3377) — do NOT drop this when copying this fixture; it is
    # the repo's template for live-FalkorDB tests. FalkorDB builds RANGE indices
    # asynchronously too, not just fulltext ones: measured on a 200k-node graph,
    # CALL db.indexes() read '[Indexing] N/200000: UNDER CONSTRUCTION' for 40
    # consecutive polls without ever reaching OPERATIONAL. Querying an
    # under-construction index silently succeeds for queries the engine would
    # otherwise reject, so without this barrier the tests below are a
    # false-green generator (measured 2 of 6 false successes on the sibling
    # fulltext module before the barrier landed).
    #
    # On this fixture's single-node graph the build finishes before the first
    # poll, so the barrier costs one CALL db.indexes() round-trip in practice.
    # If this fixture is ever widened to seed bulk data, raise the module-wide
    # @pytest.mark.timeout(15) alongside it — the barrier's own budget is 10s.
    await await_index_operational(graph)
    try:
        yield graph
    finally:
        # Teardown: best-effort delete the graph, then close the underlying
        # FalkorDB client so its connection pool does not leak across tests.
        with contextlib.suppress(Exception):
            await graph.delete()
        with contextlib.suppress(Exception):
            await client.aclose()


class TestCallDbIndexesOverRoQuery:
    """Pin that FalkorDB accepts CALL db.indexes() over GRAPH.RO_QUERY (GRAPH.RO_QUERY path).

    Task 530 / esc-486-49: list_indices() is the sole stored-procedure call on the
    read-only path in graphiti_client.py.  If this test ever starts failing (i.e.,
    FalkorDB returns a write-only / non-read-only error), revert list_indices() to
    use ``graph.query(...)`` instead of ``graph.ro_query(...)`` and update this test
    to pin the new behavior.

    Verified against FalkorDB module v41800 (4.18.0).
    """

    @pytest.mark.asyncio
    async def test_ro_query_accepts_call_db_indexes(self, live_test_graph):
        """FalkorDB must accept CALL db.indexes() sent as GRAPH.RO_QUERY without error."""
        result = await live_test_graph.ro_query('CALL db.indexes()')
        assert result.result_set is not None
        assert len(result.result_set) >= 1
        # The first (and only) index should be on label Entity
        assert result.result_set[0][0] == 'Entity'

    @pytest.mark.asyncio
    async def test_db_indexes_result_shape_matches_the_barriers_assumptions(self):
        """PREMISE PIN for the index-readiness barrier — task 3377.

        This is NOT a RED-first TDD test; it is expected to PASS on
        introduction. A timing race cannot be reliably RED-tested: on a
        single-node graph the index build completes before the first poll
        (measured: 6 of 6 polls already read OPERATIONAL), so an unbarriered
        fixture passes by luck — which is precisely the false-green mechanism
        the barrier removes. Manufacturing a reliable RED would mean seeding
        bulk data to widen the window, producing a slow and inherently flaky
        test that blows the module's 15s timeout budget.

        Its durable job is different: ``_fm_helpers.await_index_operational``
        resolves the readiness column BY NAME (``status``) from
        ``result.header`` and matches its value EXACTLY against
        ``'OPERATIONAL'``. If a FalkorDB upgrade renames or reorders that
        column, or changes the ready sentinel, the barrier silently degrades
        into a no-op. This test makes that fail loudly in the integration lane
        instead.

        Deliberately does NOT use the ``live_test_graph`` fixture and does NOT
        call the barrier. Riding the fixture would make this test a restatement
        of the fixture's own postcondition: the barrier already raises unless
        ``status`` is present and every row reads ``'OPERATIONAL'``, so the
        assertions could never fail without the fixture having errored first —
        drift would surface as a fixture ERROR on every test in the module
        rather than as this test's FAIL. It therefore provisions its own
        throwaway graph and reads ``CALL db.indexes()`` raw.

        The poll below is on the INVERSE condition — "no row is still building"
        — via the generic :func:`poll_until`, not the barrier's "every row is
        OPERATIONAL". That separation is what gives this test independent
        signal: if the ready sentinel were renamed, the barrier would merely
        hang and time out, while this test settles and then names the new value
        it found.

        Uses ``graph.query`` (not ``ro_query``) deliberately — the same path
        the barrier itself uses; the RO-path acceptance is pinned by the
        sibling test above.

        Verified against FalkorDB module v41800 (4.18.0), whose header is
        ['label', 'properties', 'types', 'options', 'language', 'stopwords',
        'entitytype', 'status', 'info'].
        """
        client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
        graph = client.select_graph(f'{TEST_GRAPH}_shape_pin')
        try:
            await graph.query('CREATE (:Entity {name: $n})', {'n': 'shape-pin'})
            await graph.query('CREATE INDEX FOR (n:Entity) ON (n.name)')

            # (1) Header shape: rows are (type, name) 2-tuples, so the column
            # NAME is col[1]. The barrier's `[col[1] for col in result.header]`
            # is meaningless if this ever stops holding.
            result = await graph.query('CALL db.indexes()')
            assert result.header, 'CALL db.indexes() returned an empty header'
            assert all(len(col) == 2 for col in result.header), (
                f'CALL db.indexes() header rows are no longer (type, name) '
                f'2-tuples (header={result.header!r}); '
                f'_fm_helpers.await_index_operational reads the column name as '
                f'col[1] and can no longer be trusted.'
            )

            # (2) A column literally named `status` exists — the barrier
            # resolves readiness from it by name.
            header_names = [col[1] for col in result.header]
            assert 'status' in header_names, (
                f'CALL db.indexes() has no "status" column (header={header_names}). '
                f'_fm_helpers.await_index_operational resolves readiness from this '
                f'column by name; without it the barrier cannot be trusted.'
            )
            status_idx = header_names.index('status')

            # (3) The build settles, and its terminal value is EXACTLY
            # 'OPERATIONAL'. Polled on the inverse ("nothing still building")
            # so this does not restate the barrier's own exit condition; 8s
            # leaves headroom inside the module-wide timeout(15) for the
            # round-trips above and the teardown below.
            observed: set[str] = set()

            async def _build_settled() -> bool:
                res = await graph.query('CALL db.indexes()')
                statuses = [row[status_idx] for row in res.result_set]
                observed.update(statuses)
                return bool(statuses) and not any(
                    'UNDER CONSTRUCTION' in s for s in statuses
                )

            await poll_until(
                _build_settled,
                timeout=8.0,
                interval=0.05,
                message=(
                    f'index build never settled on the shape-pin graph '
                    f'(statuses seen={sorted(observed)!r})'
                ),
            )

            assert observed >= {'OPERATIONAL'}, (
                f'the index build settled but never reported the exact ready '
                f'sentinel "OPERATIONAL" (statuses seen={sorted(observed)!r}). '
                f'_fm_helpers.await_index_operational matches that string exactly '
                f'and fail-closed, so a renamed sentinel turns the barrier into a '
                f'permanent timeout. Re-verify the barrier against this FalkorDB '
                f'version and update its predicate.'
            )
            unknown = {s for s in observed if s != 'OPERATIONAL'}
            assert all('UNDER CONSTRUCTION' in s for s in unknown), (
                f'CALL db.indexes() reported unrecognised status value(s) '
                f'{sorted(unknown)!r}. Documented values are the exact '
                f'"OPERATIONAL" and the progress-prefixed '
                f'"[Indexing] N/M: UNDER CONSTRUCTION".'
            )
        finally:
            with contextlib.suppress(Exception):
                await graph.delete()
            with contextlib.suppress(Exception):
                await client.aclose()


class TestBackendListIndicesLive:
    """Pin the end-to-end GraphitiBackend.list_indices() row-parsing path against live FalkorDB.

    Complements TestCallDbIndexesOverRoQuery: while that class verifies RPC-level acceptance,
    this class verifies the full path consumed by drop_vector_indices() — that list_indices()
    correctly parses the [label, field, type, entity_type] column layout from a live response.

    Task 530 / esc-486-49.  See also: tests/test_reindex.py::TestListIndices for unit tests.
    """

    @pytest.mark.asyncio
    async def test_list_indices_returns_records_live(self, mock_config, live_test_graph):
        """GraphitiBackend.list_indices() parses live FalkorDB index records correctly."""
        backend = GraphitiBackend(mock_config)
        # Inject a real driver directly — list_indices only needs _driver, not a full
        # Graphiti client stack (see _require_driver vs _require_client in graphiti_client.py).
        # Skip initialize() so the enumeration-based _ensure_indices pass in
        # GraphitiBackend.initialize() does not build Graphiti indices on the fixture graph.
        backend._driver = _MultiTenantFalkorDriver(host=FALKOR_HOST, port=FALKOR_PORT)
        try:
            records = await backend.list_indices(group_id=TEST_GRAPH)
            assert len(records) >= 1
            for rec in records:
                assert set(rec.keys()) >= {'label', 'field', 'type', 'entity_type'}
            # At least one record should be the Entity.name index we created in the fixture.
            # Note: FalkorDB returns field names as a list (e.g. ['name']), not a bare string.
            entity_records = [r for r in records if r['label'] == 'Entity']
            assert len(entity_records) >= 1
            # The field value is a list of indexed property names (and 'name' in 'name'
            # is also True, so this works uniformly whether field_val is a list or str).
            field_val = entity_records[0]['field']
            assert 'name' in field_val
        finally:
            await backend.close()
