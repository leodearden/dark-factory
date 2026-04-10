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
import os
import uuid

import pytest
import pytest_asyncio
from falkordb import FalkorDB as _SyncFalkorDB
from falkordb.asyncio import FalkorDB

from fused_memory.backends.graphiti_client import GraphitiBackend, _MultiTenantFalkorDriver

FALKOR_HOST: str = os.environ.get('FALKOR_HOST', 'localhost')
FALKOR_PORT: int = int(os.environ.get('FALKOR_PORT', '6379'))
# Per-run graph name so concurrent test runs (xdist, CI matrix on a shared
# FalkorDB) do not race on the same graph and wipe each other's fixtures.
TEST_GRAPH: str = f'_test_530_list_indices_integration_{uuid.uuid4().hex[:8]}'


def _falkor_available() -> bool:
    """FalkorDB-native reachability probe.

    Uses the sync ``falkordb.FalkorDB`` client at module import time so the
    skip-guard does not depend on ``redis`` (which is not a declared
    dependency of fused-memory — only a transitive of graphiti-core[falkordb]).
    If falkordb ever switches transport, this probe fails loudly instead of
    silently disappearing.
    """
    try:
        client = _SyncFalkorDB(
            host=FALKOR_HOST, port=FALKOR_PORT, socket_connect_timeout=2
        )
        try:
            client.select_graph('_probe').query('RETURN 1')
        finally:
            with contextlib.suppress(Exception):
                client.close()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _falkor_available(), reason='FalkorDB not reachable'),
    pytest.mark.timeout(15),
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
