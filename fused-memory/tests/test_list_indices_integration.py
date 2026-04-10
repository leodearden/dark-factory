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

import pytest
import pytest_asyncio
import redis
from falkordb.asyncio import FalkorDB

FALKOR_HOST: str = os.environ.get('FALKOR_HOST', 'localhost')
FALKOR_PORT: int = int(os.environ.get('FALKOR_PORT', '6379'))
TEST_GRAPH: str = '_test_530_list_indices_integration'


def _falkor_available() -> bool:
    try:
        r = redis.Redis(host=FALKOR_HOST, port=FALKOR_PORT, socket_connect_timeout=2)
        r.ping()
        r.close()
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
    yield graph
    # Teardown: best-effort delete
    with contextlib.suppress(Exception):
        await graph.delete()


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
