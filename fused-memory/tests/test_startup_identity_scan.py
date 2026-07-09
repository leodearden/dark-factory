"""Tests for the W6-ε startup identity-integrity scan (task 2210).

Covers:
- GraphitiBackend._scan_duplicate_entity_names — dup-NODE alarm (B5)
- GraphitiBackend._repair_duplicate_edge_uuids — dup-uuid-EDGE repair (B6)
- GraphitiBackend._run_startup_identity_scan — per-graph orchestrator
- GraphitiBackend.initialize() wiring

An idempotent startup identity-integrity pass, run per-graph over
list_graphs() during initialize(). Two responsibilities: (a) dup-NODE
alarm — detect exact-name duplicate Entity nodes and emit a LOUD WARN
naming group_id+name (a safety net for the write-time identity gate, task
2198/α + task 2202/β — a POSITIVE DETECTION signal, not input rejection);
(b) dup-uuid-EDGE repair — a one-shot idempotent migration that re-mints
fresh uuids (+ superseded_edge_uuid) on legacy dup-uuid RELATES_TO edges so
per-uuid count(*)<=1, while PRESERVING the edge set. No-op on a clean
graph. Unblocks W6-ζ (uuid-keyed read dedup).
"""
from __future__ import annotations

import contextlib
import logging
import os
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from _fm_helpers import extract_cypher, extract_params
from falkordb import FalkorDB as _SyncFalkorDB
from falkordb.asyncio import FalkorDB

from fused_memory.backends.graphiti_client import GraphitiBackend, _MultiTenantFalkorDriver

# ---------------------------------------------------------------------------
# Live-FalkorDB integration test harness (task 2210 W6-ε)
# ---------------------------------------------------------------------------
#
# The trailing live-FalkorDB test class pins the true B5/B6 semantics (real
# count(*) grouping, per-uuid uniqueness, edge-set preservation, idempotent
# re-run, group_id exclusion of a task-2115 foreign clone) against a REAL
# FalkorDB server -- the mock tests above can only pin the Cypher
# string/params shape, not FalkorDB's actual grouping/count semantics.
# Duplicated (not shared via conftest.py) from test_merge_entities.py's
# harness (itself duplicated from test_refresh_entity_summary.py /
# test_list_indices_integration.py), per the established per-module
# duplication convention that sidesteps the `sys.modules['conftest']`
# collision documented in _fm_helpers.py.
#
# Skipped automatically when FalkorDB is not reachable; the mock tests above
# guarantee coverage in that case.
# ---------------------------------------------------------------------------

FALKOR_HOST: str = os.environ.get('FALKOR_HOST', 'localhost')
FALKOR_PORT: int = int(os.environ.get('FALKOR_PORT', '6379'))


def _falkor_available() -> bool:
    """FalkorDB-native reachability probe (mirrors test_merge_entities.py)."""
    try:
        client = _SyncFalkorDB(host=FALKOR_HOST, port=FALKOR_PORT, socket_connect_timeout=2)
        try:
            client.select_graph('_probe').query('RETURN 1')
        finally:
            with contextlib.suppress(Exception):
                client.close()
        return True
    except Exception:
        return False


@pytest_asyncio.fixture
async def startup_scan_live_graph():
    """Provision a throwaway, uniquely-named FalkorDB graph, yield it, then clean up.

    A fresh graph name is minted on every invocation (rather than a shared
    module-level constant) so this module's live tests never share state,
    even under pytest-xdist or a shared FalkorDB instance.
    """
    graph_name = f'_test_2210_startup_scan_{uuid.uuid4().hex[:8]}'
    client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    with contextlib.suppress(Exception):
        stale = client.select_graph(graph_name)
        await stale.delete()
    graph = client.select_graph(graph_name)
    try:
        yield graph_name, graph
    finally:
        with contextlib.suppress(Exception):
            await graph.delete()
        with contextlib.suppress(Exception):
            await client.aclose()


# ---------------------------------------------------------------------------
# step-1/step-2: GraphitiBackend._scan_duplicate_entity_names — B5 dup-node alarm
# ---------------------------------------------------------------------------

class TestScanDuplicateEntityNames:
    """GraphitiBackend._scan_duplicate_entity_names(group_id) — dup-NODE alarm (B5)."""

    @pytest.mark.asyncio
    async def test_returns_duplicates_and_emits_warning(
        self, mock_config, make_backend, make_graph_mock, caplog,
    ):
        """A duplicated name returns [(name, count)] and fires a WARN naming
        both the group_id and the duplicated name."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[['Foo', 2]])
        backend._driver._get_graph = MagicMock(return_value=graph)

        with caplog.at_level(logging.WARNING, logger='fused_memory.backends.graphiti_client'):
            result = await backend._scan_duplicate_entity_names(group_id='g1')

        assert result == [('Foo', 2)]
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('g1' in r.message and 'Foo' in r.message for r in warnings), (
            f'expected a WARNING naming group_id and name, got: {[r.message for r in warnings]}'
        )

    @pytest.mark.asyncio
    async def test_cypher_is_group_id_scoped(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """2115-advisory: the dup-name scan is scoped by an explicit
        `n.group_id = $group_id` property predicate, not just the graph key
        selected via _graph_for — a foreign-group_id clone must not alarm
        here."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[])
        backend._driver._get_graph = MagicMock(return_value=graph)

        await backend._scan_duplicate_entity_names(group_id='g1')

        graph.ro_query.assert_awaited_once()
        call = graph.ro_query.call_args
        assert 'n.group_id = $group_id' in extract_cypher(call)
        assert extract_params(call).get('group_id') == 'g1'

    @pytest.mark.asyncio
    async def test_clean_graph_no_duplicates_no_warning(
        self, mock_config, make_backend, make_graph_mock, caplog,
    ):
        """No duplicate names -> empty list and no WARNING emitted."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[])
        backend._driver._get_graph = MagicMock(return_value=graph)

        with caplog.at_level(logging.WARNING, logger='fused_memory.backends.graphiti_client'):
            result = await backend._scan_duplicate_entity_names(group_id='g1')

        assert result == []
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# step-3/step-4: GraphitiBackend._repair_duplicate_edge_uuids — B6 dup-uuid-edge repair
# ---------------------------------------------------------------------------

class TestRepairDuplicateEdgeUuids:
    """GraphitiBackend._repair_duplicate_edge_uuids(group_id) — dup-uuid-EDGE repair (B6)."""

    @pytest.mark.asyncio
    async def test_remints_all_but_first_edge_in_dup_group(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """A three-edge dup-uuid group re-mints the last two (the survivor
        keeps eids[0]'s original uuid), returning the re-mint count. Neither
        the detection query nor the per-edge repair queries carry a
        group_id predicate -- the repair is graph-key-scoped so EVERY
        RELATES_TO edge reaches per-uuid uniqueness, regardless of its
        group_id property."""
        backend = make_backend(mock_config)

        async def router(cypher, params=None):
            result = MagicMock()
            if 'collect(ID(e))' in cypher:
                result.result_set = [['shared-uuid', [101, 102, 103]]]
            else:
                result.result_set = []
            return result

        graph = make_graph_mock([])
        graph.query = AsyncMock(side_effect=router)
        backend._driver._get_graph = MagicMock(return_value=graph)

        result = await backend._repair_duplicate_edge_uuids(group_id='g1')

        assert result == 2

        all_calls = graph.query.call_args_list
        detection_calls = [c for c in all_calls if 'collect(ID(e))' in extract_cypher(c)]
        repair_calls = [c for c in all_calls if 'WHERE ID(e) = $eid' in extract_cypher(c)]

        assert len(detection_calls) == 1
        assert 'group_id' not in extract_cypher(detection_calls[0])
        assert 'group_id' not in extract_params(detection_calls[0])

        assert len(repair_calls) == 2, (
            f'expected one per-edge repair call per re-minted edge, got '
            f'{len(repair_calls)}: {[extract_cypher(c) for c in repair_calls]}'
        )

        seen_new_uuids = set()
        seen_eids = set()
        for call in repair_calls:
            cypher = extract_cypher(call)
            params = extract_params(call)

            assert 'SET' in cypher
            assert 'e.uuid = $new_uuid' in cypher
            assert 'e.superseded_edge_uuid = $old_uuid' in cypher
            assert 'e.uuid = old.uuid' not in cypher
            assert 'group_id' not in cypher
            assert 'group_id' not in params

            eid = params.get('eid')
            assert eid in (102, 103), f'survivor eid 101 must not be repaired, got {eid}'
            seen_eids.add(eid)

            assert params.get('old_uuid') == 'shared-uuid'

            new_uuid = params.get('new_uuid')
            assert new_uuid is not None
            assert uuid.UUID(new_uuid).version == 4
            seen_new_uuids.add(new_uuid)

        assert seen_eids == {102, 103}
        assert len(seen_new_uuids) == 2, 'each re-minted edge must get a DISTINCT fresh uuid'

    @pytest.mark.asyncio
    async def test_clean_graph_no_dup_uuids_is_noop(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """No dup-uuid groups -> return 0 and zero repair queries (only the
        detection query ran)."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)

        result = await backend._repair_duplicate_edge_uuids(group_id='g1')

        assert result == 0
        graph.query.assert_awaited_once()


# ---------------------------------------------------------------------------
# step-5/step-6: GraphitiBackend._run_startup_identity_scan — per-graph orchestrator
# ---------------------------------------------------------------------------

class TestRunStartupIdentityScan:
    """GraphitiBackend._run_startup_identity_scan() — per-graph orchestrator."""

    @pytest.mark.asyncio
    async def test_scans_and_repairs_every_graph_and_aggregates_stats(
        self, mock_config, make_backend,
    ):
        backend = make_backend(mock_config)
        backend.list_graphs = AsyncMock(return_value=['g1', 'g2'])
        backend._scan_duplicate_entity_names = AsyncMock(return_value=[('Foo', 2)])
        backend._repair_duplicate_edge_uuids = AsyncMock(return_value=3)

        stats = await backend._run_startup_identity_scan()

        assert backend._scan_duplicate_entity_names.await_count == 2
        assert backend._repair_duplicate_edge_uuids.await_count == 2
        backend._scan_duplicate_entity_names.assert_any_call('g1')
        backend._scan_duplicate_entity_names.assert_any_call('g2')
        backend._repair_duplicate_edge_uuids.assert_any_call('g1')
        backend._repair_duplicate_edge_uuids.assert_any_call('g2')

        assert stats['graphs_scanned'] == 2
        assert stats['dup_name_groups'] == 2  # one duplicated name per graph, summed
        assert stats['edges_repaired'] == 6  # 3 per graph, summed

    @pytest.mark.asyncio
    async def test_one_graph_failure_does_not_abort_the_sweep(
        self, mock_config, make_backend, caplog,
    ):
        """Per-graph best-effort: a raise on one graph is caught and logged,
        and the sweep still processes the remaining graphs (mirrors the
        initialize() index-setup loop idiom, graphiti_client.py:378-385)."""
        backend = make_backend(mock_config)
        backend.list_graphs = AsyncMock(return_value=['g1', 'g2'])
        backend._scan_duplicate_entity_names = AsyncMock(return_value=[])

        async def repair_side_effect(group_id):
            if group_id == 'g1':
                raise RuntimeError('boom')
            return 3

        backend._repair_duplicate_edge_uuids = AsyncMock(side_effect=repair_side_effect)

        with caplog.at_level(logging.WARNING, logger='fused_memory.backends.graphiti_client'):
            stats = await backend._run_startup_identity_scan()  # must not raise

        assert backend._repair_duplicate_edge_uuids.await_count == 2  # g2 still processed
        assert stats['graphs_scanned'] == 2
        assert stats['edges_repaired'] == 3  # only g2's 3 counted; g1 failed
        assert any(r.levelno >= logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# step-7/step-8: GraphitiBackend.initialize() wiring
# ---------------------------------------------------------------------------

class TestInitializeRunsIdentityScan:
    """initialize() must run the startup identity-integrity scan (task 2210).

    Mirrors the preflight-wiring pattern in
    test_openai_responses_preflight.py:132-158: monkeypatch
    _MultiTenantFalkorDriver / Graphiti to mocks and disable the openai
    api_key so initialize() never attempts a real FalkorDB/LLM connection.
    """

    @pytest.mark.asyncio
    async def test_initialize_awaits_run_startup_identity_scan(
        self, mock_config, monkeypatch,
    ):
        """initialize() must await _run_startup_identity_scan() during startup."""
        import fused_memory.backends.graphiti_client as graphiti_client_module

        mock_config.llm.providers.openai.api_key = ''  # skip the OpenAI preflight/client
        monkeypatch.setattr(
            graphiti_client_module, '_MultiTenantFalkorDriver', MagicMock(),
        )
        monkeypatch.setattr(
            graphiti_client_module, 'Graphiti', MagicMock(return_value=AsyncMock()),
        )

        backend = GraphitiBackend(mock_config)
        backend._run_startup_identity_scan = AsyncMock(return_value={})

        await backend.initialize()

        backend._run_startup_identity_scan.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_survives_scan_failure(
        self, mock_config, monkeypatch,
    ):
        """A total scan failure must never break backend startup — the scan
        is a safety net, not a startup gate."""
        import fused_memory.backends.graphiti_client as graphiti_client_module

        mock_config.llm.providers.openai.api_key = ''
        monkeypatch.setattr(
            graphiti_client_module, '_MultiTenantFalkorDriver', MagicMock(),
        )
        monkeypatch.setattr(
            graphiti_client_module, 'Graphiti', MagicMock(return_value=AsyncMock()),
        )

        backend = GraphitiBackend(mock_config)
        backend._run_startup_identity_scan = AsyncMock(side_effect=RuntimeError('boom'))

        await backend.initialize()  # must not raise


# ---------------------------------------------------------------------------
# step-9: live-FalkorDB semantic pins — authoritative B5+B6 signal
# ---------------------------------------------------------------------------
#
# The mock tests above can only pin the Cypher string/params shape, not
# FalkorDB's actual count(*) grouping, its enforcement of per-uuid
# uniqueness, or whether ID(e) enumeration + `WHERE ID(e) = $eid` keying
# genuinely rewrites the intended edge while leaving the rest untouched.
# These tests pin the true B5/B6 semantics against a REAL FalkorDB server,
# mirroring test_merge_entities.py:1021-1131 (task 2207 W6-δ's own live
# pin for the sibling redirect_node_edges fix).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _falkor_available(), reason='FalkorDB not reachable')
@pytest.mark.timeout(15)
class TestScanDuplicateEntityNamesLiveFalkorDB:
    """Pin the true B5 dup-node alarm semantics against a REAL FalkorDB
    server: real count(*) grouping by name, and the `n.group_id =
    $group_id` property predicate genuinely excludes a foreign-group_id
    clone physically stored in the same graph key (the task-2115 leak
    scenario) from the alarm."""

    @pytest.mark.asyncio
    async def test_scan_finds_real_duplicates_excludes_unique_and_foreign_group(
        self, mock_config, startup_scan_live_graph, caplog,
    ):
        graph_name, graph = startup_scan_live_graph

        # Two same-group 'Foo' nodes (a real duplicate), one unique 'Bar',
        # and a foreign-group_id 'Foo' clone physically planted in this same
        # graph key (the task-2115 leak scenario) that must NOT count.
        await graph.query(
            "CREATE "
            "(:Entity {uuid: 'foo1', name: 'Foo', group_id: $gid}), "
            "(:Entity {uuid: 'foo2', name: 'Foo', group_id: $gid}), "
            "(:Entity {uuid: 'bar1', name: 'Bar', group_id: $gid}), "
            "(:Entity {uuid: 'foreign1', name: 'Foo', group_id: 'foreign'})",
            {'gid': graph_name},
        )

        backend = GraphitiBackend(mock_config)
        backend._driver = _MultiTenantFalkorDriver(host=FALKOR_HOST, port=FALKOR_PORT)
        try:
            with caplog.at_level(logging.WARNING, logger='fused_memory.backends.graphiti_client'):
                result = await backend._scan_duplicate_entity_names(graph_name)

            # Real count(*) grouping: only the same-group 'Foo' pair counts.
            # 'Bar' is unique (excluded); the foreign-group 'Foo' clone is
            # excluded by the `n.group_id = $group_id` predicate.
            assert result == [('Foo', 2)]

            warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert any(graph_name in r.message and 'Foo' in r.message for r in warnings), (
                f'expected a WARNING naming the group and duplicated name, got: '
                f'{[r.message for r in warnings]}'
            )
        finally:
            await backend.close()


@pytest.mark.skipif(not _falkor_available(), reason='FalkorDB not reachable')
@pytest.mark.timeout(15)
class TestRepairDuplicateEdgeUuidsLiveFalkorDB:
    """Pin the true B6 dup-uuid-edge repair semantics against a REAL
    FalkorDB server: after repair, no RELATES_TO uuid is shared by more than
    one edge, the re-minted edge carries a superseded_edge_uuid audit prop
    equal to the uuid it replaced, no edges are lost, fact_embedding
    survives intact, and a second call is a true no-op (idempotent).

    Seeds a PRE-EXISTING dup-uuid pair between two DIFFERENT node pairs --
    the discriminating scenario for this repair (a clean distinct-uuid seed
    would already be graph-wide unique and would not exercise the repair
    path at all).
    """

    @pytest.mark.asyncio
    async def test_repair_fixes_real_dup_uuids_no_loss_preserves_embeddings_idempotent(
        self, mock_config, startup_scan_live_graph,
    ):
        graph_name, graph = startup_scan_live_graph

        await graph.query(
            "CREATE (:Entity {uuid: 'a', name: 'A'}), "
            "(:Entity {uuid: 'b', name: 'B'}), "
            "(:Entity {uuid: 'c', name: 'C'}), "
            "(:Entity {uuid: 'd', name: 'D'})"
        )

        async def seed_edge(src, dst, edge_uuid, fact, embedding):
            await graph.query(
                'MATCH (a:Entity {uuid: $src}), (b:Entity {uuid: $dst}) '
                'CREATE (a)-[e:RELATES_TO {uuid: $edge_uuid, name: $name, fact: $fact}]->(b) '
                'SET e.fact_embedding = vecf32($embedding)',
                {
                    'src': src, 'dst': dst, 'edge_uuid': edge_uuid,
                    'name': 'rel', 'fact': fact, 'embedding': embedding,
                },
            )

        # One distinct-uuid edge (must be left untouched) plus a
        # PRE-EXISTING dup-uuid pair ('dup') between two DIFFERENT node
        # pairs.
        await seed_edge('a', 'b', 'distinct1', 'a relates to b', [1.0, 2.0])
        await seed_edge('c', 'd', 'dup', 'c relates to d', [3.0, 4.0])
        await seed_edge('a', 'd', 'dup', 'a relates to d', [5.0, 6.0])

        backend = GraphitiBackend(mock_config)
        backend._driver = _MultiTenantFalkorDriver(host=FALKOR_HOST, port=FALKOR_PORT)
        try:
            result = await backend._repair_duplicate_edge_uuids(graph_name)
            # (a) One dup-uuid group of size 2 -> re-mint 1, keep 1 survivor.
            assert result == 1

            # (b) Graph-wide: no uuid is shared by more than one RELATES_TO
            # edge anymore.
            dup_check = await graph.query(
                'MATCH ()-[e:RELATES_TO]->() '
                'WITH e.uuid AS u, count(*) AS c '
                'WHERE c > 1 '
                'RETURN count(u)'
            )
            assert dup_check.result_set[0][0] == 0

            # (c) Edge set preserved: still exactly 3 RELATES_TO edges (none
            # lost or merged by the repair).
            all_edges = await graph.query(
                'MATCH ()-[e:RELATES_TO]->() '
                'RETURN e.uuid, e.superseded_edge_uuid, e.fact_embedding, e.fact'
            )
            rows = all_edges.result_set
            assert len(rows) == 3

            # (d) The re-minted edge carries superseded_edge_uuid == 'dup'
            # and its fact_embedding survived intact.
            reminted = [r for r in rows if r[1] == 'dup']
            assert len(reminted) == 1
            new_uuid, superseded, embedding, fact = reminted[0]
            assert new_uuid != 'dup'
            assert fact in ('c relates to d', 'a relates to d')
            expected_embedding = [3.0, 4.0] if fact == 'c relates to d' else [5.0, 6.0]
            assert list(embedding) == pytest.approx(expected_embedding)

            # The dup group's survivor kept its original uuid untouched...
            survivor = [r for r in rows if r[0] == 'dup']
            assert len(survivor) == 1
            assert survivor[0][1] is None  # never repaired -> no superseded_edge_uuid

            # ...and the unrelated distinct-uuid edge is unaffected.
            distinct = [r for r in rows if r[0] == 'distinct1']
            assert len(distinct) == 1
            assert distinct[0][1] is None
            assert list(distinct[0][2]) == pytest.approx([1.0, 2.0])

            # (e) Idempotent: a second call finds nothing left to repair.
            result2 = await backend._repair_duplicate_edge_uuids(graph_name)
            assert result2 == 0

            dup_check2 = await graph.query(
                'MATCH ()-[e:RELATES_TO]->() '
                'WITH e.uuid AS u, count(*) AS c '
                'WHERE c > 1 '
                'RETURN count(u)'
            )
            assert dup_check2.result_set[0][0] == 0
        finally:
            await backend.close()
