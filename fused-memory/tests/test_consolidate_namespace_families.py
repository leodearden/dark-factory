"""Tests for scripts/consolidate_namespace_families.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution -- mirrors the pattern in
test_purge_knowlive_namespace.py.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'consolidate_namespace_families.py'


def _load_module() -> types.ModuleType:
    """Load consolidate_namespace_families.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators work correctly.
    """
    mod_name = 'consolidate_namespace_families'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()


# ===========================================================================
# Helpers
# ===========================================================================

def _make_graph_mock(rows: list[list] | None = None) -> MagicMock:
    """Minimal stand-in for conftest.make_graph_mock, scoped to this test
    module so it is usable without the fixture (kept consistent with its
    shape)."""
    result = MagicMock()
    result.result_set = rows if rows is not None else []
    graph = MagicMock()
    graph.ro_query = AsyncMock(return_value=result)
    graph.query = AsyncMock(return_value=result)
    graph.delete = AsyncMock(return_value=None)
    return graph


def _make_qdrant_mock(points: list | None = None) -> AsyncMock:
    """AsyncMock stand-in for an AsyncQdrantClient exposing scroll/upsert/
    delete_collection -- the raw transport consolidate_namespace_families
    reaches via memory.mem0._get_async_qdrant()."""
    client = AsyncMock()
    client.scroll = AsyncMock(return_value=(points if points is not None else [], None))
    client.upsert = AsyncMock(return_value=None)
    client.delete_collection = AsyncMock(return_value=None)
    return client


def _make_point(
    point_id: str,
    payload: dict | None = None,
    vector: list[float] | None = None,
) -> MagicMock:
    """Build a Qdrant-scroll-shaped point stand-in (id/payload/vector)."""
    point = MagicMock()
    point.id = point_id
    point.payload = payload if payload is not None else {}
    point.vector = vector if vector is not None else [0.1, 0.2, 0.3]
    return point


# ===========================================================================
# Tests: reviewable-config constants
# ===========================================================================

class TestGraphFamilyAliases:
    """Tests for the module constant GRAPH_FAMILY_ALIASES."""

    def test_maps_siblings_to_underscore_canonical(self):
        """Hyphenated/no-separator siblings map to the underscore-canonical key."""
        assert _mod.GRAPH_FAMILY_ALIASES['know-live'] == 'know_live'
        assert _mod.GRAPH_FAMILY_ALIASES['knowlive'] == 'know_live'
        assert _mod.GRAPH_FAMILY_ALIASES['pump-web-ui'] == 'pump_web_ui'

    def test_excludes_solar_family(self):
        """PRD Open Q1 default is keep-separate: no solar-family key/value
        appears anywhere in the alias map (neither as a sibling key nor as a
        canonical target)."""
        solar_names = {'my_solar_challenge', 'solar_challenge_platform'}
        assert not (solar_names & set(_mod.GRAPH_FAMILY_ALIASES.keys()))
        assert not (solar_names & set(_mod.GRAPH_FAMILY_ALIASES.values()))


class TestCollectionMerges:
    """Tests for the module constant COLLECTION_MERGES."""

    def test_maps_legacy_sources_to_fused_project_targets(self):
        """Representative legacy/divergent sources map to their fused_<project> target."""
        assert _mod.COLLECTION_MERGES['fused_dark-factory'] == 'fused_dark_factory'
        assert _mod.COLLECTION_MERGES['reify_reify'] == 'fused_reify'
        assert _mod.COLLECTION_MERGES['autopilot_video_autopilot_video'] == 'fused_autopilot_video'

    def test_does_not_auto_merge_ambiguous_collections(self):
        """PRD Open Q2 defers reify_ (empty project id) and fused_fused_memory
        to ι human review -- neither is a key in COLLECTION_MERGES."""
        assert 'reify_' not in _mod.COLLECTION_MERGES
        assert 'fused_fused_memory' not in _mod.COLLECTION_MERGES


class TestJunkKeys:
    """Tests for the module constant JUNK_KEYS."""

    def test_includes_the_six_explicit_keys(self):
        """JUNK_KEYS includes every explicitly-named junk graph key."""
        expected = {
            'dark-factory', '-home-leo-src-dark-factory',
            'my-project', 'test-project', 'default', '1098',
        }
        assert expected <= set(_mod.JUNK_KEYS)


# ===========================================================================
# Tests: build_consolidation_report
# ===========================================================================

class TestBuildConsolidationReport:
    """Tests for the pure function build_consolidation_report(...)."""

    def test_report_shape(self):
        """Returned dict has exactly the four top-level manifest keys."""
        report = _mod.build_consolidation_report([], [], [], dry_run=True)

        assert set(report.keys()) == {
            'dry_run', 'graph_family_merges', 'collection_merges', 'junk_key_deletions',
        }

    def test_dry_run_passed_through_verbatim(self):
        """dry_run is passed through verbatim, not hardcoded, in either direction."""
        report_true = _mod.build_consolidation_report([], [], [], dry_run=True)
        report_false = _mod.build_consolidation_report([], [], [], dry_run=False)

        assert report_true['dry_run'] is True
        assert report_false['dry_run'] is False

    def test_sections_are_lists_of_given_items(self):
        """Each section is exactly the list of per-item dicts given, in order."""
        graph_items = [{'sibling': 'know-live', 'canonical': 'know_live', 'disposition': 'MERGE'}]
        collection_items = [{'source': 'reify_reify', 'target': 'fused_reify', 'disposition': 'MERGE'}]
        junk_items = [{'key': 'my-project', 'node_count': 0, 'disposition': 'DELETE'}]

        report = _mod.build_consolidation_report(
            graph_items, collection_items, junk_items, dry_run=True,
        )

        assert report['graph_family_merges'] == graph_items
        assert report['collection_merges'] == collection_items
        assert report['junk_key_deletions'] == junk_items

    def test_empty_inputs_produce_empty_lists(self):
        """Empty inputs for all three sections produce empty lists, with dry_run preserved."""
        report = _mod.build_consolidation_report([], [], [], dry_run=False)

        assert report['graph_family_merges'] == []
        assert report['collection_merges'] == []
        assert report['junk_key_deletions'] == []
        assert report['dry_run'] is False

    def test_no_io_pure_function(self):
        """build_consolidation_report performs no I/O -- calling twice with
        identical inputs is referentially stable in content."""
        graph_items = [{'sibling': 'knowlive', 'canonical': 'know_live', 'disposition': 'MERGE'}]
        collection_items = [{'source': 'fused_dark-factory', 'target': 'fused_dark_factory'}]
        junk_items = [{'key': 'default', 'node_count': 0, 'disposition': 'DELETE'}]

        r1 = _mod.build_consolidation_report(graph_items, collection_items, junk_items, dry_run=True)
        r2 = _mod.build_consolidation_report(graph_items, collection_items, junk_items, dry_run=True)

        assert r1 == r2


# ===========================================================================
# Tests: has_unresolved
# ===========================================================================

class TestHasUnresolved:
    """Tests for the pure function has_unresolved(report) -> bool."""

    def test_all_clean_empty_sections_returns_false(self):
        """An all-clean report with empty sections returns False."""
        report = _mod.build_consolidation_report([], [], [], dry_run=True)

        assert _mod.has_unresolved(report) is False

    def test_all_clean_merge_and_delete_dispositions_returns_false(self):
        """Sections containing only MERGE/DELETE dispositions (no
        UNRESOLVED) return False."""
        graph_items = [{'sibling': 'know-live', 'canonical': 'know_live', 'disposition': 'MERGE'}]
        collection_items = [{'source': 'reify_reify', 'target': 'fused_reify', 'disposition': 'MERGE'}]
        junk_items = [{'key': 'my-project', 'node_count': 0, 'disposition': 'DELETE'}]
        report = _mod.build_consolidation_report(
            graph_items, collection_items, junk_items, dry_run=True,
        )

        assert _mod.has_unresolved(report) is False

    def test_unresolved_in_graph_family_merges_returns_true(self):
        """A single UNRESOLVED disposition in graph_family_merges returns True."""
        graph_items = [{'sibling': 'know-live', 'canonical': 'know_live', 'disposition': 'UNRESOLVED'}]
        report = _mod.build_consolidation_report(graph_items, [], [], dry_run=True)

        assert _mod.has_unresolved(report) is True

    def test_unresolved_in_collection_merges_returns_true(self):
        """A single UNRESOLVED disposition in collection_merges returns True."""
        collection_items = [
            {'source': 'fused_dark-factory', 'target': 'fused_dark_factory', 'disposition': 'UNRESOLVED'},
        ]
        report = _mod.build_consolidation_report([], collection_items, [], dry_run=True)

        assert _mod.has_unresolved(report) is True

    def test_unresolved_in_junk_key_deletions_returns_true(self):
        """A single UNRESOLVED disposition in junk_key_deletions returns True."""
        junk_items = [{'key': 'my-project', 'node_count': 3, 'disposition': 'UNRESOLVED'}]
        report = _mod.build_consolidation_report([], [], junk_items, dry_run=True)

        assert _mod.has_unresolved(report) is True

    def test_mixed_report_with_one_unresolved_returns_true(self):
        """A mixed report (several MERGE/DELETE items plus one UNRESOLVED
        anywhere) returns True."""
        graph_items = [
            {'sibling': 'know-live', 'canonical': 'know_live', 'disposition': 'MERGE'},
            {'sibling': 'pump-web-ui', 'canonical': 'pump_web_ui', 'disposition': 'MERGE'},
        ]
        collection_items = [
            {'source': 'reify_reify', 'target': 'fused_reify', 'disposition': 'MERGE'},
        ]
        junk_items = [
            {'key': 'default', 'node_count': 0, 'disposition': 'DELETE'},
            {'key': 'my-project', 'node_count': 3, 'disposition': 'UNRESOLVED'},
        ]
        report = _mod.build_consolidation_report(
            graph_items, collection_items, junk_items, dry_run=False,
        )

        assert _mod.has_unresolved(report) is True


# ===========================================================================
# Tests: rewrite_point_payload_user_id / canonical_user_id_for
# ===========================================================================

class TestRewritePointPayloadUserId:
    """Tests for the pure function rewrite_point_payload_user_id(payload, canonical_user_id)."""

    def test_sets_user_id_to_canonical(self):
        """Returned payload carries the canonical user_id."""
        payload = {'user_id': 'dark-factory', 'data': 'hello'}

        result = _mod.rewrite_point_payload_user_id(payload, 'dark_factory')

        assert result['user_id'] == 'dark_factory'

    def test_preserves_other_keys_unchanged(self):
        """Every other payload key is preserved unchanged."""
        payload = {'user_id': 'reify', 'data': 'hello', 'metadata': {'a': 1}, 'created_at': 'x'}

        result = _mod.rewrite_point_payload_user_id(payload, 'reify')

        assert result['data'] == 'hello'
        assert result['metadata'] == {'a': 1}
        assert result['created_at'] == 'x'

    def test_does_not_mutate_input(self):
        """The input payload dict is not mutated in place."""
        payload = {'user_id': 'autopilot_video_autopilot_video', 'data': 'hello'}
        original = dict(payload)

        _mod.rewrite_point_payload_user_id(payload, 'autopilot_video')

        assert payload == original


class TestCanonicalUserIdFor:
    """Tests for the pure function canonical_user_id_for(target_collection)."""

    def test_derives_project_id_from_fused_target(self):
        """The canonical user_id is the target collection with the fused_ prefix stripped."""
        assert _mod.canonical_user_id_for('fused_dark_factory') == 'dark_factory'
        assert _mod.canonical_user_id_for('fused_reify') == 'reify'
        assert _mod.canonical_user_id_for('fused_autopilot_video') == 'autopilot_video'


# ===========================================================================
# Tests: enumerate_graph_entity_nodes / count_graph_nodes
# ===========================================================================

class TestEnumerateGraphEntityNodes:
    """Tests for async enumerate_graph_entity_nodes(graphiti, key, *, limit)."""

    @pytest.mark.asyncio
    async def test_calls_graph_for_with_key(self):
        """graphiti._graph_for is called with the key argument."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        await _mod.enumerate_graph_entity_nodes(graphiti, 'know-live', limit=1000)

        graphiti._graph_for.assert_called_once_with('know-live')

    @pytest.mark.asyncio
    async def test_uses_ro_query_not_query(self):
        """Enumeration is read-only: ro_query is used, .query is NEVER called."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        await _mod.enumerate_graph_entity_nodes(graphiti, 'know-live', limit=1000)

        graph.ro_query.assert_called_once()
        graph.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_matches_entity_label(self):
        """The Cypher issued matches :Entity nodes specifically."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        await _mod.enumerate_graph_entity_nodes(graphiti, 'know-live', limit=1000)

        cypher = graph.ro_query.call_args.args[0]
        assert ':Entity' in cypher

    @pytest.mark.asyncio
    async def test_normalizes_rows_to_uuid_name_dicts(self):
        """result_set rows [uuid, name] are normalized to [{'uuid','name'}, ...]."""
        graphiti = MagicMock()
        rows = [['uuid-1', 'Node A'], ['uuid-2', 'Node B']]
        graph = _make_graph_mock(rows)
        graphiti._graph_for = MagicMock(return_value=graph)

        result = await _mod.enumerate_graph_entity_nodes(graphiti, 'know-live', limit=1000)

        assert result == [
            {'uuid': 'uuid-1', 'name': 'Node A'},
            {'uuid': 'uuid-2', 'name': 'Node B'},
        ]

    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty_list(self):
        """An empty result_set returns an empty list, not an error."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        result = await _mod.enumerate_graph_entity_nodes(graphiti, 'know-live', limit=1000)

        assert result == []

    @pytest.mark.asyncio
    async def test_warns_when_row_count_hits_limit(self, caplog):
        """No-silent-caps: hitting the limit logs a WARNING."""
        graphiti = MagicMock()
        rows = [[f'uuid-{i}', f'Node {i}'] for i in range(3)]
        graph = _make_graph_mock(rows)
        graphiti._graph_for = MagicMock(return_value=graph)

        with caplog.at_level('WARNING'):
            await _mod.enumerate_graph_entity_nodes(graphiti, 'know-live', limit=3)

        assert any('limit' in rec.message.lower() for rec in caplog.records), (
            f'Expected a limit-related WARNING, got: {[r.message for r in caplog.records]}'
        )

    @pytest.mark.asyncio
    async def test_no_warning_when_under_limit(self, caplog):
        """Row count below the limit does not log a WARNING."""
        graphiti = MagicMock()
        rows = [['uuid-1', 'Node A']]
        graph = _make_graph_mock(rows)
        graphiti._graph_for = MagicMock(return_value=graph)

        with caplog.at_level('WARNING'):
            await _mod.enumerate_graph_entity_nodes(graphiti, 'know-live', limit=1000)

        assert caplog.records == []


class TestCountGraphNodes:
    """Tests for async count_graph_nodes(graphiti, key)."""

    @pytest.mark.asyncio
    async def test_calls_graph_for_with_key(self):
        """graphiti._graph_for is called with the key argument."""
        graphiti = MagicMock()
        graph = _make_graph_mock([[0]])
        graphiti._graph_for = MagicMock(return_value=graph)

        await _mod.count_graph_nodes(graphiti, 'my-project')

        graphiti._graph_for.assert_called_once_with('my-project')

    @pytest.mark.asyncio
    async def test_uses_ro_query_not_query(self):
        """Counting is read-only: ro_query is used, .query is NEVER called."""
        graphiti = MagicMock()
        graph = _make_graph_mock([[0]])
        graphiti._graph_for = MagicMock(return_value=graph)

        await _mod.count_graph_nodes(graphiti, 'my-project')

        graph.ro_query.assert_called_once()
        graph.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_integer_count(self):
        """Returns the scalar count(n) value as an int."""
        graphiti = MagicMock()
        graph = _make_graph_mock([[7]])
        graphiti._graph_for = MagicMock(return_value=graph)

        result = await _mod.count_graph_nodes(graphiti, 'my-project')

        assert result == 7

    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_graph(self):
        """A graph with no nodes returns 0."""
        graphiti = MagicMock()
        graph = _make_graph_mock([[0]])
        graphiti._graph_for = MagicMock(return_value=graph)

        result = await _mod.count_graph_nodes(graphiti, 'default')

        assert result == 0


# ===========================================================================
# Tests: merge_graph_family
# ===========================================================================

class TestMergeGraphFamily:
    """Tests for async merge_graph_family(graphiti, sibling, canonical, node_rows)."""

    @pytest.mark.asyncio
    async def test_calls_move_once_per_node_with_rewrite_group_id(self, monkeypatch):
        """move_entity_across_graphs is called once per node row, with
        source=sibling, target=canonical, and rewrite_group_id=canonical --
        the Phase-2 identity rewrite (PRD decision 6)."""
        node_rows = [{'uuid': 'uuid-1', 'name': 'A'}, {'uuid': 'uuid-2', 'name': 'B'}]
        move_mock = AsyncMock(side_effect=[
            _mod.MoveResult(uuid='uuid-1', source_graph='know-live', target_graph='know_live'),
            _mod.MoveResult(uuid='uuid-2', source_graph='know-live', target_graph='know_live'),
        ])
        monkeypatch.setattr(_mod, 'move_entity_across_graphs', move_mock)
        graphiti = MagicMock()

        await _mod.merge_graph_family(graphiti, 'know-live', 'know_live', node_rows)

        assert move_mock.call_count == 2
        for call, row in zip(move_mock.call_args_list, node_rows, strict=True):
            assert call.args == (graphiti, row['uuid'], 'know-live', 'know_live')
            assert call.kwargs == {'rewrite_group_id': 'know_live'}

    @pytest.mark.asyncio
    async def test_tallies_move_results(self, monkeypatch):
        """The MoveResults' edges/mentions counters are summed, and
        nodes_moved reflects the number of nodes processed."""
        node_rows = [{'uuid': 'uuid-1', 'name': 'A'}, {'uuid': 'uuid-2', 'name': 'B'}]
        move_mock = AsyncMock(side_effect=[
            _mod.MoveResult(
                uuid='uuid-1', source_graph='know-live', target_graph='know_live',
                edges_moved=2, edges_skipped=1, mentions_moved=3, mentions_skipped=0,
            ),
            _mod.MoveResult(
                uuid='uuid-2', source_graph='know-live', target_graph='know_live',
                edges_moved=1, edges_skipped=0, mentions_moved=0, mentions_skipped=1,
            ),
        ])
        monkeypatch.setattr(_mod, 'move_entity_across_graphs', move_mock)
        graphiti = MagicMock()

        summary = await _mod.merge_graph_family(graphiti, 'know-live', 'know_live', node_rows)

        assert summary['nodes_moved'] == 2
        assert summary['edges_moved'] == 3
        assert summary['edges_skipped'] == 1
        assert summary['mentions_moved'] == 3
        assert summary['mentions_skipped'] == 1

    @pytest.mark.asyncio
    async def test_empty_node_rows_returns_zeroed_summary_without_calling_move(self, monkeypatch):
        """No node rows -> move_entity_across_graphs is never called, and the
        summary is all-zero."""
        move_mock = AsyncMock()
        monkeypatch.setattr(_mod, 'move_entity_across_graphs', move_mock)
        graphiti = MagicMock()

        summary = await _mod.merge_graph_family(graphiti, 'know-live', 'know_live', [])

        move_mock.assert_not_called()
        assert summary == {
            'nodes_moved': 0, 'edges_moved': 0, 'edges_skipped': 0,
            'mentions_moved': 0, 'mentions_skipped': 0,
        }


# ===========================================================================
# Tests: scroll_collection_points
# ===========================================================================

class TestScrollCollectionPoints:
    """Tests for async scroll_collection_points(qdrant_client, collection, *, limit)."""

    @pytest.mark.asyncio
    async def test_calls_scroll_with_payload_and_vectors(self):
        """scroll is called with with_payload=True AND with_vectors=True --
        omitting with_vectors would silently drop embeddings."""
        client = _make_qdrant_mock([])

        await _mod.scroll_collection_points(client, 'fused_dark-factory', limit=1000)

        client.scroll.assert_called_once_with(
            collection_name='fused_dark-factory',
            with_payload=True,
            with_vectors=True,
            limit=1000,
        )

    @pytest.mark.asyncio
    async def test_returns_the_points(self):
        """Returns the points list from the scroll result."""
        points = [_make_point('p1'), _make_point('p2')]
        client = _make_qdrant_mock(points)

        result = await _mod.scroll_collection_points(client, 'reify_reify', limit=1000)

        assert result == points

    @pytest.mark.asyncio
    async def test_warns_when_point_count_hits_limit(self, caplog):
        """No-silent-caps: hitting the limit logs a WARNING."""
        points = [_make_point(f'p{i}') for i in range(3)]
        client = _make_qdrant_mock(points)

        with caplog.at_level('WARNING'):
            await _mod.scroll_collection_points(client, 'reify_reify', limit=3)

        assert any('limit' in rec.message.lower() for rec in caplog.records), (
            f'Expected a limit-related WARNING, got: {[r.message for r in caplog.records]}'
        )

    @pytest.mark.asyncio
    async def test_no_warning_when_under_limit(self, caplog):
        """Point count below the limit does not log a WARNING."""
        points = [_make_point('p1')]
        client = _make_qdrant_mock(points)

        with caplog.at_level('WARNING'):
            await _mod.scroll_collection_points(client, 'reify_reify', limit=1000)

        assert caplog.records == []


# ===========================================================================
# Tests: merge_collection
# ===========================================================================

class TestMergeCollection:
    """Tests for async merge_collection(qdrant_client, source, target,
    canonical_user_id, points, *, capped)."""

    @pytest.mark.asyncio
    async def test_upserts_into_target_with_canonical_user_id(self):
        """upsert is called with collection_name=target and each point's
        payload user_id rewritten to canonical -- original id/vector preserved."""
        points = [
            _make_point('p1', payload={'user_id': 'dark-factory', 'data': 'a'}, vector=[0.1, 0.2]),
            _make_point('p2', payload={'user_id': 'dark-factory', 'data': 'b'}, vector=[0.3, 0.4]),
        ]
        client = _make_qdrant_mock()

        await _mod.merge_collection(
            client, 'fused_dark-factory', 'fused_dark_factory', 'dark_factory',
            points, capped=False,
        )

        client.upsert.assert_called_once()
        upsert_kwargs = client.upsert.call_args.kwargs
        assert upsert_kwargs['collection_name'] == 'fused_dark_factory'
        upserted = upsert_kwargs['points']
        assert len(upserted) == 2
        assert {p.id for p in upserted} == {'p1', 'p2'}
        for p in upserted:
            assert p.payload['user_id'] == 'dark_factory'
        by_id = {p.id: p for p in upserted}
        assert by_id['p1'].vector == [0.1, 0.2]
        assert by_id['p1'].payload['data'] == 'a'
        assert by_id['p2'].vector == [0.3, 0.4]

    @pytest.mark.asyncio
    async def test_deletes_source_when_not_capped(self):
        """A fully-drained (not capped) scroll deletes the source collection."""
        client = _make_qdrant_mock()
        points = [_make_point('p1')]

        await _mod.merge_collection(
            client, 'reify_reify', 'fused_reify', 'reify', points, capped=False,
        )

        client.delete_collection.assert_called_once_with('reify_reify')

    @pytest.mark.asyncio
    async def test_does_not_delete_source_when_capped(self):
        """A capped (possibly-incomplete) scroll does NOT delete the source
        collection -- no data loss on a partial migration."""
        client = _make_qdrant_mock()
        points = [_make_point('p1')]

        await _mod.merge_collection(
            client, 'reify_reify', 'fused_reify', 'reify', points, capped=True,
        )

        client.delete_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_summary_dict(self):
        """Returns a summary with points_upserted count and source_deleted flag."""
        client = _make_qdrant_mock()
        points = [_make_point('p1'), _make_point('p2')]

        result = await _mod.merge_collection(
            client, 'reify_reify', 'fused_reify', 'reify', points, capped=False,
        )

        assert result == {'points_upserted': 2, 'source_deleted': True}

    @pytest.mark.asyncio
    async def test_capped_summary_reports_source_not_deleted(self):
        """capped=True summary reports source_deleted=False."""
        client = _make_qdrant_mock()
        points = [_make_point('p1')]

        result = await _mod.merge_collection(
            client, 'reify_reify', 'fused_reify', 'reify', points, capped=True,
        )

        assert result == {'points_upserted': 1, 'source_deleted': False}


# ===========================================================================
# Tests: delete_junk_key
# ===========================================================================

class TestDeleteJunkKey:
    """Tests for async delete_junk_key(graphiti, key, node_count)."""

    @pytest.mark.asyncio
    async def test_calls_graph_for_with_exact_key(self):
        """graphiti._graph_for is called with the exact key argument."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        await _mod.delete_junk_key(graphiti, 'my-project', 0)

        graphiti._graph_for.assert_called_once_with('my-project')

    @pytest.mark.asyncio
    async def test_zero_count_deletes_and_returns_delete_disposition(self):
        """node_count==0 -> GRAPH.DELETE via graph.delete(), disposition DELETE."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        disposition = await _mod.delete_junk_key(graphiti, 'my-project', 0)

        graph.delete.assert_called_once()
        assert disposition == 'DELETE'

    @pytest.mark.asyncio
    async def test_nonzero_count_does_not_delete_and_returns_unresolved(self):
        """node_count>0 -> .delete() is NEVER called, disposition UNRESOLVED
        (deletion blocked -- no data loss)."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        disposition = await _mod.delete_junk_key(graphiti, 'my-project', 5)

        graph.delete.assert_not_called()
        assert disposition == 'UNRESOLVED'


# ===========================================================================
# Helpers: run() fixtures
#
# run() queries the SAME graph key for two distinct purposes -- entity
# enumeration (a family sibling) and total-node counting (every junk-key
# candidate, INCLUDING that same sibling, per the "plus emptied family
# siblings" junk-key policy) -- so a single graph mock must answer both
# ro_query shapes correctly. These helpers are shared by TestRunDryRun and
# TestRunApply.
# ===========================================================================

def _make_run_graph_mock(
    entity_rows: list[list] | None = None,
    total_count: int | None = None,
) -> MagicMock:
    """Graph mock whose ro_query dispatches on the Cypher text: the
    Entity-enumeration query returns *entity_rows*, the total-count query
    returns [[*total_count*]]. Defaults total_count to len(entity_rows) when
    not given explicitly."""
    entity_rows = entity_rows if entity_rows is not None else []
    if total_count is None:
        total_count = len(entity_rows)

    async def _ro_query(cypher: str, params: dict | None = None):
        result = MagicMock()
        result.result_set = entity_rows if 'Entity' in cypher else [[total_count]]
        return result

    graph = MagicMock()
    graph.ro_query = AsyncMock(side_effect=_ro_query)
    graph.query = AsyncMock()
    graph.delete = AsyncMock(return_value=None)
    return graph


def _make_run_graphiti_mock(
    entity_rows_by_key: dict[str, list] | None = None,
    total_count_by_key: dict[str, int] | None = None,
) -> MagicMock:
    """MagicMock graphiti whose _graph_for(key) MEMOIZES one graph mock per
    key -- the same mock instance is returned across repeat calls for a key
    -- configured via _make_run_graph_mock. Keys without an explicit fixture
    default to an empty/zero-count graph. The per-key mocks are exposed via
    ._graphs_by_key for post-call assertions (e.g. .delete/.query
    not-called)."""
    entity_rows_by_key = entity_rows_by_key or {}
    total_count_by_key = total_count_by_key or {}
    graphs: dict[str, MagicMock] = {}

    def _graph_for(key: str) -> MagicMock:
        if key not in graphs:
            rows = entity_rows_by_key.get(key, [])
            count = total_count_by_key.get(key, len(rows))
            graphs[key] = _make_run_graph_mock(rows, count)
        return graphs[key]

    graphiti = MagicMock()
    graphiti._graph_for = MagicMock(side_effect=_graph_for)
    graphiti._graphs_by_key = graphs
    return graphiti


def _make_run_qdrant_mock(points_by_collection: dict[str, list] | None = None) -> AsyncMock:
    """AsyncMock qdrant client whose scroll() dispatches on collection_name;
    unlisted collections scroll as empty."""
    points_by_collection = points_by_collection or {}

    async def _scroll(*, collection_name: str, **_kwargs):
        return (points_by_collection.get(collection_name, []), None)

    client = AsyncMock()
    client.scroll = AsyncMock(side_effect=_scroll)
    client.upsert = AsyncMock(return_value=None)
    client.delete_collection = AsyncMock(return_value=None)
    return client


def _make_run_memory_service(graphiti: MagicMock, qdrant_client: AsyncMock) -> MagicMock:
    """MagicMock memory_service wired the way run() consumes it: .graphiti
    directly, and the raw Qdrant transport via .mem0._get_async_qdrant()
    (PRD reuse note: mem0_client.py's _get_async_qdrant)."""
    memory_service = MagicMock()
    memory_service.graphiti = graphiti
    memory_service.mem0 = MagicMock()
    memory_service.mem0._get_async_qdrant = AsyncMock(return_value=qdrant_client)
    return memory_service


def _run_args(apply: bool = False, **overrides):
    """SimpleNamespace stand-in for argparse.Namespace, mirroring the
    purge_knowlive_namespace test suite's _args() helper."""
    import types as _types
    base = {'apply': apply}
    base.update(overrides)
    return _types.SimpleNamespace(**base)


# ===========================================================================
# Tests: run() -- dry-run path
# ===========================================================================

class TestRunDryRun:
    """Tests for async run(args, memory_service, *, limit) in dry-run (args.apply=False)."""

    def _scenario(self):
        """One sibling with entity rows (know-live, unmerged); one non-empty
        JUNK_KEYS entry (my-project); one collection with points
        (fused_dark-factory). Every other alias/collection/key falls back to
        the empty/zero-count default."""
        graphiti = _make_run_graphiti_mock(
            entity_rows_by_key={'know-live': [['uuid-1', 'Node A']]},
            total_count_by_key={'my-project': 3},
        )
        qdrant_client = _make_run_qdrant_mock(
            points_by_collection={
                'fused_dark-factory': [_make_point('p1', payload={'user_id': 'dark-factory'})],
            },
        )
        memory_service = _make_run_memory_service(graphiti, qdrant_client)
        return memory_service, graphiti, qdrant_client

    @pytest.mark.asyncio
    async def test_dry_run_touches_nothing(self, monkeypatch):
        """Dry-run performs ZERO mutations across every backend: no
        move_entity_across_graphs, no qdrant upsert/delete_collection, no
        GRAPH.DELETE, and read-only ro_query only (.query is NEVER called
        on any graph)."""
        move_mock = AsyncMock()
        monkeypatch.setattr(_mod, 'move_entity_across_graphs', move_mock)
        memory_service, graphiti, qdrant_client = self._scenario()

        await _mod.run(_run_args(apply=False), memory_service, limit=1000)

        move_mock.assert_not_called()
        qdrant_client.upsert.assert_not_called()
        qdrant_client.delete_collection.assert_not_called()
        assert graphiti._graphs_by_key, 'expected at least one graph to have been touched'
        for graph in graphiti._graphs_by_key.values():
            graph.delete.assert_not_called()
            graph.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_report_shape_and_dispositions(self):
        """dry_run=True; a sibling with entity rows is MERGE; a collection
        with points is MERGE; a non-empty JUNK_KEYS entry is UNRESOLVED even
        in dry-run; an unmerged sibling is ALSO UNRESOLVED as a junk-key
        candidate (its entities are still present, node_count > 0)."""
        memory_service, _, _ = self._scenario()

        report = await _mod.run(_run_args(apply=False), memory_service, limit=1000)

        assert report['dry_run'] is True

        graph_by_sibling = {item['sibling']: item for item in report['graph_family_merges']}
        assert graph_by_sibling['know-live']['canonical'] == 'know_live'
        assert graph_by_sibling['know-live']['node_count'] == 1
        assert graph_by_sibling['know-live']['node_uuids'] == ['uuid-1']
        assert graph_by_sibling['know-live']['disposition'] == 'MERGE'

        collection_by_source = {item['source']: item for item in report['collection_merges']}
        assert collection_by_source['fused_dark-factory']['target'] == 'fused_dark_factory'
        assert collection_by_source['fused_dark-factory']['point_count'] == 1
        assert collection_by_source['fused_dark-factory']['disposition'] == 'MERGE'

        junk_by_key = {item['key']: item for item in report['junk_key_deletions']}
        assert junk_by_key['my-project']['node_count'] == 3
        assert junk_by_key['my-project']['disposition'] == 'UNRESOLVED'
        assert junk_by_key['test-project']['node_count'] == 0
        assert junk_by_key['test-project']['disposition'] == 'DELETE'
        assert junk_by_key['know-live']['node_count'] == 1
        assert junk_by_key['know-live']['disposition'] == 'UNRESOLVED'

    @pytest.mark.asyncio
    async def test_dry_run_collection_unresolved_when_scroll_capped(self):
        """A collection whose scroll hits --limit is UNRESOLVED, not MERGE:
        no-silent-caps -- a capped scroll may be incomplete, so it must not
        be reported as a clean merge."""
        points = [_make_point(f'p{i}') for i in range(2)]
        graphiti = _make_run_graphiti_mock()
        qdrant_client = _make_run_qdrant_mock(
            points_by_collection={'fused_dark-factory': points},
        )
        memory_service = _make_run_memory_service(graphiti, qdrant_client)

        report = await _mod.run(_run_args(apply=False), memory_service, limit=2)

        collection_by_source = {item['source']: item for item in report['collection_merges']}
        assert collection_by_source['fused_dark-factory']['disposition'] == 'UNRESOLVED'

    @pytest.mark.asyncio
    async def test_dry_run_sibling_unresolved_when_enumeration_capped(self):
        """A sibling whose Entity enumeration hits --limit is UNRESOLVED,
        not MERGE (same no-silent-caps guard as the collection scroll)."""
        rows = [['uuid-1', 'Node A'], ['uuid-2', 'Node B']]
        graphiti = _make_run_graphiti_mock(entity_rows_by_key={'know-live': rows})
        qdrant_client = _make_run_qdrant_mock()
        memory_service = _make_run_memory_service(graphiti, qdrant_client)

        report = await _mod.run(_run_args(apply=False), memory_service, limit=2)

        graph_by_sibling = {item['sibling']: item for item in report['graph_family_merges']}
        assert graph_by_sibling['know-live']['disposition'] == 'UNRESOLVED'


# ===========================================================================
# Tests: run() -- apply path
# ===========================================================================

class TestRunApply:
    """Tests for async run(args, memory_service, *, limit) in apply mode
    (args.apply=True). Reuses the fixtures declared above (shared with
    TestRunDryRun)."""

    def _scenario(self):
        """One mergeable sibling (know-live, 1 entity row; its total node
        count is fixed at 0 to stand in for the post-move state, so it is
        ALSO DELETE-able as a junk key in the same --apply pass); one
        collection with points (fused_dark-factory); one non-empty
        JUNK_KEYS entry (my-project, count 3); one empty JUNK_KEYS entry
        (test-project, count 0)."""
        graphiti = _make_run_graphiti_mock(
            entity_rows_by_key={'know-live': [['uuid-1', 'Node A']]},
            total_count_by_key={'know-live': 0, 'my-project': 3, 'test-project': 0},
        )
        qdrant_client = _make_run_qdrant_mock(
            points_by_collection={
                'fused_dark-factory': [_make_point('p1', payload={'user_id': 'dark-factory'})],
            },
        )
        memory_service = _make_run_memory_service(graphiti, qdrant_client)
        return memory_service, graphiti, qdrant_client

    def _move_mock(self, monkeypatch) -> AsyncMock:
        """Patch move_entity_across_graphs so the family-merge branch (which
        --apply always exercises for a non-capped sibling) never reaches the
        real ε primitive."""
        move_mock = AsyncMock(
            return_value=_mod.MoveResult(
                uuid='uuid-1', source_graph='know-live', target_graph='know_live',
                edges_moved=2, edges_skipped=0, mentions_moved=1, mentions_skipped=0,
            ),
        )
        monkeypatch.setattr(_mod, 'move_entity_across_graphs', move_mock)
        return move_mock

    @pytest.mark.asyncio
    async def test_apply_moves_each_sibling_entity_with_rewrite_group_id(self, monkeypatch):
        """move_entity_across_graphs is called once per sibling entity row,
        with source=sibling, target=canonical, and rewrite_group_id=canonical
        -- the Phase-2 identity rewrite (PRD decision 6) -- and the know-live
        graph_family item carries the merge summary (nodes_moved etc.)."""
        move_mock = self._move_mock(monkeypatch)
        memory_service, graphiti, _ = self._scenario()

        report = await _mod.run(_run_args(apply=True), memory_service, limit=1000)

        move_mock.assert_called_once()
        call = move_mock.call_args
        assert call.args == (graphiti, 'uuid-1', 'know-live', 'know_live')
        assert call.kwargs == {'rewrite_group_id': 'know_live'}

        graph_by_sibling = {item['sibling']: item for item in report['graph_family_merges']}
        know_live_item = graph_by_sibling['know-live']
        assert know_live_item['nodes_moved'] == 1
        assert know_live_item['edges_moved'] == 2
        assert know_live_item['mentions_moved'] == 1

    @pytest.mark.asyncio
    async def test_apply_upserts_with_canonical_user_id_and_deletes_source_collection(self, monkeypatch):
        """qdrant upsert is called with collection_name=target and the
        point's payload user_id rewritten to canonical; the not-capped
        source collection is deleted; the collection item carries
        points_upserted/source_deleted=True."""
        self._move_mock(monkeypatch)
        memory_service, _, qdrant_client = self._scenario()

        report = await _mod.run(_run_args(apply=True), memory_service, limit=1000)

        upsert_calls = [
            c for c in qdrant_client.upsert.call_args_list
            if c.kwargs.get('collection_name') == 'fused_dark_factory'
        ]
        assert len(upsert_calls) == 1
        upserted = upsert_calls[0].kwargs['points']
        assert len(upserted) == 1
        assert upserted[0].payload['user_id'] == 'dark_factory'

        qdrant_client.delete_collection.assert_any_call('fused_dark-factory')

        collection_by_source = {item['source']: item for item in report['collection_merges']}
        dark_factory_item = collection_by_source['fused_dark-factory']
        assert dark_factory_item['points_upserted'] == 1
        assert dark_factory_item['source_deleted'] is True

    @pytest.mark.asyncio
    async def test_apply_deletes_zero_count_junk_key_and_leaves_nonzero_unresolved(self, monkeypatch):
        """delete_junk_key deletes the zero-count key (test-project,
        disposition DELETE) but leaves the non-empty one (my-project,
        disposition UNRESOLVED) untouched -- .delete() is never called on
        it. The know-live sibling, whose total count is fixed at 0 (the
        post-move state), is likewise DELETE-able in the same pass."""
        self._move_mock(monkeypatch)
        memory_service, graphiti, _ = self._scenario()

        report = await _mod.run(_run_args(apply=True), memory_service, limit=1000)

        graphiti._graphs_by_key['test-project'].delete.assert_called_once()
        graphiti._graphs_by_key['my-project'].delete.assert_not_called()

        junk_by_key = {item['key']: item for item in report['junk_key_deletions']}
        assert junk_by_key['test-project']['disposition'] == 'DELETE'
        assert junk_by_key['my-project']['disposition'] == 'UNRESOLVED'
        assert junk_by_key['know-live']['disposition'] == 'DELETE'

    @pytest.mark.asyncio
    async def test_apply_report_dry_run_is_false(self, monkeypatch):
        """report['dry_run'] is False when args.apply is True."""
        self._move_mock(monkeypatch)
        memory_service, _, _ = self._scenario()

        report = await _mod.run(_run_args(apply=True), memory_service, limit=1000)

        assert report['dry_run'] is False

    @pytest.mark.asyncio
    async def test_apply_capped_collection_not_deleted_and_stays_unresolved(self, monkeypatch):
        """A collection whose scroll hits --limit is NOT deleted even with
        --apply (no data loss on a partial migration) and its disposition
        stays UNRESOLVED."""
        self._move_mock(monkeypatch)
        points = [_make_point(f'p{i}') for i in range(2)]
        graphiti = _make_run_graphiti_mock()
        qdrant_client = _make_run_qdrant_mock(
            points_by_collection={'fused_dark-factory': points},
        )
        memory_service = _make_run_memory_service(graphiti, qdrant_client)

        report = await _mod.run(_run_args(apply=True), memory_service, limit=2)

        capped_deletes = [
            c for c in qdrant_client.delete_collection.call_args_list
            if c.args and c.args[0] == 'fused_dark-factory'
        ]
        assert capped_deletes == []

        collection_by_source = {item['source']: item for item in report['collection_merges']}
        assert collection_by_source['fused_dark-factory']['disposition'] == 'UNRESOLVED'
