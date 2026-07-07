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
