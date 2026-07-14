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
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from fused_memory.maintenance.cross_graph_move import SubgraphEdgeResult

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


def _make_count_result(count: int | None) -> MagicMock:
    """Qdrant CountResult stand-in exposing .count -- the shape returned by
    AsyncQdrantClient.count(...). *count* may be None to simulate the
    indeterminate-count case (see TestCountCollectionPoints)."""
    result = MagicMock()
    result.count = count
    return result


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
        assert _mod.COLLECTION_MERGES['fused_pump-web-ui'] == 'fused_pump_web_ui'

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


class TestEmptyCollectionCleanup:
    """Tests for the module constant EMPTY_COLLECTION_CLEANUP."""

    def test_contains_exactly_the_seven_empty_strays(self):
        """EMPTY_COLLECTION_CLEANUP contains exactly the 7 known-empty
        divergent/junk Qdrant collections."""
        expected = {
            'fused_knowlive', 'fused_know-live', 'fused_autopilot-video',
            'fused_my-project', 'fused_1098', 'fused_default',
            'fused_-home-leo-src-dark-factory',
        }
        assert set(_mod.EMPTY_COLLECTION_CLEANUP) == expected

    def test_does_not_include_ambiguous_collections(self):
        """PRD Open Q2 defers reify_ (empty project id) and
        fused_fused_memory to ι human review -- neither appears here, same
        as COLLECTION_MERGES."""
        assert 'reify_' not in _mod.EMPTY_COLLECTION_CLEANUP
        assert 'fused_fused_memory' not in _mod.EMPTY_COLLECTION_CLEANUP


# ===========================================================================
# Tests: build_consolidation_report
# ===========================================================================

class TestBuildConsolidationReport:
    """Tests for the pure function build_consolidation_report(...)."""

    def test_report_shape(self):
        """Returned dict has exactly the five top-level manifest keys."""
        report = _mod.build_consolidation_report([], [], [], [], dry_run=True)

        assert set(report.keys()) == {
            'dry_run', 'graph_family_merges', 'collection_merges',
            'junk_key_deletions', 'empty_collection_deletions',
        }

    def test_dry_run_passed_through_verbatim(self):
        """dry_run is passed through verbatim, not hardcoded, in either direction."""
        report_true = _mod.build_consolidation_report([], [], [], [], dry_run=True)
        report_false = _mod.build_consolidation_report([], [], [], [], dry_run=False)

        assert report_true['dry_run'] is True
        assert report_false['dry_run'] is False

    def test_sections_are_lists_of_given_items(self):
        """Each section is exactly the list of per-item dicts given, in order."""
        graph_items = [{'sibling': 'know-live', 'canonical': 'know_live', 'disposition': 'MERGE'}]
        collection_items = [{'source': 'reify_reify', 'target': 'fused_reify', 'disposition': 'MERGE'}]
        junk_items = [{'key': 'my-project', 'node_count': 0, 'disposition': 'DELETE'}]
        empty_collection_items = [
            {'collection': 'fused_knowlive', 'point_count': 0, 'disposition': 'DELETE'},
        ]

        report = _mod.build_consolidation_report(
            graph_items, collection_items, junk_items, empty_collection_items, dry_run=True,
        )

        assert report['graph_family_merges'] == graph_items
        assert report['collection_merges'] == collection_items
        assert report['junk_key_deletions'] == junk_items
        assert report['empty_collection_deletions'] == empty_collection_items

    def test_empty_inputs_produce_empty_lists(self):
        """Empty inputs for all four sections produce empty lists, with dry_run preserved."""
        report = _mod.build_consolidation_report([], [], [], [], dry_run=False)

        assert report['graph_family_merges'] == []
        assert report['collection_merges'] == []
        assert report['junk_key_deletions'] == []
        assert report['empty_collection_deletions'] == []
        assert report['dry_run'] is False

    def test_no_io_pure_function(self):
        """build_consolidation_report performs no I/O -- calling twice with
        identical inputs is referentially stable in content."""
        graph_items = [{'sibling': 'knowlive', 'canonical': 'know_live', 'disposition': 'MERGE'}]
        collection_items = [{'source': 'fused_dark-factory', 'target': 'fused_dark_factory'}]
        junk_items = [{'key': 'default', 'node_count': 0, 'disposition': 'DELETE'}]
        empty_collection_items = [
            {'collection': 'fused_default', 'point_count': 0, 'disposition': 'DELETE'},
        ]

        r1 = _mod.build_consolidation_report(
            graph_items, collection_items, junk_items, empty_collection_items, dry_run=True,
        )
        r2 = _mod.build_consolidation_report(
            graph_items, collection_items, junk_items, empty_collection_items, dry_run=True,
        )

        assert r1 == r2


# ===========================================================================
# Tests: has_unresolved
# ===========================================================================

class TestHasUnresolved:
    """Tests for the pure function has_unresolved(report) -> bool."""

    def test_all_clean_empty_sections_returns_false(self):
        """An all-clean report with empty sections returns False."""
        report = _mod.build_consolidation_report([], [], [], [], dry_run=True)

        assert _mod.has_unresolved(report) is False

    def test_all_clean_merge_and_delete_dispositions_returns_false(self):
        """Sections containing only MERGE/DELETE dispositions (no
        UNRESOLVED) return False."""
        graph_items = [{'sibling': 'know-live', 'canonical': 'know_live', 'disposition': 'MERGE'}]
        collection_items = [{'source': 'reify_reify', 'target': 'fused_reify', 'disposition': 'MERGE'}]
        junk_items = [{'key': 'my-project', 'node_count': 0, 'disposition': 'DELETE'}]
        empty_collection_items = [
            {'collection': 'fused_knowlive', 'point_count': 0, 'disposition': 'DELETE'},
        ]
        report = _mod.build_consolidation_report(
            graph_items, collection_items, junk_items, empty_collection_items, dry_run=True,
        )

        assert _mod.has_unresolved(report) is False

    def test_unresolved_in_graph_family_merges_returns_true(self):
        """A single UNRESOLVED disposition in graph_family_merges returns True."""
        graph_items = [{'sibling': 'know-live', 'canonical': 'know_live', 'disposition': 'UNRESOLVED'}]
        report = _mod.build_consolidation_report(graph_items, [], [], [], dry_run=True)

        assert _mod.has_unresolved(report) is True

    def test_unresolved_in_collection_merges_returns_true(self):
        """A single UNRESOLVED disposition in collection_merges returns True."""
        collection_items = [
            {'source': 'fused_dark-factory', 'target': 'fused_dark_factory', 'disposition': 'UNRESOLVED'},
        ]
        report = _mod.build_consolidation_report([], collection_items, [], [], dry_run=True)

        assert _mod.has_unresolved(report) is True

    def test_unresolved_in_junk_key_deletions_returns_true(self):
        """A single UNRESOLVED disposition in junk_key_deletions returns True."""
        junk_items = [{'key': 'my-project', 'node_count': 3, 'disposition': 'UNRESOLVED'}]
        report = _mod.build_consolidation_report([], [], junk_items, [], dry_run=True)

        assert _mod.has_unresolved(report) is True

    def test_unresolved_in_empty_collection_deletions_returns_true(self):
        """A single UNRESOLVED disposition in empty_collection_deletions returns True."""
        empty_collection_items = [
            {'collection': 'fused_knowlive', 'point_count': 3, 'disposition': 'UNRESOLVED'},
        ]
        report = _mod.build_consolidation_report([], [], [], empty_collection_items, dry_run=True)

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
        empty_collection_items = [
            {'collection': 'fused_knowlive', 'point_count': 0, 'disposition': 'DELETE'},
        ]
        report = _mod.build_consolidation_report(
            graph_items, collection_items, junk_items, empty_collection_items, dry_run=False,
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


class TestEnumerateGraphEpisodicNodes:
    """Tests for async enumerate_graph_episodic_nodes(graphiti, key, *, limit)."""

    @pytest.mark.asyncio
    async def test_calls_graph_for_with_key(self):
        """graphiti._graph_for is called with the key argument."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        await _mod.enumerate_graph_episodic_nodes(graphiti, 'know-live', limit=1000)

        graphiti._graph_for.assert_called_once_with('know-live')

    @pytest.mark.asyncio
    async def test_uses_ro_query_not_query(self):
        """Enumeration is read-only: ro_query is used, .query is NEVER called."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        await _mod.enumerate_graph_episodic_nodes(graphiti, 'know-live', limit=1000)

        graph.ro_query.assert_called_once()
        graph.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_matches_episodic_label(self):
        """The Cypher issued matches :Episodic nodes specifically."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        await _mod.enumerate_graph_episodic_nodes(graphiti, 'know-live', limit=1000)

        cypher = graph.ro_query.call_args.args[0]
        assert ':Episodic' in cypher

    @pytest.mark.asyncio
    async def test_normalizes_rows_to_uuid_dicts(self):
        """result_set rows [uuid] are normalized to [{'uuid': ...}, ...]."""
        graphiti = MagicMock()
        rows = [['episode-uuid-1'], ['episode-uuid-2']]
        graph = _make_graph_mock(rows)
        graphiti._graph_for = MagicMock(return_value=graph)

        result = await _mod.enumerate_graph_episodic_nodes(graphiti, 'know-live', limit=1000)

        assert result == [
            {'uuid': 'episode-uuid-1'},
            {'uuid': 'episode-uuid-2'},
        ]

    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty_list(self):
        """An empty result_set returns an empty list, not an error."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        result = await _mod.enumerate_graph_episodic_nodes(graphiti, 'know-live', limit=1000)

        assert result == []

    @pytest.mark.asyncio
    async def test_warns_when_row_count_hits_limit(self, caplog):
        """No-silent-caps: hitting the limit logs a WARNING."""
        graphiti = MagicMock()
        rows = [[f'episode-uuid-{i}'] for i in range(3)]
        graph = _make_graph_mock(rows)
        graphiti._graph_for = MagicMock(return_value=graph)

        with caplog.at_level('WARNING'):
            await _mod.enumerate_graph_episodic_nodes(graphiti, 'know-live', limit=3)

        assert any('limit' in rec.message.lower() for rec in caplog.records), (
            f'Expected a limit-related WARNING, got: {[r.message for r in caplog.records]}'
        )

    @pytest.mark.asyncio
    async def test_no_warning_when_under_limit(self, caplog):
        """Row count below the limit does not log a WARNING."""
        graphiti = MagicMock()
        rows = [['episode-uuid-1']]
        graph = _make_graph_mock(rows)
        graphiti._graph_for = MagicMock(return_value=graph)

        with caplog.at_level('WARNING'):
            await _mod.enumerate_graph_episodic_nodes(graphiti, 'know-live', limit=1000)

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
    """Tests for async merge_graph_family(graphiti, sibling, canonical,
    entity_rows, episode_rows) -- the three-phase barrier-ordered apply
    (task 2502, template: migrate_cross_graph_leak.py's run()): Phase A
    creates every entity + episode in canonical BEFORE any delete; Phase B
    recreates every intra-family RELATES_TO edge and MENTIONS link in ONE
    batched call; Phase C deletes every non-blocked/non-failed source.
    """

    def _patch_primitives(
        self, monkeypatch, *,
        create_node_side_effect=None,
        create_episode_side_effect=None,
        recreate_result=None,
        recreate_side_effect=None,
        delete_node_side_effect=None,
        delete_episode_side_effect=None,
    ):
        """Patch every three-phase primitive merge_graph_family drives (all
        five are new module-level names this task adds to the script's
        import block -- absent until step-8 lands, hence raising=False),
        returning the mocks for call-order/argument assertions.

        *recreate_side_effect*, when given (typically an exception
        instance), makes recreate_subgraph_relationships RAISE instead of
        returning *recreate_result* -- simulating a SYSTEMIC Phase-B abort,
        as opposed to a per-item isolated failure (expressed instead via a
        *recreate_result* carrying a non-empty ``blocked`` list)."""
        create_node_mock = AsyncMock(side_effect=create_node_side_effect)
        create_episode_mock = AsyncMock(side_effect=create_episode_side_effect)
        recreate_mock = AsyncMock(
            return_value=recreate_result if recreate_result is not None else SubgraphEdgeResult(),
            side_effect=recreate_side_effect,
        )
        delete_node_mock = AsyncMock(side_effect=delete_node_side_effect)
        delete_episode_mock = AsyncMock(side_effect=delete_episode_side_effect)

        monkeypatch.setattr(_mod, 'create_moved_node', create_node_mock, raising=False)
        monkeypatch.setattr(_mod, 'create_moved_episode', create_episode_mock, raising=False)
        monkeypatch.setattr(_mod, 'recreate_subgraph_relationships', recreate_mock, raising=False)
        monkeypatch.setattr(_mod, 'delete_source_node', delete_node_mock, raising=False)
        monkeypatch.setattr(_mod, 'delete_source_episode', delete_episode_mock, raising=False)
        return {
            'create_node': create_node_mock,
            'create_episode': create_episode_mock,
            'recreate': recreate_mock,
            'delete_node': delete_node_mock,
            'delete_episode': delete_episode_mock,
        }

    @pytest.mark.asyncio
    async def test_barrier_ordering_all_creates_before_any_delete(self, monkeypatch):
        """Every create_moved_node/create_moved_episode call is awaited
        BEFORE any delete_source_node/delete_source_episode call -- the
        barrier ordering that guarantees a co-moving intra-family RELATES_TO
        edge (or a MENTIONS link onto a just-relocated episode) survives.
        """
        call_order = []

        def _record(tag):
            def _fn(graphiti_arg, uuid, *rest, **kwargs):
                call_order.append((tag, uuid))
            return _fn

        mocks = self._patch_primitives(
            monkeypatch,
            create_node_side_effect=_record('create_node'),
            create_episode_side_effect=_record('create_episode'),
            delete_node_side_effect=_record('delete_node'),
            delete_episode_side_effect=_record('delete_episode'),
        )

        entity_rows = [{'uuid': 'entity-1', 'name': 'Alice'}, {'uuid': 'entity-2', 'name': 'Bob'}]
        episode_rows = [{'uuid': 'episode-1'}]

        await _mod.merge_graph_family(
            MagicMock(), 'know-live', 'know_live', entity_rows, episode_rows,
        )

        tags = [t for t, _u in call_order]
        last_create_idx = max(i for i, t in enumerate(tags) if t in ('create_node', 'create_episode'))
        first_delete_idx = min(i for i, t in enumerate(tags) if t in ('delete_node', 'delete_episode'))
        assert last_create_idx < first_delete_idx

        assert {u for t, u in call_order if t == 'create_node'} == {'entity-1', 'entity-2'}
        assert {u for t, u in call_order if t == 'create_episode'} == {'episode-1'}
        assert {u for t, u in call_order if t == 'delete_node'} == {'entity-1', 'entity-2'}
        assert {u for t, u in call_order if t == 'delete_episode'} == {'episode-1'}
        mocks['recreate'].assert_awaited_once()

    @pytest.mark.asyncio
    async def test_creates_use_rewrite_group_id_canonical(self, monkeypatch):
        """Every create_moved_node/create_moved_episode call is given
        rewrite_group_id=canonical -- the Phase-2 identity rewrite (PRD
        decision 6)."""
        mocks = self._patch_primitives(monkeypatch)
        entity_rows = [{'uuid': 'entity-1', 'name': 'Alice'}]
        episode_rows = [{'uuid': 'episode-1'}]

        await _mod.merge_graph_family(
            MagicMock(), 'know-live', 'know_live', entity_rows, episode_rows,
        )

        node_call = mocks['create_node'].call_args
        assert node_call.args == (ANY, 'entity-1', 'know-live', 'know_live')
        assert node_call.kwargs == {'rewrite_group_id': 'know_live'}

        episode_call = mocks['create_episode'].call_args
        assert episode_call.args == (ANY, 'episode-1', 'know-live', 'know_live')
        assert episode_call.kwargs == {'rewrite_group_id': 'know_live'}

    @pytest.mark.asyncio
    async def test_recreate_subgraph_relationships_called_once_with_move_specs(self, monkeypatch):
        """recreate_subgraph_relationships is called EXACTLY once with a
        MOVE spec (source=sibling, target=canonical) for every entity --
        NOT once per entity -- so a co-moving intra-family edge shared by
        two specs in this batch is deduped and recreated exactly once."""
        mocks = self._patch_primitives(monkeypatch)
        entity_rows = [{'uuid': 'entity-1', 'name': 'Alice'}, {'uuid': 'entity-2', 'name': 'Bob'}]

        await _mod.merge_graph_family(MagicMock(), 'know-live', 'know_live', entity_rows, [])

        mocks['recreate'].assert_awaited_once()
        specs = mocks['recreate'].call_args.args[1]
        assert {s['uuid'] for s in specs} == {'entity-1', 'entity-2'}
        for spec in specs:
            assert spec['disposition'] == 'MOVE'
            assert spec['source_graph'] == 'know-live'
            assert spec['target_graph'] == 'know_live'

    @pytest.mark.asyncio
    async def test_deletes_run_after_phase_b_for_every_non_blocked_source(self, monkeypatch):
        """delete_source_node/delete_source_episode run for every
        entity/episode once Phase A + Phase B have both completed cleanly.
        """
        mocks = self._patch_primitives(monkeypatch)
        entity_rows = [{'uuid': 'entity-1', 'name': 'Alice'}, {'uuid': 'entity-2', 'name': 'Bob'}]
        episode_rows = [{'uuid': 'episode-1'}]

        await _mod.merge_graph_family(
            MagicMock(), 'know-live', 'know_live', entity_rows, episode_rows,
        )

        assert mocks['delete_node'].await_count == 2
        deleted_node_uuids = {c.args[1] for c in mocks['delete_node'].call_args_list}
        assert deleted_node_uuids == {'entity-1', 'entity-2'}
        for c in mocks['delete_node'].call_args_list:
            assert c.args[2] == 'know-live'  # source_graph

        mocks['delete_episode'].assert_awaited_once()
        episode_call = mocks['delete_episode'].call_args
        assert episode_call.args[1] == 'episode-1'
        assert episode_call.args[2] == 'know-live'

    @pytest.mark.asyncio
    async def test_withholds_deletion_for_phase_a_create_failure(self, monkeypatch):
        """A uuid whose Phase-A create_moved_node raises is isolated: its
        source deletion is WITHHELD (create-before-delete preserved) and it
        is excluded from the Phase-B batch, but the OTHER entity still
        proceeds through Phase B/C cleanly."""
        def _create_node_side_effect(graphiti_arg, uuid, *rest, **kwargs):
            if uuid == 'entity-1':
                raise RuntimeError('boom')

        mocks = self._patch_primitives(
            monkeypatch, create_node_side_effect=_create_node_side_effect,
        )
        entity_rows = [{'uuid': 'entity-1', 'name': 'Alice'}, {'uuid': 'entity-2', 'name': 'Bob'}]

        summary = await _mod.merge_graph_family(
            MagicMock(), 'know-live', 'know_live', entity_rows, [],
        )

        deleted_node_uuids = {c.args[1] for c in mocks['delete_node'].call_args_list}
        assert 'entity-1' not in deleted_node_uuids
        assert 'entity-2' in deleted_node_uuids

        # The failed entity must not be offered to Phase B either -- its
        # target-graph copy was never created.
        specs = mocks['recreate'].call_args.args[1]
        assert {s['uuid'] for s in specs} == {'entity-2'}

        assert summary['nodes_blocked'] == 1
        assert summary['nodes_moved'] == 1

    @pytest.mark.asyncio
    async def test_withholds_deletion_for_phase_b_blocked_uuid(self, monkeypatch):
        """A uuid named in Phase B's SubgraphEdgeResult.blocked has its
        source deletion WITHHELD -- the un-recreated edge/mention still
        exists only in source, so deleting it would destroy the only copy.
        """
        blocked_result = SubgraphEdgeResult(
            edges_recreated=0,
            blocked=[{'kind': 'edge', 'uuid': 'edge-1', 'reason': 'boom', 'node_uuids': ['entity-1']}],
        )
        mocks = self._patch_primitives(monkeypatch, recreate_result=blocked_result)
        entity_rows = [{'uuid': 'entity-1', 'name': 'Alice'}, {'uuid': 'entity-2', 'name': 'Bob'}]

        summary = await _mod.merge_graph_family(
            MagicMock(), 'know-live', 'know_live', entity_rows, [],
        )

        deleted_node_uuids = {c.args[1] for c in mocks['delete_node'].call_args_list}
        assert 'entity-1' not in deleted_node_uuids
        assert 'entity-2' in deleted_node_uuids
        assert summary['blocked'] == blocked_result.blocked
        assert summary['nodes_blocked'] == 1
        assert summary['nodes_moved'] == 1

    @pytest.mark.asyncio
    async def test_withholds_episode_deletion_for_phase_b_blocked_mention(self, monkeypatch):
        """A Phase-B blocked MENTIONS item carries episode_uuid (the
        source-resident Episodic node whose MENTIONS recreate raised) --
        that episode's Phase-C deletion must be WITHHELD too, or the
        un-recreated MENTIONS link (which now exists only in source) would
        be destroyed. Regression for the data-loss-barrier-gap reviewer
        finding: today only entities named in blocked['node_uuids'] are
        withheld, so this episode was deleted anyway. No create_failed
        entity here, so the guarded MENTIONS-topology read (Scenario 2)
        never fires and a bare MagicMock graphiti suffices.
        """
        blocked_result = SubgraphEdgeResult(
            blocked=[{
                'kind': 'mention', 'uuid': 'm1', 'reason': 'boom',
                'node_uuids': ['entity-1'], 'episode_uuid': 'episode-1',
            }],
        )
        mocks = self._patch_primitives(monkeypatch, recreate_result=blocked_result)
        entity_rows = [{'uuid': 'entity-1'}]
        episode_rows = [{'uuid': 'episode-1'}, {'uuid': 'episode-2'}]

        summary = await _mod.merge_graph_family(
            MagicMock(), 'know-live', 'know_live', entity_rows, episode_rows,
        )

        deleted_episode_uuids = {c.args[1] for c in mocks['delete_episode'].call_args_list}
        assert 'episode-1' not in deleted_episode_uuids
        assert 'episode-2' in deleted_episode_uuids
        assert summary['episodes_blocked'] == 1
        assert summary['episodes_moved'] == 1

    @pytest.mark.asyncio
    async def test_withholds_episode_deletion_for_entity_create_failure_mentioned_by_episode(
        self, monkeypatch,
    ):
        """An entity whose Phase-A create_moved_node RAISES is dropped from
        entity_specs, so Phase B never reads/recreates its incident
        MENTIONS at all -- not counted in blocked, not in mentions_skipped.
        The mentioning episode's OWN Phase-A create succeeded (so it is not
        in episode_create_failed) and it is not named in Phase B's blocked
        list either. Without a guarded sibling MENTIONS-topology probe, its
        Phase-C deletion would proceed and destroy the un-recreated
        source-only MENTIONS link -- the second data-loss-barrier-gap
        scenario. The probe MUST be read-only (ro_query, never .query).
        """
        def _create_node_side_effect(graphiti_arg, uuid, *rest, **kwargs):
            if uuid == 'entity-1':
                raise RuntimeError('boom')

        mocks = self._patch_primitives(
            monkeypatch, create_node_side_effect=_create_node_side_effect,
        )

        graph_mock = _make_graph_mock([])

        async def _ro_query(cypher, params=None):
            params = params or {}
            if 'MENTIONS' in cypher and params.get('uuid') == 'entity-1':
                return MagicMock(result_set=[['episode-1']])
            return MagicMock(result_set=[])

        graph_mock.ro_query = AsyncMock(side_effect=_ro_query)

        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(return_value=graph_mock)

        entity_rows = [{'uuid': 'entity-1'}, {'uuid': 'entity-2'}]
        episode_rows = [{'uuid': 'episode-1'}, {'uuid': 'episode-2'}]

        summary = await _mod.merge_graph_family(
            graphiti, 'know-live', 'know_live', entity_rows, episode_rows,
        )

        deleted_episode_uuids = {c.args[1] for c in mocks['delete_episode'].call_args_list}
        assert 'episode-1' not in deleted_episode_uuids
        assert 'episode-2' in deleted_episode_uuids
        assert summary['episodes_blocked'] >= 1

        graphiti._graph_for.assert_any_call('know-live')
        graph_mock.ro_query.assert_awaited()
        graph_mock.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_withholds_node_deletion_for_episode_create_failure_mentioning_entity(
        self, monkeypatch,
    ):
        """The MIRROR IMAGE of the scenario above (reviewer follow-up,
        data-loss-barrier-gap case (3)): an episode whose Phase-A
        create_moved_episode RAISES is correctly withheld via
        episode_create_failed -- but any entity it MENTIONS is unaffected by
        THAT failure: the entity's OWN Phase-A create succeeded, so it is
        offered to Phase B, whose MENTIONS CREATE MATCHes the
        never-created-in-target episode and finds nothing -- silently
        counted in mentions_skipped, NOT blocked (only a raising CREATE
        lands in blocked; a no-op MATCH does not raise). Without a guarded
        REVERSE-direction sibling MENTIONS-topology probe, the entity's
        Phase-C deletion would proceed and destroy the source-only MENTIONS
        link. The probe MUST be read-only (ro_query, never .query).
        """
        def _create_episode_side_effect(graphiti_arg, uuid, *rest, **kwargs):
            if uuid == 'episode-1':
                raise RuntimeError('boom')

        mocks = self._patch_primitives(
            monkeypatch, create_episode_side_effect=_create_episode_side_effect,
        )

        graph_mock = _make_graph_mock([])

        async def _ro_query(cypher, params=None):
            params = params or {}
            if 'MENTIONS' in cypher and params.get('uuid') == 'episode-1':
                return MagicMock(result_set=[['entity-1']])
            return MagicMock(result_set=[])

        graph_mock.ro_query = AsyncMock(side_effect=_ro_query)

        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(return_value=graph_mock)

        entity_rows = [{'uuid': 'entity-1'}, {'uuid': 'entity-2'}]
        episode_rows = [{'uuid': 'episode-1'}, {'uuid': 'episode-2'}]

        summary = await _mod.merge_graph_family(
            graphiti, 'know-live', 'know_live', entity_rows, episode_rows,
        )

        deleted_node_uuids = {c.args[1] for c in mocks['delete_node'].call_args_list}
        assert 'entity-1' not in deleted_node_uuids
        assert 'entity-2' in deleted_node_uuids
        assert summary['nodes_blocked'] >= 1

        graphiti._graph_for.assert_any_call('know-live')
        graph_mock.ro_query.assert_awaited()
        graph_mock.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_phase_b_systemic_abort_withholds_all_source_deletions(self, monkeypatch):
        """A SYSTEMIC (batch-level) Phase-B abort -- recreate_subgraph_
        relationships RAISES instead of isolating the failure onto
        SubgraphEdgeResult.blocked -- must withhold the WHOLE batch's source
        deletions, not just the uuids named in the recovered
        exc.partial_result.blocked: partial_result.blocked only names items
        that failed in isolation BEFORE the abort -- it never names the
        edges/mentions the batch never reached, which now exist only in
        source. Regression for the reviewer's data-loss-create-before-delete
        finding: today only entity-1 (named in partial_result.blocked) is
        withheld, so entity-2 and episode-1 -- neither ever confirmed
        recreated -- get deleted anyway, nodes_blocked/episodes_blocked
        under-count, and the family would stay MERGE.
        """
        partial_result = SubgraphEdgeResult(
            edges_recreated=1,
            blocked=[{'kind': 'edge', 'uuid': 'edge-1', 'reason': 'boom', 'node_uuids': ['entity-1']}],
        )
        abort_exc = RuntimeError('phase b systemic failure')
        abort_exc.partial_result = partial_result  # type: ignore[attr-defined]
        mocks = self._patch_primitives(monkeypatch, recreate_side_effect=abort_exc)
        entity_rows = [{'uuid': 'entity-1', 'name': 'A'}, {'uuid': 'entity-2', 'name': 'B'}]
        episode_rows = [{'uuid': 'episode-1'}]

        summary = await _mod.merge_graph_family(
            MagicMock(), 'know-live', 'know_live', entity_rows, episode_rows,
        )

        assert mocks['delete_node'].await_count == 0
        assert mocks['delete_episode'].await_count == 0
        assert summary['nodes_blocked'] == 2
        assert summary['episodes_blocked'] == 1
        assert summary['nodes_moved'] == 0
        assert summary['episodes_moved'] == 0
        # The partial tally is still surfaced, not discarded.
        assert summary['edges_recreated'] == 1
        assert summary['blocked'] == partial_result.blocked

    @pytest.mark.asyncio
    async def test_phase_b_systemic_abort_without_partial_result_withholds_all_source_deletions(
        self, monkeypatch,
    ):
        """The defensive branch: an abort exception with NO partial_result
        attribute (getattr -> None) leaves edge_result zeroed -- but must
        STILL withhold every source deletion. The absence of a usable
        partial tally is not evidence that nothing needs blocking; it is
        the worst case (we know NOTHING about what Phase B did or didn't
        reach), so every source is withheld.
        """
        abort_exc = RuntimeError('phase b systemic failure, no partial result attached')
        mocks = self._patch_primitives(monkeypatch, recreate_side_effect=abort_exc)
        entity_rows = [{'uuid': 'entity-1', 'name': 'A'}, {'uuid': 'entity-2', 'name': 'B'}]
        episode_rows = [{'uuid': 'episode-1'}]

        summary = await _mod.merge_graph_family(
            MagicMock(), 'know-live', 'know_live', entity_rows, episode_rows,
        )

        assert mocks['delete_node'].await_count == 0
        assert mocks['delete_episode'].await_count == 0
        assert summary['nodes_blocked'] == 2
        assert summary['episodes_blocked'] == 1
        assert summary['edges_recreated'] == 0
        assert summary['blocked'] == []

    @pytest.mark.asyncio
    async def test_summary_tallies_and_surfaces_dropped_and_blocked(self, monkeypatch):
        """The returned summary tallies nodes_moved/episodes_moved/
        edges_recreated/edges_skipped/mentions_recreated/mentions_skipped
        and carries dropped_cross_target + blocked -- never silently
        dropped."""
        recreate_result = SubgraphEdgeResult(
            edges_recreated=1, edges_skipped=1,
            mentions_recreated=1, mentions_skipped=0,
            dropped_cross_target=[{'edge_uuid': 'edge-2', 'reason': 'cross-target'}],
        )
        self._patch_primitives(monkeypatch, recreate_result=recreate_result)
        entity_rows = [{'uuid': 'entity-1', 'name': 'Alice'}, {'uuid': 'entity-2', 'name': 'Bob'}]
        episode_rows = [{'uuid': 'episode-1'}]

        summary = await _mod.merge_graph_family(
            MagicMock(), 'know-live', 'know_live', entity_rows, episode_rows,
        )

        assert summary['nodes_moved'] == 2
        assert summary['episodes_moved'] == 1
        assert summary['nodes_blocked'] == 0
        assert summary['episodes_blocked'] == 0
        assert summary['edges_recreated'] == 1
        assert summary['edges_skipped'] == 1
        assert summary['mentions_recreated'] == 1
        assert summary['mentions_skipped'] == 0
        assert summary['dropped_cross_target'] == recreate_result.dropped_cross_target
        assert summary['blocked'] == []

    @pytest.mark.asyncio
    async def test_empty_family_makes_no_primitive_calls_and_zeroed_summary(self, monkeypatch):
        """No entity rows and no episode rows -> none of the three-phase
        primitives are called, and the summary is all-zero/empty."""
        mocks = self._patch_primitives(monkeypatch)

        summary = await _mod.merge_graph_family(MagicMock(), 'know-live', 'know_live', [], [])

        mocks['create_node'].assert_not_called()
        mocks['create_episode'].assert_not_called()
        mocks['recreate'].assert_not_called()
        mocks['delete_node'].assert_not_called()
        mocks['delete_episode'].assert_not_called()

        assert summary == {
            'nodes_moved': 0, 'episodes_moved': 0,
            'nodes_blocked': 0, 'episodes_blocked': 0,
            'edges_recreated': 0, 'edges_skipped': 0,
            'mentions_recreated': 0, 'mentions_skipped': 0,
            'dropped_cross_target': [], 'blocked': [],
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
# Tests: count_collection_points
# ===========================================================================

class TestCountCollectionPoints:
    """Tests for async count_collection_points(qdrant_client, collection)."""

    @pytest.mark.asyncio
    async def test_calls_count_with_collection_name(self):
        """qdrant_client.count is called with collection_name=collection."""
        client = AsyncMock()
        client.count = AsyncMock(return_value=_make_count_result(0))

        await _mod.count_collection_points(client, 'fused_knowlive')

        client.count.assert_called_once_with(collection_name='fused_knowlive')

    @pytest.mark.asyncio
    async def test_returns_the_qdrant_count(self):
        """Returns the .count value from the Qdrant CountResult."""
        client = AsyncMock()
        client.count = AsyncMock(return_value=_make_count_result(5))

        result = await _mod.count_collection_points(client, 'fused_knowlive')

        assert result == 5

    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_collection(self):
        """A collection with no points returns 0."""
        client = AsyncMock()
        client.count = AsyncMock(return_value=_make_count_result(0))

        result = await _mod.count_collection_points(client, 'fused_default')

        assert result == 0

    @pytest.mark.asyncio
    async def test_none_count_raises(self):
        """An explicit .count is None (indeterminate) -- reviewer follow-up:
        a deletion guard must fail CLOSED (raise, block the delete) rather
        than silently defaulting to 0 (which would fail OPEN and authorize
        delete_empty_collection on an unreadable count)."""
        client = AsyncMock()
        client.count = AsyncMock(return_value=_make_count_result(None))

        with pytest.raises(ValueError):
            await _mod.count_collection_points(client, 'fused_knowlive')

    @pytest.mark.asyncio
    async def test_missing_count_attribute_raises(self):
        """A CountResult with no .count attribute at all is likewise
        indeterminate -- raises rather than silently treating it as 0."""
        client = AsyncMock()
        client.count = AsyncMock(return_value=MagicMock(spec=[]))

        with pytest.raises(ValueError):
            await _mod.count_collection_points(client, 'fused_knowlive')


# ===========================================================================
# Tests: delete_empty_collection
# ===========================================================================

class TestDeleteEmptyCollection:
    """Tests for async delete_empty_collection(qdrant_client, collection, point_count)."""

    @pytest.mark.asyncio
    async def test_zero_count_deletes_and_returns_delete_disposition(self):
        """point_count==0 -> delete_collection() is called, disposition DELETE."""
        client = _make_qdrant_mock()

        disposition = await _mod.delete_empty_collection(client, 'fused_knowlive', 0)

        client.delete_collection.assert_called_once_with('fused_knowlive')
        assert disposition == 'DELETE'

    @pytest.mark.asyncio
    async def test_nonzero_count_does_not_delete_and_returns_unresolved(self):
        """point_count>0 -> delete_collection() is NEVER called, disposition
        UNRESOLVED (deletion blocked -- no data loss)."""
        client = _make_qdrant_mock()

        disposition = await _mod.delete_empty_collection(client, 'fused_knowlive', 5)

        client.delete_collection.assert_not_called()
        assert disposition == 'UNRESOLVED'

    @pytest.mark.asyncio
    async def test_raising_delete_returns_unresolved(self):
        """A raising delete_collection() is caught and reported UNRESOLVED
        rather than propagating -- best-effort, mirrors delete_junk_key."""
        client = _make_qdrant_mock()
        client.delete_collection = AsyncMock(side_effect=RuntimeError('boom'))

        disposition = await _mod.delete_empty_collection(client, 'fused_knowlive', 0)

        assert disposition == 'UNRESOLVED'


# ===========================================================================
# Helpers: run() fixtures
#
# run() queries the SAME graph key for THREE distinct purposes -- Entity
# enumeration (a family sibling), Episodic enumeration (ditto, task 2502),
# and total-node counting (every junk-key candidate, INCLUDING that same
# sibling, per the "plus emptied family siblings" junk-key policy) -- so a
# single graph mock must answer all three ro_query shapes correctly. These
# helpers are shared by TestRunDryRun and TestRunApply.
# ===========================================================================

def _make_run_graph_mock(
    entity_rows: list[list] | None = None,
    episode_rows: list[list] | None = None,
    total_count: int | None = None,
) -> MagicMock:
    """Graph mock whose ro_query dispatches on the Cypher text: the
    Entity-enumeration query returns *entity_rows*, the Episodic-enumeration
    query returns *episode_rows*, and the total-count query returns
    [[*total_count*]]. Defaults total_count to len(entity_rows) +
    len(episode_rows) when not given explicitly -- the total-count query
    (``MATCH (n) ...``) counts every label, so both node kinds contribute."""
    entity_rows = entity_rows if entity_rows is not None else []
    episode_rows = episode_rows if episode_rows is not None else []
    if total_count is None:
        total_count = len(entity_rows) + len(episode_rows)

    async def _ro_query(cypher: str, params: dict | None = None):
        result = MagicMock()
        if 'MENTIONS' in cypher:
            # merge_graph_family's guarded MENTIONS-topology probes -- BOTH
            # directions: create-failed-entity -> mentioning episodes (task
            # 2502 step-16) and create-failed-episode -> mentioned entities
            # (reviewer follow-up, data-loss-barrier-gap case (3)). Each is a
            # dual-label query ('Episodic' AND 'Entity' both appear in its
            # MATCH), so this branch MUST be routed here, before the
            # single-label branches below, or it would be misrouted to the
            # full per-key episode/entity list instead of this
            # (unseeded-by-default) topology read.
            result.result_set = []
        elif 'Episodic' in cypher:
            result.result_set = episode_rows
        elif 'Entity' in cypher:
            result.result_set = entity_rows
        else:
            result.result_set = [[total_count]]
        return result

    graph = MagicMock()
    graph.ro_query = AsyncMock(side_effect=_ro_query)
    graph.query = AsyncMock()
    graph.delete = AsyncMock(return_value=None)
    return graph


def _make_run_graphiti_mock(
    entity_rows_by_key: dict[str, list] | None = None,
    episode_rows_by_key: dict[str, list] | None = None,
    total_count_by_key: dict[str, int] | None = None,
) -> MagicMock:
    """MagicMock graphiti whose _graph_for(key) MEMOIZES one graph mock per
    key -- the same mock instance is returned across repeat calls for a key
    -- configured via _make_run_graph_mock. Keys without an explicit fixture
    default to an empty/zero-count graph. The per-key mocks are exposed via
    ._graphs_by_key for post-call assertions (e.g. .delete/.query
    not-called)."""
    entity_rows_by_key = entity_rows_by_key or {}
    episode_rows_by_key = episode_rows_by_key or {}
    total_count_by_key = total_count_by_key or {}
    graphs: dict[str, MagicMock] = {}

    def _graph_for(key: str) -> MagicMock:
        if key not in graphs:
            rows = entity_rows_by_key.get(key, [])
            episode_rows = episode_rows_by_key.get(key, [])
            count = total_count_by_key.get(key, len(rows) + len(episode_rows))
            graphs[key] = _make_run_graph_mock(rows, episode_rows, count)
        return graphs[key]

    graphiti = MagicMock()
    graphiti._graph_for = MagicMock(side_effect=_graph_for)
    graphiti._graphs_by_key = graphs
    return graphiti


def _make_run_qdrant_mock(
    points_by_collection: dict[str, list] | None = None,
    counts_by_collection: dict[str, int] | None = None,
) -> AsyncMock:
    """AsyncMock qdrant client whose scroll()/count() dispatch on
    collection_name; unlisted collections scroll as empty and count as 0 --
    the same 'clean by default' convention as _make_run_graphiti_mock's
    per-key graphs."""
    points_by_collection = points_by_collection or {}
    counts_by_collection = counts_by_collection or {}

    async def _scroll(*, collection_name: str, **_kwargs):
        return (points_by_collection.get(collection_name, []), None)

    async def _count(*, collection_name: str, **_kwargs):
        return _make_count_result(counts_by_collection.get(collection_name, 0))

    client = AsyncMock()
    client.scroll = AsyncMock(side_effect=_scroll)
    client.count = AsyncMock(side_effect=_count)
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


def _patch_merge_primitives(
    monkeypatch, *,
    recreate_result=None,
    recreate_side_effect=None,
    create_node_side_effect=None,
    create_episode_side_effect=None,
    delete_node_side_effect=None,
    delete_episode_side_effect=None,
):
    """Patch the five three-phase primitives merge_graph_family drives
    (create_moved_node, create_moved_episode, recreate_subgraph_relationships,
    delete_source_node, delete_source_episode) for run()-level orchestration
    tests.

    Cypher-level correctness of each primitive -- and of merge_graph_family's
    own barrier ordering -- is already covered by test_cross_graph_move.py
    and this module's TestMergeGraphFamily; these run()-level tests exercise
    run()'s OWN per-sibling enumeration/dispatch/disposition-folding logic
    instead, so the primitives themselves are patched away here.

    *recreate_side_effect*, when given (typically an exception instance),
    makes recreate_subgraph_relationships RAISE (a SYSTEMIC Phase-B abort)
    instead of returning *recreate_result*."""
    create_node_mock = AsyncMock(side_effect=create_node_side_effect)
    create_episode_mock = AsyncMock(side_effect=create_episode_side_effect)
    recreate_mock = AsyncMock(
        return_value=recreate_result if recreate_result is not None else SubgraphEdgeResult(),
        side_effect=recreate_side_effect,
    )
    delete_node_mock = AsyncMock(side_effect=delete_node_side_effect)
    delete_episode_mock = AsyncMock(side_effect=delete_episode_side_effect)

    monkeypatch.setattr(_mod, 'create_moved_node', create_node_mock)
    monkeypatch.setattr(_mod, 'create_moved_episode', create_episode_mock)
    monkeypatch.setattr(_mod, 'recreate_subgraph_relationships', recreate_mock)
    monkeypatch.setattr(_mod, 'delete_source_node', delete_node_mock)
    monkeypatch.setattr(_mod, 'delete_source_episode', delete_episode_mock)
    return {
        'create_node': create_node_mock,
        'create_episode': create_episode_mock,
        'recreate': recreate_mock,
        'delete_node': delete_node_mock,
        'delete_episode': delete_episode_mock,
    }


# ===========================================================================
# Tests: run() -- dry-run path
# ===========================================================================

class TestRunDryRun:
    """Tests for async run(args, memory_service, *, limit) in dry-run (args.apply=False)."""

    def _scenario(self):
        """One sibling with entity AND episode rows (know-live, unmerged);
        one non-empty JUNK_KEYS entry (my-project); one collection with
        points (fused_dark-factory). Every other alias/collection/key falls
        back to the empty/zero-count default."""
        graphiti = _make_run_graphiti_mock(
            entity_rows_by_key={'know-live': [['uuid-1', 'Node A']]},
            episode_rows_by_key={'know-live': [['episode-1']]},
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
        """Dry-run performs ZERO mutations across every backend: none of the
        three-phase primitives are called, no qdrant upsert/delete_collection,
        no GRAPH.DELETE, and read-only ro_query only (.query is NEVER called
        on any graph -- this includes the Episodic-enumeration query)."""
        mocks = _patch_merge_primitives(monkeypatch)
        memory_service, graphiti, qdrant_client = self._scenario()

        await _mod.run(_run_args(apply=False), memory_service, limit=1000)

        mocks['create_node'].assert_not_called()
        mocks['create_episode'].assert_not_called()
        mocks['recreate'].assert_not_called()
        mocks['delete_node'].assert_not_called()
        mocks['delete_episode'].assert_not_called()
        qdrant_client.upsert.assert_not_called()
        qdrant_client.delete_collection.assert_not_called()
        assert graphiti._graphs_by_key, 'expected at least one graph to have been touched'
        for graph in graphiti._graphs_by_key.values():
            graph.delete.assert_not_called()
            graph.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_report_shape_and_dispositions(self):
        """dry_run=True; a sibling with entity rows is MERGE and carries its
        episode_count/episode_uuids too; a collection with points is MERGE;
        a non-empty JUNK_KEYS entry is UNRESOLVED even in dry-run; an
        unmerged sibling is ALSO UNRESOLVED as a junk-key candidate (its
        entities/episodes are still present, node_count > 0)."""
        memory_service, _, _ = self._scenario()

        report = await _mod.run(_run_args(apply=False), memory_service, limit=1000)

        assert report['dry_run'] is True

        graph_by_sibling = {item['sibling']: item for item in report['graph_family_merges']}
        assert graph_by_sibling['know-live']['canonical'] == 'know_live'
        assert graph_by_sibling['know-live']['node_count'] == 1
        assert graph_by_sibling['know-live']['node_uuids'] == ['uuid-1']
        assert graph_by_sibling['know-live']['episode_count'] == 1
        assert graph_by_sibling['know-live']['episode_uuids'] == ['episode-1']
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
        # know-live's total node count is entity(1) + episode(1) = 2 -- the
        # total-count query counts every label, not just :Entity.
        assert junk_by_key['know-live']['node_count'] == 2
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

    @pytest.mark.asyncio
    async def test_dry_run_sibling_unresolved_when_episode_enumeration_capped(self):
        """A sibling whose Episodic enumeration hits --limit is UNRESOLVED,
        not MERGE -- the same no-silent-caps guard as the Entity enumeration
        cap, now also applied to episode enumeration (task 2502)."""
        episode_rows = [['episode-1'], ['episode-2']]
        graphiti = _make_run_graphiti_mock(
            entity_rows_by_key={'know-live': [['uuid-1', 'Node A']]},
            episode_rows_by_key={'know-live': episode_rows},
        )
        qdrant_client = _make_run_qdrant_mock()
        memory_service = _make_run_memory_service(graphiti, qdrant_client)

        report = await _mod.run(_run_args(apply=False), memory_service, limit=2)

        graph_by_sibling = {item['sibling']: item for item in report['graph_family_merges']}
        assert graph_by_sibling['know-live']['disposition'] == 'UNRESOLVED'

    @pytest.mark.asyncio
    async def test_dry_run_empty_collection_deletions_populated_without_deleting(self):
        """empty_collection_deletions previews DELETE for a count-0 stray
        and UNRESOLVED for a non-empty one, WITHOUT calling
        delete_collection -- dry-run previews, never deletes."""
        graphiti = _make_run_graphiti_mock()
        qdrant_client = _make_run_qdrant_mock(
            counts_by_collection={'fused_knowlive': 0, 'fused_my-project': 4},
        )
        memory_service = _make_run_memory_service(graphiti, qdrant_client)

        report = await _mod.run(_run_args(apply=False), memory_service, limit=1000)

        empty_by_collection = {
            item['collection']: item for item in report['empty_collection_deletions']
        }
        assert empty_by_collection['fused_knowlive']['point_count'] == 0
        assert empty_by_collection['fused_knowlive']['disposition'] == 'DELETE'
        assert empty_by_collection['fused_my-project']['point_count'] == 4
        assert empty_by_collection['fused_my-project']['disposition'] == 'UNRESOLVED'
        qdrant_client.delete_collection.assert_not_called()


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

    @pytest.mark.asyncio
    async def test_apply_drives_three_phase_primitives_with_rewrite_group_id(self, monkeypatch):
        """create_moved_node is called once per sibling entity row, with
        source=sibling, target=canonical, and rewrite_group_id=canonical --
        the Phase-2 identity rewrite (PRD decision 6); recreate_subgraph_
        relationships and delete_source_node are each driven once
        afterwards; no episode primitive runs (this sibling has no episode
        rows); and the know-live graph_family item carries the merge
        summary (nodes_moved etc.) -- replacing the old single-call
        move_entity_across_graphs assertion with the three-phase primitives."""
        mocks = _patch_merge_primitives(monkeypatch)
        memory_service, graphiti, _ = self._scenario()

        report = await _mod.run(_run_args(apply=True), memory_service, limit=1000)

        mocks['create_node'].assert_awaited_once()
        call = mocks['create_node'].call_args
        assert call.args == (graphiti, 'uuid-1', 'know-live', 'know_live')
        assert call.kwargs == {'rewrite_group_id': 'know_live'}
        mocks['create_episode'].assert_not_called()
        mocks['recreate'].assert_awaited_once()
        mocks['delete_node'].assert_awaited_once()
        mocks['delete_episode'].assert_not_called()

        graph_by_sibling = {item['sibling']: item for item in report['graph_family_merges']}
        know_live_item = graph_by_sibling['know-live']
        assert know_live_item['nodes_moved'] == 1
        assert know_live_item['episodes_moved'] == 0
        assert know_live_item['edges_recreated'] == 0
        assert know_live_item['mentions_recreated'] == 0
        assert know_live_item['disposition'] == 'MERGE'

    @pytest.mark.asyncio
    async def test_apply_clean_multi_node_family_stays_merge_with_summary(self, monkeypatch):
        """A family with 2 entities + 1 episode, whose Phase-B batch reports
        a clean (no skip/drop/blocked) edge + mention recreate, keeps
        disposition MERGE and carries the full merge summary on the family
        item -- the structural guarantee that a clean multi-node merge
        (intra-family RELATES_TO edge + sibling-resident Episodic/MENTIONS)
        is preserved end-to-end through run()."""
        clean_result = SubgraphEdgeResult(edges_recreated=1, mentions_recreated=1)
        mocks = _patch_merge_primitives(monkeypatch, recreate_result=clean_result)
        graphiti = _make_run_graphiti_mock(
            entity_rows_by_key={'know-live': [['uuid-1', 'Node A'], ['uuid-2', 'Node B']]},
            episode_rows_by_key={'know-live': [['episode-1']]},
            total_count_by_key={'know-live': 0},
        )
        qdrant_client = _make_run_qdrant_mock()
        memory_service = _make_run_memory_service(graphiti, qdrant_client)

        report = await _mod.run(_run_args(apply=True), memory_service, limit=1000)

        assert mocks['create_node'].await_count == 2
        mocks['create_episode'].assert_awaited_once()
        assert mocks['delete_node'].await_count == 2
        mocks['delete_episode'].assert_awaited_once()

        graph_by_sibling = {item['sibling']: item for item in report['graph_family_merges']}
        know_live_item = graph_by_sibling['know-live']
        assert know_live_item['disposition'] == 'MERGE'
        assert know_live_item['nodes_moved'] == 2
        assert know_live_item['episodes_moved'] == 1
        assert know_live_item['edges_recreated'] == 1
        assert know_live_item['mentions_recreated'] == 1
        assert know_live_item['edges_skipped'] == 0
        assert know_live_item['mentions_skipped'] == 0
        assert know_live_item['dropped_cross_target'] == []
        assert know_live_item['blocked'] == []
        assert _mod.has_unresolved(report) is False

    @pytest.mark.asyncio
    async def test_apply_family_edges_skipped_downgrades_to_unresolved(self, monkeypatch):
        """A merge summary reporting edges_skipped>0 downgrades the family
        item's disposition to UNRESOLVED -- the exact silent-signal failure
        this task fixes: edge loss must not exit 0."""
        lossy_result = SubgraphEdgeResult(edges_recreated=1, edges_skipped=1)
        _patch_merge_primitives(monkeypatch, recreate_result=lossy_result)
        graphiti = _make_run_graphiti_mock(
            entity_rows_by_key={'know-live': [['uuid-1', 'Node A'], ['uuid-2', 'Node B']]},
            total_count_by_key={'know-live': 0},
        )
        qdrant_client = _make_run_qdrant_mock()
        memory_service = _make_run_memory_service(graphiti, qdrant_client)

        report = await _mod.run(_run_args(apply=True), memory_service, limit=1000)

        graph_by_sibling = {item['sibling']: item for item in report['graph_family_merges']}
        assert graph_by_sibling['know-live']['disposition'] == 'UNRESOLVED'
        assert _mod.has_unresolved(report) is True

    @pytest.mark.asyncio
    async def test_apply_family_mentions_skipped_downgrades_to_unresolved(self, monkeypatch):
        """A merge summary reporting mentions_skipped>0 downgrades the
        family item's disposition to UNRESOLVED."""
        lossy_result = SubgraphEdgeResult(mentions_recreated=0, mentions_skipped=1)
        _patch_merge_primitives(monkeypatch, recreate_result=lossy_result)
        graphiti = _make_run_graphiti_mock(
            entity_rows_by_key={'know-live': [['uuid-1', 'Node A']]},
            episode_rows_by_key={'know-live': [['episode-1']]},
            total_count_by_key={'know-live': 0},
        )
        qdrant_client = _make_run_qdrant_mock()
        memory_service = _make_run_memory_service(graphiti, qdrant_client)

        report = await _mod.run(_run_args(apply=True), memory_service, limit=1000)

        graph_by_sibling = {item['sibling']: item for item in report['graph_family_merges']}
        assert graph_by_sibling['know-live']['disposition'] == 'UNRESOLVED'
        assert _mod.has_unresolved(report) is True

    @pytest.mark.asyncio
    async def test_apply_family_dropped_cross_target_downgrades_to_unresolved(self, monkeypatch):
        """A merge summary carrying a non-empty dropped_cross_target list
        downgrades the family item's disposition to UNRESOLVED, even though
        edges_skipped/mentions_skipped are both 0."""
        lossy_result = SubgraphEdgeResult(
            dropped_cross_target=[{'edge_uuid': 'edge-1', 'reason': 'cross-target'}],
        )
        _patch_merge_primitives(monkeypatch, recreate_result=lossy_result)
        graphiti = _make_run_graphiti_mock(
            entity_rows_by_key={'know-live': [['uuid-1', 'Node A']]},
            total_count_by_key={'know-live': 0},
        )
        qdrant_client = _make_run_qdrant_mock()
        memory_service = _make_run_memory_service(graphiti, qdrant_client)

        report = await _mod.run(_run_args(apply=True), memory_service, limit=1000)

        graph_by_sibling = {item['sibling']: item for item in report['graph_family_merges']}
        assert graph_by_sibling['know-live']['disposition'] == 'UNRESOLVED'
        assert _mod.has_unresolved(report) is True

    @pytest.mark.asyncio
    async def test_apply_family_blocked_downgrades_to_unresolved_and_withholds_deletion(self, monkeypatch):
        """A merge summary carrying a non-empty blocked list downgrades the
        family item's disposition to UNRESOLVED, AND the blocked uuid's
        source deletion is withheld (create-before-delete preserved) -- the
        residual node keeps this sibling's junk-key count > 0, so its
        GRAPH.DELETE is guarded off too."""
        blocked_result = SubgraphEdgeResult(
            blocked=[{'kind': 'edge', 'uuid': 'edge-1', 'reason': 'boom', 'node_uuids': ['uuid-1']}],
        )
        mocks = _patch_merge_primitives(monkeypatch, recreate_result=blocked_result)
        graphiti = _make_run_graphiti_mock(
            entity_rows_by_key={'know-live': [['uuid-1', 'Node A']]},
            total_count_by_key={'know-live': 1},  # residual: source deletion withheld
        )
        qdrant_client = _make_run_qdrant_mock()
        memory_service = _make_run_memory_service(graphiti, qdrant_client)

        report = await _mod.run(_run_args(apply=True), memory_service, limit=1000)

        mocks['delete_node'].assert_not_called()

        graph_by_sibling = {item['sibling']: item for item in report['graph_family_merges']}
        assert graph_by_sibling['know-live']['disposition'] == 'UNRESOLVED'
        assert _mod.has_unresolved(report) is True

        junk_by_key = {item['key']: item for item in report['junk_key_deletions']}
        assert junk_by_key['know-live']['disposition'] == 'UNRESOLVED'
        graphiti._graphs_by_key['know-live'].delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_apply_family_phase_a_create_failure_downgrades_to_unresolved(self, monkeypatch):
        """A Phase-A create_moved_node failure (nodes_blocked>0 in the merge
        summary) downgrades the family item's disposition to UNRESOLVED,
        even though the SubgraphEdgeResult itself reports no skip/drop/
        blocked -- the create failure alone must not be silently absorbed --
        and the failed uuid's source deletion is withheld."""
        def _create_node_side_effect(graphiti_arg, uuid, *rest, **kwargs):
            raise RuntimeError('boom')

        mocks = _patch_merge_primitives(
            monkeypatch, create_node_side_effect=_create_node_side_effect,
        )
        graphiti = _make_run_graphiti_mock(
            entity_rows_by_key={'know-live': [['uuid-1', 'Node A']]},
            total_count_by_key={'know-live': 1},
        )
        qdrant_client = _make_run_qdrant_mock()
        memory_service = _make_run_memory_service(graphiti, qdrant_client)

        report = await _mod.run(_run_args(apply=True), memory_service, limit=1000)

        mocks['delete_node'].assert_not_called()

        graph_by_sibling = {item['sibling']: item for item in report['graph_family_merges']}
        know_live_item = graph_by_sibling['know-live']
        assert know_live_item['nodes_blocked'] == 1
        assert know_live_item['disposition'] == 'UNRESOLVED'
        assert _mod.has_unresolved(report) is True

    @pytest.mark.asyncio
    async def test_apply_family_phase_b_systemic_abort_downgrades_to_unresolved_and_withholds_deletion(
        self, monkeypatch,
    ):
        """A SYSTEMIC Phase-B abort (recreate_subgraph_relationships raises)
        must withhold the WHOLE batch's source deletions -- not just the
        uuids a recovered partial_result.blocked happens to name -- so the
        know-live family stays UNRESOLVED (non-zero --apply exit) and the
        emptied sibling's junk-key GRAPH.DELETE stays guarded off (residual
        count>0). RED today: merge_graph_family only recovers
        partial_result and falls through to Phase C, so uuid-1 gets
        deleted, nodes_blocked stays 0, the family stays MERGE, and
        --apply exits 0 (silent data loss)."""
        abort_exc = RuntimeError('phase b systemic failure')
        mocks = _patch_merge_primitives(monkeypatch, recreate_side_effect=abort_exc)
        graphiti = _make_run_graphiti_mock(
            entity_rows_by_key={'know-live': [['uuid-1', 'Node A']]},
            total_count_by_key={'know-live': 1},  # residual: source deletion withheld
        )
        qdrant_client = _make_run_qdrant_mock()
        memory_service = _make_run_memory_service(graphiti, qdrant_client)

        report = await _mod.run(_run_args(apply=True), memory_service, limit=1000)

        mocks['delete_node'].assert_not_called()
        mocks['delete_episode'].assert_not_called()

        graph_by_sibling = {item['sibling']: item for item in report['graph_family_merges']}
        assert graph_by_sibling['know-live']['disposition'] == 'UNRESOLVED'
        assert _mod.has_unresolved(report) is True

        junk_by_key = {item['key']: item for item in report['junk_key_deletions']}
        assert junk_by_key['know-live']['disposition'] == 'UNRESOLVED'
        graphiti._graphs_by_key['know-live'].delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_apply_upserts_with_canonical_user_id_and_deletes_source_collection(self, monkeypatch):
        """qdrant upsert is called with collection_name=target and the
        point's payload user_id rewritten to canonical; the not-capped
        source collection is deleted; the collection item carries
        points_upserted/source_deleted=True."""
        _patch_merge_primitives(monkeypatch)
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
        _patch_merge_primitives(monkeypatch)
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
        _patch_merge_primitives(monkeypatch)
        memory_service, _, _ = self._scenario()

        report = await _mod.run(_run_args(apply=True), memory_service, limit=1000)

        assert report['dry_run'] is False

    @pytest.mark.asyncio
    async def test_apply_capped_collection_not_deleted_and_stays_unresolved(self, monkeypatch):
        """A collection whose scroll hits --limit is NOT deleted even with
        --apply (no data loss on a partial migration) and its disposition
        stays UNRESOLVED."""
        _patch_merge_primitives(monkeypatch)
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

    @pytest.mark.asyncio
    async def test_apply_deletes_zero_count_empty_collection_and_leaves_nonzero_unresolved(self, monkeypatch):
        """delete_empty_collection deletes the zero-count stray
        (fused_knowlive, disposition DELETE) but leaves the non-empty one
        (fused_my-project, disposition UNRESOLVED) untouched --
        delete_collection is never called on it."""
        _patch_merge_primitives(monkeypatch)
        graphiti = _make_run_graphiti_mock()
        qdrant_client = _make_run_qdrant_mock(
            counts_by_collection={'fused_knowlive': 0, 'fused_my-project': 4},
        )
        memory_service = _make_run_memory_service(graphiti, qdrant_client)

        report = await _mod.run(_run_args(apply=True), memory_service, limit=1000)

        qdrant_client.delete_collection.assert_any_call('fused_knowlive')
        no_delete_calls = [
            c for c in qdrant_client.delete_collection.call_args_list
            if c.args and c.args[0] == 'fused_my-project'
        ]
        assert no_delete_calls == []

        empty_by_collection = {
            item['collection']: item for item in report['empty_collection_deletions']
        }
        assert empty_by_collection['fused_knowlive']['disposition'] == 'DELETE'
        assert empty_by_collection['fused_my-project']['disposition'] == 'UNRESOLVED'
        assert _mod.has_unresolved(report) is True
