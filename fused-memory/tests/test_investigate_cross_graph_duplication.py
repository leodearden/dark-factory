"""Tests for scripts/investigate_cross_graph_duplication.py.

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

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'investigate_cross_graph_duplication.py'


def _load_module() -> types.ModuleType:
    """Load investigate_cross_graph_duplication.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators work correctly.
    """
    mod_name = 'investigate_cross_graph_duplication'
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
    """Minimal stand-in for conftest.make_graph_mock, scoped to this test module
    so it is usable without the fixture (kept consistent with its shape;
    copied from test_purge_knowlive_namespace.py)."""
    result = MagicMock()
    result.result_set = rows if rows is not None else []
    graph = MagicMock()
    graph.ro_query = AsyncMock(return_value=result)
    graph.query = AsyncMock(return_value=result)
    return graph


# ===========================================================================
# Tests: is_path_shaped_name
# ===========================================================================

class TestIsPathShapedName:
    """Tests for is_path_shaped_name(name) -> bool."""

    @pytest.mark.parametrize('name', [
        '/home/leo/src/dark-factory',
        '-home-leo-src-dark-factory',
        '-abs-path-proj',
    ])
    def test_path_shaped_names_are_true(self, name):
        """Filesystem-path-shaped graph names are detected as True."""
        assert _mod.is_path_shaped_name(name) is True

    @pytest.mark.parametrize('name', [
        'dark_factory',
        'dark-factory',
        'know_live',
        'know-live',
        'knowlive',
        'reify',
    ])
    def test_clean_project_keys_are_false(self, name):
        """Clean project keys (no path shape) are detected as False."""
        assert _mod.is_path_shaped_name(name) is False


# ===========================================================================
# Tests: detect_collision_groups
# ===========================================================================

class TestDetectCollisionGroups:
    """Tests for detect_collision_groups(graph_names) -> dict."""

    def test_two_variants_collide_into_one_group(self):
        """dark_factory/dark-factory collide into one canonical group; the
        lone 'reify' does not, and there are no path leaks."""
        result = _mod.detect_collision_groups(['dark_factory', 'dark-factory', 'reify'])

        assert result['collisions'] == [
            {'canonical': 'dark_factory', 'variants': ['dark-factory', 'dark_factory'], 'count': 2},
        ]
        assert result['suspected_path_leaks'] == []

    def test_know_live_variants_collide(self):
        """know_live/know-live collide the same way as dark_factory/dark-factory."""
        result = _mod.detect_collision_groups(['know_live', 'know-live'])

        assert result['collisions'] == [
            {'canonical': 'know_live', 'variants': ['know-live', 'know_live'], 'count': 2},
        ]

    def test_lone_name_has_no_collision(self):
        """A single, unambiguous graph name yields no collision and no path leak."""
        result = _mod.detect_collision_groups(['reify'])

        assert result['collisions'] == []
        assert result['suspected_path_leaks'] == []

    def test_path_shaped_name_is_flagged_not_merged(self):
        """The mangled path name is reported under suspected_path_leaks and is
        NOT folded into the dark_factory collision group's variants."""
        result = _mod.detect_collision_groups(
            ['dark_factory', 'dark-factory', '-home-leo-src-dark-factory'],
        )

        assert result['suspected_path_leaks'] == ['-home-leo-src-dark-factory']
        collision = next(c for c in result['collisions'] if c['canonical'] == 'dark_factory')
        assert '-home-leo-src-dark-factory' not in collision['variants']
        assert collision['count'] == 2

    def test_legacy_knowlive_not_merged_into_know_live(self):
        """Legacy no-separator 'knowlive' stays distinct from the know_live
        collision group (task 515's re-key was cancelled; not our call to
        reverse here)."""
        result = _mod.detect_collision_groups(['know_live', 'know-live', 'knowlive'])

        collision = next(c for c in result['collisions'] if c['canonical'] == 'know_live')
        assert 'knowlive' not in collision['variants']
        assert collision['count'] == 2
        assert result['suspected_path_leaks'] == []

    def test_full_task_scenario_matches_expected_shape(self):
        """The real GRAPH.LIST from task 2116: two collision families plus one
        path leak plus two untouched clean singletons (reify, knowlive)."""
        graph_names = [
            'reify', 'dark_factory', 'dark-factory',
            '-home-leo-src-dark-factory', 'know_live', 'know-live', 'knowlive',
        ]

        result = _mod.detect_collision_groups(graph_names)

        assert result['collisions'] == [
            {'canonical': 'dark_factory', 'variants': ['dark-factory', 'dark_factory'], 'count': 2},
            {'canonical': 'know_live', 'variants': ['know-live', 'know_live'], 'count': 2},
        ]
        assert result['suspected_path_leaks'] == ['-home-leo-src-dark-factory']

    def test_deterministic_ordering(self):
        """Output list ordering does not depend on input ordering."""
        names = ['reify', 'dark-factory', 'dark_factory', 'know-live', 'know_live']

        r1 = _mod.detect_collision_groups(names)
        r2 = _mod.detect_collision_groups(list(reversed(names)))

        assert r1 == r2


# ===========================================================================
# Tests: probe_node_across_graphs
# ===========================================================================

class TestProbeNodeAcrossGraphs:
    """Tests for async probe_node_across_graphs(graphiti, uuid, graph_names) -> list[dict]."""

    @pytest.mark.asyncio
    async def test_calls_graph_for_once_per_graph_name(self):
        """graphiti._graph_for is called exactly once for each graph name."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        await _mod.probe_node_across_graphs(
            graphiti, 'uuid-1', ['reify', 'dark_factory', 'know_live'],
        )

        assert graphiti._graph_for.call_count == 3
        called_names = {c.args[0] for c in graphiti._graph_for.call_args_list}
        assert called_names == {'reify', 'dark_factory', 'know_live'}

    @pytest.mark.asyncio
    async def test_uses_ro_query_not_query(self):
        """The probe is read-only: ro_query is used once per graph, and
        query is NEVER called."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        await _mod.probe_node_across_graphs(
            graphiti, 'uuid-1', ['reify', 'dark_factory', 'know_live'],
        )

        assert graph.ro_query.call_count == 3
        graph.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_uuid_passed_as_cypher_parameter(self):
        """The target uuid is passed as a Cypher query parameter, not
        string-interpolated into the query text."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        await _mod.probe_node_across_graphs(graphiti, 'target-uuid', ['reify'])

        call = graph.ro_query.call_args
        params = call.args[1] if len(call.args) > 1 else call.kwargs.get('params')
        assert params == {'uuid': 'target-uuid'}

    @pytest.mark.asyncio
    async def test_returns_one_entry_per_graph_with_nonempty_result(self):
        """Graphs with a matching row contribute one normalized entry each;
        a graph with an empty result_set contributes none."""
        graphiti = MagicMock()

        def _graph_for(name):
            if name in ('reify', 'dark_factory'):
                return _make_graph_mock([['target-uuid', 'orchestrator', 'reify']])
            return _make_graph_mock([])  # know_live: not present

        graphiti._graph_for = MagicMock(side_effect=_graph_for)

        result = await _mod.probe_node_across_graphs(
            graphiti, 'target-uuid', ['reify', 'dark_factory', 'know_live'],
        )

        assert result == [
            {'graph': 'reify', 'uuid': 'target-uuid', 'name': 'orchestrator', 'group_id': 'reify'},
            {'graph': 'dark_factory', 'uuid': 'target-uuid', 'name': 'orchestrator', 'group_id': 'reify'},
        ]

    @pytest.mark.asyncio
    async def test_empty_result_set_is_omitted(self):
        """A graph with no matching node is omitted from the result entirely."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        result = await _mod.probe_node_across_graphs(graphiti, 'uuid-1', ['reify'])

        assert result == []

    @pytest.mark.asyncio
    async def test_empty_graph_names_returns_empty_list_without_calls(self):
        """No graph names -> no _graph_for calls, empty result."""
        graphiti = MagicMock()
        graphiti._graph_for = MagicMock()

        result = await _mod.probe_node_across_graphs(graphiti, 'uuid-1', [])

        assert result == []
        graphiti._graph_for.assert_not_called()


# ===========================================================================
# Tests: classify_config_routing
# ===========================================================================

class TestClassifyConfigRouting:
    """Tests for classify_config_routing(collision_result, presence) -> dict."""

    NO_COLLISIONS = {'collisions': [], 'suspected_path_leaks': []}

    def test_confirmed_cross_graph_node_leak_when_present_in_multiple_graphs(self):
        """Task scenario: node present in 3 graphs with group_id='reify' in
        all of them -> CONFIRMED with 'cross_graph_node_leak'."""
        presence = [
            {'graph': 'reify', 'uuid': 'u1', 'name': 'orchestrator', 'group_id': 'reify'},
            {'graph': 'dark_factory', 'uuid': 'u1', 'name': 'orchestrator', 'group_id': 'reify'},
            {'graph': 'know_live', 'uuid': 'u1', 'name': 'orchestrator', 'group_id': 'reify'},
        ]

        verdict = _mod.classify_config_routing(self.NO_COLLISIONS, presence)

        assert verdict['confirmed'] is True
        assert 'cross_graph_node_leak' in verdict['signals']

    def test_single_graph_but_mismatched_group_id_still_confirms(self):
        """Even with the node present in only one graph, a mismatch between
        that graph's canonical key and the node's recorded group_id is
        itself a routing leak signal."""
        presence = [
            {'graph': 'dark_factory', 'uuid': 'u1', 'name': 'orchestrator', 'group_id': 'reify'},
        ]

        verdict = _mod.classify_config_routing(self.NO_COLLISIONS, presence)

        assert verdict['confirmed'] is True
        assert 'cross_graph_node_leak' in verdict['signals']

    def test_confirmed_name_normalization_collision_when_collisions_present(self):
        """Non-empty collisions -> CONFIRMED with 'name_normalization_collision',
        independent of presence (which here is consistent + <=1 graph)."""
        collision_result = {
            'collisions': [
                {'canonical': 'dark_factory', 'variants': ['dark-factory', 'dark_factory'], 'count': 2},
            ],
            'suspected_path_leaks': [],
        }
        presence = [{'graph': 'reify', 'uuid': 'u1', 'name': 'orchestrator', 'group_id': 'reify'}]

        verdict = _mod.classify_config_routing(collision_result, presence)

        assert verdict['confirmed'] is True
        assert 'name_normalization_collision' in verdict['signals']
        assert 'cross_graph_node_leak' not in verdict['signals']

    def test_confirmed_suspected_path_leak_when_path_leaks_present(self):
        """Non-empty suspected_path_leaks -> CONFIRMED with 'suspected_path_leak',
        independent of presence/collisions."""
        collision_result = {'collisions': [], 'suspected_path_leaks': ['-home-leo-src-dark-factory']}
        presence = [{'graph': 'reify', 'uuid': 'u1', 'name': 'orchestrator', 'group_id': 'reify'}]

        verdict = _mod.classify_config_routing(collision_result, presence)

        assert verdict['confirmed'] is True
        assert 'suspected_path_leak' in verdict['signals']
        assert 'cross_graph_node_leak' not in verdict['signals']
        assert 'name_normalization_collision' not in verdict['signals']

    def test_denied_when_no_signals_present(self):
        """No collisions, no path leaks, node present in exactly one graph
        consistent with its own group_id -> DENIED with empty signals and
        an explanatory rationale."""
        presence = [{'graph': 'reify', 'uuid': 'u1', 'name': 'orchestrator', 'group_id': 'reify'}]

        verdict = _mod.classify_config_routing(self.NO_COLLISIONS, presence)

        assert verdict['confirmed'] is False
        assert verdict['signals'] == []
        assert 'no routing anomaly detected' in verdict['rationale'].lower()

    def test_denied_when_node_not_found_anywhere(self):
        """An empty presence list (with no collisions/leaks) is also a denial."""
        verdict = _mod.classify_config_routing(self.NO_COLLISIONS, [])

        assert verdict['confirmed'] is False
        assert verdict['signals'] == []

    def test_return_shape_has_exactly_the_three_keys(self):
        """The verdict dict has exactly {confirmed, signals, rationale}."""
        verdict = _mod.classify_config_routing(self.NO_COLLISIONS, [])

        assert set(verdict.keys()) == {'confirmed', 'signals', 'rationale'}
