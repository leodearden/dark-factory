"""Tests for scripts/investigate_cross_graph_duplication.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution -- mirrors the pattern in
test_purge_knowlive_namespace.py.
"""
from __future__ import annotations

import importlib.util
import json
import os
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

    @pytest.mark.parametrize('name', [
        'my-home-src-app',
        'user-var-config-service',
    ])
    def test_multiword_clean_keys_with_filesystem_tokens_are_not_false_positives(self, name):
        """Regression: a legitimate multi-word project key that happens to
        contain filesystem-ish tokens (home/src/var/...) across >=4 hyphen
        segments is NOT a path leak. Only an actual leading separator (or
        an embedded '/') makes a name path-shaped -- these names have
        neither, so they must not be pulled out of collision detection."""
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

    @pytest.mark.asyncio
    async def test_continues_past_raising_graph_and_records_error(self):
        """A graph whose ro_query raises (e.g. a malformed/inaccessible
        path-leak graph) is skipped and recorded into *errors* -- it does
        not abort the probe of the remaining graphs."""
        graphiti = MagicMock()

        def _graph_for(name):
            if name == '-home-leo-src-dark-factory':
                graph = MagicMock()
                graph.ro_query = AsyncMock(side_effect=RuntimeError('graph key not found'))
                graph.query = AsyncMock()
                return graph
            return _make_graph_mock([['target-uuid', 'orchestrator', 'reify']])

        graphiti._graph_for = MagicMock(side_effect=_graph_for)
        errors: list = []

        result = await _mod.probe_node_across_graphs(
            graphiti, 'target-uuid',
            ['reify', '-home-leo-src-dark-factory', 'dark_factory'],
            errors=errors,
        )

        assert result == [
            {'graph': 'reify', 'uuid': 'target-uuid', 'name': 'orchestrator', 'group_id': 'reify'},
            {'graph': 'dark_factory', 'uuid': 'target-uuid', 'name': 'orchestrator', 'group_id': 'reify'},
        ]
        assert errors == [{'graph': '-home-leo-src-dark-factory', 'error': 'graph key not found'}]

    @pytest.mark.asyncio
    async def test_raising_graph_is_swallowed_when_errors_not_supplied(self):
        """If the caller doesn't pass an *errors* accumulator, a raising
        graph is still skipped rather than propagating -- the *errors*
        param is optional, not required for the read-only guarantee."""
        graphiti = MagicMock()

        def _graph_for(name):
            if name == 'bad-graph':
                graph = MagicMock()
                graph.ro_query = AsyncMock(side_effect=RuntimeError('boom'))
                return graph
            return _make_graph_mock([])

        graphiti._graph_for = MagicMock(side_effect=_graph_for)

        result = await _mod.probe_node_across_graphs(graphiti, 'uuid-1', ['bad-graph', 'reify'])

        assert result == []


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


# ===========================================================================
# Tests: build_investigation_report
# ===========================================================================

class TestBuildInvestigationReport:
    """Tests for build_investigation_report(target_uuid, all_graphs, presence,
    collision_result, verdict) -> dict."""

    def test_report_shape_and_passthrough_fields(self):
        """Every input is threaded through to its corresponding report key,
        with graph_count derived from len(all_graphs)."""
        target_uuid = 'f02a32ea-0efd-4865-94b4-97a412d8ffda'
        all_graphs = ['reify', 'dark_factory', 'know_live']
        presence = [
            {'graph': 'reify', 'uuid': target_uuid, 'name': 'orchestrator', 'group_id': 'reify'},
            {'graph': 'dark_factory', 'uuid': target_uuid, 'name': 'orchestrator', 'group_id': 'reify'},
        ]
        collision_result = {'collisions': [], 'suspected_path_leaks': []}
        verdict = {'confirmed': True, 'signals': ['cross_graph_node_leak'], 'rationale': 'because'}

        report = _mod.build_investigation_report(
            target_uuid, all_graphs, presence, collision_result, verdict,
        )

        assert report['target_uuid'] == target_uuid
        assert report['all_graphs'] == all_graphs
        assert report['graph_count'] == 3
        assert report['present_in'] == presence
        assert report['collision_groups'] == collision_result['collisions']
        assert report['suspected_path_leaks'] == collision_result['suspected_path_leaks']
        assert report['verdict'] == verdict

    def test_scope_is_single_node_when_no_collisions_or_leaks(self):
        """No collisions and no path leaks -> scope='single_node'."""
        collision_result = {'collisions': [], 'suspected_path_leaks': []}
        verdict = {'confirmed': False, 'signals': [], 'rationale': 'none'}

        report = _mod.build_investigation_report('u1', ['reify'], [], collision_result, verdict)

        assert report['scope'] == 'single_node'

    def test_scope_is_systemic_when_collisions_present(self):
        """Non-empty collisions -> scope='systemic'."""
        collision_result = {
            'collisions': [
                {'canonical': 'dark_factory', 'variants': ['dark-factory', 'dark_factory'], 'count': 2},
            ],
            'suspected_path_leaks': [],
        }
        verdict = {'confirmed': True, 'signals': ['name_normalization_collision'], 'rationale': 'x'}

        report = _mod.build_investigation_report(
            'u1', ['dark_factory', 'dark-factory'], [], collision_result, verdict,
        )

        assert report['scope'] == 'systemic'

    def test_scope_is_systemic_when_path_leaks_present(self):
        """Non-empty suspected_path_leaks -> scope='systemic'."""
        collision_result = {'collisions': [], 'suspected_path_leaks': ['-home-leo-src-dark-factory']}
        verdict = {'confirmed': True, 'signals': ['suspected_path_leak'], 'rationale': 'x'}

        report = _mod.build_investigation_report(
            'u1', ['-home-leo-src-dark-factory'], [], collision_result, verdict,
        )

        assert report['scope'] == 'systemic'

    def test_pure_no_io_referentially_stable(self):
        """Calling twice with identical inputs returns equal dicts (pure,
        no I/O, no hidden mutable state between calls)."""
        collision_result = {'collisions': [], 'suspected_path_leaks': []}
        verdict = {'confirmed': False, 'signals': [], 'rationale': 'none'}

        r1 = _mod.build_investigation_report('u1', ['reify'], [], collision_result, verdict)
        r2 = _mod.build_investigation_report('u1', ['reify'], [], collision_result, verdict)

        assert r1 == r2

    def test_errors_defaults_to_empty_list_when_omitted(self):
        """No *errors* argument -> report['errors'] == [] (never missing,
        never None)."""
        collision_result = {'collisions': [], 'suspected_path_leaks': []}
        verdict = {'confirmed': False, 'signals': [], 'rationale': 'none'}

        report = _mod.build_investigation_report('u1', ['reify'], [], collision_result, verdict)

        assert report['errors'] == []

    def test_errors_are_surfaced_verbatim_when_supplied(self):
        """A non-empty *errors* list (recorded by probe_node_across_graphs)
        is threaded through to the report unchanged."""
        collision_result = {'collisions': [], 'suspected_path_leaks': []}
        verdict = {'confirmed': False, 'signals': [], 'rationale': 'none'}
        errors = [{'graph': 'bad-graph', 'error': 'boom'}]

        report = _mod.build_investigation_report(
            'u1', ['reify', 'bad-graph'], [], collision_result, verdict, errors=errors,
        )

        assert report['errors'] == errors


# ===========================================================================
# Tests: run() -- end-to-end orchestration
# ===========================================================================

class TestRun:
    """Tests for async run(args, memory_service) -> dict."""

    def _args(self, uuid=None):
        import types as _types
        return _types.SimpleNamespace(uuid=uuid)

    def _make_memory_service(self, graphs, presence_rows_by_graph=None):
        """AsyncMock memory_service with .graphiti wired for list_graphs
        (GRAPH.LIST) + _graph_for (one stable mock per graph name), and
        delete_memory/update_edge as AsyncMocks for the read-only guarantee.
        Returns (memory_service, graph_mocks) so tests can inspect the
        per-graph mocks directly."""
        rows_by_graph = presence_rows_by_graph or {}
        graph_mocks = {name: _make_graph_mock(rows_by_graph.get(name, [])) for name in graphs}

        memory_service = AsyncMock()
        graphiti = MagicMock()
        graphiti.list_graphs = AsyncMock(return_value=graphs)
        graphiti._graph_for = MagicMock(side_effect=lambda name: graph_mocks[name])
        memory_service.graphiti = graphiti
        memory_service.delete_memory = AsyncMock(return_value=None)
        memory_service.update_edge = AsyncMock(return_value=None)
        return memory_service, graph_mocks

    @pytest.mark.asyncio
    async def test_calls_list_graphs_for_graph_list(self):
        """memory_service.graphiti.list_graphs() enumerates GRAPH.LIST."""
        memory_service, _ = self._make_memory_service(['reify', 'dark_factory'])

        await _mod.run(self._args(uuid='some-uuid'), memory_service)

        memory_service.graphiti.list_graphs.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_report_built_end_to_end(self):
        """The returned report has present_in, collision_groups, verdict,
        and scope -- the full pipeline ran, not a partial stub."""
        graphs = ['reify', 'dark_factory', 'know_live']
        rows_by_graph = {
            'reify': [['target-uuid', 'orchestrator', 'reify']],
            'dark_factory': [['target-uuid', 'orchestrator', 'reify']],
            'know_live': [['target-uuid', 'orchestrator', 'reify']],
        }
        memory_service, _ = self._make_memory_service(graphs, rows_by_graph)

        report = await _mod.run(self._args(uuid='target-uuid'), memory_service)

        assert report['target_uuid'] == 'target-uuid'
        assert report['all_graphs'] == graphs
        assert len(report['present_in']) == 3
        assert report['collision_groups'] == []
        assert report['verdict']['confirmed'] is True
        assert 'cross_graph_node_leak' in report['verdict']['signals']
        assert report['scope'] == 'single_node'

    @pytest.mark.asyncio
    async def test_uuid_none_falls_back_to_target_node_uuid(self):
        """args.uuid=None uses the module's default TARGET_NODE_UUID."""
        memory_service, _ = self._make_memory_service(['reify'])

        report = await _mod.run(self._args(uuid=None), memory_service)

        assert report['target_uuid'] == _mod.TARGET_NODE_UUID

    @pytest.mark.asyncio
    async def test_strictly_read_only(self):
        """run() never issues a mutating graph query, memory delete, or
        edge update."""
        graphs = ['reify', 'dark_factory']
        memory_service, graph_mocks = self._make_memory_service(graphs)

        await _mod.run(self._args(uuid='some-uuid'), memory_service)

        for graph in graph_mocks.values():
            graph.query.assert_not_called()
        memory_service.delete_memory.assert_not_called()
        memory_service.update_edge.assert_not_called()

    @pytest.mark.asyncio
    async def test_probe_failure_surfaced_in_report_without_aborting_run(self):
        """A single graph raising on probe does not abort run(): the
        failure is surfaced under report['errors'] and the report is still
        built from the remaining, healthy graphs."""
        graphs = ['reify', 'bad-graph']
        rows_by_graph = {'reify': [['target-uuid', 'orchestrator', 'reify']]}
        memory_service, graph_mocks = self._make_memory_service(graphs, rows_by_graph)
        graph_mocks['bad-graph'].ro_query = AsyncMock(side_effect=RuntimeError('no such graph'))

        report = await _mod.run(self._args(uuid='target-uuid'), memory_service)

        assert report['errors'] == [{'graph': 'bad-graph', 'error': 'no such graph'}]
        assert len(report['present_in']) == 1
        assert report['present_in'][0]['graph'] == 'reify'


# ===========================================================================
# Tests: main() -- CLI / live-wiring layer
# ===========================================================================

class _StubConfig:
    """Lightweight stand-in for FusedMemoryConfig -- main() only needs to
    be able to construct one and hand it to MemoryService."""

    def __init__(self, *args, **kwargs):
        pass


class _StubMemoryService:
    """Lightweight stand-in for MemoryService -- wires a .graphiti mock
    sufficient for run() to complete end-to-end without a live FalkorDB."""

    def __init__(self, config):
        self.config = config
        graphiti = MagicMock()
        graphiti.list_graphs = AsyncMock(return_value=['reify'])
        graphiti._graph_for = MagicMock(return_value=_make_graph_mock([]))
        self.graphiti = graphiti

    async def initialize(self):
        return None

    async def close(self):
        return None


class TestMainCliWiring:
    """Tests for main()/_run_live(): argparse default resolution, --config
    -> CONFIG_PATH env wiring, and the report being JSON-serialized to
    stdout. Every other test in this module hits the pure/orchestration
    core directly via mocks; main() itself -- the thing an operator
    actually runs -- was previously untested."""

    def _patch_live_deps(self, monkeypatch):
        monkeypatch.setattr('fused_memory.config.schema.FusedMemoryConfig', _StubConfig)
        monkeypatch.setattr('fused_memory.services.memory_service.MemoryService', _StubMemoryService)

    def test_default_uuid_falls_back_to_target_node_uuid_and_prints_json(self, monkeypatch, capsys):
        """No --uuid on argv -> argparse's own default (TARGET_NODE_UUID)
        flows through end-to-end, and stdout is valid JSON matching the
        report shape."""
        self._patch_live_deps(monkeypatch)
        monkeypatch.setattr(sys, 'argv', ['investigate_cross_graph_duplication.py'])

        rc = _mod.main()

        assert rc == 0
        report = json.loads(capsys.readouterr().out)
        assert report['target_uuid'] == _mod.TARGET_NODE_UUID
        assert report['all_graphs'] == ['reify']
        assert 'verdict' in report
        assert 'scope' in report

    def test_uuid_flag_overrides_default(self, monkeypatch, capsys):
        """--uuid on argv overrides the TARGET_NODE_UUID default end-to-end."""
        self._patch_live_deps(monkeypatch)
        monkeypatch.setattr(
            sys, 'argv',
            ['investigate_cross_graph_duplication.py', '--uuid', 'custom-uuid'],
        )

        _mod.main()

        report = json.loads(capsys.readouterr().out)
        assert report['target_uuid'] == 'custom-uuid'

    def test_config_flag_sets_config_path_env_var(self, monkeypatch, capsys):
        """--config <path> sets os.environ['CONFIG_PATH'] before the live
        config/MemoryService are constructed."""
        self._patch_live_deps(monkeypatch)
        monkeypatch.delenv('CONFIG_PATH', raising=False)
        monkeypatch.setattr(
            sys, 'argv',
            ['investigate_cross_graph_duplication.py', '--config', '/tmp/some-config.yaml'],
        )

        try:
            _mod.main()
            assert os.environ['CONFIG_PATH'] == '/tmp/some-config.yaml'
        finally:
            os.environ.pop('CONFIG_PATH', None)
