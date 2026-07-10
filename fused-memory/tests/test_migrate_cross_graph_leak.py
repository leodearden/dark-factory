"""Tests for scripts/migrate_cross_graph_leak.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution -- mirrors the pattern in
test_purge_knowlive_namespace.py / test_cross_graph_move.py.

Mock-only per project convention (MagicMock/AsyncMock graphs and memory
service; no live-FalkorDB fixture). This suite asserts census/classification/
manifest correctness and dry-run/apply routing at the mock level (B6). LIVE
byte-fidelity (zero foreign nodes remaining, against a REAL FalkorDB) is
eta's live throwaway-graph rehearsal -- see the script's module docstring.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import extract_cypher, extract_params

from fused_memory.maintenance.cross_graph_move import (
    CreateResult,
    MergeResult,
    MoveResult,
    SubgraphEdgeResult,
)

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'migrate_cross_graph_leak.py'


def _load_module() -> types.ModuleType:
    """Load migrate_cross_graph_leak.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators work correctly.
    """
    mod_name = 'migrate_cross_graph_leak'
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
# Shared test helpers
# ===========================================================================

def _result(rows: list[list] | None = None) -> MagicMock:
    """Build a MagicMock FalkorDB query result with the given result_set rows."""
    result = MagicMock()
    result.result_set = rows if rows is not None else []
    return result


def _foreign_row(
    uuid: str, group_id: str, name: str = 'N', labels: list[str] | None = None,
) -> list:
    """Build a raw census result_set row: [uuid, name, group_id, labels(n)].

    Column order matches the census Cypher's RETURN clause
    (n.uuid, n.name, n.group_id, labels(n)).
    """
    return [uuid, name, group_id, labels or ['Entity']]


def _make_graph_mock(
    ro_pages: list[list[list]] | None = None,
    q_result: list[list] | None = None,
    *,
    ro_side_effect=None,
) -> MagicMock:
    """Build a MagicMock graph with AsyncMock .ro_query / .query.

    ro_pages: a list of PAGES (each page a list of raw rows). Successive
    ro_query() calls consume pages in order -- for census SKIP/LIMIT paging
    tests (a single-page list, e.g. ro_pages=[rows], is fine for helpers
    that only issue one ro_query call). Defaults to a single empty page.

    ro_side_effect: a fully custom side_effect (list or callable) for
    ro_query, overriding ro_pages -- for tests where a single graph fields
    several DIFFERENT ro_query shapes in sequence (e.g. a presence probe
    followed by edge/episode count reads).

    q_result: raw rows for a single graph.query() call. This script's
    dry-run path never mutates, but --apply routes mutation through the
    epsilon primitives (not directly through graph.query), so most tests
    never need this -- it's here for completeness/safety assertions
    (e.g. asserting graph.query is NEVER called from dry-run).
    """
    graph = MagicMock()
    if ro_side_effect is not None:
        graph.ro_query = AsyncMock(side_effect=ro_side_effect)
    else:
        pages = ro_pages if ro_pages is not None else [[]]
        graph.ro_query = AsyncMock(side_effect=[_result(page) for page in pages])
    graph.query = AsyncMock(return_value=_result(q_result))
    return graph


def _make_memory_service(graphs: dict[str, MagicMock] | None = None) -> AsyncMock:
    """AsyncMock memory_service with .graphiti wired for census + classify IO.

    *graphs* maps graph name -> MagicMock graph (as returned by
    _make_graph_mock); memory_service.graphiti._graph_for(name) looks it up
    by name. memory_service.graphiti.list_graphs() defaults to sorted(graphs)
    (callers needing a different/ordered graph list can override it after
    construction).
    """
    graphs = graphs or {}
    memory_service = AsyncMock()
    graphiti = MagicMock()
    graphiti._graph_for = MagicMock(side_effect=lambda name: graphs[name])
    graphiti.list_graphs = AsyncMock(return_value=sorted(graphs))
    graphiti._require_falkor_client = MagicMock(return_value=MagicMock())
    memory_service.graphiti = graphiti
    return memory_service


def _args(
    apply: bool = False, manifest=None, page_size: int = 1000, **overrides,
) -> types.SimpleNamespace:
    """SimpleNamespace CLI-args stand-in, mirroring the argparse.Namespace shape."""
    base = {'apply': apply, 'manifest': manifest, 'page_size': page_size}
    base.update(overrides)
    return types.SimpleNamespace(**base)


# ===========================================================================
# Tests: resolve_target_graph (step-1/2)
# ===========================================================================

class TestResolveTargetGraph:
    """Tests for pure resolve_target_graph(group_id, populated_graphs, alias_map)."""

    def test_group_id_naming_a_populated_graph_returns_it(self):
        """A group_id that names an actual populated graph is its own home
        (displaced-only node -- it just needs to move back to a graph that
        already exists)."""
        populated = {'reify', 'dark_factory', 'know_live'}

        target = _mod.resolve_target_graph('dark_factory', populated, _mod.ALIAS_MAP)

        assert target == 'dark_factory'

    def test_orphan_group_id_mapped_by_alias_map_returns_canonical_target(self):
        """An orphan group_id absent from populated_graphs but present in
        ALIAS_MAP resolves to its mapped canonical target."""
        populated = {'reify', 'dark_factory', 'know_live'}

        assert _mod.resolve_target_graph('know-live', populated, _mod.ALIAS_MAP) == 'know_live'
        assert _mod.resolve_target_graph('dark-factory', populated, _mod.ALIAS_MAP) == 'dark_factory'

    def test_unmapped_orphan_returns_none(self):
        """An orphan group_id that names neither a populated graph nor an
        ALIAS_MAP entry is UNRESOLVED (None) -- not silently dropped, not
        silently routed anywhere."""
        populated = {'reify', 'dark_factory', 'know_live'}

        target = _mod.resolve_target_graph(
            'my_solar_challenge_typo', populated, _mod.ALIAS_MAP,
        )

        assert target is None

    def test_unmapped_hyphen_spelling_is_never_auto_normalized(self):
        """A hyphen spelling absent from BOTH populated_graphs and the alias
        map returns None -- it is never silently canonicalized (e.g. via a
        blind hyphen->underscore rewrite) to a new graph name. Guards against
        a future regression that "helpfully" adds generic canonicalization
        in place of the explicit, human-reviewable ALIAS_MAP."""
        populated = {'reify', 'dark_factory', 'know_live'}

        target = _mod.resolve_target_graph('some-other-typo', populated, _mod.ALIAS_MAP)

        assert target is None


# ===========================================================================
# Tests: disposition_for (step-3/4)
# ===========================================================================

class TestDispositionFor:
    """Tests for pure disposition_for(target_graph, present_in_target) -> str."""

    def test_none_target_is_unresolved_regardless_of_presence(self):
        """target_graph is None -> UNRESOLVED, whether or not present_in_target
        is (nonsensically) passed as True."""
        assert _mod.disposition_for(None, False) == _mod.UNRESOLVED
        assert _mod.disposition_for(None, True) == _mod.UNRESOLVED

    def test_target_set_and_present_in_target_is_merge(self):
        """A resolved target that already holds this uuid is a duplicate ->
        MERGE (S6), not a move."""
        assert _mod.disposition_for('know_live', True) == _mod.MERGE

    def test_target_set_and_absent_from_target_is_move(self):
        """A resolved target that does not yet hold this uuid is a
        displaced-only node -> MOVE (S5)."""
        assert _mod.disposition_for('dark_factory', False) == _mod.MOVE

    def test_returned_values_match_module_constants(self):
        """Returned strings are exactly the module's MOVE/MERGE/UNRESOLVED
        constants, not ad-hoc equal-looking literals."""
        assert _mod.disposition_for(None, False) is _mod.UNRESOLVED
        assert _mod.disposition_for('g', True) is _mod.MERGE
        assert _mod.disposition_for('g', False) is _mod.MOVE


def _classified_node(
    uuid: str,
    *,
    name: str = 'N',
    source_graph: str = 'reify',
    target_graph: str | None = 'dark_factory',
    disposition: str = _mod.MOVE,
    edge_count: int = 0,
    episode_count: int = 0,
) -> dict:
    """Build a classified-node record dict, shaped as build_manifest expects."""
    return {
        'uuid': uuid,
        'name': name,
        'source_graph': source_graph,
        'target_graph': target_graph,
        'disposition': disposition,
        'edge_count': edge_count,
        'episode_count': episode_count,
    }


# ===========================================================================
# Tests: build_manifest (step-5/6)
# ===========================================================================

class TestBuildManifest:
    """build_manifest(classified_nodes, census_counts, *, dry_run, alias_map=ALIAS_MAP)."""

    def test_manifest_shape_and_tallies(self):
        """Manifest carries dry_run, alias_map, nodes verbatim, census counts,
        summary tallies, and unresolved_uuids."""
        nodes = [
            _classified_node('u-move', disposition=_mod.MOVE),
            _classified_node('u-merge', disposition=_mod.MERGE),
            _classified_node('u-unresolved', target_graph=None, disposition=_mod.UNRESOLVED),
        ]
        census = {'reify': 3, 'dark_factory': 10, 'know_live': 5}

        manifest = _mod.build_manifest(nodes, census, dry_run=True)

        assert manifest['dry_run'] is True
        assert manifest['alias_map'] == _mod.ALIAS_MAP
        assert manifest['nodes'] == nodes
        assert manifest['census'] == census
        assert manifest['summary'] == {'MOVE': 1, 'MERGE': 1, 'UNRESOLVED': 1, 'total': 3}
        assert manifest['unresolved_uuids'] == ['u-unresolved']

    def test_dry_run_false_is_preserved(self):
        """dry_run is passed through verbatim, not hardcoded."""
        manifest = _mod.build_manifest([], {}, dry_run=False)
        assert manifest['dry_run'] is False

    def test_alias_map_override_is_echoed(self):
        """A caller-supplied alias_map (not the module default) is echoed
        verbatim -- the manifest documents what mapping was ACTUALLY used."""
        custom_map = {'foo': 'bar'}
        manifest = _mod.build_manifest([], {}, dry_run=True, alias_map=custom_map)
        assert manifest['alias_map'] == custom_map

    def test_empty_inputs_produce_zeroed_summary(self):
        """Empty classified_nodes/census_counts -> zeroed summary, empty
        nodes/unresolved lists."""
        manifest = _mod.build_manifest([], {}, dry_run=True)

        assert manifest['nodes'] == []
        assert manifest['census'] == {}
        assert manifest['summary'] == {'MOVE': 0, 'MERGE': 0, 'UNRESOLVED': 0, 'total': 0}
        assert manifest['unresolved_uuids'] == []

    def test_unresolved_uuids_only_includes_unresolved_dispositions(self):
        """unresolved_uuids lists ONLY UNRESOLVED nodes' uuids, in order --
        MOVE/MERGE nodes are excluded even though they're still in 'nodes'."""
        nodes = [
            _classified_node('u1', disposition=_mod.MOVE),
            _classified_node('u2', target_graph=None, disposition=_mod.UNRESOLVED),
            _classified_node('u3', disposition=_mod.MERGE),
            _classified_node('u4', target_graph=None, disposition=_mod.UNRESOLVED),
        ]

        manifest = _mod.build_manifest(nodes, {}, dry_run=True)

        assert manifest['unresolved_uuids'] == ['u2', 'u4']

    def test_no_io_referentially_stable(self):
        """build_manifest performs no I/O and holds no hidden mutable state:
        calling it twice with identical inputs yields equal dicts."""
        nodes = [_classified_node('u1', disposition=_mod.MOVE)]
        census = {'reify': 1}

        r1 = _mod.build_manifest(nodes, census, dry_run=True)
        r2 = _mod.build_manifest(nodes, census, dry_run=True)

        assert r1 == r2

    def test_episodic_skip_tally_is_not_unresolved(self):
        """EPISODIC_SKIP is a distinct disposition from UNRESOLVED -- it must
        be tallied in summary (dynamically, like REKEY) but must NOT leak
        into unresolved_uuids, which collects ONLY UNRESOLVED nodes."""
        node = _classified_node(
            'u-ep', target_graph='dark_factory', disposition=_mod.EPISODIC_SKIP,
        )

        manifest = _mod.build_manifest([node], {'reify': 1}, dry_run=True)

        assert manifest['summary']['EPISODIC_SKIP'] == 1
        assert manifest['summary']['total'] == 1
        assert manifest['unresolved_uuids'] == []


# ===========================================================================
# Tests: census_foreign_nodes (step-7/8)
# ===========================================================================

class TestCensusForeignNodes:
    """Tests for async census_foreign_nodes(graphiti, graph_key, *, page_size)."""

    @pytest.mark.asyncio
    async def test_uses_graph_for_and_ro_query_not_query(self):
        """Resolves the graph via _graph_for(graph_key); reads via ro_query
        only -- .query is NEVER called (read-only census)."""
        graph = _make_graph_mock(ro_pages=[[]])
        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(return_value=graph)

        await _mod.census_foreign_nodes(graphiti, 'dark_factory', page_size=1000)

        graphiti._graph_for.assert_called_once_with('dark_factory')
        graph.ro_query.assert_awaited()
        graph.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_cypher_filters_foreign_nodes_via_skip_limit_not_collect(self):
        """Cypher shape: WHERE n.group_id <> $graph_key, RETURN
        uuid/name/group_id/labels(n), SKIP/LIMIT params -- never collect()."""
        graph = _make_graph_mock(ro_pages=[[]])
        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(return_value=graph)

        await _mod.census_foreign_nodes(graphiti, 'dark_factory', page_size=500)

        cypher = extract_cypher(graph.ro_query.call_args)
        params = extract_params(graph.ro_query.call_args)
        assert 'n.group_id <>' in cypher
        assert 'SKIP' in cypher
        assert 'LIMIT' in cypher
        assert 'collect(' not in cypher
        assert params.get('graph_key') == 'dark_factory'
        assert params.get('skip') == 0
        assert params.get('limit') == 500

    @pytest.mark.asyncio
    async def test_full_page_then_short_page_advances_skip_and_concatenates(self):
        """A full first page (len==page_size) triggers a SECOND ro_query with
        skip advanced by page_size; a short second page ends the loop, and
        the result is the concatenation of both pages."""
        page1 = [_foreign_row(f'u{i}', 'dark_factory') for i in range(3)]
        page2 = [_foreign_row('u-last', 'know_live')]
        graph = _make_graph_mock(ro_pages=[page1, page2])
        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(return_value=graph)

        result = await _mod.census_foreign_nodes(graphiti, 'reify', page_size=3)

        assert graph.ro_query.await_count == 2
        second_params = extract_params(graph.ro_query.call_args_list[1])
        assert second_params.get('skip') == 3
        assert [r['uuid'] for r in result] == ['u0', 'u1', 'u2', 'u-last']

    @pytest.mark.asyncio
    async def test_terminates_on_short_page_no_third_query(self):
        """After a short (non-full) page, no further ro_query is issued."""
        page1 = [_foreign_row('u0', 'dark_factory')]
        graph = _make_graph_mock(ro_pages=[page1])
        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(return_value=graph)

        result = await _mod.census_foreign_nodes(graphiti, 'reify', page_size=1000)

        assert graph.ro_query.await_count == 1
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty_list(self):
        """A graph with zero foreign nodes returns an empty list, not an error."""
        graph = _make_graph_mock(ro_pages=[[]])
        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(return_value=graph)

        result = await _mod.census_foreign_nodes(graphiti, 'reify', page_size=1000)

        assert result == []

    @pytest.mark.asyncio
    async def test_normalizes_rows_to_dicts_with_source_graph(self):
        """Each row normalizes to {uuid,name,group_id,labels,source_graph}."""
        row = _foreign_row('u0', 'dark_factory', name='Alice', labels=['Entity'])
        graph = _make_graph_mock(ro_pages=[[row]])
        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(return_value=graph)

        result = await _mod.census_foreign_nodes(graphiti, 'reify', page_size=1000)

        assert result == [{
            'uuid': 'u0', 'name': 'Alice', 'group_id': 'dark_factory',
            'labels': ['Entity'], 'source_graph': 'reify',
        }]

    @pytest.mark.asyncio
    async def test_full_final_page_at_cap_warns_no_silent_caps(self, caplog, monkeypatch):
        """When the page cap (MAX_CENSUS_PAGES) is hit and the last page
        fetched is STILL full, a no-silent-caps WARNING fires -- the census
        may be incomplete. Monkeypatches the cap down to 1 so a single full
        page deterministically hits it without mocking a huge page count."""
        monkeypatch.setattr(_mod, 'MAX_CENSUS_PAGES', 1)
        page1 = [_foreign_row('u0', 'dark_factory')]
        graph = _make_graph_mock(ro_pages=[page1])
        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(return_value=graph)

        with caplog.at_level('WARNING'):
            result = await _mod.census_foreign_nodes(graphiti, 'reify', page_size=1)

        assert graph.ro_query.await_count == 1
        assert len(result) == 1
        assert any(
            'incomplete' in rec.message.lower() or 'cap' in rec.message.lower()
            for rec in caplog.records
        ), f'Expected a no-silent-caps WARNING, got: {[r.message for r in caplog.records]}'

    @pytest.mark.asyncio
    async def test_no_warning_when_short_page_ends_enumeration(self, caplog):
        """A normal short-page termination does not log a WARNING."""
        graph = _make_graph_mock(ro_pages=[[_foreign_row('u0', 'dark_factory')]])
        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(return_value=graph)

        with caplog.at_level('WARNING'):
            await _mod.census_foreign_nodes(graphiti, 'reify', page_size=1000)

        assert caplog.records == []


# ===========================================================================
# Tests: node_present_in_graph / count_node_edges_episodes (step-9/10)
# ===========================================================================

class TestNodePresentInGraph:
    """Tests for async node_present_in_graph(graph, uuid) -> bool."""

    @pytest.mark.asyncio
    async def test_true_when_result_set_has_a_row(self):
        """A non-empty result_set (the uuid was found) -> True."""
        graph = _make_graph_mock(ro_pages=[[['u0']]])

        present = await _mod.node_present_in_graph(graph, 'u0')

        assert present is True

    @pytest.mark.asyncio
    async def test_false_when_result_set_empty(self):
        """An empty result_set (uuid not found) -> False."""
        graph = _make_graph_mock(ro_pages=[[]])

        present = await _mod.node_present_in_graph(graph, 'missing-uuid')

        assert present is False

    @pytest.mark.asyncio
    async def test_uses_ro_query_not_query(self):
        """Presence probe is read-only: ro_query is used, .query is NEVER called."""
        graph = _make_graph_mock(ro_pages=[[]])

        await _mod.node_present_in_graph(graph, 'u0')

        graph.ro_query.assert_awaited_once()
        graph.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_probes_by_uuid_param(self):
        """The probe's uuid param matches the uuid argument."""
        graph = _make_graph_mock(ro_pages=[[]])

        await _mod.node_present_in_graph(graph, 'target-uuid')

        params = extract_params(graph.ro_query.call_args)
        assert params.get('uuid') == 'target-uuid'


class TestCountNodeEdgesEpisodes:
    """Tests for async count_node_edges_episodes(graph, uuid) -> dict."""

    @pytest.mark.asyncio
    async def test_returns_edges_and_episodes_counts(self):
        """Returns {'edges': int, 'episodes': int} from two count() reads."""
        graph = _make_graph_mock(ro_side_effect=[
            _result([[4]]),  # RELATES_TO count
            _result([[2]]),  # MENTIONS count
        ])

        counts = await _mod.count_node_edges_episodes(graph, 'u0')

        assert counts == {'edges': 4, 'episodes': 2}

    @pytest.mark.asyncio
    async def test_zero_when_no_edges_or_episodes(self):
        """Empty result_sets (no rows -- e.g. count() over zero matches
        never returning a row) normalize to zero, not an error."""
        graph = _make_graph_mock(ro_side_effect=[
            _result([]),
            _result([]),
        ])

        counts = await _mod.count_node_edges_episodes(graph, 'u0')

        assert counts == {'edges': 0, 'episodes': 0}

    @pytest.mark.asyncio
    async def test_read_only_no_query_call(self):
        """Both count reads are read-only: .query is NEVER called."""
        graph = _make_graph_mock(ro_side_effect=[_result([[0]]), _result([[0]])])

        await _mod.count_node_edges_episodes(graph, 'u0')

        graph.query.assert_not_called()
        assert graph.ro_query.await_count == 2

    @pytest.mark.asyncio
    async def test_probes_relates_to_and_mentions_by_uuid(self):
        """First read counts RELATES_TO edges, second counts Episodic
        MENTIONS -- both scoped to the given uuid."""
        graph = _make_graph_mock(ro_side_effect=[_result([[1]]), _result([[1]])])

        await _mod.count_node_edges_episodes(graph, 'target-uuid')

        first_cypher = extract_cypher(graph.ro_query.call_args_list[0])
        first_params = extract_params(graph.ro_query.call_args_list[0])
        second_cypher = extract_cypher(graph.ro_query.call_args_list[1])
        second_params = extract_params(graph.ro_query.call_args_list[1])
        assert 'RELATES_TO' in first_cypher
        assert first_params.get('uuid') == 'target-uuid'
        assert 'MENTIONS' in second_cypher
        assert second_params.get('uuid') == 'target-uuid'


# ===========================================================================
# Tests: classify_node (step-11/12)
# ===========================================================================

def _foreign_node(
    uuid: str, group_id: str, *, name: str = 'N', source_graph: str = 'reify',
    labels: list[str] | None = None,
) -> dict:
    """Build a normalized census-row dict, as returned by census_foreign_nodes."""
    return {
        'uuid': uuid, 'name': name, 'group_id': group_id,
        'labels': labels if labels is not None else ['Entity'], 'source_graph': source_graph,
    }


class TestClassifyNode:
    """Tests for async classify_node(graphiti, node, populated_graphs, *, alias_map)."""

    @pytest.mark.asyncio
    async def test_resolved_target_absent_in_target_is_move(self):
        """A displaced-only node (group_id names a populated graph, absent
        from there) classifies MOVE. edge_count/episode_count are read from
        the SOURCE graph, not the target."""
        target_graph = _make_graph_mock(ro_pages=[[]])  # presence probe: absent
        source_graph = _make_graph_mock(ro_side_effect=[_result([[3]]), _result([[1]])])
        graphs = {'dark_factory': target_graph, 'reify': source_graph}
        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(side_effect=lambda name: graphs[name])
        node = _foreign_node('u0', 'dark_factory', name='Alice', source_graph='reify')
        populated = {'reify', 'dark_factory', 'know_live'}

        record = await _mod.classify_node(graphiti, node, populated, alias_map=_mod.ALIAS_MAP)

        assert record == {
            'uuid': 'u0', 'name': 'Alice', 'source_graph': 'reify',
            'target_graph': 'dark_factory', 'disposition': _mod.MOVE,
            'edge_count': 3, 'episode_count': 1,
        }

    @pytest.mark.asyncio
    async def test_resolved_target_present_in_target_is_merge(self):
        """A duplicate node (present in the resolved target) classifies
        MERGE, and the presence probe is issued against _graph_for(target)."""
        target_graph = _make_graph_mock(ro_pages=[[['u0']]])  # presence probe: present
        source_graph = _make_graph_mock(ro_side_effect=[_result([[0]]), _result([[0]])])
        graphs = {'dark_factory': target_graph, 'reify': source_graph}
        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(side_effect=lambda name: graphs[name])
        node = _foreign_node('u0', 'dark_factory', source_graph='reify')
        populated = {'reify', 'dark_factory'}

        record = await _mod.classify_node(graphiti, node, populated, alias_map=_mod.ALIAS_MAP)

        assert record['disposition'] == _mod.MERGE
        assert record['target_graph'] == 'dark_factory'
        graphiti._graph_for.assert_any_call('dark_factory')
        target_graph.ro_query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_alias_mapped_orphan_resolves_via_alias_map(self):
        """An orphan group_id (e.g. 'know-live') with no populated graph of
        its own resolves via ALIAS_MAP to its canonical target."""
        target_graph = _make_graph_mock(ro_pages=[[]])
        source_graph = _make_graph_mock(ro_side_effect=[_result([[0]]), _result([[0]])])
        graphs = {'know_live': target_graph, 'reify': source_graph}
        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(side_effect=lambda name: graphs[name])
        node = _foreign_node('u0', 'know-live', name='Bob', source_graph='reify')
        populated = {'reify', 'know_live'}

        record = await _mod.classify_node(graphiti, node, populated, alias_map=_mod.ALIAS_MAP)

        assert record['target_graph'] == 'know_live'
        assert record['disposition'] == _mod.MOVE

    @pytest.mark.asyncio
    async def test_unmapped_orphan_is_unresolved_with_no_presence_probe(self):
        """An orphan with no populated graph and no ALIAS_MAP entry is
        UNRESOLVED, target_graph=None, and NO presence probe is issued on
        any target -- _graph_for is called ONLY for the source graph (to
        read edge/episode counts)."""
        source_graph = _make_graph_mock(ro_side_effect=[_result([[0]]), _result([[0]])])
        graphs = {'reify': source_graph}
        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(side_effect=lambda name: graphs[name])
        node = _foreign_node('u0', 'totally-unknown', name='Carl', source_graph='reify')
        populated = {'reify', 'dark_factory'}

        record = await _mod.classify_node(graphiti, node, populated, alias_map=_mod.ALIAS_MAP)

        assert record['target_graph'] is None
        assert record['disposition'] == _mod.UNRESOLVED
        assert record['edge_count'] == 0
        assert record['episode_count'] == 0
        graphiti._graph_for.assert_called_once_with('reify')

    @pytest.mark.asyncio
    async def test_target_equals_source_is_rekey_not_merge(self):
        """A variant-spelling leak whose alias resolves to its OWN source
        graph (e.g. a node physically living in the 'know_live' graph,
        tagged group_id='know-live') classifies REKEY, not MERGE.

        Probing the resolved target here would find the node ITSELF
        (present=True), so a naive impl reaches MERGE and apply would then
        call merge_foreign_duplicate(wrong=='know_live', home=='know_live')
        -- DETACH DELETEing the node's ONLY copy. The fixed impl recognizes
        target_graph == source_graph and skips the presence probe entirely:
        the single graph's ro_query is awaited exactly twice (only the two
        count reads), never a third time for a presence probe. A callable
        side_effect keyed on Cypher shape makes this assertion meaningful
        under BOTH the current (3 awaits, MERGE) and fixed (2 awaits, REKEY)
        implementations.
        """
        def _side_effect(cypher, params=None):
            if 'RELATES_TO' in cypher:
                return _result([[3]])
            if 'MENTIONS' in cypher:
                return _result([[1]])
            # Presence-probe shape -- would (wrongly) report "present" if a
            # naive impl issues it against the node's own home graph.
            return _result([['u0']])

        graph = _make_graph_mock(ro_side_effect=_side_effect)
        graphs = {'know_live': graph}
        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(side_effect=lambda name: graphs[name])
        node = _foreign_node('u0', 'know-live', name='Alice', source_graph='know_live')
        populated = {'reify', 'dark_factory', 'know_live'}

        record = await _mod.classify_node(graphiti, node, populated, alias_map=_mod.ALIAS_MAP)

        assert record['disposition'] == _mod.REKEY
        assert record['target_graph'] == 'know_live'
        assert record['source_graph'] == 'know_live'
        assert record['edge_count'] == 3
        assert record['episode_count'] == 1
        assert graph.ro_query.await_count == 2

    @pytest.mark.asyncio
    async def test_episodic_node_is_episodic_skip_no_presence_probe(self):
        """A foreign Episodic node (labels=['Episodic']) is never actioned by
        the :Entity-scoped epsilon primitives (move_entity_across_graphs /
        merge_foreign_duplicate) or rekey_node_in_place -- classify_node
        routes it to EPISODIC_SKIP instead of MOVE/MERGE/REKEY, with NO
        presence probe issued against the target graph (only 'reify', the
        source graph, is wired into this test's graphs map -- a probe
        against 'dark_factory' would KeyError). target_graph still carries
        the informational resolve_target_graph result (where the node WOULD
        belong, for eta's live relocation); edge_count/episode_count are
        still read from the source graph."""
        source_graph = _make_graph_mock(ro_side_effect=[_result([[3]]), _result([[1]])])
        graphs = {'reify': source_graph}
        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(side_effect=lambda name: graphs[name])
        node = _foreign_node(
            'u0', 'dark_factory', name='Ep1', source_graph='reify', labels=['Episodic'],
        )
        populated = {'reify', 'dark_factory'}

        record = await _mod.classify_node(graphiti, node, populated, alias_map=_mod.ALIAS_MAP)

        assert record['disposition'] == _mod.EPISODIC_SKIP
        assert record['target_graph'] == 'dark_factory'
        assert record['edge_count'] == 3
        assert record['episode_count'] == 1
        graphiti._graph_for.assert_called_once_with('reify')

    @pytest.mark.asyncio
    @pytest.mark.parametrize('labels', [['Community'], []])
    async def test_non_entity_labels_are_episodic_skip(self, labels):
        """Any foreign node whose labels do not positively confirm :Entity
        (a Community node, or a row with no labels at all) is ALSO routed to
        EPISODIC_SKIP -- only a positively-confirmed :Entity node is ever
        actioned (fail-safe: 'Entity' not in labels, not 'Episodic' in
        labels)."""
        source_graph = _make_graph_mock(ro_side_effect=[_result([[0]]), _result([[0]])])
        graphs = {'reify': source_graph}
        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(side_effect=lambda name: graphs[name])
        node = _foreign_node('u0', 'dark_factory', source_graph='reify', labels=labels)
        populated = {'reify', 'dark_factory'}

        record = await _mod.classify_node(graphiti, node, populated, alias_map=_mod.ALIAS_MAP)

        assert record['disposition'] == _mod.EPISODIC_SKIP
        graphiti._graph_for.assert_called_once_with('reify')


# ===========================================================================
# Tests: rekey_node_in_place (step-25/26 review-fix)
# ===========================================================================

class TestRekeyNodeInPlace:
    """Tests for async rekey_node_in_place(graph, uuid, new_group_id).

    Anti-data-loss invariant: an in-place re-key must NEVER open a
    CREATE-then-DETACH-DELETE window -- it only ever SETs group_id.
    """

    @pytest.mark.asyncio
    async def test_rewrites_group_id_in_place_no_create_no_delete(self):
        """Rewrites group_id via graph.query (a mutation, not ro_query): at
        least one call carries 'SET' + 'group_id' with new_group_id as a
        param, and NO call ever carries 'CREATE' or 'DETACH DELETE'."""
        graph = _make_graph_mock()

        await _mod.rekey_node_in_place(graph, 'u0', 'know_live')

        graph.query.assert_awaited()
        graph.ro_query.assert_not_called()
        cyphers = [extract_cypher(call) for call in graph.query.call_args_list]
        params_list = [extract_params(call) for call in graph.query.call_args_list]
        assert any('SET' in c and 'group_id' in c for c in cyphers)
        assert all('CREATE' not in c for c in cyphers)
        assert all('DETACH DELETE' not in c for c in cyphers)
        assert any(p.get('gid') == 'know_live' for p in params_list)
        assert all(p.get('uuid') == 'u0' for p in params_list)


# ===========================================================================
# Tests: run() dry-run (step-13/14)
# ===========================================================================

class TestRunDryRun:
    """Tests for async run(args, memory_service) -- dry-run branch (args.apply=False)."""

    @pytest.mark.asyncio
    async def test_dry_run_enumerates_censuses_classifies_and_builds_manifest(self, monkeypatch):
        """list_graphs() is enumerated; census_foreign_nodes runs per populated
        graph; each foreign node found is classified; the result is a
        build_manifest-shaped dict with dry_run=True, per-graph census
        counts, and the classified node records.

        'dark_factory' plays two roles here (census subject AND the
        resolved target of reify's foreign node) and populated_graphs is a
        set with unordered iteration, so its mock uses a role-aware
        callable side_effect (dispatching on the Cypher shape) rather than
        a fixed-order list -- this keeps the test correct regardless of
        which graph run() happens to enumerate first.
        """
        move_mock = AsyncMock(side_effect=AssertionError('must not be called during dry-run'))
        merge_mock = AsyncMock(side_effect=AssertionError('must not be called during dry-run'))
        monkeypatch.setattr(_mod, 'move_entity_across_graphs', move_mock)
        monkeypatch.setattr(_mod, 'merge_foreign_duplicate', merge_mock)

        reify_row = _foreign_row('u0', 'dark_factory', name='Alice')
        reify_graph = _make_graph_mock(ro_side_effect=[
            _result([reify_row]),  # census: one foreign node
            _result([[3]]),        # count_node_edges_episodes: RELATES_TO
            _result([[1]]),        # count_node_edges_episodes: MENTIONS
        ])

        def _dark_factory_side_effect(cypher, params=None):
            if 'group_id <>' in cypher:
                return _result([])  # census: no foreign nodes of its own
            return _result([])  # presence probe: absent (-> MOVE)

        dark_factory_graph = _make_graph_mock(ro_side_effect=_dark_factory_side_effect)

        memory_service = _make_memory_service({
            'reify': reify_graph, 'dark_factory': dark_factory_graph,
        })
        args = _args(apply=False, page_size=1000)

        manifest = await _mod.run(args, memory_service)

        memory_service.graphiti.list_graphs.assert_awaited_once()
        assert manifest['dry_run'] is True
        assert manifest['census'] == {'reify': 1, 'dark_factory': 0}
        assert len(manifest['nodes']) == 1
        assert manifest['nodes'][0]['uuid'] == 'u0'
        assert manifest['nodes'][0]['disposition'] == _mod.MOVE
        assert manifest['nodes'][0]['target_graph'] == 'dark_factory'
        assert manifest['exit_code'] == 0

        # Nothing mutated: no .query on either graph, and neither epsilon
        # primitive was invoked.
        reify_graph.query.assert_not_awaited()
        dark_factory_graph.query.assert_not_awaited()
        move_mock.assert_not_awaited()
        merge_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dry_run_with_no_foreign_nodes_returns_empty_manifest(self):
        """No foreign nodes anywhere -> an empty-but-well-shaped manifest."""
        memory_service = _make_memory_service({
            'reify': _make_graph_mock(ro_pages=[[]]),
            'dark_factory': _make_graph_mock(ro_pages=[[]]),
        })
        args = _args(apply=False)

        manifest = await _mod.run(args, memory_service)

        assert manifest['nodes'] == []
        assert manifest['census'] == {'reify': 0, 'dark_factory': 0}
        assert manifest['summary'] == {'MOVE': 0, 'MERGE': 0, 'UNRESOLVED': 0, 'total': 0}
        assert manifest['exit_code'] == 0


# ===========================================================================
# Tests: load_reviewed_manifest + --apply-requires-manifest guard (step-15/16)
# ===========================================================================

class TestLoadReviewedManifest:
    """Tests for load_reviewed_manifest(path) and run()'s apply-requires-manifest guard."""

    def test_round_trips_a_manifest_file(self, tmp_path):
        """load_reviewed_manifest(path) parses back exactly what was written."""
        manifest = _mod.build_manifest(
            [_classified_node('u1', disposition=_mod.MOVE)], {'reify': 1}, dry_run=True,
        )
        manifest_path = tmp_path / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest))

        loaded = _mod.load_reviewed_manifest(manifest_path)

        assert loaded == manifest

    @pytest.mark.asyncio
    async def test_apply_without_manifest_refuses_with_no_mutations(self, monkeypatch):
        """--apply with no --manifest refuses: zero epsilon calls, no census
        recompute (list_graphs is never even awaited), non-zero exit_code."""
        move_mock = AsyncMock()
        merge_mock = AsyncMock()
        monkeypatch.setattr(_mod, 'move_entity_across_graphs', move_mock)
        monkeypatch.setattr(_mod, 'merge_foreign_duplicate', merge_mock)
        memory_service = _make_memory_service({'reify': _make_graph_mock()})
        args = _args(apply=True, manifest=None)

        report = await _mod.run(args, memory_service)

        move_mock.assert_not_awaited()
        merge_mock.assert_not_awaited()
        memory_service.graphiti.list_graphs.assert_not_awaited()
        assert report['exit_code'] != 0


# ===========================================================================
# Tests: run() apply dispatch to epsilon primitives (step-17/18)
# ===========================================================================

class TestRunApplyDispatch:
    """Tests for run() dispatching a reviewed manifest's nodes by disposition."""

    @pytest.mark.asyncio
    async def test_dispatches_move_and_merge_skips_unresolved(self, tmp_path, monkeypatch):
        """MOVE nodes dispatch to move_entity_across_graphs, MERGE nodes to
        merge_foreign_duplicate, and an UNRESOLVED node is NEVER passed to
        either primitive -- the report still records it as a blocked entry."""
        move_node = _classified_node(
            'u-move', source_graph='reify', target_graph='dark_factory', disposition=_mod.MOVE,
        )
        merge_node = _classified_node(
            'u-merge', source_graph='reify', target_graph='know_live', disposition=_mod.MERGE,
        )
        unresolved_node = _classified_node(
            'u-unresolved', source_graph='reify', target_graph=None, disposition=_mod.UNRESOLVED,
        )
        manifest = _mod.build_manifest(
            [move_node, merge_node, unresolved_node], {'reify': 3}, dry_run=True,
        )
        manifest_path = tmp_path / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest))

        move_mock = AsyncMock(return_value=MoveResult(
            uuid='u-move', source_graph='reify', target_graph='dark_factory',
        ))
        merge_mock = AsyncMock(return_value=MergeResult(
            uuid='u-merge', wrong_graph='reify', home_graph='know_live',
        ))
        monkeypatch.setattr(_mod, 'move_entity_across_graphs', move_mock)
        monkeypatch.setattr(_mod, 'merge_foreign_duplicate', merge_mock)

        memory_service = _make_memory_service({})
        args = _args(apply=True, manifest=str(manifest_path))

        report = await _mod.run(args, memory_service)

        move_mock.assert_awaited_once_with(
            memory_service.graphiti, 'u-move', 'reify', 'dark_factory',
            rewrite_group_id='dark_factory',
        )
        merge_mock.assert_awaited_once_with(
            memory_service.graphiti, 'u-merge', wrong_graph='reify', home_graph='know_live',
        )
        assert report['unresolved_uuids'] == ['u-unresolved']
        blocked = [r for r in report['apply_results'] if r['uuid'] == 'u-unresolved']
        assert blocked == [{
            'uuid': 'u-unresolved', 'disposition': _mod.UNRESOLVED,
            'applied': False, 'blocked': True,
        }]

    @pytest.mark.asyncio
    async def test_move_dispatch_passes_rewrite_group_id_matching_target(
        self, tmp_path, monkeypatch,
    ):
        """Regression guard (review-fix cycle 1, Bug 2): a MOVE dispatch
        must pass rewrite_group_id == the node's target_graph, so an
        alias-mapped move re-keys the moved node (and, via epsilon's
        uniform new_group_id, its edges/mentions) to its home-graph name.
        Without this, the recreated node keeps its variant group_id, stays
        foreign in its new home, and post-verify miscounts it as residual
        (a false exit 1). Pinned as its OWN focused test so a future
        refactor cannot silently drop the kwarg without a test failing."""
        move_node = _classified_node(
            'u-move', source_graph='reify', target_graph='know_live', disposition=_mod.MOVE,
        )
        manifest = _mod.build_manifest([move_node], {'reify': 1}, dry_run=True)
        manifest_path = tmp_path / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest))

        move_mock = AsyncMock(return_value=MoveResult(
            uuid='u-move', source_graph='reify', target_graph='know_live',
        ))
        monkeypatch.setattr(_mod, 'move_entity_across_graphs', move_mock)
        monkeypatch.setattr(_mod, 'merge_foreign_duplicate', AsyncMock())

        memory_service = _make_memory_service({})
        args = _args(apply=True, manifest=str(manifest_path))

        await _mod.run(args, memory_service)

        assert move_mock.await_args is not None
        call_kwargs = move_mock.await_args.kwargs
        assert call_kwargs.get('rewrite_group_id') == 'know_live'


# ===========================================================================
# Tests: run() apply skips EPISODIC_SKIP (step-31/32 review-fix, cycle 2)
# ===========================================================================

class TestRunApplyEpisodicSkip:
    """Tests for run() never dispatching an EPISODIC_SKIP node to a
    primitive -- foreign Episodic/Community/unlabeled nodes are not
    :Entity-scoped-primitive-compatible; see classify_node's label gate."""

    def _write_manifest(self, tmp_path, nodes, census) -> Path:
        manifest = _mod.build_manifest(nodes, census, dry_run=True)
        manifest_path = tmp_path / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path

    @pytest.mark.asyncio
    async def test_episodic_skip_node_never_dispatched(self, tmp_path, monkeypatch):
        """An EPISODIC_SKIP node is never passed to move_entity_across_graphs
        or merge_foreign_duplicate, and receives no rekey mutation either --
        it is recorded blocked, not applied."""
        ep_node = _classified_node(
            'u-ep', source_graph='reify', target_graph='dark_factory',
            disposition=_mod.EPISODIC_SKIP,
        )
        manifest_path = self._write_manifest(tmp_path, [ep_node], {'reify': 1})

        move_mock = AsyncMock(side_effect=AssertionError('must not dispatch EPISODIC_SKIP'))
        merge_mock = AsyncMock(side_effect=AssertionError('must not dispatch EPISODIC_SKIP'))
        monkeypatch.setattr(_mod, 'move_entity_across_graphs', move_mock)
        monkeypatch.setattr(_mod, 'merge_foreign_duplicate', merge_mock)

        reify_graph = _make_graph_mock(
            ro_pages=[[_foreign_row('u-ep', 'dark_factory', labels=['Episodic'])]],
        )
        memory_service = _make_memory_service({'reify': reify_graph})
        args = _args(apply=True, manifest=str(manifest_path))

        report = await _mod.run(args, memory_service)

        move_mock.assert_not_awaited()
        merge_mock.assert_not_awaited()
        reify_graph.query.assert_not_awaited()

        ep_results = [r for r in report['apply_results'] if r['uuid'] == 'u-ep']
        assert len(ep_results) == 1
        assert ep_results[0]['applied'] is False
        assert ep_results[0]['blocked'] is True
        assert ep_results[0]['disposition'] == _mod.EPISODIC_SKIP

    @pytest.mark.asyncio
    async def test_episodic_skip_residual_matches_but_still_blocks_exit(
        self, tmp_path, monkeypatch,
    ):
        """EPISODIC_SKIP is counted in expected_residual (mirroring
        UNRESOLVED), so a graph still holding the un-actioned foreign
        Episodic node after apply reconciles (matched=True -- NOT mistaken
        for a silently-failed MOVE) -- but it still forces a non-zero,
        blocking exit_code (has_blocked), since a foreign Episodic node was
        never actually resolved."""
        ep_node = _classified_node(
            'u-ep', source_graph='reify', target_graph='dark_factory',
            disposition=_mod.EPISODIC_SKIP,
        )
        manifest_path = self._write_manifest(tmp_path, [ep_node], {'reify': 1, 'dark_factory': 0})

        # return_value is a real, zero-loss MoveResult/MergeResult (not a
        # bare AsyncMock()) -- run() now reads edges_skipped/mentions_skipped/
        # home_edge_count_after off the result (see _move_result_entry/
        # _merge_result_entry), and an unconfigured MagicMock's truthy
        # attributes and non-identical `+` result would otherwise be
        # misread as a lossy outcome, spuriously flipping 'blocked' to True.
        monkeypatch.setattr(
            _mod, 'move_entity_across_graphs',
            AsyncMock(return_value=MoveResult(uuid='u', source_graph='s', target_graph='t')),
        )
        monkeypatch.setattr(
            _mod, 'merge_foreign_duplicate',
            AsyncMock(return_value=MergeResult(uuid='u', wrong_graph='s', home_graph='t')),
        )

        memory_service = _make_memory_service({
            'reify': _make_graph_mock(
                ro_pages=[[_foreign_row('u-ep', 'dark_factory', labels=['Episodic'])]],
            ),
            'dark_factory': _make_graph_mock(ro_pages=[[]]),
        })
        args = _args(apply=True, manifest=str(manifest_path))

        report = await _mod.run(args, memory_service)

        assert report['post_verify']['matched'] is True
        assert report['exit_code'] != 0


# ===========================================================================
# Tests: run() REKEY apply + defensive source==target guard
# (step-25/26 review-fix, Bug 1 data-loss)
# ===========================================================================

class TestRunApplyRekeyAndGuard:
    """Tests for run() applying REKEY nodes and refusing malformed MOVE/MERGE
    nodes whose source_graph == target_graph (the reviewer's data-loss trap:
    dispatching either primitive with wrong==home DETACH DELETEs the node's
    only copy)."""

    def _write_manifest(self, tmp_path, nodes, census) -> Path:
        manifest = _mod.build_manifest(nodes, census, dry_run=True)
        manifest_path = tmp_path / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path

    @pytest.mark.asyncio
    async def test_apply_rekey_node_in_place_no_epsilon_call(self, tmp_path, monkeypatch):
        """A REKEY node (source_graph == target_graph) never dispatches to
        either epsilon primitive -- both are monkeypatched to explode if
        invoked, as a hard guard against the data-loss bug this fixes.
        Instead, the home graph receives an in-place group_id SET (no
        CREATE/DETACH DELETE), the node is recorded applied (not blocked),
        and a clean post-verify re-census yields exit_code 0."""
        rekey_node = _classified_node(
            'u-rekey', source_graph='know_live', target_graph='know_live',
            disposition=_mod.REKEY,
        )
        manifest_path = self._write_manifest(tmp_path, [rekey_node], {'know_live': 1})

        move_mock = AsyncMock(side_effect=AssertionError('must not be called for REKEY'))
        merge_mock = AsyncMock(side_effect=AssertionError('must not be called for REKEY'))
        monkeypatch.setattr(_mod, 'move_entity_across_graphs', move_mock)
        monkeypatch.setattr(_mod, 'merge_foreign_duplicate', merge_mock)

        home_graph = _make_graph_mock(ro_pages=[[]])  # clean post-verify re-census
        memory_service = _make_memory_service({'know_live': home_graph})
        args = _args(apply=True, manifest=str(manifest_path))

        report = await _mod.run(args, memory_service)

        move_mock.assert_not_awaited()
        merge_mock.assert_not_awaited()

        cyphers = [extract_cypher(call) for call in home_graph.query.call_args_list]
        assert cyphers, 'expected at least one graph.query mutation for the rekey'
        assert any('SET' in c and 'group_id' in c for c in cyphers)
        assert all('CREATE' not in c for c in cyphers)
        assert all('DETACH DELETE' not in c for c in cyphers)

        rekey_result = [r for r in report['apply_results'] if r['uuid'] == 'u-rekey']
        assert rekey_result == [{
            'uuid': 'u-rekey', 'disposition': _mod.REKEY, 'applied': True, 'blocked': False,
        }]
        assert report['exit_code'] == 0

    @pytest.mark.asyncio
    async def test_move_or_merge_with_source_equals_target_is_refused(self, tmp_path, monkeypatch):
        """A malformed/hand-edited manifest carrying a MOVE or MERGE node
        whose source_graph == target_graph (never produced by a correctly-
        behaving classify_node -- that's REKEY's territory) is REFUSED, not
        dispatched: dispatching it would make move_entity_across_graphs /
        merge_foreign_duplicate DETACH DELETE the node's only copy. Both
        nodes are recorded blocked and force a non-zero exit_code even
        though the mock re-census otherwise reconciles."""
        bad_move = _classified_node(
            'u-bad-move', source_graph='know_live', target_graph='know_live',
            disposition=_mod.MOVE,
        )
        bad_merge = _classified_node(
            'u-bad-merge', source_graph='reify', target_graph='reify', disposition=_mod.MERGE,
        )
        manifest_path = self._write_manifest(
            tmp_path, [bad_move, bad_merge], {'know_live': 1, 'reify': 1},
        )

        move_mock = AsyncMock()
        merge_mock = AsyncMock()
        monkeypatch.setattr(_mod, 'move_entity_across_graphs', move_mock)
        monkeypatch.setattr(_mod, 'merge_foreign_duplicate', merge_mock)

        memory_service = _make_memory_service({
            'know_live': _make_graph_mock(ro_pages=[[]]),
            'reify': _make_graph_mock(ro_pages=[[]]),
        })
        args = _args(apply=True, manifest=str(manifest_path))

        report = await _mod.run(args, memory_service)

        move_mock.assert_not_awaited()
        merge_mock.assert_not_awaited()
        blocked_uuids = {r['uuid'] for r in report['apply_results'] if r['blocked']}
        assert blocked_uuids == {'u-bad-move', 'u-bad-merge'}
        assert report['post_verify']['matched'] is True
        assert report['exit_code'] != 0


# ===========================================================================
# Tests: run() apply loop isolates per-node primitive exceptions
# (step-33/34 review-fix, cycle 2)
# ===========================================================================

class TestRunApplyExceptionIsolation:
    """Tests for run()'s apply loop never letting one node's primitive
    exception abort the whole batch. A hand-edited manifest mislabeling an
    Episodic node as MOVE (escaping the classify-time EPISODIC_SKIP gate),
    a ForeignDuplicateSuspectedError, or a transient FalkorDB error must not
    lose the record of every OTHER node's outcome, and post-verify must
    still run."""

    def _write_manifest(self, tmp_path, nodes, census) -> Path:
        manifest = _mod.build_manifest(nodes, census, dry_run=True)
        manifest_path = tmp_path / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path

    @pytest.mark.asyncio
    async def test_failed_dispatch_does_not_abort_remaining_nodes(self, tmp_path, monkeypatch):
        """A FIRST node whose primitive dispatch raises is recorded blocked
        with the error captured -- but the loop continues: the SECOND node
        is still dispatched, post-verify still runs, and run() returns a
        report dict rather than propagating the exception. The failure
        forces a non-zero exit_code."""
        failing_move = _classified_node(
            'u-move', source_graph='reify', target_graph='dark_factory', disposition=_mod.MOVE,
        )
        succeeding_merge = _classified_node(
            'u-merge', source_graph='know_live', target_graph='pump_web_ui',
            disposition=_mod.MERGE,
        )
        manifest_path = self._write_manifest(
            tmp_path, [failing_move, succeeding_merge], {'reify': 1, 'know_live': 1},
        )

        # merge_mock's return_value is a real, zero-loss MergeResult (not a
        # bare AsyncMock()) -- run() reads edges_recreated/home_edge_count_before/
        # home_edge_count_after off the result (see _merge_result_entry), and an
        # unconfigured MagicMock's truthy attributes and non-identical `+` result
        # would otherwise be misread as a lossy outcome, spuriously flipping
        # 'blocked' to True (same pitfall documented on the sibling tests above).
        move_mock = AsyncMock(side_effect=RuntimeError('simulated epsilon failure'))
        merge_mock = AsyncMock(return_value=MergeResult(
            uuid='u-merge', wrong_graph='know_live', home_graph='pump_web_ui',
        ))
        monkeypatch.setattr(_mod, 'move_entity_across_graphs', move_mock)
        monkeypatch.setattr(_mod, 'merge_foreign_duplicate', merge_mock)

        memory_service = _make_memory_service({
            'reify': _make_graph_mock(ro_pages=[[]]),
            'know_live': _make_graph_mock(ro_pages=[[]]),
        })
        args = _args(apply=True, manifest=str(manifest_path))

        report = await _mod.run(args, memory_service)

        move_result = [r for r in report['apply_results'] if r['uuid'] == 'u-move']
        assert len(move_result) == 1
        assert move_result[0]['applied'] is False
        assert move_result[0]['blocked'] is True
        assert 'simulated epsilon failure' in move_result[0]['error']

        merge_mock.assert_awaited_once()
        merge_result = [r for r in report['apply_results'] if r['uuid'] == 'u-merge']
        assert merge_result == [{
            'uuid': 'u-merge', 'disposition': _mod.MERGE, 'applied': True, 'blocked': False,
            'edges_recreated': 0, 'home_edge_count_before': 0, 'home_edge_count_after': 0,
        }]

        assert 'post_verify' in report
        assert report['exit_code'] != 0


# ===========================================================================
# Tests: run() post-verify re-census (step-19/20)
# ===========================================================================

class TestRunPostVerify:
    """Tests for run()'s post-apply re-census verification and exit_code."""

    def _write_manifest(self, tmp_path, nodes, census) -> Path:
        manifest = _mod.build_manifest(nodes, census, dry_run=True)
        manifest_path = tmp_path / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path

    @pytest.mark.asyncio
    async def test_clean_reconciliation_after_apply_is_exit_zero(self, tmp_path, monkeypatch):
        """A clean post-apply re-census (the moved node is gone from both
        graphs, no UNRESOLVED nodes in the manifest) reconciles exactly with
        the expected (all-zero) residual -> exit_code 0 and a post_verify
        block marking matched."""
        move_node = _classified_node(
            'u-move', source_graph='reify', target_graph='dark_factory', disposition=_mod.MOVE,
        )
        manifest_path = self._write_manifest(tmp_path, [move_node], {'reify': 1, 'dark_factory': 0})

        # return_value is a real, zero-loss MoveResult/MergeResult (not a
        # bare AsyncMock()) -- run() now reads edges_skipped/mentions_skipped/
        # home_edge_count_after off the result (see _move_result_entry/
        # _merge_result_entry), and an unconfigured MagicMock's truthy
        # attributes and non-identical `+` result would otherwise be
        # misread as a lossy outcome, spuriously flipping 'blocked' to True.
        monkeypatch.setattr(
            _mod, 'move_entity_across_graphs',
            AsyncMock(return_value=MoveResult(uuid='u', source_graph='s', target_graph='t')),
        )
        monkeypatch.setattr(
            _mod, 'merge_foreign_duplicate',
            AsyncMock(return_value=MergeResult(uuid='u', wrong_graph='s', home_graph='t')),
        )

        memory_service = _make_memory_service({
            'reify': _make_graph_mock(ro_pages=[[]]),
            'dark_factory': _make_graph_mock(ro_pages=[[]]),
        })
        args = _args(apply=True, manifest=str(manifest_path))

        report = await _mod.run(args, memory_service)

        memory_service.graphiti.list_graphs.assert_awaited_once()
        assert report['post_verify'] == {
            'matched': True,
            'expected': {'reify': 0, 'dark_factory': 0},
            'actual': {'reify': 0, 'dark_factory': 0},
        }
        assert report['exit_code'] == 0

    @pytest.mark.asyncio
    async def test_residual_foreign_node_after_apply_is_nonzero_exit(self, tmp_path, monkeypatch):
        """If the post-apply re-census still finds a foreign node in a graph
        that should now be clean (e.g. a MOVE that silently failed to remove
        the source copy), the residual count mismatches the expectation ->
        non-zero exit_code."""
        move_node = _classified_node(
            'u-move', source_graph='reify', target_graph='dark_factory', disposition=_mod.MOVE,
        )
        manifest_path = self._write_manifest(tmp_path, [move_node], {'reify': 1, 'dark_factory': 0})

        # return_value is a real, zero-loss MoveResult/MergeResult (not a
        # bare AsyncMock()) -- run() now reads edges_skipped/mentions_skipped/
        # home_edge_count_after off the result (see _move_result_entry/
        # _merge_result_entry), and an unconfigured MagicMock's truthy
        # attributes and non-identical `+` result would otherwise be
        # misread as a lossy outcome, spuriously flipping 'blocked' to True.
        monkeypatch.setattr(
            _mod, 'move_entity_across_graphs',
            AsyncMock(return_value=MoveResult(uuid='u', source_graph='s', target_graph='t')),
        )
        monkeypatch.setattr(
            _mod, 'merge_foreign_duplicate',
            AsyncMock(return_value=MergeResult(uuid='u', wrong_graph='s', home_graph='t')),
        )

        memory_service = _make_memory_service({
            'reify': _make_graph_mock(ro_pages=[[_foreign_row('u-move', 'dark_factory')]]),
            'dark_factory': _make_graph_mock(ro_pages=[[]]),
        })
        args = _args(apply=True, manifest=str(manifest_path))

        report = await _mod.run(args, memory_service)

        assert report['post_verify']['matched'] is False
        assert report['exit_code'] != 0

    @pytest.mark.asyncio
    async def test_any_unresolved_node_forces_nonzero_exit_even_if_counts_reconcile(
        self, tmp_path, monkeypatch,
    ):
        """An UNRESOLVED node blocks --apply's success regardless of whether
        the post-apply re-census otherwise reconciles: the residual foreign
        node it necessarily leaves behind IS the expected residual (matched
        can be True), but the manifest containing any UNRESOLVED node still
        forces a non-zero, blocking exit_code."""
        unresolved_node = _classified_node(
            'u-unresolved', source_graph='reify', target_graph=None, disposition=_mod.UNRESOLVED,
        )
        manifest_path = self._write_manifest(
            tmp_path, [unresolved_node], {'reify': 1, 'dark_factory': 0},
        )

        # return_value is a real, zero-loss MoveResult/MergeResult (not a
        # bare AsyncMock()) -- run() now reads edges_skipped/mentions_skipped/
        # home_edge_count_after off the result (see _move_result_entry/
        # _merge_result_entry), and an unconfigured MagicMock's truthy
        # attributes and non-identical `+` result would otherwise be
        # misread as a lossy outcome, spuriously flipping 'blocked' to True.
        monkeypatch.setattr(
            _mod, 'move_entity_across_graphs',
            AsyncMock(return_value=MoveResult(uuid='u', source_graph='s', target_graph='t')),
        )
        monkeypatch.setattr(
            _mod, 'merge_foreign_duplicate',
            AsyncMock(return_value=MergeResult(uuid='u', wrong_graph='s', home_graph='t')),
        )

        memory_service = _make_memory_service({
            'reify': _make_graph_mock(
                ro_pages=[[_foreign_row('u-unresolved', 'my_solar_challenge_typo')]],
            ),
            'dark_factory': _make_graph_mock(ro_pages=[[]]),
        })
        args = _args(apply=True, manifest=str(manifest_path))

        report = await _mod.run(args, memory_service)

        assert report['post_verify']['matched'] is True
        assert report['exit_code'] != 0


# ===========================================================================
# Tests: run() apply three-phase barrier ordering (task 2415, step-11/12)
# ===========================================================================

class TestRunApplyPhaseBarrier:
    """Pins the barrier-ordering invariant of the three-phase apply (CGL-eta
    follow-up, task 2415): run() must drive Phase A (create_moved_node, once
    per MOVE node) to completion for the WHOLE batch before Phase B
    (recreate_subgraph_relationships, a single batched call), and Phase B to
    completion before Phase C (delete_source_node, once per MOVE/MERGE
    node). This ordering is what prevents a co-moving neighbour's shared
    RELATES_TO edge from being silently skipped and then destroyed before it
    is ever recreated -- see the module docstring and cross_graph_move.py's
    "Residual hazard" note on the old single-call move_entity_across_graphs.
    """

    def _write_manifest(self, tmp_path, nodes, census) -> Path:
        manifest = _mod.build_manifest(nodes, census, dry_run=True)
        manifest_path = tmp_path / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path

    @pytest.mark.asyncio
    async def test_all_phase_a_creates_precede_phase_b_which_precedes_all_phase_c_deletes(
        self, tmp_path, monkeypatch,
    ):
        """A manifest with two MOVE nodes and one MERGE node: every Phase-A
        create_moved_node call is logged before the single, batched Phase-B
        recreate_subgraph_relationships call, which is logged before every
        Phase-C delete_source_node call."""
        move_a = _classified_node(
            'u-move-a', source_graph='reify', target_graph='dark_factory', disposition=_mod.MOVE,
        )
        move_b = _classified_node(
            'u-move-b', source_graph='reify', target_graph='dark_factory', disposition=_mod.MOVE,
        )
        merge_c = _classified_node(
            'u-merge-c', source_graph='reify', target_graph='know_live', disposition=_mod.MERGE,
        )
        manifest_path = self._write_manifest(
            tmp_path, [move_a, move_b, merge_c], {'reify': 3},
        )

        call_log: list[tuple] = []

        async def _fake_create(graphiti, uuid, source_graph, target_graph, **kwargs):
            call_log.append(('create', uuid))
            return CreateResult(uuid=uuid, source_graph=source_graph, target_graph=target_graph)

        async def _fake_recreate(graphiti, specs):
            call_log.append(('recreate', tuple(sorted(spec['uuid'] for spec in specs))))
            return SubgraphEdgeResult()

        async def _fake_delete(graphiti, uuid, source_graph):
            call_log.append(('delete', uuid))

        monkeypatch.setattr(_mod, 'create_moved_node', AsyncMock(side_effect=_fake_create))
        monkeypatch.setattr(
            _mod, 'recreate_subgraph_relationships', AsyncMock(side_effect=_fake_recreate),
        )
        monkeypatch.setattr(_mod, 'delete_source_node', AsyncMock(side_effect=_fake_delete))

        memory_service = _make_memory_service({
            'reify': _make_graph_mock(ro_pages=[[]]),
            'dark_factory': _make_graph_mock(ro_pages=[[]]),
            'know_live': _make_graph_mock(ro_pages=[[]]),
        })
        args = _args(apply=True, manifest=str(manifest_path))

        await _mod.run(args, memory_service)

        kinds = [entry[0] for entry in call_log]
        create_positions = [i for i, kind in enumerate(kinds) if kind == 'create']
        recreate_positions = [i for i, kind in enumerate(kinds) if kind == 'recreate']
        delete_positions = [i for i, kind in enumerate(kinds) if kind == 'delete']

        # MOVE nodes each get a Phase-A create; the MERGE node does not (its
        # home copy already exists -- see create_moved_node's docstring).
        assert len(create_positions) == 2
        assert {call_log[i][1] for i in create_positions} == {'u-move-a', 'u-move-b'}
        # A single, batched Phase-B call over the WHOLE batch (MOVE + MERGE).
        assert len(recreate_positions) == 1
        assert set(call_log[recreate_positions[0]][1]) == {
            'u-move-a', 'u-move-b', 'u-merge-c',
        }
        # Every MOVE/MERGE node gets a Phase-C delete.
        assert len(delete_positions) == 3
        assert {call_log[i][1] for i in delete_positions} == {
            'u-move-a', 'u-move-b', 'u-merge-c',
        }

        assert max(create_positions) < min(recreate_positions), (
            'every Phase-A create must precede the Phase-B recreate call'
        )
        assert max(recreate_positions) < min(delete_positions), (
            'the Phase-B recreate call must precede every Phase-C delete'
        )


# ===========================================================================
# Tests: run() apply preserves a co-moving pair's shared edge
# (task 2415, step-13/14)
# ===========================================================================

class TestRunApplyCoMovingPreserved:
    """Proves the three-phase barrier-ordered apply preserves a co-moving
    pair's shared RELATES_TO edge -- the whole point of task 2415 -- by
    folding recreate_subgraph_relationships's SubgraphEdgeResult into the
    report instead of discarding it.
    """

    def _write_manifest(self, tmp_path, nodes, census) -> Path:
        manifest = _mod.build_manifest(nodes, census, dry_run=True)
        manifest_path = tmp_path / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path

    @pytest.mark.asyncio
    async def test_co_moving_pair_edge_recreated_count_folded_into_report(
        self, tmp_path, monkeypatch,
    ):
        """Two MOVE nodes (u-A, u-B) sharing a RELATES_TO edge, both moving
        reify -> dark_factory: run() passes BOTH specs to a single
        recreate_subgraph_relationships call, folds its edges_recreated
        count into report['edges_recreated'], records both nodes applied
        (not blocked for edge loss), and -- with a clean post-verify
        re-census -- yields exit_code 0. Proves the co-moving edge is
        preserved, not lost."""
        move_a = _classified_node(
            'u-A', source_graph='reify', target_graph='dark_factory', disposition=_mod.MOVE,
        )
        move_b = _classified_node(
            'u-B', source_graph='reify', target_graph='dark_factory', disposition=_mod.MOVE,
        )
        manifest_path = self._write_manifest(
            tmp_path, [move_a, move_b], {'reify': 2, 'dark_factory': 0},
        )

        monkeypatch.setattr(_mod, 'create_moved_node', AsyncMock(return_value=None))
        monkeypatch.setattr(_mod, 'delete_source_node', AsyncMock(return_value=None))
        recreate_mock = AsyncMock(return_value=SubgraphEdgeResult(
            edges_recreated=1, edges_skipped=0, dropped_cross_target=[],
        ))
        monkeypatch.setattr(_mod, 'recreate_subgraph_relationships', recreate_mock)

        memory_service = _make_memory_service({
            'reify': _make_graph_mock(ro_pages=[[]]),
            'dark_factory': _make_graph_mock(ro_pages=[[]]),
        })
        args = _args(apply=True, manifest=str(manifest_path))

        report = await _mod.run(args, memory_service)

        recreate_mock.assert_awaited_once()
        specs_arg = recreate_mock.await_args.args[1]
        assert {spec['uuid'] for spec in specs_arg} == {'u-A', 'u-B'}

        assert report['edges_recreated'] == 1

        applied = {r['uuid']: r for r in report['apply_results']}
        assert applied['u-A']['applied'] is True
        assert applied['u-A']['blocked'] is False
        assert applied['u-B']['applied'] is True
        assert applied['u-B']['blocked'] is False

        assert report['post_verify']['matched'] is True
        assert report['exit_code'] == 0


# ===========================================================================
# Tests: run() apply surfaces cross-target-dropped edges
# (task 2415, step-15/16)
# ===========================================================================

class TestRunApplyCrossTargetDropped:
    """Proves a cross-target-dropped edge (Phase B's dropped_cross_target --
    two endpoints resolving to DIFFERENT target graphs, which FalkorDB
    cannot recreate in either) is surfaced verbatim for human review, never
    silently omitted, and forces a blocking exit even when the post-verify
    count re-census otherwise reconciles.
    """

    def _write_manifest(self, tmp_path, nodes, census) -> Path:
        manifest = _mod.build_manifest(nodes, census, dry_run=True)
        manifest_path = tmp_path / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path

    @pytest.mark.asyncio
    async def test_dropped_cross_target_edges_surfaced_and_forces_blocking_exit(
        self, tmp_path, monkeypatch,
    ):
        """recreate_subgraph_relationships (monkeypatched) returns one
        dropped_cross_target record; run()'s report carries it verbatim
        under report['dropped_cross_target_edges'] (with its reason), and
        the non-empty list forces a non-zero/blocking exit_code even though
        the post-verify re-census otherwise reconciles cleanly."""
        move_node = _classified_node(
            'u-X', source_graph='reify', target_graph='dark_factory', disposition=_mod.MOVE,
        )
        manifest_path = self._write_manifest(tmp_path, [move_node], {'reify': 1, 'dark_factory': 0})

        dropped_record = {
            'edge_uuid': 'e-1',
            'src_uuid': 'u-X',
            'dst_uuid': 'u-Y',
            'src_target': 'dark_factory',
            'dst_target': 'know_live',
            'reason': (
                'edge endpoints resolve to different target graphs (or an '
                'undeliverable non-migrating endpoint) -- cross-graph '
                'RELATES_TO edges are unsupported; needs manual review'
            ),
        }

        monkeypatch.setattr(_mod, 'create_moved_node', AsyncMock(return_value=None))
        monkeypatch.setattr(_mod, 'delete_source_node', AsyncMock(return_value=None))
        monkeypatch.setattr(
            _mod, 'recreate_subgraph_relationships',
            AsyncMock(return_value=SubgraphEdgeResult(
                edges_recreated=0, edges_skipped=0, dropped_cross_target=[dropped_record],
            )),
        )

        memory_service = _make_memory_service({
            'reify': _make_graph_mock(ro_pages=[[]]),
            'dark_factory': _make_graph_mock(ro_pages=[[]]),
        })
        args = _args(apply=True, manifest=str(manifest_path))

        report = await _mod.run(args, memory_service)

        assert report['dropped_cross_target_edges'] == [dropped_record]
        assert 'needs manual review' in report['dropped_cross_target_edges'][0]['reason']

        assert report['post_verify']['matched'] is True
        assert report['exit_code'] != 0


# ===========================================================================
# Tests: build_arg_parser (step-21/22)
# ===========================================================================

class TestBuildArgParser:
    """Tests for build_arg_parser() -- the testable half of the CLI surface.

    main() itself (live MemoryService wiring) is untested, matching
    purge_knowlive_namespace.py's convention -- see the module docstring.
    """

    def test_defaults_with_no_argv(self):
        """No flags -> apply=False, manifest=None, page_size=DEFAULT_PAGE_SIZE, config=None."""
        parser = _mod.build_arg_parser()

        args = parser.parse_args([])

        assert args.apply is False
        assert args.manifest is None
        assert args.page_size == _mod.DEFAULT_PAGE_SIZE
        assert args.config is None

    def test_overrides_are_parsed(self):
        """Every flag can be overridden from argv."""
        parser = _mod.build_arg_parser()

        args = parser.parse_args([
            '--apply',
            '--manifest', 'm.json',
            '--page-size', '500',
            '--config', 'c.yaml',
        ])

        assert args.apply is True
        assert args.manifest == 'm.json'
        assert args.page_size == 500
        assert args.config == 'c.yaml'
