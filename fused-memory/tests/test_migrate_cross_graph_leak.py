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
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import extract_cypher, extract_params

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
) -> dict:
    """Build a normalized census-row dict, as returned by census_foreign_nodes."""
    return {
        'uuid': uuid, 'name': name, 'group_id': group_id,
        'labels': ['Entity'], 'source_graph': source_graph,
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
        reify_graph.query.assert_not_called()
        dark_factory_graph.query.assert_not_called()
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
