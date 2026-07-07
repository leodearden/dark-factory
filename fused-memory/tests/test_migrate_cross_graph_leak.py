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
