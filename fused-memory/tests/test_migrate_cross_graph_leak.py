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
