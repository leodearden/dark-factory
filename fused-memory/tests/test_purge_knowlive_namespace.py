"""Tests for scripts/purge_knowlive_namespace.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution -- mirrors the pattern in
test_sweep_orphan_flag_markers.py / test_cleanup_count_snapshots.py.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'purge_knowlive_namespace.py'


def _load_module() -> types.ModuleType:
    """Load purge_knowlive_namespace.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators work correctly.
    """
    mod_name = 'purge_knowlive_namespace'
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

def _graphiti_row(uuid: str, labels: list[str] | None = None, name: str = 'Some Entity') -> dict:
    """Build a normalized Graphiti node row dict."""
    return {'uuid': uuid, 'labels': labels or ['Entity'], 'name': name}


def _mem0_member(id: str) -> dict:
    """Build a scroll-shaped Mem0 member dict."""
    return {'id': id, 'created_at': '2026-01-01T00:00:00Z', 'metadata': {'project_id': 'knowlive'}}


def _make_graph_mock(rows: list[list] | None = None) -> MagicMock:
    """Minimal stand-in for conftest.make_graph_mock, scoped to this test module
    so it is usable without the fixture (kept consistent with its shape)."""
    result = MagicMock()
    result.result_set = rows if rows is not None else []
    graph = MagicMock()
    graph.ro_query = AsyncMock(return_value=result)
    graph.query = AsyncMock(return_value=result)
    return graph


# ===========================================================================
# Tests: build_purge_report
# ===========================================================================

class TestBuildPurgeReport:
    """Tests for the pure function build_purge_report(...)."""

    def test_report_shape_and_counts(self):
        """Returned dict has namespace, dry_run, graphiti/mem0/stale_flags blocks
        with counts derived from input lengths."""
        graphiti_rows = [_graphiti_row('g1'), _graphiti_row('g2'), _graphiti_row('g3')]
        mem0_members = [_mem0_member('m1'), _mem0_member('m2')]
        flag_uuids = ('flag-a', 'flag-b')

        report = _mod.build_purge_report(
            'knowlive', graphiti_rows, mem0_members, flag_uuids, dry_run=True,
        )

        assert report['namespace'] == 'knowlive'
        assert report['dry_run'] is True
        assert report['graphiti']['count'] == 3
        assert report['graphiti']['node_uuids'] == ['g1', 'g2', 'g3']
        assert report['mem0']['count'] == 2
        assert report['mem0']['memory_ids'] == ['m1', 'm2']
        assert report['stale_flags']['uuids'] == ['flag-a', 'flag-b']

    def test_dry_run_false_is_preserved(self):
        """dry_run is passed through verbatim, not hardcoded."""
        report = _mod.build_purge_report('knowlive', [], [], (), dry_run=False)
        assert report['dry_run'] is False

    def test_empty_inputs_produce_zero_counts(self):
        """Empty graphiti/mem0/flag inputs produce zero counts and empty id lists."""
        report = _mod.build_purge_report('knowlive', [], [], (), dry_run=True)
        assert report['graphiti'] == {'count': 0, 'node_uuids': []}
        assert report['mem0'] == {'count': 0, 'memory_ids': []}
        assert report['stale_flags'] == {'uuids': []}

    def test_no_io_pure_function(self):
        """build_purge_report performs no I/O -- plain dicts/lists in, dict out."""
        graphiti_rows = [_graphiti_row('g1')]
        mem0_members = [_mem0_member('m1')]
        # Calling twice with identical inputs must be referentially stable in
        # content (no hidden mutable state / side effects between calls).
        r1 = _mod.build_purge_report('knowlive', graphiti_rows, mem0_members, ('f1',), dry_run=True)
        r2 = _mod.build_purge_report('knowlive', graphiti_rows, mem0_members, ('f1',), dry_run=True)
        assert r1 == r2


# ===========================================================================
# Tests: enumerate_graphiti_namespace
# ===========================================================================

class TestEnumerateGraphitiNamespace:
    """Tests for async enumerate_graphiti_namespace(graphiti, namespace, *, limit)."""

    @pytest.mark.asyncio
    async def test_calls_graph_for_with_namespace(self):
        """graphiti._graph_for is called with the namespace argument."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        await _mod.enumerate_graphiti_namespace(graphiti, 'knowlive', limit=1000)

        graphiti._graph_for.assert_called_once_with('knowlive')

    @pytest.mark.asyncio
    async def test_uses_ro_query_not_query(self):
        """Enumeration is read-only: ro_query is used, .query is NEVER called."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        await _mod.enumerate_graphiti_namespace(graphiti, 'knowlive', limit=1000)

        graph.ro_query.assert_called_once()
        graph.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_normalizes_rows_to_dicts(self):
        """result_set rows [uuid, labels, name] are normalized to
        [{'uuid','labels','name'}, ...]."""
        graphiti = MagicMock()
        rows = [
            ['uuid-1', ['Entity'], 'Node A'],
            ['uuid-2', ['Entity', 'Episodic'], 'Node B'],
        ]
        graph = _make_graph_mock(rows)
        graphiti._graph_for = MagicMock(return_value=graph)

        result = await _mod.enumerate_graphiti_namespace(graphiti, 'knowlive', limit=1000)

        assert result == [
            {'uuid': 'uuid-1', 'labels': ['Entity'], 'name': 'Node A'},
            {'uuid': 'uuid-2', 'labels': ['Entity', 'Episodic'], 'name': 'Node B'},
        ]

    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty_list(self):
        """An empty result_set returns an empty list, not an error."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        result = await _mod.enumerate_graphiti_namespace(graphiti, 'knowlive', limit=1000)

        assert result == []

    @pytest.mark.asyncio
    async def test_warns_when_row_count_hits_limit(self, caplog):
        """No-silent-caps: hitting the limit logs a WARNING that enumeration
        may be incomplete."""
        graphiti = MagicMock()
        rows = [[f'uuid-{i}', ['Entity'], f'Node {i}'] for i in range(3)]
        graph = _make_graph_mock(rows)
        graphiti._graph_for = MagicMock(return_value=graph)

        with caplog.at_level('WARNING'):
            await _mod.enumerate_graphiti_namespace(graphiti, 'knowlive', limit=3)

        assert any('limit' in rec.message.lower() for rec in caplog.records), (
            f'Expected a limit-related WARNING, got: {[r.message for r in caplog.records]}'
        )

    @pytest.mark.asyncio
    async def test_no_warning_when_under_limit(self, caplog):
        """Row count below the limit does not log a WARNING."""
        graphiti = MagicMock()
        rows = [['uuid-1', ['Entity'], 'Node A']]
        graph = _make_graph_mock(rows)
        graphiti._graph_for = MagicMock(return_value=graph)

        with caplog.at_level('WARNING'):
            await _mod.enumerate_graphiti_namespace(graphiti, 'knowlive', limit=1000)

        assert caplog.records == []
