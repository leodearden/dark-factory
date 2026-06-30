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


# ===========================================================================
# Tests: enumerate_mem0_namespace
# ===========================================================================

class TestEnumerateMem0Namespace:
    """Tests for async enumerate_mem0_namespace(memory_service, namespace, *, limit)."""

    @pytest.mark.asyncio
    async def test_calls_count_and_scroll_with_project_id_and_empty_filters(self):
        """Both count_memories_by_metadata and get_memories_by_metadata are
        called with project_id=namespace and filters={} (deterministic, not
        semantic search)."""
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(return_value=2)
        memory_service.get_memories_by_metadata = AsyncMock(
            return_value=[_mem0_member('m1'), _mem0_member('m2')]
        )
        memory_service.search = AsyncMock()

        await _mod.enumerate_mem0_namespace(memory_service, 'knowlive', limit=1000)

        memory_service.count_memories_by_metadata.assert_called_once_with(
            project_id='knowlive', filters={},
        )
        scroll_kwargs = memory_service.get_memories_by_metadata.call_args.kwargs
        assert scroll_kwargs.get('project_id') == 'knowlive'
        assert scroll_kwargs.get('filters') == {}
        memory_service.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_members_and_count_tuple(self):
        """Returns (members, count) -- the full list plus the authoritative count."""
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(return_value=2)
        members = [_mem0_member('m1'), _mem0_member('m2')]
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)

        result = await _mod.enumerate_mem0_namespace(memory_service, 'knowlive', limit=1000)

        assert result == (members, 2)

    @pytest.mark.asyncio
    async def test_warns_when_scroll_undercounts(self, caplog):
        """No-silent-caps: fewer enumerated members than the authoritative count
        logs a WARNING (scroll cap reached)."""
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(return_value=5)
        memory_service.get_memories_by_metadata = AsyncMock(
            return_value=[_mem0_member('m1'), _mem0_member('m2')]
        )

        with caplog.at_level('WARNING'):
            await _mod.enumerate_mem0_namespace(memory_service, 'knowlive', limit=2)

        assert any('scroll' in rec.message.lower() or 'limit' in rec.message.lower()
                    for rec in caplog.records), (
            f'Expected a scroll-cap WARNING, got: {[r.message for r in caplog.records]}'
        )

    @pytest.mark.asyncio
    async def test_no_warning_when_fully_enumerated(self, caplog):
        """member count == authoritative count -> no WARNING."""
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(return_value=2)
        memory_service.get_memories_by_metadata = AsyncMock(
            return_value=[_mem0_member('m1'), _mem0_member('m2')]
        )

        with caplog.at_level('WARNING'):
            await _mod.enumerate_mem0_namespace(memory_service, 'knowlive', limit=1000)

        assert caplog.records == []


# ===========================================================================
# Tests: purge_graphiti_namespace
# ===========================================================================

class TestPurgeGraphitiNamespace:
    """Tests for async purge_graphiti_namespace(graphiti, namespace)."""

    @pytest.mark.asyncio
    async def test_calls_graph_for_with_namespace(self):
        """graphiti._graph_for is called with the namespace argument."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        await _mod.purge_graphiti_namespace(graphiti, 'knowlive')

        graphiti._graph_for.assert_called_once_with('knowlive')

    @pytest.mark.asyncio
    async def test_issues_detach_delete_write_query_not_ro_query(self):
        """Purge is a WRITE: graph.query (NOT ro_query) is called with a
        'MATCH (n) DETACH DELETE n'-shaped Cypher statement."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        await _mod.purge_graphiti_namespace(graphiti, 'knowlive')

        graph.query.assert_called_once()
        cypher = graph.query.call_args.args[0]
        assert 'DETACH DELETE' in cypher
        graph.ro_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_ok_result_dict_on_success(self):
        """A successful purge returns a result dict with ok=True and no error."""
        graphiti = MagicMock()
        graph = _make_graph_mock([])
        graphiti._graph_for = MagicMock(return_value=graph)

        result = await _mod.purge_graphiti_namespace(graphiti, 'knowlive')

        assert result == {'ok': True, 'error': None}

    @pytest.mark.asyncio
    async def test_best_effort_raising_query_is_caught_not_raised(self):
        """A raising graph.query is caught (best-effort): the exception does
        NOT propagate, and is instead recorded in the result via ok=False
        plus an error message."""
        graphiti = MagicMock()
        graph = MagicMock()
        graph.query = AsyncMock(side_effect=RuntimeError('FalkorDB connection lost'))
        graph.ro_query = AsyncMock()
        graphiti._graph_for = MagicMock(return_value=graph)

        # Must not raise.
        result = await _mod.purge_graphiti_namespace(graphiti, 'knowlive')

        assert result['ok'] is False
        assert 'FalkorDB connection lost' in result['error']


# ===========================================================================
# Tests: purge_mem0_namespace
# ===========================================================================

class TestPurgeMem0Namespace:
    """Tests for async purge_mem0_namespace(memory_service, namespace, members)."""

    @pytest.mark.asyncio
    async def test_calls_delete_once_per_member(self):
        """delete_memory is called exactly once per member with correct kwargs."""
        memory_service = AsyncMock()
        memory_service.delete_memory = AsyncMock(return_value=None)

        members = [_mem0_member('m1'), _mem0_member('m2')]
        result = await _mod.purge_mem0_namespace(memory_service, 'knowlive', members)

        assert memory_service.delete_memory.call_count == 2
        calls = memory_service.delete_memory.call_args_list
        called_ids = {c.kwargs.get('memory_id') for c in calls}
        assert called_ids == {'m1', 'm2'}
        for c in calls:
            assert c.kwargs.get('store') == 'mem0', f'Expected store=mem0: {c}'
            assert c.kwargs.get('project_id') == 'knowlive', f'Expected project_id=knowlive: {c}'
            assert c.kwargs.get('_source') == 'purge_knowlive_namespace', (
                f'Expected _source=purge_knowlive_namespace: {c}'
            )
        assert result['deleted'] == 2

    @pytest.mark.asyncio
    async def test_best_effort_one_failure_does_not_abort_batch(self):
        """A single delete_memory failure does not abort remaining deletes or raise."""
        memory_service = AsyncMock()
        memory_service.delete_memory = AsyncMock(
            side_effect=[RuntimeError('Qdrant error'), None]
        )

        members = [_mem0_member('m1'), _mem0_member('m2')]
        # Must not raise.
        result = await _mod.purge_mem0_namespace(memory_service, 'knowlive', members)

        assert memory_service.delete_memory.call_count == 2
        assert result['deleted'] == 1
        assert 'm1' in result['failed']

    @pytest.mark.asyncio
    async def test_empty_members_returns_zero_deleted(self):
        """No members -> no delete calls, deleted=0, failed=[]."""
        memory_service = AsyncMock()
        memory_service.delete_memory = AsyncMock(return_value=None)

        result = await _mod.purge_mem0_namespace(memory_service, 'knowlive', [])

        memory_service.delete_memory.assert_not_called()
        assert result['deleted'] == 0
        assert result['failed'] == []


# ===========================================================================
# Tests: retire_stale_flags
# ===========================================================================

class TestRetireStaleFlags:
    """Tests for async retire_stale_flags(memory_service, edge_uuids, host_project,
    *, invalidation_time)."""

    @pytest.mark.asyncio
    async def test_default_args_target_the_two_canonical_uuids(self):
        """With no explicit edge_uuids/host_project, the two canonical
        stale-flag uuids and FLAG_HOST_PROJECT are targeted."""
        memory_service = AsyncMock()
        memory_service.update_edge = AsyncMock(return_value=None)
        invalidation_time = datetime(2026, 6, 30, 21, 0, 0, tzinfo=UTC)

        await _mod.retire_stale_flags(memory_service, invalidation_time=invalidation_time)

        assert memory_service.update_edge.call_count == 2
        called_uuids = {c.kwargs.get('edge_uuid') for c in memory_service.update_edge.call_args_list}
        assert called_uuids == set(_mod.STALE_FLAG_EDGE_UUIDS)
        for c in memory_service.update_edge.call_args_list:
            assert c.kwargs.get('project_id') == _mod.FLAG_HOST_PROJECT
            assert c.kwargs.get('invalid_at') == invalidation_time
            assert c.kwargs.get('_source') == 'purge_knowlive_namespace'

    @pytest.mark.asyncio
    async def test_explicit_args_override_defaults(self):
        """Explicit edge_uuids/host_project are used verbatim instead of the
        module defaults."""
        memory_service = AsyncMock()
        memory_service.update_edge = AsyncMock(return_value=None)
        invalidation_time = datetime(2026, 6, 30, 21, 0, 0, tzinfo=UTC)

        await _mod.retire_stale_flags(
            memory_service, ('custom-uuid',), 'other_project',
            invalidation_time=invalidation_time,
        )

        memory_service.update_edge.assert_called_once_with(
            edge_uuid='custom-uuid',
            project_id='other_project',
            invalid_at=invalidation_time,
            _source='purge_knowlive_namespace',
        )

    @pytest.mark.asyncio
    async def test_best_effort_partial_failure_populates_failed(self):
        """A single failing update_edge does not abort the batch or raise."""
        memory_service = AsyncMock()
        memory_service.update_edge = AsyncMock(
            side_effect=[RuntimeError('graph unavailable'), None]
        )
        invalidation_time = datetime(2026, 6, 30, 21, 0, 0, tzinfo=UTC)

        # Must not raise.
        result = await _mod.retire_stale_flags(
            memory_service, ('uuid-a', 'uuid-b'), 'dark_factory',
            invalidation_time=invalidation_time,
        )

        assert memory_service.update_edge.call_count == 2
        assert result['invalidated'] == 1
        assert 'uuid-a' in result['failed']

    @pytest.mark.asyncio
    async def test_all_succeed_returns_full_invalidated_count_and_no_failed(self):
        """All updates succeeding returns invalidated == len(edge_uuids), failed == []."""
        memory_service = AsyncMock()
        memory_service.update_edge = AsyncMock(return_value=None)
        invalidation_time = datetime(2026, 6, 30, 21, 0, 0, tzinfo=UTC)

        result = await _mod.retire_stale_flags(
            memory_service, ('uuid-a', 'uuid-b'), 'dark_factory',
            invalidation_time=invalidation_time,
        )

        assert result == {'invalidated': 2, 'failed': []}


# ===========================================================================
# Tests: run() -- dry-run path
# ===========================================================================

class TestRunDryRun:
    """Tests for async run(args, memory_service, *, invalidation_time) in dry-run."""

    def _args(self, apply: bool = False, **overrides):
        import types as _types
        base = {'apply': apply}
        base.update(overrides)
        return _types.SimpleNamespace(**base)

    def _make_memory_service(
        self, graphiti_rows: list[list] | None = None,
        mem0_members: list[dict] | None = None,
        mem0_count: int | None = None,
    ):
        """AsyncMock memory_service with a MagicMock .graphiti wired for
        enumerate_graphiti_namespace / enumerate_mem0_namespace."""
        memory_service = AsyncMock()
        graph = _make_graph_mock(graphiti_rows or [])
        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(return_value=graph)
        memory_service.graphiti = graphiti
        members = mem0_members or []
        memory_service.count_memories_by_metadata = AsyncMock(
            return_value=mem0_count if mem0_count is not None else len(members)
        )
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)
        memory_service.update_edge = AsyncMock(return_value=None)
        return memory_service, graph

    @pytest.mark.asyncio
    async def test_dry_run_does_not_mutate(self):
        """Dry-run (args.apply=False): no DETACH DELETE, no delete_memory,
        no update_edge -- nothing is mutated."""
        rows = [['uuid-1', ['Entity'], 'Node A']]
        members = [_mem0_member('m1')]
        memory_service, graph = self._make_memory_service(
            graphiti_rows=rows, mem0_members=members,
        )
        invalidation_time = datetime(2026, 6, 30, 21, 0, 0, tzinfo=UTC)

        await _mod.run(
            self._args(apply=False), memory_service,
            invalidation_time=invalidation_time,
        )

        graph.query.assert_not_called()
        memory_service.delete_memory.assert_not_called()
        memory_service.update_edge.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_report_shape(self):
        """Dry-run report has dry_run=True, enumerated graphiti node_uuids +
        mem0 memory_ids, and the stale_flags uuids."""
        rows = [['uuid-1', ['Entity'], 'Node A'], ['uuid-2', ['Entity'], 'Node B']]
        members = [_mem0_member('m1'), _mem0_member('m2')]
        memory_service, _ = self._make_memory_service(
            graphiti_rows=rows, mem0_members=members,
        )
        invalidation_time = datetime(2026, 6, 30, 21, 0, 0, tzinfo=UTC)

        report = await _mod.run(
            self._args(apply=False), memory_service,
            invalidation_time=invalidation_time,
        )

        assert report['namespace'] == 'knowlive'
        assert report['dry_run'] is True
        assert report['graphiti']['count'] == 2
        assert report['graphiti']['node_uuids'] == ['uuid-1', 'uuid-2']
        assert report['mem0']['count'] == 2
        assert report['mem0']['memory_ids'] == ['m1', 'm2']
        assert report['stale_flags']['uuids'] == list(_mod.STALE_FLAG_EDGE_UUIDS)

    @pytest.mark.asyncio
    async def test_dry_run_uses_default_namespace_when_unset(self):
        """With no args.namespace, LEGACY_NAMESPACE is used to enumerate both stores."""
        memory_service, _ = self._make_memory_service()
        invalidation_time = datetime(2026, 6, 30, 21, 0, 0, tzinfo=UTC)

        await _mod.run(
            self._args(apply=False), memory_service,
            invalidation_time=invalidation_time,
        )

        memory_service.graphiti._graph_for.assert_called_once_with(_mod.LEGACY_NAMESPACE)
        memory_service.count_memories_by_metadata.assert_called_once_with(
            project_id=_mod.LEGACY_NAMESPACE, filters={},
        )

    @pytest.mark.asyncio
    async def test_reports_enumeration_capped_when_graphiti_limit_hit(self):
        """The manifest self-documents an incomplete graphiti enumeration:
        report['graphiti']['enumeration_capped'] is True when the row count
        hits --limit, since --apply's DETACH DELETE is unbounded by limit and
        would silently delete nodes absent from this recovery record."""
        rows = [['uuid-1', ['Entity'], 'Node A'], ['uuid-2', ['Entity'], 'Node B']]
        memory_service, _ = self._make_memory_service(graphiti_rows=rows)
        invalidation_time = datetime(2026, 6, 30, 21, 0, 0, tzinfo=UTC)

        report = await _mod.run(
            self._args(apply=False, limit=2), memory_service,
            invalidation_time=invalidation_time,
        )

        assert report['graphiti']['enumeration_capped'] is True

    @pytest.mark.asyncio
    async def test_reports_enumeration_not_capped_when_under_graphiti_limit(self):
        """Row count below --limit -> enumeration_capped=False."""
        rows = [['uuid-1', ['Entity'], 'Node A']]
        memory_service, _ = self._make_memory_service(graphiti_rows=rows)
        invalidation_time = datetime(2026, 6, 30, 21, 0, 0, tzinfo=UTC)

        report = await _mod.run(
            self._args(apply=False, limit=1000), memory_service,
            invalidation_time=invalidation_time,
        )

        assert report['graphiti']['enumeration_capped'] is False

    @pytest.mark.asyncio
    async def test_reports_authoritative_mem0_count_distinct_from_enumerated(self):
        """report['mem0']['count'] is the authoritative count_memories_by_metadata
        total, not len(memory_ids) -- a scroll-capped namespace must not be
        under-reported in the manifest. report['mem0']['enumerated'] carries the
        enumerated/deletable slice size separately."""
        members = [_mem0_member('m1'), _mem0_member('m2')]
        memory_service, _ = self._make_memory_service(
            mem0_members=members, mem0_count=5,
        )
        invalidation_time = datetime(2026, 6, 30, 21, 0, 0, tzinfo=UTC)

        report = await _mod.run(
            self._args(apply=False, limit=2), memory_service,
            invalidation_time=invalidation_time,
        )

        assert report['mem0']['count'] == 5
        assert report['mem0']['enumerated'] == 2
        assert report['mem0']['memory_ids'] == ['m1', 'm2']


# ===========================================================================
# Tests: run() -- apply path
# ===========================================================================

class TestRunApply:
    """Tests for async run(args, memory_service, *, invalidation_time) in apply mode."""

    def _args(self, apply: bool = True, **overrides):
        import types as _types
        base = {'apply': apply}
        base.update(overrides)
        return _types.SimpleNamespace(**base)

    def _make_memory_service(
        self, graphiti_rows: list[list] | None = None,
        mem0_members: list[dict] | None = None,
        mem0_count: int | None = None,
        delete_side_effect=None,
    ):
        """AsyncMock memory_service with a MagicMock .graphiti wired for both
        enumeration and purge calls (mirrors TestRunDryRun._make_memory_service)."""
        memory_service = AsyncMock()
        graph = _make_graph_mock(graphiti_rows or [])
        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(return_value=graph)
        memory_service.graphiti = graphiti
        members = mem0_members or []
        memory_service.count_memories_by_metadata = AsyncMock(
            return_value=mem0_count if mem0_count is not None else len(members)
        )
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        if delete_side_effect is not None:
            memory_service.delete_memory = AsyncMock(side_effect=delete_side_effect)
        else:
            memory_service.delete_memory = AsyncMock(return_value=None)
        memory_service.update_edge = AsyncMock(return_value=None)
        return memory_service, graph

    @pytest.mark.asyncio
    async def test_apply_purges_graphiti_and_never_targets_another_namespace(self):
        """purge issues a DETACH DELETE write via _graph_for(namespace); EVERY
        _graph_for call (enumerate, purge, after-recount) targets 'knowlive'
        only -- never know_live/dark_factory (safety guard)."""
        rows = [['uuid-1', ['Entity'], 'Node A']]
        memory_service, graph = self._make_memory_service(graphiti_rows=rows)
        invalidation_time = datetime(2026, 6, 30, 21, 0, 0, tzinfo=UTC)

        await _mod.run(
            self._args(apply=True), memory_service,
            invalidation_time=invalidation_time,
        )

        assert len(memory_service.graphiti._graph_for.call_args_list) >= 2
        for call in memory_service.graphiti._graph_for.call_args_list:
            assert call.args == ('knowlive',), (
                f'_graph_for called with unexpected namespace: {call}'
            )
        assert graph.query.call_count == 1
        cypher = graph.query.call_args.args[0]
        assert 'DETACH DELETE' in cypher

    @pytest.mark.asyncio
    async def test_apply_deletes_each_enumerated_mem0_member(self):
        """delete_memory is called once per enumerated mem0 member."""
        members = [_mem0_member('m1'), _mem0_member('m2')]
        memory_service, _ = self._make_memory_service(mem0_members=members)
        invalidation_time = datetime(2026, 6, 30, 21, 0, 0, tzinfo=UTC)

        await _mod.run(
            self._args(apply=True), memory_service,
            invalidation_time=invalidation_time,
        )

        assert memory_service.delete_memory.call_count == 2
        called_ids = {
            c.kwargs.get('memory_id') for c in memory_service.delete_memory.call_args_list
        }
        assert called_ids == {'m1', 'm2'}

    @pytest.mark.asyncio
    async def test_apply_invalidates_each_stale_flag_with_host_project_and_time(self):
        """update_edge is called once per stale-flag uuid with
        project_id=FLAG_HOST_PROJECT and invalid_at=invalidation_time."""
        memory_service, _ = self._make_memory_service()
        invalidation_time = datetime(2026, 6, 30, 21, 0, 0, tzinfo=UTC)

        await _mod.run(
            self._args(apply=True), memory_service,
            invalidation_time=invalidation_time,
        )

        assert memory_service.update_edge.call_count == 2
        called_uuids = {
            c.kwargs.get('edge_uuid') for c in memory_service.update_edge.call_args_list
        }
        assert called_uuids == set(_mod.STALE_FLAG_EDGE_UUIDS)
        for c in memory_service.update_edge.call_args_list:
            assert c.kwargs.get('project_id') == _mod.FLAG_HOST_PROJECT
            assert c.kwargs.get('invalid_at') == invalidation_time

    @pytest.mark.asyncio
    async def test_apply_report_has_after_recounts_and_tallies(self):
        """Apply report has dry_run=False, deleted/invalidated tallies, empty
        failed lists on full success, and an 'after' recount block."""
        rows = [['uuid-1', ['Entity'], 'Node A']]
        members = [_mem0_member('m1'), _mem0_member('m2')]
        memory_service, _ = self._make_memory_service(
            graphiti_rows=rows, mem0_members=members,
        )
        invalidation_time = datetime(2026, 6, 30, 21, 0, 0, tzinfo=UTC)

        report = await _mod.run(
            self._args(apply=True), memory_service,
            invalidation_time=invalidation_time,
        )

        assert report['dry_run'] is False
        assert report['deleted'] == 2
        assert report['mem0_failed'] == []
        assert report['invalidated'] == 2
        assert report['flags_failed'] == []
        assert report['graphiti']['purge'] == {'ok': True, 'error': None}
        assert report['after'] == {'graphiti_count': 1, 'mem0_count': 2}

    @pytest.mark.asyncio
    async def test_apply_tolerates_partial_mem0_delete_failure(self):
        """A single failing mem0 delete does not abort the run or raise;
        report['mem0_failed'] records it and the rest of the report still
        builds normally."""
        members = [_mem0_member('m1'), _mem0_member('m2')]
        memory_service, _ = self._make_memory_service(
            mem0_members=members,
            delete_side_effect=[RuntimeError('Qdrant error'), None],
        )
        invalidation_time = datetime(2026, 6, 30, 21, 0, 0, tzinfo=UTC)

        # Must not raise.
        report = await _mod.run(
            self._args(apply=True), memory_service,
            invalidation_time=invalidation_time,
        )

        assert report['deleted'] == 1
        assert report['mem0_failed'] == ['m1']

    @pytest.mark.asyncio
    async def test_apply_tolerates_graphiti_purge_failure(self):
        """A raising graph.query (Graphiti purge failure) does not abort run():
        report['graphiti']['purge'] surfaces ok=False + the error, and the rest
        of the apply path (mem0 deletes, flag invalidation, after-recounts)
        still completes -- purge_graphiti_namespace's best-effort handling is
        exercised through run(), not just in isolation."""
        rows = [['uuid-1', ['Entity'], 'Node A']]
        result = MagicMock()
        result.result_set = rows
        graph = MagicMock()
        graph.ro_query = AsyncMock(return_value=result)
        graph.query = AsyncMock(side_effect=RuntimeError('FalkorDB connection lost'))
        graphiti = MagicMock()
        graphiti._graph_for = MagicMock(return_value=graph)

        memory_service = AsyncMock()
        memory_service.graphiti = graphiti
        members = [_mem0_member('m1')]
        memory_service.count_memories_by_metadata = AsyncMock(return_value=len(members))
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)
        memory_service.update_edge = AsyncMock(return_value=None)
        invalidation_time = datetime(2026, 6, 30, 21, 0, 0, tzinfo=UTC)

        # Must not raise.
        report = await _mod.run(
            self._args(apply=True), memory_service,
            invalidation_time=invalidation_time,
        )

        assert report['graphiti']['purge']['ok'] is False
        assert 'FalkorDB connection lost' in report['graphiti']['purge']['error']
        # Best-effort: mem0 deletes and flag invalidation still ran to completion.
        assert report['deleted'] == 1
        assert report['invalidated'] == 2
        assert report['after'] == {'graphiti_count': 1, 'mem0_count': 1}

    @pytest.mark.asyncio
    async def test_apply_tolerates_partial_flag_invalidation_failure(self):
        """A single failing update_edge does not abort run(): report['flags_failed']
        records the failed uuid and the rest of the report still builds normally
        -- retire_stale_flags's best-effort handling is exercised through run(),
        not just in isolation."""
        memory_service, _ = self._make_memory_service()
        memory_service.update_edge = AsyncMock(
            side_effect=[RuntimeError('graph unavailable'), None]
        )
        invalidation_time = datetime(2026, 6, 30, 21, 0, 0, tzinfo=UTC)

        # Must not raise.
        report = await _mod.run(
            self._args(apply=True), memory_service,
            invalidation_time=invalidation_time,
        )

        assert report['invalidated'] == 1
        assert report['flags_failed'] == [_mod.STALE_FLAG_EDGE_UUIDS[0]]
