"""Tests for cleanup_count_snapshots.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution — mirrors the pattern in test_audit_duplicate_tasks.py.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'cleanup_count_snapshots.py'


def _load_module() -> types.ModuleType:
    """Load cleanup_count_snapshots.py from its file path.

    The module is registered in sys.modules under its name so that
    @dataclass and other reflection-based decorators work correctly
    (they call sys.modules.get(cls.__module__)).
    """
    mod_name = 'cleanup_count_snapshots'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass __module__ lookup
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()
EdgeMatch = _mod.EdgeMatch
EntityScanResult = _mod.EntityScanResult


# ===========================================================================
# Helpers
# ===========================================================================

def _entity(uuid: str, name: str, summary: str = '') -> dict:
    return {'uuid': uuid, 'name': name, 'summary': summary}


def _edge(uuid: str, fact: str, name: str = '') -> dict:
    return {'uuid': uuid, 'fact': fact, 'name': name}


# ===========================================================================
# Tests: scan_entities_for_snapshots
# ===========================================================================

class TestScanEntitiesForSnapshots:
    """Tests for scan_entities_for_snapshots(project_id, entities, edges_by_entity)."""

    def _call(self, project_id, entities, edges_by_entity):
        return _mod.scan_entities_for_snapshots(project_id, entities, edges_by_entity)

    def test_snapshot_edge_flagged(self):
        """An edge whose fact contains count-snapshot text is flagged."""
        entities = [_entity('e1', 'Reify')]
        edges_by_entity = {
            'e1': [_edge('edge-abc', 'Reify project: 1505 done / 148 cancelled')],
        }
        results = self._call('dark_factory', entities, edges_by_entity)
        assert len(results) == 1
        r = results[0]
        assert r.entity_uuid == 'e1'
        assert len(r.edge_matches) == 1
        m = r.edge_matches[0]
        assert m.edge_uuid == 'edge-abc'
        assert '1505 done / 148 cancelled' in m.fact_excerpt
        assert m.project_id == 'dark_factory'
        assert 'e1' in m.entity_uuids

    def test_clean_edge_not_flagged(self):
        """An edge with a clean fact is NOT flagged."""
        entities = [_entity('e2', 'SomeEntity')]
        edges_by_entity = {
            'e2': [_edge('edge-clean', 'foo relates to bar')],
        }
        results = self._call('proj', entities, edges_by_entity)
        assert len(results) == 1
        r = results[0]
        assert r.edge_matches == []
        assert r.summary_matched is False

    def test_double_attributed_edge_deduped(self):
        """An edge attributed to two entities yields exactly ONE EdgeMatch
        whose entity_uuids contains both endpoints."""
        # Simulate get_all_valid_edges double-attribution: same edge_uuid
        # appears under both 'e1' and 'e2'.
        entities = [
            _entity('e1', 'EntityA'),
            _entity('e2', 'EntityB'),
        ]
        shared_edge = _edge('edge-shared', 'Reify: 200 done / 50 pending')
        edges_by_entity = {
            'e1': [shared_edge],
            'e2': [shared_edge],
        }
        results = self._call('proj', entities, edges_by_entity)
        # Gather all EdgeMatches across entity results
        all_matches: list[EdgeMatch] = []
        for r in results:
            all_matches.extend(r.edge_matches)
        # Only one unique EdgeMatch by edge_uuid
        unique_uuids = {m.edge_uuid for m in all_matches}
        assert unique_uuids == {'edge-shared'}
        # That match must reference both entity endpoints
        combined_entity_uuids: set[str] = set()
        for m in all_matches:
            combined_entity_uuids.update(m.entity_uuids)
        assert 'e1' in combined_entity_uuids
        assert 'e2' in combined_entity_uuids

    def test_summary_matched_when_summary_contains_snapshot(self):
        """summary_matched=True and summary_excerpt are set when the entity's
        summary text contains a count-snapshot pattern."""
        summary_text = 'Current state: 3355 done, 290 cancelled as of last sync.'
        entities = [_entity('e3', 'StatsEntity', summary=summary_text)]
        edges_by_entity: dict = {}
        results = self._call('proj', entities, edges_by_entity)
        assert len(results) == 1
        r = results[0]
        assert r.summary_matched is True
        assert r.summary_excerpt is not None
        assert '3355 done' in r.summary_excerpt

    def test_no_summary_match_for_clean_summary(self):
        """summary_matched=False when summary has no snapshot text."""
        entities = [_entity('e4', 'CleanEntity', summary='Normal entity summary.')]
        edges_by_entity: dict = {}
        results = self._call('proj', entities, edges_by_entity)
        assert results[0].summary_matched is False
        assert results[0].summary_excerpt is None

    def test_entity_with_no_edges(self):
        """Entity absent from edges_by_entity still returns a result (no matches)."""
        entities = [_entity('e5', 'NoEdges')]
        results = self._call('proj', entities, {})
        assert len(results) == 1
        assert results[0].edge_matches == []

    def test_results_ordered_by_entity_uuid(self):
        """Results are ordered deterministically by entity uuid."""
        entities = [
            _entity('zzz', 'Last'),
            _entity('aaa', 'First'),
        ]
        results = self._call('proj', entities, {})
        assert results[0].entity_uuid == 'aaa'
        assert results[1].entity_uuid == 'zzz'


# ===========================================================================
# Tests: build_audit_memory_payload
# ===========================================================================

class TestBuildAuditMemoryPayload:
    """Tests for build_audit_memory_payload(match, entity_uuid, now_iso)."""

    def _match(self, edge_uuid='edge-001', fact_excerpt='1505 done / 148 cancelled', project_id='proj') -> EdgeMatch:
        return EdgeMatch(edge_uuid=edge_uuid, fact_excerpt=fact_excerpt, project_id=project_id, entity_uuids=['e1'])

    def test_returns_expected_keys(self):
        match = self._match()
        now_iso = '2026-05-28T12:00:00+00:00'
        payload = _mod.build_audit_memory_payload(match, 'e1', now_iso)
        assert set(payload.keys()) >= {'content', 'category', 'agent_id', 'project_id', 'metadata'}

    def test_content_format(self):
        match = self._match()
        payload = _mod.build_audit_memory_payload(match, 'e1', '2026-05-28T12:00:00+00:00')
        expected = (
            'Count-snapshot cleanup: invalidated edge edge-001 on entity e1 '
            '(project=proj); original fact: 1505 done / 148 cancelled'
        )
        assert payload['content'] == expected

    def test_category_and_agent(self):
        match = self._match()
        payload = _mod.build_audit_memory_payload(match, 'e1', 'now')
        assert payload['category'] == 'observations_and_summaries'
        assert payload['agent_id'] == 'cleanup-count-snapshots'

    def test_project_id_propagated(self):
        match = self._match(project_id='my_project')
        payload = _mod.build_audit_memory_payload(match, 'e1', 'now')
        assert payload['project_id'] == 'my_project'

    def test_metadata_shape(self):
        match = self._match(edge_uuid='edge-XYZ', project_id='p1')
        now_iso = '2026-05-28T12:00:00+00:00'
        payload = _mod.build_audit_memory_payload(match, 'entity-9', now_iso)
        meta = payload['metadata']
        assert meta['kind'] == 'count_snapshot_cleanup_audit'
        assert meta['edge_uuid'] == 'edge-XYZ'
        assert meta['entity_uuid'] == 'entity-9'
        assert meta['project_id'] == 'p1'
        assert meta['invalidated_at'] == now_iso
        assert 'fact_text_original' in meta

    def test_fact_text_original_truncated_to_500(self):
        long_fact = 'X' * 600
        match = self._match(fact_excerpt=long_fact)
        payload = _mod.build_audit_memory_payload(match, 'e1', 'now')
        assert len(payload['metadata']['fact_text_original']) == 500

    def test_fact_text_original_short_not_truncated(self):
        match = self._match(fact_excerpt='short fact')
        payload = _mod.build_audit_memory_payload(match, 'e1', 'now')
        assert payload['metadata']['fact_text_original'] == 'short fact'


# ===========================================================================
# Tests: build_audit_report + format_summary_table
# ===========================================================================

class TestBuildAuditReport:
    """Tests for build_audit_report(scan_results_by_project, applied_edges,
    failed_refreshes, dry_run, limit_per_project, generated_at)."""

    def _make_result(self, pid: str, euuid: str, edge_uuid: str) -> EntityScanResult:
        m = EdgeMatch(edge_uuid=edge_uuid, fact_excerpt='1505 done', project_id=pid, entity_uuids=[euuid])
        return EntityScanResult(project_id=pid, entity_uuid=euuid, entity_name='E', edge_matches=[m])

    def test_top_level_keys(self):
        results = {'proj': [self._make_result('proj', 'e1', 'edg1')]}
        report = _mod.build_audit_report(
            scan_results_by_project=results,
            applied_edges=set(),
            failed_refreshes=[],
            dry_run=True,
            limit_per_project=1000,
            generated_at='2026-05-28T00:00:00',
        )
        assert set(report.keys()) >= {'dry_run', 'generated_at', 'projects', 'matches', 'totals'}

    def test_dry_run_flag(self):
        report = _mod.build_audit_report({}, set(), [], True, 1000, 'now')
        assert report['dry_run'] is True
        report2 = _mod.build_audit_report({}, set(), [], False, 1000, 'now')
        assert report2['dry_run'] is False

    def test_match_entry_shape(self):
        results = {'proj': [self._make_result('proj', 'e1', 'edg1')]}
        report = _mod.build_audit_report(results, set(), [], True, 1000, 'now')
        assert len(report['matches']) == 1
        m = report['matches'][0]
        assert m['edge_uuid'] == 'edg1'
        assert 'entity_uuids' in m
        assert m['project_id'] == 'proj'
        assert 'fact_excerpt' in m
        assert 'invalidated' in m

    def test_invalidated_flag_reflects_applied_edges(self):
        results = {'proj': [self._make_result('proj', 'e1', 'edg1')]}
        # Not applied
        r1 = _mod.build_audit_report(results, set(), [], True, 1000, 'now')
        assert r1['matches'][0]['invalidated'] is False
        # Applied
        r2 = _mod.build_audit_report(results, {'edg1'}, [], False, 1000, 'now')
        assert r2['matches'][0]['invalidated'] is True

    def test_per_project_summary(self):
        results = {'proj': [self._make_result('proj', 'e1', 'edg1')]}
        report = _mod.build_audit_report(results, {'edg1'}, [], False, 1000, 'now')
        assert 'proj' in report['projects']
        p = report['projects']['proj']
        assert 'entities_scanned' in p
        assert 'edges_matched' in p
        assert 'edges_invalidated' in p
        assert 'refresh_failures' in p

    def test_totals_aggregate(self):
        r1 = self._make_result('p1', 'e1', 'edg1')
        r2 = self._make_result('p2', 'e2', 'edg2')
        results = {'p1': [r1], 'p2': [r2]}
        report = _mod.build_audit_report(results, {'edg1', 'edg2'}, [], False, 1000, 'now')
        totals = report['totals']
        assert totals['edges_matched'] == 2
        assert totals['edges_invalidated'] == 2

    def test_failed_refreshes_in_report(self):
        results = {}
        failures = [{'entity_uuid': 'e1', 'error': 'NodeNotFound'}]
        report = _mod.build_audit_report(results, set(), failures, False, 1000, 'now')
        assert report['totals'].get('refresh_failures') == 1 or \
               any('refresh_failures' in str(v) for v in report.values())

    def test_deterministic_match_ordering(self):
        """Matches are ordered deterministically (by edge_uuid)."""
        m1 = EdgeMatch(edge_uuid='zzz', fact_excerpt='snap', project_id='p', entity_uuids=['e1'])
        m2 = EdgeMatch(edge_uuid='aaa', fact_excerpt='snap', project_id='p', entity_uuids=['e2'])
        r = EntityScanResult(project_id='p', entity_uuid='e1', entity_name='E', edge_matches=[m1, m2])
        results = {'p': [r]}
        report = _mod.build_audit_report(results, set(), [], True, 1000, 'now')
        uuids = [m['edge_uuid'] for m in report['matches']]
        assert uuids == sorted(uuids)


class TestFormatSummaryTable:
    """Tests for format_summary_table(report)."""

    def _report(self):
        r = _mod.build_audit_report(
            scan_results_by_project={
                'p1': [EntityScanResult('p1', 'e1', 'E1', [
                    EdgeMatch('edg1', 'snap', 'p1', ['e1']),
                ])],
            },
            applied_edges={'edg1'},
            failed_refreshes=[],
            dry_run=False,
            limit_per_project=1000,
            generated_at='2026-05-28',
        )
        return r

    def test_returns_string(self):
        table = _mod.format_summary_table(self._report())
        assert isinstance(table, str)

    def test_contains_project_row(self):
        table = _mod.format_summary_table(self._report())
        assert 'p1' in table

    def test_contains_totals_row(self):
        table = _mod.format_summary_table(self._report())
        assert 'TOTAL' in table.upper() or 'total' in table.lower()


# ===========================================================================
# Tests: select_projects + check_limit_cap
# ===========================================================================

class TestSelectProjects:
    """Tests for select_projects(known_map, project_id_filter)."""

    def _call(self, known_map, project_id_filter=None):
        return _mod.select_projects(known_map, project_id_filter)

    def test_no_filter_returns_all_sorted(self):
        known = {'zzz': '/path/zzz', 'aaa': '/path/aaa', 'mmm': '/path/mmm'}
        result = self._call(known)
        assert result == ['aaa', 'mmm', 'zzz']

    def test_valid_filter_returns_single(self):
        known = {'p1': '/p1', 'p2': '/p2'}
        result = self._call(known, 'p1')
        assert result == ['p1']

    def test_unknown_filter_raises_valueerror(self):
        known = {'p1': '/p1', 'p2': '/p2'}
        with pytest.raises(ValueError, match='p1'):
            self._call(known, 'unknown')


class TestCheckLimitCap:
    """Tests for check_limit_cap(per_project_entity_counts, limit, yes_i_am_sure)."""

    def _call(self, counts, limit=1000, yes_i_am_sure=False):
        return _mod.check_limit_cap(counts, limit, yes_i_am_sure)

    def test_no_exceeding_projects(self):
        counts = {'p1': 100, 'p2': 500}
        exceeding, abort = self._call(counts, limit=1000)
        assert exceeding == []
        assert abort is False

    def test_exceeding_project_aborts_by_default(self):
        counts = {'p1': 2000}
        exceeding, abort = self._call(counts, limit=1000)
        assert 'p1' in exceeding
        assert abort is True

    def test_yes_i_am_sure_overrides_abort(self):
        counts = {'p1': 2000}
        exceeding, abort = self._call(counts, limit=1000, yes_i_am_sure=True)
        assert 'p1' in exceeding
        assert abort is False

    def test_multiple_exceeding_projects(self):
        counts = {'p1': 2000, 'p2': 1500, 'p3': 100}
        exceeding, abort = self._call(counts, limit=1000)
        assert set(exceeding) == {'p1', 'p2'}
        assert abort is True


# ===========================================================================
# Tests: apply_cleanup  (async)
# ===========================================================================

class TestApplyCleanup:
    """Async tests for apply_cleanup(memory, scan_results, now)."""

    def _make_memory(self):
        m = MagicMock()
        m.update_edge = AsyncMock(return_value=None)
        m.add_memory = AsyncMock(return_value=None)
        m.refresh_entity_summary = AsyncMock(return_value=None)
        return m

    def _make_scan_result(self, pid='proj', entity_uuid='e1', edge_uuid='edg1') -> EntityScanResult:
        match = EdgeMatch(edge_uuid=edge_uuid, fact_excerpt='1505 done / 148 cancelled',
                          project_id=pid, entity_uuids=[entity_uuid])
        return EntityScanResult(project_id=pid, entity_uuid=entity_uuid,
                                entity_name='Entity', edge_matches=[match])

    @pytest.mark.asyncio
    async def test_update_edge_called_once_per_unique_edge(self):
        """update_edge is called exactly once per unique edge_uuid."""
        memory = self._make_memory()
        now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC)
        scan_results = [self._make_scan_result()]
        await _mod.apply_cleanup(memory, scan_results, now)
        memory.update_edge.assert_awaited_once()
        call_kwargs = memory.update_edge.call_args
        assert call_kwargs.kwargs.get('edge_uuid') == 'edg1' or \
               (call_kwargs.args and call_kwargs.args[0] == 'edg1')

    @pytest.mark.asyncio
    async def test_doubly_attributed_edge_invalidated_once(self):
        """A doubly-attributed edge (same uuid under two entity results) is
        invalidated exactly once, not twice."""
        memory = self._make_memory()
        now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC)
        # Simulate double attribution: same edge_uuid in two entity results
        shared_match = EdgeMatch(edge_uuid='shared-edge', fact_excerpt='1505 done / 148 cancelled',
                                 project_id='proj', entity_uuids=['e1', 'e2'])
        r1 = EntityScanResult(project_id='proj', entity_uuid='e1', entity_name='E1',
                              edge_matches=[shared_match])
        r2 = EntityScanResult(project_id='proj', entity_uuid='e2', entity_name='E2',
                              edge_matches=[shared_match])
        await _mod.apply_cleanup(memory, [r1, r2], now)
        # Despite two entity results referencing the same edge, update_edge is called once
        assert memory.update_edge.await_count == 1

    @pytest.mark.asyncio
    async def test_add_memory_called_per_invalidated_edge(self):
        """add_memory is called once per invalidated edge with correct kwargs."""
        memory = self._make_memory()
        now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC)
        scan_results = [self._make_scan_result()]
        await _mod.apply_cleanup(memory, scan_results, now)
        memory.add_memory.assert_awaited_once()
        kwargs = memory.add_memory.call_args.kwargs
        assert kwargs.get('category') == 'observations_and_summaries'
        assert kwargs.get('agent_id') == 'cleanup-count-snapshots'
        assert kwargs.get('metadata', {}).get('kind') == 'count_snapshot_cleanup_audit'

    @pytest.mark.asyncio
    async def test_refresh_entity_summary_called_per_entity(self):
        """refresh_entity_summary is called once per affected entity_uuid."""
        memory = self._make_memory()
        now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC)
        scan_results = [self._make_scan_result()]
        await _mod.apply_cleanup(memory, scan_results, now)
        memory.refresh_entity_summary.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_refresh_failure_is_non_fatal(self):
        """A refresh failure does not abort processing; it is recorded in failed_refreshes."""
        memory = self._make_memory()
        memory.refresh_entity_summary = AsyncMock(side_effect=Exception('NodeNotFoundError'))
        now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC)

        # Two separate entities with different edges so both are processed
        r1 = EntityScanResult('proj', 'e1', 'E1', [
            EdgeMatch('edg1', 'snap', 'proj', ['e1']),
        ])
        r2 = EntityScanResult('proj', 'e2', 'E2', [
            EdgeMatch('edg2', 'snap', 'proj', ['e2']),
        ])
        result = await _mod.apply_cleanup(memory, [r1, r2], now)
        # Both edges should still be invalidated (update_edge called twice)
        assert memory.update_edge.await_count == 2
        # Failed refreshes recorded
        assert len(result['failed_refreshes']) == 2

    @pytest.mark.asyncio
    async def test_returns_applied_edges_and_failed_refreshes(self):
        """Return value includes applied_edges (set) and failed_refreshes (list)."""
        memory = self._make_memory()
        now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC)
        scan_results = [self._make_scan_result()]
        result = await _mod.apply_cleanup(memory, scan_results, now)
        assert 'applied_edges' in result
        assert 'edg1' in result['applied_edges']
        assert 'failed_refreshes' in result
        assert result['failed_refreshes'] == []


# ===========================================================================
# Tests: run (async, end-to-end)
# ===========================================================================

class TestRun:
    """Async end-to-end tests for run(args, *, memory)."""

    def _make_memory(self, entities=None, edges_by_entity=None):
        """Build an AsyncMock memory whose graphiti returns fixture data."""
        m = MagicMock()
        m.update_edge = AsyncMock(return_value=None)
        m.add_memory = AsyncMock(return_value=None)
        m.refresh_entity_summary = AsyncMock(return_value=None)
        # graphiti sub-object
        g = MagicMock()
        g.list_entity_nodes = AsyncMock(return_value=entities or [])
        g.get_all_valid_edges = AsyncMock(return_value=edges_by_entity or {})
        m.graphiti = g
        return m

    def _args(self, apply=False, project_id=None, limit_per_project=1000, yes_i_am_sure=False):
        import argparse
        ns = argparse.Namespace(
            apply=apply,
            project_id=project_id,
            limit_per_project=limit_per_project,
            yes_i_am_sure=yes_i_am_sure,
        )
        return ns

    def _known_map(self, pid='dark_factory'):
        return {pid: '/some/path'}

    @pytest.mark.asyncio
    async def test_dry_run_no_writes(self):
        """Default dry-run: no write calls are made."""
        entities = [{'uuid': 'e1', 'name': 'E1', 'summary': ''}]
        edges = {'e1': [{'uuid': 'snap-edge', 'fact': '1505 done / 148 cancelled', 'name': ''}]}
        memory = self._make_memory(entities=entities, edges_by_entity=edges)
        args = self._args(apply=False, project_id='dark_factory')
        known_map = self._known_map('dark_factory')

        report = await _mod.run(args, memory=memory, known_projects_map=known_map)

        memory.update_edge.assert_not_awaited()
        memory.add_memory.assert_not_awaited()
        memory.refresh_entity_summary.assert_not_awaited()
        assert report['dry_run'] is True
        # The matched edge appears in the report
        matched_uuids = [m['edge_uuid'] for m in report.get('matches', [])]
        assert 'snap-edge' in matched_uuids

    @pytest.mark.asyncio
    async def test_apply_invokes_writes(self):
        """--apply flag causes update_edge to be awaited."""
        entities = [{'uuid': 'e1', 'name': 'E1', 'summary': ''}]
        edges = {'e1': [{'uuid': 'snap-edge', 'fact': '1505 done / 148 cancelled', 'name': ''}]}
        memory = self._make_memory(entities=entities, edges_by_entity=edges)
        args = self._args(apply=True, project_id='dark_factory')
        known_map = self._known_map('dark_factory')

        report = await _mod.run(args, memory=memory, known_projects_map=known_map)

        memory.update_edge.assert_awaited_once()
        assert report['dry_run'] is False

    @pytest.mark.asyncio
    async def test_limit_cap_aborts_before_writes(self):
        """When entity count exceeds limit and yes_i_am_sure=False, run aborts before writes."""
        # 3 entities
        entities = [
            {'uuid': f'e{i}', 'name': f'E{i}', 'summary': ''} for i in range(3)
        ]
        memory = self._make_memory(entities=entities, edges_by_entity={})
        args = self._args(apply=True, project_id='dark_factory',
                          limit_per_project=2, yes_i_am_sure=False)
        known_map = self._known_map('dark_factory')

        result = await _mod.run(args, memory=memory, known_projects_map=known_map)

        memory.update_edge.assert_not_awaited()
        # Result should indicate abort
        assert result.get('aborted') is True or 'aborted' in str(result)
