"""Tests for cleanup_count_snapshots.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution — mirrors the pattern in test_audit_duplicate_tasks.py.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import complete_paged_read, incomplete_paged_read

from fused_memory.backends.graphiti_client import (
    INCOMPLETE_PAGE_CAP,
    INCOMPLETE_SHORT_READ,
    IncompleteEnumerationError,
)

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


@pytest.fixture(autouse=True)
def _neutralise_store_mutation_preflight(monkeypatch):
    """Keep this MOCK-unit suite independent of the REAL ``~/.mem0``.

    ``run(..., apply=True)`` runs a fail-closed capability preflight before it
    enumerates or scans (task 4293). That probe touches the real filesystem, so
    without this fixture every ``--apply`` test would pass or fail according to
    whether the machine running pytest happens to be able to write mem0's
    history directory -- and it genuinely cannot inside an agent sandbox, which
    is the whole reason the guard exists. This suite is deliberately MOCK-unit
    (a MagicMock/AsyncMock memory, no live Graphiti), so the environment must
    not be an input to it.

    ``TestRunApplyStoreMutationPreflight`` re-rigs this per test -- to refuse,
    to record, or to pass -- so the guard's own behaviour is still pinned
    explicitly rather than assumed away.

    Deliberately NOT ``raising=False``: if the guard is ever removed from the
    script this fixture must break loudly rather than silently no-op.
    """
    monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', lambda **_kw: None)


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
        assert report['totals']['refresh_failures'] == 1

    def test_deterministic_match_ordering(self):
        """Matches are ordered deterministically (by edge_uuid)."""
        m1 = EdgeMatch(edge_uuid='zzz', fact_excerpt='snap', project_id='p', entity_uuids=['e1'])
        m2 = EdgeMatch(edge_uuid='aaa', fact_excerpt='snap', project_id='p', entity_uuids=['e2'])
        r = EntityScanResult(project_id='p', entity_uuid='e1', entity_name='E', edge_matches=[m1, m2])
        results = {'p': [r]}
        report = _mod.build_audit_report(results, set(), [], True, 1000, 'now')
        uuids = [m['edge_uuid'] for m in report['matches']]
        assert uuids == sorted(uuids)

    def test_summaries_matched_in_report(self):
        """Entities with summary_matched=True appear in report['summaries_matched']."""
        r = EntityScanResult(
            project_id='proj',
            entity_uuid='e-sm',
            entity_name='SummaryEntity',
            edge_matches=[],
            summary_matched=True,
            summary_excerpt='3355 done / 290 cancelled',
        )
        report = _mod.build_audit_report({'proj': [r]}, set(), [], True, 1000, 'now')
        assert 'summaries_matched' in report
        assert len(report['summaries_matched']) == 1
        sm = report['summaries_matched'][0]
        assert sm['entity_uuid'] == 'e-sm'
        assert sm['project_id'] == 'proj'
        assert '3355 done' in (sm['summary_excerpt'] or '')

    def test_summaries_matched_empty_when_none(self):
        """summaries_matched is an empty list when no entity has summary_matched."""
        r = EntityScanResult('proj', 'e1', 'E', [], False, None)
        report = _mod.build_audit_report({'proj': [r]}, set(), [], True, 1000, 'now')
        assert report['summaries_matched'] == []

    def test_failed_invalidations_in_report(self):
        """failed_invalidations passed in are surfaced in the report."""
        fi = [{'edge_uuid': 'edg1', 'project_id': 'proj', 'error': 'err', 'phase': 'update_edge'}]
        report = _mod.build_audit_report({}, set(), [], True, 1000, 'now', failed_invalidations=fi)
        assert report['failed_invalidations'] == fi

    def test_failed_invalidations_defaults_to_empty(self):
        """failed_invalidations defaults to [] when not passed."""
        report = _mod.build_audit_report({}, set(), [], True, 1000, 'now')
        assert report['failed_invalidations'] == []

    # -- enumeration completeness (task 4386) -------------------------------
    #
    # ``run()`` reads the whole graph twice per project and the counts above
    # describe whatever those two reads returned.  Threading the per-read
    # completeness into the report is what lets an operator tell a clean audit
    # over the WHOLE corpus from an equally clean-looking audit over a
    # truncated one.  Exercised here against the pure helper, so the shape is
    # pinned independently of the async plumbing that supplies it.

    def test_enumeration_keys_are_merged_into_the_named_project(self):
        """Each project gets ITS OWN read's verdict, not a run-wide one."""
        results = {
            'p1': [self._make_result('p1', 'e1', 'edg1')],
            'p2': [self._make_result('p2', 'e2', 'edg2')],
        }
        report = _mod.build_audit_report(
            scan_results_by_project=results,
            applied_edges=set(),
            failed_refreshes=[],
            dry_run=True,
            limit_per_project=1000,
            generated_at='now',
            enumeration_by_project={
                'p1': {
                    'entities_complete': True,
                    'entities_incomplete_kind': None,
                    'edges_complete': False,
                    'edges_incomplete_kind': INCOMPLETE_SHORT_READ,
                },
                'p2': {
                    'entities_complete': True,
                    'entities_incomplete_kind': None,
                    'edges_complete': True,
                    'edges_incomplete_kind': None,
                },
            },
        )
        assert report['projects']['p1']['edges_complete'] is False, (
            f"p1's edge read was partial; got {report['projects']['p1']!r}"
        )
        assert report['projects']['p1']['edges_incomplete_kind'] == INCOMPLETE_SHORT_READ
        assert report['projects']['p2']['edges_complete'] is True, (
            "p2's own read was complete — a partial read for p1 must not mark "
            f"it; got {report['projects']['p2']!r}"
        )
        assert report['projects']['p2']['edges_incomplete_kind'] is None

    def test_a_project_absent_from_the_mapping_still_gets_all_four_keys(self):
        """Uniform shape on EVERY path, so a consumer never has to branch.

        The keys are absent from the mapping for the paths that return before
        a read happens at all.  Defaulting to None there — rather than
        omitting them — keeps `report['projects'][pid]['edges_complete']` a
        safe subscript everywhere, and None already means exactly the right
        thing: no corpus was observed, so nothing is claimed about it.
        """
        results = {'proj': [self._make_result('proj', 'e1', 'edg1')]}
        report = _mod.build_audit_report(
            scan_results_by_project=results,
            applied_edges=set(),
            failed_refreshes=[],
            dry_run=True,
            limit_per_project=1000,
            generated_at='now',
            enumeration_by_project={},
        )
        p = report['projects']['proj']
        for key in (
            'entities_complete', 'entities_incomplete_kind',
            'edges_complete', 'edges_incomplete_kind',
        ):
            assert key in p, f'{key} must be present even when unknown; got {p!r}'
            assert p[key] is None, f'{key} must default to None (unknown); got {p!r}'

    def test_enumeration_defaults_to_unknown_when_the_argument_is_omitted(self):
        """The parameter is optional, and omitting it is not a completeness claim."""
        results = {'proj': [self._make_result('proj', 'e1', 'edg1')]}
        report = _mod.build_audit_report(results, set(), [], True, 1000, 'now')
        p = report['projects']['proj']
        assert p['entities_complete'] is None
        assert p['edges_complete'] is None
        assert report['totals']['incomplete_enumerations'] == 1, (
            'A project whose reads are UNKNOWN is not a project known to be '
            f"whole; got totals={report['totals']!r}"
        )

    def test_incomplete_enumerations_counts_projects_not_reads(self):
        """One project with BOTH reads partial counts ONCE, not twice.

        The total answers "how many projects is this report unreliable for",
        which is a per-project question.  Counting reads would report 2 for a
        single affected project and read as twice the damage.
        """
        results = {
            'p1': [self._make_result('p1', 'e1', 'edg1')],
            'p2': [self._make_result('p2', 'e2', 'edg2')],
        }
        report = _mod.build_audit_report(
            scan_results_by_project=results,
            applied_edges=set(),
            failed_refreshes=[],
            dry_run=True,
            limit_per_project=1000,
            generated_at='now',
            enumeration_by_project={
                'p1': {
                    'entities_complete': False,
                    'entities_incomplete_kind': INCOMPLETE_SHORT_READ,
                    'edges_complete': False,
                    'edges_incomplete_kind': INCOMPLETE_SHORT_READ,
                },
                'p2': {
                    'entities_complete': True,
                    'entities_incomplete_kind': None,
                    'edges_complete': True,
                    'edges_incomplete_kind': None,
                },
            },
        )
        assert report['totals']['incomplete_enumerations'] == 1, (
            'p1 is ONE partial project even though BOTH of its reads were '
            f"partial; got totals={report['totals']!r}"
        )

    def test_incomplete_enumerations_is_zero_when_every_read_is_proven_complete(self):
        results = {'proj': [self._make_result('proj', 'e1', 'edg1')]}
        report = _mod.build_audit_report(
            scan_results_by_project=results,
            applied_edges=set(),
            failed_refreshes=[],
            dry_run=True,
            limit_per_project=1000,
            generated_at='now',
            enumeration_by_project={
                'proj': {
                    'entities_complete': True,
                    'entities_incomplete_kind': None,
                    'edges_complete': True,
                    'edges_incomplete_kind': None,
                },
            },
        )
        assert report['totals']['incomplete_enumerations'] == 0


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

    # -- corpus completeness column + warning (task 4386) -------------------
    #
    # This table is what an operator actually reads (it goes to stderr, beside
    # the machine-readable JSON on stdout).  The whole point of the task is
    # that a truncated corpus stops being a WARNING buried in a backend log
    # and becomes something the person deciding whether to --apply sees.
    #
    # Asserted via ONE stable substring per contract, never on exact prose or
    # column alignment: pinning the layout would make every future column
    # widening a test failure, which teaches the next reader to edit the
    # assertion rather than think about it.

    def _report_with(self, enumeration_by_project, pids=('p1',)):
        return _mod.build_audit_report(
            scan_results_by_project={
                pid: [EntityScanResult(pid, f'e-{pid}', f'E-{pid}', [
                    EdgeMatch(f'edg-{pid}', 'snap', pid, [f'e-{pid}']),
                ])]
                for pid in pids
            },
            applied_edges=set(),
            failed_refreshes=[],
            dry_run=True,
            limit_per_project=1000,
            generated_at='2026-05-28',
            enumeration_by_project=enumeration_by_project,
        )

    @staticmethod
    def _row_for(table: str, pid: str) -> str:
        matches = [line for line in table.splitlines() if line.startswith(pid)]
        assert len(matches) == 1, f'expected exactly one {pid!r} row in:\n{table}'
        return matches[0]

    @staticmethod
    def _warning_lines(table: str) -> list[str]:
        """The trailing warning line(s), isolated from the table body.

        Asserting against the WHOLE table cannot distinguish "the warning
        names p1" from "a p1 row exists", which is always true — so the
        contract has to be checked against this line specifically.
        """
        return [line for line in table.splitlines() if line.startswith('WARNING:')]

    def test_a_whole_corpus_renders_ok_and_warns_about_nothing(self):
        table = _mod.format_summary_table(self._report_with({
            'p1': {
                'entities_complete': True, 'entities_incomplete_kind': None,
                'edges_complete': True, 'edges_incomplete_kind': None,
            },
        }))
        assert 'ok' in self._row_for(table, 'p1'), (
            f'A proven-whole corpus must be marked ok; got:\n{table}'
        )
        assert 'PARTIAL' not in table, (
            f'Nothing was partial, so nothing may claim it was; got:\n{table}'
        )

    def test_a_partial_corpus_renders_PARTIAL(self):
        table = _mod.format_summary_table(self._report_with({
            'p1': {
                'entities_complete': True, 'entities_incomplete_kind': None,
                'edges_complete': False, 'edges_incomplete_kind': INCOMPLETE_SHORT_READ,
            },
        }))
        assert 'PARTIAL' in self._row_for(table, 'p1'), (
            f'Either read being partial makes the project partial; got:\n{table}'
        )

    def test_an_unknown_corpus_renders_a_question_mark(self):
        """UNKNOWN is not ok.

        Rendering an unmeasured read as ok would be the one failure this whole
        signal exists to prevent: a run that never looked reading as a run that
        looked and found everything.
        """
        table = _mod.format_summary_table(self._report_with({}))
        row = self._row_for(table, 'p1')
        assert '?' in row, f'An unmeasured corpus must not render as ok; got:\n{table}'
        assert 'ok' not in row, f'got:\n{table}'

    def test_a_partial_corpus_appends_a_loud_warning_line(self):
        """The operator's next move is to invalidate edges, which is exactly
        the move that is unsafe on a partial read — so the table says plainly
        that an absence of matches there proves nothing."""
        table = _mod.format_summary_table(self._report_with({
            'p1': {
                'entities_complete': False, 'entities_incomplete_kind': INCOMPLETE_SHORT_READ,
                'edges_complete': True, 'edges_incomplete_kind': None,
            },
        }))
        warnings = self._warning_lines(table)
        assert len(warnings) == 1, (
            f'A partial corpus must append exactly one warning; got:\n{table}'
        )
        assert 'not proof' in warnings[0].lower(), (
            'A partial corpus must carry the "absence of matches is not proof '
            f'of cleanliness" warning; got:\n{table}'
        )
        assert 'p1' in warnings[0], (
            f'The warning must name the affected project; got:\n{table}'
        )

    def test_the_warning_names_the_partial_project_and_not_the_whole_one(self):
        """The naming is the whole point of the warning.

        With one project the "names the affected project" assertion is
        untestable against the table as a whole — 'p1' is in it either way,
        because 'p1' has a ROW. Two projects, one partial, makes the
        interpolation load-bearing: drop it and the warning still fires but
        stops telling the operator WHERE it is unsafe to --apply, which on a
        many-project run is the difference between a usable warning and one
        that condemns everything.
        """
        table = _mod.format_summary_table(self._report_with(
            {
                'p1': {
                    'entities_complete': False,
                    'entities_incomplete_kind': INCOMPLETE_SHORT_READ,
                    'edges_complete': True, 'edges_incomplete_kind': None,
                },
                'p2': {
                    'entities_complete': True, 'entities_incomplete_kind': None,
                    'edges_complete': True, 'edges_incomplete_kind': None,
                },
            },
            pids=('p1', 'p2'),
        ))
        warnings = self._warning_lines(table)
        assert len(warnings) == 1, f'expected exactly one warning; got:\n{table}'
        assert 'p1' in warnings[0], (
            f'The warning must name the PARTIAL project; got:\n{table}'
        )
        assert 'p2' not in warnings[0], (
            'The warning must not condemn a project whose corpus was proven '
            f'whole; got:\n{table}'
        )
        # And the per-project cells still disagree, so the row and the warning
        # tell the same story.
        assert 'PARTIAL' in self._row_for(table, 'p1'), f'got:\n{table}'
        assert 'ok' in self._row_for(table, 'p2'), f'got:\n{table}'

    # -- the TOTALS row's Corpus cell --------------------------------------
    #
    # The roll-up is a count of PROJECTS, but it lands in a column whose
    # per-project vocabulary is words.  Rendered bare it put a `0` or a `2`
    # in a column of `ok`/`PARTIAL`/`?` — and `0` is not `ok`, so the one
    # cell summarising whether the whole report is trustworthy read
    # ambiguously.  (amendment, reviewer_comprehensive observability finding)

    @staticmethod
    def _total_row(table: str) -> str:
        matches = [line for line in table.splitlines() if line.startswith('TOTAL')]
        assert len(matches) == 1, f'expected exactly one TOTAL row in:\n{table}'
        return matches[0]

    def test_the_totals_corpus_cell_reads_ok_when_every_project_is_whole(self):
        table = _mod.format_summary_table(self._report_with({
            'p1': {
                'entities_complete': True, 'entities_incomplete_kind': None,
                'edges_complete': True, 'edges_incomplete_kind': None,
            },
        }))
        total = self._total_row(table)
        assert total.rstrip().endswith('ok'), (
            'With nothing partial the roll-up must speak the column\'s own '
            f'vocabulary, not render a bare 0; got:\n{table}'
        )
        assert not total.rstrip().endswith('0'), (
            f'A bare 0 in a column of words reads ambiguously; got:\n{table}'
        )

    def test_the_totals_corpus_cell_labels_the_partial_count(self):
        """Two not-ok projects for two DIFFERENT reasons — one observed and
        found partial, one never measured — because the count rolls both up.
        That is also why the label is lowercase `partial` rather than the
        per-project `PARTIAL` token: p2 was never observed, so claiming it
        was observed-and-incomplete would overstate what is known."""
        table = _mod.format_summary_table(self._report_with(
            {
                'p1': {
                    'entities_complete': False,
                    'entities_incomplete_kind': INCOMPLETE_SHORT_READ,
                    'edges_complete': True, 'edges_incomplete_kind': None,
                },
                # p2 deliberately absent -> UNKNOWN, which is also not `ok`.
            },
            pids=('p1', 'p2'),
        ))
        total = self._total_row(table)
        assert '2 partial' in total, (
            'The roll-up must be an explicitly-labelled count, not a bare '
            f'integer; got:\n{table}'
        )
        assert 'PARTIAL' in self._row_for(table, 'p1'), f'got:\n{table}'
        assert '?' in self._row_for(table, 'p2'), f'got:\n{table}'

    def test_a_whole_corpus_appends_no_warning_line(self):
        table = _mod.format_summary_table(self._report_with({
            'p1': {
                'entities_complete': True, 'entities_incomplete_kind': None,
                'edges_complete': True, 'edges_incomplete_kind': None,
            },
        }))
        assert 'not proof' not in table.lower(), (
            f'Nothing was partial, so the warning must not fire; got:\n{table}'
        )


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
    async def test_update_edge_failure_non_fatal(self):
        """When update_edge raises, the edge is skipped, failure recorded in
        failed_invalidations, and remaining edges are still processed."""
        memory = self._make_memory()
        memory.update_edge = AsyncMock(side_effect=Exception('StoreError'))
        now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC)
        r1 = EntityScanResult('proj', 'e1', 'E1', [
            EdgeMatch('edg1', 'snap', 'proj', ['e1']),
        ])
        r2 = EntityScanResult('proj', 'e2', 'E2', [
            EdgeMatch('edg2', 'snap', 'proj', ['e2']),
        ])
        result = await _mod.apply_cleanup(memory, [r1, r2], now)
        # update_edge was attempted for both edges
        assert memory.update_edge.await_count == 2
        # Neither edge was invalidated (both failed)
        assert result['applied_edges'] == set()
        # Both failures recorded in failed_invalidations with phase=update_edge
        assert len(result['failed_invalidations']) == 2
        phases = {f['phase'] for f in result['failed_invalidations']}
        assert phases == {'update_edge'}
        # add_memory never called (edges were skipped)
        memory.add_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_add_memory_failure_after_update_recorded(self):
        """When update_edge succeeds but add_memory fails, the edge IS in
        applied_edges (already invalidated) but the failure is surfaced in
        failed_invalidations with phase=add_memory."""
        memory = self._make_memory()
        memory.add_memory = AsyncMock(side_effect=Exception('MemError'))
        now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC)
        scan_results = [self._make_scan_result()]
        result = await _mod.apply_cleanup(memory, scan_results, now)
        # Edge was invalidated
        assert 'edg1' in result['applied_edges']
        # Failure surfaced in failed_invalidations
        assert len(result['failed_invalidations']) == 1
        fi = result['failed_invalidations'][0]
        assert fi['phase'] == 'add_memory'
        assert fi['edge_uuid'] == 'edg1'

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
        """Return value includes applied_edges, failed_refreshes, and failed_invalidations."""
        memory = self._make_memory()
        now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=UTC)
        scan_results = [self._make_scan_result()]
        result = await _mod.apply_cleanup(memory, scan_results, now)
        assert 'applied_edges' in result
        assert 'edg1' in result['applied_edges']
        assert 'failed_refreshes' in result
        assert result['failed_refreshes'] == []
        assert 'failed_invalidations' in result
        assert result['failed_invalidations'] == []


# ===========================================================================
# Tests: run (async, end-to-end)
# ===========================================================================

class TestRun:
    """Async end-to-end tests for run(args, *, memory)."""

    def _make_memory(
        self, entities=None, edges_by_entity=None, *,
        entities_paged=None, edges_paged=None,
    ):
        """Build an AsyncMock memory whose graphiti returns fixture data.

        Both whole-graph reads are the ``enumerate_*`` pair, which returns
        ``(collection, PagedRead)`` — the script reads them directly for the
        completeness signal rather than through the ``list_entity_nodes`` /
        ``get_all_valid_edges`` shims, which discard it (task 4386).  The
        default here is a PROVEN-complete read, matching what the shims
        guaranteed by raising: every pre-existing test in this class asserts
        behaviour over a whole corpus, so that is the honest double for them.
        Pass ``entities_paged`` / ``edges_paged`` (an ``incomplete_paged_read``)
        to drive the partial-corpus paths.
        """
        m = MagicMock()
        m.update_edge = AsyncMock(return_value=None)
        m.add_memory = AsyncMock(return_value=None)
        m.refresh_entity_summary = AsyncMock(return_value=None)
        # graphiti sub-object
        g = MagicMock()
        entities = entities or []
        edges_by_entity = edges_by_entity or {}
        g.enumerate_entity_nodes = AsyncMock(return_value=(
            entities,
            entities_paged if entities_paged is not None
            else complete_paged_read(rows_seen=len(entities)),
        ))
        g.enumerate_all_valid_edges = AsyncMock(return_value=(
            edges_by_entity,
            edges_paged if edges_paged is not None
            else complete_paged_read(
                rows_seen=sum(len(v) for v in edges_by_entity.values()),
            ),
        ))
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

        # Per-project read completeness (task 4386). The counts above describe
        # SOME corpus; without these keys the report never says which one, so a
        # clean audit over the whole graph is indistinguishable in the JSON from
        # a clean audit over whatever half of it a truncated read returned.
        proj = report['projects']['dark_factory']
        assert proj['entities_complete'] is True, (
            f'A proven-complete node read must be surfaced as True; got {proj!r}'
        )
        assert proj['entities_incomplete_kind'] is None, (
            f'A complete read has no incompleteness kind; got {proj!r}'
        )
        assert proj['edges_complete'] is True, (
            f'A proven-complete edge read must be surfaced as True; got {proj!r}'
        )
        assert proj['edges_incomplete_kind'] is None, (
            f'A complete read has no incompleteness kind; got {proj!r}'
        )
        assert report['totals']['incomplete_enumerations'] == 0, (
            'Both reads were proven complete, so no project is partial; got '
            f"totals={report['totals']!r}"
        )

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
        memory.graphiti.enumerate_all_valid_edges.assert_not_awaited()
        # Result should indicate abort
        assert result.get('aborted') is True
        assert result.get('exceeding_projects') == ['dark_factory']

    @pytest.mark.asyncio
    async def test_unknown_project_id_emits_abort_output(self, capsys):
        """run() with an unknown --project-id prints a JSON abort payload on
        stdout and a human-readable message on stderr; no enumeration happens."""
        memory = self._make_memory(entities=[], edges_by_entity={})
        args = self._args(apply=False, project_id='unknown-id')
        known_map = {'dark_factory': '/p'}

        result = await _mod.run(args, memory=memory, known_projects_map=known_map)

        # (i) return-value contract preserved
        assert result.get('aborted') is True
        assert 'error' in result

        # (ii) stdout: non-empty, valid JSON, contains aborted + error
        captured = capsys.readouterr()
        assert captured.out.strip(), 'stdout should be non-empty on abort'
        out_data = json.loads(captured.out)
        assert out_data.get('aborted') is True
        assert 'error' in out_data

        # (iii) stderr: non-empty, mentions the unknown project_id
        assert captured.err.strip(), 'stderr should be non-empty on abort'
        assert 'unknown-id' in captured.err

        # (iv) enumeration did NOT happen (abort before the entity-fetch loop)
        memory.graphiti.enumerate_entity_nodes.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_limit_cap_abort_emits_abort_output(self, capsys):
        """When entity count exceeds limit, run() prints a JSON abort payload on
        stdout (with exceeding_projects) and a human-readable hint on stderr."""
        entities = [
            {'uuid': f'e{i}', 'name': f'E{i}', 'summary': ''} for i in range(3)
        ]
        memory = self._make_memory(entities=entities, edges_by_entity={})
        args = self._args(apply=True, project_id='dark_factory',
                          limit_per_project=2, yes_i_am_sure=False)
        known_map = self._known_map('dark_factory')

        result = await _mod.run(args, memory=memory, known_projects_map=known_map)

        # Return-value contract preserved (existing test still passes)
        assert result.get('aborted') is True
        assert result.get('exceeding_projects') == ['dark_factory']

        # stdout: non-empty, valid JSON, aborted + exceeding_projects
        captured = capsys.readouterr()
        assert captured.out.strip(), 'stdout should be non-empty on limit-cap abort'
        out_data = json.loads(captured.out)
        assert out_data.get('aborted') is True
        assert out_data.get('exceeding_projects') == ['dark_factory']

        # stderr: non-empty, names the over-cap project
        assert captured.err.strip(), 'stderr should be non-empty on limit-cap abort'
        assert 'dark_factory' in captured.err


    # -- partial and refused corpora (task 4386) ----------------------------

    @pytest.mark.asyncio
    async def test_an_empirically_partial_edge_read_is_reported_and_still_scanned(self):
        """EMPIRICAL incompleteness warns and proceeds — it does not abort.

        A census disagreeing by a few rows is the expected signature of a
        graph being written to mid-read, so raising would take a routine audit
        down for something that self-heals next run.  The scan therefore
        proceeds over what WAS fetched, and the report says the corpus was
        partial so the operator can weigh the result accordingly.
        """
        entities = [_entity('e1', 'E1')]
        edges = {'e1': [{'uuid': 'snap-edge', 'fact': '1505 done / 148 cancelled', 'name': ''}]}
        memory = self._make_memory(
            entities=entities,
            edges_by_entity=edges,
            edges_paged=incomplete_paged_read(
                INCOMPLETE_SHORT_READ, rows_seen=1, expected_rows=9,
            ),
        )

        report = await _mod.run(
            self._args(apply=False, project_id='dark_factory'),
            memory=memory,
            known_projects_map=self._known_map('dark_factory'),
        )

        proj = report['projects']['dark_factory']
        assert proj['edges_complete'] is False, (
            f'A partial edge read must be reported as False; got {proj!r}'
        )
        assert proj['edges_incomplete_kind'] == INCOMPLETE_SHORT_READ, (
            f'The kind is the stable discriminator; got {proj!r}'
        )
        assert proj['entities_complete'] is True, (
            'The two reads are INDEPENDENT — a partial edge read says nothing '
            f'about the node read, which was whole; got {proj!r}'
        )
        assert report['totals']['incomplete_enumerations'] == 1, (
            f"got totals={report['totals']!r}"
        )
        # ...and the scan still ran over what was fetched.
        assert 'snap-edge' in [m['edge_uuid'] for m in report['matches']], (
            'An empirical incompleteness must not discard the rows that WERE '
            f"fetched; got matches={report['matches']!r}"
        )

    @pytest.mark.asyncio
    async def test_an_empirically_partial_node_read_is_reported(self):
        """The node read carries its own verdict, under its own keys."""
        memory = self._make_memory(
            entities=[_entity('e1', 'E1')],
            edges_by_entity={},
            entities_paged=incomplete_paged_read(
                INCOMPLETE_SHORT_READ, rows_seen=1, expected_rows=40,
            ),
        )

        report = await _mod.run(
            self._args(apply=False, project_id='dark_factory'),
            memory=memory,
            known_projects_map=self._known_map('dark_factory'),
        )

        proj = report['projects']['dark_factory']
        assert proj['entities_complete'] is False, f'got {proj!r}'
        assert proj['entities_incomplete_kind'] == INCOMPLETE_SHORT_READ, f'got {proj!r}'
        assert proj['edges_complete'] is True, (
            f'The edge read was whole and must say so; got {proj!r}'
        )
        assert report['totals']['incomplete_enumerations'] == 1, (
            'One project is partial, however many of its reads were; got '
            f"totals={report['totals']!r}"
        )

    @pytest.mark.asyncio
    async def test_the_cap_abort_payload_carries_the_node_read_completeness(self):
        """A real fail-open, not a nicety.

        ``check_limit_cap`` decides the cap on ``len(entities)`` from the NODE
        read, so a TRUNCATED node read UNDER-counts and an oversized project
        can slip UNDER the cap — the cap silently stops protecting exactly the
        projects it exists for.  The abort returns before ``build_audit_report``
        runs, so the report path cannot carry this and the payload must.
        """
        entities = [_entity(f'e{i}', f'E{i}') for i in range(3)]
        memory = self._make_memory(
            entities=entities,
            edges_by_entity={},
            entities_paged=incomplete_paged_read(
                INCOMPLETE_SHORT_READ, rows_seen=3, expected_rows=300,
            ),
        )

        result = await _mod.run(
            self._args(apply=True, project_id='dark_factory',
                       limit_per_project=2, yes_i_am_sure=False),
            memory=memory,
            known_projects_map=self._known_map('dark_factory'),
        )

        assert result.get('aborted') is True
        assert 'entities_complete' in result, (
            'The cap was decided on a count from a read that may have been '
            f'truncated; the payload must say so. got {result!r}'
        )
        assert result['entities_complete'] == {'dark_factory': False}, (
            f'got {result!r}'
        )
        assert result['entities_incomplete_kind'] == {
            'dark_factory': INCOMPLETE_SHORT_READ,
        }, f'got {result!r}'

    @pytest.mark.asyncio
    async def test_a_structurally_refused_node_read_aborts_before_any_write(self):
        """The fail-closed guard survives the move off the raising shim.

        ``enumerate_*`` never raises, so this script re-applies the shared
        policy itself.  If that call were ever dropped, a page-capped read
        would be taken for the whole corpus and ``--apply`` would invalidate
        edges over a fabricated one.  Structural kinds are deterministic and
        non-transient, so raising on them cannot flap.
        """
        memory = self._make_memory(
            entities=[_entity('e1', 'E1')],
            edges_by_entity={},
            entities_paged=incomplete_paged_read(INCOMPLETE_PAGE_CAP, rows_seen=1),
        )

        with pytest.raises(IncompleteEnumerationError):
            await _mod.run(
                self._args(apply=True, project_id='dark_factory'),
                memory=memory,
                known_projects_map=self._known_map('dark_factory'),
            )

        memory.update_edge.assert_not_awaited()
        memory.graphiti.enumerate_all_valid_edges.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_structurally_refused_edge_read_aborts_before_any_write(self):
        """Same guard on the second read: the node read succeeding is not a
        licence to write back over a fabricated edge corpus."""
        memory = self._make_memory(
            entities=[_entity('e1', 'E1')],
            edges_by_entity={},
            edges_paged=incomplete_paged_read(INCOMPLETE_PAGE_CAP),
        )

        with pytest.raises(IncompleteEnumerationError):
            await _mod.run(
                self._args(apply=True, project_id='dark_factory'),
                memory=memory,
                known_projects_map=self._known_map('dark_factory'),
            )

        memory.update_edge.assert_not_awaited()


# ===========================================================================
# Tests: --apply store-mutation preflight
# ===========================================================================

class TestRunApplyStoreMutationPreflight:
    """``--apply`` refuses to START when this process cannot write mem0's store.

    Ported from ``test_sweep_toolcall_xml_leak.TestRunApplyStoreMutationPreflight``
    (task 3686), which is the in-repo precedent for this contract.

    ``apply_cleanup`` invalidates edges, writes an audit memory per edge and
    refreshes entity summaries, each in a sequential loop with its OWN
    best-effort ``except Exception``. ``StoreMutationUnavailable`` subclasses
    ``RuntimeError``, so a probe inside any of those loops would be absorbed
    into ``failed_invalidations`` / ``failed_refreshes`` rows -- and
    ``update_edge`` invalidates the edge BEFORE its audit memory is written, so
    each of those "failures" would be an already-invalidated edge with no
    rollback record. Only a run-wide probe ahead of the scan bounds that.
    """

    def _make_memory(self):
        """Mirror of ``TestRun._make_memory``, wired for a snapshot edge that
        would drive all three mutations."""
        m = MagicMock()
        m.update_edge = AsyncMock(return_value=None)
        m.add_memory = AsyncMock(return_value=None)
        m.refresh_entity_summary = AsyncMock(return_value=None)
        g = MagicMock()
        g.enumerate_entity_nodes = AsyncMock(return_value=(
            [_entity('e1', 'E1')],
            complete_paged_read(rows_seen=1),
        ))
        g.enumerate_all_valid_edges = AsyncMock(return_value=(
            {'e1': [{'uuid': 'snap-edge', 'fact': '1505 done / 148 cancelled', 'name': ''}]},
            complete_paged_read(rows_seen=1),
        ))
        m.graphiti = g
        return m

    def _args(self, apply=True, project_id='dark_factory'):
        """Mirror of ``TestRun._args``."""
        import argparse
        return argparse.Namespace(
            apply=apply,
            project_id=project_id,
            limit_per_project=1000,
            yes_i_am_sure=False,
        )

    def _known_map(self, pid='dark_factory'):
        return {pid: '/some/path'}

    @staticmethod
    def _deny(monkeypatch):
        """Rig the preflight to refuse, as it would inside an agent sandbox."""
        def _raise(*_args, **_kwargs):
            raise _mod.StoreMutationUnavailable('SENTINEL-store-unwritable')

        monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', _raise)

    @staticmethod
    def _fail_closed_records(caplog) -> list:
        """The guard site's OWN diagnosis.

        ``main`` has no handler at all here -- ``_run_live`` re-raises through
        its ``finally`` and ``asyncio.run`` lets it out -- so this ERROR record
        is the ONLY place the operator is told what was refused and what to do
        instead. Pinned on the fail-closed marker and the remedy noun ONLY, so
        every other word of the message stays free to reword.

        Asserting on message CONTENT is deliberate, and is the narrow exception
        to the repo's don't-pin-guard-message-prose norm (task 3799): the record
        this test is about is defined BY its content -- mere record-existence
        would still pass if the whole diagnosis were replaced by "boom",
        precisely the regression this exists to catch. Verified non-vacuous:
        mutating the marker in the script turns this assertion red (task 4127
        amendment).
        """
        return [
            rec for rec in caplog.records
            if rec.name == 'cleanup_count_snapshots'
            and rec.levelname == 'ERROR'
            and 'NOT started (fail-closed)' in rec.getMessage()
            and 'MCP server' in rec.getMessage()
        ]

    @pytest.mark.asyncio
    async def test_apply_performs_zero_mutations_when_the_store_is_unwritable(
        self, monkeypatch
    ):
        """The whole point: refuse to start rather than half-complete.

        ALL THREE mutation entry points are asserted un-awaited, not just one:
        each sits behind its own swallowing ``except Exception``, so a
        zero-mutation claim covering only one of them would be vacuous.
        """
        self._deny(monkeypatch)
        memory = self._make_memory()

        with pytest.raises(
            _mod.StoreMutationUnavailable, match='SENTINEL-store-unwritable'
        ):
            await _mod.run(
                self._args(apply=True),
                memory=memory,
                known_projects_map=self._known_map(),
            )

        memory.update_edge.assert_not_awaited()
        memory.add_memory.assert_not_awaited()
        memory.refresh_entity_summary.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_guard_sits_before_every_backend_read(self, monkeypatch):
        """It aborts without a single round-trip: neither the first-pass entity
        enumeration nor the second-pass edge scan is paid for by a run that was
        never going to be allowed to mutate."""
        self._deny(monkeypatch)
        memory = self._make_memory()

        with pytest.raises(_mod.StoreMutationUnavailable):
            await _mod.run(
                self._args(apply=True),
                memory=memory,
                known_projects_map=self._known_map(),
            )

        memory.graphiti.enumerate_entity_nodes.assert_not_awaited()
        memory.graphiti.enumerate_all_valid_edges.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_dry_run_is_never_gated_on_write_capability(self, monkeypatch):
        """A read-only run mutates nothing, so it must not require the ability
        to mutate -- the audit report stays obtainable from anywhere, with the
        deny still installed."""
        self._deny(monkeypatch)
        memory = self._make_memory()

        report = await _mod.run(
            self._args(apply=False),
            memory=memory,
            known_projects_map=self._known_map(),
        )

        assert report['dry_run'] is True
        assert not report.get('aborted')
        memory.graphiti.enumerate_entity_nodes.assert_awaited()
        memory.graphiti.enumerate_all_valid_edges.assert_awaited()
        memory.update_edge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_apply_is_unchanged_when_the_preflight_passes(self, monkeypatch):
        """Happy path: a writable environment cleans up exactly as before."""
        monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', lambda **_kw: None)
        memory = self._make_memory()

        report = await _mod.run(
            self._args(apply=True),
            memory=memory,
            known_projects_map=self._known_map(),
        )

        assert report['dry_run'] is False
        assert memory.update_edge.await_count == 1
        assert memory.add_memory.await_count == 1
        assert memory.refresh_entity_summary.await_count == 1

    @pytest.mark.asyncio
    async def test_the_probe_names_the_operation_being_gated(self, monkeypatch):
        """The refusal has to be attributable in a log, so the operation string
        identifies this script and its mutating mode -- and it is probed once
        for the RUN, not once per project or per edge."""
        calls: list[dict] = []
        monkeypatch.setattr(
            _mod, 'assert_store_mutation_allowed', lambda **kw: calls.append(kw)
        )
        memory = self._make_memory()

        await _mod.run(
            self._args(apply=True),
            memory=memory,
            known_projects_map=self._known_map(),
        )

        assert len(calls) == 1, 'probed ONCE per run, not once per edge'
        assert 'cleanup_count_snapshots' in calls[0]['operation']
        assert '--apply' in calls[0]['operation']

    def test_the_refusal_is_loud_when_driven_through_main(self, monkeypatch, caplog):
        """End-to-end surfacing through the real CLI entry point.

        ``main`` has NO blanket ``except Exception``, so the refusal escapes
        ``asyncio.run`` uncaught -- which is exactly why the guard site's own
        ``logger.error`` must carry the fail-closed diagnosis before the raise.
        Both halves are asserted: the refusal propagates out of ``main`` rather
        than being converted into a 0 exit code, AND the journal an operator
        reads carries the diagnosis.
        """
        self._deny(monkeypatch)
        memory = self._make_memory()
        monkeypatch.setattr(sys, 'argv', ['cleanup_count_snapshots.py', '--apply'])
        real_asyncio_run = asyncio.run

        def _drive(coro, *_a, **_kw):
            coro.close()  # never construct the live MemoryService
            return real_asyncio_run(
                _mod.run(
                    self._args(apply=True),
                    memory=memory,
                    known_projects_map=self._known_map(),
                )
            )

        monkeypatch.setattr(_mod.asyncio, 'run', _drive)

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(_mod.StoreMutationUnavailable),
        ):
            _mod.main()

        assert self._fail_closed_records(caplog), (
            "main has no handler, so the traceback is all an operator gets "
            'unless the guard site logs the fail-closed diagnosis itself; '
            f'got: {[rec.getMessage() for rec in caplog.records]}'
        )
        memory.update_edge.assert_not_awaited()
