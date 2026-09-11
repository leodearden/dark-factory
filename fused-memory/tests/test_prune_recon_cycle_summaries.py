"""Tests for scripts/prune_recon_cycle_summaries.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution — mirrors the pattern in test_cleanup_count_snapshots.py.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'prune_recon_cycle_summaries.py'


def _load_module() -> types.ModuleType:
    """Load prune_recon_cycle_summaries.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators (e.g. @dataclass) work correctly.
    """
    mod_name = 'prune_recon_cycle_summaries'
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


@pytest.fixture(autouse=True)
def _neutralise_store_mutation_preflight(monkeypatch):
    """Keep this MOCK-unit suite independent of the REAL ``~/.mem0``.

    ``run(..., apply=True)`` runs a fail-closed capability preflight before it
    scans (task 4127). That probe touches the real filesystem, so without this
    fixture every ``--apply`` test would pass or fail according to whether the
    machine running pytest happens to be able to write mem0's history
    directory -- and it genuinely cannot inside an agent sandbox, which is the
    whole reason the guard exists. This suite is deliberately MOCK-unit (a
    MagicMock memory service, no live Qdrant), so the environment must not be
    an input to it.

    ``TestRunApplyStoreMutationPreflight`` re-rigs this per test -- to refuse,
    to record, or to pass -- so the guard's own behaviour is still pinned
    explicitly rather than assumed away.

    Deliberately NOT ``raising=False``: if the guard is ever removed from the
    script this fixture must break loudly rather than silently no-op.
    """
    monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', lambda **_kw: None)


# ===========================================================================
# Tests: carries_remediation_history
# ===========================================================================

class TestCarriesRemediationHistory:
    """Tests for carries_remediation_history(content) -> bool.

    Fail-safe contract: deletion is irreversible, so the predicate must default
    to True (preserve) whenever content is empty, unparseable, or ambiguous.
    It only returns False for content that is CLEARLY pure-quiescent
    boilerplate — a quiescent marker present with no remediation signal.
    """

    def _call(self, content: str) -> bool:
        return _mod.carries_remediation_history(content)

    # --- Pure-quiescent boilerplate -> False ---

    def test_quiescent_cycle_marker_alone(self):
        content = 'Cycle summary: quiescent cycle, nothing to report.'
        assert self._call(content) is False

    def test_zero_new_episodes_and_zero_mutations(self):
        content = 'Stage 1 cycle summary: 0 new episodes, 0 mutations this cycle.'
        assert self._call(content) is False

    def test_no_mutations_phrase(self):
        content = 'Reviewed all recent memories; no mutations were necessary.'
        assert self._call(content) is False

    def test_zero_episodes_multi_sentence_boilerplate(self):
        content = (
            'Cycle summary for memory_consolidator run abc123: '
            '0 new episodes processed. 0 mutations applied. '
            'System is stable and quiescent cycle confirmed.'
        )
        assert self._call(content) is False

    def test_case_insensitive_quiescent_marker(self):
        content = 'CYCLE SUMMARY: QUIESCENT CYCLE. 0 MUTATIONS.'
        assert self._call(content) is False

    def test_zero_count_action_fields_are_boilerplate_not_remediation(self):
        """Real Stage 1 quiescent format: zero-valued action FIELDS
        ("memories_deleted: 0") must not trip the 'deleted' keyword."""
        content = (
            'FLAGS EMITTED THIS CYCLE: none\n'
            'flag_ids_this_cycle: []\n'
            'MUTATIONS THIS CYCLE:\n'
            '- memories_added: 0 (excluding this summary)\n'
            '- memories_deleted: 0\n'
            '- edges_updated: 0\n'
            '- graphiti_writes_queued: 0\n'
            '- entity_refresh_failed_uuids: []'
        )
        assert self._call(content) is False

    def test_clean_cycle_stage2_boilerplate(self):
        """Real Stage 2 quiescent format: 'Clean cycle' with zero-valued
        processed-counts is deletable boilerplate."""
        content = (
            'flag_ids_processed: none\n'
            'tasks_created: none\n'
            'stage1_mem0_flags_processed: 0\n'
            'Clean cycle -- Stage 1 emitted 0 flagged items, 0 mem0 '
            'active-query flags, 0 stale escalation flags.'
        )
        assert self._call(content) is False

    def test_no_new_episodes_prose_marker(self):
        content = 'Cycle summary: No new episodes. Nothing else to report.'
        assert self._call(content) is False

    # --- Real remediation -> True ---

    def test_nonzero_action_field_preserved(self):
        """A non-zero action field ("edges_updated: 3") is remediation
        evidence even alongside quiescent zero-fields."""
        content = (
            'MUTATIONS THIS CYCLE:\n'
            '- memories_added: 0\n'
            '- memories_deleted: 0\n'
            '- edges_updated: 3'
        )
        assert self._call(content) is True

    def test_nonzero_paren_count_preserved(self):
        """Parenthesised action counts ("memories_deleted (3):") are
        remediation evidence."""
        content = (
            'flag_ids_emitted: none. No new episodes.\n'
            'MUTATIONS THIS CYCLE:\n'
            '- memories_deleted (3):\n'
            '  - b59149f3 (stale dispatch-readiness claim)'
        )
        assert self._call(content) is True

    def test_remediation_mode_summary_preserved(self):
        content = (
            'stage: memory_consolidator (remediation run). No new episodes.\n'
            'REMEDIATION FINDINGS ADDRESSED (4): ...'
        )
        assert self._call(content) is True

    def test_nonzero_flags_processed_field_preserved(self):
        content = (
            'Clean cycle otherwise. stage1_flags_processed: 2\n'
            'memories_written: 1'
        )
        assert self._call(content) is True

    def test_deleted_entity_mentioned(self):
        content = 'Cycle summary: deleted entity e47ac10b-58cc-4372-a567-0e02b2c3d479 (duplicate).'
        assert self._call(content) is True

    def test_invalidated_edge_mentioned(self):
        content = 'Invalidated edge abc-123 because it contained stale count-snapshot text.'
        assert self._call(content) is True

    def test_flag_processed_mentioned(self):
        content = 'Cycle summary: 0 new episodes. 1 flag processed for task 1942 escalation.'
        assert self._call(content) is True

    def test_merged_memory_mentioned(self):
        content = 'Cycle summary: merged memory records for duplicate entities.'
        assert self._call(content) is True

    def test_nonzero_mutations_count(self):
        content = 'Cycle summary: 3 mutations applied this cycle.'
        assert self._call(content) is True

    def test_nonzero_deletions_count(self):
        content = 'Cycle summary: 2 deletions performed to clean up stale markers.'
        assert self._call(content) is True

    def test_refreshed_entity_mentioned(self):
        content = 'Cycle summary: 0 new episodes, 0 mutations. Refreshed entity summary for e1 as follow-up.'
        assert self._call(content) is True

    def test_edge_correction_mentioned(self):
        content = 'Cycle summary: 0 mutations logged, but one edge correction applied manually.'
        assert self._call(content) is True

    # --- Fail-safe: empty / unparseable / ambiguous -> True ---

    def test_empty_string(self):
        assert self._call('') is True

    def test_whitespace_only(self):
        assert self._call('   \n\t  ') is True

    def test_ambiguous_text_with_no_markers(self):
        content = 'Unrelated free-form note with no cycle-summary markers at all.'
        assert self._call(content) is True

    def test_ambiguous_short_content(self):
        assert self._call('...') is True


# ===========================================================================
# Tests: classify_pool (pure core)
# ===========================================================================

def _summary(id: str, created_at: str | None, content: str = 'ordinary cycle summary') -> dict:
    """Build a normalized {id, created_at, content, metadata} summary dict."""
    return {'id': id, 'created_at': created_at, 'content': content, 'metadata': {}}


class TestClassifyPool:
    """classify_pool(summaries, keep_recent_n) sorts by created_at descending,
    always keeps the most-recent keep_recent_n, and among the rest keeps only
    those carrying remediation history — the rest are marked for deletion."""

    def test_pool_below_keep_recent_n_all_kept_no_deletes(self):
        summaries = [
            _summary('a', '2026-03-01T00:00:00+00:00'),
            _summary('b', '2026-02-01T00:00:00+00:00'),
        ]
        result = _mod.classify_pool(summaries, keep_recent_n=2)

        assert set(result.keep_ids) == {'a', 'b'}
        assert result.delete_ids == []
        assert result.reasons['a'] == 'recent'
        assert result.reasons['b'] == 'recent'

    def test_pool_smaller_than_keep_recent_n_all_kept(self):
        summaries = [_summary('only', '2026-01-01T00:00:00+00:00')]
        result = _mod.classify_pool(summaries, keep_recent_n=2)

        assert result.keep_ids == ['only']
        assert result.delete_ids == []

    def test_older_quiescent_member_deleted(self):
        summaries = [
            _summary('newest1', '2026-05-01T00:00:00+00:00'),
            _summary('newest2', '2026-04-01T00:00:00+00:00'),
            _summary(
                'old-quiescent', '2026-01-01T00:00:00+00:00',
                content='0 new episodes, 0 mutations. Quiescent cycle.',
            ),
        ]
        result = _mod.classify_pool(summaries, keep_recent_n=2)

        assert set(result.keep_ids) == {'newest1', 'newest2'}
        assert result.delete_ids == ['old-quiescent']
        assert result.reasons['old-quiescent'] == 'quiescent_boilerplate'
        assert result.reasons['newest1'] == 'recent'
        assert result.reasons['newest2'] == 'recent'

    def test_older_remediation_bearing_member_preserved(self):
        summaries = [
            _summary('newest1', '2026-05-01T00:00:00+00:00'),
            _summary('newest2', '2026-04-01T00:00:00+00:00'),
            _summary(
                'old-remediation', '2026-01-01T00:00:00+00:00',
                content='deleted entity e47ac10b-58cc-4372-a567-0e02b2c3d479',
            ),
        ]
        result = _mod.classify_pool(summaries, keep_recent_n=2)

        assert 'old-remediation' in result.keep_ids
        assert result.reasons['old-remediation'] == 'remediation'
        assert result.delete_ids == []

    def test_sorts_newest_first_deletes_oldest_beyond_cutoff(self):
        """4 datable members, keep_recent_n=2: the 2 newest are kept as
        'recent'; the 2 oldest (quiescent) are deleted — proves descending
        created_at ordering, not insertion order."""
        summaries = [
            _summary('third', '2026-03-01T00:00:00+00:00', content='0 mutations.'),
            _summary('newest', '2026-04-01T00:00:00+00:00', content='0 mutations.'),
            _summary('oldest', '2026-01-01T00:00:00+00:00', content='0 mutations.'),
            _summary('second', '2026-02-01T00:00:00+00:00', content='0 mutations.'),
        ]
        result = _mod.classify_pool(summaries, keep_recent_n=2)

        assert set(result.keep_ids) == {'newest', 'third'}
        assert set(result.delete_ids) == {'second', 'oldest'}

    def test_missing_created_at_sorts_last_but_eligible_for_remediation_preserve(self):
        summaries = [
            _summary('newest1', '2026-05-01T00:00:00+00:00'),
            _summary('newest2', '2026-04-01T00:00:00+00:00'),
            _summary('no-date-quiescent', None, content='0 new episodes, 0 mutations.'),
            _summary('no-date-remediation', None, content='deleted entity xyz-123'),
        ]
        result = _mod.classify_pool(summaries, keep_recent_n=2)

        assert set(result.keep_ids) == {'newest1', 'newest2', 'no-date-remediation'}
        assert result.delete_ids == ['no-date-quiescent']
        assert result.reasons['no-date-remediation'] == 'remediation'
        assert result.reasons['no-date-quiescent'] == 'quiescent_boilerplate'

    def test_unparseable_created_at_treated_same_as_missing(self):
        summaries = [
            _summary('newest1', '2026-05-01T00:00:00+00:00'),
            _summary('newest2', '2026-04-01T00:00:00+00:00'),
            _summary('bad-date-quiescent', 'not-a-real-date', content='0 mutations, no mutations.'),
        ]
        result = _mod.classify_pool(summaries, keep_recent_n=2)

        assert 'bad-date-quiescent' in result.delete_ids
        assert result.reasons['bad-date-quiescent'] == 'quiescent_boilerplate'

    def test_empty_pool_returns_empty_decision(self):
        result = _mod.classify_pool([], keep_recent_n=2)

        assert result.keep_ids == []
        assert result.delete_ids == []
        assert result.reasons == {}


# ===========================================================================
# Tests: build_prune_report / format_summary_table (pure core)
# ===========================================================================

class TestBuildPruneReport:
    """build_prune_report assembles a structured JSON-serialisable report from
    per-(project, pool) PruneDecisions."""

    def _decisions(self):
        d1 = _mod.classify_pool(
            [
                _summary('p1-new', '2026-05-01T00:00:00+00:00'),
                _summary('p1-old-quiescent', '2026-01-01T00:00:00+00:00', content='0 mutations.'),
                _summary('p1-old-remediation', '2026-01-02T00:00:00+00:00', content='deleted entity x'),
            ],
            keep_recent_n=1,
        )
        d2 = _mod.classify_pool(
            [_summary('p2-only', '2026-03-01T00:00:00+00:00')],
            keep_recent_n=2,
        )
        return {
            ('dark_factory', 'memory_consolidator'): d1,
            ('dark_factory', 'task_knowledge_sync'): d2,
        }

    def test_top_level_keys_present(self):
        report = _mod.build_prune_report(
            self._decisions(), applied_ids=set(), dry_run=True,
            generated_at='2026-07-01T00:00:00+00:00',
        )
        for key in (
            'dry_run', 'generated_at', 'projects', 'pools', 'totals',
            'deletions', 'preserved_remediation',
        ):
            assert key in report, f'missing top-level key {key!r}'

    def test_per_project_per_pool_counts(self):
        report = _mod.build_prune_report(
            self._decisions(), applied_ids=set(), dry_run=True,
            generated_at='2026-07-01T00:00:00+00:00',
        )
        mc = report['projects']['dark_factory']['memory_consolidator']
        assert mc['scanned'] == 3
        assert mc['deletable'] == 1
        assert mc['deleted'] == 0  # dry run
        assert mc['preserved_remediation'] == 1

        tks = report['projects']['dark_factory']['task_knowledge_sync']
        assert tks['scanned'] == 1
        assert tks['deletable'] == 0

    def test_deleted_flags_empty_on_dry_run(self):
        report = _mod.build_prune_report(
            self._decisions(), applied_ids=set(), dry_run=True,
            generated_at='2026-07-01T00:00:00+00:00',
        )
        assert all(d['deleted'] is False for d in report['deletions'])
        assert report['totals']['deleted'] == 0

    def test_deleted_flags_reflect_applied_ids_on_apply(self):
        report = _mod.build_prune_report(
            self._decisions(), applied_ids={'p1-old-quiescent'}, dry_run=False,
            generated_at='2026-07-01T00:00:00+00:00',
        )
        deletion = next(d for d in report['deletions'] if d['id'] == 'p1-old-quiescent')
        assert deletion['deleted'] is True
        assert report['totals']['deleted'] == 1

    def test_deletion_ordering_is_deterministic(self):
        report1 = _mod.build_prune_report(
            self._decisions(), applied_ids=set(), dry_run=True,
            generated_at='2026-07-01T00:00:00+00:00',
        )
        report2 = _mod.build_prune_report(
            self._decisions(), applied_ids=set(), dry_run=True,
            generated_at='2026-07-01T00:00:00+00:00',
        )
        assert report1['deletions'] == report2['deletions']

    def test_pools_lists_distinct_pool_names(self):
        report = _mod.build_prune_report(
            self._decisions(), applied_ids=set(), dry_run=True,
            generated_at='2026-07-01T00:00:00+00:00',
        )
        assert set(report['pools']) == {'memory_consolidator', 'task_knowledge_sync'}

    def test_empty_decisions_produces_zeroed_report(self):
        report = _mod.build_prune_report(
            {}, applied_ids=set(), dry_run=True, generated_at='2026-07-01T00:00:00+00:00',
        )
        assert report['projects'] == {}
        assert report['deletions'] == []
        assert report['totals']['scanned'] == 0


class TestFormatSummaryTable:
    """format_summary_table renders a human-readable per-(project,pool) table."""

    def _report(self, dry_run=True):
        decisions = TestBuildPruneReport()._decisions()
        return _mod.build_prune_report(
            decisions, applied_ids=set(), dry_run=dry_run,
            generated_at='2026-07-01T00:00:00+00:00',
        )

    def test_contains_per_project_pool_row(self):
        table = _mod.format_summary_table(self._report())
        assert 'dark_factory' in table
        assert 'memory_consolidator' in table
        assert 'task_knowledge_sync' in table

    def test_contains_total_row(self):
        table = _mod.format_summary_table(self._report())
        assert 'TOTAL' in table

    def test_dry_run_tag_present_when_dry_run(self):
        table = _mod.format_summary_table(self._report(dry_run=True))
        assert '[DRY RUN]' in table

    def test_dry_run_tag_absent_when_applied(self):
        table = _mod.format_summary_table(self._report(dry_run=False))
        assert '[DRY RUN]' not in table

    def test_returns_a_string(self):
        table = _mod.format_summary_table(self._report())
        assert isinstance(table, str)
        assert len(table) > 0


# ===========================================================================
# Tests: check_scan_completeness (pure helper)
# ===========================================================================

class TestCheckScanCompleteness:
    """check_scan_completeness(per_project_counts) -> list[str] flags every
    project whose scanned cycle-summary count fell short of the ground-truth
    count_by_metadata total -- i.e. the metadata-filtered scroll under-counted
    and the scan cannot be trusted for a prune decision."""

    def test_all_scanned_equals_expected_returns_empty(self):
        per_project_counts = {'p1': (5, 5), 'p2': (0, 0)}
        assert _mod.check_scan_completeness(per_project_counts) == []

    def test_single_undercounting_project_flagged(self):
        per_project_counts = {'p': (3, 8)}
        assert _mod.check_scan_completeness(per_project_counts) == ['p']

    def test_mixed_only_undercounting_ids_returned_sorted(self):
        per_project_counts = {
            'zeta': (2, 10),
            'alpha': (5, 5),
            'beta': (1, 4),
        }
        assert _mod.check_scan_completeness(per_project_counts) == ['beta', 'zeta']

    def test_empty_input_returns_empty(self):
        assert _mod.check_scan_completeness({}) == []

    def test_scanned_greater_than_expected_not_flagged(self):
        """Defensive: scanned > expected (e.g. a new cycle-summary landing
        between the scroll and the count call) is not a truncation -- only a
        strict undercount is flagged."""
        per_project_counts = {'p': (9, 8)}
        assert _mod.check_scan_completeness(per_project_counts) == []


# ===========================================================================
# Tests: apply_prune (async, live shell)
# ===========================================================================

class TestApplyPrune:
    """Async tests for apply_prune(memory, delete_ids, project_id, now).

    Mirrors the best-effort delete posture already proven for
    reconciliation.summary_pool.enforce_summary_pool_cap and
    sweep_orphan_flag_markers.delete_orphan_markers: every id is attempted
    (asyncio.gather with return_exceptions=True); a raising delete is logged
    and excluded from the returned applied set, but does not abort the batch.
    """

    def _now(self) -> datetime:
        return datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)

    def _make_memory(self, side_effect=None) -> MagicMock:
        memory = MagicMock()
        if side_effect is not None:
            memory.delete_memory = AsyncMock(side_effect=side_effect)
        else:
            memory.delete_memory = AsyncMock(return_value={'status': 'deleted'})
        return memory

    @pytest.mark.asyncio
    async def test_calls_delete_memory_once_per_id(self):
        memory = self._make_memory()
        applied = await _mod.apply_prune(memory, ['a', 'b', 'c'], 'dark_factory', self._now())

        assert memory.delete_memory.await_count == 3
        assert applied == {'a', 'b', 'c'}

    @pytest.mark.asyncio
    async def test_delete_memory_called_with_expected_kwargs(self):
        memory = self._make_memory()
        await _mod.apply_prune(memory, ['mem-1'], 'dark_factory', self._now())

        kwargs = memory.delete_memory.call_args.kwargs
        assert kwargs.get('memory_id') == 'mem-1'
        assert kwargs.get('store') == 'mem0'
        assert kwargs.get('project_id') == 'dark_factory'
        assert kwargs.get('_source') == 'prune_recon_cycle_summaries'
        assert kwargs.get('causation_id'), 'causation_id must be a non-empty shared run identifier'

    @pytest.mark.asyncio
    async def test_causation_id_shared_across_ids_in_one_apply_call(self):
        """All deletes issued by a single apply_prune call share one causation_id,
        tying them together as one traceable one-shot prune run."""
        memory = self._make_memory()
        await _mod.apply_prune(memory, ['id-1', 'id-2'], 'dark_factory', self._now())

        causation_ids = {c.kwargs.get('causation_id') for c in memory.delete_memory.call_args_list}
        assert len(causation_ids) == 1

    @pytest.mark.asyncio
    async def test_empty_delete_ids_no_calls(self):
        memory = self._make_memory()
        applied = await _mod.apply_prune(memory, [], 'dark_factory', self._now())

        memory.delete_memory.assert_not_awaited()
        assert applied == set()

    @pytest.mark.asyncio
    async def test_partial_failure_excluded_others_proceed(self):
        async def _side_effect(**kwargs):
            if kwargs.get('memory_id') == 'bad':
                raise RuntimeError('Qdrant error')
            return {'status': 'deleted'}

        memory = MagicMock()
        memory.delete_memory = AsyncMock(side_effect=_side_effect)

        applied = await _mod.apply_prune(
            memory, ['good1', 'bad', 'good2'], 'dark_factory', self._now(),
        )

        assert memory.delete_memory.await_count == 3
        assert applied == {'good1', 'good2'}


# ===========================================================================
# Tests: run (async, end-to-end live shell)
# ===========================================================================

def _cycle_summary_record(
    id: str, stage: str, created_at: str, content: str = 'ordinary cycle summary',
) -> dict:
    """Build a scroll_by_metadata()-shaped record for a cycle_summary pool member.

    Mirrors the Mem0Backend.scroll_by_metadata return shape ``{id, created_at,
    metadata}`` where ``metadata`` is the full Qdrant payload -- the summary
    text lives in ``metadata['data']`` (Mem0's payload content key), not a
    top-level ``memory`` field.
    """
    return {
        'id': id,
        'created_at': created_at,
        'metadata': {'kind': 'cycle_summary', 'stage': stage, 'data': content, 'run_id': 'r1'},
    }


def _unrelated_record(id: str, metadata: dict | None = None) -> dict:
    """Build a scroll_by_metadata()-shaped record that must NOT enter either pool."""
    return {
        'id': id,
        'created_at': '2026-01-01T00:00:00+00:00',
        'metadata': metadata if metadata is not None else {
            'kind': 'observation', 'data': 'unrelated memory content',
        },
    }


class TestRun:
    """Async end-to-end tests for run(args, *, memory, known_projects_map)."""

    def _fixture_records(self) -> list[dict]:
        """4 stage1 + 3 stage2 + 2 unrelated records.

        With keep_recent=2:
          stage1 (memory_consolidator): keeps the 2 newest ('s1-newest',
            's1-second') + 1 older remediation-bearing ('s1-old-remediation');
            deletes 1 older quiescent ('s1-old-quiescent').
          stage2 (task_knowledge_sync): keeps the 2 newest ('s2-newest',
            's2-mid'); deletes 1 older quiescent ('s2-old-quiescent').
          The 2 unrelated records must never appear in either pool.
        """
        return [
            _cycle_summary_record('s1-newest', 'memory_consolidator', '2026-06-01T00:00:00+00:00'),
            _cycle_summary_record('s1-second', 'memory_consolidator', '2026-05-01T00:00:00+00:00'),
            _cycle_summary_record(
                's1-old-remediation', 'memory_consolidator', '2026-02-01T00:00:00+00:00',
                content='deleted entity e47ac10b-58cc-4372-a567-0e02b2c3d479 (duplicate)',
            ),
            _cycle_summary_record(
                's1-old-quiescent', 'memory_consolidator', '2026-01-01T00:00:00+00:00',
                content='0 new episodes, 0 mutations. Quiescent cycle.',
            ),
            _cycle_summary_record('s2-newest', 'task_knowledge_sync', '2026-06-15T00:00:00+00:00'),
            _cycle_summary_record('s2-mid', 'task_knowledge_sync', '2026-04-01T00:00:00+00:00'),
            _cycle_summary_record(
                's2-old-quiescent', 'task_knowledge_sync', '2026-01-01T00:00:00+00:00',
                content='0 mutations. No mutations. Quiescent cycle.',
            ),
            _unrelated_record('u1-wrong-kind'),  # kind != cycle_summary
            _unrelated_record(
                'u2-wrong-stage',
                metadata={'kind': 'cycle_summary', 'stage': 'task_knowledge_analyzer'},
            ),
        ]

    def _make_memory(self, records: list[dict] | None = None) -> MagicMock:
        """Build a mock memory whose mem0 backend exposes the scroll+count
        contract. ``scroll_by_metadata`` returns only the kind=='cycle_summary'
        records (the server-side metadata filter would never surface a
        wrong-kind record like 'u1-wrong-kind'); ``count_by_metadata`` returns
        the exact length of that same list, so the happy-path fixture never
        trips the truncation guard. ``get_all`` is still mocked (as an
        AsyncMock) purely so tests can assert it was never awaited.
        """
        memory = MagicMock()
        mem0 = MagicMock()
        all_records = records if records is not None else []
        scroll_records = [
            r for r in all_records if (r.get('metadata') or {}).get('kind') == 'cycle_summary'
        ]
        mem0.get_all = AsyncMock(return_value={'results': all_records})
        mem0.scroll_by_metadata = AsyncMock(return_value=scroll_records)
        mem0.count_by_metadata = AsyncMock(return_value=len(scroll_records))
        memory.mem0 = mem0
        memory.delete_memory = AsyncMock(return_value={'status': 'deleted'})
        return memory

    def _args(
        self, apply=False, project_id=None, keep_recent=2,
        limit_per_project=1000, yes_i_am_sure=False, scan_limit=10000,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            apply=apply,
            project_id=project_id,
            keep_recent=keep_recent,
            limit_per_project=limit_per_project,
            yes_i_am_sure=yes_i_am_sure,
            scan_limit=scan_limit,
        )

    def _known_map(self, pid='dark_factory') -> dict:
        return {pid: '/some/path'}

    @pytest.mark.asyncio
    async def test_dry_run_no_deletes_and_report_shape(self):
        """Dry-run performs NO delete_memory calls; report classifies both
        pools correctly."""
        memory = self._make_memory(self._fixture_records())
        args = self._args(apply=False, project_id='dark_factory')

        report = await _mod.run(args, memory=memory, known_projects_map=self._known_map())

        memory.delete_memory.assert_not_awaited()
        assert report['dry_run'] is True

        mc = report['projects']['dark_factory']['memory_consolidator']
        assert mc['scanned'] == 4
        assert mc['deletable'] == 1
        assert mc['deleted'] == 0
        assert mc['preserved_remediation'] == 1

        tks = report['projects']['dark_factory']['task_knowledge_sync']
        assert tks['scanned'] == 3
        assert tks['deletable'] == 1
        assert tks['deleted'] == 0

    @pytest.mark.asyncio
    async def test_apply_deletes_exactly_the_classified_delete_set(self):
        """--apply deletes exactly the ids classify_pool marked for deletion —
        no more, no less — across both pools."""
        memory = self._make_memory(self._fixture_records())
        args = self._args(apply=True, project_id='dark_factory')

        report = await _mod.run(args, memory=memory, known_projects_map=self._known_map())

        called_ids = {c.kwargs.get('memory_id') for c in memory.delete_memory.call_args_list}
        assert called_ids == {'s1-old-quiescent', 's2-old-quiescent'}
        assert memory.delete_memory.await_count == 2
        assert report['dry_run'] is False
        assert report['totals']['deleted'] == 2

    @pytest.mark.asyncio
    async def test_unrelated_records_excluded_from_report(self):
        """Records with kind != cycle_summary, or an unrecognised stage, never
        appear in either pool's scanned count or in the deletions list."""
        memory = self._make_memory(self._fixture_records())
        args = self._args(apply=False, project_id='dark_factory')

        report = await _mod.run(args, memory=memory, known_projects_map=self._known_map())

        total_scanned = sum(
            pool_stats['scanned']
            for pools in report['projects'].values()
            for pool_stats in pools.values()
        )
        assert total_scanned == 7  # 4 stage1 + 3 stage2; the 2 unrelated records excluded
        all_deletion_ids = {d['id'] for d in report['deletions']}
        assert 'u1-wrong-kind' not in all_deletion_ids
        assert 'u2-wrong-stage' not in all_deletion_ids

    @pytest.mark.asyncio
    async def test_scan_uses_metadata_filtered_scroll(self):
        """scroll_by_metadata (NOT get_all) is awaited once per selected
        project, scoped to that project_id, filtered to
        {'kind': 'cycle_summary'}, with limit=args.scan_limit. Content-based
        classification still works end-to-end via metadata['data'] -- the
        older remediation-bearing s1 summary is still preserved."""
        memory = self._make_memory(self._fixture_records())
        args = self._args(apply=False, project_id='dark_factory', scan_limit=10000)

        report = await _mod.run(args, memory=memory, known_projects_map=self._known_map())

        memory.mem0.scroll_by_metadata.assert_awaited_once()
        call = memory.mem0.scroll_by_metadata.call_args
        assert call.args[0].project_id == 'dark_factory'
        assert call.args[1] == {'kind': 'cycle_summary'}
        assert call.kwargs.get('limit') == 10000
        memory.mem0.get_all.assert_not_awaited()

        mc = report['projects']['dark_factory']['memory_consolidator']
        assert mc['preserved_remediation'] == 1

    @pytest.mark.asyncio
    async def test_unknown_project_id_aborts_before_get_all(self):
        """An unrecognised --project-id aborts before any enumeration happens."""
        memory = self._make_memory([])
        args = self._args(apply=False, project_id='unknown-id')

        result = await _mod.run(args, memory=memory, known_projects_map={'dark_factory': '/p'})

        assert result.get('aborted') is True
        memory.mem0.scroll_by_metadata.assert_not_awaited()
        memory.mem0.count_by_metadata.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_scan_truncation_aborts(self, capsys):
        """If a project's ground-truth count_by_metadata total exceeds the
        number of records scroll_by_metadata actually returned, the scan
        under-counted -- run() must hard-abort before any deletion (even
        under --apply), flag the project, and point the remedy at
        --scan-limit. scan_limit is set to exactly match the fixture's
        scroll length (8) so the ground truth is consulted despite the
        len(scrolled)<scan_limit short-circuit (see
        test_scan_skips_ground_truth_when_scroll_below_limit) -- the scroll
        mock does not itself honour `limit`, so this pins the ambiguous
        at/over-limit branch that must still call count_by_metadata."""
        memory = self._make_memory(self._fixture_records())
        # Ground truth says there are far more cycle-summaries than the
        # scroll returned -- simulates a scan_limit too small for the pool.
        memory.mem0.count_by_metadata = AsyncMock(return_value=999)
        args = self._args(apply=True, project_id='dark_factory', scan_limit=8)

        result = await _mod.run(args, memory=memory, known_projects_map=self._known_map())

        assert result.get('aborted') is True
        assert result.get('scan_truncated') is True
        assert result.get('truncated_projects') == ['dark_factory']
        assert result.get('possible_undercount_projects') == []
        memory.delete_memory.assert_not_awaited()

        captured = capsys.readouterr()
        assert '--scan-limit' in captured.err
        assert 'benign' in captured.err.lower()

    @pytest.mark.asyncio
    async def test_scan_truncation_flags_possible_undercount_when_scan_returns_zero(self, capsys):
        """scroll_by_metadata returning `[]` is indistinguishable from a
        genuinely-empty pool vs. a benign under-count/race against
        count_by_metadata without cross-checking (a real Qdrant read-timeout
        now raises out of scroll_by_metadata instead of returning `[]`, see
        its docstring, so it can never reach this branch). When the ground
        truth says the pool is NOT actually empty, the truncation guard must
        still fire (the zero-scan case is never short-circuited), and the
        abort must flag this as a probable under-count/race rather than
        pointing solely at --scan-limit, which would not fix a race."""
        memory = self._make_memory([])  # scroll_by_metadata returns []
        memory.mem0.count_by_metadata = AsyncMock(return_value=42)
        args = self._args(apply=True, project_id='dark_factory')

        result = await _mod.run(args, memory=memory, known_projects_map=self._known_map())

        assert result.get('aborted') is True
        assert result.get('scan_truncated') is True
        assert result.get('possible_undercount_projects') == ['dark_factory']
        memory.delete_memory.assert_not_awaited()

        captured = capsys.readouterr()
        assert 'under-count' in captured.err.lower() or 'race' in captured.err.lower()

    @pytest.mark.asyncio
    async def test_scroll_timeout_propagates_uncaught(self):
        """A real Qdrant read-timeout now RAISES out of scroll_by_metadata
        (task 2671 removed the old swallow-into-`[]` behaviour) instead of
        returning an empty list -- this is the invariant the
        possible_undercount_projects rename and the surrounding
        docstrings/NOTE wording above hinge on. run() must let the
        TimeoutError propagate uncaught rather than silently classifying the
        pool as empty ("nothing to prune") or reaching the
        scan_truncated/abort branch at all -- a raised TimeoutError never
        produces a scan result to evaluate in the first place."""
        memory = self._make_memory([])
        memory.mem0.scroll_by_metadata = AsyncMock(
            side_effect=TimeoutError('qdrant scroll timed out'),
        )
        args = self._args(apply=True, project_id='dark_factory')

        with pytest.raises(TimeoutError):
            await _mod.run(args, memory=memory, known_projects_map=self._known_map())

        memory.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_scan_skips_ground_truth_when_scroll_below_limit(self):
        """Efficiency: when the scroll returns a non-empty result strictly
        below --scan-limit, the pool is already provably fully enumerated
        (Qdrant's scroll fills up to `limit` in one pass), so
        count_by_metadata must NOT be consulted -- halves Qdrant reads in the
        common untruncated path."""
        memory = self._make_memory(self._fixture_records())
        args = self._args(apply=False, project_id='dark_factory', scan_limit=10000)

        report = await _mod.run(args, memory=memory, known_projects_map=self._known_map())

        memory.mem0.count_by_metadata.assert_not_awaited()
        assert not report.get('aborted')

    @pytest.mark.asyncio
    async def test_no_truncation_when_scan_complete(self):
        """count_by_metadata == len(scroll) (the default fixture wiring) is
        NOT a truncation -- run() proceeds to a normal report, not an abort.
        Guards against a false-positive abort in the happy path."""
        memory = self._make_memory(self._fixture_records())
        args = self._args(apply=False, project_id='dark_factory')

        report = await _mod.run(args, memory=memory, known_projects_map=self._known_map())

        assert not report.get('aborted')
        assert report['dry_run'] is True
        assert 'totals' in report

    @pytest.mark.asyncio
    async def test_scan_limit_decoupled_from_deletion_cap(self):
        """--scan-limit governs ONLY the scroll window; --limit-per-project
        governs ONLY the deletion safety cap. A large scan_limit alongside a
        small limit_per_project still aborts on the deletion cap (not the
        scan), proving the two knobs are independent."""
        memory = self._make_memory(self._fixture_records())
        args = self._args(
            apply=False, project_id='dark_factory',
            limit_per_project=1, scan_limit=5000,
        )

        result = await _mod.run(args, memory=memory, known_projects_map=self._known_map())

        memory.mem0.scroll_by_metadata.assert_awaited_once()
        call = memory.mem0.scroll_by_metadata.call_args
        assert call.kwargs.get('limit') == 5000

        assert result.get('aborted') is True
        assert result.get('exceeding_projects') == ['dark_factory']
        assert 'scan_truncated' not in result


# ===========================================================================
# Tests: build_parser (argparse) -- --scan-limit / --limit-per-project
# ===========================================================================

class TestArgparse:
    """build_parser() constructs the CLI parser used by main(). --scan-limit
    (the scroll window) and --limit-per-project (the deletion safety cap)
    are independent flags with independent defaults."""

    def test_scan_limit_default(self):
        args = _mod.build_parser().parse_args([])
        assert args.scan_limit == _mod._DEFAULT_SCAN_LIMIT
        assert args.limit_per_project == 1000

    def test_scan_limit_and_cap_are_independent(self):
        args = _mod.build_parser().parse_args(
            ['--scan-limit', '2000', '--limit-per-project', '50'],
        )
        assert args.scan_limit == 2000
        assert args.limit_per_project == 50


# ===========================================================================
# Tests: --apply store-mutation preflight
# ===========================================================================

class TestRunApplyStoreMutationPreflight:
    """``--apply`` refuses to START when this process cannot write mem0's store.

    Ported from ``test_sweep_toolcall_xml_leak.TestRunApplyStoreMutationPreflight``
    (task 3686), which is the in-repo precedent for this contract.

    ``apply_prune`` fans its deletes out through
    ``asyncio.gather(..., return_exceptions=True)`` and tallies per-result
    outcomes rather than aborting, so in a write-denied environment it would
    destroy Qdrant points one by one and report each history-write failure as
    an individual error -- mem0's ``_delete_memory`` removes the point BEFORE
    writing its SQLite history. Only a run-wide probe ahead of the scan bounds
    that.
    """

    def _make_memory(self, records: list[dict] | None = None) -> MagicMock:
        """Mirror of ``TestRun._make_memory``."""
        memory = MagicMock()
        mem0 = MagicMock()
        all_records = records if records is not None else []
        scroll_records = [
            r for r in all_records if (r.get('metadata') or {}).get('kind') == 'cycle_summary'
        ]
        mem0.get_all = AsyncMock(return_value={'results': all_records})
        mem0.scroll_by_metadata = AsyncMock(return_value=scroll_records)
        mem0.count_by_metadata = AsyncMock(return_value=len(scroll_records))
        memory.mem0 = mem0
        memory.delete_memory = AsyncMock(return_value={'status': 'deleted'})
        return memory

    #: Pure-quiescent boilerplate. ``carries_remediation_history`` is fail-safe
    #: (anything ambiguous is PRESERVED), so a deletable fixture must be
    #: unambiguously quiescent or nothing is ever classified for deletion and
    #: the ``--apply`` assertions become vacuous.
    _QUIESCENT = 'Cycle summary: quiescent cycle. 0 new episodes, 0 mutations.'

    def _records(self) -> list[dict]:
        """Three quiescent summaries in one pool: with keep_recent=2 exactly
        one (the oldest) is deletable, so ``--apply`` has real work to refuse."""
        return [
            _cycle_summary_record(
                'm1', 'memory_consolidator', '2026-01-01T00:00:00+00:00', self._QUIESCENT,
            ),
            _cycle_summary_record(
                'm2', 'memory_consolidator', '2026-01-02T00:00:00+00:00', self._QUIESCENT,
            ),
            _cycle_summary_record(
                'm3', 'memory_consolidator', '2026-01-03T00:00:00+00:00', self._QUIESCENT,
            ),
        ]

    def _args(
        self, apply=True, project_id: str | None = 'dark_factory', keep_recent=2,
        limit_per_project=1000, yes_i_am_sure=False, scan_limit=10000,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            apply=apply,
            project_id=project_id,
            keep_recent=keep_recent,
            limit_per_project=limit_per_project,
            yes_i_am_sure=yes_i_am_sure,
            scan_limit=scan_limit,
        )

    def _known_map(self, pid='dark_factory') -> dict:
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

        ``main`` has no handler at all here -- the refusal exits the
        interpreter as an uncaught traceback -- so this ERROR record is the
        ONLY place the operator is told what was refused and what to do
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
            if rec.name == 'prune_recon_cycle_summaries'
            and rec.levelname == 'ERROR'
            and 'NOT started (fail-closed)' in rec.getMessage()
            and 'MCP server' in rec.getMessage()
        ]

    @pytest.mark.asyncio
    async def test_apply_performs_zero_mutations_when_the_store_is_unwritable(
        self, monkeypatch
    ):
        """The whole point: refuse to start rather than half-complete."""
        self._deny(monkeypatch)
        memory = self._make_memory(self._records())

        with pytest.raises(
            _mod.StoreMutationUnavailable, match='SENTINEL-store-unwritable'
        ):
            await _mod.run(
                self._args(apply=True), memory=memory,
                known_projects_map=self._known_map(),
            )

        memory.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_guard_sits_before_the_per_project_scan(self, monkeypatch):
        """It aborts without scanning. The scan loop issues one
        ``scroll_by_metadata`` plus a conditional ``count_by_metadata`` PER
        PROJECT, and ``--project-id`` is optional, so the wasted work would
        otherwise scale with the size of the project registry."""
        self._deny(monkeypatch)
        memory = self._make_memory(self._records())

        with pytest.raises(_mod.StoreMutationUnavailable):
            await _mod.run(
                self._args(apply=True, project_id=None), memory=memory,
                known_projects_map={'dark_factory': '/a', 'reify': '/b', 'knowlive': '/c'},
            )

        memory.mem0.scroll_by_metadata.assert_not_awaited()
        memory.mem0.count_by_metadata.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_dry_run_is_never_gated_on_write_capability(self, monkeypatch):
        """A read-only run mutates nothing, so it must not require the ability
        to mutate -- the prune report stays obtainable from anywhere."""
        self._deny(monkeypatch)
        memory = self._make_memory(self._records())

        report = await _mod.run(
            self._args(apply=False), memory=memory,
            known_projects_map=self._known_map(),
        )

        assert report['dry_run'] is True
        assert report['projects']['dark_factory']['memory_consolidator']['deletable'] == 1
        assert report['projects']['dark_factory']['memory_consolidator']['deleted'] == 0
        memory.mem0.scroll_by_metadata.assert_awaited()
        memory.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_apply_is_unchanged_when_the_preflight_passes(self, monkeypatch):
        """Happy path: a writable environment prunes exactly as before."""
        monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', lambda **_kw: None)
        memory = self._make_memory(self._records())

        report = await _mod.run(
            self._args(apply=True), memory=memory,
            known_projects_map=self._known_map(),
        )

        memory.delete_memory.assert_awaited_once()
        assert report['dry_run'] is False
        assert report['projects']['dark_factory']['memory_consolidator']['deleted'] == 1

    @pytest.mark.asyncio
    async def test_the_probe_names_the_operation_being_gated(self, monkeypatch):
        """The refusal has to be attributable in a log, so the operation string
        identifies this script and its mutating mode."""
        calls: list[dict] = []
        monkeypatch.setattr(
            _mod, 'assert_store_mutation_allowed', lambda **kw: calls.append(kw)
        )
        memory = self._make_memory(self._records())

        await _mod.run(
            self._args(apply=True, project_id=None), memory=memory,
            known_projects_map={'dark_factory': '/a', 'reify': '/b'},
        )

        assert len(calls) == 1, 'probed ONCE per run, not once per project or record'
        assert 'prune_recon_cycle_summaries' in calls[0]['operation']
        assert '--apply' in calls[0]['operation']

    @pytest.mark.asyncio
    async def test_the_refusal_is_not_masked_by_the_scan_truncation_abort(
        self, monkeypatch
    ):
        """``check_scan_completeness``'s hard abort returns a report-shaped
        payload through the NORMAL return path. If the preflight sat downstream
        of it, a write-denied ``--apply`` could come back as an ordinary handled
        outcome instead of a refusal. Pin that it RAISES.

        Rigged so the scan WOULD abort: an empty scroll against a non-zero
        ground-truth count is the under-count case.
        """
        self._deny(monkeypatch)
        memory = self._make_memory([])
        memory.mem0.count_by_metadata = AsyncMock(return_value=7)

        with pytest.raises(_mod.StoreMutationUnavailable):
            await _mod.run(
                self._args(apply=True), memory=memory,
                known_projects_map=self._known_map(),
            )

        memory.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_refusal_is_not_masked_by_the_limit_cap_abort(
        self, monkeypatch
    ):
        """``check_limit_cap``'s hard abort likewise returns a report-shaped
        payload through the normal return path. Rigged so the cap WOULD abort
        (``--limit-per-project 0`` against a non-zero deletable count), a
        denied ``--apply`` must still raise rather than return that payload.

        Note ``--yes-i-am-sure`` is NOT a dry-run switch -- it only overrides
        this cap -- so it must not affect the preflight either way.
        """
        self._deny(monkeypatch)
        memory = self._make_memory(self._records())

        with pytest.raises(_mod.StoreMutationUnavailable):
            await _mod.run(
                self._args(apply=True, limit_per_project=0), memory=memory,
                known_projects_map=self._known_map(),
            )

        memory.delete_memory.assert_not_awaited()

    def test_the_refusal_escapes_main_and_is_never_a_report_shaped_success(
        self, monkeypatch, caplog
    ):
        """``main`` calls ``asyncio.run`` bare -- no try/except -- and its only
        other outcomes are ``0``/``1`` derived from ``report['aborted']``. So
        the refusal must PROPAGATE all the way out and exit the interpreter
        non-zero via an uncaught traceback, never collapsing into either of
        those report-shaped returns.

        Driven through ``main`` (the ``purge``/``sweep`` suites' ``_drive``
        idiom) precisely because that is where the claim lives: asserting only
        that ``run`` raises leaves the swallowed-on-the-way-out case -- the one
        that would read as a successful prune -- untested.
        """
        self._deny(monkeypatch)
        memory = self._make_memory(self._records())
        monkeypatch.setattr(
            sys, 'argv', ['prune_recon_cycle_summaries.py', '--apply'],
        )
        real_asyncio_run = asyncio.run

        def _drive(coro, *_a, **_kw):
            coro.close()  # never construct the live MemoryService
            return real_asyncio_run(
                _mod.run(
                    self._args(apply=True), memory=memory,
                    known_projects_map=self._known_map(),
                )
            )

        monkeypatch.setattr(_mod.asyncio, 'run', _drive)

        with caplog.at_level('ERROR'), pytest.raises(
            _mod.StoreMutationUnavailable, match='SENTINEL-store-unwritable'
        ):
            _mod.main()

        assert self._fail_closed_records(caplog), (
            'nothing else explains this traceback -- the guard site must log '
            'the fail-closed diagnosis before raising; got: '
            f'{[rec.getMessage() for rec in caplog.records]}'
        )
        memory.delete_memory.assert_not_awaited()
