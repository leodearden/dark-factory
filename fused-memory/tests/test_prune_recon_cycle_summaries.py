"""Tests for scripts/prune_recon_cycle_summaries.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution — mirrors the pattern in test_cleanup_count_snapshots.py.
"""
from __future__ import annotations

import argparse
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

    # --- Real remediation -> True ---

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
    """Build a mem0.get_all()-shaped record for a cycle_summary pool member."""
    return {
        'id': id,
        'memory': content,
        'created_at': created_at,
        'metadata': {'kind': 'cycle_summary', 'stage': stage, 'run_id': 'r1'},
    }


def _unrelated_record(id: str, metadata: dict | None = None) -> dict:
    """Build a mem0.get_all()-shaped record that must NOT enter either pool."""
    return {
        'id': id,
        'memory': 'unrelated memory content',
        'created_at': '2026-01-01T00:00:00+00:00',
        'metadata': metadata if metadata is not None else {'kind': 'observation'},
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
        memory = MagicMock()
        mem0 = MagicMock()
        mem0.get_all = AsyncMock(return_value={'results': records if records is not None else []})
        memory.mem0 = mem0
        memory.delete_memory = AsyncMock(return_value={'status': 'deleted'})
        return memory

    def _args(
        self, apply=False, project_id=None, keep_recent=2,
        limit_per_project=1000, yes_i_am_sure=False,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            apply=apply,
            project_id=project_id,
            keep_recent=keep_recent,
            limit_per_project=limit_per_project,
            yes_i_am_sure=yes_i_am_sure,
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
    async def test_get_all_scoped_per_project_with_configured_limit(self):
        """mem0.get_all is awaited once per selected project, scoped to that
        project_id, with limit=args.limit_per_project (the scan/safety-cap value)."""
        memory = self._make_memory(self._fixture_records())
        args = self._args(apply=False, project_id='dark_factory', limit_per_project=500)

        await _mod.run(args, memory=memory, known_projects_map=self._known_map())

        memory.mem0.get_all.assert_awaited_once()
        call = memory.mem0.get_all.call_args
        scope_arg = call.args[0] if call.args else call.kwargs.get('scope')
        assert scope_arg.project_id == 'dark_factory'
        assert call.kwargs.get('limit') == 500

    @pytest.mark.asyncio
    async def test_unknown_project_id_aborts_before_get_all(self):
        """An unrecognised --project-id aborts before any enumeration happens."""
        memory = self._make_memory([])
        args = self._args(apply=False, project_id='unknown-id')

        result = await _mod.run(args, memory=memory, known_projects_map={'dark_factory': '/p'})

        assert result.get('aborted') is True
        memory.mem0.get_all.assert_not_awaited()
