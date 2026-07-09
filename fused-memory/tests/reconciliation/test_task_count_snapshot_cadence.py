"""Tests for task_count_snapshot_cadence — task 2278.

Hardens the Mem0 ``task_count_snapshot`` write cadence written by Stage 2
(task_knowledge_sync) as its final action each cycle: a Stage-2 freshness
stat plus a harness consecutive-full-cycle-miss escalation.

Covers (grown step-by-step per plan.json):
- TestConstants                        (step-1/2, step-3/4)
- TestExtractSnapshotWritten            (step-1/2)
- TestComputeSnapshotMissStreak         (step-1/2)
- TestEvaluateSnapshotCadence           (step-3/4)
- TestBuildStaleSnapshotFinding         (step-3/4)
- TestVerifyTaskCountSnapshotWritten    (step-5/6)
- TestRunRecordsTaskCountSnapshotWrittenStat (step-7/8)

Task 2325 follow-up (exempt never-snapshotted projects + make the Stage-2
task_count_snapshot write structural):
- TestBuildTaskCountSnapshotContent     (task 2325 step-3/4)
- TestWriteTaskCountSnapshot            (task 2325 step-5/6)
- TestRunDeterministicSnapshotWrite     (task 2325 step-7/8)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import (
    ReconciliationRun,
    RunType,
    StageId,
    StageReport,
    Watermark,
)
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    TaskKnowledgeSync,
    _verify_task_count_snapshot_written,
    _write_task_count_snapshot,
)
from fused_memory.reconciliation.task_count_snapshot_cadence import (
    ESCALATION_CATEGORY,
    SNAPSHOT_WRITTEN_STAT_KEY,
    TASK_COUNT_SNAPSHOT_CATEGORY,
    TASK_COUNT_SNAPSHOT_KIND,
    TASK_COUNT_SNAPSHOT_MISS_THRESHOLD,
    build_stale_snapshot_finding,
    build_task_count_snapshot_content,
    compute_snapshot_miss_streak,
    evaluate_snapshot_cadence,
    extract_snapshot_written,
)


def _scope(project_id: str, project_root: str) -> ProjectScope:
    """Build a ProjectScope from raw strings — DRYs the many test call sites."""
    return ProjectScope(ProjectId(project_id), ProjectRoot(project_root))


def _stage_report(stats: dict) -> StageReport:
    """Build a minimal real StageReport carrying *stats*."""
    now = datetime.now(UTC)
    return StageReport(
        stage=StageId.task_knowledge_sync,
        started_at=now,
        completed_at=now,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Assert module-level constants have expected values."""

    def test_kind_value(self):
        assert TASK_COUNT_SNAPSHOT_KIND == 'task_count_snapshot'

    def test_stat_key_value(self):
        assert SNAPSHOT_WRITTEN_STAT_KEY == 'task_count_snapshot_written'

    def test_miss_threshold_value(self):
        assert TASK_COUNT_SNAPSHOT_MISS_THRESHOLD == 2

    def test_escalation_category_value(self):
        assert ESCALATION_CATEGORY == 'recon_stale_task_count_snapshot'


# ---------------------------------------------------------------------------
# extract_snapshot_written
# ---------------------------------------------------------------------------


class TestExtractSnapshotWritten:
    """extract_snapshot_written(stage_report) -> bool | None.

    Must handle a real StageReport AND a raw dict shape identically (mirrors
    the isinstance(x, dict) guard convention used elsewhere in reconciliation,
    e.g. journal.get_run's stage_reports reconstruction).
    """

    def test_stats_1_is_true_on_stage_report(self):
        report = _stage_report({'task_count_snapshot_written': 1})
        assert extract_snapshot_written(report) is True

    def test_stats_0_is_false_on_stage_report(self):
        report = _stage_report({'task_count_snapshot_written': 0})
        assert extract_snapshot_written(report) is False

    def test_missing_key_is_none_on_stage_report(self):
        report = _stage_report({})
        assert extract_snapshot_written(report) is None

    def test_stats_1_is_true_on_raw_dict(self):
        report = {'stats': {'task_count_snapshot_written': 1}}
        assert extract_snapshot_written(report) is True

    def test_stats_0_is_false_on_raw_dict(self):
        report = {'stats': {'task_count_snapshot_written': 0}}
        assert extract_snapshot_written(report) is False

    def test_missing_key_is_none_on_raw_dict(self):
        report = {'stats': {}}
        assert extract_snapshot_written(report) is None

    def test_missing_stats_key_is_none_on_raw_dict(self):
        assert extract_snapshot_written({}) is None

    def test_none_report_is_none(self):
        assert extract_snapshot_written(None) is None


# ---------------------------------------------------------------------------
# compute_snapshot_miss_streak
# ---------------------------------------------------------------------------


class TestComputeSnapshotMissStreak:
    """compute_snapshot_miss_streak(recent_flags) -> int.

    recent_flags is most-recent-first list of bool|None; counts the leading
    run of consecutive False, stopping at the first True (a written cycle
    resets the streak) or None (unknown -> stop, fail-safe).
    """

    @pytest.mark.parametrize(
        ('recent_flags', 'expected'),
        [
            ([], 0),
            ([True], 0),
            ([False], 1),
            ([False, False], 2),
            ([False, True, False], 1),
            ([False, None, False], 1),
        ],
    )
    def test_streak(self, recent_flags, expected):
        assert compute_snapshot_miss_streak(recent_flags) == expected


# ---------------------------------------------------------------------------
# evaluate_snapshot_cadence
# ---------------------------------------------------------------------------


class TestEvaluateSnapshotCadence:
    """evaluate_snapshot_cadence(current_written, prior_flags, *, blocked, threshold) -> dict.

    Returns {'streak': int, 'escalate': bool}.
    """

    def test_current_written_true_never_escalates_regardless_of_priors(self):
        result = evaluate_snapshot_cadence(True, [False, False, False], blocked=False)
        assert result['escalate'] is False

    def test_current_written_none_never_escalates(self):
        """Unknown current cycle -> never escalate (fail-safe)."""
        result = evaluate_snapshot_cadence(None, [False, False, False], blocked=False)
        assert result['escalate'] is False

    def test_blocked_project_never_escalates_even_with_long_streak(self):
        result = evaluate_snapshot_cadence(False, [False] * 5, blocked=True)
        assert result['escalate'] is False

    def test_current_false_empty_priors_streak_1_below_threshold(self):
        result = evaluate_snapshot_cadence(False, [], blocked=False)
        assert result == {'streak': 1, 'escalate': False}

    def test_current_false_one_prior_miss_streak_2_meets_threshold(self):
        result = evaluate_snapshot_cadence(False, [False], blocked=False)
        assert result == {'streak': 2, 'escalate': True}

    def test_current_false_prior_write_resets_streak_to_1(self):
        """A prior successful write resets the streak; current miss alone is 1."""
        result = evaluate_snapshot_cadence(False, [True], blocked=False)
        assert result == {'streak': 1, 'escalate': False}


# ---------------------------------------------------------------------------
# build_stale_snapshot_finding
# ---------------------------------------------------------------------------


class TestBuildStaleSnapshotFinding:
    """build_stale_snapshot_finding(project_id) -> dict.

    Stable identity ({category, affected_ids, description}) so _escalate's
    content-fingerprint dedup folds repeats into a single pending escalation
    (mirrors _DEAD_OWNER_STORM_FINDING).
    """

    def test_category_is_escalation_category(self):
        finding = build_stale_snapshot_finding('reify')
        assert finding['category'] == 'recon_stale_task_count_snapshot'

    def test_affected_ids_scoped_to_project(self):
        finding = build_stale_snapshot_finding('reify')
        assert finding['affected_ids'] == ['task_count_snapshot:reify']

    def test_description_is_stable_and_non_empty(self):
        first = build_stale_snapshot_finding('reify')
        second = build_stale_snapshot_finding('reify')
        assert first['description']
        assert first['description'] == second['description']


# ---------------------------------------------------------------------------
# build_task_count_snapshot_content (task 2325 step-3/4)
# ---------------------------------------------------------------------------


class TestBuildTaskCountSnapshotContent:
    """build_task_count_snapshot_content(...) -> str (task 2325).

    Pure, dependency-free renderer for the deterministic task_count_snapshot
    Mem0 write content -- no I/O, no mocks needed. Mirrors the module's
    "pure compute helper" contract (see module docstring).
    """

    def test_category_constant_value(self):
        assert TASK_COUNT_SNAPSHOT_CATEGORY == 'observations_and_summaries'

    def test_content_contains_project_counts_highest_id_and_date(self):
        content = build_task_count_snapshot_content(
            'reify',
            total=42, done=18, cancelled=3, active=20, other=1,
            highest_task_id=99, as_of='2026-07-08',
        )

        assert isinstance(content, str)
        assert 'reify' in content
        assert '2026-07-08' in content
        # Labeled substrings (not bare digits) so the assertion actually pins
        # each count to its own label -- a swap of e.g. done/cancelled in the
        # renderer, or an incidental digit match (e.g. '1' inside '18'),
        # would fail here (reviewer finding, amendment round).
        assert '42 total' in content
        assert '18 done' in content
        assert '3 cancelled' in content
        assert '20 active' in content
        assert '1 other' in content
        assert 'highest task id 99' in content

    def test_content_without_as_of_is_non_empty_and_contains_counts(self):
        content = build_task_count_snapshot_content(
            'reify',
            total=42, done=18, cancelled=3, active=20, other=1,
            highest_task_id=99, as_of=None,
        )

        assert isinstance(content, str)
        assert content
        assert 'reify' in content
        assert '42' in content
        assert '99' in content


# ---------------------------------------------------------------------------
# _verify_task_count_snapshot_written (stages/task_knowledge_sync.py)
# ---------------------------------------------------------------------------


class TestVerifyTaskCountSnapshotWritten:
    """_verify_task_count_snapshot_written(memory_service, project_id, run_window_start).

    Best-effort freshness check, mirrors _verify_stage2_summary_written's
    never-raises / None-on-transient contract.
    """

    @pytest.mark.asyncio
    async def test_record_within_window_returns_true_and_uses_kind_filter(self):
        window_start = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {
                'id': 'm1',
                'created_at': (window_start + timedelta(seconds=5)).isoformat(),
                'metadata': {'kind': 'task_count_snapshot'},
            },
        ]

        result = await _verify_task_count_snapshot_written(
            memory_service, 'reify', window_start,
        )

        assert result is True
        memory_service.get_memories_by_metadata.assert_awaited_once()
        _, kwargs = memory_service.get_memories_by_metadata.await_args
        assert kwargs['filters'] == {'kind': 'task_count_snapshot'}

    @pytest.mark.asyncio
    async def test_only_records_before_window_returns_false(self):
        window_start = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {
                'id': 'm-old',
                'created_at': (window_start - timedelta(hours=1)).isoformat(),
                'metadata': {'kind': 'task_count_snapshot'},
            },
        ]

        result = await _verify_task_count_snapshot_written(
            memory_service, 'reify', window_start,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_no_records_returns_false(self):
        window_start = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = []

        result = await _verify_task_count_snapshot_written(
            memory_service, 'reify', window_start,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_none_run_window_start_returns_none_without_querying(self):
        memory_service = AsyncMock()

        result = await _verify_task_count_snapshot_written(memory_service, 'reify', None)

        assert result is None
        memory_service.get_memories_by_metadata.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_query_failure_returns_none_not_false(self):
        window_start = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.side_effect = RuntimeError('mem0 down')

        result = await _verify_task_count_snapshot_written(
            memory_service, 'reify', window_start,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_default_scroll_limit_is_1000(self):
        """Mirrors the sibling GC helpers' (_sweep_stale_flag_markers etc.)
        default scroll_limit of 1000 (reviewer finding, amendment round)."""
        window_start = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = []

        await _verify_task_count_snapshot_written(memory_service, 'reify', window_start)

        _, kwargs = memory_service.get_memories_by_metadata.await_args
        assert kwargs['limit'] == 1000

    @pytest.mark.asyncio
    async def test_explicit_scroll_limit_is_passed_through(self):
        window_start = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = []

        await _verify_task_count_snapshot_written(
            memory_service, 'reify', window_start, scroll_limit=250,
        )

        _, kwargs = memory_service.get_memories_by_metadata.await_args
        assert kwargs['limit'] == 250

    @pytest.mark.asyncio
    async def test_unsaturated_page_with_no_match_still_returns_false(self):
        """A page below scroll_limit is exhaustive -- a real confirmed miss,
        not an artifact of truncation."""
        window_start = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)
        memory_service = AsyncMock()
        old_record = {
            'id': 'm-old',
            'created_at': (window_start - timedelta(hours=1)).isoformat(),
            'metadata': {'kind': 'task_count_snapshot'},
        }
        memory_service.get_memories_by_metadata.return_value = [old_record] * 2

        result = await _verify_task_count_snapshot_written(
            memory_service, 'reify', window_start, scroll_limit=3,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_saturated_page_with_no_match_returns_none_not_false(self):
        """Reviewer finding (amendment round, robustness): a saturated scroll
        page can't confirm absence. Qdrant's scroll orders by point id, not
        created_at, and task_count_snapshot has no GC/pool-cap, so once
        matches exceed scroll_limit the freshest record can be excluded from
        the returned page. Must return None (unknown), never a confirmed
        miss -- else a truncated page eventually fires a false stale-snapshot
        escalation despite a fresh snapshot existing."""
        window_start = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)
        memory_service = AsyncMock()
        old_record = {
            'id': 'm-old',
            'created_at': (window_start - timedelta(hours=1)).isoformat(),
            'metadata': {'kind': 'task_count_snapshot'},
        }
        memory_service.get_memories_by_metadata.return_value = [old_record] * 3

        result = await _verify_task_count_snapshot_written(
            memory_service, 'reify', window_start, scroll_limit=3,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_saturated_page_with_match_still_returns_true(self):
        """A match found even in a saturated/possibly-truncated page is still
        real evidence of a fresh write -- truncation only compromises a
        miss, never a hit."""
        window_start = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)
        memory_service = AsyncMock()
        fresh_record = {
            'id': 'm-fresh',
            'created_at': (window_start + timedelta(seconds=5)).isoformat(),
            'metadata': {'kind': 'task_count_snapshot'},
        }
        old_record = {
            'id': 'm-old',
            'created_at': (window_start - timedelta(hours=1)).isoformat(),
            'metadata': {'kind': 'task_count_snapshot'},
        }
        memory_service.get_memories_by_metadata.return_value = [old_record, fresh_record]

        result = await _verify_task_count_snapshot_written(
            memory_service, 'reify', window_start, scroll_limit=2,
        )

        assert result is True


# ---------------------------------------------------------------------------
# _write_task_count_snapshot (stages/task_knowledge_sync.py) -- task 2325 step-5/6
# ---------------------------------------------------------------------------


class TestWriteTaskCountSnapshot:
    """_write_task_count_snapshot(memory_service, taskmaster, project_root,
    project_id, run_id, run_window_start) -> bool | None.

    Deterministic (non-LLM) write of the task_count_snapshot Mem0 record.
    Best-effort: never raises, mirrors the module's other I/O helpers'
    never-raises contract. Returns True (wrote) or None (skipped/failed),
    never False -- a failed best-effort write must stay "inconclusive" so
    callers fall back to the freshness read instead of recording a
    confirmed miss.
    """

    def _taskmaster(self):
        taskmaster = AsyncMock()
        taskmaster.get_tasks.return_value = {
            'tasks': [
                {'id': 1, 'status': 'pending'},
                {'id': 2, 'status': 'in-progress'},
                {'id': 3, 'status': 'done'},
                {'id': 4, 'status': 'cancelled'},
            ],
        }
        return taskmaster

    @pytest.mark.asyncio
    async def test_success_writes_once_and_returns_true(self):
        memory_service = AsyncMock()
        memory_service.add_memory.return_value = {'memory_ids': ['m1']}
        taskmaster = self._taskmaster()

        result = await _write_task_count_snapshot(
            memory_service, taskmaster, '/tmp/test', 'reify', 'run-1', None,
        )

        assert result is True
        memory_service.add_memory.assert_awaited_once()
        _, kwargs = memory_service.add_memory.await_args
        assert kwargs['category'] == 'observations_and_summaries'
        assert kwargs['metadata']['kind'] == 'task_count_snapshot'
        assert kwargs['project_id'] == 'reify'
        assert kwargs['causation_id'] == 'run-1'
        assert isinstance(kwargs['content'], str)
        assert kwargs['content']
        assert 'reify' in kwargs['content']
        # Reviewer finding (amendment round, test_coverage): pin the
        # fetch->filter_task_tree->render wiring end-to-end, not just the
        # pure renderer (TestBuildTaskCountSnapshotContent already covers
        # label mapping in isolation). The 4-task fixture above is
        # 1 pending + 1 in-progress (both ACTIVE) + 1 done + 1 cancelled, so
        # total=4, done=1, cancelled=1, active=2 -- a regression that swapped
        # tree fields (e.g. passed done_count where active is expected) would
        # fail one of these without failing the pure-renderer test.
        assert '4 total' in kwargs['content']
        assert '1 done' in kwargs['content']
        assert '1 cancelled' in kwargs['content']
        assert '2 active' in kwargs['content']

    @pytest.mark.asyncio
    async def test_add_memory_failure_returns_none_not_raise(self):
        memory_service = AsyncMock()
        memory_service.add_memory.side_effect = RuntimeError('mem0 down')
        taskmaster = self._taskmaster()

        result = await _write_task_count_snapshot(
            memory_service, taskmaster, '/tmp/test', 'reify', 'run-1', None,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_no_taskmaster_returns_none_without_writing(self):
        memory_service = AsyncMock()

        result = await _write_task_count_snapshot(
            memory_service, None, '/tmp/test', 'reify', 'run-1', None,
        )

        assert result is None
        memory_service.add_memory.assert_not_awaited()


# ---------------------------------------------------------------------------
# TaskKnowledgeSync.run() wiring — report.stats['task_count_snapshot_written']
# ---------------------------------------------------------------------------


class TestRunRecordsTaskCountSnapshotWrittenStat:
    """TaskKnowledgeSync.run() records the freshness stat (step-7/8).

    Mirrors tests/test_stages.py's TestRunStage2SummaryReconstructionWiring
    harness: super().run() executes for real via a patched run_stage_via_cli
    while the module-level helper is patched directly, so these tests target
    only the run()-level wiring — not _verify_task_count_snapshot_written's
    own internals (covered by TestVerifyTaskCountSnapshotWritten above).
    """

    @pytest.fixture
    def mock_deps(self):
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        memory_service = AsyncMock()
        # count==1 short-circuits the unrelated stage2-summary verify/repair/
        # reconstruct chain so these tests stay isolated to the new stat.
        memory_service.count_memories_by_metadata.return_value = 1
        memory_service.delete_memory = AsyncMock(return_value=None)
        memory_service.search.return_value = []
        memory_service.add_memory.return_value = {'memory_ids': []}
        taskmaster = AsyncMock()
        taskmaster.get_tasks.return_value = {'tasks': []}
        return {
            'memory_service': memory_service,
            'taskmaster': taskmaster,
            'journal': AsyncMock(),
            'config': config,
        }

    def _fake_cli_result(self):
        return MagicMock(
            success=True,
            report={'flagged_items': [], 'summary': 'ok', 'stats': {}},
            llm_calls=1, tokens_used=0, cost_usd=0.0,
            model='test-model', error=None,
        )

    async def _run_with_snapshot_check(self, mock_deps, snapshot_result, run_id):
        stage = TaskKnowledgeSync(
            StageId.task_knowledge_sync,
            scope=_scope('dark_factory', '/tmp/test'),
            **mock_deps,
        )

        with (
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=self._fake_cli_result()),
            ),
            patch(
                'fused_memory.reconciliation.stages.task_knowledge_sync'
                '._verify_task_count_snapshot_written',
                new=AsyncMock(return_value=snapshot_result),
            ),
        ):
            return await stage.run(
                events=[], watermark=Watermark(project_id='dark_factory'),
                prior_reports=[], run_id=run_id,
            )

    @pytest.mark.asyncio
    async def test_helper_true_sets_stat_to_1(self, mock_deps):
        report = await self._run_with_snapshot_check(mock_deps, True, 'run-snap-true')
        assert report.stats['task_count_snapshot_written'] == 1

    @pytest.mark.asyncio
    async def test_helper_false_sets_stat_to_0(self, mock_deps):
        report = await self._run_with_snapshot_check(mock_deps, False, 'run-snap-false')
        assert report.stats['task_count_snapshot_written'] == 0

    @pytest.mark.asyncio
    async def test_helper_none_omits_stat_key(self, mock_deps):
        report = await self._run_with_snapshot_check(mock_deps, None, 'run-snap-none')
        assert 'task_count_snapshot_written' not in report.stats

    @pytest.mark.asyncio
    async def test_run_window_start_is_forwarded_to_verify_helper(self, mock_deps):
        """Wiring regression guard (reviewer finding, amendment round).

        Unlike the three tests above, this one does NOT patch
        _verify_task_count_snapshot_written -- it lets run() forward its
        real ``getattr(self, '_run_window_start', None)`` into the real
        helper, so a broken window handoff (e.g. a future edit that always
        passes None) is caught here. The direct-patch tests above would
        keep passing unchanged even under such a regression, since they
        never exercise the forwarding itself.

        journal.get_run is configured to return a real ReconciliationRun
        with a tz-aware started_at, which assemble_payload's run-window
        guard reads via ``journal.get_run(...).started_at`` to compute
        run_window_start and stash it on self._run_window_start before
        run() forwards it to the freshness helper post-flight.
        """
        window_start = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)
        mock_deps['journal'].get_run = AsyncMock(return_value=ReconciliationRun(
            id='run-window-fwd',
            project_id='dark_factory',
            run_type=RunType.full,
            trigger_reason='test',
            started_at=window_start,
        ))

        def _get_memories_by_metadata(*, project_id, filters, **kwargs):
            # Only the task_count_snapshot query gets the canned fresh
            # record; every other filter shape queried elsewhere in run()
            # (e.g. the unconditional stage1_flag_marker sweeps) sees an
            # empty pool, keeping this test isolated to the call path
            # under test.
            if filters == {'kind': TASK_COUNT_SNAPSHOT_KIND}:
                return [{
                    'id': 'm-fresh',
                    'created_at': (window_start + timedelta(seconds=5)).isoformat(),
                    'metadata': {'kind': TASK_COUNT_SNAPSHOT_KIND},
                }]
            return []

        get_memories = AsyncMock(side_effect=_get_memories_by_metadata)
        mock_deps['memory_service'].get_memories_by_metadata = get_memories

        stage = TaskKnowledgeSync(
            StageId.task_knowledge_sync,
            scope=_scope('dark_factory', '/tmp/test'),
            **mock_deps,
        )

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=self._fake_cli_result()),
        ):
            report = await stage.run(
                events=[], watermark=Watermark(project_id='dark_factory'),
                prior_reports=[], run_id='run-window-fwd',
            )

        assert report.stats['task_count_snapshot_written'] == 1
        snapshot_calls = [
            call for call in get_memories.await_args_list
            if call.kwargs.get('filters') == {'kind': TASK_COUNT_SNAPSHOT_KIND}
        ]
        assert len(snapshot_calls) == 1
        assert snapshot_calls[0].kwargs.get('project_id') == 'dark_factory'

    @pytest.mark.asyncio
    async def test_unknown_run_window_start_omits_stat_without_querying(self, mock_deps):
        """Converse of the forwarding test above (reviewer finding, amendment
        round): when journal.get_run yields no usable started_at (the
        pre-existing best-effort fallback), self._run_window_start stays
        None and run() must forward that None through -- the stat key is
        omitted and the snapshot-kind query never fires. Together with the
        prior test, this pins the stat's presence/value to the real
        self._run_window_start handoff rather than to some independent
        code path.
        """
        mock_deps['journal'].get_run = AsyncMock(return_value=None)
        get_memories = AsyncMock(return_value=[])
        mock_deps['memory_service'].get_memories_by_metadata = get_memories

        stage = TaskKnowledgeSync(
            StageId.task_knowledge_sync,
            scope=_scope('dark_factory', '/tmp/test'),
            **mock_deps,
        )

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=self._fake_cli_result()),
        ):
            report = await stage.run(
                events=[], watermark=Watermark(project_id='dark_factory'),
                prior_reports=[], run_id='run-window-none',
            )

        assert 'task_count_snapshot_written' not in report.stats
        snapshot_calls = [
            call for call in get_memories.await_args_list
            if call.kwargs.get('filters') == {'kind': TASK_COUNT_SNAPSHOT_KIND}
        ]
        assert snapshot_calls == []


# ---------------------------------------------------------------------------
# TaskKnowledgeSync.run() wiring — deterministic write vs freshness-read
# fallback (task 2325 step-7/8)
# ---------------------------------------------------------------------------


class TestRunDeterministicSnapshotWrite:
    """run() attempts the deterministic write for non-blocked projects and
    falls back to the freshness read for blocked projects (task 2325).

    Mirrors TestRunRecordsTaskCountSnapshotWrittenStat's harness: super().run()
    executes for real via a patched run_stage_via_cli while the module-level
    write/verify helpers are patched directly, isolating these tests to the
    run()-level gating decision (not either helper's own internals).
    """

    @pytest.fixture
    def mock_deps(self):
        config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata.return_value = 1
        memory_service.delete_memory = AsyncMock(return_value=None)
        memory_service.search.return_value = []
        memory_service.add_memory.return_value = {'memory_ids': []}
        taskmaster = AsyncMock()
        taskmaster.get_tasks.return_value = {'tasks': []}
        return {
            'memory_service': memory_service,
            'taskmaster': taskmaster,
            'journal': AsyncMock(),
            'config': config,
        }

    def _fake_cli_result(self):
        return MagicMock(
            success=True,
            report={'flagged_items': [], 'summary': 'ok', 'stats': {}},
            llm_calls=1, tokens_used=0, cost_usd=0.0,
            model='test-model', error=None,
        )

    @pytest.mark.asyncio
    async def test_non_blocked_project_writes_and_skips_verify(self, mock_deps):
        """project_id='reify' (non-blocked): run() calls the deterministic
        write, sets the stat straight from its True result, and never calls
        the freshness-read fallback."""
        stage = TaskKnowledgeSync(
            StageId.task_knowledge_sync,
            scope=_scope('reify', '/tmp/test'),
            **mock_deps,
        )

        with (
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=self._fake_cli_result()),
            ),
            patch(
                'fused_memory.reconciliation.stages.task_knowledge_sync'
                '._write_task_count_snapshot',
                new=AsyncMock(return_value=True),
            ) as mock_write,
            patch(
                'fused_memory.reconciliation.stages.task_knowledge_sync'
                '._verify_task_count_snapshot_written',
                new=AsyncMock(),
            ) as mock_verify,
        ):
            report = await stage.run(
                events=[], watermark=Watermark(project_id='reify'),
                prior_reports=[], run_id='run-det-write',
            )

        assert report.stats['task_count_snapshot_written'] == 1
        mock_write.assert_awaited_once()
        mock_verify.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_blocked_project_write_none_falls_back_to_verify(self, mock_deps):
        """project_id='reify' (non-blocked) but the deterministic write comes
        back inconclusive (None -- transient failure or no taskmaster): run()
        must still fall back to the freshness-read helper for the stat, the
        same as the blocked-project path (reviewer finding, amendment round).

        This pins the ``if task_count_snapshot_written is None:`` fallback
        branch itself -- the two pre-existing tests in this class only cover
        write-returns-True (verify skipped) and blocked (write skipped,
        verify used); neither exercises attempted-write-then-fallback.
        """
        stage = TaskKnowledgeSync(
            StageId.task_knowledge_sync,
            scope=_scope('reify', '/tmp/test'),
            **mock_deps,
        )

        with (
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=self._fake_cli_result()),
            ),
            patch(
                'fused_memory.reconciliation.stages.task_knowledge_sync'
                '._write_task_count_snapshot',
                new=AsyncMock(return_value=None),
            ) as mock_write,
            patch(
                'fused_memory.reconciliation.stages.task_knowledge_sync'
                '._verify_task_count_snapshot_written',
                new=AsyncMock(return_value=True),
            ) as mock_verify,
        ):
            report = await stage.run(
                events=[], watermark=Watermark(project_id='reify'),
                prior_reports=[], run_id='run-det-write-fallback',
            )

        assert report.stats['task_count_snapshot_written'] == 1
        mock_write.assert_awaited_once()
        mock_verify.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_blocked_project_skips_write_and_falls_back_to_verify(self, mock_deps):
        """project_id='dark_factory' (blocked): run() never calls the
        deterministic write and falls back to the freshness-read helper for
        the stat."""
        stage = TaskKnowledgeSync(
            StageId.task_knowledge_sync,
            scope=_scope('dark_factory', '/tmp/test'),
            **mock_deps,
        )

        with (
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=self._fake_cli_result()),
            ),
            patch(
                'fused_memory.reconciliation.stages.task_knowledge_sync'
                '._write_task_count_snapshot',
                new=AsyncMock(),
            ) as mock_write,
            patch(
                'fused_memory.reconciliation.stages.task_knowledge_sync'
                '._verify_task_count_snapshot_written',
                new=AsyncMock(return_value=False),
            ) as mock_verify,
        ):
            report = await stage.run(
                events=[], watermark=Watermark(project_id='dark_factory'),
                prior_reports=[], run_id='run-det-blocked',
            )

        assert report.stats['task_count_snapshot_written'] == 0
        mock_write.assert_not_awaited()
        mock_verify.assert_awaited_once()
