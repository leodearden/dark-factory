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
from fused_memory.reconciliation.stages import (
    task_knowledge_sync as task_knowledge_sync_module,
)
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    TaskKnowledgeSync,
    _prune_task_count_snapshots,
    _verify_task_count_snapshot_written,
    _write_task_count_snapshot,
)
from fused_memory.reconciliation.task_count_snapshot_cadence import (
    ESCALATION_CATEGORY,
    LEGACY_SNAPSHOT_WRITTEN_STAT_KEY,
    SNAPSHOT_PRUNE_ENUMERATED_STAT_KEY,
    SNAPSHOT_PRUNE_ENUMERATION_OK_STAT_KEY,
    SNAPSHOT_PRUNE_TRUNCATED_STAT_KEY,
    SNAPSHOT_PRUNED_STAT_KEY,
    SNAPSHOT_WRITTEN_STAT_KEY,
    TASK_COUNT_SNAPSHOT_CATEGORY,
    TASK_COUNT_SNAPSHOT_KIND,
    TASK_COUNT_SNAPSHOT_MISS_THRESHOLD,
    build_stale_snapshot_finding,
    build_task_count_snapshot_content,
    build_task_count_snapshot_unavailable_content,
    compute_snapshot_miss_streak,
    evaluate_snapshot_cadence,
    extract_snapshot_written,
)
from fused_memory.reconciliation.task_filter import is_count_snapshot


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


@pytest.fixture
def mock_deps():
    """Stage-2 constructor kwargs for the run()-level test harnesses below.

    Shared by TestRunRecordsTaskCountSnapshotWrittenStat and
    TestSnapshotWrittenStatKeyIsConstantDriven, which exercise the same
    ``super().run()``-for-real harness and had byte-identical copies of this
    fixture (code-duplication finding, task-3045 amendment round): two
    copies drift the moment run()'s dependencies change, and only one gets
    updated. Classes needing a different shape still define their own
    ``mock_deps``, which shadows this one.
    """
    config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
    memory_service = AsyncMock()
    # count==1 short-circuits the unrelated stage2-summary verify/repair/
    # reconstruct chain so these tests stay isolated to the snapshot stat.
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


def _fake_cli_result():
    """Canned run_stage_via_cli result — the CLI leg is patched out so these
    tests target run()'s own pre/post-flight wiring, not the subprocess."""
    return MagicMock(
        success=True,
        report={'flagged_items': [], 'summary': 'ok', 'stats': {}},
        llm_calls=1, tokens_used=0, cost_usd=0.0,
        model='test-model', error=None,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Assert module-level constants have expected values."""

    def test_kind_value(self):
        assert TASK_COUNT_SNAPSHOT_KIND == 'task_count_snapshot'

    def test_stat_key_value(self):
        """Exact-value pin — also the anti-regression guard for the task-3045
        rename: the un-namespaced spelling read as a persistence claim about
        Graphiti, which is what drove Stage 3 / the judge to report a
        "rejected write" for a snapshot that is Mem0-only BY DESIGN (see
        Snapshot Discipline, prompts/stage1.py). Pinning the exact value is
        what stops a future edit from quietly restoring a Graphiti-implying
        spelling.
        """
        assert SNAPSHOT_WRITTEN_STAT_KEY == 'task_count_snapshot_mem0_written'

    def test_prune_enumerated_stat_key_value(self):
        assert SNAPSHOT_PRUNE_ENUMERATED_STAT_KEY == 'task_count_snapshot_prune_enumerated'

    def test_pruned_stat_key_value(self):
        assert SNAPSHOT_PRUNED_STAT_KEY == 'task_count_snapshot_mem0_pruned'

    def test_legacy_written_stat_key_value(self):
        assert LEGACY_SNAPSHOT_WRITTEN_STAT_KEY == 'task_count_snapshot_written'

    def test_prune_enumeration_ok_stat_key_value(self):
        assert SNAPSHOT_PRUNE_ENUMERATION_OK_STAT_KEY == 'task_count_snapshot_prune_enumeration_ok'

    def test_prune_truncated_stat_key_value(self):
        assert SNAPSHOT_PRUNE_TRUNCATED_STAT_KEY == 'task_count_snapshot_prune_truncated'

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
        report = _stage_report({SNAPSHOT_WRITTEN_STAT_KEY: 1})
        assert extract_snapshot_written(report) is True

    def test_stats_0_is_false_on_stage_report(self):
        report = _stage_report({SNAPSHOT_WRITTEN_STAT_KEY: 0})
        assert extract_snapshot_written(report) is False

    def test_missing_key_is_none_on_stage_report(self):
        report = _stage_report({})
        assert extract_snapshot_written(report) is None

    def test_stats_1_is_true_on_raw_dict(self):
        report = {'stats': {SNAPSHOT_WRITTEN_STAT_KEY: 1}}
        assert extract_snapshot_written(report) is True

    def test_stats_0_is_false_on_raw_dict(self):
        report = {'stats': {SNAPSHOT_WRITTEN_STAT_KEY: 0}}
        assert extract_snapshot_written(report) is False

    def test_missing_key_is_none_on_raw_dict(self):
        report = {'stats': {}}
        assert extract_snapshot_written(report) is None

    def test_missing_stats_key_is_none_on_raw_dict(self):
        assert extract_snapshot_written({}) is None

    def test_none_report_is_none(self):
        assert extract_snapshot_written(None) is None

    # --- legacy-key back-compat (task 3045) ---------------------------------
    #
    # harness._maybe_escalate_stale_task_count_snapshot recomputes its
    # consecutive-miss streak from journal.get_recent_runs -- i.e. from
    # stage_reports blobs persisted by cycles that ran BEFORE the rename.
    # Without a fallback every such row reads as None,
    # compute_snapshot_miss_streak stops at the first one, and the
    # recon_stale_task_count_snapshot escalation goes silently dead instead
    # of loudly wrong.

    def test_legacy_key_1_is_true_on_stage_report(self):
        report = _stage_report({LEGACY_SNAPSHOT_WRITTEN_STAT_KEY: 1})
        assert extract_snapshot_written(report) is True

    def test_legacy_key_0_is_false_on_stage_report(self):
        report = _stage_report({LEGACY_SNAPSHOT_WRITTEN_STAT_KEY: 0})
        assert extract_snapshot_written(report) is False

    def test_legacy_key_1_is_true_on_raw_dict(self):
        report = {'stats': {LEGACY_SNAPSHOT_WRITTEN_STAT_KEY: 1}}
        assert extract_snapshot_written(report) is True

    def test_legacy_key_0_is_false_on_raw_dict(self):
        report = {'stats': {LEGACY_SNAPSHOT_WRITTEN_STAT_KEY: 0}}
        assert extract_snapshot_written(report) is False

    def test_new_key_wins_over_legacy_when_both_present_new_0(self):
        """Precedence is deterministic: the new key wins, both directions."""
        report = _stage_report({
            SNAPSHOT_WRITTEN_STAT_KEY: 0,
            LEGACY_SNAPSHOT_WRITTEN_STAT_KEY: 1,
        })
        assert extract_snapshot_written(report) is False

    def test_new_key_wins_over_legacy_when_both_present_new_1(self):
        report = _stage_report({
            SNAPSHOT_WRITTEN_STAT_KEY: 1,
            LEGACY_SNAPSHOT_WRITTEN_STAT_KEY: 0,
        })
        assert extract_snapshot_written(report) is True

    def test_neither_key_present_is_still_none(self):
        report = _stage_report({'some_unrelated_stat': 1})
        assert extract_snapshot_written(report) is None


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


class TestBuildTaskCountSnapshotUnavailableContent:
    """build_task_count_snapshot_unavailable_content(...) -> str (task 2738).

    Pure, dependency-free renderer for the UNKNOWN sentinel written by
    ``_write_task_count_snapshot`` when a zero-count project_root is not a
    readable git working tree -- i.e. the zero count is a false census from
    SqliteTaskBackend.get_tasks auto-creating an empty tasks.db, not a
    genuinely empty project. Sibling of TestBuildTaskCountSnapshotContent
    above: no I/O, no mocks needed.
    """

    def test_content_is_non_empty_string(self):
        out = build_task_count_snapshot_unavailable_content(
            'my_solar_challenge', '/home/leo/src/my-solar-challenge',
        )

        assert isinstance(out, str)
        assert out

    def test_content_is_not_mistaken_for_a_numeric_count_snapshot(self):
        out = build_task_count_snapshot_unavailable_content(
            'my_solar_challenge', '/home/leo/src/my-solar-challenge',
        )

        assert is_count_snapshot(out) is False

    def test_content_names_project_id_and_root(self):
        out = build_task_count_snapshot_unavailable_content(
            'my_solar_challenge', '/home/leo/src/my-solar-challenge',
        )

        assert 'my_solar_challenge' in out
        assert '/home/leo/src/my-solar-challenge' in out

    def test_content_signals_unavailability_and_reason(self):
        out = build_task_count_snapshot_unavailable_content(
            'my_solar_challenge', '/home/leo/src/my-solar-challenge',
        )

        assert 'UNAVAILABLE' in out or 'unavailable' in out
        # A stable, semantically-load-bearing token rather than the full
        # prose sentence -- pins "this is about git working tree
        # readability" without locking the exact wording (amendment,
        # task 2738 review).
        assert 'git working tree' in out

    def test_content_carries_no_zeroed_numeric_census(self):
        out = build_task_count_snapshot_unavailable_content(
            'my_solar_challenge', '/home/leo/src/my-solar-challenge',
        )

        assert '0 total' not in out
        assert '0 done' not in out


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
# _prune_task_count_snapshots (stages/task_knowledge_sync.py) -- task 2429 step-1/2, step-3/4
# ---------------------------------------------------------------------------


class TestPruneTaskCountSnapshots:
    """_prune_task_count_snapshots(memory_service, project_id, run_id) -> int.

    Best-effort, never-raising deletion of ALL existing kind='task_count_snapshot'
    Mem0 records, so that a subsequent add_memory leaves exactly one canonical
    snapshot. Structurally mirrors _sweep_stale_persistence_markers's
    enumerate-via-get_memories_by_metadata + parallel-delete-via-gather_collect
    template, minus the age cutoff (this prune deletes ALL matches, not just
    aged ones).
    """

    @pytest.mark.asyncio
    async def test_deletes_all_enumerated_records_and_returns_count(self):
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {'id': 'm1', 'created_at': '2026-07-01T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
            {'id': 'm2', 'created_at': '2026-07-02T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
            {'id': 'm3', 'created_at': '2026-07-03T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
        ]
        memory_service.delete_memory.return_value = None

        result = await _prune_task_count_snapshots(memory_service, 'reify', 'run-1')

        assert result == 3
        memory_service.get_memories_by_metadata.assert_awaited_once()
        _, kwargs = memory_service.get_memories_by_metadata.await_args
        assert kwargs['filters'] == {'kind': 'task_count_snapshot'}

        assert memory_service.delete_memory.await_count == 3
        deleted_ids = {
            call.kwargs.get('memory_id') for call in memory_service.delete_memory.call_args_list
        }
        assert deleted_ids == {'m1', 'm2', 'm3'}
        for call in memory_service.delete_memory.call_args_list:
            call_kwargs = call.kwargs
            assert call_kwargs.get('store') == 'mem0'
            assert call_kwargs.get('project_id') == 'reify'
            assert call_kwargs.get('causation_id') == 'run-1'

    @pytest.mark.asyncio
    async def test_empty_enumeration_returns_zero_and_no_deletes(self):
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = []

        result = await _prune_task_count_snapshots(memory_service, 'reify', 'run-1')

        assert result == 0
        memory_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_enumeration_failure_returns_zero_not_raise(self):
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.side_effect = RuntimeError('mem0 down')

        result = await _prune_task_count_snapshots(memory_service, 'reify', 'run-1')

        assert result == 0
        memory_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_partial_delete_failure_excluded_from_count(self):
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {'id': 'm1', 'created_at': '2026-07-01T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
            {'id': 'm2', 'created_at': '2026-07-02T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
            {'id': 'm3', 'created_at': '2026-07-03T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
        ]
        memory_service.delete_memory.side_effect = [
            {'status': 'deleted'}, RuntimeError('boom'), {'status': 'deleted'},
        ]

        result = await _prune_task_count_snapshots(memory_service, 'reify', 'run-1')

        assert result == 2
        assert memory_service.delete_memory.await_count == 3

    @pytest.mark.asyncio
    async def test_records_missing_id_are_skipped(self):
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {'created_at': '2026-07-01T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
            {'id': 'm2', 'created_at': '2026-07-02T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
        ]
        memory_service.delete_memory.return_value = None

        result = await _prune_task_count_snapshots(memory_service, 'reify', 'run-1')

        assert result == 1
        memory_service.delete_memory.assert_awaited_once()
        _, kwargs = memory_service.delete_memory.await_args
        assert kwargs['memory_id'] == 'm2'


# ---------------------------------------------------------------------------
# _prune_task_count_snapshots -- optional stats dict (task 2646 step-1/2, step-3/4)
# ---------------------------------------------------------------------------


class TestPruneSnapshotStats:
    """_prune_task_count_snapshots(..., stats=observed) populates observed
    with the three runtime-observability counts (task 2646).

    The crux this class exists to pin: a silent enumeration failure (the
    incident fingerprint -- prune deletes nothing, canonical write still
    proceeds) must be OBSERVABLY DISTINCT from a genuine empty result. A
    single delete-count int cannot make this distinction; enumeration_ok
    is the flag that does.
    """

    @pytest.mark.asyncio
    async def test_full_success_populates_all_three_stats(self):
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {'id': 'm1', 'created_at': '2026-07-01T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
            {'id': 'm2', 'created_at': '2026-07-02T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
            {'id': 'm3', 'created_at': '2026-07-03T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
        ]
        memory_service.delete_memory.return_value = None
        observed = {}

        result = await _prune_task_count_snapshots(
            memory_service, 'reify', 'run-1', stats=observed,
        )

        assert result == 3
        assert observed == {
            'task_count_snapshot_prune_enumerated': 3,
            SNAPSHOT_PRUNED_STAT_KEY: 3,
            'task_count_snapshot_prune_enumeration_ok': 1,
            'task_count_snapshot_prune_truncated': 0,
        }

    @pytest.mark.asyncio
    async def test_partial_delete_failure_counts_only_successes(self):
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {'id': 'm1', 'created_at': '2026-07-01T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
            {'id': 'm2', 'created_at': '2026-07-02T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
            {'id': 'm3', 'created_at': '2026-07-03T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
        ]
        memory_service.delete_memory.side_effect = [
            {'status': 'deleted'}, RuntimeError('boom'), {'status': 'deleted'},
        ]
        observed = {}

        result = await _prune_task_count_snapshots(
            memory_service, 'reify', 'run-1', stats=observed,
        )

        assert result == 2
        assert observed == {
            'task_count_snapshot_prune_enumerated': 3,
            SNAPSHOT_PRUNED_STAT_KEY: 2,
            'task_count_snapshot_prune_enumeration_ok': 1,
            'task_count_snapshot_prune_truncated': 0,
        }

    @pytest.mark.asyncio
    async def test_enumeration_failure_is_observably_distinct_from_empty(self):
        """The incident fingerprint: a silent enumeration failure must be
        distinguishable from a genuine empty result. Both return 0/leave
        enumerated=0/pruned=0, but only the failure case sets
        enumeration_ok=0 -- a single delete-count int cannot make this
        distinction."""
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.side_effect = RuntimeError('mem0 down')
        observed = {}

        result = await _prune_task_count_snapshots(
            memory_service, 'reify', 'run-1', stats=observed,
        )

        assert result == 0
        assert observed == {
            'task_count_snapshot_prune_enumerated': 0,
            SNAPSHOT_PRUNED_STAT_KEY: 0,
            'task_count_snapshot_prune_enumeration_ok': 0,
            'task_count_snapshot_prune_truncated': 0,
        }
        memory_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_genuine_empty_enumeration_sets_enumeration_ok(self):
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = []
        # Task 2655: an empty scroll now triggers a count cross-check (see
        # TestPruneSilentEmptyGuard below) -- a confirmed count of 0 is what
        # makes this a genuine empty pool rather than the swallowed-timeout
        # fingerprint (empty scroll + count > 0).
        memory_service.count_memories_by_metadata.return_value = 0
        observed = {}

        result = await _prune_task_count_snapshots(
            memory_service, 'reify', 'run-1', stats=observed,
        )

        assert result == 0
        assert observed == {
            'task_count_snapshot_prune_enumerated': 0,
            SNAPSHOT_PRUNED_STAT_KEY: 0,
            'task_count_snapshot_prune_enumeration_ok': 1,
            'task_count_snapshot_prune_truncated': 0,
        }
        memory_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_scroll_cap_reached_sets_truncated_stat(self):
        """Amendment round (reviewer finding): hitting the scroll_limit cap
        means older stale snapshots may remain unpruned this cycle -- a
        provably incomplete prune. Reported identically to a clean success
        by enumerated/pruned/enumeration_ok alone (all look the same as a
        real 2-record success); only the dedicated truncated stat makes a
        capped page observably distinct from a complete one. Mirrors the
        scroll_limit=N-with-N-records saturation idiom already used by
        TestVerifyTaskCountSnapshotWritten."""
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {'id': 'm1', 'created_at': '2026-07-01T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
            {'id': 'm2', 'created_at': '2026-07-02T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
        ]
        memory_service.delete_memory.return_value = None
        observed = {}

        result = await _prune_task_count_snapshots(
            memory_service, 'reify', 'run-1', scroll_limit=2, stats=observed,
        )

        assert result == 2
        assert observed == {
            'task_count_snapshot_prune_enumerated': 2,
            SNAPSHOT_PRUNED_STAT_KEY: 2,
            'task_count_snapshot_prune_enumeration_ok': 1,
            'task_count_snapshot_prune_truncated': 1,
        }

    @pytest.mark.asyncio
    async def test_unsaturated_page_leaves_truncated_stat_unset(self):
        """A page below scroll_limit is exhaustive -- not a truncation --
        even though it enumerates the same 2 records as the capped case
        above."""
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {'id': 'm1', 'created_at': '2026-07-01T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
            {'id': 'm2', 'created_at': '2026-07-02T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
        ]
        memory_service.delete_memory.return_value = None
        observed = {}

        result = await _prune_task_count_snapshots(
            memory_service, 'reify', 'run-1', scroll_limit=3, stats=observed,
        )

        assert result == 2
        assert observed == {
            'task_count_snapshot_prune_enumerated': 2,
            SNAPSHOT_PRUNED_STAT_KEY: 2,
            'task_count_snapshot_prune_enumeration_ok': 1,
            'task_count_snapshot_prune_truncated': 0,
        }


# ---------------------------------------------------------------------------
# _prune_task_count_snapshots -- silent-empty-enumeration guard (task 2655)
# ---------------------------------------------------------------------------


class TestPruneSilentEmptyGuard:
    """The incident fingerprint (task 2655, 5th recorded recurrence):
    ``Mem0Backend.scroll_by_metadata`` catches ``TimeoutError`` and returns
    ``[]`` (mem0_client.py:392-407), while its sibling ``count_by_metadata``
    (mem0_client.py:296-339) lets timeouts propagate. So an empty,
    NON-exceptional scroll page is ambiguous: a genuine empty pool and a
    swallowed Qdrant read timeout look identical to
    ``enumeration_ok``/``enumerated`` alone.

    This class pins the fix: on an empty scroll, cross-check via
    ``count_memories_by_metadata`` (project 2655's chosen cross-check
    primitive, since it propagates timeouts). A raised cross-check, or a
    confirmed count > 0, is the swallowed-timeout fingerprint and flips
    ``enumeration_ok`` to False; only a confirmed count of 0 keeps
    ``enumeration_ok`` True. The cross-check itself must never raise
    (preserves the prune's best-effort, never-raises contract), and must
    not run at all on a non-empty scroll (hot path unaffected).
    """

    @pytest.mark.asyncio
    async def test_swallowed_timeout_fingerprint_sets_enumeration_not_ok(self):
        """(a) Empty scroll but the count cross-check reports 3 existing
        snapshots -- the swallowed-timeout fingerprint. Nothing was
        enumerated, so nothing is deleted, but enumeration_ok must be
        observably 0 rather than looking like a genuine empty pool."""
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = []
        memory_service.count_memories_by_metadata.return_value = 3
        observed = {}

        result = await _prune_task_count_snapshots(
            memory_service, 'reify', 'run-1', stats=observed,
        )

        assert result == 0
        assert observed == {
            'task_count_snapshot_prune_enumerated': 0,
            SNAPSHOT_PRUNED_STAT_KEY: 0,
            'task_count_snapshot_prune_enumeration_ok': 0,
            'task_count_snapshot_prune_truncated': 0,
        }
        memory_service.delete_memory.assert_not_awaited()
        memory_service.count_memories_by_metadata.assert_awaited_once()
        _, kwargs = memory_service.count_memories_by_metadata.await_args
        assert kwargs['project_id'] == 'reify'
        assert kwargs['filters'] == {'kind': TASK_COUNT_SNAPSHOT_KIND}

    @pytest.mark.asyncio
    async def test_count_cross_check_raises_sets_enumeration_not_ok_never_raises(self):
        """(b) Empty scroll and the count cross-check itself raises -- still
        degrades to enumeration_ok=0 rather than propagating (never-raise
        contract preserved even for the new cross-check)."""
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = []
        memory_service.count_memories_by_metadata.side_effect = RuntimeError('mem0 down')
        observed = {}

        result = await _prune_task_count_snapshots(
            memory_service, 'reify', 'run-1', stats=observed,
        )

        assert result == 0
        assert observed == {
            'task_count_snapshot_prune_enumerated': 0,
            SNAPSHOT_PRUNED_STAT_KEY: 0,
            'task_count_snapshot_prune_enumeration_ok': 0,
            'task_count_snapshot_prune_truncated': 0,
        }
        memory_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_confirmed_zero_count_keeps_enumeration_ok(self):
        """(c) Empty scroll AND the count cross-check confirms 0 -- a
        genuine empty pool, not the swallowed-timeout fingerprint."""
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = []
        memory_service.count_memories_by_metadata.return_value = 0
        observed = {}

        result = await _prune_task_count_snapshots(
            memory_service, 'reify', 'run-1', stats=observed,
        )

        assert result == 0
        assert observed == {
            'task_count_snapshot_prune_enumerated': 0,
            SNAPSHOT_PRUNED_STAT_KEY: 0,
            'task_count_snapshot_prune_enumeration_ok': 1,
            'task_count_snapshot_prune_truncated': 0,
        }
        memory_service.count_memories_by_metadata.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_empty_scroll_never_invokes_count_cross_check(self):
        """(d) Hot path unaffected: a non-empty scroll page must not pay for
        the extra count cross-check call at all, and enumeration_ok stays
        True as before."""
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {'id': 'm1', 'created_at': '2026-07-01T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
            {'id': 'm2', 'created_at': '2026-07-02T00:00:00+00:00', 'metadata': {'kind': 'task_count_snapshot'}},
        ]
        memory_service.delete_memory.return_value = None
        observed = {}

        result = await _prune_task_count_snapshots(
            memory_service, 'reify', 'run-1', stats=observed,
        )

        assert result == 2
        assert observed == {
            'task_count_snapshot_prune_enumerated': 2,
            SNAPSHOT_PRUNED_STAT_KEY: 2,
            'task_count_snapshot_prune_enumeration_ok': 1,
            'task_count_snapshot_prune_truncated': 0,
        }
        memory_service.count_memories_by_metadata.assert_not_awaited()


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
        memory_service.get_memories_by_metadata.return_value = []
        # Task 2655: an empty scroll now triggers the prune's count
        # cross-check (TestPruneSilentEmptyGuard); a confirmed count of 0
        # is what makes this a genuine empty pool that the write-gate
        # (TestWriteTaskCountSnapshot below) lets through.
        memory_service.count_memories_by_metadata.return_value = 0
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
        memory_service.get_memories_by_metadata.return_value = []
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
        # Guard: never delete existing snapshots without a replacement in
        # hand -- the taskmaster=None early-return must precede the prune.
        memory_service.get_memories_by_metadata.assert_not_awaited()
        memory_service.delete_memory.assert_not_awaited()
        memory_service.add_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_prunes_existing_snapshots_before_writing(self):
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {
                'id': 'stale-1',
                'created_at': '2026-07-01T00:00:00+00:00',
                'metadata': {'kind': 'task_count_snapshot'},
            },
            {
                'id': 'stale-2',
                'created_at': '2026-07-02T00:00:00+00:00',
                'metadata': {'kind': 'task_count_snapshot'},
            },
        ]
        memory_service.delete_memory.return_value = None
        memory_service.add_memory.return_value = {'memory_ids': ['fresh-1']}
        taskmaster = self._taskmaster()

        result = await _write_task_count_snapshot(
            memory_service, taskmaster, '/tmp/test', 'reify', 'run-1', None,
        )

        assert result is True
        # Both stale ids pruned...
        assert memory_service.delete_memory.await_count == 2
        deleted_ids = {
            call.kwargs.get('memory_id') for call in memory_service.delete_memory.call_args_list
        }
        assert deleted_ids == {'stale-1', 'stale-2'}
        # ...and exactly one canonical snapshot written -- net one survivor.
        memory_service.add_memory.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_prune_runs_before_write_not_after(self):
        """A regression that writes-then-prunes must fail this ordering check."""
        call_order = []
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {
                'id': 'stale-1',
                'created_at': '2026-07-01T00:00:00+00:00',
                'metadata': {'kind': 'task_count_snapshot'},
            },
        ]

        async def _delete(*args, **kwargs):
            call_order.append('delete_memory')
            return None

        async def _add(*args, **kwargs):
            call_order.append('add_memory')
            return {'memory_ids': ['fresh-1']}

        memory_service.delete_memory.side_effect = _delete
        memory_service.add_memory.side_effect = _add
        taskmaster = self._taskmaster()

        result = await _write_task_count_snapshot(
            memory_service, taskmaster, '/tmp/test', 'reify', 'run-1', None,
        )

        assert result is True
        assert call_order == ['delete_memory', 'add_memory']

    @pytest.mark.asyncio
    async def test_add_memory_failure_after_successful_prune_returns_none(self):
        """Pins the accepted zero-snapshot window (task 2429 review).

        If the prune deletes existing snapshots but the subsequent
        ``add_memory`` then fails, this cycle leaves zero
        ``task_count_snapshot`` records for the project until the next
        cycle's write succeeds -- an accepted, self-correcting trade-off
        (see the docstring note on ``_write_task_count_snapshot``), not an
        oversight. This test pins that intended post-condition so it can't
        silently regress into something else unnoticed.
        """
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {
                'id': 'stale-1',
                'created_at': '2026-07-01T00:00:00+00:00',
                'metadata': {'kind': 'task_count_snapshot'},
            },
            {
                'id': 'stale-2',
                'created_at': '2026-07-02T00:00:00+00:00',
                'metadata': {'kind': 'task_count_snapshot'},
            },
        ]
        memory_service.delete_memory.return_value = None
        memory_service.add_memory.side_effect = RuntimeError('mem0 down')
        taskmaster = self._taskmaster()

        result = await _write_task_count_snapshot(
            memory_service, taskmaster, '/tmp/test', 'reify', 'run-1', None,
        )

        assert result is None
        # The prune still ran (and "succeeded") before the write failed --
        # both stale ids were deleted, leaving zero snapshots this cycle.
        assert memory_service.delete_memory.await_count == 2
        deleted_ids = {
            call.kwargs.get('memory_id') for call in memory_service.delete_memory.call_args_list
        }
        assert deleted_ids == {'stale-1', 'stale-2'}
        memory_service.add_memory.assert_awaited_once()


# ---------------------------------------------------------------------------
# _write_task_count_snapshot -- UNKNOWN sentinel for a non-git-working-tree
# zero-count project_root (task 2738)
# ---------------------------------------------------------------------------


class TestWriteTaskCountSnapshotUnavailableSentinel:
    """_write_task_count_snapshot writes an UNKNOWN sentinel instead of a
    zeroed numeric record when a zero-count ``project_root`` is not a
    readable git working tree.

    ``SqliteTaskBackend.get_tasks`` auto-creates an empty ``tasks.db`` and
    returns ``{'tasks': []}`` for ANY path (never raising), so a zero-count
    tree at a non-git ``project_root`` (e.g. a project whose repo was
    deleted) is a false census, indistinguishable at the data layer from a
    genuinely empty project. The writer disambiguates via
    ``resolve_main_checkout`` (raises ``ValueError`` iff *project_root* is
    not inside a readable git working tree), gated ONLY on
    ``tree.total_count == 0`` so the common non-empty path never pays the
    git-subprocess cost.
    """

    def _taskmaster_empty(self):
        taskmaster = AsyncMock()
        taskmaster.get_tasks.return_value = {'tasks': []}
        return taskmaster

    def _taskmaster_non_empty(self):
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

    def _memory_service(self):
        # Clean pool -> prune enumeration_ok=1 -> the write proceeds to
        # add_memory so the test can inspect the persisted record (mirrors
        # TestWriteTaskCountSnapshot.test_success_writes_once_and_returns_true).
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = []
        memory_service.count_memories_by_metadata.return_value = 0
        memory_service.add_memory.return_value = {'memory_ids': ['m1']}
        return memory_service

    @staticmethod
    def _persisted(memory_service) -> dict:
        """Read back the kwargs passed to the single add_memory call."""
        return memory_service.add_memory.await_args.kwargs

    @pytest.mark.asyncio
    async def test_zero_count_non_git_real_tmpdir_writes_unavailable_sentinel(self, tmp_path):
        """Faithful my_solar_challenge repro: a real non-git tmp dir drives
        the real (unpatched) resolve_main_checkout, which raises ValueError."""
        memory_service = self._memory_service()
        taskmaster = self._taskmaster_empty()

        result = await _write_task_count_snapshot(
            memory_service, taskmaster, str(tmp_path), 'my_solar_challenge', 'run-1', None,
        )

        assert result is True
        memory_service.add_memory.assert_awaited_once()
        kwargs = self._persisted(memory_service)
        assert kwargs['metadata']['snapshot_status'] == 'unavailable'
        assert kwargs['metadata']['kind'] == 'task_count_snapshot'
        assert '0 total' not in kwargs['content']
        assert '0 done' not in kwargs['content']
        assert is_count_snapshot(kwargs['content']) is False

    @pytest.mark.asyncio
    async def test_zero_count_non_git_patched_valueerror_writes_sentinel(self):
        """Deterministic mirror of the real-tmpdir case above, regardless of
        host git availability. create=True: resolve_main_checkout is not
        yet imported into task_knowledge_sync's namespace on pre-fix code,
        so patch must be able to install (and later remove) it either way."""
        memory_service = self._memory_service()
        taskmaster = self._taskmaster_empty()

        with patch(
            'fused_memory.reconciliation.stages.task_knowledge_sync.resolve_main_checkout',
            side_effect=ValueError('/some/path is not inside a git working tree'),
            create=True,
        ):
            result = await _write_task_count_snapshot(
                memory_service, taskmaster, '/some/path', 'my_solar_challenge', 'run-1', None,
            )

        assert result is True
        memory_service.add_memory.assert_awaited_once()
        kwargs = self._persisted(memory_service)
        assert kwargs['metadata']['snapshot_status'] == 'unavailable'
        assert kwargs['metadata']['kind'] == 'task_count_snapshot'
        assert '0 total' not in kwargs['content']
        assert '0 done' not in kwargs['content']
        assert is_count_snapshot(kwargs['content']) is False

    @pytest.mark.asyncio
    async def test_zero_count_git_project_root_writes_normal_zero_record(self):
        """GUARD: a genuinely empty project whose root IS a readable git
        working tree still gets its legitimate numeric zero snapshot, with
        no snapshot_status key. True both before and after the fix."""
        memory_service = self._memory_service()
        taskmaster = self._taskmaster_empty()

        with patch(
            'fused_memory.reconciliation.stages.task_knowledge_sync.resolve_main_checkout',
            return_value='/main/checkout',
            create=True,
        ):
            result = await _write_task_count_snapshot(
                memory_service, taskmaster, '/main/checkout/sub', 'reify', 'run-1', None,
            )

        assert result is True
        memory_service.add_memory.assert_awaited_once()
        kwargs = self._persisted(memory_service)
        assert '0 total' in kwargs['content']
        assert 'snapshot_status' not in kwargs['metadata']

    @pytest.mark.asyncio
    async def test_non_empty_tree_never_invokes_git_check(self):
        """GUARD: pins the total==0 short-circuit -- a non-empty tree never
        pays the git-subprocess cost. True both before and after the fix."""
        memory_service = self._memory_service()
        taskmaster = self._taskmaster_non_empty()

        with patch(
            'fused_memory.reconciliation.stages.task_knowledge_sync.resolve_main_checkout',
            create=True,
        ) as mocked_resolve:
            result = await _write_task_count_snapshot(
                memory_service, taskmaster, '/tmp/test', 'reify', 'run-1', None,
            )

        assert result is True
        kwargs = self._persisted(memory_service)
        assert '4 total' in kwargs['content']
        assert 'snapshot_status' not in kwargs['metadata']
        mocked_resolve.assert_not_called()


# ---------------------------------------------------------------------------
# _write_task_count_snapshot -- enumeration-failure write-gate (task 2655)
# ---------------------------------------------------------------------------


class TestWriteTaskCountSnapshotEnumerationGate:
    """_write_task_count_snapshot skips the deterministic write (returns
    None, never calls add_memory) when the prune reports
    enumeration_ok=0 (task 2655).

    Writing a fresh snapshot when the prior ones could not be
    enumerated/pruned is exactly what grows the byte-identical duplicate
    pile (the recurring incident this task exists to fix). Skipping is
    self-correcting: the next healthy cycle enumerates and prunes all
    accumulated duplicates, then writes one. Deliberately supersedes task
    2646's pinned "write still proceeds on enumeration failure" behavior --
    see TestRunSurfacesPruneObservability.
    test_live_silent_enumeration_failure_surfaces_as_not_ok, updated
    alongside this class.
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
    async def test_enumeration_raises_skips_write(self):
        """(a) The prune's own scroll call raises outright -- the write
        must be skipped (returns None, add_memory never called) rather
        than proceeding to add a fresh snapshot on top of an unprunable
        pile."""
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.side_effect = RuntimeError('mem0 down')
        taskmaster = self._taskmaster()

        result = await _write_task_count_snapshot(
            memory_service, taskmaster, '/tmp/test', 'reify', 'run-1', None,
        )

        assert result is None
        memory_service.add_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_timeout_fingerprint_skips_write(self):
        """(b) Empty scroll + count cross-check reports 2 existing
        snapshots -- the swallowed-timeout fingerprint. The write must be
        skipped so the cycle doesn't add another duplicate on top of the
        un-enumerable pile."""
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = []
        memory_service.count_memories_by_metadata.return_value = 2
        taskmaster = self._taskmaster()
        observed = {}

        result = await _write_task_count_snapshot(
            memory_service, taskmaster, '/tmp/test', 'reify', 'run-1', None,
            stats=observed,
        )

        assert result is None
        memory_service.add_memory.assert_not_awaited()
        assert observed[SNAPSHOT_PRUNE_ENUMERATION_OK_STAT_KEY] == 0

    @pytest.mark.asyncio
    async def test_healthy_enumeration_still_writes(self):
        """(c) Positive guard: a normal non-empty scroll (enumeration_ok=1)
        must still write -- the gate must not over-skip a healthy cycle."""
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {
                'id': 'stale-1',
                'created_at': '2026-07-01T00:00:00+00:00',
                'metadata': {'kind': 'task_count_snapshot'},
            },
        ]
        memory_service.delete_memory.return_value = None
        memory_service.add_memory.return_value = {'memory_ids': ['fresh-1']}
        taskmaster = self._taskmaster()

        result = await _write_task_count_snapshot(
            memory_service, taskmaster, '/tmp/test', 'reify', 'run-1', None,
        )

        assert result is True
        memory_service.add_memory.assert_awaited_once()


# ---------------------------------------------------------------------------
# _write_task_count_snapshot -- threading the stats dict (task 2646 step-5/6)
# ---------------------------------------------------------------------------


class TestWriteThreadsPruneStats:
    """_write_task_count_snapshot(..., stats=observed) threads its stats dict
    straight into _prune_task_count_snapshots, so the prune's runtime-
    observability counts are populated only when the prune is actually
    reached (task 2646).
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
    async def test_success_threads_prune_stats(self):
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {
                'id': 'stale-1',
                'created_at': '2026-07-01T00:00:00+00:00',
                'metadata': {'kind': 'task_count_snapshot'},
            },
            {
                'id': 'stale-2',
                'created_at': '2026-07-02T00:00:00+00:00',
                'metadata': {'kind': 'task_count_snapshot'},
            },
        ]
        memory_service.delete_memory.return_value = None
        memory_service.add_memory.return_value = {'memory_ids': ['fresh-1']}
        taskmaster = self._taskmaster()
        observed = {}

        result = await _write_task_count_snapshot(
            memory_service, taskmaster, '/tmp/test', 'reify', 'run-1', None,
            stats=observed,
        )

        assert result is True
        assert observed == {
            'task_count_snapshot_prune_enumerated': 2,
            SNAPSHOT_PRUNED_STAT_KEY: 2,
            'task_count_snapshot_prune_enumeration_ok': 1,
            'task_count_snapshot_prune_truncated': 0,
        }

    @pytest.mark.asyncio
    async def test_no_taskmaster_leaves_stats_untouched(self):
        """The taskmaster-None early-return precedes the prune entirely, so
        stats must stay untouched (not even zeroed) -- the prune was never
        attempted."""
        memory_service = AsyncMock()
        observed = {}

        result = await _write_task_count_snapshot(
            memory_service, None, '/tmp/test', 'reify', 'run-1', None,
            stats=observed,
        )

        assert result is None
        assert observed == {}


# ---------------------------------------------------------------------------
# TaskKnowledgeSync.run() wiring — report.stats[SNAPSHOT_WRITTEN_STAT_KEY]
# ---------------------------------------------------------------------------


class TestRunRecordsTaskCountSnapshotWrittenStat:
    """TaskKnowledgeSync.run() records the freshness stat (step-7/8).

    Mirrors tests/test_stages.py's TestRunStage2SummaryReconstructionWiring
    harness: super().run() executes for real via a patched run_stage_via_cli
    while the module-level helper is patched directly, so these tests target
    only the run()-level wiring — not _verify_task_count_snapshot_written's
    own internals (covered by TestVerifyTaskCountSnapshotWritten above).

    Consumes the module-level ``mock_deps`` fixture and ``_fake_cli_result``
    helper, shared with TestSnapshotWrittenStatKeyIsConstantDriven below.
    """

    async def _run_with_snapshot_check(self, mock_deps, snapshot_result, run_id):
        stage = TaskKnowledgeSync(
            StageId.task_knowledge_sync,
            scope=_scope('dark_factory', '/tmp/test'),
            **mock_deps,
        )

        with (
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=_fake_cli_result()),
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
        assert report.stats[SNAPSHOT_WRITTEN_STAT_KEY] == 1

    @pytest.mark.asyncio
    async def test_helper_false_sets_stat_to_0(self, mock_deps):
        report = await self._run_with_snapshot_check(mock_deps, False, 'run-snap-false')
        assert report.stats[SNAPSHOT_WRITTEN_STAT_KEY] == 0

    @pytest.mark.asyncio
    async def test_helper_none_omits_stat_key(self, mock_deps):
        report = await self._run_with_snapshot_check(mock_deps, None, 'run-snap-none')
        assert SNAPSHOT_WRITTEN_STAT_KEY not in report.stats

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
            new=AsyncMock(return_value=_fake_cli_result()),
        ):
            report = await stage.run(
                events=[], watermark=Watermark(project_id='dark_factory'),
                prior_reports=[], run_id='run-window-fwd',
            )

        assert report.stats[SNAPSHOT_WRITTEN_STAT_KEY] == 1
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
            new=AsyncMock(return_value=_fake_cli_result()),
        ):
            report = await stage.run(
                events=[], watermark=Watermark(project_id='dark_factory'),
                prior_reports=[], run_id='run-window-none',
            )

        assert SNAPSHOT_WRITTEN_STAT_KEY not in report.stats
        snapshot_calls = [
            call for call in get_memories.await_args_list
            if call.kwargs.get('filters') == {'kind': TASK_COUNT_SNAPSHOT_KIND}
        ]
        assert snapshot_calls == []


# ---------------------------------------------------------------------------
# TaskKnowledgeSync.run() wiring — the freshness stat is emitted under the
# shared module constant, never a hardcoded literal (task 3045 step-1/2)
# ---------------------------------------------------------------------------


class TestSnapshotWrittenStatKeyIsConstantDriven:
    """run() must emit the freshness stat under SNAPSHOT_WRITTEN_STAT_KEY.

    Rename-hazard guard (task 3045). The producer and the sole reader
    (``extract_snapshot_written``) live in different modules: the reader
    resolves the key via the shared constant, so if the producer hardcodes
    the string literal instead, a rename of the constant reaches the reader
    but NOT the producer. Nothing would go red — the harness cadence check
    at ``harness._maybe_escalate_stale_task_count_snapshot`` would simply
    read ``None`` (inconclusive) forever and the
    ``recon_stale_task_count_snapshot`` escalation would go silently dead
    rather than loudly wrong.

    Rebinding the constant on the *producer's* module namespace is what
    proves the coupling: it only takes effect if the producer looks the
    name up at call time. ``raising=True`` (the default) additionally pins
    that the producer imports the constant at all.

    Shares TestRunRecordsTaskCountSnapshotWrittenStat's harness above via the
    module-level ``mock_deps`` fixture and ``_fake_cli_result`` helper:
    super().run() executes for real via a patched run_stage_via_cli while
    the module-level freshness helper is patched directly.
    """

    SENTINEL_KEY = '__sentinel_written_key__'

    @pytest.mark.asyncio
    async def test_stat_is_emitted_under_the_rebound_constant(
        self, mock_deps, monkeypatch,
    ):
        monkeypatch.setattr(
            task_knowledge_sync_module,
            'SNAPSHOT_WRITTEN_STAT_KEY',
            self.SENTINEL_KEY,
        )

        stage = TaskKnowledgeSync(
            StageId.task_knowledge_sync,
            scope=_scope('dark_factory', '/tmp/test'),
            **mock_deps,
        )

        with (
            patch(
                'fused_memory.reconciliation.stages.base.run_stage_via_cli',
                new=AsyncMock(return_value=_fake_cli_result()),
            ),
            patch(
                'fused_memory.reconciliation.stages.task_knowledge_sync'
                '._verify_task_count_snapshot_written',
                new=AsyncMock(return_value=True),
            ),
        ):
            report = await stage.run(
                events=[], watermark=Watermark(project_id='dark_factory'),
                prior_reports=[], run_id='run-snap-constant-driven',
            )

        assert report.stats[self.SENTINEL_KEY] == 1
        # The real spelling must NOT also appear — that would mean the
        # producer still carries a hardcoded literal alongside the constant.
        assert SNAPSHOT_WRITTEN_STAT_KEY not in report.stats


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

        assert report.stats[SNAPSHOT_WRITTEN_STAT_KEY] == 1
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

        assert report.stats[SNAPSHOT_WRITTEN_STAT_KEY] == 1
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

        assert report.stats[SNAPSHOT_WRITTEN_STAT_KEY] == 0
        mock_write.assert_not_awaited()
        mock_verify.assert_awaited_once()


# ---------------------------------------------------------------------------
# TaskKnowledgeSync.run() wiring — live-cycle prune observability (task 2646
# step-7/8)
# ---------------------------------------------------------------------------


class TestRunSurfacesPruneObservability:
    """run() surfaces the REAL (unpatched) prune's enumerate/delete/
    enumeration_ok counts in report.stats (task 2646).

    Unlike TestRunDeterministicSnapshotWrite, this class does NOT patch
    _write_task_count_snapshot -- it lets the real _write_task_count_snapshot
    -> _prune_task_count_snapshots call chain run inside a live stage.run()
    cycle. This is the only kind of test that would have caught the
    incident this task exists to guard against: the prune enumerating
    nothing at runtime while everything looked correct in code review
    (the existing run()-level tests all patch the write out, so the real
    prune never executes under test).
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

    @staticmethod
    def _seed_snapshot_records(count):
        return [
            {
                'id': f'snap-{i}',
                'created_at': '2026-07-01T00:00:00+00:00',
                'metadata': {'kind': TASK_COUNT_SNAPSHOT_KIND},
            }
            for i in range(count)
        ]

    @pytest.mark.asyncio
    async def test_live_prune_success_surfaces_actual_counts(self, mock_deps):
        """Case (a): the real prune enumerates and deletes 2 seeded stale
        snapshots; report.stats reflects the actual enumerated/deleted
        counts and enumeration_ok=1."""
        seeded = self._seed_snapshot_records(2)

        def _get_memories_by_metadata(*, project_id, filters, **kwargs):
            # Isolates the prune's counts from _sweep_stale_persistence_markers,
            # which shares this same method with a different filter shape
            # within a single run() cycle (design_decisions, task 2646 plan).
            if filters == {'kind': TASK_COUNT_SNAPSHOT_KIND}:
                return seeded
            return []

        mock_deps['memory_service'].get_memories_by_metadata = AsyncMock(
            side_effect=_get_memories_by_metadata,
        )

        stage = TaskKnowledgeSync(
            StageId.task_knowledge_sync,
            scope=_scope('reify', '/tmp/test'),
            **mock_deps,
        )

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=self._fake_cli_result()),
        ):
            report = await stage.run(
                events=[], watermark=Watermark(project_id='reify'),
                prior_reports=[], run_id='run-prune-live-ok',
            )

        assert report.stats['task_count_snapshot_prune_enumerated'] == 2
        assert report.stats[SNAPSHOT_PRUNED_STAT_KEY] == 2
        assert report.stats['task_count_snapshot_prune_enumeration_ok'] == 1
        mock_deps['memory_service'].add_memory.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_live_silent_enumeration_failure_surfaces_as_not_ok(self, mock_deps):
        """Case (b), the incident fingerprint: enumeration RAISES inside the
        real prune. report.stats must show enumeration_ok=0 / enumerated=0
        -- runtime-observably distinct from a genuine empty result -- and
        (task 2655) the canonical add_memory write must now be SKIPPED
        rather than proceeding and adding another duplicate on top of an
        unprunable pile. This supersedes task 2646's pinned
        write-proceeds-on-enumeration-failure behavior."""

        def _get_memories_by_metadata(*, project_id, filters, **kwargs):
            if filters == {'kind': TASK_COUNT_SNAPSHOT_KIND}:
                raise RuntimeError('mem0 down')
            return []

        mock_deps['memory_service'].get_memories_by_metadata = AsyncMock(
            side_effect=_get_memories_by_metadata,
        )

        stage = TaskKnowledgeSync(
            StageId.task_knowledge_sync,
            scope=_scope('reify', '/tmp/test'),
            **mock_deps,
        )

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=self._fake_cli_result()),
        ):
            report = await stage.run(
                events=[], watermark=Watermark(project_id='reify'),
                prior_reports=[], run_id='run-prune-live-fail',
            )

        assert report.stats['task_count_snapshot_prune_enumeration_ok'] == 0
        assert report.stats['task_count_snapshot_prune_enumerated'] == 0
        mock_deps['memory_service'].add_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_live_timeout_fingerprint_skips_write_no_duplicate_added(self, mock_deps):
        """Case (c), task 2655's core fix: the scroll comes back EMPTY
        (no exception -- the swallowed-timeout shape) while the count
        cross-check reports 2 existing snapshots. The real prune must
        surface enumeration_ok=0, and the canonical write must be skipped
        entirely -- no duplicate added on top of the un-enumerable pile."""

        def _get_memories_by_metadata(*, project_id, filters, **kwargs):
            if filters == {'kind': TASK_COUNT_SNAPSHOT_KIND}:
                return []
            return []

        mock_deps['memory_service'].get_memories_by_metadata = AsyncMock(
            side_effect=_get_memories_by_metadata,
        )
        mock_deps['memory_service'].count_memories_by_metadata.return_value = 2

        stage = TaskKnowledgeSync(
            StageId.task_knowledge_sync,
            scope=_scope('reify', '/tmp/test'),
            **mock_deps,
        )

        with patch(
            'fused_memory.reconciliation.stages.base.run_stage_via_cli',
            new=AsyncMock(return_value=self._fake_cli_result()),
        ):
            report = await stage.run(
                events=[], watermark=Watermark(project_id='reify'),
                prior_reports=[], run_id='run-prune-live-timeout-fingerprint',
            )

        assert report.stats['task_count_snapshot_prune_enumeration_ok'] == 0
        mock_deps['memory_service'].add_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_live_timeout_fingerprint_skips_verify_fallback(self, mock_deps):
        """Task 2655 step-5/6: when the prune reports enumeration_ok=0, run()
        must NOT fall back to _verify_task_count_snapshot_written -- that
        helper's own scroll swallows timeouts the same way and would
        mis-read the empty page as a CONFIRMED miss (False), spuriously
        growing the harness's consecutive-miss streak
        (_maybe_escalate_stale_task_count_snapshot). report.stats must
        instead leave SNAPSHOT_WRITTEN_STAT_KEY absent (inconclusive)."""

        def _get_memories_by_metadata(*, project_id, filters, **kwargs):
            if filters == {'kind': TASK_COUNT_SNAPSHOT_KIND}:
                return []
            return []

        mock_deps['memory_service'].get_memories_by_metadata = AsyncMock(
            side_effect=_get_memories_by_metadata,
        )
        mock_deps['memory_service'].count_memories_by_metadata.return_value = 2

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
                '._verify_task_count_snapshot_written',
                new=AsyncMock(),
            ) as mock_verify,
        ):
            report = await stage.run(
                events=[], watermark=Watermark(project_id='reify'),
                prior_reports=[], run_id='run-prune-live-timeout-no-verify',
            )

        assert SNAPSHOT_WRITTEN_STAT_KEY not in report.stats
        mock_verify.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_taskmaster_none_still_falls_back_to_verify(self, mock_deps):
        """Contrast case: when the prune never ran at all (no taskmaster,
        so _write_task_count_snapshot early-returns before ever calling
        _prune_task_count_snapshots), SNAPSHOT_PRUNE_ENUMERATION_OK_STAT_KEY
        is absent from report.stats entirely (not 0) -- the new step-6
        guard must read that as "unknown", not "confirmed failed", and
        still fall back to verify exactly as before this task."""
        mock_deps['taskmaster'] = None

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
                '._verify_task_count_snapshot_written',
                new=AsyncMock(return_value=True),
            ) as mock_verify,
        ):
            report = await stage.run(
                events=[], watermark=Watermark(project_id='reify'),
                prior_reports=[], run_id='run-no-taskmaster-verify-fallback',
            )

        mock_verify.assert_awaited_once()
        assert report.stats[SNAPSHOT_WRITTEN_STAT_KEY] == 1


# ---------------------------------------------------------------------------
# Stage-3 prompt coupling — the stat-vs-edge guidance names the LIVE keys
# (task 3045 step-7/8)
# ---------------------------------------------------------------------------


class TestStage3PromptNamesMem0SnapshotStats:
    """STAGE3_SYSTEM_PROMPT must name the stat keys Stage 3 actually receives.

    _format_report dumps the WHOLE stats dict verbatim
    (``json.dumps(report.stats)``) into Stage 3's payload, so the renamed
    keys land in front of the model unfiltered. The rename fixes the NAME;
    the model still has to be told the invariant, or it can reconstruct the
    same "stat claims a write but there is no Graphiti edge" false positive
    from a differently-spelled key.

    Deliberately a COUPLING test, not a prose pin, following the established
    assembled-prompt drift-guard convention (tests/test_recon_report_guidance_drift.py,
    tests/test_standing_decision_prompt_drift.py): every assertion is keyed
    off the imported constants, so a future rename cannot orphan the
    guidance — leaving the prompt warning about a key the model never sees
    while the false positive quietly returns.

    The contract is PRESENCE-ONLY. No assertion is made about wording,
    sentence structure, section ordering, or what the prompt must NOT say —
    the prompt stays free to mention the legacy spelling if a future cycle
    wants to explain the rename to the model during the back-compat window
    that extract_snapshot_written's legacy fallback keeps open.
    """

    def test_prompt_names_the_written_stat_key(self):
        from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT

        assert SNAPSHOT_WRITTEN_STAT_KEY in STAGE3_SYSTEM_PROMPT

    def test_prompt_names_the_pruned_stat_key(self):
        from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT

        assert SNAPSHOT_PRUNED_STAT_KEY in STAGE3_SYSTEM_PROMPT
