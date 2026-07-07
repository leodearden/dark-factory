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
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fused_memory.models.reconciliation import StageId, StageReport
from fused_memory.reconciliation.task_count_snapshot_cadence import (
    ESCALATION_CATEGORY,
    SNAPSHOT_WRITTEN_STAT_KEY,
    TASK_COUNT_SNAPSHOT_KIND,
    TASK_COUNT_SNAPSHOT_MISS_THRESHOLD,
    build_stale_snapshot_finding,
    compute_snapshot_miss_streak,
    evaluate_snapshot_cadence,
    extract_snapshot_written,
)


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
