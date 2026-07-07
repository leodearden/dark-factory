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
    SNAPSHOT_WRITTEN_STAT_KEY,
    TASK_COUNT_SNAPSHOT_KIND,
    compute_snapshot_miss_streak,
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
