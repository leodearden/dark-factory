"""Tests for stage1_stall_detector module — task 1201.

Covers:
- TestExtractHumanOperatorTaskIds  (step-1/2)
- TestTrackHumanOperatorStalls     (step-3/4)
- TestComputeStalledTaskIds        (step-5/6)
- TestMaybeEscalateStalledTasks    (step-7/8)
"""

from __future__ import annotations

import pytest

from fused_memory.reconciliation.stage1_stall_detector import (
    STAGE1_HUMAN_OPERATOR_STALL_THRESHOLD,
    _STAGE1_HUMAN_OPERATOR_STALL_MARKER_SOURCE,
    extract_human_operator_task_ids,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Assert module-level constants have expected values."""

    def test_threshold_is_five(self):
        assert STAGE1_HUMAN_OPERATOR_STALL_THRESHOLD == 5

    def test_marker_source_value(self):
        assert _STAGE1_HUMAN_OPERATOR_STALL_MARKER_SOURCE == 'stage1_human_operator_stall_marker'


# ---------------------------------------------------------------------------
# extract_human_operator_task_ids
# ---------------------------------------------------------------------------


class TestExtractHumanOperatorTaskIds:
    """extract_human_operator_task_ids filters and deduplicates task_ids."""

    def test_returns_task_ids_of_hor_flags(self):
        """(a) Returns task_ids where resolution_status == 'human_operator_required'."""
        flags = [
            {'task_id': '42', 'flag_type': 'assumption_invalid', 'resolution_status': 'human_operator_required'},
        ]
        result = extract_human_operator_task_ids(flags)
        assert result == ['42']

    def test_skips_other_resolution_statuses(self):
        """(b) Flags with other resolution_status values are skipped."""
        flags = [
            {'task_id': '1', 'resolution_status': 'automated'},
            {'task_id': '2', 'resolution_status': 'agent_retry'},
            {'task_id': '3', 'resolution_status': 'human_operator_required'},
        ]
        result = extract_human_operator_task_ids(flags)
        assert result == ['3']

    def test_skips_flags_without_resolution_status(self):
        """(c) Flags missing the resolution_status key are skipped."""
        flags = [
            {'task_id': '7', 'flag_type': 'missing_deliverable'},
            {'task_id': '8', 'resolution_status': 'human_operator_required'},
        ]
        result = extract_human_operator_task_ids(flags)
        assert result == ['8']

    def test_deduplicates_same_task_id_preserving_first_seen_order(self):
        """(d) Multiple flags for the same task_id deduplicate, preserving first-seen order."""
        flags = [
            {'task_id': '5', 'flag_type': 'A', 'resolution_status': 'human_operator_required'},
            {'task_id': '3', 'flag_type': 'B', 'resolution_status': 'human_operator_required'},
            {'task_id': '5', 'flag_type': 'C', 'resolution_status': 'human_operator_required'},
        ]
        result = extract_human_operator_task_ids(flags)
        # '5' seen first, '3' second; '5' not duplicated
        assert result == ['5', '3']

    def test_coerces_int_task_id_to_str(self):
        """(e) Int task_id is coerced to str; int 0 and str '0' collapse."""
        flags = [
            {'task_id': 0, 'resolution_status': 'human_operator_required'},
            {'task_id': '0', 'resolution_status': 'human_operator_required'},
        ]
        result = extract_human_operator_task_ids(flags)
        assert result == ['0']

    def test_ignores_flags_with_none_task_id(self):
        """(f) Flags with task_id=None are skipped."""
        flags = [
            {'task_id': None, 'resolution_status': 'human_operator_required'},
            {'task_id': '99', 'resolution_status': 'human_operator_required'},
        ]
        result = extract_human_operator_task_ids(flags)
        assert result == ['99']

    def test_ignores_flags_with_missing_task_id(self):
        """(f) Flags missing the task_id key entirely are skipped."""
        flags = [
            {'resolution_status': 'human_operator_required'},
            {'task_id': '10', 'resolution_status': 'human_operator_required'},
        ]
        result = extract_human_operator_task_ids(flags)
        assert result == ['10']

    def test_empty_input_returns_empty_list(self):
        """(g) Empty input returns []."""
        assert extract_human_operator_task_ids([]) == []

    def test_multiple_hor_flags_in_order(self):
        """Multiple distinct HOR task_ids are returned in input order."""
        flags = [
            {'task_id': 'b', 'resolution_status': 'human_operator_required'},
            {'task_id': 'a', 'resolution_status': 'human_operator_required'},
            {'task_id': 'c', 'resolution_status': 'human_operator_required'},
        ]
        result = extract_human_operator_task_ids(flags)
        assert result == ['b', 'a', 'c']
