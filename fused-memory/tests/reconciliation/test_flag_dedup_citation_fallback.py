"""Tests for PRD γ cutover: compute_flag_signature cited_tasks fallback."""

from __future__ import annotations

import pytest

from fused_memory.reconciliation.flag_dedup import compute_flag_signature


class TestComputeFlagSignatureFallback:
    """compute_flag_signature must fall back to cited_tasks[0].task_id when
    top-level task_id is absent (PRD γ step-18).

    RED: the current implementation only reads top-level task_id and returns
    None when it is missing, even if cited_tasks carries a valid task_id.
    Step-18 adds the fallback.
    """

    def test_top_level_task_id_preserved(self):
        """Normal path: top-level task_id + flag_type → signature unchanged."""
        result = compute_flag_signature({'task_id': '42', 'flag_type': 'orphaned'})
        assert result == ('42', 'orphaned')

    def test_cited_tasks_fallback(self):
        """When top-level task_id is absent, fall back to cited_tasks[0].task_id."""
        flag = {
            'flag_type': 'orphaned',
            'cited_tasks': [
                {'project_id': 'p', 'task_id': '7', 'title': 't'},
            ],
        }
        result = compute_flag_signature(flag)
        assert result == ('7', 'orphaned'), (
            f'Expected fallback to cited_tasks[0].task_id="7", got {result!r}'
        )

    def test_empty_cited_tasks_returns_none(self):
        """When top-level task_id is absent and cited_tasks is empty → None."""
        result = compute_flag_signature({'flag_type': 'orphaned', 'cited_tasks': []})
        assert result is None

    def test_missing_both_returns_none(self):
        """When neither task_id nor cited_tasks is present → None."""
        result = compute_flag_signature({})
        assert result is None

    def test_missing_flag_type_returns_none(self):
        """flag_type is still required even with a valid cited_tasks fallback."""
        flag = {
            'cited_tasks': [{'project_id': 'p', 'task_id': '7', 'title': 't'}],
        }
        result = compute_flag_signature(flag)
        assert result is None

    def test_integer_task_id_coerced_to_str(self):
        """Integer task_id is coerced to str (existing behaviour preserved)."""
        result = compute_flag_signature({'task_id': 42, 'flag_type': 'orphaned'})
        assert result == ('42', 'orphaned')

    def test_cited_tasks_fallback_integer_task_id(self):
        """cited_tasks fallback also coerces integer task_id to str."""
        flag = {
            'flag_type': 'orphaned',
            'cited_tasks': [{'project_id': 'p', 'task_id': 99, 'title': 't'}],
        }
        result = compute_flag_signature(flag)
        assert result == ('99', 'orphaned')
