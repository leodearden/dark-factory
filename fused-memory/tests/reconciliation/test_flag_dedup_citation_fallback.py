"""Tests for PRD γ cutover: compute_flag_signature cited_tasks fallback."""

from __future__ import annotations

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

    def test_multi_task_fallback_deterministic_sorted(self):
        """Multi-task findings produce the same signature regardless of citation order.

        Reviewer finding dedup_correctness: the prior implementation used only
        cited_tasks[0].task_id, so two findings about the same set of tasks in
        different order would produce different signatures and fail to dedup.
        The fix uses sorted(all task_ids), comma-joined.
        """
        flag_abc = {
            'flag_type': 'stale',
            'cited_tasks': [
                {'task_id': 'c', 'project_id': 'p', 'title': 'C'},
                {'task_id': 'a', 'project_id': 'p', 'title': 'A'},
                {'task_id': 'b', 'project_id': 'p', 'title': 'B'},
            ],
        }
        flag_bca = {
            'flag_type': 'stale',
            'cited_tasks': [
                {'task_id': 'b', 'project_id': 'p', 'title': 'B'},
                {'task_id': 'c', 'project_id': 'p', 'title': 'C'},
                {'task_id': 'a', 'project_id': 'p', 'title': 'A'},
            ],
        }
        sig_abc = compute_flag_signature(flag_abc)
        sig_bca = compute_flag_signature(flag_bca)
        assert sig_abc == sig_bca, (
            f'Same task set in different order must produce the same signature: '
            f'{sig_abc!r} != {sig_bca!r}'
        )
        # Sorted ids are a,b,c → comma-joined → "a,b,c"
        assert sig_abc == ('a,b,c', 'stale'), f'Unexpected: {sig_abc!r}'

    def test_multi_task_fallback_distinct_sets_differ(self):
        """Two findings with different cited_tasks sets produce different signatures."""
        flag1 = {
            'flag_type': 'stale',
            'cited_tasks': [
                {'task_id': '1', 'project_id': 'p', 'title': 'T1'},
                {'task_id': '2', 'project_id': 'p', 'title': 'T2'},
            ],
        }
        flag2 = {
            'flag_type': 'stale',
            'cited_tasks': [
                {'task_id': '1', 'project_id': 'p', 'title': 'T1'},
                {'task_id': '3', 'project_id': 'p', 'title': 'T3'},
            ],
        }
        assert compute_flag_signature(flag1) != compute_flag_signature(flag2)
