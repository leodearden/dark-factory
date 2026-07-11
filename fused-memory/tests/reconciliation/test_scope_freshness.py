"""Tests for fused_memory.reconciliation.scope_freshness — task 2417.

Reconciliation freshness pre-check: skip re-derivation of unchanged
cross-project scope-correction threads.  Grown step-by-step per plan.json:

- TestIsCrossProjectScopeCorrection   (step-1/2)
- TestScopeSignature                  (step-3/4)
- TestBuildScopeSnapshotMetadata      (step-5/6)
- TestSnapshotFreshness               (step-7/8)
- TestPrecheckBootstrap               (step-9/10)
- TestPrecheckFreshSkip               (step-11/12)
- TestPrecheckChangedAndFailOpen      (step-13/14)
"""

from __future__ import annotations


class TestIsCrossProjectScopeCorrection:
    """Tests for is_cross_project_scope_correction(finding, project_id) -> bool."""

    def test_true_for_cross_project_flag_type_with_foreign_cited_task(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {
            'flag_type': 'cross_project',
            'cited_tasks': [
                {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'},
            ],
        }
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is True

    def test_true_for_cross_project_routing_category_with_foreign_cited_task(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {
            'category': 'cross_project_routing',
            'cited_tasks': [
                {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'},
            ],
        }
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is True

    def test_false_for_same_project_only_cited_tasks(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {
            'flag_type': 'cross_project',
            'cited_tasks': [
                {'project_id': 'autopilot_video', 'task_id': '540', 'title': 'x'},
            ],
        }
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is False

    def test_false_for_non_cross_project_flag_type(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {
            'flag_type': 'task_memory_mismatch',
            'cited_tasks': [
                {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'},
            ],
        }
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is False

    def test_false_for_memory_stale_category(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {
            'category': 'memory_stale',
            'cited_tasks': [
                {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'},
            ],
        }
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is False

    def test_false_when_no_cited_tasks(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {'flag_type': 'cross_project'}
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is False

    def test_false_when_cited_tasks_empty_list(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {'flag_type': 'cross_project', 'cited_tasks': []}
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is False

    def test_false_for_empty_finding_dict(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        assert is_cross_project_scope_correction({}, 'autopilot_video') is False

    def test_false_for_malformed_finding_none(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        assert is_cross_project_scope_correction(None, 'autopilot_video') is False

    def test_false_when_cited_tasks_entries_not_dicts(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {'flag_type': 'cross_project', 'cited_tasks': ['not-a-dict']}
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is False

    def test_false_when_cited_tasks_not_a_list(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {'flag_type': 'cross_project', 'cited_tasks': 'dark_factory:2405'}
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is False

    def test_true_when_one_of_several_cited_tasks_is_foreign(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {
            'flag_type': 'cross_project',
            'cited_tasks': [
                {'project_id': 'autopilot_video', 'task_id': '540', 'title': 'x'},
                {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'y'},
            ],
        }
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is True
