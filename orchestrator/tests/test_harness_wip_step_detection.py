"""Tests for surfacing already-committed WIP safety-commits to the implementer (task 2051).

The harness auto-commits uncommitted work as a safety net before several
rebase/requeue/reclaim operations (see git_ops.py and workflow.py). Any of
these can land a still-"pending" plan step's complete implementation at
branch HEAD *before* mark_step_done is called for that step, so every
session has had to re-discover the workaround ad-hoc. This suite covers the
new detection + prompt-surfacing machinery that fixes that:

  - is_wip_safety_commit / WIP_SAFETY_COMMIT_PREFIXES (git_ops.py)
  - GitOps.get_commit_subjects (git_ops.py)
  - TaskWorkflow._detect_tip_wip_commits (workflow.py)
  - BriefingAssembler.build_implementer_prompt's wip_notice rendering (briefing.py)
  - _execute_iterations wiring the detector into the prompt builder (workflow.py)
"""

from __future__ import annotations

from orchestrator.git_ops import WIP_SAFETY_COMMIT_PREFIXES, is_wip_safety_commit

# ---------------------------------------------------------------------------
# step-1 RED: is_wip_safety_commit predicate + WIP_SAFETY_COMMIT_PREFIXES
# ---------------------------------------------------------------------------


class TestIsWipSafetyCommit:
    """Pins the predicate to the exact literal subjects the harness produces.

    Producing sites (must stay recognized):
      - workflow.py:4579  'chore: save WIP before inter-iteration rebase'
      - git_ops.py:1134/2496 'chore: save WIP before requeue rebase'
      - git_ops.py:2050  'chore: save WIP before warm-lane reclaim (task 1933)'
    """

    def test_recognizes_requeue_rebase_literal(self):
        assert is_wip_safety_commit('chore: save WIP before requeue rebase') is True

    def test_recognizes_inter_iteration_rebase_literal(self):
        assert is_wip_safety_commit('chore: save WIP before inter-iteration rebase') is True

    def test_recognizes_warm_lane_reclaim_literal(self):
        assert is_wip_safety_commit(
            'chore: save WIP before warm-lane reclaim (task 1933)',
        ) is True

    def test_rejects_normal_feat_commit(self):
        assert is_wip_safety_commit('feat: GREEN — x') is False

    def test_rejects_normal_test_commit(self):
        assert is_wip_safety_commit('test: RED — y') is False

    def test_rejects_empty_string(self):
        assert is_wip_safety_commit('') is False

    def test_rejects_near_miss_subject(self):
        assert is_wip_safety_commit('chore: save the world') is False

    def test_prefixes_tuple_matches_producing_literal(self):
        """WIP_SAFETY_COMMIT_PREFIXES is the single source of truth for the prefix."""
        assert WIP_SAFETY_COMMIT_PREFIXES == ('chore: save WIP before ',)
