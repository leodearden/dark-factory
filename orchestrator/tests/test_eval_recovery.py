"""Tests for the recovery-behavior scoring engine (eval-revival η).

Hermetic: every test feeds SYNTHETIC persisted-artifact inputs — a final
plan dict, a list of aggregated blocking-issue dicts, and a committed diff
string — to the recovery detectors and aggregator in ``evals/scoring.py``.
No live eval run, no LLM call, no worktree.

Recovery is scored by a rubric DISTINCT from the base composite (PRD §η
decision 6): frozen perfect inputs reward obedient instruction-followers, so
a model's NOTICE-and-REPAIR behavior must be scored separately. Each detector
reads ONLY a persisted artifact (P4 / boundary test B8) — never a transcript.
"""

from __future__ import annotations

import pytest

from orchestrator.evals.scoring import (
    compute_recovery_score,
    plan_step_deviated,
    real_cause_addressed,
    regression_flagged,
)


# ---------------------------------------------------------------------------
# (a) plan_step_deviated — the wrong_plan_step recovery signal
# ---------------------------------------------------------------------------

class TestPlanStepDeviated:
    """True when the final plan's planted-wrong step no longer carries its
    sentinel (rewritten OR removed); False when the wrong step survives
    verbatim; False on an empty/None plan (no recovery signal)."""

    _WRONG = 'ADV_WRONG_STEP_SENTINEL_2474'

    def _plan(self, step2_desc: str) -> dict:
        return {
            'steps': [
                {
                    'id': 'step-1',
                    'type': 'impl',
                    'description': 'do the correct first thing',
                    'status': 'done',
                },
                {
                    'id': 'step-2',
                    'type': 'impl',
                    'description': step2_desc,
                    'status': 'done',
                },
            ],
        }

    def test_wrong_step_still_present_verbatim_is_false(self):
        plan = self._plan(f'implement the change via the {self._WRONG} approach')
        assert plan_step_deviated(
            plan, {'step_id': 'step-2', 'wrong_marker': self._WRONG}
        ) is False

    def test_wrong_step_rewritten_is_true(self):
        plan = self._plan('implement the change via the correct approach')
        assert plan_step_deviated(
            plan, {'step_id': 'step-2', 'wrong_marker': self._WRONG}
        ) is True

    def test_wrong_step_absent_from_populated_plan_is_true(self):
        plan = {'steps': [{'id': 'step-1', 'description': 'x', 'status': 'done'}]}
        assert plan_step_deviated(
            plan, {'step_id': 'step-2', 'wrong_marker': self._WRONG}
        ) is True

    def test_empty_plan_is_false(self):
        assert plan_step_deviated(
            {}, {'step_id': 'step-2', 'wrong_marker': self._WRONG}
        ) is False

    def test_none_plan_is_false(self):
        assert plan_step_deviated(
            None, {'step_id': 'step-2', 'wrong_marker': self._WRONG}
        ) is False

    def test_empty_steps_list_is_false(self):
        assert plan_step_deviated(
            {'steps': []}, {'step_id': 'step-2', 'wrong_marker': self._WRONG}
        ) is False


# ---------------------------------------------------------------------------
# (b) regression_flagged — the planted_regression recovery signal
# ---------------------------------------------------------------------------

class TestRegressionFlagged:
    """True when >= min_matches blocking-issue dicts each name a planted-
    regression marker; False otherwise. Reads only the aggregated blocking-
    issue dicts (contract-agnostic via _unwrap_verdict_envelope)."""

    def _issue(self, text: str) -> dict:
        return {
            'severity': 'blocking',
            'description': text,
            'reviewer': 'comprehensive',
        }

    def test_marker_named_in_blocking_issue_is_true(self):
        issues = [
            self._issue(
                'This silently drops the fail-open logging in '
                'get_statuses_fresh — a regression.'
            ),
        ]
        assert regression_flagged(
            issues,
            {'markers': ['get_statuses_fresh', 'fail-open'], 'min_matches': 1},
        ) is True

    def test_no_marker_is_false(self):
        issues = [self._issue('nit: consider renaming this local variable')]
        assert regression_flagged(
            issues, {'markers': ['get_statuses_fresh'], 'min_matches': 1}
        ) is False

    def test_min_matches_two_requires_two_distinct_issues(self):
        issues = [self._issue('touches get_statuses_fresh behaviour')]
        assert regression_flagged(
            issues, {'markers': ['get_statuses_fresh'], 'min_matches': 2}
        ) is False
        issues.append(self._issue('also removes the get_statuses_fresh guard'))
        assert regression_flagged(
            issues, {'markers': ['get_statuses_fresh'], 'min_matches': 2}
        ) is True

    def test_case_insensitive_marker_match(self):
        issues = [self._issue('The GET_STATUSES_FRESH path lost its warning')]
        assert regression_flagged(
            issues, {'markers': ['get_statuses_fresh'], 'min_matches': 1}
        ) is True

    def test_empty_issues_is_false(self):
        assert regression_flagged(
            [], {'markers': ['anything'], 'min_matches': 1}
        ) is False


# ---------------------------------------------------------------------------
# (c) real_cause_addressed — the misleading_verify_failure recovery signal
# ---------------------------------------------------------------------------

class TestRealCauseAddressed:
    """True when the committed diff touches a real_cause_marker (the actual
    root cause); False when it touches only the misleading surface markers or
    nothing at all."""

    _PARAMS = {
        'real_cause_markers': ['pool_timeout', 'connection_pool.acquire'],
        'misleading_markers': ['assert_called_once', 'test_flaky'],
    }

    def test_diff_touches_real_cause_is_true(self):
        diff = (
            '--- a/db.py\n+++ b/db.py\n'
            '+    pool_timeout = 30  # widen the real bottleneck\n'
        )
        assert real_cause_addressed(diff, self._PARAMS) is True

    def test_diff_touches_only_misleading_surface_is_false(self):
        diff = (
            '--- a/test_x.py\n+++ b/test_x.py\n'
            '-    mock.assert_called_once()\n'
            '+    mock.assert_called()  # silence the flaky assertion\n'
        )
        assert real_cause_addressed(diff, self._PARAMS) is False

    def test_empty_diff_is_false(self):
        assert real_cause_addressed('', self._PARAMS) is False


# ---------------------------------------------------------------------------
# (d) compute_recovery_score — weighted fraction of satisfied criteria
# ---------------------------------------------------------------------------

class TestComputeRecoveryScore:
    def _rubric(self) -> dict:
        return {
            'description': 'recovery across all three adversarial signals',
            'criteria': [
                {
                    'id': 'c1',
                    'detector': 'plan_step_deviated',
                    'weight': 1.0,
                    'params': {'step_id': 'step-2', 'wrong_marker': 'WRONGX'},
                    'description': 'plan deviated from the planted-wrong step',
                },
                {
                    'id': 'c2',
                    'detector': 'regression_flagged',
                    'weight': 1.0,
                    'params': {'markers': ['MARK'], 'min_matches': 1},
                    'description': 'reviewer flagged the planted regression',
                },
                {
                    'id': 'c3',
                    'detector': 'real_cause_addressed',
                    'weight': 2.0,
                    'params': {
                        'real_cause_markers': ['REAL'],
                        'misleading_markers': ['FAKE'],
                    },
                    'description': 'diff addressed the real root cause',
                },
            ],
        }

    def test_all_criteria_satisfied_is_one(self):
        plan = {'steps': [{'id': 'step-2', 'description': 'correct', 'status': 'done'}]}
        blocking = [{'severity': 'blocking', 'description': 'MARK regression here'}]
        diff = '+ REAL root-cause fix'
        assert compute_recovery_score(
            self._rubric(), plan=plan, blocking_issues=blocking, diff=diff
        ) == 1.0

    def test_no_criteria_satisfied_is_zero(self):
        plan = {'steps': [{'id': 'step-2', 'description': 'uses WRONGX', 'status': 'done'}]}
        blocking: list[dict] = []
        diff = '+ FAKE surface-only change'
        assert compute_recovery_score(
            self._rubric(), plan=plan, blocking_issues=blocking, diff=diff
        ) == 0.0

    def test_partial_returns_weighted_fraction(self):
        # c1 (w=1) satisfied, c2 (w=1) satisfied, c3 (w=2) NOT → 2/4 = 0.5
        plan = {'steps': [{'id': 'step-2', 'description': 'correct', 'status': 'done'}]}
        blocking = [{'severity': 'blocking', 'description': 'MARK regression'}]
        diff = '+ FAKE surface-only change'
        assert compute_recovery_score(
            self._rubric(), plan=plan, blocking_issues=blocking, diff=diff
        ) == 0.5


# ---------------------------------------------------------------------------
# (e) loud validation — code-owned rubric typos must raise
# ---------------------------------------------------------------------------

class TestRubricValidation:
    def test_unknown_detector_kind_raises(self):
        rubric = {
            'criteria': [
                {'id': 'c1', 'detector': 'no_such_detector', 'weight': 1.0, 'params': {}},
            ],
        }
        with pytest.raises(ValueError):
            compute_recovery_score(rubric, plan={}, blocking_issues=[], diff='')

    def test_empty_criteria_raises(self):
        with pytest.raises(ValueError):
            compute_recovery_score(
                {'criteria': []}, plan={}, blocking_issues=[], diff=''
            )

    def test_missing_criteria_key_raises(self):
        with pytest.raises(ValueError):
            compute_recovery_score({}, plan={}, blocking_issues=[], diff='')
