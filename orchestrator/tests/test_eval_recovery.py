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

import copy
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.evals import runner
from orchestrator.evals.metrics import EvalMetrics, collect_metrics
from orchestrator.evals.scoring import (
    _RECOVERY_DETECTORS,
    compute_recovery_score,
    plan_step_deviated,
    real_cause_addressed,
    regression_flagged,
)
from orchestrator.evals.task_sampler import (
    audit_fixture_corpus,
    stratification_counts,
)

TASKS_DIR = Path(runner.__file__).parent / 'tasks'


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


# ---------------------------------------------------------------------------
# EvalMetrics.recovery_score field + collect_metrics wiring (step-3/4)
#
# Hermetic collect_metrics harness: a MagicMock workflow whose artifacts return
# synthetic recovered artifacts, with run_verification / _git_diff_stats /
# snapshots.get_diff patched — no live workflow, LLM, or git.
# ---------------------------------------------------------------------------

# A synthetic 'recovered' artifact set that satisfies all three detectors.
_RECOVERED_PLAN = {
    'steps': [
        {'id': 'step-1', 'description': 'correct first step', 'status': 'done'},
        {'id': 'step-2', 'description': 'the corrected approach', 'status': 'done'},
    ],
}
_RECOVERED_BLOCKING = [
    {'severity': 'blocking', 'description': 'planted MARK regression is present'},
]
_RECOVERED_DIFF = '--- a/db.py\n+++ b/db.py\n+    REAL root-cause fix\n'

_FULL_RUBRIC = {
    'description': 'recovery across all three adversarial signals',
    'criteria': [
        {
            'id': 'c1', 'detector': 'plan_step_deviated', 'weight': 1.0,
            'params': {'step_id': 'step-2', 'wrong_marker': 'WRONGX'},
        },
        {
            'id': 'c2', 'detector': 'regression_flagged', 'weight': 1.0,
            'params': {'markers': ['MARK'], 'min_matches': 1},
        },
        {
            'id': 'c3', 'detector': 'real_cause_addressed', 'weight': 1.0,
            'params': {'real_cause_markers': ['REAL'], 'misleading_markers': ['FAKE']},
        },
    ],
}


def _wf_metrics() -> MagicMock:
    """A WorkflowMetrics stand-in with every numeric field collect_metrics reads."""
    wm = MagicMock()
    wm.total_duration_ms = 1000
    wm.total_output_tokens = 120
    wm.total_input_tokens = 300
    wm.total_cache_read_tokens = 0
    wm.total_cache_create_tokens = 0
    wm.total_cost_usd = 0.42
    wm.total_turns = 6
    wm.execute_iterations = 3
    wm.verify_attempts = 0
    wm.judge_invocations = 0
    wm.judge_cost_usd = 0.0
    wm.judge_early_exits = 0
    return wm


def _workflow(plan: dict, blocking_issues: list[dict]) -> MagicMock:
    wf = MagicMock()
    wf.metrics = _wf_metrics()
    wf.config.env_overrides = {}
    wf.config.max_execute_iterations = 20
    wf.artifacts.read_plan.return_value = plan
    reviews = MagicMock()
    reviews.blocking_issues = blocking_issues
    reviews.suggestions = []
    wf.artifacts.aggregate_reviews.return_value = reviews
    return wf


def _verify_pass() -> MagicMock:
    v = MagicMock()
    v.passed = True
    v.lint_output = ''
    v.type_output = ''
    return v


def _collect_patches(diff: str):
    """Patch the three collect_metrics side-effect boundaries (verify / diff-stat
    / recovery diff) so the run is hermetic."""
    return (
        patch(
            'orchestrator.verify.run_verification',
            AsyncMock(return_value=_verify_pass()),
        ),
        patch(
            'orchestrator.evals.metrics._git_diff_stats',
            AsyncMock(return_value=(10, 2)),
        ),
        patch(
            'orchestrator.evals.snapshots.get_diff',
            AsyncMock(return_value=diff),
        ),
    )


class TestEvalMetricsRecoveryField:
    def test_default_is_none(self):
        assert EvalMetrics().recovery_score is None

    def test_to_dict_carries_recovery_score_key_defaulting_none(self):
        d = EvalMetrics().to_dict()
        assert 'recovery_score' in d
        assert d['recovery_score'] is None


@pytest.mark.asyncio
class TestCollectMetricsRecoveryWiring:
    async def test_adversarial_task_populates_recovery_score(self):
        task = {
            'id': 'df_task_x_adv',
            'pre_task_commit': 'base123',
            'adversarial': {'recovery_rubric': _FULL_RUBRIC},
        }
        wf = _workflow(_RECOVERED_PLAN, _RECOVERED_BLOCKING)
        p_verify, p_stat, p_diff = _collect_patches(_RECOVERED_DIFF)
        with p_verify, p_stat, p_diff:
            m = await collect_metrics(wf, Path('/fake/wt'), task)
        # A real, populated float — not the None sentinel.
        assert m.recovery_score == 1.0
        assert isinstance(m.recovery_score, float)

    async def test_non_adversarial_task_leaves_recovery_none(self):
        task = {'id': 'df_task_ordinary', 'pre_task_commit': 'base123'}
        wf = _workflow(_RECOVERED_PLAN, [])
        p_verify, p_stat, p_diff = _collect_patches('')
        with p_verify, p_stat, p_diff:
            m = await collect_metrics(wf, Path('/fake/wt'), task)
        assert m.recovery_score is None

    async def test_recovery_failure_warns_and_nulls_without_nuking_composite(
        self, caplog,
    ):
        # Empty-criteria rubric → compute_recovery_score raises → collect_metrics
        # must WARN (naming the fixture) and leave recovery_score=None, while
        # STILL returning a valid composite_score.
        task = {
            'id': 'df_task_bad_adv',
            'pre_task_commit': 'base123',
            'adversarial': {'recovery_rubric': {'criteria': []}},
        }
        wf = _workflow(_RECOVERED_PLAN, [])  # 0 blocking → composite 1.0
        p_verify, p_stat, p_diff = _collect_patches('')
        with p_verify, p_stat, p_diff, caplog.at_level(
            logging.WARNING, logger='orchestrator.evals.metrics',
        ):
            m = await collect_metrics(wf, Path('/fake/wt'), task)
        assert m.recovery_score is None
        assert m.composite_score == 1.0
        warnings = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any('df_task_bad_adv' in w for w in warnings), warnings


# ---------------------------------------------------------------------------
# The 3 adversarial fixtures — schema + semantics (step-5/6)
#
# Loads the REAL evals/tasks/*_adv_*.json files. Each is derived from a real
# revival-ζ parent, tagged with its adversarial TYPE and a SEPARATE recovery
# rubric, and lives in a DISTINCT cohort so the revival-ζ audit band is
# untouched.
# ---------------------------------------------------------------------------

_ADV_FILENAMES = [
    'df_task_2430_adv_plan.json',
    'df_task_2284_adv_regression.json',
    'df_task_2339_adv_verify.json',
]
_ADV_TYPES = frozenset(
    {'wrong_plan_step', 'planted_regression', 'misleading_verify_failure'}
)
_ETA_COHORT = 'revival-eta-adversarial'


def _load_adv_fixtures() -> list[dict]:
    return [runner.load_task(TASKS_DIR / name) for name in _ADV_FILENAMES]


def _load_all_fixtures() -> list[dict]:
    return [runner.load_task(p) for p in sorted(TASKS_DIR.glob('*.json'))]


def _recovered_artifacts(fixture: dict) -> tuple[dict, list[dict], str]:
    """A synthetic artifact set that SATISFIES this fixture's single-criterion
    rubric — derived generically from the criterion's detector + params."""
    adv = fixture['adversarial']
    crit = adv['recovery_rubric']['criteria'][0]
    params = crit['params']
    if adv['type'] == 'wrong_plan_step':
        plan = copy.deepcopy(fixture['plan'])
        for step in plan['steps']:
            if step.get('id') == params['step_id']:
                step['description'] = 'the corrected approach — sentinel removed'
        return plan, [], ''
    if adv['type'] == 'planted_regression':
        marker = params['markers'][0]
        issue = {'severity': 'blocking',
                 'description': f'reviewer flags the {marker} regression'}
        return {}, [issue], ''
    if adv['type'] == 'misleading_verify_failure':
        marker = params['real_cause_markers'][0]
        return {}, [], f'--- a/x\n+++ b/x\n+ real fix in {marker}\n'
    raise AssertionError(adv['type'])


def _not_recovered_artifacts(fixture: dict) -> tuple[dict, list[dict], str]:
    """A synthetic artifact set that FAILS this fixture's rubric (score 0.0)."""
    adv = fixture['adversarial']
    crit = adv['recovery_rubric']['criteria'][0]
    params = crit['params']
    if adv['type'] == 'wrong_plan_step':
        # The fixture's own seeded plan still carries the wrong-step sentinel.
        return copy.deepcopy(fixture['plan']), [], ''
    if adv['type'] == 'planted_regression':
        return {}, [], ''  # no blocking issue names the marker
    if adv['type'] == 'misleading_verify_failure':
        mis = params['misleading_markers'][0]
        return {}, [], f'--- a/t\n+++ b/t\n+ touched only {mis}\n'
    raise AssertionError(adv['type'])


class TestAdversarialFixtures:
    def test_all_three_files_load(self):
        fixtures = _load_adv_fixtures()
        assert len(fixtures) == 3
        for fx in fixtures:
            assert fx.get('id')

    def test_each_has_valid_adversarial_block(self):
        for fx in _load_adv_fixtures():
            adv = fx.get('adversarial')
            assert isinstance(adv, dict), fx.get('id')
            assert adv['type'] in _ADV_TYPES, adv['type']
            # derived_from names a real revival-ζ parent present in tasks/
            parent_path = TASKS_DIR / f'{adv["derived_from"]}.json'
            assert parent_path.exists(), adv['derived_from']
            parent = runner.load_task(parent_path)
            assert parent['cohort'] == 'revival-zeta', adv['derived_from']
            rubric = adv['recovery_rubric']
            assert len(rubric['criteria']) >= 1
            for crit in rubric['criteria']:
                assert crit['detector'] in _RECOVERY_DETECTORS, crit['detector']

    def test_exactly_three_distinct_types_present(self):
        types = {fx['adversarial']['type'] for fx in _load_adv_fixtures()}
        assert types == _ADV_TYPES

    def test_cohort_is_eta_and_zeta_band_untouched(self):
        all_fixtures = _load_all_fixtures()
        adv = [fx for fx in all_fixtures if fx.get('cohort') == _ETA_COHORT]
        assert len(adv) == 3
        for fx in adv:
            assert fx['cohort'] == _ETA_COHORT
            assert fx['cohort'] != 'revival-zeta'
        # The revival-ζ band audit still reports its 14-fixture band ok — the
        # adversarial fixtures (distinct cohort) don't perturb it.
        zeta = [fx for fx in all_fixtures if fx.get('cohort') == 'revival-zeta']
        report = audit_fixture_corpus(zeta, ref_exists=None)
        assert report.ok, report.failures
        assert report.count == 14
        # Whole-corpus stratification classifies the adversarial fixtures without
        # raising, and counts every fixture.
        counts = stratification_counts(all_fixtures, cohort=None)
        assert counts.total == len(all_fixtures)

    def test_wrong_plan_step_fixture_embeds_sentinel_plan(self):
        fx = runner.load_task(TASKS_DIR / 'df_task_2430_adv_plan.json')
        assert fx['adversarial']['type'] == 'wrong_plan_step'
        plan = fx['plan']
        assert plan is not None
        assert plan.get('steps')
        crit = fx['adversarial']['recovery_rubric']['criteria'][0]
        wrong_marker = crit['params']['wrong_marker']
        # The planted-wrong step carries the sentinel verbatim in the seeded plan.
        import json as _json
        assert wrong_marker in _json.dumps(plan)

    def test_each_rubric_recovered_scores_one_not_recovered_zero(self):
        for fx in _load_adv_fixtures():
            rubric = fx['adversarial']['recovery_rubric']
            r_plan, r_block, r_diff = _recovered_artifacts(fx)
            assert compute_recovery_score(
                rubric, plan=r_plan, blocking_issues=r_block, diff=r_diff,
            ) == 1.0, fx['id']
            n_plan, n_block, n_diff = _not_recovered_artifacts(fx)
            assert compute_recovery_score(
                rubric, plan=n_plan, blocking_issues=n_block, diff=n_diff,
            ) == 0.0, fx['id']


# ---------------------------------------------------------------------------
# recovery_score report column — additive interim surface (step-7/8)
#
# A distinct per-(task_id, config_name) column μ/λ consume — NOT a change to
# the Elo build_report/format_markdown schema (the full C4 composite+recovery
# row is owned by λ).
# ---------------------------------------------------------------------------

def _adv_result() -> runner.EvalResult:
    return runner.EvalResult(
        task_id='df_task_2430_adv_plan',
        config_name='opus-high',
        outcome='done',
        metrics={'recovery_score': 1.0, 'composite_score': 0.5},
        worktree_path='/tmp/wt-adv',
    )


def _ordinary_result() -> runner.EvalResult:
    # No recovery_score key at all → the null sentinel path.
    return runner.EvalResult(
        task_id='df_task_1993',
        config_name='opus-high',
        outcome='done',
        metrics={'composite_score': 1.0},
        worktree_path='/tmp/wt-ord',
    )


class TestRecoveryReport:
    def test_build_recovery_report_rows(self):
        from orchestrator.evals.report import build_recovery_report

        report = build_recovery_report([_adv_result(), _ordinary_result()])
        rows = report['rows']
        by_task = {r['task_id']: r for r in rows}

        adv = by_task['df_task_2430_adv_plan']
        assert adv['recovery_score'] == 1.0
        assert adv['adversarial'] is True
        assert adv['config_name'] == 'opus-high'

        ordinary = by_task['df_task_1993']
        assert ordinary['recovery_score'] is None  # null sentinel
        assert ordinary['adversarial'] is False

    def test_format_recovery_table_renders_column(self):
        from orchestrator.evals.report import (
            build_recovery_report,
            format_recovery_table,
        )

        report = build_recovery_report([_adv_result(), _ordinary_result()])
        table = format_recovery_table(report)
        assert 'recovery_score' in table

        adv_line = next(
            ln for ln in table.splitlines() if 'df_task_2430_adv_plan' in ln
        )
        assert '1.0' in adv_line  # populated float for the adversarial row

        ord_line = next(
            ln for ln in table.splitlines() if 'df_task_1993' in ln
        )
        assert '-' in ord_line       # null sentinel rendered as '-'
        assert '1.0' not in ord_line  # ordinary row is NOT populated
