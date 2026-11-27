"""C4's settle and scoring rules as pure functions (plans/live-shadow-eval-prd.md C4)."""

from __future__ import annotations

import pytest

from orchestrator.evals.judge import score_plan_structure
from orchestrator.evals.metrics import DECLINE_KINDS, EvalMetrics, compute_composite
from orchestrator.shadow_scoring import (
    ArchitectScore,
    ImplementerScore,
    ShadowReference,
    consequence_leg_warranted,
    score_architect_cell,
    score_implementer_cell,
)
from orchestrator.shadow_types import ReferenceKind

_LANDED = ShadowReference.landed('b' * 40)
_NONE = ShadowReference.none()

_PLAN = {
    'steps': [
        {'id': 'step-1', 'type': 'test'},
        {'id': 'step-2', 'type': 'impl'},
    ],
    'files': ['orchestrator/src/orchestrator/shadow_eval.py'],
}


def _passing(**overrides) -> EvalMetrics:
    fields = {
        'tests_pass': True,
        'lint_clean': True,
        'typecheck_clean': True,
        'plan_steps': 4,
        'review_blocking_issues': 1,
    }
    fields.update(overrides)
    return EvalMetrics(**fields)


class TestShadowReference:
    def test_landed_carries_its_sha(self):
        assert (_LANDED.kind, _LANDED.sha) == (ReferenceKind.LANDED, 'b' * 40)

    def test_none_carries_no_sha(self):
        assert (_NONE.kind, _NONE.sha) == (ReferenceKind.NONE, None)

    @pytest.mark.parametrize('sha', ['', '   ', None])
    def test_a_landed_reference_without_a_sha_is_refused(self, sha):
        with pytest.raises(ValueError, match='sha'):
            ShadowReference(ReferenceKind.LANDED, sha)

    def test_a_none_reference_with_a_sha_is_refused(self):
        with pytest.raises(ValueError, match='sha'):
            ShadowReference(ReferenceKind.NONE, 'b' * 40)


class TestScoreImplementerCell:
    @pytest.mark.parametrize(
        'metrics',
        [_passing(), _passing(tests_pass=False)],
        ids=['passing', 'failing'],
    )
    def test_landed_composite_is_compute_composite(self, metrics):
        score = score_implementer_cell(metrics, _LANDED)

        assert isinstance(score, ImplementerScore)
        assert score.reference_kind is ReferenceKind.LANDED
        assert score.composite_score == compute_composite(metrics)

    def test_a_failing_landed_run_scores_a_real_zero(self):
        assert score_implementer_cell(_passing(tests_pass=False), _LANDED).composite_score == 0.0

    @pytest.mark.parametrize(
        'metrics',
        [_passing(), _passing(tests_pass=False)],
        ids=['passing', 'failing'],
    )
    def test_unreferenced_composite_is_none_never_zero(self, metrics):
        score = score_implementer_cell(metrics, _NONE)

        assert score.composite_score is None
        assert score.reference_kind is ReferenceKind.NONE

    def test_unreferenced_still_reports_the_gates_and_review(self):
        metrics = _passing(lint_clean=False, review_blocking_issues=1)
        assert compute_composite(metrics) > 0

        score = score_implementer_cell(metrics, _NONE)

        assert (
            score.tests_pass,
            score.lint_clean,
            score.typecheck_clean,
            score.review_blocking_issues,
        ) == (True, False, True, 1)


def _architect(**overrides) -> EvalMetrics:
    fields = {'terminal_kind': 'planned', 'plan_steps': 2, 'plan_quality': 0.8}
    fields.update(overrides)
    return EvalMetrics(**fields)


class TestScoreArchitectCell:
    def test_landed_reports_the_reference_judged_plan_quality(self):
        score = score_architect_cell(_architect(), _LANDED, _PLAN)

        assert isinstance(score, ArchitectScore)
        assert score.reference_kind is ReferenceKind.LANDED
        assert score.terminal_kind == 'planned'
        assert score.plan_quality == 0.8
        assert score.plan_structure is None

    def test_landed_never_presents_a_plausibility_score_as_reference_judged(self):
        metrics = _architect(judged_without_reference=True)

        assert score_architect_cell(metrics, _LANDED, _PLAN).plan_quality is None

    def test_unreferenced_scores_plan_structure_only(self):
        score = score_architect_cell(_architect(), _NONE, _PLAN)

        assert score.plan_quality is None
        assert score.plan_structure == score_plan_structure(_PLAN)
        assert score_plan_structure(_PLAN) > 0

    def test_a_declined_leg_records_the_decline_and_scores_zero_structure(self):
        metrics = _architect(terminal_kind='already_done', plan_steps=0, plan_quality=None)

        score = score_architect_cell(metrics, _NONE, None)

        assert score.terminal_kind == 'already_done'
        assert score.plan_structure == 0.0


class TestConsequenceLegWarranted:
    def test_a_planned_leg_with_steps_warrants_leg_two(self):
        assert consequence_leg_warranted(_architect(plan_steps=3)) is True

    def test_a_planned_leg_with_no_steps_does_not(self):
        assert consequence_leg_warranted(_architect(plan_steps=0)) is False

    @pytest.mark.parametrize('kind', DECLINE_KINDS)
    def test_no_decline_warrants_leg_two_even_after_a_plan(self, kind):
        assert consequence_leg_warranted(_architect(terminal_kind=kind, plan_steps=6)) is False

    @pytest.mark.parametrize('kind', ['none', None])
    def test_an_unknown_terminal_does_not(self, kind):
        assert consequence_leg_warranted(_architect(terminal_kind=kind, plan_steps=6)) is False


@pytest.mark.parametrize(
    'call',
    [
        lambda m: score_implementer_cell(m, _LANDED),
        lambda m: score_implementer_cell(m, _NONE),
        lambda m: score_architect_cell(m, _LANDED, _PLAN),
        lambda m: score_architect_cell(m, _NONE, _PLAN),
        consequence_leg_warranted,
    ],
    ids=['impl-landed', 'impl-none', 'arch-landed', 'arch-none', 'leg-two'],
)
def test_the_rules_never_mutate_the_metrics(call):
    metrics = _architect(tests_pass=True, review_blocking_issues=2)
    before = metrics.to_dict()

    call(metrics)

    assert metrics.to_dict() == before
