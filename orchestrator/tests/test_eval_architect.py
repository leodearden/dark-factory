"""Tests for the architect eval harness (eval-revival θ).

Hermetic: every test feeds SYNTHETIC inputs — synthetic produced-plan dicts,
mocked ``invoke_agent`` (an AsyncMock returning a fixed structured verdict),
synthetic ``EvalResult``s — with the git/worktree/LLM boundaries patched. No
paid LLM run, no live worktree in CI.

θ invokes the architect LIVE against ζ's fixtures, scores the produced plan
against the real landed diff plus a plan-quality rubric, and emits a
NON-sentinel per-fixture plan-quality score. Downstream roles are FROZEN when
scoring the architect (decision 8: noise isolation + token savings), so the
harness only ever runs the architect — never the implementer/verify/review.

Mirrors ``test_eval_recovery.py`` — the recovery-scoring (η) blueprint θ
follows precisely.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.evals.metrics import EvalMetrics


# ---------------------------------------------------------------------------
# Synthetic produced-plan builders (shared across the θ tests)
# ---------------------------------------------------------------------------

def _well_formed_plan() -> dict:
    """A produced plan satisfying every structural criterion.

    Alternating test/impl TDD steps, files declared, design decisions, reuse
    items, and the ``_finalized_at`` completeness marker the architect's
    ``confirm_plan`` stamps.
    """
    return {
        'steps': [
            {'id': 'step-1', 'type': 'test', 'description': 'RED test for X',
             'status': 'done'},
            {'id': 'step-2', 'type': 'impl', 'description': 'GREEN implement X',
             'status': 'done'},
            {'id': 'step-3', 'type': 'test', 'description': 'RED test for Y',
             'status': 'done'},
            {'id': 'step-4', 'type': 'impl', 'description': 'GREEN implement Y',
             'status': 'done'},
        ],
        'files': ['pkg/mod.py', 'tests/test_mod.py'],
        'design_decisions': [
            {'decision': 'do X the clean way', 'rationale': 'because Y'},
        ],
        'reuse': [
            {'what': 'existing helper', 'where': 'pkg/util.py', 'how': 'call it'},
        ],
        '_finalized_at': '2026-07-19T00:00:00+00:00',
    }


def _degenerate_plan() -> dict:
    """A minimal plan: one lonely impl step, no alternation, no metadata."""
    return {
        'steps': [
            {'id': 'step-1', 'type': 'impl', 'description': 'do everything at once'},
        ],
    }


# ---------------------------------------------------------------------------
# EvalMetrics.plan_quality / role_under_test fields (step-1/2)
#
# The θ analogues of recovery_score/adversarial: plan_quality is None (the
# non-architect sentinel) for ordinary implementer runs and a populated float
# only for role_under_test=='architect' runs; role_under_test defaults None.
# Mirrors TestEvalMetricsRecoveryField in test_eval_recovery.py.
# ---------------------------------------------------------------------------

class TestEvalMetricsPlanQualityField:
    def test_plan_quality_default_is_none(self):
        assert EvalMetrics().plan_quality is None

    def test_role_under_test_default_is_none(self):
        assert EvalMetrics().role_under_test is None

    def test_to_dict_carries_plan_quality_key_defaulting_none(self):
        d = EvalMetrics().to_dict()
        assert 'plan_quality' in d
        assert d['plan_quality'] is None

    def test_to_dict_carries_role_under_test_key_defaulting_none(self):
        d = EvalMetrics().to_dict()
        assert 'role_under_test' in d
        assert d['role_under_test'] is None


# ---------------------------------------------------------------------------
# Deterministic structural plan-quality rubric (step-3/4)
#
# score_plan_structure(plan) -> float reads ONLY the produced plan dict — no
# LLM — and weights structural signals (has_steps, tdd_alternation, files
# declared, design decisions, reuse items, confirmed). It is the always-non-
# sentinel floor run_architect_eval degrades to if the LLM plan judge fails.
# Assertions are relative/structural (well-formed > degenerate; empty == 0.0;
# in-range), not a tuned magic threshold.
# ---------------------------------------------------------------------------

class TestScorePlanStructure:
    def test_well_formed_beats_degenerate_and_in_unit_range(self):
        from orchestrator.evals.judge import score_plan_structure

        wf = score_plan_structure(_well_formed_plan())
        deg = score_plan_structure(_degenerate_plan())
        assert isinstance(wf, float)
        assert 0.0 < wf <= 1.0
        assert wf > deg

    def test_all_criteria_satisfied_is_one(self):
        # Definitional (not a tuned threshold): a plan satisfying every
        # structural criterion earns the full weight → 1.0.
        from orchestrator.evals.judge import score_plan_structure

        assert score_plan_structure(_well_formed_plan()) == 1.0

    def test_empty_plan_is_zero(self):
        from orchestrator.evals.judge import score_plan_structure

        assert score_plan_structure({}) == 0.0

    def test_none_plan_is_zero(self):
        from orchestrator.evals.judge import score_plan_structure

        assert score_plan_structure(None) == 0.0

    def test_no_steps_is_zero_regardless_of_other_fields(self):
        # No steps → 0.0 even if files/design/reuse are present: a plan with no
        # steps is not a plan.
        from orchestrator.evals.judge import score_plan_structure

        assert score_plan_structure({'steps': []}) == 0.0
        assert score_plan_structure(
            {'files': ['a.py'], 'design_decisions': [{'decision': 'x'}]}
        ) == 0.0

    def test_result_is_always_float_clamped_to_unit_interval(self):
        from orchestrator.evals.judge import score_plan_structure

        for plan in [
            _well_formed_plan(), _degenerate_plan(), {}, None, {'steps': []},
        ]:
            s = score_plan_structure(plan)
            assert isinstance(s, float)
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# LLM plan judge judge_plan_quality (step-5/6)
#
# Mirrors run_judge's parse-and-fallback contract: invoke_agent is mocked with
# an AsyncMock whose structured_output is a fixed {plan_quality, per_criterion,
# reasoning}. The judge scores the produced plan against the REAL landed diff,
# guided by PLAN_QUALITY_RUBRIC. An unparseable/empty judge output degrades to a
# defined fallback verdict (plan_quality=None) — the run_architect_eval path
# then falls back to the deterministic score_plan_structure floor.
# ---------------------------------------------------------------------------

def _judge_task() -> dict:
    return {
        'id': 'df_task_2605',
        'name': 'implement the widget',
        'task_definition': {'description': 'implement the widget correctly'},
    }


@pytest.mark.asyncio
class TestJudgePlanQuality:
    async def test_returns_parsed_verdict_and_embeds_plan_diff_rubric(self):
        from orchestrator.evals.judge import (
            PLAN_QUALITY_RUBRIC,
            judge_plan_quality,
        )

        plan = _well_formed_plan()
        reference_diff = (
            '--- a/pkg/mod.py\n+++ b/pkg/mod.py\n+    REAL_LANDED_CHANGE = 1\n'
        )
        verdict_payload = {
            'plan_quality': 0.83,
            'per_criterion': {'has_steps': 1.0, 'tdd_alternation': 1.0},
            'reasoning': 'The plan anticipates the landed diff well.',
        }
        fake = MagicMock()
        fake.structured_output = verdict_payload
        fake.output = json.dumps(verdict_payload)
        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=fake),
        ) as mock_invoke:
            verdict = await judge_plan_quality(plan, reference_diff, _judge_task())

        assert isinstance(verdict.plan_quality, float)
        assert verdict.plan_quality == 0.83
        assert verdict.per_criterion == {'has_steps': 1.0, 'tdd_alternation': 1.0}
        assert 'landed diff' in verdict.reasoning

        # The built prompt embeds the produced plan, the landed reference diff,
        # and the rubric criterion names.
        prompt = mock_invoke.call_args.kwargs['prompt']
        assert 'REAL_LANDED_CHANGE' in prompt        # the reference diff
        assert 'GREEN implement X' in prompt          # a produced-plan step
        for crit in PLAN_QUALITY_RUBRIC['criteria']:
            assert crit['name'] in prompt             # rubric criterion names

    async def test_structured_output_none_falls_back_to_json_output(self):
        # structured_output empty but output carries valid JSON → parsed.
        from orchestrator.evals.judge import judge_plan_quality

        payload = {'plan_quality': 0.5, 'per_criterion': {}, 'reasoning': 'ok'}
        fake = MagicMock()
        fake.structured_output = None
        fake.output = json.dumps(payload)
        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=fake),
        ):
            verdict = await judge_plan_quality(
                _well_formed_plan(), 'diff', _judge_task(),
            )
        assert verdict.plan_quality == 0.5

    async def test_unparseable_output_degrades_to_fallback_verdict(self):
        from orchestrator.evals.judge import PlanQualityVerdict, judge_plan_quality

        fake = MagicMock()
        fake.structured_output = None
        fake.output = 'not json at all {{{'
        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=fake),
        ):
            verdict = await judge_plan_quality(
                _well_formed_plan(), 'diff', _judge_task(),
            )
        # A defined fallback verdict — no crash; plan_quality is the None
        # sentinel so run_architect_eval degrades to score_plan_structure.
        assert isinstance(verdict, PlanQualityVerdict)
        assert verdict.plan_quality is None
        assert isinstance(verdict.reasoning, str)

    async def test_missing_plan_quality_key_degrades_to_fallback(self):
        from orchestrator.evals.judge import judge_plan_quality

        # Valid JSON but no plan_quality field → fallback, not a KeyError crash.
        payload = {'per_criterion': {}, 'reasoning': 'forgot the score'}
        fake = MagicMock()
        fake.structured_output = payload
        fake.output = json.dumps(payload)
        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=fake),
        ):
            verdict = await judge_plan_quality(
                _well_formed_plan(), 'diff', _judge_task(),
            )
        assert verdict.plan_quality is None
