"""Tests for orchestrator.evals.judge's plan-quality instruments.

``score_plan_structure``, ``clamp_unit_score`` and ``judge_plan_quality``, with
``invoke_agent`` patched so no paid LLM call is ever made.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _architect_plan_shapes import create_plan_stub, stub_with_steps, well_formed_plan
from _eval_refusal import REFUSAL_TEXT, refused_agent_result

from orchestrator.evals.judge import PlanQualityVerdict
from orchestrator.evals.measurement import (
    MeasurementStage,
    Scored,
    Unmeasured,
    UnmeasuredCause,
)

# ---------------------------------------------------------------------------
# Synthetic plan shapes only the judge instruments are fed. The builders the
# other architect-eval files share live in _architect_plan_shapes.py.
# ---------------------------------------------------------------------------

def _degenerate_plan() -> dict:
    """A minimal plan: one lonely impl step, no alternation, no metadata."""
    return {
        'steps': [
            {'id': 'step-1', 'type': 'impl', 'description': 'do everything at once'},
        ],
    }


# Every artifact shape that carries NO model content to derive a score from —
# absent, empty, not-a-dict, or the header-only stub in either empty spelling.
# The predicate's own tests live in test_eval_plan_predicates.py.
_UNSCORABLE_PLAN_SHAPES = [
    {},
    None,
    [],
    'x',
    {'steps': []},
    {'steps': None},
    create_plan_stub(),
    stub_with_steps(None),
]


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

        wf = score_plan_structure(well_formed_plan())
        deg = score_plan_structure(_degenerate_plan())
        assert isinstance(wf, float)
        assert 0.0 < wf <= 1.0
        assert wf > deg

    def test_all_criteria_satisfied_is_one(self):
        # Definitional (not a tuned threshold): a plan satisfying every
        # structural criterion earns the full weight → 1.0.
        from orchestrator.evals.judge import score_plan_structure

        assert score_plan_structure(well_formed_plan()) == 1.0

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
            well_formed_plan(), _degenerate_plan(), {}, None, {'steps': []},
        ]:
            s = score_plan_structure(plan)
            assert isinstance(s, float)
            assert 0.0 <= s <= 1.0

    # -- Fail-loud contract on a malformed CODE-OWNED rubric --------------
    # The rubric is code-owned, so a typo (unregistered criterion), an empty
    # criteria list, or a non-positive total weight must raise ValueError rather
    # than silently mis-score (loud-over-silent / structured-facts-at-failure).
    # These raise paths only fire when the plan HAS steps — an empty/None plan
    # short-circuits to 0.0 before any criterion is dispatched — so each test
    # feeds a well-formed plan alongside the malformed rubric.

    def test_unknown_criterion_name_raises_value_error(self):
        from orchestrator.evals.judge import score_plan_structure

        bad_rubric = {'criteria': [{'name': 'no_such_detector', 'weight': 1.0}]}
        with pytest.raises(ValueError, match='unknown plan-quality criterion'):
            score_plan_structure(well_formed_plan(), rubric=bad_rubric)

    def test_empty_criteria_list_raises_value_error(self):
        from orchestrator.evals.judge import score_plan_structure

        with pytest.raises(ValueError, match='no criteria'):
            score_plan_structure(well_formed_plan(), rubric={'criteria': []})

    def test_non_positive_total_weight_raises_value_error(self):
        from orchestrator.evals.judge import score_plan_structure

        zero_weight_rubric = {'criteria': [{'name': 'has_steps', 'weight': 0.0}]}
        with pytest.raises(ValueError, match='total weight'):
            score_plan_structure(well_formed_plan(), rubric=zero_weight_rubric)


# ---------------------------------------------------------------------------
# clamp_unit_score — THE single output-range contract for BOTH plan-quality
# instruments (task 3410)
#
# score_plan_structure and judge_plan_quality reduce onto the SAME axis
# (EvalMetrics.plan_quality), so — exactly as plan_predicates.is_scorable_plan is the
# ONE scorability predicate both instruments call — clamp_unit_score is the
# ONE range/rounding contract both instruments call. Pinning it directly here
# is the anti-drift anchor: a future edit to either instrument cannot drift
# the range behaviour without also breaking this class.
# ---------------------------------------------------------------------------

class TestClampUnitScore:
    @pytest.mark.parametrize('score', [0.0, 0.5, 1.0])
    def test_in_range_values_pass_through_unchanged(self, score):
        from orchestrator.evals.judge import clamp_unit_score

        assert clamp_unit_score(score) == score

    def test_above_range_clamps_to_the_upper_bound(self):
        from orchestrator.evals.judge import clamp_unit_score

        assert clamp_unit_score(1.5) == 1.0
        assert clamp_unit_score(42) == 1.0

    def test_below_range_clamps_to_the_lower_bound(self):
        from orchestrator.evals.judge import clamp_unit_score

        assert clamp_unit_score(-0.4) == 0.0

    def test_result_is_rounded_to_four_decimal_places(self):
        from orchestrator.evals.judge import clamp_unit_score

        # Exact — verified by execution, not a tuned tolerance.
        assert clamp_unit_score(0.123456789) == 0.1235


# ---------------------------------------------------------------------------
# LLM plan judge judge_plan_quality (step-5/6)
#
# invoke_agent is mocked with an AsyncMock whose structured_output is a fixed
# {plan_quality, per_criterion, reasoning}. The judge scores the produced plan
# against the REAL landed diff, guided by PLAN_QUALITY_RUBRIC, and returns
# Scored[PlanQualityVerdict] | Unmeasured: an unparseable, null or non-finite
# answer is Unmeasured(judge, unparseable), and run_architect_eval then falls
# back to the deterministic score_plan_structure floor.
# ---------------------------------------------------------------------------

def _judge_task() -> dict:
    return {
        'id': 'df_task_2605',
        'name': 'implement the widget',
        'task_definition': {'description': 'implement the widget correctly'},
    }


def _judge_result_scoring(raw, *, via: str) -> MagicMock:
    """A judge invoke_agent result reporting ``plan_quality=raw``, delivered via
    *via* (``'structured_output'`` or ``'json_output'``).

    That delivery choice is the ONLY difference between the two mocks. It
    matters in PRODUCTION — ``PLAN_QUALITY_SCHEMA``'s ``{'minimum': 0.0,
    'maximum': 1.0}`` binds a real ``structured_output`` response but not the
    ``json.loads(result.output)`` fallback (see ``clamp_unit_score``'s
    docstring in judge.py) — but NOT in this mock: a ``MagicMock`` enforces no
    schema either way, so both deliveries reach the parser with *raw*
    unchecked here. What the ``via`` parametrization actually pins is that the
    clamp / non-finite handling lives AFTER the ``structured_output or
    json.loads(...)`` merge point, so neither delivery path can regress
    independently of the other.
    """
    payload = {'plan_quality': raw, 'per_criterion': {}, 'reasoning': 'r'}
    fake = MagicMock()
    if via == 'structured_output':
        fake.structured_output = payload
    elif via == 'json_output':
        fake.structured_output = None
    else:
        raise ValueError(f'unknown via: {via!r}')
    fake.output = json.dumps(payload)
    return fake


def _scored_verdict(outcome) -> PlanQualityVerdict:
    """The verdict inside a ``Scored`` plan-judge outcome, asserting it is one."""
    assert isinstance(outcome, Scored), outcome
    return outcome.value


def _assert_judge_unparseable(outcome) -> Unmeasured:
    assert isinstance(outcome, Unmeasured), outcome
    assert outcome.stage is MeasurementStage.JUDGE
    assert outcome.cause is UnmeasuredCause.UNPARSEABLE
    return outcome


def _measurement_warnings(caplog) -> list[str]:
    """Every WARNING-or-worse message measured_call emitted, in order."""
    return [
        r.getMessage() for r in caplog.records
        if r.name == 'orchestrator.evals.measurement' and r.levelno >= logging.WARNING
    ]


@pytest.mark.asyncio
class TestJudgePlanQuality:
    async def test_returns_parsed_verdict_and_embeds_plan_diff_rubric(self):
        from orchestrator.evals.judge import (
            PLAN_QUALITY_RUBRIC,
            judge_plan_quality,
        )

        plan = well_formed_plan()
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
            outcome = await judge_plan_quality(plan, reference_diff, _judge_task())

        verdict = _scored_verdict(outcome)
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
            outcome = await judge_plan_quality(
                well_formed_plan(), 'diff', _judge_task(),
            )
        assert _scored_verdict(outcome).plan_quality == 0.5

    async def test_unparseable_output_is_unmeasured_and_names_the_cell(self, caplog):
        from orchestrator.evals.judge import judge_plan_quality

        caplog.set_level(logging.WARNING)
        fake = MagicMock()
        fake.structured_output = None
        fake.output = 'not json at all {{{'
        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=fake),
        ):
            outcome = await judge_plan_quality(
                well_formed_plan(), 'diff', _judge_task(),
            )
        # No crash, and no fabricated score: run_architect_eval degrades to
        # score_plan_structure on this outcome.
        _assert_judge_unparseable(outcome)
        assert any('df_task_2605' in w for w in _measurement_warnings(caplog))

    async def test_cap_refused_judge_invoke_is_a_cap_hit(self):
        # The judge's OWN invoke can 429 — historically indistinguishable from
        # an unparseable answer. The typed cause makes the infra cause legible.
        from orchestrator.evals.judge import judge_plan_quality

        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=refused_agent_result(cost_usd=0.0)),
        ):
            outcome = await judge_plan_quality(
                well_formed_plan(), 'diff', _judge_task(),
            )

        assert isinstance(outcome, Unmeasured)
        assert outcome.stage is MeasurementStage.JUDGE
        assert outcome.cause is UnmeasuredCause.CAP_HIT

    async def test_unparseable_but_successful_judge_output_is_not_a_refusal(self):
        # The CONTRAST: a judge that really answered and simply produced
        # garbage is unparseable, never a transport refusal.
        from shared.cli_invoke import AgentResult

        from orchestrator.evals.judge import judge_plan_quality

        answered = AgentResult(
            success=True, output='not json at all {{{',
            cost_usd=0.4, duration_ms=3000, turns=1,
        )
        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=answered),
        ):
            outcome = await judge_plan_quality(
                well_formed_plan(), 'diff', _judge_task(),
            )

        _assert_judge_unparseable(outcome)

    async def test_missing_plan_quality_key_is_unparseable(self):
        from orchestrator.evals.judge import judge_plan_quality

        # Valid JSON but no plan_quality field → unparseable, not a KeyError crash.
        payload = {'per_criterion': {}, 'reasoning': 'forgot the score'}
        fake = MagicMock()
        fake.structured_output = payload
        fake.output = json.dumps(payload)
        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=fake),
        ):
            outcome = await judge_plan_quality(
                well_formed_plan(), 'diff', _judge_task(),
            )
        _assert_judge_unparseable(outcome)

    @pytest.mark.parametrize('via', ['structured_output', 'json_output'])
    async def test_explicit_null_plan_quality_is_unparseable_and_loud(
        self, via, caplog,
    ):
        """A null answer is not a judgement, and it no longer degrades silently."""
        from orchestrator.evals.judge import judge_plan_quality

        caplog.set_level(logging.WARNING)
        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=_judge_result_scoring(None, via=via)),
        ):
            outcome = await judge_plan_quality(
                well_formed_plan(), 'diff', _judge_task(),
            )

        _assert_judge_unparseable(outcome)
        warnings = _measurement_warnings(caplog)
        assert len(warnings) == 1
        assert 'df_task_2605' in warnings[0]

    # -- Output-range contract (task 3410) --------------------------------
    # score_plan_structure has always clamped+rounded to [0, 1]/4dp
    # (judge.py:396). judge_plan_quality's parse block did not (see
    # clamp_unit_score's docstring in judge.py for why the schema bound alone
    # is not enough). Both delivery paths are covered so that hole cannot
    # hide behind a structured-output-only test.

    @pytest.mark.parametrize('via', ['structured_output', 'json_output'])
    @pytest.mark.parametrize(('raw', 'expected'), [
        pytest.param(1.5, 1.0, id='above-range-clamps-high'),
        pytest.param(42, 1.0, id='far-above-range-clamps-high'),
        pytest.param(-0.4, 0.0, id='below-range-clamps-low'),
        pytest.param(0.123456789, 0.1235, id='rounds-to-4dp-parity-with-floor'),
    ])
    async def test_parsed_score_is_clamped_and_rounded_regardless_of_delivery_path(
        self, raw, expected, via,
    ):
        from orchestrator.evals.judge import judge_plan_quality

        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=_judge_result_scoring(raw, via=via)),
        ):
            outcome = await judge_plan_quality(
                well_formed_plan(), 'diff', _judge_task(),
            )

        verdict = _scored_verdict(outcome)
        assert isinstance(verdict.plan_quality, float)
        assert 0.0 <= verdict.plan_quality <= 1.0
        assert verdict.plan_quality == expected

    # -- Loud-over-silent: an out-of-contract answer leaves a WARNING (3410) --
    # docs/legibility/design-invariants.md loud-over-silent /
    # structured-facts-at-failure — the same discipline the unjudgeable-
    # artifact guard's WARNING already follows (judge.py:514).

    @pytest.mark.parametrize('via', ['structured_output', 'json_output'])
    @pytest.mark.parametrize(('raw', 'expected'), [
        pytest.param(1.5, 1.0, id='above-range'),
        pytest.param(-0.4, 0.0, id='below-range'),
    ])
    async def test_out_of_range_answer_leaves_exactly_one_warning_naming_it(
        self, via, raw, expected, caplog,
    ):
        from orchestrator.evals.judge import judge_plan_quality

        caplog.set_level(logging.WARNING, logger='orchestrator.evals.judge')
        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=_judge_result_scoring(raw, via=via)),
        ):
            outcome = await judge_plan_quality(
                well_formed_plan(), 'diff', _judge_task(),
            )

        # Both clamp directions are pinned, not just the above-range case —
        # the warning text is built from raw_quality!r / plan_quality!r, so a
        # future edit that only warns on raw_quality > 1.0 would keep every
        # above-range assertion green while silently dropping the low-side
        # signal.
        warnings = _judge_warnings(caplog)
        assert len(warnings) == 1
        assert 'df_task_2605' in warnings[0]  # WHICH cell to go look at
        assert repr(raw) in warnings[0]       # the raw out-of-range value
        assert repr(expected) in warnings[0]  # the substituted value

        # The verdict is otherwise intact: a corrected CONTENT answer, never
        # an infra refusal (3118 exclusion shape) or a parse failure.
        verdict = _scored_verdict(outcome)
        assert verdict.plan_quality == expected
        assert verdict.per_criterion == {}
        assert verdict.reasoning == 'r'

    async def test_in_range_answer_emits_no_warning(self, caplog):
        """CONTROL — without this, the warning fires on every judge call and
        stops meaning anything.
        """
        from orchestrator.evals.judge import judge_plan_quality

        caplog.set_level(logging.WARNING, logger='orchestrator.evals.judge')
        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(
                return_value=_judge_result_scoring(0.83, via='structured_output'),
            ),
        ):
            outcome = await judge_plan_quality(
                well_formed_plan(), 'diff', _judge_task(),
            )

        assert _scored_verdict(outcome).plan_quality == 0.83
        assert _judge_warnings(caplog) == []

    # -- The hole a bare clamp leaves open: non-finite answers (task 3410, --
    # -- narrowed by task 4190) ---------------------------------------------
    # A bare clamp does NOT keep plan_quality in [0, 1] for NaN: NaN is
    # unordered, so it passes straight through unclamped (see
    # clamp_unit_score's docstring in judge.py for the mechanics). A bare
    # clamp DOES keep +/-Infinity in [0, 1] — but only by fabricating the
    # best/worst possible score (1.0/0.0) on a surface
    # cell_admission.mean_plan_quality -> select_survivors actually ranks on. Both
    # are equally NOT a judgement, so both are Unmeasured(judge, unparseable).
    # Both are reachable through the SAME schema-bypassing
    # json.loads(result.output) fallback — CPython's json.loads accepts the
    # bare literals NaN and Infinity, and overflows 1e999 to inf — not
    # hypothetical.
    #
    # This is a NARROWING of the task-3410 clamp+warn doctrine, not a
    # reversal: FINITE out-of-range answers (e.g. 1.5) stay orderable and
    # intent-preserving, and keep the clamp+warn path pinned by the sibling
    # tests directly above (~:930-994).

    @pytest.mark.parametrize('via', ['structured_output', 'json_output'])
    @pytest.mark.parametrize('raw', [
        pytest.param(float('nan'), id='nan'),
        pytest.param(float('inf'), id='positive-infinity'),
        pytest.param(float('-inf'), id='negative-infinity'),
    ])
    async def test_non_finite_answer_is_unparseable(self, raw, via, caplog):
        from orchestrator.evals.judge import judge_plan_quality

        caplog.set_level(logging.WARNING)
        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=_judge_result_scoring(raw, via=via)),
        ):
            outcome = await judge_plan_quality(
                well_formed_plan(), 'diff', _judge_task(),
            )

        # run_architect_eval degrades to the deterministic structural floor on
        # this — never a fabricated pool-best 1.0 laundered into a number a
        # downstream mean could be poisoned by. A nonsense answer is a CONTENT
        # failure, never the 3118 infra-refusal exclusion shape.
        unmeasured = _assert_judge_unparseable(outcome)
        assert repr(raw) in unmeasured.reason

        warnings = _measurement_warnings(caplog)
        assert len(warnings) == 1
        assert 'df_task_2605' in warnings[0]  # WHICH cell to go look at
        assert repr(raw) in warnings[0]       # WHAT was wrong


# ---------------------------------------------------------------------------
# The plan judge's OWN invocation spend (eval-revival υ, task 6734)
#
# run_architect_eval never sees the judge's AgentResult, only the returned
# Scored | Unmeasured outcome — so the judge invocation's cost rides the
# outcome's cost_usd, on every path, whatever the verdict turned out to be.
# ---------------------------------------------------------------------------

class TestPlanJudgeOutcomeCarriesItsSpend:
    @pytest.mark.asyncio
    async def test_success_path_carries_the_invocations_own_spend(self):
        from orchestrator.evals.judge import judge_plan_quality

        payload = {'plan_quality': 0.83, 'per_criterion': {}, 'reasoning': 'r'}
        fake = MagicMock()
        fake.structured_output = payload
        fake.output = json.dumps(payload)
        fake.cost_usd = 0.42
        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=fake),
        ):
            outcome = await judge_plan_quality(
                well_formed_plan(), 'diff', _judge_task(),
            )

        assert isinstance(outcome, Scored)
        assert outcome.cost_usd == pytest.approx(0.42)

    # -- The PRODUCER coerces: cost_usd is a real float at construction ----

    @pytest.mark.asyncio
    async def test_success_path_with_unset_mock_cost_degrades_to_zero(self):
        """A bare MagicMock invoke result that never sets ``cost_usd`` must
        read back a real ``0.0`` float, not a Mock instance and not a
        fabricated ``1.0`` (MagicMock's default ``__float__``).

        This is what makes the producer-side coercion safe for the judge
        tests that construct exactly this kind of bare mock and never touch
        ``cost_usd``.
        """
        from orchestrator.evals.judge import judge_plan_quality

        payload = {'plan_quality': 0.83, 'per_criterion': {}, 'reasoning': 'r'}
        fake = MagicMock()
        fake.structured_output = payload
        fake.output = json.dumps(payload)
        # cost_usd deliberately left UNSET — fake.cost_usd is an
        # auto-generated MagicMock child.
        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=fake),
        ):
            outcome = await judge_plan_quality(
                well_formed_plan(), 'diff', _judge_task(),
            )

        assert outcome.cost_usd == 0.0
        assert isinstance(outcome.cost_usd, float)

    @pytest.mark.asyncio
    async def test_success_path_with_non_finite_cost_degrades_to_zero(self):
        """A judge invocation reporting a non-finite cost_usd (NaN) must not
        ride into the verdict unchanged — it would poison every downstream
        sum/mean once folded into an architect cell's cost_usd."""
        from orchestrator.evals.judge import judge_plan_quality

        payload = {'plan_quality': 0.83, 'per_criterion': {}, 'reasoning': 'r'}
        fake = MagicMock()
        fake.structured_output = payload
        fake.output = json.dumps(payload)
        fake.cost_usd = float('nan')
        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=fake),
        ):
            outcome = await judge_plan_quality(
                well_formed_plan(), 'diff', _judge_task(),
            )

        assert outcome.cost_usd == 0.0

    # -- Degraded POST-invoke paths still carry real spend (step-3/4) ------
    # A refused/unparseable/non-finite judge answer still consumed whatever
    # the provider charged — reporting $0 there would understate real spend
    # exactly the way this task's bug does. Mirrors the content-failure /
    # infra-failure discipline tasks 3118/3302/3303 already established.

    @pytest.mark.asyncio
    async def test_transport_refusal_path_carries_the_invocations_own_spend(self):
        from shared.cli_invoke import AgentResult

        from orchestrator.evals.judge import judge_plan_quality

        refused = AgentResult(
            success=False,
            output=REFUSAL_TEXT,
            cost_usd=0.07,
            duration_ms=1200,
            turns=0,
            subtype='error',
            api_error_status=429,
        )
        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=refused),
        ):
            outcome = await judge_plan_quality(
                well_formed_plan(), 'diff', _judge_task(),
            )

        assert isinstance(outcome, Unmeasured)
        assert outcome.cause is UnmeasuredCause.CAP_HIT
        assert outcome.cost_usd == pytest.approx(0.07)

    @pytest.mark.asyncio
    async def test_parse_failure_path_carries_the_invocations_own_spend(self):
        from orchestrator.evals.judge import judge_plan_quality

        fake = MagicMock()
        fake.structured_output = None
        fake.output = 'not json at all {{{'
        fake.cost_usd = 0.31
        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=fake),
        ):
            outcome = await judge_plan_quality(
                well_formed_plan(), 'diff', _judge_task(),
            )

        _assert_judge_unparseable(outcome)
        assert outcome.cost_usd == pytest.approx(0.31)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('raw', [
        pytest.param(float('nan'), id='nan'),
        pytest.param(float('inf'), id='positive-infinity'),
        pytest.param(float('-inf'), id='negative-infinity'),
    ])
    async def test_non_finite_path_carries_the_invocations_own_spend(self, raw):
        from orchestrator.evals.judge import judge_plan_quality

        fake = _judge_result_scoring(raw, via='structured_output')
        fake.cost_usd = 0.29
        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=fake),
        ):
            outcome = await judge_plan_quality(
                well_formed_plan(), 'diff', _judge_task(),
            )

        _assert_judge_unparseable(outcome)
        assert outcome.cost_usd == pytest.approx(0.29)

    # -- The PRE-invoke refusal fabricates no spend (step-3/4) --------------
    # Nothing to judge, so invoke_agent is never awaited — recording spend
    # here would fabricate a cost for an invocation that never happened.

    @pytest.mark.asyncio
    @pytest.mark.parametrize('plan', _UNSCORABLE_PLAN_SHAPES)
    async def test_pre_invoke_refusal_records_no_fabricated_spend(self, plan):
        from orchestrator.evals.judge import judge_plan_quality

        mock_invoke = AsyncMock()
        with patch('orchestrator.evals.judge.invoke_agent', mock_invoke):
            outcome = await judge_plan_quality(plan, 'diff', _judge_task())

        mock_invoke.assert_not_awaited()
        assert isinstance(outcome, Scored)
        assert outcome.cost_usd == 0.0


# ---------------------------------------------------------------------------
# The plan judge REFUSES an unjudgeable artifact (task 3303)
#
# The reported defect, from the 2026-07-29 corpus cell
# ``reify_task_12__architect-opus-high__52c66767.json``: ``plan_steps=0``
# alongside ``plan_quality=0.31`` — an artifact carrying NOTHING to judge,
# scored a confident-looking third of a point. That number cannot have come
# from the deterministic floor at all: PLAN_QUALITY_RUBRIC's weights sum to
# 8.0 and score_plan_structure returns ``round(satisfied_weight / 8, 4)``, so
# its outputs are multiples of 0.125 and 0.31 is not one. It came from the LLM
# judge, which scored an empty artifact because it had no scorability guard of
# its own — its coherence rested ENTIRELY on its one caller remembering to gate.
#
# Task 3302 closed the two OUTER halves: the CALL SITE (run_architect_eval gates
# on is_scorable_plan) and the READER (cell_admission.plan_quality_score floors
# an on-disk no-plan cell). This class pins the INSTRUMENT itself, so a second
# caller — a backfill/re-scoring script, a new eval path, prompt_opt, a resume
# wave — cannot silently re-open the same defect.
#
# Parametrized over the _UNSCORABLE_PLAN_SHAPES corpus at the top of this file.
#
# ONE call per shape pins every property of the refusal, because they are four
# independent facts about a single return value from a single code path — and
# the CONTROL carries the only falsifiable half of the coherence identity
# ``plan_quality == score_plan_structure(plan)``: over the unscorable corpus
# both sides are 0.0 by construction (score_plan_structure short-circuits on
# the very predicate the guard consults), so it is only on a SCORABLE plan,
# where the judge is consulted and the floor is not, that the two instruments
# have a degree of freedom to disagree at all.
# ---------------------------------------------------------------------------

def _confident_judge_result() -> MagicMock:
    """What the UNGATED judge really does with an artifact carrying no content.

    A high, confident-looking score with per-criterion detail — at the call
    site indistinguishable from a real judgement of a real plan. This is the
    payload the guard must never let through.
    """
    payload = {
        'plan_quality': 0.95,
        'per_criterion': {'has_steps': 1.0},
        'reasoning': 'looks fine',
    }
    fake = MagicMock()
    fake.structured_output = payload
    fake.output = json.dumps(payload)
    return fake


async def _judge_with_a_confident_llm(plan, task=None):
    """Await ``judge_plan_quality`` on *plan* with the ungated judge mocked.

    The fixture lives HERE, once: every test below differs only in the plan (or
    task) it passes, so a future change to the judge's call signature is a
    one-place edit rather than one per test. Returns the ``invoke_agent`` mock
    alongside the outcome, since "was the LLM consulted at all?" is half of
    what these tests pin.
    """
    from orchestrator.evals.judge import judge_plan_quality

    with patch(
        'orchestrator.evals.judge.invoke_agent',
        AsyncMock(return_value=_confident_judge_result()),
    ) as mock_invoke:
        outcome = await judge_plan_quality(
            plan, 'diff', _judge_task() if task is None else task,
        )
    return outcome, mock_invoke


def _judge_warnings(caplog) -> list[str]:
    """Every WARNING-or-worse message the judge module emitted, in order."""
    return [
        r.getMessage() for r in caplog.records
        if r.name == 'orchestrator.evals.judge' and r.levelno >= logging.WARNING
    ]


@pytest.mark.asyncio
class TestJudgePlanQualityRefusesAnUnjudgeableArtifact:
    @pytest.mark.parametrize('plan', _UNSCORABLE_PLAN_SHAPES)
    async def test_an_unjudgeable_artifact_is_refused_and_floored(self, plan, caplog):
        """Every property of the refusal, on ONE judge call per artifact shape.

        Deliberately one test rather than one per property: these are four
        independent facts about a single return value from a single code path,
        so splitting them would buy four copies of the fixture and four times
        the invocations without covering anything more.
        """
        from orchestrator.evals.judge import score_plan_structure

        caplog.set_level(logging.WARNING, logger='orchestrator.evals.judge')
        outcome, mock_invoke = await _judge_with_a_confident_llm(plan)
        floor = score_plan_structure(plan)

        # A definite CONTENT verdict, Scored: never Unmeasured (parse failure /
        # transport refusal) and never the cap-tainted exclusion shape — the
        # content-failure vs infra-failure distinction tasks 3118 and 3302 both
        # turn on. A stepless plan from a healthy judge call is a REAL 0.0.
        verdict = _scored_verdict(outcome)
        # The floor's own value, DERIVED — never the judge's confident 0.95.
        assert verdict.plan_quality == floor == 0.0
        # Nothing to judge, so no spend: the refusal happens BEFORE the prompt
        # is built and before invoke_agent is awaited.
        mock_invoke.assert_not_awaited()
        assert outcome.cost_usd == 0.0
        assert verdict.per_criterion == {}
        # ``reasoning`` is the only part of this verdict a human reads off a
        # persisted cell, so it must EXPLAIN the 0.0 rather than leave the cell
        # indistinguishable from a judge that genuinely scored zero. Substance,
        # not a prose pin.
        assert 'no steps' in verdict.reasoning
        assert str(floor) in verdict.reasoning
        # LOUD, never a silent floor: reaching the guard means a caller did NOT
        # gate — the precise situation that produced the reported artifact — so
        # per the repo's loud-over-silent / structured-facts-at-failure
        # invariant (docs/legibility/design-invariants.md) it leaves exactly one
        # WARNING behind. Substance again, not exact prose: a wording-brittle
        # pin would fail on any rephrase without indicating a behaviour change.
        warnings = _judge_warnings(caplog)
        assert len(warnings) == 1
        assert 'df_task_2605' in warnings[0]       # WHICH cell to go look at
        assert str(floor) in warnings[0]           # what was substituted
        # The LLM was never consulted — distinguishing this record from a judge
        # that answered and happened to say 0.0.
        assert 'skip' in warnings[0].lower()

    @pytest.mark.parametrize('plan', [
        pytest.param(well_formed_plan(), id='CONTROL-well-formed'),
        pytest.param(_degenerate_plan(), id='CONTROL-degenerate-but-scorable'),
    ])
    async def test_a_scorable_plan_still_reaches_the_judge(self, plan, caplog):
        """CONTROL — and the falsifiable half of the coherence identity.

        The guard keys on scorability, never on quality: it asks "is there
        content to judge?", so a DEGENERATE plan (one lonely impl step — poor,
        but real) keeps earning its low score from the judge rather than being
        laundered into a floor.

        This is also the only place the two instruments have a degree of
        freedom to disagree. Over _UNSCORABLE_PLAN_SHAPES, asserting
        ``verdict.plan_quality == score_plan_structure(plan)`` is unfalsifiable
        — the floor short-circuits on the very predicate the guard consults, so
        both sides are unconditionally 0.0. Here the floor and the judge return
        genuinely different numbers, and the assertion that the JUDGE's stands
        is what pins the floor as a refusal value rather than a universal
        substitute.
        """
        from orchestrator.evals.judge import score_plan_structure
        from orchestrator.evals.plan_predicates import is_scorable_plan

        caplog.set_level(logging.WARNING, logger='orchestrator.evals.judge')
        assert is_scorable_plan(plan)
        outcome, mock_invoke = await _judge_with_a_confident_llm(plan)
        verdict = _scored_verdict(outcome)

        mock_invoke.assert_awaited_once()
        assert verdict.plan_quality == 0.95
        assert score_plan_structure(plan) > 0.0
        assert verdict.plan_quality != score_plan_structure(plan)
        # The signal stays rare and meaningful: the normal path emits no
        # refusal warning, or the warning stops meaning "a caller did not gate".
        assert _judge_warnings(caplog) == []

    async def test_the_refusal_names_the_cell_even_without_a_task_id(self, caplog):
        """A second caller may key its task dict by ``name`` alone.

        That caller is precisely who this guard exists to defend against, so
        reading ``id`` alone would print "unknown" exactly when the log's whole
        purpose — telling the reader WHICH cell to go look at — matters most.
        (``run_judge`` and the plan-judge prompt builder both key off ``name``
        first, so a name-only task dict is a shape the module already accepts.)
        """
        caplog.set_level(logging.WARNING, logger='orchestrator.evals.judge')
        await _judge_with_a_confident_llm({}, task={'name': 'the widget task'})

        warnings = _judge_warnings(caplog)
        assert len(warnings) == 1
        assert 'the widget task' in warnings[0]
        assert 'unknown' not in warnings[0]
