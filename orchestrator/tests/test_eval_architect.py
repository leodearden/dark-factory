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

import contextlib
import json
from pathlib import Path
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
# EvalMetrics.invocation_error / cap_tainted fields (task 3118 step-1/2)
#
# The infra-failure markers that let a reader tell "the model wrote a terrible
# plan" from "we never got to ask the model": ``invocation_error`` records WHAT
# happened at the transport layer (stage-prefixed), ``cap_tainted`` is the
# machine-checkable exclusion predicate every aggregate keys on. Both are
# default-safe so a result JSON persisted before these fields existed reads
# back unchanged.
# ---------------------------------------------------------------------------

class TestEvalMetricsInvocationErrorField:
    def test_invocation_error_default_is_none(self):
        assert EvalMetrics().invocation_error is None

    def test_cap_tainted_default_is_false(self):
        assert EvalMetrics().cap_tainted is False

    def test_to_dict_carries_invocation_error_key_defaulting_none(self):
        d = EvalMetrics().to_dict()
        assert 'invocation_error' in d
        assert d['invocation_error'] is None

    def test_to_dict_carries_cap_tainted_key_defaulting_false(self):
        d = EvalMetrics().to_dict()
        assert 'cap_tainted' in d
        assert d['cap_tainted'] is False


# ---------------------------------------------------------------------------
# detect_invocation_error — the pure transport-refusal classifier (3118 step-3/4)
#
# Delegates to the existing shared.invocation_outcome.classify_invocation seam
# (do NOT re-implement 429/cap string matching in the eval layer) plus a
# structured api_error_status==429 fallback, since 429 is deliberately excluded
# from the AuthFailed variant and routes to the cap-TEXT tier — a body-less 429
# would otherwise classify as Failure(unclassified).
#
# The load-bearing property is asymmetric: an INFRA refusal must be marked, but
# an ORDINARY content failure (an architect that ran fine and produced a bad or
# absent plan) must NOT be — that is a real reliability signal and must keep
# scoring 0.0 rather than being laundered into an exclusion.
# ---------------------------------------------------------------------------

# The exact payload the Claude CLI emitted during the 2026-07-27 architect-effort
# campaign, which produced the 3 hand-excluded fixtures this task automates away
# (plans/eval-architect-effort-verdict-2026-07-27.md, "Data hygiene").
_CAP_TEXT = "You've hit your session limit · resets 8pm"


def _cap_agent_result():
    from shared.cli_invoke import AgentResult

    return AgentResult(
        success=False,
        output=_CAP_TEXT,
        cost_usd=0.0,
        duration_ms=1200,
        turns=0,
        subtype='error',
        api_error_status=429,
    )


class TestDetectInvocationError:
    def test_none_result_is_not_an_invocation_error(self):
        from orchestrator.evals.metrics import detect_invocation_error

        assert detect_invocation_error(None) is None

    def test_successful_run_quoting_cap_text_is_not_tainted(self):
        # The no-false-taint property, inherited free from classify_invocation's
        # OK short-circuit: a HEALTHY run whose output merely QUOTES a cap string
        # (e.g. an agent discussing a usage-limit message) is not an infra
        # refusal and must never be excluded from the aggregate.
        from orchestrator.evals.metrics import detect_invocation_error
        from shared.cli_invoke import AgentResult

        ok = AgentResult(
            success=True, output=f'the CLI printed: {_CAP_TEXT}',
            cost_usd=1.5, duration_ms=9000, turns=12,
        )
        assert detect_invocation_error(ok) is None

    def test_campaign_429_payload_yields_marker_naming_the_cap(self):
        from orchestrator.evals.metrics import detect_invocation_error

        marker = detect_invocation_error(_cap_agent_result())
        assert isinstance(marker, str) and marker
        assert 'cap' in marker.lower()
        # The marker quotes the REASON, so a human reading the result JSON sees
        # the forensic evidence, not just a boolean.
        assert 'session limit' in marker

    def test_body_less_429_yields_marker_via_structured_fallback(self):
        # 429 is deliberately EXCLUDED from AuthFailed (it routes to the cap-text
        # tier), so a 429 with no cap body classifies as a wedge/unclassified
        # failure — the structured api_error_status fallback must still catch it.
        from orchestrator.evals.metrics import detect_invocation_error
        from shared.cli_invoke import AgentResult

        bare = AgentResult(
            success=False, output='', stderr='', cost_usd=0.0,
            duration_ms=800, turns=0, api_error_status=429,
        )
        marker = detect_invocation_error(bare)
        assert isinstance(marker, str) and marker
        assert '429' in marker

    def test_auth_failure_yields_auth_marker(self):
        from orchestrator.evals.metrics import detect_invocation_error
        from shared.cli_invoke import AgentResult

        unauthorized = AgentResult(
            success=False, output='', cost_usd=0.0, duration_ms=200,
            turns=0, api_error_status=401,
        )
        marker = detect_invocation_error(unauthorized)
        assert isinstance(marker, str) and marker
        assert 'auth' in marker.lower()
        assert '401' in marker

    def test_ordinary_content_failure_is_not_tainted(self):
        # THE load-bearing negative: an architect that really ran (non-zero cost,
        # real turns) and simply failed to produce a good plan is a CONTENT
        # signal — it must keep scoring 0.0, never be excluded as infra.
        from orchestrator.evals.metrics import detect_invocation_error
        from shared.cli_invoke import AgentResult

        content_failure = AgentResult(
            success=False, output='I could not complete the plan',
            cost_usd=1.2, duration_ms=45000, turns=8, api_error_status=None,
        )
        assert detect_invocation_error(content_failure) is None


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
            score_plan_structure(_well_formed_plan(), rubric=bad_rubric)

    def test_empty_criteria_list_raises_value_error(self):
        from orchestrator.evals.judge import score_plan_structure

        with pytest.raises(ValueError, match='no criteria'):
            score_plan_structure(_well_formed_plan(), rubric={'criteria': []})

    def test_non_positive_total_weight_raises_value_error(self):
        from orchestrator.evals.judge import score_plan_structure

        zero_weight_rubric = {'criteria': [{'name': 'has_steps', 'weight': 0.0}]}
        with pytest.raises(ValueError, match='total weight'):
            score_plan_structure(_well_formed_plan(), rubric=zero_weight_rubric)


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


# ---------------------------------------------------------------------------
# Architect-role candidate in configs.py (step-7/8)
#
# EvalConfig gains role (default 'implementer' → every existing EVAL_CONFIGS
# entry unchanged); ARCHITECT_EVAL_CONFIGS carries role=='architect' candidates;
# get_config_by_name resolves them. Pure-data assertions, no LLM.
# ---------------------------------------------------------------------------

class TestArchitectEvalConfigs:
    def test_evalconfig_role_defaults_to_implementer(self):
        from orchestrator.evals.configs import EvalConfig

        cfg = EvalConfig('x', 'claude', 'opus', 'high')
        assert cfg.role == 'implementer'

    def test_existing_eval_configs_all_resolve_implementer(self):
        from orchestrator.evals.configs import EVAL_CONFIGS

        assert EVAL_CONFIGS  # non-empty
        assert all(c.role == 'implementer' for c in EVAL_CONFIGS)

    def test_architect_eval_configs_exist_with_architect_role(self):
        from orchestrator.evals.configs import ARCHITECT_EVAL_CONFIGS

        assert len(ARCHITECT_EVAL_CONFIGS) >= 1
        assert all(c.role == 'architect' for c in ARCHITECT_EVAL_CONFIGS)
        # Candidate names are unique (they key results/lookup).
        names = [c.name for c in ARCHITECT_EVAL_CONFIGS]
        assert len(names) == len(set(names))

    def test_get_config_by_name_resolves_architect_candidate(self):
        from orchestrator.evals.configs import (
            ARCHITECT_EVAL_CONFIGS,
            get_config_by_name,
        )

        name = ARCHITECT_EVAL_CONFIGS[0].name
        cfg = get_config_by_name(name)
        assert cfg is not None
        assert cfg.name == name
        assert cfg.role == 'architect'

    def test_architect_fable_candidate_added_existing_byte_unchanged(self):
        """eval-revival π: architect-fable-high joins ARCHITECT_EVAL_CONFIGS.

        Parity discipline (eval-revival decision 11): the three pre-existing
        candidates stay byte-unchanged (dataclass equality) when a new
        candidate is appended; the new candidate is config-only — it rides
        run_ofat_stage's existing generic role='architect' branch (task 2478).
        """
        from orchestrator.evals.configs import ARCHITECT_EVAL_CONFIGS, EvalConfig

        existing = [
            EvalConfig('architect-opus-high', 'claude', 'opus', 'high', role='architect'),
            EvalConfig('architect-opus-max', 'claude', 'opus', 'max', role='architect'),
            EvalConfig('architect-sonnet-high', 'claude', 'sonnet', 'high', role='architect'),
        ]
        by_name = {c.name: c for c in ARCHITECT_EVAL_CONFIGS}
        for expected in existing:
            assert by_name[expected.name] == expected

        fable = by_name['architect-fable-high']
        assert fable.backend == 'claude'
        assert fable.model == 'claude-fable-5'
        assert fable.effort == 'high'
        assert fable.role == 'architect'

        assert {c.name for c in ARCHITECT_EVAL_CONFIGS} == {
            'architect-opus-high',
            'architect-opus-max',
            'architect-sonnet-high',
            'architect-fable-high',
        }


# ---------------------------------------------------------------------------
# run_architect_eval — the plan-only architect eval entry (step-9/10)
#
# Reuses the _run_plan_only invocation sequence (create_eval_worktree at
# pre_task_commit → TaskArtifacts.init → briefing.build_architect_prompt →
# invoke_agent(ARCHITECT) → read_plan), materializes the landed reference diff,
# scores via judge_plan_quality (degrading to score_plan_structure), and
# persists an EvalResult with role_under_test='architect'. DOWNSTREAM roles are
# FROZEN — only the architect runs. All boundaries patched (hermetic).
# ---------------------------------------------------------------------------

def _arch_task() -> dict:
    return {
        'id': 'df_task_2605',
        'name': 'implement the widget',
        'project_root': '/fake/project',
        'pre_task_commit': 'basecommit123',
        'reference': {'post_task_commit': 'postcommit456'},
        'task_definition': {'title': 'Widget', 'description': 'implement the widget'},
    }


async def _run_architect_eval_hermetic(
    cfg,
    *,
    produced_plan: dict,
    judge_return=None,
    judge_side_effect=None,
    arch_success: bool = True,
    invoke_side_effect=None,
    arch_result=None,
):
    """Drive run_architect_eval with every git/worktree/LLM boundary patched.

    Returns ``(result, mocks)`` where ``mocks`` exposes the invoke/verify/save/
    judge mocks for assertions. ``invoke_side_effect`` lets a test make the live
    architect invoke raise (e.g. ``TimeoutError`` to simulate --timeout expiry).

    ``arch_result``, when supplied, is used VERBATIM as ``invoke_agent``'s
    return value instead of the default duck-typed MagicMock — the injection
    point the cap tests use to feed a REAL ``shared.cli_invoke.AgentResult``
    carrying the campaign 429 payload (``api_error_status=429`` +
    "You've hit your session limit · resets 8pm"), which the MagicMock cannot
    express because every attribute access on it returns a truthy Mock.
    """
    from orchestrator.evals import runner
    from orchestrator.evals.judge import PlanQualityVerdict

    if judge_return is None and judge_side_effect is None:
        judge_return = PlanQualityVerdict(
            plan_quality=0.77, per_criterion={}, reasoning='good',
        )

    invoke_return = arch_result if arch_result is not None else MagicMock(
        success=arch_success, cost_usd=1.23, duration_ms=4567, output='done',
    )
    mock_invoke = AsyncMock(return_value=invoke_return, side_effect=invoke_side_effect)

    artifacts_instance = MagicMock()
    artifacts_instance.read_plan.return_value = produced_plan

    briefing_instance = MagicMock()
    briefing_instance.build_architect_prompt = AsyncMock(return_value='ARCH PROMPT')

    mock_judge = AsyncMock(return_value=judge_return, side_effect=judge_side_effect)
    mock_verify = AsyncMock()
    mock_save = MagicMock()

    with contextlib.ExitStack() as es:
        p = es.enter_context
        p(patch('orchestrator.evals.snapshots.create_eval_worktree',
                AsyncMock(return_value=(Path('/fake/wt'), 'run-abc'))))
        p(patch('orchestrator.evals.snapshots.cleanup_eval_worktree', AsyncMock()))
        p(patch('orchestrator.evals.snapshots.get_diff_between_commits',
                AsyncMock(return_value='--- a/x\n+++ b/x\n+ landed change\n')))
        p(patch('orchestrator.agents.invoke.invoke_agent', mock_invoke))
        p(patch('orchestrator.artifacts.TaskArtifacts',
                MagicMock(return_value=artifacts_instance)))
        p(patch('orchestrator.agents.briefing.BriefingAssembler',
                MagicMock(return_value=briefing_instance)))
        p(patch('orchestrator.evals.runner.build_eval_orch_config',
                MagicMock(return_value=MagicMock())))
        p(patch('orchestrator.evals.judge.judge_plan_quality', mock_judge))
        p(patch('orchestrator.evals.runner.save_result', mock_save))
        p(patch('orchestrator.evals.runner.load_task',
                MagicMock(return_value=_arch_task())))
        p(patch('orchestrator.verify.run_verification', mock_verify))
        result = await runner.run_architect_eval(
            Path('/fake/task.json'), cfg, base_config=MagicMock(),
        )
    return result, {
        'invoke': mock_invoke, 'verify': mock_verify,
        'save': mock_save, 'judge': mock_judge,
    }


@pytest.mark.asyncio
class TestRunArchitectEval:
    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role='architect',
        )

    async def test_scores_plan_freezes_downstream_and_persists(self):
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(), produced_plan=_well_formed_plan(),
        )

        # NON-sentinel plan_quality float + role_under_test stamped.
        assert isinstance(result.metrics['plan_quality'], float)
        assert result.metrics['plan_quality'] == 0.77
        assert result.metrics['role_under_test'] == 'architect'
        assert result.config_name == 'architect-sonnet-high'
        assert result.task_id == 'df_task_2605'

        # Architect invoked with the CANDIDATE's model/backend/effort.
        kw = mocks['invoke'].call_args.kwargs
        assert kw['model'] == 'sonnet'
        assert kw['backend'] == 'claude'
        assert kw['effort'] == 'high'

        # DOWNSTREAM roles FROZEN: exactly ONE agent invocation (the architect),
        # and verification never runs.
        assert mocks['invoke'].call_count == 1
        mocks['verify'].assert_not_called()

        # Persisted via save_result.
        mocks['save'].assert_called_once()

    async def test_judge_failure_degrades_to_deterministic_floor(self):
        from orchestrator.evals.judge import PlanQualityVerdict, score_plan_structure

        plan = _well_formed_plan()
        # Judge returns the None sentinel (parse failure) → run_architect_eval
        # must degrade to the deterministic structural floor, never a null.
        result, _ = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=plan,
            judge_return=PlanQualityVerdict(
                plan_quality=None, per_criterion={}, reasoning='parse failure',
            ),
        )
        assert result.metrics['plan_quality'] is not None
        assert result.metrics['plan_quality'] == score_plan_structure(plan)
        assert result.metrics['role_under_test'] == 'architect'

    async def test_judge_raising_still_yields_non_sentinel_score(self):
        from orchestrator.evals.judge import score_plan_structure

        plan = _well_formed_plan()
        # Even if the judge RAISES, plan_quality must be the deterministic floor.
        result, _ = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=plan,
            judge_side_effect=RuntimeError('judge exploded'),
        )
        assert result.metrics['plan_quality'] == score_plan_structure(plan)
        assert result.metrics['plan_quality'] is not None

    async def test_architect_timeout_maps_to_timeout_outcome(self):
        # A hung architect invoke surfaces as TimeoutError — what asyncio.wait_for
        # raises when the operator's --timeout expires. run_architect_eval must
        # honor the timeout and map it to outcome='timeout' (mirroring run_eval),
        # NOT the generic 'blocked', while still emitting a non-sentinel
        # plan_quality float.
        result, _ = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=_well_formed_plan(),
            invoke_side_effect=TimeoutError(),
        )
        assert result.outcome == 'timeout'
        assert isinstance(result.metrics['plan_quality'], float)
        assert result.metrics['role_under_test'] == 'architect'

    async def test_wires_plan_tools_mcp_into_architect_invoke(self):
        # BUG 1: the eval architect must be wired with plan-tools MCP exactly
        # like real dispatch — otherwise read_plan() returns {} → plan_steps=0 →
        # plan_quality collapses to the structural floor (esc-df_task_12-3).
        # run_architect_eval must pass a non-None mcp_config to invoke_agent whose
        # 'plan-tools' entry launches the direct-interpreter no-uv hot path against
        # the RELOCATED .task-meta/<name>/ root (matching production
        # _inject_plan_tools_mcp), so the plan-tools server writes plan.json where
        # the TaskArtifacts(worktree, meta_root=_meta_root_for_worktree(worktree))
        # readback reads it.
        import sys

        from orchestrator.workflow import _meta_root_for_worktree

        _, mocks = await _run_architect_eval_hermetic(
            self._cfg(), produced_plan=_well_formed_plan(),
        )

        # Computed OUTSIDE the harness ExitStack (above) so the TaskArtifacts
        # patch is inactive and the REAL relocated path is produced. This uses
        # workflow's own unpatched TaskArtifacts binding — the harness patches
        # only orchestrator.artifacts.TaskArtifacts.
        expected_meta_root = _meta_root_for_worktree(Path('/fake/wt'))
        assert str(expected_meta_root) == '/fake/.task-meta/wt'

        mcp_config = mocks['invoke'].call_args.kwargs.get('mcp_config')
        assert mcp_config is not None, 'run_architect_eval passed no mcp_config'
        plan_tools = mcp_config['mcpServers']['plan-tools']
        assert plan_tools['command'] == sys.executable
        assert plan_tools['args'] == [
            '-m', 'orchestrator.mcp.plan_tools',
            '--worktree', str(Path('/fake/wt')),
            '--meta-root', str(expected_meta_root),
        ]


# ---------------------------------------------------------------------------
# plan_quality report column — additive interim surface (step-11/12)
#
# A distinct per-(task_id, config_name, role_under_test) column μ/λ consume in
# the interim — NOT a change to the Elo build_report/format_markdown schema (the
# full C4 composite+plan_quality row is owned by λ, which θ does not depend on).
# Mirrors η's TestRecoveryReport precisely: a populated float for architect runs,
# the '-' null sentinel otherwise, sorted deterministically, rendering
# byte-identically regardless of input order.
# ---------------------------------------------------------------------------

def _architect_result(
    task_id: str = 'df_task_2605',
    config_name: str = 'architect-sonnet-high',
    plan_quality: float = 0.75,
):
    from orchestrator.evals.runner import EvalResult

    return EvalResult(
        task_id=task_id,
        config_name=config_name,
        outcome='done',
        metrics={
            'role_under_test': 'architect',
            'plan_quality': plan_quality,
            'composite_score': 0.0,
        },
        worktree_path='/tmp/wt-arch',
    )


def _implementer_result(
    task_id: str = 'df_task_1993',
    config_name: str = 'opus-high',
):
    from orchestrator.evals.runner import EvalResult

    # No plan_quality / role_under_test keys at all → the null sentinel path.
    return EvalResult(
        task_id=task_id,
        config_name=config_name,
        outcome='done',
        metrics={'composite_score': 1.0},
        worktree_path='/tmp/wt-impl',
    )


class TestPlanQualityReport:
    def test_build_plan_quality_report_rows(self):
        from orchestrator.evals.report import build_plan_quality_report

        report = build_plan_quality_report(
            [_architect_result(), _implementer_result()]
        )
        rows = report['rows']
        by_task = {r['task_id']: r for r in rows}

        arch = by_task['df_task_2605']
        assert arch['plan_quality'] == 0.75
        assert arch['role_under_test'] == 'architect'
        assert arch['config_name'] == 'architect-sonnet-high'

        impl = by_task['df_task_1993']
        assert impl['plan_quality'] is None  # null sentinel
        assert impl['role_under_test'] is None

    def test_rows_sorted_by_task_id_then_config_name(self):
        from orchestrator.evals.report import build_plan_quality_report

        # Deliberately unsorted input across both task_id and config_name.
        results = [
            _architect_result(task_id='df_task_2605', config_name='architect-sonnet-high'),
            _architect_result(task_id='df_task_1000', config_name='architect-opus-high'),
            _architect_result(task_id='df_task_2605', config_name='architect-opus-high'),
        ]
        report = build_plan_quality_report(results)
        keys = [(r['task_id'], r['config_name']) for r in report['rows']]
        assert keys == sorted(keys)

    def test_format_plan_quality_table_renders_column(self):
        from orchestrator.evals.report import (
            build_plan_quality_report,
            format_plan_quality_table,
        )

        report = build_plan_quality_report(
            [_architect_result(), _implementer_result()]
        )
        table = format_plan_quality_table(report)
        assert 'plan_quality' in table

        arch_line = next(
            ln for ln in table.splitlines() if 'df_task_2605' in ln
        )
        assert '0.7500' in arch_line     # populated float for the architect row
        assert 'architect' in arch_line  # role_under_test column

        impl_line = next(
            ln for ln in table.splitlines() if 'df_task_1993' in ln
        )
        assert '-' in impl_line           # null sentinel rendered as '-'
        assert '0.7500' not in impl_line  # implementer row is NOT populated

    def test_table_renders_byte_identically_regardless_of_input_order(self):
        from orchestrator.evals.report import (
            build_plan_quality_report,
            format_plan_quality_table,
        )

        results = [_architect_result(), _implementer_result()]
        a = format_plan_quality_table(build_plan_quality_report(results))
        b = format_plan_quality_table(
            build_plan_quality_report(list(reversed(results)))
        )
        assert a == b  # deterministic (sorted rows, no wall-clock dependence)


# ---------------------------------------------------------------------------
# CLI dispatch — _run_single_eval routes architect candidates (step-13/14)
#
# The user-observable signal: `orchestrator eval` with an architect-role
# candidate (role=='architect') routes to run_architect_eval (NOT run_eval) and
# surfaces the per-fixture plan_quality score. An ordinary implementer config
# still routes to run_eval, unchanged. Hermetic: both runners and config
# resolution are patched — no live worktree, no LLM.
# ---------------------------------------------------------------------------

def _fake_eval_result(**over):
    from orchestrator.evals.runner import EvalResult

    defaults: dict = dict(
        task_id='df_task_2605',
        config_name='x',
        outcome='done',
        metrics={'composite_score': 1.0},
        worktree_path='/tmp/wt',
        wall_clock_ms=1000,
    )
    defaults.update(over)
    return EvalResult(**defaults)


def _dispatch_single_eval(cfg, capsys):
    """Drive ``cli._run_single_eval`` with a resolved config, both runners patched.

    ``get_config_by_name`` is patched to return ``cfg`` so the dispatch is
    exercised purely on ``cfg.role``. Returns ``(out, run_eval, run_architect)``.
    """
    from orchestrator import cli

    arch_result = _fake_eval_result(
        config_name=cfg.name,
        metrics={'role_under_test': 'architect', 'plan_quality': 0.75},
    )
    impl_result = _fake_eval_result(config_name=cfg.name)

    mock_run_eval = AsyncMock(return_value=impl_result)
    mock_run_arch = AsyncMock(return_value=arch_result)

    with contextlib.ExitStack() as es:
        p = es.enter_context
        p(patch('orchestrator.evals.runner.run_eval', mock_run_eval))
        p(patch('orchestrator.evals.runner.run_architect_eval', mock_run_arch))
        p(patch('orchestrator.evals.configs.get_config_by_name',
                MagicMock(return_value=cfg)))
        cli._run_single_eval(
            Path('/fake/task.json'), cfg.name, base_config=MagicMock(),
        )
    out = capsys.readouterr().out
    return out, mock_run_eval, mock_run_arch


class TestCliArchitectDispatch:
    def _arch_cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role='architect',
        )

    def _impl_cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig('opus-high', 'claude', 'opus', 'high')

    def test_architect_config_routes_to_run_architect_eval(self, capsys):
        _, run_eval, run_arch = _dispatch_single_eval(self._arch_cfg(), capsys)
        run_arch.assert_called_once()
        run_eval.assert_not_called()

    def test_architect_dispatch_surfaces_plan_quality(self, capsys):
        out, _, _ = _dispatch_single_eval(self._arch_cfg(), capsys)
        # The per-fixture plan-quality score is echoed to the operator.
        assert 'plan_quality' in out
        assert '0.75' in out

    def test_implementer_config_still_routes_to_run_eval(self, capsys):
        _, run_eval, run_arch = _dispatch_single_eval(self._impl_cfg(), capsys)
        run_eval.assert_called_once()
        run_arch.assert_not_called()
