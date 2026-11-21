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

import functools
import logging
from pathlib import Path

import pytest
from _architect_eval_harness import (
    PROXY_ENV,
    arch_task_no_reference,
    architect_prompt_sent,
    only_record,
    run_architect_eval_hermetic,
    token_agent_result,
)
from _architect_plan_shapes import create_plan_stub, stub_with_steps, well_formed_plan
from _eval_refusal import REFUSAL_TEXT, refused_agent_result
from shared.cli_invoke import AllAccountsCappedException

from orchestrator.evals.judge import PlanQualityVerdict
from orchestrator.evals.measurement import (
    MeasurementStage,
    Scored,
    Unmeasured,
    UnmeasuredCause,
)
from orchestrator.evals.metrics import EvalMetrics, RoleUnderTest
from orchestrator.evals.results import ResultStore

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
# EvalMetrics.invocation_errors / cap_tainted fields (task 3118, task 6734)
#
# The infra-failure markers that let a reader tell "the model wrote a terrible
# plan" from "we never got to ask the model": ``invocation_errors`` records
# WHAT happened, one typed record per unmeasured stage; ``cap_tainted`` is the
# machine-checkable exclusion predicate every aggregate keys on.
# ---------------------------------------------------------------------------

class TestEvalMetricsInvocationErrorField:
    def test_invocation_errors_default_is_none(self):
        assert EvalMetrics().invocation_errors is None

    def test_cap_tainted_default_is_false(self):
        assert EvalMetrics().cap_tainted is False

    def test_to_dict_carries_invocation_errors_key_defaulting_none(self):
        d = EvalMetrics().to_dict()
        assert 'invocation_errors' in d
        assert d['invocation_errors'] is None
        assert 'invocation_error' not in d

    def test_to_dict_carries_cap_tainted_key_defaulting_false(self):
        d = EvalMetrics().to_dict()
        assert 'cap_tainted' in d
        assert d['cap_tainted'] is False

    def test_judged_without_reference_default_is_false(self):
        """False, not None: an absent marker means "no degradation observed".

        Exactly how ``cap_tainted`` behaves — a result persisted before the
        field existed reads back as not-degraded rather than as unknown.
        """
        assert EvalMetrics().judged_without_reference is False

    def test_to_dict_carries_judged_without_reference_key_defaulting_false(
        self,
    ):
        """The key must SERIALIZE on healthy cells, carrying False.

        Load-bearing cross-consumer contract (task 3628 σ → task 3632):
        ``scripts/run_fable_trial_v2_campaign.py`` decides per-cell reference
        validity by KEY PRESENCE — its ``MARKER_KEY not in metrics`` means "not
        known-good", never False, and the count it consumes from
        ``cell_admission.judged_blind_count`` is None for any config with even one
        keyless cell. If this key
        were emitted only when True (a conditional dict update, or filtering
        falsy fields out of the persisted dict), every HEALTHY cell would be
        indistinguishable from a pre-σ one and that consumer would report None
        for every candidate forever.

        Spelled as an ``in`` check FOLLOWED BY an ``is False`` check on
        purpose: a single ``d['judged_without_reference'] is False`` would pass
        just as well against a conditional-emit implementation that omits the
        key precisely in the healthy case.
        """
        d = EvalMetrics().to_dict()
        assert 'judged_without_reference' in d
        assert d['judged_without_reference'] is False


# ---------------------------------------------------------------------------
# Real AgentResult refusal shapes the runner tests feed the architect. The
# classifier's own unit tests live in test_eval_measurement.py.
# ---------------------------------------------------------------------------

def _codex_cap_agent_result():
    """A CODEX cap body — matched only by CODEX_CAP_PATTERNS, never by claude's.

    Deliberately carries no ``api_error_status`` and none of the claude
    CAP_HIT_PREFIXES, so the ONLY route to a marker is the codex-backend table:
    the payload is a live probe of whether ``backend=`` actually reached
    ``classify_invocation``.
    """
    from shared.cli_invoke import AgentResult

    return AgentResult(
        success=False,
        output='stream error: usage limit reached for this account',
        cost_usd=0.0,
        duration_ms=900,
        turns=0,
        subtype='error',
    )


def _wedged_agent_result():
    """A zero-output CLI wedge: full timeout, zero transcript turns, zero cost."""
    from shared.cli_invoke import AgentResult

    return AgentResult(
        success=False,
        output='',
        cost_usd=0.0,
        duration_ms=600_000,
        turns=0,
        timed_out=True,
        transcript_turns=0,
    )


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

@pytest.mark.asyncio
class TestRunArchitectEval:
    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    async def test_scores_plan_freezes_downstream_and_persists(self):
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(),
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

        # Persisted through the store.
        mocks['save'].assert_called_once()

    async def test_unparseable_judge_degrades_to_floor_and_records_it(self):
        from orchestrator.evals.judge import score_plan_structure

        plan = well_formed_plan()
        # An unparseable judge answer → run_architect_eval degrades to the
        # deterministic structural floor, never a null — and, unlike before
        # task 6734, the cell now says the judge's answer was unusable.
        result, _ = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=plan,
            judge_return=Unmeasured(
                MeasurementStage.JUDGE, UnmeasuredCause.UNPARSEABLE,
                'JSONDecodeError: Expecting value', cost_usd=0.6,
            ),
        )
        assert result.metrics['plan_quality'] == score_plan_structure(plan)
        assert result.metrics['role_under_test'] == 'architect'
        assert result.metrics['invocation_errors'] == [{
            'stage': 'judge', 'cause': 'unparseable',
            'reason': 'JSONDecodeError: Expecting value',
        }]
        assert result.metrics['cap_tainted'] is False
        assert result.metrics['judge_invocations'] == 1
        assert result.metrics['judge_cost_usd'] == pytest.approx(0.6)
        assert result.metrics['judged_without_reference'] is False

    async def test_judge_raising_still_yields_non_sentinel_score(self):
        from orchestrator.evals.judge import score_plan_structure

        plan = well_formed_plan()
        # Even if the judge RAISES, plan_quality must be the deterministic floor.
        result, _ = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=plan,
            judge_side_effect=RuntimeError('judge exploded'),
        )
        assert result.metrics['plan_quality'] == score_plan_structure(plan)
        assert result.metrics['plan_quality'] is not None

        # Amendment (reviewer test-coverage): a judge that RAISES must not
        # fabricate spend — no invocation credited, no cost, and the cell's
        # cost_usd is the architect's own spend alone (the harness's default
        # invoke_agent mock reports cost_usd=1.23). A regression that moved
        # judge_invocations=1 above the raise, or invented a cost inside the
        # except block, would previously go unnoticed here.
        assert result.metrics['judge_cost_usd'] == 0.0
        assert result.metrics['judge_invocations'] == 0
        assert result.metrics['cost_usd'] == pytest.approx(1.23)
        # ...but the raise must not look identical to a judge that was
        # SKIPPED either: it is recorded so the cell reads "spend unknown",
        # not silently judge-free.
        assert result.metrics['invocation_errors'] == [{
            'stage': 'judge', 'cause': 'raised',
            'reason': 'RuntimeError: judge exploded',
        }]

    async def test_architect_timeout_maps_to_timeout_outcome(self):
        # A hung architect invoke surfaces as TimeoutError — what asyncio.wait_for
        # raises when the operator's --timeout expires. run_architect_eval must
        # honor the timeout and map it to outcome='timeout' (mirroring run_eval),
        # NOT the generic 'blocked', while still emitting a non-sentinel
        # plan_quality float.
        result, _ = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            invoke_side_effect=TimeoutError(),
        )
        assert result.outcome == 'timeout'
        assert isinstance(result.metrics['plan_quality'], float)
        assert result.metrics['role_under_test'] == 'architect'

    async def test_timeout_is_marked_but_keeps_scoring_on_content(self):
        """A timeout is MARKED yet deliberately NOT tainted (asymmetry pinned).

        Marked: without a marker the cell was byte-indistinguishable from a
        genuinely terrible plan — the same defect the cap path removes.

        Not tainted: unlike a cap hit (a property of the SCHEDULE), a timeout is
        CANDIDATE-attributable — the model was asked and did not finish inside
        the operator's budget. Excluding it would let a pathologically slow
        candidate dodge the penalty its competitors paid, so the cell keeps
        scoring on content and carries the reliability signal in BOTH
        ``outcome='timeout'`` and ``invocation_errors``.
        """
        result, _ = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            invoke_side_effect=TimeoutError(),
        )
        assert result.outcome == 'timeout'
        assert result.metrics['invocation_errors'] == [{
            'stage': 'architect', 'cause': 'timeout',
            'reason': 'no answer within 60m',
        }]
        assert result.metrics['cap_tainted'] is False
        assert result.metrics['plan_quality'] is not None

    @pytest.mark.parametrize('arch_result_factory', [
        None,
        pytest.param(functools.partial(refused_agent_result, cost_usd=0.0),
                     id='refused_agent_result'),
    ])
    async def test_plan_only_cell_records_tests_pass_as_unknown(
        self, arch_result_factory,
    ):
        """A plan-only cell collected NO test signal — ``tests_pass`` is None.

        ``run_architect_eval`` freezes implementer/debugger/reviewer/verify, so
        no test is ever run. The two wrong answers are both live risks:

        - ``False`` (today's dataclass default) reads as "the tests FAILED",
          which is what hard-gates ``blend_composite`` to 0.0 and collapses
          every architect row's composite — the defect this task removes.
        - ``True`` would fabricate a pass for a run that never invoked verify,
          and would additionally let a ~$0.30/60s plan-only cell set the
          per-fixture cost/latency FLOOR that ``build_composite_report`` draws
          from PASSING trials, deflating the ~$5/900s full-workflow implementer
          rows that ``ofat_candidates()`` puts in the same result set.

        Pinned on BOTH the healthy and the cap-tainted path: the absence of a
        test signal is a property of the plan-only MODE, not of the outcome.
        """
        result, _ = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            arch_result=arch_result_factory() if arch_result_factory else None,
        )

        assert result.metrics['tests_pass'] is None
        assert result.metrics['tests_pass'] is not False
        assert result.metrics['tests_pass'] is not True

    async def test_cap_refusal_that_left_a_plan_keeps_the_structural_floor(self):
        """A cap landing MID-run, after plan.json was already written.

        The common shape of a session-limit hit during a long campaign: the
        architect wrote its plan through plan-tools MCP, THEN the CLI 429'd. The
        taint decision must consult the ARTIFACT, not just the refusal — a real
        plan is a real content measurement, and nulling it would both throw that
        measurement away and persist a self-contradictory cell (``plan_steps``
        > 0 alongside "we never got to ask the model").
        """
        from orchestrator.evals.judge import score_plan_structure

        plan = well_formed_plan()
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=plan,
            arch_result=refused_agent_result(cost_usd=0.0),
        )

        # The measurement SURVIVES: scored on the deterministic floor, kept in
        # the aggregate — symmetric with the judge-only refusal.
        assert result.metrics['plan_quality'] == score_plan_structure(plan)
        assert result.metrics['cap_tainted'] is False
        # ...and the cell is self-consistent: a plan with steps, not a null.
        assert result.metrics['plan_steps'] == len(plan['steps'])

        # The refusal is still RECORDED, so a reader knows the LLM judge never
        # ran on this cell...
        assert only_record(result.metrics) == {
            'stage': 'architect', 'cause': 'cap_hit', 'reason': REFUSAL_TEXT,
        }
        # ...and the judge is still skipped: in the same cap window it would 429
        # too, and the floor is the exact degradation a judge failure takes.
        mocks['judge'].assert_not_called()

    @pytest.mark.parametrize('empty_steps', [[], None])
    async def test_cap_after_create_plan_stub_is_tainted_not_scored_zero(
        self, empty_steps,
    ):
        """The middle case: a cap landing right after ``create_plan``.

        The architect reached exactly ONE plan-tools call before the 429, so
        ``plan.json`` holds the header-only stub ``_create_plan`` writes — a
        TRUTHY dict with zero steps. That is NOT a content measurement: there
        are no steps to score.

        BEFORE this fix the runner's taint predicate was raw dict truthiness
        (``not plan``), so this stub read as "a real plan landed" → the cell was
        left ``cap_tainted=False`` and handed to ``score_plan_structure``, which
        short-circuits to 0.0 for want of steps. Verified against the harness:
        ``cap_tainted=False, plan_quality=0.0``. That fabricated zero is then
        AVERAGED into ``mean_plan_quality`` with neither exclusion surface
        counting it — precisely the defect this task exists to remove, reopened
        by the one plan shape a mid-run cap most often produces.

        Parameterised over both empty spellings (``[]`` and ``None``) so the
        predicate is pinned to "has steps", not to the empty-list spelling.
        """
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=stub_with_steps(empty_steps),
            arch_result=refused_agent_result(cost_usd=0.0),
        )

        assert result.metrics['cap_tainted'] is True
        assert result.metrics['plan_quality'] is None
        assert result.metrics['plan_quality'] != 0.0
        assert result.metrics['plan_steps'] == 0

        record = only_record(result.metrics)
        assert (record['stage'], record['cause']) == ('architect', 'cap_hit')

        # Nothing to judge, and inside the cap window the judge would 429 too.
        mocks['judge'].assert_not_called()

    async def test_wedged_architect_invoke_is_marked_and_excluded(self):
        # A zero-output wedge (full timeout, zero transcript turns, zero cost)
        # produced no model answer at all — unmeasurable for the same reason a
        # 429 is, so it must not be scored 0.0 either (reviewer: robustness).
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan={},
            arch_result=_wedged_agent_result(),
        )
        assert result.metrics['cap_tainted'] is True
        assert result.metrics['plan_quality'] is None
        record = only_record(result.metrics)
        assert (record['stage'], record['cause']) == ('architect', 'wedge')
        mocks['judge'].assert_not_called()

    async def test_harness_exception_is_marked_and_excluded(self):
        # OUR crash, not the candidate's: the architect was never even asked, so
        # charging a fabricated 0.0 to the candidate would be plainly wrong.
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan={},
            invoke_side_effect=RuntimeError('worktree exploded'),
        )
        assert result.outcome == 'blocked'
        assert result.metrics['cap_tainted'] is True
        assert result.metrics['plan_quality'] is None
        record = only_record(result.metrics)
        assert (record['stage'], record['cause']) == ('architect', 'harness_error')
        assert record['reason'].startswith('RuntimeError')
        mocks['judge'].assert_not_called()

    async def test_config_backend_reaches_the_invocation_classifier(self):
        """The ``backend=`` argument is load-bearing, so pin it end-to-end.

        ``classify_invocation``'s codex cap table fires ONLY for the codex
        backend. Feeding a codex-shaped cap body through a codex candidate must
        mark the cell; the SAME body through a claude candidate must not — a
        regression that dropped or hard-coded ``backend=config.backend`` would
        silently un-mark every codex/gemini cap hit while every claude test kept
        passing.
        """
        from orchestrator.evals.configs import EvalConfig

        codex_cfg = EvalConfig(
            'architect-codex-high', 'codex', 'gpt-5', 'high', role=RoleUnderTest.ARCHITECT,
        )
        result, _ = await run_architect_eval_hermetic(
            codex_cfg, produced_plan={}, arch_result=_codex_cap_agent_result(),
        )
        record = only_record(result.metrics)
        assert (record['stage'], record['cause']) == ('architect', 'cap_hit')
        assert result.metrics['cap_tainted'] is True

        # Same payload, claude candidate: the codex table is never consulted.
        claude_result, _ = await run_architect_eval_hermetic(
            self._cfg(), produced_plan={}, arch_result=_codex_cap_agent_result(),
        )
        assert claude_result.metrics['invocation_errors'] is None
        assert claude_result.metrics['cap_tainted'] is False

    async def test_cap_refused_architect_invoke_is_marked_not_scored_zero(self):
        """A CLI 429 must be MARKED as infra, never scored as a terrible plan.

        BEFORE this fix the exact same input recorded ``plan_quality=0.0`` with
        no marker: ``result.success`` was False, so the plan artifact read back
        empty, the plan judge was invoked anyway (429ing in the same cap
        window), its parse-failure sentinel degraded to
        ``score_plan_structure({}) == 0.0``, and the persisted result JSON was
        byte-indistinguishable from a genuinely terrible plan. Recovering those
        cells cost a hand-correlation of result-JSON mtimes against 429 payloads
        in the run log (plans/eval-architect-effort-verdict-2026-07-27.md,
        "Data hygiene": 3 of 22 fixtures).
        """
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan={},
            arch_result=refused_agent_result(cost_usd=0.0),
        )

        # The cell is marked as NOT a content measurement, naming the refused
        # STAGE, the typed cause and the CLI's own words.
        assert result.metrics['cap_tainted'] is True
        assert result.metrics['invocation_errors'] == [
            {'stage': 'architect', 'cause': 'cap_hit', 'reason': REFUSAL_TEXT},
        ]

        # ...so the score is the explicit null, NOT a fabricated zero.
        assert result.metrics['plan_quality'] is None
        assert result.metrics['plan_quality'] != 0.0
        assert result.metrics['role_under_test'] == 'architect'
        assert result.metrics['plan_steps'] == 0

        # The plan judge is SKIPPED entirely: there is nothing to judge, and in
        # a cap window that invocation would 429 too — the second-order failure
        # that manufactured the 0.0.
        mocks['judge'].assert_not_called()
        # Still persisted, so the marked cell is recoverable from the JSON.
        mocks['save'].assert_called_once()

    async def test_judge_cap_keeps_structural_floor_and_records_marker(self):
        """A JUDGE-only refusal is marked but must NOT taint the cell.

        The architect ran fine and produced a real plan, so the deterministic
        structural score is genuinely derived from model CONTENT. Nulling it
        would throw away a valid measurement and shrink n for no reason — taint
        is keyed strictly on the ARCHITECT-side refusal, the case where no model
        content exists at all. The marker is still stamped so a reader knows the
        LLM judge never ran on this cell.
        """
        from orchestrator.evals.judge import score_plan_structure

        plan = well_formed_plan()
        result, _ = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=plan,
            judge_return=Unmeasured(
                MeasurementStage.JUDGE, UnmeasuredCause.CAP_HIT, REFUSAL_TEXT,
                cost_usd=0.4,
            ),
        )

        # The content-derived floor is KEPT — not nulled, not excluded.
        assert result.metrics['plan_quality'] == score_plan_structure(plan)
        assert result.metrics['cap_tainted'] is False

        # ...but the judge's refusal is recorded, named by STAGE, and the
        # refused call's spend is still charged.
        assert result.metrics['invocation_errors'] == [
            {'stage': 'judge', 'cause': 'cap_hit', 'reason': REFUSAL_TEXT},
        ]
        assert result.metrics['judge_invocations'] == 1
        assert result.metrics['judge_cost_usd'] == pytest.approx(0.4)

    async def test_healthy_run_carries_no_marker_and_is_not_tainted(self):
        # The untouched baseline: nothing refused anywhere → no record at all.
        result, _ = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(),
        )
        assert result.metrics['invocation_errors'] is None
        assert result.metrics['cap_tainted'] is False
        assert result.metrics['plan_quality'] == 0.77

    async def test_wires_plan_tools_mcp_into_architect_invoke(self):
        # BUG 1: the eval architect must be wired with plan-tools MCP exactly
        # like real dispatch — otherwise read_plan() returns {} → plan_steps=0 →
        # plan_quality collapses to the structural floor (esc-df_task_12-3).
        # run_architect_eval must pass a non-None mcp_config to invoke_agent whose
        # 'plan-tools' entry launches the direct-interpreter no-uv hot path against
        # the RELOCATED .task-meta/<name>/ root (matching production
        # _inject_plan_tools_mcp), so the plan-tools server writes plan.json where
        # the TaskArtifacts(worktree, meta_root=_meta_root_for_worktree(worktree))
        # readback reads it. That meta root is also its task root (task 3155):
        # the eval stays single-root, since parallel trials of one task id would
        # otherwise share one .task-meta/task-<id>.
        import sys

        from orchestrator.agents.roles import ARCHITECT
        from orchestrator.workflow import _meta_root_for_worktree

        _, mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(),
        )

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
            '--task-root', str(expected_meta_root),
            '--grants-enforced',
            *[
                token for name in sorted(ARCHITECT.plan_tools_grants)
                for token in ('--grant', name)
            ],
        ]


@pytest.mark.asyncio
class TestArchitectCellCarriesReplayFrame:
    """The architect is briefed in the replay frame and the cell says so (4844)."""

    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    async def test_architect_prompt_is_production_briefing_plus_replay_frame(self):
        _, mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(),
        )

        assert mocks['invoke'].call_args.kwargs['prompt'] == architect_prompt_sent()

    async def test_cell_is_stamped_with_the_replay_frame(self):
        from orchestrator.evals.replay_frame import REPLAY_FRAME_ID

        result, _ = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(),
        )

        assert result.metrics['replay_frame'] == REPLAY_FRAME_ID

    @pytest.mark.parametrize('unmeasurable_cell', [
        pytest.param(
            lambda: {'produced_plan': well_formed_plan(),
                     'invoke_side_effect': RuntimeError('boom')},
            id='harness-error',
        ),
        pytest.param(
            lambda: {'produced_plan': {}, 'arch_result': refused_agent_result(cost_usd=0.0)},
            id='cap-tainted',
        ),
    ])
    async def test_stamp_survives_unmeasurable_cells(self, unmeasurable_cell):
        from orchestrator.evals.replay_frame import REPLAY_FRAME_ID

        result, _ = await run_architect_eval_hermetic(
            self._cfg(), **unmeasurable_cell(),
        )

        assert result.metrics['replay_frame'] == REPLAY_FRAME_ID


# ---------------------------------------------------------------------------
# Architect cell records the plan judge's spend (step-5/6, eval-revival υ)
#
# run_architect_eval built its EvalMetrics from the architect's OWN spend
# only, discarding the plan judge's opus/effort=max invocation cost entirely
# — every architect cell persisted judge_cost_usd=0.0. judge_cost_usd is a
# SUBSET of cost_usd, not disjoint (metrics.py:69-71), so the cell's cost_usd
# must become architect spend + judge spend, with judge_cost_usd as the
# subset breakdown — mirroring the implementer path's existing semantics
# (metrics.py:597-598 / workflow.py:7872-7875).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestArchitectCellRecordsJudgeSpend:
    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    async def test_judge_spend_recorded_and_folded_into_cell_cost(self):
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            judge_return=Scored(
                PlanQualityVerdict(plan_quality=0.77, per_criterion={}, reasoning='good'),
                cost_usd=0.42,
            ),
        )

        # The RETURNED result...
        assert result.metrics['judge_cost_usd'] == pytest.approx(0.42)
        assert result.metrics['judge_invocations'] == 1
        # ...the SUBSET invariant: architect spend (1.23, the harness default)
        # PLUS the judge spend, never the architect spend alone.
        assert result.metrics['cost_usd'] == pytest.approx(1.23 + 0.42)

        # ...and what was actually PERSISTED through the store, not just what
        # the function happens to return.
        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['judge_cost_usd'] == pytest.approx(0.42)
        assert persisted['judge_invocations'] == 1
        assert persisted['cost_usd'] == pytest.approx(1.23 + 0.42)


@pytest.mark.asyncio
class TestArchitectCellStampsTokenUsageAndProxySignal:
    """Task 3656 step-1: the architect cell must record the run's TOKEN USAGE
    and its PROXY signal, not leave all three at their dataclass defaults.

    These are the three inputs :func:`~orchestrator.evals.metrics.resolve_cost_usd`
    needs to resolve cost provenance per Invariant P5. Without them the
    price-table path is unreachable on the architect side no matter what the
    resolver is asked, so they are pinned BEFORE the resolution itself
    (step-5). Every assertion checks the PERSISTED cell as well as the returned
    result — the persisted JSON is what the report surfaces and an operator read.
    """

    def _cfg(self, **kw):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high',
            role=RoleUnderTest.ARCHITECT, **kw,
        )

    async def test_reported_token_counts_are_stamped(self):
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            arch_result=token_agent_result(12_000, 3_000),
        )

        assert result.metrics['input_tokens'] == 12_000
        assert result.metrics['output_tokens'] == 3_000

        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['input_tokens'] == 12_000
        assert persisted['output_tokens'] == 3_000

    async def test_unreported_token_counts_persist_as_zero_not_none(self):
        """``AgentResult`` declares both ``int | None``; a provider that did not
        report must persist an honest ``0``, never a ``None`` that would poison
        the arithmetic in ``resolve_cost_usd`` (and ``json`` a bare null into a
        field the report layer treats as an int)."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            arch_result=token_agent_result(None, None),
        )

        assert result.metrics['input_tokens'] == 0
        assert result.metrics['output_tokens'] == 0
        assert result.metrics['input_tokens'] is not None
        assert result.metrics['output_tokens'] is not None

        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['input_tokens'] == 0
        assert persisted['output_tokens'] == 0
        # Same ``int | None`` contract for the cache counts (this AgentResult
        # leaves both at their None default).
        assert persisted['cache_read_tokens'] == 0
        assert persisted['cache_create_tokens'] == 0

    async def test_cache_tokens_and_turns_are_stamped(self):
        """The REST of the token profile ``collect_metrics`` stamps.

        These price nothing, but a native Claude run is cache-read dominated:
        a cell reporting 12k input / 3k output beside a zeroed cache block
        reads as a far smaller run than it was, so the persisted evidence must
        be symmetric with the implementer path's (reviewer: completeness).
        """
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(),
        )

        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['cache_read_tokens'] == 48_000
        assert persisted['cache_create_tokens'] == 6_000
        assert persisted['turns_used'] == 9
        # output_tokens ÷ that leg's own duration — the identical formula
        # collect_metrics uses, over the same duration workflow_duration_ms
        # carries.
        assert persisted['workflow_duration_ms'] == 4567
        assert persisted['tokens_per_second'] == pytest.approx(
            round(3_000 / (4567 / 1000), 2),
        )
        assert result.metrics['cache_read_tokens'] == 48_000

    async def test_zero_duration_cell_reports_zero_tokens_per_second(self):
        """A timed-out cell has no duration to divide by — 0.0, not a
        ZeroDivisionError that would lose the whole cell."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            invoke_side_effect=TimeoutError(),
        )

        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['tokens_per_second'] == 0.0
        assert persisted['turns_used'] == 0
        assert result.metrics['tokens_per_second'] == 0.0

    async def test_proxied_candidate_is_flagged_local(self):
        """``ANTHROPIC_BASE_URL`` set == a PROXIED endpoint, the signal Invariant
        P5 keys on to stop trusting the CLI's own cost figure."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(env_overrides={'ANTHROPIC_BASE_URL': 'http://localhost:8000/v1'}),
            produced_plan=well_formed_plan(),
        )

        assert result.metrics['is_local_model'] is True
        assert mocks['save'].call_args.args[0].metrics['is_local_model'] is True

    async def test_native_candidate_is_not_flagged_local(self):
        """Today's shape (``architect-opus-max`` / ``architect-fable-max``):
        native cloud, no proxy, so the CLI cost stays trustworthy."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(env_overrides={}),
            produced_plan=well_formed_plan(),
        )

        assert result.metrics['is_local_model'] is False
        assert mocks['save'].call_args.args[0].metrics['is_local_model'] is False


@pytest.mark.asyncio
class TestArchitectCellResolvesCostProvenance:
    """Task 3656 step-5: the architect cell's cost must be RESOLVED through
    :func:`~orchestrator.evals.metrics.resolve_cost_usd`, not copied from the
    raw CLI figure with an UNVERIFIED ``cost_source='cli'`` dataclass default.

    Invariant P5: a PROXIED endpoint's CLI cost figure is untrustworthy, so a
    proxied architect candidate must land on the price table (or the defined,
    LOUD fallback) rather than keeping the proxy's number. ``collect_metrics``
    (the implementer path) has resolved through this seam since task 2820;
    ``run_architect_eval`` bypassed it entirely.

    The plan judge is held to ZERO spend throughout (the harness's default
    ``PlanQualityVerdict`` carries ``cost_usd=0.0``) so these cases isolate the
    ARCHITECT component — composing the two-component label is step-7's.
    """

    def _cfg(self, **kw):
        from orchestrator.evals.configs import EvalConfig

        kw.setdefault('model', 'sonnet')
        return EvalConfig(
            'architect-candidate', 'claude', effort='high', role=RoleUnderTest.ARCHITECT, **kw,
        )

    async def test_native_candidate_keeps_the_cli_figure_verbatim(self):
        """REGRESSION PIN for every architect cell that has already landed.

        ``architect-opus-max`` / ``architect-fable-max`` are native cloud
        (``env_overrides={}``) and unlisted in the price table, so
        ``resolve_cost_usd`` takes its ``not is_local_model`` branch and returns
        ``(cli_cost_usd, 'cli')`` VERBATIM. Both the figure and the label must
        be byte-identical to today's — the change bites only a PROXIED
        candidate, and turns today's UNVERIFIED 'cli' into a DERIVED one.
        """
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(env_overrides={}),
            produced_plan=well_formed_plan(),
            orch_prices={},
        )

        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['cost_source'] == 'cli'
        assert persisted['cost_usd'] == pytest.approx(1.23)
        assert result.metrics['cost_source'] == 'cli'
        assert result.metrics['cost_usd'] == pytest.approx(1.23)

    async def test_priced_proxied_candidate_uses_the_price_table(self):
        """A proxied candidate whose model IS listed must be priced from the
        table, NOT from the proxy's own (wrong) CLI figure."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(model='qwen3-coder', env_overrides=dict(PROXY_ENV)),
            produced_plan=well_formed_plan(),
            arch_result=token_agent_result(1_000_000, 500_000, cost_usd=99.0),
            orch_prices={'qwen3-coder': {'input_per_1m': 0.30, 'output_per_1m': 1.20}},
        )

        expected = (1_000_000 * 0.30 + 500_000 * 1.20) / 1_000_000
        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['cost_source'] == 'price_table'
        assert persisted['cost_usd'] == pytest.approx(expected)
        assert persisted['cost_usd'] == pytest.approx(0.90)
        # Emphatically NOT the raw proxied CLI figure.
        assert persisted['cost_usd'] != pytest.approx(99.0)
        assert result.metrics['cost_source'] == 'price_table'

    async def test_unpriced_proxied_candidate_warns_and_uses_defined_fallback(
        self, caplog,
    ):
        """A proxied candidate with NO price entry degrades to the DEFINED
        ``_FALLBACK_PRICE`` and says so loudly — never a silent raw-CLI
        number (the loud-over-silent-degradation norm)."""
        from orchestrator.evals.metrics import _FALLBACK_PRICE

        with caplog.at_level(logging.WARNING, logger='orchestrator.evals.metrics'):
            result, mocks = await run_architect_eval_hermetic(
                self._cfg(model='qwen3-coder', env_overrides=dict(PROXY_ENV)),
                produced_plan=well_formed_plan(),
                arch_result=token_agent_result(1_000_000, 500_000, cost_usd=99.0),
                orch_prices={},
            )

        # Import the constant rather than hardcoding 2.0/8.0, so a future rate
        # change updates this expectation instead of leaving a stale magic
        # number (the single-home discipline of task 2459).
        expected = (
            1_000_000 * _FALLBACK_PRICE['input_per_1m']
            + 500_000 * _FALLBACK_PRICE['output_per_1m']
        ) / 1_000_000
        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['cost_source'] == 'unpriced_proxy'
        assert persisted['cost_usd'] == pytest.approx(expected)
        assert persisted['cost_usd'] != pytest.approx(99.0)
        assert result.metrics['cost_source'] == 'unpriced_proxy'

        warnings = [
            r.message for r in caplog.records
            if r.levelno >= logging.WARNING
            and 'No configured price for proxied-endpoint model' in r.message
        ]
        assert warnings, (
            'Expected the LOUD unpriced-proxy WARNING from '
            'orchestrator.evals.metrics; got '
            f'{[r.message for r in caplog.records]}'
        )

    async def test_orch_config_never_built_still_lands_a_cell(self):
        """The price table lives on the eval orch config, which is built INSIDE
        the try — so a harness crash before it is bound must leave an explicit
        ``None`` the resolver tolerates, never an ``UnboundLocalError`` /
        ``AttributeError`` that loses the whole cell.

        That path never invoked the architect (0 tokens, $0.00), so no price
        table could have changed the number anyway.
        """
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(env_overrides={}),
            produced_plan=well_formed_plan(),
            orch_config_side_effect=RuntimeError('config build exploded'),
        )

        persisted = mocks['save'].call_args.args[0].metrics
        # The cell LANDED — marked and excluded, not lost.
        assert persisted['cap_tainted'] is True
        assert [r['cause'] for r in persisted['invocation_errors']] == ['harness_error']
        assert persisted['cost_usd'] == pytest.approx(0.0)
        # A concrete label, not merely "some member of the vocabulary": nothing
        # was spent, so the resolution is skipped and the documented 'cli'
        # default stands (reviewer: an `in {the whole vocabulary}` check cannot
        # fail, so it pinned nothing).
        assert persisted['cost_source'] == 'cli'
        assert result.metrics['cap_tainted'] is True

    async def test_zero_spend_proxied_cell_does_not_warn_about_pricing(
        self, caplog,
    ):
        """A cell that spent NOTHING has no provenance to resolve.

        The timeout / harness-error / pre-invoke-cap paths never bind
        ``result``, so they reach the resolution with 0 tokens and $0.00.
        Resolving anyway would fire the LOUD unpriced-proxy WARNING for spend
        that never happened on EVERY timed-out proxied cell — training
        operators to ignore the warning that matters — and would label a $0.00
        figure 'unpriced_proxy' as though the degradation were real. The
        loud-over-silent norm is about REAL degradation (reviewer: log-noise).
        """
        with caplog.at_level(logging.WARNING, logger='orchestrator.evals.metrics'):
            result, mocks = await run_architect_eval_hermetic(
                self._cfg(model='qwen3-coder', env_overrides=dict(PROXY_ENV)),
                produced_plan=well_formed_plan(),
                invoke_side_effect=TimeoutError(),
                orch_prices={},
            )

        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['cost_usd'] == pytest.approx(0.0)
        assert persisted['input_tokens'] == 0
        assert persisted['output_tokens'] == 0
        assert persisted['cost_source'] == 'cli'
        assert result.metrics['cost_source'] == 'cli'
        assert not [
            r for r in caplog.records
            if 'No configured price for proxied-endpoint model' in r.message
        ], (
            'A $0.00 / 0-token cell must not emit the unpriced-proxy '
            f'degradation WARNING; got {[r.message for r in caplog.records]}'
        )


@pytest.mark.asyncio
class TestArchitectCellCostSourceIsComposed:
    """Task 3656 step-7: the cell's ``cost_usd`` is a TWO-COMPONENT sum
    (architect spend + plan-judge spend), so its single ``cost_source`` must
    not silently describe two differently-sourced components.

    Only the ARCHITECT component is resolved. The plan judge is always a
    native-cloud opus call — ``judge_plan_quality`` takes neither the
    candidate's model nor its ``env_overrides`` — so its CLI figure is
    trustworthy by construction, and re-resolving it against the candidate's
    price table would price opus tokens at a vLLM rate. The label therefore
    COMPOSES (``metrics.compose_cost_source``): agreement → that source,
    disagreement with judge dollars actually in the sum → ``'mixed'``.
    """

    def _cfg(self, **kw):
        from orchestrator.evals.configs import EvalConfig

        kw.setdefault('model', 'sonnet')
        return EvalConfig(
            'architect-candidate', 'claude', effort='high', role=RoleUnderTest.ARCHITECT, **kw,
        )

    def _spending_judge(self):
        return Scored(
            PlanQualityVerdict(plan_quality=0.77, per_criterion={}, reasoning='good'),
            cost_usd=0.42,
        )

    async def test_price_table_architect_plus_cli_judge_reads_mixed(self):
        """A price-table architect component beside a CLI judge component: the
        figure is the RESOLVED architect cost plus the judge's CLI cost, and the
        label says the two disagree rather than claiming either one."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(model='qwen3-coder', env_overrides=dict(PROXY_ENV)),
            produced_plan=well_formed_plan(),
            judge_return=self._spending_judge(),
            arch_result=token_agent_result(1_000_000, 500_000, cost_usd=99.0),
            orch_prices={'qwen3-coder': {'input_per_1m': 0.30, 'output_per_1m': 1.20}},
        )

        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['cost_usd'] == pytest.approx(0.90 + 0.42)
        assert persisted['judge_cost_usd'] == pytest.approx(0.42)
        assert persisted['cost_source'] == 'mixed'
        assert result.metrics['cost_source'] == 'mixed'

    async def test_todays_native_shape_with_a_spending_judge_stays_cli(self):
        """REGRESSION PIN: both components are CLI-sourced, so a spending judge
        must NOT tip today's architect cells into 'mixed'. Keeps every landed
        result and ``TestArchitectCellRecordsJudgeSpend`` byte-identical."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(env_overrides={}),
            produced_plan=well_formed_plan(),
            judge_return=self._spending_judge(),
            orch_prices={},
        )

        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['cost_source'] == 'cli'
        assert persisted['cost_usd'] == pytest.approx(1.23 + 0.42)
        assert result.metrics['cost_source'] == 'cli'

    async def test_judge_skipped_labels_the_architect_component_alone(self):
        """An unscorable plan SKIPS the judge (``create_plan_stub`` is the
        header-only artifact ``create_plan`` writes), so no judge dollars enter
        the sum and the label describes the architect component alone — NOT
        'mixed', which would announce a blend that never happened."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(model='qwen3-coder', env_overrides=dict(PROXY_ENV)),
            produced_plan=create_plan_stub(),
            judge_return=self._spending_judge(),
            arch_result=token_agent_result(1_000_000, 500_000, cost_usd=99.0),
            orch_prices={'qwen3-coder': {'input_per_1m': 0.30, 'output_per_1m': 1.20}},
        )

        mocks['judge'].assert_not_called()
        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['judge_cost_usd'] == pytest.approx(0.0)
        assert persisted['cost_source'] == 'price_table'
        assert persisted['cost_usd'] == pytest.approx(0.90)
        assert result.metrics['cost_source'] == 'price_table'


@pytest.mark.asyncio
class TestJudgeSkippedBranchesRecordNoJudgeSpend:
    """A judge that never ran leaves no judge spend behind on any branch."""

    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    # -- (a) NO FABRICATED SPEND on every judge-skipped branch ----------------
    # Each of these branches already asserts ``mocks['judge'].assert_not_called()``
    # elsewhere in this file; here the pin is on the COST fields specifically —
    # a judge that never ran must never leave a nonzero judge_cost_usd/
    # judge_invocations behind, and the cell's cost_usd must equal the
    # ARCHITECT's own spend for that branch, nothing more.

    async def test_cap_tainted_branch_records_architect_spend_only(self):
        """Cap landed right after ``create_plan``: header-only stub, tainted."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=stub_with_steps([]),
            arch_result=refused_agent_result(cost_usd=0.0),
        )
        mocks['judge'].assert_not_called()
        assert result.metrics['judge_cost_usd'] == 0.0
        assert result.metrics['judge_invocations'] == 0
        # refused_agent_result(cost_usd=0.0) itself spends $0.00 — no fabricated judge
        # dollars riding along with a refusal that burned nothing.
        assert result.metrics['cost_usd'] == pytest.approx(0.0)

    async def test_cap_refusal_with_a_plan_records_architect_spend_only(self):
        """Cap landed MID-run, after a real plan with steps had landed."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            arch_result=refused_agent_result(cost_usd=0.0),
        )
        mocks['judge'].assert_not_called()
        assert result.metrics['judge_cost_usd'] == 0.0
        assert result.metrics['judge_invocations'] == 0
        assert result.metrics['cost_usd'] == pytest.approx(0.0)

    async def test_healthy_stepless_plan_records_architect_spend_only(self):
        """Architect ran fine (no cap) but produced nothing worth judging."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan={},
            arch_success=True,
        )
        mocks['judge'].assert_not_called()
        assert result.metrics['judge_cost_usd'] == 0.0
        assert result.metrics['judge_invocations'] == 0
        # No arch_result override here — the harness's default architect
        # mock reports cost_usd=1.23, and that must be the WHOLE cell cost.
        assert result.metrics['cost_usd'] == pytest.approx(1.23)


# ---------------------------------------------------------------------------
# Task 3302: gate the LLM plan judge at the SOURCE.
#
# run_architect_eval's healthy branch called judge_plan_quality with no
# scorability gate, and judge_plan_quality had no guard of its own — it returned
# whatever the LLM said in [0, 1]. So a HEALTHY architect that produced a
# stepless artifact persisted the self-contradictory cell
# `cap_tainted=False, plan_steps=0, plan_quality=0.9`, which is exactly the
# two-scorer disagreement score_plan_structure's anti-fabrication short-circuit
# exists to prevent (Graphiti e2066ec6). Gating here keeps plan_steps and the
# persisted plan_quality consistent for every NEW cell.
#
# Task 3303 then closed the other half — the INSTRUMENT refuses such an artifact
# itself, pinned by
# test_eval_judge_plan_quality.py::TestJudgePlanQualityRefusesAnUnjudgeableArtifact — so
# the two blocks are one story: 3302 gated the CALL SITE, 3303 gated the
# INSTRUMENT, and both consult the one is_scorable_plan predicate. This class
# stays load-bearing regardless: it pins the runner-side consequences the
# instrument cannot reach from where it stands — the taint decision, the
# task_id × config.name log line, and that no opus call is made at all.
#
# _STEPLESS_PLANS is deliberately the narrower dict-only subset of
# test_eval_judge_plan_quality.py::_UNSCORABLE_PLAN_SHAPES: run_architect_eval does
# `plan = artifacts.read_plan() or {}`, so it can only ever pass a dict, whereas
# the instrument's own corpus must also cover None / [] / 'x'.
# ---------------------------------------------------------------------------

_STEPLESS_PLANS = [
    pytest.param({}, id='empty-dict'),
    pytest.param({'steps': []}, id='explicit-empty-steps'),
    pytest.param(
        {'task_id': 't', 'title': 'x', 'analysis': 'a', 'files': [], 'steps': []},
        id='header-only-create_plan-stub',
    ),
]


@pytest.mark.asyncio
class TestSteplessPlanIsNeverJudged:
    """A healthy architect that produced nothing scores the structural floor."""

    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    @staticmethod
    def _confident_judge():
        # What the ungated judge really does with an unjudgeable artifact.
        return Scored(
            PlanQualityVerdict(plan_quality=0.9, per_criterion={}, reasoning='looks fine'),
            cost_usd=0.0,
        )

    @pytest.mark.parametrize('plan', _STEPLESS_PLANS)
    async def test_stepless_plan_scores_the_structural_floor(self, plan):
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=plan,
            judge_return=self._confident_judge(),
            arch_success=True,
        )
        persisted = mocks['save'].call_args.args[0].metrics

        # The deterministic floor, NOT the judge's 0.9.
        assert persisted['plan_quality'] == 0.0
        assert persisted['plan_steps'] == 0
        assert result.metrics['plan_quality'] == 0.0

    @pytest.mark.parametrize('plan', _STEPLESS_PLANS)
    async def test_stepless_plan_is_a_content_failure_not_an_exclusion(self, plan):
        """No infra failure occurred: the architect ran fine and answered with
        nothing. That is worth 0.0, never the cap-tainted exclusion a transport
        refusal earns (task 3118)."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=plan,
            judge_return=self._confident_judge(),
            arch_success=True,
        )
        persisted = mocks['save'].call_args.args[0].metrics

        assert persisted['cap_tainted'] is False
        assert persisted['invocation_errors'] is None
        assert result.metrics['cap_tainted'] is False

    @pytest.mark.parametrize('plan', _STEPLESS_PLANS)
    async def test_the_plan_judge_is_never_invoked(self, plan):
        """Nothing to judge — and inside a cap window the call would 429 anyway,
        so an opus invocation on an unjudgeable artifact is pure waste (the same
        justification the arch_unmeasurable branch already records)."""
        _, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=plan,
            judge_return=self._confident_judge(),
            arch_success=True,
        )
        mocks['judge'].assert_not_awaited()

    async def test_a_real_plan_still_awaits_the_judge(self):
        """The control: the gate fires ONLY on a stepless artifact."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(),
        )
        persisted = mocks['save'].call_args.args[0].metrics

        mocks['judge'].assert_awaited_once()
        assert persisted['plan_quality'] == 0.77
        assert persisted['plan_steps'] > 0
        assert persisted['cap_tainted'] is False

    @pytest.mark.parametrize('plan', _STEPLESS_PLANS + [
        pytest.param(None, id='no-plan-artifact'),
        pytest.param(well_formed_plan(), id='real-plan'),
    ])
    async def test_persisted_metrics_agree_with_the_artifact_level_twin(self, plan):
        """produced_a_plan(PERSISTED metrics) == is_scorable_plan(artifact).

        The equivalence the report layer depends on: it only ever sees a
        persisted metrics dict and cannot call is_scorable_plan, which reads the
        plan ARTIFACT. Driven through the REAL runner (reviewer: test-quality) —
        asserting against a metrics dict the test built itself would only pin
        produced_a_plan against a copy of run_architect_eval's derivation living
        in this file, and would stay green while the two surfaces diverged.
        """
        from orchestrator.evals.plan_predicates import (
            is_scorable_plan,
            produced_a_plan,
        )

        _, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=plan,
            judge_return=self._confident_judge(),
            arch_success=True,
        )
        persisted = mocks['save'].call_args.args[0].metrics

        assert produced_a_plan(persisted) is is_scorable_plan(plan)


# ---------------------------------------------------------------------------
# judged_without_reference — the validity marker (eval-revival σ, task 3628)
#
# True on EXACTLY the cells whose persisted plan_quality is the LLM judge's own
# number produced from an EMPTY reference_diff. The three negative twins below
# define the predicate as much as the positive does: a cap-tainted cell has no
# plan_quality to bound, and both floor paths score via score_plan_structure,
# which never consults a reference. Keying on `not reference_diff` at the
# materialization site alone would fire on ~every no-plan cell in a hard
# campaign — precisely the population the consumer must see PAST — so the
# negatives are what lock the semantics against that over-broad implementation.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestJudgedWithoutReference:
    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    async def test_judged_blind_cell_is_marked_and_still_scored(self):
        """POSITIVE: the judge ran without ground truth — mark it, keep it.

        The cell is NOT excluded: plan_quality is still the judge's float and
        the cell stays in every pool at that score. The marker bounds VALIDITY,
        not membership.
        """
        _, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            task_override=arch_task_no_reference(),
            reference_diff='',
        )
        # Assert on the cell handed to the store — that cell is what both the
        # report layer and the downstream campaign driver actually read.
        persisted = mocks['save'].call_args.args[0].metrics

        assert persisted['judged_without_reference'] is True
        assert persisted['plan_quality'] == 0.77
        assert persisted['cap_tainted'] is False

    async def test_healthy_cell_carries_the_key_as_false(self):
        """NEGATIVE (healthy) — and the key must be PRESENT, not omitted.

        The presence assertion is the cross-consumer contract:
        ``scripts/run_fable_trial_v2_campaign.py`` (task 3632) reads per-cell
        reference validity by KEY PRESENCE, so a healthy cell must carry the
        key with False rather than omit it. Omitting it would make every
        healthy cell read as "never measured" downstream — forever, and
        silently. This is the assertion that catches a conditional-emit
        implementation, which every other case in this class would pass.
        """
        _, mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(),
        )
        persisted = mocks['save'].call_args.args[0].metrics

        assert 'judged_without_reference' in persisted
        assert persisted['judged_without_reference'] is False

    async def test_cap_tainted_cell_is_not_marked(self):
        """NEGATIVE (cap-tainted): nothing was scored, so nothing to bound.

        The judge is skipped entirely, plan_quality is None, and the cell is
        already counted — disjointly — by cap_excluded. Marking it too would
        double-count one cell across two counts that must stay disjoint.
        """
        _, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan={},
            arch_result=refused_agent_result(cost_usd=0.0),
            task_override=arch_task_no_reference(),
            reference_diff='',
        )
        persisted = mocks['save'].call_args.args[0].metrics

        assert persisted['plan_quality'] is None
        assert persisted['cap_tainted'] is True
        assert persisted['judged_without_reference'] is False

    @pytest.mark.parametrize('plan', _STEPLESS_PLANS)
    async def test_structural_floor_cell_is_not_marked(self, plan):
        """NEGATIVE (structural floor): the floor never consults a reference.

        ``score_plan_structure`` derives its number from the plan alone, so the
        score is valid ground-truth-independently — there is no validity to
        bound. The judge-not-called assertion ties the marker to the judge path
        rather than to the emptiness of reference_diff.

        Parametrized over ``_STEPLESS_PLANS``, not the wider
        ``test_eval_judge_plan_quality.py::_UNSCORABLE_PLAN_SHAPES``: this is a
        RUNNER-level test, and ``run_architect_eval`` does
        ``plan = artifacts.read_plan() or {}``, so it can only ever be handed a
        dict (see the note above
        ``TestSteplessPlanIsNeverJudged``, which scopes itself the same way).
        """
        _, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=plan,
            task_override=arch_task_no_reference(),
            reference_diff='',
        )
        persisted = mocks['save'].call_args.args[0].metrics

        assert persisted['judged_without_reference'] is False
        mocks['judge'].assert_not_called()

    async def test_judge_degraded_to_floor_is_not_marked(self):
        """NEGATIVE (judge failed → floor): the PERSISTED number is the floor.

        The judge WAS asked without a reference, but its answer was discarded,
        so the score that landed is again reference-independent. The marker
        asserts something about the persisted score, not about what was asked.
        """
        from orchestrator.evals.judge import score_plan_structure

        plan = well_formed_plan()
        _, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=plan,
            judge_side_effect=RuntimeError('judge exploded'),
            task_override=arch_task_no_reference(),
            reference_diff='',
        )
        persisted = mocks['save'].call_args.args[0].metrics

        assert persisted['plan_quality'] == score_plan_structure(plan)
        assert persisted['judged_without_reference'] is False

    async def test_missing_reference_is_logged_loudly(self, caplog):
        """The INV-2 log half beside the structured half.

        A fixture with no reference block must be loud at RUN time — the whole
        point of σ is that the v1 degradation was findable only by archaeology.
        """
        with caplog.at_level(logging.WARNING):
            await run_architect_eval_hermetic(
                self._cfg(),
                produced_plan=well_formed_plan(),
                task_override=arch_task_no_reference(),
                reference_diff='',
            )

        warnings = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any(
            'df_task_2605' in m and 'reference' in m.lower() for m in warnings
        ), f'no warning naming the task and the missing reference: {warnings}'

    async def test_missing_reference_warning_states_only_a_fixture_fact(
        self, caplog,
    ):
        """The step-6 line must not claim anything about a judge score.

        It fires BEFORE the architect's outcome is known, so on a cap-tainted
        cell — which skips the judge entirely — a claim that "any plan judge
        score will be plausibility-based" is simply false (reviewer:
        robustness). The claim belongs to the step-7 warning, which fires
        exactly where the judge is invoked blind, and which must not appear on
        this path at all.
        """
        with caplog.at_level(logging.WARNING):
            await run_architect_eval_hermetic(
                self._cfg(),
                produced_plan={},
                arch_result=refused_agent_result(cost_usd=0.0),
                task_override=arch_task_no_reference(),
                reference_diff='',
            )

        warnings = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        fixture_lines = [m for m in warnings if 'reference.post_task_commit' in m]
        assert len(fixture_lines) == 1, warnings
        # The judge never ran here, so NOTHING may assert how its score was
        # produced — neither the removed 'PLAUSIBILITY' overclaim nor the
        # step-7 line.
        assert 'plausibility' not in fixture_lines[0].lower()
        assert not any('EMPTY reference diff' in m for m in warnings), warnings

    async def test_blind_judge_claim_is_made_once_by_step_7(self, caplog):
        """On the ONE path where the claim is true, it is stated ONCE.

        The pair of warnings must not double-report: the fixture-level line
        names the authoring defect, the step-7 line names the consequence for
        the score, and neither repeats the other.
        """
        with caplog.at_level(logging.WARNING):
            await run_architect_eval_hermetic(
                self._cfg(),
                produced_plan=well_formed_plan(),
                task_override=arch_task_no_reference(),
                reference_diff='',
            )

        warnings = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        fixture_lines = [m for m in warnings if 'reference.post_task_commit' in m]
        judge_lines = [m for m in warnings if 'EMPTY reference diff' in m]

        assert len(fixture_lines) == 1, warnings
        assert len(judge_lines) == 1, warnings
        assert 'EMPTY reference diff' not in fixture_lines[0]




def _decline_artifact(reported_at: str = '2026-07-19T12:00:00+00:00') -> dict:
    """A decline artifact in the shape every ``artifacts.write_*`` persists."""
    return {'evidence': 'because', 'reason': 'because', 'reported_at': reported_at}


@pytest.mark.asyncio
class TestArchitectCellPersistsTerminalKind:
    """Task 4760: the cell now says WHY ``plan_steps == 0``, without forensics.

    Before this, an architect that took an explicit plan-tools decline exit
    persisted a cell byte-indistinguishable from one that silently failed to
    plan. The decline artifact was written and then rmtree'd by
    ``cleanup_eval_worktree`` before anything read it, so tranche-1's 47
    no-plan cells could be split into causes only from
    ``~/.claude/projects/*run-<id>/`` transcripts.
    """

    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    async def test_ordinary_plan_cell_persists_planned(self):
        result, _ = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(),
        )

        assert result.metrics['terminal_kind'] == 'planned'

    @pytest.mark.parametrize('kind', [
        'false_premise', 'already_done', 'blocking_dependency',
        'unactionable', 'ready_to_merge',
    ])
    async def test_decline_cell_persists_its_kind_and_stays_scored(self, kind):
        """THE tranche-1 shape, now legible: no plan, an explicit decline.

        ``plan_steps`` and ``outcome`` are UNCHANGED — a decline is a correct
        refusal, not a content failure, so nothing about the existing cell moves.
        """
        result, _ = await run_architect_eval_hermetic(
            self._cfg(), produced_plan={},
            declines={kind: _decline_artifact()},
        )

        assert result.metrics['terminal_kind'] == kind
        assert result.metrics['plan_steps'] == 0
        assert result.outcome == 'done'

    async def test_silent_no_plan_cell_persists_none(self):
        """The genuine "could not plan" outcome stays DISTINGUISHABLE from a
        decline — which is the entire point of the field.
        """
        result, _ = await run_architect_eval_hermetic(
            self._cfg(), produced_plan={},
        )

        assert result.metrics['terminal_kind'] == 'none'
        assert result.metrics['plan_steps'] == 0

    async def test_plan_then_later_decline_persists_both_facts(self):
        """The reify_task_4026 / run e522b1b0 ordering cell.

        BOTH facts survive, so a downstream reader can bucket it either way and
        neither reading is destroyed — planRate keeps its historical derivation.
        """
        plan = well_formed_plan()
        plan['steps'] = [
            *plan['steps'],
            {'id': 'step-5', 'type': 'test', 'description': 'RED test for Z'},
            {'id': 'step-6', 'type': 'impl', 'description': 'GREEN implement Z'},
        ]
        plan['_finalized_at'] = '2026-07-19T10:00:00+00:00'

        result, _ = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=plan,
            declines={'already_done': _decline_artifact('2026-07-19T11:00:00+00:00')},
        )

        assert result.metrics['plan_steps'] == 6
        assert result.metrics['terminal_kind'] == 'already_done'

    async def test_declines_are_read_before_the_worktree_is_cleaned_up(self):
        """THE ordering ``cleanup_eval_worktree``'s rmtree would destroy.

        ``snapshots.cleanup_eval_worktree`` rmtree's the relocated
        ``.task-meta/<name>/`` root, so a read placed ANYWHERE after it — which
        is everywhere post-``finally`` — would find nothing. Sampling the read
        count at the moment cleanup is awaited is the only way to prove the
        artifact was read while it still existed.
        """
        at_cleanup: dict[str, int] = {}
        doubles: dict = {}

        async def _sample_at_cleanup(*_args, **_kwargs):
            at_cleanup['reads'] = doubles[
                'artifacts'].read_false_premise.call_count

        result, mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan={},
            declines={'false_premise': _decline_artifact()},
            cleanup_side_effect=_sample_at_cleanup,
            expose_mocks=doubles,
        )

        assert mocks['cleanup'].await_count == 1
        assert at_cleanup['reads'] >= 1, (
            "the decline artifacts were still unread when "
            "cleanup_eval_worktree rmtree'd the meta root"
        )
        assert result.metrics['terminal_kind'] == 'false_premise'

    async def test_timeout_cell_still_persists_the_decline_kind(self):
        """ONE read site covers all four exits — including the timeout, whose
        plan is never read at all.
        """
        result, _ = await run_architect_eval_hermetic(
            self._cfg(), produced_plan={},
            invoke_side_effect=TimeoutError,
            declines={'blocking_dependency': _decline_artifact()},
        )

        assert result.outcome == 'timeout'
        assert result.metrics['terminal_kind'] == 'blocking_dependency'

    async def test_harness_error_cell_persists_none_and_does_not_raise(self):
        """``artifacts`` is never constructed, so there is nothing to read —
        and the ``finally``-block read must not turn that into a SECOND crash.
        """
        result, _ = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(),
            orch_config_side_effect=RuntimeError('eval orch config exploded'),
        )

        assert result.outcome == 'blocked'
        assert result.metrics['terminal_kind'] == 'none'

    async def test_raising_decline_reader_leaves_the_cell_scored(self):
        """A corrupt artifact costs THAT KIND's visibility, never the cell: the
        terminal kind still resolves from the plan alone.
        """
        result, _ = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(),
            decline_reader_errors={'already_done': OSError('corrupt json')},
        )

        assert result.outcome == 'done'
        assert isinstance(result.metrics['plan_quality'], float)
        assert result.metrics['terminal_kind'] == 'planned'


# ---------------------------------------------------------------------------
# Characterization of the persisted scoring fields, one row per exit (task 6737)
#
# Recorded against run_architect_eval BEFORE it was split into invoke →
# classify → assemble. Only each row's harness call may change; the expected
# literals are the evidence that the split moved no persisted field.
# ---------------------------------------------------------------------------

_SCORING_FIELDS = (
    'plan_quality', 'cap_tainted', 'invocation_errors', 'plan_steps',
    'judge_invocations', 'judge_cost_usd', 'judged_without_reference',
    'cost_usd', 'cost_source', 'terminal_kind', 'tests_pass',
    'role_under_test', 'replay_frame', 'input_tokens', 'output_tokens',
    'cache_read_tokens', 'cache_create_tokens', 'turns_used',
    'tokens_per_second', 'workflow_duration_ms', 'is_local_model',
)


def _characterized_cfg(**over):
    from orchestrator.evals.configs import EvalConfig

    return EvalConfig(
        over.pop('name', 'architect-sonnet-high'), 'claude',
        over.pop('model', 'sonnet'), 'high', role=RoleUnderTest.ARCHITECT, **over,
    )


# The default hermetic invoke's usage (see run_architect_eval_hermetic) and
# the all-zero usage of an attempt that never returned.
_DEFAULT_INVOKE_USAGE = {
    'input_tokens': 12000, 'output_tokens': 3000,
    'cache_read_tokens': 48000, 'cache_create_tokens': 6000,
    'turns_used': 9, 'tokens_per_second': 656.89,
    'workflow_duration_ms': 4567, 'is_local_model': False,
}
_NO_USAGE = {
    'input_tokens': 0, 'output_tokens': 0,
    'cache_read_tokens': 0, 'cache_create_tokens': 0,
    'turns_used': 0, 'tokens_per_second': 0.0,
    'workflow_duration_ms': 0, 'is_local_model': False,
}
_ARCHITECT_CELL_STAMPS = {
    'tests_pass': None, 'role_under_test': 'architect',
    'replay_frame': 'honest-frame-v1',
}
_JUDGED_077 = {
    'plan_quality': 0.77, 'cap_tainted': False, 'invocation_errors': None,
    'plan_steps': 4, 'judge_invocations': 1, 'judge_cost_usd': 0.0,
    'terminal_kind': 'planned',
}
_FLOORED_STEPLESS = {
    'outcome': 'done', 'plan_quality': 0.0, 'cap_tainted': False,
    'invocation_errors': None, 'plan_steps': 0, 'judge_invocations': 0,
    'judge_cost_usd': 0.0, 'judged_without_reference': False,
    'cost_usd': 1.23, 'cost_source': 'cli',
}
_CAP_HIT_RECORD = {
    'stage': 'architect', 'cause': 'cap_hit',
    'reason': "You've hit your session limit · resets 8pm",
}
_CAP_EXHAUSTED_RECORD = {
    'stage': 'architect', 'cause': 'cap_exhausted',
    'reason': 'x: all accounts capped after 3 retries (10.0s elapsed)',
}
_UNSPENT_UNJUDGED = {
    'judge_invocations': 0, 'judge_cost_usd': 0.0,
    'judged_without_reference': False, 'cost_usd': 0.0, 'cost_source': 'cli',
}


_PERSISTED_FIELD_ROWS = [
    pytest.param(
        lambda: {'produced_plan': well_formed_plan()},
        {
            **_ARCHITECT_CELL_STAMPS, **_DEFAULT_INVOKE_USAGE, **_JUDGED_077,
            'outcome': 'done', 'judged_without_reference': False,
            'cost_usd': 1.23, 'cost_source': 'cli',
        },
        id='healthy-scorable-plan',
    ),
    pytest.param(
        lambda: {'produced_plan': {}},
        {
            **_ARCHITECT_CELL_STAMPS, **_DEFAULT_INVOKE_USAGE, **_FLOORED_STEPLESS,
            'terminal_kind': 'none',
        },
        id='healthy-stepless-plan',
    ),
    pytest.param(
        lambda: {'produced_plan': create_plan_stub()},
        {
            **_ARCHITECT_CELL_STAMPS, **_DEFAULT_INVOKE_USAGE, **_FLOORED_STEPLESS,
            'terminal_kind': 'none',
        },
        id='healthy-create-plan-stub',
    ),
    pytest.param(
        lambda: {'produced_plan': {}, 'arch_result': refused_agent_result(cost_usd=0.0)},
        {
            **_ARCHITECT_CELL_STAMPS, **_NO_USAGE, **_UNSPENT_UNJUDGED,
            'outcome': 'blocked', 'plan_quality': None, 'cap_tainted': True,
            'invocation_errors': [_CAP_HIT_RECORD], 'plan_steps': 0,
            'terminal_kind': 'none', 'workflow_duration_ms': 1200,
        },
        id='cap-refusal-no-plan',
    ),
    pytest.param(
        lambda: {'produced_plan': well_formed_plan(), 'arch_result': refused_agent_result(cost_usd=0.0)},
        {
            **_ARCHITECT_CELL_STAMPS, **_NO_USAGE, **_UNSPENT_UNJUDGED,
            'outcome': 'blocked', 'plan_quality': 1.0, 'cap_tainted': False,
            'invocation_errors': [_CAP_HIT_RECORD], 'plan_steps': 4,
            'terminal_kind': 'planned', 'workflow_duration_ms': 1200,
        },
        id='cap-refusal-scorable-plan',
    ),
    pytest.param(
        lambda: {'produced_plan': well_formed_plan(), 'invoke_side_effect': TimeoutError()},
        {
            **_ARCHITECT_CELL_STAMPS, **_NO_USAGE, **_UNSPENT_UNJUDGED,
            'outcome': 'timeout', 'plan_quality': 0.0, 'cap_tainted': False,
            'invocation_errors': [{
                'stage': 'architect', 'cause': 'timeout',
                'reason': 'no answer within 60m',
            }],
            'plan_steps': 0, 'terminal_kind': 'none',
        },
        id='timeout',
    ),
    pytest.param(
        lambda: {
            'produced_plan': well_formed_plan(),
            'invoke_side_effect': RuntimeError('boom'),
        },
        {
            **_ARCHITECT_CELL_STAMPS, **_NO_USAGE, **_UNSPENT_UNJUDGED,
            'outcome': 'blocked', 'plan_quality': None, 'cap_tainted': True,
            'invocation_errors': [{
                'stage': 'architect', 'cause': 'harness_error',
                'reason': 'RuntimeError: boom',
            }],
            'plan_steps': 0, 'terminal_kind': 'none',
        },
        id='harness-error-at-invoke',
    ),
    pytest.param(
        lambda: {
            'produced_plan': well_formed_plan(),
            'orch_config_side_effect': RuntimeError('x'),
        },
        {
            **_ARCHITECT_CELL_STAMPS, **_NO_USAGE, **_UNSPENT_UNJUDGED,
            'outcome': 'blocked', 'plan_quality': None, 'cap_tainted': True,
            'invocation_errors': [{
                'stage': 'architect', 'cause': 'harness_error',
                'reason': 'RuntimeError: x',
            }],
            'plan_steps': 0, 'terminal_kind': 'none',
        },
        id='harness-error-at-orch-config',
    ),
    pytest.param(
        lambda: {
            'produced_plan': well_formed_plan(),
            'invoke_side_effect': AllAccountsCappedException(3, 10.0, 'x'),
        },
        {
            **_ARCHITECT_CELL_STAMPS, **_NO_USAGE, **_UNSPENT_UNJUDGED,
            'outcome': 'blocked', 'plan_quality': 1.0, 'cap_tainted': False,
            'invocation_errors': [_CAP_EXHAUSTED_RECORD], 'plan_steps': 4,
            'terminal_kind': 'planned',
        },
        id='cap-exhausted-scorable-plan',
    ),
    pytest.param(
        lambda: {
            'produced_plan': {},
            'invoke_side_effect': AllAccountsCappedException(3, 10.0, 'x'),
        },
        {
            **_ARCHITECT_CELL_STAMPS, **_NO_USAGE, **_UNSPENT_UNJUDGED,
            'outcome': 'blocked', 'plan_quality': None, 'cap_tainted': True,
            'invocation_errors': [_CAP_EXHAUSTED_RECORD], 'plan_steps': 0,
            'terminal_kind': 'none',
        },
        id='cap-exhausted-no-plan',
    ),
    pytest.param(
        lambda: {
            'produced_plan': well_formed_plan(),
            'judge_return': Unmeasured(
                MeasurementStage.JUDGE, UnmeasuredCause.UNPARSEABLE,
                'JSONDecodeError: Expecting value', cost_usd=0.6,
            ),
        },
        {
            **_ARCHITECT_CELL_STAMPS, **_DEFAULT_INVOKE_USAGE,
            'outcome': 'done', 'plan_quality': 1.0, 'cap_tainted': False,
            'invocation_errors': [{
                'stage': 'judge', 'cause': 'unparseable',
                'reason': 'JSONDecodeError: Expecting value',
            }],
            'plan_steps': 4, 'judge_invocations': 1, 'judge_cost_usd': 0.6,
            'judged_without_reference': False, 'cost_usd': 1.83,
            'cost_source': 'cli', 'terminal_kind': 'planned',
        },
        id='judge-unparseable',
    ),
    pytest.param(
        lambda: {
            'produced_plan': well_formed_plan(),
            'judge_side_effect': RuntimeError('judge exploded'),
        },
        {
            **_ARCHITECT_CELL_STAMPS, **_DEFAULT_INVOKE_USAGE,
            'outcome': 'done', 'plan_quality': 1.0, 'cap_tainted': False,
            'invocation_errors': [{
                'stage': 'judge', 'cause': 'raised',
                'reason': 'RuntimeError: judge exploded',
            }],
            'plan_steps': 4, 'judge_invocations': 0, 'judge_cost_usd': 0.0,
            'judged_without_reference': False, 'cost_usd': 1.23,
            'cost_source': 'cli', 'terminal_kind': 'planned',
        },
        id='judge-raised',
    ),
    pytest.param(
        lambda: {
            'produced_plan': well_formed_plan(),
            'task_override': arch_task_no_reference(), 'reference_diff': '',
        },
        {
            **_ARCHITECT_CELL_STAMPS, **_DEFAULT_INVOKE_USAGE, **_JUDGED_077,
            'outcome': 'done', 'judged_without_reference': True,
            'cost_usd': 1.23, 'cost_source': 'cli',
        },
        id='judge-blind-no-reference',
    ),
    pytest.param(
        lambda: {
            'cfg': _characterized_cfg(
                name='architect-candidate', model='qwen3-coder',
                env_overrides=dict(PROXY_ENV),
            ),
            'produced_plan': well_formed_plan(),
            'arch_result': token_agent_result(1_000_000, 500_000, cost_usd=99.0),
            'orch_prices': {
                'qwen3-coder': {'input_per_1m': 0.30, 'output_per_1m': 1.20},
            },
        },
        {
            **_ARCHITECT_CELL_STAMPS, **_JUDGED_077,
            'outcome': 'done', 'judged_without_reference': False,
            'cost_usd': 0.9, 'cost_source': 'price_table',
            'input_tokens': 1000000, 'output_tokens': 500000,
            'cache_read_tokens': 0, 'cache_create_tokens': 0, 'turns_used': 3,
            'tokens_per_second': 109481.06, 'workflow_duration_ms': 4567,
            'is_local_model': True,
        },
        id='proxied-candidate-priced',
    ),
    pytest.param(
        lambda: {
            'produced_plan': {},
            'declines': {'false_premise': _decline_artifact()},
        },
        {
            **_ARCHITECT_CELL_STAMPS, **_DEFAULT_INVOKE_USAGE, **_FLOORED_STEPLESS,
            'terminal_kind': 'false_premise',
        },
        id='false-premise-decline',
    ),
]


@pytest.mark.asyncio
class TestPersistedScoringFieldsAreStable:
    @pytest.mark.parametrize(('scenario', 'expected'), _PERSISTED_FIELD_ROWS)
    async def test_persisted_scoring_fields(self, scenario, expected, tmp_path):
        harness_kwargs = scenario()
        cfg = harness_kwargs.pop('cfg', None) or _characterized_cfg()
        store = ResultStore(tmp_path)

        await run_architect_eval_hermetic(cfg, store=store, **harness_kwargs)

        (cell,) = store.load_all()
        observed = {'outcome': cell.outcome} | {
            field: cell.metrics[field] for field in _SCORING_FIELDS
        }
        assert observed == pytest.approx(expected)
