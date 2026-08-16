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
import logging
import types
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


def _create_plan_stub() -> dict:
    """The header-only plan ``_create_plan`` writes on the architect's FIRST call.

    Verbatim shape from ``orchestrator/src/orchestrator/mcp/plan_tools.py``
    (``_create_plan``, lines 146-156): a TRUTHY dict carrying ZERO steps. This is
    the artifact a session cap leaves behind when it lands mid-run right after
    ``create_plan`` — the most common mid-run cap shape, since ``create_plan`` is
    the only plan-tools call the architect can reach before the 429.
    """
    return {
        'task_id': 't',
        'title': 'x',
        'analysis': 'a',
        'files': [],
        'prerequisites': [],
        'steps': [],
        'design_decisions': [],
        'reuse': [],
    }


def _stub_with_steps(steps) -> dict:
    """``_create_plan_stub`` with its ``steps`` replaced (``[]`` vs ``None``)."""
    stub = _create_plan_stub()
    stub['steps'] = steps
    return stub


# Every artifact shape that carries NO model content to derive a score from —
# absent, empty, not-a-dict, or the header-only stub in either empty spelling.
# Shared by the is_scorable_plan tests and the anti-drift parity assertion, so
# the two can never be extended out of step with each other.
_UNSCORABLE_PLAN_SHAPES = [
    {},
    None,
    [],
    'x',
    {'steps': []},
    {'steps': None},
    _create_plan_stub(),
    _stub_with_steps(None),
]


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
        ``report._judged_blind_count`` is None for any config with even one
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


def _flushed_server_error_result(status: int = 529):
    """The 2026-07-29 incident shape: the watchdog SIGTERM-killed a wedged CLI
    and the CLI flushed its result JSON on the way out with api_error_status
    set, so ``timed_out`` and a 5xx status are BOTH true. Mirrors
    shared/tests/test_server_error.py:61-77 (``_incident_result``), which is the
    source of truth for what a flushed 5xx looks like.
    """
    from shared.cli_invoke import AgentResult

    return AgentResult(
        success=False, output='', subtype='error_empty_output',
        timed_out=True, transcript_turns=0, duration_ms=127_000,
        api_error_status=status,
    )


def _cli_local_error_result(marker: str):
    """A local CLI/usage fault — explicitly NOT a cap (the reify-3604 fix)."""
    from shared.cli_invoke import AgentResult

    return AgentResult(
        success=False,
        output=f'error: {marker}',
        cost_usd=0.0,
        duration_ms=300,
        turns=0,
        subtype='error',
    )


def _server_error_result(status: int = 529):
    """A fast, zero-cost server-side (5xx) refusal — the provider was degraded,
    the model never answered. Deliberately NOT timed out and carrying no cap
    body, so the ONLY route to a marker is classify_invocation's ServerError
    tier (shared/src/shared/invocation_outcome.py:523).
    """
    from shared.cli_invoke import AgentResult

    return AgentResult(
        success=False, output='', cost_usd=0.0,
        duration_ms=1100, turns=0, subtype='error', api_error_status=status,
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
        from shared.cli_invoke import AgentResult

        from orchestrator.evals.metrics import detect_invocation_error

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
        # A bodied 429 is a cap hit, never the new 5xx bucket — regression pin
        # for the server_error tier, folded in here rather than as a parallel
        # test (reviewer: duplication).
        assert marker.partition(':')[0] == 'cap_hit'
        assert 'server_error' not in marker

    def test_body_less_429_yields_marker_via_structured_fallback(self):
        # 429 is deliberately EXCLUDED from AuthFailed (it routes to the cap-text
        # tier), so a 429 with no cap body classifies as a wedge/unclassified
        # failure — the structured api_error_status fallback must still catch it.
        from shared.cli_invoke import AgentResult

        from orchestrator.evals.metrics import detect_invocation_error

        bare = AgentResult(
            success=False, output='', stderr='', cost_usd=0.0,
            duration_ms=800, turns=0, api_error_status=429,
        )
        marker = detect_invocation_error(bare)
        assert isinstance(marker, str) and marker
        assert '429' in marker
        # 429 sits below the 5xx band, so a body-less 429 must keep reaching
        # this raw-status fallback rather than drifting into the server_error
        # bucket a bodied 5xx gets — folded in here rather than as a parallel
        # test (reviewer: duplication).
        assert marker.partition(':')[0] == 'api_error'
        assert 'server_error' not in marker

    def test_auth_failure_yields_auth_marker(self):
        from shared.cli_invoke import AgentResult

        from orchestrator.evals.metrics import detect_invocation_error

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
        from shared.cli_invoke import AgentResult

        from orchestrator.evals.metrics import detect_invocation_error

        content_failure = AgentResult(
            success=False, output='I could not complete the plan',
            cost_usd=1.2, duration_ms=45000, turns=8, api_error_status=None,
        )
        assert detect_invocation_error(content_failure) is None

    def test_zero_output_wedge_yields_wedge_marker(self):
        # A full-timeout invocation with zero transcript turns produced NO model
        # answer at all — unmeasurable for exactly the reason a 429 is, so it
        # must be marked rather than scored 0.0 (reviewer: robustness).
        from orchestrator.evals.metrics import detect_invocation_error

        marker = detect_invocation_error(_wedged_agent_result())
        assert isinstance(marker, str) and marker
        assert 'wedge' in marker.lower()

    def test_body_less_429_outranks_the_wedge_tier(self):
        # A 429 that ALSO timed out with zero turns must report the 429 it is,
        # not a generic wedge — the structured status is the more specific fact.
        from shared.cli_invoke import AgentResult

        from orchestrator.evals.metrics import detect_invocation_error

        wedged_429 = AgentResult(
            success=False, output='', cost_usd=0.0, duration_ms=600_000,
            turns=0, timed_out=True, transcript_turns=0, api_error_status=429,
        )
        marker = detect_invocation_error(wedged_429)
        assert marker is not None and '429' in marker
        assert 'wedge' not in marker.lower()

    def test_near_cap_warning_is_not_an_invocation_error(self):
        # NearCap is a WARNING that a cap is imminent — this invocation was NOT
        # refused, so whatever it produced is a real measurement. Pinned so the
        # deliberate None can't silently become a marker (and an exclusion).
        from shared.cli_invoke import AgentResult

        from orchestrator.evals.metrics import detect_invocation_error

        near = AgentResult(
            success=False,
            output="You're close to your usage limit for this session",
            cost_usd=0.9, duration_ms=30_000, turns=6,
        )
        assert detect_invocation_error(near) is None

    def test_cli_local_error_is_not_an_invocation_error(self):
        # CliLocalError is explicitly NOT a cap (the reify-3604 precedence fix).
        # It is a local harness/CLI fault whose classification the eval layer
        # deliberately does not launder into a cap-shaped exclusion.
        from shared.invocation_outcome import (
            NON_CAP_CLI_ERROR_MARKERS,
            CliLocalError,
            classify_invocation,
        )

        from orchestrator.evals.metrics import detect_invocation_error

        local = _cli_local_error_result(next(iter(NON_CAP_CLI_ERROR_MARKERS)))
        # Guard the fixture: it really does reach the CliLocalError tier, so
        # this stays a pin on the VARIANT rather than on an unclassified blob.
        assert isinstance(
            classify_invocation(local, strict_confirm=True), CliLocalError
        )
        assert detect_invocation_error(local) is None

    # -- backend threading -------------------------------------------------
    #
    # classify_invocation's codex/gemini cap tables fire ONLY for their own
    # backend, so the `backend=` argument is load-bearing: dropping or
    # hard-coding it would silently un-mark every codex/gemini cap hit while
    # every claude test still passed.

    def test_codex_cap_pattern_fires_only_under_the_codex_backend(self):
        from orchestrator.evals.metrics import detect_invocation_error

        codex_capped = _codex_cap_agent_result()

        marker = detect_invocation_error(codex_capped, backend='codex')
        assert isinstance(marker, str) and marker
        assert 'cap' in marker.lower()

        # Same payload, DEFAULT (claude) backend: the codex table is not
        # consulted and no claude cap prefix matches, so it is not an infra
        # refusal at all.
        assert detect_invocation_error(codex_capped) is None

    # -- 5xx (ServerError) attribution --------------------------------------

    def test_server_error_5xx_yields_server_error_marker(self):
        from orchestrator.evals.metrics import detect_invocation_error

        marker = detect_invocation_error(_server_error_result(529))
        assert isinstance(marker, str)
        # report._taint_cause (report.py:1194-1216) reduces the stage-prefixed
        # marker to this first colon-delimited token as the report's by-cause
        # exclusion key (accumulated at report.py:1313) — the prefix is a
        # behavioural contract, not cosmetics, hence the exact-token assert
        # rather than a loose `in`.
        assert marker.partition(':')[0] == 'server_error'
        assert '529' in marker

    @pytest.mark.parametrize('status', [500, 502, 503, 529, 599])
    def test_every_5xx_status_in_the_band_is_attributed(self, status: int):
        # Mirrors the band shared/tests/test_server_error.py::TestIsServerErrorStatus
        # pins, so the eval layer cannot drift into a 529-only special case.
        from orchestrator.evals.metrics import detect_invocation_error

        marker = detect_invocation_error(_server_error_result(status))
        assert isinstance(marker, str)
        assert marker.partition(':')[0] == 'server_error'
        assert str(status) in marker

    # -- timed-out-but-actually-5xx: the shape 3314 regressed to None -------

    def test_timed_out_flushed_5xx_is_attributed_as_a_server_error_not_dropped(self):
        # The headline regression case: a watchdog-SIGTERM-flushed result
        # carrying BOTH timed_out=True and a 5xx api_error_status. Before
        # 3314's ServerError tier existed this classified ZeroOutputWedge and
        # was marked; afterwards it fell through to None until this task.
        from orchestrator.evals.metrics import detect_invocation_error

        marker = detect_invocation_error(_flushed_server_error_result(529))
        assert marker is not None
        assert marker.partition(':')[0] == 'server_error'
        assert '529' in marker

    def test_timed_out_flushed_5xx_outranks_the_wedge_tier(self):
        # A SIGTERM-flushed 5xx is a PROVIDER outage, not a local wedge; the
        # upstream ServerError tier exists precisely because ranking the
        # wedge first discarded the 5xx evidence, so the eval marker must not
        # re-flatten it. Mirrors test_body_less_429_outranks_the_wedge_tier.
        from orchestrator.evals.metrics import detect_invocation_error

        marker = detect_invocation_error(_flushed_server_error_result(529))
        assert marker is not None
        assert 'wedge' not in marker.lower()

    def test_plain_wedge_without_api_error_status_is_still_a_wedge(self):
        # The converse guard: stops the new arm from swallowing the genuine
        # local-wedge signal when there is no api_error_status at all.
        from orchestrator.evals.metrics import detect_invocation_error

        marker = detect_invocation_error(_wedged_agent_result())
        assert marker is not None
        assert marker.partition(':')[0] == 'wedge'
        assert 'server_error' not in marker

    def test_near_cap_warning_outranks_the_server_error_tier(self):
        # classify_invocation ranks NearCap ABOVE ServerError (invocation_
        # outcome.py:502-505 vs :523): a real 5xx whose body ALSO carries a
        # near-cap warning classifies NearCap, not ServerError, so this must
        # still return None — deliberately, not a gap. This is the genuinely
        # interesting boundary the 5xx arm introduced (reviewer:
        # test-coverage): a provider-refused invocation (api_error_status=529)
        # whose body happens to warn of an imminent cap is scored on content,
        # not excluded as infra, because the near-cap text is the more
        # specific, actionable signal.
        from shared.cli_invoke import AgentResult

        from orchestrator.evals.metrics import detect_invocation_error

        near_with_5xx = AgentResult(
            success=False,
            output="You're close to your usage limit for this session",
            cost_usd=0.9, duration_ms=30_000, turns=6,
            api_error_status=529,
        )
        assert detect_invocation_error(near_with_5xx) is None


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
# clamp_unit_score — THE single output-range contract for BOTH plan-quality
# instruments (task 3410)
#
# score_plan_structure and judge_plan_quality reduce onto the SAME axis
# (EvalMetrics.plan_quality), so — exactly as is_scorable_plan (below) is the
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
# is_scorable_plan — THE single "does this artifact carry model content?" test
# (3118 step-19/20)
#
# Extracted because the taint decision and the structural floor's 0.0
# short-circuit were two INDEPENDENT tests that DISAGREED: run_architect_eval
# used raw dict truthiness (`not plan`) while score_plan_structure used
# `not plan.get('steps')`. `_create_plan` persists a TRUTHY header-only dict
# with zero steps, so a cap landing right after the architect's first
# plan-tools call fell straight into the gap — untainted, then floored to a
# fabricated 0.0, the exact defect this task exists to remove.
#
# Both call sites now share this one function, so the parity below is
# structural, not a convention someone has to remember.
# ---------------------------------------------------------------------------

class TestIsScorablePlan:
    @pytest.mark.parametrize('plan', _UNSCORABLE_PLAN_SHAPES)
    def test_shapes_without_steps_are_not_scorable(self, plan):
        from orchestrator.evals.judge import is_scorable_plan

        assert is_scorable_plan(plan) is False

    def test_a_plan_with_steps_is_scorable(self):
        from orchestrator.evals.judge import is_scorable_plan

        assert is_scorable_plan(_well_formed_plan()) is True
        # Even a poor plan is SCORABLE — the predicate asks "is there content to
        # measure?", never "is the content good?". A bad plan must keep earning
        # its low score rather than being laundered into an exclusion.
        assert is_scorable_plan(_degenerate_plan()) is True

    @pytest.mark.parametrize('plan', _UNSCORABLE_PLAN_SHAPES)
    def test_rejected_shapes_are_exactly_the_structural_floor_zero(self, plan):
        """ANTI-DRIFT: the predicate and the floor's short-circuit must agree.

        The blocking defect was two "is this a real plan?" tests disagreeing.
        Every shape the predicate rejects must be a shape the floor scores 0.0
        by short-circuit — so a cell can never be left untainted while the floor
        simultaneously refuses to derive a content score for it.
        """
        from orchestrator.evals.judge import is_scorable_plan, score_plan_structure

        assert is_scorable_plan(plan) is False
        assert score_plan_structure(plan) == 0.0


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
    clamp/NaN handling lives AFTER the ``structured_output or
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

    async def test_cap_refused_judge_invoke_records_invocation_error(self):
        # The judge's OWN invoke can 429 — historically indistinguishable from
        # an unparseable answer, because both produced the same bare
        # plan_quality=None verdict. The marker makes the infra cause legible.
        from orchestrator.evals.judge import judge_plan_quality

        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=_cap_agent_result()),
        ):
            verdict = await judge_plan_quality(
                _well_formed_plan(), 'diff', _judge_task(),
            )

        assert verdict.plan_quality is None
        assert isinstance(verdict.invocation_error, str)
        assert verdict.invocation_error
        assert 'cap' in verdict.invocation_error.lower()

    async def test_unparseable_but_successful_judge_output_carries_no_marker(self):
        # The CONTRAST that makes the marker meaningful: a judge that really
        # answered and simply produced garbage is a CONTENT failure, so its
        # None plan_quality stays unmarked and run_architect_eval keeps
        # degrading to the deterministic structural floor.
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
            verdict = await judge_plan_quality(
                _well_formed_plan(), 'diff', _judge_task(),
            )

        assert verdict.plan_quality is None
        assert verdict.invocation_error is None

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
            verdict = await judge_plan_quality(
                _well_formed_plan(), 'diff', _judge_task(),
            )

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
            verdict = await judge_plan_quality(
                _well_formed_plan(), 'diff', _judge_task(),
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
        assert verdict.plan_quality == expected
        assert verdict.invocation_error is None
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
            verdict = await judge_plan_quality(
                _well_formed_plan(), 'diff', _judge_task(),
            )

        assert verdict.plan_quality == 0.83
        assert _judge_warnings(caplog) == []

    # -- The hole a bare clamp leaves open: non-finite answers (task 3410, --
    # -- narrowed by task 4190) ---------------------------------------------
    # A bare clamp does NOT keep plan_quality in [0, 1] for NaN: NaN is
    # unordered, so it passes straight through unclamped (see
    # clamp_unit_score's docstring in judge.py for the mechanics). A bare
    # clamp DOES keep +/-Infinity in [0, 1] — but only by fabricating the
    # best/worst possible score (1.0/0.0) on a surface
    # report._mean_plan_quality -> select_survivors actually ranks on. Both
    # are equally NOT a judgement, so both degrade to the None sentinel.
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
    async def test_non_finite_answer_degrades_to_the_none_sentinel(
        self, raw, via, caplog,
    ):
        from orchestrator.evals.judge import judge_plan_quality

        caplog.set_level(logging.WARNING, logger='orchestrator.evals.judge')
        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=_judge_result_scoring(raw, via=via)),
        ):
            verdict = await judge_plan_quality(
                _well_formed_plan(), 'diff', _judge_task(),
            )

        # The existing sentinel run_architect_eval already degrades to the
        # deterministic structural floor on — never a fabricated pool-best
        # 1.0 laundered into a number a downstream mean could be poisoned by.
        assert verdict.plan_quality is None
        # A nonsense answer is a CONTENT failure, never the 3118 infra-refusal
        # exclusion shape.
        assert verdict.invocation_error is None
        # Same degraded shape the parse-failure fallback uses — pinned here
        # too, not just plan_quality, so a future edit can't quietly leave
        # per_criterion/reasoning out of step with that documented shape.
        assert verdict.per_criterion == {}
        assert repr(raw) in verdict.reasoning

        warnings = _judge_warnings(caplog)
        assert len(warnings) == 1
        assert 'df_task_2605' in warnings[0]  # WHICH cell to go look at
        assert repr(raw) in warnings[0]       # WHAT was wrong


# ---------------------------------------------------------------------------
# PlanQualityVerdict.cost_usd (step-1/2, eval-revival υ) — the judge's OWN
# invocation spend
#
# run_architect_eval never sees the judge's AgentResult, only the returned
# PlanQualityVerdict — so the judge invocation's cost has to ride the return
# value. A trailing, defaulted field (mirroring invocation_error,
# judge.py:458-463) keeps every existing 3-arg construction site — including
# every judge test above predating this task — reading back cost_usd=0.0
# rather than crashing.
# ---------------------------------------------------------------------------

class TestPlanQualityVerdictCarriesJudgeSpend:
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
            verdict = await judge_plan_quality(
                _well_formed_plan(), 'diff', _judge_task(),
            )

        assert verdict.cost_usd == pytest.approx(0.42)

    # -- Amendment (reviewer design-coherence): the PRODUCER coerces too ----
    # PlanQualityVerdict.cost_usd is declared ``float``; these pin that the
    # declared type is actually true at construction, not just enforced by
    # the runner's defensive read (_verdict_cost_usd / coerce_cost_usd).

    @pytest.mark.asyncio
    async def test_success_path_with_unset_mock_cost_degrades_to_zero(self):
        """A bare MagicMock invoke result that never sets ``cost_usd`` must
        read back a real ``0.0`` float, not a Mock instance and not a
        fabricated ``1.0`` (MagicMock's default ``__float__``).

        This is what makes the producer-side coercion safe for the ~20
        pre-existing judge tests that construct exactly this kind of bare
        mock and never touch ``cost_usd``: none of them assert on the field,
        and this pins the value they'd now get if they did.
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
            verdict = await judge_plan_quality(
                _well_formed_plan(), 'diff', _judge_task(),
            )

        assert verdict.cost_usd == 0.0
        assert isinstance(verdict.cost_usd, float)

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
            verdict = await judge_plan_quality(
                _well_formed_plan(), 'diff', _judge_task(),
            )

        assert verdict.cost_usd == 0.0

    def test_legacy_three_arg_construction_reads_back_zero_spend(self):
        """Every pre-existing construction site (and any older persisted /
        monkeypatched verdict) must keep working and read back "no spend
        recorded" rather than crashing."""
        from orchestrator.evals.judge import PlanQualityVerdict

        verdict = PlanQualityVerdict(
            plan_quality=0.5, per_criterion={}, reasoning='ok',
        )
        assert verdict.cost_usd == 0.0

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
            output=_CAP_TEXT,
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
            verdict = await judge_plan_quality(
                _well_formed_plan(), 'diff', _judge_task(),
            )

        assert isinstance(verdict.invocation_error, str) and verdict.invocation_error
        assert verdict.cost_usd == pytest.approx(0.07)

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
            verdict = await judge_plan_quality(
                _well_formed_plan(), 'diff', _judge_task(),
            )

        assert verdict.plan_quality is None
        assert verdict.cost_usd == pytest.approx(0.31)

    @pytest.mark.asyncio
    async def test_nan_path_carries_the_invocations_own_spend(self):
        from orchestrator.evals.judge import judge_plan_quality

        fake = _judge_result_scoring(float('nan'), via='structured_output')
        fake.cost_usd = 0.29
        with patch(
            'orchestrator.evals.judge.invoke_agent',
            AsyncMock(return_value=fake),
        ):
            verdict = await judge_plan_quality(
                _well_formed_plan(), 'diff', _judge_task(),
            )

        assert verdict.plan_quality is None
        assert verdict.cost_usd == pytest.approx(0.29)

    # -- The PRE-invoke refusal fabricates no spend (step-3/4) --------------
    # Nothing to judge, so invoke_agent is never awaited — recording spend
    # here would fabricate a cost for an invocation that never happened.

    @pytest.mark.asyncio
    @pytest.mark.parametrize('plan', _UNSCORABLE_PLAN_SHAPES)
    async def test_pre_invoke_refusal_records_no_fabricated_spend(self, plan):
        from orchestrator.evals.judge import judge_plan_quality

        mock_invoke = AsyncMock()
        with patch('orchestrator.evals.judge.invoke_agent', mock_invoke):
            verdict = await judge_plan_quality(plan, 'diff', _judge_task())

        mock_invoke.assert_not_awaited()
        assert verdict.cost_usd == 0.0


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
# Task 3302 closed the two OUTER halves: the CALL SITE (run_architect_eval
# gates on is_scorable_plan) and the READER (report._plan_quality_score floors
# an on-disk no-plan cell). This class pins the INSTRUMENT itself, so a second
# caller — a backfill/re-scoring script, a new eval path, prompt_opt, a resume
# wave — cannot silently re-open the same defect.
#
# Parametrized over the SHARED _UNSCORABLE_PLAN_SHAPES corpus (line 103), the
# same list TestIsScorablePlan uses: any shape added to the predicate's corpus
# automatically extends the judge path's coverage, so the two instruments can
# never be tested out of step with each other.
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
    alongside the verdict, since "was the LLM consulted at all?" is half of
    what these tests pin.
    """
    from orchestrator.evals.judge import judge_plan_quality

    with patch(
        'orchestrator.evals.judge.invoke_agent',
        AsyncMock(return_value=_confident_judge_result()),
    ) as mock_invoke:
        verdict = await judge_plan_quality(
            plan, 'diff', _judge_task() if task is None else task,
        )
    return verdict, mock_invoke


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
        verdict, mock_invoke = await _judge_with_a_confident_llm(plan)
        floor = score_plan_structure(plan)

        # The floor's own value, DERIVED — never the judge's confident 0.95.
        assert verdict.plan_quality == floor == 0.0
        # Nothing to judge, so no spend: the refusal happens BEFORE the prompt
        # is built and before invoke_agent is awaited.
        mock_invoke.assert_not_awaited()
        # A definite CONTENT verdict: never the None sentinel (parse failure /
        # transport refusal) and never the cap-tainted exclusion shape — the
        # content-failure vs infra-failure distinction tasks 3118 and 3302 both
        # turn on. A stepless plan from a healthy judge call is a REAL 0.0.
        assert verdict.invocation_error is None
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
        pytest.param(_well_formed_plan(), id='CONTROL-well-formed'),
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
        from orchestrator.evals.judge import is_scorable_plan, score_plan_structure

        caplog.set_level(logging.WARNING, logger='orchestrator.evals.judge')
        assert is_scorable_plan(plan)
        verdict, mock_invoke = await _judge_with_a_confident_llm(plan)

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
        """eval-revival π/ρ: architect-fable-high and architect-fable-max join
        ARCHITECT_EVAL_CONFIGS.

        Parity discipline (eval-revival decision 11): the four pre-existing
        candidates stay byte-unchanged (dataclass equality) when a new
        candidate is appended; the new candidate is config-only — it rides
        run_ofat_stage's existing generic role='architect' branch (task 2478).
        """
        from orchestrator.evals.configs import ARCHITECT_EVAL_CONFIGS, EvalConfig

        existing = [
            EvalConfig('architect-opus-high', 'claude', 'opus', 'high', role='architect'),
            EvalConfig('architect-opus-max', 'claude', 'opus', 'max', role='architect'),
            EvalConfig('architect-sonnet-high', 'claude', 'sonnet', 'high', role='architect'),
            EvalConfig('architect-fable-high', 'claude', 'claude-fable-5', 'high', role='architect'),
        ]
        by_name = {c.name: c for c in ARCHITECT_EVAL_CONFIGS}
        for expected in existing:
            assert by_name[expected.name] == expected

        fable = by_name['architect-fable-high']
        assert fable.backend == 'claude'
        assert fable.model == 'claude-fable-5'
        assert fable.effort == 'high'
        assert fable.role == 'architect'

        # architect-fable-max's field-level shape (backend/model/effort/role)
        # is pinned by test_architect_fable_max_is_effort_matched_to_opus_incumbent
        # below, not duplicated here — this test's job is parity + membership.
        assert {c.name for c in ARCHITECT_EVAL_CONFIGS} == {
            'architect-opus-high',
            'architect-opus-max',
            'architect-sonnet-high',
            'architect-fable-high',
            'architect-fable-max',
        }

    def test_architect_fable_max_is_effort_matched_to_opus_incumbent(self):
        """eval-revival ρ: architect-fable-max is the effort-matched fable variant.

        Per the OFAT methodology (mirroring JUDGE_EVAL_CONFIGS: "effort fixed →
        ONLY the model varies"), architect-fable-max holds effort/role/backend
        equal to its architect-opus-max incumbent and varies EXACTLY the model
        axis. v1's screen compared fable@high against opus@max — a confound —
        so fable-trial-v2 δ's stage-2 screen needs this effort-matched pair to
        isolate the model delta cleanly.
        """
        from orchestrator.evals.configs import get_config_by_name, ofat_candidates

        cfg = get_config_by_name('architect-fable-max')
        assert cfg is not None
        assert cfg.backend == 'claude'
        assert cfg.model == 'claude-fable-5'
        assert cfg.effort == 'max'
        assert cfg.role == 'architect'
        # A native cloud claude candidate, never a proxied bundle (the same
        # discriminator _cloud_implementer_incumbents uses).
        assert not cfg.env_overrides

        incumbent = get_config_by_name('architect-opus-max')
        assert incumbent is not None
        assert cfg.effort == incumbent.effort
        assert cfg.role == incumbent.role
        assert cfg.backend == incumbent.backend
        assert cfg.model != incumbent.model

        assert 'architect-fable-max' in {
            c.name for c in ofat_candidates() if c.role == 'architect'
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


def _arch_task_no_reference() -> dict:
    """``_arch_task()`` minus the ``reference`` key.

    The shape the three back-filled fixtures had before task 3628 — a
    top-level ``post_task_commit`` with no ``reference`` block — which
    ``run_architect_eval`` reduces to an EMPTY reference diff.
    """
    task = _arch_task()
    task.pop('reference', None)
    return task


async def _run_architect_eval_hermetic(
    cfg,
    *,
    produced_plan: dict,
    judge_return=None,
    judge_side_effect=None,
    arch_success: bool = True,
    invoke_side_effect=None,
    arch_result=None,
    orch_prices: dict | None = None,
    orch_config_side_effect=None,
    task_override=None,
    reference_diff: str = '--- a/x\n+++ b/x\n+ landed change\n',
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

    ``orch_prices`` seeds the PRICE TABLE on the patched ``build_eval_orch_config``
    return, which the runner reads to resolve the architect component's cost
    provenance (Invariant P5). It defaults to an EMPTY dict rather than a bare
    ``MagicMock`` deliberately: a bare MagicMock's ``.get(model)`` returns a
    truthy Mock, which would send ``resolve_cost_usd`` down its price-table
    branch and multiply Mocks into ``cost_usd``. The default ``invoke_agent``
    return likewise carries REAL integer token/turn counts for EVERY usage leaf
    the runner reads (input/output, both cache counts, turns) for the same
    reason — a truthy Mock is one that ``or 0`` cannot rescue, and it would
    land verbatim in the persisted cell. Both make the harness model reality so
    a RED here is caused by production behaviour, never by the double.

    ``orch_config_side_effect`` makes the patched ``build_eval_orch_config``
    RAISE — the harness-crash shape where the eval orch config (and therefore
    the price table) is never bound at all. It has to live here rather than in
    a caller-side ``patch``: this helper patches the same name from inside, so
    an outer patch would be shadowed.

    ``task_override`` replaces the task dict ``load_task`` returns (e.g.
    ``_arch_task_no_reference()``), and ``reference_diff`` is what the patched
    ``get_diff_between_commits`` returns — both defaulting to today's hardcoded
    values so every existing caller keeps byte-identical behavior. Together
    they drive the ``judged_without_reference`` cases (task 3628).
    """
    from orchestrator.evals import runner
    from orchestrator.evals.judge import PlanQualityVerdict

    if judge_return is None and judge_side_effect is None:
        judge_return = PlanQualityVerdict(
            plan_quality=0.77, per_criterion={}, reasoning='good',
        )

    invoke_return = arch_result if arch_result is not None else MagicMock(
        success=arch_success, cost_usd=1.23, duration_ms=4567, output='done',
        input_tokens=12_000, output_tokens=3_000,
        cache_read_tokens=48_000, cache_create_tokens=6_000, turns=9,
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
                AsyncMock(return_value=reference_diff)))
        p(patch('orchestrator.agents.invoke.invoke_agent', mock_invoke))
        p(patch('orchestrator.artifacts.TaskArtifacts',
                MagicMock(return_value=artifacts_instance)))
        p(patch('orchestrator.agents.briefing.BriefingAssembler',
                MagicMock(return_value=briefing_instance)))
        p(patch('orchestrator.evals.runner.build_eval_orch_config',
                MagicMock(
                    return_value=MagicMock(
                        prices=orch_prices if orch_prices is not None else {},
                    ),
                    side_effect=orch_config_side_effect,
                )))
        p(patch('orchestrator.evals.judge.judge_plan_quality', mock_judge))
        p(patch('orchestrator.evals.runner.save_result', mock_save))
        p(patch('orchestrator.evals.runner.load_task',
                MagicMock(return_value=(
                    _arch_task() if task_override is None else task_override
                ))))
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
        assert 'judge:raised' in (result.metrics['invocation_error'] or '')

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

    async def test_timeout_is_marked_but_keeps_scoring_on_content(self):
        """A timeout is MARKED yet deliberately NOT tainted (asymmetry pinned).

        Marked: without a marker the cell was byte-indistinguishable from a
        genuinely terrible plan — the same defect the cap path removes.

        Not tainted: unlike a cap hit (a property of the SCHEDULE), a timeout is
        CANDIDATE-attributable — the model was asked and did not finish inside
        the operator's budget. Excluding it would let a pathologically slow
        candidate dodge the penalty its competitors paid, so the cell keeps
        scoring on content and carries the reliability signal in BOTH
        ``outcome='timeout'`` and ``invocation_error``.
        """
        result, _ = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=_well_formed_plan(),
            invoke_side_effect=TimeoutError(),
        )
        assert result.outcome == 'timeout'
        marker = result.metrics['invocation_error']
        assert isinstance(marker, str) and marker.startswith('architect:')
        assert 'timeout' in marker.lower()
        assert result.metrics['cap_tainted'] is False
        assert result.metrics['plan_quality'] is not None

    @pytest.mark.parametrize('arch_result_factory', [None, _cap_agent_result])
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
        result, _ = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=_well_formed_plan(),
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

        plan = _well_formed_plan()
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=plan,
            arch_result=_cap_agent_result(),
        )

        # The measurement SURVIVES: scored on the deterministic floor, kept in
        # the aggregate — symmetric with the judge-only refusal.
        assert result.metrics['plan_quality'] == score_plan_structure(plan)
        assert result.metrics['cap_tainted'] is False
        # ...and the cell is self-consistent: a plan with steps, not a null.
        assert result.metrics['plan_steps'] == len(plan['steps'])

        # The refusal is still RECORDED, so a reader knows the LLM judge never
        # ran on this cell...
        marker = result.metrics['invocation_error']
        assert isinstance(marker, str) and marker.startswith('architect:')
        assert 'cap' in marker.lower()
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
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=_stub_with_steps(empty_steps),
            arch_result=_cap_agent_result(),
        )

        assert result.metrics['cap_tainted'] is True
        assert result.metrics['plan_quality'] is None
        assert result.metrics['plan_quality'] != 0.0
        assert result.metrics['plan_steps'] == 0

        marker = result.metrics['invocation_error']
        assert isinstance(marker, str) and marker.startswith('architect:')
        assert 'cap' in marker.lower()

        # Nothing to judge, and inside the cap window the judge would 429 too.
        mocks['judge'].assert_not_called()

    async def test_wedged_architect_invoke_is_marked_and_excluded(self):
        # A zero-output wedge (full timeout, zero transcript turns, zero cost)
        # produced no model answer at all — unmeasurable for the same reason a
        # 429 is, so it must not be scored 0.0 either (reviewer: robustness).
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan={},
            arch_result=_wedged_agent_result(),
        )
        assert result.metrics['cap_tainted'] is True
        assert result.metrics['plan_quality'] is None
        marker = result.metrics['invocation_error']
        assert isinstance(marker, str) and marker.startswith('architect:')
        assert 'wedge' in marker.lower()
        mocks['judge'].assert_not_called()

    async def test_harness_exception_is_marked_and_excluded(self):
        # OUR crash, not the candidate's: the architect was never even asked, so
        # charging a fabricated 0.0 to the candidate would be plainly wrong.
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan={},
            invoke_side_effect=RuntimeError('worktree exploded'),
        )
        assert result.outcome == 'blocked'
        assert result.metrics['cap_tainted'] is True
        assert result.metrics['plan_quality'] is None
        marker = result.metrics['invocation_error']
        assert isinstance(marker, str) and marker.startswith('architect:')
        assert 'harness_error' in marker
        assert 'RuntimeError' in marker
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
            'architect-codex-high', 'codex', 'gpt-5', 'high', role='architect',
        )
        result, _ = await _run_architect_eval_hermetic(
            codex_cfg, produced_plan={}, arch_result=_codex_cap_agent_result(),
        )
        marker = result.metrics['invocation_error']
        assert isinstance(marker, str) and marker.startswith('architect:')
        assert 'cap' in marker.lower()
        assert result.metrics['cap_tainted'] is True

        # Same payload, claude candidate: the codex table is never consulted.
        claude_result, _ = await _run_architect_eval_hermetic(
            self._cfg(), produced_plan={}, arch_result=_codex_cap_agent_result(),
        )
        assert claude_result.metrics['invocation_error'] is None
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
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan={},
            arch_result=_cap_agent_result(),
        )

        # The cell is marked as NOT a content measurement...
        assert result.metrics['cap_tainted'] is True
        marker = result.metrics['invocation_error']
        assert isinstance(marker, str) and marker
        assert marker.startswith('architect:')   # names the refused STAGE
        assert 'cap' in marker.lower()           # ...and the refusal itself

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
        from orchestrator.evals.judge import PlanQualityVerdict, score_plan_structure

        plan = _well_formed_plan()
        result, _ = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=plan,
            judge_return=PlanQualityVerdict(
                plan_quality=None,
                per_criterion={},
                reasoning='plan judge invocation refused: cap_hit: ...',
                invocation_error=f'cap_hit: {_CAP_TEXT}',
            ),
        )

        # The content-derived floor is KEPT — not nulled, not excluded.
        assert result.metrics['plan_quality'] is not None
        assert result.metrics['plan_quality'] == score_plan_structure(plan)
        assert result.metrics['cap_tainted'] is False

        # ...but the judge's refusal is recorded, named by STAGE.
        marker = result.metrics['invocation_error']
        assert isinstance(marker, str) and marker
        assert marker.startswith('judge:')
        assert 'cap' in marker.lower()

    async def test_healthy_run_carries_no_marker_and_is_not_tainted(self):
        # The untouched baseline: nothing refused anywhere → no marker at all.
        result, _ = await _run_architect_eval_hermetic(
            self._cfg(), produced_plan=_well_formed_plan(),
        )
        assert result.metrics['invocation_error'] is None
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
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role='architect',
        )

    async def test_judge_spend_recorded_and_folded_into_cell_cost(self):
        from orchestrator.evals.judge import PlanQualityVerdict

        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=_well_formed_plan(),
            judge_return=PlanQualityVerdict(
                plan_quality=0.77, per_criterion={}, reasoning='good',
                cost_usd=0.42,
            ),
        )

        # The RETURNED result...
        assert result.metrics['judge_cost_usd'] == pytest.approx(0.42)
        assert result.metrics['judge_invocations'] == 1
        # ...the SUBSET invariant: architect spend (1.23, the harness default)
        # PLUS the judge spend, never the architect spend alone.
        assert result.metrics['cost_usd'] == pytest.approx(1.23 + 0.42)

        # ...and what was actually PERSISTED via save_result, not just what
        # the function happens to return.
        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['judge_cost_usd'] == pytest.approx(0.42)
        assert persisted['judge_invocations'] == 1
        assert persisted['cost_usd'] == pytest.approx(1.23 + 0.42)


def _token_agent_result(input_tokens, output_tokens, *, cost_usd: float = 1.23):
    """A REAL ``AgentResult`` carrying explicit token counts.

    Real, not a MagicMock, because the ``int | None`` "provider did not report"
    shape is exactly what these tests pin — a MagicMock cannot express a
    genuine ``None`` on an attribute the runner reads.
    """
    from shared.cli_invoke import AgentResult

    return AgentResult(
        success=True,
        output='done',
        cost_usd=cost_usd,
        duration_ms=4567,
        turns=3,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


@pytest.mark.asyncio
class TestArchitectCellStampsTokenUsageAndProxySignal:
    """Task 3656 step-1: the architect cell must record the run's TOKEN USAGE
    and its PROXY signal, not leave all three at their dataclass defaults.

    These are the three inputs :func:`~orchestrator.evals.metrics.resolve_cost_usd`
    needs to resolve cost provenance per Invariant P5. Without them the
    price-table path is unreachable on the architect side no matter what the
    resolver is asked, so they are pinned BEFORE the resolution itself
    (step-5). Every assertion checks the PERSISTED cell as well as the returned
    result — the persisted JSON is what ``report.py`` and an operator read.
    """

    def _cfg(self, **kw):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high',
            role='architect', **kw,
        )

    async def test_reported_token_counts_are_stamped(self):
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=_well_formed_plan(),
            arch_result=_token_agent_result(12_000, 3_000),
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
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=_well_formed_plan(),
            arch_result=_token_agent_result(None, None),
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
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(), produced_plan=_well_formed_plan(),
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
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=_well_formed_plan(),
            invoke_side_effect=TimeoutError(),
        )

        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['tokens_per_second'] == 0.0
        assert persisted['turns_used'] == 0
        assert result.metrics['tokens_per_second'] == 0.0

    async def test_proxied_candidate_is_flagged_local(self):
        """``ANTHROPIC_BASE_URL`` set == a PROXIED endpoint, the signal Invariant
        P5 keys on to stop trusting the CLI's own cost figure."""
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(env_overrides={'ANTHROPIC_BASE_URL': 'http://localhost:8000/v1'}),
            produced_plan=_well_formed_plan(),
        )

        assert result.metrics['is_local_model'] is True
        assert mocks['save'].call_args.args[0].metrics['is_local_model'] is True

    async def test_native_candidate_is_not_flagged_local(self):
        """Today's shape (``architect-opus-max`` / ``architect-fable-max``):
        native cloud, no proxy, so the CLI cost stays trustworthy."""
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(env_overrides={}),
            produced_plan=_well_formed_plan(),
        )

        assert result.metrics['is_local_model'] is False
        assert mocks['save'].call_args.args[0].metrics['is_local_model'] is False


_PROXY_ENV = {'ANTHROPIC_BASE_URL': 'http://localhost:8000/v1'}


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
            'architect-candidate', 'claude', effort='high', role='architect', **kw,
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
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(env_overrides={}),
            produced_plan=_well_formed_plan(),
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
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(model='qwen3-coder', env_overrides=dict(_PROXY_ENV)),
            produced_plan=_well_formed_plan(),
            arch_result=_token_agent_result(1_000_000, 500_000, cost_usd=99.0),
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
            result, mocks = await _run_architect_eval_hermetic(
                self._cfg(model='qwen3-coder', env_overrides=dict(_PROXY_ENV)),
                produced_plan=_well_formed_plan(),
                arch_result=_token_agent_result(1_000_000, 500_000, cost_usd=99.0),
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
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(env_overrides={}),
            produced_plan=_well_formed_plan(),
            orch_config_side_effect=RuntimeError('config build exploded'),
        )

        persisted = mocks['save'].call_args.args[0].metrics
        # The cell LANDED — marked and excluded, not lost.
        assert persisted['cap_tainted'] is True
        assert 'harness_error' in (persisted['invocation_error'] or '')
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
            result, mocks = await _run_architect_eval_hermetic(
                self._cfg(model='qwen3-coder', env_overrides=dict(_PROXY_ENV)),
                produced_plan=_well_formed_plan(),
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
            'architect-candidate', 'claude', effort='high', role='architect', **kw,
        )

    def _spending_judge(self):
        from orchestrator.evals.judge import PlanQualityVerdict

        return PlanQualityVerdict(
            plan_quality=0.77, per_criterion={}, reasoning='good', cost_usd=0.42,
        )

    async def test_price_table_architect_plus_cli_judge_reads_mixed(self):
        """A price-table architect component beside a CLI judge component: the
        figure is the RESOLVED architect cost plus the judge's CLI cost, and the
        label says the two disagree rather than claiming either one."""
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(model='qwen3-coder', env_overrides=dict(_PROXY_ENV)),
            produced_plan=_well_formed_plan(),
            judge_return=self._spending_judge(),
            arch_result=_token_agent_result(1_000_000, 500_000, cost_usd=99.0),
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
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(env_overrides={}),
            produced_plan=_well_formed_plan(),
            judge_return=self._spending_judge(),
            orch_prices={},
        )

        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['cost_source'] == 'cli'
        assert persisted['cost_usd'] == pytest.approx(1.23 + 0.42)
        assert result.metrics['cost_source'] == 'cli'

    async def test_judge_skipped_labels_the_architect_component_alone(self):
        """An unscorable plan SKIPS the judge (``_create_plan_stub`` is the
        header-only artifact ``create_plan`` writes), so no judge dollars enter
        the sum and the label describes the architect component alone — NOT
        'mixed', which would announce a blend that never happened."""
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(model='qwen3-coder', env_overrides=dict(_PROXY_ENV)),
            produced_plan=_create_plan_stub(),
            judge_return=self._spending_judge(),
            arch_result=_token_agent_result(1_000_000, 500_000, cost_usd=99.0),
            orch_prices={'qwen3-coder': {'input_per_1m': 0.30, 'output_per_1m': 1.20}},
        )

        mocks['judge'].assert_not_called()
        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['judge_cost_usd'] == pytest.approx(0.0)
        assert persisted['cost_source'] == 'price_table'
        assert persisted['cost_usd'] == pytest.approx(0.90)
        assert result.metrics['cost_source'] == 'price_table'


@pytest.mark.asyncio
class TestArchitectRunnerJudgeCostReadIsDefensive:
    """Step-7: the runner's read of ``verdict.cost_usd`` must not fabricate
    spend on a judge-skipped branch, and an unreadable verdict field must
    degrade gracefully rather than crash the cell or mis-attribute the
    failure to "the judge raised" (the judge DID run and DID answer; only
    its cost figure is unreadable).
    """

    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role='architect',
        )

    # -- (a) NO FABRICATED SPEND on every judge-skipped branch ----------------
    # Each of these branches already asserts ``mocks['judge'].assert_not_called()``
    # elsewhere in this file; here the pin is on the COST fields specifically —
    # a judge that never ran must never leave a nonzero judge_cost_usd/
    # judge_invocations behind, and the cell's cost_usd must equal the
    # ARCHITECT's own spend for that branch, nothing more.

    async def test_cap_tainted_branch_records_architect_spend_only(self):
        """Cap landed right after ``create_plan``: header-only stub, tainted."""
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=_stub_with_steps([]),
            arch_result=_cap_agent_result(),
        )
        mocks['judge'].assert_not_called()
        assert result.metrics['judge_cost_usd'] == 0.0
        assert result.metrics['judge_invocations'] == 0
        # _cap_agent_result() itself spends $0.00 — no fabricated judge
        # dollars riding along with a refusal that burned nothing.
        assert result.metrics['cost_usd'] == pytest.approx(0.0)

    async def test_cap_refusal_with_a_plan_records_architect_spend_only(self):
        """Cap landed MID-run, after a real plan with steps had landed."""
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=_well_formed_plan(),
            arch_result=_cap_agent_result(),
        )
        mocks['judge'].assert_not_called()
        assert result.metrics['judge_cost_usd'] == 0.0
        assert result.metrics['judge_invocations'] == 0
        assert result.metrics['cost_usd'] == pytest.approx(0.0)

    async def test_healthy_stepless_plan_records_architect_spend_only(self):
        """Architect ran fine (no cap) but produced nothing worth judging."""
        result, mocks = await _run_architect_eval_hermetic(
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

    # -- (b) an unreadable or nonsensical verdict.cost_usd must not damage
    #    the cell ------------------------------------------------------------

    @pytest.mark.parametrize('bad_verdict', [
        pytest.param(
            types.SimpleNamespace(
                plan_quality=0.77, per_criterion={}, reasoning='r',
                invocation_error=None,
            ),
            id='missing-cost_usd-field',
        ),
        pytest.param(
            types.SimpleNamespace(
                plan_quality=0.77, per_criterion={}, reasoning='r',
                invocation_error=None, cost_usd=None,
            ),
            id='cost_usd-is-None',
        ),
        # Amendment (reviewer robustness): a READABLE but NON-FINITE or
        # negative cost_usd is just as dangerous as a missing one —
        # `arch_cost_usd + judge_cost_usd` would otherwise poison the cell's
        # cost_usd with NaN/Infinity (which json.dump emits as bare,
        # non-standard tokens) or a nonsense negative total, and NaN
        # propagates through every downstream report.py mean for the whole
        # config row.
        pytest.param(
            types.SimpleNamespace(
                plan_quality=0.77, per_criterion={}, reasoning='r',
                invocation_error=None, cost_usd=float('nan'),
            ),
            id='cost_usd-is-nan',
        ),
        pytest.param(
            types.SimpleNamespace(
                plan_quality=0.77, per_criterion={}, reasoning='r',
                invocation_error=None, cost_usd=float('inf'),
            ),
            id='cost_usd-is-infinite',
        ),
        pytest.param(
            types.SimpleNamespace(
                plan_quality=0.77, per_criterion={}, reasoning='r',
                invocation_error=None, cost_usd=-3.5,
            ),
            id='cost_usd-is-negative',
        ),
    ])
    async def test_unreadable_verdict_cost_degrades_the_field_not_the_cell(
        self, bad_verdict, caplog,
    ):
        """A legacy/monkeypatched verdict lacking a readable, or carrying a
        nonsensical, ``cost_usd``.

        Must not crash the cell (``arch_cost_usd + verdict.cost_usd`` when the
        field is ``None`` raises ``TypeError`` today, outside any try/except),
        must not lose the fact that the judge WAS called, and must not log
        "plan judge raised" — that warning means the judge invocation itself
        failed, which is false here: the judge ran and answered fine, only
        this ONE field on its verdict is unreadable or nonsensical (NaN,
        Infinity, negative).
        """
        caplog.set_level(logging.WARNING, logger='orchestrator.evals.runner')

        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=_well_formed_plan(),
            judge_return=bad_verdict,
        )

        mocks['judge'].assert_awaited_once()
        # The judge's real answer survives...
        assert result.metrics['plan_quality'] == 0.77
        # ...the unreadable cost degrades to 0.0, never a fabricated number...
        assert result.metrics['judge_cost_usd'] == 0.0
        # ...but the invocation itself is NOT lost: the judge DID run.
        assert result.metrics['judge_invocations'] == 1
        assert result.metrics['cost_usd'] == pytest.approx(1.23)

        assert not any(
            'plan judge raised' in r.getMessage() for r in caplog.records
        )


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
# itself, pinned by TestJudgePlanQualityRefusesAnUnjudgeableArtifact above — so
# the two blocks are one story: 3302 gated the CALL SITE, 3303 gated the
# INSTRUMENT, and both consult the one is_scorable_plan predicate. This class
# stays load-bearing regardless: it pins the runner-side consequences the
# instrument cannot reach from where it stands — the taint decision, the
# task_id × config.name log line, and that no opus call is made at all.
#
# _STEPLESS_PLANS is deliberately the narrower dict-only subset of
# _UNSCORABLE_PLAN_SHAPES (line 103): run_architect_eval does
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
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role='architect',
        )

    @staticmethod
    def _confident_judge():
        from orchestrator.evals.judge import PlanQualityVerdict

        # What the ungated judge really does with an unjudgeable artifact.
        return PlanQualityVerdict(
            plan_quality=0.9, per_criterion={}, reasoning='looks fine',
        )

    @pytest.mark.parametrize('plan', _STEPLESS_PLANS)
    async def test_stepless_plan_scores_the_structural_floor(self, plan):
        result, mocks = await _run_architect_eval_hermetic(
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
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=plan,
            judge_return=self._confident_judge(),
            arch_success=True,
        )
        persisted = mocks['save'].call_args.args[0].metrics

        assert persisted['cap_tainted'] is False
        assert persisted['invocation_error'] is None
        assert result.metrics['cap_tainted'] is False

    @pytest.mark.parametrize('plan', _STEPLESS_PLANS)
    async def test_the_plan_judge_is_never_invoked(self, plan):
        """Nothing to judge — and inside a cap window the call would 429 anyway,
        so an opus invocation on an unjudgeable artifact is pure waste (the same
        justification the arch_unmeasurable branch already records)."""
        _, mocks = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=plan,
            judge_return=self._confident_judge(),
            arch_success=True,
        )
        mocks['judge'].assert_not_awaited()

    async def test_a_real_plan_still_awaits_the_judge(self):
        """The control: the gate fires ONLY on a stepless artifact."""
        result, mocks = await _run_architect_eval_hermetic(
            self._cfg(), produced_plan=_well_formed_plan(),
        )
        persisted = mocks['save'].call_args.args[0].metrics

        mocks['judge'].assert_awaited_once()
        assert persisted['plan_quality'] == 0.77
        assert persisted['plan_steps'] > 0
        assert persisted['cap_tainted'] is False

    @pytest.mark.parametrize('plan', _STEPLESS_PLANS + [
        pytest.param(None, id='no-plan-artifact'),
        pytest.param(_well_formed_plan(), id='real-plan'),
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
        from orchestrator.evals.judge import is_scorable_plan
        from orchestrator.evals.metrics import produced_a_plan

        _, mocks = await _run_architect_eval_hermetic(
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
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role='architect',
        )

    async def test_judged_blind_cell_is_marked_and_still_scored(self):
        """POSITIVE: the judge ran without ground truth — mark it, keep it.

        The cell is NOT excluded: plan_quality is still the judge's float and
        the cell stays in every pool at that score. The marker bounds VALIDITY,
        not membership.
        """
        _, mocks = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=_well_formed_plan(),
            task_override=_arch_task_no_reference(),
            reference_diff='',
        )
        # Assert on the dict handed to save_result — that dict is what both the
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
        _, mocks = await _run_architect_eval_hermetic(
            self._cfg(), produced_plan=_well_formed_plan(),
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
        _, mocks = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan={},
            arch_result=_cap_agent_result(),
            task_override=_arch_task_no_reference(),
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
        ``_UNSCORABLE_PLAN_SHAPES``: this is a RUNNER-level test, and
        ``run_architect_eval`` does ``plan = artifacts.read_plan() or {}``, so
        it can only ever be handed a dict (see the note above
        ``TestSteplessPlanIsNeverJudged``, which scopes itself the same way).
        """
        _, mocks = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=plan,
            task_override=_arch_task_no_reference(),
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

        plan = _well_formed_plan()
        _, mocks = await _run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=plan,
            judge_side_effect=RuntimeError('judge exploded'),
            task_override=_arch_task_no_reference(),
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
            await _run_architect_eval_hermetic(
                self._cfg(),
                produced_plan=_well_formed_plan(),
                task_override=_arch_task_no_reference(),
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
            await _run_architect_eval_hermetic(
                self._cfg(),
                produced_plan={},
                arch_result=_cap_agent_result(),
                task_override=_arch_task_no_reference(),
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
            await _run_architect_eval_hermetic(
                self._cfg(),
                produced_plan=_well_formed_plan(),
                task_override=_arch_task_no_reference(),
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
    plan_steps: int = 6,
    judged_without_reference: bool | None = False,
):
    """An architect cell that DID produce a plan.

    ``plan_steps`` defaults NONZERO (task 3302): a ``plan_quality`` is a score
    over the steps a plan actually carried, and ``plan_steps > 0`` is the
    predicate the report layer reads to know one exists. A stepless cell is the
    distinct no-plan shape, requested explicitly by the tests that exercise it.

    ``judged_without_reference`` (task 3628) mints a cell whose score came from
    a judge handed an EMPTY reference diff. THREE-VALUED on purpose, because
    the report layer is: ``True``/``False`` both WRITE the key — the production
    shape, since ``EvalMetrics.to_dict`` emits it on every cell — while
    ``None`` OMITS it, minting the pre-σ legacy shape whose validity was never
    measured. The default is an explicit ``False``: a healthy modern cell, not
    a legacy one.
    """
    from orchestrator.evals.runner import EvalResult

    metrics = {
        'role_under_test': 'architect',
        'plan_quality': plan_quality,
        'plan_steps': plan_steps,
        'composite_score': 0.0,
    }
    if judged_without_reference is not None:
        metrics['judged_without_reference'] = judged_without_reference

    return EvalResult(
        task_id=task_id,
        config_name=config_name,
        outcome='done',
        metrics=metrics,
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


def _cap_tainted_result(
    task_id: str = 'df_task_3118',
    config_name: str = 'architect-sonnet-high',
    invocation_error: str | None = None,
):
    """An architect cell whose invocation was refused — NOT a measurement.

    ``invocation_error`` defaults to the campaign cap payload; pass a different
    stage-prefixed marker (e.g. ``'architect:model_not_found: ...'``) to build a
    cell excluded for a PERMANENT rather than a transient cause.
    """
    from orchestrator.evals.runner import EvalResult

    return EvalResult(
        task_id=task_id,
        config_name=config_name,
        outcome='blocked',
        metrics={
            'role_under_test': 'architect',
            'plan_quality': None,
            'cap_tainted': True,
            'invocation_error': (
                invocation_error or f'architect:cap_hit: {_CAP_TEXT}'
            ),
            'composite_score': 0.0,
        },
        worktree_path='/tmp/wt-arch-capped',
    )


_MEAN_SECTION_HEADER = 'plan_quality by config:'


def _mean_section_line(table: str, config_name: str) -> str:
    """The per-config-mean line for *config_name*, scoped to the mean SECTION.

    Scoped deliberately: a config name also appears in the per-cell rows above,
    so a bare ``in`` search over the whole table would match the wrong line.
    """
    lines = table.splitlines()
    start = lines.index(_MEAN_SECTION_HEADER)
    return next(ln for ln in lines[start:] if ln.startswith(config_name))


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

    # -- cap-tainted exclusion (task 3118) --------------------------------
    # A cap-tainted cell is an INFRA failure, not a content measurement, so the
    # aggregate must EXCLUDE it and COUNT the exclusion — never average in a
    # fabricated zero, which would penalise whichever candidate happened to be
    # scheduled inside a cap window.

    def test_rows_carry_cap_tainted_and_invocation_error(self):
        from orchestrator.evals.report import build_plan_quality_report

        report = build_plan_quality_report(
            [_architect_result(), _cap_tainted_result(), _implementer_result()]
        )
        by_task = {r['task_id']: r for r in report['rows']}

        capped = by_task['df_task_3118']
        assert capped['cap_tainted'] is True
        assert capped['invocation_error'].startswith('architect:')
        assert capped['plan_quality'] is None

        # Legacy results, whose metrics predate both keys, read back default-safe.
        assert by_task['df_task_2605']['cap_tainted'] is False
        assert by_task['df_task_2605']['invocation_error'] is None
        assert by_task['df_task_1993']['cap_tainted'] is False

    def test_per_config_mean_excludes_cap_tainted_cells(self):
        from orchestrator.evals.report import build_plan_quality_report

        cfg = 'architect-opus-high'
        report = build_plan_quality_report([
            _architect_result(task_id='t1', config_name=cfg, plan_quality=0.9),
            _architect_result(task_id='t2', config_name=cfg, plan_quality=0.7),
            _cap_tainted_result(task_id='t3', config_name=cfg),
        ])
        configs = {c['config_name']: c for c in report['configs']}
        agg = configs[cfg]

        # (0.9 + 0.7) / 2 — NOT (0.9 + 0.7 + 0.0) / 3 == 0.5333.
        assert agg['mean_plan_quality'] == 0.8
        assert agg['n'] == 2
        assert agg['cap_excluded'] == 1
        assert agg['total'] == 3

    def test_report_level_cap_excluded_total(self):
        from orchestrator.evals.report import build_plan_quality_report

        report = build_plan_quality_report([
            _architect_result(task_id='t1', config_name='a', plan_quality=0.9),
            _cap_tainted_result(task_id='t2', config_name='a'),
            _cap_tainted_result(task_id='t3', config_name='b'),
        ])
        assert report['cap_excluded'] == 2

    # -- judged_without_reference (task 3628) --------------------------------
    #
    # The θ-surface twin of cap_excluded, with one decisive difference: it
    # BOUNDS validity rather than excluding. The flagged cell stays in `n` and
    # in `mean_plan_quality` at its real score.

    def test_rows_distinguish_measured_false_from_an_absent_marker(self):
        """THREE-VALUED: True, False, and None-for-never-measured.

        The row must not collapse absence into False (reviewer:
        architecture-coherence). A metrics dict written before σ landed never
        had its validity measured, and reading it back as False asserts the
        reassuring answer to a question the instrument never asked — the exact
        inversion of the rule the sibling consumer
        ``scripts/run_fable_trial_v2_campaign.py`` enforces per cell.
        """
        from orchestrator.evals.report import build_plan_quality_report

        report = build_plan_quality_report([
            _architect_result(task_id='t1', judged_without_reference=True),
            _architect_result(task_id='t2', judged_without_reference=False),
            _architect_result(task_id='t3', judged_without_reference=None),
        ])
        by_task = {r['task_id']: r for r in report['rows']}

        assert by_task['t1']['judged_without_reference'] is True
        # MEASURED clean — the instrument ran and found a reference diff.
        assert by_task['t2']['judged_without_reference'] is False
        # NEVER MEASURED — a metrics dict predating the field.
        assert by_task['t3']['judged_without_reference'] is None

    def test_per_config_count_bounds_but_does_not_exclude(self):
        """The flagged cells are COUNTED yet still averaged in.

        The single most likely misimplementation is to treat this like
        cap_excluded and drop the cell from the pool. It is not an exclusion:
        the score is real, only the confidence in it is reduced. Pinned by
        asserting the mean is over all THREE cells, not just the healthy one.
        """
        from orchestrator.evals.report import build_plan_quality_report

        cfg = 'architect-opus-high'
        report = build_plan_quality_report([
            _architect_result(
                task_id='t1', config_name=cfg, plan_quality=0.9,
                judged_without_reference=True,
            ),
            _architect_result(
                task_id='t2', config_name=cfg, plan_quality=0.6,
                judged_without_reference=True,
            ),
            _architect_result(task_id='t3', config_name=cfg, plan_quality=0.3),
        ])
        agg = {c['config_name']: c for c in report['configs']}[cfg]

        assert agg['judged_without_reference'] == 2
        assert agg['n'] == 3
        assert agg['mean_plan_quality'] == 0.6  # (0.9 + 0.6 + 0.3) / 3

    def test_cap_tainted_cell_is_not_counted_as_judged_blind(self):
        """Disjoint from cap_excluded: the θ count is scoped to the ADMITTED pool.

        A cap-tainted cell has no plan_quality to bound, so even carrying the
        flag it must land in cap_excluded ALONE — exactly how `no_plan` is
        scoped on this surface.
        """
        from orchestrator.evals.report import build_plan_quality_report

        cfg = 'architect-opus-high'
        capped = _cap_tainted_result(task_id='t2', config_name=cfg)
        capped.metrics['judged_without_reference'] = True

        report = build_plan_quality_report([
            _architect_result(task_id='t1', config_name=cfg, plan_quality=0.9),
            capped,
        ])
        agg = {c['config_name']: c for c in report['configs']}[cfg]

        assert agg['cap_excluded'] == 1
        assert agg['judged_without_reference'] == 0

    def test_report_level_judged_without_reference_total(self):
        from orchestrator.evals.report import build_plan_quality_report

        report = build_plan_quality_report([
            _architect_result(
                task_id='t1', config_name='a', judged_without_reference=True,
            ),
            _architect_result(
                task_id='t2', config_name='b', judged_without_reference=True,
            ),
            _architect_result(task_id='t3', config_name='b'),
        ])
        assert report['judged_without_reference'] == 2
        assert sum(
            c['judged_without_reference'] for c in report['configs']
        ) == report['judged_without_reference']

    def test_one_keyless_admitted_cell_makes_the_count_unmeasured(self):
        """ABSENCE POISONS THE COUNT: None, never a partial sum reading as whole.

        A config whose pool mixes measured and pre-σ cells cannot report how
        many of its scores were judged blind — it can only report that it does
        not know. Reporting the measured ones' count (here 1) would understate
        the bound while wearing the appearance of a complete answer. This is
        ``report._judged_blind_count``'s rule, which task 3632's campaign
        script consumes rather than re-deriving, so the two surfaces of one
        field answer the same question identically.
        """
        from orchestrator.evals.report import build_plan_quality_report

        cfg = 'architect-opus-high'
        report = build_plan_quality_report([
            _architect_result(
                task_id='t1', config_name=cfg, judged_without_reference=True,
            ),
            _architect_result(
                task_id='t2', config_name=cfg, judged_without_reference=None,
            ),
        ])
        agg = {c['config_name']: c for c in report['configs']}[cfg]

        assert agg['judged_without_reference'] is None
        # …with the size of the unmeasured population beside it: 1-of-2 and
        # 2-of-2 are different findings and the None alone cannot say which.
        assert agg['judged_without_reference_unmeasured'] == 1
        # The cells are still in the pool at their real scores — an unmeasured
        # BOUND excludes nothing, exactly as a measured one does not.
        assert agg['n'] == 2
        # …and the campaign-wide total is unknown for the same reason.
        assert report['judged_without_reference'] is None
        assert report['judged_without_reference_unmeasured'] == 1

    def test_a_fully_measured_campaign_still_reports_a_number(self):
        """The None is not contagious to a corpus the instrument DID measure.

        The negative twin of the test above: absence poisons the count, but a
        campaign whose every admitted cell carries the key reports a real
        count — including a real ZERO, which is the one case where 0 is the
        honest answer.
        """
        from orchestrator.evals.report import build_plan_quality_report

        cfg = 'architect-opus-high'
        report = build_plan_quality_report([
            _architect_result(
                task_id='t1', config_name=cfg, judged_without_reference=False,
            ),
            _architect_result(
                task_id='t2', config_name=cfg, judged_without_reference=False,
            ),
        ])
        agg = {c['config_name']: c for c in report['configs']}[cfg]

        assert agg['judged_without_reference'] == 0
        assert agg['judged_without_reference_unmeasured'] == 0
        assert report['judged_without_reference'] == 0

    def test_a_keyless_cap_tainted_cell_does_not_poison_the_count(self):
        """Scoped to the ADMITTED pool, absence included.

        A cap-tainted cell has no plan_quality to bound whether or not it
        carries the marker, so its keylessness must not make the config's bound
        unknown — otherwise every campaign with one cap-window cell would read
        UNMEASURED forever.
        """
        from orchestrator.evals.report import build_plan_quality_report

        cfg = 'architect-opus-high'
        capped = _cap_tainted_result(task_id='t2', config_name=cfg)
        capped.metrics.pop('judged_without_reference', None)

        report = build_plan_quality_report([
            _architect_result(
                task_id='t1', config_name=cfg, judged_without_reference=True,
            ),
            capped,
        ])
        agg = {c['config_name']: c for c in report['configs']}[cfg]

        assert agg['cap_excluded'] == 1
        assert agg['judged_without_reference'] == 1
        assert agg['judged_without_reference_unmeasured'] == 0

    def test_non_architect_rows_are_not_counted(self):
        """Architect-scoped, like every other count on this surface."""
        from orchestrator.evals.report import build_plan_quality_report

        impl = _implementer_result(task_id='t2', config_name='opus-high')
        impl.metrics['judged_without_reference'] = True

        report = build_plan_quality_report([
            _architect_result(task_id='t1', config_name='a'), impl,
        ])
        assert report['judged_without_reference'] == 0

    def test_exclusions_are_broken_out_by_cause(self):
        # The causes are NOT interchangeable: a cap hit is transient and
        # schedule-attributable ('rerun after the window'), a model-not-found is
        # a PERMANENT candidate-config error ('this can never run'). A single
        # total would let the latter masquerade as the former and hide a dead
        # config behind n=0 / mean=None (reviewer: design-coherence).
        from orchestrator.evals.report import build_plan_quality_report

        report = build_plan_quality_report([
            _cap_tainted_result(task_id='t1', config_name='a'),
            _cap_tainted_result(task_id='t2', config_name='a'),
            _cap_tainted_result(
                task_id='t3', config_name='b',
                invocation_error='architect:model_not_found: no such model',
            ),
            # An end-to-end pin (reviewer: test-coverage) that a server_error
            # marker survives the FULL report pipeline — not just
            # detect_invocation_error's unit-level token — and lands in its
            # own cap_excluded_by_cause bucket rather than being folded into
            # 'unknown' by a marker-shape change.
            _cap_tainted_result(
                task_id='t4', config_name='b',
                invocation_error='architect:server_error: HTTP 529',
            ),
        ])
        assert report['cap_excluded'] == 4
        assert report['cap_excluded_by_cause'] == {
            'cap_hit': 2, 'model_not_found': 1, 'server_error': 1,
        }
        # Key-sorted, so the dict renders byte-deterministically.
        causes = list(report['cap_excluded_by_cause'])
        assert causes == sorted(causes)

    def test_unparseable_marker_buckets_as_unknown_not_a_real_cause(self):
        # A mis-shaped marker must show up as its own bucket rather than being
        # silently folded into a real cause.
        from orchestrator.evals.report import build_plan_quality_report

        report = build_plan_quality_report([
            _cap_tainted_result(task_id='t1', config_name='a',
                                invocation_error='architect:'),
        ])
        assert report['cap_excluded_by_cause'] == {'unknown': 1}

    def test_all_cells_tainted_yields_null_mean_never_zero(self):
        from orchestrator.evals.report import build_plan_quality_report

        report = build_plan_quality_report([
            _cap_tainted_result(task_id='t1', config_name='doomed'),
            _cap_tainted_result(task_id='t2', config_name='doomed'),
        ])
        agg = {c['config_name']: c for c in report['configs']}['doomed']
        assert agg['mean_plan_quality'] is None
        assert agg['mean_plan_quality'] != 0.0
        assert agg['n'] == 0
        assert agg['cap_excluded'] == 2

    def test_implementer_rows_stay_out_of_the_architect_aggregate(self):
        from orchestrator.evals.report import build_plan_quality_report

        report = build_plan_quality_report(
            [_architect_result(), _implementer_result()]
        )
        assert [c['config_name'] for c in report['configs']] == [
            'architect-sonnet-high',
        ]
        # ...while both still appear as ROWS, sorted deterministically.
        keys = [(r['task_id'], r['config_name']) for r in report['rows']]
        assert keys == sorted(keys)
        assert len(keys) == 2

    def test_configs_sorted_by_config_name(self):
        from orchestrator.evals.report import build_plan_quality_report

        report = build_plan_quality_report([
            _architect_result(task_id='t1', config_name='z-cfg'),
            _architect_result(task_id='t2', config_name='a-cfg'),
        ])
        names = [c['config_name'] for c in report['configs']]
        assert names == sorted(names)

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

    # -- rendered exclusion surface (task 3118) ---------------------------

    def test_cap_tainted_cell_renders_excluded_marker_not_a_zero(self):
        from orchestrator.evals.report import (
            build_plan_quality_report,
            format_plan_quality_table,
        )

        table = format_plan_quality_table(build_plan_quality_report(
            [_architect_result(), _cap_tainted_result(), _implementer_result()]
        ))
        capped_line = next(
            ln for ln in table.splitlines() if 'df_task_3118' in ln
        )
        # Explicit exclusion marker, WITH its reason — never a score...
        assert 'excluded' in capped_line
        assert '0.0000' not in capped_line
        assert 'cap_hit' in capped_line
        # ...and visibly distinct from the '-' null sentinel a non-architect row
        # uses, so "not an architect run" cannot be confused with "architect run
        # we could not measure".
        impl_line = next(ln for ln in table.splitlines() if 'df_task_1993' in ln)
        assert 'excluded' not in impl_line

    def test_table_renders_exclusion_summary_and_per_config_means(self):
        from orchestrator.evals.report import (
            build_plan_quality_report,
            format_plan_quality_table,
        )

        cfg = 'architect-opus-high'
        table = format_plan_quality_table(build_plan_quality_report([
            _architect_result(task_id='t1', config_name=cfg, plan_quality=0.9),
            _architect_result(task_id='t2', config_name=cfg, plan_quality=0.7),
            _cap_tainted_result(task_id='t3', config_name=cfg),
        ]))

        # The reader sees "n=2 of 3, 1 excluded", not a silently shrunk mean.
        summary = next(
            ln for ln in table.splitlines() if ln.startswith('excluded:')
        )
        assert '1' in summary and '3' in summary
        # ...with the CAUSE named, not a bare count.
        assert 'cap_hit' in summary

        mean_line = _mean_section_line(table, cfg)
        assert '0.8000' in mean_line   # the EXCLUDING mean, not 0.5333
        assert '0.5333' not in table

    def test_summary_names_a_permanent_cause_as_itself(self):
        # A model-not-found exclusion must not render as a cap window: the
        # operator's next action differs entirely ('fix the config' vs 'rerun').
        from orchestrator.evals.report import (
            build_plan_quality_report,
            format_plan_quality_table,
        )

        table = format_plan_quality_table(build_plan_quality_report([
            _cap_tainted_result(
                task_id='t1', config_name='dead-cfg',
                invocation_error='architect:model_not_found: no such model',
            ),
        ]))
        summary = next(
            ln for ln in table.splitlines() if ln.startswith('excluded:')
        )
        assert 'model_not_found: 1' in summary
        assert 'cap' not in summary.lower()

    def test_config_with_no_scored_cells_renders_dash_not_zero(self):
        from orchestrator.evals.report import (
            build_plan_quality_report,
            format_plan_quality_table,
        )

        table = format_plan_quality_table(build_plan_quality_report([
            _cap_tainted_result(task_id='t1', config_name='doomed'),
            _cap_tainted_result(task_id='t2', config_name='doomed'),
        ]))
        mean_line = _mean_section_line(table, 'doomed')
        assert mean_line.endswith('-')
        assert '0.0000' not in mean_line

    def test_exclusion_table_is_byte_identical_regardless_of_input_order(self):
        from orchestrator.evals.report import (
            build_plan_quality_report,
            format_plan_quality_table,
        )

        results = [
            _architect_result(task_id='t1', config_name='a', plan_quality=0.9),
            _cap_tainted_result(task_id='t2', config_name='a'),
            _architect_result(task_id='t3', config_name='b', plan_quality=0.5),
            _implementer_result(),
        ]
        a = format_plan_quality_table(build_plan_quality_report(results))
        b = format_plan_quality_table(
            build_plan_quality_report(list(reversed(results)))
        )
        assert a == b


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


def _dispatch_single_eval(cfg, capsys, arch_metrics=None):
    """Drive ``cli._run_single_eval`` with a resolved config, both runners patched.

    ``get_config_by_name`` is patched to return ``cfg`` so the dispatch is
    exercised purely on ``cfg.role``. ``arch_metrics`` overrides the fake
    architect result's metrics dict. Returns ``(out, run_eval, run_architect)``.
    """
    from orchestrator import cli

    arch_result = _fake_eval_result(
        config_name=cfg.name,
        metrics=arch_metrics or {
            'role_under_test': 'architect', 'plan_quality': 0.75,
        },
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

    def test_cap_tainted_cell_echo_names_the_taint(self, capsys):
        # An operator watching the run must see the infra refusal LIVE, not a
        # bare `plan_quality=None` that reads like a scoring quirk.
        marker = f'architect:cap_hit: {_CAP_TEXT}'
        out, _, _ = _dispatch_single_eval(
            self._arch_cfg(), capsys,
            arch_metrics={
                'role_under_test': 'architect',
                'plan_quality': None,
                'cap_tainted': True,
                'invocation_error': marker,
            },
        )
        # Scoped to the PER-CELL echo line ('plan_quality=' with the '='), not
        # the whole capture: the plan-quality table printed after the loop
        # already names the exclusion, and asserting over `out` would pass on
        # that alone while the live per-cell line still read a bare None.
        echo_line = next(
            ln for ln in out.splitlines() if 'plan_quality=' in ln
        )
        # Cause-neutral label + the marker naming the ACTUAL cause: a permanent
        # config error must not read as a transient cap window.
        assert 'unmeasurable' in echo_line
        assert marker in echo_line

    def test_model_not_found_echo_does_not_read_as_a_cap_window(self, capsys):
        # The same taint flag covers a PERMANENT candidate-configuration error.
        # An operator must not be told to "rerun after the cap resets" for a
        # model that does not exist (reviewer: design-coherence).
        marker = 'architect:model_not_found: no such model gpt-9'
        out, _, _ = _dispatch_single_eval(
            self._arch_cfg(), capsys,
            arch_metrics={
                'role_under_test': 'architect',
                'plan_quality': None,
                'cap_tainted': True,
                'invocation_error': marker,
            },
        )
        echo_line = next(
            ln for ln in out.splitlines() if 'plan_quality=' in ln
        )
        assert 'model_not_found' in echo_line
        assert 'cap' not in echo_line.lower()

    def test_implementer_config_still_routes_to_run_eval(self, capsys):
        _, run_eval, run_arch = _dispatch_single_eval(self._impl_cfg(), capsys)
        run_eval.assert_called_once()
        run_arch.assert_not_called()
