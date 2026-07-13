"""Task beta: model-not-found terminal classification (routing PRD,
plans/adaptive-model-routing-prd.md) -- shared.invocation_outcome +
shared.cli_invoke.

Fixtures follow the same AgentResult-fixture + classify_invocation(
strict_confirm=True) convention as test_invocation_outcome.py's
TestClassifyInvocation* suites. Kept in its own module-local file (not
appended to test_invocation_outcome.py / test_cli_invoke.py, and no
conftest.py edit) -- mirrors orchestrator/tests/test_routing.py's stated
no-conftest-edit rationale (verify.py's has_conftest would otherwise force a
full owning-package suite fallback at merge-verify time).
"""

from __future__ import annotations

from shared.cli_invoke import AgentFailureKind, AgentResult, classify_agent_failure
from shared.invocation_outcome import (
    OK,
    CapHit,
    Failure,
    ModelNotFound,
    classify_invocation,
)

NOT_FOUND_ERROR_BODY = (
    '{"type":"error","error":{"type":"not_found_error",'
    '"message":"model: bogus-model-9"}}'
)


class TestClassifyInvocationModelNotFound:
    """A model-not-found API error is zero-cost / near-instant / <=1 turn and
    must classify as the TERMINAL ModelNotFound variant rather than fall
    through to the unclassified Failure the retry loop's heuristic cap
    safety-net then mislabels as a CapHit -- the root cause of the
    'TRANSIENT -> whole-pool churn' bug this task fixes."""

    def test_404_status_with_not_found_error_body_is_model_not_found(self):
        result = AgentResult(
            success=False,
            api_error_status=404,
            output=(
                '{"type":"error","error":{"type":"not_found_error",'
                '"message":"model: bogus-model-9"}}'
            ),
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, ModelNotFound)

    def test_marker_in_output_with_no_status_is_model_not_found(self):
        """A body-only variant -- no structured api_error_status at all --
        must still be caught by the MODEL_NOT_FOUND_MARKERS substring scan."""
        result = AgentResult(
            success=False,
            output='Error: model not found: bogus-model-9',
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, ModelNotFound)

    def test_successful_run_quoting_marker_stays_ok(self):
        """A successful invocation that merely quotes 'not_found_error' in its
        own output (e.g. an agent discussing an API error format) must stay
        OK -- the marker scan sits BELOW the `if result.success`
        short-circuit, mirroring the existing CliLocalError/CapHit
        false-positive guard (test_success_true_outranks_incidental_cli_marker_text
        in test_invocation_outcome.py)."""
        result = AgentResult(
            success=True,
            output='The API returned a not_found_error for the old model id, as expected.',
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert outcome == OK()
        assert not isinstance(outcome, ModelNotFound)

    def test_429_cap_body_is_cap_hit_not_model_not_found(self):
        result = AgentResult(
            success=False,
            output="You're out of extra usage for this billing period. Your plan resets in 2h.",
            api_error_status=429,
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, CapHit)
        assert not isinstance(outcome, ModelNotFound)

    def test_generic_500_with_unrelated_text_is_failure_not_model_not_found(self):
        result = AgentResult(
            success=False,
            output='internal server error, please retry',
            api_error_status=500,
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, Failure)
        assert not isinstance(outcome, ModelNotFound)


class TestClassifyAgentFailureModelNotFound:
    """classify_agent_failure must map a model-not-found result to the
    distinct TERMINAL AgentFailureKind.MODEL_NOT_FOUND, NOT the transient
    API_ERROR kind -- a 404 sets api_error_status and would otherwise be
    mis-tagged transient by the existing api_error_status->API_ERROR rule
    (cli_invoke.py's decision-rule #4)."""

    def test_model_not_found_is_agent_failure_kind_member(self):
        assert AgentFailureKind.MODEL_NOT_FOUND.value == 'model_not_found'

    def test_404_not_found_error_maps_to_model_not_found_kind_not_api_error(self):
        result = AgentResult(success=False, api_error_status=404, output=NOT_FOUND_ERROR_BODY)
        classified = classify_agent_failure(result)
        assert classified.kind == AgentFailureKind.MODEL_NOT_FOUND
        assert classified.kind != AgentFailureKind.API_ERROR

    def test_summary_conveys_terminal_no_retry_semantics(self):
        result = AgentResult(success=False, api_error_status=404, output=NOT_FOUND_ERROR_BODY)
        classified = classify_agent_failure(result)
        summary_lower = classified.summary.lower()
        assert 'terminal' in summary_lower or 'no retry' in summary_lower or 'no-retry' in summary_lower

    def test_genuine_transient_500_still_maps_to_api_error(self):
        """Regression guard: the new branch must not swallow real transient
        API errors -- only a model-not-found outcome should divert away from
        the API_ERROR rule."""
        result = AgentResult(
            success=False, api_error_status=500, output='internal server error, please retry'
        )
        classified = classify_agent_failure(result)
        assert classified.kind == AgentFailureKind.API_ERROR
