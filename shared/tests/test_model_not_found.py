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

from shared.cli_invoke import AgentResult
from shared.invocation_outcome import (
    OK,
    CapHit,
    Failure,
    ModelNotFound,
    classify_invocation,
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
