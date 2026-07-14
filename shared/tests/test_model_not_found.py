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

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.cli_invoke import (
    AgentFailureKind,
    AgentResult,
    AllAccountsCappedException,
    classify_agent_failure,
    invoke_with_cap_retry,
)
from shared.config_models import AccountConfig, UsageCapConfig
from shared.invocation_outcome import (
    OK,
    CapHit,
    Failure,
    ModelNotFound,
    classify_invocation,
)
from shared.usage_gate import AccountPhase, UsageGate

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


# ---------------------------------------------------------------------------
# Terminal no-churn drive: invoke_with_cap_retry must treat a model-not-found
# result as TERMINAL -- no cross-account failover, no account transitioned to
# capped/auth_failed, and AllAccountsCappedException never raised. Driven
# through a REAL multi-account UsageGate (never a mock) with a scripted fake
# invoke_fn, mirroring test_invocation_outcome_boundary.py's boundary-harness
# style (make_boundary_gate), replicated module-locally per this task's
# design decision (no cross-test-module import, no conftest.py edit).
# ---------------------------------------------------------------------------


def _make_gate(names: list[str]) -> UsageGate:
    """Build a REAL, minimal multi-account UsageGate for the drive below.

    Mirrors test_invocation_outcome_boundary.py's make_boundary_gate: fake
    per-account env tokens so ``UsageGate._init_accounts`` resolves real
    ``AccountState`` objects, ``gate._run_probe`` AsyncMock'd so no real
    ``claude`` subprocess is ever spawned, and ``_start_account_resume_probe``
    / ``_start_auth_reprobe`` neutralized (instance-level no-op
    ``MagicMock``s) so entering CAPPED/AUTH_FAILED via ``_transition`` never
    schedules a real background asyncio task -- every transition stays
    synchronous and deterministic (load-bearing here: with both pool
    accounts capped and no background resume task ever spawned,
    ``before_invoke`` would otherwise block forever waiting for the gate to
    reopen).
    """
    acct_cfgs = []
    env_vars: dict[str, str] = {}
    for name in names:
        env_key = f'TEST_TOKEN_{name.upper().replace("-", "_")}'
        env_vars[env_key] = f'fake-token-{name}'
        acct_cfgs.append(AccountConfig(name=name, oauth_token_env=env_key))

    config = UsageCapConfig(accounts=acct_cfgs)
    with patch.dict(os.environ, env_vars):
        gate = UsageGate(config)

    gate._run_probe = AsyncMock(return_value=True)
    gate._start_account_resume_probe = MagicMock()
    gate._start_auth_reprobe = MagicMock()
    return gate


class _CountingModelNotFoundCli:
    """Scripted ``invoke_fn`` that always returns the SAME model-not-found
    ``AgentResult``, counting its calls.

    Unlike test_invocation_outcome_boundary.py's ``_ScriptedCli`` (which
    raises once its scripted results are exhausted), this fake must
    tolerate being called more than once: under the CURRENT (pre-fix)
    implementation the heuristic cap net misclassifies this zero-cost,
    instant, model-not-found result as a cap hit and fails over to every
    account in the pool before ``max_cap_retries`` bounds the loop --
    exactly the churn this RED test observes (and that the GREEN fix in
    step-10 eliminates).
    """

    def __init__(self, result: AgentResult) -> None:
        self._result = result
        self.calls = 0

    async def __call__(self, **kwargs: object) -> AgentResult:
        self.calls += 1
        return self._result


@pytest.mark.asyncio
class TestInvokeWithCapRetryModelNotFoundTerminalNoChurn:
    """The user-observable anti-churn signal (task beta's raison d'etre): a
    model-not-found result must be TERMINAL through the real
    invoke_with_cap_retry loop -- no per-account failover, no account
    transitioned to capped/auth_failed, and AllAccountsCappedException never
    raised.

    Currently FAILS: the heuristic cap net (cli_invoke.py's zero-cost /
    near-instant / <=1-turn safety net, ~line 1090) misclassifies this
    zero-cost, instant, model-not-found result as a cap hit, so BOTH pool
    accounts get marked CAPPED and invoke_fn is called twice before
    max_cap_retries=2 raises AllAccountsCappedException -- the exact
    'TRANSIENT -> whole-pool churn' bug this task fixes.
    """

    async def test_model_not_found_is_terminal_with_no_failover_or_cap_transition(self):
        gate = _make_gate(['acct-0', 'acct-1'])
        result = AgentResult(
            success=False,
            api_error_status=404,
            output=NOT_FOUND_ERROR_BODY,
        )
        cli = _CountingModelNotFoundCli(result)

        with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
            try:
                returned = await invoke_with_cap_retry(
                    gate, 'model-not-found[terminal]',
                    invoke_fn=cli,
                    max_cap_retries=2,
                    backend='claude',
                    prompt='hi',
                )
            except AllAccountsCappedException as exc:
                pytest.fail(
                    f'invoke_with_cap_retry raised AllAccountsCappedException '
                    f'after {cli.calls} call(s) for a terminal model-not-found '
                    f'result -- expected it to be returned directly with no '
                    f'cross-account failover: {exc}'
                )

        assert cli.calls == 1, (
            f'expected exactly one invocation (terminal, no cross-account '
            f'failover), the retry loop made {cli.calls}'
        )
        assert returned.success is False
        assert returned is result

        for acct in gate._accounts:
            assert acct.phase not in (AccountPhase.CAPPED, AccountPhase.AUTH_FAILED), (
                f'expected no account marked capped/auth_failed after a '
                f'terminal model-not-found result, got {acct.name}={acct.phase}'
            )
