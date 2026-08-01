"""Task alpha: server-side (HTTP 5xx) API-error classification
(plans/server-side-api-error-handling-prd.md, contract C1) --
shared.cli_invoke + shared.invocation_outcome.

Covers PRD boundary rows 1 (classification half: a watchdog-killed CLI whose
SIGTERM-flushed result JSON carries ``api_error_status=529`` must classify as a
server error, not a zero-output wedge / TIMED_OUT), 9 (a fast zero-cost 529 must
not trip ``invoke_with_cap_retry``'s heuristic cap safety-net and mark a healthy
account CAPPED) and 10 (the 429/cap-body and 401 failover paths must not move).

Fixtures follow the same AgentResult-fixture + ``classify_invocation(
strict_confirm=True)`` convention as test_invocation_outcome.py's
TestClassifyInvocation* suites. Kept in its own module-local file (not appended
to test_invocation_outcome.py / test_cli_invoke.py, and no conftest.py edit) --
mirrors the rationale stated at the top of test_model_not_found.py: verify.py's
``has_conftest`` would otherwise force a full owning-package suite fallback at
merge-verify time.
"""

from __future__ import annotations

import dataclasses
import os
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.cli_invoke import (
    AgentFailureKind,
    AgentResult,
    AllAccountsCappedException,
    classify_agent_failure,
    invoke_with_cap_retry,
    is_server_error_status,
    is_zero_output_timeout,
)
from shared.config_models import AccountConfig, UsageCapConfig
from shared.invocation_outcome import (
    OK,
    AuthFailed,
    CapHit,
    CliLocalError,
    Failure,
    ModelNotFound,
    ServerError,
    ZeroOutputWedge,
    classify_invocation,
)
from shared.usage_gate import AccountPhase, UsageGate

# The 529 "Overloaded" body the provider returns during a server-side outage.
OVERLOADED_BODY = (
    '{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}'
)

# A real cap body: matches CAP_HIT_PREFIXES ("You're out of extra") AND a
# CAP_CONFIRM_KEYWORD ("resets"), so it is recognised under strict_confirm=True.
CAP_BODY = "You're out of extra usage · resets in 2h"


def _incident_result(status: int = 529) -> AgentResult:
    """PRD boundary row 1: the 2026-07-29 shape — the watchdog SIGTERM-killed a
    wedged CLI, and the CLI flushed its result JSON on the way out with
    ``api_error_status`` set.  ``timed_out`` and a 5xx status are BOTH true, so
    this fixture is exactly where the 5xx-vs-timed_out precedence is decided.
    """
    return AgentResult(
        success=False,
        output='',
        subtype='error_empty_output',
        timed_out=True,
        transcript_turns=0,
        duration_ms=127000,
        api_error_status=status,
    )


class TestIsServerErrorStatus:
    """``is_server_error_status`` is the single canonical 5xx predicate (INV-5):
    the ``ServerError`` classification tier, ``classify_agent_failure``'s 5xx
    rule and -- via ``shared``'s re-export -- the orchestrator consumers landing
    in PRD tasks beta/gamma/delta all call THIS function, never an inline
    ``500 <= n <= 599``.
    """

    @pytest.mark.parametrize('status', [500, 501, 502, 503, 529, 598, 599])
    def test_5xx_statuses_are_server_errors(self, status: int):
        assert is_server_error_status(status) is True

    @pytest.mark.parametrize(
        'status',
        [
            None,  # no structured status was reported -- never a server error
            0,
            200,
            401,  # AuthFailed keeps its own routing
            403,
            404,  # ModelNotFound keeps its own routing
            429,  # the cap carve-out must stay outside the 5xx band
            499,  # one below the floor
            600,  # one above the ceiling
            700,
            -1,
        ],
    )
    def test_non_5xx_statuses_are_not_server_errors(self, status: int | None):
        assert is_server_error_status(status) is False


class TestServerErrorStatusPublicSurface:
    """The predicate must be reachable as ``from shared import
    is_server_error_status`` -- that re-export is what makes it a single source
    for the orchestrator consumers in PRD tasks beta/gamma/delta.

    Membership in ``cli_invoke.__all__`` / ``shared.__all__`` is NOT asserted
    here: test_public_api.py already pins both as exact sets (and derives
    ``shared.__all__`` as the union of the module ``__all__``s).  What that
    suite does not assert -- and what this one does -- is the IDENTITY of the
    re-exported object.
    """

    def test_reexported_from_shared_package(self):
        from shared import is_server_error_status as reexported

        assert reexported is is_server_error_status


# The ``ServerError`` variant contract itself (InvocationOutcome membership,
# status round-trip, frozen-ness) lives in test_invocation_outcome.py alongside
# every other variant's -- that module is the single home for sum-type variant
# contracts, so it is not restated here.


class TestClassifyInvocationServerError:
    """``classify_invocation`` gains a ServerError tier ranked BELOW
    CapHit/NearCap (a 5xx body never carries cap prefixes, and 429-body
    semantics must not move) and ABOVE ZeroOutputWedge (a SIGTERM-flushed 529
    on a watchdog-killed CLI is a provider outage, not a local wedge).
    """

    def test_fast_zero_cost_529_is_server_error(self):
        result = AgentResult(
            success=False,
            output=OVERLOADED_BODY,
            api_error_status=529,
            cost_usd=0.0,
            turns=0,
            duration_ms=1200,
        )
        assert classify_invocation(result, strict_confirm=True) == ServerError(status=529)

    @pytest.mark.parametrize('status', [500, 599])
    def test_boundary_statuses_are_server_errors(self, status: int):
        result = AgentResult(success=False, output='Internal error', api_error_status=status)
        assert classify_invocation(result, strict_confirm=True) == ServerError(status=status)

    def test_503_with_prose_body_is_server_error(self):
        result = AgentResult(
            success=False,
            output='Service Unavailable: upstream is temporarily unreachable',
            api_error_status=503,
        )
        assert classify_invocation(result, strict_confirm=True) == ServerError(status=503)

    def test_incident_shape_is_server_error_not_wedge(self):
        """PRD boundary row 1 — the regression this task exists to fix."""
        outcome = classify_invocation(_incident_result(), strict_confirm=True)
        assert outcome == ServerError(status=529)
        assert not isinstance(outcome, ZeroOutputWedge)

    def test_decision_2_lock_shape_predicate_stays_cause_blind(self):
        """PRD decision 2: ``is_zero_output_timeout`` must stay a pure SHAPE
        predicate ("did this run produce anything") — its other consumers (the
        workflow zero-output circuit breaker, the resume-wedge guard) want a
        cause-blind answer.  The cause-awareness lives in the classifier
        instead.  Both halves are asserted on the SAME fixture so a future edit
        cannot quietly make the predicate cause-aware.

        RESIDUAL GAP this lock deliberately leaves open: keeping the predicate
        cause-blind means workflow.py's zero-output hang circuit breaker (which
        keys on it) still counts a 5xx-caused timeout toward
        ``consecutive_zero_output`` and can block the task as an infra_issue --
        the misfiling this PRD exists to eliminate.  This task closes it at the
        classifier and cap-retry layers only; making that workflow consumer
        cause-aware is PRD task gamma's job.
        """
        result = _incident_result()
        assert is_zero_output_timeout(result) is True
        assert isinstance(classify_invocation(result, strict_confirm=True), ServerError)


class TestClassifyInvocationServerErrorPrecedence:
    """The neighbouring tiers must not move."""

    def test_success_short_circuits_above_server_error(self):
        result = AgentResult(success=True, output='done', api_error_status=529)
        assert isinstance(classify_invocation(result, strict_confirm=True), OK)

    @pytest.mark.parametrize('status', [401, 403])
    def test_auth_failed_outranks_server_error(self, status: int):
        result = AgentResult(success=False, output='Unauthorized', api_error_status=status)
        assert classify_invocation(result, strict_confirm=True) == AuthFailed(status=status)

    def test_404_stays_model_not_found(self):
        result = AgentResult(
            success=False,
            output='{"type":"error","error":{"type":"not_found_error"}}',
            api_error_status=404,
        )
        assert isinstance(classify_invocation(result, strict_confirm=True), ModelNotFound)

    def test_429_with_cap_body_stays_cap_hit(self):
        result = AgentResult(success=False, output=CAP_BODY, api_error_status=429)
        assert isinstance(classify_invocation(result, strict_confirm=True), CapHit)

    def test_499_is_below_the_5xx_floor(self):
        result = AgentResult(success=False, output='client closed request', api_error_status=499)
        assert isinstance(classify_invocation(result, strict_confirm=True), Failure)

    def test_cap_body_outranks_server_error(self):
        """A 5xx status carrying a REAL cap body still classifies as CapHit —
        the ServerError tier sits below cap detection precisely so cap
        accounting is untouched."""
        result = AgentResult(success=False, output=CAP_BODY, api_error_status=503)
        assert isinstance(classify_invocation(result, strict_confirm=True), CapHit)

    def test_cli_local_error_outranks_server_error(self):
        result = AgentResult(
            success=False,
            output='',
            stderr='Session id abc is already in use',
            api_error_status=503,
        )
        assert isinstance(classify_invocation(result, strict_confirm=True), CliLocalError)

    def test_model_not_found_marker_outranks_server_error(self):
        result = AgentResult(
            success=False,
            output='{"type":"error","error":{"type":"not_found_error"}}',
            api_error_status=503,
        )
        assert isinstance(classify_invocation(result, strict_confirm=True), ModelNotFound)


# shared/ must not import orchestrator/ (layering doctrine), so this is a
# byte-identical LOCAL copy of orchestrator/src/orchestrator/scheduler.py's
# ``_API_ERROR_REASON_RE``.  The scheduler's transient-requeue lane keys on this
# marker to read the status out of block_reason, so the ``agent API error: HTTP
# <status>`` prefix is a cross-module contract: this test pins the PRODUCING
# side of it, and PRD task beta pins the consuming side.
_API_ERROR_REASON_RE = re.compile(r'agent API error: HTTP (\d{3})')


class TestClassifyAgentFailureServerError:
    """PRD boundary row 1 (classification half): ``classify_agent_failure``
    gains a 5xx rule ranked ABOVE ``timed_out``.

    A watchdog SIGTERM kill flushes the CLI's result JSON with
    ``api_error_status`` set (2026-07-29 incident), so ranking ``timed_out``
    first discarded the 5xx evidence and misfiled a provider outage as a
    zero-output wedge — and the marker the scheduler's transient requeue lane
    keys on was never produced.
    """

    def test_incident_shape_is_api_error_not_timed_out(self):
        classified = classify_agent_failure(_incident_result())
        assert classified.kind is AgentFailureKind.API_ERROR
        assert classified.kind is not AgentFailureKind.TIMED_OUT

    def test_incident_shape_summary_carries_the_verbatim_marker(self):
        summary = classify_agent_failure(_incident_result()).summary
        assert summary.startswith('agent API error: HTTP 529')
        match = _API_ERROR_REASON_RE.search(summary)
        assert match is not None, f'scheduler marker regex did not match {summary!r}'
        assert match.group(1) == '529'

    def test_incident_shape_summary_appends_kill_context(self):
        """The prefix is a contract; the suffix is free-form operator forensics
        (PRD open question 5) and is emitted only on the timed-out path."""
        summary = classify_agent_failure(_incident_result()).summary
        assert 'killed' in summary
        assert '127' in summary, f'expected the elapsed seconds in {summary!r}'
        assert 'transcript_turns=0' in summary

    def test_incident_shape_diagnostic_detail_unchanged(self):
        detail = classify_agent_failure(_incident_result()).diagnostic_detail
        assert 'api_error_status=529' in detail
        assert 'transcript_turns=0' in detail

    def test_non_timed_out_529_summary_has_no_kill_context(self):
        """Pins that the suffix is conditional on ``timed_out`` — the
        non-timed-out summary stays byte-identical to today's."""
        result = AgentResult(success=False, output=OVERLOADED_BODY, api_error_status=529)
        classified = classify_agent_failure(result)
        assert classified.kind is AgentFailureKind.API_ERROR
        assert classified.summary == 'agent API error: HTTP 529'

    @pytest.mark.parametrize('status', [500, 599])
    def test_timed_out_5xx_boundaries_are_api_error(self, status: int):
        classified = classify_agent_failure(_incident_result(status))
        assert classified.kind is AgentFailureKind.API_ERROR
        match = _API_ERROR_REASON_RE.search(classified.summary)
        assert match is not None
        assert match.group(1) == str(status)

    def test_timed_out_499_stays_timed_out(self):
        """Below the 5xx floor — unchanged."""
        assert classify_agent_failure(_incident_result(499)).kind is AgentFailureKind.TIMED_OUT


class TestClassifyAgentFailureServerErrorPrecedence:
    """Every neighbouring rule must stay byte-identical."""

    def test_sigkill_subpath_with_no_flushed_json_stays_a_wedge(self):
        """PRD boundary row 2: the SIGKILL sub-path flushes no JSON, so there is
        no ``api_error_status`` to key on and the wedge phrasing is unchanged."""
        result = AgentResult(
            success=False,
            output='',
            subtype='error_empty_output',
            timed_out=True,
            transcript_turns=0,
            duration_ms=127000,
            api_error_status=None,
        )
        classified = classify_agent_failure(result)
        assert classified.kind is AgentFailureKind.TIMED_OUT
        assert 'wedge' in classified.summary
        assert 'productive' not in classified.summary

    def test_timed_out_with_progress_and_no_status_is_unchanged(self):
        result = AgentResult(
            success=False,
            output='',
            timed_out=True,
            transcript_turns=13,
            duration_ms=127000,
        )
        classified = classify_agent_failure(result)
        assert classified.kind is AgentFailureKind.TIMED_OUT
        assert 'productive' in classified.summary

    def test_productive_kill_with_a_5xx_stays_timed_out(self):
        """A PRODUCTIVE wall-clock kill (transcript_turns > 0) that happens to
        carry a 5xx keeps its TIMED_OUT kind and its truthful productive
        phrasing.  The 5xx rule exists for the pre-first-token outage shape;
        claiming a run that did 13 turns of real work was 'killed
        pre-first-token' is exactly the untruthful phrasing task 2360 fix #3 /
        reify-4827 removed, and re-kinding it would also drop it out of
        dry_run_unblock's infra-failure kinds and into the scheduler's
        transient-requeue lane.  The 5xx evidence survives in
        ``diagnostic_detail``.
        """
        result = AgentResult(
            success=False,
            output='',
            timed_out=True,
            transcript_turns=13,
            duration_ms=127000,
            api_error_status=503,
        )
        classified = classify_agent_failure(result)
        assert classified.kind is AgentFailureKind.TIMED_OUT
        assert 'productive' in classified.summary
        assert 'pre-first-token' not in classified.summary
        assert _API_ERROR_REASON_RE.search(classified.summary) is None, (
            'a productive kill must not emit the transient-requeue marker'
        )
        assert 'api_error_status=503' in classified.diagnostic_detail

    def test_timed_out_5xx_with_unknown_transcript_turns_claims_neither(self):
        """``transcript_turns=None`` means the transcript was never read — not
        evidence of progress, and equally not evidence the kill landed before
        the first token.  The rule still fires (the count is not positive), but
        the suffix must not assert 'pre-first-token'."""
        result = AgentResult(
            success=False,
            output='',
            timed_out=True,
            transcript_turns=None,
            duration_ms=127000,
            api_error_status=529,
        )
        classified = classify_agent_failure(result)
        assert classified.kind is AgentFailureKind.API_ERROR
        assert classified.summary.startswith('agent API error: HTTP 529')
        assert 'pre-first-token' not in classified.summary
        assert 'transcript_turns=unknown' in classified.summary

    def test_error_max_turns_with_a_5xx_stays_max_turns(self):
        """Saturation detection (workflow's ``_stamp_simple_saturated`` keys on
        MAX_TURNS) must survive an incidental 5xx: the 5xx rule outranks the
        TIMEOUT rule only, not the max_turns rule below it."""
        result = AgentResult(
            success=False,
            output='',
            subtype='error_max_turns',
            turns=200,
            output_tokens=4096,
            api_error_status=503,
        )
        assert classify_agent_failure(result).kind is AgentFailureKind.MAX_TURNS

    def test_model_not_found_marker_outranks_the_5xx_rule(self):
        """INV-5: the two classifiers must agree.  ``classify_invocation``
        ranks ModelNotFound ABOVE ServerError (mirrored by
        TestClassifyInvocationServerErrorPrecedence.
        test_model_not_found_marker_outranks_server_error), and
        ``invoke_with_cap_retry`` treats ModelNotFound as TERMINAL — so if this
        rule claimed the result, the scheduler would read the transient marker
        out of block_reason and requeue a run the retry loop already gave up
        on.
        """
        result = AgentResult(
            success=False,
            output='{"type":"error","error":{"type":"not_found_error"}}',
            api_error_status=503,
        )
        classified = classify_agent_failure(result)
        assert classified.kind is AgentFailureKind.MODEL_NOT_FOUND
        assert _API_ERROR_REASON_RE.search(classified.summary) is None, (
            'a terminal ModelNotFound must not emit the transient-requeue marker'
        )
        # Both classifiers agree on this shape.
        assert isinstance(classify_invocation(result, strict_confirm=True), ModelNotFound)

    def test_ended_awaiting_background_outranks_the_5xx_rule(self):
        result = AgentResult(
            success=False,
            output='',
            ended_awaiting_background=True,
            api_error_status=529,
        )
        classified = classify_agent_failure(result)
        assert classified.kind is AgentFailureKind.ENDED_AWAITING_BACKGROUND

    @pytest.mark.parametrize('status', [401, 403])
    def test_auth_statuses_stay_api_error(self, status: int):
        result = AgentResult(success=False, output='Unauthorized', api_error_status=status)
        assert classify_agent_failure(result).kind is AgentFailureKind.API_ERROR

    def test_404_stays_model_not_found(self):
        result = AgentResult(
            success=False,
            output='{"type":"error","error":{"type":"not_found_error"}}',
            api_error_status=404,
        )
        assert classify_agent_failure(result).kind is AgentFailureKind.MODEL_NOT_FOUND

    def test_success_with_5xx_status_stays_success(self):
        result = AgentResult(success=True, output='done', api_error_status=529)
        assert classify_agent_failure(result).kind is AgentFailureKind.SUCCESS


# ---------------------------------------------------------------------------
# invoke_with_cap_retry drives (PRD boundary rows 9, 1 cap-retry half, 10).
#
# Every drive runs a REAL multi-account UsageGate (never a mock) with a
# scripted fake invoke_fn, so "account NOT capped, no failover" is observed
# through genuine AccountPhase transitions rather than asserted against a mock.
# Replicated module-locally from test_model_not_found.py:153-206 per this
# task's design decision (no cross-test-module import, no conftest.py edit).
# ---------------------------------------------------------------------------


def _make_gate(names: list[str]) -> UsageGate:
    """Build a REAL, minimal multi-account UsageGate for the drives below.

    Fake per-account env tokens so ``UsageGate._init_accounts`` resolves real
    ``AccountState`` objects; ``gate._run_probe`` AsyncMock'd so no real
    ``claude`` subprocess is ever spawned; ``_start_account_resume_probe`` /
    ``_start_auth_reprobe`` neutralized (instance-level no-op ``MagicMock``s)
    so entering CAPPED/AUTH_FAILED via ``_transition`` never schedules a real
    background asyncio task — every transition stays synchronous and
    deterministic, and ``before_invoke`` can never block waiting for a resume
    probe that will never run.
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


class _CountingCli:
    """Scripted ``invoke_fn`` that always returns the SAME ``AgentResult``,
    counting its calls.

    Deliberately tolerates being called more than once (rather than raising
    once "exhausted"): before this task's fix the heuristic cap net
    misclassified a fast zero-cost 529 as a cap hit and failed over across the
    whole pool, so a regression must show up as an OBSERVED call count rather
    than be masked behind a harness exception.
    """

    def __init__(self, result: AgentResult) -> None:
        self._result = result
        self.calls = 0

    async def __call__(self, **kwargs: object) -> AgentResult:
        self.calls += 1
        return self._result


class _ScriptedCli:
    """Scripted ``invoke_fn`` returning each queued result in order, repeating
    the last one once the script is exhausted (so a regressed loop shows up as
    a call count, not a harness error)."""

    def __init__(self, *results: AgentResult) -> None:
        self._results = list(results)
        self.calls = 0

    async def __call__(self, **kwargs: object) -> AgentResult:
        idx = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[idx]


class _SpyCostStore:
    """Records ``save_invocation`` / ``save_account_event`` kwargs so the
    drives can assert that NO ``cap_hit`` account event is written for a
    server error (PRD decision 4: no account mutation)."""

    def __init__(self) -> None:
        self.invocations: list[dict] = []
        self.account_events: list[dict] = []

    async def save_invocation(self, **kwargs: object) -> None:
        self.invocations.append(dict(kwargs))

    async def save_account_event(self, **kwargs: object) -> None:
        self.account_events.append(dict(kwargs))


def _cap_hit_events(store: _SpyCostStore) -> list[dict]:
    return [e for e in store.account_events if e.get('event_type') == 'cap_hit']


@pytest.mark.asyncio
class TestInvokeWithCapRetryServerError:
    """PRD decision 4: a server error is NOT account-scoped — the 2026-07-29
    incident data showed the FRESHEST account carrying the highest failure
    rate, i.e. the provider was degraded, not any one account's quota.  So the
    retry loop must neither fail over nor mutate account state on it; the
    failed result goes back to the caller, which owns the transient requeue.
    """

    async def test_row9_fast_zero_cost_529_does_not_cap_or_failover(self):
        """PRD boundary row 9. Regression guard: before this fix the heuristic
        cap net (zero-cost / <=1 turn / <5s) marked a healthy acct-0 CAPPED and
        failed over pointlessly."""
        gate = _make_gate(['acct-0', 'acct-1'])
        result = AgentResult(
            success=False,
            output=OVERLOADED_BODY,
            api_error_status=529,
            cost_usd=0.0,
            turns=0,
            duration_ms=1200,
        )
        cli = _CountingCli(result)
        store = _SpyCostStore()

        with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
            try:
                returned = await invoke_with_cap_retry(
                    gate, 'server-error[row9]',
                    invoke_fn=cli,
                    max_cap_retries=2,
                    backend='claude',
                    cost_store=store,  # type: ignore[arg-type]
                    prompt='hi',
                )
            except AllAccountsCappedException as exc:
                pytest.fail(
                    f'invoke_with_cap_retry raised AllAccountsCappedException after '
                    f'{cli.calls} call(s) for a server-side 529 — expected the result '
                    f'returned directly with no cross-account failover: {exc}'
                )

        assert cli.calls == 1, (
            f'expected exactly one invocation (no cross-account failover on a '
            f'not-account-scoped server error), the retry loop made {cli.calls}'
        )
        assert returned is result
        assert returned.success is False

        for acct in gate._accounts:
            assert acct.phase not in (AccountPhase.CAPPED, AccountPhase.AUTH_FAILED), (
                f'expected no account mutated by a server error, got '
                f'{acct.name}={acct.phase}'
            )
        assert _cap_hit_events(store) == [], (
            f'a server error must not write a cap_hit account event, got '
            f'{_cap_hit_events(store)}'
        )

    async def test_row1_timed_out_529_on_resume_is_terminal_no_fresh_retry(self):
        """PRD boundary row 1 (cap-retry half). The ServerError branch exits the
        loop, so the wedge guard's fresh-retry churn never happens — and the
        orphaned provider session is still never re-resumed (decision 2's
        intent, preserved by exiting rather than by widening the guard)."""
        gate = _make_gate(['acct-0', 'acct-1'])
        result = dataclasses.replace(_incident_result(), session_id='sess-wedged-529')
        cli = _CountingCli(result)
        store = _SpyCostStore()

        with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
            try:
                returned = await invoke_with_cap_retry(
                    gate, 'server-error[row1]',
                    invoke_fn=cli,
                    max_cap_retries=2,
                    backend='claude',
                    cost_store=store,  # type: ignore[arg-type]
                    prompt='hi',
                    resume_session_id='sess-wedged-529',
                )
            except AllAccountsCappedException as exc:
                pytest.fail(
                    f'invoke_with_cap_retry raised AllAccountsCappedException after '
                    f'{cli.calls} call(s) for a timed-out 529: {exc}'
                )

        assert cli.calls == 1, (
            f'expected exactly one invocation (no fresh-retry churn), the retry '
            f'loop made {cli.calls}'
        )
        assert returned is result

        for acct in gate._accounts:
            assert acct.phase not in (AccountPhase.CAPPED, AccountPhase.AUTH_FAILED), (
                f'expected no account mutated by a timed-out server error, got '
                f'{acct.name}={acct.phase}'
            )
        assert _cap_hit_events(store) == []


@pytest.mark.asyncio
class TestInvokeWithCapRetryNonServerErrorUnchanged:
    """PRD boundary row 10 — the 'did not move' locks. These are GREEN both
    before and after the fix; they exist so a regression in the cap/auth
    failover paths is caught by this module rather than only by test_cap_retry.
    """

    async def test_429_cap_body_still_caps_and_fails_over(self):
        gate = _make_gate(['acct-0', 'acct-1'])
        capped = gate._accounts[0]
        cap_result = AgentResult(
            success=False,
            output=CAP_BODY,
            api_error_status=429,
            cost_usd=0.01,  # blocks the heuristic net — exercise the real detector
        )
        ok_result = AgentResult(success=True, output='done', cost_usd=0.01)
        cli = _ScriptedCli(cap_result, ok_result)

        with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
            returned = await invoke_with_cap_retry(
                gate, 'server-error[row10-cap]',
                invoke_fn=cli,
                backend='claude',
                prompt='hi',
            )

        assert returned.success is True
        assert cli.calls == 2, f'expected cap-then-failover (2 calls), got {cli.calls}'
        assert capped.phase == AccountPhase.CAPPED

    async def test_401_still_auth_fails_and_fails_over(self):
        gate = _make_gate(['acct-0', 'acct-1'])
        auth_acct = gate._accounts[0]
        auth_result = AgentResult(
            success=False,
            output='Unauthorized',
            api_error_status=401,
            cost_usd=0.01,
        )
        ok_result = AgentResult(success=True, output='done', cost_usd=0.01)
        cli = _ScriptedCli(auth_result, ok_result)

        with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
            returned = await invoke_with_cap_retry(
                gate, 'server-error[row10-auth]',
                invoke_fn=cli,
                backend='claude',
                prompt='hi',
            )

        assert returned.success is True
        assert cli.calls == 2, f'expected auth-then-failover (2 calls), got {cli.calls}'
        assert auth_acct.phase == AccountPhase.AUTH_FAILED

    async def test_non_5xx_zero_cost_instant_failure_still_trips_the_heuristic_net(self):
        """Pins that the cap-net guard is NARROWED to ServerError, not a blanket
        disable: a zero-cost instant 499 with no recognisable body still reaches
        the heuristic net and caps the account exactly as it does today."""
        gate = _make_gate(['acct-0', 'acct-1'])
        capped = gate._accounts[0]
        result = AgentResult(
            success=False,
            output='client closed request',
            api_error_status=499,
            cost_usd=0.0,
            turns=0,
            duration_ms=100,
        )
        ok_result = AgentResult(success=True, output='done', cost_usd=0.01)
        cli = _ScriptedCli(result, ok_result)

        with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
            returned = await invoke_with_cap_retry(
                gate, 'server-error[non-5xx-net]',
                invoke_fn=cli,
                backend='claude',
                prompt='hi',
            )

        assert returned.success is True
        assert capped.phase == AccountPhase.CAPPED, (
            'the heuristic cap net must still fire for a non-5xx zero-cost '
            'instant failure'
        )
