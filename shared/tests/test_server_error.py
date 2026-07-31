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
import re

import pytest

from shared.cli_invoke import (
    AgentFailureKind,
    AgentResult,
    classify_agent_failure,
    is_server_error_status,
    is_zero_output_timeout,
)
from shared.invocation_outcome import (
    OK,
    AuthFailed,
    CapHit,
    CliLocalError,
    Failure,
    InvocationOutcome,
    ModelNotFound,
    ServerError,
    ZeroOutputWedge,
    classify_invocation,
)

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
    """

    def test_reexported_from_shared_package(self):
        from shared import is_server_error_status as reexported

        assert reexported is is_server_error_status

    def test_listed_in_cli_invoke_all(self):
        from shared import cli_invoke

        assert 'is_server_error_status' in cli_invoke.__all__

    def test_listed_in_shared_all(self):
        import shared

        assert 'is_server_error_status' in shared.__all__


class TestServerErrorVariant:
    """``ServerError`` is a frozen ``InvocationOutcome`` variant carrying the
    HTTP status, so a consumer can log/route on the exact code."""

    def test_is_an_invocation_outcome(self):
        assert isinstance(ServerError(status=529), InvocationOutcome)

    def test_round_trips_status(self):
        assert ServerError(status=529).status == 529

    def test_is_frozen(self):
        outcome = ServerError(status=529)
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.status = 500  # type: ignore[misc]

    def test_equality_is_status_scoped(self):
        assert ServerError(status=529) == ServerError(status=529)
        assert ServerError(status=529) != ServerError(status=500)
        assert ServerError(status=529) != Failure(kind='unclassified')


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
