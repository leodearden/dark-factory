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

import pytest

from shared.cli_invoke import (
    AgentResult,
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
