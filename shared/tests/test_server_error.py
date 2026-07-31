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

import pytest

from shared.cli_invoke import is_server_error_status


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
