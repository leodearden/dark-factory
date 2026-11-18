"""Tests for POST /api/v2/dashboard/curator/cancel.

Step-by-step TDD:
  step-1  validation → 400 (RED before endpoint exists)
  step-3  happy-path cancelled / no_op → 200 verbatim
  step-5  not_found → 404 verbatim
  step-7  all servers unreachable → 502
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATCH_TARGET = 'dashboard.data.memory.mcp_tool_call'
_INVALIDATE_TARGET = 'dashboard.data.memory.invalidate_session'


@dataclass(frozen=True)
class _PerUrl:
    """Scripted cancel_ticket outcomes keyed by URL, for fan-out tests."""

    outcomes: Mapping[str, object]


def _cancel_script(outcome: object) -> Callable[..., Awaitable[object]]:
    """Script ``mcp_tool_call`` to raise or return *outcome*, or ``_PerUrl.outcomes[url]``."""

    async def _scripted_mcp_tool_call(
        client: object, url: str, tool_name: str, arguments: object, **kwargs: object
    ) -> object:
        result = outcome.outcomes[url] if isinstance(outcome, _PerUrl) else outcome
        if isinstance(result, BaseException):
            raise result
        return result

    return _scripted_mcp_tool_call


@contextmanager
def _patched_cancel_seams(outcome: object) -> Iterator[tuple[AsyncMock, list[str]]]:
    """Script ``mcp_tool_call`` with :func:`_cancel_script` and record ``invalidate_session``."""
    invalidated: list[str] = []
    mock_mcp = AsyncMock(side_effect=_cancel_script(outcome))
    with (
        patch(_PATCH_TARGET, new=mock_mcp),
        patch(_INVALIDATE_TARGET, new=invalidated.append),
    ):
        yield mock_mcp, invalidated


def _dials(mock_mcp: AsyncMock) -> list[tuple[str, str]]:
    """Every ``(url, tool)`` the request dialed, in order."""
    return [(c.args[1], c.args[2]) for c in mock_mcp.call_args_list]


# ---------------------------------------------------------------------------
# step-1: invalid ticket_id → 400, no MCP call
# ---------------------------------------------------------------------------

_INVALID_BODIES = [
    pytest.param(None, id='non-dict-body'),  # send as raw bytes
    pytest.param({}, id='missing-key'),
    pytest.param({'ticket_id': 123}, id='non-string'),
    pytest.param({'ticket_id': ''}, id='empty-string'),
    pytest.param({'ticket_id': 'wrong_prefix_abc'}, id='wrong-prefix'),
]


@pytest.mark.parametrize('body', _INVALID_BODIES)
def test_invalid_ticket_id_returns_400(client, body):
    """Invalid / missing ticket_id must return 400 without calling mcp_tool_call."""
    with patch(_PATCH_TARGET, new=AsyncMock()) as mock_mcp:
        if body is None:
            # Simulate non-JSON / non-dict body
            resp = client.post(
                '/api/v2/dashboard/curator/cancel',
                content=b'not json at all',
                headers={'Content-Type': 'application/json'},
            )
        else:
            resp = client.post('/api/v2/dashboard/curator/cancel', json=body)

    assert resp.status_code == 400
    data = resp.json()
    assert data.get('error') == 'invalid_ticket_id'
    mock_mcp.assert_not_called()


# ---------------------------------------------------------------------------
# step-3: happy-path → 200 verbatim, mcp_tool_call invoked correctly
# ---------------------------------------------------------------------------

_SUCCESS_SHAPES = [
    pytest.param(
        {'status': 'cancelled', 'ticket_id': 'tkt_abc'},
        id='cancelled',
    ),
    pytest.param(
        {'status': 'completed', 'ticket_id': 'tkt_abc', 'no_op': True},
        id='no_op',
    ),
]


@pytest.mark.parametrize('mcp_result', _SUCCESS_SHAPES)
def test_successful_proxy_forwards_verbatim(client, mcp_result):
    """Valid ticket_id → 200, body is the exact MCP result, call args correct."""
    with patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_result)) as mock_mcp:
        resp = client.post(
            '/api/v2/dashboard/curator/cancel',
            json={'ticket_id': 'tkt_abc'},
        )

    assert resp.status_code == 200
    assert resp.json() == mcp_result

    # The cancel handler must call cancel_ticket exactly once with the correct
    # args.  The positional destructure below assumes positional invocation.
    (call,) = mock_mcp.call_args_list
    _client_arg, url_arg, tool_arg, args_arg = call.args
    assert tool_arg == 'cancel_ticket'
    assert args_arg == {'ticket_id': 'tkt_abc'}
    # The URL must be exactly the first entry of the config THIS app resolved,
    # read from app.state rather than from DEFAULT_FUSED_MEMORY_URLS.  The
    # contract under test is "the handler dials the configured URL", not "the
    # configured URL is the packaged default" — test_scaffold.py owns the
    # latter.  Deriving also keeps this green while the suite points
    # DASHBOARD_FUSED_MEMORY_URLS at a dead port (task 5185).
    assert url_arg == client.app.state.config.fused_memory_urls[0]


# ---------------------------------------------------------------------------
# step-5: not_found → 404 verbatim
# ---------------------------------------------------------------------------


def test_not_found_returns_404(client):
    """MCP not_found response → 404 with the body forwarded verbatim."""
    mcp_result = {'error': 'not_found', 'ticket_id': 'tkt_missing'}
    with patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_result)):
        resp = client.post(
            '/api/v2/dashboard/curator/cancel',
            json={'ticket_id': 'tkt_missing'},
        )

    assert resp.status_code == 404
    assert resp.json() == mcp_result


# ---------------------------------------------------------------------------
# step-7: every caught failure type → 502
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('exc', 'expected_error'),
    [
        (httpx.ConnectError('refused'), 'fused_memory_unreachable'),
        (httpx.TimeoutException('timed out'), 'fused_memory_outcome_unknown'),
        (
            httpx.HTTPStatusError(
                '500 Internal Server Error',
                request=httpx.Request('POST', 'http://x'),
                response=httpx.Response(500, request=httpx.Request('POST', 'http://x')),
            ),
            'fused_memory_outcome_unknown',
        ),
        (ValueError('bad MCP payload'), 'fused_memory_outcome_unknown'),
    ],
    ids=['ConnectError', 'TimeoutException', 'HTTPStatusError', 'ValueError'],
)
def test_each_caught_failure_type_returns_502(client, exc, expected_error):
    """Each of the four caught exception types → 502, keyed by whether delivery is ruled out.

    This is a regression-pin test: the production except clause catches
    ConnectError, TimeoutException, HTTPStatusError, and ValueError. Any future
    refactor that drops one of these types will fail this parametrized test,
    surfacing the contract break immediately. Of these four, only ConnectError
    proves the cancel never reached the server (the full set is
    ``mcp_fanout.py::_UNDELIVERED_FAILURES``), so only it is
    ``fused_memory_unreachable``; the others may have landed and answer
    ``fused_memory_outcome_unknown``.
    """
    with _patched_cancel_seams(exc) as (mock_mcp, _invalidated):
        resp = client.post(
            '/api/v2/dashboard/curator/cancel',
            json={'ticket_id': 'tkt_xyz'},
        )

    assert resp.status_code == 502
    data = resp.json()
    assert data.get('error') == expected_error
    assert 'detail' in data
    assert isinstance(data['detail'], str)
    # Default config has exactly one URL → exactly one cancel_ticket dial
    assert _dials(mock_mcp) == [(client.app.state.config.fused_memory_urls[0], 'cancel_ticket')]


# ---------------------------------------------------------------------------
# step-3 (task-1285): invalidate_session called on transport error
# ---------------------------------------------------------------------------


def test_cancel_handler_invalidates_session_on_transport_error(client):
    """Transport error → invalidate_session called once with the failing URL.

    Verifies that the cancel handler routes through invalidate_session (the
    public helper) rather than reaching into memory_data._sessions directly,
    ensuring the module-boundary contract introduced in task-1285/step-4.
    """
    configured_url = client.app.state.config.fused_memory_urls[0]

    with _patched_cancel_seams(httpx.ConnectError('refused')) as (_mock_mcp, invalidated):
        resp = client.post(
            '/api/v2/dashboard/curator/cancel',
            json={'ticket_id': 'tkt_xyz'},
        )

    assert resp.status_code == 502
    assert invalidated == [configured_url]


# ---------------------------------------------------------------------------
# Two-URL fan-out tests (suggestion 3): fallback and short-circuit
# ---------------------------------------------------------------------------


def test_two_url_fallback_url0_fails_url1_succeeds(two_url_client):
    """URL[0] raises ConnectError; URL[1] returns cancelled → overall 200.

    Verifies that the for-loop actually falls through to the next server on a
    transport error (rather than short-circuiting and returning 502 too early).
    """
    mcp_result = {'status': 'cancelled', 'ticket_id': 'tkt_abc'}
    outcomes = _PerUrl({
        'http://localhost:9000': httpx.ConnectError('refused'),
        'http://localhost:9001': mcp_result,
    })

    with _patched_cancel_seams(outcomes) as (mock_mcp, _invalidated):
        resp = two_url_client.post(
            '/api/v2/dashboard/curator/cancel',
            json={'ticket_id': 'tkt_abc'},
        )

    assert resp.status_code == 200
    assert resp.json() == mcp_result
    # Both URLs must have been tried, in order
    assert _dials(mock_mcp) == [
        ('http://localhost:9000', 'cancel_ticket'),
        ('http://localhost:9001', 'cancel_ticket'),
    ]


def test_two_url_not_found_short_circuits_loop(two_url_client):
    """URL[0] returns not_found → 404 immediately; URL[1] is never called.

    Verifies the first-success-or-not_found semantics: an authoritative MCP
    response (including not_found) stops the loop, so we don't incorrectly
    report 404 when the ticket might live on a different server.
    """
    mcp_result = {'error': 'not_found', 'ticket_id': 'tkt_missing'}

    with _patched_cancel_seams(mcp_result) as (mock_mcp, _invalidated):
        resp = two_url_client.post(
            '/api/v2/dashboard/curator/cancel',
            json={'ticket_id': 'tkt_missing'},
        )

    assert resp.status_code == 404
    assert resp.json() == mcp_result
    # Only URL[0] must have been called; URL[1] must be skipped
    assert _dials(mock_mcp) == [('http://localhost:9000', 'cancel_ticket')]


def test_two_url_all_unreachable_returns_502_with_both_urls(two_url_client):
    """Both URLs raise ConnectError → 502 with detail mentioning both URLs.

    Verifies the fan-out exhaustion path: all configured fused-memory servers
    are unreachable, so the handler returns 502 with the error key
    'fused_memory_unreachable' and includes each server URL in the detail string.
    """
    with _patched_cancel_seams(httpx.ConnectError('refused')) as (mock_mcp, _invalidated):
        resp = two_url_client.post(
            '/api/v2/dashboard/curator/cancel',
            json={'ticket_id': 'tkt_xyz'},
        )

    assert resp.status_code == 502
    data = resp.json()
    assert data.get('error') == 'fused_memory_unreachable'
    detail = data.get('detail', '')
    # Both server URLs must appear in the error detail
    assert 'localhost:9000' in detail
    assert 'localhost:9001' in detail
    assert _dials(mock_mcp) == [
        ('http://localhost:9000', 'cancel_ticket'),
        ('http://localhost:9001', 'cancel_ticket'),
    ]


def test_two_url_all_unreachable_invalidates_each_session_in_order(two_url_client):
    """Regression pin: per-URL invalidate_session is called once per failing URL, in order.

    For each URL that raises a transport error the handler must call
    memory_data.invalidate_session(url) exactly once before moving to the
    next server.  Any future refactor that batches these calls (e.g. moves
    invalidation to after the loop), removes them, or skips them for certain
    error types will fail this assertion immediately.
    """
    urls = ['http://localhost:9000', 'http://localhost:9001']
    with _patched_cancel_seams(httpx.ConnectError('refused')) as (_mock_mcp, invalidated):
        resp = two_url_client.post(
            '/api/v2/dashboard/curator/cancel',
            json={'ticket_id': 'tkt_xyz'},
        )

    assert resp.status_code == 502
    assert invalidated == urls


# ---------------------------------------------------------------------------
# task-2218 step-7: guard the not_found short-circuit ahead of the pending
# first_success conversion — an authoritative not_found from url[0] must NOT
# fan out to url[1].
# ---------------------------------------------------------------------------


def test_cancel_handler_not_found_does_not_fan_out(two_url_client):
    """not_found from url[0] is authoritative → 404 verbatim; url[1] untouched.

    Regression guard ahead of routing cancel_ticket's loop onto
    ``mcp_fanout.first_success``: first_success stops at the first ``_call``
    that does not raise, so a not_found JSONResponse must short-circuit the
    fan-out exactly like the original for-loop's early
    ``return JSONResponse(result, 404)`` — cancel_ticket must be dialed
    exactly once (never falling through to url[1]).
    """
    mcp_result = {'error': 'not_found', 'ticket_id': 'tkt_missing'}

    with _patched_cancel_seams(mcp_result) as (mock_mcp, _invalidated):
        resp = two_url_client.post(
            '/api/v2/dashboard/curator/cancel',
            json={'ticket_id': 'tkt_missing'},
        )

    assert resp.status_code == 404
    assert resp.json() == mcp_result
    assert _dials(mock_mcp) == [('http://localhost:9000', 'cancel_ticket')]
