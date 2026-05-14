"""Tests for POST /api/v2/dashboard/curator/cancel.

Step-by-step TDD:
  step-1  validation → 400 (RED before endpoint exists)
  step-3  happy-path cancelled / no_op → 200 verbatim
  step-5  not_found → 404 verbatim
  step-7  all servers unreachable → 502
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest

from dashboard.app import _CANCEL_DETAIL_EXC_CHAR_LIMIT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATCH_TARGET = 'dashboard.data.memory.mcp_tool_call'


def _tool_name(c):
    """Return the tool_name argument from an mcp_tool_call mock call.

    Reads c.args[2] when the call was made positionally (the current handler
    convention) with a c.kwargs.get('tool_name') fallback so that the
    call-list filter still works if a background task happens to use keyword
    arguments.  This only covers the filter step; the positional destructure
    immediately after filtering still assumes positional invocation.
    """
    if len(c.args) > 2:
        return c.args[2]
    return c.kwargs.get('tool_name')


# ---------------------------------------------------------------------------
# step-1: invalid ticket_id → 400, no MCP call
# ---------------------------------------------------------------------------

_INVALID_BODIES = [
    pytest.param(None, id='non-dict-body'),        # send as raw bytes
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
    # Background tasks (metrics loop) may call mcp_tool_call for get_queue_stats
    # or get_status probes; filter those out and assert the remainder is empty so
    # no handler-level MCP call was made on the invalid-input fast-path.
    _BACKGROUND_TOOLS = {'get_queue_stats', 'get_status'}
    handler_calls = [c for c in mock_mcp.call_args_list if _tool_name(c) not in _BACKGROUND_TOOLS]
    assert handler_calls == [], (
        f'Expected no handler MCP calls for invalid input, got: {handler_calls}'
    )


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
    # args.  Background tasks (metrics loop) may also call mcp_tool_call for
    # get_status / get_queue_stats probes during the test window; filter those
    # out so the assertion targets only the handler's own MCP call.
    # _tool_name() filters the call list by tool name; the positional
    # destructure below still assumes positional invocation.
    cancel_calls = [c for c in mock_mcp.call_args_list if _tool_name(c) == 'cancel_ticket']
    assert len(cancel_calls) == 1, (
        f'Expected exactly 1 cancel_ticket call, got {len(cancel_calls)}: {cancel_calls}'
    )
    _client_arg, url_arg, tool_arg, args_arg = cancel_calls[0].args
    assert tool_arg == 'cancel_ticket'
    assert args_arg == {'ticket_id': 'tkt_abc'}
    # The URL must be exactly the first entry in the default config
    from dashboard.config import DEFAULT_FUSED_MEMORY_URLS
    assert url_arg == DEFAULT_FUSED_MEMORY_URLS[0]


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
# step-7: all servers unreachable → 502
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'exc',
    [
        httpx.ConnectError('refused'),
        httpx.TimeoutException('timed out'),
        httpx.HTTPStatusError(
            '500 Internal Server Error',
            request=httpx.Request('POST', 'http://x'),
            response=httpx.Response(500, request=httpx.Request('POST', 'http://x')),
        ),
        ValueError('bad MCP payload'),
    ],
    ids=['ConnectError', 'TimeoutException', 'HTTPStatusError', 'ValueError'],
)
def test_all_servers_unreachable_returns_502(client, exc):
    """Each of the four caught exception types → 502 fused_memory_unreachable.

    This is a regression-pin test: the production except clause catches
    ConnectError, TimeoutException, HTTPStatusError, and ValueError. Any future
    refactor that drops one of these types will fail this parametrized test,
    surfacing the contract break immediately.
    """
    with patch(
        _PATCH_TARGET,
        new=AsyncMock(side_effect=exc),
    ) as mock_mcp:
        resp = client.post(
            '/api/v2/dashboard/curator/cancel',
            json={'ticket_id': 'tkt_xyz'},
        )

    assert resp.status_code == 502
    data = resp.json()
    assert data.get('error') == 'fused_memory_unreachable'
    assert 'detail' in data
    assert isinstance(data['detail'], str)
    # Default config has exactly one URL → exactly one MCP attempt
    assert mock_mcp.call_count == 1


# ---------------------------------------------------------------------------
# Two-URL fan-out tests (suggestion 3): fallback and short-circuit
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# step-3 (task-1285): invalidate_session called on transport error
# ---------------------------------------------------------------------------


def test_cancel_handler_invalidates_session_on_transport_error(client):
    """Transport error → invalidate_session called once with the failing URL.

    Verifies that the cancel handler routes through invalidate_session (the
    public helper) rather than reaching into memory_data._sessions directly,
    ensuring the module-boundary contract introduced in task-1285/step-4.
    """
    _INVALIDATE_TARGET = 'dashboard.app.memory_data.invalidate_session'
    from dashboard.config import DEFAULT_FUSED_MEMORY_URLS

    with (
        patch(
            _PATCH_TARGET,
            new=AsyncMock(side_effect=httpx.ConnectError('refused')),
        ),
        patch(_INVALIDATE_TARGET, new=MagicMock()) as mock_invalidate,
    ):
        resp = client.post(
            '/api/v2/dashboard/curator/cancel',
            json={'ticket_id': 'tkt_xyz'},
        )

    assert resp.status_code == 502
    # invalidate_session must have been called exactly once with the failing URL
    assert mock_invalidate.call_args_list == [call(DEFAULT_FUSED_MEMORY_URLS[0])]


# ---------------------------------------------------------------------------
# step-5 (task-1285): logger.warning + str(exc) in 502 detail
# ---------------------------------------------------------------------------


def test_cancel_handler_logs_warning_and_includes_exc_in_detail(client, caplog):
    """Transport error → WARNING logged and str(exc) included in 502 detail.

    Three simultaneous assertions:
      (a) response is 502
      (b) a WARNING-level record is emitted by the 'dashboard.app' logger
          whose message contains both the URL and the exception text
      (c) the response body's 'detail' field contains the full exception
          message ('connection refused: port 8002'), not just the type name
    """
    exc_msg = 'connection refused: port 8002'
    with patch(
        _PATCH_TARGET,
        new=AsyncMock(side_effect=httpx.ConnectError(exc_msg)),
    ), caplog.at_level(logging.WARNING, logger='dashboard.app'):
        resp = client.post(
            '/api/v2/dashboard/curator/cancel',
            json={'ticket_id': 'tkt_xyz'},
        )

    # (a) 502
    assert resp.status_code == 502

    # (b) WARNING log
    warning_records = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and r.name == 'dashboard.app'
    ]
    assert warning_records, 'Expected at least one WARNING from dashboard.app'
    combined_msg = ' '.join(r.getMessage() for r in warning_records)
    assert 'cancel_ticket failed' in combined_msg
    assert exc_msg in combined_msg

    # (c) detail includes full exception message (not just type name)
    detail = resp.json().get('detail', '')
    assert exc_msg in detail, f'Expected "{exc_msg}" in detail, got: {detail!r}'


def test_cancel_handler_502_detail_truncates_exception_text(client, caplog):
    """502 detail is capped at _CANCEL_DETAIL_EXC_CHAR_LIMIT chars; WARNING keeps full text.

    Guards against arbitrary-length response-body leakage via the 502 detail
    field: the cancel handler interpolates str(exc) in the detail string, which
    can be unbounded when the upstream MCP server returns a large error body.

    Uses ValueError('X' * 300) as the exception fixture because:
      - ValueError is already in the handler's caught-exception tuple
        (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError, ValueError)
      - str(ValueError(s)) == s is part of the stdlib BaseException.__str__
        contract (single-arg case returns self.args[0] as-is), so the assertion
        directly pins the handler's truncation behaviour rather than depending on
        httpx.HTTPStatusError.__str__ formatting, which is not contractual and
        could change across httpx versions.

    Assertions:
      (a) response is 502
      (b) detail contains the first _CANCEL_DETAIL_EXC_CHAR_LIMIT chars of the
          exception message AND does NOT contain _CANCEL_DETAIL_EXC_CHAR_LIMIT+1
          consecutive 'X' chars (i.e. is truncated)
      (c) the WARNING log record still has the FULL 300-char message
          (ops log must not lose information)
    """
    long_msg = 'X' * 300
    exc = ValueError(long_msg)
    with patch(
        _PATCH_TARGET,
        new=AsyncMock(side_effect=exc),
    ), caplog.at_level(logging.WARNING, logger='dashboard.app'):
        resp = client.post(
            '/api/v2/dashboard/curator/cancel',
            json={'ticket_id': 'tkt_xyz'},
        )

    # (a) 502
    assert resp.status_code == 502

    # (b) detail is capped at _CANCEL_DETAIL_EXC_CHAR_LIMIT chars
    detail = resp.json().get('detail', '')
    assert 'X' * _CANCEL_DETAIL_EXC_CHAR_LIMIT in detail, (
        'Expected first _CANCEL_DETAIL_EXC_CHAR_LIMIT X chars in detail'
    )
    assert 'X' * (_CANCEL_DETAIL_EXC_CHAR_LIMIT + 1) not in detail, (
        'Detail must not contain _CANCEL_DETAIL_EXC_CHAR_LIMIT+1 X chars (leaked exc text)'
    )

    # (c) WARNING log retains full exception text
    warning_records = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and r.name == 'dashboard.app'
    ]
    assert warning_records, 'Expected at least one WARNING from dashboard.app'
    combined_msg = ' '.join(r.getMessage() for r in warning_records)
    assert long_msg in combined_msg, 'WARNING log must contain full (untruncated) exception text'


def test_cancel_handler_502_detail_bounded_for_httpstatuserror(client):
    """HTTPStatusError with a large response body yields a bounded 502 detail.

    httpx.HTTPStatusError.__str__ embeds the response body in its string
    representation, so a large upstream error response could leak through the
    502 detail field.  This test verifies that _CANCEL_DETAIL_EXC_CHAR_LIMIT
    bounds the detail length without asserting on the exact HTTPStatusError
    __str__ format (which is not part of the httpx public contract and could
    change across httpx versions).

    Complements test_cancel_handler_502_detail_truncates_exception_text (which
    uses ValueError for a stable contract) by exercising the actual upstream
    scenario the truncation cap was designed to defend against.
    """
    large_body = b'E' * 10000
    exc = httpx.HTTPStatusError(
        'Server Error',
        request=httpx.Request('POST', 'http://x'),
        response=httpx.Response(
            500,
            content=large_body,
            request=httpx.Request('POST', 'http://x'),
        ),
    )
    with patch(_PATCH_TARGET, new=AsyncMock(side_effect=exc)):
        resp = client.post(
            '/api/v2/dashboard/curator/cancel',
            json={'ticket_id': 'tkt_xyz'},
        )

    assert resp.status_code == 502
    detail = resp.json().get('detail', '')
    # Each per-URL error segment is '{url}: {typename}: {str(exc)[:LIMIT]}'.
    # With the default single-URL config the total is bounded by the limit
    # plus a small fixed overhead (URL + type name + separators).  Use 3×
    # the limit as a generous ceiling so the assertion catches truncation
    # failures without depending on URL/typename lengths.
    max_detail = 3 * _CANCEL_DETAIL_EXC_CHAR_LIMIT
    assert len(detail) <= max_detail, (
        f'502 detail must be bounded (≤{max_detail} chars); got {len(detail)} chars'
    )


def test_two_url_fallback_url0_fails_url1_succeeds(two_url_client):
    """URL[0] raises ConnectError; URL[1] returns cancelled → overall 200.

    Verifies that the for-loop actually falls through to the next server on a
    transport error (rather than short-circuiting and returning 502 too early).
    """
    mcp_result = {'status': 'cancelled', 'ticket_id': 'tkt_abc'}
    side_effects = [httpx.ConnectError('refused'), mcp_result]

    with patch(_PATCH_TARGET, new=AsyncMock(side_effect=side_effects)) as mock_mcp:
        resp = two_url_client.post(
            '/api/v2/dashboard/curator/cancel',
            json={'ticket_id': 'tkt_abc'},
        )

    assert resp.status_code == 200
    assert resp.json() == mcp_result
    # Both URLs must have been tried
    assert mock_mcp.call_count == 2
    urls_called = [c.args[1] for c in mock_mcp.call_args_list]
    assert urls_called == ['http://localhost:9000', 'http://localhost:9001']


def test_two_url_not_found_short_circuits_loop(two_url_client):
    """URL[0] returns not_found → 404 immediately; URL[1] is never called.

    Verifies the first-success-or-not_found semantics: an authoritative MCP
    response (including not_found) stops the loop, so we don't incorrectly
    report 404 when the ticket might live on a different server.
    """
    mcp_result = {'error': 'not_found', 'ticket_id': 'tkt_missing'}

    with patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_result)) as mock_mcp:
        resp = two_url_client.post(
            '/api/v2/dashboard/curator/cancel',
            json={'ticket_id': 'tkt_missing'},
        )

    assert resp.status_code == 404
    assert resp.json() == mcp_result
    # Only URL[0] must have been called; URL[1] must be skipped
    assert mock_mcp.call_count == 1
    assert mock_mcp.call_args.args[1] == 'http://localhost:9000'


def test_two_url_all_unreachable_returns_502_with_both_urls(two_url_client):
    """Both URLs raise ConnectError → 502 with detail mentioning both URLs.

    Verifies the fan-out exhaustion path: all configured fused-memory servers
    are unreachable, so the handler returns 502 with the error key
    'fused_memory_unreachable' and includes each server URL in the detail string.
    """
    with patch(
        _PATCH_TARGET,
        new=AsyncMock(side_effect=httpx.ConnectError('refused')),
    ) as mock_mcp:
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
    assert mock_mcp.call_count == 2


def test_two_url_all_unreachable_invalidates_each_session_in_order(two_url_client):
    """Regression pin: per-URL invalidate_session is called once per failing URL, in order.

    For each URL that raises a transport error the handler must call
    memory_data.invalidate_session(url) exactly once before moving to the
    next server.  Any future refactor that batches these calls (e.g. moves
    invalidation to after the loop), removes them, or skips them for certain
    error types will fail this assertion immediately.
    """
    _INVALIDATE_TARGET = 'dashboard.app.memory_data.invalidate_session'
    with (
        patch(
            _PATCH_TARGET,
            new=AsyncMock(side_effect=httpx.ConnectError('refused')),
        ),
        patch(_INVALIDATE_TARGET, new=MagicMock()) as mock_invalidate,
    ):
        two_url_client.post(
            '/api/v2/dashboard/curator/cancel',
            json={'ticket_id': 'tkt_xyz'},
        )

    assert mock_invalidate.call_args_list == [
        call('http://localhost:9000'),
        call('http://localhost:9001'),
    ]
