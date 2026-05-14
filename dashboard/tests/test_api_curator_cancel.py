"""Tests for POST /api/v2/dashboard/curator/cancel.

Step-by-step TDD:
  step-1  validation → 400 (RED before endpoint exists)
  step-3  happy-path cancelled / no_op → 200 verbatim
  step-5  not_found → 404 verbatim
  step-7  all servers unreachable → 502
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATCH_TARGET = 'dashboard.data.memory.mcp_tool_call'


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
    assert mock_mcp.call_count == 0


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

    mock_mcp.assert_called_once()
    _client_arg, url_arg, tool_arg, args_arg = mock_mcp.call_args.args
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


def test_all_servers_unreachable_returns_502(client):
    """ConnectError from every URL → 502 with fused_memory_unreachable envelope."""
    with patch(
        _PATCH_TARGET,
        new=AsyncMock(side_effect=httpx.ConnectError('refused')),
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
    _INVALIDATE_TARGET = 'dashboard.data.memory.invalidate_session'
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


def _two_url_client(two_url_config):
    """Return a TestClient with the two-URL config overriding app.state.config."""
    from dashboard.app import app

    ctx = TestClient(app)
    ctx.__enter__()
    app.state.config = two_url_config
    return ctx


def test_two_url_fallback_url0_fails_url1_succeeds(two_url_config):
    """URL[0] raises ConnectError; URL[1] returns cancelled → overall 200.

    Verifies that the for-loop actually falls through to the next server on a
    transport error (rather than short-circuiting and returning 502 too early).
    """
    mcp_result = {'status': 'cancelled', 'ticket_id': 'tkt_abc'}
    side_effects = [httpx.ConnectError('refused'), mcp_result]

    ctx = _two_url_client(two_url_config)
    try:
        with patch(_PATCH_TARGET, new=AsyncMock(side_effect=side_effects)) as mock_mcp:
            resp = ctx.post(
                '/api/v2/dashboard/curator/cancel',
                json={'ticket_id': 'tkt_abc'},
            )
    finally:
        ctx.__exit__(None, None, None)

    assert resp.status_code == 200
    assert resp.json() == mcp_result
    # Both URLs must have been tried
    assert mock_mcp.call_count == 2
    urls_called = [call.args[1] for call in mock_mcp.call_args_list]
    assert urls_called == ['http://localhost:9000', 'http://localhost:9001']


def test_two_url_not_found_short_circuits_loop(two_url_config):
    """URL[0] returns not_found → 404 immediately; URL[1] is never called.

    Verifies the first-success-or-not_found semantics: an authoritative MCP
    response (including not_found) stops the loop, so we don't incorrectly
    report 404 when the ticket might live on a different server.
    """
    mcp_result = {'error': 'not_found', 'ticket_id': 'tkt_missing'}

    ctx = _two_url_client(two_url_config)
    try:
        with patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_result)) as mock_mcp:
            resp = ctx.post(
                '/api/v2/dashboard/curator/cancel',
                json={'ticket_id': 'tkt_missing'},
            )
    finally:
        ctx.__exit__(None, None, None)

    assert resp.status_code == 404
    assert resp.json() == mcp_result
    # Only URL[0] must have been called; URL[1] must be skipped
    assert mock_mcp.call_count == 1
    assert mock_mcp.call_args.args[1] == 'http://localhost:9000'


def test_two_url_all_unreachable_returns_502_with_both_urls(two_url_config):
    """Both URLs raise ConnectError → 502 with detail mentioning both URLs."""
    ctx = _two_url_client(two_url_config)
    try:
        with patch(
            _PATCH_TARGET,
            new=AsyncMock(side_effect=httpx.ConnectError('refused')),
        ) as mock_mcp:
            resp = ctx.post(
                '/api/v2/dashboard/curator/cancel',
                json={'ticket_id': 'tkt_xyz'},
            )
    finally:
        ctx.__exit__(None, None, None)

    assert resp.status_code == 502
    data = resp.json()
    assert data.get('error') == 'fused_memory_unreachable'
    detail = data.get('detail', '')
    # Both server URLs must appear in the error detail
    assert 'localhost:9000' in detail
    assert 'localhost:9001' in detail
    assert mock_mcp.call_count == 2
