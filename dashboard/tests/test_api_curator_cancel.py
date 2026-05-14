"""Tests for POST /api/v2/dashboard/curator/cancel.

Step-by-step TDD:
  step-1  validation → 400 (RED before endpoint exists)
  step-3  happy-path cancelled / no_op → 200 verbatim
  step-5  not_found → 404 verbatim
  step-7  all servers unreachable → 502
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

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
    # The URL must come from the app's default fused_memory_urls[0]
    assert 'localhost' in url_arg


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
