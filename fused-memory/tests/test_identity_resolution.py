"""Unit tests for _resolve_identity in server/tools.py."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

from mcp.server.fastmcp import Context

from fused_memory.server.tools import _resolve_identity

# ---------------------------------------------------------------------------
# Helpers — lightweight mock objects (no full MCP server needed)
# ---------------------------------------------------------------------------

def _make_ctx(
    client_name: str | None = 'claude-code',
    session_header: str | None = 'abc-123-session',
    *,
    has_client_params: bool = True,
    has_request: bool = True,
) -> Context[Any, Any, Any]:
    """Build a minimal Context-shaped object."""
    ctx = SimpleNamespace()

    if has_client_params and client_name is not None:
        ctx.session = SimpleNamespace(
            client_params=SimpleNamespace(
                clientInfo=SimpleNamespace(name=client_name)
            )
        )
    elif not has_client_params:
        # Stateless HTTP — no client_params at all
        ctx.session = SimpleNamespace(client_params=None)
    else:
        ctx.session = SimpleNamespace(
            client_params=SimpleNamespace(clientInfo=SimpleNamespace(name=None))
        )

    if has_request and session_header is not None:
        headers = {'mcp-session-id': session_header}
        ctx.request_context = SimpleNamespace(
            request=SimpleNamespace(headers=MagicMock(get=lambda k, d=None: headers.get(k, d)))
        )
    elif not has_request:
        # stdio transport — no request_context
        ctx.request_context = None
    else:
        ctx.request_context = SimpleNamespace(
            request=SimpleNamespace(headers=MagicMock(get=lambda k, d=None: None))
        )

    return cast(Context[Any, Any, Any], ctx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResolveIdentity:
    """Test identity resolution logic."""

    def test_both_explicit_returned_unchanged(self):
        """Explicit values override any context."""
        ctx = _make_ctx(client_name='claude-code', session_header='sess-xyz')
        aid, sid = _resolve_identity('my-agent', 'my-session', ctx)
        assert aid == 'my-agent'
        assert sid == 'my-session'

    def test_both_none_derived_from_context(self):
        """When caller passes nothing, derive both from MCP context."""
        ctx = _make_ctx(client_name='claude-code', session_header='abc-123-session')
        aid, sid = _resolve_identity(None, None, ctx)
        assert aid == 'claude-code'
        assert sid == 'abc-123-session'

    def test_mixed_explicit_agent_derived_session(self):
        """Explicit agent_id preserved, session_id derived."""
        ctx = _make_ctx(client_name='claude-code', session_header='sess-456')
        aid, sid = _resolve_identity('orchestrator-7', None, ctx)
        assert aid == 'orchestrator-7'
        assert sid == 'sess-456'

    def test_mixed_derived_agent_explicit_session(self):
        """agent_id derived, explicit session_id preserved."""
        ctx = _make_ctx(client_name='dashboard', session_header='sess-789')
        aid, sid = _resolve_identity(None, 'explicit-sess', ctx)
        assert aid == 'dashboard'
        assert sid == 'explicit-sess'

    def test_context_is_none(self):
        """No context at all — both stay None."""
        aid, sid = _resolve_identity(None, None, None)
        assert aid is None
        assert sid is None

    def test_no_client_params_stateless(self):
        """Stateless HTTP — no client_params → agent_id stays None."""
        ctx = _make_ctx(has_client_params=False, session_header='sess-abc')
        aid, sid = _resolve_identity(None, None, ctx)
        assert aid is None
        assert sid == 'sess-abc'

    def test_no_request_stdio(self):
        """stdio transport — no request → session_id stays None, agent_id still derived."""
        ctx = _make_ctx(client_name='claude-code', has_request=False)
        aid, sid = _resolve_identity(None, None, ctx)
        assert aid == 'claude-code'
        assert sid is None

    def test_explicit_values_with_none_context(self):
        """Explicit values returned even when ctx is None."""
        aid, sid = _resolve_identity('my-agent', 'my-session', None)
        assert aid == 'my-agent'
        assert sid == 'my-session'
