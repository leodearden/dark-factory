"""Tests for _ASGIExceptionShield's shutdown-aware status code (task 2705).

Covers:
- Normal operation: an exception escaping the wrapped app still yields a
  bare 500 (unchanged pre-existing behaviour).
- Shutdown window: once the module-level operator-stop flag is set, the
  same exception yields 503 with a Retry-After header instead — orchestrator
  retry loops treat {502,503,504} as retryable but not 500, so this turns a
  clean-restart race into a retryable blip instead of a hard client failure.
"""

from collections.abc import Callable

import pytest

import fused_memory.server.main as main_module
from fused_memory.server.main import _ASGIExceptionShield


class _RaisingApp:
    """Minimal ASGI app that raises before sending any response."""

    async def __call__(self, scope, receive, send):
        raise RuntimeError('boom')


def _make_shield_call(shield: _ASGIExceptionShield) -> tuple[list[dict], Callable]:
    """Build a `sent` sink plus an `invoke()` coroutine for driving `shield`.

    Unlike `_collect`, this hands back `sent` before the call happens, so a
    caller whose shielded call RAISES (e.g. KeyboardInterrupt/SystemExit
    propagation) can still inspect whatever was sent up to that point.
    """
    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    async def receive() -> dict:
        return {'type': 'http.request'}

    scope = {'type': 'http', 'method': 'POST', 'path': '/mcp/'}

    async def invoke() -> None:
        await shield(scope, receive, send)

    return sent, invoke


async def _collect(shield: _ASGIExceptionShield) -> list[dict]:
    sent, invoke = _make_shield_call(shield)
    await invoke()
    return sent


class TestASGIExceptionShieldNormalOperation:
    @pytest.mark.asyncio
    async def test_exception_yields_bare_500(self, monkeypatch):
        monkeypatch.setattr(main_module, '_operator_stop_received', False)
        shield = _ASGIExceptionShield(_RaisingApp())

        sent = await _collect(shield)

        start = next(m for m in sent if m['type'] == 'http.response.start')
        assert start['status'] == 500
        headers = dict(start['headers'])
        assert b'retry-after' not in headers


class TestASGIExceptionShieldDuringShutdown:
    @pytest.mark.asyncio
    async def test_exception_yields_503_with_retry_after(self, monkeypatch):
        monkeypatch.setattr(main_module, '_operator_stop_received', True)
        shield = _ASGIExceptionShield(_RaisingApp())

        sent = await _collect(shield)

        start = next(m for m in sent if m['type'] == 'http.response.start')
        assert start['status'] == 503
        headers = dict(start['headers'])
        assert headers[b'retry-after'] == b'5'

    def test_is_shutdown_initiated_reads_operator_stop_flag(self, monkeypatch):
        monkeypatch.setattr(main_module, '_operator_stop_received', False)
        assert main_module._is_shutdown_initiated() is False

        monkeypatch.setattr(main_module, '_operator_stop_received', True)
        assert main_module._is_shutdown_initiated() is True
