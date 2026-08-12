"""Tests for _ASGIExceptionShield's shutdown-aware status codes and
BaseException containment (task 2705; extended by task 4067).

Covers:
- Normal operation: an exception escaping the wrapped app still yields a
  bare 500 (unchanged pre-existing behaviour). [task 2705]
- Shutdown window: once the module-level operator-stop flag is set, the
  same exception yields 503 with a Retry-After header instead — orchestrator
  retry loops treat {502,503,504} as retryable but not 500, so this turns a
  clean-restart race into a retryable blip instead of a hard client failure.
  [task 2705]
- CancelledError containment: asyncio.CancelledError — a BaseException, the
  shield's motivating 2026-04-23 failure mode — is swallowed rather than
  re-raised, and still yields 500/503 per the shutdown flag. RuntimeError
  alone (used by the two cases above) was insufficient: it is an Exception,
  so a refactor narrowing `except BaseException` to `except Exception`
  would leave those tests green while re-opening the wedge. [task 4067;
  provenance: a code-review suggestion recovered from task 2705, see
  task 4023]
- response_started suppression: once the wrapped app has already sent
  `http.response.start`, the fallback response is suppressed entirely — no
  second start message, no fallback body — while the exception is still
  contained. [task 4067, see task 4023]
- Interpreter exit: KeyboardInterrupt/SystemExit still propagate out of the
  shield uncontained, with nothing sent before the re-raise. [task 4067,
  see task 4023]
"""

import asyncio
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


class _CancelledApp:
    """Minimal ASGI app that raises asyncio.CancelledError before sending anything."""

    async def __call__(self, scope, receive, send):
        raise asyncio.CancelledError()


class TestASGIExceptionShieldCancelledError:
    """Pins the shield's motivating failure mode: the 2026-04-23 wedge.

    `asyncio.CancelledError` is a BaseException, not an Exception — unlike
    the RuntimeError used elsewhere in this file. A refactor that narrowed
    `except BaseException` to `except Exception` in
    `_ASGIExceptionShield.__call__` would leave every other test in this
    file green while letting CancelledError cascade into the shared task
    group again, exactly like the incident this class exists to prevent.
    This test is the one that would catch that regression.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('stop_received', 'expected_status'), [(False, 500), (True, 503)])
    async def test_cancelled_error_is_swallowed(self, monkeypatch, stop_received, expected_status):
        monkeypatch.setattr(main_module, '_operator_stop_received', stop_received)
        shield = _ASGIExceptionShield(_CancelledApp())

        # Reaching this line at all proves the CancelledError was swallowed
        # by the shield rather than propagating out of `await shield(...)`.
        sent = await _collect(shield)

        start = next(m for m in sent if m['type'] == 'http.response.start')
        assert start['status'] == expected_status
        assert any(m['type'] == 'http.response.body' for m in sent)
        if expected_status == 503:
            assert dict(start['headers'])[b'retry-after'] == b'5'


class _StartedThenRaisingApp:
    """Minimal ASGI app that sends a real response start, then raises."""

    async def __call__(self, scope, receive, send):
        await send(
            {
                'type': 'http.response.start',
                'status': 200,
                'headers': [(b'content-type', b'application/json')],
            }
        )
        raise RuntimeError('boom after response start')


class TestASGIExceptionShieldResponseStarted:
    """Pins the `if not response_started:` branch — previously uncovered.

    Once the wrapped app has already sent `http.response.start`, the
    shield must not stomp a fallback 500/503 on top of an in-flight
    response: the entire fallback start+body pair sits under
    `if not response_started`, so both must be skipped, not just the
    start message.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize('stop_received', [False, True])
    async def test_no_fallback_response_after_start(self, monkeypatch, stop_received):
        monkeypatch.setattr(main_module, '_operator_stop_received', stop_received)
        shield = _ASGIExceptionShield(_StartedThenRaisingApp())

        # Returning normally proves the RuntimeError was still contained.
        sent = await _collect(shield)

        starts = [m for m in sent if m['type'] == 'http.response.start']
        assert len(starts) == 1
        assert starts[0]['status'] == 200
        # Stronger than "no second start": proves the whole fallback block
        # (start + body) was skipped, not just the start message deduped.
        assert not any(m['type'] == 'http.response.body' for m in sent)


class _BaseExceptionApp:
    """Minimal ASGI app that raises a given BaseException before sending anything."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def __call__(self, scope, receive, send):
        raise self._exc


class TestASGIExceptionShieldInterpreterExit:
    """Pins the `except (KeyboardInterrupt, SystemExit): raise` clause.

    This clause bounds the deliberate CancelledError swallow exercised by
    TestASGIExceptionShieldCancelledError above: without it, that test's
    assertion could be satisfied by a shield that swallows every
    BaseException indiscriminately — including a legitimate interpreter
    shutdown signal. KeyboardInterrupt and SystemExit must still propagate
    out of `await shield(...)`, with nothing fabricated on the wire before
    the re-raise.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize('exc_type', [KeyboardInterrupt, SystemExit])
    async def test_interpreter_exit_propagates_uncontained(self, monkeypatch, exc_type):
        monkeypatch.setattr(main_module, '_operator_stop_received', False)
        sent, invoke = _make_shield_call(_ASGIExceptionShield(_BaseExceptionApp(exc_type())))

        with pytest.raises(exc_type):
            await invoke()

        # Nothing at all emitted before the re-raise — the shield does not
        # fabricate a 500 on the way out of a legitimate interpreter exit.
        assert sent == []
