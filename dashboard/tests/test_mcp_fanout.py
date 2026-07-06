"""Tests for dashboard.data.mcp_fanout — the shared first_success fan-out helper.

first_success() lifts the canonical "try each configured URL in order, return
the first success, invalidate the session and fall through on failure" pattern
out of dashboard.data.memory._first_success (and, imminently, the three
tasks.py loops) into one reusable helper. These tests cover all of its
genuinely-new (not previously tested anywhere) behaviour: failover order,
per-exception-type fall-through with session invalidation, the all-fail
sentinel, the success/no-invalidation path, and uncaught-exception
propagation.
"""

from __future__ import annotations

import httpx
import pytest

from dashboard.data.mcp_fanout import first_success


def _offline_result(errors: list[str]) -> dict:
    """Canonical offline sentinel shape shared by memory.py and tasks.py."""
    return {'offline': True, 'error': '; '.join(errors)}


def _http_status_error(url: str = 'http://x') -> httpx.HTTPStatusError:
    """Build a real httpx.HTTPStatusError (request/response are required kwargs)."""
    request = httpx.Request('POST', f'{url}/mcp')
    response = httpx.Response(500, request=request)
    return httpx.HTTPStatusError('Server Error', request=request, response=response)


@pytest.fixture(autouse=True)
def _clean_sessions():
    """Reset the memory module's session cache before and after each test.

    Several tests below pre-populate a session via memory._get_session to
    prove first_success's failure path really invalidates it end-to-end.
    """
    from dashboard.data.memory import reset_sessions
    reset_sessions()
    yield
    reset_sessions()


# ── (a) failover order + short-circuit ──────────────────────────────


class TestFirstSuccessFailoverOrder:
    """First failing URL is attempted, second (succeeding) URL short-circuits."""

    async def test_first_fails_second_succeeds_third_never_attempted(self):
        attempted: list[str] = []
        urls = ['http://a', 'http://b', 'http://c']

        async def call(url):
            attempted.append(url)
            if url == 'http://a':
                raise httpx.ConnectError('refused')
            return f'{url}-payload'

        result = await first_success(
            urls, call, log_label='test', offline_result=_offline_result,
        )

        assert result == 'http://b-payload'
        assert attempted == ['http://a', 'http://b'], (
            f'expected A then B only (C never attempted), got {attempted}'
        )


# ── (b) per-exception failover + session invalidation ───────────────


class TestFirstSuccessPerExceptionFailover:
    """Each caught exception type falls through AND invalidates the failing session."""

    @pytest.mark.parametrize(
        'exc',
        [
            httpx.ConnectError('refused'),
            httpx.TimeoutException('timed out'),
            _http_status_error(),
            ValueError('bad result'),
        ],
        ids=['ConnectError', 'TimeoutException', 'HTTPStatusError', 'ValueError'],
    )
    async def test_falls_through_and_invalidates_failing_session(self, exc):
        from dashboard.data.memory import _get_session, _sessions

        _get_session('http://x')
        assert 'http://x' in _sessions

        async def call(url):
            if url == 'http://x':
                raise exc
            return 'ok'

        result = await first_success(
            ['http://x', 'http://y'], call,
            log_label='test', offline_result=_offline_result,
        )

        assert result == 'ok'
        assert 'http://x' not in _sessions, (
            "first_success must invalidate the failing url's session"
        )


# ── (c) all-fail → offline_result(errors) ────────────────────────────


class TestFirstSuccessAllFail:
    """When every URL fails, returns offline_result(errors) with per-URL detail."""

    async def test_all_fail_returns_offline_result_with_per_url_errors(self):
        urls = ['http://a', 'http://b']

        async def call(url):
            raise httpx.ConnectError(f'{url} refused')

        result = await first_success(
            urls, call, log_label='test', offline_result=_offline_result,
        )

        assert result['offline'] is True
        assert 'http://a' in result['error']
        assert 'http://b' in result['error']
        assert 'refused' in result['error']


# ── (d) success path: no invalidation, exactly one call ──────────────


class TestFirstSuccessSuccessPath:
    """A first-try success must not invalidate any session and calls exactly once."""

    async def test_success_does_not_invalidate_and_calls_once(self):
        from dashboard.data.memory import _get_session, _sessions

        _get_session('http://a')  # pre-populate — must survive a success

        call_count = 0

        async def call(url):
            nonlocal call_count
            call_count += 1
            return 'payload'

        result = await first_success(
            ['http://a', 'http://b'], call,
            log_label='test', offline_result=_offline_result,
        )

        assert result == 'payload'
        assert call_count == 1
        assert 'http://a' in _sessions, 'success path must not invalidate any session'


# ── (e) uncaught exception types propagate ────────────────────────────


class TestFirstSuccessUncaughtException:
    """An exception type outside the caught set must propagate, not be swallowed."""

    async def test_uncaught_exception_type_propagates(self):
        async def call(url):
            raise KeyError('boom')

        with pytest.raises(KeyError):
            await first_success(
                ['http://a'], call, log_label='test', offline_result=_offline_result,
            )
