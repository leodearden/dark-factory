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

import asyncio
import logging

import httpx
import pytest

from dashboard.data.mcp_fanout import TTLCache, first_success


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


# ── (c2) per-URL failures are surfaced at WARNING, on transition ─────


class TestFirstSuccessLogsFailuresAtWarning:
    """A failing URL must leave a journal trace at WARNING, not DEBUG.

    The dashboard runs at the default WARNING root level, so a DEBUG log
    here means a *total* fused-memory/escalation outage produces no journal
    record at all — the fan-out silently degrades to the offline sentinel
    and the operator sees only an "offline" pill with no cause. Same class
    of fix as task 1814.

    The opposite failure is equally real: first_success is on ~8 hot paths
    behind a 2s UI poll, so WARNING-per-failure would turn a *sustained*
    outage into hundreds of identical lines a minute. The policy is therefore
    transition-only — see TestFanoutFailureThrottling below.
    """

    async def test_each_failing_url_logs_one_warning(self, caplog):
        urls = ['http://a', 'http://b']

        async def call(url):
            raise httpx.ConnectError(f'{url} refused')

        with caplog.at_level(logging.WARNING, logger='dashboard.data.mcp_fanout'):
            result = await first_success(
                urls, call, log_label='get_memory_status',
                offline_result=_offline_result,
            )

        assert result['offline'] is True, 'still returns the caller offline sentinel'

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'dashboard.data.mcp_fanout'
        ]
        assert len(warnings) == len(urls), (
            f'expected one WARNING per failing URL, got {len(warnings)}: '
            f'{[r.getMessage() for r in warnings]}'
        )
        messages = [r.getMessage() for r in warnings]
        for url, message in zip(urls, messages, strict=True):
            assert 'get_memory_status' in message, (
                f'warning must name the log_label, got: {message}'
            )
            assert url in message, f'warning must name the failing url, got: {message}'
            assert 'refused' in message, (
                f'warning must carry the underlying error, got: {message}'
            )


# ── (c3) sustained failure must not become a log flood ───────────────


class TestFanoutFailureThrottling:
    """WARNING on transition; DEBUG for the repeats in between.

    A total fused-memory outage lasts minutes-to-hours while the UI polls
    every 2s across ~8 fan-out paths. One WARNING per failure would emit
    order 150-300 lines/minute indefinitely, burying the first (diagnostic)
    line and growing the journal without bound. The operator is served by the
    first occurrence and by the recovery line, not by the 10,000th repeat.
    """

    @staticmethod
    async def _fail_once(label: str = 'probe', url: str = 'http://a') -> None:
        async def call(_url):
            raise httpx.ConnectError('refused')

        await first_success(
            [url], call, log_label=label, offline_result=_offline_result,
        )

    async def _succeed_once(self, label: str = 'probe', url: str = 'http://a'):
        async def call(_url):
            return 'ok'

        return await first_success(
            [url], call, log_label=label, offline_result=_offline_result,
        )

    async def test_repeat_failures_are_demoted_to_debug(self, caplog):
        with caplog.at_level(logging.DEBUG, logger='dashboard.data.mcp_fanout'):
            for _ in range(5):
                await self._fail_once()

        records = [r for r in caplog.records if r.name == 'dashboard.data.mcp_fanout']
        warnings = [r for r in records if r.levelno == logging.WARNING]
        debugs = [r for r in records if r.levelno == logging.DEBUG]

        assert len(warnings) == 1, (
            f'a sustained streak must warn exactly once, got '
            f'{[r.getMessage() for r in warnings]}'
        )
        assert len(debugs) == 4, (
            f'the 4 repeats must still be recorded at DEBUG, got '
            f'{[r.getMessage() for r in debugs]}'
        )

    async def test_streaks_are_tracked_per_label_and_url(self, caplog):
        with caplog.at_level(logging.WARNING, logger='dashboard.data.mcp_fanout'):
            await self._fail_once(label='probe', url='http://a')
            await self._fail_once(label='probe', url='http://b')
            await self._fail_once(label='other', url='http://a')

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'dashboard.data.mcp_fanout'
        ]
        assert len(warnings) == 3, (
            'each (log_label, url) pair reports independently, got '
            f'{[r.getMessage() for r in warnings]}'
        )

    async def test_recovery_closes_the_streak_and_re_arms_the_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger='dashboard.data.mcp_fanout'):
            await self._fail_once()
            await self._fail_once()  # demoted to DEBUG
            assert await self._succeed_once() == 'ok'
            await self._fail_once()  # streak closed → warns again

        messages = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'dashboard.data.mcp_fanout'
        ]
        assert len(messages) == 3, (
            f'expected open + recovery + re-open, got {messages}'
        )
        assert 'recovered' in messages[1], (
            f'the streak needs a visible closing bracket, got {messages[1]}'
        )
        assert '2 consecutive' in messages[1], (
            f'recovery should report the streak length, got {messages[1]}'
        )

    async def test_reset_sessions_clears_streak_state(self, caplog):
        from dashboard.data.memory import reset_sessions

        await self._fail_once()
        reset_sessions()
        caplog.clear()  # drop the opening WARNING captured above

        with caplog.at_level(logging.WARNING, logger='dashboard.data.mcp_fanout'):
            await self._fail_once()

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'dashboard.data.mcp_fanout'
        ]
        assert len(warnings) == 1, (
            'reset_sessions must clear throttling state so one test cannot '
            "silently demote the next test's first WARNING to DEBUG"
        )

    async def test_log_failures_false_suppresses_the_fanout_report(self, caplog):
        """The two app.py proxies log their own detailed WARNING at the call site."""
        async def call(_url):
            raise httpx.ConnectError('refused')

        with caplog.at_level(logging.DEBUG, logger='dashboard.data.mcp_fanout'):
            result = await first_success(
                ['http://a'], call, log_label='cancel_ticket',
                offline_result=_offline_result, log_failures=False,
            )

        assert result['offline'] is True, 'still returns the caller offline sentinel'
        assert not [
            r for r in caplog.records if r.name == 'dashboard.data.mcp_fanout'
        ], 'log_failures=False must leave reporting entirely to the caller'


# ── (c4) an empty-str exception must still name a cause ──────────────


class TestFanoutFailureNamesTheExceptionType:
    """``httpx.PoolTimeout`` stringifies to '' — the type name is the signal.

    A PoolTimeout means *this client's own* connection pool is saturated, not
    that the server is down; with a bare ``str(exc)`` both the WARNING and the
    offline sentinel would read "failed for <url>: " and name nothing. The
    distinction matters now that the shared client carries an explicit
    ``limits=`` bound (task 3871, app.py).
    """

    async def test_pool_timeout_is_named_in_the_warning_and_the_sentinel(self, caplog):
        async def call(_url):
            # An empty message is exactly how httpx raises this in practice.
            raise httpx.PoolTimeout('')

        with caplog.at_level(logging.WARNING, logger='dashboard.data.mcp_fanout'):
            result = await first_success(
                ['http://a'], call, log_label='get_status',
                offline_result=_offline_result,
            )

        warnings = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'dashboard.data.mcp_fanout'
        ]
        assert len(warnings) == 1
        assert 'PoolTimeout' in warnings[0], (
            f'pool saturation must be distinguishable from a dead endpoint, '
            f'got: {warnings[0]}'
        )
        assert 'PoolTimeout' in result['error'], (
            f'the offline sentinel must name a cause too, got: {result["error"]}'
        )


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


# ── TTLCache ═══════════════════════════════════════════════════════
#
# Generalizes scheduler.py's _scheduler_cache + _scheduler_refresh_lock +
# double-checked-locking pattern into a reusable single-flight short-TTL
# cache keyed by an arbitrary string.


# ── (a) fresh hit — second call within TTL does not re-run refresh ──


class TestTTLCacheFreshHit:
    async def test_second_call_within_ttl_does_not_refresh(self):
        cache = TTLCache(ttl_seconds=60.0)
        calls = 0

        async def refresh():
            nonlocal calls
            calls += 1
            return {'v': calls}

        first = await cache.get_or_refresh('k', refresh)
        second = await cache.get_or_refresh('k', refresh)

        assert calls == 1, 'expected refresh to run exactly once for a warm cache'
        assert first == second == {'v': 1}


# ── (b) callable-ttl expiry ───────────────────────────────────────────


class TestTTLCacheCallableTTLExpiry:
    async def test_callable_ttl_expiry_triggers_refetch(self):
        """Proves the mechanism fetch_tasks relies on for its runtime monkeypatch.

        _FETCH_TASKS_TTL_SECONDS is monkeypatched at runtime in
        test_tasks.py::test_fetch_tasks_ttl_expiry_refetches; TTLCache must
        resolve a callable ttl_seconds at each freshness check (not once at
        construction) for that to keep working.
        """
        ttl_box = {'v': 60.0}
        cache = TTLCache(ttl_seconds=lambda: ttl_box['v'])
        calls = 0

        async def refresh():
            nonlocal calls
            calls += 1
            return calls

        await cache.get_or_refresh('k', refresh)
        assert calls == 1

        ttl_box['v'] = 0.0  # every entry is now immediately stale
        result = await cache.get_or_refresh('k', refresh)

        assert calls == 2, 'expected a re-run once the callable ttl reports 0.0'
        assert result == 2


# ── (c) single-flight (double-checked locking) ────────────────────────


class TestTTLCacheSingleFlight:
    async def test_concurrent_cold_callers_single_flight(self):
        cache = TTLCache(ttl_seconds=60.0)
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def refresh():
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return 'value'

        task1 = asyncio.create_task(cache.get_or_refresh('k', refresh))
        task2 = asyncio.create_task(cache.get_or_refresh('k', refresh))

        await started.wait()
        # Give the second (would-be-cold) caller a chance to reach the lock
        # and block on it before we release the in-flight refresh.
        await asyncio.sleep(0)
        release.set()

        result1, result2 = await asyncio.gather(task1, task2)

        assert calls == 1, 'only the first caller should have run refresh'
        assert result1 == result2 == 'value'


# ── (c2) non-cacheable results do not single-flight ────────────────────


class TestTTLCacheNonCacheableConcurrency:
    """Pins a documented caveat: single-flight collapse requires cache_ok.

    When cache_ok(value) is False (e.g. an offline/error marker), nothing
    is stored, so a lock-queued waiter's post-lock freshness re-check still
    misses and it reruns refresh itself. Concurrent cold callers during an
    outage therefore each perform their own refresh — serialized one at a
    time by the per-key lock, not collapsed onto a single call — rather
    than sharing one in-flight result the way cacheable values do (see
    TestTTLCacheSingleFlight above).
    """

    async def test_non_cacheable_concurrent_callers_each_rerun_refresh(self):
        cache = TTLCache(ttl_seconds=60.0)
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def refresh():
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return {'offline': True}

        task1 = asyncio.create_task(
            cache.get_or_refresh('k', refresh, cache_ok=lambda v: False)
        )
        task2 = asyncio.create_task(
            cache.get_or_refresh('k', refresh, cache_ok=lambda v: False)
        )

        await started.wait()
        # Give the second (lock-queued) caller a chance to block on the
        # per-key lock before the first in-flight refresh is released.
        await asyncio.sleep(0)
        release.set()

        result1, result2 = await asyncio.gather(task1, task2)

        assert calls == 2, (
            'a non-cacheable result must not be reused by the next lock '
            'waiter — each one reruns refresh since nothing was stored'
        )
        assert result1 == result2 == {'offline': True}


# ── (d) clear() resets store + locks ──────────────────────────────────


class TestTTLCacheClear:
    async def test_clear_resets_store_and_forces_refetch(self):
        cache = TTLCache(ttl_seconds=60.0)
        calls = 0

        async def refresh():
            nonlocal calls
            calls += 1
            return calls

        await cache.get_or_refresh('k', refresh)
        cache.clear()
        result = await cache.get_or_refresh('k', refresh)

        assert calls == 2, 'clear() must force the next call to re-run refresh'
        assert result == 2


# ── (e) cache_ok predicate — generalized store-only-on-success ───────


class TestTTLCacheCacheOk:
    async def test_cache_ok_false_does_not_store(self):
        cache = TTLCache(ttl_seconds=60.0)
        calls = 0

        async def refresh():
            nonlocal calls
            calls += 1
            return {'offline': True}  # not a list

        await cache.get_or_refresh('k', refresh, cache_ok=lambda v: isinstance(v, list))
        await cache.get_or_refresh('k', refresh, cache_ok=lambda v: isinstance(v, list))

        assert calls == 2, 'cache_ok=False must prevent storing, forcing a re-run'
