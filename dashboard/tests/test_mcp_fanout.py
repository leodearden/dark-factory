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
import time
import types

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


# ── (c4b) an already-rendered cause must not be prefixed twice ───────


class TestPreformattedFanoutError:
    """A call site that already rendered 'Type: message' must not be re-prefixed.

    ``first_success`` renders every caught exception through ``describe_exc``,
    which unconditionally prepends ``type(exc).__name__``. A call site that has
    already formatted the *real* cause and re-raises it therefore reached the
    operator doubled — ``'ValueError: ConnectError: refused'`` in the
    cancel_ticket 502 detail and in the dashboard's offline pill.
    """

    def test_is_a_value_error_subclass(self):
        from dashboard.data.mcp_fanout import PreformattedFanoutError

        assert issubclass(PreformattedFanoutError, ValueError), (
            "first_success's catch tuple and callers' own `except ValueError` "
            'must keep working unchanged'
        )

    def test_describe_exc_returns_the_message_verbatim(self):
        from dashboard.data.mcp_fanout import PreformattedFanoutError, describe_exc

        exc = PreformattedFanoutError('ConnectError: refused')
        assert describe_exc(exc) == 'ConnectError: refused', (
            'an already-rendered cause must not gain a second type prefix'
        )

    def test_a_plain_value_error_is_still_prefixed(self):
        from dashboard.data.mcp_fanout import describe_exc

        assert describe_exc(ValueError('ConnectError: refused')) == (
            'ValueError: ConnectError: refused'
        ), 'the opt-out is the marker type only — plain ValueError is unchanged'

    def test_an_empty_message_still_names_a_type(self):
        from dashboard.data.mcp_fanout import PreformattedFanoutError, describe_exc

        # The content-free-log-line wart describe_exc exists to prevent
        # (httpx.PoolTimeout stringifies to '') must not re-enter here.
        assert describe_exc(PreformattedFanoutError('')) == 'PreformattedFanoutError'

    async def test_first_success_emits_a_single_prefix_end_to_end(self, caplog):
        from dashboard.data.mcp_fanout import PreformattedFanoutError

        async def call(_url):
            raise PreformattedFanoutError('ConnectError: refused')

        with caplog.at_level(logging.WARNING, logger='dashboard.data.mcp_fanout'):
            result = await first_success(
                ['http://a'], call, log_label='cancel_ticket',
                offline_result=_offline_result,
            )

        assert result['error'] == 'http://a: ConnectError: refused', (
            f'the offline sentinel must carry one prefix, got {result["error"]!r}'
        )
        assert 'ValueError: ConnectError' not in result['error']

        warnings = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'dashboard.data.mcp_fanout'
        ]
        assert len(warnings) == 1
        assert 'ValueError: ConnectError' not in warnings[0], (
            f'the WARNING must not double the prefix either, got {warnings[0]}'
        )


# ── (c5) per-project-root label discrimination ───────────────────────


class TestFanoutLabel:
    """``fanout_label`` composes the throttle key's per-project-root discriminator.

    The streak key is ``(log_label, url)``, and ONE fused-memory URL serves
    every project_root — so a fan-out caller parameterized by project_root that
    passes a fixed literal label collapses every root onto one key. Because
    ``note_fanout_success`` *pops* that key, a healthy root's success in the
    same poll cycle clears a broken root's open streak, re-arming the opening
    WARNING (plus a 'recovered' WARNING) every cycle — the exact sustained
    flood the transition-only policy exists to prevent.
    """

    def test_composes_base_and_project_basename(self):
        from dashboard.data.mcp_fanout import fanout_label

        assert fanout_label('fetch_tasks', '/home/leo/src/dark-factory') == (
            'fetch_tasks[dark-factory]'
        ), 'must generalize scheduler.py\'s existing base[label] shape'

    def test_trailing_slash_is_stripped(self):
        from dashboard.data.mcp_fanout import fanout_label

        assert fanout_label('fetch_tasks', '/a/b/') == 'fetch_tasks[b]', (
            'a configured root with a trailing slash must not yield an empty label'
        )

    def test_pathlib_path_yields_the_same_label_as_the_equivalent_str(self):
        from pathlib import Path

        from dashboard.data.mcp_fanout import fanout_label

        # Callers pass both: metrics passes a Path-derived str, scheduler a Path.
        assert fanout_label('list_tickets', Path('/a/b')) == fanout_label(
            'list_tickets', '/a/b'
        ) == 'list_tickets[b]'

    def test_root_without_a_basename_falls_back_to_the_full_string(self):
        from dashboard.data.mcp_fanout import fanout_label

        assert fanout_label('fetch_tasks', '/') == 'fetch_tasks[/]', (
            'a basename-less root must not degrade to a content-free "[]"'
        )

    def test_distinct_roots_produce_distinct_labels(self):
        from dashboard.data.mcp_fanout import fanout_label

        # This is the property the throttle key actually depends on.
        assert fanout_label('fetch_tasks', '/srv/proj-a') != fanout_label(
            'fetch_tasks', '/srv/proj-b'
        )

    def test_label_is_composed_from_the_single_project_label_definition(self):
        """``fanout_label`` must delegate the basename rule, not re-derive it.

        ``project_label`` is the one definition ``active_tasks._project_label``
        and ``redux_api._project_label`` are meant to collapse onto; a future
        edit that inlines the rule here again would silently re-fork it.
        """
        from pathlib import Path

        from dashboard.data.mcp_fanout import fanout_label, project_label

        for root in ('/home/leo/src/dark-factory', '/a/b/', '/', Path('/srv/proj-a')):
            assert fanout_label('fetch_tasks', root) == f'fetch_tasks[{project_label(root)}]'

    def test_same_basename_roots_share_a_label_by_design(self):
        """Discrimination is by basename, so same-named roots still collapse.

        Documented, inherited behaviour rather than an oversight: scheduler's
        ``label_to_root`` and redux_api's per-project payload already key on the
        same basename, so diverging to full paths here alone would make this one
        log label inconsistent with every other project label the UI renders.
        Pinned so the assumption is executable, not only prose in the docstring.
        """
        from dashboard.data.mcp_fanout import fanout_label

        assert fanout_label('fetch_tasks', '/srv/team-a/app') == fanout_label(
            'fetch_tasks', '/srv/team-b/app'
        ) == 'fetch_tasks[app]'

    async def test_distinct_roots_keep_independent_streaks_on_one_url(self, caplog):
        """End-to-end: the discriminated labels really do decouple the streaks."""
        from dashboard.data.mcp_fanout import fanout_label

        url = 'http://shared-fused-memory:8765'

        async def failing(_url):
            raise httpx.ConnectError('refused')

        async def healthy(_url):
            return 'ok'

        with caplog.at_level(logging.DEBUG, logger='dashboard.data.mcp_fanout'):
            for _ in range(3):
                await first_success(
                    [url], failing,
                    log_label=fanout_label('fetch_tasks', '/srv/proj-a'),
                    offline_result=_offline_result,
                )
                await first_success(
                    [url], healthy,
                    log_label=fanout_label('fetch_tasks', '/srv/proj-b'),
                    offline_result=_offline_result,
                )

        warnings = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'dashboard.data.mcp_fanout'
        ]
        assert len(warnings) == 1, (
            f"root B's success must not pop root A's streak, got {warnings}"
        )
        assert 'proj-a' in warnings[0], (
            f'the operator must be able to tell which root is down, got {warnings[0]}'
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


class TestTTLCacheEvictsExpiredKeys:
    """A high-cardinality key space must not grow the store without bound.

    Regression for the task-3857 review finding. ``fetch_tasks``' cache key
    includes the paging ``offset``, and ``active_tasks``' terminal-window
    caller computes it from a live task count that grows on every completion.
    Because ``TTLCache`` evicted nothing, each retired offset permanently
    retained a shaped-task list (rows carrying description/details/metadata)
    plus an ``asyncio.Lock``.
    """

    @pytest.mark.parametrize('n_keys', [50, 200, 800])
    async def test_store_size_is_independent_of_how_many_keys_are_minted(
        self, monkeypatch, n_keys
    ):
        """Resident size is set by the eviction horizon, NOT by cardinality.

        This is the assertion that actually discriminates a bounded store from
        a leaking one: sweeping 50, 200 and 800 one-shot keys must all settle
        at the SAME small resident size. A leak would grow with *n_keys*, and
        a quantize-the-offset fix (rejected during this review) would still
        grow with it — just more slowly.

        Eviction is LAZY — it runs on a cold miss, not on a background timer —
        so the newest few entries always post-date the last sweep. The bound
        is therefore a small constant, not zero.
        """
        import dashboard.data.mcp_fanout as fanout_mod

        cache: TTLCache[list] = TTLCache(ttl_seconds=lambda: 20.0)

        clock = {'t': 0.0}
        monkeypatch.setattr(
            fanout_mod, 'time', types.SimpleNamespace(monotonic=lambda: clock['t'])
        )

        # Each iteration mints a NEW key and never revisits it — exactly the
        # shape of an offset that advances with every completed task.
        for i in range(n_keys):
            await cache.get_or_refresh(f'root|s=done|p=400|o={i}', _four_hundred_rows)
            clock['t'] += 30.0  # > TTL, so the previous key is retired

        # horizon 80s / 30s per key => at most ceil(80/30) + 1 == 4 entries can
        # post-date the last sweep. Pinned as an absolute constant so the test
        # fails if the store ever starts tracking n_keys.
        assert len(cache._store) <= 4, (
            f'{n_keys} distinct one-shot keys must not accumulate store slots; '
            f'got {len(cache._store)} — the store is tracking cardinality'
        )
        assert len(cache._locks) <= 4, (
            f'retired keys must not retain locks forever; got {len(cache._locks)}'
        )
        # No survivor may be arbitrarily old: everything still resident was
        # either swept-and-kept or stored after the last sweep.
        oldest = min(stamp for stamp, _ in cache._store.values())
        assert clock['t'] - oldest <= 20.0 * TTLCache._EVICTION_TTL_MULTIPLE + 30.0, (
            'a resident entry is older than one horizon plus one sweep interval'
        )

    async def test_eviction_never_drops_a_still_servable_entry(self, monkeypatch):
        """Entries inside the TTL survive a sweep — eviction is memory-only."""
        import dashboard.data.mcp_fanout as fanout_mod

        cache: TTLCache[list] = TTLCache(ttl_seconds=lambda: 20.0)
        clock = {'t': 0.0}
        monkeypatch.setattr(
            fanout_mod, 'time', types.SimpleNamespace(monotonic=lambda: clock['t'])
        )

        await cache.get_or_refresh('keep-me', _four_hundred_rows)
        clock['t'] += 1.0
        # A cold miss on another key triggers the sweep.
        await cache.get_or_refresh('other', _four_hundred_rows)

        assert cache.get_fresh('keep-me') is not None, (
            'a sweep must not evict an entry still inside its TTL'
        )

    async def test_sweep_does_not_disturb_an_in_flight_single_flight(self, monkeypatch):
        """A lock held by an in-flight refresh survives a concurrent sweep."""
        import asyncio

        import dashboard.data.mcp_fanout as fanout_mod

        cache: TTLCache[list] = TTLCache(ttl_seconds=lambda: 20.0)
        clock = {'t': 0.0}
        monkeypatch.setattr(
            fanout_mod, 'time', types.SimpleNamespace(monotonic=lambda: clock['t'])
        )

        started = asyncio.Event()
        release = asyncio.Event()
        calls = {'n': 0}

        async def _slow() -> list:
            calls['n'] += 1
            started.set()
            await release.wait()
            return [{'id': 1}]

        # Park a refresh for 'slow' holding its lock...
        task_a = asyncio.create_task(cache.get_or_refresh('slow', _slow))
        await started.wait()
        # ...then queue a second waiter on the SAME key, and age the clock so a
        # sweep on an unrelated cold miss would consider stale keys evictable.
        task_b = asyncio.create_task(cache.get_or_refresh('slow', _slow))
        await asyncio.sleep(0)
        clock['t'] += 1000.0
        await cache.get_or_refresh('unrelated', _four_hundred_rows)

        release.set()
        await task_a
        await task_b
        assert calls['n'] == 1, (
            'the sweep must not break single-flight: the queued waiter should '
            f'have reused the in-flight result, but refresh ran {calls["n"]} times'
        )


async def _four_hundred_rows() -> list:
    """Stand-in for a terminal-window page: 400 rows carrying heavy fields."""
    return [{'id': i, 'description': 'x' * 64} for i in range(400)]


class TestTTLCacheKeepsLocksWithQueuedWaiters:
    """A lock with a queued waiter survives a sweep (task 3857 amendment).

    ``_evict_expired`` used to drop any lock whose ``locked()`` read False and
    whose key was absent from ``_store``. ``asyncio.Lock.release()`` clears
    ``_locked`` and merely SCHEDULES the first waiter's future, so between the
    release and the waiter actually resuming there is a real window in which a
    lock in active use reads as idle. The key is absent from ``_store`` during
    exactly the case that matters — an outage, where ``cache_ok`` stores
    nothing — so a concurrent cold miss sweeping in that window would delete
    the lock, the next caller would mint a fresh one, and single-flight would
    be silently lost.
    """

    async def test_released_but_not_yet_resumed_lock_is_not_reclaimed(self):
        cache: TTLCache[list] = TTLCache(ttl_seconds=lambda: 20.0)

        lock = cache._locks.setdefault('hot-key', asyncio.Lock())
        await lock.acquire()
        waiter = asyncio.create_task(lock.acquire())
        await asyncio.sleep(0)  # let the waiter reach acquire() and queue
        assert lock.locked()

        lock.release()  # schedules the waiter; does NOT resume it yet

        # The precise window the old predicate could not see.
        assert not lock.locked(), 'precondition: release() clears the flag'
        assert getattr(lock, '_waiters', None), 'precondition: a waiter is queued'
        assert 'hot-key' not in cache._store, 'precondition: nothing cacheable stored'

        cache._evict_expired()

        assert 'hot-key' in cache._locks, (
            'a lock with a queued waiter must survive the sweep — dropping it '
            'lets the next caller mint a second lock and refresh concurrently '
            'with the waiter'
        )
        assert cache._locks['hot-key'] is lock, (
            'the surviving lock must be the SAME object the waiter is queued on'
        )

        await waiter
        lock.release()

    async def test_a_genuinely_idle_lock_is_still_reclaimed(self):
        """The waiter probe must not turn the sweep into a no-op."""
        cache: TTLCache[list] = TTLCache(ttl_seconds=lambda: 20.0)

        idle = cache._locks.setdefault('cold-key', asyncio.Lock())
        assert not idle.locked()
        assert not getattr(idle, '_waiters', None)

        cache._evict_expired()

        assert 'cold-key' not in cache._locks, (
            'an unheld, unawaited lock for an absent key is still reclaimable'
        )


class TestTTLCacheBoundedLockAcquisition:
    """A never-returning refresh must not wedge every later caller of that key.

    Regression for the 2026-08-27 incident: a refresh that never returned
    (parked forever inside httpcore) held the
    ``/home/leo/src/reify|s=*|p=*|o=0`` key's lock for 19.8h with 7 waiters
    queued behind it, while the other 8 TTLCache-backed roots stayed healthy
    throughout.

    Every test here monkeypatches the new ``_LOCK_ACQUIRE_TIMEOUT_SECONDS``
    module constant down to a short REAL value and relies on the real clock —
    no fake clock anywhere in this class. The wedging refresh stub parks on a
    genuinely unresolved ``asyncio.Event`` that is NEVER set; a sleep-based
    stub would pass against the current (unbounded) code and prove nothing.
    """

    @staticmethod
    def _wedging_refresh():
        """Counting refresh stub: call #1 wedges forever, call #2+ returns 'value'."""
        entered = asyncio.Event()
        wedged = asyncio.Event()
        calls = {'n': 0}

        async def _refresh():
            calls['n'] += 1
            if calls['n'] == 1:
                entered.set()
                await wedged.wait()  # never set — genuinely unresolved
                raise AssertionError('unreachable: the wedged event is never set')
            return 'value'

        return _refresh, entered, calls

    async def _wedge_key(self, cache, key='k'):
        """Start a caller that wedges *key* forever; return (task, refresh, calls)."""
        refresh, entered, calls = self._wedging_refresh()
        task = asyncio.create_task(cache.get_or_refresh(key, refresh))
        await entered.wait()
        return task, refresh, calls

    @staticmethod
    async def _unwedge(task):
        """Cancel a still-parked wedging task and confirm no orphaned task remains."""
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_a_never_returning_refresh_does_not_wedge_the_next_caller_of_that_key(
        self, monkeypatch
    ):
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)
        first_task, refresh, calls = await self._wedge_key(cache)

        try:
            # No exception -- in particular no bare TimeoutError -- may escape
            # to the caller. On current (unbounded) code this outer wait_for is
            # the RED harness: it fires after 5s because get_or_refresh itself
            # never returns (the lock is never released).
            second = await asyncio.wait_for(
                cache.get_or_refresh('k', refresh), timeout=5.0
            )
        finally:
            await self._unwedge(first_task)

        assert second == 'value'
        assert calls['n'] == 2, 'expected the wedged call plus exactly one bypass call'

    async def test_the_bypassed_refresh_is_cached_so_later_callers_never_touch_the_lock(
        self, monkeypatch
    ):
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)
        first_task, refresh, calls = await self._wedge_key(cache)

        try:
            await asyncio.wait_for(cache.get_or_refresh('k', refresh), timeout=5.0)

            assert cache.get_fresh('k') is not None, (
                'the bypassed refresh must be stored -- nothing was ever stored '
                'for the wedged key during the incident, which is why every '
                'later caller kept queueing'
            )

            third = await cache.get_or_refresh('k', refresh)
            assert third == 'value'
            assert calls['n'] == 2, (
                'a later caller must be served from the store, not touch the '
                'still-wedged lock or run another refresh'
            )
        finally:
            await self._unwedge(first_task)

    async def test_a_wedged_key_degrades_only_itself(self, monkeypatch):
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)
        first_task, refresh_a, _ = await self._wedge_key(cache, key='a')

        b_calls = {'n': 0}

        async def refresh_b():
            b_calls['n'] += 1
            return 'b-value'

        try:
            b_result = await asyncio.wait_for(
                cache.get_or_refresh('b', refresh_b), timeout=5.0
            )
            a_second = await asyncio.wait_for(
                cache.get_or_refresh('a', refresh_a), timeout=5.0
            )
        finally:
            await self._unwedge(first_task)

        assert b_result == 'b-value'
        assert b_calls['n'] == 1, (
            "an unrelated key must resolve normally, with no bypass, while "
            "key 'a' is wedged"
        )
        assert a_second == 'value', "key 'a''s second caller must still return"

    def test_lock_acquire_timeout_is_a_finite_named_module_constant(self):
        import dashboard.data.mcp_fanout as fanout_mod

        timeout = fanout_mod._LOCK_ACQUIRE_TIMEOUT_SECONDS
        assert isinstance(timeout, float)
        assert 0 < timeout < 120, (
            'a future None (or non-positive/unbounded) value would silently '
            'restore the unbounded wait this task exists to remove'
        )


class TestTTLCacheBypassRechecksFreshness:
    """A timed-out waiter must re-check freshness BEFORE spending a bypass refresh.

    The timeout window is exactly when another caller can have filled the
    entry — a bypass that skipped the re-check would spend a duplicate MCP
    round trip (and clobber a newer value with an older one) for no benefit.
    This mirrors the post-lock double-check on the normal acquisition path:
    freshness first, refresh second, never reordered.
    """

    async def test_a_timed_out_waiter_serves_a_value_that_landed_while_it_waited(
        self, monkeypatch
    ):
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)

        # Guarantee the acquisition times out — the same reach-into-privates
        # idiom TestTTLCacheKeepsLocksWithQueuedWaiters already uses.
        lock = cache._locks.setdefault('k', asyncio.Lock())
        await lock.acquire()

        calls = {'n': 0}

        async def _refresh():
            calls['n'] += 1
            return 'refreshed'

        # The key must be COLD when the waiter starts, so its own
        # top-of-function freshness check misses and it genuinely reaches the
        # bounded lock wait rather than being served by the pre-existing
        # lock-free fast path (which would make this test pass vacuously,
        # regardless of the re-check under test). Seeding only after it is
        # parked is what stands in for "a third party stored while this
        # caller was parked".
        waiter = asyncio.create_task(cache.get_or_refresh('k', _refresh))
        await asyncio.sleep(0)  # let it reach the lock and start waiting
        cache._store['k'] = (time.monotonic(), 'landed')

        try:
            result = await asyncio.wait_for(waiter, 5.0)
        finally:
            lock.release()

        assert result == 'landed'
        assert calls['n'] == 0, (
            'a timed-out waiter must serve the value already in hand rather '
            'than issuing another MCP round trip'
        )

    async def test_a_timed_out_waiter_with_no_stored_value_still_runs_its_own_refresh(
        self, monkeypatch
    ):
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)

        lock = cache._locks.setdefault('k', asyncio.Lock())
        await lock.acquire()

        calls = {'n': 0}

        async def _refresh():
            calls['n'] += 1
            return 'refreshed'

        try:
            result = await asyncio.wait_for(cache.get_or_refresh('k', _refresh), 5.0)
        finally:
            lock.release()

        assert result == 'refreshed'
        assert calls['n'] == 1, (
            'the re-check must not be "fixed" into an unconditional cache '
            'read that starves a genuinely cold key'
        )


async def _named_refresh() -> str:
    """Module-level refresh stub whose ``__qualname__`` is a stable, asserted string.

    All eight live TTLCache call sites pass a locally-defined closure
    (``fetch_tasks.<locals>._refresh``, ...), so the qualname is what
    disambiguates *which* cache instance bypassed without a ``name=``
    constructor argument. This module-level stand-in gives the tests below a
    fixed, predictable qualname to assert against.
    """
    return 'value'


class TestTTLCacheLockBypassLogging:
    """A lock-acquisition bypass must be VISIBLE, but not a log flood.

    A silent bypass would hide the next occurrence — the same invisibility
    that let the 2026-08-27 incident run 19.8h unnoticed. Mirrors the
    transition-only WARNING policy already established for fan-out failures
    (see TestFanoutFailureThrottling above): WARNING on the first bypass of a
    streak and every ``_LOCK_BYPASS_REWARN_EVERY``-th thereafter, DEBUG for
    the repeats, WARNING again on recovery.
    """

    @staticmethod
    async def _force_bypass(cache, key, refresh):
        """Force exactly one bypass for *key*: hold its lock, call get_or_refresh, release.

        Always passes ``cache_ok=False`` so the key never actually warms up —
        a ``cache_ok=True`` bypass would store a value and let the very next
        call take the lock-free warm fast path, never touching the lock (or
        this logging) again.
        """
        lock = cache._locks.setdefault(key, asyncio.Lock())
        await lock.acquire()
        try:
            return await asyncio.wait_for(
                cache.get_or_refresh(key, refresh, cache_ok=lambda v: False), 5.0
            )
        finally:
            lock.release()

    async def test_first_bypass_warns_naming_the_key_and_the_refresh(
        self, monkeypatch, caplog
    ):
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)

        with caplog.at_level(logging.WARNING, logger='dashboard.data.mcp_fanout'):
            await self._force_bypass(cache, 'my-key', _named_refresh)

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'dashboard.data.mcp_fanout'
        ]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert 'my-key' in message, f'warning must name the key, got: {message}'
        assert _named_refresh.__qualname__ in message, (
            f'warning must name the refresh callable, got: {message}'
        )

    async def test_repeated_bypasses_for_one_key_do_not_flood_warning(
        self, monkeypatch, caplog
    ):
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)

        with caplog.at_level(logging.DEBUG, logger='dashboard.data.mcp_fanout'):
            for _ in range(5):
                await self._force_bypass(cache, 'k', _named_refresh)

        records = [r for r in caplog.records if r.name == 'dashboard.data.mcp_fanout']
        warnings = [r for r in records if r.levelno == logging.WARNING]
        debugs = [r for r in records if r.levelno == logging.DEBUG]

        assert len(warnings) == 1, (
            f'a sustained bypass streak must warn exactly once, got '
            f'{[r.getMessage() for r in warnings]}'
        )
        assert len(debugs) == 4, (
            f'the 4 repeats must still be recorded at DEBUG, got '
            f'{[r.getMessage() for r in debugs]}'
        )

    async def test_a_long_bypass_streak_still_heartbeats_at_warning(
        self, monkeypatch, caplog
    ):
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        monkeypatch.setattr(fanout_mod, '_LOCK_BYPASS_REWARN_EVERY', 3)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)

        with caplog.at_level(logging.WARNING, logger='dashboard.data.mcp_fanout'):
            for _ in range(3):
                await self._force_bypass(cache, 'k', _named_refresh)

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'dashboard.data.mcp_fanout'
        ]
        assert len(warnings) == 2, (
            f'an outage outliving log rotation must still heartbeat, got '
            f'{[r.getMessage() for r in warnings]}'
        )

    async def test_streaks_are_tracked_per_key(self, monkeypatch, caplog):
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)

        with caplog.at_level(logging.WARNING, logger='dashboard.data.mcp_fanout'):
            await self._force_bypass(cache, 'a', _named_refresh)
            await self._force_bypass(cache, 'a', _named_refresh)  # demoted to DEBUG
            await self._force_bypass(cache, 'b', _named_refresh)  # independent streak

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'dashboard.data.mcp_fanout'
        ]
        assert len(warnings) == 2, (
            f"each key's streak reports independently, got "
            f'{[r.getMessage() for r in warnings]}'
        )

    async def test_a_recovered_key_logs_a_closing_warning_and_re_arms(
        self, monkeypatch, caplog
    ):
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)
        key = 'recovering-key'

        with caplog.at_level(logging.WARNING, logger='dashboard.data.mcp_fanout'):
            await self._force_bypass(cache, key, _named_refresh)
            await self._force_bypass(cache, key, _named_refresh)  # demoted to DEBUG

            # Nobody holds the lock now, so this acquires NORMALLY. cache_ok
            # stays False so the key remains cold and the NEXT call also
            # goes through the lock rather than the warm fast path.
            await cache.get_or_refresh(key, _named_refresh, cache_ok=lambda v: False)

            # A further normal acquisition logs nothing more (streak already closed).
            await cache.get_or_refresh(key, _named_refresh, cache_ok=lambda v: False)

            # A later bypass re-arms the opening WARNING.
            await self._force_bypass(cache, key, _named_refresh)

        messages = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'dashboard.data.mcp_fanout'
        ]
        assert len(messages) == 3, f'expected open + recovery + re-open, got {messages}'
        assert key in messages[1] and 'recovered' in messages[1], (
            f'the streak needs a visible closing bracket naming the key, got {messages[1]}'
        )

    async def test_clear_resets_bypass_streak_state(self, monkeypatch, caplog):
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)

        await self._force_bypass(cache, 'k', _named_refresh)
        cache.clear()
        caplog.clear()  # drop the opening WARNING captured above

        with caplog.at_level(logging.WARNING, logger='dashboard.data.mcp_fanout'):
            await self._force_bypass(cache, 'k', _named_refresh)

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.name == 'dashboard.data.mcp_fanout'
        ]
        assert len(warnings) == 1, (
            'clear() must drop open bypass streaks so a later bypass warns '
            f'again, got {[r.getMessage() for r in warnings]}'
        )


class TestTTLCacheBypassStreaksPruning:
    """A bypass-streak counter must not outlive both its lock and its value.

    Before this, ``_bypass_streaks`` was only ever cleared wholesale by
    ``clear()`` -- nothing pruned individual keys. ``_note_lock_acquired``
    pops a key only on a NORMAL cold acquisition, and a key that self-heals
    via a bypassed store is thereafter served from the warm path, so its
    streak entry would survive indefinitely. This contradicts the class
    docstring's "Key space is bounded by disuse, not by cardinality" claim
    for this third per-key dict (task 3857 review, extended here).
    """

    async def test_a_bypass_streak_is_pruned_once_its_lock_and_value_are_both_gone(
        self, monkeypatch
    ):
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)

        # Force one bypass for 'k' (cache_ok=False so nothing is stored and
        # the key stays cold), leaving an open streak entry behind.
        lock = cache._locks.setdefault('k', asyncio.Lock())
        await lock.acquire()
        try:
            await asyncio.wait_for(
                cache.get_or_refresh('k', _named_refresh, cache_ok=lambda v: False),
                5.0,
            )
        finally:
            lock.release()

        assert cache._bypass_streaks.get('k') == 1, 'precondition: an open streak exists'
        assert not lock.locked(), 'precondition: the lock was released'
        assert not getattr(lock, '_waiters', None), 'precondition: nobody is queued'
        assert 'k' not in cache._store, 'precondition: nothing cacheable was stored'

        cache._evict_expired()

        assert 'k' not in cache._locks, 'precondition: the idle lock is reclaimed'
        assert 'k' not in cache._bypass_streaks, (
            'a bypass streak for a key with no live lock and no cached value '
            'must be pruned, not left to persist forever'
        )

    async def test_a_streak_survives_while_its_lock_is_still_live(self, monkeypatch):
        """The prune must not fire early and demote an ACTIVE streak's next warning."""
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)

        lock = cache._locks.setdefault('k', asyncio.Lock())
        await lock.acquire()
        try:
            await asyncio.wait_for(
                cache.get_or_refresh('k', _named_refresh, cache_ok=lambda v: False),
                5.0,
            )
            assert cache._bypass_streaks.get('k') == 1

            # An unrelated cold miss triggers a sweep while 'k's lock is
            # STILL held by this test.
            await cache.get_or_refresh('other', _named_refresh)

            assert 'k' in cache._bypass_streaks, (
                "a streak whose lock is still live must survive the sweep -- "
                "pruning it here would demote the next bypass's WARNING to "
                "DEBUG for a wedge that never actually recovered"
            )
        finally:
            lock.release()


class TestTTLCacheBoundsBypassConcurrency:
    """Concurrent bypasses for ONE key must not accumulate without bound.

    Before this, every timed-out caller for a wedged key started its OWN
    unlocked refresh: during a TRUE wedge (a refresh that never returns --
    this task's own incident), callers accumulate at the dashboard's poll
    rate (~2-3s) while each pins a connection on the shared httpx client,
    which risks pool saturation (httpx.PoolTimeout) for UNRELATED endpoints
    -- the opposite of what test_a_wedged_key_degrades_only_itself asserts
    (key isolation, not resource isolation). Bounding bypass concurrency to
    one shared refresh per key closes this without touching any call site.
    """

    async def test_many_timed_out_callers_for_one_key_share_a_single_refresh(
        self, monkeypatch
    ):
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)
        calls = {'n': 0}
        entered = asyncio.Event()
        release = asyncio.Event()

        async def _refresh():
            calls['n'] += 1
            entered.set()
            await release.wait()
            return 'value'

        # Hold the key's lock directly so the FIRST caller below times out
        # acquiring it -- the same idiom TestTTLCacheBypassRechecksFreshness
        # already uses -- and becomes the bypass creator through the real
        # get_or_refresh -> lock-timeout -> _bypass_refresh path.
        lock = cache._locks.setdefault('k', asyncio.Lock())
        await lock.acquire()

        try:
            first = asyncio.create_task(cache.get_or_refresh('k', _refresh))
            await asyncio.wait_for(entered.wait(), timeout=5.0)

            # Raise the bound back up now that the shared bypass has
            # started: the lock-timeout phase (forcing the first caller to
            # time out acquiring the REAL lock) and this
            # sharing-observation phase (proving 19 MORE timed-out callers
            # join that ONE refresh) want different bounds.
            #
            # MEASURED HAZARD: launching the other 19 through
            # get_or_refresh (racing the SAME real lock, as an earlier
            # version of this test did) does NOT work even with the raise
            # placed immediately after entered.wait() fires. asyncio.
            # create_task only SCHEDULES; none of the 19 gets to run its
            # own `await asyncio.wait_for(lock.acquire(), ...)` until this
            # coroutine yields, and by then they all reach their OWN
            # lock-timeout in the SAME event-loop batch the FIRST caller's
            # timeout fired in -- entirely before `entered` is even set
            # (that requires the shared bypass task to be scheduled AND
            # run, which happens one loop turn later). So all 19 read the
            # bound and lock in their join deadline BEFORE this line can
            # possibly run, reproducibly costing one extra shared refresh
            # (calls['n'] == 2, not 1) regardless of how soon after
            # entered.wait() the raise happens. Repro + fix measured
            # directly against this branch.
            #
            # Calling the private _bypass_refresh directly, AFTER the
            # raise, sidesteps the batching: it is exactly what
            # get_or_refresh's own next line runs the instant a caller
            # times out on the lock, so it faithfully simulates "19 more
            # timed-out callers of this key" without re-racing the
            # lock-timeout that isn't what this test is pinning. Works
            # because the constant is read as a module global at call time
            # (the _FANOUT_REWARN_EVERY idiom), so the raise takes effect
            # on the very next read.
            monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 5.0)

            more = [
                asyncio.create_task(
                    cache._bypass_refresh('k', _refresh, lambda v: True)
                )
                for _ in range(19)
            ]
            # Let every one of the 19 timed-out callers get a chance to
            # join the one shared bypass refresh rather than start its own.
            await asyncio.sleep(0.2)
            assert calls['n'] == 1, (
                f'expected exactly one bypass refresh to have started for '
                f"20 concurrent timed-out callers, got {calls['n']}"
            )

            release.set()
            results = await asyncio.wait_for(
                asyncio.gather(first, *more), timeout=5.0
            )
        finally:
            lock.release()

        assert results == ['value'] * 20
        assert calls['n'] == 1, (
            'concurrent timed-out callers for one key must share a single '
            'bypass refresh, not accumulate one refresh per caller'
        )


class TestTTLCacheBoundedBypassInheritance:
    """A caller must not inherit another caller's stall past the bound.

    Reviewer finding (mcp_fanout.py:627-639, pre-fix): ``_bypass_refresh``
    joined a running bypass with ``return await asyncio.shield(running)`` --
    no deadline -- and ``_bypass_tasks[key]`` was cleared only by the
    task's own done-callback. So if the SHARED bypass also never returns
    (the likely case: the wedge is endpoint/session-level, and
    ``first_success`` invalidates a cached MCP session only on an
    *exception*, never on a hang), caller #3 and every caller after it
    inherits a dead task forever.

    REPRODUCED against the pre-fix code before writing this class: bound
    monkeypatched to 0.05, refresh parked on a never-set Event -- caller #3
    never returned, ``refresh`` was called exactly 2 times, and
    ``cache._bypass_tasks['k']`` stayed pending. Confirmed.

    Invariant pinned here: a caller waits at most
    ``_LOCK_ACQUIRE_TIMEOUT_SECONDS`` on the lock, and at most that again on
    any single refresh it did not start -- and never on one already known
    to have outlived that bound. Every method monkeypatches
    ``_LOCK_ACQUIRE_TIMEOUT_SECONDS`` to 0.05 and uses the REAL clock.
    These tests deliberately create tasks that never complete -- on the
    pre-fix code the pile is larger (a joiner that should have been
    promoted instead hangs forever too) -- so teardown cancels every
    outstanding task via ``_cancel_all`` rather than assuming they finished.
    """

    @staticmethod
    def _task_of(entry):
        """Return the Task inside a ``_bypass_tasks`` entry, tuple or bare.

        Handles both the pre-step-8 shape (``dict[str, Task]``) and the
        post-step-8 shape (``dict[str, tuple[float, Task]]``), so teardown
        code written once works unchanged against either version of the
        map. Passes a bare ``None`` through unchanged.
        """
        return entry[1] if isinstance(entry, tuple) else entry

    @staticmethod
    async def _cancel_all(*tasks):
        """Cancel every still-pending task and await all of them.

        Safe with a mix of pending, already-done, and ``None`` entries --
        cancelling a completed task is a no-op and ``None`` is dropped, so
        callers can pass optional captured handles unconditionally.
        """
        live = [t for t in tasks if t is not None]
        for t in live:
            t.cancel()
        if live:
            await asyncio.gather(*live, return_exceptions=True)

    @staticmethod
    def _always_wedging_refresh():
        """Counting refresh stub: EVERY call parks on a never-set Event forever.

        Unlike ``TestTTLCacheBoundedLockAcquisition._wedging_refresh`` (whose
        call #2+ resolves), nothing here ever resolves on its own -- these
        tests pin the invariant in the worst case, where a caller promoted
        off a dead shared bypass still cannot itself return.
        """
        entered = asyncio.Event()
        wedged = asyncio.Event()
        calls = {'n': 0}

        async def _refresh():
            calls['n'] += 1
            if calls['n'] == 1:
                entered.set()
            await wedged.wait()  # never set -- genuinely unresolved
            raise AssertionError('unreachable: the wedged event is never set')

        return _refresh, entered, calls

    async def test_a_bypass_that_never_returns_does_not_wedge_the_next_caller(
        self, monkeypatch
    ):
        """The reviewer's explicitly requested regression test.

        Calls #1 and #2 park on a never-set Event; call #3 returns 'value'
        immediately -- the realistic production shape, since each fresh MCP
        call carries its own per-HTTP-request budget (``mcp_tool_call``
        threads ``timeout`` into every httpx post), so a fresh attempt CAN
        return while an older one stays wedged.
        """
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)
        calls = {'n': 0}
        entered = asyncio.Event()
        wedged = asyncio.Event()  # never set

        async def _refresh():
            calls['n'] += 1
            if calls['n'] == 1:
                entered.set()
            if calls['n'] <= 2:
                await wedged.wait()
                raise AssertionError('unreachable: the wedged event is never set')
            return 'value'

        holder = asyncio.create_task(cache.get_or_refresh('k', _refresh))
        await asyncio.wait_for(entered.wait(), timeout=5.0)

        second = asyncio.create_task(cache.get_or_refresh('k', _refresh))
        superseded = None
        try:
            # Let caller #2 time out acquiring the lock, become the bypass
            # creator, and wedge inside the refresh it started.
            await asyncio.sleep(0.2)
            superseded = cache._bypass_tasks.get('k')
            assert superseded is not None, (
                'precondition: caller #2 must have registered a bypass'
            )

            third = await asyncio.wait_for(
                cache.get_or_refresh('k', _refresh), timeout=5.0
            )

            assert third == 'value'
            assert cache.get_fresh('k') == 'value'
        finally:
            current = cache._bypass_tasks.get('k')
            await self._cancel_all(
                holder, second, self._task_of(superseded), self._task_of(current),
            )

    async def test_a_caller_never_waits_past_the_bound_on_a_refresh_it_did_not_start(
        self, monkeypatch
    ):
        """The reviewer's exact repro shape: EVERY call wedges.

        Caller #3 must still be promoted off the dead shared task within one
        bound, even though nothing it can reach ever returns -- so caller #3
        itself still does not RETURN in this test either. That is correct
        and out of this layer's reach: its own (promoted) refresh never
        returns, which is the pre-existing cold-caller exposure
        ``get_or_refresh`` cannot remove while it stays a total function (no
        call site can observe a raised TimeoutError).
        """
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)
        _refresh, entered, calls = self._always_wedging_refresh()

        holder = asyncio.create_task(cache.get_or_refresh('k', _refresh))
        await asyncio.wait_for(entered.wait(), timeout=5.0)

        second = asyncio.create_task(cache.get_or_refresh('k', _refresh))
        superseded = None
        third = None
        try:
            await asyncio.sleep(0.2)  # let caller #2 time out and become the bypass creator
            superseded = cache._bypass_tasks.get('k')
            assert superseded is not None, (
                'precondition: caller #2 must have registered a bypass'
            )

            third = asyncio.create_task(cache.get_or_refresh('k', _refresh))
            await asyncio.sleep(0.25)  # ~5 bound-windows -- plenty for the promotion

            assert calls['n'] == 3, (
                'caller #3 must be promoted to running its OWN refresh '
                f"instead of parking on the dead shared task forever, got "
                f"{calls['n']}"
            )
            current = cache._bypass_tasks.get('k')
            assert current is None or self._task_of(current) is not self._task_of(
                superseded
            ), (
                'the map must no longer track the task caller #2 created -- '
                'it is provably over-age and nobody may still be waiting on it'
            )
        finally:
            current = cache._bypass_tasks.get('k')
            await self._cancel_all(
                holder, second, third,
                self._task_of(superseded), self._task_of(current),
            )

    async def test_a_re_armed_bypass_is_counted_and_logged(self, monkeypatch, caplog):
        """The reviewer's "count each such re-arm through _note_lock_bypass".

        Abandoning an over-age shared bypass is not a repeat of the
        ORIGINAL lock-acquisition timeout -- the lock did not time out
        again, the inherited BYPASS did -- so it must log its own,
        differently-worded record, while still going through the same
        per-key streak counter so the existing WARNING/DEBUG throttle
        covers it. The streak is already > 1 by the time the re-arm fires
        here, so the re-arm record lands at DEBUG under that throttle --
        asserted at DEBUG, not WARNING, for that reason.
        """
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)
        _refresh, entered, calls = self._always_wedging_refresh()

        holder = asyncio.create_task(cache.get_or_refresh('k', _refresh))
        await asyncio.wait_for(entered.wait(), timeout=5.0)

        second = asyncio.create_task(cache.get_or_refresh('k', _refresh))
        superseded = None
        third = None
        try:
            with caplog.at_level(logging.DEBUG, logger='dashboard.data.mcp_fanout'):
                await asyncio.sleep(0.2)
                superseded = cache._bypass_tasks.get('k')
                assert superseded is not None, (
                    'precondition: caller #2 must have registered a bypass'
                )

                third = asyncio.create_task(cache.get_or_refresh('k', _refresh))
                await asyncio.sleep(0.25)

            records = [
                r for r in caplog.records if r.name == 'dashboard.data.mcp_fanout'
            ]
            rearm_records = [r for r in records if 'inherited' in r.getMessage()]
            assert rearm_records, (
                'abandoning the over-age shared bypass must log its own '
                'record -- the lock did not time out again, the inherited '
                f'bypass did -- got: {[r.getMessage() for r in records]}'
            )
            assert all(r.levelno == logging.DEBUG for r in rearm_records), (
                'the streak is already > 1 by the time the re-arm fires '
                f'here, so it lands at DEBUG under the existing throttle, '
                f'got levels {[logging.getLevelName(r.levelno) for r in rearm_records]}'
            )
            assert cache._bypass_streaks.get('k', 0) > 2, (
                'the re-arm must be counted through the same per-key streak '
                "counter as the two lock-acquisition timeouts (second's and "
                f"third's), not a separate uncounted path; streak="
                f"{cache._bypass_streaks.get('k')}"
            )
        finally:
            current = cache._bypass_tasks.get('k')
            await self._cancel_all(
                holder, second, third,
                self._task_of(superseded), self._task_of(current),
            )

    async def test_bypass_concurrency_stays_bounded_while_callers_pile_up(
        self, monkeypatch
    ):
        """Fence, not a RED -- must hold both before and after step-8.

        Guards the resource property ``_bypass_refresh`` exists for: even
        once bypass inheritance is bounded (so a wedge causes roughly one
        new refresh per bound-window rather than zero), it must never
        regress to one refresh PER CALLER -- the pre-existing
        one-refresh-per-timed-out-caller shape
        ``TestTTLCacheBoundsBypassConcurrency`` was written to close -- and
        at most one bypass task may be tracked for the key at any instant.
        """
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)
        calls = {'n': 0}
        release = asyncio.Event()

        async def _refresh():
            calls['n'] += 1
            await release.wait()
            return 'value'

        # Hold the key's lock directly so every caller below times out
        # acquiring it -- the same idiom TestTTLCacheBoundsBypassConcurrency
        # already uses.
        lock = cache._locks.setdefault('k', asyncio.Lock())
        await lock.acquire()

        try:
            waiters = [
                asyncio.create_task(cache.get_or_refresh('k', _refresh))
                for _ in range(20)
            ]
            start = time.monotonic()
            await asyncio.sleep(0.2)
            elapsed = time.monotonic() - start

            assert calls['n'] < 20, (
                '20 concurrent timed-out callers must never each start '
                f"their own refresh, got {calls['n']}"
            )
            assert (
                calls['n']
                <= 2 + elapsed / fanout_mod._LOCK_ACQUIRE_TIMEOUT_SECONDS
            ), (
                f"at most about one new refresh per bound-window is "
                f"expected, got {calls['n']} over {elapsed:.3f}s"
            )
            assert len(cache._bypass_tasks) <= 1, (
                'at most one tracked live bypass per key at any instant, '
                f'got {len(cache._bypass_tasks)}'
            )
        finally:
            release.set()
            lock.release()
            await asyncio.wait_for(
                asyncio.gather(*waiters, return_exceptions=True), timeout=5.0
            )



class TestTTLCacheBoundsLiveBypassesPerKey:
    """A wedged key must never accumulate live refreshes without bound.

    Second reviewer finding (task 4789): bounding INHERITANCE made the
    supersession loop create a NEW bypass task roughly once per
    bound-window and never cancel the abandoned one, so during a TRUE
    wedge in-flight refreshes for that key grow forever (~1 per 15s,
    ~4/min). Each parked refresh pins a connection on the process-wide
    ``httpx.AsyncClient``, whose pool is at least 100 connections
    (``app._build_http_limits``, ``_HTTP_MIN_CONNECTIONS``), so after
    ~25 minutes the SHARED pool saturates and every unrelated endpoint
    family starts raising ``httpx.PoolTimeout`` -- converting the
    incident's per-key outage (3 of 14 endpoints dead, 11 healthy for the
    full 19.8h) into a whole-dashboard one.

    ``TestTTLCacheBoundedBypassInheritance::test_bypass_concurrency_stays_bounded_while_callers_pile_up``
    does not catch this: it asserts the RATE
    (``calls['n'] <= 2 + elapsed / bound``) over a single 0.2s window, and
    a rate bound holds forever while the TOTAL still diverges. These tests
    assert the TOTAL, over many bound-windows.

    Every method monkeypatches ``_LOCK_ACQUIRE_TIMEOUT_SECONDS`` to a small
    value and uses the REAL clock. They deliberately create tasks that
    never complete, so teardown cancels everything outstanding.
    """

    @staticmethod
    async def _cancel_all(*tasks):
        live = [t for t in tasks if t is not None]
        for t in live:
            t.cancel()
        if live:
            await asyncio.gather(*live, return_exceptions=True)

    async def test_total_live_refreshes_stay_under_the_cap_across_many_windows(
        self, monkeypatch
    ):
        """The reviewer's explicitly requested regression test.

        Drives many bound-windows' worth of timed-out callers against a
        refresh that NEVER returns, and asserts the total number of started
        refreshes stays at or under ``_MAX_LIVE_BYPASSES_PER_KEY``. On the
        pre-fix code this grows by one per window without limit.
        """
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.02)
        # Pinned explicitly, and asserted against the LITERAL below rather
        # than against the constant: an assertion phrased as
        # `calls['n'] <= fanout_mod._MAX_LIVE_BYPASSES_PER_KEY` is vacuous
        # -- MEASURED, it passes against an effectively-uncapped build
        # because both sides move together, which is exactly the pre-fix
        # code this test exists to fail against.
        monkeypatch.setattr(fanout_mod, '_MAX_LIVE_BYPASSES_PER_KEY', 3)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)
        calls = {'n': 0}
        wedged = asyncio.Event()  # never set -- a genuinely unresolved future

        async def _refresh():
            calls['n'] += 1
            await wedged.wait()
            raise AssertionError('unreachable: the wedged event is never set')

        # Hold the key's lock directly so every caller below times out
        # acquiring it, the same idiom the sibling bypass tests use.
        lock = cache._locks.setdefault('k', asyncio.Lock())
        await lock.acquire()

        waiters = []
        try:
            # ~15 bound-windows, with fresh callers arriving throughout --
            # far more than enough for the pre-fix one-per-window growth to
            # blow past the cap.
            start = time.monotonic()
            while time.monotonic() - start < 0.30:
                waiters.append(
                    asyncio.create_task(cache.get_or_refresh('k', _refresh))
                )
                await asyncio.sleep(0.01)
            elapsed = time.monotonic() - start
            windows = elapsed / fanout_mod._LOCK_ACQUIRE_TIMEOUT_SECONDS

            assert windows >= 5, (
                f'precondition: the run must cover several bound-windows to '
                f'distinguish a TOTAL bound from a RATE bound, got {windows:.1f}'
            )
            assert calls['n'] <= 3, (
                'a wedged key must never start more than 3 live refreshes '
                f"however long the wedge lasts, got {calls['n']} over "
                f'{windows:.1f} bound-windows -- each one pins a connection '
                'on the shared httpx pool'
            )
            live = [t for t in cache._live_bypasses.get('k', []) if not t.done()]
            assert len(live) <= 3, (
                f'the live roster itself must respect the cap, got {len(live)}'
            )
        finally:
            lock.release()
            await self._cancel_all(*waiters, *cache._live_bypasses.get('k', []))

    async def test_the_cap_is_per_key_so_a_wedge_cannot_starve_other_keys(
        self, monkeypatch
    ):
        """Acceptance #2: unaffected keys stay unaffected.

        The cap is the resource-isolation mechanism, so it must not itself
        become a cross-key coupling: a healthy key must still refresh
        normally while a different key sits pinned at its cap.
        """
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.02)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)
        wedged = asyncio.Event()  # never set

        async def _wedging_refresh():
            await wedged.wait()
            raise AssertionError('unreachable: the wedged event is never set')

        async def _healthy_refresh():
            return 'healthy'

        wedged_lock = cache._locks.setdefault('wedged', asyncio.Lock())
        await wedged_lock.acquire()

        waiters = []
        try:
            start = time.monotonic()
            while time.monotonic() - start < 0.20:
                waiters.append(
                    asyncio.create_task(
                        cache.get_or_refresh('wedged', _wedging_refresh)
                    )
                )
                await asyncio.sleep(0.01)

            assert cache._live_bypasses.get('wedged'), (
                'precondition: the wedged key must have live bypasses pinned'
            )

            got = await asyncio.wait_for(
                cache.get_or_refresh('healthy', _healthy_refresh), timeout=5.0
            )
            assert got == 'healthy'
            assert cache.get_fresh('healthy') == 'healthy'
            assert 'healthy' not in cache._live_bypasses, (
                'a key that never timed out must never appear in the live '
                'bypass roster at all'
            )
        finally:
            wedged_lock.release()
            await self._cancel_all(*waiters, *cache._live_bypasses.get('wedged', []))

    async def test_declining_to_re_arm_at_the_cap_is_logged(self, monkeypatch, caplog):
        """A silent cap would hide the next occurrence.

        Same reasoning as the bypass and re-arm records: the 19.8h incident
        ran unnoticed because nothing on the wedged path logged. Reaching
        the cap is a distinct, operator-relevant state -- the key is now
        deliberately NOT getting fresh attempts -- so it gets its own
        worded record, throttled through the same per-key streak counter.
        """
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.02)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)
        wedged = asyncio.Event()  # never set

        async def _refresh():
            await wedged.wait()
            raise AssertionError('unreachable: the wedged event is never set')

        lock = cache._locks.setdefault('k', asyncio.Lock())
        await lock.acquire()

        waiters = []
        try:
            with caplog.at_level(logging.DEBUG, logger='dashboard.data.mcp_fanout'):
                start = time.monotonic()
                while time.monotonic() - start < 0.30:
                    waiters.append(
                        asyncio.create_task(cache.get_or_refresh('k', _refresh))
                    )
                    await asyncio.sleep(0.01)

            records = [
                r for r in caplog.records if r.name == 'dashboard.data.mcp_fanout'
            ]
            # Matched on a phrase from the cap record ITSELF, not the bare
            # word 'cap': _note_lock_bypass interpolates
            # ``refresh.__qualname__`` into every record, and this method's
            # own name contains 'cap', so the looser matcher passed against
            # an uncapped build (MEASURED) -- a vacuous assertion.
            cap_records = [
                r for r in records
                if 'live bypass refreshes' in r.getMessage()
                or 'live-bypass cap' in r.getMessage()
            ]
            assert cap_records, (
                'declining to re-arm at the live-bypass cap must leave its '
                f'own trace, got: {[r.getMessage() for r in records]}'
            )
        finally:
            lock.release()
            await self._cancel_all(*waiters, *cache._live_bypasses.get('k', []))

    async def test_a_finished_bypass_frees_a_cap_slot(self, monkeypatch):
        """The cap counts LIVE work, not lifetime attempts.

        A transiently slow key that recovers must not stay permanently
        capped: once its refreshes complete they stop holding connections,
        so they must stop counting. Pinned because the obvious wrong
        implementation -- a monotonic per-key counter -- passes every
        wedge test above and permanently wedges a key that recovered.
        """
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.02)
        monkeypatch.setattr(fanout_mod, '_MAX_LIVE_BYPASSES_PER_KEY', 3)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)
        calls = {'n': 0}
        release = asyncio.Event()

        async def _refresh():
            calls['n'] += 1
            await release.wait()
            return f"value-{calls['n']}"

        lock = cache._locks.setdefault('k', asyncio.Lock())
        await lock.acquire()

        waiters = []
        try:
            start = time.monotonic()
            while time.monotonic() - start < 0.20:
                waiters.append(
                    asyncio.create_task(cache.get_or_refresh('k', _refresh))
                )
                await asyncio.sleep(0.01)

            assert calls['n'] <= 3

            # Let every in-flight bypass finish, then confirm the roster
            # drains and a later caller can still get a fresh refresh.
            release.set()
            await asyncio.wait_for(
                asyncio.gather(*waiters, return_exceptions=True), timeout=5.0
            )
            waiters = []
            await asyncio.sleep(0)  # let done-callbacks run
            assert not cache._live_bypasses.get('k'), (
                'completed bypasses must be reclaimed from the live roster, '
                f'got {cache._live_bypasses.get("k")}'
            )

            before = calls['n']
            cache.clear()
            cache._locks.setdefault('k', asyncio.Lock())
            later = await asyncio.wait_for(
                cache.get_or_refresh('k', _refresh), timeout=5.0
            )
            assert later.startswith('value-')
            assert calls['n'] == before + 1, (
                'a key whose bypasses all finished must not stay capped'
            )
        finally:
            lock.release()
            await self._cancel_all(*waiters)


class TestTTLCacheEvictsDeadBypassEntries:
    """A DEAD bypass entry must actually be reclaimed, not merely harmless.

    Reviewer finding: "``_bypass_tasks`` is not self-bounding as line 456
    claims (an entry outlives its refresh exactly when the refresh never
    completes)." Step-8 makes such an entry harmless to CALLERS -- they no
    longer inherit it past its bound -- but does not reclaim it: it is
    replaced only if some later caller re-arms that same key, so a key
    that wedges and then goes quiet retains its tuple, and the task it
    references, indefinitely. That contradicts the class's own "growth and
    reclamation are coupled by construction" paragraph, which task 3857
    added precisely because a high-cardinality key space had been
    retaining per-key state forever. This class pins that ``_evict_expired``
    actually drops a dead entry, closing that claim for this third
    per-key structure.

    Reaches ``_evict_expired`` the way the module actually reaches it:
    through a cold miss on an UNRELATED key, never called directly. Reuses
    the fake-clock idiom the eviction tests already use --
    ``monkeypatch.setattr(fanout_mod, 'time', types.SimpleNamespace(monotonic=...))``
    (module-local per pre-1) -- so the entry's ``started_at`` and the
    sweep's ``now`` read the SAME clock and stay self-consistent, while
    asyncio's own timers (and thus ``_LOCK_ACQUIRE_TIMEOUT_SECONDS`` itself)
    keep running on the real clock.
    """

    async def test_an_over_age_bypass_entry_is_reclaimed_by_a_later_sweep(
        self, monkeypatch
    ):
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=lambda: 20.0)
        clock = {'t': 0.0}
        monkeypatch.setattr(
            fanout_mod, 'time', types.SimpleNamespace(monotonic=lambda: clock['t'])
        )

        entered = asyncio.Event()
        wedged = asyncio.Event()  # never set

        async def _refresh_a():
            entered.set()
            await wedged.wait()
            raise AssertionError('unreachable: the wedged event is never set')

        # Hold key 'a''s lock directly so the caller below times out
        # acquiring it and becomes the bypass creator -- the same idiom
        # TestTTLCacheBypassRechecksFreshness already uses. The REAL clock
        # still drives _LOCK_ACQUIRE_TIMEOUT_SECONDS (mcp_fanout's own
        # `time` name is stubbed, not asyncio's loop clock -- see pre-1).
        lock = cache._locks.setdefault('a', asyncio.Lock())
        await lock.acquire()
        holder = asyncio.create_task(cache.get_or_refresh('a', _refresh_a))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        lock.release()

        assert 'a' in cache._bypass_tasks, 'precondition: a bypass was installed for a'
        task = cache._bypass_tasks['a'][1]

        try:
            # Advance the FAKE clock -- both the entry's started_at and the
            # sweep's `now` read it, so ages stay self-consistent -- past
            # the bound, with no real waiting needed.
            clock['t'] = 1000.0

            async def _refresh_b():
                return 'b-value'

            result = await cache.get_or_refresh('b', _refresh_b)

            assert result == 'b-value'
            assert 'a' not in cache._bypass_tasks, (
                'an over-age bypass entry must be reclaimed by a later '
                'sweep, not retained indefinitely once the key goes quiet'
            )
            assert task.cancelled() is False, (
                'reclamation here means stop TRACKING, not cancel -- the '
                'abandoned task keeps running to completion in the '
                'background (see the design decision on '
                'abandon-dont-cancel)'
            )
        finally:
            task.cancel()
            await asyncio.gather(task, holder, return_exceptions=True)

    async def test_a_within_bound_bypass_entry_is_never_reclaimed(self, monkeypatch):
        """Eviction-safety analogue of test_eviction_never_drops_a_still_servable_entry.

        Pruning a WITHIN-bound entry would be a real bug: later callers
        would each start their own refresh instead of sharing the live
        one, silently regressing the exact per-caller-fanout property
        TestTTLCacheBoundsBypassConcurrency and step-7(d) fence.
        """
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=lambda: 20.0)
        clock = {'t': 0.0}
        monkeypatch.setattr(
            fanout_mod, 'time', types.SimpleNamespace(monotonic=lambda: clock['t'])
        )

        entered = asyncio.Event()
        wedged = asyncio.Event()  # never set

        async def _refresh_a():
            entered.set()
            await wedged.wait()
            raise AssertionError('unreachable: the wedged event is never set')

        lock = cache._locks.setdefault('a', asyncio.Lock())
        await lock.acquire()
        holder = asyncio.create_task(cache.get_or_refresh('a', _refresh_a))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        lock.release()

        assert 'a' in cache._bypass_tasks, 'precondition: a bypass was installed for a'
        task = cache._bypass_tasks['a'][1]

        try:
            # Do NOT advance the clock past the bound (0.05) -- the entry
            # is still live.
            clock['t'] = 0.01

            async def _refresh_b():
                return 'b-value'

            result = await cache.get_or_refresh('b', _refresh_b)

            assert result == 'b-value'
            assert 'a' in cache._bypass_tasks, (
                'a WITHIN-bound bypass entry must survive the sweep -- '
                'pruning it would silently restore one-refresh-per-caller '
                'for every later timed-out caller of this key'
            )
            assert cache._bypass_tasks['a'][1] is task, (
                'the surviving entry must be the SAME task, not a replacement'
            )
        finally:
            task.cancel()
            await asyncio.gather(task, holder, return_exceptions=True)

    async def test_a_completed_bypass_entry_is_gone_without_needing_a_sweep(
        self, monkeypatch
    ):
        """Pins which mechanism owns which case.

        Passes both before and after step-10: a bypass whose refresh
        RETURNS is dropped by its own done-callback with no sweep
        involved, so a future reader does not assume the sweep is
        load-bearing for the common (completing) path -- only for a key
        that wedges and then goes quiet (the case above).
        """
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)

        lock = cache._locks.setdefault('a', asyncio.Lock())
        await lock.acquire()

        async def _refresh_a():
            return 'a-value'

        try:
            result = await asyncio.wait_for(
                cache.get_or_refresh('a', _refresh_a), 5.0
            )
        finally:
            lock.release()

        assert result == 'a-value'
        assert 'a' not in cache._bypass_tasks, (
            'a bypass whose refresh RETURNS must be dropped by its own '
            'done-callback -- no sweep needed'
        )


class TestTTLCacheSweepsOnTheBypassPath:
    """A bypass must sweep expired entries too, not only the locked path.

    get_or_refresh's docstring says a bypass is the ONLY traffic a wedged
    key sees, so skipping the sweep there would create a sweep-starved
    regime during exactly the outage the eviction work (task 3857) was
    added for -- but every TestTTLCacheEvictsExpiredKeys test reaches
    _evict_expired via the LOCKED path only. This pins the sweep to the
    bypass path specifically.
    """

    async def test_a_bypass_reclaims_an_unrelated_ancient_entry(self, monkeypatch):
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=lambda: 20.0)
        clock = {'t': 0.0}
        monkeypatch.setattr(
            fanout_mod, 'time', types.SimpleNamespace(monotonic=lambda: clock['t'])
        )

        # An ancient, unrelated entry: long past the eviction horizon
        # (20.0 * _EVICTION_TTL_MULTIPLE == 80.0). Its lock (if any) is idle.
        cache._store['ancient'] = (0.0, 'stale-row')
        clock['t'] = 1000.0

        # Hold the target key's lock directly so the bypass path fires -- the
        # same idiom TestTTLCacheBypassRechecksFreshness already uses. The
        # REAL clock still drives _LOCK_ACQUIRE_TIMEOUT_SECONDS (mcp_fanout's
        # own `time` name is stubbed, not asyncio's loop clock -- see pre-1).
        lock = cache._locks.setdefault('k', asyncio.Lock())
        await lock.acquire()

        async def _refresh():
            return 'value'

        try:
            result = await asyncio.wait_for(cache.get_or_refresh('k', _refresh), 5.0)
        finally:
            lock.release()

        assert result == 'value'
        assert 'ancient' not in cache._store, (
            'a bypass must sweep expired entries -- the bypass path is the '
            'ONLY traffic a wedged key sees, so skipping the sweep there '
            'would starve reclamation for the duration of the outage'
        )


class TestTTLCacheBypassCannotClobberANewerValue:
    """A late-returning locked refresh must not overwrite a newer bypass value.

    _refresh_and_store's post-refresh store is the only guard between a
    slow (or truly wedged) LOCKED refresh and a value some other, faster
    bypass refresh already stored for the same key while the locked one was
    still running. Without the staleness guard, the locked refresh's
    eventual (stale) result would unconditionally overwrite the newer one --
    AND stamp it with a fresh time.monotonic(), making the stale data look
    maximally fresh for a full TTL window. This never mattered while the
    lock made the two paths mutually exclusive; the bypass path breaks that
    exclusivity by design (task 4789).
    """

    async def test_a_late_returning_locked_refresh_does_not_clobber_a_newer_bypass(
        self, monkeypatch
    ):
        import dashboard.data.mcp_fanout as fanout_mod

        monkeypatch.setattr(fanout_mod, '_LOCK_ACQUIRE_TIMEOUT_SECONDS', 0.05)
        cache: TTLCache[str] = TTLCache(ttl_seconds=60.0)

        original_entered = asyncio.Event()
        original_may_return = asyncio.Event()

        async def _original_refresh():
            original_entered.set()
            await original_may_return.wait()
            return 'stale-original'

        async def _second_callers_refresh():
            return 'fresh-bypass'

        # The ORIGINAL caller acquires the lock normally (nothing else holds
        # it yet) and then wedges inside its own refresh.
        original_task = asyncio.create_task(
            cache.get_or_refresh('k', _original_refresh)
        )
        await asyncio.wait_for(original_entered.wait(), timeout=5.0)

        # A second caller times out acquiring the still-held lock and
        # bypasses, storing a fresher value.
        bypass_result = await asyncio.wait_for(
            cache.get_or_refresh('k', _second_callers_refresh), timeout=5.0
        )
        assert bypass_result == 'fresh-bypass'
        assert cache.get_fresh('k') == 'fresh-bypass'

        # Now let the original, slow refresh finally resolve. It must not
        # clobber the newer bypass-stored value.
        original_may_return.set()
        original_result = await asyncio.wait_for(original_task, timeout=5.0)

        assert original_result == 'stale-original', (
            'the original caller still gets its own refresh result back'
        )
        assert cache.get_fresh('k') == 'fresh-bypass', (
            'a late-returning locked refresh must not clobber a newer value '
            'a bypass already stored for the same key'
        )
