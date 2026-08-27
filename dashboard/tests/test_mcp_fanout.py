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
