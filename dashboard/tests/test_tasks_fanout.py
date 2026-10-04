"""Tests for the cached fan-out every tasks read rides.

The file covers ``_cached_fanout``, its positive and negative TTL caches as
observed through ``fetch_tasks``, marker precedence, and per-root
failure-streak isolation. The streak class also covers ``fetch_statuses``'
uncached fan-out.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from _canned_mcp import canned_get_tasks_result

import dashboard.data.tasks as tasks_mod

# ---------------------------------------------------------------------------
# TestFetchTasksCache — per-project_root TTL cache inside fetch_tasks
# (step-1 core-contract tests RED; step-3 TTL-expiry test RED)
# ---------------------------------------------------------------------------


class TestFetchTasksCache:
    """Per-project_root TTL cache inside fetch_tasks.

    RED sources (step-1): _fetch_tasks_cache_clear does not yet exist in
    dashboard.data.tasks — the autouse fixture raises AttributeError on every
    test, giving the expected RED for the whole class.

    After step-2 adds the cache infrastructure, tests (a)-(d) go GREEN.
    test_fetch_tasks_ttl_expiry_refetches (step-3) stays RED until step-4
    wires the monotonic freshness check.
    """

    @pytest.fixture(autouse=True)
    def reset_fetch_tasks_cache(self):
        """Clear the per-project cache before (and after) each test.

        RED until step-2 adds dashboard.data.tasks._fetch_tasks_cache_clear.
        """
        import dashboard.data.tasks as tasks_mod
        tasks_mod._fetch_tasks_cache_clear()
        yield
        tasks_mod._fetch_tasks_cache_clear()

    async def test_fetch_tasks_within_ttl_issues_single_mcp_call(
        self, dummy_client, dummy_config
    ):
        """Two fetch_tasks calls for the same root within the TTL issue exactly ONE MCP call.

        Also asserts the shaped DONE task preserves updated_at == the input updatedAt
        and status == 'done', proving the cached set includes done tasks unchanged.

        RED until step-2: no cache → call_count == 2.
        """
        import dashboard.data.tasks as tasks_mod
        from dashboard.data.tasks import fetch_tasks

        mock_mcp = AsyncMock(return_value=canned_get_tasks_result())
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            result1 = await fetch_tasks(dummy_client, dummy_config, '/proj/A')
            result2 = await fetch_tasks(dummy_client, dummy_config, '/proj/A')

        assert mock_mcp.call_count == 1, (
            f'expected exactly 1 MCP call within TTL, got {mock_mcp.call_count}'
        )
        # The single call was 'get_tasks' for the correct project_root.
        # The unnarrowed arguments dict must stay byte-identical to the
        # pre-narrowing shape (the four full-tree callers depend on it); the
        # per-request budget rides as a keyword, never inside the dict.
        call = mock_mcp.call_args_list[0]
        positional = call.args
        assert positional[2] == 'get_tasks'
        assert positional[3] == {'project_root': '/proj/A'}
        assert call.kwargs.get('timeout') == tasks_mod.DEFAULT_PER_CALL_TIMEOUT
        # Both calls return equal shaped lists.
        assert isinstance(result1, list)
        assert result1 == result2
        # The shaped DONE task preserves updated_at (recency key for ordering + display).
        done_tasks = [t for t in result1 if t.get('status') == 'done']
        assert len(done_tasks) == 1
        assert done_tasks[0]['updated_at'] == '2026-05-29T10:00:00+00:00'

    async def test_fetch_tasks_distinct_projects_cached_separately(
        self, dummy_client, dummy_config
    ):
        """Two distinct project_roots each trigger their own MCP call (no cross-keying).

        Guards against a global/un-keyed cache; per-root results must match
        the respective canned data (call_count == 2).
        """
        from dashboard.data.tasks import fetch_tasks

        task_a_raw = {
            'id': '1', 'title': 'Task A', 'status': 'pending',
            'dependencies': [], 'metadata': {},
        }
        task_b_raw = {
            'id': '2', 'title': 'Task B', 'status': 'done',
            'updatedAt': '2026-06-01T00:00:00+00:00',
            'dependencies': [], 'metadata': {},
        }

        async def _per_root(client, url, tool, args, **_kw):
            root = args.get('project_root', '')
            if root == '/proj/A':
                return {'tasks': [task_a_raw]}
            if root == '/proj/B':
                return {'tasks': [task_b_raw]}
            return {'tasks': []}

        mock_mcp = AsyncMock(side_effect=_per_root)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            result_a = await fetch_tasks(dummy_client, dummy_config, '/proj/A')
            result_b = await fetch_tasks(dummy_client, dummy_config, '/proj/B')

        assert mock_mcp.call_count == 2, (
            f'expected 2 MCP calls for 2 distinct project_roots, got {mock_mcp.call_count}'
        )
        assert len(result_a) == 1
        assert result_a[0]['title'] == 'Task A'
        assert len(result_b) == 1
        assert result_b[0]['title'] == 'Task B'

    async def test_fetch_tasks_offline_not_cached(
        self, monkeypatch, dummy_client, dummy_config
    ):
        """Offline markers ({offline: True}) are never pinned in the POSITIVE cache.

        First call: ConnectError → offline dict returned.
        Second call: mock recovers → MCP call issued (call_count == 2); list returned.

        Task 3857 note: an offline marker is now held briefly in the SEPARATE
        negative cache (``_FETCH_TASKS_NEGATIVE_TTL_SECONDS``, ~5 s) so a
        broken root stops being re-walked on every 3 s poll. That window is
        collapsed to zero here so this test keeps asserting what it always
        asserted — a failure never pins itself for the 20 s POSITIVE TTL, and
        recovery is observed as soon as an attempt is allowed to run. The
        negative window itself is covered by ``TestFetchTasksNegativeCache``.
        """
        import dashboard.data.tasks as tasks_mod
        from dashboard.data.tasks import fetch_tasks

        monkeypatch.setattr(tasks_mod, '_FETCH_TASKS_NEGATIVE_TTL_SECONDS', 0.0)

        task_c_raw = {
            'id': '3', 'title': 'Task C', 'status': 'pending',
            'dependencies': [], 'metadata': {},
        }
        mock_mcp = AsyncMock(side_effect=[
            httpx.ConnectError('refused'),
            {'tasks': [task_c_raw]},
        ])
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            offline_result = await fetch_tasks(dummy_client, dummy_config, '/proj/C')
            success_result = await fetch_tasks(dummy_client, dummy_config, '/proj/C')

        assert mock_mcp.call_count == 2, (
            f'expected 2 MCP calls (offline not cached), got {mock_mcp.call_count}'
        )
        # First result is the offline marker dict.
        assert isinstance(offline_result, dict)
        assert offline_result.get('offline') is True
        # Second result is a list with the recovered task.
        assert isinstance(success_result, list)
        assert len(success_result) == 1
        assert success_result[0]['title'] == 'Task C'
        # The marker never entered the positive cache — the recovered list did.
        key = tasks_mod._TasksRead('/proj/C', None, tasks_mod._CompleteRead(None))
        assert tasks_mod._fetch_tasks_cache.get_fresh(key) == success_result

    async def test_fetch_tasks_returned_list_is_a_copy(
        self, dummy_client, dummy_config
    ):
        """Mutating the returned list must not corrupt the cached entry (copy isolation).

        A shallow list() copy is returned on each cache hit so callers cannot
        mutate the internally stored list.
        """
        from dashboard.data.tasks import fetch_tasks

        mock_mcp = AsyncMock(return_value=canned_get_tasks_result())
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            first = await fetch_tasks(dummy_client, dummy_config, '/proj/D')
            # Mutate the returned list — must NOT affect the cached entry.
            first.clear()
            second = await fetch_tasks(dummy_client, dummy_config, '/proj/D')

        assert second != [], 'mutating first result must not clear the cache entry'
        assert len(second) == 2, 'cache should still hold both shaped tasks after mutation'
        assert mock_mcp.call_count == 1, 'only one MCP call should have been issued'

    async def test_fetch_tasks_ttl_expiry_refetches(
        self, monkeypatch, dummy_client, dummy_config
    ):
        """When the cached entry exceeds TTL the next call issues a fresh MCP call.

        Monkeypatches _FETCH_TASKS_TTL_SECONDS=0.0 so any stored entry is
        immediately stale.  Two sequential calls must produce call_count == 2.

        RED until step-4: step-2 serves on presence (no freshness check) so
        count stays 1, failing this assertion.
        """
        import dashboard.data.tasks as tasks_mod
        from dashboard.data.tasks import fetch_tasks

        monkeypatch.setattr(tasks_mod, '_FETCH_TASKS_TTL_SECONDS', 0.0)

        mock_mcp = AsyncMock(return_value=canned_get_tasks_result())
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            await fetch_tasks(dummy_client, dummy_config, '/proj/E')
            await fetch_tasks(dummy_client, dummy_config, '/proj/E')

        assert mock_mcp.call_count == 2, (
            f'expected 2 MCP calls after TTL expiry (TTL=0.0), got {mock_mcp.call_count}'
        )

    async def test_fetch_tasks_list_copy_is_shallow_not_deep(
        self, dummy_client, dummy_config
    ):
        """Documents shallow-copy contract: inner task dict mutation IS visible in cache.

        ``list()`` provides list-level isolation only (proven by
        ``test_fetch_tasks_returned_list_is_a_copy``).  Inner task dicts are
        *shared* references between the returned list and the cached tuple;
        mutating a field in a returned task dict WILL be reflected in
        subsequent within-TTL cache hits.

        This test documents the contract boundary so that:
        (a) callers know not to mutate returned task dicts in place, and
        (b) if the implementation switches to ``copy.deepcopy`` this test will
            fail, flagging the contract change.

        Current callers (active_tasks, shape_escalations) build fresh rows and
        do not mutate source dicts, so there is no live bug today.
        """
        from dashboard.data.tasks import fetch_tasks

        mock_mcp = AsyncMock(return_value=canned_get_tasks_result())
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            first = await fetch_tasks(dummy_client, dummy_config, '/proj/G')
            assert first, 'expected at least one task from canned result'
            # Mutate a field in the first task dict — this touches the shared
            # object stored in the cache (shallow copy only guards the list wrapper).
            first[0]['__shallow_copy_marker__'] = True
            # Second call within TTL — served from cache without a new MCP call.
            second = await fetch_tasks(dummy_client, dummy_config, '/proj/G')

        assert mock_mcp.call_count == 1, 'expected single MCP call (both within TTL)'
        # Shallow copy: the inner dict mutation IS visible in the cached entry.
        # This assertion documents the known contract; if it fails, the
        # implementation has switched to deepcopy (update docstring accordingly).
        assert second[0].get('__shallow_copy_marker__') is True, (
            'list() is shallow — inner dict mutation is shared with the cache; '
            'callers must not mutate returned task dicts in place'
        )

    async def test_fetch_tasks_concurrent_cold_callers_single_flight(
        self, dummy_client, dummy_config
    ):
        """Two concurrent fetch_tasks calls on a cold cache collapse onto ONE MCP call.

        A shared asyncio.Event gates mcp_tool_call so both coroutines genuinely
        overlap while the cache is cold, rather than serializing by accident.

        RED until step-7: fetch_tasks has no single-flight lock today, so both
        cold callers reach mcp_tool_call independently (call_count == 2). GREEN
        once fetch_tasks routes through TTLCache.get_or_refresh, whose per-key
        lock makes the second caller wait for (and then reuse) the first
        caller's in-flight result instead of issuing its own MCP call.
        """
        from dashboard.data.tasks import fetch_tasks

        gate = asyncio.Event()

        async def _gated(client, url, tool, args, **_kw):
            await gate.wait()
            return canned_get_tasks_result()

        mock_mcp = AsyncMock(side_effect=_gated)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            task1 = asyncio.create_task(fetch_tasks(dummy_client, dummy_config, '/proj/SF'))
            task2 = asyncio.create_task(fetch_tasks(dummy_client, dummy_config, '/proj/SF'))

            # Let both coroutines reach mcp_tool_call (or block behind the
            # single-flight lock) before releasing the gate.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            gate.set()

            result1, result2 = await asyncio.gather(task1, task2)

        assert mock_mcp.call_count == 1, (
            f'expected single-flight (1 MCP call for 2 concurrent cold callers '
            f'on the same project_root), got {mock_mcp.call_count}'
        )
        assert result1 == result2


# ---------------------------------------------------------------------------
# TestFetchTasksNegativeCache — a failing root must stop being the expensive
# path (task 3857 step-5 RED)
# ---------------------------------------------------------------------------


class TestFetchTasksNegativeCache:
    """``cache_ok`` stores successes only, so failure is the expensive path.

    A healthy root rides the ~20 s positive TTL; a broken one re-walks its
    whole tree on every 3 s UI poll. The fix is a SECOND, much shorter TTL
    cache for offline markers — the retry is suppressed, the degradation
    signal is not.
    """

    @pytest.fixture(autouse=True)
    def reset_fetch_tasks_cache(self):
        import dashboard.data.tasks as tasks_mod
        tasks_mod._fetch_tasks_cache_clear()
        yield
        tasks_mod._fetch_tasks_cache_clear()

    async def test_second_attempt_within_negative_ttl_is_suppressed(
        self, dummy_client, dummy_config
    ):
        """(a) The retry is suppressed; the offline marker is still returned."""
        from dashboard.data.tasks import fetch_tasks

        mock_mcp = AsyncMock(side_effect=httpx.ConnectError('refused'))
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            first = await fetch_tasks(dummy_client, dummy_config, '/proj/NEG')
            calls_after_first = mock_mcp.call_count
            second = await fetch_tasks(dummy_client, dummy_config, '/proj/NEG')

        assert calls_after_first >= 1, 'the first attempt must actually try'
        assert mock_mcp.call_count == calls_after_first, (
            'the second attempt within the negative TTL must issue no MCP call, '
            f'got {mock_mcp.call_count - calls_after_first} extra'
        )
        # Degradation stays visible to BOTH callers — only the retry is suppressed.
        for marker in (first, second):
            assert isinstance(marker, dict)
            assert marker.get('offline') is True
            assert 'error' in marker

    async def test_negative_ttl_expiry_retries(
        self, monkeypatch, dummy_client, dummy_config
    ):
        """(b) A zero negative TTL makes every attempt live again."""
        import dashboard.data.tasks as tasks_mod
        from dashboard.data.tasks import fetch_tasks

        mock_mcp = AsyncMock(side_effect=httpx.ConnectError('refused'))
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            await fetch_tasks(dummy_client, dummy_config, '/proj/NEG2')
            after_first = mock_mcp.call_count
            await fetch_tasks(dummy_client, dummy_config, '/proj/NEG2')
            assert mock_mcp.call_count == after_first, 'sanity: suppressed while fresh'

            monkeypatch.setattr(
                tasks_mod, '_FETCH_TASKS_NEGATIVE_TTL_SECONDS', 0.0,
            )
            await fetch_tasks(dummy_client, dummy_config, '/proj/NEG2')

        assert mock_mcp.call_count > after_first, (
            'an expired negative entry must let the next call retry'
        )

    async def test_negative_entry_does_not_suppress_a_different_narrowing(
        self, dummy_client, dummy_config
    ):
        """(c) The negative entry shares the positive key function.

        A broken narrowed read must not mask a healthy differently-narrowed
        one — otherwise one failing call would blind the whole tab.
        """
        from dashboard.data.tasks import fetch_tasks

        async def _fail_only_active(client, url, tool, args, **_kw):
            if args.get('statuses') == ['in-progress']:
                raise httpx.ConnectError('refused')
            return {'tasks': [{
                'id': '1', 'title': 'OK ROW', 'status': 'done',
                'dependencies': [], 'metadata': {},
            }]}

        mock_mcp = AsyncMock(side_effect=_fail_only_active)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            broken = await fetch_tasks(
                dummy_client, dummy_config, '/proj/NEG3', statuses=['in-progress'],
            )
            other = await fetch_tasks(
                dummy_client, dummy_config, '/proj/NEG3', statuses=['done'],
            )

        assert isinstance(broken, dict) and broken.get('offline') is True
        assert isinstance(other, list), (
            'a negative entry for one narrowing must not suppress another'
        )
        assert [t['title'] for t in other] == ['OK ROW']

    async def test_negative_entry_does_not_suppress_a_different_root(
        self, dummy_client, dummy_config
    ):
        """(c) One broken root must not blind a healthy sibling root."""
        from dashboard.data.tasks import fetch_tasks

        async def _fail_one_root(client, url, tool, args, **_kw):
            if args['project_root'] == '/proj/BROKEN':
                raise httpx.ConnectError('refused')
            return {'tasks': [{
                'id': '1', 'title': 'HEALTHY ROW', 'status': 'pending',
                'dependencies': [], 'metadata': {},
            }]}

        mock_mcp = AsyncMock(side_effect=_fail_one_root)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            broken = await fetch_tasks(dummy_client, dummy_config, '/proj/BROKEN')
            healthy = await fetch_tasks(dummy_client, dummy_config, '/proj/HEALTHY')

        assert isinstance(broken, dict) and broken.get('offline') is True
        assert isinstance(healthy, list)
        assert [t['title'] for t in healthy] == ['HEALTHY ROW']

    async def test_cache_clear_drops_the_negative_entry_too(
        self, dummy_client, dummy_config
    ):
        """(d) The test/admin clear hook must reset BOTH stores."""
        import dashboard.data.tasks as tasks_mod
        from dashboard.data.tasks import fetch_tasks

        mock_mcp = AsyncMock(side_effect=httpx.ConnectError('refused'))
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            await fetch_tasks(dummy_client, dummy_config, '/proj/NEG4')
            after_first = mock_mcp.call_count
            tasks_mod._fetch_tasks_cache_clear()
            await fetch_tasks(dummy_client, dummy_config, '/proj/NEG4')

        assert mock_mcp.call_count > after_first, (
            '_fetch_tasks_cache_clear() must clear the negative cache as well, '
            'else a cleared cache still suppresses retries'
        )

    async def test_success_never_lands_in_the_negative_cache(
        self, dummy_client, dummy_config
    ):
        """(e) The positive path is untouched by the negative cache."""
        import dashboard.data.tasks as tasks_mod
        from dashboard.data.tasks import fetch_tasks

        mock_mcp = AsyncMock(return_value=canned_get_tasks_result())
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            first = await fetch_tasks(dummy_client, dummy_config, '/proj/NEG5')
            second = await fetch_tasks(dummy_client, dummy_config, '/proj/NEG5')

        assert mock_mcp.call_count == 1, 'positive TTL still single-flights'
        assert isinstance(first, list) and first == second
        key = tasks_mod._TasksRead('/proj/NEG5', None, tasks_mod._CompleteRead(None))
        assert tasks_mod._fetch_tasks_negative_cache.get_fresh(key) is None, (
            'a successful fetch must never be stored as a negative entry'
        )

    async def test_a_concurrent_success_is_never_shadowed_by_the_marker(
        self, dummy_client, dummy_config
    ):
        """(f) A fresh SUCCESS must win over a fresh offline marker.

        The negative lookup sits before, and outside of, the positive cache's
        per-key lock, so the two can both be fresh at once.  ``TTLCache``
        documents that a ``cache_ok``-rejected value stores nothing and "the
        next lock-queued waiter runs its own refresh in turn", which is exactly
        the interleaving driven here: two concurrent callers for one key (the
        routine case — ``app._load_task_cards``, ``data.orchestrator``,
        ``data.merge_queue`` and ``data.burndown`` all fetch the same
        unnarrowed key on the same poll), waiter A fails and writes a 5 s
        marker, waiter B then succeeds and writes a 20 s positive entry.

        Every caller for the next ~5 s then got the offline marker while valid
        fresh data sat in the positive cache — a false offline banner.  The
        module comment asserted this could not happen ("no success can occur
        inside the window to contradict it"), which held only single-threaded.

        Note what is NOT asserted: retry suppression is unchanged.  The third
        call below must issue no new MCP attempt — it is served from the
        positive cache, so the marker still costs the broken root nothing.
        """
        import asyncio

        import dashboard.data.tasks as tasks_mod
        from dashboard.data.tasks import fetch_tasks

        attempts = 0

        async def _fail_then_succeed(client, url, tool, args, **_kw):
            nonlocal attempts
            attempts += 1
            mine = attempts
            await asyncio.sleep(0)  # yield, so both callers are in flight
            if mine == 1:
                raise httpx.ConnectError('refused')
            return canned_get_tasks_result()

        with patch('dashboard.data.tasks.mcp_tool_call', new=_fail_then_succeed):
            first, second = await asyncio.gather(
                fetch_tasks(dummy_client, dummy_config, '/proj/NEG7'),
                fetch_tasks(dummy_client, dummy_config, '/proj/NEG7'),
            )
            attempts_after_race = attempts
            third = await fetch_tasks(dummy_client, dummy_config, '/proj/NEG7')

        key = tasks_mod._TasksRead('/proj/NEG7', None, tasks_mod._CompleteRead(None))
        # Precondition: the race really did leave BOTH entries fresh. Without
        # this the test could pass for the wrong reason (e.g. no marker stored).
        assert tasks_mod._fetch_tasks_negative_cache.get_fresh(key) is not None, (
            'this test is only meaningful if the failure did store a marker'
        )
        assert tasks_mod._fetch_tasks_cache.get_fresh(key) is not None, (
            'this test is only meaningful if the success did store a positive entry'
        )
        assert {isinstance(first, list), isinstance(second, list)} == {True, False}, (
            'the interleaving under test is one failure and one success, got '
            f'{type(first).__name__} and {type(second).__name__}'
        )
        assert isinstance(third, list), (
            'a caller after the race got the offline marker while fresh, valid '
            'data sat in the positive cache — a false offline banner over rows '
            f'that had already loaded: {third}'
        )
        assert attempts == attempts_after_race, (
            'the post-race call must be served from the positive cache, not '
            'become a fresh attempt — retry suppression is not the thing being '
            'relaxed here'
        )

    async def test_offline_marker_still_never_lands_in_the_positive_cache(
        self, dummy_client, dummy_config
    ):
        """(e) The existing "offline markers are not cached positively" contract."""
        import dashboard.data.tasks as tasks_mod
        from dashboard.data.tasks import fetch_tasks

        mock_mcp = AsyncMock(side_effect=httpx.ConnectError('refused'))
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            await fetch_tasks(dummy_client, dummy_config, '/proj/NEG6')

        key = tasks_mod._TasksRead('/proj/NEG6', None, tasks_mod._CompleteRead(None))
        assert tasks_mod._fetch_tasks_cache.get_fresh(key) is None, (
            'an offline marker must not pin itself in the positive cache'
        )

    def test_negative_ttl_sits_between_the_poll_and_the_positive_ttl(self):
        """The 5 s choice is pinned against the two real clocks it was picked from.

        Shorter than the positive TTL so an outage is re-probed several times
        per success window; longer than ``data.js``'s 3 s ``POLL_INTERVAL_MS``
        so a broken root costs at most one attempt per two polls rather than
        one per poll.
        """
        import re
        from pathlib import Path

        import dashboard.data.tasks as tasks_mod

        data_js = (
            Path(tasks_mod.__file__).resolve().parents[1]
            / 'static' / 'redux' / 'data.js'
        )
        match = re.search(
            r'POLL_INTERVAL_MS\s*=\s*(\d+)', data_js.read_text(),
        )
        assert match is not None, f'POLL_INTERVAL_MS not found in {data_js}'
        poll_seconds = int(match.group(1)) / 1000.0

        negative = tasks_mod._FETCH_TASKS_NEGATIVE_TTL_SECONDS
        assert negative > poll_seconds, (
            f'negative TTL {negative}s must exceed the {poll_seconds}s UI poll '
            'interval, else a broken root is re-walked on every poll'
        )
        assert negative < tasks_mod._FETCH_TASKS_TTL_SECONDS, (
            f'negative TTL {negative}s must be shorter than the positive TTL '
            f'{tasks_mod._FETCH_TASKS_TTL_SECONDS}s so an outage is re-probed '
            'several times per success window'
        )


# ---------------------------------------------------------------------------
# cross-project fan-out streak isolation (task 4133)
# ---------------------------------------------------------------------------


class TestFanoutStreakIsolationAcrossProjectRoots:
    """One broken project_root must not re-arm its WARNING every poll cycle.

    The mcp_fanout throttle key is ``(log_label, url)`` and ONE fused-memory
    URL serves every project_root, so a fixed literal ``log_label`` collapses
    all roots onto one key. ``note_fanout_success`` *pops* that key, so a
    healthy root's success in the same poll cycle clears the broken root's
    open streak — its next failure is ``streak == 1`` again, emitting both an
    opening and a 'recovered' WARNING every cycle, indefinitely. That is the
    exact sustained flood the transition-only policy (task 3871) exists to
    prevent, reintroduced through the key rather than the level.
    """

    @pytest.fixture(autouse=True)
    def _clean_state(self):
        """Give each test clean streak AND cache state.

        Without reset_sessions an earlier test's open streak would silently
        demote this test's expected opening WARNING to DEBUG — the exact
        failure mode reset_failure_streaks was added for.
        """
        import dashboard.data.tasks as tasks_mod
        from dashboard.data.memory import reset_sessions

        reset_sessions()
        tasks_mod._fetch_tasks_cache_clear()
        yield
        reset_sessions()
        tasks_mod._fetch_tasks_cache_clear()

    @staticmethod
    def _per_root_side_effect(root_a: str, payload: dict):
        """Raise ConnectError for root A; return *payload* for any other root."""
        async def _call(client, url, tool, args, **_kw):
            if args.get('project_root') == root_a:
                raise httpx.ConnectError('refused')
            return payload
        return _call

    @staticmethod
    def _fanout_records(caplog):
        records = [r for r in caplog.records if r.name == 'dashboard.data.mcp_fanout']
        warnings = [r.getMessage() for r in records if r.levelno == logging.WARNING]
        return warnings

    async def test_fetch_tasks_broken_root_warns_once_across_poll_cycles(
        self, dummy_client, dummy_config, tmp_path, caplog
    ):
        """Three poll cycles, one shared URL, root A broken and root B healthy."""
        import dashboard.data.tasks as tasks_mod
        from dashboard.data.tasks import fetch_tasks

        assert len(dummy_config.fused_memory_urls) == 1, (
            'the collapse only shows up when both roots share one URL'
        )
        root_a = str(tmp_path / 'proj-a')
        root_b = str(tmp_path / 'proj-b')

        mock_mcp = AsyncMock(
            side_effect=self._per_root_side_effect(root_a, canned_get_tasks_result())
        )
        with caplog.at_level(logging.DEBUG, logger='dashboard.data.mcp_fanout'), \
                patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            for _ in range(3):
                # Clear the 20 s TTL cache so root B genuinely re-polls each
                # cycle, as the live UI does once its entry expires.
                tasks_mod._fetch_tasks_cache_clear()
                a_result = await fetch_tasks(dummy_client, dummy_config, root_a)
                b_result = await fetch_tasks(dummy_client, dummy_config, root_b)

        # fetch_tasks returns ``list[dict] | dict``; narrow to the offline
        # marker branch before reading it (same shape as the cache tests above).
        assert isinstance(a_result, dict), 'root A must return the offline marker'
        assert a_result.get('offline') is True, 'root A must be failing'
        assert isinstance(b_result, list), 'root B must be healthy'

        warnings = self._fanout_records(caplog)
        assert len(warnings) == 1, (
            f'a broken root must warn once, not once per poll cycle, got {warnings}'
        )
        assert not [m for m in warnings if 'recovered' in m], (
            f"root B's success must not close root A's streak, got {warnings}"
        )
        assert 'proj-a' in warnings[0], (
            'the WARNING must name the failing project_root — with one shared '
            f'URL it is the only way to tell which root is down, got {warnings[0]}'
        )

    async def test_fetch_statuses_broken_root_warns_once_across_poll_cycles(
        self, dummy_client, dummy_config, tmp_path, caplog
    ):
        """Same contract for the burndown collector's fetch_statuses path."""
        from dashboard.data.tasks import fetch_statuses

        root_a = str(tmp_path / 'proj-a')
        root_b = str(tmp_path / 'proj-b')

        mock_mcp = AsyncMock(
            side_effect=self._per_root_side_effect(
                root_a, {'statuses': {'1': 'done', '2': 'pending'}}
            )
        )
        # fetch_statuses holds no cache of its own (task 5587), so BOTH
        # roots genuinely re-read every cycle — which is exactly the streak
        # this asserts on.
        with caplog.at_level(logging.DEBUG, logger='dashboard.data.mcp_fanout'), \
                patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            for _ in range(3):
                a_result = await fetch_statuses(dummy_client, dummy_config, root_a)
                b_result = await fetch_statuses(dummy_client, dummy_config, root_b)

        assert a_result.get('offline') is True, 'root A must be failing'
        assert b_result == {1: 'done', 2: 'pending'}, 'root B must be healthy'

        warnings = self._fanout_records(caplog)
        assert len(warnings) == 1, (
            f'a broken root must warn once, not once per poll cycle, got {warnings}'
        )
        assert not [m for m in warnings if 'recovered' in m], (
            f"root B's success must not close root A's streak, got {warnings}"
        )
        assert 'proj-a' in warnings[0], (
            f'the WARNING must name the failing project_root, got {warnings[0]}'
        )


class TestCachedFanoutCore:
    """`_cached_fanout(config, read, strategy, label)` — the ONE place
    the fan-out / caching / marker policy lives.

    RED source (step-6): `_cached_fanout` does not exist as a named unit; the
    policy is inline at the tail of `fetch_tasks`. Naming it is what makes the
    public split NON-duplicating: afterwards the two public reads differ ONLY
    in the record they build and the strategy they bind, so there is no second
    copy of the fan-out / caching / marker handling to drift out of step.
    """

    @pytest.fixture(autouse=True)
    def reset_fetch_tasks_cache(self):
        tasks_mod._fetch_tasks_cache_clear()
        yield
        tasks_mod._fetch_tasks_cache_clear()

    @staticmethod
    def _read(root='/proj/core', statuses=None, mode=None):
        return tasks_mod._TasksRead(
            root, statuses, mode if mode is not None else tasks_mod._CompleteRead(None)
        )

    # -- (a) parameterised by a per-URL read STRATEGY -----------------------

    async def test_the_strategy_is_invoked_once_per_url_in_config_order(
        self, dummy_client, two_url_config
    ):
        """The core owns the fan-out; the strategy owns what one url does.

        Order is asserted, not just membership: `first_success` tries urls in
        the configured order and the first success wins, so a core that
        reordered them would silently change which server the dashboard
        prefers.
        """
        seen: list[str] = []

        async def _strategy(url):
            seen.append(url)
            raise ValueError('every server is down')

        result = await tasks_mod._cached_fanout(
            two_url_config, self._read('/proj/core-order'),
            _strategy, 'fetch_tasks',
        )

        assert seen == list(two_url_config.fused_memory_urls)
        assert isinstance(result, dict) and result.get('offline') is True

    async def test_a_value_error_from_the_strategy_falls_through(
        self, dummy_client, two_url_config
    ):
        """`ValueError` is `first_success`'s documented soft-failure signal.

        A strategy raising it must cost the next url an attempt rather than
        collapsing the whole read to the offline marker.
        """
        rows = [{'id': 1, 'title': 'served by the second url'}]

        async def _strategy(url):
            if url == two_url_config.fused_memory_urls[0]:
                raise ValueError('first url said no')
            return rows

        result = await tasks_mod._cached_fanout(
            two_url_config, self._read('/proj/core-fallthrough'),
            _strategy, 'fetch_tasks',
        )

        assert result == rows

    # -- (b) the positive cache stores successes ONLY, and copies the list --

    async def test_a_success_is_cached_and_the_strategy_runs_once(
        self, dummy_client, dummy_config
    ):
        """Second read inside the TTL is served from the positive cache."""
        read = self._read('/proj/core-hit')
        calls = 0

        async def _strategy(_url):
            nonlocal calls
            calls += 1
            return [{'id': 1, 'title': 'row'}]

        first = await tasks_mod._cached_fanout(
            dummy_config, read, _strategy, 'fetch_tasks',
        )
        second = await tasks_mod._cached_fanout(
            dummy_config, read, _strategy, 'fetch_tasks',
        )

        assert first == second == [{'id': 1, 'title': 'row'}]
        assert calls == 1, 'the second read must be served from the cache'

    async def test_the_offline_marker_never_enters_the_positive_cache(
        self, dummy_client, dummy_config
    ):
        """`cache_ok=lambda v: isinstance(v, list)` — successes only.

        A marker in the POSITIVE cache would pin the offline banner for the
        full ~20 s positive TTL instead of the ~5 s negative one, so this is
        the guard that keeps the two TTLs meaning what they say.
        """
        read = self._read('/proj/core-nocache')

        async def _strategy(_url):
            raise ValueError('down')

        marker = await tasks_mod._cached_fanout(
            dummy_config, read, _strategy, 'fetch_tasks',
        )

        assert isinstance(marker, dict) and marker.get('offline') is True
        assert tasks_mod._fetch_tasks_cache.get_fresh(read) is None

    async def test_the_returned_list_is_a_fresh_copy_of_shared_elements(
        self, dummy_client, dummy_config
    ):
        """The documented SHALLOW copy contract, both halves of it.

        List-level mutation is isolated (`result.clear()` cannot empty the
        cached entry); element-level mutation is NOT (the inner dicts are
        shared references). Both halves are pinned because callers are
        documented to rely on the first and warned about the second.
        """
        read = self._read('/proj/core-copy')
        row = {'id': 1, 'status': 'pending'}

        async def _strategy(_url):
            return [row]

        first = await tasks_mod._cached_fanout(
            dummy_config, read, _strategy, 'fetch_tasks',
        )
        first.clear()

        second = await tasks_mod._cached_fanout(
            dummy_config, read, _strategy, 'fetch_tasks',
        )
        assert len(second) == 1, 'list-level mutation must not reach the cache'
        assert second is not first

        # ...and the LIMIT of that isolation, stated as fact rather than hope.
        assert second[0] is row

    # -- (c) both caches, ONE key ------------------------------------------

    async def test_both_caches_are_keyed_by_the_identical_record(
        self, dummy_client, dummy_config
    ):
        """INV-5: a split key is exactly how positive and negative drift apart.

        The SAME `_TasksRead` object must retrieve the marker from the negative
        cache and the rows from the positive one. If the two ever grew separate
        encoders, a recovered root would repopulate a key its own marker lookup
        no longer reads.
        """
        read = self._read('/proj/core-onekey')

        async def _fail(_url):
            raise ValueError('down')

        marker = await tasks_mod._cached_fanout(
            dummy_config, read, _fail, 'fetch_tasks',
        )
        assert tasks_mod._fetch_tasks_negative_cache.get_fresh(read) == marker, (
            'the marker must be written under the read record itself'
        )

        rows = [{'id': 1, 'title': 'recovered'}]

        async def _ok(_url):
            return rows

        # An equal-but-distinct record must reach the same entries: the key is
        # the record's VALUE, not its identity.
        twin = self._read('/proj/core-onekey')
        assert twin == read and twin is not read
        assert tasks_mod._fetch_tasks_negative_cache.get_fresh(twin) == marker

        tasks_mod._fetch_tasks_negative_cache.clear()
        await tasks_mod._cached_fanout(dummy_config, read, _ok, 'fetch_tasks')
        assert tasks_mod._fetch_tasks_cache.get_fresh(twin) == rows

    # -- (d) marker precedence, BOTH directions -----------------------------

    async def test_a_lone_fresh_marker_suppresses_the_attempt(
        self, dummy_client, dummy_config
    ):
        """The retry is suppressed; the degradation signal is not."""
        read = self._read('/proj/core-suppress')
        marker = {'offline': True, 'error': 'earlier failure'}

        async def _mark():
            return marker

        await tasks_mod._fetch_tasks_negative_cache.get_or_refresh(read, _mark)

        called = False

        async def _strategy(_url):
            nonlocal called
            called = True
            return []

        result = await tasks_mod._cached_fanout(
            dummy_config, read, _strategy, 'fetch_tasks',
        )

        assert result == marker, 'the marker is still RETURNED to the caller'
        assert not called, 'a fresh marker must cost no MCP attempt'

    async def test_a_fresh_positive_entry_outranks_a_fresh_marker(
        self, dummy_client, dummy_config
    ):
        """A demonstrated success beats a retry-suppression hint.

        Both caches CAN hold a fresh entry for one key — waiter A's failure
        writes a 5 s marker while waiter B's success writes a 20 s positive
        entry. Serving the marker there would put a false offline banner over
        rows already loaded.
        """
        read = self._read('/proj/core-precedence')
        rows = [{'id': 1, 'title': 'already loaded'}]

        async def _ok(_url):
            return rows

        await tasks_mod._cached_fanout(dummy_config, read, _ok, 'fetch_tasks')

        async def _mark():
            return {'offline': True, 'error': 'a concurrent failure'}

        await tasks_mod._fetch_tasks_negative_cache.get_or_refresh(read, _mark)
        assert tasks_mod._fetch_tasks_negative_cache.get_fresh(read) is not None

        calls = 0

        async def _strategy(_url):
            nonlocal calls
            calls += 1
            return rows

        result = await tasks_mod._cached_fanout(
            dummy_config, read, _strategy, 'fetch_tasks',
        )

        assert result == rows, 'the fresh positive entry wins'
        assert calls == 0, 'falling through costs no MCP call — the entry is fresh'

    # -- (e) exactly one core per public read -------------------------------

    async def test_fetch_tasks_routes_through_the_core(
        self, dummy_client, dummy_config, monkeypatch
    ):
        """The structural guard against a second copy of the policy.

        `fetch_tasks` must delegate rather than carry its own fan-out, cache
        and marker handling — otherwise the public split would duplicate all
        three and they would drift.
        """
        seen: list[tasks_mod._TasksRead] = []

        async def _fake(config, read, strategy, label, *, cached=True):
            seen.append(read)
            assert config is dummy_config
            assert callable(strategy)
            assert label == 'fetch_tasks'
            assert cached is True, 'the default read still rides the cache'
            return []

        monkeypatch.setattr(tasks_mod, '_cached_fanout', _fake)
        result = await tasks_mod.fetch_tasks(
            dummy_client, dummy_config, '/proj/core-route', timeout=7.5,
        )

        assert result == []
        assert seen == [
            tasks_mod._TasksRead(
                '/proj/core-route', None, tasks_mod._CompleteRead(None)
            )
        ], 'exactly ONE core call, carrying the record the public read built'

    async def test_each_public_read_fans_out_under_its_own_name(
        self, dummy_client, dummy_config, monkeypatch
    ):
        """The fan-out label NAMES the read that failed, per read.

        `first_success` keys its per-url failure streak on `(log_label, url)`
        and `fanout_label` composes that key, so a label shared by both public
        reads would throttle them as one stream and report a failed page read
        under `fetch_tasks[...]` — telling an operator the wrong function
        broke, which is precisely what naming the contract was meant to end.
        """
        labels: list[str] = []

        async def _fake(config, read, strategy, label, *, cached=True):
            labels.append(label)
            return []

        monkeypatch.setattr(tasks_mod, '_cached_fanout', _fake)
        await tasks_mod.fetch_tasks(dummy_client, dummy_config, '/proj/labelled')
        await tasks_mod.fetch_task_page(
            dummy_client, dummy_config, '/proj/labelled', page_size=5, offset=0,
        )

        assert labels == ['fetch_tasks', 'fetch_task_page']


# ---------------------------------------------------------------------------
# TestEveryReadIsLive — PRD decision 20: one cache per datum, owned by the
# snapshot unit, so nothing beneath it holds a task read
# ---------------------------------------------------------------------------


class TestEveryReadIsLive:
    """Every task read reaches the substrate on every call.

    The snapshot unit (``task_snapshot.SNAPSHOT_TTL_SECONDS``) is the one cache
    a task datum has. A TTL under it would let a read answer with rows older
    than the ``as_of`` the unit stamps on them, and a retry-suppression window
    under it would hide a recovery the unit is about to ask for.

    Each case uses its own project root, so no other test's reads can answer
    for these.
    """

    async def test_two_identical_whole_tree_reads_both_reach_the_substrate(
        self, dummy_client, dummy_config
    ):
        mock_mcp = AsyncMock(side_effect=lambda *_a, **_k: canned_get_tasks_result())
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            first = await tasks_mod.fetch_tasks(dummy_client, dummy_config, '/proj/live-tree')
            second = await tasks_mod.fetch_tasks(dummy_client, dummy_config, '/proj/live-tree')

        assert mock_mcp.call_count == 2, (
            f'two identical fetch_tasks calls made {mock_mcp.call_count} '
            'substrate call(s); a stored answer served the repeat'
        )
        assert isinstance(first, list) and isinstance(second, list)
        assert first == second

    async def test_two_identical_page_reads_both_reach_the_substrate(
        self, dummy_client, dummy_config
    ):
        mock_mcp = AsyncMock(side_effect=lambda *_a, **_k: canned_get_tasks_result())
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            for _ in range(2):
                page = await tasks_mod.fetch_task_page(
                    dummy_client, dummy_config, '/proj/live-page',
                    page_size=2, offset=0,
                )
                assert isinstance(page, list)

        assert mock_mcp.call_count == 2, (
            f'two identical fetch_task_page calls made {mock_mcp.call_count} '
            'substrate call(s); a stored answer served the repeat'
        )

    async def test_a_failed_read_is_retried_on_the_very_next_call(
        self, dummy_client, dummy_config
    ):
        reachable = False
        calls: list[bool] = []

        async def _mcp(*_args, **_kwargs):
            calls.append(reachable)
            if not reachable:
                raise httpx.ReadTimeout('canned get_tasks read timeout')
            return canned_get_tasks_result()

        with patch('dashboard.data.tasks.mcp_tool_call', new=_mcp):
            failed = await tasks_mod.fetch_tasks(dummy_client, dummy_config, '/proj/live-retry')
            reachable = True
            recovered = await tasks_mod.fetch_tasks(dummy_client, dummy_config, '/proj/live-retry')

        assert isinstance(failed, dict) and failed.get('offline') is True, failed
        assert True in calls, (
            'the read after a failure never reached the substrate: the failure '
            'suppressed its own retry, so a recovered root stays reported '
            f'offline (substrate calls, by reachability: {calls})'
        )
        assert isinstance(recovered, list) and len(recovered) == 2, recovered

    def test_the_terminal_window_is_read_fresh_per_request(
        self, client, tmp_path, monkeypatch
    ):
        """Its ``as_of`` is the request instant, so the rows must be that instant's.

        The window is asked for on a user action, not polled, so a read per
        request is the whole cost of making the stamp true.
        """
        from _canned_mcp import CannedMCP, _raw_row

        from dashboard.config import DashboardConfig

        root = tmp_path / 'live-window'
        root.mkdir()
        client.app.state.config = DashboardConfig(project_root=root)
        monkeypatch.setattr(
            'dashboard.data.active_tasks.fetch_task_runtime', AsyncMock(return_value={}),
        )
        pairs = [(1, 'in-progress'), (2, 'done'), (3, 'done'), (4, 'cancelled')]
        canned = CannedMCP(
            rows=[_raw_row(task_id, status) for task_id, status in pairs],
            status_map=dict(pairs),
        )

        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            for _ in range(2):
                resp = client.get('/api/v2/dashboard/tasks?terminal=live-window')
                assert resp.status_code == 200, resp.text
                assert resp.json()['TASKS_TERMINAL:live-window']['value'], resp.json()

        window_reads = [
            call for call in canned.calls_to('get_tasks')
            if 'page_size' in call['args']
        ]
        assert len(window_reads) == 2, (
            f'two ?terminal= requests issued {len(window_reads)} terminal page '
            'read(s); a stored page answered a request whose as_of claims now'
        )
