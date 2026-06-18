"""Unit tests for dashboard.data.tasks._shape_task and fetch_external_statuses.

Focus: the field-mapping contract at the MCP→dashboard boundary and the
fetch_external_statuses short-circuit + fail-safe semantics.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from dashboard.data.tasks import _shape_task

# ---------------------------------------------------------------------------
# updated_at preservation (step-1/step-2)
# ---------------------------------------------------------------------------


def test_shape_task_preserves_updated_at():
    """_shape_task must carry MCP 'updatedAt' through as 'updated_at'."""
    raw = {
        'id': '7',
        'title': 'my task',
        'status': 'done',
        'updatedAt': '2026-05-29T10:00:00+00:00',
        'dependencies': [],
        'metadata': {},
    }
    shaped = _shape_task(raw)
    assert shaped is not None
    assert shaped['updated_at'] == '2026-05-29T10:00:00+00:00'


def test_shape_task_updated_at_none_when_absent():
    """updated_at must be None (not KeyError) when updatedAt is missing."""
    raw = {
        'id': '8',
        'title': 'other task',
        'status': 'pending',
        'dependencies': [],
        'metadata': {},
    }
    shaped = _shape_task(raw)
    assert shaped is not None
    # Must be present in the dict with value None (not missing key)
    assert 'updated_at' in shaped
    assert shaped['updated_at'] is None


def test_shape_task_updated_at_none_when_explicitly_null():
    """updated_at must be None when updatedAt is explicitly None."""
    raw = {
        'id': '9',
        'title': 'null task',
        'status': 'in-progress',
        'updatedAt': None,
        'dependencies': [],
        'metadata': {},
    }
    shaped = _shape_task(raw)
    assert shaped is not None
    assert shaped['updated_at'] is None


# ---------------------------------------------------------------------------
# Existing invariants: id coercion, None on invalid id
# ---------------------------------------------------------------------------


def test_shape_task_coerces_string_id_to_int():
    raw = {'id': '42', 'title': 'x', 'status': 'pending', 'dependencies': []}
    shaped = _shape_task(raw)
    assert shaped is not None
    assert shaped['id'] == 42


def test_shape_task_returns_none_on_missing_id():
    assert _shape_task({'title': 'no id', 'status': 'pending'}) is None


def test_shape_task_returns_none_on_non_numeric_id():
    assert _shape_task({'id': 'abc', 'title': 'bad id', 'status': 'pending'}) is None


# ---------------------------------------------------------------------------
# fetch_external_statuses (step-3 / step-4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_external_statuses_empty_deps_short_circuits(dummy_config):
    """fetch_external_statuses(deps=[]) returns {} immediately without any MCP call."""
    from dashboard.data.tasks import fetch_external_statuses

    called = []

    async def _fail_if_called(*args, **kwargs):
        called.append(args)
        raise AssertionError('mcp_tool_call must not be called when deps=[]')

    with patch('dashboard.data.tasks.mcp_tool_call', side_effect=_fail_if_called):
        async with httpx.AsyncClient() as client:
            result = await fetch_external_statuses(client, dummy_config, [])

    assert result == {}
    assert called == [], 'mcp_tool_call must not be invoked for empty deps'


@pytest.mark.asyncio
async def test_fetch_external_statuses_returns_bare_status_map(dummy_config):
    """fetch_external_statuses returns the BARE {dep: status} map on success."""
    from dashboard.data.tasks import fetch_external_statuses

    bare_map = {'dark_factory:13': 'done', 'reify:8': 'unknown_task'}

    async def _fake_mcp(client, url, tool, args):
        assert tool == 'get_external_statuses'
        assert args == {'deps': ['dark_factory:13', 'reify:8']}
        return bare_map

    with patch('dashboard.data.tasks.mcp_tool_call', side_effect=_fake_mcp):
        async with httpx.AsyncClient() as client:
            result = await fetch_external_statuses(
                client, dummy_config, ['dark_factory:13', 'reify:8']
            )

    assert result == {'dark_factory:13': 'done', 'reify:8': 'unknown_task'}


@pytest.mark.asyncio
async def test_fetch_external_statuses_returns_empty_on_connect_error(dummy_config):
    """fetch_external_statuses returns the offline marker on ConnectError (all URLs exhausted)."""
    from dashboard.data.tasks import fetch_external_statuses

    async def _raise_connect(*args, **kwargs):
        raise httpx.ConnectError('refused')

    with patch('dashboard.data.tasks.mcp_tool_call', side_effect=_raise_connect):
        async with httpx.AsyncClient() as client:
            result = await fetch_external_statuses(client, dummy_config, ['dark_factory:13'])

    assert result.get('offline') is True
    assert result.get('error')


@pytest.mark.asyncio
async def test_fetch_external_statuses_returns_empty_on_non_dict_result(dummy_config):
    """fetch_external_statuses returns the offline marker if MCP returns a non-dict (all URLs exhausted)."""
    from dashboard.data.tasks import fetch_external_statuses

    async def _bad_result(*args, **kwargs):
        return ['not', 'a', 'dict']

    with patch('dashboard.data.tasks.mcp_tool_call', side_effect=_bad_result):
        async with httpx.AsyncClient() as client:
            result = await fetch_external_statuses(client, dummy_config, ['dark_factory:13'])

    assert result.get('offline') is True
    assert result.get('error')


@pytest.mark.asyncio
async def test_fetch_external_statuses_failover_on_error_dict(two_url_config):
    """fetch_external_statuses continues to the next URL when the first returns an error dict.

    Multi-server failover must not be silently lost when mcp_tool_call returns a
    structured error (e.g. {'error': '...'}). The second URL succeeds and its map
    is returned.
    """
    from dashboard.data.tasks import fetch_external_statuses

    good_map = {'dark_factory:13': 'done'}
    calls: list[str] = []

    async def _two_urls(client, url, tool, args):
        calls.append(url)
        if url == two_url_config.fused_memory_urls[0]:
            return {'error': 'server overloaded'}
        return good_map

    with patch('dashboard.data.tasks.mcp_tool_call', side_effect=_two_urls):
        async with httpx.AsyncClient() as client:
            result = await fetch_external_statuses(client, two_url_config, ['dark_factory:13'])

    assert result == good_map, 'should fall through to second URL on error dict'
    assert len(calls) == 2, 'both URLs should have been tried'


@pytest.mark.asyncio
async def test_fetch_external_statuses_failover_on_empty_dict(two_url_config):
    """fetch_external_statuses continues to the next URL when the first returns an empty dict.

    An empty {} result (e.g. from a parse failure) is a soft failure and should
    trigger multi-server failover rather than returning an empty map prematurely.
    """
    from dashboard.data.tasks import fetch_external_statuses

    good_map = {'dark_factory:13': 'in-progress'}
    calls: list[str] = []

    async def _two_urls(client, url, tool, args):
        calls.append(url)
        if url == two_url_config.fused_memory_urls[0]:
            return {}  # empty — soft failure
        return good_map

    with patch('dashboard.data.tasks.mcp_tool_call', side_effect=_two_urls):
        async with httpx.AsyncClient() as client:
            result = await fetch_external_statuses(client, two_url_config, ['dark_factory:13'])

    assert result == good_map, 'should fall through to second URL on empty dict'
    assert len(calls) == 2, 'both URLs should have been tried'


@pytest.mark.asyncio
async def test_fetch_external_statuses_returns_offline_marker_on_connect_error(dummy_config):
    """fetch_external_statuses must return {'offline':True,'error':...} when ALL URLs fail.

    Fails today because the all-fail path returns {}, indistinguishable from empty deps.
    """
    from dashboard.data.tasks import fetch_external_statuses

    async def _raise_connect(*args, **kwargs):
        raise httpx.ConnectError('refused')

    with patch('dashboard.data.tasks.mcp_tool_call', side_effect=_raise_connect):
        async with httpx.AsyncClient() as client:
            result = await fetch_external_statuses(client, dummy_config, ['dark_factory:13'])

    assert result.get('offline') is True, f'expected offline=True, got: {result}'
    assert result.get('error'), f'expected non-empty error string, got: {result}'


@pytest.mark.asyncio
async def test_fetch_external_statuses_returns_offline_marker_on_all_non_dict(dummy_config):
    """fetch_external_statuses must return offline marker when all URLs return non-dicts.

    Fails today because non-dict results just continue to the empty {} fallback.
    """
    from dashboard.data.tasks import fetch_external_statuses

    async def _bad_result(*args, **kwargs):
        return ['not', 'a', 'dict']

    with patch('dashboard.data.tasks.mcp_tool_call', side_effect=_bad_result):
        async with httpx.AsyncClient() as client:
            result = await fetch_external_statuses(client, dummy_config, ['dark_factory:13'])

    assert result.get('offline') is True, f'expected offline=True, got: {result}'
    assert result.get('error'), f'expected non-empty error string, got: {result}'


@pytest.mark.asyncio
async def test_fetch_external_statuses_empty_deps_still_returns_empty_dict(dummy_config):
    """Empty deps must still return {} (benign short-circuit), NOT the offline marker."""
    from dashboard.data.tasks import fetch_external_statuses

    with patch('dashboard.data.tasks.mcp_tool_call', side_effect=AssertionError('should not be called')):
        async with httpx.AsyncClient() as client:
            result = await fetch_external_statuses(client, dummy_config, [])

    assert result == {}, f'empty deps must return {{}}, got: {result}'


# ---------------------------------------------------------------------------
# TestFetchTasksCache — per-project_root TTL cache inside fetch_tasks
# (step-1 core-contract tests RED; step-3 TTL-expiry test RED)
# ---------------------------------------------------------------------------

# Canned raw MCP get_tasks rows used across cache tests.
_CACHE_DONE_TASK_RAW = {
    'id': '7',
    'title': 'A done task',
    'status': 'done',
    'updatedAt': '2026-05-29T10:00:00+00:00',
    'description': 'finished',
    'details': '',
    'dependencies': [],
    'metadata': {},
}
_CACHE_PENDING_TASK_RAW = {
    'id': '8',
    'title': 'A pending task',
    'status': 'pending',
    'description': '',
    'details': '',
    'dependencies': [],
    'metadata': {},
}
_CANNED_GET_TASKS_RESULT = {'tasks': [_CACHE_DONE_TASK_RAW, _CACHE_PENDING_TASK_RAW]}


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
        from dashboard.data.tasks import fetch_tasks

        mock_mcp = AsyncMock(return_value=_CANNED_GET_TASKS_RESULT)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            result1 = await fetch_tasks(dummy_client, dummy_config, '/proj/A')
            result2 = await fetch_tasks(dummy_client, dummy_config, '/proj/A')

        assert mock_mcp.call_count == 1, (
            f'expected exactly 1 MCP call within TTL, got {mock_mcp.call_count}'
        )
        # The single call was 'get_tasks' for the correct project_root.
        positional = mock_mcp.call_args_list[0].args
        assert positional[2] == 'get_tasks'
        assert positional[3] == {'project_root': '/proj/A'}
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

        async def _per_root(client, url, tool, args):
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
        self, dummy_client, dummy_config
    ):
        """Offline markers ({offline: True}) are never pinned in the cache.

        First call: ConnectError → offline dict returned.
        Second call: mock recovers → MCP call issued (call_count == 2); list returned.
        """
        from dashboard.data.tasks import fetch_tasks

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

    async def test_fetch_tasks_returned_list_is_a_copy(
        self, dummy_client, dummy_config
    ):
        """Mutating the returned list must not corrupt the cached entry (copy isolation).

        A shallow list() copy is returned on each cache hit so callers cannot
        mutate the internally stored list.
        """
        from dashboard.data.tasks import fetch_tasks

        mock_mcp = AsyncMock(return_value=_CANNED_GET_TASKS_RESULT)
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

        mock_mcp = AsyncMock(return_value=_CANNED_GET_TASKS_RESULT)
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

        mock_mcp = AsyncMock(return_value=_CANNED_GET_TASKS_RESULT)
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
