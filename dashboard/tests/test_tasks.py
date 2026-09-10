"""Unit tests for dashboard.data.tasks._shape_task and fetch_external_statuses.

Focus: the field-mapping contract at the MCP→dashboard boundary and the
fetch_external_statuses short-circuit + fail-safe semantics.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import dashboard.data.tasks as tasks_mod
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


class TestTasksReadRecord:
    """The structured cache key: `_OnePage` | `_CompleteRead` inside `_TasksRead`.

    RED source (step-2): dashboard.data.tasks has no _TasksRead yet.

    These records replace the hand-encoded key string built by
    ``_fetch_tasks_cache_key`` (the ``*`` sentinel, ``|`` separators and
    ``\x1f`` unit separator). Both defects that encoding produced are
    reproduced here as assertions, so neither can come back:

      * OFFSET DRIFT — the encoder rendered ``|o={offset}`` unconditionally
        while ``offset`` only reached the wire when ``page_size`` was set, so
        two reads with a byte-identical wire request minted two entries.
        After this change the only read carrying an offset is ``_OnePage``,
        which always sends one, so the invalid combination is not
        constructible.
      * STATUSES ORDER — ``['a','b']`` and ``['b','a']`` rendered as
        ``s=a\x1fb`` vs ``s=b\x1fa``: two entries for one order-insensitive
        SQL ``IN`` list. ``frozenset`` collapses them.
    """

    def test_the_three_records_exist(self):
        """_OnePage, _CompleteRead and _TasksRead are importable records."""
        assert dataclasses.is_dataclass(tasks_mod._OnePage)
        assert dataclasses.is_dataclass(tasks_mod._CompleteRead)
        assert dataclasses.is_dataclass(tasks_mod._TasksRead)

    @pytest.mark.parametrize(
        ('name', 'build'),
        [
            ('_OnePage', lambda m: m._OnePage(10, 0)),
            ('_CompleteRead', lambda m: m._CompleteRead(None)),
            (
                '_TasksRead',
                lambda m: m._TasksRead('/r', None, m._CompleteRead(None)),
            ),
        ],
    )
    def test_each_record_is_frozen_slotted_and_hashable(self, name, build):
        """A cache key that can be mutated or grow a __dict__ is not a key.

        Frozen because a dict key mutated after insertion is unfindable;
        slotted because these are minted per read and the key space is bounded
        by disuse rather than cardinality (TTLCache._evict_expired), so the
        per-instance dict is pure waste.
        """
        record = build(tasks_mod)

        field = dataclasses.fields(record)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(record, field, 'mutated')

        assert not hasattr(record, '__dict__'), (
            f'{name} is not slotted — it grew a __dict__'
        )

        # Hashable AND usable as a real dict key, which is the actual
        # requirement — hash() alone would pass for an unhashable-by-eq type.
        assert isinstance(hash(record), int)
        assert {record: 'v'}[build(tasks_mod)] == 'v'

    @pytest.mark.parametrize(
        ('name', 'call'),
        [
            ('_OnePage', lambda m: m._OnePage(10)),
            ('_CompleteRead', lambda m: m._CompleteRead()),
            ('_TasksRead', lambda m: m._TasksRead('/r', None)),
        ],
    )
    def test_no_field_carries_a_default(self, name, call):
        """An omitted field must be a loud TypeError, never a silent key.

        This is the runtime shadow of the pyright construction check. With a
        default, a future read mode that forgets to enter a field mints a
        valid-but-WRONG key and collides silently with an unrelated read;
        without one it fails at construction, and it fails even where pyright
        is not run.
        """
        with pytest.raises(TypeError):
            call(tasks_mod)

    def test_statuses_order_does_not_change_the_key(self):
        """['a','b'] and ['b','a'] are ONE read, so they are ONE key.

        The old encoder rendered s=a\x1fb vs s=b\x1fa — two entries for an
        identical, order-insensitive SQL IN list. Reproduced live on this tip
        before the change.
        """
        ab = tasks_mod._TasksRead(
            '/r', frozenset({'a', 'b'}), tasks_mod._CompleteRead(None)
        )
        ba = tasks_mod._TasksRead(
            '/r', frozenset({'b', 'a'}), tasks_mod._CompleteRead(None)
        )
        assert ab == ba
        assert hash(ab) == hash(ba)

    def test_none_statuses_is_distinct_from_empty_statuses(self):
        """None (whole tree) and frozenset() (no tasks at all) are opposites.

        The old encoder spelled these ``s=*`` vs ``s=`` and the distinction was
        load-bearing: collapsing them onto one key serves an empty list as if
        it were the full tree.
        """
        whole_tree = tasks_mod._TasksRead(
            '/r', None, tasks_mod._CompleteRead(None)
        )
        nothing = tasks_mod._TasksRead(
            '/r', frozenset(), tasks_mod._CompleteRead(None)
        )
        assert whole_tree != nothing

    def test_a_page_and_a_same_size_walk_are_distinct_and_named(self):
        """THE esc-4360-7 COLLISION, as the user-observable signal.

        A 10-row PAGE and the complete tree walked 10 rows at a time differ in
        length and content while agreeing on every other component. The
        discriminator is now the record's TYPE — readable as a named field,
        not parsed out of a string.
        """
        page = tasks_mod._TasksRead('/r', None, tasks_mod._OnePage(10, 0))
        walk = tasks_mod._TasksRead('/r', None, tasks_mod._CompleteRead(10))

        assert page != walk
        assert type(page.mode) is not type(walk.mode)

        # `isinstance` rather than a bare attribute read, because that is how
        # production code must consume this union — and pyright ENFORCES it:
        # reading `.page_size` off the un-narrowed `_OnePage | _CompleteRead`
        # is a type error (MEASURED). That the checker refuses the shortcut is
        # itself the proof that the discriminator is real rather than
        # decorative, which is exactly what this test exists to pin.
        assert isinstance(page.mode, tasks_mod._OnePage)
        assert isinstance(walk.mode, tasks_mod._CompleteRead)
        assert page.mode.page_size == 10
        assert walk.mode.chunk_size == 10

    def test_wire_arguments_omits_statuses_entirely_when_none(self):
        """statuses=None means "whole tree": the key is absent from the wire."""
        read = tasks_mod._TasksRead('/r', None, tasks_mod._CompleteRead(None))
        assert read.wire_arguments(None) == {'project_root': '/r'}

    def test_wire_arguments_sends_an_empty_statuses_list(self):
        """frozenset() must be SENT, not dropped.

        Guard on ``is not None``, never truthiness — ``frozenset()`` is falsy,
        so a truthiness guard would silently turn "no tasks at all" into
        "whole tree".
        """
        read = tasks_mod._TasksRead(
            '/r', frozenset(), tasks_mod._CompleteRead(None)
        )
        assert read.wire_arguments(None) == {'project_root': '/r', 'statuses': []}

    def test_wire_arguments_canonicalises_statuses_order(self):
        """A frozenset has no iteration order, so the wire list is sorted.

        Deterministic wire bytes are strictly better than frozenset order,
        which would make every log line and mock assertion nondeterministic.
        """
        read = tasks_mod._TasksRead(
            '/r', frozenset({'pending', 'done', 'blocked'}),
            tasks_mod._CompleteRead(None),
        )
        assert read.wire_arguments(None)['statuses'] == ['blocked', 'done', 'pending']

    def test_wire_arguments_adds_page_size_and_offset_together_or_not_at_all(self):
        """The pair is what the tool needs; half of it is the drift defect.

        The old encoder put ``offset`` in the KEY unconditionally while the
        wire only carried it alongside ``page_size``. One encoder building both
        is what makes that disagreement unrepresentable.
        """
        read = tasks_mod._TasksRead('/r', None, tasks_mod._CompleteRead(None))

        without = read.wire_arguments(None)
        assert 'page_size' not in without and 'offset' not in without

        windowed = read.wire_arguments(tasks_mod._OnePage(25, 50))
        assert windowed['page_size'] == 25
        assert windowed['offset'] == 50

    def test_a_walk_varies_the_window_across_pages_of_one_record(self):
        """The walk re-uses one record across pages, varying only the window.

        Each page of a chunked complete read is the SAME _TasksRead with a
        different window, which is why the encoder takes a window at all. A
        ``_CompleteRead`` mode fixes no window of its own, so this is the one
        mode that may supply one.
        """
        read = tasks_mod._TasksRead(
            '/r', frozenset({'done'}), tasks_mod._CompleteRead(10)
        )
        first = read.wire_arguments(tasks_mod._OnePage(10, 0))
        second = read.wire_arguments(tasks_mod._OnePage(10, 10))

        assert first == {
            'project_root': '/r', 'statuses': ['done'], 'page_size': 10, 'offset': 0,
        }
        assert second['offset'] == 10

    def test_a_page_read_takes_its_wire_window_from_its_own_mode(self):
        """The bytes on the wire come from the field the KEY hashes.

        A page read passes no window: `_OnePage` already fixes one, so the
        request is derived from `mode` itself. That is what makes the offset
        drift the old two-encoder shape produced unrepresentable rather than
        merely absent — there is no second copy to disagree with.
        """
        read = tasks_mod._TasksRead('/r', None, tasks_mod._OnePage(25, 50))

        assert read.wire_arguments(None) == {
            'project_root': '/r', 'page_size': 25, 'offset': 50,
        }

    def test_a_page_read_refuses_a_second_window(self):
        """Offering a window to a page read RAISES rather than overriding.

        Silently preferring either one would reintroduce exactly the key/wire
        disagreement this record exists to eliminate: the key would carry
        `mode` while the wire carried the argument. `TypeError`, not
        `ValueError` — `first_success` treats `ValueError` as a soft per-url
        failure and would launder a programming error into an offline marker.
        """
        read = tasks_mod._TasksRead('/r', None, tasks_mod._OnePage(25, 50))

        with pytest.raises(TypeError):
            read.wire_arguments(tasks_mod._OnePage(25, 75))


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

        mock_mcp = AsyncMock(return_value=_CANNED_GET_TASKS_RESULT)
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
            return _CANNED_GET_TASKS_RESULT

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
# TestFetchTasksNarrowing — server-side narrowing args on the get_tasks wire
# (task 3857 step-1 RED; step-3 adds the cache-key discrimination tests)
# ---------------------------------------------------------------------------


class TestFetchTasksNarrowing:
    """``fetch_tasks``'s narrowing arguments as a wire contract.

    The whole point of the narrowing work is that the *server* does the
    filtering, so what matters is the arguments dict that actually crosses
    the MCP boundary — not what the dashboard discards afterwards. These
    tests therefore assert on ``mcp_tool_call``'s recorded call args.

    The unnarrowed shape is pinned byte-identical because four callers
    (``app._load_task_cards``, ``data/orchestrator.py``,
    ``data/merge_queue.py``, ``data/burndown.py``) still need the full tree
    and must be unaffected by this change.
    """

    @pytest.fixture(autouse=True)
    def reset_fetch_tasks_cache(self):
        import dashboard.data.tasks as tasks_mod
        tasks_mod._fetch_tasks_cache_clear()
        yield
        tasks_mod._fetch_tasks_cache_clear()

    @staticmethod
    def _args_of(mock_mcp, index=0):
        """Return the arguments dict of the *index*-th recorded MCP call."""
        return mock_mcp.call_args_list[index].args[3]

    async def test_unnarrowed_call_sends_project_root_only(
        self, dummy_client, dummy_config
    ):
        """(a) No narrowing → the arguments dict is EXACTLY {'project_root': ...}.

        Backward-compatibility guard for the four full-tree callers: no
        ``statuses``, no ``page_size``, no ``offset`` key may appear.
        """
        from dashboard.data.tasks import fetch_tasks

        mock_mcp = AsyncMock(return_value=_CANNED_GET_TASKS_RESULT)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            await fetch_tasks(dummy_client, dummy_config, '/proj/A')

        assert mock_mcp.call_count == 1
        assert self._args_of(mock_mcp) == {'project_root': '/proj/A'}, (
            'the unnarrowed arguments dict must stay byte-identical for the '
            'four full-tree callers'
        )

    async def test_statuses_forwarded_in_canonical_order(
        self, dummy_client, dummy_config
    ):
        """(b) ``statuses`` crosses the wire ``sorted()`` — canonical, not verbatim.

        AMENDS the former `test_statuses_forwarded_verbatim`, which asserted
        the list was "not re-sorted". Statuses are now held as a
        `frozenset[str]` so that `['a','b']` and `['b','a']` hit ONE cache
        entry, and a frozenset has no iteration order — something must
        canonicalise the wire list. `sorted()` is the only sane choice:
        arbitrary frozenset order would make the wire bytes nondeterministic
        across runs and this very assertion FLAKY.

        Safe against production: `get_tasks` turns `statuses` into a SQL `IN`
        list, which is order-insensitive by construction, and both live
        narrowing call sites already pass `sorted(...)`. So the wire does not
        actually move — the test just pins a strictly stronger property.
        """
        from dashboard.data.tasks import fetch_tasks

        mock_mcp = AsyncMock(return_value=_CANNED_GET_TASKS_RESULT)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            await fetch_tasks(
                dummy_client, dummy_config, '/proj/A',
                # Deliberately UNSORTED input, so a pass-through implementation
                # could not satisfy this by accident.
                statuses=['pending', 'in-progress'],
            )

        assert self._args_of(mock_mcp) == {
            'project_root': '/proj/A',
            'statuses': ['in-progress', 'pending'],
        }

    async def test_empty_statuses_list_is_sent_not_dropped(
        self, dummy_client, dummy_config
    ):
        """``statuses=[]`` is a valid 'return nothing' request, distinct from None.

        A falsy-check implementation would drop it and silently request the
        whole tree — the exact defect this task exists to remove.
        """
        from dashboard.data.tasks import fetch_tasks

        mock_mcp = AsyncMock(return_value={'tasks': []})
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            await fetch_tasks(dummy_client, dummy_config, '/proj/A', statuses=[])

        assert self._args_of(mock_mcp) == {'project_root': '/proj/A', 'statuses': []}

    async def test_page_size_and_offset_added_together(
        self, dummy_client, dummy_config
    ):
        """(c) ``page_size``/``offset`` add exactly those two keys.

        Now a `fetch_task_page` call: a window is the PARTIAL read's whole
        reason to exist, so it moved with the contract.
        """
        from dashboard.data.tasks import fetch_task_page

        mock_mcp = AsyncMock(return_value=_CANNED_GET_TASKS_RESULT)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            await fetch_task_page(
                dummy_client, dummy_config, '/proj/A', page_size=100, offset=25,
            )

        assert self._args_of(mock_mcp) == {
            'project_root': '/proj/A',
            'page_size': 100,
            'offset': 25,
        }

    # RETIRED: `test_offset_omitted_when_page_size_is_none`, which called
    # `fetch_tasks(..., offset=25)` and asserted the wire stayed
    # `{'project_root': '/proj/A'}`. `fetch_tasks` no longer HAS an offset to
    # omit — the only read carrying one is `fetch_task_page`, which always
    # sends it. `TestPublicReadContracts.test_fetch_tasks_rejects_offset` is
    # the strictly stronger statement that replaces it: unrepresentable beats
    # representable-but-dropped.

    async def test_statuses_and_page_size_compose(self, dummy_client, dummy_config):
        """The terminal-window call shape: narrowed statuses PLUS a bounded window."""
        from dashboard.data.tasks import fetch_task_page

        mock_mcp = AsyncMock(return_value=_CANNED_GET_TASKS_RESULT)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            await fetch_task_page(
                dummy_client, dummy_config, '/proj/A',
                statuses=['cancelled', 'done'], page_size=400, offset=3600,
            )

        assert self._args_of(mock_mcp) == {
            'project_root': '/proj/A',
            'statuses': ['cancelled', 'done'],
            'page_size': 400,
            'offset': 3600,
        }

    @pytest.mark.parametrize(('reader', 'kwargs'), [
        ('fetch_tasks', {}),
        ('fetch_tasks', {'statuses': ['pending']}),
        # The windowed case routes to the PARTIAL read — `fetch_tasks` has no
        # window any more. Both public readers must carry the budget, which is
        # why the parametrize now varies the function too.
        ('fetch_task_page', {'page_size': 10, 'offset': 5}),
    ])
    async def test_every_call_carries_the_per_request_budget(
        self, dummy_client, dummy_config, reader, kwargs
    ):
        """(d) Every call — narrowed or not — passes ``timeout=`` as a keyword."""
        import dashboard.data.tasks as tasks_mod

        read = getattr(tasks_mod, reader)
        mock_mcp = AsyncMock(return_value=_CANNED_GET_TASKS_RESULT)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            await read(dummy_client, dummy_config, '/proj/A', **kwargs)

        call = mock_mcp.call_args_list[0]
        assert call.kwargs.get('timeout') == tasks_mod.DEFAULT_PER_CALL_TIMEOUT

    async def test_explicit_timeout_overrides_the_default(
        self, dummy_client, dummy_config
    ):
        """A caller may tighten the per-request budget further."""
        from dashboard.data.tasks import fetch_tasks

        mock_mcp = AsyncMock(return_value=_CANNED_GET_TASKS_RESULT)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            await fetch_tasks(dummy_client, dummy_config, '/proj/A', timeout=0.5)

        assert mock_mcp.call_args_list[0].kwargs.get('timeout') == 0.5

    def test_default_per_call_timeout_only_ever_tightens(self):
        """(d) The budget must be strictly BELOW ``mcp_tool_call``'s own default.

        Standing guard for the plan's MUST NOT: this task never raises a
        probe budget. Read out of the live signature so a future widening of
        ``mcp_tool_call``'s default cannot silently relax this.
        """
        import inspect

        import dashboard.data.memory as memory_mod
        import dashboard.data.tasks as tasks_mod

        mcp_default = inspect.signature(
            memory_mod.mcp_tool_call
        ).parameters['timeout'].default
        assert isinstance(mcp_default, (int, float))
        assert mcp_default > tasks_mod.DEFAULT_PER_CALL_TIMEOUT, (
            f'DEFAULT_PER_CALL_TIMEOUT={tasks_mod.DEFAULT_PER_CALL_TIMEOUT} must be '
            f'strictly tighter than mcp_tool_call default={mcp_default}'
        )
        assert tasks_mod.DEFAULT_PER_CALL_TIMEOUT > 0


    # -----------------------------------------------------------------
    # Cache-key discrimination (task 3857 step-3)
    #
    # fetch_tasks has five callers and only ONE of them narrows. Keying the
    # TTL cache on the bare project_root would let active_tasks' narrowed
    # entry be served to app._load_task_cards / merge_queue.load_task_titles
    # / burndown.collect_snapshot / data.orchestrator, silently truncating
    # them for up to the 20 s TTL — non-deterministically, depending on
    # which caller raced in first.
    #
    # Each test below uses distinct per-narrowing payloads so a cross-served
    # entry is detectable by CONTENT, not merely by call count.
    # -----------------------------------------------------------------

    @staticmethod
    def _payload(task_id: int, title: str, status: str = 'pending') -> dict:
        return {'tasks': [{
            'id': str(task_id), 'title': title, 'status': status,
            'dependencies': [], 'metadata': {},
        }]}

    async def test_differing_statuses_key_separately(
        self, dummy_client, dummy_config
    ):
        """(a) Same root, different ``statuses``, within TTL → two MCP calls."""
        from dashboard.data.tasks import fetch_tasks

        active_payload = self._payload(1, 'ACTIVE ROW', 'in-progress')
        terminal_payload = self._payload(2, 'TERMINAL ROW', 'done')

        async def _by_statuses(client, url, tool, args, **_kw):
            if args.get('statuses') == ['in-progress']:
                return active_payload
            if args.get('statuses') == ['done']:
                return terminal_payload
            raise AssertionError(f'unexpected statuses: {args.get("statuses")!r}')

        mock_mcp = AsyncMock(side_effect=_by_statuses)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            active = await fetch_tasks(
                dummy_client, dummy_config, '/proj/K', statuses=['in-progress'],
            )
            terminal = await fetch_tasks(
                dummy_client, dummy_config, '/proj/K', statuses=['done'],
            )

        assert mock_mcp.call_count == 2, (
            f'differing statuses must key separately, got {mock_mcp.call_count} call(s)'
        )
        assert [t['title'] for t in active] == ['ACTIVE ROW']
        assert [t['title'] for t in terminal] == ['TERMINAL ROW'], (
            'the second narrowing was served the first narrowing’s rows'
        )

    async def test_narrowed_entry_never_served_to_the_full_tree_caller(
        self, dummy_client, dummy_config
    ):
        """(b) A narrowed call must not poison the unnarrowed callers' entry.

        This is the concrete production bug: ``active_tasks`` narrows while
        ``app._load_task_cards`` / ``merge_queue.load_task_titles`` /
        ``burndown.collect_snapshot`` do not.
        """
        from dashboard.data.tasks import fetch_tasks

        narrowed_payload = self._payload(1, 'NARROWED ONLY', 'in-progress')
        full_payload = {'tasks': [
            {'id': '1', 'title': 'NARROWED ONLY', 'status': 'in-progress',
             'dependencies': [], 'metadata': {}},
            {'id': '2', 'title': 'FULL TREE EXTRA', 'status': 'done',
             'dependencies': [], 'metadata': {}},
        ]}

        async def _by_narrowing(client, url, tool, args, **_kw):
            return narrowed_payload if 'statuses' in args else full_payload

        mock_mcp = AsyncMock(side_effect=_by_narrowing)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            await fetch_tasks(
                dummy_client, dummy_config, '/proj/L', statuses=['in-progress'],
            )
            full = await fetch_tasks(dummy_client, dummy_config, '/proj/L')

        assert mock_mcp.call_count == 2, (
            'the unnarrowed caller must issue its own MCP call, not ride the '
            f'narrowed entry (got {mock_mcp.call_count} call(s))'
        )
        assert [t['title'] for t in full] == ['NARROWED ONLY', 'FULL TREE EXTRA'], (
            'the full-tree caller was served a status-filtered subset'
        )

    async def test_unnarrowed_entry_never_served_to_the_narrowed_caller(
        self, dummy_client, dummy_config
    ):
        """(b, reversed) Order must not matter — the full entry is not a narrowed one."""
        from dashboard.data.tasks import fetch_tasks

        async def _by_narrowing(client, url, tool, args, **_kw):
            if 'statuses' in args:
                return self._payload(1, 'NARROWED ONLY', 'in-progress')
            return {'tasks': [
                {'id': '1', 'title': 'NARROWED ONLY', 'status': 'in-progress',
                 'dependencies': [], 'metadata': {}},
                {'id': '2', 'title': 'FULL TREE EXTRA', 'status': 'done',
                 'dependencies': [], 'metadata': {}},
            ]}

        mock_mcp = AsyncMock(side_effect=_by_narrowing)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            await fetch_tasks(dummy_client, dummy_config, '/proj/M')
            narrowed = await fetch_tasks(
                dummy_client, dummy_config, '/proj/M', statuses=['in-progress'],
            )

        assert mock_mcp.call_count == 2
        assert [t['title'] for t in narrowed] == ['NARROWED ONLY']

    async def test_none_statuses_keys_distinctly_from_empty_list(
        self, dummy_client, dummy_config
    ):
        """``statuses=None`` and ``statuses=[]`` are opposite requests, not one key."""
        from dashboard.data.tasks import fetch_tasks

        async def _by_narrowing(client, url, tool, args, **_kw):
            if args.get('statuses') == []:
                return {'tasks': []}
            return self._payload(1, 'FULL TREE', 'pending')

        mock_mcp = AsyncMock(side_effect=_by_narrowing)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            full = await fetch_tasks(dummy_client, dummy_config, '/proj/N')
            empty = await fetch_tasks(dummy_client, dummy_config, '/proj/N', statuses=[])

        assert mock_mcp.call_count == 2
        assert [t['title'] for t in full] == ['FULL TREE']
        assert empty == []

    async def test_differing_page_size_and_offset_key_separately(
        self, dummy_client, dummy_config
    ):
        """(c) The window position is part of the identity of a result."""
        from dashboard.data.tasks import fetch_task_page

        async def _by_window(client, url, tool, args, **_kw):
            return self._payload(
                args.get('offset', 0) or 1,
                f'window p={args.get("page_size")} o={args.get("offset")}',
            )

        mock_mcp = AsyncMock(side_effect=_by_window)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            first = await fetch_task_page(
                dummy_client, dummy_config, '/proj/O', page_size=10, offset=0,
            )
            second = await fetch_task_page(
                dummy_client, dummy_config, '/proj/O', page_size=10, offset=10,
            )
            third = await fetch_task_page(
                dummy_client, dummy_config, '/proj/O', page_size=20, offset=10,
            )

        assert mock_mcp.call_count == 3, (
            f'page_size/offset must key separately, got {mock_mcp.call_count} call(s)'
        )
        assert first[0]['title'] == 'window p=10 o=0'
        assert second[0]['title'] == 'window p=10 o=10'
        assert third[0]['title'] == 'window p=20 o=10'

    async def test_identical_narrowing_still_single_flights(
        self, dummy_client, dummy_config
    ):
        """(d) Regression guard — the existing single-flight contract is unchanged."""
        from dashboard.data.tasks import fetch_task_page

        mock_mcp = AsyncMock(return_value=_CANNED_GET_TASKS_RESULT)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            first = await fetch_task_page(
                dummy_client, dummy_config, '/proj/P',
                statuses=['pending'], page_size=50, offset=5,
            )
            second = await fetch_task_page(
                dummy_client, dummy_config, '/proj/P',
                statuses=['pending'], page_size=50, offset=5,
            )

        assert mock_mcp.call_count == 1, (
            f'identical narrowing within TTL must reuse the entry, got '
            f'{mock_mcp.call_count} call(s)'
        )
        assert first == second
        assert isinstance(first, list)

    async def test_narrowed_keys_stay_per_project_root(
        self, dummy_client, dummy_config
    ):
        """The same narrowing on two roots must not collapse onto one entry."""
        from dashboard.data.tasks import fetch_tasks

        async def _by_root(client, url, tool, args, **_kw):
            return self._payload(1, f'row for {args["project_root"]}')

        mock_mcp = AsyncMock(side_effect=_by_root)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            a = await fetch_tasks(
                dummy_client, dummy_config, '/proj/Q', statuses=['pending'],
            )
            b = await fetch_tasks(
                dummy_client, dummy_config, '/proj/R', statuses=['pending'],
            )

        assert mock_mcp.call_count == 2
        assert a[0]['title'] == 'row for /proj/Q'
        assert b[0]['title'] == 'row for /proj/R'


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

        mock_mcp = AsyncMock(return_value=_CANNED_GET_TASKS_RESULT)
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
            return _CANNED_GET_TASKS_RESULT

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
            side_effect=self._per_root_side_effect(root_a, _CANNED_GET_TASKS_RESULT)
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
        # fetch_statuses caches SUCCESSES only, so the failing root genuinely
        # re-polls every cycle — which is exactly the streak this asserts on.
        # (The healthy root rides its 5 s entry; that does not touch the
        # per-label throttle being measured here.)
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


# ---------------------------------------------------------------------------
# TestFetchTasksPagination — opt-in page_size seam (task 4360, finding 4)
# ---------------------------------------------------------------------------


def _paged_task_raw(tid: int) -> dict:
    """One raw MCP get_tasks row, distinguishable by id."""
    return {
        'id': str(tid),
        'title': f'task {tid}',
        'status': 'in-progress',
        'description': '',
        'details': '',
        'dependencies': [],
        'metadata': {},
    }


def _paging_mcp(tasks: list[dict], calls: list[dict], *, envelope: bool = True):
    """An ``mcp_tool_call`` stub that pages exactly as MCP ``get_tasks`` does.

    Mirrors fused-memory ``server/tools.py`` (the ``_pagination_meta``
    envelope): when ``page_size`` is absent the full list comes back with NO
    ``pagination`` key at all — the backward-compatible shape every existing
    caller relies on.  When it is present, the response carries the page plus
    ``{total, offset, page_size, returned, has_more}``.

    ``envelope=False`` stands in for an OLDER fused-memory that ignores
    ``page_size`` and answers with a bare, unpaginated list.
    """

    async def _call(_client, _url, tool, args, **_kwargs):
        assert tool == 'get_tasks', tool
        calls.append(dict(args))
        page_size = args.get('page_size')
        if page_size is None or not envelope:
            return {'tasks': list(tasks)}
        offset = args.get('offset', 0)
        page = tasks[offset:offset + page_size]
        return {
            'tasks': page,
            'pagination': {
                'total': len(tasks),
                'offset': offset,
                'page_size': page_size,
                'returned': len(page),
                'has_more': offset + len(page) < len(tasks),
            },
        }

    return _call


class TestFetchTasksPagination:
    """``fetch_tasks(..., chunk_size=N)`` — opt-in, delivery-only pagination.

    The burndown collector reads the whole task tree every cycle and writes one
    row per cycle into an APPEND-ONLY history table.  An oversize MCP response
    is rejected wholesale, the collector's ``not isinstance(result, list)``
    guard logs and continues, and that cycle's row is never written — a
    permanent hole no later cycle backfills.  There is no field-limited read to
    reach for (MCP ``get_tasks`` exposes only project_root/tag/page_size/
    offset/statuses and the backend query is ``SELECT *``), so bounding the
    per-RESPONSE size is the available lever.

    Pagination must be a pure delivery detail: opt-in, invisible to every
    existing caller, and assembling to exactly the list a single request would
    have produced.
    """

    @pytest.fixture(autouse=True)
    def reset_fetch_tasks_cache(self):
        import dashboard.data.tasks as tasks_mod
        tasks_mod._fetch_tasks_cache_clear()
        yield
        tasks_mod._fetch_tasks_cache_clear()

    async def test_pages_are_assembled_in_order(self, dummy_client, dummy_config):
        """7 tasks at page_size=3 → 3 requests, all 7 rows, original order."""
        from dashboard.data.tasks import fetch_tasks

        tasks = [_paged_task_raw(i) for i in range(1, 8)]
        calls: list[dict] = []
        with patch(
            'dashboard.data.tasks.mcp_tool_call',
            new=AsyncMock(side_effect=_paging_mcp(tasks, calls)),
        ):
            result = await fetch_tasks(
                dummy_client, dummy_config, '/proj/pag', chunk_size=3,
            )

        assert isinstance(result, list)
        assert [t['id'] for t in result] == list(range(1, 8)), (
            f'every page must be accumulated, in order; got {result!r}'
        )
        assert [c.get('offset') for c in calls] == [0, 3, 6], (
            f'offset must advance by the returned count; got {calls!r}'
        )
        assert all(c.get('page_size') == 3 for c in calls), calls

    async def test_default_call_is_byte_identical_to_today(
        self, dummy_client, dummy_config,
    ):
        """No page_size → ONE request carrying no page_size/offset key.

        This is the backward-compatibility guarantee for ``active_tasks`` and
        every other request-path caller: they must keep sending exactly
        ``{'project_root': ...}``, so the wire shape they have always sent —
        and the unpaginated response they have always parsed — is unchanged.
        """
        from dashboard.data.tasks import fetch_tasks

        tasks = [_paged_task_raw(i) for i in range(1, 8)]
        calls: list[dict] = []
        with patch(
            'dashboard.data.tasks.mcp_tool_call',
            new=AsyncMock(side_effect=_paging_mcp(tasks, calls)),
        ):
            result = await fetch_tasks(dummy_client, dummy_config, '/proj/plain')

        assert isinstance(result, list)
        assert len(result) == 7
        assert calls == [{'project_root': '/proj/plain'}], (
            f'the unpaginated request shape must not change; got {calls!r}'
        )

    async def test_a_server_that_ignores_page_size_terminates_after_one_page(
        self, dummy_client, dummy_config,
    ):
        """No ``pagination`` key back → take that page and stop.

        An older fused-memory ignores ``page_size`` and answers with the whole
        bare list.  Without an explicit terminator the loop has no ``total`` to
        compare against and would either spin or silently truncate; it must
        degrade to exactly today's single-request behaviour instead.
        """
        from dashboard.data.tasks import fetch_tasks

        tasks = [_paged_task_raw(i) for i in range(1, 8)]
        calls: list[dict] = []
        with patch(
            'dashboard.data.tasks.mcp_tool_call',
            new=AsyncMock(side_effect=_paging_mcp(tasks, calls, envelope=False)),
        ):
            result = await fetch_tasks(
                dummy_client, dummy_config, '/proj/old', chunk_size=3,
            )

        assert isinstance(result, list)
        assert [t['id'] for t in result] == list(range(1, 8))
        assert len(calls) == 1, (
            f'an envelope-less response must end the loop; got {calls!r}'
        )

    async def test_a_non_advancing_server_yields_the_offline_marker_not_a_truncated_list(
        self, dummy_client, dummy_config,
    ):
        """``returned == 0`` with rows still owed → offline marker, never ``[]``.

        A hang is no longer the hazard: the loop terminates either way.  The
        hazard is the SHAPE of what a bounded-but-incomplete read hands back.
        ``fetch_tasks``'s contract distinguishes only ``list`` (a complete
        success) from the ``{'offline': True}`` marker, so a TRUNCATED list is
        indistinguishable from a complete one at every call site.
        ``collect_snapshot`` triages on ``isinstance(result, list)`` and would
        write ``_count_zones([])`` — a confident zero — into the APPEND-ONLY
        ``snapshots`` table for a tree the server itself reported as holding 99
        rows.  A fabricated dip in an append-only chart is unfalsifiable after
        the fact; a gap is visible.  So: raise, fall through the fan-out, and
        surface the marker the collector already skips on.
        """
        from dashboard.data.tasks import fetch_tasks

        calls: list[dict] = []

        async def _stuck(_client, _url, tool, args, **_kwargs):
            calls.append(dict(args))
            return {
                'tasks': [],
                'pagination': {
                    'total': 99, 'offset': args.get('offset', 0),
                    'page_size': args.get('page_size'), 'returned': 0,
                    'has_more': True,
                },
            }

        with patch(
            'dashboard.data.tasks.mcp_tool_call', new=AsyncMock(side_effect=_stuck),
        ):
            result = await fetch_tasks(
                dummy_client, dummy_config, '/proj/stuck', chunk_size=3,
            )

        assert isinstance(result, dict), (
            f'a truncated read must NOT be reported as a list; got {result!r}'
        )
        assert result.get('offline') is True, result
        error = str(result.get('error', ''))
        assert '/proj/stuck' in error, (
            f'the error must name the project root so an operator can grep it; got {error!r}'
        )
        assert 'truncat' in error.lower(), (
            f'the error must name the truncation as the cause; got {error!r}'
        )
        # Exactly one call per configured URL: proves BOTH that the loop cannot
        # spin AND that the failure fell through the fan-out (each URL tried
        # once) rather than being swallowed at the first one.
        assert len(calls) == len(dummy_config.fused_memory_urls), (
            f'a non-advancing page must end that URL\'s loop; got {calls!r}'
        )

    async def test_a_non_int_pagination_envelope_yields_the_offline_marker(
        self, dummy_client, dummy_config,
    ):
        """``total``/``returned`` not both ints → unverifiable, so not a success.

        With a malformed envelope there is no way to decide whether the page in
        hand is the whole tree, so completeness is UNVERIFIABLE.  An
        unverifiable read must not be reported as a complete one — the same
        reasoning as the non-advancing case, and the same remedy.
        """
        from dashboard.data.tasks import fetch_tasks

        calls: list[dict] = []

        async def _garbled(_client, _url, tool, args, **_kwargs):
            calls.append(dict(args))
            return {
                'tasks': [_paged_task_raw(1)],
                'pagination': {
                    'total': 'many', 'offset': args.get('offset', 0),
                    'page_size': args.get('page_size'), 'returned': None,
                    'has_more': True,
                },
            }

        with patch(
            'dashboard.data.tasks.mcp_tool_call', new=AsyncMock(side_effect=_garbled),
        ):
            result = await fetch_tasks(
                dummy_client, dummy_config, '/proj/garbled', chunk_size=3,
            )

        assert isinstance(result, dict) and result.get('offline') is True, (
            f'a non-int envelope must not be reported as a complete list; got {result!r}'
        )
        assert '/proj/garbled' in str(result.get('error', '')), result
        assert len(calls) == len(dummy_config.fused_memory_urls), calls

    async def test_a_partial_read_that_stalls_yields_the_marker_not_the_partial_rows(
        self, dummy_client, dummy_config,
    ):
        """Pages 0-1 arrive, then the server stalls → marker, and NO partial rows.

        The most insidious case.  An all-zero row at least looks odd in a chart;
        a PARTIAL count looks entirely plausible, so nobody ever goes looking.
        The partial rows must not leak out as a list.
        """
        from dashboard.data.tasks import fetch_tasks

        tasks = [_paged_task_raw(i) for i in range(1, 8)]
        calls: list[dict] = []

        async def _stalls_after_two_pages(_client, _url, tool, args, **_kwargs):
            calls.append(dict(args))
            offset = args.get('offset', 0)
            page_size = args.get('page_size')
            page = tasks[offset:offset + page_size] if offset < 4 else []
            return {
                'tasks': page,
                'pagination': {
                    'total': len(tasks), 'offset': offset,
                    'page_size': page_size, 'returned': len(page),
                    'has_more': True,
                },
            }

        with patch(
            'dashboard.data.tasks.mcp_tool_call',
            new=AsyncMock(side_effect=_stalls_after_two_pages),
        ):
            result = await fetch_tasks(
                dummy_client, dummy_config, '/proj/stall', chunk_size=2,
            )

        assert isinstance(result, dict) and result.get('offline') is True, (
            f'a partial read must not be handed back as a list; got {result!r}'
        )
        assert '/proj/stall' in str(result.get('error', '')), result
        # 3 requests per URL: offsets 0 and 2 serve rows, offset 4 stalls.
        assert [c.get('offset') for c in calls] == (
            [0, 2, 4] * len(dummy_config.fused_memory_urls)
        ), calls

    async def test_a_genuinely_empty_tree_still_returns_an_empty_list(
        self, dummy_client, dummy_config,
    ):
        """``total == 0, returned == 0`` is a COMPLETE read of an empty project.

        The anti-over-raise guard.  This case passes today and must keep
        passing: a naive "raise whenever ``returned <= 0``" would convert every
        empty project into a permanent burndown hole and suppress its
        legitimate all-zero row — the exact inverse of the bug being fixed, and
        just as invisible.  ``[]`` here is a true zero, not a manufactured one.
        """
        from dashboard.data.tasks import fetch_tasks

        calls: list[dict] = []

        async def _empty(_client, _url, tool, args, **_kwargs):
            calls.append(dict(args))
            return {
                'tasks': [],
                'pagination': {
                    'total': 0, 'offset': args.get('offset', 0),
                    'page_size': args.get('page_size'), 'returned': 0,
                    'has_more': False,
                },
            }

        with patch(
            'dashboard.data.tasks.mcp_tool_call', new=AsyncMock(side_effect=_empty),
        ):
            result = await fetch_tasks(
                dummy_client, dummy_config, '/proj/empty', chunk_size=3,
            )

        assert result == [], (
            f'an empty project is a complete read, not a truncation; got {result!r}'
        )
        assert len(calls) == 1, (
            f'a complete empty read must not fall through the fan-out; got {calls!r}'
        )

    async def test_a_server_whose_returned_count_disagrees_with_the_page_yields_the_marker(
        self, dummy_client, dummy_config,
    ):
        """``returned`` is cross-checked against the rows actually delivered.

        The walk ADVANCES on the server's self-reported ``returned``, so an
        unchecked counter is the one remaining way a bounded-but-incomplete read
        reaches a caller as a plain ``list``.  A server (or a proxy/serialiser
        that clips a page) claiming ``returned=10`` while shipping 4 rows skips
        6 rows per page, terminates NORMALLY at ``offset >= total``, and hands
        back a 40-row list for a 100-task tree — which ``collect_snapshot``
        triages as healthy and writes into the append-only ``snapshots`` table
        as fact.  That is precisely the plausible-dip-nobody-falsifies artefact
        the all-or-nothing rule exists to prevent, so it must surface the marker.

        ``len(shaped)`` cannot stand in for the raw page length: ``_shape_all``
        drops rows whose ``_shape_task`` returns None (an unparseable id), so a
        legitimately-shaped page can be shorter than what arrived.
        """
        from dashboard.data.tasks import fetch_tasks

        tasks = [_paged_task_raw(i) for i in range(1, 101)]
        calls: list[dict] = []

        async def _over_reports(_client, _url, tool, args, **_kwargs):
            calls.append(dict(args))
            offset = args.get('offset', 0)
            page_size = args.get('page_size') or 0
            # Ships 4 rows but claims it sent a full page.
            page = tasks[offset:offset + 4]
            return {
                'tasks': page,
                'pagination': {
                    'total': len(tasks), 'offset': offset,
                    'page_size': page_size, 'returned': page_size,
                    'has_more': True,
                },
            }

        with patch(
            'dashboard.data.tasks.mcp_tool_call', new=AsyncMock(side_effect=_over_reports),
        ):
            result = await fetch_tasks(
                dummy_client, dummy_config, '/proj/miscount', chunk_size=10,
            )

        assert isinstance(result, dict), (
            f'a page whose delivered row count contradicts its own counter is '
            f'not a verified read and must NOT be reported as a list; got {result!r}'
        )
        assert result.get('offline') is True, result
        error = str(result.get('error', ''))
        assert '/proj/miscount' in error, (
            f'the error must name the project root so an operator can grep it; got {error!r}'
        )
        assert 'returned=10' in error and '4 row' in error, (
            f'the error must report BOTH the claimed and the delivered count so '
            f'the disagreement is diagnosable; got {error!r}'
        )
        # One call per configured URL: the disagreement is caught on the FIRST
        # page, so no rows are ever accumulated on a bad counter.
        assert len(calls) == len(dummy_config.fused_memory_urls), (
            f'the miscount must end that URL\'s walk immediately; got {calls!r}'
        )

    async def test_a_total_that_grows_mid_walk_yields_the_marker(
        self, dummy_client, dummy_config,
    ):
        """A ``total`` that GROWS mid-walk is a raced read, not a snapshot.

        ``total`` is the loop's only terminator.  If it climbs while the walk is
        in flight, the tree was written underneath the read: the assembled pages
        come from different states of the world and are a coherent snapshot of
        neither.  It also un-bounds the page budget derived from the first
        response.  Both reasons say the same thing — refuse the read rather than
        hand back a list stitched from two trees.
        """
        from dashboard.data.tasks import fetch_tasks

        tasks = [_paged_task_raw(i) for i in range(1, 101)]
        calls: list[dict] = []

        async def _growing(_client, _url, tool, args, **_kwargs):
            calls.append(dict(args))
            offset = args.get('offset', 0)
            page_size = args.get('page_size') or 0
            page = tasks[offset:offset + page_size]
            # The tree is being filed into while the walk runs: the first page
            # sees a 20-task tree, the second a 50-task one.  Per-URL state, so
            # the fan-out's second attempt observes the same growth.
            total = 20 if offset == 0 else 50
            return {
                'tasks': page,
                'pagination': {
                    'total': total, 'offset': offset,
                    'page_size': page_size, 'returned': len(page),
                    'has_more': True,
                },
            }

        with patch(
            'dashboard.data.tasks.mcp_tool_call', new=AsyncMock(side_effect=_growing),
        ):
            result = await fetch_tasks(
                dummy_client, dummy_config, '/proj/racing', chunk_size=10,
            )

        assert isinstance(result, dict), (
            f'pages stitched from a changing tree are not one snapshot; got {result!r}'
        )
        assert result.get('offline') is True, result
        error = str(result.get('error', ''))
        assert '/proj/racing' in error, error
        assert 'grew from 20 to 50' in error, (
            f'the error must report the observed growth so the race is '
            f'diagnosable; got {error!r}'
        )

    async def test_an_endlessly_short_paged_server_cannot_walk_unbounded(
        self, dummy_client, dummy_config,
    ):
        """The walk is bounded by a budget derived from the FIRST ``total``.

        ``total`` alone does not bound the number of ROUND TRIPS: the loop
        advances by the rows actually delivered, so a server that answers every
        request with one row — while honestly reporting ``returned=1`` — stretches
        a ``ceil(total/page_size)`` walk into a ``total``-request one.  Those are
        sequential, on the SAME shared ``httpx.AsyncClient`` the 2 s render polls
        use, which is the ``httpx.PoolTimeout`` starvation hazard the burndown
        size probe exists to bound; and ``_burndown_loop`` wraps
        ``collect_snapshot`` in a bare ``except Exception`` with no
        ``asyncio.wait_for``, so a wedged walk just stops producing snapshots
        indefinitely instead of failing loudly.

        So the budget is derived ONCE from the first response and enforced:
        ``ceil(100/10) + 2 == 12`` requests, not 100.
        """
        from dashboard.data.tasks import fetch_tasks

        tasks = [_paged_task_raw(i) for i in range(1, 101)]
        calls: list[dict] = []

        async def _one_row_at_a_time(_client, _url, tool, args, **_kwargs):
            calls.append(dict(args))
            offset = args.get('offset', 0)
            # Honest counters throughout — the page is simply always short.
            page = tasks[offset:offset + 1]
            return {
                'tasks': page,
                'pagination': {
                    'total': len(tasks), 'offset': offset,
                    'page_size': args.get('page_size'), 'returned': len(page),
                    'has_more': True,
                },
            }

        with patch(
            'dashboard.data.tasks.mcp_tool_call',
            new=AsyncMock(side_effect=_one_row_at_a_time),
        ):
            result = await fetch_tasks(
                dummy_client, dummy_config, '/proj/dribble', chunk_size=10,
            )

        assert isinstance(result, dict), (
            f'a walk that blew its budget is not a verified read; got {result!r}'
        )
        assert result.get('offline') is True, result
        error = str(result.get('error', ''))
        assert '/proj/dribble' in error, error
        assert 'budget' in error.lower(), (
            f'the error must name the budget as the cause; got {error!r}'
        )
        # 12 = ceil(100/10) + 2 per configured URL.  The load-bearing assertion:
        # without the budget this stub serves 100 requests per URL, so an exact
        # count is what proves the walk is bounded rather than merely finite.
        expected = 12 * len(dummy_config.fused_memory_urls)
        assert len(calls) == expected, (
            f'the walk must stop at its {12}-page budget per URL, not run to '
            f'total; issued {len(calls)} request(s), expected {expected}'
        )

    async def test_the_truncation_marker_is_not_cached(
        self, dummy_client, dummy_config,
    ):
        """A truncation marker is refused by the POSITIVE cache.

        Restated for main's negative cache (task 3857).  The original claim —
        "must return real rows on the very next call" — is no longer the
        contract: a marker IS held by ``_fetch_tasks_negative_cache`` for
        ``_FETCH_TASKS_NEGATIVE_TTL_SECONDS`` (5 s), deliberately, so a broken
        root stops being the expensive path.  What must still hold, and what
        this test now checks, is the two-part property the finding actually
        cared about: the marker is never admitted to the POSITIVE cache
        (``cache_ok=lambda v: isinstance(v, list)``), and it is not retained
        beyond the short negative TTL — past it, a healthy server yields real
        rows.  The 5 s window is unreachable by the only caller that walks:
        ``burndown.collect_snapshot`` runs once per ``_SAMPLE_INTERVAL_SECONDS``
        (600 s).
        """
        from dashboard.data import tasks as tasks_mod
        from dashboard.data.tasks import fetch_tasks

        tasks = [_paged_task_raw(i) for i in range(1, 8)]
        healthy: list[bool] = [False]
        calls: list[dict] = []

        async def _flaky(_client, _url, tool, args, **_kwargs):
            calls.append(dict(args))
            offset = args.get('offset', 0)
            page_size = args.get('page_size')
            if not healthy[0]:
                return {
                    'tasks': [],
                    'pagination': {
                        'total': 99, 'offset': offset, 'page_size': page_size,
                        'returned': 0, 'has_more': True,
                    },
                }
            page = tasks[offset:offset + page_size]
            return {
                'tasks': page,
                'pagination': {
                    'total': len(tasks), 'offset': offset, 'page_size': page_size,
                    'returned': len(page), 'has_more': offset + len(page) < len(tasks),
                },
            }

        with patch(
            'dashboard.data.tasks.mcp_tool_call', new=AsyncMock(side_effect=_flaky),
        ):
            first = await fetch_tasks(
                dummy_client, dummy_config, '/proj/flaky', chunk_size=3,
            )
            assert isinstance(first, dict) and first.get('offline') is True, first

            # The marker was refused by the POSITIVE cache.  Asserted on the
            # walk's own key, so a mis-keyed entry cannot pass this by hiding
            # under a different key.
            key = tasks_mod._TasksRead(
                '/proj/flaky', None, tasks_mod._CompleteRead(3),
            )
            assert tasks_mod._fetch_tasks_cache.get_fresh(key) is None, (
                'the offline marker must never enter the positive cache'
            )

            healthy[0] = True
            # Expire the NEGATIVE entry rather than sleeping out its 5 s TTL.
            # Retention past that window is what the finding forbids; retention
            # inside it is main's deliberate retry suppression.
            tasks_mod._fetch_tasks_negative_cache.clear()
            second = await fetch_tasks(
                dummy_client, dummy_config, '/proj/flaky', chunk_size=3,
            )

        assert isinstance(second, list), (
            f'the marker must not outlive the negative TTL; got {second!r}'
        )
        assert [t['id'] for t in second] == list(range(1, 8)), second

    async def test_the_assembled_list_is_what_gets_cached(
        self, dummy_client, dummy_config,
    ):
        """The cache holds the ASSEMBLED list, under the walk's own key.

        Caching per PAGE would break the documented ~20 s TTL contract and
        re-issue the whole fan-out on every render.  The key is the whole
        `_TasksRead` record, and its `mode` being a `_CompleteRead` rather
        than an `_OnePage` is what keeps this assembled entry from being
        served to a caller that asked for one page of the same size (and
        vice versa).
        """
        from dashboard.data.tasks import fetch_tasks

        tasks = [_paged_task_raw(i) for i in range(1, 8)]
        calls: list[dict] = []
        with patch(
            'dashboard.data.tasks.mcp_tool_call',
            new=AsyncMock(side_effect=_paging_mcp(tasks, calls)),
        ):
            first = await fetch_tasks(
                dummy_client, dummy_config, '/proj/cached', chunk_size=3,
            )
            assert len(calls) == 3
            second = await fetch_tasks(
                dummy_client, dummy_config, '/proj/cached', chunk_size=3,
            )

        assert len(calls) == 3, (
            f'the second call must be served from cache; got {calls!r}'
        )
        assert first == second
        assert [t['id'] for t in second] == list(range(1, 8))

    async def test_a_walk_and_a_same_size_page_do_not_share_a_cache_entry(
        self, dummy_client, dummy_config,
    ):
        """The RETURN CONTRACT MUST discriminate the cache key (esc-4360-8).

        `fetch_task_page(page_size=3, offset=0)` is ONE 3-row page;
        `fetch_tasks(chunk_size=3)` is the complete 7-row tree walked 3 rows
        at a time. Every other key component is identical, so without the
        discriminator whichever ran first inside the 20 s TTL would be served
        to the other — handing a one-page caller the whole tree, or handing
        `burndown.collect_snapshot` a 3-row page to write into an APPEND-ONLY
        history table as the project's true size.

        Since task 5018 the discriminator is the `_TasksRead.mode` record's
        TYPE (`_OnePage` vs `_CompleteRead`) rather than a `paginate` flag, and
        the two contracts are separate FUNCTIONS. This test is the unit-level
        twin of `TestPublicReadContracts`' end-to-end signal and is KEPT: it
        asserts the two results DIFFER, not merely that two keys differ, so it
        fails if the discriminator is dropped and cannot pass by accident the
        way an `in`-only key assertion could.
        """
        from dashboard.data.tasks import fetch_task_page, fetch_tasks

        tasks = [_paged_task_raw(i) for i in range(1, 8)]
        calls: list[dict] = []
        with patch(
            'dashboard.data.tasks.mcp_tool_call',
            new=AsyncMock(side_effect=_paging_mcp(tasks, calls)),
        ):
            walked = await fetch_tasks(
                dummy_client, dummy_config, '/proj/disc', chunk_size=3,
            )
            one_page = await fetch_task_page(
                dummy_client, dummy_config, '/proj/disc', page_size=3, offset=0,
            )

        assert [t['id'] for t in walked] == list(range(1, 8)), walked
        assert [t['id'] for t in one_page] == [1, 2, 3], (
            f'a page read must return exactly one page, got {one_page!r}'
        )

    async def test_statuses_and_timeout_reach_every_page_of_a_walk(
        self, dummy_client, dummy_config,
    ):
        """Both must be on EVERY page request, not just the first.

        `server/tools.py::get_tasks` applies the status filter BEFORE its
        in-memory slice, so `total` is the FILTERED count; a page sent without
        the filter both over-reads and desynchronises the walk's terminator.
        `timeout` is a per-HTTP-request budget, so a page issued without it
        silently falls back to the default.
        """
        from dashboard.data.tasks import fetch_tasks

        tasks = [_paged_task_raw(i) for i in range(1, 8)]
        calls: list[dict] = []
        seen_timeouts: list[object] = []
        paging = _paging_mcp(tasks, calls)

        async def _spy(client, url, tool, args, **kwargs):
            seen_timeouts.append(kwargs.get('timeout'))
            return await paging(client, url, tool, args, **kwargs)

        with patch(
            'dashboard.data.tasks.mcp_tool_call', new=AsyncMock(side_effect=_spy),
        ):
            rows = await fetch_tasks(
                dummy_client, dummy_config, '/proj/thread',
                statuses=['pending'], chunk_size=3, timeout=7.5,
            )

        assert isinstance(rows, list), rows
        assert len(calls) >= 3, calls
        assert all(c.get('statuses') == ['pending'] for c in calls), (
            f'every page must carry the status filter, got {calls!r}'
        )
        assert seen_timeouts == [7.5] * len(calls), (
            f'every page must carry the per-request timeout, got {seen_timeouts!r}'
        )


async def _returned_or_raised(awaitable):
    """Await *awaitable*, returning either its value or the ValueError it raised.

    Lets a test assert on the DIFFERENCE between the two — `pytest.raises`
    discards the returned value, which is the very thing an all-or-nothing
    rule needs to show in its failure message.
    """
    try:
        return await awaitable
    except ValueError as exc:
        return exc


class TestPagePrimitive:
    """`_fetch_page` — one MCP call against ONE url, envelope PRESERVED.

    RED source (step-4): `_fetch_page` does not exist as a named unit; it is
    currently the anonymous `_request`/`_shape_all` closure pair inside
    `fetch_tasks`, which DISCARDS `result['pagination']`. The walk needs
    `returned`/`total`, so returning the pair is the whole point of naming it.
    """

    @staticmethod
    def _read(statuses=None):
        return tasks_mod._TasksRead(
            '/proj/page', statuses, tasks_mod._CompleteRead(None)
        )

    async def test_returns_shaped_rows_and_the_raw_envelope(
        self, dummy_client, monkeypatch
    ):
        """The `pagination` envelope is RETURNED, not dropped on the floor."""
        envelope = {'returned': 2, 'total': 7}

        async def _fake(_client, _url, tool, args, **_kwargs):
            assert tool == 'get_tasks'
            assert args == {
                'project_root': '/proj/page', 'page_size': 2, 'offset': 4,
            }
            return {
                'tasks': [_paged_task_raw(1), _paged_task_raw(2)],
                'pagination': envelope,
            }

        monkeypatch.setattr(tasks_mod, 'mcp_tool_call', _fake)
        page = await tasks_mod._fetch_page(
            dummy_client, 'http://x', self._read(), tasks_mod._OnePage(2, 4), 5.0,
        )
        rows = page.rows

        assert page.pagination == envelope
        assert page.delivered == 2, 'the RAW row count the server sent'
        # _shape_task coerces the id to an int; the raw MCP row carries a str.
        assert [r['id'] for r in rows] == [1, 2]
        # _shape_task-shaped, not raw MCP rows: 'updatedAt' becomes 'updated_at'.
        assert 'updated_at' in rows[0] and 'updatedAt' not in rows[0]

    async def test_pagination_is_none_when_the_server_omits_the_envelope(
        self, dummy_client, monkeypatch
    ):
        """An older fused-memory answers with a bare list and no envelope.

        `None` rather than `{}` so the walk can tell "no envelope" from "an
        envelope with nothing usable in it" — those take different branches.
        """
        async def _fake(_client, _url, _tool, _args, **_kwargs):
            return {'tasks': [_paged_task_raw(1)]}

        monkeypatch.setattr(tasks_mod, 'mcp_tool_call', _fake)
        page = await tasks_mod._fetch_page(
            dummy_client, 'http://x', self._read(), None, 5.0,
        )
        assert page.pagination is None
        assert len(page.rows) == 1

    async def test_a_structured_mcp_error_raises_value_error(
        self, dummy_client, monkeypatch
    ):
        """ValueError is `first_success`'s documented soft-failure signal.

        It must be raised for `'error' in result and 'tasks' not in result`
        specifically, so the fan-out falls through to the next URL rather than
        treating an error payload as an empty tree.
        """
        async def _fake(_client, _url, _tool, _args, **_kwargs):
            return {'error': 'boom'}

        monkeypatch.setattr(tasks_mod, 'mcp_tool_call', _fake)
        with pytest.raises(ValueError, match='boom'):
            await tasks_mod._fetch_page(
                dummy_client, 'http://x', self._read(), None, 5.0,
            )

    async def test_an_error_alongside_tasks_is_not_a_failure(
        self, dummy_client, monkeypatch
    ):
        """`error` WITH `tasks` is a partial-warning payload, not a failure.

        Pinned because the guard is `'error' in result and 'tasks' not in
        result`; loosening it to `'error' in result` would turn a served tree
        into a fan-out failure.
        """
        async def _fake(_client, _url, _tool, _args, **_kwargs):
            return {'error': 'partial', 'tasks': [_paged_task_raw(1)]}

        monkeypatch.setattr(tasks_mod, 'mcp_tool_call', _fake)
        page = await tasks_mod._fetch_page(
            dummy_client, 'http://x', self._read(), None, 5.0,
        )
        assert len(page.rows) == 1

    async def test_the_window_and_statuses_come_from_wire_arguments(
        self, dummy_client, monkeypatch
    ):
        """One encoder builds the request — `_fetch_page` does not roll its own."""
        seen: list[dict] = []

        async def _fake(_client, _url, _tool, args, **_kwargs):
            seen.append(dict(args))
            return {'tasks': [], 'pagination': {'returned': 0, 'total': 0}}

        monkeypatch.setattr(tasks_mod, 'mcp_tool_call', _fake)
        read = self._read(frozenset({'done', 'blocked'}))
        await tasks_mod._fetch_page(
            dummy_client, 'http://x', read, tasks_mod._OnePage(10, 20), 5.0,
        )
        assert seen == [read.wire_arguments(tasks_mod._OnePage(10, 20))]
        assert seen[0]['statuses'] == ['blocked', 'done']


class TestWalkPages:
    """`_walk_pages(page_fn, read, chunk_size)` — assembly over an INJECTED page fn.

    RED source (step-4): `_walk_pages` does not exist as a named unit.

    Taking the page function as a PARAMETER is what makes the five
    completeness failures drivable by a fake, replacing task 4360's
    `monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', ...)`
    string-path seam with a real one.
    """

    @staticmethod
    def _read(statuses=None, root='/proj/walk'):
        return tasks_mod._TasksRead(root, statuses, tasks_mod._CompleteRead(3))

    @staticmethod
    def _pager(pages, calls=None):
        """Build a page fn serving *pages* in order, recording its windows.

        Each entry is ``(rows, pagination)`` or ``(rows, pagination, delivered)``;
        *delivered* defaults to ``len(rows)`` since a well-formed page shapes
        one-for-one.
        """
        served = iter(pages)

        async def _page_fn(window):
            if calls is not None:
                calls.append(window)
            entry = next(served)
            rows, pagination = entry[0], entry[1]
            delivered = entry[2] if len(entry) > 2 else len(rows)
            return tasks_mod._Page(rows, delivered, pagination)

        return _page_fn

    # ---- (e) the PINNED-URL constraint, asserted structurally -------------

    def test_walk_pages_cannot_fan_out(self):
        """`_walk_pages` accepts no url/urls/config parameter.

        Load-bearing, not style: `first_success` tries urls IN ORDER, so a walk
        that could fan out mid-walk would assemble pages from DIFFERENT servers
        and silently invalidate the grown-`total` coherence check — the pages
        would be from different states of two different worlds while every
        counter still looked self-consistent. A structural pin is what stops a
        later edit reintroducing that quietly; a behavioural test cannot, since
        the bug only shows up with two disagreeing servers.
        """
        params = set(inspect.signature(tasks_mod._walk_pages).parameters)
        assert not (params & {'url', 'urls', 'config', 'client'}), (
            'the walk must be bound to ONE already-pinned url by its caller'
        )

    # ---- (c) the two COMPLETE cases: a true zero, not a truncation --------

    async def test_an_empty_tree_returns_empty_rather_than_raising(self):
        """`total <= 0` is a complete read of an empty tree.

        Get this wrong and every empty project becomes a permanent burndown
        hole instead of its legitimate all-zero row.
        """
        page_fn = self._pager([([], {'returned': 0, 'total': 0})])
        assert await tasks_mod._walk_pages(page_fn, self._read(), 3) == []

    async def test_an_exhausted_tree_returns_its_rows(self):
        """`offset >= total` with an empty page is exhaustion, not truncation."""
        rows = [_shape_task(_paged_task_raw(i)) for i in (1, 2)]
        page_fn = self._pager([
            ([r for r in rows if r], {'returned': 2, 'total': 2}),
        ])
        out = await tasks_mod._walk_pages(page_fn, self._read(), 3)
        assert [r['id'] for r in out] == [1, 2]

    # ---- (d) a server that ignores paging ---------------------------------

    async def test_a_server_that_ignores_paging_stops_after_one_page(self):
        """No envelope means this response IS the answer — take it and stop.

        Looping blind would either spin or re-request the same rows forever.
        """
        calls: list = []
        rows = [r for r in (_shape_task(_paged_task_raw(i)) for i in range(1, 8)) if r]
        page_fn = self._pager([(rows, None)], calls)

        out = await tasks_mod._walk_pages(page_fn, self._read(), 3)
        assert len(out) == 7
        assert len(calls) == 1, 'a bare-list server must be asked exactly once'

    # ---- (f) how the walk advances, and what reaches every page -----------

    async def test_offsets_advance_by_rows_delivered_not_by_chunk_size(self):
        """`walk_offset += len(page)`, not `+= chunk_size`.

        A server free to return a SHORT page (fewer rows than asked for) while
        still owing rows must be re-asked from where it actually stopped.
        Advancing by the requested chunk would skip the gap silently.
        """
        calls: list = []
        p1 = [r for r in (_shape_task(_paged_task_raw(i)) for i in (1, 2)) if r]
        p2 = [r for r in (_shape_task(_paged_task_raw(i)) for i in (3, 4)) if r]
        page_fn = self._pager(
            [(p1, {'returned': 2, 'total': 4}), (p2, {'returned': 2, 'total': 4})],
            calls,
        )

        out = await tasks_mod._walk_pages(page_fn, self._read(), 3)
        assert [r['id'] for r in out] == [1, 2, 3, 4]
        # Asked for 3, given 2 -> the next window starts at 2, NOT at 3.
        assert [(w.page_size, w.offset) for w in calls] == [(3, 0), (3, 2)]

    async def test_an_unparseable_row_does_not_look_like_a_truncated_page(self):
        """`returned` is cross-checked against DELIVERED, never `len(rows)`.

        `_shape_task` returns None for a missing or non-integer id, so a page
        of 10 that the server really did send can shape to 9. Comparing the
        server's `returned` against the SHAPED count would read that as a
        truncated page and take the whole project offline — and would advance
        the walk short, so the next page re-reads rows already held.

        This is the hazard `_Page.delivered` exists to make unmissable: before
        the extraction the raw page was in scope and the distinction lived in a
        comment; a named field cannot be silently dropped by a later edit.
        """
        calls: list = []
        # Three rows arrive; the middle one has an unparseable id, so exactly
        # two survive shaping while the server correctly reports returned=3.
        good = [r for r in (_shape_task(_paged_task_raw(i)) for i in (1, 3)) if r]
        page_fn = self._pager(
            [(good, {'returned': 3, 'total': 3}, 3)], calls,
        )

        out = await tasks_mod._walk_pages(page_fn, self._read(), 3)
        assert [r['id'] for r in out] == [1, 3], (
            'the shaped rows are returned, minus the unparseable one'
        )
        assert len(calls) == 1, 'a complete page must not be re-requested'

    async def test_statuses_reach_every_page(self):
        """The tool filters BEFORE its in-memory slice.

        So `total` is the FILTERED count; an unfiltered page would both
        over-read and desynchronise the walk's terminator.
        """
        calls: list = []
        read = self._read(frozenset({'done'}))
        p1 = [r for r in (_shape_task(_paged_task_raw(i)) for i in (1, 2, 3)) if r]
        p2 = [r for r in (_shape_task(_paged_task_raw(4)),) if r]
        page_fn = self._pager(
            [(p1, {'returned': 3, 'total': 4}), (p2, {'returned': 1, 'total': 4})],
            calls,
        )

        await tasks_mod._walk_pages(page_fn, read, 3)
        assert len(calls) == 2
        for window in calls:
            assert read.wire_arguments(window)['statuses'] == ['done']

    # ---- (b) the five completeness failures, all-or-nothing ---------------

    @pytest.mark.parametrize(
        ('label', 'pages'),
        [
            (
                'non-int envelope counters',
                [([{'id': '1'}], {'returned': '1', 'total': 'many'})],
            ),
            (
                'empty page with rows still owed',
                [([], {'returned': 0, 'total': 99})],
            ),
            (
                'returned disagrees with the page actually sent',
                [([{'id': '1'}], {'returned': 10, 'total': 99})],
            ),
            (
                'total grew mid-walk',
                [
                    ([{'id': '1'}, {'id': '2'}, {'id': '3'}],
                     {'returned': 3, 'total': 9}),
                    ([{'id': '4'}, {'id': '5'}, {'id': '6'}],
                     {'returned': 3, 'total': 99}),
                ],
            ),
            (
                'page budget exceeded',
                # total=9, chunk=3 -> budget ceil(9/3)+2 = 5 pages. A server
                # that keeps answering 1 row while claiming 9 amplifies the
                # walk; it is caught here rather than paid for.
                [([{'id': str(i)}], {'returned': 1, 'total': 9}) for i in range(1, 9)],
            ),
        ],
    )
    async def test_each_completeness_failure_raises_and_discards(self, label, pages):
        """ALL-OR-NOTHING: a truncated read must never come back as a list.

        `fetch_tasks`' contract distinguishes only `list` (a complete success)
        from the offline marker, so a truncated list is indistinguishable from
        a complete one at EVERY call site — `collect_snapshot` would write a
        confident undercount into the APPEND-ONLY snapshots table, and a
        plausible dip in an append-only chart is unfalsifiable after the fact
        while a gap is visible. `ValueError` is `first_success`'s soft-failure
        signal, so the fan-out tries the next URL and, on exhaustion, yields
        the marker `collect_snapshot` already skips on.

        The assertion that MATTERS is the second one: raising while handing
        back partial rows would defeat the entire point.
        """
        page_fn = self._pager(pages)

        # Deliberately NOT `pytest.raises`: the property under test is that
        # nothing is RETURNED, so the returned value has to be captured and
        # shown in the failure message. In the last two cases rows really were
        # accumulated before the raise, and handing those back is exactly the
        # silent truncation this rule exists to prevent.
        outcome = await _returned_or_raised(
            tasks_mod._walk_pages(page_fn, self._read(), 3)
        )
        assert isinstance(outcome, ValueError), (
            f'{label}: returned {outcome!r} instead of raising — '
            'partial rows must be DISCARDED, never handed back'
        )
        assert '/proj/walk' in str(outcome), (
            'the message must name the root, or an operator cannot act on it'
        )

    async def test_a_raising_page_fn_propagates(self):
        """`_fetch_page`'s own ValueError rides the same fall-through path.

        The walk adds completeness failures; it does not swallow transport
        ones.
        """
        async def _page_fn(_window):
            raise ValueError('transport said no')

        with pytest.raises(ValueError, match='transport said no'):
            await tasks_mod._walk_pages(_page_fn, self._read(), 3)


class TestPublicReadContracts:
    """The public split is on the RETURN CONTRACT, not on a transport flag.

    RED source (step-8): `fetch_task_page` does not exist and `fetch_tasks`
    still takes `offset`/`paginate`.

    One function returns a PARTIAL answer and says so in its name; the other
    returns the COMPLETE set, always. `chunk_size` selects how that complete
    set crosses the wire and never what it contains. esc-4360-7 was two reads
    with opposite contracts sharing one signature and one cache key — this
    class pins that they can no longer be confused, at the public surface.
    """

    @pytest.fixture(autouse=True)
    def reset_fetch_tasks_cache(self):
        tasks_mod._fetch_tasks_cache_clear()
        yield
        tasks_mod._fetch_tasks_cache_clear()

    # -- (a) the partial read announces itself -----------------------------

    def test_fetch_task_page_exists_and_is_a_coroutine_function(self):
        assert inspect.iscoroutinefunction(tasks_mod.fetch_task_page)

    @pytest.mark.parametrize('omitted', ['page_size', 'offset'])
    async def test_page_size_and_offset_are_both_required(
        self, dummy_client, dummy_config, omitted
    ):
        """A partial read with an implicit window is the defect, not a nicety.

        Defaulting either one would let a caller ask for "a page" without
        saying WHICH page, and the answer would silently be the first — the
        shape that made a page and a whole tree look alike.
        """
        # `dict[str, Any]`, not the inferred `dict[str, int]`: pyright checks
        # `**kwargs` against every keyword parameter, and a narrower value type
        # is reported against `statuses: list[str] | None`.
        kwargs: dict[str, Any] = {'page_size': 10, 'offset': 0}
        kwargs.pop(omitted)

        with pytest.raises(TypeError, match=omitted):
            await tasks_mod.fetch_task_page(
                dummy_client, dummy_config, '/proj/req', **kwargs
            )

    # -- (b) the complete read has no window at all -------------------------

    @pytest.mark.parametrize('rejected', ['offset', 'paginate'])
    async def test_fetch_tasks_rejects_the_retired_arguments(
        self, dummy_client, dummy_config, rejected
    ):
        """THE root-cause fix, stated as a contract.

        `offset` used to enter the cache key unconditionally while only
        reaching the wire alongside `page_size`, so `offset=5` and `offset=7`
        minted two entries for a byte-identical request. Removing it means the
        only read that HAS an offset is the one that ALWAYS sends it — the key
        and the wire cannot disagree again because the disagreement is no
        longer expressible.

        `paginate` is gone for the same reason at the contract level: the walk
        vs slice distinction is now WHICH FUNCTION you call.
        """
        retired: dict[str, Any] = {rejected: 5}
        with pytest.raises(TypeError, match=rejected):
            await tasks_mod.fetch_tasks(
                dummy_client, dummy_config, '/proj/gone', **retired,
            )

    # -- (c) chunk size is transport, never contract ------------------------

    async def test_chunked_and_unchunked_return_the_same_complete_set(
        self, dummy_client, dummy_config
    ):
        """A tree of 7 read whole, and read 3 at a time, must agree exactly."""
        tasks = [_paged_task_raw(i) for i in range(1, 8)]
        calls: list[dict] = []
        with patch(
            'dashboard.data.tasks.mcp_tool_call',
            new=AsyncMock(side_effect=_paging_mcp(tasks, calls)),
        ):
            unchunked = await tasks_mod.fetch_tasks(
                dummy_client, dummy_config, '/proj/same',
            )
            chunked = await tasks_mod.fetch_tasks(
                dummy_client, dummy_config, '/proj/same', chunk_size=3,
            )

        assert [t['id'] for t in unchunked] == list(range(1, 8))
        assert chunked == unchunked, (
            'chunk_size selects TRANSPORT, never the contract — both reads '
            'are the complete set'
        )

    # -- (d) the esc-4360-7 signal, end to end ------------------------------

    async def test_a_page_and_a_same_size_walk_are_different_answers(
        self, dummy_client, dummy_config
    ):
        """THE user-observable signal, at the public surface.

        `fetch_task_page(page_size=10, offset=0)` and
        `fetch_tasks(chunk_size=10)` against ONE project must issue TWO MCP
        reads and return DIFFERENT answers. They can no longer share a cache
        entry, and — the part that matters for the next reader — they can no
        longer be confused for one another at the call site either.
        """
        tasks = [_paged_task_raw(i) for i in range(1, 26)]
        calls: list[dict] = []
        with patch(
            'dashboard.data.tasks.mcp_tool_call',
            new=AsyncMock(side_effect=_paging_mcp(tasks, calls)),
        ):
            page = await tasks_mod.fetch_task_page(
                dummy_client, dummy_config, '/proj/signal', page_size=10, offset=0,
            )
            whole = await tasks_mod.fetch_tasks(
                dummy_client, dummy_config, '/proj/signal', chunk_size=10,
            )

        assert [t['id'] for t in page] == list(range(1, 11)), page
        assert [t['id'] for t in whole] == list(range(1, 26)), whole
        assert page != whole
        assert len(calls) == 4, (
            f'one page read plus a three-page walk, got {calls!r} — a shared '
            'cache entry would show up here as a missing read'
        )

    # -- (e) statuses order-insensitivity, end to end -----------------------

    async def test_reordered_statuses_hit_one_cache_entry(
        self, dummy_client, dummy_config
    ):
        """The second measured defect, closed at the public surface.

        `['a','b']` and `['b','a']` are the same SQL `IN` list, so they must be
        the same read. They used to mint `s=a\\x1fb` and `s=b\\x1fa` — two
        entries, two round trips, one answer.
        """
        mock_mcp = AsyncMock(return_value=_CANNED_GET_TASKS_RESULT)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            first = await tasks_mod.fetch_tasks(
                dummy_client, dummy_config, '/proj/order',
                statuses=['pending', 'in-progress'],
            )
            second = await tasks_mod.fetch_tasks(
                dummy_client, dummy_config, '/proj/order',
                statuses=['in-progress', 'pending'],
            )

        assert mock_mcp.call_count == 1, (
            f'an order-only difference must reuse the entry, got '
            f'{mock_mcp.call_count} call(s)'
        )
        assert first == second

    async def test_genuinely_different_statuses_still_key_separately(
        self, dummy_client, dummy_config
    ):
        """The order fix must not over-collapse into a status-blind key."""
        async def _by_statuses(client, url, tool, args, **_kw):
            return {'tasks': [{
                'id': '1', 'title': f'rows for {args.get("statuses")}',
                'status': 'done', 'dependencies': [], 'metadata': {},
            }]}

        mock_mcp = AsyncMock(side_effect=_by_statuses)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            done = await tasks_mod.fetch_tasks(
                dummy_client, dummy_config, '/proj/distinct', statuses=['done'],
            )
            pending = await tasks_mod.fetch_tasks(
                dummy_client, dummy_config, '/proj/distinct', statuses=['pending'],
            )
            whole = await tasks_mod.fetch_tasks(
                dummy_client, dummy_config, '/proj/distinct',
            )
            nothing = await tasks_mod.fetch_tasks(
                dummy_client, dummy_config, '/proj/distinct', statuses=[],
            )

        assert mock_mcp.call_count == 4, (
            f'four distinct narrowings, got {mock_mcp.call_count} call(s)'
        )
        assert done[0]['title'] != pending[0]['title']
        # `None` (whole tree) and `[]` (nothing at all) are OPPOSITE requests
        # and must never collapse onto one entry — `frozenset()` is falsy, so
        # a truthiness guard anywhere on this path would do exactly that.
        assert whole[0]['title'] != nothing[0]['title']


class TestCachedFanoutCore:
    """`_cached_fanout(client, config, read, strategy, timeout)` — the ONE place
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
            dummy_client, two_url_config, self._read('/proj/core-order'),
            _strategy, 5.0,
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
            dummy_client, two_url_config, self._read('/proj/core-fallthrough'),
            _strategy, 5.0,
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
            dummy_client, dummy_config, read, _strategy, 5.0,
        )
        second = await tasks_mod._cached_fanout(
            dummy_client, dummy_config, read, _strategy, 5.0,
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
            dummy_client, dummy_config, read, _strategy, 5.0,
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
            dummy_client, dummy_config, read, _strategy, 5.0,
        )
        first.clear()

        second = await tasks_mod._cached_fanout(
            dummy_client, dummy_config, read, _strategy, 5.0,
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
            dummy_client, dummy_config, read, _fail, 5.0,
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
        await tasks_mod._cached_fanout(dummy_client, dummy_config, read, _ok, 5.0)
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
            dummy_client, dummy_config, read, _strategy, 5.0,
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

        await tasks_mod._cached_fanout(dummy_client, dummy_config, read, _ok, 5.0)

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
            dummy_client, dummy_config, read, _strategy, 5.0,
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

        async def _fake(client, config, read, strategy, timeout):
            seen.append(read)
            assert client is dummy_client
            assert config is dummy_config
            assert callable(strategy)
            assert timeout == 7.5
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


class TestFetchStatusesCache:
    """Per-project_root TTL cache inside fetch_statuses (task 3857 amendment).

    ``fetch_statuses`` was the one per-project MCP call in the narrowed design
    with no cache, and ``active_tasks._shape_one_project`` issues it
    unconditionally.  Both /api/v2/dashboard/tasks and
    /api/v2/dashboard/scheduler reach that function on every 3 s poll, so
    uncached it cost two full-population ``get_statuses`` reads per root per
    poll — trading wire bytes for backend queries.  These tests pin the
    collapse, and pin that it never comes at the cost of a stale failure or a
    poisoned entry.
    """

    @pytest.fixture(autouse=True)
    def reset_fetch_statuses_cache(self):
        import dashboard.data.tasks as tasks_mod
        tasks_mod._fetch_statuses_cache_clear()
        yield
        tasks_mod._fetch_statuses_cache_clear()

    async def test_within_ttl_issues_single_mcp_call(self, dummy_client, dummy_config):
        """Two reads of one root within the TTL collapse to ONE get_statuses call.

        This is the whole point of the amendment: the tasks and scheduler
        endpoints poll the same root within the same 3 s window.
        """
        import dashboard.data.tasks as tasks_mod
        from dashboard.data.tasks import fetch_statuses

        mock_mcp = AsyncMock(return_value={'statuses': {'1': 'done', '2': 'pending'}})
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            first = await fetch_statuses(dummy_client, dummy_config, '/proj/A')
            second = await fetch_statuses(dummy_client, dummy_config, '/proj/A')

        assert mock_mcp.call_count == 1, (
            f'expected 1 MCP call within TTL, got {mock_mcp.call_count}'
        )
        assert first == {1: 'done', 2: 'pending'}
        assert first == second
        call = mock_mcp.call_args_list[0]
        assert call.args[2] == 'get_statuses'
        assert call.args[3] == {'project_root': '/proj/A'}
        assert call.kwargs.get('timeout') == tasks_mod.DEFAULT_PER_CALL_TIMEOUT

    async def test_distinct_roots_cached_separately(self, dummy_client, dummy_config):
        """One root's map must never be served for another (no global keying)."""
        from dashboard.data.tasks import fetch_statuses

        async def _per_root(client, url, tool, args, **_kw):
            if args.get('project_root') == '/proj/A':
                return {'statuses': {'1': 'done'}}
            return {'statuses': {'2': 'pending'}}

        mock_mcp = AsyncMock(side_effect=_per_root)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            a = await fetch_statuses(dummy_client, dummy_config, '/proj/A')
            b = await fetch_statuses(dummy_client, dummy_config, '/proj/B')

        assert mock_mcp.call_count == 2
        assert a == {1: 'done'}
        assert b == {2: 'pending'}

    async def test_offline_marker_is_not_cached(self, dummy_client, dummy_config):
        """A failed read is re-probed on the next poll — recovery is not deferred.

        Unlike ``fetch_tasks`` there is deliberately no negative cache here: a
        failed ``get_statuses`` costs one bounded per-call timeout, not a tree
        walk, and its marker is what puts a root in
        TASKS_COUNT_UNKNOWN_PROJECTS — a visible degradation that must clear
        on the first poll after recovery.
        """
        from dashboard.data.tasks import fetch_statuses

        mock_mcp = AsyncMock(side_effect=[
            httpx.ConnectError('refused'),
            {'statuses': {'5': 'done'}},
        ])
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            offline = await fetch_statuses(dummy_client, dummy_config, '/proj/C')
            recovered = await fetch_statuses(dummy_client, dummy_config, '/proj/C')

        assert mock_mcp.call_count == 2, (
            f'an offline marker must not be cached, got {mock_mcp.call_count} calls'
        )
        assert offline.get('offline') is True
        assert recovered == {5: 'done'}

    async def test_ttl_expiry_refetches(self, monkeypatch, dummy_client, dummy_config):
        """Past the TTL the map is re-read — the cache bounds cost, it does not pin state."""
        import dashboard.data.tasks as tasks_mod
        from dashboard.data.tasks import fetch_statuses

        monkeypatch.setattr(tasks_mod, '_FETCH_STATUSES_TTL_SECONDS', 0.0)
        mock_mcp = AsyncMock(return_value={'statuses': {'1': 'pending'}})
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            await fetch_statuses(dummy_client, dummy_config, '/proj/A')
            await fetch_statuses(dummy_client, dummy_config, '/proj/A')

        assert mock_mcp.call_count == 2, (
            f'expected a refetch once the TTL lapsed, got {mock_mcp.call_count}'
        )

    async def test_returned_map_is_a_copy(self, dummy_client, dummy_config):
        """A caller mutating the returned map must not poison the cached entry.

        ``_shape_one_project`` and ``collect_done_counts`` both iterate this
        map; a future caller that filters it in place would otherwise corrupt
        every subsequent reader for the whole TTL window.
        """
        from dashboard.data.tasks import fetch_statuses

        mock_mcp = AsyncMock(return_value={'statuses': {'1': 'done', '2': 'pending'}})
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            first = await fetch_statuses(dummy_client, dummy_config, '/proj/A')
            assert isinstance(first, dict)
            first.clear()
            second = await fetch_statuses(dummy_client, dummy_config, '/proj/A')

        assert mock_mcp.call_count == 1, 'still one call — the entry is warm'
        assert second == {1: 'done', 2: 'pending'}, (
            f'the cached map was mutated through the returned reference: {second}'
        )

    async def test_ttl_is_shorter_than_the_fetch_tasks_ttl(self):
        """The status map must be the FRESHER half of any row+map pair.

        ``collect_tasks_with_counts`` puts DONE_COUNTS (this map) beside
        ACTIVE_TASKS rows (``fetch_tasks``); both are cached, so the skew is
        real either way.  Keeping this TTL strictly shorter makes the skew
        one-directional — the count can be newer than the rows, never staler —
        which is the property both docstrings claim.
        """
        import dashboard.data.tasks as tasks_mod

        assert (
            tasks_mod._FETCH_STATUSES_TTL_SECONDS < tasks_mod._FETCH_TASKS_TTL_SECONDS
        ), (
            'fetch_statuses TTL must stay strictly under the fetch_tasks TTL: '
            f'{tasks_mod._FETCH_STATUSES_TTL_SECONDS} vs '
            f'{tasks_mod._FETCH_TASKS_TTL_SECONDS}'
        )

    async def test_clear_hook_does_not_reach_into_the_fetch_tasks_caches(
        self, dummy_client, dummy_config
    ):
        """The two clear hooks stay separate — each names exactly what it clears."""
        import dashboard.data.tasks as tasks_mod
        from dashboard.data.tasks import fetch_statuses, fetch_tasks

        tasks_mod._fetch_tasks_cache_clear()
        mock_mcp = AsyncMock(side_effect=lambda c, u, tool, args, **kw: (
            {'statuses': {'1': 'done'}} if tool == 'get_statuses'
            else {'tasks': []}
        ))
        try:
            with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
                await fetch_tasks(dummy_client, dummy_config, '/proj/A')
                await fetch_statuses(dummy_client, dummy_config, '/proj/A')

                tasks_mod._fetch_statuses_cache_clear()
                # fetch_tasks stays warm; fetch_statuses re-reads.
                await fetch_tasks(dummy_client, dummy_config, '/proj/A')
                await fetch_statuses(dummy_client, dummy_config, '/proj/A')
        finally:
            tasks_mod._fetch_tasks_cache_clear()

        tools = [c.args[2] for c in mock_mcp.call_args_list]
        assert tools.count('get_tasks') == 1, (
            f'_fetch_statuses_cache_clear must not evict fetch_tasks, got {tools}'
        )
        assert tools.count('get_statuses') == 2, (
            f'_fetch_statuses_cache_clear must evict fetch_statuses, got {tools}'
        )
