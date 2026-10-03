"""Unit tests for dashboard.data.tasks._shape_task and fetch_external_statuses.

Focus: the field-mapping contract at the MCP→dashboard boundary and the
fetch_external_statuses short-circuit + fail-safe semantics.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from _canned_mcp import CANNED_GET_TASKS_RESULT

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


def test_shape_task_drops_details_but_keeps_description():
    """Only the Task Detail pane renders ``details``, and it fetches its own.

    ``description`` stays: ``redux_api.shape_escalations`` embeds this whole
    dict as each escalation row's ``task``, and the Escalations drawer renders
    ``task.description``.
    """
    raw = {
        'id': '12',
        'title': 'x',
        'status': 'pending',
        'description': 'the escalation drawer shows this',
        'details': 'implementation notes ' * 500,
    }
    shaped = _shape_task(raw)
    assert shaped is not None
    assert 'details' not in shaped
    assert shaped['description'] == 'the escalation drawer shows this'


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
    def test_each_record_is_frozen_and_usable_as_a_dict_key(self, name, build):
        """A cache key that can be mutated after insertion is not a key.

        These are the two properties a cache key genuinely needs, and both are
        asserted against live behaviour rather than against the decorator's
        arguments: a mutated key is unfindable in the dict it was filed under,
        and an equal-but-distinct rebuild must find the entry the original
        wrote.
        """
        record = build(tasks_mod)

        field = dataclasses.fields(record)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(record, field, 'mutated')

        # A real dict round trip, not `hash()` alone — that would pass for a
        # type whose __eq__ and __hash__ disagree, which is precisely the way
        # a key silently stops finding its own entry.
        assert {record: 'v'}[build(tasks_mod)] == 'v', (
            f'{name} does not round-trip as a dict key'
        )

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

        mock_mcp = AsyncMock(return_value=CANNED_GET_TASKS_RESULT)
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

        mock_mcp = AsyncMock(return_value=CANNED_GET_TASKS_RESULT)
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

        mock_mcp = AsyncMock(return_value=CANNED_GET_TASKS_RESULT)
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

        mock_mcp = AsyncMock(return_value=CANNED_GET_TASKS_RESULT)
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
        mock_mcp = AsyncMock(return_value=CANNED_GET_TASKS_RESULT)
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            await read(dummy_client, dummy_config, '/proj/A', **kwargs)

        call = mock_mcp.call_args_list[0]
        assert call.kwargs.get('timeout') == tasks_mod.DEFAULT_PER_CALL_TIMEOUT

    async def test_explicit_timeout_overrides_the_default(
        self, dummy_client, dummy_config
    ):
        """A caller may tighten the per-request budget further."""
        from dashboard.data.tasks import fetch_tasks

        mock_mcp = AsyncMock(return_value=CANNED_GET_TASKS_RESULT)
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

        mock_mcp = AsyncMock(return_value=CANNED_GET_TASKS_RESULT)
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
