"""Tests for the ``_TasksRead`` record and the wire arguments it derives.

Covered directly and as observed through ``fetch_tasks``/``fetch_task_page``:
statuses canonicalisation, the page window and the per-request timeout.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import AsyncMock, patch

import pytest
from _canned_mcp import canned_get_tasks_result

import dashboard.data.tasks as tasks_mod


class TestTasksReadRecord:
    """The structured read record: `_OnePage` | `_CompleteRead` inside `_TasksRead`.

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
# (task 3857)
# ---------------------------------------------------------------------------


class TestFetchTasksNarrowing:
    """``fetch_tasks``'s narrowing arguments as a wire contract.

    The whole point of the narrowing work is that the *server* does the
    filtering, so what matters is the arguments dict that actually crosses
    the MCP boundary — not what the dashboard discards afterwards. These
    tests therefore assert on ``mcp_tool_call``'s recorded call args.

    The unnarrowed shape is pinned byte-identical because the full-tree
    caller (``app._fanout_probe_completion``, the /healthz probe) must be
    unaffected by narrowing.
    """

    @staticmethod
    def _args_of(mock_mcp, index=0):
        """Return the arguments dict of the *index*-th recorded MCP call."""
        return mock_mcp.call_args_list[index].args[3]

    async def test_unnarrowed_call_sends_project_root_only(
        self, dummy_client, dummy_config
    ):
        """(a) No narrowing → the arguments dict is EXACTLY {'project_root': ...}.

        Backward-compatibility guard for the full-tree caller: no
        ``statuses``, no ``page_size``, no ``offset`` key may appear.
        """
        from dashboard.data.tasks import fetch_tasks

        mock_mcp = AsyncMock(return_value=canned_get_tasks_result())
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            await fetch_tasks(dummy_client, dummy_config, '/proj/A')

        assert mock_mcp.call_count == 1
        assert self._args_of(mock_mcp) == {'project_root': '/proj/A'}, (
            'the unnarrowed arguments dict must stay byte-identical for the '
            'full-tree caller'
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

        mock_mcp = AsyncMock(return_value=canned_get_tasks_result())
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

        mock_mcp = AsyncMock(return_value=canned_get_tasks_result())
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
    # sends it.
    # `test_tasks_pagination.py::TestPublicReadContracts::test_fetch_tasks_rejects_the_retired_arguments`
    # is the strictly stronger statement that replaces it: unrepresentable
    # beats representable-but-dropped.

    async def test_statuses_and_page_size_compose(self, dummy_client, dummy_config):
        """The terminal-window call shape: narrowed statuses PLUS a bounded window."""
        from dashboard.data.tasks import fetch_task_page

        mock_mcp = AsyncMock(return_value=canned_get_tasks_result())
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
        mock_mcp = AsyncMock(return_value=canned_get_tasks_result())
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            await read(dummy_client, dummy_config, '/proj/A', **kwargs)

        call = mock_mcp.call_args_list[0]
        assert call.kwargs.get('timeout') == tasks_mod.DEFAULT_PER_CALL_TIMEOUT

    async def test_explicit_timeout_overrides_the_default(
        self, dummy_client, dummy_config
    ):
        """A caller may tighten the per-request budget further."""
        from dashboard.data.tasks import fetch_tasks

        mock_mcp = AsyncMock(return_value=canned_get_tasks_result())
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
