"""Tests for how a tasks read returns ONE page versus the COMPLETE set.

The public split is on the return contract: ``fetch_task_page`` answers one
explicit window of the ascending-id task list, and ``fetch_tasks`` answers the
complete set, always, in one unpaginated request. These tests drive both
through the fan-out with only ``mcp_tool_call`` replaced.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from _canned_mcp import canned_get_tasks_result

import dashboard.data.tasks as tasks_mod


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


def _paging_mcp(tasks: list[dict], calls: list[dict]):
    """An ``mcp_tool_call`` stub that pages exactly as MCP ``get_tasks`` does.

    Mirrors fused-memory ``server/tools.py`` (the ``_pagination_meta``
    envelope): when ``page_size`` is absent the full list comes back with NO
    ``pagination`` key at all — the backward-compatible shape every existing
    caller relies on.  When it is present, the response carries the page plus
    ``{total, offset, page_size, returned, has_more}``.
    """

    async def _call(_client, _url, tool, args, **_kwargs):
        assert tool == 'get_tasks', tool
        calls.append(dict(args))
        page_size = args.get('page_size')
        if page_size is None:
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


class TestPublicReadContracts:
    """The public split is on the RETURN CONTRACT, not on a transport flag.

    One function returns a PARTIAL answer and says so in its name; the other
    returns the COMPLETE set, always. esc-4360-7 was two reads with opposite
    contracts sharing one signature and one cache key — this class pins that
    they can no longer be confused, at the public surface.
    """

    # -- (a) the partial read announces itself -----------------------------

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

    @pytest.mark.parametrize('rejected', ['offset', 'paginate', 'chunk_size'])
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

        `chunk_size` selected a walk no caller used, and removing it leaves the
        complete read with exactly one transport.
        """
        retired: dict[str, Any] = {rejected: 5}
        with pytest.raises(TypeError, match=rejected):
            await tasks_mod.fetch_tasks(
                dummy_client, dummy_config, '/proj/gone', **retired,
            )

    # -- (c) the esc-4360-7 signal, end to end ------------------------------

    async def test_a_page_and_the_complete_set_are_different_answers(
        self, dummy_client, dummy_config
    ):
        """THE user-observable signal, at the public surface.

        `fetch_task_page(page_size=10, offset=0)` and `fetch_tasks()` against
        ONE project must issue TWO MCP reads and return DIFFERENT answers. They
        can no longer share a cache entry, and — the part that matters for the
        next reader — they can no longer be confused for one another at the
        call site either.
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
                dummy_client, dummy_config, '/proj/signal',
            )

        assert [t['id'] for t in page] == list(range(1, 11)), page
        assert [t['id'] for t in whole] == list(range(1, 26)), whole
        assert page != whole
        assert len(calls) == 2, (
            f'one page read plus one whole-tree read, got {calls!r} — a shared '
            'cache entry would show up here as a missing read'
        )
        assert calls[1] == {'project_root': '/proj/signal'}, (
            f'the complete read sends no window; got {calls[1]!r}'
        )

    # -- (d) statuses order-insensitivity, end to end -----------------------

    async def test_reordered_statuses_send_one_wire_request(
        self, dummy_client, dummy_config
    ):
        """The second measured defect, closed at the public surface.

        `['a','b']` and `['b','a']` are the same SQL `IN` list, so they must be
        the same read, and the same bytes on the wire.
        """
        mock_mcp = AsyncMock(return_value=canned_get_tasks_result())
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            first = await tasks_mod.fetch_tasks(
                dummy_client, dummy_config, '/proj/order',
                statuses=['pending', 'in-progress'],
            )
            second = await tasks_mod.fetch_tasks(
                dummy_client, dummy_config, '/proj/order',
                statuses=['in-progress', 'pending'],
            )

        sent = [call.args[3] for call in mock_mcp.call_args_list]
        assert len(sent) == 2 and sent[0] == sent[1], (
            f'an order-only difference must send identical arguments, got {sent}'
        )
        assert sent[0]['statuses'] == ['in-progress', 'pending']
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
