"""Tests for shared.status_paging (task 3396).

The walk is driven through its public seam with an in-test page reader that
pages an ordered population the way
``fused_memory/server/tools.py::_pagination_meta`` does: ``page_size`` echoes
the REQUESTED size, ``returned`` is the count SERVED, and ``has_more`` is
``offset + returned < total``.  The request shape and the dispatched-reply
reader are driven with plain argument dicts and JSON-RPC reply bodies.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable

import pytest

from shared.mcp_envelope import EnvelopeParseError
from shared.status_paging import (
    STATUSES_SAFE_PAGE_SIZE,
    StatusPage,
    StatusPagingError,
    read_status_page,
    status_page_arguments,
    walk_status_pages,
)
from shared.testing_mcp_stub import tool_reply_body

_LABEL = '/proj/paged'
_ABSENT = object()
"""Marks an envelope key a test deletes, as opposed to one it sets to ``None``."""

_PageReader = Callable[[int], Awaitable[StatusPage[str]]]


def _population(size: int) -> dict[str, str]:
    return {str(tid): 'done' if tid % 2 else 'pending' for tid in range(1, size + 1)}


class _PagedServer:
    """Serves *population* in order, clamping each request to an optional server *cap*."""

    def __init__(self, population: dict[str, str], *, requested: int, cap: int | None = None):
        self._items = list(population.items())
        self._requested = requested
        self._served = requested if cap is None else min(requested, cap)
        self.offsets: list[int] = []

    def envelope(self, offset: int, returned: int) -> dict[str, object]:
        total = len(self._items)
        return {
            'total': total,
            'offset': offset,
            'page_size': self._requested,
            'returned': returned,
            'has_more': offset + returned < total,
        }

    def window(self, offset: int) -> dict[str, str]:
        return dict(self._items[offset:offset + self._served])

    async def __call__(self, offset: int) -> StatusPage[str]:
        self.offsets.append(offset)
        window = self.window(offset)
        return StatusPage(window, len(window), self.envelope(offset, len(window)))


_Rewrite = Callable[[dict[str, str], dict[str, object]], StatusPage[str]]
"""Turns the first page's honest window and envelope into the page actually sent."""


def _first_page_rewritten(server: _PagedServer, rewrite: _Rewrite) -> _PageReader:
    async def _read(offset: int) -> StatusPage[str]:
        if offset > 0:
            return await server(offset)
        window = server.window(0)
        return rewrite(window, server.envelope(0, len(window)))

    return _read


async def _refusal(page_fn: _PageReader) -> StatusPagingError:
    """Walk *page_fn*, expecting a refusal that names the walk's label."""
    with pytest.raises(StatusPagingError) as refused:
        await walk_status_pages(page_fn, _LABEL)
    assert _LABEL in str(refused.value), (
        f'a refusal must name the read it refused, got {refused.value!r}'
    )
    return refused.value


class TestAWalkThatCompletes:
    async def test_a_page_without_pagination_is_the_whole_answer(self):
        """No envelope means COMPLETE: one read, and looping blind would re-read forever."""
        population = _population(3)
        offsets: list[int] = []

        async def _whole(offset: int) -> StatusPage[str]:
            offsets.append(offset)
            return StatusPage(dict(population), len(population), None)

        assert await walk_status_pages(_whole, _LABEL) == population
        assert offsets == [0]

    async def test_a_clamping_server_is_tiled_by_what_each_page_served(self):
        """Requested 2000, served 3: the walk steps by ``returned``, never by ``page_size``."""
        population = _population(10)
        server = _PagedServer(population, requested=2000, cap=3)

        assert await walk_status_pages(server, _LABEL) == population
        assert server.offsets == [0, 3, 6, 9]

    async def test_the_returned_map_is_freshly_built(self):
        population = _population(4)
        server = _PagedServer(population, requested=2)

        first = await walk_status_pages(server, _LABEL)
        first.clear()

        assert await walk_status_pages(server, _LABEL) == population


class TestAnUnverifiableEnvelopeIsRefused:
    @pytest.mark.parametrize('pagination', [
        pytest.param(['has_more', True], id='a-list'),
        pytest.param('has_more=true', id='a-string'),
    ])
    async def test_a_pagination_that_is_not_a_mapping_is_refused(self, pagination):
        """Present but unreadable is unverifiable, and is not read as absent (complete)."""
        server = _PagedServer(_population(10), requested=4)

        def _garbled(window: dict[str, str], envelope: dict[str, object]) -> StatusPage[str]:
            return StatusPage(window, len(window), pagination)

        error = await _refusal(_first_page_rewritten(server, _garbled))
        assert 'unverifiable' in str(error)

    @pytest.mark.parametrize('key, value', [
        pytest.param('has_more', _ABSENT, id='has_more-absent'),
        pytest.param('has_more', 'true', id='has_more-a-string'),
        pytest.param('has_more', 1, id='has_more-an-int'),
        pytest.param('returned', _ABSENT, id='returned-absent'),
        pytest.param('returned', None, id='returned-null'),
        pytest.param('returned', '4', id='returned-a-string'),
    ])
    async def test_an_envelope_without_a_usable_has_more_or_returned_is_refused(
        self, key, value,
    ):
        server = _PagedServer(_population(10), requested=4)

        def _unusable(window: dict[str, str], envelope: dict[str, object]) -> StatusPage[str]:
            if value is _ABSENT:
                del envelope[key]
            else:
                envelope[key] = value
            return StatusPage(window, len(window), envelope)

        await _refusal(_first_page_rewritten(server, _unusable))

    @pytest.mark.parametrize('total', [
        pytest.param(_ABSENT, id='absent'),
        pytest.param(None, id='null'),
        pytest.param('10', id='a-string'),
    ])
    async def test_more_owed_with_no_usable_total_is_refused(self, total):
        """The page budget is derived from ``total``, so a walk that must continue needs it."""
        server = _PagedServer(_population(10), requested=4)

        def _untotalled(window: dict[str, str], envelope: dict[str, object]) -> StatusPage[str]:
            if total is _ABSENT:
                del envelope['total']
            else:
                envelope['total'] = total
            return StatusPage(window, len(window), envelope)

        error = await _refusal(_first_page_rewritten(server, _untotalled))
        assert 'total' in str(error)


class TestAnInconsistentPageIsRefused:
    @pytest.mark.parametrize('shipped, claimed', [
        pytest.param(2, 4, id='a-page-clipped-in-flight'),
        pytest.param(4, 2, id='a-page-that-under-reports'),
    ])
    async def test_returned_must_match_what_the_page_delivered(self, shipped, claimed):
        """The walk advances on ``returned``, so a miscounted page would skip or re-read entries."""
        server = _PagedServer(_population(10), requested=4)

        def _miscounted(window: dict[str, str], envelope: dict[str, object]) -> StatusPage[str]:
            shipped_window = dict(list(window.items())[:shipped])
            return StatusPage(shipped_window, shipped, {**envelope, 'returned': claimed})

        error = await _refusal(_first_page_rewritten(server, _miscounted))
        assert 'inconsistent' in str(error)

    async def test_more_owed_but_nothing_served_is_refused(self):
        """``has_more`` with ``returned=0`` cannot advance the offset, so it cannot be paged past."""
        server = _PagedServer(_population(10), requested=4)

        def _empty(window: dict[str, str], envelope: dict[str, object]) -> StatusPage[str]:
            return StatusPage({}, 0, {**server.envelope(0, 0), 'has_more': True})

        error = await _refusal(_first_page_rewritten(server, _empty))
        assert 'cannot advance' in str(error)


class TestAWalkIsBounded:
    async def test_a_server_that_never_lowers_has_more_is_cut_off_at_its_budget(self):
        """The budget is ``ceil(total / first returned) + 2`` pages, from the SERVED rate."""
        population = _population(10)
        server = _PagedServer(population, requested=2000, cap=4)
        calls: list[int] = []

        async def _endless(offset: int) -> StatusPage[str]:
            # Honest page sizes, but every page claims more remains and the
            # window wraps round instead of running dry, so only the budget
            # can stop the walk.
            calls.append(offset)
            wrapped = offset % len(population)
            window = server.window(wrapped)
            envelope = {**server.envelope(wrapped, len(window)), 'has_more': True}
            return StatusPage(window, len(window), envelope)

        error = await _refusal(_endless)

        assert 'budget' in str(error)
        assert len(calls) == math.ceil(len(population) / 4) + 2


class TestAFailureIsNeverAPartialMap:
    async def test_a_page_reader_failure_on_a_later_page_propagates(self):
        server = _PagedServer(_population(10), requested=4)

        async def _breaks_at_the_second_page(offset: int) -> StatusPage[str]:
            if offset > 0:
                raise OSError('connection reset mid-walk')
            return await server(offset)

        with pytest.raises(OSError, match='connection reset mid-walk'):
            await walk_status_pages(_breaks_at_the_second_page, _LABEL)

    def test_a_paging_refusal_is_a_value_error(self):
        """The dashboard's ``first_success`` reads ``ValueError`` as its soft-failure signal."""
        assert issubclass(StatusPagingError, ValueError)


class TestTheRequestForAPage:
    _BASE = {'project_root': '/p', 'ids': ['1', '2']}

    def test_the_first_page_is_the_callers_own_request(self):
        base = dict(self._BASE)

        request = status_page_arguments(base, 0)
        request['page_size'] = 1

        assert status_page_arguments(base, 0) == self._BASE
        assert base == self._BASE

    def test_a_continuation_names_its_page_size(self):
        """``get_statuses`` ignores ``offset`` when ``page_size`` is omitted."""
        base = dict(self._BASE)

        assert status_page_arguments(base, 7) == {
            **self._BASE, 'page_size': STATUSES_SAFE_PAGE_SIZE, 'offset': 7,
        }
        assert base == self._BASE


class TestReadingADispatchedPage:
    def test_a_reply_without_pagination_reads_as_a_complete_page(self):
        reply = tool_reply_body({'statuses': {'1': 'done', '2': 'pending'}})

        assert read_status_page(reply) == StatusPage({'1': 'done', '2': 'pending'}, 2, None)

    def test_the_pagination_envelope_is_passed_through_verbatim(self):
        envelope = {'total': 5, 'offset': 0, 'page_size': 2, 'returned': 2, 'has_more': True}
        reply = tool_reply_body({'statuses': {'1': 'done', '2': 'done'}, 'pagination': envelope})

        assert read_status_page(reply).pagination == envelope

    def test_a_pagination_that_is_not_a_mapping_is_left_for_the_walk(self):
        reply = tool_reply_body({'statuses': {'1': 'done'}, 'pagination': ['has_more', True]})

        assert read_status_page(reply).pagination == ['has_more', True]

    @pytest.mark.parametrize('reply', [
        pytest.param(tool_reply_body({'tasks': []}), id='no-statuses-key'),
        pytest.param(tool_reply_body({'statuses': ['1', '2']}), id='statuses-a-list'),
        pytest.param({
            'jsonrpc': '2.0', 'id': 1,
            'result': {
                'content': [{'type': 'text', 'text': 'Error executing tool get_statuses: boom'}],
                'isError': True,
            },
        }, id='a-tool-error'),
        pytest.param({
            'jsonrpc': '2.0', 'id': 1,
            'error': {'code': -32602, 'message': 'Invalid params'},
        }, id='a-protocol-error'),
    ])
    def test_an_unreadable_reply_raises_the_envelope_error(self, reply):
        with pytest.raises(EnvelopeParseError):
            read_status_page(reply)
