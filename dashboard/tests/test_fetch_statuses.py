"""Tests for dashboard.data.tasks.fetch_statuses.

It walks ``get_statuses``' pagination envelope to one complete status map and
holds no cache of its own (task 5587).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from _canned_mcp import CannedMCP

_ABSENT = object()
"""Marks a pagination key the test deletes, as opposed to one it sets to ``None``."""


class TestFetchStatusesPaging:
    """``fetch_statuses`` walks the status map, and owns no cache (task 5587).

    Two facts, tested together because they are two halves of one change.

    PAGING.  ``get_statuses`` answers a ``page_size``/``offset`` request with
    the five-key envelope ``fused_memory/server/tools.py::_pagination_meta``
    builds: ``page_size`` echoes the REQUESTED size, ``returned`` is the count
    SERVED, and ``has_more`` is ``offset + returned < total``.  The walk
    advances by ``returned`` until ``has_more`` is False, and reads the
    ABSENCE of the envelope as a complete answer.  ``get_statuses``' own
    docstring says to advance by ``page_size``; its implementation does not
    support that, and these tests follow the implementation.  An unpaginated
    read of a large tree is rejected wholesale by the MCP transport, which is
    the incident that pagination exists to close.

    NO CACHE.  The 5 s TTL cache is gone.  After this task the only consumer
    of this read is ``task_snapshot.acquire_snapshot``, which owns a 15 s unit
    TTL of its own — a second cache layered under it is precisely the
    duplicated-staleness the one-datum-one-path PRD (decision 20) removes, and
    it would let the unit serve a map older than the ``as_of`` it stamps.
    """

    @staticmethod
    def _statuses_calls(canned):
        return [call['args'] for call in canned.calls_to('get_statuses')]

    async def test_a_paged_tree_returns_one_merged_map(
        self, dummy_client, dummy_config
    ):
        """(a) 4500 statuses served in pages arrive as ONE complete map."""
        from dashboard.data.tasks import fetch_statuses

        population = {tid: 'done' if tid % 2 else 'pending'
                      for tid in range(1, 4501)}
        canned = CannedMCP(status_map=population, status_page_size=2000)

        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            result = await fetch_statuses(dummy_client, dummy_config, '/proj/big')

        assert result == population, (
            f'expected all {len(population)} statuses merged into one map, '
            f'got {len(result)}'
        )

    async def test_every_page_is_bounded_and_the_walk_stops_on_has_more(
        self, dummy_client, dummy_config
    ):
        """(b) Each request carries a safe page_size and an offset; no page follows the last.

        The 2000 bound is the measured point at which a page still fits the
        documented-safe MCP response envelope — past it the response is
        rejected WHOLESALE while the caller believes it paged correctly.
        """
        import dashboard.data.tasks as tasks_mod
        from dashboard.data.tasks import fetch_statuses

        canned = CannedMCP(status_map={tid: 'pending' for tid in range(1, 4501)},
                           status_page_size=2000)

        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            await fetch_statuses(dummy_client, dummy_config, '/proj/big')

        requests = self._statuses_calls(canned)
        assert [args.get('offset') for args in requests] == [0, 2000, 4000], (
            f'the walk must tile the population once, got {requests}'
        )
        assert all(args['page_size'] <= tasks_mod.STATUSES_SAFE_PAGE_SIZE
                   for args in requests), (
            f'every page must stay inside the safe transport bound, got {requests}'
        )
        assert len(requests) == 3, (
            'the last page reports has_more=False, so no fourth request may '
            f'follow it, got {requests}'
        )

    async def test_a_clamping_server_is_walked_to_completion(
        self, dummy_client, dummy_config
    ):
        """A server that serves fewer entries than asked, with has_more still True, is fully tiled.

        This is the case the old fake made unrepresentable. It put the SERVED
        count in ``page_size``, which is the reverse of what
        ``_pagination_meta`` emits. So a walker keyed on ``page_size`` passed
        here, and against the real envelope it would have skipped the
        difference on every clamped page and handed ``build_census`` an
        incomplete census.
        """
        from dashboard.data.tasks import fetch_statuses

        population = {tid: 'done' for tid in range(1, 11)}
        # Server cap 4, serving 3 against a request of 2000: requested and
        # served differ, which is the only configuration that can tell the two
        # advance rules apart.
        canned = CannedMCP(status_map=population, status_page_size=4,
                           short_page_by=1)

        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            result = await fetch_statuses(dummy_client, dummy_config, '/proj/clamped')

        assert result == population, f'a skipped window under-reports, got {result}'
        offsets = [args.get('offset') for args in self._statuses_calls(canned)]
        assert offsets == [0, 3, 6, 9], (
            f'each window must start where the previous one ended, got {offsets}'
        )

    async def test_each_offset_is_the_previous_offset_plus_returned(
        self, dummy_client, dummy_config
    ):
        """The walk advances by what each page SERVED, read off that page.

        Asserted as the identity rather than as a list of numbers, so it holds
        for any page geometry. The server derives ``has_more = offset +
        returned < total``, and that terminator only tiles the population for
        a walk that advances by exactly ``returned``.
        """
        from dashboard.data.tasks import fetch_statuses

        canned = CannedMCP(status_map={tid: 'pending' for tid in range(1, 12)},
                           status_page_size=5, short_page_by=2)
        envelopes: list[dict] = []

        async def _recorded(client, url, tool, args, **kwargs):
            page = await canned(client, url, tool, args, **kwargs)
            envelopes.append(page['pagination'])
            return page

        with patch('dashboard.data.tasks.mcp_tool_call', new=_recorded):
            await fetch_statuses(dummy_client, dummy_config, '/proj/tiled')

        assert len(envelopes) > 1, envelopes
        for before, after in zip(envelopes, envelopes[1:], strict=False):
            assert after['offset'] == before['offset'] + before['returned'], envelopes
        assert envelopes[-1]['has_more'] is False, envelopes

    @pytest.mark.parametrize('shipped, claimed', [
        pytest.param(2, 4, id='a-page-clipped-in-flight'),
        pytest.param(4, 2, id='a-page-that-under-reports'),
    ])
    async def test_a_page_whose_returned_disagrees_with_its_entries_is_refused(
        self, dummy_client, dummy_config, shipped, claimed,
    ):
        """``returned`` is cross-checked against the entries that actually arrived.

        The walk advances on the self-reported counter, so the counter has to
        agree with the page. A page clipped in flight that still claims its
        full count would skip the clipped entries and then terminate normally.
        A page that under-reports would re-read entries already held. Neither
        can be detected afterwards, so the read is refused: the caller gets
        the offline marker, never a short map. ``_walk_pages`` performs the
        same check on the ``get_tasks`` envelope.
        """
        from dashboard.data.tasks import fetch_statuses

        canned = CannedMCP(status_map={tid: 'done' for tid in range(1, 11)},
                           status_page_size=4)

        async def _inconsistent(client, url, tool, args, **kwargs):
            page = await canned(client, url, tool, args, **kwargs)
            if args.get('offset') == 0:
                page = {
                    'statuses': dict(list(page['statuses'].items())[:shipped]),
                    'pagination': {**page['pagination'], 'returned': claimed},
                }
            return page

        with patch('dashboard.data.tasks.mcp_tool_call', new=_inconsistent):
            result = await fetch_statuses(dummy_client, dummy_config, '/proj/clipped')

        assert result.get('offline') is True, (
            f'an inconsistent page must refuse the read, got a map of {len(result)}'
        )

    async def test_an_unparseable_id_is_dropped_without_refusing_the_read(
        self, dummy_client, dummy_config
    ):
        """The cross-check counts what the SERVER sent, not what survived parsing.

        An entry whose id is not an integer is dropped from the map. Counting
        only the parsed entries against ``returned`` would turn that one entry
        into an offline marker for the whole project. ``_walk_pages`` draws
        the same line with ``_Page.delivered``.
        """
        from dashboard.data.tasks import fetch_statuses

        canned = CannedMCP(status_map={tid: 'done' for tid in range(1, 5)},
                           status_page_size=2000)

        async def _one_bad_id(client, url, tool, args, **kwargs):
            page = await canned(client, url, tool, args, **kwargs)
            statuses = dict(page['statuses'])
            statuses.pop('4')
            statuses['not-an-id'] = 'done'
            return {**page, 'statuses': statuses}

        with patch('dashboard.data.tasks.mcp_tool_call', new=_one_bad_id):
            result = await fetch_statuses(dummy_client, dummy_config, '/proj/odd')

        assert result == {1: 'done', 2: 'done', 3: 'done'}

    @pytest.mark.parametrize('returned', [
        pytest.param(_ABSENT, id='absent'),
        pytest.param(None, id='null'),
        pytest.param('4', id='a-string'),
    ])
    async def test_a_page_without_a_usable_returned_is_refused(
        self, dummy_client, dummy_config, returned,
    ):
        """No usable ``returned`` means completeness is unverifiable.

        The same rule as a missing ``has_more``: the walk advances on this
        counter, and an unverifiable read must not be reported as a complete
        one.
        """
        from dashboard.data.tasks import fetch_statuses

        canned = CannedMCP(status_map={tid: 'done' for tid in range(1, 11)},
                           status_page_size=4)

        async def _uncounted(client, url, tool, args, **kwargs):
            page = await canned(client, url, tool, args, **kwargs)
            envelope = dict(page['pagination'])
            if returned is _ABSENT:
                del envelope['returned']
            else:
                envelope['returned'] = returned
            return {**page, 'pagination': envelope}

        with patch('dashboard.data.tasks.mcp_tool_call', new=_uncounted):
            result = await fetch_statuses(dummy_client, dummy_config, '/proj/uncounted')

        assert result.get('offline') is True, (
            f'an uncounted page must refuse the read, got a map of {len(result)}'
        )

    async def test_a_server_that_never_lowers_has_more_is_cut_off_at_its_page_budget(
        self, dummy_client, dummy_config
    ):
        """The budget is ``ceil(total / first returned) + 2`` pages per url.

        ``has_more`` is the walk's only terminator and the server reports it,
        so a server that never lowers it would spin the walk on the httpx
        client the render polls share. The budget comes from the first page's
        SERVED count because that is the rate the walk advances at. Derived
        from the requested 2000 instead, it would allow three pages here, and
        it would refuse a clamping server that is merely slow to finish.
        """
        from dashboard.data.tasks import fetch_statuses

        population = {tid: 'pending' for tid in range(1, 11)}
        canned = CannedMCP(status_map=population, status_page_size=4)

        async def _endless(client, url, tool, args, **kwargs):
            # Honest page sizes, but every page claims more remains, and the
            # window wraps round instead of running dry, so only the budget
            # can stop the walk.
            wrapped = {**args, 'offset': (args.get('offset') or 0) % len(population)}
            page = await canned(client, url, tool, wrapped, **kwargs)
            return {**page, 'pagination': {**page['pagination'], 'has_more': True}}

        with patch('dashboard.data.tasks.mcp_tool_call', new=_endless):
            result = await fetch_statuses(dummy_client, dummy_config, '/proj/endless')

        assert result.get('offline') is True, result
        assert 'budget' in str(result.get('error', '')).lower(), result
        budget = -(-len(population) // 4) + 2
        expected = budget * len(dummy_config.fused_memory_urls)
        assert len(canned.calls_to('get_statuses')) == expected, (
            f'the walk must stop at its {budget}-page budget per url; issued '
            f'{len(canned.calls_to("get_statuses"))}, expected {expected}'
        )

    async def test_a_response_without_pagination_is_the_whole_answer(
        self, dummy_client, dummy_config
    ):
        """(d) The absence of the envelope means COMPLETE — one call, backward compatible.

        An older fused-memory ignores ``page_size`` and answers with the whole
        map; looping blind on that would re-request the same rows forever.
        """
        from dashboard.data.tasks import fetch_statuses

        population = {1: 'done', 2: 'pending', 3: 'in-progress'}
        canned = CannedMCP(status_map=population)

        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            result = await fetch_statuses(dummy_client, dummy_config, '/proj/old')

        assert result == population
        assert len(canned.calls_to('get_statuses')) == 1, (
            'a complete response must terminate the walk immediately, got '
            f'{self._statuses_calls(canned)}'
        )

    async def test_a_failure_mid_walk_returns_the_offline_marker(
        self, dummy_client, dummy_config
    ):
        """(e) A broken page yields the marker, never a silently short map.

        A partial map is indistinguishable from a complete one at the call
        site, and ``build_census`` would tally it into a confident
        under-report of ``total`` — the uniform lie the census exists to
        refuse.
        """
        from dashboard.data.tasks import fetch_statuses

        canned = CannedMCP(status_map={tid: 'done' for tid in range(1, 4501)},
                           status_page_size=2000)
        canned.fail_when = lambda call: call['args'].get('offset') == 2000

        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            result = await fetch_statuses(dummy_client, dummy_config, '/proj/broken')

        assert result.get('offline') is True, (
            f'a truncated walk must not be served as a map, got {type(result)} '
            f'of length {len(result)}'
        )

    async def test_the_returned_map_is_a_copy(self, dummy_client, dummy_config):
        """(f) A caller may mutate what it was handed without poisoning a later read."""
        from dashboard.data.tasks import fetch_statuses

        canned = CannedMCP(status_map={1: 'done', 2: 'pending'},
                           status_page_size=1)

        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            first = await fetch_statuses(dummy_client, dummy_config, '/proj/A')
            assert isinstance(first, dict)
            first.clear()
            second = await fetch_statuses(dummy_client, dummy_config, '/proj/A')

        assert second == {1: 'done', 2: 'pending'}, (
            f'the map was mutated through the returned reference: {second}'
        )

    async def test_two_consecutive_reads_issue_two_walks(
        self, dummy_client, dummy_config
    ):
        """The cache is gone: nothing between the caller and the substrate.

        The unit above owns the only TTL on this path, so a read here is
        always a live one — which is what lets the snapshot stamp an ``as_of``
        it can stand behind.
        """
        from dashboard.data.tasks import fetch_statuses

        canned = CannedMCP(status_map={1: 'done'}, status_page_size=2000)

        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            await fetch_statuses(dummy_client, dummy_config, '/proj/A')
            await fetch_statuses(dummy_client, dummy_config, '/proj/A')

        assert len(canned.calls_to('get_statuses')) == 2, (
            'a cached read would collapse these to one, and the unit above '
            'would then serve a map older than the as_of it stamps'
        )
