"""Tests for how a tasks read returns ONE page versus the COMPLETE set.

The file covers ``_fetch_page`` (one MCP call, envelope preserved),
``_walk_pages`` (assembly over an injected page fn, with all-or-nothing
completeness), ``fetch_tasks(chunk_size=N)`` end to end through the fan-out,
and the public ``fetch_task_page``/``fetch_tasks`` split. The completeness rule
is asserted at both layers on purpose: TestWalkPages owns the six failure
detections over an injected page fn, while TestFetchTasksPagination drives the
public read and owns the operator-facing error text, the request sequence, the
per-URL page budget and the per-request timeout.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from _canned_mcp import canned_get_tasks_result

import dashboard.data.tasks as tasks_mod
from dashboard.data.tasks import _shape_task

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

    async def test_a_walk_never_assembles_pages_from_two_servers(
        self, dummy_client, two_url_config
    ):
        """A walk that fails mid-way RESTARTS on the next url; it never resumes.

        `first_success` tries urls IN ORDER, so a walk free to fan out per page
        would assemble one list from DIFFERENT servers and silently invalidate
        the changed-`total` coherence check: pages from two states of the world,
        with every counter still self-consistent. Binding ONE url for the whole
        walk is what makes that unrepresentable. The two servers here serve
        DISJOINT id ranges, so a mixed answer is visible rather than plausible —
        which is the only way this property can be caught after the fact.
        """
        from dashboard.data.tasks import fetch_tasks

        first_url, second_url = two_url_config.fused_memory_urls
        healthy = [_paged_task_raw(i) for i in range(101, 108)]
        drops_out = [_paged_task_raw(i) for i in range(1, 8)]
        calls: list[tuple[str, dict]] = []

        async def _call(_client, url, tool, args, **_kwargs):
            assert tool == 'get_tasks', tool
            calls.append((url, dict(args)))
            if url == first_url and args.get('offset', 0) >= 3:
                return {'error': 'this server drops out after one page'}
            source = drops_out if url == first_url else healthy
            offset, page_size = args.get('offset', 0), args['page_size']
            page = source[offset:offset + page_size]
            return {
                'tasks': page,
                'pagination': {
                    'total': len(source),
                    'offset': offset,
                    'page_size': page_size,
                    'returned': len(page),
                    'has_more': offset + len(page) < len(source),
                },
            }

        with patch(
            'dashboard.data.tasks.mcp_tool_call', new=AsyncMock(side_effect=_call),
        ):
            result = await fetch_tasks(
                dummy_client, two_url_config, '/proj/split-brain', chunk_size=3,
            )

        assert isinstance(result, list)
        first_offsets = [args['offset'] for url, args in calls if url == first_url]
        assert first_offsets == [0, 3], (
            'the scenario requires the first url to serve one page and THEN '
            f'fail mid-walk; got {first_offsets!r}'
        )
        assert [t['id'] for t in result] == list(range(101, 108)), (
            f'one server must serve the WHOLE walk; got a mixture: {result!r}'
        )
        second_offsets = [args['offset'] for url, args in calls if url == second_url]
        assert second_offsets[0] == 0, (
            'the surviving url restarts the walk rather than resuming it at the '
            f'offset the first one died on; got {second_offsets!r}'
        )

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
        assert 'changed from 20 to 50' in error, (
            f'the error must report the observed change so the race is '
            f'diagnosable; got {error!r}'
        )

    async def test_a_total_that_shrinks_mid_walk_yields_the_marker(
        self, dummy_client, dummy_config,
    ):
        """A SHRINKING ``total`` is the same incoherence with a worse ending.

        The grow case at least un-bounds the page budget, so it trips something
        either way.  A shrink does not: the loop terminates EARLY on
        ``walk_offset >= total`` with every counter self-consistent and hands
        the assembled prefix back as a plain ``list``, which
        ``collect_snapshot`` triages on ``isinstance(result, list)`` and writes
        into the append-only ``snapshots`` table as fact.  The server re-slices
        the now-shorter list, so the pages after the deletion skip rows
        outright — the undercount is not even a prefix of the real tree.  Hence
        the coherence check is on INEQUALITY, not growth.
        """
        from dashboard.data.tasks import fetch_tasks

        tasks = [_paged_task_raw(i) for i in range(1, 101)]

        async def _shrinking(_client, _url, tool, args, **_kwargs):
            offset = args.get('offset', 0)
            page_size = args.get('page_size') or 0
            # The 45 lowest-id tasks are deleted once the walk is under way,
            # so pages after the first are sliced out of a 55-task tree.
            live = tasks if offset == 0 else tasks[45:]
            page = live[offset:offset + page_size]
            return {
                'tasks': page,
                'pagination': {
                    'total': len(live), 'offset': offset,
                    'page_size': page_size, 'returned': len(page),
                    'has_more': True,
                },
            }

        with patch(
            'dashboard.data.tasks.mcp_tool_call', new=AsyncMock(side_effect=_shrinking),
        ):
            result = await fetch_tasks(
                dummy_client, dummy_config, '/proj/deleting', chunk_size=10,
            )

        assert isinstance(result, dict), (
            f'55 rows sliced out of a shrinking tree are not the whole tree, '
            f'and a list here would be written to snapshots as one; got {result!r}'
        )
        assert result.get('offline') is True, result
        error = str(result.get('error', ''))
        assert '/proj/deleting' in error, error
        assert 'changed from 100 to 55' in error, (
            f'the error must report the observed change so the race is '
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

    async def test_a_truncation_marker_is_not_retained(
        self, dummy_client, dummy_config,
    ):
        """A truncated walk's marker answers only the call that met it.

        Nothing holds it (task 5598 retired both ``fetch_tasks`` caches), so
        the very next call against a healthy server returns real rows.
        """
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

            healthy[0] = True
            second = await fetch_tasks(
                dummy_client, dummy_config, '/proj/flaky', chunk_size=3,
            )

        assert isinstance(second, list), (
            f'the marker must not outlive the call that met it; got {second!r}'
        )
        assert [t['id'] for t in second] == list(range(1, 8)), second

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
    """`_walk_pages(page_fn, project_root, chunk_size)` — assembly over an INJECTED page fn.

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

    # The PINNED-URL constraint is not asserted here. It is behavioural, and
    # lives in TestFetchTasksPagination, which drives two servers serving
    # DISJOINT id ranges so a mixed answer is visible rather than plausible:
    #     test_a_walk_never_assembles_pages_from_two_servers

    # ---- (c) the two COMPLETE cases: a true zero, not a truncation --------

    async def test_an_empty_tree_returns_empty_rather_than_raising(self):
        """`total <= 0` is a complete read of an empty tree.

        Get this wrong and every empty project becomes a permanent burndown
        hole instead of its legitimate all-zero row.
        """
        page_fn = self._pager([([], {'returned': 0, 'total': 0})])
        assert await tasks_mod._walk_pages(page_fn, '/proj/walk', 3) == []

    async def test_an_exhausted_tree_returns_its_rows(self):
        """`offset >= total` with an empty page is exhaustion, not truncation."""
        rows = [_shape_task(_paged_task_raw(i)) for i in (1, 2)]
        page_fn = self._pager([
            ([r for r in rows if r], {'returned': 2, 'total': 2}),
        ])
        out = await tasks_mod._walk_pages(page_fn, '/proj/walk', 3)
        assert [r['id'] for r in out] == [1, 2]

    # ---- (d) a server that ignores paging ---------------------------------

    async def test_a_server_that_ignores_paging_stops_after_one_page(self):
        """No envelope means this response IS the answer — take it and stop.

        Looping blind would either spin or re-request the same rows forever.
        """
        calls: list = []
        rows = [r for r in (_shape_task(_paged_task_raw(i)) for i in range(1, 8)) if r]
        page_fn = self._pager([(rows, None)], calls)

        out = await tasks_mod._walk_pages(page_fn, '/proj/walk', 3)
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

        out = await tasks_mod._walk_pages(page_fn, '/proj/walk', 3)
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

        out = await tasks_mod._walk_pages(page_fn, '/proj/walk', 3)
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

        await tasks_mod._walk_pages(page_fn, read.project_root, 3)
        assert len(calls) == 2
        for window in calls:
            assert read.wire_arguments(window)['statuses'] == ['done']

    # ---- (b) the five completeness failures, all-or-nothing ---------------
    #      Six rows, five failures: the changed-`total` check fires both ways.

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
                'total changed mid-walk — grew',
                [
                    ([{'id': '1'}, {'id': '2'}, {'id': '3'}],
                     {'returned': 3, 'total': 9}),
                    ([{'id': '4'}, {'id': '5'}, {'id': '6'}],
                     {'returned': 3, 'total': 99}),
                ],
            ),
            (
                'total changed mid-walk — shrank',
                # The check is on INEQUALITY, so this fires on page 2. Left to
                # a `>` check it would not: the loop would break at
                # `walk_offset >= 4` and hand back 6 of 9 rows as a complete
                # list, which is the worse of the two endings.
                [
                    ([{'id': '1'}, {'id': '2'}, {'id': '3'}],
                     {'returned': 3, 'total': 9}),
                    ([{'id': '4'}, {'id': '5'}, {'id': '6'}],
                     {'returned': 3, 'total': 4}),
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
            tasks_mod._walk_pages(page_fn, '/proj/walk', 3)
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
            await tasks_mod._walk_pages(_page_fn, '/proj/walk', 3)


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
