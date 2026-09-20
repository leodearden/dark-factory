"""The one task snapshot unit — ``dashboard.data.task_snapshot``.

Every test here drives the unit through its PUBLIC seam: the canned
fused-memory substrate below is patched at ``dashboard.data.tasks.mcp_tool_call``
(a public name), so the real ``fetch_tasks`` / ``fetch_statuses`` / cache /
fan-out path runs underneath and no private name is reached into.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

# ---------------------------------------------------------------------------
# CannedMCP — the fused-memory substrate, emulated faithfully enough that a
# test above it is not vacuous (task 5587 pre-1)
# ---------------------------------------------------------------------------


class CannedMCP:
    """A stand-in for :func:`dashboard.data.memory.mcp_tool_call`.

    Faithful to the two substrate contracts the unit above it depends on,
    because a paging loop tested against a non-paging fake tests nothing:

    * ``get_tasks`` applies ``statuses`` as a row filter SERVER-side, then
      slices ``page_size``/``offset`` over a list ordered by ascending ``id``
      — so reaching the high-id end requires a computed offset. No
      ``pagination`` envelope, matching ``test_active_tasks.py::_canned_mcp``:
      the unit's row read is unpaginated and the terminal window discards the
      envelope, so emitting one would fake a contract nothing here reads.
    * ``get_statuses`` slices a deterministic total order and answers with the
      five-key ``pagination`` envelope
      (``total``/``offset``/``page_size``/``returned``/``has_more``) that
      ``fused_memory/server/tools.py::get_statuses`` documents — including
      serving a page SMALLER than requested, which is the case that
      distinguishes a loop advancing by the SERVED ``pagination['page_size']``
      from one advancing by the size it asked for.

    Args:
        rows: Raw MCP ``get_tasks`` rows (string ids), as
            :func:`_raw_row` builds them.
        status_map: ``{int id: status}`` — the population ``get_statuses``
            serves.
        status_page_size: The server's own page cap. ``None`` (default) is the
            backward-compatible mode: the whole map in one response with NO
            ``pagination`` key, which the contract defines as COMPLETE.
        short_page_by: Serve this many FEWER statuses per page than the
            smaller of the requested and the server cap, reporting the served
            size in ``pagination['page_size']``. Zero disables it.

    Mutable after construction so one instance can change behaviour between a
    test's two acquisitions. :attr:`fail_when` is a predicate over the
    recorded call — ``{'tool', 'args', 'kwargs'}`` — and every call it accepts
    raises ``httpx.ReadTimeout``. A predicate rather than a set of tool names
    because the two failures worth injecting are not the same shape: "this
    half is unreachable now" keys on the tool, while "the walk breaks at its
    SECOND page" keys on the offset, and only the latter distinguishes an
    offline marker from a silently short map.
    """

    def __init__(
        self,
        rows=(),
        status_map=None,
        *,
        status_page_size: int | None = None,
        short_page_by: int = 0,
    ) -> None:
        self.rows = [dict(row) for row in rows]
        self.status_map = dict(status_map or {})
        self.status_page_size = status_page_size
        self.short_page_by = short_page_by
        self.fail_when: Callable[[dict], bool] = lambda call: False
        self.calls: list[dict] = []

    def calls_to(self, tool: str) -> list[dict]:
        """Every recorded call to *tool*, in order."""
        return [call for call in self.calls if call['tool'] == tool]

    async def __call__(self, client, url, tool, args, **kwargs):
        # ``kwargs`` is recorded too: the per-request budget rides as the
        # ``timeout=`` keyword and never inside ``args``, so it is only
        # assertable at the wire if it is kept.
        call = {'tool': tool, 'args': dict(args), 'kwargs': dict(kwargs)}
        self.calls.append(call)
        if self.fail_when(call):
            raise httpx.ReadTimeout(f'canned {tool} read timeout')
        if tool == 'get_statuses':
            return self._statuses(args)
        if tool == 'get_tasks':
            return self._tasks(args)
        raise AssertionError(f'unexpected tool {tool!r}')

    def _statuses(self, args: dict) -> dict:
        ordered = sorted(self.status_map.items(), key=lambda item: int(item[0]))
        if self.status_page_size is None:
            return {'statuses': {str(tid): status for tid, status in ordered}}

        requested = args.get('page_size') or self.status_page_size
        served = max(1, min(requested, self.status_page_size) - self.short_page_by)
        offset = args.get('offset') or 0
        window = ordered[offset:offset + served]
        return {
            'statuses': {str(tid): status for tid, status in window},
            'pagination': {
                'total': len(ordered),
                'offset': offset,
                # What was ACTUALLY served, which is what the caller must
                # advance by — see get_statuses' "Why page_size is not
                # clamped" note.
                'page_size': served,
                'returned': len(window),
                'has_more': offset + len(window) < len(ordered),
            },
        }

    def _tasks(self, args: dict) -> dict:
        statuses = args.get('statuses')
        selected = [
            row for row in self.rows
            if statuses is None or row.get('status') in statuses
        ]
        selected.sort(key=lambda row: int(row['id']))  # ORDER BY id ASC
        page_size = args.get('page_size')
        if page_size is not None:
            start = args.get('offset') or 0
            selected = selected[start:start + page_size]
        return {'tasks': selected}


def _raw_row(task_id, status, **overrides) -> dict:
    """One raw MCP ``get_tasks`` row, distinguishable by id."""
    row = {
        'id': str(task_id),
        'title': f'task {task_id}',
        'status': status,
        'dependencies': [],
        'metadata': {},
        'updatedAt': '2026-01-01T00:00:00+00:00',
    }
    row.update(overrides)
    return row


class TestCannedMCP:
    """The emulation's own fidelity, since every test above it rests on it."""

    async def test_paged_statuses_carry_the_five_key_envelope(self):
        canned = CannedMCP(status_map={i: 'done' for i in range(1, 6)},
                           status_page_size=2)
        page = await canned(None, 'u', 'get_statuses',
                            {'project_root': '/p', 'page_size': 2, 'offset': 0})

        assert page['pagination'] == {
            'total': 5, 'offset': 0, 'page_size': 2, 'returned': 2, 'has_more': True,
        }
        assert page['statuses'] == {'1': 'done', '2': 'done'}

    async def test_a_short_page_reports_the_size_it_served(self):
        """The case that separates advancing by SERVED from advancing by requested."""
        canned = CannedMCP(status_map={i: 'done' for i in range(1, 6)},
                           status_page_size=4, short_page_by=1)
        page = await canned(None, 'u', 'get_statuses',
                            {'project_root': '/p', 'page_size': 4, 'offset': 0})

        assert page['pagination']['page_size'] == 3, 'served, not requested'
        assert page['pagination']['returned'] == 3
        assert len(page['statuses']) == 3

    async def test_no_page_size_configured_means_no_pagination_key(self):
        """The absence of the key is the substrate's spelling of COMPLETE."""
        canned = CannedMCP(status_map={1: 'done', 2: 'pending'})
        page = await canned(None, 'u', 'get_statuses', {'project_root': '/p'})

        assert 'pagination' not in page
        assert page['statuses'] == {'1': 'done', '2': 'pending'}

    async def test_an_injected_failure_raises_read_timeout(self):
        canned = CannedMCP(status_map={1: 'done'})
        canned.fail_when = lambda call: call['tool'] == 'get_statuses'

        with pytest.raises(httpx.ReadTimeout):
            await canned(None, 'u', 'get_statuses', {'project_root': '/p'})

    async def test_get_tasks_filters_then_slices_by_ascending_id(self):
        canned = CannedMCP(rows=[_raw_row(i, 'done' if i % 2 else 'pending')
                                 for i in range(1, 8)])
        page = await canned(None, 'u', 'get_tasks',
                            {'project_root': '/p', 'statuses': ['done'],
                             'page_size': 2, 'offset': 1})

        assert [row['id'] for row in page['tasks']] == ['3', '5']
