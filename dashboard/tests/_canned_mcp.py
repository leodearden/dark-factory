"""The fused-memory substrate dashboard tests drive task reads against.

Shared by every test that patches ``dashboard.data.tasks.mcp_tool_call`` so
the real ``fetch_tasks`` / ``fetch_statuses`` / cache / fan-out path runs
underneath — one faithful emulation rather than a hand-rolled fake per test
module. Its own fidelity tests live in
``tests/test_task_snapshot.py::TestCannedMCP``. :func:`canned_get_tasks_result`
builds the canned two-row ``get_tasks`` answer the cache, narrowing and
contract tests stub with.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping

import httpx


class CannedMCP:
    """A stand-in for :func:`dashboard.data.memory.mcp_tool_call`.

    Faithful to the substrate contracts the units above it depend on,
    because a paging loop tested against a non-paging fake tests nothing:

    * ``get_tasks`` applies ``statuses`` as a row filter SERVER-side, then
      slices ``page_size``/``offset`` over a list ordered by ascending ``id``
      — so reaching the high-id end requires a computed offset. No
      ``pagination`` envelope, matching ``test_active_tasks.py::_canned_mcp``:
      the unit's row read is unpaginated and the terminal window discards the
      envelope, so emitting one would fake a contract nothing here reads.
    * ``get_statuses`` slices a deterministic total order and answers with the
      five-key ``pagination`` envelope
      (``total``/``offset``/``page_size``/``returned``/``has_more``) exactly
      as ``fused_memory/server/tools.py::_pagination_meta`` builds it:
      ``page_size`` echoes the REQUESTED size verbatim, ``returned`` is the
      count actually SERVED, and ``has_more`` is ``offset + returned <
      total``. That includes serving a page SMALLER than requested, which is
      the case that tells a loop advancing by ``returned`` apart from one
      advancing by the size it asked for.
    * ``get_task`` answers ONE row by id from the same :attr:`rows`, whatever
      its status, or — for an id it does not hold — the structured
      ``TaskNotFoundError`` answer fused-memory gives. It honours a per-id
      delay, and :attr:`max_in_flight` records the most ``get_task`` calls
      ever outstanding at once, so a bounded fan-out is assertable.

    Args:
        rows: Raw MCP ``get_tasks`` rows (string ids), as
            :func:`_raw_row` builds them.
        status_map: ``{int id: status}`` — the population ``get_statuses``
            serves.
        status_page_size: The server's own page cap. ``None`` (default) is the
            backward-compatible mode: the whole map in one response with NO
            ``pagination`` key, which the contract defines as COMPLETE.
        short_page_by: Serve this many FEWER statuses per page than the
            smaller of the requested and the server cap. The served count is
            reported in ``pagination['returned']``, like any other page. Zero
            disables it.
        task_delays: ``{int id: seconds}`` — how long ``get_task`` takes to
            answer for that id. Absent ids answer at once.

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
        task_delays: Mapping[int, float] | None = None,
    ) -> None:
        self.rows = [dict(row) for row in rows]
        self.status_map = dict(status_map or {})
        self.status_page_size = status_page_size
        self.short_page_by = short_page_by
        self.task_delays = dict(task_delays or {})
        self.fail_when: Callable[[dict], bool] = lambda call: False
        self.calls: list[dict] = []
        self.max_in_flight = 0
        self._in_flight = 0

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
        if tool == 'get_task':
            return await self._task(args)
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
                'page_size': requested,
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

    async def _task(self, args: dict) -> dict:
        task_id = int(args['id'])
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            await asyncio.sleep(self.task_delays.get(task_id, 0))
        finally:
            self._in_flight -= 1
        for row in self.rows:
            if int(row['id']) == task_id:
                return dict(row)
        return {
            'error': f'No tasks found for ID(s): {task_id}',
            'error_type': 'TaskNotFoundError',
        }


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


def canned_get_tasks_result() -> dict:
    """A fresh two-row ``get_tasks`` answer, so no caller can mutate another's."""
    return {
        'tasks': [
            {
                'id': '7',
                'title': 'A done task',
                'status': 'done',
                'updatedAt': '2026-05-29T10:00:00+00:00',
                'description': 'finished',
                'details': '',
                'dependencies': [],
                'metadata': {},
            },
            {
                'id': '8',
                'title': 'A pending task',
                'status': 'pending',
                'description': '',
                'details': '',
                'dependencies': [],
                'metadata': {},
            },
        ],
    }
