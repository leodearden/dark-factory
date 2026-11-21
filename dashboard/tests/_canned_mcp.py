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
import re
from collections.abc import Callable, Mapping
from pathlib import Path

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
    * ``get_tasks`` with ``fields`` projects each served row exactly as
      ``fused_memory/backends/task_projection.py::TaskProjection.apply`` does
      (PRD dashboard-prd-lanes §5.1): ``id`` always, the requested top-level
      keys, and — when any ``metadata.<key>`` is requested — ``metadata``
      holding only the requested keys PRESENT on the row. Requesting
      ``metadata`` itself keeps the whole dict. Mirrored, not imported:
      dashboard tests may not depend on fused-memory.
    * ``fields`` is validated first, as ``fused_memory/server/tools.py::_validate_fields``
      does through ``task_projection.py::parse_task_projection``: an entry
      that is neither a row key nor ``metadata.<key>`` with ``<key>``
      matching ``[A-Za-z_][A-Za-z0-9_]*`` fails the WHOLE read with the
      server's ``ValidationError`` payload, which the dashboard reads as that
      URL failing.
    * With :attr:`audit_projection` set, a ``get_tasks`` carrying ``fields``
      serves its rows UNPROJECTED instead, and every ``get`` / ``[]`` / ``in``
      of a key the projection would have dropped lands in
      :attr:`unprojected_reads`, spelled as it is in ``fields``
      (``'title'``, ``'metadata.routing'``). Full rows keep every branch that
      tests a value reachable, so a reader of an omitted key is caught whether
      or not the projection would have hidden its answer.
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
        metadata_projection_delay: Seconds a metadata projection
            (:func:`is_metadata_projection`, the lineage read) waits before
            answering. A row read carrying ``fields`` is not delayed. Zero
            answers at once.

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
        metadata_projection_delay: float = 0.0,
    ) -> None:
        self.rows = [dict(row) for row in rows]
        self.status_map = dict(status_map or {})
        self.status_page_size = status_page_size
        self.short_page_by = short_page_by
        self.task_delays = dict(task_delays or {})
        self.metadata_projection_delay = metadata_projection_delay
        self.audit_projection: bool = False
        self.unprojected_reads: set[str] = set()
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
            if is_metadata_projection(args):
                await asyncio.sleep(self.metadata_projection_delay)
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
        fields = args.get('fields')
        refusal = _projection_refusal(fields)
        if refusal is not None:
            return refusal
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
        if fields is not None and self.audit_projection:
            selected = [_audited(row, fields, self.unprojected_reads) for row in selected]
        elif fields is not None:
            selected = [_project(row, fields) for row in selected]
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


_METADATA_FIELD_PREFIX = 'metadata.'
_TASK_ROW_KEYS: tuple[str, ...] = (
    'id', 'title', 'description', 'details', 'testStrategy', 'status', 'dependencies',
    'priority', 'subtasks', 'updatedAt', 'metadata', 'claimant_run_id', 'heartbeat_at',
    'candidate_key', 'created_at', 'created_at_source',
)
"""``task_projection.py::TASK_ROW_KEYS``: the top-level keys a projection may name."""
_METADATA_KEY_PATTERN = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
_ALLOWED_FIELDS: tuple[str, ...] = (*_TASK_ROW_KEYS, f'{_METADATA_FIELD_PREFIX}<key>')


def _projectable(entry: object) -> bool:
    """Whether ``parse_task_projection`` accepts *entry* as one ``fields`` entry."""
    if not isinstance(entry, str):
        return False
    if entry in _TASK_ROW_KEYS:
        return True
    return entry.startswith(_METADATA_FIELD_PREFIX) and bool(
        _METADATA_KEY_PATTERN.fullmatch(entry.removeprefix(_METADATA_FIELD_PREFIX)),
    )


def _projection_refusal(fields: object) -> dict | None:
    """The server's answer to a ``get_tasks`` whose *fields* it refuses, else ``None``."""
    if fields is None:
        return None
    if isinstance(fields, (list, tuple)):
        unprojectable = [entry for entry in fields if not _projectable(entry)]
        if not unprojectable:
            return None
        refused = unprojectable[0]
    else:
        refused = fields
    return {
        'error': (
            f'cannot project {refused!r}: fields must be a list of entries drawn from '
            f'{list(_ALLOWED_FIELDS)}, where <key> matches {_METADATA_KEY_PATTERN.pattern}'
        ),
        'error_type': 'ValidationError',
        'field': refused,
        'allowed': list(_ALLOWED_FIELDS),
    }


def is_metadata_projection(args: Mapping) -> bool:
    """Whether a ``get_tasks`` *args* asks for ids and metadata keys only.

    That is the shape ``dashboard.data.tasks::fetch_metadata_projection`` builds
    for the lineage read, and no row read can take it: a row read names
    top-level keys such as ``title``.
    """
    fields = args.get('fields')
    return fields is not None and all(
        entry == 'id' or entry.startswith(_METADATA_FIELD_PREFIX) for entry in fields
    )


def _row_keys(fields: list[str]) -> set[str]:
    """The top-level keys a projection to *fields* keeps."""
    return {'id'} | {entry for entry in fields if not entry.startswith(_METADATA_FIELD_PREFIX)}


def _metadata_keys(fields: list[str]) -> list[str]:
    """The metadata keys a projection to *fields* names."""
    return [
        entry.removeprefix(_METADATA_FIELD_PREFIX)
        for entry in fields if entry.startswith(_METADATA_FIELD_PREFIX)
    ]


class _AuditedDict(dict):
    """A served dict that records each read of a key outside *allowed* into *reads*."""

    def __init__(self, data: Mapping, *, allowed: set[str], spelling: str, reads: set[str]) -> None:
        super().__init__(data)
        self._allowed = allowed
        self._spelling = spelling
        self._reads = reads

    def _note(self, key) -> None:
        if key not in self._allowed:
            self._reads.add(f'{self._spelling}{key}')

    def get(self, key, default=None):
        self._note(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self._note(key)
        return super().__getitem__(key)

    def __contains__(self, key) -> bool:
        self._note(key)
        return super().__contains__(key)


def _audited(row: dict, fields: list[str], reads: set[str]) -> dict:
    """*row* served whole, recording reads of what a projection to *fields* would drop."""
    row_keys = _row_keys(fields)
    metadata_keys = set(_metadata_keys(fields))
    served = dict(row)
    metadata = served.get('metadata')
    if isinstance(metadata, dict) and 'metadata' not in row_keys:
        served['metadata'] = _AuditedDict(
            metadata, allowed=metadata_keys, spelling=_METADATA_FIELD_PREFIX, reads=reads,
        )
    served_keys = row_keys | ({'metadata'} if metadata_keys else set())
    return _AuditedDict(served, allowed=served_keys, spelling='', reads=reads)


def _project(row: dict, fields: list[str]) -> dict:
    """*row* as α's ``TaskProjection.apply`` projects it for *fields*."""
    row_keys = _row_keys(fields)
    projected = {key: value for key, value in row.items() if key in row_keys}
    metadata_keys = _metadata_keys(fields)
    if metadata_keys and 'metadata' not in row_keys:
        metadata = row.get('metadata') or {}
        projected['metadata'] = {key: metadata[key] for key in metadata_keys if key in metadata}
    return projected


class ByRoot:
    """One canned substrate per project root, routed by the read's ``project_root``.

    Keyed by the root's directory name, which is the project label the
    dashboard derives for it. ``get_external_statuses`` names no root: it
    answers each ``project:id`` dep from the substrate labelled ``project``,
    flat, with fused-memory's ``unknown_project`` / ``unknown_task`` sentinels
    for one no substrate holds.
    """

    def __init__(self, by_label: Mapping[str, CannedMCP]) -> None:
        self._by_label = by_label

    async def __call__(self, client, url, tool, args, **kwargs):
        if tool == 'get_external_statuses':
            return {dep: self._external_status(dep) for dep in args['deps']}
        substrate = self._by_label[Path(args['project_root']).name]
        return await substrate(client, url, tool, args, **kwargs)

    def _external_status(self, dep: str) -> str:
        label, _, task_id = dep.partition(':')
        substrate = self._by_label.get(label)
        if substrate is None:
            return 'unknown_project'
        return substrate.status_map.get(int(task_id), 'unknown_task')


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
