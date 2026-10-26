"""An in-process fused-memory MCP server for the deferral scripts' suites.

A plain module rather than a conftest fixture, the same convention as
``legibility_task_store_fake``: conftest appends this directory to
``sys.path``, so the test files import it by bare name. ``FakeFusedMemory.post``
has the ``httpx.post`` signature ``legibility.census_trigger.post_mcp_envelope``
calls, so a test installs it with ``install_fake_httpx(fake.post)``.

``set_task_status`` emulates β's guarded re-stamp (``restamp_only``) through
the shared model: ``task_deferral.parse_request`` decides every pure refusal,
and only the two store checks β corroborates (the carrier exists, the carrier
is not the row itself) are modelled here.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from shared.task_deferral import (
    CarriedByRequest,
    DeferralInvalid,
    DeferralInvalidReason,
    DeferralRefusalCode,
    InvalidRecord,
    parse_record,
    parse_request,
    record_to_json,
    stamp,
)

METADATA_LOOKUP_LIMIT_CAP = 100


class FakeResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body
        self.status_code = 200
        self.headers = {'content-type': 'application/json'}
        self.text = repr(body)

    def json(self) -> dict[str, Any]:
        return self._body

    def raise_for_status(self) -> None:
        return None


def _metadata(row: Mapping[str, Any]) -> Any:
    return row.get('metadata')


def _int_id(row: Mapping[str, Any]) -> int:
    return int(row['id'])


class FakeFusedMemory:
    """*projects* maps a project root string to its get_tasks-wire row dicts.
    A root not in *projects*, or in ``unreadable``, answers an error payload.
    With *echo* False, or for a root in ``unechoing``, a successful re-stamp
    omits ``deferral``, as a server that predates the deferral gate would."""

    def __init__(self, projects: Mapping[str, Sequence[Mapping[str, Any]]], *, echo: bool = True) -> None:
        self.projects: dict[str, list[dict[str, Any]]] = {
            root: [copy.deepcopy(dict(row)) for row in rows] for root, rows in projects.items()
        }
        self.echo = echo
        self.unreadable: set[str] = set()
        self.unechoing: set[str] = set()
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._moves: dict[tuple[str, int], str] = {}
        self._kind_refusals: dict[tuple[str, int, str], DeferralInvalidReason] = {}

    def move_before_write(self, root: str, task_id: int, status: str) -> None:
        """Just before *task_id*'s next write, another writer moves the row to *status*."""
        self._moves[(root, task_id)] = status

    def refuse_kind(self, root: str, task_id: int, kind: str, reason_code: DeferralInvalidReason) -> None:
        """Every write of a *kind* record to *task_id* is refused deferral_invalid *reason_code*."""
        self._kind_refusals[(root, task_id, kind)] = reason_code

    def row(self, root: str, task_id: int) -> dict[str, Any]:
        (found,) = [row for row in self.projects[root] if _int_id(row) == task_id]
        return found

    def calls_to(self, tool: str) -> list[dict[str, Any]]:
        return [arguments for name, arguments in self.calls if name == tool]

    def post(self, url: str, *, json: dict[str, Any], headers: Mapping[str, str], timeout: float) -> FakeResponse:
        params = json['params']
        tool, arguments = params['name'], copy.deepcopy(params['arguments'])
        self.calls.append((tool, arguments))
        return FakeResponse(
            {'jsonrpc': '2.0', 'id': json.get('id'), 'result': {'structuredContent': self._answer(tool, arguments)}}
        )

    def _answer(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        root = arguments.get('project_root')
        if root in self.unreadable or root not in self.projects:
            return {'error': f'project_root {root!r} is not a known project', 'error_type': 'ValidationError'}
        rows = self.projects[root]
        match tool:
            case 'get_tasks':
                return self._get_tasks(root, rows, arguments)
            case 'get_statuses':
                return self._get_statuses(rows, arguments)
            case 'find_tasks_by_metadata':
                return self._find_tasks_by_metadata(rows, arguments)
            case 'set_task_status':
                return self._set_task_status(root, rows, arguments)
        return {'error': f'the fake has no tool {tool!r}', 'error_type': 'ValidationError'}

    @staticmethod
    def _get_tasks(root: str, rows: list[dict[str, Any]], arguments: Mapping[str, Any]) -> dict[str, Any]:
        statuses = arguments.get('statuses')
        selected = [row for row in rows if statuses is None or row['status'] in statuses]
        return {'tasks': copy.deepcopy(selected), 'project_root': root}

    @staticmethod
    def _get_statuses(rows: list[dict[str, Any]], arguments: Mapping[str, Any]) -> dict[str, Any]:
        wanted = arguments.get('ids')
        by_id = {str(row['id']): row['status'] for row in rows}
        if wanted is None:
            return {'statuses': by_id}
        return {'statuses': {task_id: by_id[task_id] for task_id in wanted if task_id in by_id}}

    @staticmethod
    def _find_tasks_by_metadata(rows: list[dict[str, Any]], arguments: Mapping[str, Any]) -> dict[str, Any]:
        key = arguments['key']
        matches = sorted(
            (row for row in rows if isinstance(_metadata(row), dict) and key in _metadata(row)), key=_int_id
        )
        offset = arguments.get('offset', 0)
        page_size = min(arguments.get('limit', 50), METADATA_LOOKUP_LIMIT_CAP)
        page = matches[offset : offset + page_size]
        return {
            'key': key,
            'results': [
                {'task_id': _int_id(row), 'title': row['title'], 'status': row['status'], 'priority': 'medium'}
                for row in page
            ],
            'tasks_with_key': len(matches),
            'pagination': {
                'total': len(matches),
                'offset': offset,
                'page_size': page_size,
                'returned': len(page),
                'has_more': offset + len(page) < len(matches),
            },
        }

    def _set_task_status(self, root: str, rows: list[dict[str, Any]], arguments: Mapping[str, Any]) -> dict[str, Any]:
        task_id = int(arguments['id'])
        moved = self._moves.pop((root, task_id), None)
        found = [row for row in rows if _int_id(row) == task_id]
        if not found:
            return {'success': False, 'error': f'task {task_id} not found', 'error_type': 'TaskNotFoundError'}
        (row,) = found
        if moved is not None:
            row['status'] = moved
        if arguments.get('status') != 'deferred' or not arguments.get('restamp_only'):
            return {'error': 'the fake models only the restamp_only write to deferred', 'error_type': 'ValidationError'}
        now = datetime.now(UTC)
        try:
            request = parse_request(arguments.get('deferral'), now=now)
            self._corroborate(rows, request)
        except DeferralInvalid as exc:
            return _invalid(task_id, exc.reason, exc.field, exc.value)
        changed = self._changed(row, task_id, arguments)
        if changed is not None:
            return changed
        if isinstance(request, CarriedByRequest) and request.carrier_task_id == task_id:
            return _invalid(task_id, DeferralInvalidReason.CARRIER_IS_SELF, 'carrier_task_id', task_id)
        forced = self._kind_refusals.get((root, task_id, request.kind.value))
        if forced is not None:
            return _invalid(task_id, forced, 'kind', request.kind.value)
        record = record_to_json(stamp(request, stamped_at=now, stamped_by=arguments.get('agent_id'), deferred_at=None))
        metadata = row['metadata'] if isinstance(row.get('metadata'), dict) else {}
        row['metadata'] = {**metadata, 'deferral': record}
        reply: dict[str, Any] = {'success': True, 'task_id': str(task_id), 'status': 'deferred'}
        return {**reply, 'deferral': record} if self.echo and root not in self.unechoing else reply

    @staticmethod
    def _corroborate(rows: list[dict[str, Any]], request: object) -> None:
        if isinstance(request, CarriedByRequest) and request.carrier_task_id not in {_int_id(row) for row in rows}:
            raise DeferralInvalid(DeferralInvalidReason.CARRIER_NOT_FOUND, 'carrier_task_id', request.carrier_task_id)

    @staticmethod
    def _changed(row: Mapping[str, Any], task_id: int, arguments: Mapping[str, Any]) -> dict[str, Any] | None:
        metadata = _metadata(row)
        record = parse_record(metadata.get('deferral')) if isinstance(metadata, dict) else None
        current = None if record is None or isinstance(record, InvalidRecord) else record.stamped_at
        if row['status'] == 'deferred' and current is None and arguments.get('expected_stamped_at') is None:
            return None
        return {
            'success': False,
            'error': DeferralRefusalCode.CHANGED.value,
            'task_id': str(task_id),
            'expected_stamped_at': arguments.get('expected_stamped_at'),
            'current_stamped_at': None if current is None else current.isoformat(),
            'current_status': row['status'],
        }


def _invalid(task_id: int, reason: DeferralInvalidReason, field: str, value: object) -> dict[str, Any]:
    return {
        'success': False,
        'error': DeferralRefusalCode.INVALID.value,
        'task_id': str(task_id),
        'reason_code': reason.value,
        'field': field,
        'value': value,
    }
