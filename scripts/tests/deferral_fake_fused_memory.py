"""An in-process fused-memory MCP server for the deferral scripts' suites.

A plain module rather than a conftest fixture, the same convention as
``legibility_task_store_fake``: conftest appends this directory to
``sys.path``, so the test files import it by bare name. ``FakeFusedMemory.post``
has the ``httpx.post`` signature ``legibility.census_trigger.post_mcp_envelope``
calls, so a test installs it with ``install_fake_httpx(fake.post)``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

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
    A root not in *projects*, or in ``unreadable``, answers an error payload."""

    def __init__(self, projects: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        self.projects: dict[str, list[dict[str, Any]]] = {
            root: [copy.deepcopy(dict(row)) for row in rows] for root, rows in projects.items()
        }
        self.unreadable: set[str] = set()
        self.calls: list[tuple[str, dict[str, Any]]] = []

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
