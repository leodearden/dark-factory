#!/usr/bin/env python3
"""A stdlib stand-in for fused-memory's MCP server over a JSON task store, for the review-all fixture.

It answers ``initialize``, ``notifications/initialized`` and ``tools/call`` for
``find_tasks_by_metadata`` and ``get_task`` over ``--store`` (re-read on every
call), with the server's lookup, paging and byte-budget semantics and task
7178's ``include_rows`` widening. It prints its endpoint URL as the first
stdout line, then serves until SIGTERM or SIGINT.

Flags, one per axis: ``--no-include-rows`` (the pre-widening server),
``--reject-unknown-args`` (undeclared arguments refused, not dropped),
``--byte-budget N``, ``--limit-cap N`` (the clamp on ``limit``; 0 is a server
that reports ``has_more`` with no rows), ``--drop-row-key KEY``,
``--find-error TEXT`` (every lookup answers that tool error), ``--sse``
(event-stream framing) and ``--journal PATH`` (one JSON line per request).
"""
from __future__ import annotations

import argparse
import functools
import importlib
import json
import re
import signal
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_SHARED_SRC = str(Path(__file__).resolve().parents[5] / 'shared' / 'src')
if _SHARED_SRC not in sys.path:
    sys.path.insert(0, _SHARED_SRC)
_TASK_STATUSES = frozenset(s.value for s in importlib.import_module('shared.task_statuses').TaskStatus)

ENDPOINT_PATH = '/mcp'
LIMIT_CAP = 100
DEFAULT_LIMIT = 50
DEFAULT_BYTE_BUDGET = 48_000
_KEY_PATTERN = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_-]*')
_ROW_FIELDS = ('task_id', 'title', 'status', 'priority', 'updated_at')
_FIND_PARAMS = frozenset({
    'project_root', 'key', 'value', 'statuses', 'limit', 'offset', 'tag', 'include_value',
})
_GET_TASK_PARAMS = frozenset({'id', 'project_root', 'tag'})
_REQUIRED_PARAMS: Mapping[str, frozenset[str]] = {
    'find_tasks_by_metadata': frozenset({'project_root', 'key'}),
    'get_task': frozenset({'id', 'project_root'}),
}


@dataclass(frozen=True)
class Behaviour:
    store: Path
    journal: Path | None
    include_rows: bool
    reject_unknown_args: bool
    byte_budget: int
    limit_cap: int
    drop_row_key: str | None
    find_error: str | None
    sse: bool


class ToolValidationError(Exception):
    """A tools/call the server refuses before the tool body runs (FastMCP's isError result)."""


def _validation_error(message: str) -> dict[str, Any]:
    return {'error': message, 'error_type': 'ValidationError'}


def _json_bytes(value: Any) -> int:
    """Serialized size as ``tools.py::_json_bytes`` measures it."""
    return len(json.dumps(value).encode('utf-8'))


def _load_tasks(behaviour: Behaviour) -> list[dict[str, Any]]:
    return sorted(json.loads(behaviour.store.read_text())['tasks'], key=lambda task: task['id'])


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _paging_error(limit: Any, offset: Any) -> dict[str, Any] | None:
    if not _is_int(limit) or limit < 1:
        return _validation_error(f'limit must be an integer >= 1; got {limit!r}')
    if not _is_int(offset) or offset < 0:
        return _validation_error(f'offset must be an integer >= 0; got {offset!r}')
    return None


def _statuses_error(statuses: Any) -> dict[str, Any] | None:
    if statuses is None:
        return None
    if not isinstance(statuses, list) or not all(isinstance(s, str) for s in statuses):
        return _validation_error(f'statuses must be a list of status strings; got {statuses!r}')
    unknown = sorted(set(statuses) - _TASK_STATUSES)
    if unknown:
        return _validation_error(f'unknown statuses {unknown}; valid: {sorted(_TASK_STATUSES)}')
    return None


def _query_error(arguments: Mapping[str, Any], behaviour: Behaviour) -> dict[str, Any] | None:
    """``backends/metadata_lookup.py::validate_metadata_query``'s rules."""
    key, value = arguments.get('key'), arguments.get('value')
    if not isinstance(key, str) or not _KEY_PATTERN.fullmatch(key):
        return _validation_error(f'key must match {_KEY_PATTERN.pattern}; got {key!r}')
    if value is not None and not (isinstance(value, str) or _is_int(value)):
        return _validation_error(f'value must be a str or int; got {value!r}')
    flags = ['include_value'] + (['include_rows'] if behaviour.include_rows else [])
    for flag in flags:
        if not isinstance(arguments.get(flag, False), bool):
            return _validation_error(f'{flag} must be a bool; got {arguments[flag]!r}')
    return None


def _matches(stored: Any, value: Any) -> bool:
    """Scalar equality or list membership, type-strict: 40 does not match "40"."""
    candidates = stored if isinstance(stored, list) else [stored]
    return any(type(item) is type(value) and item == value for item in candidates)


def _row(task: Mapping[str, Any], key: str, arguments: Mapping[str, Any],
         behaviour: Behaviour) -> dict[str, Any]:
    row = {
        'task_id': task['id'], 'title': task['title'], 'status': task['status'],
        'priority': task['priority'], 'updated_at': task['updatedAt'],
    }
    if arguments.get('include_value'):
        row['matched_value'] = task['metadata'][key]
    if behaviour.include_rows and arguments.get('include_rows'):
        row['metadata'] = task['metadata']
        row['dependencies'] = sorted(task.get('dependencies', []))
        row.pop(behaviour.drop_row_key or '', None)
    return row


def _elided(row: Mapping[str, Any]) -> dict[str, Any]:
    elided = dict(row)
    if 'metadata' in row:
        elided['metadata'] = {'_elided': True, 'bytes': _json_bytes(row['metadata'])}
        elided['truncated_row'] = True
    if 'matched_value' in row:
        elided['matched_value'] = {'_elided': True, 'bytes': _json_bytes(row['matched_value'])}
        elided['truncated_value'] = True
    return elided


def _fit_rows_to_budget(rows: Sequence[Mapping[str, Any]], budget: int) -> list[Mapping[str, Any]]:
    """``tools.py::_fit_rows_to_budget``: the longest prefix that fits; a lone big row is elided."""
    size = len('[]')
    for count, row in enumerate(rows):
        size += _json_bytes(row) + (len(', ') if count else 0)
        if size > budget:
            return list(rows[:count]) if count else [_elided(row)]
    return list(rows)


def _pagination(total: int, offset: int, page_size: int, returned: int) -> dict[str, Any]:
    """``tools.py::_pagination_meta``."""
    return {
        'total': total, 'offset': offset, 'page_size': page_size, 'returned': returned,
        'has_more': offset + returned < total,
    }


def find_tasks_by_metadata(arguments: Mapping[str, Any], behaviour: Behaviour) -> dict[str, Any]:
    if behaviour.find_error is not None:
        return _validation_error(behaviour.find_error)
    limit, offset = arguments.get('limit', DEFAULT_LIMIT), arguments.get('offset', 0)
    statuses = arguments.get('statuses')
    error = (_paging_error(limit, offset) or _statuses_error(statuses)
             or _query_error(arguments, behaviour))
    if error is not None:
        return error
    key, value = arguments['key'], arguments.get('value')
    searched = [t for t in _load_tasks(behaviour) if statuses is None or t['status'] in statuses]
    carrying = [t for t in searched if key in t['metadata']]
    matching = [t for t in carrying if value is None or _matches(t['metadata'][key], value)]
    page_size = min(limit, behaviour.limit_cap)
    rows = [_row(t, key, arguments, behaviour) for t in matching[offset:offset + page_size]]
    if arguments.get('include_value') or (behaviour.include_rows and arguments.get('include_rows')):
        rows = _fit_rows_to_budget(rows, behaviour.byte_budget)
    reply: dict[str, Any] = {
        'project_id': Path(arguments['project_root']).name,
        'project_root': arguments['project_root'],
        'key': key, 'value': value, 'statuses': statuses,
        'results': rows,
        'tasks_with_key': len(carrying),
        'pagination': _pagination(len(matching), offset, page_size, len(rows)),
    }
    if behaviour.include_rows:
        reply['include_rows'] = arguments.get('include_rows', False)
    return reply


def get_task(arguments: Mapping[str, Any], behaviour: Behaviour) -> dict[str, Any]:
    wanted = str(arguments['id'])
    for task in _load_tasks(behaviour):
        if str(task['id']) == wanted:
            return {
                'id': task['id'], 'title': task['title'],
                'description': task.get('description', ''), 'details': '', 'testStrategy': '',
                'status': task['status'], 'dependencies': task.get('dependencies', []),
                'priority': task['priority'], 'subtasks': [], 'updatedAt': task['updatedAt'],
                'metadata': task['metadata'], 'project_root': arguments['project_root'],
            }
    return {
        'error': f'TASKMASTER_TOOL_ERROR: No tasks found for ID(s): {wanted}',
        'error_type': 'TaskNotFoundError',
    }


_Tool = Callable[[Mapping[str, Any], Behaviour], dict[str, Any]]
_TOOLS: Mapping[str, _Tool] = {
    'find_tasks_by_metadata': find_tasks_by_metadata,
    'get_task': get_task,
}


def _declared_params(tool: str, behaviour: Behaviour) -> frozenset[str]:
    if tool == 'get_task':
        return _GET_TASK_PARAMS
    return _FIND_PARAMS | ({'include_rows'} if behaviour.include_rows else set())


def _accepted_arguments(tool: str, arguments: Mapping[str, Any], behaviour: Behaviour) -> dict:
    """Arguments as FastMCP's argument model sees them: undeclared ones dropped (extra='ignore')."""
    declared = _declared_params(tool, behaviour)
    problems = [f'{name}\n  Extra inputs are not permitted [type=extra_forbidden]'
                for name in sorted(set(arguments) - declared) if behaviour.reject_unknown_args]
    problems += [f'{name}\n  Field required [type=missing]'
                 for name in sorted(_REQUIRED_PARAMS[tool] - set(arguments))]
    if problems:
        raise ToolValidationError(
            f'Error executing tool {tool}: {len(problems)} validation error(s) for '
            f'{tool}Arguments\n' + '\n'.join(problems)
        )
    return {name: value for name, value in arguments.items() if name in declared}


def _tool_result(params: Mapping[str, Any], behaviour: Behaviour) -> dict[str, Any]:
    name = params.get('name')
    tool = _TOOLS.get(name) if isinstance(name, str) else None
    if tool is None:
        return {'content': [{'type': 'text', 'text': f'Unknown tool: {name}'}], 'isError': True}
    try:
        arguments = _accepted_arguments(name, params.get('arguments') or {}, behaviour)
    except ToolValidationError as refused:
        return {'content': [{'type': 'text', 'text': str(refused)}], 'isError': True}
    payload = tool(arguments, behaviour)
    return {
        'content': [{'type': 'text', 'text': json.dumps(payload, indent=2)}],
        'structuredContent': payload,
        'isError': False,
    }


def _initialize_result(params: Mapping[str, Any], behaviour: Behaviour) -> dict[str, Any]:
    return {
        'protocolVersion': params.get('protocolVersion', '2024-11-05'),
        'capabilities': {'tools': {'listChanged': False}},
        'serverInfo': {'name': 'review-all-fake-store', 'version': '1.0'},
    }


_METHODS: Mapping[str, Callable[[Mapping[str, Any], Behaviour], dict[str, Any]]] = {
    'initialize': _initialize_result,
    'tools/call': _tool_result,
}
_NOTIFICATIONS = frozenset({'notifications/initialized'})
_JOURNAL_LOCK = threading.Lock()
_JOURNALED_HEADERS = ('accept', 'content-type', 'mcp-session-id')


class McpHandler(BaseHTTPRequestHandler):
    def __init__(self, *args: Any, behaviour: Behaviour, **kwargs: Any) -> None:
        self.behaviour = behaviour
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get('Content-Length') or 0))
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        self._journal(body)
        if self.path != ENDPOINT_PATH:
            self._send(404, b'Not Found', 'text/plain')
        elif not isinstance(body, dict):
            self._reply({'jsonrpc': '2.0', 'id': None,
                         'error': {'code': -32700, 'message': 'Parse error'}}, status=400)
        elif body.get('method') in _NOTIFICATIONS:
            self._send(202, b'', None)
        else:
            self._reply(self._dispatch(body))

    def _dispatch(self, body: Mapping[str, Any]) -> dict[str, Any]:
        name = body.get('method')
        method = _METHODS.get(name) if isinstance(name, str) else None
        if method is None:
            error = {'code': -32601, 'message': f'Method not found: {name}'}
            return {'jsonrpc': '2.0', 'id': body.get('id'), 'error': error}
        result = method(body.get('params') or {}, self.behaviour)
        return {'jsonrpc': '2.0', 'id': body.get('id'), 'result': result}

    def _journal(self, body: Any) -> None:
        if self.behaviour.journal is None:
            return
        entry = {
            'path': self.path,
            'headers': {name: self.headers.get(name) for name in _JOURNALED_HEADERS},
            'body': body,
        }
        with _JOURNAL_LOCK, self.behaviour.journal.open('a') as journal:
            journal.write(json.dumps(entry) + '\n')

    def _reply(self, message: Mapping[str, Any], status: int = 200) -> None:
        if self.behaviour.sse:
            frame = f'event: message\ndata: {json.dumps(message)}\n\n'
            self._send(status, frame.encode(), 'text/event-stream')
        else:
            self._send(status, json.dumps(message).encode(), 'application/json')

    def _send(self, status: int, payload: bytes, content_type: str | None) -> None:
        self.send_response(status)
        if content_type is not None:
            self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--journal', type=Path)
    parser.add_argument('--no-include-rows', action='store_true')
    parser.add_argument('--reject-unknown-args', action='store_true')
    parser.add_argument('--byte-budget', type=int, default=DEFAULT_BYTE_BUDGET)
    parser.add_argument('--limit-cap', type=int, default=LIMIT_CAP)
    parser.add_argument('--drop-row-key')
    parser.add_argument('--find-error')
    parser.add_argument('--sse', action='store_true')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    behaviour = Behaviour(
        store=args.store.resolve(), journal=args.journal, include_rows=not args.no_include_rows,
        reject_unknown_args=args.reject_unknown_args, byte_budget=args.byte_budget,
        limit_cap=args.limit_cap, drop_row_key=args.drop_row_key, find_error=args.find_error,
        sse=args.sse,
    )
    server = ThreadingHTTPServer(
        (args.host, args.port), functools.partial(McpHandler, behaviour=behaviour),
    )
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    host, port = server.server_address[:2]
    print(f'http://{host}:{port}{ENDPOINT_PATH}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
