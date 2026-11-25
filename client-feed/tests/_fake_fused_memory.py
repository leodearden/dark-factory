"""A stateful in-memory fused-memory answering the defect filer's nine closed tools in the real tools' reply shapes.

Defect entries fold through the real shared.defects rules, and submit_task's string arguments pass the real
markup-guard predicate, so the filer meets the refusals fused-memory would give it. Faults are injected per tool:
an error-dict reply, a raised McpCallError, or a SimulatedCrash before or after the nth call is applied.
"""

from __future__ import annotations

import copy
import json
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from _stores import TasksDb
from pydantic import ValidationError
from shared.defects import (
    Declare,
    DefectDeclaration,
    IllegalDefectTransition,
    MalformedDefectRef,
    apply_event,
    parse_declare_event,
    parse_outcome_event,
    parse_ref,
)
from shared.mcp_session import McpCallError, McpFailureKind
from shared.toolcall_markup import detect_for

from client_feed.sources.tasks import TASK_TAG

TERMINAL_STATUSES = frozenset({'done', 'cancelled'})
_LOOKUP_LIMIT_CAP = 100

Reply = dict[str, Any]


class SimulatedCrash(BaseException):
    """The filer's process dying at an injected point; a BaseException so no `except Exception` absorbs it."""


def _refused(code: str, detail: str, **fields: object) -> Reply:
    return {'success': False, 'error': code, **fields, 'detail': detail}


def _not_found(task_id: object) -> Reply:
    return {'error': f'Task {task_id} not found', 'error_type': 'TaskNotFound'}


def _declarations(entries: object) -> list[DefectDeclaration] | None:
    """A task's defect entries, or None when any does not read, as the real lookup leaves such a task out of its carriers."""
    if not isinstance(entries, list):
        return None
    try:
        return [DefectDeclaration.model_validate(entry) for entry in entries]
    except ValidationError:
        return None


class FakeFusedMemory:
    """fused-memory for one project root; `call_tool` is the in-process ToolCaller, `handle` the StubMcpServer handler."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = str(project_root)
        self._lock = threading.RLock()
        self._tasks: dict[int, dict[str, Any]] = {}
        self._next_id = 1
        self._calls: list[tuple[str, dict[str, Any]]] = []
        self._seen: Counter[str] = Counter()
        self._refusals: dict[str, list[Reply]] = {}
        self._failures: dict[str, list[McpCallError]] = {}
        self._crash_before: set[tuple[str, int]] = set()
        self._crash_after: set[tuple[str, int]] = set()
        self._tools: Mapping[str, Callable[[dict[str, Any]], Reply]] = {
            'submit_task': self._submit_task,
            'lookup_defect_ref': self._lookup_defect_ref,
            'find_tasks_by_metadata': self._find_tasks_by_metadata,
            'get_task': self._get_task,
            'add_dependency': self._add_dependency,
            'commit_planning': self._commit_planning,
            'declare_defect': self._declare_defect,
            'record_defect_outcome': self._record_defect_outcome,
            'set_task_status': self._set_task_status,
        }

    # --- what the tests read -------------------------------------------------

    @property
    def calls(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        """Every call as sent, (tool name, arguments), refused and crashed ones included."""
        with self._lock:
            return tuple((name, copy.deepcopy(arguments)) for name, arguments in self._calls)

    def task(self, task_id: int | str) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._tasks[int(task_id)])

    def tasks(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(copy.deepcopy(task) for _, task in sorted(self._tasks.items()))

    # --- fault injection and seeding ------------------------------------------

    def refuse_next(self, tool: str, reply: Reply) -> None:
        with self._lock:
            self._refusals.setdefault(tool, []).append(dict(reply))

    def fail_next(self, tool: str, error: McpCallError) -> None:
        with self._lock:
            self._failures.setdefault(tool, []).append(error)

    def crash_before(self, tool: str, nth: int) -> None:
        with self._lock:
            self._crash_before.add((tool, nth))

    def crash_after(self, tool: str, nth: int) -> None:
        with self._lock:
            self._crash_after.add((tool, nth))

    def seed_task(
        self,
        *,
        title: str = 'seeded',
        status: str = 'pending',
        metadata: Mapping[str, Any] | None = None,
        dependencies: Iterable[int] = (),
        task_kind: str = 'normal',
        priority: str = 'medium',
    ) -> int:
        """A task some other writer filed: a decoy, or a hand-filed carrier."""
        with self._lock:
            return self._create(
                title=title,
                description='',
                details=None,
                status=status,
                metadata=dict(metadata or {}),
                dependencies=list(dependencies),
                task_kind=task_kind,
                priority=priority,
            )

    def set_status(self, task_id: int | str, status: str) -> None:
        """A status another actor gave a task (a worker started it, a merge landed it); no call is recorded."""
        with self._lock:
            self._tasks[int(task_id)]['status'] = status

    def set_metadata(self, task_id: int | str, key: str, value: object) -> None:
        """Metadata another writer gave a task (a hand edit, a schema change); no call is recorded."""
        with self._lock:
            self._tasks[int(task_id)]['metadata'][key] = copy.deepcopy(value)

    def write_tasks_db(self, tasks: TasksDb, *, updated_at: str) -> None:
        """Every task held, written into a real tasks.db in the project's tag, so the publisher's readers see them."""
        for task in self.tasks():
            tasks.add(
                task['id'],
                title=task['title'],
                status=task['status'],
                updated_at=updated_at,
                tag=TASK_TAG,
                description=task['description'],
                details=task['details'],
                metadata=task['metadata'],
                depends_on=task['dependencies'],
            )

    # --- the tool surface ------------------------------------------------------

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Reply:
        sent = json.loads(json.dumps(dict(arguments)))
        with self._lock:
            self._calls.append((name, copy.deepcopy(sent)))
            self._seen[name] += 1
            nth = self._seen[name]
            if (name, nth) in self._crash_before:
                raise SimulatedCrash(f'before {name} #{nth}')
            injected = self._injected(name)
            if injected is not None:
                return injected
            reply = self._dispatch(name, sent)
            if (name, nth) in self._crash_after:
                raise SimulatedCrash(f'after {name} #{nth}')
            return reply

    def handle(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """The StubMcpServer ToolHandler: whatever call_tool raises, the stub answers as an isError reply."""
        return self.call_tool(name, arguments)

    def _injected(self, name: str) -> Reply | None:
        failures = self._failures.get(name)
        if failures:
            raise failures.pop(0)
        refusals = self._refusals.get(name)
        if refusals:
            return refusals.pop(0)
        return None

    def _dispatch(self, name: str, arguments: dict[str, Any]) -> Reply:
        tool = self._tools.get(name)
        if tool is None:
            raise McpCallError(McpFailureKind.TOOL_ERROR, f'Unknown tool: {name}')
        assert arguments.get('project_root') == self.project_root, (
            f'{name} sent project_root {arguments.get("project_root")!r}, not {self.project_root!r}'
        )
        return tool(arguments)

    # --- tools -----------------------------------------------------------------

    def _create(self, **fields: Any) -> int:
        task_id = self._next_id
        self._next_id += 1
        self._tasks[task_id] = {'id': task_id, **fields}
        return task_id

    def _held(self, task_id: object) -> dict[str, Any] | None:
        try:
            return self._tasks.get(int(str(task_id)))
        except ValueError:
            return None

    def _guard_markup(self, arguments: Mapping[str, Any]) -> None:
        for param, value in arguments.items():
            literal = detect_for(value, param)
            if literal is not None:
                raise McpCallError(
                    McpFailureKind.TOOL_ERROR,
                    f'Error calling tool submit_task: {param} carries MCP markup {literal!r}',
                )

    def _submit_task(self, arguments: dict[str, Any]) -> Reply:
        assert arguments.get('planning_mode') is True, 'the filer submits only under a planning hold'
        self._guard_markup(arguments)
        metadata = arguments.get('metadata') or {}
        assert isinstance(metadata, dict), f'metadata is sent as a mapping, got {type(metadata).__name__}'
        try:
            defects = [
                Declare(declaration=entry).declaration.model_dump(mode='json')
                for entry in metadata.get('defects', [])
            ]
        except ValidationError as exc:
            return _refused('defect_creation_refused', str(exc))
        task_id = self._create(
            title=arguments.get('title'),
            description=arguments.get('description'),
            details=arguments.get('details'),
            status='deferred',
            metadata={**metadata, **({'defects': defects} if defects else {})},
            dependencies=[],
            task_kind=arguments.get('task_kind', 'normal'),
            priority=arguments.get('priority'),
        )
        return {'task_id': str(task_id), 'status': 'deferred', 'planning_mode': True}

    def _lookup_defect_ref(self, arguments: dict[str, Any]) -> Reply:
        ref = arguments['ref']
        try:
            parse_ref(ref)
        except MalformedDefectRef as exc:
            return {'error': str(exc), 'error_type': 'ValidationError'}
        carriers: list[Reply] = []
        unreadable: list[int] = []
        for task_id, task in sorted(self._tasks.items()):
            if 'defects' not in task['metadata']:
                continue
            declarations = _declarations(task['metadata']['defects'])
            if declarations is None:
                unreadable.append(task_id)
                continue
            carriers.extend(
                {'task_id': task_id, 'task_status': task['status'], 'defect_status': declaration.status.value}
                for declaration in declarations
                if declaration.ref == ref
            )
        return {
            'project_root': self.project_root,
            'ref': ref,
            'live': bool(carriers),
            'carriers': carriers,
            'unreadable_task_ids': unreadable,
        }

    def _find_tasks_by_metadata(self, arguments: dict[str, Any]) -> Reply:
        key, value = arguments['key'], arguments.get('value')
        limit = min(int(arguments.get('limit', 50)), _LOOKUP_LIMIT_CAP)
        offset = int(arguments.get('offset', 0))

        def matches(task: Mapping[str, Any]) -> bool:
            if key not in task['metadata']:
                return False
            held = task['metadata'][key]
            return value is None or held == value or (isinstance(held, list) and value in held)

        found = [task for _, task in sorted(self._tasks.items()) if matches(task)]
        page = found[offset : offset + limit]
        return {
            'project_root': self.project_root,
            'key': key,
            'value': value,
            'results': [
                {
                    'task_id': task['id'],
                    'title': task['title'],
                    'status': task['status'],
                    'priority': task['priority'],
                    'updated_at': '2026-11-25T00:00:00+00:00',
                }
                for task in page
            ],
            'tasks_with_key': len(found),
            'pagination': {
                'total': len(found),
                'offset': offset,
                'page_size': limit,
                'returned': len(page),
                'has_more': offset + len(page) < len(found),
            },
        }

    def _get_task(self, arguments: dict[str, Any]) -> Reply:
        task = self._held(arguments['id'])
        if task is None:
            return _not_found(arguments['id'])
        return {**copy.deepcopy(task), 'project_root': self.project_root}

    def _add_dependency(self, arguments: dict[str, Any]) -> Reply:
        task, depends_on = self._held(arguments['id']), self._held(arguments['depends_on'])
        if task is None or depends_on is None:
            return _not_found(arguments['id'] if task is None else arguments['depends_on'])
        if depends_on['id'] not in task['dependencies']:
            task['dependencies'].append(depends_on['id'])
        return {'success': True, 'task_id': task['id'], 'depends_on': depends_on['id']}

    def _commit_planning(self, arguments: dict[str, Any]) -> Reply:
        results: list[dict[str, Any]] = []
        for text in (part.strip() for part in arguments['task_ids'].split(',')):
            task = self._held(text)
            if task is None:
                results.append({'task_id': text, 'result': _not_found(text)})
                continue
            if task['status'] == 'deferred':
                task['status'] = 'pending'
            results.append({'task_id': text, 'result': {'success': True, 'status': task['status']}})
        return {
            'success': all('error' not in item['result'] for item in results),
            'results': results,
        }

    def _fold(self, task_id: object, event: Any) -> Reply:
        task = self._held(task_id)
        if task is None:
            return _not_found(task_id)
        if task['status'] in TERMINAL_STATUSES:
            return _refused(
                'defect_write_to_terminal_task',
                f'task {task_id} is {task["status"]}',
                task_id=str(task_id),
            )
        entries: list[dict[str, Any]] = task['metadata'].setdefault('defects', [])
        index = next((i for i, entry in enumerate(entries) if entry.get('ref') == event.ref), None)
        current = None if index is None else DefectDeclaration.model_validate(entries[index])
        try:
            folded = apply_event(current, event).model_dump(mode='json')
        except IllegalDefectTransition as exc:
            return _refused('defect_transition_refused', str(exc), task_id=str(task_id))
        if index is None:
            entries.append(folded)
        else:
            entries[index] = folded
        return {'success': True, 'task_id': str(task_id), 'ref': event.ref, 'status': folded['status']}

    def _declare_defect(self, arguments: dict[str, Any]) -> Reply:
        try:
            event = parse_declare_event(arguments['event'])
        except ValidationError as exc:
            return _refused('defect_event_invalid', str(exc), task_id=arguments['task_id'])
        return self._fold(arguments['task_id'], event)

    def _record_defect_outcome(self, arguments: dict[str, Any]) -> Reply:
        try:
            event = parse_outcome_event(arguments['event'])
        except ValidationError as exc:
            return _refused('defect_event_invalid', str(exc), task_id=arguments['task_id'])
        if event.ref != arguments['ref']:
            return _refused(
                'defect_ref_mismatch',
                f'the call names ref {arguments["ref"]} but its event concerns {event.ref}',
                task_id=arguments['task_id'],
            )
        return self._fold(arguments['task_id'], event)

    def _set_task_status(self, arguments: dict[str, Any]) -> Reply:
        task = self._held(arguments['id'])
        if task is None:
            return _not_found(arguments['id'])
        task['status'] = arguments['status']
        return {'success': True, 'task_id': str(task['id']), 'status': task['status']}
