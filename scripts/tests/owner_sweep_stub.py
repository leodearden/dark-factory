"""An in-process double of both MCP servers the INV-12 owner sweep calls, and a harness to run it.

``StubMcp`` answers at ``scripts/exception_owner_sweep.py``'s ``ToolCall`` port
signature, over a real temp tasks.db that it writes and the sweep only reads.
The harness runs one scan-and-act over it.  A plain module, imported by bare
name: ``scripts/tests/conftest.py`` puts this directory on ``sys.path``.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path

from _task_db_scan import decode_metadata, tasks_db_path
from exception_owner_sweep import ActOutcome, Ports, RegisterView, Tally, act, scan
from fused_memory.middleware.candidate_key import compute_candidate_key

NOW = datetime.datetime(2026, 10, 17, 6, 0, tzinfo=datetime.UTC)

#: Where the stub's curator inserts the tasks it creates: clear of every seeded id.
FIRST_CREATED_ID = 9001


def insert_task(db: Path, task_id: int, metadata: Mapping[str, object], *, title: str | None = None) -> None:
    """A pending master-tag task, as a server-side writer would create it."""
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO tasks (tag, id, title, status, metadata, updated_at) "
            "VALUES ('master', ?, ?, 'pending', ?, '2026-10-17T06:00:00+00:00')",
            (task_id, title or f'task {task_id}', json.dumps(dict(metadata))),
        )


class StubMcp:
    """Both MCP servers, in process, at the sweep's ToolCall port signature.

    Records every call as ``(server, tool, args)``.  On the fused-memory side it
    plays the curator: a submitted candidate resolves per ``submit_outcome``,
    and a created task is INSERTED into the temp store with the submitted
    metadata.  ``'pending'`` answers the unpersisted ``timeout`` of a curator
    that has not finished, so the task may still appear after the sweep has
    stopped waiting.  Like the store's ``(tag, candidate_key)`` index, which
    covers done rows, it folds a created submission into any non-cancelled row
    of the same key.  The stub writes; the sweep only reads.

    On the escalation side it keeps the archive: every ``escalate_info`` is a
    record in ``filed`` with the real ``esc-<task_id>-<seq>`` id, and
    ``get_task_escalations`` answers from it at every level and status.
    :meth:`route_info_notes` plays the info-L0 router: it converts a note into
    a task titled by the note's summary, which folds onto any non-cancelled
    row of the same candidate key, as a submission does.
    """

    def __init__(
        self,
        db: Path,
        *,
        owner_tickets: Mapping[str, dict] | None = None,
        submit_outcome: str = 'created',
        combine_into: int | None = None,
        fail_on: frozenset[str] = frozenset(),
    ) -> None:
        self.db = db
        self.owner_tickets = dict(owner_tickets or {})
        self.submit_outcome = submit_outcome
        self.combine_into = combine_into
        self.fail_on = fail_on
        self.calls: list[tuple[str, str, dict]] = []
        self.pending: list[dict] = []
        self.filed: list[dict] = []
        self.submitted: dict[str, dict] = {}
        self.next_task_id = FIRST_CREATED_ID

    def tool_calls(self, tool: str) -> list[dict]:
        return [args for _, called, args in self.calls if called == tool]

    def _record(self, server: str, tool: str, args: Mapping[str, object]) -> None:
        self.calls.append((server, tool, dict(args)))
        if tool in self.fail_on:
            raise ConnectionError(f'{tool}: connection refused')

    def fused_memory(self, tool: str, args: Mapping[str, object]) -> Mapping[str, object]:
        self._record('fused-memory', tool, args)
        if tool == 'submit_task':
            if self.submit_outcome == 'error':
                return {'error': 'curator queue closed', 'error_type': 'RuntimeError'}
            ticket = f'tkt_S{len(self.submitted) + 1}'
            self.submitted[ticket] = dict(args)
            return {'ticket': ticket}
        if tool == 'resolve_ticket':
            ticket = str(args['ticket'])
            if ticket in self.owner_tickets:
                return self.owner_tickets[ticket]
            return self._resolve_submitted(self.submitted[ticket])
        if tool == 'update_task':
            return self._update(int(str(args['id'])), args['metadata'])
        raise AssertionError(f'unexpected fused-memory tool {tool}')

    def _colliding_row(self, args: dict) -> int | None:
        key = compute_candidate_key(args['title'], args['metadata'].get('files'))
        if key is None:
            return None
        with sqlite3.connect(self.db) as conn:
            rows = conn.execute(
                "SELECT id, title, metadata FROM tasks WHERE tag = 'master' AND status != 'cancelled'"
            ).fetchall()
        return next(
            (
                row_id
                for row_id, title, raw in rows
                if compute_candidate_key(title, decode_metadata(raw).get('files')) == key
            ),
            None,
        )

    def _resolve_submitted(self, args: dict) -> dict:
        if self.submit_outcome == 'created':
            colliding = self._colliding_row(args)
            if colliding is not None:
                return {'status': 'combined', 'task_id': str(colliding), 'reason': 'candidate_key_collision'}
            task_id = self.next_task_id
            self.next_task_id += 1
            insert_task(self.db, task_id, args['metadata'], title=args['title'])
            return {'status': 'created', 'task_id': str(task_id)}
        if self.submit_outcome == 'combined':
            return {'status': 'combined', 'task_id': str(self.combine_into)}
        if self.submit_outcome == 'refused':
            return {'status': 'refused', 'reason': 'premise_cancelled'}
        if self.submit_outcome == 'pending':
            return {'status': 'failed', 'reason': 'timeout', 'task_id': None}
        raise AssertionError(f'unexpected submit_outcome {self.submit_outcome}')

    def _update(self, task_id: int, metadata: object) -> dict:
        assert isinstance(metadata, dict)
        with sqlite3.connect(self.db) as conn:
            (raw,) = conn.execute(
                "SELECT metadata FROM tasks WHERE tag = 'master' AND id = ?", (task_id,)
            ).fetchone()
            merged = {**json.loads(raw or '{}'), **metadata}
            conn.execute(
                "UPDATE tasks SET metadata = ? WHERE tag = 'master' AND id = ?",
                (json.dumps(merged), task_id),
            )
        return {'id': task_id, 'updated': True}

    def escalation(self, tool: str, args: Mapping[str, object]) -> Mapping[str, object]:
        self._record('escalation', tool, args)
        if tool == 'get_pending_escalations':
            return {'result': [{'task_id': entry['task_id']} for entry in self.pending]}
        if tool == 'get_task_escalations':
            fields = ('id', 'task_id', 'status', 'level')
            return {
                'result': [
                    {field: record[field] for field in fields}
                    for record in self.filed
                    if record['task_id'] == args['task_id']
                ]
            }
        if tool == 'escalate_info':
            record = self._file_record(dict(args), level=0, status='pending')
            self.pending.append(record)
            return {'id': record['id'], 'status': 'queued', 'level': 0}
        raise AssertionError(f'unexpected escalation tool {tool}')

    def _file_record(self, args: dict, *, level: int, status: str) -> dict:
        seq = 1 + sum(record['task_id'] == args['task_id'] for record in self.filed)
        record = {**args, 'id': f"esc-{args['task_id']}-{seq}", 'status': status, 'level': level}
        self.filed.append(record)
        return record

    def route_info_notes(self, *, convert_to: int | None = None, promote: bool = False) -> None:
        """Play ``orchestrator/src/orchestrator/escalation_router.py``: dismiss every pending note.

        ``promote`` first files a dismissed L1 under the same ``task_id`` and
        summary, the orphan-reaper / watcher shape that produced task 7693.
        ``convert_to`` converts the one pending note into a pending task with that id,
        titled by its summary and declaring no files
        (``escalation/src/escalation/disposition.py::_curator_candidate``), and
        carrying the escalation id it was converted from (the L1's when
        promoted).  Like the curator, the conversion folds onto any
        non-cancelled row of the same candidate key and then inserts nothing.
        None dismisses only, as when the curator combined the conversion
        elsewhere.
        """
        if convert_to is not None and len(self.pending) != 1:
            raise AssertionError(
                f'convert_to={convert_to} names one task, but {len(self.pending)} notes are pending'
            )
        for record in self.pending:
            record['status'] = 'dismissed'
            converted_from = record['id']
            if promote:
                converted_from = self._file_record(
                    {'task_id': record['task_id'], 'summary': record['summary']}, level=1, status='dismissed'
                )['id']
            if convert_to is not None:
                title = record['summary'].strip()
                metadata = {'escalation_id': converted_from, 'source': 'info-l0-router'}
                if self._colliding_row({'title': title, 'metadata': metadata}) is None:
                    insert_task(self.db, convert_to, metadata, title=title)
        self.pending = []


def ports_for(project_root: Path, stub: StubMcp) -> Ports:
    return Ports(
        project_root=project_root,
        db=tasks_db_path(project_root),
        fused_memory=stub.fused_memory,
        escalation=stub.escalation,
    )


def act_once(register: RegisterView, ports: Ports, now: datetime.datetime = NOW) -> Tally:
    return act(scan(register, ports, now=now), ports, now=now)


def sweep_once(register: RegisterView, ports: Ports, now: datetime.datetime = NOW) -> ActOutcome:
    return act_once(register, ports, now).counts()


def set_status(db: Path, task_id: int, status: str) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE tasks SET status = ? WHERE tag = 'master' AND id = ?", (status, task_id))
