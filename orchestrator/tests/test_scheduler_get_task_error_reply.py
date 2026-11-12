"""Scheduler.get_task must never hand back a fused-memory error reply as a task row.

Every fused-memory get_task failure arrives as an ordinary tool reply carrying
``{'error': str, 'error_type': str}``, never as a transport error. These tests
drive a real ``Scheduler`` through its public ``mcp_session=`` seam with a
session double that scripts the get_task reply.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from _orch_helpers import HermeticMcpSession
from shared.task_read import TaskAbsent, TaskFound, TaskReadFailed

from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import Scheduler

# Deleted task: sqlite_task_backend.get_task raises TaskNotFoundError, which
# fused-memory/src/fused_memory/server/tool_errors.py::mcp_tool_errors converts.
TASK_NOT_FOUND = {'error': 'Task 999 not found', 'error_type': 'TaskNotFoundError'}

# A synthetic escalation anchor id that is not numeric.
INVALID_TASK_ID = {
    'error': "INVALID_TASK_ID: non-numeric task id: 'infra'",
    'error_type': 'TaskmasterError',
}

# fused-memory/src/fused_memory/server/main.py::_safe_call_tool backstop.
SERVER_BACKSTOP = {'error': 'handler escaped', 'error_type': 'ToolError'}

# fused-memory/src/fused_memory/server/tools.py::_normalize_project_root.
BAD_PROJECT_ROOT = {
    'error': 'project_root is not a git checkout',
    'error_type': 'ValidationError',
}

WELL_FORMED_ROW = {
    'id': '42',
    'title': 'T',
    'status': 'pending',
    'metadata': '{"files": ["a.py"]}',
}


class _GetTaskReplySession(HermeticMcpSession):
    def __init__(self, get_task_result: dict) -> None:
        super().__init__()
        self._get_task_result = get_task_result
        self.get_task_ids: list[str] = []

    async def call_tool(
        self,
        name: str,
        arguments: dict,
        timeout: float = 30,
    ) -> dict:
        if name != 'get_task':
            return await super().call_tool(name, arguments, timeout=timeout)
        self.get_task_ids.append(arguments['id'])
        return {
            'jsonrpc': '2.0',
            'id': self._next_id(),
            'result': self._get_task_result,
        }


def _text_block(payload: dict) -> dict:
    return {'type': 'text', 'text': json.dumps(payload)}


def _text_only(payload: dict) -> dict:
    return {'content': [_text_block(payload)]}


def _structured_and_text(payload: dict) -> dict:
    return {'structuredContent': payload, 'content': [_text_block(payload)]}


def _is_error(payload: dict) -> dict:
    return {'isError': True, 'content': [_text_block(payload)]}


def _scheduler(tmp_path: Path, session: _GetTaskReplySession) -> Scheduler:
    return Scheduler(OrchestratorConfig(project_root=tmp_path), mcp_session=session)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'envelope',
    [_text_only, _structured_and_text, _is_error],
    ids=['text-only', 'structured-content', 'is-error'],
)
@pytest.mark.parametrize(
    ('task_id', 'payload'),
    [
        ('999', TASK_NOT_FOUND),
        ('infra', INVALID_TASK_ID),
        ('7', SERVER_BACKSTOP),
        ('8', BAD_PROJECT_ROOT),
    ],
    ids=['task-not-found', 'invalid-task-id', 'server-backstop', 'bad-project-root'],
)
async def test_get_task_answers_none_for_any_error_reply(
    tmp_path: Path, envelope, task_id: str, payload: dict,
) -> None:
    session = _GetTaskReplySession(envelope(payload))
    scheduler = _scheduler(tmp_path, session)

    assert await scheduler.get_task(task_id) is None
    assert session.get_task_ids == [task_id]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'envelope',
    [_text_only, _structured_and_text],
    ids=['text-only', 'structured-content'],
)
async def test_task_row_is_returned_with_metadata_normalised(
    tmp_path: Path, envelope,
) -> None:
    session = _GetTaskReplySession(envelope(dict(WELL_FORMED_ROW)))
    scheduler = _scheduler(tmp_path, session)

    task = await scheduler.get_task('42')

    assert task is not None
    assert (task['id'], task['title'], task['status']) == ('42', 'T', 'pending')
    assert task['metadata'] == {'files': ['a.py']}


@pytest.mark.asyncio
async def test_data_wrapped_task_row_is_unwrapped(tmp_path: Path) -> None:
    row = {'id': '42', 'status': 'done', 'metadata': {}}
    session = _GetTaskReplySession(_text_only({'data': row}))
    scheduler = _scheduler(tmp_path, session)

    assert await scheduler.get_task('42') == row


def _warnings_from(caplog: pytest.LogCaptureFixture, *loggers: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.WARNING and r.name in loggers]


@pytest.mark.asyncio
async def test_error_reply_logs_one_warning_naming_task_and_error_type(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger='orchestrator.scheduler')
    session = _GetTaskReplySession(_text_only(INVALID_TASK_ID))
    scheduler = _scheduler(tmp_path, session)

    assert await scheduler.get_task('infra') is None

    records = _warnings_from(caplog, 'orchestrator.scheduler')
    assert len(records) == 1, [r.getMessage() for r in records]
    message = records[0].getMessage()
    assert 'infra' in message
    assert 'TaskmasterError' in message


@pytest.mark.asyncio
async def test_task_row_logs_no_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger='orchestrator.scheduler')
    caplog.set_level(logging.WARNING, logger='shared.mcp_envelope')
    session = _GetTaskReplySession(_text_only(WELL_FORMED_ROW))
    scheduler = _scheduler(tmp_path, session)

    assert await scheduler.get_task('42') is not None

    assert _warnings_from(caplog, 'orchestrator.scheduler', 'shared.mcp_envelope') == []


class _UnreachableSession(HermeticMcpSession):
    async def call_tool(
        self,
        name: str,
        arguments: dict,
        timeout: float = 30,
    ) -> dict:
        raise RuntimeError('fused-memory unreachable')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'envelope',
    [_text_only, _structured_and_text],
    ids=['text-only', 'structured-content'],
)
async def test_read_task_reports_task_not_found_as_absence(
    tmp_path: Path, envelope,
) -> None:
    session = _GetTaskReplySession(envelope(TASK_NOT_FOUND))
    scheduler = _scheduler(tmp_path, session)

    assert await scheduler.read_task('999') == TaskAbsent()
    assert session.get_task_ids == ['999']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('envelope', 'task_id', 'payload'),
    [
        (envelope, task_id, payload)
        for envelope in (_text_only, _structured_and_text, _is_error)
        for task_id, payload in (
            ('infra', INVALID_TASK_ID),
            ('7', SERVER_BACKSTOP),
            ('8', BAD_PROJECT_ROOT),
        )
    ] + [(_is_error, '999', TASK_NOT_FOUND)],
    ids=[
        f'{envelope_id}-{payload_id}'
        for envelope_id in ('text-only', 'structured-content', 'is-error')
        for payload_id in ('invalid-task-id', 'server-backstop', 'bad-project-root')
    ] + ['is-error-task-not-found'],
)
async def test_read_task_reports_any_other_error_reply_as_a_failed_read(
    tmp_path: Path, envelope, task_id: str, payload: dict,
) -> None:
    session = _GetTaskReplySession(envelope(payload))
    scheduler = _scheduler(tmp_path, session)

    assert isinstance(await scheduler.read_task(task_id), TaskReadFailed)


@pytest.mark.asyncio
async def test_read_task_finds_a_row_with_metadata_normalised(tmp_path: Path) -> None:
    session = _GetTaskReplySession(_text_only(dict(WELL_FORMED_ROW)))
    scheduler = _scheduler(tmp_path, session)

    read = await scheduler.read_task('42')

    assert isinstance(read, TaskFound)
    assert read.task['metadata'] == {'files': ['a.py']}


@pytest.mark.asyncio
async def test_read_task_normalises_a_copy_and_leaves_the_reply_as_sent(tmp_path: Path) -> None:
    reply = _structured_and_text(dict(WELL_FORMED_ROW))
    scheduler = _scheduler(tmp_path, _GetTaskReplySession(reply))

    read = await scheduler.read_task('42')

    assert isinstance(read, TaskFound)
    assert read.task['metadata'] == {'files': ['a.py']}
    assert reply['structuredContent']['metadata'] == '{"files": ["a.py"]}'


@pytest.mark.asyncio
async def test_read_task_reports_a_dispatch_fault_as_a_failed_read(tmp_path: Path) -> None:
    scheduler = Scheduler(
        OrchestratorConfig(project_root=tmp_path), mcp_session=_UnreachableSession(),
    )

    read = await scheduler.read_task('42')

    assert isinstance(read, TaskReadFailed)
    assert 'RuntimeError' in read.reason


@pytest.mark.asyncio
async def test_read_task_failed_read_logs_one_warning_naming_task_and_error_type(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger='orchestrator.scheduler')
    session = _GetTaskReplySession(_text_only(INVALID_TASK_ID))
    scheduler = _scheduler(tmp_path, session)

    assert isinstance(await scheduler.read_task('infra'), TaskReadFailed)

    records = _warnings_from(caplog, 'orchestrator.scheduler')
    assert len(records) == 1, [r.getMessage() for r in records]
    message = records[0].getMessage()
    assert 'infra' in message
    assert 'TaskmasterError' in message


@pytest.mark.asyncio
async def test_read_task_absence_logs_no_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger='orchestrator.scheduler')
    session = _GetTaskReplySession(_text_only(TASK_NOT_FOUND))
    scheduler = _scheduler(tmp_path, session)

    assert await scheduler.read_task('999') == TaskAbsent()

    assert _warnings_from(caplog, 'orchestrator.scheduler') == []
