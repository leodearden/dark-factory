"""Scheduler.get_task must never hand back a fused-memory error reply as a task row.

Every fused-memory get_task failure arrives as an ordinary tool reply carrying
``{'error': str, 'error_type': str}``, never as a transport error. These tests
drive a real ``Scheduler`` through its public ``mcp_session=`` seam with a
session double that scripts the get_task reply.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from _orch_helpers import HermeticMcpSession

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
async def test_error_reply_is_absence_not_a_task_row(
    tmp_path: Path, envelope, task_id: str, payload: dict,
) -> None:
    session = _GetTaskReplySession(envelope(payload))
    scheduler = _scheduler(tmp_path, session)

    assert await scheduler.get_task(task_id) is None
    assert session.get_task_ids == [task_id]


@pytest.mark.asyncio
async def test_task_row_is_returned_with_metadata_normalised(tmp_path: Path) -> None:
    session = _GetTaskReplySession(_text_only(WELL_FORMED_ROW))
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
