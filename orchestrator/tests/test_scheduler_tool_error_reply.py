"""A fused-memory tool reply carrying ``isError`` is a rejection, never a landed write.

FastMCP answers a pydantic ValidationError, or any ToolError raised outside a
tool body, with ``isError: True`` and plain prose rather than a JSON payload.
These tests drive a real ``Scheduler`` through its public ``mcp_session=``
seam with a session double that scripts the set_task_status and
set_task_claimant replies.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _orch_helpers import HermeticMcpSession

from orchestrator.config import OrchestratorConfig
from orchestrator.pause_policy import PauseClass
from orchestrator.scheduler import (
    DoneGateRejection,
    Scheduler,
    SchedulerCallbacks,
    SetTaskStatusRejected,
    extract_rejection,
    extract_structured_rejection,
    is_transient_rejection,
)

SCHEDULER_LOGGER = 'orchestrator.scheduler'

# Measured FastMCP reply to a set_task_status call whose status failed validation.
VALIDATION_ERROR_TEXT = (
    'Error executing tool set_task_status: 1 validation error for '
    'set_task_statusArguments\nstatus\n  Input should be a valid string '
    '[type=string_type, input_value=None, input_type=NoneType]'
)

TIMEOUT_ERROR_TEXT = (
    "Error executing tool set_task_status: TimeoutError('ensure_connected timed out')"
)

DONE_GATE_PAYLOAD = {'error': 'done_gate_missing_files', 'missing_files': ['a.py']}

# A whole JSON-RPC body with an error member and no result.
JSONRPC_ERROR_BODY = {
    'jsonrpc': '2.0',
    'id': 9,
    'error': {'code': -32602, 'message': 'Invalid params: unknown tool set_task_status'},
}


class _ScriptedReplySession(HermeticMcpSession):
    """Answers scripted tools from a per-tool queue and records each call it serves.

    A scripted entry is either a CallToolResult, wrapped into a JSON-RPC body,
    or a whole JSON-RPC body (it carries ``jsonrpc``), returned verbatim. A call
    past the end of a tool's script fails the test at once: ``pytest.fail`` is a
    BaseException, so a retrying ``except Exception`` caller cannot absorb it.
    """

    def __init__(self, replies: dict[str, list[dict]]) -> None:
        super().__init__()
        self._replies = {name: list(queue) for name, queue in replies.items()}
        self.calls: list[tuple[str, dict]] = []

    def calls_to(self, name: str) -> list[dict]:
        return [arguments for called, arguments in self.calls if called == name]

    async def call_tool(
        self,
        name: str,
        arguments: dict,
        timeout: float = 30,
    ) -> dict:
        if name not in self._replies:
            return await super().call_tool(name, arguments, timeout=timeout)
        self.calls.append((name, arguments))
        if not self._replies[name]:
            pytest.fail(f'unexpected extra {name} call: {arguments!r}')
        reply = self._replies[name].pop(0)
        if 'jsonrpc' in reply:
            return reply
        return {'jsonrpc': '2.0', 'id': self._next_id(), 'result': reply}


def _text_block(text: str) -> dict:
    return {'type': 'text', 'text': text}


def _is_error_text(text: str) -> dict:
    return {'isError': True, 'content': [_text_block(text)]}


def _is_error_json(payload: dict) -> dict:
    return _is_error_text(json.dumps(payload))


def _clean(payload: dict) -> dict:
    return {'content': [_text_block(json.dumps(payload))]}


def _body(result: dict) -> dict:
    return {'jsonrpc': '2.0', 'id': 3, 'result': result}


def _scheduler(
    tmp_path: Path,
    session: _ScriptedReplySession,
    *,
    on_park_stop_trip: AsyncMock | None = None,
) -> Scheduler:
    return Scheduler(
        OrchestratorConfig(project_root=tmp_path, park_stop_parked_threshold=1),
        mcp_session=session,
        callbacks=SchedulerCallbacks(on_park_stop_trip=on_park_stop_trip),
    )


def _scheduler_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == SCHEDULER_LOGGER
    ]


# --- The reply parsers -------------------------------------------------------


def test_plain_text_is_error_is_a_rejection_carrying_its_prose() -> None:
    body = _body(_is_error_text(VALIDATION_ERROR_TEXT))

    rendered = extract_rejection(body)

    assert rendered is not None
    assert 'IS_ERROR' in rendered
    assert '1 validation error' in rendered
    assert extract_structured_rejection(body) == {
        'error': 'IS_ERROR', 'hint': VALIDATION_ERROR_TEXT,
    }


@pytest.mark.parametrize(
    'result',
    [
        {'isError': True, 'content': []},
        _is_error_json({'message': 'nope'}),
    ],
    ids=['no-text-block', 'json-without-error-key'],
)
def test_is_error_without_a_structured_error_is_a_rejection(result: dict) -> None:
    body = _body(result)

    assert extract_rejection(body) is not None
    structured = extract_structured_rejection(body)
    assert structured is not None
    assert structured['error'] == 'IS_ERROR'


def test_json_rpc_error_body_is_a_rejection_naming_code_and_message() -> None:
    rendered = extract_rejection(JSONRPC_ERROR_BODY)

    assert rendered is not None
    assert 'JSON_RPC_ERROR' in rendered
    assert '-32602' in rendered
    assert 'Invalid params' in rendered
    structured = extract_structured_rejection(JSONRPC_ERROR_BODY)
    assert structured is not None
    assert structured['error'] == 'JSON_RPC_ERROR'


def test_structured_error_code_survives_an_is_error_reply() -> None:
    body = _body(_is_error_json(DONE_GATE_PAYLOAD))

    assert extract_structured_rejection(body) == DONE_GATE_PAYLOAD


@pytest.mark.parametrize(
    ('text', 'transient'),
    [(TIMEOUT_ERROR_TEXT, True), (VALIDATION_ERROR_TEXT, False)],
    ids=['timeout-prose', 'validation-prose'],
)
def test_is_error_prose_classifies_transience_by_its_text(text: str, transient: bool) -> None:
    rejection = extract_rejection(_body(_is_error_text(text)))

    assert is_transient_rejection(rejection) is transient


@pytest.mark.parametrize(
    'response',
    [
        {},
        None,
        {'result': _clean({'id': '42', 'status': 'pending'})},
        {'result': {'isError': False, **_clean({'id': '42', 'status': 'pending'})}},
    ],
    ids=['empty-dict', 'none', 'clean-body', 'is-error-false'],
)
def test_landed_write_is_not_a_rejection(response: dict | None) -> None:
    assert extract_rejection(response) is None
    assert extract_structured_rejection(response) is None


# --- Scheduler.set_task_status -------------------------------------------------


@pytest.mark.asyncio
async def test_is_error_blocked_write_raises_and_is_not_counted_as_parked(
    tmp_path: Path,
) -> None:
    session = _ScriptedReplySession(
        {'set_task_status': [_is_error_text(VALIDATION_ERROR_TEXT)]},
    )
    on_trip = AsyncMock()
    scheduler = _scheduler(tmp_path, session, on_park_stop_trip=on_trip)

    with pytest.raises(SetTaskStatusRejected) as excinfo:
        await scheduler.set_task_status('42', 'blocked')

    assert excinfo.value.error_code == 'IS_ERROR'
    assert '1 validation error' in excinfo.value.raw
    assert len(session.calls_to('set_task_status')) == 1
    assert scheduler.parked_window_churn_count == 0
    assert scheduler.is_paused is False
    on_trip.assert_not_called()


@pytest.mark.asyncio
async def test_json_rpc_error_blocked_write_raises_and_is_not_counted_as_parked(
    tmp_path: Path,
) -> None:
    session = _ScriptedReplySession({'set_task_status': [JSONRPC_ERROR_BODY]})
    on_trip = AsyncMock()
    scheduler = _scheduler(tmp_path, session, on_park_stop_trip=on_trip)

    with pytest.raises(SetTaskStatusRejected) as excinfo:
        await scheduler.set_task_status('42', 'blocked')

    assert excinfo.value.error_code == 'JSON_RPC_ERROR'
    assert len(session.calls_to('set_task_status')) == 1
    assert scheduler.parked_window_churn_count == 0
    assert scheduler.is_paused is False
    on_trip.assert_not_called()


@pytest.mark.asyncio
async def test_landed_blocked_write_is_counted_and_trips_park_stop(tmp_path: Path) -> None:
    session = _ScriptedReplySession(
        {'set_task_status': [_clean({'id': '42', 'status': 'blocked'})]},
    )
    scheduler = _scheduler(tmp_path, session, on_park_stop_trip=AsyncMock())

    await scheduler.set_task_status('42', 'blocked')

    assert scheduler.parked_window_churn_count == 1
    assert scheduler.is_paused is True
    assert scheduler.pause_class is PauseClass.PARK_STOP


@pytest.mark.asyncio
async def test_structured_done_gate_under_is_error_raises_done_gate_rejection(
    tmp_path: Path,
) -> None:
    session = _ScriptedReplySession(
        {'set_task_status': [_is_error_json(DONE_GATE_PAYLOAD)]},
    )
    scheduler = _scheduler(tmp_path, session)

    with pytest.raises(DoneGateRejection) as excinfo:
        await scheduler.set_task_status('42', 'pending')

    assert excinfo.value.missing_files == ['a.py']


# --- Scheduler.set_task_claimant -----------------------------------------------


@pytest.mark.asyncio
async def test_is_error_claimant_write_logs_one_rejection_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=SCHEDULER_LOGGER)
    session = _ScriptedReplySession(
        {'set_task_claimant': [_is_error_text(VALIDATION_ERROR_TEXT)]},
    )
    scheduler = _scheduler(tmp_path, session)

    await scheduler.set_task_claimant('42', heartbeat_at='2026-11-07T00:00:00+00:00')

    warnings = _scheduler_warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert '42' in message
    assert '1 validation error' in message


@pytest.mark.asyncio
async def test_json_rpc_error_claimant_clear_logs_one_rejection_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=SCHEDULER_LOGGER)
    session = _ScriptedReplySession({'set_task_claimant': [JSONRPC_ERROR_BODY]})
    scheduler = _scheduler(tmp_path, session)

    await scheduler.set_task_claimant('42', claimant_run_id=None, heartbeat_at=None)

    warnings = _scheduler_warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert '42' in message
    assert 'Invalid params' in message


@pytest.mark.asyncio
async def test_landed_claimant_write_logs_no_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=SCHEDULER_LOGGER)
    session = _ScriptedReplySession({'set_task_claimant': [_clean({'id': '42'})]})
    scheduler = _scheduler(tmp_path, session)

    await scheduler.set_task_claimant('42', heartbeat_at='2026-11-07T00:00:00+00:00')

    assert _scheduler_warnings(caplog) == []


# --- Scheduler.update_task -----------------------------------------------------


@pytest.mark.asyncio
async def test_json_rpc_error_metadata_update_fails_and_logs_code_and_message(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR, logger=SCHEDULER_LOGGER)
    session = _ScriptedReplySession({'update_task': [JSONRPC_ERROR_BODY]})
    scheduler = _scheduler(tmp_path, session)

    updated = await scheduler.update_task('42', {'k': 'v'})

    assert updated is False
    assert len(session.calls_to('update_task')) == 1
    message = ' '.join(r.getMessage() for r in _scheduler_warnings(caplog))
    assert '-32602' in message
    assert 'Invalid params' in message
