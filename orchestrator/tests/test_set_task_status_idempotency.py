"""Scheduler.set_task_status replays one client_op_id across its own retries.

The shared ``McpSession`` annotates a copy of each call's arguments, so a key
it mints lives for one logical call. The scheduler retries ABOVE the session,
and each of those attempts is a new logical call; only a key the scheduler
supplies itself survives them. Real sockets, the real shared client, and the
same ``fm_retry_backoffs`` seam the sibling ``TestSetTaskStatusForwarding``
tests shorten.
"""

from __future__ import annotations

import pytest
from shared.mcp_session import McpSession
from shared.testing_mcp_stub import StubMcpServer

from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import Scheduler

_TRANSIENT_REJECTION = {
    'error': "TimeoutError('ensure_connected timed out')",
    'error_type': 'TimeoutError',
}


def _scheduler(server: StubMcpServer) -> Scheduler:
    session = McpSession(server.url, client_name='orchestrator')
    return Scheduler(OrchestratorConfig(max_per_module=1), mcp_session=session)


def _shorten_outer_retries(monkeypatch: pytest.MonkeyPatch, backoffs: list[float]) -> None:
    monkeypatch.setattr('orchestrator.scheduler.fm_retry_backoffs', lambda *a, **kw: list(backoffs))


def _tool_call_op_ids(server: StubMcpServer) -> list[str | None]:
    return [
        r.body['params']['arguments'].get('client_op_id')
        for r in server.requests
        if isinstance(r.body, dict) and r.body.get('method') == 'tools/call'
    ]


@pytest.mark.asyncio
async def test_an_outer_retry_after_transport_exhaustion_replays_the_same_key(monkeypatch):
    _shorten_outer_retries(monkeypatch, [0.0])
    with StubMcpServer(stateful=False, unavailable_tool_calls=1) as server:
        await _scheduler(server).set_task_status('7', 'in-progress')
        op_ids = _tool_call_op_ids(server)

    assert len(op_ids) == 2
    assert None not in op_ids
    assert op_ids[0] == op_ids[1]


@pytest.mark.asyncio
async def test_outer_retries_on_a_transient_rejection_replay_the_same_key(monkeypatch):
    _shorten_outer_retries(monkeypatch, [0.0, 0.0])
    with StubMcpServer(stateful=False, payload=_TRANSIENT_REJECTION) as server:
        with pytest.raises(RuntimeError, match='3 transient retries'):
            await _scheduler(server).set_task_status('7', 'in-progress')
        op_ids = _tool_call_op_ids(server)

    assert len(op_ids) == 3
    assert len(set(op_ids)) == 1
    assert None not in op_ids
