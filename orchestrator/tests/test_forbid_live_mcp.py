"""Guard tests for conftest.py::forbid_live_mcp (tasks 2644, 7802): a live
fused-memory MCP round-trip fails the test at once, even through a retrying caller."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from shared.mcp_session import McpSession

import orchestrator.mcp_lifecycle
from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import Scheduler

_PROMPT_SECS = 5
_REAL_INITIALIZE = McpSession.initialize
_REAL_CALL = McpSession.call


@pytest.fixture
def bare_scheduler(tmp_path, monkeypatch) -> Scheduler:
    monkeypatch.setattr(orchestrator.mcp_lifecycle, '_session', None)
    return Scheduler(OrchestratorConfig(project_root=tmp_path))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('tool', 'call'),
    [
        pytest.param(
            'set_task_status', lambda s: s.set_task_status('1', 'done'), id='set_task_status',
        ),
        pytest.param('get_tasks', lambda s: s.get_tasks(), id='get_tasks'),
    ],
)
async def test_a_live_round_trip_fails_the_test_promptly_naming_tool_and_url(
    forbid_live_mcp: list[tuple[str, str]],
    bare_scheduler: Scheduler,
    tool: str,
    call: Callable[[Scheduler], Awaitable[object]],
) -> None:
    with pytest.raises(pytest.fail.Exception) as excinfo:
        async with asyncio.timeout(_PROMPT_SECS):
            await call(bare_scheduler)

    message = str(excinfo.value)
    assert tool in message
    assert bare_scheduler.config.fused_memory.url in message
    assert len(forbid_live_mcp) == 1
    forbid_live_mcp.clear()


@pytest.mark.asyncio
async def test_initialize_is_refused_naming_the_url(
    forbid_live_mcp: list[tuple[str, str]], tmp_path,
) -> None:
    url = OrchestratorConfig(project_root=tmp_path).fused_memory.url
    with pytest.raises(pytest.fail.Exception) as excinfo:
        async with asyncio.timeout(_PROMPT_SECS):
            await McpSession(url, client_name='orchestrator').initialize()

    message = str(excinfo.value)
    assert 'initialize' in message
    assert url in message
    assert len(forbid_live_mcp) == 1
    forbid_live_mcp.clear()


def test_every_test_is_guarded_without_requesting_the_fixture() -> None:
    assert McpSession.initialize is not _REAL_INITIALIZE
    assert McpSession.call is not _REAL_CALL


@pytest.mark.real_mcp_session
def test_real_mcp_session_marker_leaves_mcp_session_unpatched() -> None:
    assert McpSession.initialize is _REAL_INITIALIZE
    assert McpSession.call is _REAL_CALL
