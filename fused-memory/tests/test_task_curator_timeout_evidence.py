"""TaskCurator LLM call sites: tool scoping and timeout evidence (task 3995).

Kept apart from ``test_task_curator.py`` so that file does not grow further;
the curator helpers are imported from it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from test_task_curator import _agent_result, _make_config

from fused_memory.middleware.task_curator import CandidateTask, TaskCurator

_INVOKE = 'fused_memory.middleware.task_curator.invoke_with_cap_retry'

_SINGLE_OK = {'action': 'create', 'justification': 'x'}
_BATCH_OK = {'decisions': [
    {'candidate_index': 0, 'action': 'create', 'justification': 'x0'},
    {'candidate_index': 1, 'action': 'create', 'justification': 'x1'},
]}


async def _call_single(curator: TaskCurator, invoke: AsyncMock) -> None:
    with patch(_INVOKE, new=invoke):
        await curator._call_llm(
            CandidateTask(title='T'),
            pool=[],
            pool_sizes={'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0},
            start=0.0,
            project_id='p',
            project_root='/p',
        )


async def _call_batch(curator: TaskCurator, invoke: AsyncMock) -> None:
    with patch(_INVOKE, new=invoke):
        await curator._call_llm_batch(
            [CandidateTask(title='T0'), CandidateTask(title='T1')],
            pools=[[], []],
            pool_sizes_list=[{}, {}],
            start=0.0,
            project_id='p',
            project_root='/p',
        )


CallSite = Callable[[TaskCurator, AsyncMock], Awaitable[None]]

_CALL_SITES = [
    pytest.param(_call_single, _SINGLE_OK, id='single'),
    pytest.param(_call_batch, _BATCH_OK, id='batch'),
]


async def _successful_call_kwargs(
    drive: CallSite, structured: dict[str, Any], curator: TaskCurator,
) -> dict[str, Any]:
    invoke = AsyncMock(return_value=_agent_result(structured))
    await drive(curator, invoke)
    return invoke.call_args.kwargs


class TestCuratorMcpScoping:
    """Both call sites scope MCP to zero servers, strictly.

    Neither of the other two guards covers MCP. The ``'*'`` → ``--tools ''``
    substitution filters built-in and deferred tools only. The neutral cwd
    removes only the project ``.mcp.json``: at a neutral cwd an account-scoped
    claude.ai connector still connected and exposed
    ``mcp__claude_ai_Claude_Docs__create`` / ``update`` / ``delete``
    (measured on CLI 2.1.283).
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('drive', 'structured'), _CALL_SITES)
    async def test_passes_zero_server_strict_mcp_config(self, drive, structured):
        curator = TaskCurator(config=_make_config(), taskmaster=None)

        kwargs = await _successful_call_kwargs(drive, structured, curator)

        assert kwargs.get('mcp_config') == {'mcpServers': {}}
        assert kwargs.get('strict_mcp_config') is True
