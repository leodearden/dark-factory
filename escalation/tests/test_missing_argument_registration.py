"""The escalation server refuses a missing required argument with a structured payload.

Task 5979 (reify #7913): a ``merge_request`` call without ``worktree`` came
back as raw pydantic text that never said whether anything had run. Every call
here goes through ``fastmcp.Client``, because ``tool.fn`` / ``tool.run``
bypass middleware.
"""
from __future__ import annotations

import asyncio
import json
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from shared.mcp_missing_arguments import MISSING_ARGUMENT_CODE, MissingArgumentMiddleware
from shared.toolcall_markup import INVOKE_CLOSER, closer_for

from escalation.queue import EscalationQueue
from escalation.server import create_server

try:
    from orchestrator.config import OrchestratorConfig  # type: ignore[reportMissingImports]
    from orchestrator.merge_queue import (  # type: ignore[reportMissingImports]
        InFlightMergeRegistry,
    )
    _ORCHESTRATOR_AVAILABLE = True
except ImportError:
    _ORCHESTRATOR_AVAILABLE = False
    OrchestratorConfig: Any = None  # type: ignore[assignment,misc]
    InFlightMergeRegistry: Any = None  # type: ignore[assignment,misc]


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed'
    ),
]


SIGHTING_CALL = {
    'task_id': '5099',
    'branch': '5099',
    'description': 'manual re-submission',
    'verified_green': False,
}


class Wired:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.queue = EscalationQueue(tmp_path / 'esc')
        self.mq: asyncio.Queue = asyncio.Queue()
        self.harness = types.SimpleNamespace(
            git_ops=types.SimpleNamespace(
                resolve_branch_sha=AsyncMock(return_value=None),
                is_ancestor=AsyncMock(return_value=False),
                find_inflight_merge_worktree=AsyncMock(return_value=None),
            ),
            scheduler=types.SimpleNamespace(get_task=AsyncMock(return_value={'metadata': {}})),
        )
        self.server = create_server(
            self.queue,
            merge_queue=self.mq,
            orch_config=OrchestratorConfig(project_root=tmp_path / 'repo'),
            harness=self.harness,
            merge_inflight_registry=InFlightMergeRegistry(),
            startup_sweep=False,
        )

    async def call(self, tool: str, arguments: dict[str, Any]):
        async with Client(self.server) as client:
            return await client.call_tool(tool, arguments)

    async def refusal(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with pytest.raises(ToolError) as excinfo:
            await asyncio.wait_for(self.call(tool, arguments), timeout=5.0)
        return json.loads(str(excinfo.value))


@pytest.fixture
def wired(tmp_path: Path) -> Wired:
    return Wired(tmp_path)


async def test_middleware_is_registered_exactly_once(wired):
    guards = [m for m in wired.server.middleware if isinstance(m, MissingArgumentMiddleware)]

    assert len(guards) == 1


async def test_merge_request_without_worktree_is_refused_structurally(wired):
    payload = await wired.refusal('merge_request', SIGHTING_CALL)

    assert payload['code'] == MISSING_ARGUMENT_CODE
    assert payload['tool'] == 'merge_request'
    assert [m['name'] for m in payload['missing']] == ['worktree']
    description = payload['missing'][0]['schema'].get('description')
    assert isinstance(description, str) and description
    assert 'worktree=' in payload['example_call']


async def test_refusal_happens_before_any_slow_path(wired):
    await wired.refusal('merge_request', SIGHTING_CALL)

    wired.harness.git_ops.resolve_branch_sha.assert_not_awaited()
    wired.harness.scheduler.get_task.assert_not_awaited()
    assert wired.mq.empty()


async def test_a_complete_merge_request_is_still_queued(wired):
    result = await wired.call(
        'merge_request',
        {**SIGHTING_CALL, 'worktree': str(wired.tmp_path / 'wt'), 'wait_secs': 0},
    )

    assert result.data['status'] == 'queued'
    assert wired.mq.qsize() == 1


def _description_that_swallowed(worktree: str) -> str:
    # '\x3c' spells '<': an envelope literal typed verbatim here would have to
    # cross the authoring agent's own tool call (shared/src/shared/toolcall_markup.py).
    return (
        SIGHTING_CALL['description']
        + closer_for('description')
        + '\n\x3cworktree>'
        + worktree
        + closer_for('worktree')
        + INVOKE_CLOSER
    )


async def test_a_worktree_swallowed_by_leaked_markup_is_repaired_not_refused(wired):
    worktree = str(wired.tmp_path / 'wt')
    leaked_call = {**SIGHTING_CALL, 'description': _description_that_swallowed(worktree)}

    result = await asyncio.wait_for(wired.call('merge_request', leaked_call), timeout=5.0)

    assert result.data['status'] == 'queued'
    assert wired.mq.get_nowait().worktree == Path(worktree)


async def test_coverage_is_server_wide(wired):
    payload = await wired.refusal(
        'escalate_info',
        {'task_id': '5979', 'agent_role': 'implementer', 'category': 'cleanup_needed'},
    )

    assert payload['code'] == MISSING_ARGUMENT_CODE
    assert payload['tool'] == 'escalate_info'
    assert [m['name'] for m in payload['missing']] == ['summary']
    assert wired.queue.get_by_task('5979') == []
