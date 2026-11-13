"""The running server's task-metadata capability report, and its refusal half.

PRD plans/completion-driven-triggers-prd.md C2 "Capability report", row 16:
``get_status`` lists ``TASK_METADATA_CAPABILITIES`` at top level, and a
``before_done`` key this server does not declare is refused with
``UnsupportedTaskMetadata`` before anything is persisted.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _fastmcp_tool_result import tool_result_dict
from _hermetic_task_stack import hermetic_task_stack
from shared.task_metadata import TASK_METADATA_CAPABILITIES

from fused_memory.server.tools import create_mcp_server

pytestmark = pytest.mark.asyncio


def _status_service(status: dict) -> AsyncMock:
    service = AsyncMock()
    service.get_status = AsyncMock(return_value=status)
    return service


async def _get_status(status: dict) -> dict:
    server = create_mcp_server(_status_service(status))
    return tool_result_dict(await server._tool_manager.call_tool('get_status', {}))


async def test_get_status_reports_the_capabilities_at_top_level() -> None:
    result = await _get_status({'graphiti': {'connected': True}, 'mem0': {'connected': True}})
    assert result['task_metadata_capabilities'] == sorted(TASK_METADATA_CAPABILITIES)


async def test_an_error_status_carries_no_capabilities() -> None:
    result = await _get_status({'error': 'memory service unavailable'})
    assert 'task_metadata_capabilities' not in result


async def test_an_unknown_before_done_key_is_refused_at_submit(tmp_path: Path) -> None:
    script = tmp_path / 'run.sh'
    script.write_text('#!/bin/sh\nexit 0\n')
    os.chmod(script, 0o755)
    title = 'Predicate with a key from a newer recipe'
    async with hermetic_task_stack(tmp_path) as (server, _interceptor):
        refused = tool_result_dict(await server._tool_manager.call_tool('submit_task', {
            'project_root': str(tmp_path),
            'title': title,
            'description': 'A predicate gate whose before_done names recheck_mins.',
            'priority': 'medium',
            'planning_mode': True,
            'task_kind': 'deterministic',
            'metadata': {
                'milestone': {'mode': 'dated', 'at': '2026-11-03T00:00:00+00:00'},
                'before_done': {
                    'kind': 'predicate', 'script': 'run.sh', 'timeout_secs': 60,
                    'recheck_mins': 86400,
                },
            },
        }))
        listing = tool_result_dict(await server._tool_manager.call_tool(
            'get_tasks', {'project_root': str(tmp_path)},
        ))

    assert refused.get('error_type') == 'UnsupportedTaskMetadata', refused
    assert 'recheck_mins' in refused['error']
    assert refused['task_metadata_capabilities'] == sorted(TASK_METADATA_CAPABILITIES)
    assert title not in [task['title'] for task in listing['tasks']]
