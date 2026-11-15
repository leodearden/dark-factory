"""Boundary rows for the gate-episode stamps at the status choke point (task 3638).

A deterministic task's ``gate_escalated_at`` and ``gate_escalated_preconditions``
describe its OPEN gate episode. Each row drives the public MCP ``set_task_status``
tool against ``_hermetic_task_stack``'s real SQLite store and event buffer.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from _hermetic_task_stack import hermetic_task_stack

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.stage1_stall_detector import (
    extract_stalled_gate_backlog_task_ids,
)

_STAMP_AT = '2026-08-02T19:07:41.617308+00:00'
_PRECONDITIONS = {'dependencies': [], 'external_deps': [], 'milestone': None}
_GATE_KEYS = ('gate_escalated_at', 'gate_escalated_preconditions')
_FAR_PAST_THE_STAMP = datetime(2030, 1, 1, tzinfo=UTC)


def _pure_gate_metadata(**stamps: Any) -> dict[str, Any]:
    return {
        'task_kind': 'deterministic',
        'always_escalates': True,
        'before_done': None,
        'source': 'gate-stamp-fixture',
        **stamps,
    }


_STAMPED = _pure_gate_metadata(
    gate_escalated_at=_STAMP_AT, gate_escalated_preconditions=_PRECONDITIONS,
)


@dataclass
class Stack:
    root: str
    backend: SqliteTaskBackend
    interceptor: TaskInterceptor
    server: Any

    async def call(self, tool: str, **args: Any) -> Any:
        return await self.server._tool_manager.call_tool(
            tool, {'project_root': self.root, **args},
        )

    async def seed(self, status: str, title: str, metadata: dict[str, Any]) -> str:
        created = await self.backend.add_task(
            project_root=self.root, title=title, status=status,
            metadata=json.dumps(metadata),
        )
        return str(created['id'])

    async def read(self, task_id: str) -> dict:
        return await self.interceptor.get_task(task_id, self.root)


@pytest_asyncio.fixture
async def stack(tmp_path: Path) -> AsyncIterator[Stack]:
    async with hermetic_task_stack(tmp_path) as (server, interceptor):
        backend = interceptor.taskmaster
        assert isinstance(backend, SqliteTaskBackend)
        yield Stack(root=str(tmp_path), backend=backend, interceptor=interceptor, server=server)


def _assert_no_gate_stamp(metadata: dict[str, Any]) -> None:
    for key in _GATE_KEYS:
        assert key not in metadata, (key, metadata)


def _gate_close_provenance(task_id: str) -> dict[str, Any]:
    return {
        'kind': 'deterministic-gate',
        'note': 'pure gate resolved',
        'escalation_id': f'esc-{task_id}-1',
    }


@pytest.mark.asyncio
async def test_runner_close_drops_both_stamps_and_keeps_the_rest(stack: Stack) -> None:
    task_id = await stack.seed('blocked', 'the 2864 close', _STAMPED)

    result = await stack.call(
        'set_task_status', id=task_id, status='done', agent_id='orchestrator',
        done_provenance=_gate_close_provenance(task_id),
    )

    assert 'error' not in result, result
    row = await stack.read(task_id)
    assert row['status'] == 'done'
    metadata = row['metadata']
    _assert_no_gate_stamp(metadata)
    assert metadata['done_provenance']['kind'] == 'deterministic-gate'
    assert metadata['done_provenance']['escalation_id'] == f'esc-{task_id}-1'
    assert metadata['task_kind'] == 'deterministic'
    assert metadata['always_escalates'] is True
    assert metadata['source'] == 'gate-stamp-fixture'


@pytest.mark.asyncio
async def test_cancel_drops_both_stamps(stack: Stack) -> None:
    task_id = await stack.seed('blocked', 'cancelled gate', _STAMPED)

    result = await stack.call('set_task_status', id=task_id, status='cancelled')

    assert 'error' not in result, result
    row = await stack.read(task_id)
    assert row['status'] == 'cancelled'
    _assert_no_gate_stamp(row['metadata'])
    assert row['metadata']['source'] == 'gate-stamp-fixture'


@pytest.mark.asyncio
async def test_reopening_a_legacy_stale_close_drops_its_stamp(stack: Stack) -> None:
    legacy = _pure_gate_metadata(gate_escalated_at=_STAMP_AT)
    task_id = await stack.seed('done', 'the pre-9520 3421 shape', legacy)

    result = await stack.call(
        'set_task_status', id=task_id, status='blocked',
        reopen_reason='gate reopened for a fresh human decision',
    )

    assert 'error' not in result, result
    row = await stack.read(task_id)
    assert row['status'] == 'blocked'
    _assert_no_gate_stamp(row['metadata'])
    assert extract_stalled_gate_backlog_task_ids([row], now=_FAR_PAST_THE_STAMP) == []


@pytest.mark.asyncio
async def test_reopening_a_done_deploy_drops_its_stamp(stack: Stack) -> None:
    deploy = {
        'task_kind': 'deterministic',
        'always_escalates': True,
        'before_done': {'kind': 'deploy', 'script': 'scripts/deploy.sh', 'timeout_secs': 60},
        'gate_escalated_at': _STAMP_AT,
        'source': 'gate-stamp-fixture',
    }
    task_id = await stack.seed('done', 'an act-then-ask deploy', deploy)

    result = await stack.call(
        'set_task_status', id=task_id, status='pending',
        reopen_reason='deploy reopened for a re-run',
    )

    assert 'error' not in result, result
    row = await stack.read(task_id)
    assert row['status'] == 'pending'
    _assert_no_gate_stamp(row['metadata'])
    assert row['metadata']['before_done'] == deploy['before_done']


@pytest.mark.asyncio
@pytest.mark.parametrize(('seeded', 'target'), [('blocked', 'pending'), ('pending', 'blocked')])
async def test_non_terminal_edges_keep_both_stamps(
    stack: Stack, seeded: str, target: str,
) -> None:
    task_id = await stack.seed(seeded, f'held gate {seeded}->{target}', _STAMPED)

    result = await stack.call('set_task_status', id=task_id, status=target)

    assert 'error' not in result, result
    row = await stack.read(task_id)
    assert row['status'] == target
    assert row['metadata']['gate_escalated_at'] == _STAMP_AT
    assert row['metadata']['gate_escalated_preconditions'] == _PRECONDITIONS


@pytest.mark.asyncio
async def test_unstamped_close_invents_no_key(stack: Stack) -> None:
    seeded = _pure_gate_metadata()
    task_id = await stack.seed('blocked', 'never stamped', seeded)

    result = await stack.call(
        'set_task_status', id=task_id, status='done', agent_id='orchestrator',
        done_provenance=_gate_close_provenance(task_id),
    )

    assert 'error' not in result, result
    metadata = (await stack.read(task_id))['metadata']
    _assert_no_gate_stamp(metadata)
    assert {key: metadata[key] for key in seeded} == seeded
    assert set(metadata) == {*seeded, 'done_provenance'}
