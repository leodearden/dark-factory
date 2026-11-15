"""Boundary rows for the gate-episode stamps at the status choke point (task 3638).

A pure gate's ``gate_escalated_at`` and ``gate_escalated_preconditions`` describe
its OPEN gate episode. Each row drives the public MCP ``set_task_status`` tool
against a real SQLite store, a real write journal and a real event buffer;
nothing private is patched.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import FusedMemoryConfig, TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.stage1_stall_detector import (
    extract_stalled_gate_backlog_task_ids,
)
from fused_memory.reconciliation.targeted import TargetedReconciler
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.write_journal import WriteJournal

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

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


class RecordingReconciler(TargetedReconciler):
    """A targeted reconciler that records each request instead of reconciling."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def reconcile_task(
        self,
        task_id: str,
        transition: str,
        project_id: str,
        task_before: dict,
        project_root: str,
        reopen_reason: str | None = None,
    ) -> dict:
        self.calls.append({'task_id': task_id, 'transition': transition})
        return {}


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


def _git(root: Path, *args: str) -> None:
    subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)


def _git_main_checkout(root: Path) -> str:
    """A real main checkout, so the tool layer's project_root resolution runs unpatched."""
    root.mkdir()
    _git(root, 'init', '-q', '-b', 'main')
    _git(root, 'config', 'user.email', 'gate-stamp@test')
    _git(root, 'config', 'user.name', 'gate-stamp')
    (root / 'README').write_text('gate stamp fixture\n')
    _git(root, 'add', 'README')
    _git(root, 'commit', '-q', '-m', 'seed')
    return str(root.resolve())


@pytest_asyncio.fixture
async def stack(tmp_path: Path) -> AsyncIterator[Stack]:
    root = _git_main_checkout(tmp_path / 'proj')
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=root))
    await backend.start()
    event_buffer = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
    await event_buffer.initialize()
    journal = WriteJournal(tmp_path / 'wj')
    await journal.initialize()
    ticket_store = TicketStore(tmp_path / 'tickets.db')
    await ticket_store.initialize()
    interceptor = TaskInterceptor(
        backend, RecordingReconciler(), event_buffer,
        ticket_store=ticket_store, config=FusedMemoryConfig(),
    )
    interceptor.set_write_journal(journal)
    server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
    try:
        yield Stack(root=root, backend=backend, interceptor=interceptor, server=server)
    finally:
        await interceptor.close()
        await journal.close()
        await event_buffer.close()
        await backend.close()


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
