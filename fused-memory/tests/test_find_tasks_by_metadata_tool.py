"""find_tasks_by_metadata over a real SQLite task store.

Two faces of one lookup (plans/task-metadata-lookup-prd.md): the in-process
``TaskInterceptor.find_tasks_by_metadata`` seam and the MCP tool. Every case
arranges its store through the backend's public writes, or through the
``shared.project_layout.tasks_db`` file for states no writer produces.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from shared.project_layout import tasks_db
from shared.task_statuses import ACTIVE

from fused_memory.backends.metadata_lookup import InvalidMetadataLookupError
from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server

_OVER_BUDGET_BYTES = 48_000
_SLIM_ROW_KEYS = {'task_id', 'title', 'status', 'priority', 'updated_at'}


def _git_init_root(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(['git', 'init', '-q', str(path)], check=True, capture_output=True)
    return str(path.resolve())


def _write_garbage_store(root: str) -> None:
    db = tasks_db(root)
    db.parent.mkdir(parents=True)
    db.write_bytes(b'not a sqlite database' * 100)


@dataclass(frozen=True)
class LookupStack:
    root: str
    backend: SqliteTaskBackend
    interceptor: TaskInterceptor
    server: Any
    seed: Callable[..., Awaitable[int]]

    async def call(self, **args: Any) -> Any:
        return await self.server._tool_manager.call_tool(
            'find_tasks_by_metadata', {'project_root': self.root, **args},
        )

    def corrupt(self, task_id: int, raw: str) -> None:
        with sqlite3.connect(tasks_db(self.root)) as conn:
            conn.execute('UPDATE tasks SET metadata = ? WHERE id = ?', (raw, task_id))


@pytest_asyncio.fixture
async def lookup_stack(tmp_path: Path) -> AsyncIterator[LookupStack]:
    root = _git_init_root(tmp_path / 'project')
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=root))
    await backend.start()
    interceptor = TaskInterceptor(backend, None, EventBuffer(db_path=None))
    server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
    seeded = 0

    async def seed(status: str = 'pending', metadata: dict[str, Any] | None = None) -> int:
        nonlocal seeded
        seeded += 1
        added = await backend.add_task(
            project_root=root,
            title=f'task {seeded}',
            description=f'seeded lookup fixture task {seeded}',
            metadata=json.dumps(metadata) if metadata is not None else None,
            status=status,
        )
        return int(added['id'])

    try:
        yield LookupStack(root, backend, interceptor, server, seed)
    finally:
        await interceptor.close()
        await backend.close()


# ── In-process face: TaskInterceptor.find_tasks_by_metadata ─────────────


@pytest.mark.asyncio
async def test_presence_query_over_active_statuses_returns_whole_values(lookup_stack):
    chain_a = {'skill': 'review', 'run_id': 'r-1'}
    chain_b = {'skill': 'census', 'run_id': 'r-2', 'superseded_by': None}
    pending_id = await lookup_stack.seed('pending', {'trigger_chain': chain_a})
    deferred_id = await lookup_stack.seed('deferred', {'trigger_chain': chain_b})
    await lookup_stack.seed('done', {'trigger_chain': {'skill': 'review'}})

    found = await lookup_stack.interceptor.find_tasks_by_metadata(
        lookup_stack.root, 'trigger_chain', None,
        statuses=list(ACTIVE), include_value=True,
    )

    assert [row['task_id'] for row in found['results']] == [pending_id, deferred_id]
    assert all(isinstance(row['task_id'], int) for row in found['results'])
    assert [row['matched_value'] for row in found['results']] == [chain_a, chain_b]
    assert found['total'] == 2
    assert found['tasks_with_key'] == 2
    assert found['malformed_metadata_ids'] == []


@pytest.mark.asyncio
async def test_in_process_lookup_applies_no_byte_budget(lookup_stack):
    blob = 'a' * 60_000
    assert len(json.dumps(blob).encode('utf-8')) > _OVER_BUDGET_BYTES
    task_id = await lookup_stack.seed('pending', {'x_blob': blob})

    found = await lookup_stack.interceptor.find_tasks_by_metadata(
        lookup_stack.root, 'x_blob', include_value=True,
    )

    assert [row['task_id'] for row in found['results']] == [task_id]
    row = found['results'][0]
    assert row['matched_value'] == blob
    assert 'truncated_value' not in row


@pytest.mark.asyncio
async def test_rows_omit_the_value_unless_asked(lookup_stack):
    await lookup_stack.seed('pending', {'x_k': 'v'})

    found = await lookup_stack.interceptor.find_tasks_by_metadata(
        lookup_stack.root, 'x_k', 'v',
    )

    assert len(found['results']) == 1
    assert set(found['results'][0]) == _SLIM_ROW_KEYS


@pytest.mark.asyncio
async def test_in_process_paging_tiles_the_match_set(lookup_stack):
    ids = [await lookup_stack.seed('pending', {'x_k': 'v'}) for _ in range(5)]
    lookup = lookup_stack.interceptor.find_tasks_by_metadata

    first = await lookup(lookup_stack.root, 'x_k', 'v', limit=3)
    rest = await lookup(lookup_stack.root, 'x_k', 'v', limit=3, offset=3)
    beyond = await lookup(lookup_stack.root, 'x_k', 'v', limit=3, offset=10)

    assert [row['task_id'] for row in first['results']] == ids[:3]
    assert [row['task_id'] for row in rest['results']] == ids[3:]
    assert beyond['results'] == []
    assert first['total'] == rest['total'] == beyond['total'] == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('arg', 'overrides'),
    [
        ('key', {'key': 'a"b'}),
        ('value', {'value': True}),
        ('include_value', {'include_value': 'yes'}),
        ('limit', {'limit': 0}),
    ],
)
async def test_in_process_lookup_enforces_its_own_preconditions(lookup_stack, arg, overrides):
    call: dict[str, Any] = {'key': 'x_k', **overrides}
    offending = overrides[arg]

    with pytest.raises(InvalidMetadataLookupError) as raised:
        await lookup_stack.interceptor.find_tasks_by_metadata(lookup_stack.root, **call)

    assert isinstance(raised.value, ValueError)
    assert arg in str(raised.value)
    assert repr(offending) in str(raised.value)


@pytest.mark.asyncio
async def test_unreadable_store_raises_rather_than_reporting_no_match(lookup_stack, tmp_path):
    broken_root = _git_init_root(tmp_path / 'broken')
    _write_garbage_store(broken_root)

    with pytest.raises(sqlite3.DatabaseError):
        await lookup_stack.interceptor.find_tasks_by_metadata(broken_root, 'x_k')
