"""get_tasks(fields=...): the typed row projection of plans/dashboard-prd-lanes-prd.md §5.1.

Three faces of one projection: the pure ``task_projection`` module, the
in-process ``TaskInterceptor.get_tasks`` seam over a real SQLite store, and
the MCP tool.
"""

from __future__ import annotations

import contextlib
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

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.backends.task_projection import (
    ALLOWED_FIELDS,
    TASK_ROW_KEYS,
    InvalidTaskProjectionField,
    TaskProjection,
    parse_task_projection,
)
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server

_FULL_ROW: dict[str, Any] = {
    'id': '7',
    'title': 'a task',
    'description': 'what it does',
    'details': 'how',
    'testStrategy': 'check it',
    'status': 'pending',
    'dependencies': [3],
    'priority': 'medium',
    'subtasks': [],
    'updatedAt': '2026-10-22T00:00:00Z',
    'metadata': {'prd_path': 'plans/x.md', 'other': 1},
    'claimant_run_id': None,
    'heartbeat_at': None,
    'candidate_key': None,
}


def _project(fields: object, row: dict[str, Any] = _FULL_ROW) -> dict[str, Any]:
    projection = parse_task_projection(fields)
    assert isinstance(projection, TaskProjection)
    return projection.apply(row)


# ── Pure projection contract ────────────────────────────────────────────


def test_the_literal_row_carries_exactly_the_allow_listed_keys():
    assert tuple(_FULL_ROW) == TASK_ROW_KEYS


def test_none_means_no_projection():
    assert parse_task_projection(None) is None


def test_requested_metadata_keys_present_on_the_task_are_kept_and_absent_ones_omitted():
    projected = _project(['id', 'metadata.prd_path', 'metadata.spawned_from'])

    assert projected == {'id': '7', 'metadata': {'prd_path': 'plans/x.md'}}


def test_id_is_always_returned():
    assert _project(['status']) == {'id': '7', 'status': 'pending'}
    assert _project([]) == {'id': '7'}


def test_no_metadata_entry_means_no_metadata_key():
    assert 'metadata' not in _project(['title'])


def test_a_coerced_empty_metadata_projects_an_empty_dict():
    row = {**_FULL_ROW, 'metadata': {}}

    assert _project(['metadata.prd_path'], row) == {'id': '7', 'metadata': {}}


def test_whole_metadata_subsumes_its_sub_keys():
    projected = _project(['metadata', 'metadata.prd_path'])

    assert projected == {'id': '7', 'metadata': _FULL_ROW['metadata']}


def test_duplicate_entries_collapse():
    duplicated = _project(['id', 'id', 'metadata.prd_path', 'metadata.prd_path'])

    assert duplicated == _project(['id', 'metadata.prd_path'])


def test_every_row_key_is_accepted_from_a_list_or_a_tuple():
    assert _project(list(TASK_ROW_KEYS)) == _FULL_ROW
    assert _project(tuple(TASK_ROW_KEYS)) == _FULL_ROW


@pytest.mark.parametrize(
    'entry',
    ['foo', '', 'metadata.', 'metadata.a-b', 'metadata.a.b', 'metadata.1a', 5],
)
def test_an_invalid_entry_is_named_verbatim_with_the_allowed_set(entry):
    with pytest.raises(InvalidTaskProjectionField) as caught:
        parse_task_projection(['id', entry])

    assert caught.value.field == entry
    assert caught.value.allowed == ALLOWED_FIELDS
    assert repr(entry) in str(caught.value)


@pytest.mark.parametrize('argument', ['id', {'id': 1}])
def test_a_non_list_argument_is_named_whole(argument):
    with pytest.raises(InvalidTaskProjectionField) as caught:
        parse_task_projection(argument)

    assert caught.value.field == argument
    assert caught.value.allowed == ALLOWED_FIELDS


def test_the_allowed_set_is_every_row_key_plus_the_metadata_form():
    assert (*TASK_ROW_KEYS, 'metadata.<key>') == ALLOWED_FIELDS


# ── Real-store stack ────────────────────────────────────────────────────


def _git_init_root(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(['git', 'init', '-q', str(path)], check=True, capture_output=True)
    return str(path.resolve())


@dataclass(frozen=True)
class ProjectionStack:
    root: str
    backend: SqliteTaskBackend
    interceptor: TaskInterceptor
    server: Any
    seed: Callable[..., Awaitable[str]]

    async def call(self, **args: Any) -> Any:
        return await self.server._tool_manager.call_tool(
            'get_tasks', {'project_root': self.root, **args},
        )

    def corrupt(self, task_id: str, raw: str) -> None:
        with contextlib.closing(sqlite3.connect(tasks_db(self.root))) as conn, conn:
            conn.execute('UPDATE tasks SET metadata = ? WHERE id = ?', (raw, int(task_id)))


@pytest_asyncio.fixture
async def projection_stack(tmp_path: Path) -> AsyncIterator[ProjectionStack]:
    root = _git_init_root(tmp_path / 'project')
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=root))
    await backend.start()
    interceptor = TaskInterceptor(backend, None, EventBuffer(db_path=None))
    server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
    seeded = 0

    async def seed(status: str = 'pending', metadata: dict[str, Any] | None = None) -> str:
        nonlocal seeded
        seeded += 1
        added = await backend.add_task(
            project_root=root,
            title=f'task {seeded}',
            description=f'seeded projection fixture task {seeded}',
            metadata=json.dumps(metadata) if metadata is not None else None,
            status=status,
        )
        return str(added['id'])

    try:
        yield ProjectionStack(root, backend, interceptor, server, seed)
    finally:
        await interceptor.close()
        await backend.close()


# ── In-process face: TaskInterceptor.get_tasks ──────────────────────────


@pytest.mark.asyncio
async def test_the_allow_list_is_the_key_tuple_the_store_emits(projection_stack):
    await projection_stack.seed('pending', {'prd_path': 'p'})

    rows = (await projection_stack.backend.get_tasks(projection_stack.root))['tasks']

    assert tuple(rows[0]) == TASK_ROW_KEYS


@pytest.mark.asyncio
async def test_in_process_projection_keeps_only_requested_metadata_present(projection_stack):
    both = await projection_stack.seed('pending', {'prd_path': 'p', 'spawned_from': '12'})
    one = await projection_stack.seed('pending', {'prd_path': 'p'})
    none = await projection_stack.seed('pending', None)

    result = await projection_stack.interceptor.get_tasks(
        projection_stack.root,
        fields=['id', 'metadata.prd_path', 'metadata.spawned_from'],
    )

    assert result['tasks'] == [
        {'id': both, 'metadata': {'prd_path': 'p', 'spawned_from': '12'}},
        {'id': one, 'metadata': {'prd_path': 'p'}},
        {'id': none, 'metadata': {}},
    ]


@pytest.mark.asyncio
async def test_in_process_projection_composes_with_statuses(projection_stack):
    await projection_stack.seed('pending')
    done = await projection_stack.seed('done')

    result = await projection_stack.interceptor.get_tasks(
        projection_stack.root, statuses=['done'], fields=['status'],
    )

    assert result['tasks'] == [{'id': done, 'status': 'done'}]


@pytest.mark.asyncio
async def test_malformed_metadata_projects_as_empty_and_never_aborts(projection_stack):
    healthy = await projection_stack.seed('pending', {'prd_path': 'p'})
    unparseable = await projection_stack.seed('pending', {'prd_path': 'q'})
    not_an_object = await projection_stack.seed('pending', {'prd_path': 'r'})
    projection_stack.corrupt(unparseable, 'not json {')
    projection_stack.corrupt(not_an_object, '[1, 2]')

    result = await projection_stack.interceptor.get_tasks(
        projection_stack.root, fields=['metadata.prd_path'],
    )

    assert result['tasks'] == [
        {'id': healthy, 'metadata': {'prd_path': 'p'}},
        {'id': unparseable, 'metadata': {}},
        {'id': not_an_object, 'metadata': {}},
    ]


@pytest.mark.asyncio
async def test_the_backend_rejects_an_invalid_entry_in_process(projection_stack):
    await projection_stack.seed('pending')

    with pytest.raises(InvalidTaskProjectionField) as caught:
        await projection_stack.interceptor.get_tasks(projection_stack.root, fields=['foo'])

    assert caught.value.field == 'foo'


@pytest.mark.asyncio
async def test_a_tuple_projection_works_in_process(projection_stack):
    task_id = await projection_stack.seed('pending')

    result = await projection_stack.interceptor.get_tasks(projection_stack.root, fields=('id',))

    assert result['tasks'] == [{'id': task_id}]


@pytest.mark.asyncio
async def test_none_fields_returns_the_unprojected_rows(projection_stack):
    await projection_stack.seed('pending', {'prd_path': 'p'})
    await projection_stack.seed('done', None)

    explicit = await projection_stack.interceptor.get_tasks(projection_stack.root, fields=None)
    omitted = await projection_stack.interceptor.get_tasks(projection_stack.root)

    assert explicit == omitted
