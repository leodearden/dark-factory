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
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

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

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

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
    'created_at': '2026-10-22T00:00:00.000Z',
    'created_at_source': 'stamped',
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


def test_a_requested_created_at_key_a_store_without_the_columns_lacks_is_omitted():
    unmigrated = {
        key: value
        for key, value in _FULL_ROW.items()
        if key not in ('created_at', 'created_at_source')
    }

    assert _project(['created_at', 'created_at_source'], unmigrated) == {'id': '7'}


def test_a_requested_core_row_key_the_row_lacks_fails_loudly_rather_than_vanishing():
    broken = {key: value for key, value in _FULL_ROW.items() if key != 'status'}

    with pytest.raises(KeyError, match='status'):
        _project(['status'], broken)


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

    async def bulk_seed(self, n: int) -> list[tuple[str, str]]:
        """Seed ``n`` tasks cycling over ``_BULK_STATUS_CYCLE``; return their (id, status).

        Only the first goes through ``add_task``, which creates the store's own
        schema; the rest are INSERTed into that store file for volume.
        """
        statuses = [_BULK_STATUS_CYCLE[i % len(_BULK_STATUS_CYCLE)] for i in range(n)]
        first = int(await self.seed(statuses[0], _bulk_metadata(0)))
        rows = [
            (first + i, f'bulk task {i}', status, _json_or_none(_bulk_metadata(i)))
            for i, status in enumerate(statuses) if i > 0
        ]
        with contextlib.closing(sqlite3.connect(tasks_db(self.root))) as conn, conn:
            conn.executemany(
                "INSERT INTO tasks (tag, id, title, status, metadata, updated_at)"
                " VALUES ('master', ?, ?, ?, ?, '2026-10-22T00:00:00Z')",
                rows,
            )
        return [(str(first + i), status) for i, status in enumerate(statuses)]


_BULK_STATUS_CYCLE = ('done', 'pending', 'cancelled')


def _bulk_metadata(i: int) -> dict[str, Any] | None:
    return {'prd_path': f'plans/{i}.md'} if i % 2 == 0 else None


def _json_or_none(metadata: dict[str, Any] | None) -> str | None:
    return json.dumps(metadata) if metadata is not None else None


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
            metadata=_json_or_none(metadata),
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


# ── MCP tool face: boundary rows 1–3 (plans/dashboard-prd-lanes-prd.md §6) ──

_REQUESTED = ['id', 'metadata.prd_path', 'metadata.spawned_from']


@pytest.mark.asyncio
async def test_row1_the_tool_projects_each_row_to_id_and_requested_metadata(projection_stack):
    seeded = {
        await projection_stack.seed('pending', {'prd_path': 'plans/a.md', 'other': 1}): {
            'prd_path': 'plans/a.md',
        },
        await projection_stack.seed('done', {
            'prd_path': 'plans/b.md', 'spawned_from': '41', 'unrelated': [1, 2],
        }): {'prd_path': 'plans/b.md', 'spawned_from': '41'},
        await projection_stack.seed('pending', {'unrelated': 'x'}): {},
        await projection_stack.seed('blocked', None): {},
    }

    result = await projection_stack.call(fields=_REQUESTED)

    assert {row['id']: row['metadata'] for row in result['tasks']} == seeded
    assert all(set(row) == {'id', 'metadata'} for row in result['tasks'])
    assert result['project_root'] == projection_stack.root
    assert 'project_id' in result
    assert 'pagination' not in result


@pytest.mark.asyncio
async def test_row1_none_fields_is_byte_identical_to_omitting_it(projection_stack):
    await projection_stack.seed('pending', {'prd_path': 'plans/a.md'})
    await projection_stack.seed('done', None)

    explicit = await projection_stack.call(fields=None)
    omitted = await projection_stack.call()
    unprojected = await projection_stack.backend.get_tasks(projection_stack.root)

    assert json.dumps(explicit, sort_keys=False) == json.dumps(omitted, sort_keys=False)
    assert explicit['tasks'] == unprojected['tasks']
    assert all(tuple(row) == TASK_ROW_KEYS for row in explicit['tasks'])


@pytest.mark.asyncio
@pytest.mark.parametrize('entry', ['foo', 'metadata.a-b'])
async def test_row2_an_invalid_entry_is_a_validation_error_naming_it(projection_stack, entry):
    await projection_stack.seed('pending')

    result = await projection_stack.call(fields=[entry])

    assert set(result) == {'error', 'error_type', 'field', 'allowed'}
    assert result['error_type'] == 'ValidationError'
    assert result['field'] == entry
    assert result['allowed'] == list(ALLOWED_FIELDS)
    assert entry in result['error']


@pytest.mark.asyncio
async def test_row2_a_bare_string_is_a_validation_error_not_a_pydantic_one(projection_stack):
    await projection_stack.seed('pending')

    result = await projection_stack.call(fields='id')

    assert result['error_type'] == 'ValidationError'
    assert result['field'] == 'id'
    assert result['allowed'] == list(ALLOWED_FIELDS)


async def _walk_pages(
    stack: ProjectionStack, **args: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    envelopes: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = await stack.call(statuses=['done', 'pending'], page_size=2000, offset=offset, **args)
        rows.extend(page['tasks'])
        envelopes.append(page['pagination'])
        if not page['pagination']['has_more']:
            return rows, envelopes
        offset += page['pagination']['page_size']


@pytest.mark.asyncio
async def test_row3_projected_pages_tile_the_filtered_population(projection_stack):
    seeded = await projection_stack.bulk_seed(4500)
    expected_ids = [task_id for task_id, status in seeded if status in ('done', 'pending')]

    rows, envelopes = await _walk_pages(
        projection_stack, fields=['id', 'status', 'metadata.prd_path'],
    )
    _, unprojected_envelopes = await _walk_pages(projection_stack)

    assert len(expected_ids) == 3000
    assert [row['id'] for row in rows] == expected_ids
    assert len(envelopes) == 2
    assert all(
        set(envelope) == {'total', 'offset', 'page_size', 'returned', 'has_more'}
        for envelope in envelopes
    )
    assert envelopes == unprojected_envelopes
    assert all(set(row) <= {'id', 'status', 'metadata'} for row in rows)
    assert all(row['status'] in ('done', 'pending') for row in rows)
