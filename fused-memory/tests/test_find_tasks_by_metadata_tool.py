"""find_tasks_by_metadata over a real SQLite task store.

Two faces of one lookup (plans/task-metadata-lookup-prd.md): the in-process
``TaskInterceptor.find_tasks_by_metadata`` seam and the MCP tool. Every case
arranges its store through the backend's public writes, or through the
``shared.project_layout.tasks_db`` file for states no writer produces.
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
from shared.task_statuses import ACTIVE

from fused_memory.backends.metadata_lookup import InvalidMetadataLookupError
from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.models.scope import resolve_project_id
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server

_OVER_BUDGET_BYTES = 48_000
_SLIM_ROW_KEYS = {'task_id', 'title', 'status', 'priority', 'updated_at'}
_TOOL = 'find_tasks_by_metadata'


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
            _TOOL, {'project_root': self.root, **args},
        )

    def corrupt(self, task_id: int, raw: str) -> None:
        with contextlib.closing(sqlite3.connect(tasks_db(self.root))) as conn, conn:
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


# ── MCP face: the find_tasks_by_metadata tool ───────────────────────────


def _ids(response: dict[str, Any]) -> list[int]:
    return [row['task_id'] for row in response['results']]


@pytest_asyncio.fixture
async def unbacked_call(lookup_stack) -> AsyncIterator[Callable[..., Awaitable[Any]]]:
    """The tool over an interceptor with no task store: any store access errors."""
    interceptor = TaskInterceptor(None, None, EventBuffer(db_path=None))
    server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)

    async def call(**args: Any) -> Any:
        return await server._tool_manager.call_tool(
            _TOOL, {'project_root': lookup_stack.root, **args},
        )

    try:
        yield call
    finally:
        await interceptor.close()


@pytest.mark.asyncio
async def test_scalar_match_spans_every_status_with_the_contract_envelope(lookup_stack):
    finding = {'x_finding_key': 'fk-aaaaaaaaaaaa'}
    ids = [await lookup_stack.seed(status, finding) for status in ('pending', 'deferred', 'cancelled')]
    await lookup_stack.seed('pending', {'x_unrelated': 'fk-aaaaaaaaaaaa'})

    response = await lookup_stack.call(key='x_finding_key', value='fk-aaaaaaaaaaaa')

    assert _ids(response) == ids
    assert response['tasks_with_key'] == 3
    assert response['pagination'] == {
        'total': 3, 'offset': 0, 'page_size': 50, 'returned': 3, 'has_more': False,
    }
    assert 'malformed_metadata_ids' not in response
    assert response['project_id'] == resolve_project_id(lookup_stack.root)
    assert response['project_root'] == lookup_stack.root
    assert response['key'] == 'x_finding_key'
    assert response['value'] == 'fk-aaaaaaaaaaaa'
    assert response['statuses'] is None


@pytest.mark.asyncio
async def test_a_list_valued_key_matches_by_membership(lookup_stack):
    task_id = await lookup_stack.seed('pending', {'x_finding_key': ['fk-a', 'fk-b']})

    slim = await lookup_stack.call(key='x_finding_key', value='fk-b')
    full = await lookup_stack.call(key='x_finding_key', value='fk-b', include_value=True)

    assert _ids(slim) == [task_id]
    assert full['results'][0]['matched_value'] == ['fk-a', 'fk-b']


@pytest.mark.asyncio
async def test_presence_matches_a_json_null(lookup_stack):
    null_id = await lookup_stack.seed('pending', {'x_k': None})
    await lookup_stack.seed('pending', {'x_other': 1})

    slim = await lookup_stack.call(key='x_k')
    full = await lookup_stack.call(key='x_k', include_value=True)

    assert _ids(slim) == [null_id]
    assert full['results'][0]['matched_value'] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('key', 'value', 'matches'),
    [
        ('related_tasks', 40, True),
        ('related_tasks', '40', False),
        ('x_flag', 1, False),
    ],
)
async def test_matching_is_type_strict(lookup_stack, key, value, matches):
    task_id = await lookup_stack.seed('pending', {'related_tasks': [12, 40], 'x_flag': True})

    response = await lookup_stack.call(key=key, value=value)

    assert _ids(response) == ([task_id] if matches else [])
    assert response['tasks_with_key'] == 1


@pytest.mark.asyncio
async def test_a_valid_call_without_a_store_reports_the_store_error(unbacked_call):
    response = await unbacked_call(key='x_k')

    assert response['error_type'] == 'RuntimeError'
    assert 'results' not in response


def _assert_validation_error(response: dict[str, Any], arg: str, offending: object) -> None:
    assert response['error_type'] == 'ValidationError'
    assert arg in response['error']
    assert repr(offending) in response['error']
    assert 'results' not in response


@pytest.mark.asyncio
@pytest.mark.parametrize('value', [True, False, 1.5, {'a': 1}, ['a']])
async def test_unsupported_value_types_are_rejected_before_the_store(
    lookup_stack, unbacked_call, value,
):
    _assert_validation_error(await lookup_stack.call(key='x_k', value=value), 'value', value)
    _assert_validation_error(await unbacked_call(key='x_k', value=value), 'value', value)


@pytest.mark.asyncio
@pytest.mark.parametrize('key', ['a"; DROP', '', '-lead', 'a.b', 'a b', 'abc\n'])
async def test_keys_outside_the_grammar_are_rejected_before_the_store(
    lookup_stack, unbacked_call, key,
):
    seeded = await lookup_stack.seed('pending', {'x_k': 'v'})

    _assert_validation_error(await lookup_stack.call(key=key), 'key', key)
    _assert_validation_error(await unbacked_call(key=key), 'key', key)
    assert _ids(await lookup_stack.call(key='x_k', value='v')) == [seeded]


@pytest.mark.asyncio
@pytest.mark.parametrize('include_value', ['yes', 1])
async def test_include_value_must_be_a_real_bool(lookup_stack, unbacked_call, include_value):
    response = await lookup_stack.call(key='x_k', include_value=include_value)
    _assert_validation_error(response, 'include_value', include_value)
    response = await unbacked_call(key='x_k', include_value=include_value)
    _assert_validation_error(response, 'include_value', include_value)


@pytest.mark.asyncio
@pytest.mark.parametrize('statuses', ['cancelled', [None]])
async def test_statuses_must_be_a_list_of_strings(lookup_stack, unbacked_call, statuses):
    _assert_validation_error(
        await lookup_stack.call(key='x_k', statuses=statuses), 'statuses', statuses,
    )
    _assert_validation_error(await unbacked_call(key='x_k', statuses=statuses), 'statuses', statuses)


@pytest.mark.asyncio
async def test_pages_tile_a_large_match_set(lookup_stack):
    ids = [await lookup_stack.seed('pending', {'x_k': 'v'}) for _ in range(120)]

    first = await lookup_stack.call(key='x_k', value='v', limit=100)
    second = await lookup_stack.call(key='x_k', value='v', limit=100, offset=100)

    assert (first['pagination']['returned'], first['pagination']['has_more']) == (100, True)
    assert (second['pagination']['returned'], second['pagination']['has_more']) == (20, False)
    assert _ids(first) + _ids(second) == ids
    assert ids == sorted(set(ids))


@pytest.mark.asyncio
async def test_limit_is_clamped_to_the_cap_and_must_be_positive(lookup_stack, unbacked_call):
    await lookup_stack.seed('pending', {'x_k': 'v'})

    clamped = await lookup_stack.call(key='x_k', limit=500)

    assert clamped['pagination']['page_size'] == 100
    _assert_validation_error(await lookup_stack.call(key='x_k', limit=0), 'limit', 0)
    _assert_validation_error(await unbacked_call(key='x_k', limit=0), 'limit', 0)


@pytest.mark.asyncio
@pytest.mark.parametrize('raw', ['NOT JSON', '[1, 2]'])
async def test_a_malformed_row_is_reported_and_skipped(lookup_stack, raw):
    good = [await lookup_stack.seed('pending', {'x_k': 'v'}) for _ in range(2)]
    corrupt_id = await lookup_stack.seed('pending', {'x_k': 'v'})
    lookup_stack.corrupt(corrupt_id, raw)

    response = await lookup_stack.call(key='x_k', value='v')

    assert response['malformed_metadata_ids'] == [corrupt_id]
    assert _ids(response) == good


@pytest.mark.asyncio
async def test_the_status_filter_scopes_results_and_key_counts(lookup_stack):
    await lookup_stack.seed('pending', {'x_k': 'v'})
    cancelled = [await lookup_stack.seed('cancelled', {'x_k': 'v'}) for _ in range(2)]

    response = await lookup_stack.call(key='x_k', value='v', statuses=['cancelled'])

    assert _ids(response) == cancelled
    assert response['tasks_with_key'] == 2
    assert response['statuses'] == ['cancelled']


@pytest.mark.asyncio
async def test_a_key_nobody_carries_says_so(lookup_stack):
    await lookup_stack.seed('pending', {'x_k': 'v'})

    response = await lookup_stack.call(key='x_nobody')

    assert response['results'] == []
    assert response['tasks_with_key'] == 0
    assert response['pagination']['total'] == 0
    assert response['pagination']['has_more'] is False


@pytest.mark.asyncio
async def test_the_default_omits_values_and_applies_no_byte_cut(lookup_stack):
    big = 'a' * 60_000
    assert len(json.dumps(big).encode('utf-8')) > _OVER_BUDGET_BYTES
    await lookup_stack.seed('pending', {'x_blob': big})
    await lookup_stack.seed('pending', {'x_blob': 'small'})

    response = await lookup_stack.call(key='x_blob')

    assert response['pagination']['returned'] == 2
    for row in response['results']:
        assert 'matched_value' not in row
        assert 'truncated_value' not in row


@pytest.mark.asyncio
async def test_a_store_error_is_an_error_payload_never_an_empty_page(lookup_stack, tmp_path):
    broken_root = _git_init_root(tmp_path / 'broken')
    _write_garbage_store(broken_root)

    response = await lookup_stack.server._tool_manager.call_tool(
        _TOOL, {'project_root': broken_root, 'key': 'x_k'},
    )

    assert 'error' in response
    assert 'error_type' in response
    assert 'results' not in response
