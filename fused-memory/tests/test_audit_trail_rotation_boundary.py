"""Audit-trail rotation at the user-observable boundary (task 5771, docs/task-authoring.md §10).

A REAL SqliteTaskBackend behind a REAL TaskInterceptor: a recon-stage
``update_task`` that pushes a task past the threshold comes back bounded, with
the rotated-out text in the archive and the outcome in the response.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
import pytest_asyncio

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor, interceptor_write_succeeded
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.audit_trail_rotation import (
    ROLLUP_KEY,
    ROTATE_THRESHOLD_BYTES,
    task_payload_bytes,
)
from fused_memory.reconciliation.event_buffer import EventBuffer

RECON_AGENT = 'recon-stage-task_knowledge_sync'
GATE_MARKERS = {
    'task_kind': 'deterministic',
    'operational_mode': 'gate',
    'always_escalates': True,
    'execution_class': 'decision',
    'files': ['docs/task-authoring.md'],
}
DETAILS = 'Ruling: none yet. The substantive question stays open.'
BLOCKS = [f'Cycle {n:02d} relay.\n' + 'evidence unchanged; ' * 37 for n in range(20)]
DESCRIPTION = '\n\n'.join(BLOCKS)
NEW_BLOCK = 'Cycle 20 relay: the newest finding.\n' + 'fresh evidence; ' * 380
RELAY_KEYS = {
    'x_relay_2026_09_20': {'relay': 'first'},
    'x_relay_2026_09_21': {'relay': 'second'},
}


class FakeArchive:
    """``after_write`` runs once the content is stored, before the read-back."""

    def __init__(
        self,
        *,
        write_error: Exception | None = None,
        after_write: Callable[[], Awaitable[Any]] | None = None,
    ):
        self.write_error = write_error
        self.after_write = after_write
        self.store: dict[str, str] = {}
        self.discarded: list[str] = []

    async def write(self, *, project_id: str, content: str, metadata: dict[str, Any]) -> str | None:
        if self.write_error is not None:
            raise self.write_error
        memory_id = str(uuid.uuid4())
        self.store[memory_id] = content
        if self.after_write is not None:
            await self.after_write()
        return memory_id

    async def read(self, *, project_id: str, memory_id: str) -> str | None:
        return self.store.get(memory_id)

    async def discard(self, *, project_id: str, memory_id: str) -> None:
        self.discarded.append(memory_id)
        self.store.pop(memory_id, None)


class RecordingCurator:
    """The two TaskCurator methods an update_task re-embed and a close reach."""

    def __init__(self):
        self.reembedded: list[tuple[str, Any]] = []

    async def reembed_task(self, task_id: str, candidate: Any, project_id: str) -> None:
        self.reembedded.append((task_id, candidate))

    async def close(self) -> None:
        pass


@pytest_asyncio.fixture
async def stack(tmp_path):
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await backend.start()
    unique = uuid.uuid4().hex[:8]
    event_buffer = EventBuffer(db_path=tmp_path / f'events_{unique}.db', buffer_size_threshold=100)
    await event_buffer.initialize()
    store = TicketStore(tmp_path / f'tickets_{unique}.db')
    await store.initialize()
    interceptor = TaskInterceptor(backend, None, event_buffer, config=None, ticket_store=store)
    root = str(tmp_path)
    await backend.add_task(
        project_root=root,
        title='Gate: decide the thing',
        description=DESCRIPTION,
        details=DETAILS,
        metadata=json.dumps(GATE_MARKERS),
    )
    try:
        yield interceptor, backend, root
    finally:
        await interceptor.close()
        await backend.close()
        await event_buffer.close()


async def recon_cycle(interceptor: TaskInterceptor, root: str, agent_id: str | None) -> dict:
    return await interceptor.update_task(
        '1',
        root,
        description=DESCRIPTION + '\n\n' + NEW_BLOCK,
        metadata=json.dumps(RELAY_KEYS),
        agent_id=agent_id,
    )


@pytest.mark.asyncio
async def test_recon_write_past_the_threshold_comes_back_bounded(stack):
    interceptor, backend, root = stack
    archive = FakeArchive()
    interceptor.set_audit_trail_archive(archive)
    before = await backend.get_task('1', root)

    result = await recon_cycle(interceptor, root, RECON_AGENT)

    after = await backend.get_task('1', root)
    assert task_payload_bytes(after) < ROTATE_THRESHOLD_BYTES
    assert result['audit_trail_rotation']['status'] == 'rotated'
    assert {**result['updated_task'], 'id': after['id']} == after
    record = after['metadata'][ROLLUP_KEY]['rotations'][0]
    archived = archive.store[record['archive_memory_id']]
    assert BLOCKS[1] in archived and BLOCKS[1] not in after['description']
    assert after['description'].startswith(BLOCKS[0])
    assert after['description'].endswith(NEW_BLOCK)
    for column in ('status', 'details', 'title'):
        assert after[column] == before[column]
    for key, value in GATE_MARKERS.items():
        assert after['metadata'][key] == value
    assert after['metadata']['pending_since'] == before['metadata']['pending_since']
    assert not any(key.startswith('x_relay_2026') for key in after['metadata'])
    assert [entry['value'] for entry in after['metadata']['x_relay_history']] == [
        RELAY_KEYS['x_relay_2026_09_21'],
        RELAY_KEYS['x_relay_2026_09_20'],
    ]
    queries = after['metadata']['memory_hints']['queries']
    assert any(record['archive_memory_id'] in query for query in queries)


@pytest.mark.asyncio
async def test_non_recon_write_is_never_rotated(stack):
    interceptor, backend, root = stack
    archive = FakeArchive()
    interceptor.set_audit_trail_archive(archive)

    result = await recon_cycle(interceptor, root, None)

    after = await backend.get_task('1', root)
    assert 'audit_trail_rotation' not in result
    assert archive.store == {}
    assert set(RELAY_KEYS) <= set(after['metadata'])
    assert after['description'].endswith(NEW_BLOCK) and BLOCKS[1] in after['description']


@pytest.mark.asyncio
async def test_rotation_is_dormant_until_an_archive_is_wired(stack):
    interceptor, backend, root = stack

    result = await recon_cycle(interceptor, root, RECON_AGENT)

    after = await backend.get_task('1', root)
    assert interceptor_write_succeeded(result)
    assert 'audit_trail_rotation' not in result
    assert set(RELAY_KEYS) <= set(after['metadata'])
    assert ROLLUP_KEY not in after['metadata']


@pytest.mark.asyncio
async def test_archive_failure_reports_error_and_removes_nothing(stack):
    interceptor, backend, root = stack
    interceptor.set_audit_trail_archive(FakeArchive(write_error=RuntimeError('mem0 down')))

    result = await recon_cycle(interceptor, root, RECON_AGENT)

    after = await backend.get_task('1', root)
    assert interceptor_write_succeeded(result)
    assert result['audit_trail_rotation']['status'] == 'error'
    assert 'mem0 down' in result['audit_trail_rotation']['error']
    assert after['description'] == DESCRIPTION + '\n\n' + NEW_BLOCK
    assert set(RELAY_KEYS) <= set(after['metadata'])
    assert ROLLUP_KEY not in after['metadata']


@pytest.mark.asyncio
async def test_details_dominated_task_is_reported_unrotatable(stack):
    interceptor, backend, root = stack
    archive = FakeArchive()
    interceptor.set_audit_trail_archive(archive)
    await backend.add_task(
        project_root=root,
        title='Gate with a long ruling',
        description='One block only.',
        details='d' * 41_000,
        metadata=json.dumps(GATE_MARKERS),
    )

    result = await interceptor.update_task(
        '2', root, metadata=json.dumps({'x_note': 'cycle 9'}), agent_id=RECON_AGENT
    )

    after = await backend.get_task('2', root)
    outcome = result['audit_trail_rotation']
    assert outcome['status'] == 'unrotatable'
    assert outcome['unrotatable']['column_bytes']['details'] == 41_000
    assert archive.store == {}
    assert after['description'] == 'One block only.'
    assert after['details'] == 'd' * 41_000
    assert after['metadata']['x_note'] == 'cycle 9'
    assert ROLLUP_KEY not in after['metadata']


@pytest.mark.asyncio
async def test_a_task_changed_between_planning_and_commit_is_left_alone(stack):
    interceptor, backend, root = stack

    async def concurrent_edit() -> None:
        await backend.update_task('1', root, metadata=json.dumps({'x_concurrent': 'edit'}))

    archive = FakeArchive(after_write=concurrent_edit)
    interceptor.set_audit_trail_archive(archive)

    result = await recon_cycle(interceptor, root, RECON_AGENT)

    after = await backend.get_task('1', root)
    outcome = result['audit_trail_rotation']
    assert outcome['status'] == 'superseded'
    assert outcome['archive_discarded'] is True
    assert archive.discarded == [outcome['archive_memory_id']]
    assert archive.store == {}
    assert after['description'] == DESCRIPTION + '\n\n' + NEW_BLOCK
    assert after['metadata']['x_concurrent'] == 'edit'
    assert set(RELAY_KEYS) <= set(after['metadata'])
    assert ROLLUP_KEY not in after['metadata']


@pytest.mark.asyncio
async def test_a_description_only_the_rotation_rewrote_is_re_embedded(stack):
    interceptor, backend, root = stack
    await backend.update_task('1', root, description=DESCRIPTION + '\n\n' + NEW_BLOCK)
    curator = RecordingCurator()
    interceptor._curator = curator  # type: ignore[assignment]
    interceptor.set_audit_trail_archive(FakeArchive())

    result = await interceptor.update_task(
        '1', root, metadata=json.dumps({'x_note': 'cycle 21'}), agent_id=RECON_AGENT
    )
    await interceptor.drain()

    after = await backend.get_task('1', root)
    assert result['audit_trail_rotation']['description_rewritten'] is True
    [(task_id, candidate)] = curator.reembedded
    assert task_id == '1'
    assert candidate.description == after['description']


@pytest.mark.asyncio
async def test_a_small_recon_write_comes_back_unchanged(stack):
    interceptor, backend, root = stack
    archive = FakeArchive()
    interceptor.set_audit_trail_archive(archive)
    await backend.add_task(
        project_root=root,
        title='A small task',
        description='One block only.',
        metadata=json.dumps(GATE_MARKERS),
    )

    result = await interceptor.update_task(
        '2', root, metadata=json.dumps({'x_note': 'cycle 1'}), agent_id=RECON_AGENT
    )

    after = await backend.get_task('2', root)
    assert interceptor_write_succeeded(result)
    assert 'audit_trail_rotation' not in result
    assert archive.store == {}
    assert {**result['updated_task'], 'id': after['id']} == after
    assert ROLLUP_KEY not in after['metadata']
