"""Terminal-entry claimant handling at the interceptor status funnel (task 4866).

Drives a real SqliteTaskBackend behind a real TaskInterceptor and reads every
outcome back through ``get_task``, the path a task-store reader observes.
Contracts: docs/prds/claimant-invariant-enforcement.md C4-E2..C4-E5 (the
clear) and docs/prds/claimant-invariant-detection.md D-5/E-2 (the
``claimant_exception`` stamp on an honoured explicit claimant).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from _fm_helpers import _init_git_repo, make_db_without_claimant_columns

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.event_buffer import EventBuffer

CLAIMANT = 'run-a/sess-1/pid=11'


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'terminal_claimant_eb.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@pytest_asyncio.fixture
async def backend(tmp_path):
    b = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await b.start()
    yield b
    await b.close()


@pytest.fixture
def interceptor(backend, event_buffer):
    return TaskInterceptor(backend, None, event_buffer)


@pytest.fixture
def root(tmp_path) -> str:
    return str(tmp_path)


async def _claimed_in_progress(
    interceptor: TaskInterceptor,
    backend: SqliteTaskBackend,
    project_root: str,
    *,
    title: str = 'claimed task',
    claimant: str = CLAIMANT,
    heartbeat: str | None = None,
) -> str:
    added = await backend.add_task(project_root=project_root, title=title)
    task_id = str(added['id'])
    result = await interceptor.set_task_status(
        task_id, 'in-progress', project_root,
        claimant_run_id=claimant, heartbeat_at=heartbeat or _now_iso(),
    )
    assert 'error' not in result, result
    return task_id


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['done', 'cancelled'])
async def test_terminal_write_clears_both_claimant_columns(status, interceptor, backend, root):
    task_id = await _claimed_in_progress(interceptor, backend, root)

    result = await interceptor.set_task_status(task_id, status, root)

    assert 'error' not in result, result
    task = await backend.get_task(task_id, project_root=root)
    assert task['status'] == status
    assert task['claimant_run_id'] is None
    assert task['heartbeat_at'] is None


@pytest.mark.asyncio
async def test_csv_terminal_write_clears_every_id(interceptor, backend, root):
    ids = [
        await _claimed_in_progress(interceptor, backend, root, title=f'claimed {n}')
        for n in range(3)
    ]

    result = await interceptor.set_task_status(','.join(ids), 'cancelled', root)

    assert result['success'] is True, result
    for task_id in ids:
        task = await backend.get_task(task_id, project_root=root)
        assert task['claimant_run_id'] is None, task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'status', ['blocked', 'deferred', 'review', 'infra-hold', 'merge-deferred', 'pending'],
)
async def test_non_terminal_write_preserves_claimant(status, interceptor, backend, root):
    heartbeat = _now_iso()
    task_id = await _claimed_in_progress(interceptor, backend, root, heartbeat=heartbeat)

    result = await interceptor.set_task_status(task_id, status, root)

    assert 'error' not in result, result
    task = await backend.get_task(task_id, project_root=root)
    assert task['status'] == status
    assert task['claimant_run_id'] == CLAIMANT
    assert task['heartbeat_at'] == heartbeat


@pytest.mark.asyncio
async def test_explicit_claimant_on_terminal_write_is_honoured(interceptor, backend, root):
    task_id = await _claimed_in_progress(interceptor, backend, root)
    heartbeat = _now_iso()

    result = await interceptor.set_task_status(
        task_id, 'done', root, claimant_run_id='run/s/pid=1', heartbeat_at=heartbeat,
    )

    assert 'error' not in result, result
    task = await backend.get_task(task_id, project_root=root)
    assert task['claimant_run_id'] == 'run/s/pid=1'
    assert task['heartbeat_at'] == heartbeat


@pytest.mark.asyncio
async def test_same_status_terminal_write_stays_a_true_no_op(interceptor, backend, root):
    task_id = await _claimed_in_progress(interceptor, backend, root)
    assert 'error' not in await interceptor.set_task_status(task_id, 'done', root)
    await interceptor.set_task_claimant(
        task_id, root, claimant_run_id='leaked/s/pid=9', heartbeat_at=_now_iso(),
    )

    result = await interceptor.set_task_status(task_id, 'done', root)

    assert result.get('no_op') is True, result
    task = await backend.get_task(task_id, project_root=root)
    assert task['claimant_run_id'] == 'leaked/s/pid=9'


@pytest.mark.asyncio
async def test_terminal_write_on_pre_migration_connection_succeeds(tmp_path, event_buffer, caplog):
    project_root = str(tmp_path / 'proj')
    make_db_without_claimant_columns(
        Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db', status='in-progress',
    )
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await backend.start()
    try:
        interceptor = TaskInterceptor(backend, None, event_buffer)
        with caplog.at_level(logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'):
            result = await interceptor.set_task_status('1', 'done', project_root)
        task = await backend.get_task('1', project_root=project_root)
    finally:
        await backend.close()

    assert 'error' not in result, result
    assert task['status'] == 'done'
    assert any(
        'columns absent' in record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ), [record.getMessage() for record in caplog.records]


async def _terminal_write_with_explicit_claimant(
    interceptor: TaskInterceptor,
    backend: SqliteTaskBackend,
    project_root: str,
    status: str,
) -> tuple[dict, datetime, datetime]:
    task_id = await _claimed_in_progress(interceptor, backend, project_root)
    t0 = datetime.now(UTC)
    result = await interceptor.set_task_status(
        task_id, status, project_root, claimant_run_id='run/s/pid=1', agent_id='steward-x',
    )
    t1 = datetime.now(UTC)
    assert 'error' not in result, result
    return await backend.get_task(task_id, project_root=project_root), t0, t1


def _assert_exception_stamp(task: dict, status: str, t0: datetime, t1: datetime) -> None:
    stamp = task['metadata']['claimant_exception']
    assert stamp['claimant_run_id'] == 'run/s/pid=1'
    assert stamp['target_status'] == status
    assert stamp['agent_id'] == 'steward-x'
    assert stamp['tag'] is None
    stamped_at = datetime.fromisoformat(stamp['stamped_at'])
    assert stamped_at.utcoffset() == timedelta(0)
    assert t0 <= stamped_at <= t1
    assert task['claimant_run_id'] == 'run/s/pid=1'


@pytest.mark.asyncio
async def test_explicit_claimant_terminal_write_stamps_exception(interceptor, backend, root):
    task, t0, t1 = await _terminal_write_with_explicit_claimant(interceptor, backend, root, 'done')

    _assert_exception_stamp(task, 'done', t0, t1)


@pytest.mark.asyncio
async def test_explicit_claimant_cancel_stamps_target_status_cancelled(interceptor, backend, root):
    task, t0, t1 = await _terminal_write_with_explicit_claimant(interceptor, backend, root, 'cancelled')

    _assert_exception_stamp(task, 'cancelled', t0, t1)


@pytest.mark.asyncio
async def test_exception_stamp_is_atomic_with_done_provenance(tmp_path, interceptor, backend, root):
    sha = _init_git_repo(tmp_path)
    task_id = await _claimed_in_progress(interceptor, backend, root)

    result = await interceptor.set_task_status(
        task_id, 'done', root,
        done_provenance={'kind': 'merged', 'commit': sha},
        claimant_run_id='run/s/pid=1',
    )

    assert 'error' not in result, result
    task = await backend.get_task(task_id, project_root=root)
    assert task['status'] == 'done'
    assert task['metadata']['done_provenance']['kind'] == 'merged'
    assert task['metadata']['claimant_exception']['claimant_run_id'] == 'run/s/pid=1'


@pytest.mark.asyncio
async def test_plain_terminal_write_stamps_nothing(interceptor, backend, root):
    task_id = await _claimed_in_progress(interceptor, backend, root)

    await interceptor.set_task_status(task_id, 'done', root)

    task = await backend.get_task(task_id, project_root=root)
    assert 'claimant_exception' not in task['metadata']
    assert task['claimant_run_id'] is None


@pytest.mark.asyncio
async def test_non_terminal_explicit_claimant_stamps_nothing(interceptor, backend, root):
    task_id = await _claimed_in_progress(interceptor, backend, root)

    await interceptor.set_task_status(task_id, 'blocked', root, claimant_run_id='run/s/pid=2')

    task = await backend.get_task(task_id, project_root=root)
    assert 'claimant_exception' not in task['metadata']


@pytest.mark.asyncio
async def test_same_status_no_op_stamps_nothing(interceptor, backend, root):
    task_id = await _claimed_in_progress(interceptor, backend, root)
    await interceptor.set_task_status(task_id, 'done', root)

    result = await interceptor.set_task_status(task_id, 'done', root, claimant_run_id='run/s/pid=3')

    assert result.get('no_op') is True, result
    task = await backend.get_task(task_id, project_root=root)
    assert 'claimant_exception' not in task['metadata']
    assert task['claimant_run_id'] is None


@pytest.mark.asyncio
async def test_explicit_null_claimant_on_terminal_write_stamps_nothing(interceptor, backend, root):
    task_id = await _claimed_in_progress(interceptor, backend, root)

    await interceptor.set_task_status(task_id, 'done', root, claimant_run_id=None)

    task = await backend.get_task(task_id, project_root=root)
    assert 'claimant_exception' not in task['metadata']
    assert task['claimant_run_id'] is None


def _exception_log_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith('claimant_exception:')
    ]


@pytest.mark.asyncio
async def test_exception_stamp_logs_structured_warning(interceptor, backend, root, caplog):
    with caplog.at_level(logging.WARNING, logger='fused_memory.middleware.task_interceptor'):
        task, _, _ = await _terminal_write_with_explicit_claimant(interceptor, backend, root, 'done')
    messages = _exception_log_messages(caplog)
    assert len(messages) == 1, messages
    assert f'task_id={task["id"]}' in messages[0]
    assert 'done' in messages[0]
    assert 'run/s/pid=1' in messages[0]

    caplog.clear()
    plain_id = await _claimed_in_progress(interceptor, backend, root, title='plain task')
    with caplog.at_level(logging.WARNING, logger='fused_memory.middleware.task_interceptor'):
        await interceptor.set_task_status(plain_id, 'done', root)
    assert _exception_log_messages(caplog) == []
