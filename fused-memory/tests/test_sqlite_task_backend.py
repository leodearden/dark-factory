"""Contract tests for :class:`SqliteTaskBackend`.

The wire-shape surface mirrors
``tests/test_taskmaster_client_contract.py`` so the dual-compare soak runs
clean: every method here returns the same DTO shapes the legacy
TaskmasterBackend wrappers do.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import pytest_asyncio

from fused_memory.backends.sqlite_task_backend import (
    SqliteTaskBackend,
    _format_task_id,
    _parse_task_id,
)
from fused_memory.backends.taskmaster_types import TaskmasterError
from fused_memory.config.schema import TaskmasterConfig


@pytest_asyncio.fixture
async def backend(tmp_path):
    cfg = TaskmasterConfig(project_root=str(tmp_path), backend_mode='sqlite')
    b = SqliteTaskBackend(cfg)
    await b.start()
    yield b
    await b.close()


@pytest_asyncio.fixture
async def project_root(tmp_path):
    return str(tmp_path / 'proj')


# ── ID parsing ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    'raw,expected_id,expected_parent',
    [
        ('5', 5, None),
        ('  10 ', 10, None),
        (7, 7, None),
        ('292.1', 1, 292),
        ('100.42', 42, 100),
    ],
)
def test_parse_task_id_accepts_dotted_and_bare(raw, expected_id, expected_parent):
    tid, parent = _parse_task_id(raw)
    assert tid == expected_id
    assert parent == expected_parent


@pytest.mark.parametrize('raw', ['', 'abc', '1.2.3', '5.x', 'x.5'])
def test_parse_task_id_rejects_malformed(raw):
    with pytest.raises(TaskmasterError) as exc:
        _parse_task_id(raw)
    assert exc.value.code == 'INVALID_TASK_ID'


def test_format_task_id_round_trips():
    assert _format_task_id(7, None) == '7'
    assert _format_task_id(2, 7) == '7.2'


# ── Lifecycle ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_close_idempotent(tmp_path):
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg)
    await b.start()
    await b.start()  # idempotent
    assert b.connected is True
    assert b.restart_count == 1
    await b.close()
    await b.close()  # idempotent
    assert b.connected is False


@pytest.mark.asyncio
async def test_is_alive_reports_state(backend, project_root):
    alive, err = await backend.is_alive()
    assert alive is True
    assert err is None
    await backend.close()
    alive, err = await backend.is_alive()
    assert alive is False


# ── add_task / get_task / get_tasks ────────────────────────────────


@pytest.mark.asyncio
async def test_add_task_then_get_returns_dto(backend, project_root):
    dto = await backend.add_task(
        project_root=project_root, title='First', description='desc',
        details='details', priority='high',
    )
    assert dto['id'] == '1'
    assert 'Successfully added' in dto['message']

    one = await backend.get_task('1', project_root=project_root)
    assert one['id'] == 1  # singular get returns int per Taskmaster wire
    assert one['title'] == 'First'
    assert one['priority'] == 'high'
    assert one['status'] == 'pending'
    assert one['subtasks'] == []

    listing = await backend.get_tasks(project_root=project_root)
    assert isinstance(listing['tasks'], list)
    assert listing['tasks'][0]['id'] == '1'  # plural get_tasks returns string


@pytest.mark.asyncio
async def test_add_task_increments_id(backend, project_root):
    await backend.add_task(project_root=project_root, title='one')
    await backend.add_task(project_root=project_root, title='two')
    listing = await backend.get_tasks(project_root=project_root)
    assert sorted(t['id'] for t in listing['tasks']) == ['1', '2']


@pytest.mark.asyncio
async def test_add_task_promotes_prompt_to_title(backend, project_root):
    dto = await backend.add_task(
        project_root=project_root,
        prompt='Build a frobinator that does X\n\nDetails here',
    )
    one = await backend.get_task(dto['id'], project_root=project_root)
    assert one['title'].startswith('Build a frobinator')
    assert 'Details here' in one['description']


@pytest.mark.asyncio
async def test_add_task_without_title_or_prompt_raises(backend, project_root):
    with pytest.raises(TaskmasterError) as exc:
        await backend.add_task(project_root=project_root)
    assert 'prompt' in exc.value.message


@pytest.mark.asyncio
async def test_get_task_not_found_raises(backend, project_root):
    await backend.add_task(project_root=project_root, title='one')
    with pytest.raises(TaskmasterError) as exc:
        await backend.get_task('999', project_root=project_root)
    assert exc.value.code == 'TASKMASTER_TOOL_ERROR'
    assert 'No tasks found' in exc.value.message


# ── set_task_status ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_task_status_returns_per_id_payload(backend, project_root):
    await backend.add_task(project_root=project_root, title='x')
    result = await backend.set_task_status(
        '1', 'done', project_root=project_root,
    )
    assert 'done' in result['message']
    assert result['tasks'] == [{
        'taskId': '1',
        'oldStatus': 'pending',
        'newStatus': 'done',
    }]


@pytest.mark.asyncio
async def test_set_task_status_unknown_id_raises(backend, project_root):
    with pytest.raises(TaskmasterError):
        await backend.set_task_status('99', 'done', project_root=project_root)


# ── update_task ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_task_appends_metadata(backend, project_root):
    await backend.add_task(
        project_root=project_root,
        title='x',
        metadata=json.dumps({'prd': 'old.md'}),
    )
    dto = await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'modules': ['src']}),
        append=True,
    )
    assert dto['updated'] is True
    one = await backend.get_task('1', project_root=project_root)
    assert one['metadata']['prd'] == 'old.md'
    assert one['metadata']['modules'] == ['src']


@pytest.mark.asyncio
async def test_update_task_overwrites_metadata_without_append(backend, project_root):
    await backend.add_task(
        project_root=project_root, title='x',
        metadata=json.dumps({'prd': 'old.md'}),
    )
    await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'prd': 'new.md'}),
    )
    one = await backend.get_task('1', project_root=project_root)
    assert one['metadata'] == {'prd': 'new.md'}


# ── add_subtask / nested IDs ───────────────────────────────────────


@pytest.mark.asyncio
async def test_add_subtask_returns_dotted_id(backend, project_root):
    await backend.add_task(project_root=project_root, title='parent')
    dto = await backend.add_subtask(
        '1', project_root=project_root, title='Sub one',
    )
    assert dto['id'] == '1.1'
    assert dto['parent_id'] == '1'
    assert 'created' in dto['message']
    assert dto['subtask']['title'] == 'Sub one'


@pytest.mark.asyncio
async def test_add_subtask_increments_local_id(backend, project_root):
    await backend.add_task(project_root=project_root, title='parent')
    a = await backend.add_subtask('1', project_root=project_root, title='A')
    b = await backend.add_subtask('1', project_root=project_root, title='B')
    assert (a['id'], b['id']) == ('1.1', '1.2')

    parent = await backend.get_task('1', project_root=project_root)
    assert [s['id'] for s in parent['subtasks']] == [1, 2]


@pytest.mark.asyncio
async def test_add_subtask_unknown_parent_raises(backend, project_root):
    with pytest.raises(TaskmasterError) as exc:
        await backend.add_subtask('99', project_root=project_root, title='x')
    assert 'Parent task not found' in exc.value.message


@pytest.mark.asyncio
async def test_set_status_on_subtask_via_dotted_id(backend, project_root):
    await backend.add_task(project_root=project_root, title='parent')
    await backend.add_subtask('1', project_root=project_root, title='S')
    result = await backend.set_task_status(
        '1.1', 'done', project_root=project_root,
    )
    assert result['tasks'][0]['taskId'] == '1.1'
    parent = await backend.get_task('1', project_root=project_root)
    assert parent['subtasks'][0]['status'] == 'done'


# ── remove_tasks with cascade ──────────────────────────────────────


@pytest.mark.asyncio
async def test_remove_tasks_cascades_to_subtasks(backend, project_root):
    await backend.add_task(project_root=project_root, title='parent')
    await backend.add_subtask('1', project_root=project_root, title='A')
    await backend.add_subtask('1', project_root=project_root, title='B')

    dto = await backend.remove_tasks(['1'], project_root=project_root)

    assert dto['successful'] == 3
    assert dto['failed'] == 0
    assert sorted(dto['removed_ids']) == ['1', '1.1', '1.2']

    listing = await backend.get_tasks(project_root=project_root)
    assert listing['tasks'] == []


@pytest.mark.asyncio
async def test_remove_tasks_unknown_id_returns_failure_dto(backend, project_root):
    dto = await backend.remove_tasks(['99'], project_root=project_root)
    assert dto['successful'] == 0
    assert dto['failed'] == 1
    assert dto['removed_ids'] == []


@pytest.mark.asyncio
async def test_remove_tasks_batch_mixed_existing_missing(backend, project_root):
    # Two top-levels exist (1, 2); 3 and 99 do not.
    await backend.add_task(project_root=project_root, title='alpha')
    await backend.add_task(project_root=project_root, title='beta')

    dto = await backend.remove_tasks(
        ['1', '2', '3', '99'], project_root=project_root,
    )

    assert dto['successful'] == 2
    assert dto['failed'] == 2
    assert sorted(dto['removed_ids']) == ['1', '2']
    assert '3' in dto['message']
    assert '99' in dto['message']

    listing = await backend.get_tasks(project_root=project_root)
    assert listing['tasks'] == []


@pytest.mark.asyncio
async def test_remove_tasks_cascades_with_explicit_subtask(backend, project_root):
    # Caller asks to remove parent 1 AND its subtask 1.1; cascade naturally
    # pulls in 1.1 as well — must report once, not twice.
    await backend.add_task(project_root=project_root, title='parent')
    await backend.add_subtask('1', project_root=project_root, title='A')
    await backend.add_subtask('1', project_root=project_root, title='B')

    dto = await backend.remove_tasks(
        ['1', '1.1'], project_root=project_root,
    )

    # 1, 1.1, 1.2 — three rows actually deleted; no double-count of 1.1.
    assert dto['successful'] == 3
    assert dto['failed'] == 0
    assert sorted(dto['removed_ids']) == ['1', '1.1', '1.2']
    assert dto['removed_ids'].count('1.1') == 1


@pytest.mark.asyncio
async def test_remove_tasks_atomicity_on_malformed_id(backend, project_root):
    await backend.add_task(project_root=project_root, title='alpha')
    await backend.add_task(project_root=project_root, title='beta')

    with pytest.raises(TaskmasterError):
        # 'oops' is not a parseable id — the whole batch fails before any
        # delete runs. Verify state is unchanged afterwards.
        await backend.remove_tasks(
            ['1', 'oops', '2'], project_root=project_root,
        )

    listing = await backend.get_tasks(project_root=project_root)
    assert sorted(t['id'] for t in listing['tasks']) == ['1', '2']


@pytest.mark.asyncio
async def test_remove_tasks_empty_list_is_noop(backend, project_root):
    dto = await backend.remove_tasks([], project_root=project_root)
    assert dto['successful'] == 0
    assert dto['failed'] == 0
    assert dto['removed_ids'] == []


# ── Dependencies ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_and_remove_dependency_round_trip(backend, project_root):
    await backend.add_task(project_root=project_root, title='a')
    await backend.add_task(project_root=project_root, title='b')

    add = await backend.add_dependency('2', '1', project_root=project_root)
    assert add['id'] == '2' and add['dependency_id'] == '1'

    listing = await backend.get_tasks(project_root=project_root)
    by_id = {t['id']: t for t in listing['tasks']}
    assert by_id['2']['dependencies'] == [1]

    remove = await backend.remove_dependency(
        '2', '1', project_root=project_root,
    )
    assert remove['id'] == '2'
    listing = await backend.get_tasks(project_root=project_root)
    by_id = {t['id']: t for t in listing['tasks']}
    assert by_id['2']['dependencies'] == []


@pytest.mark.asyncio
async def test_add_dependency_self_loop_raises(backend, project_root):
    await backend.add_task(project_root=project_root, title='a')
    with pytest.raises(TaskmasterError):
        await backend.add_dependency('1', '1', project_root=project_root)


@pytest.mark.asyncio
async def test_validate_dependencies_reports_dangling(backend, project_root):
    await backend.add_task(project_root=project_root, title='a')
    await backend.add_task(project_root=project_root, title='b')
    await backend.add_dependency('2', '1', project_root=project_root)
    # Remove the target so the dependency on it dangles.
    await backend.remove_tasks(['1'], project_root=project_root)
    res = await backend.validate_dependencies(project_root=project_root)
    assert 'Dangling dependencies' in res['message']
    assert '2 -> 1' in res['message']


@pytest.mark.asyncio
async def test_validate_dependencies_clean_returns_success(backend, project_root):
    await backend.add_task(project_root=project_root, title='a')
    res = await backend.validate_dependencies(project_root=project_root)
    assert res['message'] == 'Dependencies validated successfully'


# ── Persistence on disk ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_db_file_lives_at_taskmaster_tasks_dir(backend, project_root):
    await backend.add_task(project_root=project_root, title='x')
    expected = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    assert expected.exists()


@pytest.mark.asyncio
async def test_state_survives_close_and_reopen(tmp_path):
    cfg = TaskmasterConfig(project_root=str(tmp_path), backend_mode='sqlite')
    project_root = str(tmp_path / 'proj')
    b1 = SqliteTaskBackend(cfg)
    await b1.start()
    await b1.add_task(project_root=project_root, title='persisted')
    await b1.close()

    b2 = SqliteTaskBackend(cfg)
    await b2.start()
    listing = await b2.get_tasks(project_root=project_root)
    assert [t['title'] for t in listing['tasks']] == ['persisted']
    await b2.close()


# ── Concurrency ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_add_task_yields_unique_ids(backend, project_root):
    """The per-project write lock must serialise id allocation."""
    coros = [
        backend.add_task(project_root=project_root, title=f't{i}')
        for i in range(20)
    ]
    results = await asyncio.gather(*coros)
    ids = sorted(int(r['id']) for r in results)
    assert ids == list(range(1, 21))


# ── Cancellation hardening ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_status_cancellation_leaves_connection_clean(
    backend, project_root,
):
    """A cancellation arriving while ``set_task_status`` is queued behind
    the write_lock must not leave the connection mid-transaction.

    Reproduces the soak-cancel signature: hold the per-project write lock,
    queue a ``set_task_status`` against it, cancel the awaiter via
    ``wait_for(timeout=0.001)``, then assert the next ``set_task_status``
    applies cleanly. Pre-fix (Exception-only suppress + unshielded
    rollback) the connection could end up holding an open BEGIN, which
    surfaces as ``cannot start a transaction within a transaction``
    on the next mutation.
    """
    # Seed: one task to flip.
    await backend.add_task(project_root=project_root, title='t0')
    assert (await backend.get_task('1', project_root))['status'] == 'pending'

    # Acquire the per-project write lock so the next set_task_status blocks.
    lock = backend._write_lock(project_root)
    await lock.acquire()
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                backend.set_task_status('1', 'in-progress', project_root),
                timeout=0.001,
            )
    finally:
        lock.release()

    # Connection state must be clean: the next mutation succeeds.
    res = await backend.set_task_status('1', 'done', project_root)
    assert res['tasks'][0]['newStatus'] == 'done'
    assert (await backend.get_task('1', project_root))['status'] == 'done'
