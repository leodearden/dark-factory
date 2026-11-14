"""``SqliteTaskBackend.get_task_facts``: one ids-scoped read of status, title, dependencies and done provenance."""

from __future__ import annotations

import json
import sqlite3

import pytest
import pytest_asyncio
from shared.project_layout import tasks_db

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig


@pytest_asyncio.fixture
async def backend(tmp_path):
    b = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await b.start()
    yield b
    await b.close()


@pytest.fixture
def project_root(tmp_path):
    return str(tmp_path / 'proj')


async def _add(backend: SqliteTaskBackend, project_root: str, title: str) -> str:
    return (await backend.add_task(project_root=project_root, title=title))['id']


def _set_out_of_band(project_root: str, task_id: str, *, status: str | None = None,
                     metadata: str | None = None) -> None:
    with sqlite3.connect(tasks_db(project_root)) as conn:
        if status is not None:
            conn.execute('UPDATE tasks SET status=? WHERE id=?', (status, int(task_id)))
        if metadata is not None:
            conn.execute('UPDATE tasks SET metadata=? WHERE id=?', (metadata, int(task_id)))
    conn.close()


def _provenance(**fields: object) -> str:
    return json.dumps({'done_provenance': fields})


@pytest.mark.asyncio
async def test_reply_holds_only_ids_the_store_has(backend, project_root):
    tid = await _add(backend, project_root, 'Alpha')

    facts = await backend.get_task_facts([tid, '999', 'abc', '3.1'], project_root)

    assert set(facts) == {tid}
    assert await backend.get_task_facts([], project_root) == {}
    assert await backend.get_task_facts(['abc', '3.1'], project_root) == {}


@pytest.mark.asyncio
async def test_each_row_carries_the_stored_status_and_title(backend, project_root):
    first = await _add(backend, project_root, 'Alpha')
    second = await _add(backend, project_root, 'Beta')
    _set_out_of_band(project_root, second, status='in-progress')

    facts = await backend.get_task_facts([first, second], project_root)

    assert (facts[first]['status'], facts[first]['title']) == ('pending', 'Alpha')
    assert (facts[second]['status'], facts[second]['title']) == ('in-progress', 'Beta')


@pytest.mark.asyncio
async def test_dependencies_list_each_dependency_with_its_own_live_status(backend, project_root):
    dependent = await _add(backend, project_root, 'Dependent')
    later = await _add(backend, project_root, 'Later')
    earlier = await _add(backend, project_root, 'Earlier')
    await backend.add_dependency(dependent, earlier, project_root)
    await backend.add_dependency(dependent, later, project_root)
    _set_out_of_band(project_root, earlier, status='done')

    facts = await backend.get_task_facts([dependent], project_root)

    assert facts[dependent]['dependencies'] == [
        {'id': later, 'status': 'pending'},
        {'id': earlier, 'status': 'done'},
    ]
    assert set(facts) == {dependent}


@pytest.mark.asyncio
async def test_a_task_without_dependencies_has_an_empty_list(backend, project_root):
    tid = await _add(backend, project_root, 'Alone')

    facts = await backend.get_task_facts([tid], project_root)

    assert facts[tid]['dependencies'] == []


@pytest.mark.asyncio
async def test_a_dependency_left_dangling_by_remove_tasks_has_status_none(backend, project_root):
    dependent = await _add(backend, project_root, 'Dependent')
    gone = await _add(backend, project_root, 'Gone')
    await backend.add_dependency(dependent, gone, project_root)
    await backend.remove_tasks([gone], project_root)

    facts = await backend.get_task_facts([dependent, gone], project_root)

    assert facts[dependent]['dependencies'] == [{'id': gone, 'status': None}]
    assert gone not in facts


@pytest.mark.asyncio
async def test_done_provenance_kind_and_commit_come_from_metadata(backend, project_root):
    tid = await _add(backend, project_root, 'Finished')
    _set_out_of_band(
        project_root, tid, status='done',
        metadata=_provenance(kind='merged', commit='abc123def'),
    )

    row = (await backend.get_task_facts([tid], project_root))[tid]

    assert (row['done_provenance_kind'], row['done_provenance_commit']) == ('merged', 'abc123def')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'metadata',
    [
        pytest.param(None, id='no-metadata'),
        pytest.param(json.dumps({'other': 1}), id='no-done-provenance'),
        pytest.param(_provenance(kind='', commit='   '), id='blank'),
        pytest.param(_provenance(kind=7, commit=12345), id='not-a-json-string'),
        pytest.param(json.dumps({'done_provenance': 'merged'}), id='provenance-not-an-object'),
        pytest.param('{not json', id='malformed-json'),
    ],
)
async def test_done_provenance_reads_as_none_when_absent_blank_or_malformed(
    backend, project_root, metadata,
):
    tid = await _add(backend, project_root, 'Finished')
    _set_out_of_band(project_root, tid, status='done', metadata=metadata)

    row = (await backend.get_task_facts([tid], project_root))[tid]

    assert (row['done_provenance_kind'], row['done_provenance_commit']) == (None, None)
    assert row['status'] == 'done'


@pytest.mark.asyncio
async def test_provenance_values_are_stripped(backend, project_root):
    tid = await _add(backend, project_root, 'Finished')
    _set_out_of_band(project_root, tid, status='done', metadata=_provenance(kind=' merged ', commit=' abc '))

    row = (await backend.get_task_facts([tid], project_root))[tid]

    assert (row['done_provenance_kind'], row['done_provenance_commit']) == ('merged', 'abc')
