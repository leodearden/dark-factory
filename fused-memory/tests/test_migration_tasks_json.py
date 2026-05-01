"""Round-trip test for the tasks.json → SQLite migration script.

Builds a synthetic tasks.json, runs the migration in-process, then reads
back through :class:`SqliteTaskBackend.get_tasks` and asserts a deep-equal
match (after volatile-field normalisation).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig

# Import via path so the script can run as a module from anywhere — mirrors
# how the migration script is invoked in production.
import sys

_SCRIPTS_DIR = (
    Path(__file__).resolve().parent.parent / 'scripts'
)
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import migrate_tasks_json_to_sqlite as migration  # type: ignore[import]


SAMPLE_TASKS = {
    'master': {
        'tasks': [
            {
                'id': 1,
                'title': 'Top-level alpha',
                'description': 'first',
                'details': 'doit',
                'testStrategy': '',
                'status': 'done',
                'dependencies': [],
                'priority': 'high',
                'subtasks': [],
                'updatedAt': '2026-04-01T10:00:00.000Z',
                'metadata': {'prd': '/some/prd.md'},
            },
            {
                'id': 2,
                'title': 'Top-level beta',
                'description': 'second',
                'details': '',
                'testStrategy': '',
                'status': 'pending',
                'dependencies': [1],
                'priority': 'medium',
                'subtasks': [
                    {
                        'id': 1,
                        'title': 'Sub one',
                        'description': '',
                        'details': '',
                        'status': 'pending',
                        'dependencies': [],
                        'parentTaskId': 2,
                        'parentId': 'undefined',
                        'updatedAt': '2026-04-01T11:00:00.000Z',
                    },
                    {
                        'id': 2,
                        'title': 'Sub two',
                        'description': '',
                        'details': '',
                        'status': 'in-progress',
                        'dependencies': [],
                        'parentTaskId': 2,
                        'parentId': 'undefined',
                        'updatedAt': '2026-04-01T12:00:00.000Z',
                    },
                ],
                'updatedAt': '2026-04-01T11:30:00.000Z',
                'metadata': {},
            },
        ],
        'metadata': {
            'version': '1.0.0',
            'lastModified': '2026-04-01T12:00:00.000Z',
        },
    },
}


def _build_project(root: Path) -> None:
    json_path = root / '.taskmaster' / 'tasks' / 'tasks.json'
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(SAMPLE_TASKS, indent=2))


@pytest.mark.asyncio
async def test_migration_round_trip_matches_source(tmp_path):
    project_root = tmp_path / 'proj'
    _build_project(project_root)

    failures = await migration.main_async(
        [project_root], replace=False,
    )
    assert failures == 0
    assert (
        project_root / '.taskmaster' / 'tasks' / 'tasks.db'
    ).exists()

    cfg = TaskmasterConfig(project_root=str(project_root), backend_mode='sqlite')
    backend = SqliteTaskBackend(cfg)
    await backend.start()
    try:
        listing = await backend.get_tasks(project_root=str(project_root))
    finally:
        await backend.close()

    by_id = {t['id']: t for t in listing['tasks']}
    assert set(by_id) == {'1', '2'}

    alpha = by_id['1']
    assert alpha['title'] == 'Top-level alpha'
    assert alpha['status'] == 'done'
    assert alpha['priority'] == 'high'
    assert alpha['metadata'] == {'prd': '/some/prd.md'}
    assert alpha['subtasks'] == []

    beta = by_id['2']
    assert beta['dependencies'] == [1]
    sub_titles = sorted(s['title'] for s in beta['subtasks'])
    assert sub_titles == ['Sub one', 'Sub two']
    assert {s['status'] for s in beta['subtasks']} == {'pending', 'in-progress'}


@pytest.mark.asyncio
async def test_migration_refuses_overwrite_without_replace_flag(tmp_path):
    project_root = tmp_path / 'proj'
    _build_project(project_root)
    # First run creates the DB.
    assert await migration.main_async([project_root], replace=False) == 0
    # Second run should refuse and report a failure.
    assert await migration.main_async([project_root], replace=False) == 1


@pytest.mark.asyncio
async def test_migration_replace_flag_reapplies(tmp_path):
    project_root = tmp_path / 'proj'
    _build_project(project_root)
    assert await migration.main_async([project_root], replace=False) == 0
    # Mutate JSON, re-run with --replace.
    json_path = project_root / '.taskmaster' / 'tasks' / 'tasks.json'
    payload = json.loads(json_path.read_text())
    payload['master']['tasks'][0]['title'] = 'Renamed alpha'
    json_path.write_text(json.dumps(payload))
    assert await migration.main_async([project_root], replace=True) == 0

    cfg = TaskmasterConfig(project_root=str(project_root), backend_mode='sqlite')
    backend = SqliteTaskBackend(cfg)
    await backend.start()
    try:
        one = await backend.get_task('1', project_root=str(project_root))
    finally:
        await backend.close()
    assert one['title'] == 'Renamed alpha'
