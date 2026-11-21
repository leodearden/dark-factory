"""Tests for scripts/delete_task_count_snapshots.py.

The script removes the Mem0 ``kind='task_count_snapshot'`` records that
reconciliation stopped writing in task 10081. The memory double mirrors the
real ``MemoryService.get_memories_by_metadata`` return shape: each record's
``metadata`` is its full payload dict.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _fm_helpers import load_script_module
from _store_mutation_preflight_contract import (
    SENTINEL,
    deny,
    fail_closed_records,
    neutralise_fixture,
)

from fused_memory.utils.store_mutation_preflight import StoreMutationUnavailable

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'delete_task_count_snapshots.py'

_mod = load_script_module(SCRIPT_PATH, mod_name='delete_task_count_snapshots')

_neutralise = neutralise_fixture(
    _mod,
    note="""``run`` runs the preflight under ``--apply`` before it enumerates
    any project (task 10081). The memory service is an AsyncMock, never a live
    store; ``test_apply_refuses_before_any_read_when_the_store_is_unwritable``
    re-rigs the guard to refuse.""",
)

SNAPSHOT_KIND = 'task_count_snapshot'
KNOWN_PROJECTS = {'reify': '/r/reify', 'dark_factory': '/r/df'}


def _record(memory_id: str, kind: str = SNAPSHOT_KIND) -> dict:
    return {
        'id': memory_id,
        'created_at': '2026-11-01T00:00:00+00:00',
        'metadata': {'kind': kind, 'stage': 'task_knowledge_sync', 'data': '42 total'},
    }


def _memory(records_by_project: dict[str, list[dict]]) -> AsyncMock:
    memory = AsyncMock()

    async def get_memories_by_metadata(project_id, filters, limit=1000):
        return list(records_by_project.get(project_id, []))

    memory.get_memories_by_metadata = AsyncMock(side_effect=get_memories_by_metadata)
    memory.delete_memory = AsyncMock(return_value={'status': 'deleted'})
    return memory


def _args(*argv: str):
    return _mod.build_parser().parse_args(list(argv))


async def _run(memory, *argv: str, known_projects_map=None) -> dict:
    return await _mod.run(
        _args(*argv),
        memory=memory,
        known_projects_map=KNOWN_PROJECTS if known_projects_map is None else known_projects_map,
    )


def _enumerated_projects(memory) -> list[str]:
    return [
        call.kwargs.get('project_id', call.args[0] if call.args else None)
        for call in memory.get_memories_by_metadata.await_args_list
    ]


@pytest.mark.asyncio
async def test_dry_run_lists_matches_and_deletes_nothing():
    memory = _memory({'reify': [_record('snap-reify')]})

    report = await _run(memory)

    assert report['dry_run'] is True
    assert report['projects']['reify']['matched'] == ['snap-reify']
    assert report['projects']['dark_factory']['matched'] == []
    memory.delete_memory.assert_not_awaited()
    assert _mod.resolve_exit_code(report) == 0


@pytest.mark.asyncio
async def test_apply_deletes_each_match_in_its_own_project():
    memory = _memory({
        'reify': [_record('snap-reify')],
        'dark_factory': [_record('snap-df')],
    })

    report = await _run(memory, '--apply')

    deletes = {
        (call.kwargs['memory_id'], call.kwargs['store'], call.kwargs['project_id'])
        for call in memory.delete_memory.await_args_list
    }
    assert deletes == {('snap-reify', 'mem0', 'reify'), ('snap-df', 'mem0', 'dark_factory')}
    assert memory.delete_memory.await_count == 2
    assert report['dry_run'] is False
    assert report['projects']['reify']['deleted'] == ['snap-reify']
    assert report['projects']['dark_factory']['deleted'] == ['snap-df']
    assert _mod.resolve_exit_code(report) == 0


@pytest.mark.asyncio
async def test_apply_never_deletes_a_record_of_another_kind():
    memory = _memory({'reify': [_record('cycle-1', kind='cycle_summary'), _record('snap-reify')]})

    report = await _run(memory, '--apply')

    deleted_ids = [call.kwargs['memory_id'] for call in memory.delete_memory.await_args_list]
    assert deleted_ids == ['snap-reify']
    assert report['projects']['reify']['matched'] == ['snap-reify']


@pytest.mark.asyncio
async def test_project_id_limits_enumeration_to_that_project():
    memory = _memory({'reify': [_record('snap-reify')], 'dark_factory': [_record('snap-df')]})

    report = await _run(memory, '--apply', '--project-id', 'reify')

    assert _enumerated_projects(memory) == ['reify']
    assert list(report['projects']) == ['reify']
    assert [call.kwargs['memory_id'] for call in memory.delete_memory.await_args_list] == [
        'snap-reify',
    ]


@pytest.mark.asyncio
async def test_unknown_project_id_aborts_without_reading_or_deleting():
    memory = _memory({'reify': [_record('snap-reify')]})

    report = await _run(memory, '--apply', '--project-id', 'no_such_project')

    assert report['aborted'] is True
    memory.get_memories_by_metadata.assert_not_awaited()
    memory.delete_memory.assert_not_awaited()
    assert _mod.resolve_exit_code(report) != 0


@pytest.mark.asyncio
async def test_a_failed_delete_is_reported_and_the_others_still_run():
    memory = _memory({'reify': [_record('snap-a'), _record('snap-b')]})

    async def delete_memory(*, memory_id, **kwargs):
        if memory_id == 'snap-a':
            raise RuntimeError('qdrant unavailable')
        return {'status': 'deleted'}

    memory.delete_memory = AsyncMock(side_effect=delete_memory)

    report = await _run(memory, '--apply', '--project-id', 'reify')

    assert report['projects']['reify']['failed'] == ['snap-a']
    assert report['projects']['reify']['deleted'] == ['snap-b']
    assert memory.delete_memory.await_count == 2
    assert _mod.resolve_exit_code(report) != 0


@pytest.mark.asyncio
async def test_apply_refuses_before_any_read_when_the_store_is_unwritable(
    monkeypatch, caplog,
):
    deny(_mod, monkeypatch)
    memory = _memory({'reify': [_record('snap-reify')]})

    with pytest.raises(StoreMutationUnavailable, match=SENTINEL):
        await _run(memory, '--apply')

    memory.get_memories_by_metadata.assert_not_awaited()
    memory.delete_memory.assert_not_awaited()
    assert fail_closed_records(caplog, 'delete_task_count_snapshots')
