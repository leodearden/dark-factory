"""The startup backfill that holds the pending follow-up stock behind the holder.

plans/follow-up-filing-shape-prd.md §3 decision 12 and boundary row 17. Rows are seeded straight into
the store, as the stock filed before the hold existed is, and read back through ``get_task``.
"""

from __future__ import annotations

import ast
import logging
import pathlib
from typing import Any

import pytest
from _ast_guard import calls_named, parse_python_module
from _submit_stack import Stack
from _submit_stack import passthrough_main_checkout as passthrough_main_checkout
from _submit_stack import stack as stack
from shared.followup_hold_records import FOLLOWUP_HOLD_METADATA_KEY, FOLLOWUP_HOLDER_METADATA_KEY

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.reconciliation import EventType
from fused_memory.models.scope import resolve_project_id
from fused_memory.server import main as server_main

_FOLLOWUP = {'source': 'agent-followup', 'spawn_context': 'agent_followup', 'spawned_from': '101'}
_REVIEW = {'spawn_context': 'review_suggestions'}
_EARLIER_HOLD = {'holder': 1, 'held_at': '2026-11-01T00:00:00+00:00'}


async def _row(stack: Stack, task_id: str) -> dict[str, Any]:
    result = await stack.call('get_task', id=task_id)
    assert 'error' not in result, result
    return result


def _dependencies(row: dict[str, Any]) -> list[str]:
    return [str(dependency) for dependency in row.get('dependencies') or []]


async def _holder_ids(stack: Stack) -> list[str]:
    found = await stack.call('find_tasks_by_metadata', key=FOLLOWUP_HOLDER_METADATA_KEY)
    assert 'error' not in found, found
    return [str(row['task_id']) for row in found['results']]


async def _seed_stock(stack: Stack) -> tuple[list[str], list[str]]:
    """Three pending follow-ups the hold rule matches, and the rows it must leave alone."""
    blocker = await stack.seed('An ordinary pending task', {})
    matching = [
        await stack.seed('Low agent follow-up', dict(_FOLLOWUP), priority='low'),
        await stack.seed('Low reviewer suggestion', dict(_REVIEW), priority='low'),
        await stack.seed('Polish agent follow-up', dict(_FOLLOWUP), priority='polish'),
    ]
    await stack.backend.add_dependency(matching[2], blocker, stack.root)
    untouched = [
        blocker,
        await stack.seed('Medium agent follow-up', dict(_FOLLOWUP), priority='medium'),
        await stack.seed('Done low follow-up', dict(_FOLLOWUP), priority='low', status='done'),
        await stack.seed('Planned low follow-up', {**_FOLLOWUP, 'human_decomposed': True}, priority='low'),
        await stack.seed('PRD low follow-up', {**_FOLLOWUP, 'prd_path': 'plans/x-prd.md'}, priority='low'),
        await stack.seed(
            'Released low follow-up', {**_FOLLOWUP, FOLLOWUP_HOLD_METADATA_KEY: _EARLIER_HOLD}, priority='low',
        ),
    ]
    return matching, untouched


@pytest.mark.asyncio
async def test_the_backfill_holds_every_matching_pending_followup_in_one_event(stack: Stack) -> None:
    matching, untouched = await _seed_stock(stack)
    prior = {task_id: _dependencies(await _row(stack, task_id)) for task_id in matching}
    untouched_before = {task_id: await _row(stack, task_id) for task_id in untouched}

    count = await stack.interceptor.backfill_followup_holds(stack.root)

    assert count == 3
    [holder] = await _holder_ids(stack)
    for task_id in matching:
        row = await _row(stack, task_id)
        assert _dependencies(row) == [*prior[task_id], holder], task_id
        assert str(row['metadata'][FOLLOWUP_HOLD_METADATA_KEY]['holder']) == holder
    for task_id, before in untouched_before.items():
        assert await _row(stack, task_id) == before, task_id
    assert await stack.events(EventType.followup_hold_backfilled) == [
        {'project_id': resolve_project_id(stack.root), 'count': 3},
    ]
    assert await stack.events(EventType.task_modified) == []


@pytest.mark.asyncio
async def test_a_second_backfill_adds_nothing_and_still_reports(stack: Stack) -> None:
    matching, _untouched = await _seed_stock(stack)
    await stack.interceptor.backfill_followup_holds(stack.root)
    held = {task_id: _dependencies(await _row(stack, task_id)) for task_id in matching}

    count = await stack.interceptor.backfill_followup_holds(stack.root)

    assert count == 0
    for task_id in matching:
        assert _dependencies(await _row(stack, task_id)) == held[task_id], task_id
    assert [event['count'] for event in await stack.events(EventType.followup_hold_backfilled)] == [3, 0]


@pytest.mark.asyncio
async def test_with_nothing_to_hold_no_holder_is_minted(stack: Stack) -> None:
    await stack.seed('Medium agent follow-up', dict(_FOLLOWUP), priority='medium')

    count = await stack.interceptor.backfill_followup_holds(stack.root)

    assert count == 0
    assert await _holder_ids(stack) == []
    assert await stack.events(EventType.followup_hold_backfilled) == [
        {'project_id': resolve_project_id(stack.root), 'count': 0},
    ]


def test_the_backfill_is_enabled_by_default() -> None:
    assert FusedMemoryConfig().followup_hold.backfill_enabled is True


def test_the_backfill_can_be_switched_off() -> None:
    config = FusedMemoryConfig.model_validate({'followup_hold': {'backfill_enabled': False}})

    assert config.followup_hold.backfill_enabled is False


class _RecordingBackfiller:
    def __init__(self, failing: set[str]) -> None:
        self.seen: list[str] = []
        self._failing = failing

    async def backfill_followup_holds(self, project_root: str) -> int:
        self.seen.append(project_root)
        if project_root in self._failing:
            raise RuntimeError(f'store unreadable under {project_root}')
        return 2


@pytest.fixture
def startup_roots(tmp_path: pathlib.Path) -> dict[str, str]:
    for name in ('first', 'broken', 'last'):
        (tmp_path / name).mkdir()
    return {
        name: str(tmp_path / name) for name in ('first', 'phantom', 'broken', 'last')
    }


@pytest.mark.asyncio
async def test_startup_backfills_every_real_root_and_isolates_a_failure(
    startup_roots: dict[str, str], caplog: pytest.LogCaptureFixture,
) -> None:
    backfiller = _RecordingBackfiller(failing={startup_roots['broken']})
    caplog.set_level(logging.WARNING, logger='fused_memory.server.main')

    await server_main._backfill_followup_holds(FusedMemoryConfig(), backfiller, startup_roots.values())

    assert backfiller.seen == [startup_roots['first'], startup_roots['broken'], startup_roots['last']]
    warned = [record for record in caplog.records if record.levelno == logging.WARNING]
    failed = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert any(startup_roots['phantom'] in record.getMessage() for record in warned), warned
    assert [startup_roots['broken'] in record.getMessage() for record in failed] == [True], failed


@pytest.mark.asyncio
async def test_a_disabled_backfill_touches_no_project(startup_roots: dict[str, str]) -> None:
    backfiller = _RecordingBackfiller(failing=set())
    config = FusedMemoryConfig.model_validate({'followup_hold': {'backfill_enabled': False}})

    await server_main._backfill_followup_holds(config, backfiller, startup_roots.values())

    assert backfiller.seen == []


def _run_server() -> ast.AsyncFunctionDef:
    tree = parse_python_module(pathlib.Path(server_main.__file__).resolve())
    found = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == 'run_server'
    ]
    assert len(found) == 1, 'run_server not found in server/main.py'
    return found[0]


def _is_known_projects_values(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'values'
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == '_known_projects_map'
    )


def test_run_server_awaits_the_backfill_after_the_interceptor_starts_and_before_serving() -> None:
    run_server = _run_server()
    backfills = calls_named(run_server, '_backfill_followup_holds')
    awaited = [
        node.value for node in ast.walk(run_server)
        if isinstance(node, ast.Await) and isinstance(node.value, ast.Call)
        and node.value in backfills
    ]
    assert len(backfills) == 1 and awaited == backfills, (
        'run_server must await _backfill_followup_holds exactly once, never spawn it'
    )
    [backfill] = backfills
    assert any(_is_known_projects_values(argument) for argument in backfill.args), (
        'the backfill must walk the _known_projects_map snapshot the re-mint sweep walks'
    )
    starts = [
        call for call in calls_named(run_server, 'start')
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name) and call.func.value.id == 'task_interceptor'
    ]
    servers = calls_named(run_server, 'create_mcp_server')
    assert len(starts) == 2, 'expected task_interceptor.start() in both reconciliation branches'
    assert len(servers) == 1
    assert max(call.lineno for call in starts) < backfill.lineno < servers[0].lineno
