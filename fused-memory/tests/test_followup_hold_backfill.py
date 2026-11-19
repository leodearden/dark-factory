"""The startup backfill that holds the pending follow-up stock behind the holder.

plans/follow-up-filing-shape-prd.md §3 decision 12 and boundary row 17. Rows are seeded straight into
the store, as the stock filed before the hold existed is, and read back through ``get_task``.
"""

from __future__ import annotations

import ast
import json
import logging
import pathlib
from collections.abc import Sequence
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
_MARKER = {'project_id': 'proj', 'minted_at': '2026-11-08T00:00:00+00:00'}


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
    prior = {task_id: await stack.dependencies(task_id) for task_id in matching}
    untouched_before = {task_id: await stack.row(task_id) for task_id in untouched}

    count = await stack.interceptor.backfill_followup_holds(stack.root, known_roots=[stack.root])

    assert count == 3
    [holder] = await stack.holder_ids()
    for task_id in matching:
        assert await stack.dependencies(task_id) == [*prior[task_id], holder], task_id
        assert str((await stack.metadata(task_id))[FOLLOWUP_HOLD_METADATA_KEY]['holder']) == holder
    for task_id, before in untouched_before.items():
        assert await stack.row(task_id) == before, task_id
    assert await stack.events(EventType.followup_hold_backfilled) == [
        {'project_id': resolve_project_id(stack.root), 'count': 3},
    ]
    assert await stack.events(EventType.task_modified) == []
    assert await stack.events(EventType.task_created) == [
        {'operation': 'add_task', 'task_id': holder, 'source': 'followup-holder-mint'},
    ]


@pytest.mark.asyncio
async def test_a_second_backfill_adds_nothing_and_still_reports(stack: Stack) -> None:
    matching, _untouched = await _seed_stock(stack)
    await stack.interceptor.backfill_followup_holds(stack.root, known_roots=[stack.root])
    held = {task_id: await stack.dependencies(task_id) for task_id in matching}

    count = await stack.interceptor.backfill_followup_holds(stack.root, known_roots=[stack.root])

    assert count == 0
    for task_id in matching:
        assert await stack.dependencies(task_id) == held[task_id], task_id
    assert [event['count'] for event in await stack.events(EventType.followup_hold_backfilled)] == [3, 0]


@pytest.mark.asyncio
async def test_with_nothing_to_hold_no_holder_is_minted(stack: Stack) -> None:
    await stack.seed('Medium agent follow-up', dict(_FOLLOWUP), priority='medium')

    count = await stack.interceptor.backfill_followup_holds(stack.root, known_roots=[stack.root])

    assert count == 0
    assert await stack.holder_ids() == []
    assert await stack.events(EventType.followup_hold_backfilled) == [
        {'project_id': resolve_project_id(stack.root), 'count': 0},
    ]


async def _seed_depending(stack: Stack, title: str, depends_on: str, **fields: str) -> str:
    dependent = await stack.seed(title, {}, **fields)
    await stack.backend.add_dependency(dependent, depends_on, stack.root)
    return dependent


@pytest.mark.asyncio
async def test_the_backfill_leaves_unheld_every_followup_an_unheld_task_waits_on(stack: Stack) -> None:
    def low_followup(title: str) -> dict[str, Any]:
        return {'title': title, 'metadata': dict(_FOLLOWUP), 'priority': 'low'}

    x, x2, x5, a, b, x3, x4 = [
        await stack.seed(**low_followup(f'Low follow-up {name}'))
        for name in ('X', 'X2', 'X5', 'A', 'B', 'X3', 'X4')
    ]
    await _seed_depending(stack, 'Pending task waiting on X', x)
    await stack.backend.add_dependency(x, x2, stack.root)
    await _seed_depending(stack, 'In-progress task waiting on X5', x5, status='in-progress')
    await stack.backend.add_dependency(a, b, stack.root)
    await _seed_depending(stack, 'Deferred task on X3', x3, status='deferred')
    await _seed_depending(stack, 'Done task on X4', x4, status='done')
    waited_on_before = {task_id: await stack.row(task_id) for task_id in (x, x2, x5)}

    count = await stack.interceptor.backfill_followup_holds(stack.root, known_roots=[stack.root])

    assert count == 4
    [holder] = await stack.holder_ids()
    for task_id in (a, b, x3, x4):
        assert holder in await stack.dependencies(task_id), task_id
        assert str((await stack.metadata(task_id))[FOLLOWUP_HOLD_METADATA_KEY]['holder']) == holder, task_id
    for task_id, before in waited_on_before.items():
        assert await stack.row(task_id) == before, task_id
    assert [event['count'] for event in await stack.events(EventType.followup_hold_backfilled)] == [4]


async def _seed_dependent_in(
    stack: Stack, project_root: str, title: str, depends_on: str, **fields: str,
) -> None:
    created = await stack.backend.add_task(project_root=project_root, title=title, metadata='{}', **fields)
    await stack.backend.add_dependency(str(created['id']), depends_on, project_root)


@pytest.mark.asyncio
async def test_a_followup_another_project_waits_on_is_not_held(stack: Stack) -> None:
    other_root = str(pathlib.Path(stack.root).parent / 'other')
    this_project = resolve_project_id(stack.root)
    x = await stack.seed('Low follow-up X', dict(_FOLLOWUP), priority='low')
    z = await stack.seed('Low follow-up Z', dict(_FOLLOWUP), priority='low')
    await _seed_dependent_in(stack, other_root, 'Pending task waiting on X', f'{this_project}:{x}')
    await _seed_dependent_in(
        stack, other_root, 'Deferred task on Z', f'{this_project}:{z}', status='deferred',
    )
    x_before = await stack.row(x)

    count = await stack.interceptor.backfill_followup_holds(
        stack.root, known_roots=[stack.root, other_root],
    )

    assert count == 1
    [holder] = await stack.holder_ids()
    assert holder in await stack.dependencies(z)
    assert await stack.row(x) == x_before


@pytest.mark.asyncio
async def test_a_followup_already_behind_the_holder_without_a_record_is_left_alone(stack: Stack) -> None:
    holder = await stack.seed('Follow-up holder (proj)', {FOLLOWUP_HOLDER_METADATA_KEY: _MARKER})
    edged = await stack.seed('Low follow-up already behind the holder', dict(_FOLLOWUP), priority='low')
    await stack.backend.add_dependency(edged, holder, stack.root)
    edged_before = await stack.row(edged)

    count = await stack.interceptor.backfill_followup_holds(stack.root, known_roots=[stack.root])

    assert count == 0
    assert await stack.row(edged) == edged_before
    assert await stack.holder_ids() == [holder]


@pytest.mark.asyncio
async def test_an_unreadable_known_project_holds_nothing_and_reports_nothing(stack: Stack) -> None:
    unreadable_root = pathlib.Path(stack.root).parent / 'unreadable'
    unreadable_root.write_text('a file where a project directory should be')
    followup = await stack.seed('Low follow-up X', dict(_FOLLOWUP), priority='low')
    followup_before = await stack.row(followup)

    with pytest.raises(OSError):
        await stack.interceptor.backfill_followup_holds(
            stack.root, known_roots=[stack.root, str(unreadable_root)],
        )

    assert await stack.row(followup) == followup_before
    assert await stack.holder_ids() == []
    assert await stack.events(EventType.followup_hold_backfilled) == []


@pytest.mark.asyncio
async def test_an_external_dep_naming_this_project_unnormalised_still_counts_as_a_waiter(stack: Stack) -> None:
    other_root = str(pathlib.Path(stack.root).parent / 'other')
    followup = await stack.seed('Low follow-up X', dict(_FOLLOWUP), priority='low')
    unnormalised = f'{resolve_project_id(stack.root).upper()}:{followup}'
    await stack.backend.add_task(
        project_root=other_root, title='Pending task waiting on X',
        metadata=json.dumps({'external_deps': [unnormalised]}),
    )
    followup_before = await stack.row(followup)

    count = await stack.interceptor.backfill_followup_holds(
        stack.root, known_roots=[stack.root, other_root],
    )

    assert count == 0
    assert await stack.row(followup) == followup_before


def test_the_backfill_is_enabled_by_default() -> None:
    assert FusedMemoryConfig().followup_hold.backfill_enabled is True


def test_the_backfill_can_be_switched_off() -> None:
    config = FusedMemoryConfig.model_validate({'followup_hold': {'backfill_enabled': False}})

    assert config.followup_hold.backfill_enabled is False


class _RecordingBackfiller:
    def __init__(self, failing: set[str]) -> None:
        self.seen: list[str] = []
        self.known_roots: list[list[str]] = []
        self._failing = failing

    async def backfill_followup_holds(self, project_root: str, *, known_roots: Sequence[str]) -> int:
        self.seen.append(project_root)
        self.known_roots.append(list(known_roots))
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

    await server_main.backfill_followup_holds_at_startup(FusedMemoryConfig(), backfiller, startup_roots.values())

    real_roots = [startup_roots['first'], startup_roots['broken'], startup_roots['last']]
    assert backfiller.seen == real_roots
    assert backfiller.known_roots == [real_roots] * len(real_roots)
    warned = [record for record in caplog.records if record.levelno == logging.WARNING]
    failed = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert any(startup_roots['phantom'] in record.getMessage() for record in warned), warned
    assert [startup_roots['broken'] in record.getMessage() for record in failed] == [True], failed


@pytest.mark.asyncio
async def test_a_disabled_backfill_touches_no_project(startup_roots: dict[str, str]) -> None:
    backfiller = _RecordingBackfiller(failing=set())
    config = FusedMemoryConfig.model_validate({'followup_hold': {'backfill_enabled': False}})

    await server_main.backfill_followup_holds_at_startup(config, backfiller, startup_roots.values())

    assert backfiller.seen == []


def _run_server() -> ast.AsyncFunctionDef:
    tree = parse_python_module(pathlib.Path(server_main.__file__).resolve())
    found = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == 'run_server'
    ]
    assert len(found) == 1, 'run_server not found in server/main.py'
    return found[0]


def test_run_server_awaits_the_backfill_after_the_interceptor_starts_and_before_serving() -> None:
    run_server = _run_server()
    backfills = calls_named(run_server, 'backfill_followup_holds_at_startup')
    awaited = [
        node.value for node in ast.walk(run_server)
        if isinstance(node, ast.Await) and isinstance(node.value, ast.Call)
        and node.value in backfills
    ]
    assert len(backfills) == 1 and awaited == backfills, (
        'run_server must await backfill_followup_holds_at_startup exactly once, never spawn it'
    )
    [backfill] = backfills
    [remint_sweep] = calls_named(run_server, '_start_remint_sweep')
    assert ast.dump(backfill.args[-1]) == ast.dump(remint_sweep.args[-1]), (
        'the backfill must walk the same known-project roots the re-mint sweep walks'
    )
    starts = [
        call for call in calls_named(run_server, 'start')
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name) and call.func.value.id == 'task_interceptor'
    ]
    servers = calls_named(run_server, 'create_mcp_server')
    assert starts, 'expected run_server to start the task interceptor'
    assert len(servers) == 1
    assert max(call.lineno for call in starts) < backfill.lineno < servers[0].lineno
