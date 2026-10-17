"""run_server starts the recurrence re-mint sweep and tears it down (task 6572).

``_start_remint_sweep`` is exercised by behaviour. ``run_server`` binds two
uvicorn ports and constructs every store, so it cannot be driven from a unit
test; that it calls the helper unconditionally and cancels the handle is a
structural check over its PARSED body. The handle's name is derived from the
assignment target, never spelled out.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib

import pytest
import pytest_asyncio
from _ast_guard import calls_named, parse_python_module
from _fm_helpers import poll_until

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.server import main as server_main


class _RecordingRemint:
    def __init__(self, roots: list[str]) -> None:
        self.seen: list[str] = []
        self.swept_all = asyncio.Event()
        self._roots = roots

    async def __call__(self, root: str) -> None:
        self.seen.append(root)
        if set(self._roots) <= set(self.seen):
            self.swept_all.set()


@pytest.fixture
def roots(tmp_path) -> list[str]:
    made = [tmp_path / 'a', tmp_path / 'b']
    for root in made:
        root.mkdir()
    return [str(root) for root in made]


@pytest_asyncio.fixture
async def backend(tmp_path):
    b = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await b.start()
    yield b
    await b.close()


def test_no_sweep_is_started_without_a_task_backend(roots):
    remint = _RecordingRemint(roots)

    assert server_main._start_remint_sweep(None, remint, roots) is None


@pytest.mark.asyncio
async def test_the_started_sweep_visits_every_root_and_cancels_cleanly(backend, roots):
    remint = _RecordingRemint(roots)

    task = server_main._start_remint_sweep(backend, remint, iter(roots))

    assert task is not None
    assert task.get_name() == 'recurrence_remint_sweep'
    await poll_until(remint.swept_all.is_set)
    assert remint.seen[:2] == roots
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def _run_server() -> ast.AsyncFunctionDef | ast.FunctionDef:
    tree = parse_python_module(pathlib.Path(server_main.__file__).resolve())
    run_server = next(
        (
            node for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
            and node.name == 'run_server'
        ),
        None,
    )
    assert run_server is not None, 'run_server not found in server/main.py'
    return run_server


def _cancels_and_awaits(body: list[ast.stmt], handle: str) -> bool:
    cancelled = any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == handle
        for stmt in body
        for call in calls_named(stmt, 'cancel')
    )
    awaited = any(
        isinstance(node, ast.Await)
        and isinstance(node.value, ast.Name)
        and node.value.id == handle
        for stmt in body
        for node in ast.walk(stmt)
    )
    return cancelled and awaited


def test_run_server_starts_the_sweep_unconditionally_and_cancels_it_from_a_finally_block():
    run_server = _run_server()
    spawns = [
        node for node in ast.walk(run_server)
        if isinstance(node, ast.Assign | ast.AnnAssign)
        and node.value is not None
        and calls_named(node.value, '_start_remint_sweep')
    ]
    assert len(spawns) == 1, (
        'run_server must start the recurrence re-mint sweep exactly once and keep '
        f'the handle; found {len(spawns)} assignments from it'
    )
    spawn = spawns[0]
    assert spawn in run_server.body, (
        "the sweep must be started from run_server's own top-level body, not under a "
        'branch: _start_remint_sweep decides itself whether a sweep runs'
    )
    bound = spawn.targets if isinstance(spawn, ast.Assign) else [spawn.target]
    targets = [t for t in bound if isinstance(t, ast.Name)]
    assert targets, 'the started sweep must be bound to a name, not discarded'
    handle = targets[0].id

    assert any(
        _cancels_and_awaits(node.finalbody, handle)
        for node in ast.walk(run_server)
        if isinstance(node, ast.Try)
    ), f'teardown must cancel AND await `{handle}` from a `finally` block'
