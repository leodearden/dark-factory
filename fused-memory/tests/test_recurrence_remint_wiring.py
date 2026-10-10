"""run_server spawns the recurrence re-mint sweep and tears it down (task 6572).

Structural check over run_server's PARSED body. ``run_server`` binds two
uvicorn ports and constructs every store, so it cannot be driven from a unit
test; a missing spawn would leave broken chains unrepaired in production with
every other test green. The handle's name is derived from the spawn's
assignment target, never spelled out.
"""

from __future__ import annotations

import ast
import pathlib

from _ast_guard import calls_named, parse_python_module

from fused_memory.server import main as server_main


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


def test_run_server_spawns_the_remint_sweep_and_cancels_it_from_a_finally_block():
    run_server = _run_server()
    spawns = [
        node for node in ast.walk(run_server)
        if isinstance(node, ast.Assign | ast.AnnAssign)
        and node.value is not None
        and calls_named(node.value, 'run_remint_sweep_loop')
    ]
    assert len(spawns) == 1, (
        'run_server must spawn the recurrence re-mint sweep exactly once and keep '
        f'the handle; found {len(spawns)} assignments from it'
    )
    spawn = spawns[0]
    bound = spawn.targets if isinstance(spawn, ast.Assign) else [spawn.target]
    targets = [t for t in bound if isinstance(t, ast.Name)]
    assert targets, 'the spawned sweep must be bound to a name, not discarded'
    handle = targets[0].id

    assert any(
        _cancels_and_awaits(node.finalbody, handle)
        for node in ast.walk(run_server)
        if isinstance(node, ast.Try)
    ), f'teardown must cancel AND await `{handle}` from a `finally` block'
