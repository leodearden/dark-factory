"""run_server starts the embedder endpoint probe and hands it to get_status.

``create_mcp_server`` adds the ``embedder`` and ``config_overlay`` blocks only
when it is given a probe, so a dropped wire would silently remove both from
the deployed get_status. ``run_server`` binds uvicorn ports and constructs
every store, so it cannot be driven from a unit test; the wiring is a
structural check over its PARSED body. The probe's name is derived from the
assignment target, never spelled out.
"""

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


def _probe_name(run_server: ast.AsyncFunctionDef | ast.FunctionDef) -> str:
    constructions = [
        stmt for stmt in run_server.body
        if isinstance(stmt, ast.Assign)
        and calls_named(stmt.value, 'EmbedderEndpointProbe')
    ]
    assert len(constructions) == 1, (
        "run_server's own top-level body must construct exactly one "
        f'EmbedderEndpointProbe and bind it; found {len(constructions)}'
    )
    [target] = constructions[0].targets
    assert isinstance(target, ast.Name), 'the probe must be bound to a plain name'
    return target.id


def test_run_server_starts_the_probe_unconditionally():
    run_server = _run_server()
    probe = _probe_name(run_server)

    starts = [
        call
        for stmt in run_server.body if isinstance(stmt, ast.Expr)
        for call in calls_named(stmt, 'start')
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == probe
    ]
    assert len(starts) == 1, (
        f'run_server must call `{probe}.start()` once from its own top-level body'
    )


def test_run_server_passes_the_probe_to_the_mcp_server():
    run_server = _run_server()
    probe = _probe_name(run_server)

    passed = [
        call for call in calls_named(run_server, 'create_mcp_server')
        if any(
            keyword.arg == 'embedder_probe'
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == probe
            for keyword in call.keywords
        )
    ]
    assert passed, f'create_mcp_server must be called with embedder_probe={probe}'
