"""run_server builds ONE reconciliation usage gate and hands it to both recon owners (task 11783).

Without the gate, the targeted verifier's CLI calls fall back to the ambient
~/.claude login. Below run_server the gate's path is pinned by behaviour:
test_recon_usage_gate.py (the builder), test_targeted.py and
test_verify_cli_invocation.py (TargetedReconciler -> CodebaseVerifier ->
AgentLoop -> the CLI). ``run_server`` binds two uvicorn ports and constructs
every store, so it cannot be driven from a unit test. That it builds the gate
once and passes that one gate to the harness and to the targeted reconciler is
a structural check over its PARSED body, as in test_invocations_store_wiring.py.
The gate's name is derived from the assignment target, never spelled out.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from _ast_guard import calls_named, parse_python_module

from fused_memory.server import main as server_main

BUILDER = 'build_reconciliation_usage_gate'


def _run_server() -> ast.AsyncFunctionDef:
    tree = parse_python_module(pathlib.Path(server_main.__file__).resolve())
    (run_server,) = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == 'run_server'
    ]
    return run_server


def _recon_usage_gate_name(run_server: ast.AsyncFunctionDef) -> str:
    """The name run_server binds its one ``build_reconciliation_usage_gate(...)`` result to."""
    builds = calls_named(run_server, BUILDER)
    assert len(builds) == 1, (
        f'run_server must call {BUILDER} exactly once, so every recon owner shares one '
        f'gate and its cap state; it calls it {len(builds)} times'
    )
    (build,) = builds
    (assign,) = [
        node for node in ast.walk(run_server)
        if isinstance(node, ast.Assign) and node.value is build
    ]
    (target,) = assign.targets
    assert isinstance(target, ast.Name)
    return target.id


@pytest.mark.parametrize('owner', ['ReconciliationHarness', 'TargetedReconciler'])
def test_run_server_hands_the_recon_owner_its_one_usage_gate(owner: str):
    run_server = _run_server()
    gate = _recon_usage_gate_name(run_server)

    (construction,) = calls_named(run_server, owner)

    usage_gates = [kw.value for kw in construction.keywords if kw.arg == 'usage_gate']
    assert len(usage_gates) == 1, f'{owner}(...) in run_server passes no usage_gate'
    (usage_gate,) = usage_gates
    assert isinstance(usage_gate, ast.Name) and usage_gate.id == gate, (
        f'{owner}(...) in run_server must be given {gate}, the one reconciliation usage gate'
    )
