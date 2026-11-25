"""run_server hands its invocations store to recon, so recon's LLM calls land rows (task 10414).

Every owner below run_server is pinned by behaviour: the stages, the judge and
the harness (test_stages.py, test_judge.py, test_harness.py) and the targeted
reconciler (test_targeted.py). ``run_server`` binds two uvicorn ports and
constructs every store, so it cannot be driven from a unit test. That it passes
the store to the harness and to the targeted reconciler is a structural check
over its PARSED body, as in test_recurrence_remint_wiring.py. The store's name
is derived from the assignment target, never spelled out.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from _ast_guard import calls_named, parse_python_module

from fused_memory.server import main as server_main


def _run_server() -> ast.AsyncFunctionDef:
    tree = parse_python_module(pathlib.Path(server_main.__file__).resolve())
    (run_server,) = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == 'run_server'
    ]
    return run_server


def _invocations_store_name(run_server: ast.AsyncFunctionDef) -> str:
    """The name run_server binds the store ``_setup_curator_usage_gate`` opens to."""
    (setup,) = calls_named(run_server, '_setup_curator_usage_gate')
    (assign,) = [
        node for node in ast.walk(run_server)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Await)
        and node.value.value is setup
    ]
    (target,) = assign.targets
    assert isinstance(target, ast.Tuple)
    store = target.elts[0]
    assert isinstance(store, ast.Name)
    return store.id


@pytest.mark.parametrize('owner', ['ReconciliationHarness', 'TargetedReconciler'])
def test_run_server_hands_the_recon_owner_its_invocations_store(owner: str):
    run_server = _run_server()
    store = _invocations_store_name(run_server)

    (construction,) = calls_named(run_server, owner)

    cost_stores = [kw.value for kw in construction.keywords if kw.arg == 'cost_store']
    assert len(cost_stores) == 1, f'{owner}(...) in run_server passes no cost_store'
    (cost_store,) = cost_stores
    assert isinstance(cost_store, ast.Name) and cost_store.id == store, (
        f'{owner}(...) in run_server must be given {store}, the invocations store'
    )
