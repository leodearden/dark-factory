"""run_server hands its invocations store to every consumer: recon and the WAL checkpoint loop.

Recon's LLM calls land rows in the store (task 10414), and the periodic WAL
checkpoint loop truncates its WAL (task 11880). Every owner below run_server is
pinned by behaviour: the stages, the judge and the harness (test_stages.py,
test_judge.py, test_harness.py), the targeted reconciler (test_targeted.py) and
checkpoint-target collection (test_periodic_checkpoint.py). ``run_server`` binds
two uvicorn ports and constructs every store, so it cannot be driven from a unit
test. That it passes the store to each consumer is a structural check over its
PARSED body, as in test_recurrence_remint_wiring.py. The store's name is
derived from the assignment target, never spelled out.
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


@pytest.mark.parametrize(
    ('callee', 'keyword'),
    [
        ('ReconciliationHarness', 'cost_store'),
        ('TargetedReconciler', 'cost_store'),
        ('_collect_checkpoint_targets', 'invocations_store'),
    ],
)
def test_run_server_hands_each_consumer_its_invocations_store(callee: str, keyword: str):
    run_server = _run_server()
    store = _invocations_store_name(run_server)

    (call,) = calls_named(run_server, callee)

    values = [kw.value for kw in call.keywords if kw.arg == keyword]
    assert len(values) == 1, f'{callee}(...) in run_server passes no {keyword}'
    (value,) = values
    assert isinstance(value, ast.Name) and value.id == store, (
        f'{callee}(...) in run_server must be given {keyword}={store}, the invocations store'
    )
