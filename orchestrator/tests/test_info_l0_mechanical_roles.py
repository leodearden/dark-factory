"""Referential integrity of ``escalation.classify.INFO_L0_MECHANICAL_ROLES``
against the orchestrator filers it names (plans/info-l0-disposition-router-prd.md D8).

The registry repeats, as strings, the ``agent_role`` each mechanical filer
stamps on its info L0s.  A filer that renames its role drops out of the
status-info leg and lands on the curator instead (D8 fail-loud); this test
fails at commit time instead, by requiring every registered role to still be an
``agent_role=`` value somewhere under orchestrator/src/orchestrator.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT
from escalation.classify import INFO_L0_MECHANICAL_ROLES

# rglob()s and ast.parse()s every *.py under the orchestrator src tree, so this
# module belongs to the whole-tree-scanner family (_orch_helpers.py::WHOLE_TREE_SCAN_TEST_TIMEOUT).
pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_ORCH_SRC_DIR = Path(__file__).resolve().parent.parent / 'src' / 'orchestrator'


def _string_constants_by_name(trees: list[ast.Module]) -> dict[str, set[str]]:
    """Every ``NAME = '...'`` or ``NAME: T = '...'`` binding, at any depth, by name."""
    bindings: dict[str, set[str]] = {}
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    bindings.setdefault(target.id, set()).add(value.value)
    return bindings


def _resolved_strings(value: ast.expr, bindings: dict[str, set[str]]) -> set[str]:
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return {value.value}
    if isinstance(value, ast.Name):
        return bindings.get(value.id, set())
    if isinstance(value, ast.Attribute):
        return bindings.get(value.attr, set())
    return set()


@pytest.fixture(scope='module')
def filed_agent_roles() -> set[str]:
    """Every string passed as an ``agent_role=`` keyword in the orchestrator src tree."""
    trees = [
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for path in sorted(_ORCH_SRC_DIR.rglob('*.py'))
    ]
    bindings = _string_constants_by_name(trees)
    return {
        role
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and node.arg == 'agent_role'
        for role in _resolved_strings(node.value, bindings)
    }


@pytest.mark.parametrize('role', sorted(INFO_L0_MECHANICAL_ROLES))
def test_registered_role_is_still_filed_by_the_orchestrator(role: str, filed_agent_roles: set[str]):
    assert role in filed_agent_roles, (
        f'{role!r} is in escalation.classify.INFO_L0_MECHANICAL_ROLES but no '
        f'agent_role= under {_ORCH_SRC_DIR} files it any more; update the registry '
        'to the filer\'s new role.'
    )


def test_scan_sees_a_role_outside_the_registry(filed_agent_roles: set[str]):
    """Vacuity floor: the bare 'orchestrator' role (the done-step tripwire's) is filed."""
    assert 'orchestrator' in filed_agent_roles
