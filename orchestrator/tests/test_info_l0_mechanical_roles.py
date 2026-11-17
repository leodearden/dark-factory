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
from dataclasses import dataclass
from pathlib import Path

import pytest
from _tree_scan import (
    ORCHESTRATOR_SRC,
    ORCHESTRATOR_SRC_DIR,
    TREE_PASS_XDIST_GROUP,
    ParsedModule,
    TreeAnalysis,
    TreePass,
)
from escalation.classify import INFO_L0_MECHANICAL_ROLES
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

# The family mark and shared group: shared/src/shared/whole_tree_guard_family.py::WHOLE_TREE_SCAN_TEST_TIMEOUT, _tree_scan.py.
pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP),
]


@dataclass(frozen=True)
class _RoleFacts:
    """One file's string-constant bindings and its ``agent_role=`` values, before resolution.

    Resolution folds every file's bindings first, because a role constant may be
    defined in one module and passed in another.
    """

    string_bindings: tuple[tuple[str, str], ...]
    literal_roles: tuple[str, ...]
    role_names: tuple[str, ...]


def _string_constant_bindings(tree: ast.Module) -> tuple[tuple[str, str], ...]:
    """Every ``NAME = '...'`` or ``NAME: T = '...'`` binding in *tree*, at any depth."""
    bindings: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        bindings.extend(
            (target.id, value.value) for target in targets if isinstance(target, ast.Name)
        )
    return tuple(bindings)


def _role_facts(module: ParsedModule) -> _RoleFacts:
    """A string ``agent_role=`` is a literal role; a bare or dotted name is resolved by its trailing name."""
    roles = [
        node.value
        for node in ast.walk(module.tree)
        if isinstance(node, ast.keyword) and node.arg == 'agent_role'
    ]
    return _RoleFacts(
        string_bindings=_string_constant_bindings(module.tree),
        literal_roles=tuple(
            role.value
            for role in roles
            if isinstance(role, ast.Constant) and isinstance(role.value, str)
        ),
        role_names=tuple(
            role.id if isinstance(role, ast.Name) else role.attr
            for role in roles
            if isinstance(role, ast.Name | ast.Attribute)
        ),
    )


ANALYSIS = TreeAnalysis(name=Path(__file__).stem, files=ORCHESTRATOR_SRC, extract=_role_facts)
TREE_ANALYSES = (ANALYSIS,)


@pytest.fixture(scope='module')
def filed_agent_roles(tree_pass: TreePass) -> set[str]:
    """Every string passed as an ``agent_role=`` keyword in the orchestrator src tree."""
    files = tree_pass.facts(ANALYSIS).values()
    bindings: dict[str, set[str]] = {}
    for facts in files:
        for name, value in facts.string_bindings:
            bindings.setdefault(name, set()).add(value)
    return {
        role
        for facts in files
        for role in (
            *facts.literal_roles,
            *(value for name in facts.role_names for value in bindings.get(name, ())),
        )
    }


@pytest.mark.parametrize('role', sorted(INFO_L0_MECHANICAL_ROLES))
def test_registered_role_is_still_filed_by_the_orchestrator(role: str, filed_agent_roles: set[str]):
    assert role in filed_agent_roles, (
        f'{role!r} is in escalation.classify.INFO_L0_MECHANICAL_ROLES but no '
        f'agent_role= under {ORCHESTRATOR_SRC_DIR} files it any more; update the registry '
        'to the filer\'s new role.'
    )


def test_scan_sees_a_role_outside_the_registry(filed_agent_roles: set[str]):
    """Vacuity floor: the bare 'orchestrator' role (the done-step tripwire's) is filed."""
    assert 'orchestrator' in filed_agent_roles
