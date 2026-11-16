"""Structural guard: the merge-phase scaffolding has ONE home (task 6780).

Both consumers take it from orchestrator/tests/_merge_phase_fixtures.py (SPOT,
heuristic 11), and neither uses a sibling test module as an import source, so
each test file makes sense on its own (heuristic 13).
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

# A module import, not a name import: importing a fixture by name would
# register it in this test module.
import _merge_phase_fixtures as fixtures
import pytest

SHARED = (
    'git_repo',
    'config',
    'git_ops',
    'task_assignment',
    'PLAN',
    '_build_workflow',
    '_make_advanced_worktree',
    '_make_silent_steward',
    '_stub_merge_internals',
    '_submit_gating_l0',
    '_time_the_wait',
    '_wire_resolve_callback',
)

CONSUMERS = ('test_workflow_merge_gating_strand', 'test_info_l2_never_gates')


@pytest.mark.parametrize('name', CONSUMERS)
def test_each_consumer_binds_the_support_modules_objects(name: str) -> None:
    consumer = importlib.import_module(name)
    bound = [n for n in SHARED if hasattr(consumer, n)]
    assert bound, (
        f'{name} binds none of the shared merge-phase scaffolding, so this '
        'check asserted nothing.'
    )
    # Identity, never equality: an equal re-paste passes a value comparison.
    separate = [n for n in bound if getattr(consumer, n) is not getattr(fixtures, n)]
    assert not separate, (
        f'{name} binds its own copies of {separate}. Their one home is '
        'orchestrator/tests/_merge_phase_fixtures.py; import them from there.'
    )


@pytest.mark.parametrize('name', CONSUMERS)
def test_no_consumer_imports_a_sibling_test_module(name: str) -> None:
    tree = ast.parse((Path(__file__).parent / f'{name}.py').read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    sibling_tests = sorted(m for m in imported if m.startswith('test_'))
    assert not sibling_tests, (
        f'{name} imports from the test module(s) {sibling_tests}. Import '
        'shared scaffolding from a _-prefixed support module instead.'
    )
