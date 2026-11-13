"""Guard: ``workflow.py`` is the only production-source file constructing ``TaskWorkflow(``.

PRD eval-framework-revival §α, Invariant P2 / Boundary test B2.  ``build_workflow()``
is the SINGLE construction point for ``TaskWorkflow``, so a new mandatory
constructor parameter breaks both dispatch sites at once rather than drifting.
This is the one static sweep for that invariant.  test_workflow_factory.py checks
that both dispatch sites route through the factory, and
test_eval_boundary_suite.py::test_b2_factory_single_construction_point proves at
runtime that a new mandatory parameter breaks it.

Scoped to orchestrator/src/orchestrator/ because ~70 test modules legitimately
construct TaskWorkflow directly and would false-fail a repo-wide guard.
"""

from __future__ import annotations

import re
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
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

# The family mark and shared group: shared/src/shared/whole_tree_guard_family.py::WHOLE_TREE_SCAN_TEST_TIMEOUT, _tree_scan.py.
pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP),
]

# ``TaskWorkflow(`` not preceded by a word char or dot — a construction call,
# not ``build_workflow`` / ``class TaskWorkflow:`` / an attribute access.
_CONSTRUCT_RE = re.compile(r"(?<![\w.])TaskWorkflow\(")


def _constructs_taskworkflow(module: ParsedModule) -> bool:
    return any(
        _CONSTRUCT_RE.search(line.split('#', 1)[0])  # drop trailing comment
        for line in module.source.splitlines()
    )


ANALYSIS = TreeAnalysis(
    name=Path(__file__).stem, files=ORCHESTRATOR_SRC, extract=_constructs_taskworkflow,
)
TREE_ANALYSES = (ANALYSIS,)


def test_single_taskworkflow_construction_point(tree_pass: TreePass):
    """Only ``workflow.py`` may construct ``TaskWorkflow(`` in production source."""
    offenders = {
        path.relative_to(ORCHESTRATOR_SRC_DIR).as_posix()
        for path, constructs in tree_pass.facts(ANALYSIS).items()
        if constructs
    }
    assert offenders == {'workflow.py'}, (
        'Direct TaskWorkflow(...) construction is only permitted in workflow.py '
        '(the build_workflow() factory body, the single construction point). '
        f'Offending files: {sorted(offenders)}. Route construction through '
        'build_workflow().'
    )
