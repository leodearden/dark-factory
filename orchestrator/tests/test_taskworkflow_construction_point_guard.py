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
from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT

import orchestrator

# Ceiling rationale: _orch_helpers.py::WHOLE_TREE_SCAN_TEST_TIMEOUT.
pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

# ``TaskWorkflow(`` not preceded by a word char or dot — a construction call,
# not ``build_workflow`` / ``class TaskWorkflow:`` / an attribute access.
_CONSTRUCT_RE = re.compile(r"(?<![\w.])TaskWorkflow\(")


def _orchestrator_src_root() -> Path:
    return Path(orchestrator.__file__).resolve().parent


def test_single_taskworkflow_construction_point():
    """Only ``workflow.py`` may construct ``TaskWorkflow(`` in production source."""
    src_root = _orchestrator_src_root()
    offenders: set[str] = set()
    for path in src_root.rglob('*.py'):
        text = path.read_text(encoding='utf-8', errors='ignore')
        for line in text.splitlines():
            code = line.split('#', 1)[0]  # drop trailing comment
            if _CONSTRUCT_RE.search(code):
                offenders.add(path.relative_to(src_root).as_posix())
                break
    assert offenders == {'workflow.py'}, (
        'Direct TaskWorkflow(...) construction is only permitted in workflow.py '
        '(the build_workflow() factory body, the single construction point). '
        f'Offending files: {sorted(offenders)}. Route construction through '
        'build_workflow().'
    )
