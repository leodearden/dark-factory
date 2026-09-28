"""The plan DOCUMENT substrate the two plan-tools markup suites share.

``test_plan_tools_markup_repair`` and ``test_plan_tools_markup_self_name`` both
drive plan-tools' lazy read-repair, so both need the same three things and
neither may own a second copy: a ``TaskArtifacts`` over a temp worktree, a
clean plan document to poison, and the same reading of what landed on disk.

The refusal memo belongs here for a sharper reason than tidiness. It is
process-local by design and therefore shared by every test in a pytest RUN,
across modules — and both suites report refusals on the SAME locator
(``design_decisions[0].rationale``). An autouse clear that lived in only one of
them would make the other's refusal rows depend on collection order.

Kept separate from ``_markup_helpers``, which the verdict-tools suite also
imports: nothing about a plan document belongs in a module that server shares.
"""

from __future__ import annotations

import json

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import plan_tools

#: The clean ``design_decisions[0].decision`` every specimen is built around.
DECISION_PROSE = (
    'Repair the plan lazily on read rather than sweeping the fleet, because a '
    'sweep would have to quiesce every running task first.'
)


# ---------------------------------------------------------------------------
# The fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture()
def plan_artifacts(tmp_path):
    """TaskArtifacts over a temp worktree — mirrors ``test_plan_tools_server``."""
    a = TaskArtifacts(tmp_path)
    a.init('test-1', 'Test task', 'A test')
    return a


@pytest.fixture(autouse=True)
def isolate_the_refusal_memo():
    """Clear ``_REPORTED_REFUSALS`` around every test in an importing module.

    The memo is process-local by design (one plan-tools subprocess per agent
    invocation, so "once per process" is "once per session"), which under pytest
    means one set shared by every test in the run. Clearing it here keeps the
    suite order-independent instead of making a later test depend on whether an
    earlier one happened to report the same locator.
    """
    plan_tools._REPORTED_REFUSALS.clear()
    yield
    plan_tools._REPORTED_REFUSALS.clear()


def corrupt_plan(**overrides) -> dict:
    """Return a complete, VALID plan dict whose fields can then be poisoned.

    Every call builds a fresh, independent document (no shared mutable state),
    so a test may poison ``plan['design_decisions'][0]['rationale']`` in place.
    Keyword *overrides* replace whole top-level keys, which is how a test swaps
    in its own collection (e.g. four decisions instead of the default two).

    The default document is entirely CLEAN: nothing here trips ``detect()``, so
    any fact a test observes came from the field it poisoned.
    """
    plan: dict = {
        'task_id': 'test-1',
        'title': 'A test plan',
        'analysis': 'Clean analysis prose describing the approach.',
        'files': ['orchestrator/src/orchestrator/mcp/plan_tools.py'],
        'prerequisites': [
            {
                'id': 'pre-1',
                'description': 'Clean prerequisite prose.',
                'status': 'pending',
                'commit': None,
                'tests': [],
            },
        ],
        'steps': [
            {
                'id': 'step-1',
                'type': 'test',
                'description': 'Clean step prose for the first step.',
                'status': 'pending',
                'commit': None,
            },
            {
                'id': 'step-2',
                'type': 'impl',
                'description': 'Clean step prose for the second step.',
                'status': 'pending',
                'commit': None,
            },
        ],
        'design_decisions': [
            {'decision': DECISION_PROSE, 'rationale': 'Clean rationale prose.'},
            {'decision': 'A second clean decision.', 'rationale': 'A second clean rationale.'},
        ],
        'reuse': [
            {
                'what': 'The shared detector',
                'where': 'shared/src/shared/toolcall_markup.py',
                'how': 'Clean reuse prose.',
            },
            {
                'what': 'The plan artifact reader',
                'where': 'orchestrator/src/orchestrator/artifacts.py',
                'how': 'A second clean reuse prose.',
            },
        ],
    }
    plan.update(overrides)
    return plan


def on_disk(artifacts) -> dict:
    return json.loads((artifacts.root / 'plan.json').read_text())
