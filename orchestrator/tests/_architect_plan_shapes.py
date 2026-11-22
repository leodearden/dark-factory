"""The synthetic produced-plan dicts the architect eval tests hand to the plan judge and to the cell.

Imported by bare module name (``orchestrator/tests`` has no ``__init__.py``).
"""

from __future__ import annotations


def well_formed_plan() -> dict:
    """A produced plan satisfying every structural criterion.

    Alternating test/impl TDD steps, files declared, design decisions, reuse
    items, and the ``_finalized_at`` completeness marker the architect's
    ``confirm_plan`` stamps.
    """
    return {
        'steps': [
            {'id': 'step-1', 'type': 'test', 'description': 'RED test for X',
             'status': 'done'},
            {'id': 'step-2', 'type': 'impl', 'description': 'GREEN implement X',
             'status': 'done'},
            {'id': 'step-3', 'type': 'test', 'description': 'RED test for Y',
             'status': 'done'},
            {'id': 'step-4', 'type': 'impl', 'description': 'GREEN implement Y',
             'status': 'done'},
        ],
        'files': ['pkg/mod.py', 'tests/test_mod.py'],
        'design_decisions': [
            {'decision': 'do X the clean way', 'rationale': 'because Y'},
        ],
        'reuse': [
            {'what': 'existing helper', 'where': 'pkg/util.py', 'how': 'call it'},
        ],
        '_finalized_at': '2026-07-19T00:00:00+00:00',
    }


def create_plan_stub() -> dict:
    """The header-only plan ``_create_plan`` writes on the architect's FIRST call.

    Verbatim shape from ``orchestrator/src/orchestrator/mcp/plan_tools.py``
    (``_create_plan``, lines 146-156): a TRUTHY dict carrying ZERO steps. This is
    the artifact a session cap leaves behind when it lands mid-run right after
    ``create_plan`` — the most common mid-run cap shape, since ``create_plan`` is
    the only plan-tools call the architect can reach before the 429.
    """
    return {
        'task_id': 't',
        'title': 'x',
        'analysis': 'a',
        'files': [],
        'prerequisites': [],
        'steps': [],
        'design_decisions': [],
        'reuse': [],
    }


def stub_with_steps(steps) -> dict:
    """``create_plan_stub`` with its ``steps`` replaced (``[]`` vs ``None``)."""
    stub = create_plan_stub()
    stub['steps'] = steps
    return stub
