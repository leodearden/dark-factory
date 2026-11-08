"""Answers whether a plan artifact, or a persisted eval cell, carries a plan.

The rationale for each predicate lives in docs/plan-scoring-and-judge.md
("The plan-production predicate (task 3302)" and "The anti-fabrication floor
applies uniformly (task 3303)").
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeGuard


def is_scorable_plan(plan: object) -> TypeGuard[dict]:
    """True iff *plan* is a dict carrying at least one step.

    Absent, empty, non-dict and the header-only stub ``create_plan`` writes on
    the architect's first plan-tools call (truthy, ``steps: []``) are all NOT
    plans. ``judge.score_plan_structure``'s ``0.0`` short-circuit, the plan
    judge's refusal and ``run_architect_eval``'s cap-taint gate all call this
    one function, so they cannot disagree about what a plan is. It is a
    different question from the ``has_steps`` rubric criterion, which scores a
    real plan.
    """
    return isinstance(plan, dict) and bool(plan.get('steps'))


def plan_confirmed(plan: dict) -> bool:
    """True when the plan carries the ``_finalized_at`` completeness marker.

    ``confirm_plan`` (and ``artifacts.stamp_plan_provenance``) stamp it once the
    architect finalizes a complete plan.
    """
    return bool(plan.get('_finalized_at'))


def produced_a_plan(metrics: Mapping[str, Any]) -> bool:
    """True iff a persisted cell's ``plan_steps`` is positive.

    The metrics-dict twin of :func:`is_scorable_plan`: ``run_architect_eval``
    persists ``plan_steps = len(plan.get('steps') or [])``. Never decided from
    ``plan_quality`` (the number being decided) nor ``outcome == 'done'``. A
    missing key and an explicit ``None`` both mean "no plan".
    """
    return int(metrics.get('plan_steps') or 0) > 0
