"""The implementer brief variant's closed vocabulary and the bounds each member implies (plans/live-shadow-eval-prd.md decision 14)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any

__all__ = [
    'ONE_STEP_BOUND_DIVISOR',
    'ONE_STEP_ITERATION_SLACK',
    'ImplementerBriefVariant',
    'execute_iteration_cap',
    'invocation_bounds',
]


class ImplementerBriefVariant(StrEnum):
    """Whether one implementer invocation works the whole plan or exactly one step.

    Each member IS the YAML/stored value. Every policy below branches on an
    identity test against ``ONE_STEP``, so anything that is not that member
    (a test double's config attribute included) behaves as ``WHOLE_PLAN``.
    """

    WHOLE_PLAN = 'whole-plan'
    ONE_STEP = 'one-step'


ONE_STEP_ITERATION_SLACK = 3
ONE_STEP_BOUND_DIVISOR = 3


def execute_iteration_cap(
    variant: object,
    max_execute_iterations: int,
    read_plan: Callable[[], Mapping[str, Any]],
) -> int:
    """The EXECUTE loop's iteration bound; *read_plan* is called only under one-step.

    One-step runs one iteration per plan item, so its bound is every item the
    plan holds (prerequisites and steps, done or not) plus a fixed slack.
    """
    if variant is not ImplementerBriefVariant.ONE_STEP:
        return max_execute_iterations
    plan = read_plan()
    items = [*plan.get('prerequisites', []), *plan.get('steps', [])]
    return sum(isinstance(item, dict) for item in items) + ONE_STEP_ITERATION_SLACK


def invocation_bounds(
    variant: object,
    role_name: str,
    budget_usd: float,
    max_turns: int,
) -> tuple[float, int]:
    """The ``(budget_usd, max_turns)`` one invocation of *role_name* is sent."""
    if variant is not ImplementerBriefVariant.ONE_STEP or role_name != 'implementer':
        return budget_usd, max_turns
    return (
        budget_usd / ONE_STEP_BOUND_DIVISOR,
        math.ceil(max_turns / ONE_STEP_BOUND_DIVISOR),
    )
