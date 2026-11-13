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
    'is_one_step',
]


class ImplementerBriefVariant(StrEnum):
    """Whether one implementer invocation works the whole plan or exactly one step.

    Each member IS the YAML/stored value. Every policy on this axis branches
    on :func:`is_one_step`, so anything that is not the ``ONE_STEP`` member
    behaves as ``WHOLE_PLAN``.
    """

    WHOLE_PLAN = 'whole-plan'
    ONE_STEP = 'one-step'


ONE_STEP_ITERATION_SLACK = 3
ONE_STEP_BOUND_DIVISOR = 3


def is_one_step(variant: ImplementerBriefVariant) -> bool:
    """The one membership rule for the brief, the iteration cap and the bounds: identity with ``ONE_STEP``.

    A value that skipped validation (a raw ``'one-step'`` str from an
    unvalidated ``model_copy(update=...)``, a test double's config attribute)
    is not the member, so all three agree on ``WHOLE_PLAN`` for it.
    """
    return variant is ImplementerBriefVariant.ONE_STEP


def execute_iteration_cap(
    variant: ImplementerBriefVariant,
    max_execute_iterations: int,
    read_plan: Callable[[], Mapping[str, Any]],
) -> int:
    """The EXECUTE loop's iteration bound; *read_plan* is called only under one-step.

    One-step runs one iteration per plan item, so its bound is every item the
    plan holds (prerequisites and steps, done or not) plus a fixed slack.
    """
    if not is_one_step(variant):
        return max_execute_iterations
    plan = read_plan()
    items = [*plan.get('prerequisites', []), *plan.get('steps', [])]
    return sum(isinstance(item, dict) for item in items) + ONE_STEP_ITERATION_SLACK


def invocation_bounds(
    variant: ImplementerBriefVariant,
    role_name: str,
    budget_usd: float,
    max_turns: int,
) -> tuple[float, int]:
    """The ``(budget_usd, max_turns)`` one invocation of *role_name* is sent."""
    if not is_one_step(variant) or role_name != 'implementer':
        return budget_usd, max_turns
    return (
        budget_usd / ONE_STEP_BOUND_DIVISOR,
        math.ceil(max_turns / ONE_STEP_BOUND_DIVISOR),
    )
