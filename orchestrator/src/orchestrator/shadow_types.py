"""The closed column vocabularies of a shadow-eval cell (plans/live-shadow-eval-prd.md C1).

Each member IS the value stored in its ``shadow_cells`` TEXT column and is
read back with ``Enum(raw)``. Stdlib-only, so ``run_store``, ``config`` and
``orchestrator.evals`` all import it downward.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    'CostBasis',
    'PlanSource',
    'ReferenceKind',
    'ShadowCellState',
    'ShadowShape',
]


class ShadowShape(StrEnum):
    """The closed vocabulary of shadow-cell shapes (C1, ``shadow_cells.shape``).

    A ``StrEnum`` rather than a bare ``str`` parameter: the vocabulary is
    closed, so a member IS the value stored in the TEXT column and read back
    with ``ShadowShape(raw)``, while the plan rule below lives beside the
    vocabulary it constrains instead of being re-derived at each call site.

    The vocabulary lives here so run_store, config and evals all import it downward.
    """

    IMPLEMENTER = 'implementer'
    ARCHITECT = 'architect'
    ARCHITECT_CONSEQUENCE = 'architect-consequence'
    END_TO_END = 'end-to-end'

    @property
    def requires_plan(self) -> bool:
        """Whether a cell of this shape must be built with a plan.

        Tracks C1's ``plan_source`` column: ``'production'`` for
        ``implementer`` (the cell runs production's own accepted plan),
        ``'candidate'`` for ``architect-consequence`` (consequence leg 2, the
        incumbent implementer run over the candidate architect's plan), and
        NULL for the two shapes whose architect is live — ``architect`` (the
        leg-1 architect cell) and ``end-to-end`` — neither of whose runners
        reads a plan at all.

        Plan presence is therefore a total function of shape, which is what
        lets ``build_live_fixture`` refuse in both directions rather than
        letting a stray plan be silently ignored.
        """
        return self in (ShadowShape.IMPLEMENTER, ShadowShape.ARCHITECT_CONSEQUENCE)


class CostBasis(StrEnum):
    """How an arm is billed: the closed cost-basis vocabulary of live-shadow PRD decision 16."""

    METERED = 'metered'
    SUBSCRIPTION = 'subscription'


class ShadowCellState(StrEnum):
    """C1's cell lifecycle, in order (``shadow_cells.state``)."""

    SAMPLED = 'sampled'
    OPENED = 'opened'
    RUNNING = 'running'
    AWAITING_REFERENCE = 'awaiting_reference'
    SETTLED = 'settled'
    CAP_EXCLUDED = 'cap_excluded'
    EXPIRED = 'expired'
    FAILED = 'failed'

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STATES

    @property
    def requires_reason(self) -> bool:
        """A cell that ended without settling must say why (C1)."""
        return self.is_terminal and self is not ShadowCellState.SETTLED


_TERMINAL_STATES = frozenset({
    ShadowCellState.SETTLED,
    ShadowCellState.CAP_EXCLUDED,
    ShadowCellState.EXPIRED,
    ShadowCellState.FAILED,
})


class ReferenceKind(StrEnum):
    """What production outcome a cell is scored against (``shadow_cells.reference_kind``)."""

    LANDED = 'landed'
    NONE = 'none'


class PlanSource(StrEnum):
    """Whose plan a plan-consuming cell executed (``shadow_cells.plan_source``)."""

    PRODUCTION = 'production'
    CANDIDATE = 'candidate'
