"""What an eval cell refuses, so that it writes no production state.

An eval cell runs a real ``TaskWorkflow`` against fixture worktrees. It is built
with none of the stores through which a production workflow writes, and a store
handed to the cell factory is refused before any workflow exists
(plans/review-all-s1-one-escalation-filing-entry-point-prd.md §Contracts C11).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

from escalation.queue import EscalationQueue
from shared.cost_store import CostStore

from orchestrator.event_store import EventStore


class EvalHermeticityError(ValueError):
    """An eval cell was handed a store through which it would write production state."""

    def __init__(self, field: str, value: object) -> None:
        self.field = field
        super().__init__(
            f'eval cell handed {field}={value!r}: an eval cell is built with no '
            f'production store, so it writes no production state'
        )


@dataclass(frozen=True)
class WorkflowStores:
    """The ``build_workflow`` collaborators through which a ``TaskWorkflow`` writes production state.

    Fields are named as ``build_workflow``'s own keyword arguments. The
    workflow derives its escalation filer from its queue, so refusing the
    queue refuses the filer.
    """

    escalation_queue: EscalationQueue | None = None
    steward_factory: Callable[..., object] | None = None
    event_store: EventStore | None = None
    cost_store: CostStore | None = None


NO_STORES = WorkflowStores()


def refuse_production_stores(stores: WorkflowStores) -> None:
    """Raise ``EvalHermeticityError`` for the first store *stores* holds, in declaration order."""
    for field in dataclasses.fields(stores):
        value = getattr(stores, field.name)
        if value is not None:
            raise EvalHermeticityError(field.name, value)
