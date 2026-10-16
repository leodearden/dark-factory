"""What admission cost a gated verify plan, recorded as one ``verify_admission_plan`` event.

A plan is one ``verify.py::run_scoped_verification`` or ``run_full_verification``
call. It stamps its arrival once, every test leg it runs joins the admission gate
under that stamp, and each admitted leg returns an ``AdmissionLeg``. The gate
(``verify_admission_gate.py``) owns the ordering and ``verify.py`` runs the legs.
This module owns the leg record and the plan event. The plan payload's shape is
the comment on ``event_store.py::EventType.verify_admission_plan``; a leg's is
``AdmissionLeg``. Design: plans/verify-admission-fifo-prd.md.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from shared.verify_admission import is_gated_role

from orchestrator.event_store import EventStore, EventType
from orchestrator.verify_admission_gate import (
    AdmissionGrant,
    AdmissionOrder,
    AdmissionPlan,
    AdmissionRequest,
)

PlanOutcome = Literal['returned', 'raised']


def begin_plan(role: str, *, task_id: str | None) -> AdmissionPlan | None:
    """A gated *role*'s plan stamp; None for an ungated role, which never reaches the gate."""
    return AdmissionPlan.begin(role=role, task_id=task_id) if is_gated_role(role) else None


def plan_request(
    plan: AdmissionPlan | None, module: str | None, scope_kind: str | None = None,
) -> AdmissionRequest | None:
    """*plan*'s admission request for one module's leg; None for a plan-less (ungated) role."""
    return None if plan is None else plan.request(module=module, scope_kind=scope_kind)


@dataclass(frozen=True)
class AdmissionLeg:
    """One admitted test leg: its turn in the gate, the flock wait inside that turn,
    how long it held the turn, and how the leg ended.

    Three intervals, all in seconds. ``waited_secs`` runs from the leg's arrival
    at the gate to its grant. ``flock_wait_secs`` runs from the grant until the
    cross-process flock slot was held. ``held_secs`` runs from the grant until
    the turn was handed back, so it includes the flock wait, as the watch's
    ``HolderView.held_secs`` does. ``slot_held`` False means the flock failed
    open and the leg ran without cross-process gating.

    Every field is JSON-native, so the record crosses ``VerifyResult``'s generic
    wire codec and becomes an event payload with no conversion of its own.
    """

    plan_id: str
    module: str | None
    scope_kind: str | None
    order: AdmissionOrder
    arrived_at: str
    granted_at: str
    waited_secs: float
    depth_at_arrival: int
    slot_held: bool
    flock_wait_secs: float
    held_secs: float
    rc: int
    timed_out: bool

    @classmethod
    def of(
        cls, grant: AdmissionGrant, *, slot_taken: float, released: float, rc: int, timed_out: bool,
    ) -> AdmissionLeg:
        """The leg *grant* admitted. *slot_taken* and *released* are monotonic times."""
        return cls(
            plan_id=grant.request.plan_id,
            module=grant.request.module,
            scope_kind=grant.request.scope_kind,
            order=grant.order,
            arrived_at=grant.arrived_at.isoformat(),
            granted_at=grant.granted_at.isoformat(),
            waited_secs=grant.waited_secs,
            depth_at_arrival=grant.depth_at_arrival,
            slot_held=grant.slot_held,
            flock_wait_secs=slot_taken - grant.granted,
            held_secs=released - grant.granted,
            rc=rc,
            timed_out=timed_out,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def legs_from_wire(legs: Iterable[AdmissionLeg | Mapping[str, Any]]) -> tuple[AdmissionLeg, ...]:
    """*legs* as records. ``verify_runner.py::result_from_dict`` rebuilds a
    VerifyResult generically, so its legs arrive as plain dicts."""
    return tuple(leg if isinstance(leg, AdmissionLeg) else AdmissionLeg(**leg) for leg in legs)


def emit_admission_plan(
    event_store: EventStore | None,
    plan: AdmissionPlan | None,
    legs: Sequence[AdmissionLeg],
    *,
    outcome: PlanOutcome,
) -> None:
    """Record *plan* once it has returned or raised.

    A plan that returned with no admitted leg is not recorded. A plan that
    raised has no legs to report, because they never came back with it. Its
    row says that its admission went unmeasured.
    """
    if event_store is None or plan is None or (outcome == 'returned' and not legs):
        return
    orders = {leg.order for leg in legs}
    event_store.emit(
        EventType.verify_admission_plan,
        task_id=plan.task_id,
        phase='verify',
        role=plan.role,
        data={
            'plan_id': plan.plan_id,
            'task_id': plan.task_id,
            'role': plan.role,
            'outcome': outcome,
            'order': orders.pop() if len(orders) == 1 else None,
            'plan_arrival_at': plan.arrived_at.isoformat(),
            'completed_at': datetime.now(UTC).isoformat(),
            'legs': [leg.to_dict() for leg in legs],
        },
    )
