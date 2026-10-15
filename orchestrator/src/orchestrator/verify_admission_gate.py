"""In-process ordered admission gate in front of the task-verify flock slot.

Gated verify legs (roles ``task`` and ``background``) queue here as awaitables,
and the gate decides whose turn it is to go on and take the cross-process flock
slot in ``shared.verify_admission``. The gate does no I/O and never touches the
flock. Under ``order='arrival'`` the turn goes to the lowest ``(role class, plan
arrival, leg sequence)``; under ``'unordered'`` it goes to a uniformly random
waiter, as the flock's poll race did before the gate existed. Facts about a wait
leave by return value (the grant) and through a pure ``snapshot()``. Design,
contract and invariants: plans/verify-admission-fifo-prd.md.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import random
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Literal, get_args

from shared.verify_admission import is_gated_role

AdmissionOrder = Literal['unordered', 'arrival']

ROLE_CLASSES: Mapping[str, int] = MappingProxyType({'task': 0, 'background': 1})


def role_class(role: str) -> int:
    """The ordering class of a gated *role*; raises ValueError for any other role."""
    if not is_gated_role(role):
        raise ValueError(
            f'only gated verify roles reach the admission gate: role={role!r} is not gated',
        )
    if role not in ROLE_CLASSES:
        raise ValueError(
            f'only gated verify roles reach the admission gate: role={role!r} has no role class',
        )
    return ROLE_CLASSES[role]


@dataclass(frozen=True)
class AdmissionPolicy:
    slots: int
    order: AdmissionOrder
    promote_after_secs: float

    def __post_init__(self) -> None:
        if self.slots < 1:
            raise ValueError(f'AdmissionPolicy.slots must be >= 1: slots={self.slots!r}')
        if self.order not in get_args(AdmissionOrder):
            raise ValueError(
                f'AdmissionPolicy.order must be one of {get_args(AdmissionOrder)}: '
                f'order={self.order!r}',
            )
        if self.promote_after_secs < 0:
            raise ValueError(
                'AdmissionPolicy.promote_after_secs must be >= 0: '
                f'promote_after_secs={self.promote_after_secs!r}',
            )


@dataclass(frozen=True)
class AdmissionRequest:
    role: str
    plan_id: str
    plan_arrival: float
    task_id: str | None
    module: str | None
    scope_kind: str | None

    def __post_init__(self) -> None:
        role_class(self.role)


@dataclass(frozen=True)
class AdmissionPlan:
    """One verify call's identity and arrival stamp, shared by every leg it admits."""

    plan_id: str
    role: str
    task_id: str | None
    arrival: float
    arrived_at: datetime

    def __post_init__(self) -> None:
        role_class(self.role)

    @classmethod
    def begin(cls, *, role: str, task_id: str | None) -> AdmissionPlan:
        return cls(
            plan_id=uuid.uuid4().hex,
            role=role,
            task_id=task_id,
            arrival=time.monotonic(),
            arrived_at=datetime.now(UTC),
        )

    def request(self, *, module: str | None, scope_kind: str | None) -> AdmissionRequest:
        return AdmissionRequest(
            role=self.role,
            plan_id=self.plan_id,
            plan_arrival=self.arrival,
            task_id=self.task_id,
            module=module,
            scope_kind=scope_kind,
        )


@dataclass(frozen=True)
class AdmissionGrant:
    request: AdmissionRequest
    order: AdmissionOrder
    arrived_at: datetime
    granted_at: datetime
    granted: float
    """Monotonic twin of ``granted_at``. A turn is held from here, which is
    what ``HolderView.held_secs`` measures."""
    waited_secs: float
    depth_at_arrival: int
    slot_held: bool = False
    """Whether the cross-process flock slot is held. The gate never acquires the
    flock, so a grant straight from ``admit`` carries False and
    ``verify.py::_admission_slot`` replaces it with the flock's answer."""


@dataclass(frozen=True)
class WaiterView:
    request: AdmissionRequest
    seq: int
    age_secs: float


@dataclass(frozen=True)
class HolderView:
    request: AdmissionRequest
    seq: int
    held_secs: float


@dataclass(frozen=True)
class AdmissionSnapshot:
    waiters: tuple[WaiterView, ...]
    holders: tuple[HolderView, ...]

    @property
    def depth(self) -> int:
        return len(self.waiters)

    @property
    def oldest(self) -> WaiterView | None:
        return max(self.waiters, key=lambda w: w.age_secs, default=None)

    @property
    def oldest_wait_secs(self) -> float | None:
        oldest = self.oldest
        return None if oldest is None else oldest.age_secs


@dataclass(frozen=True)
class _Turn:
    order: AdmissionOrder
    granted: float
    granted_at: datetime


@dataclass(frozen=True, eq=False)
class _Waiter:
    request: AdmissionRequest
    seq: int
    arrival: float
    arrived_at: datetime
    depth_at_arrival: int
    turn: asyncio.Future[_Turn]


class AdmissionGate:
    """Ordered turns for one flock slots directory. Loop-confined: every call
    must come from the one event loop that admitted its waiters."""

    def __init__(self) -> None:
        self._queued: dict[int, _Waiter] = {}
        self._holding: dict[int, _Waiter] = {}
        self._seq = itertools.count()

    @contextlib.asynccontextmanager
    async def admit(
        self, request: AdmissionRequest, policy: Callable[[], AdmissionPolicy],
    ) -> AsyncIterator[AdmissionGrant]:
        """Wait for this request's turn, then hold it for the body of the ``async with``.

        *policy* is read at every dispatch, so a reloaded slot count or order
        applies to the very next grant.
        """
        role_class(request.role)
        waiter = _Waiter(
            request=request,
            seq=next(self._seq),
            arrival=time.monotonic(),
            arrived_at=datetime.now(UTC),
            depth_at_arrival=len(self._queued),
            turn=asyncio.get_running_loop().create_future(),
        )
        self._queued[waiter.seq] = waiter
        try:
            self._dispatch(policy())
            turn = await waiter.turn
            yield AdmissionGrant(
                request=request,
                order=turn.order,
                arrived_at=waiter.arrived_at,
                granted_at=turn.granted_at,
                granted=turn.granted,
                waited_secs=turn.granted - waiter.arrival,
                depth_at_arrival=waiter.depth_at_arrival,
            )
        finally:
            self._leave(waiter, policy)

    def snapshot(self) -> AdmissionSnapshot:
        now = time.monotonic()
        waiters = tuple(
            WaiterView(request=w.request, seq=w.seq, age_secs=now - w.arrival)
            for w in sorted(self._queued.values(), key=lambda w: w.seq)
        )
        holders = tuple(
            HolderView(request=w.request, seq=w.seq, held_secs=now - w.turn.result().granted)
            for w in sorted(self._holding.values(), key=lambda w: w.seq)
        )
        return AdmissionSnapshot(waiters=waiters, holders=holders)

    def _leave(self, waiter: _Waiter, policy: Callable[[], AdmissionPolicy]) -> None:
        # The single queue exit (PRD invariant 1): a waiter still queued just
        # leaves; one already granted, even if cancelled before it resumed,
        # hands its turn on.
        if self._queued.pop(waiter.seq, None) is not None:
            return
        if self._holding.pop(waiter.seq, None) is not None:
            self._dispatch(policy())

    def _dispatch(self, policy: AdmissionPolicy) -> None:
        while self._queued and len(self._holding) < policy.slots:
            waiter = self._pick(policy)
            del self._queued[waiter.seq]
            if waiter.turn.done():
                continue
            self._holding[waiter.seq] = waiter
            waiter.turn.set_result(
                _Turn(order=policy.order, granted=time.monotonic(), granted_at=datetime.now(UTC)),
            )

    def _pick(self, policy: AdmissionPolicy) -> _Waiter:
        if policy.order == 'unordered':
            return random.choice(list(self._queued.values()))
        now = time.monotonic()

        def arrival_key(waiter: _Waiter) -> tuple[int, float, int]:
            promoted = now - waiter.arrival >= policy.promote_after_secs
            effective_class = 0 if promoted else role_class(waiter.request.role)
            return (effective_class, waiter.request.plan_arrival, waiter.seq)

        return min(self._queued.values(), key=arrival_key)


_GATES: dict[Path, AdmissionGate] = {}


def gate_for(slots_dir: Path | str) -> AdmissionGate:
    """The process-wide gate for *slots_dir*, keyed by its resolved path."""
    key = Path(slots_dir).resolve(strict=False)
    gate = _GATES.get(key)
    if gate is None:
        gate = _GATES[key] = AdmissionGate()
    return gate
