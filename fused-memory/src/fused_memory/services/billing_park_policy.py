"""The durable queue's park policy for exhausted provider billing (task 9944).

Each ``BillingSource`` has its own guard, and the guard decides what that
source's exhaustion means: when its parked rows re-probe, who is told when a
row parks, and what a landed parked write clears. The policy only classifies
and dispatches, so a new source is a new guard, never a new branch here.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol

from fused_memory.services.durable_queue import ParkDecision, ParkedWriteEvent
from fused_memory.services.provider_billing import (
    BillingExhaustion,
    BillingSource,
    classify_billing_exhaustion,
)

_CLEARED_SITE = 'durable_queue'


class BillingSourceGuard(Protocol):
    """One billing source's side of parking."""

    def park_until(self) -> float | None:
        """When this source's parked rows next re-probe (epoch), or None for the queue's horizon."""
        ...

    async def on_parked(self, exhaustion: BillingExhaustion, *, site: str) -> None: ...

    async def on_cleared(self, *, site: str) -> None: ...


class BillingParkPolicy:
    """Parks the durable writes a provider refuses for exhausted billing.

    Satisfies ``durable_queue.py::ParkPolicy``; a parked write that lands is
    the recovery evidence, and every guard hears of it.
    """

    def __init__(self, guards: Mapping[BillingSource, BillingSourceGuard]) -> None:
        missing = [source.value for source in BillingSource if source not in guards]
        if missing:
            raise ValueError(f'BillingParkPolicy needs a guard for every billing source; missing {missing}')
        self._guards: Mapping[BillingSource, BillingSourceGuard] = MappingProxyType(dict(guards))

    def park_decision(self, exc: BaseException) -> ParkDecision | None:
        exhaustion = classify_billing_exhaustion(exc)
        if exhaustion is None:
            return None
        return ParkDecision(
            reason=f'{exhaustion.provider} {exhaustion.source.value} exhausted ({exhaustion.code})',
            retry_at=self._guards[exhaustion.source].park_until(),
        )

    async def on_parked(self, event: ParkedWriteEvent) -> None:
        exhaustion = classify_billing_exhaustion(event.exc)
        if exhaustion is not None:
            await self._guards[exhaustion.source].on_parked(
                exhaustion, site=f'durable_queue:{event.operation}',
            )

    async def on_cleared(self) -> None:
        for guard in self._guards.values():
            await guard.on_cleared(site=_CLEARED_SITE)
