"""The merger waits here, before its look-ahead pop, for the verifier to dispatch the speculative item it just queued.

The decision is what the item's successors stack on, as its dispatch left it
(its chain tip when it chained), or None. The contract is
``plans/deep-merge-ahead-prd.md`` (dispatch contract).
"""
from __future__ import annotations

import asyncio
import dataclasses


@dataclasses.dataclass(frozen=True)
class HoldEnded:
    """A hold that is over, with its successor base (None when no dispatch decided)."""

    successor_base: str | None


class LookaheadHold:
    """Pending successor-base decisions, keyed by the held item's request id."""

    def __init__(self) -> None:
        self._decisions: dict[str, asyncio.Future[str | None]] = {}

    def hold(self, request_id: str) -> None:
        """Register a pending decision for *request_id*."""
        if request_id in self._decisions:
            raise ValueError(f'look-ahead hold already registered for {request_id}')
        self._decisions[request_id] = asyncio.get_running_loop().create_future()

    def decide(self, request_id: str, successor_base: str | None) -> None:
        """Publish what *request_id*'s successors stack on; a no-op when nothing holds it."""
        decision = self._decisions.get(request_id)
        if decision is not None and not decision.done():
            decision.set_result(successor_base)

    def is_held(self, request_id: str) -> bool:
        return request_id in self._decisions

    def release_all(self) -> None:
        """Decide None for every pending hold (lane stop)."""
        for request_id in list(self._decisions):
            self.decide(request_id, None)

    async def wait(
        self,
        request_id: str,
        *,
        alive: asyncio.Future,
        wake: asyncio.Future | None = None,
    ) -> HoldEnded | None:
        """Wait until *request_id*'s hold ends: decided, or undecided once
        *alive* (the verifier) has finished; an unheld id has ended already.

        Returns None, with the hold still standing, when *wake* finished first.
        """
        decision = self._decisions.get(request_id)
        if decision is None:
            return HoldEnded(None)
        watched = {decision, alive} if wake is None else {decision, alive, wake}
        try:
            await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            self._decisions.pop(request_id, None)
            raise
        if not (decision.done() or alive.done()):
            return None
        self._decisions.pop(request_id, None)
        return HoldEnded(decision.result() if decision.done() else None)
