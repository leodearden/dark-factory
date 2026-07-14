"""BackgroundService + LifecycleRegistry — one reusable seam collapsing the
eleven background-loop/service lifecycles in harness.py.

PRD ``plans/harness-supervision-prd.md`` §5.3 (LR-1/2/3): a bounded, uniform
start/stop contract for every long-lived harness background task so the
recurring shutdown-hang class (survey 2.3; tasks 108/161/162/169/875/1080)
becomes structurally impossible.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Parity default (Open-Q Q5): the only current failure backoff across the
# eleven harness background loops is this fixed constant — mirrors
# harness._BG_LOOP_FAILURE_BACKOFF_SECS verbatim (see
# test_background_service.py::test_default_backoff_secs_matches_harness_parity_constant).
DEFAULT_BACKOFF_SECS: float = 60.0


@dataclass(frozen=True)
class BackoffPolicy:
    """Backoff delay applied after a failed BackgroundService pass.

    Minimal constant-delay value object: Open-Q Q5 parity requires carrying
    the CURRENT backoff verbatim, and the only current failure backoff across
    the eleven loops is a fixed constant (no exponential/attempt-dependent
    shape exists today). ``delay_for`` still accepts an ``attempt`` argument
    so the call-site shape stays stable if a non-constant policy is
    introduced later.
    """

    delay_secs: float

    def delay_for(self, attempt: int) -> float:
        del attempt  # constant policy: same delay regardless of attempt
        return self.delay_secs


@dataclass
class BackgroundService:
    """A named, periodic, sleep-first background loop with a bounded stop.

    PRD §5.3: the uniform seam for the seven harness sweeps that share the
    identical ``while True: sleep(interval); await pass(); ...`` shape
    (hoisted verbatim from ``_no_landings_breaker_loop`` /
    ``_warm_lane_gc_loop`` — see harness.py). ``start()``/``stop()`` are
    idempotent; ``stop_timeout_secs`` is read by ``LifecycleRegistry.stop_all``
    to bound how long a wedging service may delay shutdown (LR-2).
    """

    name: str
    pass_fn: Callable[[], Awaitable[None]]
    interval_secs: float
    backoff: BackoffPolicy
    stop_timeout_secs: float
    max_failure_logs: int
    _task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _consecutive_failures: int = field(default=0, init=False, repr=False)

    def start(self) -> None:
        """Start the loop task, deduping against an already-live task."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop(), name=self.name)

    async def _loop(self) -> None:
        """Run the pass once.

        Stub sufficient for start() idempotency (step-3): the canonical
        sleep-first / exception-bounded loop body lands in step-6.
        """
        await self.pass_fn()
