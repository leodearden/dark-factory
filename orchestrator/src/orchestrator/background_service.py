"""BackgroundService + LifecycleRegistry — one reusable seam collapsing the
eleven background-loop/service lifecycles in harness.py.

PRD ``plans/harness-supervision-prd.md`` §5.3 (LR-1/2/3): a bounded, uniform
start/stop contract for every long-lived harness background task so the
recurring shutdown-hang class (survey 2.3; tasks 108/161/162/169/875/1080)
becomes structurally impossible.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

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
        """Wake periodically and run the pass.

        Sleep-first (hoisted verbatim from ``_no_landings_breaker_loop`` /
        ``_warm_lane_gc_loop`` — see harness.py): ``CancelledError`` re-raises
        so the loop task ends promptly on cancellation; a plain ``Exception``
        is bounded-logged (task 1907 shape: a one-line ``logger.error`` —
        NEVER ``logger.exception``, whose O(frame-count) traceback formatting
        on a pathological traceback can exceed a per-test timeout) capped at
        ``max_failure_logs`` CONSECUTIVE records, then backs off via
        ``self.backoff`` so a pass that fails immediately can never
        tight-spin. A successful pass resets the consecutive-failure counter
        so a later failure streak logs again.
        """
        while True:
            try:
                await asyncio.sleep(self.interval_secs)
                await self.pass_fn()
                self._consecutive_failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._consecutive_failures < self.max_failure_logs:
                    logger.error(
                        '%s pass failed: %s: %s',
                        self.name,
                        type(exc).__name__,
                        exc,
                    )
                self._consecutive_failures += 1
                await asyncio.sleep(self.backoff.delay_for(self._consecutive_failures))

    async def stop(self) -> None:
        """Cancel the loop task, suppress CancelledError, and clear the slot.

        Idempotent: a second call (or a call when never started) is a clean
        no-op. Mirrors the cancel/suppress/None idiom shared by every
        existing ``_stop_*`` method in harness.py. The bounded wait (LR-2)
        is the registry's job (``LifecycleRegistry.stop_all``, step-12), not
        this method's — a wedging ``pass_fn`` that ignores cancellation would
        otherwise hang this ``await`` forever.
        """
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None


@runtime_checkable
class LifecycleService(Protocol):
    """Structural contract every LifecycleRegistry member satisfies.

    Both ``BackgroundService`` and ``ManagedService`` satisfy this shape
    without explicitly inheriting from it — ``LifecycleRegistry`` stays
    duck-typed (LR-3); this Protocol exists only to give ``register()`` and
    ``services`` a precise type instead of ``Any``.

    ``start()`` is typed ``Awaitable[None] | None`` rather than requiring a
    coroutine function: ``BackgroundService.start()`` is deliberately plain
    sync (fire-and-forget ``asyncio.create_task``, nothing to await —
    step-4), while ``ManagedService.start()`` wraps bespoke async setup
    (e.g. building an aiohttp server) and must be awaited before the next
    service starts. ``LifecycleRegistry.start_all()`` awaits the result
    only when it is itself awaitable, so either shape conforms.
    """

    name: str
    stop_timeout_secs: float

    def start(self) -> Awaitable[None] | None: ...

    async def stop(self) -> None: ...


@dataclass
class ManagedService:
    """Thin adapter wrapping a bespoke lifecycle service's existing
    ``_start_*``/``_stop_*`` methods so it registers in the same
    ``LifecycleRegistry`` as ``BackgroundService``, internals unchanged.

    The four bespoke harness services (merge-worker, offline-lane,
    escalation-server, watcher-supervisor) have custom build/teardown that
    doesn't fit the uniform ``BackgroundService`` pass_fn/interval shape
    (stateful backoff, aiohttp server, lockfile release, coordinator
    construction — see plan.json design_decisions). This adapter gives them
    the same ordered-start / bounded-reverse-stop contract (LR-1/2/3)
    without rewriting their internals.
    """

    name: str
    start_fn: Callable[[], Awaitable[None] | None]
    stop_fn: Callable[[], Awaitable[None]]
    stop_timeout_secs: float

    async def start(self) -> None:
        """Delegate to ``start_fn``, awaiting it only if it is awaitable.

        Mirrors ``LifecycleRegistry.start_all``'s tolerance: most wrapped
        ``_start_*`` methods are ``async def``, but at least one (harness's
        watcher-supervisor) is plain sync — its internal gate logic never
        awaits anything, so there is nothing to await.
        """
        result = self.start_fn()
        if inspect.isawaitable(result):
            await result

    async def stop(self) -> None:
        await self.stop_fn()


class LifecycleRegistry:
    """Ordered registry of lifecycle services with a bounded start/stop ladder.

    PRD §5.3 (LR-3): ``register(svc)`` appends in declared order and
    ``start_all()`` awaits each service's ``start()`` in that same order.
    ``stop_all()`` walks the REVERSE of registration order, bounding each
    ``stop()`` so one wedging service can never hang shutdown (LR-2 / S1).

    Services satisfy the ``LifecycleService`` Protocol structurally; both
    ``BackgroundService`` and ``ManagedService`` do so without explicit
    inheritance.
    """

    def __init__(self) -> None:
        self._services: list[LifecycleService] = []

    @property
    def services(self) -> list[LifecycleService]:
        """The registered services, in declared (start) order."""
        return list(self._services)

    def register(self, service: LifecycleService) -> None:
        """Append ``service`` to the registry, in declared order."""
        self._services.append(service)

    async def start_all(self) -> None:
        """Await each registered service's start(), in registration order.

        ``start()`` may be a plain sync call (``BackgroundService``) or a
        coroutine function (``ManagedService``) — see ``LifecycleService``.
        The result is awaited only when it is itself awaitable, so a
        service's start-up work (if any) still fully completes before the
        next service's start() begins (LR-3).
        """
        for service in self._services:
            result = service.start()
            if inspect.isawaitable(result):
                await result

    async def stop_all(self) -> None:
        """Stop every registered service in REVERSE registration order.

        Each ``stop()`` is bounded by the service's own ``stop_timeout_secs``
        (LR-2 / S1): a wedging service is abandoned with a WARNING and the
        ladder proceeds to the next service; a ``stop()`` that raises a
        plain ``Exception`` is caught and logged so one failing service can
        never abort the ladder. This is the structural elimination of the
        shutdown-hang class this module exists to fix.
        """
        for service in reversed(self._services):
            try:
                await asyncio.wait_for(service.stop(), timeout=service.stop_timeout_secs)
            except TimeoutError:
                logger.warning(
                    '%s did not stop within %.2fs — abandoning',
                    service.name,
                    service.stop_timeout_secs,
                )
            except Exception as exc:
                logger.warning('%s stop() failed: %s', service.name, exc)
