"""In-process registry of reconciliation full cycles currently in flight.

Source of the machine-readable ``recon_busy`` signal on the fused-memory
``/health`` endpoint (task 2703 δ). ``ReconciliationHarness.run_full_cycle``
wraps its stage-execution block in::

    with self._active_runs.track(run_id, project_id, started_at) as active:
        for stage in stages:
            active.stage(stage.stage_id.value)
            ...

so a full cycle in flight is a structured entry, and — because the entry is
cleared in the context manager's ``finally`` — it is cleared on EVERY exit
path: a normal return, an ``Exception``, or the ``asyncio.CancelledError``
that a timeout/shutdown raises mid-stage (survey E1). That last case is why
this is a context manager and not a pair of add/remove calls: a restart
cancelled mid-cycle must not leak a phantom-busy entry.

Pure and stdlib-only: no I/O, no awaits. The harness runs a single-threaded
asyncio loop, so ``snapshot()`` reads are atomic with respect to the
``track``/``stage``/exit mutations (there is no ``await`` between them), and
``snapshot()`` never raises.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


class _RunHandle:
    """Handle yielded by ``ActiveRunRegistry.track``; ``stage(name)`` mutates
    the current stage of this run's entry.

    The handle holds the same ``entry`` dict the registry stores, so a
    ``stage()`` update is visible to ``snapshot()`` immediately.
    """

    def __init__(self, entry: dict) -> None:
        self._entry = entry

    def stage(self, name: str | None) -> None:
        """Set the current stage name for this run's entry."""
        self._entry["stage"] = name


class ActiveRunRegistry:
    """Tracks reconciliation full cycles currently in flight, keyed by run_id."""

    def __init__(self) -> None:
        # run_id -> entry dict. Insertion order is preserved by dict, so
        # snapshot() reports runs in the order they started.
        self._runs: dict[str, dict] = {}

    @contextmanager
    def track(
        self, run_id: str, project_id: str, started_at: str
    ) -> Iterator[_RunHandle]:
        """Mark *run_id* active for the duration of the ``with`` block.

        The entry is registered on entry and cleared in the ``finally`` — so
        it clears on a normal return, on an ``Exception``, and on a
        ``BaseException`` such as the ``asyncio.CancelledError`` raised when a
        cycle is cancelled mid-stage. Yields a :class:`_RunHandle` whose
        ``stage(name)`` updates the entry's current stage.
        """
        entry: dict = {
            "project_id": project_id,
            "run_id": run_id,
            "stage": None,
            "started_at": started_at,
        }
        self._runs[run_id] = entry
        try:
            yield _RunHandle(entry)
        finally:
            self._runs.pop(run_id, None)

    def snapshot(self) -> list[dict]:
        """Return the in-flight runs as independent, JSON-serializable dicts
        in insertion order. Never raises; the returned dicts are shallow
        copies, so mutating them does not alter registry state."""
        return [dict(entry) for entry in self._runs.values()]
