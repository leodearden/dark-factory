"""Post-merge staleness hook: restart fused-memory.service after code-touching merges.

Design
------
Detection happens in the ``SpeculativeMergeWorker`` 'done' path (the
chokepoint) via an injected ``on_merge_landed`` callback.  Execution is
decoupled and deferred: the ``StaleServiceRestartCoordinator`` arms a pending
flag on detection and fires the restart only when the orchestrator's run-loop
idle branch (no dispatched agents) calls ``maybe_restart(agents_idle=True)``
AND the debounce window has elapsed since the last fused-memory merge.

This guarantees:
- No dispatched agent races with the restart.
- A burst of fused-memory merges coalesces into exactly ONE restart.
- The restart script (``scripts/restart-fused-memory.sh --drain``) runs
  detached / fire-and-forget so the ~150 s drain+health never blocks the
  orchestrator event loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from orchestrator.event_store import EventStore
    from orchestrator.git_ops import GitOps

from orchestrator.event_store import EventType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path-filter helper
# ---------------------------------------------------------------------------


def diff_touches_watched_paths(
    changed_files: list[str],
    prefixes: list[str],
) -> bool:
    """Return True iff any file in *changed_files* is under a watched prefix.

    The match is boundary-safe: ``fused-memory/src/foo`` matches prefix
    ``fused-memory/src/`` but ``fused-memory/srcfoo`` does NOT.

    A file path *p* matches prefix *q* when:
    - ``p == q``  (exact match, e.g. the directory itself), or
    - ``p.startswith(q.rstrip('/') + '/')``  (strictly under the directory).
    """
    if not changed_files or not prefixes:
        return False
    for path in changed_files:
        for prefix in prefixes:
            boundary = prefix.rstrip('/') + '/'
            if path == prefix or path.startswith(boundary):
                return True
    return False


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class StaleServiceRestartCoordinator:
    """Arms and fires a debounced restart of fused-memory.service.

    Parameters
    ----------
    git_ops:
        Live GitOps instance; used to compute the landed diff via
        ``get_merge_diff_files(base_sha, head_sha)``.
    event_store:
        Live EventStore; used to emit ``EventType.service_restart``.
        May be None (no-op emit).
    watch_prefixes:
        File path prefixes that trigger a restart when present in the diff.
        Default: ``['fused-memory/src/']``.
    debounce_secs:
        Minimum quiet-time (seconds, monotonic) since the LAST
        fused-memory-touching merge before the restart fires.
        Default: 120 s.
    enabled:
        Global on/off switch.  When False, ``note_merge`` short-circuits
        immediately without calling git_ops.
    script_path:
        Path to the restart script, relative to ``project_root``.
    project_root:
        Absolute root of the project repo.
    restart_executor:
        Optional injectable async callable (no args, no return value) that
        performs the actual restart.  When None (the default) a fire-and-forget
        ``asyncio.create_subprocess_exec`` of ``<project_root>/<script_path>
        --drain`` is used.
    clock:
        Injectable callable returning a monotonic float (default:
        ``time.monotonic``).  Override in tests for determinism.
    """

    def __init__(
        self,
        *,
        git_ops: 'GitOps',
        event_store: 'EventStore | None',
        watch_prefixes: list[str] | None = None,
        debounce_secs: float = 120.0,
        enabled: bool = True,
        script_path: str = 'scripts/restart-fused-memory.sh',
        project_root: str | Path = '.',
        restart_executor: Callable[[], Awaitable[object]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._git_ops = git_ops
        self._event_store = event_store
        self._watch_prefixes: list[str] = (
            watch_prefixes if watch_prefixes is not None else ['fused-memory/src/']
        )
        self._debounce_secs = debounce_secs
        self._enabled = enabled
        self._script_path = script_path
        self._project_root = Path(project_root)
        self._restart_executor = restart_executor
        self._clock = clock

        # State
        self._pending: bool = False
        self._last_request_monotonic: float = 0.0
        self._last_restart_monotonic: float = 0.0
        # Accumulated trigger metadata for the current pending burst
        self._trigger_task_ids: list[str] = []
        self._trigger_merge_shas: list[str] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_pending(self) -> bool:
        """True if a restart has been armed and not yet fired."""
        return self._pending

    @property
    def enabled(self) -> bool:
        """True if the coordinator is enabled."""
        return self._enabled

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    async def note_merge(
        self,
        task_id: str,
        base_sha: str,
        head_sha: str,
    ) -> bool:
        """Called by ``SpeculativeMergeWorker`` when a merge lands on main.

        Returns True when this merge armed (or re-armed) the pending-restart
        flag; False when it was ignored (disabled, no watched files, git error).
        """
        if not self._enabled:
            return False

        changed = await self._git_ops.get_merge_diff_files(base_sha, head_sha)
        if not diff_touches_watched_paths(changed, self._watch_prefixes):
            return False

        # Arm / re-arm
        self._pending = True
        self._last_request_monotonic = self._clock()
        self._trigger_task_ids.append(task_id)
        self._trigger_merge_shas.append(head_sha)
        logger.info(
            'fused-memory restart pending: merge %s for task %s touched %d'
            ' watched file(s)',
            head_sha[:12],
            task_id,
            len([f for f in changed if diff_touches_watched_paths([f], self._watch_prefixes)]),
        )
        return True

    async def maybe_restart(self, *, agents_idle: bool) -> bool:
        """Fire the restart if all gate conditions are met.

        Conditions: enabled AND pending AND agents_idle AND debounce elapsed.

        Returns True when the restart was fired; False otherwise (no side
        effects — caller may call again on the next idle tick).
        """
        if not (
            self._enabled
            and self._pending
            and agents_idle
            and (self._clock() - self._last_request_monotonic >= self._debounce_secs)
        ):
            return False

        # Snapshot trigger metadata before clearing
        trigger_task_ids = list(self._trigger_task_ids)
        trigger_merge_shas = list(self._trigger_merge_shas)

        executor = self._restart_executor or self._default_restart_executor
        await executor()

        # Emit observability event
        if self._event_store is not None:
            self._event_store.emit(
                EventType.service_restart,
                task_id=trigger_task_ids[0] if trigger_task_ids else 'unknown',
                phase='service_restart',
                data={
                    'service': 'fused-memory',
                    'script': str(self._script_path),
                    'trigger_task_ids': trigger_task_ids,
                    'merge_shas': trigger_merge_shas,
                    'reason': 'post_merge_fused_memory_code_change',
                },
            )

        logger.warning(
            'Restarting fused-memory.service (post-merge staleness hook):'
            ' triggered by tasks %s, merges %s',
            trigger_task_ids,
            [s[:12] for s in trigger_merge_shas],
        )

        # Clear pending state
        self._pending = False
        self._trigger_task_ids = []
        self._trigger_merge_shas = []
        self._last_restart_monotonic = self._clock()

        return True

    async def _default_restart_executor(self) -> None:
        """Fire-and-forget: spawn the restart script detached."""
        script = self._project_root / self._script_path
        await asyncio.create_subprocess_exec(
            str(script),
            '--drain',
            start_new_session=True,
        )
        # Intentionally NOT awaiting the process exit — the ~150 s
        # drain+health must not block the orchestrator event loop.
