"""Post-merge staleness hook: restart a service after code-touching merges.

Design
------
Detection happens in the ``SpeculativeMergeWorker`` 'done' path (the
chokepoint) via an injected ``on_merge_landed`` callback.  Execution is
decoupled and deferred: the ``StaleServiceRestartCoordinator`` arms a pending
flag on detection and fires the restart only when the gate conditions are met
(see ``maybe_restart``).

This guarantees:
- Burst of merges coalesces into exactly ONE restart.
- The restart script runs detached / fire-and-forget so it never blocks the
  orchestrator event loop.

Two instances are used in production:

* **fused-memory** (``service_name='fused-memory'``, ``require_idle=True``):
  fires only at the run-loop's idle quiet-window (no dispatched agents).
  Script: ``scripts/restart-fused-memory.sh --drain``.

* **dashboard** (``service_name='dashboard'``, ``require_idle=False``):
  leaf service — fires even while agents are dispatching (promptly on the
  busy-wait branch).  Script: ``scripts/restart-dashboard.sh`` (no drain).
"""

from __future__ import annotations

import asyncio  # noqa: F401 — no direct call site in this module (the actual

# asyncio.create_subprocess_exec now lives in proc_supervision.RestartPlan.execute()),
# but this binding is load-bearing: tests patch it via the
# 'orchestrator.service_restart.asyncio.create_subprocess_exec' dotted path,
# which requires the 'asyncio' module object to be reachable as an attribute
# of this module (patching an attribute on that shared module object patches
# it for every importer, including proc_supervision). Removing this import
# breaks that patch path with AttributeError — see test_fleet_staleness_composition.py.
import json
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.event_store import EventStore
    from orchestrator.git_ops import GitOps
    from orchestrator.proc_supervision import EscalationSpec

from orchestrator.event_store import EventType
from orchestrator.proc_supervision import RestartDisposition, RestartPlan

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
# Cgroup-escaping detached restart (U2 — task 1973)
# ---------------------------------------------------------------------------


async def schedule_detached_systemd_restart(
    *,
    script: str,
    script_args: list[str],
    project_root: str | Path,
    transient_unit: str,
    on_active_secs: int,
    on_failure_escalation: EscalationSpec | None = None,
    runner=None,
):
    """Schedule a detached, cgroup-escaping self-restart via ``systemd-run --user``.

    Thin ``proc_supervision.RestartPlan`` caller (M1, task 2237) — the gap is
    closed by RP-4: builds a same-unit plan (``own_unit=target_unit=
    transient_unit``, ``verify=None``) which routes ``RestartPlan.execute()``
    to the DETACHED systemd-run path, carrying ``--working-directory=<cwd>``
    (RP-3) and, when *on_failure_escalation* is supplied, a ``/bin/sh -c``
    on-failure escalation-submit wrapper (RP-4) so a LATER fire-time failure
    of the deferred payload (e.g. a failed MainPID/timestamp verify inside
    ``scripts/restart-orchestrator.sh``) files a born-at-L2 escalation
    instead of being traceable only via journald
    (``journalctl --user -u <transient_unit>``).

    ``--on-active=<on_active_secs>`` defers execution so this call returns
    immediately after *registering* the transient unit; the restart payload
    fires later, under the user systemd manager's own cgroup — which is what
    lets it survive a same-cgroup ``systemctl restart`` SIGKILL under
    ``KillMode=control-group``. ``--collect`` removes the transient unit once
    it exits (success or failure) so repeated fires never collide on a
    lingering unit name.

    Parameters
    ----------
    script:
        Path to the restart script, relative to *project_root*.
    script_args:
        Extra positional args appended after the script path (e.g. ``['--drain']``).
    project_root:
        Absolute root of the project repo (``str`` or ``Path`` — e.g.
        ``Config.project_root`` is a ``Path``); the script path is resolved
        as ``Path(project_root) / script``.
    transient_unit:
        Name of the transient systemd-run unit (e.g.
        ``orch-selfrestart-on-merge-3.service``).
    on_active_secs:
        Seconds to defer before the transient unit fires.
    on_failure_escalation:
        Optional ``EscalationSpec`` filed (via the RP-4 ``/bin/sh -c``
        on-failure branch) if the deferred payload itself exits non-zero at
        fire time. ``None`` (default) builds a valid, unbranched payload —
        no on-failure reporting.
    runner:
        Optional injectable async callable with the same signature as
        ``asyncio.create_subprocess_exec`` (for testing). Defaults to
        ``asyncio.create_subprocess_exec``.

    Raises
    ------
    RuntimeError:
        When the ``systemd-run`` *registration* itself exits non-zero (a
        ``RestartDisposition.REGISTRATION_FAILED`` outcome). The error
        carries the last 2000 characters of combined stdout/stderr —
        preserves the coordinator's transient-retry contract
        (``StaleServiceRestartCoordinator.maybe_restart`` treats any
        non-FileNotFoundError/PermissionError exception as retryable).
    """
    plan = RestartPlan(
        script=Path(script),
        args=list(script_args),
        cwd=Path(project_root),
        target_unit=transient_unit,
        own_unit=transient_unit,
        on_failure_escalation=on_failure_escalation,
        verify=None,
        transient_unit=transient_unit,
        on_active_secs=on_active_secs,
    )
    outcome = await plan.execute(runner=runner)
    if outcome.disposition == RestartDisposition.REGISTRATION_FAILED:
        raise RuntimeError(
            f'systemd-run registration of {transient_unit} failed: {outcome.detail}'
        )


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class StaleServiceRestartCoordinator:
    """Arms and fires a debounced restart of a service after code-touching merges.

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
        watched-path merge before the restart fires.
        Default: 120 s.
    enabled:
        Global on/off switch.  When False, ``note_merge`` short-circuits
        immediately without calling git_ops.
    script_path:
        Path to the restart script, relative to ``project_root``.
    project_root:
        Root of the project repo.  Resolved to an absolute path at
        construction time (``Path(project_root).resolve()``) regardless of
        what is passed in — ``RestartPlan.__post_init__`` (task 2237) raises
        ``ValueError`` on a non-absolute ``cwd``, and ``_default_restart_executor``
        builds its ``RestartPlan`` from ``self._project_root`` directly, so a
        relative or default (``'.'``) ``project_root`` must never reach it
        unresolved — that would turn every restart attempt into a permanent
        ``ValueError`` that ``maybe_restart`` misclassifies as transient and
        retries forever.
    restart_executor:
        Optional injectable async callable (no args, no return value) that
        performs the actual restart.  When None (the default) a fire-and-forget
        ``asyncio.create_subprocess_exec`` of ``<project_root>/<script_path>
        [*script_args]`` is used.
    clock:
        Injectable callable returning a monotonic float (default:
        ``time.monotonic``).  Override in tests for determinism.
    service_name:
        Human-readable service identifier injected into log strings and event
        data (``data['service']``, ``data['reason']``).  Default: ``'fused-memory'``
        so the existing fused-memory instance renders byte-identical text.
    require_idle:
        When True (default), ``maybe_restart`` only fires when ``agents_idle``
        is True — the orchestrator's idle quiet-window.  When False (leaf
        services such as the dashboard), the gate becomes
        ``(agents_idle or not require_idle)`` so the restart fires even while
        agents are dispatching.
    restart_precondition:
        Optional injectable sync callable (no args) returning bool.  When
        provided, ``maybe_restart`` evaluates it AFTER the
        enabled/pending/idle/debounce gate and only fires when it returns
        truthy.  A falsy return OR a raised exception is treated as "not
        satisfied" (fail-safe — pending is left set so the next idle tick
        retries).  Default None preserves byte-identical existing behaviour
        for coordinators that don't need an extra gate (fused-memory,
        dashboard).  Used by the orchestrator coordinator to defer restart
        until the merge pipeline is drained (see
        ``Harness._merge_pipeline_idle``).
    max_executor_failures:
        Number of consecutive TRANSIENT restart-executor failures (any
        exception other than ``FileNotFoundError``/``PermissionError``, e.g. a
        momentary ``systemd-run`` registration hiccup) to tolerate before
        giving up.  Each transient failure below this bound retains pending
        and trigger metadata so the next idle tick retries — symmetric with
        the ``restart_precondition`` fail-safe path.  Once the bound is
        reached, pending and trigger metadata are cleared and a loud ERROR is
        logged (a new ``note_merge`` will re-arm).  The counter resets to 0
        on a successful fire.  Default: 3.
    min_interval_secs:
        Minimum WALL-CLOCK seconds enforced between successive successful
        fires — a restart-safe rate cap on top of the (monotonic, single-burst)
        debounce.  ``0.0`` (the default) DISABLES the cap entirely, preserving
        byte-identical behaviour for coordinators that don't need it
        (fused-memory, dashboard).  When ``> 0`` and the last fire was less
        than ``min_interval_secs`` ago, ``maybe_restart`` defers WITHOUT
        clearing pending (it fires as soon as the window elapses).  Used by the
        orchestrator's own self-redeploy coordinator to cap deploy-on-commit
        churn to ~1 per 8 h.
    wall_clock:
        Injectable callable returning a WALL-CLOCK epoch float (default:
        ``time.time``).  This is deliberately SEPARATE from ``clock`` (the
        monotonic debounce source): the min-interval cap must survive process
        restarts, and ``time.monotonic`` resets to ~0 on every boot, so it
        cannot anchor a persisted last-fire timestamp.  Override in tests for
        determinism.
    state_path:
        Optional path to a JSON file persisting the last successful fire's
        wall-clock timestamp, so the min-interval cap survives an orchestrator
        restart / redeploy.  ``None`` (default) → the cap is in-memory only
        (no persistence).  At construction the file is read (fail-open: a
        missing / corrupt / unreadable file → no seeded timestamp, never
        raises); after each successful fire it is rewritten atomically.
    """

    def __init__(
        self,
        *,
        git_ops: GitOps,
        event_store: EventStore | None,
        watch_prefixes: list[str] | None = None,
        debounce_secs: float = 120.0,
        enabled: bool = True,
        script_path: str = 'scripts/restart-fused-memory.sh',
        project_root: str | Path = '.',
        restart_executor: Callable[[], Awaitable[object]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        service_name: str = 'fused-memory',
        require_idle: bool = True,
        script_args: list[str] | None = None,
        restart_precondition: Callable[[], bool] | None = None,
        max_executor_failures: int = 3,
        min_interval_secs: float = 0.0,
        wall_clock: Callable[[], float] = time.time,
        state_path: Path | None = None,
    ) -> None:
        self._git_ops = git_ops
        self._event_store = event_store
        self._watch_prefixes: list[str] = (
            watch_prefixes if watch_prefixes is not None else ['fused-memory/src/']
        )
        self._debounce_secs = debounce_secs
        self._enabled = enabled
        self._script_path = script_path
        # .resolve() (not a bare Path(...)) so a relative or default ('.')
        # project_root can never reach RestartPlan.__post_init__, which raises
        # ValueError on a non-absolute cwd (task 2237) — see the project_root
        # docstring above.
        self._project_root = Path(project_root).resolve()
        self._restart_executor = restart_executor
        self._clock = clock
        self._service_name = service_name
        # When False (leaf services), the restart fires even while agents are
        # dispatching — agents_idle is not required in the gate.  Default True
        # preserves the existing fused-memory behaviour (idle-only).
        self._require_idle = require_idle
        # script_args are unpacked as positional args after the script path in
        # _default_restart_executor.  None → ['--drain'] to preserve the
        # existing fused-memory spawn signature exactly.
        self._script_args: list[str] = script_args if script_args is not None else ['--drain']
        # Optional extra gate evaluated after enabled/pending/idle/debounce.
        # Default None → the gate check below is skipped entirely, so
        # existing coordinators (fused-memory, dashboard) are unaffected.
        self._restart_precondition = restart_precondition
        # Bound on consecutive TRANSIENT executor failures before giving up.
        self._max_executor_failures = max_executor_failures
        # Restart-safe rate cap (wall-clock). 0.0 → disabled (byte-identical to
        # the pre-cap behaviour). The wall_clock source is intentionally NOT
        # self._clock (monotonic) because the cap must survive a process
        # restart, and monotonic resets on every boot.
        self._min_interval_secs = min_interval_secs
        self._wall_clock = wall_clock
        self._state_path = state_path
        # Persisted epoch of the last successful fire (None → never fired, or
        # no persisted state). Seeded fail-open from state_path at construction:
        # a missing / corrupt / unreadable file must never raise here.
        self._last_fire_wall: float | None = self._load_last_fire_wall()

        # State
        self._pending: bool = False
        self._last_request_monotonic: float = 0.0
        # Accumulated trigger metadata for the current pending burst
        self._trigger_task_ids: list[str] = []
        self._trigger_merge_shas: list[str] = []
        # Consecutive TRANSIENT executor failures since the last successful
        # fire (NOT reset by note_merge re-arming — see maybe_restart).
        self._consecutive_executor_failures: int = 0

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
        *,
        prefetched_diff: list[str] | None = None,
    ) -> bool:
        """Called by ``SpeculativeMergeWorker`` when a merge lands on main.

        Returns True when this merge armed (or re-armed) the pending-restart
        flag; False when it was ignored (disabled, no watched files, git error).

        Parameters
        ----------
        prefetched_diff:
            Optional pre-computed list of changed file paths (base_sha..head_sha).
            When provided, the git diff invocation is skipped and this list is used
            directly, which avoids redundant git calls when multiple coordinators
            are notified for the same merge (see ``Harness._note_merge_all``).
        """
        if not self._enabled:
            return False

        if prefetched_diff is not None:
            changed = prefetched_diff
        else:
            changed, err = await self._git_ops.get_merge_diff_files(base_sha, head_sha)
            if err is not None:
                return False
        # Compute the matched subset once — used for both the gate decision and
        # the log count, avoiding a redundant O(files × prefixes) re-scan.
        watched_changed = [
            f for f in changed if diff_touches_watched_paths([f], self._watch_prefixes)
        ]
        if not watched_changed:
            return False

        # Arm / re-arm
        self._pending = True
        self._last_request_monotonic = self._clock()
        self._trigger_task_ids.append(task_id)
        self._trigger_merge_shas.append(head_sha)
        logger.info(
            f'{self._service_name} restart pending: merge %s for task %s touched %d'
            ' watched file(s)',
            head_sha[:12],
            task_id,
            len(watched_changed),
        )
        return True

    async def maybe_restart(self, *, agents_idle: bool) -> bool:
        """Fire the restart if all gate conditions are met.

        Conditions: enabled AND pending AND agents_idle AND debounce elapsed
        AND (no restart_precondition OR restart_precondition() is truthy).

        Returns True when the restart was fired; False otherwise (no side
        effects — caller may call again on the next idle tick).
        """
        if not (
            self._enabled
            and self._pending
            and (agents_idle or not self._require_idle)
            and (self._clock() - self._last_request_monotonic >= self._debounce_secs)
        ):
            return False

        if self._restart_precondition is not None:
            try:
                satisfied = self._restart_precondition()
            except Exception:
                logger.warning(
                    f'{self._service_name} restart_precondition raised; deferring'
                    ' (fail-safe) — pending stays set for the next idle tick.',
                    exc_info=True,
                )
                return False
            if not satisfied:
                logger.info(
                    f'{self._service_name} restart deferred: restart_precondition'
                    ' not satisfied (pending retained).',
                )
                return False

        # Restart-safe minimum-interval rate cap (wall-clock). Enforced AFTER
        # every other gate so it only ever gates an otherwise-ready fire. When
        # the window hasn't elapsed we defer WITHOUT clearing pending, so the
        # restart fires on the first idle tick after the window opens.
        if (
            self._min_interval_secs > 0
            and self._last_fire_wall is not None
        ):
            elapsed = self._wall_clock() - self._last_fire_wall
            if elapsed < self._min_interval_secs:
                logger.info(
                    f'{self._service_name} skip redeploy: %.0fs since last,'
                    ' %.0fs cap (pending retained).',
                    elapsed,
                    self._min_interval_secs,
                )
                return False

        # Snapshot trigger metadata before clearing
        trigger_task_ids = list(self._trigger_task_ids)
        trigger_merge_shas = list(self._trigger_merge_shas)

        executor = self._restart_executor or self._default_restart_executor
        try:
            await executor()
        except (FileNotFoundError, PermissionError):
            # PERMANENT: a missing or non-executable script will fail identically
            # on every idle tick, so retrying would crash-loop. Clear pending
            # (fail-open) — a new note_merge will re-arm. NOTE: only these two
            # concrete subclasses are treated as permanent — any other OSError
            # (e.g. ENOSPC, EMFILE, a transient subprocess-spawn error) is NOT
            # caught here and falls through to the TRANSIENT branch below.
            logger.warning(
                f'{self._service_name} restart executor failed; clearing pending to avoid'
                ' a crash-loop on a permanently-missing or non-executable script.'
                ' A new note_merge will re-arm.',
                exc_info=True,
            )
            # Clear pending so subsequent idle ticks don't re-attempt and
            # crash the orchestrator (fail-open, matching the design intent).
            self._pending = False
            self._trigger_task_ids = []
            self._trigger_merge_shas = []
            self._consecutive_executor_failures = 0
            return False
        except Exception:
            # TRANSIENT: e.g. a momentary systemd-run registration hiccup
            # (RuntimeError from schedule_detached_systemd_restart). Retain
            # pending and trigger metadata — symmetric with the
            # restart_precondition fail-safe path — so the next idle tick
            # retries instead of silently dropping the restart. Bounded by
            # max_executor_failures: a persistent transient failure degrades
            # loudly instead of retrying forever.
            self._consecutive_executor_failures += 1
            if self._consecutive_executor_failures >= self._max_executor_failures:
                logger.error(
                    f'{self._service_name} restart executor failed'
                    f' {self._consecutive_executor_failures} consecutive times;'
                    ' giving up and clearing pending. A new note_merge will re-arm.',
                    exc_info=True,
                )
                self._pending = False
                self._trigger_task_ids = []
                self._trigger_merge_shas = []
                self._consecutive_executor_failures = 0
                return False
            logger.warning(
                f'{self._service_name} restart executor failed (transient, attempt'
                f' {self._consecutive_executor_failures}/{self._max_executor_failures});'
                ' retaining pending — will retry on the next idle tick.',
                exc_info=True,
            )
            return False

        # Emit observability event
        if self._event_store is not None:
            self._event_store.emit(
                EventType.service_restart,
                task_id=trigger_task_ids[0] if trigger_task_ids else 'unknown',
                phase='service_restart',
                data={
                    'service': self._service_name,
                    'script': str(self._script_path),
                    'trigger_task_ids': trigger_task_ids,
                    'merge_shas': trigger_merge_shas,
                    'reason': f"post_merge_{self._service_name.replace('-', '_')}_code_change",
                },
            )

        logger.warning(
            f'Restarting {self._service_name}.service (post-merge staleness hook):'
            ' triggered by tasks %s, merges %s',
            trigger_task_ids,
            [s[:12] for s in trigger_merge_shas],
        )

        # Clear pending state
        self._pending = False
        self._trigger_task_ids = []
        self._trigger_merge_shas = []
        # A successful fire resets the bound: the counter tracks consecutive
        # failures since the last successful fire, not a lifetime tally.
        # (note_merge re-arming does NOT reset it — only success/exhaustion do.)
        self._consecutive_executor_failures = 0

        # Stamp the min-interval rate cap AFTER the fire succeeded, then persist
        # it (restart-safe). Persist failures are non-fatal: the restart already
        # happened, so a lost timestamp merely relaxes the cap once — never
        # blocks the pipeline.
        self._last_fire_wall = self._wall_clock()
        self._persist_last_fire_wall(self._last_fire_wall)

        return True

    async def _default_restart_executor(self) -> None:
        """Fire-and-forget: spawn the restart script detached.

        Thin ``proc_supervision.RestartPlan`` caller (M1, task 2237) —
        fused-memory/dashboard LEAF restart. Positional args after the
        script path are taken from ``self._script_args`` (e.g. ``['--drain']``
        for fused-memory, ``[]`` for the dashboard). ``target_unit``/
        ``own_unit`` are both ``''`` (unused for this shape: ``verify=None``
        and ``transient_unit=None`` route ``RestartPlan.execute()`` straight
        to the leaf plain-spawn path, which never consults them).
        """
        plan = RestartPlan(
            script=self._project_root / self._script_path,
            args=list(self._script_args),
            cwd=self._project_root,
            target_unit='',
            own_unit='',
            on_failure_escalation=None,
            verify=None,
            transient_unit=None,
        )
        await plan.execute()
        # Intentionally NOT awaiting the process exit — the script runs
        # detached so its health-poll never blocks the orchestrator event loop.
        # (RestartPlan.execute()'s leaf plain-spawn path does not await it
        # either — see proc_supervision._execute_detached_leaf_plain_spawn.)

    # ------------------------------------------------------------------
    # Min-interval rate-cap persistence (restart-safe)
    # ------------------------------------------------------------------

    def _load_last_fire_wall(self) -> float | None:
        """Read the persisted last-fire epoch from ``state_path`` (fail-open).

        Returns the stored ``ts`` float, or ``None`` when there is no
        ``state_path``, the file is missing, or the file is corrupt /
        unreadable / malformed.  MUST never raise — a broken state file simply
        relaxes the cap (treated as "never fired") rather than crashing
        construction.
        """
        if self._state_path is None:
            return None
        try:
            raw = json.loads(self._state_path.read_text(encoding='utf-8'))
            ts = raw['ts']
            return float(ts)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.warning(
                '%s restart: ignoring unreadable/corrupt rate-cap state at %s: %s',
                self._service_name,
                self._state_path,
                exc,
            )
            return None

    def _persist_last_fire_wall(self, ts: float) -> None:
        """Atomically persist the last-fire epoch to ``state_path`` (non-fatal).

        Mirrors ``MergeQueueStore._save_raw``: lazy ``mkdir(parents=True)`` +
        tmp-file write + ``os.replace``.  A failure here is swallowed to a
        WARNING — the restart already fired, so a lost timestamp must not block
        anything.
        """
        if self._state_path is None:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'ts': ts,
                'iso': datetime.fromtimestamp(ts, tz=UTC).isoformat(),
            }
            tmp = self._state_path.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(payload), encoding='utf-8')
            tmp.replace(self._state_path)
        except OSError as exc:
            logger.warning(
                '%s restart: failed to persist rate-cap state to %s: %s',
                self._service_name,
                self._state_path,
                exc,
            )
