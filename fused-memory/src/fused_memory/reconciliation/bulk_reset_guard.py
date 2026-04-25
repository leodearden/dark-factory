"""Defence-in-depth bulk-reset circuit-breaker (task 918, refined task 1016).

Detects bursts of task-status reversals within a sliding time window and
halts further reversals once the configured per-kind threshold is crossed.

Background
----------
Two autopilot_video bulk resets on 2026-04-21 and 2026-04-22 pushed large
batches of tasks from done/in-progress back to pending despite the
orchestrator's ``_safe_stash_pop_with_recovery`` fix already being deployed.
This guard is a second line of defence at the fused-memory reconciliation
layer: it limits blast radius when the primary prevention fails.

Task 1016 motivation
--------------------
The 2026-04-24 reify startup-stranded-task reconciler reverted 27
in-progress tasks to pending in ~2 s (escalation id:
``esc-bulk-reset-reify-2026-04-24T070944_6456580000``).  The original
single shared threshold (10 / 60 s) fired even though zero done→pending
transitions occurred.  Task 1016 splits the shared counter into two
independent per-kind counters with independent thresholds:
  - ``done_to_pending`` (default 10/60 s): catches the March-2026
    ``advance_main`` data-loss pattern (task 918).
  - ``in_progress_to_pending`` (default 100/60 s): allows the 27-task
    startup stranded-task reconcile while still catching pathological runaways.

Architecture mirrors :class:`~fused_memory.reconciliation.backlog_policy.BacklogPolicy`:
  - Constructed with config knobs (enabled, done_threshold,
    in_progress_threshold, window_seconds, …).
  - Passed into :class:`~fused_memory.middleware.task_interceptor.TaskInterceptor`.
  - Called via ``observe_attempt`` in ``_apply_status_transition`` *before*
    the terminal-exit gate so it catches both legitimate and illegitimate
    reversal patterns.
  - Emits L1 escalation JSON under ``<project_root>/data/escalations/``
    with a kind slug (``done`` or ``in-progress``) in the filename and a
    ``kind`` field in the JSON body for at-a-glance triage.

Split-counter design
--------------------
Each ``_GuardState`` holds two independent sliding-window deques:
  - ``done_entries``        — accumulates done→pending reversals
  - ``in_progress_entries`` — accumulates in-progress→pending reversals

The shared per-project scalars ``last_escalation_ts`` and
``last_write_failure_ts`` are NOT split per-kind: a simultaneous storm of
both kinds would, with per-kind rate limits, emit two escalations within
seconds.  Sharing the rate limit preserves the "no escalation-directory
flood" guarantee; the ``kind`` field on the verdict and escalation file
still lets operators distinguish which counter tripped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _reversal_kind(
    old_status: str, new_status: str
) -> Literal['done_to_pending', 'in_progress_to_pending'] | None:
    """Classify a status transition into a guarded reversal kind, or None.

    Exactly two patterns qualify:
      * ``done``        → ``pending``  → ``'done_to_pending'``
      * ``in-progress`` → ``pending``  → ``'in_progress_to_pending'``

    All other transitions (including ``blocked``→``pending``) are not
    reversals and do not consume window slots.  Returns ``None`` for those.
    """
    if new_status != 'pending':
        return None
    if old_status == 'done':
        return 'done_to_pending'
    if old_status == 'in-progress':
        return 'in_progress_to_pending'
    return None


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BulkResetVerdict:
    """Outcome of a single :meth:`BulkResetGuard.observe_attempt` call.

    Attributes
    ----------
    kind:
        The reversal kind that tripped the guard: ``'done_to_pending'`` or
        ``'in_progress_to_pending'``.  ``None`` for ``outcome='ok'`` verdicts
        where the circuit-breaker was not tripped.
    """

    outcome: Literal['ok', 'rejection', 'escalated']
    affected_task_ids: tuple[str, ...] = ()
    triggering_timestamps: tuple[str, ...] = ()
    threshold: int = 0
    window_seconds: float = 0.0
    project_id: str = ''
    error_type: str = 'BulkResetGuardTripped'
    escalation_path: str | None = None
    kind: Literal['done_to_pending', 'in_progress_to_pending'] | None = None

    @property
    def is_rejection(self) -> bool:
        """True for both ``rejection`` and ``escalated`` outcomes."""
        return self.outcome in ('rejection', 'escalated')

    # Cap the number of task IDs and timestamps included in the MCP error payload.
    # During a bulk-reset storm the deque can grow large; serialising thousands of
    # IDs per per-id error response wastes bandwidth.  Stewards should consult the
    # escalation file for the full list.
    _MAX_PAYLOAD_IDS: int = 50

    def to_error_dict(self) -> dict:
        """Structured MCP error payload.

        Returns an empty dict for ``ok`` outcomes.  Returns a dict with
        ``success=False`` for ``rejection`` / ``escalated`` outcomes so the
        task interceptor's CSV aggregator flags the attempt correctly.

        ``affected_task_ids`` and ``triggering_timestamps`` are capped at
        ``_MAX_PAYLOAD_IDS`` entries to bound response size during storms;
        ``affected_task_ids_total`` carries the full count.
        """
        if self.outcome == 'ok':
            return {}
        total = len(self.affected_task_ids)
        task_ids = list(self.affected_task_ids[:self._MAX_PAYLOAD_IDS])
        timestamps = list(self.triggering_timestamps[:self._MAX_PAYLOAD_IDS])
        payload: dict = {
            'success': False,
            'error': (
                f'BulkResetGuardTripped: {total} reversal '
                f'attempts within {self.window_seconds}s window exceeded threshold '
                f'{self.threshold} for project {self.project_id}'
            ),
            'error_type': self.error_type,
            'affected_task_ids': task_ids,
            'affected_task_ids_total': total,
            'triggering_timestamps': timestamps,
            'threshold': self.threshold,
            'window_seconds': self.window_seconds,
            'project_id': self.project_id,
            'hint': (
                'The bulk-reset circuit-breaker (task 918) has tripped. '
                'This is a defence-in-depth guard: no further reversals will '
                'be applied until the sliding window expires. '
                'Inspect the escalation file (if present), review the affected '
                'task IDs, and manually un-reverse any tasks that were '
                'incorrectly reset.'
            ),
        }
        if self.outcome == 'escalated' and self.escalation_path is not None:
            payload['escalation_path'] = self.escalation_path
        if self.kind is not None:
            payload['kind'] = self.kind
        return payload


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

@dataclass
class _Entry:
    ts: float
    task_id: str


@dataclass
class _GuardState:
    done_entries: deque[_Entry] = field(default_factory=deque)
    in_progress_entries: deque[_Entry] = field(default_factory=deque)
    last_escalation_ts: float = 0.0
    last_write_failure_ts: float = 0.0


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

class BulkResetGuard:
    """Sliding-window circuit-breaker that halts bulk task-status reversals.

    Parameters
    ----------
    enabled:
        When ``False`` every call returns ``ok`` immediately; no state is
        mutated.  Lets operators disable the guard without code changes.
    done_threshold:
        Number of ``done``→``pending`` reversal attempts within
        ``window_seconds`` that trips the circuit.  Up to ``done_threshold``
        reversals are allowed; the ``(done_threshold+1)``-th attempt is the
        first rejection.  Default 10 — inherited from the original
        single-threshold design (task 918).
    in_progress_threshold:
        Number of ``in-progress``→``pending`` reversal attempts within
        ``window_seconds`` that trips the circuit.  Default 100 — set
        comfortably above the 27-task startup stranded-task reconcile seen
        in the 2026-04-24 reify incident while still catching pathological
        runaways.
    window_seconds:
        Length of the sliding window in seconds.  Entries older than
        ``now - window_seconds`` are pruned before each check.
    escalation_rate_limit_seconds:
        Minimum gap between escalation file writes for the same project.
        Subsequent trips within this period return ``rejection`` instead of
        ``escalated`` so the escalation directory does not fill up.
    time_provider:
        Injectable clock (``Callable[[], float]``) — defaults to
        ``time.time``.  Tests inject a fake clock for deterministic window
        tests.
    escalations_fallback_dir:
        Optional fallback directory for escalation writes.  Used as the base
        of ``<fallback_dir>/data/escalations/`` when ``project_root`` is
        empty or ``None``.  Production paths always supply ``project_root``
        via ``observe_attempt``; this parameter is primarily for tests that
        construct the guard without a real project tree.
    write_failure_backoff_seconds:
        Minimum gap between write attempts for the same project after an
        ``OSError`` from ``path.write_text`` or ``esc_dir.mkdir``.  During
        this backoff window subsequent trips return ``rejection`` without
        retrying the write, preventing tight-loop retries on a flaky mount.
        Defaults to ``60.0`` s — shorter than ``escalation_rate_limit_seconds``
        so transient failures recover quickly once the mount stabilises.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        done_threshold: int = 10,
        in_progress_threshold: int = 100,
        window_seconds: float = 60.0,
        escalation_rate_limit_seconds: float = 900.0,
        write_failure_backoff_seconds: float = 60.0,
        time_provider: Callable[[], float] = time.time,
        escalations_fallback_dir: Path | None = None,
    ) -> None:
        self._enabled = enabled
        self._done_threshold = done_threshold
        self._in_progress_threshold = in_progress_threshold
        self._window_seconds = window_seconds
        self._rate_limit_seconds = escalation_rate_limit_seconds
        self._write_failure_backoff_seconds = write_failure_backoff_seconds
        self._now = time_provider
        self._fallback_dir = escalations_fallback_dir
        self._state: dict[str, _GuardState] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def observe_attempt(
        self,
        *,
        project_id: str,
        task_id: str,
        old_status: str,
        new_status: str,
        project_root: str,
    ) -> BulkResetVerdict:
        """Observe one status-transition attempt and enforce the guard.

        Must be called *before* the actual taskmaster mutation so that the
        guard can short-circuit the mutation on rejection.

        Returns
        -------
        BulkResetVerdict
            ``outcome='ok'``        — transition is allowed.
            ``outcome='escalated'`` — threshold crossed; escalation file written.
            ``outcome='rejection'`` — threshold crossed; escalation rate-limited
                                      or write failed.
        """
        # 1. Disabled fast-path — no state touched.
        if not self._enabled:
            return BulkResetVerdict(outcome='ok', project_id=project_id)

        # 2. Non-reversal fast-path — ignore; do not touch the deque.
        kind = _reversal_kind(old_status, new_status)
        if kind is None:
            return BulkResetVerdict(outcome='ok', project_id=project_id)

        now = self._now()

        # Lock-order contract: callers (notably
        # ``TaskInterceptor._apply_status_transition``) may already hold a
        # per-project ``_write_lock`` when invoking this method.
        # ``observe_attempt`` therefore acquires only ``self._lock`` (a
        # guard-internal global asyncio.Lock) and MUST NOT attempt to
        # acquire any per-project lock — doing so would invert the
        # established order (per-project → guard-global) and risk deadlock
        # under contention with concurrent writers on other projects.
        # All work inside ``self._lock`` is bounded O(window) deque ops.
        async with self._lock:
            state = self._state.setdefault(project_id, _GuardState())

            # 3. Prune expired entries.
            cutoff = now - self._window_seconds
            while state.done_entries and state.done_entries[0].ts < cutoff:
                state.done_entries.popleft()
            while state.in_progress_entries and state.in_progress_entries[0].ts < cutoff:
                state.in_progress_entries.popleft()

            # 4. Route to the per-kind deque and select the matching threshold.
            if kind == 'done_to_pending':
                entries = state.done_entries
                threshold = self._done_threshold
            else:
                entries = state.in_progress_entries
                threshold = self._in_progress_threshold

            # 5. Record this attempt (always, even if we will reject it).
            entries.append(_Entry(ts=now, task_id=task_id))

            # 6. Check threshold.
            # Trip fires when the count EXCEEDS threshold (len > threshold),
            # i.e. the (threshold+1)-th attempt in the window is the first
            # rejection.  This means "up to threshold reversals are allowed;
            # the next one trips the circuit-breaker."
            if len(entries) <= threshold:
                return BulkResetVerdict(outcome='ok', project_id=project_id)

            # 7. Threshold crossed — collect window contents for the verdict.
            # Only entries from the tripped deque are included in
            # affected_task_ids; the other deque's IDs are captured separately
            # so the escalation JSON can surface cross-kind context for
            # operators triaging the incident without mixing the two counters.
            affected_ids = tuple(e.task_id for e in entries)
            trig_ts = tuple(
                datetime.fromtimestamp(e.ts, tz=UTC).isoformat()
                for e in entries
            )
            # Cross-kind window context (may be empty if the other counter has
            # not accumulated any entries within the current window).
            other_entries = (
                state.in_progress_entries
                if kind == 'done_to_pending'
                else state.done_entries
            )
            other_kind_ids = tuple(e.task_id for e in other_entries)

        # 8. Try to write escalation (outside lock to avoid I/O under lock).
        esc_path = await self._maybe_write_escalation(
            project_id=project_id,
            project_root=project_root,
            affected_task_ids=affected_ids,
            triggering_timestamps=trig_ts,
            other_kind_task_ids=other_kind_ids,
            kind=kind,
        )

        if esc_path is not None:
            return BulkResetVerdict(
                outcome='escalated',
                affected_task_ids=affected_ids,
                triggering_timestamps=trig_ts,
                threshold=threshold,
                window_seconds=self._window_seconds,
                project_id=project_id,
                escalation_path=str(esc_path),
                kind=kind,
            )
        return BulkResetVerdict(
            outcome='rejection',
            affected_task_ids=affected_ids,
            triggering_timestamps=trig_ts,
            threshold=threshold,
            window_seconds=self._window_seconds,
            project_id=project_id,
            kind=kind,
        )

    # ------------------------------------------------------------------
    # Escalation write
    # ------------------------------------------------------------------

    async def _record_write_failure(self, project_id: str) -> None:
        """Record a write-failure timestamp for per-project backoff tracking.

        The timestamp is captured fresh at call time (i.e. after the failed I/O
        has already returned) so the post-I/O elapsed time is included in the
        backoff window.  This prevents a long, slow I/O failure from appearing
        'recent' when measured against a stale timestamp captured before the
        operation started.

        Acquires the lock briefly so the update is visible to all concurrent
        callers of ``_maybe_write_escalation``.
        """
        now = self._now()
        async with self._lock:
            state = self._state.setdefault(project_id, _GuardState())
            state.last_write_failure_ts = now

    async def _maybe_write_escalation(
        self,
        *,
        project_id: str,
        project_root: str,
        affected_task_ids: tuple[str, ...],
        triggering_timestamps: tuple[str, ...],
        other_kind_task_ids: tuple[str, ...] = (),
        kind: Literal['done_to_pending', 'in_progress_to_pending'],
    ) -> Path | None:
        """Write an L1 escalation JSON unless rate-limited.

        Returns the path on success, or ``None`` when rate-limited / write
        failed (callers produce ``outcome='rejection'`` in that case).

        The I/O try/except blocks catch ``Exception`` (not ``OSError``) on
        purpose so that any unexpected error (e.g. ValueError from an odd
        encoding, AttributeError from a stub Path subclass) is converted to a
        ``'rejection'`` verdict rather than propagating out of
        ``observe_attempt``.  ``KeyboardInterrupt`` and ``SystemExit`` are NOT
        caught — ``Exception`` (not ``BaseException``) is used deliberately.

        Parameters
        ----------
        other_kind_task_ids:
            Task IDs from the *other* reversal kind's deque that were active
            in the current window at trip time.  Recorded in the escalation
            JSON as ``other_kind_task_ids_in_window`` so operators triaging
            the incident have full window context without needing to inspect
            the guard's in-memory state.
        """
        now = self._now()

        async with self._lock:
            state = self._state.setdefault(project_id, _GuardState())
            # NOTE: last_escalation_ts is updated AFTER a successful write (below)
            # so that a disk failure does not silently suppress the next escalation
            # for the full rate-limit period (900 s by default).
            if (now - state.last_escalation_ts) < self._rate_limit_seconds:
                # Use WARNING (not INFO) because the suppressed escalation is a
                # real incident event — an operator may be investigating why no
                # file was written for a kind that tripped inside the rate-limit
                # window.  The log message names the kind explicitly so the
                # operator knows which counter was silenced.
                logger.warning(
                    'bulk_reset_guard: escalation suppressed (rate-limited) for %s '
                    '[kind=%s, %.0fs since last write, limit=%.0fs] — '
                    'no new escalation file will be written; '
                    'check the most recent escalation file for this project',
                    project_id,
                    kind,
                    now - state.last_escalation_ts,
                    self._rate_limit_seconds,
                )
                return None
            # Write-failure backoff: suppress I/O retries for
            # write_failure_backoff_seconds after an OSError from mkdir or
            # write_text.  last_write_failure_ts is set by _record_write_failure
            # and is per-project so a flaky mount for one project does not
            # block sibling projects.
            if (now - state.last_write_failure_ts) < self._write_failure_backoff_seconds:
                logger.info(
                    'bulk_reset_guard: write-failure backoff active for %s '
                    '(%.0fs since last failure)',
                    project_id,
                    now - state.last_write_failure_ts,
                )
                return None

        # Use project_root when available; fall back to the constructor-supplied dir.
        _esc_base: Path | None = None
        if project_root:
            _esc_base = Path(project_root) / 'data' / 'escalations'
        elif self._fallback_dir is not None:
            _esc_base = self._fallback_dir / 'data' / 'escalations'

        if _esc_base is None:
            logger.error('bulk_reset_guard: no escalation directory available for %s', project_id)
            return None

        esc_dir = _esc_base
        try:
            await asyncio.to_thread(esc_dir.mkdir, parents=True, exist_ok=True)
        except Exception:
            # Intentional broad catch: the guard's contract is 'never let
            # reconciliation fail because escalation I/O failed'.  A non-OSError
            # (e.g. unexpected encoding, AttributeError from a stub Path subclass)
            # must NOT propagate out of observe_attempt.  logger.exception preserves
            # the full traceback so operators can still diagnose the root cause.
            logger.exception(
                'bulk_reset_guard: failed to create escalation dir %s', esc_dir,
            )
            await self._record_write_failure(project_id)
            return None

        ts = datetime.fromtimestamp(now, tz=UTC).isoformat()
        safe_ts = ts.replace(':', '').replace('+', '').replace('.', '_')
        # Build a kind slug for the filename so operators can distinguish
        # data-loss events (done) from benign startup-reconcile runaways
        # (in-progress) without opening the file.
        kind_slug = 'done' if kind == 'done_to_pending' else 'in-progress'
        tripped_threshold = (
            self._done_threshold if kind == 'done_to_pending'
            else self._in_progress_threshold
        )
        # Include a sanitised project_id in the filename so two projects that trip
        # within the same microsecond (or whose timestamps collide after stripping)
        # do not silently overwrite each other.
        safe_pid = ''.join(c if c.isalnum() or c in '-_' else '_' for c in project_id)
        esc_id = f'esc-bulk-reset-{kind_slug}-{safe_pid}-{safe_ts}'
        path = esc_dir / f'{esc_id}.json'

        record = {
            'id': esc_id,
            'task_id': None,
            'agent_role': 'fused-memory',
            'severity': 'blocking',
            'category': 'infra_issue',
            'kind': kind,
            'summary': (
                f'Bulk task-status reversal detected for {project_id}: '
                f'{len(affected_task_ids)} {kind_slug}→pending reversals in '
                f'{self._window_seconds}s (threshold={tripped_threshold})'
            ),
            'detail': (
                f'The bulk-reset circuit-breaker (task 918) tripped for project '
                f'{project_id}. {len(affected_task_ids)} {kind_slug}→pending '
                f'reversal attempts landed within a {self._window_seconds}s window, '
                f'exceeding the {kind_slug}→pending threshold of {tripped_threshold}. '
                f'The first triggering attempt was at '
                f'{triggering_timestamps[0] if triggering_timestamps else "unknown"}.'
            ),
            'suggested_action': (
                'Inspect the affected_task_ids; manually revert any tasks that '
                'were incorrectly reset. Check orchestrator logs for a bulk-reset '
                'trigger and ensure _safe_stash_pop_with_recovery is behaving '
                'correctly.'
            ),
            'timestamp': ts,
            'status': 'pending',
            'level': 1,
            'workflow_state': 'infra',
            'affected_task_ids': list(affected_task_ids),
            'triggering_timestamps': list(triggering_timestamps),
            # Cross-kind context: task IDs from the other reversal kind that
            # were active in the same window at trip time.  May be empty when
            # only one kind was active.  Provided so operators can see the full
            # window picture without needing the guard's in-memory state.
            'other_kind_task_ids_in_window': list(other_kind_task_ids),
            'threshold': tripped_threshold,
            'thresholds': {
                'done_to_pending': self._done_threshold,
                'in_progress_to_pending': self._in_progress_threshold,
            },
            'window_seconds': self._window_seconds,
            'project_id': project_id,
        }
        try:
            await asyncio.to_thread(path.write_text, json.dumps(record, indent=2), encoding='utf-8')
            logger.warning(
                'bulk_reset_guard: wrote L1 escalation %s '
                '(kind=%s, affected=%d, threshold=%d)',
                path,
                kind,
                len(affected_task_ids),
                tripped_threshold,
            )
        except Exception:
            # Same intentional broad catch as the mkdir block above — see docstring.
            logger.exception(
                'bulk_reset_guard: failed to write escalation %s', path,
            )
            await self._record_write_failure(project_id)
            return None

        # Only advance the rate-limit timestamp AFTER a successful write.
        # This ensures a disk hiccup does not lock out escalations for 900 s.
        async with self._lock:
            state2 = self._state.setdefault(project_id, _GuardState())
            state2.last_escalation_ts = now

        return path
