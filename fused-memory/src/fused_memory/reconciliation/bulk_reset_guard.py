"""Defence-in-depth bulk-reset circuit-breaker (task 918).

Detects bursts of done→pending or in-progress→pending task-status reversals
within a sliding time window and halts further reversals once the configured
threshold is crossed.

Background: Two autopilot_video bulk resets on 2026-04-21 and 2026-04-22
pushed large batches of tasks from done/in-progress back to pending despite
the orchestrator's ``_safe_stash_pop_with_recovery`` fix already being
deployed.  This guard is a second line of defence at the fused-memory
reconciliation layer: it limits blast radius when the primary prevention fails.

Architecture mirrors :class:`~fused_memory.reconciliation.backlog_policy.BacklogPolicy`:
  - Constructed with config knobs (enabled, threshold, window_seconds, …).
  - Passed into :class:`~fused_memory.middleware.task_interceptor.TaskInterceptor`.
  - Called via ``observe_attempt`` in ``_apply_status_transition`` *before*
    the terminal-exit gate so it catches both legitimate and illegitimate
    reversal patterns.
  - Emits L1 escalation JSON under ``<project_root>/data/escalations/``.
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

def _is_reversal(old_status: str, new_status: str) -> bool:
    """Return True iff the transition is a guarded reversal.

    Exactly two patterns qualify:
      * ``done``        → ``pending``
      * ``in-progress`` → ``pending``

    All other transitions (including blocked→pending) are not reversals and
    do not consume window slots.
    """
    return new_status == 'pending' and old_status in ('done', 'in-progress')


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BulkResetVerdict:
    """Outcome of a single :meth:`BulkResetGuard.observe_attempt` call."""

    outcome: Literal['ok', 'rejection', 'escalated']
    affected_task_ids: tuple[str, ...] = ()
    triggering_timestamps: tuple[str, ...] = ()
    threshold: int = 0
    window_seconds: float = 0.0
    project_id: str = ''
    error_type: str = 'BulkResetGuardTripped'
    escalation_path: str | None = None

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
    entries: deque[_Entry] = field(default_factory=deque)
    last_escalation_ts: float = 0.0


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
    threshold:
        Number of reversal attempts within ``window_seconds`` that trips the
        circuit.  Up to ``threshold`` reversals are allowed; the
        ``(threshold+1)``-th attempt in the window is the first to be
        rejected.  For example, ``threshold=10`` means the 11th reversal in
        the window is the first rejection.
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
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        threshold: int = 10,
        window_seconds: float = 60.0,
        escalation_rate_limit_seconds: float = 900.0,
        time_provider: Callable[[], float] = time.time,
        escalations_fallback_dir: Path | None = None,
    ) -> None:
        self._enabled = enabled
        self._threshold = threshold
        self._window_seconds = window_seconds
        self._rate_limit_seconds = escalation_rate_limit_seconds
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
        if not _is_reversal(old_status, new_status):
            return BulkResetVerdict(outcome='ok', project_id=project_id)

        now = self._now()

        async with self._lock:
            state = self._state.setdefault(project_id, _GuardState())

            # 3. Prune expired entries.
            cutoff = now - self._window_seconds
            while state.entries and state.entries[0].ts < cutoff:
                state.entries.popleft()

            # 3a. Evict idle per-project state to prevent unbounded map growth.
            # A project is considered idle when its window is empty AND the last
            # escalation is older than the rate-limit period (no imminent re-trip).
            if not state.entries and (now - state.last_escalation_ts) >= self._rate_limit_seconds:
                self._state.pop(project_id, None)
                # Re-acquire a fresh state for this attempt (setdefault below adds it back).
                state = _GuardState()
                self._state[project_id] = state

            # 4. Record this attempt (always, even if we will reject it).
            state.entries.append(_Entry(ts=now, task_id=task_id))

            # 5. Check threshold.
            # Trip fires when the count EXCEEDS threshold (len > threshold),
            # i.e. the (threshold+1)-th attempt in the window is the first
            # rejection.  This means "up to threshold reversals are allowed;
            # the next one trips the circuit-breaker."  Steps 5, 13, 15 all
            # assume threshold=3 allows exactly three ok reversals.
            if len(state.entries) <= self._threshold:
                return BulkResetVerdict(outcome='ok', project_id=project_id)

            # 6. Threshold crossed — collect window contents for the verdict.
            affected_ids = tuple(e.task_id for e in state.entries)
            trig_ts = tuple(
                datetime.fromtimestamp(e.ts, tz=UTC).isoformat()
                for e in state.entries
            )

        # 7. Try to write escalation (outside lock to avoid I/O under lock).
        esc_path = await self._maybe_write_escalation(
            project_id=project_id,
            project_root=project_root,
            affected_task_ids=affected_ids,
            triggering_timestamps=trig_ts,
        )

        if esc_path is not None:
            return BulkResetVerdict(
                outcome='escalated',
                affected_task_ids=affected_ids,
                triggering_timestamps=trig_ts,
                threshold=self._threshold,
                window_seconds=self._window_seconds,
                project_id=project_id,
                escalation_path=str(esc_path),
            )
        return BulkResetVerdict(
            outcome='rejection',
            affected_task_ids=affected_ids,
            triggering_timestamps=trig_ts,
            threshold=self._threshold,
            window_seconds=self._window_seconds,
            project_id=project_id,
        )

    # ------------------------------------------------------------------
    # Escalation write
    # ------------------------------------------------------------------

    async def _maybe_write_escalation(
        self,
        *,
        project_id: str,
        project_root: str,
        affected_task_ids: tuple[str, ...],
        triggering_timestamps: tuple[str, ...],
    ) -> Path | None:
        """Write an L1 escalation JSON unless rate-limited.

        Returns the path on success, or ``None`` when rate-limited / write
        failed (callers produce ``outcome='rejection'`` in that case).
        """
        now = self._now()

        async with self._lock:
            state = self._state.setdefault(project_id, _GuardState())
            if (now - state.last_escalation_ts) < self._rate_limit_seconds:
                logger.info(
                    'bulk_reset_guard: rate-limited escalation for %s '
                    '(%.0fs since last)',
                    project_id,
                    now - state.last_escalation_ts,
                )
                return None
            # NOTE: last_escalation_ts is updated AFTER a successful write (below)
            # so that a disk failure does not silently suppress the next escalation
            # for the full rate-limit period (900 s by default).

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
        await asyncio.to_thread(esc_dir.mkdir, parents=True, exist_ok=True)

        ts = datetime.fromtimestamp(now, tz=UTC).isoformat()
        safe_ts = ts.replace(':', '').replace('+', '').replace('.', '_')
        # Include a sanitised project_id in the filename so two projects that trip
        # within the same microsecond (or whose timestamps collide after stripping)
        # do not silently overwrite each other.
        safe_pid = ''.join(c if c.isalnum() or c in '-_' else '_' for c in project_id)
        esc_id = f'esc-bulk-reset-{safe_pid}-{safe_ts}'
        path = esc_dir / f'{esc_id}.json'

        record = {
            'id': esc_id,
            'task_id': None,
            'agent_role': 'fused-memory',
            'severity': 'blocking',
            'category': 'infra_issue',
            'summary': (
                f'Bulk task-status reversal detected for {project_id}: '
                f'{len(affected_task_ids)} reversals in {self._window_seconds}s '
                f'(threshold={self._threshold})'
            ),
            'detail': (
                f'The bulk-reset circuit-breaker (task 918) tripped for project '
                f'{project_id}. {len(affected_task_ids)} done→pending or '
                f'in-progress→pending reversal attempts landed within a '
                f'{self._window_seconds}s window, exceeding the threshold of '
                f'{self._threshold}. The first triggering attempt was at '
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
            'threshold': self._threshold,
            'window_seconds': self._window_seconds,
            'project_id': project_id,
        }
        try:
            await asyncio.to_thread(path.write_text, json.dumps(record, indent=2), encoding='utf-8')
            logger.warning(
                'bulk_reset_guard: wrote L1 escalation %s '
                '(affected=%d, threshold=%d)',
                path,
                len(affected_task_ids),
                self._threshold,
            )
        except OSError as exc:
            logger.error(
                'bulk_reset_guard: failed to write escalation %s: %s', path, exc,
            )
            return None

        # Only advance the rate-limit timestamp AFTER a successful write.
        # This ensures a disk hiccup does not lock out escalations for 900 s.
        async with self._lock:
            state2 = self._state.setdefault(project_id, _GuardState())
            state2.last_escalation_ts = now

        return path
