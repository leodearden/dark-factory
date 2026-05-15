"""Digest helpers for AFK hardening — per-N-escalation markdown digests + EWA trip.

Task 1327: Every N escalation events the harness writes an append-only markdown
digest summarising recent activity, and tracks an EWA of escalations/done that
pauses the scheduler when it trips.

All I/O in this module is fail-open: helpers return sentinels / zeros / None
and log warnings rather than raising.  The digest is observability, not a
correctness gate.

Design decisions (see plan.json):
- Pure, Harness-free helpers here; harness.py owns the trigger and state.
- EWA state is process-local (reset on restart — consistent with park-stop counters).
- write_digest_entry never raises; digest_dir is auto-created if missing.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from escalation.models import Escalation
from escalation.queue import iter_all_escalation_paths

logger = logging.getLogger(__name__)

# The 'done' outcome value — must match WorkflowOutcome.DONE.value in workflow.py.
# Kept as a local constant to avoid importing the full workflow module in a digest helper.
_DONE_OUTCOME = 'done'

# ---------------------------------------------------------------------------
# EWA math
# ---------------------------------------------------------------------------


def update_ewa(
    prev_ewa: float,
    escalations_in_step: int,
    done_in_step: int,
    alpha: float,
) -> float:
    """Compute one EWA step.

    EWA(t+1) = alpha * (escalations_in_step / max(done_in_step, 1))
               + (1 - alpha) * prev_ewa

    done_in_step == 0 uses denominator 1 so a step with escalations and zero
    completions (the worst-case signal) pushes EWA up rather than crashing.

    No exception handling — pure arithmetic.
    """
    ratio = escalations_in_step / max(done_in_step, 1)
    return alpha * ratio + (1 - alpha) * prev_ewa


# ---------------------------------------------------------------------------
# Escalation aggregation
# ---------------------------------------------------------------------------


@dataclass
class EscalationStats:
    """Aggregated escalation statistics for a digest window."""

    # Mapping of (category, level, status) → count for in-window escalations
    category_level_status_counts: dict[tuple[str, int, str], int] = field(default_factory=dict)
    # ISO timestamps of first and last in-window escalations (None if empty)
    first_ts: str | None = None
    last_ts: str | None = None
    # Total count of dedupe children across all in-window escalations
    dedupe_children_total: int = 0
    # Count of in-window escalations with status == 'pending'
    pending_total: int = 0


def aggregate_escalations(
    escalations_dir: Path,
    window_start_iso: str,
    window_end_iso: str,
) -> EscalationStats:
    """Aggregate escalation statistics for the given time window.

    Walks escalations_dir (root + archive subtree) using iter_all_escalation_paths
    which handles root-precedence deduplication. Filters by
    window_start_iso <= esc.timestamp <= window_end_iso.

    Fail-open: on any per-file error, logs a warning and continues.
    """
    stats = EscalationStats()

    for path in iter_all_escalation_paths(escalations_dir):
        try:
            esc = Escalation.from_json(path.read_text())
        except Exception:
            logger.warning('aggregate_escalations: failed to parse %s', path, exc_info=True)
            continue

        ts = esc.timestamp or ''
        if not ts:
            continue
        if ts < window_start_iso or ts > window_end_iso:
            continue

        # Update counts
        key = (esc.category, esc.level, esc.status)
        stats.category_level_status_counts[key] = (
            stats.category_level_status_counts.get(key, 0) + 1
        )

        # Track first / last timestamps in window
        if stats.first_ts is None or ts < stats.first_ts:
            stats.first_ts = ts
        if stats.last_ts is None or ts > stats.last_ts:
            stats.last_ts = ts

        # Dedupe children
        stats.dedupe_children_total += len(esc.dedupe_children or [])

        # Pending
        if esc.status == 'pending':
            stats.pending_total += 1

    return stats


# ---------------------------------------------------------------------------
# Done count from EventStore
# ---------------------------------------------------------------------------


def count_done_in_window(
    events_db: Path,
    window_start_iso: str,
    window_end_iso: str,
) -> int:
    """Count task_completed events with outcome='done' inside the given window.

    Uses sqlite3 directly (read-only).  Fail-open: any exception returns 0.
    Missing DB returns 0.
    """
    try:
        conn = sqlite3.connect(str(events_db))
        try:
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM events "
                "WHERE event_type = 'task_completed' "
                "  AND timestamp BETWEEN ? AND ? "
                "  AND json_extract(data, '$.outcome') = ?",
                (window_start_iso, window_end_iso, _DONE_OUTCOME),
            ).fetchone()
            return int(count)
        finally:
            conn.close()
    except Exception:
        logger.debug('count_done_in_window: failed (fail-open)', exc_info=True)
        return 0
