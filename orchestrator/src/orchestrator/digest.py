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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shared.cost_store import CostStore

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


# ---------------------------------------------------------------------------
# Cost statistics from CostStore
# ---------------------------------------------------------------------------


@dataclass
class CostStats:
    """Cost aggregation for a digest window and trailing-24h."""

    watcher_cost_in_window: float = 0.0
    total_cost_in_window: float = 0.0
    watcher_cost_24h: float = 0.0
    total_cost_24h: float = 0.0


_ZERO_COST = CostStats()


async def cost_in_window(
    cost_store: CostStore | None,
    window_start_iso: str,
    window_end_iso: str,
) -> CostStats:
    """Return cost aggregation for the digest window and trailing-24h.

    Mirrors the SQL pattern from Harness._enforce_cost_ceilings (single query
    with conditional aggregation) issuing one query for the window and one for
    trailing-24h.  Uses cost_store._require_conn() exactly as
    _trailing_24h_cost_usd does.

    Fail-open: cost_store is None or any exception → CostStats with all zeros.
    """
    if cost_store is None:
        return CostStats()
    try:
        conn = cost_store._require_conn()  # type: ignore[attr-defined]

        # Window query: watcher cost + total cost inside [window_start, window_end]
        cur = await conn.execute(
            'SELECT '
            '  COALESCE(SUM(cost_usd), 0.0), '
            '  COALESCE(SUM(CASE WHEN role LIKE ? THEN cost_usd END), 0.0) '
            'FROM invocations '
            'WHERE completed_at BETWEEN ? AND ?',
            ('%watcher%', window_start_iso, window_end_iso),
        )
        row_win = await cur.fetchone()
        await cur.close()
        total_win = float(row_win[0]) if row_win and row_win[0] is not None else 0.0
        watcher_win = float(row_win[1]) if row_win and row_win[1] is not None else 0.0

        # Trailing-24h query
        cutoff_24h = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
        cur = await conn.execute(
            'SELECT '
            '  COALESCE(SUM(cost_usd), 0.0), '
            '  COALESCE(SUM(CASE WHEN role LIKE ? THEN cost_usd END), 0.0) '
            'FROM invocations '
            'WHERE completed_at >= ?',
            ('%watcher%', cutoff_24h),
        )
        row_24h = await cur.fetchone()
        await cur.close()
        total_24h = float(row_24h[0]) if row_24h and row_24h[0] is not None else 0.0
        watcher_24h = float(row_24h[1]) if row_24h and row_24h[1] is not None else 0.0

        return CostStats(
            watcher_cost_in_window=watcher_win,
            total_cost_in_window=total_win,
            watcher_cost_24h=watcher_24h,
            total_cost_24h=total_24h,
        )
    except Exception:
        logger.warning('cost_in_window: query failed (fail-open)', exc_info=True)
        return CostStats()


# ---------------------------------------------------------------------------
# Digest inputs and markdown rendering
# ---------------------------------------------------------------------------


@dataclass
class DigestInputs:
    """All data needed to render one digest entry."""

    window_start_iso: str
    window_end_iso: str
    escalation_stats: EscalationStats
    done_count: int
    cost_stats: CostStats
    parked_live: int
    parked_window_churn: int
    ewa_value: float
    ewa_threshold: float
    tripped: bool
    # Dict of flag_name → bool; only True flags are rendered in Anomalies
    anomaly_flags: dict[str, bool]
    # Free-text cluster descriptions (or empty list for "none detected")
    watcher_clusters: list[str]
    # Free-text proposal summaries (or empty list for "none queued")
    dry_run_proposals: list[str]


def render_digest_markdown(inputs: DigestInputs) -> str:
    """Render a DigestInputs as a markdown string.

    All sections are always present; empty lists render as 'none detected' /
    'none queued'.  Costs are formatted as $X.XX.  No I/O — pure string
    computation.
    """
    lines: list[str] = []

    # Header
    lines.append('# Digest')
    lines.append('')

    # Window
    lines.append('## Window')
    lines.append(f'- Start: {inputs.window_start_iso}')
    lines.append(f'- End:   {inputs.window_end_iso}')
    lines.append('')

    # Escalation outcomes
    lines.append('## Escalation outcomes')
    counts = inputs.escalation_stats.category_level_status_counts
    if counts:
        lines.append('| category | level | status | count |')
        lines.append('|----------|-------|--------|-------|')
        for (category, level, status), count in sorted(counts.items()):
            lines.append(f'| {category} | {level} | {status} | {count} |')
    else:
        lines.append('_none_')
    esc = inputs.escalation_stats
    lines.append(f'- Pending: {esc.pending_total}')
    lines.append(f'- Dedupe children: {esc.dedupe_children_total}')
    if esc.first_ts:
        lines.append(f'- First: {esc.first_ts}')
        lines.append(f'- Last:  {esc.last_ts}')
    lines.append('')

    # Tasks done
    lines.append('## Tasks done in window')
    lines.append(f'{inputs.done_count}')
    lines.append('')

    # Cost
    cs = inputs.cost_stats
    lines.append('## Cost')
    lines.append(f'- Watcher (window):  ${cs.watcher_cost_in_window:.2f}')
    lines.append(f'- Total   (window):  ${cs.total_cost_in_window:.2f}')
    lines.append(f'- Watcher (24h):     ${cs.watcher_cost_24h:.2f}')
    lines.append(f'- Total   (24h):     ${cs.total_cost_24h:.2f}')
    lines.append('')

    # Parked tasks
    lines.append('## Parked tasks')
    lines.append(f'- Live parked:    {inputs.parked_live}')
    lines.append(f'- Window churn:   {inputs.parked_window_churn}')
    lines.append('')

    # EWA
    tripped_marker = ' — **TRIPPED**' if inputs.tripped else ''
    lines.append('## EWA')
    lines.append(
        f'- Value / threshold: {inputs.ewa_value:.4f} / {inputs.ewa_threshold:.4f}'
        f'{tripped_marker}'
    )
    lines.append('')

    # Anomalies
    lines.append('## Anomalies')
    active = [name for name, val in sorted(inputs.anomaly_flags.items()) if val]
    if active:
        for name in active:
            lines.append(f'- {name}')
    else:
        lines.append('_none_')
    lines.append('')

    # Cross-escalation patterns
    lines.append('## Cross-escalation patterns')
    if inputs.watcher_clusters:
        for cluster in inputs.watcher_clusters:
            lines.append(f'- {cluster}')
    else:
        lines.append('none detected')
    lines.append('')

    # Dry-run proposals
    lines.append('## Dry-run proposals')
    if inputs.dry_run_proposals:
        for proposal in inputs.dry_run_proposals:
            lines.append(f'- {proposal}')
    else:
        lines.append('none queued')
    lines.append('')

    return '\n'.join(lines)
