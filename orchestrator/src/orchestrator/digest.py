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

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.cost_store import CostStore

from escalation.models import Escalation
from escalation.queue import iter_all_escalation_paths

from orchestrator.workflow import WorkflowOutcome

logger = logging.getLogger(__name__)

# Canonical 'done' outcome value sourced directly from WorkflowOutcome so that
# any future rename is caught at import time rather than silently zeroing done counts.
_DONE_OUTCOME: str = WorkflowOutcome.DONE.value

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

    Caller contract: escalations_in_step must be > 0.  In normal flow
    _maybe_write_digest only calls this when escalations_in_step >= N (the
    digest gate guarantees this invariant).  A ValueError is raised so that
    any caller that violates the contract is loud rather than silently
    returning a stale value — keeping the unreachable path unreachable.

    done_in_step == 0 (with esc > 0) uses denominator 1 so a step with
    escalations and zero completions (the worst-case signal) pushes EWA up
    rather than crashing.

    No other exception handling — pure arithmetic.
    """
    if escalations_in_step == 0:
        raise ValueError(
            'update_ewa: escalations_in_step must be > 0; '
            'the digest gate guarantees this — passing 0 is a caller error'
        )
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

    Performance: applies two layers of prefiltering before full JSON parse:
      1. Directory-level: archive/YYYY-MM-DD/ dirs strictly before window_start
         date are skipped entirely (archive layout is date-partitioned).
      2. File-level: reads only the 'timestamp' field from raw JSON and skips
         the file if it is outside the window before constructing a full
         Escalation object.

    Timezone robustness: timestamp comparison uses datetime.fromisoformat()
    rather than lexicographic string comparison, so 'Z' and '+00:00' suffixes
    are handled correctly.  Falls back to lexicographic comparison when
    fromisoformat() cannot parse.

    Fail-open: on any per-file error, logs a warning and continues.
    """
    stats = EscalationStats()

    # Parse window bounds as aware datetimes once for the whole loop.
    try:
        window_start_dt = datetime.fromisoformat(window_start_iso)
        window_end_dt = datetime.fromisoformat(window_end_iso)
    except ValueError:
        logger.warning(
            'aggregate_escalations: cannot parse window bounds %r / %r — returning empty stats',
            window_start_iso, window_end_iso,
        )
        return stats

    # window_start date prefix for directory-level prefilter (YYYY-MM-DD).
    window_start_date = window_start_iso[:10]

    for path in iter_all_escalation_paths(escalations_dir):
        # (1) Directory-level prefilter: skip archive/YYYY-MM-DD/ dirs entirely
        # when the date partition is strictly before the window start date.
        # archive layout: <escalations_dir>/archive/YYYY-MM-DD/esc-*.json
        try:
            rel = path.relative_to(escalations_dir)
            rel_parts = rel.parts
            if (
                len(rel_parts) >= 3
                and rel_parts[0] == 'archive'
                and len(rel_parts[1]) == 10
                and rel_parts[1][4] == '-'
                and rel_parts[1][7] == '-'
                and rel_parts[1] < window_start_date
            ):
                continue
        except ValueError:
            pass  # path not relative to escalations_dir — skip dir check

        # (2) File-level prefilter: read raw JSON and parse only 'timestamp'
        # before constructing the full Escalation object.
        try:
            text = path.read_text()
        except Exception:
            logger.warning('aggregate_escalations: failed to read %s', path, exc_info=True)
            continue

        ts: str = ''
        ts_dt: datetime | None = None
        try:
            raw = json.loads(text)
            ts = raw.get('timestamp') or ''
        except Exception:
            pass  # malformed JSON — attempt full parse below, let from_json report it

        if ts:
            try:
                ts_dt = datetime.fromisoformat(ts)
                if ts_dt < window_start_dt or ts_dt > window_end_dt:
                    continue
            except ValueError:
                # fromisoformat failed (e.g. unusual suffix) — fall back to lexicographic
                if ts < window_start_iso or ts > window_end_iso:
                    continue
        else:
            continue  # no timestamp → skip

        # Full parse only for in-window files.
        try:
            esc = Escalation.from_json(text)
        except Exception:
            logger.warning('aggregate_escalations: failed to parse %s', path, exc_info=True)
            continue

        esc_ts = esc.timestamp or ts  # prefer parsed timestamp for stats

        # Update counts
        key = (esc.category, esc.level, esc.status)
        stats.category_level_status_counts[key] = (
            stats.category_level_status_counts.get(key, 0) + 1
        )

        # Track first / last timestamps in window (use raw ts for consistency)
        if stats.first_ts is None or esc_ts < stats.first_ts:
            stats.first_ts = esc_ts
        if stats.last_ts is None or esc_ts > stats.last_ts:
            stats.last_ts = esc_ts

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
    Missing DB returns 0 (DEBUG); other failures return 0 (WARNING).
    """
    try:
        if not Path(events_db).exists():
            logger.debug('count_done_in_window: DB not found (fail-open): %s', events_db)
            return 0
        # Use Path.resolve().as_uri() so the path is always absolute (as_uri()
        # requires an absolute path) and special characters are percent-encoded
        # before appending the ?mode=ro query string.  resolve() follows symlinks;
        # this is intentional — the resolved target is what SQLite will open and
        # callers that need symlink-identity preservation should pass an absolute
        # path directly.
        db_uri = Path(events_db).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True)
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
        # TOCTOU guard: if the file disappeared between the structural check
        # above and sqlite3.connect(), re-detect that as a missing-DB (DEBUG)
        # rather than an unexpected failure (WARNING).
        if not Path(events_db).exists():
            logger.debug('count_done_in_window: DB not found (fail-open): %s', events_db)
        else:
            logger.warning('count_done_in_window: failed (fail-open)', exc_info=True)
        return 0


# ---------------------------------------------------------------------------
# Merge-disposition counts from EventStore (task 2384 γ, mechanism M2 of
# plans/merge-skew-attribution-prd.md, boundary row 7).
#
# GROUPs merge_attempt events by their optional 'disposition' payload key
# (task 2381 α's MergeFailureDisposition, persisted by
# merge_queue._emit_merge_attempt) so operator stats can separate
# integration_skew from branch_bug/indeterminate/flakes instead of lumping
# every merge failure into one undifferentiated bucket.
# ---------------------------------------------------------------------------


def merge_disposition_counts(
    events_db: Path,
    window_start_iso: str,
    window_end_iso: str,
) -> dict[str, int]:
    """Count merge_attempt events inside the window, grouped by disposition.

    Rows with no ``'disposition'`` key (pre-α/β emitters, or non-orchestrator
    submit paths) are excluded — the map only covers actually-attributed
    merge failures.

    Uses sqlite3 directly (read-only).  Fail-open: any exception returns {}.
    Missing DB returns {} (DEBUG); other failures return {} (WARNING).
    """
    try:
        if not Path(events_db).exists():
            logger.debug('merge_disposition_counts: DB not found (fail-open): %s', events_db)
            return {}
        # See count_done_in_window above for the resolve().as_uri()+?mode=ro rationale.
        db_uri = Path(events_db).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True)
        try:
            rows = conn.execute(
                "SELECT json_extract(data, '$.disposition') AS disp, COUNT(*) FROM events "
                "WHERE event_type = 'merge_attempt' "
                "  AND timestamp BETWEEN ? AND ? "
                "  AND json_extract(data, '$.disposition') IS NOT NULL "
                "GROUP BY disp",
                (window_start_iso, window_end_iso),
            ).fetchall()
            return {disp: int(count) for disp, count in rows}
        finally:
            conn.close()
    except Exception:
        # TOCTOU guard: if the file disappeared between the structural check
        # above and sqlite3.connect(), re-detect that as a missing-DB (DEBUG)
        # rather than an unexpected failure (WARNING).
        if not Path(events_db).exists():
            logger.debug('merge_disposition_counts: DB not found (fail-open): %s', events_db)
        else:
            logger.warning('merge_disposition_counts: failed (fail-open)', exc_info=True)
        return {}


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


async def cost_in_window(
    cost_store: CostStore | None,
    window_start_iso: str,
    window_end_iso: str,
) -> CostStats:
    """Return cost aggregation for the digest window and trailing-24h.

    Delegates to ``CostStore.cost_totals_in_window`` — once for the explicit
    window bounds and once for the trailing-24h window ending at now.

    The trailing-24h upper bound is ``now_iso`` (a captured snapshot): any
    invocations written after that snapshot are silently excluded, which is
    acceptable for informational cost reporting.

    Fail-open: cost_store is None or any exception → CostStats with all zeros.
    """
    if cost_store is None:
        return CostStats()
    try:
        now = datetime.now(UTC)
        now_iso = now.isoformat()
        cutoff_24h_iso = (now - timedelta(hours=24)).isoformat()

        total_win, watcher_win = await cost_store.cost_totals_in_window(
            window_start_iso, window_end_iso
        )
        total_24h, watcher_24h = await cost_store.cost_totals_in_window(
            cutoff_24h_iso, now_iso
        )

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


# ---------------------------------------------------------------------------
# write_digest_entry — atomic file write, never raises
# ---------------------------------------------------------------------------


@dataclass
class DigestResult:
    """Return value from write_digest_entry.

    path is None when the write failed (fail-open — log warning, no raise).
    """

    path: Path | None
    tripped: bool
    ewa_value: float


def write_digest_entry(digest_dir: Path, inputs: DigestInputs) -> DigestResult:
    """Write a digest markdown file to digest_dir and return a DigestResult.

    - mkdir(parents=True, exist_ok=True) so the dir is auto-created.
    - File named digest-<YYYYMMDDTHHmmSS_ffffff>.md — microsecond suffix
      prevents collisions on rapid back-to-back calls.
    - Written atomically via a tmp file + rename so a partial write is never
      left behind.
    - NEVER raises: any exception logs a warning and returns
      DigestResult(path=None, tripped=inputs.tripped, ewa_value=inputs.ewa_value).

    Task 1327 AFK hardening.
    """
    try:
        digest_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime('%Y%m%dT%H%M%S_%f')
        filename = f'digest-{ts}.md'
        target = digest_dir / filename
        # Atomic write: tmp → rename
        tmp = digest_dir / f'.tmp-{filename}'
        markdown = render_digest_markdown(inputs)
        tmp.write_text(markdown, encoding='utf-8')
        tmp.rename(target)
        return DigestResult(path=target, tripped=inputs.tripped, ewa_value=inputs.ewa_value)
    except Exception:
        logger.warning('write_digest_entry: failed to write digest (fail-open)', exc_info=True)
        return DigestResult(path=None, tripped=inputs.tripped, ewa_value=inputs.ewa_value)
