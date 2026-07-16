"""Escalation lifecycle analytics — archive aggregator for the dashboard.

Backend data layer for plans/escalation-lifecycle-dashboard-prd.md Seam 2
(task gamma / 2658). Produces the payload for
``GET /api/v2/dashboard/escalation-analytics``: per-project origin/lifespan/
workflow aggregates over the escalation archive, plus regime markers and a
``parse_failures`` count (INV-4 — every skipped record is loud AND counted).

This module is a PURE-SYNC core: :func:`build_escalation_analytics` does a
per-request archive walk with no ``asyncio``. The route (``dashboard.app``)
wraps the call in ``asyncio.to_thread`` behind a short TTL cache so a cold
~10k-record walk never blocks the event loop. Tests exercise this module's
functions directly.

Clock discipline: the only permitted clock read is via
:func:`dashboard.data.utils.resolve_now`, threaded through once from
:func:`build_escalation_analytics` — see ``test_clock_discipline.py``.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Module-relative default: dashboard/regime-markers.yaml (the top-level
# `dashboard/` package dir — three levels up from
# dashboard/src/dashboard/data/escalation_analytics.py).
_DEFAULT_REGIME_MARKERS_PATH = Path(__file__).resolve().parents[3] / 'regime-markers.yaml'


def load_regime_markers(path: Path | None = None) -> tuple[list[dict], int]:
    """Load hand-curated regime markers from a committed YAML file.

    Args:
        path: Path to the YAML file. Defaults to the committed
            ``dashboard/regime-markers.yaml``.

    Returns:
        ``(markers, parse_failures_delta)``. Never raises:

        - Missing file -> ``([], 0)`` (absent is not a failure).
        - Unparseable YAML or a non-list top level -> ``([], 1)`` + WARNING
          (row 9 substrate: the endpoint must never 500 on a broken markers
          file — it degrades to an empty list and counts the failure).

        Each returned marker is normalized to ``{date, label, tasks}``
        (``tasks`` defaults to ``[]``). ``date`` is coerced to ``str`` when
        YAML parses an unquoted ``YYYY-MM-DD`` scalar as a ``datetime.date``
        object — otherwise the payload would fail JSON serialization at the
        route layer.
    """
    p = path if path is not None else _DEFAULT_REGIME_MARKERS_PATH
    if not p.exists():
        return [], 0

    try:
        with p.open() as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        logger.warning('load_regime_markers: failed to read/parse %s: %s', p, exc)
        return [], 1

    if not isinstance(data, list):
        logger.warning(
            'load_regime_markers: %s top level is not a list (got %s)',
            p, type(data).__name__,
        )
        return [], 1

    markers: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            logger.warning('load_regime_markers: skipping non-mapping entry in %s: %r', p, item)
            continue
        raw_date = item.get('date')
        markers.append({
            'date': raw_date.isoformat() if isinstance(raw_date, date) else raw_date,
            'label': item.get('label'),
            'tasks': item.get('tasks') or [],
        })
    return markers, 0


# ---------------------------------------------------------------------------
# runs.db done-per-day (esc_per_done_daily substrate)
# ---------------------------------------------------------------------------


def _done_by_day(runs_db: Path) -> dict[str, int]:
    """Return ``{date: count}`` of ``outcome='done'`` task_results rows.

    Bucketed by ``date(completed_at)``. Sync, read-only (``mode=ro`` URI),
    fail-open — mirrors ``orchestrator.digest._query_events_ro``'s discipline
    applied to ``runs.db`` instead of an events DB: a missing DB logs at
    DEBUG and returns ``{}``; any other failure logs at WARNING and returns
    ``{}``. Never raises.
    """
    runs_db = Path(runs_db)
    try:
        if not runs_db.exists():
            logger.debug('_done_by_day: DB not found (fail-open): %s', runs_db)
            return {}
        db_uri = runs_db.resolve().as_uri() + '?mode=ro'
        conn = sqlite3.connect(db_uri, uri=True)
        try:
            rows = conn.execute(
                "SELECT date(completed_at), COUNT(*) FROM task_results "
                "WHERE outcome = 'done' AND completed_at IS NOT NULL AND completed_at != '' "
                "GROUP BY date(completed_at)"
            ).fetchall()
            return {row[0]: row[1] for row in rows if row[0] is not None}
        finally:
            conn.close()
    except Exception:
        # TOCTOU guard: re-detect a since-vanished DB as missing (DEBUG)
        # rather than an unexpected failure (WARNING).
        if not runs_db.exists():
            logger.debug('_done_by_day: DB not found (fail-open): %s', runs_db)
            return {}
        logger.warning('_done_by_day: query failed for %s', runs_db, exc_info=True)
        return {}
