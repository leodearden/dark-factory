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
