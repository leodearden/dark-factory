"""Readout helpers for the rebase-distance → verify-cost telemetry (task 1802).

Two public functions:

- ``load_rebase_verify_cost_rows(event_store)`` — delegate to
  ``EventStore.fetch_events_by_type`` to retrieve all recorded
  ``rebase_verify_cost`` events for the current run.

- ``summarize_rebase_verify_cost(rows)`` — pure function that groups rows
  by cohort and computes per-cohort counts and median distance/verify-secs,
  so the distance→cost distribution readout is reproducible from collected
  samples.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.event_store import EventStore


def load_rebase_verify_cost_rows(event_store: EventStore) -> list[dict]:
    """Return all ``rebase_verify_cost`` events for the current run.

    Delegates to :meth:`EventStore.fetch_events_by_type`; each element is a
    row dict whose ``data`` key is already parsed to a ``dict``.
    """
    return event_store.fetch_events_by_type('rebase_verify_cost')


def summarize_rebase_verify_cost(rows: list[dict]) -> dict:
    """Summarise rebase-verify-cost rows grouped by cohort.

    Accepts rows as returned by :func:`load_rebase_verify_cost_rows` (each
    row has a nested ``data`` dict) **or** flat dicts (no ``data`` wrapper)
    — the data fields are read from ``row['data']`` when present, otherwise
    from the row itself.

    Returns a dict keyed by cohort name, each with:
    - ``n``: int — number of rows in this cohort (all rows, including those
      with the -1 sentinel distance returned by ``get_rebase_distance`` on
      git error).
    - ``distance_p50``: float | None — median ``distance_commits`` computed
      only over rows whose distance is ≥ 0 (i.e. valid measurements).
      ``None`` when every row in the cohort carries the -1 sentinel (all
      measurements failed).  The 'unknown' cohort will always have
      ``distance_p50=None`` because ``classify_rebase_cohort`` only assigns
      that label when ``distance < 0``.
    - ``verify_secs_p50``: float — median ``next_verify_wall_secs``.

    Returns ``{}`` for an empty input.
    """
    if not rows:
        return {}

    cohort_distances: dict[str, list[float]] = {}
    cohort_verify_secs: dict[str, list[float]] = {}
    cohort_counts: dict[str, int] = {}

    for row in rows:
        # Support both fetch_events_by_type rows (nested 'data') and flat dicts.
        payload: dict = row.get('data', row)  # type: ignore[assignment]
        cohort = payload.get('cohort', 'unknown')
        distance = float(payload.get('distance_commits', 0))
        verify_secs = float(payload.get('next_verify_wall_secs', 0.0))

        cohort_counts[cohort] = cohort_counts.get(cohort, 0) + 1
        # Exclude -1 sentinel (failed measurement) from the distance distribution
        # so it does not corrupt the median for callers treating distance_p50
        # numerically.  Rows are still counted in 'n'.
        if distance >= 0:
            cohort_distances.setdefault(cohort, []).append(distance)
        cohort_verify_secs.setdefault(cohort, []).append(verify_secs)

    result: dict = {}
    for cohort, count in cohort_counts.items():
        valid_distances = cohort_distances.get(cohort, [])
        verify_list = cohort_verify_secs.get(cohort, [])
        result[cohort] = {
            'n': count,
            'distance_p50': statistics.median(valid_distances) if valid_distances else None,
            'verify_secs_p50': statistics.median(verify_list) if verify_list else 0.0,
        }

    return result
