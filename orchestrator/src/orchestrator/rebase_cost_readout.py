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
    - ``n``: int — number of rows in this cohort
    - ``distance_p50``: float — median ``distance_commits``
    - ``verify_secs_p50``: float — median ``next_verify_wall_secs``

    Returns ``{}`` for an empty input.
    """
    if not rows:
        return {}

    cohort_distances: dict[str, list[float]] = {}
    cohort_verify_secs: dict[str, list[float]] = {}

    for row in rows:
        # Support both fetch_events_by_type rows (nested 'data') and flat dicts.
        payload: dict = row.get('data', row)  # type: ignore[assignment]
        cohort = payload.get('cohort', 'unknown')
        distance = float(payload.get('distance_commits', 0))
        verify_secs = float(payload.get('next_verify_wall_secs', 0.0))

        cohort_distances.setdefault(cohort, []).append(distance)
        cohort_verify_secs.setdefault(cohort, []).append(verify_secs)

    result: dict = {}
    for cohort, distances in cohort_distances.items():
        verify_list = cohort_verify_secs[cohort]
        result[cohort] = {
            'n': len(distances),
            'distance_p50': statistics.median(distances),
            'verify_secs_p50': statistics.median(verify_list),
        }

    return result
