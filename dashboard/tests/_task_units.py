"""Canned ``collect_tasks_with_counts`` results for tests that fake the collector.

The real collector hands back one ``TaskSnapshot`` per configured root, keyed
by project label, and each measured unit carries that root's SHAPED rows.
:func:`units_by_label` builds that dict from canned rows, so a fake carries
its rows where the real collector does: on the units.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from types import MappingProxyType

from dashboard.data.census import build_census
from dashboard.data.datum import Datum, DatumState
from dashboard.data.task_snapshot import (
    FRESHNESS_BOUND_SECONDS,
    SnapshotFailure,
    TaskSnapshot,
    unmeasured_snapshot,
)

_AS_OF = datetime(2026, 1, 1, tzinfo=UTC)


def measured_unit(rows: Iterable[dict], *, as_of: datetime = _AS_OF) -> TaskSnapshot:
    """A wholly measured unit carrying *rows*, which ``classify`` routes to OK."""
    return TaskSnapshot(
        census=Datum(
            build_census({}), as_of, DatumState.FRESH, None, FRESHNESS_BOUND_SECONDS,
        ),
        rows=Datum(list(rows), as_of, DatumState.FRESH, None, FRESHNESS_BOUND_SECONDS),
        in_progress_live=0, in_progress_stranded=0, skew_seconds=0,
        status_map=MappingProxyType({}), failure=SnapshotFailure.NONE,
    )


def units_by_label(
    active: Iterable[dict] = (),
    *,
    offline: Iterable[str] = (),
    degraded: Iterable[str] = (),
) -> dict[str, TaskSnapshot]:
    """``{label: unit}``: *active* rows grouped by ``row['project']``, plus unmeasured roots.

    Each label in *offline* gets an ``UNREACHABLE`` unit and each in
    *degraded* a ``BUDGET`` one. A label cannot both carry rows and be
    unmeasured, so the three label sets must be disjoint.
    """
    rows_by_label: dict[str, list[dict]] = {}
    for row in active:
        rows_by_label.setdefault(row['project'], []).append(row)
    offline, degraded = list(offline), list(degraded)
    overlaps = {
        'active & offline': set(rows_by_label) & set(offline),
        'active & degraded': set(rows_by_label) & set(degraded),
        'offline & degraded': set(offline) & set(degraded),
    }
    clashing = {pair: sorted(labels) for pair, labels in overlaps.items() if labels}
    assert not clashing, f'a label must name one kind of unit; overlapping: {clashing}'

    def unmeasured(label: str, failure: SnapshotFailure) -> TaskSnapshot:
        return unmeasured_snapshot(label, now=_AS_OF, reason='test', failure=failure)

    return {
        **{label: measured_unit(rows) for label, rows in rows_by_label.items()},
        **{label: unmeasured(label, SnapshotFailure.UNREACHABLE) for label in offline},
        **{label: unmeasured(label, SnapshotFailure.BUDGET) for label in degraded},
    }
