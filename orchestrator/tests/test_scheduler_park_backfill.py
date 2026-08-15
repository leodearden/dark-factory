"""EASY-backfill admission through parks.

Task 3823 / plans/scheduler-dispatch-scoring-and-lock-layer-prd.md task η
(contract C7).

C7's predicate: a candidate ``c`` blocked ONLY by another task ``p``'s park
may be admitted through it when

    predicted_hold(c) * backfill_safety_factor <= provable_assembly_delay(p)

and ``p``'s park is not older than ``backfill_max_park_age_secs``.  The
acquisition itself still goes through ``try_acquire`` under the lock — only
the PARK gate is bypassed, never the held-lock limit gate (INV-3), so an
admitted-but-actually-contended candidate simply fails to acquire and does
not dispatch.

Everything numeric here is computed from a synthetic hold history with a
hand-known median, so every predicted / bound / realized figure is an exact
literal.  No fitted thresholds, no tolerances.

Evidence base:
plans/evidence/scheduler-scoring-2026-08-06/PARKING_MODEL_REPORT.md
  :116-126  module hold-history median is the ONLY predictor with positive R²
  :248-258  the P4 recommendation, 7-9% overstay at safety ×2.5, and the named
            casualty of having no park-age cutoff
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import Scheduler

FIXED_DT = datetime(2026, 8, 1, 0, 0, 0, tzinfo=UTC)


def _make_scheduler(**config_overrides) -> Scheduler:
    """A bare Scheduler on a frozen wall clock.

    ``max_per_module=1`` matches the shape the parking model measured (one
    holder per module), and the frozen clock makes park age and realized hold
    durations exact rather than approximate.
    """
    config = OrchestratorConfig(max_per_module=1, **config_overrides)
    return Scheduler(config, wall_time_source=lambda: FIXED_DT)


def _task(task_id: str, files: list[str]) -> dict:
    return {
        'id': task_id,
        'title': f'Task {task_id}',
        'status': 'pending',
        'dependencies': [],
        'metadata': {'files': files},
    }


def _feed_spans(scheduler: Scheduler, modules: list[str], durations: list[float]) -> None:
    """Record complete hold spans on *modules* through the LIVE feed.

    Deliberately drives ``observe_acquired``/``observe_released`` rather than
    ``record()``: that is the path production takes via ``_emit_lock_event``,
    so these tests exercise the same seam the scheduler does.  Each span uses
    a distinct synthetic holder id and a disjoint time interval, so no two
    overlap and the recorded duration is exactly the value asked for.
    """
    at = 0.0
    for index, duration in enumerate(durations):
        holder = f'seed-{index}'
        scheduler._hold_history.observe_acquired(holder, modules, at=at)
        scheduler._hold_history.observe_released(holder, modules, at=at + duration)
        at += duration + 1.0


# ===========================================================================
# backfill_min_samples is the ONE source of truth for the refuse floor (INV-5)
# ===========================================================================
#
# hold_history.py:591-593 reserves this leaf for task η by name: HoldHistory
# carries a module DEFAULT so it can stand alone without the config object,
# and the Scheduler's construction site is the seam where the configured
# value becomes live.  Until that seam exists, editing backfill_min_samples
# is inert — an operator tightening the floor would see no change in
# behaviour, which is exactly the silent degradation the project's
# loud-over-silent norm forbids.
#
# Asserted BEHAVIOURALLY (feed N-1 spans -> None, feed the Nth -> the exact
# median) rather than by reading HoldHistory._min_samples: the private
# attribute could be set correctly while the prediction path read something
# else, and it is the prediction that gates admission.


def test_configured_floor_above_the_module_default_is_honoured():
    """backfill_min_samples=5: four spans still refuse, the fifth predicts."""
    scheduler = _make_scheduler(backfill_min_samples=5)
    task = _task('1', ['orchestrator/src/orchestrator/scheduler.py'])
    modules = scheduler._get_modules(task)

    _feed_spans(scheduler, modules, [10.0, 20.0, 30.0, 40.0])
    assert scheduler.predicted_hold(task) is None, (
        'four samples is below the configured floor of five — the predictor '
        'must refuse, not answer'
    )

    _feed_spans(scheduler, modules, [10.0, 20.0, 30.0, 40.0, 50.0])
    # Window default is 10, so all nine recorded samples are retained:
    # [10,20,30,40] + [10,20,30,40,50] sorted -> median (5th of 9) = 30.0.
    assert scheduler.predicted_hold(task) == pytest.approx(30.0)


def test_default_floor_is_three():
    """With no override the floor is 3: two spans refuse, three predict."""
    scheduler = _make_scheduler()
    task = _task('1', ['orchestrator/src/orchestrator/scheduler.py'])
    modules = scheduler._get_modules(task)

    _feed_spans(scheduler, modules, [100.0, 200.0])
    assert scheduler.predicted_hold(task) is None, (
        'two samples is below the default floor of three'
    )

    scheduler2 = _make_scheduler()
    _feed_spans(scheduler2, modules, [100.0, 200.0, 300.0])
    assert scheduler2.predicted_hold(task) == pytest.approx(200.0)


def test_configured_floor_below_the_module_default_is_honoured():
    """backfill_min_samples=1: a single span predicts.

    The mirror image of the raise-the-floor case, and the one that catches a
    hard-coded ``HoldHistory()`` — with the ctor default of 3 still winning,
    one span would answer None and the config value would be silently inert.
    """
    scheduler = _make_scheduler(backfill_min_samples=1)
    task = _task('1', ['orchestrator/src/orchestrator/scheduler.py'])
    modules = scheduler._get_modules(task)

    _feed_spans(scheduler, modules, [42.0])

    assert scheduler.predicted_hold(task) == pytest.approx(42.0)
