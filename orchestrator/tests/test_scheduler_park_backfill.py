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
    durations exact rather than approximate.  It is a DEFAULT, not a pin: the
    two-holders-on-one-module case needs a limit of 2 to have two holders at
    all, and overriding it must not raise a duplicate-kwarg TypeError.
    """
    config = OrchestratorConfig(**{'max_per_module': 1, **config_overrides})
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


def _hold(
    scheduler: Scheduler, holder: str, modules: list[str], *, elapsed: float
) -> None:
    """*holder* really takes the lock on *modules*, having held for *elapsed*.

    Takes the REAL lock (so ``blocking_holders`` sees it) and opens the hold
    in the history at ``FIXED_DT - elapsed`` (so ``predicted_remaining`` sees
    it), which is the pair of facts production establishes together through
    ``_emit_lock_event``.  Call this BEFORE installing the park that waits on
    the module: a park blocks a non-owner's acquire, so the reverse order
    would leave the holder holding nothing.
    """
    assert scheduler.lock_table.try_acquire(holder, modules), (
        f'{holder} could not take {modules} — test setup is wrong, not the code'
    )
    scheduler._hold_history.observe_acquired(
        holder, modules, at=FIXED_DT.timestamp() - elapsed
    )


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


# ===========================================================================
# _provable_assembly_delay(p) — the gap a park actually has to lend
# ===========================================================================
#
# C7: min over p's still-blocked park modules of
#     max(0, predicted_hold(holder) - elapsed_hold)
#
# Three choices here all point the same way, and each test names which one it
# pins:
#   * MIN over modules, not max.  p cannot assemble until EVERY parked module
#     frees, so the max is p's TRUE wait — and lending the true wait means
#     lending a number no single observation proves.  The min is the provable
#     LOWER bound, and a lower bound is the only safe direction when the
#     number is spent admitting somebody else through.
#   * an empty blocked set is 0.0, not "infinite patience": p can assemble
#     right now, so there is no gap at all.
#   * a holder with no prediction contributes 0.0.  "Provable" is in the name;
#     unprovable must not read as generous.
#
# Every figure below is exact — the seeded medians are hand-chosen odd-length
# windows and the elapsed times are literals, so predicted_remaining is an
# integer-valued literal, never a tolerance.

def test_park_on_free_modules_lends_no_gap():
    """All of p's parked modules are free → 0.0 (p can assemble NOW)."""
    scheduler = _make_scheduler()
    scheduler.lock_table.install_parks('p', ['mod-a', 'mod-b'], 'medium')

    assert scheduler._provable_assembly_delay('p') == 0.0


def test_owner_with_no_parks_lends_no_gap():
    """An owner that never parked has no wait to lend → 0.0."""
    scheduler = _make_scheduler()

    assert scheduler._provable_assembly_delay('never-parked') == 0.0


def test_single_blocked_module_lends_that_holders_remaining():
    """One blocked module, one holder with 900s left → 900.0."""
    scheduler = _make_scheduler()
    # median of [900, 1000, 1100] is exactly 1000.0
    _feed_spans(scheduler, ['mod-a'], [900.0, 1000.0, 1100.0])
    _hold(scheduler, 'h', ['mod-a'], elapsed=100.0)
    scheduler.lock_table.install_parks('p', ['mod-a'], 'medium')

    assert scheduler.predicted_remaining('h') == pytest.approx(900.0)
    assert scheduler._provable_assembly_delay('p') == pytest.approx(900.0)


def test_two_blocked_modules_lend_the_minimum_not_the_maximum():
    """Remainders 900 and 300 → 300.0.

    p's TRUE wait is 900 (it needs both), so returning the max would be
    "correct" about p and wrong about what is provable.  The min is the lower
    bound, and this test is the one that catches a max/min flip — with equal
    remainders the two are indistinguishable.
    """
    scheduler = _make_scheduler()
    _feed_spans(scheduler, ['mod-a'], [900.0, 1000.0, 1100.0])   # median 1000
    _feed_spans(scheduler, ['mod-b'], [300.0, 400.0, 500.0])     # median 400
    _hold(scheduler, 'h-a', ['mod-a'], elapsed=100.0)            # 1000-100 = 900
    _hold(scheduler, 'h-b', ['mod-b'], elapsed=100.0)            # 400-100  = 300
    scheduler.lock_table.install_parks('p', ['mod-a', 'mod-b'], 'medium')

    assert scheduler.predicted_remaining('h-a') == pytest.approx(900.0)
    assert scheduler.predicted_remaining('h-b') == pytest.approx(300.0)
    assert scheduler._provable_assembly_delay('p') == pytest.approx(300.0)


def test_unpredictable_holder_drives_the_whole_gap_to_zero():
    """A holder below the sample floor contributes 0.0, so the min is 0.0.

    Not "skip the module we cannot predict" — skipping would let the ONE
    well-evidenced module certify the whole park, which is precisely the
    empty-history-certifies-structure failure PRD :459-461 forbids.
    """
    scheduler = _make_scheduler()
    _feed_spans(scheduler, ['mod-a'], [900.0, 1000.0, 1100.0])   # median 1000
    _feed_spans(scheduler, ['mod-b'], [50.0])                    # 1 < floor of 3
    _hold(scheduler, 'h-a', ['mod-a'], elapsed=100.0)
    _hold(scheduler, 'h-b', ['mod-b'], elapsed=10.0)
    scheduler.lock_table.install_parks('p', ['mod-a', 'mod-b'], 'medium')

    assert scheduler.predicted_remaining('h-a') == pytest.approx(900.0)
    assert scheduler.predicted_remaining('h-b') is None, (
        'setup guard: mod-b must be BELOW the floor, so the holder is '
        'genuinely unpredictable rather than merely quick'
    )
    assert scheduler._provable_assembly_delay('p') == 0.0


def test_overdue_holder_lends_zero_via_the_zero_path_not_the_none_path():
    """An overdue holder yields 0.0, and it arrives as 0.0, not as None.

    ζ keeps ``0.0`` ("predicted to have finished already, and hasn't") apart
    from ``None`` ("no prediction at all").  Both refuse admission here, but
    they are different live facts and η must not collapse them.
    """
    scheduler = _make_scheduler()
    _feed_spans(scheduler, ['mod-a'], [900.0, 1000.0, 1100.0])   # median 1000
    _hold(scheduler, 'h', ['mod-a'], elapsed=5000.0)             # 5x its median
    scheduler.lock_table.install_parks('p', ['mod-a'], 'medium')

    remaining = scheduler.predicted_remaining('h')
    assert remaining is not None, 'overdue is 0.0, NOT the absence of an answer'
    assert remaining == 0.0
    assert scheduler._provable_assembly_delay('p') == 0.0


def test_module_held_by_two_takes_the_soonest_holders_remaining():
    """Two holders on one module → the MIN of their remainders.

    The module regains headroom the moment the SOONEST holder releases, so
    the later holder's remainder is not what p is waiting on.  Needs
    ``max_per_module=2`` to have two holders at all — at 2 holders the module
    is still at its limit, so it is genuinely blocked.
    """
    scheduler = _make_scheduler(max_per_module=2)
    _feed_spans(scheduler, ['mod-a'], [900.0, 1000.0, 1100.0])   # median 1000
    _hold(scheduler, 'h-slow', ['mod-a'], elapsed=100.0)         # 900 left
    _hold(scheduler, 'h-soon', ['mod-a'], elapsed=700.0)         # 300 left
    scheduler.lock_table.install_parks('p', ['mod-a'], 'medium')

    assert scheduler.lock_table.blocking_holders(['mod-a'], exclude_task='p') == {
        'mod-a': ['h-slow', 'h-soon'],  # sorted by id, not by remaining
    }, 'setup guard: the module must actually be AT its limit of 2'
    assert scheduler._provable_assembly_delay('p') == pytest.approx(300.0)
