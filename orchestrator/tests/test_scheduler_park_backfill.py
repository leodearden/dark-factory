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

import ast
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

import pytest
from _recording_event_store import _RecordingEventStore

from orchestrator import scheduler as scheduler_module
from orchestrator.config import OrchestratorConfig, apply_reload
from orchestrator.scheduler import Scheduler, _BackfillGrant

if TYPE_CHECKING:  # the fake below is duck-typed; the real class is only a type
    from escalation.queue import EscalationQueue

FIXED_DT = datetime(2026, 8, 1, 0, 0, 0, tzinfo=UTC)


class _Clock:
    """A SETTABLE wall clock, shared by the Scheduler and its lock table.

    ``Scheduler`` passes its ``_wall_now`` down to the ``ModuleLockTable``, so
    one instance drives both park install stamps and hold elapsed times —
    which is exactly the coupling production has, and the reason park age can
    be driven without poking ``_park_install_at``.

    Tests set ``.now`` directly rather than only advancing it: several install
    a park in the PAST and then return to ``FIXED_DT``, so the park has a
    chosen age while every other timestamp stays at the same reference.
    """

    def __init__(self, now: datetime = FIXED_DT) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _make_scheduler(
    *, clock: _Clock | None = None, event_store=None, **config_overrides
) -> Scheduler:
    """A bare Scheduler on a frozen (or caller-driven) wall clock.

    ``max_per_module=1`` matches the shape the parking model measured (one
    holder per module), and the frozen clock makes park age and realized hold
    durations exact rather than approximate.  It is a DEFAULT, not a pin: the
    two-holders-on-one-module case needs a limit of 2 to have two holders at
    all, and overriding it must not raise a duplicate-kwarg TypeError.
    """
    config = OrchestratorConfig(**{'max_per_module': 1, **config_overrides})
    return Scheduler(
        config,
        wall_time_source=clock or (lambda: FIXED_DT),
        event_store=event_store,
    )


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
        holder, modules, at=scheduler._wall_now().timestamp() - elapsed
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


# --- ... and it is GREEN-TIER, so a reload has to actually move it ---------
#
# All four backfill leaves are declared hot-reloadable (config.py:5043-5046,
# pinned by test_config.py::test_backfill_leaves_are_green_tier_reloadable) so
# production overstay data can settle PRD Open Q3 without a restart.  Three of
# them are read through ``self.config.X`` at call time and genuinely retune
# live.  The floor is the odd one out: its only consumer is the predictor,
# built ONCE in ``Scheduler.__init__``, so a value captured there would be
# frozen for the process era — ``reload_config`` would report the leaf under
# ``applied`` with ``reloaded: true`` and the predictor would keep certifying
# from the old floor until restart.  That is a silent fail-soft on a config
# contract, and it is what these two tests exist to forbid.
#
# Driven through the REAL ``apply_reload`` (the same call the reload MCP tool
# makes) rather than a hand-written ``object.__setattr__``, so the tier
# declaration and the observable behaviour are tied together end to end.


def _reload_floor(scheduler: Scheduler, floor: int) -> dict:
    """Hot-apply a new ``backfill_min_samples`` to a LIVE scheduler's config."""
    result = apply_reload(
        scheduler.config,
        OrchestratorConfig(max_per_module=1, backfill_min_samples=floor),
    )
    assert result['reloaded'] is True, result
    assert 'backfill_min_samples' in result['applied'], (
        f'the leaf is declared green-tier but was not applied: {result}'
    )
    return result


def test_hot_reloading_the_floor_down_makes_the_predictor_answer():
    """10 -> 3, same Scheduler object, no reconstruction."""
    scheduler = _make_scheduler(backfill_min_samples=10)
    task = _task('1', ['orchestrator/src/orchestrator/scheduler.py'])
    modules = scheduler._get_modules(task)

    _feed_spans(scheduler, modules, [100.0, 200.0, 300.0])
    assert scheduler.predicted_hold(task) is None, (
        'three samples is below the configured floor of ten'
    )

    _reload_floor(scheduler, 3)

    assert scheduler.predicted_hold(task) == pytest.approx(200.0), (
        'the operator loosened a green-tier leaf and reload_config reported it '
        'applied — the predictor must honour it without a restart'
    )


def test_hot_reloading_the_floor_up_makes_the_predictor_refuse_again():
    """3 -> 10, the tighten direction: the one that matters for safety.

    An operator raising the floor is withdrawing certification authority from
    thin histories.  If that edit is inert, backfills keep being admitted on
    evidence the operator has just declared insufficient.
    """
    scheduler = _make_scheduler(backfill_min_samples=3)
    task = _task('1', ['orchestrator/src/orchestrator/scheduler.py'])
    modules = scheduler._get_modules(task)

    _feed_spans(scheduler, modules, [100.0, 200.0, 300.0])
    assert scheduler.predicted_hold(task) == pytest.approx(200.0)

    _reload_floor(scheduler, 10)

    assert scheduler.predicted_hold(task) is None, (
        'the floor was raised past the available evidence — the predictor must '
        'stop answering, not keep certifying from the captured old floor'
    )


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


# ===========================================================================
# _backfill_admission — C7's predicate as a truth table
# ===========================================================================
#
# A PURE DECISION: returns a grant on admit, None on refuse, and mutates
# nothing.  The acquisition it enables still goes through try_acquire under
# the lock (INV-3), so a wrong "admit" costs a failed acquire, never a
# double-held module.
#
# The standard shape every case below perturbs exactly one clause of:
#
#     h  holds  GAP_MODULE           (predicted median 1000s, elapsed 100s
#                                     -> 900s provably left)
#     p  parked on [c's module, GAP_MODULE], installed 60s ago
#     c  wants its own module, blocked ONLY by p's park, module median 100s
#
#     bound = 100 * 2.5 = 250  <=  gap = 900   -> ADMIT
#
# Every figure is exact: odd-length synthetic windows give integral medians,
# and the elapsed times are literals.

CANDIDATE_FILES = ['pkg_c/src/pkg_c/thing.py']
GAP_MODULES = ['gapmod/src/gapmod/held.py']
CANDIDATE_SPANS = (50.0, 100.0, 150.0)      # median 100.0
GAP_SPANS = (900.0, 1000.0, 1100.0)         # median 1000.0


def _park_blocked_candidate(
    scheduler: Scheduler,
    clock: _Clock,
    *,
    candidate_spans: tuple[float, ...] = CANDIDATE_SPANS,
    gap_spans: tuple[float, ...] = GAP_SPANS,
    gap_elapsed: float | None = 100.0,
    park_age: float = 60.0,
    owner: str = 'p',
) -> tuple[dict, list[str]]:
    """Build the standard C7 shape; return the candidate ``(task, modules)``.

    ``gap_elapsed=None`` means NOBODY holds the gap module, so *owner* can
    assemble right now and has no gap to lend.

    Order matters and is production's order: the holder takes its lock BEFORE
    any park exists (a park blocks a foreign acquire), and the park is stamped
    by winding the shared clock back ``park_age`` seconds and forward again,
    so the age comes from the real ``install_parks`` stamping seam rather than
    from poking ``_park_install_at``.
    """
    task = _task('c', CANDIDATE_FILES)
    modules = scheduler._get_modules(task)

    if gap_elapsed is not None:
        assert scheduler.lock_table.try_acquire('h', GAP_MODULES), (
            'setup: the gap holder must take its lock before any park exists'
        )

    clock.now = FIXED_DT - timedelta(seconds=park_age)
    scheduler.lock_table.install_parks(owner, modules + GAP_MODULES, 'medium')
    clock.now = FIXED_DT

    if candidate_spans:
        _feed_spans(scheduler, modules, list(candidate_spans))
    if gap_spans:
        _feed_spans(scheduler, GAP_MODULES, list(gap_spans))
    if gap_elapsed is not None:
        scheduler._hold_history.observe_acquired(
            'h', GAP_MODULES, at=FIXED_DT.timestamp() - gap_elapsed
        )
    return task, modules


# --- Admits ---------------------------------------------------------------


def test_ordinary_admit_returns_a_grant_carrying_every_admission_fact():
    """The happy path, and the full grant payload (INV-2).

    predicted AND the bound it was compared against are both on the record,
    so the later overstay settlement can say WHY a hold was a breach without
    anyone reconstructing the arithmetic from log lines.
    """
    clock = _Clock()
    scheduler = _make_scheduler(clock=clock)
    task, modules = _park_blocked_candidate(scheduler, clock)
    parks_before = scheduler.lock_table.snapshot_parks()

    grant = scheduler._backfill_admission('c', task, modules)

    assert grant is not None
    assert grant.task_id == 'c'
    assert grant.modules == tuple(modules)
    assert grant.park_owners == ('p',)
    assert grant.predicted_hold == pytest.approx(100.0)
    assert grant.safety_factor == pytest.approx(2.5)
    assert grant.admission_bound == pytest.approx(250.0)
    assert grant.provable_assembly_delay == pytest.approx(900.0)
    assert grant.granted_at == 0.0, (
        'granted_at is stamped at DISPATCH, not at admission: admission is a '
        'pure decision that try_acquire may still discard'
    )
    # Pure decision — no state moved.
    assert scheduler.lock_table.snapshot_parks() == parks_before
    # snapshot_holders is {module: task_id}, so the candidate is looked for
    # among the VALUES.
    assert 'c' not in scheduler.lock_table.snapshot_holders().values(), (
        'admission decides; try_acquire is what actually takes the lock'
    )


def test_bound_exactly_equal_to_the_gap_admits():
    """C7's operator is ``<=``: meeting the gap exactly is an admit.

    An off-by-one to ``<`` here would be invisible in production — it would
    just look like slightly fewer backfills — so the equality case gets its
    own test rather than riding along with the ordinary admit.
    """
    clock = _Clock()
    scheduler = _make_scheduler(clock=clock)
    # 1000 - 750 = 250 left, and the bound is 100 * 2.5 = 250.
    task, modules = _park_blocked_candidate(scheduler, clock, gap_elapsed=750.0)

    grant = scheduler._backfill_admission('c', task, modules)

    assert grant is not None
    assert grant.admission_bound == pytest.approx(grant.provable_assembly_delay)


def test_park_exactly_at_the_age_cutoff_admits():
    """``age > cutoff`` refuses, so age == cutoff still admits."""
    clock = _Clock()
    scheduler = _make_scheduler(clock=clock)
    task, modules = _park_blocked_candidate(scheduler, clock, park_age=3600.0)

    assert scheduler._backfill_admission('c', task, modules) is not None


def test_the_configured_safety_factor_is_read_not_inlined():
    """The same scenario admits at 2.5 and refuses at 4.0.

    PRD Open Q3 ships 2.5 and lets production overstay data settle it, which
    only works if the leaf is actually consulted.  A constant inlined at the
    comparison would pass every other test in this file.
    """
    gap_elapsed = 700.0  # 1000 - 700 = 300 left

    clock_lenient = _Clock()
    lenient = _make_scheduler(clock=clock_lenient, backfill_safety_factor=2.5)
    task, modules = _park_blocked_candidate(lenient, clock_lenient, gap_elapsed=gap_elapsed)
    grant = lenient._backfill_admission('c', task, modules)
    assert grant is not None, '100 * 2.5 = 250 fits in a 300s gap'
    assert grant.admission_bound == pytest.approx(250.0)

    clock_strict = _Clock()
    strict = _make_scheduler(clock=clock_strict, backfill_safety_factor=4.0)
    task, modules = _park_blocked_candidate(strict, clock_strict, gap_elapsed=gap_elapsed)
    assert strict._backfill_admission('c', task, modules) is None, (
        '100 * 4.0 = 400 does NOT fit in the same 300s gap'
    )


# --- Refusals: one named test per clause ----------------------------------


def test_refuses_a_candidate_that_is_not_park_blocked_at_all():
    """Nothing to backfill through → None; the ordinary outcome stands."""
    clock = _Clock()
    scheduler = _make_scheduler(clock=clock)
    task = _task('c', CANDIDATE_FILES)
    modules = scheduler._get_modules(task)
    _feed_spans(scheduler, modules, list(CANDIDATE_SPANS))

    assert scheduler._backfill_admission('c', task, modules) is None


def test_refuses_a_candidate_blocked_by_a_real_holder_rather_than_a_park():
    """A HOLDER is not a park.

    Backfill borrows an idle gap in front of a waiting task; it has no answer
    for a module that is genuinely occupied, and PRD section 7 rejects
    preempting a held lock outright.
    """
    clock = _Clock()
    scheduler = _make_scheduler(clock=clock)
    task = _task('c', CANDIDATE_FILES)
    modules = scheduler._get_modules(task)
    _feed_spans(scheduler, modules, list(CANDIDATE_SPANS))
    _hold(scheduler, 'occupier', modules, elapsed=10.0)

    assert not scheduler.lock_table.try_acquire('c', modules), (
        'setup guard: c must really be blocked, just not by a park'
    )
    assert scheduler._backfill_admission('c', task, modules) is None


def test_the_kill_switch_refuses_an_otherwise_perfect_admit():
    """backfill_enabled=False → None even when every other clause passes."""
    clock = _Clock()
    scheduler = _make_scheduler(clock=clock, backfill_enabled=False)
    task, modules = _park_blocked_candidate(scheduler, clock)

    assert scheduler._backfill_admission('c', task, modules) is None


def test_an_empty_hold_history_refuses():
    """PRD: "An empty history must refuse, not admit."

    A predicate that accepts the empty case certifies STRUCTURE (c is blocked
    by exactly one park) rather than CAPABILITY (c's hold is short enough to
    fit) — the whole point of the safety factor is lost.
    """
    clock = _Clock()
    scheduler = _make_scheduler(clock=clock)
    task, modules = _park_blocked_candidate(scheduler, clock, candidate_spans=())

    assert scheduler.predicted_hold(task) is None, (
        'setup guard: the candidate must have NO samples on its own module'
    )
    assert scheduler._backfill_admission('c', task, modules) is None


def test_below_the_sample_floor_refuses():
    """Two samples under the default floor of three → None."""
    clock = _Clock()
    scheduler = _make_scheduler(clock=clock)
    task, modules = _park_blocked_candidate(
        scheduler, clock, candidate_spans=(50.0, 150.0)
    )

    assert scheduler.predicted_hold(task) is None
    assert scheduler._backfill_admission('c', task, modules) is None


def test_a_park_older_than_the_cutoff_refuses():
    """The clause the model demanded by name.

    Without an age cutoff, reify starver 4956 flips from
    eventually-dispatched to never-dispatched in-window
    (PARKING_MODEL_REPORT.md:256): a park that keeps being back-filled past
    is a park that never assembles.
    """
    clock = _Clock()
    scheduler = _make_scheduler(clock=clock)
    task, modules = _park_blocked_candidate(scheduler, clock, park_age=3601.0)

    assert scheduler._backfill_admission('c', task, modules) is None


def test_an_unparseable_park_install_stamp_refuses():
    """Unknown age must refuse, never read as "brand new".

    park_age_secs answers None here (deliberately unlike the dashboard's
    display-oriented fail-soft-to-0), and admission must honour it: a soft
    zero would make a corrupt stamp the most permissive state in the system.
    """
    clock = _Clock()
    scheduler = _make_scheduler(clock=clock)
    task, modules = _park_blocked_candidate(scheduler, clock)
    scheduler.lock_table._park_install_at['p'] = 'not-a-timestamp'

    assert scheduler.lock_table.park_age_secs('p', now=FIXED_DT) is None
    assert scheduler._backfill_admission('c', task, modules) is None


def test_a_missing_park_install_stamp_refuses():
    """Same answer when the stamp is absent rather than corrupt."""
    clock = _Clock()
    scheduler = _make_scheduler(clock=clock)
    task, modules = _park_blocked_candidate(scheduler, clock)
    del scheduler.lock_table._park_install_at['p']

    assert scheduler.lock_table.park_age_secs('p', now=FIXED_DT) is None
    assert scheduler._backfill_admission('c', task, modules) is None


def test_a_predicted_hold_larger_than_the_gap_refuses():
    """bound 250 against a 200s gap → None."""
    clock = _Clock()
    scheduler = _make_scheduler(clock=clock)
    # 1000 - 800 = 200 left, under the bound of 100 * 2.5 = 250.
    task, modules = _park_blocked_candidate(scheduler, clock, gap_elapsed=800.0)

    assert scheduler._provable_assembly_delay('p') == pytest.approx(200.0)
    assert scheduler._backfill_admission('c', task, modules) is None


def test_a_park_waiting_on_nothing_refuses_even_a_tiny_candidate():
    """No gap to lend → None, however short the candidate's hold.

    p is parked but nothing blocks it, so it assembles on the next tick.  Any
    borrowed time here comes straight out of p's own dispatch.
    """
    clock = _Clock()
    scheduler = _make_scheduler(clock=clock)
    task, modules = _park_blocked_candidate(
        scheduler, clock, candidate_spans=(1.0, 1.0, 1.0), gap_elapsed=None
    )

    assert scheduler.predicted_hold(task) == pytest.approx(1.0)
    assert scheduler._provable_assembly_delay('p') == 0.0
    assert scheduler._backfill_admission('c', task, modules) is None


def test_two_blocking_parks_refuse_when_only_one_of_them_qualifies():
    """Admission is a CONJUNCTION over every blocking park.

    Acquisition has to bypass ALL of them to dispatch, so "find one willing
    lender" would admit a candidate that then fails to acquire — churn with
    no dispatch, and a grant event describing a borrow that never happened.
    """

    def _build(*, with_second_park: bool) -> tuple[Scheduler, dict, list[str]]:
        clock = _Clock()
        scheduler = _make_scheduler(clock=clock)
        task = _task('c', ['pkg_c/src/pkg_c/one.py', 'pkg_c/src/pkg_c/two.py'])
        modules = scheduler._get_modules(task)
        assert len(modules) == 2, 'setup: the candidate needs two distinct modules'

        assert scheduler.lock_table.try_acquire('h', GAP_MODULES)
        # p1 is a model citizen: fresh park, 900s of provable wait.
        clock.now = FIXED_DT - timedelta(seconds=60.0)
        scheduler.lock_table.install_parks('p1', [modules[0]] + GAP_MODULES, 'medium')
        if with_second_park:
            # p2 fails every clause it can: ancient, and waiting on nothing.
            clock.now = FIXED_DT - timedelta(seconds=7200.0)
            scheduler.lock_table.install_parks('p2', [modules[1]], 'medium')
        clock.now = FIXED_DT

        _feed_spans(scheduler, modules, list(CANDIDATE_SPANS))
        _feed_spans(scheduler, GAP_MODULES, list(GAP_SPANS))
        scheduler._hold_history.observe_acquired(
            'h', GAP_MODULES, at=FIXED_DT.timestamp() - 100.0
        )
        return scheduler, task, modules

    both, task, modules = _build(with_second_park=True)
    assert both.lock_table.park_owners_blocking('c', modules) == ['p1', 'p2']
    assert both._backfill_admission('c', task, modules) is None

    # Non-vacuity: p1 alone DOES qualify, so the refusal above is the
    # conjunction talking, not a broken scenario.
    only_p1, task, modules = _build(with_second_park=False)
    assert only_p1._backfill_admission('c', task, modules) is not None


# ===========================================================================
# Boundary row 8: the whole predicate through the REAL dispatch scan
# ===========================================================================
#
# Everything above tests the pieces in isolation.  These drive
# ``acquire_next()`` on a scheduler with a real ModuleLockTable, a real
# TickContext and a recording event store, so the wiring itself — the second
# try_acquire, the grant emission, and everything the backfill must NOT
# disturb — is what is under test.
#
# The world each case builds is C7's own scenario:
#
#   h  holds  GAP_MODULE, with 900s provably left
#   p  a high-priority STARVER, parked on [c's module, GAP_MODULE] 60s ago,
#      unable to assemble because h holds GAP_MODULE
#   c  a low-priority narrow candidate wanting only its own module, blocked
#      solely by p's park, predicted hold 100s -> bound 250s
#
# p must be a real task in the candidate list, not just a park owner: the
# park-GC phase evicts any park whose owner is missing from tasks_by_id, so a
# synthetic owner would be gone before the scored phase ever ran.

PARK_OWNER_FILES = ['pkg_c/src/pkg_c/thing.py', 'gapmod/src/gapmod/held.py']
# A SECOND candidate module that p is NOT parked on, used only by the INV-3
# case.  It has to be outside p's park set: an occupier sitting on p's own
# parked module would also shrink p's provable gap — correctly, and all the
# way past the bound, since c's predicted hold is the median of that very
# module and the safety factor then guarantees bound > gap.  Admission would
# refuse on the arithmetic and never reach the lock gate the test is about.
OCCUPIED_FILES = ['pkg_d/src/pkg_d/busy.py']


def _scored_world(
    scheduler: Scheduler,
    clock: _Clock,
    *,
    candidate_spans: tuple[float, ...] = CANDIDATE_SPANS,
    park_age: float = 60.0,
    occupier: bool = False,
) -> list[dict]:
    """Build C7's dispatch scenario; return the candidate task list.

    ``occupier=True`` widens the candidate to a SECOND module — one p is not
    parked on — and puts a real foreign HOLDER there, so admission still
    passes but ``try_acquire`` must refuse: the INV-3 case.
    """
    candidate_files = list(CANDIDATE_FILES) + (OCCUPIED_FILES if occupier else [])
    candidate = _task('c', candidate_files)
    candidate['priority'] = 'low'
    starver = _task('p', PARK_OWNER_FILES)
    starver['priority'] = 'high'
    modules = scheduler._get_modules(candidate)

    assert scheduler.lock_table.try_acquire('h', GAP_MODULES)
    if occupier:
        occupied = scheduler._get_modules(_task('occupier', OCCUPIED_FILES))
        assert scheduler.lock_table.try_acquire('occupier', occupied)

    clock.now = FIXED_DT - timedelta(seconds=park_age)
    scheduler.lock_table.install_parks('p', scheduler._get_modules(starver), 'high')
    clock.now = FIXED_DT

    if candidate_spans:
        _feed_spans(scheduler, modules, list(candidate_spans))
    _feed_spans(scheduler, GAP_MODULES, list(GAP_SPANS))
    scheduler._hold_history.observe_acquired(
        'h', GAP_MODULES, at=FIXED_DT.timestamp() - 100.0
    )
    return [candidate, starver]


def _events(store, name: str) -> list[dict]:
    """Payloads of every recorded event whose type string ends in *name*."""
    return [
        payload
        for event_type, payload in store.events
        if event_type.split('.')[-1] == name or event_type == name
    ]


@pytest.mark.asyncio
async def test_backfill_dispatches_the_candidate_and_emits_one_grant():
    """The grant path end to end, through the real scored scan."""
    clock = _Clock()
    store = _RecordingEventStore()
    scheduler = _make_scheduler(clock=clock, event_store=store)
    scheduler.finish_startup()
    tasks = _scored_world(scheduler, clock)
    scheduler.get_tasks = AsyncMock(return_value=tasks)

    assignment = await scheduler.acquire_next()

    assert assignment is not None, 'the backfiller must actually dispatch'
    assert assignment.task_id == 'c'

    grants = _events(store, 'park_backfill_granted')
    assert len(grants) == 1, f'expected exactly one grant event, got {grants}'
    assert grants[0]['task_id'] == 'c'
    data = grants[0]['data']
    assert data['predicted_hold'] == pytest.approx(100.0)
    assert data['safety_factor'] == pytest.approx(2.5)
    assert data['admission_bound'] == pytest.approx(250.0)
    assert data['provable_assembly_delay'] == pytest.approx(900.0)
    assert data['park_owners'] == ['p']
    assert data['modules'] == list(scheduler._get_modules(tasks[0]))


@pytest.mark.asyncio
async def test_a_backfilled_dispatch_still_routes_through_the_lock_chokepoint():
    """lock_acquired fires and the predictor sees the acquire.

    The backfill changes the try_acquire CALL, never the emit, so the
    single-writer guard stays green and the grant's own hold is itself
    measured — which is what makes the overstay rate observable at all.
    """
    clock = _Clock()
    store = _RecordingEventStore()
    scheduler = _make_scheduler(clock=clock, event_store=store)
    scheduler.finish_startup()
    tasks = _scored_world(scheduler, clock)
    scheduler.get_tasks = AsyncMock(return_value=tasks)

    assignment = await scheduler.acquire_next()
    assert assignment is not None

    acquired = [e for e in _events(store, 'lock_acquired') if e['task_id'] == 'c']
    assert len(acquired) == 1, 'the backfilled dispatch must emit lock_acquired'
    assert scheduler._hold_history.open_modules('c') == list(
        scheduler._get_modules(tasks[0])
    ), 'the predictor must see a backfilled acquire like any other'


@pytest.mark.asyncio
async def test_a_candidate_with_no_history_does_not_backfill():
    """No samples → no dispatch, no grant.  An empty history refuses."""
    clock = _Clock()
    store = _RecordingEventStore()
    scheduler = _make_scheduler(clock=clock, event_store=store)
    scheduler.finish_startup()
    tasks = _scored_world(scheduler, clock, candidate_spans=())
    scheduler.get_tasks = AsyncMock(return_value=tasks)

    assignment = await scheduler.acquire_next()

    assert assignment is None
    assert _events(store, 'park_backfill_granted') == []


@pytest.mark.asyncio
async def test_a_park_older_than_the_cutoff_does_not_admit_a_backfiller():
    """The age clause holds through the real scan, not just in the predicate."""
    clock = _Clock()
    store = _RecordingEventStore()
    scheduler = _make_scheduler(clock=clock, event_store=store)
    scheduler.finish_startup()
    tasks = _scored_world(scheduler, clock, park_age=3601.0)
    scheduler.get_tasks = AsyncMock(return_value=tasks)

    assignment = await scheduler.acquire_next()

    assert assignment is None
    assert _events(store, 'park_backfill_granted') == []


@pytest.mark.asyncio
async def test_try_acquire_not_admission_is_the_acting_basis():
    """INV-3: admission passes, a real HOLDER occupies the module, no dispatch.

    The scoring-time read is advisory — the lock layer decides.  Admission
    deliberately does not precompute "c's modules are otherwise free"; it
    lets the untouched limit gate on the second try_acquire answer that on
    live state.  So an admitted-but-contended candidate costs one failed
    acquire and emits NOTHING, rather than producing a grant record for a
    borrow that never happened.
    """
    clock = _Clock()
    store = _RecordingEventStore()
    scheduler = _make_scheduler(clock=clock, event_store=store)
    scheduler.finish_startup()
    tasks = _scored_world(scheduler, clock, occupier=True)
    scheduler.get_tasks = AsyncMock(return_value=tasks)
    candidate, modules = tasks[0], scheduler._get_modules(tasks[0])

    # The predicate itself is happy — it is the lock layer that refuses.
    assert scheduler._backfill_admission('c', candidate, modules) is not None

    assignment = await scheduler.acquire_next()

    assert assignment is None
    assert _events(store, 'park_backfill_granted') == []


@pytest.mark.asyncio
async def test_the_backfiller_borrows_the_park_it_does_not_consume_it():
    """Backfill leaves p's park exactly as it found it.

    Not a preemption and not a reservation use: p keeps its park, its
    modules and its rank, and ``reservation_used`` stays reserved for an
    owner consuming its OWN park.  The park-consumption branch in the scored
    loop is by construction never reached by a backfiller.
    """
    clock = _Clock()
    store = _RecordingEventStore()
    scheduler = _make_scheduler(clock=clock, event_store=store)
    scheduler.finish_startup()
    tasks = _scored_world(scheduler, clock)
    scheduler.get_tasks = AsyncMock(return_value=tasks)
    stacks_before = scheduler.lock_table.snapshot_park_stacks()

    assignment = await scheduler.acquire_next()
    assert assignment is not None and assignment.task_id == 'c'

    assert [e for e in _events(store, 'reservation_used') if e['task_id'] == 'p'] == []
    parks = scheduler.lock_table.snapshot_parks()
    assert 'p' in parks, 'the park must survive the backfill'
    assert scheduler._get_modules(tasks[0])[0] in parks['p']['modules']
    assert scheduler.lock_table.snapshot_park_stacks() == stacks_before, (
        'ranks and stack order are untouched — backfill borrows, it never evicts'
    )


# ===========================================================================
# PRD section 7 rejection guards — what this task deliberately did NOT build
# ===========================================================================
#
# Two mechanisms were rejected on measured grounds and must STAY rejected.  A
# later reader with the backfill machinery in front of them has every reason
# to think "while we're here, park below rank 1" or "parks should expire" is
# the natural next move; these make that drift loud instead of silent.
#
# AST, not text grep: a call can span lines, and the enclosing function is
# exactly what each invariant is about.

_SRC_DIR = Path(__file__).parent.parent / 'src'


def _enclosing_function_ranges(tree: ast.Module) -> list[tuple[str, int, int]]:
    """``(name, lineno, end_lineno)`` for every function/method in *tree*."""
    return [
        (node.name, node.lineno, node.end_lineno or node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _call_sites(path: Path, method_name: str) -> list[tuple[str, int, str]]:
    """``(file, line, enclosing function)`` for each ``….<method_name>(…)``."""
    tree = ast.parse(path.read_text())
    ranges = _enclosing_function_ranges(tree)
    sites: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == method_name):
            continue
        # A method DEFINITION is not a call site; only Call nodes reach here.
        enclosing = [name for name, lo, hi in ranges if lo <= node.lineno <= hi]
        sites.append((path.name, node.lineno, enclosing[-1] if enclosing else '<module>'))
    return sites


def _emit_sites(path: Path, event_name: str) -> list[tuple[str, int, str]]:
    """``(file, line, enclosing function)`` for each ``…emit(EventType.<event>…)``."""
    tree = ast.parse(path.read_text())
    ranges = _enclosing_function_ranges(tree)
    sites: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == 'emit'):
            continue
        args = list(node.args) + [kw.value for kw in node.keywords if kw.arg == 'event_type']
        for arg in args:
            if (
                isinstance(arg, ast.Attribute)
                and arg.attr == event_name
                and isinstance(arg.value, ast.Name)
                and arg.value.id == 'EventType'
            ):
                enclosing = [name for name, lo, hi in ranges if lo <= node.lineno <= hi]
                sites.append((path.name, node.lineno, enclosing[-1] if enclosing else '<module>'))
    return sites


def test_backfill_installs_no_park():
    """``install_parks`` keeps exactly its two pre-existing callers.

    Below-rank-1 park INSTALLATION is rejected by PRD section 7 on measured
    grounds.  Backfill grants a candidate passage through an existing park;
    it must never create one, or the park population — the thing the parking
    model measured — changes shape underneath the evidence.
    """
    sites = [
        site
        for path in sorted(_SRC_DIR.rglob('*.py'))
        for site in _call_sites(path, 'install_parks')
    ]

    assert sorted(fn for _f, _l, fn in sites) == [
        '_bump_skip_and_maybe_park',
        '_phase_reserve_now',
    ], f'install_parks callers changed: {sites}'


def test_backfill_adds_no_park_lease_or_expiry():
    """``reservation_expired`` still has exactly one emit site, in park GC.

    Park leases were removed deliberately (task 1228) and PRD section 7
    rejects reintroducing them.  ``park_age_secs`` is a READ used to REFUSE
    an admission — it must never become a reason to evict.  A second emit
    site would mean wall-clock park death came back in through this door.
    """
    sites = [
        site
        for path in sorted(_SRC_DIR.rglob('*.py'))
        for site in _emit_sites(path, 'reservation_expired')
    ]

    assert len(sites) == 1, f'expected one reservation_expired emit site, got {sites}'
    assert sites[0][2] == '_phase_park_gc', (
        f'reservation_expired emitted from {sites[0][2]}: park eviction stays '
        f'owner-state-driven, never wall-clock-driven'
    )


def test_the_pin_loop_is_not_given_backfill():
    """``_phase_select_pins`` still calls plain ``try_acquire``.

    C7 anchors on the scored loop.  Pins already bypass fairness entirely, so
    admitting them through parks as well would hand the one path with no
    fairness accounting a second override — and the safety factor exists
    precisely to bound what the scored loop borrows.
    """
    tree = ast.parse((_SRC_DIR / 'orchestrator' / 'scheduler.py').read_text())
    pins = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == '_phase_select_pins'
    ]
    assert len(pins) == 1, 'expected exactly one _phase_select_pins definition'

    acquires = [
        node
        for node in ast.walk(pins[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr.startswith('try_acquire')
    ]
    assert acquires, 'meta: the pin loop must still take locks at all'
    for call in acquires:
        assert not [kw for kw in call.keywords if kw.arg == 'admitted_parks'], (
            f'_phase_select_pins line {call.lineno} passes admitted_parks — '
            f'backfill is scoped to the scored loop'
        )


def test_the_ast_scanners_can_actually_find_their_targets(tmp_path):
    """Meta-test: a scanner that finds nothing would pass every guard above.

    The probe goes in ``tmp_path``, never in the tests tree: this suite runs
    under pytest-xdist, so a fixed shared path is a race, and a hard interrupt
    mid-test would strand an untracked .py file where the guards' own rglob
    would find it.
    """
    probe = tmp_path / '_backfill_guard_probe.py'
    probe.write_text(
        'from x import EventType\n'
        'def f(self):\n'
        '    self.lock_table.install_parks(\n'
        '        "A", ["m"], "high")\n'
        '    self.event_store.emit(\n'
        '        EventType.reservation_expired, task_id="A", data={})\n'
    )

    assert [(line, fn) for _f, line, fn in _call_sites(probe, 'install_parks')] == [(3, 'f')]
    assert [(line, fn) for _f, line, fn in _emit_sites(probe, 'reservation_expired')] == [(5, 'f')]


# ===========================================================================
# Overstay settlement at release
# ===========================================================================
#
# Admission promised a bound; release is where that promise is judged.  The
# comparison is realized-vs-bound, and BOTH numbers ride on the event, so no
# operator has to reconstruct the arithmetic from log lines to see why a hold
# counted as a breach (INV-2).
#
# The clock is settable, so every realized duration below is exact.


def _record_grant(
    scheduler: Scheduler,
    task_id: str = 'c',
    *,
    predicted_hold: float = 200.0,
    admission_bound: float = 500.0,
) -> None:
    """Record one grant as a dispatch would, at the clock's current instant."""
    scheduler._record_backfill_grant(
        _BackfillGrant(
            task_id=task_id,
            modules=('mod-c',),
            park_owners=('p',),
            predicted_hold=predicted_hold,
            safety_factor=2.5,
            admission_bound=admission_bound,
            provable_assembly_delay=900.0,
        )
    )


def _settled_after(seconds: float, **grant_kwargs):
    """Grant at T, release at T + *seconds*; return ``(scheduler, store)``."""
    clock = _Clock()
    store = _RecordingEventStore()
    scheduler = _make_scheduler(clock=clock, event_store=store)
    assert scheduler.lock_table.try_acquire('c', ['mod-c'])
    _record_grant(scheduler, **grant_kwargs)
    clock.now = FIXED_DT + timedelta(seconds=seconds)
    scheduler.release('c')
    return scheduler, store


def test_a_hold_inside_the_bound_is_not_an_overstay():
    """400s realized against a 500s bound → nothing fired."""
    scheduler, store = _settled_after(400.0)

    assert _events(store, 'park_backfill_overstay') == []
    assert scheduler._backfill_overstay_total == 0


def test_a_hold_exactly_at_the_bound_is_not_an_overstay():
    """Meeting the bound is not breaching it.

    The bound is what admission PROMISED, so ``realized > bound`` is the
    breach test.  Landing exactly on it is the best possible outcome for a
    prediction, and counting it as a breach would inflate the measured rate
    the safety factor is meant to be tuned from.
    """
    scheduler, store = _settled_after(500.0)

    assert _events(store, 'park_backfill_overstay') == []
    assert scheduler._backfill_overstay_total == 0


def test_a_hold_past_the_bound_emits_one_overstay_with_predicted_and_realized():
    """900s realized against a 500s bound → one event, 400s over."""
    scheduler, store = _settled_after(900.0)

    overstays = _events(store, 'park_backfill_overstay')
    assert len(overstays) == 1
    assert overstays[0]['task_id'] == 'c'
    data = overstays[0]['data']
    assert data['predicted_hold'] == pytest.approx(200.0)
    assert data['safety_factor'] == pytest.approx(2.5)
    assert data['admission_bound'] == pytest.approx(500.0)
    assert data['realized_hold'] == pytest.approx(900.0)
    assert data['overstay_secs'] == pytest.approx(400.0)
    assert data['park_owners'] == ['p']
    assert scheduler._backfill_overstay_total == 1


def test_releasing_a_task_that_never_backfilled_settles_nothing():
    """No grant, no event, no counter movement."""
    clock = _Clock()
    store = _RecordingEventStore()
    scheduler = _make_scheduler(clock=clock, event_store=store)
    assert scheduler.lock_table.try_acquire('ordinary', ['mod-x'])

    clock.now = FIXED_DT + timedelta(seconds=99999.0)
    scheduler.release('ordinary')

    assert _events(store, 'park_backfill_overstay') == []
    assert scheduler._backfill_grants_total == 0
    assert scheduler._backfill_overstay_total == 0


def test_the_grant_is_popped_so_a_second_release_double_counts_nothing():
    """Settlement consumes the grant.

    Release is called from more than one path (and defensively re-called on
    cleanup), so a grant left in place would let one breach be counted twice
    and quietly inflate the rate that drives the escalation.
    """
    scheduler, store = _settled_after(900.0)
    assert scheduler._backfill_overstay_total == 1

    scheduler.release('c')

    assert len(_events(store, 'park_backfill_overstay')) == 1
    assert scheduler._backfill_overstay_total == 1
    assert 'c' not in scheduler._backfill_grants


def test_the_rate_denominator_is_grants_not_releases():
    """Three grants, one breach → 3 and 1."""
    clock = _Clock()
    store = _RecordingEventStore()
    scheduler = _make_scheduler(clock=clock, event_store=store)
    for index, realized in enumerate((100.0, 900.0, 200.0)):
        task_id = f'c{index}'
        assert scheduler.lock_table.try_acquire(task_id, [f'mod-{index}'])
        clock.now = FIXED_DT
        _record_grant(scheduler, task_id)
        clock.now = FIXED_DT + timedelta(seconds=realized)
        scheduler.release(task_id)

    assert scheduler._backfill_grants_total == 3
    assert scheduler._backfill_overstay_total == 1


def test_the_live_grant_map_is_bounded():
    """One more grant than the cap evicts the OLDEST, never grows past it.

    A grant whose owner never releases in this process era (crash, lost
    release) would otherwise leak an entry forever.  Same discipline as the
    bounded live-open map in the predictor: a leak backstop, not a policy
    limit — max_concurrent_tasks keeps the real population orders of
    magnitude below the cap.
    """
    clock = _Clock()
    scheduler = _make_scheduler(clock=clock)
    cap = scheduler_module._BACKFILL_GRANTS_MAX
    for index in range(cap):
        _record_grant(scheduler, f'c{index}')
    assert len(scheduler._backfill_grants) == cap
    assert 'c0' in scheduler._backfill_grants

    _record_grant(scheduler, 'one-too-many')

    assert len(scheduler._backfill_grants) == cap
    assert 'c0' not in scheduler._backfill_grants, 'the OLDEST entry is the one evicted'
    assert 'one-too-many' in scheduler._backfill_grants


def test_an_unreleased_grant_is_silently_lost_with_the_process_era():
    """No release, no settlement — and no exception either.

    In-memory grants share the fate of in-memory hold spans: a process era
    ends and they go with it.  That is a deliberate non-event, not an error
    condition to raise on.
    """
    clock = _Clock()
    store = _RecordingEventStore()
    scheduler = _make_scheduler(clock=clock, event_store=store)
    _record_grant(scheduler, 'never-released')

    clock.now = FIXED_DT + timedelta(seconds=99999.0)

    assert 'never-released' in scheduler._backfill_grants
    assert _events(store, 'park_backfill_overstay') == []
    assert scheduler._backfill_overstay_total == 0


# ===========================================================================
# INV-4: a rate-threshold escalation, not a per-event one
# ===========================================================================
#
# The model measured 7-9% overstay at safety x2.5
# (PARKING_MODEL_REPORT.md:254-255).  A single breach is that model working
# as designed; a sustained rate several times higher means the predictor has
# drifted or the factor is wrong.  So the signal is a RATE over a denominator
# floor, filed once per episode — not one escalation per overstay, which is
# the storm INV-4 exists to forbid.


class _FakeEscalationQueue:
    """Minimal queue exposing only what the filer touches."""

    def __init__(self, *, open_l1: bool = False, submit_raises: bool = False) -> None:
        self.submitted: list = []
        self._open_l1 = open_l1
        self._submit_raises = submit_raises
        self.checked: list[str] = []

    def has_open_l1(self, sentinel: str) -> bool:
        self.checked.append(sentinel)
        return self._open_l1

    def make_id(self, sentinel: str) -> str:
        return f'esc-{sentinel}'

    def submit(self, escalation) -> None:
        if self._submit_raises:
            raise RuntimeError('escalation queue is down')
        self.submitted.append(escalation)


def _drive_grants(scheduler: Scheduler, clock: _Clock, *, grants: int, overstays: int):
    """Record *grants* grants, then release *overstays* of them past the bound.

    All grants are recorded BEFORE any release, so the denominator is already
    at its final value when the first breach settles — otherwise the rate
    would cross the threshold on an early partial denominator and the test
    would be asserting an accident of ordering.
    """
    clock.now = FIXED_DT
    for index in range(grants):
        _record_grant(scheduler, f'c{index}')
    for index in range(grants):
        clock.now = FIXED_DT + timedelta(seconds=900.0 if index < overstays else 100.0)
        scheduler.release(f'c{index}')


def _escalating_scheduler(**queue_kwargs):
    clock = _Clock()
    scheduler = _make_scheduler(clock=clock, event_store=_RecordingEventStore())
    queue = _FakeEscalationQueue(**queue_kwargs)
    # The fake implements exactly the three members the filer touches
    # (has_open_l1 / make_id / submit); the cast is the stand-in, not a claim
    # that it is a full EscalationQueue.
    scheduler.escalation_queue = cast('EscalationQueue', queue)
    return scheduler, clock, queue


def test_the_escalation_policy_constants_are_what_the_model_supports():
    """0.25 rate over a floor of 8 grants — pinned, with the reasoning.

    The threshold is ~3x the measured 7-9%, and the floor is reachable within
    days at the modelled 23-122 grants/30d.  Code constants rather than config
    leaves: C7 mandates exactly three config literals, and escalation
    thresholds are code properties here (the _FM_READ_FAILURE_ESCALATION
    _THRESHOLD precedent).
    """
    assert pytest.approx(0.25) == scheduler_module._BACKFILL_OVERSTAY_RATE_THRESHOLD
    assert scheduler_module._BACKFILL_OVERSTAY_MIN_GRANTS == 8
    assert isinstance(scheduler_module._BACKFILL_OVERSTAY_SENTINEL, str)


def test_a_perfect_overstay_rate_below_the_grant_floor_files_nothing():
    """4 grants, 4 breaches — 100%, and still silent.

    A 1-of-1 breach is noise, not a rate.  Without a denominator floor the
    very first back-filled dispatch that ran long would file an L1 claiming
    a 100% overstay rate.
    """
    scheduler, clock, queue = _escalating_scheduler()

    _drive_grants(scheduler, clock, grants=4, overstays=4)

    assert scheduler._backfill_overstay_total == 4
    assert queue.submitted == []


def test_a_sustained_overstay_rate_files_exactly_one_l1():
    """8 grants, 2 breaches = 25% → one L1 naming rate, counts, factor, remedy."""
    scheduler, clock, queue = _escalating_scheduler()

    _drive_grants(scheduler, clock, grants=8, overstays=2)

    assert len(queue.submitted) == 1, f'expected one L1, got {queue.submitted}'
    esc = queue.submitted[0]
    assert esc.agent_role == 'orchestrator-scheduler'
    assert esc.severity == 'blocking'
    assert esc.level == 1
    assert esc.category == 'risk_identified'
    detail = esc.detail
    # The MEASURED rate and both counts, so the claim is checkable.
    assert '25' in detail, 'the measured rate must appear'
    assert '8' in detail and '2' in detail, 'both counts must appear'
    # The configured factor and the concrete remedy.
    assert 'backfill_safety_factor' in detail
    assert '2.5' in detail
    assert '2.9' in detail, (
        'the remedy is to raise the factor toward the measured x2.9 upper '
        'bracket — an escalation without a next action is just an alarm'
    )


def test_an_already_open_l1_suppresses_further_filings():
    """One open escalation per episode, not one per release."""
    scheduler, clock, queue = _escalating_scheduler(open_l1=True)

    _drive_grants(scheduler, clock, grants=8, overstays=4)

    assert queue.submitted == []
    assert queue.checked, 'dedup must actually consult has_open_l1'


def test_the_normal_measured_regime_stays_silent():
    """1 breach in 20 grants (5%, near the modelled 7-9%) files nothing.

    If the ordinary regime escalates, the escalation carries no information
    and operators learn to close it unread.
    """
    scheduler, clock, queue = _escalating_scheduler()

    _drive_grants(scheduler, clock, grants=20, overstays=1)

    assert scheduler._backfill_overstay_total == 1
    assert queue.submitted == []


def test_no_escalation_queue_warns_and_does_not_raise(caplog):
    """Bare-Scheduler unit tests stay green, and the gap is stated out loud."""
    clock = _Clock()
    scheduler = _make_scheduler(clock=clock, event_store=_RecordingEventStore())
    assert scheduler.escalation_queue is None

    with caplog.at_level(logging.WARNING):
        _drive_grants(scheduler, clock, grants=8, overstays=4)

    assert scheduler._backfill_overstay_total == 4
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_a_failing_submit_never_breaks_the_release():
    """Filing is best-effort; settlement and release complete regardless.

    A release that raised because escalation filing failed would strand the
    task's locks — turning a reporting problem into a dispatch outage.
    """
    scheduler, clock, queue = _escalating_scheduler(submit_raises=True)

    _drive_grants(scheduler, clock, grants=8, overstays=4)

    assert scheduler._backfill_overstay_total == 4
    assert queue.submitted == []
    assert scheduler._backfill_grants == {}, 'every grant still settled'
