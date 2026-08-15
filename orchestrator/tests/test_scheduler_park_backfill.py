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

from datetime import UTC, datetime, timedelta

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import Scheduler

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


def _make_scheduler(*, clock: _Clock | None = None, **config_overrides) -> Scheduler:
    """A bare Scheduler on a frozen (or caller-driven) wall clock.

    ``max_per_module=1`` matches the shape the parking model measured (one
    holder per module), and the frozen clock makes park age and realized hold
    durations exact rather than approximate.  It is a DEFAULT, not a pin: the
    two-holders-on-one-module case needs a limit of 2 to have two holders at
    all, and overriding it must not raise a duplicate-kwarg TypeError.
    """
    config = OrchestratorConfig(**{'max_per_module': 1, **config_overrides})
    return Scheduler(config, wall_time_source=clock or (lambda: FIXED_DT))


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
