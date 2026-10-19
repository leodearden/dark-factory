"""End-to-end scheduler boundary test — PSI saturation transition (task 2329,
DA4 of PRD docs/prds/dispatch-admission-load-cap.md).

DA4 is the batch leaf / integration gate for the L3b dispatch-admission load
cap: it drives the REAL ``Scheduler.acquire_next()`` selection + a real
``ModuleLockTable`` + a real SQLite ``EventStore`` across a PSI saturation
TRANSITION (idle -> saturated -> idle) with a burst of ready HEAVY
(normal-kind) tasks plus at least one DETERMINISTIC task, proving the PRD §6
work-conserving + deadlock-free invariants end-to-end — not just the DA3 gate
(task 2328)'s isolated single-tick unit tests in
``test_scheduler_dispatch_admission.py``.

This is a TEST-ONLY task — no new production module.  The "impl" steps in
this file build a reusable multi-tick transition harness (``_PsiFeed`` +
``_Driver``) that models the harness main loop itself (live in-flight
tracking, max_concurrent enforcement, and task completion via the REAL
``scheduler.release()``), since ``acquire_next()`` itself enforces neither.

Covers four scenarios (see the ``TestPsiSaturationTransition`` class):
  test_full_transition_idle_saturated_idle
      Scenario 1 — full idle -> saturated -> idle transition end-to-end.
  test_floor_holds_under_sustained_saturation
      Scenario 2 — deadlock-freedom at the peak of SUSTAINED saturation.
  test_recovery_restores_full_dispatch_up_to_cap
      Scenario 3 — work-conserving restore up to max_concurrent_tasks with
      NO residual hold after recovery.
  test_runqueue_transition_through_the_real_reader
      Scenario 4 — PRD plans/load-throttle-harmonisation-prd.md boundary
      row 6: idle -> runqueue-saturated -> idle through the REAL reader over
      a fixture cgroup tree (task 3590).

Helpers are MODULE-LOCAL (not conftest.py) per the DA2/DA3 rationale: a
conftest.py edit trips verify.py's has_conftest scoped-verify fallback.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock

import _psi_admission_yaml
import pytest
from _psi_admission_yaml import cpu_arm_cfg
from shared.psi import PsiSample, read_psi_sample

from orchestrator.config import FusedMemoryConfig, OrchestratorConfig, PsiAdmissionConfig
from orchestrator.event_store import EventStore
from orchestrator.scheduler import Scheduler

_nondefault_psi_admission_yaml = _psi_admission_yaml.nondefault_psi_admission_yaml

# ---------------------------------------------------------------------------
# Minimal task-dict + PSI-sample helpers (mirrored from
# test_scheduler_dispatch_admission.py:44-93 — same minimal shape acquire_next
# reads, so DA4's fixtures match DA3's unit-test fixtures field-for-field).
# ---------------------------------------------------------------------------


def _heavy_task(
    task_id: str,
    priority: str = 'medium',
    files: list[str] | None = None,
) -> dict:
    """Minimal normal-kind (heavy) pending-task dict carrying every field
    acquire_next reads.  Defaults to a file DISTINCT per task_id so
    max_per_module=1 never masks the PSI gate with lock contention."""
    return {
        'id': task_id,
        'title': f'Task {task_id}',
        'status': 'pending',
        'priority': priority,
        'dependencies': [],
        'metadata': {'task_kind': 'normal', 'files': files or [f'mod{task_id}']},
    }


def _det_task(
    task_id: str,
    priority: str = 'medium',
    files: list[str] | None = None,
) -> dict:
    """Minimal deterministic-kind pending-task dict carrying every field
    acquire_next reads."""
    return {
        'id': task_id,
        'title': f'Task {task_id}',
        'status': 'pending',
        'priority': priority,
        'dependencies': [],
        'metadata': {'task_kind': 'deterministic', 'files': files or [f'mod{task_id}']},
    }


def _psi(
    *,
    cpu_some10: float = 0.0,
    mem_some10: float = 0.0,
    mem_full10: float = 0.0,
    io_some10: float = 0.0,
    read_ok: bool = True,
) -> PsiSample:
    """Build a PsiSample; all metrics idle (0.0) unless overridden."""
    return PsiSample(
        cpu_some10=cpu_some10,
        mem_some10=mem_some10,
        mem_full10=mem_full10,
        io_some10=io_some10,
        read_ok=read_ok,
    )


# ---------------------------------------------------------------------------
# Multi-tick transition harness
#
# A controllable clock, a mutable PSI feed, and a driver that wraps one real
# Scheduler + real EventStore and models the pieces of the harness main loop
# that acquire_next() deliberately does not own: live in-flight tracking via
# get_tasks, scripted task completion (_Driver.complete, via the REAL
# scheduler.release()), and a max_concurrent cap (_Driver.run_until_quiescent)
# mirroring the harness asyncio.Semaphore. Every dispatch decision itself is
# still made by the real scheduler/lock-table/event-store — the driver only
# supplies the surrounding bookkeeping.
# ---------------------------------------------------------------------------


class _FakeClock:
    """Minimal controllable monotonic-style clock (mirrors DA3's, in
    test_scheduler_dispatch_admission.py:258-268)."""

    def __init__(self, start: float = 1000.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, secs: float) -> None:
        self._t += secs


class _PsiFeed:
    """Mutable PSI sample holder assignable to ``scheduler._read_psi_sample``.

    Starts idle (all metrics 0.0, read_ok=True).  ``set_saturated`` accepts
    any subset of the four PSI metrics as kwargs (unset metrics stay at their
    idle default of 0.0) so callers only need to name the metric(s) driving
    saturation.
    """

    def __init__(self) -> None:
        self._sample: PsiSample = _psi()

    def __call__(self) -> PsiSample:
        return self._sample

    def set_idle(self) -> None:
        self._sample = _psi()

    def set_saturated(
        self,
        *,
        cpu_some10: float = 0.0,
        mem_some10: float = 0.0,
        mem_full10: float = 0.0,
        io_some10: float = 0.0,
    ) -> None:
        self._sample = _psi(
            cpu_some10=cpu_some10,
            mem_some10=mem_some10,
            mem_full10=mem_full10,
            io_some10=io_some10,
            read_ok=True,
        )

    def set_unreadable(self) -> None:
        self._sample = _psi(read_ok=False)


class _Driver:
    """Drives one real Scheduler across many ``acquire_next()`` ticks.

    Models the pieces of the harness main loop that ``acquire_next()``
    itself deliberately does not own: a live ``get_tasks`` view over an
    active-task list, scripted completion, and a max-concurrent cap.  Every
    dispatch decision is made by the REAL scheduler/lock-table/event-store —
    the driver only supplies the surrounding bookkeeping.
    """

    def __init__(self, scheduler: Scheduler, event_store: EventStore) -> None:
        self.scheduler = scheduler
        self.event_store = event_store
        self._active: list[dict] = []
        self._completed: set[str] = set()
        self._heavy_in_flight: set[str] = set()
        self._det_dispatched: set[str] = set()

    @classmethod
    def build(
        cls,
        config: OrchestratorConfig,
        tasks: list[dict],
        *,
        event_store: EventStore,
        time_source,
        psi_feed: _PsiFeed | None = None,
    ) -> _Driver:
        """Without *psi_feed* the scheduler keeps the reader bound at construction."""
        scheduler = Scheduler(config, event_store=event_store, time_source=time_source)
        scheduler.finish_startup()
        if psi_feed is not None:
            scheduler._read_psi_sample = psi_feed
        driver = cls(scheduler, event_store)
        driver._active = list(tasks)
        scheduler.get_tasks = AsyncMock(side_effect=driver._get_tasks)
        return driver

    async def _get_tasks(self, *_args, **_kwargs) -> list[dict]:
        return [t for t in self._active if t['id'] not in self._completed]

    async def tick(self):
        """Drive one acquire_next() tick; record the outcome in the ledger."""
        result = await self.scheduler.acquire_next()
        if result is not None:
            if self.scheduler.is_deterministic(result.task):
                self._det_dispatched.add(result.task_id)
            else:
                self._heavy_in_flight.add(result.task_id)
        return result

    async def run_ticks(self, n: int) -> list:
        return [await self.tick() for _ in range(n)]

    async def complete(self, task_id: str) -> None:
        """Model a task finishing: release its module lock via the REAL
        scheduler.release(), then drop it from the active list so a fixed
        get_tasks backlog never re-offers a finished task as a candidate."""
        self.scheduler.release(task_id)
        self._completed.add(task_id)
        self._heavy_in_flight.discard(task_id)
        self._det_dispatched.discard(task_id)

    async def run_until_quiescent(self, max_in_flight: int) -> None:
        """Drive ticks until heavy in-flight fills ``max_in_flight`` or no
        further candidate is eligible to dispatch.

        Mirrors the harness asyncio.Semaphore (harness.py:1427): unlike PSI
        admission, ``acquire_next()`` itself enforces no max-concurrent cap,
        so the driver must stop ISSUING ticks once ``max_in_flight`` heavies
        are in flight rather than let the scheduler over-dispatch past the
        lane's concurrency limit.  A tick returning None (no free-slot
        candidate left to dispatch) also stops the loop, so a drained
        backlog can never spin forever.
        """
        while self.heavy_in_flight() < max_in_flight:
            result = await self.tick()
            if result is None:
                break

    def heavy_in_flight(self) -> int:
        return len(self._heavy_in_flight)

    def dispatched_ids(self) -> set[str]:
        return set(self._heavy_in_flight) | set(self._det_dispatched)

    def deferred_events(self) -> list[dict]:
        return self.event_store.fetch_events_by_type('dispatch_deferred')

    def deferred_events_since(self, event_id: int) -> list[dict]:
        """dispatch_deferred events strictly after *event_id* (a prior
        window's high-water id) — partitions the event stream into the
        saturated window vs. post-recovery for a no-residual-hold assert."""
        return [e for e in self.deferred_events() if e['id'] > event_id]

    def lock_acquired_events(self) -> list[dict]:
        return self.event_store.fetch_events_by_type('lock_acquired')


_IDLE_PRESSURE = (
    'some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n'
    'full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n'
)
_VERIFY_SCOPE_CGROUP = (
    '/user.slice/user-1000.slice/user@1000.service/df.slice/df-x.slice/'
    'df-verify-x-0123456789ab.scope'
)


class _LoadFixtureTree:
    """A fixture /proc + sysfs tree for the REAL ``shared.psi.read_psi_sample``.

    Host PSI reads idle. The reading process sits in a verify scope under
    ``df-x.slice`` (PRD plans/load-throttle-harmonisation-prd.md §6.3 row 1),
    with ``cpu.pressure`` at the slice and none at the scope leaf, so the own
    reading succeeds only when the resolver is given project_id ``'x'``. The
    test sets ``procs_running``.
    """

    def __init__(self, root: Path) -> None:
        self.proc_stat = root / 'proc' / 'stat'
        self.proc_cgroup = root / 'proc' / 'self' / 'cgroup'
        self.cgroup_root = root / 'sys' / 'fs' / 'cgroup'
        self.proc_cgroup.parent.mkdir(parents=True)
        self.proc_cgroup.write_text(f'0::{_VERIFY_SCOPE_CGROUP}\n')
        scope_dir = self.cgroup_root / _VERIFY_SCOPE_CGROUP.lstrip('/')
        scope_dir.mkdir(parents=True)
        (scope_dir.parent / 'cpu.pressure').write_text(
            'some avg10=12.50 avg60=10.00 avg300=5.00 total=1000\n'
            'full avg10=1.00 avg60=0.50 avg300=0.10 total=100\n'
        )
        self.set_procs_running(0)

    def set_procs_running(self, running: int) -> None:
        self.proc_stat.write_text(f'cpu  1 2 3 4\nprocs_running {running}\nprocs_blocked 0\n')

    def reader(self) -> Callable[..., PsiSample]:
        """The real reader over this tree; project_id is left to the caller."""
        return functools.partial(
            read_psi_sample,
            read=lambda _name: _IDLE_PRESSURE,
            proc_stat_path=self.proc_stat,
            proc_cgroup_path=self.proc_cgroup,
            cgroup_root=self.cgroup_root,
        )


# ---------------------------------------------------------------------------
# Scenario 1 — full idle -> saturated -> idle transition end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPsiSaturationTransition:
    """DA4 integration gate: multi-tick PSI saturation transitions driven
    against the REAL Scheduler.acquire_next() + ModuleLockTable + SQLite
    EventStore (task 2329)."""

    async def test_full_transition_idle_saturated_idle(self, tmp_path) -> None:
        """Scenario 1: idle -> saturated -> idle, full transition.

        Backlog: 4 heavy (H1..H4, distinct files/modules) + 1 deterministic
        (D1).  Heavies outrank D1 (priority='high' vs D1's default 'medium')
        so the idle phase dispatches heavies first, never D1 — isolating D1's
        later dispatch as evidence of the deterministic exemption rather than
        plain scoring luck.
        """
        event_store = EventStore(db_path=tmp_path / 'psi.db', run_id='da4-s1')
        clock = _FakeClock()
        config = OrchestratorConfig(
            max_per_module=1, max_concurrent_tasks=6, psi_admission=cpu_arm_cfg(),
        )

        tasks = [
            _heavy_task('H1', priority='high', files=['src/h1.py']),
            _heavy_task('H2', priority='high', files=['src/h2.py']),
            _heavy_task('H3', priority='high', files=['src/h3.py']),
            _heavy_task('H4', priority='high', files=['src/h4.py']),
            _det_task('D1', files=['src/d1.py']),
        ]

        feed = _PsiFeed()  # starts idle
        driver = _Driver.build(
            config, tasks, event_store=event_store, time_source=clock, psi_feed=feed,
        )

        # --- (A) idle: heavies dispatch, zero dispatch_deferred ---
        await driver.run_ticks(2)
        assert driver.heavy_in_flight() == 2, (
            'two heavy tasks should be in flight after 2 idle ticks'
        )
        assert driver.deferred_events() == [], 'an idle host must never defer dispatch'

        # --- (B) saturated: heavies held, deterministic still dispatches ---
        feed.set_saturated(cpu_some10=90.0)
        saturated_sample = feed()
        await driver.run_ticks(4)

        assert driver.heavy_in_flight() == 2, 'saturation must not admit new heavy dispatch'
        assert 'D1' in driver.dispatched_ids(), (
            'deterministic tasks are exempt from the PSI hold'
        )

        deferred_during_saturation = driver.deferred_events()
        assert len(deferred_during_saturation) >= 1, (
            'saturation must defer at least one heavy dispatch'
        )
        assert deferred_during_saturation[0]['data'] == {
            'metric': 'cpu_some_avg10',
            'in_flight': 2,
            'floor': 1,
            'sample': asdict(saturated_sample),
        }
        assert all(
            e['task_id'] in {'H1', 'H2', 'H3', 'H4'} for e in deferred_during_saturation
        ), 'dispatch_deferred must only ever name a HEAVY candidate, never the deterministic task'
        assert driver.heavy_in_flight() >= config.psi_admission.min_inflight_floor, (
            'in-flight heavy must never fall below the anti-deadlock floor while held'
        )

        # --- (C) recovery: remaining heavies dispatch, no residual hold ---
        feed.set_idle()
        await driver.run_ticks(2)

        assert driver.heavy_in_flight() == 4, (
            'recovery must dispatch the remaining heavies (work-conserving)'
        )
        assert driver.dispatched_ids() == {'H1', 'H2', 'H3', 'H4', 'D1'}
        assert driver.deferred_events() == deferred_during_saturation, (
            'no NEW dispatch_deferred events may occur after the recovery flip'
        )

        lock_events = driver.lock_acquired_events()
        assert {e['task_id'] for e in lock_events} == {'H1', 'H2', 'H3', 'H4', 'D1'}, (
            'the real ModuleLockTable must record a lock_acquired event for '
            'every dispatched task, heavy and deterministic alike'
        )

    # -----------------------------------------------------------------------
    # Scenario 2 — deadlock-freedom at the PEAK of sustained saturation
    #
    # Drains an in-flight heavy via ``_Driver.complete()`` while PSI stays
    # saturated throughout, proving the hold predicate
    # (saturated AND in_flight>=floor) makes "held with < floor in flight"
    # unreachable by construction.
    # -----------------------------------------------------------------------

    async def test_floor_holds_under_sustained_saturation(self, tmp_path) -> None:
        """Scenario 2: deadlock-freedom at the PEAK of sustained saturation.

        PSI is saturated for the ENTIRE scenario (never flipped back to
        idle).  With min_inflight_floor=2, completing ONE of the 2 in-flight
        heavies drops in-flight below the floor -- psi_hold=(saturated AND
        in_flight>=floor) makes "hold with < floor in flight" unreachable by
        construction, so a fresh heavy is re-admitted immediately even though
        PSI itself never recovered.  Repeated twice to show it is not a
        one-off.  6 heavies leaves 2 untouched in the backlog so the final
        deterministic dispatch also coincides with a live heavy deferral.
        """
        event_store = EventStore(db_path=tmp_path / 'psi.db', run_id='da4-s2')
        clock = _FakeClock()
        config = OrchestratorConfig(
            max_per_module=1,
            psi_admission=cpu_arm_cfg(min_inflight_floor=2),
        )

        tasks = [
            _heavy_task(f'H{i}', priority='high', files=[f'src/h{i}.py'])
            for i in range(1, 7)
        ]
        tasks.append(_det_task('D1', files=['src/d1.py']))

        feed = _PsiFeed()
        feed.set_saturated(cpu_some10=90.0)  # saturated from tick 1, never recovers
        driver = _Driver.build(
            config, tasks, event_store=event_store, time_source=clock, psi_feed=feed,
        )

        # Ramp-up: in_flight starts at 0 (< floor), so the anti-deadlock floor
        # lets dispatch proceed normally even though PSI is already saturated.
        await driver.run_ticks(2)
        assert driver.heavy_in_flight() == 2, 'floor reached: 2 heavies in flight'

        # --- First drain -> re-admit cycle ---
        drained_1 = next(iter(driver._heavy_in_flight))
        await driver.complete(drained_1)
        assert driver.heavy_in_flight() == 1, (
            'draining one heavy must drop in-flight below the floor'
        )

        await driver.tick()
        assert driver.heavy_in_flight() == 2, (
            'a fresh heavy must be RE-ADMITTED even though PSI never recovered -- '
            'hold with < floor in flight is unreachable by construction'
        )

        # --- Second drain -> re-admit cycle (not a one-off) ---
        drained_2 = next(iter(driver._heavy_in_flight))
        await driver.complete(drained_2)
        assert driver.heavy_in_flight() == 1

        await driver.tick()
        assert driver.heavy_in_flight() == 2, 'the re-admit behaviour repeats on a second drain'

        # --- Deterministic dispatch is never blocked by sustained saturation ---
        await driver.run_ticks(2)
        assert 'D1' in driver.dispatched_ids(), (
            'the deterministic task must dispatch at some point despite sustained saturation'
        )
        assert driver.heavy_in_flight() >= config.psi_admission.min_inflight_floor, (
            'in-flight heavy must never be left below the floor while heavy work '
            'and a free slot remain'
        )

    # -----------------------------------------------------------------------
    # Scenario 3 — work-conserving restore up to max_concurrent_tasks
    #
    # Recovers via ``_Driver.run_until_quiescent()`` (fills heavy in-flight up
    # to the max_concurrent_tasks cap, since acquire_next itself enforces no
    # such cap) and partitions the event stream with
    # ``deferred_events_since()`` to prove no dispatch_deferred event occurs
    # after the recovery flip (no residual hold).
    # -----------------------------------------------------------------------

    async def test_recovery_restores_full_dispatch_up_to_cap(self, tmp_path) -> None:
        """Scenario 3: work-conserving restore up to max_concurrent_tasks,
        with NO residual PSI hold after recovery.

        6 heavies (cap=4) + 1 deterministic.  1 heavy dispatches while idle,
        then PSI flips to saturated (in-flight=1 >= default floor=1) so the
        remaining heavies are held while D1 still dispatches.  On recovery,
        run_until_quiescent fills heavy in-flight up to the max_concurrent_
        tasks cap (harness-modeled -- acquire_next itself does not enforce
        it) -- the 2 leftover heavies wait on the CAP, not on PSI.  No
        dispatch_deferred event may occur after the recovery flip.
        """
        event_store = EventStore(db_path=tmp_path / 'psi.db', run_id='da4-s3')
        clock = _FakeClock()
        config = OrchestratorConfig(
            max_per_module=1, max_concurrent_tasks=4, psi_admission=cpu_arm_cfg(),
        )

        heavy_tasks = [
            _heavy_task(f'H{i}', priority='high', files=[f'src/h{i}.py'])
            for i in range(1, 7)
        ]
        heavy_ids = {t['id'] for t in heavy_tasks}
        tasks = [*heavy_tasks, _det_task('D1', files=['src/d1.py'])]

        feed = _PsiFeed()  # starts idle
        driver = _Driver.build(
            config, tasks, event_store=event_store, time_source=clock, psi_feed=feed,
        )

        # 1 heavy already in flight before saturation begins.
        await driver.tick()
        assert driver.heavy_in_flight() == 1

        # --- Saturated phase: new heavy dispatch held, in-flight stays 1 ---
        feed.set_saturated(cpu_some10=90.0)
        await driver.run_ticks(3)

        assert driver.heavy_in_flight() == 1, 'saturation must hold all further heavy dispatch'
        assert 'D1' in driver.dispatched_ids(), (
            'the deterministic task must still dispatch while heavy dispatch is held'
        )
        deferred_during_saturation = driver.deferred_events()
        assert len(deferred_during_saturation) >= 1, (
            'saturation must defer at least one heavy dispatch'
        )
        high_water_id = deferred_during_saturation[-1]['id']

        # --- Recovery: fill up to the max_concurrent_tasks cap, no more ---
        feed.set_idle()
        await driver.run_until_quiescent(max_in_flight=config.max_concurrent_tasks)

        assert driver.heavy_in_flight() == 4, (
            'recovery must be work-conserving up to the lane cap'
        )
        still_waiting = heavy_ids - driver.dispatched_ids()
        assert len(still_waiting) == 2, (
            f'the 2 leftover heavies must wait on the CAP, not on PSI: {still_waiting!r}'
        )

        # --- No residual hold: zero dispatch_deferred events after recovery ---
        assert driver.deferred_events_since(high_water_id) == [], (
            'no dispatch_deferred event may occur after the recovery flip'
        )

    # -----------------------------------------------------------------------
    # Scenario 4 — PRD plans/load-throttle-harmonisation-prd.md boundary row 6
    # -----------------------------------------------------------------------

    async def test_runqueue_transition_through_the_real_reader(
        self, tmp_path, monkeypatch
    ) -> None:
        """Scenario 4: idle -> runqueue-saturated -> idle, with no injected
        feed: the scheduler reads through the REAL reader over a fixture tree.

        The own-cgroup reading resolves to ``df-x.slice`` only if the
        scheduler handed the reader its ``fused_memory.project_id``.
        """
        tree = _LoadFixtureTree(tmp_path / 'host')
        # Before construction: the scheduler binds the module-level reader.
        monkeypatch.setattr('orchestrator.scheduler.read_psi_sample', tree.reader())
        event_store = EventStore(db_path=tmp_path / 'psi.db', run_id='row-6')
        config = OrchestratorConfig(
            max_per_module=1,
            max_concurrent_tasks=6,
            psi_admission=PsiAdmissionConfig(runqueue_ratio=4.0),
            fused_memory=FusedMemoryConfig(project_id='x'),
        )
        heavy_ids = {'H1', 'H2', 'H3', 'H4'}
        tasks = [
            *(_heavy_task(t, priority='high', files=[f'src/{t}.py']) for t in sorted(heavy_ids)),
            _det_task('D1', files=['src/d1.py']),
        ]
        driver = _Driver.build(config, tasks, event_store=event_store, time_source=_FakeClock())

        # --- (A) idle ---
        await driver.run_ticks(2)
        assert driver.heavy_in_flight() == 2
        assert driver.deferred_events() == []

        # --- (B) runqueue-saturated: far above 4.0 for any affinity mask ---
        tree.set_procs_running(1_000_000)
        await driver.run_ticks(4)

        assert driver.heavy_in_flight() == 2
        assert 'D1' in driver.dispatched_ids()
        deferred = driver.deferred_events()
        assert deferred
        assert all(e['task_id'] in heavy_ids for e in deferred)
        first = deferred[0]['data']
        assert first['metric'] == 'runqueue_ratio'
        sample = first['sample']
        assert sample['read_ok'] is True
        assert sample['runqueue_read_ok'] is True
        assert sample['runqueue_ratio'] == pytest.approx(
            1_000_000 / len(os.sched_getaffinity(0))
        )
        assert sample['own_read_ok'] is True
        assert sample['own_cgroup'].endswith('/df-x.slice')
        assert sample['own_cpu_some10'] == 12.5

        # --- (C) idle again: full dispatch restored, no new deferral ---
        tree.set_procs_running(0)
        await driver.run_ticks(2)

        assert driver.heavy_in_flight() == 4
        assert driver.deferred_events() == deferred
