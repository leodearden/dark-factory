"""Tests for the scheduler's dispatch-admission gate (task 2328, DA3 of PRD
docs/prds/dispatch-admission-load-cap.md).

Covers:
  step-1  (RED)   Throttle direction (scored loop) + event payload: a HEAVY
                  (normal-kind) candidate is deferred under PSI saturation
                  while a DETERMINISTIC candidate scored lower in the same
                  tick still dispatches; exactly one ``dispatch_deferred``
                  event fires with the ``{metric, in_flight, floor, sample}``
                  payload (task 3590: ``sample`` is the whole PsiSample).
  step-3  (GUARD) Work-conserving (idle PSI never throttles) + floor/deadlock
                  -freedom (0 in-flight, or in-flight below a configured
                  floor, always dispatches) + the reported metric's rank
                  (PRD plans/load-throttle-harmonisation-prd.md D10).
  step-4  (GUARD) Disabled gate (no PSI read at all), fail-open on
                  unreadable PSI (DA-D6), once-per-tick event dedup, and
                  rate-limited deferral logging.
  step-5  (RED)   Pinned-loop parity (DA-D5) — a pinned HEAVY candidate is
                  deferred the same as a scored one; a pinned DETERMINISTIC
                  candidate remains exempt.

Mirrors test_scheduler_landed_dispatch_gate.py / test_scheduler_deterministic.py
conventions: module-local task-dict helpers, ``scheduler.get_tasks =
AsyncMock(return_value=[...])``, driving ``acquire_next()`` directly,
``_RecordingEventStore`` for event assertions, and ``OverrideStore`` +
``tmp_path`` for pin parity.
"""

from __future__ import annotations

from dataclasses import asdict
from unittest.mock import AsyncMock, Mock

import pytest
from _psi_admission_yaml import write_config_with_nondefault_psi_admission
from _recording_event_store import _RecordingEventStore
from shared.psi import PsiSample

from orchestrator.config import OrchestratorConfig, PsiAdmissionConfig
from orchestrator.scheduler import Scheduler


@pytest.fixture(autouse=True)
def _nondefault_psi_admission_yaml(monkeypatch, tmp_path) -> None:
    """Run every test here against a yaml whose psi_admission differs from code."""
    monkeypatch.setenv(
        'ORCH_CONFIG_PATH', str(write_config_with_nondefault_psi_admission(tmp_path))
    )

# ---------------------------------------------------------------------------
# Minimal task-dict + PSI-sample helpers
# ---------------------------------------------------------------------------


def _heavy_task(
    task_id: str,
    priority: str = 'medium',
    files: list[str] | None = None,
) -> dict:
    """Minimal normal-kind (heavy) pending-task dict carrying every field
    acquire_next reads."""
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
    runqueue_ratio: float = 0.0,
    runqueue_read_ok: bool = False,
    own_cpu_some10: float = 0.0,
    own_cgroup: str = '',
    own_read_ok: bool = False,
    cpu_some60: float = 0.0,
) -> PsiSample:
    """Build a PsiSample; all metrics idle (0.0) unless overridden."""
    return PsiSample(
        cpu_some10=cpu_some10,
        mem_some10=mem_some10,
        mem_full10=mem_full10,
        io_some10=io_some10,
        read_ok=read_ok,
        runqueue_ratio=runqueue_ratio,
        runqueue_read_ok=runqueue_read_ok,
        own_cpu_some10=own_cpu_some10,
        own_cgroup=own_cgroup,
        own_read_ok=own_read_ok,
        cpu_some60=cpu_some60,
    )


def _psi_cfg(**overrides) -> PsiAdmissionConfig:
    """A fresh gate config whose CPU arm, which these tests drive, is explicit.

    Fresh per call, never a shared instance: apply_reload mutates submodels in
    place.
    """
    return PsiAdmissionConfig(cpu_some_avg10=85.0, **overrides)


async def _sole_deferral_data(sample: PsiSample, psi_cfg: PsiAdmissionConfig) -> dict:
    """Hold one heavy candidate on *sample*; return its dispatch_deferred data."""
    event_store = _RecordingEventStore()
    scheduler = Scheduler(
        OrchestratorConfig(max_per_module=1, psi_admission=psi_cfg),
        event_store=event_store,  # type: ignore[arg-type]
    )
    scheduler.finish_startup()
    scheduler._read_psi_sample = lambda: sample
    scheduler._dispatched = {'inflight1'}
    scheduler.get_tasks = AsyncMock(return_value=[_heavy_task('H')])

    assert await scheduler.acquire_next() is None
    deferred_events = [e for e in event_store.events if e[0] == 'dispatch_deferred']
    assert len(deferred_events) == 1
    return deferred_events[0][1]['data']


# ---------------------------------------------------------------------------
# step-1 — Throttle direction (scored loop) + event payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDispatchAdmissionThrottleDirection:
    """A saturated host defers a HEAVY candidate but still dispatches a
    DETERMINISTIC candidate scored lower in the same tick."""

    async def test_heavy_deferred_deterministic_dispatches_same_tick(self) -> None:
        event_store = _RecordingEventStore()
        scheduler = Scheduler(
            OrchestratorConfig(max_per_module=1, psi_admission=_psi_cfg()),
            event_store=event_store,  # type: ignore[arg-type]
        )
        scheduler.finish_startup()
        # cpu_some10=90 is OVER this file's explicit cpu_some_avg10=85.0 threshold.
        sample = _psi(cpu_some10=90.0, read_ok=True)
        scheduler._read_psi_sample = lambda: sample
        # >= default min_inflight_floor=1, so the floor does not suppress the hold.
        scheduler._dispatched = {'inflight1'}

        # H must outscore D so the scored loop tries H first (skips it, then
        # dispatches D) — otherwise D would dispatch and return before H is
        # ever evaluated, and no dispatch_deferred event would fire.
        task_h = _heavy_task('H', priority='high')
        task_d = _det_task('D', priority='medium')
        scheduler.get_tasks = AsyncMock(return_value=[task_h, task_d])

        result = await scheduler.acquire_next()

        assert result is not None
        assert result.task_id == 'D', 'deterministic candidate must still dispatch this tick'
        assert 'H' not in scheduler._dispatched, 'heavy candidate must be deferred under saturation'
        assert 'D' in scheduler._dispatched

        deferred_events = [e for e in event_store.events if e[0] == 'dispatch_deferred']
        assert len(deferred_events) == 1, 'exactly one dispatch_deferred event this tick'
        assert deferred_events[0][1]['data'] == {
            'metric': 'cpu_some_avg10',
            'in_flight': 1,
            'floor': 1,
            'sample': asdict(sample),
        }

    async def test_payload_carries_every_component_of_the_sample(self) -> None:
        sample = _psi(
            mem_full10=5.0,
            runqueue_ratio=2.0,
            runqueue_read_ok=True,
            own_cpu_some10=12.5,
            own_cgroup='/user.slice/user-1000.slice/user@1000.service/df.slice/df-x.slice',
            own_read_ok=True,
            cpu_some60=33.0,
        )

        data = await _sole_deferral_data(sample, _psi_cfg())

        assert data['metric'] == 'mem_full_avg10'
        assert data['sample'] == asdict(sample)


# ---------------------------------------------------------------------------
# step-3 — Work-conserving + floor/deadlock-freedom + metric ranking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDispatchAdmissionWorkConserving:
    """An idle host never throttles — ALL heavy candidates dispatch across
    successive ticks, with zero dispatch_deferred events, even when
    in-flight is already at/above the floor."""

    async def test_idle_psi_dispatches_all_heavy_candidates(self) -> None:
        event_store = _RecordingEventStore()
        scheduler = Scheduler(
            OrchestratorConfig(max_per_module=1, psi_admission=_psi_cfg()),
            event_store=event_store,  # type: ignore[arg-type]
        )
        scheduler.finish_startup()
        scheduler._read_psi_sample = lambda: _psi(read_ok=True)  # all metrics 0.0 — idle
        scheduler._dispatched = {'inflight1'}  # already >= default floor=1

        tasks = [
            _heavy_task('H1', files=['src/h1.py']),
            _heavy_task('H2', files=['src/h2.py']),
            _heavy_task('H3', files=['src/h3.py']),
        ]
        scheduler.get_tasks = AsyncMock(return_value=tasks)

        dispatched_ids = []
        for _ in range(3):
            result = await scheduler.acquire_next()
            assert result is not None
            dispatched_ids.append(result.task_id)

        assert set(dispatched_ids) == {'H1', 'H2', 'H3'}
        assert {'H1', 'H2', 'H3'} <= scheduler._dispatched
        assert [e for e in event_store.events if e[0] == 'dispatch_deferred'] == []


@pytest.mark.asyncio
class TestDispatchAdmissionFloorDeadlockFreedom:
    """The anti-deadlock floor: a hold can never engage with fewer than
    ``min_inflight_floor`` tasks in flight on this orchestrator — "hold with
    nothing running" is unreachable by construction."""

    async def test_zero_in_flight_still_dispatches_under_saturation(self) -> None:
        event_store = _RecordingEventStore()
        scheduler = Scheduler(
            OrchestratorConfig(max_per_module=1, psi_admission=_psi_cfg()),
            event_store=event_store,  # type: ignore[arg-type]
        )
        scheduler.finish_startup()
        scheduler._read_psi_sample = lambda: _psi(cpu_some10=90.0, read_ok=True)
        # scheduler._dispatched left empty — in_flight=0 < default floor=1.

        task_h = _heavy_task('H')
        scheduler.get_tasks = AsyncMock(return_value=[task_h])

        result = await scheduler.acquire_next()

        assert result is not None
        assert result.task_id == 'H'
        assert 'H' in scheduler._dispatched
        assert [e for e in event_store.events if e[0] == 'dispatch_deferred'] == []

    async def test_in_flight_below_configured_floor_still_dispatches(self) -> None:
        event_store = _RecordingEventStore()
        config = OrchestratorConfig(
            max_per_module=1,
            psi_admission=_psi_cfg(min_inflight_floor=2),
        )
        scheduler = Scheduler(config, event_store=event_store)  # type: ignore[arg-type]
        scheduler.finish_startup()
        scheduler._read_psi_sample = lambda: _psi(cpu_some10=90.0, read_ok=True)
        scheduler._dispatched = {'inflight1'}  # 1 < configured floor=2

        task_h = _heavy_task('H')
        scheduler.get_tasks = AsyncMock(return_value=[task_h])

        result = await scheduler.acquire_next()

        assert result is not None
        assert result.task_id == 'H'
        assert 'H' in scheduler._dispatched
        assert [e for e in event_store.events if e[0] == 'dispatch_deferred'] == []


@pytest.mark.asyncio
class TestDispatchAdmissionMetricRanking:
    """The metric reported in the dispatch_deferred event is the highest-ranked
    tripping arm, in PRD plans/load-throttle-harmonisation-prd.md D10 rank:
    mem_full_avg10 > runqueue_ratio > own_cpu_some_avg10 > mem_some_avg10 >
    io_some_avg10 > cpu_some_avg10 (the legacy CPU arm ranks last)."""

    async def test_cpu_and_io_both_saturated_reports_io(self) -> None:
        data = await _sole_deferral_data(_psi(cpu_some10=90.0, io_some10=50.0), _psi_cfg())

        assert data['metric'] == 'io_some_avg10'

    async def test_io_only_saturated_reports_io(self) -> None:
        data = await _sole_deferral_data(_psi(io_some10=50.0), _psi_cfg())

        assert data['metric'] == 'io_some_avg10'

    async def test_mem_some_outranks_io(self) -> None:
        data = await _sole_deferral_data(_psi(mem_some10=20.0, io_some10=50.0), _psi_cfg())

        assert data['metric'] == 'mem_some_avg10'

    async def test_mem_full_outranks_every_other_arm(self) -> None:
        sample = _psi(cpu_some10=90.0, mem_some10=99.0, mem_full10=50.0, io_some10=99.0)

        data = await _sole_deferral_data(sample, _psi_cfg())

        assert data['metric'] == 'mem_full_avg10'


@pytest.mark.asyncio
class TestDispatchAdmissionV2Arms:
    """The v2 vocabulary through the gate, PRD
    plans/load-throttle-harmonisation-prd.md boundary rows 4, 5 and 7."""

    async def test_default_config_ignores_host_cpu_runqueue_and_own_pressure(self) -> None:
        """Row 4: the legacy CPU arm and both v2 arms are off by default."""
        event_store = _RecordingEventStore()
        scheduler = Scheduler(
            OrchestratorConfig(max_per_module=1, psi_admission=PsiAdmissionConfig()),
            event_store=event_store,  # type: ignore[arg-type]
        )
        scheduler.finish_startup()
        scheduler._read_psi_sample = lambda: _psi(
            cpu_some10=99.0,
            runqueue_ratio=9.0,
            runqueue_read_ok=True,
            own_cpu_some10=99.0,
            own_read_ok=True,
        )
        scheduler._dispatched = {'inflight1'}
        scheduler.get_tasks = AsyncMock(return_value=[_heavy_task('H')])

        result = await scheduler.acquire_next()

        assert result is not None
        assert result.task_id == 'H'
        assert [e for e in event_store.events if e[0] == 'dispatch_deferred'] == []

    async def test_runqueue_arm_holds_and_outranks_mem_some(self) -> None:
        """Row 5: a runqueue ratio over its threshold holds."""
        sample = _psi(mem_some10=20.0, runqueue_ratio=4.3, runqueue_read_ok=True)

        data = await _sole_deferral_data(sample, PsiAdmissionConfig(runqueue_ratio=4.0))

        assert data['metric'] == 'runqueue_ratio'

    async def test_mem_full_outranks_the_runqueue_arm(self) -> None:
        sample = _psi(
            mem_some10=20.0, mem_full10=5.0, runqueue_ratio=4.3, runqueue_read_ok=True,
        )

        data = await _sole_deferral_data(sample, PsiAdmissionConfig(runqueue_ratio=4.0))

        assert data['metric'] == 'mem_full_avg10'

    async def test_own_cgroup_arm_holds_on_an_idle_host(self) -> None:
        """Row 7: the orchestrator's own cgroup pressure holds by itself."""
        own_cgroup = '/user.slice/user-1000.slice/user@1000.service/df.slice/df-x.slice'
        sample = _psi(own_cpu_some10=55.0, own_read_ok=True, own_cgroup=own_cgroup)

        data = await _sole_deferral_data(sample, PsiAdmissionConfig(own_cpu_some_avg10=50.0))

        assert data['metric'] == 'own_cpu_some_avg10'
        assert data['sample']['own_cgroup'] == own_cgroup


# ---------------------------------------------------------------------------
# step-4 — Robustness: disabled gate, fail-open, once-per-tick/rate-limit
# ---------------------------------------------------------------------------


class _FakeClock:
    """Minimal controllable monotonic-style clock for rate-limit tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, secs: float) -> None:
        self._t += secs


@pytest.mark.asyncio
class TestDispatchAdmissionDisabledGate:
    """psi_admission.enabled=False short-circuits the gate entirely — no PSI
    read (no per-tick reader cost), no hold, heavy dispatch proceeds exactly
    as it did before DA3."""

    async def test_disabled_gate_never_reads_psi_and_dispatches_normally(self) -> None:
        event_store = _RecordingEventStore()
        config = OrchestratorConfig(
            max_per_module=1,
            psi_admission=_psi_cfg(enabled=False),
        )
        scheduler = Scheduler(config, event_store=event_store)  # type: ignore[arg-type]
        scheduler.finish_startup()
        reader = Mock(return_value=_psi(cpu_some10=99.0, read_ok=True))
        scheduler._read_psi_sample = reader
        scheduler._dispatched = {'inflight1'}

        task_h = _heavy_task('H')
        scheduler.get_tasks = AsyncMock(return_value=[task_h])

        result = await scheduler.acquire_next()

        assert result is not None
        assert result.task_id == 'H'
        assert 'H' in scheduler._dispatched
        reader.assert_not_called()
        assert [e for e in event_store.events if e[0] == 'dispatch_deferred'] == []


@pytest.mark.asyncio
class TestDispatchAdmissionFailOpen:
    """DA-D6: an unreadable PSI sample must never wedge dispatch — the heavy
    candidate dispatches normally, with a loud WARNING instead of a hold."""

    async def test_unreadable_psi_dispatches_and_logs_warning(self, caplog) -> None:
        caplog.set_level('WARNING')
        event_store = _RecordingEventStore()
        scheduler = Scheduler(
            OrchestratorConfig(max_per_module=1, psi_admission=_psi_cfg()),
            event_store=event_store,  # type: ignore[arg-type]
        )
        scheduler.finish_startup()
        scheduler._read_psi_sample = lambda: _psi(cpu_some10=99.0, read_ok=False)
        scheduler._dispatched = {'inflight1'}

        task_h = _heavy_task('H')
        scheduler.get_tasks = AsyncMock(return_value=[task_h])

        result = await scheduler.acquire_next()

        assert result is not None, 'saturated()==False when read_ok=False — must not hold'
        assert result.task_id == 'H'
        assert 'H' in scheduler._dispatched
        assert [e for e in event_store.events if e[0] == 'dispatch_deferred'] == []
        assert any(record.levelname == 'WARNING' for record in caplog.records), (
            'an unreadable PSI sample should be logged, not silently swallowed'
        )


@pytest.mark.asyncio
class TestDispatchAdmissionOnceDedupAndRateLimit:
    """Exactly one dispatch_deferred event per tick's FIRST heavy skip, and
    the deferral WARNING log stream is separately rate-limited across ticks
    (the event stream is not — a dashboard needs one signal per tick)."""

    async def test_single_tick_multiple_heavy_skips_emit_one_event(self) -> None:
        event_store = _RecordingEventStore()
        scheduler = Scheduler(
            OrchestratorConfig(max_per_module=1, psi_admission=_psi_cfg()),
            event_store=event_store,  # type: ignore[arg-type]
        )
        scheduler.finish_startup()
        scheduler._read_psi_sample = lambda: _psi(cpu_some10=90.0, read_ok=True)
        scheduler._dispatched = {'inflight1'}

        tasks = [_heavy_task('H1', priority='high'), _heavy_task('H2', priority='medium')]
        scheduler.get_tasks = AsyncMock(return_value=tasks)

        result = await scheduler.acquire_next()

        assert result is None, 'both candidates are heavy and held — nothing dispatches this tick'
        assert 'H1' not in scheduler._dispatched
        assert 'H2' not in scheduler._dispatched
        deferred_events = [e for e in event_store.events if e[0] == 'dispatch_deferred']
        assert len(deferred_events) == 1, 'only the FIRST heavy skip emits an event this tick'

    async def test_two_saturated_ticks_within_window_emit_two_events_one_warning(
        self, caplog
    ) -> None:
        caplog.set_level('WARNING')
        event_store = _RecordingEventStore()
        clock = _FakeClock()
        scheduler = Scheduler(
            OrchestratorConfig(max_per_module=1, psi_admission=_psi_cfg()),
            event_store=event_store,  # type: ignore[arg-type]
            time_source=clock,
        )
        scheduler.finish_startup()
        scheduler._read_psi_sample = lambda: _psi(cpu_some10=90.0, read_ok=True)
        scheduler._dispatched = {'inflight1'}

        task_h = _heavy_task('H')
        scheduler.get_tasks = AsyncMock(return_value=[task_h])

        result1 = await scheduler.acquire_next()
        assert result1 is None
        clock.advance(10.0)  # well within the 180s rate-limit window
        result2 = await scheduler.acquire_next()
        assert result2 is None

        deferred_events = [e for e in event_store.events if e[0] == 'dispatch_deferred']
        assert len(deferred_events) == 2, 'one event per tick regardless of the log rate-limit'

        deferral_warnings = [
            r for r in caplog.records
            if r.levelname == 'WARNING' and 'deferring heavy' in r.getMessage()
        ]
        assert len(deferral_warnings) == 1, 'the WARNING log stream is rate-limited across ticks'
        assert 'cpu_some_avg10' in deferral_warnings[0].getMessage()


# ---------------------------------------------------------------------------
# step-5 — Pinned-loop parity (DA-D5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDispatchAdmissionPinnedLoopParity:
    """A pin doesn't reduce host load — the pinned loop must honor the same
    psi_hold decision as the scored loop (DA-D5).  Deterministic pinned
    candidates remain exempt."""

    async def test_pinned_heavy_deferred_nonpinned_deterministic_dispatches(
        self, tmp_path
    ) -> None:
        from orchestrator.overrides import OverrideStore

        event_store = _RecordingEventStore()
        config = OrchestratorConfig(max_per_module=1, psi_admission=_psi_cfg())
        store = OverrideStore(tmp_path / 'o.db')
        store.set_override('/proj', 'Z', pinned=True)

        scheduler = Scheduler(config, event_store=event_store, override_store=store)  # type: ignore[arg-type]
        scheduler.finish_startup()
        scheduler._project_root = '/proj'
        scheduler._read_psi_sample = lambda: _psi(cpu_some10=90.0, read_ok=True)
        scheduler._dispatched = {'inflight1'}

        task_z = _heavy_task('Z', files=['z/src.py'])
        task_y = _det_task('Y', files=['y/src.py'])
        scheduler.get_tasks = AsyncMock(return_value=[task_z, task_y])

        result = await scheduler.acquire_next()

        assert result is not None
        assert result.task_id == 'Y', (
            'Y (deterministic, unpinned) should dispatch; Z (pinned heavy) must not'
        )
        assert 'Z' not in scheduler._dispatched
        assert 'Y' in scheduler._dispatched
        deferred_events = [e for e in event_store.events if e[0] == 'dispatch_deferred']
        assert len(deferred_events) == 1

    async def test_pinned_deterministic_dispatches_under_saturation(self, tmp_path) -> None:
        from orchestrator.overrides import OverrideStore

        event_store = _RecordingEventStore()
        config = OrchestratorConfig(max_per_module=1, psi_admission=_psi_cfg())
        store = OverrideStore(tmp_path / 'o.db')
        store.set_override('/proj', 'W', pinned=True)

        scheduler = Scheduler(config, event_store=event_store, override_store=store)  # type: ignore[arg-type]
        scheduler.finish_startup()
        scheduler._project_root = '/proj'
        scheduler._read_psi_sample = lambda: _psi(cpu_some10=90.0, read_ok=True)
        scheduler._dispatched = {'inflight1'}

        task_w = _det_task('W', files=['w/src.py'])
        scheduler.get_tasks = AsyncMock(return_value=[task_w])

        result = await scheduler.acquire_next()

        assert result is not None
        assert result.task_id == 'W', 'deterministic exemption applies in the pin loop too'
        assert 'W' in scheduler._dispatched
