"""Tests for ``SchedulerCallbacks`` — the constructor-injected Harness↔Scheduler
callback seam (task 2235, W10-α).

Replaces the nine post-construction ``scheduler._on_*`` monkey-patches with a
single frozen dataclass passed to ``Scheduler.__init__``.

Covers:
  (a) ``SchedulerCallbacks`` shape — frozen dataclass, 9 fields, all default
      ``None``.
  (b) ``Scheduler(config, callbacks=...)`` construction, with and without an
      explicit ``callbacks=`` argument (omitting it must collapse to an
      all-``None`` ``SchedulerCallbacks`` so bare-Scheduler unit tests are
      unaffected).
  (c) Constructor-injected hooks actually FIRE on their triggers.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from unittest.mock import AsyncMock, Mock

import pytest

from orchestrator.config import OrchestratorConfig, StarvationWatchdogConfig
from orchestrator.scheduler import Scheduler, SchedulerCallbacks


class TestSchedulerCallbacksDataclass:
    """(a) SchedulerCallbacks: frozen dataclass, 9 fields, all default None."""

    def test_all_nine_fields_default_none(self):
        callbacks = SchedulerCallbacks()
        assert callbacks.on_park_stop_trip is None
        assert callbacks.on_external_dep_block is None
        assert callbacks.on_starvation_warn is None
        assert callbacks.on_starvation_resolve is None
        assert callbacks.warm_base_health_probe is None
        assert callbacks.on_warm_base_warn is None
        assert callbacks.on_warm_base_promote_l2 is None
        assert callbacks.on_warm_base_resolve is None
        assert callbacks.suppress_blocked_write is None

    def test_field_count_is_exactly_nine(self):
        field_names = {f.name for f in dataclasses.fields(SchedulerCallbacks)}
        assert field_names == {
            'on_park_stop_trip',
            'on_external_dep_block',
            'on_starvation_warn',
            'on_starvation_resolve',
            'warm_base_health_probe',
            'on_warm_base_warn',
            'on_warm_base_promote_l2',
            'on_warm_base_resolve',
            'suppress_blocked_write',
        }

    def test_is_frozen(self):
        callbacks = SchedulerCallbacks()
        with pytest.raises(dataclasses.FrozenInstanceError):
            callbacks.on_park_stop_trip = lambda reason: None


class TestSchedulerConstructorCallbacks:
    """(b) Scheduler(config, callbacks=...) construction with/without callbacks."""

    def test_constructs_with_callbacks(self):
        async def on_trip(reason: str) -> None:
            pass

        callbacks = SchedulerCallbacks(on_park_stop_trip=on_trip)
        scheduler = Scheduler(OrchestratorConfig(), callbacks=callbacks)
        assert scheduler is not None

    def test_constructs_without_callbacks_defaults_all_none(self):
        """Omitting callbacks= must not break bare-Scheduler unit tests."""
        scheduler = Scheduler(OrchestratorConfig())
        assert scheduler is not None


class TestConstructorInjectedHooksFire:
    """(c) Constructor-injected hooks fire on their triggers."""

    @pytest.mark.asyncio
    async def test_on_park_stop_trip_fires(self, monkeypatch):
        """A constructor-injected on_park_stop_trip fires at the trip threshold."""
        spy = AsyncMock()
        config = OrchestratorConfig(
            max_per_module=1,
            park_stop_parked_threshold=1,
            park_stop_parked_window_hours=1.0,
        )
        scheduler = Scheduler(
            config, callbacks=SchedulerCallbacks(on_park_stop_trip=spy)
        )

        mock = AsyncMock(return_value={})
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock)

        await scheduler.set_task_status('1', 'blocked')
        # Yield to the event loop so the ensure_future'd callback can execute.
        await asyncio.sleep(0)

        spy.assert_called_once()

    @pytest.mark.asyncio
    async def test_suppress_blocked_write_consulted(self):
        """A constructor-injected suppress_blocked_write short-circuits the write.

        No ``mcp_call`` monkeypatch is installed — if the suppression check
        were bypassed, ``dispatch_tool`` would attempt a real network call and
        this test would hang/error instead of completing.
        """
        suppress = Mock(return_value=True)
        config = OrchestratorConfig(max_per_module=1)
        scheduler = Scheduler(
            config, callbacks=SchedulerCallbacks(suppress_blocked_write=suppress)
        )

        await scheduler.set_task_status('T1', 'blocked')

        suppress.assert_called_once_with('T1')
        assert len(scheduler._blocked_transitions) == 0, (
            'Suppressed write must not record a blocked transition'
        )

    @pytest.mark.asyncio
    async def test_on_starvation_warn_fires(self):
        """A constructor-injected on_starvation_warn fires once both gates cross."""
        t: list[float] = [0.0]

        def fake_clock() -> float:
            return t[0]

        spy = AsyncMock()
        config = OrchestratorConfig(
            max_per_module=1,
            starvation_watchdog=StarvationWatchdogConfig(
                enabled=True, skip_threshold=3, idle_secs=100.0,
            ),
        )
        # Keep fairness parks from installing before the watchdog threshold
        # (mirrors the isolation trick in TestStarvationWatchdog).
        config.fairness.skip_threshold = 9999
        scheduler = Scheduler(
            config,
            time_source=fake_clock,
            callbacks=SchedulerCallbacks(on_starvation_warn=spy),
        )

        task = {
            'id': 'starved',
            'title': 'Starved task',
            'status': 'pending',
            'priority': 'medium',
            'dependencies': [],
            'metadata': {'files': ['backend/main.py']},
        }
        # Seed a held lock on a dispatched 'seed' task so 'starved' can never
        # acquire its module (loop-exhausted path -> skip_count bumped each tick).
        scheduler.lock_table.try_acquire('seed', ['backend'])
        scheduler._dispatched.add('seed')
        scheduler.get_tasks = AsyncMock(return_value=[task])
        scheduler.finish_startup()

        # Drive 3 ticks to accumulate skip_count >= skip_threshold (3).
        for _ in range(3):
            await scheduler.acquire_next()

        # Advance clock past idle_secs (100s) and tick once more.
        t[0] = 150.0
        await scheduler.acquire_next()

        spy.assert_called_once()
        call_args = spy.call_args
        assert call_args.args[0] == 'starved'


class TestLivenessAccessors:
    """Public liveness accessors (task 2235 step-3/4): ``is_dispatched(tid)``
    and ``requeue_history(tid)``.

    Replace the harness's direct reach-ins into ``scheduler._dispatched`` /
    ``scheduler._requeue_history`` with a stable public surface.
    """

    def test_is_dispatched_false_when_absent(self):
        scheduler = Scheduler(OrchestratorConfig())
        assert scheduler.is_dispatched('T1') is False

    def test_is_dispatched_true_when_in_dispatched_set(self):
        scheduler = Scheduler(OrchestratorConfig())
        scheduler._dispatched.add('T1')
        assert scheduler.is_dispatched('T1') is True

    def test_is_dispatched_false_after_discard(self):
        scheduler = Scheduler(OrchestratorConfig())
        scheduler._dispatched.add('T1')
        scheduler._dispatched.discard('T1')
        assert scheduler.is_dispatched('T1') is False

    def test_requeue_history_empty_list_when_absent(self):
        scheduler = Scheduler(OrchestratorConfig())
        assert scheduler.requeue_history('T1') == []

    def test_requeue_history_returns_recorded_entries(self):
        scheduler = Scheduler(OrchestratorConfig())
        scheduler.record_requeue(
            'T1',
            phase='impl',
            reason='boom',
            detail='detail text',
            run_id='run-1',
            cost_usd=0.5,
        )

        history = scheduler.requeue_history('T1')

        assert len(history) == 1
        record = history[0]
        assert record.attempt == 1
        assert record.phase == 'impl'
        assert record.reason == 'boom'
        assert record.detail == 'detail text'
        assert record.run_id == 'run-1'
        assert record.cost_usd == 0.5

    def test_requeue_history_mutating_returned_list_does_not_affect_internal_state(
        self,
    ):
        """``requeue_history`` must return a copy — mutating it must not leak
        into the scheduler's internal bookkeeping."""
        scheduler = Scheduler(OrchestratorConfig())
        scheduler.record_requeue(
            'T1',
            phase='impl',
            reason='boom',
            detail='detail text',
            run_id='run-1',
            cost_usd=0.0,
        )

        history = scheduler.requeue_history('T1')
        history.append('not-a-real-record')
        history.clear()

        assert len(scheduler.requeue_history('T1')) == 1


class TestWorkflowCancelGraceAndActivelyHeld:
    """(step-5) Cancel-grace stamp + ``is_actively_held`` liveness accessor.

    The cancel-grace stamp (``_workflow_cancel_at``), grace constant
    (``_RECONCILE_CANCEL_GRACE_S``) and predicate move from the Harness to
    the Scheduler (task 2235) — they sit beside ``_dispatched`` /
    ``lock_table._held``, whose single writer is the Scheduler.  Semantics
    mirror the harness's current ``_workflow_cancel_recent`` exactly.
    """

    def test_grace_constant_defaults_to_30(self):
        scheduler = Scheduler(OrchestratorConfig())
        assert scheduler._RECONCILE_CANCEL_GRACE_S == 30.0

    def test_workflow_cancel_recent_false_when_never_cancelled(self):
        scheduler = Scheduler(OrchestratorConfig())
        assert scheduler.workflow_cancel_recent('T1') is False

    def test_workflow_cancel_recent_true_within_grace_window(self):
        scheduler = Scheduler(OrchestratorConfig())
        scheduler.note_workflow_cancelled('T1')
        assert scheduler.workflow_cancel_recent('T1') is True

    def test_workflow_cancel_recent_false_after_grace_elapses(self):
        scheduler = Scheduler(OrchestratorConfig())
        # Tiny grace window so the test doesn't need to sleep 30s.
        scheduler._RECONCILE_CANCEL_GRACE_S = 0.01
        scheduler.note_workflow_cancelled('T1')
        time.sleep(0.05)
        assert scheduler.workflow_cancel_recent('T1') is False

    def test_clear_workflow_cancel_makes_it_false_immediately(self):
        scheduler = Scheduler(OrchestratorConfig())
        scheduler.note_workflow_cancelled('T1')
        assert scheduler.workflow_cancel_recent('T1') is True

        scheduler.clear_workflow_cancel('T1')

        assert scheduler.workflow_cancel_recent('T1') is False

    def test_is_actively_held_false_by_default(self):
        scheduler = Scheduler(OrchestratorConfig())
        assert scheduler.is_actively_held('T1') is False

    def test_is_actively_held_true_when_dispatched(self):
        scheduler = Scheduler(OrchestratorConfig())
        scheduler._dispatched.add('T1')
        assert scheduler.is_actively_held('T1') is True

    def test_is_actively_held_true_when_lock_held(self):
        scheduler = Scheduler(OrchestratorConfig())
        scheduler.lock_table.try_acquire('T1', ['backend'])
        assert scheduler.is_actively_held('T1') is True

    def test_is_actively_held_true_when_only_recent_cancel_stamp(self):
        scheduler = Scheduler(OrchestratorConfig())
        scheduler.note_workflow_cancelled('T1')
        assert scheduler.is_actively_held('T1') is True

    def test_is_actively_held_false_when_none_apply(self):
        scheduler = Scheduler(OrchestratorConfig())
        scheduler._dispatched.add('other-dispatched-task')
        scheduler.lock_table.try_acquire('other-held-task', ['backend'])
        scheduler.note_workflow_cancelled('other-cancelled-task')

        assert scheduler.is_actively_held('T1') is False


class TestStartupGate:
    """(step-7) ``acquire_next()`` must not run before ``finish_startup()``.

    Turns the comment-only "sweeps run before the first tick" invariant into
    a runtime check: a fresh Scheduler starts unstarted; ``acquire_next()``
    asserts ``started`` before doing any work; ``finish_startup()`` flips the
    latch and is idempotent.
    """

    def test_fresh_scheduler_is_not_started(self):
        scheduler = Scheduler(OrchestratorConfig())
        assert scheduler.started is False

    @pytest.mark.asyncio
    async def test_acquire_next_raises_before_finish_startup(self):
        scheduler = Scheduler(OrchestratorConfig())
        with pytest.raises(AssertionError):
            await scheduler.acquire_next()

    @pytest.mark.asyncio
    async def test_acquire_next_runs_normally_after_finish_startup(self):
        scheduler = Scheduler(OrchestratorConfig())
        scheduler.get_tasks = AsyncMock(return_value=[])

        scheduler.finish_startup()

        assert scheduler.started is True
        result = await scheduler.acquire_next()
        assert result is None

    def test_finish_startup_is_idempotent(self):
        scheduler = Scheduler(OrchestratorConfig())
        scheduler.finish_startup()
        scheduler.finish_startup()
        assert scheduler.started is True
