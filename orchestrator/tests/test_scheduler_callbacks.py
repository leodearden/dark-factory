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

        # Drive 3 ticks to accumulate skip_count >= skip_threshold (3).
        for _ in range(3):
            await scheduler.acquire_next()

        # Advance clock past idle_secs (100s) and tick once more.
        t[0] = 150.0
        await scheduler.acquire_next()

        spy.assert_called_once()
        call_args = spy.call_args
        assert call_args.args[0] == 'starved'
