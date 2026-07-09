"""Scheduler warm-base hard-down watchdog (task 2061).

Mirrors ``TestStarvationWatchdog`` (test_scheduler.py) in shape: a per-tick
watchdog method driven directly, an injected async probe + three injected
async callbacks, and a mutable-list fake clock.  Structurally different in
one respect — this is a SINGLETON latch (one host-scoped base, not a
per-task dict/set) that additionally HALTS dispatch (``acquire_next``
returns ``None``) while engaged, mirroring the ``_paused`` park-stop
short-circuit.

RED: ``Scheduler._apply_warm_base_hard_down_watchdog`` and the
``_warm_base_hard_down`` / ``_warm_base_hard_down_since`` /
``_warm_base_l2_promoted`` / ``_warm_base_health_probe`` /
``_on_warm_base_warn`` / ``_on_warm_base_promote_l2`` / ``_on_warm_base_resolve``
attributes do not exist yet — every test below fails with AttributeError.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import AsyncMock

import pytest

from orchestrator.config import OrchestratorConfig, WarmBaseHardDownConfig
from orchestrator.scheduler import Scheduler


class TestWarmBaseHardDownWatchdog:
    """Tests for Scheduler._apply_warm_base_hard_down_watchdog + the acquire_next halt."""

    def _make_scheduler(
        self,
        *,
        enabled: bool = True,
        l2_window_secs: float = 100.0,
    ) -> tuple[Scheduler, list]:
        """Build a Scheduler with the watchdog config tuned small and a mutable clock."""
        t: list[float] = [0.0]

        def fake_clock() -> float:
            return t[0]

        config = OrchestratorConfig(
            max_per_module=1,
            warm_base_hard_down=WarmBaseHardDownConfig(
                enabled=enabled,
                l2_window_secs=l2_window_secs,
            ),
        )
        scheduler = Scheduler(config, time_source=fake_clock)
        scheduler.finish_startup()
        return scheduler, t

    def _install_callbacks(
        self, scheduler: Scheduler, *, probe_result: str = 'absent',
    ) -> tuple[AsyncMock, AsyncMock, AsyncMock, AsyncMock]:
        probe = AsyncMock(return_value=probe_result)
        warn = AsyncMock()
        promote = AsyncMock()
        resolve = AsyncMock()
        scheduler._callbacks = dataclasses.replace(scheduler._callbacks, warm_base_health_probe=probe)
        scheduler._callbacks = dataclasses.replace(scheduler._callbacks, on_warm_base_warn=warn)
        scheduler._callbacks = dataclasses.replace(scheduler._callbacks, on_warm_base_promote_l2=promote)
        scheduler._callbacks = dataclasses.replace(scheduler._callbacks, on_warm_base_resolve=resolve)
        return probe, warn, promote, resolve

    @pytest.mark.asyncio
    async def test_absent_engages_latch_and_warns_once(self):
        """(1) probe='absent' across N ticks → warn fires exactly once, no
        promote, and the latch (_warm_base_hard_down) is engaged."""
        scheduler, t = self._make_scheduler()
        probe, warn, promote, resolve = self._install_callbacks(
            scheduler, probe_result='absent',
        )

        for _ in range(3):
            await scheduler._apply_warm_base_hard_down_watchdog()

        warn.assert_called_once()
        promote.assert_not_called()
        assert scheduler._warm_base_hard_down is True

    @pytest.mark.asyncio
    async def test_halt_short_circuits_acquire_next(self):
        """(2) With the latch engaged, acquire_next() returns None immediately
        — no get_tasks call, no dispatch."""
        scheduler, t = self._make_scheduler()
        self._install_callbacks(scheduler, probe_result='absent')

        # Engage the latch first.
        await scheduler._apply_warm_base_hard_down_watchdog()
        assert scheduler._warm_base_hard_down is True

        get_tasks_mock = AsyncMock(return_value=[])
        scheduler.get_tasks = get_tasks_mock

        result = await scheduler.acquire_next()

        assert result is None
        get_tasks_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_within_window_does_not_promote(self):
        """(3) Latch engaged at t=0; clock advanced to < l2_window_secs (100s)
        while still absent → promote must NOT fire."""
        scheduler, t = self._make_scheduler(l2_window_secs=100.0)
        probe, warn, promote, resolve = self._install_callbacks(
            scheduler, probe_result='absent',
        )

        await scheduler._apply_warm_base_hard_down_watchdog()
        assert scheduler._warm_base_hard_down is True

        t[0] = 50.0
        await scheduler._apply_warm_base_hard_down_watchdog()

        promote.assert_not_called()

    @pytest.mark.asyncio
    async def test_past_window_promotes_once_then_dedups(self):
        """(4) Past l2_window_secs while still absent → promote fires exactly
        once; further ticks (still absent) do NOT re-promote."""
        scheduler, t = self._make_scheduler(l2_window_secs=100.0)
        probe, warn, promote, resolve = self._install_callbacks(
            scheduler, probe_result='absent',
        )

        await scheduler._apply_warm_base_hard_down_watchdog()
        assert scheduler._warm_base_hard_down is True

        t[0] = 150.0
        await scheduler._apply_warm_base_hard_down_watchdog()

        promote.assert_called_once()

        # Further ticks must not re-fire (dedup).
        for _ in range(3):
            await scheduler._apply_warm_base_hard_down_watchdog()

        promote.assert_called_once()
        assert scheduler._warm_base_l2_promoted is True

    @pytest.mark.asyncio
    async def test_probe_recovers_resolves_and_clears_latch(self):
        """(5) probe flips to 'ok' → resolve fires exactly once, the latch is
        fully cleared, and acquire_next() no longer halts on this latch."""
        scheduler, t = self._make_scheduler(l2_window_secs=100.0)
        probe, warn, promote, resolve = self._install_callbacks(
            scheduler, probe_result='absent',
        )

        await scheduler._apply_warm_base_hard_down_watchdog()
        t[0] = 150.0
        await scheduler._apply_warm_base_hard_down_watchdog()
        assert scheduler._warm_base_hard_down is True
        assert scheduler._warm_base_l2_promoted is True

        # Base recovers.
        probe.return_value = 'ok'
        await scheduler._apply_warm_base_hard_down_watchdog()

        resolve.assert_called_once()
        assert scheduler._warm_base_hard_down is False
        assert scheduler._warm_base_l2_promoted is False
        assert scheduler._warm_base_hard_down_since is None

        # acquire_next() must proceed past the (now-cleared) halt check.
        get_tasks_mock = AsyncMock(return_value=[])
        scheduler.get_tasks = get_tasks_mock
        result = await scheduler.acquire_next()

        assert result is None  # no tasks — but for an ordinary reason, not the halt
        get_tasks_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_indeterminate_never_engages(self):
        """(6) probe='indeterminate' from the start → fail-safe: no warn, no halt."""
        scheduler, t = self._make_scheduler()
        probe, warn, promote, resolve = self._install_callbacks(
            scheduler, probe_result='indeterminate',
        )

        for _ in range(3):
            await scheduler._apply_warm_base_hard_down_watchdog()

        warn.assert_not_called()
        promote.assert_not_called()
        resolve.assert_not_called()
        assert scheduler._warm_base_hard_down is False

    @pytest.mark.asyncio
    async def test_disabled_is_a_noop(self):
        """(7) enabled=False → watchdog never even consults the probe."""
        scheduler, t = self._make_scheduler(enabled=False)
        probe, warn, promote, resolve = self._install_callbacks(
            scheduler, probe_result='absent',
        )

        await scheduler._apply_warm_base_hard_down_watchdog()

        probe.assert_not_called()
        warn.assert_not_called()
        assert scheduler._warm_base_hard_down is False

    @pytest.mark.asyncio
    async def test_disable_while_latched_releases_latch_and_resumes_dispatch(self):
        """(a) Disabling the watchdog mid-incident (operator hot-reload) must
        RELEASE an engaged latch, not wedge it forever.  The acquire_next
        halt (scheduler.py ~3287) is NOT gated on cfg.enabled, and
        warm_base_hard_down.enabled is a green-tier hot-reloadable field
        (config.py:2595) — so if the disabled early-return left the latch
        set, an operator disabling the watchdog to escape an incident would
        instead permanently freeze all dispatch host-wide."""
        scheduler, t = self._make_scheduler(enabled=True)
        probe, warn, promote, resolve = self._install_callbacks(
            scheduler, probe_result='absent',
        )

        await scheduler._apply_warm_base_hard_down_watchdog()
        assert scheduler._warm_base_hard_down is True

        # Operator hot-reloads config, disabling the watchdog mid-incident.
        scheduler.config.warm_base_hard_down.enabled = False

        await scheduler._apply_warm_base_hard_down_watchdog()

        assert scheduler._warm_base_hard_down is False
        assert scheduler._warm_base_hard_down_since is None
        assert scheduler._warm_base_l2_promoted is False
        resolve.assert_called_once()
        # The disabled early-return precedes the probe call — not re-consulted.
        assert probe.call_count == 1

        # Dispatch must resume past the now-cleared halt.
        get_tasks_mock = AsyncMock(return_value=[])
        scheduler.get_tasks = get_tasks_mock
        result = await scheduler.acquire_next()

        assert result is None  # no tasks — but for an ordinary reason, not the halt
        get_tasks_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_disable_while_unlatched_is_noop_no_resolve(self):
        """(b) enabled=False from the start (latch never engaged) must stay a
        pure no-op — the new clear+resolve is gated on the latch actually
        being engaged, preserving test_disabled_is_a_noop semantics."""
        scheduler, t = self._make_scheduler(enabled=False)
        probe, warn, promote, resolve = self._install_callbacks(
            scheduler, probe_result='absent',
        )

        await scheduler._apply_warm_base_hard_down_watchdog()

        resolve.assert_not_called()
        assert scheduler._warm_base_hard_down is False

    @pytest.mark.asyncio
    async def test_probe_raises_is_swallowed_and_holds_state(self):
        """(8) The probe callback raising must not crash the tick, and must
        leave the latch state unchanged (fail-safe hold)."""
        scheduler, t = self._make_scheduler()
        scheduler._callbacks = dataclasses.replace(scheduler._callbacks, warm_base_health_probe=AsyncMock(side_effect=RuntimeError('boom')))
        warn = AsyncMock()
        promote = AsyncMock()
        resolve = AsyncMock()
        scheduler._callbacks = dataclasses.replace(scheduler._callbacks, on_warm_base_warn=warn)
        scheduler._callbacks = dataclasses.replace(scheduler._callbacks, on_warm_base_promote_l2=promote)
        scheduler._callbacks = dataclasses.replace(scheduler._callbacks, on_warm_base_resolve=resolve)

        # Must not raise.
        await scheduler._apply_warm_base_hard_down_watchdog()

        warn.assert_not_called()
        promote.assert_not_called()
        resolve.assert_not_called()
        assert scheduler._warm_base_hard_down is False
