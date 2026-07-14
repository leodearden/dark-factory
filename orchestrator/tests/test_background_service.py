"""Tests for orchestrator.background_service (W10-η).

PRD ``plans/harness-supervision-prd.md`` §5.3 (LR-1/2/3): one reusable seam
collapsing the eleven background-loop/service lifecycles in harness.py so
the recurring shutdown-hang class (survey 2.3; tasks 108/161/162/169/875/
1080) becomes structurally impossible.

Steps covered:
  step-1   RED   — BackoffPolicy constant-delay contract
  step-2   GREEN — BackoffPolicy + DEFAULT_BACKOFF_SECS
  step-3   RED   — BackgroundService.start() idempotency
  step-4   GREEN — BackgroundService dataclass + idempotent start()
  step-5   RED   — BackgroundService loop contract (S2 / LR-1)
  step-6   GREEN — BackgroundService._loop() canonical sleep-first body
  step-7   RED   — BackgroundService.stop() contract
  step-8   GREEN — BackgroundService.stop()
  step-9   RED   — LifecycleRegistry.register()/start_all() (LR-3 start side)
  step-10  GREEN — LifecycleRegistry register/start_all
  step-11  RED   — LifecycleRegistry.stop_all() (S1 / LR-2)
  step-12  GREEN — LifecycleRegistry.stop_all()
  step-13  RED   — ManagedService adapter + LifecycleService Protocol
  step-14  GREEN — ManagedService + LifecycleService

This module imports ``orchestrator.background_service`` symbols LOCALLY
inside each test (not at module scope) so a not-yet-implemented symbol
never breaks collection of the rest of the file during the RED steps —
mirrors test_merge_queue_lifecycle_registry.py's / test_item_lifecycle.py's
convention.

This project's pyproject.toml does NOT set ``asyncio_mode = "auto"``, so
pytest-asyncio's default "strict" mode applies: every coroutine test needs
an explicit ``@pytest.mark.asyncio`` marker, applied PER TEST (never at
class or module level, since a class-level mark would error the sync tests
sharing that class) — mirrors test_merge_queue_resource_audit.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from unittest.mock import AsyncMock, patch

import pytest


class TestBackoffPolicy:
    """step-1: BackoffPolicy constant-delay contract."""

    def test_constant_delay_regardless_of_attempt(self) -> None:
        from orchestrator.background_service import BackoffPolicy

        policy = BackoffPolicy(60.0)

        assert policy.delay_for(0) == 60.0
        assert policy.delay_for(1) == 60.0
        assert policy.delay_for(5) == 60.0
        assert policy.delay_for(100) == 60.0

    def test_default_backoff_secs_matches_harness_parity_constant(self) -> None:
        from orchestrator.background_service import DEFAULT_BACKOFF_SECS
        from orchestrator.harness import _BG_LOOP_FAILURE_BACKOFF_SECS

        assert DEFAULT_BACKOFF_SECS == _BG_LOOP_FAILURE_BACKOFF_SECS
        assert DEFAULT_BACKOFF_SECS == 60.0

    def test_delay_secs_defaults_to_default_backoff_secs(self) -> None:
        """Amendment (reviewer_comprehensive): DEFAULT_BACKOFF_SECS is now
        load-bearing as BackoffPolicy's own default, rather than a
        parity-test-only constant — a bare ``BackoffPolicy()`` already
        carries the parity value."""
        from orchestrator.background_service import DEFAULT_BACKOFF_SECS, BackoffPolicy

        policy = BackoffPolicy()

        assert policy.delay_secs == DEFAULT_BACKOFF_SECS
        assert policy.delay_for(0) == DEFAULT_BACKOFF_SECS


def _make_service(pass_fn, **overrides):
    """Construct a BackgroundService with test-friendly defaults.

    Local import (not module-level) so a not-yet-implemented
    ``BackgroundService`` never breaks collection of ``TestBackoffPolicy``.
    """
    from orchestrator.background_service import BackgroundService, BackoffPolicy

    kwargs = {
        'name': 'test-service',
        'pass_fn': pass_fn,
        'interval_secs': 0.01,
        'backoff': BackoffPolicy(60.0),
        'stop_timeout_secs': 1.0,
        'max_failure_logs': 3,
    }
    kwargs.update(overrides)
    return BackgroundService(**kwargs)


class _RecordingSleep:
    """Async-callable ``asyncio.sleep`` stand-in recording each delay."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class TestBackgroundServiceStart:
    """step-3: BackgroundService.start() idempotency."""

    @pytest.mark.asyncio
    async def test_start_creates_exactly_one_named_task(self) -> None:
        blocker = asyncio.Event()

        async def pass_fn() -> None:
            await blocker.wait()

        svc = _make_service(pass_fn)

        svc.start()
        task = svc._task
        assert task is not None
        assert isinstance(task, asyncio.Task)
        assert task.get_name() == 'test-service'
        assert not task.done()

        blocker.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_second_start_while_running_is_a_noop(self) -> None:
        blocker = asyncio.Event()

        async def pass_fn() -> None:
            await blocker.wait()

        svc = _make_service(pass_fn)

        svc.start()
        first_task = svc._task
        svc.start()

        assert first_task is not None
        assert svc._task is first_task

        blocker.set()
        first_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await first_task

    @pytest.mark.asyncio
    async def test_start_after_task_finished_creates_fresh_task(self) -> None:
        blocker = asyncio.Event()

        async def pass_fn() -> None:
            await blocker.wait()

        svc = _make_service(pass_fn)

        svc.start()
        first_task = svc._task
        assert first_task is not None
        first_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await first_task
        assert first_task.done()

        svc.start()
        second_task = svc._task

        assert second_task is not None
        assert second_task is not first_task
        assert not second_task.done()

        blocker.set()
        second_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await second_task


class TestBackgroundServiceLoop:
    """step-5: BackgroundService loop contract (S2 / LR-1)."""

    @pytest.mark.asyncio
    async def test_cancelled_error_from_pass_fn_propagates(self) -> None:
        async def pass_fn() -> None:
            raise asyncio.CancelledError()

        svc = _make_service(pass_fn, interval_secs=0)
        svc.start()
        assert svc._task is not None

        with pytest.raises(asyncio.CancelledError):
            await svc._task

        assert svc._task.cancelled()

    @pytest.mark.asyncio
    async def test_pass_runs_after_interval_sleep_each_iteration(self) -> None:
        """Sleep-first: each iteration sleeps BEFORE running the pass."""
        # Captured BEFORE patching: the patch target below resolves to the
        # shared ``asyncio`` module object (background_service does a plain
        # ``import asyncio`), so patching ``.sleep`` on it mutates
        # ``asyncio.sleep`` process-wide for the ``with`` block's duration.
        # ``fake_sleep`` must yield via THIS captured reference, not a fresh
        # ``asyncio.sleep`` lookup, or it would recurse into itself.
        real_sleep = asyncio.sleep
        order: list[str] = []
        done_event = asyncio.Event()

        async def fake_sleep(_delay: float) -> None:
            order.append('sleep')
            await real_sleep(0)

        async def pass_fn() -> None:
            order.append('pass')
            if order.count('pass') >= 3:
                done_event.set()

        svc = _make_service(pass_fn, interval_secs=5.0)

        with patch('orchestrator.background_service.asyncio.sleep', new=fake_sleep):
            svc.start()
            assert svc._task is not None
            await asyncio.wait_for(done_event.wait(), timeout=5)
            svc._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await svc._task

        assert order[:6] == ['sleep', 'pass', 'sleep', 'pass', 'sleep', 'pass']

    @pytest.mark.asyncio
    async def test_plain_exception_bounded_logged_backs_off_then_recovers(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Failures log at most max_failure_logs consecutive ERRORs (no
        traceback), back off, and a later success resets the counter so a
        fresh failure streak logs again."""
        from orchestrator.background_service import BackoffPolicy

        # 4 consecutive fails (exceeds max_failure_logs=3) -> 1 success
        # (resets the counter) -> 1 more fail (must log again, proving reset).
        outcomes = ['fail', 'fail', 'fail', 'fail', 'ok', 'fail']
        calls: list[int] = []
        done_event = asyncio.Event()
        hang_forever = asyncio.Event()

        async def scripted_pass() -> None:
            idx = len(calls)
            calls.append(idx)
            if idx >= len(outcomes):
                done_event.set()
                await hang_forever.wait()
                return
            if outcomes[idx] == 'fail':
                raise RuntimeError(f'boom-{idx}')

        svc = _make_service(
            scripted_pass,
            interval_secs=0,
            backoff=BackoffPolicy(42.0),
            max_failure_logs=3,
        )
        recording_sleep = _RecordingSleep()

        with (
            patch('orchestrator.background_service.asyncio.sleep', new=recording_sleep),
            caplog.at_level(logging.ERROR, logger='orchestrator.background_service'),
        ):
            svc.start()
            assert svc._task is not None
            await asyncio.wait_for(done_event.wait(), timeout=5)
            svc._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await svc._task

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) == 4
        for record in error_records:
            assert record.exc_info is None, 'must use logger.error, never logger.exception'

        # Backoff delay (42.0) follows each of the 5 failing passes;
        # interval (0) follows the 1 successful pass — proves the failure
        # path routes through BackoffPolicy rather than the plain interval.
        assert recording_sleep.delays.count(42.0) == 5
        assert recording_sleep.delays.count(0) >= 1


class TestBackgroundServiceStop:
    """step-7: BackgroundService.stop() contract."""

    @pytest.mark.asyncio
    async def test_stop_cancels_suppresses_and_clears_slot(self) -> None:
        blocker = asyncio.Event()

        async def pass_fn() -> None:
            await blocker.wait()

        svc = _make_service(pass_fn, interval_secs=0)
        svc.start()
        assert svc._task is not None

        await asyncio.wait_for(svc.stop(), timeout=5)

        assert svc._task is None

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self) -> None:
        blocker = asyncio.Event()

        async def pass_fn() -> None:
            await blocker.wait()

        svc = _make_service(pass_fn, interval_secs=0)
        svc.start()

        await svc.stop()
        assert svc._task is None

        # Second stop(): clean no-op, must not raise.
        await svc.stop()
        assert svc._task is None

    @pytest.mark.asyncio
    async def test_stop_never_started_is_noop(self) -> None:
        svc = _make_service(AsyncMock())

        await svc.stop()

        assert svc._task is None

    @pytest.mark.asyncio
    async def test_stop_well_behaved_task_finishes_promptly(self) -> None:
        """A loop mid-pass on a cooperative (non-blocking) pass_fn stops
        promptly under cancellation — the bounded-stop contract that
        LifecycleRegistry.stop_all's wait_for relies on (S1)."""

        async def pass_fn() -> None:
            await asyncio.sleep(0)

        svc = _make_service(pass_fn, interval_secs=1000)
        svc.start()

        await asyncio.wait_for(svc.stop(), timeout=1)

        assert svc._task is None

    @pytest.mark.asyncio
    async def test_stop_logs_warning_on_unexpected_task_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Amendment (reviewer_comprehensive): a non-CancelledError exception
        surfacing from the loop task at teardown (e.g. a genuine bug outside
        the pass_fn contract, which _loop already bounded-logs on its own)
        is logged at WARNING rather than silently discarded."""
        svc = _make_service(AsyncMock())

        async def _boom() -> None:
            raise RuntimeError('unexpected loop bug')

        # Run the replacement task to completion (with its exception) BEFORE
        # handing it to stop(), so `.cancel()` on an already-done task is a
        # deterministic no-op and `await self._task` immediately re-raises
        # the stored exception — no timing race with real cancellation.
        task = asyncio.create_task(_boom(), name='test-service')
        with contextlib.suppress(RuntimeError):
            await task
        assert task.done()
        svc._task = task

        with caplog.at_level(logging.WARNING, logger='orchestrator.background_service'):
            await svc.stop()

        assert svc._task is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert 'test-service' in warnings[0].message
        assert 'unexpected loop bug' in warnings[0].message
        assert warnings[0].exc_info is None, 'must use logger.warning, never logger.exception'


class _RecordingService:
    """Fake lifecycle service recording start()/stop() calls into a shared
    list, in call order.

    Duck-types the eventual ``LifecycleService`` Protocol (name, async
    start(), async stop(), stop_timeout_secs — see step-14) without
    depending on it, so LifecycleRegistry can be tested before
    ManagedService/LifecycleService exist.
    """

    def __init__(self, name: str, calls: list[str], stop_timeout_secs: float = 1.0) -> None:
        self.name = name
        self._calls = calls
        self.stop_timeout_secs = stop_timeout_secs

    async def start(self) -> None:
        self._calls.append(f'{self.name}:start')

    async def stop(self) -> None:
        self._calls.append(f'{self.name}:stop')


class TestLifecycleRegistryRegister:
    """step-9: LifecycleRegistry.register() preserves declared order (LR-3)."""

    def test_register_preserves_declared_order(self) -> None:
        from orchestrator.background_service import LifecycleRegistry

        calls: list[str] = []
        registry = LifecycleRegistry()
        svc_a = _RecordingService('a', calls)
        svc_b = _RecordingService('b', calls)
        svc_c = _RecordingService('c', calls)

        registry.register(svc_a)
        registry.register(svc_b)
        registry.register(svc_c)

        assert [s.name for s in registry.services] == ['a', 'b', 'c']

    def test_services_reflects_registration_order_not_name_order(self) -> None:
        """register() order is preserved regardless of name — `services` is
        a plain ordered sequence, not something that re-sorts."""
        from orchestrator.background_service import LifecycleRegistry

        calls: list[str] = []
        registry = LifecycleRegistry()
        for name in ['zebra', 'apple', 'mango']:
            registry.register(_RecordingService(name, calls))

        assert [s.name for s in registry.services] == ['zebra', 'apple', 'mango']


class TestLifecycleRegistryStartAll:
    """step-9: LifecycleRegistry.start_all() awaits start() in order (LR-3)."""

    @pytest.mark.asyncio
    async def test_start_all_awaits_each_start_in_registration_order(self) -> None:
        from orchestrator.background_service import LifecycleRegistry

        calls: list[str] = []
        registry = LifecycleRegistry()
        for name in ['a', 'b', 'c']:
            registry.register(_RecordingService(name, calls))

        await registry.start_all()

        assert calls == ['a:start', 'b:start', 'c:start']

    @pytest.mark.asyncio
    async def test_start_all_awaits_sequentially_not_concurrently(self) -> None:
        """A slow first service's start() fully completes before the next
        service's start() begins — proves start_all() awaits each start()
        in turn rather than firing them concurrently (asyncio.gather would
        let the fast second service finish first)."""
        from orchestrator.background_service import LifecycleRegistry

        calls: list[str] = []
        registry = LifecycleRegistry()

        class _SlowStart:
            name = 'slow'
            stop_timeout_secs = 1.0

            async def start(self) -> None:
                await asyncio.sleep(0.05)
                calls.append('slow:start-done')

            async def stop(self) -> None:
                calls.append('slow:stop')

        class _FastStart:
            name = 'fast'
            stop_timeout_secs = 1.0

            async def start(self) -> None:
                calls.append('fast:start-done')

            async def stop(self) -> None:
                calls.append('fast:stop')

        registry.register(_SlowStart())
        registry.register(_FastStart())

        await registry.start_all()

        assert calls == ['slow:start-done', 'fast:start-done']

    @pytest.mark.asyncio
    async def test_start_all_logs_info_per_service(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Amendment (reviewer_comprehensive): preserves the visibility the
        old per-service ``_start_*`` methods used to log individually —
        operators can confirm from logs which services actually started."""
        from orchestrator.background_service import LifecycleRegistry

        calls: list[str] = []
        registry = LifecycleRegistry()
        for name in ['a', 'b']:
            registry.register(_RecordingService(name, calls))

        with caplog.at_level(logging.INFO, logger='orchestrator.background_service'):
            await registry.start_all()

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert [r.message for r in info_records] == ['a started', 'b started']


class TestLifecycleRegistryStopAll:
    """step-11: LifecycleRegistry.stop_all() (S1 / LR-2 — hang-class
    elimination). Every async test here is itself bound in
    ``asyncio.wait_for`` so a regression that reintroduces the hang class
    fails the test instead of wedging the suite."""

    @pytest.mark.asyncio
    async def test_stop_all_calls_stop_in_strict_reverse_order(self) -> None:
        from orchestrator.background_service import LifecycleRegistry

        calls: list[str] = []
        registry = LifecycleRegistry()
        for name in ['a', 'b', 'c']:
            registry.register(_RecordingService(name, calls))

        await asyncio.wait_for(registry.stop_all(), timeout=5)

        assert calls == ['c:stop', 'b:stop', 'a:stop']

    @pytest.mark.asyncio
    async def test_stop_all_abandons_wedging_service_and_completes_ladder(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A service whose stop() never returns is abandoned after its own
        stop_timeout_secs with a WARNING, and stop_all() still proceeds to
        stop every OTHER service in reverse order — this is the structural
        elimination of the shutdown-hang class (S1)."""
        from orchestrator.background_service import LifecycleRegistry

        calls: list[str] = []
        registry = LifecycleRegistry()

        class _WedgingService:
            name = 'wedge'
            stop_timeout_secs = 0.05

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                await asyncio.Event().wait()  # never set: wedges forever

        registry.register(_RecordingService('a', calls))
        registry.register(_WedgingService())
        registry.register(_RecordingService('c', calls))

        with caplog.at_level(logging.WARNING, logger='orchestrator.background_service'):
            await asyncio.wait_for(registry.stop_all(), timeout=5)

        # Reverse order c, wedge, a — wedge is abandoned but a is still reached.
        assert calls == ['c:stop', 'a:stop']

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert 'wedge' in warnings[0].message
        assert 'did not stop' in warnings[0].message

    @pytest.mark.asyncio
    async def test_stop_all_catches_stop_exception_and_continues_ladder(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A service whose stop() raises a plain Exception is caught+logged
        (WARNING) rather than propagating, so one failure cannot abort the
        ladder."""
        from orchestrator.background_service import LifecycleRegistry

        calls: list[str] = []
        registry = LifecycleRegistry()

        class _RaisingService:
            name = 'raiser'
            stop_timeout_secs = 1.0

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                raise RuntimeError('stop boom')

        registry.register(_RecordingService('a', calls))
        registry.register(_RaisingService())
        registry.register(_RecordingService('c', calls))

        with caplog.at_level(logging.WARNING, logger='orchestrator.background_service'):
            await asyncio.wait_for(registry.stop_all(), timeout=5)

        assert calls == ['c:stop', 'a:stop']

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert 'raiser' in warnings[0].message
        assert 'stop() failed' in warnings[0].message

    @pytest.mark.asyncio
    async def test_stop_all_logs_info_per_clean_stop(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Amendment (reviewer_comprehensive): preserves the visibility the
        old per-service ``_stop_*`` methods used to log individually —
        operators can confirm shutdown progressed through the ladder."""
        from orchestrator.background_service import LifecycleRegistry

        calls: list[str] = []
        registry = LifecycleRegistry()
        for name in ['a', 'b']:
            registry.register(_RecordingService(name, calls))

        with caplog.at_level(logging.INFO, logger='orchestrator.background_service'):
            await asyncio.wait_for(registry.stop_all(), timeout=5)

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        # Reverse registration order: 'b' stops before 'a'.
        assert [r.message for r in info_records] == ['b stopped', 'a stopped']


class TestManagedService:
    """step-13: ManagedService adapter (+ LifecycleService Protocol) — a
    thin wrapper so the four bespoke lifecycle services (merge-worker,
    offline-lane, escalation-server, watcher-supervisor) register in the
    same LifecycleRegistry as BackgroundService, keeping their existing
    ``_start_*``/``_stop_*`` methods verbatim."""

    @pytest.mark.asyncio
    async def test_start_and_stop_delegate_to_wrapped_fns(self) -> None:
        from orchestrator.background_service import ManagedService

        calls: list[str] = []

        async def start_fn() -> None:
            calls.append('started')

        async def stop_fn() -> None:
            calls.append('stopped')

        svc = ManagedService(
            name='bespoke', start_fn=start_fn, stop_fn=stop_fn, stop_timeout_secs=1.0
        )

        await svc.start()
        await svc.stop()

        assert calls == ['started', 'stopped']

    def test_exposes_name_and_stop_timeout_secs(self) -> None:
        from orchestrator.background_service import ManagedService

        svc = ManagedService(
            name='bespoke', start_fn=AsyncMock(), stop_fn=AsyncMock(), stop_timeout_secs=2.5
        )

        assert svc.name == 'bespoke'
        assert svc.stop_timeout_secs == 2.5

    @pytest.mark.asyncio
    async def test_composes_with_background_service_and_honors_bounded_stop(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A ManagedService registers alongside a BackgroundService in the
        same LifecycleRegistry; a wedging ManagedService.stop() is
        abandoned by stop_all() exactly like a wedging BackgroundService
        would be (S1), and the other (BackgroundService) member still
        stops — proving the adapter honors the uniform bounded-stop
        contract rather than needing special-casing."""
        from orchestrator.background_service import LifecycleRegistry, ManagedService

        async def wedging_stop() -> None:
            await asyncio.Event().wait()

        wedge = ManagedService(
            name='wedge-managed',
            start_fn=AsyncMock(),
            stop_fn=wedging_stop,
            stop_timeout_secs=0.05,
        )

        blocker = asyncio.Event()

        async def bg_pass() -> None:
            await blocker.wait()

        background = _make_service(bg_pass, name='background', interval_secs=0)

        registry = LifecycleRegistry()
        registry.register(wedge)
        registry.register(background)

        await registry.start_all()
        assert background._task is not None

        with caplog.at_level(logging.WARNING, logger='orchestrator.background_service'):
            await asyncio.wait_for(registry.stop_all(), timeout=5)

        # background (registered second) stops first under reverse order;
        # the wedging ManagedService is abandoned, not hung on forever.
        assert background._task is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert 'wedge-managed' in warnings[0].message
        assert 'did not stop' in warnings[0].message
