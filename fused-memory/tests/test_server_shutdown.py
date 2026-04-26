"""Tests for the _graceful_shutdown helper in fused_memory.server.main."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.server.main import _graceful_shutdown


class TestGracefulShutdownCallsMemoryServiceClose:
    @pytest.mark.asyncio
    async def test_shutdown_calls_memory_service_close(self):
        """_graceful_shutdown must await memory_service.close() once."""
        memory_service = MagicMock()
        memory_service.close = AsyncMock()

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=None,
            harness_loop_task=None,
            recon_journal=None,
        )

        memory_service.close.assert_awaited_once()


class TestGracefulShutdownClosesReconciliationJournal:
    @pytest.mark.asyncio
    async def test_shutdown_closes_reconciliation_journal(self):
        """_graceful_shutdown must await recon_journal.close() once."""
        memory_service = MagicMock()
        memory_service.close = AsyncMock()

        recon_journal = MagicMock()
        recon_journal.close = AsyncMock()

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=None,
            harness_loop_task=None,
            recon_journal=recon_journal,
        )

        recon_journal.close.assert_awaited_once()


class TestGracefulShutdownJournalClosedDespiteMemoryServiceError:
    @pytest.mark.asyncio
    async def test_recon_journal_closed_even_when_memory_service_close_raises(self):
        """recon_journal.close() must be called even when memory_service.close() raises.

        Verifies the independent try/except guard around memory_service.close() is
        load-bearing — a mock that raises proves the guard is needed.  If the guard
        were removed the RuntimeError would propagate and journal close would never run.
        """
        memory_service = MagicMock()
        memory_service.close = AsyncMock(side_effect=RuntimeError('memory close failed'))

        recon_journal = MagicMock()
        recon_journal.close = AsyncMock()

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=None,
            harness_loop_task=None,
            recon_journal=recon_journal,
        )

        recon_journal.close.assert_awaited_once()


class TestGracefulShutdownClosesTaskInterceptor:
    @pytest.mark.asyncio
    async def test_shutdown_closes_task_interceptor(self):
        """_graceful_shutdown must await task_interceptor.close() on happy path."""
        memory_service = MagicMock()
        memory_service.close = AsyncMock()

        task_interceptor = MagicMock()
        task_interceptor.drain = AsyncMock()
        task_interceptor.close = AsyncMock()

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=task_interceptor,
            harness_loop_task=None,
            recon_journal=None,
        )

        task_interceptor.close.assert_awaited_once()


class TestGracefulShutdownResilientToCloseError:
    @pytest.mark.asyncio
    async def test_shutdown_resilient_to_interceptor_close_error(self):
        """memory_service.close() must be called even if task_interceptor.close() raises."""
        memory_service = MagicMock()
        memory_service.close = AsyncMock()

        task_interceptor = MagicMock()
        task_interceptor.drain = AsyncMock()
        task_interceptor.close = AsyncMock(side_effect=RuntimeError('close failed'))

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=task_interceptor,
            harness_loop_task=None,
            recon_journal=None,
        )

        memory_service.close.assert_awaited_once()


class TestGracefulShutdownResilientToDrainError:
    @pytest.mark.asyncio
    async def test_shutdown_resilient_to_drain_error(self):
        """memory_service.close() must be called even if task_interceptor.drain() raises."""
        memory_service = MagicMock()
        memory_service.close = AsyncMock()

        task_interceptor = MagicMock()
        task_interceptor.drain = AsyncMock(side_effect=RuntimeError('drain failed'))
        task_interceptor.close = AsyncMock()

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=task_interceptor,
            harness_loop_task=None,
            recon_journal=None,
        )

        memory_service.close.assert_awaited_once()


class TestGracefulShutdownDrainsTaskInterceptor:
    @pytest.mark.asyncio
    async def test_shutdown_drains_task_interceptor(self):
        """_graceful_shutdown must await task_interceptor.drain() on happy path."""
        memory_service = MagicMock()
        memory_service.close = AsyncMock()

        task_interceptor = MagicMock()
        task_interceptor.drain = AsyncMock()
        task_interceptor.close = AsyncMock()

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=task_interceptor,
            harness_loop_task=None,
            recon_journal=None,
        )

        task_interceptor.drain.assert_awaited_once()


class TestGracefulShutdownCancelsHarnessLoopTask:
    @pytest.mark.asyncio
    async def test_shutdown_cancels_harness_loop_task(self):
        """_graceful_shutdown must cancel the harness loop asyncio.Task."""
        memory_service = MagicMock()
        memory_service.close = AsyncMock()

        # Create a real asyncio Task wrapping an infinite sleep
        async def _infinite():
            await asyncio.sleep(9999)

        harness_loop_task = asyncio.create_task(_infinite())

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=None,
            harness_loop_task=harness_loop_task,
            recon_journal=None,
        )

        assert harness_loop_task.cancelled()


class TestGracefulShutdownLogsHarnessTaskException:
    @pytest.mark.asyncio
    async def test_harness_task_exception_logged_not_swallowed(self):
        """Non-CancelledError from harness_loop_task must be logged via logger.exception().

        This test FAILS with the current code because the except clause catches
        (CancelledError, Exception) with bare ``pass``, silently discarding real errors.

        The task is allowed to run and raise RuntimeError *before* _graceful_shutdown
        is called.  cancel() becomes a no-op (task already done), and await raises the
        stored RuntimeError — which must be logged, not swallowed.
        """
        memory_service = MagicMock()
        memory_service.close = AsyncMock()

        async def _raises_runtime_error():
            raise RuntimeError('unexpected harness crash')

        harness_loop_task = asyncio.create_task(_raises_runtime_error())
        # Let the task run and store the RuntimeError before we pass it to _graceful_shutdown
        await asyncio.sleep(0)

        with patch('fused_memory.server.main.logger') as mock_logger:
            await _graceful_shutdown(
                memory_service=memory_service,
                task_interceptor=None,
                harness_loop_task=harness_loop_task,
                recon_journal=None,
            )

        mock_logger.exception.assert_called_once()


class TestGracefulShutdownHarnessTaskTimeout:
    @pytest.mark.timeout(2)
    @pytest.mark.asyncio
    async def test_shutdown_completes_even_when_harness_task_hangs_in_cleanup(self):
        """_graceful_shutdown must complete within a bounded time even if the harness task
        hangs in its cancellation-cleanup phase (e.g. doing long cleanup work after
        catching CancelledError).

        This test FAILS with the current code because the bare ``await harness_loop_task``
        has no timeout — once the task catches the first CancelledError and enters its
        cleanup branch it hangs indefinitely, blocking shutdown forever.

        After step-4 wraps the await in
        ``asyncio.wait_for(harness_loop_task, timeout=_HARNESS_CANCEL_TIMEOUT)``
        the patched 0.01s timeout fires, the cleanup sleep is interrupted by a second
        cancel, and _graceful_shutdown proceeds within the pytest-timeout window.
        """
        memory_service = MagicMock()
        memory_service.close = AsyncMock()

        # Simulates a harness that hangs indefinitely in its cleanup after being cancelled.
        # It DOES respond to a *second* cancellation (no uncancel()), so asyncio.wait_for
        # can interrupt it — but without an internal timeout the first await is stuck.
        async def _hangs_in_cleanup():
            try:
                await asyncio.sleep(9999)
            except asyncio.CancelledError:
                await asyncio.sleep(9999)  # cleanup work that hangs; cancellable

        harness_loop_task = asyncio.create_task(_hangs_in_cleanup())
        await asyncio.sleep(0)  # let the task start and reach its first await

        with patch('fused_memory.server.main._HARNESS_CANCEL_TIMEOUT', 0.01):
            await _graceful_shutdown(
                memory_service=memory_service,
                task_interceptor=None,
                harness_loop_task=harness_loop_task,
                recon_journal=None,
            )

        # If we reach here, _graceful_shutdown completed (didn't hang indefinitely)
        memory_service.close.assert_awaited_once()


class TestGracefulShutdownJournalClosedDespiteDrainError:
    @pytest.mark.asyncio
    async def test_recon_journal_closed_even_when_drain_raises(self):
        """recon_journal.close() must be called even when task_interceptor.drain() raises.

        Verifies the independent try/except guard around drain() — a drain failure
        must not prevent journal cleanup.
        """
        memory_service = MagicMock()
        memory_service.close = AsyncMock()

        task_interceptor = MagicMock()
        task_interceptor.drain = AsyncMock(side_effect=RuntimeError('drain failed'))
        task_interceptor.close = AsyncMock()

        recon_journal = MagicMock()
        recon_journal.close = AsyncMock()

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=task_interceptor,
            harness_loop_task=None,
            recon_journal=recon_journal,
        )

        recon_journal.close.assert_awaited_once()


class TestGracefulShutdownFiveStepOrdering:
    @pytest.mark.asyncio
    async def test_shutdown_steps_execute_in_correct_order(self):
        """_graceful_shutdown must execute exactly five steps in order:
        1. drain  2. interceptor_close  3. harness_cancel  4. memory_close  5. journal_close.

        Uses side_effect callbacks to append step names to a shared list,
        then asserts the list matches the expected sequence.
        """
        call_order: list[str] = []

        memory_service = MagicMock()
        memory_service.close = AsyncMock(
            side_effect=lambda: call_order.append('memory_close')
        )

        task_interceptor = MagicMock()
        task_interceptor.drain = AsyncMock(
            side_effect=lambda: call_order.append('drain')
        )
        task_interceptor.close = AsyncMock(
            side_effect=lambda: call_order.append('interceptor_close')
        )

        recon_journal = MagicMock()
        recon_journal.close = AsyncMock(
            side_effect=lambda: call_order.append('journal_close')
        )

        # Real asyncio Task that cancels quickly and records the cancel step
        async def _harness():
            await asyncio.sleep(9999)

        harness_loop_task = asyncio.create_task(_harness())

        original_cancel = harness_loop_task.cancel

        def _tracking_cancel(*args, **kwargs):
            call_order.append('harness_cancel')
            return original_cancel(*args, **kwargs)

        harness_loop_task.cancel = _tracking_cancel

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=task_interceptor,
            harness_loop_task=harness_loop_task,
            recon_journal=recon_journal,
        )

        assert call_order == ['drain', 'interceptor_close', 'harness_cancel', 'memory_close', 'journal_close']


class TestGracefulShutdownClosesEventQueue:
    """WP-B: EventQueue must close after task_interceptor and BEFORE memory_service.

    The drainer writes into the SQLite event buffer that memory_service owns,
    so closing the memory_service first would race with the final flush.
    """

    @pytest.mark.asyncio
    async def test_event_queue_closed_before_memory_service(self):
        call_order: list[str] = []

        memory_service = MagicMock()
        memory_service.close = AsyncMock(
            side_effect=lambda: call_order.append('memory_close')
        )

        task_interceptor = MagicMock()
        task_interceptor.drain = AsyncMock(
            side_effect=lambda: call_order.append('drain')
        )
        task_interceptor.close = AsyncMock(
            side_effect=lambda: call_order.append('interceptor_close')
        )

        event_queue = MagicMock()
        event_queue.close = AsyncMock(
            side_effect=lambda: call_order.append('event_queue_close')
        )

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=task_interceptor,
            harness_loop_task=None,
            recon_journal=None,
            event_queue=event_queue,
        )

        assert 'event_queue_close' in call_order
        eq_idx = call_order.index('event_queue_close')
        mem_idx = call_order.index('memory_close')
        ic_idx = call_order.index('interceptor_close')
        assert ic_idx < eq_idx < mem_idx, (
            f'expected interceptor_close < event_queue_close < memory_close, '
            f'got order: {call_order}'
        )

    @pytest.mark.asyncio
    async def test_event_queue_closed_even_when_interceptor_drain_raises(self):
        """Independent try/except: drain failure must not skip event_queue.close."""
        memory_service = MagicMock()
        memory_service.close = AsyncMock()

        task_interceptor = MagicMock()
        task_interceptor.drain = AsyncMock(side_effect=RuntimeError('drain failed'))
        task_interceptor.close = AsyncMock()

        event_queue = MagicMock()
        event_queue.close = AsyncMock()

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=task_interceptor,
            harness_loop_task=None,
            recon_journal=None,
            event_queue=event_queue,
        )

        event_queue.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_memory_service_closed_even_when_event_queue_close_raises(self):
        """A broken event_queue.close must not block memory_service.close."""
        memory_service = MagicMock()
        memory_service.close = AsyncMock()

        event_queue = MagicMock()
        event_queue.close = AsyncMock(side_effect=RuntimeError('queue close failed'))

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=None,
            harness_loop_task=None,
            recon_journal=None,
            event_queue=event_queue,
        )

        memory_service.close.assert_awaited_once()


class TestGracefulShutdownDoesNotArmForceExitWatchdog:
    @pytest.mark.asyncio
    async def test_shutdown_does_not_arm_force_exit_timer(self):
        """_graceful_shutdown must NOT arm the force-exit watchdog (Task 1080 regression).

        Calling _graceful_shutdown directly (as done in every test in this file) must
        not leave a 45s daemon threading.Timer behind.  If it does, a long pytest run
        will be killed by os._exit(1) mid-suite — no individual test failure, just a
        truncated run with a non-zero exit code.
        """
        import fused_memory.server.main as main_mod

        # Ensure clean state before the call.
        main_mod._cancel_force_exit()
        assert main_mod._shutdown_watchdog is None, 'precondition: no watchdog before call'

        memory_service = MagicMock()
        memory_service.close = AsyncMock()

        try:
            await _graceful_shutdown(
                memory_service=memory_service,
                task_interceptor=None,
                harness_loop_task=None,
                recon_journal=None,
            )

            assert main_mod._shutdown_watchdog is None, (
                'Task 1080 regression: _graceful_shutdown armed a 45s os._exit(1) watchdog. '
                'The watchdog must only be armed by _shutdown_with_watchdog (the lifespan-only wrapper).'
            )
        finally:
            main_mod._cancel_force_exit()


class TestShutdownWithWatchdog:
    @pytest.mark.asyncio
    async def test_arms_force_exit_watchdog_before_invoking_graceful_shutdown(self):
        """_shutdown_with_watchdog must arm the watchdog BEFORE delegating to _graceful_shutdown.

        The spy records whether _shutdown_watchdog was set at the moment the
        delegate was called.  A value of True in armed_state proves the arm
        happened before the call, not after.
        """
        import fused_memory.server.main as main_mod

        # Ensure clean state.
        main_mod._cancel_force_exit()
        assert main_mod._shutdown_watchdog is None, 'precondition: no watchdog before call'

        armed_state: list[bool] = []

        async def _spy(**kwargs):  # type: ignore[override]
            armed_state.append(main_mod._shutdown_watchdog is not None)

        try:
            with patch.object(main_mod, '_graceful_shutdown', _spy):
                await main_mod._shutdown_with_watchdog(
                    memory_service=MagicMock(close=AsyncMock()),
                    task_interceptor=None,
                    harness_loop_task=None,
                    recon_journal=None,
                )

            assert armed_state == [True], (
                '_shutdown_with_watchdog must arm the watchdog before calling _graceful_shutdown'
            )
        finally:
            main_mod._cancel_force_exit()

    @pytest.mark.asyncio
    async def test_forwards_all_kwargs_to_graceful_shutdown(self):
        """_shutdown_with_watchdog must forward all six kwargs to _graceful_shutdown unchanged.

        Especially important for the optional event_queue and sqlite_watchdog args — if
        either is dropped the production shutdown skips flushing the bounded write queue
        or cancelling the SQLite watchdog.

        The test guards against silent kwarg drops by first asserting that the full set of
        forwarded keys exactly matches the six expected names, then asserting per-key
        identity against unique non-None MagicMock sentinels.
        """
        import fused_memory.server.main as main_mod

        main_mod._cancel_force_exit()

        memory_service = MagicMock(close=AsyncMock())
        task_interceptor = MagicMock(drain=AsyncMock(), close=AsyncMock())
        harness_loop_task = MagicMock()
        recon_journal = MagicMock(close=AsyncMock())
        event_queue = MagicMock(close=AsyncMock())
        sqlite_watchdog = MagicMock(close=AsyncMock())

        captured: dict = {}

        async def _spy(**kwargs):  # type: ignore[override]
            captured.update(kwargs)

        try:
            with patch.object(main_mod, '_graceful_shutdown', _spy):
                await main_mod._shutdown_with_watchdog(
                    memory_service=memory_service,
                    task_interceptor=task_interceptor,
                    harness_loop_task=harness_loop_task,
                    recon_journal=recon_journal,
                    event_queue=event_queue,
                    sqlite_watchdog=sqlite_watchdog,
                )

            expected_keys = {
                'memory_service', 'task_interceptor', 'harness_loop_task',
                'recon_journal', 'event_queue', 'sqlite_watchdog',
            }
            assert captured.keys() == expected_keys, (
                f'_shutdown_with_watchdog dropped or added a forwarded kwarg: '
                f'expected {expected_keys}, got {set(captured.keys())}'
            )
            assert captured['memory_service'] is memory_service
            assert captured['task_interceptor'] is task_interceptor
            assert captured['harness_loop_task'] is harness_loop_task
            assert captured['recon_journal'] is recon_journal
            assert captured['event_queue'] is event_queue
            assert captured['sqlite_watchdog'] is sqlite_watchdog
        finally:
            main_mod._cancel_force_exit()
