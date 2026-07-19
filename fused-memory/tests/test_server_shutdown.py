"""Tests for the _graceful_shutdown helper in fused_memory.server.main."""

import asyncio
import contextlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.server.main import _build_uvicorn_config, _graceful_shutdown, _run_shielded


def _parse_timeout_stop_sec(unit_file_path: Path) -> float:
    """Extract the numeric value of a non-comment ``TimeoutStopSec=`` directive
    from a systemd unit file.

    Skips blank/comment (``#``, ``;``) lines so a commented-out directive can't
    silently match. Raises ``AssertionError`` (rather than returning a default)
    if the directive is absent, so a future template rewrite that drops the
    directive fails loudly instead of the comparison being skipped.
    """
    for raw_line in unit_file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("TimeoutStopSec="):
            return float(line.split("=", 1)[1])
    raise AssertionError(f"TimeoutStopSec= directive not found in {unit_file_path}")


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


class TestGracefulShutdownClosesCuratorCostStore:
    """step-5: _graceful_shutdown must await curator_cost_store.close() when provided."""

    @pytest.mark.asyncio
    async def test_curator_cost_store_closed_on_happy_path(self):
        """curator_cost_store.close() must be awaited once on the happy path."""
        memory_service = MagicMock()
        memory_service.close = AsyncMock()

        curator_cost_store = MagicMock()
        curator_cost_store.close = AsyncMock()

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=None,
            harness_loop_task=None,
            recon_journal=None,
            curator_cost_store=curator_cost_store,
        )

        curator_cost_store.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_curator_cost_store_closed_even_when_memory_service_raises(self):
        """curator_cost_store.close() must be awaited even if memory_service.close() raises.

        Mirrors TestGracefulShutdownJournalClosedDespiteMemoryServiceError in
        test_server_shutdown.py — each cleanup step runs under its own shield.
        """
        memory_service = MagicMock()
        memory_service.close = AsyncMock(side_effect=RuntimeError('memory close failed'))

        curator_cost_store = MagicMock()
        curator_cost_store.close = AsyncMock()

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=None,
            harness_loop_task=None,
            recon_journal=None,
            curator_cost_store=curator_cost_store,
        )

        curator_cost_store.close.assert_awaited_once()


class TestGracefulShutdownQuiescesCuratorGate:
    """curator_usage_gate.shutdown() must be awaited before curator_cost_store.close().

    Fixes the race where a 429-driven cap event fires just before shutdown
    and its background save_account_event task writes to an already-closed
    aiosqlite connection (reviewer suggestion 2).
    """

    @pytest.mark.asyncio
    async def test_gate_shutdown_called_when_provided(self):
        """curator_usage_gate.shutdown() is awaited once when the gate is provided."""
        memory_service = MagicMock(close=AsyncMock())
        curator_usage_gate = MagicMock(shutdown=AsyncMock())

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=None,
            harness_loop_task=None,
            recon_journal=None,
            curator_usage_gate=curator_usage_gate,
        )

        curator_usage_gate.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gate_shutdown_before_cost_store_close(self):
        """curator_usage_gate.shutdown() must complete before curator_cost_store.close().

        Ordering is critical: gate.shutdown() cancels/drains background tasks
        that may still be writing to the CostStore's SQLite connection.
        Closing the store first would leave those tasks writing to a closed
        connection, causing silent failures or aiosqlite errors.
        """
        call_order: list[str] = []
        memory_service = MagicMock(close=AsyncMock())

        curator_usage_gate = MagicMock(
            shutdown=AsyncMock(side_effect=lambda: call_order.append('gate_shutdown'))
        )
        curator_cost_store = MagicMock(
            close=AsyncMock(side_effect=lambda: call_order.append('store_close'))
        )

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=None,
            harness_loop_task=None,
            recon_journal=None,
            curator_usage_gate=curator_usage_gate,
            curator_cost_store=curator_cost_store,
        )

        assert call_order == ['gate_shutdown', 'store_close'], (
            f'Expected gate_shutdown before store_close, got: {call_order}'
        )

    @pytest.mark.asyncio
    async def test_gate_shutdown_called_even_when_task_interceptor_raises(self):
        """curator_usage_gate.shutdown() runs even if task_interceptor steps raise.

        Each step is shielded, so a failure in task_interceptor.drain/close
        must not prevent the gate from being quiesced.
        """
        memory_service = MagicMock(close=AsyncMock())
        task_interceptor = MagicMock(
            drain=AsyncMock(side_effect=RuntimeError('drain failed')),
            close=AsyncMock(),
        )
        curator_usage_gate = MagicMock(shutdown=AsyncMock())

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=task_interceptor,
            harness_loop_task=None,
            recon_journal=None,
            curator_usage_gate=curator_usage_gate,
        )

        curator_usage_gate.shutdown.assert_awaited_once()


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
    # The repo default timeout_method="thread" KILLS the pytest worker on
    # expiry, so a 2s outer guard is fragile under host CPU oversubscription.
    # The real behavioral assertion is the internal
    # patch('..._HARNESS_CANCEL_TIMEOUT', 0.01) below, which proves shutdown
    # completes despite a hung harness task; widening this outer guard only
    # gives scheduling headroom and is behavior-preserving.
    @pytest.mark.timeout(15)
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


class TestRunShieldedCallerCancelledCompletesInner:
    """Fix A: caller-cancellation must let the shielded inner task run to completion
    when it can finish within the remaining step budget — preventing the orphan
    detached task that wedged ``asyncio.run()`` on 2026-04-28 00:32:15.
    """

    @pytest.mark.timeout(5)
    @pytest.mark.asyncio
    async def test_run_shielded_caller_cancel_waits_for_inner_to_finish(self):
        """When the caller of _run_shielded is cancelled, the inner cleanup task
        must finish (not be abandoned as an orphan) provided it can do so within
        the remaining step budget.
        """
        inner_started = asyncio.Event()
        inner_finished = False

        async def _inner_cleanup():
            nonlocal inner_finished
            inner_started.set()
            # Short, well within the timeout — completes despite caller-cancel.
            await asyncio.sleep(0.2)
            inner_finished = True

        async def _outer():
            await _run_shielded('test_step', _inner_cleanup, timeout=2.0)

        outer_task = asyncio.create_task(_outer())
        await inner_started.wait()
        outer_task.cancel()

        # The outer task may itself raise CancelledError when awaited, but the
        # CONTRACT is that the inner cleanup runs to completion regardless.
        with contextlib.suppress(asyncio.CancelledError):
            await outer_task

        assert inner_finished, (
            'inner cleanup task was abandoned when caller was cancelled — '
            'this is the orphan-detached-task bug that wedged asyncio.run()'
        )


class TestRunShieldedCallerCancelledExceedsBudget:
    """Fix A: when the inner cleanup cannot finish within the remaining budget
    after caller-cancel, _run_shielded must explicitly cancel it instead of
    leaving it as a detached orphan.
    """

    @pytest.mark.timeout(5)
    @pytest.mark.asyncio
    async def test_run_shielded_cancels_inner_when_budget_exhausted(self):
        """If the inner step exceeds its budget after caller-cancel, the inner
        task must be cancelled (not orphaned) so asyncio.run() can shut down
        cleanly.
        """
        inner_started = asyncio.Event()
        inner_was_cancelled = False

        async def _inner_cleanup():
            nonlocal inner_was_cancelled
            inner_started.set()
            try:
                await asyncio.sleep(60)  # would exceed any sane budget
            except asyncio.CancelledError:
                inner_was_cancelled = True
                raise

        async def _outer():
            # Tight budget so we hit the deadline-expired branch quickly.
            await _run_shielded('test_step', _inner_cleanup, timeout=0.2)

        outer_task = asyncio.create_task(_outer())
        await inner_started.wait()
        outer_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await outer_task

        # _run_shielded must have explicitly cancelled the inner task once
        # the deadline expired — otherwise the inner would still be running.
        assert inner_was_cancelled, (
            'inner cleanup task was left as a detached orphan after the '
            'caller was cancelled and the step budget expired'
        )


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


class TestGracefulShutdownMemoryCloseBudget:
    """Task 2701: the memory_service.close step must run under the enlarged
    _MEMORY_CLOSE_STEP_TIMEOUT — not the flat _CLEANUP_STEP_TIMEOUT default — so
    its six sequentially-bounded sub-closes all fit within one step budget."""

    @pytest.mark.asyncio
    async def test_memory_service_close_step_uses_memory_close_step_timeout(self):
        import fused_memory.server.main as main_mod

        memory_service = MagicMock()
        memory_service.close = AsyncMock()

        recorded: list[tuple] = []

        async def _record(name, coro_factory, timeout=main_mod._CLEANUP_STEP_TIMEOUT):
            recorded.append((name, timeout))

        with patch.object(main_mod, '_run_shielded', _record):
            await _graceful_shutdown(
                memory_service=memory_service,
                task_interceptor=None,
                harness_loop_task=None,
                recon_journal=None,
            )

        steps = dict(recorded)
        assert 'memory_service.close' in steps, (
            f'memory_service.close step was not dispatched: {steps}'
        )
        assert steps['memory_service.close'] == main_mod._MEMORY_CLOSE_STEP_TIMEOUT, (
            'memory_service.close must run under _MEMORY_CLOSE_STEP_TIMEOUT, not the '
            f'flat per-step default; got {steps["memory_service.close"]}'
        )


class TestShutdownBudgetArithmetic:
    """Survey finding D1: the bounded uvicorn graceful-shutdown wait is serialized
    BEFORE _shutdown_with_watchdog arms _FORCE_EXIT_BUDGET, so worst-case shutdown
    is approximately graceful_shutdown_timeout + _FORCE_EXIT_BUDGET.  This must stay
    under systemd's TimeoutStopSec so the in-process force-exit watchdog provably
    fires before systemd SIGKILLs the cgroup (preserving exit-code control: exit 0
    for operator stop).
    """

    def test_default_budget_stays_under_systemd_timeout_stop_sec(self):
        import fused_memory.server.main as main_mod
        from fused_memory.config.schema import ServerConfig

        default_timeout = ServerConfig().graceful_shutdown_timeout
        assert default_timeout + main_mod._FORCE_EXIT_BUDGET < main_mod._SYSTEMD_TIMEOUT_STOP_SECS

    def test_schema_mirror_constants_match_main(self):
        """ServerConfig._validate_graceful_shutdown_timeout_budget duplicates
        _FORCE_EXIT_BUDGET/_SYSTEMD_TIMEOUT_STOP_SECS as private
        _MIRROR_* constants in config/schema.py (importing main.py there would
        be circular, since main.py imports FusedMemoryConfig from schema.py).
        Guard that the mirror doesn't silently drift from the source of truth.
        """
        import fused_memory.server.main as main_mod
        from fused_memory.config import schema as schema_mod

        assert schema_mod._MIRROR_FORCE_EXIT_BUDGET == main_mod._FORCE_EXIT_BUDGET
        assert (
            schema_mod._MIRROR_SYSTEMD_TIMEOUT_STOP_SECS
            == main_mod._SYSTEMD_TIMEOUT_STOP_SECS
        )

    def test_systemd_timeout_stop_secs_matches_unit_file(self):
        """Compare the constant against the *actual* TimeoutStopSec= value
        parsed from scripts/fused-memory.service.template, not a restated
        literal — so drift in either direction (template edited without the
        constant, or vice versa) fails this test instead of passing silently.
        """
        import fused_memory.server.main as main_mod

        unit_file = (
            Path(__file__).resolve().parent.parent.parent
            / "scripts"
            / "fused-memory.service.template"
        )
        parsed_timeout_stop_sec = _parse_timeout_stop_sec(unit_file)
        assert parsed_timeout_stop_sec == main_mod._SYSTEMD_TIMEOUT_STOP_SECS, (
            f"_SYSTEMD_TIMEOUT_STOP_SECS ({main_mod._SYSTEMD_TIMEOUT_STOP_SECS}) no "
            f"longer matches TimeoutStopSec={parsed_timeout_stop_sec} in {unit_file}. "
            "Update the constant in fused_memory/server/main.py so the "
            "force-exit-before-SIGKILL invariant stays honest."
        )

    def test_memory_close_step_budget_dominates_bounded_close(self):
        """Task 2701 documented basis: the memory_service.close step budget must
        exceed the flat per-step default AND dominate the worst-case bounded
        close (6 sub-closes each ≤ _SUBCLOSE_TIMEOUT), while the force-exit and
        systemd constants stay UNCHANGED so the schema.py _MIRROR_* guard and the
        systemd-template parity test both remain valid.
        """
        import fused_memory.server.main as main_mod
        from fused_memory.services import memory_service as ms_mod

        # Enlarged relative to the flat per-step default.
        assert main_mod._MEMORY_CLOSE_STEP_TIMEOUT > main_mod._CLEANUP_STEP_TIMEOUT
        # Dominates the worst-case bounded close: 6 sub-closes each capped at
        # _SUBCLOSE_TIMEOUT run sequentially inside the one step.
        assert main_mod._MEMORY_CLOSE_STEP_TIMEOUT >= 6 * ms_mod._SUBCLOSE_TIMEOUT

        # Illustrative worst-case step sum with the resized memory-close budget
        # still fits within the force-exit budget: five flat 5s steps
        # (drain, close, sqlite_watchdog, event_queue, journal_close)
        # + harness_cancel(25) + memory_close.
        worst_case_step_sum = (
            5 * main_mod._CLEANUP_STEP_TIMEOUT
            + main_mod._HARNESS_CANCEL_TIMEOUT
            + main_mod._MEMORY_CLOSE_STEP_TIMEOUT
        )
        assert worst_case_step_sum <= main_mod._FORCE_EXIT_BUDGET

        # Regression: the mirrored/parity constants are UNCHANGED so the
        # schema.py _MIRROR_* guard and the systemd-template parity test
        # (both outside this task's scope) stay valid.
        assert main_mod._FORCE_EXIT_BUDGET == 75.0
        assert main_mod._SYSTEMD_TIMEOUT_STOP_SECS == 90.0


class _DummyASGIApp:
    """Minimal placeholder ASGI app. uvicorn.Config never calls it at
    construction time, so this only needs to satisfy the type — matches the
    `_RaisingApp` placeholder pattern in tests/server/test_asgi_exception_shield.py.
    """

    async def __call__(self, scope, receive, send):
        raise NotImplementedError


class TestBuildUvicornConfig:
    """Survey finding D1: _build_uvicorn_config is the single tested seam that
    routes both the primary and recon-report uvicorn.Config construction sites
    through ServerConfig.graceful_shutdown_timeout, so the internal graceful-shutdown
    wait is always bounded instead of defaulting to None (unbounded).
    """

    def test_graceful_shutdown_timeout_applied(self):
        config = _build_uvicorn_config(
            _DummyASGIApp(),
            host='127.0.0.1',
            port=8000,
            graceful_shutdown_timeout=10,
        )
        assert config.timeout_graceful_shutdown == 10

    def test_keepalive_timeout_applied_when_provided(self):
        config = _build_uvicorn_config(
            _DummyASGIApp(),
            host='127.0.0.1',
            port=8000,
            graceful_shutdown_timeout=10,
            keepalive_timeout=30,
        )
        assert config.timeout_keep_alive == 30

    def test_keepalive_timeout_left_at_uvicorn_default_when_omitted(self):
        """When keepalive_timeout is omitted, the helper must NOT force
        timeout_keep_alive — uvicorn's own default (5s) applies unchanged.
        Matches the pre-existing recon-report site, which never set
        timeout_keep_alive.
        """
        config = _build_uvicorn_config(
            _DummyASGIApp(),
            host='127.0.0.1',
            port=8000,
            graceful_shutdown_timeout=10,
        )
        assert config.timeout_keep_alive == 5  # uvicorn.Config's own default

    def test_host_and_port_pass_through(self):
        config = _build_uvicorn_config(
            _DummyASGIApp(),
            host='0.0.0.0',
            port=9123,
            graceful_shutdown_timeout=10,
        )
        assert config.host == '0.0.0.0'
        assert config.port == 9123
