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


class TestGracefulShutdownResilientToDrainError:
    @pytest.mark.asyncio
    async def test_shutdown_resilient_to_drain_error(self):
        """memory_service.close() must be called even if task_interceptor.drain() raises."""
        memory_service = MagicMock()
        memory_service.close = AsyncMock()

        task_interceptor = MagicMock()
        task_interceptor.drain = AsyncMock(side_effect=RuntimeError('drain failed'))

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=task_interceptor,
            harness_loop_task=None,
            recon_journal=None,
        )

        memory_service.close.assert_awaited_once()


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
