"""Tests for the _graceful_shutdown helper in fused_memory.server.main."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

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

        memory_service.close.assert_called_once()


class TestGracefulShutdownCancelsHarnessLoopTask:
    @pytest.mark.asyncio
    async def test_shutdown_cancels_harness_loop_task(self):
        """_graceful_shutdown must cancel the harness loop asyncio.Task."""
        memory_service = MagicMock()
        memory_service.close = AsyncMock()

        # Create a real asyncio Task wrapping an infinite sleep
        async def _infinite():
            await asyncio.sleep(9999)

        harness_loop_task = asyncio.ensure_future(_infinite())

        await _graceful_shutdown(
            memory_service=memory_service,
            task_interceptor=None,
            harness_loop_task=harness_loop_task,
            recon_journal=None,
        )

        assert harness_loop_task.cancelled()
