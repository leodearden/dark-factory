"""``server/main.py::_start_mem0_retry_loop`` runs the service's retry pass on the service's own policy.

As in ``tests/test_audit_trail_archive_wiring.py``, the wiring is proved by
driving what it built against a recording stub, not by a call-args assertion
on the wiring call.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from fused_memory.memory_metadata import SUPERSEDED_BY_METADATA_KEY
from fused_memory.services.mem0_add_outcome import Mem0AddOutcome
from fused_memory.services.mem0_write_retry import DEFAULT_MEM0_RETRY_POLICY, Mem0RetryPolicy

_OLD = 'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee'


class _RecordingService:
    """A structural stand-in for the members the retry loop and its supersession landing hook read."""

    def __init__(self) -> None:
        self.mem0_retry_policy = Mem0RetryPolicy(
            initial_delay_seconds=1,
            max_delay_seconds=2,
            poll_interval_seconds=0.01,
            batch_size=1,
        )
        self.pass_kwargs: list[dict[str, Any]] = []
        self.passed = asyncio.Event()
        self.update_memory = AsyncMock(return_value={'status': 'updated', 'store': 'mem0'})

    async def retry_due_mem0_intents(self, **kwargs: Any) -> None:
        self.pass_kwargs.append(kwargs)
        self.passed.set()


def _intent(metadata: dict[str, Any]) -> dict[str, Any]:
    """A ``mem0_intents`` row as the journal hands it back: metadata is a JSON column."""
    return {
        'id': 'intent-1',
        'write_op_id': 'op-1',
        'project_id': 'dark_factory',
        'agent_id': 'claude-task-9',
        'session_id': 'session-9',
        'causation_id': 'cause-9',
        'metadata': json.dumps(metadata),
        'status': 'retrying',
    }


async def _first_pass(service: _RecordingService) -> dict[str, Any]:
    from fused_memory.server.main import _start_mem0_retry_loop  # noqa: PLC0415

    task = _start_mem0_retry_loop(service)
    try:
        # noqa: wall-clock-deadline — the deadline is the assertion: a pass inside half the default policy's interval proves the loop read the service's own policy
        await asyncio.wait_for(
            service.passed.wait(), timeout=DEFAULT_MEM0_RETRY_POLICY.poll_interval_seconds / 2
        )
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    return service.pass_kwargs[0]


class TestTheRetryLoopWiring:
    @pytest.mark.asyncio
    async def test_a_pass_runs_on_the_services_own_poll_interval(self):
        service = _RecordingService()

        await _first_pass(service)

        assert service.pass_kwargs

    @pytest.mark.asyncio
    async def test_the_pass_marks_a_landed_writes_supersedes(self):
        service = _RecordingService()

        on_landed = (await _first_pass(service))['on_landed']
        await on_landed(
            _intent({'supersedes': [_OLD]}),
            Mem0AddOutcome(memory_ids=('new-id',), empty_result_error=None),
        )

        service.update_memory.assert_awaited_once()
        await_args = service.update_memory.await_args
        assert await_args is not None
        kwargs = await_args.kwargs
        assert kwargs['memory_id'] == _OLD
        assert kwargs['metadata_patch'] == {SUPERSEDED_BY_METADATA_KEY: 'new-id'}
