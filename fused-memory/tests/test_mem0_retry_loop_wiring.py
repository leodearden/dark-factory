"""``server/main.py`` runs the service's Mem0 retry pass on its own policy, and marks supersedes whenever a re-issue lands.

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
    """A structural stand-in for the members the retry loop, startup recovery and their supersession landing hook read."""

    def __init__(
        self, *, backlog_alarm_age_seconds: float = 3600, pass_error: Exception | None = None
    ) -> None:
        self.mem0_retry_policy = Mem0RetryPolicy(
            initial_delay_seconds=1,
            max_delay_seconds=2,
            poll_interval_seconds=0.01,
            batch_size=1,
            backlog_alarm_age_seconds=backlog_alarm_age_seconds,
        )
        self.pass_kwargs: list[dict[str, Any]] = []
        self.recover_kwargs: list[dict[str, Any]] = []
        self.passed = asyncio.Event()
        self.third_pass = asyncio.Event()
        self.backlog_checks = 0
        self.backlog_checked = asyncio.Event()
        self._pass_error = pass_error
        self.update_memory = AsyncMock(return_value={'status': 'updated', 'store': 'mem0'})

    async def retry_due_mem0_intents(self, **kwargs: Any) -> None:
        self.pass_kwargs.append(kwargs)
        self.passed.set()
        if len(self.pass_kwargs) >= 3:
            self.third_pass.set()
        if self._pass_error is not None:
            raise self._pass_error

    async def check_mem0_retry_backlog(self) -> list[str]:
        self.backlog_checks += 1
        self.backlog_checked.set()
        return []

    async def recover_mem0_intents(self, **kwargs: Any) -> dict[str, int]:
        self.recover_kwargs.append(kwargs)
        return {}


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


async def _run_loop_until(service: _RecordingService, event: asyncio.Event) -> None:
    """Fail unless ``event`` fires sooner than the default policy's first pass could, proving the loop read the service's own policy."""
    from fused_memory.server.main import _start_mem0_retry_loop

    loop = asyncio.get_running_loop()
    started = loop.time()
    task = _start_mem0_retry_loop(service)
    try:
        await event.wait()
        assert loop.time() - started < DEFAULT_MEM0_RETRY_POLICY.poll_interval_seconds / 2
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _first_pass(service: _RecordingService) -> dict[str, Any]:
    await _run_loop_until(service, service.passed)
    return service.pass_kwargs[0]


async def _assert_the_hook_marks_supersedes(service: _RecordingService, on_landed: Any) -> None:
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

        await _assert_the_hook_marks_supersedes(service, on_landed)


class TestTheBacklogCheckWiring:
    @pytest.mark.asyncio
    async def test_a_failing_retry_pass_does_not_stop_the_backlog_check(self):
        # An alarm age of 12 poll intervals puts the check on the poll interval.
        service = _RecordingService(
            backlog_alarm_age_seconds=0.12, pass_error=RuntimeError('journal locked')
        )

        await _run_loop_until(service, service.backlog_checked)

        assert service.backlog_checks >= 1

    @pytest.mark.asyncio
    async def test_the_backlog_is_checked_on_the_policys_check_interval_not_every_pass(self):
        service = _RecordingService(backlog_alarm_age_seconds=3600)

        await _run_loop_until(service, service.third_pass)

        assert service.backlog_checks == 0


class TestTheStartupRecoveryWiring:
    @pytest.mark.asyncio
    async def test_a_recovered_writes_supersedes_are_marked_when_it_lands(self):
        from fused_memory.server.main import _recover_mem0_intents_at_startup

        service = _RecordingService()

        await _recover_mem0_intents_at_startup(service)

        (recover_kwargs,) = service.recover_kwargs
        await _assert_the_hook_marks_supersedes(service, recover_kwargs['on_landed'])
