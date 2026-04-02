"""Tests for UsageGate — race condition fixes and robustness improvements."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import UsageGate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_gate(
    account_names: list[str],
    *,
    cost_store=None,
    wait_for_reset: bool = False,
    session_budget_usd: float | None = None,
) -> UsageGate:
    """Create a UsageGate with the given account names and auto-generated env vars.

    Tokens are injected via patch.dict during construction so _init_accounts()
    finds them.  After construction the gate's _accounts list holds AccountState
    objects with token='fake-token-<name>'.
    """
    acct_cfgs = []
    env_vars: dict[str, str] = {}
    for name in account_names:
        env_key = f'TEST_TOKEN_{name.upper().replace("-", "_")}'
        env_vars[env_key] = f'fake-token-{name}'
        acct_cfgs.append(AccountConfig(name=name, oauth_token_env=env_key))

    config = UsageCapConfig(
        accounts=acct_cfgs,
        wait_for_reset=wait_for_reset,
        session_budget_usd=session_budget_usd,
    )

    with patch.dict(os.environ, env_vars):
        gate = UsageGate(config, cost_store=cost_store)

    return gate


def make_mock_cost_store() -> AsyncMock:
    """Return an AsyncMock that satisfies the CostStore interface."""
    store = AsyncMock()
    store.save_account_event = AsyncMock(return_value=None)
    return store


def _fire_with_failing_create_task(
    gate: UsageGate,
    error_msg: str,
    *,
    close_coro: bool = True,
) -> tuple[MagicMock, list]:
    """Invoke gate._fire_cost_event with a mock loop whose create_task raises RuntimeError.

    Sets up a MagicMock loop whose create_task side_effect raises
    RuntimeError(error_msg), patches asyncio.get_running_loop to return
    that loop, then calls _fire_cost_event('acct-A', 'cap_hit', '{}').

    Returns (mock_loop, captured_coro) so callers can make specific assertions.
    When close_coro=True (default), the side_effect closes the coroutine before
    raising to suppress ResourceWarning about unawaited coroutines.
    """
    mock_loop = MagicMock()
    captured_coro: list = []

    def raising_create_task(coro, **kwargs):
        captured_coro.append(coro)
        if close_coro:
            coro.close()
        raise RuntimeError(error_msg)

    mock_loop.create_task.side_effect = raising_create_task

    with patch('shared.usage_gate.asyncio.get_running_loop', return_value=mock_loop):
        gate._fire_cost_event('acct-A', 'cap_hit', '{}')

    return mock_loop, captured_coro


# ---------------------------------------------------------------------------
# before_invoke race condition — _last_account_name updated before
#         the failover cost event fires; event uses _fire_cost_event not
#         await _write_cost_event.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBeforeInvokeRaceCondition:
    """_last_account_name is updated before the failover event fires."""

    async def test_last_account_name_updated_before_fire_cost_event(self):
        """_last_account_name must equal the NEW account name when _fire_cost_event is called.

        Verifies that _last_account_name is updated to acct.name BEFORE
        _fire_cost_event is called, so the event carries the new account name
        rather than the stale previous one.  Also verifies the details payload
        is json.dumps({'from': 'acct-A', 'to': 'acct-B'}).
        """
        import json

        # cost_store must be set so the `if self._cost_store:` guard in before_invoke
        # allows _fire_cost_event to be called.
        gate = make_gate(['acct-A', 'acct-B'], cost_store=make_mock_cost_store())

        # Simulate acct-A already used, now capped
        gate._accounts[0].capped = True
        gate._last_account_name = 'acct-A'

        captured_name_at_call: list[str | None] = []
        captured_details: list[str] = []

        def capture_name(account_name: str, event_type: str, details: str) -> None:
            # Record gate._last_account_name and the details arg at the moment the event fires
            captured_name_at_call.append(gate._last_account_name)
            captured_details.append(details)

        with patch.object(gate, '_fire_cost_event', side_effect=capture_name):
            token = await gate.before_invoke()

        assert token == 'fake-token-acct-B'
        # _fire_cost_event must have been called once
        assert len(captured_name_at_call) == 1, (
            '_fire_cost_event was not called — before_invoke still uses _write_cost_event'
        )
        # At call time, _last_account_name must already reflect the new account
        assert captured_name_at_call[0] == 'acct-B', (
            f'Expected acct-B but got {captured_name_at_call[0]!r} — '
            'race: _last_account_name not yet updated when event fired'
        )
        # The details payload must carry the correct from/to account names
        assert captured_details[0] == json.dumps({'from': 'acct-A', 'to': 'acct-B'}), (
            f'Expected details={json.dumps({"from": "acct-A", "to": "acct-B"})!r}, '
            f'got {captured_details[0]!r}'
        )

    async def test_failover_uses_fire_cost_event_not_write_cost_event(self):
        """before_invoke uses _fire_cost_event (fire-and-forget) not await _write_cost_event.

        Verifies the failover path calls the non-blocking _fire_cost_event
        instead of awaiting _write_cost_event, keeping the OAuth token return
        off the blocking critical path.
        """
        # cost_store must be set so the `if self._cost_store:` guard allows the event.
        gate = make_gate(['acct-A', 'acct-B'], cost_store=make_mock_cost_store())

        gate._accounts[0].capped = True
        gate._last_account_name = 'acct-A'

        with (
            patch.object(gate, '_fire_cost_event') as mock_fire,
            patch.object(gate, '_write_cost_event', new_callable=AsyncMock) as mock_write,
        ):
            await gate.before_invoke()

        mock_fire.assert_called_once()
        mock_write.assert_not_called()


# ---------------------------------------------------------------------------
# _fire_cost_event stores the Task to prevent GC; done-callback removes it
# from the set after completion.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFireCostEventTaskStorage:
    """_fire_cost_event stores the task reference in _background_tasks to prevent GC."""

    async def test_task_stored_immediately_after_fire(self):
        """Task is in gate._background_tasks right after _fire_cost_event returns.

        Verifies that the asyncio.Task created by _fire_cost_event is stored
        in _background_tasks immediately, preventing it from being garbage-collected
        before it completes.
        """
        gate = make_gate(['acct-A'], cost_store=make_mock_cost_store())

        gate._fire_cost_event('acct-A', 'cap_hit', '{"reason": "test"}')

        assert hasattr(gate, '_background_tasks'), (
            '_background_tasks set not found — add it in __init__'
        )
        assert len(gate._background_tasks) == 1, (
            f'Expected 1 task, found {len(gate._background_tasks)} — '
            'task reference is not being stored'
        )

    async def test_task_removed_after_completion(self):
        """After the task finishes, done_callback must remove it from _background_tasks."""
        gate = make_gate(['acct-A'], cost_store=make_mock_cost_store())

        gate._fire_cost_event('acct-A', 'cap_hit', '{"reason": "test"}')

        # Deterministically drain all in-flight tasks to completion.
        await asyncio.gather(*list(gate._background_tasks))

        assert len(gate._background_tasks) == 0, (
            f'Expected empty set after completion, but {len(gate._background_tasks)} task(s) remain — '
            'done_callback not calling discard()'
        )


# ---------------------------------------------------------------------------
# RuntimeError handling in _fire_cost_event is narrowed to only the
# get_running_loop() call; errors from create_task() are caught non-fatally.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFireCostEventRuntimeErrorHandling:
    """Only 'no running event loop' RuntimeError is caught; create_task errors are caught and logged (non-fatal)."""

    async def test_no_running_loop_logs_warning(self, caplog):
        """RuntimeError from get_running_loop() is caught and a warning is logged.

        Verifies that when asyncio.get_running_loop() raises RuntimeError (no
        event loop), _fire_cost_event logs a warning rather than silently
        swallowing the exception.
        """
        gate = make_gate(['acct-A'], cost_store=make_mock_cost_store())

        with (
            patch('shared.usage_gate.asyncio.get_running_loop',
                  side_effect=RuntimeError('no running event loop')),
            caplog.at_level(logging.WARNING, logger='shared.usage_gate'),
        ):
            gate._fire_cost_event('acct-A', 'cap_hit', '{}')

        assert any(
            'no running event loop for cost event' in record.message.lower()
            for record in caplog.records
        ), (
            f'Expected a warning log containing "no running event loop for cost event", '
            f'got: {[r.message for r in caplog.records]}'
        )

    async def test_create_task_error_is_non_fatal(self, caplog):
        """RuntimeError from loop.create_task() must be caught, NOT propagate.

        Post-review: _fire_cost_event is fire-and-forget telemetry on the
        before_invoke() critical path. A RuntimeError from create_task()
        (e.g. 'Event loop is closed' during shutdown) must never crash callers.
        The method must return normally after logging a warning.
        """
        gate = make_gate(['acct-A'], cost_store=make_mock_cost_store())

        # Must NOT raise — create_task error is non-fatal
        with caplog.at_level(logging.WARNING, logger='shared.usage_gate'):
            _mock_loop, captured_coro = _fire_with_failing_create_task(
                gate, 'scheduler is shutting down'
            )

        assert len(captured_coro) == 1, 'create_task should have been called once'

    async def test_create_task_error_logs_warning(self, caplog):
        """A warning must be logged when loop.create_task() raises RuntimeError.

        The log message must include the event_type, account_name, and the
        original exception: 'Failed to schedule cost event %s/%s: %s'.
        """
        gate = make_gate(['acct-A'], cost_store=make_mock_cost_store())

        with caplog.at_level(logging.WARNING, logger='shared.usage_gate'):
            _fire_with_failing_create_task(gate, 'event loop is closed')

        assert any(
            'Failed to schedule cost event' in record.message
            and 'cap_hit' in record.message
            and 'acct-A' in record.message
            for record in caplog.records
        ), (
            f'Expected warning with "Failed to schedule cost event cap_hit/acct-A", '
            f'got: {[r.message for r in caplog.records]}'
        )


# ---------------------------------------------------------------------------
# Coroutine leak — production code closes the coro when loop.create_task
# raises RuntimeError.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFireCostEventCoroutineLeak:
    """_fire_cost_event closes the coroutine when create_task raises to prevent a resource leak."""

    async def test_coroutine_closed_on_create_task_failure(self):
        """Production code must close the coroutine when loop.create_task raises RuntimeError.

        Unlike test_create_task_error_is_non_fatal (which has the mock close the
        coro before raising), this mock does NOT close the coro.  The test then
        checks that production code has closed it by asserting coro.cr_frame is
        None — a coroutine's frame pointer is set to None only after .close() or
        after the coroutine runs to completion.
        """
        gate = make_gate(['acct-A'], cost_store=make_mock_cost_store())

        mock_loop = MagicMock()
        captured_coro: list = []

        def raising_create_task(coro, **kwargs):
            # Do NOT close the coro here — let production code do it.
            captured_coro.append(coro)
            raise RuntimeError('event loop is closed')

        mock_loop.create_task.side_effect = raising_create_task

        with patch('shared.usage_gate.asyncio.get_running_loop', return_value=mock_loop):
            gate._fire_cost_event('acct-A', 'cap_hit', '{}')

        assert len(captured_coro) == 1, 'create_task should have been called once'
        coro = captured_coro[0]
        # Production code must have called coro.close(); after close(), cr_frame is None.
        assert coro.cr_frame is None, (
            'coro.cr_frame is not None — production code did not call coro.close() '
            'when loop.create_task raised RuntimeError (coroutine leak)'
        )


# ---------------------------------------------------------------------------
# json.dumps guard in _handle_cap_detected when cost_store is None.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHandleCapDetectedJsonDumpsGuard:
    """json.dumps and _fire_cost_event are not called when cost_store is None."""

    async def test_json_dumps_not_called_without_cost_store(self):
        """json.dumps is not called for the cap_hit event when cost_store=None.

        Verifies that the entire _fire_cost_event call (including its json.dumps
        argument) is guarded by `if self._cost_store:`, avoiding unnecessary
        serialization when no store is configured.
        """
        gate = make_gate(['acct-A'], cost_store=None)
        token = gate._accounts[0].token  # use valid token for _find_account_by_token

        with patch('shared.usage_gate.json.dumps') as mock_dumps:
            gate._handle_cap_detected('usage limit hit', None, token)

        mock_dumps.assert_not_called()

    async def test_fire_cost_event_not_called_without_cost_store(self):
        """_fire_cost_event must not be called when cost_store=None."""
        gate = make_gate(['acct-A'], cost_store=None)
        token = gate._accounts[0].token

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate._handle_cap_detected('usage limit hit', None, token)

        mock_fire.assert_not_called()


# ---------------------------------------------------------------------------
# json.dumps guard in _account_resume_probe_loop when cost_store is None.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAccountResumeProbeLoopJsonDumpsGuard:
    """_write_cost_event is not called in the probe loop when cost_store=None."""

    async def test_write_cost_event_not_called_without_cost_store(self):
        """_write_cost_event is not called when cost_store=None.

        Verifies that the _write_cost_event call in _account_resume_probe_loop
        is guarded by `if self._cost_store:`, so no call is made when no store
        is configured.
        """
        gate = make_gate(['acct-A'], cost_store=None)
        acct = gate._accounts[0]
        acct.capped = True
        # resets_at in the past → sleep_for=0, probe runs immediately
        acct.resets_at = datetime.now(UTC) - timedelta(minutes=5)

        with patch.object(gate, '_write_cost_event', new_callable=AsyncMock) as mock_write:
            await gate._account_resume_probe_loop(acct)

        mock_write.assert_not_called()


# ---------------------------------------------------------------------------
# shutdown() drains _background_tasks by cancelling and awaiting in-flight tasks.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestShutdownBackgroundTasks:
    """shutdown() cancels and awaits in-flight background tasks, leaving _background_tasks empty."""

    async def test_shutdown_cancels_and_awaits_background_tasks(self):
        """shutdown() cancels in-flight cost-event tasks and drains _background_tasks.

        Verifies that shutdown() iterates _background_tasks, cancels each task,
        and awaits them (suppressing CancelledError), leaving _background_tasks
        empty and preventing 'Task was destroyed but it is pending' warnings.
        """
        gate = make_gate(['acct-A'], cost_store=make_mock_cost_store())

        # Make the cost-store block so tasks stay in-flight when shutdown() is called.
        block_event = asyncio.Event()

        async def blocking_save(*args, **kwargs):
            await block_event.wait()

        gate._cost_store.save_account_event.side_effect = blocking_save

        # Fire two events — creates two tasks in _background_tasks.
        gate._fire_cost_event('acct-A', 'cap_hit', '{}')
        gate._fire_cost_event('acct-A', 'cap_hit', '{}')

        # Yield to the event loop so the tasks start running and block on blocking_save.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert len(gate._background_tasks) == 2, (
            f'Expected 2 in-flight tasks, got {len(gate._background_tasks)}'
        )

        # shutdown() must cancel and drain the in-flight tasks.
        await gate.shutdown()

        assert len(gate._background_tasks) == 0, (
            f'Expected empty _background_tasks after shutdown(), '
            f'got {len(gate._background_tasks)} remaining — '
            'shutdown() is not draining _background_tasks'
        )

    async def test_shutdown_with_empty_background_tasks(self):
        """shutdown() must succeed without error when _background_tasks is empty.

        Regression guard: the new draining loop must handle the empty-set case
        gracefully (iterating over an empty set is a no-op).
        """
        gate = make_gate(['acct-A'], cost_store=make_mock_cost_store())

        # No events fired → set is empty.
        assert len(gate._background_tasks) == 0

        # Must not raise.
        await gate.shutdown()

        assert len(gate._background_tasks) == 0


# ---------------------------------------------------------------------------
# before_invoke() guards the failover event with `if self._cost_store:`.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBeforeInvokeGuardConsistency:
    """Failover event in before_invoke() respects the cost_store guard."""

    async def test_before_invoke_failover_no_json_dumps_without_cost_store(self):
        """json.dumps and _fire_cost_event are NOT called when cost_store=None.

        Verifies that the entire failover-event block in before_invoke() is
        guarded by `if self._cost_store:`, matching the pattern in
        _handle_cap_detected and _account_resume_probe_loop.
        """
        gate = make_gate(['acct-A', 'acct-B'], cost_store=None)

        # Simulate acct-A already used, now capped — triggers failover to acct-B.
        gate._accounts[0].capped = True
        gate._last_account_name = 'acct-A'

        with (
            patch('shared.usage_gate.json.dumps') as mock_dumps,
            patch.object(gate, '_fire_cost_event') as mock_fire,
        ):
            token = await gate.before_invoke()

        assert token == 'fake-token-acct-B'
        mock_dumps.assert_not_called()
        mock_fire.assert_not_called()

    async def test_before_invoke_failover_fires_event_with_cost_store(self):
        """_fire_cost_event MUST be called during failover when cost_store is set.

        Positive control for the guard: when cost_store is not None, the failover
        event must still fire.
        """
        gate = make_gate(['acct-A', 'acct-B'], cost_store=make_mock_cost_store())

        # Simulate acct-A already used, now capped — triggers failover to acct-B.
        gate._accounts[0].capped = True
        gate._last_account_name = 'acct-A'

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            token = await gate.before_invoke()

        assert token == 'fake-token-acct-B'
        mock_fire.assert_called_once()
        call_args = mock_fire.call_args
        assert call_args[0][0] == 'acct-B'   # account_name
        assert call_args[0][1] == 'failover'  # event_type
