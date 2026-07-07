"""Tests for UsageGate — race condition fixes and robustness improvements."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import (
    _LEGAL_TRANSITIONS,
    AccountPhase,
    AccountState,
    IllegalTransitionError,
    UsageGate,
)

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

    # Mock _run_probe to prevent real subprocess spawning in tests
    gate._run_probe = AsyncMock(return_value=True)
    return gate


def make_mock_cost_store() -> AsyncMock:
    """Return an AsyncMock that satisfies the CostStore interface."""
    store = AsyncMock()
    store.save_account_event = AsyncMock(return_value=None)
    return store


# ---------------------------------------------------------------------------
# step-1: before_invoke race condition — _last_account_name updated before
#         the failover cost event fires; event uses _fire_cost_event not
#         await _write_cost_event.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBeforeInvokeRaceCondition:
    """step-1: _last_account_name updated before failover event fires."""

    async def test_last_account_name_updated_before_fire_cost_event(self):
        """_last_account_name must equal the NEW account name when _fire_cost_event is called.

        Race condition: currently _write_cost_event is awaited BEFORE
        _last_account_name is updated, so the event fires with the stale
        (old) account name still in self._last_account_name.

        After the fix, _last_account_name is set to acct.name BEFORE
        _fire_cost_event is called.
        """
        # cost_store must be set so the `if self._cost_store:` guard in before_invoke
        # allows _fire_cost_event to be called (task-355 step-8 guard).
        gate = make_gate(['acct-A', 'acct-B'], cost_store=make_mock_cost_store())

        # Simulate acct-A already used, now capped
        gate._accounts[0].capped = True
        gate._last_account_name = 'acct-A'

        captured_name_at_call: list[str | None] = []

        def capture_name(account_name: str, event_type: str, details: str) -> None:
            # Record gate._last_account_name at the moment the event fires
            captured_name_at_call.append(gate._last_account_name)

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

    async def test_failover_uses_fire_cost_event_not_write_cost_event(self):
        """before_invoke must use _fire_cost_event (fire-and-forget) not await _write_cost_event.

        Currently before_invoke does `await self._write_cost_event(...)` which
        blocks the critical-path return of the OAuth token.  After the fix it
        must call `self._fire_cost_event(...)` (non-blocking) instead.
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
# step-3: _fire_cost_event stores the Task to prevent GC; done-callback
#         removes it from the set after completion.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFireCostEventTaskStorage:
    """step-3: _fire_cost_event stores task reference in _background_tasks."""

    async def test_task_stored_immediately_after_fire(self):
        """Task must be in gate._background_tasks right after _fire_cost_event returns.

        Currently loop.create_task() result is discarded, so _background_tasks
        either doesn't exist or is empty.
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

        # Give the event loop a chance to run the coroutine to completion
        await asyncio.sleep(0)
        await asyncio.sleep(0)  # two yields to ensure done-callback fires

        assert len(gate._background_tasks) == 0, (
            f'Expected empty set after completion, but {len(gate._background_tasks)} task(s) remain — '
            'done_callback not calling discard()'
        )


# ---------------------------------------------------------------------------
# step-5: RuntimeError handling in _fire_cost_event is narrowed to only
#         the get_running_loop() call; errors from create_task() propagate.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFireCostEventRuntimeErrorHandling:
    """step-5: only 'no running event loop' RuntimeError is caught; create_task errors propagate."""

    async def test_no_running_loop_logs_warning(self, caplog):
        """RuntimeError from get_running_loop() must be caught and a warning logged.

        Currently the broad except swallows it silently (no log). After the fix,
        a logger.warning is emitted with the event type and account name.
        """
        gate = make_gate(['acct-A'], cost_store=make_mock_cost_store())

        with (
            patch('shared.usage_gate.asyncio.get_running_loop',
                  side_effect=RuntimeError('no running event loop')),
            caplog.at_level(logging.WARNING, logger='shared.usage_gate'),
        ):
            gate._fire_cost_event('acct-A', 'cap_hit', '{}')

        assert any(
            'no running event loop' in record.message.lower()
            or 'cap_hit' in record.message
            or 'acct-A' in record.message
            for record in caplog.records
        ), (
            f'Expected a warning log for missing event loop, got: {[r.message for r in caplog.records]}'
        )

    async def test_create_task_error_is_non_fatal(self, caplog):
        """RuntimeError from loop.create_task() must be caught, NOT propagate.

        Post-review: _fire_cost_event is fire-and-forget telemetry on the
        before_invoke() critical path. A RuntimeError from create_task()
        (e.g. 'Event loop is closed' during shutdown) must never crash callers.
        The method must return normally after logging a warning.
        """
        gate = make_gate(['acct-A'], cost_store=make_mock_cost_store())

        mock_loop = MagicMock()

        captured_coro: list = []

        def raising_create_task(coro, **kwargs):
            # Close the coroutine to avoid ResourceWarning about unawaited coro
            captured_coro.append(coro)
            coro.close()
            raise RuntimeError('scheduler is shutting down')

        mock_loop.create_task.side_effect = raising_create_task

        # Must NOT raise — create_task error is non-fatal
        with (
            patch('shared.usage_gate.asyncio.get_running_loop', return_value=mock_loop),
            caplog.at_level(logging.WARNING, logger='shared.usage_gate'),
        ):
            gate._fire_cost_event('acct-A', 'cap_hit', '{}')  # no exception raised

        assert len(captured_coro) == 1, 'create_task should have been called once'

    async def test_create_task_error_logs_warning(self, caplog):
        """A warning must be logged when loop.create_task() raises RuntimeError.

        The log message must include the event_type, account_name, and the
        original exception: 'Failed to schedule cost event %s/%s: %s'.
        """
        gate = make_gate(['acct-A'], cost_store=make_mock_cost_store())

        mock_loop = MagicMock()

        def raising_create_task(coro, **kwargs):
            coro.close()
            raise RuntimeError('event loop is closed')

        mock_loop.create_task.side_effect = raising_create_task

        with (
            patch('shared.usage_gate.asyncio.get_running_loop', return_value=mock_loop),
            caplog.at_level(logging.WARNING, logger='shared.usage_gate'),
        ):
            gate._fire_cost_event('acct-A', 'cap_hit', '{}')

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
# task-355 step-1: coroutine leak — production code must close the coro when
#                  loop.create_task raises RuntimeError.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFireCostEventCoroutineLeak:
    """task-355 step-1: _fire_cost_event must close coroutine when create_task raises."""

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
# step-7: json.dumps guard in _handle_cap_detected when cost_store is None.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHandleCapDetectedJsonDumpsGuard:
    """step-7: json.dumps and _fire_cost_event not called when cost_store is None."""

    async def test_json_dumps_not_called_without_cost_store(self):
        """json.dumps must not be called for the cap_hit event when cost_store=None.

        Currently line 264 unconditionally calls json.dumps({'reason': reason})
        as an argument to _fire_cost_event, even though _fire_cost_event would
        return immediately (cost_store is None).  After the fix, the entire
        _fire_cost_event call is wrapped in `if self._cost_store:`.
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
# step-9: json.dumps guard in _account_resume_probe_loop when cost_store is None.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAccountResumeProbLoopJsonDumpsGuard:
    """step-9: _write_cost_event must not be called in probe loop when cost_store=None."""

    async def test_write_cost_event_not_called_without_cost_store(self):
        """_write_cost_event must not be called when cost_store=None.

        Currently line 410 calls `await self._write_cost_event(...)` with
        json.dumps unconditionally, even when cost_store=None (_write_cost_event
        returns early, but it's still called).  After the fix, the call is
        wrapped in `if self._cost_store:`.
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
# task-355 step-3 & step-4: shutdown() must drain _background_tasks.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestShutdownBackgroundTasks:
    """task-355 steps 3+4: shutdown() must cancel and await in-flight background tasks."""

    async def test_shutdown_cancels_and_awaits_background_tasks(self):
        """shutdown() must cancel in-flight cost-event tasks and drain _background_tasks.

        Currently shutdown() only cancels per-account resume probe tasks and
        ignores _background_tasks.  In-flight tasks remain pending after shutdown,
        producing 'Task was destroyed but it is pending' warnings.

        After the fix, shutdown() iterates _background_tasks, cancels each, and
        awaits them (suppressing CancelledError), leaving _background_tasks empty.
        """
        gate = make_gate(['acct-A'], cost_store=make_mock_cost_store())

        # Make the cost-store block so tasks stay in-flight when shutdown() is called.
        block_event = asyncio.Event()

        async def blocking_save(*args, **kwargs):
            await block_event.wait()

        # gate._cost_store is CostStore | None; the fixture injects an
        # AsyncMock.  Assert + suppress the MagicMock.side_effect attribute
        # assignment against the bound-method type.
        assert gate._cost_store is not None
        gate._cost_store.save_account_event.side_effect = blocking_save  # type: ignore[attr-defined]

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
# task-355 step-6 & step-7: before_invoke() must guard the failover event
#                            with `if self._cost_store:`.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBeforeInvokeGuardConsistency:
    """task-355 steps 6+7: failover event in before_invoke() must respect cost_store guard."""

    async def test_before_invoke_failover_no_json_dumps_without_cost_store(self):
        """json.dumps and _fire_cost_event must NOT be called when cost_store=None.

        Currently lines 187-191 call json.dumps({'from': ..., 'to': ...}) as an
        argument to _fire_cost_event unconditionally, even when cost_store=None.
        _fire_cost_event would early-return, but json.dumps is still evaluated.

        After the fix, the entire block is wrapped in `if self._cost_store:`,
        matching the pattern in _handle_cap_detected and _account_resume_probe_loop.
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


# ---------------------------------------------------------------------------
# UsageGate probe process-group tests (step-23 / step-24)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestUsageGateProbeProcessGroup:
    """_run_probe must spawn its subprocess in a fresh process group."""

    @pytest.fixture
    def gate_with_account(self) -> tuple[UsageGate, AccountState]:
        """Return a (gate, acct) pair with _probe_config_dir mocked."""
        gate = make_gate(['probe-acct'])
        # Replace the mock inserted by make_gate with the real method for testing.
        del gate._run_probe  # remove instance-level mock; fall back to class method
        acct = gate._accounts[0]
        # Stub the probe config dir to avoid real filesystem side-effects.
        gate._probe_config_dir = MagicMock()
        gate._probe_config_dir.path = Path('/tmp/probe-test')
        gate._probe_config_dir.write_credentials = MagicMock()
        return gate, acct

    async def test_usage_gate_probe_passes_start_new_session_true(
        self, gate_with_account: tuple[UsageGate, AccountState]
    ) -> None:
        """create_subprocess_exec is called with start_new_session=True.

        Failing test — _run_probe does not pass that kwarg yet.
        """
        gate, acct = gate_with_account
        captured_kwargs: dict = {}

        async def fake_exec(*args: object, **kwargs: object) -> MagicMock:
            captured_kwargs.update(kwargs)
            proc = MagicMock()
            proc.returncode = 0
            proc.communicate = AsyncMock(return_value=(b'{"ok": true}', b''))
            return proc

        with patch('shared.usage_gate.asyncio.create_subprocess_exec',
                   side_effect=fake_exec):
            await gate._run_probe(acct)

        assert captured_kwargs.get('start_new_session') is True, (
            'create_subprocess_exec must be called with start_new_session=True'
        )

    async def test_usage_gate_probe_cleanup_uses_terminate_process_group(
        self, gate_with_account: tuple[UsageGate, AccountState]
    ) -> None:
        """TimeoutError branch must await terminate_process_group, not bare kill.

        Failing test — shared.usage_gate does not import terminate_process_group yet.
        """
        gate, acct = gate_with_account

        proc = MagicMock()
        proc.returncode = None
        proc.communicate = AsyncMock(side_effect=TimeoutError())
        proc.kill = MagicMock()
        proc.wait = AsyncMock()

        with (
            patch('shared.usage_gate.asyncio.create_subprocess_exec',
                  return_value=proc),
            patch('shared.usage_gate.terminate_process_group',
                  new_callable=AsyncMock) as mock_tpg,
        ):
            result = await gate._run_probe(acct)

        assert result is False  # timed out → False
        mock_tpg.assert_awaited_once()


# ---------------------------------------------------------------------------
# _run_probe exit-code classification — distinguish local budget cap
# (API accepted the request) from real Anthropic-side failures.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestUsageGateProbeExitCodeClassification:
    """_run_probe must NOT treat local $0.01 budget exhaustion as a cap."""

    @pytest.fixture
    def gate_with_account(self) -> tuple[UsageGate, AccountState]:
        """Return a (gate, acct) pair with _probe_config_dir mocked."""
        gate = make_gate(['probe-acct'])
        del gate._run_probe  # fall back to class method
        acct = gate._accounts[0]
        gate._probe_config_dir = MagicMock()
        gate._probe_config_dir.path = Path('/tmp/probe-test')
        gate._probe_config_dir.write_credentials = MagicMock()
        return gate, acct

    async def test_probe_local_budget_exhaustion_returns_true(
        self, gate_with_account: tuple[UsageGate, AccountState]
    ) -> None:
        """Non-zero exit + subtype=error_max_budget_usd → success (API accepted)."""
        gate, acct = gate_with_account
        stdout = (
            b'{"subtype":"error_max_budget_usd",'
            b'"total_cost_usd":0.053,'
            b'"errors":["Reached maximum budget ($0.01)"]}'
        )
        proc = MagicMock()
        proc.returncode = 1
        proc.pid = 12345
        proc.communicate = AsyncMock(return_value=(stdout, b''))

        with patch('shared.usage_gate.asyncio.create_subprocess_exec',
                   return_value=proc):
            result = await gate._run_probe(acct)

        assert result is True

    async def test_probe_non_budget_error_still_returns_false(
        self, gate_with_account: tuple[UsageGate, AccountState]
    ) -> None:
        """Non-zero exit + any other subtype → failure (unchanged behavior)."""
        gate, acct = gate_with_account
        stdout = b'{"subtype":"error_api","errors":["auth failed"]}'
        proc = MagicMock()
        proc.returncode = 1
        proc.pid = 12345
        proc.communicate = AsyncMock(return_value=(stdout, b''))

        with patch('shared.usage_gate.asyncio.create_subprocess_exec',
                   return_value=proc):
            result = await gate._run_probe(acct)

        assert result is False

    async def test_probe_cap_prefix_wins_over_budget_json(
        self, gate_with_account: tuple[UsageGate, AccountState]
    ) -> None:
        """A cap prefix in stderr forces False even if stdout has budget JSON.

        Preserves the conservative bias documented at usage_gate._run_probe —
        any whiff of a real cap message keeps the account paused.
        """
        gate, acct = gate_with_account
        stderr = b"You've hit your 5-hour usage limit. Resets at 3pm."
        stdout = (
            b'{"subtype":"error_max_budget_usd",'
            b'"total_cost_usd":0.053,'
            b'"errors":["Reached maximum budget ($0.01)"]}'
        )
        proc = MagicMock()
        proc.returncode = 1
        proc.pid = 12345
        proc.communicate = AsyncMock(return_value=(stdout, stderr))

        with patch('shared.usage_gate.asyncio.create_subprocess_exec',
                   return_value=proc):
            result = await gate._run_probe(acct)

        assert result is False


# ---------------------------------------------------------------------------
# wait_for_open — used by curator worker for AllAccountsCappedException defer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWaitForOpen:
    """Used by :meth:`fused_memory.middleware.task_interceptor.TaskInterceptor._curator_worker`
    to defer a whole batch when ``invoke_with_cap_retry`` raises
    :exc:`AllAccountsCappedException`.  Returns True when at least one
    account becomes available; False on timeout."""

    async def test_returns_true_immediately_when_open(self):
        """Fast path: not currently paused → returns True without sleeping."""
        gate = make_gate(['A'])
        # Default: account is not capped → gate is open.
        assert gate.is_paused is False
        assert await gate.wait_for_open(timeout=0.5) is True

    async def test_returns_true_after_account_uncaps_via_event(self):
        """Cap → other coroutine clears cap and sets _open → wait_for_open
        returns True."""
        gate = make_gate(['A'])
        acct = gate._accounts[0]
        acct.capped = True
        gate._open.clear()
        assert gate.is_paused is True

        async def uncap_after_delay():
            await asyncio.sleep(0.05)
            acct.capped = False
            gate._open.set()

        task = asyncio.create_task(uncap_after_delay())
        try:
            ok = await gate.wait_for_open(timeout=1.0)
        finally:
            await task
        assert ok is True

    async def test_returns_false_on_timeout_when_still_capped(self):
        """All-capped + no reset → wait_for_open returns False after timeout."""
        gate = make_gate(['A'])
        acct = gate._accounts[0]
        acct.capped = True
        # Push reset far into the future so _refresh_capped_accounts won't
        # flip it during the wait.
        acct.resets_at = datetime.now(UTC) + timedelta(hours=1)
        gate._open.clear()

        ok = await gate.wait_for_open(timeout=0.05)
        assert ok is False

    async def test_returns_true_when_reset_already_passed(self):
        """If an account's ``resets_at`` is in the past at call time,
        ``_refresh_capped_accounts`` flips it to probing and we open up."""
        gate = make_gate(['A'])
        acct = gate._accounts[0]
        acct.capped = True
        acct.resets_at = datetime.now(UTC) - timedelta(seconds=1)
        gate._open.clear()

        ok = await gate.wait_for_open(timeout=0.5)
        assert ok is True
        assert acct.capped is False  # refreshed → uncapped (probing)

    async def test_no_accounts_configured_is_treated_as_open(self):
        """No-account mode (e.g. tests) returns True immediately; the gate
        cannot be paused if it has no accounts."""
        gate = make_gate([])  # no accounts configured
        assert await gate.wait_for_open(timeout=0.05) is True


# ---------------------------------------------------------------------------
# soonest_resets_at property
# ---------------------------------------------------------------------------


class TestSoonestResetsAt:
    """UsageGate.soonest_resets_at returns the earliest resets_at across capped
    accounts, or None when that information is unavailable."""

    def test_returns_none_when_no_account_is_capped(self):
        """When no account is capped the property returns None."""
        gate = make_gate(['A', 'B'])
        # Default state: no account is capped.
        assert gate.soonest_resets_at is None

    def test_returns_earliest_resets_at_across_capped_accounts(self):
        """With two capped accounts, returns the minimum resets_at."""
        gate = make_gate(['A', 'B'])
        earlier = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        later = datetime(2025, 1, 1, 14, 0, 0, tzinfo=UTC)

        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = later
        gate._accounts[1].capped = True
        gate._accounts[1].resets_at = earlier

        assert gate.soonest_resets_at == earlier

    def test_ignores_non_capped_accounts_and_capped_with_none_resets_at(self):
        """Non-capped accounts and capped accounts with resets_at=None are ignored;
        only capped accounts with a known resets_at contribute."""
        gate = make_gate(['A', 'B', 'C'])
        known = datetime(2025, 6, 15, 8, 0, 0, tzinfo=UTC)

        # A: capped, resets_at known
        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = known
        # B: capped, resets_at unknown (None)
        gate._accounts[1].capped = True
        gate._accounts[1].resets_at = None
        # C: not capped, has a resets_at (should be ignored)
        gate._accounts[2].capped = False
        gate._accounts[2].resets_at = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)

        assert gate.soonest_resets_at == known

    def test_returns_none_when_all_capped_accounts_have_unknown_resets_at(self):
        """If every capped account has resets_at=None the property returns None."""
        gate = make_gate(['A', 'B'])
        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = None
        gate._accounts[1].capped = True
        gate._accounts[1].resets_at = None

        assert gate.soonest_resets_at is None


# ---------------------------------------------------------------------------
# step-01/02: AccountPhase scaffolding — explicit `phase` field on
# AccountState + a legacy-compat property shim for the old boolean flags
# (capped/probing/probe_in_flight/auth_failed), plus the UsageGate
# `_shutting_down` guard field.  AccountPhase / IllegalTransitionError are
# imported *inside* each test (not at module scope) since they do not exist
# yet on the base branch — this keeps the rest of the file collectible while
# these specific tests fail red.
# ---------------------------------------------------------------------------


class TestAccountPhaseScaffolding:
    """step-01: AccountPhase(StrEnum) + IllegalTransitionError + compat shim."""

    def test_account_phase_importable_with_exact_members(self):
        from shared.usage_gate import AccountPhase

        assert {m.name for m in AccountPhase} == {
            'AVAILABLE', 'PROBING', 'PROBE_IN_FLIGHT', 'CAPPED', 'AUTH_FAILED',
        }

    def test_illegal_transition_error_importable(self):
        from shared.usage_gate import IllegalTransitionError

        assert issubclass(IllegalTransitionError, Exception)

    def test_fresh_account_defaults_to_available_phase(self):
        from shared.usage_gate import AccountPhase

        acct = AccountState(name='acct-A', token='tok')
        assert acct.phase == AccountPhase.AVAILABLE
        assert acct.capped is False
        assert acct.probing is False
        assert acct.probe_in_flight is False
        assert acct.auth_failed is False
        assert acct.near_cap is False

    def test_capped_setter_maps_to_phase(self):
        from shared.usage_gate import AccountPhase

        acct = AccountState(name='acct-A', token='tok')
        acct.capped = True
        assert acct.phase == AccountPhase.CAPPED
        assert acct.capped is True
        assert acct.probing is False
        assert acct.probe_in_flight is False
        assert acct.auth_failed is False

    def test_probing_setter_maps_to_phase(self):
        from shared.usage_gate import AccountPhase

        acct = AccountState(name='acct-A', token='tok')
        acct.probing = True
        assert acct.phase == AccountPhase.PROBING
        assert acct.probing is True
        assert acct.capped is False
        assert acct.probe_in_flight is False
        assert acct.auth_failed is False

    def test_probe_in_flight_setter_maps_to_phase(self):
        from shared.usage_gate import AccountPhase

        acct = AccountState(name='acct-A', token='tok')
        acct.probe_in_flight = True
        assert acct.phase == AccountPhase.PROBE_IN_FLIGHT
        assert acct.probe_in_flight is True
        assert acct.capped is False
        assert acct.probing is False
        assert acct.auth_failed is False

    def test_auth_failed_setter_maps_to_phase(self):
        from shared.usage_gate import AccountPhase

        acct = AccountState(name='acct-A', token='tok')
        acct.auth_failed = True
        assert acct.phase == AccountPhase.AUTH_FAILED
        assert acct.auth_failed is True
        assert acct.capped is False
        assert acct.probing is False
        assert acct.probe_in_flight is False

    def test_mutual_exclusion_by_construction(self):
        """Setting capped=True then auth_failed=True leaves only auth_failed True.

        Guards against the pre-refactor bug class where independent boolean
        fields could both be True simultaneously.
        """
        acct = AccountState(name='acct-A', token='tok')
        acct.capped = True
        acct.auth_failed = True
        assert acct.auth_failed is True
        assert acct.capped is False

    def test_setter_clear_of_non_current_flag_is_noop(self):
        """Clearing a flag that does NOT match the current phase is a no-op."""
        from shared.usage_gate import AccountPhase

        acct = AccountState(name='acct-A', token='tok')
        acct.probe_in_flight = True
        acct.probing = False  # probing is not the current phase -> no-op
        assert acct.phase == AccountPhase.PROBE_IN_FLIGHT
        assert acct.probe_in_flight is True

    def test_setter_clear_of_current_flag_resets_to_available(self):
        """Clearing the flag matching the CURRENT phase resets to AVAILABLE."""
        from shared.usage_gate import AccountPhase

        acct = AccountState(name='acct-A', token='tok')
        acct.capped = True
        acct.capped = False
        assert acct.phase == AccountPhase.AVAILABLE

    def test_near_cap_is_independent_of_phase_and_bool_setters(self):
        from shared.usage_gate import AccountPhase

        acct = AccountState(name='acct-A', token='tok')
        acct.near_cap = True
        assert acct.phase == AccountPhase.AVAILABLE
        assert acct.near_cap is True

        # The legacy bool setters must not touch near_cap.
        acct.capped = True
        assert acct.near_cap is True

    def test_usage_gate_has_shutting_down_flag_defaulting_false(self):
        gate = make_gate(['acct-A'])
        assert gate._shutting_down is False


# ---------------------------------------------------------------------------
# step-03/04: UsageGate._transition — sole-writer phase mutation + centralized
# _open recompute (DD-5: gate._open.is_set() <=> any account in
# {AVAILABLE, PROBING}). No legality checking yet (added step-05/06); every
# edge exercised here is legal per the eventual _LEGAL_TRANSITIONS table so
# these assertions keep holding once legality lands.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTransitionCore:
    """step-03: _transition phase-write + centralized _open recompute (DD-5)."""

    async def test_transition_to_capped_clears_open_and_near_cap(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.near_cap = True  # AVAILABLE -> CAPPED is a legal edge

        gate._transition(acct, AccountPhase.CAPPED)

        assert acct.phase == AccountPhase.CAPPED
        assert acct.near_cap is False
        assert gate._open.is_set() is False

    async def test_transition_to_probing_opens_gate(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.CAPPED
        gate._open.clear()

        gate._transition(acct, AccountPhase.PROBING)  # CAPPED -> PROBING

        assert acct.phase == AccountPhase.PROBING
        assert gate._open.is_set() is True

    async def test_transition_to_probe_in_flight_closes_gate(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.PROBING

        gate._transition(acct, AccountPhase.PROBE_IN_FLIGHT)  # PROBING -> PROBE_IN_FLIGHT

        assert acct.phase == AccountPhase.PROBE_IN_FLIGHT
        assert gate._open.is_set() is False

    async def test_transition_to_available_opens_gate(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.PROBE_IN_FLIGHT
        gate._open.clear()

        gate._transition(acct, AccountPhase.AVAILABLE)  # PROBE_IN_FLIGHT -> AVAILABLE

        assert acct.phase == AccountPhase.AVAILABLE
        assert gate._open.is_set() is True

    async def test_multi_account_open_stays_set_when_sibling_available(self):
        """Capping account A must not close the gate while B is still
        AVAILABLE — the centralized recompute considers ALL accounts, not
        just the one just mutated (unlike the old unconditional _open.clear()
        sprinkled across ~10 call sites)."""
        gate = make_gate(['acct-A', 'acct-B'], wait_for_reset=False, cost_store=None)
        a, b = gate._accounts

        gate._transition(a, AccountPhase.CAPPED)  # AVAILABLE -> CAPPED

        assert a.phase == AccountPhase.CAPPED
        assert b.phase == AccountPhase.AVAILABLE
        assert gate._open.is_set() is True


# ---------------------------------------------------------------------------
# step-05/06: _transition legality — _LEGAL_TRANSITIONS table (PRD §7.3),
# raise-before-mutate on illegal edges (B1), and the force=True escape hatch
# reserved for _on_sighup_async's operator-driven hard reset.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTransitionLegality:
    """step-05: illegal edges raise IllegalTransitionError; state unchanged."""

    async def test_illegal_edge_raises_and_state_unchanged(self):
        """AUTH_FAILED -> PROBE_IN_FLIGHT is not in the legal-transitions table."""
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.auth_failed = True
        open_before = gate._open.is_set()

        with pytest.raises(IllegalTransitionError):
            gate._transition(acct, AccountPhase.PROBE_IN_FLIGHT)

        assert acct.phase == AccountPhase.AUTH_FAILED
        assert gate._open.is_set() == open_before

    async def test_capped_to_available_without_force_raises(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.CAPPED

        with pytest.raises(IllegalTransitionError):
            gate._transition(acct, AccountPhase.AVAILABLE)

        assert acct.phase == AccountPhase.CAPPED

    async def test_capped_self_loop_raises(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.CAPPED

        with pytest.raises(IllegalTransitionError):
            gate._transition(acct, AccountPhase.CAPPED)

        assert acct.phase == AccountPhase.CAPPED

    async def test_available_to_probing_raises(self):
        """AVAILABLE may only go to CAPPED or AUTH_FAILED — PROBING is only
        reachable from CAPPED (post-reset) or PROBE_IN_FLIGHT (demotion)."""
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        assert acct.phase == AccountPhase.AVAILABLE

        with pytest.raises(IllegalTransitionError):
            gate._transition(acct, AccountPhase.PROBING)

        assert acct.phase == AccountPhase.AVAILABLE

    async def test_illegal_edge_logs_error(self, caplog):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.CAPPED

        with (
            caplog.at_level(logging.ERROR, logger='shared.usage_gate'),
            pytest.raises(IllegalTransitionError),
        ):
            gate._transition(acct, AccountPhase.AVAILABLE)

        assert any(
            'capped' in record.message.lower()
            and 'available' in record.message.lower()
            and acct.name in record.message
            for record in caplog.records
        ), (
            f'Expected a logger.error naming the illegal edge, '
            f'got: {[r.message for r in caplog.records]}'
        )

    async def test_force_bypasses_legality_check(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.CAPPED
        gate._open.clear()

        gate._transition(acct, AccountPhase.AVAILABLE, force=True)  # must not raise

        assert acct.phase == AccountPhase.AVAILABLE
        assert gate._open.is_set() is True

    async def test_legal_edges_from_transition_core_still_do_not_raise(self):
        """Regression guard: the full legal chain from TestTransitionCore
        (AVAILABLE -> CAPPED -> PROBING -> PROBE_IN_FLIGHT -> AVAILABLE)
        must not raise now that legality is enforced."""
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]

        gate._transition(acct, AccountPhase.CAPPED)
        gate._transition(acct, AccountPhase.PROBING)
        gate._transition(acct, AccountPhase.PROBE_IN_FLIGHT)
        gate._transition(acct, AccountPhase.AVAILABLE)

        assert acct.phase == AccountPhase.AVAILABLE


# ---------------------------------------------------------------------------
# step-07/08: _transition owns per-phase side effects — background-task
# lifecycle (start resume/reprobe task on entering CAPPED/AUTH_FAILED, stamp
# pause_started_at/auth_failed_at) and the B10 shutdown guard (no new
# probe/reprobe task starts while gate._shutting_down is True).
# ---------------------------------------------------------------------------


async def _drain_task(task: asyncio.Task | None) -> None:
    """Cancel *task* and await its settling, so a background task started by
    _transition doesn't leak past the end of the current test."""
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
class TestTransitionSideEffectsLifecycle:
    """step-07: _transition starts the per-phase background task and stamps
    the matching timestamp; _shutting_down suppresses both starts (B10)."""

    async def test_transition_to_capped_starts_resume_task_and_stamps_pause(self):
        gate = make_gate(['acct-A'], wait_for_reset=True, cost_store=None)
        acct = gate._accounts[0]
        assert acct.resume_task is None
        assert acct.pause_started_at is None

        gate._transition(acct, AccountPhase.CAPPED)

        assert acct.resume_task is not None
        assert acct.pause_started_at is not None
        await _drain_task(acct.resume_task)

    async def test_transition_to_auth_failed_starts_reprobe_task_and_stamps_auth_failed_at(self):
        gate = make_gate(['acct-A'], wait_for_reset=True, cost_store=None)
        acct = gate._accounts[0]
        assert acct.auth_reprobe_task is None
        assert acct.auth_failed_at is None

        gate._transition(acct, AccountPhase.AUTH_FAILED)

        assert acct.auth_reprobe_task is not None
        assert acct.auth_failed_at is not None
        await _drain_task(acct.auth_reprobe_task)

    async def test_transition_to_capped_cancels_pending_auth_reprobe_task(self):
        """AUTH_FAILED -> CAPPED (the sole demotion edge) must cancel the
        now-stale auth_reprobe_task rather than leave it running."""
        gate = make_gate(['acct-A'], wait_for_reset=True, cost_store=None)
        acct = gate._accounts[0]
        gate._transition(acct, AccountPhase.AUTH_FAILED)
        stale_reprobe_task = acct.auth_reprobe_task
        assert stale_reprobe_task is not None

        gate._transition(acct, AccountPhase.CAPPED)

        await asyncio.sleep(0)
        assert stale_reprobe_task.cancelled() or stale_reprobe_task.done()
        await _drain_task(acct.resume_task)

    async def test_shutting_down_blocks_new_resume_task_on_capped(self):
        """B10: _shutting_down=True => a CapHit transition spawns no probe task."""
        gate = make_gate(['acct-A'], wait_for_reset=True, cost_store=None)
        acct = gate._accounts[0]
        gate._shutting_down = True

        gate._transition(acct, AccountPhase.CAPPED)

        assert acct.phase == AccountPhase.CAPPED
        assert acct.resume_task is None

    async def test_shutting_down_blocks_new_reprobe_task_on_auth_failed(self):
        gate = make_gate(['acct-A'], wait_for_reset=True, cost_store=None)
        acct = gate._accounts[0]
        gate._shutting_down = True

        gate._transition(acct, AccountPhase.AUTH_FAILED)

        assert acct.phase == AccountPhase.AUTH_FAILED
        assert acct.auth_reprobe_task is None

    # -----------------------------------------------------------------
    # step-3: cost-event emission, gate-level pause bookkeeping, and
    # recovery bookkeeping (probe_count reset / pause-time consumption /
    # auth_failed_at clearing) owned by _transition itself.
    # -----------------------------------------------------------------

    async def test_transition_to_capped_fires_cap_hit_event_with_reason_and_resets_at(self):
        gate = make_gate(['acct-A'], wait_for_reset=True, cost_store=make_mock_cost_store())
        acct = gate._accounts[0]
        resets_at = datetime.now(UTC) + timedelta(hours=1)

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate._transition(acct, AccountPhase.CAPPED, resets_at=resets_at, reason='hit')

        mock_fire.assert_called_once()
        account_name, event_type, details = mock_fire.call_args[0]
        assert account_name == acct.name
        assert event_type == 'cap_hit'
        parsed = json.loads(details)
        assert parsed['reason'] == 'hit'
        assert parsed['resets_at'] == resets_at.isoformat()
        await _drain_task(acct.resume_task)

    async def test_transition_to_auth_failed_fires_auth_failed_event_with_reason_and_resets_at(self):
        gate = make_gate(['acct-A'], wait_for_reset=True, cost_store=make_mock_cost_store())
        acct = gate._accounts[0]
        resets_at = datetime.now(UTC) + timedelta(hours=2)

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate._transition(acct, AccountPhase.AUTH_FAILED, resets_at=resets_at, reason='403')

        mock_fire.assert_called_once()
        account_name, event_type, details = mock_fire.call_args[0]
        assert account_name == acct.name
        assert event_type == 'auth_failed'
        parsed = json.loads(details)
        assert parsed['reason'] == '403'
        assert parsed['resets_at'] == resets_at.isoformat()
        await _drain_task(acct.auth_reprobe_task)

    async def test_transition_to_capped_without_cost_store_no_json_dumps_or_fire(self):
        gate = make_gate(['acct-A'], wait_for_reset=True, cost_store=None)
        acct = gate._accounts[0]

        with (
            patch('shared.usage_gate.json.dumps') as mock_dumps,
            patch.object(gate, '_fire_cost_event') as mock_fire,
        ):
            gate._transition(acct, AccountPhase.CAPPED, reason='hit')

        mock_dumps.assert_not_called()
        mock_fire.assert_not_called()
        await _drain_task(acct.resume_task)

    async def test_transition_to_auth_failed_without_cost_store_no_json_dumps_or_fire(self):
        gate = make_gate(['acct-A'], wait_for_reset=True, cost_store=None)
        acct = gate._accounts[0]

        with (
            patch('shared.usage_gate.json.dumps') as mock_dumps,
            patch.object(gate, '_fire_cost_event') as mock_fire,
        ):
            gate._transition(acct, AccountPhase.AUTH_FAILED, reason='403')

        mock_dumps.assert_not_called()
        mock_fire.assert_not_called()
        await _drain_task(acct.auth_reprobe_task)

    async def test_transition_to_capped_stamps_gate_pause_when_all_blocked(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        assert gate._paused_reason == ''
        assert gate._pause_started_at is None

        gate._transition(acct, AccountPhase.CAPPED, reason='cap reason xyz')

        assert gate._paused_reason != ''
        assert 'cap reason xyz' in gate._paused_reason
        assert gate._pause_started_at is not None

    async def test_transition_to_capped_leaves_paused_reason_untouched_when_sibling_available(self):
        gate = make_gate(['acct-A', 'acct-B'], wait_for_reset=False, cost_store=None)
        a, b = gate._accounts
        assert gate._paused_reason == ''

        gate._transition(a, AccountPhase.CAPPED, reason='a capped')

        assert b.phase == AccountPhase.AVAILABLE
        assert gate._paused_reason == ''
        assert gate._pause_started_at is None

    async def test_transition_from_capped_to_probing_resets_probe_count_and_consumes_pause(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.CAPPED
        acct.probe_count = 3
        acct.pause_started_at = datetime.now(UTC) - timedelta(seconds=120)

        gate._transition(acct, AccountPhase.PROBING)

        assert acct.probe_count == 0
        assert acct.pause_started_at is None
        assert 119 <= gate._total_pause_secs <= 121

    async def test_transition_from_auth_failed_to_available_clears_auth_failed_at(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.AUTH_FAILED
        acct.auth_failed_at = datetime.now(UTC)

        gate._transition(acct, AccountPhase.AVAILABLE)

        assert acct.auth_failed_at is None

    async def test_recovery_from_capped_does_not_fire_resumed_event(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=make_mock_cost_store())
        acct = gate._accounts[0]
        acct.phase = AccountPhase.CAPPED
        acct.pause_started_at = datetime.now(UTC)

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate._transition(acct, AccountPhase.PROBING)

        mock_fire.assert_not_called()

    async def test_recovery_from_auth_failed_does_not_fire_auth_resumed_event(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=make_mock_cost_store())
        acct = gate._accounts[0]
        acct.phase = AccountPhase.AUTH_FAILED
        acct.auth_failed_at = datetime.now(UTC)

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate._transition(acct, AccountPhase.AVAILABLE)

        mock_fire.assert_not_called()


# ---------------------------------------------------------------------------
# step-5/6: _handle_cap_detected migrated onto _transition (sole writer) —
# the ~10 inline flag writes + open-recompute + cost-event firing this
# handler used to own directly are retired in favor of one _transition call.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHandleCapDetectedViaTransition:
    """step-5: _handle_cap_detected routes through _transition."""

    async def test_marks_phase_capped_and_starts_resume_task(self):
        gate = make_gate(['acct-A'], wait_for_reset=True, cost_store=None)
        acct = gate._accounts[0]
        resets_at = datetime.now(UTC) + timedelta(hours=1)

        result = gate._handle_cap_detected('usage limit', resets_at, acct.token)

        assert result is True
        assert acct.phase == AccountPhase.CAPPED
        assert acct.resume_task is not None
        await _drain_task(acct.resume_task)

    async def test_fires_exactly_one_cap_hit_event(self):
        cost_store = make_mock_cost_store()
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=cost_store)
        acct = gate._accounts[0]
        resets_at = datetime.now(UTC) + timedelta(hours=1)

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate._handle_cap_detected('usage limit', resets_at, acct.token)

        mock_fire.assert_called_once()
        _, event_type, _ = mock_fire.call_args[0]
        assert event_type == 'cap_hit'

    async def test_open_recomputed_centrally_single_account_clears(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]

        gate._handle_cap_detected('usage limit', None, acct.token)

        assert gate._open.is_set() is False

    async def test_open_recomputed_centrally_sibling_available_stays_set(self):
        gate = make_gate(['acct-A', 'acct-B'], wait_for_reset=False, cost_store=None)
        a, b = gate._accounts

        gate._handle_cap_detected('usage limit', None, a.token)

        assert b.phase == AccountPhase.AVAILABLE
        assert gate._open.is_set() is True

    async def test_repeat_call_on_same_account_does_not_raise(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]

        gate._handle_cap_detected('first', None, acct.token)
        result = gate._handle_cap_detected('second', None, acct.token)  # must not raise

        assert result is True
        assert acct.phase == AccountPhase.CAPPED

    async def test_repeat_call_on_capped_account_fires_cap_hit_only_once(self):
        """The old hand-rolled handler re-fired cap_hit on every repeat
        detection (no idempotency check). Routing through _transition's
        legality guard makes a same-phase repeat call a pure no-op."""
        cost_store = make_mock_cost_store()
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=cost_store)
        acct = gate._accounts[0]

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate._handle_cap_detected('first', None, acct.token)
            gate._handle_cap_detected('second', None, acct.token)

        assert mock_fire.call_count == 1

    async def test_demotes_auth_failed_account_to_capped_with_auth_cleared(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.auth_failed = True
        assert acct.phase == AccountPhase.AUTH_FAILED

        result = gate._handle_cap_detected('usage limit', None, acct.token)

        assert result is True
        assert acct.phase == AccountPhase.CAPPED
        assert acct.auth_failed is False

    async def test_demotes_auth_failed_account_cancels_stale_auth_reprobe_task(self):
        """The old handler never cancelled a running auth_reprobe_task on
        demotion. Routing through _transition's entering-CAPPED effects must
        cancel it — mirrors test_transition_to_capped_cancels_pending_auth_reprobe_task."""
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        gate._transition(acct, AccountPhase.AUTH_FAILED)
        stale_task = acct.auth_reprobe_task
        assert stale_task is not None
        # Let the reprobe task actually start running and suspend inside its
        # `await asyncio.sleep(interval)` (interval defaults to 3600s) —
        # otherwise it never gets a chance to run before the phase flips and
        # its own `while acct.auth_failed` guard would trivially end it,
        # masking whether _transition's entering-CAPPED effects truly cancel it.
        await asyncio.sleep(0)
        assert stale_task.done() is False

        gate._handle_cap_detected('usage limit', None, acct.token)

        await asyncio.sleep(0)
        assert stale_task.cancelled() or stale_task.done()
        assert acct.phase == AccountPhase.CAPPED


# ---------------------------------------------------------------------------
# step-7/8: _handle_auth_failure migrated onto _transition (sole writer).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHandleAuthFailureViaTransition:
    """step-7: _handle_auth_failure routes through _transition."""

    async def test_marks_phase_auth_failed_and_stamps_auth_failed_at(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]

        result = gate._handle_auth_failure('403 forbidden', acct.token)

        assert result is True
        assert acct.phase == AccountPhase.AUTH_FAILED
        assert acct.auth_failed_at is not None

    async def test_starts_auth_reprobe_task(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]

        gate._handle_auth_failure('403 forbidden', acct.token)

        assert acct.auth_reprobe_task is not None
        await _drain_task(acct.auth_reprobe_task)

    async def test_fires_exactly_one_auth_failed_event(self):
        cost_store = make_mock_cost_store()
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=cost_store)
        acct = gate._accounts[0]

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate._handle_auth_failure('403 forbidden', acct.token)

        mock_fire.assert_called_once()
        _, event_type, _ = mock_fire.call_args[0]
        assert event_type == 'auth_failed'

    async def test_event_details_include_resets_at_only_when_reason_mentions_resets(self):
        cost_store = make_mock_cost_store()
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=cost_store)
        acct = gate._accounts[0]

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate._handle_auth_failure(
                "HTTP 429: You're out of extra usage · resets in 3h", acct.token,
            )

        _, _, details_json = mock_fire.call_args[0]
        details = json.loads(details_json)
        assert 'resets_at' in details

    async def test_event_details_omit_resets_at_for_pure_auth_error(self):
        cost_store = make_mock_cost_store()
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=cost_store)
        acct = gate._accounts[0]

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate._handle_auth_failure('HTTP 403: token has been revoked', acct.token)

        _, _, details_json = mock_fire.call_args[0]
        details = json.loads(details_json)
        assert 'resets_at' not in details

    async def test_open_recomputed_centrally_single_account_clears(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]

        gate._handle_auth_failure('403 forbidden', acct.token)

        assert gate._open.is_set() is False

    async def test_open_recomputed_centrally_sibling_available_stays_set(self):
        gate = make_gate(['acct-A', 'acct-B'], wait_for_reset=False, cost_store=None)
        a, b = gate._accounts

        gate._handle_auth_failure('403 forbidden', a.token)

        assert b.phase == AccountPhase.AVAILABLE
        assert gate._open.is_set() is True

    async def test_repeat_call_on_same_account_does_not_raise(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]

        gate._handle_auth_failure('first', acct.token)
        result = gate._handle_auth_failure('second', acct.token)  # must not raise

        assert result is True
        assert acct.phase == AccountPhase.AUTH_FAILED

    async def test_repeat_call_on_auth_failed_account_fires_event_only_once(self):
        """Mirrors the cap_hit idempotency fix: the old handler re-fired
        auth_failed on every repeat detection with no guard at all."""
        cost_store = make_mock_cost_store()
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=cost_store)
        acct = gate._accounts[0]

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            gate._handle_auth_failure('first', acct.token)
            gate._handle_auth_failure('second', acct.token)

        assert mock_fire.call_count == 1


# ---------------------------------------------------------------------------
# step-9/10: the CAPPED->PROBING uncap paths (_refresh_capped_accounts and
# _account_resume_probe_loop success) migrated onto _transition.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestUncapViaTransition:
    """step-9: _refresh_capped_accounts and _account_resume_probe_loop route
    the CAPPED->PROBING uncap edge through _transition (probe_count reset +
    pause-time consumption + centralized _open recompute owned by _transition,
    replacing the two sites' own inline field writes)."""

    async def test_refresh_capped_accounts_past_reset_transitions_to_probing(self):
        gate = make_gate(['a'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.CAPPED
        acct.probe_count = 3
        acct.resets_at = datetime.now(UTC) - timedelta(minutes=1)
        acct.pause_started_at = datetime.now(UTC) - timedelta(seconds=120)
        gate._total_pause_secs = 0.0

        result = await gate._refresh_capped_accounts()

        assert result is True
        assert acct.phase == AccountPhase.PROBING
        assert acct.probe_count == 0
        assert acct.pause_started_at is None
        assert 119 <= gate._total_pause_secs <= 121
        assert gate._open.is_set() is True

    async def test_refresh_capped_accounts_future_reset_stays_capped(self):
        gate = make_gate(['a'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.CAPPED
        acct.resets_at = datetime.now(UTC) + timedelta(hours=1)

        result = await gate._refresh_capped_accounts()

        assert result is False
        assert acct.phase == AccountPhase.CAPPED

    async def test_probe_loop_success_transitions_to_probing_via_transition(self):
        gate = make_gate(['a'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.CAPPED
        acct.probe_count = 0
        acct.resets_at = datetime.now(UTC) - timedelta(minutes=1)
        acct.pause_started_at = datetime.now(UTC) - timedelta(seconds=60)
        gate._total_pause_secs = 0.0
        gate._run_probe = AsyncMock(return_value=True)

        await asyncio.wait_for(gate._account_resume_probe_loop(acct), timeout=5)

        assert acct.phase == AccountPhase.PROBING
        assert acct.probe_count == 0
        assert acct.pause_started_at is None
        assert gate._total_pause_secs >= 59
        assert gate._open.is_set() is True

    async def test_probe_loop_success_still_fires_labeled_resumed_event(self):
        gate = make_gate(['a'], wait_for_reset=False, cost_store=make_mock_cost_store())
        acct = gate._accounts[0]
        acct.phase = AccountPhase.CAPPED
        acct.probe_count = 0
        acct.resets_at = datetime.now(UTC) - timedelta(minutes=1)
        gate._run_probe = AsyncMock(return_value=True)

        with patch.object(gate, '_write_cost_event', new_callable=AsyncMock) as mock_write:
            await asyncio.wait_for(gate._account_resume_probe_loop(acct), timeout=5)

        mock_write.assert_awaited_once()
        account_name, event_type, details_json = mock_write.call_args[0]
        assert account_name == acct.name
        assert event_type == 'resumed'
        details = json.loads(details_json)
        assert details['label'] == 'probe #1 confirmed'

    async def test_probe_loop_success_no_ops_when_phase_raced_away(self):
        """Regression (esc-2128-2): _refresh_capped_accounts runs outside
        self._lock and targets the same resets_at event, so it can win the
        CAPPED->PROBING race while _account_resume_probe_loop awaits _run_probe.
        When the probe then returns ok=True from a non-CAPPED phase, the success
        block must be a pure no-op — NOT call _transition(acct, PROBING), which
        would raise IllegalTransitionError inside the fire-and-forget task
        (PROBING/AVAILABLE/PROBE_IN_FLIGHT all lack PROBING as a legal
        successor). Mirrors the auth-reprobe guard in _reprobe_account."""
        gate = make_gate(['a'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.CAPPED
        acct.probe_count = 0
        acct.resets_at = datetime.now(UTC) - timedelta(minutes=1)

        async def racing_probe(acct):
            # Simulate _refresh_capped_accounts uncapping this account mid-await.
            acct.phase = AccountPhase.PROBING
            return True

        gate._run_probe = racing_probe

        # Must not raise IllegalTransitionError (PROBING->PROBING is illegal).
        await asyncio.wait_for(gate._account_resume_probe_loop(acct), timeout=5)

        assert acct.phase == AccountPhase.PROBING


# ---------------------------------------------------------------------------
# step-11/12: the auth-recovery (_reprobe_account) and demotion (_run_probe)
# paths migrated onto _transition.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAuthRecoveryViaTransition:
    """step-11: _reprobe_account and _run_probe's demotion branch route the
    AUTH_FAILED->AVAILABLE and AUTH_FAILED->CAPPED edges through _transition."""

    async def test_reprobe_account_success_transitions_auth_failed_to_available(self):
        gate = make_gate(['a'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.AUTH_FAILED
        acct.auth_failed_at = datetime.now(UTC)
        gate._run_probe = AsyncMock(return_value=True)

        await gate._reprobe_account(acct)

        assert acct.phase == AccountPhase.AVAILABLE
        assert acct.auth_failed_at is None
        assert gate._open.is_set() is True

    async def test_reprobe_account_success_still_fires_auth_resumed_event(self):
        cost_store = make_mock_cost_store()
        gate = make_gate(['a'], wait_for_reset=False, cost_store=cost_store)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.AUTH_FAILED
        gate._run_probe = AsyncMock(return_value=True)

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            await gate._reprobe_account(acct)

        mock_fire.assert_called_once()
        account_name, event_type, _ = mock_fire.call_args[0]
        assert account_name == acct.name
        assert event_type == 'auth_resumed'

    async def test_reprobe_account_repeat_call_fires_auth_resumed_only_once(self):
        """The old handler unconditionally rewrote auth_failed/auth_failed_at
        and re-fired 'auth_resumed' on every successful reprobe call, even
        once the account was already AVAILABLE from a prior success — the
        same class of repeat-call bug fixed for _handle_cap_detected (step-6)
        and _handle_auth_failure (step-8)."""
        cost_store = make_mock_cost_store()
        gate = make_gate(['a'], wait_for_reset=False, cost_store=cost_store)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.AUTH_FAILED
        gate._run_probe = AsyncMock(return_value=True)

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            await gate._reprobe_account(acct)
            await gate._reprobe_account(acct)

        assert mock_fire.call_count == 1

    async def test_reprobe_account_on_already_available_account_is_noop(self):
        """SIGHUP's fan-out calls _reprobe_account on EVERY account after
        force-resetting them all to AVAILABLE — including ones that were
        never auth_failed. Must not raise IllegalTransitionError."""
        gate = make_gate(['a'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        assert acct.phase == AccountPhase.AVAILABLE
        gate._run_probe = AsyncMock(return_value=True)

        await gate._reprobe_account(acct)  # must not raise

        assert acct.phase == AccountPhase.AVAILABLE

    async def test_run_probe_demotion_transitions_auth_failed_to_capped(self):
        """Mirrors the out-of-scope
        test_usage_gate_exhaustive.TestAuthReprobeDemoteOnCapMessage contract:
        an AUTH_FAILED account whose reprobe output contains a cap prefix is
        demoted to CAPPED (the single legal AUTH_FAILED->CAPPED edge)."""
        gate = make_gate(['a'], wait_for_reset=True, cost_store=make_mock_cost_store())
        del gate._run_probe  # fall back to the real method under test
        gate._probe_config_dir = MagicMock()
        gate._probe_config_dir.path = Path('/tmp/probe-test')
        gate._probe_config_dir.write_credentials = MagicMock()

        acct = gate._accounts[0]
        acct.phase = AccountPhase.AUTH_FAILED
        acct.auth_failed_at = datetime.now(UTC)

        async def _idle_loop() -> None:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(60)
        stale_reprobe_task = asyncio.get_event_loop().create_task(_idle_loop())
        acct.auth_reprobe_task = stale_reprobe_task

        cap_stderr = b"You're out of extra usage \xc2\xb7 resets in 2h"
        proc = MagicMock()
        proc.returncode = 1
        proc.pid = 12345
        proc.communicate = AsyncMock(return_value=(b'', cap_stderr))

        with patch('shared.usage_gate.asyncio.create_subprocess_exec', return_value=proc):
            ok = await gate._run_probe(acct)

        assert ok is False
        assert acct.phase == AccountPhase.CAPPED
        assert acct.auth_failed_at is None
        assert acct.resets_at is not None
        delta = acct.resets_at - datetime.now(UTC)
        assert timedelta(minutes=90) < delta < timedelta(hours=3)
        # Re-read acct.auth_reprobe_task (not the captured stale_reprobe_task
        # local) — this mirrors exactly what the sibling test asserts: the
        # *field* must end up None or settled, not merely that the original
        # task object was asked to cancel (cancellation only takes effect at
        # the task's next suspension point, which may not have happened yet).
        assert (
            acct.auth_reprobe_task is None
            or acct.auth_reprobe_task.cancelled()
            or acct.auth_reprobe_task.done()
        )
        assert stale_reprobe_task is not None  # sanity: it really was set
        assert acct.resume_task is not None
        await _drain_task(acct.resume_task)

    async def test_run_probe_demotion_fires_cap_hit_event(self):
        gate = make_gate(['a'], wait_for_reset=True, cost_store=make_mock_cost_store())
        del gate._run_probe
        gate._probe_config_dir = MagicMock()
        gate._probe_config_dir.path = Path('/tmp/probe-test')
        gate._probe_config_dir.write_credentials = MagicMock()

        acct = gate._accounts[0]
        acct.phase = AccountPhase.AUTH_FAILED

        cap_stderr = b"You're out of extra usage \xc2\xb7 resets in 2h"
        proc = MagicMock()
        proc.returncode = 1
        proc.pid = 12345
        proc.communicate = AsyncMock(return_value=(b'', cap_stderr))

        with (
            patch('shared.usage_gate.asyncio.create_subprocess_exec', return_value=proc),
            patch.object(gate, '_fire_cost_event') as mock_fire,
        ):
            await gate._run_probe(acct)

        mock_fire.assert_called_once()
        _, event_type, _ = mock_fire.call_args[0]
        assert event_type == 'cap_hit'
        await _drain_task(acct.resume_task)

    async def test_run_probe_still_returns_false_on_cap_when_not_auth_failed(self):
        """Preserve the existing _run_probe return-False-on-cap contract for
        a plain (non-auth_failed) account — TestUsageGateProbeExitCodeClassification
        coverage, pinned here too since it shares this method."""
        gate = make_gate(['a'], wait_for_reset=False, cost_store=None)
        del gate._run_probe
        gate._probe_config_dir = MagicMock()
        gate._probe_config_dir.path = Path('/tmp/probe-test')
        gate._probe_config_dir.write_credentials = MagicMock()

        acct = gate._accounts[0]
        assert acct.phase == AccountPhase.AVAILABLE

        cap_stderr = b"You're out of extra usage \xc2\xb7 resets in 2h"
        proc = MagicMock()
        proc.returncode = 1
        proc.pid = 12345
        proc.communicate = AsyncMock(return_value=(b'', cap_stderr))

        with patch('shared.usage_gate.asyncio.create_subprocess_exec', return_value=proc):
            ok = await gate._run_probe(acct)

        assert ok is False
        assert acct.phase == AccountPhase.AVAILABLE


# ---------------------------------------------------------------------------
# step-13/14: the PROBE_IN_FLIGHT edges (confirm_account_ok, release_probe_slot,
# and before_invoke's probe-claim) migrated onto _transition.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProbeSlotViaTransition:
    """step-13: confirm_account_ok / release_probe_slot / before_invoke's
    probe-claim route the PROBE_IN_FLIGHT<->AVAILABLE/PROBING edges through
    _transition (probe_count reset + centralized _open recompute owned by
    _transition, replacing each site's own inline field writes and
    _open.set()/clear())."""

    async def test_confirm_account_ok_on_probe_in_flight_transitions_via_transition(self):
        gate = make_gate(['a'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.PROBE_IN_FLIGHT
        acct.probe_count = 2
        acct.near_cap = True
        gate._open.clear()

        with patch.object(gate, '_transition', wraps=gate._transition) as mock_transition:
            gate.confirm_account_ok(acct.token)

        mock_transition.assert_called_once_with(acct, AccountPhase.AVAILABLE)
        assert acct.phase == AccountPhase.AVAILABLE
        assert acct.probe_count == 0
        assert acct.near_cap is False
        assert gate._open.is_set() is True

    async def test_confirm_account_ok_on_non_probe_in_flight_does_not_transition(self):
        gate = make_gate(['a'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.near_cap = True
        assert acct.phase == AccountPhase.AVAILABLE

        with patch.object(gate, '_transition', wraps=gate._transition) as mock_transition:
            gate.confirm_account_ok(acct.token)  # must not raise

        mock_transition.assert_not_called()
        assert acct.near_cap is False
        assert acct.phase == AccountPhase.AVAILABLE

    async def test_release_probe_slot_on_probe_in_flight_transitions_via_transition(self):
        gate = make_gate(['a'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.PROBE_IN_FLIGHT
        acct.probe_count = 4
        gate._open.clear()

        with patch.object(gate, '_transition', wraps=gate._transition) as mock_transition:
            gate.release_probe_slot(acct.token)

        mock_transition.assert_called_once_with(
            acct, AccountPhase.AVAILABLE, clear_near_cap=False,
        )
        assert acct.phase == AccountPhase.AVAILABLE
        assert acct.probe_count == 0
        assert gate._open.is_set() is True

    async def test_release_probe_slot_does_not_clear_near_cap(self):
        """release_probe_slot fires on an exception path unrelated to cap
        status — unlike confirm_account_ok, it must leave a stale near_cap
        warning untouched (mirrors the out-of-scope
        test_usage_gate_exhaustive.TestReleaseProbeSlot.
        test_does_not_touch_near_cap_flag contract, which _transition's
        default blanket near_cap clear would otherwise break)."""
        gate = make_gate(['a'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.PROBE_IN_FLIGHT
        acct.near_cap = True

        gate.release_probe_slot(acct.token)

        assert acct.phase == AccountPhase.AVAILABLE
        assert acct.near_cap is True

    async def test_release_probe_slot_noop_when_token_none(self):
        gate = make_gate(['a'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.PROBE_IN_FLIGHT

        gate.release_probe_slot(None)  # must not raise

        assert acct.phase == AccountPhase.PROBE_IN_FLIGHT

    async def test_release_probe_slot_noop_when_token_unknown(self):
        gate = make_gate(['a'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.PROBE_IN_FLIGHT

        gate.release_probe_slot('unknown-token')  # must not raise

        assert acct.phase == AccountPhase.PROBE_IN_FLIGHT

    async def test_release_probe_slot_noop_when_not_probe_in_flight(self):
        gate = make_gate(['a'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        assert acct.phase == AccountPhase.AVAILABLE

        with patch.object(gate, '_transition', wraps=gate._transition) as mock_transition:
            gate.release_probe_slot(acct.token)  # must not raise

        mock_transition.assert_not_called()
        assert acct.phase == AccountPhase.AVAILABLE

    async def test_before_invoke_claims_probing_account_via_transition(self):
        gate = make_gate(['a'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]
        acct.phase = AccountPhase.PROBING

        with patch.object(gate, '_transition', wraps=gate._transition) as mock_transition:
            token = await gate.before_invoke()

        mock_transition.assert_called_once_with(acct, AccountPhase.PROBE_IN_FLIGHT)
        assert token == acct.token
        assert acct.phase == AccountPhase.PROBE_IN_FLIGHT
        assert gate._open.is_set() is False


# ---------------------------------------------------------------------------
# step-15/16: B8 — _on_sighup_async's per-account hard reset migrated onto a
# single forced _transition(acct, AVAILABLE, force=True) call, replacing the
# old 9-field inline reset + standalone self._open.set().
# ---------------------------------------------------------------------------


async def _idle_task() -> asyncio.Task:
    async def _hang() -> None:
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(60)
    return asyncio.get_event_loop().create_task(_hang())


@pytest.mark.asyncio
class TestSighupUncapsViaTransition:
    """step-15: _on_sighup_async drives every account to AVAILABLE through
    one forced _transition call each (DD-4)."""

    async def test_sighup_forces_every_account_to_available_via_transition(self):
        gate = make_gate(['a', 'b'], wait_for_reset=False, cost_store=None)
        a, b = gate._accounts
        a.phase = AccountPhase.CAPPED
        a.resets_at = datetime.now(UTC) + timedelta(hours=1)
        a.pause_started_at = datetime.now(UTC) - timedelta(seconds=30)
        a.probe_count = 2
        a.resume_task = await _idle_task()

        b.phase = AccountPhase.AUTH_FAILED
        b.auth_failed_at = datetime.now(UTC)
        b.auth_reprobe_task = await _idle_task()

        gate._open.clear()
        gate._paused_reason = 'some prior reason'
        gate._reprobe_account = AsyncMock(return_value=None)

        with (
            patch.object(gate, '_transition', wraps=gate._transition) as mock_transition,
            patch('shared.usage_gate.load_dotenv'),
        ):
            await gate._on_sighup_async()

        assert mock_transition.call_count == 2
        for call in mock_transition.call_args_list:
            args, kwargs = call
            assert args[1] == AccountPhase.AVAILABLE
            assert kwargs.get('force') is True

        assert a.phase == AccountPhase.AVAILABLE
        assert b.phase == AccountPhase.AVAILABLE
        await asyncio.sleep(0)
        assert a.resume_task.cancelled() or a.resume_task.done()
        assert b.auth_reprobe_task.cancelled() or b.auth_reprobe_task.done()
        assert a.resets_at is None
        assert a.pause_started_at is None
        assert a.probe_count == 0
        assert b.auth_failed_at is None
        assert gate._open.is_set() is True
        assert gate._paused_reason == ''


# ---------------------------------------------------------------------------
# step-17/18: B2 (DD-5) — the centralized _open recompute invariant
# (gate._open.is_set() <=> any(acct.phase in {AVAILABLE, PROBING})) must hold
# after EVERY _transition call. hypothesis is NOT installed (import fails),
# so this is hand-rolled: an exhaustive walk of every _LEGAL_TRANSITIONS edge
# plus a seeded random.Random multi-account walk, mirroring what a
# hypothesis @given(...) strategy would otherwise generate.
# ---------------------------------------------------------------------------


def _open_invariant_holds(gate: UsageGate) -> bool:
    expected_open = any(
        a.phase in (AccountPhase.AVAILABLE, AccountPhase.PROBING)
        for a in gate._accounts
    )
    return gate._open.is_set() == expected_open


@pytest.mark.asyncio
class TestOpenInvariantProperty:
    """step-17: DD-5 property — _open.is_set() <=> any account in
    {AVAILABLE, PROBING} — holds after every legal _transition edge, both
    exhaustively and over long seeded random walks."""

    async def test_exhaustive_every_legal_edge_preserves_invariant(self):
        for from_phase, targets in _LEGAL_TRANSITIONS.items():
            for to_phase in targets:
                gate = make_gate(['a'], wait_for_reset=False, cost_store=None)
                acct = gate._accounts[0]
                acct.phase = from_phase

                gate._transition(acct, to_phase)

                assert _open_invariant_holds(gate), (
                    f'{from_phase} -> {to_phase} broke the DD-5 _open invariant'
                )
                await _drain_task(acct.resume_task)
                await _drain_task(acct.auth_reprobe_task)

    async def test_random_walk_two_accounts_preserves_invariant(self):
        rng = random.Random(1234)
        gate = make_gate(['a', 'b'], wait_for_reset=False, cost_store=None)
        assert _open_invariant_holds(gate)

        for _ in range(300):
            acct = rng.choice(gate._accounts)
            legal = sorted(_LEGAL_TRANSITIONS.get(acct.phase, frozenset()))
            if not legal:
                continue
            gate._transition(acct, rng.choice(legal))
            assert _open_invariant_holds(gate)

        for acct in gate._accounts:
            await _drain_task(acct.resume_task)
            await _drain_task(acct.auth_reprobe_task)

    async def test_random_walk_three_accounts_preserves_invariant(self):
        rng = random.Random(5678)
        gate = make_gate(['a', 'b', 'c'], wait_for_reset=False, cost_store=None)
        assert _open_invariant_holds(gate)

        for _ in range(300):
            acct = rng.choice(gate._accounts)
            legal = sorted(_LEGAL_TRANSITIONS.get(acct.phase, frozenset()))
            if not legal:
                continue
            gate._transition(acct, rng.choice(legal))
            assert _open_invariant_holds(gate)

        for acct in gate._accounts:
            await _drain_task(acct.resume_task)
            await _drain_task(acct.auth_reprobe_task)

    async def test_force_reset_preserves_invariant(self):
        gate = make_gate(['a', 'b'], wait_for_reset=False, cost_store=None)
        a, b = gate._accounts
        a.phase = AccountPhase.CAPPED
        b.phase = AccountPhase.AUTH_FAILED

        gate._transition(a, AccountPhase.AVAILABLE, force=True)
        assert _open_invariant_holds(gate)

        gate._transition(b, AccountPhase.AVAILABLE, force=True)
        assert _open_invariant_holds(gate)


# ---------------------------------------------------------------------------
# step-19: regression for the review-flagged CAPPED->AUTH_FAILED illegal-edge
# crash (usage_gate.py:744). A concurrent sibling task can cap an account
# (via _handle_cap_detected) after before_invoke already handed it out
# AVAILABLE; when the in-flight caller's request then fails with a 403, the
# unconditional _transition(acct, AUTH_FAILED) call raised
# IllegalTransitionError because CAPPED->AUTH_FAILED is not a legal edge
# (_LEGAL_TRANSITIONS[CAPPED] == {PROBING}). _handle_auth_failure must
# short-circuit to a no-op for an already-CAPPED account instead of crashing.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHandleAuthFailureOnCappedAccountIsNoop:
    """step-19: _handle_auth_failure on an already-CAPPED account is a no-op
    that does not raise, does not demote the cap, and still returns True."""

    async def test_capped_account_auth_failure_does_not_raise_and_stays_capped(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=make_mock_cost_store())
        acct = gate._accounts[0]
        acct.phase = AccountPhase.CAPPED
        acct.resets_at = datetime.now(UTC) + timedelta(hours=1)
        stashed_resets_at = acct.resets_at

        with patch.object(gate, '_fire_cost_event') as mock_fire:
            result = gate._handle_auth_failure('HTTP 403: forbidden', acct.token)

        assert result is True
        assert acct.phase == AccountPhase.CAPPED
        assert acct.resets_at == stashed_resets_at
        mock_fire.assert_not_called()

    async def test_available_account_still_transitions_to_auth_failed(self):
        gate = make_gate(['acct-A'], wait_for_reset=False, cost_store=None)
        acct = gate._accounts[0]

        result = gate._handle_auth_failure('HTTP 403: forbidden', acct.token)

        assert result is True
        assert acct.phase == AccountPhase.AUTH_FAILED
