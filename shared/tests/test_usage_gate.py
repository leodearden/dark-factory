"""Tests for UsageGate — race condition fixes and robustness improvements."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import AccountState, UsageGate

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
