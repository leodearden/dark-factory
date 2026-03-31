"""Tests for UsageGate — race condition fixes and robustness improvements."""

from __future__ import annotations

import asyncio
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
        gate = make_gate(['acct-A', 'acct-B'])

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
        gate = make_gate(['acct-A', 'acct-B'])

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
