"""Tests for curator UsageGate + CostStore wiring in fused_memory.server.main."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from fused_memory.server.main import _resolve_curator_cost_store_path
from shared.config_models import AccountConfig, UsageCapConfig
from shared.cost_store import CostStore
from shared.usage_gate import UsageGate


# ---------------------------------------------------------------------------
# step-1: end-to-end — cap_hit event reaches the account_events table
# ---------------------------------------------------------------------------


class TestCuratorCapEventPersisted:
    """step-1: UsageGate + real CostStore integration — cap_hit written to DB."""

    @pytest.mark.asyncio
    async def test_cap_hit_written_to_cost_store(self, tmp_path):
        """_handle_cap_detected must persist a cap_hit row to account_events.

        This test confirms the UsageGate → CostStore plumbing works end-to-end
        before it is wired into run_server in step-2.

        Real aiosqlite writes involve thread-pool I/O and need more than 2
        asyncio.sleep(0) ticks to complete. We loop until _background_tasks
        drains (the done-callback calls discard()) to avoid flakiness.
        wait_for_reset=False avoids creating a long-lived probe task that
        would linger past the test boundary.
        """
        db_path = tmp_path / 'curator_events.db'

        store = CostStore(db_path)
        await store.open()
        try:
            # Build UsageGate with the real store; inject fake token env var.
            # wait_for_reset=False prevents a long-lived resume probe task.
            acct_cfg = AccountConfig(name='acct-a', oauth_token_env='TEST_TOKEN_ACCT_A')
            config = UsageCapConfig(accounts=[acct_cfg], wait_for_reset=False)

            with patch.dict(os.environ, {'TEST_TOKEN_ACCT_A': 'fake-token-acct-a'}):
                gate = UsageGate(config, cost_store=store)

            # Mock _run_probe to prevent real subprocess spawning.
            gate._run_probe = AsyncMock(return_value=True)

            # Fire the cap event.
            gate._handle_cap_detected('test cap reason', None, gate._accounts[0].token)

            # Drain: yield until all fire-and-forget write tasks complete.
            # Real aiosqlite involves thread I/O so needs more than 2 ticks.
            for _ in range(50):
                if not gate._background_tasks:
                    break
                await asyncio.sleep(0)

            # Verify via raw aiosqlite read (independent connection).
            async with aiosqlite.connect(db_path) as conn:
                cursor = await conn.execute(
                    'SELECT account_name, event_type, details FROM account_events'
                )
                rows = await cursor.fetchall()
        finally:
            await store.close()

        assert len(rows) == 1, f'Expected 1 row in account_events, got {len(rows)}: {rows}'
        account_name, event_type, details_str = rows[0]
        assert account_name == 'acct-a', f'account_name mismatch: {account_name!r}'
        assert event_type == 'cap_hit', f'event_type mismatch: {event_type!r}'
        details = json.loads(details_str)
        assert 'reason' in details, f'"reason" key missing from details: {details}'
        assert details['reason'] == 'test cap reason', (
            f'reason mismatch: {details["reason"]!r}'
        )


# ---------------------------------------------------------------------------
# step-3: _resolve_curator_cost_store_path helper
# ---------------------------------------------------------------------------


class TestResolveCuratorCostStorePath:
    """step-3: Path helper returns correct db path based on reconciliation config."""

    def test_returns_data_dir_path_when_reconciliation_configured(self):
        """With reconciliation config, path is <data_dir>/curator_events.db."""
        config = MagicMock()
        config.reconciliation = MagicMock()
        config.reconciliation.data_dir = '/tmp/recon'

        path = _resolve_curator_cost_store_path(config)

        assert path == Path('/tmp/recon/curator_events.db')

    def test_returns_fallback_path_when_no_reconciliation(self):
        """Without reconciliation config, path falls back to ./data/curator_events.db."""
        config = MagicMock()
        config.reconciliation = None

        path = _resolve_curator_cost_store_path(config)

        assert path == Path('./data/curator_events.db')
