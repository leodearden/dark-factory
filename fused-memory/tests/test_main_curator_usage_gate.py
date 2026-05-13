"""Tests for curator UsageGate + CostStore wiring in fused_memory.server.main."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest
from shared.config_models import AccountConfig, UsageCapConfig
from shared.cost_store import CostStore
from shared.usage_gate import UsageGate

from fused_memory.server.main import _resolve_curator_cost_store_path

# ---------------------------------------------------------------------------
# step-1: end-to-end — cap_hit event reaches the account_events table
#
# This exercises the UsageGate ↔ CostStore interface (shared-package behaviour)
# via the same constructor kwarg used in run_server after suggestion-1 amendment.
# Moving the test to shared/tests/ is out of scope (locked module boundary).
# ---------------------------------------------------------------------------


class TestCuratorCapEventPersisted:
    """UsageGate + real CostStore integration — cap_hit written to DB."""

    @pytest.mark.asyncio
    async def test_cap_hit_written_to_cost_store(self, tmp_path):
        """_handle_cap_detected must persist a cap_hit row to account_events.

        Real aiosqlite writes involve thread-pool I/O and need more than 2
        asyncio.sleep(0) ticks to complete. We gather on _background_tasks
        directly so the assertion never races against the background write.
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

            # Drain: gather on the background tasks captured synchronously
            # right after the fire.  asyncio.gather waits until all tasks
            # complete (including their aiosqlite thread-pool I/O rounds),
            # which is more reliable than a fixed sleep(0) loop.
            pending = list(gate._background_tasks)
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            # Verify via raw aiosqlite read (independent connection).
            async with aiosqlite.connect(db_path) as conn:
                cursor = await conn.execute(
                    'SELECT account_name, event_type, details FROM account_events'
                )
                rows = list(await cursor.fetchall())
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


@pytest.mark.parametrize(
    ('reconciliation_cfg', 'expected'),
    [
        (None, Path('./data/curator_events.db')),
        (MagicMock(data_dir='/tmp/recon'), Path('/tmp/recon/curator_events.db')),
    ],
    ids=['fallback', 'with_data_dir'],
)
def test_resolve_curator_cost_store_path(reconciliation_cfg, expected):
    """Path helper returns correct db path based on reconciliation config."""
    config = MagicMock()
    config.reconciliation = reconciliation_cfg
    assert _resolve_curator_cost_store_path(config) == expected

