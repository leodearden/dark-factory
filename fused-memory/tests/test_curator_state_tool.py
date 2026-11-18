"""Tests for the get_curator_state MCP tool in tools.py.

TDD steps:
  step-1  RED — create_mcp_server rejects curator_usage_gate kwarg; no tool registered
  step-3  RED — tool ignores live gate; assertions on paused/paused_reason/etc. fail
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import UsageGate

from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig
from fused_memory.middleware.task_curator import TaskCurator
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_mock_service() -> AsyncMock:
    """Minimal AsyncMock MemoryService for tool tests."""
    svc = AsyncMock()
    svc.durable_queue = None
    return svc


# ---------------------------------------------------------------------------
# step-1 RED: curator_usage_gate=None → unpaused defaults
# ---------------------------------------------------------------------------


class TestGetCuratorStateNoneGate:
    """get_curator_state returns safe defaults when curator_usage_gate is None."""

    @pytest.mark.asyncio
    async def test_get_curator_state_returns_unpaused_defaults_when_gate_none(self):
        """Tool returns {paused: False, paused_reason: None, ...} when no gate is wired."""
        svc = _make_mock_service()
        server = create_mcp_server(svc, curator_usage_gate=None)

        result = await server._tool_manager.call_tool('get_curator_state', {})

        assert result == {
            'paused': False,
            'paused_reason': None,
            'soonest_open_at': None,
            'account_count': 0,
            'registries': [],
        }, f'Unexpected result: {result!r}'


# ---------------------------------------------------------------------------
# step-3 RED: live gate (paused) → correct paused state forwarded
# ---------------------------------------------------------------------------


class TestGetCuratorStateLiveGate:
    """get_curator_state reads live UsageGate state when a gate is wired."""

    @pytest.mark.asyncio
    async def test_get_curator_state_returns_gate_state_when_paused(self, tmp_path):
        """Tool correctly maps gate's observable paused state to the MCP response.

        Drives gate state directly via internal attributes to isolate the tool's
        read path from the gate's transition machinery (probe tasks, background
        scheduling), per reviewer feedback on the original _handle_cap_detected
        approach.
        """
        from shared.cost_store import CostStore

        db_path = tmp_path / 'curator_events.db'
        store = CostStore(db_path)
        await store.open()
        gate: UsageGate | None = None
        try:
            acct_cfg = AccountConfig(name='acct-a', oauth_token_env='TEST_TOKEN_ACCT_A')
            config = UsageCapConfig(accounts=[acct_cfg], wait_for_reset=False)

            with __import__('unittest.mock', fromlist=['patch']).patch.dict(
                os.environ, {'TEST_TOKEN_ACCT_A': 'fake-token-acct-a'}
            ):
                gate = UsageGate(config, cost_store=store)

            # Drive paused state directly: set the underlying attributes that
            # is_paused, paused_reason, soonest_resets_at, and account_count read from.
            reset_dt = datetime(2026, 6, 1, tzinfo=UTC)
            gate._accounts[0].capped = True
            gate._accounts[0].resets_at = reset_dt
            gate._paused_reason = 'All accounts capped (last: synthetic cap)'

            svc = _make_mock_service()
            server = create_mcp_server(svc, curator_usage_gate=gate)

            result = await server._tool_manager.call_tool('get_curator_state', {})

            assert result == {
                'paused': True,
                'paused_reason': 'All accounts capped (last: synthetic cap)',
                'soonest_open_at': '2026-06-01T00:00:00+00:00',
                'account_count': 1,
                'registries': [],
            }, f'Unexpected result: {result!r}'
        finally:
            if gate is not None:
                await gate.shutdown()
            await store.close()


# ---------------------------------------------------------------------------
# task 6622: each curator guard registry's load state
# ---------------------------------------------------------------------------


class TestGetCuratorStateRegistries:
    """get_curator_state reports each guard registry's load state, so a guard
    that failed to load is distinguishable from one that loaded empty."""

    @pytest.mark.asyncio
    async def test_reports_each_registry_state_from_the_wired_curator(self, tmp_path):
        missing = tmp_path / 'blocklist.yaml'
        config = FusedMemoryConfig()
        config.curator = CuratorConfig(cancelled_premise_blocklist_path=str(missing))
        curator = TaskCurator(config=config, taskmaster=None, cwd=tmp_path)
        await curator.startup_self_check('p', str(tmp_path))
        interceptor = TaskInterceptor(
            None, None, EventBuffer(db_path=None), curator=curator,
        )
        server = create_mcp_server(_make_mock_service(), task_interceptor=interceptor)

        result = await server._tool_manager.call_tool('get_curator_state', {})

        registries = result['registries']
        assert [r['label'] for r in registries] == [
            'blocklist', 'recon-premise', 'operational-ask',
        ]
        blocklist, premise, operational = registries
        assert blocklist['state'] == 'failed'
        assert blocklist['path'] == str(missing)
        assert blocklist['failures'] == 1
        assert 'file not found' in blocklist['error']
        assert premise['state'] == 'unconfigured'
        assert operational['state'] == 'unconfigured'
        assert result['paused'] is False
        assert result['paused_reason'] is None
        assert result['soonest_open_at'] is None
        assert result['account_count'] == 0

    @pytest.mark.asyncio
    async def test_reading_state_never_constructs_a_curator(self):
        interceptor = TaskInterceptor(
            None, None, EventBuffer(db_path=None), config=FusedMemoryConfig(),
        )
        server = create_mcp_server(_make_mock_service(), task_interceptor=interceptor)

        result = await server._tool_manager.call_tool('get_curator_state', {})

        assert result['registries'] == []
        assert interceptor.curator_registry_health() == ()
