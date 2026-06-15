"""Tests for curator UsageGate + CostStore wiring in fused_memory.server.main."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest
from _fm_helpers import pydantic_spec
from shared.config_models import AccountConfig, UsageCapConfig
from shared.cost_store import CostStore
from shared.usage_gate import UsageGate

from fused_memory.config.schema import FusedMemoryConfig
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
    config = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
    config.reconciliation = reconciliation_cfg
    assert _resolve_curator_cost_store_path(config) == expected


# ---------------------------------------------------------------------------
# step-13: _resolve_curator_escalator_state_path helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('reconciliation_cfg', 'expected'),
    [
        (None, Path('./data/curator_escalator_state.json')),
        (MagicMock(data_dir='/tmp/recon'), Path('/tmp/recon/curator_escalator_state.json')),
    ],
    ids=['fallback', 'with_data_dir'],
)
def test_resolve_curator_escalator_state_path(reconciliation_cfg, expected):
    """Path helper returns correct state-file path based on reconciliation config."""
    # Lazy import so a missing helper produces a test failure, not a module
    # import error that would break ALL tests in this file.
    from fused_memory.server.main import _resolve_curator_escalator_state_path  # noqa: PLC0415

    config = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
    config.reconciliation = reconciliation_cfg
    assert _resolve_curator_escalator_state_path(config) == expected


# ---------------------------------------------------------------------------
# step-1 / step-3: TestCuratorUsageGateLeakGuard
#
# Tests for the extracted _setup_curator_usage_gate helper that guards the
# open CostStore against a UsageGate construction failure.
# ---------------------------------------------------------------------------


class TestCuratorUsageGateLeakGuard:
    """Leak-guard: CostStore.close() must be awaited when UsageGate init raises."""

    @pytest.mark.asyncio
    async def test_setup_closes_cost_store_when_gate_init_raises(self, tmp_path):
        """If UsageGate.__init__ raises, the open CostStore must be closed cleanly.

        This is the primary regression test for the leak bug: before the fix,
        a UsageGate construction failure left the open aiosqlite connection alive
        because _graceful_shutdown never had a chance to run.
        """
        # Lazy import — ImportError here until step-2 adds the helper; that is
        # the expected "red" state for this TDD step.
        from fused_memory.server.main import _setup_curator_usage_gate  # noqa: PLC0415

        acct_cfg = AccountConfig(name='acct-a', oauth_token_env='TEST_TOKEN_ACCT_A')
        usage_cap_cfg = UsageCapConfig(accounts=[acct_cfg], wait_for_reset=False)
        config = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
        config.usage_cap = usage_cap_cfg
        config.reconciliation = MagicMock(data_dir=str(tmp_path))

        # Spy on open() to capture the store instance created inside the helper.
        opened_stores: list[CostStore] = []
        original_open = CostStore.open

        async def capturing_open(self_store: CostStore) -> None:
            opened_stores.append(self_store)
            return await original_open(self_store)

        # Spy on close() to count how many times it was awaited.
        close_count: list[int] = [0]
        original_close = CostStore.close

        async def spy_close(self_store: CostStore) -> None:
            close_count[0] += 1
            return await original_close(self_store)

        with (
            patch.dict(os.environ, {'TEST_TOKEN_ACCT_A': 'fake-token'}),
            patch.object(CostStore, 'open', capturing_open),
            patch.object(CostStore, 'close', spy_close),
            patch.object(
                UsageGate,
                '__init__',
                side_effect=RuntimeError('synthetic init failure'),
            ),
            pytest.raises(RuntimeError, match='synthetic init failure'),
        ):
            await _setup_curator_usage_gate(config)

        assert len(opened_stores) == 1, 'CostStore.open() was not called'
        assert close_count[0] == 1, (
            f'Expected CostStore.close() to be awaited exactly once, got {close_count[0]}'
        )

    @pytest.mark.asyncio
    async def test_setup_closes_resources_when_success_path_logger_raises(self, tmp_path):
        """Success-path logger.info must be inside the try block: both gate.shutdown()
        and store.close() must run when the log call raises unexpectedly.

        On current main the trailing logger.info sits outside the try/except, so
        neither cleanup runs — this test will FAIL until step-2 folds the log
        call into the guarded block.
        """
        import fused_memory.server.main as main_module  # noqa: PLC0415
        from fused_memory.server.main import _setup_curator_usage_gate  # noqa: PLC0415

        acct_cfg = AccountConfig(name='acct-a', oauth_token_env='TEST_TOKEN_ACCT_A')
        usage_cap_cfg = UsageCapConfig(accounts=[acct_cfg], wait_for_reset=False)
        config = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
        config.usage_cap = usage_cap_cfg
        config.reconciliation = MagicMock(data_dir=str(tmp_path))

        # Spy on CostStore.close() to count awaits.
        close_count: list[int] = [0]
        original_close = CostStore.close

        async def spy_close(self_store: CostStore) -> None:
            close_count[0] += 1
            return await original_close(self_store)

        # Spy on UsageGate.shutdown() to count awaits.
        shutdown_count: list[int] = [0]
        original_shutdown = UsageGate.shutdown

        async def spy_shutdown(self_gate: UsageGate) -> None:
            shutdown_count[0] += 1
            return await original_shutdown(self_gate)

        # Patch logger.info to raise only on the success-path gate log call.
        # raiser_invoked tracks whether the raiser actually branched into the raise
        # path, so the test fails loudly if the 'Curator usage gate:' substring
        # ever drifts (rather than silently passing via a never-reached raise).
        raiser_invoked: list[bool] = [False]

        def logger_info_raiser(msg: str, *args: object, **kwargs: object) -> None:
            if 'Curator usage gate:' in str(msg):
                raiser_invoked[0] = True
                raise RuntimeError('synthetic logger failure')

        with (
            patch.dict(os.environ, {'TEST_TOKEN_ACCT_A': 'fake-token'}),
            patch.object(CostStore, 'close', spy_close),
            patch.object(UsageGate, 'shutdown', spy_shutdown),
            patch.object(main_module.logger, 'info', side_effect=logger_info_raiser),
            pytest.raises(RuntimeError, match='synthetic logger failure'),
        ):
            await _setup_curator_usage_gate(config)

        assert raiser_invoked[0], (
            "logger_info_raiser was never triggered on 'Curator usage gate:' — "
            'the log prefix may have drifted; update the substring match'
        )
        assert close_count[0] == 1, (
            f'Expected CostStore.close() to be awaited exactly once, got {close_count[0]}'
        )
        assert shutdown_count[0] == 1, (
            f'Expected UsageGate.shutdown() to be awaited exactly once, '
            f'got {shutdown_count[0]}'
        )

    @pytest.mark.asyncio
    async def test_setup_returns_store_and_gate_on_success(self, tmp_path):
        """Happy path: helper returns both open store and gate; store stays open."""
        from fused_memory.server.main import _setup_curator_usage_gate  # noqa: PLC0415

        acct_cfg = AccountConfig(name='acct-a', oauth_token_env='TEST_TOKEN_ACCT_A')
        usage_cap_cfg = UsageCapConfig(
            accounts=[acct_cfg],
            wait_for_reset=False,
        )
        config = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
        config.usage_cap = usage_cap_cfg
        config.reconciliation = MagicMock(data_dir=str(tmp_path))

        with patch.dict(os.environ, {'TEST_TOKEN_ACCT_A': 'fake-token'}):
            store, gate = await _setup_curator_usage_gate(config)

        try:
            assert store is not None, 'Expected a CostStore, got None'
            assert gate is not None, 'Expected a UsageGate, got None'
            assert gate.account_count == 1, (
                f'Expected 1 account in gate, got {gate.account_count}'
            )
        finally:
            # Clean teardown: close the gate's background tasks and the store.
            if gate is not None:
                await gate.shutdown()
            if store is not None:
                await store.close()

    def test_setup_returns_none_when_usage_cap_is_none(self):
        """Helper returns (None, None) immediately when config.usage_cap is None."""
        import asyncio  # noqa: PLC0415

        from fused_memory.server.main import _setup_curator_usage_gate  # noqa: PLC0415

        config = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
        config.usage_cap = None

        result = asyncio.run(_setup_curator_usage_gate(config))

        assert result == (None, None), f'Expected (None, None), got {result!r}'

    def test_setup_returns_none_when_usage_cap_disabled(self):
        """Helper returns (None, None) immediately when usage_cap.enabled is False."""
        import asyncio  # noqa: PLC0415

        from fused_memory.server.main import _setup_curator_usage_gate  # noqa: PLC0415

        config = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
        config.usage_cap = UsageCapConfig(enabled=False)

        result = asyncio.run(_setup_curator_usage_gate(config))

        assert result == (None, None), f'Expected (None, None), got {result!r}'

    @pytest.mark.asyncio
    async def test_setup_preserves_original_error_when_cleanup_cancelled(self, tmp_path):
        """Original UsageGate init error must survive even if CostStore.close() raises
        CancelledError mid-cleanup.

        CostStore.close is mocked to raise CancelledError, which means the real
        close never runs inside the helper. The capturing_open spy captures the
        store instance so we can release the orphaned aiosqlite connection +
        WAL background thread after the with block exits (where patches are
        gone and the real close is available again).
        """
        from fused_memory.server.main import _setup_curator_usage_gate  # noqa: PLC0415

        acct_cfg = AccountConfig(name='acct-a', oauth_token_env='TEST_TOKEN_ACCT_A')
        usage_cap_cfg = UsageCapConfig(accounts=[acct_cfg], wait_for_reset=False)
        config = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
        config.usage_cap = usage_cap_cfg
        config.reconciliation = MagicMock(data_dir=str(tmp_path))

        # close() raises CancelledError — simulates a cancel arriving mid-cleanup.
        cancel_close = AsyncMock(side_effect=asyncio.CancelledError())

        # Spy on open() to capture the store instance created inside the helper.
        opened_stores: list[CostStore] = []
        original_open = CostStore.open

        async def capturing_open(self_store: CostStore) -> None:
            opened_stores.append(self_store)
            return await original_open(self_store)

        with (
            patch.dict(os.environ, {'TEST_TOKEN_ACCT_A': 'fake-token'}),
            patch.object(CostStore, 'open', capturing_open),
            patch.object(CostStore, 'close', cancel_close),
            patch.object(
                UsageGate,
                '__init__',
                side_effect=RuntimeError('original init failure'),
            ),
            pytest.raises(RuntimeError, match='original init failure'),
        ):
            await _setup_curator_usage_gate(config)

        assert len(opened_stores) == 1, 'CostStore.open() was not called'
        assert cancel_close.await_count == 1, (
            f'Expected CostStore.close() patch to be awaited exactly once, '
            f'got {cancel_close.await_count}'
        )

        # CostStore.close was replaced by an AsyncMock that raises CancelledError, so
        # the real close() never ran inside the helper — the aiosqlite connection +
        # WAL background thread are GUARANTEED to be leaked at this point (the mock
        # replaces the method entirely; there is no path where the helper could have
        # released the connection). Patches are now gone after the with block exits,
        # so this invokes the real CostStore.close() to release the leaked resources.
        # Nothing to assert about leak state: the mock guarantees the leak, so the
        # only useful action is cleanup.
        await opened_stores[0].close()

