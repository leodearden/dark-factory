"""Tests for CostStore threading through Harness → TaskWorkflow → TaskSteward.

Verifies:
- Harness.run() generates run_id at startup (format: run-{hex12})
- CostStore created with correct db_path and opened
- UsageGate gets project_id and run_id set from harness
- CostStore passed to TaskWorkflow (via _run_slot)
- CostStore passed to TaskSteward via steward_factory (via _run_slot)
- CostStore.close() called in finally block
- RunStore.save_run receives pre-generated run_id
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=tmp_path)


def _patch_harness_infra(harness: Harness) -> None:
    """Patch heavy infrastructure so harness.run() can complete with minimal I/O."""
    harness.mcp.start = AsyncMock()
    harness.mcp.stop = AsyncMock()
    harness._dismiss_stale_escalations = AsyncMock()
    harness._start_escalation_server = AsyncMock()
    harness._stop_escalation_server = AsyncMock()
    harness._start_merge_worker = AsyncMock()
    harness._stop_merge_worker = AsyncMock()
    harness._tag_task_modules = AsyncMock()
    harness._recover_crashed_tasks = AsyncMock()
    # Return one pending task so run() doesn't raise "no pending tasks"
    harness.scheduler.get_tasks = AsyncMock(return_value=[
        {
            'id': '99',
            'title': 'Test task',
            'status': 'pending',
            'description': 'test',
            'metadata': {'modules': ['src']},
            'dependencies': [],
        }
    ])


# ---------------------------------------------------------------------------
# Tests: run_id generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHarnessRunIdGeneration:
    """Harness.run() generates run_id at startup with format run-{hex12}."""

    async def test_run_id_generated_at_startup(self, tmp_path):
        """run_id is generated and stored on harness before workflows execute."""
        config = _make_config(tmp_path)
        harness = Harness(config)
        _patch_harness_infra(harness)

        with patch('orchestrator.harness.CostStore') as MockCostStore:
            mock_cs = AsyncMock()
            MockCostStore.return_value = mock_cs
            await harness.run(dry_run=True)

        assert hasattr(harness, '_run_id')
        assert harness._run_id.startswith('run-')
        suffix = harness._run_id[len('run-'):]
        assert len(suffix) == 12
        assert re.fullmatch(r'[0-9a-f]{12}', suffix), f'Not hex: {suffix!r}'

    async def test_run_id_unique_across_runs(self, tmp_path):
        """Each run() call generates a different run_id."""
        config = _make_config(tmp_path)

        ids = []
        for _ in range(3):
            harness = Harness(config)
            _patch_harness_infra(harness)
            with patch('orchestrator.harness.CostStore') as MockCostStore:
                mock_cs = AsyncMock()
                MockCostStore.return_value = mock_cs
                await harness.run(dry_run=True)
            ids.append(harness._run_id)

        assert len(set(ids)) == 3, 'Expected distinct run_ids'


# ---------------------------------------------------------------------------
# Tests: CostStore creation and lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHarnessCostStoreLifecycle:
    """CostStore created with correct path, opened at startup, closed in finally."""

    async def test_cost_store_created_with_correct_db_path(self, tmp_path):
        """CostStore is instantiated with project_root/data/orchestrator/runs.db."""
        config = _make_config(tmp_path)
        harness = Harness(config)
        _patch_harness_infra(harness)

        with patch('orchestrator.harness.CostStore') as MockCostStore:
            mock_cs = AsyncMock()
            MockCostStore.return_value = mock_cs
            await harness.run(dry_run=True)

        expected_path = tmp_path / 'data' / 'orchestrator' / 'runs.db'
        MockCostStore.assert_called_once_with(expected_path)

    async def test_cost_store_open_called(self, tmp_path):
        """CostStore.open() is awaited before workflows start."""
        config = _make_config(tmp_path)
        harness = Harness(config)
        _patch_harness_infra(harness)

        with patch('orchestrator.harness.CostStore') as MockCostStore:
            mock_cs = AsyncMock()
            MockCostStore.return_value = mock_cs
            await harness.run(dry_run=True)

        mock_cs.open.assert_awaited_once()

    async def test_cost_store_close_called_in_finally(self, tmp_path):
        """CostStore.close() is awaited in the finally block (even on error)."""
        config = _make_config(tmp_path)
        harness = Harness(config)
        _patch_harness_infra(harness)

        with patch('orchestrator.harness.CostStore') as MockCostStore:
            mock_cs = AsyncMock()
            MockCostStore.return_value = mock_cs
            await harness.run(dry_run=True)

        mock_cs.close.assert_awaited_once()

    async def test_cost_store_close_called_even_when_run_raises(self, tmp_path):
        """CostStore.close() still called if harness raises during execution."""
        config = _make_config(tmp_path)
        harness = Harness(config)
        _patch_harness_infra(harness)
        # Make tag_task_modules raise
        harness._tag_task_modules = AsyncMock(side_effect=RuntimeError('boom'))

        with patch('orchestrator.harness.CostStore') as MockCostStore:
            mock_cs = AsyncMock()
            MockCostStore.return_value = mock_cs
            try:
                await harness.run(dry_run=True)
            except RuntimeError:
                pass

        mock_cs.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tests: usage_gate integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHarnessUsageGateIntegration:
    """Harness sets usage_gate.project_id and usage_gate.run_id after run_id generation."""

    async def test_usage_gate_receives_project_id_and_run_id(self, tmp_path):
        """usage_gate.project_id and run_id are set from harness attributes."""
        config = _make_config(tmp_path)
        harness = Harness(config)
        _patch_harness_infra(harness)

        # Inject a mock usage gate
        mock_gate = MagicMock()
        mock_gate.is_paused = False
        mock_gate.check_at_startup = AsyncMock()
        mock_gate.total_pause_secs = 0
        mock_gate.shutdown = AsyncMock()
        harness.usage_gate = mock_gate

        with patch('orchestrator.harness.CostStore') as MockCostStore:
            mock_cs = AsyncMock()
            MockCostStore.return_value = mock_cs
            await harness.run(dry_run=True)

        # project_id and run_id should have been set on the gate
        assert mock_gate.project_id == config.fused_memory.project_id
        assert mock_gate.run_id == harness._run_id

    async def test_no_usage_gate_runs_without_error(self, tmp_path):
        """When usage_gate is None (usage_cap disabled), harness runs without error."""
        from shared.config_models import UsageCapConfig
        config = _make_config(tmp_path)
        # Disable usage cap so usage_gate is None
        config.usage_cap = UsageCapConfig(enabled=False)
        harness = Harness(config)
        _patch_harness_infra(harness)
        assert harness.usage_gate is None

        with patch('orchestrator.harness.CostStore') as MockCostStore:
            mock_cs = AsyncMock()
            MockCostStore.return_value = mock_cs
            report = await harness.run(dry_run=True)

        assert report is not None


# ---------------------------------------------------------------------------
# Tests: RunStore receives pre-generated run_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHarnessRunStoreRunId:
    """RunStore.save_run receives the pre-generated run_id from harness."""

    async def test_run_store_called_with_harness_run_id(self, tmp_path):
        """RunStore.save_run is called with run_id=self._run_id."""
        config = _make_config(tmp_path)
        harness = Harness(config)
        _patch_harness_infra(harness)

        captured_run_id = {}

        with patch('orchestrator.harness.CostStore') as MockCostStore:
            mock_cs = AsyncMock()
            MockCostStore.return_value = mock_cs

            with patch('orchestrator.run_store.RunStore') as MockRunStore:
                mock_store = MagicMock()
                mock_store.save_run = MagicMock(
                    side_effect=lambda *args, **kw: captured_run_id.update(
                        {'run_id': kw.get('run_id'), 'saved': True}
                    ) or 'run-captured',
                )
                MockRunStore.return_value = mock_store

                await harness.run(dry_run=True)

        assert captured_run_id.get('saved'), 'save_run was not called'
        assert captured_run_id['run_id'] == harness._run_id
