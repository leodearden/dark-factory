"""Tests for the escalation-watcher-auto subprocess supervisor.

Task 1326 — AFK hardening: escalation-watcher-auto skill + orch subprocess supervisor.

Steps covered by this file:
  step-3: TestWatcherConfig — OrchestratorConfig field presence + defaults
  step-5: TestWatcherSupervisorLifecycle — start/stop/idempotent lifecycle
  step-7: TestRunWatcherRotation — _run_watcher_rotation invoke contract
  step-9: TestWatcherSupervisorLoopClassification — clean/unclean backoff
  step-11: TestWatcherCrashloopTrip — crashloop detection + pause_scheduler
  step-13: TestWatcherSupervisorWiring — __init__ attrs + run() source guard
"""

from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness


# ---------------------------------------------------------------------------
# step-3: Config field presence and defaults
# ---------------------------------------------------------------------------

class TestWatcherConfig:
    """OrchestratorConfig exposes all spec'd watcher_* fields with correct defaults."""

    def test_watcher_supervisor_enabled_default_true(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_supervisor_enabled is True

    def test_watcher_subprocess_restart_backoff_secs_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_subprocess_restart_backoff_secs == 30.0

    def test_watcher_rotation_escalations_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_rotation_escalations == 50

    def test_watcher_rotation_hours_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_rotation_hours == 4.0

    def test_watcher_max_crashloop_restarts_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_max_crashloop_restarts == 5

    def test_watcher_crashloop_window_secs_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_crashloop_window_secs == 600

    # Invocation knobs
    def test_watcher_model_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_model == 'opus'

    def test_watcher_rotation_budget_usd_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_rotation_budget_usd == 40.0

    def test_watcher_max_turns_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_max_turns == 400

    def test_watcher_effort_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_effort == 'high'

    def test_watcher_backend_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_backend == 'claude'


# ---------------------------------------------------------------------------
# step-5: Supervisor lifecycle — start / stop / idempotent
# ---------------------------------------------------------------------------

def _make_lifecycle_harness(tmp_path: Path, *, enabled: bool = True) -> Harness:
    """Build a minimal Harness via __new__ with lifecycle attributes injected."""
    from collections import deque

    h = Harness.__new__(Harness)
    config = OrchestratorConfig(project_root=tmp_path)
    config = config.model_copy(update={'watcher_supervisor_enabled': enabled})
    h.config = config
    h._watcher_supervisor_task = None
    h._watcher_unclean_exits: deque = deque()
    return h


class TestWatcherSupervisorLifecycle:
    """_start/_stop_watcher_supervisor lifecycle.

    Uses MagicMock tasks rather than real asyncio coroutines to avoid the
    unawaited-coroutine warnings that pytest turns into errors
    ('error:coroutine .* was never awaited:RuntimeWarning' filterwarning).
    asyncio.create_task is patched at the module level so no real scheduler
    overhead is incurred either.
    """

    def test_start_noop_when_disabled(self, tmp_path: Path) -> None:
        """When disabled, _start_watcher_supervisor is a no-op."""
        h = _make_lifecycle_harness(tmp_path, enabled=False)
        with patch('orchestrator.harness.asyncio.create_task') as mock_ct:
            h._start_watcher_supervisor()
            mock_ct.assert_not_called()
        assert h._watcher_supervisor_task is None

    def test_start_creates_named_task(self, tmp_path: Path) -> None:
        """When enabled, creates an asyncio.Task named 'watcher-supervisor'."""
        h = _make_lifecycle_harness(tmp_path, enabled=True)
        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.done.return_value = False
        mock_task.get_name.return_value = 'watcher-supervisor'

        # side_effect closes the coroutine immediately to prevent
        # "coroutine was never awaited" RuntimeWarning (which pytest 3.13 turns
        # into a test failure via PytestUnraisableExceptionWarning).
        def _create_task(coro, *, name=None):
            coro.close()
            return mock_task

        with patch('orchestrator.harness.asyncio.create_task', side_effect=_create_task) as mock_ct:
            h._start_watcher_supervisor()

        assert h._watcher_supervisor_task is mock_task
        # Verify create_task was called with name='watcher-supervisor'
        _, kwargs = mock_ct.call_args
        assert kwargs.get('name') == 'watcher-supervisor'

    def test_start_idempotent(self, tmp_path: Path) -> None:
        """A second call while the task is still alive does not create a second task."""
        h = _make_lifecycle_harness(tmp_path, enabled=True)
        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.done.return_value = False  # task is alive

        def _create_task(coro, *, name=None):
            coro.close()
            return mock_task

        with patch('orchestrator.harness.asyncio.create_task', side_effect=_create_task) as mock_ct:
            h._start_watcher_supervisor()
            h._start_watcher_supervisor()  # second call: no-op

        # create_task called exactly once (idempotent)
        mock_ct.assert_called_once()
        assert h._watcher_supervisor_task is mock_task

    @pytest.mark.asyncio
    async def test_stop_cancels_and_resets(self, tmp_path: Path) -> None:
        """_stop_watcher_supervisor cancels the task and resets to None."""
        h = _make_lifecycle_harness(tmp_path, enabled=True)

        # Use a real asyncio Task so await-on-cancel works without
        # triggering unawaited-coroutine warnings (instance-level __await__
        # is ignored by Python's dunder lookup which always goes through type).
        async def _eternal() -> None:
            await asyncio.sleep(10_000)

        task = asyncio.create_task(_eternal())
        h._watcher_supervisor_task = task
        await h._stop_watcher_supervisor()

        assert task.cancelled()
        assert h._watcher_supervisor_task is None

    @pytest.mark.asyncio
    async def test_stop_noop_when_none(self, tmp_path: Path) -> None:
        """_stop_watcher_supervisor with no task is a no-op."""
        h = _make_lifecycle_harness(tmp_path)
        # Should not raise
        await h._stop_watcher_supervisor()


# ---------------------------------------------------------------------------
# step-7: _run_watcher_rotation invoke contract
# ---------------------------------------------------------------------------

def _make_rotation_harness(tmp_path: Path) -> Harness:
    """Minimal Harness for _run_watcher_rotation tests."""
    from collections import deque

    h = Harness.__new__(Harness)
    config = OrchestratorConfig(project_root=tmp_path)
    h.config = config
    h._watcher_supervisor_task = None
    h._watcher_unclean_exits: deque = deque()
    h.usage_gate = None
    h.cost_store = MagicMock()
    h._run_id = 'run-test-rotation-001'
    # Stub mcp: mcp_config_json returns a sentinel dict
    h.mcp = MagicMock()
    h.mcp.mcp_config_json.return_value = {'mcp': 'config-sentinel'}
    return h


class TestRunWatcherRotation:
    """_run_watcher_rotation invoke contract tests."""

    @pytest.mark.asyncio
    async def test_calls_load_skill_system_prompt(self, tmp_path: Path) -> None:
        """system_prompt comes from load_skill_system_prompt('escalation-watcher-auto')."""
        from shared.cli_invoke import AgentResult

        h = _make_rotation_harness(tmp_path)
        captured: dict = {}

        async def fake_invoke_with_cap_retry(usage_gate, label, *, invoke_fn, **kwargs):
            captured['system_prompt'] = kwargs.get('system_prompt', '')
            return AgentResult(success=True, output='')

        with patch('orchestrator.harness.invoke_with_cap_retry', fake_invoke_with_cap_retry):
            from orchestrator.agents.skill_prompt import load_skill_system_prompt
            expected = load_skill_system_prompt('escalation-watcher-auto')
            await h._run_watcher_rotation()

        assert captured['system_prompt'] == expected

    @pytest.mark.asyncio
    async def test_role_and_cost_store(self, tmp_path: Path) -> None:
        """Invoked with role='escalation-watcher-auto' and the harness's cost_store."""
        from shared.cli_invoke import AgentResult

        h = _make_rotation_harness(tmp_path)
        captured: dict = {}

        async def fake_invoke(usage_gate, label, *, invoke_fn, cost_store, role, run_id, project_id, **kwargs):
            captured['role'] = role
            captured['cost_store'] = cost_store
            captured['run_id'] = run_id
            captured['project_id'] = project_id
            captured['invoke_fn'] = invoke_fn
            return AgentResult(success=True, output='')

        with patch('orchestrator.harness.invoke_with_cap_retry', fake_invoke):
            await h._run_watcher_rotation()

        assert captured['role'] == 'escalation-watcher-auto'
        assert captured['cost_store'] is h.cost_store
        assert captured['run_id'] == 'run-test-rotation-001'
        assert captured['project_id'] == h.config.fused_memory.project_id

    @pytest.mark.asyncio
    async def test_invoke_fn_is_invoke_agent(self, tmp_path: Path) -> None:
        """invoke_fn must be the orchestrator's invoke_agent."""
        from orchestrator.agents.invoke import invoke_agent
        from shared.cli_invoke import AgentResult

        h = _make_rotation_harness(tmp_path)
        captured: dict = {}

        async def fake_invoke(usage_gate, label, *, invoke_fn, **kwargs):
            captured['invoke_fn'] = invoke_fn
            return AgentResult(success=True, output='')

        with patch('orchestrator.harness.invoke_with_cap_retry', fake_invoke):
            await h._run_watcher_rotation()

        assert captured['invoke_fn'] is invoke_agent

    @pytest.mark.asyncio
    async def test_timeout_seconds_includes_grace(self, tmp_path: Path) -> None:
        """timeout_seconds = rotation_hours * 3600 + grace (> 0)."""
        from shared.cli_invoke import AgentResult

        h = _make_rotation_harness(tmp_path)
        captured: dict = {}

        async def fake_invoke(usage_gate, label, *, invoke_fn, **kwargs):
            captured['timeout_seconds'] = kwargs.get('timeout_seconds')
            return AgentResult(success=True, output='')

        with patch('orchestrator.harness.invoke_with_cap_retry', fake_invoke):
            await h._run_watcher_rotation()

        expected_min = h.config.watcher_rotation_hours * 3600
        assert captured['timeout_seconds'] is not None
        assert captured['timeout_seconds'] > expected_min, (
            'timeout_seconds must be > rotation_hours*3600 (grace not added)'
        )

    @pytest.mark.asyncio
    async def test_budget_and_model(self, tmp_path: Path) -> None:
        """max_budget_usd and model come from config knobs."""
        from shared.cli_invoke import AgentResult

        h = _make_rotation_harness(tmp_path)
        captured: dict = {}

        async def fake_invoke(usage_gate, label, *, invoke_fn, **kwargs):
            captured.update(kwargs)
            return AgentResult(success=True, output='')

        with patch('orchestrator.harness.invoke_with_cap_retry', fake_invoke):
            await h._run_watcher_rotation()

        assert captured['max_budget_usd'] == h.config.watcher_rotation_budget_usd
        assert captured['model'] == h.config.watcher_model

    @pytest.mark.asyncio
    async def test_user_prompt_contains_rotation_limits(self, tmp_path: Path) -> None:
        """User prompt embeds ROTATION_ESCALATIONS and ROTATION_HOURS values."""
        from shared.cli_invoke import AgentResult

        h = _make_rotation_harness(tmp_path)
        captured: dict = {}

        async def fake_invoke(usage_gate, label, *, invoke_fn, **kwargs):
            captured['prompt'] = kwargs.get('prompt', '')
            return AgentResult(success=True, output='')

        with patch('orchestrator.harness.invoke_with_cap_retry', fake_invoke):
            await h._run_watcher_rotation()

        prompt = captured['prompt']
        assert str(h.config.watcher_rotation_escalations) in prompt, (
            'ROTATION_ESCALATIONS value not found in user prompt'
        )
        assert str(h.config.watcher_rotation_hours) in prompt, (
            'ROTATION_HOURS value not found in user prompt'
        )

    @pytest.mark.asyncio
    async def test_mcp_config_from_escalation_url(self, tmp_path: Path) -> None:
        """mcp_config comes from self.mcp.mcp_config_json with escalation_url."""
        from shared.cli_invoke import AgentResult

        h = _make_rotation_harness(tmp_path)
        captured: dict = {}

        async def fake_invoke(usage_gate, label, *, invoke_fn, **kwargs):
            captured.update(kwargs)
            return AgentResult(success=True, output='')

        with patch('orchestrator.harness.invoke_with_cap_retry', fake_invoke):
            await h._run_watcher_rotation()

        expected_url = f'http://{h.config.escalation.host}:{h.config.escalation.port}/mcp'
        h.mcp.mcp_config_json.assert_called_once_with(escalation_url=expected_url)
        assert captured.get('mcp_config') == {'mcp': 'config-sentinel'}
