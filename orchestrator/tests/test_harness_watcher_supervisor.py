"""Tests for the escalation-watcher-auto subprocess supervisor.

Task 1326 — AFK hardening: escalation-watcher-auto skill + orch subprocess supervisor.

Steps covered by this file:
  step-3:  TestWatcherConfig — OrchestratorConfig field presence + defaults
  step-5:  TestWatcherSupervisorLifecycle — start/stop/idempotent lifecycle
  step-7:  TestRunWatcherRotation — _run_watcher_rotation invoke contract
  step-9:  TestWatcherSupervisorLoopClassification — clean/unclean backoff
  step-11: TestWatcherCrashloopTrip — crashloop detection + pause_scheduler
  step-13: TestWatcherSupervisorWiring — __init__ attrs + bound-method checks
  step-15: TestWatcherSupervisorRunWiring — behavioral run()-lifecycle ordering
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

    # Misconfigured-clean-exit cost-runaway guard (task 1388)
    def test_watcher_misconfigured_min_rotation_secs_default(self, tmp_path: Path) -> None:
        """Clean rotations shorter than this are classified as degenerate (misconfigured)."""
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_misconfigured_min_rotation_secs == 120.0

    def test_watcher_max_misconfigured_clean_exits_default(self, tmp_path: Path) -> None:
        """After this many degenerate clean exits in the window, trip watcher_misconfigured."""
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.watcher_max_misconfigured_clean_exits == 5


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
    h._watcher_unclean_exits = deque()
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
    h._watcher_unclean_exits = deque()
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
        from shared.cli_invoke import AgentResult

        from orchestrator.agents.invoke import invoke_agent

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
        h.mcp.mcp_config_json.assert_called_once_with(escalation_url=expected_url)  # type: ignore[union-attr]
        assert captured.get('mcp_config') == {'mcp': 'config-sentinel'}

    @pytest.mark.asyncio
    async def test_tool_restrictions_passed(self, tmp_path: Path) -> None:
        """invoke_with_cap_retry receives non-empty allowed_tools and disallowed_tools.

        Defence-in-depth: the autonomous agent runs unattended for up to 4h, so
        its tool surface must be explicitly bounded (per UnblockAutoConfig precedent).
        Edit and Write must appear in disallowed_tools; mcp__escalation__resolve_issue
        must appear in allowed_tools (autonomous dispatch path).
        """
        from shared.cli_invoke import AgentResult

        h = _make_rotation_harness(tmp_path)
        captured: dict = {}

        async def fake_invoke(usage_gate, label, *, invoke_fn, **kwargs):
            captured['allowed_tools'] = kwargs.get('allowed_tools')
            captured['disallowed_tools'] = kwargs.get('disallowed_tools')
            return AgentResult(success=True, output='')

        with patch('orchestrator.harness.invoke_with_cap_retry', fake_invoke):
            await h._run_watcher_rotation()

        allowed = captured.get('allowed_tools')
        disallowed = captured.get('disallowed_tools')

        assert allowed is not None and len(allowed) > 0, (
            'allowed_tools must be a non-empty list'
        )
        assert disallowed is not None and len(disallowed) > 0, (
            'disallowed_tools must be a non-empty list'
        )
        # Autonomous dispatch requires escalation resolution
        assert any('resolve_issue' in t for t in allowed), (
            'mcp__escalation__resolve_issue must be in allowed_tools for autonomous dispatch'
        )
        # Code-edit tools must be blocked
        assert 'Edit' in disallowed, 'Edit must be in disallowed_tools (no code edits)'
        assert 'Write' in disallowed, 'Write must be in disallowed_tools (no code edits)'


# ---------------------------------------------------------------------------
# step-9: Supervisor loop classification — clean/unclean backoff
# ---------------------------------------------------------------------------

def _make_loop_harness(tmp_path: Path) -> Harness:
    """Minimal Harness for _watcher_supervisor_loop tests."""
    from collections import deque

    h = Harness.__new__(Harness)
    config = OrchestratorConfig(project_root=tmp_path)
    h.config = config
    h._watcher_supervisor_task = None
    h._watcher_unclean_exits = deque()
    h._watcher_degenerate_clean_exits = deque()  # cost-runaway guard (task 1388)
    h.usage_gate = None
    h.cost_store = MagicMock()
    h._run_id = 'run-loop-test-001'
    h.mcp = MagicMock()
    h.mcp.mcp_config_json.return_value = {'mcp': 'stub'}
    return h


class TestWatcherSupervisorLoopClassification:
    """_watcher_supervisor_loop clean-vs-unclean classification and backoff.

    Strategy: patch _run_watcher_rotation to return controlled results and
    asyncio.sleep in the harness module to record calls + raise StopAsyncIteration
    after a fixed number of iterations to break the loop under test.
    """

    @pytest.mark.asyncio
    async def test_clean_exit_floor_sleep_no_unclean(self, tmp_path: Path) -> None:
        """Clean exit (success=True, timed_out=False) → floor sleep of restart_backoff_secs,
        no exponential backoff, and the unclean-exit deque stays empty.

        The floor sleep on the clean path prevents back-to-back opus invocations when
        the agent self-exits near-instantly (misconfiguration guard).
        """
        from shared.cli_invoke import AgentResult

        h = _make_loop_harness(tmp_path)
        rotation_calls = 0
        sleep_durations: list[float] = []

        # rotation always succeeds cleanly; after 2 calls break the loop
        async def fake_rotation() -> AgentResult:
            nonlocal rotation_calls
            rotation_calls += 1
            if rotation_calls >= 2:
                raise asyncio.CancelledError()
            return AgentResult(success=True, output='', timed_out=False)

        async def fake_sleep(duration: float) -> None:
            sleep_durations.append(duration)

        h._run_watcher_rotation = fake_rotation  # type: ignore[method-assign]

        with patch('orchestrator.harness.asyncio.sleep', fake_sleep), pytest.raises(asyncio.CancelledError):
            await h._watcher_supervisor_loop()

        # At least one rotation ran
        assert rotation_calls >= 1
        # A floor sleep should occur after the clean rotation (cost-runaway guard).
        assert len(sleep_durations) == 1, (
            f'Expected exactly 1 floor sleep after clean exit; got {sleep_durations}'
        )
        assert sleep_durations[0] == pytest.approx(
            h.config.watcher_subprocess_restart_backoff_secs
        ), (
            f'Floor sleep should equal restart_backoff_secs '
            f'({h.config.watcher_subprocess_restart_backoff_secs}); got {sleep_durations[0]}'
        )
        # Unclean-exit deque must stay empty (clean path, not reclassified)
        assert len(h._watcher_unclean_exits) == 0

    @pytest.mark.asyncio
    async def test_unclean_exit_triggers_backoff(self, tmp_path: Path) -> None:
        """Unclean exit (success=False) → backoff sleep of base secs, deque grows."""
        from shared.cli_invoke import AgentResult

        h = _make_loop_harness(tmp_path)
        rotation_calls = 0
        sleep_durations: list[float] = []

        async def fake_rotation() -> AgentResult:
            nonlocal rotation_calls
            rotation_calls += 1
            if rotation_calls >= 2:
                raise asyncio.CancelledError()
            return AgentResult(success=False, output='', timed_out=False)

        async def fake_sleep(duration: float) -> None:
            sleep_durations.append(duration)

        h._run_watcher_rotation = fake_rotation  # type: ignore[method-assign]

        with patch('orchestrator.harness.asyncio.sleep', fake_sleep), pytest.raises(asyncio.CancelledError):
            await h._watcher_supervisor_loop()

        base = h.config.watcher_subprocess_restart_backoff_secs
        assert len(sleep_durations) >= 1, 'Expected backoff sleep after unclean exit'
        assert sleep_durations[0] == pytest.approx(base), (
            f'First backoff should equal base {base}s; got {sleep_durations[0]}'
        )
        assert len(h._watcher_unclean_exits) >= 1

    @pytest.mark.asyncio
    async def test_timed_out_counts_as_unclean(self, tmp_path: Path) -> None:
        """timed_out=True (even if success=True) counts as an unclean exit."""
        from shared.cli_invoke import AgentResult

        h = _make_loop_harness(tmp_path)
        rotation_calls = 0
        sleep_durations: list[float] = []

        async def fake_rotation() -> AgentResult:
            nonlocal rotation_calls
            rotation_calls += 1
            if rotation_calls >= 2:
                raise asyncio.CancelledError()
            # timed_out=True → unclean regardless of success flag
            return AgentResult(success=True, output='', timed_out=True)

        async def fake_sleep(duration: float) -> None:
            sleep_durations.append(duration)

        h._run_watcher_rotation = fake_rotation  # type: ignore[method-assign]

        with patch('orchestrator.harness.asyncio.sleep', fake_sleep), pytest.raises(asyncio.CancelledError):
            await h._watcher_supervisor_loop()

        assert len(sleep_durations) >= 1, 'timed_out exit must trigger backoff'
        assert len(h._watcher_unclean_exits) >= 1

    @pytest.mark.asyncio
    async def test_consecutive_unclean_doubles_backoff(self, tmp_path: Path) -> None:
        """Second consecutive unclean exit doubles the backoff (exponential)."""
        from shared.cli_invoke import AgentResult

        # Use a short enough window to not evict entries
        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_crashloop_restarts': 99,  # disable crashloop trip
            'watcher_crashloop_window_secs': 9999,
        })
        rotation_calls = 0
        sleep_durations: list[float] = []

        async def fake_rotation() -> AgentResult:
            nonlocal rotation_calls
            rotation_calls += 1
            if rotation_calls >= 3:
                raise asyncio.CancelledError()
            return AgentResult(success=False, output='')

        async def fake_sleep(duration: float) -> None:
            sleep_durations.append(duration)

        h._run_watcher_rotation = fake_rotation  # type: ignore[method-assign]

        with patch('orchestrator.harness.asyncio.sleep', fake_sleep), pytest.raises(asyncio.CancelledError):
            await h._watcher_supervisor_loop()

        assert len(sleep_durations) >= 2, (
            f'Expected 2 backoff sleeps for 2 unclean exits; got {sleep_durations}'
        )
        base = h.config.watcher_subprocess_restart_backoff_secs
        assert sleep_durations[0] == pytest.approx(base)
        # Second sleep should be larger (exponential — at least 1.5x base)
        assert sleep_durations[1] >= base * 1.5, (
            f'Second backoff {sleep_durations[1]} should be > 1.5*base {base*1.5}'
        )

    @pytest.mark.asyncio
    async def test_clean_exit_resets_backoff(self, tmp_path: Path) -> None:
        """A clean exit after unclean exits resets the backoff to base.

        Sequence unclean→clean→unclean produces 3 sleeps:
          [unclean_backoff, clean_floor, unclean_base_again]
        The clean-path floor sleep uses watcher_subprocess_restart_backoff_secs (same
        value as base backoff), so all three sleeps are equal to base.  The key
        assertion is that the unclean after the clean reset uses base (not doubled).
        """
        from shared.cli_invoke import AgentResult

        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_crashloop_restarts': 99,
            'watcher_crashloop_window_secs': 9999,
        })
        # Sequence: unclean, clean, unclean, then cancel
        results = [
            AgentResult(success=False, output=''),   # unclean → backoff(base)
            AgentResult(success=True, output=''),    # clean   → reset + floor(base)
            AgentResult(success=False, output=''),   # unclean → base backoff again
        ]
        idx = 0
        sleep_durations: list[float] = []

        async def fake_rotation() -> AgentResult:
            nonlocal idx
            if idx >= len(results):
                raise asyncio.CancelledError()
            r = results[idx]
            idx += 1
            return r

        async def fake_sleep(duration: float) -> None:
            sleep_durations.append(duration)

        h._run_watcher_rotation = fake_rotation  # type: ignore[method-assign]

        with patch('orchestrator.harness.asyncio.sleep', fake_sleep), pytest.raises(asyncio.CancelledError):
            await h._watcher_supervisor_loop()

        base = h.config.watcher_subprocess_restart_backoff_secs
        # 3 sleeps: unclean backoff | clean floor | unclean base-reset
        assert len(sleep_durations) == 3, (
            f'Expected 3 sleeps (unclean/clean-floor/unclean); got {sleep_durations}'
        )
        # The clean-path floor and the unclean base happen to be the same value.
        # The critical assertion: post-reset unclean uses base (not doubled).
        assert sleep_durations[2] == pytest.approx(base), (
            f'Post-reset backoff should be base {base}s; got {sleep_durations[2]}'
        )


# ---------------------------------------------------------------------------
# step-11: Crashloop trip — pause_scheduler + window eviction
# ---------------------------------------------------------------------------

class TestWatcherCrashloopTrip:
    """Crashloop detection: N unclean exits in a window trips pause_scheduler."""

    @pytest.mark.asyncio
    async def test_crashloop_trips_pause_scheduler(self, tmp_path: Path) -> None:
        """After max_crashloop_restarts unclean exits within the window, pause_scheduler
        is called exactly once with reason='watcher_crashloop' and the loop exits."""
        from shared.cli_invoke import AgentResult

        max_restarts = 3
        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_crashloop_restarts': max_restarts,
            'watcher_crashloop_window_secs': 600,
            'watcher_subprocess_restart_backoff_secs': 0.0,
        })

        rotation_calls = 0
        pause_calls: list[str] = []

        async def fake_rotation() -> AgentResult:
            nonlocal rotation_calls
            rotation_calls += 1
            return AgentResult(success=False, output='')

        async def fake_pause_scheduler(reason: str) -> None:
            pause_calls.append(reason)

        async def fake_sleep(duration: float) -> None:
            pass  # instant, no blocking

        h._run_watcher_rotation = fake_rotation          # type: ignore[method-assign]
        h.pause_scheduler = fake_pause_scheduler         # type: ignore[method-assign]

        # patch monotonic to return a stable time (all exits within the window)
        import time as _time_mod
        stable_time = _time_mod.monotonic()
        with patch('orchestrator.harness.asyncio.sleep', fake_sleep), \
             patch('orchestrator.harness.time.monotonic', return_value=stable_time):
            # Loop should exit after max_restarts unclean exits
            await h._watcher_supervisor_loop()

        # pause_scheduler called exactly once with the crashloop reason
        assert pause_calls == ['watcher_crashloop'], (
            f'Expected pause_scheduler("watcher_crashloop") once; got {pause_calls}'
        )
        # Loop exited (returned) — no further rotations after the trip
        assert rotation_calls == max_restarts, (
            f'Expected exactly {max_restarts} rotations before trip; got {rotation_calls}'
        )

    @pytest.mark.asyncio
    async def test_crashloop_does_not_trip_below_threshold(self, tmp_path: Path) -> None:
        """Fewer than max_crashloop_restarts unclean exits do NOT trip pause_scheduler."""
        from shared.cli_invoke import AgentResult

        max_restarts = 3
        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_crashloop_restarts': max_restarts,
            'watcher_crashloop_window_secs': 600,
            'watcher_subprocess_restart_backoff_secs': 0.0,
        })

        rotation_calls = 0
        pause_calls: list[str] = []

        async def fake_rotation() -> AgentResult:
            nonlocal rotation_calls
            rotation_calls += 1
            if rotation_calls >= max_restarts:  # cancel before trip
                raise asyncio.CancelledError()
            return AgentResult(success=False, output='')

        async def fake_pause_scheduler(reason: str) -> None:
            pause_calls.append(reason)

        h._run_watcher_rotation = fake_rotation          # type: ignore[method-assign]
        h.pause_scheduler = fake_pause_scheduler         # type: ignore[method-assign]

        with patch('orchestrator.harness.asyncio.sleep', AsyncMock()), pytest.raises(asyncio.CancelledError):
            await h._watcher_supervisor_loop()

        assert pause_calls == [], (
            f'pause_scheduler should NOT be called below threshold; got {pause_calls}'
        )

    @pytest.mark.asyncio
    async def test_old_exits_outside_window_are_evicted(self, tmp_path: Path) -> None:
        """Unclean exits older than watcher_crashloop_window_secs are evicted
        so they do not contribute to the crashloop count."""
        import time as _time_mod

        from shared.cli_invoke import AgentResult

        max_restarts = 3
        window = 600
        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_crashloop_restarts': max_restarts,
            'watcher_crashloop_window_secs': window,
            'watcher_subprocess_restart_backoff_secs': 0.0,
        })

        pause_calls: list[str] = []
        rotation_calls = 0

        # Monotonic clock advances to put first exits outside the window.
        # Sequence: 2 old exits (at t=0), then clock jumps past window, then
        # (max_restarts - 1) exits (insufficient alone to trip), then cancel.
        old_time = _time_mod.monotonic()
        new_time = old_time + window + 1  # beyond the window
        time_sequence = iter(
            [old_time, old_time]            # first 2 unclean exits: old
            + [new_time] * (max_restarts)   # subsequent exits: recent
        )

        async def fake_rotation() -> AgentResult:
            nonlocal rotation_calls
            rotation_calls += 1
            if rotation_calls > max_restarts + 1:
                raise asyncio.CancelledError()
            return AgentResult(success=False, output='')

        async def fake_pause_scheduler(reason: str) -> None:
            pause_calls.append(reason)

        h._run_watcher_rotation = fake_rotation          # type: ignore[method-assign]
        h.pause_scheduler = fake_pause_scheduler         # type: ignore[method-assign]

        def fake_monotonic() -> float:
            try:
                return next(time_sequence)
            except StopIteration:
                return new_time

        with (
            patch('orchestrator.harness.asyncio.sleep', AsyncMock()),
            patch('orchestrator.harness.time.monotonic', side_effect=fake_monotonic),
            pytest.raises(asyncio.CancelledError),
        ):
            # Loop should cancel (not trip) because old exits are evicted
            await h._watcher_supervisor_loop()

        assert pause_calls == [], (
            'Old exits outside window should be evicted; pause_scheduler must NOT trip'
        )


# ---------------------------------------------------------------------------
# task-1388: Misconfigured-clean-exit cost-runaway guard
# ---------------------------------------------------------------------------

class TestWatcherMisconfiguredGuard:
    """Fast degenerate-clean exit guard: trips watcher_misconfigured after N fast-clean exits.

    A 'degenerate clean' exit is one where success=True but the rotation
    completed in less than watcher_misconfigured_min_rotation_secs seconds
    (indicating an empty queue, SKILL.md drift, or misconfigured env).
    """

    @pytest.mark.asyncio
    async def test_fast_clean_exit_appends_degenerate_deque(self, tmp_path: Path) -> None:
        """A fast clean exit (duration < min_rotation_secs) appends to _watcher_degenerate_clean_exits."""
        import time as _time_mod

        from shared.cli_invoke import AgentResult

        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_misconfigured_clean_exits': 99,  # disable trip during this test
            'watcher_subprocess_restart_backoff_secs': 0.0,
        })

        # Rotation duration 1.0s — well under default 120s threshold
        t0 = _time_mod.monotonic()
        time_seq = iter([t0, t0 + 1.0])  # start, end for one rotation

        def fake_monotonic() -> float:
            try:
                return next(time_seq)
            except StopIteration:
                return t0 + 1.0

        rotation_calls = 0

        async def fake_rotation() -> AgentResult:
            nonlocal rotation_calls
            rotation_calls += 1
            if rotation_calls >= 2:
                raise asyncio.CancelledError()
            return AgentResult(success=True, output='', timed_out=False)

        h._run_watcher_rotation = fake_rotation  # type: ignore[method-assign]

        with (
            patch('orchestrator.harness.asyncio.sleep', AsyncMock()),
            patch('orchestrator.harness.time.monotonic', side_effect=fake_monotonic),
            pytest.raises(asyncio.CancelledError),
        ):
            await h._watcher_supervisor_loop()

        assert len(h._watcher_degenerate_clean_exits) == 1, (
            f'Expected 1 degenerate-clean entry after fast rotation; '
            f'got {len(h._watcher_degenerate_clean_exits)}'
        )

    @pytest.mark.asyncio
    async def test_slow_clean_exit_does_not_append(self, tmp_path: Path) -> None:
        """A slow clean exit (duration >= min_rotation_secs) does NOT append to the degenerate deque."""
        import time as _time_mod

        from shared.cli_invoke import AgentResult

        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_misconfigured_min_rotation_secs': 120.0,
            'watcher_max_misconfigured_clean_exits': 99,
            'watcher_subprocess_restart_backoff_secs': 0.0,
        })

        # Rotation duration 200s — above the 120s threshold
        t0 = _time_mod.monotonic()
        time_seq = iter([t0, t0 + 200.0])  # start, end for one rotation

        def fake_monotonic() -> float:
            try:
                return next(time_seq)
            except StopIteration:
                return t0 + 200.0

        rotation_calls = 0

        async def fake_rotation() -> AgentResult:
            nonlocal rotation_calls
            rotation_calls += 1
            if rotation_calls >= 2:
                raise asyncio.CancelledError()
            return AgentResult(success=True, output='', timed_out=False)

        h._run_watcher_rotation = fake_rotation  # type: ignore[method-assign]

        with (
            patch('orchestrator.harness.asyncio.sleep', AsyncMock()),
            patch('orchestrator.harness.time.monotonic', side_effect=fake_monotonic),
            pytest.raises(asyncio.CancelledError),
        ):
            await h._watcher_supervisor_loop()

        assert len(h._watcher_degenerate_clean_exits) == 0, (
            f'Expected 0 degenerate-clean entries for slow rotation; '
            f'got {len(h._watcher_degenerate_clean_exits)}'
        )


# ---------------------------------------------------------------------------
# step-13: Wiring — __init__ attrs + run() source guard
# ---------------------------------------------------------------------------

def _make_real_harness(tmp_path: Path) -> Harness:
    """Construct a real Harness via __init__ with infrastructure classes mocked.

    Patches McpLifecycle/Scheduler/BriefingAssembler so no real servers start,
    but __init__ runs fully — including the attribute assignments under test.
    """
    config = OrchestratorConfig(project_root=tmp_path)
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        return Harness(config)


class TestWatcherSupervisorWiring:
    """Verify real Harness.__init__ sets the expected attributes and bound methods exist.

    Uses _make_real_harness (real __init__ with infrastructure mocked) instead of
    _make_lifecycle_harness (__new__ + manual injection) so the assertions actually
    guard against __init__ failing to initialize these attributes.
    """

    def test_init_sets_watcher_supervisor_task_none(self, tmp_path: Path) -> None:
        """Real Harness.__init__ sets _watcher_supervisor_task=None."""
        h = _make_real_harness(tmp_path)
        assert h._watcher_supervisor_task is None

    def test_init_sets_watcher_unclean_exits_empty_deque(self, tmp_path: Path) -> None:
        """Real Harness.__init__ sets _watcher_unclean_exits to an empty deque."""
        h = _make_real_harness(tmp_path)
        assert isinstance(h._watcher_unclean_exits, deque)
        assert len(h._watcher_unclean_exits) == 0

    def test_init_sets_watcher_degenerate_clean_exits_empty_deque(self, tmp_path: Path) -> None:
        """Real Harness.__init__ sets _watcher_degenerate_clean_exits to an empty deque.

        This deque is the cost-runaway guard for fast-clean exits (task 1388).
        """
        h = _make_real_harness(tmp_path)
        assert isinstance(h._watcher_degenerate_clean_exits, deque)
        assert len(h._watcher_degenerate_clean_exits) == 0

    def test_start_and_stop_are_bound_methods(self, tmp_path: Path) -> None:
        """_start_watcher_supervisor and _stop_watcher_supervisor are callable."""
        h = _make_lifecycle_harness(tmp_path)
        assert callable(h._start_watcher_supervisor)
        assert callable(h._stop_watcher_supervisor)


# ---------------------------------------------------------------------------
# step-15: Behavioral run()-lifecycle ordering test
# ---------------------------------------------------------------------------


def _make_run_wired_harness(tmp_path: Path) -> tuple:
    """Real Harness with all run() side-effects stubbed + shared-parent call-order mock.

    Uses a real OrchestratorConfig (like test_harness_park_stop._make_harness_with_mocks)
    to avoid spec_set restrictions on properties not in model_fields.  Patches
    McpLifecycle/Scheduler/BriefingAssembler during construction so no real servers start.

    Returns ``(harness, parent_mock)`` where ``parent_mock.mock_calls`` records
    every attached child-mock call in chronological order.
    """
    config = OrchestratorConfig(project_root=tmp_path)

    with patch('orchestrator.harness.McpLifecycle') as mock_mcp_cls, \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(config)

    mock_mcp = mock_mcp_cls.return_value
    mock_mcp.start = AsyncMock()
    mock_mcp.stop = AsyncMock()

    h.git_ops = MagicMock()
    h.git_ops.has_dirty_working_tree = AsyncMock(return_value=None)
    h.git_ops.worktree_base = tmp_path / '.worktrees'

    # Mock all startup side-effect methods not under call-order observation.
    h._start_escalation_server = AsyncMock()
    h._start_merge_worker = AsyncMock()
    h._tag_task_modules = AsyncMock()
    h._recover_crashed_tasks = AsyncMock()
    h._reconcile_stranded_in_progress = AsyncMock()
    h._tag_prd_metadata = AsyncMock()
    h._start_orphan_l0_reaper = MagicMock()
    h._start_stranded_reconcile = MagicMock()

    # Scheduler: one pending task so run() proceeds to the acquire_next loop,
    # which then raises RuntimeError('stop') to halt immediately after startup.
    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[{'id': '1', 'status': 'pending'}])
    h.scheduler.get_statuses = AsyncMock(return_value=({'1': 'pending'}, None))
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.acquire_next = AsyncMock(side_effect=RuntimeError('stop'))

    # Shared parent mock — all attached children's calls are recorded here in
    # chronological order, enabling relative-ordering assertions without reading
    # source text.
    parent = MagicMock(name='lifecycle_order')

    # Children: AsyncMock for async methods, MagicMock for sync.
    dismiss_mock = AsyncMock()
    start_terminal_mock = MagicMock()
    start_watcher_mock = MagicMock()
    stop_terminal_mock = AsyncMock()
    stop_watcher_mock = AsyncMock()
    mcp_stop_mock = AsyncMock()

    parent.attach_mock(dismiss_mock, '_dismiss_stale_escalations')
    parent.attach_mock(start_terminal_mock, '_start_terminal_status_watcher')
    parent.attach_mock(start_watcher_mock, '_start_watcher_supervisor')
    parent.attach_mock(stop_terminal_mock, '_stop_terminal_status_watcher')
    parent.attach_mock(stop_watcher_mock, '_stop_watcher_supervisor')
    parent.attach_mock(mcp_stop_mock, 'mcp_stop')

    h._dismiss_stale_escalations = dismiss_mock
    h._start_terminal_status_watcher = start_terminal_mock
    h._start_watcher_supervisor = start_watcher_mock
    h._stop_terminal_status_watcher = stop_terminal_mock
    h._stop_watcher_supervisor = stop_watcher_mock
    mock_mcp.stop = mcp_stop_mock

    return h, parent


class TestWatcherSupervisorRunWiring:
    """Behavioral run()-lifecycle test replacing the brittle source-guard meta-tests.

    Drives a real ``Harness.run()`` with all side-effects stubbed and verifies
    that ``_start_watcher_supervisor`` / ``_stop_watcher_supervisor`` are
    actually invoked — and in the correct startup/shutdown phase — via a shared
    parent mock's ``mock_calls`` list (chronological call order).

    Replaces the two source-introspection guards removed from
    ``TestWatcherSupervisorWiring`` (step-13): those used ``source.find()`` on
    harness.py text and matched the method name in docstrings/comments, making
    them pass even when ``run()`` never called the methods, and breaking on
    behaviour-preserving refactors. Step-16 proves these replacements detect
    un-wiring.
    """

    @pytest.mark.asyncio
    async def test_start_watcher_supervisor_called_once(self, tmp_path: Path) -> None:
        """_start_watcher_supervisor is called exactly once during run()."""
        h, _parent = _make_run_wired_harness(tmp_path)
        with pytest.raises(RuntimeError):
            await h.run(prd_path=None)
        h._start_watcher_supervisor.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_watcher_supervisor_after_dismiss_stale(self, tmp_path: Path) -> None:
        """Startup block: _start_watcher_supervisor appears AFTER _dismiss_stale_escalations
        in the shared parent's mock_calls (chronological call order)."""
        h, parent = _make_run_wired_harness(tmp_path)
        with pytest.raises(RuntimeError):
            await h.run(prd_path=None)

        call_strs = [str(c) for c in parent.mock_calls]

        dismiss_idx = next(
            (i for i, s in enumerate(call_strs) if '_dismiss_stale_escalations' in s), -1
        )
        start_idx = next(
            (i for i, s in enumerate(call_strs) if '_start_watcher_supervisor' in s), -1
        )

        assert dismiss_idx >= 0, (
            f'_dismiss_stale_escalations not found in mock_calls: {call_strs}'
        )
        assert start_idx >= 0, (
            f'_start_watcher_supervisor not found in mock_calls: {call_strs}'
        )
        assert start_idx > dismiss_idx, (
            f'_start_watcher_supervisor (idx={start_idx}) must appear AFTER '
            f'_dismiss_stale_escalations (idx={dismiss_idx}); calls: {call_strs}'
        )

    @pytest.mark.asyncio
    async def test_stop_watcher_supervisor_awaited_after_stop_terminal(
        self, tmp_path: Path
    ) -> None:
        """Shutdown finally block: _stop_watcher_supervisor is awaited and appears
        AFTER _stop_terminal_status_watcher in the shared parent's mock_calls."""
        h, parent = _make_run_wired_harness(tmp_path)
        with pytest.raises(RuntimeError):
            await h.run(prd_path=None)

        # Must have been awaited (not just called)
        h._stop_watcher_supervisor.assert_awaited()

        call_strs = [str(c) for c in parent.mock_calls]

        stop_terminal_idx = next(
            (i for i, s in enumerate(call_strs) if '_stop_terminal_status_watcher' in s), -1
        )
        stop_watcher_idx = next(
            (i for i, s in enumerate(call_strs) if '_stop_watcher_supervisor' in s), -1
        )

        assert stop_terminal_idx >= 0, (
            f'_stop_terminal_status_watcher not found in mock_calls: {call_strs}'
        )
        assert stop_watcher_idx >= 0, (
            f'_stop_watcher_supervisor not found in mock_calls: {call_strs}'
        )
        assert stop_watcher_idx > stop_terminal_idx, (
            f'_stop_watcher_supervisor (idx={stop_watcher_idx}) must appear after '
            f'_stop_terminal_status_watcher (idx={stop_terminal_idx}); '
            f'calls: {call_strs}'
        )
