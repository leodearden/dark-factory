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
    async def test_clean_exit_no_backoff(self, tmp_path: Path) -> None:
        """Clean exit (success=True, timed_out=False) → no backoff sleep, deque stays empty."""
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
        # No backoff sleep should have been recorded (clean exits → immediate restart)
        assert sleep_durations == [], (
            f'Expected no backoff sleep after clean exit; got {sleep_durations}'
        )
        # Unclean-exit deque must stay empty
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
        """A clean exit after unclean exits resets the backoff to base."""
        from shared.cli_invoke import AgentResult

        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_crashloop_restarts': 99,
            'watcher_crashloop_window_secs': 9999,
        })
        # Sequence: unclean, clean, unclean, then cancel
        results = [
            AgentResult(success=False, output=''),   # unclean → backoff
            AgentResult(success=True, output=''),    # clean   → reset, no backoff
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
        # Should have 2 backoff sleeps (from rotation 1 and rotation 3)
        # No sleep between rotations 2→3 (clean reset → immediate restart)
        assert len(sleep_durations) == 2, (
            f'Expected 2 backoff sleeps (unclean/clean/unclean); got {sleep_durations}'
        )
        # After the clean reset, next unclean should use base again (not doubled)
        assert sleep_durations[1] == pytest.approx(base), (
            f'Post-reset backoff should be base {base}s; got {sleep_durations[1]}'
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
# step-13: Wiring — __init__ attrs + run() source guard
# ---------------------------------------------------------------------------

class TestWatcherSupervisorWiring:
    """Verify __init__ sets the expected attributes and run() wires the lifecycle calls."""

    def test_init_sets_watcher_supervisor_task_none(self, tmp_path: Path) -> None:
        """Harness.__new__ + __init__ sets _watcher_supervisor_task=None."""
        h = _make_lifecycle_harness(tmp_path)
        assert h._watcher_supervisor_task is None

    def test_init_sets_watcher_unclean_exits_empty_deque(self, tmp_path: Path) -> None:
        """Harness.__new__ + __init__ sets _watcher_unclean_exits to an empty deque."""
        h = _make_lifecycle_harness(tmp_path)
        assert isinstance(h._watcher_unclean_exits, deque)
        assert len(h._watcher_unclean_exits) == 0

    def test_start_and_stop_are_bound_methods(self, tmp_path: Path) -> None:
        """_start_watcher_supervisor and _stop_watcher_supervisor are callable."""
        h = _make_lifecycle_harness(tmp_path)
        assert callable(h._start_watcher_supervisor)
        assert callable(h._stop_watcher_supervisor)

    def test_run_calls_start_watcher_supervisor_after_dismiss_stale(self) -> None:
        """Source guard: _start_watcher_supervisor() appears in run() after
        the _dismiss_stale_escalations() call site and alongside the other
        _start_* calls (mirrors terminal_status_watcher, orphan_l0_reaper)."""
        harness_src = Path(__file__).parent.parent / 'src' / 'orchestrator' / 'harness.py'
        source = harness_src.read_text()

        dismiss_pos = source.find('_dismiss_stale_escalations()')
        assert dismiss_pos != -1, '_dismiss_stale_escalations() call not found in harness.py'

        start_pos = source.find('_start_watcher_supervisor()')
        assert start_pos != -1, (
            '_start_watcher_supervisor() call not found in harness.py run() — '
            'step-14 wiring not yet applied'
        )

        # The start call must come AFTER the dismiss call in source order
        assert start_pos > dismiss_pos, (
            '_start_watcher_supervisor() must appear after _dismiss_stale_escalations() '
            f'in run(); dismiss_pos={dismiss_pos} start_pos={start_pos}'
        )

    def test_run_calls_stop_watcher_supervisor_in_finally_block(self) -> None:
        """Source guard: _stop_watcher_supervisor() appears in the finally shutdown
        block alongside _stop_terminal_status_watcher()."""
        harness_src = Path(__file__).parent.parent / 'src' / 'orchestrator' / 'harness.py'
        source = harness_src.read_text()

        stop_terminal_pos = source.find('_stop_terminal_status_watcher()')
        assert stop_terminal_pos != -1, '_stop_terminal_status_watcher() not found'

        stop_supervisor_pos = source.find('_stop_watcher_supervisor()')
        assert stop_supervisor_pos != -1, (
            '_stop_watcher_supervisor() call not found in harness.py run() finally block — '
            'step-14 wiring not yet applied'
        )

        # Both must be within ~100 lines of each other (same shutdown block)
        line_stop_terminal = source[:stop_terminal_pos].count('\n')
        line_stop_supervisor = source[:stop_supervisor_pos].count('\n')
        assert abs(line_stop_terminal - line_stop_supervisor) < 15, (
            f'_stop_watcher_supervisor() (line {line_stop_supervisor}) should be '
            f'near _stop_terminal_status_watcher() (line {line_stop_terminal}) in the '
            f'finally shutdown block'
        )
