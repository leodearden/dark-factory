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
import logging
import time
from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import _init_harness_state_for_test
from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import (
    _WATCHER_ALLOWED_TOOLS,
    _WATCHER_MAX_BACKOFF_SECS,
    _WATCHER_TIMEOUT_GRACE_SECS,
    Harness,
)

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
    _init_harness_state_for_test(h)  # task 1449: initialise digest counters
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
    _init_harness_state_for_test(h)  # task 1449: initialise digest counters
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
        """timeout_seconds == rotation_hours * 3600 + _WATCHER_TIMEOUT_GRACE_SECS exactly.

        Pinned to exact equality so a refactor that drops the grace addition causes this test to fail.
        """
        from shared.cli_invoke import AgentResult

        h = _make_rotation_harness(tmp_path)
        captured: dict = {}

        async def fake_invoke(usage_gate, label, *, invoke_fn, **kwargs):
            captured['timeout_seconds'] = kwargs.get('timeout_seconds')
            return AgentResult(success=True, output='')

        with patch('orchestrator.harness.invoke_with_cap_retry', fake_invoke):
            await h._run_watcher_rotation()

        expected = h.config.watcher_rotation_hours * 3600 + _WATCHER_TIMEOUT_GRACE_SECS
        assert captured['timeout_seconds'] is not None
        assert captured['timeout_seconds'] == expected, (
            f'timeout_seconds must equal rotation_hours*3600 + _WATCHER_TIMEOUT_GRACE_SECS '
            f'({expected}); got {captured["timeout_seconds"]}'
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

    @pytest.mark.asyncio
    async def test_env_overrides_injects_bash_max_timeout_ms(self, tmp_path: Path) -> None:
        """invoke_with_cap_retry receives env_overrides == {'BASH_MAX_TIMEOUT_MS': '<ms>'}.

        Exact-dict equality:
          - pins the injected value against the known literal for the default config
          - also asserts BASH_DEFAULT_TIMEOUT_MS is NOT injected (belt-only per D1)
        Default: rotation_hours=4.0, grace=300.0 → 14700 s → '14700000' ms.
        """
        from shared.cli_invoke import AgentResult

        h = _make_rotation_harness(tmp_path)
        captured: dict = {}

        async def fake_invoke(usage_gate, label, *, invoke_fn, **kwargs):
            captured['env_overrides'] = kwargs.get('env_overrides')
            return AgentResult(success=True, output='')

        with patch('orchestrator.harness.invoke_with_cap_retry', fake_invoke):
            await h._run_watcher_rotation()

        # Assert against the known literal for the default config to catch formula drift.
        # Derivation: rotation_hours=4.0, grace=300.0 → 14700 s → 14700000 ms.
        assert captured.get('env_overrides') == {'BASH_MAX_TIMEOUT_MS': '14700000'}, (
            f'env_overrides must be exactly {{"BASH_MAX_TIMEOUT_MS": "14700000"}}; '
            f'got {captured.get("env_overrides")!r}'
        )

    @pytest.mark.asyncio
    async def test_rotation_start_logs_bash_max_timeout_ms(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An INFO record is emitted on orchestrator.harness during rotation start.

        Proves the operator-visible dispatch log fires; the exact wording is not
        pinned so benign log-message rewording does not break this test.
        """
        from shared.cli_invoke import AgentResult

        h = _make_rotation_harness(tmp_path)

        async def fake_invoke(usage_gate, label, *, invoke_fn, **kwargs):
            return AgentResult(success=True, output='')

        caplog.set_level(logging.INFO, logger='orchestrator.harness')
        with patch('orchestrator.harness.invoke_with_cap_retry', fake_invoke):
            await h._run_watcher_rotation()

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert info_records, (
            f'Expected at least one INFO record on logger orchestrator.harness during rotation; '
            f'got records: {[r.getMessage() for r in caplog.records]}'
        )


# ---------------------------------------------------------------------------
# step-9: Supervisor loop classification — clean/unclean backoff
# ---------------------------------------------------------------------------

def _make_loop_harness(tmp_path: Path) -> Harness:
    """Minimal Harness for _watcher_supervisor_loop tests."""
    from collections import deque

    h = Harness.__new__(Harness)
    _init_harness_state_for_test(h)  # task 1449: initialise digest counters
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


def _build_monotonic_timestamps(
    durations: list[float],
    *,
    expect_cancelled: bool = True,
) -> list[float]:
    """Build paired (start, end) monotonic timestamps for a sequence of rotations.

    Each entry *d* in *durations* contributes ``[t0, t0 + d]`` to the list,
    representing the two ``time.monotonic()`` calls made inside the supervisor
    for each rotation (one before, one after).

    When *expect_cancelled* is ``True`` (the default), one trailing ``t0``
    is appended to cover the start-of-rotation ``time.monotonic()`` call
    that happens before the cancelling (n+1)th ``fake_rotation`` raises
    ``CancelledError``.

    This helper centralises the pairing logic shared by
    ``_run_supervisor_with_rotation_durations`` and any inline test that
    needs deterministic monotonic control (e.g. tests with mixed
    clean/unclean sequences).  Eliminates the risk of off-by-one errors
    when the sequence is hand-built in multiple places.
    """

    t0 = time.monotonic()
    timestamps: list[float] = []
    for dur in durations:
        timestamps.extend([t0, t0 + dur])
    if expect_cancelled:
        timestamps.append(t0)
    return timestamps


async def _run_supervisor_with_rotation_durations(
    h: Harness,
    rotation_durations_secs: list[float],
    *,
    expect_cancelled: bool = True,
) -> list[float]:
    """Drive _watcher_supervisor_loop with controlled per-rotation durations.

    Each entry in *rotation_durations_secs* becomes one paired (start, end)
    timestamp consumed by the patched ``time.monotonic``.

    When *expect_cancelled* is ``True`` (the default), a final start timestamp
    is appended and ``fake_rotation`` raises ``CancelledError`` on the (n+1)th
    call so the loop terminates cleanly via ``pytest.raises``.

    When *expect_cancelled* is ``False`` the loop is expected to return normally
    (e.g. because a guard trip fires on the last provided rotation), so no
    trailing cancellation is injected and ``pytest.raises`` is omitted.  The
    caller is responsible for configuring any trip-triggering state (e.g. setting
    ``h.pause_scheduler``) before calling this helper.

    Returns the list of floor/backoff sleep durations recorded via the patched
    ``asyncio.sleep``.  The caller is responsible for configuring *h* (config
    overrides, etc.) before calling this helper.
    """
    from shared.cli_invoke import AgentResult

    sleep_durations: list[float] = []
    rotation_calls = 0
    n = len(rotation_durations_secs)

    timestamps = _build_monotonic_timestamps(rotation_durations_secs, expect_cancelled=expect_cancelled)
    monotonic_sequence = iter(timestamps)

    def fake_monotonic() -> float:
        return next(monotonic_sequence)

    async def fake_rotation() -> AgentResult:
        nonlocal rotation_calls
        rotation_calls += 1
        if expect_cancelled and rotation_calls > n:
            raise asyncio.CancelledError()
        return AgentResult(success=True, output='', timed_out=False)

    async def recording_sleep(duration: float) -> None:
        sleep_durations.append(duration)

    h._run_watcher_rotation = fake_rotation  # type: ignore[method-assign]

    if expect_cancelled:
        with (
            patch('orchestrator.harness.asyncio.sleep', recording_sleep),
            patch('orchestrator.harness.time.monotonic', side_effect=fake_monotonic),
            pytest.raises(asyncio.CancelledError),
        ):
            await h._watcher_supervisor_loop()
    else:
        with (
            patch('orchestrator.harness.asyncio.sleep', recording_sleep),
            patch('orchestrator.harness.time.monotonic', side_effect=fake_monotonic),
        ):
            await h._watcher_supervisor_loop()

    return sleep_durations


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

        Sequence [unclean, unclean, clean, unclean] produces 4 sleeps:
          sleep[0] = base          (consecutive=1)
          sleep[1] = 2*base        (consecutive=2, escalated)
          sleep[2] = base          (clean floor)
          sleep[3] = base          (consecutive reset → 1 again)

        The reset is OBSERVABLE because sleep[1] is escalated (2*base) and
        sleep[3] drops back to base.  Asserting sleep[3] < sleep[1] proves the
        consecutive counter was zeroed — a broken reset would yield sleep[3]=4*base.
        """
        from shared.cli_invoke import AgentResult

        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_crashloop_restarts': 99,
            'watcher_crashloop_window_secs': 9999,
            'watcher_max_misconfigured_clean_exits': 99,  # disable misconfig trip
        })
        # Sequence: unclean, unclean, clean, unclean, then cancel
        results = [
            AgentResult(success=False, output=''),   # unclean → backoff(base, consecutive=1)
            AgentResult(success=False, output=''),   # unclean → backoff(2*base, consecutive=2)
            AgentResult(success=True, output=''),    # clean   → reset consecutive + floor(base)
            AgentResult(success=False, output=''),   # unclean → base backoff (consecutive reset to 1)
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
        # 4 sleeps: unclean | escalated-unclean | clean-floor | reset-unclean
        assert len(sleep_durations) == 4, (
            f'Expected 4 sleeps (u/u/clean-floor/u); got {sleep_durations}'
        )
        assert sleep_durations[1] == pytest.approx(2 * base), (
            f'Pre-clean backoff should be 2*base ({2*base}s); got {sleep_durations[1]}'
        )
        assert sleep_durations[3] == pytest.approx(base), (
            f'Post-reset backoff should be base ({base}s); got {sleep_durations[3]}'
        )
        # This is the key observable assertion: reset worked iff post-clean < pre-clean.
        assert sleep_durations[3] < sleep_durations[1], (
            f'Post-reset backoff {sleep_durations[3]} must be < pre-clean escalated '
            f'backoff {sleep_durations[1]} (proves consecutive counter was zeroed)'
        )

    @pytest.mark.asyncio
    async def test_rotation_exception_classified_unclean(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """When _run_watcher_rotation raises an exception (not CancelledError),
        the supervisor treats it as an unclean exit: the unclean deque grows and
        backoff sleep is applied.

        Coverage backfill (S4a): verifies the existing
        `except Exception: result = None` sentinel path at harness.py:2753-2758.
        Also asserts logger.exception fires so a silent swallow (bare except: pass)
        would be caught.
        """
        from shared.cli_invoke import AgentResult

        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_crashloop_restarts': 99,    # disable crashloop trip
            'watcher_max_misconfigured_clean_exits': 99,  # disable misconfig trip
            'watcher_crashloop_window_secs': 9999,
            'watcher_subprocess_restart_backoff_secs': 1.0,
        })

        rotation_calls = 0
        sleep_durations: list[float] = []

        async def fake_rotation() -> AgentResult:
            nonlocal rotation_calls
            rotation_calls += 1
            if rotation_calls == 1:
                raise RuntimeError('boom')   # exception path under test
            raise asyncio.CancelledError()   # exit on second call

        async def fake_sleep(duration: float) -> None:
            sleep_durations.append(duration)

        h._run_watcher_rotation = fake_rotation  # type: ignore[method-assign]

        caplog.set_level(logging.ERROR, logger='orchestrator.harness')
        with patch('orchestrator.harness.asyncio.sleep', fake_sleep), pytest.raises(asyncio.CancelledError):
            await h._watcher_supervisor_loop()

        # Raised exception must be classified as unclean: deque grows
        assert len(h._watcher_unclean_exits) == 1, (
            f'RuntimeError from rotation should be classified unclean; '
            f'_watcher_unclean_exits has {len(h._watcher_unclean_exits)} entries'
        )
        # Backoff sleep must be applied (not skipped on the exception path)
        assert len(sleep_durations) >= 1, (
            'Backoff sleep must occur after exception-classified unclean exit'
        )
        # logger.exception must have fired — a silent swallow would not emit this record
        assert any(
            'rotation raised unexpected exception' in r.getMessage()
            for r in caplog.records
        ), (
            'Expected logger.exception with "rotation raised unexpected exception" '
            f'in orchestrator.harness; records: {[r.getMessage() for r in caplog.records]}'
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
        stable_time = time.monotonic()
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
        # Each loop iteration makes 2 monotonic calls (start + end); sequences
        # are [start1, end1, start2, end2, ...] pairs.
        old_time = time.monotonic()
        new_time = old_time + window + 1  # beyond the window
        time_sequence = iter(
            [old_time] * 4              # 2 old iterations × (start, end) each
            + [new_time] * (max_restarts * 2)   # subsequent iterations × 2 calls
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

    @pytest.mark.asyncio
    async def test_pause_scheduler_failure_still_stops_supervisor(
        self, tmp_path: Path
    ) -> None:
        """When pause_scheduler raises on a crashloop trip, the supervisor must
        still stop (not silently continue).

        Regression test (S2/S4b): the broad try/except around the unclean path
        previously swallowed pause_scheduler failures, causing the loop to restart
        the agent indefinitely instead of stopping.
        """

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
            # Guard: cancel after max_restarts * 2 iterations to prevent an infinite
            # loop when the S2 defect is present (makes RED state fail fast).
            if rotation_calls > max_restarts * 2:
                raise asyncio.CancelledError()
            return AgentResult(success=False, output='')

        async def raising_pause_scheduler(reason: str) -> None:
            pause_calls.append(reason)
            raise RuntimeError('pause failed')

        h._run_watcher_rotation = fake_rotation              # type: ignore[method-assign]
        h.pause_scheduler = raising_pause_scheduler          # type: ignore[method-assign]

        stable_time = time.monotonic()
        with (
            patch('orchestrator.harness.asyncio.sleep', AsyncMock()),
            patch('orchestrator.harness.time.monotonic', return_value=stable_time),
        ):
            # Supervisor must return even though pause_scheduler raises.
            # In RED state: CancelledError fires at max_restarts*2+1 rotations, and
            # rotation_calls != max_restarts assertion reveals the defect.
            await h._watcher_supervisor_loop()

        # pause_scheduler was called exactly once with the crashloop reason
        assert pause_calls == ['watcher_crashloop'], (
            f'Expected pause_scheduler("watcher_crashloop") once; got {pause_calls}'
        )
        # No 4th rotation after the failing trip (supervisor stopped)
        assert rotation_calls == max_restarts, (
            f'Expected exactly {max_restarts} rotations; got {rotation_calls} '
            f'(supervisor did not stop after failing pause)'
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

        from shared.cli_invoke import AgentResult

        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_misconfigured_clean_exits': 99,  # disable trip during this test
            'watcher_subprocess_restart_backoff_secs': 0.0,
        })

        # Rotation duration 1.0s — well under default 120s threshold
        t0 = time.monotonic()
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

        from shared.cli_invoke import AgentResult

        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_misconfigured_min_rotation_secs': 120.0,
            'watcher_max_misconfigured_clean_exits': 99,
            'watcher_subprocess_restart_backoff_secs': 0.0,
        })

        # Rotation duration 200s — above the 120s threshold
        t0 = time.monotonic()
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

    @pytest.mark.asyncio
    async def test_misconfigured_trips_pause_scheduler(self, tmp_path: Path) -> None:
        """After watcher_max_misconfigured_clean_exits fast-clean exits in the window,
        pause_scheduler is called once with 'watcher_misconfigured' and the loop exits."""

        from shared.cli_invoke import AgentResult

        max_misconfig = 3
        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_misconfigured_clean_exits': max_misconfig,
            'watcher_crashloop_window_secs': 600,
            'watcher_subprocess_restart_backoff_secs': 0.0,
            'watcher_misconfigured_min_rotation_secs': 120.0,
            'watcher_max_crashloop_restarts': 99,  # disable crashloop trip
        })

        rotation_calls = 0
        pause_calls: list[str] = []

        # All rotations are fast-clean (duration 1s < 120s).
        # Guard: cancel after max_misconfig * 2 rotations to prevent an infinite loop
        # when the threshold trip is not yet implemented (makes RED state fail fast).
        t0 = time.monotonic()

        async def fake_rotation() -> AgentResult:
            nonlocal rotation_calls
            rotation_calls += 1
            if rotation_calls > max_misconfig * 2:
                raise asyncio.CancelledError()
            return AgentResult(success=True, output='', timed_out=False)

        async def fake_pause_scheduler(reason: str) -> None:
            pause_calls.append(reason)

        h._run_watcher_rotation = fake_rotation          # type: ignore[method-assign]
        h.pause_scheduler = fake_pause_scheduler         # type: ignore[method-assign]

        # Each iteration: start=t0, end=t0+1.0 (duration 1s < 120s threshold)
        call_count = 0

        def fake_monotonic() -> float:
            nonlocal call_count
            call_count += 1
            # start call (odd): t0; end call (even): t0 + 1.0
            if call_count % 2 == 1:
                return t0
            return t0 + 1.0

        with (
            patch('orchestrator.harness.asyncio.sleep', AsyncMock()),
            patch('orchestrator.harness.time.monotonic', side_effect=fake_monotonic),
        ):
            # Loop should exit after max_misconfig fast-clean exits (no CancelledError)
            await h._watcher_supervisor_loop()

        assert pause_calls == ['watcher_misconfigured'], (
            f'Expected pause_scheduler("watcher_misconfigured") once; got {pause_calls}'
        )
        assert rotation_calls == max_misconfig, (
            f'Expected exactly {max_misconfig} rotations before trip; got {rotation_calls}'
        )

    @pytest.mark.asyncio
    async def test_old_degenerate_exits_outside_window_are_evicted(self, tmp_path: Path) -> None:
        """Degenerate-clean exits older than watcher_crashloop_window_secs are evicted
        so they do not contribute to the misconfigured count.

        Sequence: 2 old fast-clean exits, time-jump past window, 2 more new fast-clean
        exits (total in-window = 2 < max_misconfig=3) → no trip, loop cancels normally.

        Uses a paired-values iterator (start, end per iteration) so BOTH monotonic
        calls within a single rotation use matched timestamps — ensures iteration 2
        also has a short duration and is classified as degenerate (the previous
        impl used iter_count tracking which made iteration 2's duration ≈ window+2s,
        so only 1 old entry was actually added instead of the intended 2).
        """

        from shared.cli_invoke import AgentResult

        max_misconfig = 3
        window = 600
        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_misconfigured_clean_exits': max_misconfig,
            'watcher_crashloop_window_secs': window,
            'watcher_subprocess_restart_backoff_secs': 0.0,
            'watcher_misconfigured_min_rotation_secs': 120.0,
            'watcher_max_crashloop_restarts': 99,  # disable crashloop trip
        })

        pause_calls: list[str] = []
        rotation_calls = 0

        # Drive 2 fast-clean exits at old_time, then jump past window,
        # then drive 2 more fast-clean exits (< max_misconfig=3), then cancel.
        old_time = time.monotonic()
        new_time = old_time + window + 1  # beyond the window

        # Paired iterator: (start, end) per iteration — both calls within one
        # rotation use matched timestamps so duration is always 1s (< 120s threshold).
        # iter 1: old start, old end
        # iter 2: old start, old end
        # iter 3: new start, new end  (old entries evicted here)
        # iter 4: new start, new end
        # iter 5: new start only (CancelledError raised before end)
        monotonic_sequence = iter(
            [old_time, old_time + 1.0,   # iter 1 — old fast-clean
             old_time, old_time + 1.0,   # iter 2 — old fast-clean
             new_time, new_time + 1.0,   # iter 3 — new fast-clean (old entries evicted)
             new_time, new_time + 1.0,   # iter 4 — new fast-clean
             new_time]                   # iter 5 start (before CancelledError)
        )

        def fake_monotonic() -> float:
            return next(monotonic_sequence)

        # Track deque size at the start of each rotation to verify that 2 old
        # entries were actually in the deque before the time jump.
        deque_snapshots: dict[int, int] = {}

        async def fake_rotation() -> AgentResult:
            nonlocal rotation_calls
            rotation_calls += 1
            # Snapshot is taken BEFORE this iteration's deque update, so
            # snapshot[3] == 2 proves two old entries existed before eviction.
            deque_snapshots[rotation_calls] = len(h._watcher_degenerate_clean_exits)
            if rotation_calls > max_misconfig + 1:
                raise asyncio.CancelledError()
            return AgentResult(success=True, output='', timed_out=False)

        async def fake_pause_scheduler(reason: str) -> None:
            pause_calls.append(reason)

        h._run_watcher_rotation = fake_rotation          # type: ignore[method-assign]
        h.pause_scheduler = fake_pause_scheduler         # type: ignore[method-assign]

        with (
            patch('orchestrator.harness.asyncio.sleep', AsyncMock()),
            patch('orchestrator.harness.time.monotonic', side_effect=fake_monotonic),
            pytest.raises(asyncio.CancelledError),
        ):
            # Loop should cancel (not trip) because old entries are evicted
            await h._watcher_supervisor_loop()

        assert pause_calls == [], (
            'Old degenerate exits outside window should be evicted; '
            'pause_scheduler must NOT trip'
        )
        # Verify that 2 old entries were genuinely added before the time jump
        # (snapshot taken at start of iter 3, before iter 3's deque update).
        assert deque_snapshots.get(3) == 2, (
            f'Expected 2 old entries in deque at start of iteration 3 '
            f'(before time-jump eviction); got {deque_snapshots.get(3)}'
        )
        # After eviction at iter 3 and iter 4: only 2 new entries remain.
        assert len(h._watcher_degenerate_clean_exits) == 2, (
            f'Expected 2 in-window entries after old entries evicted; '
            f'got {len(h._watcher_degenerate_clean_exits)}'
        )
        # All remaining entries must be from new_time (old_time entries evicted).
        assert all(t >= new_time for t in h._watcher_degenerate_clean_exits), (
            'All remaining deque entries must use new_time (old entries evicted)'
        )

    @pytest.mark.asyncio
    async def test_misconfigured_pause_scheduler_failure_still_stops_supervisor(
        self, tmp_path: Path
    ) -> None:
        """When pause_scheduler raises on a misconfigured trip, the supervisor must
        still stop (not silently continue).

        Regression test (symmetry with S2/crashloop): the defensive try/except
        added in step-6 around pause_scheduler('watcher_misconfigured') must
        return regardless of whether pause_scheduler itself raises.
        """

        from shared.cli_invoke import AgentResult

        max_misconfig = 3
        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_misconfigured_clean_exits': max_misconfig,
            'watcher_crashloop_window_secs': 600,
            'watcher_subprocess_restart_backoff_secs': 0.0,
            'watcher_misconfigured_min_rotation_secs': 120.0,
            'watcher_max_crashloop_restarts': 99,  # disable crashloop trip
        })

        rotation_calls = 0
        pause_calls: list[str] = []

        # Guard: cancel after max_misconfig * 2 iterations to prevent an infinite
        # loop when the defensive wrapper is absent (makes failure fast).
        async def fake_rotation() -> AgentResult:
            nonlocal rotation_calls
            rotation_calls += 1
            if rotation_calls > max_misconfig * 2:
                raise asyncio.CancelledError()
            return AgentResult(success=True, output='', timed_out=False)

        async def raising_pause_scheduler(reason: str) -> None:
            pause_calls.append(reason)
            raise RuntimeError('pause failed')

        h._run_watcher_rotation = fake_rotation              # type: ignore[method-assign]
        h.pause_scheduler = raising_pause_scheduler          # type: ignore[method-assign]

        t0 = time.monotonic()
        call_count = 0

        def fake_monotonic() -> float:
            nonlocal call_count
            call_count += 1
            # odd call = start (t0), even call = end (t0 + 1.0s < 120s threshold)
            return t0 if call_count % 2 == 1 else t0 + 1.0

        with (
            patch('orchestrator.harness.asyncio.sleep', AsyncMock()),
            patch('orchestrator.harness.time.monotonic', side_effect=fake_monotonic),
        ):
            # Supervisor must return even though pause_scheduler raises.
            await h._watcher_supervisor_loop()

        assert pause_calls == ['watcher_misconfigured'], (
            f'Expected pause_scheduler("watcher_misconfigured") once; got {pause_calls}'
        )
        assert rotation_calls == max_misconfig, (
            f'Expected exactly {max_misconfig} rotations; got {rotation_calls} '
            f'(supervisor did not stop after failing pause)'
        )

    @pytest.mark.asyncio
    async def test_degenerate_clean_floor_grows_exponentially(self, tmp_path: Path) -> None:
        """Consecutive degenerate-clean exits grow the restart floor exponentially.

        Each fast-clean rotation (duration < watcher_misconfigured_min_rotation_secs)
        that does NOT trip the guard should sleep base * 2^(consecutive-1) — mirroring
        the unclean-exit exponential pattern.  With watcher_subprocess_restart_backoff_secs=1.0
        and 4 consecutive degenerate exits, expected floor durations are [1.0, 2.0, 4.0, 8.0].

        RED: currently the production code sleeps a flat watcher_subprocess_restart_backoff_secs
        on every clean rotation, so all 4 sleeps would be 1.0 instead of growing.
        """
        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_misconfigured_clean_exits': 99,   # disable trip
            'watcher_misconfigured_min_rotation_secs': 120.0,
            'watcher_subprocess_restart_backoff_secs': 1.0,
            'watcher_crashloop_window_secs': 600,
            'watcher_max_crashloop_restarts': 99,          # disable crashloop trip
        })
        # 4 consecutive degenerate-clean rotations (1s each, well under 120s threshold)
        sleep_durations = await _run_supervisor_with_rotation_durations(h, [1.0, 1.0, 1.0, 1.0])
        assert sleep_durations == pytest.approx([1.0, 2.0, 4.0, 8.0]), (
            f'Expected exponential floor [1.0, 2.0, 4.0, 8.0] for 4 consecutive '
            f'degenerate-clean exits; got {sleep_durations}'
        )

    @pytest.mark.asyncio
    async def test_healthy_clean_resets_degenerate_floor(self, tmp_path: Path) -> None:
        """A healthy-duration clean rotation resets the degenerate-clean floor counter.

        Sequence: degenerate (1s), degenerate (1s), HEALTHY (200s), degenerate (1s).
        With watcher_subprocess_restart_backoff_secs=1.0 and min_rotation_secs=120.0:
          sleep[0] = 1.0  (consecutive_degenerate=1)
          sleep[1] = 2.0  (consecutive_degenerate=2, escalated)
          sleep[2] = 1.0  (healthy flat base — queue had real work)
          sleep[3] = 1.0  (key assertion: reset to consecutive_degenerate=1, NOT 3)

        Without the reset, sleep[3] would be 4.0 (counter continuing from 3).

        RED: after step-2 the counter is never zeroed on a healthy rotation, so
        sleep[3] == 4.0 instead of 1.0.
        """
        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_misconfigured_clean_exits': 99,   # disable trip
            'watcher_misconfigured_min_rotation_secs': 120.0,
            'watcher_subprocess_restart_backoff_secs': 1.0,
            'watcher_crashloop_window_secs': 600,
            'watcher_max_crashloop_restarts': 99,          # disable crashloop trip
        })
        # degen (1s), degen (1s), HEALTHY (200s), degen (1s)
        # key: the post-healthy degenerate must sleep 1.0 (counter reset), not 4.0
        sleep_durations = await _run_supervisor_with_rotation_durations(
            h, [1.0, 1.0, 200.0, 1.0]
        )
        assert sleep_durations == pytest.approx([1.0, 2.0, 1.0, 1.0]), (
            f'Expected [1.0, 2.0, 1.0, 1.0] — healthy rotation at index 2 must reset '
            f'the degenerate counter so index 3 is base (not 4.0); got {sleep_durations}'
        )

    @pytest.mark.asyncio
    async def test_degenerate_clean_floor_saturates_at_cap(self, tmp_path: Path) -> None:
        """The exponential degenerate-clean floor is capped at _WATCHER_MAX_BACKOFF_SECS.

        Guards the ``min(..., _WATCHER_MAX_BACKOFF_SECS)`` saturation cap in
        ``_watcher_supervisor_loop`` (task 1430, degenerate-clean branch) against
        a regression that drops it.  With base=10.0 and 10
        consecutive degenerate-clean exits the uncapped sequence would be:
          [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
        The 10th value (5120.0) exceeds _WATCHER_MAX_BACKOFF_SECS=3600.0 by ~42 %,
        so a missing cap is unambiguous — the test would observe 5120 instead of 3600.
        Earlier values (all ≤ 2560) fall below the cap, confirming the cap only
        triggers when it should.
        """
        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_misconfigured_clean_exits': 99,   # disable trip
            'watcher_misconfigured_min_rotation_secs': 120.0,
            'watcher_subprocess_restart_backoff_secs': 10.0,
            'watcher_crashloop_window_secs': 600,
            'watcher_max_crashloop_restarts': 99,          # disable crashloop trip
        })
        # 10 consecutive degenerate-clean rotations (1s each, well under 120s threshold)
        sleep_durations = await _run_supervisor_with_rotation_durations(h, [1.0] * 10)
        assert len(sleep_durations) == 10, (
            f'Expected 10 sleep durations (one per rotation); got {len(sleep_durations)}'
        )
        assert sleep_durations == pytest.approx([
            10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0,
            _WATCHER_MAX_BACKOFF_SECS,
        ]), (
            f'Expected exponential sequence capped at {_WATCHER_MAX_BACKOFF_SECS} on '
            f'the 10th rotation (uncapped value would be 5120.0); got {sleep_durations}'
        )
        # Extra cross-check: every sleep must respect the cap
        assert all(s <= _WATCHER_MAX_BACKOFF_SECS for s in sleep_durations), (
            f'One or more sleeps exceeded _WATCHER_MAX_BACKOFF_SECS={_WATCHER_MAX_BACKOFF_SECS}: '
            f'{sleep_durations}'
        )

    @pytest.mark.asyncio
    async def test_degenerate_clean_trip_fires_during_exponential_growth(
        self, tmp_path: Path
    ) -> None:
        """Trip fires correctly while the exponential floor is actively growing.

        Task 1430 moved ``consecutive_degenerate_clean += 1`` *before* the
        ``_check_watcher_guard`` call in ``_watcher_supervisor_loop`` so the
        floor calculation always uses the post-increment counter.  This test
        verifies that ordering:

        - Iters 1 and 2: no trip (deque length < max_misconfig=3); floor is
          base * 2^(n-1) using the post-increment counter → sleeps [1.0, 2.0].
        - Iter 3: trip arms (deque length == max_misconfig=3) → pause_scheduler
          called, loop returns; no sleep issued for iter 3.

        Complements the existing ``test_misconfigured_trips_pause_scheduler`` which
        uses base=0.0 (floor always 0, increment ordering invisible).  With base=1.0
        a regression moving the increment to *after* the guard call would yield
        floors [base, base] = [1.0, 1.0] instead of [1.0, 2.0], catching the ordering
        bug while still firing the trip on iter 3.
        """
        max_misconfig = 3
        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_misconfigured_clean_exits': max_misconfig,
            'watcher_misconfigured_min_rotation_secs': 120.0,
            'watcher_subprocess_restart_backoff_secs': 1.0,  # NON-zero: floor is observable
            'watcher_crashloop_window_secs': 600,
            'watcher_max_crashloop_restarts': 99,  # disable crashloop trip
        })

        pause_calls: list[str] = []

        async def fake_pause_scheduler(reason: str) -> None:
            pause_calls.append(reason)

        h.pause_scheduler = fake_pause_scheduler  # type: ignore[method-assign]

        # 3 rotations of 1s each (all < 120s threshold → degenerate-clean).
        # On iter 3 the trip arms and the supervisor returns normally — not via CancelledError.
        sleep_durations = await _run_supervisor_with_rotation_durations(
            h, [1.0, 1.0, 1.0], expect_cancelled=False
        )

        assert pause_calls == ['watcher_misconfigured'], (
            f'Expected pause_scheduler("watcher_misconfigured") once; got {pause_calls}'
        )
        # Key assertion: floor sequence proves pre-guard increment ordering.
        # Iter 1: consecutive=1 → floor = 1.0 * 2^0 = 1.0
        # Iter 2: consecutive=2 → floor = 1.0 * 2^1 = 2.0
        # Iter 3: trip fires → no sleep (returns before reaching floor calculation)
        # len(sleep_durations)==2 also confirms exactly 3 rotations ran (2 slept + 1 tripped).
        assert sleep_durations == pytest.approx([1.0, 2.0]), (
            f'Expected sleep_durations [1.0, 2.0] (pre-guard increment gives base*2^0 '
            f'then base*2^1 before trip arms on iter 3); got {sleep_durations}'
        )

    @pytest.mark.asyncio
    async def test_degenerate_clean_floor_does_not_overflow_at_high_consecutive(
        self, tmp_path: Path
    ) -> None:
        """Degenerate-clean floor computation does not raise OverflowError at high consecutive.

        With an unbounded exponent, ``2 ** (consecutive - 1)`` overflows to
        OverflowError at consecutive > ~1024 (Python cannot represent 2^1024 as float).
        The fix clamps the exponent to 60 before multiplying, so the outer
        ``min(..., _WATCHER_MAX_BACKOFF_SECS)`` still applies and the sleep saturates
        at the ceiling without ever raising.

        RED: without the exponent clamp the loop crashes with OverflowError on the
        ~1025th rotation, so the assert on len(sleep_durations) would never be
        reached — the test fails with OverflowError propagating from
        _watcher_supervisor_loop.
        """
        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_misconfigured_clean_exits': 99999,   # disable trip
            'watcher_max_crashloop_restarts': 99999,           # disable crashloop trip
            'watcher_misconfigured_min_rotation_secs': 120.0,
            'watcher_subprocess_restart_backoff_secs': 1.0,
            'watcher_crashloop_window_secs': 999999,           # deque never evicts, trip never fires
        })
        # 1100 consecutive degenerate-clean rotations (1s each, well under 120s threshold).
        # Beyond consecutive ~1024 the unguarded 2^(consecutive-1) overflows float range.
        sleep_durations = await _run_supervisor_with_rotation_durations(h, [1.0] * 1100)
        assert len(sleep_durations) == 1100, (
            f'Expected 1100 sleep durations (one per rotation); got {len(sleep_durations)}'
        )
        assert all(s <= _WATCHER_MAX_BACKOFF_SECS for s in sleep_durations), (
            f'One or more sleeps exceeded _WATCHER_MAX_BACKOFF_SECS={_WATCHER_MAX_BACKOFF_SECS}: '
            f'{[s for s in sleep_durations if s > _WATCHER_MAX_BACKOFF_SECS]}'
        )
        # Tail values must saturate at the ceiling (not silently drop to 0 or raise)
        assert sleep_durations[-1] == pytest.approx(_WATCHER_MAX_BACKOFF_SECS), (
            f'Final sleep should saturate at {_WATCHER_MAX_BACKOFF_SECS}s; got {sleep_durations[-1]}'
        )

    @pytest.mark.asyncio
    async def test_unclean_backoff_does_not_overflow_at_high_consecutive(
        self, tmp_path: Path
    ) -> None:
        """Unclean-exit backoff does not raise OverflowError at high consecutive.

        Mirrors test_degenerate_clean_floor_does_not_overflow_at_high_consecutive
        for the unclean-exit exponential backoff site (same overflow bug, different
        code path — uses consecutive_unclean instead of consecutive_degenerate_clean).
        No time.monotonic patch needed: the unclean path is duration-independent.

        RED: without the exponent clamp the unclean-path OverflowError is caught by
        the broad ``except Exception`` handler (harness.py) and the loop degrades
        gracefully to the base-backoff fallback (~1.0 s per iteration) rather than
        crashing.  The discriminating assertion is ``sleep_durations[-1]``: with the
        clamp it saturates at ``_WATCHER_MAX_BACKOFF_SECS`` (3600.0 s); without it
        the final sleep is the degraded ~1.0 s fallback.  (Contrast the
        degenerate-clean path, whose overflow computation is outside its try block,
        so that test's crash-based RED claim is accurate.)
        """
        from shared.cli_invoke import AgentResult

        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_crashloop_restarts': 99999,   # disable crashloop trip
            'watcher_crashloop_window_secs': 999999,   # deque never evicts, trip never fires
            'watcher_subprocess_restart_backoff_secs': 1.0,
        })
        rotation_calls = 0
        sleep_durations: list[float] = []

        async def fake_rotation() -> AgentResult:
            nonlocal rotation_calls
            rotation_calls += 1
            if rotation_calls > 1100:
                raise asyncio.CancelledError()
            return AgentResult(success=False, output='')

        async def fake_sleep(duration: float) -> None:
            sleep_durations.append(duration)

        h._run_watcher_rotation = fake_rotation  # type: ignore[method-assign]

        with patch('orchestrator.harness.asyncio.sleep', fake_sleep), pytest.raises(asyncio.CancelledError):
            await h._watcher_supervisor_loop()

        assert len(sleep_durations) >= 1100, (
            f'Expected >= 1100 sleep durations; got {len(sleep_durations)}'
        )
        assert all(s <= _WATCHER_MAX_BACKOFF_SECS for s in sleep_durations), (
            f'One or more sleeps exceeded _WATCHER_MAX_BACKOFF_SECS={_WATCHER_MAX_BACKOFF_SECS}: '
            f'{[s for s in sleep_durations if s > _WATCHER_MAX_BACKOFF_SECS]}'
        )
        # Tail values must saturate at the ceiling (not silently drop to 0 or raise).
        # Check the last element as the primary discriminating signal: without the
        # clamp the OverflowError is swallowed by except Exception and the fallback
        # sleep(base≈1.0) runs instead of sleep(3600.0).
        assert sleep_durations[-1] == pytest.approx(_WATCHER_MAX_BACKOFF_SECS), (
            f'Final sleep should saturate at {_WATCHER_MAX_BACKOFF_SECS}s; got {sleep_durations[-1]}'
        )
        # Bulk check: the post-1024 tail (where overflow would occur without fix) must
        # all saturate, not all degrade to the fallback base rate.
        post_overflow_tail = sleep_durations[1024:]
        assert post_overflow_tail, 'Expected at least one sleep beyond iteration 1024'
        assert all(s == pytest.approx(_WATCHER_MAX_BACKOFF_SECS) for s in post_overflow_tail), (
            f'Post-overflow tail sleeps should all be {_WATCHER_MAX_BACKOFF_SECS}s; '
            f'first 5 tail values: {post_overflow_tail[:5]}'
        )

    @pytest.mark.asyncio
    async def test_unclean_exit_resets_degenerate_clean_floor(self, tmp_path: Path) -> None:
        """An unclean exit resets the degenerate-clean floor counter (symmetric reset).

        The clean path already resets ``consecutive_unclean = 0`` on both healthy AND
        degenerate-clean exits.  By symmetry, the unclean path should reset
        ``consecutive_degenerate_clean = 0`` so that degenerate-clean bursts are
        isolated — an interleaved unclean exit breaks any in-progress degen burst.

        Sequence (4 rotations then cancel):
          [degen-clean(1s), degen-clean(1s), unclean, degen-clean(1s)]

        With watcher_subprocess_restart_backoff_secs=1.0:
          sleep[0] = 1.0  (consecutive_degenerate=1 → base*2^0)
          sleep[1] = 2.0  (consecutive_degenerate=2 → base*2^1)
          sleep[2] = 1.0  (unclean, consecutive_unclean=1 → base*2^0)
          sleep[3] = 1.0  (KEY: post-unclean degen resets to consecutive_degenerate=1)

        RED: without the reset, sleep[3] == 4.0 (consecutive_degenerate=3 → base*2^2),
        because the unclean exit does not zero consecutive_degenerate_clean.
        """
        from shared.cli_invoke import AgentResult

        h = _make_loop_harness(tmp_path)
        h.config = h.config.model_copy(update={
            'watcher_max_misconfigured_clean_exits': 99,   # disable misconfig trip
            'watcher_max_crashloop_restarts': 99,           # disable crashloop trip
            'watcher_misconfigured_min_rotation_secs': 120.0,
            'watcher_subprocess_restart_backoff_secs': 1.0,
            'watcher_crashloop_window_secs': 600,
        })

        # 4 rotations: degen-clean, degen-clean, unclean, degen-clean, then cancel
        results = [
            AgentResult(success=True, output=''),    # degen-clean → floor(1.0)
            AgentResult(success=True, output=''),    # degen-clean → floor(2.0)
            AgentResult(success=False, output=''),   # unclean     → backoff(1.0) + reset degen counter
            AgentResult(success=True, output=''),    # degen-clean → floor(1.0) — KEY: reset worked
        ]
        idx = 0
        sleep_durations: list[float] = []

        # Build paired (start, end) timestamps for all 4 rotations + 1 cancel start
        # via the shared helper so the pairing logic isn't duplicated.
        # Each rotation uses duration 1.0 s (well under 120 s threshold → degenerate).
        # The unclean rotation still consumes a start/end pair (classification is by
        # result, not duration), so duration 1.0 is harmless for that slot.
        timestamps = _build_monotonic_timestamps([1.0] * 4, expect_cancelled=True)

        monotonic_iter = iter(timestamps)

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

        with (
            patch('orchestrator.harness.asyncio.sleep', fake_sleep),
            patch('orchestrator.harness.time.monotonic', side_effect=lambda: next(monotonic_iter)),
            pytest.raises(asyncio.CancelledError),
        ):
            await h._watcher_supervisor_loop()

        base = h.config.watcher_subprocess_restart_backoff_secs
        assert len(sleep_durations) == 4, (
            f'Expected 4 sleeps (degen/degen/unclean/degen); got {sleep_durations}'
        )
        # Verify the first two degenerate-clean floors to confirm exponential growth is working
        assert sleep_durations[0] == pytest.approx(base), (
            f'sleep[0] should be base={base}s (consecutive_degenerate=1); got {sleep_durations[0]}'
        )
        assert sleep_durations[1] == pytest.approx(2 * base), (
            f'sleep[1] should be 2*base={2*base}s (consecutive_degenerate=2); got {sleep_durations[1]}'
        )
        # Verify the unclean backoff
        assert sleep_durations[2] == pytest.approx(base), (
            f'sleep[2] should be base={base}s (unclean, consecutive_unclean=1); got {sleep_durations[2]}'
        )
        # KEY: the post-unclean degen floor must reset to base (not continue to 4.0)
        assert sleep_durations[3] == pytest.approx(base), (
            f'sleep[3] should be base={base}s — unclean exit must reset consecutive_degenerate_clean '
            f'so the next degenerate burst starts fresh; got {sleep_durations[3]} '
            f'(without fix: 4.0 = base*2^2, counter not reset)'
        )
        # The observable symmetric-reset assertion: post-unclean degen < pre-unclean escalated
        assert sleep_durations[3] < sleep_durations[1], (
            f'Post-unclean degenerate floor {sleep_durations[3]} must be < pre-unclean escalated '
            f'{sleep_durations[1]} — proves consecutive_degenerate_clean was zeroed by unclean exit'
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


# ---------------------------------------------------------------------------
# task 1449: regression guard — _maybe_write_digest must surface AttributeError
# ---------------------------------------------------------------------------

class TestMaybeWriteDigestSurfacesMissingState:
    """Regression guard: both _maybe_write_digest and the supervisor wrapper
    must re-raise AttributeError instead of swallowing it.

    Task 1449 narrowed the catch-all ``except Exception`` at two layers:

    * ``_maybe_write_digest``'s own catch-all (harness.py:3390-3393): now
      re-raises ``AttributeError`` so programming-error drift surfaces to tests.
    * The supervisor wrapper (harness.py:2762-2769): applies the same pattern
      so the re-raised error is not re-swallowed at the outer layer.

    Without the narrowing, both layers silently convert the ``AttributeError``
    raised by the missing ``_escalation_event_count`` access to a
    ``logger.warning(...)`` call — tests pass, the bug goes undetected.

    After the narrowing (step-4), both tests below pass: the ``AttributeError``
    propagates to the test runner unchanged.
    """

    @pytest.mark.asyncio
    async def test_maybe_write_digest_surfaces_attribute_error(
        self, tmp_path: Path,
    ) -> None:
        """_maybe_write_digest raises AttributeError when digest counters are absent.

        This test FAILS before step-4 because _maybe_write_digest's catch-all
        (harness.py:3390-3393) swallows the AttributeError and converts it to
        logger.warning(..., exc_info=True).  After step-4 inserts
        ``except AttributeError: raise``, the error propagates.
        """
        h = Harness.__new__(Harness)
        # Provide only config — deliberately omit _init_harness_state_for_test
        # so the four digest counters are absent.
        h.config = OrchestratorConfig(project_root=tmp_path)

        with pytest.raises(AttributeError, match='_escalation_event_count'):
            await h._maybe_write_digest()

    @pytest.mark.asyncio
    async def test_supervisor_loop_wrapper_surfaces_attribute_error(
        self, tmp_path: Path,
    ) -> None:
        """_watcher_supervisor_loop's wrapper re-raises AttributeError from _maybe_write_digest.

        This test FAILS before step-4 because the supervisor wrapper at
        harness.py:2762-2769 swallows the AttributeError (even after step-4
        narrows _maybe_write_digest's own catch-all — two layers must both be
        narrowed).

        Strategy: patch _run_watcher_rotation to return a clean AgentResult
        on the first call, then raise CancelledError on the second call.
        Before step-4: AttributeError is swallowed; the loop continues through
        the degenerate-clean path, sleeps (patched to be instant), and
        terminates with CancelledError — pytest.raises(AttributeError) FAILS.
        After step-4: AttributeError propagates immediately from the first
        iteration's _maybe_write_digest call — pytest.raises(AttributeError)
        PASSES.
        """
        from collections import deque

        from shared.cli_invoke import AgentResult

        # Build same shape as _make_loop_harness but WITHOUT the digest counters.
        h = Harness.__new__(Harness)
        # NOTE: intentionally no _init_harness_state_for_test(h) call here.
        h.config = OrchestratorConfig(project_root=tmp_path)
        h._watcher_supervisor_task = None
        h._watcher_unclean_exits = deque()
        h._watcher_degenerate_clean_exits = deque()

        rotation_calls = 0

        async def fake_rotation() -> AgentResult:
            nonlocal rotation_calls
            rotation_calls += 1
            if rotation_calls > 1:
                raise asyncio.CancelledError()
            return AgentResult(success=True, output='', timed_out=False)

        h._run_watcher_rotation = fake_rotation  # type: ignore[method-assign]

        # time.monotonic: return 0.0 for all calls (start=end=0.0, duration=0).
        # asyncio.sleep: instant no-op so the degenerate-clean backoff floor
        # doesn't actually sleep.
        with (
            patch('orchestrator.harness.time.monotonic', return_value=0.0),
            patch('orchestrator.harness.asyncio.sleep', new=AsyncMock()),
            pytest.raises(AttributeError, match='_escalation_event_count'),
        ):
            await h._watcher_supervisor_loop()

    @pytest.mark.asyncio
    async def test_supervisor_loop_wrapper_surfaces_non_counter_attribute_error(
        self, tmp_path: Path,
    ) -> None:
        """Supervisor wrapper re-raises ANY AttributeError, not just _escalation_event_count.

        Documents the full blast radius of the narrowed catch-all (task 1449):
        ``except AttributeError: raise`` in the supervisor wrapper fires for
        any ``AttributeError`` raised inside ``_maybe_write_digest``, not only
        the missing digest-counter case.  This is intentional — all
        programming-error ``AttributeError``\\s (missing state due to any
        ``__init__``-gap, renamed attribute, deleted dependency) surface to
        the test suite rather than being silently converted to a warning log.

        A future refactor that removes or renames an attribute accessed later
        in ``_maybe_write_digest`` (e.g. ``self.scheduler``) will be caught
        here: the test asserts the error propagates with the correct message,
        not a misleading green result.
        """
        from shared.cli_invoke import AgentResult

        h = Harness.__new__(Harness)
        h.config = OrchestratorConfig(project_root=tmp_path)
        h._watcher_supervisor_task = None
        h._watcher_unclean_exits = deque()
        h._watcher_degenerate_clean_exits = deque()

        async def fake_rotation() -> AgentResult:
            return AgentResult(success=True, output='', timed_out=False)

        h._run_watcher_rotation = fake_rotation  # type: ignore[method-assign]

        # Simulate a non-counter AttributeError inside _maybe_write_digest —
        # e.g. a future refactor removes self.scheduler or renames it.
        # Patching _maybe_write_digest directly isolates the supervisor-wrapper
        # layer: we confirm the wrapper's ``except AttributeError: raise``
        # propagates the error regardless of its origin.
        digest_mock = AsyncMock(side_effect=AttributeError('scheduler'))
        with (
            patch('orchestrator.harness.time.monotonic', return_value=0.0),
            patch('orchestrator.harness.asyncio.sleep', new=AsyncMock()),
            patch.object(h, '_maybe_write_digest', digest_mock),
            pytest.raises(AttributeError, match='scheduler'),
        ):
            await h._watcher_supervisor_loop()


# ---------------------------------------------------------------------------
# task 1501: supervisor-failsafe — file/dedup/resolve L2 outage escalation
# ---------------------------------------------------------------------------

def _make_harness_with_queue(tmp_path: Path) -> tuple[Harness, EscalationQueue]:
    """Build a Harness with a real EscalationQueue for outage-L2 tests.

    Uses the same __new__ + _init_harness_state_for_test pattern as
    _make_loop_harness / _make_lifecycle_harness, but also attaches a real
    EscalationQueue rooted at tmp_path / 'esc' so submit / find_pending_l2_by_root_cause
    / resolve execute the real disk path without mocking queue internals.
    """
    h = Harness.__new__(Harness)
    _init_harness_state_for_test(h)
    config = OrchestratorConfig(project_root=tmp_path)
    h.config = config
    h._watcher_supervisor_task = None
    h._watcher_unclean_exits = deque()
    h._watcher_degenerate_clean_exits = deque()

    queue = EscalationQueue(tmp_path / 'esc')
    h._escalation_queue = queue
    return h, queue


def _submit_sample_l1(queue: EscalationQueue, task_id: str = 'task-sample-1') -> str:
    """Submit a minimal L1 escalation and return its id."""
    from escalation.models import Escalation
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role='test-agent',
        severity='blocking',
        category='infra_issue',
        summary='sample L1 for outage test',
        level=1,
    )
    queue.submit(esc)
    return esc.id


class TestFileWatcherOutageL2:
    """_file_watcher_outage_l2: creates an L2 outage escalation."""

    def test_creates_new_l2_when_none_exists(self, tmp_path: Path) -> None:
        """First call files exactly one L2 with the expected attributes.

        Pre-submit 2 sample L1 escalations so the pending-L1 count is non-zero,
        then verify the filed L2 includes the reason string and the L1 count.
        """
        h, queue = _make_harness_with_queue(tmp_path)

        # Pre-populate queue with 2 L1 escalations so count is non-zero
        _submit_sample_l1(queue, 'task-a')
        _submit_sample_l1(queue, 'task-b')

        h._file_watcher_outage_l2('watcher_crashloop')

        pending = queue.get_pending()
        l2s = [e for e in pending if e.level == 2]
        assert len(l2s) == 1, (
            f'Expected exactly 1 L2 escalation after first call; got {len(l2s)}: {l2s}'
        )
        l2 = l2s[0]
        assert l2.root_cause == h._WATCHER_OUTAGE_ROOT_CAUSE, (
            f'root_cause must be {h._WATCHER_OUTAGE_ROOT_CAUSE!r}; got {l2.root_cause!r}'
        )
        assert l2.task_id == h._WATCHER_OUTAGE_SENTINEL, (
            f'task_id must be {h._WATCHER_OUTAGE_SENTINEL!r}; got {l2.task_id!r}'
        )
        assert l2.severity == 'urgent', (
            f'severity must be "urgent"; got {l2.severity!r}'
        )
        assert l2.agent_role == 'orchestrator-watcher-supervisor', (
            f'agent_role must be "orchestrator-watcher-supervisor"; got {l2.agent_role!r}'
        )
        assert 'watcher_crashloop' in l2.summary, (
            f'summary must contain the reason "watcher_crashloop"; got {l2.summary!r}'
        )
        # L1 count: 2 pre-submitted L1s
        assert '2' in l2.summary, (
            f'summary must mention pending-L1 count (2); got {l2.summary!r}'
        )

    def test_idempotent_no_duplicate_on_second_call(self, tmp_path: Path) -> None:
        """Multiple calls while an open L2 exists must NOT create duplicate L2s.

        Calls _file_watcher_outage_l2 three times (same reason twice, then a
        different reason) and asserts exactly ONE L2 is still pending, proving
        the dedup guard fires on the 2nd and 3rd calls.
        """
        h, queue = _make_harness_with_queue(tmp_path)

        h._file_watcher_outage_l2('watcher_crashloop')
        h._file_watcher_outage_l2('watcher_crashloop')          # same reason — dedup
        h._file_watcher_outage_l2('watcher_misconfigured')      # different reason — still dedup

        l2s = [e for e in queue.get_pending() if e.level == 2]
        assert len(l2s) == 1, (
            f'Expected exactly 1 L2 after 3 calls; got {len(l2s)}: {l2s}'
        )

    def test_noop_when_queue_is_none(self, tmp_path: Path) -> None:
        """When _escalation_queue is None, _file_watcher_outage_l2 must be a no-op.

        Bare-Harness unit tests (no queue attached) must stay green.
        """
        h = Harness.__new__(Harness)
        _init_harness_state_for_test(h)
        h.config = OrchestratorConfig(project_root=tmp_path)
        h._watcher_supervisor_task = None
        h._watcher_unclean_exits = deque()
        h._escalation_queue = None  # bare-Harness

        # Must not raise
        h._file_watcher_outage_l2('watcher_crashloop')


# ---------------------------------------------------------------------------
# task 1501: _resolve_watcher_outage_l2 resolves the open outage L2
# ---------------------------------------------------------------------------

class TestResolveWatcherOutageL2:
    """_resolve_watcher_outage_l2: resolves an open outage L2 or is a no-op."""

    def test_resolves_open_l2_when_present(self, tmp_path: Path) -> None:
        """File an outage L2, then resolve it — must no longer be pending."""
        h, queue = _make_harness_with_queue(tmp_path)

        # File the outage L2
        h._file_watcher_outage_l2('watcher_crashloop')
        l2s_before = [e for e in queue.get_pending() if e.level == 2]
        assert len(l2s_before) == 1, 'precondition: one pending L2'
        l2_id = l2s_before[0].id

        # Resolve
        h._resolve_watcher_outage_l2()

        # Must no longer be in pending
        l2s_after = [e for e in queue.get_pending() if e.level == 2]
        assert len(l2s_after) == 0, (
            f'Expected 0 pending L2s after resolve; got {l2s_after}'
        )
        # Must be resolved in the full queue
        resolved = queue.get(l2_id)
        assert resolved is not None, f'escalation {l2_id} not found in queue'
        assert resolved.status == 'resolved', (
            f'Expected status="resolved"; got {resolved.status!r}'
        )

    def test_noop_when_no_open_l2(self, tmp_path: Path) -> None:
        """When no pending L2 with root_cause='watcher_supervisor_down', must be a no-op."""
        h, queue = _make_harness_with_queue(tmp_path)

        # No L2 in queue
        pending_before = list(queue.get_pending())

        # Must not raise
        h._resolve_watcher_outage_l2()

        # Queue unchanged
        pending_after = list(queue.get_pending())
        assert pending_after == pending_before, (
            f'Queue should be unchanged; before={pending_before} after={pending_after}'
        )

    def test_noop_when_queue_is_none(self, tmp_path: Path) -> None:
        """When _escalation_queue is None, _resolve_watcher_outage_l2 must be a no-op."""
        h = Harness.__new__(Harness)
        _init_harness_state_for_test(h)
        h.config = OrchestratorConfig(project_root=tmp_path)
        h._escalation_queue = None

        # Must not raise
        h._resolve_watcher_outage_l2()


# ---------------------------------------------------------------------------
# task 1501: _check_watcher_guard trip must also file the outage L2
# ---------------------------------------------------------------------------

class TestCheckWatcherGuardFilesL2Outage:
    """When _check_watcher_guard trips, _file_watcher_outage_l2 is also called."""

    @pytest.mark.asyncio
    async def test_trip_files_outage_l2_crashloop(self, tmp_path: Path) -> None:
        """Crashloop trip: L2 outage escalation is filed with reason='watcher_crashloop'."""
        max_count = 3
        window = 600.0
        now = time.monotonic()

        h, queue = _make_harness_with_queue(tmp_path)
        h.pause_scheduler = AsyncMock()  # type: ignore[method-assign]

        # Pre-populate exits_deque with max_count - 1 entries so the trip fires
        # exactly on the max_count-th call (the one made by _check_watcher_guard).
        exits_deque = deque([now] * (max_count - 1))

        tripped = await h._check_watcher_guard(
            exits_deque, 'watcher_crashloop', max_count, window, now
        )

        assert tripped is True, 'Guard should have tripped'
        h.pause_scheduler.assert_awaited_once_with('watcher_crashloop')

        l2s = [e for e in queue.get_pending() if e.level == 2]
        assert len(l2s) == 1, (
            f'Expected 1 L2 after crashloop trip; got {len(l2s)}'
        )
        assert 'watcher_crashloop' in l2s[0].summary, (
            f'L2 summary must contain "watcher_crashloop"; got {l2s[0].summary!r}'
        )
        assert l2s[0].root_cause == h._WATCHER_OUTAGE_ROOT_CAUSE

    @pytest.mark.asyncio
    async def test_trip_files_outage_l2_misconfigured(self, tmp_path: Path) -> None:
        """Misconfigured trip: L2 outage escalation is filed with reason='watcher_misconfigured'."""
        max_count = 3
        window = 600.0
        now = time.monotonic()

        h, queue = _make_harness_with_queue(tmp_path)
        h.pause_scheduler = AsyncMock()  # type: ignore[method-assign]

        exits_deque = deque([now] * (max_count - 1))

        tripped = await h._check_watcher_guard(
            exits_deque, 'watcher_misconfigured', max_count, window, now
        )

        assert tripped is True, 'Guard should have tripped'

        l2s = [e for e in queue.get_pending() if e.level == 2]
        assert len(l2s) == 1, (
            f'Expected 1 L2 after misconfigured trip; got {len(l2s)}'
        )
        assert 'watcher_misconfigured' in l2s[0].summary, (
            f'L2 summary must contain "watcher_misconfigured"; got {l2s[0].summary!r}'
        )

    @pytest.mark.asyncio
    async def test_no_l2_when_guard_does_not_trip(self, tmp_path: Path) -> None:
        """When guard does NOT trip (below threshold), no L2 is filed."""
        max_count = 5
        window = 600.0
        now = time.monotonic()

        h, queue = _make_harness_with_queue(tmp_path)
        h.pause_scheduler = AsyncMock()  # type: ignore[method-assign]

        exits_deque: deque[float] = deque()  # empty — first entry added by guard itself

        tripped = await h._check_watcher_guard(
            exits_deque, 'watcher_crashloop', max_count, window, now
        )

        assert tripped is False, 'Guard should not have tripped'
        l2s = [e for e in queue.get_pending() if e.level == 2]
        assert len(l2s) == 0, (
            f'Expected 0 L2s when guard does not trip; got {l2s}'
        )


# ---------------------------------------------------------------------------
# task 1501: _start_watcher_supervisor disabled branch must file outage L2
# ---------------------------------------------------------------------------

class TestStartWatcherSupervisorDisabledFilesL2:
    """When watcher_supervisor_enabled=False, _start_watcher_supervisor must file an L2."""

    def test_disabled_branch_files_outage_l2(self, tmp_path: Path) -> None:
        """With watcher_supervisor_enabled=False, calling _start_watcher_supervisor
        must: (a) not create an asyncio.Task, (b) file an L2 outage escalation
        with summary containing 'watcher_supervisor_enabled=false'.
        """
        h, queue = _make_harness_with_queue(tmp_path)
        h.config = h.config.model_copy(update={'watcher_supervisor_enabled': False})

        with patch('orchestrator.harness.asyncio.create_task') as mock_ct:
            h._start_watcher_supervisor()
            mock_ct.assert_not_called()

        assert h._watcher_supervisor_task is None

        l2s = [e for e in queue.get_pending() if e.level == 2]
        assert len(l2s) == 1, (
            f'Expected 1 L2 outage escalation when supervisor disabled; got {len(l2s)}'
        )
        assert 'watcher_supervisor_enabled=false' in l2s[0].summary.lower(), (
            f'L2 summary must mention watcher_supervisor_enabled=false; got {l2s[0].summary!r}'
        )
        assert l2s[0].root_cause == h._WATCHER_OUTAGE_ROOT_CAUSE


# ---------------------------------------------------------------------------
# task 1501: _enforce_cost_ceilings watcher ceiling trip must file outage L2
# ---------------------------------------------------------------------------

class TestEnforceCostCeilingsFilesL2:
    """_enforce_cost_ceilings watcher ceiling trip must file an L2 outage escalation."""

    @pytest.mark.asyncio
    async def test_watcher_ceiling_trip_files_outage_l2(self, tmp_path: Path) -> None:
        """When the watcher 24h cost ceiling trips, an L2 outage escalation is filed.

        Build a Harness with a real EscalationQueue, stub cost_store whose
        cost_totals_in_window returns (0.0, watcher_ceiling + 1.0) so the
        watcher branch trips.  Stub pause_scheduler as AsyncMock and scheduler
        with is_paused=False.  Assert:
        - pause_scheduler was awaited once with 'cost_ceiling_watcher_exceeded'
        - queue.get_pending() contains an L2 with root_cause='watcher_supervisor_down'
          and summary containing 'cost_ceiling_watcher_exceeded'
        """
        h, queue = _make_harness_with_queue(tmp_path)

        watcher_ceiling = h.config.watcher_daily_cost_ceiling_usd
        # Stub scheduler with is_paused=False so the early return is not triggered
        h.scheduler = MagicMock()
        h.scheduler.is_paused = False
        # Stub cost_store so the watcher branch trips
        h.cost_store = MagicMock()
        h.cost_store.cost_totals_in_window = AsyncMock(
            return_value=(0.0, watcher_ceiling + 1.0)
        )
        h.pause_scheduler = AsyncMock()

        await h._enforce_cost_ceilings()

        h.pause_scheduler.assert_awaited_once_with('cost_ceiling_watcher_exceeded')

        l2s = [e for e in queue.get_pending() if e.level == 2]
        assert len(l2s) == 1, (
            f'Expected 1 L2 outage escalation after watcher ceiling trip; got {len(l2s)}'
        )
        assert l2s[0].root_cause == h._WATCHER_OUTAGE_ROOT_CAUSE, (
            f'Expected root_cause={h._WATCHER_OUTAGE_ROOT_CAUSE!r}; got {l2s[0].root_cause!r}'
        )
        assert 'cost_ceiling_watcher_exceeded' in l2s[0].summary, (
            f'L2 summary must contain "cost_ceiling_watcher_exceeded"; got {l2s[0].summary!r}'
        )


# ---------------------------------------------------------------------------
# task 1501: _watcher_supervisor_loop healthy-clean branch resolves outage L2
# ---------------------------------------------------------------------------

def _make_loop_harness_with_queue(tmp_path: Path) -> tuple[Harness, EscalationQueue]:
    """Extend _make_loop_harness with a real EscalationQueue for recovery tests."""
    h = _make_loop_harness(tmp_path)
    queue = EscalationQueue(tmp_path / 'esc')
    h._escalation_queue = queue
    return h, queue


class TestWatcherSupervisorLoopRecoveryResolvesL2:
    """_watcher_supervisor_loop healthy-clean branch resolves the open outage L2."""

    @pytest.mark.asyncio
    async def test_healthy_clean_rotation_resolves_outage_l2(self, tmp_path: Path) -> None:
        """A healthy-clean rotation (duration >= min) resolves the open outage L2.

        Pre-file an outage L2, drive _watcher_supervisor_loop with one rotation
        lasting >= watcher_misconfigured_min_rotation_secs (healthy-clean path),
        then assert the L2 is no longer pending (it was resolved).
        """
        h, queue = _make_loop_harness_with_queue(tmp_path)
        min_secs = 30.0
        h.config = h.config.model_copy(update={'watcher_misconfigured_min_rotation_secs': min_secs})
        h.pause_scheduler = AsyncMock()  # type: ignore[method-assign]

        # Pre-file the outage L2
        h._file_watcher_outage_l2('watcher_crashloop')
        pending_before = [
            e for e in queue.get_pending()
            if e.level == 2 and e.root_cause == h._WATCHER_OUTAGE_ROOT_CAUSE
        ]
        assert len(pending_before) == 1, 'Pre-condition: outage L2 must be filed before loop runs'

        # Drive with one healthy-clean rotation (duration > min_secs)
        await _run_supervisor_with_rotation_durations(h, [min_secs + 1.0])

        pending_after = [
            e for e in queue.get_pending()
            if e.level == 2 and e.root_cause == h._WATCHER_OUTAGE_ROOT_CAUSE
        ]
        assert len(pending_after) == 0, (
            f'Expected outage L2 to be resolved after healthy-clean rotation; '
            f'still pending: {pending_after}'
        )

    @pytest.mark.asyncio
    async def test_degenerate_clean_does_NOT_resolve_outage_l2(self, tmp_path: Path) -> None:
        """A degenerate-clean rotation (duration < min) does NOT resolve the open outage L2.

        Pre-file an outage L2, drive _watcher_supervisor_loop with one rotation
        lasting < watcher_misconfigured_min_rotation_secs (degenerate-clean path),
        and assert the outage L2 remains pending.
        """
        h, queue = _make_loop_harness_with_queue(tmp_path)
        min_secs = 30.0
        h.config = h.config.model_copy(update={'watcher_misconfigured_min_rotation_secs': min_secs})
        h.pause_scheduler = AsyncMock()  # type: ignore[method-assign]

        # Pre-file the outage L2
        h._file_watcher_outage_l2('watcher_crashloop')
        pending_before = [
            e for e in queue.get_pending()
            if e.level == 2 and e.root_cause == h._WATCHER_OUTAGE_ROOT_CAUSE
        ]
        assert len(pending_before) == 1, 'Pre-condition: outage L2 must be filed before loop runs'

        # Drive with one degenerate-clean rotation (duration < min_secs)
        await _run_supervisor_with_rotation_durations(h, [min_secs - 1.0])

        pending_after = [
            e for e in queue.get_pending()
            if e.level == 2 and e.root_cause == h._WATCHER_OUTAGE_ROOT_CAUSE
        ]
        assert len(pending_after) == 1, (
            f'Expected outage L2 to remain pending after degenerate-clean rotation; '
            f'got {len(pending_after)} pending'
        )


# ---------------------------------------------------------------------------
# task 1501: _WATCHER_ALLOWED_TOOLS must include promote_to_l2
# ---------------------------------------------------------------------------

class TestWatcherAllowedTools:
    """_WATCHER_ALLOWED_TOOLS includes the L2-promotion tool (task 1501)."""

    def test_promote_to_l2_in_allowed_tools(self) -> None:
        """mcp__escalation__promote_to_l2 must be in _WATCHER_ALLOWED_TOOLS.

        The autonomous L1-watcher agent can promote escalations to L2 when they
        exceed its scope — this requires the promote_to_l2 MCP tool in its
        allowed-tools allowlist (consumer-per-level contract from
        plans/escalation-l2-tiering.md).
        """
        assert 'mcp__escalation__promote_to_l2' in _WATCHER_ALLOWED_TOOLS, (
            'mcp__escalation__promote_to_l2 must be in _WATCHER_ALLOWED_TOOLS '
            'so the autonomous watcher can promote L1→L2 escalations; '
            f'current list: {_WATCHER_ALLOWED_TOOLS}'
        )
