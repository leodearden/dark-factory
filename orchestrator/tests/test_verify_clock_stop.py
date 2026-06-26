"""Tests for the clock-stop verify timeout feature (task 1916).

Covers:
- OrchestratorConfig field defaults and defaults.yaml integration (step-1/step-2)
- OrchestratorConfig model_validator for enabled markers (step-3/step-4)
- _match_clock_marker unit tests (step-5/step-6)
- _run_cmd clock-stop behavioral integration tests (step-7/step-8)
- max_total_secs cumulative-stop cap (step-9/step-10)
- _run_or_skip_timed / run_verification wiring test (step-11/step-12)
"""

from __future__ import annotations

import asyncio
from importlib import resources as pkg_resources
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orchestrator.config import OrchestratorConfig


def _load_package_defaults() -> dict:
    """Read the shipped defaults.yaml so tests stay in sync automatically."""
    defaults_file = pkg_resources.files('orchestrator') / 'defaults.yaml'
    return yaml.safe_load(defaults_file.read_text())


# ── Step-1 / Step-2: Config default tests ─────────────────────────────────────


class TestClockStopConfigDefaults:
    """OrchestratorConfig exposes clock-stop fields with the correct Pydantic
    defaults AND the shipped defaults.yaml activates the feature."""

    def test_pydantic_default_enabled_is_false(self):
        """Raw Pydantic field default for verify_clock_stop_enabled is False.

        This keeps directly-constructed OrchestratorConfig (and all existing
        _run_cmd test doubles that bypass the yaml settings source) opt-out by
        default, exactly mirroring verify_cold_command_timeout_secs (Pydantic
        default None / defaults.yaml ships 5400).
        """
        field_info = OrchestratorConfig.model_fields['verify_clock_stop_enabled']
        assert field_info.default is False

    def test_pydantic_default_marker_stop(self):
        """Raw Pydantic default for verify_clock_stop_marker_stop."""
        field_info = OrchestratorConfig.model_fields['verify_clock_stop_marker_stop']
        assert field_info.default == '@@REIFY_CLOCK_STOP@@'

    def test_pydantic_default_marker_heartbeat(self):
        """Raw Pydantic default for verify_clock_stop_marker_heartbeat."""
        field_info = OrchestratorConfig.model_fields['verify_clock_stop_marker_heartbeat']
        assert field_info.default == '@@REIFY_CLOCK_HEARTBEAT@@'

    def test_pydantic_default_marker_start(self):
        """Raw Pydantic default for verify_clock_stop_marker_start."""
        field_info = OrchestratorConfig.model_fields['verify_clock_stop_marker_start']
        assert field_info.default == '@@REIFY_CLOCK_START@@'

    def test_pydantic_default_heartbeat_idle_max(self):
        """Raw Pydantic default for verify_clock_stop_heartbeat_idle_max is 180.0."""
        field_info = OrchestratorConfig.model_fields['verify_clock_stop_heartbeat_idle_max']
        assert field_info.default == 180.0

    def test_pydantic_default_max_total_secs(self):
        """Raw Pydantic default for verify_clock_stop_max_total_secs is 0.0 (unlimited)."""
        field_info = OrchestratorConfig.model_fields['verify_clock_stop_max_total_secs']
        assert field_info.default == 0.0

    def test_defaults_yaml_enables_clock_stop(self, monkeypatch, tmp_path):
        """defaults.yaml ships verify_clock_stop_enabled: true so the deployed
        orchestrator activates the seam for reify without a separate ops config
        change (mirrors the cold-timeout activation pattern)."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        assert config.verify_clock_stop_enabled is True

    def test_defaults_yaml_marker_stop(self, monkeypatch, tmp_path):
        """defaults.yaml ships the @@REIFY_CLOCK_STOP@@ marker string."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        defaults = _load_package_defaults()
        assert config.verify_clock_stop_marker_stop == defaults['verify_clock_stop_marker_stop']
        assert config.verify_clock_stop_marker_stop == '@@REIFY_CLOCK_STOP@@'

    def test_defaults_yaml_marker_heartbeat(self, monkeypatch, tmp_path):
        """defaults.yaml ships the @@REIFY_CLOCK_HEARTBEAT@@ marker string."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        defaults = _load_package_defaults()
        assert config.verify_clock_stop_marker_heartbeat == defaults['verify_clock_stop_marker_heartbeat']
        assert config.verify_clock_stop_marker_heartbeat == '@@REIFY_CLOCK_HEARTBEAT@@'

    def test_defaults_yaml_marker_start(self, monkeypatch, tmp_path):
        """defaults.yaml ships the @@REIFY_CLOCK_START@@ marker string."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        defaults = _load_package_defaults()
        assert config.verify_clock_stop_marker_start == defaults['verify_clock_stop_marker_start']
        assert config.verify_clock_stop_marker_start == '@@REIFY_CLOCK_START@@'

    def test_defaults_yaml_heartbeat_idle_max(self, monkeypatch, tmp_path):
        """defaults.yaml ships verify_clock_stop_heartbeat_idle_max == 180."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        defaults = _load_package_defaults()
        assert config.verify_clock_stop_heartbeat_idle_max == defaults['verify_clock_stop_heartbeat_idle_max']
        assert config.verify_clock_stop_heartbeat_idle_max == 180.0

    def test_defaults_yaml_max_total_secs(self, monkeypatch, tmp_path):
        """defaults.yaml ships verify_clock_stop_max_total_secs == 0 (unlimited)."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        defaults = _load_package_defaults()
        assert config.verify_clock_stop_max_total_secs == defaults['verify_clock_stop_max_total_secs']
        assert config.verify_clock_stop_max_total_secs == 0.0


# ── Step-3 / Step-4: Validator tests ──────────────────────────────────────────


class TestClockStopConfigValidator:
    """model_validator fires when enabled=True: markers must be non-empty and distinct."""

    def test_enabled_with_empty_stop_marker_raises(self):
        """Constructing with enabled=True and empty marker_stop raises ValidationError."""
        with pytest.raises(ValidationError):
            OrchestratorConfig(
                verify_clock_stop_enabled=True,
                verify_clock_stop_marker_stop='',
            )

    def test_enabled_with_empty_heartbeat_marker_raises(self):
        """Constructing with enabled=True and empty marker_heartbeat raises ValidationError."""
        with pytest.raises(ValidationError):
            OrchestratorConfig(
                verify_clock_stop_enabled=True,
                verify_clock_stop_marker_heartbeat='',
            )

    def test_enabled_with_empty_start_marker_raises(self):
        """Constructing with enabled=True and empty marker_start raises ValidationError."""
        with pytest.raises(ValidationError):
            OrchestratorConfig(
                verify_clock_stop_enabled=True,
                verify_clock_stop_marker_start='',
            )

    def test_enabled_with_duplicate_stop_heartbeat_raises(self):
        """Constructing with enabled=True and marker_stop == marker_heartbeat raises ValidationError."""
        with pytest.raises(ValidationError):
            OrchestratorConfig(
                verify_clock_stop_enabled=True,
                verify_clock_stop_marker_stop='@@SAME@@',
                verify_clock_stop_marker_heartbeat='@@SAME@@',
                verify_clock_stop_marker_start='@@DIFFERENT@@',
            )

    def test_enabled_with_duplicate_heartbeat_start_raises(self):
        """Constructing with enabled=True and marker_heartbeat == marker_start raises ValidationError."""
        with pytest.raises(ValidationError):
            OrchestratorConfig(
                verify_clock_stop_enabled=True,
                verify_clock_stop_marker_stop='@@STOP@@',
                verify_clock_stop_marker_heartbeat='@@SAME@@',
                verify_clock_stop_marker_start='@@SAME@@',
            )

    def test_enabled_with_duplicate_stop_start_raises(self):
        """Constructing with enabled=True and marker_stop == marker_start raises ValidationError."""
        with pytest.raises(ValidationError):
            OrchestratorConfig(
                verify_clock_stop_enabled=True,
                verify_clock_stop_marker_stop='@@SAME@@',
                verify_clock_stop_marker_heartbeat='@@HB@@',
                verify_clock_stop_marker_start='@@SAME@@',
            )

    def test_enabled_with_default_markers_is_valid(self):
        """Constructing with enabled=True and default markers does NOT raise."""
        # This should not raise (the default @@REIFY_CLOCK_*@@ strings are
        # non-empty and distinct).
        config = OrchestratorConfig(
            verify_clock_stop_enabled=True,
        )
        assert config.verify_clock_stop_enabled is True

    def test_enabled_with_custom_valid_markers_is_valid(self):
        """Constructing with enabled=True and valid custom markers does NOT raise."""
        config = OrchestratorConfig(
            verify_clock_stop_enabled=True,
            verify_clock_stop_marker_stop='<<STOP>>',
            verify_clock_stop_marker_heartbeat='<<HB>>',
            verify_clock_stop_marker_start='<<START>>',
        )
        assert config.verify_clock_stop_marker_stop == '<<STOP>>'

    def test_disabled_with_empty_markers_does_not_raise(self):
        """When enabled=False, blank markers do NOT raise (validator is no-op when disabled)."""
        # Should not raise even though markers are empty strings.
        config = OrchestratorConfig(
            verify_clock_stop_enabled=False,
            verify_clock_stop_marker_stop='',
            verify_clock_stop_marker_heartbeat='',
            verify_clock_stop_marker_start='',
        )
        assert config.verify_clock_stop_enabled is False


# ── Step-5 / Step-6: _match_clock_marker unit tests ───────────────────────────


class TestMatchClockMarker:
    """_match_clock_marker returns 'stop'/'heartbeat'/'start'/None by substring match."""

    @pytest.fixture()
    def cfg(self):
        """Default ClockStopConfig using the @@REIFY_CLOCK_*@@ markers."""
        from orchestrator.verify import ClockStopConfig
        return ClockStopConfig(
            marker_stop='@@REIFY_CLOCK_STOP@@',
            marker_heartbeat='@@REIFY_CLOCK_HEARTBEAT@@',
            marker_start='@@REIFY_CLOCK_START@@',
            heartbeat_idle_max=180.0,
            max_total_secs=0.0,
        )

    def test_stop_with_trailing_fields(self, cfg):
        """STOP marker with trailing reason=/pid= fields -> 'stop'."""
        from orchestrator.verify import _match_clock_marker
        line = '@@REIFY_CLOCK_STOP@@ reason=test_slot_starvation pid=1234'
        assert _match_clock_marker(line, cfg) == 'stop'

    def test_heartbeat_with_trailing_fields(self, cfg):
        """HEARTBEAT marker with trailing reason=/waited= fields -> 'heartbeat'."""
        from orchestrator.verify import _match_clock_marker
        line = '@@REIFY_CLOCK_HEARTBEAT@@ reason=psi_pressure waited=60'
        assert _match_clock_marker(line, cfg) == 'heartbeat'

    def test_start_with_trailing_fields(self, cfg):
        """START marker with trailing reason=/waited= fields -> 'start'."""
        from orchestrator.verify import _match_clock_marker
        line = '@@REIFY_CLOCK_START@@ reason=psi_pressure waited=90'
        assert _match_clock_marker(line, cfg) == 'start'

    def test_plain_log_line_returns_none(self, cfg):
        """A plain log line with no marker -> None."""
        from orchestrator.verify import _match_clock_marker
        assert _match_clock_marker('running cargo test --workspace', cfg) is None

    def test_empty_line_returns_none(self, cfg):
        """An empty line -> None."""
        from orchestrator.verify import _match_clock_marker
        assert _match_clock_marker('', cfg) is None

    def test_start_line_does_not_match_stop(self, cfg):
        """A START line does NOT match the stop classifier."""
        from orchestrator.verify import _match_clock_marker
        line = '@@REIFY_CLOCK_START@@ waited=10'
        result = _match_clock_marker(line, cfg)
        assert result == 'start', f'Expected start, got {result!r}'

    def test_stop_line_does_not_match_start(self, cfg):
        """A STOP line does NOT match the start classifier."""
        from orchestrator.verify import _match_clock_marker
        line = '@@REIFY_CLOCK_STOP@@ reason=pressure'
        result = _match_clock_marker(line, cfg)
        assert result == 'stop', f'Expected stop, got {result!r}'

    def test_custom_markers_are_honored(self):
        """ClockStopConfig custom marker strings are used for matching."""
        from orchestrator.verify import ClockStopConfig, _match_clock_marker
        cfg = ClockStopConfig(
            marker_stop='<<S>>',
            marker_heartbeat='<<H>>',
            marker_start='<<R>>',
            heartbeat_idle_max=60.0,
            max_total_secs=0.0,
        )
        assert _match_clock_marker('<<S>> reason=foo', cfg) == 'stop'
        assert _match_clock_marker('<<H>> waited=5', cfg) == 'heartbeat'
        assert _match_clock_marker('<<R>> waited=10', cfg) == 'start'
        assert _match_clock_marker('@@REIFY_CLOCK_STOP@@', cfg) is None

    def test_substring_tolerance_with_prefix(self, cfg):
        """A line with a leading harness prefix before the marker still matches."""
        from orchestrator.verify import _match_clock_marker
        line = '[harness] 2026-06-26T12:00:00 @@REIFY_CLOCK_HEARTBEAT@@ waited=30'
        assert _match_clock_marker(line, cfg) == 'heartbeat'


# ── Step-7 / Step-8: Core _run_cmd clock-stop behavioral tests ────────────────


class TestRunCmdClockStop:
    """Real-subprocess _run_cmd integration tests for the clock-stop state machine.

    Each test spawns a real bash stub and asserts on the (rc, out, timed_out)
    tuple.  Timing margins are large (sub-second budgets vs multi-second sleeps)
    so assertions are robust to CI timing jitter.
    """

    @staticmethod
    def _make_cfg(
        heartbeat_idle_max: float = 5.0,
        max_total_secs: float = 0.0,
    ):
        from orchestrator.verify import ClockStopConfig
        return ClockStopConfig(
            marker_stop='@@REIFY_CLOCK_STOP@@',
            marker_heartbeat='@@REIFY_CLOCK_HEARTBEAT@@',
            marker_start='@@REIFY_CLOCK_START@@',
            heartbeat_idle_max=heartbeat_idle_max,
            max_total_secs=max_total_secs,
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_happy_exclude_span(self, tmp_path: Path):
        """(a) HAPPY/EXCLUDE-SPAN: stopped span is excluded from wall-clock budget.

        timeout=1.0, heartbeat_idle_max=2.0.  Stub emits STOP, heartbeats for
        ~1.5s (past the 1.0 budget), then START, then exits 0.  The stopped span
        is excluded so the wall-clock budget is NOT consumed during the wait.
        Expected: rc==0, timed_out is False.
        """
        from orchestrator.verify import _run_cmd
        log_path = tmp_path / 'verify.log'
        cmd = (
            'echo "@@REIFY_CLOCK_STOP@@ reason=test"; '
            'for i in 1 2 3 4 5; do sleep 0.3; echo "@@REIFY_CLOCK_HEARTBEAT@@ waited=$i"; done; '
            'echo "@@REIFY_CLOCK_START@@ waited=1.5"; '
            'exit 0'
        )
        cfg = self._make_cfg(heartbeat_idle_max=2.0)
        rc, out, timed_out = await _run_cmd(
            cmd, tmp_path, timeout=1.0, log_path=log_path, clock_stop=cfg,
        )
        assert rc == 0, f'Expected rc==0; got rc={rc}, out={out!r}'
        assert timed_out is False, f'Expected timed_out=False; got timed_out={timed_out}'

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_heartbeat_idle_kill(self, tmp_path: Path):
        """(b) HEARTBEAT-IDLE KILL: silent stop triggers idle backstop.

        timeout=10.0, heartbeat_idle_max=0.5.  Stub emits STOP then sleeps 5s
        silently.  Expected: timed_out is True, killed well before 5s.
        """
        from orchestrator.verify import _run_cmd
        log_path = tmp_path / 'verify.log'
        cmd = (
            'echo "@@REIFY_CLOCK_STOP@@ reason=pressure"; '
            'sleep 5'
        )
        cfg = self._make_cfg(heartbeat_idle_max=0.5)
        rc, out, timed_out = await _run_cmd(
            cmd, tmp_path, timeout=10.0, log_path=log_path, clock_stop=cfg,
        )
        assert timed_out is True, f'Expected timed_out=True; got timed_out={timed_out}'

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_resumed_budget_kill(self, tmp_path: Path):
        """(c) RESUMED-BUDGET KILL: post-START wall-clock fires.

        timeout=1.0, heartbeat_idle_max=5.0.  Stub emits STOP, one HEARTBEAT,
        START quickly (~0.2s stopped), then sleeps 5s.  Expected: timed_out is
        True (resumed wall-clock fires at ~1.0s post-start).
        """
        from orchestrator.verify import _run_cmd
        log_path = tmp_path / 'verify.log'
        cmd = (
            'echo "@@REIFY_CLOCK_STOP@@ reason=pressure"; '
            'sleep 0.1; '
            'echo "@@REIFY_CLOCK_HEARTBEAT@@ waited=0.1"; '
            'sleep 0.1; '
            'echo "@@REIFY_CLOCK_START@@ waited=0.2"; '
            'sleep 5'
        )
        cfg = self._make_cfg(heartbeat_idle_max=5.0)
        rc, out, timed_out = await _run_cmd(
            cmd, tmp_path, timeout=1.0, log_path=log_path, clock_stop=cfg,
        )
        assert timed_out is True, f'Expected timed_out=True; got timed_out={timed_out}'

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_enabled_no_markers_normal_hang(self, tmp_path: Path):
        """(d) ENABLED-NO-MARKERS: clock_stop enabled but stub emits no markers.

        timeout=1.0.  Stub just sleeps 5s.  Normal hang detection fires.
        Expected: timed_out is True.
        """
        from orchestrator.verify import _run_cmd
        log_path = tmp_path / 'verify.log'
        cmd = 'sleep 5'
        cfg = self._make_cfg()
        rc, out, timed_out = await _run_cmd(
            cmd, tmp_path, timeout=1.0, log_path=log_path, clock_stop=cfg,
        )
        assert timed_out is True, f'Expected timed_out=True; got timed_out={timed_out}'

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_gate_disabled_markers_ignored(self, tmp_path: Path):
        """(e) GATE/DISABLED: clock_stop=None; markers in output are ignored.

        timeout=1.0.  Stub emits STOP + heartbeats past budget, but clock_stop
        is None so the stopped span is NOT excluded.  Expected: timed_out is True.
        """
        from orchestrator.verify import _run_cmd
        log_path = tmp_path / 'verify.log'
        cmd = (
            'echo "@@REIFY_CLOCK_STOP@@ reason=test"; '
            'for i in 1 2 3; do sleep 0.4; echo "@@REIFY_CLOCK_HEARTBEAT@@ waited=$i"; done; '
            'exit 0'
        )
        # No clock_stop kwarg (default None) — legacy behaviour
        rc, out, timed_out = await _run_cmd(
            cmd, tmp_path, timeout=1.0, log_path=log_path,
        )
        assert timed_out is True, f'Expected timed_out=True; got timed_out={timed_out}'


# ── Step-9 / Step-10: max_total_secs cap tests ────────────────────────────────


class TestRunCmdMaxTotalSecs:
    """max_total_secs cumulative-stopped-time cap."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_max_total_secs_exceeded(self, tmp_path: Path):
        """max_total_secs=0.5; stub HEARTBEATs forever (never STARTs).

        timeout=10.0 (wall-clock won't fire), heartbeat_idle_max=5.0 (idle
        won't fire).  Stub emits STOP then heartbeats every 0.2s.
        Expected: timed_out is True, killed at ~0.5s cumulative stopped time.
        """
        from orchestrator.verify import ClockStopConfig, _run_cmd
        log_path = tmp_path / 'verify.log'
        cmd = (
            'echo "@@REIFY_CLOCK_STOP@@ reason=pressure"; '
            'for i in $(seq 1 20); do sleep 0.2; echo "@@REIFY_CLOCK_HEARTBEAT@@ waited=$i"; done; '
            'exit 0'
        )
        cfg = ClockStopConfig(
            marker_stop='@@REIFY_CLOCK_STOP@@',
            marker_heartbeat='@@REIFY_CLOCK_HEARTBEAT@@',
            marker_start='@@REIFY_CLOCK_START@@',
            heartbeat_idle_max=5.0,
            max_total_secs=0.5,
        )
        rc, out, timed_out = await _run_cmd(
            cmd, tmp_path, timeout=10.0, log_path=log_path, clock_stop=cfg,
        )
        assert timed_out is True, f'Expected timed_out=True; got timed_out={timed_out}'

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_max_total_secs_zero_means_unlimited(self, tmp_path: Path):
        """max_total_secs=0 (unlimited); heartbeating stub is NOT killed by total cap.

        We run for only 1s total (timeout=1.5, but stub does STOP + quick
        heartbeat + START + exit 0) to keep the test fast.  Confirms 0 == no cap.
        """
        from orchestrator.verify import ClockStopConfig, _run_cmd
        log_path = tmp_path / 'verify.log'
        cmd = (
            'echo "@@REIFY_CLOCK_STOP@@ reason=pressure"; '
            'sleep 0.2; '
            'echo "@@REIFY_CLOCK_HEARTBEAT@@ waited=0.2"; '
            'sleep 0.2; '
            'echo "@@REIFY_CLOCK_START@@ waited=0.4"; '
            'exit 0'
        )
        cfg = ClockStopConfig(
            marker_stop='@@REIFY_CLOCK_STOP@@',
            marker_heartbeat='@@REIFY_CLOCK_HEARTBEAT@@',
            marker_start='@@REIFY_CLOCK_START@@',
            heartbeat_idle_max=5.0,
            max_total_secs=0.0,  # unlimited
        )
        rc, out, timed_out = await _run_cmd(
            cmd, tmp_path, timeout=3.0, log_path=log_path, clock_stop=cfg,
        )
        assert rc == 0, f'Expected rc==0; got rc={rc}'
        assert timed_out is False, f'Expected timed_out=False; got timed_out={timed_out}'


# ── Step-11 / Step-12: Wiring test (_run_or_skip_timed / run_verification) ────


class TestClockStopWiring:
    """_run_cmd receives clock_stop only when config.verify_clock_stop_enabled=True."""

    @pytest.mark.asyncio
    async def test_run_cmd_gets_clock_stop_when_enabled(self, tmp_path: Path):
        """When verify_clock_stop_enabled=True, _run_cmd is called with a
        ClockStopConfig whose fields match the config values."""
        from unittest.mock import AsyncMock, patch

        from orchestrator.verify import ClockStopConfig, run_verification

        captured_kwargs: dict = {}

        async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            captured_kwargs.update(kwargs)
            return 0, '', False

        config = OrchestratorConfig(
            verify_clock_stop_enabled=True,
            verify_clock_stop_marker_stop='@@REIFY_CLOCK_STOP@@',
            verify_clock_stop_marker_heartbeat='@@REIFY_CLOCK_HEARTBEAT@@',
            verify_clock_stop_marker_start='@@REIFY_CLOCK_START@@',
            verify_clock_stop_heartbeat_idle_max=180.0,
            verify_clock_stop_max_total_secs=0.0,
        )
        worktree = tmp_path / 'wt'
        worktree.mkdir()

        with patch('orchestrator.verify._run_cmd', side_effect=spy_run_cmd):
            await run_verification(
                worktree=worktree,
                config=config,
                attempt_id=None,
            )

        assert 'clock_stop' in captured_kwargs, (
            f'Expected clock_stop kwarg; got: {captured_kwargs!r}'
        )
        cs = captured_kwargs['clock_stop']
        assert isinstance(cs, ClockStopConfig), f'Expected ClockStopConfig, got {type(cs)}'
        assert cs.marker_stop == '@@REIFY_CLOCK_STOP@@'
        assert cs.marker_heartbeat == '@@REIFY_CLOCK_HEARTBEAT@@'
        assert cs.marker_start == '@@REIFY_CLOCK_START@@'
        assert cs.heartbeat_idle_max == 180.0
        assert cs.max_total_secs == 0.0

    @pytest.mark.asyncio
    async def test_run_cmd_no_clock_stop_when_disabled(self, tmp_path: Path):
        """When verify_clock_stop_enabled=False, _run_cmd is NOT called with
        clock_stop kwarg (legacy call signature preserved)."""
        from unittest.mock import patch

        from orchestrator.verify import run_verification

        captured_kwargs: dict = {}

        async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            captured_kwargs.update(kwargs)
            return 0, '', False

        config = OrchestratorConfig(
            verify_clock_stop_enabled=False,
        )
        worktree = tmp_path / 'wt'
        worktree.mkdir()

        with patch('orchestrator.verify._run_cmd', side_effect=spy_run_cmd):
            await run_verification(
                worktree=worktree,
                config=config,
                attempt_id=None,
            )

        assert 'clock_stop' not in captured_kwargs, (
            f'clock_stop should not be passed when disabled; got: {captured_kwargs!r}'
        )
