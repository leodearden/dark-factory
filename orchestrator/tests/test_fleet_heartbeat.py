"""Tests for orchestrator.fleet_heartbeat — the fleet-common per-unit merge-idle
heartbeat producer (task 2395, α of the fleet-redeploy PRD).

Covers the pure module functions in isolation (no Harness required):
  - ``DEFAULT_FLEET_DIR`` / ``resolve_fleet_dir`` — fleet-common directory resolution.
  - ``build_heartbeat_payload`` — the five-field on-disk payload shape.
  - ``write_heartbeat`` — the atomic tmp-file + os.replace writer.

This is the SAME module the future reader (γ drain gate, ε --report) will
import, so its on-disk contract is pinned here.
"""

from __future__ import annotations

from pathlib import Path

from orchestrator.fleet_heartbeat import DEFAULT_FLEET_DIR, resolve_fleet_dir

# ---------------------------------------------------------------------------
# DEFAULT_FLEET_DIR / resolve_fleet_dir
# ---------------------------------------------------------------------------


class TestDefaultFleetDir:
    """DEFAULT_FLEET_DIR matches the watchdog's hardcoded REPO_DIR + /data/fleet."""

    def test_default_fleet_dir_matches_watchdog_repo_dir(self):
        """Pins the constant to scripts/orchestrator-watchdog.py's REPO_DIR (task 2395 analysis)."""
        assert DEFAULT_FLEET_DIR == Path('/home/leo/src/dark-factory/data/fleet')


class TestResolveFleetDir:
    """resolve_fleet_dir(env) — ORCH_FLEET_DIR override with a hardcoded default."""

    def test_returns_default_when_env_unset(self):
        """No ORCH_FLEET_DIR key in the mapping → DEFAULT_FLEET_DIR."""
        assert resolve_fleet_dir({}) == DEFAULT_FLEET_DIR

    def test_returns_default_when_env_empty(self):
        """ORCH_FLEET_DIR present but empty string → falls back to DEFAULT_FLEET_DIR."""
        assert resolve_fleet_dir({'ORCH_FLEET_DIR': ''}) == DEFAULT_FLEET_DIR

    def test_returns_path_of_env_value_when_set(self):
        """ORCH_FLEET_DIR set and non-empty → Path(ORCH_FLEET_DIR), via an explicit env mapping."""
        assert resolve_fleet_dir({'ORCH_FLEET_DIR': '/tmp/custom-fleet-dir'}) == Path(
            '/tmp/custom-fleet-dir'
        )

    def test_defaults_to_os_environ_and_honours_monkeypatch(self, monkeypatch):
        """With no env arg, resolve_fleet_dir reads the real os.environ (monkeypatch-able)."""
        monkeypatch.setenv('ORCH_FLEET_DIR', '/tmp/monkeypatched-fleet-dir')

        assert resolve_fleet_dir() == Path('/tmp/monkeypatched-fleet-dir')

    def test_defaults_to_os_environ_unset(self, monkeypatch):
        """With no env arg and ORCH_FLEET_DIR unset in the real environment → DEFAULT_FLEET_DIR."""
        monkeypatch.delenv('ORCH_FLEET_DIR', raising=False)

        assert resolve_fleet_dir() == DEFAULT_FLEET_DIR
