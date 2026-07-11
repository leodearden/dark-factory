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

import json
from pathlib import Path

from orchestrator.fleet_heartbeat import (
    DEFAULT_FLEET_DIR,
    build_heartbeat_payload,
    resolve_fleet_dir,
    write_heartbeat,
)

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


# ---------------------------------------------------------------------------
# build_heartbeat_payload
# ---------------------------------------------------------------------------


class TestBuildHeartbeatPayload:
    """build_heartbeat_payload(...) — the five-field on-disk payload shape."""

    def test_returns_exactly_five_fields_with_type_fidelity(self):
        """Payload has exactly {unit, merge_idle, depth, queue_empty, ts_epoch}, values/types preserved."""
        payload = build_heartbeat_payload(
            unit='orchestrator-reify.service',
            merge_idle=True,
            depth=0,
            queue_empty=True,
            ts_epoch=1234567890.5,
        )

        assert set(payload.keys()) == {'unit', 'merge_idle', 'depth', 'queue_empty', 'ts_epoch'}
        assert payload['unit'] == 'orchestrator-reify.service'
        assert isinstance(payload['unit'], str)
        assert payload['merge_idle'] is True
        assert isinstance(payload['merge_idle'], bool)
        assert payload['depth'] == 0
        assert isinstance(payload['depth'], int)
        assert payload['queue_empty'] is True
        assert isinstance(payload['queue_empty'], bool)
        assert payload['ts_epoch'] == 1234567890.5
        assert isinstance(payload['ts_epoch'], float)

    def test_busy_values_preserved(self):
        """A busy/non-idle tick's values pass through unchanged (no truthy coercion)."""
        payload = build_heartbeat_payload(
            unit='orchestrator-dark-factory.service',
            merge_idle=False,
            depth=3,
            queue_empty=False,
            ts_epoch=42.0,
        )

        assert payload == {
            'unit': 'orchestrator-dark-factory.service',
            'merge_idle': False,
            'depth': 3,
            'queue_empty': False,
            'ts_epoch': 42.0,
        }


# ---------------------------------------------------------------------------
# write_heartbeat
# ---------------------------------------------------------------------------


class TestWriteHeartbeat:
    """write_heartbeat(fleet_dir, unit, payload) — atomic tmp-file + os.replace writer."""

    def test_writes_unit_json_and_creates_missing_parent_dirs(self, tmp_path):
        """Writes <fleet_dir>/<unit>.json, creating missing nested parent dirs."""
        fleet_dir = tmp_path / 'deep' / 'nested' / 'fleet'
        payload = build_heartbeat_payload(
            unit='orchestrator-reify.service',
            merge_idle=True,
            depth=0,
            queue_empty=True,
            ts_epoch=111.0,
        )

        result = write_heartbeat(fleet_dir, 'orchestrator-reify.service', payload)

        expected = fleet_dir / 'orchestrator-reify.service.json'
        assert expected.exists(), (
            'Expected heartbeat file to be created, including missing parent dirs'
        )
        assert result == expected

    def test_written_content_round_trips_the_exact_payload(self, tmp_path):
        """File content json.loads back to the exact payload passed in."""
        payload = build_heartbeat_payload(
            unit='orchestrator-dark-factory.service',
            merge_idle=False,
            depth=2,
            queue_empty=False,
            ts_epoch=222.5,
        )

        result = write_heartbeat(tmp_path, 'orchestrator-dark-factory.service', payload)

        assert json.loads(result.read_text()) == payload

    def test_no_leftover_tmp_file_after_write(self, tmp_path):
        """No <unit>.json.tmp remains after the atomic rename."""
        payload = build_heartbeat_payload(
            unit='orchestrator-reify.service',
            merge_idle=True,
            depth=0,
            queue_empty=True,
            ts_epoch=333.0,
        )

        write_heartbeat(tmp_path, 'orchestrator-reify.service', payload)

        assert not (tmp_path / 'orchestrator-reify.service.json.tmp').exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == ['orchestrator-reify.service.json']

    def test_returns_the_final_path(self, tmp_path):
        """Returns the final on-disk Path (not the tmp path)."""
        payload = build_heartbeat_payload(
            unit='orchestrator-reify.service',
            merge_idle=True,
            depth=0,
            queue_empty=True,
            ts_epoch=444.0,
        )

        result = write_heartbeat(tmp_path, 'orchestrator-reify.service', payload)

        assert result == tmp_path / 'orchestrator-reify.service.json'

    def test_empty_unit_falls_back_to_deterministic_filename(self, tmp_path):
        """unit='' never produces a file literally named '.json'; falls back deterministically."""
        payload = build_heartbeat_payload(
            unit='',
            merge_idle=True,
            depth=0,
            queue_empty=True,
            ts_epoch=555.0,
        )

        result = write_heartbeat(tmp_path, '', payload)

        assert result.name != '.json'
        assert result.exists()
        assert result == tmp_path / 'unknown-unit.json'
