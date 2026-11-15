"""Tests for scripts/laptop_battery_guard.py, the low-battery clean-shutdown
guard for leo-laptop (task 9051).

Every fake power_supply tree carries ACAD online=1 and a USB-C source node
reading online=1, the readings that stayed up while the battery ran flat on
2026-11-08 and 2026-11-10. So each WARN and POWEROFF case here also proves
that AC presence cannot suppress the guard.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from cli_subprocess_timeout import cli_timeout_from_env
from laptop_battery_guard import (
    BatteryReadError,
    BatteryReading,
    Thresholds,
    Verdict,
    decide,
    read_batteries,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

THRESHOLDS = Thresholds(floor_percent=10, warn_margin_percent=5)

GUARD = Path(__file__).parent.parent / "laptop_battery_guard.py"
CLI_TIMEOUT = cli_timeout_from_env("LAPTOP_BATTERY_GUARD_CLI_TIMEOUT")

SHIMS = {
    "systemctl": 'echo "systemctl $*" >>"$FAKE_STATE/calls.log"\nexit "${FAKE_SYSTEMCTL_RC:-0}"\n',
    "systemd-run": 'echo "systemd-run $*" >>"$FAKE_STATE/calls.log"\nexit 0\n',
    "fake-poweroff": 'echo "poweroff $*" >>"$FAKE_STATE/calls.log"\nexit "${FAKE_POWEROFF_RC:-0}"\n',
}
KILL_SCOPES = "systemctl --user --machine=leo@.host kill --signal=SIGKILL df-verify-*.scope"
STOP_SCOPES = "systemctl --user --machine=leo@.host stop df-verify-*.scope"


def _write_node(node: Path, files: dict[str, str]) -> None:
    node.mkdir(parents=True)
    for name, value in files.items():
        (node / name).write_text(f"{value}\n")


def power_supply(tmp_path, batteries, extra=()):
    """A fake /sys/class/power_supply: AC always online, plus *batteries*.

    *batteries* maps a BAT* name to (status, capacity). *extra* is a sequence
    of (node name, files) pairs for further non-BAT* nodes.
    """
    root = tmp_path / "power_supply"
    _write_node(root / "ACAD", {"type": "Mains", "online": "1"})
    _write_node(root / "ucsi-source-psy-USBC000:003", {"type": "USB", "online": "1"})
    for name, files in extra:
        _write_node(root / name, files)
    for name, (status, capacity) in batteries.items():
        _write_node(
            root / name, {"type": "Battery", "status": status, "capacity": str(capacity)},
        )
    return root


def readings(*pairs):
    return tuple(
        BatteryReading(name=f"BAT{i}", status=status, capacity=capacity)
        for i, (status, capacity) in enumerate(pairs)
    )


class TestReadBatteries:
    def test_no_battery_dirs_reads_empty(self, tmp_path):
        assert read_batteries(power_supply(tmp_path, {})) == ()

    def test_reads_status_and_capacity(self, tmp_path):
        root = power_supply(tmp_path, {"BAT1": ("Discharging", 42)})
        assert read_batteries(root) == (
            BatteryReading(name="BAT1", status="Discharging", capacity=42),
        )

    def test_peripheral_battery_is_not_read(self, tmp_path):
        mouse = ("hidpp_battery_0", {"type": "Battery", "status": "Discharging", "capacity": "5"})
        assert read_batteries(power_supply(tmp_path, {}, extra=[mouse])) == ()

    def test_unparseable_capacity_names_the_file(self, tmp_path):
        root = power_supply(tmp_path, {"BAT1": ("Discharging", "garbage")})
        with pytest.raises(BatteryReadError, match=re.escape(str(root / "BAT1" / "capacity"))):
            read_batteries(root)

    def test_missing_capacity_names_the_file(self, tmp_path):
        root = power_supply(tmp_path, {"BAT1": ("Discharging", 42)})
        (root / "BAT1" / "capacity").unlink()
        with pytest.raises(BatteryReadError, match=re.escape(str(root / "BAT1" / "capacity"))):
            read_batteries(root)


class TestDecide:
    @pytest.mark.parametrize(
        ("status", "capacity", "verdict"),
        [
            ("Charging", 5, Verdict.OK),
            ("Not charging", 3, Verdict.OK),
            ("Full", 100, Verdict.OK),
            ("Discharging", 16, Verdict.OK),
            ("Discharging", 15, Verdict.WARN),
            ("Discharging", 11, Verdict.WARN),
            ("Discharging", 10, Verdict.POWEROFF),
            ("Discharging", 0, Verdict.POWEROFF),
        ],
    )
    def test_single_battery(self, status, capacity, verdict):
        assert decide(readings((status, capacity)), THRESHOLDS) is verdict

    def test_no_battery(self):
        assert decide((), THRESHOLDS) is Verdict.NO_BATTERY

    def test_highest_capacity_stands_for_the_pack(self):
        pack = readings(("Not charging", 80), ("Discharging", 5))
        assert decide(pack, THRESHOLDS) is Verdict.OK

    def test_every_pack_low_powers_off(self):
        pack = readings(("Discharging", 9), ("Discharging", 7))
        assert decide(pack, THRESHOLDS) is Verdict.POWEROFF


class TestThresholds:
    @pytest.mark.parametrize(
        ("kwargs", "field", "value"),
        [
            ({"floor_percent": 101, "warn_margin_percent": 5}, "floor_percent", "101"),
            ({"floor_percent": -1, "warn_margin_percent": 5}, "floor_percent", "-1"),
            ({"floor_percent": 10, "warn_margin_percent": -1}, "warn_margin_percent", "-1"),
        ],
    )
    def test_rejects_out_of_range(self, kwargs, field, value):
        with pytest.raises(ValueError) as excinfo:
            Thresholds(**kwargs)
        assert field in str(excinfo.value)
        assert value in str(excinfo.value)

    def test_accepts_zero_floor_and_margin(self):
        assert Thresholds(floor_percent=0, warn_margin_percent=0).floor_percent == 0


@pytest.fixture
def world(tmp_path):
    """Shims for systemctl, systemd-run and the poweroff command, all logging
    to one calls file so the order of the guard's actions can be asserted."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in SHIMS.items():
        shim = bin_dir / name
        shim.write_text("#!/usr/bin/env bash\n" + body)
        shim.chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()
    return {"tmp": tmp_path, "bin": bin_dir, "state": state}


def run(world, batteries, *args, timeout=CLI_TIMEOUT, **env_overrides):
    """Run the guard CLI against a fresh fake tree holding *batteries*."""
    root = power_supply(world["tmp"], batteries)
    env = {
        **os.environ,
        "PATH": f"{world['bin']}{os.pathsep}{os.environ['PATH']}",
        "FAKE_STATE": str(world["state"]),
        "BATTERY_GUARD_POWER_SUPPLY_DIR": str(root),
        "BATTERY_GUARD_POWEROFF_CMD": f"{world['bin'] / 'fake-poweroff'} --now",
        "BATTERY_GUARD_SCOPE_USER": "leo",
        **env_overrides,
    }
    return subprocess.run(
        [sys.executable, str(GUARD), *args],
        env=env, capture_output=True, text=True, timeout=timeout,
    )


def calls(world):
    log = world["state"] / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


class TestCliQuiet:
    def test_no_battery_is_silent(self, world):
        result = run(world, {})
        assert (result.returncode, result.stdout, calls(world)) == (0, "", [])

    @pytest.mark.parametrize(("status", "capacity"), [("Charging", 5), ("Not charging", 3)])
    def test_not_discharging_takes_no_action(self, world, status, capacity):
        result = run(world, {"BAT1": (status, capacity)})
        assert (result.returncode, calls(world)) == (0, [])


class TestCliWarn:
    def test_warn_band_logs_and_notifies_only(self, world):
        result = run(world, {"BAT1": ("Discharging", 11)})
        assert result.returncode == 0, result.stderr
        lines = result.stdout.splitlines()
        assert len(lines) == 1
        assert all(part in lines[0] for part in ("BAT1", "Discharging", "11%", "floor 10%"))
        made = calls(world)
        assert len(made) == 1
        assert made[0].startswith("systemd-run ")
        assert all(part in made[0] for part in ("--user", "--machine=leo@.host", "notify-send"))

    def test_zero_margin_disables_the_warning(self, world):
        result = run(world, {"BAT1": ("Discharging", 11)}, BATTERY_GUARD_WARN_MARGIN_PERCENT="0")
        assert (result.returncode, calls(world)) == (0, [])


class TestCliPoweroff:
    def test_stops_verify_scopes_then_powers_off(self, world):
        result = run(world, {"BAT1": ("Discharging", 10)})
        assert result.returncode == 0, result.stderr
        assert "BAT1 Discharging 10%" in result.stdout
        assert "df-verify-*.scope" in result.stdout
        assert "poweroff" in result.stdout
        made = calls(world)
        assert made.index(KILL_SCOPES) < made.index(STOP_SCOPES) < made.index("poweroff --now")
        assert made[-1] == "poweroff --now"

    def test_scope_stop_failure_never_blocks_the_poweroff(self, world):
        result = run(world, {"BAT1": ("Discharging", 10)}, FAKE_SYSTEMCTL_RC="1")
        assert result.returncode == 0, result.stderr
        assert calls(world)[-1] == "poweroff --now"
        output = result.stdout + result.stderr
        assert "stop df-verify-*.scope" in output
        assert "failed" in output

    def test_poweroff_failure_exits_1(self, world):
        result = run(world, {"BAT1": ("Discharging", 10)}, FAKE_POWEROFF_RC="1")
        assert result.returncode == 1
        assert "poweroff request failed" in result.stderr

    def test_floor_is_configurable(self, world):
        result = run(world, {"BAT1": ("Discharging", 15)}, BATTERY_GUARD_FLOOR_PERCENT="20")
        assert result.returncode == 0, result.stderr
        assert "poweroff --now" in calls(world)


class TestCliDryRun:
    def test_reports_the_verdict_without_acting(self, world):
        result = run(world, {"BAT1": ("Discharging", 10)}, "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "BAT1 Discharging 10%" in result.stdout
        assert "POWEROFF" in result.stdout
        assert calls(world) == []

    def test_reports_a_missing_battery(self, world):
        result = run(world, {}, "--dry-run")
        assert result.returncode == 0, result.stderr
        assert "no battery" in result.stdout
        assert str(world["tmp"] / "power_supply") in result.stdout


class TestCliRefusals:
    def test_malformed_env_value_exits_2(self, world):
        result = run(world, {"BAT1": ("Discharging", 5)}, BATTERY_GUARD_FLOOR_PERCENT="ten")
        assert result.returncode == 2
        assert "BATTERY_GUARD_FLOOR_PERCENT" in result.stderr
        assert "ten" in result.stderr
        assert calls(world) == []

    def test_unreadable_battery_exits_1_without_acting(self, world):
        result = run(world, {"BAT1": ("Discharging", "garbage")})
        assert result.returncode == 1
        assert str(world["tmp"] / "power_supply" / "BAT1" / "capacity") in result.stderr
        assert calls(world) == []
