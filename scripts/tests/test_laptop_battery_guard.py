"""Tests for scripts/laptop_battery_guard.py, the low-battery clean-shutdown
guard for leo-laptop (task 9051).

Every fake power_supply tree carries ACAD online=1 and a USB-C source node
reading online=1, the readings that stayed up while the battery ran flat on
2026-11-08 and 2026-11-10. So each WARN and POWEROFF case here also proves
that AC presence cannot suppress the guard.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
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
