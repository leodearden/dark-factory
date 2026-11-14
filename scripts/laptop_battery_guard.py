#!/usr/bin/env python3
"""Low-battery clean-shutdown guard for leo-laptop (task 9051).

AC "online" stays 1 when a weak USB-C supply cannot keep up with verify
load, so the battery can run flat while UPower believes the laptop is on
mains. This guard therefore decides from BAT*/status and BAT*/capacity
alone and never reads AC presence.

Discharging at or below the floor: stop the user's df-verify-*.scope units
and power off. Discharging within the warn margin above the floor: warn.

Environment knobs, each defaulting to the module constant of the same
name: BATTERY_GUARD_POWER_SUPPLY_DIR, BATTERY_GUARD_FLOOR_PERCENT,
BATTERY_GUARD_WARN_MARGIN_PERCENT, BATTERY_GUARD_SCOPE_USER and
BATTERY_GUARD_POWEROFF_CMD.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

DEFAULT_POWER_SUPPLY_DIR = Path("/sys/class/power_supply")
DEFAULT_FLOOR_PERCENT = 10
DEFAULT_WARN_MARGIN_PERCENT = 5
DEFAULT_SCOPE_USER = "leo"
DEFAULT_POWEROFF_CMD = "systemctl poweroff --check-inhibitors=no"
SCOPE_PATTERN = "df-verify-*.scope"

DISCHARGING = "Discharging"


@dataclass(frozen=True)
class BatteryReading:
    name: str
    status: str
    capacity: int


class Verdict(Enum):
    NO_BATTERY = "no battery"
    OK = "ok"
    WARN = "warn"
    POWEROFF = "poweroff"


@dataclass(frozen=True)
class Thresholds:
    floor_percent: int
    warn_margin_percent: int

    def __post_init__(self) -> None:
        for name in ("floor_percent", "warn_margin_percent"):
            value = getattr(self, name)
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be within 0..100, got {value}")


class BatteryReadError(Exception):
    pass


def _read_file(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        raise BatteryReadError(f"{path}: {exc}") from exc


def _read_battery(node: Path) -> BatteryReading:
    status = _read_file(node / "status").strip()
    capacity_path = node / "capacity"
    raw_capacity = _read_file(capacity_path)
    try:
        capacity = int(raw_capacity)
    except ValueError as exc:
        raise BatteryReadError(f"{capacity_path}: not an integer: {raw_capacity!r}") from exc
    return BatteryReading(name=node.name, status=status, capacity=capacity)


def read_batteries(power_supply_dir: Path) -> tuple[BatteryReading, ...]:
    return tuple(_read_battery(node) for node in sorted(power_supply_dir.glob("BAT*")))


def decide(readings: Sequence[BatteryReading], thresholds: Thresholds) -> Verdict:
    if not readings:
        return Verdict.NO_BATTERY
    if not any(r.status == DISCHARGING for r in readings):
        return Verdict.OK
    level = max(r.capacity for r in readings)
    if level <= thresholds.floor_percent:
        return Verdict.POWEROFF
    if level <= thresholds.floor_percent + thresholds.warn_margin_percent:
        return Verdict.WARN
    return Verdict.OK


def describe(readings: Sequence[BatteryReading]) -> str:
    return ", ".join(f"{r.name} {r.status} {r.capacity}%" for r in readings)


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("main() lands in task 9051 step-4")
