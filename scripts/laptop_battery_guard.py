#!/usr/bin/env python3
"""Low-battery clean-shutdown guard for leo-laptop (task 9051).

AC "online" stays 1 when a weak USB-C supply cannot keep up with verify
load, so the battery can run flat while UPower believes the laptop is on
mains. This guard therefore decides from BAT*/status and BAT*/capacity
alone and never reads AC presence.

Discharging at or below the floor: stop the user's df-verify-*.scope units
and power off. Discharging within the warn margin above the floor: warn.

Environment knobs, each defaulting to the matching DEFAULT_* constant:
BATTERY_GUARD_POWER_SUPPLY_DIR, BATTERY_GUARD_FLOOR_PERCENT,
BATTERY_GUARD_WARN_MARGIN_PERCENT, BATTERY_GUARD_SCOPE_USER and
BATTERY_GUARD_POWEROFF_CMD (split like a shell command line).
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

DEFAULT_POWER_SUPPLY_DIR = Path("/sys/class/power_supply")
DEFAULT_FLOOR_PERCENT = 10
DEFAULT_WARN_MARGIN_PERCENT = 5
DEFAULT_SCOPE_USER = "leo"
DEFAULT_POWEROFF_CMD = "systemctl poweroff --check-inhibitors=no"
SCOPE_PATTERN = "df-verify-*.scope"

NOTIFY_TIMEOUT_SECS = 15
SYSTEMCTL_TIMEOUT_SECS = 30
POWEROFF_TIMEOUT_SECS = 60

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
                raise ThresholdError(name, value)


class ThresholdError(ValueError):
    def __init__(self, field: str, value: int) -> None:
        super().__init__(f"{field} must be within 0..100, got {value}")
        self.field = field


class BatteryReadError(Exception):
    pass


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class GuardConfig:
    power_supply_dir: Path
    thresholds: Thresholds
    scope_user: str
    poweroff_cmd: tuple[str, ...]


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


_THRESHOLD_ENV = (
    ("floor_percent", "BATTERY_GUARD_FLOOR_PERCENT", DEFAULT_FLOOR_PERCENT),
    ("warn_margin_percent", "BATTERY_GUARD_WARN_MARGIN_PERCENT", DEFAULT_WARN_MARGIN_PERCENT),
)


def _env_int(environ: Mapping[str, str], var: str, default: int) -> int:
    raw = environ.get(var)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{var}={raw}: not an integer") from None


def _thresholds_from_env(environ: Mapping[str, str]) -> Thresholds:
    values = {field: _env_int(environ, var, default) for field, var, default in _THRESHOLD_ENV}
    try:
        return Thresholds(**values)
    except ThresholdError as exc:
        var = next(var for field, var, _ in _THRESHOLD_ENV if field == exc.field)
        raise ConfigError(f"{var}={environ[var]}: {exc}") from None


def _command_from_env(environ: Mapping[str, str], var: str, default: str) -> tuple[str, ...]:
    raw = environ.get(var, default)
    try:
        argv = tuple(shlex.split(raw))
    except ValueError as exc:
        raise ConfigError(f"{var}={raw}: {exc}") from None
    if not argv:
        raise ConfigError(f"{var}={raw}: empty command")
    return argv


def load_config(environ: Mapping[str, str]) -> GuardConfig:
    return GuardConfig(
        power_supply_dir=Path(
            environ.get("BATTERY_GUARD_POWER_SUPPLY_DIR", str(DEFAULT_POWER_SUPPLY_DIR)),
        ),
        thresholds=_thresholds_from_env(environ),
        scope_user=environ.get("BATTERY_GUARD_SCOPE_USER", DEFAULT_SCOPE_USER),
        poweroff_cmd=_command_from_env(
            environ, "BATTERY_GUARD_POWEROFF_CMD", DEFAULT_POWEROFF_CMD,
        ),
    )


def _say(text: str) -> None:
    print(f"battery-guard: {text}", flush=True)


def _complain(text: str) -> None:
    print(f"battery-guard: {text}", file=sys.stderr, flush=True)


def _run(cmd: Sequence[str], timeout: float) -> bool:
    shown = " ".join(cmd)
    try:
        completed = subprocess.run(cmd, stdin=subprocess.DEVNULL, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        _complain(f"{shown} failed: {exc}")
        return False
    if completed.returncode != 0:
        _complain(f"{shown} failed (rc={completed.returncode})")
        return False
    return True


def _user_manager(user: str) -> list[str]:
    return ["--user", f"--machine={user}@.host"]


def _notify(user: str, message: str) -> None:
    _run(
        [
            "systemd-run", *_user_manager(user), "--collect", "--quiet",
            "notify-send", "--urgency=critical", "Battery guard", message,
        ],
        NOTIFY_TIMEOUT_SECS,
    )


def _stop_verify_scopes(user: str) -> None:
    systemctl = ["systemctl", *_user_manager(user)]
    _run([*systemctl, "kill", "--signal=SIGKILL", SCOPE_PATTERN], SYSTEMCTL_TIMEOUT_SECS)
    _run([*systemctl, "stop", SCOPE_PATTERN], SYSTEMCTL_TIMEOUT_SECS)


def _warn(config: GuardConfig, readings: Sequence[BatteryReading]) -> None:
    floor = config.thresholds.floor_percent
    message = (
        f"{describe(readings)} (floor {floor}%): connect a stronger charger; "
        f"poweroff at {floor}%"
    )
    _say(f"WARNING {message}")
    _notify(config.scope_user, message)


def _poweroff(config: GuardConfig, readings: Sequence[BatteryReading]) -> int:
    message = (
        f"{describe(readings)} <= floor {config.thresholds.floor_percent}%: "
        f"stopping {SCOPE_PATTERN} for {config.scope_user}, then powering off"
    )
    _say(f"POWEROFF {message}")
    _notify(config.scope_user, message)
    _stop_verify_scopes(config.scope_user)
    if not _run(config.poweroff_cmd, POWEROFF_TIMEOUT_SECS):
        _complain("ERROR poweroff request failed")
        return 1
    _say("poweroff requested")
    return 0


def _dry_run_report(
    config: GuardConfig, readings: Sequence[BatteryReading], verdict: Verdict,
) -> str:
    reading = describe(readings) if readings else f"no battery under {config.power_supply_dir}"
    thresholds = config.thresholds
    return (
        f"{reading} (floor {thresholds.floor_percent}%, "
        f"warn margin {thresholds.warn_margin_percent}%): {verdict.name}"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the reading, thresholds and verdict; act on nothing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = load_config(os.environ)
    except ConfigError as exc:
        _complain(f"ERROR {exc}")
        return 2
    try:
        readings = read_batteries(config.power_supply_dir)
    except BatteryReadError as exc:
        _complain(f"ERROR cannot read battery: {exc}")
        return 1
    verdict = decide(readings, config.thresholds)
    if args.dry_run:
        _say(_dry_run_report(config, readings, verdict))
        return 0
    if verdict is Verdict.WARN:
        _warn(config, readings)
    elif verdict is Verdict.POWEROFF:
        return _poweroff(config, readings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
