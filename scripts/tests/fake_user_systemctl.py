"""A stand-in for `systemctl --user`, for installer tests in a scratch HOME.

It really runs a service: `start` and `restart` read the unit file the
installer wrote under $HOME/.config/systemd/user, apply its Environment= and
EnvironmentFile= lines, and spawn its one-line ExecStart= in a new session.
Every invocation's argv is appended as a JSON list to
$FAKE_SYSTEMD_STATE/systemctl.log; a service's pid is kept in
$FAKE_SYSTEMD_STATE/<unit>.pid and an enabled unit leaves <unit>.enabled.
A service counts as running until every thread of its process has exited, as
systemd's stop waits for the unit's cgroup to empty.
A timer is active exactly when it is enabled.
"""
from __future__ import annotations

import contextlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

STATE = Path(os.environ["FAKE_SYSTEMD_STATE"])
UNIT_DIR = Path(os.environ["HOME"]) / ".config" / "systemd" / "user"
EXITED_STATES = {"Z", "X"}
STOP_DEADLINE_SECS = 5


def task_state(stat: Path) -> str:
    """The state letter in one /proc stat file; X once the task has gone."""
    try:
        return stat.read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return "X"


def live_pid(unit: str) -> int:
    """The unit's pid while any thread of its process still runs, else 0."""
    try:
        pid = int((STATE / f"{unit}.pid").read_text())
        tasks = list(Path(f"/proc/{pid}/task").iterdir())
    except (OSError, ValueError):
        return 0
    running = any(task_state(task / "stat") not in EXITED_STATES for task in tasks)
    return pid if running else 0


def is_enabled(unit: str) -> bool:
    return (STATE / f"{unit}.enabled").exists()


def stop(unit: str) -> bool:
    """SIGKILL the unit's process group; True once every thread of it has exited."""
    pid = live_pid(unit)
    if not pid:
        return True
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)
    deadline = time.monotonic() + STOP_DEADLINE_SECS
    while live_pid(unit) and time.monotonic() < deadline:
        time.sleep(0.05)
    if live_pid(unit):
        print(
            f"fake systemctl: {unit} (pid {pid}) still running"
            f" {STOP_DEADLINE_SECS}s after SIGKILL",
            file=sys.stderr,
        )
        return False
    return True


def start(unit: str) -> int:
    if not unit.endswith(".service") or live_pid(unit):
        return 0
    env = dict(os.environ)
    command: list[str] = []
    for line in (UNIT_DIR / unit).read_text().splitlines():
        key, _, value = line.partition("=")
        if key == "Environment":
            name, _, setting = value.partition("=")
            env[name] = setting
        elif key == "EnvironmentFile":
            for env_line in Path(value.lstrip("-")).read_text().splitlines():
                name, sep, setting = env_line.partition("=")
                if sep:
                    env[name] = setting
        elif key == "ExecStart":
            command = shlex.split(value)
    if not command:
        print(f"{unit} has no ExecStart=", file=sys.stderr)
        return 1
    with (STATE / f"{unit}.log").open("ab") as out:
        proc = subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (STATE / f"{unit}.pid").write_text(str(proc.pid))
    return 0


def enable(units: list[str], now: bool) -> int:
    for unit in units:
        if not (UNIT_DIR / unit).is_file():
            print(f"Unit file {unit} does not exist.", file=sys.stderr)
            return 1
        (STATE / f"{unit}.enabled").touch()
        if now and start(unit):
            return 1
    return 0


def report(answer: bool, yes: str, no: str, quiet: bool) -> int:
    if not quiet:
        print(yes if answer else no)
    return 0 if answer else 3


def show(rest: list[str]) -> int:
    prop = rest[rest.index("-p") + 1]
    unit = [arg for arg in rest if not arg.startswith("-") and arg != prop][-1]
    if prop != "MainPID":
        print(f"fake systemctl: unsupported property {prop}", file=sys.stderr)
        return 1
    print(live_pid(unit))
    return 0


def main(argv: list[str]) -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "systemctl.log").open("a") as log:
        log.write(json.dumps(argv) + "\n")
    if argv[:1] != ["--user"]:
        print("fake systemctl: only --user is supported", file=sys.stderr)
        return 1
    command, rest = argv[1], argv[2:]
    units = [arg for arg in rest if not arg.startswith("-")]
    quiet = "--quiet" in rest or "-q" in rest
    if command == "daemon-reload":
        return 0
    if command == "enable":
        return enable(units, now="--now" in rest)
    if command == "start":
        return max((start(unit) for unit in units), default=0)
    if command == "restart":
        if not all([stop(unit) for unit in units]):
            return 1
        return max((start(unit) for unit in units), default=0)
    if command == "stop":
        return 0 if all([stop(unit) for unit in units]) else 1
    if command == "is-active":
        unit = units[-1]
        active = live_pid(unit) > 0 if unit.endswith(".service") else is_enabled(unit)
        return report(active, "active", "inactive", quiet)
    if command == "is-enabled":
        return report(is_enabled(units[-1]), "enabled", "disabled", quiet)
    if command == "show":
        return show(rest)
    print(f"fake systemctl: unsupported command {command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
