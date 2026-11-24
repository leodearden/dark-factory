"""fake_user_systemctl.py, driven through its CLI, against a unit whose main thread exits first.

zombie_leader_service.py's thread-group leader reads Z in /proc while a worker
thread still holds its listening port and its instance lock. Real systemd
counts such a unit as running until its cgroup is empty, so the fake must too:
a restart that starts the new instance before every thread of the old one has
gone races it for the port.

Past its stop deadline, a stop or restart of a unit still running must fail
naming it, never start a second instance beside it.
"""
from __future__ import annotations

import errno
import fcntl
import os
import shlex
import socket
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pytest
from cli_subprocess_timeout import cli_timeout_from_env
from fake_user_systemctl_state import kill_all_units, pid_file, unit_pid
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from zombie_leader_service import SYS_EXIT, instance_lock

pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.skipif(
        os.uname().machine not in SYS_EXIT,
        reason=f"no SYS_exit number known for {os.uname().machine}",
    ),
]

HERE = Path(__file__).resolve().parent
FAKE_SYSTEMCTL = HERE / "fake_user_systemctl.py"
SERVICE = HERE / "zombie_leader_service.py"
CLI_TIMEOUT = cli_timeout_from_env("FAKE_USER_SYSTEMCTL_TEST_TIMEOUT")
UNIT = "zombie-leader.service"
INACTIVE = 3


def bind_errno(port: int) -> int:
    """0 if a fresh socket can bind 127.0.0.1:port, else the errno bind() raised."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as refused:
            assert refused.errno is not None, refused
            return refused.errno
    return 0


@dataclass(frozen=True)
class UnitWorld:
    env: dict[str, str] = field(repr=False)
    state: Path
    ready: Path
    release: Path
    locks: Path

    def systemctl(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-I", str(FAKE_SYSTEMCTL), "--user", *args],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT,
        )

    def pid(self) -> int:
        return unit_pid(self.state, UNIT)

    @cached_property
    def port(self) -> int:
        """The port the first instance started listens on; blocks until its leader reads Z."""
        return int(self.ready.read_text())

    def any_thread_runs(self, pid: int) -> bool:
        """Whether instance pid still holds its lock, which only its last thread's exit frees."""
        lock = os.open(instance_lock(self.locks, pid), os.O_RDONLY)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        finally:
            os.close(lock)
        return False

    def wait_until_inactive(self) -> None:
        """Return once is-active reports the unit inactive; each check is a CLI run of its own."""
        while self.systemctl("is-active", UNIT).returncode != INACTIVE:
            continue

    def kill_all(self) -> None:
        self.release.touch()
        kill_all_units(self.state)


@pytest.fixture
def world(tmp_path: Path) -> Iterator[UnitWorld]:
    """A scratch HOME holding zombie-leader.service's unit file, and the fake's state."""
    home = tmp_path / "home"
    unit_dir = home / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    ready, release, locks = tmp_path / "ready", tmp_path / "release", tmp_path / "locks"
    os.mkfifo(ready)
    locks.mkdir()
    command = [sys.executable, "-I", str(SERVICE), str(ready), str(release), str(locks)]
    (unit_dir / UNIT).write_text(f"ExecStart={shlex.join(command)}\n")
    world = UnitWorld(
        env={**os.environ, "HOME": str(home), "FAKE_SYSTEMD_STATE": str(tmp_path / "state")},
        state=tmp_path / "state",
        ready=ready,
        release=release,
        locks=locks,
    )
    world.state.mkdir()
    try:
        yield world
    finally:
        world.kill_all()


@pytest.fixture
def started(world: UnitWorld) -> UnitWorld:
    """zombie-leader.service started through the fake, its leader already Z."""
    proc = world.systemctl("start", UNIT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert world.port
    return world


@pytest.fixture
def ungrouped_child() -> Iterator[subprocess.Popen[bytes]]:
    """A live process that leads no process group, so killpg on its pid signals nothing."""
    child = subprocess.Popen(["sleep", "infinity"])
    try:
        yield child
    finally:
        child.kill()
        child.wait()


def test_a_unit_whose_main_thread_exited_is_active_while_a_thread_runs(
    started: UnitWorld,
) -> None:
    is_active = started.systemctl("is-active", UNIT)
    assert (is_active.returncode, is_active.stdout.strip()) == (0, "active")
    main_pid = started.systemctl("show", "-p", "MainPID", "--value", UNIT)
    assert main_pid.stdout.strip() == str(started.pid())
    assert bind_errno(started.port) == errno.EADDRINUSE

    started.release.touch()
    started.wait_until_inactive()


def test_restart_leaves_no_thread_of_the_old_instance(started: UnitWorld) -> None:
    old_pid = started.pid()

    proc = started.systemctl("restart", UNIT)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not started.any_thread_runs(old_pid), f"a thread of the old instance (pid {old_pid}) runs"
    assert started.pid() != old_pid


@pytest.mark.parametrize("command", ["restart", "stop"])
def test_a_unit_still_running_at_the_stop_deadline_fails_naming_it(
    world: UnitWorld, ungrouped_child: subprocess.Popen[bytes], command: str
) -> None:
    pid_file(world.state, UNIT).write_text(str(ungrouped_child.pid))

    proc = world.systemctl(command, UNIT, FAKE_SYSTEMD_STOP_DEADLINE_SECS="0")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert f"{UNIT} (pid {ungrouped_child.pid}) still running" in proc.stderr
    assert unit_pid(world.state, UNIT) == ungrouped_child.pid
    assert ungrouped_child.poll() is None
