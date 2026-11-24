"""fake_user_systemctl.py, driven through its CLI, against a unit whose main thread exits first.

zombie_leader_service.py's thread-group leader reads Z in /proc while a worker
thread still holds its listening port. Real systemd counts such a unit as
running until its cgroup is empty, so the fake must too: a restart that starts
the new instance before every thread of the old one has gone races it for the
port.
"""
from __future__ import annotations

import errno
import os
import shlex
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from cli_subprocess_timeout import cli_timeout_from_env
from fake_user_systemctl_state import kill_all_units, log_file, unit_pid
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from zombie_leader_service import SYS_EXIT

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


def eventually(condition: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + CLI_TIMEOUT
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


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
    env: dict[str, str]
    state: Path
    ready: Path
    release: Path

    def systemctl(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-I", str(FAKE_SYSTEMCTL), "--user", *args],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT,
        )

    def pid(self) -> int:
        return unit_pid(self.state, UNIT)

    def unit_log(self) -> str:
        log = log_file(self.state, UNIT)
        return log.read_text(errors="replace") if log.exists() else "(no log)"

    def wait_until_ready(self) -> int:
        """The port the running instance listens on, once its leader reads Z."""
        assert eventually(self.ready.exists), (
            f"{UNIT} wrote no {self.ready} within {CLI_TIMEOUT}s; its log:\n{self.unit_log()}"
        )
        return int(self.ready.read_text())

    def kill_all(self) -> None:
        self.release.touch()
        kill_all_units(self.state)


@pytest.fixture
def started(tmp_path: Path) -> Iterator[UnitWorld]:
    """zombie-leader.service started through the fake, its leader already Z."""
    home = tmp_path / "home"
    unit_dir = home / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    ready, release = tmp_path / "ready", tmp_path / "release"
    command = [sys.executable, "-I", str(SERVICE), str(ready), str(release)]
    (unit_dir / UNIT).write_text(f"ExecStart={shlex.join(command)}\n")
    world = UnitWorld(
        env={**os.environ, "HOME": str(home), "FAKE_SYSTEMD_STATE": str(tmp_path / "state")},
        state=tmp_path / "state",
        ready=ready,
        release=release,
    )
    try:
        proc = world.systemctl("start", UNIT)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        world.wait_until_ready()
        yield world
    finally:
        world.kill_all()


def test_a_unit_whose_main_thread_exited_is_active_while_a_thread_runs(
    started: UnitWorld,
) -> None:
    port = int(started.ready.read_text())

    is_active = started.systemctl("is-active", UNIT)
    assert (is_active.returncode, is_active.stdout.strip()) == (0, "active")
    main_pid = started.systemctl("show", "-p", "MainPID", "--value", UNIT)
    assert main_pid.stdout.strip() == str(started.pid())
    assert bind_errno(port) == errno.EADDRINUSE

    started.release.touch()
    assert eventually(lambda: started.systemctl("is-active", UNIT).returncode == INACTIVE)


def test_restart_leaves_no_thread_of_the_old_instance(started: UnitWorld) -> None:
    old_port, old_pid = int(started.ready.read_text()), started.pid()
    started.ready.unlink()

    proc = started.systemctl("restart", UNIT)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert bind_errno(old_port) == 0, f"a thread of the old instance (pid {old_pid}) holds its port"
    assert started.pid() != old_pid
