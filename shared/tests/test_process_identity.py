"""shared.process_identity against real child processes and fake proc roots.

Boundary rows 12 (liveness) and 26 (unreadable /proc) of
plans/deferral-flip-condition-prd.md §6. Fake proc roots go through the public
``proc_root`` keyword; nothing private is patched.
"""

from __future__ import annotations

import dataclasses
import os
import signal
import socket
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from shared.process_identity import (
    Liveness,
    ProcessIdentity,
    capture,
    command_name,
    liveness,
)
from test_pure_stdlib_leaves import run_python_child

_LIVE_BOOT_ID = Path('/proc/sys/kernel/random/boot_id')
_FORGED_BOOT_ID = '00000000-0000-0000-0000-000000000000'
_COMM_WITH_PARENS = 'a) b (c d'
_RENAME_THEN_SLEEP = (
    'import sys, time\n'
    'sys.stdin.readline()\n'
    f'open("/proc/self/comm", "w").write({_COMM_WITH_PARENS!r})\n'
    'print("ready", flush=True)\n'
    'time.sleep(600)\n'
)


@pytest.fixture
def spawn() -> Iterator[Callable[..., subprocess.Popen]]:
    """Start real children; kill and reap every one at teardown, even if already reaped."""
    children: list[subprocess.Popen] = []

    def _spawn(argv: list[str], **popen_kwargs) -> subprocess.Popen:
        child = subprocess.Popen(argv, **popen_kwargs)
        children.append(child)
        return child

    yield _spawn
    for child in children:
        child.kill()
        child.wait()
        for stream in (child.stdin, child.stdout):
            if stream is not None:
                stream.close()


@pytest.fixture
def live_child(spawn) -> subprocess.Popen:
    return spawn(['sleep', '600'])


def _captured(pid: int) -> ProcessIdentity:
    identity = capture(pid)
    assert identity is not None, f'pid {pid} should be capturable while it runs'
    return identity


def _kill_and_reap(child: subprocess.Popen) -> None:
    child.kill()
    child.wait()


def _fake_proc_root(root: Path, *, boot_id: str) -> Path:
    (root / '1').mkdir(parents=True)
    boot_id_path = root / 'sys' / 'kernel' / 'random' / 'boot_id'
    boot_id_path.parent.mkdir(parents=True)
    boot_id_path.write_text(f'{boot_id}\n')
    return root


def test_capture_reads_the_live_identity(live_child):
    identity = capture(live_child.pid)

    assert isinstance(identity, ProcessIdentity)
    assert identity.pid == live_child.pid
    assert identity.host == socket.gethostname()
    assert identity.boot_id == _LIVE_BOOT_ID.read_text().strip()
    assert capture(live_child.pid) == identity


def test_identity_is_frozen(live_child):
    identity = _captured(live_child.pid)

    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.pid = 1  # type: ignore[misc]


def test_running_process_is_alive(live_child):
    assert liveness(_captured(live_child.pid)) is Liveness.ALIVE
    assert liveness(_captured(os.getpid())) is Liveness.ALIVE


def test_exited_and_reaped_process_is_dead(live_child):
    identity = _captured(live_child.pid)

    _kill_and_reap(live_child)

    assert liveness(identity) is Liveness.DEAD
    assert capture(live_child.pid) is None


def test_reboot_is_death(live_child):
    rebooted = dataclasses.replace(_captured(live_child.pid), boot_id=_FORGED_BOOT_ID)

    assert liveness(rebooted) is Liveness.DEAD


def test_reused_pid_is_death(live_child):
    identity = _captured(live_child.pid)
    reused = dataclasses.replace(identity, start_ticks=identity.start_ticks + 1)

    assert liveness(reused) is Liveness.DEAD


def test_other_host_is_decided_before_any_proc_read(live_child, tmp_path):
    elsewhere = dataclasses.replace(
        _captured(live_child.pid), host='not-' + socket.gethostname()
    )

    assert liveness(elsewhere) is Liveness.OTHER_HOST
    assert liveness(elsewhere, proc_root=tmp_path / 'absent') is Liveness.OTHER_HOST


def test_unreaped_zombie_is_dead(live_child):
    identity = _captured(live_child.pid)

    os.kill(live_child.pid, signal.SIGKILL)
    os.waitid(os.P_PID, live_child.pid, os.WEXITED | os.WNOWAIT)

    assert liveness(identity) is Liveness.DEAD
    assert capture(live_child.pid) is None


def test_start_ticks_are_read_after_the_last_paren_of_comm(spawn):
    child = spawn(
        [sys.executable, '-c', _RENAME_THEN_SLEEP],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdin is not None and child.stdout is not None
    before_rename = _captured(child.pid)

    child.stdin.write('go\n')
    child.stdin.flush()
    assert child.stdout.readline().strip() == 'ready'

    assert command_name(child.pid) == _COMM_WITH_PARENS
    assert _captured(child.pid) == before_rename


def test_command_name_of_a_running_and_an_exited_process(live_child):
    assert command_name(live_child.pid) == 'sleep'

    _kill_and_reap(live_child)

    assert command_name(live_child.pid) is None


def test_absent_pid_with_visible_pid_1_is_positive_evidence_of_death(live_child, tmp_path):
    identity = _captured(live_child.pid)
    fake = _fake_proc_root(tmp_path / 'proc', boot_id=identity.boot_id)

    assert liveness(identity, proc_root=fake) is Liveness.DEAD
    assert capture(identity.pid, proc_root=fake) is None
    assert command_name(identity.pid, proc_root=fake) is None


def test_printed_liveness_of_the_current_process_reads_alive():
    printed = run_python_child(
        'import json, os\n'
        'from shared.process_identity import capture, liveness\n'
        'print(json.dumps(str(liveness(capture(os.getpid())))))\n'
    )

    assert printed == 'ALIVE'
