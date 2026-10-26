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
from _child_processes import spawned_children
from test_pure_stdlib_leaves import run_python_child

from shared.process_identity import (
    Liveness,
    ProcessIdentity,
    ProcessIdentityUnreadable,
    capture,
    command_name,
    liveness,
)

_LIVE_BOOT_ID = Path('/proc/sys/kernel/random/boot_id')
_FORGED_BOOT_ID = '00000000-0000-0000-0000-000000000000'
_COMM_WITH_PARENS = 'a) b (c d'
_PR_SET_NAME = 15
# prctl rather than a write to /proc/self/comm: the agent sandbox denies that write.
_RENAME_THEN_SLEEP = (
    'import ctypes, sys, time\n'
    'sys.stdin.readline()\n'
    f'ctypes.CDLL(None).prctl({_PR_SET_NAME}, {_COMM_WITH_PARENS.encode()!r}, 0, 0, 0)\n'
    'print("ready", flush=True)\n'
    'time.sleep(600)\n'
)


@pytest.fixture
def spawn() -> Iterator[Callable[..., subprocess.Popen]]:
    with spawned_children() as spawn_child:
        yield spawn_child


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


def _fake_proc_root(root: Path, *, boot_id: str | None) -> Path:
    (root / '1').mkdir(parents=True)
    if boot_id is not None:
        boot_id_path = root / 'sys' / 'kernel' / 'random' / 'boot_id'
        boot_id_path.parent.mkdir(parents=True)
        boot_id_path.write_text(f'{boot_id}\n')
    return root


def _write_pid_files(root: Path, pid: int, *, stat: str) -> list[Path]:
    pid_dir = root / str(pid)
    pid_dir.mkdir()
    (pid_dir / 'stat').write_text(stat)
    (pid_dir / 'comm').write_text('sleep\n')
    return [pid_dir / 'stat', pid_dir / 'comm']


def test_capture_reads_the_live_identity(live_child):
    identity = capture(live_child.pid)

    assert isinstance(identity, ProcessIdentity)
    assert identity.pid == live_child.pid
    assert identity.host == socket.gethostname()
    assert identity.boot_id == _LIVE_BOOT_ID.read_text().strip()
    assert capture(live_child.pid) == identity


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


def _live_stat_line(pid: int) -> str:
    return (Path('/proc') / str(pid) / 'stat').read_text()


def _with_start_ticks(stat: str, start_ticks: str) -> str:
    head, _, tail = stat.rpartition(')')
    fields = tail.split()
    fields[19] = start_ticks
    return f'{head}) {" ".join(fields)}\n'


def _proc_not_mounted(root: Path, identity: ProcessIdentity) -> Path:
    return root / 'absent'


def _every_pid_invisible(root: Path, identity: ProcessIdentity) -> Path:
    root.mkdir()
    return root


def _pid_files_mode_000(root: Path, identity: ProcessIdentity) -> Path:
    fake = _fake_proc_root(root, boot_id=identity.boot_id)
    for path in _write_pid_files(fake, identity.pid, stat=_live_stat_line(identity.pid)):
        path.chmod(0)
    return fake


def _malformed_stat(stat_of: Callable[[int], str]) -> Callable[[Path, ProcessIdentity], Path]:
    def build(root: Path, identity: ProcessIdentity) -> Path:
        fake = _fake_proc_root(root, boot_id=identity.boot_id)
        _write_pid_files(fake, identity.pid, stat=stat_of(identity.pid))
        return fake

    return build


def _boot_id_missing(root: Path, identity: ProcessIdentity) -> Path:
    fake = _fake_proc_root(root, boot_id=None)
    _write_pid_files(fake, identity.pid, stat=_live_stat_line(identity.pid))
    return fake


_ALL_READS = ('capture', 'liveness', 'command_name')
_STAT_READS = ('capture', 'liveness')
_UNREADABLE_PROC_ROOTS = [
    pytest.param(_proc_not_mounted, _ALL_READS, id='proc-not-mounted'),
    pytest.param(_every_pid_invisible, _ALL_READS, id='hidepid-invisible'),
    pytest.param(
        _pid_files_mode_000,
        _ALL_READS,
        id='eacces',
        marks=pytest.mark.skipif(os.geteuid() == 0, reason='root ignores file modes'),
    ),
    pytest.param(
        _malformed_stat(lambda pid: _live_stat_line(pid).replace(')', ' ')),
        _STAT_READS,
        id='stat-without-paren',
    ),
    pytest.param(
        _malformed_stat(lambda pid: f'{pid} (sleep) S 1 2 3\n'),
        _STAT_READS,
        id='stat-too-few-fields',
    ),
    pytest.param(
        _malformed_stat(lambda pid: _with_start_ticks(_live_stat_line(pid), 'soon')),
        _STAT_READS,
        id='stat-start-ticks-not-a-number',
    ),
    pytest.param(_boot_id_missing, _STAT_READS, id='boot-id-missing'),
]


def _read(name: str, identity: ProcessIdentity, proc_root: Path) -> object:
    if name == 'capture':
        return capture(identity.pid, proc_root=proc_root)
    if name == 'liveness':
        return liveness(identity, proc_root=proc_root)
    return command_name(identity.pid, proc_root=proc_root)


@pytest.fixture
def removable_tmp_path(tmp_path) -> Iterator[Path]:
    """tmp_path whose mode-000 files are made removable again at teardown."""
    yield tmp_path
    for path in tmp_path.rglob('*'):
        path.chmod(0o700)


@pytest.mark.parametrize(('build_proc_root', 'reads'), _UNREADABLE_PROC_ROOTS)
def test_unreadable_proc_raises_and_never_reads_as_dead(
    build_proc_root, reads, live_child, removable_tmp_path
):
    identity = _captured(live_child.pid)
    proc_root = build_proc_root(removable_tmp_path / 'proc', identity)

    for name in reads:
        with pytest.raises(ProcessIdentityUnreadable) as raised:
            answer = _read(name, identity, proc_root)
            pytest.fail(f'{name} answered {answer!r} from an unreadable proc root')
        assert str(raised.value.path) in str(raised.value), name
        assert raised.value.path.is_relative_to(proc_root), name


def test_unreadable_is_not_an_os_error():
    assert not issubclass(ProcessIdentityUnreadable, OSError)
