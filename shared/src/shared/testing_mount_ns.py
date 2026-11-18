"""Boundary row 19 of ``plans/canonical-build-path-shared-cache-prd.md`` §6, as live processes.

:func:`namespaced_holder` starts a process in its own mount namespace that sees a lane at a
bind-mount point, references it only by that path, and whose visible parent has been killed.
:func:`unreadable_foreign_namespace_process` starts a same-user process in a foreign mount
namespace whose root the scanner cannot read. This is the one fixture every dark-factory C11
implementation's tests use (``shared/tests`` and ``orchestrator/tests``), so it ships in
``shared`` like :mod:`shared.testing_stdin`. Each context manager asserts its scenario's
premise before yielding, so a pass can never come from an untranslated match, and calls
``pytest.skip`` naming what failed where the host cannot build the scenario. pytest is
imported only to skip, so the module imports cleanly outside a test run, like
:mod:`shared.testing_cli_runner`.
"""

from __future__ import annotations

import contextlib
import functools
import os
import select
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import NoReturn

from shared.proc_group import process_group_terminated

#: Absolute, because the AppArmor profile that lets bwrap create namespaces attaches by path.
BWRAP = Path('/usr/bin/bwrap')

#: Backstops against broken infrastructure, not timing assertions.
_READY_BACKSTOP_S = 60.0
_TERMINATION_BACKSTOP_S = 30.0
_POLL_S = 0.05

HELD_FILE_SIZE = 4096


class Hold(Enum):
    """What the namespaced process holds on the lane, always through the mount-point path.

    ``CWD`` sleeps with its cwd at ``<mount_point>/sub``; ``CWD_BUSY`` spins there.
    ``OPEN_FILE`` keeps an fd open on ``<mount_point>/sub/held.bin`` with cwd ``/``.
    ``MAPPING`` maps that file and closes every fd on it, with cwd ``/``.
    """

    CWD = auto()
    CWD_BUSY = auto()
    OPEN_FILE = auto()
    MAPPING = auto()

    @property
    def in_lane_cwd(self) -> bool:
        return self in (Hold.CWD, Hold.CWD_BUSY)


@dataclass(frozen=True)
class HeldProcess:
    pid: int
    pgid: int


_READY = "import os\nprint('ready', os.getpid(), flush=True)\n"
_SLEEP = 'import time\ntime.sleep(600)\n'
_SPIN = 'import time\nend = time.monotonic() + 600\nwhile time.monotonic() < end:\n    pass\n'
_OPEN = "import sys\nheld = open(sys.argv[1], 'rb')\n"

#: Maps the file at ``sys.argv[1]``, then closes every fd on it. ``mmap.mmap`` keeps a private
#: dup of the fd it maps, so the mapping is then the process's only reference to the file.
MAP_THEN_CLOSE_FDS = """
import mmap, os, sys
path = sys.argv[1]
with open(path, 'rb') as mapped:
    mapping = mmap.mmap(mapped.fileno(), 0, prot=mmap.PROT_READ)
for fd in os.listdir('/proc/self/fd'):
    try:
        if os.readlink(f'/proc/self/fd/{fd}') == path:
            os.close(int(fd))
    except OSError:
        pass
"""

_HOLDER_SCRIPTS = {
    Hold.CWD: _READY + _SLEEP,
    Hold.CWD_BUSY: _READY + _SPIN,
    Hold.OPEN_FILE: _OPEN + _READY + _SLEEP,
    Hold.MAPPING: MAP_THEN_CLOSE_FDS + _READY + _SLEEP,
}

_UNSHARED_SLEEPER = """
import ctypes, os, sys, time
try:
    os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNS)
except (AttributeError, OSError) as exc:
    print('refused', repr(exc), flush=True)
    sys.exit(0)
ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)
print('ready', os.getpid(), flush=True)
time.sleep(600)
"""


@contextmanager
def namespaced_holder(lane: Path, mount_point: Path, hold: Hold) -> Iterator[HeldProcess]:
    """A process holding *hold* on *lane* only through *mount_point*, its visible parent killed.

    Creates ``lane/sub/held.bin`` and the empty *mount_point*, then runs the holder under
    ``bwrap --bind lane mount_point`` in a new session. The outer bwrap is SIGKILLed and
    reaped before yielding; the holder keeps the outer's pgid. On exit the group is killed
    and polled until it has terminated.
    """
    _skip_unless_bwrap_creates_mount_namespaces()
    held = mount_point / 'sub' / 'held.bin'
    (lane / 'sub').mkdir(parents=True, exist_ok=True)
    (lane / 'sub' / 'held.bin').write_bytes(b'\0' * HELD_FILE_SIZE)
    mount_point.mkdir(parents=True, exist_ok=True)
    chdir = mount_point / 'sub' if hold.in_lane_cwd else Path('/')
    outer = subprocess.Popen(
        [
            str(BWRAP), '--dev-bind', '/', '/', '--bind', str(lane), str(mount_point),
            '--chdir', str(chdir), '--', sys.executable, '-c', _HOLDER_SCRIPTS[hold], str(held),
        ],
        cwd='/',
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        pid = _ready_pid(outer, f'bwrap holder ({hold.name})')
        _assert_held_only_through_mount_point(pid, lane, mount_point, hold)
        outer.send_signal(signal.SIGKILL)
        outer.wait()
        yield HeldProcess(pid=pid, pgid=outer.pid)
    finally:
        _kill_group_and_await_termination(outer)


@contextmanager
def unreadable_foreign_namespace_process(cwd: Path) -> Iterator[HeldProcess]:
    """A same-user process in a foreign mount namespace whose root cannot be read.

    The child unshares its user and mount namespaces in-process, with no exec, then makes
    itself non-dumpable, so ``/proc/<pid>/root``, ``cwd``, ``fd`` and ``maps`` refuse the
    scanner while ``mountinfo`` and ``ns/mnt`` stay readable.
    """
    child = subprocess.Popen(
        [sys.executable, '-c', _UNSHARED_SLEEPER],
        cwd=str(cwd),
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = _ready_line(child, 'unshared sleeper')
        if line.startswith('refused'):
            _skip(f'os.unshare(CLONE_NEWUSER | CLONE_NEWNS) refused: {line.strip()}')
        pid = _parse_ready(line, 'unshared sleeper', child)
        try:
            os.stat(f'/proc/{pid}/root')
        except PermissionError:
            pass
        else:
            _skip(
                f'/proc/{pid}/root of a non-dumpable foreign-namespace process is readable '
                f'here (CAP_SYS_PTRACE?), so the unreadable-root scenario cannot be built'
            )
        yield HeldProcess(pid=pid, pgid=child.pid)
    finally:
        _kill_group_and_await_termination(child)


def _skip(reason: str) -> NoReturn:
    import pytest

    pytest.skip(reason)


def _skip_unless_bwrap_creates_mount_namespaces() -> None:
    reason = _why_bwrap_cannot_create_mount_namespaces()
    if reason is not None:
        _skip(reason)


@functools.cache
def _why_bwrap_cannot_create_mount_namespaces() -> str | None:
    """Probed once per process: the answer does not change within a test session."""
    if not BWRAP.is_file():
        return f'{BWRAP} is absent, so no mount namespace can be built'
    probe = subprocess.run(
        [str(BWRAP), '--dev-bind', '/', '/', '--', sys.executable, '-c', ''],
        cwd='/',
        capture_output=True,
        text=True,
        timeout=_READY_BACKSTOP_S,
    )
    if probe.returncode != 0:
        return (
            f'{BWRAP} cannot create a mount namespace here (exit {probe.returncode}): '
            f'{probe.stderr.strip()!r}; e.g. a landlocked session (PRD S9)'
        )
    return None


def _ready_line(child: subprocess.Popen[str], what: str) -> str:
    assert child.stdout is not None
    readable, _, _ = select.select([child.stdout], [], [], _READY_BACKSTOP_S)
    if not readable:
        raise AssertionError(f'{what} printed nothing within {_READY_BACKSTOP_S}s')
    return child.stdout.readline()


def _parse_ready(line: str, what: str, child: subprocess.Popen[str]) -> int:
    word, _, pid = line.partition(' ')
    if word != 'ready' or not pid.strip().isdigit():
        assert child.stderr is not None
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        raise AssertionError(
            f'{what} did not become ready: stdout {line!r}, stderr {child.stderr.read()!r}'
        )
    return int(pid)


def _ready_pid(child: subprocess.Popen[str], what: str) -> int:
    return _parse_ready(_ready_line(child, what), what, child)


def _at_or_under(path: str, root: Path) -> bool:
    return path.startswith('/') and Path(path).is_relative_to(root)


def _fd_targets(pid: int) -> list[str]:
    fd_dir = Path(f'/proc/{pid}/fd')
    return [os.readlink(fd) for fd in fd_dir.iterdir()]


def _mapped_paths(pid: int) -> list[str]:
    lines = os.fsdecode(Path(f'/proc/{pid}/maps').read_bytes()).split('\n')
    return [parts[5] for parts in (line.split(maxsplit=5) for line in lines) if len(parts) == 6]


def _assert_held_only_through_mount_point(
    pid: int, lane: Path, mount_point: Path, hold: Hold
) -> None:
    """The host must see the hold at the mount-point path and nowhere at the lane's own path."""
    cwd = os.readlink(f'/proc/{pid}/cwd')
    fds = _fd_targets(pid)
    maps = _mapped_paths(pid)
    seen = f'cwd {cwd!r}, fds {fds!r}'
    assert not any(_at_or_under(path, lane) for path in (cwd, *fds, *maps)), (
        f'holder {pid} names the lane {lane} by its own path, so a scan could pass untranslated: '
        f'{seen}'
    )
    if hold.in_lane_cwd:
        assert _at_or_under(cwd, mount_point), f'holder {pid} cwd is not under {mount_point}: {seen}'
        return
    assert not _at_or_under(cwd, mount_point), f'holder {pid} cwd is in the lane: {seen}'
    held = str(mount_point / 'sub' / 'held.bin')
    if hold is Hold.OPEN_FILE:
        assert held in fds, f'holder {pid} has no fd on {held}: {seen}'
    else:
        assert held in maps, f'holder {pid} does not map {held}: {seen}'
        assert not any(_at_or_under(fd, mount_point) for fd in fds), (
            f'holder {pid} still has an fd in the lane: {seen}'
        )


def _kill_group_and_await_termination(leader: subprocess.Popen[str]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(leader.pid, signal.SIGKILL)
    leader.wait()
    for stream in (leader.stdout, leader.stderr):
        if stream is not None:
            stream.close()
    deadline = time.monotonic() + _TERMINATION_BACKSTOP_S
    while not process_group_terminated(leader.pid):
        if time.monotonic() >= deadline:
            raise AssertionError(
                f'process group {leader.pid} still running {_TERMINATION_BACKSTOP_S}s after SIGKILL'
            )
        time.sleep(_POLL_S)
