"""A real process with the argv of a shared ``claude agents`` session host.

``session_identity.is_shared_session_host`` reads ``/proc/<pid>/cmdline``, so
tests drive it with a live process whose argv has the host's shape rather
than by patching the rule.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

REQUIRES_PROC = pytest.mark.skipif(
    not Path('/proc/self/cmdline').exists(), reason='host detection reads /proc'
)


@contextmanager
def running_shared_host(host_dir: Path) -> Iterator[int]:
    """Yield the pid of a live ``<python> agents --inherit-permission-mode ...`` process."""
    (host_dir / 'agents').write_text('import time\ntime.sleep(600)\n', encoding='utf-8')
    proc = subprocess.Popen(
        [sys.executable, 'agents', '--inherit-permission-mode', 'bypassPermissions'],
        cwd=host_dir,
    )
    try:
        _await_host_argv(proc.pid)
        yield proc.pid
    finally:
        proc.kill()
        proc.wait(timeout=10)


def _await_host_argv(pid: int) -> None:
    """Wait for ``/proc/<pid>/cmdline`` to show the host argv.

    Popen returns once the child's exec has closed its CLOEXEC pipe, which the
    kernel does before it publishes the new argv, so a loaded host can read an
    empty cmdline for a moment after Popen returns.
    """
    cmdline = Path(f'/proc/{pid}/cmdline')
    deadline = time.monotonic() + 10
    while cmdline.read_bytes().split(b'\0')[1:2] != [b'agents']:
        if time.monotonic() > deadline:
            pytest.fail(f'{cmdline} never showed the `agents` argv')
        time.sleep(0.01)
