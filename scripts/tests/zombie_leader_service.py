"""A service whose main thread exits while a worker thread keeps it, and its port, alive.

Usage: zombie_leader_service.py <ready-fifo> <release-file> <lock-dir>

It flocks its instance_lock under <lock-dir>, listens on a loopback port,
starts a worker thread, then ends only its main thread with the raw exit
syscall. The thread-group leader then reads Z in /proc while the process still
holds the listening socket and the lock: the state a multithreaded service
passes through on its way out. The lock's descriptor is never closed, so the
lock is released only when the last thread exits and the descriptor table goes.
Once the leader reads Z the worker writes the port into <ready-fifo>, which
blocks until the caller reads it. The worker returns, ending the process, when
<release-file> exists.
"""
from __future__ import annotations

import ctypes
import fcntl
import os
import socket
import sys
import threading
import time
from pathlib import Path

from shared.testing_own_time import state_letter

SYS_EXIT = {"x86_64": 60, "aarch64": 93}
POLL_SECS = 0.01


def instance_lock(lock_dir: Path, pid: int) -> Path:
    """The file the instance with this pid holds flocked until its last thread exits."""
    return lock_dir / str(pid)


def leader_state(pid: int) -> str:
    return state_letter(Path(f"/proc/{pid}/stat").read_text())


def serve(listener: socket.socket, ready: Path, release: Path) -> None:
    while leader_state(os.getpid()) != "Z":
        time.sleep(POLL_SECS)
    ready.write_text(str(listener.getsockname()[1]))
    while not release.exists():
        time.sleep(POLL_SECS)


def main(ready: Path, release: Path, lock_dir: Path) -> None:
    lock = os.open(instance_lock(lock_dir, os.getpid()), os.O_WRONLY | os.O_CREAT, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    threading.Thread(target=serve, args=(listener, ready, release)).start()
    ctypes.CDLL(None).syscall(SYS_EXIT[os.uname().machine], 0)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
