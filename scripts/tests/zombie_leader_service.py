"""A service whose main thread exits while a worker thread keeps it, and its port, alive.

Usage: zombie_leader_service.py <ready-file> <release-file>

It listens on a loopback port, starts a worker thread, then ends only its main
thread with the raw exit syscall. The thread-group leader then reads Z in
/proc while the process still holds the listening socket: the state a
multithreaded service passes through on its way out. Once the leader reads Z
the worker writes the port to <ready-file>, and it returns, ending the
process, when <release-file> exists.
"""
from __future__ import annotations

import ctypes
import os
import socket
import sys
import threading
import time
from pathlib import Path

from shared.safe_io import atomic_write_text
from shared.testing_own_time import state_letter

SYS_EXIT = {"x86_64": 60, "aarch64": 93}
POLL_SECS = 0.01


def leader_state(pid: int) -> str:
    return state_letter(Path(f"/proc/{pid}/stat").read_text())


def serve(listener: socket.socket, ready: Path, release: Path) -> None:
    while leader_state(os.getpid()) != "Z":
        time.sleep(POLL_SECS)
    atomic_write_text(ready, str(listener.getsockname()[1]))
    while not release.exists():
        time.sleep(POLL_SECS)


def main(ready: Path, release: Path) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    threading.Thread(target=serve, args=(listener, ready, release)).start()
    ctypes.CDLL(None).syscall(SYS_EXIT[os.uname().machine], 0)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
