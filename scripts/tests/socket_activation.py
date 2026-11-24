"""Run a server on a loopback port this process bound first, handed over by sd_listen_fds.

The port is bound and listening before the server starts and is never
released, so no other process can take it between choosing it and serving
it. The consumer is test_build_cache_store.py::private_rclone.
"""
from __future__ import annotations

import contextlib
import socket
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import IO

_SD_LISTEN_HANDOFF = 'exec 3<&0 0</dev/null; LISTEN_PID=$$ exec "$@"'


@contextlib.contextmanager
def socket_activated(
    command_for_port: Callable[[int], Sequence[str]],
    *,
    env: Mapping[str, str],
    output: int | IO[bytes],
) -> Iterator[tuple[subprocess.Popen[bytes], int]]:
    """The running server and its port; leaving the context kills and reaps it."""
    child_env = {key: value for key, value in env.items() if not key.startswith("LISTEN_")}
    child_env["LISTEN_FDS"] = "1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        proc = subprocess.Popen(
            ["sh", "-c", _SD_LISTEN_HANDOFF, "sh", *command_for_port(port)],
            stdin=listener.fileno(),
            env=child_env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        yield proc, port
    finally:
        proc.kill()
        proc.wait(timeout=10)
