"""socket_activated: a server on a loopback port the caller bound first."""
from __future__ import annotations

import errno
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from socket_activation import socket_activated

_PAUSE_FOREVER = ["-c", "import signal; signal.pause()"]

_SERVE_ONE_GREETING_ON_FD_3 = """\
import os, socket, stat, sys
assert os.environ["LISTEN_FDS"] == "1"
assert int(os.environ["LISTEN_PID"]) == os.getpid()
assert "LISTEN_FDNAMES" not in os.environ
assert not stat.S_ISSOCK(os.fstat(0).st_mode)
listener = socket.socket(fileno=3)
assert listener.getsockname()[1] == int(sys.argv[1])
conn, _ = listener.accept()
conn.sendall(b"served-on-fd-3")
conn.close()
"""


def test_the_port_is_held_before_the_child_ever_listens() -> None:
    with socket_activated(
        lambda _port: [sys.executable, "-I", *_PAUSE_FOREVER],
        env=dict(os.environ),
        output=subprocess.DEVNULL,
    ) as (proc, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as squatter:
            with pytest.raises(OSError) as refused:
                squatter.bind(("127.0.0.1", port))
        assert refused.value.errno == errno.EADDRINUSE
    assert proc.poll() is not None


def test_the_child_serves_on_fd_3_addressed_to_its_own_pid(tmp_path: Path) -> None:
    stale = {**os.environ, "LISTEN_PID": "1", "LISTEN_FDNAMES": "stale"}
    with (tmp_path / "child.log").open("wb") as log:
        with socket_activated(
            lambda port: [sys.executable, "-I", "-c", _SERVE_ONE_GREETING_ON_FD_3, str(port)],
            env=stale,
            output=log,
        ) as (_proc, port):
            with socket.create_connection(("127.0.0.1", port)) as client:
                assert client.recv(64) == b"served-on-fd-3"
