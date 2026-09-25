"""Tests for shared.systemd_listeners.take_systemd_listeners.

Each case runs a child interpreter whose fd 3 is a real listening socket,
exactly as systemd hands it over, and reads the child's verdict from stdout.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys

import pytest

_CHILD = """
import json, os, sys
from shared.systemd_listeners import take_systemd_listeners
listeners = take_systemd_listeners()
report = {
    'ports': sorted(listeners),
    'inheritable': [os.get_inheritable(s.fileno()) for s in listeners.values()],
    'env_left': sorted(k for k in ('LISTEN_FDS', 'LISTEN_PID', 'LISTEN_FDNAMES') if k in os.environ),
    'second_call': sorted(take_systemd_listeners()),
}
print(json.dumps(report), flush=True)
if listeners and '--serve-one' in sys.argv:
    conn, _ = next(iter(listeners.values())).accept()
    conn.sendall(b'served-by-child')
    conn.close()
"""


@pytest.fixture
def listening_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    sock.listen(8)
    yield sock
    sock.close()


def _spawn(sock: socket.socket, env_extra: dict[str, str], *, argv_prefix=(), serve_one=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith('LISTEN_')}
    env.update(env_extra)
    args = [*argv_prefix, sys.executable, '-c', _CHILD]
    if serve_one:
        args.append('--serve-one')
    fd = sock.fileno()
    # close_fds would run AFTER preexec_fn and close the fd 3 it just made;
    # Python's own fds are non-inheritable, so only fd 3 crosses the exec.
    return subprocess.Popen(
        args,
        env=env,
        stdout=subprocess.PIPE,
        text=True,
        close_fds=False,
        preexec_fn=lambda: os.dup2(fd, 3),
    )


def _report(proc: subprocess.Popen) -> dict:
    line = proc.stdout.readline()
    assert line, 'child printed no report'
    return json.loads(line)


def test_adopts_a_socket_whose_listen_pid_is_the_parent(listening_socket):
    """The production shape: ``uv run`` is LISTEN_PID, Python is its child."""
    port = listening_socket.getsockname()[1]
    proc = _spawn(
        listening_socket,
        {'LISTEN_FDS': '1', 'LISTEN_PID': str(os.getpid()), 'LISTEN_FDNAMES': 'x'},
    )
    report = _report(proc)
    assert proc.wait(timeout=30) == 0
    assert report == {
        'ports': [port],
        'inheritable': [False],
        'env_left': [],
        'second_call': [],
    }


def test_adopts_a_socket_whose_listen_pid_is_the_process_itself(listening_socket):
    port = listening_socket.getsockname()[1]
    proc = _spawn(
        listening_socket,
        {'LISTEN_FDS': '1'},
        argv_prefix=('sh', '-c', 'LISTEN_PID=$$ exec "$@"', 'sh'),
    )
    report = _report(proc)
    assert proc.wait(timeout=30) == 0
    assert report['ports'] == [port]


def test_ignores_sockets_addressed_to_another_process_but_still_consumes_the_env(
    listening_socket,
):
    proc = _spawn(listening_socket, {'LISTEN_FDS': '1', 'LISTEN_PID': '1'})
    report = _report(proc)
    assert proc.wait(timeout=30) == 0
    assert report['ports'] == []
    assert report['env_left'] == []


def test_returns_nothing_when_not_socket_activated(listening_socket):
    proc = _spawn(listening_socket, {})
    report = _report(proc)
    assert proc.wait(timeout=30) == 0
    assert report['ports'] == []


def test_the_adopted_socket_is_the_live_listener(listening_socket):
    """A client connecting to the systemd-held port is served by the child."""
    port = listening_socket.getsockname()[1]
    proc = _spawn(
        listening_socket,
        {'LISTEN_FDS': '1', 'LISTEN_PID': str(os.getpid())},
        serve_one=True,
    )
    assert _report(proc)['ports'] == [port]
    listening_socket.close()
    with socket.create_connection(('127.0.0.1', port), timeout=10) as client:
        assert client.recv(64) == b'served-by-child'
    assert proc.wait(timeout=30) == 0
