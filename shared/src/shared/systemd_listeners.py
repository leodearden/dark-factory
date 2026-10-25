"""Adopt listening sockets passed by systemd socket activation (sd_listen_fds).

A ``<unit>.socket`` keeps the listening socket bound while its service
restarts, so clients connecting mid-restart queue in the kernel backlog
instead of being refused. Claude Code's HTTP MCP client gives up for good
after ~15s of refused connections; a restart takes ~50s.

``uv run`` spawns the Python server as a child and passes fd 3 through, so
systemd's LISTEN_PID names the ``uv`` parent rather than this process —
both are accepted.
"""

import os
import socket
from collections.abc import MutableMapping

SD_LISTEN_FDS_START = 3
_ACTIVATION_ENV = ('LISTEN_FDS', 'LISTEN_PID', 'LISTEN_FDNAMES')


def take_systemd_listeners(
    environ: MutableMapping[str, str] | None = None,
) -> dict[int, socket.socket]:
    """Return systemd-passed TCP listening sockets keyed by local port.

    Consumes the activation variables from *environ* (default ``os.environ``)
    so child processes never mistake them for their own; a second call
    therefore returns ``{}``. Returns ``{}`` when the process was not socket
    activated. Adopted sockets are made non-inheritable.
    """
    env = os.environ if environ is None else environ
    count = int(env.get('LISTEN_FDS') or 0)
    owner = int(env.get('LISTEN_PID') or 0)
    for name in _ACTIVATION_ENV:
        env.pop(name, None)
    if count < 1 or owner not in (os.getpid(), os.getppid()):
        return {}
    listeners: dict[int, socket.socket] = {}
    for fd in range(SD_LISTEN_FDS_START, SD_LISTEN_FDS_START + count):
        sock = socket.socket(fileno=fd)
        if sock.type != socket.SOCK_STREAM or sock.family not in (
            socket.AF_INET,
            socket.AF_INET6,
        ):
            sock.detach()
            continue
        sock.set_inheritable(False)
        listeners[sock.getsockname()[1]] = sock
    return listeners
