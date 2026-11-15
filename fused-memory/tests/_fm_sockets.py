"""Loopback sockets a test aims a client at: a refused port, and a listener that records being dialled.

Lives outside conftest.py, imported by name, under the convention stated in
_fm_helpers' module docstring.
"""

import socket
from collections.abc import Iterator
from contextlib import closing, contextmanager


@contextmanager
def refused_port() -> Iterator[int]:
    """A loopback port bound but not listening, so a connect to it is refused while it is held."""
    with closing(socket.socket()) as reservation:
        reservation.bind(('127.0.0.1', 0))
        yield reservation.getsockname()[1]


@contextmanager
def silent_listener() -> Iterator[socket.socket]:
    """A port that completes TCP handshakes and never answers, so a client that dials it is detectable."""
    with closing(socket.socket()) as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        listener.setblocking(False)
        yield listener


def has_pending_connection(listener: socket.socket) -> bool:
    try:
        connection, _ = listener.accept()
    except BlockingIOError:
        return False
    connection.close()
    return True
