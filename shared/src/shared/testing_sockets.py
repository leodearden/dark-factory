"""Loopback sockets a test aims a client at when nothing must answer, for tests.

A port bound to 127.0.0.1:0 and then closed is free again at once, so on a
shared host any concurrent bind (another xdist worker's stub server, another
verify) can be handed it before the client under test connects. This module
keeps the port reserved instead. It is a pure-stdlib leaf, registered in
shared/tests/test_pure_stdlib_leaves.py.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ['refused_port']


@contextmanager
def refused_port() -> Iterator[int]:
    """A loopback port held bound, with no listen() and no SO_REUSEADDR, for the with-block: every connect to it is refused and no other socket can bind it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(('127.0.0.1', 0))
        yield reservation.getsockname()[1]
