"""The loopback sockets in _fm_sockets behave as their callers rely on."""

import errno
import socket

import pytest
from _fm_sockets import has_pending_connection, refused_port, silent_listener


def test_a_refused_port_turns_connects_away_and_cannot_be_bound_while_held() -> None:
    with refused_port() as port:
        with pytest.raises(ConnectionRefusedError):
            socket.create_connection(('127.0.0.1', port), timeout=5)
        with socket.socket() as server, pytest.raises(OSError) as bind:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('127.0.0.1', port))

    assert bind.value.errno == errno.EADDRINUSE


def test_a_silent_listener_reports_whether_anything_dialled_it() -> None:
    with silent_listener() as undialled, silent_listener() as dialled:
        with socket.create_connection(dialled.getsockname(), timeout=5):
            assert has_pending_connection(dialled)
        assert not has_pending_connection(undialled)
