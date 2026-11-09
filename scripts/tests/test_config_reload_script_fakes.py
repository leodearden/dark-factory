"""Contract tests for config_reload_script_fakes.ClosedPort, the held dead-port reservation."""
from __future__ import annotations

import errno
import socket

import pytest
from config_reload_script_fakes import ClosedPort


@pytest.mark.parametrize("address", ["127.0.0.1", "0.0.0.0"])
def test_no_server_can_bind_a_held_closed_port(address):
    with ClosedPort() as closed, socket.socket() as rival:
        rival.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with pytest.raises(OSError) as refused:
            rival.bind((address, closed.port))
    assert refused.value.errno == errno.EADDRINUSE


def test_a_connect_to_a_held_closed_port_is_refused():
    with ClosedPort() as closed, pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", closed.port), timeout=10)


def test_a_closed_port_offers_no_port_outside_its_with_block():
    with pytest.raises(RuntimeError):
        _ = ClosedPort().port
    with ClosedPort() as closed:
        pass
    with pytest.raises(RuntimeError):
        _ = closed.port
