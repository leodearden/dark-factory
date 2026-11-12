"""Contract tests for config_reload_script_fakes.ClosedPort, the held dead-port reservation."""
from __future__ import annotations

import errno
import socket

import pytest
from config_reload_script_fakes import ClosedPort


def _ipv4_rival() -> socket.socket:
    return socket.socket(socket.AF_INET)


def _dual_stack_rival() -> socket.socket:
    rival = socket.socket(socket.AF_INET6)
    rival.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    return rival


@pytest.mark.parametrize(
    ("make_rival", "address"),
    [
        pytest.param(_ipv4_rival, "127.0.0.1", id="ipv4-loopback"),
        pytest.param(_ipv4_rival, "0.0.0.0", id="ipv4-any"),
        pytest.param(_dual_stack_rival, "::", id="dual-stack-any",
                     marks=pytest.mark.skipif(not socket.has_ipv6, reason="no IPv6 on this host")),
    ],
)
def test_no_server_can_bind_a_held_closed_port(make_rival, address):
    with ClosedPort() as closed, make_rival() as rival:
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


def test_a_held_closed_port_refuses_a_second_entry_and_keeps_its_reservation():
    with ClosedPort() as closed:
        held = closed.port
        with pytest.raises(RuntimeError), closed:
            pass
        assert closed.port == held
