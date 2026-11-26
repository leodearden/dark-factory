"""Contract tests for shared.testing_sockets.refused_port, the held dead-port reservation."""
from __future__ import annotations

import errno
import socket

import pytest
from shared.testing_sockets import refused_port


def _ipv4_rival() -> socket.socket:
    return socket.socket(socket.AF_INET)


def _dual_stack_rival() -> socket.socket:
    rival = socket.socket(socket.AF_INET6)
    rival.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    return rival


def test_a_connect_to_a_held_refused_port_is_refused():
    with refused_port() as port, pytest.raises(ConnectionRefusedError):
        socket.create_connection(('127.0.0.1', port), timeout=5)


@pytest.mark.parametrize(
    ('make_rival', 'address'),
    [
        pytest.param(_ipv4_rival, '127.0.0.1', id='ipv4-loopback'),
        pytest.param(_ipv4_rival, '0.0.0.0', id='ipv4-any'),
        pytest.param(_dual_stack_rival, '::', id='dual-stack-any',
                     marks=pytest.mark.skipif(not socket.has_ipv6, reason='no IPv6 on this host')),
    ],
)
def test_no_server_can_bind_a_held_refused_port(make_rival, address):
    with refused_port() as port, make_rival() as rival:
        rival.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with pytest.raises(OSError) as refused:
            rival.bind((address, port))
    assert refused.value.errno == errno.EADDRINUSE
