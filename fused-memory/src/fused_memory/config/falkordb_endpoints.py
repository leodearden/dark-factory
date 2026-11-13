"""The FalkorDB servers fused-memory knows by address: production, and the
scratch server that tests and evals write to (task 7792; OPERATIONS.md
"The scratch FalkorDB")."""

from __future__ import annotations

import ipaddress
import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from fused_memory.config.schema import FusedMemoryConfig

FALKOR_HOST_ENV = 'FALKOR_HOST'
FALKOR_PORT_ENV = 'FALKOR_PORT'

_DEFAULT_HOST = 'localhost'
_REDIS_DEFAULT_PORT = 6379
_THIS_HOST_NAMES = frozenset({'', 'localhost', 'localhost.localdomain', 'ip6-localhost', 'ip6-loopback'})


@dataclass(frozen=True)
class FalkorEndpoint:
    host: str
    port: int

    @property
    def uri(self) -> str:
        return f'redis://{self.host}:{self.port}'

    @classmethod
    def from_uri(cls, uri: str) -> FalkorEndpoint:
        parts = urlsplit(uri)
        return cls(parts.hostname or _DEFAULT_HOST, parts.port or _REDIS_DEFAULT_PORT)


PRODUCTION_FALKORDB = FalkorEndpoint(_DEFAULT_HOST, _REDIS_DEFAULT_PORT)
SCRATCH_FALKORDB = FalkorEndpoint(_DEFAULT_HOST, 6380)


class ProductionFalkorDBTarget(RuntimeError):
    """A scratch FalkorDB target that resolves to the production server."""


def is_production_falkordb(endpoint: FalkorEndpoint) -> bool:
    return endpoint.port == PRODUCTION_FALKORDB.port and _names_this_host(endpoint.host)


def _names_this_host(host: str) -> bool:
    """Whether a connect to *host* lands on this machine, in any spelling: localhost, loopback, unspecified."""
    name = host.strip().strip('[]').rstrip('.').lower()
    if name in _THIS_HOST_NAMES or name.endswith('.localhost'):
        return True
    address = _numeric_address(name)
    if address is None:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback or address.is_unspecified


def _numeric_address(name: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """*name* as an IP address, accepting the short IPv4 forms a resolver accepts (``127.1``)."""
    try:
        return ipaddress.ip_address(name)
    except ValueError:
        pass
    try:
        return ipaddress.IPv4Address(socket.inet_aton(name))
    except OSError:
        return None


def with_falkordb(cfg: FusedMemoryConfig, endpoint: FalkorEndpoint) -> FusedMemoryConfig:
    """*cfg* with its FalkorDB moved to *endpoint*, which takes no password; every other field kept."""
    falkordb = cfg.graphiti.falkordb.model_copy(update={'uri': endpoint.uri, 'password': None})
    return cfg.model_copy(update={'graphiti': cfg.graphiti.model_copy(update={'falkordb': falkordb})})


def scratch_falkordb_from_env(environ: Mapping[str, str] = os.environ) -> FalkorEndpoint:
    """Resolve the scratch server from FALKOR_HOST/FALKOR_PORT, refusing production.

    Unset or empty variables take SCRATCH_FALKORDB's host and port.

    Raises:
        RuntimeError: FALKOR_PORT is set but is not an integer.
        ProductionFalkorDBTarget: the variables aim at the production server.
    """
    raw_host = environ.get(FALKOR_HOST_ENV, '').strip()
    raw_port = environ.get(FALKOR_PORT_ENV, '').strip()
    endpoint = FalkorEndpoint(
        raw_host or SCRATCH_FALKORDB.host,
        _port_from(raw_port) if raw_port else SCRATCH_FALKORDB.port,
    )
    if is_production_falkordb(endpoint):
        raise ProductionFalkorDBTarget(
            f'{FALKOR_HOST_ENV}={endpoint.host!r} {FALKOR_PORT_ENV}={raw_port!r} aims at '
            f'the production FalkorDB ({PRODUCTION_FALKORDB.uri}). Tests and evals '
            f'write only to the scratch server; unset both to use '
            f'{SCRATCH_FALKORDB.uri}.'
        )
    return endpoint


def _port_from(raw_port: str) -> int:
    try:
        return int(raw_port)
    except ValueError:
        raise RuntimeError(
            f'{FALKOR_PORT_ENV} must be an integer, got {raw_port!r}. Unset it to use '
            f'the scratch default ({SCRATCH_FALKORDB.port}), or set it to the port '
            f'your scratch FalkorDB listens on.'
        ) from None
