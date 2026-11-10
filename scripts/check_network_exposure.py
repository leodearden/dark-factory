#!/usr/bin/env python3
"""Behavioural loopback-only probe of every factory port.

Contract: plans/mcp-network-exposure-prd.md §3 "Contract of the exposure check".
Stdlib only, so it runs under a plain system python3.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import socket
from typing import Final, TypeVar

TOOL: Final = "network_exposure"

SOCKET_UNIT_PATTERNS: Final = ("fused-memory.socket", "orchestrator-*.socket")
LIST_SOCKETS_ARGV: Final = ("systemctl", "--user", "list-sockets", "-o", "json")
IP_ADDR_ARGV: Final = ("ip", "-j", "addr")

_LOOPBACK_SCOPE: Final = "host"
_LINK_SCOPE: Final = "link"
_FAMILIES: Final = {"inet": socket.AF_INET, "inet6": socket.AF_INET6}


class SourceUnreadable(Exception):
    """A structured source could not be read, or did not have the expected shape."""

    def __init__(self, command: tuple[str, ...], reason: str) -> None:
        super().__init__(f"{' '.join(command)}: {reason}")
        self.command = command
        self.reason = reason


@dataclasses.dataclass(frozen=True)
class OwnedPort:
    port: int
    owner: str


@dataclasses.dataclass(frozen=True)
class HostAddress:
    ip: str
    family: socket.AddressFamily
    scope_id: int
    ifname: str

    def sockaddr(self, port: int) -> tuple[str, int] | tuple[str, int, int, int]:
        if self.family == socket.AF_INET6:
            return (self.ip, port, 0, self.scope_id)
        return (self.ip, port)

    @property
    def label(self) -> str:
        if self.scope_id:
            return f"{self.ip}%{self.ifname}"
        return f"{self.ip} ({self.ifname})"


def _require_list(value: object, command: tuple[str, ...], what: str) -> list[object]:
    if not isinstance(value, list):
        raise SourceUnreadable(command, f"{what} is {type(value).__name__}, not a list")
    return value


def _require_object(value: object, command: tuple[str, ...], what: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SourceUnreadable(command, f"{what} is {type(value).__name__}, not an object")
    return value


_Field = TypeVar("_Field")


def _require_field(
    row: dict[str, object], key: str, kind: type[_Field], command: tuple[str, ...]
) -> _Field:
    value = row.get(key)
    if not isinstance(value, kind):
        raise SourceUnreadable(command, f"entry {row!r} has no {kind.__name__} {key!r}")
    return value


def _tcp_port(listen: str) -> int | None:
    _, _, tail = listen.rpartition(":")
    return int(tail) if tail.isdigit() else None


def _is_factory_unit(unit: str) -> bool:
    return any(fnmatch.fnmatchcase(unit, pattern) for pattern in SOCKET_UNIT_PATTERNS)


def socket_unit_ports(rows: object) -> list[OwnedPort]:
    """The TCP port of every factory socket unit row; the listen's host part is ignored."""
    owned: list[OwnedPort] = []
    for raw in _require_list(rows, LIST_SOCKETS_ARGV, "the socket list"):
        row = _require_object(raw, LIST_SOCKETS_ARGV, "a socket row")
        listen = _require_field(row, "listen", str, LIST_SOCKETS_ARGV)
        unit = _require_field(row, "unit", str, LIST_SOCKETS_ARGV)
        port = _tcp_port(listen)
        if port is not None and _is_factory_unit(unit):
            owned.append(OwnedPort(port=port, owner=unit))
    return owned


def _link_addresses(link: dict[str, object]) -> list[HostAddress]:
    ifname = _require_field(link, "ifname", str, IP_ADDR_ARGV)
    ifindex = _require_field(link, "ifindex", int, IP_ADDR_ARGV)
    addresses: list[HostAddress] = []
    for raw in _require_list(link.get("addr_info", []), IP_ADDR_ARGV, f"{ifname}'s addr_info"):
        info = _require_object(raw, IP_ADDR_ARGV, "an addr_info entry")
        family_name = _require_field(info, "family", str, IP_ADDR_ARGV)
        local = _require_field(info, "local", str, IP_ADDR_ARGV)
        scope = _require_field(info, "scope", str, IP_ADDR_ARGV)
        family = _FAMILIES.get(family_name)
        if family is None or scope == _LOOPBACK_SCOPE:
            continue
        scope_id = ifindex if family == socket.AF_INET6 and scope == _LINK_SCOPE else 0
        addresses.append(HostAddress(ip=local, family=family, scope_id=scope_id, ifname=ifname))
    return addresses


def host_addresses(links: object) -> list[HostAddress]:
    """Every address whose scope is not "host" (the kernel's loopback label), deduplicated."""
    seen: dict[HostAddress, None] = {}
    for raw in _require_list(links, IP_ADDR_ARGV, "the link list"):
        link = _require_object(raw, IP_ADDR_ARGV, "a link")
        seen.update(dict.fromkeys(_link_addresses(link)))
    return list(seen)
