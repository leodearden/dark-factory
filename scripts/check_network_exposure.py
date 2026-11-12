#!/usr/bin/env python3
"""Behavioural loopback-only probe of every factory port.

Contract: plans/mcp-network-exposure-prd.md §3 "Contract of the exposure check".
Stdlib only, so it runs under a plain system python3.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import enum
import errno
import fnmatch
import itertools
import json
import socket
import subprocess
import sys
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Final, TextIO, TypeAlias, TypeVar

from systemd_unit_parity import emit

TOOL: Final = "network_exposure"

SOCKET_UNIT_PATTERNS: Final = ("fused-memory.socket", "orchestrator-*.socket")
LIST_SOCKETS_ARGV: Final = ("systemctl", "--user", "list-sockets", "-o", "json")
IP_ADDR_ARGV: Final = ("ip", "-j", "addr")
SOURCE_TIMEOUT_SECS: Final = 30
DEFAULT_CONNECT_TIMEOUT_SECS: Final = 1.0
PROBE_WORKERS: Final = 32

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


Sockaddr: TypeAlias = tuple[str, int] | tuple[str, int, int, int]


@dataclasses.dataclass(frozen=True)
class HostAddress:
    ip: str
    family: socket.AddressFamily
    scope_id: int
    ifname: str

    def sockaddr(self, port: int) -> Sockaddr:
        if self.family == socket.AF_INET6:
            return (self.ip, port, 0, self.scope_id)
        return (self.ip, port)

    @property
    def label(self) -> str:
        if self.scope_id:
            return f"{self.ip}%{self.ifname}"
        return f"{self.ip} ({self.ifname})"


LOOPBACK: Final = HostAddress(ip="127.0.0.1", family=socket.AF_INET, scope_id=0, ifname="lo")


# Host port -> compose service, for every `ports` publish in
# fused-memory/docker/docker-compose.yml; pinned there by
# tests/scripts/test_check_network_exposure.py.
COMPOSE_PUBLISHED_PORTS: Final[Mapping[int, str]] = types.MappingProxyType(
    {6379: "falkordb", 6380: "falkordb-scratch", 6333: "qdrant", 6334: "qdrant", 8002: "fused-mcp", 8420: "embedder"}
)
RECON_REPORT_PORT: Final = 8003

STATIC_PORTS: Final[tuple[OwnedPort, ...]] = (
    *(
        OwnedPort(port=port, owner=f"compose service {service}")
        for port, service in COMPOSE_PUBLISHED_PORTS.items()
    ),
    OwnedPort(
        port=RECON_REPORT_PORT,
        owner="fused-memory recon-report (ServerConfig.recon_report_port)",
    ),
)


def merge_owners(*sources: Iterable[OwnedPort]) -> dict[int, tuple[str, ...]]:
    """Port -> every owner of it, once each in source order, sorted by port."""
    owners: dict[int, dict[str, None]] = {}
    for source in sources:
        for owned in source:
            owners.setdefault(owned.port, {})[owned.owner] = None
    return {port: tuple(owners[port]) for port in sorted(owners)}


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


class ExposureExit(enum.IntEnum):
    """What the check's exit status means.

    LOOPBACK_ONLY: every probed port refused every off-loopback connect and
    accepted on 127.0.0.1, and both sources were read.
    FINDING: anything else, including a source that could not be read.

    2 is never produced under the [network_exposure] tag: it belongs to
    argparse rejecting a flag and to python3 refusing a missing script.
    setup-host.sh relies on that to believe a status only when the tag is present.
    """

    LOOPBACK_ONLY = 0
    FINDING = 1


class Outcome(enum.Enum):
    ACCEPTED = "was accepted"
    REFUSED = "was refused"
    TIMED_OUT = "timed out"
    FAILED = "failed"


@dataclasses.dataclass(frozen=True)
class ProbeResult:
    outcome: Outcome
    errno_name: str = ""

    @property
    def description(self) -> str:
        if self.outcome is Outcome.FAILED:
            return f"failed with {self.errno_name}"
        return self.outcome.value


def _errno_name(exc: OSError) -> str:
    if exc.errno is None:
        return str(exc)
    return errno.errorcode.get(exc.errno, str(exc.errno))


def probe(family: socket.AddressFamily, sockaddr: Sockaddr, timeout: float) -> ProbeResult:
    """One TCP connect, classified."""
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(sockaddr)
    except ConnectionRefusedError:
        return ProbeResult(Outcome.REFUSED)
    except TimeoutError:
        return ProbeResult(Outcome.TIMED_OUT)
    except OSError as exc:
        return ProbeResult(Outcome.FAILED, _errno_name(exc))
    return ProbeResult(Outcome.ACCEPTED)


def read_json_source(argv: tuple[str, ...]) -> object:
    """Run *argv* and parse its stdout as JSON, or raise SourceUnreadable saying why not."""
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=SOURCE_TIMEOUT_SECS,
            check=False,
        )
    except FileNotFoundError:
        raise SourceUnreadable(argv, "command not found on PATH") from None
    except subprocess.TimeoutExpired:
        raise SourceUnreadable(argv, f"no answer within {SOURCE_TIMEOUT_SECS} s") from None
    except OSError as exc:
        raise SourceUnreadable(argv, f"could not start ({_errno_name(exc)})") from None
    if completed.returncode != 0:
        raise SourceUnreadable(argv, f"exited {completed.returncode}: {completed.stderr.strip()}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SourceUnreadable(argv, f"output is not JSON ({exc})") from None


class FindingKind(enum.Enum):
    SOURCE_ERROR = "source-error"
    EXPOSED = "exposed"
    TIMED_OUT = "timed-out"
    UNVERIFIED = "unverified"
    NOT_LISTENING = "not-listening"


_OFF_LOOPBACK_FINDINGS: Final = {
    Outcome.ACCEPTED: FindingKind.EXPOSED,
    Outcome.TIMED_OUT: FindingKind.TIMED_OUT,
    Outcome.FAILED: FindingKind.UNVERIFIED,
}


def _finding_line(kind: FindingKind, text: str) -> str:
    return f"[{kind.value}] {text}"


@dataclasses.dataclass(frozen=True)
class Target:
    port: int
    address: HostAddress
    loopback: bool


@dataclasses.dataclass(frozen=True)
class Finding:
    kind: FindingKind
    target: Target
    owners: tuple[str, ...]
    result: ProbeResult

    def render(self) -> str:
        subject = f"port {self.target.port} ({', '.join(self.owners)})"
        label = self.target.address.label
        if self.kind is FindingKind.EXPOSED:
            text = f"{subject} accepted a connect on {label}"
        elif self.kind is FindingKind.NOT_LISTENING:
            text = f"{subject}: a connect on {label} {self.result.description}, so nothing serves it on loopback"
        else:
            text = f"{subject}: a connect on {label} {self.result.description} instead of being refused"
        return _finding_line(self.kind, text)


def _port_targets(port: int, addresses: Sequence[HostAddress]) -> list[Target]:
    return [
        *(Target(port, address, loopback=False) for address in addresses),
        Target(port, LOOPBACK, loopback=True),
    ]


def _finding(target: Target, owners: tuple[str, ...], result: ProbeResult) -> Finding | None:
    if target.loopback:
        kind = None if result.outcome is Outcome.ACCEPTED else FindingKind.NOT_LISTENING
    else:
        kind = _OFF_LOOPBACK_FINDINGS.get(result.outcome)
    return None if kind is None else Finding(kind, target, owners, result)


def _probe_target(target: Target, timeout: float) -> ProbeResult:
    return probe(target.address.family, target.address.sockaddr(target.port), timeout)


def _probe_all(targets: Sequence[Target], timeout: float) -> list[ProbeResult]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=PROBE_WORKERS) as pool:
        return list(pool.map(_probe_target, targets, itertools.repeat(timeout)))


_Parsed = TypeVar("_Parsed")


def _read_source(
    argv: tuple[str, ...], parse: Callable[[object], list[_Parsed]]
) -> tuple[list[_Parsed], SourceUnreadable | None]:
    try:
        return parse(read_json_source(argv)), None
    except SourceUnreadable as exc:
        return [], exc


def _source_error_line(error: SourceUnreadable) -> str:
    return _finding_line(FindingKind.SOURCE_ERROR, f"{' '.join(error.command)}: {error.reason}")


def run(
    *,
    static_ports: Sequence[OwnedPort] = STATIC_PORTS,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECS,
    out: TextIO,
) -> ExposureExit:
    """Probe every factory port on every off-loopback address and on 127.0.0.1; report and judge."""
    socket_ports, socket_error = _read_source(LIST_SOCKETS_ARGV, socket_unit_ports)
    addresses, address_error = _read_source(IP_ADDR_ARGV, host_addresses)
    errors = [error for error in (socket_error, address_error) if error is not None]
    owners = merge_owners(socket_ports, static_ports)
    targets = [target for port in owners for target in _port_targets(port, addresses)]
    results = _probe_all(targets, connect_timeout)
    findings = [
        finding
        for target, result in zip(targets, results, strict=True)
        if (finding := _finding(target, owners[target.port], result)) is not None
    ]
    emit(TOOL, f"probing {len(owners)} port(s) on {len(addresses)} non-loopback address(es) and 127.0.0.1", out=out)
    return _report([*map(_source_error_line, errors), *(finding.render() for finding in findings)], out)


def _report(failure_lines: Sequence[str], out: TextIO) -> ExposureExit:
    for line in failure_lines:
        emit(TOOL, line, out=out)
    if failure_lines:
        emit(TOOL, f"[finding] {len(failure_lines)} finding(s) above", out=out)
        return ExposureExit.FINDING
    emit(TOOL, "[ok] every probed port refuses off-loopback connects and accepts on 127.0.0.1", out=out)
    return ExposureExit.LOOPBACK_ONLY


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Exit 0 only if every factory port refuses TCP connects on every "
            "non-loopback address of this host and accepts on 127.0.0.1; "
            "every report line is tagged [network_exposure]."
        )
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=DEFAULT_CONNECT_TIMEOUT_SECS,
        metavar="SECS",
        help="per-connect timeout; a timeout is reported as a failure (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return int(run(connect_timeout=args.connect_timeout, out=sys.stdout))


if __name__ == "__main__":
    sys.exit(main())
