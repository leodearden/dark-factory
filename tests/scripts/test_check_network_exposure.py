"""Tests for scripts/check_network_exposure.py (contract: plans/mcp-network-exposure-prd.md §3).

The structured-source parsers are driven with JSON shaped like the live
`systemctl --user list-sockets -o json` and `ip -j addr` outputs.
"""

from __future__ import annotations

import ast
import pathlib
import socket

import check_network_exposure as exposure
import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).parents[2]
COMPOSE_PATH = REPO_ROOT / "fused-memory" / "docker" / "docker-compose.yml"
SCHEMA_PATH = REPO_ROOT / "fused-memory" / "src" / "fused_memory" / "config" / "schema.py"
FM_CONFIG_PATH = REPO_ROOT / "fused-memory" / "config" / "config.yaml"

# ---------------------------------------------------------------------------
# socket_unit_ports: `systemctl --user list-sockets -o json`
# ---------------------------------------------------------------------------

_SOCKET_ROWS = [
    {"listen": "0.0.0.0:8002", "unit": "fused-memory.socket", "activates": "fused-memory.service"},
    {"listen": "127.0.0.1:8103", "unit": "fused-memory.socket", "activates": "fused-memory.service"},
    {
        "listen": "127.0.0.1:8102",
        "unit": "orchestrator-dark-factory.socket",
        "activates": "orchestrator-dark-factory.service",
    },
    {"listen": "[::]:8104", "unit": "orchestrator-v6.socket", "activates": "orchestrator-v6.service"},
    {"listen": "/run/user/1000/bus", "unit": "dbus.socket", "activates": "dbus.service"},
    {"listen": "127.0.0.1:9999", "unit": "gpg-agent.socket", "activates": "gpg-agent.service"},
    {"listen": "/run/user/1000/fm.sock", "unit": "fused-memory.socket", "activates": "fused-memory.service"},
]


def test_socket_unit_ports_selects_factory_units_and_takes_only_the_port():
    assert exposure.socket_unit_ports(_SOCKET_ROWS) == [
        exposure.OwnedPort(port=8002, owner="fused-memory.socket"),
        exposure.OwnedPort(port=8103, owner="fused-memory.socket"),
        exposure.OwnedPort(port=8102, owner="orchestrator-dark-factory.socket"),
        exposure.OwnedPort(port=8104, owner="orchestrator-v6.socket"),
    ]


def test_socket_unit_ports_of_no_rows_is_empty():
    assert exposure.socket_unit_ports([]) == []


@pytest.mark.parametrize(
    "rows",
    [
        {"listen": "0.0.0.0:8002", "unit": "fused-memory.socket"},
        ["fused-memory.socket"],
        [{"unit": "fused-memory.socket"}],
        [{"listen": "0.0.0.0:8002"}],
        [{"listen": 8002, "unit": "fused-memory.socket"}],
    ],
    ids=["not-a-list", "row-not-object", "no-listen", "no-unit", "listen-not-string"],
)
def test_socket_unit_ports_rejects_a_wrong_shape(rows: object):
    with pytest.raises(exposure.SourceUnreadable):
        exposure.socket_unit_ports(rows)


# ---------------------------------------------------------------------------
# host_addresses: `ip -j addr`
# ---------------------------------------------------------------------------


def _addr(family: str, local: str, scope: str) -> dict[str, object]:
    return {"family": family, "local": local, "prefixlen": 24, "scope": scope}


def _link(ifindex: int, ifname: str, *addrs: dict[str, object]) -> dict[str, object]:
    return {"ifindex": ifindex, "ifname": ifname, "link_type": "ether", "addr_info": list(addrs)}


_LINKS = [
    _link(1, "lo", _addr("inet", "127.0.0.1", "host"), _addr("inet6", "::1", "host")),
    _link(
        2,
        "enp39s0",
        _addr("inet", "192.168.1.105", "global"),
        _addr("inet6", "2a0e:1d47:c504:c500:2ef0:5dff:fe25:b556", "global"),
        _addr("inet6", "fe80::2ef0:5dff:fe25:b556", "link"),
    ),
    _link(3, "wlo1"),
    _link(4, "tailscale0", _addr("inet", "100.107.28.56", "global")),
    _link(5, "docker0", _addr("inet", "172.17.0.1", "global")),
]


def _by_ip(addresses: list[exposure.HostAddress]) -> dict[str, exposure.HostAddress]:
    return {address.ip: address for address in addresses}


def test_host_addresses_excludes_host_scope_and_keeps_every_other_address():
    assert set(_by_ip(exposure.host_addresses(_LINKS))) == {
        "192.168.1.105",
        "2a0e:1d47:c504:c500:2ef0:5dff:fe25:b556",
        "fe80::2ef0:5dff:fe25:b556",
        "100.107.28.56",
        "172.17.0.1",
    }


def test_host_addresses_carries_family_scope_id_and_link_name():
    by_ip = _by_ip(exposure.host_addresses(_LINKS))
    assert by_ip["192.168.1.105"] == exposure.HostAddress(
        ip="192.168.1.105", family=socket.AF_INET, scope_id=0, ifname="enp39s0"
    )
    assert by_ip["2a0e:1d47:c504:c500:2ef0:5dff:fe25:b556"] == exposure.HostAddress(
        ip="2a0e:1d47:c504:c500:2ef0:5dff:fe25:b556",
        family=socket.AF_INET6,
        scope_id=0,
        ifname="enp39s0",
    )
    assert by_ip["172.17.0.1"].ifname == "docker0"


def test_host_addresses_scopes_an_ipv6_link_local_to_its_link():
    link_local = _by_ip(exposure.host_addresses(_LINKS))["fe80::2ef0:5dff:fe25:b556"]
    assert link_local.family == socket.AF_INET6
    assert link_local.scope_id == 2


def test_host_addresses_decides_by_scope_not_by_interface():
    links = [_link(1, "lo", _addr("inet", "127.0.0.1", "host"), _addr("inet", "10.9.8.7", "global"))]
    assert [address.ip for address in exposure.host_addresses(links)] == ["10.9.8.7"]


def test_host_addresses_dedupes_an_address_listed_twice():
    links = [
        _link(2, "eth0", _addr("inet", "10.0.0.5", "global"), _addr("inet", "10.0.0.5", "global")),
    ]
    assert [address.ip for address in exposure.host_addresses(links)] == ["10.0.0.5"]


@pytest.mark.parametrize(
    "links",
    [
        {"ifindex": 1},
        ["lo"],
        [{"ifindex": 2, "ifname": "eth0", "addr_info": [{"family": "inet", "scope": "global"}]}],
        [{"ifindex": 2, "ifname": "eth0", "addr_info": [{"local": "10.0.0.5", "scope": "global"}]}],
        [{"ifindex": 2, "ifname": "eth0", "addr_info": [{"family": "inet", "local": "10.0.0.5"}]}],
        [{"ifindex": 2, "ifname": "eth0", "addr_info": "10.0.0.5"}],
    ],
    ids=["not-a-list", "link-not-object", "no-local", "no-family", "no-scope", "addr-info-not-list"],
)
def test_host_addresses_rejects_a_wrong_shape(links: object):
    with pytest.raises(exposure.SourceUnreadable):
        exposure.host_addresses(links)


def test_sockaddr_is_a_pair_for_ipv4_and_a_scoped_quad_for_ipv6():
    v4 = exposure.HostAddress(ip="10.0.0.5", family=socket.AF_INET, scope_id=0, ifname="eth0")
    v6 = exposure.HostAddress(ip="fe80::1", family=socket.AF_INET6, scope_id=7, ifname="eth0")
    assert v4.sockaddr(8002) == ("10.0.0.5", 8002)
    assert v6.sockaddr(8002) == ("fe80::1", 8002, 0, 7)


# ---------------------------------------------------------------------------
# The static table is pinned to its homes (SPOT / INV-5)
# ---------------------------------------------------------------------------


def _published_host_port(service: str, entry: object) -> int:
    if isinstance(entry, dict) and "published" in entry:
        return int(entry["published"])
    if isinstance(entry, str) and entry.count(":") >= 1:
        host_port = entry.split(":")[-2]
        if host_port.isdigit():
            return int(host_port)
    pytest.fail(f"compose service {service!r} has a ports entry this pin cannot read: {entry!r}")


def _compose_published_ports() -> dict[int, str]:
    services = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))["services"]
    return {
        _published_host_port(name, entry): name
        for name, service in services.items()
        for entry in service.get("ports", [])
    }


def _field_default(class_name: str, field_name: str) -> object:
    """The ``default=`` of the ``Field(...)`` assigned to *field_name* in *class_name*."""
    tree = ast.parse(SCHEMA_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for statement in node.body:
            if (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == field_name
                and isinstance(statement.value, ast.Call)
            ):
                for keyword in statement.value.keywords:
                    if keyword.arg == "default":
                        return ast.literal_eval(keyword.value)
    pytest.fail(f"{SCHEMA_PATH}::{class_name}.{field_name} has no Field(default=...) to pin against")


def test_compose_table_equals_every_compose_publish():
    assert _compose_published_ports() == dict(exposure.COMPOSE_PUBLISHED_PORTS)


def test_recon_report_port_equals_the_server_config_default():
    assert exposure.RECON_REPORT_PORT == _field_default("ServerConfig", "recon_report_port")


def test_config_yaml_does_not_move_the_recon_report_port():
    server = yaml.safe_load(FM_CONFIG_PATH.read_text(encoding="utf-8"))["server"]
    assert server.get("recon_report_port", exposure.RECON_REPORT_PORT) == exposure.RECON_REPORT_PORT


def test_static_ports_owns_one_port_per_table_entry():
    compose = {
        exposure.OwnedPort(port=port, owner=f"compose service {service}")
        for port, service in exposure.COMPOSE_PUBLISHED_PORTS.items()
    }
    recon = {
        owned for owned in exposure.STATIC_PORTS if owned.port == exposure.RECON_REPORT_PORT
    }
    assert len(exposure.STATIC_PORTS) == len(exposure.COMPOSE_PUBLISHED_PORTS) + 1
    assert set(exposure.STATIC_PORTS) == compose | recon
    assert [owned.owner for owned in recon] == [
        "fused-memory recon-report (ServerConfig.recon_report_port)"
    ]


def test_merge_owners_names_every_owner_of_a_port_once_in_source_order():
    sockets = [
        exposure.OwnedPort(port=8002, owner="fused-memory.socket"),
        exposure.OwnedPort(port=8002, owner="fused-memory.socket"),
        exposure.OwnedPort(port=8103, owner="fused-memory.socket"),
    ]
    static = [
        exposure.OwnedPort(port=6379, owner="compose service falkordb"),
        exposure.OwnedPort(port=8002, owner="compose service fused-mcp"),
    ]
    assert exposure.merge_owners(sockets, static) == {
        6379: "compose service falkordb",
        8002: "fused-memory.socket, compose service fused-mcp",
        8103: "fused-memory.socket",
    }
