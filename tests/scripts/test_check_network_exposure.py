"""Tests for scripts/check_network_exposure.py (contract: plans/mcp-network-exposure-prd.md §3).

The structured-source parsers are driven with JSON shaped like the live
`systemctl --user list-sockets -o json` and `ip -j addr` outputs.
"""

from __future__ import annotations

import ast
import collections.abc
import io
import json
import os
import pathlib
import shlex
import shutil
import socket
import subprocess
import sys
import time

import check_network_exposure as exposure
import pytest
import yaml
from setup_host_sections import checker_repo, run_section, slice_section, usage_error_checker
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from shell_sections import stub_bin_dir, write_stub

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = pathlib.Path(__file__).parents[2]
COMPOSE_PATH = REPO_ROOT / "fused-memory" / "docker" / "docker-compose.yml"
SCHEMA_PATH = REPO_ROOT / "fused-memory" / "src" / "fused_memory" / "config" / "schema.py"
FM_CONFIG_PATH = REPO_ROOT / "fused-memory" / "config" / "config.yaml"
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_network_exposure.py"
REPORT_TAG = "[network_exposure]"

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
    assert _field_default("ServerConfig", "recon_report_port") == exposure.RECON_REPORT_PORT


def test_config_yaml_does_not_move_the_recon_report_port():
    server = yaml.safe_load(FM_CONFIG_PATH.read_text(encoding="utf-8"))["server"]
    assert server.get("recon_report_port", exposure.RECON_REPORT_PORT) == exposure.RECON_REPORT_PORT


def test_static_ports_owns_one_port_per_table_entry():
    assert sorted(owned.port for owned in exposure.STATIC_PORTS) == sorted(
        [*exposure.COMPOSE_PUBLISHED_PORTS, exposure.RECON_REPORT_PORT]
    )
    owner_of = {owned.port: owned.owner for owned in exposure.STATIC_PORTS}
    for port, service in exposure.COMPOSE_PUBLISHED_PORTS.items():
        assert service in owner_of[port]
    assert "ServerConfig.recon_report_port" in owner_of[exposure.RECON_REPORT_PORT]


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
        6379: ("compose service falkordb",),
        8002: ("fused-memory.socket", "compose service fused-mcp"),
        8103: ("fused-memory.socket",),
    }


# ---------------------------------------------------------------------------
# run(): behavioural probes against in-process listeners, fake sources on PATH
# ---------------------------------------------------------------------------

FINDING_TOKENS = (
    "[exposed]",
    "[timed-out]",
    "[unverified]",
    "[not-listening]",
    "[source-error]",
    "[finding]",
)


class FakeHost:
    """Fake `systemctl` / `ip` on PATH, each printing a JSON payload the test chose."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.bin_dir = stub_bin_dir(root)

    def serve(self, command: str, payload: object) -> None:
        self.serve_bytes(command, json.dumps(payload).encode("utf-8"))

    def serve_bytes(self, command: str, output: bytes) -> None:
        data = self.root / f"{command}.out"
        data.write_bytes(output)
        argv_log = shlex.quote(str(self.root / f"{command}.argv"))
        write_stub(self.bin_dir, command, f'printf "%s\\n" "$*" >> {argv_log}\ncat {shlex.quote(str(data))}\n')

    def stub(self, command: str, body: str) -> None:
        write_stub(self.bin_dir, command, body)

    def argv(self, command: str) -> list[str]:
        return (self.root / f"{command}.argv").read_text(encoding="utf-8").splitlines()


def _fake_links(*extra: dict[str, object]) -> list[dict[str, object]]:
    return [
        _link(1, "lo", _addr("inet", "127.0.0.1", "host"), _addr("inet6", "::1", "host")),
        _link(2, "fake0", _addr("inet", "127.0.0.2", "global"), *extra),
    ]


def _row(port: int, unit: str, host: str = "127.0.0.1") -> dict[str, str]:
    return {"listen": f"{host}:{port}", "unit": unit, "activates": unit.replace(".socket", ".service")}


@pytest.fixture
def fake_host(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> FakeHost:
    host = FakeHost(tmp_path)
    host.serve("systemctl", [])
    host.serve("ip", _fake_links())
    monkeypatch.setenv("PATH", f"{host.bin_dir}:{os.environ['PATH']}")
    return host


@pytest.fixture
def held() -> collections.abc.Iterator[list[socket.socket]]:
    sockets: list[socket.socket] = []
    yield sockets
    for sock in sockets:
        sock.close()


def _listen(held: list[socket.socket], host: str) -> int:
    server = socket.create_server((host, 0))
    held.append(server)
    return server.getsockname()[1]


def _bound_not_listening(held: list[socket.socket]) -> int:
    """A port that REFUSES for the whole test: held bound, never listening, no SO_REUSEADDR."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    held.append(sock)
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


def _full_backlog(held: list[socket.socket]) -> int:
    """A wildcard port whose accept queue is full, so the next connect times out."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    held.append(server)
    server.bind(("0.0.0.0", 0))
    server.listen(0)
    port = server.getsockname()[1]
    held.append(socket.create_connection(("127.0.0.1", port)))
    return port


def _run(
    static_ports: collections.abc.Sequence[exposure.OwnedPort] = (),
    connect_timeout: float = 1.0,
) -> tuple[exposure.ExposureExit, list[str]]:
    out = io.StringIO()
    verdict = exposure.run(static_ports=static_ports, connect_timeout=connect_timeout, out=out)
    lines = out.getvalue().splitlines()
    assert lines, "run() printed nothing"
    untagged = [line for line in lines if not line.startswith("[network_exposure] ")]
    assert not untagged, f"untagged lines: {untagged}"
    return verdict, lines


def _line_index(lines: list[str], *needles: str) -> int:
    matches = [i for i, line in enumerate(lines) if all(needle in line for needle in needles)]
    assert len(matches) == 1, f"expected one line containing {needles}, got {matches} in {lines}"
    return matches[0]


def _findings(lines: list[str]) -> list[str]:
    return [line for line in lines if any(token in line for token in FINDING_TOKENS)]


def test_run_reports_loopback_only_for_a_loopback_listener(fake_host: FakeHost, held):
    port = _listen(held, "127.0.0.1")
    fake_host.serve("systemctl", [_row(port, "orchestrator-x.socket")])

    verdict, lines = _run()

    assert verdict == exposure.ExposureExit.LOOPBACK_ONLY == 0
    assert _findings(lines) == []
    _line_index(lines, "[ok]")
    _line_index(lines, "probing 1 port(s) on 1 non-loopback address(es) and 127.0.0.1")
    assert fake_host.argv("systemctl") == ["--user list-sockets -o json"]


def test_run_names_an_exposed_socket_port_its_address_and_owner(fake_host: FakeHost, held):
    port = _listen(held, "0.0.0.0")
    fake_host.serve("systemctl", [_row(port, "fused-memory.socket", host="0.0.0.0")])

    verdict, lines = _run()

    assert verdict == exposure.ExposureExit.FINDING == 1
    _line_index(lines, "[exposed]", f"port {port}", "127.0.0.2", "fused-memory.socket")
    assert len(_findings(lines)) == 2, lines  # the [exposed] line and the [finding] verdict


def test_run_probes_and_owns_static_ports(fake_host: FakeHost, held):
    port = _listen(held, "0.0.0.0")

    verdict, lines = _run(static_ports=(exposure.OwnedPort(port, "compose service falkordb"),))

    assert verdict == exposure.ExposureExit.FINDING
    _line_index(lines, "[exposed]", f"port {port}", "127.0.0.2", "compose service falkordb")


def test_run_names_every_owner_of_a_port(fake_host: FakeHost, held):
    port = _listen(held, "0.0.0.0")
    fake_host.serve("systemctl", [_row(port, "fused-memory.socket", host="0.0.0.0")])

    verdict, lines = _run(static_ports=(exposure.OwnedPort(port, "compose service fused-mcp"),))

    assert verdict == exposure.ExposureExit.FINDING
    _line_index(lines, "[exposed]", f"port {port}", "fused-memory.socket, compose service fused-mcp")


def test_run_fails_a_port_nothing_accepts_on_loopback(fake_host: FakeHost, held):
    port = _bound_not_listening(held)
    fake_host.serve("systemctl", [_row(port, "orchestrator-down.socket")])

    verdict, lines = _run()

    assert verdict == exposure.ExposureExit.FINDING
    _line_index(lines, "[not-listening]", f"port {port}", "orchestrator-down.socket", "127.0.0.1")
    assert not [line for line in lines if "[exposed]" in line]


def test_run_names_a_timeout_as_a_failure(fake_host: FakeHost, held):
    port = _full_backlog(held)
    fake_host.serve("systemctl", [_row(port, "fused-memory.socket", host="0.0.0.0")])

    verdict, lines = _run(connect_timeout=0.3)

    assert verdict == exposure.ExposureExit.FINDING
    _line_index(lines, "[timed-out]", f"port {port}", "127.0.0.2")


def test_run_probes_ipv6_addresses(fake_host: FakeHost, held):
    if not socket.has_dualstack_ipv6():
        pytest.skip("this host cannot bind a dual-stack IPv6 socket")
    try:
        server = socket.create_server(("::", 0), family=socket.AF_INET6, dualstack_ipv6=True)
    except OSError as exc:
        pytest.skip(f"this host cannot bind a dual-stack IPv6 socket: {exc}")
    held.append(server)
    port = server.getsockname()[1]
    try:
        socket.create_connection(("::1", port), timeout=1.0).close()
    except OSError as exc:
        pytest.skip(f"this host cannot connect to ::1: {exc}")
    fake_host.serve("ip", _fake_links(_addr("inet6", "::1", "global")))
    fake_host.serve("systemctl", [_row(port, "orchestrator-v6.socket", host="[::]")])

    verdict, lines = _run()

    assert verdict == exposure.ExposureExit.FINDING
    _line_index(lines, "[exposed]", f"port {port}", "::1 (fake0)")


def test_run_reports_an_address_it_could_not_probe_as_unverified(fake_host: FakeHost, held):
    port = _listen(held, "127.0.0.1")
    fake_host.serve("ip", [_link(99999, "ghost0", _addr("inet6", "fe80::1", "link"))])
    fake_host.serve("systemctl", [_row(port, "orchestrator-x.socket")])

    verdict, lines = _run()

    assert verdict == exposure.ExposureExit.FINDING
    _line_index(lines, "[unverified]", f"port {port}", "fe80::1%ghost0")


def test_run_reads_a_source_whose_output_is_not_utf8(fake_host: FakeHost, held):
    port = _listen(held, "127.0.0.1")
    fake_host.serve("systemctl", [_row(port, "orchestrator-x.socket")])
    fake_host.serve_bytes("ip", json.dumps(_fake_links()).encode("utf-8").replace(b"fake0", b"fake\xff0"))

    verdict, lines = _run()

    assert verdict == exposure.ExposureExit.LOOPBACK_ONLY, lines
    _line_index(lines, "probing 1 port(s) on 1 non-loopback address(es)")


def _systemctl_fails(host: FakeHost, monkeypatch: pytest.MonkeyPatch) -> None:
    host.stub("systemctl", 'echo "Failed to connect to bus: No medium found" >&2\nexit 1\n')


def _ip_prints_garbage(host: FakeHost, monkeypatch: pytest.MonkeyPatch) -> None:
    host.stub("ip", "echo 'not json'\n")


def _ip_absent(host: FakeHost, monkeypatch: pytest.MonkeyPatch) -> None:
    """PATH becomes the fake bin dir alone, holding systemctl and what its stub needs."""
    (host.bin_dir / "ip").unlink()
    for tool in ("bash", "cat"):
        found = shutil.which(tool)
        assert found is not None, f"{tool} is not on PATH"
        (host.bin_dir / tool).symlink_to(found)
    monkeypatch.setenv("PATH", str(host.bin_dir))


@pytest.mark.parametrize(
    ("break_source", "command", "reason"),
    [
        (_systemctl_fails, "systemctl --user list-sockets -o json", "Failed to connect to bus"),
        (_ip_prints_garbage, "ip -j addr", "not JSON"),
        (_ip_absent, "ip -j addr", "not found"),
    ],
    ids=["systemctl-exits-1", "ip-prints-non-json", "ip-absent"],
)
def test_run_reports_an_unreadable_source_and_still_probes_the_rest(
    fake_host: FakeHost, held, monkeypatch: pytest.MonkeyPatch, break_source, command: str, reason: str
):
    port = _bound_not_listening(held)
    break_source(fake_host, monkeypatch)

    verdict, lines = _run(static_ports=(exposure.OwnedPort(port, "compose service falkordb"),))

    assert verdict == exposure.ExposureExit.FINDING
    _line_index(lines, "[source-error]", command, reason)
    _line_index(lines, "probing 1 port(s)")
    _line_index(lines, "[not-listening]", f"port {port}", "127.0.0.1")


def test_run_reports_findings_in_ascending_port_order(fake_host: FakeHost, held):
    low, high = sorted((_listen(held, "0.0.0.0"), _listen(held, "0.0.0.0")))
    fake_host.serve(
        "systemctl",
        [_row(high, "orchestrator-b.socket", "0.0.0.0"), _row(low, "orchestrator-a.socket", "0.0.0.0")],
    )

    verdict, lines = _run()

    assert verdict == exposure.ExposureExit.FINDING
    assert _line_index(lines, "[exposed]", f"port {low}") < _line_index(lines, "[exposed]", f"port {high}")


# ---------------------------------------------------------------------------
# The CLI's exit contract: 0/1 always tagged, 2 never tagged
# ---------------------------------------------------------------------------


def _cli(tmp_path: pathlib.Path, *args: str, script: pathlib.Path = SCRIPT_PATH) -> tuple[int, str]:
    """Run the real script with systemctl printing no rows and ip printing garbage.

    The unreadable `ip` source makes the exit 1 whatever this host's ports are.
    """
    host = FakeHost(tmp_path)
    host.serve("systemctl", [])
    host.stub("ip", "echo 'not json'\n")
    env = {**os.environ, "PATH": f"{host.bin_dir}:{os.environ['PATH']}"}
    result = subprocess.run(
        [sys.executable, str(script), *args], capture_output=True, text=True, env=env, timeout=120
    )
    return result.returncode, result.stdout + result.stderr


def _assert_every_line_tagged(output: str) -> None:
    lines = output.splitlines()
    assert lines, "the CLI printed nothing"
    untagged = [line for line in lines if not line.startswith(f"{REPORT_TAG} ")]
    assert not untagged, f"untagged lines: {untagged}"


def test_cli_reports_a_source_error_tagged_and_exits_1(tmp_path: pathlib.Path):
    rc, output = _cli(tmp_path)

    assert rc == exposure.ExposureExit.FINDING, output
    _assert_every_line_tagged(output)
    _line_index(output.splitlines(), "[source-error]", "ip -j addr")


def test_cli_connect_timeout_bounds_each_connect(fake_host: FakeHost, held, capsys: pytest.CaptureFixture[str]):
    port = _full_backlog(held)
    fake_host.serve("systemctl", [_row(port, "fused-memory.socket", host="0.0.0.0")])
    longer_than_default = exposure.DEFAULT_CONNECT_TIMEOUT_SECS + 0.5

    started = time.monotonic()
    rc = exposure.main(["--connect-timeout", str(longer_than_default)])
    elapsed = time.monotonic() - started

    assert rc == exposure.ExposureExit.FINDING
    _line_index(capsys.readouterr().out.splitlines(), "[timed-out]", f"port {port}", "127.0.0.2")
    assert elapsed >= longer_than_default, f"the connect gave up after {elapsed:.2f} s"


@pytest.mark.parametrize(
    "args", [("--connect-timeout", "soon"), ("--bogus",)], ids=["non-numeric-timeout", "unknown-flag"]
)
def test_cli_usage_error_exits_2_without_the_tag(tmp_path: pathlib.Path, args: tuple[str, ...]):
    rc, output = _cli(tmp_path, *args)

    assert rc == 2, output
    assert REPORT_TAG not in output, f"an argparse usage error must not look like a report:\n{output}"


def test_cli_missing_script_exits_2_without_the_tag(tmp_path: pathlib.Path):
    rc, output = _cli(tmp_path, script=tmp_path / "check_network_exposure_RENAMED.py")

    assert rc == 2, output
    assert REPORT_TAG not in output, f"a missing script must not look like a report:\n{output}"


def test_script_is_executable():
    assert os.access(SCRIPT_PATH, os.X_OK), f"{SCRIPT_PATH} is not executable"


# ---------------------------------------------------------------------------
# setup-host.sh §11: the status is believed only with the tag
# ---------------------------------------------------------------------------

_GATE_START = "_net_exposure_script="
_GATE_END = "\nfi\n"


def _stub_checker(report: str, status: int) -> str:
    return f"import sys\nprint({report!r})\nsys.exit({status})\n"


def _run_gate(
    tmp_path: pathlib.Path, *, body: str | None, with_checker: bool = True
) -> subprocess.CompletedProcess:
    repo = checker_repo(tmp_path, SCRIPT_PATH.name, body=body, with_checker=with_checker)
    result = run_section(
        tmp_path, slice_section(_GATE_START, _GATE_END), repo_root=repo, unit_dir=tmp_path / "units"
    )
    assert result.returncode == 0, (
        f"fail() only printfs; the health-check section must not abort.\n{result.stdout}\n{result.stderr}"
    )
    return result


def test_gate_reports_ok_for_a_tagged_loopback_only_report(tmp_path: pathlib.Path):
    result = _run_gate(tmp_path, body=_stub_checker(f"{REPORT_TAG} [ok] loopback only", 0))

    assert "OK " in result.stdout, result.stdout
    assert "FAIL " not in result.stdout, result.stdout


def test_gate_fails_on_a_tagged_finding_and_reprints_the_report(tmp_path: pathlib.Path):
    report = f"{REPORT_TAG} [exposed] port 8002 (fused-memory.socket) accepted a connect on 192.168.1.105"
    result = _run_gate(tmp_path, body=_stub_checker(report, 1))

    assert report in result.stdout, result.stdout
    assert f"FAIL   see the {REPORT_TAG} report above." in result.stdout, result.stdout
    assert "OK " not in result.stdout, result.stdout


def test_gate_fails_on_a_tagged_exit_2(tmp_path: pathlib.Path):
    result = _run_gate(tmp_path, body=_stub_checker(f"{REPORT_TAG} [ok] loopback only", 2))

    assert "FAIL " in result.stdout, result.stdout
    assert "OK " not in result.stdout, result.stdout


def test_gate_does_not_believe_an_untagged_report(tmp_path: pathlib.Path):
    result = _run_gate(tmp_path, body=_stub_checker("[ok] loopback only", 0))

    assert f"produced no {REPORT_TAG} report" in result.stdout, result.stdout
    assert "OK " not in result.stdout, result.stdout


def test_gate_fails_on_a_usage_error(tmp_path: pathlib.Path):
    body = usage_error_checker(SCRIPT_PATH.name, "[-h] [--connect-timeout SECS]", "--connect-timeout")
    result = _run_gate(tmp_path, body=body)

    assert f"produced no {REPORT_TAG} report" in result.stdout, result.stdout
    assert "OK " not in result.stdout, result.stdout


def test_gate_fails_when_the_checker_is_missing(tmp_path: pathlib.Path):
    result = _run_gate(tmp_path, body=None, with_checker=False)

    missing = tmp_path / "repo" / "scripts" / SCRIPT_PATH.name
    assert f"FAIL Network exposure check missing: {missing}" in result.stdout, result.stdout
    assert "OK " not in result.stdout, result.stdout
