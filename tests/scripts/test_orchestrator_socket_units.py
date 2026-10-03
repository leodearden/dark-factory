"""Every orchestrator's escalation port is held by a systemd socket unit.

`orchestrator-<project>.socket` keeps the escalation MCP port bound while the
service restarts, so Claude Code's HTTP MCP client (which gives up for good
after ~15s of refused connections) survives the restart. Every committed
orchestrator service has one. For units the watchdog covers, the port must be
the one scripts/orchestrator-watchdog.py::WATCHED gives that unit — itself
pinned to every project's escalation.port by
tests/scripts/test_orchestrator_watchdog.py.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest
from systemd_unit_invariants import (
    ALL_ORCHESTRATOR_SERVICE_FILES,
    REPO_ROOT,
    parse_sections,
)

SCRIPTS = REPO_ROOT / "scripts"
HOOK = "ExecStopPost=-/home/leo/src/dark-factory/scripts/stop-socket-unless-restarting.sh %n"


def _watched() -> list[tuple[int, str]]:
    spec = importlib.util.spec_from_file_location(
        "orchestrator_watchdog", SCRIPTS / "orchestrator-watchdog.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.WATCHED)


WATCHED = _watched()


def _directives(path: pathlib.Path, section: str) -> list[str]:
    lines = parse_sections(path.read_text(encoding="utf-8"))[section]
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


SERVICES = sorted(
    p.name for p in ALL_ORCHESTRATOR_SERVICE_FILES if p.name != "orchestrator-watchdog.service"
)


def test_every_orchestrator_service_and_only_those_have_a_socket_unit() -> None:
    assert {unit for _, unit in WATCHED} <= set(SERVICES)
    sockets = {p.name for p in SCRIPTS.glob("orchestrator-*.socket.template")}
    assert sockets == {unit.removesuffix(".service") + ".socket.template" for unit in SERVICES}


@pytest.mark.parametrize(("port", "unit"), WATCHED, ids=[u for _, u in WATCHED])
def test_the_socket_holds_the_watched_escalation_port(port: int, unit: str) -> None:
    socket_unit = SCRIPTS / (unit.removesuffix(".service") + ".socket.template")
    assert _directives(socket_unit, "Socket") == [f"ListenStream=127.0.0.1:{port}"]


@pytest.mark.parametrize("unit", SERVICES)
def test_the_service_is_wired_to_its_socket(unit: str) -> None:
    service = SCRIPTS / unit
    socket_name = unit.removesuffix(".service") + ".socket"
    unit_section = _directives(service, "Unit")
    after = [d.split("=", 1)[1].split() for d in unit_section if d.startswith("After=")]
    assert any(socket_name in names for names in after)
    assert f"Wants={socket_name}" in unit_section
    assert [d for d in _directives(service, "Service") if d.startswith("ExecStopPost=")] == [HOOK]
    assert f"Also={socket_name}" in _directives(service, "Install")
