"""scripts/falkordb_aof_gate.py: the fused-memory unit's FalkorDB compose-up, gated on AOF liveness (task 6330).

Docker is a fake runner that answers the gate's read commands from canned
state and records every command; the host side of the bind mount is a tmp dir.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from falkordb_aof_gate import CommandResult, main

COMPOSE = "/repo/fused-memory/docker/docker-compose.yml"
CONTAINER = "dad0940ab28a"
UP_ALL = ("docker", "compose", "-f", COMPOSE, "up", "-d", "falkordb", "qdrant")
UP_QDRANT = ("docker", "compose", "-f", COMPOSE, "up", "-d", "qdrant")
START_EXISTING = ("docker", "start", CONTAINER)
ACTIONS = {UP_ALL, UP_QDRANT, START_EXISTING}
BASE_SIZE = 3000
INCR_SIZE = 500


def _info(**overrides: str) -> str:
    fields = {
        "loading": "0",
        "aof_enabled": "1",
        "aof_rewrite_in_progress": "0",
        "aof_last_write_status": "ok",
        "aof_current_size": str(BASE_SIZE + INCR_SIZE),
    }
    fields.update(overrides)
    return "# Persistence\r\n" + "".join(f"{key}:{value}\r\n" for key, value in fields.items())


def _write_aof(data_dir: Path, *, incr_size: int = INCR_SIZE) -> None:
    aof_dir = data_dir / "appendonlydir"
    aof_dir.mkdir(parents=True)
    (aof_dir / "appendonly.aof.13.base.rdb").write_bytes(b"b" * BASE_SIZE)
    (aof_dir / "appendonly.aof.13.incr.aof").write_bytes(b"i" * incr_size)
    (aof_dir / "appendonly.aof.manifest").write_text(
        "file appendonly.aof.13.base.rdb seq 13 type b\n"
        "file appendonly.aof.13.incr.aof seq 13 type i startoffset 0\n"
        "file appendonly.aof.12.incr.aof seq 12 type h\n"
    )


class FakeDocker:
    def __init__(
        self,
        data_dir: Path,
        *,
        status: str | None = "running",
        appendonly: str = "yes",
        infos: Sequence[str] = (),
        mounted: bool = True,
        redis_cli_fails: bool = False,
        ps_fails: bool = False,
        action_rc: int = 0,
    ):
        self.data_dir = data_dir
        self.status = status
        self.appendonly = appendonly
        self.infos = list(infos) or [_info(), _info()]
        self.mounted = mounted
        self.redis_cli_fails = redis_cli_fails
        self.ps_fails = ps_fails
        self.action_rc = action_rc
        self.commands: list[tuple[str, ...]] = []

    @property
    def actions(self) -> list[tuple[str, ...]]:
        return [command for command in self.commands if command in ACTIONS]

    def __call__(self, argv: Sequence[str]) -> CommandResult:
        command = tuple(argv)
        self.commands.append(command)
        if command in ACTIONS:
            return CommandResult(self.action_rc, "", "")
        if command[4:] == ("ps", "-a", "-q", "falkordb"):
            if self.ps_fails:
                return CommandResult(1, "", "Cannot connect to the Docker daemon")
            return CommandResult(0, f"{CONTAINER}\n" if self.status else "", "")
        if command[4:] == ("config", "--format", "json"):
            volume = {"type": "bind", "source": str(self.data_dir), "target": "/var/lib/falkordb/data"}
            return CommandResult(0, json.dumps({"services": {"falkordb": {"volumes": [volume]}}}), "")
        if command[:2] == ("docker", "inspect"):
            target = "/var/lib/falkordb/data" if self.mounted else "/elsewhere"
            mount = {"Type": "bind", "Source": str(self.data_dir), "Destination": target}
            return CommandResult(0, json.dumps([{"State": {"Status": self.status}, "Mounts": [mount]}]), "")
        if command[:3] == ("docker", "exec", CONTAINER):
            return self._redis_cli(command[3:])
        raise AssertionError(f"unexpected docker command {command!r}")

    def _redis_cli(self, args: tuple[str, ...]) -> CommandResult:
        if self.redis_cli_fails:
            return CommandResult(1, "", "LOADING Redis is loading the dataset in memory")
        if args[:4] == ("redis-cli", "--json", "CONFIG", "GET"):
            config = {
                "appendonly": self.appendonly,
                "dir": "/var/lib/falkordb/data",
                "appenddirname": "appendonlydir",
                "appendfilename": "appendonly.aof",
            }
            return CommandResult(0, json.dumps(config), "")
        if args == ("redis-cli", "INFO", "persistence"):
            return CommandResult(0, self.infos.pop(0), "")
        raise AssertionError(f"unexpected redis-cli call {args!r}")


def _gate(docker: FakeDocker, *extra: str) -> int:
    return main(["--compose-file", COMPOSE, *extra], run=docker)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "falkordb"


def test_a_running_container_with_a_current_aof_is_recreated(data_dir, capsys):
    _write_aof(data_dir)
    docker = FakeDocker(data_dir)

    assert _gate(docker) == 0

    assert docker.actions == [UP_ALL]
    assert "AOF current, recreate allowed" in capsys.readouterr().err


def test_an_aof_that_grew_between_the_two_info_reads_is_current(data_dir):
    _write_aof(data_dir)
    total = BASE_SIZE + INCR_SIZE
    docker = FakeDocker(
        data_dir, infos=[_info(aof_current_size=str(total - 10)), _info(aof_current_size=str(total + 10))],
    )

    assert _gate(docker) == 0
    assert docker.actions == [UP_ALL]


@pytest.mark.parametrize(
    ("label", "docker_kwargs", "incr_size"),
    [
        ("appendonly off", {"appendonly": "no"}, INCR_SIZE),
        ("last write failed", {"infos": [_info(aof_last_write_status="err")] * 2}, INCR_SIZE),
        ("rewrite in progress", {"infos": [_info(), _info(aof_rewrite_in_progress="1")]}, INCR_SIZE),
        ("still loading", {"infos": [_info(loading="1")] * 2}, INCR_SIZE),
        ("aof not enabled", {"infos": [_info(aof_enabled="0")] * 2}, INCR_SIZE),
        ("host incr behind the server", {}, INCR_SIZE - 1),
        ("data dir not host-mounted", {"mounted": False}, INCR_SIZE),
        ("redis-cli unreadable", {"redis_cli_fails": True}, INCR_SIZE),
    ],
)
def test_a_running_container_without_a_current_aof_is_left_alone(
    data_dir, capsys, label, docker_kwargs, incr_size,
):
    _write_aof(data_dir, incr_size=incr_size)
    docker = FakeDocker(data_dir, **docker_kwargs)

    assert _gate(docker) == 0, label

    assert docker.actions == [UP_QDRANT], label
    err = capsys.readouterr().err
    assert "recreate REFUSED" in err, label
    assert "CONFIG SET appendonly yes" in err, label


def test_a_running_container_with_no_manifest_on_the_host_is_left_alone(data_dir):
    data_dir.mkdir()
    docker = FakeDocker(data_dir)

    assert _gate(docker) == 0
    assert docker.actions == [UP_QDRANT]


def test_a_stopped_container_is_started_as_it_is_never_recreated(data_dir, capsys):
    docker = FakeDocker(data_dir, status="exited")

    assert _gate(docker) == 0

    assert docker.actions == [START_EXISTING, UP_QDRANT]
    assert "next fused-memory restart" in capsys.readouterr().err


@pytest.mark.parametrize("status", ["restarting", "paused", "dead", "removing"])
def test_an_unsettled_container_is_left_alone(data_dir, capsys, status):
    docker = FakeDocker(data_dir, status=status)

    assert _gate(docker) == 0

    assert docker.actions == [UP_QDRANT]
    assert "recreate REFUSED" in capsys.readouterr().err


def test_no_container_and_no_persisted_data_creates_it(data_dir):
    docker = FakeDocker(data_dir, status=None)

    assert _gate(docker) == 0
    assert docker.actions == [UP_ALL]


@pytest.mark.parametrize("artifact", ["dump.rdb", "appendonlydir"])
def test_no_container_over_persisted_data_is_not_created(data_dir, capsys, artifact):
    data_dir.mkdir()
    if artifact == "appendonlydir":
        (data_dir / artifact).mkdir()
    else:
        (data_dir / artifact).write_bytes(b"x")
    docker = FakeDocker(data_dir, status=None)

    assert _gate(docker) == 0

    assert docker.actions == [UP_QDRANT]
    assert "create REFUSED" in capsys.readouterr().err


def test_an_unreachable_docker_touches_no_falkordb(data_dir):
    docker = FakeDocker(data_dir, ps_fails=True)

    _gate(docker)

    assert docker.actions == [UP_QDRANT]


def test_a_failed_compose_up_fails_the_gate(data_dir):
    _write_aof(data_dir)
    docker = FakeDocker(data_dir, action_rc=1)

    assert _gate(docker) == 1


@pytest.mark.parametrize(
    ("docker_kwargs", "expected_rc"), [({}, 0), ({"appendonly": "no"}, 1)],
)
def test_check_only_reports_the_verdict_and_acts_on_nothing(data_dir, capsys, docker_kwargs, expected_rc):
    _write_aof(data_dir)
    docker = FakeDocker(data_dir, **docker_kwargs)

    assert _gate(docker, "--check-only") == expected_rc

    assert docker.actions == []
    assert capsys.readouterr().err


def test_under_systemd_a_refusal_is_logged_at_error_priority(data_dir, capsys, monkeypatch):
    monkeypatch.setenv("JOURNAL_STREAM", "8:12345")
    _write_aof(data_dir)

    _gate(FakeDocker(data_dir, appendonly="no"))

    refusal = [line for line in capsys.readouterr().err.splitlines() if "REFUSED" in line]
    assert refusal and all(line.startswith("<3>") for line in refusal)
