"""scripts/falkordb_aof_gate.py: the fused-memory unit's FalkorDB compose-up, gated on AOF liveness (task 6330).

Docker is a fake runner that answers the gate's read commands from canned
state and records every command; the host side of the bind mount is a tmp dir.
Like real Redis, the fake's INFO omits the aof_* size fields while aof_enabled:0.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from falkordb_aof_gate import CommandResult, main

COMPOSE = "/repo/fused-memory/docker/docker-compose.yml"
CONTAINER = "dad0940ab28a"
DATA_DIR_IN_CONTAINER = "/var/lib/falkordb/data"
UP_ALL = ("docker", "compose", "-f", COMPOSE, "up", "-d", "falkordb", "qdrant")
UP_QDRANT = ("docker", "compose", "-f", COMPOSE, "up", "-d", "qdrant")
START_EXISTING = ("docker", "start", CONTAINER)
ACTIONS = {UP_ALL, UP_QDRANT, START_EXISTING}
AOF_REDIS_ARGS = "--appendonly yes --appendfsync everysec"
BASE_SIZE = 3000
INCR_SIZE = 500

ENABLE_AOF = "CONFIG SET appendonly yes"
WAIT = "wait until"
FIX_DISK = "disk"
USE_MAIN_COMPOSE = "main checkout"
BY_HAND = "by hand"
BOOTSTRAP = "Bootstrapping FalkorDB's AOF from dump.rdb"
SETTLE = "settle"


def _info(**overrides: str) -> str:
    fields = {
        "loading": "0",
        "aof_enabled": "1",
        "aof_rewrite_in_progress": "0",
        "aof_last_write_status": "ok",
        "aof_current_size": str(BASE_SIZE + INCR_SIZE),
        "aof_base_size": str(BASE_SIZE),
    }
    fields.update(overrides)
    if fields["aof_enabled"] == "0":
        fields.pop("aof_current_size")
        fields.pop("aof_base_size")
    return "# Persistence\r\n" + "".join(f"{key}:{value}\r\n" for key, value in fields.items())


AOF_OFF_INFOS = [_info(aof_enabled="0")] * 2


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
        compose_data_dir: Path | None = None,
        redis_args: str | None = AOF_REDIS_ARGS,
        redis_cli_fails: bool = False,
        exec_raises: Exception | None = None,
        ps_fails: bool = False,
        action_rc: int = 0,
    ):
        self.data_dir = data_dir
        self.status = status
        self.appendonly = appendonly
        self.infos = list(infos) or [_info(), _info()]
        self.mounted = mounted
        self.compose_data_dir = compose_data_dir or data_dir
        self.redis_args = redis_args
        self.redis_cli_fails = redis_cli_fails
        self.exec_raises = exec_raises
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
        if command[4:] == ("config", "--no-env-resolution", "--format", "json"):
            volume = {"type": "bind", "source": str(self.compose_data_dir), "target": DATA_DIR_IN_CONTAINER}
            return CommandResult(0, json.dumps({"services": {"falkordb": {"volumes": [volume]}}}), "")
        if command[:2] == ("docker", "inspect"):
            return CommandResult(0, json.dumps([self._inspect()]), "")
        if command[:3] == ("docker", "exec", CONTAINER):
            if self.exec_raises is not None:
                raise self.exec_raises
            return self._redis_cli(command[3:])
        raise AssertionError(f"unexpected docker command {command!r}")

    def _inspect(self) -> dict:
        target = DATA_DIR_IN_CONTAINER if self.mounted else "/elsewhere"
        env = ["PATH=/usr/bin", "FALKORDB_ARGS=MAX_QUEUED_QUERIES 25 TIMEOUT 30000 RESULTSET_SIZE 10000"]
        if self.redis_args is not None:
            env.append(f"REDIS_ARGS={self.redis_args}")
        return {
            "State": {"Status": self.status},
            "Mounts": [{"Type": "bind", "Source": str(self.data_dir), "Destination": target}],
            "Config": {"Env": env},
        }

    def _redis_cli(self, args: tuple[str, ...]) -> CommandResult:
        if self.redis_cli_fails:
            return CommandResult(1, "", "LOADING Redis is loading the dataset in memory")
        if args[:4] == ("redis-cli", "--json", "CONFIG", "GET"):
            config = {
                "appendonly": self.appendonly,
                "dir": DATA_DIR_IN_CONTAINER,
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
    ("label", "docker_kwargs", "incr_size", "remedy"),
    [
        ("appendonly off", {"appendonly": "no", "infos": AOF_OFF_INFOS}, INCR_SIZE, ENABLE_AOF),
        ("aof not enabled", {"infos": AOF_OFF_INFOS}, INCR_SIZE, ENABLE_AOF),
        ("last write failed", {"infos": [_info(aof_last_write_status="err")] * 2}, INCR_SIZE, FIX_DISK),
        ("rewrite in progress", {"infos": [_info(), _info(aof_rewrite_in_progress="1")]}, INCR_SIZE, WAIT),
        ("still loading", {"infos": [_info(loading="1")] * 2}, INCR_SIZE, WAIT),
        ("host incr behind the server", {}, INCR_SIZE - 1, BY_HAND),
        ("live data dir not bind-mounted", {"mounted": False}, INCR_SIZE, USE_MAIN_COMPOSE),
        ("redis-cli unreadable", {"redis_cli_fails": True}, INCR_SIZE, BY_HAND),
    ],
)
def test_a_running_container_without_a_current_aof_is_left_alone_with_its_own_remedy(
    data_dir, capsys, label, docker_kwargs, incr_size, remedy,
):
    _write_aof(data_dir, incr_size=incr_size)
    docker = FakeDocker(data_dir, **docker_kwargs)

    assert _gate(docker) == 0, label

    assert docker.actions == [UP_QDRANT], label
    err = capsys.readouterr().err
    assert "recreate REFUSED" in err, label
    assert remedy in err, (label, err)


@pytest.mark.parametrize("extra", [(), ("--check-only",)])
def test_the_post_crash_state_refuses_rather_than_raising(data_dir, capsys, extra):
    """The 10-26 shape: AOF off after a crash restart, a stale manifest on the host, no size in INFO."""
    _write_aof(data_dir)
    docker = FakeDocker(data_dir, appendonly="no", infos=AOF_OFF_INFOS)

    rc = _gate(docker, *extra)

    assert rc == (1 if extra else 0)
    assert docker.actions == ([] if extra else [UP_QDRANT])
    err = capsys.readouterr().err
    assert "recreate REFUSED" in err and ENABLE_AOF in err


def test_a_running_container_with_no_manifest_on_the_host_is_left_alone(data_dir, capsys):
    data_dir.mkdir()
    docker = FakeDocker(data_dir)

    assert _gate(docker) == 0
    assert docker.actions == [UP_QDRANT]
    assert ENABLE_AOF in capsys.readouterr().err


@pytest.mark.parametrize("status", ["running", "exited"])
def test_a_compose_file_binding_another_data_dir_never_touches_falkordb(data_dir, tmp_path, capsys, status):
    _write_aof(data_dir)
    docker = FakeDocker(data_dir, status=status, compose_data_dir=tmp_path / "worktree" / "falkordb")

    assert _gate(docker) == 0

    assert docker.actions == [UP_QDRANT]
    err = capsys.readouterr().err
    assert "REFUSED" in err and USE_MAIN_COMPOSE in err


@pytest.mark.parametrize("status", ["exited", "created"])
def test_a_stopped_container_that_boots_with_aof_is_started_as_it_is(data_dir, capsys, status):
    docker = FakeDocker(data_dir, status=status)

    assert _gate(docker) == 0

    assert docker.actions == [START_EXISTING, UP_QDRANT]
    assert "next fused-memory restart" in capsys.readouterr().err


@pytest.mark.parametrize("redis_args", [None, "", "--appendonly no", "--appendfsync everysec"])
def test_a_stopped_container_that_boots_without_aof_is_not_started(data_dir, capsys, redis_args):
    data_dir.mkdir()
    (data_dir / "dump.rdb").write_bytes(b"x")
    docker = FakeDocker(data_dir, status="exited", redis_args=redis_args)

    assert _gate(docker) == 0

    assert docker.actions == [UP_QDRANT]
    err = capsys.readouterr().err
    assert "start REFUSED" in err and BOOTSTRAP in err


@pytest.mark.parametrize("status", ["restarting", "paused", "dead", "removing"])
def test_an_unsettled_container_is_left_alone(data_dir, capsys, status):
    docker = FakeDocker(data_dir, status=status)

    assert _gate(docker) == 0

    assert docker.actions == [UP_QDRANT]
    err = capsys.readouterr().err
    assert "recreate REFUSED" in err and SETTLE in err


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
    err = capsys.readouterr().err
    assert "create REFUSED" in err and BOOTSTRAP in err


def test_an_unreachable_docker_touches_no_falkordb(data_dir, capsys):
    docker = FakeDocker(data_dir, ps_fails=True)

    _gate(docker)

    assert docker.actions == [UP_QDRANT]
    assert BY_HAND in capsys.readouterr().err


def test_an_unexpected_failure_inside_the_gate_refuses_and_logs_the_traceback(data_dir, capsys):
    _write_aof(data_dir)
    docker = FakeDocker(data_dir, exec_raises=RuntimeError("boom"))

    assert _gate(docker) == 0

    assert docker.actions == [UP_QDRANT]
    err = capsys.readouterr().err
    assert "Traceback" in err and "boom" in err
    assert "REFUSED" in err and BY_HAND in err


def test_a_failed_compose_up_fails_the_gate(data_dir):
    _write_aof(data_dir)
    docker = FakeDocker(data_dir, action_rc=1)

    assert _gate(docker) == 1


@pytest.mark.parametrize(
    ("docker_kwargs", "expected_rc"), [({}, 0), ({"appendonly": "no", "infos": AOF_OFF_INFOS}, 1)],
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

    _gate(FakeDocker(data_dir, appendonly="no", infos=AOF_OFF_INFOS))

    refusal = [line for line in capsys.readouterr().err.splitlines() if "REFUSED" in line]
    assert refusal and all(line.startswith("<3>") for line in refusal)


def test_the_unit_template_brings_the_stores_up_only_through_the_gate():
    template = Path(__file__).resolve().parents[1] / "fused-memory.service.template"
    exec_start_pre = [
        line for line in template.read_text().splitlines() if line.startswith("ExecStartPre=")
    ]

    assert any("scripts/falkordb_aof_gate.py --compose-file" in line for line in exec_start_pre), exec_start_pre
    assert not any("docker compose" in line for line in exec_start_pre), (
        "a bare compose-up would recreate falkordb without the AOF gate"
    )
