#!/usr/bin/env python3
"""Bring up fused-memory's backing stores, recreating FalkorDB only onto a current AOF (task 6330).

The fused-memory unit's ExecStartPre (and setup-host.sh) runs this in place of
a bare ``docker compose up -d falkordb qdrant``. Once docker-compose.yml
changes, that compose-up recreates the falkordb container, and the new server
starts with ``--appendonly yes``. Redis then loads ONLY the AOF named by its
manifest and ignores dump.rdb, so a stale AOF silently drops every write made
since it went stale (and no AOF at all starts the server empty).

A recreate is allowed only when the compose file binds the live container's
own data dir and the live server proves its AOF is current: appendonly yes,
aof_enabled:1, loading:0, aof_last_write_status:ok, aof_rewrite_in_progress:0,
and the manifest's base + incr files on the HOST side of the bind mount total
exactly what the server says it has appended (aof_current_size), bracketed by
two INFO reads taken around the stat. That is "current" whatever the write
rate: an idle server's incr mtime does not move, but its size still equals
aof_current_size.

On a refusal falkordb is not touched, qdrant is still brought up, and each
problem is logged at error priority with its own remedy. Stdlib only, so it
runs under the system python3 with no environment set up.
"""

import argparse
import enum
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

DATA_DIR_IN_CONTAINER = "/var/lib/falkordb/data"
PERSISTED_ARTIFACTS = ("dump.rdb", "appendonlydir")
LOADED_MANIFEST_TYPES = ("b", "i")
STOPPED_STATES = ("created", "exited")
COMMAND_TIMEOUT_SECS = 240
RUNBOOK = 'OPERATIONS.md "FalkorDB settings and the recreate gate"'
ENABLE_AOF_STEPS = (
    "run `docker exec docker-falkordb-1 redis-cli CONFIG SET appendonly yes`, wait until "
    "`redis-cli INFO persistence` shows aof_rewrite_in_progress:0"
)


class Remedy(enum.Enum):
    ENABLE_AOF = f"{ENABLE_AOF_STEPS}, then restart fused-memory"
    WAIT = (
        "wait until `docker exec docker-falkordb-1 redis-cli INFO persistence` shows loading:0 and "
        "aof_rewrite_in_progress:0, then restart fused-memory"
    )
    FIX_DISK = (
        "the server cannot write its AOF: check free space and IO errors on the disk holding the "
        "host data dir, then restart fused-memory"
    )
    USE_MAIN_COMPOSE = (
        "run the gate with the compose file whose falkordb volume is the live container's data dir, "
        "i.e. the main checkout's fused-memory/docker/docker-compose.yml, never a worktree's"
    )
    SETTLE = "wait for the container to settle (`docker ps -a`), then restart fused-memory"
    BOOTSTRAP = (
        "decide which copy is newest (OPERATIONS.md \"Deciding which FalkorDB copy is newest\") before "
        "removing any container, then start falkordb onto that copy"
    )
    INVESTIGATE = (
        "inspect `docker ps -a` and `docker exec docker-falkordb-1 redis-cli INFO persistence` "
        "by hand before restarting fused-memory"
    )


@dataclass(frozen=True)
class Problem:
    finding: str
    remedy: Remedy


class Action(enum.Enum):
    RECONCILE_BOTH = "compose up falkordb and qdrant"
    START_EXISTING = "start the existing falkordb container, compose up qdrant"
    LEAVE_FALKORDB = "leave falkordb as it is, compose up qdrant"


@dataclass(frozen=True)
class Verdict:
    action: Action
    summary: str
    problems: tuple[Problem, ...] = ()
    container: str | None = None

    @property
    def recreate_allowed(self) -> bool:
        return self.action is Action.RECONCILE_BOTH

    @property
    def message(self) -> str:
        if not self.problems:
            return self.summary
        findings = "; ".join(problem.finding for problem in self.problems)
        remedies = " ALSO: ".join(dict.fromkeys(problem.remedy.value for problem in self.problems))
        return f"{self.summary} ({findings}). REMEDY: {remedies} ({RUNBOOK})"


def _refuse(summary: str, *problems: Problem, container: str | None = None) -> Verdict:
    unique = tuple(dict.fromkeys(problems))
    return Verdict(Action.LEAVE_FALKORDB, summary, unique, container)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[[Sequence[str]], CommandResult]


def run_command(argv: Sequence[str]) -> CommandResult:
    try:
        done = subprocess.run(list(argv), capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECS)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CommandResult(127, "", repr(exc))
    return CommandResult(done.returncode, done.stdout, done.stderr)


class ProbeFailed(Exception):
    pass


def redis_cli(container_id: str, *args: str) -> list[str]:
    return ["docker", "exec", container_id, "redis-cli", *args]


class Docker:
    def __init__(self, run: Runner, compose_file: str):
        self._run = run
        self.compose_file = compose_file

    def read(self, argv: Sequence[str]) -> str:
        result = self._run(argv)
        if result.returncode != 0:
            raise ProbeFailed(f"`{' '.join(argv)}` exited {result.returncode}: {result.stderr.strip()}")
        return result.stdout

    def compose(self, *args: str) -> list[str]:
        return ["docker", "compose", "-f", self.compose_file, *args]

    def act(self, argv: Sequence[str]) -> int:
        result = self._run(argv)
        for line in (result.stdout + result.stderr).splitlines():
            log("info", line)
        return result.returncode


@dataclass(frozen=True)
class Container:
    id: str
    status: str
    data_mount: str | None
    redis_args: str

    @classmethod
    def parse(cls, container_id: str, inspect_json: str) -> "Container":
        details = json.loads(inspect_json)[0]
        data_mount = next(
            (m["Source"] for m in details.get("Mounts") or [] if m.get("Destination") == DATA_DIR_IN_CONTAINER),
            None,
        )
        env = dict(entry.partition("=")[::2] for entry in details["Config"]["Env"] or [])
        return cls(container_id, details["State"]["Status"], data_mount, env.get("REDIS_ARGS", ""))

    @property
    def boots_with_aof(self) -> bool:
        tokens = self.redis_args.split()
        return any(name == "--appendonly" and value == "yes" for name, value in zip(tokens, tokens[1:], strict=False))


AOF_CONFIG_KEYS = ("appendonly", "dir", "appenddirname", "appendfilename")


@dataclass(frozen=True)
class ServerAofConfig:
    appendonly: str
    dir: str
    appenddirname: str
    appendfilename: str

    @classmethod
    def parse(cls, config_json: str) -> "ServerAofConfig":
        try:
            config = json.loads(config_json)
            return cls(*(str(config[name]) for name in AOF_CONFIG_KEYS))
        except (ValueError, LookupError, TypeError) as exc:
            raise ValueError(f"CONFIG GET {' '.join(AOF_CONFIG_KEYS)} replied: {config_json.strip()}") from exc

    @classmethod
    def read(cls, docker: Docker, container_id: str) -> "ServerAofConfig":
        return cls.parse(docker.read(redis_cli(container_id, "--json", "CONFIG", "GET", *AOF_CONFIG_KEYS)))

    @property
    def manifest_name(self) -> str:
        return f"{self.appendfilename}.manifest"


def parse_info_fields(info_text: str) -> dict[str, str]:
    lines = (line.partition(":") for line in info_text.splitlines() if line and not line.startswith("#"))
    return {key: value.strip() for key, _, value in lines}


def loaded_aof_file_names(manifest_text: str) -> list[str]:
    names = []
    for line in manifest_text.splitlines():
        tokens = line.split()
        entry = dict(zip(tokens[0::2], tokens[1::2], strict=False))
        if entry.get("type") in LOADED_MANIFEST_TYPES:
            names.append(entry["file"])
    return names


@dataclass(frozen=True)
class Persistence:
    """One ``INFO persistence`` read. Redis omits the aof_* sizes while the AOF is off."""

    loading: str | None
    aof_enabled: str | None
    aof_rewrite_in_progress: str | None
    aof_last_write_status: str | None
    aof_current_size: int | None

    @classmethod
    def parse(cls, info_text: str) -> "Persistence":
        fields = parse_info_fields(info_text)
        size = fields.get("aof_current_size", "")
        return cls(
            fields.get("loading"),
            fields.get("aof_enabled"),
            fields.get("aof_rewrite_in_progress"),
            fields.get("aof_last_write_status"),
            int(size) if size.isdigit() else None,
        )

    def problems(self) -> list[Problem]:
        found = []
        if self.aof_enabled != "1":
            found.append(Problem(f"aof_enabled:{self.aof_enabled}", Remedy.ENABLE_AOF))
        if self.loading != "0":
            found.append(Problem(f"loading:{self.loading}", Remedy.WAIT))
        if self.aof_rewrite_in_progress != "0":
            found.append(Problem(f"aof_rewrite_in_progress:{self.aof_rewrite_in_progress}", Remedy.WAIT))
        if self.aof_last_write_status != "ok":
            found.append(Problem(f"aof_last_write_status:{self.aof_last_write_status}", Remedy.FIX_DISK))
        return found


def decide(docker: Docker) -> Verdict:
    try:
        return _decide(docker)
    except ProbeFailed as exc:
        return _refuse("recreate REFUSED: cannot read the falkordb state", Problem(str(exc), Remedy.INVESTIGATE))
    except Exception as exc:
        for line in traceback.format_exc().splitlines():
            log("error", line)
        return _refuse("recreate REFUSED: the gate could not decide", Problem(repr(exc), Remedy.INVESTIGATE))


def _decide(docker: Docker) -> Verdict:
    container_id = docker.read(docker.compose("ps", "-a", "-q", "falkordb")).strip()
    compose_data_dir = _compose_data_dir(docker)
    if not container_id:
        if _in_linked_worktree(docker):
            return _refuse(
                "create REFUSED: the compose file is in a linked git worktree",
                Problem(f"{docker.compose_file} would create falkordb over {compose_data_dir}", Remedy.USE_MAIN_COMPOSE),
            )
        return _verdict_without_container(compose_data_dir)
    container = Container.parse(container_id, docker.read(["docker", "inspect", container_id]))
    if container.data_mount is None or not _same_dir(container.data_mount, compose_data_dir):
        return _refuse(
            "recreate REFUSED: the compose file and the live container disagree on the data dir",
            Problem(
                f"{docker.compose_file} binds {compose_data_dir}, the container mounts {container.data_mount}",
                Remedy.USE_MAIN_COMPOSE,
            ),
            container=container.id,
        )
    if container.status == "running":
        return _verdict_for_running(docker, container)
    if container.status in STOPPED_STATES:
        return _verdict_for_stopped(container)
    return _refuse(
        "recreate REFUSED: falkordb is not running",
        Problem(f"container status is {container.status}", Remedy.SETTLE),
        container=container.id,
    )


def _compose_data_dir(docker: Docker) -> str:
    config = json.loads(docker.read(docker.compose("config", "--no-env-resolution", "--format", "json")))
    volumes = config["services"]["falkordb"]["volumes"]
    sources = [v["source"] for v in volumes if v.get("target") == DATA_DIR_IN_CONTAINER]
    if not sources:
        raise ProbeFailed(f"{docker.compose_file} mounts nothing at {DATA_DIR_IN_CONTAINER}")
    return sources[0]


def _in_linked_worktree(docker: Docker) -> bool:
    compose_dir = str(Path(docker.compose_file).parent)
    try:
        dirs = docker.read(["git", "-C", compose_dir, "rev-parse", "--path-format=absolute",
                            "--git-dir", "--git-common-dir"]).split()
    except ProbeFailed:
        return False
    return len(dirs) == 2 and not _same_dir(dirs[0], dirs[1])


def _same_dir(left: str, right: str) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _verdict_for_running(docker: Docker, container: Container) -> Verdict:
    problems = _aof_problems(docker, container)
    if problems:
        return _refuse("recreate REFUSED: AOF not current", *problems, container=container.id)
    return Verdict(Action.RECONCILE_BOTH, "AOF current, recreate allowed", container=container.id)


def _verdict_for_stopped(container: Container) -> Verdict:
    if container.boots_with_aof:
        return Verdict(
            Action.START_EXISTING,
            f"falkordb is {container.status} and boots with appendonly yes; starting the existing "
            "container unchanged, because its AOF cannot be read while it is stopped. Any pending "
            "recreate waits for the next fused-memory restart, when the gate can read the live AOF",
            container=container.id,
        )
    return _refuse(
        "start REFUSED: the stopped container would boot with appendonly no",
        Problem(
            f"falkordb is {container.status} and its REDIS_ARGS ({container.redis_args!r}) do not enable "
            "the AOF, so it would load dump.rdb and drop every write only the AOF holds",
            Remedy.BOOTSTRAP,
        ),
        container=container.id,
    )


def _aof_problems(docker: Docker, container: Container) -> list[Problem]:
    info = redis_cli(container.id, "INFO", "persistence")
    try:
        config = ServerAofConfig.read(docker, container.id)
        before = Persistence.parse(docker.read(info))
        on_disk = _manifest_bytes_on_host(container.data_mount, config)
        after = Persistence.parse(docker.read(info))
    except (ProbeFailed, ValueError, LookupError, TypeError, OSError) as exc:
        return [Problem(f"cannot read the live AOF state: {exc}", Remedy.INVESTIGATE)]

    problems = []
    if config.appendonly != "yes":
        problems.append(Problem(f"appendonly is {config.appendonly}", Remedy.ENABLE_AOF))
    problems += before.problems() + after.problems()
    if config.dir != DATA_DIR_IN_CONTAINER:
        problems.append(Problem(f"the server writes to {config.dir}, not the mounted {DATA_DIR_IN_CONTAINER}",
                                Remedy.INVESTIGATE))
    elif on_disk is None:
        problems.append(Problem(f"no AOF manifest under {container.data_mount}", Remedy.ENABLE_AOF))
    elif before.aof_current_size is None or after.aof_current_size is None:
        problems.append(Problem("INFO reports no aof_current_size", Remedy.ENABLE_AOF))
    elif not before.aof_current_size <= on_disk <= after.aof_current_size:
        problems.append(Problem(
            f"host AOF files total {on_disk} bytes, the server reports "
            f"{before.aof_current_size}..{after.aof_current_size}",
            Remedy.INVESTIGATE,
        ))
    return list(dict.fromkeys(problems))


def _manifest_bytes_on_host(host_dir: str | None, config: ServerAofConfig) -> int | None:
    if host_dir is None:
        return None
    aof_dir = Path(host_dir) / config.appenddirname
    manifest = aof_dir / config.manifest_name
    if not manifest.is_file():
        return None
    return sum((aof_dir / name).stat().st_size for name in loaded_aof_file_names(manifest.read_text()))


def _verdict_without_container(host_dir: str) -> Verdict:
    persisted = [name for name in PERSISTED_ARTIFACTS if (Path(host_dir) / name).exists()]
    if not persisted:
        return Verdict(Action.RECONCILE_BOTH, f"no falkordb container and no persisted data in {host_dir}; creating it")
    return _refuse(
        "create REFUSED: no falkordb container over persisted data",
        Problem(
            f"{host_dir} holds {', '.join(persisted)}; a new container boots with appendonly yes and loads "
            "only the AOF, so a stale or missing AOF would drop data. fused-memory will not reach FalkorDB "
            "until it is brought up by hand",
            Remedy.BOOTSTRAP,
        ),
    )


def act(docker: Docker, verdict: Verdict) -> int:
    if verdict.action is Action.RECONCILE_BOTH:
        return docker.act(docker.compose("up", "-d", "falkordb", "qdrant"))
    start_rc = 0
    if verdict.action is Action.START_EXISTING and verdict.container:
        start_rc = docker.act(["docker", "start", verdict.container])
    return docker.act(docker.compose("up", "-d", "qdrant")) or start_rc


_JOURNAL_PRIORITY = {"error": "<3>", "warning": "<4>", "info": "<6>"}


def log(level: str, message: str) -> None:
    prefix = _JOURNAL_PRIORITY[level] if os.environ.get("JOURNAL_STREAM") else ""
    print(f"{prefix}[falkordb-aof-gate] {message}", file=sys.stderr, flush=True)


def _level(verdict: Verdict) -> str:
    if verdict.recreate_allowed:
        return "info"
    return "warning" if verdict.action is Action.START_EXISTING else "error"


def main(argv: Sequence[str] | None = None, *, run: Runner = run_command) -> int:
    parser = argparse.ArgumentParser(
        description="Bring up the backing stores, recreating FalkorDB only onto a current AOF.")
    parser.add_argument("--compose-file", required=True)
    parser.add_argument("--check-only", action="store_true",
                        help="report the verdict and act on nothing; exit 0 if a recreate is allowed, else 1")
    args = parser.parse_args(argv)

    docker = Docker(run, args.compose_file)
    verdict = decide(docker)
    log(_level(verdict), verdict.message)
    if args.check_only:
        return 0 if verdict.recreate_allowed else 1
    log("info", f"action: {verdict.action.value}")
    return act(docker, verdict)


if __name__ == "__main__":
    sys.exit(main())
