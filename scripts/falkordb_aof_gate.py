#!/usr/bin/env python3
"""Bring up fused-memory's backing stores, recreating FalkorDB only onto a current AOF (task 6330).

The fused-memory unit's ExecStartPre runs this in place of a bare
``docker compose up -d falkordb qdrant``. Once docker-compose.yml changes, that
compose-up recreates the falkordb container, and the new server starts with
``--appendonly yes``. Redis then loads ONLY the AOF named by its manifest and
ignores dump.rdb, so a stale AOF silently drops every write made since it went
stale (and no AOF at all starts the server empty).

A recreate is allowed only when the live server proves its AOF is current:
appendonly yes, aof_enabled:1, loading:0, aof_last_write_status:ok,
aof_rewrite_in_progress:0, and the manifest's base + incr files on the HOST
side of the bind mount total exactly what the server says it has appended
(aof_current_size), bracketed by two INFO reads taken around the stat. That is
"current" whatever the write rate: an idle server's incr mtime does not move,
but its size still equals aof_current_size.

A refusal never takes fused-memory down: falkordb is left as it is, qdrant is
still brought up, and the remedy is logged at error priority. Stdlib only, so
it runs under the system python3 with no environment set up.
"""

import argparse
import enum
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

DATA_DIR_IN_CONTAINER = "/var/lib/falkordb/data"
PERSISTED_ARTIFACTS = ("dump.rdb", "appendonlydir")
LOADED_MANIFEST_TYPES = ("b", "i")
COMMAND_TIMEOUT_SECS = 240
REMEDY = (
    "remedy: `docker exec docker-falkordb-1 redis-cli CONFIG SET appendonly yes`, wait until "
    "`redis-cli INFO persistence` shows aof_rewrite_in_progress:0, then restart fused-memory "
    "(OPERATIONS.md \"FalkorDB settings and the recreate gate\")"
)


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


class Action(enum.Enum):
    RECONCILE_BOTH = "compose up falkordb and qdrant"
    START_EXISTING = "start the existing falkordb container, compose up qdrant"
    LEAVE_FALKORDB = "leave falkordb as it is, compose up qdrant"


@dataclass(frozen=True)
class Verdict:
    action: Action
    reason: str

    @property
    def recreate_allowed(self) -> bool:
        return self.action is Action.RECONCILE_BOTH


class ProbeFailed(Exception):
    pass


class Docker:
    def __init__(self, run: Runner, compose_file: str):
        self._run = run
        self._compose = ["docker", "compose", "-f", compose_file]

    def read(self, argv: Sequence[str]) -> str:
        result = self._run(argv)
        if result.returncode != 0:
            raise ProbeFailed(f"`{' '.join(argv)}` exited {result.returncode}: {result.stderr.strip()}")
        return result.stdout

    def compose(self, *args: str) -> list[str]:
        return [*self._compose, *args]

    def act(self, argv: Sequence[str]) -> int:
        result = self._run(argv)
        for line in (result.stdout + result.stderr).splitlines():
            log("info", line)
        return result.returncode


def decide(docker: Docker) -> tuple[Verdict, str | None]:
    try:
        container = docker.read(docker.compose("ps", "-a", "-q", "falkordb")).strip()
        if not container:
            return _verdict_without_container(docker), None
        status, mounts = _inspect(docker, container)
        if status == "running":
            return _verdict_for_running(docker, container, mounts), container
    except ProbeFailed as exc:
        return Verdict(Action.LEAVE_FALKORDB, f"recreate REFUSED: cannot read falkordb state ({exc}); {REMEDY}"), None
    if status in ("created", "exited"):
        return Verdict(
            Action.START_EXISTING,
            f"falkordb is {status}; starting the existing container unchanged, because its AOF "
            "cannot be read while it is stopped. Any pending recreate waits for the next "
            "fused-memory restart, when the gate can read the live AOF",
        ), container
    return Verdict(
        Action.LEAVE_FALKORDB,
        f"recreate REFUSED: falkordb container is {status}, not running; {REMEDY}",
    ), container


def _inspect(docker: Docker, container: str) -> tuple[str, dict[str, str]]:
    try:
        details = json.loads(docker.read(["docker", "inspect", container]))[0]
        mounts = {m["Destination"]: m["Source"] for m in details["Mounts"] if m.get("Type") == "bind"}
        return details["State"]["Status"], mounts
    except (ValueError, LookupError, TypeError) as exc:
        raise ProbeFailed(f"unexpected `docker inspect` output: {exc!r}") from exc


def _verdict_for_running(docker: Docker, container: str, mounts: dict[str, str]) -> Verdict:
    problems = _aof_problems(docker, container, mounts)
    if problems:
        return Verdict(Action.LEAVE_FALKORDB, f"recreate REFUSED: AOF not current ({'; '.join(problems)}); {REMEDY}")
    return Verdict(Action.RECONCILE_BOTH, "AOF current, recreate allowed")


def _aof_problems(docker: Docker, container: str, mounts: dict[str, str]) -> list[str]:
    redis_cli = ["docker", "exec", container, "redis-cli"]
    try:
        config = json.loads(docker.read([*redis_cli, "--json", "CONFIG", "GET",
                                         "appendonly", "dir", "appenddirname", "appendfilename"]))
        before = _parse_info(docker.read([*redis_cli, "INFO", "persistence"]))
        host_dir = mounts.get(config["dir"])
        on_disk = _manifest_bytes_on_host(host_dir, config) if host_dir else None
        after = _parse_info(docker.read([*redis_cli, "INFO", "persistence"]))
    except (ProbeFailed, ValueError, LookupError, TypeError, OSError) as exc:
        return [f"cannot read the live AOF state: {exc}"]

    problems = []
    if config["appendonly"] != "yes":
        problems.append(f"appendonly is {config['appendonly']}")
    for info in (before, after):
        for field, wanted in (("aof_enabled", "1"), ("loading", "0"),
                              ("aof_last_write_status", "ok"), ("aof_rewrite_in_progress", "0")):
            if info.get(field) != wanted:
                problems.append(f"{field}:{info.get(field)}")
    if host_dir is None:
        problems.append(f"{config['dir']} is not bind-mounted from the host")
    elif on_disk is None:
        problems.append(f"no AOF manifest under {host_dir}")
    elif not int(before["aof_current_size"]) <= on_disk <= int(after["aof_current_size"]):
        problems.append(
            f"host AOF files total {on_disk} bytes, server reports "
            f"{before['aof_current_size']}..{after['aof_current_size']}"
        )
    return list(dict.fromkeys(problems))


def _parse_info(text: str) -> dict[str, str]:
    pairs = (line.partition(":") for line in text.splitlines() if line and not line.startswith("#"))
    return {key: value.strip() for key, _, value in pairs}


def _manifest_bytes_on_host(host_dir: str, config: dict[str, str]) -> int | None:
    aof_dir = Path(host_dir) / config["appenddirname"]
    manifest = aof_dir / f"{config['appendfilename']}.manifest"
    if not manifest.is_file():
        return None
    total = 0
    for line in manifest.read_text().splitlines():
        tokens = line.split()
        entry = dict(zip(tokens[0::2], tokens[1::2], strict=False))
        if entry.get("type") in LOADED_MANIFEST_TYPES:
            total += (aof_dir / entry["file"]).stat().st_size
    return total


def _verdict_without_container(docker: Docker) -> Verdict:
    try:
        config_json = docker.read(docker.compose("config", "--no-env-resolution", "--format", "json"))
        services = json.loads(config_json)["services"]
        host_dir = next(
            v["source"] for v in services["falkordb"]["volumes"] if v.get("target") == DATA_DIR_IN_CONTAINER
        )
    except (ProbeFailed, ValueError, LookupError, TypeError, StopIteration) as exc:
        return Verdict(Action.LEAVE_FALKORDB, f"create REFUSED: cannot resolve the falkordb data dir ({exc!r})")
    persisted = [name for name in PERSISTED_ARTIFACTS if (Path(host_dir) / name).exists()]
    if not persisted:
        return Verdict(Action.RECONCILE_BOTH, f"no falkordb container and no persisted data in {host_dir}; creating it")
    return Verdict(
        Action.LEAVE_FALKORDB,
        f"create REFUSED: no falkordb container, and {host_dir} holds {', '.join(persisted)}. A new "
        "container starts with appendonly yes and loads only the AOF, so a stale or missing AOF "
        "would drop data. fused-memory will not reach FalkorDB until it is created by hand "
        "(OPERATIONS.md \"FalkorDB settings and the recreate gate\")",
    )


def act(docker: Docker, verdict: Verdict, container: str | None) -> int:
    if verdict.action is Action.RECONCILE_BOTH:
        return docker.act(docker.compose("up", "-d", "falkordb", "qdrant"))
    start_rc = 0
    if verdict.action is Action.START_EXISTING and container:
        start_rc = docker.act(["docker", "start", container])
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
    parser = argparse.ArgumentParser(description="Bring up the backing stores, recreating FalkorDB only onto a current AOF.")
    parser.add_argument("--compose-file", required=True)
    parser.add_argument("--check-only", action="store_true",
                        help="report the verdict and act on nothing; exit 0 if a recreate is allowed, else 1")
    args = parser.parse_args(argv)

    docker = Docker(run, args.compose_file)
    verdict, container = decide(docker)
    log(_level(verdict), verdict.reason)
    if args.check_only:
        return 0 if verdict.recreate_allowed else 1
    log("info", f"action: {verdict.action.value}")
    return act(docker, verdict, container)


if __name__ == "__main__":
    sys.exit(main())
