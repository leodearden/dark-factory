#!/usr/bin/env python3
"""Copy a running FalkorDB's restore set for export onto a freshly rewritten AOF (task 6334).

With appendonly yes (task 6330) a restored server loads ONLY the AOF files its
manifest names and ignores dump.rdb. The manifest moves only on an AOF rewrite,
while saves keep refreshing dump.rdb, so a plain copy of the live data dir
carries a manifest older than dump.rdb, and scripts/import-data.sh refuses
exactly that. After a rewrite redis also unlinks the superseded (type h) files
in a background thread, so a whole-dir copy taken then races those unlinks.

So the server is driven to a fresh rewrite first: wait until it is idle,
require the AOF on, BGSAVE, then BGREWRITEAOF, which leaves a manifest newer
than dump.rdb by construction. Only dump.rdb, the manifest and the base/incr
files the copied manifest names are then copied, mtimes kept. The source is the
container's own data mount, never an argument, so a worktree's copy of the
export cannot flush the live server and then copy an empty dir. Stdlib only.
"""

import argparse
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from falkordb_aof_gate import (
    DATA_DIR_IN_CONTAINER,
    Container,
    Docker,
    ProbeFailed,
    Remedy,
    Runner,
    ServerAofConfig,
    loaded_aof_file_names,
    parse_info_fields,
    redis_cli,
    run_command,
)

RDB_FILE = "dump.rdb"
STARTED_REPLY_PREFIX = "Background"


class ExportFailed(Exception):
    pass


@dataclass(frozen=True)
class SaveState:
    """One ``INFO persistence`` read, reduced to what a flush-and-copy depends on."""

    loading: str
    aof_enabled: str
    rdb_bgsave_in_progress: str
    aof_rewrite_in_progress: str
    aof_rewrite_scheduled: str
    rdb_saves: int
    aof_rewrites: int
    rdb_last_bgsave_status: str
    aof_last_bgrewrite_status: str
    aof_last_write_status: str

    @classmethod
    def parse(cls, info_text: str) -> "SaveState":
        fields = parse_info_fields(info_text)
        try:
            return cls(
                loading=fields["loading"],
                aof_enabled=fields["aof_enabled"],
                rdb_bgsave_in_progress=fields["rdb_bgsave_in_progress"],
                aof_rewrite_in_progress=fields["aof_rewrite_in_progress"],
                aof_rewrite_scheduled=fields["aof_rewrite_scheduled"],
                rdb_saves=int(fields["rdb_saves"]),
                aof_rewrites=int(fields["aof_rewrites"]),
                rdb_last_bgsave_status=fields["rdb_last_bgsave_status"],
                aof_last_bgrewrite_status=fields["aof_last_bgrewrite_status"],
                aof_last_write_status=fields["aof_last_write_status"],
            )
        except KeyError as exc:
            raise ExportFailed(f"INFO persistence has no {exc.args[0]} field") from exc

    @property
    def idle(self) -> bool:
        return (self.loading, self.rdb_bgsave_in_progress, self.aof_rewrite_in_progress,
                self.aof_rewrite_scheduled) == ("0", "0", "0", "0")

    def describe(self) -> str:
        return " ".join(f"{name}:{value}" for name, value in vars(self).items())


@dataclass(frozen=True)
class Flush:
    command: str
    counter: str
    status: str

    def count(self, state: SaveState) -> int:
        return getattr(state, self.counter)

    def outcome(self, state: SaveState) -> str:
        return getattr(state, self.status)


SAVE = Flush("BGSAVE", "rdb_saves", "rdb_last_bgsave_status")
REWRITE = Flush("BGREWRITEAOF", "aof_rewrites", "aof_last_bgrewrite_status")


class Server:
    """The running falkordb container's redis, read and driven through ``docker exec``."""

    def __init__(self, docker: Docker, container_id: str, args: argparse.Namespace,
                 sleep: Callable[[float], None], clock: Callable[[], float]):
        self._docker = docker
        self._container_id = container_id
        self._timeout_secs: float = args.timeout_secs
        self._poll_secs: float = args.poll_secs
        self._sleep = sleep
        self._clock = clock

    def info(self) -> SaveState:
        return SaveState.parse(self._docker.read(redis_cli(self._container_id, "INFO", "persistence")))

    def send(self, command: str) -> str:
        return self._docker.read(redis_cli(self._container_id, command)).strip()

    def wait_until(self, done: Callable[[SaveState], bool], what: str) -> SaveState:
        deadline = self._clock() + self._timeout_secs
        while True:
            state = self.info()
            if done(state):
                return state
            if self._clock() >= deadline:
                raise ExportFailed(
                    f"timed out after {self._timeout_secs:g}s waiting for {what}; "
                    f"last INFO persistence read: {state.describe()}"
                )
            self._sleep(self._poll_secs)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a running FalkorDB's restore set for export onto a freshly rewritten AOF.")
    parser.add_argument("--compose-file", required=True)
    parser.add_argument("--dest", required=True, type=Path,
                        help="directory to receive dump.rdb and the AOF restore set")
    parser.add_argument("--timeout-secs", type=float, default=1800,
                        help="bound on each wait; a BGSAVE of the live 3.9 GB dataset measured 79s")
    parser.add_argument("--poll-secs", type=float, default=2)
    return parser.parse_args(argv)


def log(message: str) -> None:
    print(f"[falkordb-export-copy] {message}", file=sys.stderr, flush=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    run: Runner = run_command,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    args = parse_args(argv)
    docker = Docker(run, args.compose_file)
    try:
        log(export(docker, args, sleep, clock))
    except (ExportFailed, ProbeFailed) as exc:
        log(f"FAILED: {exc}")
        return 1
    return 0


def export(docker: Docker, args: argparse.Namespace,
           sleep: Callable[[float], None], clock: Callable[[], float]) -> str:
    container_id, host_dir = _running_falkordb(docker)
    server = Server(docker, container_id, args, sleep, clock)
    config = ServerAofConfig.read(docker, container_id)
    _require_aof(config, server.wait_until(lambda s: s.idle, "the server to be idle"))
    _flush(server, SAVE)
    state = _flush(server, REWRITE)
    copied = _copy_restore_set(host_dir, config, args.dest)
    return (f"copied {RDB_FILE}, {config.manifest_name} and {copied} base/incr file(s) "
            f"to {args.dest} at aof_rewrites:{state.aof_rewrites}")


def _running_falkordb(docker: Docker) -> tuple[str, Path]:
    container_id = docker.read(docker.compose("ps", "-q", "falkordb")).strip()
    if not container_id:
        raise ExportFailed(f"{docker.compose_file} has no running falkordb container")
    container = Container.parse(container_id, docker.read(["docker", "inspect", container_id]))
    if container.status != "running":
        raise ExportFailed(f"falkordb container {container.id} is {container.status}, not running")
    if container.data_mount is None:
        raise ExportFailed(f"falkordb container {container.id} mounts nothing at {DATA_DIR_IN_CONTAINER}")
    return container.id, Path(container.data_mount)


def _require_aof(config: ServerAofConfig, state: SaveState) -> None:
    problems = []
    if config.appendonly != "yes":
        problems.append(f"appendonly is {config.appendonly}")
    if state.aof_enabled != "1":
        problems.append(f"aof_enabled:{state.aof_enabled}")
    if config.dir != DATA_DIR_IN_CONTAINER:
        problems.append(f"the server writes to {config.dir}, not the mounted {DATA_DIR_IN_CONTAINER}")
    if problems:
        raise ExportFailed(
            f"{'; '.join(problems)}. Nothing was flushed or copied: import-data.sh would refuse an export "
            f"without a current AOF. REMEDY: {Remedy.ENABLE_AOF.value}, then re-run the export"
        )


def _flush(server: Server, flush: Flush) -> SaveState:
    before = server.wait_until(lambda s: s.idle, f"the server to be idle before {flush.command}")
    reply = server.send(flush.command)
    if not reply.startswith(STARTED_REPLY_PREFIX):
        raise ExportFailed(f"{flush.command} was refused: {reply}")
    log(f"{flush.command}: {reply}")
    after = server.wait_until(
        lambda s: s.idle and (flush.count(s) > flush.count(before) or flush.outcome(s) != "ok"),
        f"{flush.command} to finish",
    )
    if flush.outcome(after) != "ok":
        raise ExportFailed(f"{flush.command} failed: {flush.status}:{flush.outcome(after)}")
    log(f"{flush.command} done: {flush.counter} {flush.count(before)}→{flush.count(after)}")
    return after


def _copy_restore_set(host_dir: Path, config: ServerAofConfig, dest: Path) -> int:
    manifest = Path(config.appenddirname) / config.manifest_name
    (dest / config.appenddirname).mkdir(parents=True, exist_ok=True)
    if (host_dir / RDB_FILE).exists():
        shutil.copy2(host_dir / RDB_FILE, dest / RDB_FILE)
    shutil.copy2(host_dir / manifest, dest / manifest)
    names = loaded_aof_file_names((dest / manifest).read_text())
    for name in names:
        shutil.copy2(host_dir / config.appenddirname / name, dest / config.appenddirname / name)
    return len(names)


if __name__ == "__main__":
    sys.exit(main())
