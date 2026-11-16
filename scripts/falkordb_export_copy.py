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
files the copied manifest names are then copied, mtimes kept, between the INFO
read that saw the rewrite finish and one taken after the copy, which must show
no save or rewrite landing in between. The destination's manifest is removed
before the copy and written last, from the text the file list was read from;
any failure after that removes it again, so a failed export, or a failed re-run
over an earlier one, holds no manifest and import-data.sh refuses it. A raced
copy is not retried: it means
something still writes FalkorDB, which a silent second attempt would hide. The
source is the container's own data mount, never an argument, so a worktree's
copy of the export cannot flush the live server and then copy an empty dir.
Stdlib only.
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
    ENABLE_AOF_STEPS,
    Container,
    Docker,
    ProbeFailed,
    Runner,
    ServerAofConfig,
    loaded_aof_file_names,
    parse_info_fields,
    redis_cli,
    run_command,
)

RDB_FILE = "dump.rdb"
STARTED_REPLY_PREFIX = "Background"
STILL_WRITING = "something is still writing to FalkorDB: stop it, then re-run the export"


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
    """A background flush: the INFO field counting its completions and the one reporting its outcome,
    each named for messages and read through a typed accessor."""

    command: str
    counter: str
    count: Callable[[SaveState], int]
    status: str
    outcome: Callable[[SaveState], str]


SAVE = Flush("BGSAVE", "rdb_saves", lambda s: s.rdb_saves,
             "rdb_last_bgsave_status", lambda s: s.rdb_last_bgsave_status)
REWRITE = Flush("BGREWRITEAOF", "aof_rewrites", lambda s: s.aof_rewrites,
                "aof_last_bgrewrite_status", lambda s: s.aof_last_bgrewrite_status)


class Server:
    """The running falkordb container's redis, read and driven through ``docker exec``."""

    def __init__(self, docker: Docker, container_id: str, *, timeout_secs: float, poll_secs: float,
                 sleep: Callable[[float], None], clock: Callable[[], float]):
        self._docker = docker
        self._container_id = container_id
        self._timeout_secs = timeout_secs
        self._poll_secs = poll_secs
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
    except (ValueError, LookupError, TypeError) as exc:
        log(f"FAILED: docker or redis-cli gave a reply this helper cannot read ({type(exc).__name__}: {exc})")
        return 1
    return 0


def export(docker: Docker, args: argparse.Namespace,
           sleep: Callable[[float], None], clock: Callable[[], float]) -> str:
    container_id, host_dir = _running_falkordb(docker)
    server = Server(docker, container_id, timeout_secs=args.timeout_secs, poll_secs=args.poll_secs,
                    sleep=sleep, clock=clock)
    config = ServerAofConfig.read(docker, container_id)
    _require_mounted_dir(config, container_id, host_dir)
    _require_aof(config, server.wait_until(lambda s: s.idle, "the server to be idle"))
    _flush(server, SAVE)
    rewritten = _flush(server, REWRITE)
    copied, state = _bracketed_copy(server, rewritten, host_dir, config, args.dest)
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


def _require_mounted_dir(config: ServerAofConfig, container_id: str, host_dir: Path) -> None:
    if config.dir != DATA_DIR_IN_CONTAINER:
        raise ExportFailed(
            f"the server writes to {config.dir}, but {host_dir} is mounted at {DATA_DIR_IN_CONTAINER}, so the "
            f"files there are not the ones it saves. Nothing was flushed or copied. REMEDY: inspect "
            f"`docker exec {container_id} redis-cli CONFIG GET dir` and `docker inspect {container_id}` by hand; "
            "files saved outside the mount may exist only inside the container, so copy them out before "
            "recreating it, then re-run the export"
        )


def _require_aof(config: ServerAofConfig, state: SaveState) -> None:
    problems = []
    if config.appendonly != "yes":
        problems.append(f"appendonly is {config.appendonly}")
    if state.aof_enabled != "1":
        problems.append(f"aof_enabled:{state.aof_enabled}")
    if problems:
        raise ExportFailed(
            f"{'; '.join(problems)}. Nothing was flushed or copied: import-data.sh would refuse an export "
            f"without a current AOF. REMEDY: {ENABLE_AOF_STEPS}, then re-run the export"
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


def _bracketed_copy(
    server: Server, before: SaveState, host_dir: Path, config: ServerAofConfig, dest: Path,
) -> tuple[int, SaveState]:
    copied_manifest = dest / config.appenddirname / config.manifest_name
    try:
        copied_manifest.unlink(missing_ok=True)
        copied = _copy_restore_set(host_dir, config, dest)
        after = server.info()
        _require_steady(before, after)
    except BaseException:
        copied_manifest.unlink(missing_ok=True)
        log(f"left no {config.manifest_name} under {dest}, so import-data.sh refuses this partial export")
        raise
    return copied, after


def _require_steady(before: SaveState, after: SaveState) -> None:
    problems = [
        f"{flush.counter} {flush.count(before)}→{flush.count(after)}"
        for flush in (SAVE, REWRITE) if flush.count(before) != flush.count(after)
    ]
    for when, state in (("before", before), ("after", after)):
        if not state.idle or state.aof_last_write_status != "ok":
            problems.append(f"{when} the copy: {state.describe()}")
    if problems:
        raise ExportFailed(f"the copy is not one consistent snapshot ({'; '.join(problems)}); {STILL_WRITING}")


def _copy_restore_set(host_dir: Path, config: ServerAofConfig, dest: Path) -> int:
    manifest = Path(config.appenddirname) / config.manifest_name
    try:
        manifest_bytes = (host_dir / manifest).read_bytes()
        names = loaded_aof_file_names(manifest_bytes.decode())
        (dest / config.appenddirname).mkdir(parents=True, exist_ok=True)
        if (host_dir / RDB_FILE).exists():
            shutil.copy2(host_dir / RDB_FILE, dest / RDB_FILE)
        for name in names:
            shutil.copy2(host_dir / config.appenddirname / name, dest / config.appenddirname / name)
        (dest / manifest).write_bytes(manifest_bytes)
        shutil.copystat(host_dir / manifest, dest / manifest)
    except OSError as exc:
        raise ExportFailed(
            f"copying the restore set failed ({exc}); a named file that vanished means a rewrite replaced it "
            f"mid-copy, so {STILL_WRITING}"
        ) from exc
    return len(names)


if __name__ == "__main__":
    sys.exit(main())
