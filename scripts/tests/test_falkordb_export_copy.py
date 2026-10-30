"""scripts/falkordb_export_copy.py: export a running FalkorDB onto a freshly rewritten AOF (task 6334).

FakeFalkorDB answers the helper's docker and redis-cli commands from a small
stateful model of one redis server whose host data dir is a tmp dir. BGSAVE and
BGREWRITEAOF start a child that finishes a few INFO polls later and writes real
files with strictly increasing mtimes. Like redis-cli 8.6.2 (measured), an error
reply arrives on stdout with exit 0. FakeClock stands in for time.

The seeded data dir is the measured live state: an AOF manifest OLDER than
dump.rdb, plus a stray temp-1.rdb.
"""

import json
import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from falkordb_aof_gate import CommandResult
from falkordb_export_copy import main

COMPOSE = "/repo/fused-memory/docker/docker-compose.yml"
CONTAINER = "dad0940ab28a"
DATA_DIR_IN_CONTAINER = "/var/lib/falkordb/data"
REDIS_CLI = ("docker", "exec", CONTAINER, "redis-cli")
CONFIG_GET = (*REDIS_CLI, "--json", "CONFIG", "GET", "appendonly", "dir", "appenddirname", "appendfilename")
INFO = (*REDIS_CLI, "INFO", "persistence")
BGSAVE = (*REDIS_CLI, "BGSAVE")
BGREWRITEAOF = (*REDIS_CLI, "BGREWRITEAOF")
CHILD_POLLS = 2
TIMEOUT_SECS = 60
POLL_SECS = 1
FIRST_MTIME_NS = 1_793_000_000 * 10**9
MANIFEST = "appendonlydir/appendonly.aof.manifest"


def _manifest(seq: int, history_seq: int | None) -> str:
    lines = [f"file appendonly.aof.{seq}.base.rdb seq {seq} type b"]
    if history_seq is not None:
        lines += [
            f"file appendonly.aof.{history_seq}.base.rdb seq {history_seq} type h",
            f"file appendonly.aof.{history_seq}.incr.aof seq {history_seq} type h",
        ]
    lines.append(f"file appendonly.aof.{seq}.incr.aof seq {seq} type i startoffset 0")
    return "\n".join(lines) + "\n"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, secs: float) -> None:
        self.now += secs


class FakeFalkorDB:
    def __init__(self, data_dir: Path, *, appendonly: str = "yes", **info: str):
        self.data_dir = data_dir
        self.aof_dir = data_dir / "appendonlydir"
        self.appendonly = appendonly
        self.info = {
            "loading": "0",
            "aof_enabled": "1",
            "rdb_bgsave_in_progress": "0",
            "aof_rewrite_in_progress": "0",
            "aof_rewrite_scheduled": "0",
            "rdb_saves": "200",
            "aof_rewrites": "0",
            "rdb_last_bgsave_status": "ok",
            "aof_last_bgrewrite_status": "ok",
            "aof_last_write_status": "ok",
        }
        self.info.update(info)
        self.commands: list[tuple[str, ...]] = []
        self._mtime_ns = FIRST_MTIME_NS
        self._seq = 1
        self._child: str | None = None
        self._child_polls = 0
        self._seed()

    @property
    def sent(self) -> list[tuple[str, ...]]:
        return [command for command in self.commands if command in (BGSAVE, BGREWRITEAOF)]

    def _seed(self) -> None:
        self.aof_dir.mkdir(parents=True)
        self._write(self.aof_dir / "appendonly.aof.1.base.rdb", b"base-1")
        self._write(self.aof_dir / "appendonly.aof.1.incr.aof", b"incr-1")
        self._write(self.aof_dir / "appendonly.aof.manifest", _manifest(1, None).encode())
        self._write(self.data_dir / "temp-1.rdb", b"stale temp")
        self._write(self.data_dir / "dump.rdb", b"dump-200")

    def _write(self, path: Path, data: bytes) -> None:
        path.write_bytes(data)
        self._mtime_ns += 10**9
        os.utime(path, ns=(self._mtime_ns, self._mtime_ns))

    def __call__(self, argv: Sequence[str]) -> CommandResult:
        command = tuple(argv)
        self.commands.append(command)
        if command == ("docker", "compose", "-f", COMPOSE, "ps", "-q", "falkordb"):
            return CommandResult(0, f"{CONTAINER}\n", "")
        if command == ("docker", "inspect", CONTAINER):
            return CommandResult(0, json.dumps([self._inspect()]), "")
        if command == CONFIG_GET:
            config = {
                "appendonly": self.appendonly,
                "dir": DATA_DIR_IN_CONTAINER,
                "appenddirname": "appendonlydir",
                "appendfilename": "appendonly.aof",
            }
            return CommandResult(0, json.dumps(config) + "\n", "")
        if command == INFO:
            self._tick()
            return CommandResult(0, self._render_info(), "")
        if command == BGSAVE:
            return CommandResult(0, self._bgsave() + "\n", "")
        if command == BGREWRITEAOF:
            return CommandResult(0, self._bgrewriteaof() + "\n", "")
        raise AssertionError(f"unexpected command {command!r}")

    def _inspect(self) -> dict:
        return {
            "State": {"Status": "running"},
            "Mounts": [{"Type": "bind", "Source": str(self.data_dir), "Destination": DATA_DIR_IN_CONTAINER}],
            "Config": {"Env": ["PATH=/usr/bin", "REDIS_ARGS=--appendonly yes --appendfsync everysec"]},
        }

    def _render_info(self) -> str:
        return "# Persistence\r\n" + "".join(f"{key}:{value}\r\n" for key, value in self.info.items())

    def _bgsave(self) -> str:
        if self.info["rdb_bgsave_in_progress"] == "1":
            return "ERR Background save already in progress"
        if self.info["aof_rewrite_in_progress"] == "1":
            return (
                "ERR An AOF log rewriting in progress: can't BGSAVE right now. "
                "Use BGSAVE SCHEDULE in order to schedule a BGSAVE whenever possible."
            )
        self._start_child("rdb_bgsave_in_progress")
        return "Background saving started"

    def _bgrewriteaof(self) -> str:
        if self.info["aof_rewrite_in_progress"] == "1":
            return "ERR Background append only file rewriting already in progress"
        if self.info["rdb_bgsave_in_progress"] == "1":
            self.info["aof_rewrite_scheduled"] = "1"
            return "Background append only file rewriting scheduled"
        self._start_child("aof_rewrite_in_progress")
        return "Background append only file rewriting started"

    def _start_child(self, field: str) -> None:
        self.info[field] = "1"
        self._child, self._child_polls = field, CHILD_POLLS

    def _tick(self) -> None:
        if self._child is None:
            return
        self._child_polls -= 1
        if self._child_polls > 0:
            return
        field, self._child = self._child, None
        self.info[field] = "0"
        if field == "rdb_bgsave_in_progress":
            self._finish_save()
        else:
            self._finish_rewrite()

    def _finish_save(self) -> None:
        saves = int(self.info["rdb_saves"]) + 1
        self._write(self.data_dir / "dump.rdb", f"dump-{saves}".encode())
        self.info["rdb_saves"] = str(saves)

    def _finish_rewrite(self) -> None:
        previous, self._seq = self._seq, self._seq + 1
        self._write(self.aof_dir / f"appendonly.aof.{self._seq}.base.rdb", f"base-{self._seq}".encode())
        self._write(self.aof_dir / f"appendonly.aof.{self._seq}.incr.aof", f"incr-{self._seq}".encode())
        self._write(self.aof_dir / "appendonly.aof.manifest", _manifest(self._seq, previous).encode())
        self.info["aof_rewrites"] = str(int(self.info["aof_rewrites"]) + 1)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "falkordb"


@pytest.fixture
def dest(tmp_path: Path) -> Path:
    return tmp_path / "export" / "fused-memory-data" / "falkordb"


def _export(fake: FakeFalkorDB, dest: Path, clock: FakeClock | None = None) -> int:
    clock = clock or FakeClock()
    argv = [
        "--compose-file", COMPOSE, "--dest", str(dest),
        "--timeout-secs", str(TIMEOUT_SECS), "--poll-secs", str(POLL_SECS),
    ]
    return main(argv, run=fake, sleep=clock.sleep, clock=clock.monotonic)


def _files_under(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}


def test_a_running_server_is_saved_then_rewritten_then_copied(data_dir, dest):
    fake = FakeFalkorDB(data_dir)
    manifest_before = (data_dir / MANIFEST).stat().st_mtime_ns
    assert manifest_before < (data_dir / "dump.rdb").stat().st_mtime_ns

    assert _export(fake, dest) == 0

    assert fake.sent == [BGSAVE, BGREWRITEAOF]
    copied = _files_under(dest)
    assert copied == {
        "dump.rdb",
        MANIFEST,
        "appendonlydir/appendonly.aof.2.base.rdb",
        "appendonlydir/appendonly.aof.2.incr.aof",
    }
    assert (dest / MANIFEST).read_text() == _manifest(2, 1)
    for name in copied:
        assert (dest / name).read_bytes() == (data_dir / name).read_bytes(), name
        assert (dest / name).stat().st_mtime_ns == (data_dir / name).stat().st_mtime_ns, name
    assert (dest / MANIFEST).stat().st_mtime_ns >= (dest / "dump.rdb").stat().st_mtime_ns


def test_only_the_restore_set_is_copied(data_dir, dest):
    fake = FakeFalkorDB(data_dir)

    assert _export(fake, dest) == 0

    copied = _files_under(dest)
    for history in ("appendonly.aof.1.base.rdb", "appendonly.aof.1.incr.aof"):
        assert (data_dir / "appendonlydir" / history).exists(), history
        assert f"appendonlydir/{history}" not in copied
    assert "temp-1.rdb" not in copied


@pytest.mark.parametrize(
    ("server", "field"),
    [({"appendonly": "no"}, "appendonly"), ({"aof_enabled": "0"}, "aof_enabled")],
    ids=["config-appendonly-no", "info-aof-enabled-0"],
)
def test_a_server_with_the_aof_off_is_refused_before_anything_is_written(data_dir, dest, capsys, server, field):
    fake = FakeFalkorDB(data_dir, **server)

    assert _export(fake, dest) == 1

    assert fake.sent == []
    assert not dest.exists()
    err = capsys.readouterr().err
    assert field in err
    assert "import-data.sh" in err and "refuse" in err
