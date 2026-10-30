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
CHILD_STATUS = {
    "rdb_bgsave_in_progress": "rdb_last_bgsave_status",
    "aof_rewrite_in_progress": "aof_last_bgrewrite_status",
}
MODULE_FORK_REPLY = (
    "ERR Another child process is active (AOF?): can't BGSAVE right now. "
    "Use BGSAVE SCHEDULE in order to schedule a BGSAVE whenever possible."
)


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
    """Knobs: ``in_flight`` starts a child before the export, ``loading_polls`` keeps loading:1 for
    that many INFO reads, ``stuck`` names a child that never finishes, ``fails`` one that ends with
    status err, ``rejects`` a command answered with an error reply, and ``exec_fails`` makes every
    ``docker exec`` fail as on a container that went away."""

    def __init__(
        self,
        data_dir: Path,
        *,
        appendonly: str = "yes",
        listed: bool = True,
        status: str = "running",
        in_flight: str | None = None,
        loading_polls: int = 0,
        stuck: str | None = None,
        fails: str | None = None,
        rejects: tuple[str, ...] | None = None,
        exec_fails: bool = False,
        **info: str,
    ):
        self.data_dir = data_dir
        self.aof_dir = data_dir / "appendonlydir"
        self.appendonly = appendonly
        self.listed = listed
        self.status = status
        self.loading_polls = loading_polls
        self.stuck = stuck
        self.fails = fails
        self.rejects = rejects
        self.exec_fails = exec_fails
        self.info = {
            "loading": "1" if loading_polls else "0",
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
        self.replies: list[str] = []
        self._mtime_ns = FIRST_MTIME_NS
        self._seq = 1
        self._child: str | None = None
        self._child_polls = 0
        self._seed()
        if in_flight is not None:
            self._start_child(in_flight)

    @property
    def sent(self) -> list[tuple[str, ...]]:
        return [command for command in self.commands if command in (BGSAVE, BGREWRITEAOF)]

    @property
    def execs(self) -> list[tuple[str, ...]]:
        return [command for command in self.commands if command[:2] == ("docker", "exec")]

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
            return CommandResult(0, f"{CONTAINER}\n" if self.listed else "", "")
        if command == ("docker", "inspect", CONTAINER):
            return CommandResult(0, json.dumps([self._inspect()]), "")
        if command[:3] == ("docker", "exec", CONTAINER) and self.exec_fails:
            return CommandResult(1, "", f"Error response from daemon: container {CONTAINER} is not running")
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
        if command in (BGSAVE, BGREWRITEAOF):
            reply = self._reply(command[-1])
            self.replies.append(reply)
            return CommandResult(0, reply + "\n", "")
        raise AssertionError(f"unexpected command {command!r}")

    def _inspect(self) -> dict:
        return {
            "State": {"Status": self.status},
            "Mounts": [{"Type": "bind", "Source": str(self.data_dir), "Destination": DATA_DIR_IN_CONTAINER}],
            "Config": {"Env": ["PATH=/usr/bin", "REDIS_ARGS=--appendonly yes --appendfsync everysec"]},
        }

    def _render_info(self) -> str:
        return "# Persistence\r\n" + "".join(f"{key}:{value}\r\n" for key, value in self.info.items())

    def _reply(self, name: str) -> str:
        if self.info["loading"] == "1":
            return "LOADING Redis is loading the dataset in memory"
        if self.rejects and name in self.rejects:
            return MODULE_FORK_REPLY
        return self._bgsave() if name == "BGSAVE" else self._bgrewriteaof()

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
        if self.loading_polls:
            self.loading_polls -= 1
            self.info["loading"] = "1" if self.loading_polls else "0"
        if self._child is None or self._child == self.stuck:
            return
        self._child_polls -= 1
        if self._child_polls > 0:
            return
        field, self._child = self._child, None
        self.info[field] = "0"
        if field == self.fails:
            self.info[CHILD_STATUS[field]] = "err"
        elif field == "rdb_bgsave_in_progress":
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


@pytest.mark.parametrize("in_flight", ["rdb_bgsave_in_progress", "aof_rewrite_in_progress"])
def test_a_busy_server_is_waited_out_before_the_save(data_dir, dest, in_flight):
    fake = FakeFalkorDB(data_dir, in_flight=in_flight)

    assert _export(fake, dest) == 0

    assert fake.sent == [BGSAVE, BGREWRITEAOF]
    assert all(reply.startswith("Background") for reply in fake.replies), fake.replies
    assert (dest / MANIFEST).is_file()


def test_a_server_still_loading_is_waited_out(data_dir, dest):
    fake = FakeFalkorDB(data_dir, loading_polls=3)

    assert _export(fake, dest) == 0

    assert all(reply.startswith("Background") for reply in fake.replies), fake.replies
    assert (dest / MANIFEST).is_file()


def test_a_rewrite_that_never_finishes_times_out(data_dir, dest, capsys):
    fake = FakeFalkorDB(data_dir, stuck="aof_rewrite_in_progress")
    clock = FakeClock()

    assert _export(fake, dest, clock) == 1

    assert clock.now <= TIMEOUT_SECS + 10 * POLL_SECS
    err = capsys.readouterr().err
    assert "BGREWRITEAOF" in err and f"{TIMEOUT_SECS}s" in err
    assert "aof_rewrite_in_progress:1" in err
    assert not (dest / MANIFEST).exists()


@pytest.mark.parametrize("child", ["rdb_bgsave_in_progress", "aof_rewrite_in_progress"])
def test_a_failed_save_or_rewrite_fails_the_export(data_dir, dest, capsys, child):
    fake = FakeFalkorDB(data_dir, fails=child)
    clock = FakeClock()

    assert _export(fake, dest, clock) == 1

    assert clock.now < TIMEOUT_SECS
    assert f"{CHILD_STATUS[child]}:err" in capsys.readouterr().err
    assert not (dest / MANIFEST).exists()


def test_a_rejected_command_fails_without_waiting_out_the_timeout(data_dir, dest, capsys):
    fake = FakeFalkorDB(data_dir, rejects=("BGSAVE",))
    clock = FakeClock()

    assert _export(fake, dest, clock) == 1

    assert clock.now < TIMEOUT_SECS
    assert fake.sent == [BGSAVE]
    assert MODULE_FORK_REPLY in capsys.readouterr().err
    assert not (dest / MANIFEST).exists()


def test_an_unreachable_server_fails_the_export(data_dir, dest, capsys):
    fake = FakeFalkorDB(data_dir, exec_fails=True)

    assert _export(fake, dest) == 1

    err = capsys.readouterr().err
    assert f"docker exec {CONTAINER} redis-cli" in err
    assert "Error response from daemon" in err
    assert fake.sent == []
    assert not dest.exists()


@pytest.mark.parametrize(
    ("server", "finding"),
    [({"listed": False}, "no running falkordb container"), ({"status": "exited"}, "is exited, not running")],
    ids=["no-container", "exited"],
)
def test_a_container_that_is_not_running_fails_the_export(data_dir, dest, capsys, server, finding):
    fake = FakeFalkorDB(data_dir, **server)

    assert _export(fake, dest) == 1

    assert fake.execs == []
    assert finding in capsys.readouterr().err
    assert not dest.exists()
