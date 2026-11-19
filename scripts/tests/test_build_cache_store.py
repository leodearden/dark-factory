"""The shared build-cache store: its pruner and its installer.

Both live under scripts/deploy/ (PRD C13 of
plans/canonical-build-path-shared-cache-prd.md). The pruner is driven as a CLI
over a fixture store laid out as sccache's remote layout `<k0>/<k1>/<k2>/<key>`,
with a `systemctl` shim on PATH standing in for the user manager. Tests that
need a WebDAV server run a private real rclone on loopback; no test touches the
real user manager, ~/.config, ~/.cache or ports 4226/4227.
"""
from __future__ import annotations

import base64
import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from cli_subprocess_timeout import cli_timeout_from_env
from laptop_battery_guard_fixtures import write_shims

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
PRUNER = DEPLOY / "build_cache_store_prune.py"
CLI_TIMEOUT = cli_timeout_from_env("BUILD_CACHE_STORE_TEST_TIMEOUT")

STORE_UNIT = "build-cache-store.service"
DAY = 86400.0
HOUR = 3600.0

needs_rclone = pytest.mark.skipif(
    shutil.which("rclone") is None,
    reason="rclone absent: the store tests need a real rclone (gate ι installs it on leo-laptop)",
)

_SYSTEMCTL_SHIM = """\
printf '%s\\n' "$*" >> "$SYSTEMCTL_LOG"
if [ "${1:-}" = --user ] && [ "${2:-}" = show ]; then
  printf '%s\\n' "${FAKE_RCLONE_PID:-0}"
fi
exit "${FAKE_SYSTEMCTL_RC:-0}"
"""


def put_object(
    store: Path, key: str, size: int, mtime_age: float, atime_age: float | None = None
) -> Path:
    """A store object of *size* bytes last modified *mtime_age* seconds ago."""
    path = store / key[0] / key[1] / key[2] / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    now = time.time()
    access_age = mtime_age if atime_age is None else atime_age
    os.utime(path, (now - access_age, now - mtime_age))
    return path


class PrunerWorld:
    def __init__(self, tmp_path: Path) -> None:
        self.store = tmp_path / "store"
        self.store.mkdir()
        self.bin = tmp_path / "bin"
        write_shims(self.bin, {"systemctl": _SYSTEMCTL_SHIM})
        self.systemctl_log = tmp_path / "systemctl.log"

    def run(
        self, *args: str, cap: int, rclone_pid: int = 0, systemctl_rc: int = 0
    ) -> tuple[int, dict[str, Any]]:
        env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "SYSTEMCTL_LOG": str(self.systemctl_log),
            "FAKE_RCLONE_PID": str(rclone_pid),
            "FAKE_SYSTEMCTL_RC": str(systemctl_rc),
        }
        proc = subprocess.run(
            [sys.executable, "-I", str(PRUNER), *args, "--cap-bytes", str(cap)],
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT,
            env=env,
        )
        report = json.loads(proc.stdout) if proc.stdout.strip() else {}
        return proc.returncode, report

    def prune(self, *extra: str, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        return self.run(str(self.store), *extra, **kwargs)

    def systemctl_calls(self) -> list[str]:
        if not self.systemctl_log.exists():
            return []
        return self.systemctl_log.read_text().splitlines()


@pytest.fixture
def world(tmp_path: Path) -> PrunerWorld:
    return PrunerWorld(tmp_path)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_request(
    method: str,
    url: str,
    auth: tuple[str, str] | None = None,
    data: bytes | None = None,
) -> tuple[int, bytes]:
    """Status and body; a body cut short is returned as far as it got."""
    request = urllib.request.Request(url, data=data, method=method)
    if auth is not None:
        token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")
    try:
        with _NO_PROXY.open(request, timeout=5) as response:
            try:
                return response.status, response.read()
            except http.client.IncompleteRead as cut:
                return response.status, cut.partial
    except urllib.error.HTTPError as err:
        return err.code, b""


def wait_for_http(url: str, deadline_secs: float = 15.0) -> None:
    deadline = time.monotonic() + deadline_secs
    while time.monotonic() < deadline:
        try:
            http_request("GET", url)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.1)
    raise AssertionError(f"nothing answered at {url} within {deadline_secs}s")


@pytest.fixture
def private_rclone(tmp_path: Path) -> Iterator[tuple[subprocess.Popen[bytes], str, tuple[str, str]]]:
    """A real rclone serving tmp_path/'store' on a private loopback port."""
    store = tmp_path / "store"
    store.mkdir(exist_ok=True)
    port = free_port()
    auth = ("tester", "secret-for-tests")
    env = {
        **os.environ,
        "RCLONE_CONFIG": "/dev/null",
        "RCLONE_USER": auth[0],
        "RCLONE_PASS": auth[1],
    }
    with (tmp_path / "rclone.log").open("wb") as log:
        proc = subprocess.Popen(
            ["rclone", "serve", "webdav", str(store), "--addr", f"127.0.0.1:{port}"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        base = f"http://127.0.0.1:{port}"
        wait_for_http(base + "/")
        yield proc, base, auth
    finally:
        proc.kill()
        proc.wait(timeout=10)


class TestPrune:
    def test_over_cap_removes_exactly_the_oldest(self, world: PrunerWorld) -> None:
        paths = [
            put_object(world.store, f"{age}aa{age}key", 100, mtime_age=age * DAY)
            for age in range(1, 6)
        ]
        code, report = world.prune(cap=300)
        assert code == 0
        survivors = {path for path in paths if path.exists()}
        assert survivors == set(paths[:3])
        assert report["removed"] == 2
        assert report["removed_bytes"] == 200
        assert report["remaining_bytes"] == 300
        assert report["remaining_bytes"] <= report["cap_bytes"]
        assert report["over_cap"] is False

    def test_last_use_is_the_later_of_atime_and_mtime(self, world: PrunerWorld) -> None:
        read_recently = put_object(
            world.store, "aaa-read", 1000, mtime_age=4 * DAY, atime_age=HOUR
        )
        untouched = put_object(world.store, "bbb-idle", 1000, mtime_age=2 * DAY)
        code, report = world.prune(cap=1500)
        assert code == 0
        assert read_recently.exists()
        assert not untouched.exists()
        assert report["removed"] == 1

    def test_young_files_are_never_removed(self, world: PrunerWorld) -> None:
        old = [
            put_object(world.store, f"{age}old", 100, mtime_age=age * DAY)
            for age in (1, 2)
        ]
        young = put_object(world.store, "new-put", 1000, mtime_age=60)
        code, report = world.prune(cap=10)
        assert code == 0
        assert young.exists()
        assert not any(path.exists() for path in old)
        assert report["skipped_young"] == 1
        assert report["over_cap"] is True
        assert report["remaining_bytes"] == 1000

    def test_directories_are_never_removed(self, world: PrunerWorld) -> None:
        emptied = put_object(world.store, "abc-one", 100, mtime_age=DAY).parent
        precreated = world.store / "f" / "f" / "f"
        precreated.mkdir(parents=True)
        code, report = world.prune(cap=0)
        assert code == 0
        assert report["removed"] == 1
        assert emptied.is_dir()
        assert precreated.is_dir()

    def test_check_removes_nothing_and_never_calls_systemctl(
        self, world: PrunerWorld
    ) -> None:
        paths = [
            put_object(world.store, f"{age}chk", 100, mtime_age=age * DAY)
            for age in range(1, 4)
        ]
        code, report = world.prune("--check", cap=100)
        assert code == 0
        assert all(path.exists() for path in paths)
        assert report["check"] is True
        assert report["removed"] == 2
        assert report["signalled"] is None
        assert world.systemctl_calls() == []

    def test_under_cap_removes_nothing_and_never_calls_systemctl(
        self, world: PrunerWorld
    ) -> None:
        path = put_object(world.store, "abc-under", 100, mtime_age=3 * DAY)
        code, report = world.prune(cap=1000)
        assert code == 0
        assert path.exists()
        assert report["removed"] == 0
        assert report["over_cap"] is False
        assert world.systemctl_calls() == []

    def test_store_not_running_still_prunes(self, world: PrunerWorld) -> None:
        path = put_object(world.store, "abc-gone", 100, mtime_age=3 * DAY)
        code, report = world.prune(cap=0, rclone_pid=0)
        assert code == 0
        assert not path.exists()
        assert report["signalled"] is None
        assert world.systemctl_calls() == [
            f"--user show -p MainPID --value {STORE_UNIT}"
        ]

    def test_signals_the_store_units_main_pid_once(
        self, world: PrunerWorld, tmp_path: Path
    ) -> None:
        hups = tmp_path / "hups"
        ready = tmp_path / "ready"
        listener = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-c",
                "import signal, sys, time\n"
                "hups, ready = sys.argv[1], sys.argv[2]\n"
                "signal.signal(signal.SIGHUP, lambda *_: open(hups, 'a').write('hup\\n'))\n"
                "open(ready, 'w').close()\n"
                "time.sleep(60)\n",
                str(hups),
                str(ready),
            ]
        )
        try:
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            for key in ("abc-1", "abd-2", "abe-3"):
                put_object(world.store, key, 100, mtime_age=3 * DAY)
            code, report = world.prune(cap=0, rclone_pid=listener.pid)
            deadline = time.monotonic() + 5
            while not hups.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
        finally:
            listener.kill()
            listener.wait(timeout=10)
        assert code == 0
        assert report["removed"] == 3
        assert report["signalled"] == listener.pid
        assert hups.read_text() == "hup\n"

    def test_systemctl_failure_exits_one(self, world: PrunerWorld) -> None:
        put_object(world.store, "abc-fail", 100, mtime_age=3 * DAY)
        code, _ = world.prune(cap=0, systemctl_rc=1)
        assert code == 1

    def test_missing_store_root_exits_one(self, world: PrunerWorld, tmp_path: Path) -> None:
        code, _ = world.run(str(tmp_path / "no-such-store"), cap=0)
        assert code == 1

    @needs_rclone
    def test_pruned_object_is_404_through_a_running_rclone(
        self,
        world: PrunerWorld,
        private_rclone: tuple[subprocess.Popen[bytes], str, tuple[str, str]],
    ) -> None:
        proc, base, auth = private_rclone
        key = "0f0" + "e" * 61
        (world.store / "0" / "f" / "0").mkdir(parents=True)
        url = f"{base}/0/f/0/{key}"
        body = os.urandom(4096)
        assert http_request("PUT", url, auth, body)[0] in (200, 201, 204)
        assert http_request("GET", url, auth) == (200, body)

        on_disk = world.store / "0" / "f" / "0" / key
        aged = time.time() - 3 * DAY
        os.utime(on_disk, (aged, aged))
        code, report = world.prune(cap=0, rclone_pid=proc.pid)
        assert code == 0
        assert report["signalled"] == proc.pid

        deadline = time.monotonic() + 5
        status = http_request("GET", url, auth)[0]
        while status != 404 and time.monotonic() < deadline:
            time.sleep(0.1)
            status = http_request("GET", url, auth)[0]
        assert status == 404
        assert proc.poll() is None
