"""The shared build-cache store: its pruner and its installer.

Both live under scripts/deploy/ (PRD C13 of
plans/canonical-build-path-shared-cache-prd.md). The pruner is driven as a CLI
over a fixture store laid out as sccache's remote layout `<k0>/<k1>/<k2>/<key>`,
with a `systemctl` shim on PATH standing in for the user manager. The installer
runs in a scratch HOME against fake_user_systemctl.py, which really launches the
store unit it wrote, plus `tailscale` and `loginctl` shims. Tests that need a
WebDAV server run a private real rclone on loopback; no test touches the real
user manager, ~/.config, ~/.cache or ports 4226/4227.
"""
from __future__ import annotations

import base64
import contextlib
import http.client
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from cli_subprocess_timeout import cli_timeout_from_env
from laptop_battery_guard_fixtures import write_shims
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
PRUNER = DEPLOY / "build_cache_store_prune.py"
INSTALLER = DEPLOY / "install_build_cache_store.sh"
FAKE_SYSTEMCTL = Path(__file__).resolve().parent / "fake_user_systemctl.py"
CLI_TIMEOUT = cli_timeout_from_env("BUILD_CACHE_STORE_TEST_TIMEOUT")

STORE_UNIT = "build-cache-store.service"
PRUNE_SERVICE = "build-cache-store-prune.service"
PRUNE_TIMER = "build-cache-store-prune.timer"
STORE_UNITS = (STORE_UNIT, PRUNE_SERVICE, PRUNE_TIMER)
FAKE_TAILSCALE_ADDR = "127.0.0.2"
DEFAULT_CAP_BYTES = 300 << 30
HEX = "0123456789abcdef"
SYSTEMCTL_MUTATIONS = {"daemon-reload", "enable", "disable", "start", "restart", "stop"}
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


_TAILSCALE_SHIM = """\
if [ "${1:-}" = ip ] && [ "${2:-}" = -4 ]; then
  printf '%s\\n' "${FAKE_TS_ADDR-127.0.0.2}"
  exit 0
fi
exit 1
"""

_LOGINCTL_SHIM = """\
printf '%s\\n' "${FAKE_LINGER:-yes}"
"""


class StoreWorld:
    """A scratch HOME, a fake user manager and a private port for the installer."""

    def __init__(self, tmp_path: Path) -> None:
        self.home = tmp_path / "home"
        self.home.mkdir()
        self.state = tmp_path / "state"
        self.state.mkdir()
        self.bin = tmp_path / "bin"
        write_shims(
            self.bin,
            {
                "tailscale": _TAILSCALE_SHIM,
                "loginctl": _LOGINCTL_SHIM,
                "systemctl": f'exec {shlex.quote(sys.executable)} -I {shlex.quote(str(FAKE_SYSTEMCTL))} "$@"\n',
            },
        )
        self.port = free_port()
        inherited = {k: v for k, v in os.environ.items() if not k.startswith("XDG_")}
        self.env = {
            **inherited,
            "HOME": str(self.home),
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_SYSTEMD_STATE": str(self.state),
            "BUILD_CACHE_STORE_PORT": str(self.port),
        }
        self.store = self.home / ".cache" / "build-cache-store"
        self.credentials_file = self.home / ".config" / "build-cache-store" / "credentials"
        self.unit_dir = self.home / ".config" / "systemd" / "user"
        self.pruner_copy = self.home / ".local" / "lib" / "build-cache-store" / PRUNER.name

    def stub_rclone(self) -> None:
        """An rclone that preflight can resolve but nothing ever serves from."""
        write_shims(self.bin, {"rclone": "exit 0\n"})

    def install(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(INSTALLER), *args],
            env={**self.env, **env},
            cwd=self.home.parent,
            capture_output=True,
            text=True,
            timeout=2 * CLI_TIMEOUT,
        )

    def credentials(self) -> dict[str, str]:
        pairs = (line.partition("=") for line in self.credentials_file.read_text().splitlines())
        return {name: value for name, sep, value in pairs if sep}

    def auth(self) -> tuple[str, str]:
        credentials = self.credentials()
        return credentials["RCLONE_USER"], credentials["RCLONE_PASS"]

    def unit_text(self, unit: str) -> str:
        return (self.unit_dir / unit).read_text()

    def exec_start(self, unit: str) -> list[str]:
        lines = [
            line.removeprefix("ExecStart=")
            for line in self.unit_text(unit).splitlines()
            if line.startswith("ExecStart=")
        ]
        assert len(lines) == 1, lines
        return shlex.split(lines[0])

    def systemctl_calls(self) -> list[list[str]]:
        log = self.state / "systemctl.log"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    def mutations(self) -> list[list[str]]:
        return [
            call for call in self.systemctl_calls() if SYSTEMCTL_MUTATIONS & set(call)
        ]

    def store_pid(self) -> int:
        return int((self.state / f"{STORE_UNIT}.pid").read_text())

    def is_enabled(self, unit: str) -> bool:
        return (self.state / f"{unit}.enabled").exists()

    def kill_services(self) -> None:
        for pid_file in self.state.glob("*.pid"):
            with contextlib.suppress(ProcessLookupError, ValueError):
                os.killpg(int(pid_file.read_text()), signal.SIGKILL)


@pytest.fixture
def store_world(tmp_path: Path) -> Iterator[StoreWorld]:
    world = StoreWorld(tmp_path)
    try:
        yield world
    finally:
        world.kill_services()


@pytest.fixture
def installed(store_world: StoreWorld) -> StoreWorld:
    proc = store_world.install()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return store_world


def leaf_layout() -> set[Path]:
    return (
        {Path(a) for a in HEX}
        | {Path(a, b) for a in HEX for b in HEX}
        | {Path(a, b, c) for a in HEX for b in HEX for c in HEX}
    )


def addr_values(command: list[str]) -> list[str]:
    return [command[i + 1] for i, arg in enumerate(command) if arg == "--addr"]


@needs_rclone
class TestStoreInstall:
    def test_apply_lays_out_store_credentials_and_pruner(self, installed: StoreWorld) -> None:
        dirs = {path.relative_to(installed.store) for path in installed.store.rglob("*") if path.is_dir()}
        assert len(dirs) == 16 + 256 + 4096
        assert dirs == leaf_layout()
        assert installed.credentials_file.stat().st_mode & 0o777 == 0o600
        assert set(installed.credentials()) >= {"RCLONE_USER", "RCLONE_PASS"}
        assert installed.pruner_copy.read_bytes() == PRUNER.read_bytes()
        assert os.access(installed.pruner_copy, os.X_OK)

    def test_store_round_trips_on_both_addresses(self, installed: StoreWorld) -> None:
        auth = installed.auth()
        for address in ("127.0.0.1", FAKE_TAILSCALE_ADDR):
            url = f"http://{address}:{installed.port}/0/0/0/{os.urandom(8).hex()}"
            body = os.urandom(2048)
            assert http_request("PUT", url, auth, body)[0] in (200, 201, 204)
            assert http_request("GET", url, auth) == (200, body)
            assert http_request("GET", url)[0] == 401

    def test_store_unit_binds_only_loopback_and_tailscale(self, installed: StoreWorld) -> None:
        command = installed.exec_start(STORE_UNIT)
        assert addr_values(command) == [
            f"127.0.0.1:{installed.port}",
            f"{FAKE_TAILSCALE_ADDR}:{installed.port}",
        ]
        text = installed.unit_text(STORE_UNIT)
        lines = text.splitlines()
        assert "Restart=always" in lines
        assert "Environment=RCLONE_CONFIG=/dev/null" in lines
        assert installed.auth()[1] not in text

    def test_prune_timer_is_enabled_and_runs_the_installed_pruner(
        self, installed: StoreWorld
    ) -> None:
        assert installed.is_enabled(PRUNE_TIMER)
        assert installed.is_enabled(STORE_UNIT)
        assert installed.exec_start(PRUNE_SERVICE) == [
            "/usr/bin/python3",
            "-I",
            str(installed.pruner_copy),
            str(installed.store),
            "--cap-bytes",
            str(DEFAULT_CAP_BYTES),
            "--rclone-unit",
            STORE_UNIT,
        ]

    def test_check_after_apply_passes(self, installed: StoreWorld) -> None:
        proc = installed.install("--check")
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_reapply_changes_nothing_and_keeps_the_store_running(
        self, installed: StoreWorld
    ) -> None:
        before = {unit: installed.unit_text(unit) for unit in STORE_UNITS}
        credentials = installed.credentials_file.read_bytes()
        pid = installed.store_pid()
        calls_before = len(installed.systemctl_calls())

        proc = installed.install()

        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert {unit: installed.unit_text(unit) for unit in STORE_UNITS} == before
        assert installed.credentials_file.read_bytes() == credentials
        assert installed.store_pid() == pid
        rerun_calls = installed.systemctl_calls()[calls_before:]
        assert not [call for call in rerun_calls if "restart" in call]

    def test_check_detects_a_drifted_unit(self, installed: StoreWorld) -> None:
        with (installed.unit_dir / STORE_UNIT).open("a") as unit:
            unit.write("# hand edit\n")
        proc = installed.install("--check")
        assert proc.returncode == 1
        assert STORE_UNIT in proc.stderr


class TestStoreCheck:
    def test_check_on_a_fresh_home_fails_and_writes_nothing(
        self, store_world: StoreWorld
    ) -> None:
        store_world.stub_rclone()
        proc = store_world.install("--check")
        assert proc.returncode == 1
        assert str(store_world.credentials_file) in proc.stderr
        assert str(store_world.store) in proc.stderr
        assert list(store_world.home.iterdir()) == []
        assert store_world.mutations() == []


class TestStoreRefusals:
    @pytest.mark.parametrize(
        ("env", "named"),
        [
            ({"FAKE_TS_ADDR": ""}, "tailscale"),
            ({"FAKE_TS_ADDR": "0.0.0.0"}, "0.0.0.0"),
            ({"FAKE_LINGER": "no"}, "Linger"),
        ],
        ids=["empty-tailscale-address", "wildcard-address", "lingering-off"],
    )
    def test_preflight_refuses_and_writes_nothing(
        self, store_world: StoreWorld, env: dict[str, str], named: str
    ) -> None:
        store_world.stub_rclone()
        proc = store_world.install(**env)
        assert proc.returncode == 1
        assert named in proc.stderr
        assert list(store_world.home.iterdir()) == []
        assert store_world.mutations() == []

    def test_unexpected_argument_exits_two(self, store_world: StoreWorld) -> None:
        store_world.stub_rclone()
        proc = store_world.install("--bogus")
        assert proc.returncode == 2
        assert list(store_world.home.iterdir()) == []


HOST_CONFIG_KEYS = {
    "enabled",
    "store_url",
    "store_user",
    "store_password",
    "fallback_dir",
    "fallback_cap",
    "policy",
}


def write_credentials(path: Path, lines: str) -> Path:
    path.write_text(lines)
    path.chmod(0o600)
    return path


class TestHostConfig:
    def test_laptop_writes_only_the_host_config(
        self, store_world: StoreWorld, tmp_path: Path
    ) -> None:
        credentials = write_credentials(
            tmp_path / "copied-credentials", "RCLONE_USER=reify-canon\nRCLONE_PASS=0123abcd\n"
        )
        host_config = tmp_path / "cfg" / "reify-canon" / "host.toml"
        store_url = f"http://{FAKE_TAILSCALE_ADDR}:{store_world.port}"

        proc = store_world.install(
            "--host-config",
            BUILD_CACHE_STORE_CREDENTIALS=str(credentials),
            REIFY_CANON_HOST_CONFIG=str(host_config),
            REIFY_CANON_ENABLED="true",
            REIFY_CANON_POLICY="mixed",
            BUILD_CACHE_STORE_URL=store_url,
        )

        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert host_config.stat().st_mode & 0o777 == 0o600
        assert host_config.parent.stat().st_mode & 0o777 == 0o700
        config = tomllib.loads(host_config.read_text())
        assert set(config) == HOST_CONFIG_KEYS
        assert config["enabled"] is True
        assert config["policy"] == "mixed"
        assert config["store_url"] == store_url
        assert config["store_user"] == "reify-canon"
        assert config["store_password"] == "0123abcd"
        assert config["fallback_dir"].startswith(f"{store_world.home}/")
        assert config["fallback_cap"] == "50G"
        assert not store_world.store.exists()
        assert not store_world.unit_dir.exists()
        assert store_world.mutations() == []

    def test_defaults_are_disabled_shared_and_loopback(
        self, store_world: StoreWorld, tmp_path: Path
    ) -> None:
        credentials = write_credentials(
            tmp_path / "credentials", "RCLONE_USER=reify-canon\nRCLONE_PASS=feed\n"
        )
        proc = store_world.install(
            "--host-config", BUILD_CACHE_STORE_CREDENTIALS=str(credentials)
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        host_config = store_world.home / ".config" / "reify-canon" / "host.toml"
        config = tomllib.loads(host_config.read_text())
        assert config["enabled"] is False
        assert config["policy"] == "shared"
        assert config["store_url"] == f"http://127.0.0.1:{store_world.port}"

    def test_rerun_is_byte_identical(self, store_world: StoreWorld, tmp_path: Path) -> None:
        credentials = write_credentials(
            tmp_path / "credentials", "RCLONE_USER=reify-canon\nRCLONE_PASS=feed\n"
        )
        host_config = tmp_path / "reify-canon" / "host.toml"
        env = {
            "BUILD_CACHE_STORE_CREDENTIALS": str(credentials),
            "REIFY_CANON_HOST_CONFIG": str(host_config),
            "REIFY_CANON_ENABLED": "true",
        }
        assert store_world.install("--host-config", **env).returncode == 0
        first = host_config.read_bytes()
        assert store_world.install("--host-config", **env).returncode == 0
        assert host_config.read_bytes() == first

    @pytest.mark.parametrize(
        ("env", "credential_lines"),
        [
            ({"REIFY_CANON_POLICY": "fast"}, "RCLONE_USER=u\nRCLONE_PASS=p\n"),
            ({"REIFY_CANON_ENABLED": "yes"}, "RCLONE_USER=u\nRCLONE_PASS=p\n"),
            ({}, "RCLONE_USER=u\n"),
        ],
        ids=["unknown-policy", "non-bool-enabled", "no-password"],
    )
    def test_invalid_input_writes_no_file(
        self,
        store_world: StoreWorld,
        tmp_path: Path,
        env: dict[str, str],
        credential_lines: str,
    ) -> None:
        credentials = write_credentials(tmp_path / "credentials", credential_lines)
        host_config = tmp_path / "reify-canon" / "host.toml"
        proc = store_world.install(
            "--host-config",
            BUILD_CACHE_STORE_CREDENTIALS=str(credentials),
            REIFY_CANON_HOST_CONFIG=str(host_config),
            **env,
        )
        assert proc.returncode == 1
        assert not host_config.exists()

    @needs_rclone
    def test_check_proves_the_store_takes_this_hosts_credentials(
        self, installed: StoreWorld
    ) -> None:
        assert installed.install("--host-config").returncode == 0
        host_config = installed.home / ".config" / "reify-canon" / "host.toml"

        proc = installed.install("--check", "--host-config")
        assert proc.returncode == 0, proc.stdout + proc.stderr

        text = host_config.read_text()
        password = tomllib.loads(text)["store_password"]
        host_config.write_text(text.replace(password, "wrong-password"))
        proc = installed.install("--host-config", "--check")
        assert proc.returncode == 1
        assert "credentials" in proc.stderr

        host_config.write_text(text)
        host_config.chmod(0o644)
        proc = installed.install("--check", "--host-config")
        assert proc.returncode == 1
        assert "644" in proc.stderr
