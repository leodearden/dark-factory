"""Tests for the laptop battery guard's committed system units and for
install_laptop_battery_guard.sh, which installs them on leo-laptop (task 9051).

The installer tests copy scripts/tests/test_install_laptop_merge_verify_trim.py's
fake-ssh pattern: the ssh shim drops the host argument and runs the rest
through `bash -c` from $HOME, so the script's real single-quoted-heredoc
payloads and the piped guard run on this machine. sudo, systemctl and
journalctl are shims too, and REMOTE_DESTDIR roots every install path in a
temp dir, so this suite never touches a real host, /etc or system manager.
The one production-default run (an empty REMOTE_DESTDIR) uses a sudo shim
that only records its arguments.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from laptop_battery_guard_fixtures import power_supply, write_shims
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

DEPLOY = Path(__file__).parent.parent / "deploy"
GUARD_SRC = Path(__file__).parent.parent / "laptop_battery_guard.py"
UNIT = "laptop-battery-guard"
SERVICE = DEPLOY / f"{UNIT}.service"
TIMER = DEPLOY / f"{UNIT}.timer"
PYTHON_PREFIX = "/usr/bin/python3 "
SCRIPT = DEPLOY / "install_laptop_battery_guard.sh"
STAGE_REL = ".cache/laptop-battery-guard-staging"
READING = "BAT1 Discharging 42%"

SHIMS = {
    "ssh": 'shift\ncd "$HOME"\nexec bash -c "$*"\n',
    "sudo": (
        '[ "${1:-}" = "-n" ] && shift\n'
        '[ -z "${FAKE_NO_SUDO:-}" ] || exit 1\n'
        'if [ -n "${FAKE_SUDO_LOG_ONLY:-}" ]; then echo "$*" >>"$FAKE_STATE/sudo.log"; exit 0; fi\n'
        'exec "$@"\n'
    ),
    "systemctl": (
        'echo "$*" >>"$FAKE_STATE/systemctl.log"\n'
        'case "$*" in\n'
        '  "is-enabled "*) echo "${FAKE_IS_ENABLED:-enabled}" ;;\n'
        '  "is-active "*) echo "${FAKE_IS_ACTIVE:-active}" ;;\n'
        '  "start "*) exit "${FAKE_START_RC:-0}" ;;\n'
        '  "show -p Result --value "*) echo "${FAKE_RESULT:-success}" ;;\n'
        '  *) exit 0 ;;\n'
        'esac\n'
    ),
    "journalctl": 'echo "$*" >>"$FAKE_STATE/journalctl.log"\n',
}


def _directive(path, key):
    """Values of every `key=` line. Comment lines start with '#' or ';', so
    they never match."""
    prefix = f"{key}="
    return [
        line[len(prefix):].strip()
        for line in path.read_text().splitlines()
        if line.startswith(prefix)
    ]


class TestTimer:
    def test_polls_every_30s(self):
        assert _directive(TIMER, "OnUnitActiveSec") == ["30s"]

    def test_accuracy_keeps_the_poll_at_30s(self):
        assert _directive(TIMER, "AccuracySec") == ["1s"]

    def test_first_poll_a_minute_after_boot(self):
        assert _directive(TIMER, "OnBootSec") == ["1min"]

    def test_enabled_by_timers_target(self):
        assert _directive(TIMER, "WantedBy") == ["timers.target"]

    def test_declares_no_calendar(self):
        assert _directive(TIMER, "OnCalendar") == []


class TestService:
    def test_is_oneshot(self):
        assert _directive(SERVICE, "Type") == ["oneshot"]

    def test_runs_the_installed_guard_with_system_python(self):
        assert _directive(SERVICE, "ExecStart") == [
            f"{PYTHON_PREFIX}/usr/local/lib/{UNIT}/laptop_battery_guard.py",
        ]

    def test_runs_as_root(self):
        assert _directive(SERVICE, "User") == []

    def test_documents_the_installed_guard(self):
        (exec_start,) = _directive(SERVICE, "ExecStart")
        assert _directive(SERVICE, "Documentation") == [
            f"file://{exec_start.removeprefix(PYTHON_PREFIX)}",
        ]


def test_guard_source_is_executable():
    assert os.access(GUARD_SRC, os.X_OK)


@pytest.fixture
def world(tmp_path):
    """Fake host home, DESTDIR root, shim state and a battery reading 42%."""
    dirs = {name: tmp_path / name for name in ("home", "remote-root", "state", "bin")}
    for path in dirs.values():
        path.mkdir()
    write_shims(dirs["bin"], SHIMS)
    return {**dirs, "power_supply": power_supply(tmp_path, {"BAT1": ("Discharging", 42)})}


def run(world, *args, **env_overrides):
    env = {
        **os.environ,
        "PATH": f"{world['bin']}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(world["home"]),
        "SSH": str(world["bin"] / "ssh"),
        "LAPTOP_HOST": "dummy-host",
        "REMOTE_DESTDIR": str(world["remote-root"]),
        "BACKUP_LABEL": "testlabel",
        "FAKE_STATE": str(world["state"]),
        "BATTERY_GUARD_POWER_SUPPLY_DIR": str(world["power_supply"]),
        **env_overrides,
    }
    return subprocess.run(
        [str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60,
    )


def exec_target(world):
    (exec_start,) = _directive(SERVICE, "ExecStart")
    return world["remote-root"] / exec_start.removeprefix(PYTHON_PREFIX).lstrip("/")


def unit_dir(world):
    return world["remote-root"] / "etc/systemd/system"


def installed_files(world):
    return sorted(p for p in world["remote-root"].rglob("*") if p.is_file())


def shim_calls(world, shim):
    log = world["state"] / f"{shim}.log"
    return log.read_text().splitlines() if log.exists() else []


def systemctl_calls(world):
    return shim_calls(world, "systemctl")


def assert_installed(world):
    target = exec_target(world)
    assert os.access(target, os.X_OK)
    assert target.read_bytes() == GUARD_SRC.read_bytes()
    for source in (SERVICE, TIMER):
        assert (unit_dir(world) / source.name).read_bytes() == source.read_bytes()


class TestApply:
    def test_installs_enables_and_dry_runs_the_installed_guard(self, world):
        result = run(world)
        assert result.returncode == 0, result.stderr
        assert_installed(world)
        assert not (world["home"] / STAGE_REL).exists()
        calls = systemctl_calls(world)
        assert (
            calls.index("daemon-reload")
            < calls.index(f"enable --now {UNIT}.timer")
            < calls.index(f"start {UNIT}.service")
        )
        assert result.stdout.count(READING) == 2, "preflight and installed dry runs"
        assert "Result=success" in result.stdout

    def test_runs_the_guard_on_the_unit_interpreter_not_path_python3(self, world):
        write_shims(world["bin"], {"python3": 'echo "PATH python3 ran" >&2\nexit 97\n'})
        result = run(world)
        assert result.returncode == 0, result.stderr
        assert "PATH python3 ran" not in result.stderr
        assert result.stdout.count(READING) == 2
        assert PYTHON_PREFIX.strip() in result.stdout

    def test_empty_destdir_installs_to_the_absolute_paths(self, world):
        """The production default. ssh flattens its argv, so an unquoted empty
        REMOTE_DESTDIR would vanish and shift every later argument."""
        run(world, REMOTE_DESTDIR="", FAKE_SUDO_LOG_ONLY="1")
        stage = world["home"] / STAGE_REL
        lib = f"/usr/local/lib/{UNIT}"
        sudo = shim_calls(world, "sudo")
        assert f"mkdir -p {lib} /etc/systemd/system" in sudo
        for name in (f"{UNIT}.service", f"{UNIT}.timer"):
            assert f"install -m 0644 {stage}/{name} /etc/systemd/system/{name}" in sudo
        assert (
            f"install -m 0755 {stage}/laptop_battery_guard.py {lib}/laptop_battery_guard.py"
            in sudo
        )
        assert installed_files(world) == []

    def test_reapply_is_idempotent(self, world):
        assert run(world).returncode == 0
        result = run(world)
        assert result.returncode == 0, result.stderr
        assert_installed(world)
        assert list(world["remote-root"].rglob("*.bak-testlabel")) == []

    def test_backs_up_a_differing_unit(self, world):
        old_timer = unit_dir(world) / f"{UNIT}.timer"
        old_timer.parent.mkdir(parents=True)
        old_timer.write_text("[Timer]\nOnUnitActiveSec=5min\n")
        result = run(world)
        assert result.returncode == 0, result.stderr
        backup = world["remote-root"] / f"usr/local/lib/{UNIT}/{UNIT}.timer.bak-testlabel"
        assert backup.read_text() == "[Timer]\nOnUnitActiveSec=5min\n"
        assert old_timer.read_bytes() == TIMER.read_bytes()


class TestCheck:
    @pytest.mark.parametrize("flag", ["--check", "--dry-run"])
    def test_prints_the_reading_and_changes_nothing(self, world, flag):
        result = run(world, flag)
        assert result.returncode == 0, result.stderr
        assert READING in result.stdout
        assert "Nothing changed" in result.stdout
        assert installed_files(world) == []
        calls = systemctl_calls(world)
        assert not [c for c in calls if "enable" in c or "daemon-reload" in c]


class TestRefusals:
    def test_without_passwordless_sudo(self, world):
        result = run(world, FAKE_NO_SUDO="1")
        assert result.returncode == 1
        assert "passwordless sudo" in result.stderr
        assert installed_files(world) == []

    def test_unreachable_host(self, world):
        result = run(world, SSH="false")
        assert result.returncode == 1
        assert "ssh dummy-host failed" in result.stderr
        assert installed_files(world) == []

    def test_inactive_timer_fails(self, world):
        result = run(world, FAKE_IS_ACTIVE="inactive")
        assert result.returncode == 1
        assert f"{UNIT}.timer" in result.stderr
        assert "not active" in result.stderr

    @pytest.mark.parametrize("env", [{"FAKE_START_RC": "1"}, {"FAKE_RESULT": "exit-code"}])
    def test_failed_first_run_fails_and_shows_the_journal(self, world, env):
        result = run(world, **env)
        assert result.returncode == 1
        assert f"{UNIT}.service" in result.stderr
        assert "Result=success" not in result.stdout
        assert any(f"-u {UNIT}.service" in call for call in shim_calls(world, "journalctl"))

    def test_unexpected_argument(self, world):
        result = run(world, "--force")
        assert result.returncode == 2
        assert "unexpected argument" in result.stderr
