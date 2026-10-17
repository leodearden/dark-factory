"""Tests for install_laptop_merge_verify_trim.sh — drives the script via
subprocess against a fake reify source checkout and a fake "remote" home.

Mirrors scripts/tests/test_enable_laptop_persistent_worktree.py's fake-ssh
pattern: the ssh shim drops the host argument and runs the rest through
`bash -c` from $HOME, the way a real ssh login shell would, so the script's
real single-quoted-heredoc payloads run on this machine. systemctl, crontab,
findmnt and journalctl are shims on PATH that record calls and keep state in
temp files; this suite never touches a real host, crontab or user unit.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "deploy" / "install_laptop_merge_verify_trim.sh"
UNIT = "reify-merge-verify-trim"
LIB_REL = ".local/lib/reify-verify-host"

GOOD_TRIM = "#!/usr/bin/env bash\nset -euo pipefail\necho trimmed\n"
GOOD_GC = "#!/usr/bin/env bash\ncase \"$wt\" in *_merge-verify*) continue ;; esac\n"
GOOD_SERVICE = (
    "[Service]\nType=oneshot\n"
    f"ExecStart=%h/{LIB_REL}/prune-unconsulted-target.sh --window-days 3\n"
)
GOOD_TIMER = "[Timer]\nOnCalendar=hourly\n[Install]\nWantedBy=timers.target\n"

SHIMS = {
    "ssh": 'shift\ncd "$HOME"\nexec bash -c "$*"\n',
    "systemctl": (
        'echo "$*" >>"$FAKE_STATE/systemctl.log"\n'
        'case "$*" in\n'
        '  "--user show-environment") [ -z "${FAKE_NO_BUS:-}" ] ;;\n'
        '  "--user is-enabled "*) echo enabled ;;\n'
        '  "--user is-active "*) echo active ;;\n'
        '  "--user start "*) exit "${FAKE_START_RC:-0}" ;;\n'
        '  "--user show -p Result --value "*) echo "${FAKE_RESULT:-success}" ;;\n'
        '  *) exit 0 ;;\n'
        'esac\n'
    ),
    "crontab": (
        'if [ "$1" = "-l" ]; then\n'
        '  [ -f "$FAKE_STATE/crontab" ] || { echo "no crontab for leo" >&2; exit 1; }\n'
        '  cat "$FAKE_STATE/crontab"\n'
        'elif [ "$1" = "-" ]; then\n'
        '  cat >"$FAKE_STATE/crontab"\n'
        'fi\n'
    ),
    "findmnt": 'echo "${FAKE_MOUNT_OPTS:-rw,relatime}"\n',
    "journalctl": 'echo "fake journal for $*"\n',
}


@pytest.fixture
def world(tmp_path):
    """Fake reify source checkout, fake remote home + reify tree, shims."""
    source = tmp_path / "reify-src"
    for rel, text in {
        "scripts/prune-unconsulted-target.sh": GOOD_TRIM,
        "scripts/gc-worktree-targets.sh": GOOD_GC,
        f"deploy/systemd/{UNIT}.service": GOOD_SERVICE,
        f"deploy/systemd/{UNIT}.timer": GOOD_TIMER,
    }.items():
        path = source / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (source / "scripts/prune-unconsulted-target.sh").chmod(0o755)

    remote_reify = tmp_path / "remote-reify"
    (remote_reify / ".worktrees/_merge-verify/target").mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    cron_old = remote_reify / "scripts/gc-worktree-targets.sh"
    (state / "crontab").write_text(
        f"0 */2 * * * {cron_old} >> /tmp/gc-worktree-targets.log 2>&1\n"
        "15 3 * * * /usr/bin/true\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in SHIMS.items():
        shim = bin_dir / name
        shim.write_text("#!/usr/bin/env bash\n" + body)
        shim.chmod(0o755)
    return {
        "source": source, "remote_reify": remote_reify, "home": home,
        "state": state, "bin": bin_dir, "cron_old": cron_old,
    }


def run(world, *args, **env_overrides):
    env = {
        **os.environ,
        "PATH": f"{world['bin']}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(world["home"]),
        "XDG_CONFIG_HOME": str(world["home"] / ".config"),
        "SSH": str(world["bin"] / "ssh"),
        "LAPTOP_HOST": "dummy-host",
        "REIFY_SOURCE_ROOT": str(world["source"]),
        "REMOTE_REIFY_ROOT": str(world["remote_reify"]),
        "BACKUP_LABEL": "testlabel",
        "FAKE_STATE": str(world["state"]),
        **env_overrides,
    }
    return subprocess.run(
        [str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60,
    )


def lib(world):
    return world["home"] / LIB_REL


def unit_dir(world):
    return world["home"] / ".config/systemd/user"


def systemctl_calls(world):
    log = world["state"] / "systemctl.log"
    return log.read_text().splitlines() if log.exists() else []


def assert_nothing_installed(world):
    assert not (unit_dir(world) / f"{UNIT}.timer").exists()
    assert not (lib(world) / "prune-unconsulted-target.sh").exists()
    assert str(world["cron_old"]) in (world["state"] / "crontab").read_text()
    assert not any("enable" in c for c in systemctl_calls(world))


def test_apply_installs_repoints_cron_enables_and_runs_once(world):
    result = run(world)
    assert result.returncode == 0, result.stderr
    installed_trim = lib(world) / "prune-unconsulted-target.sh"
    assert installed_trim.read_text() == GOOD_TRIM
    assert os.access(installed_trim, os.X_OK)
    assert os.access(lib(world) / "gc-worktree-targets.sh", os.X_OK)
    assert (unit_dir(world) / f"{UNIT}.service").read_text() == GOOD_SERVICE
    assert (unit_dir(world) / f"{UNIT}.timer").read_text() == GOOD_TIMER
    assert not (lib(world) / ".staging").exists()

    crontab = (world["state"] / "crontab").read_text()
    assert f"0 */2 * * * {lib(world)}/gc-worktree-targets.sh >> /tmp/gc-worktree-targets.log 2>&1" in crontab
    assert str(world["cron_old"]) not in crontab
    assert "15 3 * * * /usr/bin/true" in crontab
    assert str(world["cron_old"]) in (lib(world) / "crontab.bak-testlabel").read_text()

    calls = systemctl_calls(world)
    assert "--user daemon-reload" in calls
    assert f"--user enable --now {UNIT}.timer" in calls
    assert calls.index(f"--user enable --now {UNIT}.timer") < calls.index(f"--user start {UNIT}.service")


def test_second_apply_is_a_no_op_for_the_crontab(world):
    assert run(world).returncode == 0
    crontab_after_first = (world["state"] / "crontab").read_text()
    result = run(world)
    assert result.returncode == 0, result.stderr
    assert "already runs" in result.stdout
    assert (world["state"] / "crontab").read_text() == crontab_after_first


def test_check_mode_changes_nothing(world):
    result = run(world, "--check")
    assert result.returncode == 0, result.stderr
    assert "Nothing changed" in result.stdout
    assert_nothing_installed(world)


@pytest.mark.parametrize("rel", [
    "scripts/prune-unconsulted-target.sh",
    "scripts/gc-worktree-targets.sh",
    f"deploy/systemd/{UNIT}.service",
    f"deploy/systemd/{UNIT}.timer",
])
def test_missing_deliverable_refuses_before_touching_the_host(world, rel):
    (world["source"] / rel).unlink()
    result = run(world)
    assert result.returncode == 1
    assert "deliverable missing" in result.stderr
    assert_nothing_installed(world)


@pytest.mark.parametrize("mutate, message", [
    (lambda s: (s / "scripts/prune-unconsulted-target.sh").write_text(
        GOOD_TRIM + 'source "$(dirname "$0")/lib.sh"\n'), "sources another file"),
    (lambda s: (s / "scripts/prune-unconsulted-target.sh").chmod(0o644), "not executable"),
    (lambda s: (s / f"deploy/systemd/{UNIT}.service").write_text(
        "[Service]\nExecStart=/home/leo/src/reify/scripts/prune-unconsulted-target.sh\n"),
     "installed copy"),
    (lambda s: (s / "scripts/gc-worktree-targets.sh").write_text("#!/usr/bin/env bash\n"),
     "still wipe the warm target"),
])
def test_contract_violation_refuses(world, mutate, message):
    mutate(world["source"])
    result = run(world)
    assert result.returncode == 1
    assert message in result.stderr
    assert_nothing_installed(world)


@pytest.mark.parametrize("env, message", [
    ({"FAKE_MOUNT_OPTS": "rw,noatime"}, "noatime"),
    ({"FAKE_NO_BUS": "1"}, "no systemd --user bus"),
])
def test_preflight_failure_refuses(world, env, message):
    result = run(world, **env)
    assert result.returncode == 1
    assert message in result.stderr
    assert_nothing_installed(world)


def test_missing_target_dir_refuses(world):
    (world["remote_reify"] / ".worktrees/_merge-verify/target").rmdir()
    result = run(world)
    assert result.returncode == 1
    assert "missing" in result.stderr
    assert_nothing_installed(world)


@pytest.mark.parametrize("env", [{"FAKE_START_RC": "1"}, {"FAKE_RESULT": "exit-code"}])
def test_failed_first_run_exits_non_zero(world, env):
    result = run(world, **env)
    assert result.returncode == 1
    assert "fake journal" in result.stderr


def test_unexpected_argument_is_cli_misuse(world):
    result = run(world, "--force")
    assert result.returncode == 2
    assert "unexpected argument" in result.stderr
