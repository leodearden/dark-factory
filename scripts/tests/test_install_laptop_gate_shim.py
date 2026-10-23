"""Tests for install_laptop_gate_shim.sh — drives the script via subprocess
against a fake orchestrator wrapper in a temp directory (task 6565).

The ssh shim drops the host argument and runs the rest through `bash -c`
locally, so the script's real single-quoted-heredoc payload runs on this
machine. The fake wrapper mirrors the shape of leo-laptop's
/usr/local/bin/orchestrator (PATH export, 600 s watchdog default, START/EXIT
log lines, subshell-form exec line) with the target binary replaced by a stub.
The fake `sudo` stands in for privilege over a read-only wrapper directory.
This suite never touches a real host.
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "deploy" / "install_laptop_gate_shim.sh"

REAL_BIN = "/home/leo/src/dark-factory/.venv/bin/orchestrator"
OLD_LINE = f'{REAL_BIN} "$@" 2> >(tee -a "$_LOG" >&2)'
NEW_LINE = f'"${{ORCH_GATE_BIN:-{REAL_BIN}}}" "$@" 2> >(tee -a "$_LOG" >&2)'

WRAPPER_TEMPLATE = """#!/bin/bash
export PATH="/home/leo/.cargo/bin:$PATH"
export ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS="${{ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS:-600}}"
_LOG="${{TEST_LOG:?}}"
echo "=== START $1 shell-pid=$$" >> "$_LOG"
{preamble}{exec_line}
_rc=$?
echo "=== EXIT rc=$_rc" >> "$_LOG"
exit $_rc
"""


IGNORES_GATE_BIN = (
    f'function {REAL_BIN} {{ echo "stub $*"; }}\n'
    "unset ORCH_GATE_BIN\n"
)


def wrapper_text(exec_line=OLD_LINE, drop=None, preamble=""):
    text = WRAPPER_TEMPLATE.format(exec_line=exec_line, preamble=preamble)
    if drop:
        text = "".join(ln for ln in text.splitlines(keepends=True) if drop not in ln)
    return text


@pytest.fixture
def world(tmp_path):
    bindir = tmp_path / "usr-local-bin"
    bindir.mkdir()
    shim = bindir / "orchestrator"
    shim.write_text(wrapper_text())
    shim.chmod(0o775)
    shims = tmp_path / "shims"
    shims.mkdir()
    (shims / "ssh").write_text('#!/usr/bin/env bash\nshift\nexec bash -c "$*"\n')
    (shims / "ssh").chmod(0o755)
    return {"tmp": tmp_path, "bindir": bindir, "shim": shim, "shims": shims}


def add_fake_sudo(world, *, available=True):
    """`sudo -n <cmd>` runs <cmd> with the wrapper directory temporarily writable."""
    body = (
        'shift\n[ "$1" = true ] && exit 0\n'
        f'chmod u+w "{world["bindir"]}"\n"$@"\nrc=$?\n'
        f'chmod u-w "{world["bindir"]}"\nexit $rc\n'
        if available else "exit 1\n"
    )
    sudo = world["shims"] / "sudo"
    sudo.write_text("#!/usr/bin/env bash\n" + body)
    sudo.chmod(0o755)


def run(world, *args):
    env = {
        **os.environ,
        "PATH": f"{world['shims']}{os.pathsep}{os.environ['PATH']}",
        "SSH": str(world["shims"] / "ssh"),
        "LAPTOP_HOST": "dummy-host",
        "SHIM_PATH": str(world["shim"]),
        "BACKUP_LABEL": "testlabel",
        "TEST_LOG": str(world["tmp"] / "invocations.log"),
    }
    return subprocess.run([str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60)


def invoke_wrapper(world, *args, **env):
    return subprocess.run(
        [str(world["shim"]), *args],
        env={**os.environ, "TEST_LOG": str(world["tmp"] / "invocations.log"), **env},
        capture_output=True, text=True, timeout=30,
    )


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK)


def test_apply_rewrites_only_the_exec_line_and_keeps_host_facts(world):
    result = run(world)
    assert result.returncode == 0, result.stderr
    assert world["shim"].read_text() == wrapper_text(NEW_LINE)
    assert stat.S_IMODE(world["shim"].stat().st_mode) == 0o775
    assert (world["bindir"] / "orchestrator.bak-testlabel").read_text() == wrapper_text()


def test_installed_wrapper_runs_the_gate_bin_and_keeps_the_exit_log_line(world):
    run(world)
    out = invoke_wrapper(world, "verify-merge", ORCH_GATE_BIN="/bin/echo")
    assert out.returncode == 0
    assert out.stdout.strip() == "verify-merge"
    log = (world["tmp"] / "invocations.log").read_text()
    assert "START verify-merge" in log and "EXIT rc=0" in log


def test_unset_gate_bin_defaults_to_the_shared_tree_binary(world):
    run(world)
    assert f"${{ORCH_GATE_BIN:-{REAL_BIN}}}" in world["shim"].read_text()


def test_reapply_is_a_noop_that_exits_zero(world):
    run(world)
    first = world["shim"].read_text()
    (world["bindir"] / "orchestrator.bak-testlabel").unlink()
    result = run(world)
    assert result.returncode == 0, result.stderr
    assert world["shim"].read_text() == first
    assert not (world["bindir"] / "orchestrator.bak-testlabel").exists()


def test_check_fails_on_the_old_wrapper_and_writes_nothing(world):
    result = run(world, "--check")
    assert result.returncode != 0
    assert "does not honour ORCH_GATE_BIN" in result.stderr
    assert world["shim"].read_text() == wrapper_text()
    assert not (world["bindir"] / "orchestrator.bak-testlabel").exists()


def test_check_passes_on_the_new_wrapper(world):
    run(world)
    result = run(world, "--check")
    assert result.returncode == 0, result.stderr
    assert "honours ORCH_GATE_BIN" in result.stdout


def test_check_fails_when_the_wrapper_is_missing(world):
    world["shim"].unlink()
    result = run(world, "--check")
    assert result.returncode != 0
    assert "does not exist" in result.stderr


@pytest.mark.parametrize(
    "marker",
    ["export PATH=", "ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS", "_LOG=", " START ", " EXIT rc="],
)
def test_wrapper_that_lost_a_host_fact_is_refused_untouched(world, marker):
    world["shim"].write_text(wrapper_text(drop=marker))
    before = world["shim"].read_text()
    result = run(world)
    assert result.returncode != 0
    assert "refusing to touch" in result.stderr
    assert world["shim"].read_text() == before


def test_unrecognised_exec_line_is_refused_untouched(world):
    world["shim"].write_text(wrapper_text("/opt/other/orchestrator \"$@\" 2> >(tee -a \"$_LOG\" >&2)"))
    before = world["shim"].read_text()
    result = run(world)
    assert result.returncode != 0
    assert "not recognised" in result.stderr
    assert world["shim"].read_text() == before


def assert_wrapper_is_untouched_and_executable(world, before):
    assert world["shim"].read_bytes() == before
    assert os.access(world["shim"], os.X_OK)


def test_apply_whose_probe_fails_restores_the_backup_and_exits_nonzero(world):
    """The wrapper stubs its default binary and unsets ORCH_GATE_BIN, so the
    edited exec line is ignored: the post-install probe must roll back."""
    world["shim"].write_text(wrapper_text(preamble=IGNORES_GATE_BIN))
    before = world["shim"].read_bytes()
    result = run(world)
    assert result.returncode != 0
    assert "wrapper ignores ORCH_GATE_BIN" in result.stderr
    assert "restored" in result.stderr
    assert_wrapper_is_untouched_and_executable(world, before)


def test_check_fails_on_an_installed_wrapper_that_ignores_gate_bin(world):
    world["shim"].write_text(wrapper_text(NEW_LINE, preamble=IGNORES_GATE_BIN))
    before = world["shim"].read_bytes()
    result = run(world, "--check")
    assert result.returncode != 0
    assert "wrapper ignores ORCH_GATE_BIN" in result.stderr
    assert_wrapper_is_untouched_and_executable(world, before)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
class TestUnwritableWrapperDirectory:
    @pytest.fixture(autouse=True)
    def readonly_dir(self, world):
        world["bindir"].chmod(0o555)
        yield
        world["bindir"].chmod(0o755)

    def test_without_sudo_states_it_writes_nothing_and_exits_nonzero(self, world):
        add_fake_sudo(world, available=False)
        before = world["shim"].read_text()
        result = run(world)
        assert result.returncode != 0
        assert "not writable" in result.stderr
        assert "Nothing was written" in result.stderr
        assert "sudo sed -i" in result.stderr
        assert world["shim"].read_text() == before
        assert sorted(p.name for p in world["bindir"].iterdir()) == ["orchestrator"]

    def test_with_passwordless_sudo_installs_by_atomic_rename(self, world):
        add_fake_sudo(world)
        inode = world["shim"].stat().st_ino
        result = run(world)
        assert result.returncode == 0, result.stderr
        assert world["shim"].read_text() == wrapper_text(NEW_LINE)
        assert world["shim"].stat().st_ino != inode
        assert (world["bindir"] / "orchestrator.bak-testlabel").read_text() == wrapper_text()
        assert sorted(p.name for p in world["bindir"].iterdir()) == [
            "orchestrator", "orchestrator.bak-testlabel",
        ]

    def test_sudo_that_passes_the_probe_but_refuses_the_install_changes_nothing(self, world):
        sudo = world["shims"] / "sudo"
        sudo.write_text('#!/usr/bin/env bash\nshift\n[ "$1" = true ] && exit 0\nexit 1\n')
        sudo.chmod(0o755)
        before = world["shim"].read_text()
        result = run(world)
        assert result.returncode != 0
        assert "is unchanged" in result.stderr and "sudo sed -i" in result.stderr
        assert world["shim"].read_text() == before
        assert sorted(p.name for p in world["bindir"].iterdir()) == ["orchestrator"]

    def test_check_needs_no_write_access(self, world):
        add_fake_sudo(world, available=False)
        result = run(world, "--check")
        assert result.returncode != 0
        assert "does not honour ORCH_GATE_BIN" in result.stderr


def test_unknown_argument_exits_2(world):
    result = run(world, "--bogus")
    assert result.returncode == 2
