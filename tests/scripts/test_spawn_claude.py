"""Tests for skills/spawn/spawn-claude.sh exit-code contract.

Headless: no real display needed.  A fake ``claude`` binary on PATH provides
parametrisable exit codes, and fake terminal binaries route to specific dispatch
branches by name (the script dispatches on the first word of $CLAUDE_TERMINAL_CMD).
"""

from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import textwrap
import time

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[2]
SPAWN_SCRIPT = REPO_ROOT / "skills" / "spawn" / "spawn-claude.sh"

# Branch routing: the script dispatches on the first word of $CLAUDE_TERMINAL_CMD.
FOREGROUND_NAMES = ["gnome-terminal", "xterm", "kitty"]
DETACHING_NAMES = ["konsole", "custom-term"]  # konsole branch + custom *) branch
ALL_NAMES = FOREGROUND_NAMES + DETACHING_NAMES

# ---------------------------------------------------------------------------
# Fake terminal scripts
# ---------------------------------------------------------------------------

# Universal FOREGROUND fake terminal: finds the first 'bash' token in argv and
# exec's from there.  Works for every emulator branch's argv shape:
#   gnome-terminal --wait [--title=T] -- bash -c "$inner"
#   xterm [-T title]  -e bash -c "$inner"
#   kitty [--title T] bash -c "$inner"
#   konsole [-p ...]  -e bash -c "$inner"
#   custom-term       -- bash -c "$inner"   (via eval in the *) branch)
_FOREGROUND_TERM_SCRIPT = textwrap.dedent("""\
    #!/usr/bin/env bash
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == "bash" ]]; then
        exec "$@"
      fi
      shift
    done
    exit 1
""")

# Detaching fake terminal: runs the payload in a new session (setsid), records
# the session-leader pid to a file, then exits 0 (konsole-like).  The test
# sends SIGHUP to the leader's pgid to simulate window-close.
_DETACHING_TERM_TEMPLATE = textwrap.dedent("""\
    #!/usr/bin/env bash
    # Find 'bash' in argv, then run that payload as a detached session leader.
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == "bash" ]]; then
        break
      fi
      shift
    done
    # $@ is now: bash -c "$inner"
    # Run it in a new session so it gets its own pgid.
    setsid "$@" &
    leader_pid=$!
    # Write the pid so the test can signal the group.
    echo "$leader_pid" > {pidfile}
    exit 0
""")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bin_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    return d


def _write_fake_claude(bin_dir: pathlib.Path, exit_code: int) -> None:
    p = bin_dir / "claude"
    p.write_text(f"#!/usr/bin/env bash\nexit {exit_code}\n")
    p.chmod(0o755)


def _write_foreground_terminal(bin_dir: pathlib.Path, name: str) -> None:
    p = bin_dir / name
    p.write_text(_FOREGROUND_TERM_SCRIPT)
    p.chmod(0o755)


def _write_detaching_terminal(
    bin_dir: pathlib.Path, name: str, pidfile: pathlib.Path
) -> None:
    script = _DETACHING_TERM_TEMPLATE.format(pidfile=str(pidfile))
    p = bin_dir / name
    p.write_text(script)
    p.chmod(0o755)


def _base_env(bin_dir: pathlib.Path, terminal_name: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(bin_dir) + ":" + env.get("PATH", "")
    env["CLAUDE_TERMINAL_CMD"] = terminal_name
    env.pop("ESCALATION_TERMINAL_CMD", None)
    # Keep the genuine-launcher-failure grace short so tests don't hang.
    env["SPAWN_LAUNCH_GRACE_SECS"] = "2"
    return env


def _run_spawn(
    env: dict[str, str],
    cwd: pathlib.Path,
    *,
    timeout: int = 30,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(SPAWN_SCRIPT), str(cwd), "false", "", "test prompt"],
        env=env,
        capture_output=True,
        timeout=timeout,
    )


# ===========================================================================
# Step-1 tests: exit-code propagation
# ===========================================================================
# RED today for exit-3: gnome-terminal/xterm/kitty use ``|| exit 127``;
# konsole/custom use ``wait $! || { rm sentinel; exit 127; }`` — both conflate
# a non-zero session exit (propagated through the emulator) with launcher
# failure.  GREEN after step-2.


@pytest.mark.parametrize("terminal_name", ALL_NAMES)
def test_session_exit_0_propagates(tmp_path: pathlib.Path, terminal_name: str) -> None:
    """A spawned session that exits 0 must yield exit 0 from spawn-claude.sh."""
    bin_dir = _make_bin_dir(tmp_path)
    _write_fake_claude(bin_dir, exit_code=0)
    _write_foreground_terminal(bin_dir, terminal_name)
    env = _base_env(bin_dir, terminal_name)

    result = _run_spawn(env, tmp_path)
    assert result.returncode == 0, (
        f"[{terminal_name}] Expected exit 0 (session success), "
        f"got {result.returncode}\nstderr: {result.stderr.decode()}"
    )


@pytest.mark.parametrize("terminal_name", ALL_NAMES)
def test_session_nonzero_exit_propagates_not_127(
    tmp_path: pathlib.Path, terminal_name: str
) -> None:
    """A non-zero session exit (3) must propagate; must NOT be conflated with 127.

    RED today for all branches: the ``|| exit 127`` (foreground) and
    ``wait $! || { rm sentinel; exit 127; }`` (detaching) patterns both
    interpret the emulator's non-zero exit as a launcher failure and discard
    the sentinel.
    """
    bin_dir = _make_bin_dir(tmp_path)
    _write_fake_claude(bin_dir, exit_code=3)
    _write_foreground_terminal(bin_dir, terminal_name)
    env = _base_env(bin_dir, terminal_name)

    result = _run_spawn(env, tmp_path)
    assert result.returncode == 3, (
        f"[{terminal_name}] Expected exit 3 (session exit code), "
        f"got {result.returncode}\nstderr: {result.stderr.decode()}"
    )


# ===========================================================================
# Step-3 tests: window-close sentinel contract
# ===========================================================================
# RED today: inner payload has no trap, so a SIGHUP to the foreground group
# kills the inner bash before `echo $ec > sentinel` runs; await_sentinel hangs.
# GREEN after step-4.


@pytest.mark.parametrize("terminal_name", DETACHING_NAMES)
def test_window_close_yields_129_not_hang(
    tmp_path: pathlib.Path, terminal_name: str
) -> None:
    """Closing the terminal window while the session is alive must yield 129, not hang.

    RED today: the sentinel is never written on SIGHUP (no trap), so
    await_sentinel hangs forever.  Fixed by the inner-payload EXIT trap.
    """
    bin_dir = _make_bin_dir(tmp_path)

    # Long-running claude: sleep 300 (default SIGHUP disposition: terminate).
    fake_claude = bin_dir / "claude"
    fake_claude.write_text("#!/usr/bin/env bash\nexec sleep 300\n")
    fake_claude.chmod(0o755)

    pidfile = tmp_path / "leader.pid"
    _write_detaching_terminal(bin_dir, terminal_name, pidfile)
    env = _base_env(bin_dir, terminal_name)

    # Launch spawn-claude.sh in its OWN session so signaling the payload group
    # can't reach the pytest process.
    proc = subprocess.Popen(
        [str(SPAWN_SCRIPT), str(tmp_path), "false", "", "test prompt"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    # Wait for the fake terminal to write the leader pid (max 5 s).
    deadline = time.monotonic() + 5.0
    while not pidfile.exists() and time.monotonic() < deadline:
        time.sleep(0.05)

    assert pidfile.exists(), (
        "Fake terminal did not write leader pid within 5s — test harness broken"
    )

    leader_pid = int(pidfile.read_text().strip())
    # Send SIGHUP to the entire process group of the session leader.
    os.killpg(leader_pid, signal.SIGHUP)

    # spawn-claude.sh must complete within 10 s and return 129.
    try:
        rc = proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail(
            f"[{terminal_name}] spawn-claude.sh hung after window-close "
            f"(await_sentinel never unblocked — SIGHUP sentinel gap not fixed)"
        )
    assert rc == 129, (
        f"[{terminal_name}] Expected exit 129 (window closed while alive), "
        f"got {rc}\nstderr: {proc.stderr.read().decode()}"  # type: ignore[union-attr]
    )


# ===========================================================================
# Step-5 tests: contract-guard tests for 127, 126, 2
# ===========================================================================
# These exercise paths that should already be correct; they lock the contract
# before step-6 touches every branch.


@pytest.mark.parametrize("terminal_name", ["xterm", "custom-term"])
def test_genuine_launcher_failure_yields_127(
    tmp_path: pathlib.Path, terminal_name: str
) -> None:
    """A terminal that exits non-zero WITHOUT writing the sentinel → exit 127.

    The launcher genuinely failed to start the payload (e.g. the emulator
    binary crashed before opening a window).  This is the true launcher-
    failure case and must yield 127.
    """
    bin_dir = _make_bin_dir(tmp_path)
    _write_fake_claude(bin_dir, exit_code=0)

    # Fake terminal that exits immediately with rc=1, never running the payload
    # (so no sentinel is ever written).
    fail_term = bin_dir / terminal_name
    fail_term.write_text("#!/usr/bin/env bash\nexit 1\n")
    fail_term.chmod(0o755)

    env = _base_env(bin_dir, terminal_name)
    result = _run_spawn(env, tmp_path, timeout=15)
    assert result.returncode == 127, (
        f"[{terminal_name}] Genuine launcher failure must yield 127, "
        f"got {result.returncode}\nstderr: {result.stderr.decode()}"
    )


@pytest.mark.skipif(
    __import__("platform").system() == "Darwin",
    reason="exit-126 path requires non-Darwin host",
)
def test_no_emulator_found_yields_126(tmp_path: pathlib.Path) -> None:
    """When no emulator is found, spawn-claude.sh must exit 126."""
    import shutil as _shutil

    bin_dir = _make_bin_dir(tmp_path)
    _write_fake_claude(bin_dir, exit_code=0)

    # Build a minimal system-bin that has only the utilities the script needs
    # but NOT any terminal emulator (gnome-terminal/konsole/kitty/xterm).
    # We look up each utility by its absolute path using the full system PATH
    # so we get the real binary, then symlink it in our minimal dir.
    sys_bin = tmp_path / "sys_bin"
    sys_bin.mkdir()
    for util in ["bash", "mktemp", "sleep", "cat", "rm", "uname"]:
        src = _shutil.which(util)
        if src:
            (sys_bin / util).symlink_to(src)

    env = dict(os.environ)
    env["PATH"] = str(bin_dir) + ":" + str(sys_bin)
    env.pop("CLAUDE_TERMINAL_CMD", None)
    env.pop("ESCALATION_TERMINAL_CMD", None)

    result = _run_spawn(env, tmp_path, timeout=10)
    assert result.returncode == 126, (
        f"No emulator must yield 126, got {result.returncode}\n"
        f"stderr: {result.stderr.decode()}"
    )


def test_bad_usage_yields_2(tmp_path: pathlib.Path) -> None:
    """Passing wrong number of arguments must exit 2."""
    result = subprocess.run(
        [str(SPAWN_SCRIPT), "only-one-arg"],
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 2, (
        f"Bad usage must yield 2, got {result.returncode}"
    )
