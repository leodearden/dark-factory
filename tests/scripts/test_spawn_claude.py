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
import sys
import textwrap
import time
from datetime import UTC, datetime, timedelta

import pytest  # pyright: ignore[reportMissingImports]

REPO_ROOT = pathlib.Path(__file__).parents[2]
SPAWN_SCRIPT = REPO_ROOT / "skills" / "spawn" / "spawn-claude.sh"

# Insert this worktree's orchestrator/src onto sys.path (ahead of any
# editable install pointing at a different checkout) so `import
# orchestrator.session_registry` resolves to the module spawn-claude.sh
# itself invokes by absolute path (task 2285) -- letting this file assert
# in-process against the exact same record/reap contract. Also gives the
# task-2298 (Fleet Cockpit C7) two-way boundary test direct, in-process
# access to session_hooks.run_session_start -- the already-landed C1/C2
# consumer of the CLAUDE_SPAWN_PARENT_ID this file's sibling-mode tests
# export.
_ORCH_SRC = REPO_ROOT / "orchestrator" / "src"
if str(_ORCH_SRC) not in sys.path:
    sys.path.insert(0, str(_ORCH_SRC))

from orchestrator import session_hooks  # noqa: E402  # pyright: ignore[reportAttributeAccessIssue]
from orchestrator import session_registry  # noqa: E402  # pyright: ignore[reportAttributeAccessIssue]

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
#
# Real terminal emulators (konsole, gnome-terminal, xterm, kitty, macOS
# Terminal) reset child signal dispositions to SIG_DFL before launching the
# child shell.  This fake terminal must do the same: `env --default-signal=HUP,TERM`
# un-ignores HUP and TERM across the exec boundary so the payload bash's
# `trap 'exit 129' HUP` (and `trap 'exit 143' TERM`) can actually install and
# fire.  Without this reset, an inherited SIGHUP=SIG_IGN (from a detached CI
# harness or a preexec_fn in tests) silently makes the trap a POSIX no-op —
# a non-interactive bash cannot trap a signal that was SIG_IGN on entry.
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
    # env --default-signal=HUP,TERM resets those dispositions to SIG_DFL,
    # mirroring what a real terminal emulator does for its child shell.
    setsid env --default-signal=HUP,TERM "$@" &
    leader_pid=$!
    # Publish the leader pid IMMEDIATELY — the readiness marker written by the
    # fake claude binary (which spawn-claude.sh invokes only AFTER arming the
    # EXIT/HUP/TERM traps) is the synchronization gate for SIGHUP delivery.
    # A blind pre-publish sleep is a timing-fragile substitute and is
    # load-sensitive under full-suite xdist contention (task-1925).
    echo "$leader_pid" > {pidfile}
    exit 0
""")

# Stress-variant detaching terminal: injects a trap-install delay INSIDE the
# payload (before $inner runs) to expose timing-fragile synchronization.
# Publishes the leader pid IMMEDIATELY — no blind sleep — so the readiness
# marker (written by the fake claude once spawn-claude.sh's traps are armed)
# is the only valid synchronization gate for SIGHUP delivery.
#
# {delay}   — float seconds injected before $inner (e.g. 1.0)
# {pidfile} — path where the leader pid is written
_STRESS_DETACHING_TERM_TEMPLATE = textwrap.dedent("""\
    #!/usr/bin/env bash
    # Stress variant: inject a trap-install delay before the payload runs.
    # Simulates slow bash startup to expose timing-fragile synchronization.
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == "bash" ]]; then
        break
      fi
      shift
    done
    # $1=bash  $2=-c  $3=<inner script>
    inner="$3"
    # Prepend delay BEFORE inner runs (before traps are armed in the payload).
    setsid env --default-signal=HUP,TERM bash -c "sleep {delay}; $inner" &
    leader_pid=$!
    # Publish pid IMMEDIATELY — the readiness marker (written by the fake
    # claude binary once spawn-claude.sh has armed the EXIT/HUP/TERM traps)
    # is the synchronization gate, not a blind sleep.
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


def _write_fake_claude_with_readiness(
    bin_dir: pathlib.Path, readyfile: pathlib.Path
) -> None:
    """Write a fake ``claude`` that signals readiness then sleeps indefinitely.

    spawn-claude.sh's $inner arms EXIT, HUP, and TERM traps BEFORE invoking
    ``claude``, so the moment this binary writes *readyfile* the HUP trap is
    guaranteed to be installed in the enclosing bash process.  This makes
    SIGHUP-after-readiness-gate deterministic regardless of load.
    """
    p = bin_dir / "claude"
    p.write_text(
        f"#!/usr/bin/env bash\n"
        f"echo ready > {readyfile!s}\n"
        f"exec sleep 300\n"
    )
    p.chmod(0o755)


def _write_fake_claude_writing_result(bin_dir: pathlib.Path) -> None:
    """Write a fake ``claude`` that writes an outcome-header result file to
    ``$CLAUDE_SPAWN_RESULT_FILE`` (falling back to /dev/null when unset —
    the pre-T5 / fail-soft shape), then exits 0.
    """
    p = bin_dir / "claude"
    p.write_text(
        "#!/usr/bin/env bash\n"
        "cat > \"${CLAUDE_SPAWN_RESULT_FILE:-/dev/null}\" <<'EOF'\n"
        "---\n"
        "outcome: done\n"
        "changed: none\n"
        "action_needed: none\n"
        "---\n"
        "Test prose.\n"
        "EOF\n"
        "exit 0\n"
    )
    p.chmod(0o755)


def _write_fake_claude_capturing_prompt(
    bin_dir: pathlib.Path, capture_file: pathlib.Path
) -> None:
    """Write a fake ``claude`` that captures its prompt argument (``$1``) to
    *capture_file*, then exits 0.

    With ``skip_perms="false"`` (the default baked into ``_run_spawn``),
    ``$flags`` is empty, so ``$1`` is exactly the prompt string
    spawn-claude.sh assembled into ``$inner`` -- letting a test inspect
    whatever transformation (e.g. a result-handback trailer) the script
    applied before invoking claude.
    """
    p = bin_dir / "claude"
    p.write_text(
        f"#!/usr/bin/env bash\n"
        f'printf %s "$1" > {capture_file!s}\n'
        f"exit 0\n"
    )
    p.chmod(0o755)


def _write_fake_claude_capturing_prompt_and_writing_result(
    bin_dir: pathlib.Path, capture_file: pathlib.Path
) -> None:
    """Write a fake ``claude`` that captures its prompt argument (``$1``) to
    *capture_file* AND writes an outcome-header result file to
    ``$CLAUDE_SPAWN_RESULT_FILE`` (falling back to /dev/null when unset),
    then exits 0.

    Combines ``_write_fake_claude_capturing_prompt`` and
    ``_write_fake_claude_writing_result`` so a single spawn can lock BOTH
    halves of the result-handback protocol at once: whether a trailer was
    appended to the prompt, and whether anything landed in a result.md.
    """
    p = bin_dir / "claude"
    p.write_text(
        f"#!/usr/bin/env bash\n"
        f'printf %s "$1" > {capture_file!s}\n'
        "cat > \"${CLAUDE_SPAWN_RESULT_FILE:-/dev/null}\" <<'EOF'\n"
        "---\n"
        "outcome: done\n"
        "changed: none\n"
        "action_needed: none\n"
        "---\n"
        "Test prose.\n"
        "EOF\n"
        "exit 0\n"
    )
    p.chmod(0o755)


def _wait_for_path(path: pathlib.Path, timeout: float) -> None:
    """Poll until *path* exists, raising ``AssertionError`` on timeout."""
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"Timed out after {timeout}s waiting for {path} to appear"
            )
        time.sleep(0.05)


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
    # Isolate the session-registry writes spawn-claude.sh now performs
    # (task 2285) to a tmp dir sibling of bin_dir, so this suite never reads
    # or writes the real ~/.claude/fleet tree.
    env["CLAUDE_FLEET_ROOT"] = str(bin_dir.parent / "fleet")
    # Isolate the started-watchdog's transcript-appearance probe (task 2286,
    # Attention Rail T4) to a fresh, empty tmp dir sibling of bin_dir, so the
    # watchdog never scans (or finds stray evidence in) the real
    # ~/.claude/projects tree.
    env["CLAUDE_PROJECTS_DIR"] = str(bin_dir.parent / "projects")
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

    Hardened (task-1925) to use an explicit readiness handshake instead of a
    blind pre-signal sleep, making SIGHUP-after-trap-install deterministic
    under full-suite xdist load.

    Synchronization contract (load-independent):
      1. _DETACHING_TERM_TEMPLATE publishes the leader pid IMMEDIATELY after
         setsid (no blind sleep).
      2. The fake claude writes a readiness marker file before exec sleep 300.
      3. spawn-claude.sh's $inner arms EXIT/HUP/TERM traps BEFORE invoking
         claude, so the readiness marker is a sound proof that the HUP trap
         is installed.
      4. The test waits for BOTH the pidfile AND the readiness marker before
         sending SIGHUP — SIGHUP is always delivered after trap installation.
    """
    bin_dir = _make_bin_dir(tmp_path)

    pidfile = tmp_path / "leader.pid"
    readyfile = tmp_path / "claude.ready"

    # Fake claude that writes a readiness marker then sleeps indefinitely.
    # The marker is written only after spawn-claude.sh's $inner has armed the
    # EXIT/HUP/TERM traps, so it is a deterministic proof the HUP trap is live.
    _write_fake_claude_with_readiness(bin_dir, readyfile)

    _write_detaching_terminal(bin_dir, terminal_name, pidfile)
    env = _base_env(bin_dir, terminal_name)

    # Launch spawn-claude.sh in its OWN session so signaling the payload group
    # can't reach the pytest process.
    #
    # preexec_fn forces SIGHUP=SIG_IGN on the spawned process (and its whole
    # descendant chain) BEFORE exec.  This reproduces the detached-harness
    # condition deterministically: SIG_IGN is inherited across fork+exec, and
    # a non-interactive bash cannot trap a signal that was SIG_IGN on entry
    # (POSIX), so the payload's `trap 'exit 129' HUP` becomes a silent no-op
    # unless the fake terminal resets the disposition — exactly what a real
    # terminal emulator does for its child shell.
    proc = subprocess.Popen(
        [str(SPAWN_SCRIPT), str(tmp_path), "false", "", "test prompt"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN),
    )

    # Wait for BOTH the leader pid AND the readiness marker.
    # The pidfile appears first (published immediately by the terminal);
    # the readyfile appears only after spawn-claude.sh has armed its traps and
    # invoked the fake claude — proof that SIGHUP will land on a live HUP trap.
    _wait_for_path(pidfile, timeout=5.0)
    _wait_for_path(readyfile, timeout=10.0)

    leader_pid = int(pidfile.read_text().strip())
    # Send SIGHUP to the entire process group of the session leader.
    # HUP trap is now guaranteed to be installed → exit 129.
    os.killpg(leader_pid, signal.SIGHUP)

    # Must-not-hang guard — NOT a latency SLA.
    # The success path takes ~2-4s (await_sentinel 2s poll + pidfile handshake).
    # A genuine hang is infinite (no sentinel ever written), so 15s cleanly
    # separates pass from hang while staying well under the global 60s
    # pytest-timeout (shared/pyproject.toml, timeout_method=signal), so this
    # descriptive pytest.fail still fires before the blunt signal-kill.
    try:
        rc = proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail(
            f"[{terminal_name}] spawn-claude.sh hung after window-close "
            f"(await_sentinel never unblocked — SIGHUP sentinel gap not fixed)"
        )
    else:
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


# ===========================================================================
# task-1925 step-4: readiness-handshake stress test
# ===========================================================================
# Guards that a >=1s trap-install delay does NOT cause a hang when SIGHUP is
# gated on the post-trap readiness marker rather than a blind pre-signal sleep.
#
# Design rationale: spawn-claude.sh's $inner arms EXIT/HUP/TERM traps BEFORE
# calling `claude`, so the fake `claude` writing a readiness file is a sound
# proof that the HUP trap is already armed.  The stress terminal template
# injects a {delay}s sleep BEFORE $inner runs, ensuring that a blind 0.2s
# sleep would deliver SIGHUP before the HUP trap is installed → default
# SIGHUP terminates the payload → no sentinel → hang.  With the readiness
# gate, SIGHUP is deferred until after the trap is in place → exit 129.


def test_window_close_129_robust_to_delayed_trap_install(
    tmp_path: pathlib.Path,
) -> None:
    """SIGHUP after readiness gate yields 129 even with a >=1s trap-install delay.

    Stress test for the readiness-handshake synchronization mechanism
    (task-1925 step-4).

    The stress terminal template injects a {DELAY}s sleep BEFORE $inner runs,
    so the HUP trap cannot be installed within DELAY seconds of the leader pid
    being published.  A blind 0.2s pre-signal sleep would deliver SIGHUP
    before the trap is armed → payload terminated by SIG_DFL → no sentinel →
    hang.  With the readiness gate (waiting for the fake claude's marker file,
    which is written only after spawn-claude.sh has armed the traps), SIGHUP
    is always delivered after trap installation → exit 129.
    """
    DELAY = 1.0  # exceeds the legacy 200ms blind-sleep margin
    terminal_name = "custom-term"
    bin_dir = _make_bin_dir(tmp_path)

    pidfile = tmp_path / "leader.pid"
    readyfile = tmp_path / "claude.ready"

    # Fake claude that signals readiness BEFORE sleeping (proves traps are armed,
    # since spawn-claude.sh installs traps before invoking claude).
    _write_fake_claude_with_readiness(bin_dir, readyfile)

    # Stress terminal: injects DELAY before $inner, publishes pid immediately.
    script = _STRESS_DETACHING_TERM_TEMPLATE.format(
        delay=DELAY, pidfile=str(pidfile)
    )
    term = bin_dir / terminal_name
    term.write_text(script)
    term.chmod(0o755)

    env = _base_env(bin_dir, terminal_name)

    proc = subprocess.Popen(
        [str(SPAWN_SCRIPT), str(tmp_path), "false", "", "test prompt"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN),
    )

    # Gate on BOTH the leader pid AND the readiness marker (post-trap proof).
    # The pid is published immediately; the readiness file appears only after
    # the DELAY + $inner trap-install sequence completes.
    _wait_for_path(pidfile, timeout=5.0)
    _wait_for_path(readyfile, timeout=5.0 + DELAY + 5.0)  # generous budget

    leader_pid = int(pidfile.read_text().strip())
    # SIGHUP arrives after the HUP trap is armed — must yield exit 129.
    os.killpg(leader_pid, signal.SIGHUP)

    try:
        rc = proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail(
            f"[{terminal_name}] spawn-claude.sh hung after window-close with "
            f"{DELAY}s trap-install delay — readiness gate not effective"
        )
    else:
        assert rc == 129, (
            f"[{terminal_name}] Expected exit 129 (window closed while alive "
            f"with {DELAY}s trap-install delay), got {rc}\n"
            f"stderr: {proc.stderr.read().decode()}"  # type: ignore[union-attr]
        )


# ===========================================================================
# task-2285 step-11: session-registry record lifecycle
# ===========================================================================
# Real spawn-claude.sh run, CLAUDE_FLEET_ROOT isolated to a tmp dir (via
# _base_env). The registry write must be purely additive: the pre-existing
# exit-code contract (result.returncode == the fake claude's own code) holds
# unchanged alongside the new launching -> exited record.json lifecycle.


def _find_one_record(fleet_root: pathlib.Path) -> pathlib.Path:
    """Return the single record.json under *fleet_root*/sessions/, or fail loudly."""
    records = list((fleet_root / "sessions").glob("*/record.json"))
    assert len(records) == 1, f"expected exactly one record.json, found {records}"
    return records[0]


@pytest.mark.parametrize("exit_code", [0, 3])
def test_spawn_writes_session_record_lifecycle(
    tmp_path: pathlib.Path, exit_code: int,
) -> None:
    """A real spawn writes launching -> exited with the session's own exit code."""
    bin_dir = _make_bin_dir(tmp_path)
    _write_fake_claude(bin_dir, exit_code=exit_code)
    _write_foreground_terminal(bin_dir, "xterm")
    env = _base_env(bin_dir, "xterm")

    result = _run_spawn(env, tmp_path)

    assert result.returncode == exit_code, (
        f"registry wiring must be additive to the exit-code contract: "
        f"expected {exit_code}, got {result.returncode}\n"
        f"stderr: {result.stderr.decode()}"
    )

    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    record_path = _find_one_record(fleet_root)
    record = session_registry.SessionRecord.from_json(record_path.read_text())
    assert record.status == session_registry.Status.EXITED
    assert record.exit_code == exit_code


# ===========================================================================
# task-2285 step-13: session-registry fail-soft on forced registry failure
# ===========================================================================
# CLAUDE_FLEET_ROOT is pointed at a subpath UNDER a pre-created regular file,
# so any mkdir the registry attempts raises NotADirectoryError deterministically
# (not reliant on filesystem permission semantics, which vary across CI users/
# containers). This pins the hard requirement (design decision 3): a registry
# fault must be loud (caller-visible stderr) but must NEVER change
# spawn-claude.sh's own exit code, and must leave no record dir behind.


@pytest.mark.parametrize("exit_code", [0, 3])
def test_spawn_fail_soft_on_unwritable_fleet_root(
    tmp_path: pathlib.Path, exit_code: int,
) -> None:
    """A forced registry-write failure must not affect the exit-code contract."""
    bin_dir = _make_bin_dir(tmp_path)
    _write_fake_claude(bin_dir, exit_code=exit_code)
    _write_foreground_terminal(bin_dir, "xterm")
    env = _base_env(bin_dir, "xterm")

    # Shadow _base_env's CLAUDE_FLEET_ROOT with one that can never be created:
    # a path nested UNDER a pre-existing regular file.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a regular file, not a directory\n")
    env["CLAUDE_FLEET_ROOT"] = str(blocker / "fleet")

    result = _run_spawn(env, tmp_path)

    assert result.returncode == exit_code, (
        f"a registry fault must never change the exit-code contract: "
        f"expected {exit_code}, got {result.returncode}\n"
        f"stderr: {result.stderr.decode()}"
    )

    stderr = result.stderr.decode()
    assert "session_registry" in stderr and "failed" in stderr, (
        f"expected a loud registry-fault line on the spawn's stderr, got:\n{stderr}"
    )

    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    assert not fleet_root.exists(), (
        f"no record dir should have been created under an unwritable fleet root, "
        f"but {fleet_root} exists"
    )


# ===========================================================================
# task-2285 step-15: G5 two-way boundary test (write -> refresh -> reap)
# ===========================================================================
# The shared session-registry contract (PRD G5): a record written by one
# process (spawn-claude.sh, a real bash subprocess) must be findable and
# refreshable under the SAME slug key by a wholly separate write (the future
# T6 SessionStart hook, simulated here via the `refresh` CLI), and the result
# must still reap correctly afterward. This is the seam every downstream
# Attention Rail task (T4/T5/T6/T7) builds on.

# Matches the guaranteed-dead-pid convention established in
# orchestrator/tests/test_session_registry.py (_DEAD_PID = 2**31 - 1).
_DEAD_PID = 2**31 - 1


def test_registry_write_refresh_reap_two_way_boundary(tmp_path: pathlib.Path) -> None:
    """write (real spawn) -> refresh (simulated hook, same key) -> reap."""
    # --- Producer: a REAL spawn-claude.sh run writes launching -> exited --
    bin_dir = _make_bin_dir(tmp_path)
    _write_fake_claude(bin_dir, exit_code=0)
    _write_foreground_terminal(bin_dir, "xterm")
    env = _base_env(bin_dir, "xterm")

    result = _run_spawn(env, tmp_path)
    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    record_path = _find_one_record(fleet_root)
    slug_dir = record_path.parent
    written = session_registry.SessionRecord.from_json(record_path.read_text())
    assert written.status == session_registry.Status.EXITED

    # Backdate the mtime so the refresh's heartbeat bump is unambiguous even
    # on filesystems with coarse mtime resolution.
    backdated_ts = record_path.stat().st_mtime - 100
    os.utime(record_path, (backdated_ts, backdated_ts))

    # --- Consumer: a simulated hook write refreshes under the SAME key ----
    # The exact CLI invocation the T6 SessionStart hook will make.
    rc = session_registry.main(
        ["refresh", "--record", str(slug_dir), "--status", "running"]
    )
    assert rc == 0

    refreshed_paths = list((fleet_root / "sessions").glob("*/record.json"))
    assert refreshed_paths == [record_path], (
        f"refresh must update the SAME record.json path (same key), "
        f"found: {refreshed_paths}"
    )
    refreshed = session_registry.SessionRecord.from_json(record_path.read_text())
    assert refreshed.session_slug == written.session_slug
    assert refreshed.status == session_registry.Status.RUNNING
    assert record_path.stat().st_mtime > backdated_ts, (
        "refresh must bump the record's mtime heartbeat"
    )

    # --- Upsert-on-absent: a hand-launched session with no prior write ----
    # (the T6 hand-launched-capture path -- no spawn-claude.sh write exists
    # for this slug at all).
    absent_slug = "unblock-df-999-424242"
    assert not session_registry.record_path_for_slug(absent_slug, root=fleet_root).is_file()

    upserted = session_registry.refresh_record(
        absent_slug, root=fleet_root, status=session_registry.Status.AWAITING_INPUT,
    )
    assert upserted.session_slug == absent_slug
    assert upserted.status == session_registry.Status.AWAITING_INPUT
    assert upserted.schema_version == session_registry.SCHEMA_VERSION
    assert upserted.start_ts, "an upserted record must still get a populated start_ts"
    upserted_path = session_registry.record_path_for_slug(absent_slug, root=fleet_root)
    assert upserted_path.is_file()
    assert session_registry.read_record(absent_slug, root=fleet_root) == upserted

    # --- The hook-refreshed record still reaps correctly -------------------
    # Force a guaranteed-dead launcher_pid (the real spawn's own $$ has
    # already exited by now, but relying on incidental PID death would be
    # racy under PID reuse) and age it past the non-terminal heartbeat TTL,
    # so it reaps via the stale_pid rule now that its status is RUNNING
    # (non-terminal) after the hook's refresh.
    dead_record = session_registry.SessionRecord.from_json(record_path.read_text())
    dead_record.launcher_pid = _DEAD_PID
    record_path.write_text(dead_record.to_json())
    now = datetime.now(UTC)
    stale_ts = (
        now - session_registry.NON_TERMINAL_HEARTBEAT_TTL - timedelta(hours=1)
    ).timestamp()
    os.utime(record_path, (stale_ts, stale_ts))

    reaped = session_registry.reap_stale_records(root=fleet_root, now=now)
    reaped_by_slug = {r.session_slug: r.reason for r in reaped}
    assert reaped_by_slug.get(dead_record.session_slug) == "stale_pid"
    assert not slug_dir.exists()
    assert absent_slug not in reaped_by_slug, (
        "the freshly-upserted record must not be reaped -- its heartbeat is new"
    )


# ===========================================================================
# task-2286 (Attention Rail T4): failed-to-start detection
# ===========================================================================
# The 2026-07-06 incident: a detaching launcher reports success (exit 0) but
# never actually starts claude -- no sentinel, no transcript, ever.
# resolve_detached's launch_rc==0 branch then calls the UNBOUNDED
# await_sentinel and hangs forever. A backgrounded started-watchdog must
# detect this within a bounded grace (SPAWN_STARTED_GRACE_SECS) and surface a
# distinct exit code + a loud stderr line + a failed-to-start session-registry
# record, instead of hanging.


def test_failed_to_start_detected_on_detached_exit0(tmp_path: pathlib.Path) -> None:
    """A detaching launcher that exits 0 WITHOUT starting claude must be
    flagged failed-to-start within SPAWN_STARTED_GRACE_SECS, not hang forever.

    This is the exact incident shape: the launcher (routed to the `*)`
    dispatch branch as "custom-term") exits 0 immediately without ever
    exec'ing the `bash -c "$inner"` payload -- so no claude process ever
    runs, no sentinel is ever written, and no transcript ever appears under
    $CLAUDE_PROJECTS_DIR.

    RED today: resolve_detached's launch_rc==0 (no sentinel) branch calls the
    unbounded await_sentinel, which loops forever since the sentinel is never
    written -> proc.wait(timeout=20) raises TimeoutExpired -> pytest.fail.
    """
    bin_dir = _make_bin_dir(tmp_path)

    # Fake DETACHING terminal that never runs the payload at all: no claude,
    # no sentinel, no transcript. Adapts the exit-0-without-payload idiom of
    # test_genuine_launcher_failure_yields_127 (which uses exit 1 for a
    # genuine launcher crash) to exit 0 -- the incident is a launcher that
    # reports SUCCESS while silently never starting claude.
    term = bin_dir / "custom-term"
    term.write_text("#!/usr/bin/env bash\nexit 0\n")
    term.chmod(0o755)

    env = _base_env(bin_dir, "custom-term")
    env["SPAWN_STARTED_GRACE_SECS"] = "2"

    proc = subprocess.Popen(
        [str(SPAWN_SCRIPT), str(tmp_path), "false", "", "test prompt"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    # Must-not-hang guard, not a latency SLA -- mirrors
    # test_window_close_yields_129_not_hang's Popen+wait(timeout) pattern.
    # Pre-impl this hangs forever (unbounded await_sentinel), so a bounded
    # wait cleanly separates pass/fail from an infinite hang.
    try:
        rc = proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail(
            "spawn-claude.sh hung after a detached launcher exited 0 without "
            "starting claude (no started-watchdog / unbounded await_sentinel "
            "not broken)"
        )
    else:
        stderr = proc.stderr.read().decode()  # type: ignore[union-attr]
        assert rc == 144, (
            f"Expected exit 144 (EXIT_FAILED_TO_START), got {rc}\nstderr: {stderr}"
        )
        assert "failed-to-start" in stderr.lower(), (
            f"expected a loud failed-to-start line on stderr, got:\n{stderr}"
        )

        fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
        record_path = _find_one_record(fleet_root)
        record = session_registry.SessionRecord.from_json(record_path.read_text())
        assert record.status == session_registry.Status.FAILED_TO_START, (
            f"expected registry status failed-to-start, got {record.status}"
        )


def test_transcript_appearance_suppresses_flag(tmp_path: pathlib.Path) -> None:
    """A fresh transcript file must suppress the failed-to-start flag.

    Proves the transcript detector is load-bearing. Uses a DETACHING launcher
    (custom-term, routing through resolve_detached's launch_rc==0 branch --
    the incident path) whose fake claude writes a transcript file under
    $CLAUDE_PROJECTS_DIR/<enc>/ the moment it starts, then sleeps well past
    the shrunk started-grace before exiting and letting $inner write the
    sentinel. <enc> mirrors session_registry.transcript_path_for_cwd's
    encoding: cwd with every '/' and '.' replaced by '-'.

    Grace/sleep margins are deliberately generous, not tight: the watchdog's
    deadline is computed from `date +%s` (whole-second resolution), so a
    nominal N-second grace can grant anywhere from just-over-0 to just-under
    (N+1) seconds of real time depending on where in the current second the
    watchdog starts. A grace=1s/sleep=2s pairing is flaky under load for
    exactly this reason (observed empirically); grace=3s/sleep=8s leaves
    enough headroom to absorb that +/-1s truncation noise on a busy host.

    RED today: step-2's watchdog only knows the exit sentinel, so with grace
    shorter than the sentinel delay it flags failed-to-start well before the
    sentinel appears -> rc 144, not 0. GREEN after step-4 wires the
    transcript probe.
    """
    bin_dir = _make_bin_dir(tmp_path)

    enc = str(tmp_path).replace("/", "-").replace(".", "-")

    # Fake claude: write the transcript file immediately (mirroring a real
    # Claude Code session creating ~/.claude/projects/<enc>/*.jsonl the moment
    # it starts), then outlast the shrunk started-grace before exiting -- so
    # only the transcript probe (not the sentinel) can suppress the flag.
    claude = bin_dir / "claude"
    claude.write_text(
        "#!/usr/bin/env bash\n"
        f'mkdir -p "$CLAUDE_PROJECTS_DIR/{enc}"\n'
        f'touch "$CLAUDE_PROJECTS_DIR/{enc}/session.jsonl"\n'
        "sleep 8\n"
        "exit 0\n"
    )
    claude.chmod(0o755)

    pidfile = tmp_path / "leader.pid"
    _write_detaching_terminal(bin_dir, "custom-term", pidfile)

    env = _base_env(bin_dir, "custom-term")
    env["SPAWN_STARTED_GRACE_SECS"] = "3"

    result = _run_spawn(env, tmp_path, timeout=20)

    stderr = result.stderr.decode()
    assert result.returncode == 0, (
        f"Expected exit 0 (transcript evidence must suppress the flag), "
        f"got {result.returncode}\nstderr: {stderr}"
    )
    assert result.returncode != 144, (
        f"transcript evidence must suppress EXIT_FAILED_TO_START, got 144\n"
        f"stderr: {stderr}"
    )
    assert "failed-to-start" not in stderr.lower(), (
        f"transcript evidence must suppress the loud failed-to-start line, got:\n{stderr}"
    )

    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    record_path = _find_one_record(fleet_root)
    record = session_registry.SessionRecord.from_json(record_path.read_text())
    assert record.status == session_registry.Status.EXITED, (
        f"expected registry status exited, got {record.status}"
    )


def test_foreground_claude_descendant_suppresses_flag_without_transcript(
    tmp_path: pathlib.Path,
) -> None:
    """A live foreground `claude` descendant with NO transcript must suppress
    the flag -- exercises _claude_descendant_alive as the LOAD-BEARING
    evidence (the only positive signal available on this path).

    Uses a FOREGROUND launcher (xterm) whose fake claude writes no transcript
    at all under $CLAUDE_PROJECTS_DIR, but stays alive (sleeping) well past
    the shrunk started-grace before exiting. Since xterm's fake terminal
    `exec`s into the payload bash (see _FOREGROUND_TERM_SCRIPT), claude runs
    as a direct descendant of spawn-claude.sh's own $$ -- unlike a detached
    launcher (setsid + background job, reparented once the launcher process
    exits), where this probe is correctly always empty.

    Grace/sleep margins mirror test_transcript_appearance_suppresses_flag's
    documented rationale: the watchdog's deadline is computed from whole-second
    `date +%s`, so a nominal N-second grace can grant anywhere from
    just-over-0 to just-under (N+1) seconds of real time. grace=2s/sleep=6s
    leaves ample headroom to absorb that truncation noise on a busy host.
    """
    bin_dir = _make_bin_dir(tmp_path)
    _write_foreground_terminal(bin_dir, "xterm")

    # Fake claude: writes NO transcript anywhere, just outlasts the shrunk
    # started-grace before exiting -- so only _claude_descendant_alive (not
    # the transcript probe) can suppress the flag.
    claude = bin_dir / "claude"
    claude.write_text("#!/usr/bin/env bash\nsleep 6\nexit 0\n")
    claude.chmod(0o755)

    env = _base_env(bin_dir, "xterm")
    env["SPAWN_STARTED_GRACE_SECS"] = "2"

    result = _run_spawn(env, tmp_path, timeout=20)

    stderr = result.stderr.decode()
    assert result.returncode == 0, (
        f"Expected exit 0 (live claude descendant must suppress the flag), "
        f"got {result.returncode}\nstderr: {stderr}"
    )
    assert result.returncode != 144, (
        f"a live claude descendant (no transcript yet) must suppress "
        f"EXIT_FAILED_TO_START, got 144\nstderr: {stderr}"
    )
    assert "failed-to-start" not in stderr.lower(), (
        f"live claude descendant must suppress the loud failed-to-start "
        f"line, got:\n{stderr}"
    )

    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    record_path = _find_one_record(fleet_root)
    record = session_registry.SessionRecord.from_json(record_path.read_text())
    assert record.status == session_registry.Status.EXITED, (
        f"expected registry status exited, got {record.status}"
    )


def test_foreground_launcher_failure_prefers_127_over_started_grace_race(
    tmp_path: pathlib.Path,
) -> None:
    """A foreground genuine launcher failure must yield 127 even when the
    caller has set SPAWN_STARTED_GRACE_SECS well below SPAWN_LAUNCH_GRACE_SECS.

    Regression guard for the review finding on resolve_foreground: the
    started-watchdog runs concurrently with _wait_sentinel_grace, and with a
    started-grace shorter than the launch-grace it reliably flags
    failed-to-start (writes fts_marker) well before _wait_sentinel_grace's
    own deadline elapses -- both timers are driven by the exact same root
    cause (the payload never ran), so this is deterministic, not a coin
    flip. resolve_foreground must still report the pre-existing, more
    specific 127 launcher-failure verdict in that case, not let a
    same-cause 144 win a race it happens to reach first.

    Uses a large gap (1s started-grace vs 5s launch-grace) so the watchdog's
    worst-case flag time (just under 2s, per the whole-second `date +%s`
    truncation noise documented elsewhere in this file) lands comfortably
    before _wait_sentinel_grace's ~4-5s deadline -- eliminating scheduling
    jitter as a source of flakiness.
    """
    bin_dir = _make_bin_dir(tmp_path)
    _write_fake_claude(bin_dir, exit_code=0)

    # Fake FOREGROUND terminal that exits immediately with rc=1, never running
    # the payload (so no sentinel is ever written) -- the same
    # genuine-launcher-failure shape as test_genuine_launcher_failure_yields_127,
    # now deliberately racing a misconfigured started-grace.
    fail_term = bin_dir / "xterm"
    fail_term.write_text("#!/usr/bin/env bash\nexit 1\n")
    fail_term.chmod(0o755)

    env = _base_env(bin_dir, "xterm")
    env["SPAWN_LAUNCH_GRACE_SECS"] = "5"
    env["SPAWN_STARTED_GRACE_SECS"] = "1"

    result = _run_spawn(env, tmp_path, timeout=20)
    assert result.returncode == 127, (
        f"Foreground launcher failure must yield 127 even when "
        f"SPAWN_STARTED_GRACE_SECS <= SPAWN_LAUNCH_GRACE_SECS, "
        f"got {result.returncode}\nstderr: {result.stderr.decode()}"
    )


def test_normal_spawn_exit0_not_flagged(tmp_path: pathlib.Path) -> None:
    """A normal, fast-exiting spawn must never be flagged failed-to-start.

    Regression guard (task's "a normal spawn is NOT flagged" requirement):
    locks that the started-watchdog's sentinel short-circuit means an
    ordinary fast session is never flagged, even while the started-grace
    watchdog is running concurrently in the background. Already green after
    step-2 (the sentinel check alone satisfies it) -- this pins the contract
    before step-4 adds more evidence probes.
    """
    bin_dir = _make_bin_dir(tmp_path)
    _write_fake_claude(bin_dir, exit_code=0)
    _write_foreground_terminal(bin_dir, "xterm")
    env = _base_env(bin_dir, "xterm")
    env["SPAWN_STARTED_GRACE_SECS"] = "2"

    result = _run_spawn(env, tmp_path)

    stderr = result.stderr.decode()
    assert result.returncode == 0, (
        f"Expected exit 0 (normal spawn), got {result.returncode}\nstderr: {stderr}"
    )
    assert "failed-to-start" not in stderr.lower(), (
        f"a normal spawn must never be flagged failed-to-start, got:\n{stderr}"
    )

    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    record_path = _find_one_record(fleet_root)
    record = session_registry.SessionRecord.from_json(record_path.read_text())
    assert record.status == session_registry.Status.EXITED, (
        f"expected registry status exited, got {record.status}"
    )


# ===========================================================================
# task-2287 (Attention Rail T5): spawn result-handback protocol
# ===========================================================================
# Exit codes are demoted to liveness-only; the semantic outcome channel is an
# explicit result.md written by the spawned session into its own session-
# registry record dir. spawn-claude.sh exports CLAUDE_SPAWN_RESULT_FILE
# (derived from SESSION_RECORD_DIR, byte-identical to the record's own
# result_file) and appends a prompt trailer pointing the session at it.


def test_spawn_exports_result_file_and_session_writes_it(
    tmp_path: pathlib.Path,
) -> None:
    """The spawned session must be able to write its outcome to
    $CLAUDE_SPAWN_RESULT_FILE, landing exactly at the session record's own
    result_file path.

    RED today: CLAUDE_SPAWN_RESULT_FILE is not exported into $inner, so the
    fake claude's write falls through to /dev/null and no file lands at the
    record's result_file path.
    """
    bin_dir = _make_bin_dir(tmp_path)
    _write_fake_claude_writing_result(bin_dir)
    _write_foreground_terminal(bin_dir, "xterm")
    env = _base_env(bin_dir, "xterm")

    result = _run_spawn(env, tmp_path)
    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    record_path = _find_one_record(fleet_root)
    record = session_registry.SessionRecord.from_json(record_path.read_text())

    expected_result_file = str(record_path.parent / "result.md")
    assert record.result_file == expected_result_file, (
        f"expected record.result_file == {expected_result_file}, "
        f"got {record.result_file}"
    )
    assert record.result_file is not None  # narrows str | None for the type checker

    result_file = pathlib.Path(record.result_file)
    assert result_file.is_file(), (
        f"expected a result.md written to {result_file} by the session"
    )
    assert "outcome: done" in result_file.read_text()


def test_spawn_appends_result_handback_trailer_to_prompt(
    tmp_path: pathlib.Path,
) -> None:
    """spawn-claude.sh must append a result-handback trailer to the prompt,
    pointing the spawned session at its own session record's result_file.

    Uses skip_perms="false" (the default baked into _run_spawn) so $flags is
    empty and $1 is exactly the prompt string spawn-claude.sh assembled --
    letting this test inspect the literal string claude received.

    RED today: no trailer is appended, so the captured prompt equals the
    bare original ("test prompt") with no result_file path anywhere in it.
    """
    bin_dir = _make_bin_dir(tmp_path)
    capture_file = tmp_path / "captured_prompt.txt"
    _write_fake_claude_capturing_prompt(bin_dir, capture_file)
    _write_foreground_terminal(bin_dir, "xterm")
    env = _base_env(bin_dir, "xterm")

    result = _run_spawn(env, tmp_path)
    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    record_path = _find_one_record(fleet_root)
    record = session_registry.SessionRecord.from_json(record_path.read_text())
    assert record.result_file, "expected a populated result_file on the record"

    captured_prompt = capture_file.read_text()
    assert "test prompt" in captured_prompt, (
        f"expected the original prompt to survive in the captured prompt, "
        f"got:\n{captured_prompt}"
    )
    assert record.result_file in captured_prompt, (
        f"expected a result-handback trailer referencing "
        f"{record.result_file!r} to be appended to the prompt, "
        f"got:\n{captured_prompt}"
    )


def test_spawn_fail_soft_skips_result_handback_when_registry_faults(
    tmp_path: pathlib.Path,
) -> None:
    """A forced registry-write failure must cleanly skip the ENTIRE
    result-handback protocol -- no env export, no prompt trailer, no bogus
    result.md anywhere -- while leaving the pre-existing exit-code contract
    and the loud registry-fault stderr line untouched.

    Green-on-arrival guard (matches test_spawn_fail_soft_on_unwritable_fleet_root's
    precedent): steps 2/4/6 already gate the result_file, the env export, and
    the prompt trailer on a non-empty CLAUDE_SPAWN_RESULT_FILE, and
    SESSION_RECORD_DIR is empty whenever the registry `launching` write
    faults, so this should already pass without further implementation
    changes. Locks the fail-soft contract before any future change to the
    result-handback wiring.

    Also captures the literal prompt claude received (via
    _write_fake_claude_capturing_prompt_and_writing_result) and asserts it
    is byte-identical to the bare original -- directly locking the "no
    prompt trailer" half of the contract instead of leaving it implied by
    the gating logic alone.
    """
    bin_dir = _make_bin_dir(tmp_path)
    capture_file = tmp_path / "captured_prompt.txt"
    _write_fake_claude_capturing_prompt_and_writing_result(bin_dir, capture_file)
    _write_foreground_terminal(bin_dir, "xterm")
    env = _base_env(bin_dir, "xterm")

    # Shadow _base_env's CLAUDE_FLEET_ROOT with one that can never be created:
    # a path nested UNDER a pre-existing regular file (same trick as
    # test_spawn_fail_soft_on_unwritable_fleet_root).
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a regular file, not a directory\n")
    env["CLAUDE_FLEET_ROOT"] = str(blocker / "fleet")

    result = _run_spawn(env, tmp_path)

    assert result.returncode == 0, (
        f"a result-file allocation failure must never change the exit-code "
        f"contract: expected 0 (the session's own code), got "
        f"{result.returncode}\nstderr: {result.stderr.decode()}"
    )

    stderr = result.stderr.decode()
    assert "session_registry" in stderr and "failed" in stderr, (
        f"expected a loud registry-fault line on the spawn's stderr, got:\n{stderr}"
    )

    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    assert not fleet_root.exists(), (
        f"no record dir should have been created under an unwritable fleet "
        f"root, but {fleet_root} exists"
    )
    assert not list(tmp_path.rglob("result.md")), (
        "no result.md should be created anywhere when the registry write faults"
    )

    captured_prompt = capture_file.read_text()
    assert captured_prompt == "test prompt", (
        "expected NO result-handback trailer to be appended when the "
        "registry write faults -- the captured prompt must equal the bare "
        f"original with nothing appended, got:\n{captured_prompt!r}"
    )
    assert "result.md" not in captured_prompt, (
        f"expected no 'result.md' substring anywhere in the prompt claude "
        f"received, got:\n{captured_prompt!r}"
    )


# ===========================================================================
# task-2291 step-7: Fleet Cockpit C1 spawn env exports (child/parent identity)
# ===========================================================================
# CLAUDE_SPAWN_SESSION_ID/CLAUDE_SPAWN_PARENT_ID let the spawned child (and
# its own descendants) discover its own registry identity and its direct
# spawner's, without re-deriving either from SESSION_RECORD_DIR. Default
# 'child' spawn_mode: the child's parent-of-record is the DIRECT spawner --
# this spawn-claude.sh invocation's own inherited CLAUDE_SPAWN_SESSION_ID.
# Both exports mirror result_export's no-op-when-empty idiom, so a registry
# fault (SESSION_RECORD_DIR empty) cleanly exports neither.


def _write_fake_claude_capturing_env(
    bin_dir: pathlib.Path, capture_file: pathlib.Path
) -> None:
    """Write a fake ``claude`` that captures its own CLAUDE_SPAWN_SESSION_ID
    and CLAUDE_SPAWN_PARENT_ID env vars to *capture_file*, then exits 0.

    Modeled on ``_write_fake_claude_capturing_prompt`` -- lets a test inspect
    exactly what the two Fleet Cockpit C1 identity exports resolved to for
    the spawned child process.
    """
    p = bin_dir / "claude"
    p.write_text(
        "#!/usr/bin/env bash\n"
        "{\n"
        '  echo "SESSION=${CLAUDE_SPAWN_SESSION_ID:-}"\n'
        '  echo "PARENT=${CLAUDE_SPAWN_PARENT_ID:-}"\n'
        f"}} > {capture_file!s}\n"
        "exit 0\n"
    )
    p.chmod(0o755)


def _parse_captured_env(capture_file: pathlib.Path) -> dict[str, str]:
    """Parse the ``KEY=value`` lines written by _write_fake_claude_capturing_env."""
    parsed: dict[str, str] = {}
    for line in capture_file.read_text().splitlines():
        key, _, value = line.partition("=")
        parsed[key] = value
    return parsed


def test_spawn_exports_session_and_parent_ids(tmp_path: pathlib.Path) -> None:
    """The child sees its OWN new session_slug as CLAUDE_SPAWN_SESSION_ID, and
    the spawner's own inherited CLAUDE_SPAWN_SESSION_ID as
    CLAUDE_SPAWN_PARENT_ID (default 'child' spawn_mode: parent-of-record is
    the direct spawner).
    """
    bin_dir = _make_bin_dir(tmp_path)
    capture_file = tmp_path / "captured_env.txt"
    _write_fake_claude_capturing_env(bin_dir, capture_file)
    _write_foreground_terminal(bin_dir, "xterm")
    env = _base_env(bin_dir, "xterm")
    # Simulate this spawn-claude.sh invocation itself being run BY an
    # already-spawned parent session -- the spawner's own inherited identity
    # token that this new child's CLAUDE_SPAWN_PARENT_ID must carry forward.
    env.pop("CLAUDE_SPAWN_PARENT_ID", None)
    env["CLAUDE_SPAWN_SESSION_ID"] = "root-df-1-1"

    result = _run_spawn(env, tmp_path)
    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    record_path = _find_one_record(fleet_root)
    record = session_registry.SessionRecord.from_json(record_path.read_text())

    captured = _parse_captured_env(capture_file)
    assert captured.get("SESSION") == record.session_slug, (
        f"expected the child's own new session_slug, got {captured!r}"
    )
    assert captured.get("PARENT") == "root-df-1-1", (
        f"expected the spawner's own inherited session id as the parent, got {captured!r}"
    )


def test_spawn_parent_id_empty_for_human_root(tmp_path: pathlib.Path) -> None:
    """A human-launched root has no CLAUDE_SPAWN_SESSION_ID of its own to
    inherit, so the child's CLAUDE_SPAWN_PARENT_ID must be empty -- while the
    child still gets its own freshly-minted CLAUDE_SPAWN_SESSION_ID.
    """
    bin_dir = _make_bin_dir(tmp_path)
    capture_file = tmp_path / "captured_env.txt"
    _write_fake_claude_capturing_env(bin_dir, capture_file)
    _write_foreground_terminal(bin_dir, "xterm")
    env = _base_env(bin_dir, "xterm")
    env.pop("CLAUDE_SPAWN_SESSION_ID", None)
    env.pop("CLAUDE_SPAWN_PARENT_ID", None)

    result = _run_spawn(env, tmp_path)
    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    record_path = _find_one_record(fleet_root)
    record = session_registry.SessionRecord.from_json(record_path.read_text())

    captured = _parse_captured_env(capture_file)
    assert captured.get("SESSION") == record.session_slug
    assert captured.get("PARENT") == "", (
        f"a human-launched root has no parent to report, got {captured!r}"
    )


def test_spawn_id_exports_fail_soft_when_registry_faults(tmp_path: pathlib.Path) -> None:
    """A forced registry-write failure (SESSION_RECORD_DIR empty) must yield
    clean no-op exports -- with no ambient CLAUDE_SPAWN_SESSION_ID/PARENT_ID
    to leak through, the child sees both empty -- while the pre-existing
    exit-code contract is untouched (mirrors
    test_spawn_fail_soft_on_unwritable_fleet_root).

    Green-on-arrival guard (matches
    test_spawn_fail_soft_skips_result_handback_when_registry_faults'
    precedent): step-8 gates both exports on a non-empty SESSION_RECORD_DIR,
    so with no ambient identity env to leak through this already passes
    pre-implementation too -- it locks the fail-soft contract rather than
    demonstrating a RED->GREEN transition.
    """
    bin_dir = _make_bin_dir(tmp_path)
    capture_file = tmp_path / "captured_env.txt"
    _write_fake_claude_capturing_env(bin_dir, capture_file)
    _write_foreground_terminal(bin_dir, "xterm")
    env = _base_env(bin_dir, "xterm")
    env.pop("CLAUDE_SPAWN_SESSION_ID", None)
    env.pop("CLAUDE_SPAWN_PARENT_ID", None)

    # Shadow _base_env's CLAUDE_FLEET_ROOT with one that can never be
    # created: a path nested UNDER a pre-existing regular file (same trick
    # as test_spawn_fail_soft_on_unwritable_fleet_root).
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a regular file, not a directory\n")
    env["CLAUDE_FLEET_ROOT"] = str(blocker / "fleet")

    result = _run_spawn(env, tmp_path)

    assert result.returncode == 0, (
        f"a registry fault must never change the exit-code contract: "
        f"got {result.returncode}\nstderr: {result.stderr.decode()}"
    )

    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    assert not fleet_root.exists(), (
        f"no record dir should have been created under an unwritable fleet "
        f"root, but {fleet_root} exists"
    )

    captured = _parse_captured_env(capture_file)
    assert captured.get("SESSION") == "", f"expected a clean no-op export, got {captured!r}"
    assert captured.get("PARENT") == "", f"expected a clean no-op export, got {captured!r}"


# ===========================================================================
# task-2297 (Fleet Cockpit C6): tmux lane for long-runner watcher sessions
# ===========================================================================
# CLAUDE_SPAWN_BACKEND=tmux opts into a crash-survivable, reattachable tmux
# window instead of any terminal emulator -- bypassing emulator discovery
# entirely so no display/emulator is needed at all. `tmux new-window`/
# `new-session` return immediately while the tmux server owns the payload --
# exactly the DETACHED semantics the konsole/custom-launcher branches already
# get via resolve_detached -- so the window (and its registry record)
# outlives spawn-claude.sh itself, which is what makes the lane reattachable.


def _write_fake_tmux(
    bin_dir: pathlib.Path, marker: pathlib.Path, target: str = "fleet-proj:0"
) -> None:
    """Write a fake ``tmux`` binary (Fleet Cockpit C6 tmux lane).

    ``has-session`` always reports absent (exit 1), forcing spawn-claude.sh's
    tmux-lane branch down the ``new-session`` path (never ``new-window``) --
    these tests don't care which of the two creates the window, only that
    ONE of them does. ``new-session``/``new-window`` touch *marker* (proves
    the tmux lane -- not an emulator -- ran), locate ``bash`` in argv
    (mirrors ``_DETACHING_TERM_TEMPLATE``), run the payload detached via
    ``setsid env --default-signal=HUP,TERM`` (the same signal-disposition
    reset a real terminal gives its child shell), print a synthetic
    ``"<session>:<window>"`` *target* (mirrors tmux's own ``-P -F`` output),
    and exit 0 -- so the command substitution ``$(tmux ...)`` in
    spawn-claude.sh returns almost immediately while the payload keeps
    running in the background, matching a real `tmux new-window` call.
    """
    script = textwrap.dedent(f"""\
        #!/usr/bin/env bash
        case "$1" in
          has-session)
            exit 1
            ;;
          new-session|new-window)
            touch {marker!s}
            while [[ $# -gt 0 ]]; do
              if [[ "$1" == "bash" ]]; then
                break
              fi
              shift
            done
            setsid env --default-signal=HUP,TERM "$@" >/dev/null 2>&1 &
            echo "{target}"
            exit 0
            ;;
          *)
            exit 1
            ;;
        esac
    """)
    p = bin_dir / "tmux"
    p.write_text(script)
    p.chmod(0o755)


def _write_marker_only_failing_terminal(
    bin_dir: pathlib.Path, name: str, marker: pathlib.Path
) -> None:
    """Write a fake foreground emulator that touches *marker* then exits 1
    WITHOUT running the payload (no sentinel is ever written).

    Used to prove the tmux lane bypasses emulator discovery entirely --
    CLAUDE_TERMINAL_CMD is set to *name* alongside CLAUDE_SPAWN_BACKEND=tmux,
    so if the tmux branch ever fell through to emulator discovery this
    binary would run and leave *marker* behind.
    """
    p = bin_dir / name
    p.write_text(f"#!/usr/bin/env bash\ntouch {marker!s}\nexit 1\n")
    p.chmod(0o755)


def _write_fake_claude_writing_result_with_exit_code(
    bin_dir: pathlib.Path, exit_code: int
) -> None:
    """Like ``_write_fake_claude_writing_result``, but exits *exit_code*
    instead of the hardcoded 0 -- lets a test lock the tmux lane's exit-code
    contract (the session's own code) and its result-handback delivery
    together in one spawn.
    """
    p = bin_dir / "claude"
    p.write_text(
        "#!/usr/bin/env bash\n"
        "cat > \"${CLAUDE_SPAWN_RESULT_FILE:-/dev/null}\" <<'EOF'\n"
        "---\n"
        "outcome: done\n"
        "changed: none\n"
        "action_needed: none\n"
        "---\n"
        "Test prose.\n"
        "EOF\n"
        f"exit {exit_code}\n"
    )
    p.chmod(0o755)


@pytest.mark.parametrize("exit_code", [0, 3])
def test_tmux_backend_routes_and_runs_session(
    tmp_path: pathlib.Path, exit_code: int,
) -> None:
    """CLAUDE_SPAWN_BACKEND=tmux must route to the tmux lane -- bypassing
    the configured emulator entirely -- and still deliver the full
    exit-code + session-record + result-handback contract.

    RED today: without a tmux branch, CLAUDE_SPAWN_BACKEND is silently
    ignored and ordinary emulator discovery picks CLAUDE_TERMINAL_CMD=xterm.
    The fake xterm below touches emulator_used and exits 1 WITHOUT running
    the payload (no sentinel is ever written), so resolve_foreground's
    genuine-launcher-failure path fires -> exit 127, not exit_code; no
    EXITED record is ever written. GREEN after step-4.
    """
    bin_dir = _make_bin_dir(tmp_path)
    _write_fake_claude_writing_result_with_exit_code(bin_dir, exit_code)

    tmux_marker = tmp_path / "tmux_used"
    emulator_marker = tmp_path / "emulator_used"
    _write_fake_tmux(bin_dir, tmux_marker)
    _write_marker_only_failing_terminal(bin_dir, "xterm", emulator_marker)

    env = _base_env(bin_dir, "xterm")
    env["CLAUDE_SPAWN_BACKEND"] = "tmux"
    env["CLAUDE_SPAWN_PROJECT"] = "proj"

    result = _run_spawn(env, tmp_path)

    assert result.returncode == exit_code, (
        f"expected the session's own exit code {exit_code} via the tmux "
        f"lane, got {result.returncode}\nstderr: {result.stderr.decode()}"
    )
    assert not emulator_marker.exists(), (
        "CLAUDE_SPAWN_BACKEND=tmux must bypass emulator discovery entirely "
        "-- the configured CLAUDE_TERMINAL_CMD=xterm must never run"
    )
    assert tmux_marker.exists(), (
        "expected the tmux lane (new-session/new-window) to have run"
    )

    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    record_path = _find_one_record(fleet_root)
    assert record_path.exists(), "the session record must persist after exit"
    record = session_registry.SessionRecord.from_json(record_path.read_text())
    assert record.status == session_registry.Status.EXITED, (
        f"expected registry status exited, got {record.status}"
    )
    assert record.exit_code == exit_code

    assert record.result_file is not None
    result_file = pathlib.Path(record.result_file)
    assert result_file.is_file(), (
        f"expected a result.md written to {result_file} by the session"
    )
    assert "outcome:" in result_file.read_text()


def test_tmux_backend_stamps_display_record(tmp_path: pathlib.Path) -> None:
    """A successful tmux-lane spawn must stamp the record's display: kind
    'tmux', tmux_target matching the tmux `-P -F` output, and wm_title
    matching the title argument passed to spawn-claude.sh.

    RED today: step-4 runs the tmux lane but never calls the session-registry
    `set-display` verb, so record.display stays None -- the record `launching`
    creates never populates it, and nothing else writes it. GREEN after
    step-6 wires the post-window-creation stamp (built on step-2's
    `set-display` CLI verb, already landed).
    """
    bin_dir = _make_bin_dir(tmp_path)
    _write_fake_claude_writing_result_with_exit_code(bin_dir, 0)

    tmux_marker = tmp_path / "tmux_used"
    emulator_marker = tmp_path / "emulator_used"
    target = "fleet-proj:0"
    _write_fake_tmux(bin_dir, tmux_marker, target=target)
    _write_marker_only_failing_terminal(bin_dir, "xterm", emulator_marker)

    env = _base_env(bin_dir, "xterm")
    env["CLAUDE_SPAWN_BACKEND"] = "tmux"
    env["CLAUDE_SPAWN_PROJECT"] = "proj"

    # Use a distinctive, non-empty title (unlike _run_spawn's hardcoded "")
    # so the wm_title assertion below actually exercises the wiring instead
    # of trivially matching an empty default.
    title = "tmux-lane-display-test"
    result = subprocess.run(
        [str(SPAWN_SCRIPT), str(tmp_path), "false", title, "test prompt"],
        env=env,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    record_path = _find_one_record(fleet_root)
    record = session_registry.SessionRecord.from_json(record_path.read_text())

    assert record.display is not None, (
        "expected a stamped display on a successful tmux-lane spawn"
    )
    assert record.display.kind == "tmux"
    assert record.display.tmux_target == target, (
        f"expected the display's tmux_target to match the tmux -P -F "
        f"output {target!r}, got {record.display.tmux_target!r}"
    )
    assert record.display.wm_title == title, (
        f"expected the display's wm_title to match the title passed to "
        f"spawn-claude.sh ({title!r}), got {record.display.wm_title!r}"
    )


@pytest.mark.skipif(
    __import__("platform").system() == "Darwin",
    reason="exit-126 path requires non-Darwin host",
)
def test_tmux_backend_missing_tmux_yields_126(tmp_path: pathlib.Path) -> None:
    """CLAUDE_SPAWN_BACKEND=tmux with no `tmux` binary on PATH must exit 126
    with a loud stderr line mentioning tmux -- mirroring the no-emulator 126
    path (test_no_emulator_found_yields_126) rather than a bare
    command-not-found 127.

    RED today: step-4's tmux-lane case has no availability guard, so
    `tmux has-session ...`/`tmux new-session ...` are themselves
    command-not-found (bash exit 127) -> resolve_detached's launch_rc!=0
    branch -> exit 127, not 126. GREEN after step-8.
    """
    import shutil as _shutil

    bin_dir = _make_bin_dir(tmp_path)
    _write_fake_claude(bin_dir, exit_code=0)

    # Minimal system-bin with only the utilities the script needs -- NO
    # tmux, NO terminal emulator (mirrors test_no_emulator_found_yields_126's
    # sys_bin exactly, including the deliberate exclusion of python3, which
    # makes the session-registry `launching` write no-op so this test needs
    # no CLAUDE_FLEET_ROOT isolation).
    sys_bin = tmp_path / "sys_bin"
    sys_bin.mkdir()
    for util in ["bash", "mktemp", "sleep", "cat", "rm", "uname"]:
        src = _shutil.which(util)
        if src:
            (sys_bin / util).symlink_to(src)

    env = dict(os.environ)
    env["PATH"] = str(bin_dir) + ":" + str(sys_bin)
    env["CLAUDE_SPAWN_BACKEND"] = "tmux"
    env["SPAWN_LAUNCH_GRACE_SECS"] = "2"
    env.pop("CLAUDE_TERMINAL_CMD", None)
    env.pop("ESCALATION_TERMINAL_CMD", None)

    result = _run_spawn(env, tmp_path, timeout=10)
    assert result.returncode == 126, (
        f"missing tmux in tmux-backend mode must yield 126, got "
        f"{result.returncode}\nstderr: {result.stderr.decode()}"
    )
    stderr = result.stderr.decode()
    assert "tmux" in stderr.lower(), (
        f"expected a loud stderr line mentioning tmux, got:\n{stderr}"
    )


# ===========================================================================
# task-2298 step-1: Fleet Cockpit C7 sibling spawn mode (parent-of-record)
# ===========================================================================
# CLAUDE_SPAWN_MODE=sibling changes which identity the child inherits as its
# own CLAUDE_SPAWN_PARENT_ID: not the direct spawner (this spawn-claude.sh
# invocation's own CLAUDE_SPAWN_SESSION_ID -- the 'child' default exercised
# by the C1 tests above), but the spawner's OWN inherited
# CLAUDE_SPAWN_PARENT_ID -- the shared ancestor. All three tests below route
# through CLAUDE_SPAWN_BACKEND=tmux (Fleet Cockpit C6, see _write_fake_tmux
# above) rather than a foreground emulator, and poll for captured_env.txt via
# _wait_for_path rather than assuming it is present the instant _run_spawn
# returns: the eventual GREEN state (step-4, fire-and-forget) returns from
# spawn-claude.sh before the detached child has necessarily finished writing
# it, so these tests must not depend on the pre-step-4 blocking-wait timing
# they'd otherwise get for free.


def _run_sibling_capture_spawn(
    tmp_path: pathlib.Path,
    *,
    spawner_session_id: str | None,
    spawner_parent_id: str | None,
) -> tuple[subprocess.CompletedProcess[bytes], dict[str, str], pathlib.Path]:
    """Run a CLAUDE_SPAWN_MODE=sibling spawn behind the tmux backend and
    return ``(result, parsed captured env, fleet_root)``.

    Shared setup for the three sibling parent-of-record tests below.
    """
    bin_dir = _make_bin_dir(tmp_path)
    capture_file = tmp_path / "captured_env.txt"
    _write_fake_claude_capturing_env(bin_dir, capture_file)
    tmux_marker = tmp_path / "tmux_used"
    _write_fake_tmux(bin_dir, tmux_marker)

    env = _base_env(bin_dir, "xterm")
    env["CLAUDE_SPAWN_BACKEND"] = "tmux"
    env["CLAUDE_SPAWN_PROJECT"] = "proj"
    env["CLAUDE_SPAWN_MODE"] = "sibling"
    if spawner_session_id is None:
        env.pop("CLAUDE_SPAWN_SESSION_ID", None)
    else:
        env["CLAUDE_SPAWN_SESSION_ID"] = spawner_session_id
    if spawner_parent_id is None:
        env.pop("CLAUDE_SPAWN_PARENT_ID", None)
    else:
        env["CLAUDE_SPAWN_PARENT_ID"] = spawner_parent_id

    result = _run_spawn(env, tmp_path)
    _wait_for_path(capture_file, timeout=5.0)
    captured = _parse_captured_env(capture_file)
    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    return result, captured, fleet_root


def test_sibling_mode_parents_at_shared_ancestor(tmp_path: pathlib.Path) -> None:
    """Sibling mode: the child's CLAUDE_SPAWN_PARENT_ID must be the
    spawner's OWN inherited parent ("A", the shared ancestor) -- NOT the
    spawner's own session id ("P", the direct spawner). The child still
    gets its own freshly-minted CLAUDE_SPAWN_SESSION_ID either way.

    RED today: CLAUDE_SPAWN_MODE is ignored (default 'child' behavior), so
    parent_id_export always resolves to the spawner's own
    CLAUDE_SPAWN_SESSION_ID ("P") regardless of spawn_mode. GREEN after
    step-2.
    """
    result, captured, fleet_root = _run_sibling_capture_spawn(
        tmp_path, spawner_session_id="P", spawner_parent_id="A",
    )
    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    record_path = _find_one_record(fleet_root)
    record = session_registry.SessionRecord.from_json(record_path.read_text())

    assert captured.get("SESSION") == record.session_slug, (
        f"expected the child's own new session_slug, got {captured!r}"
    )
    assert captured.get("PARENT") == "A", (
        f"sibling mode must parent the child at the spawner's OWN parent "
        f"(shared ancestor 'A'), not the direct spawner 'P', got {captured!r}"
    )


def test_sibling_mode_null_parent_becomes_root(tmp_path: pathlib.Path) -> None:
    """Sibling mode with no CLAUDE_SPAWN_PARENT_ID of its own (the spawner
    is itself root, or was hand-launched) -- the child's
    CLAUDE_SPAWN_PARENT_ID must be empty (null -> root), not silently fall
    back to the spawner's own session id.

    RED today: same root cause as test_sibling_mode_parents_at_shared_ancestor
    -- CLAUDE_SPAWN_MODE is ignored, so parent_id_export uses the spawner's
    own CLAUDE_SPAWN_SESSION_ID ("P") instead of staying empty.
    """
    result, captured, _fleet_root = _run_sibling_capture_spawn(
        tmp_path, spawner_session_id="P", spawner_parent_id=None,
    )
    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    assert captured.get("PARENT") == "", (
        f"a spawner with no parent of its own must yield an empty (root) "
        f"parent for a sibling child, got {captured!r}"
    )


def test_sibling_parentage_two_way_into_hook_record(tmp_path: pathlib.Path) -> None:
    """B7 boundary (C7 spawn-side export <-> C1/C2 hook-side schema): the
    CLAUDE_SPAWN_PARENT_ID a sibling spawn actually exports must, when fed
    to the already-landed SessionStart hook
    (orchestrator.session_hooks.run_session_start), land as
    record.parent_session_id -- proving the two sides of the seam compose
    end-to-end, not just independently.

    RED today: the sibling spawn exports "P" (the direct spawner) instead of
    "A" (the shared ancestor), so the fed-through value fails both the
    equals-"A" and not-equals-"P" assertions below.
    """
    result, captured, fleet_root = _run_sibling_capture_spawn(
        tmp_path, spawner_session_id="P", spawner_parent_id="A",
    )
    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    hook_input = {"session_id": "child-hook-boundary", "cwd": str(tmp_path)}
    hook_env = {"CLAUDE_SPAWN_PARENT_ID": captured.get("PARENT", "")}
    record = session_hooks.run_session_start(hook_input, hook_env, root=fleet_root)

    assert record.parent_session_id == "A", (
        f"expected the sibling-exported parent id to land as "
        f"record.parent_session_id, got {record.parent_session_id!r}"
    )
    assert record.parent_session_id != "P", (
        "record.parent_session_id must not be the direct spawner's own "
        "session id in sibling mode"
    )


# ===========================================================================
# task-2298 step-3: Fleet Cockpit C7 sibling mode is fire-and-forget
# ===========================================================================
# The /prd author->decompose handoff needs to spawn its sibling and exit
# cleanly, not babysit it -- so sibling mode must return once the child is
# LAUNCHED, without blocking on the sentinel until it exits. Proven by
# marker presence (started) + absence (done) rather than a wall-clock
# tolerance: sleep_secs is chosen generously large relative to how long
# _run_spawn itself takes to return, not tuned to any specific latency
# budget.


def _write_fake_claude_slow_with_markers(
    bin_dir: pathlib.Path, started: pathlib.Path, done: pathlib.Path, sleep_secs: float = 5,
) -> None:
    """Write a fake ``claude`` that touches *started* immediately, sleeps
    *sleep_secs*, then touches *done* and exits 0.

    Used to prove fire-and-forget: the spawner returns after the child is
    launched (started exists) but WITHOUT waiting for it to finish (done
    does not exist yet) -- a marker-presence check, not a timing tolerance.
    """
    p = bin_dir / "claude"
    p.write_text(
        f"#!/usr/bin/env bash\n"
        f"touch {started!s}\n"
        f"sleep {sleep_secs}\n"
        f"touch {done!s}\n"
        f"exit 0\n"
    )
    p.chmod(0o755)


def test_sibling_mode_is_fire_and_forget(tmp_path: pathlib.Path) -> None:
    """Sibling mode must return from spawn-claude.sh once the child is
    LAUNCHED, without blocking on the sentinel until it exits.

    Premise-free: the done-marker absence is checked as a snapshot taken
    immediately after ``_run_spawn`` returns (before any further polling
    delay), and the started-marker's eventual presence is confirmed via
    ``_wait_for_path`` (robust to scheduling jitter, not a race against the
    assertion above it).

    RED today: sibling mode still falls through to
    resolve_detached->await_sentinel, so _run_spawn blocks until the child
    exits and touches done -- done.exists() is True by the time _run_spawn
    returns, failing the "must be absent" assertion below. GREEN after
    step-4.
    """
    bin_dir = _make_bin_dir(tmp_path)
    started = tmp_path / "started"
    done = tmp_path / "done"
    _write_fake_claude_slow_with_markers(bin_dir, started, done, sleep_secs=5)

    tmux_marker = tmp_path / "tmux_used"
    _write_fake_tmux(bin_dir, tmux_marker)

    env = _base_env(bin_dir, "xterm")
    env["CLAUDE_SPAWN_BACKEND"] = "tmux"
    env["CLAUDE_SPAWN_PROJECT"] = "proj"
    env["CLAUDE_SPAWN_MODE"] = "sibling"

    result = _run_spawn(env, tmp_path)

    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    # Snapshot immediately -- must reflect the state at the moment
    # spawn-claude.sh returned, not after the started-marker poll below.
    done_existed_at_return = done.exists()
    assert not done_existed_at_return, (
        "sibling mode must return WITHOUT waiting for the child to finish "
        "-- the done-marker must not exist yet at the moment "
        "spawn-claude.sh returns"
    )

    _wait_for_path(started, timeout=5.0)

    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    record_path = _find_one_record(fleet_root)
    record = session_registry.SessionRecord.from_json(record_path.read_text())
    assert record.status == session_registry.Status.RUNNING, (
        f"expected a best-effort refresh to RUNNING, got {record.status}"
    )
