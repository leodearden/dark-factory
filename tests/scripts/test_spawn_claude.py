"""Tests for skills/spawn/spawn-claude.sh exit-code contract.

Headless: no real display needed.  A fake ``claude`` binary on PATH provides
parametrisable exit codes, and fake terminal binaries route to specific dispatch
branches by name (the script dispatches on the first word of $CLAUDE_TERMINAL_CMD).
"""

from __future__ import annotations

import math
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

# noqa must sit on the STATEMENT's first line: E402 is reported at the start of
# the import, so the per-name noqas ruff's I001 fix left on lines below suppress
# nothing. The pyright ignores stay per-name, where each attribute is flagged.
from orchestrator import (  # noqa: E402
    session_hooks,  # pyright: ignore[reportAttributeAccessIssue]
    session_registry,  # pyright: ignore[reportAttributeAccessIssue]
)

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
    """Poll until *path* exists, raising ``AssertionError`` on timeout.

    Low-level primitive only -- direct callers should prefer
    _wait_for_path_scaled (below) for a load-adaptive budget instead of a
    fixed timeout; this function remains only as its poll implementation.
    """
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"Timed out after {timeout}s waiting for {path} to appear"
            )
        time.sleep(0.05)


# _READINESS_WAIT_CAP_SECS: measured, not guessed -- on this host (nproc 32,
# /proc/loadavg 100.32 => load-per-core 3.14), whole-test wall for every
# _wait_for_path-gated test was <= 3.15s (konsole 2.46s, custom-term 1.72s,
# sibling lanes 0.94-1.16s), and whole-test wall upper-bounds any single gate
# inside it -- so a 30s ceiling is ~12x the worst observed gate.
#
# Deliberately NOT raised to 60 the way _NOT_FLAGGED_GRACE_BASE_SECS (below)
# uses cap_secs=60: a started-grace is an upper bound the watchdog polls to
# and the happy path never pays (see the comment above
# _NOT_FLAGGED_GRACE_BASE_SECS), whereas a readiness-wait cap IS paid in full
# on the failure path, so it stays tight rather than inheriting that raise.
#
# Headroom against the real per-test ceiling: `pytest --collect-only` reports
# configfile: pyproject.toml (the repo root, which sets no `timeout`); the
# value that actually governs this file is the --timeout=300 passed by
# scripts/orchestrator.yaml:17. Worst case in the busiest rewired test
# (test_window_close_129_robust_to_delayed_trap_install, whose readyfile
# gate overrides cap_secs to 60 -- see _wait_for_path_scaled) is
# 30 (pidfile) + 60 + 1.0 DELAY (readyfile) + 30 (proc.wait) = 121s,
# comfortably inside 300s.
_READINESS_WAIT_CAP_SECS = 30


def _wait_for_path_scaled(
    path: pathlib.Path,
    base_secs: int,
    *,
    extra_secs: float = 0.0,
    cap_secs: int = _READINESS_WAIT_CAP_SECS,
) -> float:
    """Wait for *path* with a load-scaled budget, and return the budget used.

    A fixed _wait_for_path timeout races a host-load-dependent subprocess
    startup chain -- observed once as
    test_window_close_yields_129_not_hang[konsole] failing at
    _wait_for_path(pidfile, timeout=5.0) during task 3451's step-7
    full-suite verify, passing in isolation and on immediate rerun.

    Returning the budget makes the policy assertable on an already-existing
    path with zero sleeping -- the direct analogue of _set_started_grace
    returning the int it wrote into env (see below), and the reason no
    forbidden source-grepping meta-test is needed to pin the fix.

    Floored at base_secs: an idle host (load-per-core <= 1) returns
    base_secs unchanged, so every rewired call site stays byte-identical to
    its old fixed pin on an unloaded host.

    extra_secs exists for gates that sit behind a DELIBERATELY INJECTED,
    wall-clock-fixed sleep (today only
    test_window_close_129_robust_to_delayed_trap_install's DELAY = 1.0,
    injected by _STRESS_DETACHING_TERM_TEMPLATE before $inner runs). Such a
    sleep does not stretch with host load, so it is added UNSCALED and is
    NOT subject to cap_secs -- only the load-dependent startup chain around
    it is scaled.

    cap_secs overrides _READINESS_WAIT_CAP_SECS for a call site whose
    pre-existing budget already exceeded it -- mirroring
    _NOT_FLAGGED_GRACE_BASE_SECS's own cap raise (30 -> 60) below. Today
    only test_window_close_129_robust_to_delayed_trap_install's readyfile
    gate needs this: its old inline form summed two INDEPENDENTLY-capped
    _load_scaled_grace(5) halves (up to 2*30=60s under load), so collapsing
    it onto the default single 30s cap would nearly halve its loaded-host
    protection.
    """
    budget = _load_scaled_grace(base_secs, cap_secs=cap_secs) + extra_secs
    _wait_for_path(path, timeout=budget)
    return budget


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


def _hermetic_environ() -> dict[str, str]:
    """Return a copy of the process environment with every known ambient
    leak scrubbed -- the shared base for every env-construction site in
    this file.
    """
    env = dict(os.environ)
    env.pop("ESCALATION_TERMINAL_CMD", None)
    # The host's ~/.claude/settings.json env block (belt-and-braces layer of
    # the transcript-persistence fix) injects this into every Bash subprocess
    # -- including this pytest run. Drop it so the persistence-export tests
    # assert spawn-claude.sh's OWN unconditional export, not an ambient leak.
    env.pop("CLAUDE_CODE_FORCE_SESSION_PERSISTENCE", None)
    # skills/spawn/spawn-claude.sh exports CLAUDE_SPAWN_SESSION_ID/PARENT_ID/
    # WM_TITLE/RESULT_FILE into every session it launches (orchestrator adds
    # ROLE/PROJECT/TASK_ID on top), so a suite run from INSIDE a spawned
    # session -- e.g. an L2 escalation-watcher /unblock session running this
    # suite before submitting a merge -- inherits them, while the merge
    # worker's clean systemd unit never does, hiding the leak from CI.
    # Prefix-generic by design (3rd point-fix of this class; the set keeps
    # growing) -- sibling mechanism for the orchestrator suite:
    # orchestrator/tests/test_session_hooks.py::_clear_claude_spawn_env
    # (task 2643).
    for key in [k for k in env if k.startswith("CLAUDE_SPAWN_")]:
        env.pop(key, None)
    return env


def _base_env(bin_dir: pathlib.Path, terminal_name: str) -> dict[str, str]:
    env = _hermetic_environ()
    env["PATH"] = str(bin_dir) + ":" + env.get("PATH", "")
    env["CLAUDE_TERMINAL_CMD"] = terminal_name
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


# _SPAWN_RUN_CAP_SECS: derived, not tuned. Worst-case single-test
# composition on this channel is one _run_spawn/proc.wait budget plus at
# most one _wait_for_path_scaled readiness gate (verified across every
# _run_spawn call site that is followed by a _wait_for_path_scaled call:
# _run_sibling_capture_spawn, test_sibling_mode_is_fire_and_forget, and
# test_sibling_mode_foreground_emulator_is_fire_and_forget -- named by
# function rather than line number, since line numbers rot as the file
# shifts), so 120 + _READINESS_WAIT_CAP_SECS (30) = 150s, 2x headroom
# inside the governing --timeout=300 (scripts/orchestrator.yaml's
# test_command key -- the repo-root pyproject.toml sets no timeout and
# shared/pyproject.toml's timeout=60 does not govern this file). Measured
# happy path for the flaking test: 1.36-2.19s per param (n=6) at
# load-per-core 2.2, so 120 is ~55x.
#
# Deliberately LARGER than _READINESS_WAIT_CAP_SECS (30): a readiness-wait
# cap is paid in full on the failure path, whereas a subprocess wall-clock
# bound is paid only when the child genuinely hangs -- the happy path
# returns the instant the child exits. At the load-per-core 6.6 task 3451
# documented for this host, base 30 scales to ceil(30*6.6)=198, so a cap of
# 30 or 60 would discard most of the headroom this change exists to buy.
_SPAWN_RUN_CAP_SECS = 120


def _spawn_run_budget(base_secs: int) -> int:
    """Load-scale a whole-invocation must-not-hang bound, floored and capped.

    This is a must-not-hang guard, NOT a latency SLA -- every _run_spawn
    caller's real contract is an exit code, not a wall-clock duration.

    Returns the budget so the policy is assertable with zero sleeping,
    which is why no source-grepping meta-test is needed to pin the fix
    (same rationale as _wait_for_path_scaled above and _set_started_grace
    below).

    Delegates entirely to _load_scaled_grace, which floors at base_secs: an
    idle host is byte-identical to the pre-existing fixed pins at every
    _run_spawn call site, so this change can only lengthen a budget under
    contention, never shorten one.
    """
    return _load_scaled_grace(base_secs, cap_secs=_SPAWN_RUN_CAP_SECS)


def _run_spawn(
    env: dict[str, str],
    cwd: pathlib.Path,
    *,
    timeout: int = 30,
    title: str = "",
    scale_timeout: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    """Invoke spawn-claude.sh and return its completed process.

    `timeout` is a load-scaled BASE, not a fixed ceiling: it is a
    must-not-hang guard, not a latency SLA -- every caller's real contract
    is an exit code, not a wall-clock duration. Routed through
    _spawn_run_budget by default, whose _load_scaled_grace floor makes an
    idle host byte-identical to today's fixed pins at all ~25 call sites,
    so this can only lengthen the bound under contention, never shorten
    one.

    scale_timeout=False is the documented opt-out for call sites whose
    verdict is load-INSENSITIVE (the no-emulator/no-tmux 126 sites, per
    task 3486's audit) -- there, scaling would only make a genuine
    regression take longer to report.

    `timeout` must be passed as an UNSCALED base. A caller that already
    computed a load-adaptive value (e.g. via _set_started_grace or
    _load_scaled_grace) and adds fixed margin on top -- as the
    "must-not-be-flagged" family below does (grace + sleep + margin) --
    must also pass scale_timeout=False, or _spawn_run_budget scales an
    already-scaled number a second time, discarding that site's own
    derivation under load instead of honoring it.
    """
    budget = _spawn_run_budget(timeout) if scale_timeout else timeout
    return subprocess.run(
        [str(SPAWN_SCRIPT), str(cwd), "false", title, "test prompt"],
        env=env,
        capture_output=True,
        timeout=budget,
    )


# ===========================================================================
# task-3062: hermetic environment scrub for CLAUDE_SPAWN_* ambient leakage
# ===========================================================================
# Every test in this file that builds a child env from `dict(os.environ)`
# inherits whatever CLAUDE_SPAWN_* vars happen to be set in the *runner's*
# own environment -- real inside any spawned session (every L2
# escalation-watcher /unblock session runs this suite before submitting a
# merge) but invisible on the merge worker's systemd unit, which starts
# clean. That asymmetry already produced two point-fixes in `_base_env`
# (ESCALATION_TERMINAL_CMD, CLAUDE_CODE_FORCE_SESSION_PERSISTENCE) plus a
# latent, still-passing false-negative below (test_no_emulator_found_yields_126
# silently takes the tmux branch under an ambient CLAUDE_SPAWN_BACKEND=tmux).
# The tests here pin the fix deterministically, by setting the leaking
# variable themselves rather than depending on the runner's ambient state.


def test_base_env_scrubs_every_claude_spawn_var(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_base_env` must drop every CLAUDE_SPAWN_*-prefixed var from the
    inherited environment -- not just the vars this file happens to name
    today.

    Sets the four vars spawn-claude.sh itself splices into a spawned
    session (SESSION_ID, PARENT_ID, WM_TITLE, RESULT_FILE), the three the
    orchestrator adds on top (ROLE, PROJECT, TASK_ID -- see task 2940's
    extension to
    orchestrator/tests/test_session_hooks.py::_clear_claude_spawn_env), and
    a synthetic CLAUDE_SPAWN_FUTURE_KNOB that exists nowhere in the
    codebase. The synthetic one is the whole point: it is the only
    assertion that can distinguish a prefix-generic scrub from a fourth
    named enumeration.
    """
    for var in (
        "CLAUDE_SPAWN_SESSION_ID",
        "CLAUDE_SPAWN_PARENT_ID",
        "CLAUDE_SPAWN_WM_TITLE",
        "CLAUDE_SPAWN_RESULT_FILE",
        "CLAUDE_SPAWN_ROLE",
        "CLAUDE_SPAWN_PROJECT",
        "CLAUDE_SPAWN_TASK_ID",
        "CLAUDE_SPAWN_FUTURE_KNOB",
    ):
        monkeypatch.setenv(var, "leak")
    monkeypatch.setenv("ESCALATION_TERMINAL_CMD", "leak")
    monkeypatch.setenv("CLAUDE_CODE_FORCE_SESSION_PERSISTENCE", "leak")

    bin_dir = _make_bin_dir(tmp_path)
    env = _base_env(bin_dir, "xterm")

    leaked = [k for k in env if k.startswith("CLAUDE_SPAWN_")]
    assert leaked == [], f"expected no CLAUDE_SPAWN_* vars to survive, found {leaked}"
    assert "ESCALATION_TERMINAL_CMD" not in env
    assert "CLAUDE_CODE_FORCE_SESSION_PERSISTENCE" not in env

    # The scrub must not be over-broad: the positive setup still survives.
    assert env["CLAUDE_TERMINAL_CMD"] == "xterm"
    assert env["SPAWN_LAUNCH_GRACE_SECS"] == "2"
    assert "CLAUDE_FLEET_ROOT" in env
    assert "CLAUDE_PROJECTS_DIR" in env
    assert env["PATH"].startswith(str(bin_dir) + ":")


def test_spawn_omits_wm_title_export_when_ambient_wm_title_leaks(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end reproduction of the reported false-red, made deterministic.

    The sibling test below (test_spawn_omits_wm_title_export_when_title_empty)
    only fails when the *runner's* own environment happens to carry
    CLAUDE_SPAWN_WM_TITLE -- true inside a spawned session but never true on
    the merge worker's clean systemd unit. This test sets the leak itself
    via monkeypatch, so it fails on the merge worker too.
    """
    monkeypatch.setenv("CLAUDE_SPAWN_WM_TITLE", "ambient-leak-sentinel")

    bin_dir = _make_bin_dir(tmp_path)
    capture_file = tmp_path / "captured_env.txt"
    _write_fake_claude_capturing_env(bin_dir, capture_file)
    _write_foreground_terminal(bin_dir, "xterm")
    env = _base_env(bin_dir, "xterm")

    result = _run_spawn(env, tmp_path, title="")
    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    captured = _parse_captured_env(capture_file)
    assert captured.get("CLAUDE_SPAWN_WM_TITLE", "") == "", (
        f"expected no wm-title export for an empty title, got {captured!r}"
    )


@pytest.mark.skipif(
    __import__("platform").system() == "Darwin",
    reason="exit-126 path requires non-Darwin host",
)
def test_no_emulator_found_yields_126_ignores_ambient_spawn_backend(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the OTHER `dict(os.environ)` sites in this file are hermetic
    too, not just `_base_env`: an ambient CLAUDE_SPAWN_BACKEND=tmux must not
    reroute this no-emulator scenario down the tmux lane.

    Both leak paths exit 126 (spawn-claude.sh's "tmux not found" branch vs.
    its "no terminal emulator found" branch), so an exit-code-only assertion
    can't discriminate between them -- confirmed empirically that
    `CLAUDE_SPAWN_BACKEND=tmux pytest ...::test_no_emulator_found_yields_126`
    passes today despite taking the wrong branch. This test asserts on
    stderr content instead.

    RED before this file's own _hermetic_environ() fix: a raw
    `dict(os.environ)` (this test's own construction, mirroring
    test_no_emulator_found_yields_126 and test_tmux_backend_missing_tmux_
    yields_126 below) passes the ambient CLAUDE_SPAWN_BACKEND straight
    through, so the run takes the tmux branch and stderr says "tmux not
    found" instead of "no terminal emulator found".
    """
    import shutil as _shutil

    monkeypatch.setenv("CLAUDE_SPAWN_BACKEND", "tmux")

    bin_dir = _make_bin_dir(tmp_path)
    _write_fake_claude(bin_dir, exit_code=0)

    # Minimal system-bin with only the utilities the script needs -- NO
    # tmux, NO terminal emulator (mirrors test_no_emulator_found_yields_126's
    # sys_bin exactly).
    sys_bin = tmp_path / "sys_bin"
    sys_bin.mkdir()
    for util in ["bash", "mktemp", "sleep", "cat", "rm", "uname"]:
        src = _shutil.which(util)
        if src:
            (sys_bin / util).symlink_to(src)

    env = _hermetic_environ()
    env["PATH"] = str(bin_dir) + ":" + str(sys_bin)
    env.pop("CLAUDE_TERMINAL_CMD", None)

    # Task 3599 audit: same no-emulator availability-guard shape as
    # test_no_emulator_found_yields_126 below -- rc==126 is load-INSENSITIVE
    # (task 3486), so this stays unscaled rather than inheriting the new
    # load-scaled default.
    result = _run_spawn(env, tmp_path, timeout=10, scale_timeout=False)
    stderr = result.stderr.decode()
    assert result.returncode == 126, (
        f"expected 126, got {result.returncode}\nstderr: {stderr}"
    )
    assert "no terminal emulator found" in stderr, (
        f"expected the no-emulator branch (ambient CLAUDE_SPAWN_BACKEND must "
        f"not leak through), got:\n{stderr}"
    )
    assert "tmux" not in stderr.lower(), (
        f"expected the tmux branch NOT to run, got:\n{stderr}"
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

    Synchronization contract -- the ORDERING below is load-independent:
      1. _DETACHING_TERM_TEMPLATE publishes the leader pid IMMEDIATELY after
         setsid (no blind sleep).
      2. The fake claude writes a readiness marker file before exec sleep 300.
      3. spawn-claude.sh's $inner arms EXIT/HUP/TERM traps BEFORE invoking
         claude, so the readiness marker is a sound proof that the HUP trap
         is installed.
      4. The test waits for BOTH the pidfile AND the readiness marker before
         sending SIGHUP — SIGHUP is always delivered after trap installation.

    The WAIT BUDGETS around that ordering are a separate matter and are NOT
    load-independent -- they are load-scaled via _wait_for_path_scaled (task
    3486), after a burst-load excursion past a fixed 5.0s pidfile timeout was
    observed here in test_window_close_yields_129_not_hang[konsole].
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
    # Budgets are load-scaled (task 3486's _wait_for_path_scaled) rather than
    # fixed, since a burst-load excursion past a fixed 5.0s pidfile timeout is
    # exactly the flake that was observed here (this test, konsole lane).
    _wait_for_path_scaled(pidfile, 5)
    _wait_for_path_scaled(readyfile, 10)

    leader_pid = int(pidfile.read_text().strip())
    # Send SIGHUP to the entire process group of the session leader.
    # HUP trap is now guaranteed to be installed → exit 129.
    os.killpg(leader_pid, signal.SIGHUP)

    # Must-not-hang guard — NOT a latency SLA.
    # The success path takes ~2-4s (await_sentinel 2s poll + pidfile handshake).
    # A genuine hang is infinite (no sentinel ever written), so a load-scaled
    # budget (base 15s, capped at _READINESS_WAIT_CAP_SECS -- named
    # explicitly rather than relying on _load_scaled_grace's own matching
    # default) cleanly separates pass from hang while staying well under the
    # governing 300s pytest-timeout: this file's rootdir/configfile is the
    # repo-root pyproject.toml (verified via `pytest --collect-only`), whose
    # [tool.pytest.ini_options] sets no `timeout` of its own -- so
    # shared/pyproject.toml's timeout=60 does NOT govern this file, and the
    # real ceiling is the --timeout=300 scripts/orchestrator.yaml:17 passes.
    # This descriptive pytest.fail still fires well before that blunt kill.
    #
    # Deliberately NOT routed through _spawn_run_budget/_SPAWN_RUN_CAP_SECS
    # (task 3599): this readiness-adjacent policy (cap 30) is a separate
    # policy owner from that whole-invocation channel (cap 120) -- see the
    # explicit divergence note in
    # test_failed_to_start_detected_on_detached_exit0, the one Popen+wait
    # site in this file that DOES use _spawn_run_budget.
    try:
        rc = proc.wait(timeout=_load_scaled_grace(15, cap_secs=_READINESS_WAIT_CAP_SECS))
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
    # Task 3599: dropped the fixed timeout=15 pin -- measured happy path
    # 1.36-2.19s per param (n=6) at load-per-core 2.2 in this worktree, yet
    # the old 15s bound was exceeded under merge-verify contention
    # (escalation esc-3495-1, log
    # data/verify-logs/3495/attempt-1.scripts.test-20260803T151949_260976Z.log).
    # This was the only _run_spawn call in the file that LOWERED the bound
    # below the default for a load-SENSITIVE assertion, and gained nothing
    # by doing so: the contract is returncode == 127, not a latency SLA.
    # Now inherits _run_spawn's load-scaled 30s default.
    result = _run_spawn(env, tmp_path)
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

    env = _hermetic_environ()
    env["PATH"] = str(bin_dir) + ":" + str(sys_bin)
    env.pop("CLAUDE_TERMINAL_CMD", None)

    # Task 3599 audit: rc==126 is load-INSENSITIVE (no emulator on PATH
    # fails the availability guard immediately; task 3486 measured 0.05s),
    # so this stays unscaled rather than inheriting the new load-scaled
    # default.
    result = _run_spawn(env, tmp_path, timeout=10, scale_timeout=False)
    assert result.returncode == 126, (
        f"No emulator must yield 126, got {result.returncode}\n"
        f"stderr: {result.stderr.decode()}"
    )


def test_bad_usage_yields_2(tmp_path: pathlib.Path) -> None:
    """Passing wrong number of arguments must exit 2."""
    result = subprocess.run(
        [str(SPAWN_SCRIPT), "only-one-arg"],
        capture_output=True,
        timeout=_spawn_run_budget(5),
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
    # the DELAY + $inner trap-install sequence completes. Both gates route
    # through _wait_for_path_scaled (task 3486) -- the single policy owner
    # for every readiness gate in this file, rather than hand-rolling the
    # load scaling inline -- so a load-slowed-but-correct run doesn't
    # spuriously time out. DELAY is a wall-clock-fixed injected sleep, not
    # load-dependent, so it is passed as extra_secs and added unscaled on
    # top of the scaled base.
    #
    # readyfile passes cap_secs=60 (not the default 30): the old inline form
    # summed two INDEPENDENTLY-capped _load_scaled_grace(5) halves, up to
    # 2*30=60s under load. A bare _wait_for_path_scaled(readyfile, 10) --
    # single 30s cap -- would nearly halve that loaded-host budget (e.g.
    # 61s -> 31s at the load-per-core 6.6 recorded near
    # _NOT_FLAGGED_GRACE_BASE_SECS below); cap_secs=60 mirrors that same
    # constant's own cap raise and keeps this gate's loaded-host protection
    # >= what it replaces. The idle-host floor is unaffected by the cap
    # either way: 5 + DELAY + 5 == _load_scaled_grace(10) + DELAY == 11.0s.
    _wait_for_path_scaled(pidfile, 5)
    _wait_for_path_scaled(readyfile, 10, extra_secs=DELAY, cap_secs=60)

    leader_pid = int(pidfile.read_text().strip())
    # SIGHUP arrives after the HUP trap is armed — must yield exit 129.
    os.killpg(leader_pid, signal.SIGHUP)

    # Deliberately NOT routed through _spawn_run_budget/_SPAWN_RUN_CAP_SECS
    # (task 3599) -- same readiness-adjacent policy (cap 30), left untouched
    # for the same reason as test_window_close_yields_129_not_hang above.
    try:
        rc = proc.wait(timeout=_load_scaled_grace(15, cap_secs=_READINESS_WAIT_CAP_SECS))
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
    written -> proc.wait(timeout=_spawn_run_budget(20)) raises
    TimeoutExpired -> pytest.fail.
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
    # Task 3451 audit: deliberately NOT routed through _set_started_grace.
    # This test asserts the flag MUST fire, and its launcher exits 0
    # without ever running the payload, so no evidence (sentinel,
    # transcript, or claude descendant) can EVER appear -- the watchdog
    # fires regardless of grace, and the verdict is load-insensitive. The
    # short pin only bounds how long the test waits for that inevitable
    # flag; _set_started_grace would merely make it slower.
    env["SPAWN_STARTED_GRACE_SECS"] = "2"

    proc = subprocess.Popen(
        [str(SPAWN_SCRIPT), str(tmp_path), "false", "", "test prompt"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    # Must-not-hang guard, not a latency SLA. Pre-impl this hangs forever
    # (unbounded await_sentinel), so a bounded wait cleanly separates
    # pass/fail from an infinite hang. Task 3599: the bound is now
    # load-scaled via _spawn_run_budget -- the same whole-invocation
    # wall-clock policy _run_spawn itself uses, reached here via Popen+wait
    # instead of subprocess.run. The rc==144 verdict IS load-sensitive (it
    # needs the watchdog to flag AND the parent to exit), unlike the
    # SPAWN_STARTED_GRACE_SECS pin above, which stays fixed on its own,
    # different channel.
    #
    # Deliberate divergence, stated explicitly rather than left implicit:
    # this file has two OTHER Popen+wait must-not-hang sites
    # (test_window_close_yields_129_not_hang and
    # test_window_close_129_robust_to_delayed_trap_install), both of which
    # stay on _load_scaled_grace(15, cap_secs=_READINESS_WAIT_CAP_SECS)
    # (cap 30) -- task 3486's readiness-gate policy, which this task's plan
    # explicitly left untouched as outside its defect. This site does NOT
    # mirror that pattern; it shares _spawn_run_budget's larger cap (120)
    # instead, because it is the same whole-invocation wall-clock channel
    # _run_spawn covers, just reached via Popen+wait rather than
    # subprocess.run.
    try:
        rc = proc.wait(timeout=_spawn_run_budget(20))
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


# ===========================================================================
# Task 2733 step-3/4: _load_scaled_grace -- load-adaptive SPAWN_STARTED_GRACE_SECS
# ===========================================================================
# Second recurrence of a started-grace flake in this file (task 2367 already
# bumped the fixed 1s/2s -> 3s/8s six days ago). A fixed bump chases a moving
# target as host load climbs; _load_scaled_grace instead scales the grace by
# load-per-core -- floored at base_secs (an idle host is byte-identical to
# today) and capped at cap_secs (a pathological host stays bounded).


def _load_scaled_grace(base_secs: int, *, cap_secs: int = 30) -> int:
    """Scale a started-grace budget by host load-per-core, floored and capped.

    A fixed started-grace chases a moving target as host load climbs (this
    is the SECOND recurrence of a started-grace flake in this file -- task
    2367 already bumped a fixed 1s/2s -> 3s/8s six days ago). Load-per-core
    headroom tracks the actual contention that delays the fake claude
    startup chain, instead of chasing that moving target with another
    one-off bump.

    Floored at base_secs: an idle host (loadavg_1min <= cpu_count) returns
    base_secs unchanged, so this is byte-identical to the pre-existing fixed
    grace there -- no regression. Capped at cap_secs so a pathologically
    loaded host stays bounded. Fails safe to base_secs if getloadavg is
    unavailable on this platform.
    """
    try:
        load1 = os.getloadavg()[0]
    except (OSError, AttributeError):
        return base_secs
    factor = max(1.0, load1 / (os.cpu_count() or 1))
    return max(base_secs, min(cap_secs, math.ceil(base_secs * factor)))


# _NOT_FLAGGED_GRACE_BASE_SECS: raised from 2 to 8 (task 3451). Derived, not
# tuned -- on this host (nproc 32, /proc/loadavg 212 => load-per-core 6.6),
# n=3 runs of the normal fast spawn shape (delay=0, grace=2, foreground
# xterm, fake claude exiting 0) took 2.13s / 3.10s / 4.71s wall. The old 2s
# pin sat BELOW that entire observed range -- the complete explanation of
# the flake. 8 > 4.71 gives 1.7x margin from the floor alone, before load
# scaling multiplies on top: at that same load the full policy yields
# min(60, ceil(8 * 6.6)) = 53s, ~11x the worst measured happy path.
#
# The larger grace is free: measured wall-clock is NOT proportional to
# grace (grace=2 -> 2.13-4.71s vs grace=90 (unpinned) -> 2.54-7.28s,
# overlapping ranges) because _cleanup (skills/spawn/spawn-claude.sh:107)
# kills the backgrounded watchdog at parent exit, so the grace is only an
# upper bound the watchdog polls to, never a wait the happy path pays.
#
# cap_secs=60 (below) is RAISED by this change from _load_scaled_grace's
# own default cap of 30 -- not unchanged. The raise is load-bearing, not
# cosmetic: at load-per-core 6.6, ceil(8*6.6)=53 would otherwise be
# clamped down to 30, discarding most of the load headroom the base bump
# from 2 to 8 was meant to buy. 60 stays strictly below the 90s production
# default (skills/spawn/spawn-claude.sh:89), so this pin never tests an
# unreachable configuration.
_NOT_FLAGGED_GRACE_BASE_SECS = 8


def _set_started_grace(env: dict[str, str]) -> int:
    """Compute and write the load-adaptive started-grace for tests that
    assert the failed-to-start flag must NOT fire.

    Delegates entirely to _load_scaled_grace so such tests inherit the
    load-adaptive policy by default instead of each hand-picking a fixed
    number (the third recurrence of a started-grace flake in this file:
    task 2367 bumped 1s/2s -> 3s/8s, task 2733 added _load_scaled_grace, and
    2733 missed this site). Writing the env var is part of the contract, not
    a side effect a caller must remember to do -- it is what makes the
    policy deterministically unit-testable (assert the returned int and the
    string that landed in env) without a source-grepping meta-test to prove
    call sites were rewired.
    """
    grace = _load_scaled_grace(_NOT_FLAGGED_GRACE_BASE_SECS, cap_secs=60)
    env["SPAWN_STARTED_GRACE_SECS"] = str(grace)
    return grace


def test_load_scaled_grace_idle_host_returns_base_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load-per-core <= 1 (idle host) floors at base_secs -- no scaling up."""
    monkeypatch.setattr(os, "getloadavg", lambda: (10.0, 10.0, 10.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 32)

    assert _load_scaled_grace(3, cap_secs=30) == 3


def test_load_scaled_grace_scales_up_with_load_per_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load-per-core > 1 scales grace up to ceil(base_secs * loadavg / cpu_count)."""
    monkeypatch.setattr(os, "getloadavg", lambda: (64.0, 64.0, 64.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 32)

    assert _load_scaled_grace(3, cap_secs=30) == 6


def test_load_scaled_grace_clamps_to_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pathological load is clamped at cap_secs instead of growing unbounded."""
    monkeypatch.setattr(os, "getloadavg", lambda: (3200.0, 3200.0, 3200.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 32)

    assert _load_scaled_grace(3, cap_secs=30) == 30


def test_load_scaled_grace_getloadavg_error_returns_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """getloadavg unavailable (OSError or AttributeError) fails safe to base_secs."""

    def _raise_oserror() -> tuple[float, float, float]:
        raise OSError("getloadavg not supported on this platform")

    monkeypatch.setattr(os, "getloadavg", _raise_oserror)
    assert _load_scaled_grace(3, cap_secs=30) == 3

    def _raise_attributeerror() -> tuple[float, float, float]:
        raise AttributeError("os has no getloadavg on this platform")

    monkeypatch.setattr(os, "getloadavg", _raise_attributeerror)
    assert _load_scaled_grace(3, cap_secs=30) == 3


# ===========================================================================
# Task 3486: _wait_for_path_scaled -- load-scaled readiness-gate policy
# ===========================================================================
# FOURTH recurrence of the fixed-timeout-vs-load-dependent-startup flake
# class in this file: task 2367 bumped a fixed started-grace 1s/2s -> 3s/8s;
# task 2733 added _load_scaled_grace above; task 3451 added _set_started_grace
# for the started-grace family; task 3486 (here) covers the _wait_for_path
# readiness-gate family. Observed instance:
# test_window_close_yields_129_not_hang[konsole] timing out at
# _wait_for_path(pidfile, timeout=5.0) during task 3451's step-7 full-suite
# verify -- passing in isolation and on immediate rerun (a burst-load
# excursion past a fixed pin, not a genuine hang).


def test_wait_for_path_scaled_returns_load_scaled_budget_on_loaded_host(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a loaded host, the returned budget matches _load_scaled_grace's own
    output -- used as the oracle for arg-forwarding (cap_secs in
    particular), so a change to how arguments reach _load_scaled_grace is
    pinned here without duplicating its floor/scale/clamp/error-safe
    arithmetic, already pinned once by the test_load_scaled_grace_* family
    above. That oracle alone can't catch a bug shared by both functions, so
    a literal expected value is also pinned below (96.0 loadavg / 32 cores
    => load-per-core 3.0, base 5 => ceil(5 * 3.0) = 15).

    The path already exists, so the call returns immediately: this is what
    makes the policy assertable without ever sleeping.
    """
    monkeypatch.setattr(os, "getloadavg", lambda: (96.0, 96.0, 96.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 32)

    existing = tmp_path / "already-there"
    existing.touch()

    budget = _wait_for_path_scaled(existing, 5)
    assert budget == 15
    assert budget == _load_scaled_grace(5, cap_secs=_READINESS_WAIT_CAP_SECS)


def test_wait_for_path_scaled_idle_host_floors_at_base(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idle host (load-per-core < 1) floors at base_secs unchanged --
    pinning the no-regression property that every rewired call site stays
    byte-identical to the fixed pin it replaces on an unloaded host.
    """
    monkeypatch.setattr(os, "getloadavg", lambda: (10.0, 10.0, 10.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 32)

    existing = tmp_path / "already-there"
    existing.touch()

    assert _wait_for_path_scaled(existing, 5) == 5


def test_wait_for_path_scaled_enforces_the_scaled_budget_not_the_base(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagation test -- the one property the return value alone cannot
    prove: a buggy implementation could return the scaled number but still
    pass the raw, unscaled base (or, for the second case below, a budget
    without extra_secs) through to _wait_for_path. Points at a path that
    never appears so the real enforced timeout is observable both via the
    raised message and via measured wall-clock.

    Deliberately sized at ~2s/~3s of real wall-clock (base_secs=1 x
    load-per-core 2.0 => scaled budget 2; plus extra_secs=1.0 => 3); a
    larger base would only make the suite slower without pinning anything
    further.
    """
    monkeypatch.setattr(os, "getloadavg", lambda: (64.0, 64.0, 64.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 32)

    missing = tmp_path / "never-appears"

    start = time.monotonic()
    with pytest.raises(AssertionError, match=r"Timed out after 2"):
        _wait_for_path_scaled(missing, 1)
    elapsed = time.monotonic() - start

    assert elapsed >= 2.0, (
        f"expected the SCALED budget (2s), not the 1s base, to be enforced; "
        f"only waited {elapsed:.2f}s"
    )

    # extra_secs must reach _wait_for_path too, not just the return value --
    # a buggy impl could compute `scaled + extra_secs` for the return but
    # pass only `scaled` through to _wait_for_path, which the assertion
    # above alone (extra_secs defaults to 0.0 there) would not catch.
    start = time.monotonic()
    with pytest.raises(AssertionError, match=r"Timed out after 3"):
        _wait_for_path_scaled(missing, 1, extra_secs=1.0)
    elapsed = time.monotonic() - start

    assert elapsed >= 3.0, (
        f"expected the SCALED budget + extra_secs (3s) to be enforced; "
        f"only waited {elapsed:.2f}s"
    )


def test_wait_for_path_scaled_adds_extra_secs_unscaled(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """extra_secs is added ON TOP of the scaled base, unscaled by load and
    exempt from the cap -- the one gate shape a bare (path, base_secs)
    signature cannot express:
    test_window_close_129_robust_to_delayed_trap_install's readyfile gate,
    which waits out a deliberately injected, wall-clock-fixed sleep (DELAY)
    on top of the load-dependent subprocess-startup chain. A `sleep 1.0` in
    a shell script takes 1.0s regardless of host load, so scaling it would
    inflate the budget for a component that provably does not stretch; and
    clamping it would silently eat a delay the test deliberately injected.

    Both cases use an already-existing path so the call returns instantly.
    """
    existing = tmp_path / "already-there"
    existing.touch()

    # Loaded host (load-per-core 2.0): extra_secs is added on top of the
    # scaled base, not folded into the scaling itself.
    monkeypatch.setattr(os, "getloadavg", lambda: (64.0, 64.0, 64.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    assert _wait_for_path_scaled(
        existing, 10, extra_secs=1.0
    ) == _load_scaled_grace(10, cap_secs=_READINESS_WAIT_CAP_SECS) + 1.0

    # Pathological host: the cap clamps only the scaled part; extra_secs
    # survives the clamp untouched. Mirrors test_load_scaled_grace_clamps_to_cap
    # one layer up.
    monkeypatch.setattr(os, "getloadavg", lambda: (3200.0, 3200.0, 3200.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    assert (
        _wait_for_path_scaled(existing, 10, extra_secs=1.0)
        == _READINESS_WAIT_CAP_SECS + 1.0
    )


def test_wait_for_path_scaled_cap_secs_override_widens_the_clamp(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cap_secs overrides _READINESS_WAIT_CAP_SECS per call site -- the knob
    test_window_close_129_robust_to_delayed_trap_install's readyfile gate
    relies on (cap_secs=60) to keep its loaded-host budget >= the two
    independently-capped _load_scaled_grace(5) halves it replaced. Pinned
    here at the policy layer so the override itself is tested once, rather
    than only implicitly through that call site. Mirrors
    test_load_scaled_grace_clamps_to_cap one layer up.
    """
    monkeypatch.setattr(os, "getloadavg", lambda: (3200.0, 3200.0, 3200.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 32)

    existing = tmp_path / "already-there"
    existing.touch()

    # Default cap_secs still clamps at _READINESS_WAIT_CAP_SECS...
    assert _wait_for_path_scaled(existing, 10) == _READINESS_WAIT_CAP_SECS
    # ...but an explicit wider cap_secs clamps there instead.
    assert _wait_for_path_scaled(existing, 10, cap_secs=60) == 60


# ===========================================================================
# Task 3599: _spawn_run_budget -- load-scaled must-not-hang guard for _run_spawn
# ===========================================================================
# FIFTH recurrence of the fixed-timeout-vs-load-dependent-startup flake class
# in this file: task 2367 fixed-bumped a started-grace 1s/2s -> 3s/8s; task
# 2733 added _load_scaled_grace; task 3451 added _set_started_grace; task 3486
# added _wait_for_path_scaled for the readiness-gate family; task 3599 (here)
# covers the whole-invocation wall-clock channel -- the `timeout` _run_spawn
# hands to subprocess.run. Observed instance:
# test_genuine_launcher_failure_yields_127[xterm] raising
# subprocess.TimeoutExpired after a fixed 15s in merge worktree
# _merge-dd5a8aa6 (escalation esc-3495-1, archived log
# data/verify-logs/3495/attempt-1.scripts.test-20260803T151949_260976Z.log)
# while passing in isolation.


def test_spawn_run_cap_leaves_headroom_inside_governing_timeout() -> None:
    """_SPAWN_RUN_CAP_SECS must leave headroom inside the governing
    --timeout=300 (scripts/orchestrator.yaml's test_command key) even
    stacked with one _wait_for_path_scaled readiness gate -- the worst-case
    single-test composition _run_sibling_capture_spawn,
    test_sibling_mode_is_fire_and_forget, and
    test_sibling_mode_foreground_emulator_is_fire_and_forget each exercise
    (a _run_spawn call followed by a _wait_for_path_scaled call).

    Deliberately just this one static invariant between the two module-
    level constants, no monkeypatching: the scale/floor/clamp arithmetic
    _spawn_run_budget delegates to is already pinned by the
    test_load_scaled_grace_* family above, and _spawn_run_budget's own
    forwarding of it is pinned against real subprocess.run calls by the
    test_run_spawn_* family below -- re-deriving that arithmetic a third
    time here would be pure duplication (task 3599 amendment; a prior
    revision of this test file had three test_spawn_run_budget_* tests
    doing exactly that).
    """
    assert _SPAWN_RUN_CAP_SECS + _READINESS_WAIT_CAP_SECS < 300


def _capture_spawn_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, **kwargs
) -> float:
    """Invoke _run_spawn with subprocess.run monkeypatched to a no-op stub
    that records the `timeout` kwarg it was handed, so the load-scaling
    policy is pinned as runtime behaviour (the actual argument
    subprocess.run receives) without ever launching a real subprocess or
    sleeping.
    """
    captured: dict[str, float] = {}

    def _fake_run(argv, **kw):
        captured["timeout"] = kw["timeout"]
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    _run_spawn({}, tmp_path, **kwargs)
    return captured["timeout"]


def test_run_spawn_scales_its_timeout_under_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A fixed wall-clock bound races a host-load-dependent startup chain --
    observed as test_genuine_launcher_failure_yields_127[xterm] timing out
    at a fixed 15s under merge-verify contention. The default `timeout=30`
    _run_spawn hands to subprocess.run must itself be load-adaptive, not a
    fixed 30, so every one of the ~20 call sites on the bare default is
    covered by a single fix.
    """
    monkeypatch.setattr(os, "getloadavg", lambda: (96.0, 96.0, 96.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 32)

    timeout = _capture_spawn_timeout(monkeypatch, tmp_path)
    assert timeout == _spawn_run_budget(30) == 90


def test_run_spawn_explicit_timeout_is_a_scaled_base_not_a_fixed_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """An explicit `timeout=` argument is treated as a BASE routed through
    _spawn_run_budget, not a ceiling -- so a caller that dialed down its
    bound (e.g. the old timeout=15 at the reported flake site) still gets
    load protection instead of racing the same fixed pin under contention.
    """
    monkeypatch.setattr(os, "getloadavg", lambda: (96.0, 96.0, 96.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 32)

    timeout = _capture_spawn_timeout(monkeypatch, tmp_path, timeout=20)
    assert timeout == _spawn_run_budget(20) == 60


def test_run_spawn_idle_host_timeout_is_byte_identical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The no-regression guarantee for every existing call site: on an idle
    host (load-per-core < 1), both the default and an explicit timeout
    reach subprocess.run completely unchanged. Passes both before and
    after step-4 -- intentionally; this is the no-regression guard, not a
    RED test.
    """
    monkeypatch.setattr(os, "getloadavg", lambda: (10.0, 10.0, 10.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 32)

    assert _capture_spawn_timeout(monkeypatch, tmp_path) == 30
    assert _capture_spawn_timeout(monkeypatch, tmp_path, timeout=10) == 10


def test_run_spawn_scale_timeout_false_forwards_base_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """scale_timeout=False is the documented opt-out that preserves task
    3486's audited decision for the load-INSENSITIVE 126/no-emulator sites
    (rc==126 is decided by an immediate availability-guard failure; load-
    scaling would only make a genuine regression take longer to report).
    Pinned as a contract here rather than left as an undocumented
    convention, so a future edit to _run_spawn cannot silently drop it.
    """
    monkeypatch.setattr(os, "getloadavg", lambda: (96.0, 96.0, 96.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 32)

    timeout = _capture_spawn_timeout(
        monkeypatch, tmp_path, scale_timeout=False, timeout=10
    )
    assert timeout == 10


# ===========================================================================
# Task 3451: _set_started_grace -- shared started-grace policy for the
# "must NOT be flagged failed-to-start" test family
# ===========================================================================
# Third recurrence of a started-grace flake in this file (task 2367 bumped a
# fixed 1s/2s -> 3s/8s; task 2733 added _load_scaled_grace above but missed
# wiring test_normal_spawn_exit0_not_flagged to it, leaving it pinned at a
# fixed "2" against a 90s production default -- skills/spawn/spawn-claude.sh:89).
# _set_started_grace both computes the load-scaled grace via
# _load_scaled_grace AND writes it into env["SPAWN_STARTED_GRACE_SECS"], so
# the fix is deterministically unit-testable here -- assert the returned int
# and the string that landed in env -- instead of requiring a forbidden
# source-grepping meta-test to prove call sites were rewired.


def test_set_started_grace_writes_env_matching_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_set_started_grace delegates to _load_scaled_grace and writes the
    identical value into env["SPAWN_STARTED_GRACE_SECS"] as a string.

    The floor/scale/cap arithmetic itself is already pinned three ways by
    the test_load_scaled_grace_* tests above (idle/scale/clamp/error-safe);
    re-deriving that same arithmetic here through _set_started_grace would
    just be duplicate coverage of task 2733's tests. The only contract that
    is genuinely new at this layer is that _set_started_grace's return
    value and the string it writes into env agree -- so this test uses
    _load_scaled_grace itself as the oracle rather than hardcoding an
    expected number, on an IDLE host (load-per-core < 1) where
    _load_scaled_grace floors at the bare base unchanged.

    That idle-host floor is also where _NOT_FLAGGED_GRACE_BASE_SECS's own
    value must clear the parent script's measured happy-path startup
    latency -- not just be a low fixed number. MEASURED, not guessed: on
    this host (nproc 32, /proc/loadavg 212 => load-per-core 6.6) three runs
    of the normal fast spawn shape (delay=0, grace=2, foreground xterm,
    fake claude exiting 0) took 2.13s / 3.10s / 4.71s wall -- the whole
    observed range sits ABOVE the old 2s pin, which is the complete
    explanation of the reported flake in test_normal_spawn_exit0_not_flagged
    (registry status intermittently failed-to-start instead of exited). A
    floor of 8s clears the 4.71s worst case with a 1.7x margin, before any
    load scaling multiplies on top of it.
    """
    monkeypatch.setattr(os, "getloadavg", lambda: (10.0, 10.0, 10.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 32)

    env: dict[str, str] = {}
    grace = _set_started_grace(env)

    assert grace == _load_scaled_grace(_NOT_FLAGGED_GRACE_BASE_SECS, cap_secs=60)
    assert env["SPAWN_STARTED_GRACE_SECS"] == str(grace)
    assert _NOT_FLAGGED_GRACE_BASE_SECS >= 8, (
        "must-not-be-flagged started-grace floor must clear the measured "
        f"worst-case happy-path spawn latency (4.71s at load-per-core 6.6); "
        f"got {_NOT_FLAGGED_GRACE_BASE_SECS}"
    )


def test_transcript_appearance_suppresses_flag(tmp_path: pathlib.Path) -> None:
    """A fresh transcript file must suppress the failed-to-start flag.

    Proves the transcript detector is load-bearing. Uses a DETACHING launcher
    (custom-term, routing through resolve_detached's launch_rc==0 branch --
    the incident path) whose fake claude writes a transcript file under
    $CLAUDE_PROJECTS_DIR/<enc>/ the moment it starts, then sleeps a fixed 8s
    before exiting and letting $inner write the sentinel -- so the
    transcript evidence is available from t~0, long before the sentinel can
    possibly exist, and _started_watchdog (which polls continuously and
    returns on first evidence) observes only the transcript. <enc> is COMPUTED
    by session_registry.encode_cwd (the canonical: every '/', '.' and '_' maps
    to '-', case preserved) rather than restated in prose or hand-copied here
    -- see the comment at the assignment below for why.

    Grace is load-adaptive via _set_started_grace (task 3451), which shares
    one policy across all three must-not-be-flagged sites in this file.
    Originally task 2733's bare _load_scaled_grace(3) -- but a base of 3s
    was ALSO below the measured 4.71s worst-case happy-path chain latency
    (load-per-core 6.6), leaving residual exposure at low-but-nonzero load.
    Under merge-verify xdist contention the fake launcher->claude->transcript
    startup chain can take longer than any fixed margin -- this is the THIRD
    recurrence of this exact flake (task 2367 already bumped the fixed value
    1s/2s -> 3s/8s six days before task 2733's fix, which task 3451 now
    supersedes here).

    The fake-claude sleep stays FIXED at 8s, decoupled from the now-larger
    grace: the only validity requirement is that the exit sentinel lands
    AFTER the transcript evidence appears, and since the transcript file
    persists once written (and the watchdog polls continuously within the
    grace window), a fixed modest sleep keeps the sentinel strictly after
    evidence while keeping the happy path fast (runs ~startup+sleep, not
    ~startup+grace). The outer _run_spawn timeout is grace-relative
    (grace + sleep + margin) so the same load that enlarges grace cannot
    convert a 144-flake into a subprocess-TimeoutExpired flake.
    """
    bin_dir = _make_bin_dir(tmp_path)

    # THE canonical encoder, not a hand-copied expression. A local
    # str.replace chain here is a mirror of the code under test, so it moves in
    # lockstep with a bug in that code and can never detect one -- exactly how
    # the missing '_' -> '-' rule survived a fully green suite (task 3272).
    # encode_cwd is itself pinned to hard-coded real on-disk dir names by
    # test_legibility_inventory.py's TestEncoderLockstep, so calling it gives
    # this fixture a real oracle transitively. (A hard-coded literal, the
    # strongest option, is not available: tmp_path is generated per run.)
    enc = session_registry.encode_cwd(str(tmp_path))

    # Fake claude: write the transcript file immediately (mirroring a real
    # Claude Code session creating ~/.claude/projects/<enc>/*.jsonl the moment
    # it starts), then sleep a fixed 8s before exiting -- the transcript
    # lands at t~0, long before the sentinel can exist, so only the
    # transcript probe (not the sentinel) can suppress the flag.
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
    grace = _set_started_grace(env)

    # scale_timeout=False (task 3599 amendment): `grace` is already
    # load-scaled (_set_started_grace delegates to _load_scaled_grace), so
    # routing grace + 8 + 6 through _spawn_run_budget would scale an
    # already-scaled number a second time -- e.g. at load-per-core 3.0,
    # grace=24 and the sum 38 would become ceil(38*3)=114, discarding this
    # margin's own derivation instead of honoring it.
    result = _run_spawn(env, tmp_path, timeout=grace + 8 + 6, scale_timeout=False)

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
    at all under $CLAUDE_PROJECTS_DIR, but stays alive (sleeping) for a fixed
    6s before exiting -- so the watchdog observes it as a live descendant
    long before it exits, regardless of grace. Since xterm's fake terminal
    `exec`s into the payload bash (see _FOREGROUND_TERM_SCRIPT), claude runs
    as a direct descendant of spawn-claude.sh's own $$ -- unlike a detached
    launcher (setsid + background job, reparented once the launcher process
    exits), where this probe is correctly always empty.

    Grace is load-adaptive via _set_started_grace (task 3451), the same
    policy as test_transcript_appearance_suppresses_flag. Originally task
    2733's bare _load_scaled_grace(2) -- but a base of 2s was ALSO below
    the measured 4.71s worst-case happy-path chain latency (load-per-core
    6.6), leaving residual exposure at low-but-nonzero load. All three
    must-not-be-flagged sites in this file now share one policy.

    The fake-claude sleep stays FIXED at 6s, decoupled from the now-larger
    grace: the only validity requirement is that the exit sentinel lands
    AFTER _claude_descendant_alive observes the live descendant, and since
    the 6s live window hugely exceeds the watchdog's 0.25s poll interval (and
    the watchdog polls continuously within the grace window), a fixed modest
    sleep keeps the sentinel strictly after evidence while keeping the happy
    path fast. The outer _run_spawn timeout is grace-relative
    (grace + sleep + margin) so the same load that enlarges grace cannot
    convert a 144-flake into a subprocess-TimeoutExpired flake.
    """
    bin_dir = _make_bin_dir(tmp_path)
    _write_foreground_terminal(bin_dir, "xterm")

    # Fake claude: writes NO transcript anywhere, just stays alive (sleeping)
    # for a fixed 6s before exiting -- so only _claude_descendant_alive (not
    # the transcript probe) can suppress the flag.
    claude = bin_dir / "claude"
    claude.write_text("#!/usr/bin/env bash\nsleep 6\nexit 0\n")
    claude.chmod(0o755)

    env = _base_env(bin_dir, "xterm")
    grace = _set_started_grace(env)

    # scale_timeout=False (task 3599 amendment): `grace` is already
    # load-scaled (_set_started_grace delegates to _load_scaled_grace), so
    # a second pass through _spawn_run_budget would double-count contention
    # -- see the identical rationale on the transcript-evidence test above.
    result = _run_spawn(env, tmp_path, timeout=grace + 6 + 6, scale_timeout=False)

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
    # Task 3451 audit: deliberately NOT routed through _set_started_grace.
    # Here the SHORT grace IS the premise -- it must stay well below the
    # SPAWN_LAUNCH_GRACE_SECS="5" set on the line above, and load-scaling a
    # base of 1 (via _set_started_grace or _load_scaled_grace) could exceed
    # 5 under contention, inverting the exact 127-vs-144 ordering this test
    # exists to pin. This is the opposite family from _set_started_grace's
    # must-not-fire tests: here the flag firing fast is fine, so long as
    # resolve_foreground's 127 verdict still wins the race.
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

    Task 3451: fixes a load-sensitive flake from pinning
    SPAWN_STARTED_GRACE_SECS to a fixed "2" against a 90s production
    default. Under merge-verify contention the parent's own
    launcher->claude->sentinel chain outran the fixed 2s window while all
    three watchdog probes (sentinel, transcript, live claude descendant)
    were still empty, so the watchdog overwrote the registry record with
    failed-to-start AFTER the parent had already written exited, and
    _cleanup (skills/spawn/spawn-claude.sh:107) then killed the watchdog
    before its stderr echo -- which is exactly why the reported failure
    showed registry=failed-to-start with a CLEAN stderr: it passed the
    "failed-to-start" not in stderr assertion below and failed only the
    final registry-status assertion. Now uses _set_started_grace, the same
    load-adaptive policy as the sibling must-not-be-flagged tests, with a
    grace-relative _run_spawn timeout so the same load that enlarges the
    grace cannot convert this into a subprocess.TimeoutExpired flake
    instead.
    """
    bin_dir = _make_bin_dir(tmp_path)
    _write_fake_claude(bin_dir, exit_code=0)
    _write_foreground_terminal(bin_dir, "xterm")
    env = _base_env(bin_dir, "xterm")
    grace = _set_started_grace(env)

    # scale_timeout=False (task 3599 amendment): `grace` is already
    # load-scaled (_set_started_grace delegates to _load_scaled_grace) --
    # see the identical double-scaling rationale on the two must-not-be-
    # flagged tests above.
    result = _run_spawn(env, tmp_path, timeout=grace + 20, scale_timeout=False)

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
# Model selection: CLAUDE_SPAWN_MODEL forwards to `claude --model <value>`
# ===========================================================================
# The trivial, first-class way to pin the spawned session's model: an env var
# baked into the payload string alongside $flags, so it reaches claude even
# under daemon-owned emulators that don't inherit the caller's environment.
# Unset = no --model on the argv (inherit the spawner's default), keeping every
# existing caller byte-identical.


def _write_fake_claude_capturing_argv(
    bin_dir: pathlib.Path, capture_file: pathlib.Path
) -> None:
    """Write a fake ``claude`` that captures its full argv (NUL-delimited) to
    *capture_file*, then exits 0.

    NUL-delimited rather than newline-delimited on purpose: the prompt is a
    single argv element that may itself contain newlines (the result-handback
    trailer spawn-claude.sh appends is multi-line), so a NUL separator keeps
    argv boundaries unambiguous. Read back with ``_read_argv`` below.

    Lets a test inspect the exact flag list spawn-claude.sh assembled into
    ``$inner`` -- e.g. whether a ``--model <value>`` pair was spliced in, and
    that the prompt still survives as the final argument.
    """
    p = bin_dir / "claude"
    p.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\0" "$@" > {capture_file!s}\n'
        "exit 0\n"
    )
    p.chmod(0o755)


def _read_argv(capture_file: pathlib.Path) -> list[str]:
    """Read a NUL-delimited argv capture (see _write_fake_claude_capturing_argv),
    dropping the trailing empty element left by the final NUL terminator.
    """
    raw = capture_file.read_bytes().decode()
    parts = raw.split("\0")
    if parts and parts[-1] == "":
        parts.pop()
    return parts


def test_spawn_model_env_adds_model_flag(tmp_path: pathlib.Path) -> None:
    """CLAUDE_SPAWN_MODEL=<value> must splice ``--model <value>`` into the
    claude invocation, ahead of the prompt.

    Uses skip_perms="false" (the default baked into _run_spawn) so $flags is
    otherwise empty and the argv is exactly ``--model <value> <prompt>``.
    """
    bin_dir = _make_bin_dir(tmp_path)
    capture_file = tmp_path / "captured_argv.txt"
    _write_fake_claude_capturing_argv(bin_dir, capture_file)
    _write_foreground_terminal(bin_dir, "xterm")
    env = _base_env(bin_dir, "xterm")
    env["CLAUDE_SPAWN_MODEL"] = "claude-fable-5"

    result = _run_spawn(env, tmp_path)
    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    argv = _read_argv(capture_file)
    assert "--model" in argv, f"expected --model in the claude argv, got: {argv!r}"
    assert argv[argv.index("--model") + 1] == "claude-fable-5", (
        f"expected the model value to follow --model, got: {argv!r}"
    )
    # The prompt (plus the result-handback trailer) must still be the final
    # argument, unaffected by the flag.
    assert argv[-1].startswith("test prompt"), (
        f"expected the prompt to remain the final argv entry, got: {argv!r}"
    )


def test_spawn_no_model_env_omits_model_flag(tmp_path: pathlib.Path) -> None:
    """With CLAUDE_SPAWN_MODEL unset, no ``--model`` may appear on the argv --
    the spawned session inherits the spawner's default model (existing-caller
    behavior is byte-identical).
    """
    bin_dir = _make_bin_dir(tmp_path)
    capture_file = tmp_path / "captured_argv.txt"
    _write_fake_claude_capturing_argv(bin_dir, capture_file)
    _write_foreground_terminal(bin_dir, "xterm")
    env = _base_env(bin_dir, "xterm")
    env.pop("CLAUDE_SPAWN_MODEL", None)

    result = _run_spawn(env, tmp_path)
    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    argv = _read_argv(capture_file)
    assert "--model" not in argv, (
        f"expected NO --model on the argv when CLAUDE_SPAWN_MODEL is unset, "
        f"got: {argv!r}"
    )
    # The prompt (plus its result-handback trailer) is the ONLY argv entry.
    assert len(argv) == 1 and argv[0].startswith("test prompt"), (
        f"expected the prompt as the only argv entry, got: {argv!r}"
    )


def test_spawn_model_env_precedes_raw_claude_args(tmp_path: pathlib.Path) -> None:
    """A raw ``--model`` in CLAUDE_SPAWN_CLAUDE_ARGS is spliced AFTER the
    dedicated CLAUDE_SPAWN_MODEL one, so it is the last ``--model`` on the argv
    -- and claude uses the last ``--model`` it sees, letting the escape hatch
    override the dedicated var when a caller sets both.
    """
    bin_dir = _make_bin_dir(tmp_path)
    capture_file = tmp_path / "captured_argv.txt"
    _write_fake_claude_capturing_argv(bin_dir, capture_file)
    _write_foreground_terminal(bin_dir, "xterm")
    env = _base_env(bin_dir, "xterm")
    env["CLAUDE_SPAWN_MODEL"] = "haiku"
    env["CLAUDE_SPAWN_CLAUDE_ARGS"] = "--model opus"

    result = _run_spawn(env, tmp_path)
    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    argv = _read_argv(capture_file)
    model_idxs = [i for i, tok in enumerate(argv) if tok == "--model"]
    assert len(model_idxs) == 2, f"expected two --model flags, got: {argv!r}"
    # Dedicated var first, raw passthrough (the winner) last.
    assert argv[model_idxs[0] + 1] == "haiku", f"got: {argv!r}"
    assert argv[model_idxs[1] + 1] == "opus", f"got: {argv!r}"
    assert model_idxs[0] < model_idxs[1], f"got: {argv!r}"


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
    """Write a fake ``claude`` that captures its own CLAUDE_SPAWN_SESSION_ID,
    CLAUDE_SPAWN_PARENT_ID, CLAUDE_SPAWN_WM_TITLE, and
    CLAUDE_CODE_FORCE_SESSION_PERSISTENCE env vars to *capture_file*, then
    exits 0.

    Modeled on ``_write_fake_claude_capturing_prompt`` -- lets a test inspect
    exactly what the Fleet Cockpit C1 identity exports (and, since task 2510,
    the C10-fix window-title marker export; and the unconditional
    transcript-persistence override) resolved to for the spawned child
    process.
    """
    p = bin_dir / "claude"
    p.write_text(
        "#!/usr/bin/env bash\n"
        "{\n"
        '  echo "SESSION=${CLAUDE_SPAWN_SESSION_ID:-}"\n'
        '  echo "PARENT=${CLAUDE_SPAWN_PARENT_ID:-}"\n'
        '  echo "CLAUDE_SPAWN_WM_TITLE=${CLAUDE_SPAWN_WM_TITLE:-}"\n'
        '  echo "PERSIST=${CLAUDE_CODE_FORCE_SESSION_PERSISTENCE:-}"\n'
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
    # Transcript-persistence override: exported unconditionally so a spawned
    # interactive session inheriting CLAUDE_CODE_CHILD_SESSION still saves its
    # ~/.claude/projects transcript (Claude Code >=2.1.208 suppression).
    assert captured.get("PERSIST") == "1", (
        f"expected the unconditional persistence override, got {captured!r}"
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
    # The persistence override is deliberately NOT gated on the registry: it
    # must still be exported on the fail-soft path, or a registry fault would
    # silently reintroduce transcript loss.
    assert captured.get("PERSIST") == "1", (
        f"persistence override must survive a registry fault, got {captured!r}"
    )


# ===========================================================================
# task-2510 (Fleet Cockpit C10 fix): CLAUDE_SPAWN_WM_TITLE export
# ===========================================================================
# The spawned session's own SessionStart hook (orchestrator rail C2,
# session_hooks._resolve_display) needs the EXACT window title this script
# handed the terminal emulator to resolve the live X11 window id via
# `wmctrl -l` -- this script keys its own LAUNCHING record on launcher_pid
# while the hook keys on session_id (a different record), so the marker must
# travel through the environment, mirroring CLAUDE_SPAWN_SESSION_ID/
# PARENT_ID/RESULT_FILE above.


def test_spawn_exports_wm_title_when_title_nonempty(tmp_path: pathlib.Path) -> None:
    """A non-empty title arg is exported into the spawned session as
    CLAUDE_SPAWN_WM_TITLE, byte-identical to the title handed to the
    emulator.
    """
    bin_dir = _make_bin_dir(tmp_path)
    capture_file = tmp_path / "captured_env.txt"
    _write_fake_claude_capturing_env(bin_dir, capture_file)
    _write_foreground_terminal(bin_dir, "xterm")
    env = _base_env(bin_dir, "xterm")

    result = _run_spawn(env, tmp_path, title="focus:df#2510 alpha")
    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    captured = _parse_captured_env(capture_file)
    assert captured.get("CLAUDE_SPAWN_WM_TITLE") == "focus:df#2510 alpha"


def test_spawn_omits_wm_title_export_when_title_empty(tmp_path: pathlib.Path) -> None:
    """Additive, no-op behavior: an empty title (the pre-task-2510 default,
    e.g. every existing caller in this suite) must not export
    CLAUDE_SPAWN_WM_TITLE at all -- it comes through empty in the captured
    env, exactly like the fail-soft PARENT/SESSION no-op cases above.
    """
    bin_dir = _make_bin_dir(tmp_path)
    capture_file = tmp_path / "captured_env.txt"
    _write_fake_claude_capturing_env(bin_dir, capture_file)
    _write_foreground_terminal(bin_dir, "xterm")
    env = _base_env(bin_dir, "xterm")

    result = _run_spawn(env, tmp_path, title="")
    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    captured = _parse_captured_env(capture_file)
    assert captured.get("CLAUDE_SPAWN_WM_TITLE", "") == "", (
        f"expected no wm-title export for an empty title, got {captured!r}"
    )


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

    # Use a distinctive, non-empty title so the wm_title assertion below
    # actually exercises the wiring instead of trivially matching an empty
    # default.
    title = "tmux-lane-display-test"
    result = _run_spawn(env, tmp_path, title=title)
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

    env = _hermetic_environ()
    env["PATH"] = str(bin_dir) + ":" + str(sys_bin)
    env["CLAUDE_SPAWN_BACKEND"] = "tmux"
    # Task 3486 audit: SPAWN_LAUNCH_GRACE_SECS="2" and timeout=10 below stay
    # FIXED on purpose -- rc==126 is load-INSENSITIVE (no tmux/emulator on
    # PATH fails the availability guard immediately; load-scaling would only
    # make a genuine regression take longer to report). Measured: 0.05s.
    # Task 3599 audit: _run_spawn's timeout now load-scales by default, so
    # the explicit scale_timeout=False below is what keeps this pin FIXED --
    # both audits are one decision, not a reversal.
    env["SPAWN_LAUNCH_GRACE_SECS"] = "2"
    env.pop("CLAUDE_TERMINAL_CMD", None)

    result = _run_spawn(env, tmp_path, timeout=10, scale_timeout=False)
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
    _wait_for_path_scaled(capture_file, 5)
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
# budget -- large enough that the snapshot-on-return assertion stays
# unambiguous even on a heavily-loaded CI runner.
#
# Two tests exercise two structurally different detach paths: the tmux lane
# below (test_sibling_mode_is_fire_and_forget) was ALREADY detached by the
# tmux server before this task, so sibling mode there only swaps
# resolve_detached for resolve_sibling; the foreground-emulator lane further
# down (test_sibling_mode_foreground_emulator_is_fire_and_forget) drives the
# NEW explicit `setsid <emu> ... </dev/null >/dev/null 2>&1 &` detach this
# task adds to the xterm/kitty/mac-terminal/custom branches -- the code most
# likely to regress and, absent that second test, entirely uncovered.


def _write_fake_claude_slow_with_markers(
    bin_dir: pathlib.Path, started: pathlib.Path, done: pathlib.Path, sleep_secs: float = 20,
) -> None:
    """Write a fake ``claude`` that touches *started* immediately, sleeps
    *sleep_secs*, then touches *done* and exits 0.

    Used to prove fire-and-forget: the spawner returns after the child is
    launched (started exists) but WITHOUT waiting for it to finish (done
    does not exist yet) -- a marker-presence check, not a timing tolerance.
    The default is deliberately generous (20s, comfortably inside
    ``_run_spawn``'s 30s default timeout): the assertion is taken
    immediately on return, not after the sleep, so a larger value costs
    nothing in wall-clock but makes the "spawn-claude.sh reliably returns
    well before the child finishes" premise robust under CI load.
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
    _write_fake_claude_slow_with_markers(bin_dir, started, done, sleep_secs=20)

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

    # Safe to lengthen under load: the done-marker snapshot carrying the
    # actual fire-and-forget assertion was already taken above, and in
    # fire-and-forget mode nothing rewrites the registry record after
    # launch, so the record.status == RUNNING assertion below has no
    # upper-bound dependency on how long this wait took (task 3486 audit).
    _wait_for_path_scaled(started, 5)

    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    record_path = _find_one_record(fleet_root)
    record = session_registry.SessionRecord.from_json(record_path.read_text())
    assert record.status == session_registry.Status.RUNNING, (
        f"expected a best-effort refresh to RUNNING, got {record.status}"
    )


def test_sibling_mode_foreground_emulator_is_fire_and_forget(tmp_path: pathlib.Path) -> None:
    """Fire-and-forget must also hold for a foreground terminal emulator
    (xterm) -- not just the tmux lane covered by
    test_sibling_mode_is_fire_and_forget above.

    The tmux lane was ALREADY detached by the tmux server before this task,
    so sibling mode there only swaps resolve_detached for resolve_sibling.
    The xterm branch is different: it needed a brand-new explicit detach
    (``setsid xterm ... </dev/null >/dev/null 2>&1 &``, replacing the plain
    foreground ``xterm ...`` call that resolve_foreground otherwise waits
    on) -- the code most likely to regress, and, without this test, never
    exercised by any sibling-mode test.

    Premise-free and doubles as a pipe-holding check, not just a timing
    check: ``_write_foreground_terminal``'s fake xterm does `exec bash -c
    "$inner"` -- if spawn-claude.sh ever dropped the `setsid` backgrounding
    or the `</dev/null >/dev/null 2>&1` stdio redirect, that exec'd process
    (and the slow fake claude beneath it) would inherit and hold open THIS
    test's own captured stdout/stderr pipe. subprocess.run's
    capture_output read would then block until the child closes those
    descriptors -- i.e. until it finishes sleeping and exits, ~sleep_secs
    later -- so the done-marker would already exist (or the call would
    approach its timeout) by the time control returned, not just fail a
    bare marker check.

    RED before step-4: sibling mode fell through to resolve_foreground's
    plain, attached ``xterm "${args[@]}"`` call, so _run_spawn blocked until
    the fake claude finished sleeping and touched done. GREEN after step-4's
    setsid + stdio-redirect + resolve_sibling for the xterm branch.
    """
    bin_dir = _make_bin_dir(tmp_path)
    started = tmp_path / "started"
    done = tmp_path / "done"
    _write_fake_claude_slow_with_markers(bin_dir, started, done, sleep_secs=20)
    _write_foreground_terminal(bin_dir, "xterm")

    env = _base_env(bin_dir, "xterm")
    env["CLAUDE_SPAWN_MODE"] = "sibling"

    result = _run_spawn(env, tmp_path)

    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    # Snapshot immediately -- must reflect the state at the moment
    # spawn-claude.sh returned, not after the started-marker poll below.
    # Reaching this line at all (rather than subprocess.run blocking on a
    # held-open pipe) is itself part of what's being asserted -- see the
    # docstring above.
    done_existed_at_return = done.exists()
    assert not done_existed_at_return, (
        "sibling mode via a foreground emulator (xterm) must return "
        "WITHOUT waiting for the child to finish -- the done-marker must "
        "not exist yet at the moment spawn-claude.sh returns, and the "
        "caller's stdout/stderr pipe must not be held open by an "
        "undetached child"
    )

    _wait_for_path_scaled(started, 5)

    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    record_path = _find_one_record(fleet_root)
    record = session_registry.SessionRecord.from_json(record_path.read_text())
    assert record.status == session_registry.Status.RUNNING, (
        f"expected a best-effort refresh to RUNNING, got {record.status}"
    )


# ===========================================================================
# task-4015: CLAUDE_SPAWN_* launch inputs are consumed, then REMOVED from the
# spawned session's own environment
# ===========================================================================
# spawn-claude.sh's inputs (CLAUDE_SPAWN_CLAUDE_ARGS / MODE / MODEL / BACKEND /
# TMUX_SESSION / ...) reach it by ordinary environment inheritance, and until
# task 4015 nothing removed them from the environment the payload then exec'd
# `claude` with. A spawned session therefore carried its OWN launch parameters
# forward, and its Bash tool handed them straight back to the next
# spawn-claude.sh it ran -- so an inherited value was re-served as apparent
# caller intent one level down.
#
# The 2026-08-11 incident: a fleet crash-recovery launch sets
# CLAUDE_SPAWN_CLAUDE_ARGS="--resume <that session's own id>". The recovered
# session inherited it, and its next /spawn -- meant to be a FRESH sibling --
# came up as a live resume of the spawner instead.
#
# Every test below poisons the namespace EXPLICITLY on top of _base_env's
# prefix scrub (see test_base_env_scrubs_every_claude_spawn_var), rather than
# depending on the runner's ambient state, so they fail on the merge worker's
# clean systemd unit too -- same rationale as the task-3062 tests above.


def _write_fake_claude_capturing_spawn_namespace(
    bin_dir: pathlib.Path, capture_file: pathlib.Path
) -> None:
    """Write a fake ``claude`` that captures the ENTIRE ``CLAUDE_SPAWN_*``
    namespace visible in its own environment to *capture_file*, then exits 0.

    Prefix-generic on purpose -- unlike ``_write_fake_claude_capturing_env``,
    which echoes four vars by name. The whole point of the task-4015 tests is
    that a var this file never names (a future launch knob) must not survive
    into the child either, so the capture itself cannot be an enumeration.

    The trailing ``exit 0`` is load-bearing: ``grep`` exits 1 when it matches
    nothing, and an EMPTY namespace is a PASSING outcome here -- without the
    explicit exit, the fully-scrubbed case would surface as a nonzero spawn
    exit code rather than as the pass it is.

    Emits ``KEY=value`` lines, so the capture reads back through the existing
    ``_parse_captured_env`` -- no new parser.
    """
    p = bin_dir / "claude"
    p.write_text(
        "#!/usr/bin/env bash\n"
        f"env | grep '^CLAUDE_SPAWN_' | sort > {capture_file!s}\n"
        "exit 0\n"
    )
    p.chmod(0o755)


def test_spawn_unsets_inherited_launch_inputs_from_child_env(
    tmp_path: pathlib.Path,
) -> None:
    """The four pure-input launch knobs must NOT be visible in the spawned
    session's own environment, however they reached this invocation.

    The poison values are chosen so neither reroutes the launcher away from
    the foreground-xterm path under test: ``MODE=child`` is the default, and
    a ``BACKEND`` of anything other than ``tmux`` falls through to normal
    emulator discovery. What is asserted is purely what the CHILD can see.

    The positive half matters just as much: the scrub must not be
    over-broad, so the child must still see its own freshly-computed
    CLAUDE_SPAWN_SESSION_ID and CLAUDE_SPAWN_RESULT_FILE -- the values THIS
    spawn computed for THAT child, not inherited ones.

    RED today: nothing in the payload unsets the namespace, so all four
    inherited vars reach the child by plain environment inheritance.
    """
    bin_dir = _make_bin_dir(tmp_path)
    capture_file = tmp_path / "captured_namespace.txt"
    _write_fake_claude_capturing_spawn_namespace(bin_dir, capture_file)
    _write_foreground_terminal(bin_dir, "xterm")

    env = _base_env(bin_dir, "xterm")
    env["CLAUDE_SPAWN_CLAUDE_ARGS"] = "--resume poisoned-parent-session"
    env["CLAUDE_SPAWN_MODE"] = "child"
    env["CLAUDE_SPAWN_MODEL"] = "haiku"
    env["CLAUDE_SPAWN_BACKEND"] = "leak-not-tmux"

    result = _run_spawn(env, tmp_path)
    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    captured = _parse_captured_env(capture_file)
    for var in (
        "CLAUDE_SPAWN_CLAUDE_ARGS",
        "CLAUDE_SPAWN_MODE",
        "CLAUDE_SPAWN_MODEL",
        "CLAUDE_SPAWN_BACKEND",
    ):
        assert var not in captured, (
            f"{var} is a per-launch INPUT: consumed by this invocation, then "
            f"removed from the child's environment so it cannot be re-served "
            f"to a grandchild. Child saw: {captured!r}"
        )

    # Not over-broad: this child's OWN computed values must survive.
    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    record_path = _find_one_record(fleet_root)
    record = session_registry.SessionRecord.from_json(record_path.read_text())
    assert captured.get("CLAUDE_SPAWN_SESSION_ID") == record.session_slug, (
        f"the child must still see its own new session slug "
        f"{record.session_slug!r}, got {captured!r}"
    )
    assert captured.get("CLAUDE_SPAWN_RESULT_FILE") == record.result_file, (
        f"the child must still see the result file allocated for THIS spawn "
        f"({record.result_file!r}), got {captured!r}"
    )


def test_spawn_unset_is_prefix_generic_not_an_enumerated_list(
    tmp_path: pathlib.Path,
) -> None:
    """The scrub must cover the whole ``CLAUDE_SPAWN_*`` namespace, not a
    hand-maintained list of the names known today.

    CLAUDE_SPAWN_FUTURE_KNOB exists nowhere in the codebase, which is the
    entire point: it is the only assertion that can distinguish a
    prefix-generic ``${!CLAUDE_SPAWN_@}`` sweep from a fourth named
    enumeration that a future launch knob would silently fall outside of.
    Same idiom this file already applies to its OWN scrub in
    test_base_env_scrubs_every_claude_spawn_var -- kept deliberately
    identical so the harness-side and script-side scrubs read the same.

    A real input var is poisoned alongside it so a regression that drops the
    sweep entirely fails here too, not only in the test above.
    """
    bin_dir = _make_bin_dir(tmp_path)
    capture_file = tmp_path / "captured_namespace.txt"
    _write_fake_claude_capturing_spawn_namespace(bin_dir, capture_file)
    _write_foreground_terminal(bin_dir, "xterm")

    env = _base_env(bin_dir, "xterm")
    env["CLAUDE_SPAWN_FUTURE_KNOB"] = "leak"
    env["CLAUDE_SPAWN_CLAUDE_ARGS"] = "--resume poisoned-parent-session"

    result = _run_spawn(env, tmp_path)
    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    captured = _parse_captured_env(capture_file)
    assert "CLAUDE_SPAWN_FUTURE_KNOB" not in captured, (
        "the unset must be prefix-generic (${!CLAUDE_SPAWN_@}), so a var "
        "that exists nowhere in the codebase is stripped too -- an "
        "enumerated name list would leak every knob added after it was "
        f"written. Child saw: {captured!r}"
    )
    assert "CLAUDE_SPAWN_CLAUDE_ARGS" not in captured, (
        f"the known input var must be stripped as well, got {captured!r}"
    )


def test_spawn_registry_fault_unsets_parent_result_file_rather_than_inheriting_it(
    tmp_path: pathlib.Path,
) -> None:
    """On a session-registry fault the child must see NO
    CLAUDE_SPAWN_RESULT_FILE at all -- never the SPAWNER's inherited one.

    This case gets its own test rather than an assertion folded into the
    prefix-genericity one above because it is the only one that silently
    corrupts a PARENT session's outcome record: with the registry faulted,
    SESSION_RECORD_DIR is empty, so ``result_export`` is the empty string
    and nothing is exported -- and before task 4015 the parent's inherited
    value simply survived into the child, which would then dutifully write
    its own outcome over its spawner's result.md.

    Uses the same deterministic fault injection as
    test_spawn_fail_soft_skips_result_handback_when_registry_faults: a
    CLAUDE_FLEET_ROOT nested under a pre-existing regular file, so the
    registry's mkdir raises NotADirectoryError rather than depending on
    permission semantics that vary across CI users and containers.

    RED if the unset were nested inside the ``if [ -n
    "$CLAUDE_SPAWN_RESULT_FILE" ]`` / ``if [ -n "$SESSION_RECORD_DIR" ]``
    guards, whose bodies are skipped on exactly this path: the skip path
    must UNSET, not merely decline to set. Fail-soft must also stay
    fail-soft -- the exit-code contract is asserted unchanged.
    """
    bin_dir = _make_bin_dir(tmp_path)
    capture_file = tmp_path / "captured_namespace.txt"
    _write_fake_claude_capturing_spawn_namespace(bin_dir, capture_file)
    _write_foreground_terminal(bin_dir, "xterm")

    env = _base_env(bin_dir, "xterm")
    parent_result = "/parent/session/result.md"
    env["CLAUDE_SPAWN_RESULT_FILE"] = parent_result

    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a regular file, not a directory\n")
    env["CLAUDE_FLEET_ROOT"] = str(blocker / "fleet")

    result = _run_spawn(env, tmp_path)
    assert result.returncode == 0, (
        f"a registry fault must never change the exit-code contract: "
        f"expected 0, got {result.returncode}\nstderr: {result.stderr.decode()}"
    )

    captured = _parse_captured_env(capture_file)
    assert captured.get("CLAUDE_SPAWN_RESULT_FILE") != parent_result, (
        "a child must never inherit its SPAWNER's result file -- it would "
        f"overwrite the parent's outcome record. Child saw: {captured!r}"
    )
    assert "CLAUDE_SPAWN_RESULT_FILE" not in captured, (
        "with no record dir of its own the child must see NO result file at "
        f"all, not a stale inherited one. Child saw: {captured!r}"
    )


def _write_fake_claude_dumping_full_env(
    bin_dir: pathlib.Path,
    capture_file: pathlib.Path,
    argv_file: pathlib.Path | None = None,
) -> None:
    """Write a fake ``claude`` that dumps its ENTIRE environment (NUL-delimited,
    via ``env -0``) to *capture_file* -- and, when *argv_file* is given, its
    own argv alongside -- then exits 0.

    The full environment, not just the CLAUDE_SPAWN_* slice: this is exactly
    what the spawned session's Bash tool would hand to the next
    spawn-claude.sh it runs, so it can be fed straight back in as a second
    invocation's environment (see the two-level test below).

    NUL-delimited for the same reason ``_write_fake_claude_capturing_argv``
    is: an environment value may itself contain newlines, so a newline
    separator could not delimit entries unambiguously. Read back with
    ``_read_env0`` / ``_read_argv``.
    """
    p = bin_dir / "claude"
    body = f"env -0 > {capture_file!s}\n"
    if argv_file is not None:
        body += f'printf "%s\\0" "$@" > {argv_file!s}\n'
    p.write_text("#!/usr/bin/env bash\n" + body + "exit 0\n")
    p.chmod(0o755)


def _read_env0(capture_file: pathlib.Path) -> dict[str, str]:
    """Read a NUL-delimited ``env -0`` capture into a dict.

    Partitions each entry on its FIRST ``=`` (a value may contain further
    ``=`` characters) and drops the trailing empty element left by the final
    NUL terminator -- the env-shaped counterpart of ``_read_argv``.
    """
    parsed: dict[str, str] = {}
    for entry in capture_file.read_bytes().decode().split("\0"):
        if not entry:
            continue
        key, _, value = entry.partition("=")
        parsed[key] = value
    return parsed


def test_spawned_session_cannot_reserve_its_own_launch_args_to_a_grandchild(
    tmp_path: pathlib.Path,
) -> None:
    """The 2026-08-11 incident, reproduced end-to-end across TWO real spawns.

    Level 1 is a fleet crash-recovery launch:
    CLAUDE_SPAWN_CLAUDE_ARGS="--resume <a session id>". That is a DELIBERATE
    input and the direct child must honour it -- asserted below on the
    level-1 child's own argv. Requirement (4): inputs are consumed before
    the unset, and from inside the script a deliberate command-prefix
    assignment is indistinguishable from an inherited one, so the direct
    child's argv is NOT where this is fixed. Do not "fix" that path -- it
    would delete the documented CLAUDE_SPAWN_CLAUDE_ARGS passthrough and
    break test_spawn_model_env_precedes_raw_claude_args.

    Level 2 is where the fix bites. The recovered session's own environment
    -- captured verbatim at level 1, exactly what its Bash tool would hand
    to the next spawn-claude.sh it runs -- is fed back in as a second
    invocation's environment. That grandchild is meant to be FRESH, and
    before task 4015 it came up as a live resume of its spawner because the
    inherited CLAUDE_SPAWN_CLAUDE_ARGS was re-served as apparent caller
    intent.

    Two real script invocations rather than a fake claude that spawns
    recursively: the level-1 run has fully completed before level 2 starts,
    so the fake ``claude`` can simply be rewritten in between and nothing
    races.
    """
    bin_dir = _make_bin_dir(tmp_path)
    child_env_file = tmp_path / "child_env.bin"
    child_argv_file = tmp_path / "child_argv.bin"
    _write_fake_claude_dumping_full_env(bin_dir, child_env_file, child_argv_file)
    _write_foreground_terminal(bin_dir, "xterm")

    env = _base_env(bin_dir, "xterm")
    env["CLAUDE_SPAWN_CLAUDE_ARGS"] = "--resume poisoned-parent-session"

    # --- level 1: the crash-recovery launch -------------------------------
    result = _run_spawn(env, tmp_path)
    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    child_argv = _read_argv(child_argv_file)
    assert "--resume" in child_argv and "poisoned-parent-session" in child_argv, (
        f"the DIRECT child must still receive args deliberately passed on "
        f"this invocation (requirement 4) -- got: {child_argv!r}"
    )

    child_env = _read_env0(child_env_file)
    assert "CLAUDE_SPAWN_CLAUDE_ARGS" not in child_env, (
        f"the recovered session must not carry its own launch args forward "
        f"in its environment, got: "
        f"{ {k: v for k, v in child_env.items() if k.startswith('CLAUDE_SPAWN_')} !r}"
    )

    # --- level 2: what that session's next /spawn would actually run ------
    grandchild_argv_file = tmp_path / "grandchild_argv.bin"
    _write_fake_claude_capturing_argv(bin_dir, grandchild_argv_file)

    result2 = _run_spawn(child_env, tmp_path)
    assert result2.returncode == 0, f"stderr: {result2.stderr.decode()}"

    grandchild_argv = _read_argv(grandchild_argv_file)
    assert "--resume" not in grandchild_argv, (
        f"a session spawned from within the recovered session must be FRESH, "
        f"not a resume of its spawner -- got: {grandchild_argv!r}"
    )
    assert not any("poisoned-parent-session" in tok for tok in grandchild_argv), (
        f"the spawner's own resume target must not reach the grandchild's "
        f"argv anywhere, got: {grandchild_argv!r}"
    )


def test_sanitization_preserves_launcher_stamped_record_identity(
    tmp_path: pathlib.Path,
) -> None:
    """The not-over-broad lock, in the two-way style of
    test_sibling_parentage_two_way_into_hook_record: identity must survive
    on the session-registry RECORD even though it no longer travels in the
    child's environment.

    CLAUDE_SPAWN_ROLE / TASK_ID are launcher-side inputs. They are stamped
    by the ``python3 ... launching`` write, which runs in its own subprocess
    BEFORE and OUTSIDE $inner -- so a payload-level unset cannot reach it.
    The child itself has no consumer for them: session_hooks.run_session_start
    reads identity from the environment only in its ``except
    FileNotFoundError`` branch, and a spawn-claude.sh child provably has a
    record at its slug already (CLAUDE_SPAWN_SESSION_ID is exported only
    when SESSION_RECORD_DIR is non-empty), so that branch is unreachable
    here.

    Stripping them is in fact corrective: a leaked inherited identity used
    to resolve as ``implementer:<project>#<id>`` on an unrelated session --
    the same leak orchestrator/tests/test_session_hooks.py::_clear_claude_
    spawn_env exists to defend against.
    """
    bin_dir = _make_bin_dir(tmp_path)
    capture_file = tmp_path / "captured_namespace.txt"
    _write_fake_claude_capturing_spawn_namespace(bin_dir, capture_file)
    _write_foreground_terminal(bin_dir, "xterm")

    env = _base_env(bin_dir, "xterm")
    env["CLAUDE_SPAWN_ROLE"] = "implementer"
    env["CLAUDE_SPAWN_TASK_ID"] = "9999"

    result = _run_spawn(env, tmp_path)
    assert result.returncode == 0, f"stderr: {result.stderr.decode()}"

    fleet_root = pathlib.Path(env["CLAUDE_FLEET_ROOT"])
    record_path = _find_one_record(fleet_root)
    record = session_registry.SessionRecord.from_json(record_path.read_text())
    assert record.role == "implementer", (
        f"the launching write runs before/outside $inner, so the payload's "
        f"unset must not disturb the stamped role, got {record.role!r}"
    )
    assert record.task_id == "9999", (
        f"same for the stamped task_id, got {record.task_id!r}"
    )

    captured = _parse_captured_env(capture_file)
    assert "CLAUDE_SPAWN_ROLE" not in captured, (
        f"identity belongs on the record, not in the child's environment "
        f"where it would be re-served to the next spawn: {captured!r}"
    )
    assert "CLAUDE_SPAWN_TASK_ID" not in captured, (
        f"same for the task id: {captured!r}"
    )
