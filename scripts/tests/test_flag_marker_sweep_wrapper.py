"""Tests for scripts/fused-memory-flag-marker-sweep.sh -- the committed
nightly DRAIN action for stage1_flag_marker dead-weight records (task 2693,
follow-up to task 2596's previously-unwired sweep).

Drives the wrapper via subprocess with the FLAG_MARKER_SWEEP_CMD test seam
pointed at a fake recorder executable (records its argv to a JSON state
file) -- mirrors test_install_trickle_timer.py's fake-systemctl harness.
Real uv/fused_memory/live stores are never touched.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

WRAPPER = Path(__file__).parent.parent / "fused-memory-flag-marker-sweep.sh"


# ---------------------------------------------------------------------------
# Fake sweep-command recorder (marker-file + configurable exit code)
# ---------------------------------------------------------------------------

_FAKE_RECORDER_SRC = '''#!/usr/bin/env python3
"""Fake sweep-invocation recorder for testing
fused-memory-flag-marker-sweep.sh. Records argv[1:] (the sweep script path
and its flags) and a snapshot of os.environ into a JSON state file at
$FAKE_SWEEP_STATE, then exits with $FAKE_SWEEP_EXIT_CODE (default 0).

One executable serves BOTH call kinds the wrapper makes (task 2917 EDIT 1).
When `--list-known-projects` is in argv it plays the RESOLUTION call: it
prints each whitespace-separated entry of $FAKE_SWEEP_KNOWN_PROJECTS on its
own line and exits $FAKE_SWEEP_LIST_EXIT_CODE (defaulting to 0 when that var
named at least one project, else 1 -- mirroring the real script, which exits
non-zero when the registry resolves empty). Otherwise it plays a SWEEP call
and exits $FAKE_SWEEP_EXIT_CODE.

The list-exit and sweep-exit seams are deliberately SEPARATE vars: the
continue-after-failure test sets a non-zero SWEEP exit and must not
simultaneously break project resolution.
"""
import json
import os
import sys

state_path = os.environ["FAKE_SWEEP_STATE"]
with open(state_path) as f:
    state = json.load(f)
state.setdefault("calls", []).append(sys.argv[1:])
state.setdefault("envs", []).append(dict(os.environ))
with open(state_path, "w") as f:
    json.dump(state, f)

if "--list-known-projects" in sys.argv[1:]:
    known = os.environ.get("FAKE_SWEEP_KNOWN_PROJECTS", "").split()
    for project_id in known:
        print(project_id)
    sys.exit(int(
        os.environ.get("FAKE_SWEEP_LIST_EXIT_CODE", "0" if known else "1")
    ))

sys.exit(int(os.environ.get("FAKE_SWEEP_EXIT_CODE", "0")))
'''


def _fake_recorder(tmp_path):
    """Write an executable fake sweep-command recorder into <tmp_path>/bin/
    and its backing JSON state file. Returns (bin_dir, state_path)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "fake-sweep-recorder"
    fake.write_text(_FAKE_RECORDER_SRC)
    fake.chmod(0o755)

    state_path = tmp_path / "sweep_state.json"
    state_path.write_text(json.dumps({"calls": []}))
    return bin_dir, state_path


def _recorded_calls(state_path):
    return json.loads(state_path.read_text())["calls"]


def _recorded_envs(state_path):
    return json.loads(state_path.read_text())["envs"]


# ---------------------------------------------------------------------------
# Script driver
# ---------------------------------------------------------------------------

def _run_wrapper(
    tmp_path, *, exit_code=0, extra_env=None, dotenv_contents=None,
    project_ids="dark_factory",
):
    """Run fused-memory-flag-marker-sweep.sh with FLAG_MARKER_SWEEP_CMD
    pointed at the fake recorder and REPO pointed at a tmp dir with no
    `.env` (so the wrapper's `source .env` is a no-op under test) -- unless
    `dotenv_contents` is given, in which case a `$REPO/.env` file with that
    content is written first so the sourcing branch itself is exercised.

    `project_ids` seeds FLAG_MARKER_SWEEP_PROJECT_IDS, the explicit override
    for the per-project sweep loop (task 2917 EDIT 1). It defaults to the
    single `dark_factory` so every pre-existing single-call test keeps its
    one-call contract; pass None to leave the var UNSET (and scrubbed from
    the inherited environment) and exercise the wrapper's own registry
    resolution instead."""
    bin_dir, state_path = _fake_recorder(tmp_path)

    fake_repo = tmp_path / "fake-repo"
    fake_repo.mkdir(exist_ok=True)
    if dotenv_contents is not None:
        (fake_repo / ".env").write_text(dotenv_contents)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["FLAG_MARKER_SWEEP_CMD"] = "fake-sweep-recorder"
    env["FAKE_SWEEP_STATE"] = str(state_path)
    env["FAKE_SWEEP_EXIT_CODE"] = str(exit_code)
    env["REPO"] = str(fake_repo)
    if project_ids is None:
        env.pop("FLAG_MARKER_SWEEP_PROJECT_IDS", None)
    else:
        env["FLAG_MARKER_SWEEP_PROJECT_IDS"] = project_ids
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        env=env, capture_output=True, text=True, timeout=30,
    )
    return result, state_path


# ---------------------------------------------------------------------------
# step-1: RED -- fused-memory-flag-marker-sweep.sh
# ---------------------------------------------------------------------------

def test_wrapper_is_executable():
    assert os.access(WRAPPER, os.X_OK), (
        f"Expected {WRAPPER} to be executable (os.X_OK); it is not. "
        f"Run: chmod +x {WRAPPER}"
    )


def test_wrapper_invokes_sweep_with_apply_and_terminal_drain(tmp_path):
    result, state_path = _run_wrapper(tmp_path)

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    calls = _recorded_calls(state_path)
    assert len(calls) == 1, f"calls={calls!r}"
    argv = calls[0]
    assert any(
        a.endswith("fused-memory/scripts/sweep_orphan_flag_markers.py") for a in argv
    ), f"Expected the sweep script path in argv={argv!r}"
    assert "--apply" in argv, f"argv={argv!r}"
    assert "--terminal-drain" in argv, f"argv={argv!r}"
    # Task 2917 EDIT 1: every sweep invocation is now explicitly scoped to a
    # project_id rather than riding the sweep parser's own default.
    assert "--project-id" in argv, f"argv={argv!r}"
    assert argv[argv.index("--project-id") + 1] == "dark_factory", f"argv={argv!r}"


def test_wrapper_propagates_nonzero_exit(tmp_path):
    result, _state_path = _run_wrapper(tmp_path, exit_code=7)

    assert result.returncode != 0, (
        f"Expected a non-zero wrapper exit when the sweep command fails; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_wrapper_sources_dotenv_and_propagates_to_sweep(tmp_path):
    """The wrapper's stated core purpose (module docstring: it 'must run
    under the SERVICE env, not a bare shell, or the census silently
    narrows') is to source $REPO/.env and export CONFIG_PATH/PROJECT_ROOT/
    FALKORDB_URI so the sweep runs with the right service environment.
    Every other test points REPO at a directory with no .env, so the
    sourcing branch and export propagation are otherwise never exercised.
    This writes a real `.env` and confirms both an explicitly-exported var
    (FALKORDB_URI, overriding the wrapper's own default) and a plain .env
    var the wrapper never names itself (propagated only via `set -a`) reach
    the sweep process -- a regression that broke sourcing or dropped the
    `set -a` export would leave both at their fallback/absent values while
    every other test in this file stayed green."""
    result, state_path = _run_wrapper(
        tmp_path,
        dotenv_contents=(
            "FALKORDB_URI=redis://test-env-host:1234\n"
            "FLAG_MARKER_SWEEP_TEST_SENTINEL=from-dot-env\n"
        ),
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    envs = _recorded_envs(state_path)
    assert len(envs) == 1, f"envs={envs!r}"
    recorded_env = envs[0]
    assert recorded_env.get("FALKORDB_URI") == "redis://test-env-host:1234", (
        f"Expected the sourced .env's FALKORDB_URI to reach the sweep "
        f"process (overriding the wrapper's own default); "
        f"recorded_env={recorded_env!r}"
    )
    assert recorded_env.get("FLAG_MARKER_SWEEP_TEST_SENTINEL") == "from-dot-env", (
        f"Expected a plain .env var the wrapper never names itself to "
        f"still reach the sweep process via `set -a`; "
        f"recorded_env={recorded_env!r}"
    )


# ---------------------------------------------------------------------------
# Fake `uv` shim (pins the default FLAG_MARKER_SWEEP_CMD prefix)
# ---------------------------------------------------------------------------

_FAKE_UV_SRC = '''#!/usr/bin/env python3
"""Fake `uv` shim for pinning fused-memory-flag-marker-sweep.sh's default
FLAG_MARKER_SWEEP_CMD prefix (`uv run --frozen --project "$FM" python`).
Records argv[1:] into a JSON state file at $FAKE_SWEEP_STATE and exits 0.
Never invokes real uv / fused_memory / live stores.
"""
import json
import os
import sys

state_path = os.environ["FAKE_SWEEP_STATE"]
with open(state_path) as f:
    state = json.load(f)
state.setdefault("calls", []).append(sys.argv[1:])
with open(state_path, "w") as f:
    json.dump(state, f)

sys.exit(0)
'''


def _fake_uv(tmp_path):
    """Write an executable fake `uv` into <tmp_path>/uv-bin/ and its backing
    JSON state file. Returns (bin_dir, state_path)."""
    bin_dir = tmp_path / "uv-bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "uv"
    fake.write_text(_FAKE_UV_SRC)
    fake.chmod(0o755)

    state_path = tmp_path / "uv_state.json"
    state_path.write_text(json.dumps({"calls": []}))
    return bin_dir, state_path


def test_wrapper_default_prefix_invokes_uv_run_frozen_project(tmp_path):
    """Pins the default FLAG_MARKER_SWEEP_CMD prefix -- `uv run --frozen
    --project "$FM" python`, an unquoted array expansion with an embedded
    quoted "$FM" -- which every other test in this file bypasses by
    overriding FLAG_MARKER_SWEEP_CMD with a bare recorder. A future edit
    that broke that quoting (e.g. dropping the inner quotes around $FM)
    would pass every other test while silently mis-invoking (or
    word-splitting) the real `uv` in production. REPO is deliberately given
    a space in its path (`FM` inherits it) so a dropped-quote regression
    actually manifests here as a word-split argv rather than passing by
    coincidence -- the same case the reviewer manually verified by hand."""
    uv_bin_dir, state_path = _fake_uv(tmp_path)

    fake_repo = tmp_path / "fake repo with spaces"
    fake_repo.mkdir(exist_ok=True)
    fake_fm = fake_repo / "fused-memory"

    env = dict(os.environ)
    env["PATH"] = f"{uv_bin_dir}{os.pathsep}{env['PATH']}"
    env["FAKE_SWEEP_STATE"] = str(state_path)
    env["REPO"] = str(fake_repo)
    env.pop("FLAG_MARKER_SWEEP_CMD", None)
    # Pin the project list explicitly so this stays a test of the INTERPRETER
    # PREFIX and nothing else: with the override unset the wrapper would make
    # an extra `--list-known-projects` resolution call through this same fake
    # `uv`, and the exactly-one-call assertion below would be measuring
    # registry resolution rather than argv quoting.
    env["FLAG_MARKER_SWEEP_PROJECT_IDS"] = "dark_factory"

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    calls = _recorded_calls(state_path)
    assert len(calls) == 1, f"calls={calls!r}"
    assert calls[0] == [
        "run", "--frozen", "--project", str(fake_fm), "python",
        str(fake_fm / "scripts" / "sweep_orphan_flag_markers.py"),
        "--apply", "--terminal-drain", "--project-id", "dark_factory",
    ], f"argv={calls[0]!r}"


def test_wrapper_resolves_uv_by_absolute_path_when_path_omits_it(tmp_path):
    """Task 2917 EDIT 3. Pins that the wrapper resolves `uv` to an ABSOLUTE
    path rather than trusting PATH.

    OBSERVED production failure (journalctl --user -u
    fused-memory-flag-marker-sweep.service):

        Aug 18 09:02:44 ... fused-memory-flag-marker-sweep.sh[65377]:
            .../fused-memory-flag-marker-sweep.sh: line 46: exec: uv: not found
        Aug 18 09:02:44 ... fused-memory-flag-marker-sweep.service:
            Main process exited, code=exited, status=127/n/a

    That line is immediately preceded by a `-- Boot ... --` marker and the
    next normal timer firing succeeded, so the failure is specific to the
    unit's `Persistent=true` BOOT CATCH-UP run, which fires before the login
    session pushes the user PATH into the systemd user manager. `uv` lives
    in /home/leo/.local/bin, which is absent from that minimal boot PATH.

    Simulated here by scrubbing PATH down to /usr/bin:/bin (no `uv`
    anywhere on it) while pointing the wrapper's UV_BIN override at the fake
    uv recorder. A wrapper that still relies on a bare `uv` word exits 127
    without ever invoking it."""
    uv_bin_dir, state_path = _fake_uv(tmp_path)

    fake_repo = tmp_path / "fake-repo"
    fake_repo.mkdir(exist_ok=True)

    env = dict(os.environ)
    env["PATH"] = "/usr/bin:/bin"
    env["FAKE_SWEEP_STATE"] = str(state_path)
    env["REPO"] = str(fake_repo)
    env["UV_BIN"] = str(uv_bin_dir / "uv")
    env.pop("FLAG_MARKER_SWEEP_CMD", None)

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        env=env, capture_output=True, text=True, timeout=30,
    )

    assert result.returncode == 0, (
        f"Expected the wrapper to resolve uv via UV_BIN under a boot-catch-up "
        f"PATH that omits it (returncode 127 == the OBSERVED regression); "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    calls = _recorded_calls(state_path)
    assert len(calls) >= 1, (
        f"Expected the absolute-path uv to actually be invoked; calls={calls!r} "
        f"stderr={result.stderr!r}"
    )


def test_wrapper_fails_loud_when_uv_cannot_be_resolved(tmp_path):
    """A missing interpreter must be DIAGNOSABLE from the journal, not a bare
    shell 127 (loud-over-silent-degradation).

    Scrubs PATH to /usr/bin:/bin, leaves UV_BIN unset, and repoints HOME at a
    tmp dir so the $HOME/.local/bin/uv fallback misses too. The wrapper must
    exit non-zero AND print its own ERROR:-prefixed line naming `uv`."""
    if os.path.exists("/usr/local/bin/uv"):
        pytest.skip(
            "/usr/local/bin/uv exists on this host, so the wrapper's last-resort "
            "fallback resolves and the unresolvable-uv path cannot be exercised"
        )

    fake_repo = tmp_path / "fake-repo"
    fake_repo.mkdir(exist_ok=True)
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir(exist_ok=True)

    env = dict(os.environ)
    env["PATH"] = "/usr/bin:/bin"
    env["REPO"] = str(fake_repo)
    env["HOME"] = str(fake_home)
    env.pop("UV_BIN", None)
    env.pop("FLAG_MARKER_SWEEP_CMD", None)

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        env=env, capture_output=True, text=True, timeout=30,
    )

    assert result.returncode != 0, (
        f"Expected a non-zero exit when uv cannot be resolved; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "ERROR:" in result.stderr, (
        f"Expected the wrapper's own ERROR:-prefixed diagnostic rather than a "
        f"bare shell 127; stderr={result.stderr!r}"
    )
    assert "uv" in result.stderr, (
        f"Expected the diagnostic to name `uv`; stderr={result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# step-9: RED -- the per-project sweep loop (task 2917 EDIT 1)
# ---------------------------------------------------------------------------

def _project_ids_of(calls):
    """Extract the --project-id value from each recorded SWEEP call (the
    resolution call, which carries --list-known-projects, is skipped)."""
    out = []
    for argv in calls:
        if "--list-known-projects" in argv:
            continue
        assert "--project-id" in argv, f"sweep call without --project-id: {argv!r}"
        out.append(argv[argv.index("--project-id") + 1])
    return out


def test_wrapper_sweeps_every_configured_project_id(tmp_path):
    """The defect this closes: the wrapper used to `exec` the sweep exactly
    ONCE, with no --project-id, so it rode the sweep parser's own
    `dark_factory` default while the per-project census output read as if the
    whole fleet had been drained. Every registered project must get its own
    invocation."""
    result, state_path = _run_wrapper(tmp_path, project_ids="dark_factory reify")

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    calls = _recorded_calls(state_path)
    swept = _project_ids_of(calls)
    assert sorted(swept) == ["dark_factory", "reify"], (
        f"Expected exactly one sweep per configured project_id; "
        f"swept={swept!r} calls={calls!r}"
    )
    for argv in calls:
        if "--list-known-projects" in argv:
            continue
        assert "--apply" in argv and "--terminal-drain" in argv, (
            f"Every per-project sweep keeps the nightly drain argv; argv={argv!r}"
        )

    # Census honesty: the journal must name each project actually swept, so an
    # operator can read coverage off the log rather than inferring it.
    combined = result.stdout + result.stderr
    for project_id in ("dark_factory", "reify"):
        assert f"project_id={project_id}" in combined, (
            f"Expected a per-project progress line naming {project_id!r}; "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )


def test_wrapper_continues_sweeping_after_one_project_fails(tmp_path):
    """A failing project must not silently truncate the fleet sweep. The
    wrapper drops `exec` and does NOT `set -e` out of the loop: every project
    is still attempted, the failure is named on stderr, and the overall exit
    is non-zero so a partial nightly drain is loud rather than swallowed."""
    result, state_path = _run_wrapper(
        tmp_path, exit_code=7, project_ids="dark_factory reify",
    )

    calls = _recorded_calls(state_path)
    swept = _project_ids_of(calls)
    assert sorted(swept) == ["dark_factory", "reify"], (
        f"Expected BOTH projects to be attempted despite the first failing "
        f"(no exec, no short-circuit); swept={swept!r} calls={calls!r}"
    )
    assert result.returncode != 0, (
        f"Expected a non-zero overall exit when a project's sweep failed; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "ERROR:" in result.stderr, (
        f"Expected an ERROR:-prefixed stderr line for the failing project; "
        f"stderr={result.stderr!r}"
    )
    assert "dark_factory" in result.stderr, (
        f"Expected the failing project_id to be named on stderr; "
        f"stderr={result.stderr!r}"
    )
