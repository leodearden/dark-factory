"""Tests for scripts/fused-memory-flag-marker-check.sh -- the read-only
CHECK counterpart to fused-memory-flag-marker-sweep.sh (task 2596's
--check/backlog_verdict predicate mode).

Drives the wrapper via subprocess with the FLAG_MARKER_SWEEP_CMD test seam
pointed at a fake recorder executable (records its argv to a JSON state
file) -- mirrors test_flag_marker_sweep_wrapper.py's fake-recorder harness.
Real uv/fused_memory/live stores are never touched.

Task 4591: this wrapper had the same bare-`uv`-from-PATH fragility that the
sibling sweep wrapper actually tripped via its systemd unit's
Persistent=true boot catch-up run (OBSERVED 2026-08-18 09:02:44 -- see
fused-memory-flag-marker-sweep.sh's own header). This wrapper has no systemd
unit or before_done predicate wired to it today (the watch gate is retired,
task 3923), so the defect was LATENT rather than observed here directly --
these tests pin the same absolute-uv-resolution fix so a future re-wiring
doesn't reinherit it.

Task 3498 ratified this wrapper's surviving role (an ad-hoc census
instrument) and made its read-only claim a property rather than prose: the
exec line passes caller args straight through and the sweep accepts
``--check --apply`` together, so the wrapper now refuses every spelling the
sweep's own parser reads as ``apply=True``. The refusal tests at the end pin
that, with the refusal set derived from the sweep's parser rather than
restated. Task 3498 also pins argument ORDER and verbatim exit-code relay,
which the baseline tests above check only loosely.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

WRAPPER = Path(__file__).parent.parent / "fused-memory-flag-marker-check.sh"

# Resolved HERE, in the parent, because the uv-resolution tests below hand the
# child an empty PATH -- `subprocess.run(["bash", ...])` looks argv[0] up in the
# CHILD's env, so a bare "bash" is itself unfindable under a scrubbed PATH
# (FileNotFoundError, nothing to do with the wrapper). An absolute interpreter
# keeps the scrub aimed at `uv` alone, which is what these tests are about.
BASH = shutil.which("bash") or "/bin/bash"


# ---------------------------------------------------------------------------
# Fake check-command recorder (marker-file + configurable exit code)
# ---------------------------------------------------------------------------

_FAKE_RECORDER_SRC = '''#!/usr/bin/env python3
"""Fake check-invocation recorder for testing
fused-memory-flag-marker-check.sh. Records argv[1:] and a snapshot of
os.environ into a JSON state file at $FAKE_CHECK_STATE, then exits with
$FAKE_CHECK_EXIT_CODE (default 0).
"""
import json
import os
import sys

state_path = os.environ["FAKE_CHECK_STATE"]
with open(state_path) as f:
    state = json.load(f)
state.setdefault("calls", []).append(sys.argv[1:])
state.setdefault("envs", []).append(dict(os.environ))
with open(state_path, "w") as f:
    json.dump(state, f)

sys.exit(int(os.environ.get("FAKE_CHECK_EXIT_CODE", "0")))
'''


def _fake_recorder(tmp_path):
    """Write an executable fake check-command recorder into <tmp_path>/bin/
    and its backing JSON state file. Returns (bin_dir, state_path)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "fake-check-recorder"
    fake.write_text(_FAKE_RECORDER_SRC)
    fake.chmod(0o755)

    state_path = tmp_path / "check_state.json"
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
    tmp_path, *, exit_code=0, extra_env=None, dotenv_contents=None, args=None,
):
    """Run fused-memory-flag-marker-check.sh with FLAG_MARKER_SWEEP_CMD
    pointed at the fake recorder and REPO pointed at a tmp dir with no
    `.env` (so the wrapper's `source .env` is a no-op under test) -- unless
    `dotenv_contents` is given, in which case a `$REPO/.env` file with that
    content is written first so the sourcing branch itself is exercised."""
    bin_dir, state_path = _fake_recorder(tmp_path)

    fake_repo = tmp_path / "fake-repo"
    fake_repo.mkdir(exist_ok=True)
    if dotenv_contents is not None:
        (fake_repo / ".env").write_text(dotenv_contents)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["FLAG_MARKER_SWEEP_CMD"] = "fake-check-recorder"
    env["FAKE_CHECK_STATE"] = str(state_path)
    env["FAKE_CHECK_EXIT_CODE"] = str(exit_code)
    env["REPO"] = str(fake_repo)
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        [BASH, str(WRAPPER), *(args or [])],
        env=env, capture_output=True, text=True, timeout=30,
    )
    return result, state_path


# ---------------------------------------------------------------------------
# Baseline behaviour (passthrough, --check flag, exit propagation)
# ---------------------------------------------------------------------------

def test_wrapper_is_executable():
    assert os.access(WRAPPER, os.X_OK), (
        f"Expected {WRAPPER} to be executable (os.X_OK); it is not. "
        f"Run: chmod +x {WRAPPER}"
    )


def test_wrapper_invokes_check_with_check_flag(tmp_path):
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
    assert "--check" in argv, f"argv={argv!r}"


def test_wrapper_passes_extra_args_through(tmp_path):
    result, state_path = _run_wrapper(
        tmp_path, args=["--project-id", "reify", "--max-backlog", "0"],
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    argv = _recorded_calls(state_path)[0]
    assert "--project-id" in argv and "reify" in argv, f"argv={argv!r}"
    assert "--max-backlog" in argv and "0" in argv, f"argv={argv!r}"


def test_wrapper_propagates_nonzero_exit(tmp_path):
    result, _state_path = _run_wrapper(tmp_path, exit_code=1)

    assert result.returncode != 0, (
        f"Expected a non-zero wrapper exit when the check command fails; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_wrapper_sources_dotenv_and_propagates_to_check(tmp_path):
    """Mirrors the sweep wrapper's equivalent test: confirms the wrapper's
    stated core purpose (source $REPO/.env, export CONFIG_PATH/PROJECT_ROOT/
    FALKORDB_URI) actually reaches the check process."""
    result, state_path = _run_wrapper(
        tmp_path,
        dotenv_contents=(
            "FALKORDB_URI=redis://test-env-host:1234\n"
            "FLAG_MARKER_CHECK_TEST_SENTINEL=from-dot-env\n"
        ),
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    envs = _recorded_envs(state_path)
    assert len(envs) == 1, f"envs={envs!r}"
    recorded_env = envs[0]
    assert recorded_env.get("FALKORDB_URI") == "redis://test-env-host:1234", (
        f"Expected the sourced .env's FALKORDB_URI to reach the check "
        f"process; recorded_env={recorded_env!r}"
    )
    assert recorded_env.get("FLAG_MARKER_CHECK_TEST_SENTINEL") == "from-dot-env", (
        f"Expected a plain .env var to reach the check process via `set -a`; "
        f"recorded_env={recorded_env!r}"
    )


# ---------------------------------------------------------------------------
# Fake `uv` shim (pins the default FLAG_MARKER_SWEEP_CMD prefix)
# ---------------------------------------------------------------------------

_FAKE_UV_SRC = '''"""Fake `uv` shim for pinning fused-memory-flag-marker-check.sh's default
FLAG_MARKER_SWEEP_CMD prefix (`uv run --frozen --project "$FM" python`).
Records argv[1:] into a JSON state file at $FAKE_CHECK_STATE and exits 0.
Never invokes real uv / fused_memory / live stores.
"""
import json
import os
import sys

state_path = os.environ["FAKE_CHECK_STATE"]
with open(state_path) as f:
    state = json.load(f)
state.setdefault("calls", []).append(sys.argv[1:])
with open(state_path, "w") as f:
    json.dump(state, f)

sys.exit(0)
'''


def _write_fake_uv(path):
    """Write the fake `uv` shim to `path`, executable, with an ABSOLUTE
    interpreter shebang.

    Absolute, not `#!/usr/bin/env python3`: the wrapper EXECs this shim, so it
    inherits the child's PATH -- and the uv-resolution tests below scrub that
    PATH to an empty dir, where `env` cannot find python3. A relative shebang
    fails 127 there, which is indistinguishable from the very bug under test.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n" + _FAKE_UV_SRC)
    path.chmod(0o755)


def _fake_uv(tmp_path):
    """Write an executable fake `uv` into <tmp_path>/uv-bin/ and its backing
    JSON state file. Returns (bin_dir, state_path)."""
    bin_dir = tmp_path / "uv-bin"
    bin_dir.mkdir(exist_ok=True)
    _write_fake_uv(bin_dir / "uv")

    state_path = tmp_path / "uv_state.json"
    state_path.write_text(json.dumps({"calls": []}))
    return bin_dir, state_path


def _empty_path_dir(tmp_path):
    """An empty directory to use as the wrapper's entire PATH.

    Task 4591 amendment (suggestion 7): scrubbing PATH to "/usr/bin:/bin"
    is host-dependent -- on a host that packages uv into /usr/bin or /bin,
    `command -v uv` succeeds, the wrapper runs the REAL uv against a tmp
    fake-repo, and the unresolvable-uv assertions fail with a confusing uv
    error instead of exercising (or skipping) the intended branch. An empty
    dir is host-independent: nothing on the wrapper's pre-exec path needs an
    external binary (`[`, `printf`, `command`, `source`, `echo` are all bash
    builtins), so an empty PATH is sufficient AND guarantees no uv is found.
    """
    path_dir = tmp_path / "empty-path"
    path_dir.mkdir(exist_ok=True)
    return str(path_dir)


def test_wrapper_default_prefix_invokes_uv_run_frozen_project(tmp_path):
    """Pins the default FLAG_MARKER_SWEEP_CMD prefix -- `uv run --frozen
    --project "$FM" python`, built literally so "$FM" survives verbatim even
    when the repo path contains spaces."""
    uv_bin_dir, state_path = _fake_uv(tmp_path)

    fake_repo = tmp_path / "fake repo with spaces"
    fake_repo.mkdir(exist_ok=True)
    fake_fm = fake_repo / "fused-memory"

    env = dict(os.environ)
    env["PATH"] = f"{uv_bin_dir}{os.pathsep}{env['PATH']}"
    env["FAKE_CHECK_STATE"] = str(state_path)
    env["REPO"] = str(fake_repo)
    env.pop("FLAG_MARKER_SWEEP_CMD", None)

    result = subprocess.run(
        [BASH, str(WRAPPER), "--project-id", "reify"],
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
        "--check", "--project-id", "reify",
    ], f"argv={calls[0]!r}"


def test_wrapper_resolves_uv_by_absolute_path_when_path_omits_it(tmp_path):
    """Task 4591 / task 2917 EDIT 3 pattern. Pins that the wrapper resolves
    `uv` to an ABSOLUTE path rather than trusting PATH.

    Simulated by scrubbing PATH down to an empty dir (no `uv` anywhere on
    it) while pointing the wrapper's UV_BIN override at the fake uv
    recorder. A wrapper that still relies on a bare `uv` word exits 127
    without ever invoking it -- the same `exec: uv: not found` / status=127
    the sweep wrapper's systemd unit hit on its boot catch-up run."""
    uv_bin_dir, state_path = _fake_uv(tmp_path)

    fake_repo = tmp_path / "fake-repo"
    fake_repo.mkdir(exist_ok=True)

    env = dict(os.environ)
    env["PATH"] = _empty_path_dir(tmp_path)
    env["FAKE_CHECK_STATE"] = str(state_path)
    env["REPO"] = str(fake_repo)
    env["UV_BIN"] = str(uv_bin_dir / "uv")
    env.pop("FLAG_MARKER_SWEEP_CMD", None)

    result = subprocess.run(
        [BASH, str(WRAPPER)],
        env=env, capture_output=True, text=True, timeout=30,
    )

    assert result.returncode == 0, (
        f"Expected the wrapper to resolve uv via UV_BIN under a PATH that "
        f"omits it (returncode 127 == the regression); "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    calls = _recorded_calls(state_path)
    assert len(calls) >= 1, (
        f"Expected the absolute-path uv to actually be invoked; calls={calls!r} "
        f"stderr={result.stderr!r}"
    )


def test_wrapper_fails_loud_when_uv_cannot_be_resolved(tmp_path):
    """A missing interpreter must be DIAGNOSABLE, not a bare shell 127
    (loud-over-silent-degradation).

    Scrubs PATH to an EMPTY dir (task 4591 amendment, suggestion 7 -- see
    _empty_path_dir; the previous "/usr/bin:/bin" was host-dependent and
    would run the real uv wherever uv is packaged into /usr/bin or /bin),
    leaves UV_BIN unset, and repoints HOME at a tmp dir so the
    $HOME/.local/bin/uv fallback misses too. The wrapper must exit non-zero
    AND print its own ERROR:-prefixed line naming `uv`.

    /usr/local/bin/uv is the one remaining unavoidable host dependence: it
    is a hardcoded absolute candidate in the ladder, so it cannot be scrubbed
    from the environment -- hence the skip rather than an assertion."""
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
    env["PATH"] = _empty_path_dir(tmp_path)
    env["REPO"] = str(fake_repo)
    env["HOME"] = str(fake_home)
    env.pop("UV_BIN", None)
    env.pop("FLAG_MARKER_SWEEP_CMD", None)

    result = subprocess.run(
        [BASH, str(WRAPPER)],
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


def test_wrapper_resolves_uv_from_home_local_bin_when_path_omits_it(tmp_path):
    """Task 4591 amendment (suggestion 3). Pins the SUCCESS side of the
    candidate loop -- the branch that actually fires in the incident this fix
    exists for.

    In the observed boot-catch-up failure there is no UV_BIN, PATH carries no
    uv, and uv is installed at $HOME/.local/bin/uv (the location the header
    calls "the measured real location"; confirmed on this host). The other two
    uv tests cover only the UV_BIN override and the total-miss branches, so a
    wrapper that honoured UV_BIN and nothing else would pass them both while
    still dying 127 on exactly the systemd run being fixed. This closes that.
    """
    _uv_bin_dir, state_path = _fake_uv(tmp_path)

    fake_repo = tmp_path / "fake-repo"
    fake_repo.mkdir(exist_ok=True)

    # A fake HOME whose .local/bin holds the recorder, so the wrapper's
    # "$HOME/.local/bin/uv" candidate -- and only that candidate -- resolves.
    fake_home = tmp_path / "fake-home"
    home_uv = fake_home / ".local" / "bin" / "uv"
    _write_fake_uv(home_uv)

    env = dict(os.environ)
    env["PATH"] = _empty_path_dir(tmp_path)
    env["HOME"] = str(fake_home)
    env["FAKE_CHECK_STATE"] = str(state_path)
    env["REPO"] = str(fake_repo)
    env.pop("UV_BIN", None)
    env.pop("FLAG_MARKER_SWEEP_CMD", None)

    result = subprocess.run(
        [BASH, str(WRAPPER)],
        env=env, capture_output=True, text=True, timeout=30,
    )

    assert result.returncode == 0, (
        f"Expected the wrapper to fall back to $HOME/.local/bin/uv under a "
        f"PATH that omits uv (returncode 127 == the boot-catch-up regression); "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    calls = _recorded_calls(state_path)
    assert len(calls) == 1, (
        f"Expected the $HOME/.local/bin/uv fallback to actually be invoked; "
        f"calls={calls!r} stderr={result.stderr!r}"
    )
    assert calls[0][:2] == ["run", "--frozen"], f"argv={calls[0]!r}"


def test_wrapper_fails_loud_when_uv_bin_is_set_but_not_executable(tmp_path):
    """Task 4591 amendment (suggestion 5). An explicit UV_BIN pin that does
    not resolve must be a LOUD failure, never a silent fall-through.

    Falling through to the PATH/candidate ladder would run a DIFFERENT uv than
    the one the operator (or a test) named -- the opposite of the
    loud-over-silent-degradation norm the ERROR: line exists to honour. Here
    UV_BIN points at a real file with the exec bit cleared while a perfectly
    good fake uv sits on PATH: a fall-through would exit 0 having invoked the
    PATH copy, so the assertions below distinguish the two behaviours rather
    than merely observing an error."""
    uv_bin_dir, state_path = _fake_uv(tmp_path)

    fake_repo = tmp_path / "fake-repo"
    fake_repo.mkdir(exist_ok=True)

    # A UV_BIN that exists but is NOT executable (the lost-exec-bit / stale-path
    # case). The ladder below it WOULD resolve -- that is the point.
    bad_uv = tmp_path / "bad-uv"
    bad_uv.write_text("#!/bin/sh\nexit 0\n")
    bad_uv.chmod(0o644)

    env = dict(os.environ)
    env["PATH"] = f"{uv_bin_dir}{os.pathsep}{_empty_path_dir(tmp_path)}"
    env["FAKE_CHECK_STATE"] = str(state_path)
    env["REPO"] = str(fake_repo)
    env["UV_BIN"] = str(bad_uv)
    env.pop("FLAG_MARKER_SWEEP_CMD", None)

    result = subprocess.run(
        [BASH, str(WRAPPER)],
        env=env, capture_output=True, text=True, timeout=30,
    )

    assert result.returncode != 0, (
        f"Expected a non-zero exit when UV_BIN is set but not executable; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "ERROR:" in result.stderr, (
        f"Expected the wrapper's own ERROR:-prefixed diagnostic; "
        f"stderr={result.stderr!r}"
    )
    assert str(bad_uv) in result.stderr, (
        f"Expected the diagnostic to name the bad UV_BIN path so an operator "
        f"can see WHICH pin failed; stderr={result.stderr!r}"
    )
    assert _recorded_calls(state_path) == [], (
        f"Expected NO uv invocation: silently falling through to the uv on "
        f"PATH would run a different uv than the one pinned. "
        f"calls={_recorded_calls(state_path)!r}"
    )


def test_wrapper_resolves_uv_after_sourcing_dotenv_so_env_can_supply_uv_bin(tmp_path):
    """Task 4591 amendment (review suggestion 4). BEHAVIOURAL pin on the
    ordering of the uv resolution relative to `set -a; source "$REPO/.env"`.

    The resolution must run AFTER the source, so a UV_BIN (or PATH) set in
    .env is visible to it. That is not a stylistic preference: it is exactly
    the remedy an operator reaches for after a minimal-boot-PATH 127, and the
    sibling fused-memory/scripts/cgl_eta_auto_apply.sh originally resolved
    BEFORE its source -- so the same .env fix would have worked here and
    failed there.

    Driven end-to-end rather than by grepping the script text: UV_BIN is
    supplied ONLY by the .env file (absent from the process env), under a PATH
    scrubbed to an empty dir and a HOME with no .local/bin/uv. If the
    resolution ran first, UV_BIN would still be unset at that moment, nothing
    else would resolve, and the wrapper would exit 127 with its ERROR: line.
    Reaching the fake uv at all is the proof.

    MUTATION-VERIFIED by hoisting the whole resolve + CHECK_CMD block above
    the `set -a` -- this test then fails with exactly the 127 + ERROR: line
    described above. Note what does NOT move the needle, since it is the
    non-obvious part: relocating the line that sources
    scripts/lib/resolve_uv.sh alone changes nothing, because it only defines
    functions and the require_uv_bin call lives inside the
    FLAG_MARKER_SWEEP_CMD else-branch further down. The invocation point is
    the invariant, not the definition point.

    (The cgl wrapper hardcodes REPO and so cannot be pointed at a fake .env;
    its copy of this invariant is pinned structurally instead, in
    fused-memory/tests/test_cgl_eta_auto_apply_wrapper.py.)
    """
    if os.path.exists("/usr/local/bin/uv"):
        pytest.skip(
            "/usr/local/bin/uv exists on this host, so the ladder resolves "
            "without .env's UV_BIN and the ordering cannot be discriminated"
        )

    uv_bin_dir, state_path = _fake_uv(tmp_path)

    fake_home = tmp_path / "fake-home"
    fake_home.mkdir(exist_ok=True)

    fake_repo = tmp_path / "fake-repo"
    fake_repo.mkdir(exist_ok=True)
    (fake_repo / ".env").write_text(f"UV_BIN={uv_bin_dir / 'uv'}\n")

    env = dict(os.environ)
    env["PATH"] = _empty_path_dir(tmp_path)
    env["HOME"] = str(fake_home)
    env["FAKE_CHECK_STATE"] = str(state_path)
    env["REPO"] = str(fake_repo)
    env.pop("UV_BIN", None)          # supplied by .env alone -- that is the point
    env.pop("FLAG_MARKER_SWEEP_CMD", None)

    result = subprocess.run(
        [BASH, str(WRAPPER)],
        env=env, capture_output=True, text=True, timeout=30,
    )

    assert result.returncode == 0, (
        f"Expected UV_BIN set in $REPO/.env to be visible to the uv "
        f"resolution. A non-zero exit here means the resolution ran BEFORE "
        f"`source $REPO/.env`, making the .env remedy for a boot-PATH 127 "
        f"silently ineffective; stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert _recorded_calls(state_path), (
        f"Expected the uv named by .env to actually be invoked; "
        f"stderr={result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# Task 3498: strict argument order and verbatim exit-code relay
# ---------------------------------------------------------------------------

def test_check_wrapper_injects_check_before_caller_args(tmp_path):
    """--check is injected by the wrapper itself and must PRECEDE the
    caller's pass-through args, whose relative order must be preserved."""
    result, state_path = _run_wrapper(
        tmp_path, args=["--project-id", "reify", "--max-backlog", "0"],
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    calls = _recorded_calls(state_path)
    assert len(calls) == 1, f"calls={calls!r}"
    argv = calls[0]
    assert argv[0].endswith(
        "fused-memory/scripts/sweep_orphan_flag_markers.py"
    ), f"Expected the sweep script path first in argv={argv!r}"
    assert argv[1:] == [
        "--check", "--project-id", "reify", "--max-backlog", "0",
    ], (
        f"Expected --check injected ahead of the caller's args, with their "
        f"relative order preserved; argv={argv!r}"
    )


def test_check_wrapper_propagates_verdict_exit_code(tmp_path):
    """The wrapper must relay the sweep's OWN exit code verbatim, not merely
    some non-zero code. Since task 3923 armed the blind-spot veto, rc=1
    carries two distinct causes (backlog over the ceiling, or an OBSERVED
    enumeration blind spot); a wrapper that normalised the code would
    destroy the consumer's ability to distinguish a verdict from a crash."""
    result, state_path = _run_wrapper(tmp_path, exit_code=1)

    assert result.returncode == 1, (
        f"Expected the sweep's own rc=1 verdict relayed verbatim, got "
        f"{result.returncode}; stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert len(_recorded_calls(state_path)) == 1, (
        f"calls={_recorded_calls(state_path)!r}"
    )


# --------------------------------------------------------------------------
# Task 3498: the advertised read-only contract, enforced.
# --------------------------------------------------------------------------

def test_check_wrapper_refuses_apply(tmp_path):
    """The header promises this wrapper 'performs NO deletions ... so
    resolve/resume re-runs are harmless'. That was prose, not a property:
    the exec line is `... --check "$@"` and _parse_args(['--check','--apply'])
    is ACCEPTED (apply=True check=True), so `--apply --terminal-drain`
    through this "read-only" wrapper ran a destructive drain. The refusal
    must happen BEFORE the sweep is invoked at all."""
    result, state_path = _run_wrapper(tmp_path, args=["--apply", "--terminal-drain"])

    assert result.returncode != 0, (
        f"Expected a non-zero refusal for --apply through the read-only "
        f"check wrapper; stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert _recorded_calls(state_path) == [], (
        f"Expected the sweep to NOT be invoked at all when --apply is "
        f"refused; calls={_recorded_calls(state_path)!r}"
    )
    assert "--apply" in result.stderr, (
        f"Expected a stderr diagnostic naming the offending --apply "
        f"argument; stderr={result.stderr!r}"
    )
    assert "fused-memory-flag-marker-sweep.sh" in result.stderr, (
        f"Expected the stderr diagnostic to point the operator at the "
        f"destructive counterpart wrapper; stderr={result.stderr!r}"
    )


@pytest.mark.parametrize("abbrev", ["--a", "--ap", "--app", "--appl"])
def test_check_wrapper_refuses_abbreviated_apply(tmp_path, abbrev):
    """argparse `allow_abbrev` defaults True, and all four of --a/--ap/--app/
    --appl were verified during task 3498's planning to reach apply=True on
    sweep_orphan_flag_markers.py. A guard matching only the literal --apply
    would therefore be bypassed by a single dropped character. This is what
    forces the guard to reject every PREFIX of --apply."""
    result, state_path = _run_wrapper(tmp_path, args=[abbrev])

    assert result.returncode != 0, (
        f"Expected {abbrev!r} (an argparse abbreviation of --apply that "
        f"reaches apply=True) to be refused; stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert _recorded_calls(state_path) == [], (
        f"Expected the sweep to NOT be invoked at all when {abbrev!r} is "
        f"refused; calls={_recorded_calls(state_path)!r}"
    )
    assert abbrev in result.stderr, (
        f"Expected a stderr diagnostic naming the offending {abbrev!r} "
        f"argument; stderr={result.stderr!r}"
    )


def test_check_wrapper_allows_non_apply_args(tmp_path):
    """Regression guard in the opposite direction: the refusal must match
    WHOLE arguments, never a substring. --no-fail-on-blind-spot (census-only
    relaxation of task 3923's veto) and a value that merely contains the
    text 'apply' must both pass through untouched."""
    result, state_path = _run_wrapper(
        tmp_path,
        args=[
            "--project-id", "apply-test",
            "--max-backlog", "0",
            "--no-fail-on-blind-spot",
            "--limit", "5000",
        ],
    )

    assert result.returncode == 0, (
        f"Expected non---apply arguments to pass through untouched; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    calls = _recorded_calls(state_path)
    assert len(calls) == 1, f"calls={calls!r}"
    assert calls[0][1:] == [
        "--check",
        "--project-id", "apply-test",
        "--max-backlog", "0",
        "--no-fail-on-blind-spot",
        "--limit", "5000",
    ], (
        f"Expected every non---apply argument forwarded verbatim after the "
        f"injected --check; argv={calls[0]!r}"
    )

    # The converse of the parser-derived test's property, on the arguments actually
    # forwarded: nothing the wrapper is willing to pass through may be a
    # spelling the sweep's own parser reads as apply=True. (Derivation
    # helper defined with that test, below.)
    reaches_apply = set(calls[0][1:]) & _derived_apply_spellings()
    assert not reaches_apply, (
        f"The wrapper forwarded {sorted(reaches_apply)!r}, which "
        f"sweep_orphan_flag_markers.py's own parser accepts as apply=True; "
        f"argv={calls[0]!r}"
    )


# --------------------------------------------------------------------------
# Parser-derived completeness: the refusal set is CHECKED against the
# sweep's own parser, not restated from it.
# --------------------------------------------------------------------------

SWEEP_SCRIPT = (
    Path(__file__).parents[2]
    / "fused-memory" / "scripts" / "sweep_orphan_flag_markers.py"
)
_PARSER_BUILDER = "_build_parser"

# argparse actions that consume no value, so `--opt=VALUE` is rejected for
# them. Anything outside this set takes a value, which makes `--opt=VALUE` a
# further accepted spelling -- see _derived_apply_spellings.
_ZERO_ARG_ACTIONS = frozenset({"store_true", "store_false", "store_const", "count"})


def _sweep_parser_options():
    """Extract ``(long_option_strings, apply_actions)`` from the sweep's own
    argparse definition by AST-parsing sweep_orphan_flag_markers.py.

    Read STATICALLY rather than imported. scripts/tests/ runs under
    ``uv run --project shared`` (dark-factory-orchestrator.yaml's
    test_command), where ``fused_memory`` -- which the sweep imports at
    module scope -- is not installed; that config's own comment records
    "scripts/tests/ imports no first-party package" as a property of the
    lane. Shelling out to the real sweep instead would need the fused-memory
    env AND would mean handing a destructive argv to a script pointed at
    live stores. ``ast.parse`` reaches the same declaration with neither
    cost, and the model it implements is validated against real argparse:
    an exhaustive sweep of every prefix and ``=VALUE`` form of every long
    option found nothing this derivation misses.
    """
    tree = ast.parse(SWEEP_SCRIPT.read_text(encoding="utf-8"))
    builder = next(
        (
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == _PARSER_BUILDER
        ),
        None,
    )
    assert builder is not None, (
        f"Expected a {_PARSER_BUILDER}() function in {SWEEP_SCRIPT}; the sweep "
        f"appears to have renamed or moved its parser builder, so this test "
        f"can no longer derive the refusal set from it. Re-point "
        f"_PARSER_BUILDER."
    )

    longs, apply_actions = [], []
    for node in ast.walk(builder):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        opts = [
            a.value for a in node.args
            if isinstance(a, ast.Constant) and isinstance(a.value, str)
        ]
        kwargs = {k.arg: k.value for k in node.keywords}
        dest = kwargs.get("dest")
        dest = dest.value if isinstance(dest, ast.Constant) else None
        action = kwargs.get("action")
        action = action.value if isinstance(action, ast.Constant) else None

        longs.extend(o for o in opts if o.startswith("--"))
        implied_dests = [
            o.lstrip("-").replace("-", "_") for o in opts if o.startswith("--")
        ]
        if dest == "apply" or "apply" in implied_dests:
            apply_actions.append((opts, action))

    return longs, apply_actions


def _derived_apply_spellings():
    """Every argv spelling sweep_orphan_flag_markers.py's parser accepts as
    ``apply=True``, derived from that parser rather than restated.

    Models argparse's ``allow_abbrev`` (default True): a ``--`` prefix of at
    least one character is accepted iff it is a prefix of exactly ONE long
    option. Today that yields --a/--ap/--app/--appl/--apply -- the five the
    wrapper's guard lists. If the sweep ever grew another ``--a...`` flag,
    the shorter prefixes would become ambiguous and drop out of this set on
    their own; if ``--apply`` were ever redefined to take a VALUE, the
    ``--apply=1`` spellings it would then accept appear here and the
    wrapper's whole-argument guard fails this test loudly instead of
    silently regressing to the deletion vector task 3498 exists to close.
    """
    longs, apply_actions = _sweep_parser_options()
    assert apply_actions, (
        f"Derived NO apply-setting action from {SWEEP_SCRIPT}'s "
        f"{_PARSER_BUILDER}(); either --apply was removed (in which case the "
        f"wrapper's guard is dead code to delete) or this derivation no "
        f"longer recognises it. Long options seen: {sorted(set(longs))!r}"
    )

    spellings = set()
    for opts, action in apply_actions:
        takes_value = action not in _ZERO_ARG_ACTIONS
        for opt in opts:
            if not opt.startswith("--"):
                spellings.add(opt)  # a short alias is never abbreviated
                continue
            for length in range(3, len(opt) + 1):  # "--" + >=1 char
                prefix = opt[:length]
                if sum(1 for o in longs if o.startswith(prefix)) == 1:
                    spellings.add(prefix)
                    if takes_value:
                        spellings.add(f"{prefix}=1")
    return spellings


def test_check_wrapper_refuses_every_apply_spelling_the_sweep_accepts(tmp_path):
    """The guard is a hand-maintained list of five literals, but its
    COMPLETENESS is a property of sweep_orphan_flag_markers.py's parser, not
    of this wrapper -- and the two refusal tests above pin only the five spellings the guard
    already names, so they can never catch drift in the thing that defines
    the set. This test derives the refusal set FROM that parser and asserts
    the wrapper refuses every member, so adding a destructive spelling to
    the sweep fails here instead of silently reopening the deletion vector.
    """
    spellings = _derived_apply_spellings()
    assert "--apply" in spellings, (
        f"Sanity: expected the literal --apply among the derived spellings, "
        f"got {sorted(spellings)!r}"
    )

    for spelling in sorted(spellings):
        result, state_path = _run_wrapper(tmp_path, args=[spelling])

        assert result.returncode != 0, (
            f"{spelling!r} is accepted by the sweep's own parser as "
            f"apply=True (derived from {_PARSER_BUILDER}() in "
            f"{SWEEP_SCRIPT.name}), but the wrapper did not refuse it. "
            f"Derived set: {sorted(spellings)!r}; stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )
        assert _recorded_calls(state_path) == [], (
            f"Expected the sweep NOT to be invoked for the apply-setting "
            f"spelling {spelling!r}; calls={_recorded_calls(state_path)!r}"
        )
