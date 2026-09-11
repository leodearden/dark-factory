"""Tests for fused-memory/scripts/cgl_eta_auto_apply.sh -- the SHELL wrapper
that is wired as the CGL-eta bulk-apply task's before_done predicate action.

Its Python siblings already have coverage (test_cgl_eta_auto_apply_impl.py,
test_cgl_eta_scheduler_gate.py, test_cgl_eta_finalize_gate.py); the wrapper
itself had none, so the shell logic that decides WHICH interpreter runs them
was unpinned. Task 4591 amendment (review suggestion 1): the same task added
342 lines of uv-resolution coverage to the sibling
scripts/fused-memory-flag-marker-check.sh while giving the identical
resolve_uv_bin() block here zero -- and this is the higher-consequence copy,
because a 127 here is a live predicate failure rather than a latent one.

WHAT THIS PINS -- the uv-resolution contract only, three rows mirroring the
check wrapper's:
  * an absolute UV_BIN is honoured under a PATH that carries no uv (the
    `exec: uv: not found` / status=127 boot-catch-up failure this fix exists
    for -- see the wrapper header);
  * an unresolvable uv fails LOUDLY with an ERROR: line, not a bare 127;
  * a UV_BIN that is SET but not executable fails loudly too, and never
    silently falls through to the ladder and runs a different uv.

WHAT IT DELIBERATELY DOES NOT DO -- record the child's environment. Unlike
the check wrapper, this one hardcodes `REPO=/home/leo/src/dark-factory` (it
is not env-overridable), so it sources the REAL repo `.env`, which carries
live API keys and OAuth tokens. The fake `uv` below therefore records argv
ONLY. Do not add an os.environ snapshot to it, however convenient: it would
write secrets into a tmp JSON file and into any failure's assertion output.
Real uv, real fused_memory and live stores are never touched -- every
invocation is intercepted by the fake.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_FM = Path(__file__).resolve().parents[1]
WRAPPER = _FM / "scripts" / "cgl_eta_auto_apply.sh"
CHECK_WRAPPER = _FM.parent / "scripts" / "fused-memory-flag-marker-check.sh"

# Resolved in the PARENT: the tests below hand the child an empty PATH, and
# subprocess looks argv[0] up in the CHILD's env, so a bare "bash" would be
# unfindable for reasons that have nothing to do with the wrapper.
BASH = shutil.which("bash") or "/bin/bash"

# The wrapper's four call sites: halt, impl, finalize, and the EXIT-trap resume.
_EXPECTED_UV_CALLS = 4


# ---------------------------------------------------------------------------
# Fake `uv` (argv-only recorder -- see the module docstring on why)
# ---------------------------------------------------------------------------

_FAKE_UV_SRC = '''"""Fake `uv` for testing cgl_eta_auto_apply.sh. Appends argv[1:] to a JSON
state file at $FAKE_UV_STATE and exits 0. Records NO environment: the wrapper
sources the real repo .env, which holds live credentials.
"""
import json
import os
import sys

state_path = os.environ["FAKE_UV_STATE"]
with open(state_path) as f:
    state = json.load(f)
state.setdefault("calls", []).append(sys.argv[1:])
with open(state_path, "w") as f:
    json.dump(state, f)

sys.exit(0)
'''


def _write_fake_uv(path):
    """Write the fake `uv` to `path`, executable, with an ABSOLUTE shebang.

    Absolute rather than `#!/usr/bin/env python3`: the wrapper invokes this
    under the scrubbed PATH the tests set, where `env` cannot find python3.
    A relative shebang would fail 127 there -- indistinguishable from the very
    bug under test.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n" + _FAKE_UV_SRC)
    path.chmod(0o755)


def _fake_uv(tmp_path):
    """Write an executable fake `uv` into <tmp_path>/"uv bin"/ plus its backing
    JSON state file. Returns (uv_path, state_path).

    The SPACE in the directory name is deliberate: it pins the quoting of
    "$UV_BIN_RESOLVED" at the wrapper's four call sites. Unquoted, the path
    word-splits and the wrapper tries to exec a directory prefix -- a
    regression a space-free tmp path would never catch.
    """
    uv_path = tmp_path / "uv bin" / "uv"
    _write_fake_uv(uv_path)

    state_path = tmp_path / "uv_state.json"
    state_path.write_text(json.dumps({"calls": []}))
    return uv_path, state_path


def _recorded_calls(state_path):
    return json.loads(state_path.read_text())["calls"]


def _empty_path_dir(tmp_path):
    """An empty directory to use as the wrapper's entire PATH.

    Host-independent, unlike "/usr/bin:/bin": on a host that packages uv into
    /usr/bin or /bin, `command -v uv` would succeed and the wrapper would run
    the REAL uv. Nothing on the wrapper's path to the uv resolution needs an
    external binary -- `[`, `printf`, `command`, `source`, `echo` and `trap`
    are all bash builtins -- so an empty PATH is sufficient AND guarantees no
    uv is found. (CGL_RUN_STAMP is preset by _base_env so the wrapper's one
    genuinely external call, `date -u`, is never reached either.)
    """
    path_dir = tmp_path / "empty-path"
    path_dir.mkdir(exist_ok=True)
    return str(path_dir)


def _base_env(tmp_path, state_path=None):
    env = dict(os.environ)
    env["PATH"] = _empty_path_dir(tmp_path)
    # Preset so `${CGL_RUN_STAMP:-$(date -u ...)}` never shells out to `date`,
    # which is not reachable under the scrubbed PATH above.
    env["CGL_RUN_STAMP"] = "20260101T000000Z"
    if state_path is not None:
        env["FAKE_UV_STATE"] = str(state_path)
    env.pop("UV_BIN", None)
    return env


def _run(env, tmp_path):
    return subprocess.run(
        [BASH, str(WRAPPER)],
        env=env, cwd=str(tmp_path), capture_output=True, text=True, timeout=60,
    )


# ---------------------------------------------------------------------------
# uv resolution
# ---------------------------------------------------------------------------

def test_wrapper_is_executable():
    assert os.access(WRAPPER, os.X_OK), (
        f"Expected {WRAPPER} to be executable (os.X_OK); it is not. "
        f"Run: chmod +x {WRAPPER}"
    )


def test_wrapper_resolves_uv_by_absolute_path_when_path_omits_it(tmp_path):
    """The row this whole fix exists for: PATH carries no uv, so a wrapper
    that still spelled the interpreter as a bare `uv` word would die
    `exec: uv: not found` / status=127 -- exactly what the sibling sweep
    wrapper's systemd unit hit on its Persistent=true boot catch-up run.
    An absolute UV_BIN must be honoured instead."""
    uv_path, state_path = _fake_uv(tmp_path)

    env = _base_env(tmp_path, state_path)
    env["UV_BIN"] = str(uv_path)

    result = _run(env, tmp_path)

    assert result.returncode == 0, (
        f"Expected the wrapper to resolve uv via UV_BIN under a PATH that "
        f"omits it (returncode 127 == the boot-catch-up regression); "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    calls = _recorded_calls(state_path)
    assert len(calls) == _EXPECTED_UV_CALLS, (
        f"Expected {_EXPECTED_UV_CALLS} uv invocations (scheduler halt, impl, "
        f"finalize gate, and the EXIT-trap resume); calls={calls!r} "
        f"stderr={result.stderr!r}"
    )
    # Every call must carry the `run --project <FM> python <script>` shape --
    # i.e. it is the resolved absolute uv being invoked, not a shell builtin
    # or some other binary that happened to be reachable.
    for argv in calls:
        assert argv[:2] == ["run", "--project"], f"argv={argv!r}"
        assert argv[3] == "python", f"argv={argv!r}"
        assert argv[4].endswith(".py"), f"argv={argv!r}"

    scripts_invoked = [Path(argv[4]).name for argv in calls]
    assert scripts_invoked == [
        "cgl_eta_scheduler_gate.py",     # halt
        "cgl_eta_auto_apply_impl.py",    # the apply itself
        "cgl_eta_finalize_gate.py",      # best-effort gate close
        "cgl_eta_scheduler_gate.py",     # EXIT-trap resume
    ], f"scripts_invoked={scripts_invoked!r}"


def test_wrapper_fails_loud_when_uv_cannot_be_resolved(tmp_path):
    """A missing interpreter must be DIAGNOSABLE, not a bare shell 127
    (loud-over-silent-degradation). Scrubs PATH to an empty dir, leaves
    UV_BIN unset, and repoints HOME at a tmp dir so the $HOME/.local/bin/uv
    fallback misses too.

    /usr/local/bin/uv is the one host dependence that cannot be scrubbed --
    it is a hardcoded absolute candidate in the ladder -- hence the skip
    rather than an assertion."""
    if os.path.exists("/usr/local/bin/uv"):
        pytest.skip(
            "/usr/local/bin/uv exists on this host, so the wrapper's last-resort "
            "fallback resolves and the unresolvable-uv path cannot be exercised"
        )

    fake_home = tmp_path / "fake-home"
    fake_home.mkdir(exist_ok=True)

    env = _base_env(tmp_path)
    env["HOME"] = str(fake_home)

    result = _run(env, tmp_path)

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


def test_wrapper_fails_loud_when_uv_bin_is_set_but_not_executable(tmp_path):
    """An explicit UV_BIN pin that does not resolve must fail LOUDLY, never
    fall through to the rest of the ladder: doing so would run a DIFFERENT uv
    than the one named. Here a perfectly good fake uv sits on PATH, so a
    fall-through would exit 0 having invoked it -- the assertions below
    distinguish the two behaviours rather than merely observing an error."""
    uv_path, state_path = _fake_uv(tmp_path)

    # Exists, but the exec bit is cleared (the stale-path / lost-exec-bit case).
    bad_uv = tmp_path / "bad-uv"
    bad_uv.write_text("#!/bin/sh\nexit 0\n")
    bad_uv.chmod(0o644)

    env = _base_env(tmp_path, state_path)
    env["PATH"] = f"{uv_path.parent}{os.pathsep}{_empty_path_dir(tmp_path)}"
    env["UV_BIN"] = str(bad_uv)

    result = _run(env, tmp_path)

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


# ---------------------------------------------------------------------------
# Structural: the resolve-after-source ordering, shared with the sibling
# ---------------------------------------------------------------------------

def _code_line_of(src, needle, label, path):
    """Line number of the first NON-COMMENT line containing `needle`.

    Comments are skipped deliberately. Both wrapper headers now DISCUSS this
    ordering in prose, quoting `source "$REPO/.env"` verbatim, so a naive
    `src.find(needle)` matches the documentation rather than the code -- which
    makes the assertion below silently vacuous. MEASURED, which is why this
    exists: the first cut of this test passed unchanged against a wrapper
    whose resolve block had been moved back above the source.
    """
    for i, line in enumerate(src.splitlines()):
        if line.lstrip().startswith("#"):
            continue
        if needle in line:
            return i
    raise AssertionError(
        f"Expected a non-comment {label} line in {path}; found none"
    )


def test_both_wrappers_resolve_uv_after_sourcing_dotenv():
    """Task 4591 amendment (review suggestion 4). Pins the ORDER, which is
    load-bearing and was asymmetric in this file's first cut: uv must be
    resolved AFTER `set -a; source "$REPO/.env"`, in BOTH wrappers.

    Resolving first would ignore a PATH or UV_BIN set in .env -- which is
    precisely the remedy an operator reaches for after a minimal-boot-PATH
    127 -- so this wrapper would have failed where its sibling succeeded.
    Asserted structurally because this wrapper hardcodes REPO and so cannot
    be pointed at a fake .env at runtime. Cheap insurance against a future
    tidy-up silently reintroducing the asymmetry.
    """
    for path in (WRAPPER, CHECK_WRAPPER):
        src = path.read_text()
        source_line = _code_line_of(
            src, 'source "$REPO/.env"', "`source .env`", path,
        )
        resolve_line = _code_line_of(
            src, "resolve_uv_bin() {", "resolve_uv_bin() definition", path,
        )
        assert source_line < resolve_line, (
            f"{path.name} resolves uv BEFORE sourcing $REPO/.env. That makes a "
            f"PATH or UV_BIN set in .env invisible to the resolution -- the "
            f"exact operator remedy for the boot-PATH 127 this ladder exists "
            f"for -- and desynchronises it from its sibling wrapper. Move the "
            f"resolve_uv_bin definition and its guard below the `set +a` block."
        )
