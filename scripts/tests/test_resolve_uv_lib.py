"""Tests for scripts/lib/resolve_uv.sh -- the one shared `uv` resolution ladder.

The lib is SOURCED, never executed, so every contract row drives it the way a
caller does: a throwaway probe script sources it under `set -euo pipefail` and
calls one of its two public functions. `resolve_uv_bin` is the silent rc 0/1/2
probe; `require_uv_bin` is the loud form callers use, which names itself by
the CALLER's basename.

Fake `uv` binaries are plain `#!/bin/sh` files: the lib only tests `-x` and
never execs them.

The wiring rows run a COPY of each real caller from a tmp tree, beside a stub
lib that announces itself. Every one of them carries a safety net, because a
mis-wired caller could otherwise reach live systems: the cgl wrapper sources
the MAIN checkout's .env and halts real schedulers, and the sync script stops
the live orchestrator fleet. See _caller_env.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from git_checkout_root import checkout_root_or_skip

REPO_ROOT = Path(__file__).resolve().parents[2]
LIB_RELPATH = "scripts/lib/resolve_uv.sh"
LIB = REPO_ROOT / LIB_RELPATH

CALLERS = (
    "scripts/fused-memory-flag-marker-sweep.sh",
    "scripts/fused-memory-flag-marker-check.sh",
    "fused-memory/scripts/cgl_eta_auto_apply.sh",
    "scripts/sync-orchestrator-env.sh",
)

STUB_MARKER = "STUB-RESOLVER-CALLED"
_STUB_LIB_SRC = (
    "resolve_uv_bin() { return 1; }\n"
    f"require_uv_bin() {{ echo {STUB_MARKER} >&2; return 99; }}\n"
)

# Resolved in the PARENT: the child gets a scrubbed PATH, under which a bare
# "bash" argv[0] would itself be unfindable.
BASH = shutil.which("bash") or "/bin/bash"

PROBE_NAME = "probe-caller.sh"

_NO_LAST_RESORT_UV = pytest.mark.skipif(
    os.path.exists("/usr/local/bin/uv"),
    reason=(
        "/usr/local/bin/uv exists on this host, so the ladder's last-resort "
        "fallback resolves and the unresolvable-uv path cannot be exercised"
    ),
)


def _write_fake_uv(path: Path, *, mode: int = 0o755) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(mode)
    return path


def _empty_dir(tmp_path: Path, name: str) -> Path:
    """A directory holding nothing -- as a PATH, it guarantees no uv is found.

    Host-independent where "/usr/bin:/bin" is not (a host may package uv
    there), and sufficient because everything the lib runs is a builtin.
    """
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    return directory


def _run_probe(
    tmp_path: Path,
    fn: str,
    *,
    path: str,
    home: Path | None,
    uv_bin: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Source the lib from a probe caller and run `fn`. `home=None` unsets HOME."""
    probe = tmp_path / PROBE_NAME
    probe.write_text('set -euo pipefail\nsource "$RESOLVE_UV_LIB"\n"$@"\n')

    env = dict(os.environ)
    env["RESOLVE_UV_LIB"] = str(LIB)
    env["PATH"] = path
    if uv_bin is None:
        env.pop("UV_BIN", None)
    else:
        env["UV_BIN"] = str(uv_bin)
    if home is None:
        env.pop("HOME", None)
    else:
        env["HOME"] = str(home)

    return subprocess.run(
        [BASH, str(probe), fn],
        env=env, capture_output=True, text=True, timeout=30, cwd=tmp_path,
    )


def _diag(result: subprocess.CompletedProcess[str]) -> str:
    return f"rc={result.returncode} stdout={result.stdout!r} stderr={result.stderr!r}"


# ---------------------------------------------------------------------------
# resolve_uv_bin -- the silent probe: rc 0 (path on stdout), 1 (nothing), 2 (bad pin)
# ---------------------------------------------------------------------------

def test_resolve_honors_executable_uv_bin_over_path(tmp_path):
    pinned = _write_fake_uv(tmp_path / "pinned" / "uv")
    on_path = _write_fake_uv(tmp_path / "on-path" / "uv")

    result = _run_probe(
        tmp_path, "resolve_uv_bin",
        path=str(on_path.parent), home=_empty_dir(tmp_path, "home"), uv_bin=pinned,
    )

    assert result.returncode == 0, _diag(result)
    assert result.stdout == str(pinned), _diag(result)
    assert result.stderr == "", _diag(result)


def test_resolve_refuses_to_fall_through_a_non_executable_uv_bin(tmp_path):
    """A set-but-unusable pin is rc 2 even though a good uv sits on PATH:
    falling through would run a DIFFERENT uv than the one named."""
    bad_uv = _write_fake_uv(tmp_path / "bad-uv", mode=0o644)
    on_path = _write_fake_uv(tmp_path / "on-path" / "uv")

    result = _run_probe(
        tmp_path, "resolve_uv_bin",
        path=str(on_path.parent), home=_empty_dir(tmp_path, "home"), uv_bin=bad_uv,
    )

    assert result.returncode == 2, _diag(result)
    assert result.stdout == "", _diag(result)
    assert result.stderr == "", _diag(result)


def test_resolve_prefers_path_over_home_local_bin(tmp_path):
    on_path = _write_fake_uv(tmp_path / "on-path" / "uv")
    home = tmp_path / "home"
    _write_fake_uv(home / ".local" / "bin" / "uv")

    result = _run_probe(
        tmp_path, "resolve_uv_bin", path=str(on_path.parent), home=home,
    )

    assert result.returncode == 0, _diag(result)
    assert result.stdout == str(on_path), _diag(result)
    assert result.stderr == "", _diag(result)


def test_resolve_falls_back_to_home_local_bin_when_path_omits_uv(tmp_path):
    """The branch that fires in the boot-catch-up incident: no UV_BIN, no uv
    on PATH, uv installed at $HOME/.local/bin/uv."""
    home = tmp_path / "home"
    home_uv = _write_fake_uv(home / ".local" / "bin" / "uv")

    result = _run_probe(
        tmp_path, "resolve_uv_bin",
        path=str(_empty_dir(tmp_path, "empty-path")), home=home,
    )

    assert result.returncode == 0, _diag(result)
    assert result.stdout == str(home_uv), _diag(result)
    assert result.stderr == "", _diag(result)


@_NO_LAST_RESORT_UV
def test_resolve_reports_a_quiet_miss_when_uv_is_nowhere(tmp_path):
    result = _run_probe(
        tmp_path, "resolve_uv_bin",
        path=str(_empty_dir(tmp_path, "empty-path")),
        home=_empty_dir(tmp_path, "home"),
    )

    assert result.returncode == 1, _diag(result)
    assert result.stdout == "", _diag(result)
    assert result.stderr == "", _diag(result)


# ---------------------------------------------------------------------------
# require_uv_bin -- the loud form: path on stdout, or `<caller>: ERROR:` + 127
# ---------------------------------------------------------------------------

def test_require_prints_the_resolved_path(tmp_path):
    on_path = _write_fake_uv(tmp_path / "on-path" / "uv")

    result = _run_probe(
        tmp_path, "require_uv_bin",
        path=str(on_path.parent), home=_empty_dir(tmp_path, "home"),
    )

    assert result.returncode == 0, _diag(result)
    assert result.stdout == str(on_path), _diag(result)


def test_require_names_a_non_executable_uv_bin_under_the_callers_name(tmp_path):
    bad_uv = _write_fake_uv(tmp_path / "bad-uv", mode=0o644)
    on_path = _write_fake_uv(tmp_path / "on-path" / "uv")

    result = _run_probe(
        tmp_path, "require_uv_bin",
        path=str(on_path.parent), home=_empty_dir(tmp_path, "home"), uv_bin=bad_uv,
    )

    assert result.returncode == 127, _diag(result)
    assert result.stdout == "", _diag(result)
    assert result.stderr.startswith(f"{PROBE_NAME}: ERROR:"), _diag(result)
    assert str(bad_uv) in result.stderr, _diag(result)


@_NO_LAST_RESORT_UV
def test_require_names_uv_and_the_searched_path_when_uv_is_nowhere(tmp_path):
    searched = str(_empty_dir(tmp_path, "empty-path"))

    result = _run_probe(
        tmp_path, "require_uv_bin", path=searched, home=_empty_dir(tmp_path, "home"),
    )

    assert result.returncode == 127, _diag(result)
    assert result.stdout == "", _diag(result)
    assert result.stderr.startswith(f"{PROBE_NAME}: ERROR:"), _diag(result)
    assert "uv" in result.stderr, _diag(result)
    assert searched in result.stderr, _diag(result)


@_NO_LAST_RESORT_UV
def test_require_reaches_its_diagnostic_with_home_unset(tmp_path):
    """Callers source the lib under `set -u`, so an unset HOME must still
    reach the designed ERROR line, not abort on `HOME: unbound variable`."""
    result = _run_probe(
        tmp_path, "require_uv_bin",
        path=str(_empty_dir(tmp_path, "empty-path")), home=None,
    )

    assert result.returncode == 127, _diag(result)
    assert result.stderr.startswith(f"{PROBE_NAME}: ERROR:"), _diag(result)
    assert "unbound variable" not in result.stderr, _diag(result)


# ---------------------------------------------------------------------------
# Wiring -- every caller sources THIS lib, found relative to itself
# ---------------------------------------------------------------------------

def _caller_tree(tmp_path: Path, relpath: str, *, with_stub_lib: bool) -> Path:
    """Copy one caller into tmp_path/tree at its repo-relative path; return the copy.

    A caller that finds the stub has located the lib relative to its OWN
    location: a $REPO-, cwd- or main-checkout-relative source never sees it.
    """
    tree = tmp_path / "tree"
    copied = tree / relpath
    copied.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / relpath, copied)
    if with_stub_lib:
        stub = tree / LIB_RELPATH
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text(_STUB_LIB_SRC)
    return copied


def _write_fake_tool(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)


def _caller_env(tmp_path: Path) -> dict[str, str]:
    """One env for every caller, built so a MIS-wired caller stays harmless.

    - UV_BIN is a NON-executable file, so any REAL lib a caller reached would
      stop at rc=2 before running uv.
    - systemctl, pgrep and sleep are fakes FIRST on PATH: systemctl only logs,
      and pgrep always reports "still alive", which makes the sync script
      abort before any `uv sync`.
    """
    fake_bin = tmp_path / "fake-bin"
    _write_fake_tool(fake_bin / "systemctl", 'echo "$*" >> "$FAKE_SYSTEMCTL_LOG"\nexit 0')
    _write_fake_tool(fake_bin / "pgrep", "exit 0")
    _write_fake_tool(fake_bin / "sleep", "exit 0")

    not_executable_uv = _write_fake_uv(tmp_path / "not-executable-uv", mode=0o644)

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}{os.pathsep}/usr/bin{os.pathsep}/bin"
    env["FAKE_SYSTEMCTL_LOG"] = str(tmp_path / "systemctl.log")
    env["UV_BIN"] = str(not_executable_uv)
    env["REPO"] = str(_empty_dir(tmp_path, "fake-repo"))
    env["HOME"] = str(_empty_dir(tmp_path, "home"))
    env["CGL_RUN_STAMP"] = "20260101T000000Z"
    env["FLAG_MARKER_SWEEP_PROJECT_IDS"] = "dark_factory"
    env.pop("FLAG_MARKER_SWEEP_CMD", None)
    env.pop("DASHBOARD_KNOWN_PROJECT_ROOTS", None)
    return env


def _run_caller(
    argv: list[str], *, cwd: Path, tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, env=_caller_env(tmp_path), cwd=cwd, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=60,
    )


def _systemctl_calls(tmp_path: Path) -> str:
    log = tmp_path / "systemctl.log"
    return log.read_text() if log.exists() else ""


@pytest.mark.parametrize("relpath", CALLERS)
def test_caller_sources_resolver_relative_to_itself(tmp_path, relpath):
    copied = _caller_tree(tmp_path, relpath, with_stub_lib=True)

    result = _run_caller([BASH, str(copied)], cwd=tmp_path, tmp_path=tmp_path)

    assert result.returncode != 0, _diag(result)
    assert STUB_MARKER in result.stderr, _diag(result)
    assert _systemctl_calls(tmp_path) == "", _diag(result)


@pytest.mark.parametrize("relpath", CALLERS)
def test_caller_finds_resolver_when_invoked_by_bare_name(tmp_path, relpath):
    copied = _caller_tree(tmp_path, relpath, with_stub_lib=True)

    result = _run_caller([BASH, copied.name], cwd=copied.parent, tmp_path=tmp_path)

    assert STUB_MARKER in result.stderr, _diag(result)


@pytest.mark.parametrize("relpath", CALLERS)
def test_caller_fails_loud_when_resolver_lib_missing(tmp_path, relpath):
    copied = _caller_tree(tmp_path, relpath, with_stub_lib=False)

    result = _run_caller([BASH, str(copied)], cwd=tmp_path, tmp_path=tmp_path)

    assert result.returncode != 0, _diag(result)
    assert f"{copied.name}: ERROR:" in result.stderr, _diag(result)
    assert "resolve_uv.sh" in result.stderr, _diag(result)
    assert _systemctl_calls(tmp_path) == "", _diag(result)


_RESOLVER_DEFINITION = re.compile(
    r"^\s*(function\s+)?(resolve_uv_bin|require_uv_bin)\s*\(\s*\)"
)


def test_only_the_lib_defines_the_resolver():
    """A private copy left in a caller, defined AFTER its source line, would
    shadow the lib's function while every behavioural row above still passed."""
    root = Path(checkout_root_or_skip())
    listed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", "*.sh"],
        capture_output=True, text=True, timeout=30, check=True,
    ).stdout

    defining = set()
    for relpath in filter(None, listed.split("\0")):
        path = root / relpath
        if not path.is_file():
            continue
        for line in path.read_text(errors="replace").splitlines():
            if not line.lstrip().startswith("#") and _RESOLVER_DEFINITION.match(line):
                defining.add(relpath)
                break

    assert defining == {LIB_RELPATH}, (
        f"Only {LIB_RELPATH} may define resolve_uv_bin/require_uv_bin; "
        f"defining files: {sorted(defining)}"
    )
