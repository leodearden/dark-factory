"""Tests for scripts/lib/resolve_uv.sh -- the one shared `uv` resolution ladder.

The lib is SOURCED, never executed, so every row drives it the way a caller
does: a throwaway probe script sources it under `set -euo pipefail` and calls
one of its two public functions. `resolve_uv_bin` is the silent rc 0/1/2
probe; `require_uv_bin` is the loud form callers use, which names itself by
the CALLER's basename.

Fake `uv` binaries are plain `#!/bin/sh` files: the lib only tests `-x` and
never execs them.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

LIB = Path(__file__).resolve().parents[1] / "lib" / "resolve_uv.sh"

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
