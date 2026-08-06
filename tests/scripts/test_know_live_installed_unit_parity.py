"""Host-coupled parity guard: the INSTALLED orchestrator-know-live.service unit.

Deliberately HOST-COUPLED — the opposite contract of every sibling parity
suite in this directory. test_check_dashboard_unit_parity.py states its rule
plainly: "All drift-logic tests run against inline fixture strings and
tmp_path directories — NEVER the host's real ~/.config/systemd/user/". This
module exists BECAUSE that rule is correct for a general parity checker but
cannot answer the one question task 3642 needs answered: has THIS host's
installed unit, and its systemd --user manager, actually been reconciled
with the committed template. scripts/check_orchestrator_unit_parity.py (a
registry-driven installed-vs-committed gate) is confirmed absent from main —
its commits live only on the unmerged task/3424 branch — so there is no
portable checker this module could instead exercise against a fixture. A
future run on a host with the unit installed and reconciled gets a live
green answer about ITS install; a fresh checkout or CI runner with no
installed unit and no user D-Bus session degrades to a skip (see the guards
below), never a false failure.

Scope is deliberately narrow: two PROPERTIES of exactly one unit
(orchestrator-know-live.service), at two layers — not a byte-parity sweep
across the fleet:

  1. FILE layer — the installed unit FILE on disk.
  2. MANAGER layer — systemd --user's LOADED view of the unit, via
     `systemctl --user show`. Not redundant with the file layer: `cp`-ing a
     corrected unit into place without `daemon-reload` leaves the manager
     holding the stale unit, so only the manager layer proves a reload
     actually happened.

Each layer checks two properties: the RestartMaxDelaySec=/RestartSteps=
pairing (systemd silently drops an unpaired cap — see
tests/scripts/systemd_unit_invariants.py) and the `--config` argument
naming the canonical dark-factory-orchestrator.yaml basename (task 3641;
CANONICAL_CONFIG_BASENAME mirrors tests/scripts/test_orchestrator_service_
files.py:543). Deliberately NOT asserted: ActiveState — liveness is the
watchdog's job (scripts/orchestrator-watchdog.py) and pinning it here would
make this suite fail during any legitimate restart window.

Importable `from systemd_unit_invariants import ...` because
tests/scripts/conftest.py puts this directory on sys.path — pytest's
--import-mode=importlib (pyproject.toml addopts) deliberately does not do
that on its own.
"""

import os
import pathlib
import shutil
import subprocess

import pytest

from systemd_unit_invariants import assert_restart_backoff_effective, restart_directive

REPO_ROOT = pathlib.Path(__file__).parents[2]

# Mirrors scripts/setup-host.sh:114 (`UNIT_DIR="$HOME/.config/systemd/user"`),
# with an XDG_CONFIG_HOME override layered on top for robustness — the
# installer itself does not honour XDG_CONFIG_HOME, so the two can only
# diverge if a future host sets that variable while the installer stays
# unchanged; on this host (XDG_CONFIG_HOME unset) both resolve to the
# identical /home/leo/.config/systemd/user, as measured.
UNIT_DIR = (
    pathlib.Path(os.environ.get("XDG_CONFIG_HOME", pathlib.Path.home() / ".config"))
    / "systemd"
    / "user"
)

UNIT_BASENAME = "orchestrator-know-live.service"
INSTALLED_UNIT_PATH = UNIT_DIR / UNIT_BASENAME

# Mirrors tests/scripts/test_orchestrator_service_files.py:543 — copied
# inline (not imported across test modules) per this module's reuse note.
CANONICAL_CONFIG_BASENAME = "dark-factory-orchestrator.yaml"

_SYSTEMCTL_SKIP_REASON = (
    "systemctl is not installed; this module requires a live systemd --user "
    "manager and has no fixture-based fallback (see module docstring)"
)


def _require_installed_unit() -> pathlib.Path:
    """Return INSTALLED_UNIT_PATH, or skip if it does not exist on this host.

    A fresh checkout or a CI runner has no ~/.config/systemd/user at all —
    that is an environment fact, not a defect in the unit, so this skips
    rather than fails.
    """
    if not INSTALLED_UNIT_PATH.exists():
        pytest.skip(
            f"{INSTALLED_UNIT_PATH} does not exist on this host (fresh "
            "checkout or CI runner with no installed orchestrator units)"
        )
    return INSTALLED_UNIT_PATH


def _config_arg_from_exec_start(exec_start_value: str) -> str | None:
    """Token immediately after `--config` in an ExecStart= value, or None.

    Mirrors test_orchestrator_service_files.py's _exec_start_config_arg
    token parse (also accepts `--config=VALUE`), copied inline rather than
    imported across test modules.
    """
    tokens = exec_start_value.split()
    for i, token in enumerate(tokens):
        if token == "--config":
            return tokens[i + 1] if i + 1 < len(tokens) else None
        if token.startswith("--config="):
            return token.split("=", 1)[1]
    return None


def _systemctl_user_show(unit: str, *properties: str) -> dict[str, str] | None:
    """Run `systemctl --user show <unit> -p <prop> ...`, parsed to a dict.

    Returns None — never raises — when the query cannot be answered at all:
    a non-zero exit, or output naming a bus connection failure ("Failed to
    connect to bus" — the shape seen from a container/CI sandbox with no
    user D-Bus session). Callers treat None as "skip", not "fail": this
    invariant requires a live systemd --user manager to answer, and its
    absence is an environment fact rather than a defect in the unit.
    """
    argv = ["systemctl", "--user", "show", unit]
    for prop in properties:
        argv += ["-p", prop]
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=30, check=False,
        )
    except OSError:
        return None
    combined = f"{result.stdout}{result.stderr}"
    if result.returncode != 0 or "failed to connect to bus" in combined.lower():
        return None
    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            parsed[key] = value
    return parsed


# ---------------------------------------------------------------------------
# FILE layer — the installed unit on disk
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("systemctl") is None, reason=_SYSTEMCTL_SKIP_REASON)
def test_installed_unit_file_restart_backoff_effective() -> None:
    """The INSTALLED unit file's RestartMaxDelaySec= must be paired with RestartSteps=.

    Measured 2026-08-06: the installed copy at ~/.config/systemd/user/
    orchestrator-know-live.service declares RestartMaxDelaySec=60 with no
    RestartSteps= line — the committed scripts/orchestrator-know-live.service
    is ahead by exactly that directive (plus its two-line explanatory
    comment), landed for the whole fleet by commit f7459a1c49 but never
    propagated to this host for know-live. Reuses
    systemd_unit_invariants.assert_restart_backoff_effective, the same
    RELATIONAL invariant test_systemd_restart_backoff.py already applies to
    the COMMITTED template — this is that same check applied to the
    INSTALLED file instead.
    """
    path = _require_installed_unit()
    assert_restart_backoff_effective(path)


@pytest.mark.skipif(shutil.which("systemctl") is None, reason=_SYSTEMCTL_SKIP_REASON)
def test_installed_unit_file_execstart_config_is_canonical() -> None:
    """The INSTALLED unit file's --config must name the canonical basename.

    Expected GREEN on arrival — a regression pin, not the RED signal this
    module exists to add. Task 3641 already canonicalized the installed
    file's ExecStart= (measured: it agrees byte-for-byte with the committed
    template on this line); this guards against a FUTURE setup-host.sh run
    re-installing a stale template that regresses it.
    """
    path = _require_installed_unit()
    exec_start = restart_directive(path, "ExecStart")
    assert exec_start is not None, f"{path} declares no ExecStart= line"
    config_arg = _config_arg_from_exec_start(exec_start)
    assert config_arg is not None, (
        f"{path}'s ExecStart= carries no --config argument: {exec_start!r}"
    )
    actual = pathlib.PurePosixPath(config_arg).name
    assert actual == CANONICAL_CONFIG_BASENAME, (
        f"{path}'s ExecStart= points --config at {config_arg!r}, whose "
        f"basename is {actual!r}. It must be the canonical "
        f"{CANONICAL_CONFIG_BASENAME!r} (CLAUDE.md; task 3641) — the "
        "dashboard's _discover_escalation_urls keys on that exact filename."
    )


# ---------------------------------------------------------------------------
# MANAGER layer — systemd --user's LOADED view of the unit
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("systemctl") is None, reason=_SYSTEMCTL_SKIP_REASON)
def test_installed_unit_manager_restart_steps_effective() -> None:
    """systemd --user's LOADED view must report RestartSteps=4, not just the file.

    Not redundant with the file-layer check above: `cp`-ing a corrected unit
    into place without `systemctl --user daemon-reload` leaves the MANAGER
    holding the old unit, so a file-only check would bless a host whose
    backoff is still inert. Measured 2026-08-06: `systemctl --user show
    orchestrator-know-live.service -p RestartSteps` reports RestartSteps=0 —
    systemd's zero-value default, confirming the manager has not reloaded
    the RestartSteps=4 line at all.
    """
    _require_installed_unit()
    shown = _systemctl_user_show(UNIT_BASENAME, "RestartSteps")
    if shown is None:
        pytest.skip(
            "systemctl --user show could not be queried (no user D-Bus "
            "session in this runner)"
        )
    steps = shown.get("RestartSteps")
    assert steps == "4", (
        f"systemctl --user show {UNIT_BASENAME} -p RestartSteps reports "
        f"RestartSteps={steps!r}, not '4'. The committed "
        "scripts/orchestrator-know-live.service declares RestartSteps=4; "
        "propagate it with `install -m 0644 scripts/orchestrator-know-live"
        f".service {INSTALLED_UNIT_PATH}` followed by `systemctl --user "
        f"daemon-reload` and `systemctl --user restart {UNIT_BASENAME}`."
    )


@pytest.mark.skipif(shutil.which("systemctl") is None, reason=_SYSTEMCTL_SKIP_REASON)
def test_installed_unit_manager_execstart_config_is_canonical() -> None:
    """systemd --user's LOADED ExecStart= must carry the canonical --config path.

    Expected GREEN on arrival, mirroring the file-layer config check above —
    the manager already agrees with the (already-canonical) installed file.
    Kept as its own assertion, at its own layer, so a future regression that
    only reaches the file (an install without a daemon-reload) or only
    reaches the manager (a hand-edited drop-in) is still caught by whichever
    layer it actually lands in.
    """
    _require_installed_unit()
    shown = _systemctl_user_show(UNIT_BASENAME, "ExecStart")
    if shown is None:
        pytest.skip(
            "systemctl --user show could not be queried (no user D-Bus "
            "session in this runner)"
        )
    exec_start = shown.get("ExecStart", "")
    assert CANONICAL_CONFIG_BASENAME in exec_start, (
        f"systemctl --user show {UNIT_BASENAME} -p ExecStart reports "
        f"{exec_start!r}, which does not contain the canonical config "
        f"basename {CANONICAL_CONFIG_BASENAME!r} (CLAUDE.md; task 3641)."
    )
