"""Tests for scripts/legibility/install-trickle-timer.sh — the real
install script that replaces the fail-loud stub committed with
plans/confusion-reduction-prd.md (PRD task epsilon).

Drives the script via subprocess with a FAKE `systemctl` shimmed onto PATH
(records every invocation, minus `--user`, into a shared JSON state file —
mirrors scripts/tests/test_deploy_w5_recon_reliability.py and
scripts/tests/test_deploy_w11_lane_lifecycle.py) plus a real invocation of
`nightly.py resolve-config` (the REAL Python resolver — no mock — with
INSTALL_TRICKLE_TIMER_PYTHON=sys.executable substituting for the default `uv
run` prefix, mirroring watcher-rearm.sh's WATCHER_REARM_PYTHON override, and
LEGIBILITY_SEARCH_ROOTS pointed at a tmp fixture project). Real systemd is
never touched.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Resolved via scripts/tests/conftest.py's sys.path insertion (scripts/ AND
# scripts/legibility/ are both on sys.path; no package __init__ needed) --
# same idiom as test_legibility_coder.py's `import coder as mod`.
import coder

SCRIPT = Path(__file__).parent.parent / "legibility" / "install-trickle-timer.sh"
TEMPLATES_DIR = Path(__file__).parent.parent  # scripts/tests/../ = scripts/

# The unit names the PRODUCTION checkout absolutely -- systemd resolves
# Environment= absolutely and the installed unit is a byte copy of the
# committed one -- so this must NOT be derived from a REPO_ROOT, which is a
# .worktrees/<id> path when the suite runs in a lane. Precedent + rationale:
# scripts/tests/test_install_reify_closure_staleness_sweep_timer.py:27-32.
PRODUCTION_CLAUDE_BIN = "/home/leo/.local/bin/claude"


# ---------------------------------------------------------------------------
# Fake systemctl (marker-file + canned `enable`/`list-timers` responses)
# ---------------------------------------------------------------------------

_FAKE_SYSTEMCTL_SRC = '''#!/usr/bin/env python3
"""Fake `systemctl` for testing install-trickle-timer.sh.

Records every invocation (minus `--user`) into a JSON state file at
$FAKE_SYSTEMCTL_STATE. `enable --now <unit>` marks each non-flag arg as an
enabled unit; `list-timers` echoes back one line per *.timer unit enabled so
far THIS RUN, unless FAKE_SYSTEMCTL_OMIT_LIST_TIMERS=1 -- simulating a
self-verify failure where `enable` nominally succeeded but the unit is
absent from `list-timers`.
"""
import json
import os
import sys

STATE_PATH = os.environ["FAKE_SYSTEMCTL_STATE"]


def _load():
    with open(STATE_PATH) as f:
        return json.load(f)


def _save(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def main(argv):
    args = [a for a in argv[1:] if a != "--user"]
    if not args:
        return 1
    verb, rest = args[0], args[1:]

    state = _load()
    state.setdefault("calls", []).append(args)

    if verb == "daemon-reload":
        _save(state)
        return 0

    if verb == "enable":
        units = [a for a in rest if not a.startswith("-")]
        enabled = state.setdefault("enabled_timers", [])
        for u in units:
            if u not in enabled:
                enabled.append(u)
        _save(state)
        return 0

    if verb == "list-timers":
        _save(state)
        if os.environ.get("FAKE_SYSTEMCTL_OMIT_LIST_TIMERS") == "1":
            print("0 timers listed.")
            return 0
        enabled = state.get("enabled_timers", [])
        for unit in enabled:
            service = unit.replace(".timer", ".service")
            print(f"Mon 2026-07-14 03:00:00 UTC  8h left  n/a  n/a  {unit}  {service}")
        print(f"{len(enabled)} timers listed.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''


def _fake_systemctl(tmp_path):
    """Write an executable fake `systemctl` into <tmp_path>/bin/systemctl and
    its backing JSON state file. Returns (bin_dir, state_path)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "systemctl"
    fake.write_text(_FAKE_SYSTEMCTL_SRC)
    fake.chmod(0o755)

    state_path = tmp_path / "systemctl_state.json"
    state_path.write_text(json.dumps({"calls": [], "enabled_timers": []}))
    return bin_dir, state_path


def _systemctl_calls(tmp_path):
    state_path = tmp_path / "systemctl_state.json"
    if not state_path.is_file():
        return []
    return json.loads(state_path.read_text())["calls"]


# ---------------------------------------------------------------------------
# Fixture project (a real docs/legibility/legibility.yaml, resolved by the
# REAL nightly.py resolve-config through LEGIBILITY_SEARCH_ROOTS)
# ---------------------------------------------------------------------------

def _write_project_config(tmp_path, *, project_id, escalation_port=8199):
    """Write a minimal valid legibility.yaml for *project_id* under a fresh
    search root, returning that search root (the PARENT dir
    resolve_config_path globs one level down from) -- mirrors
    test_legibility_nightly.py's _write_config/env-override fixture shape."""
    search_root = tmp_path / "search-root"
    project_root = search_root / project_id
    legibility_dir = project_root / "docs" / "legibility"
    legibility_dir.mkdir(parents=True, exist_ok=True)
    config_path = legibility_dir / "legibility.yaml"
    config_path.write_text(
        f"project_id: {project_id}\n"
        f"project_root: {project_root}\n"
        f"escalation_port: {escalation_port}\n"
        f"cwd_prefixes:\n"
        f"  - {project_root}\n",
        encoding="utf-8",
    )
    return search_root


# ---------------------------------------------------------------------------
# Script driver
# ---------------------------------------------------------------------------

def _run_script(tmp_path, project_id, *, env=None):
    """Run install-trickle-timer.sh <project_id> via subprocess.

    Puts a fresh fake `systemctl` on PATH (state reset each call) and
    defaults INSTALL_TRICKLE_TIMER_PYTHON=sys.executable so the script's
    `nightly.py resolve-config` delegation runs the REAL resolver directly
    -- no `uv run`/`--frozen` needed in-test. Callers supply
    XDG_CONFIG_HOME and LEGIBILITY_SEARCH_ROOTS via *env*.
    """
    bin_dir, state_path = _fake_systemctl(tmp_path)

    full_env = dict(os.environ)
    full_env["PATH"] = f"{bin_dir}{os.pathsep}{full_env['PATH']}"
    full_env["FAKE_SYSTEMCTL_STATE"] = str(state_path)
    full_env["INSTALL_TRICKLE_TIMER_PYTHON"] = sys.executable
    full_env.pop("LEGIBILITY_SEARCH_ROOTS", None)
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), project_id],
        env=full_env, capture_output=True, text=True, timeout=30,
    )


# ---------------------------------------------------------------------------
# step-25: RED -- install-trickle-timer.sh
# ---------------------------------------------------------------------------

def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), (
        f"Expected {SCRIPT} to be executable (os.X_OK); it is not. "
        f"Run: chmod +x {SCRIPT}"
    )


def test_install_copies_templates_and_enables_timer(tmp_path):
    search_root = _write_project_config(tmp_path, project_id="proj_a")
    xdg_config = tmp_path / "xdg-config"

    result = _run_script(
        tmp_path, "proj_a",
        env={"XDG_CONFIG_HOME": str(xdg_config), "LEGIBILITY_SEARCH_ROOTS": str(search_root)},
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    unit_dir = xdg_config / "systemd" / "user"
    service_path = unit_dir / "legibility-trickle@.service"
    timer_path = unit_dir / "legibility-trickle@.timer"
    assert service_path.is_file(), f"Expected {service_path} to exist after install"
    assert timer_path.is_file(), f"Expected {timer_path} to exist after install"
    assert service_path.read_bytes() == (TEMPLATES_DIR / "legibility-trickle@.service").read_bytes()
    assert timer_path.read_bytes() == (TEMPLATES_DIR / "legibility-trickle@.timer").read_bytes()

    # The installed @.service's ExecStart references nightly.py + the %i
    # instance placeholder (systemd itself expands %i at instantiation
    # time -- the template is copied verbatim, never per-instance edited).
    service_text = service_path.read_text()
    assert "nightly.py" in service_text
    assert "--project-id %i" in service_text

    calls = _systemctl_calls(tmp_path)
    assert ["daemon-reload"] in calls, f"calls={calls!r}"
    assert ["enable", "--now", "legibility-trickle@proj_a.timer"] in calls, f"calls={calls!r}"
    assert any(c[0] == "list-timers" for c in calls), f"calls={calls!r}"


def _service_environment(service_text):
    """Parse the [Service] section's `Environment=` assignments into a dict.

    Scans from the `[Service]` header to the next section header, and
    shlex.split()s each directive's right-hand side before splitting every
    token on its FIRST `=`. The shlex step handles BOTH the
    one-assignment-per-line form this template uses and systemd's legal
    space-separated multi-assignment form, so a future reflow of the unit
    onto a single Environment= line does not silently stop being checked.
    """
    env = {}
    in_service = False
    for raw_line in service_text.splitlines():
        line = raw_line.strip()
        if line.startswith("["):
            in_service = line == "[Service]"
            continue
        if not in_service or not line.startswith("Environment="):
            continue
        for token in shlex.split(line[len("Environment="):]):
            name, sep, value = token.partition("=")
            if sep:
                env[name] = value
    return env


def test_service_template_pins_claude_bin():
    """Template-content invariant (NOT install behavior): the committed
    @.service must pin LEGIBILITY_CLAUDE_BIN to an ABSOLUTE `claude` path.

    Root cause this guards (2026-08-18): coder.py resolves the CLI as
    `claude_bin or os.environ.get(_CLAUDE_BIN_ENV_VAR) or "claude"`, so
    without this pin the bare name is execvp-resolved against whatever PATH
    the `systemd --user` manager happens to hold. That manager started at
    boot with NO ~/.local/bin on PATH -- the graphical login that imports
    ~/.profile came 84 minutes later -- and the timer's Persistent=true
    catch-up run fired into exactly that window, ENOENT'ing 6/6 selected
    digests on reify and 38/38 on dark_factory in about one second. The pin
    is load-bearing, not redundant belt-and-braces; do not "clean it up".
    """
    service_text = (TEMPLATES_DIR / "legibility-trickle@.service").read_text()
    env = _service_environment(service_text)

    # Read the env-var NAME from coder rather than hardcoding the literal:
    # the pin is only useful because coder.py reads that exact variable, so
    # a rename there must fail HERE instead of leaving a live unit setting a
    # var nothing reads.
    assert coder._CLAUDE_BIN_ENV_VAR in env, (
        f"The trickle @.service must pin {coder._CLAUDE_BIN_ENV_VAR}= so the "
        f"coder survives a systemd --user manager whose PATH lacks "
        f"~/.local/bin; parsed [Service] Environment={env!r}"
    )
    value = env[coder._CLAUDE_BIN_ENV_VAR]
    assert value == PRODUCTION_CLAUDE_BIN, (
        f"Expected {coder._CLAUDE_BIN_ENV_VAR}={PRODUCTION_CLAUDE_BIN!r}, "
        f"got {value!r}"
    )
    # Asserted independently of the literal above: the entire point of the
    # fix is surviving a PATH gap, so an edit back to a bare `claude` must
    # fail even if someone also changes PRODUCTION_CLAUDE_BIN.
    assert os.path.isabs(value), (
        f"{coder._CLAUDE_BIN_ENV_VAR} must be an ABSOLUTE path (a bare name "
        f"would be PATH-resolved, which is the failure being fixed); "
        f"got {value!r}"
    )

    # Adding a second Environment= line must not displace the first.
    assert "PYTHONPATH" in env, (
        f"The pre-existing PYTHONPATH assignment must survive alongside the "
        f"claude-bin pin; parsed [Service] Environment={env!r}"
    )


def test_install_is_idempotent(tmp_path):
    search_root = _write_project_config(tmp_path, project_id="proj_a")
    xdg_config = tmp_path / "xdg-config"
    env = {"XDG_CONFIG_HOME": str(xdg_config), "LEGIBILITY_SEARCH_ROOTS": str(search_root)}

    result_1 = _run_script(tmp_path, "proj_a", env=env)
    assert result_1.returncode == 0, (
        f"stdout={result_1.stdout!r} stderr={result_1.stderr!r}"
    )

    result_2 = _run_script(tmp_path, "proj_a", env=env)
    assert result_2.returncode == 0, (
        f"Expected re-running the install to still exit 0; "
        f"stdout={result_2.stdout!r} stderr={result_2.stderr!r}"
    )


def test_install_fails_when_self_verify_omits_timer(tmp_path):
    search_root = _write_project_config(tmp_path, project_id="proj_a")
    xdg_config = tmp_path / "xdg-config"

    result = _run_script(
        tmp_path, "proj_a",
        env={
            "XDG_CONFIG_HOME": str(xdg_config),
            "LEGIBILITY_SEARCH_ROOTS": str(search_root),
            "FAKE_SYSTEMCTL_OMIT_LIST_TIMERS": "1",
        },
    )

    assert result.returncode != 0, (
        f"Expected a non-zero exit when list-timers omits the enabled timer "
        f"(self-verify failure); stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_install_fails_when_project_config_unresolvable(tmp_path):
    xdg_config = tmp_path / "xdg-config"
    empty_search_root = tmp_path / "empty-search-root"
    empty_search_root.mkdir()

    result = _run_script(
        tmp_path, "no_such_project",
        env={"XDG_CONFIG_HOME": str(xdg_config), "LEGIBILITY_SEARCH_ROOTS": str(empty_search_root)},
    )

    assert result.returncode != 0, (
        f"Expected a non-zero exit for an unresolvable project_id; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert _systemctl_calls(tmp_path) == [], (
        f"Expected NO systemctl invocation when config resolution fails first; "
        f"calls={_systemctl_calls(tmp_path)!r}"
    )
    assert not (xdg_config / "systemd" / "user" / "legibility-trickle@.service").exists(), (
        "Expected no unit file to be installed when config resolution fails"
    )
