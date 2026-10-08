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

SCRIPT = Path(__file__).parent.parent / "legibility" / "install-trickle-timer.sh"
TEMPLATES_DIR = Path(__file__).parent.parent  # scripts/tests/../ = scripts/

# The unit names the PRODUCTION install absolutely -- systemd resolves
# Environment= absolutely and the installed unit is a byte copy of the
# committed one -- so neither may be derived from a REPO_ROOT, which is a
# .worktrees/<id> path when the suite runs in a lane. Precedent + rationale:
# scripts/tests/test_install_reclaim_orphaned_worktrees_timer.py::test_install_copies_units_enables_timer_and_kicks_drain,
# which asserts that byte-copy directly.
PRODUCTION_CLAUDE_DIR = "/home/leo/.local/bin"
# The account tokens the unit reads come from THE production checkout's .env,
# never a worktree's. The leading "-" makes it optional to systemd.
PRODUCTION_ENV_FILE = "-/home/leo/src/dark-factory/.env"


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


def test_service_template_pins_a_path_that_finds_claude():
    """Template-content invariant (NOT install behavior): the committed
    @.service must pin PATH to an absolute list that reaches ``claude``.

    The shared runner spawns the BARE name ``claude``, resolved against the
    child env's PATH, which it inherits from this unit (task 6042). Root cause
    this guards (2026-08-18): the ``systemd --user`` manager started at boot
    with NO ~/.local/bin on PATH -- the graphical login that imports
    ~/.profile came 84 minutes later -- and the timer's Persistent=true
    catch-up run fired into exactly that window, ENOENT'ing 6/6 selected
    digests on reify and 38/38 on dark_factory in about one second. The pin
    is load-bearing, not redundant belt-and-braces; do not "clean it up".
    """
    service_text = (TEMPLATES_DIR / "legibility-trickle@.service").read_text()
    env = _service_environment(service_text)

    assert "PATH" in env, (
        f"The trickle @.service must pin PATH= so claude resolves under a "
        f"systemd --user manager whose PATH lacks ~/.local/bin; parsed "
        f"[Service] Environment={env!r}"
    )
    entries = env["PATH"].split(":")
    assert all(os.path.isabs(entry) for entry in entries), (
        f"every PATH entry must be ABSOLUTE -- a relative one resolves "
        f"against the unit's cwd; got {entries!r}"
    )
    assert PRODUCTION_CLAUDE_DIR in entries, entries
    assert "/usr/bin" in entries, entries
    assert entries.index(PRODUCTION_CLAUDE_DIR) < entries.index("/usr/bin"), (
        f"{PRODUCTION_CLAUDE_DIR} must come ahead of /usr/bin, so the "
        f"operator's claude wins over any system copy; got {entries!r}"
    )

    assert "LEGIBILITY_CLAUDE_BIN" not in env, (
        "nothing reads LEGIBILITY_CLAUDE_BIN any more -- the shared runner "
        "resolves claude on PATH -- so a unit still setting it is a pin that "
        "silently does nothing"
    )
    assert "PYTHONPATH" in env, (
        f"The PYTHONPATH assignment must survive alongside the PATH pin; "
        f"parsed [Service] Environment={env!r}"
    )


def _service_directive(service_text, name):
    """Every value assigned to directive *name* in the ``[Service]`` section.

    A directive can legally repeat (systemd appends), so this returns a LIST.
    Sectioned parsing rather than a substring scan over the whole file, for
    the same reason ``_service_environment`` does it: a directive in [Unit]
    or [Install] is a different directive, and a test that cannot tell them
    apart would pass on a unit systemd reads differently.
    """
    values = []
    in_service = False
    for raw_line in service_text.splitlines():
        line = raw_line.strip()
        if line.startswith("["):
            in_service = line == "[Service]"
            continue
        if in_service and line.startswith(f"{name}="):
            values.append(line[len(name) + 1:].strip())
    return values


def test_service_template_carries_the_account_pool_and_pins_no_account():
    """Template-content invariant (NOT install behavior), task 5488: the unit
    must supply the account POOL and pin no single account out of it.

    Two positives (`EnvironmentFile=`, `UnsetEnvironment=ANTHROPIC_API_KEY`;
    why each is there: OPERATIONS.md §"Legibility trickle accounts (03:00)")
    and one negative, which is the load-bearing one: re-pinning one account
    here -- what the 2026-09-14 stopgap drop-in did with max-h -- would make
    the whole pool inert again while leaving every test above green.
    """
    service_text = (TEMPLATES_DIR / "legibility-trickle@.service").read_text()

    env_files = _service_directive(service_text, "EnvironmentFile")
    assert env_files == [PRODUCTION_ENV_FILE], (
        f"The trickle @.service must name the project .env as the source of "
        f"the CLAUDE_OAUTH_TOKEN_* pool (build_pool loads it in-process too; "
        f"the unit is where the dependency is stated), OPTIONALLY: a missing "
        f".env must not stop the unit starting, because build_pool turns an "
        f"empty pool into a recorded deferral (task 5635); got "
        f"EnvironmentFile={env_files!r}"
    )

    # The literal, not a constant read from shared: this asserts what systemd
    # must UNSET, and the shared runner's matching strip
    # (shared/src/shared/cli_invoke.py::_invoke_claude) is a separate
    # mechanism for a separate process. Two independent guards of one policy
    # is the intent (docs/code-quality.md heuristic 10), not a duplication.
    assert _service_directive(service_text, "UnsetEnvironment") == [
        "ANTHROPIC_API_KEY"
    ], (
        "The trickle @.service must unset ANTHROPIC_API_KEY: the CLI prefers "
        "it over the OAuth token, so leaving it set defeats the account "
        "choice silently"
    )

    # Over DIRECTIVES, never the raw file: the comments are allowed -- and
    # want -- to name the variable they explain. Not over the parsed
    # `Environment=` dict either, because the shape being excluded is exactly
    # what the 2026-09-14 drop-in used: an `ExecStart=` reset carrying the
    # assignment inline, which no Environment= parser would ever see.
    directives = "\n".join(
        line for line in service_text.splitlines()
        if not line.lstrip().startswith(("#", ";"))
    )
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in directives, (
        "The unit must pin NO single account -- choosing one per invocation "
        "is the gate's job now. Re-pinning here reintroduces exactly the "
        "failure this task removes (the 2026-09-14 max-h drop-in), and does "
        "it invisibly: every other assertion in this file stays green."
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


# ---------------------------------------------------------------------------
# task 5488: the installer RETIRES the stale account-pin drop-in
#
# NOT cosmetic. A drop-in's `ExecStart=` reset REPLACES the unit's ExecStart
# entirely, and the 2026-09-14 stopgap
# (~/.config/systemd/user/legibility-trickle@.service.d/10-account-pin.conf)
# did exactly that, re-spelling ExecStart with
# CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_OAUTH_TOKEN_H} inline. Leave it installed
# and the trickle stays pinned to one account forever while every change in
# this task sits silently inert -- the unit file on disk saying one thing and
# the unit systemd runs saying another.
#
# test_remove_lms_dropin_wrapper.py is the in-repo precedent for this shape.
# ---------------------------------------------------------------------------

STALE_DROPIN_NAME = "10-account-pin.conf"


def _dropin_dir(xdg_config):
    return xdg_config / "systemd" / "user" / "legibility-trickle@.service.d"


def _seed_stale_dropin(xdg_config):
    """Write a byte-faithful copy of the 2026-09-14 stopgap drop-in."""
    dropin_dir = _dropin_dir(xdg_config)
    dropin_dir.mkdir(parents=True, exist_ok=True)
    path = dropin_dir / STALE_DROPIN_NAME
    path.write_text(
        "[Service]\n"
        "EnvironmentFile=/home/leo/src/dark-factory/.env\n"
        "UnsetEnvironment=ANTHROPIC_API_KEY\n"
        "ExecStart=\n"
        "ExecStart=/usr/bin/env CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_OAUTH_TOKEN_H} "
        "/home/leo/.local/bin/uv run --frozen --project shared python "
        "scripts/legibility/nightly.py run --project-id %i\n",
        encoding="utf-8",
    )
    return path


def test_install_removes_the_stale_account_pin_dropin(tmp_path):
    search_root = _write_project_config(tmp_path, project_id="proj_a")
    xdg_config = tmp_path / "xdg-config"
    stale = _seed_stale_dropin(xdg_config)

    result = _run_script(
        tmp_path, "proj_a",
        env={"XDG_CONFIG_HOME": str(xdg_config), "LEGIBILITY_SEARCH_ROOTS": str(search_root)},
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert not stale.exists(), (
        "the stale account-pin drop-in survived the install -- its ExecStart= "
        "reset replaces the unit's own, so the trickle stays pinned to one "
        "account and task 5488 is silently inert"
    )
    assert not _dropin_dir(xdg_config).exists(), (
        "an emptied .service.d directory must go too: systemd lists it as a "
        "drop-in path forever otherwise, and the next reader cannot tell an "
        "empty override directory from a live one"
    )
    assert STALE_DROPIN_NAME in result.stdout, (
        "removing a live override is a change an operator must see in the "
        "install output, not a silent side effect"
    )


def test_install_is_idempotent_with_no_dropin_present(tmp_path):
    """The removal must be a no-op when there is nothing to remove.

    Under `set -euo pipefail` a bare `rm`/`rmdir` of a missing path exits
    non-zero and would abort the whole install -- turning a first-time setup
    on a machine that never had the stopgap into a hard failure.
    """
    search_root = _write_project_config(tmp_path, project_id="proj_a")
    xdg_config = tmp_path / "xdg-config"
    env = {"XDG_CONFIG_HOME": str(xdg_config), "LEGIBILITY_SEARCH_ROOTS": str(search_root)}

    first = _run_script(tmp_path, "proj_a", env=env)
    assert first.returncode == 0, f"stdout={first.stdout!r} stderr={first.stderr!r}"

    _seed_stale_dropin(xdg_config)
    second = _run_script(tmp_path, "proj_a", env=env)
    assert second.returncode == 0

    third = _run_script(tmp_path, "proj_a", env=env)
    assert third.returncode == 0, (
        f"a re-run with the drop-in already gone must still exit 0; "
        f"stdout={third.stdout!r} stderr={third.stderr!r}"
    )
    assert STALE_DROPIN_NAME not in third.stdout, (
        "only an install that actually removed something may say so"
    )


def test_install_removes_only_the_named_dropin(tmp_path):
    """The guard that matters: an operator's OWN override must survive.

    A blanket `rm -rf .service.d` would silently delete a deliberate local
    customization -- and the operator would only discover it the next time the
    behaviour it encoded failed to happen.
    """
    search_root = _write_project_config(tmp_path, project_id="proj_a")
    xdg_config = tmp_path / "xdg-config"
    stale = _seed_stale_dropin(xdg_config)
    operator_override = _dropin_dir(xdg_config) / "20-operator-nice.conf"
    operator_override.write_text("[Service]\nNice=10\n", encoding="utf-8")

    result = _run_script(
        tmp_path, "proj_a",
        env={"XDG_CONFIG_HOME": str(xdg_config), "LEGIBILITY_SEARCH_ROOTS": str(search_root)},
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert not stale.exists()
    assert operator_override.is_file(), (
        "an unrelated drop-in must never be swept up by the retirement"
    )
    assert operator_override.read_text() == "[Service]\nNice=10\n"


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
