"""Pytest driver for scripts/tests/test_remove_lms_dropin.sh (task 4200).

That shell self-test was written under task 3750 to exercise the success path
of scripts/remove-lms-arm-worktree-dropin.sh -- a script that runs UNSANDBOXED
and UNATTENDED as a deterministic `before_done` deploy action -- somewhere
other than production.  It was correct, it was green, and NOTHING RAN IT: it
was the only .sh among 61 pytest modules in scripts/tests/, and pytest collects
only `test_*.py`.  A gate nobody runs rots silently, and the first person to
learn it had rotted would have been production.  This module is the collected
caller that closes that gap.

It is deliberately NOT `@pytest.mark.integration`.  The root pyproject.toml
addopts deselect that marker, so an integration-marked gate would be absent
from every default `verify` run -- present in the tree, checked by nobody,
which is verbatim the defect this module exists to fix.  See
scripts/tests/test_lms_verification_artifact.py:25-29 for the same reasoning
recorded for the same reason.  Instead the systemd-dependent tests are gated on
a COMPUTED skip reason, so they run for real wherever a --user manager is
reachable and skip cleanly (never error) in a sandbox or CI container that has
no session bus.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest

SELFTEST_SH = Path(__file__).parent / "test_remove_lms_dropin.sh"


# ---------------------------------------------------------------------------
# The computed skip guard
# ---------------------------------------------------------------------------

def _unit_dir() -> Path:
    """Resolve the systemd --user unit dir EXACTLY as the .sh resolves it.

    The .sh (and the script under test) both spell this
    `${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user`.  `:-` treats an EMPTY
    value as absent, so this checks truthiness rather than key membership --
    otherwise an exported-but-empty XDG_CONFIG_HOME would resolve a relative
    path under the CWD and probe a directory the .sh will never touch.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "systemd" / "user"


def _systemd_user_manager_skip_reason() -> str | None:
    """Return why the systemd-backed tests must skip, or None to run them.

    Three independent preconditions, checked in order.  Each returned string
    says why the skip is an ENVIRONMENT FACT rather than a defect, so a
    skipped verify log is not mistaken for a silently disabled gate.

    1. ``systemctl`` must be on PATH.
    2. The --user MANAGER must be reachable.  Probed with
       ``systemctl --user show -p Version``, deliberately NOT
       ``is-system-running``: the latter reports non-zero for a merely
       ``degraded`` manager, which the operator host currently is and which
       is perfectly usable for the .sh (it only ever ``daemon-reload``s), so
       gating on it would over-skip on exactly the hosts where the test
       works.  Any non-zero rc, FileNotFoundError/OSError or timeout counts
       as unreachable -- a container with no DBUS_SESSION_BUS_ADDRESS or
       XDG_RUNTIME_DIR produces the first, a stripped image the others.
    3. The unit dir must be genuinely WRITABLE, confirmed by creating and
       unlinking a real probe file rather than by trusting
       ``os.access(..., os.W_OK)``.  os.access reports the mode bits, not a
       sandbox/LSM denial -- and the observed failure in dark-factory task
       worktrees is exactly that shape: files are user-owned and rw, yet the
       write is refused ("install: cannot remove ...: Permission denied").
       A mode-bit check would report writable, the guard would not fire, and
       the wrapper would ERROR instead of skipping.
    """
    if shutil.which("systemctl") is None:
        return (
            "no systemctl on this PATH -- scripts/tests/test_remove_lms_dropin.sh "
            "drives a real systemd --user manager and cannot run here.  This is "
            "an environment fact (sandbox/CI container), not a defect in the "
            "script under test"
        )

    try:
        probe = subprocess.run(
            ["systemctl", "--user", "show", "-p", "Version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        reachable = probe.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        # FileNotFoundError is an OSError subclass, so a systemctl that
        # vanished between the which() above and here lands here too.
        reachable = False
    if not reachable:
        return (
            "no reachable systemd --user manager (no session bus / "
            "XDG_RUNTIME_DIR, or the probe timed out) -- the .sh installs a "
            "throwaway template and asserts against `systemctl --user show`, "
            "which needs a live user manager.  Environment fact, not a defect"
        )

    unit_dir = _unit_dir()
    try:
        unit_dir.mkdir(parents=True, exist_ok=True)
        write_probe = unit_dir / f".lms-selftest-writeprobe-{os.getpid()}-{uuid4().hex[:8]}"
        write_probe.write_text("")
        write_probe.unlink()
    except OSError as exc:
        return (
            f"the systemd user unit directory {unit_dir} is not writable "
            f"({exc}) -- the .sh must install its throwaway template there "
            "because the --user manager resolves its unit search path from "
            "its OWN environment, so the dir cannot be redirected to a tmp "
            "path.  Environment fact (agent write-scope), not a defect"
        )

    return None


_SKIP_REASON = _systemd_user_manager_skip_reason()


# ---------------------------------------------------------------------------
# Per-invocation isolation: a unique throwaway template name
# ---------------------------------------------------------------------------

# The single shared constant.  _unique_template() GENERATES names with this
# prefix and _prune_stale_selftest_units() only ever DELETES names with it, so
# "the prune can never touch a real unit" is structural rather than a pair of
# string literals free to drift apart.  It cannot prefix-match `lms-arm@` (the
# real unit the script under test targets) or any dark-factory unit.
_SELFTEST_PREFIX = "lms-dropin-selftest-"


def _unique_template() -> str:
    """Return a fresh throwaway systemd TEMPLATE name for one .sh invocation.

    BOTH components are load-bearing:

    * ``os.getpid()`` disambiguates CONCURRENT pytest processes.  The fleet
      runs ``max_concurrent_tasks: 48`` and all 48 share one $HOME, hence one
      ~/.config/systemd/user.  Measured at plan time with the .sh's old
      hardcoded ``lms-dropin-selftest@``: two concurrent runs BOTH fail
      deterministically, because case 4's ``rm -f "$UNIT"`` tears down the
      other run's fixture mid-test ("drop-in still present after refusal",
      "template survives the re-run", and a WorkingDirectory that resolves to
      the OTHER run's mktemp dir).
    * The uuid suffix disambiguates repeat calls WITHIN one process -- two
      tests in this module each run the .sh -- and survives PID reuse after a
      killed run left same-PID residue behind.

    The trailing "@" makes it a template: the .sh appends ".service" and
    builds its probe unit as ``${TEMPLATE}probe.service``.
    """
    return f"{_SELFTEST_PREFIX}{os.getpid()}-{uuid4().hex[:8]}@"


# How long a selftest unit must go untouched before it counts as abandoned.
# Comfortably longer than the ~2s a run takes and than the suite's own
# --timeout=300, so a live sibling is never mistaken for residue.
_STALE_AFTER_S = 3600.0


def _prune_stale_selftest_units(
    unit_dir: Path,
    *,
    now: float,
    max_age_s: float = _STALE_AFTER_S,
) -> None:
    """Best-effort removal of abandoned lms-dropin-selftest-* residue.

    The .sh cleans up after itself on every exit path via `trap cleanup EXIT`,
    but a SIGKILL has no exit path.  With the old fixed template name that was
    self-healing; with per-invocation unique names it is not, so this pays the
    cost explicitly.

    TWO SAFETY PROPERTIES, each with its own test above, because this deletes
    files out of the operator's LIVE unit directory:

    * Only ``_SELFTEST_PREFIX``-named units are ever considered -- the same
      constant ``_unique_template()`` generates with, so a real ``lms-arm@``
      or fleet unit is out of scope structurally, at any age.
    * Anything NEWER than max_age_s is left alone.  Under 48-way concurrency a
      fresh selftest unit is a running sibling's fixture, and deleting it
      would destroy that run.

    ``now`` and ``max_age_s`` are parameters rather than clock reads so the
    tests can drive this deterministically without sleeping or patching time.
    The whole body is best-effort: this is opportunistic hygiene called before
    the real work, and it must never be the thing that turns a green suite red.
    """
    try:
        if not unit_dir.is_dir():
            return
        for path in sorted(unit_dir.iterdir()):
            if not path.name.startswith(_SELFTEST_PREFIX):
                continue
            if not path.name.endswith((".service", ".service.d")):
                continue
            try:
                if now - path.stat().st_mtime <= max_age_s:
                    continue
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
            except OSError:
                # One unreadable/undeletable entry must not abort the sweep of
                # the rest -- residue accumulates, so partial progress matters.
                continue
    except OSError:
        return


# ---------------------------------------------------------------------------
# step-1: the computed skip guard
# ---------------------------------------------------------------------------

def test_skip_reason_gates_on_binary_manager_and_writable_unit_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """_systemd_user_manager_skip_reason must gate on ALL THREE preconditions.

    Every case below is host-independent: the binary lookup, the --user
    manager probe and the unit directory are all monkeypatched rather than
    read from this host's real environment, so this test asserts the same
    thing on a live systemd box and in a DBUS-less container.  That is what
    keeps this module from being a total no-op in a sandbox -- the guard
    itself is genuinely exercised everywhere, and only the two subprocess
    tests skip.  Mirrors tests/scripts/test_dashboard_service_template.py::
    test_systemd_analyze_skip_reason_requires_both_binary_and_user_runtime_dir.

      (a) systemctl absent from PATH.
      (b) systemctl present, --user manager unreachable -- in all three ways
          it can be unreachable: non-zero rc, FileNotFoundError, and a
          probe that times out.
      (c) systemctl present and the manager reachable, but the unit dir
          cannot be written.  NOT hypothetical: agent write-scope in
          dark-factory task worktrees has been observed denying writes to
          ~/.config/systemd/user ("install: cannot remove ...: Permission
          denied") even for user-owned files.  Without this case the wrapper
          would ERROR rather than skip in such a sandbox -- the opposite of
          this module's "stay green in sandboxes and CI" contract.
      (d) all three satisfied -- the guarded tests must actually RUN.

    Assertions are on `is None` / `is not None` plus a substring, never on
    exact prose, so the reason strings stay free to explain themselves.
    """
    writable_cfg = tmp_path / "cfg"
    writable_cfg.mkdir()

    def _ok_probe(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["systemctl"], returncode=0, stdout="Version=255.4\n", stderr=""
        )

    # (a) Binary absent -- gated even with a reachable manager and a
    # perfectly writable unit dir alongside it.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(writable_cfg))
    monkeypatch.setattr(subprocess, "run", _ok_probe)
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: None)
    reason = _systemd_user_manager_skip_reason()
    assert reason is not None
    assert "systemctl" in reason

    # (b) Binary present, manager unreachable -- three distinct failure
    # modes, because a container with no DBUS_SESSION_BUS_ADDRESS /
    # XDG_RUNTIME_DIR produces the non-zero-rc one while a stripped image
    # produces the others, and all three must skip rather than error.
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/systemctl")

    def _rc1_probe(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["systemctl"],
            returncode=1,
            stdout="",
            stderr="Failed to connect to bus: No medium found\n",
        )

    monkeypatch.setattr(subprocess, "run", _rc1_probe)
    reason = _systemd_user_manager_skip_reason()
    assert reason is not None
    assert "user manager" in reason

    def _missing_probe(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(2, "No such file or directory: 'systemctl'")

    monkeypatch.setattr(subprocess, "run", _missing_probe)
    reason = _systemd_user_manager_skip_reason()
    assert reason is not None
    assert "user manager" in reason

    def _timeout_probe(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="systemctl", timeout=5)

    monkeypatch.setattr(subprocess, "run", _timeout_probe)
    reason = _systemd_user_manager_skip_reason()
    assert reason is not None
    assert "user manager" in reason

    # (c) Binary present and manager reachable, but the unit dir is
    # unwritable.  Rendered here by pointing XDG_CONFIG_HOME at a regular
    # FILE, so "<file>/systemd/user" can never be created -- a denial that
    # holds for any uid, unlike a chmod, which root would sail straight
    # through and turn this case into a silent no-op.
    not_a_dir = tmp_path / "cfg-is-a-file"
    not_a_dir.write_text("")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(not_a_dir))
    monkeypatch.setattr(subprocess, "run", _ok_probe)
    reason = _systemd_user_manager_skip_reason()
    assert reason is not None
    assert "unit dir" in reason.lower() or "unit directory" in reason.lower()

    # (d) All three satisfied: the guarded tests MUST run.  A guard that
    # never returns None would skip everywhere and re-create the very rot
    # this module exists to end.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(writable_cfg))
    reason = _systemd_user_manager_skip_reason()
    assert reason is None


def test_skip_reason_falls_back_to_home_config_like_the_shell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The unit dir must resolve EXACTLY as the .sh resolves it.

    The .sh uses `${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user`, whose `:-`
    treats an EMPTY value as absent.  A Python guard checking key membership
    instead of truthiness would resolve `""/systemd/user` -> a relative path
    under the CWD, probe a directory the .sh will never touch, and report the
    wrong answer in both directions.  Pinned host-independently against a
    tmp HOME.
    """
    fake_home = tmp_path / "home"
    (fake_home / ".config").mkdir(parents=True)
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/systemctl")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            args=["systemctl"], returncode=0, stdout="Version=255.4\n", stderr=""
        ),
    )
    monkeypatch.setenv("HOME", str(fake_home))

    # Empty XDG_CONFIG_HOME must fall back to $HOME/.config, exactly as `:-`
    # does -- so the writable tmp HOME below is what gets probed, and the
    # guard returns None.
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    assert _systemd_user_manager_skip_reason() is None
    assert (fake_home / ".config" / "systemd" / "user").is_dir()

    # And an UNSET XDG_CONFIG_HOME resolves the same way.
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert _systemd_user_manager_skip_reason() is None


# ---------------------------------------------------------------------------
# step-3: RED -- the per-invocation unique template name
# ---------------------------------------------------------------------------

def test_unique_template_is_a_legal_distinct_systemd_template_name() -> None:
    """_unique_template must produce a fresh, legal systemd TEMPLATE name.

    Isolation between concurrent runs comes from the unit NAME and nothing
    else: the systemd --user manager resolves its unit search path from its
    OWN environment, so pointing XDG_CONFIG_HOME at a tmp dir would give the
    fixture a directory the manager never reads.  The .sh must therefore keep
    writing to the real ~/.config/systemd/user, and every property below is
    what makes that safe under `max_concurrent_tasks: 48` sharing one $HOME.

      (a) Trailing "@".  The .sh appends ".service" and builds its probe as
          "${TEMPLATE}probe.service"; a missing "@" silently yields a plain
          unit instead of a template, and the probe unit resolves to nothing.
      (b) Two calls differ.  Both subprocess tests in this module run the .sh,
          and a shared name between them re-creates the collision this seam
          exists to prevent.
      (c) The shared _SELFTEST_PREFIX.  The generator and the prune bind the
          SAME constant, so the prune's "never touch a real unit" property is
          structural rather than a pair of string literals free to drift.
      (d) A legal systemd unit-name charset.  Verified viable at plan time:
          PID-suffixed template names resolve correctly against a real
          manager.

    No systemd required -- this runs everywhere.
    """
    name = _unique_template()

    # (a)
    assert name.endswith("@"), (
        f"_unique_template() must return a TEMPLATE name ending in '@'; got {name!r}. "
        "The .sh builds '${TEMPLATE}probe.service' from it."
    )
    # (b)
    assert _unique_template() != _unique_template()
    # (c)
    assert name.startswith(_SELFTEST_PREFIX)
    # (d)
    assert re.fullmatch(r"[A-Za-z0-9:_.\-]+@", name), (
        f"{name!r} is not a legal systemd template unit name; systemd accepts "
        r"only [A-Za-z0-9:_.\-] before the '@'."
    )


def test_selftest_prefix_is_shared_and_cannot_match_a_real_unit() -> None:
    """The one constant both the generator and the prune key on.

    Pinned as its own assertion because the prune deletes files out of the
    operator's LIVE unit directory: if _SELFTEST_PREFIX were ever widened to
    something a real unit could start with, the prune would sweep up
    production units and this module would become the outage it was written
    to prevent.  "lms-dropin-selftest-" cannot prefix-match `lms-arm@` (the
    real unit the script under test targets) or any dark-factory unit.
    """
    assert _SELFTEST_PREFIX == "lms-dropin-selftest-"
    for real in ("lms-arm@", "fused-memory", "dark-factory-dashboard", "orchestrator"):
        assert not real.startswith(_SELFTEST_PREFIX)


# ---------------------------------------------------------------------------
# step-5: RED -- the stale-residue prune
# ---------------------------------------------------------------------------

def _write_unit(unit_dir: Path, template: str, *, age_s: float, now: float) -> tuple[Path, Path]:
    """Install a <template>.service + <template>.service.d/ pair aged age_s.

    Mirrors the exact on-disk shape the .sh's install_fixture() leaves behind,
    so the prune is driven against real residue rather than a stand-in.
    """
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit = unit_dir / f"{template}.service"
    dropin_dir = unit_dir / f"{template}.service.d"
    dropin_dir.mkdir(exist_ok=True)
    unit.write_text("[Service]\nExecStart=/bin/true\n")
    (dropin_dir / "10-worktree-3713.conf").write_text("[Service]\nWorkingDirectory=/x\n")
    stamp = now - age_s
    for path in (unit, dropin_dir / "10-worktree-3713.conf", dropin_dir):
        os.utime(path, (stamp, stamp))
    return unit, dropin_dir


def test_prune_removes_stale_selftest_residue(tmp_path: Path) -> None:
    """(a) An OLD selftest unit and its drop-in dir are both removed.

    Uniquifying the template name costs exactly one thing, and this is it.
    The .sh's old FIXED name self-healed: a run killed before its EXIT trap
    fired left residue the next run simply overwrote and removed.  Unique
    names have no such self-healing -- a SIGKILLed run (verify timeout, task
    cancellation, both routine in this fleet) strands
    lms-dropin-selftest-<pid>-<hex>@.service in the operator's LIVE unit dir
    forever, and those accumulate, slowing every daemon-reload and polluting
    `systemctl --user list-unit-files`.  Driven against tmp_path with
    hand-set mtimes: no systemd, no sleeping, no real unit dir.
    """
    now = 1_000_000.0
    unit_dir = tmp_path / "systemd" / "user"
    unit, dropin_dir = _write_unit(
        unit_dir, f"{_SELFTEST_PREFIX}999-deadbeef@", age_s=7200.0, now=now
    )

    _prune_stale_selftest_units(unit_dir, now=now, max_age_s=3600.0)

    assert not unit.exists(), f"stale {unit.name} should have been pruned"
    assert not dropin_dir.exists(), f"stale {dropin_dir.name}/ should have been pruned"


def test_prune_leaves_a_fresh_selftest_unit_alone(tmp_path: Path) -> None:
    """(b) SAFETY -- a FRESH selftest unit belongs to a RUNNING sibling.

    Under 48-way concurrency another task's .sh may be mid-run right now, and
    its fixture is a selftest-prefixed unit that is only seconds old.
    Deleting it would destroy that run's fixture and turn a green sibling red
    -- reintroducing, from the cleanup side, the very collision the unique
    template name was introduced to eliminate.  Age is the only thing
    separating "my abandoned residue" from "someone else's live fixture".
    """
    now = 1_000_000.0
    unit_dir = tmp_path / "systemd" / "user"
    unit, dropin_dir = _write_unit(
        unit_dir, f"{_SELFTEST_PREFIX}12345-cafebabe@", age_s=5.0, now=now
    )

    _prune_stale_selftest_units(unit_dir, now=now, max_age_s=3600.0)

    assert unit.exists(), "a FRESH selftest unit belongs to a concurrent run and must survive"
    assert dropin_dir.exists(), "a FRESH selftest drop-in dir must survive"


def test_prune_never_touches_a_non_selftest_unit(tmp_path: Path) -> None:
    """(c) SAFETY -- a real unit is never swept up, at ANY age.

    This prune deletes files out of the operator's live
    ~/.config/systemd/user.  `lms-arm@` is the actual unit the script under
    test targets, and the dark-factory units below are the live fleet.  All
    are aged FAR beyond max_age_s -- age must be irrelevant for anything
    lacking the selftest prefix, or this module becomes the outage it was
    written to prevent.
    """
    now = 1_000_000.0
    unit_dir = tmp_path / "systemd" / "user"
    reals = [
        _write_unit(unit_dir, "lms-arm@", age_s=86_400.0 * 30, now=now),
        _write_unit(unit_dir, "fused-memory", age_s=86_400.0 * 90, now=now),
        _write_unit(unit_dir, "dark-factory-dashboard", age_s=86_400.0 * 90, now=now),
    ]

    _prune_stale_selftest_units(unit_dir, now=now, max_age_s=3600.0)

    for unit, dropin_dir in reals:
        assert unit.exists(), f"{unit.name} is a REAL unit and must never be pruned"
        assert dropin_dir.exists(), f"{dropin_dir.name}/ is REAL and must never be pruned"


def test_prune_is_silent_when_the_unit_dir_does_not_exist(tmp_path: Path) -> None:
    """A fresh checkout or a container with no ~/.config/systemd/user.

    The prune is opportunistic hygiene called before the real work; it must
    never be the thing that turns a green suite red.
    """
    _prune_stale_selftest_units(
        tmp_path / "nope" / "systemd" / "user", now=1_000_000.0, max_age_s=3600.0
    )


# ---------------------------------------------------------------------------
# step-7: the measured concurrency defect
# ---------------------------------------------------------------------------

# Comfortably under the suite's own --timeout=300 (scripts/orchestrator.yaml),
# so a wedged run surfaces as this test's own diagnosable failure rather than
# as an opaque suite-level timeout that names no cause.
_SELFTEST_TIMEOUT_S = 120


def _run_selftest(template: str) -> subprocess.CompletedProcess[str]:
    """Drive scripts/tests/test_remove_lms_dropin.sh against ONE template name.

    The template is threaded in through the LMS_SELFTEST_TEMPLATE seam, which
    the .sh then forwards to the script under test via its existing
    LMS_UNIT_TEMPLATE seam -- so this single variable propagates through both
    halves and gives the invocation an absolute unit path no concurrent run
    shares.
    """
    env = os.environ.copy()
    env["LMS_SELFTEST_TEMPLATE"] = template
    return subprocess.run(
        ["bash", str(SELFTEST_SH)],
        capture_output=True,
        text=True,
        timeout=_SELFTEST_TIMEOUT_S,
        env=env,
        check=False,
    )


def _remove_template_residue(template: str) -> None:
    """Delete one template's unit + drop-in dir from the REAL unit directory.

    The .sh cleans up after itself via `trap cleanup EXIT`; this is the
    belt-and-braces for the path where it cannot -- a mid-test failure or an
    exception raised between launch and assertion.  Best-effort by design.
    """
    unit_dir = _unit_dir()
    try:
        (unit_dir / f"{template}.service").unlink(missing_ok=True)
        shutil.rmtree(unit_dir / f"{template}.service.d", ignore_errors=True)
    except OSError:
        pass


def _assert_selftest_passed(result: subprocess.CompletedProcess[str], template: str) -> None:
    """Assert one run is green, surfacing its full output on failure.

    A future breakage of the script under test must be diagnosable from the
    verify log ALONE -- nobody will have a live systemd box and this worktree
    in hand when it goes red.  The .sh prints a PASS/FAIL line per check, so
    embedding stdout names the exact check that broke.
    """
    detail = (
        f"\n--- template: {template}\n"
        f"--- exit code: {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert result.returncode == 0, f"{SELFTEST_SH.name} exited non-zero.{detail}"
    assert "ALL CHECKS PASSED" in result.stdout, (
        f"{SELFTEST_SH.name} did not report ALL CHECKS PASSED.{detail}"
    )


@pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON))
def test_concurrent_selftests_do_not_collide() -> None:
    """Three simultaneous runs must all pass -- the MEASURED defect, not a guess.

    The fleet runs `max_concurrent_tasks: 48` and every task shares one $HOME,
    so this .sh can and will be running in several worktrees at once.  With
    its old hardcoded `lms-dropin-selftest@`, every instance wrote and `rm`ed
    the SAME absolute unit path, and case 4's `rm -f "$UNIT"` tore down the
    other run's fixture mid-test.  Measured at plan time: two concurrent runs
    BOTH failed, deterministically --

      run A: "exit code is 0 -- expected '0', got '1'"
             "WorkingDirectory now resolves to the repo root -- expected
              '/tmp/tmp.NnkWkdITs8', got '/tmp/tmp.bUjgix0thr'"
      run B: "drop-in still present after refusal -- expected 'yes', got 'no'"
             "template survives the re-run -- expected 'yes', got 'no'"

    That makes the LMS_SELFTEST_TEMPLATE seam MANDATORY rather than a nicety.
    scripts/orchestrator.yaml warns that a red leg here "blocks every merge,
    review checkpoint and main-tip sweep repo-wide, on branches with no
    defect" (the repo root sets merge_verify_breadth: full).  Wiring the .sh
    into the collected suite UNSEAMED would therefore have converted a
    dormant-but-correct test into an intermittent repo-wide merge blocker --
    strictly worse than the rot this task set out to fix.

    THREE instances rather than two, so the overlap window is not marginal.
    """
    templates = [_unique_template() for _ in range(3)]
    assert len(set(templates)) == 3, "the generator must not hand two runs the same name"

    try:
        with ThreadPoolExecutor(max_workers=len(templates)) as pool:
            results = list(pool.map(_run_selftest, templates))
        for template, result in zip(templates, results, strict=True):
            _assert_selftest_passed(result, template)
    finally:
        for template in templates:
            _remove_template_residue(template)


# ---------------------------------------------------------------------------
# step-9: the primary anti-rot gate
# ---------------------------------------------------------------------------

def test_shell_selftest_is_executable() -> None:
    """Expected GREEN on arrival -- a regression pin.

    The .sh is mode 100755 today and this module drives it through `bash` for
    portability, so the bit is not strictly load-bearing HERE -- but the .sh
    documents itself as directly runnable by an operator
    ("Usage: scripts/tests/test_remove_lms_dropin.sh"), and a lost +x silently
    breaks that contract.  Mirrors
    test_reclaim_orphaned_worktrees_wrapper.py::test_wrapper_is_executable.
    """
    assert SELFTEST_SH.is_file(), f"{SELFTEST_SH} is missing"
    assert os.access(SELFTEST_SH, os.X_OK), (
        f"Expected {SELFTEST_SH} to be executable (os.X_OK); it is not. "
        f"Run: chmod +x {SELFTEST_SH}"
    )


@pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON))
def test_shell_selftest_passes() -> None:
    """THE assertion this task exists to create.  Expected GREEN on arrival.

    There is no RED to manufacture here and the absence of one is the point:
    scripts/tests/test_remove_lms_dropin.sh was already correct -- 12/12
    checks, ~2s -- and the defect was that NOTHING RAN IT.  It was the only
    .sh among 61 pytest modules in scripts/tests/, and pytest collects only
    `test_*.py`, so the file sat in the tree checked by nobody.  The fix is
    therefore the EXISTENCE of a collected caller, and this is it.  Same shape
    as tests/scripts/test_know_live_installed_unit_parity.py's
    "Expected GREEN on arrival -- a regression pin".

    What it protects: scripts/remove-lms-arm-worktree-dropin.sh runs
    UNSANDBOXED and UNATTENDED as a deterministic `before_done` deploy action.
    Its own TEST SEAMS comment says such a script "must not have its happy
    path first execute in production" -- a guarantee that, until this module,
    was delivered by an operator remembering to run a file rather than by
    anything mechanical.  From here the verify gate delivers it.

    Safety of running for real in the DEFAULT suite: the .sh only ever
    `daemon-reload`s -- it never starts, stops, enables or restarts any unit
    -- and it operates solely on a uniquely-named throwaway template sharing
    no name with any real unit.  Cost is ~2s against a suite measured at
    293-931s.
    """
    _prune_stale_selftest_units(_unit_dir(), now=time.time())

    template = _unique_template()
    try:
        _assert_selftest_passed(_run_selftest(template), template)
    finally:
        _remove_template_residue(template)
