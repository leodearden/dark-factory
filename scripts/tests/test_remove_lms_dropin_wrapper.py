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
    name = _unique_template()  # noqa: F821  (step-4 defines it)

    # (a)
    assert name.endswith("@"), (
        f"_unique_template() must return a TEMPLATE name ending in '@'; got {name!r}. "
        "The .sh builds '${TEMPLATE}probe.service' from it."
    )
    # (b)
    assert _unique_template() != _unique_template()  # noqa: F821
    # (c)
    assert name.startswith(_SELFTEST_PREFIX)  # noqa: F821
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
    assert _SELFTEST_PREFIX == "lms-dropin-selftest-"  # noqa: F821
    for real in ("lms-arm@", "fused-memory", "dark-factory-dashboard", "orchestrator"):
        assert not real.startswith(_SELFTEST_PREFIX)  # noqa: F821
