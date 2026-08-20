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

import shutil
import subprocess
from pathlib import Path

import pytest

SELFTEST_SH = Path(__file__).parent / "test_remove_lms_dropin.sh"


# ---------------------------------------------------------------------------
# step-1: RED -- the computed skip guard
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
    reason = _systemd_user_manager_skip_reason()  # noqa: F821  (step-2 defines it)
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
    reason = _systemd_user_manager_skip_reason()  # noqa: F821
    assert reason is not None
    assert "user manager" in reason

    def _missing_probe(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(2, "No such file or directory: 'systemctl'")

    monkeypatch.setattr(subprocess, "run", _missing_probe)
    reason = _systemd_user_manager_skip_reason()  # noqa: F821
    assert reason is not None
    assert "user manager" in reason

    def _timeout_probe(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="systemctl", timeout=5)

    monkeypatch.setattr(subprocess, "run", _timeout_probe)
    reason = _systemd_user_manager_skip_reason()  # noqa: F821
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
    reason = _systemd_user_manager_skip_reason()  # noqa: F821
    assert reason is not None
    assert "unit dir" in reason.lower() or "unit directory" in reason.lower()

    # (d) All three satisfied: the guarded tests MUST run.  A guard that
    # never returns None would skip everywhere and re-create the very rot
    # this module exists to end.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(writable_cfg))
    reason = _systemd_user_manager_skip_reason()  # noqa: F821
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
    assert _systemd_user_manager_skip_reason() is None  # noqa: F821
    assert (fake_home / ".config" / "systemd" / "user").is_dir()

    # And an UNSET XDG_CONFIG_HOME resolves the same way.
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert _systemd_user_manager_skip_reason() is None  # noqa: F821
