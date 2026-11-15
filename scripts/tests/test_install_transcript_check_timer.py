"""Tests for scripts/legibility/install-transcript-check-timer.sh — the
install script for the legibility-transcript-check systemd user timer
(task 2901, follow-up to task 2893's lost-transcript detector).

Mirrors scripts/tests/test_install_trickle_timer.py: drives the script via
subprocess with `scripts/tests/conftest.py::fake_timer_systemctl` first on PATH,
plus a real invocation of `nightly.py resolve-config` (the REAL Python resolver — no
mock — with INSTALL_TRANSCRIPT_CHECK_TIMER_PYTHON=sys.executable substituting
for the default `uv run` prefix, and LEGIBILITY_SEARCH_ROOTS pointed at a tmp
fixture project). Real systemd is never touched.

Divergences from the mirrored trickle install test:
  * SCRIPT / unit basenames re-pointed at legibility-transcript-check@.
  * The installed @.service content assertion additionally checks it
    references check_transcript_persistence.py (NOT nightly.py) and does
    NOT pass --check-preventer (DD2: the 2893 preventer-guard regex
    false-negatives against the real spawn-claude.sh, so passing it would
    make the timer exit 1 on every run — a perpetual false alarm).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPT = Path(__file__).parent.parent / "legibility" / "install-transcript-check-timer.sh"
TEMPLATES_DIR = Path(__file__).parent.parent  # scripts/tests/../ = scripts/


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

def _run_script(fake_timer_systemctl, project_id, *, env=None):
    """Run install-transcript-check-timer.sh <project_id> via subprocess.

    Puts *fake_timer_systemctl* first on PATH, reset on every call, and
    defaults INSTALL_TRANSCRIPT_CHECK_TIMER_PYTHON=sys.executable so the
    script's `nightly.py resolve-config` delegation runs the REAL resolver
    directly -- no `uv run`/`--frozen` needed in-test. Callers supply
    XDG_CONFIG_HOME and LEGIBILITY_SEARCH_ROOTS via *env*.
    """
    fake_timer_systemctl.reset()

    full_env = dict(os.environ)
    full_env["PATH"] = f"{fake_timer_systemctl.bin_dir}{os.pathsep}{full_env['PATH']}"
    full_env["INSTALL_TRANSCRIPT_CHECK_TIMER_PYTHON"] = sys.executable
    full_env.pop("LEGIBILITY_SEARCH_ROOTS", None)
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), project_id],
        env=full_env, capture_output=True, text=True, timeout=30,
    )


# ---------------------------------------------------------------------------
# step-1: RED -- install-transcript-check-timer.sh
# ---------------------------------------------------------------------------

def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), (
        f"Expected {SCRIPT} to be executable (os.X_OK); it is not. "
        f"Run: chmod +x {SCRIPT}"
    )


def test_install_copies_templates_and_enables_timer(fake_timer_systemctl, tmp_path):
    search_root = _write_project_config(tmp_path, project_id="proj_a")
    xdg_config = tmp_path / "xdg-config"

    result = _run_script(
        fake_timer_systemctl, "proj_a",
        env={"XDG_CONFIG_HOME": str(xdg_config), "LEGIBILITY_SEARCH_ROOTS": str(search_root)},
    )

    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    unit_dir = xdg_config / "systemd" / "user"
    service_path = unit_dir / "legibility-transcript-check@.service"
    timer_path = unit_dir / "legibility-transcript-check@.timer"
    assert service_path.is_file(), f"Expected {service_path} to exist after install"
    assert timer_path.is_file(), f"Expected {timer_path} to exist after install"
    # Byte-equality with the committed templates above already proves the
    # install copied them verbatim; the @.service ExecStart *shape* invariant
    # (references the detector, passes %i, and — DD2 — never --check-preventer)
    # is asserted directly against the committed template in
    # test_service_template_execstart_shape, independent of install behavior.

    calls = fake_timer_systemctl.calls()
    assert ["daemon-reload"] in calls, f"calls={calls!r}"
    assert ["enable", "--now", "legibility-transcript-check@proj_a.timer"] in calls, f"calls={calls!r}"
    assert any(c[0] == "list-timers" for c in calls), f"calls={calls!r}"


def test_service_template_execstart_shape():
    """Template-content invariant (NOT install behavior): the committed
    @.service's ExecStart must target the DETECTOR
    (check_transcript_persistence.py) with the %i instance placeholder
    (systemd expands %i at instantiation — the template is copied verbatim,
    never per-instance edited) and must NOT pass --check-preventer.

    DD2: the 2893 preventer-guard regex false-negatives against the real
    spawn-claude.sh, so passing --check-preventer would make the timer exit 1
    on EVERY run (a perpetual false alarm). This pins the ExecStart shape at
    its source, so a future template edit that reintroduced --check-preventer
    fails here regardless of the install path.
    """
    service_text = (TEMPLATES_DIR / "legibility-transcript-check@.service").read_text()
    assert "check_transcript_persistence.py" in service_text
    assert "--project-id %i" in service_text
    assert "--check-preventer" not in service_text, (
        "The transcript-check service template must NOT pass --check-preventer "
        "(DD2): the 2893 preventer-guard regex false-negatives against the real "
        "spawn-claude.sh, which would make the timer exit 1 on every run."
    )


def test_install_is_idempotent(fake_timer_systemctl, tmp_path):
    search_root = _write_project_config(tmp_path, project_id="proj_a")
    xdg_config = tmp_path / "xdg-config"
    env = {"XDG_CONFIG_HOME": str(xdg_config), "LEGIBILITY_SEARCH_ROOTS": str(search_root)}

    result_1 = _run_script(fake_timer_systemctl, "proj_a", env=env)
    assert result_1.returncode == 0, (
        f"stdout={result_1.stdout!r} stderr={result_1.stderr!r}"
    )

    result_2 = _run_script(fake_timer_systemctl, "proj_a", env=env)
    assert result_2.returncode == 0, (
        f"Expected re-running the install to still exit 0; "
        f"stdout={result_2.stdout!r} stderr={result_2.stderr!r}"
    )


def test_install_fails_when_self_verify_omits_timer(fake_timer_systemctl, tmp_path):
    search_root = _write_project_config(tmp_path, project_id="proj_a")
    xdg_config = tmp_path / "xdg-config"

    result = _run_script(
        fake_timer_systemctl, "proj_a",
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


def test_install_fails_when_project_config_unresolvable(fake_timer_systemctl, tmp_path):
    xdg_config = tmp_path / "xdg-config"
    empty_search_root = tmp_path / "empty-search-root"
    empty_search_root.mkdir()

    result = _run_script(
        fake_timer_systemctl, "no_such_project",
        env={"XDG_CONFIG_HOME": str(xdg_config), "LEGIBILITY_SEARCH_ROOTS": str(empty_search_root)},
    )

    assert result.returncode != 0, (
        f"Expected a non-zero exit for an unresolvable project_id; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert fake_timer_systemctl.calls() == [], (
        f"Expected NO systemctl invocation when config resolution fails first; "
        f"calls={fake_timer_systemctl.calls()!r}"
    )
    assert not (xdg_config / "systemd" / "user" / "legibility-transcript-check@.service").exists(), (
        "Expected no unit file to be installed when config resolution fails"
    )
