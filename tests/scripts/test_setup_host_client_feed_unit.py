"""setup-host.sh section 7a: render client-feed@.service, enable nothing.

The slice runs the real renderer against a tmp unit dir with a recording
systemctl stub (tests/scripts/setup_host_sections.py). Instances are enabled by
hand, one per client project, so the section must never enable or start one.
"""

import pathlib

import render_systemd_unit
from setup_host_sections import (
    REPO_ROOT,
    enabled_units,
    run_section,
    slice_section,
    systemctl_calls,
)

TEMPLATE = REPO_ROOT / "scripts" / "client-feed.service.template"
INSTALLED_NAME = "client-feed@.service"
UV_PATH = "/opt/uv/bin/uv"

_START = "_cf_render_script="
_END = "\nfi\n"


def _run(tmp_path: pathlib.Path, repo_root: pathlib.Path, *, trailer: str = ""):
    unit_dir = tmp_path / "units"
    result = run_section(
        tmp_path,
        slice_section(_START, _END) + trailer,
        repo_root=repo_root,
        unit_dir=unit_dir,
        env_extra={"UV_PATH": UV_PATH},
    )
    return result, unit_dir


def test_setup_host_renders_the_client_feed_template_and_enables_no_instance(
    tmp_path: pathlib.Path,
) -> None:
    result, unit_dir = _run(tmp_path, REPO_ROOT)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (unit_dir / INSTALLED_NAME).read_text(encoding="utf-8") == (
        render_systemd_unit.render_template(
            TEMPLATE.read_text(encoding="utf-8"),
            repo_root=str(REPO_ROOT),
            uv_path=UV_PATH,
        )
    )
    assert systemctl_calls(tmp_path) == [["--user", "daemon-reload"]]
    assert enabled_units(tmp_path) == []


def test_a_refused_render_leaves_the_installed_unit_alone_and_the_bootstrap_running(
    tmp_path: pathlib.Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("render_systemd_unit.py", "systemd_unit_parity.py"):
        (repo / "scripts" / name).write_text(
            (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    unit_dir = tmp_path / "units"
    unit_dir.mkdir()
    (unit_dir / INSTALLED_NAME).write_text("OLD\n", encoding="utf-8")

    result, _ = _run(tmp_path, repo, trailer="printf 'AFTER\\n'\n")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "AFTER" in result.stdout, "a refused render aborted the installer"
    fail_lines = [ln for ln in result.stdout.splitlines() if ln.startswith("FAIL ")]
    assert any(INSTALLED_NAME in ln for ln in fail_lines), result.stdout
    assert (unit_dir / INSTALLED_NAME).read_text(encoding="utf-8") == "OLD\n"
    assert systemctl_calls(tmp_path) == []
    assert enabled_units(tmp_path) == []
