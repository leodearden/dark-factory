"""setup-host.sh section 7a: render client-feed@.service and client-feed-filer@.service, enable nothing.

The slice runs the real renderer against a tmp unit dir with a recording
systemctl stub (tests/scripts/setup_host_sections.py). Instances of either unit
are enabled by hand, one per client project, so the section must never enable or
start one.
"""

import pathlib

import pytest
import render_systemd_unit
from setup_host_sections import (
    REPO_ROOT,
    enabled_units,
    run_section,
    slice_section,
    systemctl_calls,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

PUBLISHER_TEMPLATE = REPO_ROOT / "scripts" / "client-feed.service.template"
FILER_TEMPLATE = REPO_ROOT / "scripts" / "client-feed-filer.service.template"
PUBLISHER_NAME = "client-feed@.service"
FILER_NAME = "client-feed-filer@.service"
UV_PATH = "/opt/uv/bin/uv"
DAEMON_RELOAD = ["--user", "daemon-reload"]

_START = "_cf_render_script="
_END = "\nfi\n"
_END_AFTER = "client-feed-filer.service.template"
_RENDERER_FILES = ("render_systemd_unit.py", "systemd_unit_parity.py")


def _run(tmp_path: pathlib.Path, repo_root: pathlib.Path, *, trailer: str = ""):
    unit_dir = tmp_path / "units"
    result = run_section(
        tmp_path,
        slice_section(_START, _END, end_after=_END_AFTER) + trailer,
        repo_root=repo_root,
        unit_dir=unit_dir,
        env_extra={"UV_PATH": UV_PATH},
    )
    return result, unit_dir


def _render(template: pathlib.Path, repo_root: pathlib.Path) -> str:
    return render_systemd_unit.render_template(
        template.read_text(encoding="utf-8"),
        repo_root=str(repo_root),
        uv_path=UV_PATH,
    )


def _repo_copy(tmp_path: pathlib.Path, *templates: pathlib.Path) -> pathlib.Path:
    """A repo holding the renderer and only `templates`: a render of any other template is refused."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for source in (*(REPO_ROOT / "scripts" / name for name in _RENDERER_FILES), *templates):
        (repo / "scripts" / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return repo


def _installed(tmp_path: pathlib.Path, *names: str) -> pathlib.Path:
    unit_dir = tmp_path / "units"
    unit_dir.mkdir()
    for name in names:
        (unit_dir / name).write_text("OLD\n", encoding="utf-8")
    return unit_dir


def _fail_lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if line.startswith("FAIL ")]


def test_setup_host_renders_both_client_feed_templates_and_enables_no_instance(
    tmp_path: pathlib.Path,
) -> None:
    result, unit_dir = _run(tmp_path, REPO_ROOT)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (unit_dir / PUBLISHER_NAME).read_text(encoding="utf-8") == _render(PUBLISHER_TEMPLATE, REPO_ROOT)
    assert (unit_dir / FILER_NAME).read_text(encoding="utf-8") == _render(FILER_TEMPLATE, REPO_ROOT)
    calls = systemctl_calls(tmp_path)
    assert calls, "no daemon-reload after installing the units"
    assert all(call == DAEMON_RELOAD for call in calls), calls
    assert enabled_units(tmp_path) == []


def test_a_refused_render_leaves_the_installed_units_alone_and_the_bootstrap_running(
    tmp_path: pathlib.Path,
) -> None:
    repo = _repo_copy(tmp_path)
    unit_dir = _installed(tmp_path, PUBLISHER_NAME, FILER_NAME)

    result, _ = _run(tmp_path, repo, trailer="printf 'AFTER\\n'\n")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "AFTER" in result.stdout, "a refused render aborted the installer"
    fail_lines = _fail_lines(result.stdout)
    for name in (PUBLISHER_NAME, FILER_NAME):
        assert any(name in line for line in fail_lines), result.stdout
        assert (unit_dir / name).read_text(encoding="utf-8") == "OLD\n"
    assert systemctl_calls(tmp_path) == []
    assert enabled_units(tmp_path) == []


def test_a_refused_filer_render_leaves_its_installed_unit_alone_and_the_publisher_rendered(
    tmp_path: pathlib.Path,
) -> None:
    repo = _repo_copy(tmp_path, PUBLISHER_TEMPLATE)
    unit_dir = _installed(tmp_path, FILER_NAME)

    result, _ = _run(tmp_path, repo, trailer="printf 'AFTER\\n'\n")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "AFTER" in result.stdout, "a refused filer render aborted the installer"
    fail_lines = _fail_lines(result.stdout)
    assert fail_lines and all(FILER_NAME in line for line in fail_lines), result.stdout
    assert any("[client_feed_filer_unit_render]" in line for line in fail_lines), result.stdout
    assert (unit_dir / FILER_NAME).read_text(encoding="utf-8") == "OLD\n"
    assert (unit_dir / PUBLISHER_NAME).read_text(encoding="utf-8") == _render(PUBLISHER_TEMPLATE, repo)
    assert systemctl_calls(tmp_path) == [DAEMON_RELOAD]
    assert enabled_units(tmp_path) == []
