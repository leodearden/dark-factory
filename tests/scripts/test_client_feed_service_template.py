"""File-content tests for scripts/client-feed.service.template.

The template is rendered by scripts/setup-host.sh into client-feed@.service,
one instance per client project. Its restart backoff and its `uv run --no-sync`
are owned by the fleet sweeps tests/scripts/test_systemd_restart_backoff.py and
tests/scripts/test_uv_run_venv_isolation.py; this module pins the rest.
"""

import pathlib

import render_systemd_unit
from systemd_unit_invariants import logical_exec_start
from systemd_unit_parity import parse_unit_directives

REPO_ROOT = pathlib.Path(__file__).parents[2]
TEMPLATE = REPO_ROOT / "scripts" / "client-feed.service.template"

_RENDER_ROOT = "/srv/dark-factory"
_RENDER_UV = "/opt/uv/bin/uv"


def _rendered() -> str:
    return render_systemd_unit.render_template(
        TEMPLATE.read_text(encoding="utf-8"),
        repo_root=_RENDER_ROOT,
        uv_path=_RENDER_UV,
    )


def test_rendered_unit_publishes_the_instances_config_from_the_repo_root() -> None:
    rendered = _rendered()
    directives = parse_unit_directives(rendered)

    assert "__REPO_ROOT__" not in rendered
    assert "__UV_PATH__" not in rendered
    assert directives["Service"]["WorkingDirectory"] == [_RENDER_ROOT]
    assert logical_exec_start(rendered, "client-feed@.service") == (
        f"{_RENDER_UV} run --no-sync --project client-feed client-feed publish"
        " --config %h/.config/client-feed/%i.yaml"
    )


def test_unit_restarts_on_failure_logs_to_the_journal_and_can_be_enabled_per_instance() -> None:
    directives = parse_unit_directives(_rendered())
    service = directives["Service"]

    assert service["Restart"] == ["on-failure"]
    assert service["StandardOutput"] == ["journal"]
    assert service["StandardError"] == ["journal"]
    assert directives["Install"]["WantedBy"] == ["default.target"], (
        "without [Install] WantedBy=, `systemctl --user enable "
        "client-feed@<name>` has nothing to enable"
    )
