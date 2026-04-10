"""File-content tests for the dark-factory-dashboard systemd service files.

These tests read the source-controlled service definition files directly —
no systemd runtime is required.  They guard against drift between the
template (scripts/dashboard.service.template, the true source of truth used
by setup-host.sh) and the checked-in hardcoded copy
(dashboard/dark-factory-dashboard.service).

See also:
  - tests/scripts/test_run_vllm_eval_lint.py  — pattern reference
  - dashboard/src/dashboard/config.py line 84  — COMMA-separated split
"""
import pathlib

REPO_ROOT = pathlib.Path(__file__).parents[2]
TEMPLATE = REPO_ROOT / "scripts" / "dashboard.service.template"
HARDCODED = REPO_ROOT / "dashboard" / "dark-factory-dashboard.service"

TEMPLATE_EXPECTED_ENV_LINE = (
    "Environment=DASHBOARD_KNOWN_PROJECT_ROOTS="
    "__REPO_ROOT__,"
    "/home/leo/src/reify,"
    "/home/leo/src/autopilot-video"
)

HARDCODED_EXPECTED_ENV_LINE = (
    "Environment=DASHBOARD_KNOWN_PROJECT_ROOTS="
    "/home/leo/src/dark-factory,"
    "/home/leo/src/reify,"
    "/home/leo/src/autopilot-video"
)


def test_template_sets_known_project_roots() -> None:
    """scripts/dashboard.service.template must declare DASHBOARD_KNOWN_PROJECT_ROOTS with __REPO_ROOT__ sentinel."""
    content = TEMPLATE.read_text(encoding="utf-8")
    assert TEMPLATE_EXPECTED_ENV_LINE in content, (
        f"Expected line not found in {TEMPLATE}:\n  {TEMPLATE_EXPECTED_ENV_LINE!r}\n"
        "The template must use __REPO_ROOT__ as the self entry, not a hardcoded path. "
        "Add it to the [Service] section after the ExecStart block."
    )


def test_hardcoded_service_file_sets_known_project_roots() -> None:
    """dashboard/dark-factory-dashboard.service must declare DASHBOARD_KNOWN_PROJECT_ROOTS with literal path."""
    content = HARDCODED.read_text(encoding="utf-8")
    assert HARDCODED_EXPECTED_ENV_LINE in content, (
        f"Expected line not found in {HARDCODED}:\n  {HARDCODED_EXPECTED_ENV_LINE!r}\n"
        "Add it to the [Service] section after the ExecStart block."
    )


def test_known_project_roots_uses_comma_separator_not_colon() -> None:
    """Both service files must use commas (not colons) to separate project roots.

    The consumer code is ``roots.split(',')`` — a colon-separated value would
    be parsed as a single path literal and silently aggregate nothing.
    """
    colon_pattern = (
        "DASHBOARD_KNOWN_PROJECT_ROOTS="
        "/home/leo/src/dark-factory:"
    )
    for path in (TEMPLATE, HARDCODED):
        content = path.read_text(encoding="utf-8")
        assert colon_pattern not in content, (
            f"Colon-separated DASHBOARD_KNOWN_PROJECT_ROOTS found in {path}. "
            "Use commas — the parser at dashboard/src/dashboard/config.py:84 "
            "calls roots.split(',')."
        )
