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
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[2]
TEMPLATE = REPO_ROOT / "scripts" / "dashboard.service.template"
HARDCODED = REPO_ROOT / "dashboard" / "dark-factory-dashboard.service"

EXPECTED_ENV_LINE = (
    "Environment=DASHBOARD_KNOWN_PROJECT_ROOTS="
    "/home/leo/src/dark-factory,"
    "/home/leo/src/reify,"
    "/home/leo/src/autopilot-video"
)


def _assert_known_project_roots_comma_separated(path: pathlib.Path) -> None:
    """Assert that DASHBOARD_KNOWN_PROJECT_ROOTS in *path* uses commas, not colons.

    Parses the Environment= line with a regex so the check is position-independent:
    a colon anywhere in the value fails the assertion regardless of which root it
    follows.
    """
    content = path.read_text(encoding="utf-8")
    match = re.search(
        r"^Environment=DASHBOARD_KNOWN_PROJECT_ROOTS=(.*)$",
        content,
        re.MULTILINE,
    )
    assert match is not None, (
        f"Environment=DASHBOARD_KNOWN_PROJECT_ROOTS= line not found in {path}"
    )
    value = match.group(1)
    assert ":" not in value, (
        f"Colon-separated DASHBOARD_KNOWN_PROJECT_ROOTS found in {path}. "
        "Use commas — the parser at "
        "dashboard/src/dashboard/config.py \u2014 "
        "DashboardConfig.from_env handling of DASHBOARD_KNOWN_PROJECT_ROOTS "
        "calls roots.split(',')."
    )


def test_template_sets_known_project_roots() -> None:
    """scripts/dashboard.service.template must declare DASHBOARD_KNOWN_PROJECT_ROOTS."""
    content = TEMPLATE.read_text(encoding="utf-8")
    assert EXPECTED_ENV_LINE in content, (
        f"Expected line not found in {TEMPLATE}:\n  {EXPECTED_ENV_LINE!r}\n"
        "Add it to the [Service] section after the ExecStart block."
    )


def test_hardcoded_service_file_sets_known_project_roots() -> None:
    """dashboard/dark-factory-dashboard.service must declare DASHBOARD_KNOWN_PROJECT_ROOTS."""
    content = HARDCODED.read_text(encoding="utf-8")
    assert EXPECTED_ENV_LINE in content, (
        f"Expected line not found in {HARDCODED}:\n  {EXPECTED_ENV_LINE!r}\n"
        "Add it to the [Service] section after the ExecStart block."
    )


def test_comma_separator_helper_detects_colon_in_any_position(
    tmp_path: pathlib.Path,
) -> None:
    """_assert_known_project_roots_comma_separated must catch colons in any position.

    The narrow old guard (looking for '/home/leo/src/dark-factory:') fails when the
    first root is not dark-factory or the colon appears between the second and third
    roots.  This test exercises the case that the old guard cannot see.
    """
    # Bad: colon between second and third roots (old guard misses this)
    bad_file = tmp_path / "bad.service"
    bad_file.write_text(
        "[Service]\nEnvironment=DASHBOARD_KNOWN_PROJECT_ROOTS=/a,/b:/c\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        _assert_known_project_roots_comma_separated(bad_file)

    # Good: all commas, helper must not raise
    good_file = tmp_path / "good.service"
    good_file.write_text(
        "[Service]\nEnvironment=DASHBOARD_KNOWN_PROJECT_ROOTS=/a,/b,/c\n",
        encoding="utf-8",
    )
    _assert_known_project_roots_comma_separated(good_file)


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
