"""File-content tests for the dark-factory-dashboard systemd service files.

These tests read the source-controlled service definition files directly —
no systemd runtime is required.  They guard against drift between the
template (scripts/dashboard.service.template, the true source of truth used
by setup-host.sh) and the checked-in hardcoded copy
(dashboard/dark-factory-dashboard.service).

See also:
  - tests/scripts/test_run_vllm_eval_lint.py  — pattern reference
  - dashboard/src/dashboard/config.py — DashboardConfig.from_env handling of DASHBOARD_KNOWN_PROJECT_ROOTS (COMMA-separated split)
"""

import pathlib
import re

import pytest

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

    The helper parses the Environment= value and checks for any colon, so it
    guards both the literal-path form (``/home/leo/src/dark-factory:``) and the
    template's ``__REPO_ROOT__:`` sentinel form in a single, position-independent
    pass.
    """
    for path in (TEMPLATE, HARDCODED):
        _assert_known_project_roots_comma_separated(path)


def test_comment_warns_about_systemd_space_handling() -> None:
    """Both service files must carry the systemd-aware comment for DASHBOARD_KNOWN_PROJECT_ROOTS.

    The old wording 'comma-separated, no spaces' was misleading — the Python parser
    tolerates whitespace around commas.  The real hazard is systemd: spaces inside
    an Environment= value are treated as separators between variable assignments.
    The updated comment makes this explicit so future editors understand why spaces
    are forbidden.
    """
    expected_comment = (
        "# Multi-project cost aggregation "
        "(comma-separated; avoid spaces inside the value \u2014 "
        "systemd would treat them as assignment separators)"
    )
    for path in (TEMPLATE, HARDCODED):
        content = path.read_text(encoding="utf-8")
        assert expected_comment in content, (
            f"Systemd-aware comment not found in {path}:\n  {expected_comment!r}\n"
            "Update the comment above the Environment=DASHBOARD_KNOWN_PROJECT_ROOTS line."
        )
