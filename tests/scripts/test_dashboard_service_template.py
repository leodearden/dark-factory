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

# Canonical substitution values for rendering the template.
# Source of truth: setup-host.sh lines 325-329 which run:
#   sed 's|__REPO_ROOT__|$REPO_ROOT|g'   (global, unanchored, literal substitution)
#   sed 's|__UV_PATH__|$UV_PATH|g'       (global, unanchored, literal substitution)
# These must match the values in the committed dashboard/dark-factory-dashboard.service.
EXPECTED_REPO_ROOT = "/home/leo/src/dark-factory"
EXPECTED_UV_PATH = "/home/leo/.local/bin/uv"


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
    assert value.strip() != "", (
        f"DASHBOARD_KNOWN_PROJECT_ROOTS is empty or whitespace-only in {path}. "
        "An empty value would silently produce a single empty-string root after "
        "split(','), which is a misconfiguration."
    )
    assert ":" not in value, (
        f"Colon-separated DASHBOARD_KNOWN_PROJECT_ROOTS found in {path}. "
        "Use commas — the parser at "
        "dashboard/src/dashboard/config.py — "
        "DashboardConfig.from_env handling of DASHBOARD_KNOWN_PROJECT_ROOTS "
        "calls roots.split(',')."
    )


def test_template_sets_known_project_roots() -> None:
    """scripts/dashboard.service.template must declare DASHBOARD_KNOWN_PROJECT_ROOTS with __REPO_ROOT__ sentinel.

    Kept for targeted diagnostics — this property is subsumed by
    test_template_renders_to_hardcoded_file, but this test pinpoints which
    specific invariant broke if the render test fails.
    """
    content = TEMPLATE.read_text(encoding="utf-8")
    assert TEMPLATE_EXPECTED_ENV_LINE in content, (
        f"Expected line not found in {TEMPLATE}:\n  {TEMPLATE_EXPECTED_ENV_LINE!r}\n"
        "The template must use __REPO_ROOT__ as the self entry, not a hardcoded path. "
        "Add it to the [Service] section after the ExecStart block."
    )


def test_hardcoded_service_file_sets_known_project_roots() -> None:
    """dashboard/dark-factory-dashboard.service must declare DASHBOARD_KNOWN_PROJECT_ROOTS with literal path.

    Kept for targeted diagnostics — this property is subsumed by
    test_template_renders_to_hardcoded_file, but this test pinpoints which
    specific invariant broke if the render test fails.
    """
    content = HARDCODED.read_text(encoding="utf-8")
    assert HARDCODED_EXPECTED_ENV_LINE in content, (
        f"Expected line not found in {HARDCODED}:\n  {HARDCODED_EXPECTED_ENV_LINE!r}\n"
        "Add it to the [Service] section after the ExecStart block."
    )


def test_comma_separator_helper_rejects_empty_value(
    tmp_path: pathlib.Path,
) -> None:
    """_assert_known_project_roots_comma_separated must reject an empty or whitespace-only value.

    An empty DASHBOARD_KNOWN_PROJECT_ROOTS would silently produce a single empty-string
    root after split(','), which is a misconfiguration.  A whitespace-only value is
    equally broken (systemd treats spaces inside an Environment= value as separators
    between variable assignments, so the entire value would be discarded).
    """
    # Bad: empty value — regex matches, group(1) is '', helper should raise
    empty_file = tmp_path / "empty.service"
    empty_file.write_text(
        "[Service]\nEnvironment=DASHBOARD_KNOWN_PROJECT_ROOTS=\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        _assert_known_project_roots_comma_separated(empty_file)

    # Bad: whitespace-only value — group(1) is '   ', strip() is '', helper should raise
    whitespace_file = tmp_path / "whitespace.service"
    whitespace_file.write_text(
        "[Service]\nEnvironment=DASHBOARD_KNOWN_PROJECT_ROOTS=   \n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        _assert_known_project_roots_comma_separated(whitespace_file)


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

    Kept for targeted diagnostics — this property is subsumed by
    test_template_renders_to_hardcoded_file, but this test pinpoints which
    specific invariant broke if the render test fails.
    """
    for path in (TEMPLATE, HARDCODED):
        _assert_known_project_roots_comma_separated(path)


def test_comment_warns_about_systemd_space_handling() -> None:
    """Both service files must carry an intent-based comment for DASHBOARD_KNOWN_PROJECT_ROOTS.

    The check is intent-based, not prose-pinning:
    - The line immediately above Environment=DASHBOARD_KNOWN_PROJECT_ROOTS= (skipping blanks)
      must be a '#' comment that mentions both 'systemd' and 'space' (case-insensitive).
    - The old misleading phrase 'no spaces' must not appear anywhere in either file.

    This is stronger than an exact-string match: any future copy-edit that preserves the
    warning intent (systemd treats spaces as separators) will pass, while edits that remove
    or contradict the intent will fail.
    """
    for path in (TEMPLATE, HARDCODED):
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()

        # Find the Environment=DASHBOARD_KNOWN_PROJECT_ROOTS= line
        env_idx = next(
            (i for i, ln in enumerate(lines) if "Environment=DASHBOARD_KNOWN_PROJECT_ROOTS=" in ln),
            None,
        )
        assert env_idx is not None, (
            f"Environment=DASHBOARD_KNOWN_PROJECT_ROOTS= line not found in {path}"
        )

        # Walk backward to the nearest non-blank line
        comment_idx = env_idx - 1
        while comment_idx >= 0 and lines[comment_idx].strip() == "":
            comment_idx -= 1

        assert comment_idx >= 0, (
            f"No non-blank line found above Environment=DASHBOARD_KNOWN_PROJECT_ROOTS= in {path}"
        )
        comment_line = lines[comment_idx]

        assert comment_line.startswith("#"), (
            f"Line above Environment=DASHBOARD_KNOWN_PROJECT_ROOTS= in {path} "
            f"is not a '#' comment:\n  {comment_line!r}"
        )
        comment_lower = comment_line.lower()
        assert "systemd" in comment_lower, (
            f"Comment above Environment=DASHBOARD_KNOWN_PROJECT_ROOTS= in {path} "
            f"does not mention 'systemd':\n  {comment_line!r}\n"
            "Update the comment to explain the systemd space-separator hazard."
        )
        assert "space" in comment_lower, (
            f"Comment above Environment=DASHBOARD_KNOWN_PROJECT_ROOTS= in {path} "
            f"does not mention 'space':\n  {comment_line!r}\n"
            "Update the comment to warn about spaces inside the Environment= value."
        )

        # The old misleading phrase 'no spaces' must not appear anywhere in this file
        assert "no spaces" not in content.lower(), (
            f"Misleading phrase 'no spaces' found in {path}. "
            "Remove it — the real hazard is systemd's space-as-separator behavior, "
            "not the Python parser's whitespace tolerance."
        )


def test_template_renders_to_hardcoded_file() -> None:
    """Rendered template must match the committed hardcoded service file verbatim.

    This is the canonical drift-prevention invariant: applying the same substitutions
    as setup-host.sh (lines 325-329) to the template must yield the hardcoded file
    byte-for-byte.

    Substitution semantics (mirroring setup-host.sh):
        sed 's|__REPO_ROOT__|$REPO_ROOT|g'  →  str.replace('__REPO_ROOT__', EXPECTED_REPO_ROOT)
        sed 's|__UV_PATH__|$UV_PATH|g'      →  str.replace('__UV_PATH__', EXPECTED_UV_PATH)

    Both sentinels contain no regex metacharacters and no '|', so str.replace is
    semantically identical to the sed command (global, unanchored, literal substitution).

    If this test fails, the template and hardcoded file have drifted.  Re-render by
    running the sed substitutions in setup-host.sh lines 325-329 and updating
    dashboard/dark-factory-dashboard.service.
    """
    rendered = (
        TEMPLATE.read_text(encoding="utf-8")
        .replace("__REPO_ROOT__", EXPECTED_REPO_ROOT)
        .replace("__UV_PATH__", EXPECTED_UV_PATH)
    )
    hardcoded = HARDCODED.read_text(encoding="utf-8")
    assert rendered == hardcoded, (
        f"Rendered template does not match {HARDCODED}.\n"
        f"Template path: {TEMPLATE}\n"
        "The files have drifted.  Re-render by running the sed substitutions "
        "in setup-host.sh lines 325-329 and updating "
        "dashboard/dark-factory-dashboard.service."
    )
