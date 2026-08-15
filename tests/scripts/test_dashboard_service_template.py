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

import functools
import pathlib
import re
import shutil
import subprocess
import typing

import pytest
from systemd_unit_invariants import (
    assert_restart_backoff_effective as _assert_restart_backoff_effective,
)
from systemd_unit_invariants import (
    restart_directive as _restart_directive,
)

REPO_ROOT = pathlib.Path(__file__).parents[2]
TEMPLATE = REPO_ROOT / "scripts" / "dashboard.service.template"
HARDCODED = REPO_ROOT / "dashboard" / "dark-factory-dashboard.service"

TEMPLATE_EXPECTED_ENV_LINE = (
    "Environment=DASHBOARD_KNOWN_PROJECT_ROOTS="
    "__REPO_ROOT__"
)

HARDCODED_EXPECTED_ENV_LINE = (
    "Environment=DASHBOARD_KNOWN_PROJECT_ROOTS="
    "/home/leo/src/dark-factory"
)

# WorkingDirectory= is load-bearing on its own terms: ExecStart runs
# `uv run --project dashboard python -m uvicorn dashboard.app:app`, and
# `--project dashboard` is a RELATIVE path resolved against the unit's
# process cwd. systemd sets that cwd only from WorkingDirectory= (it does not
# inherit an interactive shell's), so dropping or repointing this directive
# makes uv resolve the wrong project directory, or fail outright. Both files
# also pin an ABSOLUTE path, which systemd requires: an unsubstituted
# __REPO_ROOT__ sentinel trips a fatal "WorkingDirectory= path is not
# absolute" error, corroborated in this file's
# test_systemd_analyze_verify_reports_no_ignored_directives.
# check_dashboard_unit_parity.py treats WorkingDirectory as presence-only by
# design (it compares committed-vs-installed, where the directive's value can
# legitimately diverge across hosts), so it does not guard the
# template/hardcoded pair asserted below — that guard is
# _assert_exact_unit_line, an anchored check (not a substring check),
# so a repointed or commented-out directive still fails.
TEMPLATE_EXPECTED_WORKING_DIRECTORY_LINE = "WorkingDirectory=__REPO_ROOT__"

HARDCODED_EXPECTED_WORKING_DIRECTORY_LINE = (
    "WorkingDirectory=/home/leo/src/dark-factory"
)

# DASHBOARD_PROJECT_ROOT is declared EXPLICITLY (task 3572) because
# dashboard/src/dashboard/config.py's `project_root` falls back to Path.cwd()
# when it is unset — so without the line asserted below the dashboard's entire
# data root is an undeclared SIDE EFFECT of WorkingDirectory=.  The units carry
# that rationale on the line itself; the full version, including why the
# installed copy cannot be value-compared, is on
# scripts/check_dashboard_unit_parity.py's UnitSpec.env_matches_directive.
#
# What is pinned HERE is the two-part shape: each file declares the line
# verbatim (these constants), AND the value equals WorkingDirectory= — a
# separate assertion, since both lines can be individually well-formed while
# one has been repointed.  See
# test_project_root_env_matches_working_directory_in_both_unit_files below.
TEMPLATE_EXPECTED_PROJECT_ROOT_ENV_LINE = (
    "Environment=DASHBOARD_PROJECT_ROOT=__REPO_ROOT__"
)

HARDCODED_EXPECTED_PROJECT_ROOT_ENV_LINE = (
    "Environment=DASHBOARD_PROJECT_ROOT=/home/leo/src/dark-factory"
)

# These are the literal paths baked into the committed hardcoded service file;
# the render test verifies the template expands to exactly those values.
# setup-host.sh computes REPO_ROOT and UV_PATH at runtime (from $(dirname $0)/..
# and $(command -v uv) respectively), so what it installs on a worktree or
# alternate machine may legitimately differ — the test is not asserting anything
# about the runtime install environment.
#
# Substitution semantics (setup-host.sh lines 325-329):
#   sed 's|__REPO_ROOT__|$REPO_ROOT|g'   (global, unanchored, literal substitution)
#   sed 's|__UV_PATH__|$UV_PATH|g'       (global, unanchored, literal substitution)
# Both sentinels contain no regex metacharacters and no '|', so str.replace is
# semantically identical to the sed command.
HARDCODED_SERVICE_REPO_ROOT = "/home/leo/src/dark-factory"
HARDCODED_SERVICE_UV_PATH = "/home/leo/.local/bin/uv"


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


def _assert_exact_unit_line(path: pathlib.Path, expected_line: str) -> None:
    """Assert *expected_line* appears in *path* as a whole, anchored line.

    Generic by construction — it asserts nothing about WHICH directive it is
    given, so it serves every unit line whose exact value is load-bearing.  It
    is used here for both ``WorkingDirectory=`` and
    ``Environment=DASHBOARD_PROJECT_ROOT=``; see the module comments above
    TEMPLATE_EXPECTED_WORKING_DIRECTORY_LINE and
    TEMPLATE_EXPECTED_PROJECT_ROOT_ENV_LINE for why each of those is
    load-bearing.

    A plain substring check (``expected_line in content``) is satisfied by a
    repointed value that merely starts with the expected text — e.g.
    ``WorkingDirectory=__REPO_ROOT__/dashboard`` still contains
    ``WorkingDirectory=__REPO_ROOT__`` — and by the same text sitting inside a
    ``#`` comment, which systemd never parses as a directive.  Both leave the
    unit's real cwd (or data root) wrong or unset while a substring check stays
    green.  Anchoring the full line with ``^...$`` under ``re.MULTILINE``
    rejects both: a suffix breaks the ``$`` boundary, and a leading ``#``
    breaks the ``^`` boundary.
    """
    content = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^{re.escape(expected_line)}\s*$", re.MULTILINE)
    assert pattern.search(content) is not None, (
        f"No line matching {expected_line!r} found in {path}. See the "
        "rationale comments above TEMPLATE_EXPECTED_WORKING_DIRECTORY_LINE and "
        "TEMPLATE_EXPECTED_PROJECT_ROOT_ENV_LINE for why these directives are "
        "load-bearing."
    )


def _assert_project_root_env_matches_working_directory(path: pathlib.Path) -> None:
    """Assert DASHBOARD_PROJECT_ROOT equals WorkingDirectory= WITHIN *path*.

    This is the relation the two directives must satisfy, checked INSIDE a
    single file rather than across the template/hardcoded pair.  The exact-line
    assertions above already pin each file's literal value; what they cannot
    catch is one of the two being repointed while the other stays put, since
    each remains individually well-formed.

    Checking the relation intra-file (rather than comparing one file's value to
    the other's) is also what makes it host-invariant: the committed unit
    hardcodes /home/leo/src/dark-factory while setup-host.sh renders the
    installed copy with that host's real $REPO_ROOT, so the VALUES legitimately
    differ per host while this EQUALITY holds on every one of them.
    scripts/check_dashboard_unit_parity.py's env_matches_directive applies the
    same relation to the installed copy for exactly that reason.

    Both directives are matched anchored (``^...$`` under ``re.MULTILINE``), so
    a commented-out or suffixed line does not satisfy either half — the same
    property _assert_exact_unit_line's docstring records.
    """
    content = path.read_text(encoding="utf-8")

    env_match = re.search(
        r"^Environment=DASHBOARD_PROJECT_ROOT=(.*)$", content, re.MULTILINE
    )
    assert env_match is not None, (
        f"No Environment=DASHBOARD_PROJECT_ROOT= line found in {path}. Without "
        "it the dashboard's data root falls back to Path.cwd() and is an "
        "undeclared side effect of WorkingDirectory= — see the comment above "
        "TEMPLATE_EXPECTED_PROJECT_ROOT_ENV_LINE."
    )
    wd_match = re.search(r"^WorkingDirectory=(.*)$", content, re.MULTILINE)
    assert wd_match is not None, (
        f"No WorkingDirectory= line found in {path}; the DASHBOARD_PROJECT_ROOT "
        "relation cannot be established without it."
    )

    project_root = env_match.group(1).strip()
    working_directory = wd_match.group(1).strip()
    assert project_root != "", (
        f"DASHBOARD_PROJECT_ROOT is empty or whitespace-only in {path}. "
        "config.py would then treat the empty string as the declared root "
        "rather than falling back to cwd."
    )
    assert working_directory != "", (
        f"WorkingDirectory= is empty or whitespace-only in {path}."
    )
    assert project_root == working_directory, (
        f"DASHBOARD_PROJECT_ROOT ({project_root!r}) does not equal "
        f"WorkingDirectory ({working_directory!r}) in {path}. The dashboard "
        "would then read its databases from one directory while `uv run "
        "--project dashboard` resolves relative paths against another. Repoint "
        "whichever of the two moved."
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


def test_working_directory_is_pinned_in_both_unit_files() -> None:
    """Both unit files must pin WorkingDirectory= to the repo root, exactly.

    See the module comment on TEMPLATE_EXPECTED_WORKING_DIRECTORY_LINE above
    for why the directive is load-bearing, and
    _assert_exact_unit_line's docstring for why the check is anchored
    rather than a substring match.

    Kept for targeted diagnostics — this property is subsumed by
    test_template_renders_to_hardcoded_file, but this test pinpoints which
    file's WorkingDirectory= line broke without inspecting a full-file diff.
    """
    for path, expected_line in (
        (TEMPLATE, TEMPLATE_EXPECTED_WORKING_DIRECTORY_LINE),
        (HARDCODED, HARDCODED_EXPECTED_WORKING_DIRECTORY_LINE),
    ):
        _assert_exact_unit_line(path, expected_line)


def test_project_root_env_var_is_pinned_in_both_unit_files() -> None:
    """Both unit files must declare DASHBOARD_PROJECT_ROOT explicitly (task 3572).

    See the module comment on TEMPLATE_EXPECTED_PROJECT_ROOT_ENV_LINE above for
    why the variable is declared rather than left to config.py's Path.cwd()
    fallback, and _assert_exact_unit_line's docstring for why the check is
    anchored rather than a substring match.

    The template must carry the __REPO_ROOT__ sentinel, never a hardcoded path,
    so setup-host.sh's `sed 's|__REPO_ROOT__|$REPO_ROOT|g'` makes the installed
    value track the real checkout — the same treatment
    DASHBOARD_KNOWN_PROJECT_ROOTS' self entry already gets.
    """
    for path, expected_line in (
        (TEMPLATE, TEMPLATE_EXPECTED_PROJECT_ROOT_ENV_LINE),
        (HARDCODED, HARDCODED_EXPECTED_PROJECT_ROOT_ENV_LINE),
    ):
        _assert_exact_unit_line(path, expected_line)


def test_project_root_env_matches_working_directory_in_both_unit_files() -> None:
    """DASHBOARD_PROJECT_ROOT must equal WorkingDirectory= within each unit file.

    Distinct from the exact-line tests above, which pin each directive's literal
    value independently: this pins the RELATION between them, so repointing one
    without the other fails even though both lines remain individually
    well-formed.  See _assert_project_root_env_matches_working_directory's
    docstring for why the relation is checked intra-file (it is host-invariant
    that way), and scripts/check_dashboard_unit_parity.py's env_matches_directive
    for the same relation applied to the installed copy.
    """
    for path in (TEMPLATE, HARDCODED):
        _assert_project_root_env_matches_working_directory(path)


def test_comma_separator_helper_rejects_empty_value(
    tmp_path: pathlib.Path,
) -> None:
    """_assert_known_project_roots_comma_separated must reject an empty or whitespace-only value.

    An empty DASHBOARD_KNOWN_PROJECT_ROOTS would silently produce a single empty-string
    root after split(','), which is a misconfiguration.  A whitespace-only value is
    equally broken: systemd splits unquoted Environment= values on whitespace, so a
    whitespace-only value reduces to an empty assignment.
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

    # Good: single-root value — helper must not raise (guards against over-tightening the empty check to require a comma)
    good_file = tmp_path / "single_root.service"
    good_file.write_text(
        "[Service]\nEnvironment=DASHBOARD_KNOWN_PROJECT_ROOTS=/a\n",
        encoding="utf-8",
    )
    _assert_known_project_roots_comma_separated(good_file)


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


def test_working_directory_helper_rejects_suffix_or_comment(
    tmp_path: pathlib.Path,
) -> None:
    """_assert_exact_unit_line must not be satisfied by a substring match.

    A plain ``expected in content`` check stays green if WorkingDirectory= is
    repointed to a subdirectory (``.../dashboard``) or survives only inside a
    ``#`` comment — both leave the unit's real cwd wrong while every
    substring-based assertion keeps passing.  This pins the anchored check
    against exactly those two regressions.
    """
    expected_line = "WorkingDirectory=/home/leo/src/dark-factory"

    # Bad: repointed to a subdirectory — `expected_line in content` is still
    # True, but the unit's cwd is now wrong.
    suffixed = tmp_path / "suffixed.service"
    suffixed.write_text(
        "[Service]\nWorkingDirectory=/home/leo/src/dark-factory/dashboard\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        _assert_exact_unit_line(suffixed, expected_line)

    # Bad: directive only present inside a comment — systemd never parses it.
    commented = tmp_path / "commented.service"
    commented.write_text(
        "[Service]\n# WorkingDirectory=/home/leo/src/dark-factory\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        _assert_exact_unit_line(commented, expected_line)

    # Good: exact, uncommented line — helper must not raise.
    good = tmp_path / "good.service"
    good.write_text(
        "[Service]\nWorkingDirectory=/home/leo/src/dark-factory\n",
        encoding="utf-8",
    )
    _assert_exact_unit_line(good, expected_line)


def test_project_root_env_helper_rejects_a_mismatch(tmp_path: pathlib.Path) -> None:
    """_assert_project_root_env_matches_working_directory must not silently no-op.

    Same convention as test_working_directory_helper_rejects_suffix_or_comment
    above: a relational assertion helper earns its own negatives.  Without them
    a helper whose regexes stopped matching — say after a directive was renamed
    — would find nothing, assert nothing, and keep
    test_project_root_env_matches_working_directory_in_both_unit_files green
    forever against units that had actually drifted apart.  Every case pins its
    own failure mode with ``match=`` so a bad case cannot "pass" by tripping a
    different branch than the one it names.
    """
    # Bad: the two directives disagree — the dashboard would read its databases
    # from one directory while uv resolves --project against another.
    mismatched = tmp_path / "mismatched.service"
    mismatched.write_text(
        "[Service]\n"
        "WorkingDirectory=/home/alice/src/dark-factory\n"
        "Environment=DASHBOARD_PROJECT_ROOT=/tmp/wrong\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="does not equal"):
        _assert_project_root_env_matches_working_directory(mismatched)

    # Bad: the Environment= line is absent entirely — the relation cannot hold,
    # and silence here would read as agreement.
    missing_env = tmp_path / "missing_env.service"
    missing_env.write_text(
        "[Service]\nWorkingDirectory=/home/alice/src/dark-factory\n",
        encoding="utf-8",
    )
    with pytest.raises(
        AssertionError, match="No Environment=DASHBOARD_PROJECT_ROOT= line"
    ):
        _assert_project_root_env_matches_working_directory(missing_env)

    # Bad: the Environment= line is present only inside a comment, which systemd
    # never parses — the anchored match must reject it rather than reading the
    # value out of the comment.
    commented_env = tmp_path / "commented_env.service"
    commented_env.write_text(
        "[Service]\n"
        "WorkingDirectory=/home/alice/src/dark-factory\n"
        "# Environment=DASHBOARD_PROJECT_ROOT=/home/alice/src/dark-factory\n",
        encoding="utf-8",
    )
    with pytest.raises(
        AssertionError, match="No Environment=DASHBOARD_PROJECT_ROOT= line"
    ):
        _assert_project_root_env_matches_working_directory(commented_env)

    # Bad: WorkingDirectory= absent — the other half of the relation.
    missing_wd = tmp_path / "missing_wd.service"
    missing_wd.write_text(
        "[Service]\nEnvironment=DASHBOARD_PROJECT_ROOT=/home/alice/src/dark-factory\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="No WorkingDirectory= line"):
        _assert_project_root_env_matches_working_directory(missing_wd)

    # Good: the two agree at a path that is NOT this host's — the relation is
    # host-invariant, so a correctly-configured foreign checkout must pass.
    # This is what makes the check safe to apply to an installed copy.
    good = tmp_path / "good.service"
    good.write_text(
        "[Service]\n"
        "WorkingDirectory=/home/alice/src/dark-factory\n"
        "Environment=DASHBOARD_PROJECT_ROOT=/home/alice/src/dark-factory\n",
        encoding="utf-8",
    )
    _assert_project_root_env_matches_working_directory(good)


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
    - The old misleading phrase 'no spaces' must not appear in the warning comment line above Environment=DASHBOARD_KNOWN_PROJECT_ROOTS=.

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

        # The old misleading phrase 'no spaces' must not appear in the warning comment
        assert "no spaces" not in comment_line.lower(), (
            f"Misleading phrase 'no spaces' found in the warning comment above "
            f"Environment=DASHBOARD_KNOWN_PROJECT_ROOTS= in {path}. "
            "Remove it — the real hazard is systemd's space-as-separator behavior, "
            "not the Python parser's whitespace tolerance."
        )


def test_template_renders_to_hardcoded_file() -> None:
    """Rendered template must match the committed hardcoded service file verbatim.

    This is the canonical drift-prevention invariant: applying the same substitutions
    as setup-host.sh (lines 325-329) to the template must yield the hardcoded file
    byte-for-byte.

    Substitution semantics (mirroring setup-host.sh):
        sed 's|__REPO_ROOT__|$REPO_ROOT|g'  →  str.replace('__REPO_ROOT__', HARDCODED_SERVICE_REPO_ROOT)
        sed 's|__UV_PATH__|$UV_PATH|g'      →  str.replace('__UV_PATH__', HARDCODED_SERVICE_UV_PATH)

    Both sentinels contain no regex metacharacters and no '|', so str.replace is
    semantically identical to the sed command (global, unanchored, literal substitution).

    If this test fails, the template and hardcoded file have drifted.  Re-render by
    running the sed substitutions in setup-host.sh lines 325-329 and updating
    dashboard/dark-factory-dashboard.service.
    """
    rendered = (
        TEMPLATE.read_text(encoding="utf-8")
        .replace("__REPO_ROOT__", HARDCODED_SERVICE_REPO_ROOT)
        .replace("__UV_PATH__", HARDCODED_SERVICE_UV_PATH)
    )
    hardcoded = HARDCODED.read_text(encoding="utf-8")
    assert rendered == hardcoded, (
        f"Rendered template does not match {HARDCODED}.\n"
        f"Template path: {TEMPLATE}\n"
        "The files have drifted.  Re-render by running the sed substitutions "
        "in setup-host.sh lines 325-329 and updating "
        "dashboard/dark-factory-dashboard.service."
    )


# ---------------------------------------------------------------------------
# Bounded shutdown drain
#
# Without an explicit uvicorn drain bound, a restart with a live polling client
# attached never finishes its connection drain, so systemd's TimeoutStopSec
# elapses and the unit is SIGKILLed — turning every restart into a ~16s
# contiguous dead window.  The invariant below is RELATIONAL, not existential:
# a --timeout-graceful-shutdown at or above TimeoutStopSec would still end in
# SIGKILL, so the mere presence of the flag is not the property that matters.
# ---------------------------------------------------------------------------

# Seconds that must remain between uvicorn's graceful-shutdown bound and
# systemd's SIGKILL deadline, for uvicorn's post-drain lifespan shutdown, the
# interpreter's exit, and the intermediate `uv run` parent's own teardown.
MIN_SHUTDOWN_MARGIN_SECONDS = 5

# The browser's poll interval is DERIVED from the client source, never restated
# here.  plans/dashboard-availability-prd.md records that the dashboard polls
# every 3 seconds and that the resulting keep-alive connections "never idle out
# under a 3s poll" — which is exactly why the drain never completed before the
# graceful-shutdown bound was added.  But the invariant this module protects is
# RELATIONAL (keep-alive must exceed the interval at which the browser reuses
# its connection), so a constant copied out of the JS would rot silently: bump
# the poller to 6000ms and a hardcoded `3` keeps the test green (5 > 3) while
# the real invariant (5 > 6) is violated.  Parsing the live value instead makes
# that bump fail loudly, the same way test_template_renders_to_hardcoded_file
# renders the template rather than restating its contents.
#
# CORRECTION (task 4087): deriving that value from data.js ALONE was the same
# class of rot one level up.  The browser runs more than one recurring HTTP
# poller, and each holds its own keep-alive connection and reuses it at its own
# interval — so the bound must clear the SLOWEST of them, not whichever one this
# module happened to know about.  tab_overview.jsx has polled /api/load every
# 5000ms for the entire life of this invariant while the derivation saw only
# data.js's 3000ms, leaving keep-alive=5 sitting exactly ON the boundary
# (5000 > 5000 is false) with the suite green.  CLIENT_POLLERS below makes the
# covered set explicit and _slowest_client_poll_ms() makes the bound relational
# against all of it; test_every_client_setinterval_is_registered_or_allowlisted
# makes an UNcovered new poller a loud failure rather than a silent one.
REDUX_DIR = REPO_ROOT / "dashboard" / "src" / "dashboard" / "static" / "redux"
POLL_JS = REDUX_DIR / "data.js"

# The SECOND recurring HTTP poller: HostLoadCard's /api/load poll.  It is fully
# independent of data.js's refresh — its own setInterval, its own connection,
# its own reuse interval — so the keep-alive bound has to clear it too.
LOAD_POLL_JSX = REDUX_DIR / "tab_overview.jsx"

# The poller names its period explicitly, so anchor on that name rather than on
# whichever call site happens to consume it.  An earlier revision scraped the
# sole ``setInterval(..., <literal>)`` in the file; that broke the moment the
# poller was refactored to pass a named constant (and gained unrelated
# setTimeout-based fetch deadlines), because the literal no longer appeared at
# the call site.  Binding to the declaration is both stabler and more precise.
#
# The name is a PARAMETER rather than a module constant because more than one
# client poller declares a period (see CLIENT_POLLERS below).  The patterns are
# otherwise byte-identical per name; in particular the ``^\s*(?:const|let|var)
# \s+`` anchor is what makes LOAD_POLL_INTERVAL_MS and POLL_INTERVAL_MS
# mutually non-matching, so one poller can never answer a lookup for another.
@functools.cache
def _poll_interval_patterns(const_name: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """Return the (declaration, use) regex pair for *const_name*, compiled once."""
    name = re.escape(const_name)
    return (
        re.compile(rf"^\s*(?:const|let|var)\s+{name}\s*=\s*(\d+)\s*;", re.MULTILINE),
        re.compile(rf"setInterval\([^;]*?,\s*{name}\s*\)"),
    )


def _parse_poll_interval_ms(
    source: str, origin: str, const_name: str = "POLL_INTERVAL_MS"
) -> int:
    """Return the period named *const_name* in *source*, in milliseconds.

    Reads the ``<const_name>`` declaration, and separately asserts that the
    constant is what actually drives a ``setInterval``.  Both halves must hold:
    a declaration nobody schedules on would let this module measure a dead
    constant, and a scheduler whose period we cannot read would let the
    keep-alive bound be checked against nothing.  Exactly one declaration is
    required so the period can never be ambiguous.

    *const_name* defaults to data.js's ``POLL_INTERVAL_MS``, but is a parameter
    because the browser runs more than one recurring HTTP poller and each names
    its own period — tab_overview.jsx's host-load card declares
    ``LOAD_POLL_INTERVAL_MS``.  Parameterizing the existing parser rather than
    writing a second one carries both halves of the contract above to every
    poller for free.  Every message names the REQUESTED constant, so a miss on
    one poller cannot report itself as a problem with another.
    """
    decl_re, use_re = _poll_interval_patterns(const_name)
    decls = decl_re.findall(source)
    assert len(decls) == 1, (
        f"Expected exactly one {const_name} = <literal> declaration in "
        f"{origin}, found {len(decls)}: {decls}. The keep-alive bound in this "
        "module is derived from that period; if the poller was renamed, moved, "
        "or its period made non-literal, update _parse_poll_interval_ms to "
        "identify the poll interval explicitly rather than dropping "
        "the check."
    )
    assert use_re.search(source), (
        f"{const_name} is declared in {origin} but never passed as a "
        "setInterval(...) period, so it may no longer be that poller's actual "
        "interval. Update _parse_poll_interval_ms to read "
        "whatever now schedules the poll."
    )
    return int(decls[0])


def _client_poll_interval_ms() -> int:
    """Return the browser's data-refresh poll interval, in milliseconds."""
    return _parse_poll_interval_ms(POLL_JS.read_text(encoding="utf-8"), str(POLL_JS))


class ClientPoller(typing.NamedTuple):
    """One recurring HTTP poller the browser runs, for keep-alive purposes.

    *what* is prose naming the poller, so a failure can say WHICH one set the
    keep-alive floor instead of just printing a number.
    """

    path: pathlib.Path
    const_name: str
    what: str


# Every recurring HTTP poller in the shipped client.  Membership is the property
# that matters: an unregistered poller is one the keep-alive floor does not
# clear, which is exactly how tab_overview.jsx's 5s /api/load poll went
# unnoticed.  Periods are read from the live sources at run time (never restated
# here), so there is no third copy to go stale.
#
# A new recurring poller MUST be added here.  The completeness guard
# (test_every_client_setinterval_is_registered_or_allowlisted) fails on any
# setInterval under redux/ that is neither registered here nor in
# NON_POLLING_TIMERS, so forgetting is loud rather than silent.
CLIENT_POLLERS: tuple[ClientPoller, ...] = (
    ClientPoller(POLL_JS, "POLL_INTERVAL_MS", "main data refresh"),
    ClientPoller(
        LOAD_POLL_JSX, "LOAD_POLL_INTERVAL_MS", "host-load card, /api/load"
    ),
)


def _slowest_client_poll() -> tuple[int, ClientPoller]:
    """Return (period_ms, poller) for the SLOWEST registered client poller.

    Returns the entry alongside the period so callers can name which poller set
    the keep-alive floor; a bare number leaves the reader to go find it.
    """
    measured = [
        (
            _parse_poll_interval_ms(
                entry.path.read_text(encoding="utf-8"),
                str(entry.path),
                const_name=entry.const_name,
            ),
            entry,
        )
        for entry in CLIENT_POLLERS
    ]
    return max(measured, key=lambda pair: pair[0])


def _slowest_client_poll_ms() -> int:
    """Return the slowest registered client poll interval, in milliseconds."""
    return _slowest_client_poll()[0]


# Matches a `//` line comment or a `/* ... */` block comment.  Block first, so a
# `//` inside a block does not end the line-comment match early.
_JS_COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)


def _strip_js_comments(source: str) -> str:
    """Return *source* with comments blanked out, preserving line structure.

    Comment bodies are replaced by their own newlines rather than deleted, so a
    site's line number and the file's shape survive the strip.

    Load-bearing, not cosmetic.  The live client mentions ``setInterval(...)``
    inside a ``//`` comment in tab_tasks.jsx, so a raw-text scan would report a
    call site that does not exist — training maintainers to allowlist fictions,
    and (worse) letting an allowlist entry be "satisfied" by a commented-out
    line.  This is the same hazard _uvicorn_int_flag already handles from the
    other direction, by scoping its lookup to the logical ExecStart line rather
    than the whole file, because both unit files discuss the very flags it reads
    in prose above the command.  Scope to code, never to raw file text.

    Deliberately naive about strings and regex literals: a ``//`` inside a JS
    string would over-strip.  The redux sources carry no such case, and the
    failure direction is safe — over-stripping can only HIDE a site, which the
    stale-allowlist half then surfaces rather than passing silently.
    """
    return _JS_COMMENT_RE.sub(lambda m: "\n" * m.group(0).count("\n"), source)


def _setinterval_period_args(source: str) -> list[str]:
    """Return the period argument text of every ``setInterval(`` call in *source*.

    Scans the argument list with a balanced-paren walk and takes the text after
    the LAST top-level comma.  A naive ``setInterval\\([^)]*\\)`` regex stops at
    the first ``)``, which both live non-polling timers contain in their
    callbacks — ``setInterval(() => setNow(new Date()), 1000)`` and
    ``setInterval(() => tick(n => n + 1), 2000)`` — so it would yield a garbled
    period for exactly the sites this scanner exists to classify.
    """
    periods: list[str] = []
    for match in re.finditer(r"\bsetInterval\s*\(", source):
        depth = 1
        last_comma = None
        idx = match.end()
        while idx < len(source) and depth:
            char = source[idx]
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
                if not depth:
                    break
            elif char == "," and depth == 1:
                last_comma = idx
            idx += 1
        if depth and idx >= len(source):
            # Unbalanced call — report the whole tail rather than silently
            # dropping a site the maintainer needs to see.
            periods.append(source[match.end():].strip())
            continue
        arg_start = match.end() if last_comma is None else last_comma + 1
        periods.append(source[arg_start:idx].strip())
    return periods


def _site_relpath(path: pathlib.Path) -> str:
    """Return the short ``<dir>/<file>`` key a call site is reported under.

    Two components, so the key is stable across checkouts and worktrees (an
    absolute path would make every allowlist entry machine-specific) while still
    being unambiguous and greppable within the client tree.
    """
    return f"{path.parent.name}/{path.name}"


def _unclassified_setinterval_sites(
    paths: typing.Iterable[pathlib.Path],
    known_consts: typing.AbstractSet[str],
    allowlist: typing.Mapping[tuple[str, str], str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Sort every live ``setInterval`` under *paths* into known / allowed / new.

    Returns ``(unclassified, stale)``:

    * *unclassified* — ``(relpath, period_text)`` for each call site whose period
      is neither one of *known_consts* (a registered poller's period constant)
      nor an *allowlist* key.  These are the sites nothing has decided about, so
      the keep-alive floor may not clear them.
    * *stale* — *allowlist* keys matching no live call site.  An entry that
      matches nothing checks nothing, and rots the gate into a green no-op —
      the same staleness reasoning as
      test_registry_keys_are_all_declared_in_the_committed_units in
      tests/scripts/test_check_dashboard_unit_parity.py.  Reporting it also
      keeps a commented-out site from satisfying its own allowlist entry.

    Both lists are returned rather than asserted here so the caller can name the
    concrete remedy for each; see _strip_js_comments and
    _setinterval_period_args for why the scan is done over comment-stripped
    source with a balanced-paren walk.
    """
    unclassified: list[tuple[str, str]] = []
    live: set[tuple[str, str]] = set()
    for path in paths:
        relpath = _site_relpath(path)
        code = _strip_js_comments(path.read_text(encoding="utf-8"))
        for period in _setinterval_period_args(code):
            site = (relpath, period)
            live.add(site)
            if period in known_consts or site in allowlist:
                continue
            unclassified.append(site)
    stale = sorted(key for key in allowlist if key not in live)
    return unclassified, stale


def _logical_exec_start(path: pathlib.Path) -> str:
    """Return the ExecStart= command in *path* as a single logical line.

    The dashboard unit writes ExecStart as a systemd backslash continuation
    spanning several physical lines, so a naive per-line regex would miss any
    flag that lives on a continuation line.  Joins the ExecStart= line with
    each following line while the current line ends in a backslash, dropping
    the trailing ``\\`` and collapsing continuation indentation to a single
    space.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    start_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("ExecStart=")),
        None,
    )
    assert start_idx is not None, f"No ExecStart= line found in {path}"

    parts: list[str] = []
    idx = start_idx
    while True:
        line = lines[idx].rstrip()
        continued = line.endswith("\\")
        if continued:
            line = line[:-1]
        # The first line keeps its ExecStart= prefix verbatim; continuation
        # lines are stripped so the join yields single-space separation.
        parts.append(line.rstrip() if idx == start_idx else line.strip())
        if not continued or idx + 1 >= len(lines):
            break
        idx += 1
    return " ".join(parts)


def _uvicorn_int_flag(path: pathlib.Path, flag: str) -> int | None:
    """Return the integer argument of ``--<flag>`` in *path*'s ExecStart, or None.

    Both CLI spellings are recognised.  uvicorn's parser is click-based, so
    ``--timeout-keep-alive 5`` and ``--timeout-keep-alive=5`` are equally valid
    and behave identically; accepting only the space-separated form would make
    an ``=``-form edit fail with "flag absent from ExecStart" while the flag is
    plainly there.

    The lookup is deliberately scoped to the logical ExecStart line rather than
    the whole file: both unit files discuss these same flags in the explanatory
    comment block above ExecStart, so a whole-file regex would keep reporting a
    value after the flag had actually been deleted from the command.
    """
    command = _logical_exec_start(path)
    match = re.search(rf"--{re.escape(flag)}[=\s]+(\d+)", command)
    return int(match.group(1)) if match else None


# systemd time specs this parser understands: a bare integer (seconds) or an
# integer with an s/sec/seconds suffix.  Deliberately narrow — anything else is
# reported by name so the reader extends the parser instead of guessing.
_SECONDS_SPEC_RE = re.compile(r"^(\d+)\s*(?:s|sec|secs|second|seconds)?$")


def _timeout_stop_sec(path: pathlib.Path) -> int:
    """Return the unit's TimeoutStopSec= value in seconds.

    The directive must be present: a unit without it inherits systemd's
    DefaultTimeoutStopSec, which makes any margin assertion against it
    meaningless.

    systemd accepts far more than a bare integer here — ``15s``, ``1min`` and
    ``infinity`` are all valid and all in common use — so the value is captured
    as an opaque token first and parsed second.  Matching only ``(\\d+)`` would
    report a perfectly present ``TimeoutStopSec=15s`` as a *missing* directive,
    and would give ``TimeoutStopSec=infinity`` — the single value that most
    severely breaks the margin invariant asserted below — that same misleading
    diagnosis.  The absent-directive message is therefore reserved for a
    genuinely missing line.
    """
    match = re.search(
        r"^TimeoutStopSec=(.*)$",
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match is not None, (
        f"No TimeoutStopSec= directive found in {path}. "
        "Without it the unit inherits systemd's DefaultTimeoutStopSec, so the "
        "graceful-shutdown margin cannot be verified from the unit file alone."
    )
    spec = match.group(1).strip()
    assert spec != "infinity", (
        f"TimeoutStopSec=infinity leaves the stop unbounded in {path}. "
        "systemd then never SIGKILLs a stuck shutdown, so a wedged unit hangs "
        "instead of restarting and the drain margin asserted here has no "
        "deadline to be measured against."
    )
    seconds = _SECONDS_SPEC_RE.match(spec)
    assert seconds is not None, (
        f"could not parse TimeoutStopSec={spec!r} in {path} as seconds; extend "
        "the parser. Only a bare integer or an integer with an s/sec/seconds "
        "suffix is understood today."
    )
    return int(seconds.group(1))


def _assert_drain_bounded(path: pathlib.Path) -> None:
    """Assert *path* bounds uvicorn's drain strictly below the SIGKILL deadline."""
    graceful = _uvicorn_int_flag(path, "timeout-graceful-shutdown")
    assert graceful is not None, (
        f"No --timeout-graceful-shutdown flag in the ExecStart of {path}. "
        "Without it uvicorn waits indefinitely for open connections to close; "
        "a browser polling every 3s never lets them idle out, so systemd's "
        "TimeoutStopSec elapses and the unit is SIGKILLed on every restart."
    )
    stop = _timeout_stop_sec(path)
    assert graceful < stop, (
        f"--timeout-graceful-shutdown {graceful} is not below TimeoutStopSec={stop} "
        f"in {path}. A graceful timeout at or above the stop timeout still ends in "
        "SIGKILL, so the presence of the flag alone does not bound the drain."
    )
    assert stop - graceful >= MIN_SHUTDOWN_MARGIN_SECONDS, (
        f"Only {stop - graceful}s between --timeout-graceful-shutdown {graceful} and "
        f"TimeoutStopSec={stop} in {path}; at least {MIN_SHUTDOWN_MARGIN_SECONDS}s are "
        "needed for uvicorn's post-drain lifespan shutdown, interpreter exit and the "
        "`uv run` parent's teardown."
    )


def _write_synthetic_unit(
    path: pathlib.Path,
    exec_start: str,
    timeout_stop_sec: int | str = 15,
    preamble: str = "",
) -> pathlib.Path:
    """Write a minimal synthetic unit file for guard-verification tests.

    *timeout_stop_sec* is interpolated verbatim so a test can supply any systemd
    time spec (``15s``, ``infinity``, ``1min``), not just a bare integer.
    *preamble* is inserted above ExecStart= — used to reproduce the real units'
    explanatory comment block, which mentions the very flags being looked up.
    """
    body = f"{preamble}\n" if preamble else ""
    path.write_text(
        f"[Service]\n{body}{exec_start}\nTimeoutStopSec={timeout_stop_sec}\n",
        encoding="utf-8",
    )
    return path


def test_drain_bound_guard_rejects_unbounded_units(tmp_path: pathlib.Path) -> None:
    """_assert_drain_bounded must fire on every way the drain can stay unbounded.

    Follows the same convention as test_comma_separator_helper_* above: a
    file-content assertion helper earns its own negative tests, so it cannot
    silently no-op.  Without these, a regex that failed to join the
    backslash-continued ExecStart would find no flag in ANY file and could be
    "fixed" by loosening the assertion instead of the join.

    Every case pins its own failure mode with ``match=``.  A bare
    ``pytest.raises(AssertionError)`` is satisfied by ANY assertion firing
    anywhere in the call chain, so the four bad cases — which exist precisely to
    distinguish absent / above / equal / thin-margin — would all still pass if
    they were tripping the missing-flag branch instead of the one they name.
    """
    # Bad: no --timeout-graceful-shutdown at all — the pre-fix state.
    absent = _write_synthetic_unit(
        tmp_path / "absent.service",
        "ExecStart=/usr/bin/uv run python -m uvicorn app:app --host 127.0.0.1",
    )
    with pytest.raises(AssertionError, match="No --timeout-graceful-shutdown flag"):
        _assert_drain_bounded(absent)

    # Bad: graceful timeout ABOVE the stop timeout — still SIGKILLs.
    above = _write_synthetic_unit(
        tmp_path / "above.service",
        "ExecStart=/usr/bin/uv run python -m uvicorn app:app "
        "--timeout-graceful-shutdown 20",
    )
    with pytest.raises(AssertionError, match="20 is not below TimeoutStopSec=15"):
        _assert_drain_bounded(above)

    # Bad: graceful timeout EQUAL to the stop timeout — races the SIGKILL.
    equal = _write_synthetic_unit(
        tmp_path / "equal.service",
        "ExecStart=/usr/bin/uv run python -m uvicorn app:app "
        "--timeout-graceful-shutdown 15",
    )
    with pytest.raises(AssertionError, match="15 is not below TimeoutStopSec=15"):
        _assert_drain_bounded(equal)

    # Bad: below the stop timeout but with too little margin (3s < 5s) for
    # lifespan shutdown, interpreter exit and the `uv run` parent's teardown.
    thin_margin = _write_synthetic_unit(
        tmp_path / "thin_margin.service",
        "ExecStart=/usr/bin/uv run python -m uvicorn app:app "
        "--timeout-graceful-shutdown 12",
    )
    with pytest.raises(AssertionError, match="Only 3s between"):
        _assert_drain_bounded(thin_margin)

    # Good: 8 vs 15 with the flag on a CONTINUATION line — must not raise.
    # Guards against over-tightening, and exercises the continuation join in
    # _logical_exec_start (a per-line regex would miss this flag entirely).
    good = _write_synthetic_unit(
        tmp_path / "good.service",
        "ExecStart=/usr/bin/uv run --project dashboard \\\n"
        "  python -m uvicorn app:app \\\n"
        "  --host 127.0.0.1 --port 8080 \\\n"
        "  --timeout-graceful-shutdown 8",
    )
    _assert_drain_bounded(good)

    # Good: the click ``=`` spelling is equally valid and must be accepted.
    # Rejecting it would fail with "flag absent from ExecStart" against a file
    # that plainly carries the flag.
    equals_form = _write_synthetic_unit(
        tmp_path / "equals_form.service",
        "ExecStart=/usr/bin/uv run python -m uvicorn app:app "
        "--timeout-graceful-shutdown=8",
    )
    _assert_drain_bounded(equals_form)
    assert _uvicorn_int_flag(equals_form, "timeout-graceful-shutdown") == 8


def test_uvicorn_flag_lookup_is_scoped_to_exec_start(tmp_path: pathlib.Path) -> None:
    """_uvicorn_int_flag must read the ExecStart command, not the whole file.

    Both real unit files carry an explanatory comment block above ExecStart that
    names these flags verbatim — dark-factory-dashboard.service literally
    contains "--timeout-keep-alive" followed by its value in a comment (quoted
    without the number here so this claim cannot rot on a retune).  So a future
    refactor of _uvicorn_int_flag to a whole-file regex would keep every test in
    this module green while the flag had actually been deleted from the command
    systemd runs, which is the exact regression these tests exist to catch.
    This pins the scoping so that refactor fails instead.
    """
    commented = _write_synthetic_unit(
        tmp_path / "commented.service",
        "ExecStart=/usr/bin/uv run python -m uvicorn app:app \\\n"
        "  --timeout-graceful-shutdown 8",
        preamble=(
            "# --timeout-keep-alive 5 is uvicorn's own default, pinned deliberately.\n"
            "# It is kept ABOVE the client's 3s poll interval on purpose."
        ),
    )
    assert _uvicorn_int_flag(commented, "timeout-keep-alive") is None, (
        "_uvicorn_int_flag found --timeout-keep-alive in a unit whose ExecStart "
        "does not carry it — the lookup is matching the comment block instead of "
        "the command. Scope it to _logical_exec_start()."
    )
    # The flag that IS on the command is still found, so the scoping did not
    # over-tighten into finding nothing at all.
    assert _uvicorn_int_flag(commented, "timeout-graceful-shutdown") == 8


def test_timeout_stop_sec_parses_systemd_time_specs(tmp_path: pathlib.Path) -> None:
    """_timeout_stop_sec must distinguish absent / infinity / unparseable / valid.

    systemd accepts unit suffixes and the literal ``infinity`` here.  Collapsing
    all of those into "No TimeoutStopSec= directive found" misdirects the fixer
    precisely where the diagnosis matters most: ``infinity`` is the one value
    that most severely breaks the margin invariant, and it must say so.
    """
    exec_start = (
        "ExecStart=/usr/bin/uv run python -m uvicorn app:app "
        "--timeout-graceful-shutdown 8"
    )

    # Good: bare integer, and the same value with each accepted suffix.
    for spec in ("15", "15s", "15sec", "15secs", "15second", "15seconds"):
        unit = _write_synthetic_unit(
            tmp_path / f"stop_{spec}.service", exec_start, timeout_stop_sec=spec
        )
        assert _timeout_stop_sec(unit) == 15, (
            f"TimeoutStopSec={spec} should parse as 15 seconds"
        )
        # A suffixed spec must not fall out of the drain guard either.
        _assert_drain_bounded(unit)

    # Bad: genuinely missing directive — this message is reserved for that case.
    missing = tmp_path / "missing.service"
    missing.write_text(f"[Service]\n{exec_start}\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="No TimeoutStopSec= directive found"):
        _timeout_stop_sec(missing)

    # Bad: infinity — valid systemd, but leaves the stop unbounded.
    infinite = _write_synthetic_unit(
        tmp_path / "infinite.service", exec_start, timeout_stop_sec="infinity"
    )
    with pytest.raises(AssertionError, match="TimeoutStopSec=infinity"):
        _timeout_stop_sec(infinite)

    # Bad: a spec this parser does not understand — say so, don't claim absence.
    compound = _write_synthetic_unit(
        tmp_path / "compound.service", exec_start, timeout_stop_sec="1min 30s"
    )
    with pytest.raises(AssertionError, match="could not parse"):
        _timeout_stop_sec(compound)


def test_poll_interval_parser_requires_exactly_one_interval() -> None:
    """_parse_poll_interval_ms must fail loudly on zero or ambiguous matches.

    The keep-alive bound is derived from this parse, so a silent miss would turn
    the relational assertion into a no-op against a value nobody supplied.
    """
    assert (
        _parse_poll_interval_ms(
            "const POLL_INTERVAL_MS = 3000;\n"
            "const handle = setInterval(() => pollTick(opts), POLL_INTERVAL_MS);\n",
            "synthetic",
        )
        == 3000
    )

    with pytest.raises(AssertionError, match="found 0"):
        _parse_poll_interval_ms("refreshDFData();\n", "synthetic")

    with pytest.raises(AssertionError, match="found 2"):
        _parse_poll_interval_ms(
            "const POLL_INTERVAL_MS = 3000;\n"
            "const POLL_INTERVAL_MS = 9000;\n"
            "setInterval(tick, POLL_INTERVAL_MS);\n",
            "synthetic",
        )

    # Declared but nothing schedules on it: the constant may be vestigial, so the
    # keep-alive bound would be measured against a period the browser never uses.
    with pytest.raises(AssertionError, match="never passed as a setInterval"):
        _parse_poll_interval_ms(
            "const POLL_INTERVAL_MS = 3000;\nsetInterval(tick, 250);\n", "synthetic"
        )


def test_poll_interval_parser_reads_an_arbitrary_constant_name() -> None:
    """_parse_poll_interval_ms must parse ANY named period, not just POLL_INTERVAL_MS.

    More than one client poller declares a period — data.js's POLL_INTERVAL_MS
    (the main data refresh) and tab_overview.jsx's LOAD_POLL_INTERVAL_MS (the
    host-load card's /api/load poll) — and the keep-alive lower bound is derived
    from the SLOWEST of them.  Rather than growing a second bespoke parser, the
    existing declaration-anchored one takes the constant name as a parameter, so
    both halves of its contract (exactly-one-declaration, and the declared
    constant actually schedules a setInterval) carry over to every poller for
    free instead of being re-derived per poller.

    Two properties beyond "it parses" are pinned here.  The failure messages must
    name the REQUESTED constant: a LOAD_POLL_INTERVAL_MS miss that reported
    itself as a POLL_INTERVAL_MS problem would send the fixer to the wrong file.
    And the two names must not cross-match, or a max over the poller registry
    could be computed twice from the same source with the other poller silently
    dropping out — the exact silent-coverage failure this registry exists to
    remove.
    """
    assert (
        _parse_poll_interval_ms(
            "const LOAD_POLL_INTERVAL_MS = 5000;\n"
            "const id = setInterval(fetchLoad, LOAD_POLL_INTERVAL_MS);\n",
            "synthetic",
            const_name="LOAD_POLL_INTERVAL_MS",
        )
        == 5000
    )

    # Both halves of the contract still hold for the requested name, and each
    # failure names the constant it was actually looking for.
    with pytest.raises(AssertionError, match="found 0") as absent:
        _parse_poll_interval_ms(
            "fetchLoad();\n", "synthetic", const_name="LOAD_POLL_INTERVAL_MS"
        )
    assert "LOAD_POLL_INTERVAL_MS" in str(absent.value)

    with pytest.raises(AssertionError, match="found 2") as ambiguous:
        _parse_poll_interval_ms(
            "const LOAD_POLL_INTERVAL_MS = 5000;\n"
            "const LOAD_POLL_INTERVAL_MS = 9000;\n"
            "setInterval(fetchLoad, LOAD_POLL_INTERVAL_MS);\n",
            "synthetic",
            const_name="LOAD_POLL_INTERVAL_MS",
        )
    assert "LOAD_POLL_INTERVAL_MS" in str(ambiguous.value)

    with pytest.raises(AssertionError, match="never passed as a setInterval") as dead:
        _parse_poll_interval_ms(
            "const LOAD_POLL_INTERVAL_MS = 5000;\nsetInterval(fetchLoad, 250);\n",
            "synthetic",
            const_name="LOAD_POLL_INTERVAL_MS",
        )
    assert "LOAD_POLL_INTERVAL_MS" in str(dead.value)

    # The two live names must not cross-match in EITHER direction. The
    # ``^\s*(?:const|let|var)\s+`` anchor is what keeps LOAD_POLL_INTERVAL_MS
    # from satisfying a POLL_INTERVAL_MS lookup by suffix; a substring match
    # would let one poller answer for both.
    with pytest.raises(AssertionError, match="found 0"):
        _parse_poll_interval_ms(
            "const POLL_INTERVAL_MS = 3000;\n"
            "setInterval(() => pollTick(opts), POLL_INTERVAL_MS);\n",
            "synthetic",
            const_name="LOAD_POLL_INTERVAL_MS",
        )

    with pytest.raises(AssertionError, match="found 0"):
        _parse_poll_interval_ms(
            "const LOAD_POLL_INTERVAL_MS = 5000;\n"
            "setInterval(fetchLoad, LOAD_POLL_INTERVAL_MS);\n",
            "synthetic",
        )


def _write_synthetic_client(path: pathlib.Path, source: str) -> pathlib.Path:
    """Write a synthetic client-side source file for scanner-verification tests.

    The sibling of _write_synthetic_unit for the JS side: the scanner's own
    behaviour is pinned against sources this module controls, so the live-tree
    guard is the only test coupled to the real dashboard client.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_setinterval_scanner_classifies_every_call_site(
    tmp_path: pathlib.Path,
) -> None:
    """_unclassified_setinterval_sites must sort live timers into known/allowed/new.

    This is the guard-the-guard test for the completeness check: a scanner that
    silently reported nothing would turn
    test_every_client_setinterval_is_registered_or_allowlisted into a green
    no-op, which is the exact failure mode that let a second poller go
    unnoticed in the first place.

    Two hazards get their own cases.  Comment immunity: the live tree mentions
    ``setInterval(...)`` inside a ``//`` comment at tab_tasks.jsx, so a raw-text
    scan would report a phantom site and train maintainers to allowlist
    fictions.  That is the same trap _uvicorn_int_flag already handles from the
    other direction by scoping to the logical ExecStart line rather than the
    whole file (see test_uvicorn_flag_lookup_is_scoped_to_exec_start).  And
    nested calls: a naive ``setInterval\\([^)]*\\)`` match stops at the FIRST
    ``)``, so ``setInterval(() => tick(n => n + 1), 2000)`` would yield a
    garbled period — both live non-polling timers have exactly this shape.
    """
    redux = tmp_path / "redux"

    # Registered: driven by a known poller constant, so already covered.
    registered = _write_synthetic_client(
        redux / "data.js",
        "const POLL_INTERVAL_MS = 3000;\n"
        "setInterval(() => pollTick(opts), POLL_INTERVAL_MS);\n",
    )
    # Allowlisted: a pure UI tick, and a NESTED call in the callback.
    allowlisted = _write_synthetic_client(
        redux / "app.jsx",
        "setInterval(() => setNow(new Date()), 1000);\n",
    )
    # Unregistered and unallowlisted — must be reported.
    unknown = _write_synthetic_client(
        redux / "tab_thing.jsx",
        "setInterval(fetchThing, 4000);\n",
    )
    # Also unclassified, and nested two deep: pins the balanced-paren scan.
    nested = _write_synthetic_client(
        redux / "shell.jsx",
        "setInterval(() => tick(n => n + 1), 2000);\n",
    )
    # Comments only: a `//` mention and a `/* */` block. Neither is live code.
    commented = _write_synthetic_client(
        redux / "tab_tasks.jsx",
        "// ...`setInterval(() => setNow(...), 1000)`, unrelated\n"
        "/* legacy, removed:\n"
        "setInterval(fetchOld, 7000);\n"
        "*/\n",
    )

    allowlist = {
        ("redux/app.jsx", "1000"): "clock tick — opens no connection",
        # Points at a call site that now exists only inside a comment.
        ("redux/tab_tasks.jsx", "1000"): "checks nothing anymore",
    }
    unclassified, stale = _unclassified_setinterval_sites(
        [registered, allowlisted, unknown, nested, commented],
        known_consts={"POLL_INTERVAL_MS"},
        allowlist=allowlist,
    )

    # (a) known constant, (b) allowlisted, (d) comment-only: none reported.
    # (c) the two genuinely new timers are, each named with its file and period.
    assert sorted(unclassified) == [
        ("redux/shell.jsx", "2000"),
        ("redux/tab_thing.jsx", "4000"),
    ], (
        "Scanner must report exactly the unregistered, unallowlisted call sites "
        f"— with their periods — and nothing else; got {sorted(unclassified)}. "
        "A phantom from a commented-out site would train maintainers to "
        "allowlist fictions; a missed real site is a poller the keep-alive "
        "floor does not clear."
    )

    # (e) an allowlist entry whose site is now only a comment is STALE, not
    # silently satisfied — otherwise the allowlist rots into a no-op gate.
    assert stale == [("redux/tab_tasks.jsx", "1000")], (
        "An allowlist entry matching no live call site must be reported stale; "
        f"got {stale}."
    )


def test_client_poll_interval_is_readable_from_the_shipped_client() -> None:
    """The live client source must still yield a poll interval to compare against.

    Guards the derivation itself: if data.js moves or its poller is rewritten,
    this fails here with a clear cause rather than silently weakening
    test_keep_alive_timeout_is_pinned_above_poll_interval.
    """
    assert POLL_JS.is_file(), (
        f"Client poll source not found at {POLL_JS}. The keep-alive lower bound "
        "is derived from it; update POLL_JS if the client moved."
    )
    assert _client_poll_interval_ms() > 0


def test_host_load_poller_declares_a_named_interval() -> None:
    """The host-load card's /api/load poller must expose a parseable period too.

    HostLoadCard in tab_overview.jsx runs an INDEPENDENT second HTTP poller: it
    fetches /api/load on its own setInterval, entirely separate from data.js's
    data refresh.  It therefore holds its own keep-alive connection and reuses
    it at its own interval, so its period is a load-bearing input to the
    keep-alive lower bound and must be derivable — a poller whose period cannot
    be read is a poller the bound cannot clear.

    Naming the period rather than leaving a bare ``setInterval(fetchLoad, 5000)``
    literal is what makes that derivation stable.  This module already learned
    and recorded that lesson for data.js (see the comment above
    _poll_interval_patterns): scraping the literal at the call site broke the
    moment the poller was refactored to pass a named constant, and binding to
    the declaration is both stabler and more precise.  The same reasoning
    applies here, and the declaration additionally gives the JSX a place to
    carry the "this feeds the systemd keep-alive bound" warning.
    """
    assert LOAD_POLL_JSX.is_file(), (
        f"Host-load poll source not found at {LOAD_POLL_JSX}. The keep-alive "
        "lower bound is derived from it; update LOAD_POLL_JSX if the host-load "
        "card moved."
    )
    assert (
        _parse_poll_interval_ms(
            LOAD_POLL_JSX.read_text(encoding="utf-8"),
            str(LOAD_POLL_JSX),
            const_name="LOAD_POLL_INTERVAL_MS",
        )
        > 0
    )


def test_slowest_client_poll_is_derived_from_every_registered_poller() -> None:
    """The keep-alive floor must come from the SLOWEST poller, not from data.js.

    CLIENT_POLLERS is the explicit statement of which recurring HTTP pollers the
    keep-alive bound covers, and _slowest_client_poll_ms() reduces it with max()
    — because each poller independently reuses its own connection at its own
    interval, so a bound that clears only the fastest one still closes the
    slower socket in the gap between its polls.

    The registry gets the same staleness guard as
    test_registry_keys_are_all_declared_in_the_committed_units in
    tests/scripts/test_check_dashboard_unit_parity.py: every registered path must
    exist and every registered constant must still parse, so a moved file or a
    renamed constant fails loudly here instead of silently dropping a poller out
    of the max and quietly lowering the floor.

    The last assertion pins the defect this task exists to remove: the slowest
    poller must be strictly slower than data.js's, proving the registry has not
    collapsed back to a single entry.  It had exactly one for the whole life of
    the keep-alive invariant, while a 5s /api/load poller sat unnoticed on the
    boundary.
    """
    registered = {(entry.path, entry.const_name) for entry in CLIENT_POLLERS}
    assert (POLL_JS, "POLL_INTERVAL_MS") in registered, (
        "data.js's main data refresh is missing from CLIENT_POLLERS; the "
        "keep-alive floor would stop covering it."
    )
    assert (LOAD_POLL_JSX, "LOAD_POLL_INTERVAL_MS") in registered, (
        "tab_overview.jsx's /api/load host-load poll is missing from "
        "CLIENT_POLLERS. It is an independent recurring HTTP poller and holds "
        "its own keep-alive connection, so the floor must clear it too."
    )

    periods = []
    for entry in CLIENT_POLLERS:
        assert entry.path.is_file(), (
            f"CLIENT_POLLERS registers {entry.path} ({entry.what}) but that file "
            "does not exist. Update the registry entry rather than leaving a "
            "poller silently uncovered by the keep-alive floor."
        )
        period = _parse_poll_interval_ms(
            entry.path.read_text(encoding="utf-8"),
            str(entry.path),
            const_name=entry.const_name,
        )
        assert period > 0
        periods.append(period)

    assert _slowest_client_poll_ms() == max(periods), (
        "_slowest_client_poll_ms() must be the max over every registered "
        f"poller; got {_slowest_client_poll_ms()} against {periods}."
    )

    data_js_ms = _parse_poll_interval_ms(
        POLL_JS.read_text(encoding="utf-8"), str(POLL_JS)
    )
    assert _slowest_client_poll_ms() > data_js_ms, (
        f"The slowest registered poller ({_slowest_client_poll_ms()}ms) is not "
        f"strictly slower than data.js's {data_js_ms}ms poll, so the registry "
        "has effectively collapsed back to the single-poller derivation this "
        "guard exists to prevent. tab_overview.jsx polls /api/load every 5000ms "
        "and must be part of the max."
    )


def test_every_client_setinterval_is_registered_or_allowlisted() -> None:
    """Every recurring timer in the client must be classified, not merely absent.

    This is the guard that would have caught the defect this module now carries
    a correction for.  tab_overview.jsx polled /api/load every 5000ms for the
    entire life of the keep-alive invariant; the derivation read data.js and
    nothing else, so the second poller was never part of the bound and
    keep-alive sat exactly ON its boundary (5000 > 5000 is false) with the whole
    suite green.  Nothing failed, because nothing was looking.

    So membership is checked from the CLIENT side, not just the registry side:
    every setInterval under the shipped redux tree must be accounted for by one
    of two explicit decisions, and an unclassified one fails loudly with both
    remedies named.  Files are globbed live rather than enumerated, so a new
    client file is covered the moment it lands rather than when someone
    remembers to add it here.
    """
    sources = sorted(
        path
        for path in REDUX_DIR.rglob("*")
        if path.suffix in (".js", ".jsx") and path.is_file()
    )
    assert sources, (
        f"No .js/.jsx sources found under {REDUX_DIR}. This guard would pass "
        "vacuously — update REDUX_DIR to wherever the client moved rather than "
        "leaving every client timer unchecked."
    )

    unclassified, stale = _unclassified_setinterval_sites(
        sources,
        known_consts={entry.const_name for entry in CLIENT_POLLERS},
        allowlist=NON_POLLING_TIMERS,
    )

    assert not unclassified, (
        "Unclassified setInterval call site(s) in the dashboard client: "
        f"{unclassified}. Every recurring timer must be an explicit decision, "
        "because a timer that performs HTTP holds its own keep-alive connection "
        "and reuses it at its own interval — which is how tab_overview.jsx's 5s "
        "/api/load poll went unnoticed while the keep-alive floor was derived "
        "from data.js alone.\n"
        "  - If it PERFORMS HTTP: name its period as a *_POLL_INTERVAL_MS "
        "constant and register it in CLIENT_POLLERS. If it is the slowest "
        "poller this raises the --timeout-keep-alive floor, and "
        "test_keep_alive_timeout_is_pinned_above_poll_interval will say so.\n"
        "  - If it is a pure UI tick that opens NO connection: add it to "
        "NON_POLLING_TIMERS with a reason."
    )
    assert not stale, (
        f"NON_POLLING_TIMERS entries matching no live call site: {stale}. "
        "An entry that matches nothing checks nothing, so it silently rots this "
        "gate — remove it, or fix it to match the timer it was meant to cover. "
        "Note a commented-out call site does not count as live."
    )


def test_shutdown_drain_is_bounded_in_both_unit_files() -> None:
    """Both unit files must bound uvicorn's drain below systemd's SIGKILL deadline.

    uvicorn hard-bounds the connection drain: server.py wraps
    _wait_tasks_to_complete() in asyncio.wait_for(timeout=timeout_graceful_shutdown)
    and force-cancels every remaining task on TimeoutError.  Without the flag the
    drain is unbounded and every restart ends in
    "State 'stop-sigterm' timed out. Killing." → SIGKILL.
    """
    for path in (TEMPLATE, HARDCODED):
        _assert_drain_bounded(path)


def test_keep_alive_timeout_is_pinned_above_poll_interval() -> None:
    """--timeout-keep-alive must be pinned explicitly, and pinned ABOVE the poll interval.

    The lower bound is the non-obvious half of this test.  The tempting move —
    dropping keep-alive below the slowest client poll so idle connections close
    on their own — is wrong here: it would make the server close that polling
    socket in the gap between its polls, exposing the classic
    server-closes-while-client-writes race, i.e. trading a shutdown-time stall
    for request-time failures on a change whose whole purpose is availability.
    The hard drain guarantee already comes from --timeout-graceful-shutdown, so
    keep-alive is not being asked to do that job.

    The bound must clear EVERY recurring HTTP poller, not just the busiest one.
    Each poller holds its own keep-alive connection and reuses it at its OWN
    interval, so a keep-alive above data.js's 3s data refresh but at-or-below
    tab_overview.jsx's 5s /api/load poll still exposes that race on the slower
    socket — which is precisely the state this test used to certify as green
    (see the CORRECTION note above CLIENT_POLLERS).  Hence the comparison is
    against _slowest_client_poll_ms().

    Pinning it explicitly is still worth doing beyond the bound itself: it
    surfaces the interaction that caused the incident — keep-alive exceeds the
    poll interval, which is precisely why the connection never idles out —
    makes it greppable, and gives the unit-file parity check a concrete
    directive to diff.

    Every period is read out of the live client sources rather than restated,
    so raising a poller past the pinned keep-alive fails here instead of passing
    against a stale copy of the number.  The comparison is done in milliseconds
    to keep a non-whole-second poll (e.g. 3500ms) from being floored into a
    looser bound than the client actually uses.
    """
    poll_ms, slowest = _slowest_client_poll()
    for path in (TEMPLATE, HARDCODED):
        keep_alive = _uvicorn_int_flag(path, "timeout-keep-alive")
        assert keep_alive is not None, (
            f"No --timeout-keep-alive flag in the ExecStart of {path}. "
            "Leaving it implicit hides the interaction that caused the incident: "
            "keep-alive exceeds the slowest client poll — "
            f"{poll_ms / 1000:g}s, from {slowest.what} ({slowest.path}) — which "
            "is why the polling connection never idles out."
        )
        assert keep_alive * 1000 > poll_ms, (
            f"--timeout-keep-alive {keep_alive}s is not above the slowest client "
            f"poll interval of {poll_ms / 1000:g}s — {slowest.what}, "
            f"{slowest.const_name} in {slowest.path} — in {path}. "
            "Below that poller's interval the server closes its socket in the "
            "gap between polls, exposing the server-closes-while-client-writes "
            "race and turning a shutdown fix into a source of failed polls. The "
            "drain is already bounded by --timeout-graceful-shutdown; do not "
            "retune keep-alive to compensate for it — raise keep-alive above the "
            "new poll interval (in BOTH unit files), or lower the poll interval."
        )
        stop = _timeout_stop_sec(path)
        assert keep_alive < stop, (
            f"--timeout-keep-alive {keep_alive} is not below TimeoutStopSec={stop} "
            f"in {path}. A keep-alive idle window longer than the stop timeout "
            "would be incoherent with the drain bound."
        )


# ---------------------------------------------------------------------------
# Restart backoff
#
# The invariant and its directive reader now live in systemd_unit_invariants,
# imported at the top of this module under the private aliases this file has
# always used.  They were lifted out of here by task 3408 so that the
# fleet-wide sweep in test_systemd_restart_backoff.py applies the SAME
# assertion rather than a second copy of it — see that module's docstring.
#
# Nothing asserts that the sharing holds, deliberately: the `from
# systemd_unit_invariants import (...)` at the top of this module IS the
# sharing, structurally, and a drifted private copy would be caught by that
# fleet sweep failing on whichever unit the copy stopped checking.  An earlier
# meta-test pinning it by object identity was dropped as testing the shape of
# the refactor rather than any behaviour of a systemd unit.
#
# The negative-case guard for the helper stays here, below, next to the
# drain-bound guard it is modelled on.
# ---------------------------------------------------------------------------


def _write_restart_unit(path: pathlib.Path, body: str) -> pathlib.Path:
    """Write a minimal synthetic unit carrying only a [Service] restart stanza.

    _write_synthetic_unit above is shaped for the ExecStart/TimeoutStopSec drain
    tests and would have to be contorted through its *preamble* argument to
    express a restart stanza, so the restart guards get their own one-line
    builder rather than bending a fixture that means something else.
    """
    path.write_text(f"[Service]\n{body}\n", encoding="utf-8")
    return path


def test_restart_backoff_guard_rejects_ineffective_units(
    tmp_path: pathlib.Path,
) -> None:
    """_assert_restart_backoff_effective must fire on every inert-cap spelling.

    Same convention as test_drain_bound_guard_rejects_unbounded_units above: an
    assertion helper earns its own negatives so it cannot silently no-op, and
    every case pins its own failure mode with ``match=``.  A bare
    ``pytest.raises(AssertionError)`` is satisfied by ANY assertion firing
    anywhere in the chain, so the distinct bad cases below — absent steps / zero
    steps / absent floor — would all still "pass" while tripping one shared
    branch, which is exactly the confusion they exist to rule out.
    """
    # Bad: the pre-fix state — a cap declared with nothing to interpolate over.
    no_steps = _write_restart_unit(
        tmp_path / "no_steps.service",
        "Restart=on-failure\nRestartSec=5\nRestartMaxDelaySec=60",
    )
    with pytest.raises(AssertionError, match="but no RestartSteps="):
        _assert_restart_backoff_effective(no_steps)

    # Bad: present but zero — a zero-step curve is no backoff at all, so mere
    # presence of the directive is not the property being asserted.
    zero_steps = _write_restart_unit(
        tmp_path / "zero_steps.service",
        "Restart=on-failure\nRestartSec=5\nRestartSteps=0\nRestartMaxDelaySec=60",
    )
    with pytest.raises(AssertionError, match="produces no backoff curve"):
        _assert_restart_backoff_effective(zero_steps)

    # Bad: overridden to zero by a later repeat.  systemd is LAST-WINS for these
    # scalars, so this unit's effective RestartSteps is 0 and its cap is
    # discarded — measured on systemd 255.4, which emits the pairing warning for
    # exactly this file while staying silent on the reverse order (0 then 4).
    # A first-match read of the directive would report 4 and pass, so this case
    # pins _restart_directive's last-wins semantics through the guard that
    # depends on them.
    overridden_steps = _write_restart_unit(
        tmp_path / "overridden_steps.service",
        "Restart=on-failure\nRestartSec=5\nRestartSteps=4\n"
        "RestartSteps=0\nRestartMaxDelaySec=60",
    )
    with pytest.raises(AssertionError, match="produces no backoff curve"):
        _assert_restart_backoff_effective(overridden_steps)
    assert _restart_directive(overridden_steps, "RestartSteps") == "0"

    # Bad: cap and steps, but no floor to interpolate up FROM.
    no_floor = _write_restart_unit(
        tmp_path / "no_floor.service",
        "Restart=on-failure\nRestartSteps=4\nRestartMaxDelaySec=60",
    )
    with pytest.raises(AssertionError, match="no RestartSec="):
        _assert_restart_backoff_effective(no_floor)

    # Good: the full curve — must not raise (guards against over-tightening).
    good = _write_restart_unit(
        tmp_path / "good.service",
        "Restart=on-failure\nRestartSec=5\nRestartSteps=4\nRestartMaxDelaySec=60",
    )
    _assert_restart_backoff_effective(good)

    # Good: no cap declared at all — the invariant is conditional on the cap
    # being declared, not a blanket requirement that every unit implement
    # backoff.  Pins that a later author cannot tighten it into an
    # unconditional "RestartSteps= must be present".
    no_cap = _write_restart_unit(
        tmp_path / "no_cap.service", "Restart=on-failure\nRestartSec=5"
    )
    _assert_restart_backoff_effective(no_cap)

    # Good: a suffixed spec is still a declared cap.  This is why
    # _restart_directive captures the value opaquely — a (\d+) parse would read
    # `60s` as an absent directive and skip the invariant without a word.
    suffixed = _write_restart_unit(
        tmp_path / "suffixed.service",
        "Restart=on-failure\nRestartSec=5s\nRestartSteps=4\nRestartMaxDelaySec=60s",
    )
    _assert_restart_backoff_effective(suffixed)
    assert _restart_directive(suffixed, "RestartMaxDelaySec") == "60s"

    # ...and the inert form of that same spelling must STILL be caught, so the
    # opaque capture above bought precision, not laxity.
    suffixed_no_steps = _write_restart_unit(
        tmp_path / "suffixed_no_steps.service",
        "Restart=on-failure\nRestartSec=5s\nRestartMaxDelaySec=60s",
    )
    with pytest.raises(AssertionError, match="but no RestartSteps="):
        _assert_restart_backoff_effective(suffixed_no_steps)

    # Bad: the same inert cap, spelled with the leading and around-separator
    # whitespace systemd.syntax permits.  This is a VALID unit whose cap systemd
    # parses and discards, so it must be caught — a column-0-only anchor reads
    # it as declaring no cap at all and returns early, skipping the invariant
    # without a word.  That is the one failure direction this guard cannot
    # afford: it masks the very defect the helper exists to surface, unlike the
    # opaque-value capture above, which errs toward checking MORE.
    spaced_no_steps = _write_restart_unit(
        tmp_path / "spaced_no_steps.service",
        "Restart=on-failure\n  RestartSec = 5\n\tRestartMaxDelaySec\t=\t60",
    )
    with pytest.raises(AssertionError, match="but no RestartSteps="):
        _assert_restart_backoff_effective(spaced_no_steps)
    assert _restart_directive(spaced_no_steps, "RestartMaxDelaySec") == "60"

    # Good: the whitespace-tolerant read is complete, not just detection-only —
    # a properly paired unit in that same spelling must not raise.
    spaced_good = _write_restart_unit(
        tmp_path / "spaced_good.service",
        "Restart=on-failure\n  RestartSec = 5\n  RestartSteps = 4\n"
        "  RestartMaxDelaySec = 60",
    )
    _assert_restart_backoff_effective(spaced_good)


def test_restart_backoff_is_effective_in_both_unit_files() -> None:
    """Both unit files must pair RestartMaxDelaySec= with a usable RestartSteps=.

    Without the pairing systemd discards the cap at load time, so a dashboard
    stuck in a crash loop retries every 5s indefinitely instead of backing off
    toward 60s — the unit's comment claims a backoff that the unit does not do.
    """
    for path in (TEMPLATE, HARDCODED):
        _assert_restart_backoff_effective(path)


def _ignored_directive_lines(output: str, unit: pathlib.Path) -> list[str]:
    """Return systemd-analyze lines reporting a discarded directive in *unit*.

    Split out of the test below so it can be exercised against captured
    systemd-analyze output WITHOUT a systemd runtime — the filter is the part
    that can silently no-op, and a guard that only runs where systemd is
    installed is no guard at all.  See
    test_ignored_directive_filter_matches_both_systemd_spellings for the
    negatives; the measured samples it pins are quoted there verbatim.
    """
    return [
        line
        for line in output.splitlines()
        # Case-insensitive: systemd writes "Ignoring." for a semantic discard
        # and "ignoring." for an unknown key, and on systemd < 254 the latter is
        # the ONLY form this unit's backoff directives can produce.
        if "ignoring." in line.lower()
        # Scoped to the unit under test: verify also loads dependencies, and
        # each warning names the unit it belongs to.  A semantic warning is
        # prefixed with the bare unit name, a parse warning with the full path
        # and line number.
        and (line.startswith(f"{unit.name}:") or str(unit) in line)
    ]


def test_ignored_directive_filter_matches_both_systemd_spellings() -> None:
    """_ignored_directive_lines must catch both casings and only this unit.

    The samples below are real ``systemd-analyze verify`` output captured on
    2026-08-01 (systemd 255.4), not invented strings: the capital-I form from a
    unit with RestartMaxDelaySec= and no RestartSteps=, the lowercase form from
    a deliberately misspelled ``RestartStepz=``, and the dependency noise from
    ``systemd-analyze verify --user`` on the real dashboard unit.

    This is the guard that keeps the systemd test below honest.  Both of its
    ways of degrading into a silent pass — wrong casing, or another unit's
    warning counted as this one's — are pinned here, and they are pinned
    without needing systemd, so they hold on every host.
    """
    semantic = (
        f"{HARDCODED.name}: Service has RestartMaxDelaySec= but no "
        "RestartSteps= setting. Ignoring."
    )
    parse = (
        f"{HARDCODED}:42: Unknown key name 'RestartStepz' in section "
        "'Service', ignoring."
    )
    # Another unit's identical defect, plus genuinely unrelated noise.
    foreign = (
        "orchestrator-dark-factory.service: Service has RestartMaxDelaySec= "
        "but no RestartSteps= setting. Ignoring."
    )
    noise = "Failed to bind private socket: Address already in use"

    # Caught: the capital-I semantic warning...
    assert _ignored_directive_lines(semantic, HARDCODED) == [semantic]
    # ...and the lowercase parse warning, which is the ONLY signal available on
    # systemd < 254, where both backoff directives degrade to unknown keys.
    assert _ignored_directive_lines(parse, HARDCODED) == [parse]

    # Ignored: a warning that belongs to a different unit must not be blamed on
    # this file, and non-warning chatter must not trip the assertion.
    assert _ignored_directive_lines(f"{foreign}\n{noise}", HARDCODED) == []

    # Mixed stream: exactly the two lines that name this unit, in order.
    combined = "\n".join([noise, foreign, semantic, parse])
    assert _ignored_directive_lines(combined, HARDCODED) == [semantic, parse]

    # A clean run reports nothing.
    assert _ignored_directive_lines("", HARDCODED) == []


@pytest.mark.skipif(
    shutil.which("systemd-analyze") is None,
    reason=(
        "systemd-analyze is not installed; the file-content invariants above are "
        "the primary guard and this module requires no systemd runtime"
    ),
)
def test_systemd_analyze_verify_reports_no_ignored_directives() -> None:
    """systemd itself must not report discarding any directive in the unit.

    Confirmatory corroboration of the file-content invariant above, deliberately
    skippable so the module keeps the "no systemd runtime is required" contract
    stated in its docstring.

    Five details here are load-bearing, each verified by measurement rather
    than assumed (2026-08-01, systemd 255.4):

    1. stderr is captured, not just stdout.  The warning goes to stderr while
       stdout is empty, so a stdout-only assertion would be a silent no-op that
       passes forever regardless of the unit's contents.
    2. The assertion is on the WARNING TEXT, never on ``returncode``.
       ``systemd-analyze verify`` exits 1 when the ExecStart binary does not
       exist ("Command /... is not executable"), and this unit bakes in
       /home/leo/.local/bin/uv — so a returncode check would fail on any other
       host for a reason entirely unrelated to this invariant.  (It exits 0 even
       while emitting the warnings this test hunts, so the code carries no
       signal in either direction.)
    3. It runs against HARDCODED only.  The raw template cannot be verified at
       all: its ``__REPO_ROOT__`` sentinel trips a fatal "WorkingDirectory= path
       is not absolute" error.  test_template_renders_to_hardcoded_file already
       asserts the two files are byte-identical modulo the sentinels, so the
       property carries across without verifying the same bytes twice.
    4. The match is CASE-INSENSITIVE, because systemd spells its two discard
       paths differently and only one of them is the pairing warning:
         - ``t1.service: Service has RestartMaxDelaySec= but no RestartSteps=
           setting. Ignoring.``           (capital I)
         - ``/path/t2.service:8: Unknown key name 'RestartStepz' in section
           'Service', ignoring.``          (lowercase)
       This matters precisely where the guard is needed most.  RestartSteps= AND
       RestartMaxDelaySec= both landed in systemd v254, so on any host below
       that — and scripts/setup-host.sh renders this template onto arbitrary
       machines — the whole backoff fix degrades to two unknown keys, systemd
       emits only the lowercase form, and a capital-I filter would report green
       on a unit doing no backoff at all.  Matching one casing would blind this
       test to its single most important failure mode.
    5. Warnings are SCOPED to the unit under test.  ``systemd-analyze verify``
       loads the unit's dependencies too and prefixes each warning with the
       offending unit's own name, so an unscoped filter blames this file for
       another unit's defect.  Measured: ``--user`` verification of this unit
       emits eight such lines for orchestrator-*.service and fused-memory.service
       (which genuinely lack RestartSteps=) and zero for the dashboard unit.
       Both prefix forms above are covered — bare unit name for a semantic
       warning, full path plus line number for a parse warning.

    Scoping is what makes ``--user`` safe to pass, and passing it is the point:
    this is a --user unit (WantedBy=default.target, installed under
    ~/.config/systemd/user), so system-scope verification checks it in a manager
    scope it never runs in and silently fails to resolve its user-scope
    dependencies.
    """
    result = subprocess.run(
        ["systemd-analyze", "verify", "--user", str(HARDCODED)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    combined = f"{result.stdout}{result.stderr}"
    ignored = _ignored_directive_lines(combined, HARDCODED)
    assert not ignored, (
        f"systemd-analyze verify reports directives it is IGNORING in {HARDCODED}:\n"
        + "\n".join(f"  {line}" for line in ignored)
        + "\nEach such line is a directive systemd parsed, warned about, and then "
        "discarded — the unit does not do what its text says it does. An "
        "'Unknown key name' line means this host's systemd predates the "
        "directive (RestartSteps= and RestartMaxDelaySec= both require v254), "
        "so the backoff is inert here even though the file reads correctly."
    )


# ---------------------------------------------------------------------------
# Clean-stop exit-code classification
#
# `uv run` propagates its child's signal death as a numeric exit code, so a
# perfectly graceful SIGTERM stop surfaces to systemd as exit 143 (128+15).
# Under systemd's default classification that is a FAILURE: the journal records
# "Failed with result exit-code" (status=143) and the unit is left in a failed
# state for a stop that was clean, expected and successful — observed
# 2026-07-30 22:23:53 BST on the bounded-drain unit from task 3306, with six
# keep-alive clients attached.  The cost is diagnostic, and compounding: every
# routine restart plants a false failure, so the failed state stops meaning
# anything and a real crash reads exactly like a deploy.
#
# Note there is no systemd-analyze corroboration in this section, unlike the
# restart-backoff one above.  A missing SuccessExitStatus= is a perfectly VALID
# configuration, not a misconfiguration, so systemd emits nothing about it — the
# file-content invariant is the only signal available.
# ---------------------------------------------------------------------------


def _success_exit_statuses(path: pathlib.Path) -> set[str]:
    """Return every exit status *path* declares as success, as a flat token set.

    Two systemd behaviours are load-bearing here and neither is optional:

    1. A single directive accepts a SPACE-SEPARATED LIST
       (``SuccessExitStatus=143 SIGPIPE``), so each value is split.
    2. Repeated directives ACCUMULATE rather than override, so every match is
       collected — ``re.findall``, not ``re.search``.

    A ``re.search(...).group(1)`` parser would satisfy neither: it would read a
    unit that correctly declares ``SuccessExitStatus=SIGPIPE`` on one line and
    ``SuccessExitStatus=143`` on another as missing 143, sending a future author
    to "fix" a unit that is already right.  Same failure mode _timeout_stop_sec
    guards against, where a valid ``15s`` must not be misdiagnosed as absent.

    The ``^`` anchor also keeps this reading DIRECTIVES, not prose: both real
    unit files discuss this directive by name in the comment above it, and a
    commented-out mention must never be counted as a live declaration.
    """
    values = re.findall(
        r"^SuccessExitStatus=(.*)$",
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return {token for value in values for token in value.split()}


def _assert_clean_sigterm_is_success(path: pathlib.Path) -> None:
    """Assert *path* classifies a clean SIGTERM stop (exit 143) as success."""
    statuses = _success_exit_statuses(path)
    assert statuses, (
        f"{path} declares no SuccessExitStatus= at all, so systemd applies its "
        "default classification and records every clean stop as a failure. "
        "`uv run` propagates its child's SIGTERM death as exit code 143 "
        "(128+15), which the journal reports as 'Failed with result exit-code' "
        "(status=143) — observed 2026-07-30 22:23:53 BST on the bounded-drain "
        "unit from task 3306. Declare SuccessExitStatus=143."
    )
    assert "143" in statuses, (
        f"{path} declares SuccessExitStatus={sorted(statuses)}, which does not "
        "include 143 — the status this unit actually exits with on a graceful "
        "stop. `uv run` translates its child's SIGTERM death into exit 143 "
        "(128+15), so a stop that was clean and expected is still recorded as "
        "'Failed with result exit-code'. Declaring some OTHER status successful "
        "does not fix the exit this unit really produces."
    )

    # Context for the trade being made, not an independent property: 143 being
    # "success" is precisely what stops Restart=on-failure re-firing on it, so a
    # reader of this assertion should be able to see the restart policy it
    # interacts with rather than having to go find it.
    restart = _restart_directive(path, "Restart")
    assert restart is not None, (
        f"{path} declares SuccessExitStatus but no Restart= policy. "
        "SuccessExitStatus= reclassifies exactly the exits Restart=on-failure "
        "reacts to, so the two belong together; without a restart policy the "
        "directive only relabels the recorded result and this assertion is "
        "pinning an interaction that is not there."
    )


def test_success_exit_status_parser_handles_systemd_spellings(
    tmp_path: pathlib.Path,
) -> None:
    """_success_exit_statuses must handle every spelling systemd itself accepts.

    Per the module convention, the helper earns its own negatives so it cannot
    silently no-op, and each case pins its own failure mode with ``match=``
    rather than a bare ``pytest.raises(AssertionError)`` that any assertion in
    the chain would satisfy.

    The good cases matter as much as the bad ones here: a parser too strict to
    read a valid unit would send a future author to "fix" something already
    correct, which is the more expensive error of the two.
    """
    # Bad: no directive at all — the pre-fix state.
    absent = _write_restart_unit(
        tmp_path / "absent.service", "Restart=on-failure\nRestartSec=5"
    )
    assert _success_exit_statuses(absent) == set()
    with pytest.raises(AssertionError, match="declares no SuccessExitStatus"):
        _assert_clean_sigterm_is_success(absent)

    # Bad: named only in a comment.  Not hypothetical — the real unit files
    # carry an explanatory comment naming this directive directly above it, so a
    # parser that dropped the ^ anchor would report those files as compliant
    # while the live directive had been deleted.  Same trap
    # test_uvicorn_flag_lookup_is_scoped_to_exec_start guards for the flags.
    commented = _write_restart_unit(
        tmp_path / "commented.service",
        "# SuccessExitStatus=143 because `uv run` exits 143 on SIGTERM\n"
        "Restart=on-failure\nRestartSec=5",
    )
    assert _success_exit_statuses(commented) == set(), (
        "a commented-out mention was counted as a live declaration"
    )
    with pytest.raises(AssertionError, match="declares no SuccessExitStatus"):
        _assert_clean_sigterm_is_success(commented)

    # Bad: present, but not the status this unit actually exits with.  Guards
    # against a helper that merely checks the directive exists.
    wrong = _write_restart_unit(
        tmp_path / "wrong.service", "Restart=on-failure\nSuccessExitStatus=75"
    )
    with pytest.raises(AssertionError, match="does not include 143"):
        _assert_clean_sigterm_is_success(wrong)

    # Bad: the right status but no restart policy to interact with.
    no_restart = _write_restart_unit(
        tmp_path / "no_restart.service", "SuccessExitStatus=143"
    )
    with pytest.raises(AssertionError, match="no Restart= policy"):
        _assert_clean_sigterm_is_success(no_restart)

    # Good: the plain form — must not raise.
    plain = _write_restart_unit(
        tmp_path / "plain.service", "Restart=on-failure\nSuccessExitStatus=143"
    )
    _assert_clean_sigterm_is_success(plain)

    # Good: space-separated list on ONE line.  Pins the whitespace split — a
    # whole-value parser would compare the string "143 SIGPIPE" against "143".
    listed = _write_restart_unit(
        tmp_path / "listed.service", "Restart=on-failure\nSuccessExitStatus=143 SIGPIPE"
    )
    assert _success_exit_statuses(listed) == {"143", "SIGPIPE"}
    _assert_clean_sigterm_is_success(listed)

    # Good: split across REPEATED directives, which systemd accumulates.  Pins
    # the findall — a single re.search would read only the SIGPIPE line and
    # declare this correctly-configured unit broken.
    repeated = _write_restart_unit(
        tmp_path / "repeated.service",
        "Restart=on-failure\nSuccessExitStatus=SIGPIPE\nSuccessExitStatus=143",
    )
    assert _success_exit_statuses(repeated) == {"143", "SIGPIPE"}
    _assert_clean_sigterm_is_success(repeated)


def test_clean_sigterm_stop_is_not_recorded_as_a_unit_failure() -> None:
    """Both unit files must classify exit 143 as a successful stop.

    Without it every ordinary `systemctl restart` leaves a "Failed with result
    exit-code" entry in the journal for a shutdown that worked exactly as
    designed, so a genuinely failed unit is indistinguishable from a routine one.
    """
    for path in (TEMPLATE, HARDCODED):
        _assert_clean_sigterm_is_success(path)
