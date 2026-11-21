"""The dashboard unit's directive shape, and render parity between its two committed copies.

These tests read the source-controlled service definition files directly —
no systemd runtime is required.  They pin the environment and
WorkingDirectory= lines of the template (scripts/dashboard.service.template,
the true source of truth used by setup-host.sh) and of the checked-in
hardcoded copy (dashboard/dark-factory-dashboard.service), and guard against
drift between the two.

Sibling suites:
  - tests/scripts/test_dashboard_unit_uvicorn_timeouts.py — uvicorn drain and keep-alive bounds
  - tests/scripts/test_dashboard_unit_restart_policy.py — restart backoff and clean-stop classification
  - tests/scripts/test_dashboard_client_pollers.py — the browser pollers the keep-alive bound must clear

See also:
  - tests/scripts/test_run_vllm_eval_lint.py  — pattern reference
  - dashboard/src/dashboard/config.py — DashboardConfig.from_env handling of DASHBOARD_KNOWN_PROJECT_ROOTS (COMMA-separated split)
"""

import pathlib
import re

import pytest

# The PRODUCTION substitution, imported by name — tests/scripts/conftest.py puts
# scripts/ on sys.path for exactly this. See the comment above
# HARDCODED_SERVICE_REPO_ROOT for why this module renders through the
# installer's own code rather than replaying the substitution.
import render_systemd_unit
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

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
# absolute" error, corroborated in
# tests/scripts/test_dashboard_unit_restart_policy.py::
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
# Substitution semantics: scripts/render_systemd_unit.py::render_template,
# which is the code setup-host.sh section 8 actually runs to install this unit.
# Both sentinels are replaced globally, unanchored and literally.
#
# The render test calls render_systemd_unit.render_template rather than
# replaying the substitution here, so its lockstep claim is made THROUGH the
# installer's own function and cannot drift from the installer.
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
    so the installer's __REPO_ROOT__ substitution
    (scripts/render_systemd_unit.py::render_template) makes the installed
    value track the real checkout — the same treatment
    DASHBOARD_KNOWN_PROJECT_ROOTS' self entry already gets. NOTE the treatment
    is only the same for the SUBSTITUTION: DASHBOARD_KNOWN_PROJECT_ROOTS is
    additionally PRESERVED from the installed unit across a reinstall
    (systemd_unit_parity.ENV_POLICY's host-local rows, preserved by
    scripts/render_systemd_unit.py), and this variable
    deliberately is not — its value must equal the SAME copy's
    WorkingDirectory=, which the next test pins.
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

    This is the canonical drift-prevention invariant, and it is made THROUGH the
    installer's own renderer: scripts/render_systemd_unit.py::render_template
    is the function setup-host.sh section 8 runs to install this unit, so the
    two repo-side files are held in lockstep by the same code that renders the
    host-side one. A replayed copy of the substitution would be a third
    spelling, unguarded.

    If this test fails, the template and hardcoded file have drifted.  Re-render
    with::

        python3 scripts/render_systemd_unit.py --no-preserve \\
            --template  scripts/dashboard.service.template \\
            --repo-root /home/leo/src/dark-factory \\
            --uv-path   /home/leo/.local/bin/uv \\
            --output    dashboard/dark-factory-dashboard.service

    (The two literals are HARDCODED_SERVICE_REPO_ROOT / HARDCODED_SERVICE_UV_PATH
    above — the committed copy is deliberately pinned to this host's paths, not
    to whatever checkout happens to be running the test.)

    ``--no-preserve`` IS LOAD-BEARING HERE, and its absence was a trap.  That
    CLI's whole purpose is host-side: it reads --output FIRST as the installed
    copy and carries its host-local DASHBOARD_KNOWN_PROJECT_ROOTS into the
    render.  Pointed at the REPO-side artifact, the file being read is the one
    under regeneration — so the moment the template's default for that variable
    changes, the preserving form would copy the stale committed value straight
    back in, leave this test red, and give no hint why.  It is a no-op today
    only because the two values coincide.  Never pass --no-preserve when
    rendering an INSTALLED unit; that is the clobber the renderer exists to
    prevent, and setup-host.sh does not pass it.
    """
    rendered = render_systemd_unit.render_template(
        TEMPLATE.read_text(encoding="utf-8"),
        repo_root=HARDCODED_SERVICE_REPO_ROOT,
        uv_path=HARDCODED_SERVICE_UV_PATH,
    )
    hardcoded = HARDCODED.read_text(encoding="utf-8")
    assert rendered == hardcoded, (
        f"Rendered template does not match {HARDCODED}.\n"
        f"Template path: {TEMPLATE}\n"
        "The files have drifted.  Re-render with "
        "scripts/render_systemd_unit.py — see this test's docstring for the "
        "exact invocation."
    )
