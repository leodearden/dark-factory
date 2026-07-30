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
    "__REPO_ROOT__"
)

HARDCODED_EXPECTED_ENV_LINE = (
    "Environment=DASHBOARD_KNOWN_PROJECT_ROOTS="
    "/home/leo/src/dark-factory"
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

# The browser's measured poll interval: plans/dashboard-availability-prd.md
# records that dashboard/static/redux/data.js:165 polls every 3 seconds, and
# that the resulting keep-alive connections "never idle out under a 3s poll" —
# which is exactly why the drain never completed before the graceful-shutdown
# bound was added.
CLIENT_POLL_INTERVAL_SECONDS = 3


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
    """Return the integer argument of ``--<flag>`` in *path*'s ExecStart, or None."""
    command = _logical_exec_start(path)
    match = re.search(rf"--{re.escape(flag)}\s+(\d+)", command)
    return int(match.group(1)) if match else None


def _timeout_stop_sec(path: pathlib.Path) -> int:
    """Return the unit's TimeoutStopSec= value in seconds.

    The directive must be present: a unit without it inherits systemd's
    DefaultTimeoutStopSec, which makes any margin assertion against it
    meaningless.
    """
    match = re.search(
        r"^TimeoutStopSec=(\d+)$",
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match is not None, (
        f"No TimeoutStopSec= directive found in {path}. "
        "Without it the unit inherits systemd's DefaultTimeoutStopSec, so the "
        "graceful-shutdown margin cannot be verified from the unit file alone."
    )
    return int(match.group(1))


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
    timeout_stop_sec: int = 15,
) -> pathlib.Path:
    """Write a minimal synthetic unit file for guard-verification tests."""
    path.write_text(
        f"[Service]\n{exec_start}\nTimeoutStopSec={timeout_stop_sec}\n",
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
    """
    # Bad: no --timeout-graceful-shutdown at all — the pre-fix state.
    absent = _write_synthetic_unit(
        tmp_path / "absent.service",
        "ExecStart=/usr/bin/uv run python -m uvicorn app:app --host 127.0.0.1",
    )
    with pytest.raises(AssertionError):
        _assert_drain_bounded(absent)

    # Bad: graceful timeout ABOVE the stop timeout — still SIGKILLs.
    above = _write_synthetic_unit(
        tmp_path / "above.service",
        "ExecStart=/usr/bin/uv run python -m uvicorn app:app "
        "--timeout-graceful-shutdown 20",
    )
    with pytest.raises(AssertionError):
        _assert_drain_bounded(above)

    # Bad: graceful timeout EQUAL to the stop timeout — races the SIGKILL.
    equal = _write_synthetic_unit(
        tmp_path / "equal.service",
        "ExecStart=/usr/bin/uv run python -m uvicorn app:app "
        "--timeout-graceful-shutdown 15",
    )
    with pytest.raises(AssertionError):
        _assert_drain_bounded(equal)

    # Bad: below the stop timeout but with too little margin (3s < 5s) for
    # lifespan shutdown, interpreter exit and the `uv run` parent's teardown.
    thin_margin = _write_synthetic_unit(
        tmp_path / "thin_margin.service",
        "ExecStart=/usr/bin/uv run python -m uvicorn app:app "
        "--timeout-graceful-shutdown 12",
    )
    with pytest.raises(AssertionError):
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
    dropping keep-alive below the client's 3s poll so idle connections close on
    their own — is wrong here: it would make the server close the polling socket
    in the gap between polls, exposing the classic
    server-closes-while-client-writes race, i.e. trading a shutdown-time stall
    for request-time failures on a change whose whole purpose is availability.
    The hard drain guarantee already comes from --timeout-graceful-shutdown, so
    keep-alive is not being asked to do that job.

    Pinning it explicitly (at uvicorn's own default) is still worth doing: it
    surfaces the interaction that caused the incident — the 5s default exceeds
    the 3s poll, which is precisely why the connection never idles out — makes
    it greppable, and gives the unit-file parity check a concrete directive to
    diff.
    """
    for path in (TEMPLATE, HARDCODED):
        keep_alive = _uvicorn_int_flag(path, "timeout-keep-alive")
        assert keep_alive is not None, (
            f"No --timeout-keep-alive flag in the ExecStart of {path}. "
            "Leaving it implicit hides the interaction that caused the incident: "
            "uvicorn's 5s default exceeds the client's "
            f"{CLIENT_POLL_INTERVAL_SECONDS}s poll, which is why the polling "
            "connection never idles out."
        )
        assert keep_alive > CLIENT_POLL_INTERVAL_SECONDS, (
            f"--timeout-keep-alive {keep_alive} is not above the client's "
            f"{CLIENT_POLL_INTERVAL_SECONDS}s poll interval in {path}. "
            "Below the poll interval the server closes the polling socket in the "
            "gap between polls, exposing the server-closes-while-client-writes "
            "race and turning a shutdown fix into a source of failed polls. The "
            "drain is already bounded by --timeout-graceful-shutdown; do not "
            "retune keep-alive to compensate for it."
        )
        stop = _timeout_stop_sec(path)
        assert keep_alive < stop, (
            f"--timeout-keep-alive {keep_alive} is not below TimeoutStopSec={stop} "
            f"in {path}. A keep-alive idle window longer than the stop timeout "
            "would be incoherent with the drain bound."
        )
