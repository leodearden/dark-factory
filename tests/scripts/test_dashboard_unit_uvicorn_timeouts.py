"""The dashboard unit's uvicorn timeouts are bounded by its stop deadline and the browser's polls.

In both committed copies (scripts/dashboard.service.template and
dashboard/dark-factory-dashboard.service), --timeout-graceful-shutdown must sit
below TimeoutStopSec= with margin to spare, and --timeout-keep-alive must sit
above the slowest connection-reuse interval of the browser's recurring pollers.
These tests read the files directly, so no systemd runtime is required.

Convention: each file-content assertion helper earns its own negative tests,
so it cannot silently no-op, and every bad case pins its own failure mode with
``match=``.
"""

import pathlib
import re

import pytest
from dashboard_client_pollers import slowest_client_poll
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from systemd_unit_invariants import logical_exec_start

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = pathlib.Path(__file__).parents[2]
TEMPLATE = REPO_ROOT / "scripts" / "dashboard.service.template"
HARDCODED = REPO_ROOT / "dashboard" / "dark-factory-dashboard.service"


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
    value after the flag had actually been deleted from the command.  None
    means the command lacks the flag, never that there is no command: a unit
    with no effective ExecStart= raises MalformedExecStart instead.
    """
    command = logical_exec_start(path.read_text(encoding="utf-8"), str(path))
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

    Follows this module's convention: a file-content assertion helper earns
    its own negative tests, so it cannot silently no-op.  Without these, a
    regex that failed to join the
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
    # logical_exec_start (a per-line regex would miss this flag entirely).
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
        "the command. Scope it to logical_exec_start()."
    )
    # The flag that IS on the command is still found, so the scoping did not
    # over-tighten into finding nothing at all.
    assert _uvicorn_int_flag(commented, "timeout-graceful-shutdown") == 8


def test_uvicorn_flag_lookup_reads_the_effective_exec_start(tmp_path: pathlib.Path) -> None:
    """_uvicorn_int_flag must read the LAST ExecStart=, the one systemd runs.

    A drop-in under <unit>.d/ merges by appending an empty reset and then the
    real command, so a first-match read reports a value systemd discards (the
    reasoning lives on systemd_unit_invariants.logical_exec_start).  This pins
    the deliberate first-to-last change in this suite's ExecStart= lookup.
    """
    overridden = _write_synthetic_unit(
        tmp_path / "overridden.service",
        "ExecStart=/usr/bin/uv run python -m uvicorn app:app --timeout-graceful-shutdown 20\n"
        "ExecStart=\n"
        "ExecStart=/usr/bin/uv run python -m uvicorn app:app \\\n"
        "  --timeout-graceful-shutdown 8",
    )
    graceful = _uvicorn_int_flag(overridden, "timeout-graceful-shutdown")
    assert graceful == 8, (
        f"_uvicorn_int_flag read --timeout-graceful-shutdown {graceful} from an "
        "overridden ExecStart= rather than the command systemd runs (8): the "
        "lookup must take the LAST ExecStart= assignment, joined across "
        "continuations."
    )


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
    socket (see tests/scripts/dashboard_client_pollers.py::CLIENT_POLLERS).
    Hence the comparison is against slowest_client_poll().

    That comparison is against the connection-REUSE interval, which is not the
    same as the timer period: data.js delays each request by up to
    JITTER_MAX_MS inside the in-flight window, so its requests can land
    POLL_INTERVAL_MS + JITTER_MAX_MS apart.  Comparing against the bare period
    would understate the gap the socket actually has to survive — see
    tests/scripts/dashboard_client_pollers.py::poller_reuse_ms.

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
    poll_ms, slowest = slowest_client_poll()
    for path in (TEMPLATE, HARDCODED):
        keep_alive = _uvicorn_int_flag(path, "timeout-keep-alive")
        assert keep_alive is not None, (
            f"No --timeout-keep-alive flag in the ExecStart of {path}. "
            "Leaving it implicit hides the interaction that caused the incident: "
            "keep-alive exceeds the slowest client connection-reuse interval — "
            f"{poll_ms / 1000:g}s, from {slowest.what} ({slowest.path}) — which "
            "is why the polling connection never idles out."
        )
        assert keep_alive * 1000 > poll_ms, (
            f"--timeout-keep-alive {keep_alive}s is not above the slowest client "
            f"connection-reuse interval of {poll_ms / 1000:g}s — {slowest.what}, "
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
