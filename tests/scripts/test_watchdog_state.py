"""Unit tests for scripts/_watchdog_state.py, the watchdog's persisted-state primitives.

Every function takes the caller's journal sink, so these tests pass a list's
``append`` and read what landed there; nothing here loads the watchdog or
patches a module-level logger. Failure injection stays at the stdlib boundary
(``os.replace``, ``time.time``, a real chmod).
"""

import json
import os
import pathlib
import time

import _watchdog_state as state
import pytest

# Resolves because the root conftest.py appends the repo root to sys.path.
from df_pytest_isolation import (
    CLOCK_PROVENANCE_SESSION_KEY,
    CLOCK_PROVENANCE_SOURCE_KEY,
    PYTEST_SESSION_TOKEN_ENV,
)


def _discard(_msg: str) -> None:
    pass


def _cross_device_replace(_src: str, _dst: str) -> None:
    raise OSError("cross-device link")


# ---------------------------------------------------------------------------
# Atomic JSON state files: atomic_write_json / read_json_state (task 3764)
#
# Task 3764 added THREE more persisted-state writers/readers to a watchdog that
# already had two (_stamp_fm_deploy_clock / _read_last_fm_deploy_epoch).
# Without extraction the ~20-line makedirs/mkstemp/os.replace/unlink-on-error
# dance would have been copy-pasted a third and fourth time. The extraction is
# behaviour-preserving; the deploy-clock suite in
# tests/scripts/test_orchestrator_watchdog.py is its regression net,
# re-asserted explicitly by
# tests/scripts/test_orchestrator_watchdog.py::test_fm_deploy_clock_still_roundtrips_through_shared_helpers.
# ---------------------------------------------------------------------------


def test_atomic_write_json_writes_body_and_creates_parent_dir(
    tmp_path: pathlib.Path,
) -> None:
    """atomic_write_json writes the JSON body, creating a missing parent dir.

    Asserted against the REAL on-disk file (parsed back), never a mock.
    """
    target = tmp_path / "deep" / "nested" / "state.json"
    assert not target.parent.exists()

    state.atomic_write_json(str(target), {"count": 2, "verdict": "wedged"}, _discard)

    assert target.exists(), "the write must create the target file and its parent dir"
    assert json.loads(target.read_text()) == {"count": 2, "verdict": "wedged"}


def test_atomic_write_json_leaves_no_stray_temp_file(tmp_path: pathlib.Path) -> None:
    """A successful atomic_write_json leaves ONLY the target file behind.

    The write goes via tempfile.mkstemp + os.replace; the temp file must be
    renamed away, not left accumulating in data/fused-memory/ on every tick.
    """
    target = tmp_path / "state.json"

    state.atomic_write_json(str(target), {"ts": 1}, _discard)

    survivors = sorted(p.name for p in tmp_path.iterdir())
    assert survivors == ["state.json"], f"stray temp file(s) left behind: {survivors}"


def test_atomic_write_json_fail_soft_on_replace_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A failed rename is logged and swallowed, and the temp file is unlinked.

    Fail-soft: a persistence hiccup must never crash the oneshot watchdog.
    """
    logged: list[str] = []
    monkeypatch.setattr(os, "replace", _cross_device_replace)
    target = tmp_path / "state.json"

    # Must not raise.
    state.atomic_write_json(str(target), {"ts": 1}, logged.append)

    assert not target.exists(), "a failed write must not leave a partial target"
    assert list(tmp_path.iterdir()) == [], (
        f"the temp file must be unlinked on failure: "
        f"{sorted(p.name for p in tmp_path.iterdir())}"
    )
    assert logged, "a swallowed write failure must be logged, not silent"
    assert logged[0].startswith("atomic_write_json:"), (
        f"the swallowed-failure line must name the primitive: {logged!r}"
    )


def test_atomic_write_json_fail_soft_on_unwritable_dir(tmp_path: pathlib.Path) -> None:
    """An unwritable target directory is logged and swallowed, not raised."""
    logged: list[str] = []

    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)  # r-x: cannot create the temp file inside
    try:
        # Must not raise.
        state.atomic_write_json(str(locked / "state.json"), {"ts": 1}, logged.append)
    finally:
        locked.chmod(0o700)

    assert not (locked / "state.json").exists()
    assert logged, "a swallowed write failure must be logged, not silent"


def test_atomic_write_json_reports_whether_the_write_landed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The writer returns True only after the rename succeeded, False otherwise.

    Fail-soft is not the same as fail-safe: one caller
    (scripts/orchestrator-watchdog.py::_stamp_fm_liveness_restart_clock) ARMS a
    cap, so it must be able to tell a landed write from a swallowed failure
    rather than assume the former.
    """
    assert state.atomic_write_json(str(tmp_path / "ok.json"), {"ts": 1}, _discard) is True

    monkeypatch.setattr(os, "replace", _cross_device_replace)
    assert (
        state.atomic_write_json(str(tmp_path / "bad.json"), {"ts": 1}, _discard) is False
    )


def test_read_json_state_returns_parsed_dict(tmp_path: pathlib.Path) -> None:
    """read_json_state parses a well-formed file into a dict."""
    target = tmp_path / "state.json"
    target.write_text('{"ts": 1783000000, "iso": "2026-08-06T00:00:00+00:00"}')

    assert state.read_json_state(str(target), _discard) == {
        "ts": 1783000000,
        "iso": "2026-08-06T00:00:00+00:00",
    }


def test_read_json_state_missing_file_returns_none_silently(
    tmp_path: pathlib.Path,
) -> None:
    """A missing file yields None with NO log line.

    "The state does not exist yet" is the normal case on a fresh checkout — it
    is not a degradation and must not spam the journal every tick.
    """
    logged: list[str] = []

    assert state.read_json_state(str(tmp_path / "absent.json"), logged.append) is None
    assert logged == [], f"a missing state file must not log: {logged}"


def test_read_json_state_corrupt_json_returns_none_and_logs(
    tmp_path: pathlib.Path,
) -> None:
    """A truncated/corrupt body yields None (fail-open) AND a log line."""
    logged: list[str] = []
    target = tmp_path / "corrupt.json"
    target.write_text('{"ts": 178300')  # truncated / partially written

    assert state.read_json_state(str(target), logged.append) is None
    assert logged, "a corrupt state file is a real degradation and must be logged"


def test_read_json_state_non_object_body_returns_none_and_logs(
    tmp_path: pathlib.Path,
) -> None:
    """A well-formed but NON-OBJECT body yields None AND a log line.

    ``[1, 2]`` / ``"hello"`` parse fine but carry no state, so every caller
    loses its clock or streak — indistinguishable in consequence from a
    corrupt file, and therefore a genuine degradation that must leave journal
    evidence. Before the task-3764 extraction this case raised TypeError on
    ``raw["ts"]`` inside _read_last_fm_deploy_epoch and WAS logged; a silent
    None here would make the fm deploy backstop and the liveness restart cap
    both vanish with nothing in the journal to explain it.
    """
    for body in ("[1, 2]", '"hello"', "null", "3"):
        logged: list[str] = []
        target = tmp_path / "nonobject.json"
        target.write_text(body)

        assert state.read_json_state(str(target), logged.append) is None, (
            f"a non-object body must yield None: {body!r}"
        )
        assert logged, f"a non-object state body must be logged: {body!r}"
        assert any(str(target) in m for m in logged), (
            f"the log line must name the offending path: {logged!r}"
        )


def test_read_json_state_oserror_returns_none_and_logs(tmp_path: pathlib.Path) -> None:
    """A non-FileNotFoundError OSError yields None (fail-open) AND a log line."""
    logged: list[str] = []
    target = tmp_path / "state.json"
    target.write_text('{"ts": 1}')
    target.chmod(0o000)
    try:
        result = state.read_json_state(str(target), logged.append)
    finally:
        target.chmod(0o600)

    assert result is None
    assert logged, "an unreadable state file must be logged, not silently ignored"


# ---------------------------------------------------------------------------
# Clock-layer primitives: read_clock_epoch / stamp_clock / within_min_interval
# (task 3764 amendment)
#
# read_clock_epoch / stamp_clock / within_min_interval sit one level above
# the JSON primitives and are the SINGLE definition behind every clock
# wrapper in scripts/orchestrator-watchdog.py (fleet-deploy, fm-deploy,
# fm-liveness-restart) and the unit-parity clock. The wrappers' own suites are
# the behavioural regression net; these pin the primitives' contracts directly
# so a drift between the fail-open contracts is caught at the source rather
# than once per clock.
# ---------------------------------------------------------------------------


def test_read_clock_epoch_roundtrips_a_stamped_clock(tmp_path: pathlib.Path) -> None:
    """stamp_clock writes {ts, iso}; read_clock_epoch reads it back as float."""
    clock = tmp_path / "nested" / "clock.json"
    assert not clock.parent.exists(), "the stamp must create the parent dir"

    state.stamp_clock(str(clock), _discard)

    body = json.loads(clock.read_text())
    assert "ts" in body and "iso" in body, f"clock body must carry ts+iso: {body}"
    epoch = state.read_clock_epoch(str(clock), "test clock", _discard)
    assert isinstance(epoch, float)
    assert epoch == pytest.approx(float(body["ts"]))


def test_read_clock_epoch_missing_file_returns_none_silently(
    tmp_path: pathlib.Path,
) -> None:
    """An absent clock is the normal 'never stamped' case: None, and no log spam."""
    logged: list[str] = []

    assert (
        state.read_clock_epoch(str(tmp_path / "absent.json"), "test clock", logged.append)
        is None
    )
    assert logged == [], f"a never-stamped clock must not log every tick: {logged!r}"


def test_read_clock_epoch_unusable_ts_returns_none_and_names_the_clock(
    tmp_path: pathlib.Path,
) -> None:
    """A body with a missing/unusable `ts` fails open to None AND names the label.

    The label is the whole reason it is a parameter: with several clocks
    sharing one reader, a journal line that did not say WHICH clock degraded
    would be useless to an operator.
    """
    for body in ('{"iso": "2026-08-06T00:00:00+00:00"}', '{"ts": "not-a-number"}'):
        logged: list[str] = []
        clock = tmp_path / "clock.json"
        clock.write_text(body)

        assert state.read_clock_epoch(str(clock), "fm-deploy clock", logged.append) is None
        assert any("fm-deploy clock" in m for m in logged), (
            f"the log line must name the degraded clock: {logged!r}"
        )


def test_within_min_interval_uses_the_reader_it_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate compares now - reader() against secs, honouring the passed reader.

    Taking the READER rather than a path is what keeps each wrapper's named
    epoch function the single seam tests substitute at — a wrapper can never
    silently bypass its own reader.
    """
    monkeypatch.setattr(time, "time", lambda: 1_000_000.0 + 100.0)

    assert state.within_min_interval(3600, lambda: 1_000_000.0) is True
    assert state.within_min_interval(60, lambda: 1_000_000.0) is False


def test_within_min_interval_disabled_does_not_read_the_clock() -> None:
    """secs<=0 disables the cap outright — the reader is not even called.

    A disabled cap must not depend on a readable file.
    """

    def boom() -> float:
        pytest.fail("a disabled min-interval must not read its clock")

    assert state.within_min_interval(0, boom) is False
    assert state.within_min_interval(-1, boom) is False


def test_within_min_interval_absent_clock_is_outside_the_window() -> None:
    """A None epoch counts as OUTSIDE the window (fail toward acting).

    Failing the other way would let one unreadable file suppress every future
    restart indefinitely.
    """
    assert state.within_min_interval(3600, lambda: None) is False


# ---------------------------------------------------------------------------
# min_interval_remaining_secs: the window arithmetic's one home (task 8270)
#
# within_min_interval is re-expressed on it, so a cap that must say HOW LONG
# it still holds (the per-unit liveness revive cap in
# scripts/orchestrator-watchdog.py) reads the clock once and shares the
# fail directions above rather than carrying a second copy of them.
# ---------------------------------------------------------------------------

_NOW = 1_000_000.0


def _never_read() -> float:
    pytest.fail("a disabled min-interval must not read its clock")


@pytest.fixture
def _frozen_now(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: _NOW)


@pytest.mark.usefixtures("_frozen_now")
def test_min_interval_remaining_secs_counts_down_from_the_stamp() -> None:
    assert state.min_interval_remaining_secs(900, lambda: _NOW - 100) == pytest.approx(
        800.0
    )


@pytest.mark.usefixtures("_frozen_now")
def test_min_interval_remaining_secs_is_zero_once_the_window_has_passed() -> None:
    assert state.min_interval_remaining_secs(900, lambda: _NOW - 901) == 0.0


def test_min_interval_remaining_secs_disabled_does_not_read_the_clock() -> None:
    assert state.min_interval_remaining_secs(0, _never_read) == 0.0
    assert state.min_interval_remaining_secs(-5, _never_read) == 0.0


def test_min_interval_remaining_secs_absent_clock_is_outside_the_window() -> None:
    assert state.min_interval_remaining_secs(900, lambda: None) == 0.0


@pytest.mark.usefixtures("_frozen_now")
def test_min_interval_remaining_secs_future_dated_clock_holds_longer_than_secs() -> None:
    """A clock stamped in the future (a skewed host, a test that patched time)
    holds the cap for more than *secs*, exactly as within_min_interval always did.
    """
    assert state.min_interval_remaining_secs(900, lambda: _NOW + 100) > 900


@pytest.mark.usefixtures("_frozen_now")
@pytest.mark.parametrize(
    ("secs", "epoch"),
    [
        (900, _NOW - 100),
        (900, _NOW - 901),
        (900, _NOW - 900),
        (0, _NOW - 100),
        (-5, _NOW - 100),
        (900, None),
        (900, _NOW + 100),
    ],
)
def test_within_min_interval_is_remaining_secs_above_zero(
    secs: int, epoch: float | None
) -> None:
    assert state.within_min_interval(secs, lambda: epoch) == (
        state.min_interval_remaining_secs(secs, lambda: epoch) > 0
    )


def test_an_injected_now_replaces_the_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller that owns its *now* (the crash-loop page dedup) shares this rule too."""
    monkeypatch.setattr(time, "time", lambda: pytest.fail("an injected now reads no wall clock"))

    assert state.min_interval_remaining_secs(
        900, lambda: _NOW - 100, now=_NOW
    ) == pytest.approx(800.0)
    assert state.within_min_interval(900, lambda: _NOW - 899, now=_NOW) is True
    assert state.within_min_interval(900, lambda: _NOW - 900, now=_NOW) is False


# ---------------------------------------------------------------------------
# task 4823: the clock stamp carries its own provenance
#
# The watchdog is the SECOND of the two clock writers (restart-all-orchestrators.sh
# is the first) and it owns every one of its clocks through one stamp_clock
# primitive. These pin this writer's half of the contract;
# df_pytest_isolation.py::deploy_clock_change_report states what the provenance
# is for and what it buys.
#
# Every key is named through the df_pytest_isolation constants rather than a
# literal of this file's own: that is what makes these drift pins rather than
# tautologies, since the guard reads the keys IT defines and neither tier can
# import the other (the watchdog is stdlib-only, and df_pytest_isolation is
# stdlib+pytest-only because every subproject conftest imports it).
# ---------------------------------------------------------------------------

# A stand-in for a real uuid4().hex; the shape a live session actually stamps.
_SESSION_TOKEN_SENTINEL = "3f9c1a8b2d7e4f60a5b3c1d9e7f50246"


def test_stamp_clock_records_its_writer_and_the_ambient_session_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A test-spawned stamp is TAGGED as test-spawned.

    The token is read from the ambient environment at write time, which is what
    lets a suite-wide guard attribute a write it did not make itself: every
    spawner in this repo builds its child env from dict(os.environ), so
    descendants inherit it for free.
    """
    monkeypatch.setenv(PYTEST_SESSION_TOKEN_ENV, _SESSION_TOKEN_SENTINEL)
    clock_file = tmp_path / "clock.json"

    assert state.stamp_clock(str(clock_file), _discard) is True

    body = json.loads(clock_file.read_text())
    assert body[CLOCK_PROVENANCE_SOURCE_KEY] == "orchestrator-watchdog.py"
    assert body[CLOCK_PROVENANCE_SESSION_KEY] == _SESSION_TOKEN_SENTINEL


def test_stamp_clock_records_an_empty_token_outside_a_pytest_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The shape every GENUINE deploy writes — and the guard's one forgiving input.

    ALWAYS present, never omitted: an empty string is the positive statement "no
    pytest session was an ancestor of this write", whereas a missing key is
    indistinguishable from a pre-4823 writer and correctly keeps failing closed.
    """
    monkeypatch.delenv(PYTEST_SESSION_TOKEN_ENV, raising=False)
    clock_file = tmp_path / "clock.json"

    state.stamp_clock(str(clock_file), _discard)

    body = json.loads(clock_file.read_text())
    assert CLOCK_PROVENANCE_SESSION_KEY in body, (
        f"the key must be PRESENT and empty, not omitted; got {body!r}"
    )
    assert body[CLOCK_PROVENANCE_SESSION_KEY] == ""
    assert body[CLOCK_PROVENANCE_SOURCE_KEY] == "orchestrator-watchdog.py"


def test_stamp_clock_keeps_ts_and_iso_readable(tmp_path: pathlib.Path) -> None:
    """The additive-schema guarantee the clock readers depend on.

    Every reader extracts `ts` and nothing else, so the two new keys must be
    inert to all of them. Asserted through the real reader rather than by
    eyeballing the body: a schema change that broke read_clock_epoch would
    disarm every min-interval cap at once.
    """
    clock_file = tmp_path / "clock.json"

    state.stamp_clock(str(clock_file), _discard)

    epoch = state.read_clock_epoch(str(clock_file), "test clock", _discard)
    assert epoch is not None
    assert epoch == pytest.approx(time.time(), abs=60)
    assert isinstance(json.loads(clock_file.read_text())["iso"], str)
