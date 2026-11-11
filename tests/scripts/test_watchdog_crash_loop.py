"""Tests for scripts/_watchdog_crash_loop.py, the orchestrator watchdog's crash-loop detector (task 8270).

The module is imported by name, so ``crash_loop`` is the sys.modules instance:
the same object scripts/orchestrator-watchdog.py's from-import binds. The
pure core (status, growth, observation, alarm text) needs no I/O at all.
"""

import dataclasses
import importlib.util
import pathlib
import types

import _watchdog_crash_loop as crash_loop
import pytest

# Resolves because the root conftest.py appends the repo root to sys.path.
from df_pytest_isolation import synthetic_unit

REPO_ROOT = pathlib.Path(__file__).parents[2]
CRASH_LOOP_PATH = REPO_ROOT / "scripts" / "_watchdog_crash_loop.py"

UNIT = synthetic_unit("dark-factory")
WINDOW = 1800
COUNT = 5
T = 1_800_000_000

AlarmKind = crash_loop.AlarmKind
UnitRestartHistory = crash_loop.UnitRestartHistory
UnitRestartStatus = crash_loop.UnitRestartStatus


def _status(n: int | None, result: str | None = "success") -> UnitRestartStatus:
    return UnitRestartStatus(n_restarts=n, result=result)


def _observe(
    history: UnitRestartHistory, status: UnitRestartStatus, now: int
) -> crash_loop.Observation:
    return crash_loop.observe_unit(
        UNIT, history, status, now=now, alarm_count=COUNT, window_secs=WINDOW
    )


# ---------------------------------------------------------------------------
# UnitRestartStatus
# ---------------------------------------------------------------------------


def test_unit_restart_status_is_frozen() -> None:
    status = _status(3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        status.n_restarts = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    ("result", "hit"),
    [
        ("start-limit-hit", True),
        ("success", False),
        ("exit-code", False),
        (None, False),
    ],
)
def test_start_limit_hit_only_for_that_result(result: str | None, hit: bool) -> None:
    assert _status(3, result).start_limit_hit is hit


def test_render_is_one_whitespace_free_token() -> None:
    assert _status(3).render() == "3/success"
    assert crash_loop.UNKNOWN_RESTART_STATUS.render() == "?/?"
    assert crash_loop.UNKNOWN_RESTART_STATUS == UnitRestartStatus(
        n_restarts=None, result=None
    )
    for status in (_status(10, "start-limit-hit"), crash_loop.UNKNOWN_RESTART_STATUS):
        assert len(status.render().split()) == 1


# ---------------------------------------------------------------------------
# restart_growth: counter-reset aware
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("samples", "growth"),
    [
        ((), 0),
        (((T, 4),), 0),
        (((T, 0), (T + 60, 2), (T + 120, 5)), 5),
        (((T, 3), (T + 60, 3), (T + 120, 3)), 0),
    ],
)
def test_restart_growth_sums_increases(
    samples: tuple[tuple[int, int], ...], growth: int
) -> None:
    assert crash_loop.restart_growth(samples) == growth


def test_restart_growth_counts_through_a_counter_flush() -> None:
    """A decrease means a manual start reset NRestarts: the new value is all growth."""
    assert crash_loop.restart_growth(((T, 4), (T + 60, 0), (T + 120, 3))) == 3


# ---------------------------------------------------------------------------
# observe_unit: the restart-rate alarm
# ---------------------------------------------------------------------------


def test_growth_at_the_threshold_alarms_and_stamps_the_page_epoch() -> None:
    history = UnitRestartHistory(samples=((T - 600, 0),), sampling_since=T - 600)

    obs = _observe(history, _status(5), T)

    assert obs.alarm is not None
    assert obs.alarm.kind is AlarmKind.RESTART_RATE
    assert obs.alarm.unit == UNIT
    assert obs.alarm.restarts == 5
    assert obs.alarm.window_secs == WINDOW
    assert obs.held is None
    assert obs.history.last_escalation_epoch == T


def test_growth_below_the_threshold_is_quiet() -> None:
    history = UnitRestartHistory(samples=((T - 600, 0),), sampling_since=T - 600)

    obs = _observe(history, _status(4), T)

    assert obs.alarm is None
    assert obs.held is None
    assert obs.history.last_escalation_epoch is None


def test_samples_older_than_the_window_are_pruned() -> None:
    history = UnitRestartHistory(samples=((T - 1900, 0),), sampling_since=T - 1900)

    obs = _observe(history, _status(5), T)

    assert obs.alarm is None
    assert obs.history.samples == ((T, 5),)


def test_one_page_per_window_per_unit() -> None:
    history = UnitRestartHistory(samples=((T - 600, 0),), sampling_since=T - 600)
    first = _observe(history, _status(5), T)

    held = _observe(first.history, _status(6), T + 60)
    assert held.alarm is None
    assert held.held is not None
    assert held.held.kind is AlarmKind.RESTART_RATE
    assert held.history.last_escalation_epoch == T

    again = _observe(held.history, _status(11), T + WINDOW)
    assert again.alarm is not None
    assert again.alarm.kind is AlarmKind.RESTART_RATE
    assert again.history.last_escalation_epoch == T + WINDOW


# ---------------------------------------------------------------------------
# observe_unit: start-limit-hit pages once per episode
# ---------------------------------------------------------------------------


def test_start_limit_hit_pages_once_per_episode() -> None:
    first = _observe(UnitRestartHistory(), _status(2, "start-limit-hit"), T)
    assert first.alarm is not None
    assert first.alarm.kind is AlarmKind.START_LIMIT_HIT
    assert first.history.start_limit_paged is True

    still_stuck = _observe(first.history, _status(2, "start-limit-hit"), T + WINDOW + 60)
    assert still_stuck.alarm is None
    assert still_stuck.history.start_limit_paged is True

    recovered = _observe(still_stuck.history, _status(0, "success"), T + WINDOW + 120)
    assert recovered.alarm is None
    assert recovered.history.start_limit_paged is False

    relapse = _observe(recovered.history, _status(1, "start-limit-hit"), T + 2 * WINDOW + 60)
    assert relapse.alarm is not None
    assert relapse.alarm.kind is AlarmKind.START_LIMIT_HIT


def test_start_limit_hit_inside_a_rate_pages_window_is_held_then_pages() -> None:
    history = UnitRestartHistory(samples=((T - 600, 0),), sampling_since=T - 600)
    rate = _observe(history, _status(5), T)
    assert rate.alarm is not None and rate.alarm.kind is AlarmKind.RESTART_RATE

    held = _observe(rate.history, _status(5, "start-limit-hit"), T + 60)
    assert held.alarm is None
    assert held.held is not None
    assert held.held.kind is AlarmKind.START_LIMIT_HIT
    assert held.history.start_limit_paged is False

    paged = _observe(held.history, _status(5, "start-limit-hit"), T + WINDOW)
    assert paged.alarm is not None
    assert paged.alarm.kind is AlarmKind.START_LIMIT_HIT
    assert paged.history.start_limit_paged is True


# ---------------------------------------------------------------------------
# observe_unit: unknown reads, sampling_since, immutability
# ---------------------------------------------------------------------------


def test_unknown_status_appends_nothing_and_keeps_the_episode_flag() -> None:
    history = UnitRestartHistory(
        samples=((T - 60, 2),), sampling_since=T - 60, start_limit_paged=True
    )

    obs = _observe(history, crash_loop.UNKNOWN_RESTART_STATUS, T)

    assert obs.alarm is None
    assert obs.held is None
    assert obs.history.samples == ((T - 60, 2),)
    assert obs.history.start_limit_paged is True


def test_sampling_since_is_set_once_and_survives_pruning() -> None:
    first = _observe(UnitRestartHistory(), _status(0), T)
    assert first.history.sampling_since == T

    later = _observe(first.history, _status(0), T + 7 * 86400)
    assert later.history.samples == ((T + 7 * 86400, 0),)
    assert later.history.sampling_since == T


def test_unknown_status_does_not_start_sampling() -> None:
    obs = _observe(UnitRestartHistory(), crash_loop.UNKNOWN_RESTART_STATUS, T)
    assert obs.history.sampling_since is None


def test_observe_unit_does_not_mutate_its_input() -> None:
    history = UnitRestartHistory(samples=((T - 600, 0),), sampling_since=T - 600)
    before = dataclasses.replace(history)

    _observe(history, _status(5, "start-limit-hit"), T)

    assert history == before
    with pytest.raises(dataclasses.FrozenInstanceError):
        history.start_limit_paged = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CrashLoopAlarm: the page's text
# ---------------------------------------------------------------------------


def test_alarm_sentinel_names_the_unit() -> None:
    alarm = crash_loop.CrashLoopAlarm(UNIT, AlarmKind.RESTART_RATE, 5, WINDOW)
    assert alarm.task_sentinel == f"orchestrator-crash-loop-{UNIT}"


def test_rate_alarm_text_carries_the_count_and_the_recipe() -> None:
    alarm = crash_loop.CrashLoopAlarm(UNIT, AlarmKind.RESTART_RATE, 7, WINDOW)

    summary = alarm.summary()
    assert UNIT in summary
    assert f"7 restarts in {WINDOW}s" in summary
    text = summary + "\n" + alarm.detail()
    assert f"journalctl --user -u {UNIT}" in text
    assert "exit_reason" in text


def test_start_limit_alarm_text_says_start_limit_hit() -> None:
    alarm = crash_loop.CrashLoopAlarm(UNIT, AlarmKind.START_LIMIT_HIT, 0, WINDOW)

    summary = alarm.summary()
    assert UNIT in summary
    assert "start-limit-hit" in summary
    text = summary + "\n" + alarm.detail()
    assert f"journalctl --user -u {UNIT}" in text
    assert "exit_reason" in text
    assert f"systemctl --user reset-failed {UNIT}" in text


# ---------------------------------------------------------------------------
# Env knobs, read at import time: load a FRESH copy (the by-name import is
# cached in sys.modules, so setenv after it would never reach it).
# ---------------------------------------------------------------------------


def _fresh_copy() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_watchdog_crash_loop_fresh_copy", CRASH_LOOP_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 5), ("8", 8), ("five", 5), ("0", 1), ("-3", 1)],
)
def test_alarm_count_knob(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: int
) -> None:
    if raw is None:
        monkeypatch.delenv("ORCH_RESTART_RATE_ALARM_COUNT", raising=False)
    else:
        monkeypatch.setenv("ORCH_RESTART_RATE_ALARM_COUNT", raw)
    assert _fresh_copy().RESTART_RATE_ALARM_COUNT == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 1800), ("600", 600), ("30m", 1800), ("0", 1), ("-60", 1)],
)
def test_alarm_window_knob(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: int
) -> None:
    if raw is None:
        monkeypatch.delenv("ORCH_RESTART_RATE_ALARM_WINDOW_SECS", raising=False)
    else:
        monkeypatch.setenv("ORCH_RESTART_RATE_ALARM_WINDOW_SECS", raw)
    assert _fresh_copy().RESTART_RATE_ALARM_WINDOW_SECS == expected
