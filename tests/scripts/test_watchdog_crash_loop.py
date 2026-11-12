"""Tests for scripts/_watchdog_crash_loop.py, the orchestrator watchdog's crash-loop detector (task 8270).

The module is imported by name, so ``crash_loop`` is the sys.modules instance:
the same object scripts/orchestrator-watchdog.py's from-import binds. The
pure core (status, growth, observation, alarm text) needs no I/O at all.
Every I/O test hands the detector tmp_path as its repo_dir, so its state and
queue land under tmp_path by construction, and the escalation submit runs a
recording fake uv rather than the real one.
"""

import dataclasses
import importlib.util
import json
import pathlib
import sys
import types

import _watchdog_crash_loop as crash_loop
import pytest
from fake_uv import argv_flag, recorded_argvs, write_fake_uv
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

# Resolves because the root conftest.py appends the repo root to sys.path.
from df_pytest_isolation import synthetic_unit

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = pathlib.Path(__file__).parents[2]
CRASH_LOOP_PATH = REPO_ROOT / "scripts" / "_watchdog_crash_loop.py"

UNIT = synthetic_unit("dark-factory")
WINDOW = 1800
COUNT = 5
T = 1_800_000_000

AlarmKind = crash_loop.AlarmKind
UnitRestartHistory = crash_loop.UnitRestartHistory
UnitRestartStatus = crash_loop.UnitRestartStatus


@pytest.fixture(autouse=True)
def _no_path_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ambient override would move a test's state, queue or uv out of tmp_path."""
    for name in (
        "ORCH_RESTART_RATE_STATE",
        "ORCH_CRASH_LOOP_QUEUE_DIR",
        "ORCH_WATCHDOG_UV_BIN",
    ):
        monkeypatch.delenv(name, raising=False)


def _discard(_msg: str) -> None:
    pass


def _status(n: int | None, result: str | None = "success") -> UnitRestartStatus:
    return UnitRestartStatus(n_restarts=n, result=result)


def _observe(
    history: UnitRestartHistory, status: UnitRestartStatus, now: int
) -> crash_loop.Observation:
    return crash_loop.observe_unit(
        UNIT, history, status, now=now, alarm_count=COUNT, window_secs=WINDOW
    )


def _observe_and_land(
    history: UnitRestartHistory, status: UnitRestartStatus, now: int
) -> crash_loop.Observation:
    """Observe, then treat any page the observation decided on as filed successfully."""
    obs = _observe(history, status, now)
    if obs.alarm is None:
        return obs
    return dataclasses.replace(
        obs, history=crash_loop.mark_page_landed(obs.history, obs.alarm)
    )


# ---------------------------------------------------------------------------
# UnitRestartStatus
# ---------------------------------------------------------------------------


def _assign(obj: object, field: str, value: object) -> None:
    setattr(obj, field, value)


def test_unit_restart_status_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _assign(_status(3), "n_restarts", 4)


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
    unknown = crash_loop.UNKNOWN_RESTART_STATUS
    assert unknown.render() == "?/?"
    assert (unknown.n_restarts, unknown.result) == (None, None)
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


def test_start_limit_hit_page_stamps_the_epoch_but_leaves_the_episode_open() -> None:
    """The episode is closed by mark_page_landed after a submit succeeds, not on attempt."""
    obs = _observe(UnitRestartHistory(), _status(10, "start-limit-hit"), T)

    assert obs.alarm is not None
    assert obs.alarm.kind is AlarmKind.START_LIMIT_HIT
    assert obs.history.last_escalation_epoch == T
    assert obs.history.start_limit_paged is False


def test_start_limit_hit_pages_once_per_episode() -> None:
    first = _observe_and_land(UnitRestartHistory(), _status(2, "start-limit-hit"), T)
    assert first.alarm is not None
    assert first.alarm.kind is AlarmKind.START_LIMIT_HIT
    assert first.history.start_limit_paged is True

    still_stuck = _observe_and_land(
        first.history, _status(2, "start-limit-hit"), T + WINDOW + 60
    )
    assert still_stuck.alarm is None
    assert still_stuck.history.start_limit_paged is True

    recovered = _observe_and_land(
        still_stuck.history, _status(0, "success"), T + WINDOW + 120
    )
    assert recovered.alarm is None
    assert recovered.history.start_limit_paged is False

    relapse = _observe_and_land(
        recovered.history, _status(1, "start-limit-hit"), T + 2 * WINDOW + 60
    )
    assert relapse.alarm is not None
    assert relapse.alarm.kind is AlarmKind.START_LIMIT_HIT


def test_start_limit_hit_inside_a_rate_pages_window_is_held_then_pages() -> None:
    history = UnitRestartHistory(samples=((T - 600, 0),), sampling_since=T - 600)
    rate = _observe_and_land(history, _status(5), T)
    assert rate.alarm is not None and rate.alarm.kind is AlarmKind.RESTART_RATE

    held = _observe_and_land(rate.history, _status(5, "start-limit-hit"), T + 60)
    assert held.alarm is None
    assert held.held is not None
    assert held.held.kind is AlarmKind.START_LIMIT_HIT
    assert held.history.start_limit_paged is False

    paged = _observe_and_land(held.history, _status(5, "start-limit-hit"), T + WINDOW)
    assert paged.alarm is not None
    assert paged.alarm.kind is AlarmKind.START_LIMIT_HIT
    assert paged.history.start_limit_paged is True


def test_mark_page_landed_marks_only_a_start_limit_episode() -> None:
    history = UnitRestartHistory(
        samples=((T - 60, 2), (T, 10)), sampling_since=T - 600, last_escalation_epoch=T
    )
    before = dataclasses.replace(history)
    start_limit = crash_loop.CrashLoopAlarm(UNIT, AlarmKind.START_LIMIT_HIT, 8, WINDOW)
    rate = crash_loop.CrashLoopAlarm(UNIT, AlarmKind.RESTART_RATE, 8, WINDOW)

    landed = crash_loop.mark_page_landed(history, start_limit)
    assert landed == dataclasses.replace(history, start_limit_paged=True)

    assert crash_loop.mark_page_landed(history, rate) == history
    assert history == before


def test_unlanded_start_limit_page_retries_at_window_cadence() -> None:
    stuck = _status(10, "start-limit-hit")

    first = _observe(UnitRestartHistory(), stuck, T)
    assert first.alarm is not None
    assert first.alarm.kind is AlarmKind.START_LIMIT_HIT

    held = _observe(first.history, stuck, T + 60)
    assert held.alarm is None
    assert held.held is not None
    assert held.held.kind is AlarmKind.START_LIMIT_HIT

    retry = _observe(held.history, stuck, T + WINDOW)
    assert retry.alarm is not None
    assert retry.alarm.kind is AlarmKind.START_LIMIT_HIT
    assert retry.history.last_escalation_epoch == T + WINDOW


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
        _assign(history, "start_limit_paged", True)


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
# cached in sys.modules, so setenv after it would never reach it). The copy is
# registered under its throwaway name while it executes, because dataclasses
# resolves a class's module through sys.modules.
# ---------------------------------------------------------------------------

_FRESH_COPY_NAME = "_watchdog_crash_loop_fresh_copy"


def _fresh_copy(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(_FRESH_COPY_NAME, CRASH_LOOP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _FRESH_COPY_NAME, module)
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
    count = _fresh_copy(monkeypatch).RESTART_RATE_ALARM_COUNT
    assert count == expected


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
    window = _fresh_copy(monkeypatch).RESTART_RATE_ALARM_WINDOW_SECS
    assert window == expected


# ---------------------------------------------------------------------------
# Paths: resolved at CALL time, so a setenv after import still takes effect
# ---------------------------------------------------------------------------


def test_restart_rate_state_path_defaults_under_repo_dir(tmp_path: pathlib.Path) -> None:
    assert crash_loop.restart_rate_state_path(str(tmp_path)) == str(
        tmp_path / "data" / "orchestrator" / "watchdog-restart-rate.json"
    )


def test_restart_rate_state_path_override_is_read_per_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    override = str(tmp_path / "elsewhere.json")
    monkeypatch.setenv("ORCH_RESTART_RATE_STATE", override)
    assert crash_loop.restart_rate_state_path(str(tmp_path)) == override


def test_crash_loop_queue_dir_defaults_to_the_repos_escalation_queue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    assert crash_loop.crash_loop_queue_dir(str(tmp_path)) == str(
        tmp_path / "data" / "escalations"
    )
    monkeypatch.setenv("ORCH_CRASH_LOOP_QUEUE_DIR", str(tmp_path / "q"))
    assert crash_loop.crash_loop_queue_dir(str(tmp_path)) == str(tmp_path / "q")


def test_watchdog_uv_bin_is_absolute_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The watchdog unit sets no PATH, so a bare 'uv' would not resolve."""
    assert crash_loop.watchdog_uv_bin() == "/home/leo/.local/bin/uv"
    monkeypatch.setenv("ORCH_WATCHDOG_UV_BIN", "/opt/uv")
    assert crash_loop.watchdog_uv_bin() == "/opt/uv"


# ---------------------------------------------------------------------------
# State IO: load_restart_histories / save_restart_histories
# ---------------------------------------------------------------------------


def test_histories_round_trip_through_the_state_file(tmp_path: pathlib.Path) -> None:
    path = str(tmp_path / "state" / "restart-rate.json")
    histories = {
        UNIT: UnitRestartHistory(
            samples=((T - 60, 2), (T, 5)),
            sampling_since=T - 86400,
            last_escalation_epoch=T,
            start_limit_paged=True,
        ),
        synthetic_unit("reify"): UnitRestartHistory(),
    }

    assert crash_loop.save_restart_histories(path, histories, _discard) is True

    assert crash_loop.load_restart_histories(path, _discard) == histories
    on_disk = json.loads(pathlib.Path(path).read_text())
    assert set(on_disk) == set(histories)
    assert on_disk[UNIT]["samples"] == [[T - 60, 2], [T, 5]]


def test_save_reports_a_write_that_did_not_land(tmp_path: pathlib.Path) -> None:
    blocker = tmp_path / "a-file"
    blocker.write_text("")
    logged: list[str] = []

    landed = crash_loop.save_restart_histories(
        str(blocker / "restart-rate.json"), {UNIT: UnitRestartHistory()}, logged.append
    )

    assert landed is False
    assert logged


def test_load_missing_file_is_empty_and_silent(tmp_path: pathlib.Path) -> None:
    logged: list[str] = []
    assert crash_loop.load_restart_histories(str(tmp_path / "absent.json"), logged.append) == {}
    assert logged == []


def test_load_corrupt_file_is_empty_and_logged_once(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "restart-rate.json"
    path.write_text('{"orchestrator-')
    logged: list[str] = []

    assert crash_loop.load_restart_histories(str(path), logged.append) == {}
    assert len(logged) == 1


def test_load_drops_unusable_parts_of_a_hand_edited_entry(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "restart-rate.json"
    path.write_text(
        json.dumps(
            {
                UNIT: {
                    "samples": [
                        [T - 60, 2],
                        ["soon", 3],
                        [T - 30, True],
                        [T - 20],
                        [T - 10, 4, 9],
                        "junk",
                        [T, 5],
                    ],
                    "sampling_since": T - 600,
                    "last_escalation_epoch": -5,
                    "start_limit_paged": True,
                },
                synthetic_unit("reify"): {"last_escalation_epoch": True},
                synthetic_unit("hive"): {"last_escalation_epoch": "yesterday"},
                synthetic_unit("broken"): ["not", "a", "dict"],
            }
        )
    )

    histories = crash_loop.load_restart_histories(str(path), _discard)

    assert histories[UNIT] == UnitRestartHistory(
        samples=((T - 60, 2), (T, 5)),
        sampling_since=T - 600,
        last_escalation_epoch=None,
        start_limit_paged=True,
    )
    assert histories[synthetic_unit("reify")].last_escalation_epoch is None
    assert histories[synthetic_unit("hive")].last_escalation_epoch is None
    assert synthetic_unit("broken") not in histories


# ---------------------------------------------------------------------------
# The page: crash_loop_submit_argv / file_crash_loop_escalation
# ---------------------------------------------------------------------------

_RATE_ALARM = crash_loop.CrashLoopAlarm(UNIT, AlarmKind.RESTART_RATE, 6, WINDOW)


def test_submit_argv_is_the_born_at_l2_shape(tmp_path: pathlib.Path) -> None:
    repo = str(tmp_path)
    queue = str(tmp_path / "q")

    argv = crash_loop.crash_loop_submit_argv(
        _RATE_ALARM, uv_bin="/x/uv", repo_dir=repo, queue_dir=queue
    )

    assert argv == [
        "/x/uv",
        "run",
        "--project",
        str(tmp_path / "escalation"),
        "escalation",
        "submit",
        "--queue-dir",
        queue,
        "--task",
        f"orchestrator-crash-loop-{UNIT}",
        "--severity",
        "critical",
        "--category",
        "infra_issue",
        "--summary",
        _RATE_ALARM.summary(),
        "--detail",
        _RATE_ALARM.detail(),
        "--agent-role",
        "orchestrator-watchdog",
    ]


def test_file_escalation_runs_submit_once_and_logs_the_sentinel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    uv, record = write_fake_uv(tmp_path)
    monkeypatch.setenv("ORCH_WATCHDOG_UV_BIN", str(uv))
    logged: list[str] = []

    filed = crash_loop.file_crash_loop_escalation(
        _RATE_ALARM, repo_dir=str(tmp_path), log=logged.append
    )

    assert filed is True
    assert recorded_argvs(record) == [
        crash_loop.crash_loop_submit_argv(
            _RATE_ALARM,
            uv_bin=str(uv),
            repo_dir=str(tmp_path),
            queue_dir=str(tmp_path / "data" / "escalations"),
        )
    ]
    assert any("filed" in m and _RATE_ALARM.task_sentinel in m for m in logged), logged


def test_file_escalation_reports_a_failed_submit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    uv, _record = write_fake_uv(tmp_path, exit_code=3)
    monkeypatch.setenv("ORCH_WATCHDOG_UV_BIN", str(uv))
    logged: list[str] = []

    filed = crash_loop.file_crash_loop_escalation(
        _RATE_ALARM, repo_dir=str(tmp_path), log=logged.append
    )

    assert filed is False
    assert any("exited 3" in m for m in logged), logged


def test_file_escalation_survives_a_missing_uv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setenv("ORCH_WATCHDOG_UV_BIN", str(tmp_path / "no-such-uv"))
    logged: list[str] = []

    filed = crash_loop.file_crash_loop_escalation(
        _RATE_ALARM, repo_dir=str(tmp_path), log=logged.append
    )

    assert filed is False
    assert logged


def test_file_escalation_survives_undecodable_submit_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """subprocess.run(text=True) decodes after the child exits, so the page may have landed."""

    def undecodable(*_args, **_kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(crash_loop.subprocess, "run", undecodable)
    logged: list[str] = []

    filed = crash_loop.file_crash_loop_escalation(
        _RATE_ALARM, repo_dir=str(tmp_path), log=logged.append
    )

    assert filed is False
    assert any("UnicodeDecodeError" in m and _RATE_ALARM.task_sentinel in m for m in logged)


# ---------------------------------------------------------------------------
# crash_loop_pass: one tick over every unit
# ---------------------------------------------------------------------------

OTHER = synthetic_unit("reify")


def _reader(answers: dict[str, UnitRestartStatus]):
    def read_status(unit: str) -> UnitRestartStatus:
        return answers[unit]

    return read_status


def _tick(
    tmp_path: pathlib.Path,
    answers: dict[str, UnitRestartStatus],
    now: int,
    log: list[str] | None = None,
) -> dict[str, UnitRestartStatus]:
    return crash_loop.crash_loop_pass(
        list(answers),
        _reader(answers),
        repo_dir=str(tmp_path),
        log=(log if log is not None else []).append,
        now=now,
        alarm_count=COUNT,
        window_secs=WINDOW,
    )


def _state(tmp_path: pathlib.Path) -> dict:
    return json.loads(
        pathlib.Path(crash_loop.restart_rate_state_path(str(tmp_path))).read_text()
    )


def test_pass_pages_once_for_a_restart_burst(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    uv, record = write_fake_uv(tmp_path)
    monkeypatch.setenv("ORCH_WATCHDOG_UV_BIN", str(uv))

    _tick(tmp_path, {UNIT: _status(0)}, T)
    _tick(tmp_path, {UNIT: _status(5)}, T + 600)

    calls = recorded_argvs(record)
    assert len(calls) == 1
    argv = calls[0]
    assert argv_flag(argv, "--task") == f"orchestrator-crash-loop-{UNIT}"
    assert argv_flag(argv, "--severity") == "critical"
    assert argv_flag(argv, "--category") == "infra_issue"
    assert argv_flag(argv, "--agent-role").startswith("orchestrator-")
    assert argv_flag(argv, "--queue-dir") == str(tmp_path / "data" / "escalations")

    logged: list[str] = []
    _tick(tmp_path, {UNIT: _status(6)}, T + 660, logged)

    assert len(recorded_argvs(record)) == 1
    unit_state = _state(tmp_path)[UNIT]
    assert unit_state["samples"] == [[T, 0], [T + 600, 5], [T + 660, 6]]
    assert unit_state["last_escalation_epoch"] == T + 600
    assert any("not re-filing" in m and UNIT in m for m in logged), logged


def test_pass_pages_start_limit_hit_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    uv, record = write_fake_uv(tmp_path)
    monkeypatch.setenv("ORCH_WATCHDOG_UV_BIN", str(uv))

    _tick(tmp_path, {UNIT: _status(2, "start-limit-hit")}, T)
    assert len(recorded_argvs(record)) == 1

    _tick(tmp_path, {UNIT: _status(2, "start-limit-hit")}, T + 2000)
    assert len(recorded_argvs(record)) == 1


def test_pass_retries_a_start_limit_hit_page_that_failed_to_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A broken submit must not mark the episode paged: the unit stays down meanwhile."""
    uv, record = write_fake_uv(tmp_path, exit_code=3)
    monkeypatch.setenv("ORCH_WATCHDOG_UV_BIN", str(uv))
    stuck = {UNIT: _status(10, "start-limit-hit")}

    _tick(tmp_path, stuck, T)
    assert len(recorded_argvs(record)) == 1
    assert _state(tmp_path)[UNIT]["start_limit_paged"] is False

    logged: list[str] = []
    _tick(tmp_path, stuck, T + 60, logged)
    assert len(recorded_argvs(record)) == 1
    assert any("not re-filing" in m and "attempt" in m for m in logged), logged

    _tick(tmp_path, stuck, T + WINDOW)
    assert len(recorded_argvs(record)) == 2

    write_fake_uv(tmp_path, exit_code=0)

    _tick(tmp_path, stuck, T + 2 * WINDOW)
    assert len(recorded_argvs(record)) == 3
    assert _state(tmp_path)[UNIT]["start_limit_paged"] is True

    _tick(tmp_path, stuck, T + 3 * WINDOW)
    assert len(recorded_argvs(record)) == 3


def test_pass_keeps_the_attempt_stamped_when_filing_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A raise after the decision must not unstamp it, or every tick would re-page."""
    attempts: list[crash_loop.CrashLoopAlarm] = []

    def raising_file(alarm, **_kwargs):
        attempts.append(alarm)
        raise RuntimeError("boom after submit")

    monkeypatch.setattr(crash_loop, "file_crash_loop_escalation", raising_file)

    _tick(tmp_path, {UNIT: _status(0)}, T)
    _tick(tmp_path, {UNIT: _status(5)}, T + 600)

    assert len(attempts) == 1
    unit_state = _state(tmp_path)[UNIT]
    assert unit_state["samples"] == [[T, 0], [T + 600, 5]]
    assert unit_state["last_escalation_epoch"] == T + 600

    _tick(tmp_path, {UNIT: _status(6)}, T + 660)

    assert len(attempts) == 1


def test_pass_returns_every_status_and_one_unit_cannot_blind_the_rest(
    tmp_path: pathlib.Path,
) -> None:
    def read_status(unit: str) -> UnitRestartStatus:
        if unit == UNIT:
            raise RuntimeError("systemd went away")
        return _status(1)

    logged: list[str] = []
    statuses = crash_loop.crash_loop_pass(
        [UNIT, OTHER],
        read_status,
        repo_dir=str(tmp_path),
        log=logged.append,
        now=T,
        alarm_count=COUNT,
        window_secs=WINDOW,
    )

    assert statuses == {OTHER: _status(1)}
    assert _state(tmp_path)[OTHER]["samples"] == [[T, 1]]
    assert any(UNIT in m and "systemd went away" in m for m in logged), logged


def test_pass_returns_statuses_when_the_state_cannot_be_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    blocker = tmp_path / "a-file"
    blocker.write_text("")
    monkeypatch.setenv("ORCH_RESTART_RATE_STATE", str(blocker / "restart-rate.json"))
    answers = {UNIT: _status(0), OTHER: _status(2, "exit-code")}

    assert _tick(tmp_path, answers, T) == answers


def test_pass_with_every_status_unknown_files_nothing_but_saves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    uv, record = write_fake_uv(tmp_path)
    monkeypatch.setenv("ORCH_WATCHDOG_UV_BIN", str(uv))
    unknown = crash_loop.UNKNOWN_RESTART_STATUS

    statuses = _tick(tmp_path, {UNIT: unknown, OTHER: unknown}, T)

    assert statuses == {UNIT: unknown, OTHER: unknown}
    assert recorded_argvs(record) == []
    assert set(_state(tmp_path)) == {UNIT, OTHER}
