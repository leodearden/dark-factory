"""One age anchor per task in the scheduler's active fetch (task 6821).

``AgeAnchors(tmp_path)`` is one process; a second ``AgeAnchors(tmp_path)`` on
the same root is a restart: it shares the guard file and starts with empty
memory.  The rule under test is D3 in
``plans/scheduler-dispatch-scoring-and-lock-layer-prd.md``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import shared.safe_io as _safe_io

from orchestrator.age_anchors import AGE_ANCHOR_TTL, AgeAnchor, AgeAnchors
from orchestrator.guard_state import PersistentMap, guard_path

_FILED = '2026-10-01T00:00:00+00:00'
_REOPENED = '2026-10-05T00:00:00+00:00'
_T0 = datetime(2026, 10, 1, tzinfo=UTC)


def _task(tid: str, status: str = 'pending', pending_since: str | None = None) -> dict:
    metadata: dict = {}
    if pending_since is not None:
        metadata['pending_since'] = pending_since
    return {'id': tid, 'status': status, 'metadata': metadata}


def _guard(root: Path) -> Path:
    path = guard_path(root, 'scheduler_age_anchors.json')
    assert path is not None
    return path


def _spy_on_writes(monkeypatch) -> list[Path]:
    writes: list[Path] = []
    real = _safe_io.atomic_write_text

    def recorder(path, text, **kwargs):
        writes.append(Path(path))
        return real(path, text, **kwargs)

    monkeypatch.setattr(_safe_io, 'atomic_write_text', recorder)
    return writes


def _reset_300_to_1000(anchors: AgeAnchors) -> None:
    anchors.observe([_task('300', pending_since=_FILED)], 50)
    anchors.observe([_task('300', pending_since=_REOPENED)], 1000)


class TestFirstSight:
    def test_a_numeric_id_anchors_to_itself(self, tmp_path: Path):
        anchors = AgeAnchors(tmp_path)

        anchors.observe([_task('5')], 100)

        assert anchors.age('5', 100) == 95

    def test_an_id_int_cannot_parse_anchors_to_max_id(self, tmp_path: Path):
        anchors = AgeAnchors(tmp_path)

        anchors.observe([_task('abc'), _task('²')], 100)

        assert anchors.age('abc', 100) == 0
        assert anchors.age('²', 100) == 0
        assert anchors.age('abc', 130) == 30

    def test_an_unobserved_id_has_no_age(self, tmp_path: Path):
        assert AgeAnchors(tmp_path).age('5', 100) == 0

    def test_an_empty_or_terminal_status_is_not_observed(self, tmp_path: Path):
        anchors = AgeAnchors(tmp_path)

        anchors.observe([_task('5', 'done'), _task('6', 'cancelled'), _task('7', '')], 100)

        assert anchors.tracked_ids() == frozenset()
        assert [anchors.age(tid, 100) for tid in ('5', '6', '7')] == [0, 0, 0]

    def test_a_terminal_observation_leaves_a_held_anchor_alone(self, tmp_path: Path):
        anchors = AgeAnchors(tmp_path)
        anchors.observe([_task('5', pending_since=_FILED)], 100)

        anchors.observe([_task('5', 'cancelled', pending_since=_FILED)], 100)

        assert anchors.tracked_ids() == frozenset({'5'})
        assert anchors.age('5', 100) == 95


class TestReset:
    def test_a_changed_stamp_resets_to_max_id_and_is_returned(self, tmp_path: Path):
        anchors = AgeAnchors(tmp_path)
        anchors.observe([_task('1', pending_since=_FILED)], 50)

        resets = anchors.observe([_task('1', pending_since=_REOPENED)], 100)

        assert resets == {'1': AgeAnchor(100, _REOPENED)}
        assert anchors.age('1', 100) == 0

    def test_an_unchanged_re_observation_returns_no_reset(self, tmp_path: Path):
        anchors = AgeAnchors(tmp_path)
        anchors.observe([_task('1', pending_since=_FILED)], 50)

        assert anchors.observe([_task('1', pending_since=_FILED)], 100) == {}

    def test_neither_a_first_nor_a_missing_stamp_is_a_reset(self, tmp_path: Path):
        anchors = AgeAnchors(tmp_path)
        anchors.observe([_task('7')], 100)

        assert anchors.observe([_task('7', pending_since=_FILED)], 200) == {}
        assert anchors.age('7', 200) == 193
        assert anchors.observe([_task('7')], 300) == {}
        assert anchors.age('7', 300) == 293

    def test_a_reset_is_applied_once(self, tmp_path: Path):
        _reset_300_to_1000(AgeAnchors(tmp_path))
        restarted = AgeAnchors(tmp_path)

        assert restarted.observe([_task('300', pending_since=_REOPENED)], 1050) == {}
        assert restarted.age('300', 1050) == 50

    def test_a_reopen_while_no_process_runs_is_still_reset(self, tmp_path: Path):
        AgeAnchors(tmp_path).observe([_task('300', pending_since=_FILED)], 50)
        restarted = AgeAnchors(tmp_path)

        resets = restarted.observe([_task('300', pending_since=_REOPENED)], 1000)

        assert resets == {'300': AgeAnchor(1000, _REOPENED)}


class TestDeparture:
    def test_tracked_ids_are_the_observed_active_ids(self, tmp_path: Path):
        anchors = AgeAnchors(tmp_path)
        anchors.observe([_task('5'), _task('6', 'in-progress'), _task('7', 'done')], 100)

        assert anchors.tracked_ids() == frozenset({'5', '6'})

        anchors.record_departures(['5'])

        assert anchors.tracked_ids() == frozenset({'6'})

    def test_departing_an_untracked_id_writes_nothing(self, tmp_path: Path, monkeypatch):
        anchors = AgeAnchors(tmp_path)
        anchors.observe([_task('5')], 100)
        writes = _spy_on_writes(monkeypatch)

        anchors.record_departures(['9'])

        assert anchors.tracked_ids() == frozenset({'5'})
        assert writes == []

    def test_a_return_restores_the_departed_anchor(self, tmp_path: Path):
        anchors = AgeAnchors(tmp_path)
        _reset_300_to_1000(anchors)
        anchors.record_departures(['300'])

        anchors.observe([_task('300', pending_since=_REOPENED)], 2000)

        assert anchors.age('300', 2000) == 1000

    def test_a_return_after_a_restart_restores_the_departed_anchor(self, tmp_path: Path):
        anchors = AgeAnchors(tmp_path)
        _reset_300_to_1000(anchors)
        anchors.record_departures(['300'])
        restarted = AgeAnchors(tmp_path)

        restarted.observe([_task('300', pending_since=_REOPENED)], 2000)

        assert restarted.age('300', 2000) == 1000


class TestWrites:
    def test_a_cold_start_writes_once_and_a_steady_state_never(
        self, tmp_path: Path, monkeypatch,
    ):
        tasks = [_task(str(tid), pending_since=_FILED) for tid in range(100, 150)]
        anchors = AgeAnchors(tmp_path)
        writes = _spy_on_writes(monkeypatch)

        anchors.observe(tasks, 200)
        assert writes.count(_guard(tmp_path)) == 1

        writes.clear()
        anchors.observe(tasks, 200)
        assert writes.count(_guard(tmp_path)) == 0

    def test_a_mass_departure_writes_once_and_departing_nothing_never(
        self, tmp_path: Path, monkeypatch,
    ):
        ids = [str(tid) for tid in range(100, 150)]
        anchors = AgeAnchors(tmp_path)
        anchors.observe([_task(tid) for tid in ids], 200)
        writes = _spy_on_writes(monkeypatch)

        anchors.record_departures(ids)
        assert writes.count(_guard(tmp_path)) == 1

        writes.clear()
        anchors.record_departures(ids)
        assert writes.count(_guard(tmp_path)) == 0

    def test_no_project_root_keeps_everything_in_memory(self, monkeypatch):
        writes = _spy_on_writes(monkeypatch)
        anchors = AgeAnchors(None)

        _reset_300_to_1000(anchors)
        anchors.observe([_task('5')], 1000)
        anchors.record_departures(['300'])
        anchors.observe([_task('300', pending_since=_REOPENED)], 2000)

        assert anchors.age('300', 2000) == 1000
        assert anchors.age('5', 2000) == 1995
        assert writes == []


class TestRecordLifetime:
    def test_a_record_no_write_refreshed_expires(self, tmp_path: Path):
        clock = [_T0]
        _reset_300_to_1000(AgeAnchors(tmp_path, now=lambda: clock[0]))
        clock[0] = _T0 + AGE_ANCHOR_TTL + timedelta(days=1)
        restarted = AgeAnchors(tmp_path, now=lambda: clock[0])

        restarted.observe([_task('300', pending_since=_REOPENED)], 2000)

        assert restarted.age('300', 2000) == 1700

    def test_a_departure_refreshes_the_departed_record(self, tmp_path: Path):
        clock = [_T0]
        anchors = AgeAnchors(tmp_path, now=lambda: clock[0])
        anchors.observe([_task('300', pending_since=_FILED), _task('99')], 50)
        anchors.observe([_task('300', pending_since=_REOPENED), _task('99')], 1000)

        clock[0] = _T0 + timedelta(days=20)
        anchors.observe([_task('99')], 1000)
        anchors.record_departures(['300'])

        clock[0] = _T0 + timedelta(days=40)
        restarted = AgeAnchors(tmp_path, now=lambda: clock[0])
        restarted.observe([_task('300', pending_since=_REOPENED)], 2000)

        assert restarted.age('300', 2000) == 1000

    def test_a_malformed_record_is_ignored(self, tmp_path: Path):
        PersistentMap(_guard(tmp_path), ttl=timedelta(days=30)).put_many({
            '10': {'anchor': 'not-an-int', 'pending_since': None},
            '11': 'garbage',
        })
        anchors = AgeAnchors(tmp_path)

        anchors.observe([_task('10'), _task('11')], 100)

        assert anchors.age('10', 100) == 90
        assert anchors.age('11', 100) == 89
