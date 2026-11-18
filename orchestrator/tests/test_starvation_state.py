"""The starvation watchdog's restart-durable state: refusals, announcements, the cap cohort."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from orchestrator.eligibility_clock import EligibilityClock, EligibilityClockStore
from orchestrator.starvation_state import CapBaseline, StarvationStateStore, WatchdogState
from orchestrator.starvation_watchdog import StarvationGate

ROOT = '/proj'
T0 = datetime(2026, 11, 1, tzinfo=UTC)
BASELINE = CapBaseline(
    size=4, modules=frozenset({'hot/d.py', 'hot/e.py'}), subjects=frozenset({'0'}),
)


def _db(tmp_path: Path) -> Path:
    return tmp_path / 'scheduler_eligibility.db'


def _state(tmp_path: Path, root: str = ROOT) -> WatchdogState:
    return WatchdogState(StarvationStateStore(_db(tmp_path)), root)


class _BrokenWritesStore(StarvationStateStore):
    def increment_refusals(self, project_root: str, task_ids: Iterable[str]) -> None:
        raise sqlite3.OperationalError('disk I/O error')

    def announce(self, project_root: str, task_id: str) -> None:
        raise sqlite3.OperationalError('disk I/O error')

    def wait(self, project_root: str, task_id: str, gate: StarvationGate) -> None:
        raise sqlite3.OperationalError('disk I/O error')

    def drop_waiting(self, project_root: str, task_ids: Iterable[str]) -> None:
        raise sqlite3.OperationalError('disk I/O error')

    def forget(self, project_root: str, task_ids: Iterable[str]) -> None:
        raise sqlite3.OperationalError('disk I/O error')

    def record_baseline(self, project_root: str, baseline: CapBaseline) -> None:
        raise sqlite3.OperationalError('disk I/O error')


class TestLockRefusals:
    def test_counts_scans_per_task(self, tmp_path: Path) -> None:
        state = _state(tmp_path)
        state.note_scan(['a', 'b'])
        state.note_scan(['a', 'b'])
        state.note_scan(['a'])

        assert state.refusals('a') == 3
        assert state.refusals('b') == 2
        assert state.refusals('never') == 0
        assert state.task_ids() == frozenset({'a', 'b'})

    def test_a_task_listed_twice_in_one_scan_counts_once(self, tmp_path: Path) -> None:
        state = _state(tmp_path)
        state.note_scan(['a', 'a'])

        assert state.refusals('a') == 1
        assert _state(tmp_path).refusals('a') == 1

    def test_forget_drops_the_count(self, tmp_path: Path) -> None:
        state = _state(tmp_path)
        state.note_scan(['a', 'b'])
        state.forget(['a'])

        assert state.refusals('a') == 0
        assert state.task_ids() == frozenset({'b'})


class TestRestart:
    def test_a_restart_sees_every_fact(self, tmp_path: Path) -> None:
        first = _state(tmp_path)
        first.note_scan(['a', 'b'])
        first.note_scan(['a'])
        first.announce('a')
        first.wait('b', StarvationGate.LOCK_REFUSED)
        first.wait('c', StarvationGate.DUAL)
        first.record_baseline(BASELINE)

        restarted = _state(tmp_path)

        assert restarted.refusals('a') == 2
        assert restarted.refusals('b') == 1
        assert restarted.announced() == frozenset({'a'})
        assert dict(restarted.waiting()) == {
            'b': StarvationGate.LOCK_REFUSED, 'c': StarvationGate.DUAL,
        }
        assert restarted.baseline() == BASELINE
        assert restarted.task_ids() == frozenset({'a', 'b', 'c'})

    def test_refusals_keep_counting_after_a_restart(self, tmp_path: Path) -> None:
        _state(tmp_path).note_scan(['a'])
        restarted = _state(tmp_path)
        restarted.note_scan(['a'])

        assert restarted.refusals('a') == 2
        assert _state(tmp_path).refusals('a') == 2

    def test_rewaiting_replaces_the_gate(self, tmp_path: Path) -> None:
        state = _state(tmp_path)
        state.wait('b', StarvationGate.LOCK_REFUSED)
        state.wait('b', StarvationGate.DUAL)

        assert dict(_state(tmp_path).waiting()) == {'b': StarvationGate.DUAL}

    def test_an_unrecorded_project_has_the_empty_baseline(self, tmp_path: Path) -> None:
        baseline = _state(tmp_path).baseline()

        assert baseline == CapBaseline.EMPTY
        assert baseline.size == 0
        assert baseline.modules == frozenset()
        assert baseline.subjects == frozenset()

    def test_a_recorded_baseline_is_replaced_not_merged(self, tmp_path: Path) -> None:
        state = _state(tmp_path)
        state.record_baseline(BASELINE)
        state.record_baseline(CapBaseline.EMPTY)

        assert _state(tmp_path).baseline() == CapBaseline.EMPTY

    def test_an_unreadable_baseline_restarts_from_the_empty_one(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        state = _state(tmp_path)
        state.note_scan(['a'])
        state.record_baseline(BASELINE)
        conn = sqlite3.connect(str(_db(tmp_path)))
        with conn:
            conn.execute("UPDATE cap_announcement SET modules='not json'")
        conn.close()

        with caplog.at_level(logging.WARNING, logger='orchestrator.starvation_state'):
            restarted = _state(tmp_path)

        assert restarted.baseline() == CapBaseline.EMPTY
        assert restarted.refusals('a') == 1
        assert any('not json' in r.getMessage() for r in caplog.records)

    def test_a_waiting_row_with_an_unknown_gate_is_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        state = _state(tmp_path)
        state.wait('b', StarvationGate.LOCK_REFUSED)
        state.wait('c', StarvationGate.DUAL)
        conn = sqlite3.connect(str(_db(tmp_path)))
        with conn:
            conn.execute("UPDATE cap_waiting SET gate='retired_gate' WHERE task_id='c'")
        conn.close()

        with caplog.at_level(logging.WARNING, logger='orchestrator.starvation_state'):
            restarted = _state(tmp_path)

        assert dict(restarted.waiting()) == {'b': StarvationGate.LOCK_REFUSED}
        assert any('retired_gate' in r.getMessage() for r in caplog.records)


class TestForget:
    def test_forget_drops_every_fact_of_exactly_those_tasks_durably(self, tmp_path: Path) -> None:
        state = _state(tmp_path)
        state.note_scan(['a', 'b'])
        state.announce('a')
        state.announce('b')
        state.wait('a', StarvationGate.LOCK_REFUSED)
        state.wait('b', StarvationGate.DUAL)
        state.record_baseline(BASELINE)

        state.forget(['a'])

        for view in (state, _state(tmp_path)):
            assert view.refusals('a') == 0
            assert view.refusals('b') == 1
            assert view.announced() == frozenset({'b'})
            assert dict(view.waiting()) == {'b': StarvationGate.DUAL}
            assert view.task_ids() == frozenset({'b'})
            assert view.baseline() == BASELINE

    def test_drop_waiting_drops_only_the_waiting_entry(self, tmp_path: Path) -> None:
        state = _state(tmp_path)
        state.note_scan(['a'])
        state.announce('a')
        state.wait('a', StarvationGate.LOCK_REFUSED)
        state.wait('b', StarvationGate.DUAL)

        state.drop_waiting(['a', 'b'])

        for view in (state, _state(tmp_path)):
            assert dict(view.waiting()) == {}
            assert view.refusals('a') == 1
            assert view.announced() == frozenset({'a'})
            assert view.task_ids() == frozenset({'a'})

    def test_waiting_is_a_snapshot_its_reader_may_drop_from(self, tmp_path: Path) -> None:
        state = _state(tmp_path)
        state.wait('a', StarvationGate.LOCK_REFUSED)
        state.wait('b', StarvationGate.DUAL)
        before = state.waiting()

        state.drop_waiting(tid for tid in state.waiting())

        assert dict(before) == {'a': StarvationGate.LOCK_REFUSED, 'b': StarvationGate.DUAL}
        assert dict(_state(tmp_path).waiting()) == {}

    def test_task_ids_is_the_union_of_every_per_task_fact(self, tmp_path: Path) -> None:
        state = _state(tmp_path)
        state.note_scan(['r'])
        state.announce('n')
        state.wait('w', StarvationGate.DUAL)

        assert state.task_ids() == frozenset({'r', 'n', 'w'})


class TestScope:
    def test_two_projects_in_one_file_are_isolated(self, tmp_path: Path) -> None:
        mine = _state(tmp_path)
        mine.note_scan(['a'])
        mine.announce('a')
        mine.wait('a', StarvationGate.DUAL)
        mine.record_baseline(BASELINE)

        other = _state(tmp_path, '/other')
        other.note_scan(['a'])
        other.forget(['a'])

        assert other.task_ids() == frozenset()
        assert other.baseline() == CapBaseline.EMPTY
        restarted = _state(tmp_path)
        assert restarted.refusals('a') == 1
        assert restarted.announced() == frozenset({'a'})
        assert restarted.baseline() == BASELINE

    def test_the_file_is_shared_with_the_eligibility_anchors(self, tmp_path: Path) -> None:
        EligibilityClock(EligibilityClockStore(_db(tmp_path)), ROOT).observe({'A'}, now=T0)
        state = _state(tmp_path)
        state.note_scan(['A'])
        state.record_baseline(BASELINE)
        EligibilityClock(EligibilityClockStore(_db(tmp_path)), ROOT).observe({'B'}, now=T0)

        assert EligibilityClock(EligibilityClockStore(_db(tmp_path)), ROOT).task_ids() == (
            frozenset({'A', 'B'})
        )
        restarted = _state(tmp_path)
        assert restarted.refusals('A') == 1
        assert restarted.baseline() == BASELINE

    def test_load_from_reads_the_file_a_previous_process_wrote(self, tmp_path: Path) -> None:
        _state(tmp_path).note_scan(['a'])

        assert WatchdogState.load_from(_db(tmp_path), ROOT).refusals('a') == 1


class TestFailOpen:
    def test_store_write_errors_leave_memory_authoritative(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        state = WatchdogState(_BrokenWritesStore(_db(tmp_path)), ROOT)
        with caplog.at_level(logging.WARNING, logger='orchestrator.starvation_state'):
            state.note_scan(['a', 'b'])
            state.announce('a')
            state.wait('a', StarvationGate.LOCK_REFUSED)
            state.wait('b', StarvationGate.DUAL)
            state.drop_waiting(['b'])
            state.forget(['b'])
            state.record_baseline(BASELINE)

        assert state.refusals('a') == 1
        assert state.refusals('b') == 0
        assert state.announced() == frozenset({'a'})
        assert dict(state.waiting()) == {'a': StarvationGate.LOCK_REFUSED}
        assert state.baseline() == BASELINE
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_unreadable_store_starts_empty(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        store = StarvationStateStore(_db(tmp_path))
        store.db_path.unlink()
        store.db_path.mkdir()

        with caplog.at_level(logging.WARNING, logger='orchestrator.starvation_state'):
            state = WatchdogState(store, ROOT)

        assert state.task_ids() == frozenset()
        assert state.baseline() == CapBaseline.EMPTY
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_an_unopenable_file_keeps_the_state_in_memory(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        _db(tmp_path).mkdir()

        with caplog.at_level(logging.WARNING, logger='orchestrator.starvation_state'):
            state = WatchdogState.load_from(_db(tmp_path), ROOT)
        state.note_scan(['a'])

        assert state.refusals('a') == 1
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_no_file_is_memory_only(self) -> None:
        state = WatchdogState.load_from(None, ROOT)
        state.note_scan(['a'])

        assert state.refusals('a') == 1

    def test_memory_only_state(self) -> None:
        state = WatchdogState(None, ROOT)
        state.note_scan(['a'])
        state.announce('a')
        state.wait('a', StarvationGate.DUAL)
        state.record_baseline(BASELINE)

        assert state.refusals('a') == 1
        assert state.announced() == frozenset({'a'})
        assert state.baseline() == BASELINE
        state.forget(['a'])
        assert state.task_ids() == frozenset()
