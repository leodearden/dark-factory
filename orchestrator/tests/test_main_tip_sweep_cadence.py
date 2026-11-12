"""MainSweepColdControl: the main-tip sweep's warm-by-default, periodically
cold build cadence (task 5812)."""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime, timedelta

from orchestrator.main_tip_sweep_cadence import MainSweepColdControl, SweepSeedMode
from orchestrator.run_store import RunStore

ONE_DAY = 86400.0
SHA = 'a' * 40
T0 = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
CADENCE_LOGGER = 'orchestrator.main_tip_sweep_cadence'


class FakeClock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class UnreadableLedger:
    def load_main_sweep_last_cold_verdict(self, project_id: str) -> datetime | None:
        raise sqlite3.OperationalError('database is locked')

    def save_main_sweep_cold_verdict(
        self, project_id: str, *, verdict_at: datetime, swept_sha: str,
    ) -> None:
        return None


class UnwritableLedger:
    def load_main_sweep_last_cold_verdict(self, project_id: str) -> datetime | None:
        return None

    def save_main_sweep_cold_verdict(
        self, project_id: str, *, verdict_at: datetime, swept_sha: str,
    ) -> None:
        raise sqlite3.OperationalError('disk I/O error')


def test_first_sweep_with_no_history_is_cold():
    control = MainSweepColdControl(None, 'p', clock=FakeClock())
    assert control.seed_mode(ONE_DAY) is SweepSeedMode.COLD


def test_within_interval_after_a_cold_verdict_is_warm():
    clock = FakeClock()
    control = MainSweepColdControl(None, 'p', clock=clock)
    control.record_verdict(SweepSeedMode.COLD, SHA)

    clock.advance(3600)

    assert control.seed_mode(ONE_DAY) is SweepSeedMode.WARM


def test_interval_elapsed_since_last_cold_verdict_is_cold():
    clock = FakeClock()
    control = MainSweepColdControl(None, 'p', clock=clock)
    control.record_verdict(SweepSeedMode.COLD, SHA)

    clock.advance(ONE_DAY)

    assert control.seed_mode(ONE_DAY) is SweepSeedMode.COLD


def test_warm_verdicts_do_not_reset_the_cold_clock():
    clock = FakeClock()
    control = MainSweepColdControl(None, 'p', clock=clock)
    control.record_verdict(SweepSeedMode.COLD, SHA)
    clock.advance(3600)
    control.record_verdict(SweepSeedMode.WARM, SHA)
    clock.advance(22 * 3600)
    control.record_verdict(SweepSeedMode.WARM, SHA)

    clock.advance(3600)

    assert control.seed_mode(ONE_DAY) is SweepSeedMode.COLD


def test_zero_interval_is_always_cold():
    control = MainSweepColdControl(None, 'p', clock=FakeClock())
    control.record_verdict(SweepSeedMode.COLD, SHA)
    assert control.seed_mode(0) is SweepSeedMode.COLD


def test_clock_behind_last_cold_verdict_fails_toward_cold():
    clock = FakeClock()
    control = MainSweepColdControl(None, 'p', clock=clock)
    control.record_verdict(SweepSeedMode.COLD, SHA)

    clock.advance(-3600)

    assert control.seed_mode(ONE_DAY) is SweepSeedMode.COLD


def test_last_cold_verdict_at_is_exposed():
    clock = FakeClock()
    control = MainSweepColdControl(None, 'p', clock=clock)
    assert control.last_cold_verdict_at is None

    control.record_verdict(SweepSeedMode.COLD, SHA)

    assert control.last_cold_verdict_at == T0


def test_cold_verdict_is_restored_from_the_ledger_after_a_restart(tmp_path):
    store = RunStore(tmp_path / 'runs.db')
    clock = FakeClock()
    MainSweepColdControl(store, 'p', clock=clock).record_verdict(SweepSeedMode.COLD, SHA)

    clock.advance(3600)
    restarted = MainSweepColdControl(store, 'p', clock=clock)

    assert restarted.seed_mode(ONE_DAY) is SweepSeedMode.WARM


def test_unreadable_ledger_degrades_to_cold(caplog):
    with caplog.at_level(logging.WARNING, logger=CADENCE_LOGGER):
        control = MainSweepColdControl(UnreadableLedger(), 'p', clock=FakeClock())

    assert control.seed_mode(ONE_DAY) is SweepSeedMode.COLD
    assert any(
        r.levelno == logging.WARNING and r.name == CADENCE_LOGGER for r in caplog.records
    ), caplog.text


def test_unwritable_ledger_keeps_the_in_memory_stamp(caplog):
    clock = FakeClock()
    control = MainSweepColdControl(UnwritableLedger(), 'p', clock=clock)

    with caplog.at_level(logging.WARNING, logger=CADENCE_LOGGER):
        control.record_verdict(SweepSeedMode.COLD, SHA)

    assert any(
        r.levelno == logging.WARNING and r.name == CADENCE_LOGGER for r in caplog.records
    ), caplog.text
    clock.advance(3600)
    assert control.seed_mode(ONE_DAY) is SweepSeedMode.WARM
