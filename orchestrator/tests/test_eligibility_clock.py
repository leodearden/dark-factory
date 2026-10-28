"""The durable eligibility clock: one wall-clock anchor per dispatch candidate."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from orchestrator.eligibility_clock import EligibilityClock, EligibilityClockStore

ROOT = '/proj'
T0 = datetime(2026, 10, 1, tzinfo=UTC)


def _at(hours: float) -> datetime:
    return T0 + timedelta(hours=hours)


def _store(tmp_path: Path) -> EligibilityClockStore:
    return EligibilityClockStore(tmp_path / 'scheduler_eligibility.db')


class _BrokenWritesStore(EligibilityClockStore):
    def stamp(self, project_root: str, anchors: Mapping[str, datetime]) -> None:
        raise sqlite3.OperationalError('disk I/O error')

    def forget(self, project_root: str, task_ids: Iterable[str]) -> None:
        raise sqlite3.OperationalError('disk I/O error')


class TestAnchors:
    def test_first_candidacy_stamps_once_and_never_moves(self, tmp_path: Path) -> None:
        clock = EligibilityClock(_store(tmp_path), ROOT)
        clock.observe({'A', 'B'}, now=_at(0))
        clock.observe({'A', 'B', 'C'}, now=_at(1))

        assert clock.age_secs('A', _at(5)) == 5 * 3600
        assert clock.age_secs('C', _at(5)) == 4 * 3600
        assert clock.task_ids() == frozenset({'A', 'B', 'C'})

    def test_restart_reloads_anchors_instead_of_restamping(self, tmp_path: Path) -> None:
        EligibilityClock(_store(tmp_path), ROOT).observe({'A'}, now=_at(0))

        restarted = EligibilityClock(_store(tmp_path), ROOT)
        restarted.observe({'A'}, now=_at(72))

        assert restarted.age_secs('A', _at(72)) == 72 * 3600

    def test_forget_is_durable(self, tmp_path: Path) -> None:
        clock = EligibilityClock(_store(tmp_path), ROOT)
        clock.observe({'A', 'B'}, now=_at(0))
        clock.forget({'A'})

        restarted = EligibilityClock(_store(tmp_path), ROOT)
        assert restarted.task_ids() == frozenset({'B'})
        restarted.observe({'A'}, now=_at(10))
        assert restarted.age_secs('A', _at(12)) == 2 * 3600

    def test_anchors_are_scoped_to_their_project(self, tmp_path: Path) -> None:
        EligibilityClock(_store(tmp_path), ROOT).observe({'A'}, now=_at(0))

        assert EligibilityClock(_store(tmp_path), '/other').task_ids() == frozenset()

    def test_never_observed_task_has_zero_age(self, tmp_path: Path) -> None:
        clock = EligibilityClock(_store(tmp_path), ROOT)
        assert clock.age_secs('ghost', _at(3)) == 0.0

    def test_age_is_never_negative(self, tmp_path: Path) -> None:
        clock = EligibilityClock(_store(tmp_path), ROOT)
        clock.observe({'A'}, now=_at(5))
        assert clock.age_secs('A', _at(4)) == 0.0


class TestFailOpen:
    def test_store_write_errors_leave_memory_authoritative(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        clock = EligibilityClock(
            _BrokenWritesStore(tmp_path / 'scheduler_eligibility.db'), ROOT,
        )
        with caplog.at_level(logging.WARNING, logger='orchestrator.eligibility_clock'):
            clock.observe({'A'}, now=_at(0))
            clock.observe({'A', 'B'}, now=_at(1))
            clock.forget({'B'})

        assert clock.age_secs('A', _at(2)) == 2 * 3600
        assert clock.task_ids() == frozenset({'A'})
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_unreadable_store_starts_empty(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        store = _store(tmp_path)
        store.db_path.unlink()
        store.db_path.mkdir()

        with caplog.at_level(logging.WARNING, logger='orchestrator.eligibility_clock'):
            clock = EligibilityClock(store, ROOT)

        assert clock.task_ids() == frozenset()
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_memory_only_clock(self) -> None:
        clock = EligibilityClock(None, ROOT)
        clock.observe({'A'}, now=_at(0))
        assert clock.age_secs('A', _at(1)) == 3600
        clock.forget({'A'})
        assert clock.task_ids() == frozenset()


class TestStoreFromConfig:
    def test_from_config_uses_the_canonical_path(self, tmp_path: Path) -> None:
        from orchestrator.config import OrchestratorConfig

        config = OrchestratorConfig(project_root=tmp_path)
        store = EligibilityClockStore.from_config(config)

        assert store.db_path == config.eligibility_clock_db_path
        assert store.db_path.exists()
