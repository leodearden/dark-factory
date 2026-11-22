"""Reconciliation steps aside while the allowance reserve refuses its background work.

Contract: plans/allowance-reserve-by-account-prd.md D13 and §6.4. Every test drives a real
UsageGate from ``shared.testing_reserve`` on a reserve clock that moves only when told to; the
gate is never mocked.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import pytest
from shared.testing_reserve import (
    PROTECTED,
    RELEASE_AT,
    RESERVE_ACCOUNT,
    ReserveClock,
    cap_all_but_reserve,
    reserve_gate,
)
from shared.usage_gate import UsageGate

from fused_memory.reconciliation.reserve_step_aside import (
    DEFERRAL_LOG_INTERVAL_SECS,
    ReserveStepAside,
    StepAsideSite,
)

_STEP_ASIDE_LOGGER = 'fused_memory.reconciliation.reserve_step_aside'


class MonotonicClock:
    """A monotonic clock that moves only when a test advances it."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def b2_gate(tmp_path: Path, clock: ReserveClock) -> UsageGate:
    gate = reserve_gate(tmp_path, clock=clock)
    cap_all_but_reserve(gate)
    return gate


def deferral_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records
        if record.name == _STEP_ASIDE_LOGGER and record.levelno == logging.INFO
    ]


class TestReserveStepAside:
    def test_a_refused_check_carries_the_refusal(self, tmp_path: Path) -> None:
        step_aside = ReserveStepAside(b2_gate(tmp_path, ReserveClock(PROTECTED)), clock=MonotonicClock())

        check = step_aside.check('p', site=StepAsideSite.PROJECT_LOOP)

        assert check.refused
        assert check.refusal is not None
        assert (check.refusal.account, check.refusal.release_at) == (RESERVE_ACCOUNT, RELEASE_AT)
        assert check.refusal_ended is False

    def test_the_deferral_is_logged_once_per_interval_per_project(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO, logger=_STEP_ASIDE_LOGGER)
        clock = MonotonicClock()
        step_aside = ReserveStepAside(b2_gate(tmp_path, ReserveClock(PROTECTED)), clock=clock)

        for _ in range(3):
            step_aside.check('p', site=StepAsideSite.BACKLOG_CHUNK)
            clock.advance(DEFERRAL_LOG_INTERVAL_SECS / 4)
        [first] = deferral_records(caplog)
        message = first.getMessage()
        for named in ("'p'", StepAsideSite.BACKLOG_CHUNK.value, RESERVE_ACCOUNT, RELEASE_AT.isoformat()):
            assert named in message

        clock.advance(DEFERRAL_LOG_INTERVAL_SECS)
        step_aside.check('p', site=StepAsideSite.BACKLOG_CHUNK)
        assert len(deferral_records(caplog)) == 2

        step_aside.check('q', site=StepAsideSite.JUDGE)
        third = deferral_records(caplog)[2].getMessage()
        assert "'q'" in third and StepAsideSite.JUDGE.value in third

    def test_the_first_check_after_a_refusal_reports_its_end_once(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO, logger=_STEP_ASIDE_LOGGER)
        reserve_clock = ReserveClock(PROTECTED)
        step_aside = ReserveStepAside(b2_gate(tmp_path, reserve_clock), clock=MonotonicClock())
        assert step_aside.check('p', site=StepAsideSite.PROJECT_LOOP).refused

        reserve_clock.advance(RELEASE_AT - PROTECTED + timedelta(minutes=1))
        ended = step_aside.check('p', site=StepAsideSite.PROJECT_LOOP)
        after = step_aside.check('p', site=StepAsideSite.PROJECT_LOOP)

        assert (ended.refused, ended.refusal_ended) == (False, True)
        assert (after.refused, after.refusal_ended) == (False, False)

        reserve_clock.now = PROTECTED
        assert step_aside.check('p', site=StepAsideSite.PROJECT_LOOP).refused
        assert len(deferral_records(caplog)) == 2

    def test_without_a_gate_nothing_is_refused_and_no_edge_is_seen(self) -> None:
        step_aside = ReserveStepAside(None, clock=MonotonicClock())

        checks = [step_aside.check('p', site=site) for site in StepAsideSite]

        assert [(c.refused, c.refusal, c.refusal_ended) for c in checks] == [(False, None, False)] * 3

    def test_an_open_pool_in_the_protected_window_is_not_refused(self, tmp_path: Path) -> None:
        step_aside = ReserveStepAside(
            reserve_gate(tmp_path, clock=ReserveClock(PROTECTED)), clock=MonotonicClock(),
        )

        check = step_aside.check('p', site=StepAsideSite.PROJECT_LOOP)

        assert (check.refused, check.refusal, check.refusal_ended) == (False, None, False)
