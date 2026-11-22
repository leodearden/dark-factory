"""Reconciliation steps aside while the allowance reserve refuses its background work.

Contract: plans/allowance-reserve-by-account-prd.md D13 and §6.4. Every test drives a real
UsageGate from ``shared.testing_reserve`` on a reserve clock that moves only when told to; the
gate is never mocked.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from reconciliation.harness_fixtures import (
    make_harness,
    make_harness_config,
    open_event_buffer,
    open_journal,
    stub_stage_runs,
)
from shared.testing_reserve import (
    PROTECTED,
    RELEASE_AT,
    RESERVE_ACCOUNT,
    ReserveClock,
    cap_all_but_reserve,
    reserve_gate,
)
from shared.usage_gate import UsageGate

from fused_memory.models.reconciliation import (
    AssembledPayload,
    EventSource,
    EventType,
    JudgeVerdict,
    ReconciliationEvent,
    StageReport,
    VerdictAction,
    VerdictSeverity,
)
from fused_memory.reconciliation.harness import BacklogIterator, ReconciliationHarness
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


@contextlib.asynccontextmanager
async def open_reserve_harness(
    tmp_path: Path, gate: UsageGate, **config_overrides: Any,
) -> AsyncIterator[ReconciliationHarness]:
    """A harness on real stores under *tmp_path*, its every recon call admitted by *gate*."""
    async with (
        open_journal(tmp_path / 'journal') as journal,
        open_event_buffer(tmp_path / 'events.db') as event_buffer,
    ):
        harness = make_harness(
            journal=journal,
            event_buffer=event_buffer,
            taskmaster=AsyncMock(),
            config=make_harness_config(**{'judge_enabled': False, **config_overrides}),
            known_projects={'test-project': str(tmp_path)},
            usage_gate=gate,
        )
        try:
            yield harness
        finally:
            await gate.shutdown()


def make_event() -> ReconciliationEvent:
    return ReconciliationEvent(
        id=str(uuid.uuid4()),
        type=EventType.episode_added,
        source=EventSource.agent,
        project_id='test-project',
        timestamp=datetime.now(UTC),
        payload={},
    )


async def buffer_events(harness: ReconciliationHarness, count: int) -> None:
    for _ in range(count):
        await harness.buffer.push(make_event())


def record_stage_runs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub every pipeline stage with an empty report; return the run ids they are entered for."""
    entered: list[str] = []

    async def fake_run(
        self, events, watermark, prior_reports, run_id, model=None, resume_session_id=None,
    ) -> StageReport:
        entered.append(run_id)
        now = datetime.now(UTC)
        return StageReport(stage=self.stage_id, started_at=now, completed_at=now)

    stub_stage_runs(monkeypatch, fake_run)
    return entered


def one_event_per_chunk_assembler(memory_service, taskmaster, config, project_root=''):
    """A ContextAssembler stand-in whose every chunk is the oldest peeked event."""
    assembler = MagicMock()

    async def assemble(peeked, project_id):
        if not peeked:
            return AssembledPayload(events=[])
        return AssembledPayload(events=[peeked[0]], events_remaining=len(peeked) - 1)

    assembler.assemble = assemble
    return assembler


_ASSEMBLER_PATH = 'fused_memory.reconciliation.context_assembler.ContextAssembler'


class TestDrainSitesStepAside:
    @pytest.mark.asyncio
    async def test_the_project_loop_drains_nothing_while_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO, logger=_STEP_ASIDE_LOGGER)
        entered = record_stage_runs(monkeypatch)
        async with open_reserve_harness(tmp_path, b2_gate(tmp_path, ReserveClock(PROTECTED))) as harness:
            await buffer_events(harness, 3)
            assert (await harness.buffer.should_trigger('test-project'))[0]
            sleeps = 0
            third_sleep = asyncio.Event()

            async def counting_sleep(_seconds: float) -> None:
                nonlocal sleeps
                await asyncio.sleep(0)
                sleeps += 1
                if sleeps >= 3:
                    third_sleep.set()

            with patch('fused_memory.reconciliation.harness._sleep', counting_sleep):
                loop_task = asyncio.create_task(harness._project_loop('test-project'))
                try:
                    await third_sleep.wait()
                finally:
                    loop_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await loop_task

            assert await harness.journal.get_recent_runs('test-project') == []
            assert (await harness.buffer.get_buffer_stats('test-project'))['size'] == 3
        assert entered == []
        assert len(deferral_records(caplog)) == 1

    @pytest.mark.asyncio
    async def test_a_backlog_refused_before_its_first_chunk_drains_nothing(
        self, tmp_path: Path,
    ) -> None:
        async with open_reserve_harness(tmp_path, b2_gate(tmp_path, ReserveClock(PROTECTED))) as harness:
            await buffer_events(harness, 3)
            with (
                patch(_ASSEMBLER_PATH, side_effect=one_event_per_chunk_assembler),
                patch.object(harness, 'run_full_cycle', new=AsyncMock()) as run_full_cycle,
            ):
                await BacklogIterator(
                    harness.config, harness.journal, harness.buffer, harness,
                ).run('test-project')

            assert await harness.journal.get_recent_runs('test-project') == []
            assert (await harness.buffer.get_buffer_stats('test-project'))['size'] == 3
        run_full_cycle.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_refusal_between_chunks_stops_the_backlog_before_its_consolidation(
        self, tmp_path: Path,
    ) -> None:
        gate = reserve_gate(tmp_path, clock=ReserveClock(PROTECTED))
        async with open_reserve_harness(tmp_path, gate) as harness:
            await buffer_events(harness, 3)
            reasons: list[str] = []

            async def first_chunk_caps_the_pool(project_id: str, trigger_reason: str, **_: Any) -> None:
                reasons.append(trigger_reason)
                cap_all_but_reserve(gate)

            with (
                patch(_ASSEMBLER_PATH, side_effect=one_event_per_chunk_assembler),
                patch.object(harness, 'run_full_cycle', side_effect=first_chunk_caps_the_pool),
            ):
                await BacklogIterator(
                    harness.config, harness.journal, harness.buffer, harness,
                ).run('test-project')

            assert reasons == ['backlog_chunk:1:1']
            assert await harness.buffer.count_buffered('test-project') == 2


async def wait_until(holds: Callable[[], Awaitable[bool]]) -> None:
    """Poll *holds* until it answers True; the per-test timeout is the only deadlock backstop."""
    while not await holds():
        await asyncio.sleep(0.01)


async def drive_one_project_loop_cycle(harness: ReconciliationHarness) -> None:
    """Drive the project loop until its first cycle has released the run lock, then cancel it."""
    released = asyncio.Event()
    mark_run_complete = harness.buffer.mark_run_complete

    async def witnessed(*args: Any, **kwargs: Any) -> Any:
        try:
            return await mark_run_complete(*args, **kwargs)
        finally:
            released.set()

    with patch.object(harness.buffer, 'mark_run_complete', side_effect=witnessed):
        loop_task = asyncio.create_task(harness._project_loop('test-project'))
        try:
            async def cycle_released() -> bool:
                return released.is_set() or loop_task.done()

            await wait_until(cycle_released)
        finally:
            loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await loop_task


class TestJudgesStepAside:
    @pytest.mark.asyncio
    async def test_a_refused_judge_waits_and_is_recovered_once_when_the_refusal_ends(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        record_stage_runs(monkeypatch)
        reserve_clock = ReserveClock(PROTECTED)
        gate = reserve_gate(tmp_path, clock=reserve_clock)
        async with open_reserve_harness(
            tmp_path, gate,
            judge_enabled=True, judge_infra_max_consecutive_failures=1, halt_on_judge_serious=True,
        ) as harness:
            assert harness.judge is not None
            journal = harness.journal
            reviewed: list[str] = []
            release_a = asyncio.Event()

            async def review(run_id: str) -> JudgeVerdict:
                reviewed.append(run_id)
                if run_id == reviewed[0]:
                    await release_a.wait()
                verdict = JudgeVerdict(
                    run_id=run_id, reviewed_at=datetime.now(UTC), severity=VerdictSeverity.ok,
                    findings=[], action_taken=VerdictAction.none,
                )
                await journal.add_verdict(verdict)
                return verdict

            async def a_is_in_review() -> bool:
                return bool(reviewed)

            with patch.object(harness.judge, 'review_run', new=AsyncMock(side_effect=review)):
                run_a = await harness.run_full_cycle('test-project', 'test')
                await wait_until(a_is_in_review)
                assert reviewed == [run_a.id]

                cap_all_but_reserve(gate)
                run_b = await harness.run_full_cycle('test-project', 'test')
                for _ in range(10):
                    await asyncio.sleep(0)

                assert (run_b.id, 'test-project') in await journal.get_pending_judge_runs()
                assert reviewed == [run_a.id]
                assert harness.judge.is_halted('test-project') is False

                await buffer_events(harness, 3)
                reserve_clock.advance(RELEASE_AT - PROTECTED + timedelta(minutes=1))
                await drive_one_project_loop_cycle(harness)
                [run_c] = [
                    run.id for run in await journal.get_recent_runs('test-project')
                    if run.id not in {run_a.id, run_b.id}
                ]

                async def b_and_c_reviewed() -> bool:
                    return {run_b.id, run_c} <= set(reviewed)

                await wait_until(b_and_c_reviewed)
                assert Counter(reviewed) == {run_a.id: 1, run_b.id: 1, run_c: 1}

                release_a.set()

                async def nothing_pending() -> bool:
                    return await journal.get_pending_judge_runs() == []

                await wait_until(nothing_pending)
            assert Counter(reviewed) == {run_a.id: 1, run_b.id: 1, run_c: 1}
