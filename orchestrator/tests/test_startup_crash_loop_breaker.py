"""The startup crash-loop breaker through the real ``Harness.run`` (plans/startup-hardening-prd.md §3.5).

Every start runs the real 1b and 1c0a over one seeded repo, so one runs.db,
one journal and one escalation queue carry over from start to start, as they
do across systemd restarts. The outcome is read only through public
artifacts: the runs table, the journal file, the escalation queue and the
event store.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _startup_harness import journal_merge_request, merge_lane_harness_factory
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.project_layout import runs_db, startup_failures_json

from orchestrator.event_store import EventType
from orchestrator.harness import Harness
from orchestrator.merge_lane.liveness import MergeLivenessConfigError
from orchestrator.run_store import RunStore
from orchestrator.startup_steps import CRASH_LOOP_THRESHOLD, StartupFailureStreak

SENTINEL = 'orchestrator-startup-crash-loop'


@pytest.fixture
def make_harness(tmp_path: Path) -> Callable[[], Harness]:
    return merge_lane_harness_factory(tmp_path)


def _runs(h: Harness) -> list[tuple[str | None, str | None]]:
    conn = sqlite3.connect(runs_db(h.config.project_root))
    try:
        return conn.execute(
            'SELECT startup_completed_at, exit_reason FROM runs ORDER BY rowid',
        ).fetchall()
    finally:
        conn.close()


def _streak(h: Harness) -> StartupFailureStreak | None:
    path = startup_failures_json(h.config.project_root)
    if not path.exists():
        return None
    return StartupFailureStreak.model_validate_json(path.read_text())


def _pages(tmp_path: Path) -> list[Escalation]:
    return EscalationQueue(tmp_path / 'escalations').get_by_task(SENTINEL, status='pending', level=2)


def _every_record_on_the_sentinel(tmp_path: Path) -> list[Escalation]:
    return EscalationQueue(tmp_path / 'escalations').get_by_task(SENTINEL)


async def _crash_at_2e(make: Callable[[], Harness]) -> Harness:
    h = make()
    h._reap_orphan_worktrees = AsyncMock(side_effect=SystemExit('injected 2e'))
    with pytest.raises(SystemExit):
        await h.run(prd_path=None)
    return h


async def _healthy_dry_run(make: Callable[[], Harness]) -> Harness:
    h = make()
    await h.run(prd_path=None, dry_run=True)
    return h


@pytest.mark.asyncio
class TestThreeCrashesThenQuarantine:
    async def test_three_crashes_at_2e_are_journaled_and_resolve_no_merge(
        self, make_harness, tmp_path,
    ) -> None:
        first = make_harness()
        await journal_merge_request(first, '41')
        crashed = [await _crash_at_2e(make_harness) for _ in range(CRASH_LOOP_THRESHOLD)]

        assert _runs(crashed[-1]) == [(None, 'startup_crash:2e')] * CRASH_LOOP_THRESHOLD
        streak = _streak(crashed[-1])
        assert streak is not None
        assert (streak.step, streak.exception_type, streak.consecutive) == (
            '2e', 'SystemExit', CRASH_LOOP_THRESHOLD,
        )
        assert _pages(tmp_path) == []
        for h in crashed:
            assert h.event_store is not None
            assert h.event_store.fetch_events_by_type(EventType.merge_finalized) == []

    async def test_the_next_start_skips_2e_pages_once_completes_and_clears_the_journal(
        self, make_harness, tmp_path,
    ) -> None:
        for _ in range(CRASH_LOOP_THRESHOLD):
            await _crash_at_2e(make_harness)

        h = make_harness()
        reaper = AsyncMock()
        h._reap_orphan_worktrees = reaper
        await h.run(prd_path=None, dry_run=True)

        reaper.assert_not_awaited()
        h.scheduler.finish_startup.assert_called_once()
        startup_completed_at, exit_reason = _runs(h)[-1]
        assert startup_completed_at is not None
        assert exit_reason is None
        [page] = _pages(tmp_path)
        assert '2e' in page.summary
        assert 'SystemExit' in page.summary
        assert not startup_failures_json(h.config.project_root).exists()

    async def test_a_later_healthy_start_leaves_the_one_page_alone(
        self, make_harness, tmp_path,
    ) -> None:
        for _ in range(CRASH_LOOP_THRESHOLD):
            await _crash_at_2e(make_harness)
        await _healthy_dry_run(make_harness)

        await _healthy_dry_run(make_harness)

        [page] = _every_record_on_the_sentinel(tmp_path)
        assert page.status == 'pending'
        assert page.dedupe_count == 0


async def _refused_at_1b(make: Callable[[], Harness]) -> Harness:
    """A start the ratified fatal refusal stops inside 1b, after the escalation server came up."""
    h = make()
    h._start_merge_worker = AsyncMock(side_effect=MergeLivenessConfigError('over budget'))
    with pytest.raises(MergeLivenessConfigError):
        await h.run(prd_path=None)
    return h


@pytest.mark.asyncio
class TestFatalStepAtThreshold:
    async def test_each_refused_start_counts_once(self, make_harness, tmp_path) -> None:
        for _ in range(CRASH_LOOP_THRESHOLD):
            h = await _refused_at_1b(make_harness)

        streak = _streak(h)
        assert streak is not None
        assert (streak.step, streak.consecutive) == ('1b', CRASH_LOOP_THRESHOLD)
        assert _pages(tmp_path) == []

    async def test_a_fatal_step_at_the_threshold_still_refuses_and_pages(
        self, make_harness, tmp_path,
    ) -> None:
        for _ in range(CRASH_LOOP_THRESHOLD):
            await _refused_at_1b(make_harness)

        h = await _refused_at_1b(make_harness)

        [page] = _pages(tmp_path)
        assert '1b' in page.summary
        assert 'MergeLivenessConfigError' in page.summary
        assert _runs(h)[-1] == (None, 'startup_crash:1b')

    async def test_a_further_refused_start_refreshes_the_one_page(
        self, make_harness, tmp_path,
    ) -> None:
        for _ in range(CRASH_LOOP_THRESHOLD + 1):
            await _refused_at_1b(make_harness)
        [filed] = _pages(tmp_path)

        await _refused_at_1b(make_harness)

        [refreshed] = _pages(tmp_path)
        assert refreshed.id == filed.id
        assert refreshed.dedupe_count == 1


@pytest.mark.asyncio
class TestSigtermIsNotAFailure:
    async def test_a_cancelled_start_is_recorded_as_cancelled_and_not_journaled(
        self, make_harness,
    ) -> None:
        h = make_harness()
        h._recover_crashed_tasks = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await h.run(prd_path=None)

        assert _runs(h) == [(None, 'cancelled')]
        assert not startup_failures_json(h.config.project_root).exists()


@pytest.mark.asyncio
class TestCrashOutsideAnyStep:
    async def test_an_mcp_start_crash_is_journaled_before_the_first_step_and_paged_later(
        self, make_harness, tmp_path,
    ) -> None:
        for _ in range(CRASH_LOOP_THRESHOLD):
            h = make_harness()
            h.mcp.start = AsyncMock(side_effect=RuntimeError('mcp bind'))
            with pytest.raises(RuntimeError, match='mcp bind'):
                await h.run(prd_path=None)

        assert _runs(h) == [(None, 'startup_crash:before-first-step')] * CRASH_LOOP_THRESHOLD
        streak = _streak(h)
        assert streak is not None
        assert streak.consecutive == CRASH_LOOP_THRESHOLD

        healthy = make_harness()
        await healthy.run(prd_path=None, dry_run=True)

        [page] = _pages(tmp_path)
        assert 'before-first-step' in page.summary
        assert 'RuntimeError' in page.summary
        healthy.scheduler.finish_startup.assert_called_once()
        assert not startup_failures_json(healthy.config.project_root).exists()

    async def test_a_task_tree_gate_crash_never_quarantines_the_step_before_it(
        self, make_harness,
    ) -> None:
        for _ in range(CRASH_LOOP_THRESHOLD):
            h = make_harness()
            h.scheduler.get_statuses = AsyncMock(return_value=({}, RuntimeError('fm down')))
            with pytest.raises(RuntimeError, match='fm down'):
                await h.run(prd_path=None)

        assert _runs(h) == [(None, 'startup_crash:after-1c1i')] * CRASH_LOOP_THRESHOLD

        healthy = make_harness()
        slice_weight = AsyncMock()
        healthy._ensure_project_slice_weight = slice_weight
        await healthy.run(prd_path=None, dry_run=True)

        slice_weight.assert_awaited_once()


@pytest.mark.asyncio
class TestHealthyStart:
    async def test_a_healthy_start_pages_nothing_and_records_startup_completion(
        self, make_harness, tmp_path,
    ) -> None:
        h = await _healthy_dry_run(make_harness)

        assert _every_record_on_the_sentinel(tmp_path) == []
        [(startup_completed_at, exit_reason)] = _runs(h)
        assert startup_completed_at is not None
        assert exit_reason is None


def _after_finish_startup(h: Harness, probe: Callable[[], None]) -> None:
    """Run ``probe`` once, inside ``h.run``, at the first task-tree read after startup finished."""
    probed = False
    finish_startup = MagicMock()
    h.scheduler.finish_startup = finish_startup

    def statuses() -> tuple[dict[str, str], None]:
        nonlocal probed
        if finish_startup.called and not probed:
            probed = True
            probe()
        return {'1': 'pending'}, None

    h.scheduler.get_statuses = AsyncMock(side_effect=statuses)


def _run_row(db: Path) -> tuple[str | None, str | None, str | None]:
    with closing(sqlite3.connect(db)) as conn:
        return conn.execute(
            'SELECT startup_completed_at, exit_reason, completed_at FROM runs '
            'ORDER BY rowid DESC LIMIT 1',
        ).fetchone()


@pytest.mark.asyncio
class TestStartupCompletionIsDurableBeforeExit:
    async def test_a_live_run_reads_startup_completed_before_run_returns(
        self, make_harness,
    ) -> None:
        h = make_harness()
        live_rows: list[tuple[str | None, str | None, str | None]] = []
        _after_finish_startup(h, lambda: live_rows.append(_run_row(runs_db(h.config.project_root))))

        await h.run(prd_path=None, dry_run=True)

        [(startup_completed_at, exit_reason, completed_at)] = live_rows
        assert startup_completed_at is not None
        assert exit_reason is None
        assert completed_at is None
        assert _run_row(runs_db(h.config.project_root))[0] == startup_completed_at

    async def test_a_run_sigkilled_after_startup_keeps_startup_completed_at(
        self, make_harness, tmp_path,
    ) -> None:
        h = make_harness()
        snapshot = tmp_path / 'sigkilled-runs.db'

        def copy_live_db() -> None:
            with (
                closing(sqlite3.connect(runs_db(h.config.project_root))) as live,
                closing(sqlite3.connect(snapshot)) as copy,
            ):
                live.backup(copy)

        _after_finish_startup(h, copy_live_db)

        await h.run(prd_path=None, dry_run=True)

        startup_completed_at, exit_reason, completed_at = _run_row(snapshot)
        assert startup_completed_at is not None
        assert exit_reason is None
        assert completed_at is None

    async def test_a_failing_stamp_logs_a_warning_and_startup_still_completes(
        self, make_harness, caplog,
    ) -> None:
        h = make_harness()

        with (
            patch.object(
                RunStore, 'mark_startup_completed',
                side_effect=sqlite3.OperationalError('database is locked'),
            ),
            caplog.at_level(logging.WARNING, logger='orchestrator.harness'),
        ):
            await h.run(prd_path=None, dry_run=True)

        h.scheduler.finish_startup.assert_called_once()
        db = runs_db(h.config.project_root)
        assert _run_row(db)[1] is None
        with closing(sqlite3.connect(db)) as conn:
            [(run_id,)] = conn.execute('SELECT run_id FROM runs').fetchall()
        assert any(
            record.levelno == logging.WARNING and run_id in record.getMessage()
            for record in caplog.records
        )
