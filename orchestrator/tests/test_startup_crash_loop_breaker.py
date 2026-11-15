"""The startup crash-loop breaker through the real ``Harness.run`` (plans/startup-hardening-prd.md §3.5).

Every start runs the real 1b and 1c0a over one seeded repo, so one runs.db,
one journal and one escalation queue carry over from start to start, as they
do across systemd restarts. The outcome is read only through public
artifacts: the runs table, the journal file, the escalation queue and the
event store.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _startup_harness import journal_merge_request, merge_lane_harness_factory
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.project_layout import runs_db, startup_failures_json

from orchestrator.event_store import EventType
from orchestrator.harness import Harness
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

        healthy = await _healthy_dry_run(make_harness)

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

        assert _runs(h) == [(None, 'startup_crash:after-1c1h')] * CRASH_LOOP_THRESHOLD

        healthy = make_harness()
        leftover_reaper = AsyncMock()
        healthy._run_leftover_verify_scope_reaper_pass = leftover_reaper
        await healthy.run(prd_path=None, dry_run=True)

        leftover_reaper.assert_awaited_once()


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
