"""Tests for Harness._cascade_unblock_member — auto-flip cascade-resolved L1 tasks blocked→pending.

Acceptance criteria:
  1. Cascade-resolved L1 member with task status 'blocked' → set_task_status('pending'), INFO cites L2 id.
  2. Member status 'done' → set_task_status NOT called (DEBUG-skipped; terminal members need no flip).
  3. Member status 'deferred' → set_task_status NOT called (carve-out), DEBUG logged.
  4. Member status 'in-progress' → set_task_status NOT called (carve-out), DEBUG logged.
  5. Dismissed cascade (dismiss=True) → members have status='dismissed' → no flip.
  6. Direct/steward-resolved L1 (resolved_by='steward') → no flip (guard: l2-cascade prefix).
  7. Mixed-status cascade: blocked member flipped, done member DEBUG-skipped, deferred member skipped.
  8. Wake events still fire for every member (regression: synchronous wake unaffected by async flip).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.harness import Harness

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def harness(tmp_path: Path, mock_orch_config) -> Harness:
    """Harness with mocked internals for cascade-unblock unit testing."""
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    # Replace scheduler with async mocks
    h.scheduler = MagicMock()
    h.scheduler.get_status = AsyncMock(return_value='blocked')
    h.scheduler.set_task_status = AsyncMock()

    # _merge_worker stays None — unhalt branch is skipped in all tests here
    return h


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_l1_esc(
    task_id: str = '3438',
    status: str = 'resolved',
    resolved_by: str = 'l2-cascade:esc-4000-39',
    level: int = 1,
) -> Escalation:
    return Escalation(
        id=f'esc-{task_id}-1',
        task_id=task_id,
        agent_role='workflow',
        severity='blocking',
        category='infra_issue',
        summary='x',
        level=level,
        status=status,
        resolved_by=resolved_by,
    )


# ---------------------------------------------------------------------------
# Unit tests — direct _on_escalation_resolved calls
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCascadeUnblockUnit:
    """Per-member unit cases that call _on_escalation_resolved directly."""

    async def test_criterion_1_blocked_task_flipped_to_pending(
        self, harness: Harness, caplog
    ):
        """[Criterion 1] blocked member → set_task_status('pending'), INFO cites L2 id."""
        esc = _make_l1_esc(task_id='3438', resolved_by='l2-cascade:esc-4000-39')
        harness._escalation_events['3438'] = asyncio.Event()
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        with caplog.at_level(logging.INFO):
            harness._on_escalation_resolved(esc)
            await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_awaited_once_with('3438', 'pending')  # type: ignore[attr-defined]
        assert any('esc-4000-39' in r.message for r in caplog.records), (
            "Expected INFO record citing L2 id 'esc-4000-39'"
        )

    async def test_criterion_2_done_task_not_flipped(
        self, harness: Harness, caplog
    ):
        """[Criterion 2] member status 'done' → set_task_status NOT called, DEBUG logged.

        A member completing normally while its L2 cluster is still pending is
        an expected, common outcome. Attempting a write just to produce a
        WARNING (relying on the server gate to reject it) generates noise for
        a routine case. Skip it silently with DEBUG instead.
        """
        esc = _make_l1_esc(task_id='3438', resolved_by='l2-cascade:esc-4000-39')
        harness.scheduler.get_status = AsyncMock(return_value='done')

        with caplog.at_level(logging.DEBUG):
            harness._on_escalation_resolved(esc)
            await asyncio.gather(*list(harness._background_tasks))

        # No attempt made
        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        # A DEBUG record was logged (not blocked; skipping)
        assert any(r.levelno == logging.DEBUG for r in caplog.records), (
            "Expected a DEBUG record for done (not-blocked) carve-out"
        )

    async def test_criterion_3_deferred_task_not_flipped(
        self, harness: Harness, caplog
    ):
        """[Criterion 3] member status 'deferred' → set_task_status NOT called, DEBUG logged."""
        esc = _make_l1_esc(task_id='3438', resolved_by='l2-cascade:esc-4000-39')
        harness.scheduler.get_status = AsyncMock(return_value='deferred')

        with caplog.at_level(logging.DEBUG):
            harness._on_escalation_resolved(esc)
            await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        assert any(r.levelno == logging.DEBUG for r in caplog.records), (
            "Expected a DEBUG record for deferred carve-out"
        )

    async def test_criterion_4_in_progress_task_not_flipped(
        self, harness: Harness, caplog
    ):
        """[Criterion 4] member status 'in-progress' → set_task_status NOT called, DEBUG logged."""
        esc = _make_l1_esc(task_id='3438', resolved_by='l2-cascade:esc-4000-39')
        harness.scheduler.get_status = AsyncMock(return_value='in-progress')

        with caplog.at_level(logging.DEBUG):
            harness._on_escalation_resolved(esc)
            await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        assert any(r.levelno == logging.DEBUG for r in caplog.records), (
            "Expected a DEBUG record for in-progress carve-out"
        )

    async def test_criterion_6_steward_resolved_not_flipped(
        self, harness: Harness, caplog
    ):
        """[Criterion 6] Direct/steward resolve (resolved_by='steward') → no flip."""
        esc = _make_l1_esc(task_id='9', status='resolved', resolved_by='steward')
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_merge_deferred_task_not_flipped(
        self, harness: Harness, caplog
    ):
        """[Review fix] member status 'merge-deferred' → set_task_status NOT called, DEBUG logged.

        'merge-deferred' is a LIVE non-terminal holding status for atomic-train members
        (workflow.py:475). The server terminal-exit gate does NOT protect it, so a
        set_task_status('pending') call would SUCCEED and clobber the deliberate holding
        state. The allowlist carve-out must exclude it.
        """
        esc = _make_l1_esc(task_id='3438', resolved_by='l2-cascade:esc-4000-39')
        harness.scheduler.get_status = AsyncMock(return_value='merge-deferred')

        with caplog.at_level(logging.DEBUG):
            harness._on_escalation_resolved(esc)
            await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        assert any(r.levelno == logging.DEBUG and 'merge-deferred' in r.message for r in caplog.records), (
            "Expected a DEBUG record mentioning 'merge-deferred'"
        )

    async def test_unknown_nonterminal_status_not_flipped(
        self, harness: Harness, caplog
    ):
        """[Review fix] novel non-terminal status 'review' → set_task_status NOT called.

        Pins allowlist semantics: any status not in {blocked} ∪ TERMINAL_STATUSES is
        skipped, future-proofing against newly introduced statuses.
        """
        esc = _make_l1_esc(task_id='3438', resolved_by='l2-cascade:esc-4000-39')
        harness.scheduler.get_status = AsyncMock(return_value='review')

        with caplog.at_level(logging.DEBUG):
            harness._on_escalation_resolved(esc)
            await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        assert any(r.levelno == logging.DEBUG for r in caplog.records), (
            "Expected a DEBUG record for unknown non-terminal status carve-out"
        )


# ---------------------------------------------------------------------------
# Integration tests — real EscalationQueue.resolve()
# ---------------------------------------------------------------------------

def _make_l1_in_queue(
    queue: EscalationQueue, esc_id: str, task_id: str = 'task-1'
) -> Escalation:
    esc = Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role='steward',
        severity='blocking',
        category='design_concern',
        summary='L1 test escalation',
        level=1,
    )
    queue.submit(esc)
    return esc


def _make_l2_in_queue(
    queue: EscalationQueue, l2_id: str, member_ids: list[str]
) -> Escalation:
    esc = Escalation(
        id=l2_id,
        task_id='task-cluster',
        agent_role='escalation-watcher-auto',
        severity='blocking',
        category='design_concern',
        summary='L2 cluster',
        level=2,
        root_cause='shared root cause',
        members=list(member_ids),
    )
    queue.submit(esc)
    return esc


@pytest.mark.asyncio
class TestCascadeUnblockIntegration:
    """Integration tests using a real EscalationQueue to exercise the full cascade path."""

    async def test_criterion_5_dismissed_cascade_no_flip(
        self, harness: Harness, tmp_path: Path
    ):
        """[Criterion 5] Dismissed cascade (dismiss=True) → members dismissed, no flip."""
        queue = EscalationQueue(tmp_path / 'esc')
        l1 = _make_l1_in_queue(queue, 'esc-l1-1', 'task-501')
        l2 = _make_l2_in_queue(queue, 'esc-l2-1', [l1.id])

        # Pre-seed wake event for the member
        harness._escalation_events['task-501'] = asyncio.Event()
        queue.set_resolve_callback(harness._on_escalation_resolved)

        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        queue.resolve(l2.id, 'dismissed resolution', dismiss=True)
        await asyncio.gather(*list(harness._background_tasks))

        # Member has status='dismissed', so the helper should NOT flip
        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_criterion_7_mixed_status_cascade(
        self, harness: Harness, tmp_path: Path, caplog
    ):
        """[Criterion 7] 3-member cascade: blocked→flipped, done→DEBUG-skipped, deferred→skipped."""
        queue = EscalationQueue(tmp_path / 'esc')
        l1_blocked = _make_l1_in_queue(queue, 'esc-l1-blocked', 'task-blocked')
        l1_done = _make_l1_in_queue(queue, 'esc-l1-done', 'task-done')
        l1_deferred = _make_l1_in_queue(queue, 'esc-l1-deferred', 'task-deferred')
        l2 = _make_l2_in_queue(
            queue, 'esc-l2-mixed',
            [l1_blocked.id, l1_done.id, l1_deferred.id],
        )

        queue.set_resolve_callback(harness._on_escalation_resolved)

        def _get_status_side_effect(task_id: str):
            mapping = {
                'task-blocked': 'blocked',
                'task-done': 'done',
                'task-deferred': 'deferred',
                # L2's own task_id — should be skipped (level==2 guard, not level==1)
                'task-cluster': 'pending',
            }
            return mapping.get(task_id, 'blocked')

        harness.scheduler.get_status = AsyncMock(side_effect=_get_status_side_effect)

        with caplog.at_level(logging.DEBUG):
            queue.resolve(l2.id, 'root cause fixed', dismiss=False)
            await asyncio.gather(*list(harness._background_tasks))

        # blocked task was flipped
        harness.scheduler.set_task_status.assert_any_await('task-blocked', 'pending')  # type: ignore[attr-defined]
        # done task was NOT flipped (DEBUG-skipped; normal case, not an error)
        assert not any(
            call.args == ('task-done', 'pending')
            for call in harness.scheduler.set_task_status.await_args_list  # type: ignore[attr-defined]
        ), "done task should be DEBUG-skipped, not flipped"
        # deferred task was NOT flipped
        assert not any(
            call.args == ('task-deferred', 'pending')
            for call in harness.scheduler.set_task_status.await_args_list  # type: ignore[attr-defined]
        ), "deferred task should not be flipped"
        # L2's own task ('task-cluster') was NOT flipped — level==2 guard
        assert not any(
            call.args == ('task-cluster', 'pending')
            for call in harness.scheduler.set_task_status.await_args_list  # type: ignore[attr-defined]
        ), "L2's own task should not be flipped (level==2 guard)"

    async def test_criterion_8_wake_events_still_fire(
        self, harness: Harness, tmp_path: Path
    ):
        """[Criterion 8] Wake events fire synchronously for every member even with async flip."""
        queue = EscalationQueue(tmp_path / 'esc')
        l1_a = _make_l1_in_queue(queue, 'esc-l1-a', 'task-a')
        l1_b = _make_l1_in_queue(queue, 'esc-l1-b', 'task-b')
        l2 = _make_l2_in_queue(queue, 'esc-l2-wake', [l1_a.id, l1_b.id])

        # Pre-seed wake events for both members
        event_a = asyncio.Event()
        event_b = asyncio.Event()
        harness._escalation_events['task-a'] = event_a
        harness._escalation_events['task-b'] = event_b

        queue.set_resolve_callback(harness._on_escalation_resolved)
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        # queue.resolve is synchronous; wake events should be set immediately after it returns
        queue.resolve(l2.id, 'wake regression test', dismiss=False)

        # Wake events must be set SYNCHRONOUSLY — before any async gather
        assert event_a.is_set(), "Wake event for task-a must be set synchronously"
        assert event_b.is_set(), "Wake event for task-b must be set synchronously"

        # Clean up background tasks
        await asyncio.gather(*list(harness._background_tasks))


@pytest.mark.asyncio
class TestCascadeUnblockOffLoop:
    """The resolve callback may fire OFF the orchestrator loop (sync MCP
    resolve_issue runs on a FastMCP threadpool worker).  The old bare
    asyncio.create_task raised 'no running event loop' there, silently
    dropping the flip (2026-05-29 reify).  _schedule_coro_threadsafe must
    route it back onto the captured loop instead."""

    async def test_off_loop_resolve_still_flips_blocked_member(
        self, harness: Harness
    ):
        """[Regression] blocked member is flipped even when the callback runs off-loop."""
        # run() normally captures this; set directly since we skip startup.
        harness._loop = asyncio.get_running_loop()
        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        esc = _make_l1_esc(task_id='3438', resolved_by='l2-cascade:esc-4000-39')

        # Worker thread → no running loop in that thread (the failing condition).
        await asyncio.to_thread(harness._on_escalation_resolved, esc)

        # The flip is scheduled onto harness._loop; poll until it lands.
        deadline = asyncio.get_running_loop().time() + 1.0
        while harness.scheduler.set_task_status.await_count == 0:  # type: ignore[attr-defined]
            if asyncio.get_running_loop().time() >= deadline:
                break
            await asyncio.sleep(0.02)

        harness.scheduler.set_task_status.assert_awaited_once_with('3438', 'pending')  # type: ignore[attr-defined]
