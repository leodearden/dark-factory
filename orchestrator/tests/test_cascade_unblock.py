"""Tests for Harness._cascade_unblock_member — auto-flip cascade-resolved L1 tasks blocked→pending.

Acceptance criteria:
  1. Cascade-resolved L1 member with task status 'blocked' → set_task_status('pending'), INFO cites L2 id.
  2. Member status 'done' → set_task_status NOT called (DEBUG-skipped; terminal members need no flip).
  3. Member status 'deferred' → set_task_status NOT called (carve-out), DEBUG logged.
  4. Member status 'in-progress' → CLAIMANT-LIVENESS decides, not the status:
     4a. a LIVE claimant (fresh heartbeat) → set_task_status NOT called, DEBUG logged
         (the running workflow owns its own re-pend, woken by event.set()).
     4b. NO live claimant (stranded) → set_task_status('pending').
     Re-anchored by task 3540 / PRD `plans/task-escalation-state-graph-prd.md`
     D8 (spec E9), which replaced the old blanket in-progress carve-out —
     it stranded any task whose workflow died between filing the escalation
     and exiting, leaving the row in-progress with nobody heartbeating it and
     its escalation now closed.
  5. Dismissed cascade (dismiss=True) → members have status='dismissed' → no flip.
  6. Direct/steward-resolved L1 (resolved_by='steward') → no flip (guard: l2-cascade prefix).
  7. Mixed-status cascade: blocked member flipped, stranded in-progress member
     flipped (task 3540), done member DEBUG-skipped, deferred member skipped.
  8. Wake events still fire for every member (regression: synchronous wake unaffected by async flip).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
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

    # The claimant-liveness fork (task 3540) reads
    # ``config.claimant_liveness_ttl_secs`` into ``timedelta(seconds=...)``.
    # ``mock_orch_config`` is spec_set but supplies no numeric value for it, and
    # ``timedelta(seconds=<MagicMock>)`` raises TypeError.  Pin the production
    # default.
    h.config.claimant_liveness_ttl_secs = 300.0

    # Replace scheduler with async mocks
    h.scheduler = MagicMock()
    h.scheduler.get_status = AsyncMock(return_value='blocked')
    h.scheduler.set_task_status = AsyncMock()
    # _check_reblock_guard reads fresh metadata and persists the counter; the
    # claimant-liveness gate and the write-time corroborating read both read
    # ``status``/``claimant_run_id``/``heartbeat_at`` off this same row, so it
    # must be a FULL row rather than the old ``{'id': 'x', 'metadata': {}}``
    # stub (which has no status at all).
    h.scheduler.get_task = AsyncMock(return_value=_row('blocked'))
    h.scheduler.update_task = AsyncMock(return_value=True)
    # THE trap (task 3540): a bare MagicMock attribute returns a TRUTHY child
    # mock, so every row would read as having a LIVE claimant and EVERY flip
    # would be silently skipped — a green suite asserting nothing.  Must be an
    # explicit False.
    h.scheduler.is_actively_held = MagicMock(return_value=False)

    # _merge_worker stays None — unhalt branch is skipped in all tests here
    return h


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The claimant-liveness TTL these fixtures pin onto the config knob (task 3540).
# Mirrors ``OrchestratorConfig.claimant_liveness_ttl_secs``'s production default.
_TTL_SECS: float = 300.0

# A well-formed claimant identity in ``compose_claimant_run_id`` shape.
_CLAIMANT: str = 'run-abc/sess-def/pid=4242'


def _row(
    status: str,
    *,
    task_id: str = 'x',
    claimant: str | None = None,
    heartbeat: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Build a fused-memory task row for the claimant-liveness oracle.

    Deliberately defaults to a STRANDED row (no claimant, no heartbeat) so a
    test that says nothing about liveness still exercises the flip — the
    common production shape for the orphan re-pend this suite covers.

    ``heartbeat`` is a symbolic age derived from :data:`_TTL_SECS` rather than
    a magic literal, so a TTL change cannot silently invert a fixture:
    ``'fresh'`` → now (live, given a claimant), ``'stale'`` → outside the TTL,
    ``None`` → absent.  Structural twin of ``test_resume_claimant_liveness._row``
    — kept local rather than imported so the two suites stay independently
    collectable.
    """
    now = datetime.now(UTC)
    if heartbeat == 'fresh':
        heartbeat_at: str | None = now.isoformat()
    elif heartbeat == 'stale':
        heartbeat_at = (now - timedelta(seconds=2 * _TTL_SECS)).isoformat()
    elif heartbeat is None:
        heartbeat_at = None
    else:  # pragma: no cover — fixture misuse
        raise ValueError(f'unknown heartbeat age {heartbeat!r}')

    return {
        'id': task_id,
        'status': status,
        'claimant_run_id': claimant,
        'heartbeat_at': heartbeat_at,
        'metadata': dict(metadata or {}),
    }


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

    async def test_criterion_4a_in_progress_with_live_claimant_not_flipped(
        self, harness: Harness, caplog
    ):
        """[Criterion 4a] in-progress WITH a live claimant → no flip, DEBUG logged.

        Re-anchored by task 3540 (PRD D8, spec E9).  The invariant this half
        preserves is the one the old blanket carve-out was really protecting:
        a workflow that is still running owns its own re-pend (it was woken
        synchronously by ``event.set()``), so flipping here would race it.
        Liveness — not the status string — is what establishes that.
        """
        esc = _make_l1_esc(task_id='3438', resolved_by='l2-cascade:esc-4000-39')
        harness.scheduler.get_status = AsyncMock(return_value='in-progress')
        harness.scheduler.get_task = AsyncMock(
            return_value=_row('in-progress', claimant=_CLAIMANT, heartbeat='fresh')
        )

        with caplog.at_level(logging.DEBUG):
            harness._on_escalation_resolved(esc)
            await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        assert any(
            r.levelno == logging.DEBUG and 'live claimant' in r.getMessage()
            for r in caplog.records
        ), "Expected a DEBUG record naming the live claimant as the reason for the skip"

    async def test_criterion_4b_stranded_in_progress_is_repended(
        self, harness: Harness
    ):
        """[Criterion 4b] in-progress with NO live claimant → flipped to 'pending'.

        The codification task 3540 overturns.  The old carve-out skipped this
        row on its status alone, which left every task whose workflow died
        between filing the escalation and exiting sitting in-progress with
        nobody heartbeating it and its escalation now closed — nothing left to
        advance it.
        """
        esc = _make_l1_esc(task_id='3438', resolved_by='l2-cascade:esc-4000-39')
        harness.scheduler.get_status = AsyncMock(return_value='in-progress')
        harness.scheduler.get_task = AsyncMock(
            return_value=_row('in-progress', claimant=None, heartbeat=None)
        )

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_awaited_once_with('3438', 'pending')  # type: ignore[attr-defined]

    async def test_criterion_6_active_workflow_owns_repend(
        self, harness: Harness, caplog
    ):
        """[Criterion 6, reframed] Direct resolve WITH an active workflow → no flip.

        Originally this pinned the ``l2-cascade:`` prefix guard (a
        steward-resolved L1 did not flip).  Fix #1a now flips an orphaned
        direct-resolve, so the real invariant is: when a workflow slot is
        active for the task (``task_id in _escalation_events``), that workflow
        owns its own re-pend (woken by the synchronous ``event.set()``), and
        ``_on_escalation_resolved`` must NOT also flip — otherwise it races the
        workflow.  Registering the active-workflow event must suppress the flip.
        """
        esc = _make_l1_esc(task_id='9', status='resolved', resolved_by='steward')
        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        # Active workflow owns the re-pend.
        harness._escalation_events['9'] = asyncio.Event()

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        # The active workflow was still woken synchronously.
        assert harness._escalation_events['9'].is_set()


@pytest.mark.asyncio
class TestDirectResolveOrphanUnblock:
    """Fix #1a — a directly-resolved (non-cascade) orphan L1 with NO active
    workflow re-pends its blocked task.  This is the precise inverse of the
    root cause: ``escalation-watcher-auto`` resolved the L1 directly, no
    workflow owned the re-pend, and the task stayed stranded `blocked` (3576
    incident, 2026-05-29)."""

    async def test_direct_resolve_no_active_workflow_flips_blocked(
        self, harness: Harness, caplog
    ):
        """Direct resolve (escalation-watcher-auto), level 1, status resolved,
        NO active workflow, task blocked → flips blocked→pending."""
        esc = _make_l1_esc(
            task_id='3576', status='resolved',
            resolved_by='escalation-watcher-auto',
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        # No _escalation_events['3576'] → orphaned, no active workflow.

        with caplog.at_level(logging.INFO):
            harness._on_escalation_resolved(esc)
            await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_awaited_once_with('3576', 'pending')  # type: ignore[attr-defined]
        assert any('escalation-watcher-auto' in r.message for r in caplog.records), (
            "Expected an INFO/label record citing the direct resolver"
        )

    async def test_direct_dismiss_no_active_workflow_no_flip(
        self, harness: Harness
    ):
        """Direct DISMISS (status='dismissed') → no flip.

        A dismissed escalation means 'abandon the task' — it must stay blocked.
        """
        esc = _make_l1_esc(
            task_id='3576', status='dismissed',
            resolved_by='escalation-watcher-auto',
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_direct_resolve_level0_orphan_flips(self, harness: Harness):
        """A direct-resolved L0 with NO live workflow DOES flip.

        Re-anchored by task 3540 (PRD D8, spec E9).  This test previously
        codified "only L1 is the blocking-handoff tier that re-pend keys off" —
        the level assumption the task overturns.  The premise behind it was
        "every L0 has a live workflow waiting on ``event.set()``", which is
        false for any workflow that crashed between filing the escalation and
        exiting: its ``_escalation_events`` entry has already been popped, so
        the wake sets nothing and the ``level >= 1`` gate then dropped the
        re-pend silently.  Liveness, not level, is the discriminator — see
        ``test_direct_resolve_level0_with_active_workflow_no_flip`` for the
        other half.
        """
        esc = _make_l1_esc(
            task_id='3576', status='resolved',
            resolved_by='escalation-watcher-auto', level=0,
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        # No _escalation_events['3576'] and no claimant → orphaned.

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_awaited_once_with('3576', 'pending')  # type: ignore[attr-defined]

    async def test_direct_resolve_level0_with_active_workflow_no_flip(
        self, harness: Harness
    ):
        """The preserved half of the old L0 rule: a LIVE L0 workflow still owns
        its own re-pend and must not be raced.

        The local ``_escalation_events`` guard is what suppresses this — it
        proves a live workflow in THIS process was just woken synchronously.
        """
        esc = _make_l1_esc(
            task_id='3576', status='resolved',
            resolved_by='escalation-watcher-auto', level=0,
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        harness._escalation_events['3576'] = asyncio.Event()

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        assert harness._escalation_events['3576'].is_set()

    async def test_direct_resolve_not_blocked_no_flip(self, harness: Harness):
        """Direct resolve but task already moved off `blocked` (e.g. done) →
        _cascade_unblock_member's status recheck leaves it alone."""
        esc = _make_l1_esc(
            task_id='3576', status='resolved',
            resolved_by='escalation-watcher-auto',
        )
        harness.scheduler.get_status = AsyncMock(return_value='done')

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_scheduler_pause_sentinel_excluded(self, harness: Harness):
        """The synthetic __scheduler__ pause sentinel is NOT a real task — its
        resolution has dedicated auto-resume handling, so the orphan-unblock
        path must skip it (no get_status flip, no stray background task)."""
        esc = _make_l1_esc(
            task_id=harness._SCHEDULER_PAUSE_SENTINEL, status='resolved',
            resolved_by='interactive',
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        # Not paused → the dedicated auto-resume block is also a no-op, so the
        # ONLY thing that could schedule a background task is the orphan-unblock
        # path — which must skip the sentinel.  harness.scheduler is a MagicMock
        # (fixture); narrow it so the is_paused mock attribute is settable — the
        # real Scheduler.is_paused is a read-only @property.
        assert isinstance(harness.scheduler, MagicMock)
        harness.scheduler.is_paused = False

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        assert harness._background_tasks == set(), (
            'orphan-unblock must not schedule any coro for the pause sentinel'
        )

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
        """[Criterion 7] 4-member cascade: blocked→flipped, stranded in-progress→
        flipped, done→DEBUG-skipped, deferred→skipped.

        Re-anchored by task 3540 (PRD D8, spec E9): the in-progress member is
        new here and now flips, because ``get_task`` reports it with no
        claimant.  ``get_task`` is given a per-``task_id`` side_effect matching
        ``get_status``'s so status and claimant AGREE per member — production
        reads the two independently (see ``_cascade_unblock_member``'s
        "Efficiency note"), and a fixed shared row would silently describe a
        task that cannot exist.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        l1_blocked = _make_l1_in_queue(queue, 'esc-l1-blocked', 'task-blocked')
        l1_inprog = _make_l1_in_queue(queue, 'esc-l1-inprog', 'task-in-progress')
        l1_done = _make_l1_in_queue(queue, 'esc-l1-done', 'task-done')
        l1_deferred = _make_l1_in_queue(queue, 'esc-l1-deferred', 'task-deferred')
        l2 = _make_l2_in_queue(
            queue, 'esc-l2-mixed',
            [l1_blocked.id, l1_inprog.id, l1_done.id, l1_deferred.id],
        )

        queue.set_resolve_callback(harness._on_escalation_resolved)

        _statuses = {
            'task-blocked': 'blocked',
            # Stranded: no live claimant (see _row_side_effect) → re-pends.
            'task-in-progress': 'in-progress',
            'task-done': 'done',
            'task-deferred': 'deferred',
            # L2's own task_id — should be skipped (level==2 guard, not level==1)
            'task-cluster': 'pending',
        }

        def _get_status_side_effect(task_id: str):
            return _statuses.get(task_id, 'blocked')

        def _get_task_side_effect(task_id: str):
            # Every member is claimant-less, so ONLY the status allow-list can
            # be what withholds a flip in this cascade.
            return _row(
                _statuses.get(task_id, 'blocked'),
                task_id=task_id, claimant=None, heartbeat=None,
            )

        harness.scheduler.get_status = AsyncMock(side_effect=_get_status_side_effect)
        harness.scheduler.get_task = AsyncMock(side_effect=_get_task_side_effect)

        with caplog.at_level(logging.DEBUG):
            queue.resolve(l2.id, 'root cause fixed', dismiss=False)
            await asyncio.gather(*list(harness._background_tasks))

        # blocked task was flipped
        harness.scheduler.set_task_status.assert_any_await('task-blocked', 'pending')  # type: ignore[attr-defined]
        # stranded in-progress task was ALSO flipped (task 3540)
        harness.scheduler.set_task_status.assert_any_await('task-in-progress', 'pending')  # type: ignore[attr-defined]
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
