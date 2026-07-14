"""ζ integration gate — B1/B2/B3-harness/B4/B6 + comp-2 harness face.

PRD ``plans/task-status-authority-prd.md`` §Boundary-test sketch. This
module realizes the escalation<->harness HARNESS-side rows of the 17-cell
boundary matrix: B1 (memberless born-at-L2 resume), B2 (esc-2073-15
stranded-in-progress revert), B3-harness (park's harness face, two-way with
the escalation suite's B3-server), B4 (abandon/close_only), B6 (infra-hold
first-class: resume-at-verify + reconcile-skip), and the harness face of
comp-2 (an unrecognised resolution_action -> ``effect_for`` returns None ->
no ``set_task_status`` call — the SAME table the escalation server rejects
on, proven from the other side of the seam).

Every drive goes through ``harness._on_escalation_resolved(esc)`` (a SYNC
call that schedules background coros into ``harness._background_tasks``)
followed by draining those coros, then asserts the scheduler's
``set_task_status`` target — the product's own read path on the harness
side (mirrors test_harness_action_dispatch.py / test_harness_infra_hold_repend.py).

Deliberate mock-assertion style, unlike the fused-memory/escalation ζ gate
modules: these cells assert via a ``MagicMock`` scheduler's
``set_task_status.assert_awaited_once_with(...)``, not a persisted row read
back through a real store — driving ``_on_escalation_resolved`` against a
real ``Scheduler`` is impractical here (it would need a live dispatch loop
and backing store this module has no other reason to stand up). This is the
intentionally-mocked "harness face" that complements the two-way
store/server read-path rows in the fused-memory and escalation suites:
together they cover both sides of the escalation<->harness seam, even
though this side confirms only "the harness issued the call", not "a row
landed in the intended state".
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.action_effects import ACTION_EFFECTS, ANY, effect_for
from escalation.models import Escalation

from orchestrator.harness import Harness

# B6 exercises is_infra_held indirectly (through the harness's own
# _cascade_unblock_member / _reconcile_stranded_in_progress guards); kept as
# a direct import to document the guard under test.
from orchestrator.task_status import is_infra_held  # noqa: F401

_BASE_SHA = 'a' * 40  # branch_base_sha in metadata — non-degenerate-branch fixtures
_TIP_SHA = 'b' * 40   # tip SHA different from base -> non-degenerate


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config) -> Harness:
    """Harness with mocked internals for status-authority-gate dispatch tests.

    Mirrors test_harness_action_dispatch.py:32-55 — McpLifecycle/
    OverrideStore/Scheduler/BriefingAssembler patched at construction, then
    the scheduler replaced with a MagicMock carrying AsyncMock
    get_status/set_task_status/get_task/update_task so
    ``_on_escalation_resolved``'s background coros can be driven and
    asserted through the scheduler's own call surface (the harness's
    "product read path" for a task-status write).
    """
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler.get_status = AsyncMock(return_value='blocked')
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.get_task = AsyncMock(return_value={'id': 'task', 'metadata': {}})
    h.scheduler.update_task = AsyncMock(return_value=True)

    # B2/B6 drive _revert_in_progress_if_no_live_claimant /
    # _reconcile_stranded_in_progress directly, which touch git_ops — NOT
    # mocked above (only McpLifecycle/OverrideStore/Scheduler/
    # BriefingAssembler are patched at construction). Mirrors
    # test_harness_infra_hold_repend.py:53-62: keep worktree_base real
    # (rooted under tmp_path) so a per-test worktree dir can be created, and
    # default cleanup_worktree/resolve_branch_sha so B1/B3/B4/comp2 (which
    # never touch git_ops) are unaffected.
    h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()

    def _fake_cleanup(worktree_path, tid):
        shutil.rmtree(worktree_path, ignore_errors=True)
    h.git_ops.cleanup_worktree = AsyncMock(side_effect=_fake_cleanup)
    h.git_ops.resolve_branch_sha = AsyncMock(return_value=_TIP_SHA)

    # _merge_worker stays None — unhalt branch skipped in all tests here.
    return h


def _make_esc(
    task_id: str = 'task-1',
    status: str = 'resolved',
    resolved_by: str = 'steward',
    level: int = 1,
    resolution_action: str | None = None,
) -> Escalation:
    """Mirrors test_harness_action_dispatch.py:58-76 — a generic resolved
    escalation for the B1/B3-harness/B4/comp-2 dispatch cells."""
    return Escalation(
        id=f'esc-{task_id}-1',
        task_id=task_id,
        agent_role='workflow',
        severity='blocking',
        category='infra_issue',
        summary='status-authority-gate dispatch test',
        level=level,
        status=status,
        resolved_by=resolved_by,
        resolution_action=resolution_action,
    )


def _make_infra_esc(
    task_id: str = '1883',
    status: str = 'resolved',
    level: int = 1,
    resolved_by: str | None = None,
) -> Escalation:
    """Mirrors test_harness_infra_hold_repend.py:204-221 — a minimal resolved
    infra_issue escalation for the B6 resume-at-verify cell."""
    return Escalation(
        id=f'esc-{task_id}-99',
        task_id=task_id,
        agent_role='workflow',
        severity='blocking',
        category='infra_issue',
        summary='ENOSPC during verify warm marker write',
        level=level,
        status=status,
        resolved_by=resolved_by,
    )


def _make_worktree(harness: Harness, tid: str, *, with_lock: bool = False) -> Path:
    """Create a minimal worktree directory for *tid*.

    Mirrors test_harness_infra_hold_repend.py:67-81. If ``with_lock`` is
    False (default), no plan.lock is created — simulates the no-live-
    claimant / orphan scenario that B2's stranded-revert sweep targets.
    """
    wt = harness.git_ops.worktree_base / tid
    (wt / '.task').mkdir(parents=True, exist_ok=True)
    if with_lock:
        (wt / '.task' / 'plan.lock').write_text(json.dumps({
            'session_id': f'{tid}-test',
            'locked_at': '2026-06-23T00:00:00Z',
            'owner_pid': os.getpid(),  # live PID → not reaped
        }))
    return wt


# ---------------------------------------------------------------------------
# B1 — memberless (orphan) born-at-L2 resume flips blocked->pending.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB1MemberlessBornAtL2Resume:
    """B1: a blocked task with a level=2 escalation resolved with
    resolution_action='resume', whose task_id is NOT registered in
    harness._escalation_events (no active workflow — an orphan, not an
    l2-cascade member) — flips blocked->pending via _cascade_unblock_member
    (harness.py:8949-8958)."""

    async def test_orphan_l2_resume_flips_blocked_to_pending(self, harness: Harness) -> None:
        task_id = 'zeta-b1-orphan'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='resume',
            status='resolved',
            resolved_by='interactive',
            level=2,
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        assert task_id not in harness._escalation_events, (
            'Precondition: task must be an orphan (no active workflow slot)'
        )

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            task_id, 'pending',
        )


# ---------------------------------------------------------------------------
# B3 (harness side) — park's harness face: park -> blocked, NEVER deferred.
# Two-way with escalation/tests/test_status_authority_gate.py's
# TestB3ServerParkKeepsL2Open (the escalation-side face of the same seam).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB3HarnessParkSetsBlocked:
    """B3, harness-side face: resolution_action='park' ->
    scheduler.set_task_status(task_id, <Table B's park target>) via
    _action_teardown_and_set_status (harness.py:8970-8985). Asserted against
    ``ACTION_EFFECTS[('park', ANY, ANY)].target_status`` rather than a
    hardcoded 'blocked' literal, so this test and the escalation-side
    ``TestB3ServerParkKeepsL2Open``
    (escalation/tests/test_status_authority_gate.py) stay coupled to the
    single source of truth — NEVER 'deferred'."""

    async def test_park_sets_blocked_not_deferred(self, harness: Harness) -> None:
        task_id = 'zeta-b3-park'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='park',
            status='pending',
            resolved_by='interactive',
            level=2,
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        harness.is_workflow_active = MagicMock(return_value=False)
        park_target = ACTION_EFFECTS[('park', ANY, ANY)].target_status

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            task_id, park_target,
        )
        written = {
            a.args[1] for a in harness.scheduler.set_task_status.await_args_list  # type: ignore[attr-defined]
        }
        assert 'deferred' not in written, f'park must never target deferred; wrote {written}'


# ---------------------------------------------------------------------------
# B4 — abandon -> cancelled; close_only -> no set_task_status call at all.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB4AbandonCancelledCloseOnlyNoOp:
    """B4: resolution_action='abandon' -> set_task_status(task_id,'cancelled');
    resolution_action='close_only' -> WORKFLOW_NONE early-return
    (harness.py:8915-8923) -> no set_task_status call at all."""

    async def test_abandon_sets_cancelled(self, harness: Harness) -> None:
        task_id = 'zeta-b4-abandon'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='abandon',
            status='dismissed',
            resolved_by='interactive',
            level=1,
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        harness.is_workflow_active = MagicMock(return_value=False)

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            task_id, 'cancelled',
        )

    async def test_close_only_no_set_task_status_call(self, harness: Harness) -> None:
        task_id = 'zeta-b4-close-only'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='close_only',
            status='dismissed',
            resolved_by='interactive',
            level=1,
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# comp-2, harness face — an unrecognised action -> effect_for None -> no write.
# The SAME table the escalation server rejects an unknown action on (B5).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestComp2HarnessFaceUnknownActionNoWrite:
    """comp-2 harness face: effect_for('bogus', ...) returns None
    (harness.py:8907-8913) -> warning + return, no set_task_status call — the
    SAME effect_for table the escalation server rejects an unknown action on
    (B5, escalation/tests/test_status_authority_gate.py)."""

    async def test_unknown_action_effect_none_no_set_task_status(self, harness: Harness) -> None:
        assert effect_for('bogus', 1, 'infra_issue') is None, (
            'Precondition: bogus must be unrecognised by the shared Table B'
        )
        task_id = 'zeta-comp2-bogus'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='bogus',
            status='resolved',
            resolved_by='interactive',
            level=1,
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# B2 (esc-2073-15 regression) — the accurate landed mechanism is the
# stranded-revert sweep _revert_in_progress_if_no_live_claimant, NOT the
# escalation-resume cascade (_cascade_unblock_member is blocked-only and
# DELIBERATELY carves out in-progress — test_cascade_unblock.py:143).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB2StrandedInProgressRevert:
    """B2: a stranded in-progress task with no live claimant is re-pended by
    the stranded-revert sweep (_revert_in_progress_if_no_live_claimant,
    harness.py:3719), NOT by the escalation-resume cascade
    (_cascade_unblock_member is blocked-only — a status='in-progress' task
    handed to it is a DEBUG-skipped no-op). An infra-held row is exempt.
    Mirrors test_harness_infra_hold_repend.py:88-195."""

    async def test_no_live_claimant_reverts_to_pending(self, harness: Harness) -> None:
        tid = 'zeta-b2-stranded'
        _make_worktree(harness, tid)  # no lock -> no live claimant
        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid, 'status': 'in-progress', 'metadata': {},
        })

        result = await harness._revert_in_progress_if_no_live_claimant(tid, mid_run=False)

        assert result == 'reverted', (
            f'Stranded in-progress task with no live claimant should be '
            f'reverted to pending via the stranded-revert sweep; got {result!r}'
        )
        harness.scheduler.set_task_status.assert_awaited_once_with(tid, 'pending')  # type: ignore[attr-defined]

    async def test_live_claimant_not_reverted(self, harness: Harness) -> None:
        """Contrast: a LIVE claimant (fresh plan.lock, live owner_pid) on a
        startup sweep (mid_run=False) is left intact — the stranded-revert
        sweep only reaps tasks with NO live claimant."""
        tid = 'zeta-b2-live'
        _make_worktree(harness, tid, with_lock=True)
        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid, 'status': 'in-progress', 'metadata': {},
        })

        result = await harness._revert_in_progress_if_no_live_claimant(tid, mid_run=False)

        assert result != 'reverted', (
            f'Task with a live claimant (fresh plan.lock) must not be '
            f'reverted; got {result!r}'
        )
        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_infra_held_stranded_row_exempt_from_revert(self, harness: Harness) -> None:
        """An infra-held row (status='infra-hold') on a non-degenerate branch
        is EXEMPT from the stranded-revert sweep even with no live claimant —
        the A1 guard fires before the lock-liveness classification."""
        tid = 'zeta-b2-infra-exempt'
        _make_worktree(harness, tid)  # no lock -> no live claimant
        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid,
            'status': 'infra-hold',
            'metadata': {'branch_base_sha': _BASE_SHA},
        })
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=_TIP_SHA)  # non-degenerate

        result = await harness._revert_in_progress_if_no_live_claimant(tid, mid_run=False)

        assert result != 'reverted', (
            f"An infra-held row must be exempt from the stranded-revert "
            f'sweep; got {result!r}'
        )
        for call in harness.scheduler.set_task_status.await_args_list:  # type: ignore[attr-defined]
            assert call.args[1] != 'pending', (
                f'set_task_status({tid!r}, "pending") was called despite '
                f"status='infra-hold': {call}"
            )

    async def test_cascade_unblock_member_does_not_flip_in_progress(
        self, harness: Harness, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Proves B2 is NOT served by the escalation cascade: a status=
        'in-progress' task routed through _on_escalation_resolved's
        l2-cascade path (-> _cascade_unblock_member) is a DEBUG-skipped
        no-op — _cascade_unblock_member is blocked-only and DELIBERATELY
        carves out in-progress (test_cascade_unblock.py:143,
        test_criterion_4_in_progress_task_not_flipped)."""
        task_id = 'zeta-b2-not-cascade'
        esc = _make_esc(
            task_id=task_id,
            resolution_action=None,  # legacy resolve -> maps to 'resume'
            status='resolved',
            resolved_by='l2-cascade:esc-parent-1',
            level=1,
        )
        harness.scheduler.get_status = AsyncMock(return_value='in-progress')

        with caplog.at_level(logging.DEBUG):
            harness._on_escalation_resolved(esc)
            await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        assert any(r.levelno == logging.DEBUG for r in caplog.records), (
            'Expected a DEBUG record for the in-progress carve-out'
        )


# ---------------------------------------------------------------------------
# B6 — infra-hold first-class: resolved infra_issue escalation on an
# infra-held task resumes-at-verify (in-progress), NEVER 'pending'; the
# reconcile stranded sweep skips an infra-held row; a non-infra blocked task
# still flips to pending (contrast).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB6InfraHoldFirstClass:
    """B6: infra-hold is a first-class status (task 2200/omega4). A resolved
    infra_issue escalation on an infra-held task resumes-at-verify
    (in-progress) via _cascade_unblock_member's A1 guard (harness.py:9402-
    9439), never re-pends to 'pending'; the reconcile stranded sweep skips
    an infra-held row entirely (is_infra_held guard); a non-infra blocked
    task still flips to pending. Mirrors
    test_harness_infra_hold_repend.py:224-424."""

    async def test_infra_hold_resolved_escalation_resumes_at_verify(
        self, harness: Harness,
    ) -> None:
        tid = 'zeta-b6-infra'
        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid,
            'status': 'infra-hold',
            'metadata': {'branch_base_sha': _BASE_SHA},
        })
        harness.scheduler.get_status = AsyncMock(return_value='infra-hold')
        esc = _make_infra_esc(task_id=tid)  # status='resolved' -> legacy 'resume'

        harness._escalation_events.pop(tid, None)  # orphan path

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_awaited_once_with(tid, 'in-progress')  # type: ignore[attr-defined]
        pending_calls = [
            c for c in harness.scheduler.set_task_status.await_args_list  # type: ignore[attr-defined]
            if c.args[1] == 'pending'
        ]
        assert not pending_calls, (
            f"set_task_status('pending') was called despite status='infra-hold'. "
            f'Calls: {pending_calls}'
        )

    async def test_reconcile_sweep_skips_infra_held_row(self, harness: Harness) -> None:
        tid = 'zeta-b6-sweep-skip'
        harness.scheduler.get_statuses = AsyncMock(
            return_value=({tid: 'infra-hold'}, None),
        )

        with patch.object(
            harness, '_reconcile_one_stranded', new=AsyncMock(),
        ) as mock_reconcile_one:
            count = await harness._reconcile_stranded_in_progress()

        mock_reconcile_one.assert_not_called()
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        assert count == 0

    async def test_non_infra_blocked_task_still_flips_to_pending(
        self, harness: Harness,
    ) -> None:
        """Contrast control: a plain status='blocked' task (no infra-hold)
        still resumes via the ordinary Table B resume->pending path — the
        is_infra_held pre-gate must not disturb non-infra tasks."""
        tid = 'zeta-b6-contrast'
        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid, 'status': 'blocked', 'metadata': {},
        })
        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        esc = _make_infra_esc(task_id=tid)

        harness._escalation_events.pop(tid, None)

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_awaited_once_with(tid, 'pending')  # type: ignore[attr-defined]
