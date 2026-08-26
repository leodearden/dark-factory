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
from _orch_helpers import (
    LIVE_CLAIMANT,
    claimant_row,
    wire_scheduler_liveness_mock,
)
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
    # Task 3540: is_actively_held auto-mocks TRUTHY on a bare MagicMock,
    # so every row would read as having a live claimant and every
    # resume flip would be silently skipped. Wire the real accessors.
    wire_scheduler_liveness_mock(h.scheduler)
    h.scheduler.get_status = AsyncMock(return_value='blocked')
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))
    h.scheduler.set_task_status = AsyncMock()
    # Task 3540: the row must carry 'status' too, and it must AGREE with
    # get_status above. _cascade_unblock_member re-reads the row immediately
    # before the write and re-applies the status/liveness gate to THAT
    # snapshot (INV-3), so a row that omits 'status' describes a task that
    # cannot exist and reads as "left the re-pendable statuses" -> no write.
    h.scheduler.get_task = AsyncMock(
        return_value={'id': 'task', 'status': 'blocked', 'metadata': {}}
    )
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


def _liveness_row(task_id: str, status: str, *, live: bool) -> dict:
    """A task row carrying the claimant columns the task-3540 resume fork reads.

    ``harness.py::Harness._resume_repend_liveness`` folds
    ``scheduler.is_actively_held`` (wired real by
    ``wire_scheduler_liveness_mock`` above, so False unless a test dispatches
    or locks the id) with ``shared.task_claimant.has_live_claimant`` over these
    columns.

    A thin ``live``-flavoured adapter over ``_orch_helpers.claimant_row`` — the
    SHARED builder this suite, ``test_cascade_unblock.py`` and
    ``test_resume_claimant_liveness.py`` all use, so the heartbeat ages (and
    the TTL they derive from) cannot drift apart between them.

    ``live=False`` means NO claimant at all — the stranded shape B2 is about.
    """
    return claimant_row(
        status,
        task_id=task_id,
        claimant=LIVE_CLAIMANT if live else None,
        heartbeat='fresh' if live else None,
    )


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
# B2 (esc-2073-15 regression) — TWO mechanisms now re-pend a stranded
# in-progress row, and this section pins both: the stranded-revert sweep
# harness.py::Harness._revert_in_progress_if_no_live_claimant (the periodic
# owner), and — since task 3540 — the escalation-resume cascade
# harness.py::Harness._cascade_unblock_member, which gates on CLAIMANT
# LIVENESS rather than status == 'blocked'.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB2StrandedInProgressRevert:
    """B2: a stranded in-progress task with no live claimant is re-pended.

    Two independent edges deliver that, and both are pinned here:

      - ``harness.py::Harness._revert_in_progress_if_no_live_claimant`` — the
        periodic stranded-revert sweep, which owns the row when no escalation
        resolution is involved at all.
      - ``harness.py::Harness._cascade_unblock_member`` — the escalation-resume
        cascade.  Task 3540 (PRD ``plans/task-escalation-state-graph-prd.md``
        D8, spec E9) re-anchored its gate from ``status == 'blocked'`` onto
        claimant LIVENESS, so a resolution landing on a stranded in-progress
        row now re-pends it immediately instead of waiting a sweep interval.
        Its live-claimant twin still writes nothing — a running workflow owns
        its own re-pend, having been woken synchronously by ``event.set()``.

    An infra-held row is exempt from the sweep.
    Mirrors ``test_harness_infra_hold_repend.py`` (the revert-liveness suite)."""

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

    async def test_live_looking_lock_no_longer_gates_applier(self, harness: Harness) -> None:
        """Task 2243, W10-θ2 step-16: a plan.lock recording a live owner_pid
        no longer short-circuits this applier — that determination now
        belongs to TaskGroundTruth.recovery_for's live_claimant resolution,
        which has already ruled out a live claimant before REVERT_TO_PENDING
        is dispatched here. Calling the applier directly with a live-looking
        lock now reverts (see test_harness_infra_hold_repend.py's
        TestRevertInProgressLivenessGateRetired for the full parity suite)."""
        tid = 'zeta-b2-live'
        _make_worktree(harness, tid, with_lock=True)
        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid, 'status': 'in-progress', 'metadata': {},
        })

        result = await harness._revert_in_progress_if_no_live_claimant(tid, mid_run=False)

        assert result == 'reverted', (
            f'The applier no longer re-derives plan.lock liveness; a '
            f'live-looking lock must not block the revert. Got {result!r}'
        )
        harness.scheduler.set_task_status.assert_awaited_once_with(tid, 'pending')  # type: ignore[attr-defined]

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

    async def test_cascade_unblock_member_repends_stranded_in_progress(
        self, harness: Harness,
    ) -> None:
        """B2 IS also served by the escalation cascade, as of task 3540.

        A status='in-progress' row with NO live claimant, routed through
        ``_on_escalation_resolved``'s l2-cascade path (->
        ``_cascade_unblock_member``), is re-pended immediately.

        This overturns the codification this cell previously carried (that
        ``_cascade_unblock_member`` is blocked-only and DEBUG-skips in-progress
        unconditionally). The twin re-anchor lives in
        ``test_cascade_unblock.py::TestCascadeUnblockCriteria``
        (``test_criterion_4a_in_progress_with_live_claimant_not_flipped`` /
        ``test_criterion_4b_stranded_in_progress_is_repended``); the fork's own
        contract suite is ``test_resume_claimant_liveness.py``.

        Unaffected by task 3538, and deliberately distinct from it: this row's
        status is 'in-progress' (is_infra_held False), so it never reaches the
        infra pre-gate whose target that task changed — it reaches the ordinary
        Table B resume, whose target is looked up rather than asserted as a
        literal."""
        task_id = 'zeta-b2-stranded-cascade'
        esc = _make_esc(
            task_id=task_id,
            resolution_action=None,  # legacy resolve -> maps to 'resume'
            status='resolved',
            resolved_by='l2-cascade:esc-parent-1',
            level=1,
        )
        harness.scheduler.get_status = AsyncMock(return_value='in-progress')
        harness.scheduler.get_task = AsyncMock(
            return_value=_liveness_row(task_id, 'in-progress', live=False)
        )

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        effect = effect_for('resume', esc.level, esc.category)
        assert effect is not None and effect.target_status is not None
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            task_id, effect.target_status,
        )

    async def test_cascade_unblock_member_does_not_flip_live_in_progress(
        self, harness: Harness, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The half of the old carve-out that SURVIVES task 3540.

        An in-progress row whose claimant is heartbeating inside
        ``claimant_liveness_ttl_secs`` still gets no write: the running
        workflow owns its own re-pend (it was woken synchronously by
        ``event.set()`` in ``_on_escalation_resolved``), so a flip here would
        race it. Liveness — not the status string — is what establishes
        that."""
        task_id = 'zeta-b2-live-cascade'
        esc = _make_esc(
            task_id=task_id,
            resolution_action=None,  # legacy resolve -> maps to 'resume'
            status='resolved',
            resolved_by='l2-cascade:esc-parent-1',
            level=1,
        )
        harness.scheduler.get_status = AsyncMock(return_value='in-progress')
        harness.scheduler.get_task = AsyncMock(
            return_value=_liveness_row(task_id, 'in-progress', live=True)
        )

        with caplog.at_level(logging.DEBUG):
            harness._on_escalation_resolved(esc)
            await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]
        assert any(
            r.levelno == logging.DEBUG and 'live claimant' in r.getMessage()
            for r in caplog.records
        ), 'Expected a DEBUG record naming the live claimant as the skip reason'


# ---------------------------------------------------------------------------
# B6 — infra-hold first-class: a resolved infra_issue escalation on an
# infra-held task resumes through the dedicated is_infra_held branch (never
# dropped by the blocked-only gate); the reconcile stranded sweep skips an
# infra-held row; a non-infra blocked task still flips to pending (contrast).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB6InfraHoldFirstClass:
    """B6: infra-hold is a first-class status (task 2200/omega4). A resolved
    infra_issue escalation on an infra-held task resumes via
    _cascade_unblock_member's A1 pre-gate rather than being skipped by the
    blocked-only gate (an infra-held row is never 'blocked'); the reconcile
    stranded sweep skips an infra-held row entirely (is_infra_held guard); a
    non-infra blocked task still flips to pending. Mirrors
    test_harness_infra_hold_repend.py's TestInfraHoldEscalationResolution.

    HOLD side vs RESUME side (task 3538 / PRD γ3): the sweep-skip below is the
    HOLD side and is unchanged. The RESUME side's target moved from
    'in-progress' to 'pending' — a claimant-less 'in-progress' is the stranded
    shape and is undispatchable, while resume-at-verify is branch-keyed
    (_has_prior_implementation) and survives the change. See
    test_harness_infra_resume_truthful.py."""

    async def test_infra_hold_resolved_escalation_repends_for_dispatch(
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

        harness.scheduler.set_task_status.assert_awaited_once_with(tid, 'pending')  # type: ignore[attr-defined]
        inprog_calls = [
            c for c in harness.scheduler.set_task_status.await_args_list  # type: ignore[attr-defined]
            if c.args[1] == 'in-progress'
        ]
        assert not inprog_calls, (
            f"set_task_status('in-progress') was called for status='infra-hold' — "
            f'that is the claimant-less strand shape. Calls: {inprog_calls}'
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
