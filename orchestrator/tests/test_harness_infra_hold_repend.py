"""Tests for A1 step-13/14/15/16: harness guards that prevent a verify-complete
infra_hold task from being re-pended.

Step 13 (RED): _revert_in_progress_if_no_live_claimant — task with infra_hold
on a non-degenerate branch must NOT be reverted to pending when the claimant
is gone.  Currently RED because the guard does not exist yet.

Step 15 (RED): _on_escalation_resolved — an infra_issue resolution for an
infra_hold task resumes-at-verify (set in-progress) instead of the default
resume→pending flip.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.harness import Harness

# ---------------------------------------------------------------------------
# Harness fixture (mirrors test_reconcile_stranded.py)
# ---------------------------------------------------------------------------

_BASE_SHA = 'a' * 40   # branch_base_sha in metadata
_TIP_SHA  = 'b' * 40   # tip SHA that is DIFFERENT from base → non-degenerate
_DEGEN_SHA = _BASE_SHA  # tip == base → degenerate


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config) -> Harness:
    """Create a Harness with mocked internals for unit testing infra-hold guards."""
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    # Replace scheduler with async mocks
    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[])
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.update_task = AsyncMock(return_value=True)

    # Default get_task: returns None (no metadata)
    h.scheduler.get_task = AsyncMock(return_value=None)

    # Keep worktree_base real so we can create fake worktrees
    h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()

    # Default cleanup_worktree: side_effect removes the dir
    def _fake_cleanup(worktree_path, tid):
        shutil.rmtree(worktree_path, ignore_errors=True)
    h.git_ops.cleanup_worktree = AsyncMock(side_effect=_fake_cleanup)

    # Default: branch is non-degenerate (tip ≠ base)
    h.git_ops.resolve_branch_sha = AsyncMock(return_value=_TIP_SHA)

    return h


def _make_worktree(harness: Harness, tid: str, *, with_lock: bool = False) -> Path:
    """Create a minimal worktree directory for *tid*.

    If ``with_lock`` is False (default), no plan.lock is created — simulates
    the no-live-claimant / orphan scenario that triggers a revert.
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
# Step 13: _revert_in_progress_if_no_live_claimant — infra_hold guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRevertInProgressInfraHoldGuard:
    """_revert_in_progress_if_no_live_claimant must NOT revert infra-held tasks.

    RED before step-14: function reverts regardless of metadata.
    GREEN after step-14: status='infra-hold' (task 2200/ω4; formerly
    metadata.infra_hold) + non-degenerate branch → skip revert.
    """

    async def test_infra_hold_non_degenerate_not_reverted(
        self, harness: Harness, monkeypatch
    ):
        """Task with status='infra-hold' + non-degenerate branch + no live
        claimant → intact.

        After step-14 (re-keyed onto the first-class status by task 2200/ω4)
        the guard fires and returns None (task left intact).
        """
        tid = '1883'
        _make_worktree(harness, tid)  # no lock → no live claimant

        # Task status is infra-hold (first-class status, task 2200/ω4)
        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid,
            'status': 'infra-hold',
            'metadata': {
                'branch_base_sha': _BASE_SHA,
            },
        })
        # Non-degenerate branch: tip ≠ base
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=_TIP_SHA)

        result = await harness._revert_in_progress_if_no_live_claimant(
            tid, mid_run=False,
        )

        # Must NOT revert — return None (or any non-'reverted' value) and
        # MUST NOT call set_task_status with 'pending'.
        assert result != 'reverted', (
            f'_revert_in_progress_if_no_live_claimant reverted task {tid!r} '
            f"despite status='infra-hold' and branch being non-degenerate. "
            f'An infra-hold must survive the stranded recovery sweep.'
        )
        for call in harness.scheduler.set_task_status.await_args_list:  # type: ignore[attr-defined]
            assert call.args[1] != 'pending', (
                f'set_task_status({tid!r}, "pending") was called despite '
                f"status='infra-hold': {call}"
            )

    async def test_no_infra_hold_still_reverts(self, harness: Harness):
        """Task WITHOUT infra_hold + no live claimant → reverts normally.

        This is the contrast/no-regression case: the guard must be NARROW.
        Passes both before and after step-14 (existing revert behaviour preserved).
        """
        tid = '1884'
        _make_worktree(harness, tid)  # no lock → no live claimant

        # No infra_hold in metadata
        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid,
            'metadata': {},
        })

        result = await harness._revert_in_progress_if_no_live_claimant(
            tid, mid_run=False,
        )

        assert result == 'reverted', (
            f'Task without infra_hold should be reverted to pending; got {result!r}'
        )
        harness.scheduler.set_task_status.assert_awaited_once_with(tid, 'pending')  # type: ignore[attr-defined]

    async def test_infra_hold_degenerate_branch_still_reverts(
        self, harness: Harness, monkeypatch
    ):
        """Task with status='infra-hold' BUT degenerate branch → still reverts.

        A degenerate branch (tip == branch_base_sha) has no real implementation
        commits — it was just provisioned.  The guard is narrow: infra-hold
        alone is insufficient; the branch must also be non-degenerate.

        Passes both before and after step-14 (guard requires BOTH conditions).
        """
        tid = '1885'
        _make_worktree(harness, tid)  # no lock → no live claimant

        # status is infra-hold but branch is degenerate (tip == base → no work done)
        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid,
            'status': 'infra-hold',
            'metadata': {
                'branch_base_sha': _BASE_SHA,
            },
        })
        # Degenerate: branch tip == branch_base_sha
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=_DEGEN_SHA)

        result = await harness._revert_in_progress_if_no_live_claimant(
            tid, mid_run=False,
        )

        # Degenerate branch → still reverts (no real work to preserve)
        assert result == 'reverted', (
            f"Task with status='infra-hold' on degenerate branch should still "
            f'revert; got {result!r}'
        )


# ---------------------------------------------------------------------------
# Step 15: _on_escalation_resolved — infra_hold → resume-at-verify, NOT pending
# ---------------------------------------------------------------------------

from escalation.models import Escalation  # noqa: E402 (after class definitions is fine)


def _make_infra_esc(
    task_id: str = '1883',
    status: str = 'resolved',
    level: int = 1,
    resolved_by: str | None = None,
) -> Escalation:
    """Create a minimal resolved infra_issue escalation."""
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


@pytest.mark.asyncio
class TestInfraHoldEscalationResolution:
    """_on_escalation_resolved with an infra-held task → resume-at-verify
    (in-progress), not pending.

    RED before step-16: _cascade_unblock_member always calls set_task_status('pending'),
    so the infra-held task gets wrongly re-pended.
    GREEN after step-16: is_infra_held (status='infra-hold'; task 2200/ω4,
    formerly metadata.infra_hold) intercepts the flip and sets 'in-progress'
    instead.
    """

    async def test_infra_hold_escalation_resolved_resumes_at_verify(
        self, harness: Harness
    ):
        """status='infra-hold' task: resolved infra_issue escalation →
        'in-progress', NOT 'pending'.

        After step-16 (re-keyed onto the first-class status by task 2200/ω4)
        _cascade_unblock_member checks is_infra_held before the blocked-gate
        and calls 'in-progress' instead of 'pending'.
        """
        tid = '1883'

        # Task status is infra-hold (first-class status, task 2200/ω4)
        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid,
            'status': 'infra-hold',
            'metadata': {
                'branch_base_sha': _BASE_SHA,
            },
        })
        harness.scheduler.get_status = AsyncMock(return_value='infra-hold')

        esc = _make_infra_esc(task_id=tid)  # status='resolved' → legacy 'resume' action

        # Ensure orphan path: task not in _escalation_events
        harness._escalation_events.pop(tid, None)

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        # Must NOT flip to 'pending'
        pending_calls = [
            c for c in harness.scheduler.set_task_status.await_args_list  # type: ignore[attr-defined]
            if c.args[1] == 'pending'
        ]
        assert not pending_calls, (
            f"set_task_status('pending') was called despite status='infra-hold'. "
            f'An infra-held task must resume-at-verify (in-progress), not re-pend. '
            f'Calls: {pending_calls}'
        )

        # Must flip to 'in-progress' (resume-at-verify)
        inprog_calls = [
            c for c in harness.scheduler.set_task_status.await_args_list  # type: ignore[attr-defined]
            if c.args[1] == 'in-progress'
        ]
        assert inprog_calls, (
            f"set_task_status('in-progress') was never called for status='infra-hold' "
            f'task. Expected resume-at-verify but got: '
            f'{[c.args for c in harness.scheduler.set_task_status.await_args_list]}'  # type: ignore[attr-defined]
        )

    async def test_no_infra_hold_still_repends(self, harness: Harness):
        """Task without infra_hold: resolved escalation → 'pending' (no regression).

        This is the contrast/no-regression case. Passes both before and after
        step-16 (existing behaviour preserved for non-infra_hold tasks).
        """
        tid = '1884'

        # No infra_hold in metadata
        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid,
            'metadata': {},
        })
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        esc = _make_infra_esc(task_id=tid)

        harness._escalation_events.pop(tid, None)

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        # Normal resume → flip to 'pending'
        harness.scheduler.set_task_status.assert_awaited_once_with(tid, 'pending')  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Task 2200 (ω4) step-5: the reconcile SKIP site keys on the task's
# first-class status via is_infra_held, not the retired metadata.infra_hold
# boolean.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRevertInProgressKeysOnInfraHoldStatus:
    """_revert_in_progress_if_no_live_claimant must key its hold guard on
    the task's status (is_infra_held), not the retired metadata.infra_hold
    flag.

    RED before step-6: the guard still reads metadata.infra_hold only and
    never looks at status.
    """

    async def test_status_infra_hold_non_degenerate_not_reverted(
        self, harness: Harness,
    ):
        """(i) status='infra-hold' + non-degenerate branch + no live claimant
        -> NOT reverted, even with no metadata.infra_hold flag at all — the
        first-class status alone is now the hold signal.
        """
        tid = '1883'
        _make_worktree(harness, tid)  # no lock → no live claimant

        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid,
            'status': 'infra-hold',
            'metadata': {'branch_base_sha': _BASE_SHA},
        })
        # Non-degenerate branch: tip ≠ base
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=_TIP_SHA)

        result = await harness._revert_in_progress_if_no_live_claimant(
            tid, mid_run=False,
        )

        assert result != 'reverted', (
            f"_revert_in_progress_if_no_live_claimant reverted task {tid!r} "
            f"despite status='infra-hold' on a non-degenerate branch. An "
            f'infra-hold must survive the stranded recovery sweep.'
        )
        for call in harness.scheduler.set_task_status.await_args_list:  # type: ignore[attr-defined]
            assert call.args[1] != 'pending', (
                f'set_task_status({tid!r}, "pending") was called despite '
                f"status='infra-hold': {call}"
            )

    async def test_legacy_metadata_flag_without_infra_hold_status_still_reverts(
        self, harness: Harness,
    ):
        """(ii) status='in-progress' + a LEGACY metadata.infra_hold flag
        (status != 'infra-hold') -> STILL reverted.  Proves the guard no
        longer honors the retired metadata flag by itself.
        """
        tid = '1884'
        _make_worktree(harness, tid)  # no lock → no live claimant

        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid,
            'status': 'in-progress',
            'metadata': {
                'infra_hold': {'phase': 'warm_marker', 'errno': 28},
                'branch_base_sha': _BASE_SHA,
            },
        })
        # Non-degenerate branch: tip ≠ base
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=_TIP_SHA)

        result = await harness._revert_in_progress_if_no_live_claimant(
            tid, mid_run=False,
        )

        assert result == 'reverted', (
            f'Task with a legacy metadata.infra_hold flag but '
            f"status != 'infra-hold' should still be reverted to pending "
            f'(the retired flag must not be honored on its own); '
            f'got {result!r}'
        )
        harness.scheduler.set_task_status.assert_awaited_once_with(tid, 'pending')  # type: ignore[attr-defined]


@pytest.mark.asyncio
class TestReconcileSweepExcludesInfraHold:
    """Regression lock: infra-hold rows never reach _reconcile_one_stranded.

    _RECONCILE_SWEEP_STATUSES = {'in-progress', 'blocked'} already excludes
    'infra-hold', so this is green-on-arrival — no production code change
    accompanies it.  It guards against a future edit silently adding
    'infra-hold' to the sweep set, which would re-introduce the re-pend bug
    this task's STAMP/HOLD/RESUME migration fixes.
    """

    async def test_infra_hold_row_never_reaches_reconcile_one_stranded(
        self, harness: Harness,
    ):
        tid = '1886'
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


# ---------------------------------------------------------------------------
# Task 2200 (ω4) step-7: the WRITE site _cascade_unblock_member keys on the
# task's first-class status via is_infra_held, not the retired
# metadata.infra_hold boolean, and must check it BEFORE the status != 'blocked'
# early return — an infra-held task is never 'blocked', so the pre-step-8
# early return skips the resume-at-verify branch entirely.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCascadeUnblockKeysOnInfraHoldStatus:
    """_cascade_unblock_member must key its infra-hold resume branch on
    is_infra_held(task) — the task's first-class status — checked BEFORE the
    status != 'blocked' early return, and resume-at-verify from that status
    alone (no metadata.infra_hold flag to read or clear).

    RED before step-8: the cascade reads status via get_status() and
    early-returns whenever it is not 'blocked' — an infra-held task's status
    is 'infra-hold', so it never reaches the (also still metadata-keyed)
    infra branch at all.
    """

    async def test_status_infra_hold_resumes_at_verify(
        self, harness: Harness,
    ):
        """status='infra-hold' (no metadata.infra_hold flag) on a resolved
        infra_issue L1, via the orphan path -> resume-at-verify
        ('in-progress'), never 'pending', and no update_task call (there is
        no flag to clear under the first-class status).
        """
        tid = '1887'

        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid,
            'status': 'infra-hold',
            'metadata': {},
        })
        harness.scheduler.get_status = AsyncMock(return_value='infra-hold')

        esc = _make_infra_esc(task_id=tid)  # status='resolved' → legacy 'resume' action

        # Ensure orphan path: task not in _escalation_events
        harness._escalation_events.pop(tid, None)

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        # Must resume-at-verify: 'in-progress'.
        harness.scheduler.set_task_status.assert_awaited_once_with(tid, 'in-progress')  # type: ignore[attr-defined]

        # Must NOT flip to 'pending'.
        pending_calls = [
            c for c in harness.scheduler.set_task_status.await_args_list  # type: ignore[attr-defined]
            if c.args[1] == 'pending'
        ]
        assert not pending_calls, (
            f"set_task_status('pending') was called for a status='infra-hold' "
            f'task. Calls: {pending_calls}'
        )

        # No metadata.infra_hold clear — there is no flag left to clear.
        harness.scheduler.update_task.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_blocked_no_infra_hold_still_flips_to_pending(
        self, harness: Harness,
    ):
        """Contrast/no-regression: plain status='blocked' + metadata:{} still
        flips to 'pending' via Table B — the is_infra_held pre-gate must not
        disturb the existing non-infra resume path.
        """
        tid = '1888'

        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid,
            'status': 'blocked',
            'metadata': {},
        })
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        esc = _make_infra_esc(task_id=tid)

        harness._escalation_events.pop(tid, None)

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_awaited_once_with(tid, 'pending')  # type: ignore[attr-defined]
