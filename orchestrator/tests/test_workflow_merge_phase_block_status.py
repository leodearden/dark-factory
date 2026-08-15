"""Boundary #13 (row-``blocked`` clause): a merge-phase ``_mark_blocked`` that
EXITS THE SLOT must write its park status.

PRD ``plans/task-escalation-state-graph-prd.md`` task γ2 (D6); normative spec
``docs/task-escalation-state-spec.md`` §8-E2 and §5; invariant INV-6
(status-matches-liveness — every exit from a claimed state writes its successor
status through the transition's designated choke point).

The defect: ``_mark_blocked``'s ``if not merge_phase:`` ENTRY gate suppresses the
task-status transition for the whole call, so every merge-phase BLOCKED return
left the row at ``in-progress`` with no live claimant.

The fix is NOT to delete that gate.  The Chesterton's-fence investigation
(commit 22918d5c24, recorded in ``_mark_blocked``'s ``MERGE_PHASE_RATIONALE``
docstring paragraph) found a REAL dependency on the suppression — but only for
the REQUEUED retry-in-place arm, where the row must stay ``in-progress`` under a
live claimant while ``_run_merge_phase`` retries the merge in-slot.  So the park
write moves to the BLOCKED RETURN points, and this module pins BOTH halves:

  * the park write happens on the two genuine slot-exit BLOCKED returns
    (``escalate_to_human`` short-circuit, and the final fall-through), for both
    ``merge_phase`` values and honouring ``block_status``;
  * the fence is preserved — the REQUEUED retry-in-place arm still writes
    NEITHER ``blocked`` NOR ``pending``, and the §5 steward-terminal-decision
    preserve carve-out is not clobbered.

Both halves are pinned BEHAVIOURALLY, through ``_mark_blocked`` itself.  An
earlier revision also carried source-structure meta-tests over workflow.py (a
``merge_phase=True`` call-site allowlist, and rationale-marker greps); they were
dropped because they asserted on cosmetic source shape — function names, call
nesting — rather than on the property that matters, and so failed on
behaviour-preserving refactors while adding nothing the tests below miss.  The
rationale for the surviving suppression lives in ``_mark_blocked``'s
``MERGE_PHASE_RATIONALE`` docstring paragraph and at the call site in
``_run_merge_phase``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler
from escalation.queue import EscalationQueue
from shared.task_statuses import TaskStatus

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow
from orchestrator.workflow_types import (
    StewardResolved,
    StewardTerminalDecision,
    WorkflowOutcome,
    WorkflowState,
)

TASK_ID = '3537'


def _make_config(tmp_path: Path) -> OrchestratorConfig:
    """A REAL ``OrchestratorConfig``.

    Deliberately not the shared ``mock_orch_config`` fixture: that is a
    ``MagicMock(spec_set=pydantic_spec(OrchestratorConfig))`` and raises
    ``AttributeError`` on BaseModel API, which ``_mark_blocked``'s
    ``_durable_ref_suffix`` / dry-run guards reach through.  Mirrors
    ``test_workflow_merge_blocked_surfacing._make_routing_config``.
    """
    return OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


def _make_workflow(
    tmp_path: Path, *, with_queue: bool = False,
) -> TaskWorkflow:
    """A minimal TaskWorkflow whose scheduler RECORDS every status write."""
    assignment = TaskAssignment(
        task_id=TASK_ID,
        task={
            'id': TASK_ID,
            'title': 'merge-phase park status',
            'description': '',
            'status': 'in-progress',
            'metadata': {},
            'dependencies': [],
        },
        modules=[],
    )
    queue = EscalationQueue(tmp_path / 'escalations') if with_queue else None
    workflow = TaskWorkflow(
        assignment=assignment,
        config=_make_config(tmp_path),
        git_ops=MagicMock(),
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        briefing=FakeBriefing(),    # type: ignore[arg-type]
        mcp=FakeMcp(),              # type: ignore[arg-type]
        escalation_queue=queue,
    )
    workflow.worktree = tmp_path / 'wt'
    # The dry-run investigation spawn is orthogonal to the row write and would
    # leave a pending background task behind; stub it out.
    workflow._spawn_dry_run_unblock = MagicMock()  # type: ignore[method-assign]
    return workflow


def _statuses(workflow: TaskWorkflow) -> list[str]:
    assert isinstance(workflow.scheduler, FakeScheduler)
    return workflow.scheduler.statuses.get(TASK_ID, [])


# ---------------------------------------------------------------------------
# E2 / INV-6 — the park write happens on every slot-exiting BLOCKED return
# ---------------------------------------------------------------------------

class TestMergePhaseBlockedWritesParkStatus:
    """Spec §8-E2: a BLOCKED return from ``_mark_blocked`` frees the slot, so it
    owes the row write REGARDLESS of ``merge_phase``.

    RED for ``merge_phase=True``: the ENTRY gate suppresses the write entirely,
    so ``scheduler.statuses`` is empty and the row is left ``in-progress`` with
    no live claimant (INV-6 violation).
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize('merge_phase', [False, True], ids=['plain', 'merge_phase'])
    async def test_fall_through_blocked_writes_row(
        self, tmp_path: Path, merge_phase: bool,
    ) -> None:
        workflow = _make_workflow(tmp_path)
        workflow._enter_phase(WorkflowState.MERGE)
        pre_block_state = workflow.machine.state

        outcome = await workflow._mark_blocked(
            'merge verification failed',
            detail='the branch did not verify on top of main',
            merge_phase=merge_phase,
            skip_escalation=True,
        )

        # (a) the slot is being freed
        assert outcome == WorkflowOutcome.BLOCKED
        # (b) ...so the row must say so
        assert _statuses(workflow)[-1:] == ['blocked'], (
            f'merge_phase={merge_phase}: the BLOCKED fall-through exits the slot, '
            f'so it must write the park status (INV-6); '
            f'got statuses={_statuses(workflow)!r}'
        )
        # (d) the TerminalReport still stamps the PRE-block phase
        report = workflow._terminal_report
        assert report is not None
        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.blocked_from_phase == pre_block_state

    @pytest.mark.asyncio
    @pytest.mark.parametrize('merge_phase', [False, True], ids=['plain', 'merge_phase'])
    async def test_fall_through_honours_block_status(
        self, tmp_path: Path, merge_phase: bool,
    ) -> None:
        """(c) The merge-aware write must thread ``block_status`` through, not
        hardcode ``'blocked'`` — otherwise a verify-infra STAMP lands on the
        generic status and the harness HOLD guard / RESUME cascade (both keyed
        on ``is_infra_held``) never sees it."""
        workflow = _make_workflow(tmp_path)
        workflow._enter_phase(WorkflowState.MERGE)

        outcome = await workflow._mark_blocked(
            'verify infra unavailable',
            block_status='infra-hold',
            merge_phase=merge_phase,
            skip_escalation=True,
        )

        assert outcome == WorkflowOutcome.BLOCKED
        assert _statuses(workflow)[-1:] == ['infra-hold'], (
            f'merge_phase={merge_phase}: block_status must be honoured by the '
            f'merge-aware park write; got {_statuses(workflow)!r}'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('merge_phase', [False, True], ids=['plain', 'merge_phase'])
    async def test_escalate_to_human_short_circuit_writes_row(
        self, tmp_path: Path, merge_phase: bool,
    ) -> None:
        """The ``escalate_to_human=True`` short-circuit is the OTHER genuine
        slot-exiting BLOCKED return — it skips the steward entirely and hands
        off to a human, so nothing will retry in-place."""
        workflow = _make_workflow(tmp_path, with_queue=True)
        workflow._enter_phase(WorkflowState.MERGE)

        outcome = await workflow._mark_blocked(
            'confirmed merge loop — no automated resolution path',
            escalate_to_human=True,
            merge_phase=merge_phase,
        )

        assert outcome == WorkflowOutcome.BLOCKED
        assert _statuses(workflow)[-1:] == ['blocked'], (
            f'merge_phase={merge_phase}: the escalate_to_human short-circuit '
            f'hands off to a human and exits the slot, so it must write the '
            f'park status (INV-6); got {_statuses(workflow)!r}'
        )
        # The human-facing L1 is still the sole escalation for this exit.
        assert workflow.escalation_queue is not None
        pending = workflow.escalation_queue.get_pending()
        assert len(pending) == 1 and pending[0].level == 1


# ---------------------------------------------------------------------------
# The Chesterton's fence — the two returns that must NOT write
# ---------------------------------------------------------------------------

class TestMergePhaseFencePreserved:
    """GREEN both before and after: these two returns keep the suppression.

    Deleting the ``if not merge_phase:`` ENTRY gate (the naive fix) would break
    the fence in the other direction — an INV-6 violation SM-2 cannot catch,
    because ``outcome_allows_status('requeued', BLOCKED)`` is True.
    """

    @staticmethod
    def _wire_steward(workflow: TaskWorkflow, outcome) -> None:
        workflow._ensure_steward_started = AsyncMock()  # type: ignore[method-assign]
        workflow._steward = MagicMock()
        workflow._await_steward_completion = AsyncMock(  # type: ignore[method-assign]
            return_value=outcome,
        )

    @pytest.mark.asyncio
    async def test_requeue_retry_in_place_writes_neither_status(
        self, tmp_path: Path,
    ) -> None:
        """``StewardResolved`` under ``merge_phase=True`` means
        ``_run_merge_phase`` retries the merge IN-SLOT: the claimant is still
        live, so the row must stay ``in-progress`` and the durable obligation is
        carried by ``metadata.merge_retry_pending``."""
        workflow = _make_workflow(tmp_path, with_queue=True)
        workflow._enter_phase(WorkflowState.MERGE)
        self._wire_steward(workflow, StewardResolved(resolution_text='fixed on main'))
        stamp = AsyncMock()
        workflow._stamp_merge_retry_pending = stamp  # type: ignore[method-assign]

        outcome = await workflow._mark_blocked(
            'merge verification failed', merge_phase=True,
        )

        assert outcome == WorkflowOutcome.REQUEUED
        assert 'blocked' not in _statuses(workflow), (
            'the retry-in-place arm keeps a LIVE claimant — a blocked row here '
            'would be an INV-6 violation in the opposite direction '
            f'(statuses={_statuses(workflow)!r})'
        )
        assert 'pending' not in _statuses(workflow), (
            'merge_phase=True must not re-pend: the caller retries in-slot '
            f'rather than requeueing through the scheduler (statuses={_statuses(workflow)!r})'
        )
        stamp.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_steward_terminal_decision_preserve_carve_out(
        self, tmp_path: Path,
    ) -> None:
        """Spec §5 preserve carve-out: the steward already adjudicated the row
        to ``deferred``.  ``deferred`` is NOT in ``shared.task_statuses.TERMINAL``
        (only ``{DONE, CANCELLED}``), so nothing else would stop a blanket write
        from silently clobbering a human-visible adjudication."""
        workflow = _make_workflow(tmp_path, with_queue=True)
        workflow._enter_phase(WorkflowState.MERGE)
        self._wire_steward(
            workflow, StewardTerminalDecision(new_status=TaskStatus.DEFERRED),
        )

        outcome = await workflow._mark_blocked(
            'merge verification failed', merge_phase=True,
        )

        assert outcome == WorkflowOutcome.BLOCKED
        assert 'blocked' not in _statuses(workflow), (
            "the steward's 'deferred' adjudication must be preserved, not "
            f'clobbered with blocked (statuses={_statuses(workflow)!r})'
        )
