"""Truthful REQUEUED / CANCELLED exits (PRD γ3, task 3538).

Two families of untruthful workflow exits are pinned here:

**REQUEUED without a status write.**  ``_drive()``'s ``except
WarmLaneRequeue`` clause and ``_handle_soft_cancel``'s spurious-wakeup
fallback both return ``WorkflowOutcome.REQUEUED`` while leaving the row
``in-progress``.  The harness's slot ``finally`` then nulls the claimant,
producing exactly the ``(in-progress, NULL claimant)`` shape
``shared.task_claimant.is_stranded`` is defined to detect — so a transient
capacity signal degenerates into a strand whose only recovery is the
stranded sweep, which itself refuses to act while any escalation is open.
The fix writes ``pending`` before the REQUEUED exit, through one shared
choke point (``TaskWorkflow._repend_for_requeue``).

**DONE returned against an observed ``cancelled`` row.**  Three sites read
the task status, see a member of ``TERMINAL_STATUSES``, and return
``WorkflowOutcome.DONE`` — including when the observed status is
``cancelled``.  That is both a lie (the tally counts it as completed) and a
live crash: ``_OUTCOME_ALLOWED['done'] == {DONE}``, so ``run()``'s SM-2
exit check raises ``AssertionError``.  The fix maps the observed status onto
its truthful outcome through one shared choke point
(``TaskWorkflow._observed_terminal_outcome``).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec
from _workflow_helpers import FakeScheduler
from shared.task_transitions import outcome_allows_status

from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome, WorkflowState


def _make_workflow(
    *,
    tmp_path: Path,
    task_id: str = '3538',
    scheduler: object | None = None,
) -> TaskWorkflow:
    """Minimal TaskWorkflow over a ``FakeScheduler`` (status history recorded).

    Deliberately does NOT pre-populate ``wf.worktree`` so ``run()`` reaches
    ``create_worktree`` — the boundary-#12a driver needs the warm-lane raise
    to propagate through ``_drive()``'s handlers.
    """
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'Tx', 'description': 'd'}
    assignment.modules = []  # empty → _resolve_module_configs returns [] cleanly

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = tmp_path / 'proj'

    sched = scheduler if scheduler is not None else FakeScheduler()

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=MagicMock(),
        scheduler=sched,  # type: ignore[arg-type]
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    return wf


# ---------------------------------------------------------------------------
# _observed_terminal_outcome — the shared "observed row is terminal" mapper
# ---------------------------------------------------------------------------


def test_observed_cancelled_returns_cancelled_and_enters_cancelled_phase(
    tmp_path: Path,
):
    """(a) An observed ``'cancelled'`` row maps to CANCELLED and moves the phase.

    SM-1 terminal absorption: the workflow's own machine must record that the
    run ended cancelled, not left mid-phase.
    """
    wf = _make_workflow(tmp_path=tmp_path)
    assert wf.machine.state is WorkflowState.PLAN
    assert not wf.machine.is_terminal()

    result = wf._observed_terminal_outcome('cancelled')

    assert result is WorkflowOutcome.CANCELLED
    assert wf.machine.state is WorkflowState.CANCELLED


def test_observed_cancelled_is_idempotent_when_machine_already_cancelled(
    tmp_path: Path,
):
    """(b) Already-CANCELLED machine → CANCELLED returned, no IllegalTransition.

    This is the ``_finalise_cancellation`` path: it enters CANCELLED *before*
    calling ``_handle_soft_cancel``, so the helper must not try to re-enter an
    absorbing state.
    """
    wf = _make_workflow(tmp_path=tmp_path)
    wf.state = WorkflowState.CANCELLED  # force_set — stage the terminal phase
    assert wf.machine.is_terminal()

    result = wf._observed_terminal_outcome('cancelled')

    assert result is WorkflowOutcome.CANCELLED
    assert wf.machine.state is WorkflowState.CANCELLED


def test_observed_done_returns_done_without_moving_the_phase(tmp_path: Path):
    """(c) An observed ``'done'`` row maps to DONE with the phase UNCHANGED.

    Byte-identical phase semantics to the three existing DONE exits — the DONE
    branch deliberately does not transition, so no existing phase assertion
    moves.
    """
    wf = _make_workflow(tmp_path=tmp_path)
    wf.state = WorkflowState.MERGE
    assert not wf.machine.is_terminal()

    result = wf._observed_terminal_outcome('done')

    assert result is WorkflowOutcome.DONE
    assert wf.machine.state is WorkflowState.MERGE


@pytest.mark.parametrize('status', ['cancelled', 'done'])
def test_observed_terminal_outcome_is_sm2_consistent(tmp_path: Path, status: str):
    """(d) Both terminal statuses satisfy the SM-2 exit check they are paired with.

    Asserted through ``outcome_allows_status`` — the SAME authority ``run()``
    consumes — so this test cannot drift from the production predicate.
    ``_OUTCOME_ALLOWED['done'] == {DONE}`` is exactly why returning DONE on a
    ``'cancelled'`` row raises ``AssertionError`` out of ``run()`` today.
    """
    wf = _make_workflow(tmp_path=tmp_path)

    result = wf._observed_terminal_outcome(status)

    assert outcome_allows_status(result, status) is True
