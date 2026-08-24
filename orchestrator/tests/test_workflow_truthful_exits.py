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

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec
from _workflow_helpers import FakeScheduler
from shared.task_transitions import outcome_allows_status

from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import (
    WarmLaneDiskPressure,
    WarmLanePoolExhausted,
    WarmLanePoolHardDown,
    WarmLaneRequeue,
    WarmLaneReseedContaminated,
    WarmLaneSoftPressure,
)
from orchestrator.scheduler import (
    DoneGateRejection,
    SetTaskStatusRejected,
    TerminalExitRejection,
)
from orchestrator.workflow import (
    TaskWorkflow,
    WorkflowCancelled,
    WorkflowOutcome,
    WorkflowState,
)
from orchestrator.workflow_types import classify_failure


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

    git_ops = MagicMock()
    # run()'s terminal cleanup awaits this whenever the machine ends in
    # DONE/CANCELLED — a bare MagicMock is not awaitable, so the CANCELLED
    # exits exercised below need a real coroutine here.
    git_ops.release_lane_for_terminal_task = AsyncMock()

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
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


# ---------------------------------------------------------------------------
# _repend_for_requeue — the shared "make a REQUEUED exit truthful" choke point
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repend_for_requeue_writes_pending_and_returns_none(tmp_path: Path):
    """(a) Happy path: writes ``pending`` exactly once, returns ``None``.

    ``None`` means "no terminal override — caller keeps its REQUEUED exit".
    The write must land BEFORE the caller returns, so that by the time the
    harness's slot ``finally`` nulls the claimant the row is ``pending``, not
    the ``(in-progress, NULL claimant)`` strand shape.
    """
    sched = FakeScheduler()
    wf = _make_workflow(tmp_path=tmp_path, scheduler=sched)

    result = await wf._repend_for_requeue()

    assert result is None
    assert sched.statuses[wf.task_id] == ['pending']


@pytest.mark.asyncio
async def test_repend_for_requeue_maps_cancelled_rejection_to_cancelled(
    tmp_path: Path,
):
    """(b) ``TerminalExitRejection(old_status='cancelled')`` → CANCELLED.

    Delegated to ``_observed_terminal_outcome``, so the machine also enters
    CANCELLED — the terminal row wins over the caller's REQUEUED intent.
    """
    wf = _make_workflow(tmp_path=tmp_path)
    wf.scheduler.set_task_status = AsyncMock(  # type: ignore[method-assign]
        side_effect=TerminalExitRejection(
            task_id=wf.task_id, old_status='cancelled',
            target_status='pending', raw='terminal-exit gate',
        )
    )

    result = await wf._repend_for_requeue()

    assert result is WorkflowOutcome.CANCELLED
    assert wf.machine.state is WorkflowState.CANCELLED


@pytest.mark.asyncio
async def test_repend_for_requeue_maps_done_rejection_to_done(tmp_path: Path):
    """(c) ``TerminalExitRejection(old_status='done')`` → DONE, phase unmoved."""
    wf = _make_workflow(tmp_path=tmp_path)
    wf.scheduler.set_task_status = AsyncMock(  # type: ignore[method-assign]
        side_effect=TerminalExitRejection(
            task_id=wf.task_id, old_status='done',
            target_status='pending', raw='terminal-exit gate',
        )
    )

    result = await wf._repend_for_requeue()

    assert result is WorkflowOutcome.DONE
    assert wf.machine.state is WorkflowState.PLAN


@pytest.mark.asyncio
async def test_repend_for_requeue_logs_other_rejections_and_returns_none(
    tmp_path: Path, caplog,
):
    """(d) A non-terminal ``SetTaskStatusRejected`` is loud (ERROR) but not fatal.

    The caller keeps its REQUEUED exit — the row stays ``in-progress``, which
    ``_OUTCOME_ALLOWED['requeued']`` still permits today; task θ's narrowing to
    ``{PENDING}`` is what will make this case loud at the SM-2 check.
    """
    wf = _make_workflow(tmp_path=tmp_path)
    wf.scheduler.set_task_status = AsyncMock(  # type: ignore[method-assign]
        side_effect=SetTaskStatusRejected(
            task_id=wf.task_id, error_code='unknown', raw='server said no',
        )
    )

    with caplog.at_level(logging.ERROR):
        result = await wf._repend_for_requeue()

    assert result is None
    assert any(
        rec.levelno >= logging.ERROR and 'unknown' in rec.getMessage()
        for rec in caplog.records
    ), f'expected an ERROR log naming the error_code; got {caplog.records!r}'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('exc', 'expected'),
    [
        (SetTaskStatusRejected(task_id='3538', error_code='unknown', raw='r'), None),
        (
            TerminalExitRejection(
                task_id='3538', old_status='done', target_status='pending', raw='r',
            ),
            WorkflowOutcome.DONE,
        ),
        (
            TerminalExitRejection(
                task_id='3538', old_status='cancelled', target_status='pending',
                raw='r',
            ),
            WorkflowOutcome.CANCELLED,
        ),
        (DoneGateRejection(task_id='3538', missing_files=['a.py'], raw='r'), None),
        # NOT a SetTaskStatusRejected — and that is the whole point:
        # ``Scheduler.set_task_status`` raises a BARE ``RuntimeError`` once its
        # ``fm_retry_backoffs()`` transient loop is exhausted (scheduler.py:2845),
        # i.e. the fused-memory-restarting / MCP-unreachable case, which is
        # precisely the infra degradation that GENERATES warm-lane requeues.
        (
            RuntimeError(
                'set_task_status(3538, pending) failed after 4 transient '
                'retries: TimeoutError'
            ),
            None,
        ),
        # ``dispatch_tool`` can also surface a bare transport error.
        (TimeoutError(), None),
    ],
    ids=[
        'base', 'terminal-done', 'terminal-cancelled', 'done-gate',
        'transient-retries-exhausted', 'bare-transport-error',
    ],
)
async def test_repend_for_requeue_never_reraises(
    tmp_path: Path, exc: BaseException, expected: WorkflowOutcome | None, caplog,
):
    """(e) NOTHING escapes the helper — rejection-shaped or not.

    Both call sites sit OUTSIDE ``_drive()``'s ``except SetTaskStatusRejected``
    handler — the ``WarmLaneRequeue`` clause is its SIBLING (so a raise inside
    it is not caught by later clauses of the same ``try``), and
    ``_handle_soft_cancel`` runs from inside ``run()``'s ``except
    WorkflowCancelled`` handler, while ``run()`` catches nothing else.  So any
    escape destroys the ``TerminalReport`` (the harness sees ``report is
    None``), skipping ``_apply_retry_cap`` / ``record_requeue`` /
    ``counts_against_requeue_cap`` bookkeeping entirely AND leaving the row
    ``in-progress`` for the slot ``finally`` to strand — strictly worse than
    the pre-γ3 behaviour, which always returned a REQUEUED report.

    ``None`` means "no terminal override — the caller keeps its REQUEUED
    exit"; a ``WorkflowOutcome`` means the row turned out terminal and that
    verdict wins.
    """
    wf = _make_workflow(tmp_path=tmp_path)
    wf.scheduler.set_task_status = AsyncMock(side_effect=exc)  # type: ignore[method-assign]

    with caplog.at_level(logging.ERROR):
        result = await wf._repend_for_requeue()  # must not raise

    assert result is expected
    if expected is None:
        assert any(
            rec.levelno >= logging.ERROR for rec in caplog.records
        ), f'expected a loud ERROR log for {exc!r}; got {caplog.records!r}'


@pytest.mark.asyncio
async def test_repend_for_requeue_propagates_workflow_cancelled(tmp_path: Path):
    """(f) ``WorkflowCancelled`` is deliberately NOT swallowed.

    It subclasses ``Exception`` (workflow_types.py:830), so a blanket
    ``except Exception`` arm would capture it and violate CX-1's "raised by
    CancellationScope and caught at EXACTLY ONE place — TaskWorkflow.run()",
    silently downgrading a cancellation into a REQUEUED exit.  (``asyncio.
    CancelledError`` needs no such carve-out: it derives from
    ``BaseException`` on this repo's ``>=3.11`` floor.)
    """
    wf = _make_workflow(tmp_path=tmp_path)
    wf.scheduler.set_task_status = AsyncMock(  # type: ignore[method-assign]
        side_effect=WorkflowCancelled('soft')
    )

    with pytest.raises(WorkflowCancelled) as caught:
        await wf._repend_for_requeue()

    assert caught.value.kind == 'soft'


# ---------------------------------------------------------------------------
# Boundary #12a — the warm-lane REQUEUED exit re-pends the row
# ---------------------------------------------------------------------------

WARM_LANE_EXCS = [
    WarmLanePoolExhausted("warm-lane pool exhausted for branch '3538'; requeue"),
    WarmLaneDiskPressure("warm-lane seed disk pressure for branch '3538'; requeue"),
    WarmLanePoolHardDown("warm-lane base absent (pool hard-down) for '3538'; requeue"),
    WarmLaneSoftPressure("warm-lane soft pressure for branch '3538'; requeue"),
    WarmLaneReseedContaminated("warm-lane reseed contaminated for '3538'; requeue"),
    WarmLaneRequeue("bare warm-lane requeue for branch '3538'"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'exc', WARM_LANE_EXCS, ids=[type(e).__name__ for e in WARM_LANE_EXCS],
)
async def test_warm_lane_requeue_repends_the_row(
    tmp_path: Path, exc: WarmLaneRequeue,
):
    """run()'s warm-lane REQUEUED exit leaves the row ``pending``, not stranded.

    Today this path returns REQUEUED with NO status write, so the row stays
    ``in-progress`` (written by ``_setup_worktree_and_artifacts``) and the
    harness slot ``finally`` then nulls the claimant — the exact
    ``(in-progress, NULL claimant)`` strand shape.  After the fix the last
    status row is ``pending``, so ordinary dispatch recovers the task and
    recovery no longer depends on the stranded sweep (which refuses to act
    while any escalation is open).
    """
    sched = FakeScheduler()
    wf = _make_workflow(tmp_path=tmp_path, scheduler=sched)
    wf.git_ops.create_worktree = AsyncMock(side_effect=exc)
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    report = await wf.run()

    # (1) the row is re-pended at exit
    assert sched.statuses[wf.task_id][-1] == 'pending', (
        f'expected the row re-pended before the REQUEUED exit; '
        f'history={sched.statuses[wf.task_id]!r}'
    )
    # (2) the TerminalReport is otherwise UNCHANGED (behaviour-parity guard
    #     for the disposition-table single-sourcing tests)
    disp = classify_failure(exc)
    assert report.outcome == WorkflowOutcome.REQUEUED
    assert report.reason == disp.reason_prefix
    assert report.phase == wf.machine.state
    assert report.blocked_from_phase == wf.machine.state
    assert report.counts_against_requeue_cap == disp.counts_against_requeue_cap
    # (3) no block path taken
    mark_blocked.assert_not_awaited()


@pytest.mark.asyncio
async def test_warm_lane_requeue_on_cancelled_row_returns_cancelled(tmp_path: Path):
    """A cancelled row observed by the re-pend write wins over the requeue intent.

    ``set_task_status('pending')`` raises ``TerminalExitRejection(old_status=
    'cancelled')``; ``run()`` must return CANCELLED (not REQUEUED) and the
    rejection must NOT escape — this clause is a SIBLING of ``_drive()``'s
    ``except SetTaskStatusRejected``, so nothing downstream would catch it.
    """
    sched = FakeScheduler()
    wf = _make_workflow(tmp_path=tmp_path, scheduler=sched)
    wf.git_ops.create_worktree = AsyncMock(
        side_effect=WarmLanePoolExhausted('pool exhausted; requeue')
    )

    real_set = sched.set_task_status

    async def _reject_pending(task_id: str, status: str, **kwargs):
        if status == 'pending':
            # Faithful double: the server refuses BECAUSE the row is already
            # 'cancelled', so the observable row must say so too — otherwise
            # get_status would still report 'in-progress' and run()'s SM-2
            # exit check would fire on a state production never produces.
            sched.statuses.setdefault(task_id, []).append('cancelled')
            raise TerminalExitRejection(
                task_id=task_id, old_status='cancelled',
                target_status='pending', raw='terminal-exit gate',
            )
        await real_set(task_id, status, **kwargs)

    sched.set_task_status = _reject_pending  # type: ignore[method-assign]

    report = await wf.run()

    assert report.outcome == WorkflowOutcome.CANCELLED
    assert wf.machine.state is WorkflowState.CANCELLED


@pytest.mark.asyncio
async def test_warm_lane_requeue_survives_a_dead_repend_write(tmp_path: Path):
    """END-TO-END: a re-pend write that DIES must not cost the REQUEUED report.

    ``Scheduler.set_task_status`` raises a bare ``RuntimeError`` when its
    transient-retry loop is exhausted (scheduler.py:2845) — fused-memory
    restarting / MCP unreachable, the very condition that produces warm-lane
    requeues.  Nothing downstream catches it (this clause is a SIBLING of
    ``_drive()``'s ``except SetTaskStatusRejected`` and ``except Exception``),
    so an escape makes ``run()`` raise: the harness sees no ``TerminalReport``,
    ``_apply_retry_cap`` / ``record_requeue`` never run, and the row is left
    ``in-progress`` for the slot ``finally`` to strand.  The degraded exit must
    therefore carry the SAME bookkeeping as the healthy one.
    """
    exc = WarmLanePoolExhausted("warm-lane pool exhausted for branch '3538'; requeue")
    sched = FakeScheduler()
    wf = _make_workflow(tmp_path=tmp_path, scheduler=sched)
    wf.git_ops.create_worktree = AsyncMock(side_effect=exc)
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    real_set = sched.set_task_status

    async def _die_on_pending(task_id: str, status: str, **kwargs):
        # Only the re-pend write dies — the dispatch 'in-progress' write ahead
        # of create_worktree must still land, or the run never reaches the
        # warm-lane clause at all.
        if status == 'pending':
            raise RuntimeError(
                f'set_task_status({task_id}, {status}) failed after 4 '
                f'transient retries: TimeoutError'
            )
        await real_set(task_id, status, **kwargs)

    sched.set_task_status = _die_on_pending  # type: ignore[method-assign]

    report = await wf.run()  # must NOT raise

    disp = classify_failure(exc)
    assert report.outcome == WorkflowOutcome.REQUEUED
    assert report.reason == disp.reason_prefix
    assert report.phase == wf.machine.state
    assert report.blocked_from_phase == wf.machine.state
    assert report.counts_against_requeue_cap == disp.counts_against_requeue_cap
    mark_blocked.assert_not_awaited()
    # Degraded but no worse than the pre-γ3 floor: the row is left exactly
    # where that code always left it, and SM-2 still passes
    # (``_OUTCOME_ALLOWED['requeued']`` admits IN_PROGRESS today).
    assert sched.statuses[wf.task_id][-1] == 'in-progress'


# ---------------------------------------------------------------------------
# Boundary #12b — the soft-cancel spurious-wakeup fallback re-pends the row
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSoftCancelFallbackRepends:
    """``_handle_soft_cancel``'s three-way decision table, status-write edition.

    Only case 3 (the spurious-wakeup REQUEUED fallback) changes: it must now
    write ``pending`` first, for the same reason as the warm-lane clause —
    returning REQUEUED on an ``in-progress`` row hands the harness slot
    ``finally`` a claimant-less mid-flight row to strand.  Cases 1 and 2 are
    negative controls and must not move.
    """

    def _make(self, tmp_path: Path, *, status: str) -> TaskWorkflow:
        sched = FakeScheduler()
        wf = _make_workflow(tmp_path=tmp_path, scheduler=sched)
        sched.statuses[wf.task_id] = [status]
        return wf

    async def test_spurious_wakeup_repends_before_requeued(self, tmp_path: Path):
        """Case 3: non-terminal + cancel_event NOT set → ``pending`` then REQUEUED."""
        wf = self._make(tmp_path, status='in-progress')
        assert not wf._cancel_event.is_set()

        outcome = await wf._handle_soft_cancel('merge')

        assert outcome == WorkflowOutcome.REQUEUED
        assert wf.scheduler.statuses[wf.task_id][-1] == 'pending', (  # type: ignore[attr-defined]
            'the spurious-wakeup fallback must re-pend before exiting; '
            f'history={wf.scheduler.statuses[wf.task_id]!r}'  # type: ignore[attr-defined]
        )

    async def test_spurious_wakeup_still_requeues_when_the_repend_write_dies(
        self, tmp_path: Path,
    ):
        """Case 3, degraded: a dead re-pend write still returns REQUEUED.

        This method runs from inside ``run()``'s ``except WorkflowCancelled``
        handler and ``run()`` catches nothing else, so an escaping
        ``RuntimeError`` from the exhausted transient-retry loop would take
        ``run()`` down with it.
        """
        wf = self._make(tmp_path, status='in-progress')
        wf.scheduler.set_task_status = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError(
                'set_task_status(3538, pending) failed after 4 transient '
                'retries: TimeoutError'
            )
        )
        assert not wf._cancel_event.is_set()

        outcome = await wf._handle_soft_cancel('merge')  # must not raise

        assert outcome == WorkflowOutcome.REQUEUED

    async def test_soft_cancelled_writes_no_status(self, tmp_path: Path):
        """Negative control, case 2: SOFT_CANCELLED writes NOTHING.

        ``release_workflow`` owns that park (it always follows a
        SOFT_CANCELLED exit with an explicit ``set_task_status``), so a write
        here would double-write and race the human-initiated takeover.
        """
        wf = self._make(tmp_path, status='in-progress')
        wf._cancel_event.set()

        outcome = await wf._handle_soft_cancel('merge')

        assert outcome == WorkflowOutcome.SOFT_CANCELLED
        assert wf.scheduler.statuses[wf.task_id] == ['in-progress']  # type: ignore[attr-defined]

    async def test_terminal_done_writes_no_pending(self, tmp_path: Path):
        """Negative control, case 1: a ``done`` row exits DONE with no write."""
        wf = self._make(tmp_path, status='done')
        wf._cancel_event.set()  # terminal wins over the event

        outcome = await wf._handle_soft_cancel('merge')

        assert outcome == WorkflowOutcome.DONE
        assert wf.scheduler.statuses[wf.task_id] == ['done']  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Boundary #14a — DONE-on-cancelled producer #1 (_handle_soft_cancel)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_soft_cancel_on_cancelled_row_returns_cancelled(tmp_path: Path):
    """A ``cancelled`` row reached via soft cancel exits CANCELLED, not DONE.

    ``_handle_soft_cancel``'s case 1 collapses every member of
    ``TERMINAL_STATUSES`` onto DONE, so a human cancellation is reported as a
    completion.  The machine must also be left in ``WorkflowState.CANCELLED``
    (SM-1 terminal absorption) — which ``_finalise_cancellation`` has already
    entered by the time this runs, so the helper must be idempotent there.
    """
    sched = FakeScheduler()
    wf = _make_workflow(tmp_path=tmp_path, scheduler=sched)
    sched.statuses[wf.task_id] = ['cancelled']
    wf.state = WorkflowState.CANCELLED  # as _finalise_cancellation leaves it

    outcome = await wf._handle_soft_cancel('merge')

    assert outcome == WorkflowOutcome.CANCELLED
    assert wf.machine.state is WorkflowState.CANCELLED
    # No 'pending' write: a terminal row is not requeued.
    assert sched.statuses[wf.task_id] == ['cancelled']


@pytest.mark.asyncio
async def test_soft_cancel_on_done_row_still_returns_done(tmp_path: Path):
    """Negative control: a ``done`` row still exits DONE (unchanged behaviour)."""
    sched = FakeScheduler()
    wf = _make_workflow(tmp_path=tmp_path, scheduler=sched)
    sched.statuses[wf.task_id] = ['done']
    wf.state = WorkflowState.CANCELLED

    outcome = await wf._handle_soft_cancel('merge')

    assert outcome == WorkflowOutcome.DONE


def test_done_outcome_is_not_allowed_on_a_cancelled_row():
    """Why boundary #14a is a live crash, not a mislabelling.

    ``_OUTCOME_ALLOWED['done'] == {DONE}``, so run()'s SM-2 exit check raises
    ``AssertionError`` on a DONE outcome against a ``cancelled`` row; the
    truthful CANCELLED pairing is consistent by construction.  Tally
    correctness follows mechanically — ``Harness._compute_tallies`` counts
    ``report.completed`` as ``outcome == DONE`` only, so a DONE-on-cancelled
    inflated the completed count.
    """
    assert outcome_allows_status(WorkflowOutcome.DONE, 'cancelled') is False
    assert outcome_allows_status(WorkflowOutcome.CANCELLED, 'cancelled') is True


@pytest.mark.asyncio
async def test_run_soft_cancel_on_cancelled_row_reports_cancelled(tmp_path: Path):
    """End-to-end: ``run()`` returns CANCELLED and does not trip its own SM-2.

    Today the soft-cancel path returns DONE against the ``cancelled`` row, and
    ``run()``'s exit check raises
    ``AssertionError: run()-exit SM-2: outcome ... inconsistent with status
    'cancelled'`` — so this is a crash out of ``run()``, not a cosmetic
    mislabel.
    """
    sched = FakeScheduler()
    wf = _make_workflow(tmp_path=tmp_path, scheduler=sched)
    sched.statuses[wf.task_id] = ['cancelled']

    async def _cancelled_drive():
        raise WorkflowCancelled(kind='soft')

    wf._drive = _cancelled_drive  # type: ignore[method-assign]

    report = await wf.run()

    assert report.outcome == WorkflowOutcome.CANCELLED
    assert report.phase is WorkflowState.CANCELLED
    assert wf.machine.state is WorkflowState.CANCELLED
