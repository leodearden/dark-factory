"""An architect that escalates instead of planning blocks behind an L1 (RC2).

When the architect files a blocking L0 through plan-tools MCP and exits
without writing ``.task/plan.json`` or any rejection artifact, ``_plan()``
promotes every L0 it filed, verbatim, to a new level-1 record and blocks
the task behind them. The L0s are dismissed naming their L1s, so the
finding a human must act on is open exactly once, and the stranded-blocked
sweep's own veto keeps the task from being re-pended into the same loop.

An architect that filed only ``escalate_info`` has not escalated in lieu of
a plan: the no-plan handler (with its cycle counters) runs instead, as it
does when the architect filed nothing new.

Each promotion stands alone. An L0 whose L1 cannot be written stays pending,
named in a WARNING, and does not stop the others. The task blocks behind
whichever blocking L0s did promote, and falls through to the no-plan handler
when none did.

These tests drive ``_plan()`` against a real ``EscalationQueue`` and the
real ``_mark_blocked`` / ``_handle_no_plan_failure``. The architect
subprocess (``_invoke``) is the stub, and it files its L0s the way
escalate_blocker / escalate_info would.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.cli_invoke import AgentResult

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.recovery_pins import records_pin_blocked_recovery
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

TASK_ID = '917'
BLOCKING_SUMMARY = 'Leo decision needed: RECOMMEND KEEP max_concurrent_tasks=24'
BLOCKING_DETAIL = (
    'Dispatch latency is flat at 24 slots.\n'
    'Lowering the cap trades throughput for nothing measurable.\n'
    'Hypothesis: the latency is merge-lane bound, not slot bound.'
)
SECOND_BLOCKING_SUMMARY = 'Leo decision needed: docs-only merges skip the full gate'
INFO_SUMMARY = 'Noted: helpers.foo is duplicated in two modules'


@dataclass
class _Fixture:
    wf: TaskWorkflow
    queue: EscalationQueue
    scheduler: MagicMock
    invoke: AsyncMock


def _architect_l0(
    queue: EscalationQueue, *, severity: str, summary: str, detail: str,
) -> Escalation:
    return Escalation(
        id=queue.make_id(TASK_ID),
        task_id=TASK_ID,
        agent_role='architect',
        severity=severity,
        category='design_concern',
        summary=summary,
        detail=detail,
        level=0,
        filing_claimant_run_id='architect-run/session/pid=1',
    )


def _blocking_l0(
    queue: EscalationQueue, summary: str = BLOCKING_SUMMARY,
) -> Escalation:
    return _architect_l0(
        queue, severity='blocking', summary=summary, detail=BLOCKING_DETAIL,
    )


def _info_l0(queue: EscalationQueue) -> Escalation:
    return _architect_l0(
        queue, severity='info',
        summary=INFO_SUMMARY, detail='Both copies are identical today.',
    )


class _QueueRefusingL1(EscalationQueue):
    """A real queue whose disk refuses the L1 promoted from one summary."""

    def __init__(self, queue_dir: Path, refused_summary: str):
        super().__init__(queue_dir)
        self.refused_summary = refused_summary

    def submit(self, escalation: Escalation) -> str:
        if escalation.level == 1 and escalation.summary == self.refused_summary:
            raise OSError(f'disk refused {escalation.id}')
        return super().submit(escalation)


def _make(
    tmp_path: Path,
    *,
    queue: EscalationQueue | None = None,
    event_store: EventStore | None = None,
) -> _Fixture:
    """A TaskWorkflow ready for ``_plan()``, backed by a real escalation queue."""
    assignment = MagicMock()
    assignment.task_id = TASK_ID
    assignment.task = {
        'id': TASK_ID, 'title': 'T', 'description': 'd', 'metadata': {},
    }
    assignment.modules = ['mod_a']

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = tmp_path / 'proj'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.unblock_auto.enabled = False
    config.git.branch_prefix = 'task/'

    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.get_status = AsyncMock(return_value='in-progress')
    scheduler.get_task = AsyncMock(return_value=None)

    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value='currentmain')

    queue = queue or EscalationQueue(tmp_path / 'escalations')
    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        escalation_queue=queue,
        event_store=event_store,
    )

    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    artifacts = TaskArtifacts(worktree)
    artifacts.init(TASK_ID, 'T', 'd', base_commit='oldbase')
    wf.artifacts = artifacts
    wf.worktree = worktree

    invoke = AsyncMock(return_value=AgentResult(
        success=True, output='filed L0', cost_usd=0.50,
        duration_ms=10_000, turns=8,
    ))
    wf._invoke = invoke  # type: ignore[method-assign]
    wf.briefing.build_architect_prompt = AsyncMock(return_value='prompt')
    wf.briefing.build_revalidation_prompt = AsyncMock(return_value='prompt')

    return _Fixture(wf=wf, queue=queue, scheduler=scheduler, invoke=invoke)


def _architect_files(f: _Fixture, *records: Escalation) -> None:
    """The architect run submits *records*, as escalate_blocker/_info would."""
    result = f.invoke.return_value

    def _run(*_args: object, **_kwargs: object) -> AgentResult:
        for record in records:
            f.queue.submit(record)
        return result

    f.invoke.side_effect = _run


def _open_handoffs(queue: EscalationQueue) -> list[Escalation]:
    return [e for e in queue.get_by_task(TASK_ID, status='pending') if e.level >= 1]


def _pending_l0s(queue: EscalationQueue) -> list[Escalation]:
    return [e for e in queue.get_by_task(TASK_ID, status='pending') if e.level == 0]


def _l1_carrying(handoffs: Sequence[Escalation], summary: str) -> Escalation:
    matches = [e for e in handoffs if e.summary == summary]
    assert len(matches) == 1, [(e.id, e.summary) for e in handoffs]
    return matches[0]


def _sweep_would_leave_it_blocked(queue: EscalationQueue) -> bool:
    return records_pin_blocked_recovery(
        TASK_ID, queue.get_by_task(TASK_ID, status='pending'), live_claimant=False,
    )


def _no_plan_handler_filed(queue: EscalationQueue) -> bool:
    return any('no plan.json produced' in e.summary for e in _open_handoffs(queue))


def _assert_blocked_through_no_plan_handler(
    f: _Fixture, *, not_promoted: str | None = None,
) -> None:
    handoffs = _open_handoffs(f.queue)
    assert not_promoted not in [e.summary for e in handoffs]
    assert _no_plan_handler_filed(f.queue), [e.summary for e in handoffs]


def _warned_about(caplog: pytest.LogCaptureFixture, esc_id: str) -> bool:
    return any(
        r.levelno == logging.WARNING and esc_id in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_blocking_l0_is_promoted_and_pins_the_block(tmp_path: Path):
    f = _make(tmp_path)
    l0 = _blocking_l0(f.queue)
    _architect_files(f, l0)

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.BLOCKED
    handoffs = _open_handoffs(f.queue)
    assert len(handoffs) == 1, [(e.id, e.summary) for e in handoffs]
    l1 = handoffs[0]
    assert l1.summary == BLOCKING_SUMMARY
    assert BLOCKING_DETAIL in l1.detail
    assert _pending_l0s(f.queue) == []
    dismissed = f.queue.get(l0.id)
    assert dismissed is not None
    assert dismissed.status == 'dismissed'
    assert dismissed.resolution is not None and l1.id in dismissed.resolution
    assert f.scheduler.set_task_status.await_args_list[-1].args == (TASK_ID, 'blocked')
    assert _sweep_would_leave_it_blocked(f.queue)


@pytest.mark.asyncio
async def test_every_new_l0_is_promoted_keeping_its_severity(tmp_path: Path):
    f = _make(tmp_path)
    _architect_files(f, _blocking_l0(f.queue), _info_l0(f.queue))

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.BLOCKED
    handoffs = _open_handoffs(f.queue)
    assert len(handoffs) == 2, [(e.id, e.summary) for e in handoffs]
    assert _l1_carrying(handoffs, BLOCKING_SUMMARY).severity == 'blocking'
    assert _l1_carrying(handoffs, INFO_SUMMARY).severity == 'info'
    assert _pending_l0s(f.queue) == []
    assert _sweep_would_leave_it_blocked(f.queue)


@pytest.mark.asyncio
async def test_each_promoted_l1_is_announced_once(tmp_path: Path):
    store = EventStore(tmp_path / 'runs.db', run_id='run-rc2')
    f = _make(tmp_path, event_store=store)
    _architect_files(f, _blocking_l0(f.queue), _info_l0(f.queue))

    await f.wf._plan()

    handoffs = _open_handoffs(f.queue)
    assert len(handoffs) == 2, [(e.id, e.summary) for e in handoffs]
    announced = sorted(
        (row['data']['escalation_id'], row['data']['level'])
        for row in store.fetch_events_by_type(EventType.escalation_created)
    )
    assert announced == sorted((e.id, 1) for e in handoffs)


@pytest.mark.asyncio
async def test_info_only_filing_goes_through_the_no_plan_handler(tmp_path: Path):
    f = _make(tmp_path)
    _architect_files(f, _info_l0(f.queue))

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.BLOCKED
    _assert_blocked_through_no_plan_handler(f, not_promoted=INFO_SUMMARY)


@pytest.mark.asyncio
async def test_l0_pending_before_the_architect_ran_is_not_new(tmp_path: Path):
    f = _make(tmp_path)
    f.queue.submit(_blocking_l0(f.queue))

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.BLOCKED
    _assert_blocked_through_no_plan_handler(f, not_promoted=BLOCKING_SUMMARY)


@pytest.mark.asyncio
async def test_no_l0_goes_through_the_no_plan_handler(tmp_path: Path):
    f = _make(tmp_path)

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.BLOCKED
    _assert_blocked_through_no_plan_handler(f)


@pytest.mark.asyncio
async def test_an_l0_that_fails_to_promote_does_not_stop_the_others(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
):
    queue = _QueueRefusingL1(tmp_path / 'escalations', BLOCKING_SUMMARY)
    f = _make(tmp_path, queue=queue)
    refused = _blocking_l0(queue)
    _architect_files(f, refused, _blocking_l0(queue, SECOND_BLOCKING_SUMMARY))

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.BLOCKED
    handoffs = _open_handoffs(queue)
    assert [e.summary for e in handoffs] == [SECOND_BLOCKING_SUMMARY]
    assert not _no_plan_handler_filed(queue)
    assert _warned_about(caplog, refused.id)
    assert _sweep_would_leave_it_blocked(queue)


@pytest.mark.asyncio
async def test_no_blocking_l0_promoted_falls_through_to_the_no_plan_handler(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
):
    queue = _QueueRefusingL1(tmp_path / 'escalations', BLOCKING_SUMMARY)
    f = _make(tmp_path, queue=queue)
    refused = _blocking_l0(queue)
    _architect_files(f, refused, _info_l0(queue))

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.BLOCKED
    _assert_blocked_through_no_plan_handler(f, not_promoted=BLOCKING_SUMMARY)
    assert _warned_about(caplog, refused.id)
    assert _sweep_would_leave_it_blocked(queue)
