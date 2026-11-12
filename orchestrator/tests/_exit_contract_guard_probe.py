"""The tests that test_exit_contract.py runs in a nested pytest session.

Not collected by the suite (no ``test_`` prefix).  The first test leaks a task
whose ``finally:`` records a run()-exit would-violate, so the verdict is logged
only while conftest.py's ``_drain_leaked_tasks`` cancels the leak.  The second
test leaks nothing and must not be blamed for it.
"""
import asyncio

import pytest

from orchestrator.exit_contract import judge_exit, record_exit_verdict
from orchestrator.workflow_types import WorkflowOutcome, WorkflowState

_KEEP: list[asyncio.Task] = []


async def _a_run_that_exits_blocked_over_a_done_row_when_cancelled() -> None:
    try:
        await asyncio.Event().wait()
    finally:
        record_exit_verdict(
            judge_exit(
                outcome=WorkflowOutcome.BLOCKED,
                report_phase=WorkflowState.BLOCKED,
                machine_state=WorkflowState.BLOCKED,
                status_row='done',
                write_failure=None,
            ),
            task_id='probe',
            enforce=False,
            event_store=None,
            escalation_queue=None,
            worktree=None,
            filing_claimant_run_id=None,
        )


@pytest.mark.asyncio
async def test_leaks_a_run_that_exits_while_drained() -> None:
    _KEEP.append(asyncio.get_running_loop().create_task(
        _a_run_that_exits_blocked_over_a_done_row_when_cancelled(),
        name='leaked-run',
    ))
    await asyncio.sleep(0)


def test_runs_after_the_leak() -> None:
    pass
