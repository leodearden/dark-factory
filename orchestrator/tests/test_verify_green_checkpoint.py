"""End-to-end test for the durable verified-green VERIFY checkpoint (task 2752).

Drives ``TaskWorkflow._execute_verify_review_loop`` with a real worktree, a
fully-done plan, and a real ``EventStore`` pre-seeded — under a PRIOR run_id —
with a durable ``workflow_verify`` green at the branch's current tip. On a
fresh dispatch (a NEW run_id on the same db, the cross-restart scenario) the
loop must SKIP ``_verify_debugfix_loop`` entirely: the checkpoint fires, emits
``phase_skipped(verify)``, and (because verify did not run this cycle) does NOT
re-emit ``workflow_verify`` under the new run.

step-7: failing test, written before the checkpoint exists in
``_execute_verify_review_loop`` (RED — verify runs, no phase_skipped, a fresh
workflow_verify row is emitted). step-8 inserts the checkpoint and makes it green.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.artifacts import ReviewAggregation, TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

TASK_ID = '42'
GREEN_TIP = 'tip-xyz'


def _make_e2e_workflow(tmp_path: Path, event_store: EventStore) -> TaskWorkflow:
    """A workflow with a real worktree + fully-done plan + real event store,
    mirroring ``test_suggestion_triage``'s ``_make_e2e_workflow`` but with an
    ``EventStore`` attached so the checkpoint has a durable history to read."""
    assignment = MagicMock()
    assignment.task_id = TASK_ID
    assignment.task = {'id': TASK_ID, 'title': 'Test Task', 'description': 'desc'}

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.git = GitConfig()  # branch_prefix defaults to 'task/'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.max_execute_iterations = 5
    config.inter_iteration_rebase = False
    config.steward_completion_timeout = 300.0

    mcp = MagicMock()
    mcp.url = 'http://localhost:8002'

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=MagicMock(),
        scheduler=MagicMock(),
        briefing=MagicMock(),
        mcp=mcp,
        event_store=event_store,
    )

    wt = tmp_path / 'wt'
    wt.mkdir()
    wf.worktree = wt
    wf.artifacts = TaskArtifacts(wt)
    wf.artifacts.init(TASK_ID, 'Test Task', 'desc', base_commit='deadbeef')

    plan = {
        'task_id': TASK_ID,
        'title': 'Test Task',
        'files': [],
        'analysis': 'Test analysis',
        'prerequisites': [],
        'steps': [
            {
                'id': 'step-1',
                'type': 'impl',
                'description': 'Write code',
                'status': 'done',
                'commit': 'abc123',
            },
        ],
    }
    wf.artifacts.write_plan(plan)
    wf.artifacts.stamp_plan_provenance(wf.session_id)
    return wf


class TestVerifyGreenCheckpointSkipsVerify:
    @pytest.mark.asyncio
    async def test_durable_green_at_tip_skips_verify_and_suppresses_reemit(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / 'runs.db'

        # Pre-seed a DURABLE green under a PRIOR run at the current tip.
        prior = EventStore(db_path, run_id='run-prior')
        prior.emit(
            EventType.workflow_verify,
            task_id=TASK_ID,
            data={'passed': True, 'tip_sha': GREEN_TIP, 'base_sha': 'deadbeef'},
        )

        # Fresh dispatch: a NEW run on the same db (the cross-restart scenario).
        store_now = EventStore(db_path, run_id='run-now')
        wf = _make_e2e_workflow(tmp_path, store_now)

        # Current branch tip matches the durable green's tip → checkpoint hits.
        wf._get_head_commit = AsyncMock(return_value=GREEN_TIP)  # type: ignore[method-assign]
        # Spy: must NOT be awaited when the checkpoint fires.
        wf._verify_debugfix_loop = AsyncMock(return_value=WorkflowOutcome.DONE)  # type: ignore[method-assign]
        wf._review = AsyncMock(return_value=ReviewAggregation(  # type: ignore[method-assign]
            has_blocking_issues=False,
            blocking_issues=[],
            suggestions=[],
            reviews={'analyst': {}},
        ))
        wf._suggestions_in_scope = lambda s: []  # type: ignore[method-assign]
        wf._write_suggestions_to_memory = AsyncMock()  # type: ignore[method-assign]

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE

        # (1) verify was skipped — the debugfix loop was never awaited.
        wf._verify_debugfix_loop.assert_not_awaited()

        # (2) the honest skip signal was emitted: phase_skipped(verify).
        skipped = store_now.fetch_events_by_type_all_runs(
            EventType.phase_skipped, task_id=TASK_ID
        )
        assert any(row['phase'] == 'verify' for row in skipped), skipped

        # (3) NO new workflow_verify row under the current run (suppressed) —
        #     the run-scoped reader on run-now sees none.
        assert store_now.fetch_events_by_type(EventType.workflow_verify) == []


class TestVerifyCheckpointTipFetchFailSafe:
    @pytest.mark.asyncio
    async def test_raising_get_head_commit_falls_through_to_normal_verify(
        self, tmp_path: Path
    ) -> None:
        # Regression pin (reviewer robustness finding, task 2752 amendment): a
        # RAISING _get_head_commit at the checkpoint site (e.g. git not on PATH
        # → FileNotFoundError, or a None worktree) must NOT crash
        # _execute_verify_review_loop.  It must fall through to the normal
        # verify path (the tip cannot be read → checkpoint miss →
        # _verify_debugfix_loop runs), exactly as the checkpoint comment
        # promises — never propagate out of the loop.
        db_path = tmp_path / 'runs.db'

        # A durable prior-run green DOES exist at GREEN_TIP, but it must be
        # ignored: the checkpoint can never read the current tip to compare.
        prior = EventStore(db_path, run_id='run-prior')
        prior.emit(
            EventType.workflow_verify,
            task_id=TASK_ID,
            data={'passed': True, 'tip_sha': GREEN_TIP, 'base_sha': 'deadbeef'},
        )

        store_now = EventStore(db_path, run_id='run-now')
        wf = _make_e2e_workflow(tmp_path, store_now)

        # 1st call (checkpoint site) RAISES; 2nd call (post-verify capture)
        # returns the tip so the passing-verify fall-through proceeds normally.
        wf._get_head_commit = AsyncMock(  # type: ignore[method-assign]
            side_effect=[FileNotFoundError('git'), GREEN_TIP]
        )
        wf._verify_debugfix_loop = AsyncMock(return_value=WorkflowOutcome.DONE)  # type: ignore[method-assign]
        wf._review = AsyncMock(return_value=ReviewAggregation(  # type: ignore[method-assign]
            has_blocking_issues=False,
            blocking_issues=[],
            suggestions=[],
            reviews={'analyst': {}},
        ))
        wf._suggestions_in_scope = lambda s: []  # type: ignore[method-assign]
        wf._write_suggestions_to_memory = AsyncMock()  # type: ignore[method-assign]

        # Must NOT raise despite the checkpoint tip fetch failing.
        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE

        # verify RAN (the checkpoint missed because the tip could not be read).
        wf._verify_debugfix_loop.assert_awaited_once()

        # NO phase_skipped(verify) — nothing was skipped this cycle.
        skipped = store_now.fetch_events_by_type_all_runs(
            EventType.phase_skipped, task_id=TASK_ID
        )
        assert not any(row['phase'] == 'verify' for row in skipped), skipped
