"""The merge_retry_pending clean-merge fast path passes the REAL merge-entry gate.

``workflow.py::TaskWorkflow._resume_merge_retry_if_pending`` jumps straight to
the merge phase when the stamped branch head still matches and still merges
cleanly.  This module drives that jump end to end through the real
merge-entry scope invariant and asserts it reaches the merge queue without
filing a gating escalation (task 4111; regression pin for esc-3388-9, whose
fix is commit 13679df023).
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from _workflow_helpers import _init_git_repo, _make
from escalation.queue import EscalationQueue

from orchestrator.workflow import WorkflowOutcome

PLAN_FILES = ['pkg/src/pkg/mod.py']
MERGE_SHA = 'c' * 40


@pytest.mark.asyncio
async def test_clean_merge_resume_reaches_the_queue_without_a_scope_escalation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
):
    worktree = tmp_path / 'wt'
    _init_git_repo(worktree)
    f = _make(worktree=worktree, project_root=tmp_path / 'proj', branch_on_main=False)
    wf = f.wf
    head = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=worktree,
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    f.artifacts.write_plan({
        'task_id': wf.task_id, 'title': 'T', 'files': list(PLAN_FILES), 'steps': [],
    })
    wf.task['metadata'] = {
        'files': list(PLAN_FILES),
        'merge_retry_pending': {
            'branch_head': head,
            'base_sha': 'mainsha123',
            'resolved_at': '2026-10-31T00:00:00+00:00',
        },
    }
    wf.scheduler.get_task = AsyncMock(
        return_value={'id': wf.task_id, 'metadata': {'files': list(PLAN_FILES)}},
    )

    wf.escalation_queue = queue = EscalationQueue(tmp_path / 'escalations')
    wf.config.unblock_auto.enabled = False
    wf.config.max_merge_retries = 1
    wf.config.max_pre_merge_retries = 1
    wf.config.merge_train_former_enabled = False
    wf.config.timeouts.steward = 0.1
    wf.config.steward_completion_timeout = 0.1

    wf.git_ops.merge_tree_conflicts = AsyncMock(
        return_value=SimpleNamespace(clean=True, conflicted_paths=[]),
    )
    wf.git_ops.rebase_onto_main = AsyncMock(return_value=False)
    wf.git_ops.rebind_branch_to_head = AsyncMock(return_value=True)
    wf.git_ops.get_merge_commit_diff_files = AsyncMock(return_value=([], None))

    wf.merge_queue = merge_queue = asyncio.Queue()
    requests = []

    async def _merge_worker() -> None:
        req = await merge_queue.get()
        requests.append(req)
        req.result.set_result(SimpleNamespace(
            status='done', merge_sha=MERGE_SHA,
            superseded_by=None, reason='', landed_via_chain=None,
        ))

    worker = asyncio.create_task(_merge_worker())
    try:
        outcome = await asyncio.wait_for(
            wf._resume_merge_retry_if_pending('task/50'), 20,
        )
    finally:
        worker.cancel()

    assert [esc.summary for esc in queue.get_by_task(wf.task_id)] == []
    swallowed_attribute_errors = [
        record.getMessage() for record in caplog.records
        if record.exc_info and isinstance(record.exc_info[1], AttributeError)
    ]
    assert swallowed_attribute_errors == []
    assert [r.task_files for r in requests] == [PLAN_FILES]
    assert outcome == WorkflowOutcome.DONE
    f.mark_done.assert_awaited_once_with(wf.task_id, kind='merged', sha=MERGE_SHA)
