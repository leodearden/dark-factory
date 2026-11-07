"""Archive-inclusive ``EscalationQueue.get_by_task`` scans run OFF the event loop (task 4567).

The scheduler, the merge worker, ``/health`` and the in-process escalation
server all share the orchestrator's loop, so a synchronous scan of the whole
archive inside one of these coroutines stalls every one of them.  Each test
pins THREAD IDENTITY, never timing (the stance of
``escalation/tests/test_pending_scan_off_loop.py``), and also asserts the
site's result, so the hop cannot pass by stubbing the scan out.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import test_suggestion_triage as _triage
import test_workflow_infra_thrash as _infra
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from test_workflow_escalated_steward_stall import _build_workflow, _make_advanced_worktree
from test_workflow_escalated_steward_stall import config as config
from test_workflow_escalated_steward_stall import git_ops as git_ops
from test_workflow_escalated_steward_stall import git_repo as git_repo
from test_workflow_escalated_steward_stall import task_assignment as task_assignment

from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import TaskAssignment


class _ThreadProbeQueue(EscalationQueue):
    """A REAL queue recording the thread and status filter of every task scan.

    Delegates through ``super()``, so each probed call still does the real glob
    and JSON parse.  The status is kept because only a non-``'pending'`` scan
    reaches the archive.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.scans: list[tuple[int, str | None]] = []

    def get_by_task(self, task_id, status=None, level=None, agent_role=None):
        self.scans.append((threading.get_ident(), status))
        return super().get_by_task(task_id, status, level, agent_role)

    def archive_inclusive_threads(self) -> list[int]:
        return [thread for thread, status in self.scans if status != 'pending']


def _assert_archive_scans_ran_off_loop(probe: _ThreadProbeQueue) -> None:
    threads = probe.archive_inclusive_threads()
    assert threads, f'no archive-inclusive get_by_task scan was recorded: {probe.scans}'
    # The test body runs on the event-loop thread.
    assert threading.get_ident() not in threads, (
        f'an archive-inclusive get_by_task scan ran on the event loop: {probe.scans}'
    )


def _file(
    queue: EscalationQueue,
    task_id: str,
    *,
    category: str,
    granted_files: list[str] | None = None,
) -> Escalation:
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role='implementer',
        severity='blocking',
        category=category,
        summary=f'{category} blocker',
        granted_files=list(granted_files or []),
    )
    queue.submit(esc)
    return esc


@pytest.mark.asyncio
class TestWorkflowScansRunOffLoop:
    async def test_infra_resume_thrash_classifies_off_loop(self, tmp_path: Path):
        probe = _ThreadProbeQueue(tmp_path / 'queue')
        infra = _file(probe, '99', category='infra_issue')
        probe.resolve(infra.id, 'resume', resolved_by='steward')
        f = _infra._make(
            metadata={
                'retry_ledger': {
                    'consecutive_infra_resume_failures': 1,
                    'last_infra_resume_iteration_count': 5,
                },
            },
            iteration_log=[{'iteration': i} for i in range(5)],
        )
        f.wf.escalation_queue = probe

        outcome = await f.wf._check_infra_resume_thrash()

        assert outcome is None
        ledger = _infra._persisted_metadata(f.update_task)['retry_ledger']
        assert ledger['consecutive_infra_resume_failures'] == 2
        _assert_archive_scans_ran_off_loop(probe)

    async def test_granted_files_are_collected_off_loop(self, tmp_path: Path):
        probe = _ThreadProbeQueue(tmp_path / 'queue')
        grant = _file(probe, '99', category='scope_violation')
        probe.resolve(
            grant.id, 'resume', resolved_by='steward', granted_files=['a.py', 'b.py'],
        )
        _file(probe, '99', category='scope_violation', granted_files=['c.py'])
        f = _infra._make()
        f.wf.escalation_queue = probe

        granted = await f.wf._collect_granted_files()

        assert granted == ['a.py', 'b.py']
        _assert_archive_scans_ran_off_loop(probe)

    async def test_wait_for_resolution_reads_resolutions_off_loop(
        self,
        config: OrchestratorConfig,
        git_ops: GitOps,
        task_assignment: TaskAssignment,
        tmp_path: Path,
    ):
        probe = _ThreadProbeQueue(tmp_path / 'queue')
        worktree = await _make_advanced_worktree(git_ops, task_assignment.task_id)
        workflow, _ = _build_workflow(config, git_ops, task_assignment, probe, worktree)
        first = _file(probe, task_assignment.task_id, category='task_failure')
        second = _file(probe, task_assignment.task_id, category='task_failure')
        probe.resolve(first.id, 'answer to first', resolved_by='steward')
        probe.resolve(second.id, 'answer to second', resolved_by='steward')

        carried = await asyncio.wait_for(workflow._wait_for_resolution(), 10)

        assert [c.escalation_id for c in carried] == [first.id, second.id]
        off_loop_statuses = {
            status for thread, status in probe.scans if thread != threading.get_ident()
        }
        assert {'resolved', None} <= off_loop_statuses, probe.scans
        _assert_archive_scans_ran_off_loop(probe)

    async def test_review_suggestions_dedup_reads_off_loop(self, tmp_path: Path):
        probe = _ThreadProbeQueue(tmp_path / 'queue')
        wf = _triage._make_workflow(escalation_queue=probe)
        wf.state = MagicMock(value='review')
        suggestion = {
            'reviewer': 'test_analyst', 'severity': 'suggestion',
            'location': 'src/foo.py:10', 'category': 'coverage',
            'description': 'Missing edge case', 'suggested_fix': 'Add test',
        }

        await wf._escalate_suggestions(_triage._fake_reviews([suggestion]))

        filed = probe.get_by_task('42', status='pending')
        assert [e.category for e in filed] == ['review_suggestions']
        _assert_archive_scans_ran_off_loop(probe)
