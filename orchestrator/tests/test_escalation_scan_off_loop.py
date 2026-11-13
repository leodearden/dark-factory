"""Archive-inclusive ``EscalationQueue.get_by_task`` scans run OFF the event loop (task 4567).

The scheduler, the merge worker, ``/health`` and the in-process escalation
server all share the orchestrator's loop, so a synchronous scan of the whole
archive inside one of these coroutines stalls every one of them.  Each test
pins THREAD IDENTITY, never timing (the stance of
``escalation/tests/test_pending_scan_off_loop.py``), and also asserts the
site's result, so the hop cannot pass by stubbing the scan out.
"""

from __future__ import annotations

import ast
import asyncio
import threading
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import AsyncMock, MagicMock

import pytest
import test_deterministic_runner as _det
import test_harness_deterministic_recon_sweep as _recon
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
from orchestrator.deterministic_runner import DETERMINISTIC_AGENT_ROLE, DeterministicRunner
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import WorkflowOutcome


class _Scan(NamedTuple):
    thread: int
    status: str | None
    agent_role: str | None


class _ThreadProbeQueue(EscalationQueue):
    """A REAL queue recording the thread and filters of every task scan.

    Delegates through ``super()``, so each probed call still does the real glob
    and JSON parse.  The status is kept because only a non-``'pending'`` scan
    reaches the archive.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.scans: list[_Scan] = []

    def get_by_task(self, task_id, status=None, level=None, agent_role=None):
        self.scans.append(_Scan(threading.get_ident(), status, agent_role))
        return super().get_by_task(task_id, status, level, agent_role)

    def archive_inclusive_threads(self) -> list[int]:
        return [scan.thread for scan in self.scans if scan.status != 'pending']


class _ResolvesDuringResolvedScan(_ThreadProbeQueue):
    """Resolves one record as a ``status='resolved'`` scan begins, as a concurrent resolver can."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.resolve_during_scan: str | None = None

    def get_by_task(self, task_id, status=None, level=None, agent_role=None):
        if status == 'resolved' and self.resolve_during_scan is not None:
            esc_id, self.resolve_during_scan = self.resolve_during_scan, None
            self.resolve(esc_id, 'answered mid-scan', resolved_by='steward')
        return super().get_by_task(task_id, status, level, agent_role)


def _assert_archive_scans_ran_off_loop(probe: _ThreadProbeQueue) -> None:
    threads = probe.archive_inclusive_threads()
    assert threads, f'no archive-inclusive get_by_task scan was recorded: {probe.scans}'
    # The test body runs on the event-loop thread.
    assert threading.get_ident() not in threads, (
        f'an archive-inclusive get_by_task scan ran on the event loop: {probe.scans}'
    )


def _assert_own_role_archive_scan_ran_off_loop(probe: _ThreadProbeQueue) -> None:
    assert any(
        scan.status != 'pending' and scan.agent_role == DETERMINISTIC_AGENT_ROLE
        for scan in probe.scans
    ), f'no role-scoped archive-inclusive scan was recorded: {probe.scans}'
    _assert_archive_scans_ran_off_loop(probe)


def _file(
    queue: EscalationQueue,
    task_id: str,
    *,
    category: str,
    granted_files: list[str] | None = None,
    agent_role: str = 'implementer',
    level: int = 0,
) -> Escalation:
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role=agent_role,
        severity='blocking',
        category=category,
        summary=f'{category} blocker',
        granted_files=list(granted_files or []),
        level=level,
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
            scan.status for scan in probe.scans if scan.thread != threading.get_ident()
        }
        assert {'resolved', None} <= off_loop_statuses, probe.scans
        _assert_archive_scans_ran_off_loop(probe)

    async def test_record_pending_at_entry_but_resolved_during_the_scan_answers_this_round(
        self,
        config: OrchestratorConfig,
        git_ops: GitOps,
        task_assignment: TaskAssignment,
        tmp_path: Path,
    ):
        queue = _ResolvesDuringResolvedScan(tmp_path / 'queue')
        worktree = await _make_advanced_worktree(git_ops, task_assignment.task_id)
        workflow, _ = _build_workflow(config, git_ops, task_assignment, queue, worktree)
        earlier = _file(queue, task_assignment.task_id, category='task_failure')
        queue.resolve(earlier.id, 'answer from an earlier round', resolved_by='steward')
        open_at_entry = _file(queue, task_assignment.task_id, category='task_failure')
        queue.resolve_during_scan = open_at_entry.id

        carried = await asyncio.wait_for(workflow._wait_for_resolution(), 10)

        assert queue.resolve_during_scan is None, 'the entry resolved-scan never ran'
        assert {c.escalation_id: c.answered_this_round for c in carried} == {
            earlier.id: False, open_at_entry.id: True,
        }

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


@pytest.mark.asyncio
class TestDeterministicRunnerScansRunOffLoop:
    async def test_pure_gate_resume_proof_is_read_off_loop(self, tmp_path: Path):
        probe = _ThreadProbeQueue(tmp_path)
        gate = _file(
            probe, '99', category='milestone_gate',
            agent_role=DETERMINISTIC_AGENT_ROLE, level=2,
        )
        probe.resolve(gate.id, 'human approved the gate', resolved_by='human')
        task = _det._gate_task(task_id='99', gate_escalated_at='2026-06-23T12:00:00+00:00')
        scheduler = _det._mock_scheduler(task)
        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=probe)

        outcome = await runner.run(_det._make_assignment(task))

        assert outcome == WorkflowOutcome.DONE
        scheduler.set_task_status.assert_awaited_once_with(
            '99',
            'done',
            done_provenance={
                'kind': 'deterministic-gate',
                'note': 'pure gate resolved',
                'escalation_id': gate.id,
            },
        )
        _assert_own_role_archive_scan_ran_off_loop(probe)

    async def test_deploy_resume_proof_is_read_off_loop(self, tmp_path: Path):
        probe = _ThreadProbeQueue(tmp_path)
        failure = _file(
            probe, '600', category='infra_issue',
            agent_role=DETERMINISTIC_AGENT_ROLE, level=2,
        )
        probe.resolve(failure.id, 'human verified the unit manually')
        task = _det._deploy_task(task_id='600', before_done_ran_at='2026-06-23T10:00:00+00:00')
        scheduler = _det._mock_scheduler(task)
        script_runner = AsyncMock(return_value=(0, 'ok'))
        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=probe,
            unit_inspector=AsyncMock(return_value=_det._BASELINE_UNIT_STATE),
            script_runner=script_runner,
        )

        await runner.run(_det._make_assignment(task))

        script_runner.assert_not_awaited()
        assert [c.args[1] for c in scheduler.set_task_status.await_args_list] == ['done']
        _assert_own_role_archive_scan_ran_off_loop(probe)


@pytest.mark.asyncio
async def test_recon_sweep_gate_proof_is_read_off_loop(tmp_path: Path):
    h = _recon._make_recon_harness()
    h.event_store = None
    probe = _ThreadProbeQueue(tmp_path)
    h._escalation_queue = probe
    tid = 'tid-gate'
    task = {
        'id': tid, 'status': 'blocked',
        'title': 'Human gate 5330', 'description': 'Approve reify 5330.',
        'metadata': {
            'task_kind': 'deterministic',
            'always_escalates': True,
            'before_done': None,
            'gate_escalated_at': '2026-07-01T00:05:00+00:00',
            'gate_options': ['approve', 'reject'],
        },
    }
    h.scheduler.get_tasks = AsyncMock(return_value=[task])
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))

    await h._run_deterministic_recon_sweep()

    refiled = probe.get_by_task(tid, status='pending')
    assert [(e.category, e.level) for e in refiled] == [('milestone_gate', 2)]
    _assert_own_role_archive_scan_ran_off_loop(probe)


_GUARDED_MODULES = tuple(
    Path(__file__).resolve().parents[1] / 'src' / 'orchestrator' / name
    for name in ('workflow.py', 'deterministic_runner.py', 'harness.py')
)


class _InlineScan(NamedTuple):
    function: str
    lineno: int


def _status_argument(call: ast.Call) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == 'status':
            return keyword.value
    return call.args[1] if len(call.args) >= 2 else None


def _is_pending_literal(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value == 'pending'


def _get_by_task_census(source: str) -> tuple[int, list[_InlineScan]]:
    """Count ``get_by_task`` references and list archive-inclusive calls inlined in a coroutine.

    A call is inlined in a coroutine when its NEAREST enclosing def is async.
    ``asyncio.to_thread(q.get_by_task, ...)`` passes the bound method rather
    than calling it, so it counts as a reference and is never flagged.
    """
    references = 0
    inline: list[_InlineScan] = []

    def visit(node: ast.AST, enclosing: ast.FunctionDef | ast.AsyncFunctionDef | None) -> None:
        nonlocal references
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Attribute) and child.attr == 'get_by_task':
                references += 1
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == 'get_by_task'
                and isinstance(enclosing, ast.AsyncFunctionDef)
                and not _is_pending_literal(_status_argument(child))
            ):
                inline.append(_InlineScan(enclosing.name, child.lineno))
            is_def = isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
            visit(child, child if is_def else enclosing)

    visit(ast.parse(source), None)
    return references, inline


@pytest.mark.parametrize(('source', 'flagged'), [
    ('async def f(q):\n    q.get_by_task(t)\n', [_InlineScan('f', 2)]),
    ('async def f(q):\n    q.get_by_task(t, status="resolved")\n', [_InlineScan('f', 2)]),
    ('async def f(q, s):\n    q.get_by_task(t, s)\n', [_InlineScan('f', 2)]),
    ('async def f(q):\n    q.get_by_task(t, status="pending")\n', []),
    ('async def f(q):\n    q.get_by_task(t, "pending", level=0)\n', []),
    ('async def f(q):\n    await asyncio.to_thread(q.get_by_task, t)\n', []),
    ('def f(q):\n    q.get_by_task(t)\n', []),
    ('async def f(q):\n    def g():\n        q.get_by_task(t)\n', []),
])
def test_inline_scan_detector_flags_only_archive_inclusive_coroutine_calls(source, flagged):
    _, inline = _get_by_task_census(source)

    assert inline == flagged


def test_no_archive_inclusive_get_by_task_runs_inline_in_a_coroutine():
    """No archive-inclusive ``get_by_task`` call sits lexically inside an ``async def``.

    Blind spot: a SYNC helper called from a coroutine is invisible here, which
    is why ``_collect_granted_files`` and ``_escalate_suggestions`` are async.
    """
    offenders: list[str] = []
    for path in _GUARDED_MODULES:
        references, inline = _get_by_task_census(path.read_text())
        assert references, f'{path.name}: no get_by_task reference found; the census is vacuous'
        offenders += [
            f'{path.name}:{scan.lineno} in {scan.function}()' for scan in inline
        ]

    assert not offenders, (
        'archive-inclusive get_by_task called inline in a coroutine; wrap it in '
        f'await asyncio.to_thread(queue.get_by_task, ...): {offenders}'
    )
