"""The VERIFY/DEBUGFIX loop's 5xx requeue (PRD task ι, boundary row 14).

``plans/server-side-api-error-handling-prd.md`` contract C2: a debugger killed
by a provider 5xx before doing any work must send the task back to the queue
on the FIRST such result, exactly like the execute loop does (task γ), rather
than burning the verify budget into a ``Verification attempts exhausted``
block with an L0 and a steward.

Every test drives the public seam ``TaskWorkflow.run()`` on a real git repo,
with the agent CLI (``invoke_agent``) and the verify subprocess
(``run_scoped_verification``) faked at the module boundary.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _api_outage_harness import incident_529_result, sigkill_wedge_result
from _workflow_helpers import (
    AgentStub,
    FakeScheduler,
    _build_workflow,
    _derive_meta_root_like_production,
    _init_repo,
)

from orchestrator.agents.invoke import AgentResult
from orchestrator.config import GitConfig, OrchestratorConfig, SandboxConfig
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import (
    TaskAssignment,
    TerminalExitRejection,
    is_transient_api_requeue,
)
from orchestrator.task_status import TERMINAL_STATUSES
from orchestrator.verify import VerifyResult
from orchestrator.verify_categories import FailureCategory
from orchestrator.workflow import TerminalReport, WorkflowOutcome, WorkflowState

# Autouse; named here so the agent-side doubles resolve the meta_root production does.
pytestmark = pytest.mark.usefixtures(_derive_meta_root_like_production.__name__)

TASK_ID = '42'

FAILING_VERIFY = VerifyResult(
    passed=False,
    test_output='FAILED test_farewell',
    lint_output='',
    type_output='',
    summary='tests failed',
    category=FailureCategory.TEST_FAILURE,
    cause_hint='FAILED test_lib.py::test_farewell',
)


@pytest.fixture(autouse=True)
def _no_dry_run_unblock(monkeypatch):
    monkeypatch.setattr('orchestrator.workflow.run_dry_run_unblock', AsyncMock(return_value=None))


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _init_repo(repo)
    return repo


@pytest.fixture
def config(git_repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=git_repo,
        max_concurrent_tasks=1,
        max_execute_iterations=5,
        max_verify_attempts=3,
        judge_after_each_iteration=False,
        escalate_preexisting_main_break=False,
        sandbox=SandboxConfig(enabled=False),
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


@pytest.fixture
def git_ops(config: OrchestratorConfig) -> GitOps:
    return GitOps(config.git, config.project_root)


@pytest.fixture
def task_assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id=TASK_ID,
        task={
            'id': TASK_ID,
            'title': 'Add farewell function',
            'description': 'Add a farewell(name) function to lib.py with tests',
            'status': 'pending',
            'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


class ScriptedDebuggerStub(AgentStub):
    """An AgentStub whose debugger returns *debugger_result* and commits nothing."""

    def __init__(self, debugger_result: AgentResult):
        super().__init__()
        self.debugger_result = debugger_result

    async def _debugger(self, cwd: Path) -> AgentResult:
        return self.debugger_result


class TerminalRowScheduler(FakeScheduler):
    """A row a racing writer makes *terminal_status* just before the first re-pend.

    While the row is terminal, every non-terminal write without a
    ``reopen_reason`` meets the server's terminal-exit gate and raises
    ``TerminalExitRejection``, as scheduler.py::Scheduler.set_task_status does.
    A ``reopen_reason`` is the gate's sanctioned override, so that write lands.
    """

    def __init__(self, terminal_status: str):
        super().__init__()
        self.terminal_status = terminal_status
        self.refused_writes: list[str] = []

    async def set_task_status(self, task_id: str, status: str, **kwargs) -> None:
        history = self.statuses.setdefault(task_id, [])
        if status == 'pending' and self.terminal_status not in history:
            history.append(self.terminal_status)
        row_is_terminal = history[-1:] == [self.terminal_status]
        if (
            row_is_terminal
            and status not in TERMINAL_STATUSES
            and kwargs.get('reopen_reason') is None
        ):
            self.refused_writes.append(status)
            raise TerminalExitRejection(
                task_id=task_id,
                old_status=self.terminal_status,
                target_status=status,
                raw='terminal-exit gate',
            )
        await super().set_task_status(task_id, status, **kwargs)


async def _run(
    config: OrchestratorConfig,
    git_ops: GitOps,
    assignment: TaskAssignment,
    monkeypatch: pytest.MonkeyPatch,
    debugger_result: AgentResult,
    scheduler: FakeScheduler | None = None,
) -> tuple[TerminalReport, ScriptedDebuggerStub, FakeScheduler]:
    stub = ScriptedDebuggerStub(debugger_result)
    workflow, default_scheduler = _build_workflow(config, git_ops, assignment, stub)
    if scheduler is not None:
        workflow.scheduler = scheduler
    monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
    monkeypatch.setattr(
        'orchestrator.workflow.run_scoped_verification',
        AsyncMock(return_value=FAILING_VERIFY),
    )
    report = await workflow.run()
    return report, stub, scheduler or default_scheduler


def _reviewer_calls(stub: AgentStub) -> list[str]:
    return [call for call in stub.calls if call.startswith('reviewer')]


@pytest.mark.asyncio
class TestFiveHundredKilledDebuggerRequeues:
    async def test_first_529_killed_debugger_requeues(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        report, stub, scheduler = await _run(
            config, git_ops, task_assignment, monkeypatch, incident_529_result(),
        )

        assert report.outcome is WorkflowOutcome.REQUEUED
        assert report.api_error_status == 529
        assert 'agent API error: HTTP 529' in report.reason
        assert report.reason.startswith('Debug fix failed')
        assert report.phase is WorkflowState.VERIFY
        assert scheduler.statuses[TASK_ID][-1] == 'pending'
        assert 'blocked' not in scheduler.statuses[TASK_ID]
        assert stub.calls.count('debugger') == 1
        assert _reviewer_calls(stub) == []

    async def test_requeue_is_routed_to_the_transient_lane(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        report, _, _ = await _run(
            config, git_ops, task_assignment, monkeypatch, incident_529_result(),
        )

        assert is_transient_api_requeue(
            report.reason, api_error_status=report.api_error_status,
        ) is True
        assert is_transient_api_requeue(report.reason, api_error_status=None) is True


@pytest.mark.asyncio
class TestTerminalRowDuringRepend:
    async def test_cancelled_row_ends_the_dispatch_cancelled(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        scheduler = TerminalRowScheduler('cancelled')

        report, stub, _ = await _run(
            config, git_ops, task_assignment, monkeypatch, incident_529_result(),
            scheduler=scheduler,
        )

        assert report.outcome is WorkflowOutcome.CANCELLED
        assert _reviewer_calls(stub) == []
        assert 'blocked' not in scheduler.statuses.get(TASK_ID, [])

    async def test_done_row_drops_the_requeue_and_never_reads_as_green(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        scheduler = TerminalRowScheduler('done')

        report, stub, _ = await _run(
            config, git_ops, task_assignment, monkeypatch, incident_529_result(),
            scheduler=scheduler,
        )

        dropped_requeues = config.max_verify_attempts - 1
        assert scheduler.refused_writes == ['pending'] * dropped_requeues + ['blocked']
        assert stub.calls.count('debugger') == dropped_requeues
        assert _reviewer_calls(stub) == []
        # The terminal-exit gate adjudicates the exhaustion block: a done row
        # with no on-main done_provenance is reopened as a bypass.
        assert report.outcome is WorkflowOutcome.BLOCKED
        assert scheduler.reopen_reasons[TASK_ID] == (
            'bypass detected: Verification attempts exhausted'
        )
        assert scheduler.statuses[TASK_ID][-2:] == ['done', 'blocked']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'debugger_result',
    [
        pytest.param(sigkill_wedge_result(), id='no-status-wedge'),
        pytest.param(replace(incident_529_result(), api_error_status=429), id='rate-limit-429'),
        pytest.param(
            replace(incident_529_result(), turns=5, transcript_turns=5),
            id='5xx-after-progress',
        ),
    ],
)
async def test_non_transient_debugger_failures_keep_exhausting_the_verify_budget(
    config, git_ops, task_assignment, monkeypatch, debugger_result,
):
    report, _, scheduler = await _run(
        config, git_ops, task_assignment, monkeypatch, debugger_result,
    )

    assert report.outcome is WorkflowOutcome.BLOCKED
    assert scheduler.statuses[TASK_ID][-1] == 'blocked'
    assert 'Verification attempts exhausted' in report.reason
    assert 'pending' not in scheduler.statuses[TASK_ID]
