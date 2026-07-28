"""Tests for the run()-ESCALATED steward wait (task 3170).

``run()``'s ESCALATED branch waits via :meth:`TaskWorkflow._wait_for_resolution`,
which is escalation-queue-only: it is woken by ``EscalationQueue.resolve`` →
``_resolve_callback`` → ``harness._on_escalation_resolved`` →
``_escalation_events[task_id].set()``.  It never reads the steward outcome
channel that its sibling waiter (:meth:`_await_steward_completion`, reached via
``_mark_blocked``) consumes.

Task 2248 introduced two steward give-up branches that publish a typed
``StewardInterrupted`` outcome WITHOUT dismissing their own L0.  On the
``_mark_blocked`` path that is fine (the channel carries the signal); on the
``run()``-ESCALATED path the publish goes to an unread channel and the still-
pending L0 keeps the waiter blocked forever, while the steward re-handles the
same capped escalation at loop speed.

The converged contract these tests pin:

    A steward give-up ALWAYS dismisses its own L0 before publishing an
    outcome.  No pending L0 survives a steward give-up.

Both waiters observe that single producer invariant, and the ESCALATED wait is
additionally bounded by ``steward_completion_timeout`` so a future producer bug
degrades into a loud, bounded unblock instead of a permanent strand.

Modelled on ``test_workflow_status_on_resume.py`` — the only existing module
that drives ``run()``'s ESCALATED branch deterministically.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

# ---------------------------------------------------------------------------
# Fixtures (local — mirrors test_workflow_status_on_resume.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'lib.py').write_text('def greet(name): return name\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def config(git_repo: Path) -> OrchestratorConfig:
    """Config with a SHORT ``steward_completion_timeout``.

    Only ``timeouts.steward >= steward_completion_timeout`` is validated, and
    the stock ``timeouts.steward`` (1800s) satisfies 0.5s comfortably.  Tests
    that must NOT be satisfiable by the timeout (the cross-component
    regression) raise it back up locally.
    """
    return OrchestratorConfig(
        project_root=git_repo,
        max_concurrent_tasks=1,
        steward_completion_timeout=0.5,
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
        task_id='42',
        task={
            'id': '42',
            'title': 'X',
            'description': 'Y',
            'status': 'pending',
            'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


PLAN = {
    'task_id': '42',
    'title': 'X',
    'files': ['lib.py'],
    'analysis': '',
    'prerequisites': [],
    'steps': [
        {
            'id': 'step-1',
            'type': 'impl',
            'description': '',
            'status': 'pending',
            'commit': None,
        },
    ],
    'design_decisions': [],
    'reuse': [],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_advanced_worktree(git_ops: GitOps, task_id: str) -> Path:
    """Create a worktree on a task branch ONE commit ahead of main.

    Two things depend on this: the already-on-main short-circuit in run()'s
    ESCALATED branch must not fire (it would mask the wait entirely), and
    ``_worktree_has_wip_commits`` must derive True (it is the wip probe the
    production ``_ensure_steward_started`` injects into the steward, and the
    give-up branches under test are wip-gated).
    """
    wt_info = await git_ops.create_worktree(task_id)
    wt = wt_info.path
    (wt / 'precommit.txt').write_text('test marker\n')
    await _run(['git', 'add', 'precommit.txt'], cwd=wt)
    await _run(['git', 'commit', '-m', 'test marker commit'], cwd=wt)
    return wt


def _build_workflow(
    config: OrchestratorConfig,
    git_ops: GitOps,
    assignment: TaskAssignment,
    queue: EscalationQueue,
    worktree: Path,
) -> tuple[TaskWorkflow, FakeScheduler]:
    """Wire a TaskWorkflow with all fakes for these tests.

    ``escalation_event`` is passed at construction exactly as the harness does
    at dispatch time (``_register_escalation_event``), so
    :func:`_wire_resolve_callback` always has a live event to set.

    Worktree is pre-set (eval-mode external path) so run() skips
    create_worktree and the MERGE block.
    """
    scheduler = FakeScheduler()
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,  # type: ignore[arg-type]
        briefing=FakeBriefing(),  # type: ignore[arg-type]
        mcp=FakeMcp(),  # type: ignore[arg-type]
        escalation_queue=queue,
        escalation_event=asyncio.Event(),
        initial_plan=dict(PLAN),
    )
    workflow.worktree = worktree
    return workflow, scheduler


def _wire_resolve_callback(queue: EscalationQueue, workflow: TaskWorkflow) -> None:
    """Wire the queue's resolve callback to the workflow's escalation event.

    Mirrors ``harness._on_escalation_resolved`` (harness.py:11164-11174), which
    does exactly ``self._escalation_events[escalation.task_id].set()``.  This is
    the ONLY thing that wakes ``_wait_for_resolution`` in production, so a test
    that omits it does not reproduce the production wake path at all.
    """
    def _on_resolved(esc) -> None:  # noqa: ARG001 — signature parity with harness
        event = workflow._escalation_event
        if event is not None:
            event.set()

    queue.set_resolve_callback(_on_resolved)


def _submit_l0(
    queue: EscalationQueue, task_id: str, *, category: str = 'task_failure',
) -> Escalation:
    """Submit a pending level-0 escalation and return the model (not just the id).

    The steward-side tests need the whole ``Escalation`` to hand to a real
    ``TaskSteward._handle_escalation``.
    """
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role='implementer',
        severity='blocking',
        category=category,
        summary='synthetic blocker',
        detail='synthetic detail',
    )
    queue.submit(esc)
    return esc


def _make_evrl_returner(returns: list[WorkflowOutcome]):
    """Return an AsyncMock that pops successive WorkflowOutcomes per call.

    The last value is reused if the list is exhausted, so a misconfigured test
    does not hang the workflow's outer while-true.
    """
    state = {'count': 0, 'queue': list(returns)}

    async def fake_evrl():
        state['count'] += 1
        q = state['queue']
        if len(q) > 1:
            return q.pop(0)
        return q[0] if q else WorkflowOutcome.DONE

    return AsyncMock(side_effect=fake_evrl), state
