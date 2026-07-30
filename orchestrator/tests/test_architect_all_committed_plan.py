"""Architect pre-satisfies plan steps on an already-green branch (task 3033).

PRD: ``plans/architect-already-complete-exits.md`` §γ ("architect emits an
all-committed plan on an already-green branch → 0 EXECUTE iterations").

Task 3030 (§α) landed the pre-satisfy substrate — ``TaskArtifacts.mark_step_committed``
plus the ``mark_step_committed`` plan-tools MCP tool guarded by a real on-branch
SHA check. This suite covers γ: the wiring that lets the ARCHITECT actually
reach that substrate, and the observable consequences when it does.

Covered surfaces:

  - ``ARCHITECT.allowed_tools`` grants ``mcp__plan-tools__mark_step_committed``
    (and IMPLEMENTER/DEBUGGER deliberately do NOT) — roles.py
  - ``TaskWorkflow._detect_committed_branch_work`` — the architect-facing
    counterpart of the implementer-facing ``_detect_tip_wip_commits`` (workflow.py)
  - ``BriefingAssembler.build_architect_prompt``'s ``committed_work`` rendering
    (briefing.py)
  - ``_plan()`` threading the detector into the architect briefing (workflow.py)
  - ``_execute_iterations`` emitting ``phase_skipped(phase='execute')`` and an
    honest ``agent='architect', event='execute_skipped'`` iteration-log entry
    when nothing is pending (workflow.py)

Boundary tests from the PRD:

  A1-1  an all-committed plan runs ZERO execute iterations, invokes no
        implementer, and says so in a durable event.
  A1-2  a PARTIALLY committed plan still runs the loop for the remaining
        pending steps, and does not re-derive/re-implement the committed one.
  A1-3  VERIFY remains the semantic gate — a falsely pre-satisfied step cannot
        reach MERGE without VERIFY running (pinning test only; no new gate).

Scaffolding is copied from the analogous suite
``orchestrator/tests/test_harness_wip_step_detection.py:104-239`` (task 2051 —
this work's closest precedent, since it also surfaces already-committed commits
to an agent): real temp git repo, real ``GitOps``, a real worktree from
``git_ops.create_worktree``, heavy collaborators (scheduler/briefing/mcp) mocked.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow

# ---------------------------------------------------------------------------
# Shared real-git scaffolding (pre-1)
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
    (repo / 'lib.py').write_text('x = 1\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial'], cwd=repo)


@pytest.fixture
def config(git_repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=git_repo,
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
            'id': '42', 'title': 'X', 'description': '',
            'status': 'pending', 'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


class FakeEventStore:
    """Minimal ``EventStore`` double recording every ``emit`` call.

    ``TaskWorkflow.event_store`` is optional in production — every emitter is
    guarded by ``if self.event_store:`` (e.g. workflow.py:4223) — so the real
    workflow constructor leaves it unset. Attaching this double is what lets
    the boundary tests assert on emitted events instead of on an absence.

    Also implements ``fetch_events_by_type_all_runs`` (the run-agnostic reader
    ``verify_checkpoint.green_checkpoint_at_tip`` consults) over the recorded
    rows, so A1-3 can assert the durable-green VERIFY skip is NOT satisfied by
    an all-``done`` plan.
    """

    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(
        self,
        event_type: EventType,
        *,
        task_id: str | None = None,
        phase: str | None = None,
        role: str | None = None,
        data: dict | None = None,
        cost_usd: float | None = None,
        duration_ms: int | None = None,
    ) -> None:
        self.events.append({
            'event_type': event_type,
            'task_id': task_id,
            'phase': phase,
            'role': role,
            'data': data or {},
        })

    def fetch_events_by_type_all_runs(
        self, event_type: EventType, *, task_id: str | None = None,
    ) -> list[dict]:
        return [
            e for e in self.events
            if e['event_type'] == event_type
            and (task_id is None or e['task_id'] == task_id)
        ]

    def of_type(self, event_type: EventType, phase: str | None = None) -> list[dict]:
        """Recorded rows of *event_type*, optionally narrowed to *phase*."""
        return [
            e for e in self.events
            if e['event_type'] == event_type
            and (phase is None or e['phase'] == phase)
        ]


def _make_workflow(
    config: OrchestratorConfig,
    git_ops: GitOps,
    assignment: TaskAssignment,
    worktree: Path,
) -> tuple[TaskWorkflow, TaskArtifacts]:
    """Wire a minimal TaskWorkflow with heavy collaborators mocked.

    Mirrors test_harness_wip_step_detection.py._make_workflow, extended to
    attach a :class:`FakeEventStore` so emitted events are assertable.
    """
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=MagicMock(),  # type: ignore[arg-type]
        briefing=MagicMock(),  # type: ignore[arg-type]
        mcp=MagicMock(),  # type: ignore[arg-type]
    )
    workflow.worktree = worktree
    artifacts = TaskArtifacts(worktree)
    artifacts.init('42', 'X', 'desc', base_commit='base-sha-old')
    workflow.artifacts = artifacts
    workflow.plan = {'task_id': '42', 'steps': [], 'prerequisites': []}
    workflow.event_store = FakeEventStore()  # type: ignore[assignment]
    return workflow, artifacts


# ---------------------------------------------------------------------------
# step-1 RED: the architect's capability surface for α's mark_step_committed
# ---------------------------------------------------------------------------
#
# allowed_tools is the ENFORCED tool surface — agents/invoke.py:979-985 builds
# the backend `--tools` allowlist from it (and loudly drops specs it cannot
# express), so a tool absent from allowed_tools is structurally unreachable no
# matter what the prompt says.

_MARK_COMMITTED = 'mcp__plan-tools__mark_step_committed'


class TestArchitectCanReachMarkStepCommitted:
    def test_architect_is_granted_mark_step_committed(self):
        from orchestrator.agents.roles import ARCHITECT  # noqa: PLC0415

        assert _MARK_COMMITTED in ARCHITECT.allowed_tools, (
            f'ARCHITECT.allowed_tools omits {_MARK_COMMITTED!r} — the backend '
            f'allowlist built at agents/invoke.py:979-985 would strip the call, '
            f'making the γ signal structurally unreachable. Got: '
            f'{ARCHITECT.allowed_tools!r}'
        )

    def test_implementer_is_not_granted_mark_step_committed(self):
        """Pre-satisfying is an AUTHORING-time authority, not an execution one.

        IMPLEMENTER/DEBUGGER keep only ``mark_step_done`` via
        ``_PLAN_STATUS_TOOLS``: an implementer that could self-grant ``done``
        with a description tag claiming committed provenance would defeat the
        whole TDD bookkeeping.
        """
        from orchestrator.agents.roles import IMPLEMENTER  # noqa: PLC0415

        assert _MARK_COMMITTED not in IMPLEMENTER.allowed_tools
        assert 'mcp__plan-tools__mark_step_done' in IMPLEMENTER.allowed_tools

    def test_debugger_is_not_granted_mark_step_committed(self):
        from orchestrator.agents.roles import DEBUGGER  # noqa: PLC0415

        assert _MARK_COMMITTED not in DEBUGGER.allowed_tools

    def test_architect_declares_plan_tools_family(self):
        """Keeps the inverse-capability invariant satisfied by the new grant.

        test_agent_capability_wiring.py::TestInverseCapabilityInvariant asserts
        every role allowing an ``mcp__plan-tools__`` prefix declares the
        ``plan_tools`` family; asserted here too so a grant added without the
        family declaration fails in this suite as well.
        """
        from orchestrator.agents.roles import ARCHITECT  # noqa: PLC0415

        assert 'plan_tools' in ARCHITECT.mcp_families


@pytest.mark.asyncio
class TestGrantedToolActuallyExists:
    async def test_granted_name_resolves_to_a_registered_tool(self, tmp_path):
        """The grant must never name a tool the server does not register.

        Mirrors the registration assertion at test_plan_tools_server.py:1089 —
        an allowlist entry for a non-existent tool is a silent dead grant.
        """
        from orchestrator.mcp import plan_tools  # noqa: PLC0415

        artifacts = TaskArtifacts(tmp_path)
        artifacts.init('42', 'X', 'desc', base_commit='base')
        server = plan_tools.create_server(artifacts)

        tool = await server.get_tool('mark_step_committed')

        assert tool is not None
        # The granted allowlist name is the mcp__<server>__<tool> spelling of
        # exactly this registered tool.
        assert _MARK_COMMITTED.endswith('__mark_step_committed')
