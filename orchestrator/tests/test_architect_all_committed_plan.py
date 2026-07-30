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
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

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


# ---------------------------------------------------------------------------
# step-3 RED: TaskWorkflow._detect_committed_branch_work
# ---------------------------------------------------------------------------
#
# The ARCHITECT-facing counterpart of the implementer-facing
# _detect_tip_wip_commits (workflow.py:6884). Two filters are deliberately
# absent, and this class machine-checks that divergence:
#   - no is_wip_safety_commit filter — an already-committed-green branch is
#     made of ordinary feat(...)/test(...) commits, which the WIP detector
#     cannot see at all;
#   - no dedup against already-`done` steps' recorded commits — an architect
#     re-authoring the plan needs every branch commit as candidate provenance.


@pytest.mark.asyncio
class TestDetectCommittedBranchWork:
    async def test_ordinary_commits_are_surfaced_head_first(
        self, config, git_ops, task_assignment,
    ):
        """Two ordinary (non-WIP) commits: both surfaced, HEAD-first."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'red.py').write_text('def test_x(): assert False\n')
        red_sha = await git_ops.commit(wt, 'test: RED — x')
        (wt / 'impl.py').write_text('x = 1\n')
        green_sha = await git_ops.commit(wt, 'feat: GREEN — x')
        assert red_sha and green_sha, 'Setup: expected two real commits'

        result = await workflow._detect_committed_branch_work()

        assert result == [
            {'sha': green_sha, 'subject': 'feat: GREEN — x'},
            {'sha': red_sha, 'subject': 'test: RED — x'},
        ], (
            'Expected BOTH ordinary commits HEAD-first — unlike '
            '_detect_tip_wip_commits this detector must not filter on '
            f'is_wip_safety_commit. Got {result}'
        )

    async def test_commit_recorded_on_a_done_step_is_still_surfaced(
        self, config, git_ops, task_assignment,
    ):
        """No done-step dedup: the architect needs every branch commit.

        _detect_tip_wip_commits drops a commit already recorded as a done
        step's ``commit`` (correct for the implementer's attribution loop);
        the architect is RE-AUTHORING the plan, so the same commit is
        candidate provenance for a freshly-added step and must still appear.
        """
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'impl.py').write_text('x = 1\n')
        sha = await git_ops.commit(wt, 'feat: GREEN — x')
        assert sha

        workflow.plan = {
            'task_id': '42',
            'prerequisites': [],
            'steps': [{'id': 'step-1', 'status': 'done', 'commit': sha}],
        }

        result = await workflow._detect_committed_branch_work()

        assert result == [{'sha': sha, 'subject': 'feat: GREEN — x'}]

    async def test_head_at_base_returns_empty(self, config, git_ops, task_assignment):
        """A truly-fresh first dispatch (HEAD == base_commit) surfaces nothing."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        assert await workflow._detect_committed_branch_work() == []

    async def test_unset_base_commit_returns_empty(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit('')

        (wt / 'impl.py').write_text('x = 1\n')
        await git_ops.commit(wt, 'feat: GREEN — x')

        assert await workflow._detect_committed_branch_work() == []

    @pytest.mark.parametrize('missing', ['worktree', 'git_ops', 'artifacts'])
    async def test_missing_collaborator_returns_empty(
        self, config, git_ops, task_assignment, missing,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)
        (wt / 'impl.py').write_text('x = 1\n')
        await git_ops.commit(wt, 'feat: GREEN — x')

        setattr(workflow, missing, None)

        assert await workflow._detect_committed_branch_work() == []

    async def test_git_error_returns_empty_and_does_not_raise(
        self, config, git_ops, task_assignment,
    ):
        """Best-effort posture: a false negative must never sink PLAN."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)
        (wt / 'impl.py').write_text('x = 1\n')
        await git_ops.commit(wt, 'feat: GREEN — x')

        async def _boom(*_args, **_kwargs):
            raise RuntimeError('git exploded')

        workflow.git_ops.get_commit_subjects = _boom  # type: ignore[method-assign]

        assert await workflow._detect_committed_branch_work() == []


# ---------------------------------------------------------------------------
# step-5 RED: build_architect_prompt renders the already-committed-work section
# ---------------------------------------------------------------------------


@pytest.fixture
def briefing_assembler(tmp_path: Path):
    """Real BriefingAssembler, mirroring test_briefing.py's construction."""
    from orchestrator.agents.briefing import BriefingAssembler  # noqa: PLC0415

    return BriefingAssembler(OrchestratorConfig(
        project_root=tmp_path,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    ))


@pytest.fixture
def architect_task() -> dict:
    return {
        'id': '3033',
        'title': 'Architect all-committed plan',
        'description': 'Pre-satisfy steps already committed on the branch.',
        'metadata': {'files': ['orchestrator']},
    }


_SHA_A = 'a1b2c3d4e5f60718293a4b5c6d7e8f9012345678'
_SHA_B = '0f1e2d3c4b5a69788796a5b4c3d2e1f001234567'
_COMMITTED_WORK = [
    {'sha': _SHA_A, 'subject': 'feat: GREEN — the thing'},
    {'sha': _SHA_B, 'subject': 'test: RED — the thing'},
]


@pytest.mark.asyncio
class TestArchitectPromptCommittedWorkSection:
    async def _prompt(self, briefing_assembler, architect_task, **kwargs) -> str:
        return await briefing_assembler.build_architect_prompt(
            architect_task, worktree=None, context='', **kwargs,
        )

    async def test_renders_each_commit_sha_and_subject(
        self, briefing_assembler, architect_task,
    ):
        prompt = await self._prompt(
            briefing_assembler, architect_task, committed_work=_COMMITTED_WORK,
        )

        for entry in _COMMITTED_WORK:
            assert entry['sha'][:12] in prompt, (
                f"Expected abbreviated sha {entry['sha'][:12]} in the prompt"
            )
            assert entry['subject'] in prompt

    async def test_names_mark_step_committed_and_the_pre_satisfy_instruction(
        self, briefing_assembler, architect_task,
    ):
        prompt = await self._prompt(
            briefing_assembler, architect_task, committed_work=_COMMITTED_WORK,
        )

        assert 'mark_step_committed' in prompt
        lowered = prompt.lower()
        assert 'pending' in lowered, (
            'The section must contrast pre-satisfying against leaving the step '
            'pending'
        )

    async def test_states_verify_is_the_gate(self, briefing_assembler, architect_task):
        """No-silent-green: the architect must be told VERIFY still runs."""
        prompt = await self._prompt(
            briefing_assembler, architect_task, committed_work=_COMMITTED_WORK,
        )

        assert 'VERIFY' in prompt
        lowered = prompt.lower()
        assert 'never pre-satisfy' in lowered or 'do not pre-satisfy' in lowered, (
            'The section must explicitly forbid pre-satisfying a step whose '
            'tests have not been seen to pass'
        )

    async def test_instructs_first_hand_corroboration(
        self, briefing_assembler, architect_task,
    ):
        prompt = await self._prompt(
            briefing_assembler, architect_task, committed_work=_COMMITTED_WORK,
        )

        assert 'git show' in prompt
        assert "tests" in prompt.lower()

    @pytest.mark.parametrize('committed_work', [None, []])
    async def test_no_section_when_no_committed_work(
        self, briefing_assembler, architect_task, committed_work,
    ):
        """A truly-fresh first dispatch stays byte-identical to today.

        Preserves the C-A1 anti-anchoring posture documented at briefing.py:76-81
        — no extra section, and no mention of the pre-satisfy tool at all.
        """
        prompt = await self._prompt(
            briefing_assembler, architect_task, committed_work=committed_work,
        )

        assert 'mark_step_committed' not in prompt
        assert 'Already-Committed' not in prompt

    async def test_default_call_shape_still_works(
        self, briefing_assembler, architect_task,
    ):
        """The new kwarg is keyword-only with a None default."""
        prompt = await briefing_assembler.build_architect_prompt(
            architect_task, None, '',
        )

        assert 'mark_step_committed' not in prompt
        assert architect_task['title'] in prompt

    async def test_rendering_is_identical_to_the_no_kwarg_baseline(
        self, briefing_assembler, architect_task,
    ):
        """committed_work=[] must produce exactly the pre-γ prompt."""
        baseline = await briefing_assembler.build_architect_prompt(
            architect_task, None, '',
        )
        with_empty = await self._prompt(
            briefing_assembler, architect_task, committed_work=[],
        )

        assert with_empty == baseline


# ---------------------------------------------------------------------------
# step-7 RED: _plan() threads the detector into the architect briefing
# ---------------------------------------------------------------------------
#
# A wiring test only: the assertions stop at the briefing boundary rather than
# driving _plan's outcome. Harness modelled on
# test_workflow_architect_l0_promotion.py::_make — the architect _invoke is
# stubbed to succeed without writing a plan, and _handle_no_plan_failure is
# stubbed so _plan returns immediately after the (observed) briefing call.


def _make_plan_workflow(tmp_path: Path) -> tuple[TaskWorkflow, AsyncMock]:
    from unittest.mock import AsyncMock as _AsyncMock  # noqa: PLC0415

    from _orch_helpers import pydantic_spec  # noqa: PLC0415
    from shared.cli_invoke import AgentResult  # noqa: PLC0415

    assignment = MagicMock()
    assignment.task_id = '3033'
    assignment.task = {
        'id': '3033', 'title': 'T', 'description': 'd', 'metadata': {},
    }
    assignment.modules = ['mod_a']

    cfg = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    cfg.fused_memory.project_id = 'dark_factory'
    cfg.fused_memory.url = 'http://localhost:8002'
    cfg.lock_depth = 2
    cfg.steward_completion_timeout = 300.0
    cfg.project_root = tmp_path / 'proj'
    cfg.max_review_cycles = 2
    cfg.max_amendment_rounds = 1

    scheduler = MagicMock()
    scheduler.set_task_status = _AsyncMock()
    scheduler.update_task = _AsyncMock(return_value=True)
    scheduler.get_status = _AsyncMock(return_value='in-progress')

    fake_git = MagicMock()
    fake_git.get_main_sha = _AsyncMock(return_value='currentmain')

    escalation_queue = MagicMock()
    escalation_queue.get_by_task = MagicMock(return_value=[])

    wf = TaskWorkflow(
        assignment=assignment,
        config=cfg,  # type: ignore[arg-type]
        git_ops=fake_git,  # type: ignore[arg-type]
        scheduler=scheduler,  # type: ignore[arg-type]
        briefing=MagicMock(),  # type: ignore[arg-type]
        mcp=MagicMock(),  # type: ignore[arg-type]
        escalation_queue=escalation_queue,
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    artifacts = TaskArtifacts(worktree)
    artifacts.init('3033', 'T', 'd', base_commit='oldbase')
    wf.artifacts = artifacts
    wf.worktree = worktree

    # Architect succeeds but writes no plan.json → _plan falls to the stubbed
    # no-plan handler, which is enough to observe the briefing call.
    wf._invoke = _AsyncMock(  # type: ignore[method-assign]
        return_value=AgentResult(success=True, output='ok'),
    )
    wf._mark_blocked = _AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]
    wf._handle_no_plan_failure = _AsyncMock(  # type: ignore[method-assign]
        return_value=WorkflowOutcome.BLOCKED,
    )
    build = _AsyncMock(return_value='architect-prompt')
    wf.briefing.build_architect_prompt = build
    return wf, build


@pytest.mark.asyncio
class TestPlanThreadsCommittedWorkIntoBriefing:
    async def test_detector_awaited_once_and_result_forwarded(self, tmp_path):
        wf, build = _make_plan_workflow(tmp_path)
        detected = [{'sha': _SHA_A, 'subject': 'feat: GREEN — the thing'}]
        detector = AsyncMock(return_value=detected)
        wf._detect_committed_branch_work = detector  # type: ignore[method-assign]

        await wf._plan()

        assert detector.await_count == 1, (
            f'Expected exactly one detector await, got {detector.await_count}'
        )
        build.assert_awaited()
        kwargs = build.call_args.kwargs
        assert kwargs.get('committed_work') == detected, (
            f'Expected the detector list forwarded as committed_work, got {kwargs}'
        )

    async def test_empty_detection_is_forwarded_without_crashing(self, tmp_path):
        wf, build = _make_plan_workflow(tmp_path)
        wf._detect_committed_branch_work = AsyncMock(return_value=[])  # type: ignore[method-assign]

        await wf._plan()

        build.assert_awaited()
        assert build.call_args.kwargs.get('committed_work') == []

    async def test_detector_failure_does_not_sink_plan(self, tmp_path):
        """Best-effort: a raising detector must still leave PLAN running."""
        wf, build = _make_plan_workflow(tmp_path)
        wf._detect_committed_branch_work = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError('detector exploded'),
        )

        await wf._plan()

        build.assert_awaited()
        assert wf._invoke.await_count >= 1, (  # type: ignore[attr-defined]
            'A detector failure must not prevent the architect invocation'
        )


# ---------------------------------------------------------------------------
# step-9 RED: boundary tests A1-1 / A1-2 on the EXECUTE no-op
# ---------------------------------------------------------------------------
#
# Driving pattern from test_harness_wip_step_detection.py:576-623 — stub
# _invoke with a side effect, _check_escalations → [], _get_head_commit → a sha.
# Fixtures are produced by CALLING artifacts.mark_step_committed (α's landed
# method) rather than hand-writing status='done' dicts, so they carry the
# production shape (full sha in `commit`, `[COMMITTED <sha[:12]>]` description
# tag) and break loudly if α's contract ever drifts.

_PLAN_STEPS = [
    {'id': 'step-1', 'type': 'test', 'description': 'RED — x',
     'status': 'pending', 'commit': None},
    {'id': 'step-2', 'type': 'impl', 'description': 'GREEN — x',
     'status': 'pending', 'commit': None},
]


def _write_plan(artifacts: TaskArtifacts, workflow: TaskWorkflow) -> None:
    artifacts.write_plan({
        'task_id': '42',
        'title': 'X',
        'analysis': 'A',
        'prerequisites': [
            {'id': 'pre-1', 'description': 'setup', 'status': 'pending',
             'commit': None},
        ],
        'steps': [dict(s) for s in _PLAN_STEPS],
    })
    artifacts.stamp_plan_provenance(workflow.session_id)
    workflow.plan = artifacts.read_plan()


def _stub_execute_collaborators(
    workflow: TaskWorkflow,
) -> tuple[AsyncMock, AsyncMock]:
    """Stub the heavy EXECUTE collaborators.

    Returns ``(invoke, build_implementer_prompt)``. Both mocks are handed back
    rather than read off ``workflow`` at the assertion site: a member-access
    assignment only narrows within its own scope, so outside this helper
    ``workflow.briefing.build_implementer_prompt`` resolves to its declared
    bound-method type and ``.await_count``/``.call_args`` are unexpressible.
    """
    build_implementer_prompt = AsyncMock(return_value='impl')
    workflow.briefing.build_implementer_prompt = build_implementer_prompt
    workflow._check_escalations = MagicMock(return_value=[])  # type: ignore[method-assign]
    workflow._get_head_commit = AsyncMock(return_value='head-sha')  # type: ignore[method-assign]
    invoke = AsyncMock()
    workflow._invoke = invoke  # type: ignore[method-assign]
    return invoke, build_implementer_prompt


@pytest.mark.asyncio
class TestA1AllCommittedPlanSkipsExecute:
    """A1-1: an all-committed plan runs ZERO execute iterations."""

    async def _setup(self, config, git_ops, task_assignment):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)
        _write_plan(artifacts, workflow)

        # Two real branch commits, then pre-satisfy EVERY plan item against them.
        (wt / 'red.py').write_text('def test_x(): assert True\n')
        red_sha = await git_ops.commit(wt, 'test: RED — x')
        (wt / 'impl.py').write_text('x = 1\n')
        green_sha = await git_ops.commit(wt, 'feat: GREEN — x')
        assert red_sha and green_sha, 'Setup: expected two real commits'

        assert artifacts.mark_step_committed('pre-1', red_sha) is True
        assert artifacts.mark_step_committed('step-1', red_sha) is True
        assert artifacts.mark_step_committed('step-2', green_sha) is True
        assert artifacts.get_pending_steps() == [], 'Setup: nothing may remain pending'
        workflow.plan = artifacts.read_plan()

        invoke, build_implementer_prompt = _stub_execute_collaborators(workflow)
        return workflow, artifacts, invoke, build_implementer_prompt

    async def test_zero_iterations_and_no_implementer_turn(
        self, config, git_ops, task_assignment,
    ):
        workflow, _artifacts, invoke, build_implementer_prompt = await self._setup(
            config, git_ops, task_assignment,
        )

        outcome = await workflow._execute_iterations()

        assert outcome == WorkflowOutcome.DONE
        assert workflow.metrics.execute_iterations == 0, (
            'An all-committed plan must run ZERO execute iterations, got '
            f'{workflow.metrics.execute_iterations}'
        )
        assert invoke.await_count == 0, 'No implementer LLM turn may happen'
        assert build_implementer_prompt.await_count == 0

    async def test_emits_phase_skipped_execute_event(
        self, config, git_ops, task_assignment,
    ):
        """The zero-iteration skip must be POSITIVELY observable.

        Without this event "0 iterations" is a pure absence, indistinguishable
        from a lost event log — see the structured-facts-at-failure invariant.
        """
        workflow, artifacts, _invoke, _build_prompt = await self._setup(
            config, git_ops, task_assignment,
        )

        await workflow._execute_iterations()

        skips = workflow.event_store.of_type(  # type: ignore[attr-defined]
            EventType.phase_skipped, phase='execute',
        )
        assert len(skips) == 1, (
            f'Expected exactly one phase_skipped(phase="execute") event, got {skips}'
        )
        event = skips[0]
        assert event['task_id'] == workflow.task_id
        data = event['data']
        assert data['reason'] == 'no_pending_steps'
        assert data['execute_iterations'] == 0
        plan = artifacts.read_plan()
        total = len(plan['steps']) + len(plan['prerequisites'])
        assert data['step_count'] == total, (
            f"Expected step_count={total}, got {data['step_count']}"
        )
        assert set(data['done_step_ids']) == {'pre-1', 'step-1', 'step-2'}, (
            f"Expected every pre-satisfied id, got {data['done_step_ids']}"
        )


@pytest.mark.asyncio
class TestA1PartiallyCommittedPlanStillRunsExecute:
    """A1-2: a partially-committed plan still runs the loop for what's left."""

    async def test_partial_plan_runs_loop_for_pending_step_only(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)
        artifacts.write_plan({
            'task_id': '42',
            'title': 'X',
            'analysis': 'A',
            'prerequisites': [],
            'steps': [dict(s) for s in _PLAN_STEPS],
        })
        artifacts.stamp_plan_provenance(workflow.session_id)

        (wt / 'red.py').write_text('def test_x(): assert True\n')
        red_sha = await git_ops.commit(wt, 'test: RED — x')
        assert red_sha
        assert artifacts.mark_step_committed('step-1', red_sha) is True
        workflow.plan = artifacts.read_plan()

        pending_ids = [s['id'] for s in artifacts.get_pending_steps()]
        assert pending_ids == ['step-2'], f'Setup: expected only step-2, got {pending_ids}'

        invoke, build_implementer_prompt = _stub_execute_collaborators(workflow)

        def _mark_pending_done(*_args, **_kwargs):
            from shared.cli_invoke import AgentResult  # noqa: PLC0415

            artifacts.update_step_status('step-2', 'done', 'impl-commit-sha')
            return AgentResult(success=True, output='')

        invoke.side_effect = _mark_pending_done

        outcome = await workflow._execute_iterations()

        assert outcome == WorkflowOutcome.DONE
        assert workflow.metrics.execute_iterations == 1, (
            'A partially-committed plan must still run the loop for the '
            f'remaining pending step, got {workflow.metrics.execute_iterations}'
        )
        assert workflow.event_store.of_type(  # type: ignore[attr-defined]
            EventType.phase_skipped, phase='execute',
        ) == [], 'No execute-skip event may be emitted when work remained'

        # The pre-satisfied step is neither re-derived nor re-implemented.
        handed_plan = build_implementer_prompt.call_args.args[0]
        step_1 = next(
            s for s in handed_plan['steps'] if s['id'] == 'step-1'
        )
        assert step_1['status'] == 'done'
        assert step_1['commit'] == red_sha
        assert step_1['description'].startswith(f'[COMMITTED {red_sha[:12]}]'), (
            f"Expected the [COMMITTED …] provenance tag intact, got "
            f"{step_1['description']!r}"
        )


# ---------------------------------------------------------------------------
# step-11 RED: durable honest signal for the skipped EXECUTE + boundary A1-3
# ---------------------------------------------------------------------------
#
# Why an iteration-log entry is REQUIRED, not cosmetic: _has_prior_implementation
# (workflow.py:11156) requires BOTH SHA divergence AND at least one
# _iteration_entry_is_work entry, and the merge-phase guard _recover_before_merge
# uses the iteration-log-only fallback (wt_head=None). An all-committed plan
# writes zero implementer/debugger/judge entries, so without an entry of its own
# γ would make an already-landed branch resolve has_work=False and REFUSE to
# recover-DONE — re-planning merged work.


@pytest.mark.asyncio
class TestSkippedExecuteWritesDurableLogEntry:
    async def _run_all_committed(self, config, git_ops, task_assignment):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)
        _write_plan(artifacts, workflow)

        (wt / 'red.py').write_text('def test_x(): assert True\n')
        red_sha = await git_ops.commit(wt, 'test: RED — x')
        (wt / 'impl.py').write_text('x = 1\n')
        green_sha = await git_ops.commit(wt, 'feat: GREEN — x')
        for step_id, sha in (
            ('pre-1', red_sha), ('step-1', red_sha), ('step-2', green_sha),
        ):
            assert artifacts.mark_step_committed(step_id, sha) is True
        workflow.plan = artifacts.read_plan()
        _stub_execute_collaborators(workflow)

        outcome = await workflow._execute_iterations()
        assert outcome == WorkflowOutcome.DONE
        return workflow, artifacts, green_sha

    async def test_writes_one_honest_architect_execute_skipped_entry(
        self, config, git_ops, task_assignment,
    ):
        """An HONEST label, not a forged implementer entry.

        Forging ``agent='implementer'`` would pass the existing classifier with
        zero code change but would assert in the durable record that an
        implementer ran when none did — corrupting the very log the reconciler
        and future architects read.
        """
        _workflow, artifacts, _sha = await self._run_all_committed(
            config, git_ops, task_assignment,
        )

        entries, corrupted = artifacts.read_iteration_log()

        assert corrupted == []
        assert len(entries) == 1, (
            f'Expected exactly one iteration-log entry, got {entries}'
        )
        entry = entries[0]
        assert entry['agent'] == 'architect'
        assert entry['event'] == 'execute_skipped'
        assert set(entry['steps_completed']) == {'pre-1', 'step-1', 'step-2'}
        assert entry['committed'] is True
        assert entry['source'] == 'orchestrator'

    async def test_entry_counts_as_work_so_merge_recovery_still_fires(
        self, config, git_ops, task_assignment,
    ):
        """The entry must satisfy _iteration_entry_is_work AND, with a real
        diverged HEAD, _has_prior_implementation."""
        from orchestrator.workflow import _iteration_entry_is_work  # noqa: PLC0415

        workflow, artifacts, head_sha = await self._run_all_committed(
            config, git_ops, task_assignment,
        )
        entries, _ = artifacts.read_iteration_log()

        assert _iteration_entry_is_work(entries[0]) is True

        # SHA-primary path: real diverged HEAD + the log entry ⇒ has_work.
        assert head_sha != artifacts.read_base_commit(), 'Setup: HEAD must diverge'
        assert workflow._has_prior_implementation(wt_head=head_sha).has_work is True
        # Iteration-log-only fallback (the shape _recover_before_merge uses).
        assert workflow._has_prior_implementation().has_work is True


class TestClassifierExtensionOpensNoFalseDoneHole:
    """The new architect clause must be narrow: BOTH the event name AND a
    non-empty steps_completed are required, and the pre-existing
    ``committed is False`` early-return still applies."""

    @staticmethod
    def _is_work(entry: dict) -> bool:
        from orchestrator.workflow import _iteration_entry_is_work  # noqa: PLC0415

        return _iteration_entry_is_work(entry)

    def test_architect_execute_skipped_with_no_steps_is_not_work(self):
        assert self._is_work({
            'agent': 'architect', 'event': 'execute_skipped',
            'steps_completed': [], 'committed': True,
        }) is False

    def test_architect_execute_skipped_uncommitted_is_not_work(self):
        assert self._is_work({
            'agent': 'architect', 'event': 'execute_skipped',
            'steps_completed': ['step-1'], 'committed': False,
        }) is False

    def test_architect_entry_with_other_event_is_not_work(self):
        assert self._is_work({
            'agent': 'architect', 'event': 'plan_written',
            'steps_completed': ['step-1'], 'committed': True,
        }) is False

    def test_zero_work_implementer_entry_stays_not_work(self):
        assert self._is_work({
            'agent': 'implementer', 'steps_completed': [],
        }) is False

    def test_debugger_entry_stays_work(self):
        assert self._is_work({
            'agent': 'debugger', 'steps_completed': [],
        }) is True


@pytest.mark.asyncio
class TestA1VerifyRemainsTheGate:
    """A1-3: a falsely pre-satisfied step cannot reach MERGE without VERIFY.

    Pinning test only — no new gate is implemented. It exists so a future change
    that lets an all-``done`` plan short-circuit VERIFY fails here instead of
    shipping a silent green.
    """

    async def test_zero_iteration_execute_does_not_satisfy_verify(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)
        _write_plan(artifacts, workflow)
        (wt / 'impl.py').write_text('x = 1\n')
        sha = await git_ops.commit(wt, 'feat: GREEN — x')
        for step_id in ('pre-1', 'step-1', 'step-2'):
            assert artifacts.mark_step_committed(step_id, sha) is True
        workflow.plan = artifacts.read_plan()
        _stub_execute_collaborators(workflow)

        await workflow._execute_iterations()

        assert not workflow._verify_checkpoint_hit, (
            'A zero-iteration EXECUTE must not mark the VERIFY checkpoint hit'
        )
        assert workflow.event_store.of_type(  # type: ignore[attr-defined]
            EventType.workflow_verify,
        ) == [], 'No workflow_verify event may be manufactured by the skip'
        assert workflow.event_store.of_type(  # type: ignore[attr-defined]
            EventType.phase_skipped, phase='verify',
        ) == [], 'The skip must not also skip VERIFY'

    async def test_durable_green_skip_still_requires_a_prior_green_at_tip(
        self, config, git_ops, task_assignment,
    ):
        """green_checkpoint_at_tip is not satisfied by an all-done plan.

        The only VERIFY skip is the durable verified-green checkpoint, which
        requires a PRIOR passing workflow_verify recorded at the EXACT tip — a
        condition a falsely pre-satisfied branch cannot meet.
        """
        from orchestrator.verify_checkpoint import (  # noqa: PLC0415
            green_checkpoint_at_tip,
        )

        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)
        _write_plan(artifacts, workflow)
        (wt / 'impl.py').write_text('x = 1\n')
        tip = await git_ops.commit(wt, 'feat: GREEN — x')
        for step_id in ('pre-1', 'step-1', 'step-2'):
            assert artifacts.mark_step_committed(step_id, sha=tip) is True
        workflow.plan = artifacts.read_plan()
        _stub_execute_collaborators(workflow)

        await workflow._execute_iterations()

        # An all-done plan plus the skip event is NOT a green checkpoint.
        assert green_checkpoint_at_tip(
            workflow.event_store, EventType.workflow_verify, workflow.task_id, tip,
        ) is False

        # A passing workflow_verify recorded at a DIFFERENT tip is still no skip.
        workflow.event_store.emit(  # type: ignore[attr-defined]
            EventType.workflow_verify,
            task_id=workflow.task_id,
            data={'passed': True, 'tip_sha': 'some-other-tip'},
        )
        assert green_checkpoint_at_tip(
            workflow.event_store, EventType.workflow_verify, workflow.task_id, tip,
        ) is False

        # Only a passing verify at THIS exact tip satisfies it.
        workflow.event_store.emit(  # type: ignore[attr-defined]
            EventType.workflow_verify,
            task_id=workflow.task_id,
            data={'passed': True, 'tip_sha': tip},
        )
        assert green_checkpoint_at_tip(
            workflow.event_store, EventType.workflow_verify, workflow.task_id, tip,
        ) is True


# ---------------------------------------------------------------------------
# step-13 RED: the execute_skipped work entry must fail CLOSED on provenance
# ---------------------------------------------------------------------------
#
# Review blocker (robustness / false-DONE at workflow.py:6193): the entry's
# steps_completed is derived purely from `status == 'done'` and ignores
# `commit` entirely. α's authoring-time guard `_sha_exists_on_branch`
# (mcp/plan_tools.py:514) is `git merge-base --is-ancestor <sha> HEAD`, which
# is satisfied by `base_commit` ITSELF and by every main ancestor below base.
# An architect can therefore pre-satisfy every step against base-reachable
# shas on a branch carrying ZERO commits beyond base and still get a
# `committed: True` work entry — which _iteration_entry_is_work classifies as
# real work, which lets _recover_before_merge resolve has_work=True and
# recovery-DONE a branch that implemented nothing.
#
# The fix belongs at the WRITER: only ids whose recorded commit is a genuine
# member of `base_commit..HEAD` may be named. `phase_skipped` stays
# unconditional (the skip really happened — honest observability), and now
# also reports the divergence between what the plan claims and what the
# branch can back.


_ABSENT_SHA = 'deadbeefdeadbeefdeadbeefdeadbeefdeadbeef'
"""Syntactically valid 40-hex sha that names no object in the test repo."""


async def _advance_main(config: OrchestratorConfig, marker: str) -> None:
    """Add one real commit to main so the worktree's base has an ancestor.

    Needed to build case (d): a done step whose commit is a REAL main commit
    that is nonetheless below base and therefore not beyond-base work.
    """
    root = config.project_root
    (root / f'{marker}.py').write_text(f'{marker} = 1\n')
    await _run(['git', 'add', '-A'], cwd=root)
    await _run(['git', 'commit', '-m', f'chore: {marker}'], cwd=root)


def _write_steps_plan(
    artifacts: TaskArtifacts,
    workflow: TaskWorkflow,
    steps: list[dict],
    prerequisites: list[dict] | None = None,
) -> None:
    """Write a plan with an arbitrary step/prerequisite shape."""
    artifacts.write_plan({
        'task_id': '42',
        'title': 'X',
        'analysis': 'A',
        'prerequisites': [dict(p) for p in (prerequisites or [])],
        'steps': [dict(s) for s in steps],
    })
    artifacts.stamp_plan_provenance(workflow.session_id)
    workflow.plan = artifacts.read_plan()


def _step(step_id: str) -> dict:
    return {
        'id': step_id, 'type': 'impl', 'description': f'do {step_id}',
        'status': 'pending', 'commit': None,
    }


def _skip_event(workflow: TaskWorkflow) -> dict:
    """The single phase_skipped(phase='execute') row, asserted to be unique."""
    skips = workflow.event_store.of_type(  # type: ignore[attr-defined]
        EventType.phase_skipped, phase='execute',
    )
    assert len(skips) == 1, (
        f'phase_skipped(execute) must stay unconditional and unique, got {skips}'
    )
    return skips[0]


def _skipped_entries(artifacts: TaskArtifacts) -> list[dict]:
    entries, corrupted = artifacts.read_iteration_log()
    assert corrupted == []
    return [e for e in entries if e.get('event') == 'execute_skipped']


@pytest.mark.asyncio
class TestSkippedExecuteEntryRequiresBeyondBaseProvenance:
    """The durable work entry must be backed by commits in ``base..HEAD``."""

    async def _worktree(self, config, git_ops, task_assignment):
        """Real worktree + workflow, base stamped, tip AT base."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        workflow, artifacts = _make_workflow(
            config, git_ops, task_assignment, wt_info.path,
        )
        artifacts.update_base_commit(wt_info.base_commit)
        _stub_execute_collaborators(workflow)
        return workflow, artifacts, wt_info.path, wt_info.base_commit

    # -- (a) PREMISE PIN ---------------------------------------------------

    async def test_alpha_guard_admits_base_and_below_base_shas(
        self, config, git_ops, task_assignment,
    ):
        """WHY status alone is insufficient: α's guard is reachability-only.

        If α's ``_sha_exists_on_branch`` is ever tightened to REJECT
        base-reachable shas, this assertion fails loudly and the writer-side
        filter below can be revisited rather than silently kept forever.
        """
        from orchestrator.mcp import plan_tools  # noqa: PLC0415

        await _advance_main(config, 'below_base')
        _rc, below_base_sha, _err = await _run(
            ['git', 'rev-parse', 'HEAD~1'], cwd=config.project_root,
        )
        wt_info = await git_ops.create_worktree(task_assignment.task_id)

        assert plan_tools._sha_exists_on_branch(
            wt_info.path, wt_info.base_commit,
        ) is True, 'PREMISE: the guard admits base_commit itself'
        assert plan_tools._sha_exists_on_branch(
            wt_info.path, below_base_sha,
        ) is True, 'PREMISE: the guard admits arbitrary main ancestors below base'
        assert plan_tools._sha_exists_on_branch(
            wt_info.path, _ABSENT_SHA,
        ) is False, 'PREMISE: the guard does reject a non-existent sha'

    # -- (b) every step pre-satisfied against base, zero beyond-base work --

    async def test_all_steps_against_base_sha_write_no_entry(
        self, config, git_ops, task_assignment,
    ):
        workflow, artifacts, _wt, base_sha = await self._worktree(
            config, git_ops, task_assignment,
        )
        _write_steps_plan(
            artifacts, workflow,
            steps=[_step('step-1'), _step('step-2')],
            prerequisites=[_step('pre-1')],
        )
        for item_id in ('pre-1', 'step-1', 'step-2'):
            assert artifacts.mark_step_committed(item_id, base_sha) is True
        assert artifacts.get_pending_steps() == []
        workflow.plan = artifacts.read_plan()

        outcome = await workflow._execute_iterations()

        assert outcome == WorkflowOutcome.DONE
        # Honest observability is unconditional — the skip DID happen.
        event = _skip_event(workflow)
        assert set(event['data']['done_step_ids']) == {'pre-1', 'step-1', 'step-2'}
        assert event['data']['committed_step_ids'] == [], (
            'No step may claim provenance the branch cannot back, got '
            f"{event['data']['committed_step_ids']}"
        )
        # …but no work entry, so the merge guard cannot recovery-DONE.
        assert _skipped_entries(artifacts) == [], (
            'A branch with zero commits beyond base must write no work entry'
        )
        assert workflow._has_prior_implementation(wt_head=None).has_work is False, (
            '_recover_before_merge must keep the explicitly-handled spurious '
            'merge-signal path, not recovery-DONE a branch with no work'
        )

    # -- (c) done with NO commit at all ------------------------------------

    async def test_commit_less_done_step_is_excluded(
        self, config, git_ops, task_assignment,
    ):
        """``update_step_status(id, 'done')`` permits ``commit=None``
        (artifacts.py:708) — status alone must never count as provenance,
        even on a branch that DOES carry real beyond-base commits."""
        workflow, artifacts, wt, _base = await self._worktree(
            config, git_ops, task_assignment,
        )
        _write_steps_plan(artifacts, workflow, steps=[_step('step-1')])
        (wt / 'impl.py').write_text('x = 1\n')
        real_sha = await git_ops.commit(wt, 'feat: real beyond-base work')
        assert real_sha

        artifacts.update_step_status('step-1', 'done')
        assert artifacts.get_pending_steps() == []
        assert artifacts.read_plan()['steps'][0]['commit'] is None
        workflow.plan = artifacts.read_plan()

        outcome = await workflow._execute_iterations()

        assert outcome == WorkflowOutcome.DONE
        event = _skip_event(workflow)
        assert event['data']['done_step_ids'] == ['step-1']
        assert event['data']['committed_step_ids'] == []
        assert _skipped_entries(artifacts) == [], (
            'A commit-less done step is not provenance for anything'
        )

    # -- (d) done with a commit OUTSIDE base..HEAD -------------------------

    async def test_below_base_and_absent_shas_are_excluded(
        self, config, git_ops, task_assignment,
    ):
        await _advance_main(config, 'below_base')
        _rc, below_base_sha, _err = await _run(
            ['git', 'rev-parse', 'HEAD~1'], cwd=config.project_root,
        )
        workflow, artifacts, wt, base_sha = await self._worktree(
            config, git_ops, task_assignment,
        )
        assert below_base_sha != base_sha, 'Setup: need a real ancestor below base'
        _write_steps_plan(
            artifacts, workflow, steps=[_step('step-1'), _step('step-2')],
        )
        (wt / 'impl.py').write_text('x = 1\n')
        assert await git_ops.commit(wt, 'feat: real beyond-base work')

        assert artifacts.mark_step_committed('step-1', below_base_sha) is True
        assert artifacts.mark_step_committed('step-2', _ABSENT_SHA) is True
        assert artifacts.get_pending_steps() == []
        workflow.plan = artifacts.read_plan()

        outcome = await workflow._execute_iterations()

        assert outcome == WorkflowOutcome.DONE
        event = _skip_event(workflow)
        assert set(event['data']['done_step_ids']) == {'step-1', 'step-2'}
        assert event['data']['committed_step_ids'] == [], (
            'Neither a below-base main commit nor an absent sha is beyond-base work'
        )
        assert _skipped_entries(artifacts) == []

    # -- (e) MIXED plan ----------------------------------------------------

    async def test_mixed_plan_names_only_the_beyond_base_step(
        self, config, git_ops, task_assignment,
    ):
        workflow, artifacts, wt, base_sha = await self._worktree(
            config, git_ops, task_assignment,
        )
        _write_steps_plan(
            artifacts, workflow,
            steps=[_step('step-good'), _step('step-base'), _step('step-bare')],
        )
        (wt / 'impl.py').write_text('x = 1\n')
        good_sha = await git_ops.commit(wt, 'feat: real beyond-base work')
        assert good_sha

        assert artifacts.mark_step_committed('step-good', good_sha) is True
        assert artifacts.mark_step_committed('step-base', base_sha) is True
        artifacts.update_step_status('step-bare', 'done')
        assert artifacts.get_pending_steps() == []
        workflow.plan = artifacts.read_plan()

        outcome = await workflow._execute_iterations()

        assert outcome == WorkflowOutcome.DONE
        event = _skip_event(workflow)
        assert set(event['data']['done_step_ids']) == {
            'step-good', 'step-base', 'step-bare',
        }, 'The full claimed set stays visible in the event log'
        assert event['data']['committed_step_ids'] == ['step-good'], (
            'The divergence between claim and provenance must be visible, got '
            f"{event['data']['committed_step_ids']}"
        )

        entries = _skipped_entries(artifacts)
        assert len(entries) == 1, f'Expected exactly one entry, got {entries}'
        entry = entries[0]
        assert entry['agent'] == 'architect'
        assert entry['steps_completed'] == ['step-good']
        assert 'step-base' not in entry['steps_completed'], (
            'A base-reachable sha is not beyond-base work'
        )
        assert 'step-bare' not in entry['steps_completed']
        assert entry['committed'] is True

    # -- (f) FAIL-CLOSED on a detector failure -----------------------------

    async def _all_committed_for_real(self, config, git_ops, task_assignment):
        """All steps pre-satisfied against genuine beyond-base shas."""
        workflow, artifacts, wt, _base = await self._worktree(
            config, git_ops, task_assignment,
        )
        _write_steps_plan(
            artifacts, workflow,
            steps=[_step('step-1'), _step('step-2')],
            prerequisites=[_step('pre-1')],
        )
        (wt / 'red.py').write_text('def test_x(): assert True\n')
        red_sha = await git_ops.commit(wt, 'test: RED — x')
        (wt / 'impl.py').write_text('x = 1\n')
        green_sha = await git_ops.commit(wt, 'feat: GREEN — x')
        assert red_sha and green_sha
        for item_id, sha in (
            ('pre-1', red_sha), ('step-1', red_sha), ('step-2', green_sha),
        ):
            assert artifacts.mark_step_committed(item_id, sha) is True
        assert artifacts.get_pending_steps() == []
        workflow.plan = artifacts.read_plan()
        return workflow, artifacts

    async def test_detector_raising_writes_no_entry_and_does_not_raise(
        self, config, git_ops, task_assignment,
    ):
        workflow, artifacts = await self._all_committed_for_real(
            config, git_ops, task_assignment,
        )
        workflow._detect_committed_branch_work = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError('git exploded'),
        )

        outcome = await workflow._execute_iterations()

        assert outcome == WorkflowOutcome.DONE, 'A detector failure must not sink EXECUTE'
        event = _skip_event(workflow)
        assert event['data']['committed_step_ids'] == []
        assert _skipped_entries(artifacts) == [], (
            'No evidence ⇒ no work entry (fail closed)'
        )

    async def test_detector_returning_empty_writes_no_entry(
        self, config, git_ops, task_assignment,
    ):
        workflow, artifacts = await self._all_committed_for_real(
            config, git_ops, task_assignment,
        )
        workflow._detect_committed_branch_work = AsyncMock(  # type: ignore[method-assign]
            return_value=[],
        )

        outcome = await workflow._execute_iterations()

        assert outcome == WorkflowOutcome.DONE
        event = _skip_event(workflow)
        assert event['data']['committed_step_ids'] == []
        assert _skipped_entries(artifacts) == []

    # -- (g) REGRESSION ----------------------------------------------------

    async def test_genuinely_all_committed_plan_still_writes_the_entry(
        self, config, git_ops, task_assignment,
    ):
        """The tightening rejects ONLY unbacked provenance."""
        workflow, artifacts = await self._all_committed_for_real(
            config, git_ops, task_assignment,
        )

        outcome = await workflow._execute_iterations()

        assert outcome == WorkflowOutcome.DONE
        event = _skip_event(workflow)
        assert set(event['data']['committed_step_ids']) == {
            'pre-1', 'step-1', 'step-2',
        }
        entries = _skipped_entries(artifacts)
        assert len(entries) == 1
        assert set(entries[0]['steps_completed']) == {'pre-1', 'step-1', 'step-2'}
        assert entries[0]['committed'] is True
        assert workflow._has_prior_implementation(wt_head=None).has_work is True
