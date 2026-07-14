"""Non-fixture test helpers for orchestrator workflow tests.

Lives outside conftest.py to avoid the `sys.modules['conftest']` collision
that arises when root-level pytest loads multiple subprojects' conftests in
the same process under --import-mode=importlib. See task 977 for convention
rationale.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from _orch_helpers import pydantic_spec, wire_scheduler_liveness_mock
from escalation.queue import EscalationQueue

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import _run
from orchestrator.harness import Harness
from orchestrator.landed_outbox import LandedOutbox, LandedRow, MergeProvenance
from orchestrator.workflow import TaskWorkflow
from orchestrator.workflow_types import StewardResolved


class FakeMcp:
    """Minimal McpLifecycle stand-in."""

    @property
    def url(self) -> str:
        return 'http://localhost:9999'

    def mcp_config_json(self, escalation_url: str | None = None) -> dict:
        return {'mcpServers': {}}


class FakeScheduler:
    """Scheduler that tracks status changes without HTTP calls."""

    def __init__(self):
        self.statuses: dict[str, list[str]] = {}
        self.provenance: dict[str, dict] = {}
        self.reopen_reasons: dict[str, str] = {}
        # Optional task data store for tests that exercise the bypass-detection
        # path, which calls get_task to read metadata.done_provenance.
        self.task_data: dict[str, dict] = {}
        # PRD task-status-authority C4/D4 (task 2188, omega1): last-seen
        # claimant fields, tracked the same way as provenance/reopen_reasons.
        self.claimant_run_ids: dict[str, str | None] = {}
        self.heartbeats: dict[str, str | None] = {}

    async def set_task_status(
        self,
        task_id: str,
        status: str,
        *,
        done_provenance: dict | None = None,
        reopen_reason: str | None = None,
        claimant_run_id: str | None = None,
        heartbeat_at: str | None = None,
    ) -> None:
        self.statuses.setdefault(task_id, []).append(status)
        if done_provenance is not None:
            self.provenance[task_id] = done_provenance
        if reopen_reason is not None:
            self.reopen_reasons[task_id] = reopen_reason
        if claimant_run_id is not None:
            self.claimant_run_ids[task_id] = claimant_run_id
        if heartbeat_at is not None:
            self.heartbeats[task_id] = heartbeat_at

    async def set_task_claimant(
        self,
        task_id: str,
        *,
        claimant_run_id: str | None = None,
        heartbeat_at: str | None = None,
    ) -> None:
        """Status-untouching claimant write — mirrors Scheduler.set_task_claimant."""
        self.claimant_run_ids[task_id] = claimant_run_id
        self.heartbeats[task_id] = heartbeat_at

    async def mark_done(
        self,
        task_id: str,
        *,
        kind: str,
        sha: str,
        note: str | None = None,
    ) -> None:
        """Mirror Scheduler.mark_done: forward to set_task_status with provenance."""
        provenance: dict = {'kind': kind, 'commit': sha}
        if note is not None:
            provenance['note'] = note
        await self.set_task_status(
            task_id, 'done', done_provenance=provenance,
        )

    async def handle_blast_radius_expansion(
        self,
        task_id: str,
        current: list[str],
        needed: list[str],
        /,
        *,
        persist_files: list[str] | None = None,
    ) -> bool:
        return True

    async def get_status(self, task_id: str) -> str | None:
        history = self.statuses.get(task_id)
        return history[-1] if history else None

    async def get_task(self, task_id: str) -> dict | None:
        return self.task_data.get(task_id)

    async def get_tasks(self, *, statuses: Iterable[str] | None = None) -> list[dict]:
        """Return all tracked task data as a list."""
        return list(self.task_data.values())

    async def get_statuses(
        self, ids: list[str] | None = None,
    ) -> tuple[dict[str, str], Exception | None]:
        """Return latest status per id, filtered by ids if provided."""
        latest = {tid: hist[-1] for tid, hist in self.statuses.items() if hist}
        if ids is not None:
            latest = {tid: latest[tid] for tid in ids if tid in latest}
        return latest, None

    async def update_task(
        self, task_id: str, metadata: str | dict, *, append: bool = False,
    ) -> bool:
        return True

    async def dispatch_tool(
        self, name: str, arguments: dict, *, timeout: float = 15
    ) -> dict:
        return {}

    def release(self, task_id: str) -> None:
        pass

    async def tasks_by_train(self, train_id: str, /) -> list[dict]:
        return []

    def clear_requeue_count(self, task_id: str, /) -> None:
        pass


class FakeBriefing:
    """BriefingAssembler that returns canned prompts."""

    async def build_architect_prompt(self, task: dict, worktree=None, context: str | None = None) -> str:
        return f'Plan task: {task.get("title", "")}'

    async def build_plan_completion_prompt(
        self, task: dict, partial_plan: dict, worktree=None,
        context: str | None = None,
    ) -> str:
        return f'Complete partial plan: {task.get("title", "")}'

    async def build_implementer_prompt(
        self, plan: dict, iteration_log: list, context: str | None = None,
        rebase_notice: dict | None = None, task_id: str | None = None,
        wip_notice: list[dict] | None = None,
    ) -> str:
        return 'Implement the plan'

    async def build_debugger_prompt(
        self, failures: str, plan: dict, context: str | None = None,
        task_id: str | None = None,
    ) -> str:
        return f'Fix: {failures[:100]}'

    async def build_reviewer_prompt(
        self, reviewer_type: str, diff: str, context: str | None = None
    ) -> str:
        return f'Review ({reviewer_type}): {diff[:100]}'

    async def build_completion_judge_prompt(
        self,
        plan: dict,
        iteration_log: list,
        diff: str,
        task_id: str | None = None,
        context: str | None = None,
    ) -> str:
        return f'Judge task {task_id}: plan has {len(plan.get("steps", []))} steps'

    async def build_merger_prompt(
        self, conflicts: str, task_intent: str, context: str | None = None
    ) -> str:
        return f'Merge: {conflicts[:100]}'

    async def build_resume_prompt(
        self,
        task: dict,
        plan: dict,
        escalation_summary: str,
        resolution: str,
        worktree=None,
    ) -> str:
        return f'Resume: {resolution[:100]}'

    async def build_amender_prompt(
        self, plan: dict, iteration_log: list, suggestions: list, locked_modules: list,
        context: str | None = None, task_id: str | None = None,
    ) -> str:
        return 'Amend the plan'

    async def build_revalidation_prompt(
        self, task: dict, existing_plan: dict, changed_files: list,
        worktree=None, context: str | None = None,
    ) -> str:
        return f'Revalidate: {task.get("title", "")}'

    async def build_plan_tightening_prompt(
        self, task: dict, plan: dict, not_touched: list,
        worktree=None, context: str | None = None,
    ) -> str:
        return f'Tighten plan for: {task.get("title", "")}'

    async def build_simple_task_prompt(
        self, task: dict, worktree=None, context: str | None = None,
    ) -> str:
        return f'Simple task: {task.get("title", "")}'


def _make_resolving_steward(queue: EscalationQueue, task_id: str) -> type:
    """Return a steward class that resolves all pending L0 escalations in start().

    Used by TestMarkBlockedFalseDoneGuard tests to drive _mark_blocked through
    the post-steward `if not remaining:` branch without a full workflow.run() cycle.
    """

    class _FakeSteward:
        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002
            self._outcome_channel = None
            self._wip_probe = None

        def set_outcome_channel(self, channel) -> None:
            self._outcome_channel = channel

        def set_wip_probe(self, probe) -> None:
            self._wip_probe = probe

        async def start(self) -> None:
            pending = queue.get_by_task(task_id, status='pending', level=0)
            assert pending, 'expected at least one pending L0 escalation to resolve'
            for esc in pending:
                queue.resolve(
                    esc.id, 'Resolved by FakeSteward', resolved_by='fake-steward',
                )
            if self._outcome_channel is not None:
                self._outcome_channel.put_nowait(
                    StewardResolved(resolution_text='Resolved by FakeSteward'),
                )

        async def stop(self) -> None:
            pass

    return _FakeSteward


def _make_status_setting_steward(
    queue: EscalationQueue, scheduler, task_id: str, final_status: str,
) -> type:
    """FakeSteward that resolves L0s and then sets task status to final_status.

    Simulates a steward that marked the task deferred/blocked after inspecting
    the escalation — the workflow must NOT overwrite that decision with
    pending on its auto-requeue path.
    """

    class _FakeSteward:
        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002
            self._outcome_channel = None
            self._wip_probe = None

        def set_outcome_channel(self, channel) -> None:
            self._outcome_channel = channel

        def set_wip_probe(self, probe) -> None:
            self._wip_probe = probe

        async def start(self) -> None:
            pending = queue.get_by_task(task_id, status='pending', level=0)
            for esc in pending:
                queue.resolve(esc.id, 'Resolved', resolved_by='fake-steward')
            await scheduler.set_task_status(task_id, final_status)
            if self._outcome_channel is not None:
                self._outcome_channel.put_nowait(
                    StewardResolved(resolution_text='Resolved'),
                )

        async def stop(self) -> None:
            pass

    return _FakeSteward


# ---------------------------------------------------------------------------
# Merge-provenance workflow fixture factory (moved from
# test_workflow_merge_provenance.py — task 2610, Group A).
# ---------------------------------------------------------------------------


@dataclass
class _Fixture:
    wf: TaskWorkflow
    artifacts: TaskArtifacts
    set_task_status: AsyncMock
    mark_done: AsyncMock
    update_task: AsyncMock
    is_ancestor: AsyncMock
    get_main_sha: AsyncMock


def _make(
    *,
    worktree: Path,
    project_root: Path,
    task_id: str = '50',
    branch_on_main: bool = True,
    main_sha: str = 'mainsha123',
) -> _Fixture:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = []

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = project_root

    set_task_status = AsyncMock()
    scheduler = MagicMock()
    scheduler.set_task_status = set_task_status
    # Fix 1 (mirrors test_workflow_already_done.py): workflow refreshes
    # metadata.files via update_task before set_task_status('done').
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.get_status = AsyncMock(return_value=None)

    # Forward mark_done into set_task_status so assertions can observe the
    # (task_id, 'done', done_provenance=...) call shape either way.
    async def _fake_mark_done(tid, *, kind, sha, note=None):
        provenance: dict = {'kind': kind, 'commit': sha}
        if note is not None:
            provenance['note'] = note
        await set_task_status(tid, 'done', done_provenance=provenance)
    mark_done = AsyncMock(side_effect=_fake_mark_done)
    scheduler.mark_done = mark_done

    is_ancestor = AsyncMock(return_value=branch_on_main)
    get_main_sha = AsyncMock(return_value=main_sha)
    git_ops = MagicMock()
    git_ops.is_ancestor = is_ancestor
    git_ops.get_main_sha = get_main_sha

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )

    worktree.mkdir(parents=True, exist_ok=True)
    artifacts = TaskArtifacts(worktree)
    artifacts.init(task_id, 'T', 'd', base_commit='oldbase')
    wf.artifacts = artifacts
    wf.worktree = worktree

    return _Fixture(
        wf=wf, artifacts=artifacts,
        set_task_status=set_task_status,
        mark_done=mark_done,
        update_task=scheduler.update_task,
        is_ancestor=is_ancestor,
        get_main_sha=get_main_sha,
    )


def _bind_landed_row(tmp_path: Path, *, task_id: str, advanced_sha: str) -> None:
    """Bind a real LandedOutbox (via MergeProvenance.bind) holding a row for *task_id*."""
    outbox = LandedOutbox(tmp_path / 'landed.json')
    outbox.record(LandedRow(
        task_id=task_id, branch_tip_sha='branchtip', advanced_sha=advanced_sha,
        landed_at=1.0,
    ))
    MergeProvenance.bind(outbox)


# ---------------------------------------------------------------------------
# Warm-lane workflow factory (moved from test_workflow_warm_lane_requeue.py —
# task 2610, Group B — renamed from the bare `_make_workflow` since that name
# is reused ~30x across the tests dir with different signatures).
# ---------------------------------------------------------------------------


def _make_warmlane_workflow(*, tmp_path: Path, task_id: str = '1859') -> TaskWorkflow:
    """Build a minimal TaskWorkflow with mocked deps for create_worktree tests.

    Unlike the _make_workflow in test_workflow_worktree_missing.py, this one
    does NOT pre-populate wf.worktree — we want run() to call create_worktree
    so the exception raised there propagates through run()'s exception handlers.
    """
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'Test task', 'description': 'desc'}
    assignment.modules = []  # empty → _resolve_module_configs returns [] cleanly

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = tmp_path / 'proj'
    # worktree_identity_guard_enabled is read inside _setup_worktree_and_artifacts
    # before create_worktree; let MagicMock return a truthy value (the default).

    scheduler = MagicMock()
    # get_status: return 'pending' so the pre-empt check doesn't raise TerminalExitRejection
    scheduler.get_status = AsyncMock(return_value='pending')
    scheduler.set_task_status = AsyncMock()

    git_ops = MagicMock()
    # create_worktree will be configured per-test

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    # Do NOT set wf.worktree — keep it None so run() calls create_worktree.
    return wf


# ---------------------------------------------------------------------------
# Harness factories (moved from test_harness_warm_lane_wiring.py —
# task 2610, Group C).
# ---------------------------------------------------------------------------


def _build_harness(config: OrchestratorConfig) -> Harness:
    """Construct a Harness with heavy constructors patched out."""
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        harness = Harness(config)
    # `Scheduler` is patched to a bare MagicMock; wire real (non-auto-mock)
    # liveness-accessor behaviour (task 2235: harness.py now calls
    # scheduler.is_dispatched()/.is_actively_held()/.workflow_cancel_recent()
    # instead of reaching into _dispatched/lock_table._held directly) so
    # tests below that set harness.scheduler._dispatched (or .lock_table)
    # exercise real semantics instead of an auto-mocked (always-truthy) stub.
    wire_scheduler_liveness_mock(harness.scheduler)  # type: ignore[arg-type]
    return harness


async def _init_git_repo(repo: Path) -> None:
    from orchestrator.git_ops import _run
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'init'], cwd=repo)


# ---------------------------------------------------------------------------
# Protocol-conformance assertion (static-only; never executes at runtime).
# Mirrors the if TYPE_CHECKING / SchedulerFacade conformance block near the
# bottom of test_workflow_e2e.py.
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from orchestrator.scheduler import SchedulerFacade

    _fake_scheduler_conforms: SchedulerFacade = FakeScheduler()
