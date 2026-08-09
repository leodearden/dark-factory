"""Non-fixture test helpers for orchestrator workflow tests.

Lives outside conftest.py to avoid the `sys.modules['conftest']` collision
that arises when root-level pytest loads multiple subprojects' conftests in
the same process under --import-mode=importlib. See task 977 for convention
rationale.

One exception to "non-fixture": `_derive_meta_root_like_production` (Group D,
task 2610) is an autouse pytest fixture, moved here anyway because it is
cross-imported and semantically paired with `AgentStub` — see its own
docstring. Like every other factory in this module, importing it by name is
required to activate it (see that docstring for the activation mechanics).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec, wire_scheduler_liveness_mock
from escalation.queue import EscalationQueue

from orchestrator.agents.invoke import AgentResult
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.harness import Harness
from orchestrator.landed_outbox import LandedOutbox, LandedRow, MergeProvenance
from orchestrator.mcp.verdict_tools import (
    _envelope as _verdict_envelope,
)
from orchestrator.mcp.verdict_tools import (
    _submit_review_verdict,
)
from orchestrator.module_charter import sanitize_files_for_persist
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
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
        # Scope-reconciliation choke point (task 2505): every call's
        # (current, needed, persist_files) is recorded here so tests can
        # assert on what was persisted, and blast_radius_result configures
        # the return value (default True == lock acquired successfully;
        # tests force False to simulate a sibling lock conflict).
        self.blast_radius_calls: list[tuple[list[str], list[str], list[str] | None]] = []
        self.blast_radius_result: bool = True
        # Scope-grant direct metadata.files persist seam (task 2505, step-18):
        # every update_task(task_id, metadata) call is recorded here so the
        # same-module scope-grant persist path (which writes metadata.files
        # directly, bypassing handle_blast_radius_expansion) can be asserted.
        # Purely additive — the existing no-op `return True` is preserved.
        self.update_task_calls: list[tuple[str, str | dict]] = []

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
        self.blast_radius_calls.append((current, needed, persist_files))
        return self.blast_radius_result

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
        self.update_task_calls.append((task_id, metadata))
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

    def note_merge_phase_entered(self, tid: str) -> None:
        pass

    def clear_merge_phase(self, tid: str) -> None:
        pass


class FakeMetadataBackend:
    """Merge-faithful ``metadata`` backend for workflow tests (task 3579).

    Unlike ``FakeScheduler`` — whose ``update_task`` records call arguments and
    persists nothing — this fake keeps a real ``blob`` and models BOTH:

    - fused-memory's #1827 ``metadata_mode`` contract, with the same precedence
      ``Scheduler.update_task`` resolves (scheduler.py:3793-3799):
      ``metadata_mode`` > ``append=True`` -> ``'additive'`` > ``'merge'``.
      ``'merge'`` is shallow last-write-wins (supplied keys overwrite wholesale,
      omitted keys are preserved), ``'replace'`` is a whole-blob overwrite, and
      ``'additive'`` is list-union for list values / overwrite otherwise.
    - the scheduler-side ``metadata.files`` persist that
      ``handle_blast_radius_expansion(persist_files=...)`` performs via
      ``Scheduler._persist_files_metadata`` (scheduler.py:6890),
      ``merged = {**fresh_md, 'files': sanitize_files_for_persist(files)}``.

    It exists so tests can assert on BACKEND STATE rather than merely on call
    arguments — a whole-blob write that reverts a freshly-persisted
    ``metadata.files`` is invisible to a call-argument assertion but obvious in
    ``self.blob``.

    ``update_task`` accepts *metadata* both positionally and as the ``metadata=``
    keyword on purpose: workflow stamps call it either way, and a signature
    mismatch would raise ``TypeError`` inside a caller's blanket
    ``except Exception``, silently leaving the blob untouched.

    Opt-in: wire it into a test's existing ``MagicMock()`` scheduler by
    rebinding ``update_task`` / ``handle_blast_radius_expansion`` / ``get_task``.
    ``FakeScheduler`` is deliberately NOT changed — it has ~20 importers and
    ``test_workflow_dry_run_hook.py`` pins that passing ``metadata_mode`` to it
    raises ``TypeError``.
    """

    def __init__(self, initial: dict | None = None):
        self.blob: dict = dict(initial or {})
        self.update_task_calls: list[dict] = []
        self.blast_radius_calls: list[tuple[list[str], list[str], list[str] | None]] = []
        self.blast_radius_result: bool = True

    async def update_task(
        self,
        task_id: str,
        metadata: str | dict | None = None,
        *,
        append: bool = False,
        metadata_mode: str | None = None,
    ) -> bool:
        self.update_task_calls.append({
            'task_id': task_id,
            'metadata': metadata,
            'append': append,
            'metadata_mode': metadata_mode,
        })
        payload = json.loads(metadata) if isinstance(metadata, str) else (metadata or {})
        # Mirror Scheduler.update_task's precedence exactly.
        if metadata_mode is not None:
            effective = metadata_mode
        elif append:
            effective = 'additive'
        else:
            effective = 'merge'

        if effective == 'additive':
            for key, value in payload.items():
                existing = self.blob.get(key)
                if isinstance(existing, list) and isinstance(value, list):
                    self.blob[key] = existing + value
                else:
                    self.blob[key] = value
        elif effective == 'replace':
            self.blob = dict(payload)
        else:
            self.blob = {**self.blob, **payload}
        return True

    async def handle_blast_radius_expansion(
        self,
        task_id: str,
        current: list[str],
        needed: list[str],
        /,
        *,
        persist_files: list[str] | None = None,
    ) -> bool:
        self.blast_radius_calls.append((current, needed, persist_files))
        if self.blast_radius_result and persist_files is not None:
            self.blob = {
                **self.blob,
                'files': sanitize_files_for_persist(persist_files),
            }
        return self.blast_radius_result

    async def get_task(self, task_id: str) -> dict:
        return {'id': task_id, 'metadata': dict(self.blob)}


class FakeBriefing:
    """BriefingAssembler that returns canned prompts."""

    async def build_architect_prompt(
        self, task: dict, worktree=None, context: str | None = None,
        *, include_prior_proposals: bool = False,
        committed_work: list[dict] | None = None,
    ) -> str:
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
        self, reviewer_type: str, diff: str, context: str | None = None,
        amendment_suggestions: list[dict] | None = None,
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
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'init'], cwd=repo)


# ---------------------------------------------------------------------------
# e2e factories (moved from test_workflow_e2e.py — task 2610, Group D, final
# decoupling): PLAN, _make_review, AgentStub, _build_workflow,
# _build_workflow_with_escalation, _init_repo, and the autouse fixture
# _derive_meta_root_like_production.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _derive_meta_root_like_production(monkeypatch):
    """Mirror ε1 (task 2258) meta_root derivation in the agent-side doubles.

    In production the workflow's ``self.artifacts`` is constructed with the
    derived sibling ``.task-meta/<name>`` root
    (``TaskArtifacts(worktree, _meta_root_for_worktree(worktree))``) and the
    real ``plan_tools.py`` MCP writer is launched with the *same* ``--meta-root``
    — so every reader/writer for a worktree resolves through one consistent
    ``meta_root``.

    These e2e doubles (``AgentStub._architect`` / ``_implementer`` and ~50
    other call sites) bypass the real ``plan_tools.py`` subprocess and emulate
    agent-side writes by hand-constructing ``TaskArtifacts(worktree)`` with no
    ``meta_root`` (legacy-only, ``root == worktree/'.task'``). After the
    workflow's migrate-on-write methods (``stamp_plan_provenance``/``lock_plan``)
    establish the NEW copy, those legacy-only doubles write step status only to
    the legacy copy — invisible to the workflow's relocated ``self.artifacts``
    — so ``get_pending_steps()`` never observes completion and the task blocks
    with "Execution iterations exhausted".

    Rather than thread the derived root through every call site, patch the
    constructor to default ``meta_root`` to the same sibling root production
    derives whenever it is omitted. Explicit ``meta_root`` args (the workflow's
    own and ``plan_tools.py``'s) are untouched.
    """
    orig_init = TaskArtifacts.__init__

    def _init(self, worktree, meta_root=None):
        if meta_root is None:
            meta_root = TaskArtifacts.meta_root_for(worktree.parent, worktree.name)
        orig_init(self, worktree, meta_root)

    monkeypatch.setattr(TaskArtifacts, '__init__', _init)


async def _init_repo(repo: Path):
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    # Seed with a simple Python file so the repo isn't empty
    (repo / 'lib.py').write_text('def greet(name: str) -> str:\n    return f"Hello, {name}"\n')
    (repo / 'test_lib.py').write_text(
        'from lib import greet\n\ndef test_greet():\n    assert greet("world") == "Hello, world"\n'
    )
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


PLAN = {
    'task_id': '42',
    'title': 'Add farewell function',
    'files': ['lib.py', 'test_lib.py'],
    'analysis': 'Simple function addition with TDD',
    'prerequisites': [],
    'steps': [
        {
            'id': 'step-1',
            'type': 'test',
            'description': 'Write failing test for farewell()',
            'status': 'pending',
            'commit': None,
        },
        {
            'id': 'step-2',
            'type': 'impl',
            'description': 'Implement farewell() to pass test',
            'status': 'pending',
            'commit': None,
        },
    ],
    'design_decisions': [
        {'decision': 'Mirror greet() signature', 'rationale': 'Consistency'},
    ],
    'reuse': [],
    # Models a successful architect run: the agent called confirm_plan, so the
    # plan carries the completeness marker the workflow now gates on.
    '_finalized_at': '2026-01-01T00:00:00+00:00',
}


def _make_review(reviewer: str, verdict: str = 'PASS', issues: list | None = None):
    return {
        'reviewer': reviewer,
        'verdict': verdict,
        'issues': issues or [],
        'summary': f'{reviewer} review: {verdict}',
    }


class AgentStub:
    """Deterministic agent stub that performs real file operations.

    Tracks which roles have been invoked and their order.
    """

    def __init__(
        self,
        verify_results: list[VerifyResult] | None = None,
        judge_verdicts: list | None = None,
    ):
        self.calls: list[str] = []
        # Last env_overrides seen per role — set by invoke_agent before dispatch.
        self.env_overrides_by_role: dict[str, dict[str, str] | None] = {}
        self._impl_iteration = 0
        # Sequence of verify results to return (pops from front)
        self._verify_results = list(verify_results or [VerifyResult(
            passed=True, test_output='', lint_output='', type_output='',
            summary='All checks passed',
        )])
        # Sequence of judge verdicts to return (pops from front; last persists).
        # Each entry may be a dict (structured verdict), an Exception to raise,
        # None (to simulate structured_output=None), or the string 'SUCCESS_FALSE'
        # to simulate result.success=False.
        self._judge_verdicts = list(judge_verdicts or [])

    async def invoke_agent(
        self,
        prompt: str,
        system_prompt: str,
        cwd: Path,
        model: str = 'opus',
        max_turns: int = 50,
        max_budget_usd: float = 5.0,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        mcp_config: dict | None = None,
        output_schema: dict | None = None,
        permission_mode: str = 'bypassPermissions',
        sandbox_modules: list[str] | None = None,
        sandbox_extras: list[str] | None = None,
        effort: str | None = None,
        backend: str = 'claude',
        oauth_token: str | None = None,
        resume_session_id: str | None = None,
        session_id: str | None = None,
        timeout_seconds: float | None = None,
        config_dir: Path | None = None,
        env_overrides: dict[str, str] | None = None,
        spawn_env: dict[str, str] | None = None,
        startup_grace_secs: float = 120.0,
        working_idle_secs: float | None = None,
        absolute_cap_secs: float | None = None,
        prices: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Determine role from system_prompt content, perform side effects."""
        role = self._detect_role(system_prompt)
        self.calls.append(role)
        self.env_overrides_by_role[role] = env_overrides

        if role == 'architect':
            return await self._architect(cwd)
        elif role == 'implementer':
            return await self._implementer(cwd)
        elif role == 'debugger':
            return await self._debugger(cwd)
        elif role.startswith('reviewer'):
            result = self._reviewer(role, output_schema)
            if result.structured_output is not None:
                # Mirror the real submit_review_verdict tool call by routing
                # through the actual _submit_review_verdict implementation
                # (not a hand-built write_verdict envelope), so the
                # reviewer==role identity check (verdict_tools.py:86-94) is
                # exercised on this integration path too — a regression
                # where the reviewer payload's 'reviewer' field diverges
                # from the role would otherwise pass silently here even
                # though the real tool would reject it (reviewer_comprehensive
                # amendment, task 2484). Written to the .task-meta META_ROOT
                # the workflow's self.artifacts reads from — NOT
                # TaskArtifacts(cwd) (the legacy .task root), or the write
                # would land where _run_reviewer never reads. Same root
                # pattern as the _architect stub below.
                rev_artifacts = TaskArtifacts(
                    cwd, TaskArtifacts.meta_root_for(cwd.parent, cwd.name)
                )
                so = result.structured_output
                _submit_review_verdict(
                    rev_artifacts, role, session_id or 'sid',
                    so.get('reviewer'), so.get('verdict'),
                    so.get('issues'), so.get('summary'),
                )
            return result
        elif role == 'merger':
            return self._merger()
        elif role == 'judge':
            return self._judge(cwd, session_id or 'sid')
        else:
            return AgentResult(success=True, output='OK')

    def _detect_role(self, system_prompt: str) -> str:
        # Judge check goes first — its system prompt does not contain
        # architect/implementer/reviewer substrings so ordering is safe.
        if 'completion judge' in system_prompt.lower():
            return 'judge'
        if 'TDD architect' in system_prompt:
            return 'architect'
        if 'TDD implementer' in system_prompt:
            return 'implementer'
        if 'debugger' in system_prompt.lower() and 'You are a debugger' in system_prompt:
            return 'debugger'
        if 'code reviewer' in system_prompt.lower():
            return 'reviewer_comprehensive'
        if 'merge conflict resolver' in system_prompt.lower():
            return 'merger'
        # If re-planning (architect called with review feedback)
        if 'blocking issues were found' in system_prompt or 'Update the plan' in system_prompt:
            return 'architect'
        return 'unknown'

    async def _architect(self, cwd: Path) -> AgentResult:
        """Write plan.json to the task's meta_root.

        The workflow constructs its TaskArtifacts with the relocated
        ``.task-meta`` sibling root (task 2258 W11-ε1) and reads plan.json from
        there — NOT the legacy ``<worktree>/.task``. The real agent-side
        plan-tools MCP writes to the same root via ``--meta-root``; this stub
        must mirror that or the workflow sees "no plan.json produced".
        """
        artifacts = TaskArtifacts(
            cwd, TaskArtifacts.meta_root_for(cwd.parent, cwd.name)
        )
        artifacts.root.mkdir(parents=True, exist_ok=True)
        plan = dict(PLAN)
        plan['_schema_version'] = 1
        (artifacts.root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        return AgentResult(success=True, output='Plan created', cost_usd=0.50)

    async def _implementer(self, cwd: Path) -> AgentResult:
        """Create code files and update plan.json step statuses."""
        self._impl_iteration += 1
        artifacts = TaskArtifacts(
            cwd, TaskArtifacts.meta_root_for(cwd.parent, cwd.name)
        )
        pending = artifacts.get_pending_steps()

        if not pending:
            return AgentResult(success=True, output='Nothing to do')

        completed_ids = []
        for step in pending:
            step_id = step['id']
            if step['type'] == 'test':
                # Write failing test
                (cwd / 'test_lib.py').write_text(
                    'from lib import greet, farewell\n\n'
                    'def test_greet():\n'
                    '    assert greet("world") == "Hello, world"\n\n'
                    'def test_farewell():\n'
                    '    assert farewell("world") == "Goodbye, world"\n'
                )
            elif step['type'] == 'impl':
                # Implement
                (cwd / 'lib.py').write_text(
                    'def greet(name: str) -> str:\n'
                    '    return f"Hello, {name}"\n\n'
                    'def farewell(name: str) -> str:\n'
                    '    return f"Goodbye, {name}"\n'
                )

            # Git commit
            await _run(['git', 'add', '-A'], cwd=cwd)
            await _run(['git', 'commit', '-m', f'Complete {step_id}'], cwd=cwd)
            _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=cwd)

            artifacts.update_step_status(step_id, 'done', commit=sha)
            completed_ids.append(step_id)

        return AgentResult(
            success=True, output=f'Completed {completed_ids}', cost_usd=1.00
        )

    async def _debugger(self, cwd: Path) -> AgentResult:
        """Fix whatever is broken. For our stub, ensure files are correct."""
        (cwd / 'lib.py').write_text(
            'def greet(name: str) -> str:\n'
            '    return f"Hello, {name}"\n\n'
            'def farewell(name: str) -> str:\n'
            '    return f"Goodbye, {name}"\n'
        )
        await _run(['git', 'add', '-A'], cwd=cwd)
        await _run(['git', 'commit', '-m', 'fix: correct implementation'], cwd=cwd)
        return AgentResult(success=True, output='Fixed', cost_usd=0.30)

    def _reviewer(self, role: str, output_schema: dict | None) -> AgentResult:
        """Return PASS review."""
        review = _make_review(role)
        return AgentResult(
            success=True,
            output=json.dumps(review),
            structured_output=review,
            cost_usd=0.10,
        )

    def _merger(self) -> AgentResult:
        return AgentResult(success=True, output='Merged', cost_usd=0.20)

    def _judge(self, cwd: Path, session_id: str) -> AgentResult:
        """Return the next queued judge verdict.

        If the entry is an Exception, raise it (simulates invoke failure).
        If the entry is 'SUCCESS_FALSE', return success=False.
        If the entry is None, return success=True with structured_output=None.
        Otherwise treat the entry as a verdict dict.

        Post-task-η the workflow reads the completion verdict from the
        verdict-tools artifact (``verdicts/judge.json``) ONLY — the legacy
        ``result.structured_output`` read is gone. So for a real verdict
        dict this stub must mirror the agent-side submit_completion_verdict
        tool call by routing through the actual ``_submit_completion_verdict``
        implementation, writing to the ``.task-meta`` META_ROOT the
        workflow's ``self.artifacts`` reads from (NOT ``TaskArtifacts(cwd)``,
        the legacy ``.task`` root) — same root pattern as the reviewer/
        architect stubs above. ``structured_output`` is retained belt-and-
        braces but is no longer consulted by the workflow. The envelope is
        written directly (``_envelope`` + ``write_verdict``) rather than
        through ``_submit_completion_verdict`` so a deliberately malformed
        verdict dict (missing required keys) is preserved verbatim on disk —
        exercising the workflow's required-keys fall-through
        (test_execute_iterations_continues_on_judge_malformed_output).
        """
        if not self._judge_verdicts:
            # No verdicts queued — default: incomplete
            verdict = {
                'complete': False,
                'reasoning': 'default stub verdict',
                'uncovered_plan_steps': [],
                'substantive_work': False,
            }
        elif len(self._judge_verdicts) > 1:
            verdict = self._judge_verdicts.pop(0)
        else:
            verdict = self._judge_verdicts[0]

        if isinstance(verdict, Exception):
            raise verdict
        if verdict == 'SUCCESS_FALSE':
            return AgentResult(success=False, output='', cost_usd=0.05)
        if verdict is None:
            return AgentResult(success=True, output='', structured_output=None, cost_usd=0.05)

        judge_artifacts = TaskArtifacts(
            cwd, TaskArtifacts.meta_root_for(cwd.parent, cwd.name)
        )
        judge_artifacts.write_verdict(
            'judge', _verdict_envelope('judge', session_id, verdict)
        )
        return AgentResult(
            success=True,
            output=json.dumps(verdict),
            structured_output=verdict,
            cost_usd=0.05,
        )

    def next_verify_result(self) -> VerifyResult:
        """Pop the next verify result, or return the last one forever."""
        if len(self._verify_results) > 1:
            return self._verify_results.pop(0)
        return self._verify_results[0]


def _build_workflow(
    config: OrchestratorConfig,
    git_ops: GitOps,
    assignment: TaskAssignment,
    agent_stub: AgentStub,
) -> tuple[TaskWorkflow, FakeScheduler]:
    """Wire up a TaskWorkflow with all fakes injected."""
    from _serial_merge_worker import MergeWorker

    scheduler = FakeScheduler()
    merge_queue: asyncio.Queue = asyncio.Queue()
    worker = MergeWorker(git_ops, merge_queue)
    # Start merge worker — cleaned up when event loop tears down after test
    asyncio.create_task(worker.run(), name='test-merge-worker')
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,  # type: ignore[arg-type]
        briefing=FakeBriefing(),  # type: ignore[arg-type]
        mcp=FakeMcp(),  # type: ignore[arg-type]
        merge_queue=merge_queue,
    )
    return workflow, scheduler


def _build_workflow_with_escalation(
    config: OrchestratorConfig,
    git_ops: GitOps,
    assignment: TaskAssignment,
    agent_stub: AgentStub,
    tmp_path: Path,
    *,
    spawn_merge_worker: bool = True,
) -> tuple[TaskWorkflow, FakeScheduler, EscalationQueue]:
    """Wire up a TaskWorkflow with an EscalationQueue attached.

    When ``spawn_merge_worker=False``, skips the MergeWorker/asyncio.create_task
    setup — use for tests that exercise _mark_blocked directly and never enqueue
    merge work; omitting create_task avoids the 'Task was destroyed but it is
    pending!' warning in pytest teardown.
    """
    from _serial_merge_worker import MergeWorker

    scheduler = FakeScheduler()
    queue_dir = tmp_path / 'escalation_queue'
    queue = EscalationQueue(queue_dir)
    merge_queue: asyncio.Queue = asyncio.Queue()
    if spawn_merge_worker:
        worker = MergeWorker(git_ops, merge_queue)
        asyncio.create_task(worker.run(), name='test-merge-worker')
    else:
        worker = None
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,  # type: ignore[arg-type]
        briefing=FakeBriefing(),  # type: ignore[arg-type]
        mcp=FakeMcp(),  # type: ignore[arg-type]
        escalation_queue=queue,
        merge_queue=merge_queue,
        merge_worker=worker,
    )
    return workflow, scheduler, queue


# ---------------------------------------------------------------------------
# Protocol-conformance assertion (static-only; never executes at runtime).
# Mirrors the if TYPE_CHECKING / SchedulerFacade conformance block near the
# bottom of test_workflow_e2e.py.
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from orchestrator.scheduler import SchedulerFacade

    _fake_scheduler_conforms: SchedulerFacade = FakeScheduler()
