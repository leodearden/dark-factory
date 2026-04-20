"""Per-task workflow state machine: PLAN → EXECUTE → VERIFY → REVIEW → MERGE → DONE."""

from __future__ import annotations

import asyncio
import enum
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from shared.config_dir import TaskConfigDir
from shared.cost_store import CostStore

from orchestrator.agents.briefing import COMPLETION_JUDGE_SCHEMA
from orchestrator.agents.invoke import AgentResult, invoke_with_cap_retry
from orchestrator.agents.roles import (
    ALL_REVIEWERS,
    ARCHITECT,
    DEBUGGER,
    IMPLEMENTER,
    JUDGE,
    MERGER,
    AgentRole,
)
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment, files_to_modules, normalize_lock
from orchestrator.task_status import TERMINAL_STATUSES
from orchestrator.usage_gate import SessionBudgetExhausted as _SessionBudgetExhausted
from orchestrator.verify import run_scoped_verification

# Orchestrator package directory — used to resolve ``uv run --project`` for
# the plan-tools stdio MCP server.
_ORCH_PROJECT_DIR = Path(__file__).resolve().parents[2]


class _StewardReescalated(Exception):
    """Raised when the steward re-escalates to level-1 (human intervention)."""

    def __init__(self, escalations):
        self.escalations = escalations

if TYPE_CHECKING:
    from orchestrator.usage_gate import UsageGate


# ---------------------------------------------------------------------------
# Structural protocols — allow test doubles without inheriting concrete classes
# ---------------------------------------------------------------------------


class _SchedulerLike(Protocol):
    async def set_task_status(
        self, task_id: str, status: str, /, *, done_provenance: dict | None = ...
    ) -> None: ...
    async def handle_blast_radius_expansion(
        self, task_id: str, current: list[str], needed: list[str], /
    ) -> bool: ...
    def get_cached_status(self, task_id: str, /) -> str | None: ...


class _McpLike(Protocol):
    @property
    def url(self) -> str: ...
    def mcp_config_json(self, escalation_url: str | None = None) -> dict: ...


class _BriefingLike(Protocol):
    async def build_architect_prompt(
        self, task: dict, worktree: Path | None = ..., context: str | None = ...
    ) -> str: ...
    async def build_resume_prompt(
        self,
        task: dict,
        plan: dict,
        escalation_summary: str,
        resolution: str,
        worktree: Path | None = ...,
    ) -> str: ...
    async def build_implementer_prompt(
        self, plan: dict, iteration_log: list, context: str | None = ...,
        rebase_notice: dict | None = ..., task_id: str | None = ...,
    ) -> str: ...
    async def build_amender_prompt(
        self, plan: dict, iteration_log: list[dict],
        suggestions: list[dict], locked_modules: list[str],
        context: str | None = ..., task_id: str | None = ...,
    ) -> str: ...
    async def build_debugger_prompt(
        self, failures: str, plan: dict, context: str | None = ...,
        task_id: str | None = ...,
    ) -> str: ...
    async def build_reviewer_prompt(
        self, reviewer_type: str, diff: str, context: str | None = ...
    ) -> str: ...
    async def build_merger_prompt(
        self, conflicts: str, task_intent: str, context: str | None = ...
    ) -> str: ...
    async def build_revalidation_prompt(
        self, task: dict, existing_plan: dict,
        changed_files: list[str], worktree: Path | None = ...,
        context: str | None = ...,
    ) -> str: ...
    async def build_completion_judge_prompt(
        self, plan: dict, iteration_log: list[dict], diff: str,
        task_id: str | None = ..., context: str | None = ...,
    ) -> str: ...

logger = logging.getLogger(__name__)


class WorkflowState(enum.Enum):
    PLAN = 'plan'
    EXECUTE = 'execute'
    VERIFY = 'verify'
    REVIEW = 'review'
    MERGE = 'merge'
    DONE = 'done'
    BLOCKED = 'blocked'
    ESCALATED = 'escalated'


class WorkflowOutcome(enum.Enum):
    DONE = 'done'
    BLOCKED = 'blocked'
    REQUEUED = 'requeued'
    ESCALATED = 'escalated'


@dataclass
class WorkflowMetrics:
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    agent_invocations: int = 0
    execute_iterations: int = 0
    verify_attempts: int = 0
    review_cycles: int = 0
    amendment_rounds: int = 0
    pre_merge_rebase_attempts: int = 0
    pre_merge_rebase_ok: int = 0
    advance_main_retries: int = 0
    inter_iteration_rebases: int = 0
    total_turns: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_create_tokens: int = 0
    # Completion judge metrics (ζ). judge_cost_usd is a subset of total_cost_usd,
    # not disjoint — existing budget guards/cost reports using total_cost_usd
    # continue to work unchanged.
    judge_invocations: int = 0
    judge_cost_usd: float = 0.0
    judge_early_exits: int = 0


class TaskWorkflow:
    """Per-task state machine."""

    def __init__(
        self,
        assignment: TaskAssignment,
        config: OrchestratorConfig,
        git_ops: GitOps,
        scheduler: _SchedulerLike,
        briefing: _BriefingLike,
        mcp: _McpLike,
        escalation_queue=None,
        escalation_event: asyncio.Event | None = None,
        usage_gate: UsageGate | None = None,
        initial_plan: dict | None = None,
        steward_factory=None,
        merge_queue: asyncio.Queue | None = None,
        merge_worker=None,
        event_store: EventStore | None = None,
        cost_store: CostStore | None = None,
    ):
        self.assignment = assignment
        self.config = config
        self.git_ops = git_ops
        self.scheduler = scheduler
        self.briefing = briefing
        self.mcp = mcp
        self.merge_queue = merge_queue
        # MergeWorker | SpeculativeMergeWorker | None — used by wip/unmerged
        # handlers to register halt ownership. The asyncio.Queue above carries
        # merge requests; this is the worker that owns the halt flag.
        self.merge_worker = merge_worker
        self.event_store = event_store
        self.cost_store = cost_store

        self.state = WorkflowState.PLAN
        self._phase_cost_at_entry: float = 0.0
        self.task = assignment.task
        self.task_id = assignment.task_id
        self.modules = list(assignment.modules)
        self.worktree: Path | None = None
        self._worktree_external = False  # True when worktree was pre-created (eval mode)
        self.artifacts: TaskArtifacts | None = None
        self.plan: dict = {}
        self.initial_plan = initial_plan
        self.metrics = WorkflowMetrics()

        # Per-module configs for scoped verification
        self._module_configs = self._resolve_module_configs()

        # Escalation support
        self.escalation_queue = escalation_queue
        self._escalation_event = escalation_event

        # Usage cap gate
        self.usage_gate = usage_gate

        # Unique session identifier for plan ownership (format: {task_id}-{uuid_hex[:8]})
        self.session_id = f'{self.task_id}-{uuid.uuid4().hex[:8]}'

        self._steward_factory = steward_factory
        self._steward: Any | None = None
        self._config_dir: TaskConfigDir | None = None
        self._old_plan_base: str | None = None  # base commit from prior session (for revalidation diff)

    @property
    def _task_files(self) -> list[str] | None:
        """Return the file list from the current plan, or None if unavailable/empty."""
        files = self.plan.get('files', [])
        return files if files else None

    def _enter_phase(self, new_state: WorkflowState) -> None:
        """Transition to a new workflow phase, emitting events."""
        if self.event_store:
            prev = self.state
            if prev not in (WorkflowState.DONE, WorkflowState.BLOCKED):
                cost_delta = self.metrics.total_cost_usd - self._phase_cost_at_entry
                self.event_store.emit(
                    EventType.phase_exit,
                    task_id=self.task_id, phase=prev.value,
                    cost_usd=cost_delta,
                )
            self.event_store.emit(
                EventType.phase_enter,
                task_id=self.task_id, phase=new_state.value,
            )
        self._phase_cost_at_entry = self.metrics.total_cost_usd
        self.state = new_state

    async def run(self) -> WorkflowOutcome:
        """Execute the full state machine."""
        branch_name = self.task_id
        try:
            # Set task in-progress
            await self.scheduler.set_task_status(self.task_id, 'in-progress')

            # Create worktree (captures base commit for stable diffs)
            # If worktree is already set (e.g. eval mode), skip creation
            if self.worktree is None:
                worktree_info = await self.git_ops.create_worktree(branch_name)
                self.worktree = worktree_info.path
                base_commit = worktree_info.base_commit
            else:
                # Eval mode: worktree was pre-created, skip creation and cleanup
                self._worktree_external = True
                proc = await asyncio.create_subprocess_exec(
                    'git', 'rev-parse', 'HEAD',
                    cwd=str(self.worktree),
                    stdout=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                base_commit = stdout.decode().strip()
            # Per-task config dir for credential isolation
            self._config_dir = TaskConfigDir(self.task_id)

            # Sync per-worktree venvs so imports resolve from worktree source
            if not self._worktree_external:
                await self._sync_worktree_venvs()

            self.artifacts = TaskArtifacts(self.worktree)
            # Capture old base_commit before init() overwrites metadata.json.
            # Used by _plan() to compute the diff for plan revalidation.
            self._old_plan_base = self.artifacts.read_base_commit()
            self.artifacts.init(
                self.task_id,
                self.task.get('title', ''),
                self.task.get('description', ''),
                base_commit=base_commit,
            )

            # ── .task/ contamination guard ────────────────────────────
            # After init() creates the .task/ scratch directory, verify it
            # is NOT tracked in git.  If it is (inherited from a contaminated
            # main), untrack it so agents don't accidentally commit it.
            # This is defense-in-depth: create_worktree() should have already
            # scrubbed, but the guard here catches the eval-mode path and
            # any race conditions.
            rc, tracked, _ = await _run(
                ['git', 'ls-files', '--', '.task/'],
                cwd=self.worktree,
            )
            if rc == 0 and tracked.strip():
                logger.warning(
                    'Task %s: .task/ is tracked in git (inherited contamination) '
                    '— removing from index. Files: %s',
                    self.task_id, tracked.strip()[:200],
                )
                await _run(
                    ['git', 'rm', '-r', '--cached', '--', '.task/'],
                    cwd=self.worktree,
                )
                # Don't commit the removal separately — it'll be part of
                # the first real commit the agent makes.

            # PLAN (skip if initial_plan was provided — eval mode)
            if self.initial_plan:
                plan_tid = self.initial_plan.get('task_id')
                if plan_tid and plan_tid != self.task_id:
                    logger.error(
                        f'Task {self.task_id}: initial_plan has mismatched '
                        f'task_id {plan_tid} — discarding, will re-plan'
                    )
                    self.initial_plan = None
            if self.initial_plan:
                self.artifacts.write_plan(self.initial_plan)
                self.artifacts.stamp_plan_provenance(self.session_id)
                self.artifacts.lock_plan(self.session_id)
                self.plan = self.artifacts.read_plan()
                logger.info(
                    f'Task {self.task_id}: using provided plan '
                    f'({len(self.plan.get("steps", []))} steps)'
                )
            else:
                self._enter_phase(WorkflowState.PLAN)
                plan_outcome = await self._plan()
                if plan_outcome == WorkflowOutcome.REQUEUED:
                    return WorkflowOutcome.REQUEUED
                if plan_outcome == WorkflowOutcome.BLOCKED:
                    if self.event_store:
                        self.event_store.emit(
                            EventType.waste_detected,
                            task_id=self.task_id, phase='plan',
                            data={'waste_type': 'replan_after_failure'},
                        )
                    return plan_outcome  # _plan() already called _mark_blocked

            # ── Ghost-loop early exit (before EXECUTE) ─────────────
            # If the worktree HEAD is already reachable from main, the
            # task's code was merged in a prior run that never reached
            # DONE status (e.g. post-merge memory write failed).  Skip
            # the entire execute/review/merge cycle to avoid the
            # implementer making redundant commits that defeat the
            # merge-phase ancestor check.
            #
            # NOTE: wt_head == current_main is ALSO a legitimate ghost-
            # loop case — create_worktree rebases reused worktrees onto
            # main, fast-forwarding a post-merge branch to match main
            # exactly.  The has_work check below distinguishes stale
            # branch points (no implementation) from true ghost loops.
            wt_head = await self._get_head_commit()
            current_main = await self.git_ops.get_main_sha()
            already_on_main = (
                wt_head == current_main
                or await self.git_ops.is_ancestor(wt_head, current_main)
            )
            if already_on_main and not self._worktree_external:
                # Guard: a stale branch point (requeued task that was planned
                # but never implemented, or a freshly-created worktree) also
                # satisfies the ancestor check.  Only skip if there's
                # evidence of prior implementation work.
                has_work = (
                    self._has_prior_implementation()
                    or await self.git_ops.has_uncommitted_work(self.worktree)
                )
                if has_work:
                    logger.info(
                        f'Task {self.task_id}: worktree HEAD {wt_head[:8]} '
                        f'already on main — skipping to DONE (prior merge survived)'
                    )
                else:
                    logger.info(
                        f'Task {self.task_id}: worktree HEAD {wt_head[:8]} '
                        f'is ancestor of main but no prior implementation '
                        f'— stale branch point, proceeding normally'
                    )
                    already_on_main = False
            if not already_on_main or self._worktree_external:
                # Normal path: EXECUTE + VERIFY + REVIEW loop (with escalation retry)
                while True:
                    outcome = await self._execute_verify_review_loop()
                    if outcome == WorkflowOutcome.ESCALATED:
                        self._enter_phase(WorkflowState.ESCALATED)
                        await self._ensure_steward_started()
                        logger.info(f'Task {self.task_id}: waiting for escalation resolution')
                        try:
                            resolution = await self._wait_for_resolution()
                        except _StewardReescalated:
                            return await self._mark_blocked(
                                'Steward re-escalated to human',
                                skip_escalation=True,
                            )
                        # If branch is already on main (e.g. steward merged
                        # during resolution), skip re-implementation — proceed
                        # to MERGE which will detect already_merged.
                        _, wt_head_raw, _ = await _run(
                            ['git', 'rev-parse', 'HEAD'], cwd=self.worktree,
                        )
                        esc_main_sha = await self.git_ops.get_main_sha()
                        if await self.git_ops.is_ancestor(
                            wt_head_raw.strip(), esc_main_sha,
                        ):
                            logger.info(
                                'Task %s: branch already on main after '
                                'escalation resolution — skipping '
                                're-implementation', self.task_id,
                            )
                            break

                        # Resume with resolution context
                        logger.info(f'Task {self.task_id}: resuming after escalation resolution')
                        resume_prompt = await self.briefing.build_resume_prompt(
                            self.task, self.plan,
                            '\n'.join(e.summary for e in self._check_escalations()),
                            resolution, self.worktree,
                        )
                        await self._invoke(IMPLEMENTER, resume_prompt, self.worktree)
                        continue
                    if outcome != WorkflowOutcome.DONE:
                        return outcome
                    break

                # MERGE (skip for eval mode — no merge into main)
                if not self._worktree_external:
                    self._enter_phase(WorkflowState.MERGE)

                    # Ghost-loop early exit: if branch is already on main,
                    # skip the entire merge phase (prevents infinite retry
                    # when code was merged by an external actor).
                    _, branch_head, _ = await _run(
                        ['git', 'rev-parse', 'HEAD'], cwd=self.worktree,
                    )
                    main_sha = await self.git_ops.get_main_sha()
                    already_merged = await self.git_ops.is_ancestor(
                        branch_head.strip(), main_sha,
                    )

                    # Defense-in-depth: same stale-branch-point guard as
                    # the pre-EXECUTE check.  Should rarely fire since
                    # we just ran execute, but guards against edge cases.
                    if already_merged and not self._has_prior_implementation():
                        logger.warning(
                            f'Task {self.task_id}: branch appears merged at '
                            f'merge phase but has no implementation entries '
                            f'— proceeding with merge'
                        )
                        already_merged = False

                    if not already_merged:
                        for _merge_attempt in range(self.config.max_merge_retries):
                            # Phase 1: pre-merge rebase (no lock, no queue slot)
                            # Rebase the task branch onto current main and re-verify
                            # so the queued merge phase is fast/trivial.
                            pre_rebased = False
                            for _attempt in range(self.config.max_pre_merge_retries):
                                main_before = await self.git_ops.get_main_sha()
                                if not await self.git_ops.rebase_onto_main(self.worktree):
                                    break  # true conflict — queue will detect it
                                verify = await run_scoped_verification(
                                    self.worktree, self.config, self._module_configs,
                                    task_files=self._task_files,
                                )
                                if not verify.passed:
                                    if verify.timed_out:
                                        logger.warning(
                                            f'Task {self.task_id}: post-rebase verification '
                                            f'timed out; merge queue will retry'
                                        )
                                    else:
                                        logger.warning(
                                            f'Task {self.task_id}: post-rebase verification '
                                            f'failed: {verify.summary}'
                                        )
                                        if self.event_store:
                                            self.event_store.emit(
                                                EventType.waste_detected,
                                                task_id=self.task_id, phase='merge',
                                                data={
                                                    'waste_type': 'post_rebase_verify_fail',
                                                    'summary': verify.summary[:200],
                                                },
                                            )
                                    break
                                main_after = await self.git_ops.get_main_sha()
                                if main_before == main_after:
                                    pre_rebased = True
                                    self.metrics.pre_merge_rebase_ok += 1
                                    break
                                self.metrics.pre_merge_rebase_attempts += 1
                                logger.info(
                                    f'Task {self.task_id}: main moved during pre-merge '
                                    f'rebase, retrying'
                                )

                            # Phase 2: submit to merge queue (replaces _merge_lock)
                            merge_outcome = await self._submit_to_merge_queue(
                                branch_name, pre_rebased=pre_rebased,
                                merge_phase=True,
                            )
                            if merge_outcome == WorkflowOutcome.DONE:
                                break
                            if merge_outcome != WorkflowOutcome.REQUEUED:
                                # BLOCKED — steward gave up, terminal
                                return merge_outcome

                            # Steward resolved — check if branch landed on main
                            _, bh, _ = await _run(
                                ['git', 'rev-parse', 'HEAD'], cwd=self.worktree,
                            )
                            main_sha = await self.git_ops.get_main_sha()
                            if await self.git_ops.is_ancestor(bh.strip(), main_sha):
                                logger.info(
                                    'Task %s: branch on main after steward '
                                    'resolution', self.task_id,
                                )
                                break
                            # Retry merge
                            logger.info(
                                'Task %s: retrying merge (attempt %d/%d)',
                                self.task_id, _merge_attempt + 1,
                                self.config.max_merge_retries,
                            )
                        else:
                            return await self._mark_blocked(
                                'Merge retries exhausted after steward resolutions'
                            )
                    else:
                        logger.info(
                            f'Task {self.task_id}: branch already on main '
                            f'— skipping merge'
                        )

            # SUCCESS — write completion knowledge (best-effort after merge)
            try:
                await self._write_completion_to_memory()
            except Exception as e:
                logger.warning(
                    f'Task {self.task_id}: completion memory write failed '
                    f'(non-fatal): {e}'
                )
            # Wait for steward to finish any pending work (suggestion triage, etc.)
            await self._ensure_steward_started()
            await self._await_steward_completion()
            self._enter_phase(WorkflowState.DONE)
            await self.scheduler.set_task_status(self.task_id, 'done')
            logger.info(
                f'Task {self.task_id} DONE — '
                f'cost=${self.metrics.total_cost_usd:.2f} '
                f'invocations={self.metrics.agent_invocations}'
            )
            return WorkflowOutcome.DONE

        except _SessionBudgetExhausted:
            logger.warning(f'Task {self.task_id}: session budget exhausted')
            return await self._mark_blocked('Session budget exhausted')

        except Exception as e:
            logger.exception(f'Task {self.task_id} workflow error: {e}')
            return await self._mark_blocked(f'Workflow error: {e}')

        finally:
            # Stop steward if running
            if self._steward:
                await self._steward.stop()
            # Cleanup worktree (only if done — keep for debugging if blocked)
            # Skip cleanup for externally-managed worktrees (eval mode)
            if self.state == WorkflowState.DONE and self.worktree and not self._worktree_external:
                await self.git_ops.cleanup_worktree(self.worktree, branch_name)
            # Cleanup per-task config dir
            if self._config_dir:
                self._config_dir.cleanup()

    def _resolve_module_configs(self) -> list[ModuleConfig]:
        """Collect ModuleConfigs for this task's modules.

        Groups modules by subproject prefix and returns one ModuleConfig per
        subproject that has an ``orchestrator.yaml``.  Warns for subprojects
        without configs.  Returns an empty list when no modules are assigned
        (triggers global fallback in ``run_scoped_verification``).
        """
        if not self.modules:
            return []
        seen: dict[str, ModuleConfig] = {}
        missing: set[str] = set()
        for m in self.modules:
            mc = self.config.for_module(m)
            if mc:
                seen[mc.prefix] = mc
            else:
                prefix = m.strip('/').split('/')[0]
                missing.add(prefix)
        if missing:
            logger.warning(
                'Task %s: subprojects without orchestrator.yaml: %s — '
                'these will use global verification config',
                self.task_id, missing,
            )
        return list(seen.values())

    async def _sync_worktree_venvs(self) -> None:
        """Run ``uv sync`` for task subprojects in the worktree.

        Creates per-worktree venvs so Python imports resolve from the
        worktree's source code rather than the main tree's editable installs.
        Local ``[tool.uv.sources]`` dependencies (e.g. ``shared``) are
        pulled in automatically via relative editable paths.
        """
        assert self.worktree is not None

        # Derive unique subproject prefixes from task modules
        prefixes: set[str] = set()
        for m in self.modules:
            prefix = m.strip('/').split('/')[0]
            if (self.worktree / prefix / 'pyproject.toml').exists():
                prefixes.add(prefix)

        if not prefixes:
            return

        worktree = self.worktree  # bind for closure (narrowed to Path)

        # Sync subprojects in parallel
        async def _sync(prefix: str) -> None:
            project_dir = str(worktree / prefix)
            proc = await asyncio.create_subprocess_exec(
                'uv', 'sync', '--project', project_dir,
                cwd=str(self.worktree),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                logger.warning(
                    'Task %s: uv sync failed for %s: %s',
                    self.task_id, prefix, stdout.decode()[:500],
                )
            else:
                logger.info('Task %s: synced venv for %s', self.task_id, prefix)

        await asyncio.gather(*(_sync(p) for p in sorted(prefixes)))

    async def _plan(self) -> WorkflowOutcome:
        """Invoke the architect to produce a plan."""
        assert self.worktree is not None and self.artifacts is not None

        # Defense-in-depth: if plan.lock already exists, check if it's stale.
        # If stale (self-lock from crashed session or age exceeds threshold),
        # clear it and proceed.  If held by an active session, requeue to
        # avoid duplicate execution.
        if self.artifacts.is_plan_locked() and self.artifacts.read_plan():
            cleared = self.artifacts.clear_stale_plan_lock(self.task_id)
            if not cleared:
                lock_data = self.artifacts.read_plan_lock()
                lock_owner = lock_data.get('session_id', 'unknown') if lock_data else 'unknown'
                logger.info(
                    f'Task {self.task_id}: plan.lock is held by session {lock_owner!r}, '
                    f'skipping architect — requeuing to avoid duplicate execution'
                )
                await self.scheduler.set_task_status(self.task_id, 'pending')
                return WorkflowOutcome.REQUEUED
            # Lock was stale and cleared.  If the plan has provenance
            # (complete plan from a prior session, e.g. blast-radius requeue),
            # keep it for revalidation.  If not (crashed mid-planning), delete.
            existing = self.artifacts.read_plan()
            if existing and not existing.get('_session_id'):
                plan_path = self.artifacts.root / 'plan.json'
                if plan_path.exists():
                    plan_path.unlink()

        # ── Revalidation vs. fresh planning ──────────────────────────
        # If a provenance-stamped plan already exists (blast-radius requeue),
        # build a revalidation prompt with the diff of what changed on main
        # so the architect can confirm, update, or recreate the plan.
        revalidation = False
        revalidation_changed_files: list[str] = []
        existing_plan = self.artifacts.read_plan()
        if (
            existing_plan
            and existing_plan.get('steps')
            and existing_plan.get('_session_id')
            and self._old_plan_base
        ):
            current_main = await self.git_ops.get_main_sha()
            revalidation_changed_files = await self.git_ops.get_changed_files(
                self._old_plan_base, current_main,
            )
            plan_file_set = set(existing_plan.get('files', []))
            overlap = [f for f in revalidation_changed_files if f in plan_file_set]
            logger.info(
                'Task %s: revalidating existing plan '
                '(%d changed files, %d overlap with plan)',
                self.task_id, len(revalidation_changed_files), len(overlap),
            )
            prompt = await self.briefing.build_revalidation_prompt(
                self.task, existing_plan, revalidation_changed_files,
                worktree=self.worktree,
            )
            revalidation = True
        else:
            prompt = await self.briefing.build_architect_prompt(
                self.task, worktree=self.worktree,
            )

        result: AgentResult | None = None
        for attempt in range(2):
            result = await self._invoke(ARCHITECT, prompt, self.worktree)

            if not result.success:
                logger.error(f'Task {self.task_id}: architect failed: {result.output[:200]}')
                return await self._mark_blocked(
                    'Planning failed: architect invocation failed',
                    detail=f'Architect output:\n{result.output[:2000]}',
                )

            # Detect anomalous premature exit: succeeded but suspiciously
            # few turns and low cost — likely a transient CLI issue.
            self.plan = self.artifacts.read_plan()
            if (
                attempt == 0
                and result.turns <= 2
                and result.cost_usd < 0.20
                and not self.plan
            ):
                logger.warning(
                    f'Task {self.task_id}: architect completed anomalously '
                    f'(turns={result.turns}, cost=${result.cost_usd:.2f}, '
                    f'duration={result.duration_ms}ms, output_len={len(result.output)}) '
                    f'— retrying once'
                )
                continue

            break

        assert result is not None  # range(2) always executes at least once
        if not self.plan:
            logger.error(f'Task {self.task_id}: architect produced no plan.json')
            assert result is not None  # range(2) is always non-empty; loop always assigns result
            return await self._mark_blocked(
                'Planning failed: no plan.json produced',
                detail=(
                    f'Architect succeeded but did not write .task/plan.json\n'
                    f'turns={result.turns}, cost=${result.cost_usd:.2f}, '
                    f'duration={result.duration_ms}ms\n'
                    f'Architect output:\n{result.output[:2000]}'
                ),
            )

        if not self.plan.get('steps'):
            # Normalization (in read_plan) didn't help — try a one-shot
            # repair prompt before blocking.
            logger.warning(
                'Task %s: plan has no "steps" after normalization — '
                'attempting repair prompt',
                self.task_id,
            )
            repaired = await self._repair_plan_schema()
            if repaired:
                self.plan = self.artifacts.read_plan()

        if not self.plan.get('steps'):
            plan_dump = json.dumps(self.plan, indent=2)
            logger.error(
                f'Task {self.task_id}: architect wrote plan.json but missing/empty '
                f'"steps" — full plan content: {plan_dump}'
            )
            return await self._mark_blocked(
                'Planning failed: plan missing "steps"',
                detail=f'Plan content:\n{plan_dump[:4000]}',
            )

        if outcome := await self._validate_prerequisites_or_block('initial plan'):
            return outcome

        # Stamp provenance and acquire lock
        self.artifacts.stamp_plan_provenance(self.session_id)
        self.artifacts.lock_plan(self.session_id)
        self.plan = self.artifacts.read_plan()

        if revalidation and self.event_store:
            plan_file_set = set(self.plan.get('files', []))
            self.event_store.emit(
                EventType.plan_revalidated,
                task_id=self.task_id,
                data={
                    'changed_files_count': len(revalidation_changed_files),
                    'overlap_count': len(
                        [f for f in revalidation_changed_files
                         if f in plan_file_set]
                    ),
                },
            )

        plan_files = self.plan.get('files', [])
        if not plan_files:
            return await self._mark_blocked(
                'Planning failed: plan missing "files"',
                detail=(
                    'Architect wrote plan.json without a non-empty "files" array. '
                    'Files are required to derive module locks.'
                ),
            )
        plan_modules = files_to_modules(plan_files, self.config.lock_depth)
        logger.info(
            f'Task {self.task_id}: derived {len(plan_modules)} modules '
            f'from {len(plan_files)} files: {plan_modules}'
        )

        if set(plan_modules) != set(self.modules):
            expanded = await self.scheduler.handle_blast_radius_expansion(
                self.task_id, self.modules, plan_modules
            )
            if not expanded:
                return WorkflowOutcome.REQUEUED
            self.modules = plan_modules
            self._module_configs = self._resolve_module_configs()

        # Write plan decisions to memory
        await self._write_decisions_to_memory()

        logger.info(
            f'Task {self.task_id}: plan created with '
            f'{len(self.plan.get("prerequisites", []))} prerequisites, '
            f'{len(self.plan.get("steps", []))} steps'
        )
        return WorkflowOutcome.DONE

    async def _validate_prerequisites_or_block(
        self, context: str
    ) -> WorkflowOutcome | None:
        """Validate prerequisites format; block if invalid.

        Encapsulates the try/validate/except/mark-blocked pattern shared by the
        initial-plan checkpoint and the replan checkpoint, parameterised by a
        *context* string that appears in log and escalation messages.

        Args:
            context: Short description for log/error messages, e.g.
                     ``'initial plan'`` or ``'replan'``.

        Returns:
            A :class:`WorkflowOutcome` (BLOCKED) if validation fails,
            ``None`` if prerequisites are valid.
        """
        assert self.artifacts is not None
        try:
            self.artifacts.validate_plan_prerequisites()
        except ValueError as exc:
            plan_dump = json.dumps(self.plan, indent=2)
            logger.error(
                f'Task {self.task_id}: {context} produced plan.json with invalid '
                f'prerequisites — {exc}'
            )
            return await self._mark_blocked(
                f'Planning failed ({context}): invalid prerequisites format — {exc}',
                detail=f'Plan content:\n{plan_dump[:4000]}',
            )
        return None

    async def _repair_plan_schema(self) -> bool:
        """One-shot attempt to fix a plan that is missing ``steps``.

        Sends the broken plan back to the architect with a focused repair
        prompt.  Returns True if the repaired plan now has a ``steps`` array.
        """
        assert self.artifacts is not None  # caller guarantees
        assert self.worktree is not None

        broken_plan = self.artifacts.read_plan()
        if not broken_plan:
            return False

        plan_path = str(self.worktree / '.task' / 'plan.json')
        plan_dump = json.dumps(broken_plan, indent=2)[:6000]

        repair_prompt = (
            'The architect produced a plan that is structurally invalid — '
            'it is missing the required top-level "steps" array.\n\n'
            f'Here is the broken plan content:\n\n```json\n{plan_dump}\n```\n\n'
            'The required schema is:\n'
            '```json\n'
            '{\n'
            '  "task_id": "<task id>",\n'
            '  "title": "<task title>",\n'
            '  "files": ["path/to/file1.py"],\n'
            '  "modules": ["<module1>"],\n'
            '  "analysis": "<analysis>",\n'
            '  "prerequisites": [\n'
            '    {"id": "pre-1", "description": "...", "status": "pending", "commit": null}\n'
            '  ],\n'
            '  "steps": [\n'
            '    {"id": "step-1", "type": "test", "description": "...", "status": "pending", "commit": null},\n'
            '    {"id": "step-2", "type": "impl", "description": "...", "status": "pending", "commit": null}\n'
            '  ],\n'
            '  "design_decisions": [\n'
            '    {"decision": "...", "rationale": "..."}\n'
            '  ]\n'
            '}\n'
            '```\n\n'
            'Your job: restructure the existing plan content into the required '
            'schema.  Do NOT explore the codebase or redesign the plan.  Simply '
            'reorganize the existing keys and values into the correct shape and '
            f'write the result to `{plan_path}` using the Write tool.'
        )

        try:
            await self._invoke(ARCHITECT, repair_prompt, self.worktree)
        except Exception as e:
            logger.warning(
                'Task %s: repair prompt invocation failed: %s',
                self.task_id, e,
            )
            return False

        repaired_plan = self.artifacts.read_plan()
        if repaired_plan.get('steps'):
            logger.info(
                'Task %s: repair prompt succeeded — plan now has %d steps',
                self.task_id, len(repaired_plan['steps']),
            )
            return True

        logger.warning(
            'Task %s: repair prompt did not produce a valid "steps" array',
            self.task_id,
        )
        return False

    async def _execute_verify_review_loop(self) -> WorkflowOutcome:
        """Execute → Verify → Review loop with retry limits."""
        # Clear stale merge-failure review from prior runs — prevents
        # the review phase from re-surfacing resolved merge issues.
        if self.artifacts:
            stale_merge = self.artifacts.root / 'reviews' / 'merge.json'
            if stale_merge.exists():
                logger.info('Task %s: removing stale merge.json review', self.task_id)
                stale_merge.unlink()

        review_cycle = 0
        amendment_round = 0

        while True:
            # EXECUTE
            self._enter_phase(WorkflowState.EXECUTE)
            exec_outcome = await self._execute_iterations()
            if exec_outcome == WorkflowOutcome.ESCALATED:
                return WorkflowOutcome.ESCALATED
            if exec_outcome == WorkflowOutcome.BLOCKED:
                return await self._mark_blocked('Execution iterations exhausted')

            # VERIFY + DEBUGFIX loop
            self._enter_phase(WorkflowState.VERIFY)
            verify_outcome = await self._verify_debugfix_loop()
            if verify_outcome == WorkflowOutcome.ESCALATED:
                return WorkflowOutcome.ESCALATED
            if verify_outcome == WorkflowOutcome.BLOCKED:
                return await self._mark_blocked('Verification attempts exhausted')

            # REVIEW
            self._enter_phase(WorkflowState.REVIEW)
            reviews = await self._review()
            if reviews.reviewer_errors:
                names = ', '.join(reviews.reviewer_errors)
                return await self._mark_blocked(
                    f'{len(reviews.reviewer_errors)} reviewer(s) failed with '
                    f'infrastructure errors after retries: {names}'
                )
            if not reviews.has_blocking_issues:
                # L2b: try an amendment pass before escalating suggestions.
                # In-scope suggestions (module-lock members) are applied by
                # the implementer directly — no architect, no new tasks.
                # Cap is config.max_amendment_rounds (default 1).
                in_scope = self._suggestions_in_scope(reviews.suggestions)
                if (
                    in_scope
                    and amendment_round < self.config.max_amendment_rounds
                ):
                    amendment_round += 1
                    logger.info(
                        'Task %s: amendment round %d, %d in-scope '
                        'suggestion(s) (of %d total)',
                        self.task_id, amendment_round,
                        len(in_scope), len(reviews.suggestions),
                    )
                    # Archive pre-amendment reviews so post-mortem can compare
                    if self.artifacts:
                        import shutil
                        reviews_dir = self.artifacts.root / 'reviews'
                        archive_dir = (
                            self.artifacts.root
                            / f'reviews-amend-{amendment_round}'
                        )
                        if reviews_dir.exists() and not archive_dir.exists():
                            shutil.copytree(reviews_dir, archive_dir)
                            logger.info(
                                'Task %s: archived reviews to %s',
                                self.task_id, archive_dir.name,
                            )
                    await self._amend(in_scope, amendment_round)
                    self.metrics.amendment_rounds += 1
                    continue  # re-loop: EXECUTE → VERIFY → REVIEW

                # Cap exhausted or nothing in-scope — existing DONE path
                if self.escalation_queue and reviews.suggestions:
                    self._escalate_suggestions(reviews)
                else:
                    await self._write_suggestions_to_memory(reviews)
                return WorkflowOutcome.DONE

            review_cycle += 1

            # Archive reviews from this cycle before re-plan overwrites them
            if self.artifacts:
                reviews_dir = self.artifacts.root / 'reviews'
                archive_dir = self.artifacts.root / f'reviews-cycle-{review_cycle}'
                if reviews_dir.exists() and not archive_dir.exists():
                    import shutil
                    shutil.copytree(reviews_dir, archive_dir)
                    logger.info('Task %s: archived reviews to %s', self.task_id, archive_dir.name)

            if review_cycle >= self.config.max_review_cycles:
                self._escalate_review_issues(reviews)
                return WorkflowOutcome.ESCALATED

            # Re-plan based on review feedback
            logger.info(
                f'Task {self.task_id}: review cycle {review_cycle}, '
                f'{len(reviews.blocking_issues)} blocking issues'
            )
            await self._replan(reviews)
            # Re-stamp provenance — architect may have overwritten plan.json
            assert self.artifacts is not None
            self.plan = self.artifacts.read_plan()
            if not self.plan or not self.plan.get('steps'):
                plan_dump = json.dumps(self.plan, indent=2) if self.plan else 'None'
                logger.error(
                    f'Task {self.task_id}: replan produced plan.json with '
                    f'missing/empty "steps" — full plan content: {plan_dump}'
                )
                return await self._mark_blocked(
                    'Architect replan produced no valid steps',
                    detail=f'Replan content:\n{plan_dump[:4000]}',
                )
            if outcome := await self._validate_prerequisites_or_block('replan'):
                return outcome
            self.artifacts.stamp_plan_provenance(self.session_id)
            self.metrics.review_cycles += 1

    async def _execute_iterations(self) -> WorkflowOutcome:
        """Run implementer iterations until plan is complete."""
        assert self.worktree is not None and self.artifacts is not None
        while self.artifacts.get_pending_steps():
            if self.metrics.execute_iterations >= self.config.max_execute_iterations:
                return WorkflowOutcome.BLOCKED

            # Inter-iteration rebase: keep the task branch close to main
            # so the eventual merge is less likely to conflict.  Skip on
            # the first iteration (nothing has changed yet).
            rebase_notice = None
            if (
                self.metrics.execute_iterations > 0
                and self.config.inter_iteration_rebase
            ):
                rebase_notice = await self._inter_iteration_rebase()

            # Validate plan ownership before each implementer invocation
            if not self.artifacts.validate_plan_owner(self.session_id):
                logger.error(
                    f'Task {self.task_id}: plan.json ownership mismatch — '
                    f'expected session {self.session_id}, plan has different _session_id'
                )
                self._escalate_plan_overwrite()
                return WorkflowOutcome.BLOCKED

            self.plan = self.artifacts.read_plan()
            iteration_log, corrupted = self.artifacts.read_iteration_log()
            if corrupted:
                self._escalate_corruption(corrupted)

            # Snapshot completed steps before invocation
            completed_before = {
                s['id']
                for col in ('prerequisites', 'steps')
                for s in self.plan.get(col, [])
                if isinstance(s, dict) and s.get('status') == 'done'
            }

            prompt = await self.briefing.build_implementer_prompt(
                self.plan, iteration_log, rebase_notice=rebase_notice,
                task_id=self.task_id,
            )
            result = await self._invoke(IMPLEMENTER, prompt, self.worktree)

            self.metrics.execute_iterations += 1

            # Check for escalations
            blocking = [e for e in self._check_escalations() if e.severity == 'blocking']
            if blocking:
                return WorkflowOutcome.ESCALATED

            # Re-read plan to see progress
            self.plan = self.artifacts.read_plan()

            # Compute newly completed steps and write iteration log
            completed_after = {
                s['id']
                for col in ('prerequisites', 'steps')
                for s in self.plan.get(col, [])
                if isinstance(s, dict) and s.get('status') == 'done'
            }
            newly_completed = sorted(completed_after - completed_before)
            head_commit = await self._get_head_commit()

            if newly_completed:
                step_descs = [
                    s.get('description', s['id'])
                    for col in ('prerequisites', 'steps')
                    for s in self.plan.get(col, [])
                    if isinstance(s, dict) and s.get('id') in newly_completed
                ]
                summary = '; '.join(step_descs)
            else:
                summary = 'No new steps completed'

            self.artifacts.append_iteration_log({
                'iteration': self.metrics.execute_iterations,
                'agent': 'implementer',
                'steps_attempted': newly_completed,
                'steps_completed': newly_completed,
                'commit': head_commit,
                'summary': summary,
                'source': 'orchestrator',
            })

            # Defense-in-depth: re-stamp _session_id after each implementer
            # iteration.  The plan-tools MCP server preserves it, but if the
            # model also edited plan.json directly (dropping _session_id),
            # this recovers gracefully instead of blocking.
            if not self.artifacts.validate_plan_owner(self.session_id):
                logger.warning(
                    'Task %s: _session_id mismatch after implementer — re-stamping',
                    self.task_id,
                )
            self.artifacts.stamp_plan_provenance(self.session_id)

            if not result.success:
                logger.warning(
                    f'Task {self.task_id}: implementer iteration '
                    f'{self.metrics.execute_iterations} failed'
                )

            # --- Judge: decide whether to exit early (ζ) ---
            # Opt-in via config.judge_after_each_iteration (default False).
            # Eval mode flips it on per-task. Failures fall through silently
            # to the next iteration — current behavior is preserved as worst case.
            if self.config.judge_after_each_iteration:
                judge_verdict = await self._run_completion_judge(iteration_log)
                if judge_verdict is not None and judge_verdict.get('complete') is True:
                    # Safety: reject complete=True if substantive_work=False.
                    # An empty or trivial diff cannot be a completed task.
                    if not judge_verdict.get('substantive_work', False):
                        logger.warning(
                            f'Task {self.task_id}: judge returned complete=True '
                            f'with substantive_work=False — ignoring verdict'
                        )
                    else:
                        self.metrics.judge_early_exits += 1
                        logger.info(
                            f'Task {self.task_id}: judge signaled completion at '
                            f'iteration {self.metrics.execute_iterations} — '
                            f'reasoning: {judge_verdict.get("reasoning", "")[:200]}'
                        )
                        self.artifacts.append_iteration_log({
                            'iteration': self.metrics.execute_iterations,
                            'agent': 'judge',
                            'event': 'early_exit',
                            'complete': True,
                            'substantive_work': True,
                            'uncovered_plan_steps': judge_verdict.get('uncovered_plan_steps', []),
                            'summary': judge_verdict.get('reasoning', '')[:500],
                            'source': 'orchestrator',
                        })
                        return WorkflowOutcome.DONE

        return WorkflowOutcome.DONE

    async def _run_completion_judge(
        self, iteration_log: list[dict]
    ) -> dict | None:
        """Invoke the completion judge. Returns parsed verdict dict or None on failure.

        Any failure mode (exception, success=False, malformed output) returns
        None so the caller continues the iteration loop — current behavior is
        preserved as the worst case.
        """
        assert self.worktree is not None and self.artifacts is not None

        base_commit = self.artifacts.read_base_commit()
        if base_commit:
            diff = await self.git_ops.get_diff_from_base(self.worktree, base_commit)
        else:
            diff = await self.git_ops.get_diff_from_main(self.worktree)

        if not diff or not diff.strip():
            logger.info(
                f'Task {self.task_id}: empty diff — skipping judge invocation'
            )
            return None

        prompt = await self.briefing.build_completion_judge_prompt(
            plan=self.plan,
            iteration_log=iteration_log,
            diff=diff,
            task_id=self.task_id,
        )

        pre_cost = self.metrics.total_cost_usd
        try:
            result = await self._invoke(
                JUDGE, prompt, self.worktree,
                output_schema=COMPLETION_JUDGE_SCHEMA,
            )
        except Exception as exc:
            logger.warning(
                f'Task {self.task_id}: judge invocation raised '
                f'{type(exc).__name__}: {exc} — continuing iteration loop'
            )
            return None

        # judge_cost_usd is a subset of total_cost_usd (already incremented
        # inside _invoke), tracked separately for reporting.
        self.metrics.judge_invocations += 1
        self.metrics.judge_cost_usd += (self.metrics.total_cost_usd - pre_cost)

        if not result.success:
            logger.warning(
                f'Task {self.task_id}: judge invocation returned success=False — '
                f'continuing iteration loop'
            )
            return None

        verdict = result.structured_output
        if not isinstance(verdict, dict):
            logger.warning(
                f'Task {self.task_id}: judge returned non-dict structured_output — '
                f'continuing iteration loop'
            )
            return None

        required = {'complete', 'reasoning', 'uncovered_plan_steps', 'substantive_work'}
        if not required <= verdict.keys():
            logger.warning(
                f'Task {self.task_id}: judge verdict missing keys '
                f'{required - verdict.keys()} — continuing iteration loop'
            )
            return None

        return verdict

    async def _inter_iteration_rebase(self) -> dict | None:
        """Check if main advanced past our base; if so, rebase.

        Returns a dict ``{old_base, new_base, changed_files}`` when a
        rebase was performed, or ``None`` if no rebase was needed or the
        rebase failed (failure is non-blocking — the merge phase will
        handle conflicts).
        """
        assert self.worktree is not None and self.artifacts is not None

        old_base = self.artifacts.read_base_commit()
        if not old_base:
            return None

        current_main = await self.git_ops.get_main_sha()
        if current_main == old_base:
            return None

        if not await self.git_ops.is_ancestor(old_base, current_main):
            return None  # unexpected topology — skip

        changed_files = await self.git_ops.get_changed_files(
            old_base, current_main,
        )

        # Commit any uncommitted work before rebasing
        await self.git_ops.commit(
            self.worktree, 'chore: save WIP before inter-iteration rebase',
        )

        if not await self.git_ops.rebase_onto_main(self.worktree):
            logger.warning(
                f'Task {self.task_id}: inter-iteration rebase failed, '
                f'continuing on old base'
            )
            return None

        self.artifacts.update_base_commit(current_main)
        self.metrics.inter_iteration_rebases += 1

        self.artifacts.append_iteration_log({
            'iteration': self.metrics.execute_iterations,
            'agent': 'orchestrator',
            'event': 'rebase',
            'old_base': old_base,
            'new_base': current_main,
            'files_changed_on_main': changed_files[:50],
            'source': 'orchestrator',
            'summary': (
                f'Rebased onto main ({old_base[:8]} -> {current_main[:8]}), '
                f'{len(changed_files)} files changed'
            ),
        })

        logger.info(
            f'Task {self.task_id}: rebased onto main '
            f'({old_base[:8]} -> {current_main[:8]}), '
            f'{len(changed_files)} files changed on main'
        )

        return {
            'old_base': old_base,
            'new_base': current_main,
            'changed_files': changed_files,
        }

    async def _verify_debugfix_loop(self) -> WorkflowOutcome:
        """Run verification, invoke debugger on failures."""
        assert self.worktree is not None and self.artifacts is not None
        verify_attempt = 0

        while True:
            result = await run_scoped_verification(self.worktree, self.config, self._module_configs, task_files=self._task_files)
            if result.passed:
                return WorkflowOutcome.DONE

            verify_attempt += 1
            if verify_attempt >= self.config.max_verify_attempts:
                return WorkflowOutcome.BLOCKED

            self.metrics.verify_attempts += 1
            logger.info(
                f'Task {self.task_id}: verify attempt {verify_attempt} failed: {result.summary}'
            )

            # Invoke debugger
            self.plan = self.artifacts.read_plan()
            prompt = await self.briefing.build_debugger_prompt(
                result.failure_report(), self.plan, task_id=self.task_id,
            )
            debug_result = await self._invoke(DEBUGGER, prompt, self.worktree)

            # Write debugger iteration log entry
            head_commit = await self._get_head_commit()
            self.artifacts.append_iteration_log({
                'iteration': verify_attempt,
                'agent': 'debugger',
                'steps_attempted': [],
                'steps_completed': [],
                'commit': head_commit,
                'summary': f'Debug fix for: {result.summary[:100]}',
                'source': 'orchestrator',
            })

            # Check for escalations from debugger
            blocking = [e for e in self._check_escalations() if e.severity == 'blocking']
            if blocking:
                return WorkflowOutcome.ESCALATED

            if not debug_result.success:
                logger.warning(f'Task {self.task_id}: debugger failed')

    async def _review(self):
        """Run all 5 reviewers with stagger, retry errors."""
        assert self.worktree is not None and self.artifacts is not None
        base_commit = self.artifacts.read_base_commit()
        if base_commit:
            diff = await self.git_ops.get_diff_from_base(self.worktree, base_commit)
        else:
            diff = await self.git_ops.get_diff_from_main(self.worktree)

        stagger = self.config.reviewer_stagger_secs

        # Staggered launch — spread OAuth session creation
        async def _staggered(idx: int, role: AgentRole):
            if idx > 0:
                await asyncio.sleep(idx * stagger)
            return await self._run_reviewer(role, diff)

        tasks = [_staggered(i, r) for i, r in enumerate(ALL_REVIEWERS)]
        results = list(await asyncio.gather(*tasks, return_exceptions=True))

        # Retry ERROR verdicts and exceptions
        for attempt in range(self.config.max_reviewer_retries):
            error_indices = [
                i for i, r in enumerate(results)
                if isinstance(r, Exception)
                or (isinstance(r, dict) and r.get('verdict') == 'ERROR')
            ]
            if not error_indices:
                break
            logger.info(
                f'Task {self.task_id}: retrying {len(error_indices)} failed '
                f'reviewer(s) (attempt {attempt + 1}/{self.config.max_reviewer_retries})'
            )
            for i in error_indices:
                await asyncio.sleep(stagger)
                try:
                    results[i] = await self._run_reviewer(ALL_REVIEWERS[i], diff)
                except Exception as exc:
                    results[i] = exc

        # Write results — synthesize ERROR for persistent exceptions
        for role, result in zip(ALL_REVIEWERS, results, strict=True):
            if isinstance(result, Exception):
                logger.error(
                    f'Reviewer {role.name} failed after retries: {result}',
                    exc_info=result,
                )
                result = {
                    'reviewer': role.name,
                    'verdict': 'ERROR',
                    'issues': [],
                    'summary': f'Reviewer exception: {result}',
                }
            if isinstance(result, dict):
                self.artifacts.write_review(role.name, result)

        return self.artifacts.aggregate_reviews()

    async def _run_reviewer(self, role: AgentRole, diff: str) -> dict:
        """Run a single reviewer and parse its JSON output."""
        assert self.worktree is not None
        prompt = await self.briefing.build_reviewer_prompt(role.name, diff)

        # Use structured output for reviewers
        review_schema = {
            'type': 'object',
            'properties': {
                'reviewer': {'type': 'string'},
                'verdict': {'type': 'string', 'enum': ['PASS', 'ISSUES_FOUND']},
                'issues': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'severity': {'type': 'string', 'enum': ['blocking', 'suggestion']},
                            'location': {'type': 'string'},
                            'category': {'type': 'string'},
                            'description': {'type': 'string'},
                            'suggested_fix': {'type': 'string'},
                        },
                        'required': ['severity', 'location', 'category', 'description'],
                    },
                },
                'summary': {'type': 'string'},
            },
            'required': ['reviewer', 'verdict', 'issues', 'summary'],
        }

        result = await self._invoke(
            role, prompt, self.worktree, output_schema=review_schema
        )

        if result.structured_output:
            return result.structured_output

        # Try parsing output as JSON
        try:
            return json.loads(result.output)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                f'Reviewer {role.name} produced unparseable output '
                f'(success={result.success}): {result.output[:200]}'
            )
            return {
                'reviewer': role.name,
                'verdict': 'ERROR',
                'issues': [],
                'summary': f'Reviewer error: {result.output[:200]}',
            }

    def _suggestions_in_scope(self, suggestions: list[dict]) -> list[dict]:
        """Filter suggestions to those whose location falls within a module
        this task already holds a lock for.

        Module-lock membership is the scheduler's own concurrency invariant
        (see ``scheduler.normalize_lock``). Filtering this way guarantees an
        amendment pass can't expand the task's lock footprint, and handles
        new files created inside a locked module by construction (a new path
        under a locked module normalizes to the same module key).
        """
        if not suggestions:
            return []
        locked = set(self.modules)
        if not locked:
            logger.warning(
                'Task %s: empty lock set at amendment filter time; '
                'returning zero in-scope suggestions',
                self.task_id,
            )
            return []
        depth = self.config.lock_depth
        in_scope: list[dict] = []
        for s in suggestions:
            location = (s.get('location') or '').strip()
            if not location:
                continue
            # Location format is 'src/foo.py:42' — strip the line number
            file_path = location.split(':', 1)[0].strip()
            if not file_path:
                continue
            module_key = normalize_lock(file_path, depth)
            if module_key and module_key in locked:
                in_scope.append(s)
        return in_scope

    async def _replan(self, reviews) -> None:
        """Feed review feedback back to architect for re-planning."""
        assert self.worktree is not None and self.artifacts is not None
        feedback = reviews.format_for_replan()
        self.plan = self.artifacts.read_plan()

        prompt = f"""\
The implementation was reviewed and blocking issues were found.

{feedback}

# Current Plan

```json
{json.dumps(self.plan, indent=2)}
```

# Action

Update the plan to address the blocking issues. You may add new steps to the `steps` array, but do NOT remove or reorder existing steps. Set new steps to status "pending". Write the updated plan to `.task/plan.json`.
"""
        await self._invoke(ARCHITECT, prompt, self.worktree)
        self.plan = self.artifacts.read_plan()

    async def _amend(
        self, in_scope: list[dict], amendment_round: int,
    ) -> None:
        """Invoke the implementer to apply in-scope review suggestions.

        Amendment passes skip the architect entirely — the plan is frozen,
        no new steps are added, the implementer patches the existing diff
        in place. Scope is enforced by the ``_suggestions_in_scope`` filter
        upstream (module-lock membership) and reinforced in the prompt.
        """
        assert self.worktree is not None and self.artifacts is not None

        self.plan = self.artifacts.read_plan()
        iteration_log, corrupted = self.artifacts.read_iteration_log()
        if corrupted:
            self._escalate_corruption(corrupted)

        prompt = await self.briefing.build_amender_prompt(
            plan=self.plan,
            iteration_log=iteration_log,
            suggestions=in_scope,
            locked_modules=list(self.modules),
            task_id=self.task_id,
        )
        await self._invoke(IMPLEMENTER, prompt, self.worktree)

        head_commit = await self._get_head_commit()
        self.artifacts.append_iteration_log({
            'iteration': self.metrics.execute_iterations,
            'agent': 'implementer',
            'source': 'amendment',
            'amendment_round': amendment_round,
            'suggestions_count': len(in_scope),
            'commit': head_commit,
            'summary': (
                f'Amendment round {amendment_round} '
                f'({len(in_scope)} suggestions)'
            ),
        })

        # Validate plan ownership after the pass — amendment must NOT
        # overwrite plan.json. If it did, the session_id stamp will mismatch.
        if not self.artifacts.validate_plan_owner(self.session_id):
            logger.error(
                'Task %s: plan.json ownership mismatch after amendment pass '
                '(round %d) — implementer was instructed not to touch the plan',
                self.task_id, amendment_round,
            )
            self._escalate_plan_overwrite()

    async def _submit_to_merge_queue(
        self, branch_name: str, pre_rebased: bool = False,
        *, merge_phase: bool = False,
    ) -> WorkflowOutcome:
        """Submit a merge request to the queue and await the result.

        The merge worker handles merging, verification, and CAS
        advancement of main.  Conflicts are returned immediately —
        this method resolves them in the task worktree (outside the
        queue) and re-submits.

        When *merge_phase* is True, escalations created by failure
        paths suppress task-status transitions — the caller retries
        the merge in-place instead of requeueing via the scheduler.
        """
        from orchestrator.merge_queue import MergeOutcome, MergeRequest

        assert self.worktree is not None
        assert self.merge_queue is not None

        future: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
        await self.merge_queue.put(MergeRequest(
            task_id=self.task_id,
            branch=branch_name,
            worktree=self.worktree,
            pre_rebased=pre_rebased,
            task_files=self._task_files,
            module_configs=self._module_configs,
            config=self.config,
            result=future,
        ))

        result = await future

        if result.status == 'wip_halted':
            return await self._handle_wip_conflict(result, branch_name)
        if result.status == 'done_wip_recovery':
            return await self._handle_wip_recovery(result)
        if result.status == 'wip_recovery_no_advance':
            return await self._handle_wip_recovery_no_advance(result)
        if result.status == 'unmerged_state':
            return await self._handle_unmerged_state(result, branch_name)
        if result.status == 'done':
            return WorkflowOutcome.DONE
        if result.status == 'already_merged':
            logger.info(f'Task {self.task_id}: already merged to main')
            return WorkflowOutcome.DONE
        if result.status == 'conflict':
            return await self._resolve_and_resubmit(
                branch_name, result.conflict_details,
                merge_phase=merge_phase,
            )
        # blocked — infer review category from reason
        if 'verification failed' in result.reason.lower():
            category = 'post_merge_verify'
        elif 'ff' in result.reason.lower() or 'advanced' in result.reason.lower():
            category = 'merge_ff_failed'
        else:
            category = 'merge_error'
        self._write_merge_failure_review(category, result.reason)
        return await self._mark_blocked(result.reason, merge_phase=merge_phase)

    async def _resolve_and_resubmit(
        self, branch_name: str, conflict_details: str,
        *, merge_phase: bool = False,
    ) -> WorkflowOutcome:
        """Resolve merge conflicts in the task worktree, then re-submit.

        This runs OUTSIDE the merge queue — the worker is free to process
        other merges while this task resolves its conflicts.
        """
        assert self.worktree is not None
        logger.info(
            f'Task {self.task_id}: merge conflicts detected, '
            f'resolving outside queue'
        )
        task_intent = (
            f"Task: {self.task.get('title', '')}\n"
            f"{self.task.get('description', '')}"
        )
        prompt = await self.briefing.build_merger_prompt(
            conflict_details, task_intent,
        )

        # Rebase onto current main so MERGER works on up-to-date state
        await self.git_ops.rebase_onto_main(self.worktree)
        merger_result = await self._invoke(MERGER, prompt, self.worktree)

        if not merger_result.success or 'BLOCKED' in merger_result.output.upper():
            reason = f'Merger could not resolve: {merger_result.output[:200]}'
            self._write_merge_failure_review('merger_blocked', reason)
            return await self._mark_blocked(reason, merge_phase=merge_phase)

        # Re-submit to queue (now resolved, needs fresh merge)
        return await self._submit_to_merge_queue(
            branch_name, pre_rebased=False, merge_phase=merge_phase,
        )

    async def _handle_wip_conflict(
        self, result, branch_name: str,
    ) -> WorkflowOutcome:
        """Handle a wip_halted merge outcome: create level-1 escalation and wait.

        The merge did NOT land — WIP in project_root overlaps the merge diff.
        After the human resolves (commits/stashes WIP), the task retries the merge.
        """
        overlap = result.overlap_files or []
        detail = (
            f'Merge for task {self.task_id} (branch {branch_name}) was blocked '
            f'because uncommitted work in project_root overlaps the merge diff.\n\n'
            f'Overlapping files:\n'
            + '\n'.join(f'  - {f}' for f in overlap)
            + '\n\nAction required: commit or stash the WIP, then resolve this '
            'escalation to un-halt the merge queue and retry.'
        )
        logger.warning(f'Task {self.task_id}: WIP overlap — creating level-1 escalation')

        if self.escalation_queue:
            from escalation.models import Escalation

            esc = Escalation(
                id=self.escalation_queue.make_id(self.task_id),
                task_id=self.task_id,
                agent_role='orchestrator',
                severity='blocking',
                category='wip_conflict',
                summary=f'WIP overlaps merge diff: {", ".join(overlap[:5])}',
                detail=detail,
                suggested_action='manual_intervention',
                level=1,
                worktree=str(self.worktree) if self.worktree else None,
                workflow_state=self.state.value,
            )
            self.escalation_queue.submit(esc)
            if self.merge_worker is not None:
                self.merge_worker.set_halt_owner(esc.id)
            if self.event_store:
                self.event_store.emit(
                    EventType.escalation_created,
                    task_id=self.task_id, phase=self.state.value,
                    data={'escalation_id': esc.id, 'category': 'wip_conflict',
                          'severity': 'blocking', 'summary': esc.summary[:200]},
                )

            # Wait for human resolution
            if self._escalation_event is None:
                self._escalation_event = asyncio.Event()
            self._escalation_event.clear()
            await self._escalation_event.wait()
            logger.info(f'Task {self.task_id}: WIP conflict resolved — retrying merge')

        return WorkflowOutcome.REQUEUED

    async def _handle_wip_recovery(self, result) -> WorkflowOutcome:
        """Handle a done_wip_recovery merge outcome: merge landed but WIP conflicted.

        The merge IS on main, but the user's stashed WIP conflicted during pop.
        WIP has been preserved on a recovery branch. Create a level-1 escalation
        to inform the human, then return DONE (the task's merge succeeded).
        """
        recovery_branch = result.recovery_branch or '(unknown)'
        detail = (
            f'Merge for task {self.task_id} landed on main successfully, but '
            f'the stash pop of your uncommitted WIP produced conflicts.\n\n'
            f'Your WIP has been preserved on branch: {recovery_branch}\n\n'
            f'To recover:\n'
            f'  git checkout {recovery_branch}\n'
            f'  # Review and cherry-pick or reapply your changes\n\n'
            f'Resolve this escalation to un-halt the merge queue.'
        )
        logger.warning(
            f'Task {self.task_id}: merge landed but stash pop conflicted — '
            f'WIP on {recovery_branch}'
        )

        if self.escalation_queue:
            from escalation.models import Escalation

            esc = Escalation(
                id=self.escalation_queue.make_id(self.task_id),
                task_id=self.task_id,
                agent_role='orchestrator',
                severity='blocking',
                category='wip_conflict',
                summary=f'Stash pop conflict — WIP preserved on {recovery_branch}',
                detail=detail,
                suggested_action='manual_intervention',
                level=1,
                worktree=str(self.worktree) if self.worktree else None,
                workflow_state=self.state.value,
            )
            self.escalation_queue.submit(esc)
            if self.merge_worker is not None:
                self.merge_worker.set_halt_owner(esc.id)
            if self.event_store:
                self.event_store.emit(
                    EventType.escalation_created,
                    task_id=self.task_id, phase=self.state.value,
                    data={'escalation_id': esc.id, 'category': 'wip_conflict',
                          'severity': 'blocking', 'summary': esc.summary[:200]},
                )

            # Wait for human resolution before returning DONE
            if self._escalation_event is None:
                self._escalation_event = asyncio.Event()
            self._escalation_event.clear()
            await self._escalation_event.wait()
            logger.info(f'Task {self.task_id}: WIP recovery escalation resolved')

        return WorkflowOutcome.DONE

    async def _handle_wip_recovery_no_advance(self, result) -> WorkflowOutcome:
        """Handle a wip_recovery_no_advance merge outcome.

        The merge did NOT land on main (CAS failure path). A stash pop conflict
        occurred, and WIP has been preserved on a recovery branch. Create a
        level-1 escalation to inform a human, then return BLOCKED — the task
        cannot proceed until the tree is manually inspected.

        Unlike ``_handle_wip_recovery`` (which returns DONE because the merge
        landed), this returns BLOCKED because main was NOT advanced.
        """
        recovery_branch = result.recovery_branch or '(unknown)'
        detail = (
            f'Merge for task {self.task_id} did NOT advance main. '
            f'A stash pop conflict occurred after a CAS failure, leaving '
            f'the working tree in an unresolvable state.\n\n'
            f'Your WIP has been preserved on branch: {recovery_branch}\n\n'
            f'The merge queue has been halted. To recover:\n'
            f'  git checkout {recovery_branch}\n'
            f'  # Review and reapply your changes to a clean branch\n\n'
            f'Manual intervention required — do NOT let automated tooling '
            f'resolve this escalation. Resolve this escalation to un-halt '
            f'the merge queue.'
        )
        logger.warning(
            f'Task {self.task_id}: stash pop conflicted on CAS-failure path — '
            f'merge did not advance. WIP preserved on {recovery_branch}'
        )

        if self.escalation_queue:
            from escalation.models import Escalation

            esc = Escalation(
                id=self.escalation_queue.make_id(self.task_id),
                task_id=self.task_id,
                agent_role='orchestrator',
                severity='blocking',
                category='wip_conflict',
                summary=f'Stash pop conflict (merge did not advance) — WIP on {recovery_branch}',
                detail=detail,
                suggested_action='manual_intervention',
                level=1,
                worktree=str(self.worktree) if self.worktree else None,
                workflow_state=self.state.value,
            )
            self.escalation_queue.submit(esc)
            if self.merge_worker is not None:
                self.merge_worker.set_halt_owner(esc.id)
            if self.event_store:
                self.event_store.emit(
                    EventType.escalation_created,
                    task_id=self.task_id, phase=self.state.value,
                    data={'escalation_id': esc.id, 'category': 'wip_conflict',
                          'severity': 'blocking', 'summary': esc.summary[:200]},
                )

            # Wait for human resolution — do NOT return DONE (merge did not land)
            if self._escalation_event is None:
                self._escalation_event = asyncio.Event()
            self._escalation_event.clear()
            await self._escalation_event.wait()
            logger.info(f'Task {self.task_id}: wip_recovery_no_advance escalation resolved')

        return WorkflowOutcome.BLOCKED

    async def _handle_unmerged_state(
        self, result, branch_name: str,
    ) -> WorkflowOutcome:
        """Handle an unmerged_state merge outcome.

        ``project_root`` had pre-existing UU/AA/DD markers BEFORE this merge
        attempted to advance main. The merge did NOT land, and the tree is
        already in an inconsistent state. Halt stays in effect until a human
        inspects, cleans up project_root (``git mergetool`` / manual
        resolution / ``git reset``), and resolves the escalation.
        """
        detail = (
            f'Merge for task {self.task_id} (branch {branch_name}) was '
            f'blocked because project_root already has unresolved merge '
            f'conflicts (UU/AA/DD markers) from a prior, unrelated event — '
            f'the merge queue refuses to stash/advance over a partially '
            f'resolved tree.\n\n'
            f'Action required: inspect ``git status`` in project_root, '
            f'resolve the existing merge state (``git mergetool`` / edit '
            f'the conflicted files / ``git reset`` to abandon the prior '
            f'merge), then resolve this escalation to un-halt the merge '
            f'queue.\n\n'
            f'Manual intervention required — do NOT let automated tooling '
            f'resolve this escalation.'
        )
        logger.warning(
            f'Task {self.task_id}: unmerged_state in project_root — '
            f'creating level-1 escalation'
        )

        if self.escalation_queue:
            from escalation.models import Escalation

            esc = Escalation(
                id=self.escalation_queue.make_id(self.task_id),
                task_id=self.task_id,
                agent_role='orchestrator',
                severity='blocking',
                category='unmerged_state',
                summary=(
                    'project_root has unresolved (UU/AA/DD) markers — '
                    'merge queue halted'
                ),
                detail=detail,
                suggested_action='manual_intervention',
                level=1,
                worktree=str(self.worktree) if self.worktree else None,
                workflow_state=self.state.value,
            )
            self.escalation_queue.submit(esc)
            if self.merge_worker is not None:
                self.merge_worker.set_halt_owner(esc.id)
            if self.event_store:
                self.event_store.emit(
                    EventType.escalation_created,
                    task_id=self.task_id, phase=self.state.value,
                    data={'escalation_id': esc.id, 'category': 'unmerged_state',
                          'severity': 'blocking', 'summary': esc.summary[:200]},
                )

            if self._escalation_event is None:
                self._escalation_event = asyncio.Event()
            self._escalation_event.clear()
            await self._escalation_event.wait()
            logger.info(
                f'Task {self.task_id}: unmerged_state escalation resolved'
            )

        return WorkflowOutcome.BLOCKED

    def _write_merge_failure_review(self, category: str, detail: str) -> None:
        """Write a review-format JSON describing a merge failure to .task/reviews/.

        Uses the same schema as reviewer agents so humans and retry agents
        can consume it uniformly.
        """
        if not self.artifacts:
            return
        review = {
            'reviewer': 'merge',
            'verdict': 'ISSUES_FOUND',
            'issues': [
                {
                    'severity': 'blocking',
                    'location': 'main',
                    'category': category,
                    'description': detail,
                },
            ],
            'summary': detail[:200],
        }
        self.artifacts.write_review('merge', review)

    def _select_model_for_role(self, role: AgentRole, base_model: str) -> str:
        """Override model for implementer/debugger based on task complexity."""
        if role.name not in ('implementer', 'debugger'):
            return base_model

        # Check for Rust modules (crates/ prefix is the convention)
        rust_modules = [m for m in self.modules if m.startswith('crates/')]
        if len(rust_modules) < 3:
            return base_model

        # Check step count if plan is available (always true for implementer/debugger)
        if self.plan:
            step_count = len(self.plan.get('steps', []))
            if step_count >= 12:
                logger.info(
                    'Task %s: upgrading %s to opus (%d Rust modules, %d steps)',
                    self.task_id, role.name, len(rust_modules), step_count,
                )
                return 'opus'

        return base_model

    async def _invoke(
        self,
        role: AgentRole,
        prompt: str,
        cwd: Path,
        output_schema: dict | None = None,
    ) -> AgentResult:
        """Invoke an agent with role-specific configuration."""
        # Get role-specific config overrides
        models = self.config.models
        budgets = self.config.budgets
        turns = self.config.max_turns
        effort_cfg = self.config.effort
        timeouts_cfg = self.config.timeouts
        backends_cfg = self.config.backends

        role_key = role.name.split('_')[0]

        model = getattr(models, role_key, role.default_model)
        model = self._select_model_for_role(role, model)
        budget = getattr(budgets, role_key, role.default_budget)
        max_turns_val = getattr(turns, role_key, role.default_max_turns)
        effort_val = getattr(effort_cfg, role_key, 'high')
        timeout_val = getattr(timeouts_cfg, role_key, self.config.invocation_timeout)
        backend_val = getattr(backends_cfg, role_key, 'claude')

        # Use reviewer config for all reviewer variants
        if role.name.startswith('reviewer'):
            model = models.reviewer
            budget = budgets.reviewer
            max_turns_val = turns.reviewer
            effort_val = effort_cfg.reviewer
            timeout_val = timeouts_cfg.reviewer
            backend_val = backends_cfg.reviewer

        # Determine sandbox modules based on role
        sandbox_modules = None
        if self.config.sandbox.enabled and role.name in ('implementer', 'debugger'):
            sandbox_modules = self.modules

        # Build MCP config — fused-memory always, escalation when available.
        # Judge gets MCP so its jcodemunch tools (in allowed_tools) actually
        # work; it does not use escalation tools but mcp_config_json handles
        # escalation_url=None fine.
        mcp_config = None
        if role.name in ('architect', 'implementer', 'debugger', 'merger', 'judge'):
            escalation_url = None
            if self.escalation_queue:
                esc = self.config.escalation
                escalation_url = f'http://{esc.host}:{esc.port}/mcp'
            mcp_config = self.mcp.mcp_config_json(escalation_url=escalation_url)

        # Plan-tools stdio MCP server — architect builds plans, implementer/
        # debugger marks steps done.  Per-invocation isolation: each agent
        # gets its own server bound to the worktree path.
        if role.name in ('architect', 'implementer', 'debugger') and cwd:
            if not mcp_config:
                mcp_config = {'mcpServers': {}}
            mcp_config.setdefault('mcpServers', {})['plan-tools'] = {
                'command': 'uv',
                'args': [
                    'run', '--project', str(_ORCH_PROJECT_DIR),
                    'python', '-m', 'orchestrator.mcp.plan_tools',
                    '--worktree', str(cwd),
                ],
            }

        started_at = datetime.now(UTC).isoformat()
        result = await invoke_with_cap_retry(
            usage_gate=self.usage_gate,
            label=f'Task {self.task_id} [{role.name}]',
            config_dir=self._config_dir,
            prompt=prompt,
            system_prompt=role.system_prompt,
            cwd=cwd,
            model=model,
            max_turns=max_turns_val,
            max_budget_usd=budget,
            allowed_tools=role.allowed_tools or None,
            disallowed_tools=role.disallowed_tools or None,
            mcp_config=mcp_config,
            output_schema=output_schema,
            sandbox_modules=sandbox_modules,
            effort=effort_val,
            backend=backend_val,
            timeout_seconds=timeout_val,
            # Judge always hits Claude API — propagating ANTHROPIC_BASE_URL
            # routes it through vLLM where max_model_len causes
            # ServerDisconnectedError after 2 tool-use rounds (3cd380a079).
            # Cap hits on Claude API are handled by UsageGate account failover
            # (wired in runner.py for eval mode).
            env_overrides=(self.config.env_overrides or None) if role.name in ('implementer', 'debugger') else None,
        )
        completed_at = datetime.now(UTC).isoformat()

        # Track metrics
        self.metrics.total_cost_usd += result.cost_usd
        self.metrics.total_duration_ms += result.duration_ms
        self.metrics.agent_invocations += 1
        self.metrics.total_turns += result.turns
        self.metrics.total_input_tokens += result.input_tokens or 0
        self.metrics.total_output_tokens += result.output_tokens or 0
        self.metrics.total_cache_read_tokens += result.cache_read_tokens or 0
        self.metrics.total_cache_create_tokens += result.cache_create_tokens or 0

        logger.info(
            'Task %s [%s]: success=%s cost=$%.2f turns=%d timeout=%.0fs',
            self.task_id, role.name, result.success, result.cost_usd,
            result.turns, timeout_val,
            extra={
                'task_id': self.task_id, 'role': role.name, 'model': model,
                'cost_usd': result.cost_usd, 'turns': result.turns,
                'input_tokens': result.input_tokens,
                'output_tokens': result.output_tokens,
            },
        )

        if self.event_store:
            self.event_store.emit(
                EventType.invocation_end,
                task_id=self.task_id,
                phase=self.state.value,
                role=role.name,
                cost_usd=result.cost_usd,
                duration_ms=result.duration_ms,
                data={
                    'turns': result.turns,
                    'success': result.success,
                    'subtype': result.subtype,
                    'model': model,
                    'account_name': result.account_name,
                    'input_tokens': result.input_tokens,
                    'output_tokens': result.output_tokens,
                    'cache_read_tokens': result.cache_read_tokens,
                    'cache_create_tokens': result.cache_create_tokens,
                },
            )

        if self.cost_store:
            try:
                await self.cost_store.save_invocation(
                    run_id=self.event_store.run_id if self.event_store else '',
                    task_id=self.task_id,
                    project_id=self.config.fused_memory.project_id,
                    account_name=result.account_name,
                    model=model,
                    role=role.name,
                    cost_usd=result.cost_usd,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cache_read_tokens=result.cache_read_tokens,
                    cache_create_tokens=result.cache_create_tokens,
                    duration_ms=result.duration_ms,
                    capped=False,
                    started_at=started_at,
                    completed_at=completed_at,
                )
            except Exception:
                logger.warning('Failed to save invocation cost', exc_info=True)

        return result

    def _check_escalations(self):
        """Check for pending escalations for this task."""
        if not self.escalation_queue:
            return []
        return self.escalation_queue.get_by_task(self.task_id, status='pending')

    async def _wait_for_resolution(self) -> str:
        """Wait for all level-0 pending escalations to be resolved.

        Raises ``_StewardReescalated`` if the steward re-escalated to
        level-1 (human), indicating the task should be blocked.

        When no escalation queue is available (e.g. eval mode), returns
        an empty string immediately — the caller treats this as "no
        resolution" and the workflow proceeds to ESCALATED/BLOCKED
        via its normal path.
        """
        if self.escalation_queue is None:
            logger.warning(
                'Task %s: _wait_for_resolution called without escalation_queue '
                '(eval mode?) — returning immediately',
                self.task_id,
            )
            return ''

        if self._escalation_event is None:
            self._escalation_event = asyncio.Event()

        # Wait for level-0 pending escalations to clear
        while True:
            pending_l0 = self.escalation_queue.get_by_task(
                self.task_id, status='pending', level=0,
            )
            if not pending_l0:
                break
            self._escalation_event.clear()
            await self._escalation_event.wait()

        # Check for level-1 re-escalation (steward gave up)
        pending_l1 = self.escalation_queue.get_by_task(
            self.task_id, status='pending', level=1,
        )
        if pending_l1:
            raise _StewardReescalated(pending_l1)

        # Collect resolutions
        resolved = [
            e for e in self.escalation_queue.get_by_task(self.task_id)
            if e.status == 'resolved' and e.resolution
        ]
        return '\n'.join(e.resolution for e in resolved)

    async def _get_head_commit(self) -> str:
        """Return the HEAD commit SHA for the current worktree."""
        proc = await asyncio.create_subprocess_exec(
            'git', 'rev-parse', 'HEAD',
            cwd=str(self.worktree),
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode().strip()

    def _has_prior_implementation(self) -> bool:
        """Check whether a prior run did any implementation in this worktree.

        Inspects .task/iterations.jsonl for implementer/debugger entries.
        Planning-only runs don't write these, so absence means stale branch
        point rather than a legitimately merged prior run.

        Correctness assumption: worktrees are always created fresh per task run.
        If .task/iterations.jsonl were ever carried across a worktree recreation
        (e.g. worktree reuse on the same branch), this guard could incorrectly
        return True for an empty branch and reintroduce the false-done path.
        Any future change to create_worktree that enables reuse must revisit here.
        """
        if self.artifacts is None:
            return False
        entries, _ = self.artifacts.read_iteration_log()
        return any(e.get('agent') in ('implementer', 'debugger') for e in entries)

    def _escalate_plan_overwrite(self) -> None:
        """Submit a blocking escalation when plan.json is overwritten by a foreign session."""
        summary = f'plan.json overwrite detected for task {self.task_id}'
        detail = (
            f'Expected _session_id={self.session_id} but plan.json contains a different value. '
            f'A duplicate workflow may have overwritten plan.json.'
        )
        logger.error(f'Task {self.task_id}: {summary}')

        if not self.escalation_queue:
            return

        from escalation.models import Escalation

        esc = Escalation(
            id=self.escalation_queue.make_id(self.task_id),
            task_id=self.task_id,
            agent_role='orchestrator',
            severity='blocking',
            category='infra_issue',
            summary=summary,
            detail=detail,
            suggested_action='investigate_and_retry',
            worktree=str(self.worktree) if self.worktree else None,
            workflow_state=self.state.value,
        )
        self.escalation_queue.submit(esc)
        if self.event_store:
            self.event_store.emit(
                EventType.escalation_created,
                task_id=self.task_id, phase=self.state.value,
                data={'escalation_id': esc.id, 'category': esc.category,
                      'severity': esc.severity, 'summary': summary[:200]},
            )

    def _escalate_corruption(self, corrupted: list[str]) -> None:
        """Submit an info-severity escalation for corrupted iteration log lines."""
        if not self.escalation_queue:
            logger.warning(
                'Task %s: %d corrupted iteration log lines (no escalation queue)',
                self.task_id, len(corrupted),
            )
            return

        from escalation.models import Escalation

        detail = f'{len(corrupted)} corrupted line(s):\n' + '\n'.join(
            line[:200] for line in corrupted[:10]
        )
        esc = Escalation(
            id=self.escalation_queue.make_id(self.task_id),
            task_id=self.task_id,
            agent_role='orchestrator',
            severity='info',
            category='infra_issue',
            summary=f'{len(corrupted)} corrupted iteration log line(s)',
            detail=detail,
            suggested_action='investigate_log_corruption',
            worktree=str(self.worktree) if self.worktree else None,
            workflow_state=self.state.value,
        )
        self.escalation_queue.submit(esc)

    async def _mark_blocked(
        self, reason: str, *, detail: str = '',
        skip_escalation: bool = False,
        merge_phase: bool = False,
    ) -> WorkflowOutcome:
        """Mark task as blocked and optionally create an escalation entry.

        *reason* is used as the escalation summary (truncated to 200 chars).
        *detail* is the full diagnostic context persisted in the escalation
        file; defaults to *reason* when not provided.
        *skip_escalation* suppresses escalation creation when a level-1
        escalation already exists (e.g. steward re-escalated to human).
        *merge_phase* suppresses task-status transitions (blocked/pending)
        when the caller will retry the merge in-place rather than requeueing
        through the scheduler.
        """
        if self.state == WorkflowState.DONE:
            logger.warning(
                'Task %s: already DONE, ignoring late blocked transition: %s',
                self.task_id, reason,
            )
            return WorkflowOutcome.DONE
        if not merge_phase:
            self._enter_phase(WorkflowState.BLOCKED)
            await self.scheduler.set_task_status(self.task_id, 'blocked')
        logger.warning(f'Task {self.task_id} BLOCKED: {reason}')

        if self.escalation_queue and skip_escalation:
            # Defensive cleanup: L0 should already be cleared by
            # _wait_for_resolution, but dismiss any stragglers (race
            # between the L0-empty check and the L1 check).
            remaining_l0 = self.escalation_queue.get_by_task(
                self.task_id, status='pending', level=0,
            )
            if remaining_l0:
                logger.warning(
                    'Task %s: %d L0 escalation(s) still pending at '
                    'mark_blocked(skip_escalation=True) — dismissing',
                    self.task_id, len(remaining_l0),
                )
                for esc in remaining_l0:
                    self.escalation_queue.resolve(
                        esc.id,
                        'Auto-dismissed: task blocked after steward re-escalation',
                        dismiss=True,
                        resolved_by='auto-dismissed',
                    )

        if self.escalation_queue and not skip_escalation:
            # Don't create a duplicate if level-1 already pending
            existing_l1 = self.escalation_queue.get_by_task(
                self.task_id, status='pending', level=1,
            )
            if not existing_l1:
                from escalation.models import Escalation

                esc = Escalation(
                    id=self.escalation_queue.make_id(self.task_id),
                    task_id=self.task_id,
                    agent_role='orchestrator',
                    severity='blocking',
                    category='task_failure',
                    summary=reason[:200],
                    detail=detail or reason,
                    suggested_action='investigate_and_retry',
                    worktree=str(self.worktree) if self.worktree else None,
                    workflow_state=self.state.value,
                )
                self.escalation_queue.submit(esc)

                if self.event_store:
                    self.event_store.emit(
                        EventType.escalation_created,
                        task_id=self.task_id, phase=self.state.value,
                        data={'escalation_id': esc.id, 'category': 'task_failure',
                              'severity': 'blocking', 'summary': reason[:200]},
                    )

            # Give the steward a chance to resolve the escalation
            await self._ensure_steward_started()
            if self._steward:
                await self._await_steward_completion()

                # Guard: steward may have marked the task done (terminal).
                # The scheduler rejects done→pending, but we must also
                # return the correct outcome.
                cached = self.scheduler.get_cached_status(self.task_id)
                if cached in TERMINAL_STATUSES:
                    logger.info(
                        'Task %s: status is %s after steward — not re-queueing',
                        self.task_id, cached,
                    )
                    if cached == 'done':
                        self._enter_phase(WorkflowState.DONE)
                        return WorkflowOutcome.DONE
                    return WorkflowOutcome.BLOCKED

                # If steward resolved all level-0 escalations, set task back
                # to pending so the scheduler re-picks it on the next cycle.
                remaining = self.escalation_queue.get_by_task(
                    self.task_id, status='pending', level=0,
                )
                if not remaining:
                    # Guard: if the task's branch is already merged to
                    # main, transition to DONE instead of re-queueing —
                    # prevents stash_failed / advance_main ghost loops.
                    if self.worktree and self.git_ops:
                        try:
                            _, wt_head, _ = await _run(
                                ['git', 'rev-parse', 'HEAD'],
                                cwd=self.worktree,
                            )
                            main_sha = await self.git_ops.get_main_sha()
                            if await self.git_ops.is_ancestor(
                                wt_head.strip(), main_sha,
                            ):
                                if not self._has_prior_implementation():
                                    _base = (
                                        self.artifacts.read_base_commit()
                                        if self.artifacts else None
                                    )
                                    _entries, _ = (
                                        self.artifacts.read_iteration_log()
                                        if self.artifacts else ([], [])
                                    )
                                    logger.warning(
                                        'Task %s: branch HEAD %s is ancestor '
                                        'of main %s but no implementation '
                                        'entries (base=%s, entries=%d) — '
                                        'proceeding with requeue',
                                        self.task_id,
                                        wt_head.strip()[:8],
                                        main_sha[:8],
                                        _base[:8] if _base else 'none',
                                        len(_entries),
                                    )
                                else:
                                    logger.info(
                                        'Task %s: branch already on main — '
                                        'completing instead of re-queueing',
                                        self.task_id,
                                    )
                                    self._enter_phase(WorkflowState.DONE)
                                    await self.scheduler.set_task_status(
                                        self.task_id, 'done',
                                    )
                                    return WorkflowOutcome.DONE
                        except Exception:
                            logger.warning(
                                'Task %s: merge-check failed, '
                                'proceeding with requeue',
                                self.task_id, exc_info=True,
                            )

                    if self.event_store:
                        self.event_store.emit(
                            EventType.escalation_resolved,
                            task_id=self.task_id, phase=self.state.value,
                            data={'outcome': 'requeued'},
                        )
                    if not merge_phase:
                        await self.scheduler.set_task_status(self.task_id, 'pending')
                        logger.info(
                            f'Task {self.task_id}: steward resolved blocking '
                            f'escalation, reset to pending for re-scheduling'
                        )
                    else:
                        logger.info(
                            f'Task {self.task_id}: steward resolved blocking '
                            f'escalation, caller will retry merge in-place'
                        )
                    return WorkflowOutcome.REQUEUED

        return WorkflowOutcome.BLOCKED

    async def _write_completion_to_memory(self) -> None:
        """Write task completion summary so dependent tasks find it in briefings."""
        parts = [f"Completed: {self.task.get('title', '')}"]

        desc = self.task.get('description', '')
        if desc:
            parts.append(f"Description: {desc}")

        analysis = self.plan.get('analysis', '')
        if analysis:
            parts.append(f"Analysis: {analysis}")

        decisions = self.plan.get('design_decisions', [])
        if decisions:
            decision_text = '; '.join(
                d.get('decision', '') for d in decisions[:3]
            )
            parts.append(f"Key decisions: {decision_text}")

        steps = self.plan.get('steps', [])
        done_count = sum(1 for s in steps if isinstance(s, dict) and s.get('status') == 'done')
        parts.append(f"Steps completed: {done_count}/{len(steps)}")

        if self.modules:
            parts.append(f"Modules: {', '.join(self.modules)}")

        content = '\n'.join(parts)

        try:
            import httpx as httpx_mod
            async with httpx_mod.AsyncClient() as client:
                await client.post(
                    f'{self.mcp.url}/mcp/',
                    json={
                        'jsonrpc': '2.0',
                        'id': 1,
                        'method': 'tools/call',
                        'params': {
                            'name': 'add_memory',
                            'arguments': {
                                'content': content,
                                'category': 'observations_and_summaries',
                                'project_id': self.config.fused_memory.project_id,
                                'agent_id': f'orchestrator-task-{self.task_id}',
                            },
                        },
                    },
                    timeout=10,
                )
        except Exception as e:
            logger.warning(f'Failed to write completion to memory: {e}')

    async def _write_decisions_to_memory(self) -> None:
        """Write plan design decisions to fused-memory."""
        decisions = self.plan.get('design_decisions', [])
        if not decisions:
            return
        try:
            async with __import__('httpx').AsyncClient() as client:
                for decision in decisions:
                    await client.post(
                        f'{self.mcp.url}/mcp/',
                        json={
                            'jsonrpc': '2.0',
                            'id': 1,
                            'method': 'tools/call',
                            'params': {
                                'name': 'add_memory',
                                'arguments': {
                                    'content': f"Decision: {decision['decision']}\nRationale: {decision['rationale']}",
                                    'category': 'decisions_and_rationale',
                                    'project_id': self.config.fused_memory.project_id,
                                    'agent_id': f'orchestrator-task-{self.task_id}',
                                },
                            },
                        },
                        timeout=10,
                    )
        except Exception as e:
            logger.warning(f'Failed to write decisions to memory: {e}')

    async def _write_suggestions_to_memory(self, reviews) -> None:
        """Write review suggestions (non-blocking) to memory as conventions."""
        suggestions = reviews.suggestions
        if not suggestions:
            return
        try:
            import httpx as httpx_mod
            async with httpx_mod.AsyncClient() as client:
                for suggestion in suggestions[:5]:  # cap at 5 to avoid noise
                    await client.post(
                        f'{self.mcp.url}/mcp/',
                        json={
                            'jsonrpc': '2.0',
                            'id': 1,
                            'method': 'tools/call',
                            'params': {
                                'name': 'add_memory',
                                'arguments': {
                                    'content': f"[{suggestion.get('category', '')}] {suggestion.get('description', '')}",
                                    'category': 'preferences_and_norms',
                                    'project_id': self.config.fused_memory.project_id,
                                    'agent_id': f'orchestrator-task-{self.task_id}',
                                },
                            },
                        },
                        timeout=10,
                    )
        except Exception as e:
            logger.warning(f'Failed to write suggestions to memory: {e}')

    def _escalate_suggestions(self, reviews) -> None:
        """Submit review suggestions as an info escalation for steward triage."""
        from escalation.models import Escalation

        suggestions = reviews.suggestions
        if not suggestions or not self.escalation_queue:
            return

        # Content fingerprint: skip if identical suggestions already escalated
        content_hash = hashlib.sha256(
            json.dumps(suggestions, sort_keys=True).encode(),
        ).hexdigest()[:16]

        existing = self.escalation_queue.get_by_task(self.task_id)
        for prev in existing:
            if (
                prev.category == 'review_suggestions'
                and prev.detail
                and prev.detail.startswith(f'#hash:{content_hash}#')
            ):
                logger.info(
                    'Task %s: skipping duplicate review_suggestions escalation '
                    '(content hash %s matches %s)',
                    self.task_id, content_hash, prev.id,
                )
                return

        detail = f'#hash:{content_hash}#' + json.dumps(suggestions)

        esc = Escalation(
            id=self.escalation_queue.make_id(self.task_id),
            task_id=self.task_id,
            agent_role='orchestrator',
            severity='info',
            category='review_suggestions',
            summary=f'{len(suggestions)} review suggestion(s) for triage',
            detail=detail,
            suggested_action='triage_suggestions',
            worktree=str(self.worktree) if self.worktree else None,
            workflow_state=self.state.value,
        )
        self.escalation_queue.submit(esc)
        if self.event_store:
            self.event_store.emit(
                EventType.escalation_created,
                task_id=self.task_id, phase=self.state.value,
                data={'escalation_id': esc.id, 'category': 'review_suggestions',
                      'severity': 'info', 'count': len(suggestions)},
            )
        logger.info(
            f'Task {self.task_id}: submitted {len(suggestions)} suggestions '
            f'for steward triage ({esc.id})'
        )

    def _escalate_review_issues(self, reviews) -> None:
        """Submit remaining review issues as a blocking escalation for the steward."""
        if not self.escalation_queue:
            return

        from escalation.models import Escalation

        detail = reviews.format_for_replan()
        n_blocking = len(reviews.blocking_issues)
        n_suggestions = len(reviews.suggestions)

        esc = Escalation(
            id=self.escalation_queue.make_id(self.task_id),
            task_id=self.task_id,
            agent_role='orchestrator',
            severity='blocking',
            category='review_issues',
            summary=(
                f'Review cycles exhausted with {n_blocking} blocking issue(s) '
                f'and {n_suggestions} suggestion(s)'
            ),
            detail=detail,
            suggested_action='fix_review_issues',
            worktree=str(self.worktree) if self.worktree else None,
            workflow_state=self.state.value,
        )
        self.escalation_queue.submit(esc)
        if self.event_store:
            self.event_store.emit(
                EventType.escalation_created,
                task_id=self.task_id, phase=self.state.value,
                data={'escalation_id': esc.id, 'category': 'review_issues',
                      'severity': 'blocking', 'n_blocking': n_blocking,
                      'n_suggestions': n_suggestions},
            )
        logger.info(
            f'Task {self.task_id}: escalated {n_blocking} review issues '
            f'to steward ({esc.id})'
        )

    async def _ensure_steward_started(self) -> None:
        """Start the steward lazily on first call, if factory was provided."""
        if self._steward is not None:
            return
        if not self._steward_factory or not self.worktree:
            return
        # Check if there are pending level-0 escalations worth starting for
        if self.escalation_queue:
            pending = self.escalation_queue.get_by_task(
                self.task_id, status='pending', level=0,
            )
            if not pending:
                return
        steward = self._steward_factory(self.worktree, self._config_dir)
        self._steward = steward
        await steward.start()

    async def _await_steward_completion(self) -> None:
        """Wait for the steward to finish pending work, with grace period.

        On timeout, auto-re-escalate remaining level-0 escalations to
        level 1 (steward→human) and dismiss the originals.

        Only waits if the steward is actually running — otherwise there's
        nothing to wait for.
        """
        if not self.escalation_queue or not self._steward:
            return

        timeout = self.config.steward_completion_timeout
        queue = self.escalation_queue

        def _pending_l0():
            return queue.get_by_task(self.task_id, status='pending', level=0)

        pending = _pending_l0()
        if not pending:
            return

        logger.info(
            f'Task {self.task_id}: waiting up to {timeout:.0f}s for steward completion'
        )

        if self._escalation_event is None:
            self._escalation_event = asyncio.Event()

        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            pending = _pending_l0()
            if not pending:
                logger.info(f'Task {self.task_id}: steward completed all pending work')
                return

            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break

            self._escalation_event.clear()
            try:
                await asyncio.wait_for(
                    self._escalation_event.wait(), timeout=remaining,
                )
            except TimeoutError:
                break

        # Timeout — re-escalate remaining to level 1
        from escalation.models import Escalation

        logger.warning(
            f'Task {self.task_id}: steward completion timed out after {timeout:.0f}s, '
            f're-escalating {len(pending)} item(s) to level 1'
        )
        for esc in pending:
            reesc = Escalation(
                id=queue.make_id(self.task_id),
                task_id=self.task_id,
                agent_role='steward',
                severity=esc.severity,
                category=esc.category,
                summary=f'Steward timeout: {esc.summary}',
                detail=esc.detail,
                suggested_action='manual_intervention',
                level=1,
            )
            queue.submit(reesc)
            queue.resolve(
                esc.id,
                'Auto-dismissed: steward completion timeout, re-escalated to level 1',
                dismiss=True,
            )
