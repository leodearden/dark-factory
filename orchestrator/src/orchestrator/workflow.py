"""Per-task workflow state machine: PLAN → EXECUTE → VERIFY → REVIEW → MERGE → DONE."""

from __future__ import annotations

import asyncio
import contextlib
import enum
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

from shared.cli_invoke import (
    AllAccountsCappedException,
    classify_agent_failure,
    invoke_with_cap_retry,
)
from shared.config_dir import TaskConfigDir
from shared.cost_store import CostStore

from orchestrator.agents.briefing import COMPLETION_JUDGE_SCHEMA
from orchestrator.agents.invoke import AgentResult, invoke_agent
from orchestrator.agents.roles import (
    _ESCALATION_TOOLS,
    ALL_REVIEWERS,
    ARCHITECT,
    DEBUGGER,
    IMPLEMENTER,
    JUDGE,
    MERGER,
    ROLES,
    AgentRole,
)
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment, files_to_modules, normalize_lock
from orchestrator.task_status import TERMINAL_STATUSES, WORKFLOW_PRESERVE_STATUSES
from orchestrator.usage_gate import SessionBudgetExhausted as _SessionBudgetExhausted
from orchestrator.verify import VerifyResult, run_scoped_verification

# Orchestrator package directory — used to resolve ``uv run --project`` for
# the plan-tools stdio MCP server.
_ORCH_PROJECT_DIR = Path(__file__).resolve().parents[2]

# Roles whose allowed_tools include at least one 'mcp__escalation__escalate*' tool.
# 'steward' and 'deep_reviewer' are excluded: they run in their own dispatchers
# (TaskSteward and ReviewCheckpoint respectively), not through TaskWorkflow._invoke.
# All other roles are included or excluded based on their actual allowed_tools entries in ROLES.
_ESCALATION_CAPABLE_ROLES: frozenset[str] = frozenset(
    name for name, role in ROLES.items()
    if any(t in _ESCALATION_TOOLS for t in (role.allowed_tools or []))
    and name not in {'steward', 'deep_reviewer'}
)


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
        self,
        task_id: str,
        status: str,
        /,
        *,
        done_provenance: dict | None = ...,
        reopen_reason: str | None = ...,
    ) -> None: ...
    async def handle_blast_radius_expansion(
        self, task_id: str, current: list[str], needed: list[str], /
    ) -> bool: ...
    async def get_status(self, task_id: str, /) -> str | None: ...
    async def update_task(
        self, task_id: str, metadata: str | dict,
    ) -> bool: ...
    async def dispatch_tool(
        self, name: str, arguments: dict, *, timeout: float = ...,
    ) -> dict: ...


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
    PLANNED = 'planned'
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


class _PriorImplStatus(NamedTuple):
    """Result of :meth:`TaskWorkflow._has_prior_implementation`.

    Bundles the three pieces of information that the merge-check call-site
    needs so it can avoid redundant artifact reads.
    """

    has_work: bool
    """True iff the worktree has implementation commits beyond the base."""

    entries: list[dict]
    """Full parsed iteration-log entries (may be empty)."""

    base_commit: str | None
    """SHA read from metadata.json, or None if the file is absent."""


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
        cancel_event: asyncio.Event | None = None,
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
        self._escalation_missing_warned: bool = False

        # Soft-cancel: settable from outside the workflow (e.g. by the
        # harness when reconciliation reports the task is now terminal,
        # or by the ``release_workflow`` MCP tool).  Long awaits in the
        # merge queue / steward grace period race against this event so
        # the workflow can exit promptly without waiting for a 900 s
        # timeout.  See Step 4 of the zombie-escalation fix.
        self._cancel_event: asyncio.Event = cancel_event or asyncio.Event()

        # Usage cap gate
        self.usage_gate = usage_gate

        # Unique session identifier for plan ownership (format: {task_id}-{uuid_hex[:8]})
        self.session_id = f'{self.task_id}-{uuid.uuid4().hex[:8]}'

        self._steward_factory = steward_factory
        self._steward: Any | None = None
        self._config_dir: TaskConfigDir | None = None
        self._old_plan_base: str | None = None  # base commit from prior session (for revalidation diff)
        self._merge_sha: str | None = None  # merge commit SHA set by _submit_to_merge_queue on success
        self._last_completed_role: str | None = None  # role of the last successfully-completed invocation
        self._last_verify_result: VerifyResult | None = None  # most recent failing VerifyResult from _verify_debugfix_loop
        # Block-reason surfacing for the harness-level retry cap.  Populated
        # when _mark_blocked takes the REQUEUED return path; the harness reads
        # these after workflow.run() returns to decide whether to increment
        # the per-task requeue counter.
        self._last_block_reason: str = ''
        self._last_block_detail: str = ''
        self._last_block_phase: str = ''

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

            # ── Pre-PLAN ghost-loop recovery ──────────────────────────
            # If the worktree's branch is already merged to main AND there
            # is evidence of prior implementation work, mark the task DONE
            # immediately — a prior run merged it but died before writing
            # the DONE status.  Short-circuiting here prevents the architect
            # from being invoked and keeps the run idempotent.
            #
            # NOTE: a related guard runs just below (before EXECUTE) and has
            # deliberately different semantics: the post-PLAN guard falls through
            # to the SUCCESS path (writes completion memory, uses merge-sha
            # provenance), and it also checks has_uncommitted_work (useful
            # post-execution, premature here).  If you change the
            # is_ancestor/has_work logic, check both guards.
            recovery = await self._recover_if_already_merged()
            if recovery == WorkflowOutcome.DONE:
                return recovery

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
                # WorkflowOutcome.PLANNED falls through to execute/verify/review.

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
            #
            # See also: the pre-PLAN _recover_if_already_merged() call
            # above.  That guard returns DONE directly (skips completion
            # memory); this guard falls through to the SUCCESS path which
            # writes completion memory.  The two guards cover complementary
            # failure modes — do not collapse them.
            _branch_check = await self._check_branch_on_main()
            already_on_main = _branch_check is not None
            if already_on_main and not self._worktree_external:
                # Guard: a stale branch point (requeued task that was planned
                # but never implemented, or a freshly-created worktree) also
                # satisfies the ancestor check.  Only skip if there's
                # evidence of prior implementation work.
                #
                # WT_HEAD INTENTIONALLY OMITTED — see _has_prior_implementation()
                # docstring.  This caller is reached after a genuine rebase, so
                # wt_head may equal the new base_commit even on a
                # genuinely-implemented branch.  The iteration-log fallback is the
                # correct signal here: if there's an implementer entry in the log,
                # the branch has real work and we should skip to DONE.  Passing
                # wt_head would cause the SHA-primary check to return has_work=False
                # on any rebased branch, silently discarding completed work.
                assert _branch_check is not None  # narrowing: already_on_main is True
                wt_head, _ = _branch_check
                has_work = (
                    self._has_prior_implementation().has_work
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

                        # Honor steward terminal decisions BEFORE resuming the
                        # implementer.  The steward may have set the task to
                        # done / cancelled / deferred / blocked while resolving
                        # the L0 (e.g. queued a follow-up task that is now the
                        # durable fix and deferred this one onto it).  Without
                        # this guard the resume loop keeps invoking the
                        # implementer/debugger until verify-attempt budget
                        # exhausts — burning $7-8 per cycle on a task the
                        # steward already decided to park.  Mirrors the inline
                        # returns inside _mark_blocked (~L3125–3168) but
                        # bypasses it so we do NOT file an L1 for what is an
                        # intentional steward terminal decision.
                        current_status = await self.scheduler.get_status(self.task_id)
                        if current_status == 'done':
                            self._enter_phase(WorkflowState.DONE)
                            return WorkflowOutcome.DONE
                        if current_status in WORKFLOW_PRESERVE_STATUSES:
                            logger.info(
                                'Task %s: steward set status to %s during '
                                'escalation resolution — preserving, exiting '
                                'resume loop',
                                self.task_id, current_status,
                            )
                            self._enter_phase(WorkflowState.BLOCKED)
                            return WorkflowOutcome.BLOCKED

                        # Fix 2 — anti-thrash guard for repeated infra-issue
                        # resumes on the same root cause.  Status is confirmed
                        # non-terminal here (Fix 1 guard above), so it's safe
                        # to count this as a real resume attempt.  At
                        # threshold the helper short-circuits to BLOCKED + L1
                        # so a human can intervene rather than the orchestrator
                        # dispatching the implementer/debugger again.
                        thrash_outcome = await self._check_infra_resume_thrash()
                        if thrash_outcome is not None:
                            return thrash_outcome

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
                    #
                    # WT_HEAD INTENTIONALLY OMITTED — see _has_prior_implementation()
                    # docstring.  At this call site we have just run EXECUTE and
                    # any prior rebase already happened; the iteration-log fallback
                    # is the right signal.  The base_commit rebased-head problem
                    # does not apply here because we are checking for the ABSENCE
                    # of implementation work (i.e. a spurious merge signal), and a
                    # freshly-rebased branch that completed EXECUTE will always have
                    # iteration-log entries.
                    if already_merged and not self._has_prior_implementation().has_work:
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
            await self.scheduler.set_task_status(
                self.task_id, 'done',
                done_provenance=(
                    {'kind': 'merged', 'commit': self._merge_sha}
                    if self._merge_sha else None
                ),
            )
            logger.info(
                f'Task {self.task_id} DONE — '
                f'cost=${self.metrics.total_cost_usd:.2f} '
                f'invocations={self.metrics.agent_invocations}'
            )
            return WorkflowOutcome.DONE

        except AllAccountsCappedException as e:
            logger.warning(
                f'Task {self.task_id}: all accounts capped — '
                f'{e.retries} retries in {e.elapsed_secs:.1f}s (label={e.label!r})'
            )
            return await self._mark_blocked(
                f'All accounts capped: {e.label} — {e.retries} retries in {e.elapsed_secs:.1f}s'
            )

        except _SessionBudgetExhausted as e:
            last_role = self._last_completed_role or 'n/a'
            budget_limit = self.config.usage_cap.session_budget_usd
            # Use the gate's own cumulative figure for the summary — it is the
            # value that actually exceeded the budget, whereas
            # self.metrics.total_cost_usd only advances on successful returns
            # and may lag the gate's running tally if a cap-retry or partial
            # invocation contributed cost without completing.
            reason = (
                f'Session budget exhausted: ${e.cumulative_cost:.2f} spent of '
                f'${budget_limit:.2f} budget (last completed role: {last_role})'
            )
            detail = (
                f'budget_limit=${budget_limit:.2f}\n'
                f'total_cost_usd=${self.metrics.total_cost_usd:.2f}\n'
                f'cumulative_cost (gate)=${e.cumulative_cost:.2f}\n'
                f'agent_invocations={self.metrics.agent_invocations}\n'
                f'total_turns={self.metrics.total_turns}\n'
                f'last_completed_role={last_role}'
            )
            # _mark_blocked logs "Task %s BLOCKED: %s" — only log the
            # gate-specific cross-check figure that's unique to this call site.
            logger.info(
                'Task %s: session budget exhausted (gate cumulative $%.2f)',
                self.task_id, e.cumulative_cost,
            )
            return await self._mark_blocked(reason, detail=detail)

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

    def _maybe_warn_missing_escalation(self, role_name: str) -> None:
        """Emit a single WARNING when an escalation-capable role is invoked without a queue."""
        if self._escalation_missing_warned:
            return
        if self.escalation_queue is not None:
            return
        if role_name not in _ESCALATION_CAPABLE_ROLES:
            return
        logger.warning(
            'Task %s: escalation_queue is unavailable — agent role %r would normally'
            ' have escalation tools wired',
            self.task_id, role_name,
        )
        self._escalation_missing_warned = True

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
        rebase_retry_used = False
        for _outer_attempt in range(2):  # at most one rebase-retry round-trip
            for attempt in range(2):
                result = await self._invoke(ARCHITECT, prompt, self.worktree)

                if not result.success:
                    cls = classify_agent_failure(result)
                    logger.error(
                        'Task %s: architect failed (%s): %s',
                        self.task_id, cls.kind.value, cls.summary,
                    )
                    return await self._mark_blocked(
                        f'Planning failed: {cls.summary}',
                        detail=cls.diagnostic_detail,
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

            # Architect task-rejection artifacts.  Both are terminal — handle
            # deterministically before the "no plan.json" failure path so
            # neither interacts with the consecutive_no_plan_failures cycle
            # counter.  Order matters: unactionable_task is the most decisive
            # (jumps straight to L1, bypasses steward); already_done is a
            # clean DONE; blocking_dependency may re-loop the architect.
            if self.artifacts.read_unactionable_task() is not None:
                return await self._handle_unactionable_task_report()
            if self.artifacts.read_already_done() is not None:
                return await self._handle_already_done_report()
            # Fix B: the architect may have written a blocking_dependency
            # report instead of a plan.
            if self.artifacts.read_blocking_dependency() is not None:
                dep_outcome = await self._handle_blocking_dep_report(
                    rebase_retry_used=rebase_retry_used,
                )
                if dep_outcome is not None:
                    return dep_outcome
                # Helper rebased + cleared the artifact; loop back to retry
                # the architect once.
                rebase_retry_used = True
                continue
            break

        assert result is not None  # range(2) always executes at least once
        if not self.plan:
            cls = classify_agent_failure(result)
            logger.error(
                'Task %s: architect produced no plan.json (%s): %s',
                self.task_id, cls.kind.value, cls.summary,
            )
            return await self._handle_no_plan_failure(
                f'Planning failed: no plan.json produced — {cls.summary}',
                detail=(
                    'Architect succeeded but did not write .task/plan.json\n'
                    f'{cls.diagnostic_detail}'
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
            return await self._handle_no_plan_failure(
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
            return await self._handle_no_plan_failure(
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
        return WorkflowOutcome.PLANNED

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

    async def _handle_no_plan_failure(
        self, reason: str, *, detail: str,
    ) -> WorkflowOutcome:
        """Block on a no-plan / malformed-plan failure with cycle detection.

        Fix C — increments ``consecutive_no_plan_failures`` keyed by
        ``last_no_plan_main_sha`` in the task's metadata.  When the
        counter hits ≥ 2 with the same main SHA, the no-plan loop has
        been observed and we escalate to a human directly (skip the
        steward) rather than letting the workflow re-pend.
        """
        try:
            current_main_sha = await self.git_ops.get_main_sha()
        except Exception as exc:  # noqa: BLE001 — fall through to standard path
            logger.warning(
                'Task %s: could not read main SHA for no-plan cycle counter: %s',
                self.task_id, exc,
            )
            current_main_sha = ''

        metadata = self.task.get('metadata') or {}
        if not isinstance(metadata, dict):
            metadata = {}
        last_sha = str(metadata.get('last_no_plan_main_sha') or '')
        try:
            counter = int(metadata.get('consecutive_no_plan_failures') or 0)
        except (TypeError, ValueError):
            counter = 0

        if not current_main_sha or last_sha != current_main_sha:
            counter = 1
        else:
            counter += 1

        # Persist the new counter (best-effort — never block on this).
        new_metadata = dict(metadata)
        new_metadata['last_no_plan_main_sha'] = current_main_sha
        new_metadata['consecutive_no_plan_failures'] = counter
        self.task['metadata'] = new_metadata
        try:
            await self.scheduler.update_task(self.task_id, metadata=new_metadata)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                'Task %s: failed to persist no-plan cycle counter: %s',
                self.task_id, exc,
            )

        if counter >= 2:
            logger.warning(
                'Task %s: consecutive_no_plan_failures=%d on main SHA %s — '
                'no-plan loop confirmed; escalating to human',
                self.task_id, counter, current_main_sha[:12] or '<unknown>',
            )
            full_reason = (
                f'Repeated no-plan failure (counter={counter}) on same '
                f'main SHA: {reason}'
            )
            return await self._mark_blocked(
                full_reason, detail=detail, escalate_to_human=True,
            )

        return await self._mark_blocked(reason, detail=detail)

    async def _check_infra_resume_thrash(self) -> WorkflowOutcome | None:
        """Fix 2 — detect & escalate repeated infra-issue resume thrash.

        Called from the ESCALATED branch of :meth:`run` after the steward
        has resolved the L0 and Fix 1 confirmed the task status is still
        non-terminal (pending / in-progress).  If the most recent resolved
        L0 was an ``infra_issue`` and the iteration log has not grown since
        the previous resume, increment ``consecutive_infra_resume_failures``
        in the task metadata.  At ``max_consecutive_infra_resumes``, route
        to ``_mark_blocked(escalate_to_human=True)`` instead of dispatching
        the implementer again.

        Returns:
            ``WorkflowOutcome.BLOCKED`` when the threshold is hit; ``None``
            to fall through to the existing implementer-resume path.

        The iteration-log growth signal (rather than HEAD SHA) is canonical:
        steward fix-commits and ``--allow-empty`` commits both advance HEAD
        without representing real agent progress.  The iteration log is the
        signal already used by ``_has_prior_implementation``.

        Mirrors :meth:`_handle_no_plan_failure` style — same per-task
        concurrency assumption as the existing
        ``consecutive_no_plan_failures`` writer; no new hazard.
        """
        assert self.artifacts is not None

        metadata = self.task.get('metadata') or {}
        if not isinstance(metadata, dict):
            metadata = {}

        # Determine the category of the most recent resolved L0 (the one
        # the steward just handled).  If no escalation queue is wired up
        # (e.g. eval mode), we cannot classify — fall through.
        recent_category: str | None = None
        if self.escalation_queue:
            resolved = [
                e
                for e in self.escalation_queue.get_by_task(self.task_id)
                if e.level == 0 and e.status == 'resolved'
            ]
            if resolved:
                resolved.sort(
                    key=lambda e: e.resolved_at or e.timestamp, reverse=True,
                )
                recent_category = resolved[0].category

        # Iteration-log entry count is the progress signal.
        iter_entries, _ = self.artifacts.read_iteration_log()
        current_iter_count = len(iter_entries)

        try:
            counter = int(
                metadata.get('consecutive_infra_resume_failures') or 0
            )
        except (TypeError, ValueError):
            counter = 0

        if recent_category == 'infra_issue':
            try:
                last_iter_count = int(
                    metadata.get('last_infra_resume_iteration_count') or 0
                )
            except (TypeError, ValueError):
                last_iter_count = 0
            if current_iter_count > last_iter_count:
                # Steward fix-commits will reset the counter via
                # iteration-log growth.  This is intentional: a steward
                # action is forward progress.
                counter = 1
            else:
                counter += 1
        else:
            # Non-infra category (or no resolved L0 we could classify) —
            # the thrash signal does not apply; reset.
            counter = 0

        new_metadata = dict(metadata)
        new_metadata['consecutive_infra_resume_failures'] = counter
        new_metadata['last_infra_resume_iteration_count'] = current_iter_count
        self.task['metadata'] = new_metadata
        try:
            await self.scheduler.update_task(
                self.task_id, metadata=new_metadata,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort, log and proceed
            logger.warning(
                'Task %s: failed to persist infra-resume thrash counter: %s',
                self.task_id, exc,
            )

        if counter >= self.config.max_consecutive_infra_resumes:
            logger.warning(
                'Task %s: consecutive_infra_resume_failures=%d at threshold '
                '%d — infra-issue thrash confirmed; escalating to human',
                self.task_id, counter,
                self.config.max_consecutive_infra_resumes,
            )
            return await self._mark_blocked(
                f'Repeated infra-issue resume thrash (counter={counter})',
                detail=(
                    f'category={recent_category!r}, '
                    f'iteration_log_entries={current_iter_count}, '
                    f'last_iteration_log_entries='
                    f'{metadata.get("last_infra_resume_iteration_count", 0)}'
                ),
                escalate_to_human=True,
            )

        return None

    async def _handle_blocking_dep_report(
        self, *, rebase_retry_used: bool,
    ) -> WorkflowOutcome | None:
        """Process a ``.task/blocking_dependency.json`` report from the architect.

        Caller has already verified the artifact exists.

        - If the named dep is non-terminal: register the Taskmaster
          dependency, clear the artifact, and return REQUEUED.  Status
          stays ``pending`` — the scheduler's dep-check keeps the task
          from dispatching until the dep is ``done``/``cancelled``.
        - If the dep is already terminal: a race occurred (dep landed
          between architect-start and report).  Rebase onto current
          main, clear the artifact, and return ``None`` so the caller
          retries the architect once.  When ``rebase_retry_used`` is
          already ``True``, return BLOCKED to prevent unbounded retry.
        - If the artifact is malformed (missing ``depends_on_task_id``):
          clear it and return BLOCKED.
        """
        assert self.artifacts is not None and self.worktree is not None
        report = self.artifacts.read_blocking_dependency()
        assert report is not None  # caller must have verified

        dep_id = str(report.get('depends_on_task_id') or '').strip()
        reason = report.get('reason', '')
        if not dep_id:
            logger.error(
                'Task %s: blocking_dependency.json missing depends_on_task_id; '
                'treating as planning failure', self.task_id,
            )
            self.artifacts.clear_blocking_dependency()
            return await self._mark_blocked(
                'Architect wrote malformed blocking_dependency.json '
                '(missing depends_on_task_id)',
                detail=json.dumps(report, indent=2)[:2000],
            )

        dep_status = await self.scheduler.get_status(dep_id)

        if dep_status in TERMINAL_STATUSES:
            # Race: dep landed between architect-start and report.
            self.artifacts.clear_blocking_dependency()
            if rebase_retry_used:
                return await self._mark_blocked(
                    f'Architect repeatedly reported blocking dependency on '
                    f'task {dep_id} which is already {dep_status} '
                    f'(rebase-retry already used)',
                    detail=f'reason: {reason}\nfull_report: '
                           f'{json.dumps(report, indent=2)[:1500]}',
                )
            logger.info(
                'Task %s: architect reported dep on task %s but it is %s — '
                'rebasing onto main and retrying architect once',
                self.task_id, dep_id, dep_status,
            )
            rebased = await self.git_ops.rebase_onto_main(self.worktree)
            new_main_sha = await self.git_ops.get_main_sha()
            if rebased:
                self.artifacts.update_base_commit(new_main_sha)
            else:
                logger.warning(
                    'Task %s: rebase onto main failed during blocking-dep '
                    'recovery — retrying architect on stale base anyway',
                    self.task_id,
                )
            return None

        # Non-terminal dep — register the Taskmaster dependency.
        logger.info(
            'Task %s: architect reported blocking dependency on task %s '
            '(status=%s); registering dep and requeueing — reason: %s',
            self.task_id, dep_id, dep_status, reason[:200],
        )
        try:
            await self.scheduler.dispatch_tool(
                'add_dependency',
                {
                    'id': self.task_id,
                    'depends_on': dep_id,
                    'project_root': str(self.config.project_root),
                },
                timeout=15,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort, log and proceed
            logger.warning(
                'Task %s: add_dependency(%s -> %s) failed: %s — proceeding '
                'with requeue anyway',
                self.task_id, self.task_id, dep_id, exc,
            )

        self.artifacts.clear_blocking_dependency()
        await self.scheduler.set_task_status(self.task_id, 'pending')
        return WorkflowOutcome.REQUEUED

    async def _handle_already_done_report(self) -> WorkflowOutcome:
        """Process a ``.task/already_done.json`` report from the architect.

        Caller has already verified the artifact exists.

        Validation: ``commit`` must be non-empty and reachable from main.
        ``git merge-base --is-ancestor`` returns false for both unknown SHAs
        and SHAs not on main, so this single check covers both.

        On success: set task status to ``done`` with provenance pointing
        at the architect-named commit, return ``DONE``.
        On validation failure: clear the artifact, route to ``_mark_blocked``
        without escalating to a human — this is an architect mistake
        (wrong/missing commit), not an unworkable task.
        """
        assert self.artifacts is not None
        report = self.artifacts.read_already_done()
        assert report is not None  # caller must have verified

        commit = str(report.get('commit') or '').strip()
        evidence = str(report.get('evidence') or '')

        self.artifacts.clear_already_done()

        if not commit:
            return await self._mark_blocked(
                'Architect wrote malformed already_done.json '
                '(missing commit)',
                detail=json.dumps(report, indent=2)[:2000],
            )

        main_sha = await self.git_ops.get_main_sha()
        on_main = await self.git_ops.is_ancestor(commit, main_sha)
        if not on_main:
            return await self._mark_blocked(
                f'Architect reported task already done at {commit[:12]} '
                f'but commit is not reachable from main',
                detail=(
                    f'commit: {commit}\nmain_sha: {main_sha}\n'
                    f'evidence: {evidence}'
                )[:2000],
            )

        logger.info(
            'Task %s: architect reported task already done at %s — '
            'setting status done with provenance',
            self.task_id, commit[:12],
        )
        self._enter_phase(WorkflowState.DONE)
        await self.scheduler.set_task_status(
            self.task_id, 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': commit,
                'note': (
                    f'architect-reported task already on main; '
                    f'evidence: {evidence[:400]}'
                ),
            },
        )
        return WorkflowOutcome.DONE

    async def _handle_unactionable_task_report(self) -> WorkflowOutcome:
        """Process a ``.task/unactionable_task.json`` report from the architect.

        Caller has already verified the artifact exists.

        Stops the steward early to close the small async window where a
        stale L0 from a prior PLAN attempt could be processed concurrently
        with our L1 submission.  The ``finally`` block in ``run()`` also
        stops the steward, so this is defense-in-depth.

        Then short-circuits to ``_mark_blocked(escalate_to_human=True)``,
        which submits an L1 directly without invoking the steward — the
        steward consumes only L0 escalations and cannot fix a broken spec.
        """
        assert self.artifacts is not None
        report = self.artifacts.read_unactionable_task()
        assert report is not None  # caller must have verified

        reason = str(report.get('reason') or '').strip()
        evidence = str(report.get('evidence') or '')

        if self._steward:
            await self._steward.stop()
            self._steward = None

        self.artifacts.clear_unactionable_task()

        if not reason:
            return await self._mark_blocked(
                'Architect wrote malformed unactionable_task.json '
                '(missing reason)',
                detail=json.dumps(report, indent=2)[:2000],
                escalate_to_human=True,
            )

        return await self._mark_blocked(
            f'Architect reported task unactionable: {reason}',
            detail=f'reason: {reason}\nevidence: {evidence}'[:2000],
            escalate_to_human=True,
        )

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
                detail = self._last_verify_result.failure_report() if self._last_verify_result else ''
                return await self._mark_blocked('Verification attempts exhausted', detail=detail)

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

    async def _inter_iteration_rebase(
        self, *, event_label: str = 'rebase',
    ) -> dict | None:
        """Check if main advanced past our base; if so, rebase.

        Returns a dict ``{old_base, new_base, changed_files}`` when a
        rebase was performed, or ``None`` if no rebase was needed or the
        rebase failed (failure is non-blocking — the merge phase will
        handle conflicts).

        ``event_label`` populates the ``event`` field on the
        iteration_log entry so verify-phase calls (Fix 3) can be
        distinguished from execute-phase calls in post-mortem analysis.
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

        # Commit any uncommitted work before rebasing.  ``commit()`` no-ops
        # on a clean tree so verify-phase callers (which always run on a
        # clean tree post-execute) do not produce empty WIP commits.
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
            'event': event_label,
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
            # Fix 3: rebase onto main BEFORE each verify (including the first).
            # Closes the verify-only-retry rebase gap: when main advances
            # mid-task (e.g. a sibling task fixes the env collision the
            # verify is failing on), the existing _inter_iteration_rebase
            # only fires from the EXECUTE loop — it cannot pick up new main
            # commits while we're cycling verify ↔ debugger.  The helper
            # short-circuits cheaply when current_main == old_base, so
            # firing on every retry costs at most one ``git rev-parse``.
            if self.config.rebase_before_verify:
                await self._inter_iteration_rebase(
                    event_label='verify_phase_rebase',
                )

            result = await run_scoped_verification(
                self.worktree, self.config, self._module_configs, task_files=self._task_files,
                attempt_id=verify_attempt + 1,
                task_id=self.task_id,
                archive_root=self.config.project_root / 'data' / 'verify-logs',
            )
            if not result.passed:
                self._last_verify_result = result
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
        from orchestrator.merge_queue import MergeOutcome, MergeRequest, enqueue_merge_request

        assert self.worktree is not None
        assert self.merge_queue is not None

        future: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
        merge_request = MergeRequest(
            task_id=self.task_id,
            branch=branch_name,
            worktree=self.worktree,
            pre_rebased=pre_rebased,
            task_files=self._task_files,
            module_configs=self._module_configs,
            config=self.config,
            result=future,
        )
        await enqueue_merge_request(self.merge_queue, merge_request, self.event_store)

        # Race the future against the cancel event so a human marking the
        # task done out-of-band exits the workflow promptly instead of
        # waiting for the merge worker to finish.
        result = await self._await_cancellable(future)
        if result is None:
            return await self._handle_soft_cancel('merge')

        if result.status == 'wip_halted':
            return await self._handle_wip_conflict(result, branch_name)
        if result.status == 'done_wip_recovery':
            return await self._handle_wip_recovery(result)
        if result.status == 'wip_recovery_no_advance':
            return await self._handle_wip_recovery_no_advance(result)
        if result.status == 'unmerged_state':
            return await self._handle_unmerged_state(result, branch_name)
        if result.status == 'done':
            if result.merge_sha is not None:
                self._merge_sha = result.merge_sha
            return WorkflowOutcome.DONE
        if result.status == 'already_merged':
            logger.info(f'Task {self.task_id}: already merged to main')
            return WorkflowOutcome.DONE
        if result.status == 'conflict':
            return await self._resolve_and_resubmit(
                branch_name, result.conflict_details,
                merge_phase=merge_phase,
            )
        # ``blocked`` — but first check for the worktree-missing race: if a
        # human marked the task ``done`` and removed the worktree while the
        # merge was in flight, the merge worker surfaces a known reason
        # prefix.  Re-read task status; if terminal, exit cleanly without
        # creating an escalation.
        from orchestrator.merge_queue import WORKTREE_MISSING_REASON_PREFIX
        if result.reason.startswith(WORKTREE_MISSING_REASON_PREFIX):
            try:
                status = await self.scheduler.get_status(self.task_id)
            except Exception:
                logger.exception(
                    f'Task {self.task_id}: get_status failed during '
                    f'worktree-missing fallback; falling through to blocked'
                )
                status = None
            if status in TERMINAL_STATUSES:
                logger.info(
                    f'Task {self.task_id}: worktree missing but task '
                    f'status={status!r} (terminal) — exiting DONE without '
                    f'escalation'
                )
                return WorkflowOutcome.DONE
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

        # Warn once per workflow instance when an escalation-capable role is
        # dispatched without an escalation queue wired up.
        self._maybe_warn_missing_escalation(role.name)

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
            invoke_fn=invoke_agent,
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

        # Record the last successfully-completed role (updated only on success,
        # mirrors the cost-accumulation path below — failed/raised invocations
        # do not advance either field).
        self._last_completed_role = role.name

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
        if self.escalation_queue.has_open_l1(self.task_id):
            pending_l1 = self.escalation_queue.get_by_task(
                self.task_id, status='pending', level=1,
            )
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

    async def _check_branch_on_main(self) -> tuple[str, str] | None:
        """Probe whether the worktree HEAD is reachable from main.

        Returns ``(wt_head, main_sha)`` when ``git merge-base --is-ancestor
        wt_head main_sha`` succeeds (i.e. the branch has been merged to main
        or the HEAD is exactly main).  Returns ``None`` in three cases:

        1. ``self.worktree`` or ``self.git_ops`` is None — partially-wired
           workflow; callers that reach this state should treat the branch as
           not-on-main.
        2. HEAD is not an ancestor of main — branch has unmerged commits.
        3. Any of the above when combined with the caller's own guard logic.

        Does NOT catch subprocess or git exceptions — callers wrap as needed.
        ``_recover_if_already_merged`` wraps the call in ``try/except`` and
        logs ``'merge-check failed'`` before returning None; the pre-EXECUTE
        ghost-loop guard in ``workflow.run()`` lets exceptions propagate.  The
        divergent downstream logic at each call site is intentional — do not
        collapse them.

        See also: ``_recover_if_already_merged`` (pre-PLAN guard) and the
        ghost-loop guard around ``workflow.py:431`` (pre-EXECUTE guard).
        """
        if self.worktree is None or self.git_ops is None:
            return None
        wt_head = await self._get_head_commit()
        main_sha = await self.git_ops.get_main_sha()
        if await self.git_ops.is_ancestor(wt_head, main_sha):
            return (wt_head, main_sha)
        return None

    def _has_prior_implementation(self, wt_head: str | None = None) -> _PriorImplStatus:
        """Check whether a prior run did any implementation in this worktree.

        When *wt_head* is provided (the post-execution branch HEAD), the primary
        signal is a SHA comparison: the branch has advanced past its starting
        point iff ``wt_head.strip() != base_commit``.  This is invariant to
        iteration-log format changes and avoids the false-done regression where
        a stale iteration entry triggers the guard on a branch with no commits.

        When *wt_head* is not provided, falls back to scanning
        .task/iterations.jsonl for implementer/debugger entries.
        Planning-only runs don't write these, so absence means stale branch
        point rather than a legitimately merged prior run.  This fallback is
        also used by the ghost-loop guard and the merge-phase guard, where a
        post-rebase HEAD may coincide with base_commit even on a genuinely-
        implemented branch.

        When *wt_head* IS provided but base_commit is absent (metadata.json
        not yet stamped), returns ``has_work=False`` — this is the
        fail-closed safety net for SHA-primary callers.  Refusing to fall
        back to the iteration-log scan prevents false-DONE from inherited
        .task/iterations.jsonl contamination.  See
        test_returns_none_when_wt_head_provided_but_metadata_missing for the
        regression case.

        Returns a :class:`_PriorImplStatus` NamedTuple with ``has_work``,
        ``entries`` (full iteration-log list — callers can use ``len(entries)``
        in warning breadcrumbs without a second ``read_iteration_log()`` call),
        and ``base_commit`` (may be ``None`` if metadata.json is absent).

        Correctness invariants: the iteration-log fallback relies on
        .task/iterations.jsonl entries faithfully reflecting prior work on the
        *same branch*.  Two scenarios matter:

        *Intended* — ghost-loop re-run on the same branch: create_worktree
        may rebase a reused worktree onto main, so wt_head == base_commit even
        though the branch was genuinely implemented.  The fallback correctly
        returns True here because the earlier implementer run wrote its entries
        before the rebase.  This is the scenario exploited by the pre-EXECUTE
        guard and the pre-MERGE guard; it is safe for those callers.

        *Dangerous* — orphaned log: if iterations.jsonl were somehow copied
        from a different task's branch or inherited from main contamination,
        the fallback would return True for an empty branch → false-done.
        This is why callers that hold a reliable wt_head must pass it
        explicitly.  ``_recover_if_already_merged()`` passes wt_head to
        use the SHA-primary path, preventing false-DONE on inherited
        .task/iterations.jsonl contamination — see the comment there for
        the full trade-off analysis.
        """
        if self.artifacts is None:
            return _PriorImplStatus(has_work=False, entries=[], base_commit=None)
        base_commit = self.artifacts.read_base_commit()
        entries, _ = self.artifacts.read_iteration_log()
        if wt_head is not None and base_commit is not None:
            sha_diverges = wt_head.strip() != base_commit
            has_iter_log_work = any(
                e.get('agent') in ('implementer', 'debugger') for e in entries
            )
            # Defense in depth: SHA divergence alone is racy under
            # fused-memory's tasks.json auto-commit to main (the
            # pre-positioning rev-parse in create_worktree could lag
            # the actual worktree fork point).  Iteration-log evidence
            # alone is racy under inherited orphan logs.  Require BOTH
            # signals before declaring prior implementation work.  See
            # the audit notes in
            # ~/.claude/plans/do-2-3-misty-marshmallow.md and the
            # orphan-log scenario in this method's docstring.
            return _PriorImplStatus(
                has_work=sha_diverges and has_iter_log_work,
                entries=entries,
                base_commit=base_commit,
            )
        if wt_head is not None:
            # Fail-closed: caller signaled SHA-primary semantics by passing
            # wt_head, but base_commit is absent (metadata.json not stamped).
            # Refuse to fall back to the iteration-log scan — on a worktree
            # with inherited .task/iterations.jsonl contamination this would
            # still produce has_work=True and false-DONE.  See
            # test_returns_none_when_wt_head_provided_but_metadata_missing
            # for the regression case.
            return _PriorImplStatus(has_work=False, entries=entries, base_commit=None)
        # Fallback (no wt_head): iteration-log scan for pre-EXECUTE / merge-phase guards
        return _PriorImplStatus(
            has_work=any(e.get('agent') in ('implementer', 'debugger') for e in entries),
            entries=entries,
            base_commit=base_commit,
        )

    async def _recover_if_already_merged(self) -> WorkflowOutcome | None:
        """Check if the task's branch is already on main and transition to DONE.

        Called pre-PLAN to short-circuit ghost-loop re-runs: if a prior workflow
        run merged the branch but failed before writing DONE status, this guard
        detects the merged branch and immediately marks the task done.

        Returns WorkflowOutcome.DONE if the branch is already merged to main AND
        there is prior implementation work.  Returns None in all other cases
        (branch not merged, no prior work, missing worktree/git_ops, exceptions).
        """
        # Intentional double-check: _check_branch_on_main() has its own
        # None-guard and would return None silently, but this outer check lets
        # us emit the 'skipping merge-recovery' DEBUG log so the missing-wiring
        # condition is observable at the call-site level.
        if self.worktree is None or self.git_ops is None:
            logger.debug(
                'Task %s: skipping merge-recovery (no worktree or git_ops)',
                self.task_id,
            )
            return None

        # ── Git layer ─────────────────────────────────────
        # Delegates to _check_branch_on_main() which can fail for git/infra
        # reasons (e.g. corrupted index, network mount offline).
        # Returns (wt_head, main_sha) when the branch is on main, else None.
        try:
            _git_check = await self._check_branch_on_main()
        except Exception:
            logger.warning(
                'Task %s: merge-check failed, proceeding with normal workflow',
                self.task_id, exc_info=True,
            )
            return None

        if _git_check is None:
            return None

        wt_head, main_sha = _git_check

        # ── Artifacts layer ────────────────────────────────
        # Reads that can fail for filesystem/JSON reasons (e.g. corrupted
        # iterations.jsonl, missing metadata).  Wrapped in a SEPARATE
        # try/except from the git layer so operators can distinguish the
        # root cause from the log message.
        #
        # We pass wt_head to _has_prior_implementation() so the SHA-primary
        # path is taken: has_work = (wt_head != base_commit).  This prevents
        # false-DONE when a fresh worktree (wt_head == base_commit, no real
        # commits) has inherited an on-disk .task/iterations.jsonl written
        # directly to the worktree's .task/ directory (e.g. main-branch
        # contamination where the file exists on disk but is untracked by git).
        # Without wt_head the iteration-log fallback finds the implementer entry
        # and incorrectly returns DONE for an unimplemented task — catastrophic
        # silent failure.  See test_returns_none_for_inherited_iterations_log_on_fresh_worktree
        # for the regression case this guards.
        #
        # Additional safety net: if metadata.json was never stamped
        # (base_commit is None), _has_prior_implementation() now returns
        # has_work=False from its fail-closed branch rather than falling back
        # to the iteration-log scan.  This covers edge cases where
        # artifacts.init() was not called before this guard (e.g. eval-mode
        # paths or future refactors that re-order setup).  See
        # test_returns_none_when_wt_head_provided_but_metadata_missing.
        #
        # Trade-off: if create_worktree rebased a genuinely-implemented branch
        # onto a new main tip so that wt_head == new_base_commit, the SHA-primary
        # check returns has_work=False and this guard returns None (the workflow
        # proceeds to PLAN).  The pre-EXECUTE guard at workflow.py:412-457 still
        # uses the iteration-log fallback and will catch the rebased ghost-loop
        # before EXECUTE, routing the workflow to the SUCCESS path.  Only one
        # architect invocation is wasted — bounded cost, far preferable to a
        # silent false-DONE that marks an unimplemented task complete with no
        # code written.  See _has_prior_implementation() for SHA-primary vs.
        # fallback semantics.
        try:
            status = self._has_prior_implementation(wt_head=wt_head)
            if not status.has_work:
                logger.warning(
                    'Task %s: branch HEAD %s is ancestor '
                    'of main %s but no implementation '
                    'entries (base=%s, entries=%d) — '
                    'proceeding with normal workflow',
                    self.task_id,
                    wt_head[:8],
                    main_sha[:8],
                    status.base_commit[:8] if status.base_commit else 'none',
                    len(status.entries),
                )
                return None
        except Exception:
            logger.warning(
                'Task %s: artifacts read failed during merge-check, '
                'proceeding with normal workflow',
                self.task_id, exc_info=True,
            )
            return None

        logger.info(
            'Task %s: branch already on main — completing instead of re-queueing',
            self.task_id,
        )
        self._enter_phase(WorkflowState.DONE)
        await self.scheduler.set_task_status(
            self.task_id, 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': main_sha,
                'note': (
                    'branch already on main at workflow start '
                    '(pre-PLAN recovery)'
                ),
            },
        )
        return WorkflowOutcome.DONE

    def _escalate_plan_overwrite(self) -> None:
        """Submit a blocking escalation when plan.json ownership doesn't match.

        Distinguishes two cases by reading the current _session_id:

        - **Empty/missing**: the architect failed before stamping provenance.
          This is usually a downstream effect of a planning-phase failure
          (e.g. 403/cap hit with no retry), not a duplicate workflow.
        - **Non-empty but different**: a genuine foreign session wrote plan.json.
        """
        summary = f'plan.json overwrite detected for task {self.task_id}'
        foreign_session = ''
        if self.artifacts is not None:
            try:
                plan_path = self.artifacts.root / 'plan.json'
                data = json.loads(plan_path.read_text())
                foreign_session = data.get('_session_id') or ''
            except Exception:
                pass

        if not foreign_session:
            detail = (
                f'plan.json is not stamped with the current session '
                f'(expected _session_id={self.session_id}). Most likely the '
                f'architect failed before stamping — a downstream effect of a '
                f'planning-phase failure, not a duplicate workflow.'
            )
        else:
            detail = (
                f'Expected _session_id={self.session_id} but plan.json contains '
                f'{foreign_session}. A duplicate workflow may have overwritten plan.json.'
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
        escalate_to_human: bool = False,
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
        *escalate_to_human* (Fix C) skips the steward entirely and submits
        an L1 escalation immediately.  Use when the caller has determined
        a confirmed loop / unresolvable failure that the steward cannot
        meaningfully un-stick (e.g. ≥2 consecutive no-plan failures on
        the same main SHA).
        """
        if self.state == WorkflowState.DONE:
            logger.warning(
                'Task %s: already DONE, ignoring late blocked transition: %s',
                self.task_id, reason,
            )
            return WorkflowOutcome.DONE
        # Capture the phase we were in before transitioning to BLOCKED so the
        # harness-level retry cap can report *which* phase looped.  _enter_phase
        # overwrites self.state, so stash first.
        self._last_block_phase = self.state.value
        self._last_block_reason = reason
        self._last_block_detail = detail or reason
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

        created_l0_id: str | None = None
        if self.escalation_queue and not skip_escalation:
            # Don't create a duplicate if level-1 already pending
            if not self.escalation_queue.has_open_l1(self.task_id):
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
                created_l0_id = esc.id

                if self.event_store:
                    self.event_store.emit(
                        EventType.escalation_created,
                        task_id=self.task_id, phase=self.state.value,
                        data={'escalation_id': esc.id, 'category': 'task_failure',
                              'severity': 'blocking', 'summary': reason[:200]},
                    )

            # Fix C short-circuit: the caller already determined this is
            # a confirmed loop / unresolvable failure that the steward
            # cannot un-stick.  Skip steward, submit L1, return BLOCKED.
            if escalate_to_human:
                await self._ensure_l1_escalation_for_blocked(
                    reason, detail or reason,
                )
                return WorkflowOutcome.BLOCKED

            # Give the steward a chance to resolve the escalation
            await self._ensure_steward_started()
            if self._steward:
                await self._await_steward_completion()

                # Single fresh read of the store — replaces the old cached
                # scheduler snapshots. Server-side terminal guard rejects
                # done→pending, but we still need the correct workflow
                # outcome.
                current = await self.scheduler.get_status(self.task_id)
                if current in TERMINAL_STATUSES:
                    logger.info(
                        'Task %s: status is %s after steward — not re-queueing',
                        self.task_id, current,
                    )
                    if current == 'done':
                        self._enter_phase(WorkflowState.DONE)
                        return WorkflowOutcome.DONE
                    # 'cancelled' is an intentional terminal — no L1 needed.
                    return WorkflowOutcome.BLOCKED

                # If steward resolved all level-0 escalations, set task back
                # to pending so the scheduler re-picks it on the next cycle.
                remaining = self.escalation_queue.get_by_task(
                    self.task_id, status='pending', level=0,
                )
                if not remaining:
                    # Guard: if the steward escalated to L1 (human-only),
                    # leave the task's status untouched and exit.  L0-empty
                    # alone does not mean "all clear" — an open L1 signals
                    # that the steward handed off.
                    if self.escalation_queue.has_open_l1(self.task_id):
                        logger.info(
                            'Task %s: L1 escalation open — steward handed '
                            'off to human; leaving status as-is and exiting',
                            self.task_id,
                        )
                        return WorkflowOutcome.ESCALATED

                    # Preserve steward-set deferred. Terminal statuses (done,
                    # cancelled) were caught earlier via ``current``; blocked
                    # intentionally falls through to requeue because the
                    # orchestrator's own _mark_blocked wrote it and the steward
                    # leaving it alone is indistinguishable from re-asserting
                    # it. 'deferred' is the one case the steward chooses
                    # explicitly that we must not overwrite.
                    if current == 'deferred':
                        logger.info(
                            'Task %s: steward set status to deferred — '
                            'preserving, skipping auto-requeue',
                            self.task_id,
                        )
                        return WorkflowOutcome.BLOCKED

                    # Fix A: detect dismiss-with-terminate.  When the
                    # steward's resolve_issue(terminate=True) marks the L0
                    # 'dismissed' (rather than 'resolved'), the agent
                    # signaled "I cannot fix this" — re-pending will loop
                    # the same failure.  Halt here, submit an L1 so a
                    # human can intervene, and return BLOCKED.
                    if created_l0_id is not None:
                        last_l0 = self.escalation_queue.get(created_l0_id)
                        if (
                            last_l0 is not None
                            and last_l0.status == 'dismissed'
                        ):
                            logger.warning(
                                'Task %s: steward dismissed L0 (terminate) '
                                '— halting, escalating to L1', self.task_id,
                            )
                            await self._ensure_l1_escalation_for_blocked(
                                reason, detail or reason,
                            )
                            return WorkflowOutcome.BLOCKED

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

        # Fall-through BLOCKED: either no escalation queue, or the steward
        # never resolved the L0.  Either way a human should know — submit
        # an L1 (deduped) so the task isn't silently parked.
        await self._ensure_l1_escalation_for_blocked(reason, detail or reason)
        return WorkflowOutcome.BLOCKED

    async def _ensure_l1_escalation_for_blocked(
        self, reason: str, detail: str,
    ) -> None:
        """Submit a level-1 escalation if none is open for this task.

        Called from BLOCKED-return paths so a human is signaled when
        automated handlers cannot make progress.  Idempotent — deduped
        via ``has_open_l1``.
        """
        if not self.escalation_queue:
            return
        if self.escalation_queue.has_open_l1(self.task_id):
            return
        from escalation.models import Escalation

        esc = Escalation(
            id=self.escalation_queue.make_id(self.task_id),
            task_id=self.task_id,
            agent_role='orchestrator',
            severity='blocking',
            category='task_failure',
            summary=f'Workflow blocked, no automated resolution path: {reason[:160]}',
            detail=detail or reason,
            suggested_action='manual_intervention',
            worktree=str(self.worktree) if self.worktree else None,
            workflow_state=self.state.value,
            level=1,
        )
        self.escalation_queue.submit(esc)
        if self.event_store:
            self.event_store.emit(
                EventType.escalation_created,
                task_id=self.task_id, phase=self.state.value,
                data={
                    'escalation_id': esc.id, 'category': 'task_failure',
                    'severity': 'blocking', 'level': 1,
                    'summary': reason[:200],
                },
            )
        logger.warning(
            'Task %s: submitted L1 escalation %s for unresolved BLOCKED state',
            self.task_id, esc.id,
        )

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

    async def _await_cancellable(self, awaitable):
        """Race ``awaitable`` against ``self._cancel_event``.

        Returns the awaitable's result, or ``None`` if the cancel event was
        set first.  When ``None`` is returned the caller should look up the
        scheduler's truth and decide between DONE / cancelled / normal-blocked
        via :meth:`_handle_soft_cancel`.

        If both the awaitable and the cancel event resolve in the same
        ``asyncio.wait`` window, the awaitable's result wins — the work
        already finished, no need to soft-cancel.
        """
        fut = asyncio.ensure_future(awaitable)
        cancel_task = asyncio.create_task(self._cancel_event.wait())
        try:
            done, _pending = await asyncio.wait(
                {fut, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if fut in done:
                return fut.result()
            return None
        finally:
            if not cancel_task.done():
                cancel_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await cancel_task

    async def _handle_soft_cancel(self, phase: str) -> WorkflowOutcome:
        """Decide an outcome after ``_cancel_event`` interrupted a long wait.

        Re-reads the scheduler's view of task status: if terminal, exit
        ``DONE`` (typically a human marked the task done); if not terminal,
        the cancel was likely spurious (or the workflow should be requeued)
        — fall back to ``REQUEUED`` so the harness re-runs the slot once
        the cancel condition clears.
        """
        try:
            status = await self.scheduler.get_status(self.task_id)
        except Exception:
            logger.exception(
                f'Task {self.task_id}: get_status failed during soft-cancel'
            )
            status = None
        logger.info(
            f'Task {self.task_id}: soft-cancel during {phase} — '
            f'scheduler status={status!r}'
        )
        if status in TERMINAL_STATUSES:
            return WorkflowOutcome.DONE
        return WorkflowOutcome.REQUEUED

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

            # Soft-cancel takes precedence over the steward grace period.
            if self._cancel_event.is_set():
                logger.info(
                    f'Task {self.task_id}: cancel-event set during steward grace — '
                    f'skipping remaining wait'
                )
                return

            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break

            self._escalation_event.clear()
            esc_wait = asyncio.create_task(self._escalation_event.wait())
            cancel_wait = asyncio.create_task(self._cancel_event.wait())
            try:
                done, _pending = await asyncio.wait(
                    {esc_wait, cancel_wait},
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                for t in (esc_wait, cancel_wait):
                    if not t.done():
                        t.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await t
            if not done:
                break  # timeout — fall through to re-escalation
            if cancel_wait in done:
                logger.info(
                    f'Task {self.task_id}: cancel-event fired during steward grace — '
                    f'exiting completion wait'
                )
                return

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
