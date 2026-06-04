"""Per-task workflow state machine: PLAN → EXECUTE → VERIFY → REVIEW → MERGE → DONE."""

from __future__ import annotations

import asyncio
import contextlib
import enum
import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, cast

# Runtime import of the BORN_AT_L2_SEVERITIES constant from escalation.models.
# escalation.models is listed under TYPE_CHECKING above (:77-78) for the Escalation
# type annotation; a separate runtime import is needed here because
# _is_gating_escalation evaluates the set at call time, not just for type hints.
# This is cycle-safe: escalation/src/escalation/models.py imports only stdlib and
# nothing in it imports orchestrator.
from escalation.models import BORN_AT_L2_SEVERITIES
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
    SIMPLE_TASK,
    AgentRole,
)
from orchestrator.artifacts import PLAN_SCHEMA_VERSION, TaskArtifacts
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.dry_run_unblock import run_dry_run_unblock
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, TrainMembership, _run
from orchestrator.scheduler import (
    SetTaskStatusRejected,
    TaskAssignment,
    TerminalExitRejection,
    files_to_modules,
    normalize_lock,
)
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


def _is_gating_escalation(e: Escalation) -> bool:
    """Return True if *e* should gate workflow progress (PRD C7 / decisions D4, D8).

    Gating policy — an escalation gates when ANY disjunct fires:
    1. Plain blocking L0: ``severity == 'blocking' and level == 0``
       (fresh agent-filed blocker awaiting steward triage).
    2. Born-at-L2 severity: ``severity in BORN_AT_L2_SEVERITIES``
       i.e. 'critical' or 'urgent' — these bypass the auto-watcher and route
       straight to a human regardless of level.
    3. Any level ≥ 2: ``level >= 2``
       (escalation-watcher promoted to L2; a human must decide before the
       workflow proceeds — applies to any severity, including 'info').

    Deliberately NOT gating:
    - Plain blocking L1 (``severity == 'blocking' and level == 1``): this is a
      steward hand-off from a *prior* run that is still awaiting human triage.
      Sinking the current run on a stale L1 caused the run-2 false-blocked
      outcome (esc-2911-22).  The B3 constraint preserves this property; note
      that a *prior-run* critical/urgent (``severity in BORN_AT_L2_SEVERITIES``)
      DOES gate the current run — that is the intended stop-the-line semantics
      for high-severity escalations (D4 accepted consequence).
    """
    return (
        (e.severity == 'blocking' and e.level == 0)
        or e.severity in BORN_AT_L2_SEVERITIES
        or e.level >= 2
    )


class _StewardReescalated(Exception):
    """Raised when the steward re-escalates to level-1 (consumed by the auto-watcher, which may promote to L2 for a human)."""

    def __init__(self, escalations):
        self.escalations = escalations

if TYPE_CHECKING:
    from escalation.models import Escalation, TrainState
    from escalation.queue import EscalationQueue

    from orchestrator.merge_queue import InFlightMergeRegistry
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
    async def mark_done(
        self,
        task_id: str,
        /,
        *,
        kind: str,
        sha: str,
        note: str | None = ...,
    ) -> None: ...
    async def handle_blast_radius_expansion(
        self, task_id: str, current: list[str], needed: list[str], /
    ) -> bool: ...
    async def get_status(self, task_id: str, /) -> str | None: ...
    async def get_task(self, task_id: str, /) -> dict | None: ...
    async def update_task(
        self, task_id: str, metadata: str | dict, *, append: bool = ...,
    ) -> bool: ...
    async def dispatch_tool(
        self, name: str, arguments: dict, *, timeout: float = ...,
    ) -> dict: ...
    async def get_tasks(self) -> list[dict]: ...
    async def get_statuses(
        self, ids: list[str] | None = ...,
    ) -> tuple[dict[str, str], Exception | None]: ...
    async def tasks_by_train(self, train_id: str, /) -> list[dict]: ...
    def clear_requeue_count(self, task_id: str, /) -> None: ...


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
    async def build_plan_completion_prompt(
        self, task: dict, partial_plan: dict,
        worktree: Path | None = ..., context: str | None = ...,
    ) -> str: ...
    async def build_plan_tightening_prompt(
        self, task: dict, plan: dict, not_touched: list[str],
        worktree: Path | None = ..., context: str | None = ...,
    ) -> str: ...
    async def build_simple_task_prompt(
        self, task: dict, worktree: Path | None = ...,
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
    MERGE_DEFERRED = 'merge-deferred'
    DONE = 'done'
    BLOCKED = 'blocked'
    ESCALATED = 'escalated'


class WorkflowOutcome(enum.Enum):
    DONE = 'done'
    PLANNED = 'planned'
    BLOCKED = 'blocked'
    REQUEUED = 'requeued'
    ESCALATED = 'escalated'
    CANCELLED = 'cancelled'
    MERGE_DEFERRED = 'merge-deferred'


# Matches the wrapper string ``_run_cmd`` injects when its own asyncio.wait_for
# fires.  When this is the only cause hint there is no actionable signal for
# the debugger; the verify-retry loop short-circuits to BLOCKED instead of
# burning ``max_verify_attempts × verify_command_timeout_secs``.  After the
# streamed-stdout fix in ``_run_cmd`` (Change 2) a real cause hint should
# surface on attempt 1 for any genuine in-test hang.
_OPAQUE_TIMEOUT_CAUSE_RE = re.compile(r'^Command timed out after \d+(\.\d+)?s:')

# Regexes used by ``_normalize_cause_hint`` — compiled once at module level.
# Order of application: ANSI first (so coloured file:line refs are cleaned
# before the file:line pattern matches them), then file:line, then whitespace.
_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*m')
_FILE_LINE_RE = re.compile(
    r'\b[\w./\\-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|cpp|c|h|sh|md|yaml|yml|json|toml)'
    r':\d+(:\d+)?\b'
)
_WHITESPACE_RE = re.compile(r'\s+')


def _normalize_cause_hint(hint: str | None) -> str:
    """Normalise a VerifyResult cause_hint for equality comparison.

    Strips ANSI colour escape sequences, removes file:line (and file:line:col)
    numeric tails, collapses contiguous whitespace to a single space,
    lowercases, and strips leading/trailing whitespace.

    Returns an empty string for empty or None input — never raises.

    Used by the verify-loop signature-repetition guard to detect consecutive
    identical failures even when line numbers shift between retries.
    """
    if not hint:
        return ''
    # 1. Strip ANSI colour codes (e.g. \x1b[31m...\x1b[0m) first so that
    #    coloured file:line references like \x1b[31mfoo.py:42\x1b[0m become
    #    plain foo.py:42 before the file:line pattern runs.
    result = _ANSI_ESCAPE_RE.sub('', hint)
    # 2. Strip file:line and file:line:col numeric tails
    #    (e.g. "tests/test_x.py:42" or "foo.py:42:7").
    result = _FILE_LINE_RE.sub('', result)
    # 3. Collapse contiguous whitespace (spaces, tabs, newlines) to one space.
    result = _WHITESPACE_RE.sub(' ', result)
    # 4. Lowercase and strip.
    return result.lower().strip()


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
        mcp: _McpLike | None,
        escalation_queue: EscalationQueue | None = None,
        escalation_event: asyncio.Event | None = None,
        usage_gate: UsageGate | None = None,
        initial_plan: dict | None = None,
        steward_factory=None,
        merge_queue: asyncio.Queue | None = None,
        merge_worker=None,
        merge_inflight_registry: InFlightMergeRegistry | None = None,
        event_store: EventStore | None = None,
        cost_store: CostStore | None = None,
        cancel_event: asyncio.Event | None = None,
        resume_session_id: dict | None = None,
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
        self.merge_inflight_registry = merge_inflight_registry
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
        # Base commit for the current run's worktree (set in run() right after
        # create_worktree).  Captured here so _reconcile_metadata_files_for_done
        # can diff base..merge_sha and write the actually-changed paths instead
        # of the architect's plan.files (which the merge may have squashed).
        self._base_commit: str | None = None
        self._merge_sha: str | None = None  # merge commit SHA set by _submit_to_merge_queue on success
        self._last_completed_role: str | None = None  # role of the last successfully-completed invocation
        self._last_verify_result: VerifyResult | None = None  # most recent failing VerifyResult from _verify_debugfix_loop
        # Per-run history of (category, normalised cause_hint) tuples for the
        # signature-repetition guard.  Ephemeral — intentionally not persisted
        # in task metadata because the verify loop is wholly within one
        # workflow run (unlike _check_infra_resume_thrash which crosses runs).
        self._failure_signature_history: list[tuple[str, str]] = []
        # Block-reason surfacing for the harness-level retry cap.  Populated
        # when _mark_blocked takes the REQUEUED return path; the harness reads
        # these after workflow.run() returns to decide whether to increment
        # the per-task requeue counter.
        self._last_block_reason: str = ''
        self._last_block_detail: str = ''
        self._last_block_phase: str = ''
        # Last blocked-from-merge-queue reason — captured by
        # _submit_to_merge_queue and consumed by the merge-phase thrash
        # check (Fix 3).  Cleared between merge attempts so a stale
        # reason from an earlier task slot can't poison the signature.
        self._last_merge_block_reason: str | None = None

        # Background asyncio.Tasks spawned by _spawn_dry_run_unblock.
        # Holding a strong reference prevents the event loop's weak-ref GC
        # from destroying long-running investigations before they complete.
        self._background_tasks: set = set()

        # In-task dedup cache for _route_review_suggestions_to_curator.
        # Stores the content_hash of the *most recently* routed suggestion
        # batch (scalar, not a set).  The escalation→resume cycle can re-enter
        # REVIEW→DONE with identical suggestions; this sentinel short-circuits
        # consecutive identical re-entries so the curator R4 gate never has to
        # absorb redundant N-HTTP batches.
        #
        # Boundary: sequence A→B→A will re-submit A on the third call because
        # B overwrites the cache entry.  This is intentional — the scalar
        # fast-path only eliminates *consecutive* duplicates.  The server-side
        # curator R4 idempotency gate (task_interceptor._check_escalation_idempotency)
        # is the durable source-of-truth dedup for non-consecutive repeats.
        self._last_routed_suggestion_hash: str | None = None

        # One-shot guard for the architect plan-tightening retry.  Set
        # True on first _try_narrow_plan call regardless of outcome so
        # the workflow never loops the narrowing pass on the same task.
        self._plan_tightened: bool = False

        # Crash-recovery resume: when the harness detected a surviving
        # agent-session sidecar at startup, it injects the parsed dict
        # here.  The next _invoke whose role matches will use --resume
        # against the original Claude session and a "continue" prompt
        # instead of spawning a fresh agent.  Consumed (set to None) on
        # first use so subsequent invocations are fresh.
        self._pending_resume_session_id: str | None = None
        self._pending_resume_role: str | None = None
        if resume_session_id:
            self._pending_resume_session_id = resume_session_id.get('session_id')
            self._pending_resume_role = resume_session_id.get('role')

    @property
    def _task_files(self) -> list[str] | None:
        """Return the file list from the current plan, or None if unavailable/empty."""
        files = self.plan.get('files', [])
        return files if files else None

    @property
    def _train(self) -> TrainMembership | None:
        """Return the train membership dict if this task is a train member, else None.

        Reads ``task.metadata.train`` using the same dict-or-None shape that
        git_ops and scheduler already consume (β₂/β₁).  A non-dict value
        (e.g. a stale string) is treated as absent so callers get a clean
        ``is not None`` check.
        """
        train = (self.task.get('metadata') or {}).get('train')
        return cast(TrainMembership, train) if isinstance(train, dict) else None

    async def _enter_merge_deferred(self) -> WorkflowOutcome:
        """Park this train member in the merge-deferred holding state (γ₁, PRD §9.5).

        Called after the full execute→verify→review pipeline succeeds for a
        train member.  Instead of entering the merge phase, the task transitions
        to ``status='merge-deferred'`` and waits for the group-merge worker (δ₁)
        to drive the eventual done transition once all siblings are ready.

        After writing merge-deferred, invokes ``_maybe_enqueue_group_merge()``
        (δ₂).  When the trigger fires (all members deferred, self is tip), that
        method awaits the GroupMergeRequest and returns the final outcome
        directly.  When the train is incomplete (or self is not the tip), the
        trigger returns ``None`` and this method returns ``MERGE_DEFERRED`` to
        park the workflow.

        The worktree is NOT cleaned up here — the merge-queue worker's
        post-merge ``cleanup_worktree`` never runs for merge-deferred tasks,
        so the worktree is naturally preserved as γ₁'s observable signal.
        """
        self._enter_phase(WorkflowState.MERGE_DEFERRED)
        await self.scheduler.set_task_status(self.task_id, 'merge-deferred')
        # Clear any requeue counter accumulated from prior failed attempts: the
        # task is workspace-green now, so those attempts are no longer relevant.
        # The harness's _handle_outcome_post only special-cases REQUEUED/DONE;
        # MERGE_DEFERRED falls through with requeued=False, leaving a stranded
        # counter if we don't clear it here.
        self.scheduler.clear_requeue_count(self.task_id)
        logger.info(
            'Task %s: train member workspace-green — parking in merge-deferred '
            '(group-merge worker owns done transition)',
            self.task_id,
        )
        trigger_outcome = await self._maybe_enqueue_group_merge()
        if trigger_outcome is not None:
            return trigger_outcome
        return WorkflowOutcome.MERGE_DEFERRED

    async def _maybe_enqueue_group_merge(self) -> WorkflowOutcome | None:
        """δ₂ trigger: enqueue a GroupMergeRequest when all train members are merge-deferred.

        Called immediately after ``_enter_merge_deferred`` writes the
        merge-deferred status.  Evaluates whether this task is the tip of a
        complete train and, if so, builds and enqueues a
        :class:`~orchestrator.merge_queue.GroupMergeRequest` from self's
        context (branch/worktree/task_files/module_configs/config).

        Gating (step-6 adds full guards; initial impl fires unconditionally
        when train metadata is present):

        * ``self._train is None`` → return ``None`` (non-train task; unreachable
          from ``_enter_merge_deferred`` but defensive).
        * ``members`` is empty → return ``None`` (no tasks found).
        * Any member is not merge-deferred (with self's status trusted as
          merge-deferred regardless of the get_tasks read lag) → return ``None``.
        * Self is not the tip (highest order) → return ``None`` (only the tip's
          branch contains all member commits).
        * ``self.worktree is None`` or ``self.merge_queue is None`` → log and
          return ``None``.

        Outcome mapping after awaiting the future:
        * ``None`` (soft-cancel) → ``_handle_soft_cancel('group-merge')``
        * ``result.status == 'done'`` → ``WorkflowOutcome.DONE``
        * Any other status → ``_mark_blocked(..., escalate_to_human=True)``

        Returns the mapped ``WorkflowOutcome``, or ``None`` to signal the caller
        to park the workflow as ``MERGE_DEFERRED``.
        """
        from orchestrator.merge_queue import (
            TRAIN_INCOMPLETE_REASON_PREFIX,
            GroupMergeRequest,
            MergeOutcome,  # noqa: F401 — kept for type completeness
            register_and_enqueue_merge_request,
        )

        train = self._train
        if train is None:
            return None

        train_id: str = train.get('id', '')  # type: ignore[assignment]
        if not train_id:
            return None

        members = await self.scheduler.tasks_by_train(train_id)
        if not members:
            return None

        # Guard: all members must be merge-deferred.
        # Trust self's just-written status (the get_tasks read may lag the write).
        def _effective_status(m: dict) -> str:
            return 'merge-deferred' if str(m.get('id')) == self.task_id else str(m.get('status', 'unknown'))

        if not all(_effective_status(m) == 'merge-deferred' for m in members):
            return None

        # Guard: self must be the tip (highest order → members[-1] after ascending sort).
        # Correctness depends on dispatch gating: β₁'s intra-train allowance lets
        # order=k start only once order=k-1 is merge-deferred, which guarantees the
        # tip (highest order) enters merge-deferred LAST.  The δ₁ group-merge worker
        # acts as a backstop: if this trigger does not fire (e.g. the tip re-enters
        # merge-deferred before a sibling propagates), the worker's own status pre-check
        # will reject the request and return TRAIN_INCOMPLETE, parking the train for retry.
        tip = members[-1]
        if str(tip.get('id')) != self.task_id:
            return None

        # Guard: infrastructure must be available.
        if self.worktree is None or self.merge_queue is None:
            logger.warning(
                'Task %s: train tip reached but worktree=%r merge_queue=%r — parking',
                self.task_id, self.worktree, self.merge_queue,
            )
            return None

        member_ids = [str(m.get('id')) for m in members]
        branch_name = self.task_id  # matches _submit_to_merge_queue convention

        async def _status_check(ids: list[str]) -> dict[str, str]:
            statuses, err = await self.scheduler.get_statuses(ids)
            if err is not None:
                # Log the transient MCP/read error so the silent discard is
                # visible.  We return the (possibly empty/partial) dict rather
                # than raising so the worker's existing train_incomplete pre-check
                # fires and routes back to the MERGE_DEFERRED park (see outcome
                # mapping below) instead of surfacing an unclassified 'Merge
                # worker error' that would escalate to a human.
                logger.warning(
                    'Task %s: train %r status_check got error from get_statuses '
                    '(will return partial %d-entry dict): %s',
                    self.task_id, train_id, len(statuses or {}), err,
                )
            return statuses or {}  # type: ignore[return-value]

        async def _mark_member_done(mid: str, sha: str) -> None:
            await self.scheduler.mark_done(
                mid, kind='merged', sha=sha, note=f'train {train_id}',
            )

        future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
        req = GroupMergeRequest(
            task_id=self.task_id,
            branch=branch_name,
            worktree=self.worktree,
            pre_rebased=False,
            task_files=self._task_files,
            module_configs=self._module_configs,
            config=self.config,
            result=future,
            train_id=train_id,
            member_task_ids=member_ids,
            tip_branch=branch_name,
            tip_task_id=self.task_id,
            status_check=_status_check,
            mark_member_done=_mark_member_done,
        )

        logger.info(
            'Task %s: all %d train members merge-deferred — enqueuing GroupMergeRequest '
            '(train=%r, members=%r)',
            self.task_id, len(member_ids), train_id, member_ids,
        )
        # Only the tip branch (req.branch) is registered in the in-flight
        # registry.  Member branches are intentionally out of scope here:
        # GroupMergeRequest carries member_task_ids but not member branch names,
        # so registering them would require coupling knowledge of the branch-
        # naming convention into δ₂.  The on-disk _merge-* worktree scan in
        # coalesce_or_enqueue_merge_request provides a fallback for member-branch
        # coalescing after the train merge starts executing.
        await register_and_enqueue_merge_request(
            self.merge_queue, req, self.event_store, self.merge_inflight_registry,
        )

        result = await self._await_cancellable(future)
        if result is None:
            return await self._handle_soft_cancel('group-merge')
        if result.status == 'done':
            if result.merge_sha:
                self._merge_sha = result.merge_sha
            return WorkflowOutcome.DONE
        # Distinguish transient from genuine failures.
        # train_incomplete: one or more members haven't propagated their
        # merge-deferred write yet (or a transient get_statuses error returned
        # a partial dict).  This is a timing artefact, not a real failure — park
        # the train and let δ₁ retry when the tip re-enters merge-deferred.
        if result.reason and result.reason.startswith(TRAIN_INCOMPLETE_REASON_PREFIX):
            logger.info(
                'Task %s: GroupMergeRequest returned train_incomplete for train %r '
                '— parking (MERGE_DEFERRED) for δ₁ retry. reason: %s',
                self.task_id, train_id, result.reason,
            )
            return None  # caller (_enter_merge_deferred) returns MERGE_DEFERRED
        # Orphan-halt probe: _map_advance_failure halted the merge queue (one of the
        # four halt-inducing statuses: wip_halted, done_wip_recovery,
        # wip_recovery_no_advance, unmerged_state) and halt_owner_esc_id is None,
        # so harness._on_escalation_resolved cannot unhalt the queue on L1 resolution.
        # Register the halt owner NOW so resolving the L1 auto-unhalts the queue
        # (parity with the single-task _handle_wip_conflict path — task 1599).
        # The probe is ground truth: it fires only when the queue IS halted with no
        # owner, is future-proof against new halt statuses, and never misfires on
        # genuine non-halt 'blocked' outcomes (rebase-conflict/cas_failed do NOT call
        # halt(), so is_wip_halted stays False for those paths).
        if (
            self.merge_worker is not None
            and self.merge_worker.is_wip_halted
            and self.merge_worker.halt_owner_esc_id is None
        ):
            return await self._escalate_train_halt(result, train_id)
        # Genuine failure (conflict/verify-red/rebase-conflict) → escalate to human
        # so a broken train is never silently parked forever (scenarios 5/8).
        return await self._mark_blocked(
            f'Group merge for train {train_id!r} failed: {result.status} — {result.reason}',
            escalate_to_human=True,
        )

    async def _finalise_merged_done(self) -> WorkflowOutcome:
        """Common DONE-finalisation for the happy-path merge.

        Refreshes ``metadata.files`` from the merge diff, then calls
        ``mark_done`` with ``kind='merged'``.  Extracted from ``run()`` so
        the state machine stays under pyright's complexity threshold.

        Returns ``WorkflowOutcome.DONE`` on success; ``BLOCKED`` (with L1)
        when the persistence layer refuses or ``_merge_sha`` is missing.
        """
        await self._reconcile_metadata_files_for_done()
        if not self._merge_sha:
            # Should never happen on the happy path — _submit_to_merge_queue
            # always populates _merge_sha on success.  Defensive bail-out
            # rather than passing done_provenance=None (which the server
            # gate now rejects with done_provenance_required).
            logger.error(
                'Task %s: SUCCESS path reached without _merge_sha — '
                'cannot construct done_provenance',
                self.task_id,
            )
            return await self._mark_blocked(
                'Internal: SUCCESS without merge_sha — provenance unconstructable',
                escalate_to_human=True,
            )
        try:
            await self.scheduler.mark_done(
                self.task_id, kind='merged', sha=self._merge_sha,
            )
        except SetTaskStatusRejected as exc:
            # Persistence layer refused the done write — the architect's
            # claim is contradicted by the row state.  Honest log + L1
            # instead of pretending the task is DONE.
            logger.error(
                'Task %s: set_task_status(done) rejected — %s: %s',
                self.task_id, exc.error_code, exc.raw,
            )
            return await self._mark_blocked(
                f'set_task_status(done) rejected: {exc.error_code} — {exc.raw}',
                escalate_to_human=True,
            )
        logger.info(
            f'Task {self.task_id} DONE — '
            f'cost=${self.metrics.total_cost_usd:.2f} '
            f'invocations={self.metrics.agent_invocations}'
        )
        return WorkflowOutcome.DONE

    async def _reconcile_metadata_files_for_done(self) -> None:
        """Set ``metadata.files`` to the merge-diff files before set_task_status('done').

        Truth source: ``git diff --name-only --no-renames <_base_commit>..<_merge_sha>``
        (excluding ``.task/``) — i.e. the files that actually landed on main,
        not the architect's plan.files (which the merge may have squashed,
        refactored, or rewritten).  Pre-fix, plan.files-derived metadata
        could include paths that no longer existed post-merge, tripping the
        phantom-done gate even though provenance was correct.

        Fall-back paths: when ``_merge_sha`` or ``_base_commit`` is None
        (already-on-main shortcuts, eval mode without a merge), write an
        empty list — the fused-memory gate-skip-when-verified-provenance
        branch handles the missing-files case from there.

        Sibling keys such as ``memory_hints`` and ``_causation_id`` (added by
        Stage-2 reconciliation after the workflow loaded ``self.task``) are
        preserved via the ``_merge_fresh_metadata`` read-modify-write; a
        pre-fix bare ``{'files': files}`` write clobbered them under the
        default ``append=False``.
        """
        if self._merge_sha and self._base_commit:
            files = await self.git_ops.get_merge_diff_files(
                self._base_commit, self._merge_sha,
            )
        else:
            files = []
        merged = await self._merge_fresh_metadata(
            self.task.get('metadata') or {},
            log_context='done metadata files reconcile',
        )
        merged['files'] = files
        # Optimistically update in-memory so downstream reads in this session
        # see the expected state.  In-memory is intentionally optimistic; the
        # backend is the authority and will be re-read on the next reconcile
        # cycle if this write fails (consistent with _handle_no_plan_failure).
        self.task['metadata'] = merged
        await self.scheduler.update_task(self.task_id, merged)

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

    async def _setup_worktree_and_artifacts(self, branch_name: str) -> None:
        """Set in-progress, create/inspect the worktree, init artifacts, scrub .task/.

        Extracted from ``run()`` so the state machine stays under pyright's
        complexity threshold.  Side-effects:

        - ``set_task_status('in-progress')``
        - populates ``self.worktree`` / ``self._base_commit`` /
          ``self._config_dir`` / ``self.artifacts`` / ``self._old_plan_base``
        - sets ``self._worktree_external = True`` when the caller pre-created
          the worktree (eval mode) so the cleanup path knows to leave it alone
        - syncs per-worktree venvs unless the worktree is external
        - removes any ``.task/`` paths inherited as tracked from main
        """
        # Pre-empt race: read live status BEFORE claiming 'in-progress'.
        # If the task was cancelled (or otherwise terminal) out-of-band in the
        # ~3 s window between acquire and dispatch, raise TerminalExitRejection
        # so run()'s existing 1489 handler converts it to WorkflowOutcome.CANCELLED
        # without an 'in-progress' write, no escalation, and no reopen.
        # A None / get_status failure is not in TERMINAL_STATUSES → fall through.
        live_status = await self.scheduler.get_status(self.task_id)
        if live_status in TERMINAL_STATUSES:
            raise TerminalExitRejection(
                task_id=self.task_id,
                old_status=live_status,
                target_status='in-progress',
                raw='preempt: task terminal at dispatch',
            )

        await self.scheduler.set_task_status(self.task_id, 'in-progress')

        # Create worktree (captures base commit for stable diffs).
        # If the caller pre-created the worktree (eval mode) skip creation
        # and rev-parse HEAD instead.
        if self.worktree is None:
            # Pass the live task title so create_worktree can quarantine a
            # reused worktree whose stored identity belongs to a different
            # (recycled-id) task — defense-in-depth behind Fix C's flag.
            train_meta = (self.task.get('metadata') or {}).get('train')
            worktree_info = await self.git_ops.create_worktree(
                branch_name,
                expected_title=(
                    self.task.get('title')
                    if self.config.worktree_identity_guard_enabled
                    else None
                ),
                train=cast(TrainMembership, train_meta) if isinstance(train_meta, dict) else None,
            )
            self.worktree = worktree_info.path
            base_commit = worktree_info.base_commit
        else:
            self._worktree_external = True
            proc = await asyncio.create_subprocess_exec(
                'git', 'rev-parse', 'HEAD',
                cwd=str(self.worktree),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            # Soft-fail: log and fall through rather than raising, so a
            # misconfigured eval-mode worktree degrades gracefully to
            # citation-grep alone instead of crashing setup.
            # NOTE: unlike the update_task soft-fail below (which keeps a valid
            # SHA in memory because only the metadata write failed), this path
            # has no SHA to record — _base_commit is left None so consumers
            # gating on `is not None` (e.g. the merge-queue plan-files-touched
            # check at line 3062) skip cleanly instead of receiving an empty-SHA.
            if proc.returncode != 0:
                logger.warning(
                    'Task %s: git rev-parse HEAD failed in external worktree '
                    '(rc=%s); base_commit will be None. stderr=%s',
                    self.task_id, proc.returncode,
                    stderr.decode(errors='replace').strip()[:200] or '<empty>',
                )
                base_commit = None
            else:
                base_commit = stdout.decode().strip()
        self._base_commit = base_commit
        # Record branch_base_sha in task metadata immediately after branch
        # creation so _reconcile_one_stranded (harness.py) can detect
        # zero-commit branches that sit on a main ancestor (Guard 3 in
        # the is_ancestor fast-path) and stale merge-markers from prior
        # incarnations of the same task id (stale-marker check).
        # Soft-fail: a transient update_task failure degrades gracefully
        # to citation-grep alone (Guards 1+2) instead of crashing setup.
        if base_commit:
            try:
                await self.scheduler.update_task(
                    self.task_id,
                    {'branch_base_sha': base_commit},
                    append=True,
                )
            except Exception:
                logger.warning(
                    'Task %s: failed to record branch_base_sha=%s; '
                    'reconciler will fall back to citation-grep guard',
                    self.task_id, base_commit,
                )
        # Colocate the per-task Claude config dir inside the worktree so
        # the session JSONL travels with the worktree.  Crash recovery
        # needs both the sidecar and the session file together to resume.
        self._config_dir = TaskConfigDir(
            self.task_id, base_dir=self.worktree / '.task',
        )

        if not self._worktree_external:
            await self._sync_worktree_venvs()

        self.artifacts = TaskArtifacts(self.worktree)
        # Capture old base_commit before init() overwrites metadata.json
        # — _plan() uses it for revalidation diff.
        self._old_plan_base = self.artifacts.read_base_commit()
        self.artifacts.init(
            self.task_id,
            self.task.get('title', ''),
            self.task.get('description', ''),
            base_commit=base_commit,
        )

        # ── .task/ contamination guard ────────────────────────────
        # If .task/ slipped into git on main (inherited contamination),
        # untrack it here so agents don't accidentally commit it.
        # Defense-in-depth — create_worktree() should have scrubbed, but
        # this catches the eval-mode path and race conditions.
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

    async def run(self) -> WorkflowOutcome:
        """Execute the full state machine."""
        branch_name = self.task_id
        try:
            await self._setup_worktree_and_artifacts(branch_name)
            # After setup, both attributes are guaranteed populated.
            # The asserts narrow the types so the rest of run() can
            # use self.worktree / self.artifacts without re-checking.
            assert self.worktree is not None
            assert self.artifacts is not None

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

            # ── Lever C: SIMPLE_TASK optimistic path ──────────────────
            # When the task title matches a small/well-bounded change and
            # auto_eval_redo metadata is absent, dispatch a single Sonnet
            # agent that explores, plans (via plan-tools MCP), and
            # implements end-to-end. Falls through to the architect path
            # on any failure (no plan written, unactionable artifact,
            # no steps marked done).
            if (
                not self.initial_plan
                and self.config.simple_task_enabled
                and not (self.task.get('metadata') or {}).get('auto_eval_redo')
                and not (self.task.get('metadata') or {}).get('force_full_path')
            ):
                from orchestrator.agents.triage import classify_simple_task
                if classify_simple_task(self.task):
                    self._enter_phase(WorkflowState.PLAN)
                    simple_outcome = await self._run_simple_task()
                    if simple_outcome == WorkflowOutcome.PLANNED:
                        # Plan written + step(s) marked done by SIMPLE_TASK.
                        # _execute_iterations will see no pending steps and
                        # return cleanly, allowing VERIFY to take over.
                        pass  # fall through to the post-PLAN section below
                    elif simple_outcome == WorkflowOutcome.DONE:
                        # SIMPLE_TASK reported task_already_done.
                        return simple_outcome
                    elif simple_outcome == WorkflowOutcome.BLOCKED:
                        # SIMPLE_TASK reported unactionable_task — terminal.
                        return simple_outcome
                    else:
                        # Fallthrough sentinel — drop the plan so the
                        # architect path runs cleanly.
                        if self.artifacts is not None:
                            plan_path = self.artifacts.root / 'plan.json'
                            plan_path.unlink(missing_ok=True)
                            (self.artifacts.root / 'plan.lock').unlink(
                                missing_ok=True,
                            )

            # PLAN (skip if initial_plan was provided — eval mode, or if
            # the SIMPLE_TASK path above already populated self.plan)
            if self.initial_plan:
                plan_tid = self.initial_plan.get('task_id')
                if plan_tid and plan_tid != self.task_id:
                    logger.error(
                        f'Task {self.task_id}: initial_plan has mismatched '
                        f'task_id {plan_tid} — discarding, will re-plan'
                    )
                    self.initial_plan = None
            if self.plan:
                # SIMPLE_TASK already populated self.plan — skip the
                # initial_plan / _plan() branch entirely.
                pass
            elif self.initial_plan:
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
                if plan_outcome == WorkflowOutcome.DONE:
                    # _handle_already_done_report set status=done with provenance
                    # and returned DONE.  Falling through to EXECUTE/VERIFY/MERGE
                    # wastes effort and (per the 2026-05-04 incident) can orphan a
                    # merge-queue halt if the workflow gets soft-cancelled
                    # mid-merge.
                    return plan_outcome
                if plan_outcome == WorkflowOutcome.ESCALATED:
                    # _mark_blocked returned ESCALATED: status='blocked' was already
                    # written, an L1 escalation is open, the steward handed off to
                    # a human.  Falling through to the post-PLAN ghost-loop guard
                    # would mistake any architect scratch in the worktree for
                    # "prior merge survived" and try to flip status to 'done'
                    # (the done_provenance gate currently blocks the DB write but
                    # the workflow logs a fake DONE — don't lie).  See task 2911
                    # incident 2026-05-06; mirrors 9760ba74bf for the DONE case.
                    return plan_outcome
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
                #
                # SINGLE SIGNAL — iteration log only.  We previously OR'd in
                # has_uncommitted_work as a backstop for the "merged but iteration
                # log empty" case, but that case is caught earlier by
                # _recover_if_already_merged() with SHA-primary semantics.  The
                # uncommitted backstop backfires whenever the architect leaves
                # scratch behind on a budget-exhaustion escalation — the worktree
                # has dirty files, the branch is on main, but no real work was
                # done.  Task 2911 (2026-05-06) hit exactly this and the workflow
                # tried to flip status to 'done' on an unimplemented task.
                assert _branch_check is not None  # narrowing: already_on_main is True
                wt_head, _ = _branch_check
                has_work = self._has_prior_implementation().has_work
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
                    # Train members hold at merge-deferred instead of merging;
                    # the group-merge worker (δ₁) owns the eventual done
                    # transition once all siblings are workspace-green (PRD §9.5,
                    # γ₁).  The full execute→verify→review pipeline still ran
                    # above (PRD acceptance criterion 5).  Non-train path is
                    # byte-identical — this guard only fires when metadata.train
                    # is a dict.
                    if self._train is not None:
                        return await self._enter_merge_deferred()

                    self._enter_phase(WorkflowState.MERGE)

                    # Defense-in-depth: any blocking L0 escalation, or any
                    # born-at-L2 (critical/urgent), or any level≥2 escalation
                    # created during execute/verify/review (e.g. plan-overwrite
                    # from _amend, or any future code path that escalates outside
                    # the implementer/debugger callsites) must gate the merge.
                    # Without this, an escalation queued mid-run would sit
                    # invisible while a merge proceeded.
                    # D4 accepted consequence: a pending critical/urgent from a
                    # prior incarnation DOES gate a fresh run — stop-the-line
                    # semantics.  See _is_gating_escalation for the full policy.
                    gating = [e for e in self._check_escalations() if _is_gating_escalation(e)]
                    if gating:
                        logger.warning(
                            'Task %s: %d gating escalation(s) at MERGE entry '
                            '— bailing to ESCALATED',
                            self.task_id, len(gating),
                        )
                        return WorkflowOutcome.ESCALATED

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
                                    # role='task' is explicit for γ's explicit-is-correct
                                    # invariant (mirrors merge_queue.py role='merge').
                                    # 'task' is already the default so this is documentary;
                                    # no call-site spy exists for this path — the
                                    # _verify_debugfix_loop spy in
                                    # test_workflow_verify_retry.py covers the primary site.
                                    role='task',
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
                            self._last_merge_block_reason = None
                            merge_outcome = await self._submit_to_merge_queue(
                                branch_name, pre_rebased=pre_rebased,
                                merge_phase=True,
                            )
                            if merge_outcome == WorkflowOutcome.DONE:
                                break
                            if merge_outcome != WorkflowOutcome.REQUEUED:
                                # BLOCKED — steward gave up, terminal
                                return merge_outcome

                            # Fix 3 — anti-thrash guard for repeated
                            # steward-resolved merge-phase loops on the same
                            # outcome signature.  At threshold escalates to L1
                            # rather than resubmitting the same merge.
                            if self._last_merge_block_reason is not None:
                                current_signature = hashlib.sha256(
                                    self._last_merge_block_reason.encode('utf-8'),
                                ).hexdigest()[:16]
                                prev_signature = (
                                    self.task.get('metadata') or {}
                                ).get('last_merge_outcome_signature')
                                thrash_outcome = await self._check_merge_outcome_thrash(
                                    prev_signature, current_signature,
                                )
                                if thrash_outcome is not None:
                                    return thrash_outcome

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
            return await self._finalise_merged_done()

        except AllAccountsCappedException as e:
            logger.warning(
                f'Task {self.task_id}: all accounts capped — '
                f'{e.retries} retries in {e.elapsed_secs:.1f}s (label={e.label!r})'
            )
            return await self._mark_blocked(
                f'All accounts capped: {e.label} — {e.retries} retries in {e.elapsed_secs:.1f}s',
                suggested_action='cap_wait_exceeded_sanity_bound',
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

        except SetTaskStatusRejected as exc:
            # Fast-path: a terminal-status rejection arrived out-of-band before
            # setup completed (either from the pre-empt live-status read or from
            # set_task_status('in-progress') itself).  Route through the same
            # bypass-discrimination helpers 1489 introduced for mid-flight aborts:
            # - 'cancelled'  → CANCELLED, no reopen, no escalation (sub-case 0).
            # - 'done', provenance on main  → DONE, no escalation (legitimate done).
            # - 'done', provenance missing / off-main  → reopen + L1 bypass_done +
            #   BLOCKED (bypass detected).
            # Only non-TerminalExitRejection subclasses (rare persistence errors)
            # fall through to the 'unhandled rejection' L1 path below.
            if isinstance(exc, TerminalExitRejection):
                cancelled_outcome = self._handle_cancelled_terminal_exit(exc)
                if cancelled_outcome is not None:
                    return cancelled_outcome

                # Not 'cancelled' (must be 'done' given TERMINAL_STATUSES).
                # Reuse the bypass-discrimination helper: returns BLOCKED when a
                # bypass is detected (reopen + bypass_done L1 already filed), or
                # None when provenance is legitimate (commit reachable from main).
                bypass_outcome = await self._handle_terminal_exit_on_block(
                    exc,
                    reason=(
                        f'terminal at dispatch (old_status={exc.old_status!r})'
                    ),
                    detail='preempt: task terminal at dispatch',
                )
                if bypass_outcome is not None:
                    # Bypass done detected — helper already reopened the row and
                    # filed an L1 with category='bypass_done'.
                    return bypass_outcome

                # Legitimate done: provenance commit is reachable from main.
                # The row is already terminal-done out-of-band (e.g. a human
                # marked done during the ~3s acquire→dispatch window). Accept it.
                logger.info(
                    'Task %s: already legitimately done at dispatch (old_status=%r) '
                    '— accepting, no reopen, no escalation',
                    self.task_id, exc.old_status,
                )
                self._enter_phase(WorkflowState.DONE)
                return WorkflowOutcome.DONE

            # A persistence-layer rejection escaped one of the workflow's
            # set_task_status / mark_done call sites without an explicit
            # handler.  Route to L1: the row state contradicts what the
            # workflow tried to write, which a steward retry can't unstick.
            # Caught BEFORE the broad Exception so it gets dedicated framing.
            logger.error(
                'Task %s: unhandled set_task_status rejection — %s: %s',
                self.task_id, exc.error_code, exc.raw,
            )
            return await self._mark_blocked(
                f'Unhandled set_task_status rejection: {exc.error_code} — {exc.raw}',
                escalate_to_human=True,
            )

        except Exception as e:
            logger.exception(f'Task {self.task_id} workflow error: {e}')
            return await self._mark_blocked(f'Workflow error: {e}')

        finally:
            # Stop steward if running
            if self._steward:
                await self._steward.stop()
            # Cleanup worktree (only if done AND branch is on main — preserve
            # otherwise so an agent's update_task(status='done') bypass doesn't
            # GC unmerged work). Skips externally-managed worktrees (eval mode).
            await self._maybe_cleanup_done_worktree()
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
            # Lock was stale and cleared.  Decide what to keep:
            #  - provenance-stamped or finalized → a complete plan from a prior
            #    session (e.g. blast-radius requeue) → keep for revalidation.
            #  - has steps but not finalized → a prior session was interrupted
            #    mid-build → keep the partial; the completion pass below finishes
            #    it instead of re-planning from scratch.
            #  - no steps, no provenance, not finalized → nothing recoverable →
            #    delete.
            existing = self.artifacts.read_plan()
            if (
                existing
                and not existing.get('_session_id')
                and not existing.get('_finalized_at')
                and not existing.get('steps')
            ):
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
            and not existing_plan.get('_finalized_at')
            and not existing_plan.get('_session_id')
        ):
            # ── Completion pass ───────────────────────────────────────
            # A prior session wrote steps but never finalized the plan
            # (it was cut off mid-build by a budget/turn cap or crash).
            # Hand the partial back to the architect to finish rather than
            # discard the work and re-plan from scratch.
            logger.info(
                'Task %s: resuming a partial plan (%d steps, not finalized) '
                'via the completion pass',
                self.task_id, len(existing_plan.get('steps', [])),
            )
            prompt = await self.briefing.build_plan_completion_prompt(
                self.task, existing_plan, worktree=self.worktree,
            )
        elif (
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

            # ── Lever B: revalidation skip ────────────────────────────
            # When main has gained no changes that touch the plan's files,
            # the architect's revalidation pass is a near-certain no-op.
            # Short-circuit by stamping _revalidated_at and bumping
            # base_commit ourselves, then return PLANNED. Falls through to
            # the existing architect path on any pre-flight failure.
            if (
                self.config.revalidation_skip_enabled
                and not overlap
                and await self._can_skip_revalidation(existing_plan)
            ):
                skipped = await self._apply_revalidation_skip(
                    existing_plan, current_main,
                )
                if skipped is not None:
                    return skipped
                # Pre-flight check failed inside _apply_revalidation_skip
                # (e.g. blast-radius lock denied) — fall through.

            prompt = await self.briefing.build_revalidation_prompt(
                self.task, existing_plan, revalidation_changed_files,
                worktree=self.worktree,
            )
            revalidation = True
        else:
            prompt = await self.briefing.build_architect_prompt(
                self.task, worktree=self.worktree,
            )

        # Snapshot pre-architect open L0 ids so the post-loop check can
        # detect "architect filed an L0 in lieu of writing a plan."  This is
        # the deterministic catch for RC2 — if it fires, we route directly
        # to L1 without re-invoking the architect (the no-plan loop burned
        # 16 successive Opus calls on task 917 because the cycle counter
        # reset every time main moved).
        pre_l0_ids: set[str] = set()
        if self.escalation_queue:
            pre_l0_ids = {
                e.id for e in self.escalation_queue.get_by_task(
                    self.task_id, status='pending', level=0,
                )
            }

        result: AgentResult | None = None
        rebase_retry_used = False
        for _outer_attempt in range(2):  # at most one rebase-retry round-trip
            for attempt in range(2):
                result = await self._invoke(ARCHITECT, prompt, self.worktree)

                if not result.success:
                    cls = classify_agent_failure(result)
                    # Salvage: the architect may have finalized a complete plan
                    # on disk before the CLI run hit a budget/turn cap (or was
                    # otherwise cut off).  A plan carrying ``_finalized_at`` was
                    # explicitly declared complete, so it is safe to use despite
                    # the run-level failure — the same validation gates below
                    # still apply.  Without the marker the plan is an unfinished
                    # partial; block and let the next session's completion pass
                    # finish it.
                    salvaged = self.artifacts.read_plan()
                    if (
                        salvaged
                        and salvaged.get('_finalized_at')
                        and salvaged.get('steps')
                    ):
                        logger.warning(
                            'Task %s: architect run failed (%s) but a finalized '
                            'plan (%d steps) is on disk — salvaging instead of '
                            'discarding',
                            self.task_id, cls.kind.value,
                            len(salvaged.get('steps', [])),
                        )
                        if self.event_store:
                            self.event_store.emit(
                                EventType.plan_salvaged,
                                task_id=self.task_id,
                                phase='plan',
                                data={
                                    'failure_kind': cls.kind.value,
                                    'cost_usd': result.cost_usd,
                                    'turns': result.turns,
                                    'steps': len(salvaged.get('steps', [])),
                                },
                            )
                        self.plan = salvaged
                        break  # fall through to validation / provenance / lock
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

            # Architect task-rejection artifacts.  All are terminal — handle
            # deterministically before the "no plan.json" failure path so
            # none interacts with the consecutive_no_plan_failures cycle
            # counter.  Order matters: unactionable_task and false_premise are
            # the most decisive (jump straight to L1, bypass steward);
            # already_done is a clean DONE; blocking_dependency may re-loop
            # the architect.
            if self.artifacts.read_unactionable_task() is not None:
                return await self._handle_unactionable_task_report()
            if self.artifacts.read_false_premise() is not None:
                return await self._handle_false_premise_report()
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

        # RC2 deterministic catch: the architect filed a fresh L0 escalation
        # in lieu of writing a plan.  Route straight to L1 (skip the steward,
        # skip the no-plan loop) — the architect already filed an L0, so
        # _mark_blocked must not create another one.
        if not self.plan and self.escalation_queue:
            post_l0 = self.escalation_queue.get_by_task(
                self.task_id, status='pending', level=0,
            )
            new_l0 = [e for e in post_l0 if e.id not in pre_l0_ids]
            if new_l0:
                summary = new_l0[0].summary
                logger.warning(
                    'Task %s: architect filed L0 without plan (%s) — '
                    'auto-promoting to L1 to break no-plan loop',
                    self.task_id, summary[:120],
                )
                return await self._mark_blocked(
                    f'Architect filed L0 without plan: {summary}',
                    detail=new_l0[0].detail,
                    escalate_to_human=True,
                    skip_escalation=True,  # the architect already filed one
                )

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

        # Hard-require the completeness marker.  The architect must call
        # confirm_plan() as its final action; a plan with steps but no
        # _finalized_at is an unfinished partial — either the run was cut off
        # mid-build, or the architect skipped the final call.  Route it through
        # the no-plan failure handler (which has cycle detection).  The partial
        # is left on disk, so the next session's completion pass can resume it.
        #
        # This runs BEFORE stamp_plan_provenance so a missing confirm_plan is
        # caught here rather than masked by the marker that stamping backfills.
        if not self.plan.get('_finalized_at'):
            return await self._handle_no_plan_failure(
                'Planning failed: plan not finalized '
                '(architect did not call confirm_plan)',
                detail=(
                    'The plan on disk has steps but no _finalized_at marker. '
                    'The architect must call confirm_plan() as its final action '
                    'to mark the plan complete. Treating as an incomplete plan; '
                    'a later session will resume and finish it.'
                ),
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
                # Annotate the requeue so the per-task retry-cap report can
                # name *why* — without this, three blast-radius requeues in a
                # row produce a cap-exhaust report with phase/reason='unknown'.
                additional = sorted(set(plan_modules) - set(self.modules))
                self._last_block_phase = self.state.value
                self._last_block_reason = 'plan_blast_radius_lock_conflict'
                self._last_block_detail = (
                    f'Plan expansion blocked: additional locks {additional} '
                    f'unavailable (held by other tasks). '
                    f'Held modules: {sorted(self.modules)}; '
                    f'plan modules: {sorted(plan_modules)}.'
                )
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

    async def _can_skip_revalidation(self, plan: dict) -> bool:
        """Lever B pre-flight checks for the revalidation skip.

        All checks must pass for the optimisation to apply. Conservative —
        on any uncertainty, return False so the existing architect-driven
        revalidation runs.
        """
        assert self.worktree is not None
        # Schema version must match — bumped schemas mean the architect
        # may need to refresh fields the orchestrator can't synthesise.
        if plan.get('_schema_version') != PLAN_SCHEMA_VERSION:
            logger.info(
                'Task %s: revalidation skip declined — schema mismatch '
                '(plan=%r, current=%r)',
                self.task_id,
                plan.get('_schema_version'),
                PLAN_SCHEMA_VERSION,
            )
            return False
        # Every plan file must still exist in the worktree.
        for f in plan.get('files', []):
            if not (self.worktree / f).exists():
                logger.info(
                    'Task %s: revalidation skip declined — plan file %r '
                    'missing in worktree',
                    self.task_id, f,
                )
                return False
        # Plan provenance age bound.
        created_at = plan.get('_created_at') or plan.get('_revalidated_at')
        if created_at:
            try:
                stamped = datetime.fromisoformat(str(created_at))
                if stamped.tzinfo is None:
                    stamped = stamped.replace(tzinfo=UTC)
                age_hours = (
                    datetime.now(UTC) - stamped
                ).total_seconds() / 3600.0
                if age_hours > self.config.max_revalidation_age_hours:
                    logger.info(
                        'Task %s: revalidation skip declined — plan age '
                        '%.1fh exceeds bound %.1fh',
                        self.task_id, age_hours,
                        self.config.max_revalidation_age_hours,
                    )
                    return False
            except (TypeError, ValueError):
                logger.info(
                    'Task %s: revalidation skip declined — could not parse '
                    '_created_at=%r',
                    self.task_id, created_at,
                )
                return False
        # Conservative: if the prior run blocked at REVIEW, the architect
        # path may be needed to refresh strategy. Read the iteration log
        # for the most recent block annotation.
        metadata = self.task.get('metadata') or {}
        if (
            metadata.get('last_block_phase') == 'review'
            and metadata.get('last_block_outcome') == 'blocked'
        ):
            logger.info(
                'Task %s: revalidation skip declined — prior block at REVIEW',
                self.task_id,
            )
            return False
        return True

    async def _apply_revalidation_skip(
        self, plan: dict, current_main: str,
    ) -> WorkflowOutcome | None:
        """Lever B side-effect path. Returns PLANNED on success, or None to
        signal the caller should fall through to the architect path."""
        assert self.artifacts is not None and self.worktree is not None

        # Re-derive modules from plan files; if scope grew, ask scheduler
        # to expand the lock. Denial means a sibling task holds an
        # overlapping module — fall through to the architect path which
        # already handles the requeue case.
        plan_files = plan.get('files', [])
        plan_modules = files_to_modules(plan_files, self.config.lock_depth)
        if set(plan_modules) != set(self.modules):
            expanded = await self.scheduler.handle_blast_radius_expansion(
                self.task_id, self.modules, plan_modules,
            )
            if not expanded:
                logger.info(
                    'Task %s: revalidation skip declined — blast-radius '
                    'expansion denied',
                    self.task_id,
                )
                return None
            self.modules = plan_modules
            self._module_configs = self._resolve_module_configs()

        # Bump revalidation stamp + base commit (mirrors confirm_plan).
        try:
            self.artifacts.bump_revalidation_stamp(
                self.session_id, base_commit=current_main,
            )
        except ValueError as exc:
            logger.warning(
                'Task %s: revalidation skip declined — bump_revalidation_stamp '
                'rejected plan: %s',
                self.task_id, exc,
            )
            return None

        # Acquire plan lock (the architect path also does this; the eval
        # path skips it entirely).  If another session holds the lock,
        # fall through to the architect path which has its own retry logic.
        if not self.artifacts.lock_plan(self.session_id):
            logger.info(
                'Task %s: revalidation skip declined — plan.lock contended',
                self.task_id,
            )
            return None

        self.plan = self.artifacts.read_plan()

        # Stamp optimistic-path metadata for the auto-eval hook.
        await self._stamp_optimistic_path('revalidation_skip')

        if self.event_store:
            self.event_store.emit(
                EventType.phase_skipped,
                task_id=self.task_id,
                phase='plan',
                data={
                    'reason': 'revalidation_skipped_no_overlap',
                    'plan_session_id': str(plan.get('_session_id') or ''),
                    'plan_files': plan_files,
                    'main_sha': current_main,
                },
            )

        logger.info(
            'Task %s: Lever B — revalidation skipped (overlap=0, '
            'main %s -> stamped on plan)',
            self.task_id, current_main[:12],
        )
        return WorkflowOutcome.PLANNED

    async def _run_simple_task(self) -> WorkflowOutcome:
        """Lever C — single-agent simple-task path.

        Dispatches the SIMPLE_TASK role (sonnet) to explore, register a
        plan via the plan-tools MCP server, edit the listed files, and
        mark the step(s) done in one session.

        Returns:
            ``PLANNED`` — plan.json was written and at least one step
                marked done; the workflow continues to VERIFY without
                invoking the implementer.
            ``DONE`` — SIMPLE_TASK reported the work is already on main.
            ``BLOCKED`` — SIMPLE_TASK reported the task is unactionable.
            ``REQUEUED`` (sentinel) — SIMPLE_TASK gave up partway; caller
                falls through to the architect path.
        """
        assert self.worktree is not None and self.artifacts is not None

        prompt = await self.briefing.build_simple_task_prompt(
            self.task, worktree=self.worktree,
        )

        try:
            result = await self._invoke(SIMPLE_TASK, prompt, self.worktree)
        except Exception as exc:  # noqa: BLE001 — fall through to architect
            logger.warning(
                'Task %s: SIMPLE_TASK invocation failed (%s) — '
                'falling through to architect path',
                self.task_id, exc,
            )
            return WorkflowOutcome.REQUEUED

        if not result.success:
            cls = classify_agent_failure(result)
            logger.info(
                'Task %s: SIMPLE_TASK did not succeed (%s) — '
                'falling through to architect path',
                self.task_id, cls.kind.value,
            )
            return WorkflowOutcome.REQUEUED

        # Architect-style escape hatches — same artifacts so the existing
        # handlers work unchanged.
        if self.artifacts.read_unactionable_task() is not None:
            return await self._handle_unactionable_task_report()
        if self.artifacts.read_false_premise() is not None:
            return await self._handle_false_premise_report()
        if self.artifacts.read_already_done() is not None:
            return await self._handle_already_done_report()
        if self.artifacts.read_blocking_dependency() is not None:
            # Falling through — the architect path's
            # _handle_blocking_dep_report logic re-acquires base_commit
            # context and registers the dependency cleanly.
            return WorkflowOutcome.REQUEUED

        plan = self.artifacts.read_plan()
        if not plan or not plan.get('steps'):
            logger.info(
                'Task %s: SIMPLE_TASK wrote no plan — falling through',
                self.task_id,
            )
            return WorkflowOutcome.REQUEUED

        # Verify at least one step is done — the SIMPLE_TASK contract is
        # plan + implement + mark-done. Anything less and we let the
        # architect path take over to avoid handing a half-built plan to
        # _execute_iterations.
        any_done = any(
            isinstance(s, dict) and s.get('status') == 'done'
            for col in ('prerequisites', 'steps')
            for s in plan.get(col, [])
        )
        if not any_done:
            logger.info(
                'Task %s: SIMPLE_TASK plan has no steps marked done — '
                'falling through to architect path',
                self.task_id,
            )
            return WorkflowOutcome.REQUEUED

        # Stamp provenance + acquire plan lock (same shape as _plan()).
        try:
            self.artifacts.stamp_plan_provenance(self.session_id)
        except ValueError as exc:
            logger.warning(
                'Task %s: SIMPLE_TASK provenance stamp failed (%s) — '
                'falling through',
                self.task_id, exc,
            )
            return WorkflowOutcome.REQUEUED
        self.artifacts.lock_plan(self.session_id)
        self.plan = self.artifacts.read_plan()

        # Refresh module assignments from the plan's files (handles the
        # case where the SIMPLE_TASK agent expanded scope by one file).
        plan_files = self.plan.get('files', [])
        if plan_files:
            plan_modules = files_to_modules(plan_files, self.config.lock_depth)
            if set(plan_modules) != set(self.modules):
                expanded = await self.scheduler.handle_blast_radius_expansion(
                    self.task_id, self.modules, plan_modules,
                )
                if expanded:
                    self.modules = plan_modules
                    self._module_configs = self._resolve_module_configs()

        await self._stamp_optimistic_path('simple_task')

        if self.event_store:
            self.event_store.emit(
                EventType.phase_skipped,
                task_id=self.task_id,
                phase='plan',
                data={
                    'reason': 'architect_skipped_simple_task',
                    'classifier_signals': {
                        'title': str(self.task.get('title') or ''),
                        'files': list(
                            (self.task.get('metadata') or {}).get('files') or []
                        ),
                        'priority': str(self.task.get('priority') or ''),
                    },
                    'plan_files': plan_files,
                },
            )

        logger.info(
            'Task %s: Lever C — SIMPLE_TASK produced plan with %d step(s)',
            self.task_id, len(self.plan.get('steps', [])),
        )
        return WorkflowOutcome.PLANNED

    async def _stamp_optimistic_path(self, kind: str) -> None:
        """Stamp ``metadata.optimistic_path`` on the task so the harness's
        auto-eval hook can detect that this task took the optimistic path
        on its current attempt.

        Fire-and-forget — failure logs a warning and does not block.
        """
        try:
            metadata = dict(self.task.get('metadata') or {})
            metadata['optimistic_path'] = kind
            self.task['metadata'] = metadata
            await self.scheduler.update_task(
                self.task_id, metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                'Task %s: failed to stamp optimistic_path=%s: %s',
                self.task_id, kind, exc,
            )

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
            # The repair restructures an existing complete plan into valid
            # schema (the architect was told not to redesign), so the repaired
            # plan is complete by construction.  Stamp the completeness marker
            # — the repair prompt writes plan.json via the Write tool rather
            # than confirm_plan, so the _finalized_at gate would otherwise
            # reject this plan.
            repaired_plan.setdefault('_finalized_at', datetime.now(UTC).isoformat())
            self.artifacts.write_plan(repaired_plan)
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

    async def _merge_fresh_metadata(
        self,
        in_memory_metadata: dict,
        *,
        log_context: str,
    ) -> dict:
        """Read-modify-write helper: merge in-memory metadata with the backend's.

        Fetches the backend's current task metadata and overlays it onto
        ``in_memory_metadata`` so that keys added after ``self.task`` was loaded
        (e.g. ``memory_hints`` re-attached by Stage-2 reconciliation) survive the
        next write.  Backend keys win on collision, which is the correct policy: a
        backend-side addition represents an external write that must not be
        discarded.

        If ``get_task`` fails, logs a warning containing ``log_context`` and falls
        back to ``in_memory_metadata`` alone.  Mirrors the boundary-normalisation
        pattern used in ``_handle_terminal_exit_on_block``.

        Args:
            in_memory_metadata: The in-memory metadata dict (from
                ``self.task.get('metadata') or {}``).
            log_context: Short descriptor of the write site, used verbatim in the
                fallback warning (e.g. ``'no-plan counter'``,
                ``'infra-resume thrash counter'``).

        Returns:
            A new dict ready for counter-field assignment and persist.
        """
        try:
            fresh_task = await self.scheduler.get_task(self.task_id)
        except Exception as exc:  # noqa: BLE001 — best-effort, fall back to in-memory
            logger.warning(
                'Task %s: failed to refresh metadata before %s write; '
                'falling back to in-memory metadata '
                '(memory_hints may be clobbered): %s',
                self.task_id, log_context, exc,
            )
            fresh_task = None
        # Merge: start from in-memory metadata so that locally-set keys not yet
        # persisted are preserved, then overlay fresh backend keys so that
        # backend-side additions (e.g. memory_hints from Stage-2 reconciliation)
        # win on collision.  When get_task failed (fresh_task is None) the
        # backend overlay is empty and we fall back to the in-memory copy only.
        return {
            **in_memory_metadata,
            **((fresh_task.get('metadata') or {}) if isinstance(fresh_task, dict) else {}),
        }

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
        last_sha = str(metadata.get('last_no_plan_main_sha') or '')
        try:
            counter = int(metadata.get('consecutive_no_plan_failures') or 0)
        except (TypeError, ValueError):
            counter = 0
        try:
            total = int(metadata.get('total_no_plan_failures') or 0)
        except (TypeError, ValueError):
            total = 0

        if not current_main_sha or last_sha != current_main_sha:
            counter = 1
        else:
            counter += 1

        # Total counter never resets — backstops the SHA-keyed counter when
        # main keeps moving and the per-SHA counter never reaches 2 (the
        # bug behind 16 successive Opus calls on task 917).
        total += 1

        # Read-modify-write: see _merge_fresh_metadata for the merge policy.
        new_metadata = await self._merge_fresh_metadata(
            metadata, log_context='no-plan counter',
        )
        new_metadata['last_no_plan_main_sha'] = current_main_sha
        # Counters intentionally sourced from in-memory metadata; backend overlay is only for non-counter keys (e.g. memory_hints).
        new_metadata['consecutive_no_plan_failures'] = counter
        new_metadata['total_no_plan_failures'] = total
        self.task['metadata'] = new_metadata
        try:
            await self.scheduler.update_task(self.task_id, metadata=new_metadata)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                'Task %s: failed to persist no-plan cycle counter: %s',
                self.task_id, exc,
            )

        if counter >= 2 or total >= 3:
            trigger = (
                'same-SHA counter' if counter >= 2 else 'total counter'
            )
            logger.warning(
                'Task %s: no-plan loop confirmed (%s) — '
                'consecutive=%d on main SHA %s, total=%d; escalating to human',
                self.task_id, trigger, counter,
                current_main_sha[:12] or '<unknown>', total,
            )
            full_reason = (
                f'Repeated no-plan failure (counter={counter}, total={total}) '
                f'via {trigger}: {reason}'
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

        # PRD § 9.8 — train members bypass this counter entirely.
        # The loop-guard (γ₂/task 1523) owns train verify-phase thrash;
        # the train-merge worker owns the merge phase.  The counter adds
        # no value for train members and risks false-trips on legitimate
        # merge-deferred → merge-deferred re-stamps from sibling activity.
        if isinstance(metadata.get('train'), dict):
            logger.debug(
                'Task %s: train member — bypassing infra-resume thrash counter '
                '(loop-guard owns train thrash)',
                self.task_id,
            )
            return None

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

        # Read-modify-write: see _merge_fresh_metadata for the merge policy.
        new_metadata = await self._merge_fresh_metadata(
            metadata, log_context='infra-resume thrash counter',
        )
        # Counters intentionally sourced from in-memory metadata; backend overlay is only for non-counter keys (e.g. memory_hints).
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

    async def _check_merge_outcome_thrash(
        self,
        prev_signature: str | None,
        current_signature: str,
    ) -> WorkflowOutcome | None:
        """Fix 3 — detect & escalate repeated steward-resolved merge-phase loops.

        Mirrors :meth:`_check_infra_resume_thrash` for the merge-retry loop.
        Called from the merge-phase loop after ``_submit_to_merge_queue``
        returns ``REQUEUED`` (steward resolved an L0 and the loop is about
        to resubmit).  If the merge-outcome signature matches the previous
        attempt, increment ``consecutive_merge_thrash`` in the task
        metadata.  At ``max_consecutive_merge_thrash``, route to
        ``_mark_blocked(escalate_to_human=True)`` instead of resubmitting.

        ``current_signature`` is a sha256-short fingerprint of the blocked
        ``MergeOutcome.reason``; full text would bloat metadata when verify
        failure reports run multi-kilobyte.

        Returns ``WorkflowOutcome.BLOCKED`` at threshold; ``None`` to fall
        through to the resubmit.
        """
        metadata = self.task.get('metadata') or {}

        # PRD § 9.8 — train members bypass this counter entirely.
        # The train-merge worker owns the merge phase for train members;
        # the counter adds no value and risks false-trips on legitimate
        # merge-deferred → merge-deferred re-stamps.
        if isinstance(metadata.get('train'), dict):
            logger.debug(
                'Task %s: train member — bypassing merge-outcome thrash counter '
                '(train-merge worker owns merge phase)',
                self.task_id,
            )
            return None

        try:
            counter = int(metadata.get('consecutive_merge_thrash') or 0)
        except (TypeError, ValueError):
            counter = 0

        if prev_signature is not None and prev_signature == current_signature:
            counter += 1
        else:
            # Different verdict (or first observation) → steward made
            # progress on something; reset to 1 because we just saw one.
            counter = 1

        new_metadata = dict(metadata)
        new_metadata['consecutive_merge_thrash'] = counter
        new_metadata['last_merge_outcome_signature'] = current_signature
        self.task['metadata'] = new_metadata
        try:
            await self.scheduler.update_task(
                self.task_id, metadata=new_metadata,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort, log and proceed
            logger.warning(
                'Task %s: failed to persist merge-thrash counter: %s',
                self.task_id, exc,
            )

        if counter >= self.config.max_consecutive_merge_thrash:
            logger.warning(
                'Task %s: consecutive_merge_thrash=%d at threshold %d — '
                'merge-phase thrash confirmed; escalating to human',
                self.task_id, counter,
                self.config.max_consecutive_merge_thrash,
            )
            return await self._mark_blocked(
                f'Repeated merge-phase thrash (counter={counter})',
                detail=f'merge_outcome_signature={current_signature}',
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
        await self._reconcile_metadata_files_for_done()
        try:
            await self.scheduler.mark_done(
                self.task_id,
                kind='found_on_main',
                sha=commit,
                note=(
                    f'architect-reported task already on main; '
                    f'evidence: {evidence[:400]}'
                ),
            )
        except SetTaskStatusRejected as exc:
            # Architect's claim is contradicted by the persistence layer —
            # L1-worthy: a steward retry can't reconcile a phantom-done or
            # provenance ancestor mismatch.
            logger.error(
                'Task %s: mark_done rejected after already_done report — %s: %s',
                self.task_id, exc.error_code, exc.raw,
            )
            return await self._mark_blocked(
                f'Architect already_done rejected: {exc.error_code} — {exc.raw}',
                escalate_to_human=True,
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

    async def _handle_false_premise_report(self) -> WorkflowOutcome:
        """Process a ``.task/false_premise.json`` report from the architect.

        Caller has already verified the artifact exists.

        Stops the steward early to close the small async window where a
        stale L0 from a prior PLAN attempt could be processed concurrently
        with our L1 submission.  The ``finally`` block in ``run()`` also
        stops the steward, so this is defense-in-depth.

        Then short-circuits to ``_mark_blocked(escalate_to_human=True,
        category='design_concern')``, which submits a level-1 design_concern
        escalation directly without invoking the steward — only a human/curator
        can re-spec a test premise or relocate a signal to a different task.
        """
        assert self.artifacts is not None
        report = self.artifacts.read_false_premise()
        assert report is not None  # caller must have verified

        classification = str(report.get('classification') or '')
        premise = str(report.get('premise') or '').strip()
        evidence = str(report.get('evidence') or '')
        proposed_resolution = str(report.get('proposed_resolution') or '')

        if self._steward:
            await self._steward.stop()
            self._steward = None

        if not premise:
            result = await self._mark_blocked(
                'Architect wrote malformed false_premise.json '
                '(missing premise)',
                detail=json.dumps(report, indent=2)[:2000],
                escalate_to_human=True,
                category='design_concern',
            )
        else:
            result = await self._mark_blocked(
                f'Architect reported false RED-test premise: {premise}',
                detail=(
                    f'classification: {classification}\n'
                    f'evidence: {evidence}\n'
                    f'proposed_resolution: {proposed_resolution}'
                )[:2000],
                escalate_to_human=True,
                category='design_concern',
            )
        # Clear only after the L1 escalation has been successfully submitted —
        # if _mark_blocked raises, the artifact survives for the next retry.
        self.artifacts.clear_false_premise()
        return result

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
                    amend_ok = await self._amend(in_scope, amendment_round)
                    if not amend_ok:
                        return WorkflowOutcome.ESCALATED
                    self.metrics.amendment_rounds += 1
                    continue  # re-loop: EXECUTE → VERIFY → REVIEW

                # Cap exhausted or nothing in-scope — existing DONE path.
                # Route suggestions directly to the curator intake (fire-and-
                # forget); fall back to memory write when there are none.
                # _escalate_suggestions is retained as the steward fallback
                # but is no longer called from this path.
                if reviews.suggestions:
                    await self._route_review_suggestions_to_curator(reviews)
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

            # Check for escalations.  Plain blocking L1 (already-escalated-to-human)
            # issues from prior runs of this same task stay in the queue until a
            # human or escalation-watcher resolves them; we must not let them
            # sink the current run.  Fresh L0 blocking escalations gate progress
            # here, AND born-at-L2 (critical/urgent) escalations AND any level≥2
            # escalations also gate regardless of severity — D4 accepted consequence:
            # a pending critical/urgent from a prior incarnation DOES sink a fresh
            # run (intended stop-the-line semantics).  See _is_gating_escalation.
            blocking = [e for e in self._check_escalations() if _is_gating_escalation(e)]
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
                force_workspace=self._train is not None,
                role='task',
            )
            if not result.passed:
                self._last_verify_result = result
            if result.passed:
                return WorkflowOutcome.DONE

            verify_attempt += 1
            # Fast-fail: when the verifier's own injected ``Command timed out
            # after Ns: …`` wrapper string is the only signal, retrying gives
            # the debugger nothing actionable.  Escalate to L1 after
            # ``max_opaque_timeout_attempts`` instead of burning the full
            # ``max_verify_attempts × verify_command_timeout_secs`` budget.
            # The streamed-stdout fix in ``_run_cmd`` means a real cause hint
            # should appear on attempt 1 for any genuine in-test hang;
            # persistent opaque timeouts indicate infrastructure the debugger
            # can't fix.
            if (
                result.category == 'infra_timeout'
                and _OPAQUE_TIMEOUT_CAUSE_RE.match(result.cause_hint or '')
                and verify_attempt >= self.config.max_opaque_timeout_attempts
            ):
                logger.warning(
                    'Task %s: opaque infra_timeout cap hit at attempt %d/%d '
                    '(cause_hint=%r) — escalating to L1',
                    self.task_id, verify_attempt,
                    self.config.max_opaque_timeout_attempts,
                    (result.cause_hint or '')[:120],
                )
                return WorkflowOutcome.BLOCKED
            # Signature-repetition guard: escalate to L1 after N consecutive
            # verify failures with the same (category, normalised cause_hint)
            # tuple.  Placement is AFTER the opaque-timeout cap (the more
            # specific signal) and BEFORE the global max_verify_attempts cap.
            # Uses the last-N-equal window so an early "blip" of a different
            # category does not prevent the guard from firing once the loop
            # settles into a repeat pattern.
            # Empty-string signatures ('', '') count as identical triples and
            # trigger the guard intentionally: if the verifier repeatedly
            # returns opaque output with no category or cause_hint, escalating
            # to L1 is still the right response.
            _sig = (result.category or '', _normalize_cause_hint(result.cause_hint))
            self._failure_signature_history.append(_sig)
            _N = self.config.max_failure_signature_repeat
            if (
                len(self._failure_signature_history) >= _N
                and all(s == _sig for s in self._failure_signature_history[-_N:])
            ):
                logger.warning(
                    'Task %s: %d consecutive identical verify failures '
                    '(sig=%r) — escalating to L1',
                    self.task_id, _N, _sig,
                )
                return WorkflowOutcome.BLOCKED
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

            # Check for escalations from debugger.  Same filter as the post-implementer
            # check — a stale plain blocking L1 from a prior run must not sink a
            # successful debug pass.  Born-at-L2 (critical/urgent) AND any level≥2
            # escalations also gate here — D4 accepted consequence: a pending
            # critical/urgent from a prior incarnation DOES sink a fresh run
            # (intended stop-the-line semantics).  See _is_gating_escalation.
            blocking = [e for e in self._check_escalations() if _is_gating_escalation(e)]
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
    ) -> bool:
        """Invoke the implementer to apply in-scope review suggestions.

        Amendment passes skip the architect entirely — the plan is frozen,
        no new steps are added, the implementer patches the existing diff
        in place. Scope is enforced by the ``_suggestions_in_scope`` filter
        upstream (module-lock membership) and reinforced in the prompt.

        Returns:
            True on success. False if the amendment overwrote plan.json
            (ownership mismatch) — in that case ``_escalate_plan_overwrite``
            is called and the caller MUST halt the workflow rather than
            continue as if the pass succeeded.
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
            return False

        return True

    async def _try_narrow_plan(self, not_touched: list[str]) -> bool:
        """Give the architect ONE chance to narrow the plan after the
        merge gate flagged declared-but-untouched files.

        Lenient semantics: the architect may keep some flagged entries
        (treating them as genuinely needed); the only hard constraint
        is no NEW files may be added.  The gate's re-check is the
        source of truth for pass/fail.

        Returns True when the narrowing pass should be re-checked
        against the gate, False otherwise (one-shot already fired,
        architect invocation failed, or post-pass subset check rejected
        the new plan because it introduced new files).
        """
        if self._plan_tightened:
            return False
        self._plan_tightened = True

        before = set(self.plan.get('files', []))
        prompt = await self.briefing.build_plan_tightening_prompt(
            self.task, self.plan, not_touched, worktree=self.worktree,
        )
        assert self.worktree is not None
        assert self.artifacts is not None
        result = await self._invoke(ARCHITECT, prompt, self.worktree)
        if not result.success:
            return False

        self.plan = self.artifacts.read_plan()
        after = set(self.plan.get('files', []))
        if not after.issubset(before):
            logger.warning(
                'Task %s: narrowing pass added new files %s — rejecting',
                self.task_id, sorted(after - before),
            )
            return False
        # Lenient: partial narrowing or confirm_plan() both fall through
        # to the gate's re-check below.
        return True

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
        from orchestrator.merge_queue import (
            PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
            MergeOutcome,
            MergeRequest,
            _check_plan_files_touched_in_branch,
            _emit_merge_attempt,
            register_and_enqueue_merge_request,
        )

        assert self.worktree is not None
        assert self.merge_queue is not None

        # Decision-1 pre-merge subset check: refuse to merge a branch that
        # never touched the architect's declared plan files.  Before
        # escalating, give the architect ONE bounded chance to narrow the
        # plan against current branch state — only the architect (not the
        # steward) is allowed to mediate, and only to drop genuinely
        # unneeded entries.  Adding new files is rejected by the helper's
        # subset check.  The gate's re-check is the source of truth for
        # pass/fail.
        if (
            self._task_files
            and self._base_commit is not None
        ):
            rc, branch_head, _ = await _run(
                ['git', 'rev-parse', 'HEAD'], cwd=self.worktree,
            )
            if rc == 0 and branch_head.strip():
                check = await _check_plan_files_touched_in_branch(
                    list(self._task_files),
                    self._base_commit,
                    branch_head.strip(),
                    self.git_ops,
                    task_id=self.task_id,
                )
                narrowing_succeeded = False
                if check.not_touched:
                    narrowed = await self._try_narrow_plan(check.not_touched)
                    if narrowed:
                        # Re-check the (possibly narrowed) plan against the
                        # gate.  ``_task_files`` reads from ``self.plan``,
                        # which ``_try_narrow_plan`` has refreshed from disk.
                        rc2, branch_head2, _ = await _run(
                            ['git', 'rev-parse', 'HEAD'], cwd=self.worktree,
                        )
                        if rc2 == 0 and branch_head2.strip():
                            check = await _check_plan_files_touched_in_branch(
                                list(self._task_files or []),
                                self._base_commit,
                                branch_head2.strip(),
                                self.git_ops,
                                task_id=self.task_id,
                            )
                            if not check.not_touched:
                                narrowing_succeeded = True
                if check.not_touched:
                    reason = (
                        f'{PLAN_FILES_NOT_TOUCHED_REASON_PREFIX}: '
                        f'{", ".join(check.not_touched)}. '
                        f'The architect declared these files but no commit '
                        f'on the branch touched them.  Implementation has '
                        f'not delivered against the plan.'
                    )
                    _emit_merge_attempt(
                        self.event_store, self.task_id,
                        'plan_files_not_touched',
                    )
                    return await self._mark_blocked(
                        reason,
                        merge_phase=merge_phase,
                        escalate_to_human=True,
                    )
                if narrowing_succeeded:
                    # Distinguish honest over-declaration (handled here)
                    # from human-triage-required (emitted just above).
                    _emit_merge_attempt(
                        self.event_store, self.task_id,
                        'plan_files_narrowed',
                    )

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
        await register_and_enqueue_merge_request(
            self.merge_queue, merge_request, self.event_store, self.merge_inflight_registry,
        )

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
        # Drop-guard short-circuit: a real merger-drop is the human-judgement
        # case the gate exists for.  Steward mediation (e.g. mutating plan.json
        # to silence the gate) would undermine the safeguard, so skip the L0
        # steward path entirely and submit an L1 immediately.
        from orchestrator.merge_queue import (
            DROPPED_PLAN_TARGETS_REASON_PREFIX,
            POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
            POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX,
            TRANSIENT_INFRA_REASON_PREFIX,
        )
        if result.reason.startswith(DROPPED_PLAN_TARGETS_REASON_PREFIX):
            self._write_merge_failure_review('dropped_plan_targets', result.reason)
            return await self._mark_blocked(
                result.reason,
                merge_phase=merge_phase,
                escalate_to_human=True,
            )
        # Decision-2 short-circuit: post-merge equivalence is also a
        # human-judgement case (conflict-resolution drops / rebase
        # regressions); same L1-not-steward routing.
        if result.reason.startswith(POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX):
            self._write_merge_failure_review(
                'post_merge_equivalence_failed', result.reason,
            )
            return await self._mark_blocked(
                result.reason,
                merge_phase=merge_phase,
                escalate_to_human=True,
            )
        # Post-merge unscoped type-check short-circuit: a union break caught
        # after the merge landed is also a human-judgement fix-forward case;
        # same L1-not-steward routing so the steward cannot mask the signal.
        if result.reason.startswith(POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX):
            self._write_merge_failure_review(
                'post_merge_pyright_broken', result.reason,
            )
            return await self._mark_blocked(
                result.reason,
                merge_phase=merge_phase,
                escalate_to_human=True,
            )
        # Transient-infra short-circuit: the merge worker already pruned stale
        # merge worktrees and retried the verify, and it still hit ENOSPC.  Go
        # straight to a human-facing L1 tagged ``infra_issue`` (not the
        # steward — it can't free disk).  The durable ref + infra_issue
        # category let the escalation-watcher auto-resolve if the disk has
        # recovered by read-time.
        if result.reason.startswith(TRANSIENT_INFRA_REASON_PREFIX):
            self._write_merge_failure_review('transient_infra', result.reason)
            return await self._mark_blocked(
                result.reason,
                merge_phase=merge_phase,
                escalate_to_human=True,
                category='infra_issue',
            )
        # Fix 3 — capture the merge-queue blocked reason so the merge-phase
        # loop can fingerprint it for the thrash check before resubmitting.
        self._last_merge_block_reason = result.reason
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

    def _submit_halt_owning_escalation(self, esc: Escalation) -> None:
        """Submit a halt-owning escalation and register halt ownership.  Non-waiting.

        This is the load-bearing submit → set_halt_owner ordering shared by:
          - ``_submit_halt_escalation_and_wait`` (single-task path; follows with an await)
          - ``_escalate_train_halt`` (train path; does not await — tip stays BLOCKED and
            re-dispatches when the L1 is resolved and the queue is unhalted)

        Order is significant: set_halt_owner MUST follow a successful submit.
        A registered owner with no pending escalation cannot be resolved by
        _on_escalation_resolved, so the halt would be permanent.  If submit
        raises, the exception propagates before set_halt_owner is reached and
        no orphan halt is registered.

        Callers must guard with ``if self.escalation_queue:`` before calling.
        """
        assert self.escalation_queue is not None, (
            '_submit_halt_owning_escalation requires escalation_queue; '
            'callers must guard with `if self.escalation_queue:`'
        )
        self.escalation_queue.submit(esc)  # propagates on failure; set_halt_owner NOT reached
        if self.merge_worker is not None:
            self.merge_worker.set_halt_owner(esc.id)

    async def _submit_halt_escalation_and_wait(self, esc: Escalation) -> None:
        """Submit a halt-owning escalation, register ownership, and wait for resolution.

        Delegates the load-bearing submit → set_halt_owner ordering to
        ``_submit_halt_owning_escalation``, then awaits ``_escalation_event``.

        On cancellation or any other BaseException raised after set_halt_owner
        (including event_store.emit failures or task cancellation during the
        await), releases the halt via unhalt_wip(reason='workflow_cancelled')
        iff this workflow still owns it, then re-raises so the cancellation
        propagates to the caller's Task.
        """
        assert self.escalation_queue is not None, (
            '_submit_halt_escalation_and_wait requires escalation_queue; '
            'callers must guard with `if self.escalation_queue:`'
        )
        self._submit_halt_owning_escalation(esc)  # submit + set_halt_owner (load-bearing order)
        # try/except starts here so emit, event setup, AND the await are all
        # protected — any BaseException after set_halt_owner triggers cleanup.
        try:
            if self.event_store:
                self.event_store.emit(
                    EventType.escalation_created,
                    task_id=self.task_id, phase=self.state.value,
                    data={
                        'escalation_id': esc.id,
                        'category': esc.category,
                        'severity': esc.severity,
                        'summary': esc.summary[:200],
                    },
                )
            if self._escalation_event is None:
                self._escalation_event = asyncio.Event()
            self._escalation_event.clear()
            await self._escalation_event.wait()
        except BaseException:
            if (
                self.merge_worker is not None
                and self.merge_worker.is_halt_owner(esc.id)
            ):
                self.merge_worker.unhalt_wip(reason='workflow_cancelled')
            raise

    def _build_wip_halt_escalation_text(
        self,
        status: str,
        result,
        *,
        branch_name: str | None = None,
        train_id: str | None = None,
    ) -> tuple[str, str, str]:
        """Build ``(category, summary, detail)`` for a WIP-halt level-1 escalation.

        Shared by both the single-task handlers (``branch_name=...``) and the
        train consumer ``_escalate_train_halt`` (``train_id=...``), so that
        human-readable recovery instructions live in one place and cannot drift.

        The task-context label (``'Merge for task X (branch Y)'`` vs
        ``'Train merge for task X (train Y)'``) and one train-specific
        split-brain note for ``done_wip_recovery`` are the only per-path
        variations.

        Parameters
        ----------
        status:
            The ``MergeOutcome.status`` string.  Handled: ``'unmerged_state'``,
            ``'done_wip_recovery'``, ``'wip_recovery_no_advance'``,
            ``'wip_halted'`` (default branch).
        result:
            The ``MergeOutcome`` object (for ``overlap_files``,
            ``recovery_branch``).
        branch_name:
            Single-task context — the local branch being merged.
            Mutually exclusive with ``train_id``.
        train_id:
            Train context — the train identifier string.
            Mutually exclusive with ``branch_name``.

        Returns
        -------
        ``(category, summary, detail)`` :
            Ready for ``Escalation(category=..., summary=..., detail=...)``.
        """
        is_train = train_id is not None
        task_id = self.task_id

        # Context label used in the detail intro sentence.
        if is_train:
            merge_ctx = f'Train merge for task {task_id} (train {train_id!r})'
        elif branch_name is not None:
            merge_ctx = f'Merge for task {task_id} (branch {branch_name})'
        else:
            merge_ctx = f'Merge for task {task_id}'

        if status == 'unmerged_state':
            category = 'unmerged_state'
            if is_train:
                summary = (
                    f'Train member {task_id} blocked: '
                    f'project_root has unresolved (UU/AA/DD) markers'
                )
            else:
                summary = (
                    'project_root has unresolved (UU/AA/DD) markers — '
                    'merge queue halted'
                )
            detail = (
                f'{merge_ctx} was blocked because project_root already has unresolved '
                f'merge conflicts (UU/AA/DD markers) from a prior, unrelated event — '
                f'the merge queue refuses to stash/advance over a partially resolved tree.\n\n'
                f'Action required: inspect ``git status`` in project_root, '
                f'resolve the existing merge state (``git mergetool`` / edit '
                f'the conflicted files / ``git reset`` to abandon the prior '
                f'merge), then resolve this escalation to un-halt the merge queue.\n\n'
                f'Manual intervention required — do NOT let automated tooling '
                f'resolve this escalation.'
            )

        elif status == 'done_wip_recovery':
            category = 'wip_conflict'
            recovery_branch = result.recovery_branch or '(unknown)'
            if is_train:
                summary = (
                    f'Train member {task_id}: stash pop conflict — '
                    f'WIP preserved on {recovery_branch}'
                )
                wip_clause = (
                    'uncommitted WIP produced conflicts '
                    '(split-brain: merge landed, members NOT flipped to done).'
                )
                # Task 1599 / Suggestion 4: the merge is already on main.
                # Instruct the human to flip all members done before resolving
                # so the re-dispatch does not re-attempt the already-landed merge.
                post_resolve = (
                    'Note: the merge commit is already on main.  Before resolving this '
                    'escalation, ensure all train members are marked done — the train '
                    're-dispatches on unhalt and re-entering the merge queue with an '
                    'already-merged tip may produce a confusing split result.\n\n'
                    'Resolve this escalation to un-halt the merge queue.'
                )
            else:
                summary = f'Stash pop conflict — WIP preserved on {recovery_branch}'
                wip_clause = 'your uncommitted WIP produced conflicts.'
                post_resolve = 'Resolve this escalation to un-halt the merge queue.'
            detail = (
                f'{merge_ctx} landed on main successfully, but the stash pop of '
                f'{wip_clause}\n\n'
                f'Your WIP has been preserved on branch: {recovery_branch}\n\n'
                f'To recover:\n'
                f'  git checkout {recovery_branch}\n'
                f'  # Review and cherry-pick or reapply your changes\n\n'
                f'{post_resolve}'
            )

        elif status == 'wip_recovery_no_advance':
            category = 'wip_conflict'
            recovery_branch = result.recovery_branch or '(unknown)'
            if is_train:
                summary = (
                    f'Train member {task_id}: stash pop conflict (no advance) — '
                    f'WIP on {recovery_branch}'
                )
                no_advance_intro = (
                    f'{merge_ctx} did NOT advance main (CAS failure path). '
                    f'A stash pop conflict occurred, leaving'
                )
            else:
                summary = (
                    f'Stash pop conflict (merge did not advance) — '
                    f'WIP on {recovery_branch}'
                )
                no_advance_intro = (
                    f'{merge_ctx} did NOT advance main. '
                    f'A stash pop conflict occurred after a CAS failure, leaving'
                )
            detail = (
                f'{no_advance_intro} the working tree in an unresolvable state.\n\n'
                f'Your WIP has been preserved on branch: {recovery_branch}\n\n'
                f'The merge queue has been halted. To recover:\n'
                f'  git checkout {recovery_branch}\n'
                f'  # Review and reapply your changes to a clean branch\n\n'
                f'Manual intervention required — do NOT let automated tooling '
                f'resolve this escalation. Resolve this escalation to un-halt '
                f'the merge queue.'
            )

        else:
            # wip_halted (and any future halt-inducing status the probe catches)
            category = 'wip_conflict'
            overlap = result.overlap_files or []
            if is_train:
                summary = (
                    f'Train member {task_id}: WIP overlaps merge diff: '
                    + ', '.join(overlap[:5])
                )
                subject_ctx = f'train member {task_id} (train {train_id!r})'
            else:
                summary = f'WIP overlaps merge diff: {", ".join(overlap[:5])}'
                branch_str = f' (branch {branch_name})' if branch_name else ''
                subject_ctx = f'task {task_id}{branch_str}'
            detail = (
                f'Merge for {subject_ctx} was blocked because uncommitted '
                f'work in project_root overlaps the merge diff.\n\n'
                f'Overlapping files:\n'
                + '\n'.join(f'  - {f}' for f in overlap)
                + '\n\nAction required: commit or stash the WIP, then resolve this '
                'escalation to un-halt the merge queue and retry.'
            )

        return category, summary, detail

    async def _escalate_train_halt(
        self, result, train_id: str,
    ) -> WorkflowOutcome:
        """Handle a train WIP-halt outcome: build per-status L1, own the halt, block tip.

        Called from ``_maybe_enqueue_group_merge`` when the orphan-halt probe fires::

            merge_worker is not None
            and merge_worker.is_wip_halted
            and merge_worker.halt_owner_esc_id is None

        Covers all four ``_map_advance_failure`` halt-inducing statuses:
          - ``wip_halted``             → category='wip_conflict'  (WIP overlaps diff)
          - ``done_wip_recovery``      → category='wip_conflict'  (merge landed; stash pop conflict)
          - ``wip_recovery_no_advance``→ category='wip_conflict'  (CAS failure; no advance)
          - ``unmerged_state``         → category='unmerged_state' (pre-existing UU/AA/DD)

        Unlike the single-task ``_handle_wip_conflict`` etc., this helper does NOT
        await escalation resolution — the train tip stays BLOCKED and re-dispatches
        once the halt is cleared.  This avoids reintroducing the cancellation-orphan
        surface that task 1448 hardened for inline-waiting coroutines.

        When ``escalation_queue is None`` (config-absent deployment), logs a warning
        and falls back to plain BLOCKED with no owner registered.

        Auto-recovery parity (task 1599):
          Resolving the returned L1 triggers harness._on_escalation_resolved →
          unhalt_wip() because ``_submit_halt_owning_escalation`` registered this
          workflow as halt owner.  harness._rehydrate_merge_halt re-owns the L1
          across restarts (category in {wip_conflict, unmerged_state}).
        """
        status = result.status
        category, summary, detail = self._build_wip_halt_escalation_text(
            status, result, train_id=train_id,
        )

        reason = (
            f'Train halt: merge for task {self.task_id} (train {train_id!r}) '
            f'halted the merge queue ({status})'
        )
        logger.warning(
            'Task %s: train halt (%s) — %s',
            self.task_id, status,
            'creating level-1 escalation' if self.escalation_queue
            else 'escalation_queue=None, cannot register halt owner',
        )

        if self.escalation_queue:
            from escalation.models import Escalation

            train_state = await self._build_train_state()

            # Defensive re-check: the consumer's orphan-halt probe validated
            # halt_owner_esc_id is None before calling us, but _build_train_state()
            # contains an await and another coroutine could (theoretically) set the
            # owner during that window.  In the current serial merge-worker design
            # this window is unreachable, but re-checking prevents a hard crash
            # from the 'owner already set' assertion inside set_halt_owner if the
            # worker ever becomes concurrent.  Mirror the escalation_queue=None
            # fallback: log a warning and fall through to plain BLOCKED.
            if (
                self.merge_worker is not None
                and self.merge_worker.halt_owner_esc_id is not None
            ):
                logger.warning(
                    'Task %s: halt owner set concurrently during _build_train_state '
                    '(owner: %r) — skipping duplicate set_halt_owner; plain BLOCKED',
                    self.task_id, self.merge_worker.halt_owner_esc_id,
                )
            else:
                esc = Escalation(
                    id=self.escalation_queue.make_id(self.task_id),
                    task_id=self.task_id,
                    agent_role='orchestrator',
                    severity='blocking',
                    category=category,
                    summary=summary,
                    detail=detail,
                    suggested_action='manual_intervention',
                    level=1,
                    worktree=str(self.worktree) if self.worktree else None,
                    workflow_state=self.state.value,
                    train_state=train_state,
                )
                self._submit_halt_owning_escalation(esc)
                logger.info(
                    'Task %s: train halt L1 %r submitted and halt ownership registered',
                    self.task_id, esc.id,
                )
        else:
            logger.warning(
                'Task %s: merge queue is halted (train %r) but escalation_queue is '
                'None — halt owner cannot be registered; manual unhalt_merge_queue '
                'required to clear the orphan halt',
                self.task_id, train_id,
            )

        return await self._mark_blocked(reason, detail=detail, skip_escalation=True)

    async def _handle_wip_conflict(
        self, result, branch_name: str,
    ) -> WorkflowOutcome:
        """Handle a wip_halted merge outcome: create level-1 escalation and wait.

        The merge did NOT land — WIP in project_root overlaps the merge diff.
        After the human resolves (commits/stashes WIP), the task retries the merge.
        """
        category, summary, detail = self._build_wip_halt_escalation_text(
            result.status, result, branch_name=branch_name,
        )
        logger.warning(f'Task {self.task_id}: WIP overlap — creating level-1 escalation')

        if self.escalation_queue:
            from escalation.models import Escalation

            esc = Escalation(
                id=self.escalation_queue.make_id(self.task_id),
                task_id=self.task_id,
                agent_role='orchestrator',
                severity='blocking',
                category=category,
                summary=summary,
                detail=detail,
                suggested_action='manual_intervention',
                level=1,
                worktree=str(self.worktree) if self.worktree else None,
                workflow_state=self.state.value,
            )
            await self._submit_halt_escalation_and_wait(esc)
            logger.info(f'Task {self.task_id}: WIP conflict resolved — retrying merge')

        return WorkflowOutcome.REQUEUED

    async def _handle_wip_recovery(self, result) -> WorkflowOutcome:
        """Handle a done_wip_recovery merge outcome: merge landed but WIP conflicted.

        The merge IS on main, but the user's stashed WIP conflicted during pop.
        WIP has been preserved on a recovery branch. Create a level-1 escalation
        to inform the human, then return DONE (the task's merge succeeded).
        """
        # Capture the on-main SHA so the success-path set_task_status('done', ...)
        # in run() can build a valid done_provenance. Without this, _merge_sha
        # stays None and fused-memory rejects the transition with "kind required".
        if result.merge_sha is not None:
            self._merge_sha = result.merge_sha
        recovery_branch = result.recovery_branch or '(unknown)'
        category, summary, detail = self._build_wip_halt_escalation_text(
            result.status, result,
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
                category=category,
                summary=summary,
                detail=detail,
                suggested_action='manual_intervention',
                level=1,
                worktree=str(self.worktree) if self.worktree else None,
                workflow_state=self.state.value,
            )
            await self._submit_halt_escalation_and_wait(esc)
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
        category, summary, detail = self._build_wip_halt_escalation_text(
            result.status, result,
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
                category=category,
                summary=summary,
                detail=detail,
                suggested_action='manual_intervention',
                level=1,
                worktree=str(self.worktree) if self.worktree else None,
                workflow_state=self.state.value,
            )
            await self._submit_halt_escalation_and_wait(esc)
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
        category, summary, detail = self._build_wip_halt_escalation_text(
            result.status, result, branch_name=branch_name,
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
                category=category,
                summary=summary,
                detail=detail,
                suggested_action='manual_intervention',
                level=1,
                worktree=str(self.worktree) if self.worktree else None,
                workflow_state=self.state.value,
            )
            await self._submit_halt_escalation_and_wait(esc)
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
            if self.mcp is not None:
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

        # Session-resume lifecycle: if the harness recovered a sidecar that
        # matches the role we're about to invoke, resume the prior session
        # by passing resume_session_id to invoke_with_cap_retry.  Otherwise,
        # allocate a fresh UUID up-front via --session-id so a future restart
        # can find and resume this session.  The sidecar is written before the
        # subprocess starts and cleared in the finally below — its presence
        # ⇔ "agent was in flight when the orchestrator exited".
        #
        # Prompt-substitution ownership: cli_invoke owns the resume-continuation
        # prompt swap (CRASH_RECOVERY_RESUME_PROMPT = 'continue').  Workflow
        # always passes the real task prompt so that cli_invoke's
        # original_prompt capture is correct for any fresh-fallback invocation.
        resume_session_id: str | None = None
        if (
            self._pending_resume_session_id
            and self._pending_resume_role == role.name
        ):
            session_id_val = self._pending_resume_session_id
            resume_session_id = session_id_val
            self._pending_resume_session_id = None
            self._pending_resume_role = None
            logger.info(
                'Task %s [%s]: resuming prior session %s via --resume',
                self.task_id, role.name, session_id_val,
            )
        else:
            session_id_val = str(uuid.uuid4())

        if self.artifacts is not None:
            self.artifacts.write_agent_session(
                session_id_val, role.name, datetime.now(UTC).isoformat(),
            )

        started_at = datetime.now(UTC).isoformat()
        try:
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
                session_id=session_id_val,
                resume_session_id=resume_session_id,
                # Judge always hits Claude API — propagating ANTHROPIC_BASE_URL
                # routes it through vLLM where max_model_len causes
                # ServerDisconnectedError after 2 tool-use rounds (3cd380a079).
                # Cap hits on Claude API are handled by UsageGate account failover
                # (wired in runner.py for eval mode).
                env_overrides=(self.config.env_overrides or None) if role.name in ('implementer', 'debugger') else None,
            )
        finally:
            if self.artifacts is not None:
                self.artifacts.clear_agent_session()
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
        level-1 (consumed by the auto-watcher), indicating the task should be blocked.

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

        # Collect resolutions (filter ensures resolution is non-None str for join)
        resolutions = [
            e.resolution
            for e in self.escalation_queue.get_by_task(self.task_id)
            if e.status == 'resolved' and e.resolution is not None
        ]
        return '\n'.join(resolutions)

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
        await self._reconcile_metadata_files_for_done()
        try:
            await self.scheduler.mark_done(
                self.task_id,
                kind='found_on_main',
                sha=main_sha,
                note=(
                    'branch already on main at workflow start '
                    '(pre-PLAN recovery)'
                ),
            )
        except SetTaskStatusRejected as exc:
            logger.error(
                'Task %s: pre-PLAN recovery mark_done rejected — %s: %s',
                self.task_id, exc.error_code, exc.raw,
            )
            return await self._mark_blocked(
                f'Pre-PLAN recovery rejected: {exc.error_code} — {exc.raw}',
                escalate_to_human=True,
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

    def _handle_cancelled_terminal_exit(
        self, exc: TerminalExitRejection,
    ) -> WorkflowOutcome | None:
        """Return ``WorkflowOutcome.CANCELLED`` when *exc* signals an authoritative
        user/manual cancellation (``exc.old_status == 'cancelled'``), else ``None``.

        Centralises the predicate and log message used at two sites — run()'s
        ``SetTaskStatusRejected`` guard and ``_handle_terminal_exit_on_block``'s
        sub-case 0 — so they cannot drift.
        """
        if exc.old_status == 'cancelled':
            logger.info(
                'Task %s: cancelled out-of-band; aborting gracefully '
                '(no reopen, no escalation)',
                self.task_id,
            )
            return WorkflowOutcome.CANCELLED
        return None

    async def _handle_terminal_exit_on_block(
        self,
        exc: TerminalExitRejection,
        reason: str,
        detail: str,
    ) -> WorkflowOutcome | None:
        """Detect ``update_task(status='done')`` bypass after a terminal-exit rejection.

        When ``set_task_status('blocked')`` raises ``TerminalExitRejection``
        the row is already terminal. Three sub-cases, branching on ``exc.old_status``:

        0. Cancelled (``old_status == 'cancelled'``) — a user/manual cancellation
           landed out-of-band. This is authoritative; the workflow must NOT reopen
           the row or file any escalation. Log an info line and return
           ``WorkflowOutcome.CANCELLED`` immediately (before any ``get_task``
           round-trip or provenance read).
        1. Legitimate done (``old_status == 'done'``, provenance commit reachable
           from main) — return ``None`` so the existing flow continues; the
           post-steward branch at ``current == 'done'`` returns DONE.
        2. Bypass (``old_status == 'done'``, provenance missing or commit not on
           main) — reopen the row via
           ``set_task_status('blocked', reopen_reason='bypass detected: …')``,
           file an L1 with ``category='bypass_done'``, and return
           ``WorkflowOutcome.BLOCKED``.
        """
        # Sub-case 0: authoritative user/manual cancellation — abort gracefully.
        outcome = self._handle_cancelled_terminal_exit(exc)
        if outcome is not None:
            return outcome

        task = await self.scheduler.get_task(self.task_id)
        # Scheduler.get_task normalises task['metadata'] to a dict at the
        # boundary (via _normalize_task_metadata) so we can read it directly.
        metadata: dict = (task.get('metadata') or {}) if isinstance(task, dict) else {}

        provenance = metadata.get('done_provenance') if metadata else None
        commit = ''
        if isinstance(provenance, dict):
            commit = str(provenance.get('commit') or '').strip()

        if commit:
            try:
                main_sha = await self.git_ops.get_main_sha()
                if await self.git_ops.is_ancestor(commit, main_sha):
                    logger.info(
                        'Task %s: terminal-exit on _mark_blocked but provenance '
                        'commit %s is reachable from main — legitimate done',
                        self.task_id, commit[:12],
                    )
                    return None
            except Exception:
                logger.warning(
                    'Task %s: ancestor check failed during bypass detection',
                    self.task_id, exc_info=True,
                )

        logger.warning(
            'Task %s: terminal-exit on _mark_blocked with missing or off-main '
            'done_provenance (commit=%s); reopening + filing L1 bypass_done',
            self.task_id, commit[:12] if commit else '<none>',
        )

        bypass_summary = f'bypass detected: {reason[:160]}'
        try:
            await self.scheduler.set_task_status(
                self.task_id, 'blocked', reopen_reason=bypass_summary,
            )
        except Exception:
            logger.exception(
                'Task %s: reopen with reopen_reason failed; continuing to L1',
                self.task_id,
            )

        if self.escalation_queue:
            from escalation.models import Escalation
            l1 = Escalation(
                id=self.escalation_queue.make_id(self.task_id),
                task_id=self.task_id,
                agent_role='orchestrator',
                severity='blocking',
                category='bypass_done',
                summary=(
                    f'Mark-done bypass detected for task {self.task_id}: '
                    f'row is {exc.old_status!r}, provenance commit not on main'
                ),
                detail=(
                    f'reason: {reason}\n'
                    f'detail: {detail}\n'
                    f'old_status: {exc.old_status}\n'
                    f'evidence_commit: {commit or "<none>"}\n'
                ),
                suggested_action='investigate_bypass_done',
                worktree=str(self.worktree) if self.worktree else None,
                workflow_state=self.state.value,
                level=1,
            )
            self.escalation_queue.submit(l1)
            if self.event_store:
                self.event_store.emit(
                    EventType.escalation_created,
                    task_id=self.task_id, phase=self.state.value,
                    data={
                        'escalation_id': l1.id, 'category': 'bypass_done',
                        'severity': 'blocking', 'level': 1,
                        'summary': l1.summary[:200],
                    },
                )

        return WorkflowOutcome.BLOCKED

    async def _maybe_cleanup_done_worktree(self) -> None:
        """Clean up worktree+branch only when the branch is reachable from main.

        Forensics 2026-05-08: when an agent bypassed the merge via
        ``update_task(status='done')``, the workflow returned DONE while the
        branch HEAD was still off-main. Cleanup deleted the unmerged branch;
        only the loose-object grace period kept the work recoverable.

        Mirrors the predicate at ``run()``'s ghost-loop early exit.
        """
        if (
            self.state != WorkflowState.DONE
            or self.worktree is None
            or self._worktree_external
        ):
            return
        branch_name = self.task_id
        try:
            rc, branch_head_raw, _ = await _run(
                ['git', 'rev-parse', 'HEAD'], cwd=self.worktree,
            )
        except Exception:
            logger.warning(
                'Task %s: rev-parse HEAD failed in cleanup gate; preserving worktree',
                self.task_id, exc_info=True,
            )
            return
        if rc != 0:
            logger.warning(
                'Task %s: rev-parse HEAD returned %d in cleanup gate; '
                'preserving worktree+branch for inspection',
                self.task_id, rc,
            )
            return
        branch_head = branch_head_raw.strip()
        try:
            main_sha = await self.git_ops.get_main_sha()
        except Exception:
            logger.warning(
                'Task %s: get_main_sha failed in cleanup gate; preserving worktree',
                self.task_id, exc_info=True,
            )
            return
        if await self.git_ops.is_ancestor(branch_head, main_sha):
            await self.git_ops.cleanup_worktree(self.worktree, branch_name)
            return
        # Fallback: branch tip is a pre-rebase duplicate that is NOT an ancestor
        # of main, but the recorded merge commit (_merge_sha) IS on main.  The
        # task's work was merged via a rebase-on-merge flow.  Reclaim the
        # regenerable build cache while preserving the worktree+branch for forensics.
        merge_sha = self._merge_sha
        if merge_sha and await self.git_ops.is_ancestor(merge_sha, main_sha):
            reclaimed = await self.git_ops.reclaim_worktree_build_artifacts(
                self.worktree,
            )
            logger.warning(
                'Task %s: branch HEAD %s not on main but merge commit %s is on '
                'main — reclaimed build artifacts %s; preserving worktree+branch',
                self.task_id, branch_head[:12], merge_sha[:12],
                [str(p) for p in reclaimed],
            )
            return
        logger.warning(
            'Task %s: state=DONE but branch HEAD %s not reachable from main %s '
            '— preserving worktree+branch for inspection',
            self.task_id, branch_head[:12], main_sha[:12],
        )

    async def _mark_blocked(
        self, reason: str, *, detail: str = '',
        skip_escalation: bool = False,
        merge_phase: bool = False,
        escalate_to_human: bool = False,
        suggested_action: str = 'investigate_and_retry',
        category: str = 'task_failure',
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
            _status_set_ok = False
            try:
                await self.scheduler.set_task_status(self.task_id, 'blocked')
                _status_set_ok = True
            except TerminalExitRejection as exc:
                bypass_outcome = await self._handle_terminal_exit_on_block(
                    exc, reason, detail or reason,
                )
                if bypass_outcome is not None:
                    return bypass_outcome
                # Legitimate done — fall through; the existing post-steward
                # flow below handles current==done by returning DONE.
            if _status_set_ok:
                self._spawn_dry_run_unblock(reason, detail or reason)
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
            # Fix C short-circuit: the caller determined this failure is not
            # resolvable by the steward (e.g. false premise, unactionable spec,
            # confirmed cycle).  Skip the L0 entirely — a stopped steward
            # cannot consume it — and submit only the human-facing L1.
            # This also prevents duplicate design_concern escalations in the
            # queue (an L0 the steward can never act on + an L1 for the same
            # report), keeping dashboards clean and preventing orphan-L0 reaper
            # noise.  The unactionable handler follows the same pattern.
            if escalate_to_human:
                await self._ensure_l1_escalation_for_blocked(
                    reason, detail or reason, category=category,
                )
                return WorkflowOutcome.BLOCKED

            # Don't create a duplicate if level-1 already pending
            if not self.escalation_queue.has_open_l1(self.task_id):
                from escalation.models import Escalation

                esc = Escalation(
                    id=self.escalation_queue.make_id(self.task_id),
                    task_id=self.task_id,
                    agent_role='orchestrator',
                    severity='blocking',
                    category=category,
                    summary=reason[:200],
                    detail=detail or reason,
                    suggested_action=suggested_action,
                    worktree=str(self.worktree) if self.worktree else None,
                    workflow_state=self.state.value,
                )
                self.escalation_queue.submit(esc)

                if self.event_store:
                    self.event_store.emit(
                        EventType.escalation_created,
                        task_id=self.task_id, phase=self.state.value,
                        data={'escalation_id': esc.id, 'category': category,
                              'severity': 'blocking', 'summary': reason[:200]},
                    )

            # Capture window-start for the broadened dismiss-with-terminate
            # guard below.  Any L0 whose resolved_at falls inside this window
            # is attributable to the current steward invocation — including
            # follow-on L0s the steward chains and dismisses itself, not just
            # the L0 the workflow itself just submitted (the original narrow
            # Fix A guard tracked only that one).
            steward_window_start = datetime.now(UTC).isoformat()

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
                    # Guard: if the steward escalated to L1 (auto-watcher consumer),
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

                    # Fix A (broadened): detect dismiss-with-terminate for
                    # ANY L0 on this task whose resolved_at falls inside the
                    # current steward invocation window — not just the L0
                    # the workflow itself just submitted (the original narrow
                    # Fix A guard tracked only that one).
                    #
                    # The steward may chain a follow-on L0 (e.g. an
                    # ``infra_issue`` raised while resolving the original
                    # ``task_failure``) and dismiss-with-terminate THAT one.
                    # That is still "agent gives up", so halt and submit an
                    # L1 instead of re-pending the task.
                    dismissed_l0s = self.escalation_queue.get_by_task(
                        self.task_id, status='dismissed', level=0,
                    )
                    recent_dismissals = [
                        e for e in dismissed_l0s
                        if e.resolved_at is not None
                        and e.resolved_at >= steward_window_start
                    ]
                    if recent_dismissals:
                        logger.warning(
                            'Task %s: steward dismissed %d L0(s) during this '
                            'invocation (ids=%s) — halting, escalating to L1',
                            self.task_id, len(recent_dismissals),
                            [e.id for e in recent_dismissals],
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
        await self._ensure_l1_escalation_for_blocked(reason, detail or reason, category=category)

        # Fix #2 — dismiss any still-pending L0 now that we are exiting BLOCKED.
        # The steward has finished (or never ran) and this workflow slot is
        # about to exit, so the L0's only consumer is dead.  Leaving it pending
        # strands an orphan that the orphan-L0 reaper later promotes into a
        # DUPLICATE L1 (esc-3576-234 in the 2026-05-29 incident) — pure churn,
        # since _ensure_l1_escalation_for_blocked above already filed the human-
        # facing L1.  Mirrors the defensive dismissal block at the
        # skip_escalation path above.  Guarded on escalation_queue because the
        # fall-through is also reached when no queue is wired.
        if self.escalation_queue:
            orphan_l0 = self.escalation_queue.get_by_task(
                self.task_id, status='pending', level=0,
            )
            for esc in orphan_l0:
                self.escalation_queue.resolve(
                    esc.id,
                    'Auto-dismissed: workflow exiting BLOCKED, steward done '
                    '(L1 filed for human handoff)',
                    dismiss=True,
                    resolved_by='auto-dismissed',
                )
            if orphan_l0:
                logger.info(
                    'Task %s: dismissed %d orphan L0(s) on BLOCKED fall-through '
                    '(steward consumer dead; L1 already filed)',
                    self.task_id, len(orphan_l0),
                )
        return WorkflowOutcome.BLOCKED

    def _durable_ref_suffix(self) -> str:
        """Durable git identifiers to append to an L1 escalation's detail.

        The originating worktree is ephemeral — task and merge worktrees are
        reaped (e.g. by the merge queue or a disk-pressure prune) well before
        a human reads a human-facing L1.  So cite refs that survive: the task
        branch and the SHAs bracketing its work.  ``tip`` is the merge commit
        SHA when the merge already landed, else a label pointing at the
        branch's current HEAD.
        """
        branch = f'{self.config.git.branch_prefix}{self.task_id}'
        base = (self._base_commit or '')[:12] or 'unknown'
        tip = (self._merge_sha or '')[:12] or f'{branch}@HEAD'
        return f'\n\n[durable refs] branch={branch} base={base} tip={tip}'

    async def _build_train_state(self) -> TrainState | None:
        """Build per-train context for L1 escalation payloads (PRD § 9.8).

        Returns None when this task carries no valid metadata.train (non-train
        task or malformed metadata).  When valid, returns::

            {'id': str, 'order': int,
             'parked_members': list[str],  # siblings at merge-deferred, excl. self
             'failing_member': str}        # this task's id

        Fast path: if metadata.train.members is a list, call get_statuses on
        those ids and filter.  Fallback: scan get_tasks() for tasks whose
        metadata.train.id matches; status is read directly from the task dicts
        (avoiding a redundant get_statuses round-trip — get_tasks already embeds
        per-task status in each dict).

        Reuses the TrainMembership cast/isinstance convention from task 1522
        (git_ops.py:54-62); cast and TrainMembership are already imported.
        """
        metadata = self.task.get('metadata') or {}
        train_meta = metadata.get('train')
        if not isinstance(train_meta, dict):
            return None
        train = cast(TrainMembership, train_meta)

        train_id = train.get('id')
        train_order = train.get('order')
        if train_id is None or train_order is None:
            return None

        # Discover candidate sibling task ids and their statuses.
        members_cache = train.get('members')
        if isinstance(members_cache, list):
            # Fast path — use the cached members list; fetch statuses from the server.
            candidates: list[str] = [str(m) for m in members_cache]
            statuses, _ = await self.scheduler.get_statuses(candidates)
        else:
            # Fallback scan — discover siblings via get_tasks().
            # Status is already embedded in each task dict; avoid a second round-trip.
            tasks = await self.scheduler.get_tasks()
            statuses: dict[str, str] = {str(t['id']): str(t.get('status', 'unknown')) for t in tasks}
            candidates = [
                str(t['id'])
                for t in tasks
                if isinstance((t.get('metadata') or {}).get('train'), dict)
                and cast(TrainMembership, (t.get('metadata') or {}).get('train')).get('id') == train_id
            ]

        parked_members = [
            c for c in candidates
            if c != self.task_id and statuses.get(c) == 'merge-deferred'
        ]
        return cast('TrainState', {
            'id': train_id,
            'order': train_order,
            'parked_members': parked_members,
            'failing_member': self.task_id,
        })

    async def _ensure_l1_escalation_for_blocked(
        self, reason: str, detail: str, *, category: str = 'task_failure',
    ) -> None:
        """Submit a level-1 escalation if none is open for this task.

        Called from BLOCKED-return paths so a human is signaled when
        automated handlers cannot make progress.  Idempotent — deduped
        via ``has_open_l1``.

        Cites durable refs (branch + base/tip SHAs) in ``detail`` and leaves
        ``worktree=None``: the human reads this after the worktree may be
        gone, so the ephemeral path is worse than useless.  (The L0 builder
        keeps ``worktree=`` — the steward is live and acts *in* that tree.)
        """
        if not self.escalation_queue:
            return
        if self.escalation_queue.has_open_l1(self.task_id):
            return
        from escalation.models import Escalation

        # PRD § 9.8 — inject per-train context for park-prefix derail escalations.
        # Non-train tasks get train_state=None (the Escalation field default).
        # This is the single L1 chokepoint (3 callers inside _mark_blocked at
        # ~4861/~4980/~5007), so it covers every park-prefix derail path.
        train_state = await self._build_train_state()

        esc = Escalation(
            id=self.escalation_queue.make_id(self.task_id),
            task_id=self.task_id,
            agent_role='orchestrator',
            severity='blocking',
            category=category,
            summary=f'Workflow blocked, no automated resolution path: {reason[:160]}',
            detail=(detail or reason) + self._durable_ref_suffix(),
            suggested_action='manual_intervention',
            worktree=None,
            workflow_state=self.state.value,
            level=1,
            train_state=train_state,
        )
        self.escalation_queue.submit(esc)
        if self.event_store:
            self.event_store.emit(
                EventType.escalation_created,
                task_id=self.task_id, phase=self.state.value,
                data={
                    'escalation_id': esc.id, 'category': category,
                    'severity': 'blocking', 'level': 1,
                    'summary': reason[:200],
                },
            )
        logger.warning(
            'Task %s: submitted L1 escalation %s for unresolved BLOCKED state',
            self.task_id, esc.id,
        )

    def _spawn_dry_run_unblock(self, reason: str, detail: str) -> None:
        """Fire-and-forget: spawn an autonomous dry-run investigation.

        Skips when unblock_auto is disabled. Wraps asyncio.create_task so
        _mark_blocked never awaits the investigation — it is a pure side-effect.
        Any exception inside run_dry_run_unblock is caught there and written
        as a fallback proposal entry, so unhandled task exceptions are closed.
        """
        if not getattr(self.config, 'unblock_auto', None) or not self.config.unblock_auto.enabled:
            return
        if self.worktree is None:
            logger.debug(
                'Task %s: skipping dry-run unblock hook — no worktree set',
                self.task_id,
            )
            return
        # Skip if an investigation for this task is already running (e.g. rapid
        # re-blocks during steward retries).  Duplicate proposals are unhelpful
        # and would multiply budget spend up to 600 s × N re-blocks.
        _task_name = f'unblock-auto-{self.task_id}'
        if any(t.get_name() == _task_name and not t.done() for t in self._background_tasks):
            logger.debug(
                'Task %s: dry-run investigation already in progress, skipping duplicate spawn',
                self.task_id,
            )
            return
        try:
            _task = asyncio.create_task(
                run_dry_run_unblock(
                    task_id=self.task_id,
                    worktree=str(self.worktree),
                    reason=reason,
                    detail=detail,
                    scheduler=self.scheduler,
                    mcp=self.mcp,
                    config=self.config,
                    event_store=getattr(self, 'event_store', None),
                ),
                name=_task_name,
            )
            self._background_tasks.add(_task)
            _task.add_done_callback(self._background_tasks.discard)
        except Exception as exc:
            logger.warning(
                'Task %s: failed to spawn dry-run unblock hook: %s',
                self.task_id, exc,
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

        if self.mcp is None:
            return
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
        if self.mcp is None:
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
        if self.mcp is None:
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

    async def _post_submit_tasks(self, arguments_list: list[dict]) -> None:
        """Fire-and-forget: POST all submit_task calls to the fused-memory MCP.

        Uses a single shared ``httpx.AsyncClient`` for the entire batch so only
        one TCP connection pool is opened per routing call regardless of how many
        suggestions are being submitted.  Runs inside ``asyncio.create_task`` so
        the caller returns immediately.

        Per-POST exceptions are caught and logged as warnings; a failure on one
        suggestion does not abort the remaining submissions.
        """
        if self.mcp is None:
            return
        try:
            import httpx as httpx_mod
            async with httpx_mod.AsyncClient() as client:
                for arguments in arguments_list:
                    try:
                        await client.post(
                            f'{self.mcp.url}/mcp/',
                            json={
                                'jsonrpc': '2.0',
                                'id': 1,
                                'method': 'tools/call',
                                'params': {
                                    'name': 'submit_task',
                                    'arguments': arguments,
                                },
                            },
                            timeout=10,
                        )
                    except Exception as exc:
                        logger.warning(
                            'Task %s: failed to submit curator task (fire-and-forget): %s',
                            self.task_id, exc,
                        )
        except Exception as exc:
            logger.warning(
                'Task %s: failed to open HTTP client for curator submits: %s',
                self.task_id, exc,
            )

    async def _route_review_suggestions_to_curator(self, reviews) -> None:
        """Route review suggestions directly to the curator intake (fire-and-forget).

        Inserts suggestions as CandidateTask tickets via ``submit_task`` MCP calls.
        The curator's ``_curator_worker`` drains tickets asynchronously; this method
        returns immediately after scheduling regardless of curator speed.

        Cross-submission dedup is handled by the curator R4 gate
        ``_check_escalation_idempotency`` which matches
        ``(escalation_id, suggestion_hash)`` metadata against existing
        non-cancelled tasks.  Re-submitting identical suggestions produces the
        same content_hash → same metadata → ticket marked 'combined' → 0 new rows.

        Must NOT call curate_batch — that dispatches invoke_with_cap_retry and
        can stall up to 31 minutes.  Uses asyncio.create_task +
        ``self._background_tasks`` (mirrors ``_spawn_dry_run_unblock``).

        When ``self.mcp`` is not configured (e.g. CLI / dry-run / test contexts),
        falls back to ``_escalate_suggestions`` if an escalation queue is available,
        or to ``_write_suggestions_to_memory`` otherwise. Note that
        ``_write_suggestions_to_memory`` itself requires ``self.mcp`` to write, so the
        ``(no mcp, no queue)`` case is a documented no-op — a WARNING is logged so the
        drop is auditable, but the suggestions are not persisted to any sink.
        """
        from orchestrator.review_suggestions.dedup import review_suggestion_payload_hash

        suggestions = reviews.suggestions
        if not suggestions:
            return

        # In-task dedup: compute the hash first so the cache check sits above
        # ALL routing branches (curator submit, escalation-queue fallback, no-op).
        # The escalation→resume cycle can re-enter REVIEW→DONE with an identical
        # suggestion set; short-circuiting here avoids N HTTP round-trips that
        # the curator R4 gate (_check_escalation_idempotency) would otherwise
        # have to absorb on the server side.
        content_hash = review_suggestion_payload_hash(suggestions)
        if self._last_routed_suggestion_hash == content_hash:
            logger.info(
                'Task %s: skipping duplicate curator route (hash=%s already sent)',
                self.task_id, content_hash,
            )
            return

        # Guard: fall back if MCP transport is unavailable so suggestions are
        # never silently dropped.  _escalate_suggestions is the real caller of
        # the steward-escalation fallback path.
        if self.mcp is None:
            if self.escalation_queue:
                self._escalate_suggestions(reviews)
                # Record the hash after a successful fallback so identical
                # re-entries skip the escalation queue entirely.  The queue's
                # own dedup would also catch it, but the in-task fast-path
                # avoids touching the queue at all.
                self._last_routed_suggestion_hash = content_hash
            else:
                logger.warning(
                    'Task %s: dropping %d review suggestion(s) — no MCP transport and no '
                    'escalation queue configured (CLI/dry-run/test no-op)',
                    self.task_id, len(suggestions),
                )
                # NOTE: do NOT set _last_routed_suggestion_hash here.
                # Suggestions were dropped, not routed.  The WARNING must fire
                # on every drop call to preserve audit visibility of this
                # pathological branch.
            return
        task_id = self.task_id
        project_root = str(self.config.project_root)

        all_arguments = []
        for suggestion in suggestions:
            cat = suggestion.get('category', '')
            loc = suggestion.get('location', '')
            desc = suggestion.get('description', '')
            title = f'[{cat}] {loc}: {desc[:60]}'

            all_arguments.append({
                'title': title,
                'description': desc,
                'details': json.dumps(suggestion),
                'priority': 'low',
                'project_root': project_root,
                'metadata': {
                    'spawned_from': task_id,
                    'spawn_context': 'review_suggestions',
                    'escalation_id': f'review-suggestions-{task_id}',
                    'suggestion_hash': content_hash,
                },
            })

        # Schedule the entire batch as one background task with a shared HTTP
        # client — avoids opening N separate connection pools for N suggestions.
        try:
            _task = asyncio.create_task(
                self._post_submit_tasks(all_arguments),
                name=f'route-suggestions-{task_id}',
            )
            self._background_tasks.add(_task)
            _task.add_done_callback(self._background_tasks.discard)
            # Record the hash only on successful scheduling so that a
            # create_task failure does not prevent a retry with the same
            # suggestion set.  The curator R4 gate is the durable backstop
            # for duplicates that slip through on a successful retry.
            self._last_routed_suggestion_hash = content_hash
        except Exception as exc:
            logger.warning(
                'Task %s: failed to schedule curator submits: %s',
                task_id, exc,
            )

        logger.info(
            'Task %s: scheduled %d suggestion(s) for direct curator intake '
            '(hash=%s)',
            task_id, len(suggestions), content_hash,
        )

    def _escalate_suggestions(self, reviews) -> None:
        """Submit review suggestions as an info escalation for steward triage.

        Retained as the steward-escalation fallback even though the primary
        call site now routes via ``_route_review_suggestions_to_curator``.
        The dedup helpers are shared with the curator path — both sites import
        from ``orchestrator.review_suggestions.dedup``.
        """
        from escalation.models import Escalation

        from orchestrator.review_suggestions.dedup import (
            find_prior_review_suggestion,
            hash_marker,
            review_suggestion_payload_hash,
        )

        suggestions = reviews.suggestions
        if not suggestions or not self.escalation_queue:
            return

        # Content fingerprint: skip if identical suggestions already escalated.
        content_hash = review_suggestion_payload_hash(suggestions)

        existing = self.escalation_queue.get_by_task(self.task_id)
        prior = find_prior_review_suggestion(existing, content_hash)
        if prior is not None:
            logger.info(
                'Task %s: skipping duplicate review_suggestions escalation '
                '(content hash %s matches %s)',
                self.task_id, content_hash, prior.id,
            )
            return

        detail = hash_marker(content_hash) + json.dumps(suggestions)

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
            # If the cancel event won, propagate that cancellation to the
            # underlying awaitable.  Without this, an enqueued merge request
            # becomes "orphaned": the worker still processes it and may halt
            # the queue with no workflow left to create the owning escalation.
            # See merge_queue.py: workers check req.result.cancelled() at entry
            # and before each halt_for_wip site.
            if not fut.done():
                fut.cancel()

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
        level 1 (steward→auto-watcher) and dismiss the originals.

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
