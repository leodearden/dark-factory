"""Top-level orchestration loop."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, TypeGuard

from shared.cli_invoke import AllAccountsCappedException, invoke_with_cap_retry
from shared.cost_store import CostStore

from orchestrator import digest as digest_mod
from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.invoke import invoke_agent
from orchestrator.agents.skill_prompt import load_skill_system_prompt
from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.mcp_lifecycle import McpLifecycle
from orchestrator.overrides import OverrideStore
from orchestrator.review_checkpoint import ReviewCheckpoint
from orchestrator.run_store import RunStore
from orchestrator.scheduler import (
    Scheduler,
    SetTaskStatusRejected,
    files_to_modules,
)
from orchestrator.usage_gate import UsageGate
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

if TYPE_CHECKING:
    from orchestrator.merge_queue import MergeWorker, SpeculativeMergeWorker

try:
    from escalation.queue import EscalationQueue
    from escalation.server import create_server
    HAS_ESCALATION = True
except ImportError:
    HAS_ESCALATION = False

try:
    from orchestrator.steward import TaskSteward
    HAS_STEWARD = True
except ImportError:
    HAS_STEWARD = False

logger = logging.getLogger(__name__)

# After this many consecutive set_task_status rejections for the same task in
# the stranded-in-progress / blocked sweep, escalate to L1 instead of looping.
# A genuine persistent failure (server-side validation contradicts a
# branch-on-main observation) is L1-worthy; transient PID-liveness or
# branch-resolution races should clear well before this threshold fires.
MAX_RECONCILE_FAILURES: int = 5

# Grace period added to the watcher rotation timeout on top of
# watcher_rotation_hours*3600.  Gives the agent time to emit its digest and
# exit cleanly before the supervisor kills it with a SIGTERM timeout.
_WATCHER_TIMEOUT_GRACE_SECS: float = 300.0

# Maximum backoff between unclean watcher exits (seconds).
_WATCHER_MAX_BACKOFF_SECS: float = 3600.0

# Allowed and disallowed tools for the escalation-watcher-auto rotation.
# Scoped to what escalation triage actually needs: file reads, foreground
# escalation.watcher subprocess, safe git reads, and the MCP tools for
# autonomous dispatch (update_task, add_dependency, resolve_issue).
# Defence-in-depth mirrors the unblock-auto precedent (dry_run_unblock.py).
_WATCHER_ALLOWED_TOOLS: list[str] = [
    'Read',
    'Glob',
    'Grep',
    # Foreground-blocking watcher subprocess (see SKILL.md wait-for-next-L1 step)
    'Bash(python -m escalation.watcher:*)',
    # Git reads for context
    'Bash(git log:*)',
    'Bash(git diff:*)',
    'Bash(git status:*)',
    'Bash(git show:*)',
    'Bash(git rev-parse:*)',
    'Bash(git branch:*)',
    'Bash(git ls-files:*)',
    # Escalation MCP: read + autonomous resolve
    'mcp__escalation__get_pending_escalations',
    'mcp__escalation__resolve_issue',
    # Fused-memory MCP: read + autonomous dispatch (scope_violation/dependency/cleanup)
    'mcp__fused-memory__get_task',
    'mcp__fused-memory__get_tasks',
    'mcp__fused-memory__search',
    'mcp__fused-memory__update_task',
    'mcp__fused-memory__add_dependency',
]
# Mutating tools blocked — no code edits, no destructive git, no infra commands.
_WATCHER_DISALLOWED_TOOLS: list[str] = [
    'Edit',
    'Write',
    'Bash(git commit:*)',
    'Bash(git push:*)',
    'Bash(git reset:*)',
    'Bash(git checkout:*)',
    'Bash(git restore:*)',
    'Bash(git clean:*)',
    'Bash(git merge:*)',
    'Bash(git rebase:*)',
]


def _pid_alive(pid: int) -> bool:
    """Return True if the process identified by *pid* is alive.

    Mirrors the semantics of
    fused-memory/src/fused_memory/services/orchestrator_detector.py:58-72
    without introducing a cross-package import edge.

    - Returns False for pid <= 0 (invalid).
    - Uses os.kill(pid, 0): success → alive; ProcessLookupError → dead;
      PermissionError → alive (we can see it but lack permission to signal it);
      other OSError → treat as dead.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _is_valid_sha_40(s: object) -> TypeGuard[str]:
    """Return True iff *s* is a well-formed 40-char lowercase hex SHA.

    Used to validate ``branch_base_sha`` values read from task metadata
    before comparing them against live git output.  Any non-conforming
    value is treated as missing so the reconciler falls through to the
    existing citation-grep guard rather than making a bogus comparison.
    """
    return (
        isinstance(s, str)
        and len(s) == 40
        and all(c in '0123456789abcdef' for c in s)
    )


def _acquire_project_lock(project_root: Path) -> IO:
    """Acquire an exclusive flock on a per-project lockfile.

    Returns the open file object — caller must keep a reference to it
    (closing or GC releases the lock).  Raises ``SystemExit(1)`` if
    another orchestrator instance already holds the lock.
    """
    lock_path = project_root / 'data' / 'orchestrator' / 'orchestrator.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    lock_file = open(lock_path, 'w')  # noqa: SIM115
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Another instance holds the lock — read its diagnostic info
        try:
            with open(lock_path) as f:
                info = f.read().strip()
        except OSError:
            info = '(unknown)'
        logger.error(
            'Another orchestrator is already running for this project.\n'
            f'  Lock holder: {info}\n'
            f'  Lock file:   {lock_path}\n'
            'Kill the existing instance first, or wait for it to finish.'
        )
        lock_file.close()
        raise SystemExit(1) from None

    # Write diagnostic info for anyone who tries to acquire next
    lock_file.truncate(0)
    lock_file.seek(0)
    lock_file.write(f'PID {os.getpid()} started {datetime.now(UTC).isoformat()}')
    lock_file.flush()
    return lock_file


@dataclass
class TaskReport:
    task_id: str
    title: str
    outcome: WorkflowOutcome
    cost_usd: float = 0.0
    duration_ms: int = 0
    agent_invocations: int = 0
    execute_iterations: int = 0
    verify_attempts: int = 0
    review_cycles: int = 0
    steward_cost_usd: float = 0.0
    steward_invocations: int = 0
    completed_at: str = ''
    # Block-context surfacing for the per-task retry cap.  Populated by
    # _run_slot from the workflow's stashed _last_block_* attrs when the
    # outcome is REQUEUED (harmless/empty on DONE paths).  Not persisted
    # to runs.db — purely in-memory for the cap check + cap-exhaust report.
    block_reason: str = ''
    block_detail: str = ''
    block_phase: str = ''


@dataclass
class HarnessReport:
    started_at: str = ''
    completed_at: str = ''
    total_tasks: int = 0
    completed: int = 0
    blocked: int = 0
    escalated: int = 0
    total_cost_usd: float = 0.0
    task_reports: list[TaskReport] = field(default_factory=list)
    paused_for_cap: bool = False
    cap_pause_duration_secs: float = 0.0
    review_checkpoints: int = 0
    review_findings: int = 0
    review_tasks_created: int = 0
    review_cost_usd: float = 0.0

    def summary(self) -> str:
        lines = [
            f'Orchestrator run complete: {self.completed}/{self.total_tasks} tasks done',
            f'  Blocked: {self.blocked}',
            f'  Escalated: {self.escalated}',
            f'  Total cost: ${self.total_cost_usd:.2f}',
            f'  Duration: {self.started_at} → {self.completed_at}',
        ]
        if self.paused_for_cap:
            lines.append(f'  Cap pause: {self.cap_pause_duration_secs:.0f}s total')
        if self.review_checkpoints > 0:
            lines.append(
                f'  Review checkpoints: {self.review_checkpoints} '
                f'({self.review_findings} findings, '
                f'{self.review_tasks_created} tasks, '
                f'${self.review_cost_usd:.2f})'
            )
        lines.extend([
            '',
            'Per-task results:',
        ])
        for r in self.task_reports:
            lines.append(
                f'  {r.task_id}: {r.outcome.value} '
                f'(${r.cost_usd:.2f}, {r.agent_invocations} invocations)'
            )
        return '\n'.join(lines)


class Harness:
    """Top-level orchestration loop."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        from orchestrator.agents.sandbox_dispatch import set_backend
        set_backend(config.sandbox.backend)
        self.mcp = McpLifecycle(config)
        self.git_ops = GitOps(config.git, config.project_root)
        self.scheduler = Scheduler(config, override_store=OverrideStore.from_config(config))
        # Wire the park-stop trip callback: Scheduler trips → Harness.pause_scheduler.
        # This connects in-memory trip detection to the full pause bundle
        # (persistence + event + log) defined on the Harness.  Sibling tasks
        # (cost-ceiling 1323, EWA digest 1327) call pause_scheduler() directly.
        self.scheduler._on_park_stop_trip = self.pause_scheduler
        self.briefing = BriefingAssembler(config)
        self.report = HarnessReport()
        self._recovered_plans: dict[str, dict] = {}
        # Worktrees that survived crash recovery with a stamped pre-EXECUTE
        # plan (no completed steps) but were NOT pre-loaded into
        # _recovered_plans — we want _plan() to re-run revalidation against
        # the existing stamped plan rather than skipping straight to EXECUTE
        # via initial_plan.  Membership protects them from the
        # _reconcile_stranded_in_progress sweep, which otherwise wipes any
        # worktree not in _recovered_plans.
        self._preserved_worktrees: set[str] = set()
        # Worktrees whose agent_session.json sidecar survived a crash with
        # no plan.json — the next workflow for that task resumes the prior
        # Claude session via --resume rather than spawning fresh.  Keyed
        # by task_id; value is the parsed sidecar dict.
        self._recovered_sessions: dict[str, dict] = {}

        # Usage cap gate
        self.usage_gate: UsageGate | None = (
            UsageGate(config.usage_cap) if config.usage_cap.enabled else None
        )

        # Review checkpoints
        self.review_checkpoint: ReviewCheckpoint | None = (
            ReviewCheckpoint(config, self.mcp, self.usage_gate)
            if config.review.enabled else None
        )
        self._review_running = False
        self._pending_review_task: asyncio.Task | None = None
        self._task_modules: dict[str, list[str]] = {}  # task_id -> modules

        # Escalation support
        self._escalation_queue: EscalationQueue | None = None
        self._escalation_events: dict[str, asyncio.Event] = {}
        self._escalation_task: asyncio.Task | None = None
        self._orphan_reaper_task: asyncio.Task | None = None

        # Digest + EWA trip counters (task 1327 AFK hardening).
        # Delegated to _init_digest_state() so test helpers can call the same
        # canonical code without duplicating the counter names by value (task 1449).
        self._init_digest_state()

        # Soft-cancel registry — keyed by task_id, set externally to abort
        # long workflow waits (merge queue future, steward grace period).
        # Mirrors ``_escalation_events``: created in ``_run_slot`` for each
        # active task and cleared on slot exit.  Used by ``cancel_workflow``
        # and by the reconciliation subscription that watches for terminal
        # task status transitions.  See zombie-escalation fix Step 4.
        self._workflow_cancel_events: dict[str, asyncio.Event] = {}

        # Per-task asyncio.Task handle for the _run_slot coroutine — registered
        # at _run_slot start, popped in its finally.  Enables hard-cancel
        # fallback when a workflow ignores the soft cancel_event.
        # See task 1491, ITEM 2 (hard-cancel fallback).
        self._workflow_slot_tasks: dict[str, asyncio.Task] = {}

        # Consecutive terminal-status poll counts per task.  Incremented each
        # poll a workflow is terminal but still active; reset when it is no
        # longer terminal or drops out of the active set.  Controls the
        # soft→hard cancel threshold (config.terminal_status_hard_cancel_polls).
        self._terminal_cancel_counts: dict[str, int] = {}

        # Background poll: periodically check fused-memory for active tasks
        # whose status has flipped to terminal out-of-band (typically a human
        # marking a task done via /unblock).  When detected, set the cancel
        # event so the workflow exits cleanly without churning escalations.
        # See zombie-escalation fix Step 5.
        self._terminal_status_watcher_task: asyncio.Task | None = None

        # Escalation-watcher-auto subprocess supervisor (task 1326).
        # Keeps a fresh escalation-watcher-auto agent alive via invoke_with_cap_retry
        # across multi-day AFK windows with rotation, exponential backoff, and a
        # crashloop→pause_scheduler guard.  Mirrors _terminal_status_watcher_task.
        self._watcher_supervisor_task: asyncio.Task | None = None
        # Monotonic timestamps of unclean watcher exits; used by the crashloop guard
        # to count failures within watcher_crashloop_window_secs.
        self._watcher_unclean_exits: deque[float] = deque()
        # Monotonic timestamps of degenerate-clean watcher exits (duration below
        # watcher_misconfigured_min_rotation_secs); used by the cost-runaway guard
        # (task 1388).  Separate from _watcher_unclean_exits so the trip reason
        # ('watcher_misconfigured' vs 'watcher_crashloop') is unambiguous.
        self._watcher_degenerate_clean_exits: deque[float] = deque()

        # Background sweep: periodic re-run of the startup
        # ``_reconcile_stranded_in_progress`` pass during a long run, so
        # tasks stranded by transient backend failures get unstrandred
        # without waiting for the next orchestrator restart.  See Fix 4.
        self._stranded_reconcile_task: asyncio.Task | None = None

        # Per-task consecutive-failure counter for the reconcile sweep —
        # cleared on any successful mark_done, used to gate L1 escalation.
        self._reconcile_failure_counts: dict[str, int] = {}
        # Per-task consecutive-citation-miss counter for the reconcile sweep —
        # incremented when the is_ancestor fast-path is short-circuited by a
        # missing task-id citation on main; reset on any successful flip or
        # genuine skip (e.g. open L1).  Used to escalate to L1 when the
        # reconciler refuses to auto-flip a task indefinitely.
        self._reconcile_skip_counts: dict[str, int] = {}
        # Wall-clock of the most recent _workflow_cancel_events.set() call,
        # keyed by task_id.  R3-race-guard window — the sweep skips a task
        # whose workflow was cancelled within the last
        # ``_RECONCILE_CANCEL_GRACE_S`` seconds, since the workflow's
        # finally: block may still be writing state.
        self._workflow_cancel_at: dict[str, float] = {}
        self._RECONCILE_CANCEL_GRACE_S: float = 30.0

        # Merge queue — single worker owns all main-branch advancement
        self._merge_queue: asyncio.Queue = asyncio.Queue()
        self._merge_worker: MergeWorker | SpeculativeMergeWorker | None = None
        self._merge_worker_task: asyncio.Task | None = None

        # Event store — created at run start with a generated run_id
        self.event_store: EventStore | None = None

        # Run store — incremental task result persistence (shares runs.db)
        self._run_store: RunStore | None = None
        self._run_id: str | None = None

        # Cost store — per-invocation cost tracking (shares runs.db)
        self.cost_store: CostStore | None = None

        # Auto-eval — process-local set of redo task ids dispatched during
        # this run. Used by the daily-budget query in ``_maybe_auto_eval``.
        self._auto_eval_redo_task_ids: set[str] = set()

        # Singleton lock — held for the duration of run()
        self._lock_file: IO | None = None

    def _init_digest_state(self) -> None:
        """Initialise task-1327 AFK-hardening digest counters.

        Called from ``__init__`` and from the test helper
        ``_init_harness_state_for_test`` (orchestrator/tests/_orch_helpers.py)
        so that fixtures constructing ``Harness`` via ``Harness.__new__`` call
        canonical code rather than duplicating counter names by value.

        When new digest-related counters are added to ``_maybe_write_digest``
        in the future, add them HERE — ``__init__``, the test helper, and all
        seven ``Harness.__new__``-based fixtures pick them up automatically.

        Attributes
        ----------
        _escalation_event_count:
            Incremented on every escalation submit/resolve callback.
        _last_digest_event_count:
            Snapshot of the count at the last digest write.
        _ewa_value:
            Current EWA state (process-local; resets on restart).
        _last_digest_window_end_iso:
            ISO timestamp of the last digest window's end; set to start time
            on first run.  Note: done_count comes from EventStore
            (count_done_in_window) as the single source of truth — no
            scheduler-delta counter needed (task 1421).
        """
        self._escalation_event_count: int = 0
        self._last_digest_event_count: int = 0
        self._ewa_value: float = 0.0
        self._last_digest_window_end_iso: str = ''

    async def run(
        self,
        prd_path: Path | None = None,
        dry_run: bool = False,
        delay_secs: int = 0,
        force_dirty_start: bool = False,
        retag_modules: bool = False,
    ) -> HarnessReport:
        """Execute the full orchestration pipeline.

        If *prd_path* is ``None``, skip PRD parsing and run existing tasks.
        If *delay_secs* > 0, sleep that many seconds after startup (escalation
        server runs immediately) before executing tasks.
        """
        self.report.started_at = datetime.now(UTC).isoformat()

        # Install the usage-gate SIGHUP handler now that we're inside
        # asyncio.run(). The gate's __init__ runs before the loop exists
        # (Harness is constructed in the sync body of cli.run()), so its
        # auto-registration is a no-op there. The call is idempotent —
        # gates constructed inside an event loop (evals, fused-memory)
        # already installed the handler in __init__.
        if self.usage_gate is not None:
            self.usage_gate.register_signal_handlers()

        # 0. Singleton lock — prevent concurrent orchestrators on same project
        self._lock_file = _acquire_project_lock(self.config.project_root)

        # 0a. Create event store and run store for this run
        import uuid

        run_id = f'run-{uuid.uuid4().hex[:12]}'
        self._run_id = run_id
        db_path = self.config.project_root / 'data' / 'orchestrator' / 'runs.db'
        try:
            self.event_store = EventStore(db_path, run_id)
            self.scheduler.event_store = self.event_store
        except Exception:
            logger.warning('Failed to create event store', exc_info=True)

        # 0a. Create run store and register this run immediately so
        # task_results can be written incrementally as tasks complete.
        try:
            self._run_store = RunStore(db_path)
            self._run_store.start_run(
                run_id,
                self.config.fused_memory.project_id,
                self.report.started_at,
                str(prd_path) if prd_path else '',
            )
        except Exception:
            logger.warning('Failed to create run store', exc_info=True)

        # 0a-post. Restore scheduler pause state from prior run (if any).
        await self._load_persisted_scheduler_pause()

        # 0b. Create cost store (shares runs.db with EventStore/RunStore)
        try:
            self.cost_store = CostStore(db_path)
            await self.cost_store.open()
        except Exception:
            logger.warning('Failed to create cost store', exc_info=True)

        # Wire cost store into usage gate for cap/failover/resume events
        if self.usage_gate and self.cost_store:
            self.usage_gate._cost_store = self.cost_store
            self.usage_gate._project_id = self.config.fused_memory.project_id
            self.usage_gate._run_id = run_id

        # Wire cost store into review checkpoint for review invocation costs
        if self.review_checkpoint and self.cost_store:
            self.review_checkpoint.cost_store = self.cost_store
            self.review_checkpoint.run_id = run_id

        # 0c. Refuse to start with dirty working tree (unless forced).
        # Checked before any servers start to avoid zombie processes on failure.
        if not force_dirty_start:
            dirty = await self.git_ops.has_dirty_working_tree()
            if dirty:
                self._lock_file.close()
                self._lock_file = None
                raise RuntimeError(
                    'Refusing to start: project_root has uncommitted tracked changes. '
                    'Commit or stash your work first, or pass --force-dirty-start to override.\n'
                    f'Dirty files:\n{dirty}'
                )

        # Hoisted out of the try block so the finally clause can cancel
        # in-flight workflow tasks even if an exception fires before the
        # main loop creates them.
        active: set[asyncio.Task] = set()
        task_reports: list[TaskReport] = []

        try:
            # 1. Start fused-memory HTTP server
            logger.info('Starting fused-memory HTTP server...')
            await self.mcp.start()

            # 1b. Start escalation server
            await self._start_escalation_server()

            # 1b2. Start merge worker
            await self._start_merge_worker()

            # 1c. Dismiss stale escalations from prior runs (non-fatal)
            try:
                await self._dismiss_stale_escalations()
            except Exception as e:
                logger.warning(f'Failed to dismiss stale escalations: {e}')

            # 1c0. Rehydrate merge-halt state from preserved L1s (non-fatal).
            # Must run after _dismiss_stale_escalations so we scan the
            # settled post-dismissal queue (only real L1s remain).
            try:
                self._rehydrate_merge_halt()
            except Exception as e:
                logger.warning(f'Failed to rehydrate merge halt: {e}')

            # 1c1. Start orphan L0 reaper (non-fatal) — catches escalations
            # whose task_id has no active workflow/steward (e.g. reviewer
            # emits against a synthetic task_id, or a workflow crashed
            # before its steward could claim them).
            self._start_orphan_l0_reaper()

            # 1c1b. Start terminal-status watcher (non-fatal) — cancels
            # active workflows whose task has been marked terminal
            # out-of-band (e.g. by a human via /unblock).  Without this,
            # the workflow churns through its merge/steward retry loop
            # for tens of minutes after the human has finished the task.
            self._start_terminal_status_watcher()

            # 1c1c1. Start escalation-watcher-auto supervisor (task 1326,
            # AFK hardening) — keeps a fresh autonomous L1-watcher subprocess
            # alive across multi-day AFK windows.  Depends on L1 persistence
            # (task 1321: _dismiss_stale_escalations above preserves L1 queue).
            self._start_watcher_supervisor()

            # 1c1c. Start periodic stranded-in-progress reconcile (Fix 4).
            # Catches tasks stranded by transient backend failures during a
            # long run so they don't accumulate until the next restart.
            self._start_stranded_reconcile()

            # 1c2. Delay before task execution (escalation server already running)
            if delay_secs > 0:
                hours, rem = divmod(delay_secs, 3600)
                mins, secs = divmod(rem, 60)
                parts = []
                if hours:
                    parts.append(f'{hours}h')
                if mins:
                    parts.append(f'{mins}m')
                if secs:
                    parts.append(f'{secs}s')
                human = ' '.join(parts)
                logger.info(
                    f'Delaying task execution by {human} — '
                    f'escalation server is live on port {self.config.escalation.port}'
                )
                await asyncio.sleep(delay_secs)
                logger.info('Delay complete — resuming task execution')

            # 1d. Usage cap startup check
            if self.usage_gate:
                logger.info('Checking usage cap status...')
                await self.usage_gate.check_at_startup()
                if self.usage_gate.is_paused:
                    logger.warning(f'Usage cap already hit: {self.usage_gate.paused_reason}')
                    if not self.config.usage_cap.wait_for_reset:
                        raise RuntimeError(
                            f'Usage cap hit at startup: {self.usage_gate.paused_reason}'
                        )
                    # wait_for_reset=True: probe loop is already running,
                    # workflows will block in before_invoke until gate reopens

            # 2. PRD decomposition retired with the parse_prd / Taskmaster
            #    cutover (see plans/do-1-on-a-happy-pony.md §Cycle 2). The
            #    task tree must be populated before invoking the orchestrator
            #    — typically via planning_mode + curator from an interactive
            #    session. The --prd flag is preserved for tagging existing
            #    tasks with their source PRD so review can trace provenance.
            if prd_path is not None:
                logger.info(
                    'PRD decomposition removed — assuming task tree is '
                    'already populated. Tagging tasks with source PRD: %s',
                    prd_path,
                )
                _pre_statuses, _ = await self.scheduler.get_statuses()
                pre_ids = set(_pre_statuses.keys())
                # 2a. Tag any tasks that don't yet carry a PRD with this one.
                await self._tag_prd_metadata(prd_path, pre_ids)

            existing_statuses, err = await self.scheduler.get_statuses()
            if 'pending' not in existing_statuses.values():
                if not existing_statuses:
                    # Distinguish transport failure from genuinely empty tree.
                    if err is not None:
                        raise RuntimeError(
                            f'Failed to reach fused-memory: '
                            f'{type(err).__name__}: {err}'
                        ) from err
                    # Genuinely empty: point operators at the fused-memory
                    # logs in case tasks should exist but weren't returned.
                    logger.error(
                        'get_statuses returned an empty mapping — if tasks '
                        'should exist, check fused-memory logs for transport '
                        'errors before assuming the task tree is empty.'
                    )
                raise RuntimeError(
                    'No pending tasks found. Populate the task tree via '
                    'planning_mode + curator before invoking the orchestrator.'
                )

            # 2b. Tag tasks with code modules for concurrency locking
            logger.info('Tagging tasks with code modules...')
            await self._tag_task_modules(force=retag_modules)

            # 2c. Recover crashed tasks from surviving worktrees
            await self._recover_crashed_tasks()

            # 2d. Reconcile stranded in-progress tasks (live-claimant-aware)
            await self._reconcile_stranded_in_progress()

            statuses, _ = await self.scheduler.get_statuses()
            self.report.total_tasks = sum(1 for s in statuses.values() if s == 'pending')
            logger.info(f'Task tree populated: {self.report.total_tasks} pending tasks')

            if dry_run:
                logger.info('Dry run — stopping after task population')
                return self.report

            # 3. Run workflow slots
            sem = asyncio.Semaphore(self.config.max_concurrent_tasks)

            while True:
                # Pick up any pending review task spawned by _collect_done_reports
                if self._pending_review_task is not None:
                    active.add(self._pending_review_task)
                    self._pending_review_task = None

                # If a review checkpoint is running, don't acquire new tasks —
                # wait for in-flight tasks (and the review) to complete.
                if self._review_running:
                    if not active:
                        break  # shouldn't happen — review task is in active
                    done, active = await asyncio.wait(
                        active, return_when=asyncio.FIRST_COMPLETED
                    )
                    self._collect_done_reports(done, task_reports)
                    continue

                # Check daily cost ceilings before dispatching the next task.
                # On breach, pause_scheduler() is called and acquire_next()
                # returns None (task-1322 machinery), draining in-flight work
                # and exiting the run loop cleanly.  Task 1323.
                await self._enforce_cost_ceilings()

                assignment = await self.scheduler.acquire_next()

                if assignment is None:
                    if not active:
                        # Before treating as "all done", sweep stranded
                        # in-progress.  Tasks stranded by transient backend
                        # failures may unblock pending dependents; a clean
                        # exit here would leave them for the next restart.
                        # See Fix 4 — stuck-blocked recovery.
                        changed = await self._reconcile_stranded_in_progress(
                            mid_run=True,
                        )
                        if changed > 0:
                            logger.info(
                                'Mid-run reconcile freed %d task(s) '
                                '— continuing main loop', changed,
                            )
                            continue
                        break  # all done or all blocked
                    # Wait for any active task to complete, then retry.
                    # Timeout ensures newly-added tasks are discovered
                    # within 15s even when no running task completes.
                    done, active = await asyncio.wait(
                        active, return_when=asyncio.FIRST_COMPLETED,
                        timeout=15,
                    )
                    self._collect_done_reports(done, task_reports)
                    continue

                await sem.acquire()
                self._task_modules[assignment.task_id] = list(assignment.modules)
                task = asyncio.create_task(
                    self._run_slot(assignment, sem),
                    name=f'workflow-{assignment.task_id}',
                )
                active.add(task)
                task.add_done_callback(active.discard)

            # Drain remaining
            if active:
                done, _ = await asyncio.wait(active)
                self._collect_done_reports(done, task_reports)

            self.report.task_reports = task_reports
            self.report.completed = sum(
                1 for r in task_reports if r.outcome == WorkflowOutcome.DONE
            )
            self.report.blocked = sum(
                1 for r in task_reports if r.outcome == WorkflowOutcome.BLOCKED
            )
            self.report.escalated = sum(
                1 for r in task_reports if r.outcome == WorkflowOutcome.ESCALATED
            )
            self.report.total_cost_usd = sum(r.cost_usd for r in task_reports)

            # 3b. Optional full review after all tasks complete
            if (self.review_checkpoint
                    and self.config.review.full_review_on_complete
                    and self.report.completed > 0):
                logger.info('Running full post-completion review...')
                try:
                    review_report = await self.review_checkpoint.run_full()
                    self.report.review_checkpoints += 1
                    self.report.review_findings += review_report.findings_count
                    self.report.review_tasks_created += len(review_report.tasks_created)
                    self.report.review_cost_usd += review_report.cost_usd
                    logger.info(
                        'Full review complete: %d findings, %d tasks created',
                        review_report.findings_count,
                        len(review_report.tasks_created),
                    )
                    if review_report.tasks_created:
                        try:
                            await self._tag_task_modules()
                        except Exception as tag_err:
                            logger.warning(f'Post-review module tagging failed: {tag_err}')
                except Exception as e:
                    logger.error(f'Full review failed: {e}')

        finally:
            # 4. Shutdown
            # 4a. Cancel any in-flight workflow tasks BEFORE shutting down
            # usage_gate — otherwise a cap-hit in a still-running agent can
            # spawn a fresh probe task via _handle_cap_detected AFTER
            # usage_gate.shutdown() has drained the existing ones, leaving
            # the event loop alive forever.
            if active:
                logger.info(f'Cancelling {len(active)} active workflow task(s)')
                for t in active:
                    t.cancel()
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*active, return_exceptions=True),
                        timeout=15.0,
                    )
                except TimeoutError:
                    logger.error('Workflow tasks did not drain within 15s')
                active.clear()

            self.report.completed_at = datetime.now(UTC).isoformat()

            # Finalize run metrics in SQLite (task_results were written
            # incrementally in _collect_done_reports; this updates the
            # runs row with final aggregates).
            if self._run_store and self._run_id:
                try:
                    self._run_store.finish_run(self._run_id, self.report)
                except Exception as e:
                    logger.warning(f'Failed to finalize run metrics: {e}')

            # Checkpoint all SQLite WAL files while the event loop is still
            # healthy — truncates the WAL so a subsequent crash cannot leave
            # the stores in a half-written state.  Best-effort: failures are
            # logged as warnings but never block shutdown.
            self._checkpoint_stores()

            # Save HarnessReport alongside review checkpoint reports
            if self.report.review_checkpoints > 0:
                self._save_harness_report()

            if self.usage_gate:
                if self.usage_gate.total_pause_secs > 0:
                    self.report.paused_for_cap = True
                    self.report.cap_pause_duration_secs = self.usage_gate.total_pause_secs
                try:
                    await self.usage_gate.shutdown()
                except Exception as e:
                    logger.warning(f'usage_gate.shutdown() failed: {e}')
            if self.cost_store:
                try:
                    await self.cost_store.close()
                except Exception as e:
                    logger.warning(f'cost_store.close() failed: {e}')
            try:
                await self._stop_merge_worker()
            except Exception as e:
                logger.warning(f'_stop_merge_worker() failed: {e}')
            try:
                await self._stop_orphan_l0_reaper()
            except Exception as e:
                logger.warning(f'_stop_orphan_l0_reaper() failed: {e}')
            try:
                await self._stop_terminal_status_watcher()
            except Exception as e:
                logger.warning(f'_stop_terminal_status_watcher() failed: {e}')
            try:
                await self._stop_watcher_supervisor()
            except Exception as e:
                logger.warning(f'_stop_watcher_supervisor() failed: {e}')
            try:
                await self._stop_stranded_reconcile()
            except Exception as e:
                logger.warning(f'_stop_stranded_reconcile() failed: {e}')
            try:
                await self._stop_escalation_server()
            except Exception as e:
                logger.warning(f'_stop_escalation_server() failed: {e}')
            try:
                await self.mcp.stop()
            except Exception as e:
                logger.warning(f'mcp.stop() failed: {e}')

            # 4b. Last-resort straggler sweep — catches any task the named
            # cleanup above missed (orphan probe tasks, cost-event
            # fire-and-forgets, sub-tasks spawned by merge/escalation stop).
            current = asyncio.current_task()
            stragglers = [
                t for t in asyncio.all_tasks()
                if t is not current and not t.done()
            ]
            if stragglers:
                names = [t.get_name() for t in stragglers]
                logger.warning(
                    f'Cancelling {len(stragglers)} straggler task(s): {names}'
                )
                for t in stragglers:
                    t.cancel()
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*stragglers, return_exceptions=True),
                        timeout=5.0,
                    )
                except TimeoutError:
                    still = [t.get_name() for t in stragglers if not t.done()]
                    logger.error(f'Stragglers did not die within 5s: {still}')

            # Release singleton lock
            if self._lock_file is not None:
                self._lock_file.close()
                self._lock_file = None

        logger.info(self.report.summary())
        return self.report

    async def _tag_prd_metadata(self, prd_path: Path, pre_parse_ids: set[str]) -> None:
        """Tag tasks with the PRD they were created from."""
        resolved_prd = str(prd_path.resolve())
        tasks = await self.scheduler.get_tasks()
        new_ids = {str(t.get('id', '')) for t in tasks} - pre_parse_ids

        tagged = 0
        for t in tasks:
            tid = str(t.get('id', ''))
            if not tid:
                continue
            metadata = t.get('metadata') or {}
            if metadata.get('prd'):
                continue  # already tagged
            if new_ids and tid not in new_ids:
                continue  # existed before parse_prd
            await self.scheduler.update_task(tid, {'prd': resolved_prd})
            tagged += 1

        if tagged:
            logger.info(f'Tagged {tagged} tasks with PRD: {resolved_prd}')

    async def _tag_task_modules(self, force: bool = False) -> None:
        """Invoke a Claude agent to tag each task with the files it touches.

        Uses structured output to get a JSON mapping of task_id → [files],
        then persists via scheduler.update_task() as ``metadata.files``.

        When *force* is ``True``, retag all non-done/cancelled tasks even if
        they already have file metadata.
        """
        tasks = await self.scheduler.get_tasks()

        skip_statuses = {'done', 'cancelled'}
        untagged = []
        for t in tasks:
            if t.get('status') in skip_statuses:
                continue
            if not force:
                metadata = t.get('metadata') or {}
                files = metadata.get('files', [])
                if files:
                    continue
            untagged.append(t)

        if not untagged:
            logger.info('No tasks to tag — skipping')
            return

        if force:
            logger.info(f'Force-retagging {len(untagged)} tasks with file metadata')

        # Get top-level directory listing for context
        try:
            entries = sorted(p.name for p in self.config.project_root.iterdir() if p.is_dir() and not p.name.startswith('.'))
        except OSError:
            entries = []

        task_summaries = []
        for t in untagged:
            task_summaries.append({
                'id': str(t.get('id', '')),
                'title': t.get('title', ''),
                'description': t.get('description', ''),
            })

        schema = {
            'type': 'object',
            'properties': {
                'tasks': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'id': {'type': 'string'},
                            'files': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': 'Predicted file paths (or directory paths) this task will create or modify',
                            },
                        },
                        'required': ['id', 'files'],
                    },
                },
            },
            'required': ['tasks'],
        }

        prompt = f"""\
Given these tasks and this codebase structure, predict which files each task
will create or modify.

Be specific and exhaustive with file predictions — include source files AND
test files. Use paths relative to the project root. The `files` field is used
to derive concurrency locks, so accuracy prevents unnecessary serialization.
Directory paths are accepted when an entire directory will be touched.

# Codebase top-level directories
{json.dumps(entries)}

# Tasks to tag
{json.dumps(task_summaries, indent=2)}

Output JSON matching the schema. Every task must appear in the output.
"""

        try:
            result = await invoke_with_cap_retry(
                usage_gate=self.usage_gate,
                label='Module tagging',
                invoke_fn=invoke_agent,
                prompt=prompt,
                system_prompt='You are a code module classifier. Given task descriptions and a codebase structure, determine which code modules each task will modify. Be precise and conservative.',
                cwd=self.config.project_root,
                model=self.config.models.module_tagger,
                max_turns=self.config.max_turns.module_tagger,
                max_budget_usd=self.config.budgets.module_tagger,
                output_schema=schema,
            )
        except AllAccountsCappedException as e:
            logger.warning(
                f'Module tagging skipped: all accounts capped '
                f'({e.retries} retries in {e.elapsed_secs:.1f}s)'
            )
            return

        if not result.success:
            logger.warning(f'Module tagger agent failed: {result.output[:200]}')
            return

        # Parse the structured output
        mapping = result.structured_output
        if not mapping:
            try:
                mapping = json.loads(result.output)
            except (json.JSONDecodeError, TypeError):
                logger.warning('Module tagger produced no parseable output')
                return

        tagged_count = 0
        for entry in mapping.get('tasks', []):
            task_id = str(entry.get('id', ''))
            files = entry.get('files', [])
            if task_id and files:
                await self.scheduler.update_task(
                    task_id, json.dumps({'files': files})
                )
                # Also populate in-memory cache so modules are available
                # immediately without re-fetching from taskmaster
                depth = self.config.lock_depth
                derived = files_to_modules(files, depth)
                if derived:
                    self.scheduler._module_cache[task_id] = derived
                tagged_count += 1

        logger.info(f'Tagged {tagged_count}/{len(untagged)} tasks with file metadata')
        logger.info(f'Module cache has {len(self.scheduler._module_cache)} entries')
        if self.scheduler._module_cache:
            sample = dict(list(self.scheduler._module_cache.items())[:3])
            logger.info(f'Module cache sample: {sample}')

    async def _recover_crashed_tasks(self) -> None:
        """Scan surviving worktrees and recover plans with completed work.

        For each worktree in the worktree base directory:
        - If it has a plan.json with completed steps, store the plan for
          injection into the resumed workflow.
        - Otherwise, clean up the worktree (no useful work to recover).

        Also resets any in-progress tasks to pending so acquire_next() picks
        them up.
        """
        worktree_base = self.git_ops.worktree_base
        if not worktree_base.exists():
            return

        recovered = 0
        cleaned = 0

        for entry in worktree_base.iterdir():
            if not entry.is_dir():
                continue
            task_id = entry.name
            plan_path = entry / '.task' / 'plan.json'

            if not plan_path.exists():
                # Mid-invocation crash: an agent subprocess was in flight when
                # the prior orchestrator died (no plan was written yet).  The
                # sidecar pins the Claude session UUID so the next workflow
                # can --resume it with a "continue" prompt instead of spawning
                # a fresh agent.
                sidecar_path = entry / '.task' / 'agent_session.json'
                if sidecar_path.exists():
                    try:
                        session_data = json.loads(sidecar_path.read_text())
                        logger.info(
                            f'Recovery: worktree {task_id} has agent session sidecar '
                            f'(role={session_data.get("role")}, '
                            f'session_id={session_data.get("session_id")}) '
                            f'— preserving for resume'
                        )
                        self._preserved_worktrees.add(task_id)
                        self._recovered_sessions[task_id] = session_data
                        recovered += 1
                        continue
                    except (json.JSONDecodeError, OSError) as e:
                        logger.warning(
                            f'Recovery: worktree {task_id} sidecar unreadable, '
                            f'falling back to cleanup: {e}'
                        )
                logger.info(
                    f'Recovery: worktree {task_id} has no plan — cleaning up'
                )
                await self.git_ops.cleanup_worktree(entry, task_id)
                cleaned += 1
                continue

            try:
                plan = json.loads(plan_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    f'Recovery: worktree {task_id} has corrupt plan — '
                    f'cleaning up ({e})'
                )
                await self.git_ops.cleanup_worktree(entry, task_id)
                cleaned += 1
                continue

            # Validate plan belongs to this task
            plan_task_id = plan.get('task_id')
            if plan_task_id and plan_task_id != task_id:
                logger.warning(
                    f'Recovery: worktree {task_id} has plan for task '
                    f'{plan_task_id} — task_id mismatch, cleaning up'
                )
                await self.git_ops.cleanup_worktree(entry, task_id)
                cleaned += 1
                continue

            # Check if plan has any completed steps
            # Note: some plans have prerequisites as plain strings (not dicts)
            completed = [
                s for col in ('prerequisites', 'steps')
                for s in plan.get(col, [])
                if isinstance(s, dict) and s.get('status') == 'done'
            ]

            if not completed:
                # Provenance-stamped plans deserve preservation: the architect
                # ran successfully and wrote a plan that subsequently got the
                # session_id stamp.  A common path here is the blast-radius
                # lock-conflict requeue (workflow.py:1071-1088), which leaves
                # a stamped pre-EXECUTE plan behind.  Wiping the worktree
                # forces the next acquisition to call the architect again —
                # 17-20 wasted Opus calls per 14d.  Keep the worktree, unlink
                # plan.lock, and add to _preserved_worktrees so the next
                # acquisition reuses it; _plan() will then take the
                # revalidation branch via _old_plan_base.  We deliberately
                # do NOT pre-load into _recovered_plans because that bypasses
                # _plan() entirely.
                if plan.get('_session_id'):
                    logger.info(
                        f'Recovery: worktree {task_id} has stamped plan with '
                        f'no completed steps — preserving for revalidation'
                    )
                    lock_path = entry / '.task' / 'plan.lock'
                    if lock_path.exists():
                        lock_path.unlink()
                        logger.info(
                            f'Recovery: cleared stale plan.lock for '
                            f'preserved task {task_id}'
                        )
                    # The architect already produced a stamped plan; any
                    # sidecar present is from a later (post-plan) invocation
                    # that crashed and isn't meaningful here — clear it to
                    # avoid confusing the next workflow on this task.
                    (entry / '.task' / 'agent_session.json').unlink(missing_ok=True)
                    self._preserved_worktrees.add(task_id)
                    recovered += 1
                    continue
                logger.info(
                    f'Recovery: worktree {task_id} has unstamped plan with no '
                    f'completed steps — cleaning up'
                )
                await self.git_ops.cleanup_worktree(entry, task_id)
                cleaned += 1
                continue

            total = sum(len(plan.get(col, [])) for col in ('prerequisites', 'steps'))
            logger.info(
                f'Recovery: worktree {task_id} has plan with '
                f'{len(completed)}/{total} steps done — storing for resumption'
            )
            self._recovered_plans[task_id] = plan
            # Clear stale plan.lock so the new session doesn't immediately requeue
            lock_path = entry / '.task' / 'plan.lock'
            if lock_path.exists():
                lock_path.unlink()
                logger.info(f'Recovery: cleared stale plan.lock for task {task_id}')
            recovered += 1

        if recovered or cleaned:
            logger.info(
                f'Crash recovery: {recovered} plans recovered, '
                f'{cleaned} worktrees cleaned'
            )

    async def _reconcile_stranded_in_progress(self, *, mid_run: bool = False) -> int:
        """Sweep stranded in-progress tasks back to pending (or done).

        Examines every task that is currently in-progress and checks whether
        it has a live claimant via plan.lock / owner_pid.  Any task without a
        live claimant is reverted to pending so the scheduler can re-acquire it.

        At startup (``mid_run=False``) this is called AFTER
        ``_recover_crashed_tasks()`` (which may unlink plan.lock for recovered
        worktrees) and BEFORE the first ``scheduler.acquire_next()`` call, so
        ``self.scheduler._dispatched`` is always empty.

        When ``mid_run=True`` the harness has dispatched tasks during this
        run; those are NOT stranded — they are actively held by the scheduler
        — and must be filtered before any liveness check.  Without the filter
        the sweep would race the workflow that legitimately holds the task.

        Returns the number of tasks reverted or marked done so the caller can
        decide whether to keep the main loop running (Fix 4: stuck-blocked
        recovery).

        **Already-on-main fast-path** (is_ancestor == True):
        When the task branch is already an ancestor of main, the task is
        terminal.  We resolve the branch to its 40-char commit SHA via
        ``git rev-parse --verify`` *before* ``cleanup_worktree`` (which calls
        ``git branch -D`` and would invalidate a post-cleanup rev-parse — see
        git_ops.py lines around cleanup_worktree).

        Success path: ``done_provenance={'commit': <sha>, 'note': '…'}`` —
        matches the workflow.py:656 convention so downstream consumers
        (fused-memory ``_validate_done_provenance``,
        ``invalidate_fabricated_shipping_edges.py``, Stage 2 reconciliation)
        can identify the SHA the task ended on.

        Failure path: if the branch ref vanishes between is_ancestor and the
        subsequent rev-parse (rare TOCTOU race), ``resolve_branch_sha`` returns
        None.  We fall back to note-only provenance and emit a WARNING log so
        operators can spot the race.  Reconciliation is best-effort and must
        not abort on this edge case; fused-memory accepts note-only provenance.

        **Branch-deleted fast-path** (find_merge_marker):
        ``is_ancestor`` returns False in two cases: (1) the branch ref still
        exists but isn't on main — revert is correct; (2) the branch ref is
        gone — ``git merge-base --is-ancestor`` exits non-zero because it
        cannot resolve the ref.  Case (2) is the realistic post-merge-queue
        crash scenario: ``advance_main`` succeeded and ``cleanup_worktree``
        deleted the branch, but ``set_task_status('done')`` never ran.

        To distinguish case (2) from case (1) we call
        ``git_ops.find_merge_marker(branch)`` immediately after the
        ``is_ancestor`` block.  ``find_merge_marker`` gates on
        ``resolve_branch_sha`` (returns None when the branch is still present,
        so it can only hit the git-log path when the ref is truly gone), then
        searches recent main commits for a subject matching
        ``Merge {branch} into {main_branch}`` — the format ``merge_to_main``
        writes.  A hit is treated identically to the
        is_ancestor path: pop ``_recovered_plans``, attempt
        ``cleanup_worktree`` (swallow errors), call
        ``set_task_status('done', done_provenance={'commit': marker_sha, ...})``.
        """
        statuses, _ = await self.scheduler.get_statuses()
        reverted = 0
        marked_done = 0
        log_prefix = 'Reconcile (mid-run)' if mid_run else 'Reconcile'

        # R4: sweep both 'in-progress' AND 'blocked' so out-of-band-merged
        # blocked tasks (manual `git merge` while task was blocked) get
        # marked done by the next sweep cycle.  'cancelled' and 'deferred'
        # are intentionally excluded — terminal-by-decision and
        # human-deferred respectively.
        sweep_statuses = {'in-progress', 'blocked'}
        now = time.monotonic()

        for tid, status in statuses.items():
            if status not in sweep_statuses:
                continue
            if mid_run and (
                tid in self.scheduler._dispatched
                or tid in self.scheduler.lock_table._held
            ):
                # Actively held by this run's scheduler — not stranded.
                continue

            # R3 race guard: when a workflow soft-cancel was set very
            # recently, the workflow's finally: block may still be writing
            # state (release_workflow + cleanup window).  Skip until the
            # grace period elapses so we don't revert work the workflow
            # is still finishing.
            if mid_run:
                cancelled_at = self._workflow_cancel_at.get(tid)
                if (
                    cancelled_at is not None
                    and now - cancelled_at < self._RECONCILE_CANCEL_GRACE_S
                ):
                    continue

            try:
                outcome = await self._reconcile_one_stranded(
                    tid, status, mid_run=mid_run,
                )
            except SetTaskStatusRejected as exc:
                # Persistence layer refused our write — count strikes; on
                # threshold, escalate to L1.  Honest log so future operators
                # can see *why* the task is still stranded instead of the
                # old "marked done" misnomer.  Other (unexpected) exception
                # types intentionally propagate so bugs in the sweep surface
                # rather than being silently skipped.
                count = self._reconcile_failure_counts.get(tid, 0) + 1
                self._reconcile_failure_counts[tid] = count
                logger.error(
                    '%s: failed to mark task %s done — %s: %s '
                    '(consecutive failures=%d/%d)',
                    log_prefix, tid, exc.error_code, exc.raw,
                    count, MAX_RECONCILE_FAILURES,
                )
                if count >= MAX_RECONCILE_FAILURES and self._escalation_queue:
                    self._escalate_reconcile_failure(tid, exc, count)
                continue

            if outcome == 'marked_done':
                marked_done += 1
                self._reconcile_failure_counts.pop(tid, None)
            elif outcome == 'reverted':
                reverted += 1
                self._reconcile_failure_counts.pop(tid, None)

        if reverted or marked_done:
            logger.info(
                '%s: %d stranded task(s) reverted to pending; '
                '%d marked done (branch already on main)',
                log_prefix, reverted, marked_done,
            )
        return reverted + marked_done

    async def _reconcile_one_stranded(
        self, tid: str, status: str, *, mid_run: bool,
    ) -> str | None:
        """Reconcile a single stranded task. Returns 'marked_done', 'reverted', or None.

        Raises ``SetTaskStatusRejected`` if the persistence layer refuses the
        recovery write — caller handles failure counting + escalation.
        """
        branch = f'{self.git_ops.config.branch_prefix}{tid}'

        # Fetch task metadata once for both fast-paths (is_ancestor + find_merge_marker).
        # Scheduler.get_task normalises metadata at the boundary
        # (scheduler.py:_normalize_task_metadata), so task['metadata'] is always a
        # dict whenever task is not None.  `or {}` collapses any residual None value
        # (e.g. a manually-constructed task dict that bypasses normalisation); the
        # load-bearing guard against task itself being absent is `if task else {}`.
        # The unconditional fetch is the deliberate trade-off: one MCP call per
        # stranded task even when neither fast-path fires (e.g. lock-state revert),
        # in exchange for a single source of truth for `metadata` shared by both
        # fast-paths (eliminating the duplicated per-branch get_task pattern).
        task = await self.scheduler.get_task(tid)
        metadata = (task.get('metadata') or {}) if task else {}

        # Already-on-main fast-path (is_ancestor == True).
        # NB: is_ancestor is degenerate for zero-commit branches whose tip
        # equals the main HEAD at branch-create time.  Two guards reject
        # the false-positive shape before we flip the row to done.
        if await self.git_ops.is_ancestor(branch, self.git_ops.config.main_branch):
            # Guard 1 — open L1 escalation for this task.  An L1 escalation
            # is the deliberate human-handoff signal (e.g. workflow ran
            # _mark_blocked(escalate_to_human=True) when the task was
            # declared unactionable); the reconciler must not second-guess
            # the disposition, even if the branch tip happens to sit on a
            # main ancestor.  Skip counter is intentionally NOT incremented
            # here — the L1 escalation that triggered the skip already
            # exists, so double-escalating would spam.
            if (
                self._escalation_queue is not None
                and self._escalation_queue.has_open_l1(tid)
            ):
                logger.info(
                    'Reconcile: task %s on main but has open L1 escalation; '
                    'leaving status=%s (open L1 vetoes auto-flip)',
                    tid, status,
                )
                return None

            # Guard 2 — positive citation evidence on main.  We require a
            # commit on main whose subject cites the task id; this rejects
            # the zero-commit-branch shape where is_ancestor returns True
            # trivially but no commit actually lands the task's work.
            # Resolve BEFORE cleanup_worktree (which runs `git branch -D`
            # and would invalidate a post-cleanup grep against the branch).
            citation_sha = await self.git_ops.find_task_citation_commit(
                tid,
                pattern_template=self.git_ops.config.commit_citation_pattern,
            )
            if citation_sha is None:
                count = self._reconcile_skip_counts.get(tid, 0) + 1
                self._reconcile_skip_counts[tid] = count
                logger.info(
                    'Reconcile: task %s branch on main but no commit '
                    'cites task/%s; leaving status=%s '
                    '(consecutive citation misses=%d/%d)',
                    tid, tid, status, count, MAX_RECONCILE_FAILURES,
                )
                if (
                    count >= MAX_RECONCILE_FAILURES
                    and self._escalation_queue is not None
                ):
                    self._escalate_reconcile_skip(tid, status, count)
                return None

            # Guard 3 — branch-advanced structural check.
            # Guards 1 (open L1) and 2 (citation grep) are content
            # heuristics.  This guard is structural: it rejects the
            # false-positive shape where a zero-commit branch sits on a
            # main ancestor (is_ancestor returns True trivially) even
            # though no real implementation work was pushed.  We compare
            # the live branch tip against the recorded branch_base_sha
            # written by workflow._setup_worktree_and_artifacts at
            # branch-creation time.
            #
            # Missing / malformed branch_base_sha → fall through (backward
            # compat for tasks created before this guard was deployed, or
            # if the metadata write failed transiently at creation time).
            branch_base_sha = metadata.get('branch_base_sha')
            if _is_valid_sha_40(branch_base_sha):
                branch_tip_sha = await self.git_ops.resolve_branch_sha(branch)
                if branch_tip_sha == branch_base_sha:
                    logger.info(
                        'Reconcile: task %s branch tip == branch_base_sha (%s); '
                        'branch never advanced past creation — vetoing auto-flip',
                        tid, branch_base_sha,
                    )
                    return None

            # All guards passed — clear the skip counter and flip.
            self._reconcile_skip_counts.pop(tid, None)
            note = (
                f'reconcile: branch on main while task was {status} '
                f'(out-of-band merge)'
                if status == 'blocked'
                else 'reconcile: branch already on main when stranded in-progress'
            )
            await self._mark_in_progress_done(
                tid, citation_sha, note, 'branch-already-on-main',
            )
            return 'marked_done'

        # Branch-deleted fast-path (find_merge_marker).
        # is_ancestor returned False, but the branch may simply not exist
        # any more (cleanup_worktree ran after advance_main but before
        # set_task_status).
        # Note: find_merge_marker already gates on branch-existence
        # (git_ops.py:774) — it returns None when the branch ref is still
        # live.  The stale-marker check below therefore only fires when the
        # branch is gone and a commit on main mentions the task id in its
        # merge message.
        marker_sha = await self.git_ops.find_merge_marker(branch)
        if marker_sha:
            # Stale-marker check — re-opened-branch / prior-incarnation guard.
            # A task can be re-queued (branch deleted + re-created) after a
            # prior incarnation was merged.  In that case, the prior merge's
            # commit lands on main with the same task id, and find_merge_marker
            # would return its SHA — triggering a spurious done flip for the
            # current incarnation.
            #
            # Reject the marker if it is an ancestor of branch_base_sha (i.e.
            # it pre-dates the current incarnation's creation point).
            # `find_merge_marker`'s branch-existence gate handles the orthogonal
            # case where the branch ref still exists (returns None there);
            # this check handles the case where the branch is gone but the
            # stale marker from a *prior* incarnation under the same task id
            # is already on main.
            #
            # Missing/malformed branch_base_sha → fall through (backward compat
            # for tasks created before this guard was deployed).
            branch_base_sha = metadata.get('branch_base_sha')
            if _is_valid_sha_40(branch_base_sha) and await self.git_ops.is_ancestor(
                marker_sha, branch_base_sha
            ):
                logger.info(
                    'Reconcile: task %s stale marker %s is ancestor of '
                    'branch_base_sha %s — belongs to a prior incarnation; '
                    'vetoing auto-flip',
                    tid, marker_sha, branch_base_sha,
                )
                return None

            note = (
                f'reconcile: merge marker found on main while task was {status}'
                if status == 'blocked'
                else 'reconcile: branch deleted but merge marker found on main'
            )
            await self._mark_in_progress_done(
                tid, marker_sha, note, 'branch-deleted-marker-found',
            )
            return 'marked_done'

        # No on-main evidence.  For 'blocked' tasks, leave the row alone —
        # blocked is a deliberate state and we only flip it to done on
        # observed evidence.
        if status == 'blocked':
            return None

        # 'in-progress' and not on main: classify by lock state.
        worktree_path = self.git_ops.worktree_base / tid
        lock_path = worktree_path / '.task' / 'plan.lock'

        if not lock_path.exists():
            # No worktree or no lock → orphan, revert.
            if (
                worktree_path.exists()
                and tid not in self._recovered_plans
                and tid not in self._preserved_worktrees
            ):
                try:
                    await self.git_ops.cleanup_worktree(worktree_path, tid)
                except Exception:
                    logger.warning(
                        'Reconcile: cleanup_worktree failed for task %s'
                        ' (no-lock); continuing',
                        tid, exc_info=True,
                    )
            await self.scheduler.set_task_status(tid, 'pending')
            logger.info(
                'Reconcile: reverted task %s to pending (reason=no-lock)', tid
            )
            return 'reverted'

        # Lock exists — check whether the owner is still alive.
        owner_alive = False
        try:
            lock_data = json.loads(lock_path.read_text())
            if not isinstance(lock_data, dict):
                raise ValueError('plan.lock is not a JSON object')
            owner_pid = lock_data.get('owner_pid')
            if owner_pid is None:
                logger.warning(
                    'Reconcile: plan.lock for task %s has no owner_pid;'
                    ' treating as stale',
                    tid,
                )
            else:
                try:
                    owner_alive = _pid_alive(int(owner_pid))
                except (TypeError, ValueError):
                    owner_alive = False
        except (OSError, json.JSONDecodeError, ValueError):
            owner_alive = False

        # R3: during a mid_run sweep, owner_pid is almost always the harness
        # PID (which IS alive by definition).  The dispatch-table filter at
        # the top of the loop already excluded actively-held tasks; if we
        # made it here, the workflow has exited without releasing the lock.
        # Treat the lock as stale and fall through to recovery.
        if owner_alive and not mid_run:
            return None  # Live claimant outside our run — leave alone.

        # Stale lock (or mid_run-and-not-active) — clear it and revert.
        if (
            tid not in self._recovered_plans
            and tid not in self._preserved_worktrees
        ):
            try:
                await self.git_ops.cleanup_worktree(worktree_path, tid)
            except Exception:
                logger.warning(
                    'Reconcile: cleanup_worktree failed for task %s'
                    ' (stale-lock); continuing',
                    tid, exc_info=True,
                )
            with contextlib.suppress(OSError):
                lock_path.unlink(missing_ok=True)
        else:
            with contextlib.suppress(OSError):
                lock_path.unlink()
        await self.scheduler.set_task_status(tid, 'pending')
        logger.info(
            'Reconcile: reverted task %s to pending (reason=stale-lock)', tid
        )
        return 'reverted'

    async def _mark_in_progress_done(
        self,
        tid: str,
        sha: str | None,
        note: str,
        reason: str,
    ) -> None:
        """Mark a stranded task done via ``scheduler.mark_done``.

        Thin wrapper around ``Scheduler.mark_done`` that owns the side-effect
        sequence — pop ``_recovered_plans``, attempt ``cleanup_worktree``
        (swallow errors), call ``mark_done(kind='found_on_main', ...)``
        because the sweep is *finding* tasks already on main, not actively
        merging them.

        Args:
            tid: The task id (string form, as it appears in get_statuses).
            sha: The on-main commit anchoring the recovery.  When ``None``
                (rev-parse race for a vanishing branch), log + skip — the
                next sweep retries; ``mark_done`` requires a real SHA.
            note: Free-text provenance note distinguishing the recovery
                path (already-on-main vs deleted-but-marker-found).
            reason: Short slug used in cleanup-failure WARNING and the
                success INFO log.

        Raises ``SetTaskStatusRejected`` on persistent persistence-layer
        rejection — caller decides whether to count strikes / escalate.
        """
        if sha is None:
            logger.warning(
                'Reconcile: rev-parse failed for task %s (%s) — '
                'skipping; next sweep will retry',
                tid, reason,
            )
            return
        worktree_path = self.git_ops.worktree_base / tid
        self._recovered_plans.pop(tid, None)
        self._recovered_sessions.pop(tid, None)
        if worktree_path.exists():
            try:
                await self.git_ops.cleanup_worktree(worktree_path, tid)
            except Exception:
                logger.warning(
                    'Reconcile: cleanup_worktree failed for task %s'
                    ' (%s); continuing',
                    tid, reason, exc_info=True,
                )
        await self.scheduler.mark_done(
            tid, kind='found_on_main', sha=sha, note=note,
        )
        logger.info(
            'Reconcile: marked task %s done (reason=%s)', tid, reason,
        )

    def _escalate_reconcile_failure(
        self,
        tid: str,
        exc: SetTaskStatusRejected,
        count: int,
    ) -> None:
        """Submit an L1 escalation for persistent reconcile failure.

        Fired when the same task has rejected ``mark_done`` MAX_RECONCILE_FAILURES
        consecutive sweeps in a row — i.e. the persistence layer
        contradicts a branch-on-main observation persistently enough that
        steward-mediated retry can't unstick it.
        """
        if not self._escalation_queue:
            return
        from escalation.models import Escalation

        esc = Escalation(
            id=self._escalation_queue.make_id(tid),
            task_id=tid,
            agent_role='harness-reconcile',
            severity='blocking',
            category='reconcile_persistent_rejection',
            summary=(
                f'Reconcile sweep failed to mark task {tid} done '
                f'{count}x consecutively'
            )[:200],
            detail=(
                f'set_task_status(done) rejected by fused-memory after '
                f'{count} consecutive sweeps for task {tid}.\n\n'
                f'error_code: {exc.error_code}\n'
                f'raw: {exc.raw}\n\n'
                f'The branch was observed on main (or a merge marker was '
                f'found) but the persistence layer refuses the transition. '
                f'Manual investigation required — the row may carry stale '
                f'metadata.files or the done_provenance ancestor check may '
                f'be failing.'
            ),
            suggested_action='investigate_persistence_layer_rejection',
        )
        self._escalation_queue.submit(esc)
        # Reset counter so we don't re-escalate every sweep — operator
        # action will resolve and the next true rejection (if any) starts
        # a fresh strike count.
        self._reconcile_failure_counts.pop(tid, None)

    def _escalate_reconcile_skip(
        self,
        tid: str,
        status: str,
        count: int,
    ) -> None:
        """Submit an L1 escalation for persistent citation-miss skips.

        Mirror of ``_escalate_reconcile_failure`` for the case where
        ``find_task_citation_commit`` keeps returning None despite a stable
        ``is_ancestor==True`` observation — i.e. the branch sits on main
        but nothing on main cites the task, so the reconciler refuses to
        auto-flip the row.  After ``MAX_RECONCILE_FAILURES`` consecutive
        skips, escalate so a human can resolve manually.
        """
        if not self._escalation_queue:
            return
        from escalation.models import Escalation

        esc = Escalation(
            id=self._escalation_queue.make_id(tid),
            task_id=tid,
            agent_role='harness-reconcile',
            severity='blocking',
            category='reconcile_citation_missing',
            summary=(
                f'Reconciler refuses to auto-flip task {tid}: branch on '
                f'main but no commit cites the task ({count}x consecutive)'
            )[:200],
            detail=(
                f'reconciler refuses to auto-flip task {tid}: branch on '
                f'main but no commit cites the task. Status: {status}. '
                f'Resolve manually.\n\n'
                f'Consecutive citation misses: {count} / '
                f'{MAX_RECONCILE_FAILURES}.\n\n'
                f'The is_ancestor check returns True for task/{tid}, but '
                f'no commit on main matches the configured citation '
                f'pattern (GitConfig.commit_citation_pattern, or the '
                f'built-in default if unset).  Either the merge happened '
                f'with a non-conventional commit subject, or the branch '
                f'tip is degenerate (zero-commit branch sitting on a main '
                f'ancestor).  Inspect the branch and main history; mark '
                f'the task done manually if the work landed under a '
                f'non-matching subject, or leave it in its current status '
                f'and resolve this escalation to silence the sweep.'
            ),
            suggested_action='investigate_citation_missing',
            level=1,
        )
        self._escalation_queue.submit(esc)
        self._reconcile_skip_counts.pop(tid, None)

    async def _run_slot(
        self, assignment, sem: asyncio.Semaphore
    ) -> TaskReport | None:
        """Run a single workflow slot."""
        report = None
        try:
            logger.info(
                f'Starting workflow for task {assignment.task_id}: '
                f'{assignment.task.get("title", "")}'
            )
            # Create escalation event for this task
            esc_event = None
            if self._escalation_queue:
                esc_event = asyncio.Event()
                self._escalation_events[assignment.task_id] = esc_event

            # Soft-cancel event — exposed so external code (reconciliation
            # subscriber, release_workflow MCP tool) can interrupt long
            # workflow waits when the task becomes terminal out-of-band.
            cancel_event = asyncio.Event()
            self._workflow_cancel_events[assignment.task_id] = cancel_event

            # Register the asyncio.Task handle for this slot so hard_cancel_workflow
            # can request a hard cancel if the workflow ignores the soft event.
            # current_task() is safe here because _run_slot is always scheduled
            # as an asyncio.Task (via create_task in _acquire_and_run_slot).
            self._workflow_slot_tasks[assignment.task_id] = asyncio.current_task()  # type: ignore[assignment]

            recovered_plan = self._recovered_plans.pop(assignment.task_id, None)
            recovered_session = self._recovered_sessions.pop(assignment.task_id, None)
            # Drop any preserved-worktree marker once the slot picks the task up.
            self._preserved_worktrees.discard(assignment.task_id)

            # Build steward factory — steward starts when the workflow
            # creates its worktree (it needs the path).
            steward_factory = None
            if HAS_STEWARD and self._escalation_queue:
                esc_q = self._escalation_queue  # capture for closure (narrows type)

                def _make_steward(worktree: Path, config_dir=None, *, _assign=assignment) -> TaskSteward:  # type: ignore[name-defined]
                    return TaskSteward(  # type: ignore[reportPossiblyUnbound]
                        task_id=_assign.task_id,
                        task=_assign.task,
                        worktree=worktree,
                        config=self.config,
                        mcp=self.mcp,
                        escalation_queue=esc_q,
                        briefing=self.briefing,
                        usage_gate=self.usage_gate,
                        config_dir=config_dir,
                        event_store=self.event_store,
                    )
                steward_factory = _make_steward

            workflow = TaskWorkflow(
                assignment=assignment,
                config=self.config,
                git_ops=self.git_ops,
                scheduler=self.scheduler,
                briefing=self.briefing,
                mcp=self.mcp,
                escalation_queue=self._escalation_queue,
                escalation_event=esc_event,
                usage_gate=self.usage_gate,
                initial_plan=recovered_plan,
                steward_factory=steward_factory,
                merge_queue=self._merge_queue,
                merge_worker=self._merge_worker,
                event_store=self.event_store,
                cost_store=self.cost_store,
                cancel_event=cancel_event,
                resume_session_id=recovered_session,
            )

            if self.event_store:
                self.event_store.emit(
                    EventType.task_started,
                    task_id=assignment.task_id,
                    data={'title': assignment.task.get('title', '')},
                )

            outcome = await workflow.run()

            steward_cost = 0.0
            steward_invocations = 0
            steward = workflow._steward
            if steward and hasattr(steward, 'metrics'):
                steward_cost = steward.metrics.total_cost_usd
                steward_invocations = steward.metrics.invocations

            report = TaskReport(
                task_id=assignment.task_id,
                title=assignment.task.get('title', ''),
                outcome=outcome,
                cost_usd=workflow.metrics.total_cost_usd,
                duration_ms=workflow.metrics.total_duration_ms,
                agent_invocations=workflow.metrics.agent_invocations,
                execute_iterations=workflow.metrics.execute_iterations,
                verify_attempts=workflow.metrics.verify_attempts,
                review_cycles=workflow.metrics.review_cycles,
                steward_cost_usd=steward_cost,
                steward_invocations=steward_invocations,
                completed_at=datetime.now(UTC).isoformat(),
                block_reason=workflow._last_block_reason,
                block_detail=workflow._last_block_detail,
                block_phase=workflow._last_block_phase,
            )

            if self.event_store:
                self.event_store.emit(
                    EventType.task_completed,
                    task_id=assignment.task_id,
                    cost_usd=report.cost_usd,
                    duration_ms=report.duration_ms,
                    data={
                        'outcome': outcome.value,
                        'agent_invocations': report.agent_invocations,
                        'execute_iterations': report.execute_iterations,
                        'verify_attempts': report.verify_attempts,
                        'review_cycles': report.review_cycles,
                        'steward_cost_usd': report.steward_cost_usd,
                        'steward_invocations': report.steward_invocations,
                    },
                )

            return report
        except Exception as e:
            logger.exception(f'Workflow slot error for task {assignment.task_id}: {e}')
            return TaskReport(
                task_id=assignment.task_id,
                title=assignment.task.get('title', ''),
                outcome=WorkflowOutcome.BLOCKED,
            )
        finally:
            self._escalation_events.pop(assignment.task_id, None)
            self._workflow_cancel_events.pop(assignment.task_id, None)
            self._workflow_cancel_at.pop(assignment.task_id, None)
            self._workflow_slot_tasks.pop(assignment.task_id, None)
            self._terminal_cancel_counts.pop(assignment.task_id, None)
            requeued = report is not None and report.outcome == WorkflowOutcome.REQUEUED
            if report is not None:
                requeued = await self._apply_retry_cap(
                    assignment.task_id, report, requeued,
                )
                # Auto-eval: when the optimistic path blocks at a phase we
                # care about, dispatch a sibling redo on the full architect
                # path. Best-effort — never blocks slot release.
                try:
                    await self._maybe_auto_eval(assignment, report)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        'Task %s: auto-eval hook failed (non-fatal): %s',
                        assignment.task_id, exc,
                    )
            self.scheduler.release(assignment.task_id, requeued=requeued)
            sem.release()

    async def _maybe_auto_eval(
        self, assignment, report: TaskReport,
    ) -> None:
        """Dispatch an auto-eval redo when the optimistic path blocked.

        Triggers when:
        - ``report.outcome`` is BLOCKED.
        - The original task carries ``metadata.optimistic_path`` (set by
          Lever B or Lever C).
        - The original task is NOT itself an auto-eval redo.
        - ``report.block_phase`` is in ``config.auto_eval_phases``.
        - The 24h auto-eval USD budget has not been exhausted.

        On success, the original branch + worktree are renamed with a
        ``-skip-attempt`` suffix and a sibling task is submitted via
        ``submit_task(planning_mode=True)`` (curator dedupe bypass) with
        ``metadata.force_full_path=True`` to prevent the redo from taking
        an optimistic path itself.
        """
        if not getattr(self.config, 'auto_eval_enabled', False):
            return
        if report.outcome != WorkflowOutcome.BLOCKED:
            return
        if report.block_phase not in self.config.auto_eval_phases:
            return

        task_metadata = (assignment.task.get('metadata') or {})
        optimistic_path = str(task_metadata.get('optimistic_path') or '')
        if not optimistic_path:
            return
        if task_metadata.get('auto_eval_redo'):
            return

        # Daily budget check — skip the redo when exhausted.
        used = await self._auto_eval_budget_used_24h()
        if used >= self.config.auto_eval_redo_budget_usd:
            logger.info(
                'Task %s: auto-eval budget exhausted ($%.2f used of $%.2f) — '
                'skipping redo',
                assignment.task_id, used, self.config.auto_eval_redo_budget_usd,
            )
            return

        original_id = assignment.task_id

        # Rename the original branch + worktree so the new task can use a
        # fresh task/<original_id>-redo branch (sibling task gets its own
        # branch from create_worktree on dispatch).
        old_branch = original_id
        new_branch = f'{original_id}-skip-attempt'
        old_path = self.git_ops.worktree_base / old_branch
        new_path = self.git_ops.worktree_base / new_branch
        renamed = False
        if old_path.exists():
            try:
                await self.git_ops.rename_worktree(
                    old_path=old_path,
                    new_path=new_path,
                    old_branch=old_branch,
                    new_branch=new_branch,
                )
                renamed = True
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    'Task %s: auto-eval rename failed (%s) — continuing '
                    'without rename',
                    original_id, exc,
                )

        redo_metadata = {
            'auto_eval_redo': True,
            'auto_eval_pair': str(original_id),
            'spawned_from': str(original_id),
            'force_full_path': True,
            'modules': list(task_metadata.get('modules') or []),
            'files': list(task_metadata.get('files') or []),
        }

        title = (
            f'[auto-eval redo] {assignment.task.get("title", "")}'
        )[:200]
        description = (
            f'Automated full-architect redo of task {original_id} '
            f'(blocked at phase={report.block_phase} via optimistic path '
            f'{optimistic_path!r}). Original branch+worktree renamed to '
            f'{new_branch} for forensic comparison.'
        )

        try:
            submit_result = await self.scheduler.dispatch_tool(
                'submit_task',
                {
                    'project_root': str(self.config.project_root),
                    'title': title,
                    'description': description,
                    'priority': str(
                        assignment.task.get('priority') or 'medium'
                    ),
                    'metadata': redo_metadata,
                    'planning_mode': True,
                },
                timeout=30,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                'Task %s: auto-eval submit_task raised (%s) — aborting redo',
                original_id, exc,
            )
            return

        new_task_id = self._extract_task_id(submit_result)
        if not new_task_id:
            logger.warning(
                'Task %s: auto-eval submit_task returned no task_id (%r)',
                original_id, submit_result,
            )
            return

        # Flip the new task from deferred → pending so the scheduler picks
        # it up on the next tick.
        try:
            await self.scheduler.dispatch_tool(
                'set_task_status',
                {
                    'project_root': str(self.config.project_root),
                    'id': new_task_id,
                    'status': 'pending',
                },
                timeout=15,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                'Task %s: auto-eval status flip to pending failed (%s) — '
                'redo task %s left deferred',
                original_id, exc, new_task_id,
            )

        # Cross-reference the new id back onto the original task.
        try:
            await self.scheduler.update_task(
                original_id,
                metadata={
                    **task_metadata,
                    'auto_eval_pair': str(new_task_id),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                'Task %s: auto-eval back-link update failed (%s)',
                original_id, exc,
            )

        # Track for budget accounting.
        self._auto_eval_redo_task_ids.add(str(new_task_id))

        if self.event_store:
            self.event_store.emit(
                EventType.auto_eval_dispatched,
                task_id=str(new_task_id),
                data={
                    'original_task_id': str(original_id),
                    'optimistic_path': optimistic_path,
                    'block_phase': report.block_phase,
                    'rename_succeeded': renamed,
                    'budget_used_24h': used,
                },
            )

        logger.info(
            'Task %s: auto-eval dispatched redo task %s '
            '(rename=%s, budget_used=$%.2f)',
            original_id, new_task_id, renamed, used,
        )

    async def _trailing_24h_fetch_one(
        self,
        sql: str,
        leading_params: tuple = (),
        *,
        label: str,
    ) -> tuple | None:
        """Execute an arbitrary trailing-24h aggregate query and return the first row.

        ``sql`` must be a full SELECT ending with ``... completed_at >= ?`` (the
        cutoff is appended internally as the LAST parameter).  ``leading_params``
        are passed in front of the cutoff in the same order.

        Fail-open: returns None when ``cost_store is None`` or when any
        exception is raised by ``_require_conn()`` / cursor execute / fetchone.
        Callers MUST treat None as "skip / assume zero" — dispatch must
        never be blocked by a transient cost-DB error.  The ``label`` is used
        only in the warning log.
        """
        if not sql.rstrip().endswith('completed_at >= ?'):
            raise ValueError(
                '_trailing_24h_fetch_one: sql must end with "completed_at >= ?" '
                f'(got {sql[:80]!r})'
            )
        if self.cost_store is None:
            return None
        cutoff = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
        try:
            conn = self.cost_store._require_conn()  # type: ignore[attr-defined]
            cur = await conn.execute(sql, (*leading_params, cutoff))
            row = await cur.fetchone()
            await cur.close()
            return tuple(row) if row is not None else None
        except Exception as exc:  # noqa: BLE001 — never block dispatch on this
            logger.warning(
                '%s: trailing-24h cost query failed (%s) — fail-open',
                label, exc,
            )
            return None

    async def _auto_eval_budget_used_24h(self) -> float:
        """Sum cost_usd from invocations table for known auto-eval redo
        task_ids in the trailing 24h.

        Process-local: the redo set resets on harness restart. Acceptable
        because auto-eval is rollout instrumentation that will be sunset
        once metrics stabilise.
        """
        if not self._auto_eval_redo_task_ids:
            return 0.0
        placeholders = ','.join('?' for _ in self._auto_eval_redo_task_ids)
        row = await self._trailing_24h_fetch_one(
            f'SELECT COALESCE(SUM(cost_usd), 0.0) FROM invocations '
            f'WHERE task_id IN ({placeholders}) AND completed_at >= ?',
            tuple(self._auto_eval_redo_task_ids),
            label='auto_eval_budget_used_24h',
        )
        return float(row[0]) if row and row[0] is not None else 0.0

    async def _enforce_cost_ceilings(self) -> None:
        """Check daily cost ceilings and pause the scheduler on breach.

        Runs every dispatch-loop tick (Harness.run()) immediately before
        ``scheduler.acquire_next()``.  Two checks, evaluated in order:

        1. Watcher ceiling (early warning): trailing-24h cost for
           invocations with ``role LIKE '%watcher%'`` vs
           ``config.watcher_daily_cost_ceiling_usd``.
        2. Orch-wide ceiling (safety net): trailing-24h cost for ALL
           invocations vs ``config.orch_daily_cost_ceiling_usd``.

        The first ceiling that trips wins.  When the scheduler is already
        paused (from any source), returns immediately to avoid redundant
        RunStore writes and duplicate log spam.

        Delegates to ``CostStore.cost_totals_in_window`` so the
        role-LIKE-watcher aggregation pattern lives in exactly one place.
        The trailing-24h window is ``[cutoff_24h_iso, now_iso]`` (inclusive
        ``BETWEEN``); any invocations written after the ``now_iso`` snapshot
        are silently excluded — acceptable for a fail-open dispatch guard.
        On any DB error the method returns immediately without pausing
        (fail-open: a transient query failure must never stop dispatch).
        Task 1323.
        """
        if self.scheduler.is_paused:
            return
        if self.cost_store is None:
            return
        now = datetime.now(UTC)
        now_iso = now.isoformat()
        cutoff_24h_iso = (now - timedelta(hours=24)).isoformat()
        try:
            total, watcher = await self.cost_store.cost_totals_in_window(cutoff_24h_iso, now_iso)
        except Exception as exc:  # noqa: BLE001 — never block dispatch on this
            logger.warning(
                '_enforce_cost_ceilings: trailing-24h cost query failed (%s) — fail-open',
                exc,
            )
            return

        if watcher >= self.config.watcher_daily_cost_ceiling_usd:
            logger.warning(
                '_enforce_cost_ceilings: watcher 24h cost $%.2f >= ceiling $%.2f '
                '— pausing scheduler',
                watcher, self.config.watcher_daily_cost_ceiling_usd,
            )
            await self.pause_scheduler('cost_ceiling_watcher_exceeded')
            return

        if total >= self.config.orch_daily_cost_ceiling_usd:
            logger.warning(
                '_enforce_cost_ceilings: orch-wide 24h cost $%.2f >= ceiling $%.2f '
                '— pausing scheduler',
                total, self.config.orch_daily_cost_ceiling_usd,
            )
            await self.pause_scheduler('cost_ceiling_orch_exceeded')

    @staticmethod
    def _extract_task_id(submit_result: Any) -> str | None:
        """Pull the task_id out of an MCP tools/call response wrapper.

        ``dispatch_tool`` returns the raw MCP result envelope. The shape may
        be ``{'task_id': ...}``, ``{'content': [{'text': '{...}'}], ...}``,
        or ``{'structuredContent': {...}}`` depending on transport. Normalise.
        """
        if not isinstance(submit_result, dict):
            return None
        if submit_result.get('task_id'):
            return str(submit_result['task_id'])
        for key in ('structuredContent', 'result'):
            inner = submit_result.get(key)
            if isinstance(inner, dict) and inner.get('task_id'):
                return str(inner['task_id'])
        content = submit_result.get('content')
        if isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get('type') == 'text':
                    try:
                        parsed = json.loads(str(chunk.get('text') or ''))
                    except (TypeError, ValueError):
                        continue
                    if isinstance(parsed, dict) and parsed.get('task_id'):
                        return str(parsed['task_id'])
        return None

    async def _apply_retry_cap(
        self, task_id: str, report: TaskReport, requeued: bool,
    ) -> bool:
        """Update the per-task REQUEUED counter and fire cap exhaustion.

        Called from ``_run_slot``'s finally block just before
        ``scheduler.release``.  Returns the effective *requeued* flag — False
        when cap exhaustion fires (task is blocked, cooldown is irrelevant),
        otherwise the caller's original value.
        """
        if self._run_id is None:
            return requeued
        if report.outcome == WorkflowOutcome.REQUEUED:
            attempt_cost = report.cost_usd + report.steward_cost_usd
            count = self.scheduler.record_requeue(
                task_id,
                phase=report.block_phase or 'unknown',
                reason=report.block_reason or 'unknown',
                detail=report.block_detail or '',
                run_id=self._run_id,
                cost_usd=attempt_cost,
            )
            if count >= self.config.requeue_cap:
                history = list(
                    self.scheduler._requeue_history.get(task_id, ())
                )
                cumulative_cost = sum(r.cost_usd for r in history)
                try:
                    await self.scheduler.trigger_retry_cap_exhausted(
                        task_id,
                        run_id=self._run_id,
                        cost_usd=cumulative_cost,
                        escalation_queue=self._escalation_queue,
                    )
                except Exception:
                    logger.exception(
                        'Retry-cap trigger failed for task %s', task_id,
                    )
                return False  # task is blocked, skip cooldown
        elif report.outcome == WorkflowOutcome.DONE:
            self.scheduler.clear_requeue_count(task_id)
        return requeued

    def _collect_done_reports(
        self, done: set[asyncio.Task], task_reports: list[TaskReport]
    ) -> None:
        """Extract TaskReports from completed asyncio.Tasks and track merges for review."""
        for t in done:
            # Handle review checkpoint completion
            if t.get_name() == 'review-checkpoint':
                self._review_running = False
                try:
                    t.result()  # propagate exceptions
                except Exception as e:
                    logger.error(f'Review checkpoint error: {e}')
                continue

            try:
                report = t.result()
                if report:
                    task_reports.append(report)
                    # Persist task result immediately so it survives crashes
                    if self._run_store and self._run_id:
                        try:
                            self._run_store.save_task_result(
                                self._run_id, report,
                                self.config.fused_memory.project_id,
                            )
                        except Exception as e:
                            logger.warning(
                                f'Failed to persist task result '
                                f'{report.task_id}: {e}'
                            )
                    # Track module merges for review checkpoints
                    if (report.outcome == WorkflowOutcome.DONE
                            and self.review_checkpoint):
                        modules = self._task_modules.pop(report.task_id, [])
                        self.review_checkpoint.record_merge(modules)
                        # Check if review should trigger
                        if (self.review_checkpoint.should_trigger()
                                and not self._review_running):
                            self._trigger_review_checkpoint()
            except Exception as e:
                logger.error(f'Workflow slot error: {e}')

    def _trigger_review_checkpoint(self) -> None:
        """Spawn a review checkpoint as a concurrent task.

        The main loop will pause task acquisition while the review runs
        (``_review_running`` flag) but in-flight tasks continue in their
        worktrees.
        """
        self._review_running = True
        # We can't add to active here — the caller's done set is immutable.
        # Instead, the review task is spawned and tracked; the main loop checks
        # _review_running and waits on active (which includes this task after
        # the next iteration adds it).

        # Actually, we need the task in `active` for the main loop's asyncio.wait.
        # The caller iterates `done`, so we store the task on self for the loop
        # to pick up.
        self._pending_review_task = asyncio.create_task(
            self._run_review_checkpoint(), name='review-checkpoint',
        )

    async def _run_review_checkpoint(self) -> None:
        """Execute a focused review checkpoint."""
        assert self.review_checkpoint is not None
        logger.info('Starting review checkpoint...')
        try:
            review_report = await self.review_checkpoint.run_focused()
            self.report.review_checkpoints += 1
            self.report.review_findings += review_report.findings_count
            self.report.review_tasks_created += len(review_report.tasks_created)
            self.report.review_cost_usd += review_report.cost_usd
            logger.info(
                'Review checkpoint complete: %d findings, %d tasks created, '
                'cost=$%.2f',
                review_report.findings_count,
                len(review_report.tasks_created),
                review_report.cost_usd,
            )
            # Tag newly created tasks with module metadata (agents may have
            # included modules, but re-run the batch tagger as a fallback
            # for any tasks that lack them).
            if review_report.tasks_created:
                try:
                    await self._tag_task_modules()
                except Exception as tag_err:
                    logger.warning(f'Post-review module tagging failed: {tag_err}')
        except Exception as e:
            logger.error(f'Review checkpoint failed: {e}')

    def _checkpoint_stores(self) -> None:
        """Run WAL TRUNCATE checkpoint on each SQLite store — best-effort.

        Called from the shutdown ``finally:`` block after ``finish_run()`` so
        the WAL is truncated while the event loop is still healthy.  Each store
        is checkpointed independently; a failure in one does not prevent the
        others from running.  Stores that were never created (None) are skipped
        silently — mirrors the init-failure guard at lines 437-453.

        Follows the existing harness shutdown try/except pattern used by
        ``usage_gate.shutdown()``, ``cost_store.close()``, etc.
        """
        if self._run_store:
            try:
                result = self._run_store.checkpoint()
                logger.info(f'run_store checkpoint: {result}')
            except Exception as e:
                logger.warning(f'run_store.checkpoint() failed: {e}')

        if self.event_store:
            try:
                result = self.event_store.checkpoint()
                logger.info(f'event_store checkpoint: {result}')
            except Exception as e:
                logger.warning(f'event_store.checkpoint() failed: {e}')

        override_store = self.scheduler.override_store if self.scheduler else None
        if override_store is not None:
            try:
                result = override_store.checkpoint()
                logger.info(f'override_store checkpoint: {result}')
            except Exception as e:
                logger.warning(f'override_store.checkpoint() failed: {e}')

    def _save_harness_report(self) -> None:
        """Persist HarnessReport as JSON alongside review checkpoint reports."""
        reports_dir = self.config.project_root / self.config.review.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)

        ts = self.report.started_at.replace(':', '').replace('-', '')[:15]
        path = reports_dir / f'harness-{ts}.json'

        data = {
            'started_at': self.report.started_at,
            'completed_at': self.report.completed_at,
            'total_tasks': self.report.total_tasks,
            'completed': self.report.completed,
            'blocked': self.report.blocked,
            'escalated': self.report.escalated,
            'total_cost_usd': self.report.total_cost_usd,
            'paused_for_cap': self.report.paused_for_cap,
            'cap_pause_duration_secs': self.report.cap_pause_duration_secs,
            'review_checkpoints': self.report.review_checkpoints,
            'review_findings': self.report.review_findings,
            'review_tasks_created': self.report.review_tasks_created,
            'review_cost_usd': self.report.review_cost_usd,
            'task_reports': [
                {
                    'task_id': r.task_id,
                    'title': r.title,
                    'outcome': r.outcome.value,
                    'cost_usd': r.cost_usd,
                    'duration_ms': r.duration_ms,
                    'agent_invocations': r.agent_invocations,
                    'completed_at': r.completed_at,
                }
                for r in self.report.task_reports
            ],
        }

        try:
            path.write_text(json.dumps(data, indent=2))
            logger.info('HarnessReport saved: %s', path)
        except OSError as e:
            logger.warning('Failed to save HarnessReport: %s', e)

    async def _start_merge_worker(self) -> None:
        """Start the merge queue worker as a background asyncio task.

        Uses SpeculativeMergeWorker (two-coroutine pipeline) by default.
        MergeWorker (serial) is preserved but deprecated.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker

        self._merge_worker = SpeculativeMergeWorker(
            self.git_ops, self._merge_queue, event_store=self.event_store,
        )
        self._merge_worker_task = asyncio.create_task(
            self._merge_worker.run(), name='merge-worker',
        )
        logger.info('Speculative merge worker started')

    async def _stop_merge_worker(self) -> None:
        """Stop the merge worker gracefully."""
        if self._merge_worker_task is not None and self._merge_worker is not None:
            await self._merge_worker.stop()
            self._merge_worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._merge_worker_task
            self._merge_worker_task = None
            logger.info('Merge worker stopped')

    def _build_task_status_lookup(self) -> Callable[[str], Awaitable[str | None]]:
        """Return an async callable (task_id) -> str|None backed by the scheduler.

        The returned closure is injected into the escalation MCP server as
        ``task_status_lookup`` so the chokepoint can query live task status
        without holding a direct reference to the Harness.
        """
        async def _lookup(task_id: str) -> str | None:
            return await self.scheduler.get_status(task_id)

        return _lookup

    async def _start_escalation_server(self) -> None:
        """Start the escalation MCP server as a background asyncio task."""
        if not HAS_ESCALATION:
            logger.info('Escalation package not installed — skipping escalation server')
            return
        assert HAS_ESCALATION  # narrows type: EscalationQueue and create_server are defined

        queue_dir = Path(self.config.escalation.queue_dir)
        if not queue_dir.is_absolute():
            queue_dir = self.config.project_root / queue_dir
        self._escalation_queue = EscalationQueue(queue_dir)  # type: ignore[possibly-unbound]
        self._escalation_queue.set_notify_callback(self._on_escalation)
        self._escalation_queue.set_resolve_callback(self._on_escalation_resolved)

        # Wire escalation queue into review checkpoint so it can triage
        # escalations the deep reviewer emits against the synthetic review
        # task_id (which has no workflow/steward to handle them).
        if self.review_checkpoint is not None:
            self.review_checkpoint.escalation_queue = self._escalation_queue

        mcp_server = create_server(  # type: ignore[possibly-unbound]
            self._escalation_queue,
            merge_queue=self._merge_queue,
            orch_config=self.config,
            event_store=self.event_store,
            harness=self,
            task_status_lookup=self._build_task_status_lookup(),
        )
        host = self.config.escalation.host
        port = self.config.escalation.port

        async def _serve():
            import uvicorn
            app = mcp_server.http_app()
            uv_config = uvicorn.Config(
                app, host=host, port=port, log_level='warning',
            )
            server = uvicorn.Server(uv_config)
            await server.serve()

        self._escalation_task = asyncio.create_task(_serve(), name='escalation-server')
        logger.info(f'Escalation MCP server starting on {host}:{port}')
        # Give the server a moment to bind, then verify it didn't crash
        await asyncio.sleep(0.5)
        if self._escalation_task.done():
            exc = self._escalation_task.exception()
            if exc:
                raise RuntimeError(
                    f'Escalation server failed to start on {host}:{port}: {exc}'
                ) from exc

    async def _dismiss_stale_escalations(self) -> None:
        """Dismiss stale L0 escalations left over from prior orchestrator runs.

        Called right after _start_escalation_server() so that any L0
        (agent→steward) escalations persisted in the queue directory from a
        previous (crashed or completed) run are cleared before the new run
        begins.

        L1 (steward→human) escalations are intentionally preserved across
        restart — they represent human-attention requests that were not yet
        acted on and must not be silently lost during long AFK periods.
        """
        if self._escalation_queue is None:
            return

        resolution = (
            'Auto-dismissed: orchestrator restarted — stale from prior run'
        )
        count = self._escalation_queue.dismiss_all_pending(resolution)
        if count:
            logger.info(
                f'Dismissed {count} stale L0 escalation(s) from prior run; '
                f'L1 escalations preserved across restart'
            )

    def _rehydrate_merge_halt(self) -> str | None:
        """Restore merge-halt state from preserved L1 escalations after restart.

        On restart, SpeculativeMergeWorker is constructed fresh (un-halted,
        no owner).  _dismiss_stale_escalations intentionally preserves pending
        level-1 wip_conflict/unmerged_state escalations — but NOTHING
        re-asserts the corresponding halt or re-registers the halt owner.

        This method scans the settled post-dismissal queue for preserved L1s
        of the relevant categories and restores the (halted, owner-registered)
        state, so the existing _on_escalation_resolved -> unhalt_wip path
        cleanly releases the halt when the operator resolves the L1.

        Returns the escalation id that now owns the halt, or None if no action
        was taken.
        """
        if self._merge_worker is None or self._escalation_queue is None:
            return None

        candidates = [
            esc for esc in self._escalation_queue.get_pending()
            if esc.level == 1 and esc.category in {'wip_conflict', 'unmerged_state'}
        ]
        if not candidates:
            return None

        if len(candidates) > 1:
            logger.warning(
                '_rehydrate_merge_halt: %d qualifying L1s found; registering '
                'only the most recent as halt owner.  The merge queue will '
                'resume as soon as that owner L1 is resolved — even though '
                '%d older L1(s) remain pending.  Resolve the most-recent L1 '
                'last to avoid premature queue resumption.',
                len(candidates),
                len(candidates) - 1,
            )
        esc = max(candidates, key=lambda e: datetime.fromisoformat(e.timestamp))
        reason = (
            f'Rehydrated merge halt from preserved L1 {esc.id} '
            f'(category={esc.category}) after restart'
        )
        self._merge_worker.halt_for_wip(reason)
        self._merge_worker.set_halt_owner(esc.id)
        logger.warning(reason)
        return esc.id

    async def _stop_escalation_server(self) -> None:
        """Stop the escalation server."""
        if self._escalation_task is not None:
            self._escalation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._escalation_task
            self._escalation_task = None
            logger.info('Escalation server stopped')

    def _start_orphan_l0_reaper(self) -> None:
        """Start the orphan L0 reaper as a background asyncio task.

        The reaper periodically scans pending level-0 escalations; any whose
        ``task_id`` has no active workflow (not in ``_escalation_events``)
        and whose age exceeds ``orphan_l0_timeout_secs`` is promoted to
        level 1 so the escalation-watcher can pick it up.  Without this,
        orphan L0s (e.g. emitted by the deep reviewer, or left behind by a
        crashed workflow) sit pending until the next orchestrator restart
        auto-dismisses them unread.
        """
        if not self.config.orphan_l0_reaper_enabled:
            return
        if self._escalation_queue is None:
            return
        if self._orphan_reaper_task is not None and not self._orphan_reaper_task.done():
            return
        self._orphan_reaper_task = asyncio.create_task(
            self._orphan_l0_reaper_loop(), name='orphan-l0-reaper',
        )
        logger.info(
            'Orphan L0 reaper started (timeout=%.0fs, interval=%.0fs)',
            self.config.orphan_l0_timeout_secs,
            self.config.orphan_l0_check_interval_secs,
        )

    async def _stop_orphan_l0_reaper(self) -> None:
        """Cancel the orphan L0 reaper loop."""
        if self._orphan_reaper_task is not None:
            self._orphan_reaper_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._orphan_reaper_task
            self._orphan_reaper_task = None
            logger.info('Orphan L0 reaper stopped')

    async def _orphan_l0_reaper_loop(self) -> None:
        """Wake periodically and promote orphan L0 escalations to L1."""
        interval = self.config.orphan_l0_check_interval_secs
        while True:
            try:
                await asyncio.sleep(interval)
                self._reap_orphan_l0_escalations()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception('Orphan L0 reaper pass failed')

    def _reap_orphan_l0_escalations(self) -> int:
        """Single pass: promote any overdue orphan L0 to L1.  Returns count.

        Extracted from the loop so tests can drive it deterministically.
        An escalation is an orphan when its ``task_id`` is not in
        ``_escalation_events`` (no running workflow) and it is older than
        ``orphan_l0_timeout_secs``.
        """
        if self._escalation_queue is None:
            return 0

        from escalation.models import Escalation

        timeout = self.config.orphan_l0_timeout_secs
        now = datetime.now(UTC)
        promoted = 0

        for esc in self._escalation_queue.get_pending():
            if esc.level != 0:
                continue
            if esc.task_id in self._escalation_events:
                continue  # active workflow will handle it
            try:
                age_secs = (now - datetime.fromisoformat(esc.timestamp)).total_seconds()
            except (ValueError, TypeError):
                continue
            if age_secs < timeout:
                continue

            reesc = Escalation(
                id=self._escalation_queue.make_id(esc.task_id),
                task_id=esc.task_id,
                agent_role='harness-orphan-reaper',
                severity=esc.severity,
                category=esc.category,
                summary=(
                    f'Orphan L0 ({age_secs:.0f}s old, no active workflow): '
                    f'{esc.summary}'
                ),
                detail=esc.detail,
                suggested_action='manual_intervention',
                worktree=esc.worktree,
                workflow_state=esc.workflow_state,
                level=1,
            )
            self._escalation_queue.submit(reesc)
            self._escalation_queue.resolve(
                esc.id,
                (
                    'Auto-promoted to level 1 — orphan L0 (no active '
                    f'workflow for task_id={esc.task_id})'
                ),
                dismiss=True,
                resolved_by='harness-orphan-reaper',
            )
            promoted += 1

        if promoted:
            logger.warning(
                'Orphan L0 reaper: promoted %d escalation(s) to L1', promoted,
            )
        return promoted

    def cancel_workflow(self, task_id: str) -> bool:
        """Soft-cancel an active workflow.

        Sets the per-task ``cancel_event`` so any long await inside the
        workflow (merge-queue future, steward grace period) wakes promptly.
        The workflow re-reads task status and exits ``DONE`` if terminal,
        ``REQUEUED`` otherwise.

        Returns ``True`` iff a workflow was active for ``task_id`` (i.e. the
        cancel event was registered).  Returns ``False`` when the task has
        no slot — the call is a no-op and the caller can avoid waiting.
        """
        event = self._workflow_cancel_events.get(task_id)
        if event is None:
            return False
        event.set()
        # Stamp wall-clock so the reconcile sweep can grace-skip this task
        # for the next _RECONCILE_CANCEL_GRACE_S seconds — the workflow's
        # finally: block may still be writing state (R3 race guard).
        self._workflow_cancel_at[task_id] = time.monotonic()
        return True

    def is_workflow_active(self, task_id: str) -> bool:
        """True iff a workflow slot is currently active for ``task_id``."""
        return task_id in self._workflow_cancel_events

    def hard_cancel_workflow(self, task_id: str) -> bool:
        """Hard-cancel the asyncio.Task running the workflow slot for ``task_id``.

        This is the escalation path when a workflow ignores the soft
        ``cancel_event`` (set by ``cancel_workflow``) for too long.
        Requesting asyncio.Task.cancel() forces a ``CancelledError`` into the
        coroutine's next await point; the ``_run_slot`` finally block still
        runs (CancelledError is BaseException, so it bypasses the
        ``except Exception`` guard at harness.py:1833) ensuring lock release
        and registry cleanup.

        Re-stamps ``_workflow_cancel_at`` for the R3 reconcile grace window,
        mirroring ``cancel_workflow``.

        Returns ``True`` iff a live (non-done) slot task was found and
        ``cancel()`` was requested.  Returns ``False`` when there is no
        registered slot task or it is already done — the call is a no-op and
        the caller should treat this as a no-op.
        """
        task = self._workflow_slot_tasks.get(task_id)
        if task is None or task.done():
            return False
        # Stamp wall-clock so the reconcile sweep respects the R3 grace window.
        self._workflow_cancel_at[task_id] = time.monotonic()
        task.cancel()
        return True

    # ------------------------------------------------------------------
    # Terminal-status watcher (zombie-escalation fix Step 5)
    # ------------------------------------------------------------------

    def _start_terminal_status_watcher(self) -> None:
        """Start a background poll that cancels workflows whose tasks have
        gone terminal out-of-band (e.g. a human marking a task ``done``).

        Polling avoids cross-process subscription to fused-memory's event bus.
        At expected interval (~30 s) the poll cost is one ``get_statuses``
        round-trip per active set, which is ~30 ms for a warm fused-memory.
        """
        if not self.config.terminal_status_watcher_enabled:
            return
        if (
            self._terminal_status_watcher_task is not None
            and not self._terminal_status_watcher_task.done()
        ):
            return
        self._terminal_status_watcher_task = asyncio.create_task(
            self._terminal_status_watcher_loop(),
            name='terminal-status-watcher',
        )
        logger.info(
            'Terminal-status watcher started (interval=%.0fs)',
            self.config.terminal_status_poll_interval_secs,
        )

    async def _stop_terminal_status_watcher(self) -> None:
        """Cancel the terminal-status watcher loop."""
        if self._terminal_status_watcher_task is not None:
            self._terminal_status_watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._terminal_status_watcher_task
            self._terminal_status_watcher_task = None
            logger.info('Terminal-status watcher stopped')

    async def _terminal_status_watcher_loop(self) -> None:
        """Wake periodically and cancel workflows whose task is now terminal."""
        interval = self.config.terminal_status_poll_interval_secs
        while True:
            try:
                await asyncio.sleep(interval)
                await self._scan_for_terminal_active_tasks()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception('Terminal-status watcher pass failed')

    # ------------------------------------------------------------------
    # Escalation-watcher-auto subprocess supervisor (task 1326)
    # ------------------------------------------------------------------

    def _start_watcher_supervisor(self) -> None:
        """Start the escalation-watcher-auto subprocess supervisor.

        No-op when config.watcher_supervisor_enabled is False.
        Idempotent: does nothing if the task is already alive.
        Mirrors _start_terminal_status_watcher.
        """
        if not self.config.watcher_supervisor_enabled:
            return
        if (
            self._watcher_supervisor_task is not None
            and not self._watcher_supervisor_task.done()
        ):
            return
        self._watcher_supervisor_task = asyncio.create_task(
            self._watcher_supervisor_loop(),
            name='watcher-supervisor',
        )
        logger.info(
            'Escalation-watcher-auto supervisor started '
            '(rotation_escalations=%d rotation_hours=%.1f)',
            self.config.watcher_rotation_escalations,
            self.config.watcher_rotation_hours,
        )

    async def _stop_watcher_supervisor(self) -> None:
        """Cancel the watcher supervisor loop. Mirrors _stop_terminal_status_watcher."""
        if self._watcher_supervisor_task is not None:
            self._watcher_supervisor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._watcher_supervisor_task
            self._watcher_supervisor_task = None
            logger.info('Escalation-watcher-auto supervisor stopped')

    async def _run_watcher_rotation(self):  # type: ignore[return]
        """Run one escalation-watcher-auto rotation via invoke_with_cap_retry.

        Extracted for deterministic unit testing (same rationale as
        _scan_for_terminal_active_tasks).  Returns the AgentResult from
        invoke_with_cap_retry.

        The agent is instructed to exit cleanly after ROTATION_ESCALATIONS or
        ROTATION_HOURS — whichever fires first — and to emit its digest as its
        final message.  The supervisor-side timeout is rotation_hours*3600 +
        _WATCHER_TIMEOUT_GRACE_SECS so that a wedged agent that ignores its
        own rotation instructions is force-killed (classified as unclean, feeds
        the crashloop guard).
        """

        cfg = self.config
        user_prompt = (
            f'You are running as an autonomous escalation watcher.\n'
            f'Rotation limits (injected by supervisor):\n'
            f'  ROTATION_ESCALATIONS={cfg.watcher_rotation_escalations}\n'
            f'  ROTATION_HOURS={cfg.watcher_rotation_hours}\n'
            f'\n'
            f'When you have handled ROTATION_ESCALATIONS escalations '
            f'or {cfg.watcher_rotation_hours} hours have elapsed since startup, '
            f'emit your digest as the final message and exit cleanly.\n'
            f'\n'
            f'Project root: {cfg.project_root}\n'
            f'Escalation queue: {cfg.project_root}/{cfg.escalation.queue_dir}\n'
        )
        system_prompt = load_skill_system_prompt('escalation-watcher-auto')
        escalation_url = f'http://{cfg.escalation.host}:{cfg.escalation.port}/mcp'
        mcp_config = self.mcp.mcp_config_json(escalation_url=escalation_url)
        timeout_secs = cfg.watcher_rotation_hours * 3600 + _WATCHER_TIMEOUT_GRACE_SECS

        return await invoke_with_cap_retry(
            self.usage_gate,
            'Escalation watcher (auto)',
            invoke_fn=invoke_agent,
            cost_store=self.cost_store,
            run_id=self._run_id or '',
            project_id=cfg.fused_memory.project_id,
            role='escalation-watcher-auto',
            task_id='',
            prompt=user_prompt,
            system_prompt=system_prompt,
            cwd=cfg.project_root,
            model=cfg.watcher_model,
            max_turns=cfg.watcher_max_turns,
            max_budget_usd=cfg.watcher_rotation_budget_usd,
            effort=cfg.watcher_effort,
            backend=cfg.watcher_backend,
            mcp_config=mcp_config,
            timeout_seconds=timeout_secs,
            allowed_tools=_WATCHER_ALLOWED_TOOLS,
            disallowed_tools=_WATCHER_DISALLOWED_TOOLS,
        )

    async def _watcher_supervisor_loop(self) -> None:
        """Supervisor loop — restart watcher rotations until shutdown.

        Classification:
          clean exit (success=True and not timed_out) → immediate restart,
              reset consecutive-unclean counter.
          unclean exit (success=False OR timed_out=True) → record timestamp,
              apply exponential backoff, check crashloop guard (step-12).
          asyncio.CancelledError → re-raise (clean shutdown signal).
          generic Exception → treated as unclean (logged via logger.exception).

        Crashloop detection is added in step-12.
        """
        consecutive_unclean: int = 0
        consecutive_degenerate_clean: int = 0  # task 1430: exponential floor on fast-clean exits
        while True:
            start = time.monotonic()
            try:
                result = await self._run_watcher_rotation()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    'Escalation-watcher-auto rotation raised unexpected exception'
                )
                # Treat as unclean: backoff below
                result = None  # sentinel for unclean path
            end = time.monotonic()  # captured after rotation; reused as 'now' below

            clean = (
                result is not None
                and bool(getattr(result, 'success', False))
                and not bool(getattr(result, 'timed_out', False))
            )

            # Best-effort digest check after each rotation (task 1327).
            # Single call site — applies to both clean and unclean paths.
            # Never allowed to break the supervisor: CancelledError re-raised,
            # AttributeError re-raised (task 1449: surfaces state-init drift to
            # tests rather than converting it to a silent warning log),
            # every other runtime exception logged and swallowed.
            try:
                await self._maybe_write_digest()
            except asyncio.CancelledError:
                raise
            except AttributeError:
                raise  # task 1449: surface fixture-drift / state-init bugs to tests
            except Exception:
                logger.warning(
                    '_maybe_write_digest raised unexpectedly in supervisor loop '
                    '(best-effort swallowed)',
                    exc_info=True,
                )

            if clean:
                # Healthy rotation completed — reset backoff.
                # Enforce a minimum floor between rotations even on clean exit:
                # prevents back-to-back opus invocations if the agent self-exits
                # near-instantly due to misconfiguration (empty queue, SKILL.md
                # drift, etc.).  Mirrors the terminal-status-watcher's always-sleep
                # pattern.  watcher_subprocess_restart_backoff_secs (default 30s)
                # doubles as the clean-restart floor.
                consecutive_unclean = 0
                # Cost-runaway guard: track degenerate-clean exits (task 1388).
                # A healthy rotation takes at least watcher_misconfigured_min_rotation_secs
                # seconds; one that exits faster is degenerate (empty queue, SKILL.md
                # drift, misconfigured env).
                duration = end - start
                # _check_watcher_guard: append+evict+trip, fully defensive.
                # Only called when duration is below the healthy-rotation floor;
                # reuses watcher_crashloop_window_secs as the burst-detection
                # window — semantically identical for both failure modes.
                if duration < self.config.watcher_misconfigured_min_rotation_secs:
                    # Increment before the guard call — mirrors the unclean path
                    # convention (consecutive_unclean += 1 before _check_watcher_guard).
                    consecutive_degenerate_clean += 1
                    if await self._check_watcher_guard(
                        self._watcher_degenerate_clean_exits,
                        'watcher_misconfigured',
                        self.config.watcher_max_misconfigured_clean_exits,
                        self.config.watcher_crashloop_window_secs,
                        end,
                    ):
                        return
                    # Degenerate-clean, no trip — apply exponential floor (task 1430).
                    # Fast rotation signals potential cost-runaway; grow the sleep
                    # exponentially to slow the burn rate before the trip arms.
                    # Mirrors the unclean-exit exponential backoff in the section below.
                    # Clamp exp to 60 (2**60 ≈ 1.15e18) to prevent OverflowError at very
                    # high consecutive counts; the outer min still bounds to _WATCHER_MAX_BACKOFF_SECS
                    # so observable behaviour is unchanged for any realistic consecutive value.
                    exp = min(consecutive_degenerate_clean - 1, 60)
                    floor = min(
                        self.config.watcher_subprocess_restart_backoff_secs * (2 ** exp),
                        _WATCHER_MAX_BACKOFF_SECS,
                    )
                    logger.warning(
                        'Escalation-watcher-auto rotation completed cleanly but fast '
                        '(consecutive_degenerate=%d floor=%.1fs)',
                        consecutive_degenerate_clean,
                        floor,
                    )
                    try:
                        await asyncio.sleep(floor)
                    except asyncio.CancelledError:
                        raise  # clean shutdown
                    except Exception:
                        logger.exception(
                            'Escalation-watcher-auto supervisor: unexpected error in '
                            'degenerate-clean-path floor sleep — sleeping base backoff '
                            'to avoid silent supervisor death'
                        )
                        try:
                            await asyncio.sleep(
                                self.config.watcher_subprocess_restart_backoff_secs
                            )
                        except asyncio.CancelledError:
                            raise
                else:
                    # Healthy clean (duration >= watcher_misconfigured_min_rotation_secs):
                    # positive evidence the queue had real work — reset the cost-runaway
                    # counter so the next degenerate burst starts fresh from base.
                    consecutive_degenerate_clean = 0
                    logger.info('Escalation-watcher-auto rotation completed cleanly; restarting')
                    await asyncio.sleep(self.config.watcher_subprocess_restart_backoff_secs)
                continue

            # --- Unclean exit path ---
            # Symmetric with the clean path's ``consecutive_unclean = 0`` reset (which
            # fires on both healthy and degenerate-clean exits): an unclean exit breaks
            # any in-progress degenerate-clean burst, so reset the floor counter.
            # Without this, interleaved degen/unclean workloads grow
            # consecutive_degenerate_clean unboundedly across bursts (task 1443).
            consecutive_degenerate_clean = 0
            consecutive_unclean += 1
            # _check_watcher_guard handles append+evict+trip defensively.  Called
            # outside any broad try/except so pause_scheduler failure cannot be
            # swallowed by the backoff-degradation handler (S2 defect prevention).
            if await self._check_watcher_guard(
                self._watcher_unclean_exits,
                'watcher_crashloop',
                self.config.watcher_max_crashloop_restarts,
                self.config.watcher_crashloop_window_secs,
                end,
            ):
                return

            # No trip — apply exponential backoff.
            try:
                # Clamp exp to 60; see degenerate-clean path comment above re: overflow.
                exp = min(consecutive_unclean - 1, 60)
                backoff = min(
                    self.config.watcher_subprocess_restart_backoff_secs * (2 ** exp),
                    _WATCHER_MAX_BACKOFF_SECS,
                )
                logger.warning(
                    'Escalation-watcher-auto rotation exited uncleanly '
                    '(consecutive=%d backoff=%.1fs)',
                    consecutive_unclean,
                    backoff,
                )
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise  # clean shutdown
            except Exception:
                logger.exception(
                    'Escalation-watcher-auto supervisor: unexpected error in '
                    'unclean-path backoff — sleeping base backoff to avoid '
                    'silent supervisor death'
                )
                # Degrade gracefully: sleep base backoff then continue the loop.
                # Re-raise CancelledError from the fallback sleep if received.
                try:
                    await asyncio.sleep(
                        self.config.watcher_subprocess_restart_backoff_secs
                    )
                except asyncio.CancelledError:
                    raise

    async def _check_watcher_guard(
        self,
        exits_deque: deque[float],
        reason: str,
        max_count: int,
        window_secs: float,
        now: float,
    ) -> bool:
        """Append *now* to *exits_deque*, evict stale entries, trip if threshold met.

        Returns True  → guard tripped; caller should stop the supervisor.
        Returns False → threshold not reached or bookkeeping error; caller continues.

        Bookkeeping errors (deque ops) are logged and suppressed so a rare
        collection anomaly cannot silently kill the supervisor.

        pause_scheduler failure is also caught and logged, but the method
        still returns True so a broken pause_scheduler cannot defeat the stop.
        CancelledError is always re-raised (clean shutdown signal).
        """
        try:
            exits_deque.append(now)
            while exits_deque and exits_deque[0] < now - window_secs:
                exits_deque.popleft()
        except Exception:
            logger.exception(
                'watcher %s guard: bookkeeping error — skipping trip check',
                reason,
            )
            return False
        if len(exits_deque) < max_count:
            return False
        logger.error(
            'Escalation-watcher-auto %s guard tripped '
            '(%d exits in %ds window) — pausing scheduler',
            reason,
            len(exits_deque),
            window_secs,
        )
        try:
            await self.pause_scheduler(reason)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                'pause_scheduler raised on %s trip; stopping supervisor anyway',
                reason,
            )
        return True  # always stop, even if pause_scheduler raised

    async def _scan_for_terminal_active_tasks(self) -> int:
        """Single pass: cancel any active workflow whose task is terminal.

        Returns the number of workflows cancelled.  Extracted so tests can
        drive the scan deterministically.
        """
        from orchestrator.task_status import TERMINAL_STATUSES

        active_ids = list(self._workflow_cancel_events.keys())
        if not active_ids:
            return 0
        statuses, error = await self.scheduler.get_statuses(active_ids)
        if error is not None:
            return 0
        cancelled = 0
        for task_id, status in statuses.items():
            if status in TERMINAL_STATUSES and self.cancel_workflow(task_id):
                logger.info(
                    'Terminal-status watcher: cancelling workflow for task '
                    '%s (status=%s)', task_id, status,
                )
                cancelled += 1
        return cancelled

    # ------------------------------------------------------------------
    # Stranded-in-progress periodic sweep (Fix 4)
    # ------------------------------------------------------------------

    def _start_stranded_reconcile(self) -> None:
        """Start the periodic stranded-in-progress sweep.

        Mirrors the terminal-status watcher: a long-lived asyncio.Task
        wakes every ``stranded_reconcile_interval_secs`` and re-runs
        ``_reconcile_stranded_in_progress(mid_run=True)``.  The mid_run
        filter skips tasks the scheduler is actively dispatching, so the
        sweep can't race a healthy workflow.
        """
        if not self.config.stranded_reconcile_enabled:
            return
        if (
            self._stranded_reconcile_task is not None
            and not self._stranded_reconcile_task.done()
        ):
            return
        self._stranded_reconcile_task = asyncio.create_task(
            self._stranded_reconcile_loop(),
            name='stranded-reconcile',
        )
        logger.info(
            'Stranded-in-progress reconcile started (interval=%.0fs)',
            self.config.stranded_reconcile_interval_secs,
        )

    async def _stop_stranded_reconcile(self) -> None:
        """Cancel the stranded-in-progress sweep loop."""
        if self._stranded_reconcile_task is not None:
            self._stranded_reconcile_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._stranded_reconcile_task
            self._stranded_reconcile_task = None
            logger.info('Stranded-in-progress reconcile stopped')

    async def _stranded_reconcile_loop(self) -> None:
        """Wake periodically and run the mid-run stranded sweep."""
        interval = self.config.stranded_reconcile_interval_secs
        while True:
            try:
                await asyncio.sleep(interval)
                await self._reconcile_stranded_in_progress(mid_run=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception('Stranded-in-progress reconcile pass failed')

    def _on_escalation(self, escalation) -> None:
        """Callback when any escalation is submitted — wake the waiting workflow/steward."""
        # Increment before the event-set logic so the counter reflects every submission.
        # Best-effort observability counter — incremented here from arbitrary callbacks
        # without a lock.  _maybe_write_digest snapshots it once at entry so concurrent
        # callbacks cannot cause a double-skip between the threshold check and the advance.
        # May drift by a small constant under concurrency; not a correctness gate.
        self._escalation_event_count += 1  # task 1327 AFK hardening
        event = self._escalation_events.get(escalation.task_id)
        if event:
            event.set()

    def _on_escalation_resolved(self, escalation) -> None:
        """Callback when an escalation is resolved — wake the waiting workflow."""
        # Increment for any status transition (resolved or dismissed) — both are
        # escalation events that the EWA digest needs to count.
        # Best-effort observability counter — same concurrency caveat as _on_escalation
        # above; _maybe_write_digest snapshots it at entry to avoid double-skip drift.
        self._escalation_event_count += 1  # task 1327 AFK hardening
        event = self._escalation_events.get(escalation.task_id)
        if event:
            event.set()

        # Un-halt the merge queue only when the escalation that OWNS the
        # halt resolves. Prior versions matched on category alone, which
        # let any wip_conflict resolve release the halt — leaving the real
        # blocker's escalation pending (phantom-L1 bug, esc-1888-57 on reify
        # 2026-04-16). The owner pointer is the single source of truth.
        if (
            self._merge_worker is not None
            and self._merge_worker.is_halt_owner(escalation.id)
        ):
            self._merge_worker.unhalt_wip()
            logger.info(
                'Merge queue un-halted: halt owner %s resolved', escalation.id,
            )

    def get_merge_halt_status(self) -> dict[str, Any]:
        """Inspect the merge queue's halt state for operator tooling."""
        if self._merge_worker is None:
            return {'wired': False}
        return {
            'wired': True,
            'halted': self._merge_worker.is_wip_halted,
            'owner_esc_id': self._merge_worker.halt_owner_esc_id,
        }

    def force_unhalt_merge_queue(self, reason: str) -> dict[str, Any]:
        """Operator escape hatch for orphan halts (no escalation owns the halt).

        Refuses to act if the halt has an active owning escalation — operators
        must use ``resolve_issue(owner_esc_id)`` on those, since the legitimate
        unhalt path (``Harness._on_escalation_resolved``) is intact.
        """
        if self._merge_worker is None:
            return {'unhalted': False, 'error': 'merge worker not initialised'}
        if not self._merge_worker.is_wip_halted:
            return {'unhalted': False, 'reason': 'queue not halted'}
        owner = self._merge_worker.halt_owner_esc_id
        if owner is not None and self._escalation_queue is not None:
            try:
                esc = self._escalation_queue.get(owner)
            except Exception:
                esc = None
            if esc is not None and esc.status not in ('resolved', 'dismissed'):
                return {
                    'unhalted': False,
                    'error': (
                        f'halt is owned by active escalation {owner!r} — '
                        f'use resolve_issue({owner!r}, ...) instead'
                    ),
                    'owner_esc_id': owner,
                }
        self._merge_worker.unhalt_wip()
        logger.warning(
            'Force-unhalted merge queue (prior_owner=%s, reason=%r)',
            owner, reason,
        )
        return {'unhalted': True, 'prior_owner': owner, 'reason': reason}

    # ------------------------------------------------------------------ #
    # Scheduler park-and-stop pause (task 1322)                           #
    # ------------------------------------------------------------------ #

    async def pause_scheduler(self, reason: str) -> None:
        """Pause the scheduler so acquire_next() returns None until resumed.

        1. Delegates to ``scheduler.pause(reason)`` (idempotent in-memory state).
        2. Persists via ``RunStore.save_scheduler_pause`` (best-effort).
        3. Emits ``EventType.scheduler_paused`` (best-effort).
        4. Logs a WARNING so the operator sees it.

        Called directly by sibling tasks (cost-ceiling 1323, EWA digest 1327)
        and also wired as the callback for Scheduler's park-stop trip detector.

        Race note: when this is invoked as the park-stop trip callback, the
        scheduler's synchronous latch has already set ``is_paused=True`` with
        the trip reason, so a different external caller racing in between the
        latch and this callback executing would briefly see is_paused=True
        with the trip reason but no disk row yet.  After this callback runs,
        disk and memory both reflect the trip reason — first-wins on memory,
        last-write-wins on disk for the same reason value.  Sequential
        external callers after a trip see no-op idempotent in-memory plus a
        disk overwrite with the new reason; this divergence between memory
        and disk for the in-flight pause is accepted as the cost of keeping
        the trip's status write off the synchronous critical path.
        """
        self.scheduler.pause(reason)
        logger.warning('Scheduler paused: %s', reason)

        if self._run_store and self._run_id:
            try:
                self._run_store.save_scheduler_pause(
                    project_id=self.config.fused_memory.project_id,
                    reason=reason,
                    pause_at_iso=datetime.now(UTC).isoformat(),
                    set_by_run_id=self._run_id,
                )
            except Exception:
                logger.warning('pause_scheduler: failed to persist pause state', exc_info=True)

        if self.event_store:
            self.event_store.emit(
                EventType.scheduler_paused,
                data={'reason': reason},
            )

    async def resume_scheduler(self) -> None:
        """Clear the scheduler pause so acquire_next() resumes dispatching.

        1. Delegates to ``scheduler.resume()`` (idempotent).
        2. Clears persistence via ``RunStore.clear_scheduler_pause`` (best-effort).
        3. Emits ``EventType.scheduler_resumed`` (best-effort).
        4. If the pause was caused by an EWA trip (reason starts with 'ewa_trip_'),
           resets ``_ewa_value`` to 0.0 so the next digest step does not immediately
           re-trigger a pause.  EWA decays naturally with alpha=0.3 (~N digests of
           low ratio to fall back below threshold), but an operator-driven resume
           signals that the situation is under control — a clean reset is less
           surprising than watching the EWA trip instantly again.
        5. Logs INFO.
        """
        # Snapshot pause reason BEFORE resume() clears it.
        prior_reason = self.scheduler.pause_reason or ''
        self.scheduler.resume()
        if prior_reason.startswith('ewa_trip_'):
            self._ewa_value = 0.0
            logger.info(
                'resume_scheduler: EWA value reset to 0.0 (prior pause was ewa_trip).'
            )
        logger.info('Scheduler resumed.')

        if self._run_store:
            try:
                self._run_store.clear_scheduler_pause(
                    self.config.fused_memory.project_id,
                )
            except Exception:
                logger.warning('resume_scheduler: failed to clear persisted pause', exc_info=True)

        if self.event_store:
            self.event_store.emit(EventType.scheduler_resumed)

    async def _load_persisted_scheduler_pause(self) -> None:
        """Restore scheduler pause state from runs.db on restart.

        Called once from ``Harness.run()`` after the RunStore is initialised.
        If a pause record exists, the scheduler is paused in-memory using
        ``scheduler.pause()`` directly — NOT ``pause_scheduler()`` — so that
        no duplicate persistence write or event emission occurs (the row is
        already on disk from the prior run that set it).

        Logs a WARNING with the persisted reason and pause_at so the operator
        is alerted on startup.  Any failure is caught and logged but never
        blocks startup.
        """
        if not self._run_store:
            return
        try:
            record = self._run_store.load_scheduler_pause(
                self.config.fused_memory.project_id,
            )
            if record:
                reason = record.get('reason', '<unknown reason>')
                pause_at = record.get('pause_at', '<unknown time>')
                restored_from_run_id = record.get('set_by_run_id', '<unknown>')
                logger.warning(
                    'Scheduler pause persisted from prior run — restoring. '
                    'reason=%r  pause_at=%r  set_by_run_id=%r  '
                    '(call Harness.resume_scheduler() to clear)',
                    reason,
                    pause_at,
                    restored_from_run_id,
                )
                self.scheduler.pause(reason)
                # Emit a distinct event so the timeline self-documents
                # cross-run continuity.  Operators querying the event log
                # for a run that starts with dispatch halted can see WHY
                # without having to cross-reference the previous run_id.
                if self.event_store:
                    self.event_store.emit(
                        EventType.scheduler_pause_restored,
                        data={
                            'reason': reason,
                            'pause_at': pause_at,
                            'restored_from_run_id': restored_from_run_id,
                        },
                    )
        except Exception:
            logger.warning(
                '_load_persisted_scheduler_pause: failed to read pause state',
                exc_info=True,
            )

    # ------------------------------------------------------------------ #
    # Digest + EWA trip (task 1327)                                       #
    # ------------------------------------------------------------------ #

    async def _maybe_write_digest(self) -> None:
        """Check if it is time to write a digest; if so gather data, write, and update EWA.

        Called from _watcher_supervisor_loop after each rotation (best-effort).
        Any exception is swallowed and logged as a WARNING — a failed digest must
        never break the watcher supervisor loop.

        Algorithm:
        1. Early-return when digest_enabled=False.
        2. Snapshot _escalation_event_count (best-effort; see note below).
        3. Early-return when (snapshot - last_count) < N.
        4. Snapshot escalation delta.
        5. Compute window (last window_end → now).
        6. Aggregate escalation stats (fail-open).
        7. Count done tasks in EventStore (fail-open).
           done_count from EventStore is the single source of truth for both
           the rendered digest figure and the update_ewa input — ensuring the
           operator sees exactly the number that drove the EWA decision.
        8. Query cost stats from CostStore (fail-open).
        9. Read parked-task counts from Scheduler state.
        10. Update EWA using EventStore done_count.
        11. Compute trip flag and anomaly flags.
        12. Write digest file (fail-open via write_digest_entry).
        13. Advance counters/state using the snapshot from step 2 (not the
            live counter), so that concurrent callbacks firing inside this
            function do not silently skip counted events.
        14. If tripped and not already paused, call pause_scheduler (post-write).

        Note on _escalation_event_count: callbacks fire inline on the asyncio
        event loop thread, so there are no real concurrent writers — the
        snapshot at step 2 guards against the logical interleaving where a
        callback runs at an await point inside this function, not against torn
        integer writes.  Snapshotting once at step 2 makes the threshold check
        and the advance consistent — concurrent callbacks cannot cause a
        "double-skip" where the advance overshoots the events that triggered
        this digest.  The counter is best-effort observability; a small drift
        is acceptable.

        Task 1327 AFK hardening.
        """
        try:
            # (1) Early-return if disabled.
            if not self.config.digest_enabled:
                return

            # (2) Snapshot _escalation_event_count so the threshold check (3)
            # and the advance (13) are consistent even if a concurrent callback
            # increments the live counter between those two reads.
            event_count_snapshot = self._escalation_event_count

            # (3) Early-return if not enough new events.
            diff = event_count_snapshot - self._last_digest_event_count
            if diff < self.config.digest_every_n_escalations:
                return

            # (4) Snapshot escalation delta.
            escalations_in_step = diff

            # (5) Compute window timestamps.
            window_end = datetime.now(UTC).isoformat()
            window_start = self._last_digest_window_end_iso
            if not window_start:
                # First digest: use 24h ago as window start.
                window_start = (datetime.now(UTC) - timedelta(hours=24)).isoformat()

            # (6) Gather escalation stats (fail-open via aggregate_escalations).
            escalations_dir = (
                Path(self.config.project_root) / self.config.escalation.queue_dir
            )
            escalation_stats = digest_mod.aggregate_escalations(
                escalations_dir, window_start, window_end
            )

            # (7) Count done tasks in window (fail-open via count_done_in_window).
            events_db_path = self.event_store.db_path if self.event_store else None
            done_count = (
                digest_mod.count_done_in_window(
                    events_db_path, window_start, window_end
                )
                if events_db_path is not None
                else 0
            )

            # (8) Cost stats (fail-open via cost_in_window).
            cost_stats = await digest_mod.cost_in_window(
                self.cost_store, window_start, window_end
            )

            # (9) Parked-task counts via public Scheduler properties (task 1327).
            parked_live = self.scheduler.parked_live_count
            parked_window_churn = self.scheduler.parked_window_churn_count

            # (10) Update EWA — done_count from EventStore (step 7) is the single
            # source of truth so the EWA input matches the rendered digest figure.
            new_ewa = digest_mod.update_ewa(
                prev_ewa=self._ewa_value,
                escalations_in_step=escalations_in_step,
                done_in_step=done_count,
                alpha=self.config.digest_ewa_alpha,
            )

            # (11) Trip flag and anomaly flags.
            tripped = new_ewa >= self.config.digest_ewa_threshold
            anomaly_flags = {
                'cost_spike': (
                    cost_stats.watcher_cost_in_window
                    > self.config.watcher_daily_cost_ceiling_usd * 0.5
                ),
                'park_spike': (
                    parked_window_churn >= self.config.park_stop_parked_threshold
                ),
                'ewa_above_threshold': tripped,
                'infra_dedupe_active': escalation_stats.dedupe_children_total > 0,
            }

            # (12) Resolve digest directory, assemble inputs, and write digest file
            # (fail-open via write_digest_entry — never raises).
            if self.config.digest_dir:
                digest_dir = Path(self.config.digest_dir)
            else:
                digest_dir = Path(self.config.project_root) / 'data' / 'digests'

            inputs = digest_mod.DigestInputs(
                window_start_iso=window_start,
                window_end_iso=window_end,
                escalation_stats=escalation_stats,
                done_count=done_count,
                cost_stats=cost_stats,
                parked_live=parked_live,
                parked_window_churn=parked_window_churn,
                ewa_value=new_ewa,
                ewa_threshold=self.config.digest_ewa_threshold,
                tripped=tripped,
                anomaly_flags=anomaly_flags,
                watcher_clusters=[],
                dry_run_proposals=[],
            )

            digest_mod.write_digest_entry(digest_dir, inputs)

            # (13) Advance EWA state and counters.
            # Use event_count_snapshot (not the live self._escalation_event_count)
            # so that concurrent callbacks that fired after the snapshot are not
            # silently skipped — they will be counted in the next digest step.
            self._ewa_value = new_ewa
            self._last_digest_event_count = event_count_snapshot
            self._last_digest_window_end_iso = window_end

            # (14) EWA trip: pause scheduler AFTER the digest is written so the
            # markdown captures the trip-causing state.
            if tripped and not self.scheduler.is_paused:
                await self.pause_scheduler(f'ewa_trip_{new_ewa:.4f}')

        except asyncio.CancelledError:
            raise
        except AttributeError:
            raise  # task 1449: surface fixture-drift / state-init bugs to tests
        except Exception:
            logger.warning('_maybe_write_digest: unexpected error (fail-open)', exc_info=True)
