"""Top-level orchestration loop."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING

from shared.cli_invoke import AllAccountsCappedException, invoke_with_cap_retry
from shared.cost_store import CostStore

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.invoke import invoke_agent
from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.mcp_lifecycle import McpLifecycle, mcp_call
from orchestrator.review_checkpoint import ReviewCheckpoint
from orchestrator.run_store import RunStore
from orchestrator.scheduler import Scheduler, files_to_modules
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
        self.scheduler = Scheduler(config)
        self.briefing = BriefingAssembler(config)
        self.report = HarnessReport()
        self._recovered_plans: dict[str, dict] = {}

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

        # Singleton lock — held for the duration of run()
        self._lock_file: IO | None = None

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

            # 1c1. Start orphan L0 reaper (non-fatal) — catches escalations
            # whose task_id has no active workflow/steward (e.g. reviewer
            # emits against a synthetic task_id, or a workflow crashed
            # before its steward could claim them).
            self._start_orphan_l0_reaper()

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

            # 2. Parse PRD into tasks (skipped when no PRD given)
            if prd_path is not None:
                logger.info(f'Parsing PRD: {prd_path}')
                _pre_statuses, _ = await self.scheduler.get_statuses()
                pre_ids = set(_pre_statuses.keys())
                await self._populate_tasks(prd_path)

                # 2a. Tag newly-created tasks with PRD source
                await self._tag_prd_metadata(prd_path, pre_ids)
            else:
                logger.info('No PRD given — running existing tasks')
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
                        'No PRD given and no pending tasks found. '
                        'Pass --prd to decompose a PRD, or create tasks first.'
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

                assignment = await self.scheduler.acquire_next()

                if assignment is None:
                    if not active:
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

            # Commit accumulated task status changes so they survive
            # working-tree resets and are visible to future merge worktrees.
            if self.report.completed > 0:
                sha = await self.git_ops.commit_task_statuses()
                if sha:
                    logger.info(f'Committed task statuses: {sha[:8]}')

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

    async def _populate_tasks(self, prd_path: Path) -> None:
        """Use taskmaster parse_prd to decompose PRD into tasks."""
        try:
            result = await mcp_call(
                f'{self.mcp.url}/mcp',
                'tools/call',
                {
                    'name': 'parse_prd',
                    'arguments': {
                        'input': str(prd_path.resolve()),
                        'project_root': str(self.config.project_root),
                    },
                },
                timeout=120,  # PRD parsing can be slow
            )
            content = result.get('result', {}).get('content', [{}])
            text = content[0].get('text', '') if content else ''
            if content and content[0].get('isError'):
                raise RuntimeError(f'parse_prd error: {text}')
            logger.info(f'PRD parsed: {text[:200]}')
        except RuntimeError as e:
            if 'already contains' in str(e):
                logger.info('Task tree already populated — skipping parse_prd')
            else:
                raise
        except Exception as e:
            raise RuntimeError(f'Failed to parse PRD: {e}') from e

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
        """Invoke a Claude agent to tag each task with the code modules it touches.

        Uses structured output to get a JSON mapping of task_id → [modules],
        then persists via scheduler.update_task().

        When *force* is ``True``, retag all non-done/cancelled tasks even if
        they already have module metadata.
        """
        tasks = await self.scheduler.get_tasks()

        skip_statuses = {'done', 'cancelled'}
        untagged = []
        for t in tasks:
            if t.get('status') in skip_statuses:
                continue
            if not force:
                metadata = t.get('metadata') or {}
                modules = metadata.get('modules', [])
                if modules:
                    continue
            untagged.append(t)

        if not untagged:
            logger.info('No tasks to tag — skipping')
            return

        if force:
            logger.info(f'Force-retagging {len(untagged)} tasks with module metadata')

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
                                'description': 'Predicted file paths this task will create or modify',
                            },
                            'modules': {'type': 'array', 'items': {'type': 'string'}},
                        },
                        'required': ['id', 'files', 'modules'],
                    },
                },
            },
            'required': ['tasks'],
        }

        prompt = f"""\
Given these tasks and this codebase structure, predict which files each task
will create or modify, and which code modules (directories) it will touch.

Be specific and exhaustive with file predictions — include source files AND
test files. Use paths relative to the project root. The `files` field is used
to derive concurrency locks, so accuracy prevents unnecessary serialization.

The `modules` field should list directory-level groupings (e.g. "src/backends",
"src/server", "tests") as a human-readable summary.

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
            modules = entry.get('modules', [])
            files = entry.get('files', [])
            if task_id and (modules or files):
                metadata: dict = {}
                if files:
                    metadata['files'] = files
                if modules:
                    metadata['modules'] = modules
                await self.scheduler.update_task(task_id, json.dumps(metadata))
                # Also populate in-memory cache so modules are available
                # immediately without re-fetching from taskmaster
                depth = self.config.lock_depth
                if files:
                    derived = files_to_modules(files, depth)
                    if derived:
                        self.scheduler._module_cache[task_id] = derived
                elif modules:
                    from orchestrator.scheduler import normalize_lock
                    self.scheduler._module_cache[task_id] = [
                        normalize_lock(m, depth) for m in modules
                    ]
                tagged_count += 1

        logger.info(f'Tagged {tagged_count}/{len(untagged)} tasks with module metadata')
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
                logger.info(
                    f'Recovery: worktree {task_id} has plan but no '
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

    async def _reconcile_stranded_in_progress(self) -> None:
        """Startup sweep: revert stranded in-progress tasks to pending.

        Examines every task that is currently in-progress and checks whether
        it has a live claimant via plan.lock / owner_pid.  Any task without a
        live claimant is reverted to pending so the scheduler can re-acquire it.

        This method is called AFTER _recover_crashed_tasks() (which may unlink
        plan.lock for recovered worktrees) and BEFORE the first
        scheduler.acquire_next() call, so self._dispatched is always empty here.
        """
        statuses, _ = await self.scheduler.get_statuses()
        reverted = 0

        for tid, status in statuses.items():
            if status != 'in-progress':
                continue

            worktree_path = self.git_ops.worktree_base / tid
            lock_path = worktree_path / '.task' / 'plan.lock'

            if not lock_path.exists():
                # No worktree or no lock → orphan, revert.
                # If the worktree directory exists and we haven't promised to
                # resume this task's plan, clean it up so the scheduler can
                # create a fresh worktree on re-acquisition without colliding.
                if worktree_path.exists() and tid not in self._recovered_plans:
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
                reverted += 1
                continue

            # Lock exists — check whether the owner is still alive.
            owner_alive = False
            try:
                lock_data = json.loads(lock_path.read_text())
                if not isinstance(lock_data, dict):
                    raise ValueError('plan.lock is not a JSON object')
                owner_pid = lock_data.get('owner_pid')
                if owner_pid is None:
                    # Missing or null owner_pid — surface this in logs so
                    # unexpected lock formats don't fail silently.
                    logger.warning(
                        'Reconcile: plan.lock for task %s has no owner_pid;'
                        ' treating as stale',
                        tid,
                    )
                else:
                    try:
                        owner_alive = _pid_alive(int(owner_pid))
                    except (TypeError, ValueError):
                        # owner_pid is non-numeric — treat lock as stale.
                        owner_alive = False
            except (OSError, json.JSONDecodeError, ValueError):
                # OSError: file read failure.
                # JSONDecodeError: malformed JSON text.
                # ValueError: malformed lock structure (non-dict JSON value).
                # (Unexpected exception types propagate — they indicate a bug.)
                owner_alive = False

            if owner_alive:
                # Live claimant — leave the task alone.
                continue

            # Stale lock — clear it and revert.
            if tid not in self._recovered_plans:
                # Full cleanup: remove worktree dir + branch so re-acquisition
                # creates a fresh worktree without colliding.
                try:
                    await self.git_ops.cleanup_worktree(worktree_path, tid)
                except Exception:
                    logger.warning(
                        'Reconcile: cleanup_worktree failed for task %s'
                        ' (stale-lock); continuing',
                        tid, exc_info=True,
                    )
                # Unconditionally unlink the stale lock so the next sweep
                # doesn't re-encounter it.  If cleanup_worktree succeeded the
                # whole worktree dir is gone and this is a cheap no-op;
                # if cleanup failed we still guarantee the lock is cleared.
                with contextlib.suppress(OSError):
                    lock_path.unlink(missing_ok=True)
            else:
                # Plan will be resumed — preserve worktree, only clear stale lock.
                with contextlib.suppress(OSError):
                    lock_path.unlink()
            await self.scheduler.set_task_status(tid, 'pending')
            logger.info(
                'Reconcile: reverted task %s to pending (reason=stale-lock)', tid
            )
            reverted += 1

        if reverted:
            logger.info('Reconcile: %d stranded task(s) reverted to pending', reverted)

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

            recovered_plan = self._recovered_plans.pop(assignment.task_id, None)

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
            requeued = report is not None and report.outcome == WorkflowOutcome.REQUEUED
            self.scheduler.release(assignment.task_id, requeued=requeued)
            sem.release()

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

    async def _start_escalation_server(self) -> None:
        """Start the escalation MCP server as a background asyncio task."""
        if not HAS_ESCALATION:
            logger.info('Escalation package not installed — skipping escalation server')
            return

        queue_dir = Path(self.config.escalation.queue_dir)
        if not queue_dir.is_absolute():
            queue_dir = self.config.project_root / queue_dir
        self._escalation_queue = EscalationQueue(queue_dir)  # type: ignore[possibly-undefined]
        self._escalation_queue.set_notify_callback(self._on_escalation)
        self._escalation_queue.set_resolve_callback(self._on_escalation_resolved)

        # Wire escalation queue into review checkpoint so it can triage
        # escalations the deep reviewer emits against the synthetic review
        # task_id (which has no workflow/steward to handle them).
        if self.review_checkpoint is not None:
            self.review_checkpoint.escalation_queue = self._escalation_queue

        mcp_server = create_server(self._escalation_queue, merge_queue=self._merge_queue, orch_config=self.config, event_store=self.event_store)  # type: ignore[possibly-undefined]
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
        """Dismiss all pending escalations left over from prior orchestrator runs.

        Called right after _start_escalation_server() so that any escalations
        persisted in the queue directory from a previous (crashed or completed)
        run are cleared before the new run begins.  All pending escalations at
        startup are by definition stale — a new run should never inherit
        unresolved escalations from a prior one.
        """
        if self._escalation_queue is None:
            return

        resolution = (
            'Auto-dismissed: orchestrator restarted — stale from prior run'
        )
        count = self._escalation_queue.dismiss_all_pending(resolution)
        if count:
            logger.info(f'Dismissed {count} stale escalation(s) from prior run')

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

    def _on_escalation(self, escalation) -> None:
        """Callback when any escalation is submitted — wake the waiting workflow/steward."""
        event = self._escalation_events.get(escalation.task_id)
        if event:
            event.set()

    def _on_escalation_resolved(self, escalation) -> None:
        """Callback when an escalation is resolved — wake the waiting workflow."""
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
