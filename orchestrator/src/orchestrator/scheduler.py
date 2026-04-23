"""Task selection and module lock management."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass

from shared.locking import files_to_modules, normalize_lock

from orchestrator.config import (
    DEFAULT_TIER,
    PRIORITY_RANK,
    PRIORITY_TIERS,
    TIER_BASE,
    TIER_WIDTH,
    OrchestratorConfig,
    coerce_tier,
)
from orchestrator.event_store import EventStore, EventType
from orchestrator.mcp_lifecycle import mcp_call

# task_skipped events for "effectively infinite" skip thresholds (>= this
# value) are rate-limited to a geometric schedule so the event store is not
# flooded with diagnostics for tasks that will perpetually lose the race.
_INF_SKIP_THRESHOLD: int = 1000
_GEOMETRIC_SKIP_EMIT_COUNTS: frozenset[int] = frozenset({1, 10, 100, 1000, 10000})

logger = logging.getLogger(__name__)

__all__ = [
    'normalize_lock',
    'files_to_modules',
    'TaskAssignment',
    'ModuleLockTable',
    'Scheduler',
]


@dataclass
class TaskAssignment:
    """A task that has been assigned to a workflow slot, with module locks held."""

    task_id: str
    task: dict
    modules: list[str]


class ModuleLockTable:
    """Hierarchical module locking — two modules conflict if one is a prefix
    of the other (parent/child), but siblings are independent.

    Examples (all conflict):
        autopilot/analyze  <->  autopilot/analyze/asr   (parent <-> child)
        src/server         <->  src/server               (exact match)

    Examples (no conflict):
        autopilot/analyze/asr  <->  autopilot/analyze/speech  (siblings)
    """

    def __init__(self, config: OrchestratorConfig):
        self._limits: dict[str, int] = {}
        self._held: dict[str, set[str]] = {}  # task_id -> set of held modules
        # normalized_module -> (owner_task_id, deadline_monotonic)
        self._parked: dict[str, tuple[str, float]] = {}
        self._config = config

    # --- Hierarchy helpers ---

    @staticmethod
    def _conflicts(a: str, b: str) -> bool:
        """Two modules conflict if one is a prefix of the other (or exact match)."""
        return a == b or a.startswith(b + '/') or b.startswith(a + '/')

    def _count_conflicts(self, module: str, exclude_task: str | None = None) -> int:
        """Count how many *other* tasks hold a lock that conflicts with ``module``."""
        count = 0
        for task_id, task_modules in self._held.items():
            if task_id == exclude_task:
                continue
            if any(self._conflicts(held, module) for held in task_modules):
                count += 1
        return count

    # --- Park (reservation) helpers ---

    def _is_parked_blocks(self, module: str, task_id: str, now: float) -> bool:
        """Return True iff any active park hierarchically conflicts with *module*
        and is owned by a different task.

        Expired parks are ignored (they'll be cleaned up on the next prune).
        """
        for parked_module, (owner, deadline) in self._parked.items():
            if owner == task_id:
                continue
            if deadline <= now:
                continue
            if self._conflicts(parked_module, module):
                return True
        return False

    def has_parks(self, task_id: str) -> bool:
        """Return True if *task_id* currently owns any reservation."""
        return any(owner == task_id for owner, _ in self._parked.values())

    def install_parks(
        self, task_id: str, modules: list[str], deadline: float
    ) -> list[str]:
        """Install reservations on the normalized form of *modules* for *task_id*.

        Returns the list of normalized modules actually parked.
        """
        depth = self._config.lock_depth
        installed: list[str] = []
        for m in modules:
            normalized = normalize_lock(m, depth)
            if not normalized:
                continue
            self._parked[normalized] = (task_id, deadline)
            installed.append(normalized)
        return installed

    def clear_parks_for(self, task_id: str) -> None:
        """Remove every reservation owned by *task_id*."""
        self._parked = {
            m: (owner, deadline)
            for m, (owner, deadline) in self._parked.items()
            if owner != task_id
        }

    def prune_expired_parks(self, now: float) -> list[str]:
        """Drop parks whose deadline has passed. Returns evicted owner task IDs
        (deduplicated, preserving first-seen order)."""
        expired_modules = [
            m for m, (_, deadline) in self._parked.items() if deadline <= now
        ]
        evicted: list[str] = []
        for m in expired_modules:
            owner, _ = self._parked.pop(m)
            if owner not in evicted:
                evicted.append(owner)
        return evicted

    # --- Limit lookup (unchanged) ---

    def _limit_for(self, module: str) -> int:
        module = normalize_lock(module, self._config.lock_depth)
        if module not in self._limits:
            mc = self._config.for_module(module)
            if mc and mc.module_overrides and module in mc.module_overrides:
                self._limits[module] = mc.module_overrides[module]
            elif module in self._config.module_overrides:
                self._limits[module] = self._config.module_overrides[module]
            elif mc and mc.max_per_module is not None:
                self._limits[module] = mc.max_per_module
            else:
                self._limits[module] = self._config.max_per_module
        return self._limits[module]

    # --- Public API ---

    def is_held(self, task_id: str) -> bool:
        """Return True if task_id currently holds any module locks."""
        return task_id in self._held

    def try_acquire(self, task_id: str, modules: list[str]) -> bool:
        """Non-blocking attempt to acquire all module locks.

        Uses hierarchical conflict detection: a lock on ``A/B`` conflicts with
        ``A/B/C`` (and vice-versa) but NOT with ``A/D``.  Also refuses if any
        requested module hierarchically conflicts with an active reservation
        owned by a different task (see ``install_parks``).

        Returns True if all acquired, False if any unavailable.
        """
        depth = self._config.lock_depth
        normalized = list({normalize_lock(m, depth) for m in modules})
        now = time.monotonic()

        # Check every requested module against all other tasks' held locks and
        # active reservations owned by other tasks.
        for module in normalized:
            if self._count_conflicts(module, exclude_task=task_id) >= self._limit_for(module):
                return False
            if self._is_parked_blocks(module, task_id, now):
                return False

        self._held[task_id] = set(normalized)
        logger.info(f'Task {task_id} acquired locks: {normalized}')
        return True

    def release(self, task_id: str) -> None:
        """Release all module locks held by a task."""
        modules = self._held.pop(task_id, set())
        if modules:
            logger.info(f'Task {task_id} released locks: {list(modules)}')

    def release_subset(self, task_id: str, modules: list[str]) -> list[str]:
        """Drop a subset of the task's held modules. Returns the normalized
        modules actually released (may be empty). Removes the task's entry
        entirely when no held modules remain so downstream iteration over
        ``_held`` behaves the same as after a full ``release``.
        """
        held = self._held.get(task_id)
        if not held:
            return []
        depth = self._config.lock_depth
        to_drop = {normalize_lock(m, depth) for m in modules} & held
        if not to_drop:
            return []
        held.difference_update(to_drop)
        if not held:
            del self._held[task_id]
        released = sorted(to_drop)
        logger.info(f'Task {task_id} released subset: {released}')
        return released

    def try_acquire_additional(self, task_id: str, additional: list[str]) -> bool:
        """Non-blocking attempt to expand lock set for a task."""
        depth = self._config.lock_depth
        current = self._held.get(task_id, set())
        new_modules = [
            normalize_lock(m, depth)
            for m in additional
            if normalize_lock(m, depth) not in current
        ]
        if not new_modules:
            return True

        now = time.monotonic()
        for module in new_modules:
            if self._count_conflicts(module, exclude_task=task_id) >= self._limit_for(module):
                return False
            if self._is_parked_blocks(module, task_id, now):
                return False

        self._held[task_id].update(new_modules)
        logger.info(f'Task {task_id} expanded locks: {new_modules}')
        return True


class Scheduler:
    """Selects next eligible task and manages module locks."""

    def __init__(self, config: OrchestratorConfig, event_store: EventStore | None = None):
        self.config = config
        self.lock_table = ModuleLockTable(config)
        self.event_store = event_store
        self._dispatched: set[str] = set()
        self._memory_url = config.fused_memory.url
        self._project_root = str(config.project_root)
        self._module_cache: dict[str, list[str]] = {}  # task_id -> expanded modules
        self._fallback_warned: set[str] = set()  # task IDs already warned about fallback
        self._requeue_until: dict[str, float] = {}  # task_id -> monotonic deadline
        # --- Fairness state (see orchestrator.config.FairnessConfig) ---
        self._skip_count: dict[str, int] = {}  # task_id -> consecutive top-skip count
        self._task_start_times: dict[str, float] = {}  # task_id -> monotonic start
        self._recent_durations: deque[float] = deque(
            maxlen=config.fairness.median_window
        )
        # Per-tier cap bookkeeping: remember the effective priority of every
        # currently-dispatched task so acquire_next can count slots at-or-below
        # a candidate's tier without re-walking the full task graph.
        self._dispatched_priority: dict[str, str] = {}
        # Age-anchor bookkeeping for score(): first time we see a task as
        # pending, we record its age baseline.  Cleared on transition to any
        # non-pending status so a cancelled->pending resurrection starts
        # fresh (no accumulated age).
        self._pending_anchor: dict[str, int] = {}
        self._was_non_pending: set[str] = set()

    async def get_tasks(self) -> list[dict]:
        """Fetch all tasks from fused-memory/taskmaster."""
        try:
            result = await mcp_call(
                f'{self._memory_url}/mcp',
                'tools/call',
                {
                    'name': 'get_tasks',
                    'arguments': {'project_root': self._project_root},
                },
                timeout=15,
            )
            content = result.get('result', {}).get('content', [])
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    import json
                    data = json.loads(block['text'])
                    # Handle taskmaster's nested response format: {data: {tasks: [...]}}
                    if 'data' in data and isinstance(data['data'], dict):
                        tasks = data['data'].get('tasks', [])
                    else:
                        tasks = data.get('tasks', [])
                    return tasks
        except Exception as e:
            # logger.exception preserves the traceback + exception class so the
            # next time this fires we have more than str(e) to go on — str()
            # produces bare forms like "[Errno 2] No such file or directory"
            # with no indication of which layer raised it.
            logger.exception(
                'Failed to fetch tasks: %s: %s', type(e).__name__, e,
            )
        return []

    async def set_task_status(
        self,
        task_id: str,
        status: str,
        *,
        done_provenance: dict | None = None,
        reopen_reason: str | None = None,
    ) -> None:
        """Update task status via fused-memory.

        Terminal-state enforcement lives on the server (fused-memory
        TaskInterceptor) — this method just forwards the call. Pass
        ``reopen_reason`` to exit a terminal status (done/cancelled);
        orchestrator automation never needs it.
        """
        try:
            arguments: dict = {
                'id': task_id,
                'status': status,
                'project_root': self._project_root,
            }
            if done_provenance is not None:
                arguments['done_provenance'] = done_provenance
            if reopen_reason is not None:
                arguments['reopen_reason'] = reopen_reason
            await mcp_call(
                f'{self._memory_url}/mcp',
                'tools/call',
                {
                    'name': 'set_task_status',
                    'arguments': arguments,
                },
                timeout=15,
            )
        except Exception as e:
            logger.exception(
                'Failed to set task %s status to %s: %s: %s',
                task_id, status, type(e).__name__, e,
            )

    async def get_status(self, task_id: str) -> str | None:
        """Return the current status of ``task_id``, or ``None`` on failure.

        Replaces the old client-side status cache. Each call is a fresh MCP
        round-trip; fused-memory's warm get_task is ~30 ms, so this is cheap
        at the handful of decision points that actually need the truth
        (post-steward terminal check).
        """
        try:
            result = await mcp_call(
                f'{self._memory_url}/mcp',
                'tools/call',
                {
                    'name': 'get_task',
                    'arguments': {
                        'id': task_id,
                        'project_root': self._project_root,
                    },
                },
                timeout=15,
            )
        except Exception as e:
            logger.exception(
                'Failed to get task %s status: %s: %s',
                task_id, type(e).__name__, e,
            )
            return None
        content = result.get('result', {}).get('content', [])
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                try:
                    data = json.loads(block['text'])
                except (ValueError, TypeError):
                    return None
                # Taskmaster envelope: {data: {...}} — unwrap if present.
                inner = data.get('data') if isinstance(data.get('data'), dict) else data
                status = inner.get('status') if isinstance(inner, dict) else None
                if isinstance(status, str):
                    return status
        return None

    async def update_task(self, task_id: str, metadata: str | dict) -> bool:
        """Update task metadata via fused-memory. Returns True on success."""
        # fused-memory update_task expects metadata as a JSON string
        if isinstance(metadata, dict):
            metadata = json.dumps(metadata)
        try:
            result = await mcp_call(
                f'{self._memory_url}/mcp',
                'tools/call',
                {
                    'name': 'update_task',
                    'arguments': {
                        'id': task_id,
                        'metadata': metadata,
                        'project_root': self._project_root,
                    },
                },
                timeout=15,
            )
            # MCP tool errors return in the response body, not as exceptions
            content = result.get('result', result) if isinstance(result, dict) else result
            if isinstance(content, dict) and content.get('isError'):
                text = ''
                for block in content.get('content', []):
                    if isinstance(block, dict) and block.get('type') == 'text':
                        text = block.get('text', '')
                        break
                logger.error(f'Failed to update task {task_id}: {text}')
                return False
            return True
        except Exception as e:
            logger.exception(
                'Failed to update task %s: %s: %s',
                task_id, type(e).__name__, e,
            )
            return False

    def _deps_satisfied(self, task: dict, status_map: dict[str, str]) -> bool:
        """Return True if every dependency of *task* has status 'done'.

        Handles three dependency formats:
          - dict with 'id' key: ``{'id': 1}`` or ``{'id': '1'}``
          - integer: ``1``
          - string: ``'1'``

        Emits a DEBUG log when a dependency blocks dispatch, naming the dep
        ID and its current status to aid diagnosis of premature-dispatch issues.
        """
        deps = task.get('dependencies', [])
        task_id = str(task.get('id', '?'))
        for d in deps:
            dep_id = str(d.get('id', d) if isinstance(d, dict) else d)
            dep_status = status_map.get(dep_id, 'unknown')
            if dep_status != 'done':
                logger.debug(
                    'Task %s blocked: dep %s has status %s, need done',
                    task_id,
                    dep_id,
                    dep_status,
                )
                return False
        return True

    def _compute_lease(self, tier: str = DEFAULT_TIER) -> float:
        """Compute a reservation lease from the rolling duration window.

        - Empty history → midpoint of ``[lease_min_secs, lease_max_secs]``
        - Otherwise → ``median * lease_multiplier``, clamped to bounds.

        The multiplier is resolved per-*tier* (critical/high parks carry a
        longer lease than low/polish) via
        :meth:`FairnessConfig.lease_multiplier_for`.
        """
        f = self.config.fairness
        if not self._recent_durations:
            return (f.lease_min_secs + f.lease_max_secs) / 2
        median = statistics.median(self._recent_durations)
        lease = median * f.lease_multiplier_for(tier)
        return max(f.lease_min_secs, min(lease, f.lease_max_secs))

    def _bump_skip_and_maybe_park(
        self,
        task_id: str,
        modules: list[str],
        tier: str = DEFAULT_TIER,
    ) -> None:
        """Increment *task_id*'s skip counter; install a reservation if it
        has just crossed ``skip_threshold`` and does not already hold parks.

        *tier* is the task's effective priority — it selects a per-tier
        threshold and lease multiplier.  When the per-tier threshold is
        >= ``_INF_SKIP_THRESHOLD`` (e.g. ``9999`` for low/polish in the
        default config) parking is effectively disabled and the
        ``task_skipped`` event stream is rate-limited to geometric counts.
        """
        if not task_id:
            return
        count = self._skip_count.get(task_id, 0) + 1
        self._skip_count[task_id] = count
        threshold = self.config.fairness.skip_threshold_for(tier)
        # Rate-limit task_skipped for tiers that will never park: emit only
        # at {1, 10, 100, 1000, 10000, ...} so the event store is not flooded.
        should_emit = (
            threshold < _INF_SKIP_THRESHOLD
            or count in _GEOMETRIC_SKIP_EMIT_COUNTS
        )
        if self.event_store and should_emit:
            self.event_store.emit(
                EventType.task_skipped,
                task_id=task_id,
                data={
                    'skip_count': count,
                    'modules': modules,
                    'priority': tier,
                    'threshold': threshold,
                },
            )
        if count >= threshold and not self.lock_table.has_parks(task_id):
            lease = self._compute_lease(tier)
            deadline = time.monotonic() + lease
            installed = self.lock_table.install_parks(task_id, modules, deadline)
            logger.info(
                'Task %s reserved modules %s (skip_count=%d, lease=%.1fs, tier=%s)',
                task_id, installed, count, lease, tier,
            )
            if self.event_store:
                self.event_store.emit(
                    EventType.reservation_installed,
                    task_id=task_id,
                    data={
                        'modules': installed,
                        'skip_count': count,
                        'lease_secs': lease,
                        'priority': tier,
                    },
                )

    # --- Value/h scoring helpers (P1/P2/P3) -----------------------------

    @staticmethod
    def _build_reverse_index(tasks: list[dict]) -> dict[str, set[str]]:
        """Build ``{dep_id -> {tasks_that_depend_on_dep_id}}`` in one pass.

        Replaces the O(N^2) inline dependents scan the legacy sort key used.
        """
        rev: dict[str, set[str]] = {}
        for t in tasks:
            tid = str(t.get('id', ''))
            if not tid:
                continue
            for d in t.get('dependencies', []):
                dep_id = str(d.get('id', d) if isinstance(d, dict) else d)
                if dep_id:
                    rev.setdefault(dep_id, set()).add(tid)
        return rev

    @staticmethod
    def _compute_effective_priorities(
        tasks_by_id: dict[str, dict],
        reverse_index: dict[str, set[str]],
        status_map: dict[str, str],
    ) -> dict[str, str]:
        """Priority inheritance (P1).

        ``effective_priority(t) = min-rank(own, effective(d) for d in dependents(t))``
        walking only undone dependents (``status not in {done, cancelled}``).

        Tri-state DFS guards against dependency cycles: on a cycle the task
        contributes only its own priority and a WARN is logged.
        """
        memo: dict[str, str] = {}
        visiting: set[str] = set()
        walked: set[str] = set()

        def walk(tid: str) -> str:
            if tid in memo:
                return memo[tid]
            if tid in visiting:
                logger.warning(
                    'Priority inheritance: cycle detected at task %s; using own priority only',
                    tid,
                )
                task = tasks_by_id.get(tid, {})
                return coerce_tier(task.get('priority'))
            visiting.add(tid)
            task = tasks_by_id.get(tid, {})
            own = coerce_tier(task.get('priority'))
            best_rank = PRIORITY_RANK[own]
            for parent_id in reverse_index.get(tid, ()):
                parent_status = status_map.get(parent_id, '')
                if parent_status in ('done', 'cancelled'):
                    continue
                parent_eff = walk(parent_id)
                parent_rank = PRIORITY_RANK[parent_eff]
                if parent_rank < best_rank:
                    best_rank = parent_rank
            visiting.discard(tid)
            walked.add(tid)
            result = PRIORITY_TIERS[best_rank]
            memo[tid] = result
            return result

        for tid in tasks_by_id:
            if tid not in memo:
                walk(tid)
        return memo

    @staticmethod
    def _compute_transitive_counts(
        tasks_by_id: dict[str, dict],
        reverse_index: dict[str, set[str]],
        status_map: dict[str, str],
    ) -> dict[str, int]:
        """CPM proxy (P3): BFS over the reverse-dependency graph per task,
        counting undone descendants.  Memoized per cycle, O(N+E) overall.
        """
        memo: dict[str, int] = {}

        def bfs(root: str) -> int:
            seen: set[str] = set()
            queue: deque[str] = deque([root])
            count = 0
            while queue:
                current = queue.popleft()
                for child in reverse_index.get(current, ()):
                    if child in seen:
                        continue
                    seen.add(child)
                    if status_map.get(child, '') in ('done', 'cancelled'):
                        # Walk through to find further undone descendants (they
                        # may themselves unlock work).
                        queue.append(child)
                        continue
                    count += 1
                    queue.append(child)
            return count

        for tid in tasks_by_id:
            memo[tid] = bfs(tid)
        return memo

    def _compute_age(self, task_id: str, max_id: int) -> int:
        """Return this task's age, in "newer-task-count" units.

        Anchors are lazily initialized in :meth:`_update_age_anchors`; they
        reset to the *current* max_id on resurrection so a cancelled→pending
        task does not inherit accumulated age.
        """
        anchor = self._pending_anchor.get(task_id)
        if anchor is None:
            return 0
        return max(0, max_id - anchor)

    def _update_age_anchors(self, tasks: list[dict], max_id: int) -> None:
        """Maintain per-task age anchors across ticks.

        - First time we see a task as pending with no prior non-pending
          history, anchor to its own numeric id (so genuinely-old pending
          tasks carry accumulated age from the start).
        - First time we see a task as pending after having seen it
          non-pending, anchor to *current max_id* (resurrection resets age).
        - On any non-pending observation, drop the anchor and mark the task
          as ever-non-pending so the next pending appearance is a fresh start.
        """
        for t in tasks:
            tid = str(t.get('id', ''))
            if not tid:
                continue
            status = t.get('status', '')
            if status != 'pending':
                self._pending_anchor.pop(tid, None)
                if status:
                    self._was_non_pending.add(tid)
                continue
            if tid in self._pending_anchor:
                continue
            # First-seen pending for this tid.
            if tid in self._was_non_pending:
                # Resurrection — start fresh from now.
                self._pending_anchor[tid] = max_id
            elif tid.isdigit():
                self._pending_anchor[tid] = int(tid)
            else:
                self._pending_anchor[tid] = max_id

    def _compute_score(
        self,
        tier: str,
        age: int,
        transitive_count: int,
    ) -> float:
        """Compute the total dispatch score for a task.

        ``score = TIER_BASE[tier] + min(α*age + β*log1p(trans), TIER_WIDTH - 1)``

        The combined age+CPM bonus is capped below ``TIER_WIDTH`` so bonuses
        can never bump a task across a tier boundary — priority always wins.
        """
        tier = coerce_tier(tier)
        base = TIER_BASE[tier]
        age_bonus = self.config.age_alpha * float(age)
        cpm_bonus = self.config.cpm_beta * math.log1p(max(0, transitive_count))
        bonus = min(age_bonus + cpm_bonus, float(TIER_WIDTH - 1))
        return float(base) + bonus

    def _count_dispatched_at_or_below(self, tier: str) -> int:
        """Count currently-dispatched tasks whose effective priority is at
        *tier* or lower (i.e. same rank or larger rank value)."""
        tier = coerce_tier(tier)
        rank = PRIORITY_RANK[tier]
        return sum(
            1
            for p in self._dispatched_priority.values()
            if PRIORITY_RANK.get(coerce_tier(p), PRIORITY_RANK[DEFAULT_TIER]) >= rank
        )

    def _allowed_by_tier_cap(self, tier: str) -> bool:
        """Return False iff admitting a task at *tier* would exceed the
        configured per-tier slot cap."""
        limit = self.config.tier_slot_limit(tier)
        return self._count_dispatched_at_or_below(tier) < limit

    async def acquire_next(self) -> TaskAssignment | None:
        """Find next eligible task under the value/h scoring model.

        Dispatch order is determined by ``_compute_score()``: tier base is
        dominant, age + CPM bonuses order tasks within a tier, and
        per-tier slot caps reserve headroom for higher-value work.
        """
        # Fairness: evict expired reservations and reset their owners' skip
        # counts so they can re-accumulate instead of immediately re-parking.
        now = time.monotonic()
        evicted = self.lock_table.prune_expired_parks(now)
        for owner in evicted:
            self._skip_count.pop(owner, None)
            logger.info('Task %s reservation expired', owner)
            if self.event_store:
                self.event_store.emit(
                    EventType.reservation_expired,
                    task_id=owner,
                    data={},
                )

        tasks = await self.get_tasks()
        if not tasks:
            return None

        # Status + id indices, built once per tick.
        status_map: dict[str, str] = {}
        tasks_by_id: dict[str, dict] = {}
        max_id = 0
        for t in tasks:
            tid = str(t.get('id', ''))
            if not tid:
                continue
            status_map[tid] = t.get('status', 'unknown')
            tasks_by_id[tid] = t
            if tid.isdigit():
                max_id = max(max_id, int(tid))

        # Maintain age anchors (resurrected tasks reset their anchor).
        self._update_age_anchors(tasks, max_id)

        # Build reverse index + compute effective priorities + CPM counts
        # once per tick (O(N+E)).
        reverse_index = self._build_reverse_index(tasks)
        effective_priorities = self._compute_effective_priorities(
            tasks_by_id, reverse_index, status_map
        )
        transitive_counts = self._compute_transitive_counts(
            tasks_by_id, reverse_index, status_map
        )

        # Filter to pending tasks whose deps are all done and that aren't
        # dispatched or in their post-requeue cooldown window.
        candidates: list[dict] = []
        for t in tasks:
            if t.get('status') != 'pending':
                continue
            tid_str = str(t.get('id', ''))
            if tid_str in self._dispatched:
                continue
            cooldown_deadline = self._requeue_until.get(tid_str)
            if cooldown_deadline is not None:
                if time.monotonic() < cooldown_deadline:
                    continue
                del self._requeue_until[tid_str]
            if not self._deps_satisfied(t, status_map):
                continue
            candidates.append(t)

        if not candidates:
            return None

        # Score each candidate.  Higher score wins; ties broken by task_id
        # string order (stable, FIFO-ish for numeric ids).
        scored: list[tuple[float, str, dict, str]] = []
        for t in candidates:
            tid = str(t.get('id', ''))
            pri = effective_priorities.get(tid, coerce_tier(t.get('priority')))
            age = self._compute_age(tid, max_id)
            transitive = transitive_counts.get(tid, 0)
            score = self._compute_score(pri, age, transitive)
            scored.append((score, tid, t, pri))

        scored.sort(key=lambda entry: (-entry[0], entry[1]))

        # DEBUG: log the top 3 so α/β tuning is post-hoc diagnosable.
        if logger.isEnabledFor(logging.DEBUG):
            top3 = scored[:3]
            logger.debug(
                'acquire_next top candidates: %s',
                [
                    {
                        'id': e[1],
                        'score': round(e[0], 2),
                        'pri': e[3],
                    }
                    for e in top3
                ],
            )

        # Strict top is the highest-scoring eligible candidate.  We track it
        # for fairness bookkeeping (skip counter / park installation).
        top_score, top_id, top_task, top_pri = scored[0]
        top_modules = self._get_modules(top_task)
        top_had_parks = self.lock_table.has_parks(top_id)

        cap_blocked = 0
        for _score, task_id, task, pri in scored:
            modules = self._get_modules(task)
            # Tier-cap filter: skip candidates that would exceed their tier's
            # slot budget.  Parks override caps — once a fairness reservation
            # is installed, the owner dispatches regardless.
            has_park = self.lock_table.has_parks(task_id)
            if not has_park and not self._allowed_by_tier_cap(pri):
                cap_blocked += 1
                continue
            if self.lock_table.try_acquire(task_id, modules):
                self._dispatched.add(task_id)
                self._dispatched_priority[task_id] = pri
                self._task_start_times[task_id] = time.monotonic()
                if task_id == top_id:
                    self._skip_count.pop(task_id, None)
                    if top_had_parks:
                        self.lock_table.clear_parks_for(task_id)
                        if self.event_store:
                            self.event_store.emit(
                                EventType.reservation_used,
                                task_id=task_id,
                                data={'modules': modules, 'priority': pri},
                            )
                else:
                    # A lower-ranked task won — top was passed over this tick.
                    self._bump_skip_and_maybe_park(top_id, top_modules, top_pri)
                if self.event_store:
                    self.event_store.emit(
                        EventType.lock_acquired,
                        task_id=task_id,
                        data={'modules': modules, 'priority': pri},
                    )
                return TaskAssignment(task_id=task_id, task=task, modules=modules)

        # No candidate acquired.  If at least one was blocked by a tier cap,
        # emit a single idle-diagnostic event so "why are slots idle" is
        # visible in the event store.
        if cap_blocked and self.event_store:
            self.event_store.emit(
                EventType.scheduler_tier_cap_idle,
                data={
                    'candidates_skipped_by_cap': cap_blocked,
                    'top_id': top_id,
                    'top_priority': top_pri,
                },
            )

        # Loop exhausted with no acquire — top candidate was also skipped.
        self._bump_skip_and_maybe_park(top_id, top_modules, top_pri)
        return None

    async def handle_blast_radius_expansion(
        self,
        task_id: str,
        current: list[str],
        needed: list[str],
    ) -> bool:
        """Handle plan refining blast radius (widening, narrowing, or shift).

        1. Try acquire any additional locks (needed − current)
        2. On success, release any stale locks (current − needed) so other
           tasks can acquire modules the refined plan no longer touches
        3. On acquire failure: update task with new modules, reset to pending,
           release current locks
        """
        depth = self.config.lock_depth
        current_set = {normalize_lock(m, depth) for m in current}
        needed_set = {normalize_lock(m, depth) for m in needed}
        additional = sorted(needed_set - current_set)
        stale = sorted(current_set - needed_set)
        if not additional and not stale:
            return True

        if not additional or self.lock_table.try_acquire_additional(task_id, additional):
            if stale:
                released = self.lock_table.release_subset(task_id, stale)
                if released and self.event_store:
                    self.event_store.emit(
                        EventType.lock_released,
                        task_id=task_id,
                        data={'modules': released, 'reason': 'plan_refinement'},
                    )
            logger.info(f'Task {task_id} expanded to modules: {needed}')
            return True

        # Can't acquire — reset task
        logger.warning(
            f'Task {task_id} needs modules {needed} but locks unavailable. Requeuing.'
        )
        # Cache expanded modules in memory so _get_modules uses them on retry
        self._module_cache[task_id] = sorted(needed_set)
        updated = await self.update_task(task_id, {'modules': needed})
        if not updated:
            logger.warning(
                f'Task {task_id}: metadata update failed (non-critical — '
                f'using in-memory module cache for scheduling).'
            )
        await self.set_task_status(task_id, 'pending')
        self.lock_table.release(task_id)
        return False

    def release(self, task_id: str, *, requeued: bool = False) -> None:
        """Release all module locks for a task and clear dispatch guard."""
        self._dispatched.discard(task_id)
        self._dispatched_priority.pop(task_id, None)
        if requeued:
            self._requeue_until[task_id] = (
                time.monotonic() + self.config.requeue_cooldown_secs
            )
        # Fairness: record duration for the rolling median used by _compute_lease.
        start = self._task_start_times.pop(task_id, None)
        if start is not None:
            self._recent_durations.append(time.monotonic() - start)
        modules = list(self.lock_table._held.get(task_id, set()))
        self.lock_table.release(task_id)
        # Defensive: clear any reservations still owned by this task.
        self.lock_table.clear_parks_for(task_id)
        if self.event_store and modules:
            self.event_store.emit(
                EventType.lock_released,
                task_id=task_id,
                data={'modules': modules},
            )

    def _get_modules(self, task: dict) -> list[str]:
        """Extract module list from task metadata, normalized for locking.

        Priority: in-memory cache > metadata.files > metadata.modules > fallback.
        """
        task_id = str(task.get('id', ''))
        depth = self.config.lock_depth
        # Check in-memory cache first (survives metadata update failures)
        if task_id in self._module_cache:
            return self._module_cache[task_id]
        metadata = task.get('metadata', {})
        if isinstance(metadata, str):
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                metadata = json.loads(metadata)
        if isinstance(metadata, dict):
            # Prefer file-derived modules (most accurate)
            files = metadata.get('files', [])
            if isinstance(files, list) and files:
                derived = files_to_modules(files, depth)
                if derived:
                    return derived
            # Fall back to explicitly tagged modules
            modules = metadata.get('modules', [])
            if isinstance(modules, list) and modules:
                return [normalize_lock(m, depth) for m in modules]
        # Fallback: use a generic module name based on task id
        if task_id not in self._fallback_warned:
            logger.warning(
                'Task %s: no module metadata found — using fallback lock task-%s',
                task_id,
                task_id or 'unknown',
            )
            self._fallback_warned.add(task_id)
        return [f'task-{task_id or "unknown"}']
