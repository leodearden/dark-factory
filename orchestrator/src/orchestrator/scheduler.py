"""Task selection and module lock management."""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import dataclass

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.mcp_lifecycle import mcp_call
from orchestrator.task_status import TERMINAL_STATUSES, is_valid_transition

logger = logging.getLogger(__name__)


def normalize_lock(module: str, depth: int = 2) -> str:
    """Normalize a module path to a fixed depth for lock granularity.

    e.g. normalize_lock('crates/reify-types/src/persistent.rs') -> 'crates/reify-types'
    """
    if not module:
        return module
    parts = module.strip('/').split('/')
    return '/'.join(parts[:depth])


def files_to_modules(files: list[str], depth: int) -> list[str]:
    """Derive unique module locks from a list of file paths.

    Each file path is normalized to ``depth`` components, then deduplicated.
    """
    modules: set[str] = set()
    for f in files:
        normalized = normalize_lock(f, depth)
        if normalized:
            modules.add(normalized)
    return sorted(modules)


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
        ``A/B/C`` (and vice-versa) but NOT with ``A/D``.

        Returns True if all acquired, False if any unavailable.
        """
        depth = self._config.lock_depth
        normalized = list({normalize_lock(m, depth) for m in modules})

        # Check every requested module against all other tasks' held locks
        for module in normalized:
            if self._count_conflicts(module, exclude_task=task_id) >= self._limit_for(module):
                return False

        self._held[task_id] = set(normalized)
        logger.info(f'Task {task_id} acquired locks: {normalized}')
        return True

    def release(self, task_id: str) -> None:
        """Release all module locks held by a task."""
        modules = self._held.pop(task_id, set())
        if modules:
            logger.info(f'Task {task_id} released locks: {list(modules)}')

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

        for module in new_modules:
            if self._count_conflicts(module, exclude_task=task_id) >= self._limit_for(module):
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
        self._status_cache: dict[str, str] = {}
        self._fallback_warned: set[str] = set()  # task IDs already warned about fallback
        self._requeue_until: dict[str, float] = {}  # task_id -> monotonic deadline

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
                    # Seed status cache from task list.
                    # Always trust the store — it is the source of truth.
                    # External processes may reinstate cancelled/done tasks
                    # to pending; the cache must reflect that to avoid
                    # silent rejection loops in set_task_status().
                    for t in tasks:
                        tid = str(t.get('id', ''))
                        s = t.get('status', '')
                        if tid and s:
                            old = self._status_cache.get(tid)
                            if old in TERMINAL_STATUSES and old != s:
                                logger.info(
                                    'Task %s: store status %s overrides cached %s '
                                    '(external reinstatement)',
                                    tid, s, old,
                                )
                            self._status_cache[tid] = s
                    return tasks
        except Exception as e:
            logger.error(f'Failed to fetch tasks: {e}')
        return []

    async def set_task_status(self, task_id: str, status: str) -> None:
        """Update task status via fused-memory."""
        cached = self._status_cache.get(task_id)
        if not is_valid_transition(cached, status):
            logger.warning(
                'Task %s: rejecting %s->%s (terminal state guard)', task_id, cached, status
            )
            return
        try:
            await mcp_call(
                f'{self._memory_url}/mcp',
                'tools/call',
                {
                    'name': 'set_task_status',
                    'arguments': {
                        'id': task_id,
                        'status': status,
                        'project_root': self._project_root,
                    },
                },
                timeout=15,
            )
            self._status_cache[task_id] = status
        except Exception as e:
            logger.error(f'Failed to set task {task_id} status to {status}: {e}')

    def get_cached_status(self, task_id: str) -> str | None:
        """Return the last known status for a task, or None if not yet seen."""
        return self._status_cache.get(task_id)

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
            logger.error(f'Failed to update task {task_id}: {e}')
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

    async def acquire_next(self) -> TaskAssignment | None:
        """Find next eligible task: pending, deps done, module locks available.

        Priority: explicit priority > dependency depth > task ID.
        """
        tasks = await self.get_tasks()
        if not tasks:
            return None

        # Build status map
        status_map = {}
        for t in tasks:
            tid = str(t.get('id', ''))
            status_map[tid] = t.get('status', 'unknown')

        # Filter to pending tasks whose deps are all done
        candidates = []
        for t in tasks:
            if t.get('status') != 'pending':
                continue
            if str(t.get('id', '')) in self._dispatched:
                continue
            tid_str = str(t.get('id', ''))
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

        # Sort by priority (high first), then dependency depth (deeper first), then ID
        def sort_key(t):
            priority_map = {'high': 0, 'medium': 1, 'low': 2}
            p = priority_map.get(t.get('priority', 'medium'), 1)
            # Count how many other tasks depend on this one
            tid = str(t.get('id', ''))
            dependents = sum(
                1 for other in tasks
                if any(
                    str(d.get('id', d) if isinstance(d, dict) else d) == tid
                    for d in other.get('dependencies', [])
                )
            )
            return (p, -dependents, str(t.get('id', '')))

        candidates.sort(key=sort_key)

        # Try to acquire module locks
        for task in candidates:
            modules = self._get_modules(task)
            task_id = str(task.get('id', ''))
            if self.lock_table.try_acquire(task_id, modules):
                self._dispatched.add(task_id)
                if self.event_store:
                    self.event_store.emit(
                        EventType.lock_acquired,
                        task_id=task_id,
                        data={'modules': modules},
                    )
                return TaskAssignment(task_id=task_id, task=task, modules=modules)

        return None

    async def handle_blast_radius_expansion(
        self,
        task_id: str,
        current: list[str],
        needed: list[str],
    ) -> bool:
        """Handle plan discovering wider blast radius.

        1. Try acquire additional locks
        2. If success: return True (proceed)
        3. If fail: update task with new modules, reset to pending, release current locks
        """
        depth = self.config.lock_depth
        current_normalized = [normalize_lock(m, depth) for m in current]
        needed_normalized = [normalize_lock(m, depth) for m in needed]
        additional = [m for m in needed_normalized if m not in current_normalized]
        if not additional:
            return True

        if self.lock_table.try_acquire_additional(task_id, additional):
            logger.info(f'Task {task_id} expanded to modules: {needed}')
            return True

        # Can't acquire — reset task
        logger.warning(
            f'Task {task_id} needs modules {needed} but locks unavailable. Requeuing.'
        )
        # Cache expanded modules in memory so _get_modules uses them on retry
        self._module_cache[task_id] = needed_normalized
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
        if requeued:
            self._requeue_until[task_id] = (
                time.monotonic() + self.config.requeue_cooldown_secs
            )
        modules = list(self.lock_table._held.get(task_id, set()))
        self.lock_table.release(task_id)
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
