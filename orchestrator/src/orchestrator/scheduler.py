"""Task selection and module lock management."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from orchestrator.config import OrchestratorConfig
from orchestrator.mcp_lifecycle import mcp_call

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

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.lock_table = ModuleLockTable(config)
        self._dispatched: set[str] = set()
        self._memory_url = config.fused_memory.url
        self._project_root = str(config.project_root)

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
                        return data['data'].get('tasks', [])
                    return data.get('tasks', [])
        except Exception as e:
            logger.error(f'Failed to fetch tasks: {e}')
        return []

    async def set_task_status(self, task_id: str, status: str) -> None:
        """Update task status via fused-memory."""
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
        except Exception as e:
            logger.error(f'Failed to set task {task_id} status to {status}: {e}')

    async def update_task(self, task_id: str, metadata: str) -> bool:
        """Update task metadata via fused-memory. Returns True on success."""
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
            deps = t.get('dependencies', [])
            deps_done = all(
                status_map.get(str(d.get('id', d) if isinstance(d, dict) else d)) == 'done'
                for d in deps
            )
            if not deps_done:
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
                return TaskAssignment(task_id=task_id, task=task, modules=modules)

        return None

    async def handle_blast_radius_expansion(
        self,
        task_id: str,
        current_modules: list[str],
        needed_modules: list[str],
    ) -> bool:
        """Handle plan discovering wider blast radius.

        1. Try acquire additional locks
        2. If success: return True (proceed)
        3. If fail: update task with new modules, reset to pending, release current locks
        """
        depth = self.config.lock_depth
        current_normalized = [normalize_lock(m, depth) for m in current_modules]
        needed_normalized = [normalize_lock(m, depth) for m in needed_modules]
        additional = [m for m in needed_normalized if m not in current_normalized]
        if not additional:
            return True

        if self.lock_table.try_acquire_additional(task_id, additional):
            logger.info(f'Task {task_id} expanded to modules: {needed_modules}')
            return True

        # Can't acquire — reset task
        logger.warning(
            f'Task {task_id} needs modules {needed_modules} but locks unavailable. Requeuing.'
        )
        import json
        updated = await self.update_task(task_id, json.dumps({'modules': needed_modules}))
        if not updated:
            logger.error(
                f'Task {task_id}: metadata update failed — task will retry with '
                f'original modules and may cycle. Check TASK_MASTER_ALLOW_METADATA_UPDATES.'
            )
        await self.set_task_status(task_id, 'pending')
        self.lock_table.release(task_id)
        return False

    def release(self, task_id: str) -> None:
        """Release all module locks for a task."""
        self.lock_table.release(task_id)

    def _get_modules(self, task: dict) -> list[str]:
        """Extract module list from task metadata, normalized for locking.

        Priority: metadata.files (deterministic) > metadata.modules (heuristic).
        """
        depth = self.config.lock_depth
        metadata = task.get('metadata', {})
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
        return [f'task-{task.get("id", "unknown")}']
