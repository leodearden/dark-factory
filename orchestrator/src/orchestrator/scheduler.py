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


@dataclass
class TaskAssignment:
    """A task that has been assigned to a workflow slot, with module locks held."""

    task_id: str
    task: dict
    modules: list[str]


class ModuleLockTable:
    """Per-module counters. Serialize entire workflow per module."""

    def __init__(self, config: OrchestratorConfig):
        self._limits: dict[str, int] = {}
        self._counts: dict[str, int] = {}  # module -> current holders
        self._held: dict[str, set[str]] = {}  # task_id -> set of held modules
        self._config = config

    def _limit_for(self, module: str) -> int:
        module = normalize_lock(module, self._config.lock_depth)
        if module not in self._limits:
            self._limits[module] = self._config.module_overrides.get(
                module, self._config.max_per_module
            )
        return self._limits[module]

    def _count(self, module: str) -> int:
        return self._counts.get(module, 0)

    def try_acquire(self, task_id: str, modules: list[str]) -> bool:
        """Non-blocking attempt to acquire all module locks.

        Returns True if all acquired, False if any unavailable.
        On failure, releases any partially acquired locks.
        """
        depth = self._config.lock_depth
        normalized = [normalize_lock(m, depth) for m in modules]
        acquired = []
        for module in normalized:
            if self._count(module) < self._limit_for(module):
                self._counts[module] = self._count(module) + 1
                acquired.append(module)
            else:
                # Release what we acquired
                for m in acquired:
                    self._counts[m] -= 1
                return False

        self._held[task_id] = set(normalized)
        logger.info(f'Task {task_id} acquired locks: {normalized}')
        return True

    def release(self, task_id: str) -> None:
        """Release all module locks held by a task."""
        modules = self._held.pop(task_id, set())
        for module in modules:
            self._counts[module] = max(0, self._count(module) - 1)
        if modules:
            logger.info(f'Task {task_id} released locks: {list(modules)}')

    def try_acquire_additional(self, task_id: str, additional: list[str]) -> bool:
        """Non-blocking attempt to expand lock set for a task."""
        depth = self._config.lock_depth
        current = self._held.get(task_id, set())
        new_modules = [normalize_lock(m, depth) for m in additional if normalize_lock(m, depth) not in current]
        if not new_modules:
            return True

        acquired = []
        for module in new_modules:
            if self._count(module) < self._limit_for(module):
                self._counts[module] = self._count(module) + 1
                acquired.append(module)
            else:
                for m in acquired:
                    self._counts[m] -= 1
                return False

        self._held[task_id].update(new_modules)
        logger.info(f'Task {task_id} expanded locks: {new_modules}')
        return True


class Scheduler:
    """Selects next eligible task and manages module locks."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.lock_table = ModuleLockTable(config)
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

    async def update_task(self, task_id: str, metadata: str) -> None:
        """Update task metadata via fused-memory."""
        try:
            await mcp_call(
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
        except Exception as e:
            logger.error(f'Failed to update task {task_id}: {e}')

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
        await self.update_task(task_id, json.dumps({'modules': needed_modules}))
        await self.set_task_status(task_id, 'pending')
        self.lock_table.release(task_id)
        return False

    def release(self, task_id: str) -> None:
        """Release all module locks for a task."""
        self.lock_table.release(task_id)

    def _get_modules(self, task: dict) -> list[str]:
        """Extract module list from task metadata, normalized for locking."""
        metadata = task.get('metadata', {})
        if isinstance(metadata, dict):
            modules = metadata.get('modules', [])
            if isinstance(modules, list) and modules:
                return [normalize_lock(m, self.config.lock_depth) for m in modules]
        # Fallback: use a generic module name based on task id
        return [f'task-{task.get("id", "unknown")}']
