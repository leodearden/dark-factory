"""Synchronous functions for discovering orchestrator processes and reading task artifacts.

Scans running processes, parses the Taskmaster task tree, reads per-worktree
.task/ artifacts, and combines them into a unified orchestrator status view.
All functions are synchronous (subprocess.run, file I/O) — unlike the async
memory.py and reconciliation.py modules.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import subprocess
from pathlib import Path

from dashboard.config import DashboardConfig

logger = logging.getLogger(__name__)


def _extract_task_id(dirname: str) -> str | None:
    """Normalise a worktree directory name to a numeric task ID string.

    Handles two naming conventions:
    - ``'task-{id}'`` (e.g. ``'task-7'``) — strips the prefix and returns the
      digit portion.
    - ``'{id}'`` (e.g. ``'7'``) — returns it directly.

    Returns ``None`` for any name that doesn't yield a non-empty digit string
    (e.g. ``'task-abc'``, ``'task-'``, ``'random-dir'``, ``''``).
    """
    if dirname.startswith('task-'):
        suffix = dirname[len('task-'):]
        return suffix if suffix.isdigit() and suffix else None
    return dirname if dirname.isdigit() and dirname else None


def find_running_orchestrators() -> list[dict]:
    """Scan ``ps aux`` for running orchestrator processes.

    Returns a list of dicts with keys: pid (int), prd (str), running (bool),
    started (str). Returns [] on subprocess failure or if no orchestrators found.
    """
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
        )
    except Exception:
        logger.warning('Failed to run ps aux', exc_info=True)
        return []

    orchestrators: list[dict] = []
    for line in result.stdout.splitlines():
        if 'orchestrator' not in line:
            continue
        if '--prd' not in line:
            continue
        if 'grep' in line:
            continue

        fields = line.split()
        if len(fields) < 11:
            continue

        try:
            pid = int(fields[1])
            started = fields[8]

            prd_match = re.search(r'--prd\s+(\S+)', line)
            if not prd_match:
                continue

            orchestrators.append({
                'pid': pid,
                'prd': prd_match.group(1),
                'running': True,
                'started': started,
            })
        except (ValueError, IndexError):
            logger.warning('Skipping malformed ps line: %s', line.strip())
            continue

    return orchestrators


def load_task_tree(tasks_json_path: Path) -> list[dict]:
    """Parse a Taskmaster tasks.json file into a list of task dicts.

    Supports both ``{'master': {'tasks': [...]}}`` and ``{'tasks': [...]}``
    formats. Each returned dict has keys: id, title, status, priority,
    dependencies, metadata.

    Returns [] if the file is missing, contains invalid JSON, or lacks
    the expected structure.
    """
    try:
        raw = tasks_json_path.read_text()
    except FileNotFoundError:
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning('Malformed JSON in %s', tasks_json_path)
        return []

    try:
        raw_tasks = data['master']['tasks']
    except (KeyError, TypeError):
        try:
            raw_tasks = data['tasks']
        except (KeyError, TypeError):
            logger.warning('No tasks found in %s', tasks_json_path)
            return []

    if not isinstance(raw_tasks, list):
        logger.warning('tasks is not a list in %s', tasks_json_path)
        return []

    result: list[dict] = []
    for task in raw_tasks:
        try:
            result.append({
                'id': task.get('id'),
                'title': task.get('title'),
                'status': task.get('status'),
                'priority': task.get('priority'),
                'dependencies': task.get('dependencies', []),
                'metadata': task.get('metadata', {}),
            })
        except AttributeError:
            logger.warning('Skipping non-dict task entry in %s', tasks_json_path)
            continue

    return result


def read_task_artifacts(worktree_path: Path) -> dict:
    """Read .task/ artifacts from a worktree directory.

    Returns a dict with keys:
    - metadata: parsed metadata.json dict, or None
    - phase: 'PLAN', 'EXECUTE', or 'DONE'
    - plan_progress: {'done': int, 'total': int}
    - iteration_count: number of lines in iterations.jsonl
    - review_summary: 'N/M passed' or '—' if no reviews
    """
    task_dir = worktree_path / '.task'

    # Metadata
    metadata = None
    with contextlib.suppress(FileNotFoundError, json.JSONDecodeError):
        metadata = json.loads((task_dir / 'metadata.json').read_text())

    # Plan progress, phase, and modules
    done_count = 0
    total_count = 0
    modules: list[str] = []
    try:
        plan_data = json.loads((task_dir / 'plan.json').read_text())
        steps = plan_data.get('steps', [])
        total_count = len(steps)
        done_count = sum(1 for s in steps if s.get('status') == 'done')
        modules = plan_data.get('modules', [])
    except (FileNotFoundError, json.JSONDecodeError, AttributeError, TypeError):
        pass

    if total_count == 0:
        phase = 'PLAN'
    elif done_count == total_count:
        phase = 'DONE'
    else:
        phase = 'EXECUTE'

    # Iteration count
    iteration_count = 0
    try:
        with open(task_dir / 'iterations.jsonl') as f:
            iteration_count = sum(1 for _ in f)
    except FileNotFoundError:
        pass

    # Review summary
    review_summary = '\u2014'
    reviews_dir = task_dir / 'reviews'
    if reviews_dir.is_dir():
        review_files = list(reviews_dir.glob('*.json'))
        if review_files:
            total_reviews = len(review_files)
            passed = 0
            for review_file in review_files:
                try:
                    review = json.loads(review_file.read_text())
                    if review.get('verdict') == 'PASS':
                        passed += 1
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
            review_summary = f'{passed}/{total_reviews} passed'

    return {
        'metadata': metadata,
        'phase': phase,
        'plan_progress': {'done': done_count, 'total': total_count},
        'iteration_count': iteration_count,
        'review_summary': review_summary,
        'modules': modules,
    }


def _extract_task_id(dirname: str) -> str | None:
    """Normalize a worktree directory name to a numeric task ID string.

    Handles two naming conventions:
    - Plain numeric: '7' → '7'
    - Prefixed: 'task-7' → '7'

    Returns None for non-task directories ('tmp-backup', 'task-abc', 'task-', '').
    """
    if dirname.startswith('task-'):
        suffix = dirname[len('task-'):]
        return suffix if suffix.isdigit() else None
    return dirname if dirname.isdigit() else None


def discover_orchestrators(config: DashboardConfig) -> list[dict]:
    """Discover running orchestrators and enrich with task tree and worktree data.

    For each running orchestrator process, attaches:
    - tasks: parsed task tree from tasks.json
    - worktrees: dict mapping worktree name → artifact data
    - summary: status counts {total, done, in_progress, blocked, pending}

    Returns [] if no orchestrator processes are running.
    """
    processes = find_running_orchestrators()
    if not processes:
        return []

    tasks = load_task_tree(config.tasks_json)

    # Scan worktrees
    worktrees: dict[str, dict] = {}
    worktrees_dir = config.worktrees_dir
    if worktrees_dir.is_dir():
        for subdir in sorted(worktrees_dir.iterdir()):
            if subdir.is_dir():
                task_id = _extract_task_id(subdir.name)
                if task_id is not None:
                    worktrees[task_id] = read_task_artifacts(subdir)

    # Compute summary stats from task tree
    summary = {
        'total': len(tasks),
        'done': sum(1 for t in tasks if t.get('status') == 'done'),
        'in_progress': sum(1 for t in tasks if t.get('status') == 'in-progress'),
        'blocked': sum(1 for t in tasks if t.get('status') == 'blocked'),
        'pending': sum(1 for t in tasks if t.get('status') == 'pending'),
    }

    result: list[dict] = []
    for proc in processes:
        result.append({
            **proc,
            'tasks': tasks,
            'worktrees': worktrees,
            'summary': summary,
        })

    return result
