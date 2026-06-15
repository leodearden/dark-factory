"""Functions for discovering orchestrator processes and reading task artifacts.

Scans running processes (via ``ps aux``), reads per-worktree ``.task/``
artifacts from disk, and fetches task trees from the fused-memory MCP
server (which is the source of truth post-2026-05-02 SQLite cutover).
``discover_orchestrators`` is async because the MCP call is async; the
process-scanning and worktree-artifact helpers remain sync and run via
``asyncio.to_thread`` from the async caller.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import subprocess
from pathlib import Path

import httpx

from dashboard.config import DashboardConfig
from dashboard.data.tasks import fetch_tasks

logger = logging.getLogger(__name__)


def _resolve_project_root(prd: str, default_root: Path) -> Path:
    """Find the project root for an orchestrator by walking up from its PRD path.

    Looks for a ``.taskmaster/`` directory starting from the PRD's parent.
    Falls back to *default_root* (the dashboard's own project root) if no
    ``.taskmaster/`` is found or the PRD path is relative.

    The returned Path is always canonical (symlinks resolved).  This guarantee
    is now mirrored by ``_resolve_root`` inside :func:`discover_orchestrators`:
    every branch of that helper also returns a canonical Path, so consumers of
    either function can rely on canonical-path equality without defensive
    ``.resolve()`` calls.
    """
    p = Path(prd)
    if not p.is_absolute():
        p = default_root / p
    p = p.resolve()

    for ancestor in p.parents:
        if (ancestor / '.taskmaster').is_dir():
            return ancestor
    return default_root.resolve()


def _read_project_root_from_config(config_path: str) -> Path | None:
    """Extract ``project_root`` from an orchestrator config YAML file.

    Handles ``${VAR:default}`` env-var expansion for the project_root value.
    Returns ``None`` if the file can't be read or doesn't contain project_root.
    """
    import os

    import yaml

    try:
        raw = yaml.safe_load(Path(config_path).read_text())
    except (FileNotFoundError, yaml.YAMLError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    value = raw.get('project_root')
    if not isinstance(value, str):
        return None
    # Expand ${VAR:default} patterns (matching orchestrator config.py behavior)
    expanded = re.sub(
        r'\$\{([^:}]+)(?::([^}]*))?\}',
        lambda m: os.environ.get(m.group(1), m.group(2) or ''),
        value,
    )
    p = Path(expanded)
    return p.resolve() if p.is_absolute() else None


def _scan_worktrees(worktrees_dir: Path) -> dict[int, dict]:
    """Scan a .worktrees/ directory and return {task_id: artifact_data}."""
    worktrees: dict[int, dict] = {}
    if worktrees_dir.is_dir():
        for subdir in sorted(worktrees_dir.iterdir()):
            if subdir.is_dir():
                task_id = _extract_task_id(subdir.name)
                if task_id is not None:
                    worktrees[task_id] = read_task_artifacts(subdir)
    return worktrees


def _extract_task_id(dirname: str) -> int | None:
    """Normalise a worktree directory name to a numeric task ID.

    Handles two naming conventions:
    - ``'task-{id}'`` (e.g. ``'task-7'``) — strips the prefix and returns the
      digit portion as an int.
    - ``'{id}'`` (e.g. ``'7'``) — returns it as an int.

    Returns ``None`` for any name that doesn't yield a non-empty digit string
    (e.g. ``'task-abc'``, ``'task-'``, ``'random-dir'``, ``''``).
    """
    digits: str | None = None
    if dirname.startswith('task-'):
        suffix = dirname[len('task-'):]
        digits = suffix if suffix.isdigit() and suffix else None
    else:
        digits = dirname if dirname.isdigit() and dirname else None
    return int(digits) if digits is not None else None


def find_running_orchestrators() -> list[dict]:
    """Scan ``ps aux`` for running orchestrator processes.

    Detects three launch patterns:

    1. ``orchestrator run --prd <path>`` — extracts prd path
    2. ``orchestrator run --config <path>`` — extracts config path
    3. ``orchestrator run`` (no flags) — bare run of existing tasks

    Returns a list of dicts with keys: pid (int), prd (str | None),
    config_path (str | None), running (bool), started (str).
    Returns [] on subprocess failure or if no orchestrators found.
    """
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.warning('Failed to run ps aux', exc_info=True)
        return []

    orchestrators: list[dict] = []
    for line in result.stdout.splitlines():
        if 'orchestrator' not in line:
            continue
        if 'orchestrator run' not in line:
            continue
        if 'grep' in line:
            continue

        fields = line.split()
        if len(fields) < 11:
            continue

        try:
            pid = int(fields[1])
            started = fields[8]
        except (ValueError, IndexError):
            logger.warning('Skipping malformed ps line: %s', line.strip())
            continue

        prd_match = re.search(r'--prd\s+(\S+)', line)
        config_match = re.search(r'--config\s+(\S+)', line)

        orchestrators.append({
            'pid': pid,
            'prd': prd_match.group(1) if prd_match else None,
            'config_path': config_match.group(1) if config_match else None,
            'running': True,
            'started': started,
        })

    return orchestrators


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

    # Plan progress, phase, and files
    done_count = 0
    total_count = 0
    files: list[str] = []
    try:
        plan_data = json.loads((task_dir / 'plan.json').read_text())
        steps = plan_data.get('steps', [])
        total_count = len(steps)
        done_count = sum(1 for s in steps if s.get('status') == 'done')
        files = plan_data.get('files', [])
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
    review_summary = '—'
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
        'files': files,
    }


async def discover_orchestrators(
    client: httpx.AsyncClient,
    config: DashboardConfig,
) -> list[dict]:
    """Discover running orchestrators and enrich with task tree and worktree data.

    For each running orchestrator process, attaches:
    - tasks: parsed task list fetched from fused-memory MCP
    - worktrees: dict mapping worktree name → artifact data
    - summary: status counts {total, done, in_progress, blocked, pending}

    Returns [] if no orchestrator processes are running.
    Per-project task fetches that hit MCP errors degrade to an empty list.
    """
    processes = await asyncio.to_thread(find_running_orchestrators)
    if not processes:
        return []

    def _resolve_root(proc: dict) -> Path:
        """Resolve project root from process info: prd > config > default.

        All three branches return a canonical (symlink-resolved) Path so that
        the ``groups`` dict always uses canonical keys and the ``project_root``
        emitted in each result entry is canonical without further ``.resolve()``.
        """
        if proc.get('prd'):
            return _resolve_project_root(proc['prd'], config.project_root)
        if proc.get('config_path'):
            root = _read_project_root_from_config(proc['config_path'])
            if root is not None:
                return root
        return config.project_root.resolve()

    # Group processes by resolved project root — multiple PIDs targeting the
    # same project are merged into a single entry with a 'pids' list.
    groups: dict[Path, list[dict]] = {}
    for proc in processes:
        root = _resolve_root(proc)
        groups.setdefault(root, []).append(proc)

    # Cache per-project data so we don't re-fetch the same task list
    # when multiple processes share a project root.
    project_cache: dict[Path, tuple[list[dict], dict[int, dict]]] = {}

    result: list[dict] = []
    for project_root, group in groups.items():
        if project_root not in project_cache:
            fetched = await fetch_tasks(client, config, project_root)
            tasks = fetched if isinstance(fetched, list) else []
            worktrees = await asyncio.to_thread(_scan_worktrees, project_root / '.worktrees')
            project_cache[project_root] = (tasks, worktrees)

        tasks, worktrees = project_cache[project_root]
        summary = {
            'total': len(tasks),
            'done': sum(1 for t in tasks if t.get('status') == 'done'),
            'in_progress': sum(1 for t in tasks if t.get('status') == 'in-progress'),
            'blocked': sum(1 for t in tasks if t.get('status') == 'blocked'),
            'pending': sum(1 for t in tasks if t.get('status') == 'pending'),
        }
        last_update = max(
            (t['updated_at'] for t in tasks if t.get('updated_at')),
            default=None,
        )

        # Display label: prefer PRD path, fall back to project root path
        prd = next((p['prd'] for p in group if p.get('prd')), None)
        label = prd if prd else str(project_root)

        result.append({
            'pids': [p['pid'] for p in group],
            'prd': prd,
            'label': label,
            'project_root': str(project_root),
            'running': any(p['running'] for p in group),
            'started': group[0]['started'],
            'last_update': last_update,
            'tasks': tasks,
            'worktrees': worktrees,
            'summary': summary,
        })

    return result
