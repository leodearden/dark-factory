"""Synchronous functions for discovering orchestrator processes and reading task artifacts.

Scans running processes, parses the Taskmaster task tree, reads per-worktree
.task/ artifacts, and combines them into a unified orchestrator status view.
All functions are synchronous (subprocess.run, file I/O) — unlike the async
memory.py and reconciliation.py modules.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


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

    result: list[dict] = []
    for task in raw_tasks:
        result.append({
            'id': task.get('id'),
            'title': task.get('title'),
            'status': task.get('status'),
            'priority': task.get('priority'),
            'dependencies': task.get('dependencies', []),
            'metadata': task.get('metadata', {}),
        })

    return result
