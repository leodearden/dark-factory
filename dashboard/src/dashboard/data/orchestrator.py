"""Functions for discovering orchestrator processes and task status.

Scans running processes (via ``ps aux``) and fetches task trees from the
fused-memory MCP server (which is the source of truth post-2026-05-02
SQLite cutover). ``discover_orchestrators`` is async because the MCP call
is async; the process-scanning helper remains sync and runs via
``asyncio.to_thread`` from the async caller.

FORMAT COUPLING
================
This module re-derives a format it does not own. It does NOT import the
``orchestrator`` package on purpose — ``dashboard/pyproject.toml`` depends
only on ``escalation`` + ``dark-factory-shared``, and import unification was
evaluated and rejected (see ``plans/dashboard-alignment-prd.md`` task ζ). If
the upstream format changes, this module must be updated by hand.

1. ps-scan launch patterns (:func:`find_running_orchestrators`) — the
   ``'orchestrator run'`` substring match and the ``--prd``/``--config``
   regexes re-derive the CLI surface of the ``run`` command defined in
   ``orchestrator/src/orchestrator/cli.py`` (grep that file for ``def run``).
   Anyone renaming the ``run`` command or its ``--prd``/``--config`` flags
   must update ``find_running_orchestrators`` to match.

RETIRED: this module used to re-derive a second format — the ``.task/``
artifact layout (``metadata.json``, ``plan.json``, ``iterations.jsonl``,
``reviews/*.json``) — via a hand-rolled reader (``read_task_artifacts`` /
``_scan_worktrees`` / ``_extract_task_id``). That reader has been deleted;
per-task runtime state (loops/attempts/lane/phase) is now served by the
escalation ``get_task_runtime_state`` MCP tool, with the orchestrator as
the single owner of that format (see
``plans/dashboard-task-runtime-endpoint-prd.md``). There is no longer a
"future single owner" hand-off pending for this module.
"""

from __future__ import annotations

import asyncio
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


def find_running_orchestrators() -> list[dict]:
    """Scan ``ps aux`` for running orchestrator processes.

    Detects three launch patterns:

    1. ``orchestrator run --prd <path>`` — extracts prd path
    2. ``orchestrator run --config <path>`` — extracts config path
    3. ``orchestrator run`` (no flags) — bare run of existing tasks

    Returns a list of dicts with keys: pid (int), prd (str | None),
    config_path (str | None), running (bool), started (str).
    Returns [] on subprocess failure or if no orchestrators found.

    FORMAT COUPLING: the patterns below re-derive the ``run`` command's CLI
    surface from orchestrator/src/orchestrator/cli.py (grep that file for
    ``def run``). See the module docstring's FORMAT COUPLING section.
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


async def discover_orchestrators(
    client: httpx.AsyncClient,
    config: DashboardConfig,
) -> list[dict]:
    """Discover running orchestrators and enrich with task tree data.

    For each running orchestrator process, attaches:
    - tasks: parsed task list fetched from fused-memory MCP
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
    # Cache tuple: (tasks, offline, error)
    project_cache: dict[Path, tuple[list[dict], bool, str | None]] = {}

    result: list[dict] = []
    for project_root, group in groups.items():
        if project_root not in project_cache:
            fetched = await fetch_tasks(client, config, project_root)
            if isinstance(fetched, list):
                tasks = fetched
                offline = False
                fetch_error: str | None = None
            else:
                # Offline marker: {'offline': True, 'error': ...}
                tasks = []
                offline = bool(fetched.get('offline')) if isinstance(fetched, dict) else False
                fetch_error = str(fetched.get('error', '')) if isinstance(fetched, dict) else None
            project_cache[project_root] = (tasks, offline, fetch_error)

        tasks, offline, fetch_error = project_cache[project_root]
        summary = {
            'total': len(tasks),
            'done': sum(1 for t in tasks if t.get('status') == 'done'),
            'in_progress': sum(1 for t in tasks if t.get('status') == 'in-progress'),
            'blocked': sum(1 for t in tasks if t.get('status') == 'blocked'),
            'pending': sum(1 for t in tasks if t.get('status') == 'pending'),
        }
        # Lexicographic max over ISO-8601 strings is correct here because
        # tasks.py copies updatedAt verbatim from a single upstream source, so
        # all values share the same format and UTC offset.  If the source ever
        # emits mixed offsets, switch to key=datetime.fromisoformat.
        # Scope: top-level tasks only, matching the summary counts above.
        # If subtask recency should count, flatten the task tree first.
        last_update = max(
            (t['updated_at'] for t in tasks if t.get('updated_at')),
            default=None,
        )

        # Display label: prefer PRD path, fall back to project root path
        prd = next((p['prd'] for p in group if p.get('prd')), None)
        label = prd if prd else str(project_root)

        entry: dict = {
            'pids': [p['pid'] for p in group],
            'prd': prd,
            'label': label,
            'project_root': str(project_root),
            'running': any(p['running'] for p in group),
            'started': group[0]['started'],
            'last_update': last_update,
            'tasks': tasks,
            'summary': summary,
            'offline': offline,
        }
        if fetch_error:
            entry['error'] = fetch_error
        result.append(entry)

    return result
