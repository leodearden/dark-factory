"""Functions for discovering orchestrator processes and reading task artifacts.

Scans running processes (via ``ps aux``), reads per-worktree ``.task/``
artifacts from disk, and fetches task trees from the fused-memory MCP
server (which is the source of truth post-2026-05-02 SQLite cutover).
``discover_orchestrators`` is async because the MCP call is async; the
process-scanning and worktree-artifact helpers remain sync and run via
``asyncio.to_thread`` from the async caller.

FORMAT COUPLING
================
This module re-derives two formats it does not own. It does NOT import the
``orchestrator`` package on purpose — ``dashboard/pyproject.toml`` depends
only on ``escalation`` + ``dark-factory-shared``, and import unification was
evaluated and rejected (see ``plans/dashboard-alignment-prd.md`` task ζ). If
either upstream format changes, this module must be updated by hand.

1. ps-scan launch patterns (:func:`find_running_orchestrators`) — the
   ``'orchestrator run'`` substring match and the ``--prd``/``--config``
   regexes re-derive the CLI surface of the ``run`` command defined in
   ``orchestrator/src/orchestrator/cli.py`` (grep that file for ``def run``).
   Anyone renaming the ``run`` command or its ``--prd``/``--config`` flags
   must update ``find_running_orchestrators`` to match.

2. ``.task/``/``.task-meta/`` artifact layout (:func:`read_task_artifacts`) —
   the ``metadata.json`` shape, ``plan.json``'s ``steps`` list (per-step
   ``status``, top-level ``files`` list), ``iterations.jsonl`` (read as a
   line count), and ``reviews/*.json``'s ``verdict`` field all re-derive the
   artifact layout owned by ``orchestrator/src/orchestrator/artifacts.py``.
   Note: ``artifacts.py`` tracks step ``status`` across BOTH the
   ``prerequisites`` and ``steps`` collections (see ``update_step_status``,
   ``get_pending_steps``, ``get_completed_steps``), but
   ``read_task_artifacts`` only counts ``steps`` toward ``plan_progress`` —
   ``prerequisites`` status is intentionally not surfaced here.

   W11-β relocated these artifacts to a SIBLING
   ``<worktree_base>/.task-meta/<worktree_name>`` dir (owned by
   ``TaskArtifacts.meta_root_for``; dirname owned by
   ``config.TASK_META_DIRNAME``). ``read_task_artifacts`` re-derives that
   path shape BY HAND (``worktree_path.parent / '.task-meta' /
   worktree_path.name``) since it cannot import the orchestrator package,
   and resolves each artifact new-then-old (the new path wins once it
   exists, else the legacy ``<worktree>/.task`` path), mirroring
   ``TaskArtifacts._read_path``. As of W11-ε2, ``metadata.json`` and
   ``plan.json`` resolve new-then-old; ``iterations.jsonl`` and
   ``reviews/*.json`` still read the legacy path only (relocated in a
   follow-up step). The legacy fallback is dropped entirely at task ι
   after a full green compat-window cycle.
   FUTURE SINGLE OWNER: stream W11's ``TaskArtifacts`` (see the seam table
   in ``plans/bug-hotspot-remediation-program-2026-07-06.md``) — this doc
   block is the marker W11 greps for to find and migrate this dashboard
   reader.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from pathlib import Path

import httpx
from shared.safe_io import load_json_or_warn

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


def read_task_artifacts(worktree_path: Path) -> dict:
    """Read task artifacts for a worktree — from the relocated .task-meta/
    dir (new) or the legacy .task/ dir (old).

    Returns a dict with keys:
    - metadata: parsed metadata.json dict, or None
    - phase: 'PLAN', 'EXECUTE', or 'DONE'
    - plan_progress: {'done': int, 'total': int}
    - iteration_count: number of lines in iterations.jsonl
    - review_summary: 'N/M passed' or '—' if no reviews

    FORMAT COUPLING: the artifact layout parsed below (metadata.json,
    plan.json 'steps'/'files', iterations.jsonl, reviews/*.json 'verdict')
    is owned by orchestrator/src/orchestrator/artifacts.py. Only plan.json's
    'steps' collection is counted toward plan_progress; 'prerequisites'
    status (also tracked by artifacts.py) is intentionally not surfaced.

    metadata.json and plan.json are resolved new-then-old: read from the
    relocated <worktree_base>/.task-meta/<worktree_name> dir
    (worktree_path.parent / '.task-meta' / worktree_path.name) when present,
    else fall back to the legacy <worktree>/.task dir. This hand-derives the
    path shape owned by TaskArtifacts.meta_root_for / config.TASK_META_DIRNAME
    (the dashboard cannot import the orchestrator package — see the module
    docstring's FORMAT COUPLING section) and mirrors
    TaskArtifacts._read_path's new-then-old resolution. iterations.jsonl and
    reviews/ are relocated in a follow-up step; until then they read the
    legacy path only. The legacy fallback is retired at task ι once a full
    compat-window cycle confirms every lane has migrated.

    FUTURE SINGLE OWNER: stream W11's TaskArtifacts — see the module
    docstring's FORMAT COUPLING section.
    """
    legacy_dir = worktree_path / '.task'
    meta_dir = worktree_path.parent / '.task-meta' / worktree_path.name

    def _resolve(name: str) -> Path:
        """Resolve *name* new-then-old (compat window).

        Returns meta_dir/name if it exists, else legacy_dir/name if THAT
        exists, else meta_dir/name as the canonical (non-existent) path —
        mirrors TaskArtifacts._read_path. Returning the canonical new path
        when neither copy exists keeps load_json_or_warn silent on
        benign-absent artifacts.
        """
        new_path = meta_dir / name
        if new_path.exists():
            return new_path
        legacy_path = legacy_dir / name
        if legacy_path.exists():
            return legacy_path
        return new_path

    # Metadata
    metadata, _meta_ok = load_json_or_warn(_resolve('metadata.json'), default=None)
    if not isinstance(metadata, dict):
        metadata = None

    # Plan progress, phase, and files
    done_count = 0
    total_count = 0
    files: list[str] = []
    plan_data, _ok = load_json_or_warn(_resolve('plan.json'), default=None)
    if isinstance(plan_data, dict):
        steps = plan_data.get('steps', [])
        if isinstance(steps, list):
            total_count = len(steps)
            done_count = sum(1 for s in steps if isinstance(s, dict) and s.get('status') == 'done')
        raw_files = plan_data.get('files', [])
        files = raw_files if isinstance(raw_files, list) else []

    if total_count == 0:
        phase = 'PLAN'
    elif done_count == total_count:
        phase = 'DONE'
    else:
        phase = 'EXECUTE'

    # Iteration count
    iteration_count = 0
    try:
        with open(legacy_dir / 'iterations.jsonl') as f:
            iteration_count = sum(1 for _ in f)
    except FileNotFoundError:
        pass

    # Review summary
    review_summary = '—'
    reviews_dir = legacy_dir / 'reviews'
    if reviews_dir.is_dir():
        review_files = list(reviews_dir.glob('*.json'))
        if review_files:
            total_reviews = len(review_files)
            passed = 0
            for review_file in review_files:
                review, _ok = load_json_or_warn(review_file, default=None)
                if isinstance(review, dict) and review.get('verdict') == 'PASS':
                    passed += 1
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
    # Cache tuple: (tasks, worktrees, offline, error)
    project_cache: dict[Path, tuple[list[dict], dict[int, dict], bool, str | None]] = {}

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
            worktrees = await asyncio.to_thread(_scan_worktrees, project_root / '.worktrees')
            project_cache[project_root] = (tasks, worktrees, offline, fetch_error)

        tasks, worktrees, offline, fetch_error = project_cache[project_root]
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
            'worktrees': worktrees,
            'summary': summary,
            'offline': offline,
        }
        if fetch_error:
            entry['error'] = fetch_error
        result.append(entry)

    return result
