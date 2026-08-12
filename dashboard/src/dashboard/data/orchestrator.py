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

2. Config DEFAULTS layering (:func:`read_max_concurrent_tasks`) — the
   orchestrator's effective config is ``_deep_merge(_load_defaults(),
   project_config)`` (``orchestrator/src/orchestrator/config.py``), so a key
   a project's YAML omits is still in force from
   ``orchestrator/src/orchestrator/defaults.yaml``. Reading a project YAML
   alone therefore under-reports, and for a parity DENOMINATOR that silently
   disables the alarm rather than loosening it. The one default this module
   needs is restated as
   :data:`_ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS`; anyone changing
   ``max_concurrent_tasks`` in ``defaults.yaml`` must update it (a test
   asserts the two agree whenever the orchestrator source is present).

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


def _expand_env_placeholders(value: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:default}`` in an orchestrator-config scalar.

    Matches orchestrator ``config.py`` behaviour: an unset ``VAR`` with no
    default expands to the empty string, and callers decide what that means
    (for both readers below: "unknown", never a usable value).
    """
    import os

    return re.sub(
        r'\$\{([^:}]+)(?::([^}]*))?\}',
        lambda m: os.environ.get(m.group(1), m.group(2) or ''),
        value,
    )


def _read_project_root_from_config(config_path: str) -> Path | None:
    """Extract ``project_root`` from an orchestrator config YAML file.

    Handles ``${VAR:default}`` env-var expansion for the project_root value.
    Returns ``None`` if the file can't be read or doesn't contain project_root.
    """
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
    p = Path(_expand_env_placeholders(value))
    return p.resolve() if p.is_absolute() else None


def _load_config_mapping(path: Path) -> dict | None:
    """``yaml.safe_load`` *path* and return it if it is a mapping, else ``None``.

    Never raises: this runs inside the burndown collector loop, where an
    exception would take down a whole collection cycle — strictly worse than
    an unknown value. ``ValueError`` covers ``UnicodeDecodeError`` for a
    non-text file.
    """
    import yaml

    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, ValueError, yaml.YAMLError) as exc:
        logger.debug('Unreadable orchestrator config %s: %s', path, exc)
        return None
    return raw if isinstance(raw, dict) else None


# FORMAT COUPLING item 2 (see the module docstring). Restates
# ``orchestrator/src/orchestrator/defaults.yaml``'s ``max_concurrent_tasks``,
# which ``orchestrator.config`` deep-merges UNDER every project config. A
# project that omits the key therefore runs with this cap, not with no cap —
# so the reader must layer it the same way or the parity alarm silently never
# fires for those projects (the exact E12 miss it exists to catch).
# ``dashboard/tests/test_orchestrator.py`` asserts this constant against that
# file whenever the orchestrator source is present, so drift fails a test
# rather than degrading an alarm.
_ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS = 24

_CAP_KEY_ABSENT = object()
"""Sentinel: the config mapping has no ``max_concurrent_tasks`` key at all.

Distinct from an explicit ``max_concurrent_tasks:`` (YAML null). Absent means
"the orchestrator's own default applies"; an explicit null is a config defect —
``OrchestratorConfig`` types the field ``int``, so such a config fails
validation and no orchestrator runs from it at all.
"""


def _coerce_concurrency_cap(value: object, path: Path) -> int | None:
    """Coerce a raw ``max_concurrent_tasks`` value to a usable cap, or ``None``.

    ``None`` means UNKNOWN, which the parity alarm must never conflate with
    "not breaching", and is now reserved STRICTLY for "the file is unreadable
    or the value is malformed". A malformed value logs a WARNING naming the
    file and the value, because that is a config defect an operator needs
    to see.

    An ABSENT key (*value* is :data:`_CAP_KEY_ABSENT`) is not unknown: the
    orchestrator deep-merges ``defaults.yaml`` under the project config, so
    the project runs with :data:`_ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS`.
    Returning ``None`` here would exclude the whole project from the parity
    alarm — of the live configs under ``/home/leo/src``, two omit the key.

    Boundaries:

    * ``bool`` is rejected explicitly — it is an ``int`` subclass, so a bare
      ``isinstance(value, int)`` would silently read ``true`` as a cap of 1.
    * ``0`` is KEPT as a real cap ("dispatch nothing"). Tasks still
      in-progress against a 0 cap are a genuine breach; reporting that as
      unknown would hide it.
    * Negative caps are nonsense and rejected.
    * An explicit YAML null is MALFORMED, not absent — see
      :data:`_CAP_KEY_ABSENT`.
    * A numeric *string* is accepted after ``${VAR:default}`` expansion —
      that is how a config spells an env-driven int.
    """
    if value is _CAP_KEY_ABSENT:
        logger.debug(
            'No max_concurrent_tasks in %s — applying the orchestrator default '
            'of %d (defaults.yaml is merged under every project config)',
            path,
            _ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS,
        )
        return _ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS

    raw = value
    if isinstance(value, bool):
        cap = None
    elif isinstance(value, int):
        cap = value
    elif isinstance(value, str):
        expanded = _expand_env_placeholders(value).strip()
        try:
            cap = int(expanded)
        except ValueError:
            cap = None
    else:
        cap = None

    if cap is None or cap < 0:
        logger.warning(
            'Ignoring unusable max_concurrent_tasks %r in %s — treating the '
            'concurrency cap as unknown',
            raw,
            path,
        )
        return None
    return cap


def read_max_concurrent_tasks(project_root: Path | str) -> int | None:
    """Read *project_root*'s orchestrator concurrency cap, or ``None`` if unknown.

    This is the burndown parity alarm's denominator. ``max_concurrent_tasks``
    is restart-only (red-tier: it is absent from ``config.py``'s hot-reload
    allowlist, and the scheduler semaphore is sized once at startup), but a
    burndown window spans restarts and the cap varies between projects, so it
    is TIME-VARYING across the window regardless. The collector therefore calls
    this once per snapshot and stores the answer ON the snapshot row: comparing
    a historical in-progress census against today's cap would forgive a past
    breach after a cap raise and invent one after a cut. Callers must treat
    ``None`` as UNKNOWN, never as "not breaching".

    ``None`` is reserved for a config that is ABSENT, unreadable, or carries a
    malformed value. A readable config that simply OMITS the key yields the
    orchestrator's own default
    (:data:`_ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS`), because
    ``orchestrator.config`` deep-merges ``defaults.yaml`` under every project
    config — every running orchestrator has a cap, whether or not its YAML
    spells one. Reading such a project as capless would drop it out of the
    parity alarm entirely.

    Resolution mirrors ``dashboard.config._discover_root_escalation_url`` and
    reuses its filename constants so the two cannot fork: the canonical
    ``dark-factory-orchestrator.yaml`` is authoritative once it exists on
    disk (a live config must not be masked by a stale legacy file — including
    when it omits the key, where the default applies rather than the legacy
    file's value), and only its outright absence falls through to
    ``_LEGACY_CONFIG_NAMES`` in order, taking the first readable one.

    Unlike that startup-time discovery helper, the legacy-spelling nudge here
    is logged at DEBUG, not WARNING: this runs every collection cycle
    (~10 min per project), so a WARNING would be recurring log spam rather
    than the bounded once-per-process reminder that one is.

    Never raises — see :func:`_load_config_mapping`.
    """
    from dashboard.config import _CANONICAL_CONFIG_NAME, _LEGACY_CONFIG_NAMES

    root = Path(project_root)

    canonical = root / _CANONICAL_CONFIG_NAME
    if canonical.is_file():
        data = _load_config_mapping(canonical)
        if data is None:
            return None
        return _coerce_concurrency_cap(
            data.get('max_concurrent_tasks', _CAP_KEY_ABSENT), canonical
        )

    for legacy_name in _LEGACY_CONFIG_NAMES:
        legacy_path = root / legacy_name
        if not legacy_path.is_file():
            continue
        data = _load_config_mapping(legacy_path)
        if data is None:
            continue
        cap = _coerce_concurrency_cap(
            data.get('max_concurrent_tasks', _CAP_KEY_ABSENT), legacy_path
        )
        if cap is not None:
            logger.debug(
                'Project %s: read max_concurrent_tasks from legacy config path %s '
                '(expected %s)',
                root.name,
                legacy_path,
                _CANONICAL_CONFIG_NAME,
            )
            return cap
    return None


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
