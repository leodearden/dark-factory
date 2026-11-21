"""Discovering orchestrator PROCESSES: which ones run, and against what root.

Scans running processes (via ``ps aux``) and resolves each to the project root
it targets. That is the whole job. ``discover_orchestrators`` is async only
because the ``ps`` scan runs via ``asyncio.to_thread``; the scanning helper
itself is sync.

RETIRED, and deliberately: this module used to additionally fetch EVERY
resolved root's whole task tree from fused-memory — under a two-layer budget,
a per-root cache and an offline/degraded split of its own — to compute a
five-key ``summary`` and a ``last_update`` for each entry. Those were the only
consumers of the fetch, and both are gone from the wire. The task snapshot
unit behind ``/api/v2/dashboard/tasks``
(``dashboard/src/dashboard/data/task_snapshot.py``) is the SINGLE place the
dashboard measures a task population now, so a second implementation here
would be one more copy to keep in agreement by hand — and the one that fell
behind, since nothing on this route ever validated its counts. A handler that
reads no task tree needs no budget for one, which is why the budget constants
went with it rather than being left inert.

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

2. Config DEFAULTS layering (:func:`read_run_slot_capacity`) — the
   orchestrator's effective config is ``_deep_merge(_load_defaults(), project_config)``
   (``orchestrator/src/orchestrator/config.py``), so a key a project's YAML
   omits is still in force from ``orchestrator/src/orchestrator/defaults.yaml``.
   Reading a project YAML alone therefore misreports the parity DENOMINATOR.
   The two defaults this module needs are restated as
   :data:`_ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS` and
   :data:`_ORCHESTRATOR_DEFAULT_MAIN_HEALTH_FIX_RESERVED_SLOTS`; anyone
   changing ``max_concurrent_tasks`` or ``main_health_fix_reserved_slots`` in
   ``defaults.yaml`` must update its constant (a test asserts each pair agrees
   whenever the orchestrator source is present).

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
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from dashboard.config import _CANONICAL_CONFIG_NAME, DashboardConfig

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
    raw = _load_config_mapping(Path(config_path))
    if raw is None:
        return None
    value = raw.get('project_root')
    if not isinstance(value, str):
        return None
    p = Path(_expand_env_placeholders(value))
    return p.resolve() if p.is_absolute() else None


_config_mapping_cache: dict[Path, tuple[tuple[int, int], Mapping[str, object] | None]] = {}
"""Parsed config per path, valid while the file's ``(st_mtime_ns, st_size)`` holds.

One entry per path (a changed file replaces its entry), so the cache is bounded
by the number of distinct config paths ever seen. A parse failure is cached
under the same key as a success: fixing the file changes the key.
"""


def _load_config_mapping(path: Path) -> Mapping[str, object] | None:
    """Parse *path* as YAML and return it if it is a mapping, else ``None``.

    The parse is cached on the file's ``(st_mtime_ns, st_size)``: this is
    reached from every ``/orchestrators`` request (via
    :func:`_read_project_root_from_config`) as well as the burndown collector,
    and the pure-Python YAML parse is measurable on the event loop. An edited
    file is re-parsed on the next call. A file that cannot be stat-ed is never
    cached. Every caller shares the cached mapping, so it is returned as a
    read-only view.

    Never raises: this runs inside the burndown collector loop, where an
    exception would take down a whole collection cycle — strictly worse than
    an unknown value. ``ValueError`` covers ``UnicodeDecodeError`` for a
    non-text file.
    """
    import yaml

    try:
        st = path.stat()
    except OSError as exc:
        logger.debug('Unreadable orchestrator config %s: %s', path, exc)
        return None
    key = (st.st_mtime_ns, st.st_size)
    cached = _config_mapping_cache.get(path)
    if cached is not None and cached[0] == key:
        return cached[1]

    try:
        text = path.read_text()
    except (OSError, ValueError) as exc:
        logger.debug('Unreadable orchestrator config %s: %s', path, exc)
        return None
    try:
        raw = yaml.safe_load(text)
    except (ValueError, yaml.YAMLError) as exc:
        logger.debug('Unparseable orchestrator config %s: %s', path, exc)
        raw = None
    mapping = MappingProxyType(raw) if isinstance(raw, dict) else None
    _config_mapping_cache[path] = (key, mapping)
    return mapping


# FORMAT COUPLING item 2 (see the module docstring). Restate
# ``orchestrator/src/orchestrator/defaults.yaml``'s ``max_concurrent_tasks``
# and ``main_health_fix_reserved_slots``, which ``orchestrator.config``
# deep-merges UNDER every project config. A project that omits either key
# therefore runs with these values, not with none — so the reader must layer
# them the same way or the parity alarm misjudges those projects (the exact
# E12 miss it exists to catch). ``dashboard/tests/test_orchestrator.py``
# asserts both constants against that file whenever the orchestrator source is
# present, so drift fails a test rather than degrading an alarm.
_ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS = 24
_ORCHESTRATOR_DEFAULT_MAIN_HEALTH_FIX_RESERVED_SLOTS = 1


def _coerce_slot_count(
    data: Mapping[str, object], key: str, default: int, path: Path
) -> int | None:
    """Read slot-count *key* from *data* (parsed from *path*) as a usable count, or ``None``.

    ``None`` means UNKNOWN, which the parity alarm must never conflate with
    "not breaching", and is reserved STRICTLY for "the value is malformed". A
    malformed value logs a WARNING naming the key, the value and the file,
    because that is a config defect an operator needs to see.

    An ABSENT key is not unknown: the orchestrator deep-merges
    ``defaults.yaml`` under the project config, so the project runs with
    *default*. Returning ``None`` here would exclude the whole project from
    the parity alarm — of the live configs under ``/home/leo/src``, two omit
    ``max_concurrent_tasks``.

    Boundaries:

    * An explicit YAML null (``key:`` with no value) is MALFORMED, not absent:
      ``OrchestratorConfig`` types both slot counts ``int``, so such a config
      fails validation and no orchestrator runs from it at all.
    * ``bool`` is rejected explicitly — it is an ``int`` subclass, so a bare
      ``isinstance(value, int)`` would silently read ``true`` as 1.
    * ``0`` is KEPT as a real count ("dispatch nothing" / "no reserve").
      Tasks still in-progress against a 0 cap are a genuine breach; reporting
      that as unknown would hide it.
    * Negative counts are nonsense and rejected.
    * A numeric *string* is accepted after ``${VAR:default}`` expansion —
      that is how a config spells an env-driven int.
    """
    if key not in data:
        logger.debug(
            'No %s in %s — applying the orchestrator default of %d '
            '(defaults.yaml is merged under every project config)',
            key,
            path,
            default,
        )
        return default

    value = data[key]
    if isinstance(value, bool):
        count = None
    elif isinstance(value, int):
        count = value
    elif isinstance(value, str):
        expanded = _expand_env_placeholders(value).strip()
        try:
            count = int(expanded)
        except ValueError:
            count = None
    else:
        count = None

    if count is None or count < 0:
        logger.warning(
            'Ignoring unusable %s %r in %s — treating the run-slot capacity as unknown',
            key,
            value,
            path,
        )
        return None
    return count


def _resolve_orchestrator_config(
    project_root: Path | str,
) -> tuple[Path, Mapping[str, object]] | None:
    """Read *project_root*'s ``dark-factory-orchestrator.yaml``: its path and parsed mapping.

    That is the only name read. ``None`` means the file is absent or does not
    parse to a mapping.

    The file is authoritative for EVERY key read from it: a key it omits
    takes the orchestrator default and a key it spells badly is unknown, but
    neither is ever supplied by another file.

    Never raises — see :func:`_load_config_mapping`.
    """
    canonical = Path(project_root) / _CANONICAL_CONFIG_NAME
    if not canonical.is_file():
        return None
    data = _load_config_mapping(canonical)
    return None if data is None else (canonical, data)


def read_run_slot_capacity(project_root: Path | str) -> int | None:
    """Read *project_root*'s run-slot capacity, or ``None`` if unknown.

    The capacity is ``max_concurrent_tasks + main_health_fix_reserved_slots``:
    the orchestrator holds reserved slots beyond the general pool that only a
    main-health fix may take, so a fix running on the reserve is legitimate
    occupancy, not a breach (``docs/main-health-fix-run-slot-reserve.md``).

    This is the burndown parity alarm's denominator. Both keys are
    restart-only (red-tier: absent from ``config.py``'s hot-reload allowlist,
    and the scheduler is sized once at startup), but a burndown window spans
    restarts and the capacity varies between projects, so it is TIME-VARYING
    across the window regardless. The collector therefore calls this once per
    snapshot and stores the answer ON the snapshot row: comparing a historical
    in-progress census against today's capacity would forgive a past breach
    after a raise and invent one after a cut. Callers must treat ``None`` as
    UNKNOWN, never as "not breaching".

    Both keys come from the one file :func:`_resolve_orchestrator_config`
    reads. One it OMITS gets the orchestrator's own default
    (:data:`_ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS`,
    :data:`_ORCHESTRATOR_DEFAULT_MAIN_HEALTH_FIX_RESERVED_SLOTS`), because
    ``orchestrator.config`` deep-merges ``defaults.yaml`` under every project
    config. Either key being malformed makes the whole capacity unknown:
    dropping only a bad reserve would understate capacity and false-alarm.
    """
    resolved = _resolve_orchestrator_config(project_root)
    if resolved is None:
        return None
    path, data = resolved
    general = _coerce_slot_count(
        data, 'max_concurrent_tasks', _ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS, path
    )
    reserved = _coerce_slot_count(
        data,
        'main_health_fix_reserved_slots',
        _ORCHESTRATOR_DEFAULT_MAIN_HEALTH_FIX_RESERVED_SLOTS,
        path,
    )
    if general is None or reserved is None:
        return None
    return general + reserved


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


async def discover_orchestrators(config: DashboardConfig) -> list[dict]:
    """Every running orchestrator process, grouped by the root it targets.

    Returns one entry per resolved project root — ``pids``, ``prd``, ``label``,
    ``project_root``, ``running``, ``started`` — or ``[]`` when nothing is
    running. Multiple PIDs that resolve to the same canonical root are ONE
    entry with every pid in ``pids``, which is what makes the tab's row count
    a count of projects rather than of processes.

    NO TASK TREE IS READ, and *config* is the only argument because none is
    needed for one: the MCP client this used to take is gone with the fetch.
    Task counts live on ``/api/v2/dashboard/tasks``, inside a
    ``Datum`` that says when they were measured — see
    ``dashboard/src/dashboard/data/task_snapshot.py``. Two consequences worth
    stating, since a reader may look for the old ones:

    * there is no budget here to overrun, and no degraded/offline outcome to
      report — nothing is attempted that could fail. ``ps`` is a local scan;
    * ``last_update`` is no longer emitted. It was the max ``updated_at``
      across a root's tasks, and nothing here reads a task any more.
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
        groups.setdefault(_resolve_root(proc), []).append(proc)

    result: list[dict] = []
    for project_root, group in groups.items():
        # Display label: prefer PRD path, fall back to project root path.
        prd = next((p['prd'] for p in group if p.get('prd')), None)
        result.append({
            'pids': [p['pid'] for p in group],
            'prd': prd,
            'label': prd if prd else str(project_root),
            'project_root': str(project_root),
            'running': any(p['running'] for p in group),
            'started': group[0]['started'],
        })
    return result
