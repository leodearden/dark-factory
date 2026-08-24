"""Aggregate active tasks across all known projects for the redux dashboard.

Joins three sources — task tree (via fused-memory MCP), per-task runtime
state (via the orchestrator's escalation MCP, ``get_task_runtime_state``),
and optional burst state from reconciliation — into the ``ACTIVE_TASKS``
shape consumed by the React dashboard's tasks tab.

Output shape (per task) matches ``data.js`` mock fixtures:

    {
        'id': 'dark_factory/T-19',
        'project': 'dark_factory',
        'title': '...',
        'description': '...',
        'details': '...',         # may be empty; many tasks have none
        'status': 'in-progress',
        'agent': 'claude-task-19',  # TaskRuntimeEntry.has_worktree; None if no worktree.
                                    # WORKTREE PRESENCE, NOT LIVENESS — see 'stranded'.
        'claimant_run_id': 'run-1/sess-1/pid=42',  # MCP get_tasks claim column; None if unclaimed
        'heartbeat_at': '2026-08-08T12:00:00+00:00',  # MCP get_tasks claim column; None if never
        'stranded': False,          # tasks.task_is_stranded(task, now) — in-progress with no live
                                    # claimant (null/blank claimant, or a heartbeat older than
                                    # STRANDED_HEARTBEAT_TTL). Independent of 'agent': a leftover
                                    # worktree makes 'agent' truthy while nothing is running.
        'started': 14,              # minutes since TaskRuntimeEntry.started (runtime snapshot)
        'loops': 2,                 # TaskRuntimeEntry.loops (runtime snapshot)
        'attempts': 3,              # TaskRuntimeEntry.attempts (runtime snapshot)
        'lane': '_lane-7',          # TaskRuntimeEntry.lane, or None
        'phase': 'EXECUTE',         # TaskRuntimeEntry.phase, or None
        'lane_state': 'assigned',   # TaskRuntimeEntry.lane_state, or None
        'runtime_offline': False,   # True iff this project's runtime snapshot is unreachable —
                                     # loops/attempts/started/agent/lane/phase/lane_state are then
                                     # ALL None (never a fabricated 0). A task absent from an
                                     # online snapshot instead gets honest zeros/None with
                                     # runtime_offline False; a per-task read failure on an online
                                     # snapshot yields None fields with runtime_offline still False
                                     # (honest error != offline). See _runtime_fields.
        'deps': [{'id': 'dark_factory/T-15', 'title': '...', 'done': True}, ...],
        'meta_files': ['src/...py', ...],  # taskmaster metadata.files; retained on API for
                                           # debugging/tooling — no frontend UI reads it directly
        'train': {'id': 'demo', 'order': 0},  # present when metadata.train is set; None otherwise
    }

Lock state is surfaced via the scheduler endpoint (see /api/v2/dashboard/scheduler).
The bespoke FILE_LOCKS derivation has been removed; all lock display routes through
``D.SCHEDULER.{rows,modules}`` on the frontend.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import httpx
from shared.task_runtime_state import TaskRuntimeEntry, TaskRuntimeSnapshot

from dashboard.config import DashboardConfig
from dashboard.data.task_runtime import fetch_task_runtime
from dashboard.data.tasks import (
    fetch_external_statuses,
    fetch_statuses,
    fetch_tasks,
    task_is_stranded,
)
from dashboard.data.utils import resolve_now

logger = logging.getLogger(__name__)

_ACTIVE_STATUSES = {'in-progress', 'blocked', 'pending', 'merge-deferred', 'deferred'}
# The two terminal buckets the Tasks tab renders. Sent as the ``statuses``
# filter for the bounded terminal window, so this set is what crosses the wire
# — keep it in step with the (status, cap) pairs iterated below.
_TERMINAL_STATUSES = {'done', 'cancelled'}

# Maximum done / cancelled tasks to include per project when the caller opts in
# via ``max_done_per_project`` / ``max_cancelled_per_project``.
# Kept at module level so app.py can import them.
# Upper bound on how many terminal (done + cancelled) rows are pulled per
# project per render — 8x the 50-row _MAX_DONE_PER_PROJECT render cap below.
#
# This is the ceiling: without it the Tasks tab pulled every done row in the
# tree (~4000 rows / ~40 MB on dark-factory) to render at most 50 of them.
# 8x rather than 1x because the window is selected by DESCENDING TASK ID
# while the render cap selects by ``updated_at`` — see _shape_one_project's
# docstring for why those differ and when the gap can bite.
_TERMINAL_FETCH_WINDOW = 400

_MAX_DONE_PER_PROJECT = 50
_MAX_CANCELLED_PER_PROJECT = 50

# --- Budget constants -------------------------------------------------------
#
# Written in the ``app._HEALTHZ_TOTAL_BUDGET`` / ``app._DB_PROBE_TIMEOUT``
# idiom: a per-unit budget, a whole-handler budget, and an introspectable
# roster of the units, so ``tests/test_tasks_budget.py`` can machine-check
# that the parts fit the whole instead of a human re-deriving the arithmetic
# every time one of them moves.

# The MCP calls ``_shape_one_project`` issues for ONE project root. A named
# tuple rather than a literal ``3`` deliberately: the invariant then tracks
# reality, so adding a fourth per-project call without raising the budget
# fails the structural test rather than silently overrunning in production.
_PER_PROJECT_MCP_CALLS: tuple[str, ...] = (
    'get_tasks[active]',
    'get_statuses',
    'get_tasks[terminal]',
)

# Whole-operation bound for ONE project root, enforced by ``asyncio.wait_for``
# in ``collect_tasks_with_counts``.
#
# ``tasks.DEFAULT_PER_CALL_TIMEOUT`` (2.0) * 3 calls = 6.0 <= 7.0, leaving
# 1.0 s of slack so this deadline is a real backstop for non-MCP overhead
# (JSON decode, row shaping, event-loop scheduling) rather than coinciding
# exactly with the sum of its parts — the same reasoning as healthz's
# ``_DB_PROBE_TIMEOUT * 3 = 2.7 <= _HEALTHZ_TOTAL_BUDGET = 3.0``.
#
# What that sum does and does NOT claim: it bounds the sum of the
# PER-HTTP-REQUEST budgets. It does NOT bound a cold MCP session, which
# performs three posts (initialize, notifications/initialized, tools/call) and
# so can reach ``3 * DEFAULT_PER_CALL_TIMEOUT`` for a SINGLE tool call. That
# residual is exactly what this ``wait_for`` layer exists to cap: the two
# layers are complementary, not redundant (the same two-layer note
# ``dashboard/src/dashboard/data/task_runtime.py``'s module docstring carries).
_TASKS_PER_PROJECT_BUDGET = 7.0

# Whole-handler deadline for the entire multi-project aggregation.
#
# Strictly below ``data.js``'s 30 000 ms fetch abort with 10 s of headroom for
# HTTP and JSON serialisation, so the PARTIAL payload the deadline produces is
# actually deliverable to the browser that asked for it. It replaces a
# structural worst case of roughly ``roots * 3 posts * 10 s`` with no cap at
# all.
#
# Raising any one of these three constants requires re-checking the others —
# they are mutually constrained, and ``test_tasks_budget.py`` enforces that.
# None of them may be raised toward ``memory.mcp_tool_call``'s 10 s default.
_TASKS_TOTAL_BUDGET = 20.0

# Defensive-visibility threshold: a PRD with an unusually large number of live
# done/cancelled members beyond the per-bucket cap logs a warning, so a
# pathological case is visible rather than silently inflating the payload.
#
# This comment used to also assert that the exemption "never drops rows —
# 'all live members' is the contract, not 'up to N'". _TERMINAL_FETCH_WINDOW
# made that FALSE and the correction belongs here, at the site that states the
# contract, not only in _shape_one_project's docstring: the exemption can only
# exempt rows that were FETCHED, so a live PRD's done/cancelled members with
# ids below the window's high-id end are absent from `tasks` and can never be
# exempted at all. On a tree with more terminal tasks than the window
# (dark-factory has ~4000 against a 400-row window) the contract in
# plans/dashboard-taskgraph-legibility-prd.md is therefore no longer met, and
# front-end PRD member aggregation under-counts by exactly those rows.
#
# Left as a disclosed narrowing rather than silently widened here: raising the
# window to restore it trades directly against the payload budget this task
# exists to bound, and that trade is a product decision about the PRD contract
# rather than a defect in this module. Tracked as TASK 4416, which weighs the
# three candidate resolutions (amend the PRD contract / raise the window /
# fetch live-PRD members explicitly) — a task id, not "the review notes", so a
# future reader can actually check whether it was revisited. Do not "fix" it
# here by quietly bumping the window: the point of 4416 is that the contract
# in plans/dashboard-taskgraph-legibility-prd.md and this code must agree
# either way.
_LIVE_PRD_EXEMPTION_WARN_THRESHOLD = 200


def _project_label(root: Path) -> str:
    """Display label for a project root path: the directory's basename."""
    return root.name or str(root)


def _all_project_roots(config: DashboardConfig) -> list[Path]:
    """All known project roots, deduped, primary first."""
    seen: set[Path] = {config.project_root}
    roots: list[Path] = [config.project_root]
    for r in config.known_project_roots:
        if r not in seen:
            seen.add(r)
            roots.append(r)
    return roots


def _task_uid(project: str, task_id: int) -> str:
    """Project-scoped unique id used by the React tasks tab as a map key."""
    return f'{project}/T-{task_id}'


def _minutes_since(iso: str | None, *, now: datetime | None = None) -> int:
    """Whole minutes between *iso* and *now* (UTC). 0 on parse failure / future.

    *now* defaults to the live clock via :func:`dashboard.data.utils.resolve_now`;
    pass an explicit value for deterministic results or to share one instant
    across multiple rows in an aggregation.
    """
    if not iso:
        return 0
    try:
        ts = datetime.fromisoformat(iso.replace('Z', '+00:00'))
    except ValueError:
        return 0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    delta = resolve_now(now) - ts
    minutes = int(delta.total_seconds() // 60)
    return max(minutes, 0)


def _coalesce_prd(metadata: dict) -> str | None:
    """Coalesce PRD provenance from *metadata* into a single normalized string.

    Checks ``prd_path``, then ``prd``, then ``prd_ref`` (in that precedence
    order); the first value that is a non-empty string after stripping a
    trailing ``#anchor`` or ``§section`` suffix and surrounding whitespace
    wins. Non-string values are skipped. A value that cleans to ``''`` (e.g.
    it was only a suffix) falls through to the next key. Returns ``None``
    when no key yields a non-empty result.
    """
    for key in ('prd_path', 'prd', 'prd_ref'):
        raw = metadata.get(key)
        if not isinstance(raw, str):
            continue
        cleaned = raw.split('#', 1)[0].split('§', 1)[0].strip()
        if cleaned:
            return cleaned
    return None


def _build_task_row(
    project: str,
    task: dict,
    task_id: int,
    rt: dict,
    uid: str,
    *,
    prd: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Build the common row fields shared by active and done task rows.

    Returns a dict with all fields that are identical regardless of task
    status.  Callers add status-specific fields afterwards:
    active rows add ``started`` (minutes) and ``deps``; done rows add
    ``started: 0``, ``deps: []``, and ``completed`` (ISO timestamp or '').

    Every row carries the claim projection: the two raw MCP ``get_tasks``
    columns ``claimant_run_id``/``heartbeat_at`` (``None`` when the row
    predates them or the task is unclaimed) plus the computed ``stranded``
    boolean from :func:`dashboard.data.tasks.task_is_stranded`.  They are on
    BOTH the active and the terminal row shapes so the wire shape is uniform
    (a terminal row is simply never stranded — the shared predicate gates on
    ``status == 'in-progress'``).

    ``stranded`` is deliberately independent of ``agent``: see the comment at
    the ``agent`` assignment in :func:`_runtime_fields`.

    *rt* is the runtime-fields dict produced by :func:`_runtime_fields`
    (``agent``/``loops``/``attempts``/``lane``/``phase``/``lane_state``/
    ``runtime_offline`` — ``started`` is handled separately by the caller,
    since active rows use ``rt['started']`` while terminal rows hard-code
    ``0``). Missing keys default to ``None``/``False`` so a bare ``{}`` (used
    by direct unit tests of this function) is still valid.

    *prd*, if given, is used verbatim as the row's ``prd`` value instead of
    re-deriving it from *task*'s metadata via ``_coalesce_prd`` — callers
    that already computed it (e.g. the terminal-bucket loop, to decide
    live-PRD membership) can pass it through to avoid doing the same
    split/strip work twice. Omitting it (or passing ``None``, the actual
    no-provenance value) falls back to deriving it from metadata, which is
    safe because re-deriving a true ``None`` is idempotent.

    *now* is the reference instant for the strand verdict, threaded from the
    caller's single per-pass ``resolve_now`` (see ``_shape_one_project``) —
    this function never reads the clock itself, so every row in one pass is
    judged against the same instant.
    """
    metadata = task.get('metadata') or {}
    meta_files = list(metadata.get('files') or [])
    train_meta = metadata.get('train')
    train = (
        {'id': train_meta['id'], 'order': train_meta.get('order', 0)}
        if isinstance(train_meta, dict) and train_meta.get('id')
        else None
    )
    raw_ext = metadata.get('external_deps')
    external_deps = (
        [{'id': dep, 'status': 'unknown'}
         for dep in raw_ext
         if isinstance(dep, str) and dep]
        if isinstance(raw_ext, list) else []
    )
    return {
        'id': uid,
        'project': project,
        'title': task.get('title') or '',
        'description': task.get('description') or '',
        'details': task.get('details') or '',
        'status': task.get('status'),
        'agent': rt.get('agent'),
        'loops': rt.get('loops'),
        'attempts': rt.get('attempts'),
        'lane': rt.get('lane'),
        'phase': rt.get('phase'),
        'lane_state': rt.get('lane_state'),
        'runtime_offline': rt.get('runtime_offline', False),
        'claimant_run_id': task.get('claimant_run_id'),
        'heartbeat_at': task.get('heartbeat_at'),
        'stranded': task_is_stranded(task, now=now),
        'meta_files': meta_files,
        'train': train,
        'external_deps': external_deps,
        'prd': prd if prd is not None else _coalesce_prd(metadata),
    }


def _runtime_fields(
    index: dict[int, TaskRuntimeEntry],
    is_offline: bool,
    task_id: int,
    *,
    now: datetime | None = None,
) -> dict:
    """Derive a task row's runtime-sourced fields from the project's runtime snapshot.

    Three cases:

    - *is_offline* (the project's ``get_task_runtime_state`` snapshot reported
      ``offline=True``, or no escalation URL is configured for this project at
      all): every field is ``None`` — never a fabricated ``0`` — and
      ``runtime_offline`` is ``True``.
    - *task_id* absent from *index* (the project IS online, the task just has
      no entry in its snapshot): honest zeros (``loops``/``attempts``/
      ``started`` are ``0``; ``agent``/``lane``/``phase``/``lane_state`` are
      ``None``) and ``runtime_offline`` is ``False``.
    - *task_id* present in *index*: real fields from the ``TaskRuntimeEntry``,
      which may themselves be ``None`` on a per-task artifact read failure —
      an honest per-task error, not an offline project, so ``runtime_offline``
      stays ``False`` either way.
    """
    if is_offline:
        return {
            'agent': None, 'loops': None, 'attempts': None, 'started': None,
            'lane': None, 'phase': None, 'lane_state': None,
            'runtime_offline': True,
        }
    entry = index.get(task_id)
    if entry is None:
        return {
            'agent': None, 'loops': 0, 'attempts': 0, 'started': 0,
            'lane': None, 'phase': None, 'lane_state': None,
            'runtime_offline': False,
        }
    return {
        # ``agent`` is a WORKTREE-PRESENCE signal, not evidence of liveness: it
        # is truthy whenever a ``.worktrees/<id>`` directory exists, including
        # long after the agent that created it died. Reading it as "an agent is
        # working on this" is exactly the confusion this projection removes —
        # the liveness verdict is the row's independent ``stranded`` field,
        # derived from the claim columns via ``tasks.task_is_stranded``.
        'agent': f'claude-task-{task_id}' if entry.has_worktree else None,
        'loops': entry.loops,
        'attempts': entry.attempts,
        'started': _minutes_since(entry.started, now=now),
        'lane': entry.lane,
        'phase': entry.phase,
        'lane_state': entry.lane_state,
        'runtime_offline': False,
    }


def _resolve_deps(
    task: dict,
    by_id: dict[int, dict],
    project: str,
    *,
    status_map: Mapping[int, str] | None = None,
) -> list[dict]:
    """Resolve *task*'s ``dependencies`` ids into ``{id, title, done}`` dicts.

    *by_id* is the lookup over the rows this render actually fetched.  It used
    to span the whole tree; since ``_shape_one_project`` bounded the terminal
    fetch it does not, so a done dependency outside the window would drop out
    of ``by_id`` and lose its chip entirely.

    Resolution order, most to least informative:

    1. a full row in *by_id* — real title, real status;
    2. otherwise, an id present in *status_map* — an honest PARTIAL entry:
       the ``done`` flag is authoritative, the title degrades to ``''``.
       BRANCH (2) IS THE COMMON CASE ON A LARGE TREE, not a rare fallback: it
       fires for every dependency below the terminal window's high-id end, so
       on a project with far more terminal tasks than ``_TERMINAL_FETCH_WINDOW``
       (dark-factory: ~4000 against 400) MOST of an active task's done
       dependencies resolve titleless.  ``tab_tasks.jsx`` therefore renders the
       id ALONE for these — the `` · `` separator is emitted only when a title
       exists, or the chips would read as ``3502 ·`` with nothing after it;

    3. otherwise dropped, unchanged.  The id exists nowhere the dashboard can
       see, and fabricating a chip for it would be worse than omitting it.

    Why the title degrades rather than being fetched: ``get_tasks`` exposes no
    ``ids`` filter, so resolving one missing title costs an entire extra tree
    read — precisely the unbounded fetch this design removed.  The compact
    status map is the only BOUNDED source available, and the ``done`` flag is
    the load-bearing half of the chip (it drives the strike-through), so a
    titleless-but-correct chip beats a missing one.  ``''`` is already what
    this function emits for a row with no title, so no consumer needs a new
    guard.
    """
    deps: list[dict] = []
    for dep_id in task.get('dependencies') or []:
        dep_task = by_id.get(dep_id)
        if dep_task is not None:
            deps.append({
                'id': _task_uid(project, dep_id),
                'title': dep_task.get('title') or '',
                'done': dep_task.get('status') == 'done',
            })
            continue
        dep_status = (status_map or {}).get(dep_id)
        if dep_status is None:
            continue
        deps.append({
            'id': _task_uid(project, dep_id),
            'title': '',
            'done': dep_status == 'done',
        })
    return deps


async def _shape_one_project(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: Path,
    *,
    max_done_per_project: int = 0,
    max_cancelled_per_project: int = 0,
    now: datetime | None = None,
    runtime: TaskRuntimeSnapshot | None = None,
) -> tuple[list[dict], bool, int | None]:
    """Build ``(active_tasks, offline, done_count)`` for a single project root.

    *offline* is True when the MCP fetch failed for this project; the
    caller surfaces that in the API payload so the React Tasks tab can
    show an offline banner.

    *done_count* is the project's total number of ``'done'`` tasks, **before**
    the *max_done_per_project* cap is applied.  It comes from the compact
    ``fetch_statuses`` map (``{id: status}``), NOT from the fetched rows —
    the rows are now a bounded window, so counting them would undercount any
    project with more terminal tasks than the window holds.

    **Three bounded calls, not one unbounded one.**  This function used to
    issue a single unnarrowed ``fetch_tasks`` and derive active rows, terminal
    buckets and *done_count* from that one full tree; on dark-factory that
    meant transferring ~4000 done rows (~40 MB) per render to show at most 50
    of them.  It now issues:

    1. ``fetch_tasks(statuses=sorted(_ACTIVE_STATUSES))`` — active rows,
       filtered server-side in SQL;
    2. ``fetch_statuses(...)`` — the compact map, ~95% smaller, supplying both
       *done_count* and the terminal population that positions (3).  Issued
       concurrently with (1);
    3. ``fetch_tasks(statuses=sorted(_TERMINAL_STATUSES), page_size=..., offset=...)``
       — a bounded window of terminal rows, issued ONLY when a terminal cap is
       actually requested.  ``collect_active_tasks``'s scheduler path passes
       both caps as 0 and therefore transfers no terminal row at all.

    The only component that still grows with the tree is the ~15 B/task status
    map, not the ~10 KB/task rows.

    **Scope of that win — read this before quoting the numbers.**  Every claim
    above is about THIS function and the ``/api/v2/dashboard/tasks`` payload it
    shapes.  It is NOT a claim about the process's total MCP traffic per poll.
    Four other callers still issue an UNNARROWED ``fetch_tasks`` on the same
    poll cycle — ``app._load_task_cards``, ``data/orchestrator.py``,
    ``data/merge_queue.py`` and ``data/burndown.py`` — and since the
    ``fetch_tasks`` cache key now includes the narrowing args, the Tasks tab no
    longer shares their cached full tree.  Net per poll the process therefore
    still transfers the whole tree once for those callers AND additionally
    issues this function's narrowed calls.  What this change delivers is that
    the TASKS TAB no longer pulls the full tree and no longer grows with the
    terminal tree; removing the remaining whole-tree transfer means narrowing
    those four callers too, which is separate work and is not done here.

    **Two disclosed display-semantics changes**, both caused by what
    ``get_tasks`` does and does not offer:

    * The terminal window is selected by DESCENDING TASK ID.  That is the only
      ordering available — ``SqliteTaskBackend._get_tasks_internal`` is
      ``ORDER BY id`` and ``page_size``/``offset`` slice that ascending list,
      and there is no ``ORDER BY updated_at``.  Rows inside the window are
      still sorted by ``updated_at`` descending for display, so the common
      case is unchanged (tasks are filed and completed in roughly id order).
      The divergent case is real: a long-parked low-id task completing late
      can fall outside the window.  ``_TERMINAL_FETCH_WINDOW`` is therefore 8x
      the render cap, and truncation logs a WARNING rather than capping
      silently.
    * The live-PRD terminal-member exemption below now covers only members
      INSIDE the window, for the same reason.

    Benign race: *n_terminal* comes from a separate ``get_statuses`` read, so a
    task completing between the two calls can shift the window by a row.

    If the compact map read fails while the active fetch succeeded, the project
    is NOT declared offline — the active rows are still good.  Two things
    degrade together, because both depend on the map and on nothing else:
    *done_count* is returned as ``None`` (UNKNOWN — not zero, and not
    offline; the caller omits the project from ``DONE_COUNTS``), and the
    terminal window is SKIPPED entirely, so no done/cancelled row is emitted
    for that render.  Skipping is required, not merely tidy: the window's
    offset is computed from the map's terminal population, so without the map
    the offset collapses to 0 and — since ``page_size``/``offset`` slice an
    ASCENDING-id list — would select the OLDEST terminal rows and present them
    as the tab's most recent.  A WARNING is logged.  Only an offline ACTIVE
    fetch means offline.

    When *max_done_per_project* > 0, the most-recent N done tasks
    (sorted by ``updated_at`` descending, then ``id`` descending) are
    appended to the returned list.  Each done row carries a ``completed``
    field (the ``updated_at`` ISO string or ``''``).  Active rows are
    unaffected.

    When *max_cancelled_per_project* > 0, the most-recent N cancelled tasks
    are similarly appended (same sort key, same ``completed`` field, same
    ``started: 0`` / ``deps: []`` treatment as done rows).

    *runtime* is this project's ``TaskRuntimeSnapshot`` (resolved ONCE by the
    caller — see ``collect_tasks_with_counts`` — via ``fetch_task_runtime``).
    ``None`` (no escalation URL configured for this project) is treated
    identically to ``runtime.offline``: every row's runtime-sourced fields
    degrade to an honest ``None`` via :func:`_runtime_fields`, distinct from
    the task-tree ``offline`` return value above.
    """
    project = _project_label(project_root)
    # Resolve the reference instant ONCE per build pass — never per row — so
    # every row's ``started`` and ``stranded`` verdict share one instant.
    effective_now = resolve_now(now)

    # (1) active rows, SQL-filtered server-side, and (2) the compact
    # {id: status} map — concurrently, since neither depends on the other.
    #
    # (2) is UNCONDITIONAL, including on the scheduler path where done_count is
    # discarded. It looks gateable on `wants_terminal` and is not: the map is
    # also _resolve_deps' only bounded fallback for a dependency outside the
    # fetched rows, so skipping it would silently drop dependency chips — the
    # exact regression that fallback was added to prevent — on every render
    # that path serves. done_count is the map's cheapest product, not its only
    # one.
    #
    # Its COST is bounded by fetch_statuses' own 5 s TTL cache rather than by a
    # gate here. That matters because BOTH /api/v2/dashboard/tasks and
    # /api/v2/dashboard/scheduler reach this function (via
    # collect_tasks_with_counts) on every 3 s data.js poll: uncached, one
    # unconditional call per root became two full-population get_statuses reads
    # per root per poll — trading wire bytes for backend queries, which is not
    # the trade this change set out to make. Cached, the two endpoints share
    # one read and consecutive polls collapse. See
    # tasks._FETCH_STATUSES_TTL_SECONDS for why 5 s.
    fetched, status_map = await asyncio.gather(
        fetch_tasks(client, config, project_root, statuses=sorted(_ACTIVE_STATUSES)),
        fetch_statuses(client, config, project_root),
    )
    if isinstance(fetched, dict) and fetched.get('offline'):
        return [], True, 0
    tasks = list(fetched) if isinstance(fetched, list) else []

    # The compact map is the authoritative source of done_count (a count, not
    # rows) and of the terminal population that positions the window below.
    map_offline = not isinstance(status_map, dict) or bool(status_map.get('offline'))
    status_map = (
        {} if map_offline
        else {k: v for k, v in status_map.items() if isinstance(k, int)}
    )

    # The window is POSITIONED by n_terminal (see below), which only the
    # compact map can supply. That count now rides fetch_statuses' 5 s TTL, so
    # a task completing inside that window can leave the offset one row short
    # and hold the newest done row out of the list for up to 5 s — bounded,
    # self-correcting on the next miss, and far inside the 20 s staleness the
    # terminal rows already carry from fetch_tasks' own cache. It cuts the
    # other way too: a stable n_terminal means a stable offset, so the
    # terminal fetch's cache key stops churning on every completion. Without it the offset collapses to 0, and since
    # page_size/offset slice an ASCENDING-id list, offset 0 selects the OLDEST
    # terminal rows — which are then sorted by updated_at desc and emitted as
    # the tab's "most recent" done list. Showing months-old rows as the newest
    # is a worse failure than showing none, so an unpositionable window is not
    # fetched at all.
    wants_terminal = (
        (max_done_per_project > 0 or max_cancelled_per_project > 0)
        and not map_offline
    )
    if wants_terminal:
        n_terminal = sum(1 for s in status_map.values() if s in _TERMINAL_STATUSES)
        window = _TERMINAL_FETCH_WINDOW
        if n_terminal > window:
            logger.warning(
                'project %s: %d terminal (done+cancelled) tasks exceed the '
                '%d-row fetch window — only the %d highest-id terminal rows '
                'are fetched, so a low-id task completed long after it was '
                'filed can be missing from the Tasks tab',
                project, n_terminal, window, window,
            )
        # page_size/offset slice a list ordered by ASCENDING id, so reaching
        # the high-id end requires a computed offset rather than a LIMIT.
        terminal = await fetch_tasks(
            client, config, project_root,
            statuses=sorted(_TERMINAL_STATUSES),
            page_size=window,
            offset=max(0, n_terminal - window),
        )
        if isinstance(terminal, list):
            # DEDUP, not concatenate. The two fetches are separate cached
            # reads, so a task that completed between them appears in BOTH:
            # once from the active read, once from the terminal read. Emitting
            # both yields two rows sharing one _task_uid — the id the React
            # tab uses as a map key and as its selection identity — so the
            # task renders twice, as pending AND as done.
            #
            # This is not a narrow race. fetch_tasks caches per (root,
            # narrowing) for the TTL, and the terminal key embeds an offset
            # that changes on EVERY completion — so a completion mints a fresh
            # terminal key (cold, sees 'done') while the active key is still
            # served from an entry up to a full TTL old (still 'pending').
            # Every completion would duplicate a row for up to the TTL window.
            # The pre-narrowing single fetch made this structurally impossible;
            # splitting the read is what introduced it.
            #
            # The terminal snapshot WINS: it is the newer of the two reads by
            # exactly the reasoning above, so its status is the more current.
            merged = {t.get('id'): t for t in tasks}
            merged.update({t.get('id'): t for t in terminal})
            tasks = list(merged.values())
        else:
            logger.warning(
                'project %s: terminal-window fetch failed (%s) — done and '
                'cancelled rows are omitted from this render',
                project, terminal.get('error') if isinstance(terminal, dict) else terminal,
            )

    if map_offline:
        # Degrade honestly rather than declaring an otherwise-healthy project
        # offline: the active fetch succeeded, so its rows are still good.
        # The count, though, loses its ONLY authoritative source. Counting the
        # fetched rows instead would now be a fabricated zero — the terminal
        # window was skipped just above, so no done row was fetched at all —
        # so the count is reported as UNKNOWN (None) and the caller omits the
        # project from DONE_COUNTS, the same not-zero-and-not-offline channel
        # the budget-degraded path uses.
        done_count = None
        logger.warning(
            'project %s: compact status map unavailable — done_count is '
            'UNKNOWN for this render (not zero, and not offline), and the '
            'terminal window was skipped because it cannot be positioned '
            'without the map, so no done/cancelled row is emitted',
            project,
        )
    else:
        done_count = sum(1 for s in status_map.values() if s == 'done')

    if not tasks:
        return [], False, done_count

    is_runtime_offline = runtime is None or runtime.offline
    runtime_index: dict[int, TaskRuntimeEntry] = (
        {e.task_id: e for e in runtime.tasks} if runtime is not None and not runtime.offline else {}
    )

    # Lookup table for dep title/status resolution within the same project.
    by_id: dict[int, dict] = {t['id']: t for t in tasks if isinstance(t.get('id'), int)}

    active: list[dict] = []

    for task in tasks:
        status = task.get('status')
        if status not in _ACTIVE_STATUSES:
            continue

        task_id = task['id']
        rt = _runtime_fields(runtime_index, is_runtime_offline, task_id, now=effective_now)

        uid = _task_uid(project, task_id)
        row = _build_task_row(project, task, task_id, rt, uid, now=effective_now)
        # active rows: started from the runtime entry; deps from task tree.
        row['started'] = rt['started']
        row['deps'] = _resolve_deps(task, by_id, project, status_map=status_map)
        active.append(row)

    # PRDs with at least one member still in an active status. Done/cancelled
    # members of these "live" PRDs are exempt from the terminal-bucket cap —
    # see Contract: task-row prd field in plans/dashboard-taskgraph-legibility-prd.md.
    live_prds = {row['prd'] for row in active if row.get('prd')}

    # Bounded terminal buckets: iterate over (status, cap) pairs. Rows within
    # the top-N cap keep the original done/cancelled shape; done/cancelled
    # members of a still-live PRD are additionally emitted beyond the cap
    # (with populated deps) per the live-PRD terminal-member exemption above.
    for _bkt_status, _bkt_cap in (
        ('done', max_done_per_project),
        ('cancelled', max_cancelled_per_project),
    ):
        if _bkt_cap <= 0:
            continue
        bucket_tasks = [t for t in tasks if t.get('status') == _bkt_status]
        # Sort by updated_at descending; id descending as tie-breaker.
        bucket_tasks.sort(
            key=lambda t: (t.get('updated_at') or '', t.get('id') or 0),
            reverse=True,
        )
        capped_ids = {t['id'] for t in bucket_tasks[:_bkt_cap]}
        exempted_count = 0
        for task in bucket_tasks:
            task_id = task['id']
            prd = _coalesce_prd(task.get('metadata') or {})
            is_live_member = prd is not None and prd in live_prds
            beyond_cap = task_id not in capped_ids
            if beyond_cap and not is_live_member:
                continue
            if beyond_cap:
                exempted_count += 1
            uid = _task_uid(project, task_id)
            rt = _runtime_fields(runtime_index, is_runtime_offline, task_id, now=effective_now)
            row = _build_task_row(project, task, task_id, rt, uid, prd=prd, now=effective_now)
            # terminal rows: no meaningful start time; deps only for live-PRD
            # members (the terminal-member exemption), else unsurfaced.
            row['started'] = 0
            row['deps'] = (
                _resolve_deps(task, by_id, project, status_map=status_map)
                if is_live_member else []
            )
            row['completed'] = task.get('updated_at') or ''
            active.append(row)
        if exempted_count > _LIVE_PRD_EXEMPTION_WARN_THRESHOLD:
            logger.warning(
                'project %s: live-PRD exemption emitted %d %s rows beyond the '
                'cap (max=%d) — a PRD may have an unusually large number of '
                'live terminal members',
                project, exempted_count, _bkt_status, _bkt_cap,
            )

    return active, False, done_count


async def collect_tasks_with_counts(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    *,
    max_done_per_project: int = 0,
    max_cancelled_per_project: int = 0,
    resolve_external: bool = False,
    now: datetime | None = None,
) -> tuple[list[dict], list[str], dict[str, int], list[str], list[str]]:
    """Aggregate active tasks and per-project done counts in a single MCP pass.

    Returns ``(active_tasks, offline_projects, done_counts,
    degraded_projects, count_unknown_projects)``
    where:

    - *active_tasks* is the list of active (and optionally bounded done) rows
    - *offline_projects* lists project labels whose MCP fetch failed
    - *done_counts* maps project label → total done task count (pre-cap)
    - *degraded_projects* lists project labels the budget did not deliver

    **Bounded as a whole, not merely per call.**  The walk over project roots
    is sequential, so without a deadline this function's worst case is the SUM
    of every project's worst case — unbounded in the number of configured
    roots, and behind a browser ``fetch`` that aborts at 30 s.  A
    ``loop.time()`` deadline (``_TASKS_TOTAL_BUDGET``) is taken up front and
    each project is run under ``asyncio.wait_for`` at
    ``min(remaining, _TASKS_PER_PROJECT_BUDGET)``, copying ``app.healthz``'s
    loop shape rather than inventing one.

    Expiry yields a PARTIAL payload with explicit per-project markers, never a
    truncated-but-confident one: every project that timed out or never got its
    turn is named in *degraded_projects*, and neither contributes a
    *done_counts* entry (no count was measured, so none is fabricated — not
    even a ``0``, which renders as a real "this project has zero done tasks").

    *degraded* and *offline* are DISTINCT FACTS and must never be merged by a
    consumer: *offline* means the fetch demonstrably failed (the project is
    proven unreachable), *degraded* means the budget expired first and this
    project's state is simply UNKNOWN.  Collapsing them tells an operator that
    fused-memory is down when the only thing that happened is that the handler
    ran out of time — sending them to restart a healthy service.

    When *resolve_external* is ``True``, gathers the deduped union of every
    row's ``external_deps`` ids, issues **one** batched
    ``fetch_external_statuses`` call (skipped when the union is empty), and
    overwrites each entry's status.  Deps absent from the map keep the honest
    ``'unknown'`` sentinel.  Defaults to ``False`` so the scheduler-page path
    (``collect_active_tasks``) issues no extra MCP round-trip.

    That call runs AFTER the per-project walk and is inside the same deadline:
    it is both CHECKED (skipped outright once the budget is spent) and BOUNDED
    (run under ``asyncio.wait_for`` on whatever remains).  Checked-but-unbounded
    is not enough — a small positive remainder would still admit a call that
    then took ``mcp_tool_call``'s own default — and "bounded as a whole" has to
    mean the whole, or the claim is false for the last leg of the handler.

    *now* is resolved ONCE (via :func:`dashboard.data.utils.resolve_now`) at
    this aggregation boundary and threaded into every project's
    ``_shape_one_project`` call, so every returned row's ``started`` shares
    the same instant regardless of which project it came from.

    Per-task runtime state (loops/attempts/started/agent/lane/phase/
    lane_state) is likewise fetched ONCE here — a single concurrent fan-out
    via :func:`dashboard.data.task_runtime.fetch_task_runtime` over
    ``config.escalation_urls`` — and each project's snapshot is threaded into
    its ``_shape_one_project`` call, mirroring the single-``now`` threading.

    Prefer this over calling ``collect_active_tasks`` and
    ``collect_done_counts`` concurrently: it still avoids a redundant
    per-project fan-out (one walk of the roots, one shared *now*, one shared
    runtime snapshot) rather than two independent ones.

    It no longer HALVES the round-trips, and DONE_COUNTS is no longer the same
    snapshot as the rows — both claims stood before the narrowing and neither
    survives it, so do not rely on either:

    * per project this issues the 2-3 calls enumerated in
      ``_PER_PROJECT_MCP_CALLS``, not half of what the two collectors cost;
    * DONE_COUNTS comes from the compact ``fetch_statuses`` map while the rows
      come from ``fetch_tasks``. Both are cached, at 5 s and 20 s
      respectively, so the count can be up to ~15 s NEWER than the rows it
      sits beside. The skew is one-directional by construction (the count is
      never the staler half) and is the same skew ``fetch_tasks``' own
      "Data consistency" note documents.
    """
    effective_now = resolve_now(now)
    # The deadline is taken BEFORE the runtime fan-out, so that fan-out is
    # inside the budget too rather than being free time the projects then pay
    # for. Same shape as app.healthz's whole-handler deadline.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _TASKS_TOTAL_BUDGET
    runtime_by_label = await fetch_task_runtime(client, config.escalation_urls)
    all_active: list[dict] = []
    offline_projects: list[str] = []
    done_counts: dict[str, int] = {}
    degraded_projects: list[str] = []
    # Roots whose ACTIVE rows loaded fine but whose compact status map did
    # not, so done_count is UNKNOWN and the terminal window was skipped.
    # These are NOT offline (their rows are good and current) and NOT
    # degraded (nothing timed out), so without a list of their own they
    # would appear in no marker at all — and the front end would render
    # them as a healthy project with a confident "0 done". That is the
    # invisible-failure class this task exists to close.
    count_unknown_projects: list[str] = []
    for root in _all_project_roots(config):
        label = _project_label(root)
        remaining = deadline - loop.time()
        if remaining <= 0:
            # Never got its turn. A silently missing project reads as "no
            # active work" on the Tasks tab, which is the same class of
            # invisible failure the fan-out logging policy was raised to
            # WARNING to close.
            degraded_projects.append(label)
            logger.warning(
                'project %s: skipped — the %.1fs Tasks budget was already '
                'spent before this project was reached; its rows and done '
                'count are UNKNOWN for this render (not zero, and not offline)',
                label, _TASKS_TOTAL_BUDGET,
            )
            continue
        try:
            active, offline, done_count = await asyncio.wait_for(
                _shape_one_project(
                    client, config, root,
                    max_done_per_project=max_done_per_project,
                    max_cancelled_per_project=max_cancelled_per_project,
                    now=effective_now,
                    runtime=runtime_by_label.get(label),
                ),
                timeout=min(remaining, _TASKS_PER_PROJECT_BUDGET),
            )
        except TimeoutError:
            degraded_projects.append(label)
            logger.warning(
                'project %s: exceeded its %.1fs share of the %.1fs Tasks '
                'budget (%.1fs remained) — its rows and done count are '
                'UNKNOWN for this render (not zero, and not offline)',
                label, _TASKS_PER_PROJECT_BUDGET, _TASKS_TOTAL_BUDGET, remaining,
            )
            continue
        except Exception:
            # DEFENSE IN DEPTH, and deliberately broad. The fan-out normally
            # converts a failed read into the offline marker, so nothing here
            # is a demonstrated crash — but without this clause ANY unexpected
            # exception (a decode error, a shaping bug, an httpx transport
            # error that escaped the fan-out) unwinds the whole loop and 500s
            # the handler, throwing away every healthy project already
            # collected. That is the same "one bad root blanks the whole tab"
            # failure TASKS_OFFLINE exists to close, relocated from the banner
            # to the handler, and one root must not be able to cause it.
            #
            # OFFLINE, not degraded: the read demonstrably FAILED, which is
            # what offline means. degraded is reserved for "the budget never
            # let us find out" — the distinction the two branches above draw,
            # and merging them here would undo it.
            #
            # exc_info is load-bearing: an exception absorbed into a routine
            # offline marker with no traceback is a bug that renders as an
            # outage forever. The log is what separates "fused-memory is down"
            # from "our own shaping code raised".
            offline_projects.append(label)
            logger.warning(
                'project %s: unexpected error while shaping its rows — the '
                'project is marked offline for this render so the remaining '
                'roots still render; this is a BUG, not an outage',
                label, exc_info=True,
            )
            continue
        if offline:
            offline_projects.append(label)
        elif done_count is not None:
            done_counts[label] = done_count
        else:
            # done_count is None => the compact status map read failed for an
            # otherwise-healthy project. Omitting the label from done_counts
            # keeps a fabricated 0 off the wire, but omission ALONE is not
            # enough: the client's fallback counts the done rows it received,
            # and the terminal window was deliberately skipped for exactly
            # these projects, so that fallback is always 0 and renders as a
            # confident "0 done". Naming the root here is what lets the
            # banner and the header say UNKNOWN instead.
            count_unknown_projects.append(label)
        all_active.extend(active)

    if resolve_external:
        # Gather the deduped union of external dep ids for ACTIVE (non-done) rows only.
        # Done rows' external deps are no longer actionable; skipping them avoids
        # needless MCP load and prevents 'External dependencies' chips appearing on
        # completed tasks in the dashboard (where they would be noise, not signal).
        dep_ids: set[str] = set()
        for row in all_active:
            if 'completed' in row:
                continue  # skip bounded done rows
            for entry in row.get('external_deps') or []:
                dep_ids.add(entry['id'])
        # Same deadline treatment as the per-project loop: this call runs AFTER
        # it, so without a check it would overrun the budget the loop just
        # honoured. Skipping leaves every entry on its existing 'unknown'
        # sentinel, which is the honest value for a status never read.
        ext_remaining = deadline - loop.time()
        if dep_ids and ext_remaining <= 0:
            logger.warning(
                'external dep statuses skipped for %d id(s) — the %.1fs Tasks '
                'budget was spent by the per-project walk; every external dep '
                "keeps its 'unknown' sentinel for this render",
                len(dep_ids), _TASKS_TOTAL_BUDGET,
            )
        elif dep_ids:
            # BOUNDED, not merely deadline-checked. The check above only
            # decides whether to start; without this wait_for the call itself
            # ran on mcp_tool_call's 10s-per-request default and — a cold
            # session being three posts, per fan-out URL — could overrun the
            # whole-handler budget by ~30s and blow past data.js's 30 000 ms
            # fetch abort, throwing away the very partial payload the deadline
            # exists to deliver. Same two-layer shape as the per-project loop.
            try:
                status_map = await asyncio.wait_for(
                    fetch_external_statuses(client, config, sorted(dep_ids)),
                    timeout=ext_remaining,
                )
            except TimeoutError:
                # Every entry keeps its 'unknown' sentinel — identical to the
                # skip branch above, and for the identical reason: a status
                # that was never read has no honest value but 'unknown'.
                logger.warning(
                    'external dep statuses for %d id(s) exceeded the %.1fs '
                    'remaining of the %.1fs Tasks budget — every external dep '
                    "keeps its 'unknown' sentinel for this render",
                    len(dep_ids), ext_remaining, _TASKS_TOTAL_BUDGET,
                )
                status_map = {}
            map_offline = bool(status_map.get('offline'))
            for row in all_active:
                if 'completed' in row:
                    continue  # skip bounded done rows
                for entry in row.get('external_deps') or []:
                    if map_offline:
                        entry['status'] = 'offline'
                    else:
                        entry['status'] = status_map.get(entry['id'], 'unknown')

    return (
        all_active, offline_projects, done_counts,
        degraded_projects, count_unknown_projects,
    )


async def collect_active_tasks(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    *,
    max_done_per_project: int = 0,
    max_cancelled_per_project: int = 0,
    now: datetime | None = None,
) -> tuple[list[dict], list[str]]:
    """Collect active tasks across all known projects.

    Returns ``(active_tasks, offline_projects)`` where *offline_projects* is
    the list of project labels whose MCP fetch failed.  The handler turns a
    non-empty *offline_projects* into ``offline: True`` on the dashboard payload.

    When *max_done_per_project* > 0, the most-recent N done tasks per project
    are appended to the returned list (each with a ``completed`` field).
    Default 0 leaves the return shape unchanged — scheduler.py is unaffected.

    *now* is forwarded to ``collect_tasks_with_counts`` so every row's
    ``started`` derives from a single shared instant; see that function's
    docstring for details. Defaults to the live clock.

    Lock state is surfaced via the scheduler endpoint — see
    /api/v2/dashboard/scheduler.

    Note: callers that also need per-project done counts should use
    ``collect_tasks_with_counts`` to avoid a second MCP round-trip.

    The whole-handler budget applies here too, but its *degraded_projects*
    marker is absorbed rather than forwarded: this narrower two-element
    contract has nowhere to put it, and a degraded project is emphatically NOT
    offline, so reclassifying it into *offline_projects* would be a lie.  It is
    still logged at WARNING by ``collect_tasks_with_counts``.  Callers that
    need to distinguish "unknown" from "reachable and empty" must use
    ``collect_tasks_with_counts`` directly.
    """
    active, offline, _, _, _ = await collect_tasks_with_counts(
        client, config,
        max_done_per_project=max_done_per_project,
        max_cancelled_per_project=max_cancelled_per_project,
        now=now,
    )
    return active, offline


async def collect_done_counts(
    client: httpx.AsyncClient,
    config: DashboardConfig,
) -> dict[str, int]:
    """Return a ``{project_label: done_count}`` map for all reachable projects.

    Uses the compact ``fetch_statuses`` because only a per-status count is
    needed here.  NOTE: this is no longer the burndown collector's source —
    that switched to ``fetch_tasks`` (task 3543) because the compact map
    carries no claimant columns and so cannot express the live/stranded
    split.  The two agree on the ``done`` count (both ultimately read the same
    task store), but they are separate reads at separate instants, so a task
    completing between them can show a transient off-by-one.

    All projects are fetched concurrently to minimise latency.  Projects whose
    ``fetch_statuses`` returns an offline marker are silently omitted.
    """
    roots = _all_project_roots(config)
    results = await asyncio.gather(
        *(fetch_statuses(client, config, r) for r in roots)
    )
    counts: dict[str, int] = {}
    for root, result in zip(roots, results, strict=False):
        # Skip offline markers (dict with 'offline' key).
        if isinstance(result, dict) and result.get('offline'):
            continue
        label = _project_label(root)
        counts[label] = sum(1 for s in result.values() if s == 'done')
    return counts
