"""Aggregate active tasks across all known projects for the redux dashboard.

Joins three sources — the per-root task snapshot unit
(``dashboard.data.task_snapshot``, which owns every read of the task tree),
per-task runtime state (via the orchestrator's escalation MCP,
``get_task_runtime_state``), and optional burst state from reconciliation —
into the ``ACTIVE_TASKS`` shape consumed by the React dashboard's tasks tab.

THIS MODULE READS NOTHING ITSELF beyond the runtime fan-out and the batched
external-status tail. A root's task state arrives as a ``TaskSnapshot``,
already stamped with the instant it was measured and already carrying, per
half, whether that measurement succeeded and why not. What lives here is the
WALK — the budget, the admission width, the rotation, the canonical
re-assembly — and the SHAPING of one root's rows out of one unit.

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
        'started': 14,              # minutes since TaskRuntimeEntry.started (runtime snapshot);
                                    # None when the entry's own started is None
                                    # (per-task read failure)
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
        'runtime_status': 'ok',     # WHY runtime_offline is what it is — see RuntimeStatus.
                                     # runtime_offline alone cannot tell an operator whether the
                                     # orchestrator is down ('unreachable'), the dashboard was too
                                     # starved to ask within its own probe budget
                                     # ('deadline_exceeded'), or no orchestrator is configured for
                                     # this root at all ('not_configured') — three cases that
                                     # demand opposite responses but render identically as blank
                                     # cells. Collapsing them is what made the 2026-07-30 event get
                                     # misdiagnosed as an orchestrator outage. runtime_offline is
                                     # UNCHANGED: True for every non-'ok' member.
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
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx
from shared.task_runtime_state import TaskRuntimeEntry, TaskRuntimeSnapshot
from shared.timestamps import parse_timestamp_or_warn

from dashboard.config import DashboardConfig
from dashboard.data.task_runtime import fetch_task_runtime
from dashboard.data.task_snapshot import (
    SnapshotFailure,
    SnapshotHealth,
    TaskSnapshot,
    acquire_snapshot,
    classify,
    unmeasured_snapshot,
)
from dashboard.data.tasks import fetch_external_statuses, task_is_stranded
from dashboard.data.utils import resolve_now

logger = logging.getLogger(__name__)

# Why a task row's runtime fields are what they are — the FAULT DOMAIN of the
# runtime probe, not the task (task 3517). Mirrors the ``status``-discriminator
# convention already used by ``app._probe_db`` and ``redux_api._shape_wal_status``.
#
# - 'ok'                the probe succeeded. Fields are real (or honest zeros
#                       for a task absent from the snapshot, or honest Nones on
#                       a PER-TASK read failure — the probe was still fine).
# - 'not_configured'    no escalation URL for this project root, so nothing was
#                       ever probed. Expected and permanent; NOT a fault.
# - 'unreachable'       connect refused / HTTP error / malformed payload. The
#                       ORCHESTRATOR is the fault domain; go look at it.
# - 'deadline_exceeded' the probe's own budget fired. Likely the DASHBOARD's
#                       fault under a starved event loop — the orchestrator may
#                       be perfectly healthy (2026-07-30).
# - 'unknown'           the snapshot said offline with no reason. Out-of-contract
#                       for a dashboard-synthesized snapshot; the honest sentinel.
#                       NEVER fabricate a diagnosis to fill this in.
RuntimeStatus = Literal[
    'ok', 'not_configured', 'unreachable', 'deadline_exceeded', 'unknown',
]

# --- Budget constants -------------------------------------------------------
#
# Written in the ``app._HEALTHZ_TOTAL_BUDGET`` / ``app._DB_PROBE_TIMEOUT``
# idiom: a per-unit budget, a whole-handler budget, and an introspectable
# roster of the units, so ``tests/test_tasks_budget.py`` can machine-check
# that the parts fit the whole instead of a human re-deriving the arithmetic
# every time one of them moves.
#
# The per-unit half of that pair lives in ``task_snapshot`` now, with the
# reads it bounds: ``PER_CALL_TIMEOUT`` and the ``PER_PROJECT_MCP_CALLS``
# roster. A per-read budget left behind in a module that no longer issues the
# read would be a second copy waiting to drift.

# Whole-operation bound for ONE project root, enforced by ``asyncio.wait_for``
# in ``collect_tasks_with_counts``.
#
# ``task_snapshot.PER_CALL_TIMEOUT`` (4.4) * 3 bounded operations = 13.2 <=
# 14.0, leaving 0.8 s of slack so this deadline is a real backstop for non-MCP
# overhead (JSON decode, row shaping, event-loop scheduling) rather than
# coinciding exactly with the sum of its parts — the same reasoning as
# healthz's ``_DB_PROBE_TIMEOUT * 3 = 2.7 <= _HEALTHZ_TOTAL_BUDGET = 3.0``.
#
# It moved 7.0 -> 14.0 only because ``PER_CALL_TIMEOUT`` moved 2.0 -> 4.4: the
# parts-fit-the-whole shape is unchanged and the slack is still named. It is
# NOT an independent widening, and must not be raised on its own.
#
# What that sum does and does NOT claim: it bounds the sum of the per-OPERATION
# budgets. It does NOT bound a cold MCP session, which performs three posts
# (initialize, notifications/initialized, tools/call) and so can reach
# ``3 * PER_CALL_TIMEOUT`` for a SINGLE tool call — the TAB'S per-call term,
# not the shared default, since that is what every call below threads. That
# residual is exactly what this ``wait_for`` layer exists to cap: the two
# layers are complementary, not redundant (the same two-layer note
# ``dashboard/src/dashboard/data/task_runtime.py``'s module docstring carries).
_TASKS_PER_PROJECT_BUDGET = 14.0

# Whole-handler deadline for the entire multi-project aggregation.
#
# Strictly below ``data.js``'s 30 000 ms fetch abort with 10 s of headroom for
# HTTP and JSON serialisation, so the PARTIAL payload the deadline produces is
# actually deliverable to the browser that asked for it. It replaces a
# structural worst case of roughly ``roots * 3 posts * 10 s`` with no cap at
# all.
#
# DELIBERATELY UNCHANGED at 20.0 while ``_TASKS_PER_PROJECT_BUDGET`` doubled.
# The obvious "fix" for a cold render that runs out of budget is to widen this
# toward ``data.js``'s 30 000 ms abort, and it is the wrong one: the measured
# payload is ~14 MB, and the 10 s of remaining headroom is what serialises and
# ships it. Spend that headroom on more fetching and the handler delivers a
# payload the browser has already aborted — a 15s-handler-behind-a-5s-caller,
# which is the exact failure ``test_healthz_deadline.py`` exists to prevent
# and strictly worse than degrading a root. The root-count problem is solved
# by CONCURRENCY (``_TASKS_ROOT_CONCURRENCY``) and the fairness problem by
# ROTATION, neither of which costs headroom.
#
# Raising any one of these constants requires re-checking the others — they
# are mutually constrained, and ``test_tasks_budget.py`` enforces that.
# None of them may be raised toward ``memory.mcp_tool_call``'s 10 s default.
#
# AFTER-STATE, measured 2026-09-07 (task 4884 step-20), same method as the
# before-numbers above: caches cleared, then a cold ``GET
# /api/v2/dashboard/tasks``, 10 repetitions, 9 configured roots, live
# fused-memory. RAW NUMBERS, because this is the capacity baseline the next
# change reads:
#
#   before (per-call 2.0, sequential, fixed order): 208 of 3045 active rows;
#     dark-factory, reify AND autopilot-video all marked OFFLINE.
#   after:  rows 295 / 1167 / 1562 / 1689 / 1776 / 2680 / 2806 / 2857 /
#           2924 / 3043; wall 9.8–22.4 s; payload 0.9–14.3 MB.
#
# HONEST VERDICT, both halves. FAIRNESS (rotation) holds: the degraded/offline
# set differed on every one of the 10 renders, against the journal's fixed
# trailing pair (solar-challenge-platform, pump-web-ui) every render.
# COMPLETENESS does NOT hold unconditionally: 3 of 10 renders came back fully
# clean (0 offline, 0 degraded, 2806–3043 rows in 9.8–13.4 s), the other 7
# degraded or offlined between one and six roots. So a cold render CAN now
# serve every root inside the budget, but is not guaranteed to.
#
# Deliberately NOT closed by raising this constant. Doing so would buy the
# clean render by making failure impossible, which is exactly what #4795
# acceptance 3 forbids and what ``test_tasks_budget.py`` (c) blocks — and it
# would spend the serialisation headroom the 14.3 MB payload measured above
# actually needs. The residual is the un-projected ``SELECT *`` read (task
# 4390); until that lands the honest markers are the answer, not a wider bound.
#
# CONFOUND, stated so the numbers are not over-read: these were taken with the
# production dashboard also polling the SAME single fused-memory server every
# 3 s, so they include real contention and are a pessimistic bound, not a
# quiet-system best case.
_TASKS_TOTAL_BUDGET = 20.0

# How many project roots ``collect_tasks_with_counts`` may have in flight.
#
# WHY > 1: the walk used to be SEQUENTIAL, so N roots cost the SUM of their
# per-root costs against one ``_TASKS_TOTAL_BUDGET``. At the incident's 9
# roots that sum exceeded the total, and because the walk order was fixed the
# SAME trailing roots were reported degraded on every render (the journal's
# repeated ``project pump-web-ui: skipped — the 20.0s Tasks budget was
# already spent``). At width W the worst case becomes roughly
# ``ceil(N / W) * _TASKS_PER_PROJECT_BUDGET``.
#
# WHY BOUNDED rather than a plain unbounded ``gather``: ``burndown.py``
# records the measurement (see ``_SNAPSHOT_PAGE_SIZE`` and
# ``_fetch_snapshot_tasks``). A full fan-out over every root goes against a
# SINGLE fused-memory server where the requests serialise SERVER-side anyway,
# on the SAME shared httpx client the 3 s render polls use — "a live
# httpx.PoolTimeout risk for request-path handlers". Unbounded concurrency
# would therefore buy this endpoint nothing (the server is the bottleneck)
# while spending every other endpoint's connections.
#
# NOTE that ``app._build_http_limits`` scales ``max_connections`` with the
# fleet size, which makes the POOL look like the constraint it is not. Raising
# this width because the pool grew would be reading the wrong number: the
# ceiling here is the single MCP server's own serialisation, not connections.
#
# 4 covers the measured shape of the fan-out — two big roots (dark-factory
# 5128 tasks, reify 7279) and seven that finish in under 0.1 s — in
# ``ceil(9/4) = 3`` waves. ``test_tasks_budget.py`` (f) holds it in (1, 8].
_TASKS_ROOT_CONCURRENCY = 4

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


# Admission-order rotation offset for `collect_tasks_with_counts`, advanced by
# one slot per render. Module state, like the `_*_cache` objects elsewhere in
# this package, with a matching `_reset_root_rotation()` test hook.
_root_rotation_offset: int = 0


def _reset_root_rotation() -> None:
    """Reset the admission-order rotation. Test hook.

    Same shape as the ``_*_cache_clear`` hooks in ``tasks.py``: rotation is
    module state, so a test that asserts an ORDER has to be able to start from
    a known offset rather than inherit whatever the previous test left.
    """
    global _root_rotation_offset
    _root_rotation_offset = 0


def _rotated_project_roots(config: DashboardConfig) -> list[Path]:
    """``_all_project_roots`` rotated left by one more slot on each call.

    WHAT THIS CLOSES. `collect_tasks_with_counts` cannot always serve every
    root inside `_TASKS_TOTAL_BUDGET`, and with a FIXED walk order the roots
    that lose are always the same ones — the last ones. The journal of the
    2026-08-27 incident shows exactly that: `project solar-challenge-platform:
    skipped — the 20.0s Tasks budget was already spent before this project was
    reached` and `project pump-web-ui: skipped ...`, the same trailing pair,
    render after render. Those two projects were effectively invisible on the
    Tasks tab for the duration.

    ROTATION MAKES STARVATION FAIR, NOT ABSENT. This is the load-bearing
    claim, and it is deliberately weaker than it looks: with 9 roots and a
    20.0 s budget some render will still fail to serve some roots. What
    rotation guarantees is that the starved SET rotates, so no root is
    permanently invisible. What reports the starvation is unchanged — the
    honest `TASKS_DEGRADED_PROJECTS` / `TASKS_OFFLINE_PROJECTS` markers task
    3857 built. Do not read this helper as a fix for degraded rows; read it as
    the reason a degraded row is transient rather than permanent.

    DETERMINISTIC ROUND-ROBIN, NOT RANDOMISATION. A shuffle would also spread
    the starvation, and was rejected: an operator comparing two consecutive
    renders can predict which roots were served under a round-robin and cannot
    under a shuffle, and a test can assert the former (see
    ``TestCollectTasksWithCountsFairness``) but only sample the latter.
    Reproducibility in an incident is worth more here than any property a
    random order would buy.

    SEPARATE HELPER, deliberately. ``_all_project_roots`` stays byte-identical
    and primary-first: ``app.py``, ``scheduler.py``, ``collect_done_counts``
    and ``test_app.py``'s patch point all depend on that ordering, so rotating
    in place would silently repoint every one of them at a different project.
    Only ``collect_tasks_with_counts``' admission loop calls this.

    Rotation changes ADMISSION order only. Output is re-assembled in canonical
    ``_all_project_roots`` order by the caller, so the rendered table does not
    reshuffle on every 3 s poll.
    """
    global _root_rotation_offset
    roots = _all_project_roots(config)
    if not roots:
        return roots
    offset = _root_rotation_offset % len(roots)
    _root_rotation_offset = (_root_rotation_offset + 1) % len(roots)
    return roots[offset:] + roots[:offset]


def _task_uid(project: str, task_id: int) -> str:
    """Project-scoped unique id used by the React tasks tab as a map key."""
    return f'{project}/T-{task_id}'


def _minutes_since(iso: str | None, *, now: datetime | None = None) -> int | None:
    """Whole minutes between *iso* and *now* (UTC).

    Returns ``None`` when *iso* is missing/empty — the start time is genuinely
    UNKNOWN (the per-task artifact-read-failure signal on
    ``TaskRuntimeEntry.started``), and fabricating ``0`` there would render as
    '0m running' during precisely the failure this path exists to surface
    (loud-over-silent-degradation; INV-2 structured-facts-at-failure).
    Returns ``None`` on parse failure too — a present-but-unparseable *iso*
    is upstream data damage (``TaskRuntimeEntry.started`` is typed
    ``str | None`` but pydantic does not validate ISO format), and rendering
    it as ``0`` would produce the identical misleading '0m running' the
    missing-input case above exists to avoid. Returns ``0`` for a genuine
    sub-minute-old start, and for a future timestamp (clamped by the
    ``max(minutes, 0)`` below) — a ``0`` in the payload does NOT imply parse
    failure or clock skew; only ``None`` signals an unknown start.

    *now* defaults to the live clock via :func:`dashboard.data.utils.resolve_now`;
    pass an explicit value for deterministic results or to share one instant
    across multiple rows in an aggregation.
    """
    if not iso:
        return None
    ts, ok = parse_timestamp_or_warn(iso, context='active_tasks._minutes_since')
    if not ok:
        # Logged once per call, not deduped/rate-limited: the docstring above
        # notes there is no known producer of a damaged `started` today, so
        # unbounded per-render volume is a theoretical concern, not an
        # observed one. Revisit (e.g. a per-value seen-set) if that changes.
        return None
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
    ``runtime_offline``/``runtime_status`` — ``started`` is handled separately by the caller,
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
        'runtime_status': rt.get('runtime_status', 'ok'),
        'claimant_run_id': task.get('claimant_run_id'),
        'heartbeat_at': task.get('heartbeat_at'),
        'stranded': task_is_stranded(task, now=now),
        'meta_files': meta_files,
        'train': train,
        'external_deps': external_deps,
        'prd': prd if prd is not None else _coalesce_prd(metadata),
    }


def _probe_status(runtime: TaskRuntimeSnapshot | None) -> RuntimeStatus:
    """Classify a project's runtime probe outcome — see :data:`RuntimeStatus`.

    ``None`` means the project label never appeared in the fan-out result at
    all, i.e. no escalation URL is configured for it and no probe was ever
    attempted — distinct from a probe that was attempted and failed.

    An ``offline=True`` snapshot with no ``offline_reason`` is out-of-contract
    for one the dashboard synthesized, so it degrades to ``'unknown'`` rather
    than being assigned a plausible-sounding reason we did not measure.
    """
    if runtime is None:
        return 'not_configured'
    if not runtime.offline:
        return 'ok'
    if runtime.offline_reason == 'deadline_exceeded':
        return 'deadline_exceeded'
    if runtime.offline_reason == 'unreachable':
        return 'unreachable'
    return 'unknown'


def _runtime_fields(
    index: dict[int, TaskRuntimeEntry],
    status: RuntimeStatus,
    task_id: int,
    *,
    now: datetime | None = None,
) -> dict:
    """Derive a task row's runtime-sourced fields from the project's runtime snapshot.

    *status* is this project's probe outcome from :func:`_probe_status`, and is
    emitted verbatim as the row's ``runtime_status``. ``runtime_offline`` is
    derived from it here as ``status != 'ok'`` — one source of truth — and keeps
    its EXACT prior meaning, so no downstream consumer's semantics shift.

    Three cases:

    - *status* is any non-``'ok'`` member (``'not_configured'``,
      ``'unreachable'``, ``'deadline_exceeded'``, ``'unknown'`` — we have no
      usable snapshot, whatever the cause): every field is ``None`` — never a
      fabricated ``0`` — and ``runtime_offline`` is ``True``.
    - *task_id* absent from *index* (the project IS online, the task just has
      no entry in its snapshot): honest zeros (``loops``/``attempts``/
      ``started`` are ``0``; ``agent``/``lane``/``phase``/``lane_state`` are
      ``None``) and ``runtime_offline`` is ``False``.
    - *task_id* present in *index*: real fields from the ``TaskRuntimeEntry``,
      which may themselves be ``None`` on a per-task artifact read failure —
      an honest per-task error, not an offline project, so ``runtime_offline``
      stays ``False`` and ``runtime_status`` stays ``'ok'`` either way (the
      PROBE succeeded; only this one task's artifact read did not).
    """
    if status != 'ok':
        return {
            'agent': None, 'loops': None, 'attempts': None, 'started': None,
            'lane': None, 'phase': None, 'lane_state': None,
            'runtime_offline': True, 'runtime_status': status,
        }
    entry = index.get(task_id)
    if entry is None:
        return {
            'agent': None, 'loops': 0, 'attempts': 0, 'started': 0,
            'lane': None, 'phase': None, 'lane_state': None,
            'runtime_offline': False, 'runtime_status': status,
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
        'runtime_status': status,
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
    to span the whole tree; the rows are now narrowed to
    ``shared.task_statuses.ACTIVE`` server-side, so a DONE dependency is never
    among them and would drop out of ``by_id`` and lose its chip entirely.

    Resolution order, most to least informative:

    1. a full row in *by_id* — real title, real status;
    2. otherwise, an id present in *status_map* — an honest PARTIAL entry:
       the ``done`` flag is authoritative, the title degrades to ``''``.
       BRANCH (2) IS THE COMMON CASE, not a rare fallback: it fires for EVERY
       terminal dependency, since none of them is in the narrowed rows, so
       most of an active task's done dependencies resolve titleless.
       ``tab_tasks.jsx`` therefore renders the id ALONE for these — the
       `` · `` separator is emitted only when a title exists, or the chips
       would read as ``3502 ·`` with nothing after it;

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


def _shape_one_project(
    project_root: Path,
    snapshot: TaskSnapshot,
    *,
    now: datetime,
    runtime: TaskRuntimeSnapshot | None = None,
) -> list[dict]:
    """Shape one project's ACTIVE rows out of an already-acquired *snapshot*.

    PURE given its arguments: it reads no clock, issues no MCP call and
    touches no cache. Acquiring the unit is I/O and lives in
    ``task_snapshot.acquire_snapshot``; turning it into rows is arithmetic and
    lives here. The split is what lets the shaping be exercised against a
    canned unit and the reads be exercised against a canned substrate, with
    neither test having to stand in for the other.

    It reads exactly two things off the unit:

    * ``snapshot.rows.value`` — the project's ACTIVE rows, or ``None`` when
      that half was never measured. ``None`` yields an EMPTY list, never a
      fabricated row: the unit's own ``Datum`` is what tells the consumer
      whether an empty table means "no active work" or "we could not look".
    * ``snapshot.status_map`` — the raw ``{id: status}`` map, which is
      ``_resolve_deps``' only bounded fallback for a dependency outside the
      fetched rows. The rows are narrowed to ``shared.task_statuses.ACTIVE``
      server-side, so a DONE dependency is never among them and this fallback
      is the common case rather than a rare one; skipping it would silently
      drop dependency chips, the exact regression that fallback was added to
      prevent.

    A STALE rows half is shaped exactly like a fresh one. The rows really were
    measured — just earlier — and the ``Datum`` carrying them says when; the
    alternative is blanking a project's table on one failed refresh, which is
    strictly less informative than showing what was last true.

    *now* is the caller's single resolved instant: every row's ``started`` and
    every ``stranded`` verdict share it, so two rows from different projects
    cannot disagree about what time it is.

    *runtime* is this project's ``TaskRuntimeSnapshot`` (resolved ONCE by the
    caller — see ``collect_tasks_with_counts`` — via ``fetch_task_runtime``).
    ``None`` (no escalation URL configured for this project) is treated
    identically to ``runtime.offline``: every row's runtime-sourced fields
    degrade to an honest ``None`` via :func:`_runtime_fields`, which is a
    DIFFERENT fault domain from the task-tree read the snapshot reports on.
    """
    project = _project_label(project_root)
    tasks = snapshot.rows.value or []
    if not tasks:
        return []

    runtime_status = _probe_status(runtime)
    runtime_index: dict[int, TaskRuntimeEntry] = (
        {e.task_id: e for e in runtime.tasks} if runtime is not None and not runtime.offline else {}
    )

    # Lookup table for dep title/status resolution within the same project.
    by_id: dict[int, dict] = {t['id']: t for t in tasks if isinstance(t.get('id'), int)}

    active: list[dict] = []
    for task in tasks:
        task_id = task.get('id')
        if not isinstance(task_id, int):
            continue
        rt = _runtime_fields(runtime_index, runtime_status, task_id, now=now)
        uid = _task_uid(project, task_id)
        row = _build_task_row(project, task, task_id, rt, uid, now=now)
        # active rows: started from the runtime entry; deps from task tree.
        row['started'] = rt['started']
        row['deps'] = _resolve_deps(task, by_id, project, status_map=snapshot.status_map)
        active.append(row)
    return active


def shape_terminal_rows(
    project_root: Path,
    rows: list[dict],
    *,
    now: datetime,
) -> list[dict]:
    """Shape a terminal window's raw rows into task rows for the wire.

    The sibling of :func:`_shape_one_project`, and here for the same reason:
    row shaping lives in this module so the two row shapes are built by one
    ``_build_task_row`` and cannot drift into two slightly different 20-key
    dicts. ``task_snapshot`` reads the window; this turns it into rows.

    PURE given its arguments, and deliberately thinner than the active
    shaping. Three fields differ, each because a terminal task has no live
    state to report:

    * ``started`` is ``0`` — a finished task has no elapsed runtime;
    * ``deps`` is empty — dependency chips answer "what is this waiting on",
      which is not a question about a finished task, and resolving them would
      need a status map this path never reads;
    * ``completed`` carries ``updated_at``, the only completion instant the
      substrate offers.

    No runtime probe is issued: :func:`_runtime_fields` under
    ``'not_configured'`` degrades every runtime-sourced field to an honest
    ``None`` rather than a fabricated zero. Probing for rows that by
    definition have no live agent would spend a budget to learn nothing.

    Rows are returned in the order the substrate served them (ascending id).
    Ordering for display is the consumer's, and the ``Datum``'s
    ``lower_bound`` state is what tells it the list is a window rather than a
    population.
    """
    project = _project_label(project_root)
    runtime_status = _probe_status(None)
    shaped: list[dict] = []
    for task in rows:
        task_id = task.get('id')
        if not isinstance(task_id, int):
            continue
        rt = _runtime_fields({}, runtime_status, task_id, now=now)
        row = _build_task_row(
            project, task, task_id, rt, _task_uid(project, task_id), now=now,
        )
        row['started'] = 0
        row['deps'] = []
        row['completed'] = task.get('updated_at') or ''
        shaped.append(row)
    return shaped


async def _acquire_and_shape(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    project_root: Path,
    *,
    now: datetime,
    runtime: TaskRuntimeSnapshot | None = None,
) -> tuple[list[dict], TaskSnapshot]:
    """ONE root's whole share of a render: acquire its unit, shape its rows.

    The unit the per-project budget bounds and the semaphore admits, named so
    the budget has something to point at. Both halves of the share are here
    because neither is meaningful without the other — rows with no snapshot
    have no provenance, and a snapshot with no rows renders nothing.
    """
    snapshot = await acquire_snapshot(client, config, project_root, now=now)
    rows = _shape_one_project(project_root, snapshot, now=now, runtime=runtime)
    return rows, snapshot


async def collect_tasks_with_counts(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    *,
    resolve_external: bool = False,
    now: datetime | None = None,
) -> tuple[list[dict], dict[str, TaskSnapshot]]:
    """Aggregate every root's active rows and its task snapshot in one pass.

    Returns ``(active_rows, snapshots_by_label)``, with an entry in the second
    for EVERY configured root — including the ones this render never reached,
    which carry a ``BUDGET``-kind unit rather than being absent.

    ONE ENTRY PER ROOT, not a row list beside three parallel lists of labels.
    The previous 5-tuple made "which projects are offline" and "which
    projects have a count" two separate answers that a caller had to keep in
    agreement by hand; here both are properties of the same record, and
    ``task_snapshot.classify`` is the single rule that routes one to a banner.
    A project therefore cannot be named offline while its census says it was
    measured, because there is only one thing to read.

    The entries are keyed by project LABEL, which is the directory basename,
    so two configured roots sharing one collapse to a single entry. That is
    pre-existing and deliberate: the label is what the wire and the front end
    address a project by, and the ROW assembly below is still keyed by root,
    so no root's rows are lost to the collision.

    **Bounded as a whole, not merely per call.**  A ``loop.time()`` deadline
    (``_TASKS_TOTAL_BUDGET``) is taken up front and each project is run under
    ``asyncio.wait_for`` at ``min(remaining, _TASKS_PER_PROJECT_BUDGET)``,
    copying ``app.healthz``'s loop shape rather than inventing one.

    **Concurrent at a bounded width.**  The walk used to be SEQUENTIAL, which
    made this function's worst case the SUM of every project's worst case —
    so at the 9 roots of the 2026-08-27 incident the total budget could not
    fit them all and the trailing roots degraded on every render.  Roots are
    now admitted through an ``asyncio.Semaphore(_TASKS_ROOT_CONCURRENCY)``, so
    the worst case is roughly ``ceil(roots / _TASKS_ROOT_CONCURRENCY) *
    _TASKS_PER_PROJECT_BUDGET``.  The deadline is still what BOUNDS it —
    concurrency changes the cost, not the guarantee.  The width is bounded
    rather than unbounded because the fan-out targets a single fused-memory
    server (where the requests serialise server-side regardless) over the
    shared httpx client the render polls use; see ``_TASKS_ROOT_CONCURRENCY``.

    Concurrency changes ADMISSION order only.  ``remaining`` is computed after
    a root acquires its slot — a root that waited for one pays for the wait
    rather than being handed a stale budget — and every result is collected
    into a per-root slot and re-assembled in ROOT order, so completion order
    can never reach the payload.

    Expiry yields a PARTIAL payload with explicit per-project provenance,
    never a truncated-but-confident one: a project that timed out or never got
    its turn still has an entry, and that entry's census is non-fresh rather
    than zero — no count was measured, so none is fabricated, and a ``0``
    would render as a real "this project has zero done tasks".

    Expiry is a fact about THIS handler, not about the server, which is why
    the unit carries a structured kind rather than an offline flag:
    ``SnapshotFailure.UNREACHABLE`` means the read demonstrably failed and
    ``BUDGET`` means the budget expired first, and ``task_snapshot.classify``
    is the only place they are turned into banners. Collapsing them tells an
    operator that fused-memory is down when the only thing that happened is
    that the handler ran out of time — sending them to restart a healthy
    service.

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
    this aggregation boundary and threaded into every project's share, so
    every returned row's ``started``, every ``Datum``'s ``as_of`` and every
    strand verdict share one instant regardless of which project produced it.

    Per-task runtime state (loops/attempts/started/agent/lane/phase/
    lane_state) is likewise fetched ONCE here — a single concurrent fan-out
    via :func:`dashboard.data.task_runtime.fetch_task_runtime` over
    ``config.escalation_urls`` — and each project's runtime snapshot is
    threaded into its share, mirroring the single-``now`` threading.

    Per root the DEFAULT render spends two of the three bounded operations in
    ``task_snapshot.PER_PROJECT_MCP_CALLS``; the third is reserved for the
    ``?terminal=`` request, which is the only one that asks for terminal rows.

    The count and the rows are two halves of ONE unit read under one TTL and
    stamped with one instant, so the skew between them is measured and
    reported (``snapshot.skew_seconds``) rather than implied by two caches at
    different TTLs. That skew used to be up to ~15 s and undisclosed.
    """
    effective_now = resolve_now(now)
    # The deadline is taken BEFORE the runtime fan-out, so that fan-out is
    # inside the budget too rather than being free time the projects then pay
    # for. Same shape as app.healthz's whole-handler deadline.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _TASKS_TOTAL_BUDGET
    runtime_by_label = await fetch_task_runtime(client, config.escalation_urls)
    all_active: list[dict] = []
    snapshots: dict[str, TaskSnapshot] = {}

    # ROTATED for admission, canonical for output. The rotation is what stops
    # the same trailing roots being starved on every render; see
    # _rotated_project_roots. `roots` below is the canonical order the results
    # are re-assembled in.
    roots = _all_project_roots(config)
    admission_order = _rotated_project_roots(config)
    # Admission control, not a work queue: the coroutines are all created up
    # front and the semaphore decides how many are inside _acquire_and_shape
    # at once. See _TASKS_ROOT_CONCURRENCY for why the width is bounded.
    slots = asyncio.Semaphore(_TASKS_ROOT_CONCURRENCY)

    def _unreached(
        root: Path, reason: str, failure: SnapshotFailure,
    ) -> tuple[list[dict], TaskSnapshot]:
        """No rows, plus the unit a root this render did not measure still owes."""
        return [], unmeasured_snapshot(
            root, now=effective_now, reason=reason, failure=failure,
        )

    async def _one(root: Path) -> tuple[list[dict], TaskSnapshot]:
        """Serve ONE root, returning its rows and its unit — never mutating shared state.

        Every branch returns a result instead of appending to the outer list.
        Appending from inside a concurrent coroutine would order the payload by
        COMPLETION, and the Tasks tab renders ``all_active`` directly, so the
        table would reshuffle on every 3 s poll. The caller re-assembles these
        results in ROOT order below.

        The result carries NO label: the caller pairs each one with the root
        that produced it, which is the only identity that is unique (see the
        re-assembly below).

        Every non-happy branch still returns a UNIT, never a marker flag: a
        root the budget never reached owes the wire an entry exactly as much
        as a healthy one, and its kind is what routes it to a banner. The
        reason is composed HERE, where the branch knows which budget expired
        and how much of it remained, and is then carried verbatim by both the
        WARNING and the unit — one sentence, one place, two readers.
        """
        label = _project_label(root)
        async with slots:
            # AFTER admission, deliberately: a root that queued for a slot has
            # already spent part of the whole-handler budget, and handing it a
            # `remaining` measured before the wait would let the walk overrun
            # the deadline by up to one wave.
            remaining = deadline - loop.time()
            if remaining <= 0:
                # Never got its turn. A silently missing project reads as "no
                # active work" on the Tasks tab, which is the same class of
                # invisible failure the fan-out logging policy was raised to
                # WARNING to close.
                reason = (
                    f'skipped — the {_TASKS_TOTAL_BUDGET:.1f}s Tasks budget was '
                    'already spent before this project was reached'
                )
                logger.warning(
                    'project %s: %s; its rows and census are UNKNOWN for this '
                    'render (not zero, and not offline)',
                    label, reason,
                )
                return _unreached(root, reason, SnapshotFailure.BUDGET)
            try:
                active, snapshot = await asyncio.wait_for(
                    _acquire_and_shape(
                        client, config, root,
                        now=effective_now,
                        runtime=runtime_by_label.get(label),
                    ),
                    timeout=min(remaining, _TASKS_PER_PROJECT_BUDGET),
                )
            except TimeoutError:
                reason = (
                    f'exceeded its {_TASKS_PER_PROJECT_BUDGET:.1f}s share of the '
                    f'{_TASKS_TOTAL_BUDGET:.1f}s Tasks budget '
                    f'({remaining:.1f}s remained)'
                )
                logger.warning(
                    'project %s: %s — its rows and census are UNKNOWN for this '
                    'render (not zero, and not offline)',
                    label, reason,
                )
                return _unreached(root, reason, SnapshotFailure.BUDGET)
            except Exception:
                # DEFENSE IN DEPTH, and deliberately broad. The fan-out
                # normally converts a failed read into the offline marker, so
                # nothing here is a demonstrated crash — but without this
                # clause ANY unexpected exception (a decode error, a shaping
                # bug, an httpx transport error that escaped the fan-out)
                # unwinds the whole GATHER and 500s the handler, throwing away
                # every healthy project. That is the same "one bad root blanks
                # the whole tab" failure TASKS_OFFLINE exists to close,
                # relocated from the banner to the handler, and one root must
                # not be able to cause it.
                #
                # It must stay INSIDE _one for that to hold: hoisted to the
                # gather (as return_exceptions=True) it would still catch the
                # exception, but only after asyncio.gather had already been
                # given the chance to propagate it, and the per-root offline/
                # degraded routing below would have nothing to key on.
                #
                # OFFLINE, not degraded: the read demonstrably FAILED, which is
                # what offline means. degraded is reserved for "the budget
                # never let us find out" — the distinction the two branches
                # above draw, and merging them here would undo it.
                #
                # exc_info is load-bearing: an exception absorbed into a
                # routine offline marker with no traceback is a bug that
                # renders as an outage forever. The log is what separates
                # "fused-memory is down" from "our own shaping code raised".
                logger.warning(
                    'project %s: unexpected error while shaping its rows — the '
                    'project is marked offline for this render so the remaining '
                    'roots still render; this is a BUG, not an outage',
                    label, exc_info=True,
                )
                return _unreached(
                    root,
                    'unexpected error while shaping this project (see the '
                    'WARNING and its traceback)',
                    SnapshotFailure.UNREACHABLE,
                )
        return active, snapshot

    # return_exceptions=False is correct here BECAUSE the broad `except
    # Exception` above lives INSIDE _one: nothing can escape to the gather, so
    # there is no exception for it to swallow, and a real escape (a bug in this
    # assembly code, a CancelledError) must still propagate rather than be
    # silently converted into a result object.
    results = await asyncio.gather(*(_one(root) for root in admission_order))
    # Keyed by ROOT, never by label. `_project_label` is the directory
    # BASENAME, so two configured roots can share one (``/a/proj`` and
    # ``/b/proj``) — and a label-keyed dict collapses them, which would extend
    # the survivor's rows into `all_active` TWICE (duplicate `_task_uid`s, the
    # React tab's map key) and drop the other root's rows entirely. Roots are
    # deduped by `_all_project_roots`, so this pairing is total and 1:1;
    # `strict=True` says so rather than trusting it. (`snapshots` below is
    # label-keyed because the wire addresses projects by label; the collision
    # costs an ENTRY there, never a row.)
    by_root = dict(zip(admission_order, results, strict=True))

    # CANONICAL ROOT order — neither completion order nor admission order.
    # This is the only place the shared accumulators are written.
    for root in roots:
        rows, snapshot = by_root[root]
        snapshots[_project_label(root)] = snapshot
        all_active.extend(rows)

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

    return all_active, snapshots


async def collect_active_tasks(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    *,
    now: datetime | None = None,
) -> tuple[list[dict], list[str]]:
    """Collect active tasks across all known projects.

    Returns ``(active_tasks, offline_projects)`` where *offline_projects* is
    the list of project labels whose read DEMONSTRABLY failed.  The handler
    turns a non-empty *offline_projects* into ``offline: True`` on the
    dashboard payload.

    The labels are DERIVED from each root's unit through
    ``task_snapshot.classify``, not carried alongside it: there is one place
    that decides what "offline" means, and this narrower contract reads it
    rather than restating it.

    *now* is forwarded to ``collect_tasks_with_counts`` so every row's
    ``started`` derives from a single shared instant; see that function's
    docstring for details. Defaults to the live clock.

    Lock state is surfaced via the scheduler endpoint — see
    /api/v2/dashboard/scheduler.

    The whole-handler budget applies here too, but a DEGRADED root is absorbed
    rather than forwarded: this narrower two-element contract has nowhere to
    put it, and a degraded project is emphatically NOT offline, so
    reclassifying it into *offline_projects* would be a lie.  It is still
    logged at WARNING by ``collect_tasks_with_counts``.  Callers that need to
    distinguish "unknown" from "reachable and empty" — or that want the census
    at all — must use ``collect_tasks_with_counts`` directly and read the
    units.
    """
    active, snapshots = await collect_tasks_with_counts(client, config, now=now)
    offline = [
        label for label, snapshot in snapshots.items()
        if classify(snapshot) is SnapshotHealth.OFFLINE
    ]
    return active, offline
