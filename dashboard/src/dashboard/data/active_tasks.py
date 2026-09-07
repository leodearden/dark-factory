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
from typing import Any, Literal

import httpx
from shared.task_runtime_state import TaskRuntimeEntry, TaskRuntimeSnapshot
from shared.timestamps import parse_timestamp_or_warn

from dashboard.config import DashboardConfig
from dashboard.data.task_runtime import fetch_task_runtime
from dashboard.data.tasks import (
    fetch_external_statuses,
    fetch_statuses,
    fetch_task_page,
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

# Per-HTTP-REQUEST budget for the Tasks tab's own MCP calls, threaded into
# every call ``_shape_one_project`` issues.
#
# MEASURED 2026-09-07 against the live fused-memory (localhost:8002), 9
# configured roots, caches cleared per root, per-call timeout temporarily
# raised to 10.0 so a slow root reported its real latency instead of a
# ReadTimeout: per-CALL max 2.696 s, per-ROOT wall max 2.876 s (dark-factory
# 5128 tasks, reify 7279), p95 across roots 1.695 s. Everything else was
# under 0.1 s — the distribution is two big trees and seven small ones, not a
# uniform cost.
#
# 4.4 = 1.5 * the 2.876 s per-root wall max, rounded up to one decimal.
#
# WHY 2.0 IS TOO SMALL, on evidence rather than on principle: at
# ``tasks.DEFAULT_PER_CALL_TIMEOUT`` the same measurement's truly-cold render
# marked dark-factory, reify AND autopilot-video OFFLINE and shipped 208 of
# 3045 active rows — i.e. the Tasks tab reported the three largest projects
# unreachable while fused-memory was serving them fine, which is what the
# journal's ``fetch_tasks[reify] failed for http://localhost:8002:
# ReadTimeout`` lines are. Offline is a claim that the read demonstrably
# failed; a per-call budget below the honest service time turns that claim
# into a lie on every cold render.
#
# WHY THE SERVER COST SCALES WITH TREE SIZE even though this read is
# status-narrowed: there is no field projection at any layer and the backend
# query is ``SELECT *`` feeding a fixed 14-key row — see ``fetch_tasks``'
# docstring, which records the limitation, and task 4390, which is the open
# follow-up to add projection. Until that lands, narrowing the STATUSES does
# not narrow the WORK, so a 5 000-task tree costs what a 5 000-task tree
# costs and the budget has to be sized for it.
#
# WHY THIS IS TASKS-TAB-LOCAL rather than a bump of the shared
# ``tasks.DEFAULT_PER_CALL_TIMEOUT``: that constant feeds
# ``tasks.DEFAULT_WHOLE_OPERATION_BUDGET``, which
# ``orchestrator._ORCHESTRATORS_PER_ROOT_BUDGET``,
# ``merge_queue._TASK_TITLES_BUDGET`` and ``app._TASK_CARDS_BUDGET`` all bind
# BY REFERENCE (task 4788). Raising the shared default to fix the Tasks tab
# would silently widen three unrelated route budgets — none of which fetches
# a 5 000-task tree, so none of which needs it.
# ``test_tasks_budget.py`` assertion (e) pins both halves of this.
_TASKS_PER_CALL_TIMEOUT = 4.4

# Whole-operation bound for ONE project root, enforced by ``asyncio.wait_for``
# in ``collect_tasks_with_counts``.
#
# ``_TASKS_PER_CALL_TIMEOUT`` (4.4) * 3 calls = 13.2 <= 14.0, leaving
# 0.8 s of slack so this deadline is a real backstop for non-MCP overhead
# (JSON decode, row shaping, event-loop scheduling) rather than coinciding
# exactly with the sum of its parts — the same reasoning as healthz's
# ``_DB_PROBE_TIMEOUT * 3 = 2.7 <= _HEALTHZ_TOTAL_BUDGET = 3.0``.
#
# It moved 7.0 -> 14.0 only because ``_TASKS_PER_CALL_TIMEOUT`` moved 2.0 ->
# 4.4: the parts-fit-the-whole shape is unchanged and the slack is still
# named. It is NOT an independent widening, and must not be raised on its own.
#
# What that sum does and does NOT claim: it bounds the sum of the
# PER-HTTP-REQUEST budgets. It does NOT bound a cold MCP session, which
# performs three posts (initialize, notifications/initialized, tools/call) and
# so can reach ``3 * DEFAULT_PER_CALL_TIMEOUT`` for a SINGLE tool call. That
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
    3. ``fetch_task_page(statuses=sorted(_TERMINAL_STATUSES), page_size=...,
       offset=...)``
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
    #
    # Both carry the Tasks-tab-LOCAL _TASKS_PER_CALL_TIMEOUT rather than
    # tasks.DEFAULT_PER_CALL_TIMEOUT: this tab is the only caller that reads a
    # 5 000-task tree, and at the shared 2.0 s default the measurement of
    # 2026-09-07 marked the three largest roots OFFLINE on a cold render. See
    # the constant for the numbers and for why the shared default may not move.
    fetched, status_map = await asyncio.gather(
        fetch_tasks(
            client, config, project_root,
            statuses=sorted(_ACTIVE_STATUSES),
            timeout=_TASKS_PER_CALL_TIMEOUT,
        ),
        fetch_statuses(
            client, config, project_root, timeout=_TASKS_PER_CALL_TIMEOUT,
        ),
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
        # fetch_task_page, not fetch_tasks: this read wants a PARTIAL answer
        # (the WARNING above says so), and after task 5018 that contract is in
        # the function name rather than in an argument combination.
        # page_size/offset slice a list ordered by ASCENDING id, so reaching
        # the high-id end requires a computed offset rather than a LIMIT.
        terminal = await fetch_task_page(
            client, config, project_root,
            statuses=sorted(_TERMINAL_STATUSES),
            page_size=window,
            offset=max(0, n_terminal - window),
            timeout=_TASKS_PER_CALL_TIMEOUT,
        )
        if isinstance(terminal, list):
            # DEDUP, not concatenate. The two fetches are separate cached
            # reads, so a task that completed between them appears in BOTH:
            # once from the active read, once from the terminal read. Emitting
            # both yields two rows sharing one _task_uid — the id the React
            # tab uses as a map key and as its selection identity — so the
            # task renders twice, as pending AND as done.
            #
            # This is not a narrow race. Both reads cache per (root,
            # narrowing, mode) for the TTL, and the terminal key embeds an offset
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

    runtime_status = _probe_status(runtime)
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
        rt = _runtime_fields(runtime_index, runtime_status, task_id, now=effective_now)

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
            rt = _runtime_fields(runtime_index, runtime_status, task_id, now=effective_now)
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

    # ROTATED for admission, canonical for output. The rotation is what stops
    # the same trailing roots being starved on every render; see
    # _rotated_project_roots. `roots` below is the canonical order the results
    # are re-assembled in.
    roots = _all_project_roots(config)
    admission_order = _rotated_project_roots(config)
    # Admission control, not a work queue: the coroutines are all created up
    # front and the semaphore decides how many are inside _shape_one_project
    # at once. See _TASKS_ROOT_CONCURRENCY for why the width is bounded.
    slots = asyncio.Semaphore(_TASKS_ROOT_CONCURRENCY)

    async def _one(root: Path) -> dict[str, Any]:
        """Shape ONE root, returning a result record — never mutating shared state.

        Every branch returns a record instead of appending to the outer lists.
        Appending from inside a concurrent coroutine would order the payload by
        COMPLETION, and the Tasks tab renders ``all_active`` directly, so the
        table would reshuffle on every 3 s poll. The caller re-assembles these
        records in ROOT order below.
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
                logger.warning(
                    'project %s: skipped — the %.1fs Tasks budget was already '
                    'spent before this project was reached; its rows and done '
                    'count are UNKNOWN for this render (not zero, and not offline)',
                    label, _TASKS_TOTAL_BUDGET,
                )
                return {'label': label, 'degraded': True}
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
                logger.warning(
                    'project %s: exceeded its %.1fs share of the %.1fs Tasks '
                    'budget (%.1fs remained) — its rows and done count are '
                    'UNKNOWN for this render (not zero, and not offline)',
                    label, _TASKS_PER_PROJECT_BUDGET, _TASKS_TOTAL_BUDGET, remaining,
                )
                return {'label': label, 'degraded': True}
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
                return {'label': label, 'offline': True}
        return {
            'label': label, 'active': active,
            'offline': offline, 'done_count': done_count,
        }

    # return_exceptions=False is correct here BECAUSE the broad `except
    # Exception` above lives INSIDE _one: nothing can escape to the gather, so
    # there is no exception for it to swallow, and a real escape (a bug in this
    # assembly code, a CancelledError) must still propagate rather than be
    # silently converted into a result object.
    results = await asyncio.gather(*(_one(root) for root in admission_order))
    by_label = {result['label']: result for result in results}

    # CANONICAL ROOT order — neither completion order nor admission order.
    # This is the only place the shared accumulators are written.
    for root in roots:
        result = by_label[_project_label(root)]
        label = result['label']
        if result.get('degraded'):
            degraded_projects.append(label)
            continue
        if result.get('offline'):
            offline_projects.append(label)
            continue
        done_count = result['done_count']
        if done_count is not None:
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
        all_active.extend(result['active'])

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
