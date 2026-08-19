"""Pure helper functions for scheduler-state MCP tools.

Kept separate from server/tools.py so the logic is testable without
standing up the full MCP server.  Mirrors the pattern used by the
scheduler-override tools registered in 1259.

No orchestrator imports — these helpers read on-disk files written by
the orchestrator process (a separate process).  The only coupling is the
on-disk file format (JSON snapshot + SQLite runs.db schema).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

_log = logging.getLogger(__name__)

# How long a resolved lock_depth stays memoised. ``lock_depth`` is red-tier
# (absent from the orchestrator's RELOADABLE_FIELDS) so it changes only at
# orchestrator restart; a short TTL therefore collapses the repeated reads a
# curator batch would otherwise do while still picking up a restart within a
# minute. Deliberately NOT process-lifetime: fused-memory outlives many
# orchestrator restarts, so a permanent memo could pin a stale depth forever.
_LOCK_DEPTH_TTL_SECONDS = 60.0

# (str(project_root), default) -> (monotonic_deadline, depth). Keyed on the
# default too, so a caller passing a different coarse fallback never reads
# another caller's fallback back out of the cache.
_lock_depth_cache: dict[tuple[str, int], tuple[float, int]] = {}

# project_roots already reported as running on the coarse fallback, so the
# diagnostic below is one line per project rather than one per candidate.
_lock_depth_fallback_logged: set[str] = set()


def clear_lock_depth_cache() -> None:
    """Drop the memoised lock_depth values and the one-time-log bookkeeping.

    Exists for tests, which write several snapshots within one process and
    must not see each other's memoised values; production callers rely on
    ``_LOCK_DEPTH_TTL_SECONDS`` instead. Public (not underscore-private) so
    tests do not have to reach into module internals to stay deterministic.
    """
    _lock_depth_cache.clear()
    _lock_depth_fallback_logged.clear()


def _empty_skeleton() -> dict:
    """Build and return a fresh empty skeleton dict.

    Returns a new dict on every call so callers cannot accidentally mutate
    a shared module-level object and corrupt future calls.
    """
    return {
        'skip_counts': {},
        'parks': {},
        'park_stacks': {},
        'effective_priorities': {},
        'pin_queue': [],
        'overrides': {},
        'current_holders': {},
        'is_paused': False,
        'pause_reason': None,
        'snapshot_at': None,
    }


def read_scheduler_state(project_root: Path) -> dict:
    """Read and return the scheduler state snapshot from disk.

    Reads ``<project_root>/data/orchestrator/scheduler_state.json``.
    Returns the empty skeleton dict when the file is absent, unreadable,
    or contains invalid JSON.  A missing file is normal before the first
    orchestrator tick and is handled silently; any other error is logged
    as a warning before returning the empty skeleton.
    """
    path = project_root / 'data' / 'orchestrator' / 'scheduler_state.json'
    try:
        return json.loads(path.read_bytes())
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning('read_scheduler_state: %s — returning empty skeleton', exc)
    return _empty_skeleton()


def effective_lock_depth(project_root: str | Path | None, default: int) -> int:
    """Return the scheduler's effective ``lock_depth`` for one project.

    fused-memory is ONE server serving MANY projects, and each project's
    orchestrator resolves its own ``lock_depth`` (measured 2026-08-07 across
    the fleet: 3, 4, 4, 4, 4, 10, 12 — five distinct values). A single global
    scalar in fused-memory's own config cannot match all of them, so callers
    that need scheduler-consistent lock keys must resolve the depth per
    project. This reads the value the orchestrator itself published, so there
    is no second implementation of the orchestrator's config layering (package
    ``defaults.yaml`` -> project yaml -> pydantic default) here — that layering
    lives in ``orchestrator.config``, which fused-memory deliberately does not
    import.

    STALENESS IS A NON-ISSUE FOR THIS FIELD SPECIFICALLY. ``lock_depth`` is
    red-tier (not in the orchestrator's ``RELOADABLE_FIELDS``), so it changes
    only at orchestrator restart — unlike ``parks``/``current_holders`` in the
    same snapshot, which are live state and genuinely stale between writes. A
    snapshot hours old still carries the correct depth.

    The resolved value is memoised per ``(project_root, default)`` for
    ``_LOCK_DEPTH_TTL_SECONDS``. The read is blocking and the curator calls
    this once per candidate, so without the memo a batch of N candidates for
    one project would re-read and re-parse the same ~29 KB snapshot N times
    for a value that cannot change between them.

    ``default`` is returned when the snapshot is absent (a freshly onboarded
    project whose orchestrator has never run), unreadable, or carries no
    usable ``lock_depth``; taking that path logs one line per project_root so
    an unexpectedly-missing snapshot is diagnosable from logs rather than only
    from bad curator decisions.

    Callers should pass a COARSE default rather than a guess at "the usual"
    fleet value, because the error costs are asymmetric — but note the
    asymmetry is between *degrees of unreliability*, not free-vs-costly. A
    falsely-FINE key under-matches: the overlapping task is missed outright,
    every time, and the cost is a DUPLICATE TASK — the failure the curator
    exists to prevent. A falsely-COARSE key over-matches: usually that costs
    only an LLM look at an irrelevant pool entry, but the module stream is
    sorted by ``_module_sort_key`` — ``(status, priority, task_id)``, i.e. NOT
    by relevance to the candidate — and then truncated to
    ``curator.pool_module_cap``. So a very coarse key against a deep project
    can over-match widely enough to push the genuinely overlapping task out of
    the retained window and produce the same duplicate. Coarse is still the
    right direction (it degrades only past the cap, where fine fails always),
    but it is not cost-free, which is another reason to resolve the real depth
    per project rather than lean on the fallback.
    """
    # A falsy project_root (``None`` or ``''``) means "unknown project" — take
    # the coarse default before touching the filesystem. ``None`` is in the
    # annotation because this guard deliberately accepts it, not as an
    # oversight: the guard's contract is falsiness, and a caller that has no
    # project in hand should get the coarse default by the same documented
    # path as ``''`` rather than through the broad ``except`` below.
    # ``Path('')`` normalises to ``Path('.')``,
    # so without this the read is CWD-relative and silently hands back the
    # depth of whatever project the fused-memory server process happens to be
    # rooted in: a cross-project leak, strictly worse than the coarse
    # fallback. Deliberately narrow — falsy only. Relative paths are NOT
    # rejected here; that would be a behaviour change beyond this guard, and
    # the wire boundary (``_normalize_project_root`` -> ``validate_project_root``)
    # is where absoluteness is enforced for the production path.
    if not project_root:
        return default

    key = (str(project_root), default)
    now = time.monotonic()
    cached = _lock_depth_cache.get(key)
    if cached is not None and now < cached[0]:
        return cached[1]

    depth = _read_snapshot_lock_depth(project_root)
    if depth is None:
        if key[0] not in _lock_depth_fallback_logged:
            _lock_depth_fallback_logged.add(key[0])
            _log.info(
                'effective_lock_depth: no usable lock_depth in %s — using the '
                'coarse fallback %s. Expected for a project whose orchestrator '
                'has never run; otherwise the snapshot is missing or malformed '
                'and module-lock keys will be coarser than the scheduler\'s.',
                key[0], default,
            )
        depth = default
    _lock_depth_cache[key] = (now + _LOCK_DEPTH_TTL_SECONDS, depth)
    return depth


def _read_snapshot_lock_depth(project_root: str | Path) -> int | None:
    """Return the snapshot's usable ``lock_depth``, or ``None`` if there is none.

    ``None`` means "no usable value, take the caller's default" and covers an
    absent snapshot, an unreadable/invalid-JSON file, a missing key, and a
    value that is not a positive non-bool int. Never raises — depth resolution
    escaping into ``_build_corpus`` would degrade the curator to an empty pool
    and file a duplicate task, the exact failure the coarse fallback exists to
    prevent.
    """
    try:
        state = read_scheduler_state(Path(project_root))
        # read_scheduler_state hands back json.loads() output unchecked, so a
        # body that is valid JSON but not an object ('null', '[]', '"x"')
        # arrives here as a non-dict and would make .get raise. Guard before
        # touching it — the sibling consumer
        # recon_write_policy._corroboration_verdict defends identically.
        if not isinstance(state, dict):
            _log.warning(
                'effective_lock_depth: snapshot for %s is %s, not an object',
                project_root, type(state).__name__,
            )
            return None
        depth = state.get('lock_depth')
    except Exception as exc:  # defensive: never let depth resolution raise
        _log.warning('effective_lock_depth: %s — no usable depth', exc)
        return None
    # bool is an int subclass; reject it explicitly so True never means depth 1.
    if isinstance(depth, int) and not isinstance(depth, bool) and depth > 0:
        return depth
    return None


async def read_scheduler_events(
    project_root: Path,
    since: str | None,
    limit: int,
    event_types: list[str] | None,
) -> dict:
    """Read scheduler events from runs.db, newest-first.

    Opens ``<project_root>/data/orchestrator/runs.db`` in read-only URI mode
    via aiosqlite.  Returns ``{'events': [...], 'count': <int>}`` where each
    event is a dict with keys: id, timestamp, run_id, task_id, event_type, data.

    ``data`` is the JSON-parsed payload (dict), never a raw string.

    Returns ``{'events': [], 'count': 0}`` when the database is missing.

    **Path canonicalization**: the runs.db path is passed through
    ``Path.resolve()`` before the SQLite URI is constructed, mirroring the
    ``resolve()``-then-``as_uri()`` pattern used by ``DbPool.get`` in
    ``dashboard/src/dashboard/data/db.py``.  This ensures that symlinks in
    the path are expanded before the URI is built, so the connection always targets
    the real file.  **Symlink deployments beware**: if ``project_root`` is itself
    a symlink (or contains symlink components), the SQLite connection will target
    the resolved (canonical) path rather than the symlinked one.  This is
    intentional — it avoids URI-encoding ambiguity — but it is a silent behavioral
    change relative to a raw path open.
    """
    import aiosqlite

    db_path = (project_root / 'data' / 'orchestrator' / 'runs.db').resolve()
    if not db_path.exists():
        return {'events': [], 'count': 0}

    clauses: list[str] = []
    params: list = []

    if event_types:
        placeholders = ', '.join('?' for _ in event_types)
        clauses.append(f'event_type IN ({placeholders})')
        params.extend(event_types)

    if since is not None:
        clauses.append('timestamp >= ?')
        params.append(since)

    where = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
    # NOTE: When event_types is supplied the query planner may scan by
    # timestamp and filter by event_type (or vice versa) depending on
    # available indexes.  If runs.db grows large, consider adding a
    # composite index on (event_type, timestamp DESC) in the orchestrator's
    # schema migration and verify the plan with EXPLAIN QUERY PLAN.
    sql = (
        f'SELECT id, timestamp, run_id, task_id, event_type, data '
        f'FROM events {where} '
        f'ORDER BY timestamp DESC, id DESC '
        f'LIMIT ?'
    )
    params.append(limit)

    uri = f'{db_path.as_uri()}?mode=ro'
    async with aiosqlite.connect(uri, uri=True) as db:
        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()

    events = [
        {
            'id': r[0],
            'timestamp': r[1],
            'run_id': r[2],
            'task_id': r[3],
            'event_type': r[4],
            'data': json.loads(r[5] or '{}'),
        }
        for r in rows
    ]
    return {'events': events, 'count': len(events)}
