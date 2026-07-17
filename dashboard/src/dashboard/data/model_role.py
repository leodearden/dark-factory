"""Async queries for the per-(model×role) outcome rollup (task 2534 δ).

Dashboard-side mirror of :func:`orchestrator.digest.model_role_rollup` —
the SAME GROUP-BY-(model, role) semantics (invocations LEFT JOIN
task_results on (run_id, task_id) for done/blocked rates, cap-hit rate,
$/done) plus per-role turn-cap saturation from the events table
(routing_decision max_turns vs invocation_end turns), reimplemented here in
async aiosqlite against the same runs.db file the dashboard already opens
for cost queries (:mod:`dashboard.data.costs`). Digest and dashboard run in
different processes with different DB drivers, so the query is written
twice rather than shared — see plan.json's design_decisions for task 2534.

Fail-open throughout: a missing/offline DB (``None``) or any query error
contributes nothing (``with_db``'s ``([], {})`` default flows through to an
empty ``{'rows': [], 'turn_cap_saturation': {}}`` rollup), matching the
digest side's empty-``ModelRoleRollup`` contract.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

import aiosqlite

from dashboard.data.db import with_db
from dashboard.data.utils import resolve_now

logger = logging.getLogger(__name__)

_DONE_OUTCOME = 'done'
_BLOCKED_OUTCOME = 'blocked'


def _cutoff(days: int, *, now: datetime | None = None) -> str:
    """Return ISO-format cutoff datetime for the given look-back window.

    Local copy of :func:`dashboard.data.costs._cutoff` — kept independent
    rather than imported so this module has no cross-module dependency on
    another data module's private helper (mirrors how the digest-side
    rollup owns its own ``_query_events_ro`` scaffold).
    """
    return (resolve_now(now) - timedelta(days=days)).isoformat()


async def _query_model_role_cells(
    db: aiosqlite.Connection, since: str,
) -> tuple[list[dict], dict[str, dict]]:
    """Raw per-(model,role) cells + raw per-role saturation counts for one DB.

    Returns ``(cells, saturation_raw)``:

    - ``cells``: one dict per (model, role) with the raw aggregates needed
      to both render a row AND merge correctly across DBs — in particular
      ``distinct_done_tasks`` (not just the derived ``cost_per_done``),
      since summing already-divided ``cost_per_done`` values across DBs
      would be wrong.
    - ``saturation_raw``: ``{role: {'hits', 'total', 'has_max_turns'}}`` —
      raw numerator/denominator so the aggregator can sum them across DBs
      before dividing, rather than averaging already-divided ratios.
    """
    raw_rows = await db.execute_fetchall(
        'SELECT i.model, i.role, '
        '       COUNT(*) AS invocation_count, '
        '       SUM(i.capped) AS capped_count, '
        '       SUM(i.cost_usd) AS total_cost_usd, '
        '       SUM(CASE WHEN tr.outcome = ? THEN 1 ELSE 0 END) AS done_count, '
        '       SUM(CASE WHEN tr.outcome = ? THEN 1 ELSE 0 END) AS blocked_count, '
        '       COUNT(DISTINCT CASE WHEN tr.outcome = ? THEN i.task_id END) '
        '           AS distinct_done_tasks '
        '  FROM invocations i '
        '  LEFT JOIN task_results tr '
        '    ON i.run_id = tr.run_id AND i.task_id = tr.task_id '
        ' WHERE i.completed_at >= ? '
        ' GROUP BY i.model, i.role',
        (_DONE_OUTCOME, _BLOCKED_OUTCOME, _DONE_OUTCOME, since),
    )

    cells: list[dict] = []
    for row in raw_rows:
        cells.append({
            'model': row['model'],
            'role': row['role'],
            'invocation_count': int(row['invocation_count'] or 0),
            'capped_count': int(row['capped_count'] or 0),
            'total_cost_usd': float(row['total_cost_usd'] or 0.0),
            'done_count': int(row['done_count'] or 0),
            'blocked_count': int(row['blocked_count'] or 0),
            'distinct_done_tasks': int(row['distinct_done_tasks'] or 0),
        })

    saturation_raw = await _compute_turn_cap_saturation_raw(db, since)
    return cells, saturation_raw


async def _compute_turn_cap_saturation_raw(
    db: aiosqlite.Connection, since: str,
) -> dict[str, dict]:
    """Per-role ``{'hits', 'total', 'has_max_turns'}`` from the events table.

    Mirrors ``orchestrator.digest._compute_turn_cap_saturation``: max_turns
    is role-constant config recovered via MAX(routing_decision.max_turns)
    over the window; a role with invocation_end rows but no routing_decision
    max_turns gets ``has_max_turns=False`` so the caller can report
    saturation=None (fail-open) rather than a misleading 0.0.
    """
    max_turns_rows = await db.execute_fetchall(
        "SELECT role, MAX(json_extract(data, '$.max_turns')) AS max_turns "
        '  FROM events '
        " WHERE event_type = 'routing_decision' "
        '   AND timestamp >= ? '
        '   AND role IS NOT NULL '
        ' GROUP BY role',
        (since,),
    )
    max_turns_by_role: dict[str, float] = {
        row['role']: row['max_turns'] for row in max_turns_rows
        if row['max_turns'] is not None
    }

    turns_rows = await db.execute_fetchall(
        "SELECT role, json_extract(data, '$.turns') AS turns "
        '  FROM events '
        " WHERE event_type = 'invocation_end' "
        '   AND timestamp >= ? '
        '   AND role IS NOT NULL',
        (since,),
    )

    raw: dict[str, dict] = {}
    for row in turns_rows:
        turns = row['turns']
        if turns is None:
            continue
        role = row['role']
        entry = raw.setdefault(
            role, {'hits': 0, 'total': 0, 'has_max_turns': role in max_turns_by_role},
        )
        entry['total'] += 1
        max_turns = max_turns_by_role.get(role)
        if max_turns is not None and turns >= max_turns:
            entry['hits'] += 1
    return raw


def _finalize_cell(cell: dict) -> dict:
    """Derive rates/cap-hit-rate/cost_per_done from one merged/raw cell."""
    invocation_count = cell['invocation_count']
    done_count = cell['done_count']
    blocked_count = cell['blocked_count']
    capped_count = cell['capped_count']
    total_cost_usd = cell['total_cost_usd']
    distinct_done_tasks = cell['distinct_done_tasks']

    return {
        'model': cell['model'],
        'role': cell['role'],
        'invocation_count': invocation_count,
        'done_count': done_count,
        'blocked_count': blocked_count,
        'done_rate': (done_count / invocation_count) if invocation_count else 0.0,
        'blocked_rate': (blocked_count / invocation_count) if invocation_count else 0.0,
        'capped_count': capped_count,
        'cap_hit_rate': (capped_count / invocation_count) if invocation_count else 0.0,
        'total_cost_usd': total_cost_usd,
        # None (not 0/inf) when the cell has zero done tasks — distinguishes
        # "no completed tasks yet" from "$0 per done" (matches the digest side).
        'cost_per_done': (
            (total_cost_usd / distinct_done_tasks) if distinct_done_tasks > 0 else None
        ),
    }


def _finalize_saturation(raw: dict[str, dict]) -> dict[str, float | None]:
    """Convert raw per-role hit/total counts into the public ratio map."""
    saturation: dict[str, float | None] = {}
    for role, entry in raw.items():
        if not entry['has_max_turns'] or not entry['total']:
            saturation[role] = None
        else:
            saturation[role] = entry['hits'] / entry['total']
    return saturation


async def _per_db_cells(
    db: aiosqlite.Connection | None, since: str,
) -> tuple[list[dict], dict[str, dict]]:
    """Fail-open per-DB raw query: ``([], {})`` on a None/offline/erroring DB."""
    return await with_db(
        db, lambda conn: _query_model_role_cells(conn, since), ([], {}),
    )


async def get_model_role_rollup(
    db: aiosqlite.Connection | None,
    *,
    days: int = 7,
    now: datetime | None = None,
) -> dict:
    """Per-(model, role) outcome rollup + per-role turn-cap saturation for one DB.

    Returns::

        {
            'rows': [{'model', 'role', 'invocation_count', 'done_count',
                      'blocked_count', 'done_rate', 'blocked_rate',
                      'capped_count', 'cap_hit_rate', 'total_cost_usd',
                      'cost_per_done'}, ...],
            'turn_cap_saturation': {role: float | None, ...},
        }

    Numerically identical (same columns, same rate formulas) to
    :func:`orchestrator.digest.model_role_rollup` given the same runs.db
    and window — a shared-contract cross-check test asserts this.

    Fail-open: ``db is None`` or any query error returns the empty rollup.
    """
    since = _cutoff(days, now=now)
    cells, saturation_raw = await _per_db_cells(db, since)
    return {
        'rows': [_finalize_cell(c) for c in cells],
        'turn_cap_saturation': _finalize_saturation(saturation_raw),
    }


async def aggregate_model_role_rollup(
    dbs: list[aiosqlite.Connection | None],
    *,
    days: int = 7,
    now: datetime | None = None,
) -> dict:
    """Merge :func:`get_model_role_rollup`'s raw inputs across multiple DBs.

    Per-(model, role) counts/costs are summed and rates are recomputed from
    the merged totals (not averaged across DBs). Turn-cap saturation is
    merged by summing each role's raw hit/total counts across DBs before
    dividing — an average-of-ratios would misweight a DB with few
    invocations equally against one with many.

    A ``None`` or offline DB is skipped (contributes nothing) rather than
    failing the whole aggregate — mirrors ``aggregate_cost_by_role``.

    *now* is resolved once and threaded to every per-DB cutoff so all
    queries share a single window, closing the per-DB clock-skew race.
    """
    now = resolve_now(now)
    since = _cutoff(days, now=now)
    results = await asyncio.gather(*(_per_db_cells(db, since) for db in dbs))

    merged_cells: dict[tuple[str, str], dict] = {}
    merged_saturation: dict[str, dict] = {}

    for cells, saturation_raw in results:
        for cell in cells:
            key = (cell['model'], cell['role'])
            m = merged_cells.setdefault(key, {
                'model': cell['model'], 'role': cell['role'],
                'invocation_count': 0, 'done_count': 0, 'blocked_count': 0,
                'capped_count': 0, 'total_cost_usd': 0.0, 'distinct_done_tasks': 0,
            })
            m['invocation_count'] += cell['invocation_count']
            m['done_count'] += cell['done_count']
            m['blocked_count'] += cell['blocked_count']
            m['capped_count'] += cell['capped_count']
            m['total_cost_usd'] += cell['total_cost_usd']
            m['distinct_done_tasks'] += cell['distinct_done_tasks']

        for role, entry in saturation_raw.items():
            ms = merged_saturation.setdefault(
                role, {'hits': 0, 'total': 0, 'has_max_turns': False},
            )
            ms['hits'] += entry['hits']
            ms['total'] += entry['total']
            ms['has_max_turns'] = ms['has_max_turns'] or entry['has_max_turns']

    return {
        'rows': [_finalize_cell(c) for c in merged_cells.values()],
        'turn_cap_saturation': _finalize_saturation(merged_saturation),
    }
