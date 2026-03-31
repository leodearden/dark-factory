"""Async queries for cost tracking metrics.

Reads from data/orchestrator/runs.db (invocations and account_events tables
written by CostStore) to produce per-project and per-account cost statistics.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import aiosqlite

from dashboard.data.db import with_db

logger = logging.getLogger(__name__)


def _cutoff(days: int) -> str:
    """Return ISO-format cutoff datetime for the given look-back window."""
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


# ---------------------------------------------------------------------------
# 1. Cost summary (per project)
# ---------------------------------------------------------------------------

async def get_cost_summary(
    db: aiosqlite.Connection | None,
    *,
    days: int = 7,
) -> dict[str, dict]:
    """Per-project cost summary.

    Returns {project_id: {total_spend, avg_cost_per_task, active_accounts,
                           cap_events}}.
    *total_spend* is the sum of cost_usd for all invocations in the window.
    *avg_cost_per_task* is total_spend / distinct task count (0 when no tasks).
    *active_accounts* is the count of distinct account_name values.
    *cap_events* is the count of account_events rows with event_type='cap_hit'
    scoped to the same project_id within the window.
    """
    since = _cutoff(days)

    async def _query(db: aiosqlite.Connection) -> dict[str, dict]:
        inv_rows = await db.execute_fetchall(
            'SELECT project_id, '
            '       SUM(cost_usd) AS total_spend, '
            '       COUNT(DISTINCT task_id) AS task_count, '
            '       COUNT(DISTINCT account_name) AS account_count '
            '  FROM invocations '
            ' WHERE completed_at >= ? '
            ' GROUP BY project_id',
            (since,),
        )

        # cap_hit counts per project from account_events
        cap_rows = await db.execute_fetchall(
            "SELECT project_id, COUNT(*) AS cap_count "
            '  FROM account_events '
            " WHERE event_type = 'cap_hit' AND created_at >= ? "
            ' GROUP BY project_id',
            (since,),
        )
        cap_by_project: dict[str, int] = {row[0]: row[1] for row in cap_rows}

        result: dict[str, dict] = {}
        for row in inv_rows:
            project_id, total_spend, task_count, account_count = row
            total_spend = total_spend or 0.0
            task_count = task_count or 0
            result[project_id] = {
                'total_spend': total_spend,
                'task_count': task_count,
                'avg_cost_per_task': total_spend / task_count if task_count else 0.0,
                'active_accounts': account_count or 0,
                'cap_events': cap_by_project.get(project_id, 0),
            }
        return result

    return await with_db(db, _query, {})


# ---------------------------------------------------------------------------
# 2. Cost by project (per-model breakdown)
# ---------------------------------------------------------------------------


async def get_cost_by_project(
    db: aiosqlite.Connection | None,
    *,
    days: int = 7,
) -> dict[str, list[dict]]:
    """Per-project cost broken down by model.

    Returns {project_id: [{model: str, total: float}, ...]}.
    Entries are ordered by total descending.
    """
    since = _cutoff(days)

    async def _query(db: aiosqlite.Connection) -> dict[str, list[dict]]:
        rows = await db.execute_fetchall(
            'SELECT project_id, model, SUM(cost_usd) AS total '
            '  FROM invocations '
            ' WHERE completed_at >= ? '
            ' GROUP BY project_id, model '
            ' ORDER BY total DESC',
            (since,),
        )

        result: dict[str, list[dict]] = {}
        for row in rows:
            project_id, model, total = row
            if project_id not in result:
                result[project_id] = []
            result[project_id].append({'model': model, 'total': total or 0.0})
        return result

    return await with_db(db, _query, {})


# ---------------------------------------------------------------------------
# 3. Cost by account
# ---------------------------------------------------------------------------

async def get_cost_by_account(
    db: aiosqlite.Connection | None,
    *,
    days: int = 7,
) -> dict[str, dict]:
    """Per-account cost summary with cap status.

    Returns {account_name: {spend, invocations, cap_events, last_cap, status}}.
    *status* is 'capped' if the most recent account_event is 'cap_hit' with no
    subsequent 'resumed'; otherwise 'active'.
    """
    since = _cutoff(days)

    async def _query(db: aiosqlite.Connection) -> dict[str, dict]:
        # Aggregate spend and invocation count per account
        inv_rows = await db.execute_fetchall(
            'SELECT account_name, SUM(cost_usd) AS spend, COUNT(*) AS cnt '
            '  FROM invocations '
            ' WHERE completed_at >= ? '
            ' GROUP BY account_name',
            (since,),
        )

        # Cap event counts and most-recent event per account.
        # Use MAX(CASE WHEN event_type='cap_hit' ...) so last_cap_at only holds
        # cap_hit timestamps (never resumed). Add created_at >= ? guard to the
        # correlated subquery so status is derived from in-window events only.
        evt_rows = await db.execute_fetchall(
            "SELECT account_name, "
            "       SUM(CASE WHEN event_type = 'cap_hit' THEN 1 ELSE 0 END) AS cap_count, "
            "       MAX(CASE WHEN event_type = 'cap_hit' THEN created_at END) AS last_cap_at, "
            '       (SELECT event_type FROM account_events ae2 '
            '         WHERE ae2.account_name = account_events.account_name '
            '           AND ae2.created_at >= ? '
            '         ORDER BY created_at DESC LIMIT 1) AS last_event_type '
            '  FROM account_events '
            ' WHERE created_at >= ? '
            ' GROUP BY account_name',
            (since, since),
        )

        # Cap counts and status keyed by account_name
        caps: dict[str, dict] = {}
        for row in evt_rows:
            account_name, cap_count, last_cap_at, last_event_type = row
            caps[account_name] = {
                'cap_events': cap_count or 0,
                'last_cap': last_cap_at,  # NULL when no cap_hit in window
                'status': 'capped' if last_event_type == 'cap_hit' else 'active',
            }

        result: dict[str, dict] = {}
        for row in inv_rows:
            account_name, spend, cnt = row
            cap_info = caps.get(
                account_name,
                {'cap_events': 0, 'last_cap': None, 'status': 'active'},
            )
            result[account_name] = {
                'spend': spend or 0.0,
                'invocations': cnt or 0,
                'cap_events': cap_info['cap_events'],
                'last_cap': cap_info['last_cap'],
                'status': cap_info['status'],
            }
        return result

    return await with_db(db, _query, {})

# ---------------------------------------------------------------------------
# 4. Cost by role
# ---------------------------------------------------------------------------

async def get_cost_by_role(
    db: aiosqlite.Connection | None,
    *,
    days: int = 7,
) -> dict[str, dict]:
    """Per-project cost broken down by role, then model.

    Returns {project_id: {role: {model: total}}}.
    """
    since = _cutoff(days)

    async def _query(db: aiosqlite.Connection) -> dict[str, dict]:
        rows = await db.execute_fetchall(
            'SELECT project_id, role, model, SUM(cost_usd) AS total '
            '  FROM invocations '
            ' WHERE completed_at >= ? '
            ' GROUP BY project_id, role, model',
            (since,),
        )

        result: dict[str, dict] = {}
        for row in rows:
            project_id, role, model, total = row
            if project_id not in result:
                result[project_id] = {}
            if role not in result[project_id]:
                result[project_id][role] = {}
            result[project_id][role][model] = total or 0.0
        return result

    return await with_db(db, _query, {})

# ---------------------------------------------------------------------------
# 5. Cost trend (daily time series)
# ---------------------------------------------------------------------------

async def get_cost_trend(
    db: aiosqlite.Connection | None,
    *,
    days: int = 7,
) -> dict[str, list[dict]]:
    """Per-project daily cost totals over the look-back window.

    Returns {project_id: [{day: str, total: float}, ...]}.
    Entries cover every calendar day in the window (gaps filled with 0.0),
    ordered chronologically.
    """
    now = datetime.now(UTC)
    since = (now - timedelta(days=days)).isoformat()

    # Pre-fill all days in the window (today included)
    all_days: list[str] = []
    for i in range(days - 1, -1, -1):
        day = (now - timedelta(days=i)).strftime('%Y-%m-%d')
        all_days.append(day)

    async def _query(db: aiosqlite.Connection) -> dict[str, list[dict]]:
        rows = await db.execute_fetchall(
            "SELECT project_id, DATE(completed_at) AS day, SUM(cost_usd) AS total "
            '  FROM invocations '
            ' WHERE completed_at >= ? '
            ' GROUP BY project_id, day '
            ' ORDER BY day',
            (since,),
        )

        # Index DB results
        data: dict[str, dict[str, float]] = {}
        for row in rows:
            project_id, day, total = row
            if project_id not in data:
                data[project_id] = {}
            data[project_id][day] = total or 0.0

        # Build gap-filled result
        result: dict[str, list[dict]] = {}
        for project_id, day_totals in data.items():
            result[project_id] = [
                {'day': d, 'total': day_totals.get(d, 0.0)}
                for d in all_days
            ]
        return result

    return await with_db(db, _query, {})

# ---------------------------------------------------------------------------
# 6. Account events
# ---------------------------------------------------------------------------

async def get_account_events(
    db: aiosqlite.Connection | None,
    *,
    days: int = 7,
) -> list[dict]:
    """Recent account events (cap_hit, resumed, etc.) within the window.

    Returns [{account_name, event_type, project_id, run_id, details,
               created_at}, ...] ordered by created_at DESC.
    *details* is returned as-is (string or None) — callers may parse JSON.
    """
    since = _cutoff(days)

    async def _query(db: aiosqlite.Connection) -> list[dict]:
        rows = await db.execute_fetchall(
            'SELECT account_name, event_type, project_id, run_id, '
            '       details, created_at '
            '  FROM account_events '
            ' WHERE created_at >= ? '
            ' ORDER BY created_at DESC',
            (since,),
        )
        return [
            {
                'account_name': row[0],
                'event_type': row[1],
                'project_id': row[2],
                'run_id': row[3],
                'details': row[4],
                'created_at': row[5],
            }
            for row in rows
        ]

    return await with_db(db, _query, [])


# ---------------------------------------------------------------------------
# 7. Run cost breakdown
# ---------------------------------------------------------------------------

async def get_run_cost_breakdown(
    db: aiosqlite.Connection | None,
    *,
    days: int = 7,
) -> list[dict]:
    """Per-run cost breakdown grouped by task, with invocation detail.

    Returns a list of run dicts:
        [{run_id, project_id, total_cost,
          tasks: [{task_id, title, cost,
                   invocations: [{model, role, cost_usd, account_name,
                                  duration_ms, capped}]}]}]

    Invocations with NULL task_id are grouped under task_id=None with title=None.
    LEFT JOIN with task_results provides task titles.
    """
    since = _cutoff(days)

    async def _query(db: aiosqlite.Connection) -> list[dict]:
        rows = await db.execute_fetchall(
            'SELECT i.run_id, i.project_id, i.task_id, '
            '       tr.title, '
            '       i.model, i.role, i.cost_usd, '
            '       i.account_name, i.duration_ms, i.capped '
            '  FROM invocations i '
            '  LEFT JOIN task_results tr '
            '    ON i.run_id = tr.run_id AND i.task_id = tr.task_id '
            ' WHERE i.completed_at >= ? '
            ' ORDER BY i.run_id, i.task_id, i.id',
            (since,),
        )

        # Build nested structure: run_id → task_id → invocations
        runs: dict[str, dict] = {}
        for row in rows:
            run_id, project_id, task_id, title, model, role, cost_usd, \
                account_name, duration_ms, capped = row
            cost_usd = cost_usd or 0.0

            if run_id not in runs:
                runs[run_id] = {
                    'run_id': run_id,
                    'project_id': project_id,
                    'total_cost': 0.0,
                    'tasks': {},  # task_id → task dict (temporary)
                }
            run = runs[run_id]
            run['total_cost'] += cost_usd

            if task_id not in run['tasks']:
                run['tasks'][task_id] = {
                    'task_id': task_id,
                    'title': title,
                    'cost': 0.0,
                    'invocations': [],
                }
            task = run['tasks'][task_id]
            task['cost'] += cost_usd
            task['invocations'].append({
                'model': model,
                'role': role,
                'cost_usd': cost_usd,
                'account_name': account_name,
                'duration_ms': duration_ms,
                'capped': bool(capped),
            })

        # Convert to list, converting task dict to list
        result: list[dict] = []
        for run in runs.values():
            tasks_list = list(run['tasks'].values())
            result.append({
                'run_id': run['run_id'],
                'project_id': run['project_id'],
                'total_cost': run['total_cost'],
                'tasks': tasks_list,
            })
        return result

    return await with_db(db, _query, [])
