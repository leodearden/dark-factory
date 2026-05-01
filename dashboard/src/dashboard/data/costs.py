"""Async queries for cost tracking metrics.

Reads from data/orchestrator/runs.db (invocations and account_events tables
written by CostStore) to produce per-project and per-account cost statistics.

When multiple project roots are configured, the ``aggregate_*`` functions
query each project's runs.db in parallel and merge the results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
from datetime import UTC, datetime, timedelta

import aiosqlite

from dashboard.data.db import with_db
from dashboard.data.stats_utils import percentile

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
            '       COUNT(DISTINCT account_name) AS account_count, '
            '       SUM(input_tokens) AS input_tokens, '
            '       SUM(output_tokens) AS output_tokens, '
            '       SUM(cache_read_tokens) AS cache_read_tokens, '
            '       SUM(cache_create_tokens) AS cache_create_tokens '
            '  FROM invocations '
            ' WHERE completed_at >= ? '
            ' GROUP BY project_id',
            (since,),
        )

        # Per-run cost rollups for p95 percentile. Group by run_id within
        # each project so a single run's many invocations sum to one cost.
        run_rows = await db.execute_fetchall(
            'SELECT project_id, run_id, SUM(cost_usd) AS run_cost '
            '  FROM invocations '
            ' WHERE completed_at >= ? '
            ' GROUP BY project_id, run_id',
            (since,),
        )
        run_costs_by_project: dict[str, list[float]] = {}
        for row in run_rows:
            run_costs_by_project.setdefault(row['project_id'], []).append(
                row['run_cost'] or 0.0
            )

        # cap_hit counts per project from account_events
        cap_rows = await db.execute_fetchall(
            "SELECT project_id, COUNT(*) AS cap_count "
            '  FROM account_events '
            " WHERE event_type = 'cap_hit' AND created_at >= ? "
            ' GROUP BY project_id',
            (since,),
        )
        cap_by_project: dict[str, int] = {
            row['project_id']: row['cap_count'] for row in cap_rows
        }

        result: dict[str, dict] = {}
        for row in inv_rows:
            project_id = row['project_id']
            total_spend = row['total_spend'] or 0.0
            task_count = row['task_count'] or 0
            account_count = row['account_count']
            input_t = row['input_tokens'] or 0
            output_t = row['output_tokens'] or 0
            cache_read = row['cache_read_tokens'] or 0
            cache_create = row['cache_create_tokens'] or 0
            result[project_id] = {
                'total_spend': total_spend,
                'task_count': task_count,
                'avg_cost_per_task': total_spend / task_count if task_count else 0.0,
                'active_accounts': account_count or 0,
                'cap_events': cap_by_project.get(project_id, 0),
                'tokens': {
                    'input': input_t,
                    'output': output_t,
                    'cache_read': cache_read,
                    'cache_create': cache_create,
                    'total': input_t + output_t + cache_read + cache_create,
                },
                # Raw per-run costs for p95 computation; aggregator concatenates
                # across DBs before sorting so the percentile is correct globally.
                'run_costs': run_costs_by_project.get(project_id, []),
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
            project_id = row['project_id']
            if project_id not in result:
                result[project_id] = []
            result[project_id].append({'model': row['model'], 'total': row['total'] or 0.0})
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
    """Per-account cost summary with availability status.

    Returns {account_name: {spend, invocations, cap_events, last_cap,
                            last_resumed, last_auth_fail, last_auth_resumed,
                            status, resets_at}}.

    *status* is ``'capped'`` when the account is unavailable to the
    orchestrator — i.e. the most recent unavailability event (``cap_hit`` or
    ``auth_failed``) is later than the most recent recovery event (``resumed``
    or ``auth_resumed``). Otherwise ``'active'``. The wire string stays
    ``'capped'`` for UI compatibility; semantically it means "unavailable",
    matching the gate's ``all(a.capped or a.auth_failed)`` predicate.

    Anthropic's modern cap signal arrives as HTTP 429 (routed through
    ``UsageGate._handle_auth_failure`` and persisted as ``auth_failed``), so
    folding ``auth_failed`` in is what makes the dashboard track the gate's
    actual decision rather than only the legacy stderr-cap path.

    *resets_at* is the parsed reset ISO timestamp from whichever of the
    latest ``cap_hit`` / ``auth_failed`` details payload is more recent (or
    None when not unavailable / unknown).
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

        # Per-account unavailability / recovery timestamps plus the latest
        # cap_hit and auth_failed details payloads (for resets_at extraction).
        #
        # latest_cap / latest_auth_fail pick the newest cap_hit / auth_failed
        # row per account and carry the `details` field forward. The outer
        # query computes per-account:
        #   - cap_count: how many cap_hits in the window (kept for reporting)
        #   - last_cap_at / last_auth_fail_at: most recent unavailability
        #     timestamps (either flips an account to 'capped' status)
        #   - last_resumed_at / last_auth_resumed_at: most recent recovery
        #     timestamps (either clears unavailability)
        # Status compares max(unavail) vs max(recover); 'failover' / 'near_cap'
        # are intentionally ignored.
        evt_rows = await db.execute_fetchall(
            'WITH latest_cap AS ( '
            '  SELECT account_name, details, '
            '         ROW_NUMBER() OVER '
            '           (PARTITION BY account_name ORDER BY created_at DESC) AS rn '
            '    FROM account_events '
            "   WHERE event_type = 'cap_hit' AND created_at >= ? "
            '), '
            'latest_auth_fail AS ( '
            '  SELECT account_name, details, '
            '         ROW_NUMBER() OVER '
            '           (PARTITION BY account_name ORDER BY created_at DESC) AS rn '
            '    FROM account_events '
            "   WHERE event_type = 'auth_failed' AND created_at >= ? "
            ') '
            'SELECT ae.account_name, '
            "       SUM(CASE WHEN ae.event_type = 'cap_hit' THEN 1 ELSE 0 END) AS cap_count, "
            "       MAX(CASE WHEN ae.event_type = 'cap_hit' THEN ae.created_at END) AS last_cap_at, "
            "       MAX(CASE WHEN ae.event_type = 'resumed' THEN ae.created_at END) AS last_resumed_at, "
            "       MAX(CASE WHEN ae.event_type = 'auth_failed' THEN ae.created_at END) AS last_auth_fail_at, "
            "       MAX(CASE WHEN ae.event_type = 'auth_resumed' THEN ae.created_at END) AS last_auth_resumed_at, "
            '       lc.details AS last_cap_details, '
            '       laf.details AS last_auth_fail_details '
            '  FROM account_events ae '
            '  LEFT JOIN latest_cap lc '
            '    ON ae.account_name = lc.account_name AND lc.rn = 1 '
            '  LEFT JOIN latest_auth_fail laf '
            '    ON ae.account_name = laf.account_name AND laf.rn = 1 '
            ' WHERE ae.created_at >= ? '
            ' GROUP BY ae.account_name',
            (since, since, since),
        )

        # Cap counts and status keyed by account_name
        caps: dict[str, dict] = {}
        for row in evt_rows:
            account_name = row['account_name']
            last_cap_at = row['last_cap_at']
            last_resumed_at = row['last_resumed_at']
            last_auth_fail_at = row['last_auth_fail_at']
            last_auth_resumed_at = row['last_auth_resumed_at']
            last_unavail = _max_ts(last_cap_at, last_auth_fail_at)
            last_recover = _max_ts(last_resumed_at, last_auth_resumed_at)
            # Unavailable iff most recent unavailability event has no later
            # recovery event of either kind.
            is_unavailable = last_unavail is not None and (
                last_recover is None or last_unavail > last_recover
            )
            cap_resets_at = _extract_resets_at(row['last_cap_details'])
            auth_resets_at = _extract_resets_at(row['last_auth_fail_details'])
            # `resets_at` is the resets_at paired with whichever unavailability
            # event is more recent. Surfaces ``None`` when not unavailable.
            if not is_unavailable:
                resets_at: str | None = None
            elif last_cap_at and (
                last_auth_fail_at is None or last_cap_at >= last_auth_fail_at
            ):
                resets_at = cap_resets_at
            else:
                resets_at = auth_resets_at
            caps[account_name] = {
                'cap_events': row['cap_count'] or 0,
                'last_cap': last_cap_at,  # NULL when no cap_hit in window
                'last_resumed': last_resumed_at,  # NULL when no resumed in window
                'last_auth_fail': last_auth_fail_at,
                'last_auth_resumed': last_auth_resumed_at,
                # Per-event-type resets_at fields preserved so the cross-DB
                # aggregator can pair the right one with whichever event won
                # globally.
                'cap_resets_at': cap_resets_at,
                'auth_resets_at': auth_resets_at,
                'status': 'capped' if is_unavailable else 'active',
                'resets_at': resets_at,
            }

        default_info = {
            'cap_events': 0, 'last_cap': None, 'last_resumed': None,
            'last_auth_fail': None, 'last_auth_resumed': None,
            'cap_resets_at': None, 'auth_resets_at': None,
            'status': 'active', 'resets_at': None,
        }
        result: dict[str, dict] = {}
        for row in inv_rows:
            account_name = row['account_name']
            cap_info = caps.get(account_name, default_info)
            result[account_name] = {
                'spend': row['spend'] or 0.0,
                'invocations': row['cnt'] or 0,
                'cap_events': cap_info['cap_events'],
                'last_cap': cap_info['last_cap'],
                'last_resumed': cap_info['last_resumed'],
                'last_auth_fail': cap_info['last_auth_fail'],
                'last_auth_resumed': cap_info['last_auth_resumed'],
                'cap_resets_at': cap_info['cap_resets_at'],
                'auth_resets_at': cap_info['auth_resets_at'],
                'status': cap_info['status'],
                'resets_at': cap_info['resets_at'],
            }

        # Second pass: emit accounts that have any in-window event but no
        # invocations. These are operationally significant (idle but
        # capped/auth_failed) and must be visible in the dashboard even
        # without spend.
        for account_name, cap_info in caps.items():
            if account_name not in result:
                result[account_name] = {
                    'spend': 0.0,
                    'invocations': 0,
                    'cap_events': cap_info['cap_events'],
                    'last_cap': cap_info['last_cap'],
                    'last_resumed': cap_info['last_resumed'],
                    'last_auth_fail': cap_info['last_auth_fail'],
                    'last_auth_resumed': cap_info['last_auth_resumed'],
                    'cap_resets_at': cap_info['cap_resets_at'],
                    'auth_resets_at': cap_info['auth_resets_at'],
                    'status': cap_info['status'],
                    'resets_at': cap_info['resets_at'],
                }
        return result

    return await with_db(db, _query, {})


def _max_ts(a: str | None, b: str | None) -> str | None:
    """Return the lexically greater ISO timestamp, or None when both are None."""
    if a is None:
        return b
    if b is None:
        return a
    return a if a > b else b


def _extract_resets_at(details: str | None) -> str | None:
    """Return the ISO resets_at string from a cap_hit details payload.

    Prefers the persisted ``resets_at`` field (written by shared.usage_gate
    since 2026-04-21). For older cap_hit rows — and rows written by orchestrator
    processes that haven't yet restarted to pick up the new shared code — falls
    back to parsing the reason string.
    """
    if not details:
        return None
    try:
        payload = json.loads(details)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get('resets_at')
    if isinstance(value, str) and value:
        return value
    # TRANSITIONAL FALLBACK — REMOVE ON 2026-04-28.
    # Long-running orchestrators started before 2026-04-21 still write cap_hit
    # details without the persisted resets_at field. Until they cycle naturally,
    # derive the timestamp from the reason string so the Uncaps In column isn't
    # blank. Duplicates a subset of shared.usage_gate._parse_resets_at; kept
    # minimal so drift is visible if Anthropic changes the wording.
    reason = payload.get('reason')
    if isinstance(reason, str) and reason:
        parsed = _parse_resets_from_reason(reason)
        if parsed is not None:
            return parsed.isoformat()
    return None


# TRANSITIONAL — REMOVE ON 2026-04-28 along with the fallback branch above.
def _parse_resets_from_reason(text: str) -> datetime | None:
    """Minimal mirror of shared.usage_gate._parse_resets_at.

    Handles the two formats observed in production cap_hit reason strings:
      - "resets in 3h" / "resets in 45m" / "resets in 2d"
      - "resets 7pm (Europe/London)" / "resets 3:00 AM (US/Pacific)"
    Returns None when no recognised form is present — unlike the shared parser
    we deliberately don't fall back to "1 hour from now", because a bogus
    countdown in the UI is worse than a dash.
    """
    m = re.search(r'resets\s+in\s+(\d+)\s*([hmd])', text, re.IGNORECASE)
    if m:
        amount = int(m.group(1))
        delta = {
            'h': timedelta(hours=amount),
            'm': timedelta(minutes=amount),
            'd': timedelta(days=amount),
        }[m.group(2).lower()]
        return datetime.now(UTC) + delta

    m = re.search(
        r'resets\s+(\d{1,2}(?::\d{2})?\s*[ap]m)\s*\(([^)]+)\)',
        text, re.IGNORECASE,
    )
    if m:
        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo(m.group(2).strip())
            time_str = m.group(1).strip()
            for fmt in ('%I:%M %p', '%I%p', '%I:%M%p', '%I %p'):
                try:
                    parsed_time = datetime.strptime(time_str, fmt).time()
                    break
                except ValueError:
                    continue
            else:
                return None
            now_in_tz = datetime.now(tz)
            target = now_in_tz.replace(
                hour=parsed_time.hour, minute=parsed_time.minute,
                second=0, microsecond=0,
            )
            if target <= now_in_tz:
                target += timedelta(days=1)
            return target.astimezone(UTC)
        except Exception:
            return None
    return None

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
            project_id = row['project_id']
            role = row['role']
            if project_id not in result:
                result[project_id] = {}
            if role not in result[project_id]:
                result[project_id][role] = {}
            result[project_id][role][row['model']] = row['total'] or 0.0
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
            project_id = row['project_id']
            if project_id not in data:
                data[project_id] = {}
            data[project_id][row['day']] = row['total'] or 0.0

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
    limit: int = 200,
) -> list[dict]:
    """Recent account events (cap_hit, resumed, etc.) within the window.

    Returns [{account_name, event_type, project_id, run_id, details,
               created_at}, ...] ordered by created_at DESC.
    *details* is returned as-is (string or None) — callers may parse JSON.
    *limit* caps the number of rows returned (default 200).
    """
    since = _cutoff(days)

    async def _query(db: aiosqlite.Connection) -> list[dict]:
        rows = await db.execute_fetchall(
            'SELECT account_name, event_type, project_id, run_id, '
            '       details, created_at '
            '  FROM account_events '
            ' WHERE created_at >= ? '
            ' ORDER BY created_at DESC'
            ' LIMIT ?',
            (since, limit),
        )
        return [
            {
                'account_name': row['account_name'],
                'event_type': row['event_type'],
                'project_id': row['project_id'],
                'run_id': row['run_id'],
                'details': row['details'],
                'created_at': row['created_at'],
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
    limit: int = 500,
) -> list[dict]:
    """Per-run cost breakdown grouped by task, with invocation detail.

    Returns a list of run dicts:
        [{run_id, project_id, total_cost,
          tasks: [{task_id, title, cost,
                   invocations: [{model, role, cost_usd, account_name,
                                  duration_ms, capped}]}]}]

    Invocations with NULL task_id are grouped under task_id=None with title=None.
    LEFT JOIN with task_results provides task titles.
    *limit* caps the number of invocation rows fetched from SQL (default 500).
    """
    since = _cutoff(days)

    async def _query(db: aiosqlite.Connection) -> list[dict]:
        try:
            rows = await db.execute_fetchall(
                'SELECT i.run_id, i.project_id, i.task_id, '
                '       tr.title, '
                '       i.model, i.role, i.cost_usd, '
                '       i.account_name, i.duration_ms, i.capped '
                '  FROM invocations i '
                '  LEFT JOIN task_results tr '
                '    ON i.run_id = tr.run_id AND i.task_id = tr.task_id '
                ' WHERE i.completed_at >= ? '
                ' ORDER BY i.run_id, i.task_id, i.id'
                ' LIMIT ?',
                (since, limit),
            )
        except sqlite3.OperationalError:
            # task_results table may not exist in all project DBs
            rows = await db.execute_fetchall(
                'SELECT run_id, project_id, task_id, '
                '       NULL AS title, '
                '       model, role, cost_usd, '
                '       account_name, duration_ms, capped '
                '  FROM invocations '
                ' WHERE completed_at >= ? '
                ' ORDER BY run_id, task_id, id'
                ' LIMIT ?',
                (since, limit),
            )

        # Build nested structure: run_id → task_id → invocations
        runs: dict[str, dict] = {}
        for row in rows:
            run_id = row['run_id']
            task_id = row['task_id']
            cost_usd = row['cost_usd'] or 0.0

            if run_id not in runs:
                runs[run_id] = {
                    'run_id': run_id,
                    'project_id': row['project_id'],
                    'total_cost': 0.0,
                    'tasks': {},  # task_id → task dict (temporary)
                }
            run = runs[run_id]
            run['total_cost'] += cost_usd

            if task_id not in run['tasks']:
                run['tasks'][task_id] = {
                    'task_id': task_id,
                    'title': row['title'],
                    'cost': 0.0,
                    'invocations': [],
                }
            task = run['tasks'][task_id]
            task['cost'] += cost_usd
            task['invocations'].append({
                'model': row['model'],
                'role': row['role'],
                'cost_usd': cost_usd,
                'account_name': row['account_name'],
                'duration_ms': row['duration_ms'],
                'capped': bool(row['capped']),
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


# ===========================================================================
# Multi-DB aggregation
# ===========================================================================


async def aggregate_cost_summary(
    dbs: list[aiosqlite.Connection | None],
    *,
    days: int = 7,
) -> dict[str, dict]:
    """Merge :func:`get_cost_summary` results from multiple databases.

    Token totals are summed component-wise; per-run costs are concatenated
    across DBs and a single global p95 is computed from the merged list.
    """
    results = await asyncio.gather(*(get_cost_summary(db, days=days) for db in dbs))
    merged: dict[str, dict] = {}
    for result in results:
        for pid, info in result.items():
            if pid not in merged:
                merged[pid] = dict(info)
                merged[pid]['tokens'] = dict(info.get('tokens') or _ZERO_TOKENS)
                merged[pid]['run_costs'] = list(info.get('run_costs') or [])
            else:
                m = merged[pid]
                m['total_spend'] += info['total_spend']
                m['task_count'] += info['task_count']
                m['active_accounts'] += info['active_accounts']
                m['cap_events'] += info['cap_events']
                tc = m['task_count']
                m['avg_cost_per_task'] = m['total_spend'] / tc if tc else 0.0
                tokens_in = info.get('tokens') or _ZERO_TOKENS
                for key in _TOKEN_KEYS:
                    m['tokens'][key] = m['tokens'].get(key, 0) + tokens_in.get(key, 0)
                m['run_costs'].extend(info.get('run_costs') or [])
    # Compute p95 once per project from the merged run_costs list. Keep the
    # raw list in the dict so shape_costs can compute the global p95 by
    # concatenating across projects.
    for pid, m in merged.items():
        run_costs = sorted(m.get('run_costs') or [])
        m['run_costs'] = run_costs
        m['p95_run_cost'] = percentile(run_costs, 95) if run_costs else None
    return merged


_TOKEN_KEYS = ('input', 'output', 'cache_read', 'cache_create', 'total')
_ZERO_TOKENS = {k: 0 for k in _TOKEN_KEYS}


async def aggregate_cost_by_project(
    dbs: list[aiosqlite.Connection | None],
    *,
    days: int = 7,
) -> dict[str, list[dict]]:
    """Merge :func:`get_cost_by_project` results from multiple databases."""
    results = await asyncio.gather(*(get_cost_by_project(db, days=days) for db in dbs))
    merged: dict[str, list[dict]] = {}
    for result in results:
        for pid, models in result.items():
            if pid not in merged:
                merged[pid] = list(models)
            else:
                merged[pid].extend(models)
    return merged


async def aggregate_cost_by_account(
    dbs: list[aiosqlite.Connection | None],
    *,
    days: int = 7,
) -> dict[str, dict]:
    """Merge :func:`get_cost_by_account` across databases.

    Accounts are shared across projects (same OAuth token, same Max
    subscription), so cap state is a single global thing — even though each
    project's orchestrator writes its own events. This aggregator computes
    the global ordering from ``max(last_cap)``, ``max(last_resumed)``,
    ``max(last_auth_fail)``, and ``max(last_auth_resumed)`` across all source
    DBs. An account is ``'capped'`` iff the globally newest unavailability
    event (cap_hit *or* auth_failed) has no later recovery event (resumed
    *or* auth_resumed) in ANY DB — so a stale orchestrator that saw a cap
    and then stopped no longer masks a resumed observed by another running
    orchestrator.

    The paired ``resets_at`` travels with whichever of cap_hit / auth_failed
    is globally most recent. Spend/invocations/cap_events are summed.
    """
    results = await asyncio.gather(*(get_cost_by_account(db, days=days) for db in dbs))
    merged: dict[str, dict] = {}
    for result in results:
        for acct, info in result.items():
            if acct not in merged:
                merged[acct] = dict(info)
            else:
                m = merged[acct]
                m['spend'] += info['spend']
                m['invocations'] += info['invocations']
                m['cap_events'] += info['cap_events']
                if info['last_cap'] and (not m['last_cap'] or info['last_cap'] > m['last_cap']):
                    m['last_cap'] = info['last_cap']
                    # cap_resets_at travels with the winning last_cap (it's
                    # the resets_at of *that* cap_hit, not of an auth_fail).
                    m['cap_resets_at'] = info.get('cap_resets_at')
                info_resumed = info.get('last_resumed')
                if info_resumed and (
                    not m.get('last_resumed') or info_resumed > m['last_resumed']
                ):
                    m['last_resumed'] = info_resumed
                info_auth_fail = info.get('last_auth_fail')
                if info_auth_fail and (
                    not m.get('last_auth_fail')
                    or info_auth_fail > m['last_auth_fail']
                ):
                    m['last_auth_fail'] = info_auth_fail
                    m['auth_resets_at'] = info.get('auth_resets_at')
                info_auth_resumed = info.get('last_auth_resumed')
                if info_auth_resumed and (
                    not m.get('last_auth_resumed')
                    or info_auth_resumed > m['last_auth_resumed']
                ):
                    m['last_auth_resumed'] = info_auth_resumed

    # Recompute status globally now that last_cap, last_resumed,
    # last_auth_fail, and last_auth_resumed reflect the max across all DBs.
    # Per-DB status latch would mis-report when one stale DB saw an
    # unavailability but never saw the recovery that another DB observed
    # later.
    for m in merged.values():
        last_cap = m.get('last_cap')
        last_auth_fail = m.get('last_auth_fail')
        last_resumed = m.get('last_resumed')
        last_auth_resumed = m.get('last_auth_resumed')
        last_unavail = _max_ts(last_cap, last_auth_fail)
        last_recover = _max_ts(last_resumed, last_auth_resumed)
        if last_unavail and (not last_recover or last_unavail > last_recover):
            m['status'] = 'capped'
            # resets_at is paired with whichever event was the global
            # unavailability winner.
            if last_cap and (
                last_auth_fail is None or last_cap >= last_auth_fail
            ):
                m['resets_at'] = m.get('cap_resets_at')
            else:
                m['resets_at'] = m.get('auth_resets_at')
        else:
            m['status'] = 'active'
            # resets_at is stale once we know a later recovery cleared
            # everything.
            m['resets_at'] = None
    return merged


async def aggregate_cost_by_role(
    dbs: list[aiosqlite.Connection | None],
    *,
    days: int = 7,
) -> dict[str, dict]:
    """Merge :func:`get_cost_by_role` results from multiple databases."""
    results = await asyncio.gather(*(get_cost_by_role(db, days=days) for db in dbs))
    merged: dict[str, dict] = {}
    for result in results:
        for pid, roles in result.items():
            if pid not in merged:
                merged[pid] = {}
            for role, models in roles.items():
                if role not in merged[pid]:
                    merged[pid][role] = {}
                for model, total in models.items():
                    merged[pid][role][model] = merged[pid][role].get(model, 0.0) + total
    return merged


async def aggregate_cost_trend(
    dbs: list[aiosqlite.Connection | None],
    *,
    days: int = 7,
) -> dict[str, list[dict]]:
    """Merge :func:`get_cost_trend` results from multiple databases."""
    results = await asyncio.gather(*(get_cost_trend(db, days=days) for db in dbs))
    merged: dict[str, list[dict]] = {}
    for result in results:
        for pid, series in result.items():
            if pid not in merged:
                merged[pid] = list(series)
            else:
                existing = {e['day']: e for e in merged[pid]}
                for entry in series:
                    if entry['day'] in existing:
                        existing[entry['day']]['total'] += entry['total']
                    else:
                        existing[entry['day']] = dict(entry)
                merged[pid] = sorted(existing.values(), key=lambda e: e['day'])
    return merged


async def aggregate_account_events(
    dbs: list[aiosqlite.Connection | None],
    *,
    days: int = 7,
    limit: int = 200,
) -> list[dict]:
    """Merge :func:`get_account_events` from multiple databases."""
    results = await asyncio.gather(
        *(get_account_events(db, days=days, limit=limit) for db in dbs),
    )
    combined: list[dict] = []
    for result in results:
        combined.extend(result)
    combined.sort(key=lambda e: e['created_at'], reverse=True)
    return combined[:limit]


async def aggregate_run_cost_breakdown(
    dbs: list[aiosqlite.Connection | None],
    *,
    days: int = 7,
    limit: int = 500,
) -> list[dict]:
    """Merge :func:`get_run_cost_breakdown` from multiple databases."""
    results = await asyncio.gather(
        *(get_run_cost_breakdown(db, days=days, limit=limit) for db in dbs),
    )
    combined: list[dict] = []
    for result in results:
        combined.extend(result)
    return combined
