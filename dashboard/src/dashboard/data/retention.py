"""The one retention policy for the dashboard's snapshot tables.

Applies to every snapshot table in burndown.db and metrics.db: raw rows are
kept for ``RAW_RETENTION``, one row per hour after that, and nothing after
``MAX_RETENTION``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import aiosqlite

RAW_RETENTION = timedelta(days=7)
MAX_RETENTION = timedelta(days=90)

# Single percent signs: this text reaches SQLite verbatim, never %-formatted,
# and '%%' would make every row's bucket the same constant (task 6606).
_HOUR_BUCKET = "strftime('%Y-%m-%dT%H', ts)"


@dataclass(frozen=True, slots=True)
class SqlPredicate:
    """A SQL boolean expression and the parameters it binds."""

    sql: str
    params: tuple[object, ...]


async def apply_retention(
    conn: aiosqlite.Connection,
    table: str,
    *,
    now: datetime,
    partition_by: tuple[str, ...] = (),
    prefer: SqlPredicate | None = None,
) -> None:
    """Compact *table* as of *now*; the caller owns the transaction.

    Rows older than ``RAW_RETENTION`` keep ONE row per (*partition_by*...,
    hour): the first row matching *prefer* when there is one, else the
    newest. Rows older than ``MAX_RETENTION`` are deleted. Does not commit.
    """
    _require_identifiers(table, partition_by)
    hourly_before = (now - RAW_RETENTION).isoformat()
    expire_before = (now - MAX_RETENTION).isoformat()
    partition = ', '.join((*partition_by, _HOUR_BUCKET))
    order = 'ts DESC'
    prefer_params: tuple[object, ...] = ()
    if prefer is not None:
        order = f'CASE WHEN {prefer.sql} THEN 0 ELSE 1 END, {order}'
        prefer_params = prefer.params

    await conn.execute(
        f"""
        DELETE FROM {table}
        WHERE ts < ?
          AND rowid NOT IN (
              SELECT keep_rowid FROM (
                  SELECT rowid AS keep_rowid, ROW_NUMBER() OVER (
                      PARTITION BY {partition}
                      ORDER BY {order}
                  ) AS rn
                  FROM {table}
                  WHERE ts < ?
              )
              WHERE rn = 1
          )
        """,
        (hourly_before, *prefer_params, hourly_before),
    )
    await conn.execute(f'DELETE FROM {table} WHERE ts < ?', (expire_before,))


def _require_identifiers(table: str, partition_by: tuple[str, ...]) -> None:
    for name in (table, *partition_by):
        if not name.isidentifier():
            raise ValueError(
                f'apply_retention on table {table!r}: {name!r} is not a SQL identifier'
            )
