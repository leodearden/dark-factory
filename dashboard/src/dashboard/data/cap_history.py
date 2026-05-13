"""Shared cap-interval reader for the dashboard.

Extracts ``cap_hit``/``resumed`` event pairs from ``account_events`` into
:class:`CapInterval` objects.  The Curator tab (and any future consumer) should
call :func:`read_cap_intervals` rather than inlining their own SQL; the Costs
tab keeps its own CTE query for legacy reasons (see ``costs.py`` docstring on
``get_cost_by_account``).

Public API
----------
- :class:`CapInterval` — frozen dataclass with ``account_name``, ``start``,
  ``end`` (``None`` means still capped at query time).
- :func:`read_cap_intervals` — async reader over one or more DB connections.
- :func:`merge_all_accounts_capped` — intersect intervals across all accounts.
- :func:`compute_overlap_ms` — sum capped-overlap milliseconds for a window.
- :func:`bucketise_cap_sparkline` — produce a 0/1 :class:`ChartData` sparkline.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import aiosqlite

from dashboard.data.chart_utils import ChartData
from dashboard.data.db import with_db

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# CapInterval dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CapInterval:
    """A contiguous period during which an account was capped.

    ``end=None`` means the cap was still in effect at query time (no matching
    ``resumed`` event was found within the query window).
    """

    account_name: str
    start: datetime
    end: datetime | None


# ---------------------------------------------------------------------------
# read_cap_intervals
# ---------------------------------------------------------------------------

async def read_cap_intervals(
    dbs: list[aiosqlite.Connection | None],
    *,
    days: int,
) -> list[CapInterval]:
    """Return all cap intervals across *dbs* within the last *days* days.

    Queries each DB's ``account_events`` table for ``cap_hit`` and ``resumed``
    rows in the window, partitions rows by ``account_name``, and FIFO-pairs
    them: each ``cap_hit`` consumes the next chronologically-later ``resumed``
    for the same account.  Unpaired ``cap_hit`` rows become open-ended
    intervals (``end=None``).

    Args:
        dbs: List of aiosqlite connections (``None`` entries are skipped).
        days: Look-back window in days.

    Returns:
        Flat list of :class:`CapInterval` objects across all DBs, unordered.
    """
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()

    async def _read_one(db: aiosqlite.Connection) -> list[CapInterval]:
        rows = await db.execute_fetchall(
            "SELECT account_name, event_type, created_at "
            "  FROM account_events "
            " WHERE event_type IN ('cap_hit', 'resumed') "
            "   AND created_at >= ? "
            " ORDER BY account_name, created_at",
            (cutoff,),
        )

        # Group rows by account_name preserving chronological order
        by_account: dict[str, list[tuple[str, datetime]]] = {}
        for row in rows:
            ts_str: str = row['created_at']
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            by_account.setdefault(row['account_name'], []).append(
                (row['event_type'], ts)
            )

        intervals: list[CapInterval] = []
        for account_name, events in by_account.items():
            # FIFO pairing: pending cap_hits queue up; each resumed closes the
            # oldest pending cap_hit; trailing cap_hits become open-ended.
            pending: deque[datetime] = deque()
            for event_type, ts in events:
                if event_type == 'cap_hit':
                    pending.append(ts)
                elif event_type == 'resumed' and pending:
                    start = pending.popleft()
                    intervals.append(CapInterval(account_name=account_name, start=start, end=ts))
            # Remaining pending cap_hits are open-ended
            for start in pending:
                intervals.append(CapInterval(account_name=account_name, start=start, end=None))

        return intervals

    results = await asyncio.gather(*(with_db(db, _read_one, []) for db in dbs))
    flat: list[CapInterval] = []
    for result in results:
        flat.extend(result)
    return flat


# ---------------------------------------------------------------------------
# merge_all_accounts_capped
# ---------------------------------------------------------------------------

def merge_all_accounts_capped(
    intervals: list[CapInterval],
    account_names: list[str],
) -> list[tuple[datetime, datetime | None]]:
    """Return the time windows during which ALL accounts in *account_names* were capped.

    Uses a sweep-line over interval starts and ends.  A merged window opens
    when every account in *account_names* has at least one active interval;
    it closes when any account's coverage drops to zero.

    Args:
        intervals: All known cap intervals (may include accounts not in
            *account_names*).
        account_names: Explicit set of accounts that must ALL be capped.

    Returns:
        List of ``(start, end)`` tuples where ``end=None`` means the merged
        cap window was still open at the end of the input.  Empty list when
        no such window exists.
    """
    account_set = set(account_names)
    n = len(account_set)

    if n == 0:
        return []

    # Build per-account interval lists (filter to requested accounts only)
    per_account: dict[str, list[CapInterval]] = {a: [] for a in account_set}
    for iv in intervals:
        if iv.account_name in per_account:
            per_account[iv.account_name].append(iv)

    # If any account has zero intervals it can never be capped → no overlap
    if any(len(ivs) == 0 for ivs in per_account.values()):
        return []

    # Build sweep events: (timestamp, kind, account)
    # kind: +1 = interval starts, -1 = interval ends
    # Sentinel for open-ended intervals: datetime.max (UTC)
    _MAX_DT = datetime.max.replace(tzinfo=UTC)

    events: list[tuple[datetime, int, str]] = []
    open_ended_accounts: set[str] = set()

    for account_name, ivs in per_account.items():
        for iv in ivs:
            events.append((iv.start, 1, account_name))
            end = iv.end if iv.end is not None else _MAX_DT
            if iv.end is None:
                open_ended_accounts.add(account_name)
            events.append((end, -1, account_name))

    # Sort: at equal timestamps process starts (+1) before ends (-1)
    events.sort(key=lambda e: (e[0], -e[1]))

    # Sweep
    active_count: dict[str, int] = {a: 0 for a in account_set}
    currently_all_capped = False
    window_start: datetime | None = None
    merged: list[tuple[datetime, datetime | None]] = []

    for ts, kind, account_name in events:
        active_count[account_name] += kind
        all_capped = all(active_count[a] > 0 for a in account_set)

        if all_capped and not currently_all_capped:
            # Merged window opens
            window_start = ts
            currently_all_capped = True
        elif not all_capped and currently_all_capped:
            # Merged window closes — ts is the end of the closing interval
            # The window ends when coverage drops, i.e. at this timestamp
            # (right edge exclusive-style; use ts as the endpoint)
            assert window_start is not None
            merged.append((window_start, ts))
            currently_all_capped = False
            window_start = None

    # If still inside a merged window after all events, determine end
    if currently_all_capped and window_start is not None:
        # All accounts' remaining intervals: check if they're all open-ended
        # An open-ended merged window is possible only if every account
        # contributed an open-ended interval covering the window start.
        # Simple heuristic: if every account is in open_ended_accounts,
        # the merged window is open-ended.
        if account_set <= open_ended_accounts:
            merged.append((window_start, None))
        else:
            # At least one account's last interval was closed at _MAX_DT
            # sentinel — treat the window as ending there (then discard
            # because it's the sentinel, not a real end).
            # The sweep will have processed the -1 sentinel before we exit.
            # If we're still here it means something weird happened; emit
            # with None to be safe.
            merged.append((window_start, None))

    return merged


# ---------------------------------------------------------------------------
# compute_overlap_ms
# ---------------------------------------------------------------------------

def compute_overlap_ms(
    start: datetime,
    end: datetime,
    capped: list[tuple[datetime, datetime | None]],
) -> int:
    """Return the total milliseconds in [start, end] covered by *capped* intervals.

    Args:
        start: Window start (inclusive).
        end: Window end (inclusive).
        capped: List of ``(cap_start, cap_end)`` tuples.  ``cap_end=None``
            means the cap extends past *end* (clamp to *end*).

    Returns:
        Total overlapping milliseconds, rounded to the nearest integer.
    """
    total_ms = 0
    for c_start, c_end in capped:
        effective_end = end if c_end is None else min(end, c_end)
        effective_start = max(start, c_start)
        if effective_end > effective_start:
            total_ms += int((effective_end - effective_start).total_seconds() * 1000)
    return total_ms


# ---------------------------------------------------------------------------
# bucketise_cap_sparkline
# ---------------------------------------------------------------------------

def bucketise_cap_sparkline(
    capped: list[tuple[datetime, datetime | None]],
    *,
    bucket_seconds: int = 600,
    window_hours: int = 24,
    now: datetime | None = None,
) -> ChartData:
    """Produce a 0/1 step-series showing when all accounts were capped.

    Generates buckets from ``now - window_hours`` to ``now`` at
    ``bucket_seconds`` resolution.  Each bucket is sampled at its *right edge*:
    ``1`` if any interval in *capped* is active at that timestamp, ``0``
    otherwise.

    Args:
        capped: List of ``(start, end)`` tuples (``end=None`` = still open).
        bucket_seconds: Bucket width in seconds (default 600 = 10 min).
        window_hours: Look-back window in hours (default 24).
        now: Reference time; defaults to ``datetime.now(UTC)``.

    Returns:
        :class:`ChartData` with ``labels`` (ISO right-edge timestamps) and
        ``values`` (list of 0 or 1).
    """
    effective_now = now if now is not None else datetime.now(UTC)
    start_at = effective_now - timedelta(hours=window_hours)
    num_buckets = (window_hours * 3600) // bucket_seconds

    labels: list[str] = []
    values: list[int | float] = []

    for i in range(num_buckets):
        right_edge = start_at + timedelta(seconds=bucket_seconds * (i + 1))
        label = right_edge.isoformat()
        value = 1 if any(
            c_start <= right_edge and (c_end is None or right_edge <= c_end)
            for c_start, c_end in capped
        ) else 0
        labels.append(label)
        values.append(value)

    return {'labels': labels, 'values': values}
