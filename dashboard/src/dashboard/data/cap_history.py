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
- :func:`summarize_accounts` — derive per-account capped/available counts from
  a list of :class:`CapInterval` with an optional ``total_accounts`` denominator.
- :func:`compute_capped_now_and_windows` — derive ``(capped_now, capped_windows)``
  from a list of :class:`CapInterval`; shared by ``api_curator`` and ``_sample_curator``.

Interval convention
-------------------
All interval bounds use **half-open semantics**: ``[start, end)`` — the start
instant is included, the end instant is excluded.  An ``end=None`` interval
extends indefinitely (equivalent to ``[start, ∞)``).  This convention is
applied consistently in :func:`merge_all_accounts_capped` (sweep events at
equal timestamps are processed as starts-before-ends, with zero-width merged
windows filtered out) and
:func:`bucketise_cap_sparkline` (right-edge sampling uses ``right_edge < end``).
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TypedDict

import aiosqlite

from dashboard.data.chart_utils import ChartData
from dashboard.data.db import with_db

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
# _to_utc
# ---------------------------------------------------------------------------


def _to_utc(dt: datetime | None) -> datetime:
    """Return a UTC-aware datetime from *dt*.

    Works for both a caller's *now* reference value (``None`` means "use the
    current instant") and for arbitrary event timestamps parsed from the
    database (always a concrete ``datetime``):

    - ``None`` → ``datetime.now(UTC)``
    - naive (no ``tzinfo``) → assumed UTC, attached via ``replace(tzinfo=UTC)``
    - tz-aware but non-UTC → converted via ``astimezone(UTC)``
    """
    effective = dt if dt is not None else datetime.now(UTC)
    if effective.tzinfo is None:
        return effective.replace(tzinfo=UTC)
    return effective.astimezone(UTC)


# ---------------------------------------------------------------------------
# read_cap_intervals
# ---------------------------------------------------------------------------


async def read_cap_intervals(
    dbs: list[aiosqlite.Connection | None],
    *,
    days: int,
    now: datetime | None = None,
) -> list[CapInterval]:
    """Return all cap intervals across *dbs* within the last *days* days.

    Queries each DB's ``account_events`` table for ``cap_hit`` and ``resumed``
    rows in the window, partitions rows by ``account_name``, and FIFO-pairs
    them: each ``cap_hit`` consumes the next chronologically-later ``resumed``
    for the same account.  Unpaired ``cap_hit`` rows become open-ended
    intervals (``end=None``).

    Args:
        dbs: List of aiosqlite connections (``None`` entries are skipped).
        days: Look-back window in days.  Must be > 0; raises
            :exc:`ValueError` otherwise.
        now: Reference instant for the look-back cutoff; defaults to
            ``datetime.now(UTC)``.  Pass an explicit timestamp to anchor the
            query to the same clock as the caller's other time computations
            (e.g. ticket bounds in ``_sample_curator``).  Any ``now`` value
            is normalized to UTC before computing the cutoff: naive datetimes
            (no ``tzinfo``) are assumed to be UTC (``replace(tzinfo=UTC)``);
            tz-aware but non-UTC datetimes are converted via
            ``astimezone(UTC)``.  This guarantees the cutoff string always
            carries a ``+00:00`` suffix and compares correctly via
            lexicographic string ordering against the tz-aware ISO strings
            stored in ``account_events.created_at``.

    Returns:
        Flat list of :class:`CapInterval` objects across all DBs, unordered.
    """
    if days <= 0:
        raise ValueError('days must be positive')
    effective_now = _to_utc(now)
    cutoff = (effective_now - timedelta(days=days)).isoformat()

    async def _read_one(db: aiosqlite.Connection) -> list[CapInterval]:
        rows = await db.execute_fetchall(
            'SELECT account_name, event_type, created_at '
            '  FROM account_events '
            " WHERE event_type IN ('cap_hit', 'resumed') "
            '   AND created_at >= ? '
            ' ORDER BY account_name, created_at',
            (cutoff,),
        )

        # Group rows by account_name preserving chronological order
        by_account: dict[str, list[tuple[str, datetime]]] = {}
        for row in rows:
            ts_str: str = row['created_at']
            # Canonicalise to UTC: naive rows (no tzinfo) assumed UTC;
            # tz-aware non-UTC rows converted via astimezone(UTC).
            ts = _to_utc(datetime.fromisoformat(ts_str))
            by_account.setdefault(row['account_name'], []).append((row['event_type'], ts))

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
    it closes when any account's coverage drops to zero.  Intervals are
    treated as half-open ``[start, end)`` — adjacent intervals that merely
    touch at a single timestamp produce no merged window.

    Args:
        intervals: All known cap intervals (may include accounts not in
            *account_names*).
        account_names: Explicit set of accounts that must ALL be capped.

    Returns:
        List of ``(start, end)`` tuples where ``end=None`` means the merged
        cap window was still open at the end of the input.  Empty list when
        no such window exists.

    Precondition:
        *intervals* must be **exhaustive** for the time window the caller
        cares about.  :func:`read_cap_intervals` only returns intervals whose
        ``cap_hit`` event falls within the look-back window, so an account
        that was already capped *before* the window began will appear to have
        zero intervals and trigger the early-exit ``[]`` return.  Callers
        should extend ``days`` to ensure any active cap is captured, or model
        pre-window caps explicitly.
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
    # kind: +1 = interval starts, -1 = interval ends.
    # Open-ended intervals (end=None) only emit a start event; their end is
    # treated as "infinity" so they never contribute a -1 event to the sweep.
    # After the sweep we check whether active intervals keep the merged window
    # open and whether those intervals are open-ended.

    events: list[tuple[datetime, int, str]] = []

    # Track per-account whether any open-ended interval is currently
    # "active" (i.e. started but no matching end event will close it).
    open_ended_by_account: dict[str, bool] = {a: False for a in account_set}

    for account_name, ivs in per_account.items():
        for iv in ivs:
            events.append((iv.start, 1, account_name))
            if iv.end is not None:
                events.append((iv.end, -1, account_name))
            else:
                # Mark that this account has an open-ended interval starting
                # at iv.start; it will never emit a -1 event.
                open_ended_by_account[account_name] = True

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
            # Merged window closes at this timestamp
            assert window_start is not None
            merged.append((window_start, ts))
            currently_all_capped = False
            window_start = None

    # If still inside a merged window after all real events, the window is
    # open-ended.  Because every closed interval emits a matching -1 event,
    # any account still active (count > 0) at this point MUST have at least
    # one open-ended interval — so open_ended_by_account[a] is True for every
    # account.  The assert catches bugs in event-emission logic.
    if currently_all_capped and window_start is not None:
        all_open = all(open_ended_by_account[a] for a in account_set)
        assert all_open, (
            'BUG: all accounts active at sweep end but not all have open-ended '
            'intervals — event-emission invariant violated'
        )
        merged.append((window_start, None))

    # Filter zero-width windows: adjacent intervals that merely touch at a
    # single timestamp produce start == end under half-open semantics.
    return [(s, e) for s, e in merged if e is None or s != e]


# ---------------------------------------------------------------------------
# compute_overlap_ms
# ---------------------------------------------------------------------------


def compute_overlap_ms(
    start: datetime,
    end: datetime,
    capped: list[tuple[datetime, datetime | None]],
) -> int:
    """Return the total milliseconds in [start, end) covered by *capped* intervals.

    Args:
        start: Window start (inclusive).
        end: Window end (exclusive).
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
    otherwise.  Intervals are treated as half-open ``[start, end)`` — a cap
    that ends exactly at a bucket right-edge does **not** mark that bucket as
    ``1``.

    Args:
        capped: List of ``(start, end)`` tuples (``end=None`` = still open).
        bucket_seconds: Bucket width in seconds (default 600 = 10 min).
        window_hours: Look-back window in hours (default 24).
        now: Reference time; defaults to ``datetime.now(UTC)``.  Any *now*
            value is normalised to UTC before computing the window: naive
            datetimes (no ``tzinfo``) are assumed to be UTC
            (``replace(tzinfo=UTC)``); tz-aware but non-UTC datetimes are
            converted via ``astimezone(UTC)``.

    Returns:
        :class:`ChartData` with ``labels`` (ISO right-edge timestamps) and
        ``values`` (list of 0 or 1).
    """
    effective_now = _to_utc(now)
    start_at = effective_now - timedelta(hours=window_hours)
    num_buckets = (window_hours * 3600) // bucket_seconds

    labels: list[str] = []
    values: list[int | float] = []

    # O(num_buckets * len(capped)).  capped is expected to be small (single-digit
    # merged windows over a 24 h sparkline); if it grows large, switch to a
    # pointer-advance sweep instead.
    for i in range(num_buckets):
        right_edge = start_at + timedelta(seconds=bucket_seconds * (i + 1))
        label = right_edge.isoformat()
        value = (
            1
            if any(
                c_start <= right_edge and (c_end is None or right_edge < c_end)
                for c_start, c_end in capped
            )
            else 0
        )
        labels.append(label)
        values.append(value)

    return {'labels': labels, 'values': values}


# ---------------------------------------------------------------------------
# summarize_accounts
# ---------------------------------------------------------------------------


class AccountsSummary(TypedDict):
    """Return type for :func:`summarize_accounts`."""

    total: int
    capped: int
    available: int
    capped_accounts: list[str]


def summarize_accounts(
    intervals: list[CapInterval],
    *,
    total_accounts: int | None = None,
) -> AccountsSummary:
    """Derive per-account availability counts from a list of cap intervals.

    Args:
        intervals: List of :class:`CapInterval` objects.
        total_accounts: True total of configured accounts (e.g.
            ``UsageGate.account_count``).  When provided it is used as the
            denominator so accounts with no recent cap events are counted as
            available.  Defaults to ``None`` which falls back to the number
            of distinct accounts observed in *intervals*.

    Returns:
        Dict with keys ``total``, ``capped``, ``available``, ``capped_accounts``
        where:

        - ``capped_accounts`` — sorted list of distinct account names that have
          at least one open-ended interval (currently capped).
        - ``total`` — ``max(total_accounts or 0, #distinct accounts in intervals)``.
          The ``max()`` guards against under-reporting when ``total_accounts`` is
          stale relative to actual cap-history.
        - ``capped`` — ``len(capped_accounts)``.
        - ``available`` — ``total - capped``.
    """
    capped_accounts = sorted({iv.account_name for iv in intervals if iv.end is None})
    all_accounts = {iv.account_name for iv in intervals}
    total = max(total_accounts or 0, len(all_accounts))
    capped = len(capped_accounts)
    available = total - capped
    return {
        'total': total,
        'capped': capped,
        'available': available,
        'capped_accounts': capped_accounts,
    }


# ---------------------------------------------------------------------------
# compute_capped_now_and_windows
# ---------------------------------------------------------------------------


def compute_capped_now_and_windows(
    intervals: list[CapInterval],
    total_accounts: int | None = None,
) -> tuple[int, list[tuple[datetime, datetime | None]]]:
    """Derive (capped_now, capped_windows) from a single intervals list.

    Consolidates logic that was previously duplicated in api_curator and
    _sample_curator.  Uses asymmetric semantics:

    - *capped_now* is 1 iff the account universe is non-empty AND no account
      is currently usable (``available == 0``), where universe size =
      ``max(total_accounts, #accounts seen in intervals)``.  Pass
      ``total_accounts`` (the gate's ``account_count``) so healthy accounts
      with no recent cap events are counted as available; omit it (``None``)
      to fall back to the cap-history universe (preserves existing behaviour
      when the gate is not reachable).
    - *capped_windows* uses "all accounts simultaneously capped" semantics via
      :func:`merge_all_accounts_capped` on a **sorted** ``account_names`` list
      for deterministic merge-event ordering across reruns.

    Args:
        intervals: List of :class:`CapInterval` objects.  Both call sites
            (``api_curator``, ``_sample_curator``) already hold a
            ``list[CapInterval]`` from :func:`read_cap_intervals`; the
            ``list`` annotation matches reality and avoids the overhead of
            a redundant materialisation pass.
        total_accounts: True total of configured accounts; see
            :func:`summarize_accounts`.  Defaults to ``None``.

    Returns:
        ``(capped_now, capped_windows)`` where ``capped_now`` is 0 or 1 and
        ``capped_windows`` is the output of :func:`merge_all_accounts_capped`.
    """
    summary = summarize_accounts(intervals, total_accounts=total_accounts)
    capped_now = 1 if summary['total'] > 0 and summary['available'] == 0 else 0
    account_names = sorted({iv.account_name for iv in intervals})
    capped_windows = merge_all_accounts_capped(intervals, account_names)
    return capped_now, capped_windows
