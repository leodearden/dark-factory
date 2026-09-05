#!/usr/bin/env python3
"""Reproduce the merge-lane throughput baseline from an orchestrator runs.db.

Codifies the § Background baseline queries of `plans/merge-lane-throughput-prd.md`
as one runnable report, so the PRD's downstream decompositions read one table
instead of each hand-writing its own SQL (and each getting a slightly different
answer).

Sections: landings/day, the four-segment lead-time split, verify duration by
runner, remote-host occupancy (three estimators, side by side), heartbeat queue
depth, merge_attempt/merge_finalized outcome mixes, and — behind flags — the
speculation and merge-ahead-chain sections.

STRICTLY READ-ONLY.  Every connection is a `mode=ro` SQLite URI
(:func:`_connect_ro`); this script writes nothing, files nothing, and emits no
events.  It is safe to run against a live store while the orchestrator is
merging.

WHAT THIS DOES NOT DO: it asserts no numeric target.  It reproduces a *dated*
baseline and prints what it measures.  `--window 14d` resolves relative to the
current clock, so it covers a different fortnight every day and cannot
reproduce a table whose header carries a fixed date; `--window <iso>..<iso>` is
the mechanism for that.  Each section is therefore labelled with the concrete
resolved window it was computed over, because the PRD's § Background table is
not single-windowed: most rows are 14d and four are explicitly 30d.  Do not
"fix" the script until a 30d run matches a 14d row — that would corrupt the
baseline the downstream decompositions inherit.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

# `<N>d`, N a positive integer.  Anchored: '14' and '14x' must both be
# rejected rather than silently truncated to 14 days.
_RELATIVE_RE = re.compile(r'^(\d+)d$')

_RANGE_SEP = '..'


def _iso(dt: datetime) -> str:
    """Format *dt* in the exact ISO-8601 spelling the events table stores.

    `event_store.py::EventStore.emit` writes
    ``datetime.now(UTC).isoformat()``, which renders the offset as ``+00:00``
    (never ``Z``).  Because that column is TEXT and the spelling is fixed, the
    result of this function can be used directly as a SQL string comparand and
    the comparison is a correct chronological one.
    """
    return dt.astimezone(UTC).isoformat()


def _parse_endpoint(text: str, spec: str) -> datetime:
    """Parse one ISO-8601 endpoint of a `<iso>..<iso>` window spec.

    A naive endpoint is interpreted as UTC — operators write the PRD's dated
    bounds both ways, and silently treating a naive endpoint as *local* time
    would shift the window by the host's offset without saying so.
    """
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f'bad --window {spec!r}: endpoint {text!r} is not ISO-8601 '
            f'({exc}). Expected e.g. 2026-08-20T16:10:00+00:00, or <N>d.'
        ) from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_window(spec: str, now: datetime) -> tuple[datetime, datetime]:
    """Resolve a `--window` *spec* against an injected *now* into `(lo, hi)`.

    Two forms:
      ``<N>d``          -> ``(now - N days, now)``
      ``<iso>..<iso>``  -> exactly those two instants

    Both endpoints of the returned pair are tz-aware UTC.  *now* is a
    parameter, never `datetime.now()` read inside, so every caller (and every
    test) fixes the clock explicitly.

    Raises :class:`argparse.ArgumentTypeError` — echoing the offending spec —
    for an empty, malformed, zero-length or reversed window.  A reversed range
    is rejected rather than silently swapped: it far more often means the
    operator pasted the bounds backwards than that they wanted the same window.
    """
    relative = _RELATIVE_RE.match(spec)
    if relative:
        days = int(relative.group(1))
        if days <= 0:
            raise argparse.ArgumentTypeError(
                f'bad --window {spec!r}: window must span at least one day.'
            )
        return (now - timedelta(days=days), now)

    if _RANGE_SEP in spec:
        parts = spec.split(_RANGE_SEP)
        if len(parts) != 2 or not all(p.strip() for p in parts):
            raise argparse.ArgumentTypeError(
                f'bad --window {spec!r}: the dated form takes exactly two '
                f'ISO-8601 endpoints separated by "..".'
            )
        lo = _parse_endpoint(parts[0].strip(), spec)
        hi = _parse_endpoint(parts[1].strip(), spec)
        if lo >= hi:
            raise argparse.ArgumentTypeError(
                f'bad --window {spec!r}: start {_iso(lo)} is not before end '
                f'{_iso(hi)}.'
            )
        return (lo, hi)

    raise argparse.ArgumentTypeError(
        f'bad --window {spec!r}: expected "<N>d" (e.g. 14d, relative to now) '
        f'or "<iso>..<iso>" (e.g. '
        f'2026-08-20T16:10:00+00:00..2026-09-03T16:10:00+00:00, which is how '
        f'a dated baseline is reproduced exactly).'
    )


# ---------------------------------------------------------------------------
# I/O rim — the ONLY part of this module that touches a database or a clock.
# Everything below the rim is a pure function over event dicts.
# ---------------------------------------------------------------------------


def _connect_ro(path: str | Path) -> sqlite3.Connection:
    """Open *path* strictly read-only, via a `mode=ro` SQLite URI.

    `mode=ro` (rather than a bare `sqlite3.connect`) is the house convention
    for a script that reads a LIVE store — see
    `scripts/audit_wiped_metadata_files.py`, `scripts/census_tagger_debris.py`
    and `scripts/scan_task_toolcall_leaks.py`. It buys two things a bare
    connect does not: a write attempted through this connection fails loudly
    instead of mutating an orchestrator's event store, and a typo'd path
    raises rather than silently creating an empty database that would then
    report every section as "no data".
    """
    uri = f'file:{Path(path).resolve()}?mode=ro'
    return sqlite3.connect(uri, uri=True)


def load_events(
    conn: sqlite3.Connection, event_type: str, lo: datetime, hi: datetime
) -> list[dict[str, Any]]:
    """Load `event_type` rows whose timestamp falls in ``[lo, hi)``.

    Returns a list of ``{'timestamp', 'task_id', 'data'}`` dicts ordered by
    ``id`` (insertion order), with ``data`` parsed from JSON.

    The window is half-open — ``lo`` inclusive, ``hi`` exclusive — so
    consecutive windows tile without double-counting the boundary row.  Bounds
    are compared as strings against the TEXT `timestamp` column using
    :func:`_iso`; that is a correct chronological comparison because
    `event_store.py::EventStore.emit` writes one fixed ISO-8601 spelling
    (always UTC, always a `+00:00` offset, never `Z`), which sorts
    lexicographically. The predicate hits `idx_events_type`.

    Malformed JSON, a NULL payload, or a payload that parses to a non-object
    all degrade to an empty dict rather than raising: `emit` is
    fire-and-forget, so a truncated or corrupt row is possible and must not
    abort the read of every other row in the window.
    """
    rows = conn.execute(
        'SELECT timestamp, task_id, data FROM events '
        'WHERE event_type=? AND timestamp>=? AND timestamp<? ORDER BY id',
        (event_type, _iso(lo), _iso(hi)),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for ts, task_id, data in rows:
        parsed: Any = {}
        if data:
            try:
                parsed = json.loads(data)
            except (json.JSONDecodeError, TypeError, ValueError):
                parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        out.append({'timestamp': ts, 'task_id': task_id, 'data': parsed})
    return out


# ---------------------------------------------------------------------------
# Shared numeric helpers.
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], pct: float) -> float | None:
    """Return the *pct*-th percentile of *values*, or ``None`` when empty.

    Linear interpolation between the two nearest order statistics (the
    ``numpy.percentile`` default, and what `statistics.quantiles` approximates)
    on the ascending sort of *values*: with ``k = (n - 1) * pct / 100``, the
    result is ``s[floor(k)] + (k - floor(k)) * (s[ceil(k)] - s[floor(k)])``.

    Returns ``None`` — never ``0.0`` — for an empty series.  `scripts/
    mint_hard_v2_fixtures.py::_percentile` returns 0.0 there; that would render
    "no laptop verify ran in this window" as a p50 of zero minutes, which is
    the single most consequential misreading this report can produce.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * (pct / 100.0)
    lo_i = math.floor(k)
    hi_i = math.ceil(k)
    if lo_i == hi_i:
        return float(ordered[lo_i])
    return float(ordered[lo_i] + (k - lo_i) * (ordered[hi_i] - ordered[lo_i]))


def _parse_ts(raw: str | None) -> datetime | None:
    """Parse an events-table `timestamp` into a tz-aware UTC datetime.

    A row whose timestamp is absent or unparseable yields ``None`` and is
    skipped by its caller rather than aborting the section — same
    fire-and-forget rationale as :func:`load_events`'s JSON degradation.
    """
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


# ---------------------------------------------------------------------------
# Section: landings per day.
# ---------------------------------------------------------------------------


def compute_landings_per_day(
    finalized_events: Sequence[dict[str, Any]], lo: datetime, hi: datetime
) -> dict[str, Any]:
    """Landings per UTC calendar day over the COMPLETE day buckets in the window.

    A landing is a `merge_finalized` row whose ``data['state'] == 'done'``.
    The other terminal states — 'blocked', 'superseded', 'conflict',
    'unknown_branch', 'abandoned' (see `merge_types.py::MergeOutcome.status`,
    plus 'abandoned' which `event_store.py` documents on this payload) — are
    NOT landings and are reported separately by :func:`compute_mixes`.

    Bucketing.  When *lo* falls mid-day, that leading day is a partial bucket
    and is DROPPED.  The trailing day is KEPT even when *hi* falls mid-day, and
    is named in ``partial_trailing_day`` so its under-count is legible instead
    of silent — see the asymmetry note below.  Interior days with no landing
    are zero-filled, so a quiet day pulls the median down instead of vanishing
    from the series.

    THE LEADING/TRAILING ASYMMETRY, stated plainly because it is a definitional
    choice and not an obvious one: dropping BOTH partial buckets is the more
    symmetric rule, but it is not the rule the PRD's § Background table was
    computed under, and this script exists to reproduce that table.  Over the
    dated window below, drop-both gives median 13.0 over 13 buckets — a
    different table from the one every downstream decomposition inherits.  So
    the trailing bucket is kept and LABELLED: a caller reading a live
    ``--window 14d`` run (where *hi* is "now", so the last bucket is always a
    day in progress) must read the final ``per_day`` entry as partial.

    EMPIRICALLY PINNED (plan decision 1), and re-measured on this branch.  Run
    against the live dark_factory store over the PRD's dated window
    2026-08-20T16:10:00+00:00..2026-09-03T16:10:00+00:00, this definition
    reproduces the PRD § Background row EXACTLY: median 12.0, max 27 over 14
    buckets (Aug 21 .. Sep 3).  Each neighbouring definition gives a different
    table on the same rows:
      - counting every `merge_finalized` state:  median 19.0, max 75
      - keeping the partial leading bucket:      median 11,   max 27 (15 buckets)
      - dropping the partial trailing bucket:    median 13.0, max 27 (13 buckets)
    The PRD's ``12.0`` is a ``.0`` float, i.e. the mean of the two middle values
    of an EVEN-length series, which only an even bucket count can produce.

    Returns ``{'per_day': {iso_date: count}, 'median': float | None,
    'max': int | None, 'n_days': int, 'partial_trailing_day': str | None}``.
    ``median``/``max`` are ``None`` — not ``0`` — when the window holds no
    bucket at all: "too short to contain a whole day" is a different finding
    from "a whole day passed with no landing".
    """
    at_midnight = time(0, 0, tzinfo=UTC)
    first_day = lo.date() if lo.timetz() == at_midnight else (
        lo.date() + timedelta(days=1)
    )
    # `hi` is exclusive, so a `hi` at exactly midnight completes the PREVIOUS
    # day and opens no new bucket; a mid-day `hi` leaves its own day open, and
    # that day is kept but reported as partial.
    partial_trailing_day: str | None = None
    if hi.timetz() == at_midnight:
        last_day = hi.date() - timedelta(days=1)
    else:
        last_day = hi.date()
        partial_trailing_day = last_day.isoformat()

    if first_day > last_day:
        return {'per_day': {}, 'median': None, 'max': None, 'n_days': 0,
                'partial_trailing_day': None}

    per_day: dict[str, int] = {}
    day = first_day
    while day <= last_day:
        per_day[day.isoformat()] = 0
        day += timedelta(days=1)

    for event in finalized_events:
        if event.get('data', {}).get('state') != 'done':
            continue
        ts = _parse_ts(event.get('timestamp'))
        if ts is None:
            continue
        key = ts.date().isoformat()
        if key in per_day:
            per_day[key] += 1

    counts = list(per_day.values())
    return {
        'per_day': per_day,
        'median': _percentile([float(c) for c in counts], 50),
        'max': max(counts),
        'n_days': len(counts),
        'partial_trailing_day': partial_trailing_day,
    }


def _series(values: Sequence[float]) -> dict[str, Any]:
    """Summarise a duration series as p50/p90/min/max/n, all ``None`` if empty."""
    return {
        'p50': _percentile(values, 50),
        'p90': _percentile(values, 90),
        'min': min(values) if values else None,
        'max': max(values) if values else None,
        'n': len(values),
    }


def _by_task(events: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Index events by their ``task_id`` COLUMN, preserving row order.

    The COLUMN is the only join key available: `merge_queued` / `merge_dequeued`
    carry just ``{branch, queue_depth}``, with no ``request_id`` (that key
    exists on `merge_finalized` alone), so a request-scoped join is impossible.
    Rows with a NULL task_id (e.g. `merge_heartbeat`) are dropped.
    """
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        task_id = event.get('task_id')
        if task_id is None:
            continue
        index[str(task_id)].append(event)
    return index


def _last_before(
    events: Sequence[dict[str, Any]], cutoff: datetime
) -> tuple[datetime, dict[str, Any]] | None:
    """The latest event strictly before *cutoff*, with its parsed timestamp."""
    best: tuple[datetime, dict[str, Any]] | None = None
    for event in events:
        ts = _parse_ts(event.get('timestamp'))
        if ts is None or ts >= cutoff:
            continue
        if best is None or ts > best[0]:
            best = (ts, event)
    return best


def _first_at_or_after(
    events: Sequence[dict[str, Any]], cutoff: datetime
) -> tuple[datetime, dict[str, Any]] | None:
    """The earliest event at or after *cutoff*, with its parsed timestamp."""
    best: tuple[datetime, dict[str, Any]] | None = None
    for event in events:
        ts = _parse_ts(event.get('timestamp'))
        if ts is None or ts < cutoff:
            continue
        if best is None or ts < best[0]:
            best = (ts, event)
    return best


# ---------------------------------------------------------------------------
# Section: the four-segment lead-time split.
# ---------------------------------------------------------------------------


def compute_lead_time(
    queued_events: Sequence[dict[str, Any]],
    dequeued_events: Sequence[dict[str, Any]],
    verify_events: Sequence[dict[str, Any]],
    finalized_events: Sequence[dict[str, Any]],
    lo: datetime,
    hi: datetime,
) -> dict[str, Any]:
    """Split merge lead time into queue wait, verify, and a finalize+CAS residual.

        merge_queued -> merge_dequeued -> sum(merge_verify) -> merge_finalized

    THE JOIN (plan decision 2, validated live).  For each landing — a
    `merge_finalized` row with ``data['state'] == 'done'`` — the lead-time
    origin is the **last** `merge_queued` row for that task_id **strictly
    before** the landing.  Not the first: a task re-enters the queue on
    gate_retry, cas_retry and supersede (dark_factory saw 440 `merge_queued`
    rows against 200 landings in one 14-day window), so a first-queued join
    measures "time since this task first attempted to merge", which is a
    different and much larger quantity — live it gave p50/p90 137.8/3114.8
    minutes against the correct 50.1/171.3.  And strictly before, because a
    task re-queued *after* it landed would otherwise yield a negative lead.

    Neither `merge_queued` nor `merge_dequeued` carries an in-payload
    timestamp, so every duration here is a difference of row ``timestamp``
    columns.  Verify time is the sum of ``data['duration_ms']`` over the task's
    in-window `merge_verify` rows — NOT the events table's ``duration_ms``
    COLUMN, which `event_store.py` leaves NULL for this event type.

    Segment availability differs per task, so each series carries its own ``n``:
      lead      every matched landing
      wait      landings with a `merge_dequeued` at or after the joined queue row
      verify    landings with at least one in-window `merge_verify` row — a task
                with none is EXCLUDED rather than contributing 0.0, which would
                read as an instantaneous verify
      residual  lead - wait - verify, over landings that have a wait (without
                one the remainder is not attributable to finalize+CAS).  A
                missing verify counts as 0 there, so a task whose verify rows
                fell outside the window inflates its residual.

    WINDOW EDGE.  A landing early in the window is often `unmatched` only
    because its `merge_queued` row predates *lo*; that is a truncation
    artefact, not evidence the task was never queued.  The resolved bounds are
    returned so a caller can say so.

    RE-MEASURED on this branch against the live dark_factory store over the
    PRD's dated window 2026-08-20T16:10:00+00:00..2026-09-03T16:10:00+00:00
    (n=199 matched landings, 1 unmatched at the window's leading edge):
    lead p50/p90 = 49.9/172.0 min against the PRD's 50.0/171.7, and queue wait
    p50/p90 = 0.0/110.7 against the PRD's "~0 / 110.4".  Residual p50/p90 =
    1.8/19.6, verify p50/p90 = 41.3/70.3.

    Returns ``{'lead', 'wait', 'verify', 'residual', 'matched', 'unmatched',
    'unmatched_task_ids', 'window'}``; the four series are
    :func:`_series` dicts.
    """
    queued_by_task = _by_task(queued_events)
    dequeued_by_task = _by_task(dequeued_events)
    verify_by_task = _by_task(verify_events)

    lead: list[float] = []
    wait: list[float] = []
    verify: list[float] = []
    residual: list[float] = []
    matched = 0
    unmatched_task_ids: list[str] = []

    for event in finalized_events:
        if event.get('data', {}).get('state') != 'done':
            continue
        task_id = event.get('task_id')
        finalized_ts = _parse_ts(event.get('timestamp'))
        if task_id is None or finalized_ts is None:
            continue
        task_id = str(task_id)

        joined = _last_before(queued_by_task.get(task_id, []), finalized_ts)
        if joined is None:
            unmatched_task_ids.append(task_id)
            continue
        queued_ts, _ = joined
        matched += 1

        lead_min = (finalized_ts - queued_ts).total_seconds() / 60.0
        lead.append(lead_min)

        verify_min: float | None = None
        verify_rows = [
            e for e in verify_by_task.get(task_id, [])
            if isinstance(e.get('data', {}).get('duration_ms'), (int, float))
        ]
        if verify_rows:
            verify_min = sum(
                float(e['data']['duration_ms']) for e in verify_rows
            ) / 60_000.0
            verify.append(verify_min)

        dequeued = _first_at_or_after(dequeued_by_task.get(task_id, []), queued_ts)
        if dequeued is not None:
            wait_min = (dequeued[0] - queued_ts).total_seconds() / 60.0
            wait.append(wait_min)
            residual.append(lead_min - wait_min - (verify_min or 0.0))

    return {
        'lead': _series(lead),
        'wait': _series(wait),
        'verify': _series(verify),
        'residual': _series(residual),
        'matched': matched,
        'unmatched': len(unmatched_task_ids),
        'unmatched_task_ids': unmatched_task_ids,
        'window': (_iso(lo), _iso(hi)),
    }
