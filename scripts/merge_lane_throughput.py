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
import sys
from collections import Counter, defaultdict
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


# Explicit bucket label for a payload key that is absent or None. Never
# silently dropped and never merged into a real value's bucket: an unexpected
# vocabulary shows up in the report instead of quietly vanishing.
UNKNOWN = '(unknown)'


def _duration_minutes(data: dict[str, Any]) -> float | None:
    """Minutes from a payload's ``duration_ms``, or ``None`` when unusable.

    `merge_verify` stores its duration in the JSON payload; the events table's
    ``duration_ms`` COLUMN is NULL for this event type (`event_store.py`
    populates that column only for the invocation-shaped events).  Reading the
    column instead is silently wrong rather than loudly wrong, which is why it
    is never touched here.
    """
    raw = data.get('duration_ms')
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw) / 60_000.0


# ---------------------------------------------------------------------------
# Section: verify duration by runner.
# ---------------------------------------------------------------------------


def compute_verify_by_runner(
    verify_events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Bucket `merge_verify` rows by ``data['runner']``: p50/p90/n/pass rate.

    Emitted by `verify_runner.py::VerifyRunnerPool.dispatch`; ``runner`` is the
    host name that actually ran the verify ('local' for the orchestrator host,
    otherwise a configured remote).  Durations come from
    ``data['duration_ms']`` — see :func:`_duration_minutes` for why the column
    is not read.

    A runner with no rows in the window is ABSENT from the result, never
    reported as zeros: "the laptop ran no verify in this window" and "the
    laptop verified instantly and never passed" are different findings and must
    not render identically.  A row whose ``runner`` is missing or None buckets
    under :data:`UNKNOWN` rather than being dropped, so an unexpected value
    surfaces.  Rows with no usable duration still count toward ``n`` and the
    pass rate — only ``n_durations`` and the percentiles exclude them.

    FORWARD COMPAT — ``fallback_reason``.  That key is introduced by the PRD's
    task C, which is DOWNSTREAM of this script, so on main no row carries it.
    The top-level ``fallback_key_present`` flag says whether ANY row in the
    window carried it; when it is False the per-runner ``fallback_reasons``
    tallies are empty *because the key does not exist yet*, which a caller must
    render as "not present" rather than as a zero count.  Collapsing the two
    would let a pre-task-C window read as positive evidence that busy-fallback
    dispatches never happen.

    Returns ``{'runners': {name: {...}}, 'fallback_key_present': bool}``.
    """
    durations: dict[str, list[float]] = defaultdict(list)
    passes: dict[str, list[bool]] = defaultdict(list)
    fallbacks: dict[str, Counter[str]] = defaultdict(Counter)
    fallback_key_present = False

    for event in verify_events:
        data = event.get('data', {})
        runner = data.get('runner') or UNKNOWN
        passes[runner].append(bool(data.get('passed')))
        minutes = _duration_minutes(data)
        if minutes is not None:
            durations[runner].append(minutes)
        reason = data.get('fallback_reason')
        if reason is not None:
            fallback_key_present = True
            fallbacks[runner][str(reason)] += 1

    runners: dict[str, Any] = {}
    for runner, outcomes in passes.items():
        series = _series(durations.get(runner, []))
        runners[runner] = {
            **series,
            'n': len(outcomes),
            'n_durations': series['n'],
            'pass_rate': sum(outcomes) / len(outcomes),
            'fallback_reasons': dict(fallbacks.get(runner, Counter())),
        }
    return {'runners': runners, 'fallback_key_present': fallback_key_present}


# ---------------------------------------------------------------------------
# Section: remote-host occupancy — three estimators, deliberately unreconciled.
# ---------------------------------------------------------------------------


def compute_occupancy(
    heartbeat_events: Sequence[dict[str, Any]],
    verify_events: Sequence[dict[str, Any]],
    lo: datetime,
    hi: datetime,
) -> dict[str, Any]:
    """Estimate per-host verify-slot occupancy THREE independent ways.

    `merge_heartbeat` carries ``data['hosts']``, a list of
    ``{name, is_local, slot_state, quarantined, ...}`` (see
    `verify_runner.py::_SLOT_WIRE`); ``slot_state`` is one of 'free', 'busy',
    'parked' or None, and ``hosts`` is ``[]`` before the allocator has ever
    dispatched.

    The three estimators, each reported per host and NEVER reconciled:

    ``locf_busy_fraction``
        Last-observation-carried-forward integral: each sample holds its
        ``slot_state`` until that host's next sample, and the final sample is
        carried to *hi*.  The denominator is the host's OBSERVED span — from
        its first sample to *hi*, reported as ``observed_span_minutes`` — not
        the whole window, so a host that appeared halfway through is not
        charged as idle for hours nobody looked at it.
    ``raw_sample_fraction``
        Busy samples over total samples.  Unweighted, so it over-weights
        whatever the heartbeat cadence happened to sample densely.
    ``verify_duration_fraction``
        Sum of that host's `merge_verify` ``data['duration_ms']`` over the FULL
        window span.  ``None`` when the host ran no verify in the window.

    They use different denominators and different evidence, which is part of
    why they disagree.  DO NOT collapse them (plan decision 4, PRD
    § Decomposition A).  RE-MEASURED on this branch against the live reify
    store over the dated window
    2026-08-20T16:10:00+00:00..2026-09-03T16:10:00+00:00, the laptop's spread
    was LOCF 22.2% — the PRD's row — against raw-sample 33.4% (n=2506 samples)
    against verify-duration-sum 1.3%.  The disagreement is the signal
    the downstream decomposition needs; a single blended figure would destroy
    it.

    'parked' and None count as not-busy, but every observed state is reported
    in ``slot_states`` so a quarantine ('parked') never disappears into a bland
    "not busy".

    Returns ``{'hosts': {name: {...}}, 'n_heartbeats': int,
    'window_span_minutes': float}``.
    """
    window_span_minutes = (hi - lo).total_seconds() / 60.0

    samples: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    for event in heartbeat_events:
        ts = _parse_ts(event.get('timestamp'))
        if ts is None:
            continue
        hosts = event.get('data', {}).get('hosts')
        if not isinstance(hosts, list):
            continue
        for host in hosts:
            if not isinstance(host, dict):
                continue
            name = host.get('name')
            if not name:
                continue
            samples[str(name)].append((ts, host.get('slot_state') or UNKNOWN))

    verify_minutes: dict[str, float] = defaultdict(float)
    for event in verify_events:
        data = event.get('data', {})
        minutes = _duration_minutes(data)
        if minutes is not None and data.get('runner'):
            verify_minutes[str(data['runner'])] += minutes

    hosts_out: dict[str, Any] = {}
    for name, host_samples in samples.items():
        ordered = sorted(host_samples, key=lambda pair: pair[0])
        busy_seconds = 0.0
        for i, (ts, state) in enumerate(ordered):
            end = ordered[i + 1][0] if i + 1 < len(ordered) else hi
            if state == 'busy' and end > ts:
                busy_seconds += (end - ts).total_seconds()
        observed_span_minutes = (hi - ordered[0][0]).total_seconds() / 60.0
        n_busy = sum(1 for _, state in ordered if state == 'busy')
        host_verify = verify_minutes.get(name)
        hosts_out[name] = {
            'locf_busy_fraction': (
                busy_seconds / 60.0 / observed_span_minutes
                if observed_span_minutes > 0 else None
            ),
            'observed_span_minutes': observed_span_minutes,
            'raw_sample_fraction': n_busy / len(ordered),
            'n_samples': len(ordered),
            'n_busy_samples': n_busy,
            'verify_duration_fraction': (
                host_verify / window_span_minutes
                if host_verify is not None and window_span_minutes > 0 else None
            ),
            'verify_minutes': host_verify,
            'slot_states': dict(Counter(state for _, state in ordered)),
        }

    return {
        'hosts': hosts_out,
        'n_heartbeats': len(heartbeat_events),
        'window_span_minutes': window_span_minutes,
    }


def _depth_distribution(
    events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Histogram ``data['depth']`` with a None-safe ``int()`` coercion.

    Mirrors `analyze_speculation_depth.py::compute_per_depth`: an absent, None
    or unparseable depth is SKIPPED and counted in ``n_skipped`` rather than
    raising or landing in a 0 bucket.  Historical (pre-task-2340) `merge_verify`
    rows carry no depth at all.
    """
    counts: Counter[int] = Counter()
    skipped = 0
    for event in events:
        raw = event.get('data', {}).get('depth')
        if raw is None:
            skipped += 1
            continue
        try:
            counts[int(raw)] += 1
        except (TypeError, ValueError):
            skipped += 1
    return {
        'distribution': dict(sorted(counts.items())),
        'n_coerced': sum(counts.values()),
        'n_skipped': skipped,
    }


# ---------------------------------------------------------------------------
# Section: speculation (--speculation).
# ---------------------------------------------------------------------------


def compute_speculation(
    speculative_events: Sequence[dict[str, Any]],
    voided_events: Sequence[dict[str, Any]],
    verify_events: Sequence[dict[str, Any]],
    finalized_events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Speculation depth, chain-dead void rate, and speculative-ahead share.

    DEPTH IS TWO DIFFERENT TYPES and the distributions are kept apart.
    `merge_queue.py`'s `_emit_speculative` str-coerces every payload value, so
    ``speculative_merge.data.depth`` is a STR (``"0"``), while
    ``merge_verify.data.depth`` is a native ``int | None``.  The in-line
    warning at that call site says a future consumer aggregating depth across
    both event types must not assume one type; keeping ``speculative_depth``
    and ``verify_depth`` separate avoids the trap outright rather than casting
    and hoping.

    VOID RATE = `verdict_voided` rows with ``reason == 'chain_dead'`` over
    `speculative_merge` rows.  ``chain_dead`` is currently the only value that
    field takes, but it is filtered rather than assumed, so a new reason shows
    up as a shrinking numerator instead of being silently counted.  ``point``
    ('dispatch' vs 'adoption') is tallied alongside.

    SPECULATIVE-AHEAD SHARE = landings whose task has, strictly before the
    landing, either a `speculative_merge` row or a `merge_verify` row with
    ``speculative`` truthy — reported as ``matched``/``total`` as well as a
    share, because the share alone hides how thin the denominator can be.

    Every rate is ``None``, never ``0.0``, when its denominator is empty:
    "nothing was speculated in this window" is not "speculation was tried and
    never voided".

    Returns ``{'speculative_depth', 'verify_depth', 'void_rate',
    'n_speculative', 'n_voided_chain_dead', 'void_points',
    'speculative_ahead'}``.
    """
    n_speculative = len(speculative_events)
    chain_dead = [
        e for e in voided_events
        if e.get('data', {}).get('reason') == 'chain_dead'
    ]
    void_points = Counter(
        str(e.get('data', {}).get('point') or UNKNOWN) for e in chain_dead
    )

    speculative_by_task = _by_task(speculative_events)
    verify_by_task = _by_task(verify_events)

    total_landings = 0
    matched_landings = 0
    for event in finalized_events:
        if event.get('data', {}).get('state') != 'done':
            continue
        finalized_ts = _parse_ts(event.get('timestamp'))
        task_id = event.get('task_id')
        if finalized_ts is None or task_id is None:
            continue
        total_landings += 1
        task_id = str(task_id)
        ahead = _last_before(speculative_by_task.get(task_id, []), finalized_ts)
        if ahead is None:
            speculative_verifies = [
                e for e in verify_by_task.get(task_id, [])
                if e.get('data', {}).get('speculative')
            ]
            ahead = _last_before(speculative_verifies, finalized_ts)
        if ahead is not None:
            matched_landings += 1

    return {
        'speculative_depth': _depth_distribution(speculative_events),
        'verify_depth': _depth_distribution(verify_events),
        'n_speculative': n_speculative,
        'n_voided_chain_dead': len(chain_dead),
        'void_rate': len(chain_dead) / n_speculative if n_speculative else None,
        'void_points': dict(void_points),
        'speculative_ahead': {
            'matched': matched_landings,
            'total': total_landings,
            'share': matched_landings / total_landings if total_landings else None,
        },
    }


# ---------------------------------------------------------------------------
# Section: queue depth and the outcome/state mixes.
# ---------------------------------------------------------------------------

# Mirrors `merge_types.py::_NON_TERMINAL_OUTCOMES` — the OutcomeKind members
# for which the attempt/merge is still live. Copied as literal strings, NOT
# imported: scripts/ must stay importable without the orchestrator package
# (this module's test gate runs under `uv run --project shared`). The set is
# documented there as a FROZEN CONTRACT, already mirrored by the dashboard.
_NON_TERMINAL_OUTCOMES = frozenset({
    'cas_retry', 'gate_retry', 'post_merge_generation_chained',
    'plan_files_narrowed',
})


def compute_queue_depth(
    heartbeat_events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """p50/p90/min/max/n over `merge_heartbeat`'s ``data['depth']``.

    ``depth`` is a native int (unlike `speculative_merge`'s str depth — see
    :func:`compute_speculation`).  A heartbeat with no usable depth is skipped;
    an empty window yields ``None`` percentiles rather than a depth of 0, which
    would read as "the queue was empty" instead of "nobody was looking".
    """
    depths: list[float] = []
    for event in heartbeat_events:
        raw = event.get('data', {}).get('depth')
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        depths.append(float(raw))
    series = _series(depths)
    if depths:
        series['max'] = int(max(depths))
        series['min'] = int(min(depths))
    return series


def _tally(values: Iterable[str]) -> dict[str, Any]:
    """Counter plus shares over observed strings, with a ``total``."""
    counts = Counter(values)
    total = sum(counts.values())
    return {
        'counts': dict(counts),
        'total': total,
        'shares': {k: v / total for k, v in counts.items()} if total else {},
    }


def compute_mixes(
    attempt_events: Sequence[dict[str, Any]],
    finalized_events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Tally `merge_attempt` outcomes and `merge_finalized` states.

    Values are tallied AS OBSERVED, never validated against an imported
    vocabulary.  Two reasons: `scripts/` must import without the orchestrator
    package, and — more importantly — the vocabularies are not closed in
    practice.  `merge_types.py::OutcomeKind` is the documented outcome
    vocabulary, but `workflow.py` emits ``'superseded'`` deliberately OUTSIDE
    it; `MergeOutcome.status` is the finalize-state Literal, to which
    `event_store.py` documents ``'abandoned'`` as an addition.  A value this
    script has never seen must appear in the report, not vanish from it.  A
    row with the key missing buckets under :data:`UNKNOWN`.

    Outcomes are additionally split terminal / non-terminal (see
    :data:`_NON_TERMINAL_OUTCOMES`): ``gate_retry`` and ``cas_retry`` mean the
    attempt is still live, so folding them in with terminal outcomes makes the
    mix read as a landing tally when it is not.

    Returns ``{'attempt_outcomes': {...}, 'finalize_states': {...}}``.
    """
    outcomes = _tally(
        str(e.get('data', {}).get('outcome') or UNKNOWN) for e in attempt_events
    )
    counts: dict[str, int] = outcomes['counts']
    non_terminal = {
        k: v for k, v in counts.items() if k in _NON_TERMINAL_OUTCOMES
    }
    terminal = {k: v for k, v in counts.items() if k not in _NON_TERMINAL_OUTCOMES}
    outcomes.update({
        'non_terminal': non_terminal,
        'terminal': terminal,
        'n_non_terminal': sum(non_terminal.values()),
        'n_terminal': sum(terminal.values()),
    })
    states = _tally(
        str(e.get('data', {}).get('state') or UNKNOWN) for e in finalized_events
    )
    return {'attempt_outcomes': outcomes, 'finalize_states': states}


# ---------------------------------------------------------------------------
# Section: merge-ahead chains (--chains).
# ---------------------------------------------------------------------------


def compute_chains(
    finalized_events: Sequence[dict[str, Any]],
    verify_events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Deep merge-ahead chain landings and observed chain lengths.

    ``landed_via_chain`` IS AN INT AND IS SUMMED.  `event_store.py` documents
    it on the `merge_finalized` payload as ``1`` on an item landed by a deep
    merge-ahead chain walk and ``None`` otherwise (the task 3186 delta), and
    `scripts/merge-deep-canary-predicate.sh` — the existing consumer — pins
    that unit by SUMming it.  It is not a boolean flag, and it is not the chain
    size: reading it either of those ways gives a different number the moment a
    payload ever carries a value other than 1.  A non-numeric value is counted
    in ``n_unusable_landed_via_chain`` rather than coerced.

    Chain length comes from `merge_verify`'s ``data['chain_items']``, which is
    always >= 1 and never None — a non-chained verify is a chain of ONE, not a
    missing measurement.  ``items_per_deep_verify`` therefore averages only the
    DEEP verifies (``chain_items > 1``); including the chains of one would
    drag the figure toward 1 and hide how long real chains get.

    ``chain_landed_share`` is a real ``0.0`` when landings were observed and
    none chained — that is a measurement, and returning ``None`` there would
    imply chains went unmeasured.  It is ``None`` only when the window holds no
    landing at all.

    Returns ``{'items_landed_via_chain', 'n_chain_landings', 'n_landings',
    'chain_landed_share', 'n_unusable_landed_via_chain', 'chain_items'}``.
    """
    items = 0
    n_chain_landings = 0
    n_landings = 0
    n_unusable = 0

    for event in finalized_events:
        data = event.get('data', {})
        if data.get('state') != 'done':
            continue
        n_landings += 1
        raw = data.get('landed_via_chain')
        if raw is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            n_unusable += 1
            continue
        items += int(raw)
        n_chain_landings += 1

    chain_items: list[int] = []
    for event in verify_events:
        raw = event.get('data', {}).get('chain_items')
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        chain_items.append(int(raw))
    deep = [c for c in chain_items if c > 1]

    return {
        'items_landed_via_chain': items,
        'n_chain_landings': n_chain_landings,
        'n_landings': n_landings,
        'chain_landed_share': (
            n_chain_landings / n_landings if n_landings else None
        ),
        'n_unusable_landed_via_chain': n_unusable,
        'chain_items': {
            'distribution': dict(sorted(Counter(chain_items).items())),
            'n': len(chain_items),
            'n_deep': len(deep),
            'items_per_deep_verify': sum(deep) / len(deep) if deep else None,
        },
    }


# ---------------------------------------------------------------------------
# Multi-project rim: one runs.db per project root, one labelled bundle each.
# ---------------------------------------------------------------------------

# This checkout, used when no --project-root is passed. `scripts/` sits
# directly under the project root, so `parents[1]` is that root (the same
# spelling `scripts/mint_hard_v2_fixtures.py` and `scripts/query-events.sh`
# use).
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The always-on sections, in report order. `chains` and `speculation` are
# behind flags and appended after these.
_ALWAYS_ON_SECTIONS = (
    'landings_per_day', 'lead_time', 'verify_by_runner',
    'occupancy', 'queue_depth', 'mixes',
)


def resolve_runs_db(root: str | Path) -> Path:
    """``<root>/data/orchestrator/runs.db`` — the orchestrator event store.

    That layout is fixed by `harness.py` and carries no config key, so it is
    spelled here rather than read from `dark-factory-orchestrator.yaml`; the
    same literal appears in `scripts/audit_wiped_metadata_files.py`,
    `scripts/mint_hard_v2_fixtures.py` and `scripts/query-events.sh`.
    """
    return Path(root) / 'data' / 'orchestrator' / 'runs.db'


def resolve_project_roots(values: Sequence[str] | None) -> list[Path]:
    """Resolve the repeated ``--project-root`` values, defaulting to this checkout.

    argparse's ``append`` action leaves the destination at ``None`` (not
    ``[]``) when the flag never appears, so the default is applied here rather
    than via ``default=[...]`` — an argparse list default is shared mutable
    state that ``append`` extends rather than replaces, which would silently
    add this checkout to every explicit invocation.
    """
    if not values:
        return [DEFAULT_PROJECT_ROOT]
    return [Path(value) for value in values]


def collect_project(
    root: str | Path,
    lo: datetime,
    hi: datetime,
    *,
    chains: bool = False,
    speculation: bool = False,
) -> dict[str, Any]:
    """Read one project's runs.db and return one LABELLED result bundle.

    WHY THE FLAG IS REPEATABLE rather than a project column being read
    (plan decision 5).  The `events` table has no project/project_id column at
    all — checked against the live DDL in `event_store.py::_SCHEMA`, which is
    ten columns wide and carries none.  The only ``project_id`` in the system
    belongs to `account_events` in a DIFFERENT store
    (`shared/src/shared/cost_store.py`), whose event vocabulary is disjoint
    from the merge-lane one and which shares no join key with it.  So a
    per-project breakdown is not derivable from a single root by any query:
    the project label IS the root you read, and reading several means passing
    the flag several times.  The PRD's void-rate decomposition needs exactly
    this — reify's rate against dark_factory's, side by side and never pooled.

    Returns ``{'project_root', 'runs_db', 'window', 'error', 'sections'}``.
    On failure ``error`` is a human-readable string naming the path that was
    actually looked for and ``sections`` is EMPTY — never a bundle of zeros.
    "This store could not be read" and "this project landed nothing in the
    window" are different findings and must not render identically.
    """
    root_path = Path(root)
    db_path = resolve_runs_db(root_path)
    bundle: dict[str, Any] = {
        'project_root': str(root_path),
        'runs_db': str(db_path),
        'window': (_iso(lo), _iso(hi)),
        'error': None,
        'sections': {},
    }

    if not db_path.is_file():
        bundle['error'] = (
            f'no runs.db at {db_path} (expected '
            f'<project_root>/data/orchestrator/runs.db)'
        )
        return bundle

    try:
        conn = _connect_ro(db_path)
    except sqlite3.Error as exc:
        bundle['error'] = f'cannot open {db_path} read-only: {exc}'
        return bundle

    try:
        loaded = {
            event_type: load_events(conn, event_type, lo, hi)
            for event_type in (
                'merge_queued', 'merge_dequeued', 'merge_verify',
                'merge_finalized', 'merge_heartbeat', 'merge_attempt',
                'speculative_merge', 'verdict_voided',
            )
        }
    except sqlite3.Error as exc:
        # A file that is not a database, or a truncated one, fails on the
        # first query rather than at connect time.
        bundle['error'] = f'cannot read {db_path}: {exc}'
        return bundle
    finally:
        conn.close()

    sections: dict[str, Any] = {
        'landings_per_day': compute_landings_per_day(
            loaded['merge_finalized'], lo, hi
        ),
        'lead_time': compute_lead_time(
            loaded['merge_queued'], loaded['merge_dequeued'],
            loaded['merge_verify'], loaded['merge_finalized'], lo, hi,
        ),
        'verify_by_runner': compute_verify_by_runner(loaded['merge_verify']),
        'occupancy': compute_occupancy(
            loaded['merge_heartbeat'], loaded['merge_verify'], lo, hi
        ),
        'queue_depth': compute_queue_depth(loaded['merge_heartbeat']),
        'mixes': compute_mixes(
            loaded['merge_attempt'], loaded['merge_finalized']
        ),
    }
    if speculation:
        sections['speculation'] = compute_speculation(
            loaded['speculative_merge'], loaded['verdict_voided'],
            loaded['merge_verify'], loaded['merge_finalized'],
        )
    if chains:
        sections['chains'] = compute_chains(
            loaded['merge_finalized'], loaded['merge_verify']
        )
    bundle['sections'] = sections
    return bundle


def collect_projects(
    roots: Sequence[str | Path],
    lo: datetime,
    hi: datetime,
    *,
    chains: bool = False,
    speculation: bool = False,
) -> list[dict[str, Any]]:
    """One :func:`collect_project` bundle per root, in the order given.

    A root whose store cannot be read yields an error bundle and the remaining
    roots are still collected: a typo in one ``--project-root`` must not cost
    the operator the other project's rows.
    """
    return [
        collect_project(root, lo, hi, chains=chains, speculation=speculation)
        for root in roots
    ]


def void_rate_by_project(
    bundles: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """The chain-dead void rate PER project, keyed by project root.

    The cross-project view the ``--speculation`` section exists for: the rates
    are placed beside each other, never pooled into one numerator over one
    denominator.  Pooling is not a smaller version of the same finding — it is
    a different one, and it erases the between-project spread that motivates
    the PRD's decomposition.

    A bundle that errored is ABSENT from the result rather than present with a
    rate of 0.0: a project whose store could not be read has no measured void
    rate at all.
    """
    out: dict[str, dict[str, Any]] = {}
    for bundle in bundles:
        speculation = bundle.get('sections', {}).get('speculation')
        if bundle.get('error') is not None or speculation is None:
            continue
        out[bundle['project_root']] = {
            'n_speculative': speculation['n_speculative'],
            'n_voided_chain_dead': speculation['n_voided_chain_dead'],
            'void_rate': speculation['void_rate'],
        }
    return out


def any_bundle_failed(bundles: Sequence[dict[str, Any]]) -> bool:
    """True when any project's store could not be read — drives a non-zero exit."""
    return any(bundle.get('error') is not None for bundle in bundles)


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface: repeatable ``--project-root``, ``--window``, and flags."""
    parser = argparse.ArgumentParser(
        description=(
            'Reproduce the merge-lane throughput baseline from one or more '
            'orchestrator runs.db stores. Strictly read-only.'
        ),
    )
    parser.add_argument(
        '--project-root', action='append', metavar='PATH',
        help=(
            'Project root holding data/orchestrator/runs.db. Repeat to report '
            'several projects side by side (the events table has no project '
            'column, so the root IS the project label). '
            f'Default: {DEFAULT_PROJECT_ROOT}'
        ),
    )
    parser.add_argument(
        '--window', default='14d', metavar='SPEC',
        help=(
            '"<N>d" relative to now (default: %(default)s), or '
            '"<iso>..<iso>" to reproduce a dated baseline exactly.'
        ),
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Emit JSON keyed by project root instead of the text report.',
    )
    parser.add_argument(
        '--chains', action='store_true',
        help='Add the deep merge-ahead chain section (off by default).',
    )
    parser.add_argument(
        '--speculation', action='store_true',
        help='Add the speculation section, incl. void rate by project (off by default).',
    )
    return parser


# ---------------------------------------------------------------------------
# Rendering and CLI entry point.
# ---------------------------------------------------------------------------

# Every section header starts with this prefix and carries the CONCRETE
# RESOLVED window it was computed over. The PRD's § Background table is NOT
# single-windowed — seven rows are 14d and four are 30d — so a header without
# its window lets two rows measured over different spans be read as one
# measurement (plan decision 3).
SECTION_PREFIX = '-- '

SECTION_TITLES = {
    'landings_per_day': 'landings/day',
    'lead_time': 'lead-time split',
    'verify_by_runner': 'verify duration by runner',
    'occupancy': 'host occupancy',
    'queue_depth': 'queue depth',
    'mixes': 'outcome mixes',
    'speculation': 'speculation',
    'chains': 'merge-ahead chains',
}

VOID_RATE_BY_PROJECT_TITLE = 'void rate by project'


def _section(title: str, window: tuple[str, str] | list[str]) -> str:
    """A section header stamped with the window the section was computed over."""
    lo, hi = window
    return f'{SECTION_PREFIX}{title}  [{lo} .. {hi}] --'


def _num(value: float | None, digits: int = 1) -> str:
    """A float, or ``n/a`` — never ``0.0`` — when the series was empty."""
    return 'n/a' if value is None else f'{value:.{digits}f}'


def _count(value: int | None) -> str:
    return 'n/a' if value is None else str(value)


def _pct(value: float | None, digits: int = 1) -> str:
    return 'n/a' if value is None else f'{value * 100:.{digits}f}%'


def _rate(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:.3f}'


def _format_landings(
    section: dict[str, Any], window: tuple[str, str] | list[str]
) -> list[str]:
    lines = [_section(SECTION_TITLES['landings_per_day'], window)]
    lines.append(
        f"  median {_num(section['median'])}  max {_count(section['max'])}"
        f"  over {section['n_days']} complete UTC day bucket(s)"
    )
    if section['partial_trailing_day']:
        lines.append(
            f"  NOTE {section['partial_trailing_day']} is a PARTIAL trailing "
            f'bucket (the window ends mid-day), so its count under-reports'
        )
    lines += [f'    {day}: {count}' for day, count in section['per_day'].items()]
    return lines


def _format_lead_time(
    section: dict[str, Any], window: tuple[str, str] | list[str]
) -> list[str]:
    lines = [_section(SECTION_TITLES['lead_time'], window)]
    lines.append(
        f"  {section['matched']} landing(s) joined to the last merge_queued "
        f"strictly before them; {section['unmatched']} unmatched (queued "
        f'before the window opened)'
    )
    lines.append(
        '    segment           p50      p90      min      max      n  (minutes)'
    )
    for label, key in (
        ('lead', 'lead'), ('queue wait', 'wait'),
        ('verify', 'verify'), ('finalize+CAS', 'residual'),
    ):
        series = section[key]
        lines.append(
            f"    {label:<14}{_num(series['p50']):>7}  {_num(series['p90']):>7}"
            f"  {_num(series['min']):>7}  {_num(series['max']):>7}"
            f"  {series['n']:>5}"
        )
    return lines


def _format_verify_by_runner(
    section: dict[str, Any], window: tuple[str, str] | list[str]
) -> list[str]:
    lines = [_section(SECTION_TITLES['verify_by_runner'], window)]
    if not section['runners']:
        lines.append('  no merge_verify rows in this window')
    for name in sorted(section['runners']):
        runner = section['runners'][name]
        lines.append(
            f"    {name:<12} p50 {_num(runner['p50']):>7}"
            f"  p90 {_num(runner['p90']):>7}  n {runner['n']}"
            f" (with a duration: {runner['n_durations']})"
            f"  pass {_rate(runner['pass_rate'])}"
        )
    if section['fallback_key_present']:
        for name in sorted(section['runners']):
            reasons = section['runners'][name]['fallback_reasons']
            if reasons:
                lines.append(f'      {name} fallback_reason: {reasons}')
    else:
        # NOT "0 fallbacks": no row in this window carried the key at all, so
        # this window is no evidence either way about busy-fallback dispatch.
        lines.append(
            '      fallback_reason: key not present on any row in this window'
        )
    return lines


def _format_occupancy(
    section: dict[str, Any], window: tuple[str, str] | list[str]
) -> list[str]:
    lines = [_section(SECTION_TITLES['occupancy'], window)]
    lines.append(
        '  three estimators, deliberately unreconciled — none is authoritative'
    )
    if not section['hosts']:
        lines.append(
            f"  no merge_heartbeat host samples in this window "
            f"({section['n_heartbeats']} heartbeat(s))"
        )
    for name in sorted(section['hosts']):
        host = section['hosts'][name]
        lines.append(
            f"    {name:<12} LOCF {_pct(host['locf_busy_fraction']):>7}"
            f"  raw-sample {_pct(host['raw_sample_fraction']):>7}"
            f"  verify-duration {_pct(host['verify_duration_fraction']):>7}"
        )
        lines.append(
            f"      {host['n_samples']} sample(s), {host['n_busy_samples']} busy;"
            f" slot states {host['slot_states']}"
        )
    return lines


def _format_queue_depth(
    section: dict[str, Any], window: tuple[str, str] | list[str]
) -> list[str]:
    return [
        _section(SECTION_TITLES['queue_depth'], window),
        f"  p50 {_num(section['p50'])}  p90 {_num(section['p90'])}"
        f"  min {_count(section['min'])}  max {_count(section['max'])}"
        f"  over {section['n']} heartbeat(s)",
    ]


def _format_tally(tally: dict[str, Any], indent: str = '      ') -> list[str]:
    return [
        f"{indent}{name} {count} ({_rate(tally['shares'][name])})"
        for name, count in sorted(
            tally['counts'].items(), key=lambda kv: (-kv[1], kv[0])
        )
    ]


def _format_mixes(
    section: dict[str, Any], window: tuple[str, str] | list[str]
) -> list[str]:
    outcomes = section['attempt_outcomes']
    states = section['finalize_states']
    lines = [_section(SECTION_TITLES['mixes'], window)]
    lines.append(
        f"    merge_attempt outcomes (n={outcomes['total']}): "
        f"{outcomes['n_terminal']} terminal, {outcomes['n_non_terminal']} "
        f'non-terminal (still live — NOT landings)'
    )
    lines += _format_tally(outcomes)
    lines.append(f"    merge_finalized states (n={states['total']}):")
    lines += _format_tally(states)
    return lines


def _format_speculation(
    section: dict[str, Any], window: tuple[str, str] | list[str]
) -> list[str]:
    ahead = section['speculative_ahead']
    lines = [_section(SECTION_TITLES['speculation'], window)]
    lines.append(
        f"    speculative_merge {section['n_speculative']}, chain_dead voids "
        f"{section['n_voided_chain_dead']}, void rate "
        f"{_rate(section['void_rate'])}"
    )
    lines.append(f"      void points: {section['void_points']}")
    lines.append(
        f"      landed with speculation ahead: {ahead['matched']}/"
        f"{ahead['total']} ({_rate(ahead['share'])})"
    )
    # Two distributions, never pooled: speculative_merge.depth is a STR and
    # merge_verify.depth is a native int (see compute_speculation).
    lines.append(
        f"      speculative_merge depth (str-coerced): "
        f"{section['speculative_depth']['distribution']}"
        f" (skipped {section['speculative_depth']['n_skipped']})"
    )
    lines.append(
        f"      merge_verify depth (int): "
        f"{section['verify_depth']['distribution']}"
        f" (skipped {section['verify_depth']['n_skipped']})"
    )
    return lines


def _format_chains(
    section: dict[str, Any], window: tuple[str, str] | list[str]
) -> list[str]:
    items = section['chain_items']
    lines = [_section(SECTION_TITLES['chains'], window)]
    lines.append(
        f"    items landed via chain (SUM of landed_via_chain): "
        f"{section['items_landed_via_chain']} over "
        f"{section['n_chain_landings']} chain landing(s) of "
        f"{section['n_landings']} ({_rate(section['chain_landed_share'])})"
    )
    if section['n_unusable_landed_via_chain']:
        lines.append(
            f"      {section['n_unusable_landed_via_chain']} row(s) carried a "
            f'non-numeric landed_via_chain and were not coerced'
        )
    lines.append(
        f"      chain_items distribution: {items['distribution']};"
        f" items per deep verify: {_num(items['items_per_deep_verify'], 2)}"
        f" over {items['n_deep']} deep verify(s)"
    )
    return lines


_SECTION_FORMATTERS = {
    'landings_per_day': _format_landings,
    'lead_time': _format_lead_time,
    'verify_by_runner': _format_verify_by_runner,
    'occupancy': _format_occupancy,
    'queue_depth': _format_queue_depth,
    'mixes': _format_mixes,
    'speculation': _format_speculation,
    'chains': _format_chains,
}

# Report order. A bundle only renders the sections it actually carries, so the
# flagged ones simply do not appear unless they were requested.
_SECTION_ORDER = (*_ALWAYS_ON_SECTIONS, 'speculation', 'chains')


def format_report(bundles: Sequence[dict[str, Any]]) -> str:
    """Render collected bundles as the human-readable report.

    One block per project, labelled with the project root it was read from,
    and never pooled: the projects are separate measurements of separate merge
    lanes, and the between-project spread is a finding in its own right.

    Every section header carries the concrete resolved window (see
    :data:`SECTION_PREFIX`).  A bundle that failed renders its error where its
    sections would have been — an unreadable store must not render as a quiet
    fortnight.
    """
    if not bundles:
        return ''
    lines: list[str] = []
    for bundle in bundles:
        lines.append(f"=== project: {bundle['project_root']} ===")
        lines.append(f"runs.db: {bundle['runs_db']}")
        if bundle['error'] is not None:
            lines.append(f"ERROR: {bundle['error']}")
            lines.append('')
            continue
        for key in _SECTION_ORDER:
            section = bundle['sections'].get(key)
            if section is None:
                continue
            lines.append('')
            lines += _SECTION_FORMATTERS[key](section, bundle['window'])
        lines.append('')

    by_project = void_rate_by_project(bundles)
    if by_project:
        window = bundles[0]['window']
        lines.append(
            f'=== {VOID_RATE_BY_PROJECT_TITLE}  '
            f'[{window[0]} .. {window[1]}] ==='
        )
        for root, entry in by_project.items():
            lines.append(
                f"  {root}: {entry['n_voided_chain_dead']}/"
                f"{entry['n_speculative']} = {_rate(entry['void_rate'])}"
            )
        lines.append(
            '  (side by side, never pooled: the spread between projects is '
            'the finding)'
        )
    return '\n'.join(lines).rstrip('\n')


def main(argv: Sequence[str], now: datetime | None = None) -> int:
    """CLI entry point. Returns 0, or non-zero when a project could not be read.

    *now* is injectable so a caller (and every test) can fix the clock that
    ``--window <N>d`` resolves against; production passes nothing and gets
    ``datetime.now(UTC)``.

    Exit codes: ``2`` for a malformed ``--window`` (nothing is printed to
    stdout), ``1`` when at least one project's store could not be read — the
    other projects still report, on stdout, and every failure is named on
    stderr — and ``0`` otherwise.
    """
    args = build_parser().parse_args(list(argv))
    resolved_now = now if now is not None else datetime.now(UTC)
    try:
        lo, hi = parse_window(args.window, resolved_now)
    except argparse.ArgumentTypeError as exc:
        print(f'merge_lane_throughput: {exc}', file=sys.stderr)
        return 2

    roots = resolve_project_roots(args.project_root)
    bundles = collect_projects(
        roots, lo, hi, chains=args.chains, speculation=args.speculation
    )
    for bundle in bundles:
        if bundle['error'] is not None:
            print(
                f"merge_lane_throughput: {bundle['project_root']}: "
                f"{bundle['error']}",
                file=sys.stderr,
            )

    if args.json:
        # Keyed by project root, and carrying `void_rate_by_project` on every
        # run (empty without --speculation) so the schema is stable for the
        # downstream decompositions that consume this.
        print(json.dumps({
            'window': [_iso(lo), _iso(hi)],
            'projects': {b['project_root']: b for b in bundles},
            'void_rate_by_project': void_rate_by_project(bundles),
        }, indent=2, default=str))
    else:
        print(format_report(bundles))

    return 1 if any_bundle_failed(bundles) else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
