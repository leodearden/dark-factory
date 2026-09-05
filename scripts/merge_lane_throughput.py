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
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
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
