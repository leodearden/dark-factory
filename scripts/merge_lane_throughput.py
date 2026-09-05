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
import re
from datetime import UTC, datetime, timedelta

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
