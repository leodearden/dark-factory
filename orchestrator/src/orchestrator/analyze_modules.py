"""Per-module conflict analysis — input for hand-curated orchestrator.yaml overrides.

Reads ``runs.db`` events (``lock_acquired`` / ``lock_released`` /
``task_skipped`` / ``merge_attempt``) and aggregates per-first-path-component:

* total dispatches
* average hold time
* conflict proxy — how often a dispatch on that module coincides with a
  skipped task that wanted the same module
* suggested ``max_per_module`` override

The recommendation is intentionally coarse: 1 for highly-conflicted modules,
2–3 for busy-but-mergeable ones, 4 for low-conflict modules.  Apply by hand
to the relevant subproject ``orchestrator.yaml`` and observe follow-on
``task_skipped`` rates.

Usage
-----

    uv run --project orchestrator python -m orchestrator.analyze_modules \\
        --since 7d /path/to/data/orchestrator/runs.db

Pass ``--json`` for machine-readable output suitable for piping into jq.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from orchestrator import hold_history

_DURATION_RE = re.compile(r'^(\d+)([smhd])$')


def _parse_since(value: str) -> datetime:
    """Parse a ``--since`` value into an absolute UTC datetime.

    Accepts ``7d``, ``24h``, ``90m``, ``3600s`` or an ISO-8601 timestamp.
    """
    m = _DURATION_RE.match(value.strip())
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = {
            's': timedelta(seconds=n),
            'm': timedelta(minutes=n),
            'h': timedelta(hours=n),
            'd': timedelta(days=n),
        }[unit]
        return datetime.now(UTC) - delta
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f'--since must be <N>{{s,m,h,d}} or ISO-8601; got {value!r}'
        ) from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


@dataclass
class ModuleStats:
    """Aggregated per-module dispatch / conflict counters."""

    dispatches: int = 0
    skipped_waiting: int = 0
    total_hold_secs: float = 0.0
    hold_samples: int = 0

    #: How many of ``hold_samples`` are right-censored LOWER BOUNDS — spans an
    #: era boundary or a double-acquire ended rather than a ``lock_released``.
    #: Counted in the mean (PARKING_MODEL_REPORT.md:102 — "the lock did block
    #: others until then"), but reported separately: a mean over mostly-censored
    #: evidence is biased low-to-unknown, and an operator reading `avg_hold_s`
    #: deserves to know which kind of evidence it rests on.
    truncated_holds: int = 0

    def conflict_rate(self) -> float:
        """Fraction of skips over dispatches — ~0 is idle, >1 is contended."""
        if self.dispatches == 0:
            return 0.0
        return self.skipped_waiting / self.dispatches

    def avg_hold_secs(self) -> float:
        if self.hold_samples == 0:
            return 0.0
        return self.total_hold_secs / self.hold_samples

    def truncated_fraction(self) -> float:
        """Share of hold samples whose end was imposed rather than observed.

        Returns 0.0 with no samples, matching :meth:`avg_hold_secs` — this is
        deliberately NOT ``HoldHistory.truncated_fraction``'s None-refusal.
        That refusal protects a *predictor* whose caller must not mistake
        "unknown" for "clean"; here the value is a column in a table, and its
        siblings on this dataclass already answer 0.0 for the empty case.
        """
        if self.hold_samples == 0:
            return 0.0
        return self.truncated_holds / self.hold_samples


def _iter_events(
    db_path: Path,
    since: datetime,
) -> Iterable[dict]:
    """Yield ``runs.db`` event rows shaped like ``EventStore`` returns them.

    Each row is a dict carrying ``id``, ``timestamp``, ``run_id``, ``task_id``,
    ``event_type`` and a JSON-decoded ``data`` — the same shape
    ``EventStore.fetch_events_by_type_all_runs`` produces, so
    :func:`hold_history.iter_hold_spans` can consume this stream directly
    instead of the CLI re-implementing acquire→release pairing (INV-5).

    ``service_restart`` rows are selected even though this module never counts
    them: they are one of the two era boundaries the span helper closes on.

    The read stays raw sqlite rather than going through ``EventStore``: the
    fetch API has no ``since`` predicate, so it would pull the whole store and
    filter in Python, discarding the one thing ``--since`` exists to do.
    """
    # Open read-only to avoid touching -wal/-shm sidecars; mirrors digest.py:242.
    conn = sqlite3.connect(f'{db_path.resolve().as_uri()}?mode=ro', uri=True)
    try:
        cur = conn.execute(
            'SELECT id, timestamp, run_id, task_id, event_type, data '
            'FROM events '
            'WHERE timestamp >= ? '
            "AND event_type IN "
            "('lock_acquired','lock_released','task_skipped','merge_attempt',"
            "'service_restart') "
            # iter_hold_spans needs ONE stream in the store's own order, or a
            # release could be applied before its acquire.
            'ORDER BY id ASC',
            (since.isoformat(),),
        )
        for row_id, ts, run_id, task_id, event_type, data in cur:
            try:
                data_dict = json.loads(data) if data else {}
            except (TypeError, ValueError):
                data_dict = {}
            yield {
                'id': row_id,
                'timestamp': ts,
                'run_id': run_id,
                'task_id': task_id,
                'event_type': event_type,
                'data': data_dict,
            }
    finally:
        conn.close()


def aggregate(db_path: Path, since: datetime) -> dict[str, ModuleStats]:
    """Scan ``db_path`` events since ``since`` and return per-module stats."""
    stats: dict[str, ModuleStats] = defaultdict(ModuleStats)

    # Materialized, not streamed: `iter_hold_spans` needs ONE stream in `id`
    # order and the dispatch/skip counters need the same rows.  A counting
    # generator would keep it lazy at the cost of making the counters valid
    # only once the span generator has been fully drained — an invariant with
    # no local evidence, bought for nothing in an offline CLI.  The window is
    # bounded by --since (default 7d); the unbounded worst case is the whole
    # store, ~26k rows on the 136MB production runs.db.
    rows = list(_iter_events(db_path, since))

    for row in rows:
        event_type = row['event_type']
        # The module keys are used VERBATIM: the payload already carries them
        # depth-coarsened at the emitting project's own `lock_depth`, which is
        # the exact key `ModuleLockTable._limit_for` and `module_overrides`
        # look up (scheduler.py:1505-1517).  Re-applying `normalize_lock` here
        # would need that project's `lock_depth`, and the CLI is handed a db
        # path with no config to read it from.  `dict.fromkeys` de-duplicates
        # within one event while preserving order.
        keys = dict.fromkeys(hold_history.modules_of(row))
        if event_type == 'lock_acquired':
            for k in keys:
                stats[k].dispatches += 1
        elif event_type == 'task_skipped':
            for k in keys:
                stats[k].skipped_waiting += 1

    # Hold accounting is NOT done here (PRD INV-5): one shared era-boundary
    # span-closing helper serves both the predictor and this report, so the two
    # cannot drift on what an acquire, a release or a lost process means.  The
    # `task_skipped` / `merge_attempt` rows in this list are safe to hand over:
    # they are not in the helper's interesting set and are skipped before
    # `run_id` is read, so they cannot fabricate an era boundary.
    for span in hold_history.iter_hold_spans(rows):
        entry = stats[span.module]
        entry.total_hold_secs += span.duration
        entry.hold_samples += 1
        if span.truncated:
            entry.truncated_holds += 1

    return dict(stats)


def suggest_max_per_module(entry: ModuleStats) -> int:
    """Pick a suggested ``max_per_module`` override for *entry*.

    Heuristic (conservative; applied by hand after review):

    * conflict_rate >= 2.0 → 1 (highly contended — serialize)
    * conflict_rate >= 0.5 → 2
    * conflict_rate >= 0.1 → 3
    * otherwise → 4
    """
    rate = entry.conflict_rate()
    if rate >= 2.0:
        return 1
    if rate >= 0.5:
        return 2
    if rate >= 0.1:
        return 3
    return 4


#: Module-column width.  A depth-2 lock module such as
#: ``fused-memory/src/fused_memory/reconciliation`` overflows the 32 this used
#: to be, and two distinct hot modules sharing a 32-char prefix rendered as the
#: SAME label — leaving an operator unable to tell which key the ``suggest``
#: column meant.  Wide enough for the longest key in the measured evidence.
_MODULE_COL = 56


def render_table(stats: dict[str, ModuleStats]) -> str:
    """Human-readable table sorted by descending conflict rate.

    ``trunc`` is how many of the hold samples behind ``avg_hold_s`` are
    right-censored lower bounds rather than clean releases.
    """
    lines = [
        f'{"module":<{_MODULE_COL}} {"dispatches":>12} {"skipped":>10} '
        f'{"conflict":>10} {"avg_hold_s":>12} {"trunc":>7} {"suggest":>8}'
    ]
    ranked = sorted(
        stats.items(),
        key=lambda kv: (-kv[1].conflict_rate(), -kv[1].dispatches),
    )
    for module, entry in ranked:
        # Deliberately NOT truncated to the column width: a clipped lock key is
        # not a key, and pasting one into `module_overrides` silently does
        # nothing.  An over-long module pushes its own row wide instead.
        lines.append(
            f'{module:<{_MODULE_COL}} {entry.dispatches:>12d} '
            f'{entry.skipped_waiting:>10d} {entry.conflict_rate():>10.2f} '
            f'{entry.avg_hold_secs():>12.1f} {entry.truncated_holds:>7d} '
            f'{suggest_max_per_module(entry):>8d}'
        )
    return '\n'.join(lines)


def render_json(stats: dict[str, ModuleStats]) -> str:
    """Machine-readable JSON for piping into jq."""
    payload = {
        module: {
            'dispatches': e.dispatches,
            'skipped_waiting': e.skipped_waiting,
            'conflict_rate': round(e.conflict_rate(), 4),
            'avg_hold_secs': round(e.avg_hold_secs(), 2),
            # How much of avg_hold_secs rests on censored lower bounds.
            'truncated_holds': e.truncated_holds,
            'suggested_max_per_module': suggest_max_per_module(e),
        }
        for module, e in stats.items()
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='orchestrator.analyze_modules',
        description='Per-module conflict analysis for hand-curated '
                    'orchestrator.yaml overrides.',
    )
    parser.add_argument(
        'db_path', type=Path,
        help='Path to runs.db (usually data/orchestrator/runs.db)',
    )
    parser.add_argument(
        '--since', type=_parse_since, default=_parse_since('7d'),
        help='Look-back window: <N>{s,m,h,d} or ISO timestamp (default: 7d)',
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Emit machine-readable JSON instead of a table',
    )
    parser.add_argument(
        '--min-dispatches', type=int, default=0,
        help='Drop modules with fewer than N dispatches (default: 0)',
    )
    args = parser.parse_args(argv)

    if not args.db_path.exists():
        print(f'runs.db not found: {args.db_path}', file=sys.stderr)
        return 2

    stats = aggregate(args.db_path, args.since)
    if args.min_dispatches > 0:
        stats = {
            m: e for m, e in stats.items()
            if e.dispatches >= args.min_dispatches
        }
    if not stats:
        print('(no module events in window)', file=sys.stderr)
        return 0

    if args.json:
        print(render_json(stats))
    else:
        print(render_table(stats))
    return 0


if __name__ == '__main__':
    sys.exit(main())
