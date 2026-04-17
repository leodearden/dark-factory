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
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

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
    open_acquires: dict[str, float] = field(default_factory=dict)

    def conflict_rate(self) -> float:
        """Fraction of skips over dispatches — ~0 is idle, >1 is contended."""
        if self.dispatches == 0:
            return 0.0
        return self.skipped_waiting / self.dispatches

    def avg_hold_secs(self) -> float:
        if self.hold_samples == 0:
            return 0.0
        return self.total_hold_secs / self.hold_samples


def _first_component(module: str) -> str:
    """Return the first path component (matching how for_module dispatches)."""
    return module.strip('/').split('/', 1)[0] if module else ''


def _iter_events(
    db_path: Path,
    since: datetime,
) -> Iterable[tuple[str, str, str | None, dict]]:
    """Yield (timestamp, event_type, task_id, data) tuples from runs.db."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            'SELECT timestamp, event_type, task_id, data '
            'FROM events '
            'WHERE timestamp >= ? '
            "AND event_type IN "
            "('lock_acquired','lock_released','task_skipped','merge_attempt') "
            'ORDER BY id ASC',
            (since.isoformat(),),
        )
        for ts, event_type, task_id, data in cur:
            try:
                data_dict = json.loads(data) if data else {}
            except (TypeError, ValueError):
                data_dict = {}
            yield ts, event_type, task_id, data_dict
    finally:
        conn.close()


def _parse_ts(ts: str) -> float:
    """Parse an ISO timestamp into POSIX seconds."""
    return datetime.fromisoformat(ts).timestamp()


def aggregate(db_path: Path, since: datetime) -> dict[str, ModuleStats]:
    """Scan ``db_path`` events since ``since`` and return per-module stats."""
    stats: dict[str, ModuleStats] = defaultdict(ModuleStats)
    for ts, event_type, task_id, data in _iter_events(db_path, since):
        modules = data.get('modules') or []
        if isinstance(modules, str):
            modules = [modules]
        keys = {_first_component(m) for m in modules if m}
        keys.discard('')
        if event_type == 'lock_acquired':
            posix = _parse_ts(ts)
            for k in keys:
                stats[k].dispatches += 1
                if task_id:
                    stats[k].open_acquires[task_id] = posix
        elif event_type == 'lock_released':
            posix = _parse_ts(ts)
            for k in keys:
                if task_id and task_id in stats[k].open_acquires:
                    start = stats[k].open_acquires.pop(task_id)
                    stats[k].total_hold_secs += max(0.0, posix - start)
                    stats[k].hold_samples += 1
        elif event_type == 'task_skipped':
            for k in keys:
                stats[k].skipped_waiting += 1
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


def render_table(stats: dict[str, ModuleStats]) -> str:
    """Human-readable table sorted by descending conflict rate."""
    lines = [
        f'{"module":<32} {"dispatches":>12} {"skipped":>10} '
        f'{"conflict":>10} {"avg_hold_s":>12} {"suggest":>8}'
    ]
    ranked = sorted(
        stats.items(),
        key=lambda kv: (-kv[1].conflict_rate(), -kv[1].dispatches),
    )
    for module, entry in ranked:
        lines.append(
            f'{module[:32]:<32} {entry.dispatches:>12d} '
            f'{entry.skipped_waiting:>10d} {entry.conflict_rate():>10.2f} '
            f'{entry.avg_hold_secs():>12.1f} {suggest_max_per_module(entry):>8d}'
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
