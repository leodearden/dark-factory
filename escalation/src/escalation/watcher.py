"""CLI watcher that uses inotify to watch for new escalations.

Usage: python -m escalation.watcher --queue-dir <path> [--task-id <id>]
       [--level <int>] [--ntfy-url <url>] [--timeout <secs>]
       [--exclude-id <esc-id> ...]

Watches for new .json files in the queue directory. When a matching pending
escalation appears (or is already present at startup), prints the escalation
JSON to stdout and optionally sends a push notification via ntfy.sh.

Exit-code contract:
  0    — one matching escalation printed to stdout.
  124  — --timeout expired; nothing printed (coreutils convention).
  0    — SIGTERM received (treated as a clean shutdown).

Startup order: inotify watch is armed BEFORE the initial filesystem scan so
no events are missed in the gap between the two.  If a pending match is found
during the scan, it is emitted immediately and the event loop is never entered.

Load-bearing invariants (callers MUST rely on these):

  (a) Atomic writes: all escalation-queue writers use a tmp-file+rename
      sequence (including fused-memory reconciliation).  A partial or
      in-progress JSON file is therefore impossible to observe via an inotify
      CREATE/MOVED_TO event — every event fires only after the rename has
      completed and the file is fully durable.

  (b) Wake-signal only, drains are authoritative: the watcher signals that
      *something* happened; it does not guarantee exactly-once delivery or
      that the returned escalation is still actionable.  Consumers MUST
      re-drain the queue after every watcher return (via get_pending or
      equivalent) and MUST tolerate spurious wakes (events that produce no
      matching escalation after the drain).  A spurious wake is normal and
      not an error.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from inotify_simple import INotify, flags
from shared.timestamps import parse_timestamp_or_warn

from escalation.models import BORN_AT_L2_SEVERITIES, Escalation


def _matches(
    esc: Escalation,
    task_id: str | None,
    level: int | None,
    exclude_ids: frozenset[str] | set[str] = frozenset(),
) -> bool:
    """Return True iff esc is pending and satisfies the optional filters."""
    if esc.status != 'pending':
        return False
    if esc.id in exclude_ids:  # defensive fallback: covers id/filename mismatch; filename stem pre-checks handle the normal case
        return False
    if task_id and esc.task_id != task_id:
        return False
    return not (level is not None and esc.level != level)


def _initial_scan(
    queue_dir: Path,
    task_id: str | None,
    level: int | None,
    exclude_ids: frozenset[str] | set[str] = frozenset(),
) -> Escalation | None:
    """Scan the queue directory for already-pending matching escalations.

    Returns the OLDEST by timestamp, or None if no match found.
    Malformed / unreadable JSON files are skipped (never silently dropped).
    Mirrors the get_pending + find_pending_l2_by_root_cause idiom in queue.py.
    """
    best: Escalation | None = None
    best_ts: datetime | None = None

    for path in queue_dir.glob('esc-*.json'):
        if path.stem in exclude_ids:
            continue

        try:
            esc = Escalation.from_json(path.read_text())
        except (json.JSONDecodeError, KeyError, OSError, TypeError):
            continue

        if not _matches(esc, task_id, level, exclude_ids):
            continue

        # Parse timestamp; emits a WARNING (loud-over-silent) when malformed.
        # Default datetime.min fallback: malformed entry included and sorts oldest.
        ts, _ = parse_timestamp_or_warn(esc.timestamp, context='watcher._initial_scan')

        if best_ts is None or ts < best_ts:
            best = esc
            best_ts = ts

    return best


def _read_exclude_file(path: Path | None) -> frozenset[str]:
    """Read a newline-delimited esc-id exclude list, tolerating blank lines
    and `#`-comments, and normalizing a trailing `.json` suffix like
    `--exclude-id` does (watcher.py's static exclude normalization).

    Fails open (returns an empty frozenset) when path is None or the file
    is missing/unreadable — a transient miss is retried on the next poll
    rather than treated as a fatal error.

    Re-read in full on every poll, so a caller appending a new id mid-run
    should do so as a single short-line append (e.g. `echo esc-id >> file`),
    which is atomic on POSIX filesystems. A poll racing a non-atomic,
    multi-write append could observe a partially-written final line; this is
    self-healing since the next poll re-reads the completed file.
    """
    if path is None:
        return frozenset()

    try:
        text = path.read_text()
    except OSError:
        return frozenset()

    ids: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        ids.add(stripped[:-5] if stripped.endswith('.json') else stripped)
    return frozenset(ids)


def _snapshot_pending_ids(queue_dir: Path) -> frozenset[str]:
    """Return the set of esc-ids (filename stems) present in queue_dir right now.

    Used by --baseline to freeze a launch-time exclusion snapshot so a
    watcher run only fires on items filed AFTER it started, instead of
    hand-listing every already-pending id via --exclude-id.
    """
    return frozenset(p.stem for p in queue_dir.glob('esc-*.json'))


def _send_ntfy(url: str, escalation: Escalation) -> None:
    """POST an escalation as a push notification to an ntfy.sh endpoint."""
    is_urgent = escalation.severity in (BORN_AT_L2_SEVERITIES | {'blocking'})
    title = f'[{escalation.severity.upper()}] Task {escalation.task_id}: {escalation.category}'
    body = escalation.summary
    if escalation.detail and escalation.detail != escalation.summary:
        body += f'\n\n{escalation.detail[:500]}'

    req = urllib.request.Request(url, data=body.encode('utf-8'), method='POST')
    req.add_header('Title', title)
    req.add_header('Priority', 'urgent' if is_urgent else 'default')
    req.add_header('Tags', 'rotating_light' if is_urgent else 'information_source')
    urllib.request.urlopen(req)


def _emit(esc: Escalation, ntfy_url: str | None) -> None:
    """Print escalation JSON to stdout and optionally send an ntfy push notification."""
    print(json.dumps(esc.to_dict(), indent=2))
    if ntfy_url:
        try:
            _send_ntfy(ntfy_url, esc)
        except Exception as e:
            print(f'ntfy send failed: {e}', file=sys.stderr)


def main() -> None:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    parser = argparse.ArgumentParser(description='Watch for escalation events')
    parser.add_argument('--queue-dir', required=True, help='Escalation queue directory')
    parser.add_argument('--task-id', default=None, help='Filter to a specific task ID')
    parser.add_argument('--level', type=int, default=None, help='Filter to a specific escalation level')
    parser.add_argument('--ntfy-url', default=None, help='ntfy.sh topic URL for push notifications')
    parser.add_argument(
        '--timeout', type=float, default=None,
        help='max blocking wait in seconds; on expiry exit 124',
    )
    parser.add_argument(
        '--exclude-id', action='append', default=None,
        help='esc-id to exclude from initial scan AND event loop; repeatable',
    )
    parser.add_argument(
        '--exclude-file', default=None,
        help=(
            'path to a newline-delimited esc-id exclude list, re-read each '
            'poll; repeated ids skip initial scan AND event loop'
        ),
    )
    parser.add_argument(
        '--baseline', action='store_true',
        help=(
            'snapshot pending esc-ids at launch; fire only on items NOT in '
            'that snapshot, instead of hand-listing every already-pending id'
        ),
    )
    args = parser.parse_args()

    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)

    static_exclude = frozenset(
        e[:-5] if e.endswith('.json') else e for e in (args.exclude_id or [])
    )
    exclude_file_path = Path(args.exclude_file) if args.exclude_file else None

    def current_excludes() -> frozenset[str]:
        return static_exclude | baseline_ids | _read_exclude_file(exclude_file_path)

    # Baseline is frozen ONCE here, BEFORE inotify is armed, so the race
    # window falls on the safe side: an escalation filed between this
    # snapshot and add_watch() below is absent from the snapshot (so it is
    # not permanently excluded) and, since no watch is armed yet, generates
    # no inotify event either -- but it is still on disk by the time the
    # initial scan below runs, so the scan's own directory glob picks it up
    # and fires on it normally. Snapshotting AFTER add_watch() instead would
    # let such an item be captured by the snapshot (permanently excluded)
    # while ALSO queuing an inotify event that gets silently dropped as an
    # excluded id -- i.e. a genuinely new item swallowed for the run's
    # lifetime. Unlike --exclude-file, the baseline is never re-read.
    baseline_ids = _snapshot_pending_ids(queue_dir) if args.baseline else frozenset()

    # ARM inotify so no events are missed between the scan and the watch.
    inotify = INotify()
    watch_flags = flags.CREATE | flags.MOVED_TO
    inotify.add_watch(str(queue_dir), watch_flags)

    # Initial scan: emit any already-pending escalation and exit immediately.
    match = _initial_scan(queue_dir, args.task_id, args.level, current_excludes())
    if match is not None:
        _emit(match, args.ntfy_url)
        sys.exit(0)

    # Compute monotonic deadline (None = block indefinitely).
    deadline: float | None = (
        None if args.timeout is None else time.monotonic() + args.timeout
    )

    # Event loop: wait for new files.
    while True:
        if deadline is not None:
            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
            if remaining_ms == 0:
                sys.exit(124)
            events = list(inotify.read(timeout=remaining_ms))
        else:
            events = list(inotify.read(timeout=None))

        if deadline is not None and not events:
            sys.exit(124)

        excludes = current_excludes()  # RE-READ each poll (--exclude-file may have changed)

        for event in events:
            name = event.name
            if not name or not (name.startswith('esc-') and name.endswith('.json')):
                continue

            if name[:-5] in excludes:
                continue

            path = queue_dir / name
            try:
                esc = Escalation.from_json(path.read_text())
            except (json.JSONDecodeError, KeyError, OSError, TypeError):
                continue

            if not _matches(esc, args.task_id, args.level, excludes):
                continue

            _emit(esc, args.ntfy_url)
            sys.exit(0)


if __name__ == '__main__':
    main()
