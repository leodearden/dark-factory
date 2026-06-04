"""CLI watcher that uses inotify to watch for new escalations.

Usage: python -m escalation.watcher --queue-dir <path> [--task-id <id>]
       [--ntfy-url <url>]

Watches for new .json files in the queue directory. When one appears and matches
the optional task_id filter, prints the escalation JSON to stdout (and optionally
sends a push notification via ntfy.sh).

Exits after the first matching escalation.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from inotify_simple import INotify, flags

from escalation.models import BORN_AT_L2_SEVERITIES, Escalation


def _matches(esc: Escalation, task_id: Optional[str], level: Optional[int]) -> bool:
    """Return True iff esc is pending and satisfies the optional filters."""
    if esc.status != 'pending':
        return False
    if task_id and esc.task_id != task_id:
        return False
    if level is not None and esc.level != level:
        return False
    return True


def _initial_scan(
    queue_dir: Path,
    task_id: Optional[str],
    level: Optional[int],
) -> Optional[Escalation]:
    """Scan the queue directory for already-pending matching escalations.

    Returns the OLDEST by timestamp, or None if no match found.
    Malformed / unreadable JSON files are skipped (never silently dropped).
    Mirrors the get_pending + find_pending_l2_by_root_cause idiom in queue.py.
    """
    best: Optional[Escalation] = None
    best_ts: Optional[datetime] = None

    for path in queue_dir.glob('esc-*.json'):
        try:
            esc = Escalation.from_json(path.read_text())
        except (json.JSONDecodeError, KeyError, OSError, TypeError):
            continue

        if not _matches(esc, task_id, level):
            continue

        try:
            ts = datetime.fromisoformat(esc.timestamp)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            ts = datetime.min.replace(tzinfo=UTC)

        if best_ts is None or ts < best_ts:
            best = esc
            best_ts = ts

    return best


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


def _emit(esc: Escalation, ntfy_url: Optional[str]) -> None:
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
    args = parser.parse_args()

    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)

    # ARM inotify first so no events are missed between scan and watch.
    inotify = INotify()
    watch_flags = flags.CREATE | flags.MOVED_TO
    inotify.add_watch(str(queue_dir), watch_flags)

    # Initial scan: emit any already-pending escalation and exit immediately.
    match = _initial_scan(queue_dir, args.task_id, args.level)
    if match is not None:
        _emit(match, args.ntfy_url)
        sys.exit(0)

    # Compute monotonic deadline (None = block indefinitely).
    deadline: Optional[float] = (
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

        for event in events:
            name = event.name
            if not name or not name.endswith('.json'):
                continue

            path = queue_dir / name
            try:
                esc = Escalation.from_json(path.read_text())
            except (json.JSONDecodeError, KeyError, OSError):
                continue

            if not _matches(esc, args.task_id, args.level):
                continue

            _emit(esc, args.ntfy_url)
            sys.exit(0)


if __name__ == '__main__':
    main()
