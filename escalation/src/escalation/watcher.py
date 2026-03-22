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
import sys
import urllib.request
from pathlib import Path

from inotify_simple import INotify, flags

from escalation.models import Escalation


def _send_ntfy(url: str, escalation: Escalation) -> None:
    """POST an escalation as a push notification to an ntfy.sh endpoint."""
    is_blocking = escalation.severity == 'blocking'
    title = f'[{"BLOCKING" if is_blocking else "INFO"}] Task {escalation.task_id}: {escalation.category}'
    body = escalation.summary
    if escalation.detail and escalation.detail != escalation.summary:
        body += f'\n\n{escalation.detail[:500]}'

    req = urllib.request.Request(url, data=body.encode('utf-8'), method='POST')
    req.add_header('Title', title)
    req.add_header('Priority', 'urgent' if is_blocking else 'default')
    req.add_header('Tags', 'rotating_light' if is_blocking else 'information_source')
    urllib.request.urlopen(req)


def main() -> None:
    parser = argparse.ArgumentParser(description='Watch for escalation events')
    parser.add_argument('--queue-dir', required=True, help='Escalation queue directory')
    parser.add_argument('--task-id', default=None, help='Filter to a specific task ID')
    parser.add_argument('--ntfy-url', default=None, help='ntfy.sh topic URL for push notifications')
    args = parser.parse_args()

    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)

    inotify = INotify()
    watch_flags = flags.CREATE | flags.MOVED_TO
    inotify.add_watch(str(queue_dir), watch_flags)

    while True:
        for event in inotify.read():
            name = event.name
            if not name or not name.endswith('.json'):
                continue

            path = queue_dir / name
            try:
                esc = Escalation.from_json(path.read_text())
            except (json.JSONDecodeError, KeyError, OSError):
                continue

            if esc.status != 'pending':
                continue

            if args.task_id and esc.task_id != args.task_id:
                continue

            print(json.dumps(esc.to_dict(), indent=2))

            if args.ntfy_url:
                try:
                    _send_ntfy(args.ntfy_url, esc)
                except Exception as e:
                    print(f'ntfy send failed: {e}', file=sys.stderr)

            sys.exit(0)


if __name__ == '__main__':
    main()
