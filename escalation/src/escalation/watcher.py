"""CLI watcher that uses inotify to watch for new escalations.

Usage: python -m escalation.watcher --queue-dir <path> [--task-id <id>]

Watches for new .json files in the queue directory. When one appears and matches
the optional task_id filter, prints the escalation JSON to stdout and exits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inotify_simple import INotify, flags

from escalation.models import Escalation


def main() -> None:
    parser = argparse.ArgumentParser(description='Watch for escalation events')
    parser.add_argument('--queue-dir', required=True, help='Escalation queue directory')
    parser.add_argument('--task-id', default=None, help='Filter to a specific task ID')
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

            # Print and exit
            print(json.dumps(esc.to_dict(), indent=2))
            sys.exit(0)


if __name__ == '__main__':
    main()
