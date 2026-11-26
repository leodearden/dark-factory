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

Startup order: `main` arms the inotify watch (`armed_source`) BEFORE handing
it to `watch`, which runs the initial filesystem scan before polling, so no
events are missed in the gap between the two.  If a pending match is found during the
scan, it is emitted immediately and the event loop is never entered.

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

  (c) A dropped ntfy push is visible ONLY as a marker line: the push is
      best-effort, so its failure never blocks the stdout emit and never
      changes the exit code — a re-arm whose push was dropped still reports
      `WATCHER_REARM_OUTCOME: FIRED exit=0`.  The one signal is a
      `WATCHER_NTFY_OUTCOME: FAILED` line on stderr, one per dropped push.
      `main` configures the logging stream to stderr at INFO, as every sibling
      CLI in this package does, so the line is there for a plain
      `python -m escalation.watcher` run rather than by grace of
      `logging.lastResort` — which anything configuring a root handler
      in-process would take away.
      That line is also the only way to COUNT an outage: `emit` runs at most
      once per process (its one call site in `main` exits on the next line), so a
      process-local integer could never read more than 1 — the aggregate is
      taken across the re-arm loop's stderr instead.  The marker deliberately
      copies `scripts/watcher-rearm.sh`'s `WATCHER_REARM_OUTCOME` idiom and
      its channel: stderr, keeping stdout pure escalation JSON for the
      consumers that parse it.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Protocol

from inotify_simple import Event, INotify, flags
from shared.timestamps import parse_timestamp_or_warn

from escalation.models import BORN_AT_L2_SEVERITIES, Escalation
from escalation.queue import read_escalation_for_scan

logger = logging.getLogger(__name__)

#: UnicodeDecodeError: a long-lived watcher must survive one undecodable record
#: (see ``escalation/queue.py::read_escalation_for_scan``).
_WATCHER_PARSE_ERRORS: tuple[type[BaseException], ...] = (
    json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError,
)


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
    Malformed / unreadable files are skipped with a log line from
    ``read_escalation_for_scan`` (WARNING; DEBUG for a record archived mid-scan).
    Mirrors the get_pending + find_pending_l2_by_root_cause idiom in queue.py.
    """
    best: Escalation | None = None
    best_ts: datetime | None = None

    for path in queue_dir.glob('esc-*.json'):
        if path.stem in exclude_ids:
            continue

        esc, _reason = read_escalation_for_scan(
            path, context='watcher._initial_scan', parse_errors=_WATCHER_PARSE_ERRORS,
        )
        if esc is None:
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


def _normalize_esc_id(esc_id: str) -> str:
    """Strip a trailing `.json` suffix, if present, from an esc-id.

    Shared by the static `--exclude-id` normalization in main() and by
    `_read_exclude_file` below, so both accept an id with or without the
    `.json` suffix identically and the rule can't drift between the two.
    """
    return esc_id[:-5] if esc_id.endswith('.json') else esc_id


def _read_exclude_file(path: Path | None) -> frozenset[str]:
    """Read a newline-delimited esc-id exclude list, tolerating blank lines
    and `#`-comments, and normalizing a trailing `.json` suffix via
    `_normalize_esc_id` (shared with `--exclude-id`'s static-exclude
    normalization).

    Fails open (returns an empty frozenset) when path is None or the file is
    missing, unreadable, or undecodable as text — a transient miss is
    retried on the next poll rather than treated as a fatal error.

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
    except (OSError, UnicodeDecodeError):
        return frozenset()

    ids: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        ids.add(_normalize_esc_id(stripped))
    return frozenset(ids)


def _snapshot_pending_ids(queue_dir: Path) -> frozenset[str]:
    """Return the set of esc-ids (filename stems) present in queue_dir right now.

    Used by --baseline to freeze a launch-time exclusion snapshot so a
    watcher run only fires on items filed AFTER it started, instead of
    hand-listing every already-pending id via --exclude-id.
    """
    return frozenset(p.stem for p in queue_dir.glob('esc-*.json'))


def exclusions(
    *,
    static: frozenset[str],
    baseline: frozenset[str],
    exclude_file: Path | None,
) -> Callable[[], frozenset[str]]:
    """Return the `current_excludes` callable `watch` polls.

    `static` and `baseline` are fixed for the run, and `exclude_file` is
    re-read on every call so an appended id is excluded from the next poll on.
    """
    fixed = static | baseline

    def current() -> frozenset[str]:
        return fixed | _read_exclude_file(exclude_file)

    return current


def ntfy_request(url: str, escalation: Escalation) -> urllib.request.Request:
    """Build the ntfy.sh push-notification POST for an escalation."""
    is_urgent = escalation.severity in (BORN_AT_L2_SEVERITIES | {'blocking'})
    title = f'[{escalation.severity.upper()}] Task {escalation.task_id}: {escalation.category}'
    body = escalation.summary
    if escalation.detail and escalation.detail != escalation.summary:
        body += f'\n\n{escalation.detail[:500]}'

    req = urllib.request.Request(url, data=body.encode('utf-8'), method='POST')
    req.add_header('Title', title)
    req.add_header('Priority', 'urgent' if is_urgent else 'default')
    req.add_header('Tags', 'rotating_light' if is_urgent else 'information_source')
    return req


def _send_ntfy(url: str, escalation: Escalation) -> None:
    """POST an escalation as a push notification to an ntfy.sh endpoint."""
    urllib.request.urlopen(ntfy_request(url, escalation))


def emit(
    esc: Escalation,
    ntfy_url: str | None,
    *,
    send: Callable[[str, Escalation], None] = _send_ntfy,
) -> None:
    """Print escalation JSON to stdout and optionally send an ntfy push notification.

    The push is best-effort: a failure is reported (invariant (c)) and then
    dropped, so it never costs the operator the queue item nor changes the
    exit code.
    """
    print(json.dumps(esc.to_dict(), indent=2))
    if ntfy_url:
        try:
            send(ntfy_url, esc)
        except Exception as e:
            logger.error('WATCHER_NTFY_OUTCOME: FAILED esc=%s url=%s: %s', esc.id, ntfy_url, e)


class EventSource(Protocol):
    """The read half of an inotify watch, as `watch` consumes it."""

    def read(self, timeout: int | None = None) -> Iterable[Event]: ...


@contextmanager
def armed_source(queue_dir: Path) -> Iterator[INotify]:
    """Yield an inotify source armed on `queue_dir` for every arrival.

    An arrival is renamed in (MOVED_TO, as invariant (a)'s writers do) or
    created in place (CREATE).
    """
    with INotify() as inotify:
        inotify.add_watch(str(queue_dir), flags.CREATE | flags.MOVED_TO)
        yield inotify


def _read_timeout_ms(deadline: float | None, clock: Callable[[], float]) -> int | None:
    """Milliseconds left before `deadline`, floored at 0; None blocks indefinitely."""
    if deadline is None:
        return None
    return max(0, int((deadline - clock()) * 1000))


def _first_matching_escalation(
    queue_dir: Path,
    events: Iterable[Event],
    task_id: str | None,
    level: int | None,
    excludes: frozenset[str],
) -> Escalation | None:
    """Return the escalation of the first event that names a matching pending record."""
    for event in events:
        name = event.name
        if not name or not (name.startswith('esc-') and name.endswith('.json')):
            continue

        if name[:-5] in excludes:
            continue

        esc, _reason = read_escalation_for_scan(
            queue_dir / name, context='watcher.watch', parse_errors=_WATCHER_PARSE_ERRORS,
        )
        if esc is not None and _matches(esc, task_id, level, excludes):
            return esc
    return None


def watch(
    queue_dir: Path,
    source: EventSource,
    *,
    task_id: str | None,
    level: int | None,
    timeout: float | None,
    current_excludes: Callable[[], frozenset[str]],
    clock: Callable[[], float],
) -> Escalation | None:
    """Return the first pending escalation matching the filters, or None once `timeout` expires.

    `source` must already be armed on `queue_dir` (see `armed_source`), so
    nothing filed during the initial scan is missed.  `current_excludes` is called before the scan and
    once per poll, and the `timeout` deadline is measured from the end of the
    scan.
    """
    match = _initial_scan(queue_dir, task_id, level, current_excludes())
    if match is not None:
        return match

    deadline = None if timeout is None else clock() + timeout
    while True:
        timeout_ms = _read_timeout_ms(deadline, clock)
        if timeout_ms == 0:
            return None

        events = list(source.read(timeout=timeout_ms))
        if deadline is not None and not events:
            return None

        match = _first_matching_escalation(queue_dir, events, task_id, level, current_excludes())
        if match is not None:
            return match


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    # Invariant (c)'s marker is an operator-facing STDERR line, so this process
    # has to own a handler: without one it reaches stderr only by grace of
    # `logging.lastResort`, which anything configuring a root handler in-process
    # silently takes away — and which emits the bare message with no level.
    # basicConfig defaults to stderr, so stdout stays pure escalation JSON, and
    # it is a no-op when a caller has already configured logging — the same call
    # every sibling CLI here makes (submit, sweep, archive).
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    args = _parse_args(argv)

    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)

    static_exclude = frozenset(_normalize_esc_id(e) for e in (args.exclude_id or []))
    exclude_file_path = Path(args.exclude_file) if args.exclude_file else None

    # Baseline is frozen ONCE here, BEFORE inotify is armed, so the race
    # window falls on the safe side: an escalation filed between this
    # snapshot and armed_source() below is absent from the snapshot (so it is
    # not permanently excluded) and, since no watch is armed yet, generates
    # no inotify event either -- but it is still on disk by the time the
    # initial scan below runs, so the scan's own directory glob picks it up
    # and fires on it normally. Snapshotting AFTER arming instead would
    # let such an item be captured by the snapshot (permanently excluded)
    # while ALSO queuing an inotify event that gets silently dropped as an
    # excluded id -- i.e. a genuinely new item swallowed for the run's
    # lifetime. Unlike --exclude-file, the baseline is never re-read (see
    # `exclusions`).
    baseline_ids = _snapshot_pending_ids(queue_dir) if args.baseline else frozenset()
    current_excludes = exclusions(
        static=static_exclude, baseline=baseline_ids, exclude_file=exclude_file_path,
    )

    # ARM inotify so no events are missed between the scan and the watch.
    with armed_source(queue_dir) as source:
        match = watch(
            queue_dir, source, task_id=args.task_id, level=args.level, timeout=args.timeout,
            current_excludes=current_excludes, clock=time.monotonic,
        )

    if match is None:
        sys.exit(124)
    emit(match, args.ntfy_url)
    sys.exit(0)


if __name__ == '__main__':
    main()
