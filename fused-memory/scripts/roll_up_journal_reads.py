#!/usr/bin/env python3
"""Move write_ops' successful non-search read history into read_op_counts (task 5405).

Run only after BOTH fused-memory and the dashboard are restarted on a build
containing task 5405. ``migrate`` is online-safe: short chunks, each one verified
transaction, resumable and abortable at any moment. ``status``, ``snapshot``
and ``verify`` are read-only. ``reclaim`` (VACUUM) needs fused-memory disabled
and its socket stopped. Runbook: fused-memory/scripts/README_roll_up_journal_reads.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.services.read_rollup import (
    COUNT_ONLY_READ_PREDICATE,
    COUNT_READ_SQL,
    DELETE_COUNT_ONLY_READS_SQL,
    RAW_READS_BY_ROLLUP_KEY_SQL,
)

DEFAULT_JOURNAL = Path('/home/leo/src/dark-factory/data/reconciliation/write_journal.db')
#: The config fused-memory reads when CONFIG_PATH is unset, as
#: fused_memory/server/main.py::run_server defaults it.
SERVICE_CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'config.yaml'
DEFAULT_CHUNK_ROWIDS = 5000
DEFAULT_PAUSE_SECONDS = 0.5
DEFAULT_DEPLOY_GRACE_MINUTES = 15.0
DEFAULT_WINDOW_HOURS = (1, 24, 168)
BUSY_TIMEOUT_MS = 30000
ROLLUP_TABLE = 'read_op_counts'
#: The service, and the socket that would re-activate it on the next connection.
FUSED_MEMORY_UNITS = ('fused-memory.service', 'fused-memory.socket')
_ACTIVE_STATES = frozenset({'active', 'activating', 'reloading', 'deactivating'})

_ROLLUP_KEY_COLUMNS = ('hour_start', 'project_id', 'agent_id', 'operation')

_FIRST_COUNT_ONLY_ROWID_SQL = (
    f'SELECT rowid FROM write_ops WHERE {COUNT_ONLY_READ_PREDICATE} ORDER BY rowid LIMIT 1'
)
_NEWEST_COUNT_ONLY_READ_SQL = (
    'SELECT created_at FROM write_ops INDEXED BY idx_wo_kind_time '
    f'WHERE {COUNT_ONLY_READ_PREDICATE} ORDER BY created_at DESC LIMIT 1'
)
_ROLLUP_COUNT_SQL = (
    f'SELECT count FROM {ROLLUP_TABLE} '
    'WHERE hour_start = ? AND project_id = ? AND agent_id = ? AND operation = ?'
)

# The Memory panel's numbers as an independent oracle: raw write_ops rows
# plus read_op_counts counts, in ONE statement so one read snapshot, keyed
# finer than both panel views. Restates
# dashboard/src/dashboard/data/write_journal.py::get_memory_ops, which
# fused-memory cannot import; NULL and '' operations both read 'unknown'
# there, so they share a key here. A plain literal, because
# dashboard/tests/test_write_journal.py::TestMigrationVerifyOracleMatchesPanel
# reads it with ast to hold the two to agreement.
_PANEL_COUNTS_SQL = """
SELECT hour, kind, operation, SUM(n) FROM (
    SELECT strftime('%Y-%m-%dT%H:00', created_at) AS hour, kind,
           IFNULL(NULLIF(operation, ''), 'unknown') AS operation, COUNT(*) AS n
    FROM write_ops INDEXED BY idx_wo_created
    WHERE created_at >= :start AND created_at < :end
    GROUP BY 1, 2, 3
    UNION ALL
    SELECT strftime('%Y-%m-%dT%H:00', hour_start), 'read',
           IFNULL(NULLIF(operation, ''), 'unknown'), SUM(count)
    FROM read_op_counts
    WHERE hour_start >= :start AND hour_start < :end
    GROUP BY 1, 2, 3
)
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3
"""


class PreflightError(RuntimeError):
    """The journal is not in a state this command may act on."""


class RollupVerificationError(RuntimeError):
    """A chunk's rollup deltas did not equal the raw rows it removed; rolled back."""


@dataclass(frozen=True)
class RowidSpan:
    first_rowid: int
    max_rowid: int


@dataclass(frozen=True)
class ChunkResult:
    lo: int
    hi: int
    keys: int
    rows_moved: int


@dataclass(frozen=True)
class MigrationLimits:
    chunk_rowids: int
    pause_seconds: float
    max_seconds: float | None
    max_chunks: int | None


@dataclass(frozen=True)
class MigrationReport:
    applied: bool
    chunks: int
    rows_moved: int
    seconds: float
    rows_per_second: float
    remaining: RowidSpan | None
    preview: tuple[tuple[Any, ...], ...]
    #: Each chunk's transaction time, which bounds its write-lock hold; keep it
    #: well under fused-memory's 5000 ms busy_timeout.
    mean_chunk_seconds: float = 0.0
    max_chunk_seconds: float = 0.0


@dataclass(frozen=True)
class PageFigures:
    page_size: int
    page_count: int
    freelist_count: int


@dataclass(frozen=True)
class ReclaimReport:
    applied: bool
    blockers: tuple[str, ...]
    before: PageFigures
    reclaimable_bytes: int
    before_bytes: int
    after: PageFigures | None = None
    after_bytes: int | None = None
    seconds: float | None = None


@dataclass(frozen=True)
class Window:
    start: str
    end: str
    hours: int


@dataclass(frozen=True)
class Mismatch:
    window: Window
    hour: str
    kind: str
    operation: str
    expected: int
    observed: int


def journal_hour(text: str) -> datetime:
    """``text`` as an aware UTC datetime floored to its hour; a naive value is UTC."""
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'not an ISO-8601 timestamp: {text!r}') from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def window_hours(text: str) -> tuple[int, ...]:
    try:
        hours = tuple(int(part) for part in text.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'not a comma-separated hour list: {text!r}') from exc
    if not hours or min(hours) < 1:
        raise argparse.ArgumentTypeError(f'window hours must be positive: {text!r}')
    return hours


def connect_read_only(journal: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f'{journal.resolve().as_uri()}?mode=ro', uri=True, timeout=5.0)
    conn.execute('PRAGMA query_only=ON')
    return conn


def connect_for_migration(journal: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(journal, isolation_level=None, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.execute(f'PRAGMA busy_timeout={BUSY_TIMEOUT_MS}')
    conn.execute('PRAGMA synchronous=FULL')
    return conn


def _scalar(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    return None if row is None else row[0]


def count_only_span(conn: sqlite3.Connection) -> RowidSpan | None:
    """The rowid span still holding count-only raw reads: the migration's work queue."""
    first = _scalar(conn, _FIRST_COUNT_ONLY_ROWID_SQL)
    if first is None:
        return None
    return RowidSpan(first_rowid=first, max_rowid=_scalar(conn, 'SELECT MAX(rowid) FROM write_ops'))


def status(conn: sqlite3.Connection) -> dict[str, Any]:
    require_rollup_table(conn)
    span = count_only_span(conn)
    return {
        'count_only_span': None if span is None else asdict(span),
        'rollup_rows': _scalar(conn, f'SELECT COUNT(*) FROM {ROLLUP_TABLE}'),
    }


def require_rollup_table(conn: sqlite3.Connection) -> None:
    found = _scalar(
        conn, "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (ROLLUP_TABLE,)
    )
    if found is None:
        raise PreflightError(
            f'the journal has no {ROLLUP_TABLE} table: restart fused-memory on a build '
            'containing task 5405 first'
        )


def require_no_recent_count_only_read(
    conn: sqlite3.Connection, *, grace: timedelta, now: datetime
) -> None:
    """Refuse while an old build may still journal count-only reads per row."""
    newest = _scalar(conn, _NEWEST_COUNT_ONLY_READ_SQL)
    if newest is not None and newest >= (now - grace).isoformat():
        raise PreflightError(
            f'a count-only raw read is newer than the {grace} deploy grace '
            f'(created_at {newest}): a fused-memory older than task 5405 may still be '
            'serving; restart it on the 5405 build and retry'
        )


def configured_read_retention_days() -> float:
    """fused-memory's raw read horizon, loaded as the service loads it: from
    $CONFIG_PATH, else SERVICE_CONFIG, with ${VAR:default} expanded from this
    process's environment."""
    previous = os.environ.get('CONFIG_PATH')
    os.environ['CONFIG_PATH'] = previous or str(SERVICE_CONFIG)
    try:
        return FusedMemoryConfig().write_journal.read_retention_days
    finally:
        if previous is None:
            del os.environ['CONFIG_PATH']


def panel_windows(
    end: datetime, hours: Sequence[int], *, now: datetime, read_retention_days: float
) -> tuple[Window, ...]:
    """Windows ending at *end*, refused if any reaches past the raw read horizon,
    where a startup prune may delete raw reads between snapshot and verify."""
    horizon = now - timedelta(days=read_retention_days)
    windows = tuple(
        Window(start=(end - timedelta(hours=h)).isoformat(), end=end.isoformat(), hours=h)
        for h in hours
    )
    for window in windows:
        if datetime.fromisoformat(window.start) < horizon:
            raise PreflightError(
                f'a {window.hours}h window starting {window.start} reaches past the '
                f'read retention horizon ({horizon.isoformat()})'
            )
    return windows


def panel_counts(conn: sqlite3.Connection, window: Window) -> dict[tuple[str, str, str], int]:
    rows = conn.execute(_PANEL_COUNTS_SQL, {'start': window.start, 'end': window.end})
    return {(hour, kind, operation): n for hour, kind, operation, n in rows}


def snapshot(conn: sqlite3.Connection, windows: Sequence[Window]) -> dict[str, Any]:
    require_rollup_table(conn)
    return {
        'windows': [
            {
                **asdict(window),
                'counts': [[*key, n] for key, n in sorted(panel_counts(conn, window).items())],
            }
            for window in windows
        ]
    }


def verify(conn: sqlite3.Connection, recorded: dict[str, Any]) -> list[Mismatch]:
    require_rollup_table(conn)
    mismatches = []
    for entry in recorded['windows']:
        window = Window(start=entry['start'], end=entry['end'], hours=entry['hours'])
        expected = {(hour, kind, op): n for hour, kind, op, n in entry['counts']}
        observed = panel_counts(conn, window)
        for key in sorted(expected.keys() | observed.keys()):
            if expected.get(key, 0) != observed.get(key, 0):
                mismatches.append(Mismatch(
                    window, *key, expected=expected.get(key, 0), observed=observed.get(key, 0)
                ))
    return mismatches


def _rollup_count(conn: sqlite3.Connection, key: Sequence[Any]) -> int:
    return _scalar(conn, _ROLLUP_COUNT_SQL, key) or 0


def _verify_chunk(
    lo: int,
    hi: int,
    preview: Sequence[Sequence[Any]],
    before: dict[tuple, int],
    after: dict[tuple, int],
    deleted: int,
) -> None:
    expected_deleted = sum(row[4] for row in preview)
    if deleted != expected_deleted:
        raise RollupVerificationError(
            f'chunk rowid [{lo}, {hi}): deleted {deleted} raw reads, expected {expected_deleted}'
        )
    for row in preview:
        key, n = tuple(row[:4]), row[4]
        if after[key] != before[key] + n:
            raise RollupVerificationError(
                f'chunk rowid [{lo}, {hi}): {dict(zip(_ROLLUP_KEY_COLUMNS, key, strict=True))} '
                f'count is {after[key]}, expected {before[key]} + {n} = {before[key] + n}'
            )


def migrate_chunk(conn: sqlite3.Connection, lo: int, hi: int) -> ChunkResult:
    """Move one rowid window in ONE transaction, committed only if every key's
    rollup delta equals the raw rows it removed."""
    conn.execute('BEGIN IMMEDIATE')
    try:
        preview = conn.execute(RAW_READS_BY_ROLLUP_KEY_SQL, (lo, hi)).fetchall()
        keys = [tuple(row[:4]) for row in preview]
        before = {key: _rollup_count(conn, key) for key in keys}
        conn.executemany(COUNT_READ_SQL, preview)
        deleted = conn.execute(DELETE_COUNT_ONLY_READS_SQL, (lo, hi)).rowcount
        after = {key: _rollup_count(conn, key) for key in keys}
        _verify_chunk(lo, hi, preview, before, after, deleted)
    except BaseException:
        conn.execute('ROLLBACK')
        raise
    conn.execute('COMMIT')
    return ChunkResult(lo=lo, hi=hi, keys=len(keys), rows_moved=deleted)


def preview_first_chunk(conn: sqlite3.Connection, chunk_rowids: int) -> MigrationReport:
    span = count_only_span(conn)
    preview: tuple[tuple[Any, ...], ...] = ()
    if span is not None:
        lo = span.first_rowid
        preview = tuple(
            tuple(row)
            for row in conn.execute(RAW_READS_BY_ROLLUP_KEY_SQL, (lo, lo + chunk_rowids))
        )
    return MigrationReport(
        applied=False, chunks=0, rows_moved=0, seconds=0.0, rows_per_second=0.0,
        remaining=span, preview=preview,
    )


def _limit_reached(limits: MigrationLimits, chunks: int, elapsed: float) -> bool:
    if limits.max_chunks is not None and chunks >= limits.max_chunks:
        return True
    return limits.max_seconds is not None and elapsed >= limits.max_seconds


def migrate(conn: sqlite3.Connection, limits: MigrationLimits) -> MigrationReport:
    started = time.monotonic()
    span = count_only_span(conn)
    rows_moved = 0
    chunk_seconds: list[float] = []
    if span is not None:
        lo, hi = span.first_rowid, span.max_rowid + 1
        while lo < hi and not _limit_reached(limits, len(chunk_seconds), time.monotonic() - started):
            chunk_started = time.monotonic()
            result = migrate_chunk(conn, lo, min(lo + limits.chunk_rowids, hi))
            chunk_seconds.append(time.monotonic() - chunk_started)
            rows_moved += result.rows_moved
            lo = result.hi
            if lo < hi:
                time.sleep(limits.pause_seconds)
    seconds = time.monotonic() - started
    return MigrationReport(
        applied=True,
        chunks=len(chunk_seconds),
        rows_moved=rows_moved,
        seconds=round(seconds, 3),
        rows_per_second=round(rows_moved / seconds, 1) if seconds > 0 else 0.0,
        remaining=count_only_span(conn),
        preview=(),
        mean_chunk_seconds=round(sum(chunk_seconds) / len(chunk_seconds), 3) if chunk_seconds else 0.0,
        max_chunk_seconds=round(max(chunk_seconds, default=0.0), 3),
    )


def fused_memory_unit_active() -> bool:
    """True unless systemd confirms every FUSED_MEMORY_UNITS unit is NOT active; fails closed."""
    try:
        probe = subprocess.run(
            ['systemctl', '--user', 'is-active', *FUSED_MEMORY_UNITS],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    states = probe.stdout.split()
    return len(states) != len(FUSED_MEMORY_UNITS) or bool(_ACTIVE_STATES.intersection(states))


def page_figures(conn: sqlite3.Connection) -> PageFigures:
    return PageFigures(
        page_size=_scalar(conn, 'PRAGMA page_size'),
        page_count=_scalar(conn, 'PRAGMA page_count'),
        freelist_count=_scalar(conn, 'PRAGMA freelist_count'),
    )


def on_disk_bytes(journal: Path) -> int:
    """``.db`` + ``-wal`` + ``-shm``, as WriteJournal.growth_sample counts them."""
    total = 0
    for suffix in ('', '-wal', '-shm'):
        sidecar = journal.with_name(journal.name + suffix)
        if sidecar.exists():
            total += sidecar.stat().st_size
    return total


def reclaim_blockers(conn: sqlite3.Connection, *, unit_active: Callable[[], bool]) -> list[str]:
    blockers = []
    span = count_only_span(conn)
    if span is not None:
        blockers.append(
            f'count-only raw reads remain in rowids first_rowid={span.first_rowid} '
            f'max_rowid={span.max_rowid}: finish migrate --apply first'
        )
    if unit_active():
        blockers.append(
            f'{" or ".join(FUSED_MEMORY_UNITS)} is active (or the state could not be '
            'confirmed): stop the socket and disable --now the service first'
        )
    return blockers


def use_temp_dir(conn: sqlite3.Connection, temp_dir: Path) -> None:
    """Point VACUUM's temporary copy at *temp_dir* (process-wide, as SQLite defines it)."""
    conn.execute(f"PRAGMA temp_store_directory = '{temp_dir.resolve()}'")
    if _scalar(conn, 'PRAGMA temp_store_directory') != str(temp_dir.resolve()):
        raise PreflightError(
            'this sqlite build ignores PRAGMA temp_store_directory: '
            'set SQLITE_TMPDIR in the environment instead of --temp-dir'
        )


def vacuum_in_place(conn: sqlite3.Connection) -> None:
    """VACUUM keeps the file's inode, so a long-lived reader's handle stays valid."""
    conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    conn.execute('VACUUM')
    conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')


def reclaim(
    journal: Path,
    *,
    apply: bool,
    unit_active: Callable[[], bool],
    temp_dir: Path | None = None,
) -> ReclaimReport:
    with closing(connect_read_only(journal)) as conn:
        blockers = tuple(reclaim_blockers(conn, unit_active=unit_active))
        before = page_figures(conn)
    report = ReclaimReport(
        applied=False,
        blockers=blockers,
        before=before,
        reclaimable_bytes=before.freelist_count * before.page_size,
        before_bytes=on_disk_bytes(journal),
    )
    if not apply:
        return report
    if blockers:
        raise PreflightError('; '.join(blockers))
    started = time.monotonic()
    with closing(connect_for_migration(journal)) as conn:
        if temp_dir is not None:
            use_temp_dir(conn, temp_dir)
        vacuum_in_place(conn)
        after = page_figures(conn)
    return replace(
        report,
        applied=True,
        after=after,
        after_bytes=on_disk_bytes(journal),
        seconds=round(time.monotonic() - started, 3),
    )


def _common_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--journal', type=Path, default=DEFAULT_JOURNAL)
    return common


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest='command', required=True)
    common = _common_parser()
    commands.add_parser('status', parents=[common], help='read-only progress report')

    snap = commands.add_parser('snapshot', parents=[common], help='record panel windows')
    snap.add_argument('--out', type=Path, required=True)
    snap.add_argument('--window-hours', type=window_hours, default=DEFAULT_WINDOW_HOURS)
    snap.add_argument(
        '--window-end', type=journal_hour, default=None,
        help='window end, floored to the hour (default: the hour before the current one)',
    )
    snap.add_argument(
        '--read-retention-days', type=float, default=None,
        help="fused-memory's raw read horizon (default: write_journal.read_retention_days "
        'from $CONFIG_PATH, else fused-memory/config/config.yaml, under this environment)',
    )

    check = commands.add_parser('verify', parents=[common], help='compare against a snapshot')
    check.add_argument('--snapshot', type=Path, required=True)

    move = commands.add_parser('migrate', parents=[common], help='move history in chunks')
    move.add_argument('--apply', action='store_true')
    move.add_argument('--chunk-rowids', type=int, default=DEFAULT_CHUNK_ROWIDS)
    move.add_argument('--pause-seconds', type=float, default=DEFAULT_PAUSE_SECONDS)
    move.add_argument('--max-seconds', type=float, default=None)
    move.add_argument('--max-chunks', type=int, default=None)
    move.add_argument(
        '--deploy-grace-minutes', type=float, default=DEFAULT_DEPLOY_GRACE_MINUTES
    )

    vacuum = commands.add_parser(
        'reclaim', parents=[common], help='VACUUM in place; fused-memory must be disabled'
    )
    vacuum.add_argument('--apply', action='store_true')
    vacuum.add_argument(
        '--temp-dir', type=Path, default=None,
        help="VACUUM's temporary copy (default: SQLite's own: $SQLITE_TMPDIR, $TMPDIR, "
        '/var/tmp, /usr/tmp, /tmp)',
    )
    return parser


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _run_status(args: argparse.Namespace) -> int:
    with closing(connect_read_only(args.journal)) as conn:
        _print_json(status(conn))
    return 0


def _run_snapshot(args: argparse.Namespace) -> int:
    now = datetime.now(UTC)
    end = args.window_end or journal_hour(now.isoformat()) - timedelta(hours=1)
    retention = args.read_retention_days
    if retention is None:
        retention = configured_read_retention_days()
    windows = panel_windows(end, args.window_hours, now=now, read_retention_days=retention)
    with closing(connect_read_only(args.journal)) as conn:
        recorded = snapshot(conn, windows)
    args.out.write_text(json.dumps(recorded, indent=2))
    _print_json({
        'out': str(args.out),
        'windows': [asdict(w) for w in windows],
        'read_retention_days': retention,
    })
    return 0


def _run_verify(args: argparse.Namespace) -> int:
    recorded = json.loads(args.snapshot.read_text())
    with closing(connect_read_only(args.journal)) as conn:
        mismatches = verify(conn, recorded)
    _print_json({'mismatches': [asdict(m) for m in mismatches]})
    return 1 if mismatches else 0


def _run_migrate(args: argparse.Namespace) -> int:
    grace = timedelta(minutes=args.deploy_grace_minutes)
    connect = connect_for_migration if args.apply else connect_read_only
    with closing(connect(args.journal)) as conn:
        require_rollup_table(conn)
        require_no_recent_count_only_read(conn, grace=grace, now=datetime.now(UTC))
        if not args.apply:
            report = preview_first_chunk(conn, args.chunk_rowids)
        else:
            report = migrate(conn, MigrationLimits(
                chunk_rowids=args.chunk_rowids,
                pause_seconds=args.pause_seconds,
                max_seconds=args.max_seconds,
                max_chunks=args.max_chunks,
            ))
    _print_json(asdict(report))
    return 0


def _run_reclaim(args: argparse.Namespace, unit_active: Callable[[], bool]) -> int:
    report = reclaim(
        args.journal, apply=args.apply, unit_active=unit_active, temp_dir=args.temp_dir
    )
    _print_json(asdict(report))
    return 0


_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    'status': _run_status,
    'snapshot': _run_snapshot,
    'verify': _run_verify,
    'migrate': _run_migrate,
}


def main(
    argv: list[str] | None = None,
    *,
    unit_active: Callable[[], bool] = fused_memory_unit_active,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == 'reclaim':
            return _run_reclaim(args, unit_active)
        return _COMMANDS[args.command](args)
    except (PreflightError, RollupVerificationError, sqlite3.Error, OSError) as exc:
        print(f'{args.command} refused or failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
