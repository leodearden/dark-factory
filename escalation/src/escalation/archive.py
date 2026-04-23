"""Escalation queue archive utilities.

Provides:
  - archive_dir_for_date: compute the archive subdirectory path for a resolved_at timestamp
  - prune_archive: delete dated archive subdirs older than a retention threshold
  - main: CLI entry point (`python -m escalation.archive`)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

ARCHIVE_SUBDIR = 'archive'
DATE_FORMAT = '%Y-%m-%d'


def archive_dir_for_date(queue_dir: Path, resolved_at_iso: str) -> Path:
    """Return the archive subdirectory path for the given resolved_at ISO timestamp.

    Does NOT create the directory — pure path computation.

    Args:
        queue_dir: Root queue directory (parent of ``archive/``).
        resolved_at_iso: ISO 8601 timestamp string (timezone-aware or naive).

    Returns:
        ``queue_dir / 'archive' / 'YYYY-MM-DD'`` derived from the **UTC** date.
        Timezone-aware timestamps are converted to UTC before extracting the date
        so that a +05:00 timestamp near midnight is bucketed by UTC day, not local day.
        Naive timestamps are treated as-is (assumed UTC at call site).
    """
    dt = datetime.fromisoformat(resolved_at_iso)
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC)
    date_str = dt.strftime(DATE_FORMAT)
    return Path(queue_dir) / ARCHIVE_SUBDIR / date_str


def prune_archive(
    queue_dir: Path,
    retention_days: int,
    now: datetime | None = None,
) -> int:
    """Delete dated archive subdirs older than *retention_days*.

    Only removes children of ``queue_dir/archive/`` whose names match
    ``YYYY-MM-DD``.  Loose files, non-matching directories, and other
    entries are left untouched.

    Args:
        queue_dir: Root queue directory (parent of ``archive/``).
        retention_days: Subdirs older than this many days are removed.
        now: Reference time for cutoff calculation (defaults to ``datetime.now(UTC)``).

    Returns:
        Number of subdirectories removed.
    """
    if now is None:
        now = datetime.now(UTC)

    archive_dir = Path(queue_dir) / ARCHIVE_SUBDIR
    if not archive_dir.exists():
        return 0

    cutoff = now - timedelta(days=retention_days)
    removed = 0

    for child in archive_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            parsed = datetime.strptime(child.name, DATE_FORMAT).replace(tzinfo=UTC)
        except ValueError:
            continue
        if parsed < cutoff:
            shutil.rmtree(child)
            removed += 1
            logger.info(f'Pruned archive subdir: {child}')

    return removed


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: python -m escalation.archive."""
    parser = argparse.ArgumentParser(
        description='Prune old dated archive subdirectories from an escalation queue.',
    )
    parser.add_argument(
        '--queue-dir',
        required=True,
        type=Path,
        help='Path to the escalation queue directory (parent of archive/).',
    )
    parser.add_argument(
        '--retention-days',
        type=int,
        default=30,
        help='Delete archive subdirs older than this many days (default: 30).',
    )
    args = parser.parse_args(argv)

    if not args.queue_dir.exists():
        logger.error('queue-dir does not exist: %s', args.queue_dir)
        return 2

    count = prune_archive(args.queue_dir, args.retention_days)
    archive_path = args.queue_dir / ARCHIVE_SUBDIR
    logger.info(
        'Pruned %d archive dir(s) older than %d days in %s',
        count,
        args.retention_days,
        archive_path,
    )
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    sys.exit(main())
