"""Escalation queue layout: archive subdirectory naming and terminal status names.

The stdlib-only statement, for readers outside the ``escalation`` package, of
where a resolved escalation record is archived
(``<queue_dir>/archive/<YYYY-MM-DD>/``) and of the spellings of the terminal
escalation statuses, so a process that holds no queue can read the layout
without importing that package.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    'ARCHIVE_SUBDIR',
    'DATE_FORMAT',
    'RESOLVED',
    'DISMISSED',
    'TERMINAL_STATUS_NAMES',
    'archive_dir_for_date',
]

ARCHIVE_SUBDIR = 'archive'
DATE_FORMAT = '%Y-%m-%d'

RESOLVED = 'resolved'
DISMISSED = 'dismissed'
TERMINAL_STATUS_NAMES = frozenset({RESOLVED, DISMISSED})


def archive_dir_for_date(queue_dir: Path, resolved_at_iso: str) -> Path:
    """Return the archive subdirectory for a ``resolved_at`` ISO 8601 timestamp.

    Pure path computation: does NOT create the directory. The date is the
    **UTC** date: a timezone-aware timestamp is converted to UTC first, so a
    ``+05:00`` timestamp near midnight is bucketed by UTC day, not local day.
    A naive timestamp is taken as already UTC.
    """
    dt = datetime.fromisoformat(resolved_at_iso)
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC)
    return Path(queue_dir) / ARCHIVE_SUBDIR / dt.strftime(DATE_FORMAT)
