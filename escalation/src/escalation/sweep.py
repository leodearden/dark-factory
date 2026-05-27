"""One-time sweep tool: relocate resolved/dismissed escalations from queue root to archive.

IMPORTANT: This tool is intended for a single manual deploy run only.
Do NOT run against live escalation queues from an autonomous agent.

Usage:
  python -m escalation.sweep --queue-dir <path>          # dry-run (default)
  python -m escalation.sweep --queue-dir <path> --apply  # actually moves files
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from escalation import archive
from escalation.models import Escalation

logger = logging.getLogger(__name__)


@dataclass
class SweepReport:
    archived: int = 0
    reconciled_root_wins: int = 0
    reconciled_archive_wins: int = 0
    untouched_pending: int = 0
    skipped_unparsable: int = 0
    root_before: int = 0
    root_after: int = 0


def sweep(queue_dir: Path, *, apply: bool = False) -> SweepReport:
    """Sweep resolved/dismissed escalations from queue root to archive.

    Args:
        queue_dir: Root queue directory (parent of archive/).
        apply: If False (default), classify and report without mutating disk.
               If True, actually move/remove files.

    Returns:
        SweepReport with counts for each action category.
    """
    queue_dir = Path(queue_dir)
    report = SweepReport()

    root_files = list(queue_dir.glob('esc-*.json'))
    report.root_before = len(root_files)

    for path in root_files:
        esc = Escalation.from_json(path.read_text())

        if esc.status == 'pending':
            report.untouched_pending += 1
            continue

        if esc.status in ('resolved', 'dismissed'):
            target_dir = archive.archive_dir_for_date(queue_dir, esc.resolved_at)
            if apply:
                target_dir.mkdir(parents=True, exist_ok=True)
                os.replace(path, target_dir / path.name)
            report.archived += 1
            continue

    report.root_after = (
        report.root_before
        - report.archived
        - report.reconciled_root_wins
        - report.reconciled_archive_wins
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: python -m escalation.sweep."""
    parser = argparse.ArgumentParser(
        description='Sweep resolved/dismissed escalations from queue root to archive.',
    )
    parser.add_argument(
        '--queue-dir',
        required=True,
        type=Path,
        help='Path to the escalation queue directory (parent of archive/).',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        default=False,
        help='Actually move files (default: dry-run only).',
    )
    args = parser.parse_args(argv)

    if not args.queue_dir.exists():
        logger.error('queue-dir does not exist: %s', args.queue_dir)
        return 2

    report = sweep(args.queue_dir, apply=args.apply)
    logger.info(
        'Sweep %s: archived=%d reconciled(root_wins=%d, archive_wins=%d) pending=%d skipped=%d; root: %d → %d',
        'APPLIED' if args.apply else 'DRY-RUN',
        report.archived,
        report.reconciled_root_wins,
        report.reconciled_archive_wins,
        report.untouched_pending,
        report.skipped_unparsable,
        report.root_before,
        report.root_after,
    )
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    sys.exit(main())
