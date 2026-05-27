"""One-time sweep tool: relocate resolved/dismissed escalations from queue root to archive.

IMPORTANT: This tool is intended for a single manual deploy run only.
Do NOT run against live escalation queues from an autonomous agent.

Usage:
  python -m escalation.sweep --queue-dir <path>          # dry-run (default)
  python -m escalation.sweep --queue-dir <path> --apply  # actually moves files
"""

from __future__ import annotations

import argparse
import json
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


def _pick_richer(root_esc: Escalation, archive_esc: Escalation) -> bool:
    """Return True iff root is strictly richer than archive; ties go to archive."""
    root_key = (root_esc.resolved_by is not None, root_esc.resolution_turns is not None)
    archive_key = (archive_esc.resolved_by is not None, archive_esc.resolution_turns is not None)
    return root_key > archive_key


def _build_archive_index(archive_root: Path) -> dict[str, Path]:
    """Return {stem: path} index of all esc-*.json files under archive_root."""
    if not archive_root.exists():
        return {}
    return {p.stem: p for p in archive_root.rglob('esc-*.json')}


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

    archive_root = queue_dir / archive.ARCHIVE_SUBDIR
    archive_index = _build_archive_index(archive_root)

    for path in root_files:
        try:
            esc = Escalation.from_json(path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning('skipping unparsable %s: %s', path.name, e)
            report.skipped_unparsable += 1
            continue

        if esc.status == 'pending':
            report.untouched_pending += 1
            continue

        if esc.status in ('resolved', 'dismissed'):
            if not esc.resolved_at:
                logger.warning(
                    'skipping %s: status %s but resolved_at is missing', path.name, esc.status
                )
                report.skipped_unparsable += 1
                continue
            existing_archive = archive_index.get(path.stem)

            if existing_archive is None:
                # No archive copy: move to dated subdir
                target_dir = archive.archive_dir_for_date(queue_dir, esc.resolved_at)
                if apply:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    os.replace(path, target_dir / path.name)
                report.archived += 1
            else:
                # Archive copy exists: compare richness
                archive_esc = Escalation.from_json(existing_archive.read_text())
                if _pick_richer(esc, archive_esc):
                    # Root wins: atomically overwrite the archive slot
                    if apply:
                        os.replace(path, existing_archive)
                    report.reconciled_root_wins += 1
                else:
                    # Archive wins: drop the duplicate root copy
                    if apply:
                        os.unlink(path)
                    report.reconciled_archive_wins += 1
            continue

        # Defensive: unknown status (should not occur given models.py constraints)
        logger.warning('skipping %s: unknown status %r', path.name, esc.status)
        report.skipped_unparsable += 1

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
