"""Sweep tool: relocate resolved/dismissed escalations from queue root to archive.

Relocates resolved/dismissed escalations from queue root to archive.
Run at the escalation server-start single-writer window (pre-serving) —
the one window with no concurrent queue writers.

Usage:
  python -m escalation.sweep --queue-dir <path>          # dry-run (default)
  python -m escalation.sweep --queue-dir <path> --apply  # actually moves files
"""

from __future__ import annotations

import argparse
import errno
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from escalation import archive
from escalation.models import Escalation
from escalation.queue import escalation_id_lock

logger = logging.getLogger(__name__)


@dataclass
class SweepReport:
    archived: int = 0
    reconciled_root_wins: int = 0
    reconciled_archive_wins: int = 0
    untouched_pending: int = 0
    # Counts files left in root for any skip reason: unparsable JSON, missing
    # resolved_at on a resolved/dismissed file, or unknown status value.
    # Individual cases are logged at WARNING level for operator review.
    skipped_unparsable: int = 0
    root_before: int = 0
    root_after: int = 0


def _pick_richer(root_esc: Escalation, archive_esc: Escalation) -> bool:
    """Return True iff root is strictly richer than archive; ties go to archive.

    Richness is compared lexicographically by:
      1. resolved_by present
      2. resolution_turns present
      3. len(resolution text)  — guards against silent loss of free-form text
      4. dedupe_count          — guards against losing folded-duplicate metadata
    """
    def _key(e: Escalation) -> tuple:
        return (
            e.resolved_by is not None,
            e.resolution_turns is not None,
            len(e.resolution or ''),
            e.dedupe_count,
        )
    return _key(root_esc) > _key(archive_esc)


def _build_archive_index(archive_root: Path) -> dict[str, Path]:
    """Return {stem: path} index of all esc-*.json files under archive_root.

    If the same escalation id appears in multiple dated subdirs (e.g. from a
    prior partial run or a bad import), the last path discovered wins and a
    WARNING is logged.  The operator should investigate and deduplicate manually
    before running --apply; the sweep will still proceed with the indexed copy.
    """
    if not archive_root.exists():
        return {}
    index: dict[str, Path] = {}
    for p in archive_root.rglob('esc-*.json'):
        if p.stem in index:
            logger.warning(
                'archive duplicate: %s exists at %s AND %s; using latter for reconciliation',
                p.stem, index[p.stem], p,
            )
        index[p.stem] = p
    return index


def _atomic_move(src: Path, dst: Path) -> None:
    """Move src to dst atomically where possible, with a cross-device fallback.

    os.replace is atomic on the same filesystem but raises OSError(EXDEV) when
    src and dst are on different devices (e.g. archive is a bind mount).  In
    that case we fall back to shutil.move (copy + unlink) and log a warning.
    """
    try:
        os.replace(src, dst)
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            logger.warning(
                'cross-device move detected; falling back to shutil.move for %s → %s', src, dst
            )
            shutil.move(str(src), str(dst))
        else:
            raise


def _relocate_terminal(
    queue_dir: Path,
    path: Path,
    esc: Escalation,
    *,
    apply: bool = True,
) -> bool:
    """Relocate a terminal esc file to its dated archive subdir.

    Shared by ``sweep()``'s archive-missing branch and ``reap_loose_archive_files()``
    to avoid the two copies drifting independently.

    Performs target path computation, collision guard (checked **inside** the
    per-id lock to close the TOCTOU window), and the locked atomic move.

    Pre-conditions (verified by caller):
        - ``esc.status`` is ``'resolved'`` or ``'dismissed'``
        - ``esc.resolved_at`` is not ``None``

    Args:
        queue_dir: Root queue directory (parent of ``archive/``).
        path: Source file path to relocate.
        esc: Parsed escalation record.
        apply: If True, actually move the file; if False, only check (dry-run).

    Returns:
        ``True`` if the file was (or would be) moved; ``False`` if skipped due
        to a collision — the existing target is left untouched.
    """
    target_dir = archive.archive_dir_for_date(queue_dir, esc.resolved_at)
    target_path = target_dir / path.name

    if apply:
        with escalation_id_lock(queue_dir, path.stem):
            # Re-check inside the lock to close the TOCTOU window: a concurrent
            # writer could have created target_path between our pre-check and
            # acquiring the lock.
            if target_path.exists():
                logger.warning(
                    'skipping %s: target already exists at %s (not overwriting)',
                    path.name, target_path,
                )
                return False
            target_dir.mkdir(parents=True, exist_ok=True)
            _atomic_move(path, target_path)
    else:
        # Dry-run: check for collisions without acquiring the lock.
        if target_path.exists():
            return False

    return True


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
                # No archive copy: move to dated subdir via shared helper.
                # The helper re-checks for collisions inside the per-id lock so
                # a file that appeared between index-build and lock-acquire is
                # detected atomically (TOCTOU safe).
                if _relocate_terminal(queue_dir, path, esc, apply=apply):
                    report.archived += 1
            else:
                # Archive copy exists: compare richness
                archive_esc = Escalation.from_json(existing_archive.read_text())
                if _pick_richer(esc, archive_esc):
                    # Root wins: atomically overwrite the archive slot
                    if apply:
                        with escalation_id_lock(queue_dir, path.stem):
                            _atomic_move(path, existing_archive)
                    report.reconciled_root_wins += 1
                else:
                    # Archive wins: drop the duplicate root copy
                    if apply:
                        with escalation_id_lock(queue_dir, path.stem):
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


@dataclass
class StartupSweepReport:
    """Aggregated report from a single run_startup_sweep call."""
    sweep: SweepReport
    loose_reaped: int = 0
    pruned_dirs: int = 0


def run_startup_sweep(
    queue_dir: Path,
    *,
    retention_days: int = archive.DEFAULT_RETENTION_DAYS,
    apply: bool = True,
    now: datetime | None = None,
) -> StartupSweepReport:
    """Orchestrate sweep + loose-reap + prune at escalation server start.

    Runs three passes in sequence:
      1. sweep(queue_dir, apply=apply)       — root→archive relocation
      2. reap_loose_archive_files(…)         — archive top-level→dated subdir
      3. archive.prune_archive(…)            — drop subdirs beyond retention

    Logs one INFO summary line on the ``escalation.sweep`` logger.

    Args:
        queue_dir: Root queue directory.
        retention_days: Retention threshold forwarded to prune_archive.
        apply: If False, dry-run all three passes (no disk mutations).
        now: Reference datetime for prune_archive cutoff (defaults to live UTC).

    Returns:
        StartupSweepReport with nested SweepReport plus loose_reaped and pruned_dirs.
    """
    queue_dir = Path(queue_dir)
    sweep_report = sweep(queue_dir, apply=apply)
    loose_reaped = reap_loose_archive_files(queue_dir, apply=apply)
    pruned_dirs = archive.prune_archive(queue_dir, retention_days, now=now) if apply else 0

    report = StartupSweepReport(
        sweep=sweep_report,
        loose_reaped=loose_reaped,
        pruned_dirs=pruned_dirs,
    )
    logger.info(
        'Startup sweep %s: archived=%d reconciled(root=%d archive=%d) '
        'loose_reaped=%d pruned_dirs=%d pending=%d skipped=%d; root: %d → %d',
        'APPLIED' if apply else 'DRY-RUN',
        sweep_report.archived,
        sweep_report.reconciled_root_wins,
        sweep_report.reconciled_archive_wins,
        loose_reaped,
        pruned_dirs,
        sweep_report.untouched_pending,
        sweep_report.skipped_unparsable,
        sweep_report.root_before,
        sweep_report.root_after,
    )
    return report


def reap_loose_archive_files(queue_dir: Path, *, apply: bool = True) -> int:
    """Relocate loose archive-top-level esc-*.json files into dated subdirs.

    A "loose" file is one that sits directly under ``archive/`` (not inside
    a ``YYYY-MM-DD`` dated subdir).  These accumulate when the queue's
    ``_archive_resolved`` fast-path is bypassed or when a prior sweep run
    wrote files without the dated-subdir structure.

    Each relocation is serialized per-id by ``escalation_id_lock`` so it
    does not race concurrent queue writers that hold the same id lock.

    Args:
        queue_dir: Root queue directory (parent of ``archive/``).
        apply: If True (default), actually move files.  If False, only count
               how many would be moved (dry-run).

    Returns:
        Number of files moved (or would-move when apply=False).

    Skips (with WARNING):
        - Files whose JSON cannot be parsed
        - Files whose status is not resolved/dismissed
        - Files missing ``resolved_at``
        - Files where a same-name file already exists in the target dated dir
          (never overwrite / lose data)
    """
    queue_dir = Path(queue_dir)
    archive_root = queue_dir / archive.ARCHIVE_SUBDIR
    if not archive_root.exists():
        return 0

    moved = 0
    # Materialize the glob before iterating so the directory scan completes
    # before any moves occur — mutating a directory during os.scandir/readdir
    # has unspecified behaviour on some filesystems and can cause subsequent
    # sibling entries to be skipped (mirrors sweep()'s `root_files = list(...)` pattern).
    # Only glob the archive top level — non-recursive, so dated-subdir files
    # are excluded (the glob `esc-*.json` does not recurse into subdirs).
    for path in list(archive_root.glob('esc-*.json')):
        try:
            esc = Escalation.from_json(path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning('reap_loose: skipping unparsable %s: %s', path.name, e)
            continue

        if esc.status not in ('resolved', 'dismissed'):
            logger.warning(
                'reap_loose: skipping %s: non-terminal status %r', path.name, esc.status
            )
            continue

        if not esc.resolved_at:
            logger.warning(
                'reap_loose: skipping %s: resolved_at is missing', path.name
            )
            continue

        # _relocate_terminal handles the collision guard inside the per-id lock
        # (closing the TOCTOU window) and performs the atomic move.
        if _relocate_terminal(queue_dir, path, esc, apply=apply):
            moved += 1

    return moved


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
