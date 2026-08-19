#!/usr/bin/env python3
"""Retention GC sweep over the agent-transcript archive.

Task 2731 (δ of plans/agent-transcript-archival-prd.md). The α producer hook
(:mod:`shared.transcript_archive`, task 2742) copies each finished agent
session's transcripts to a durable archive root outside the per-task worktree,
laid out as ``<root>/<task_id>/<enc>/<session_id>.jsonl`` with each archived
copy carrying the SOURCE transcript's mtime (mirrored via ``os.utime``). Nothing
prunes that tree, so it grows without bound; this sweep is the δ consumer that
enforces α's ``retention.*`` config knobs (``max_age_days`` / ``max_task_dirs``)
to keep archive disk bounded.

Retention model
---------------
Each immediate child dir of ``<root>`` is one per-task archive dir. Its
retention "age" is the newest ``.jsonl`` mtime among its descendants
(recursive), falling back to the dir's own mtime when it holds no files —
because α mirrors the source transcript mtime onto each archived copy, that
newest mtime is the real time the task's most recent agent session last wrote,
and is stable across re-archival.

A task dir is pruned when it fails EITHER bound (union):

* **age cap** — its age exceeds ``max_age_days`` (reason ``age``);
* **count cap** — it falls outside the newest ``max_task_dirs`` dirs, oldest
  first (reason ``count``).

A dir failing both is tagged ``age+count``. A NON-POSITIVE cap DISABLES that
axis (fail-safe: a mis-set ``0`` imposes no bound and never wipes the whole
archive rather than being read as "keep zero"). Both caps non-positive is a
pure no-op sweep.

Posture: best-effort, LOUD, never-raise (mirrors α's archival posture and
docs/legibility/design-invariants.md INV-4 loud-over-silent). Every dropped
dir and a summary count are logged with a stable greppable ``gc_agent_transcripts:``
prefix; a per-dir ``shutil.rmtree`` failure is logged at WARNING and counted
while its siblings are still pruned; the run always exits 0. An empty or absent
archive root is a no-op.

Usage
-----
  # Dry-run (default via --check): scan + classify + log would-prune, delete
  # nothing, exit 0. Doubles as the delivered_check for a dependent task
  # (kind='script', args=['--check'], expect exit 0).
  python3 scripts/gc_agent_transcripts.py --check

  # Enforce retention against the default archive root and caps.
  python3 scripts/gc_agent_transcripts.py

  # Override root / caps.
  python3 scripts/gc_agent_transcripts.py --root /path/to/archive \
      --max-age-days 90 --max-task-dirs 5000

  # Deterministic age reference clock (tests / reproducible runs).
  python3 scripts/gc_agent_transcripts.py --now 1000000000

Scheduling
----------
Resolves PRD open-question 3 as "default standalone script + a documented
cron/watchdog hook"; the exact cadence and an actual systemd timer are
deferred/out-of-scope for this task. Run it standalone as above, or wire the
documented hook: invoke ``scripts/gc_agent_transcripts.py`` on a low cadence
(e.g. a daily user cron entry, or a periodic call from the orchestrator
watchdog alongside the other disk-hygiene sweeps). Because the sweep is
best-effort and always exits 0, a cron/watchdog wrapper will not alarm on
routine per-dir hiccups — the machine-readable failure list in the JSON report
carries that signal instead.

This module is STDLIB-ONLY by design: it does NOT import/load
``OrchestratorConfig`` (``load_config`` is config-required and side-effectful),
so it runs standalone in any environment. Its default root/caps are module
constants that MIRROR α's config, pinned against silent drift by a drift-guard
test against ``orchestrator.config.TranscriptArchiveConfig`` / ``RetentionConfig``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger('gc_agent_transcripts')

# Stable greppable prefix on every operator-facing log line.
_LOG_PREFIX = 'gc_agent_transcripts:'

SECONDS_PER_DAY = 86_400

# Stdlib-only mirrors of α's canonical retention config
# (orchestrator.config.TranscriptArchiveConfig / RetentionConfig). Pinned
# against silent drift by test_gc_agent_transcripts.py's drift-guard test —
# see the module docstring for why the GC does not import OrchestratorConfig.
# drain_check.py's DEFAULT_FLEET_DIR sets the precedent for the hardcoded
# absolute project-root default.
DEFAULT_PROJECT_ROOT = Path('/home/leo/src/dark-factory')
# Mirrors TranscriptArchiveConfig().root — resolved against project_root, NOT
# the per-task worktree (the archive is a durable, git-ignored data dir).
ARCHIVE_ROOT_RELATIVE = 'data/orchestrator/agent-transcripts'
DEFAULT_ARCHIVE_ROOT = DEFAULT_PROJECT_ROOT / ARCHIVE_ROOT_RELATIVE
# Mirrors RetentionConfig().max_age_days / .max_task_dirs.
DEFAULT_MAX_AGE_DAYS = 90
DEFAULT_MAX_TASK_DIRS = 5000


def default_archive_root() -> Path:
    """Return the default archive root (``project_root / ARCHIVE_ROOT_RELATIVE``).

    The single resolution site for the CLI ``--root`` default, mirroring the
    ``resolve_*`` helper shape of :mod:`drain_check`. ``--root`` overrides it;
    there is no env knob (the archive root is a fixed data-dir convention).
    """
    return DEFAULT_ARCHIVE_ROOT


@dataclass
class GcDecision:
    """Classification of scanned task dirs into keep / prune sets.

    ``keep`` is the list of task-dir paths retained. ``prune`` pairs each
    dropped task-dir path with its reason (``age`` / ``count`` / ``age+count``).
    Pure value object — no I/O.
    """

    keep: list[Path] = field(default_factory=list)
    prune: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def keep_paths(self) -> set[Path]:
        """Set of retained task-dir paths."""
        return set(self.keep)

    @property
    def prune_paths(self) -> set[Path]:
        """Set of pruned task-dir paths."""
        return {path for path, _reason in self.prune}

    @property
    def reasons(self) -> dict[Path, str]:
        """Map each pruned task-dir path to its prune reason."""
        return {path: reason for path, reason in self.prune}


def select_prunable(
    task_dirs: list[tuple[Path, float]],
    now: float,
    max_age_days: int,
    max_task_dirs: int,
) -> GcDecision:
    """Classify ``(path, mtime)`` task dirs into keep / prune sets (pure).

    Prune = UNION of the two bounds; a dir failing either (or both) is dropped:

    * **age** — ``max_age_days > 0`` and ``mtime`` is strictly older than
      ``now - max_age_days * 86400``. A dir sitting exactly on the cutoff
      (``now - mtime == max_age_days*86400``) is KEPT (strict ``<``), and one
      second older is pruned.
    * **count** — dirs are ranked newest-first by ``mtime`` (deterministic
      tiebreak on path); a dir whose newest-first rank is ``>= max_task_dirs``
      falls outside the retained window.

    A non-positive cap disables that axis entirely (fail-safe: a mis-set ``0``
    imposes no bound rather than pruning everything). The prune reason is
    tagged ``age`` / ``count`` / ``age+count`` accordingly; ``keep`` is the
    complement. Result ordering is newest-first and deterministic regardless of
    input order.
    """
    age_cutoff: float | None = None
    if max_age_days > 0:
        age_cutoff = now - max_age_days * SECONDS_PER_DAY

    # Newest-first rank (mtime desc, path asc as a deterministic tiebreak).
    ordered = sorted(task_dirs, key=lambda item: (-item[1], str(item[0])))

    keep: list[Path] = []
    prune: list[tuple[Path, str]] = []
    for rank, (path, mtime) in enumerate(ordered):
        is_age = age_cutoff is not None and mtime < age_cutoff
        is_count = max_task_dirs > 0 and rank >= max_task_dirs
        if is_age and is_count:
            prune.append((path, 'age+count'))
        elif is_age:
            prune.append((path, 'age'))
        elif is_count:
            prune.append((path, 'count'))
        else:
            keep.append(path)

    return GcDecision(keep=keep, prune=prune)


def _warn_stat_skip(kind: str, path: Path, err: OSError) -> None:
    """LOUD, greppable WARNING for a ``stat`` that failed mid-scan (best-effort).

    The archive is written CONCURRENTLY by the α producer hook, so a path can
    vanish (or briefly become unreadable) between the directory walk and the
    ``stat`` call — a classic TOCTOU race. :func:`scan_task_dirs` skips the
    offending *path* and keeps sweeping rather than aborting, mirroring
    :func:`prune_task_dirs`' per-dir ``OSError`` posture and the module's
    never-raise / always-exit-0 contract. Every line carries the greppable
    ``gc_agent_transcripts:`` prefix.
    """
    logger.warning(
        '%s failed to stat %s %s (errno=%s): %s — skipping',
        _LOG_PREFIX,
        kind,
        path,
        err.errno,
        err,
    )


def scan_task_dirs(root: Path) -> list[tuple[Path, float]]:
    """Scan the archive *root* into ``(task_dir, mtime)`` pairs (filesystem I/O).

    Each immediate child DIR of *root* is one per-task archive dir. Its
    reported mtime is the NEWEST descendant file mtime (recursive), which —
    because α mirrors the source transcript mtime onto each archived copy —
    measures the task's most recent agent-session activity. A task dir holding
    no files falls back to its own directory mtime. Stray non-directory entries
    directly under *root* are skipped.

    Returns ``[]`` when *root* is absent or is not a directory (so a missing or
    empty archive is a clean no-op). Output is sorted by task-dir path for
    deterministic ordering.

    Best-effort + never-raise: because the archive is written concurrently by
    the α producer hook, any per-entry ``stat`` may fail with an ``OSError``
    (a file/dir that vanishes or becomes unreadable between the walk and the
    ``stat`` — a TOCTOU race). Every such failure is skipped LOUDLY (see
    :func:`_warn_stat_skip`) and the sweep continues, so one torn-down path
    can never abort the scan or violate the module's always-exit-0 contract.
    """
    root = Path(root)
    if not root.is_dir():
        return []

    scanned: list[tuple[Path, float]] = []
    for child in sorted(root.iterdir()):
        try:
            if not child.is_dir():
                continue
        except OSError as err:
            # The child vanished / became unstattable between iterdir and here.
            _warn_stat_skip('archive entry', child, err)
            continue

        # Newest descendant FILE mtime, guarding EVERY per-file stat: an
        # archived transcript can vanish (or briefly become unreadable) between
        # the rglob walk and the stat call. A failed file is skipped LOUDLY;
        # the remaining readable files still set the dir's age.
        file_mtimes: list[float] = []
        for descendant in child.rglob('*'):
            try:
                if descendant.is_file():
                    file_mtimes.append(descendant.stat().st_mtime)
            except OSError as err:
                _warn_stat_skip('archive file', descendant, err)

        if file_mtimes:
            mtime = max(file_mtimes)
        else:
            # No readable files — fall back to the dir's own mtime, guarding
            # that stat too (the dir itself may have just vanished).
            try:
                mtime = child.stat().st_mtime
            except OSError as err:
                _warn_stat_skip('task dir', child, err)
                continue
        scanned.append((child, mtime))
    return scanned


@dataclass
class PruneOutcome:
    """Result of a :func:`prune_task_dirs` sweep.

    ``removed`` lists the task-dir paths successfully deleted; ``failed`` lists
    those whose ``shutil.rmtree`` raised ``OSError`` (logged at WARNING and
    counted, never re-raised). In a dry-run both stay empty. Pure value object.
    """

    removed: list[Path] = field(default_factory=list)
    failed: list[Path] = field(default_factory=list)


def prune_task_dirs(
    prune_records: list[tuple[Path, str]],
    *,
    dry_run: bool,
) -> PruneOutcome:
    """Delete each pruned task dir, best-effort and LOUD (filesystem I/O).

    Iterates ``(path, reason)`` records. In ``dry_run`` mode nothing is
    deleted — each dir is logged with a stable ``would prune`` INFO line and
    skipped. Otherwise each dir is removed via ``shutil.rmtree`` inside a
    ``try/except OSError``: a success logs a LOUD per-removal INFO line and is
    appended to ``removed``; an ``OSError`` is logged at WARNING (path / reason
    / errno) and appended to ``failed`` while the sweep continues. The sweep
    NEVER re-raises, so one wedged dir cannot stop its siblings from being
    pruned. Every line carries the greppable ``gc_agent_transcripts:`` prefix.
    """
    outcome = PruneOutcome()
    for path, reason in prune_records:
        if dry_run:
            logger.info(
                '%s would prune task dir %s (reason=%s)', _LOG_PREFIX, path, reason
            )
            continue
        try:
            shutil.rmtree(path)
        except OSError as err:
            logger.warning(
                '%s failed to prune task dir %s (reason=%s, errno=%s): %s',
                _LOG_PREFIX,
                path,
                reason,
                err.errno,
                err,
            )
            outcome.failed.append(path)
        else:
            logger.info(
                '%s pruned task dir %s (reason=%s)', _LOG_PREFIX, path, reason
            )
            outcome.removed.append(path)
    return outcome


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI arg parser (unit-testable in isolation).

    ``--root`` defaults to :func:`default_archive_root`; the caps default to the
    drift-guarded α mirrors; ``--check`` is a dry-run flag; ``--now`` overrides
    the age reference clock (epoch seconds) for deterministic/reproducible runs.
    """
    parser = argparse.ArgumentParser(
        prog='gc_agent_transcripts.py',
        description=(
            'Retention GC sweep over the agent-transcript archive '
            '(age/count cap, best-effort, LOUD).'
        ),
    )
    parser.add_argument(
        '--root',
        default=str(default_archive_root()),
        help='Archive root to sweep (default: %(default)s).',
    )
    parser.add_argument(
        '--max-age-days',
        type=int,
        default=DEFAULT_MAX_AGE_DAYS,
        help=(
            'Prune task dirs older than this many days; a non-positive value '
            'disables the age axis (default: %(default)s).'
        ),
    )
    parser.add_argument(
        '--max-task-dirs',
        type=int,
        default=DEFAULT_MAX_TASK_DIRS,
        help=(
            'Keep at most this many newest task dirs, pruning older ones; a '
            'non-positive value disables the count axis (default: %(default)s).'
        ),
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help=(
            'Dry-run: scan + classify + log would-prune lines, delete nothing, '
            'exit 0. Doubles as the delivered_check for a dependent task.'
        ),
    )
    parser.add_argument(
        '--now',
        type=float,
        default=time.time(),
        help=(
            'Age reference clock in epoch seconds (default: the current time). '
            'Pin it for deterministic age-based pruning in tests.'
        ),
    )
    return parser


def build_gc_report(
    root: Path,
    scanned: list[tuple[Path, float]],
    decision: GcDecision,
    outcome: PruneOutcome,
    caps: tuple[int, int],
    now: float,
    check: bool,
) -> dict:
    """Assemble the machine-readable JSON report (pure — no I/O).

    ``removed`` is the count actually deleted (0 in a dry-run); ``pruned`` is
    the count CLASSIFIED as prunable (non-zero even in a dry-run). ``failed``
    carries the best-effort per-dir ``rmtree`` failures — the machine-readable
    failure signal (INV-4 loud-over-silent) that a cron/watchdog wrapper reads
    without alarming on the always-0 exit code.
    """
    max_age_days, max_task_dirs = caps
    reason_counts: dict[str, int] = {}
    for _path, reason in decision.prune:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        'root': str(root),
        'now': now,
        'check': check,
        'caps': {'max_age_days': max_age_days, 'max_task_dirs': max_task_dirs},
        'scanned': len(scanned),
        'kept': len(decision.keep),
        'pruned': len(decision.prune),
        'reason_counts': reason_counts,
        'removed': len(outcome.removed),
        'removed_paths': [str(p) for p in outcome.removed],
        'failed': len(outcome.failed),
        'failed_paths': [str(p) for p in outcome.failed],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the sweep: scan → classify → prune (or dry-run), report, exit 0.

    Best-effort and never-raising by design — the JSON report on stdout and the
    LOUD summary on stderr carry the signal (including any per-dir failures);
    the process always exits 0 so a cron/watchdog hook does not alarm on routine
    per-dir hiccups. ``--check`` never deletes.
    """
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)

    root = Path(args.root)
    scanned = scan_task_dirs(root)
    decision = select_prunable(
        scanned, args.now, args.max_age_days, args.max_task_dirs
    )
    outcome = prune_task_dirs(decision.prune, dry_run=args.check)

    report = build_gc_report(
        root=root,
        scanned=scanned,
        decision=decision,
        outcome=outcome,
        caps=(args.max_age_days, args.max_task_dirs),
        now=args.now,
        check=args.check,
    )
    print(json.dumps(report))

    logger.info(
        '%s sweep complete — root=%s scanned=%d kept=%d pruned=%d removed=%d '
        'failed=%d check=%s',
        _LOG_PREFIX,
        root,
        report['scanned'],
        report['kept'],
        report['pruned'],
        report['removed'],
        report['failed'],
        args.check,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
