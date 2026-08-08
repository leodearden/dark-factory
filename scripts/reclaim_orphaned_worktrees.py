#!/usr/bin/env python3
"""Periodic verified reclaim of ``.worktrees-orphaned/`` parkings (task 2980).

Automates the 2026-07-23 manual reclaim of the orchestrator's quarantine base
as a recurring, safety-preserving sweep. The PRODUCER is
:meth:`orchestrator.git_ops.GitOps.quarantine_worktree`: when a worktree is
orphaned it is MOVED (``git worktree move``, so it stays a fully registered
worktree) to ``<project_root>/.worktrees-orphaned/<branch>-<ts>`` and its branch
is renamed to ``task/<branch>-<ts>``, where
``ts = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')`` is the PARKING time. Each
parking therefore carries a preserved WIP commit on its own branch. Nothing
prunes that tree, so parkings accumulate without bound (~1.1G / 21 entries at
the 07-23 manual reclaim); this sweep is the consumer that reclaims them.

Age model
---------
Each parking's age is derived from the trailing ``-<YYYYMMDDTHHMMSSZ>`` stamp in
its DIRECTORY basename (:func:`parse_parking_dir_name`) — the parking time
stamped by ``quarantine_worktree`` — NOT the branch-creation time (which can be
days older than the parking) and NOT the directory mtime (perturbable by later
reads/writes). This is the only signal that faithfully skips freshly-parked
worktrees still referenced by in-flight triage (the 07-23 run deliberately
skipped a 1h-old parking). A basename with no valid trailing stamp is SKIPPED
with a LOUD warning — never guess an age.

A parking is reclaimed when its age STRICTLY exceeds ``--min-age-hours``
(default 48; the exact-boundary parking is KEPT). A NON-POSITIVE floor reclaims
NOTHING — the fail-safe direction is the OPPOSITE of gc_agent_transcripts
(where non-positive DISABLES the axis and keeps everything), because HERE the
age floor is the ONLY protection against reclaiming an in-flight parking, so a
mis-set ``0``/negative floor must never wipe fresh parkings.

Safety (zero content lost)
--------------------------
Per parked worktree, in order: (1) require a RESOLVABLE branch from
``git worktree list --porcelain`` (a detached / unresolvable-branch parking is
SKIPPED + logged, never removed); (2) if ``git status --porcelain`` is
non-empty, park-commit everything (``git add -A && git commit --no-verify``)
onto its branch FIRST (a ``git status`` error is treated fail-safe as
dirty->skip, mirroring ``GitOps._worktree_dirty``); (3) verify
``git rev-parse --verify refs/heads/<branch>`` resolves (content now provably
safe on a ref INDEPENDENT of the worktree); (4) ``git worktree remove --force``;
(5) ``git worktree prune`` after all removals. NEVER deletes branches; only ever
removes paths ``git worktree list`` reports as REGISTERED and under the parking
root (a band guard mirroring ``GitOps._refuse_foreign_band``). Non-registered
"skeleton" dirs are OUT of scope (too data-loss-risky to automate) — skipped +
logged for manual handling.

Posture: best-effort, LOUD, never-raise, always-exit-0 (mirrors
scripts/gc_agent_transcripts.py and docs/legibility/design-invariants.md INV-4
loud-over-silent). Every skip / removal / failure and a summary count are logged
with a stable greppable ``reclaim_orphaned_worktrees:`` prefix; a per-worktree
failure is logged + counted while its siblings are still reclaimed; the run
always exits 0. An empty or absent parking root is a no-op.

Usage
-----
  # Dry-run: scan + classify + log would-reclaim, remove/commit nothing, exit 0.
  python3 scripts/reclaim_orphaned_worktrees.py --check

  # Reclaim eligible parkings under the default repo's quarantine base.
  python3 scripts/reclaim_orphaned_worktrees.py

  # Override repo / parking root / age floor.
  python3 scripts/reclaim_orphaned_worktrees.py --repo /path/to/repo \
      --parking-root /path/to/.worktrees-orphaned --min-age-hours 72

  # Deterministic age reference clock (tests / reproducible runs).
  python3 scripts/reclaim_orphaned_worktrees.py --now 1000000000

This module is STDLIB-ONLY by design: it does NOT import/construct
``OrchestratorConfig`` / ``GitOps`` (both config-required and side-effectful) —
it re-implements the few git operations it needs via ``subprocess``, so it runs
standalone in any environment (and the wrapper needs no ``uv``/service-env,
unlike the flag-marker sweep). It reuses only the PRODUCER's naming FACTS.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger('reclaim_orphaned_worktrees')

# Stable greppable prefix on every operator-facing log line.
_LOG_PREFIX = 'reclaim_orphaned_worktrees:'

# The hardcoded absolute project-root default (drain_check.py / gc_agent_
# transcripts.py set the precedent). Overridable via --repo.
DEFAULT_PROJECT_ROOT = '/home/leo/src/dark-factory'

# The quarantine base is a direct sibling of the worktree dir (mirrors
# GitOps.quarantine_base: ``<worktree_dir>-orphaned``). Derived per-repo in
# main() as ``<repo>/.worktrees-orphaned`` unless --parking-root overrides it.
PARKING_ROOT_NAME = '.worktrees-orphaned'

# Default age floor. 48h sits in the task's suggested 48-72h band and clears the
# ~1h triage-reference window. A NON-POSITIVE floor reclaims NOTHING.
DEFAULT_MIN_AGE_HOURS = 48.0

# Trailing parking stamp: quarantine_worktree names each parking dir
# ``<branch>-<%Y%m%dT%H%M%SZ>``; anchor at END so a lane id embedding its own
# hyphens (``_lane-0-<ts>``) still parses off the final stamp.
_PARKING_STAMP_RE = re.compile(r'-(\d{8}T\d{6}Z)$')
_PARKING_STAMP_FORMAT = '%Y%m%dT%H%M%SZ'


def parse_parking_dir_name(name: str) -> datetime | None:
    """Parse a parking dir BASENAME's trailing stamp into a tz-aware UTC datetime.

    ``quarantine_worktree`` stamps each parking dir basename as
    ``<branch>-<%Y%m%dT%H%M%SZ>`` at the moment of parking, so the trailing
    stamp is the true "parked-at" time. The regex anchors ``-(\\d{8}T\\d{6}Z)$``
    at the END of the basename, so both ``2920-<ts>`` and ``_lane-0-<ts>`` (the
    lane id embeds its own hyphens) parse off the FINAL stamp.

    Returns a timezone-aware UTC :class:`datetime`, or ``None`` when the
    basename has no valid trailing stamp OR the stamp is calendar-invalid
    (``strptime`` ``ValueError``) — the parse is TOTAL and never raises, so an
    unparseable parking is skipped rather than crashing the sweep.
    """
    match = _PARKING_STAMP_RE.search(name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), _PARKING_STAMP_FORMAT).replace(
            tzinfo=UTC
        )
    except ValueError:
        return None


@dataclass
class ParkedWorktree:
    """One registered worktree under the quarantine base (a parking).

    ``path`` is its resolved on-disk location; ``branch`` is the local branch it
    has checked out (``None`` if detached / unresolvable in
    ``git worktree list --porcelain``); ``parked_at`` is the tz-aware UTC parking
    time parsed from the dir basename's trailing stamp. Pure value object.
    """

    path: Path
    branch: str | None
    parked_at: datetime


@dataclass
class ReclaimDecision:
    """Classification of scanned parkings into keep / reclaim sets.

    ``keep`` is the list of retained :class:`ParkedWorktree` records. ``reclaim``
    pairs each eligible record with its reason (currently always ``age``). Pure
    value object — no I/O.
    """

    keep: list[ParkedWorktree] = field(default_factory=list)
    reclaim: list[tuple[ParkedWorktree, str]] = field(default_factory=list)

    @property
    def keep_paths(self) -> set[Path]:
        """Set of retained parking paths."""
        return {record.path for record in self.keep}

    @property
    def reclaim_paths(self) -> set[Path]:
        """Set of reclaimable parking paths."""
        return {record.path for record, _reason in self.reclaim}

    @property
    def reasons(self) -> dict[Path, str]:
        """Map each reclaimable parking path to its reason."""
        return {record.path: reason for record, reason in self.reclaim}


def select_reclaimable(
    records: list[ParkedWorktree],
    now: float,
    min_age_hours: float,
) -> ReclaimDecision:
    """Classify parkings into keep / reclaim by the age FLOOR (pure).

    A parking is reclaimed when its age (``now - parked_at``, both in epoch
    seconds) is STRICTLY greater than ``min_age_hours * 3600`` — a parking
    sitting exactly on the floor is KEPT (strict ``>``, not ``>=``), and one
    second older is reclaimed. The reclaim reason is always ``age``.

    A NON-POSITIVE ``min_age_hours`` reclaims NOTHING (every record kept): here
    the age floor is the ONLY protection against reclaiming an in-flight
    parking, so a mis-set ``0``/negative floor must be a safe no-op — the
    OPPOSITE of gc_agent_transcripts' "non-positive disables the axis". The
    accompanying LOUD warning is emitted by :func:`main` (the single
    argument-resolution site), keeping this classifier pure.

    Output ordering is deterministic (oldest parking first, path as tiebreak)
    regardless of input order.
    """
    ordered = sorted(records, key=lambda r: (r.parked_at.timestamp(), str(r.path)))
    if min_age_hours <= 0:
        return ReclaimDecision(keep=list(ordered), reclaim=[])

    floor_seconds = min_age_hours * 3600
    keep: list[ParkedWorktree] = []
    reclaim: list[tuple[ParkedWorktree, str]] = []
    for record in ordered:
        age_seconds = now - record.parked_at.timestamp()
        if age_seconds > floor_seconds:
            reclaim.append((record, 'age'))
        else:
            keep.append(record)
    return ReclaimDecision(keep=keep, reclaim=reclaim)


def _run_git(args: list[str], cwd: Path) -> tuple[int, str, str]:
    """Run ``git <args>`` in *cwd*, returning ``(returncode, stdout, stderr)``.

    The single subprocess chokepoint used by EVERY git call in this module.
    Forces ``LC_ALL=C`` so porcelain output is locale-stable. A 120s ``timeout``
    bounds a hung git invocation (e.g. one blocking on an ``index.lock`` held by
    the concurrently-running orchestrator on the same repo). Best-effort +
    never-raise: an ``OSError`` (git binary missing, *cwd* vanished, etc.) OR a
    ``subprocess.TimeoutExpired`` (a hung call tripping the 120s bound —
    ``TimeoutExpired`` is a ``SubprocessError``, NOT an ``OSError``, so it must
    be caught explicitly) is mapped to a non-zero ``(1, '', <message>)`` rather
    than propagated. That keeps EVERY git chokepoint fail-safe — including the
    unguarded ``git worktree list`` in :func:`list_parked_worktrees` and the
    ``git worktree prune`` in :func:`main` — so the never-raise / always-exit-0
    posture holds even under routine git contention, without each caller needing
    its own ``try/except``.
    """
    env = dict(os.environ, LC_ALL='C')
    try:
        proc = subprocess.run(
            ['git', *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        return 1, '', str(err)
    return proc.returncode, proc.stdout, proc.stderr


def _is_under(path: Path, root: Path) -> bool:
    """Whether *path* is *root* itself or a descendant of it (band guard)."""
    try:
        return path == root or path.is_relative_to(root)
    except (OSError, ValueError):
        return False


def list_parked_worktrees(repo: Path, parking_root: Path) -> list[ParkedWorktree]:
    """Scan ``git worktree list --porcelain`` for parkings under *parking_root*.

    Parses each porcelain record's ``worktree <path>`` and ``branch
    refs/heads/<name>`` (``detached`` / ``bare`` records leave ``branch=None``),
    keeps only those whose RESOLVED path is under *parking_root* (the band guard
    — nothing outside the quarantine base is ever considered), and derives each
    parking's ``parked_at`` from its dir basename via
    :func:`parse_parking_dir_name`.

    A parking whose basename lacks a valid trailing stamp is SKIPPED with a LOUD
    WARNING (never guess an age). Best-effort + never-raise: a failed
    ``git worktree list`` (or absent parking root — no worktrees under it) yields
    ``[]`` LOUDLY rather than raising.
    """
    repo = Path(repo)
    parking_root = Path(parking_root)

    rc, out, err = _run_git(['worktree', 'list', '--porcelain'], cwd=repo)
    if rc != 0:
        logger.warning(
            '%s `git worktree list` failed in %s (rc=%s): %s — treating as no '
            'parkings',
            _LOG_PREFIX,
            repo,
            rc,
            err.strip(),
        )
        return []

    try:
        root_resolved = parking_root.resolve()
    except OSError:
        root_resolved = parking_root

    records: list[ParkedWorktree] = []
    for block in out.split('\n\n'):
        path: Path | None = None
        branch: str | None = None
        for line in block.splitlines():
            if line.startswith('worktree '):
                path = Path(line[len('worktree ') :]).resolve()
            elif line.startswith('branch '):
                ref = line[len('branch ') :].strip()
                head = 'refs/heads/'
                branch = ref[len(head) :] if ref.startswith(head) else ref
        if path is None or not _is_under(path, root_resolved):
            continue
        parked_at = parse_parking_dir_name(path.name)
        if parked_at is None:
            logger.warning(
                '%s skipping parking with unparseable basename %s (no trailing '
                '-<YYYYMMDDTHHMMSSZ> stamp) — never guess an age',
                _LOG_PREFIX,
                path,
            )
            continue
        records.append(ParkedWorktree(path=path, branch=branch, parked_at=parked_at))
    return records


def is_worktree_dirty(path: Path) -> bool:
    """Whether *path*'s working tree has uncommitted content (fail-safe DIRTY).

    ``git status --porcelain`` is non-empty => dirty. Any git failure (nonzero
    rc, or *path* is not a worktree / has vanished) is treated FAIL-SAFE as
    dirty (``True``) + LOUD warning — mirroring ``GitOps._worktree_dirty``'s
    fail-safe True, so a parking whose cleanliness cannot be PROVEN is never
    removed without a park-commit first. Never raises.
    """
    rc, out, err = _run_git(['status', '--porcelain'], cwd=path)
    if rc != 0:
        logger.warning(
            '%s `git status` failed in %s (rc=%s): %s — treating as DIRTY '
            '(fail-safe)',
            _LOG_PREFIX,
            path,
            rc,
            err.strip(),
        )
        return True
    return bool(out.strip())


def branch_ref_resolves(repo: Path, branch: str) -> bool:
    """Whether ``refs/heads/<branch>`` resolves in *repo* (content is on a ref).

    ``git rev-parse --verify --quiet refs/heads/<branch>`` — ``rc == 0`` means
    the branch exists (so the parking's content is safely reachable
    INDEPENDENTLY of the worktree). Any other rc (or git failure) => ``False``.
    Never raises.
    """
    rc, _out, _err = _run_git(
        ['rev-parse', '--verify', '--quiet', f'refs/heads/{branch}'], cwd=repo
    )
    return rc == 0


def park_commit(worktree: Path, reason: str) -> str | None:
    """Snapshot a dirty parking's content onto its branch (``git add -A`` +
    ``git commit --no-verify``).

    Returns ``None`` and does nothing when the worktree is already clean (a
    no-op). Otherwise stages everything (all tracked + untracked, non-ignored
    content) and commits it with ``--no-verify`` so a rejecting pre-commit hook
    can never strand the snapshot, returning the new HEAD sha. Any git failure
    is logged LOUDLY and returns ``None`` — never raises. ``--no-verify`` and
    ``add -A`` together make "dirty -> park-commit -> zero content lost" a
    provable property: the content is now on the branch ref, reachable
    independently of the worktree.
    """
    if not is_worktree_dirty(worktree):
        return None

    rc, _out, err = _run_git(['add', '-A'], cwd=worktree)
    if rc != 0:
        logger.warning(
            '%s `git add -A` failed in %s (rc=%s): %s — NOT park-committing',
            _LOG_PREFIX,
            worktree,
            rc,
            err.strip(),
        )
        return None

    rc, _out, err = _run_git(
        ['commit', '--no-verify', '-m', f'chore: park-commit reclaim ({reason})'],
        cwd=worktree,
    )
    if rc != 0:
        logger.warning(
            '%s `git commit` failed in %s (rc=%s): %s — NOT park-committing',
            _LOG_PREFIX,
            worktree,
            rc,
            err.strip(),
        )
        return None

    rc, out, err = _run_git(['rev-parse', 'HEAD'], cwd=worktree)
    if rc != 0:
        logger.warning(
            '%s `git rev-parse HEAD` failed in %s after park-commit (rc=%s): %s',
            _LOG_PREFIX,
            worktree,
            rc,
            err.strip(),
        )
        return None

    sha = out.strip()
    logger.info(
        '%s park-committed %s (reason=%s) -> %s', _LOG_PREFIX, worktree, reason, sha
    )
    return sha


def remove_worktree(repo: Path, path: Path) -> bool:
    """Remove a registered parking via ``git worktree remove --force <path>``.

    Returns ``True`` on success, ``False`` on any failure (logged LOUDLY); never
    raises. Passes NO branch-deleting flag — ``git worktree remove`` deletes only
    the working tree + the ``.git/worktrees`` admin entry and NEVER the branch
    ref, so a parking's content (already on its branch) survives the removal.
    The only destructive verb in this module, and it only ever runs on paths
    :func:`list_parked_worktrees` vetted as registered under the parking root.
    """
    rc, _out, err = _run_git(
        ['worktree', 'remove', '--force', str(path)], cwd=repo
    )
    if rc != 0:
        logger.warning(
            '%s `git worktree remove --force %s` failed (rc=%s): %s',
            _LOG_PREFIX,
            path,
            rc,
            err.strip(),
        )
        return False
    logger.info('%s removed worktree %s (branch intact)', _LOG_PREFIX, path)
    return True


@dataclass
class ReclaimOutcome:
    """Result of a :func:`reclaim_worktrees` sweep (paths, for the JSON report).

    ``reclaimed`` — parkings whose worktree was removed (branch left intact);
    ``park_committed`` — parkings whose dirty content was committed onto the
    branch before removal (a subset relationship with ``reclaimed`` in the happy
    path); ``skipped`` — parkings NOT removed by the data-loss guard
    (detached / unresolvable branch, or an un-park-committable dirty tree);
    ``failed`` — parkings whose removal (or processing) errored. Pure value
    object.
    """

    reclaimed: list[Path] = field(default_factory=list)
    park_committed: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    failed: list[Path] = field(default_factory=list)


def reclaim_worktrees(
    repo: Path,
    records: list[ParkedWorktree],
    *,
    dry_run: bool,
) -> ReclaimOutcome:
    """Reclaim each parking, in the fixed data-loss-guard order (best-effort).

    Per record: (1) require a RESOLVABLE branch (``branch is None`` — detached —
    or ``rev-parse`` fails => SKIP + LOUD, never removed); (2) in ``dry_run``,
    log ``would reclaim`` and stop; (3) if dirty, :func:`park_commit` onto the
    branch FIRST (a park-commit failure => SKIP, never removed) then re-verify
    the branch still resolves; (4) :func:`remove_worktree`, recording
    ``reclaimed`` on success or ``failed`` on failure. Content is never removed
    until it is provably on a branch ref INDEPENDENT of the worktree, so
    "zero content lost" holds.

    Each record is wrapped in ``try/except`` so one wedged parking is logged +
    counted (``failed``) while its siblings are still reclaimed — the sweep
    NEVER raises. Callers do the age selection upstream (:func:`select_reclaimable`);
    everything passed here is treated as eligible.
    """
    outcome = ReclaimOutcome()
    for record in records:
        path = record.path
        branch = record.branch
        try:
            # (1) Data-loss guard: content must be reachable on a branch ref.
            if branch is None or not branch_ref_resolves(repo, branch):
                logger.warning(
                    '%s SKIP %s — branch %r is detached/unresolvable; never '
                    'removing a parking whose content is not provably on a ref',
                    _LOG_PREFIX,
                    path,
                    branch,
                )
                outcome.skipped.append(path)
                continue

            # (2) Dry-run: classify only, mutate nothing.
            if dry_run:
                logger.info(
                    '%s would reclaim %s (branch=%s)', _LOG_PREFIX, path, branch
                )
                continue

            # (3) Dirty -> park-commit onto the branch FIRST, then re-verify.
            if is_worktree_dirty(path):
                if park_commit(path, 'reclaim') is None:
                    logger.warning(
                        '%s SKIP %s — dirty parking could not be park-committed; '
                        'NOT removing (data-loss guard)',
                        _LOG_PREFIX,
                        path,
                    )
                    outcome.skipped.append(path)
                    continue
                outcome.park_committed.append(path)
                if not branch_ref_resolves(repo, branch):
                    logger.warning(
                        '%s SKIP %s — branch %r stopped resolving after '
                        'park-commit; NOT removing (data-loss guard)',
                        _LOG_PREFIX,
                        path,
                        branch,
                    )
                    outcome.skipped.append(path)
                    continue

            # (4) Content is safe on the branch — remove the worktree.
            if remove_worktree(repo, path):
                outcome.reclaimed.append(path)
            else:
                outcome.failed.append(path)
        except Exception as err:  # noqa: BLE001 — never let one record abort the sweep
            logger.warning(
                '%s unexpected error reclaiming %s: %s — counting as failed',
                _LOG_PREFIX,
                path,
                err,
            )
            outcome.failed.append(path)
    return outcome


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI arg parser (unit-testable in isolation).

    ``--repo`` defaults to :data:`DEFAULT_PROJECT_ROOT`; ``--parking-root``
    defaults to ``None`` (derived as ``<repo>/.worktrees-orphaned`` in
    :func:`main`); ``--min-age-hours`` is the age floor (non-positive => reclaim
    nothing); ``--check`` is a dry-run flag; ``--now`` overrides the age
    reference clock (epoch seconds) for deterministic runs.
    """
    parser = argparse.ArgumentParser(
        prog='reclaim_orphaned_worktrees.py',
        description=(
            'Periodic verified reclaim of .worktrees-orphaned/ parkings '
            '(age-floored, safety-preserving, best-effort, LOUD).'
        ),
    )
    parser.add_argument(
        '--repo',
        default=DEFAULT_PROJECT_ROOT,
        help='Repository whose quarantine base is swept (default: %(default)s).',
    )
    parser.add_argument(
        '--parking-root',
        default=None,
        help=f'Quarantine base to sweep (default: <repo>/{PARKING_ROOT_NAME}).',
    )
    parser.add_argument(
        '--min-age-hours',
        type=float,
        default=DEFAULT_MIN_AGE_HOURS,
        help=(
            'Reclaim parkings STRICTLY older than this many hours; a '
            'NON-POSITIVE value reclaims NOTHING (fail-safe, protecting fresh '
            'parkings) (default: %(default)s).'
        ),
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help=(
            'Dry-run: scan + classify + log would-reclaim lines, remove/commit '
            'nothing, exit 0.'
        ),
    )
    parser.add_argument(
        '--now',
        type=float,
        default=time.time(),
        help=(
            'Age reference clock in epoch seconds (default: the current time). '
            'Pin it for deterministic age-based reclaim in tests.'
        ),
    )
    return parser


def build_report(
    root: Path,
    now: float,
    check: bool,
    min_age_hours: float,
    scanned: list[ParkedWorktree],
    decision: ReclaimDecision,
    outcome: ReclaimOutcome,
) -> dict:
    """Assemble the machine-readable JSON report (pure — no I/O).

    ``reclaimed`` is the count actually removed (0 in a dry-run); ``kept`` is the
    count classified as too-young by the age floor. ``skipped`` / ``failed``
    carry the best-effort data-loss-guard skips and per-worktree failures — the
    machine-readable signal (INV-4) a cron/watchdog wrapper reads without
    alarming on the always-0 exit code.
    """
    return {
        'root': str(root),
        'now': now,
        'check': check,
        'min_age_hours': min_age_hours,
        'scanned': len(scanned),
        'kept': len(decision.keep),
        'reclaimed': len(outcome.reclaimed),
        'reclaimed_paths': [str(p) for p in outcome.reclaimed],
        'park_committed': len(outcome.park_committed),
        'park_committed_paths': [str(p) for p in outcome.park_committed],
        'skipped': len(outcome.skipped),
        'skipped_paths': [str(p) for p in outcome.skipped],
        'failed': len(outcome.failed),
        'failed_paths': [str(p) for p in outcome.failed],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the sweep: scan → classify → reclaim (or dry-run) → prune, report.

    Best-effort and never-raising by design — the JSON report on stdout and the
    LOUD summary on stderr carry the signal (including any skips/failures); the
    process always exits 0 so a nightly timer does not alarm on a routine
    per-worktree hiccup. ``--check`` never removes/commits. A ``git worktree
    prune`` runs after real (non-check) removals to clear any stale admin
    entries.
    """
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)

    repo = Path(args.repo)
    parking_root = (
        Path(args.parking_root)
        if args.parking_root is not None
        else repo / PARKING_ROOT_NAME
    )

    if args.min_age_hours <= 0:
        logger.warning(
            '%s --min-age-hours=%s is non-positive; reclaiming NOTHING '
            '(fail-safe: the age floor is the only protection for fresh '
            'parkings)',
            _LOG_PREFIX,
            args.min_age_hours,
        )

    scanned = list_parked_worktrees(repo, parking_root)
    decision = select_reclaimable(scanned, args.now, args.min_age_hours)
    reclaim_records = [record for record, _reason in decision.reclaim]
    outcome = reclaim_worktrees(repo, reclaim_records, dry_run=args.check)

    if not args.check:
        rc, _out, err = _run_git(['worktree', 'prune'], cwd=repo)
        if rc != 0:
            logger.warning(
                '%s `git worktree prune` failed in %s (rc=%s): %s',
                _LOG_PREFIX,
                repo,
                rc,
                err.strip(),
            )

    report = build_report(
        root=parking_root,
        now=args.now,
        check=args.check,
        min_age_hours=args.min_age_hours,
        scanned=scanned,
        decision=decision,
        outcome=outcome,
    )
    print(json.dumps(report))

    logger.info(
        '%s sweep complete — root=%s scanned=%d kept=%d reclaimed=%d '
        'park_committed=%d skipped=%d failed=%d check=%s',
        _LOG_PREFIX,
        parking_root,
        report['scanned'],
        report['kept'],
        report['reclaimed'],
        report['park_committed'],
        report['skipped'],
        report['failed'],
        args.check,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
