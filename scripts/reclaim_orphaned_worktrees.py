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

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger('reclaim_orphaned_worktrees')

# Stable greppable prefix on every operator-facing log line.
_LOG_PREFIX = 'reclaim_orphaned_worktrees:'

# The hardcoded absolute project-root default (drain_check.py / gc_agent_
# transcripts.py set the precedent). Overridable via --repo.
DEFAULT_PROJECT_ROOT = '/home/leo/src/dark-factory'

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
    Forces ``LC_ALL=C`` so porcelain output is locale-stable. Best-effort +
    never-raise: an ``OSError`` (git binary missing, *cwd* vanished, etc.) is
    mapped to a non-zero ``(1, '', <message>)`` rather than propagated, so
    callers can treat a failed git call fail-safe without a ``try/except``.
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
    except OSError as err:
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
