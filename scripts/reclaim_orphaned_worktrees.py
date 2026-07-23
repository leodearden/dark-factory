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
import re
from datetime import UTC, datetime

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
