#!/usr/bin/env python3
"""Retention GC sweep over the gzipped agent-transcript archive.

Task 2731 (δ of plans/agent-transcript-archival-prd.md). The α producer hook
(:mod:`shared.transcript_archive`, task 2742) gzips each finished agent
session's transcripts to a durable archive root outside the per-task worktree,
laid out as ``<root>/<task_id>/<enc>/<session_id>.jsonl.gz`` with each ``.gz``
carrying the SOURCE transcript's mtime (mirrored via ``os.utime``). Nothing
prunes that tree, so it grows without bound; this sweep is the δ consumer that
enforces α's ``retention.*`` config knobs (``max_age_days`` / ``max_task_dirs``)
to keep archive disk bounded.

Retention model
---------------
Each immediate child dir of ``<root>`` is one per-task archive dir. Its
retention "age" is the newest ``.gz`` mtime among its descendants (recursive),
falling back to the dir's own mtime when it holds no files — because α mirrors
the source transcript mtime onto each ``.gz``, that newest mtime is the real
time the task's most recent agent session last wrote, and is stable across
re-archival.

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

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger('gc_agent_transcripts')

# Stable greppable prefix on every operator-facing log line.
_LOG_PREFIX = 'gc_agent_transcripts:'

SECONDS_PER_DAY = 86_400


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

    AGE arm only (this step): a dir is age-prunable iff ``max_age_days > 0``
    and its ``mtime`` is strictly older than ``now - max_age_days * 86400`` —
    so a dir sitting exactly on the cutoff (``now - mtime == max_age_days*86400``)
    is KEPT, and one second older is pruned. A non-positive ``max_age_days``
    disables the age axis entirely.

    ``max_task_dirs`` is accepted but treated as unlimited for now (the count
    arm lands in a later step).
    """
    del max_task_dirs  # count arm not yet implemented; accepted, treated as unlimited

    age_cutoff: float | None = None
    if max_age_days > 0:
        age_cutoff = now - max_age_days * SECONDS_PER_DAY

    keep: list[Path] = []
    prune: list[tuple[Path, str]] = []
    for path, mtime in task_dirs:
        if age_cutoff is not None and mtime < age_cutoff:
            prune.append((path, 'age'))
        else:
            keep.append(path)

    return GcDecision(keep=keep, prune=prune)
