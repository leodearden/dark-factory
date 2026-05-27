#!/usr/bin/env python3
"""One-shot backfill: collapse/dismiss the pending recon escalation pile.

Motivation: Task A7a introduced content-fingerprint deduplication for
``recon_integrity_issue`` escalations (dedupe.py / submit_or_dedupe).  The
~5,858 ``recon_integrity_issue`` records that accumulated before A7a/A7b went
live each represent a recurring finding that was filed independently every
reconciliation cycle.  This script collapses each fingerprint group into a
single canonical record (oldest survives, ``dedupe_count`` = group size,
``dedupe_fingerprint`` stamped), archives/dismisses the rest, and leaves every
blocking and non-recon category escalation completely untouched.

The collapse policy is byte-identical to the live A7a/A7b submit-time
deduplication: eligible categories are sourced from
``DedupeConfig.for_recon().infra_dedupe_categories`` (= ``('recon_integrity_issue',)``),
and the fingerprint is computed via ``compute_content_fingerprint`` with the
same inputs that the harness passes at submit time.

Usage
-----
  # Dry run (default): print JSON report, touch nothing.
  python scripts/backfill_recon_escalations.py

  # Commit the collapses.
  python scripts/backfill_recon_escalations.py --apply

  # Override the queue directory (default: ./data/reconciliation/escalations).
  python scripts/backfill_recon_escalations.py --queue-dir /path/to/queue --apply

  # Override resolved-by tag and resolution note.
  python scripts/backfill_recon_escalations.py --apply \\
      --resolved-by backfill-A7c \\
      --note "Collapsed by A7c backfill: duplicate recon_integrity_issue finding."

Safety properties:
- Dry-run is the default — no writes occur unless ``--apply`` is passed.
- Idempotent: a second ``--apply`` run is a no-op because every fingerprint
  maps to a single pending canonical after the first run (singletons are never
  collapsed).
- Only ever calls EscalationQueue methods (get_pending, get, submit, resolve);
  never enumerates, moves, or deletes raw files directly.
- Blocking escalations (infra_issue, recon_failure, etc.) are never touched.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from escalation.dedupe import DedupeConfig, compute_content_fingerprint
from escalation.models import Escalation
from escalation.queue import EscalationQueue

logger = logging.getLogger(__name__)

RESOLVED_BY: str = 'backfill-A7c'
DEFAULT_NOTE: str = (
    'Collapsed by A7c backfill: duplicate recon_integrity_issue finding. '
    'Canonical record retains full dedupe_count and dedupe_fingerprint.'
)


def finding_fingerprint(esc: Escalation) -> str:
    """Return a content fingerprint for *esc* by parsing its detail JSON.

    Reproduces the fingerprint that A7b's ``submit_or_dedupe`` will stamp at
    submit time, making backfilled canonicals forward-consistent.

    Falls back to a description-less fingerprint (``compute_content_fingerprint``
    with empty affected_ids and the escalation summary) when ``esc.detail`` is
    missing or not a valid JSON object containing the expected finding fields.
    """
    raise NotImplementedError


@dataclass
class GroupCollapse:
    """One fingerprint group that will be collapsed."""

    fingerprint: str
    canonical_id: str
    child_ids: list[str]
    group_size: int
    category: str


@dataclass
class BackfillPlan:
    """The complete plan for the backfill run."""

    collapses: list[GroupCollapse]
    pending_before: int
    eligible_total: int
    distinct_fingerprints: int
    groups_collapsed: int
    to_dismiss: int
    expected_survivors: int


def build_plan(
    pending: list[Escalation],
    eligible_categories: set[str] | None = None,
) -> BackfillPlan:
    """Analyse *pending* escalations and return a collapse plan.

    Only escalations whose ``category`` is in *eligible_categories* are
    considered for collapse.  Singleton fingerprint groups are skipped (the
    idempotency guard: after an apply every group is a singleton, so re-runs
    are no-ops).
    """
    raise NotImplementedError


def apply_plan(
    queue: EscalationQueue,
    plan: BackfillPlan,
    *,
    resolved_by: str = RESOLVED_BY,
    note: str = DEFAULT_NOTE,
) -> dict:
    """Execute *plan* against *queue*.

    For each GroupCollapse:
    - Stamps ``dedupe_count``, ``dedupe_children``, and ``dedupe_fingerprint``
      on the canonical and persists it via ``queue.submit()``.
    - Dismisses each child via ``queue.resolve(dismiss=True)``.

    Returns a dict with ``dismissed`` and ``updated`` counts.
    """
    raise NotImplementedError


def run(
    queue_dir: str | Path,
    *,
    apply: bool = False,
    resolved_by: str = RESOLVED_BY,
    note: str = DEFAULT_NOTE,
) -> dict:
    """Build a collapse plan for *queue_dir* and optionally execute it.

    Returns a report dict.  When ``apply`` is False (dry-run, the default),
    no writes are performed.
    """
    raise NotImplementedError


def main() -> int:
    """CLI entry point.  Returns an exit code."""
    raise NotImplementedError


if __name__ == '__main__':
    sys.exit(main())
