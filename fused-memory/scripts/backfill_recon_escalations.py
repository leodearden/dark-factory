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
from collections import defaultdict
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
    try:
        finding = json.loads(esc.detail)
        if not isinstance(finding, dict):
            raise TypeError('detail is not a JSON object')
        finding_category = finding.get('category', '')
        affected_ids = finding.get('affected_ids', [])
        description = finding.get('description', '')
        return compute_content_fingerprint(
            esc.category, finding_category, affected_ids, description,
        )
    except (json.JSONDecodeError, TypeError):
        return compute_content_fingerprint(esc.category, '', [], esc.summary or '')


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
    if eligible_categories is None:
        eligible_categories = set(DedupeConfig.for_recon().infra_dedupe_categories)

    pending_before = len(pending)

    eligible = [e for e in pending if e.category in eligible_categories]
    eligible_total = len(eligible)

    # Group by fingerprint, sorting each group oldest-first.
    groups: dict[str, list[Escalation]] = defaultdict(list)
    for esc in eligible:
        fp = finding_fingerprint(esc)
        groups[fp].append(esc)

    distinct_fingerprints = len(groups)

    collapses: list[GroupCollapse] = []
    for fp, members in groups.items():
        if len(members) <= 1:
            continue  # singleton — idempotency guard, nothing to collapse
        # Sort by (timestamp, id) so the oldest is first and ties are deterministic.
        members.sort(key=lambda e: (e.timestamp, e.id))
        canonical = members[0]
        children = members[1:]
        collapses.append(GroupCollapse(
            fingerprint=fp,
            canonical_id=canonical.id,
            child_ids=[c.id for c in children],
            group_size=len(members),
            category=canonical.category,
        ))

    groups_collapsed = len(collapses)
    to_dismiss = sum(len(c.child_ids) for c in collapses)
    expected_survivors = pending_before - to_dismiss

    return BackfillPlan(
        collapses=collapses,
        pending_before=pending_before,
        eligible_total=eligible_total,
        distinct_fingerprints=distinct_fingerprints,
        groups_collapsed=groups_collapsed,
        to_dismiss=to_dismiss,
        expected_survivors=expected_survivors,
    )


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
    dismissed = 0
    updated = 0

    for collapse in plan.collapses:
        canonical = queue.get(collapse.canonical_id)
        if canonical is None:
            logger.warning('Canonical %s not found; skipping group', collapse.canonical_id)
            continue

        canonical.dedupe_count = collapse.group_size
        canonical.dedupe_children = list(collapse.child_ids)
        canonical.dedupe_fingerprint = collapse.fingerprint
        queue.submit(canonical)
        updated += 1

        for child_id in collapse.child_ids:
            queue.resolve(child_id, note, dismiss=True, resolved_by=resolved_by)
            dismissed += 1

    return {'dismissed': dismissed, 'updated': updated}


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
    queue = EscalationQueue(Path(queue_dir))
    pending = queue.get_pending()
    plan = build_plan(pending)

    eligible_cats = set(DedupeConfig.for_recon().infra_dedupe_categories)
    blocking_pending = sum(
        1 for e in pending if e.category not in eligible_cats
    )

    report: dict = {
        'queue_dir': str(queue_dir),
        'pending_before': plan.pending_before,
        'eligible_total': plan.eligible_total,
        'distinct_fingerprints': plan.distinct_fingerprints,
        'groups_collapsed': plan.groups_collapsed,
        'to_dismiss': plan.to_dismiss,
        'expected_survivors': plan.expected_survivors,
        'blocking_pending': blocking_pending,
        'dry_run': not apply,
    }

    if apply:
        result = apply_plan(queue, plan, resolved_by=resolved_by, note=note)
        pending_after = len(queue.get_pending())
        report['dismissed'] = result['dismissed']
        report['updated'] = result['updated']
        report['pending_after'] = pending_after

    return report


def main() -> int:
    """CLI entry point.  Returns an exit code."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

    parser = argparse.ArgumentParser(
        description='Backfill: collapse duplicate recon_integrity_issue escalations.',
    )
    parser.add_argument(
        '--queue-dir',
        default='./data/reconciliation/escalations',
        help='Path to the escalation queue directory (default: ./data/reconciliation/escalations).',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        default=False,
        help='Perform writes. Without this flag the script is a dry run.',
    )
    parser.add_argument(
        '--resolved-by',
        default=RESOLVED_BY,
        help=f'resolved_by tag written to dismissed records (default: {RESOLVED_BY}).',
    )
    parser.add_argument(
        '--note',
        default=DEFAULT_NOTE,
        help='Resolution note written to dismissed records.',
    )
    args = parser.parse_args()

    report = run(
        args.queue_dir,
        apply=args.apply,
        resolved_by=args.resolved_by,
        note=args.note,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
