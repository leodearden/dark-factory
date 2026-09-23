#!/usr/bin/env python3
"""One-shot backfill: collapse/dismiss the pending recon escalation pile.

Motivation: Task A7a introduced content-fingerprint deduplication for
``recon_integrity_issue`` escalations (dedupe.py / submit_or_dedupe).  The
~5,858 ``recon_integrity_issue`` records that accumulated before A7a/A7b went
live each represent a recurring finding that was filed independently every
reconciliation cycle.  This script collapses each fingerprint group into a
single canonical record (oldest survives, ``dedupe_count`` = number of folded
children (= group_size − 1), matching the ``attach_dedupe_child`` semantics and
the ``dedupe_count == len(dedupe_children)`` invariant; ``dedupe_fingerprint``
stamped), archives/dismisses the rest, and leaves every
blocking and non-recon category escalation completely untouched.

The collapse policy is consistent with the A7a/A7b dedup approach: eligible
categories are sourced from ``DedupeConfig.for_recon().infra_dedupe_categories``
(= ``('recon_integrity_issue',)``), and each escalation's fingerprint is computed
from the same inputs (``escalation_category``, ``finding_category``,
``affected_ids``, ``description``) that A7b will stamp at submit time once the
harness migrates from ``_escalate()`` / ``queue.submit()`` to
``submit_or_dedupe()``.

Note: the harness currently bypasses dedup entirely — ``_escalate()`` calls
``queue.submit()`` directly and does NOT invoke ``submit_or_dedupe``, so no
``dedupe_fingerprint`` has been stamped on any existing record.  Every
``recon_integrity_issue`` that has accumulated to date is therefore unstamped.
This script's fingerprint is a forward-looking commitment to the shape A7b will
stamp once the harness is migrated — backfilled canonicals carry
``dedupe_fingerprint`` so that future ``submit_or_dedupe`` folds route correctly.

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
- The queue directory must ALREADY EXIST; ``run()`` refuses otherwise.  See the
  section below.

WHY THIS SCRIPT PREFLIGHTS ITS TARGET (a decision, task 4319)
-------------------------------------------------------------
:func:`run` refuses, before the scan, unless ``--queue-dir`` names a directory
that ALREADY exists.  The default is the RELATIVE
``./data/reconciliation/escalations``, so a run from anywhere but the project
root -- a task worktree in particular -- manufactures an empty queue and reports
``"pending_before": 0``, a false all-clear that ``main()`` below would hand back
as exit 0.

See ``fused_memory/utils/target_store_preflight.py::assert_queue_dir_exists``
for the mechanism, the probe-vs-existence argument, the prior art and the
placement rules -- that module is the single normative copy, and this note
deliberately does not restate it.

A RESIDUAL THIS TASK RECORDED AND DELIBERATELY DID NOT GUARD (task 4319)
-------------------------------------------------------------------------
:func:`apply_plan` is not transactional.  Per group it stamps the canonical via
``queue.submit()`` and THEN dismisses N children in a loop, with no rollback
between the two.  Its own idempotency guard skips any canonical that already
carries dedupe state, so a failure landing between the submit and the last
``queue.resolve`` would strand the remaining children PERMANENTLY: the re-run
sees the stamped canonical and skips the whole group.

That is recorded rather than fixed here because under a uniform write-deny it
is unreachable.  The first write-requiring syscall in every mutating queue path
is the lockfile ``os.open(..., O_CREAT | O_RDWR)`` in
``escalation/queue.py::escalation_id_lock``, taken outside any handler, so a
denial aborts on record #1 before anything is written.  The premise that would
make it reachable, and that a later reader should re-check: a PARTIAL policy
that grants the queue root but denies a subtree — e.g. ``archive/``, where
``escalation/queue.py::EscalationQueue._archive_resolved`` swallows ``OSError``
into a ``logger.warning`` by deliberate no-data-loss choice.  Landlock does not
produce that shape (its rules are path-prefix based), which is why it is a
premise and not an observation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from escalation.dedupe import DedupeConfig, compute_content_fingerprint
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from fused_memory.utils.target_store_preflight import assert_queue_dir_exists

logger = logging.getLogger(__name__)

RESOLVED_BY: str = 'backfill-A7c'
DEFAULT_NOTE: str = (
    'Collapsed by A7c backfill: duplicate recon_integrity_issue finding. '
    'Canonical record retains full dedupe_count and dedupe_fingerprint.'
)

# Sentinel: timestamps that cannot be parsed sort to the END of the group so
# they are never mistakenly selected as the canonical (oldest).
_MAX_DT: datetime = datetime.max.replace(tzinfo=UTC)


def _parse_ts(ts: str | None) -> datetime:
    """Parse an ISO 8601 timestamp string into an aware datetime.

    Normalises the ``Z`` UTC suffix to ``+00:00`` for Python < 3.11
    compatibility.  Naive timestamps (no timezone info) are assumed UTC.
    Returns ``_MAX_DT`` on any parse failure so unparseable records sort to
    the END of their fingerprint group and are never selected as canonical.

    Mirrors the ``datetime.fromisoformat`` + tzinfo-fill pattern used by
    ``find_dedupe_parent`` in escalation/dedupe.py.
    """
    if not ts:
        return _MAX_DT
    try:
        normalised = ts.replace('Z', '+00:00') if ts.endswith('Z') else ts
        dt = datetime.fromisoformat(normalised)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except (ValueError, TypeError):
        return _MAX_DT


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
        # Sort by parsed timestamp so mixed ISO formats (Z vs +00:00 vs
        # microsecond precision) all compare correctly.  _parse_ts falls back
        # to _MAX_DT on failure so unparseable records sort to the end.
        members.sort(key=lambda e: (_parse_ts(e.timestamp), e.id))
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

    Returns a dict with ``dismissed`` and ``updated`` counts plus two
    state-drift error counters: ``canonical_not_found`` (a canonical gone
    before it could be stamped) and ``children_vanished`` (a child gone before
    it could be dismissed, so its stamped canonical overstates the group).  A
    canonical that already carries dedupe state is skipped uncounted: that is
    A7b's idempotency guard, not a failure.
    """
    dismissed = 0
    updated = 0
    canonical_not_found = 0
    children_vanished = 0

    for collapse in plan.collapses:
        canonical = queue.get(collapse.canonical_id)
        if canonical is None:
            logger.warning('Canonical %s not found; skipping group', collapse.canonical_id)
            canonical_not_found += 1
            continue

        # Guard: if the canonical already carries dedupe state, A7b may already
        # be active for this fingerprint.  Skip rather than overwriting — the
        # backfill has no business meddling with A7b's in-progress work.
        if canonical.dedupe_count > 0 or canonical.dedupe_children:
            logger.warning(
                'Canonical %s already has dedupe state (count=%d, children=%d); '
                'skipping group — A7b may already be active for this fingerprint',
                collapse.canonical_id,
                canonical.dedupe_count,
                len(canonical.dedupe_children),
            )
            continue

        canonical.dedupe_count = len(collapse.child_ids)
        canonical.dedupe_children = list(collapse.child_ids)
        canonical.dedupe_fingerprint = collapse.fingerprint
        queue.submit(canonical)
        updated += 1

        for child_id in collapse.child_ids:
            result = queue.resolve(child_id, note, dismiss=True, resolved_by=resolved_by)
            if result is None:
                logger.warning(
                    'Child %s not found during resolve (state drift between '
                    'get_pending and apply); skipping',
                    child_id,
                )
                children_vanished += 1
            else:
                dismissed += 1

    return {
        'dismissed': dismissed,
        'updated': updated,
        'canonical_not_found': canonical_not_found,
        'children_vanished': children_vanished,
    }


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

    Refuses with ``TargetStoreMissing`` when *queue_dir* does not exist — see
    the module docstring.  The check lives here rather than in ``main()`` so
    programmatic callers inherit it too.
    """
    assert_queue_dir_exists(queue_dir, operation='backfill_recon_escalations')

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
        report['canonical_not_found'] = result['canonical_not_found']
        report['children_vanished'] = result['children_vanished']
        report['pending_after'] = pending_after

    return report


def resolve_exit_code(report: dict) -> int:
    """0 on a clean run, 1 when ``canonical_not_found`` or ``children_vanished``
    is non-zero.

    A missing key counts as 0; a dry-run report carries neither.
    """
    errors = report.get('canonical_not_found', 0) + report.get('children_vanished', 0)
    return 1 if errors > 0 else 0


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
    return resolve_exit_code(report)


if __name__ == '__main__':
    sys.exit(main())
