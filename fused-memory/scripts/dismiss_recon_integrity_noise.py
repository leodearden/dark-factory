#!/usr/bin/env python3
"""One-shot cleanup: dismiss the pending recon_integrity_issue noise pile.

Context (2026-05-27, "Direction 3" — see plans/afk-A7-recon-closure.md):
The A7a/A7b content-fingerprint dedup is ineffective against the real recon
queue because the records have no stable identity to fingerprint on — the LLM
emits a different ``finding.category`` (task_memory_mismatch / other /
systemic_pattern / ...) and a polluted ``affected_ids`` set (volatile memory
UUIDs, co-flagged tasks, inconsistent ``452`` / ``task-452`` / ``task_452``
encodings) every cycle.  A dry-run of the A7c backfill collapsed only 6,048 ->
5,703; even a normalized task-id key reached only ~3,670 survivors, with 1,460
findings carrying no task-id at all.

These ``recon_integrity_issue`` records are info-severity, **non-actionable**
("Non-actionable integrity finding: ...") or already-attempted-and-failed
("Unresolved after remediation: ...") residue.  Nothing consumes port 8103, so
they have never been human-facing in any actionable sense.  This script
dismisses them so the queue becomes a small, watchable signal of the genuinely
operational escalations (infra_issue, recon_failure, recon_backlog_overflow,
recon_stale_run, ...), which are LEFT UNTOUCHED.

The go-forward fix (stop recon escalating non-actionable info findings into the
queue at all) is tracked as a separate task; this script only clears the
existing pile.

Usage
-----
  # Dry run (default): print JSON report, touch nothing.
  python scripts/dismiss_recon_integrity_noise.py

  # Commit the dismissals.
  python scripts/dismiss_recon_integrity_noise.py --apply

  # Override the queue directory (default: ./data/reconciliation/escalations).
  python scripts/dismiss_recon_integrity_noise.py --queue-dir /path/to/queue --apply

Safety properties:
- Dry-run is the default — no writes occur unless ``--apply`` is passed.
- Dismissals only ever target category == 'recon_integrity_issue'.  Every other
  category (operational / blocking) is left completely untouched.
- Uses only EscalationQueue.resolve(dismiss=True), which ARCHIVES each record
  (moves it under archive/<date>/ — nothing is deleted) and is idempotent, so a
  second --apply run is a no-op over already-dismissed records.
- A fresh EscalationQueue has no resolve-callback, and the live 8103 server
  re-globs the directory on every get_pending(), so dismissals stay consistent
  with the running service without restarting it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from escalation.queue import EscalationQueue

logger = logging.getLogger(__name__)

TARGET_CATEGORY = 'recon_integrity_issue'
RESOLVED_BY = 'dismiss-recon-noise-2026-05-27'
DEFAULT_NOTE = (
    'Bulk-dismissed (Direction 3, 2026-05-27): non-actionable recon_integrity_issue '
    'noise with no consumer and no stable identity for dedup. Archived for '
    'post-mortem. Go-forward fix (stop escalating non-actionable info findings) '
    'tracked separately. See plans/afk-A7-recon-closure.md.'
)


def run(
    queue_dir: str | Path,
    *,
    apply: bool = False,
    resolved_by: str = RESOLVED_BY,
    note: str = DEFAULT_NOTE,
) -> dict:
    """Dismiss every pending recon_integrity_issue in *queue_dir*.

    When ``apply`` is False (the default) no writes occur; the report shows what
    would be dismissed and what would be kept.
    """
    queue = EscalationQueue(Path(queue_dir))
    pending = queue.get_pending()

    targets = [e for e in pending if e.category == TARGET_CATEGORY]
    kept = [e for e in pending if e.category != TARGET_CATEGORY]

    report: dict = {
        'queue_dir': str(queue_dir),
        'pending_before': len(pending),
        'to_dismiss': len(targets),
        'kept': len(kept),
        'kept_by_category': dict(Counter(e.category for e in kept).most_common()),
        'kept_by_severity': dict(Counter(e.severity for e in kept).most_common()),
        'dry_run': not apply,
    }

    if apply:
        dismissed = 0
        for esc in targets:
            result = queue.resolve(esc.id, note, dismiss=True, resolved_by=resolved_by)
            if result is None:
                logger.warning('Escalation %s vanished before resolve; skipping', esc.id)
            else:
                dismissed += 1
        report['dismissed'] = dismissed
        report['pending_after'] = len(queue.get_pending())

    return report


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s %(message)s')

    parser = argparse.ArgumentParser(
        description='Dismiss the pending recon_integrity_issue noise pile (info findings).',
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
    parser.add_argument('--resolved-by', default=RESOLVED_BY)
    parser.add_argument('--note', default=DEFAULT_NOTE)
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
