#!/usr/bin/env python3
"""Derive — and optionally close — orphaned recon stale escalations (task 3052).

MOTIVATION.  ``stage1_stall_detector.py::maybe_escalate_stalled_gate_backlog``
files a ``reconciliation_stale_gate_backlog`` L1 only while its subject task
is ``status == 'blocked'``.  Once the subject goes terminal (done/cancelled)
or vanishes from its project's task store the record is moot, can never
re-file, and nothing closes it — the reconciliation harness never resolves its
own escalation queue (the A7b contract above
``reconciliation/harness.py::_RECON_DEDUP_CONFIG``), and the orchestrator's
revalidation sweep reads a different queue and returns early on
``level != 2`` while every recon record is born at L1.

This is a derivation RULE, deliberately NOT a roster of ids: the queue
accumulates continuously.  Measured pending
``reconciliation_stale_gate_backlog`` count — 108 on 2026-08-30, 117 at
planning, **124 on 2026-09-02**.  Always re-derive; never drain a stale list.

The classification rule has exactly ONE owner, shared with the in-cycle
Stage-1 sweep: ``fused_memory.reconciliation.orphaned_recon_escalation_sweep``.
A second copy could drift and make the flag and the reap disagree about which
records are safe to close.

Closing here does not contradict A7b.  That invariant bans the HARNESS from
resolving its own queue; an operator-run one-shot over the same queue is
precedented by ``fused-memory/scripts/backfill_recon_escalations.py``, whose
dry-run-default / ``--apply`` / ``--queue-dir`` contract this script copies.
It is also the branch the watcher playbook already sanctions: "**Resolve
only** when the underlying task will genuinely stop qualifying for
re-selection" — and re-selection requires ``status == 'blocked'``, so a
terminal or absent subject provably cannot cause a re-file.

Usage
-----
  # Dry run (the default): print the JSON report, touch nothing.
  python fused-memory/scripts/derive_orphaned_recon_escalations.py

  # Close the derived set.
  python fused-memory/scripts/derive_orphaned_recon_escalations.py --apply

  # Override the queue dir and/or supply extra project roots.
  python fused-memory/scripts/derive_orphaned_recon_escalations.py \\
      --queue-dir /path/to/data/reconciliation/escalations \\
      --project-root /home/leo/src/reify --apply

WARNING — ``--apply`` MUTATES THE LIVE QUEUE.  It resolves real pending
escalation records at ``<project_root>/data/reconciliation/escalations``, which
is production operational state.  Running it is the OPERATOR's or the port-8103
watcher session's action, taken under
``skills/recon-escalation-watcher/SKILL.md``'s amended playbook row — not an
implementing agent's.

Safety properties:
- Dry run is the default; no write happens without ``--apply``.
- Idempotent: the reap set is re-derived from ``get_pending()`` each run, so
  an already-closed record is simply no longer a candidate.
- A record whose ``project_id`` cannot be parsed, or whose project is not in
  the roots map, is counted ``unresolvable`` and NEVER closed — its subject
  was never checked, so closing it would be a reap on no evidence.
- Each project's census is CROSS-TAG-COMPLETE (``list_tags`` then one
  ``get_statuses_fresh`` per tag), because ``get_statuses_fresh`` defaults to
  a single tag and a single-tag read would report a subject living elsewhere
  as absent — closing a possibly-still-``blocked`` record.
- A census read failure is tallied into ``errors`` and classifies NOTHING for
  that project.
- Only ``EscalationQueue`` methods are used (``get_pending``/``resolve``);
  raw queue files are never enumerated, moved or deleted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from escalation.queue import EscalationQueue

from fused_memory.models.scope import build_known_projects_map
from fused_memory.reconciliation.orphaned_recon_escalation_sweep import (
    _project_status_census,
    classify_orphan,
    escalation_project_id,
    select_reapable_escalations,
)

logger = logging.getLogger(__name__)

RESOLVED_BY: str = 'orphaned-recon-escalation-reaper'

# The purpose-built ``escalation.models.RESOLUTION_CLASSES`` member for exactly
# this condition (task 2724): deliberately neither 'benign' nor 'actionable',
# so a reaped record stays auditable rather than being mis-labelled.
# ``EscalationQueue.resolve`` validates it BEFORE taking the lock and raises
# ``ValueError`` on an unknown value, so a typo here fails loudly with nothing
# persisted.
RESOLUTION_CLASS: str = 'moot-terminal-subject'

DEFAULT_QUEUE_DIR: str = './data/reconciliation/escalations'

PROJECT_ROOT: str = str(Path(__file__).resolve().parents[2])


def _resolution_note(esc, classification, project_id, subject_status) -> str:
    """Free-text rationale stamped on the closed record.

    Names the evidence so a later auditor can re-derive the decision from the
    record alone rather than trusting this script's assertion.
    """
    if classification == 'terminal':
        observed = (
            f'subject task {esc.task_id} is {subject_status} (terminal) in '
            f"{project_id}'s task store"
        )
    else:
        observed = (
            f'subject task {esc.task_id} has no row in {project_id}\'s task '
            'store (checked across every tag)'
        )
    return (
        f'orphaned-recon-escalation reaper: {observed}. This '
        f'{esc.category} record is filed only while its subject is '
        "status == 'blocked' "
        '(stage1_stall_detector.py::extract_stalled_gate_backlog_task_ids), '
        'so its premise no longer holds and it cannot be re-filed — closing '
        'it cannot re-arm the filing rule.'
    )


async def run(
    queue_dir,
    project_roots,
    *,
    apply: bool = False,
    resolved_by: str = RESOLVED_BY,
    taskmaster = None,
) -> dict:
    """Derive the orphan set over *queue_dir*, and close it when *apply*.

    Args:
        queue_dir: The RECON escalation queue directory.
        project_roots: ``{project_id: project_root}``.  A record naming a
            project absent from this map is ``unresolvable``, never reaped.
        apply: ``False`` (the default) is a pure dry run — nothing is written.
        resolved_by: The ``resolved_by`` tag stamped on closed records.
        taskmaster: A ``TaskBackendProtocol``.  When ``None`` a
            ``SqliteTaskBackend`` is built from the fused-memory config and
            started/closed around the derivation.

    Returns:
        A JSON-serialisable report.  EVERY count key is present in BOTH modes
        — deliberately unlike ``backfill_recon_escalations.py``, which adds
        its apply-only keys conditionally and so forces consumers into
        ``.get()``.  ``reaped`` is simply 0 on a dry run.
    """
    owns_backend = taskmaster is None
    if owns_backend:
        from fused_memory.backends.sqlite_task_backend import (  # noqa: PLC0415
            SqliteTaskBackend,
        )
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415

        config = FusedMemoryConfig()
        if config.taskmaster is None:
            raise RuntimeError('Task backend not configured in fused-memory config')
        taskmaster = SqliteTaskBackend(config.taskmaster)
        await taskmaster.start()

    try:
        return await _derive(
            queue_dir, project_roots, taskmaster,
            apply=apply, resolved_by=resolved_by,
        )
    finally:
        if owns_backend:
            await taskmaster.close()


async def _derive(queue_dir, project_roots, taskmaster, *, apply, resolved_by) -> dict:
    """The derivation proper — see :func:`run` for the contract."""
    queue = EscalationQueue(Path(queue_dir))
    # Queue ROOT only: an archived record is already closed and is correctly
    # invisible here, which is what makes a repeat --apply a clean no-op.
    reapable = select_reapable_escalations(queue.get_pending())

    report: dict = {
        'dry_run': not apply,
        'queue_dir': str(queue_dir),
        'scanned': len(reapable),
        'terminal': 0,
        'missing': 0,
        'live': 0,
        'unresolvable': 0,
        'errors': 0,
        'reaped': 0,
        'reapable_ids': [],
    }

    censuses: dict[str, dict[str, str] | None] = {}

    for esc in reapable:
        project_id = escalation_project_id(esc)
        if project_id is None or project_id not in project_roots:
            report['unresolvable'] += 1
            continue

        if project_id not in censuses:
            censuses[project_id] = await _project_status_census(
                taskmaster, project_roots[project_id], log=logger,
            )
        census = censuses[project_id]
        if census is None:
            report['errors'] += 1
            continue

        classification = classify_orphan(esc, census)
        if classification == 'live':
            report['live'] += 1
            continue

        report[classification] += 1
        report['reapable_ids'].append(esc.id)

        if not apply:
            continue

        note = _resolution_note(
            esc, classification, project_id, census.get(str(esc.task_id)),
        )
        result = queue.resolve(
            esc.id, note,
            resolved_by=resolved_by,
            resolution_class=RESOLUTION_CLASS,
        )
        if result is None:
            logger.warning(
                'Escalation %s not found during resolve (state drift between '
                'get_pending and apply); skipping', esc.id,
            )
            continue
        report['reaped'] += 1

    if report['unresolvable']:
        logger.warning(
            '%d of %d reapable record(s) could not be scoped to a known project '
            'and were NOT classified — pass --project-root for the missing '
            'project(s), or verify those records by hand before touching them',
            report['unresolvable'], report['scanned'],
        )

    return report


def main() -> int:
    """CLI entry point.  Returns an exit code."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

    parser = argparse.ArgumentParser(
        description=(
            'Derive (and optionally close) pending recon stale escalations whose '
            'subject task went terminal or vanished.'
        ),
    )
    parser.add_argument(
        '--queue-dir',
        default=DEFAULT_QUEUE_DIR,
        help=f'Recon escalation queue directory (default: {DEFAULT_QUEUE_DIR}).',
    )
    parser.add_argument(
        '--project-root',
        action='append',
        default=None,
        help=(
            'Extra project root to include in the project_id -> root map '
            '(repeatable). Defaults to build_known_projects_map, which reads '
            'DASHBOARD_KNOWN_PROJECT_ROOTS.'
        ),
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
        help=f'resolved_by tag written to closed records (default: {RESOLVED_BY}).',
    )
    args = parser.parse_args()

    project_roots = build_known_projects_map(PROJECT_ROOT, args.project_root)
    report = asyncio.run(
        run(
            args.queue_dir,
            project_roots,
            apply=args.apply,
            resolved_by=args.resolved_by,
        ),
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
