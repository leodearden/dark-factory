#!/usr/bin/env python3
"""One-shot repair: re-point (or drop) a defective cited memory id on a finding
owned by an already-COMPLETED reconciliation run (task 3065, task 5552).

Why a script exists at all
--------------------------
The same repair is available as the ``repair_memory_citation`` MCP tool, and
both call the SAME function — ``reconciliation.citation_repair.repair_memory_citation``.
This script owns no copy of the rewrite logic; it parses flags, opens the
journal and the memory service, and delegates. Two reasons it is still worth
having:

  1. The running fused-memory server exposes the tool surface it booted with,
     so the new MCP tool is unreachable until a restart. The script acts on the
     journal directly and needs none.
  2. An operator repairing a historical audit record wants a dry run first. The
     ``apply`` flag lives INSIDE the shared function, so ``--apply``-less runs
     traverse the identical path — including both corroboration reads — and
     report exactly what the write would do.

Why the journal and not recon-report state
------------------------------------------
A closed run's recon-report state does not survive: ``recon_report_state_ttl_seconds``
defaults to 300s and ``tick()`` deletes the run's shadow rows at quiescence. The
journal's ``runs.stage_reports`` blob is the only durable home of a completed
run's findings, so that is what gets rewritten.

What it cannot do
-----------------
It cannot decide which defect class you are repairing — ``--reason`` asserts
that, and the repair CHECKS the assertion rather than trusting it. With
``--reason memory_not_found`` (the default) the victim must be CONFIRMED absent
from Mem0, else ``citation_not_dangling``. With ``--reason wrong_memory`` it
must RESOLVE, else ``citation_not_resolving``; each refusal names the other
class, so a misclassification costs a dry run rather than a bad write. Either
way a backend read that RAISES is ``verification_error``, never a repair —
unknown is neither absent nor present.

It cannot remove a still-resolving citation silently: ``--reason wrong_memory``
without a non-blank ``--justification`` is ``justification_required``. The
``citation_repairs`` record is the only surviving account of the change, so the
prose you pass is what a later reader of the audit record will have.

It cannot report a repair that changed nothing. The replacement must resolve
(else ``replacement_not_found``) and must not be the victim itself (else
``replacement_is_victim`` — that swap strips the citation and re-appends it).
And a run that is still live is refused outright, because the harness rewrites
the whole ``stage_reports`` blob at each stage end and would silently clobber
the repair.

Journal I/O that raises is reported too, not thrown: ``journal_error`` carries
the ``phase`` that failed (read / write / verify) and a hint saying whether
anything was written — a read-only data dir is the failure this path has
actually hit. And a repair that IS written but does not survive the
read-after-write check (another writer rewrote the whole blob in between) is
reported as ``repair_clobbered`` rather than a false ``repaired``. Every one of
those exits 1; only ``status: repaired`` / ``status: dry_run`` exits 0.

The incident this was written for (task 3065 — ``memory_not_found``)
--------------------------------------------------------------------
Run ``06a4466d-cdc0-49ac-8e99-e6723be39392`` (project ``reify``, completed
2026-07-26), finding ``5e85117e-51fc-4a7f-8ca7-e26078dbd3f2``, whose two cited
memories were both destroyed by a Stage-1 supersession. The surviving successor
``746b4ab9-ca3c-418b-982a-32b85bfcf94b`` names both of them in its own
``metadata.supersedes``. Two invocations repair it — the first re-points, the
second drops (the successor is already cited by then, so a second replacement
would only duplicate it):

    # 1. re-point the id the incident named
    python scripts/repair_recon_citation.py \\
        --data-dir /home/leo/src/dark-factory/data/reconciliation \\
        --target-run-id 06a4466d-cdc0-49ac-8e99-e6723be39392 \\
        --finding-id 5e85117e-51fc-4a7f-8ca7-e26078dbd3f2 \\
        --memory-id beacf7fc-b76a-4c0b-876d-f4cf6d906d42 \\
        --replacement-memory-id 746b4ab9-ca3c-418b-982a-32b85bfcf94b \\
        --apply

    # 2. drop the sibling that is also dangling (no --replacement-memory-id)
    python scripts/repair_recon_citation.py \\
        --data-dir /home/leo/src/dark-factory/data/reconciliation \\
        --target-run-id 06a4466d-cdc0-49ac-8e99-e6723be39392 \\
        --finding-id 5e85117e-51fc-4a7f-8ca7-e26078dbd3f2 \\
        --memory-id 17085708-b888-472f-bbf3-0a06634fd4db \\
        --apply

Drop ``--apply`` from either to dry-run it first. Dry-run is the default.

Status of that repair: DONE — APPLIED 2026-08-08 by the task-3065 steward. Do
NOT re-run the two invocations above; they are retained only as the worked
example of the flag shapes. Both were blocked from the task worktree
(``/home/leo/src/dark-factory/``, including ``data/``, is not writable by a
task agent — the write raised ``sqlite3.OperationalError: attempt to write a
readonly database``, and the row was verified byte-identical afterwards), and
were re-run to completion from the steward session, which does have write
access to that checkout. Sequence, all measured against the live journal:

  * pre-repair ``runs.stage_reports`` for the run re-read read-only and
    confirmed byte-identical to the blocked implementer's snapshot —
    ``sha256 ea653dc8cccf8a51a61f555adb6126056d0544f38165afac13eb0fbbe882f61c``,
    11063 bytes, no ``citation_repairs`` key.
  * both invocations re-dry-run (gates still green), then re-run with
    ``--apply``; each returned ``status: repaired``, ``removed_count: 1``.
  * post-repair blob ``sha256 2b8279a7be49d5bb…``, 11444 bytes. The finding's
    ``cited_memories`` is now exactly one entry — the successor
    ``746b4ab9-…`` with its real fingerprint (``category:
    procedural_knowledge``, ``agent_id: recon-stage-memory_consolidator``).
    Both retired ids survive only inside the new ``citation_repairs`` audit
    list (``reason: memory_not_found``, ``repaired_by:
    script:repair_recon_citation``), which is why a raw substring grep for
    them still hits — check ``cited_memories``, not the raw blob.

Rollback artifact (the pre-repair blob) is at
``/tmp/3065-rollback/pre_stage_reports.json``; note ``/tmp`` is not durable
across a reboot. Run 1 before 2 was and remains the required order: after 1
the successor is already cited, which is why 2 is drop-only rather than a
second re-point.

The incident that added ``--reason wrong_memory`` (task 5552)
-------------------------------------------------------------
Run ``cd2af61a-fe12-4222-b58c-9eb5a2070c44`` (project
``solar_challenge_platform``, status ``completed``, completed
2026-09-17T04:10:27Z), stage ``task_knowledge_sync``, finding
``7750fd64-f862-4ad8-8b1f-a9a08b1494d0``. The finding reports that task 182
gained its missing structural dependency on task 173 — Stage 2 called
``add_dependency(182, depends_on=173)`` and confirmed it with a follow-up
``get_task(182)``. Its single cited memory
``8505f9b0-dbfd-44c0-ac32-7352a85bfb6f`` RESOLVES (measured: ``get_memory_by_id``
returns ``found: true``) but is the task-168 index_health rolling summary
(``kind: index_health_confirmation_rollup``, ``task_id: 168``), which says
nothing about task 182, task 173, or any dependency. Wrong memory, not a
missing one — so the task-3065 path refuses it with ``citation_not_dangling``
and there is nothing to re-point to, because the claim's evidence is a TASK
read rather than a memory. A detach is the whole repair:

    python scripts/repair_recon_citation.py \\
        --data-dir /home/leo/src/dark-factory/data/reconciliation \\
        --target-run-id cd2af61a-fe12-4222-b58c-9eb5a2070c44 \\
        --finding-id 7750fd64-f862-4ad8-8b1f-a9a08b1494d0 \\
        --memory-id 8505f9b0-dbfd-44c0-ac32-7352a85bfb6f \\
        --reason wrong_memory \\
        --justification 'cites the task-168 index_health rollup, which makes no claim about the task-182 dependency; the claim itself was confirmed by Stage 2 via get_task(182)' \\
        --apply

Note this run is CROSS-PROJECT relative to this repo, which is why it is the
script's example and not the MCP tool's: the tool passes its own
``caller_project_id`` and would refuse with ``project_mismatch``. The script
passes none — that bypass exists for exactly this correction.

Status of that repair: NOT APPLIED as of task 5552 landing. It is retained as
the worked example of the ``wrong_memory`` flag shape, and the ids above were
read from the live journal, so a dry run (drop ``--apply``) should report the
gates green. Applying it needs write access to ``data/`` in the main checkout,
which a task agent does not have — see the task-3065 note above for the
``sqlite3.OperationalError: attempt to write a readonly database`` this raises
from a task worktree.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from fused_memory.reconciliation import citation_repair

logger = logging.getLogger(__name__)

# Stamped as the repair's provenance in place of the `run:<id>` a stage agent
# would supply. A human ran a script; the record should say so rather than
# borrow a run id that did not author the decision.
REPAIRED_BY = 'script:repair_recon_citation'


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for this script.

    Extracted from ``main()`` so tests can assert flag defaults — above all that
    ``--apply`` defaults False — without invoking the live entry point.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--target-run-id', dest='target_run_id', required=True,
        help='The COMPLETED run that owns the finding (not your own run)',
    )
    parser.add_argument(
        '--finding-id', dest='finding_id', required=True,
        help='The finding carrying the dangling citation',
    )
    parser.add_argument(
        '--memory-id', dest='memory_id', required=True,
        help='The defective cited memory id to remove. Its required state is '
             'whichever --reason names: absent, or resolving-but-wrong.',
    )
    parser.add_argument(
        '--replacement-memory-id', dest='replacement_memory_id', default=None,
        help='Live successor to cite instead (must resolve). Omit for a '
             'drop-only repair, which removes the dangling citation and cites '
             'nothing in its place.',
    )
    parser.add_argument(
        '--reason', default='memory_not_found',
        choices=['memory_not_found', 'wrong_memory'],
        help="The citation's DEFECT CLASS, which selects the corroboration the "
             "repair demands (default: memory_not_found). 'memory_not_found' "
             'asserts the cited id is CONFIRMED ABSENT from Mem0; '
             "'wrong_memory' asserts it RESOLVES but does not back the "
             'finding. Naming the wrong one is a refusal pointing at the '
             'other, never a silent reclassification. Orthogonal to drop vs '
             'swap, which is --replacement-memory-id.',
    )
    parser.add_argument(
        '--justification', default=None,
        help='Why the cited memory does not back the finding. REQUIRED with '
             '--reason wrong_memory, which removes a citation that still '
             'resolves: the citation_repairs record is then the only surviving '
             'account of the change, so state what the citation should have '
             'backed and how the claim was independently confirmed. Optional '
             'for --reason memory_not_found, whose confirmed absence is its '
             'own account, but recorded when given.',
    )
    parser.add_argument(
        '--store', default='mem0', choices=['mem0', 'graphiti'],
        help="Citation store (default: mem0). 'graphiti' is refused by the "
             'repair path — the dangling check is a Mem0/Qdrant point read and '
             'would false-flag every graph citation.',
    )
    parser.add_argument(
        '--data-dir', dest='data_dir', default=None,
        help='Reconciliation data dir holding reconciliation.db '
             '(default: the configured reconciliation.data_dir)',
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Commit the repair (default: dry-run, report only)',
    )
    return parser


async def run(args: argparse.Namespace, *, journal: Any, memory: Any) -> dict[str, Any]:
    """Delegate to the shared repair function and return its outcome verbatim.

    Deliberately thin: the dry-run/apply switch lives inside
    ``citation_repair.repair_memory_citation`` so this script and the MCP tool
    traverse one code path (INV-5).
    """
    return await citation_repair.repair_memory_citation(
        journal,
        memory,
        target_run_id=args.target_run_id,
        finding_id=args.finding_id,
        memory_id=args.memory_id,
        store=args.store,
        replacement_memory_id=args.replacement_memory_id,
        reason=args.reason,
        justification=args.justification,
        repaired_by=REPAIRED_BY,
        apply=args.apply,
    )


def exit_code_for(outcome: dict[str, Any]) -> int:
    """0 only for a resolved repair or a clean dry-run; 1 for anything else.

    An outcome carrying neither key is a contract break — it exits 1 rather
    than being read as success.
    """
    return 0 if outcome.get('status') in {'repaired', 'dry_run'} else 1


def report(outcome: dict[str, Any]) -> int:
    """Print the outcome as indented JSON and return its exit code.

    JSON, not prose: the operator gets the structured facts — which id was
    removed, which was cited, which gate refused — in a shape that survives
    being pasted into an incident record (INV-2).
    """
    print(json.dumps(outcome, indent=2, sort_keys=True, default=str))
    return exit_code_for(outcome)


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )

    args = build_parser().parse_args()

    async def _run_live() -> int:
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
        from fused_memory.reconciliation.journal import (  # noqa: PLC0415
            ReconciliationJournal,
        )
        from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

        config = FusedMemoryConfig()
        data_dir = Path(args.data_dir or config.reconciliation.data_dir)

        journal = ReconciliationJournal(data_dir)
        await journal.initialize()
        memory = MemoryService(config)
        await memory.initialize()
        try:
            outcome = await run(args, journal=journal, memory=memory)
        finally:
            if hasattr(memory, 'close'):
                await memory.close()
            await journal.close()
        return report(outcome)

    return asyncio.run(_run_live())


if __name__ == '__main__':
    sys.exit(main())
