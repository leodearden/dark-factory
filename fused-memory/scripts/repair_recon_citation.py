#!/usr/bin/env python3
"""One-shot repair: re-point (or drop) a dangling cited memory id on a finding
owned by an already-COMPLETED reconciliation run (task 3065).

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
It can only repair provenance, never rewrite a live claim. The victim citation
must be CONFIRMED absent from Mem0 (else ``citation_not_dangling``) and the
replacement must resolve (else ``replacement_not_found``); a backend read that
RAISES is ``verification_error``, never a repair — unknown is not absent. A run
that is still live is refused outright, because the harness rewrites the whole
``stage_reports`` blob at each stage end and would silently clobber the repair.

The incident this was written for
---------------------------------
Run ``06a4466d-cdc0-49ac-8e99-e6723be39392`` (project ``reify``, completed
2026-07-26), finding ``5e85117e-51fc-4a7f-8ca7-e26078dbd3f2``, whose two cited
memories were both destroyed by a Stage-1 supersession. The surviving successor
``746b4ab9-ca3c-418b-982a-32b85bfcf94b`` names both of them in its own
``metadata.supersedes``. Two invocations repair it — the first re-points, the
second drops (the successor is already cited by then, so a second replacement
would only duplicate it):

    # 1. re-point the id the incident named
    python scripts/repair_recon_citation.py \\
        --target-run-id 06a4466d-cdc0-49ac-8e99-e6723be39392 \\
        --finding-id 5e85117e-51fc-4a7f-8ca7-e26078dbd3f2 \\
        --memory-id beacf7fc-b76a-4c0b-876d-f4cf6d906d42 \\
        --replacement-memory-id 746b4ab9-ca3c-418b-982a-32b85bfcf94b \\
        --apply

    # 2. drop the sibling that is also dangling (no --replacement-memory-id)
    python scripts/repair_recon_citation.py \\
        --target-run-id 06a4466d-cdc0-49ac-8e99-e6723be39392 \\
        --finding-id 5e85117e-51fc-4a7f-8ca7-e26078dbd3f2 \\
        --memory-id 17085708-b888-472f-bbf3-0a06634fd4db \\
        --apply

Drop ``--apply`` from either to dry-run it first. Dry-run is the default.
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
        help='The dangling cited memory id to remove (must be confirmed absent)',
    )
    parser.add_argument(
        '--replacement-memory-id', dest='replacement_memory_id', default=None,
        help='Live successor to cite instead (must resolve). Omit for a '
             'drop-only repair, which removes the dangling citation and cites '
             'nothing in its place.',
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
