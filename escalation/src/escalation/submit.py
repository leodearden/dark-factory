"""Escalation submit CLI — file-backed born-at-L2 escalation writer.

Provides a script-callable entry point so a detached systemd OnFailure unit
can file a born-at-L2 escalation WITHOUT the MCP server.

Invocation forms:
    python -m escalation submit ...      (package form; requires __main__.py)
    escalation submit ...                (console-script form; requires pyproject.toml [project.scripts])

Both forms pass 'submit' as the leading token, consumed by the 'submit' subparser.

KEY SEMANTIC: EscalationQueue.submit() writes the Escalation record as-is and
does NOT compute `level`.  The born-at-L2 stamping that normally lives in
server.py's _chokepoint_or_submit closure is BYPASSED by the file-backed CLI,
so this module must replicate the sentinel-born-at-L2 contract (PRD I3) by
constructing Escalation(level=2, ...) directly.

Severity is restricted to BORN_AT_L2_SEVERITIES (critical, urgent) via argparse
choices so that a non-L2 severity is rejected at the argument boundary BEFORE
any disk write — consistent with the CLI's sole purpose of producing L2 records.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from escalation.models import BORN_AT_L2_SEVERITIES, Escalation
from escalation.queue import EscalationQueue

logger = logging.getLogger(__name__)

# Default sentinel agent_role — matches PRD invariant I3 (orchestrator-deterministic)
# and the server's sentinel namespace (orchestrator-*), keeping on-disk records
# consistent with server-filed L2s for dashboard/audit.
DEFAULT_AGENT_ROLE = 'orchestrator-deterministic'


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for filing a born-at-L2 escalation to the file-backed queue.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).

    Returns:
        0 on success; argparse exits with 2 on invalid arguments.
    """
    parser = argparse.ArgumentParser(
        description='File a born-at-L2 escalation directly to the file-backed queue.',
    )
    subparsers = parser.add_subparsers(dest='command')

    sub = subparsers.add_parser(
        'submit',
        help='Submit a born-at-L2 escalation to the queue.',
    )
    sub.add_argument(
        '--queue-dir',
        required=True,
        type=Path,
        help='Path to the escalation queue directory (created if absent).',
    )
    sub.add_argument(
        '--task',
        required=True,
        dest='task',
        help='Task ID this escalation is filed against.',
    )
    sub.add_argument(
        '--severity',
        required=True,
        choices=sorted(BORN_AT_L2_SEVERITIES),
        help=(
            f'Severity — must be one of {sorted(BORN_AT_L2_SEVERITIES)}. '
            'Only BORN_AT_L2_SEVERITIES values are accepted; the CLI exists '
            'to produce L2 records, and a non-L2 severity would silently '
            'write an L0 record defeating the purpose.'
        ),
    )
    sub.add_argument(
        '--category',
        required=True,
        help='Escalation category (e.g. infra_issue, scope_violation).',
    )
    sub.add_argument(
        '--summary',
        required=True,
        help='One-line summary of the escalation.',
    )
    sub.add_argument(
        '--detail',
        default='',
        help='Full context / detail for the escalation (default: empty).',
    )
    sub.add_argument(
        '--agent-role',
        default=DEFAULT_AGENT_ROLE,
        dest='agent_role',
        help=(
            f'Sentinel agent role (default: {DEFAULT_AGENT_ROLE!r}). '
            'Must start with orchestrator-* to satisfy the sentinel namespace contract.'
        ),
    )

    args = parser.parse_args(argv)

    if args.command != 'submit':
        parser.print_help()
        return 2

    # EscalationQueue.__init__ creates queue_dir (parents=True, exist_ok=True),
    # matching the detached OnFailure context where the dir may not yet exist.
    q = EscalationQueue(args.queue_dir)
    esc_id = q.make_id(args.task)

    # Stamp level=2 directly — the server's _chokepoint_or_submit path that
    # normally computes this is bypassed by the file-backed CLI (PRD I3).
    esc = Escalation(
        id=esc_id,
        task_id=args.task,
        agent_role=args.agent_role,
        severity=args.severity,
        category=args.category,
        summary=args.summary,
        detail=args.detail,
        level=2,
    )
    q.submit(esc)
    logger.info('Submitted born-at-L2 escalation: %s [%s]', esc.id, esc.severity)
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    sys.exit(main())
