#!/usr/bin/env python3
"""Print per-write backend telemetry (duration and LLM tokens) from the write journal.

Runs ``fused_memory.services.write_journal.OPERATOR_TELEMETRY_QUERY``, the one
copy of the query, so this report cannot drift from the schema the journal
writes. The journal is opened read-only (``mode=ro`` and ``query_only``), which
makes it safe against the serving instance's live file. Prints one JSON object
per ``backend_ops`` row, most recent first.

    uv run --frozen --project fused-memory python \\
        fused-memory/scripts/telemetry_query.py --since 2026-10-04T00:00:00Z --limit 50
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fused_memory.services.write_journal import OPERATOR_TELEMETRY_QUERY

DEFAULT_JOURNAL = Path('/home/leo/src/dark-factory/data/reconciliation/write_journal.db')
DEFAULT_WINDOW = timedelta(days=1)
DEFAULT_LIMIT = 50
CONNECT_TIMEOUT_SECONDS = 5.0


def journal_timestamp(text: str) -> str:
    """``text`` in the journal's own ``created_at`` form; a naive value is taken as UTC.

    The query compares timestamps as text, so a ``Z`` suffix or a non-UTC
    offset would shift the window unless it is normalised first.
    """
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'not an ISO-8601 timestamp: {text!r}') from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat()


def read_telemetry(journal: Path, since: str, limit: int) -> list[dict[str, Any]]:
    con = sqlite3.connect(
        f'{journal.resolve().as_uri()}?mode=ro', uri=True, timeout=CONNECT_TIMEOUT_SECONDS
    )
    try:
        con.execute('PRAGMA query_only=ON')
        con.row_factory = sqlite3.Row
        return [dict(row) for row in con.execute(OPERATOR_TELEMETRY_QUERY, (since, limit))]
    finally:
        con.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--journal', type=Path, default=DEFAULT_JOURNAL)
    parser.add_argument(
        '--since',
        type=journal_timestamp,
        default=(datetime.now(UTC) - DEFAULT_WINDOW).isoformat(),
        help='lower created_at bound, ISO-8601; naive means UTC (default: 24 hours ago)',
    )
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = read_telemetry(args.journal, args.since, args.limit)
    except sqlite3.Error as exc:
        print(f'cannot read {args.journal}: {exc}', file=sys.stderr)
        return 1
    for row in rows:
        print(json.dumps(row))
    return 0


if __name__ == '__main__':
    sys.exit(main())
