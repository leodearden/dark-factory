"""Apply missing schema migrations to the reconciliation SQLite database.

Idempotent — safe to run multiple times. Uses CREATE TABLE IF NOT EXISTS
so it only creates tables that don't already exist.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path('data/reconciliation/reconciliation.db')

_MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS chunk_boundaries (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    run_id TEXT,
    events_count INTEGER,
    status TEXT DEFAULT 'processing',
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunk_project ON chunk_boundaries(project_id);

CREATE TABLE IF NOT EXISTS run_actions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    target TEXT NOT NULL,
    operation TEXT NOT NULL,
    detail TEXT DEFAULT '{}',
    causation_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ra_run ON run_actions(run_id);
"""


def migrate(db_path: Path) -> list[str]:
    """Apply missing tables to the reconciliation DB.

    Returns list of table names that were created (empty if already up-to-date).
    """
    if not db_path.exists():
        logger.error('Database not found at %s', db_path)
        return []

    conn = sqlite3.connect(str(db_path))
    try:
        # Check which tables exist before migration
        existing = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        conn.executescript(_MIGRATION_SQL)
        conn.commit()

        # Check which tables are new
        after = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        created = sorted(after - existing)

        if created:
            logger.info('Created tables: %s', ', '.join(created))
        else:
            logger.info('Schema already up-to-date, no tables created')

        return created
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Apply missing schema migrations')
    parser.add_argument(
        '--db-path',
        type=Path,
        default=_DEFAULT_DB_PATH,
        help=f'Path to reconciliation DB (default: {_DEFAULT_DB_PATH})',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    created = migrate(args.db_path)
    if created:
        print(f'Created {len(created)} table(s): {", ".join(created)}')
    else:
        print('Schema already up-to-date.')


if __name__ == '__main__':
    main()
