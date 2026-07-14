"""Shared test fixtures for curator replay corpus unit tests (task 2496).

Builds a synthetic sqlite ``tickets.db`` mirroring the REAL schema
(``fused_memory/middleware/ticket_store.py`` ``TABLE_SQL``) so the reader/
build/audit unit tests are fully deterministic and never touch the real
gitignored ``data/reconciliation/tickets.db``. The schema below is a
documented copy -- NOT imported -- keeping the orchestrator eval package
decoupled from fused_memory (mirrors ``_reviewer_trial_mining_fixtures.py``'s
synthetic ``runs.db`` pattern).

Uniquely named (like ``_reviewer_trial_mining_fixtures.py``) so it can be
imported from test files without colliding with sibling subprojects'
conftest modules.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

# Mirrors fused_memory/middleware/ticket_store.py TABLE_SQL -- only the
# columns the curator replay reader actually consumes.
_SCHEMA = """
CREATE TABLE tickets (
    ticket_id   TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    candidate_json TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    task_id     TEXT,
    reason      TEXT,
    result_json TEXT,
    created_at  TEXT NOT NULL,
    resolved_at TEXT,
    expires_at  TEXT NOT NULL,
    escalated_at TEXT
);
"""

_DEFAULT_CANDIDATE = {
    'title': '',
    'description': '',
    'details': '',
    'files_to_modify': [],
    'priority': 'medium',
    'spawned_from': None,
}


def make_synthetic_tickets_db(path: Path, rows: list[dict]) -> Path:
    """Create a sqlite ``tickets.db`` at *path* mirroring the real schema.

    Each entry in *rows* describes one ``tickets`` row, with optional keys:

    - ``ticket_id`` (str, required)
    - ``project_id`` (str, default ``'dark_factory'``)
    - ``candidate`` (dict, default a minimal candidate titled after
      ticket_id) -- json-serialized into ``candidate_json``. Pass
      ``candidate_json`` (a raw str) instead to exercise malformed-JSON
      handling.
    - ``status`` (str, default ``'created'``) -- one of
      pending/created/combined/failed
    - ``task_id`` (str | None)
    - ``reason`` (str | None)
    - ``result`` (dict, optional) -- json-serialized into ``result_json``.
      Pass ``result_json`` (a raw str) instead to exercise malformed-JSON
      handling. Omit both to leave ``result_json`` NULL (mirrors a
      still-pending or otherwise result-less row).
    - ``created_at``/``resolved_at``/``expires_at``/``escalated_at`` (str,
      optional -- sane defaults are used so tests only need to set what
      they care about)

    Returns *path*.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_SCHEMA)
        for row in rows:
            ticket_id = row['ticket_id']

            if 'candidate_json' in row:
                candidate_json = row['candidate_json']
            else:
                candidate = {
                    **_DEFAULT_CANDIDATE,
                    'title': f'Candidate {ticket_id}',
                    **row.get('candidate', {}),
                }
                candidate_json = json.dumps(candidate)

            if 'result_json' in row:
                result_json = row['result_json']
            elif 'result' in row:
                result_json = json.dumps(row['result'])
            else:
                result_json = None

            conn.execute(
                """
                INSERT INTO tickets
                    (ticket_id, project_id, candidate_json, status, task_id,
                     reason, result_json, created_at, resolved_at, expires_at,
                     escalated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    row.get('project_id', 'dark_factory'),
                    candidate_json,
                    row.get('status', 'created'),
                    row.get('task_id'),
                    row.get('reason'),
                    result_json,
                    row.get('created_at', '2026-01-01T00:00:00+00:00'),
                    row.get('resolved_at'),
                    row.get('expires_at', '2027-01-01T00:00:00+00:00'),
                    row.get('escalated_at'),
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return path
