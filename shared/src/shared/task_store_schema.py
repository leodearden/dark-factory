"""The DDL a fresh task store (``shared.project_layout.tasks_db``) is created with.

The task store executes it on every connection open, before its migrations; indexes
that migrations add (``ux_tasks_candidate_key``) are not here. It imports only the
standard library, so stdlib-only consumers such as the script suites can build the
production shape.
"""

from typing import Final

__all__ = ['TASK_STORE_DDL']

TASK_STORE_DDL: Final = """
CREATE TABLE IF NOT EXISTS tasks (
    tag             TEXT NOT NULL DEFAULT 'master',
    id              INTEGER NOT NULL,
    title           TEXT NOT NULL,
    description     TEXT,
    details         TEXT,
    test_strategy   TEXT,
    status          TEXT NOT NULL,
    priority        TEXT,
    metadata        TEXT,
    updated_at      TEXT NOT NULL,
    claimant_run_id TEXT,
    heartbeat_at    TEXT,
    candidate_key   TEXT,
    created_at      TEXT,
    created_at_source TEXT,
    PRIMARY KEY (tag, id)
);

CREATE INDEX IF NOT EXISTS ix_tasks_status ON tasks (tag, status);

CREATE TABLE IF NOT EXISTS dependencies (
    tag        TEXT NOT NULL DEFAULT 'master',
    task_id    INTEGER NOT NULL,
    depends_on INTEGER NOT NULL,
    PRIMARY KEY (tag, task_id, depends_on)
);

-- Monotonic id high-water mark per tag sequence.  ``add_task`` allocates
-- ``max(MAX(tasks.id), max_id) + 1`` so a deleted id is NEVER reissued.
-- Applied idempotently via executescript on every connection open.
CREATE TABLE IF NOT EXISTS id_counters (
    tag    TEXT NOT NULL DEFAULT 'master',
    max_id INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (tag)
);
"""
