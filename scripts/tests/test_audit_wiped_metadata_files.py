"""Tests for scripts/audit_wiped_metadata_files.py — the READ-ONLY audit of
the DONE-path ``metadata.files`` wipe.

Task 3146: `TaskWorkflow._reconcile_metadata_files_for_done`
(orchestrator/src/orchestrator/workflow.py:2001-2021) contains an
``elif self._merge_sha: ... else: files = []`` ladder that blanks a task's
``metadata.files`` when the DONE path is reached without a merge sha. This
module tests the audit that enumerates the observable blast radius. Neither
the audit nor these tests ever mutate a task or event record.

Mirrors test_scan_task_toolcall_leaks.py: pure functions get direct pytest
coverage; ``main()`` gets subprocess coverage.

NO TEST HERE ASSERTS A COUNT OR TASK ID DERIVED FROM THE LIVE DATABASES.
tasks.db and runs.db are mutated continuously by the running orchestrator, so
a test pinning "the live DB yields N candidates" would be a guessed threshold
that goes red the moment another task merges. Every assertion runs against
synthetic temp databases built by the helpers below, whose contents the test
controls exactly.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# Temp-DB fixture builders.
#
# Both schemas MIRROR THE LIVE SCHEMAS so tests exercise real column shapes
# and NOT NULL constraints rather than invented ones:
#   - tasks:  fused-memory's sqlite_task_backend.py _SCHEMA_SQL, verified
#             read-only against /home/leo/src/dark-factory/.taskmaster/tasks/tasks.db
#   - events: orchestrator/src/orchestrator/event_store.py:23-41 (_SCHEMA)
# Only the columns the audit actually reads are included, plus the NOT NULL
# ones it must satisfy to insert a realistic row.
# ---------------------------------------------------------------------------

_TASKS_SCHEMA = """
CREATE TABLE tasks (
    tag           TEXT NOT NULL DEFAULT 'master',
    id            INTEGER NOT NULL,
    title         TEXT NOT NULL,
    description   TEXT,
    details       TEXT,
    test_strategy TEXT,
    status        TEXT NOT NULL,
    priority      TEXT,
    metadata      TEXT,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (tag, id)
);
"""

_EVENTS_SCHEMA = """
CREATE TABLE events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    run_id      TEXT    NOT NULL,
    task_id     TEXT,
    event_type  TEXT    NOT NULL,
    phase       TEXT,
    role        TEXT,
    data        TEXT    DEFAULT '{}',
    cost_usd    REAL,
    duration_ms INTEGER
);
"""


def _make_tasks_db(tmp_path: Path, rows: list[dict], name: str = "tasks.db") -> Path:
    """Build a temp tasks.db mirroring the live schema and insert *rows*.

    Each row dict may carry: ``id`` (required), ``tag`` (default 'master'),
    ``title``, ``status`` (default 'done'), ``priority``, ``updated_at``, and
    ``metadata``. ``metadata`` is passed through VERBATIM when it is a str or
    None — so a test can insert deliberately malformed JSON — and json-encoded
    when it is a dict/list.
    """
    db_path = tmp_path / name
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_TASKS_SCHEMA)
        for row in rows:
            metadata = row.get("metadata")
            if metadata is not None and not isinstance(metadata, str):
                metadata = json.dumps(metadata)
            conn.execute(
                "INSERT INTO tasks (tag, id, title, description, details, "
                "test_strategy, status, priority, metadata, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row.get("tag", "master"),
                    row["id"],
                    row.get("title", f"task {row['id']}"),
                    row.get("description"),
                    row.get("details"),
                    row.get("test_strategy"),
                    row.get("status", "done"),
                    row.get("priority", "medium"),
                    metadata,
                    row.get("updated_at", "2026-07-30T00:00:00+00:00"),
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _make_runs_db(tmp_path: Path, events: list[dict], name: str = "runs.db") -> Path:
    """Build a temp runs.db mirroring the live events schema and insert *events*.

    Each event dict may carry: ``event_type`` (required), ``task_id``,
    ``run_id``, ``timestamp``, ``phase``, ``role``, and ``data``. ``data`` is
    passed through VERBATIM when it is a str or None — so a test can insert
    malformed JSON or a NULL payload — and json-encoded otherwise. Rows are
    inserted in list order, so ``events`` order IS ascending ``id`` order,
    which is the ordering the audit relies on for "latest wins".
    """
    db_path = tmp_path / name
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_EVENTS_SCHEMA)
        for i, event in enumerate(events):
            data = event.get("data")
            if data is not None and not isinstance(data, str):
                data = json.dumps(data)
            task_id = event.get("task_id")
            conn.execute(
                "INSERT INTO events (timestamp, run_id, task_id, event_type, "
                "phase, role, data, cost_usd, duration_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.get("timestamp", f"2026-07-30T00:00:{i:02d}+00:00"),
                    event.get("run_id", "run-1"),
                    None if task_id is None else str(task_id),
                    event["event_type"],
                    event.get("phase"),
                    event.get("role"),
                    data,
                    event.get("cost_usd"),
                    event.get("duration_ms"),
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Fixture-builder self-checks: these guard the builders themselves, since every
# later test's correctness rests on them producing live-shaped rows.
# ---------------------------------------------------------------------------


def test_make_tasks_db_roundtrips_rows(tmp_path):
    db_path = _make_tasks_db(
        tmp_path,
        [
            {"id": 1, "status": "done", "metadata": {"files": ["a.py"]}},
            {"id": 2, "tag": "other", "status": "cancelled", "metadata": None},
            {"id": 3, "status": "done", "metadata": "{not json"},
        ],
    )
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT tag, id, status, metadata FROM tasks ORDER BY tag, id"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [
        ("master", 1, "done", '{"files": ["a.py"]}'),
        ("master", 3, "done", "{not json"),
        ("other", 2, "cancelled", None),
    ]


def test_make_runs_db_assigns_ascending_ids_in_list_order(tmp_path):
    db_path = _make_runs_db(
        tmp_path,
        [
            {"event_type": "set_to_plan", "task_id": 7, "data": {"files": ["m"]}},
            {"event_type": "phase_skipped", "task_id": 7, "data": None},
            {"event_type": "merge_finalized", "task_id": None, "data": "{bad"},
        ],
    )
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT id, task_id, event_type, data FROM events ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [
        (1, "7", "set_to_plan", '{"files": ["m"]}'),
        (2, "7", "phase_skipped", None),
        (3, None, "merge_finalized", "{bad"),
    ]
