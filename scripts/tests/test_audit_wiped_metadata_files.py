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

from audit_wiped_metadata_files import (
    TaskRecord,
    load_task_records,
)

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


# ---------------------------------------------------------------------------
# load_task_records — read tasks.db into TaskRecords keyed by (tag, id).
# ---------------------------------------------------------------------------


def test_load_task_records_carries_tag_id_status_and_files(tmp_path):
    db_path = _make_tasks_db(
        tmp_path,
        [{"id": 42, "status": "done", "metadata": {"files": ["a.py", "b.py"]}}],
    )
    records = load_task_records(str(db_path))

    assert list(records) == [("master", 42)]
    record = records[("master", 42)]
    assert isinstance(record, TaskRecord)
    assert record.tag == "master"
    assert record.task_id == 42
    assert record.status == "done"
    # (b) a well-formed files list yields exactly those entries.
    assert record.metadata_files == ("a.py", "b.py")


def test_load_task_records_degrades_every_unusable_metadata_shape_to_empty(tmp_path):
    """(c) + (d): empty list, absent key, NULL, malformed JSON, and a
    wrong-typed `files` all yield an empty tuple WITHOUT raising."""
    db_path = _make_tasks_db(
        tmp_path,
        [
            {"id": 1, "metadata": {"files": []}},
            {"id": 2, "metadata": {}},
            {"id": 3, "metadata": None},
            {"id": 4, "metadata": "{not valid json at all"},
            # (d) `files` is a bare string, not a list.
            {"id": 5, "metadata": {"files": "scripts/thing.py"}},
            # metadata decodes to a non-dict entirely.
            {"id": 6, "metadata": "[1, 2, 3]"},
            {"id": 7, "metadata": '"just a string"'},
            {"id": 8, "metadata": {"files": None}},
        ],
    )
    records = load_task_records(str(db_path))

    assert len(records) == 8
    for task_id in range(1, 9):
        assert records[("master", task_id)].metadata_files == (), f"task {task_id}"


def test_load_task_records_keys_by_tag_and_id_so_tags_do_not_collide(tmp_path):
    """(e) the same numeric id under two tags must not collide."""
    db_path = _make_tasks_db(
        tmp_path,
        [
            {"id": 9, "tag": "master", "status": "done", "metadata": {"files": []}},
            {"id": 9, "tag": "feature", "status": "pending", "metadata": {"files": ["x.py"]}},
        ],
    )
    records = load_task_records(str(db_path))

    assert set(records) == {("master", 9), ("feature", 9)}
    assert records[("master", 9)].metadata_files == ()
    assert records[("master", 9)].status == "done"
    assert records[("feature", 9)].metadata_files == ("x.py",)
    assert records[("feature", 9)].status == "pending"


def test_load_task_records_coerces_non_string_file_entries(tmp_path):
    """A files list carrying non-string junk is coerced/filtered rather than
    crashing a downstream str-formatting consumer."""
    db_path = _make_tasks_db(
        tmp_path, [{"id": 1, "metadata": {"files": ["a.py", None, 3, ""]}}]
    )
    records = load_task_records(str(db_path))
    assert records[("master", 1)].metadata_files == ("a.py",)


def test_load_task_records_on_empty_db_returns_empty_mapping(tmp_path):
    db_path = _make_tasks_db(tmp_path, [])
    assert load_task_records(str(db_path)) == {}
