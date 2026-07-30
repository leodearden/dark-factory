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
    FIDELITY_FILE_LEVEL,
    FIDELITY_LOCK_LEVEL,
    PlanFilesRecord,
    TaskRecord,
    load_plan_files_from_disk,
    load_plan_files_from_events,
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


# ---------------------------------------------------------------------------
# load_plan_files_from_events — recover plan scope from durable event payloads.
#
# Two event-borne sources, deliberately NOT of equal fidelity:
#   - phase_skipped.plan_files (workflow.py:4275, :4447) — TRUE file-level
#     plan.files.
#   - set_to_plan.files (scheduler.py:6987-6994) — the LOCK-level `needed`
#     (module) set, explicitly not the file-level persist set
#     (event_store.py:77-82, scheduler.py:6982-6984).
# ---------------------------------------------------------------------------


def test_load_plan_files_from_events_reads_phase_skipped_as_file_level(tmp_path):
    """(a) phase_skipped.plan_files is FILE_LEVEL."""
    db_path = _make_runs_db(
        tmp_path,
        [
            {
                "event_type": "phase_skipped",
                "task_id": 2085,
                "phase": "plan",
                "data": {
                    "reason": "revalidation_skipped_no_overlap",
                    "plan_session_id": "2085-abc",
                    "plan_files": ["orchestrator/src/orchestrator/workflow.py"],
                    "main_sha": "deadbeef",
                },
            }
        ],
    )
    records = load_plan_files_from_events(str(db_path))

    assert list(records) == ["2085"]
    record = records["2085"]
    assert isinstance(record, PlanFilesRecord)
    assert record.files == ("orchestrator/src/orchestrator/workflow.py",)
    assert record.source == "phase_skipped_event"
    assert record.fidelity == FIDELITY_FILE_LEVEL


def test_load_plan_files_from_events_reads_set_to_plan_as_lock_level(tmp_path):
    """(b) set_to_plan.files carries the lock-level `needed` set, so it is
    tagged LOCK_LEVEL and must never be presented as verbatim plan.files."""
    db_path = _make_runs_db(
        tmp_path,
        [
            {
                "event_type": "set_to_plan",
                "task_id": 2085,
                "data": {
                    "files": ["orchestrator", "shared"],
                    "released": [],
                    "acquired": ["shared"],
                    "persisted": True,
                },
            }
        ],
    )
    record = load_plan_files_from_events(str(db_path))["2085"]

    assert record.files == ("orchestrator", "shared")
    assert record.source == "set_to_plan_event"
    assert record.fidelity == FIDELITY_LOCK_LEVEL


def test_load_plan_files_from_events_file_level_beats_lock_level_either_order(tmp_path):
    """(c) FILE_LEVEL wins regardless of which row has the higher id."""
    phase_skipped = {
        "event_type": "phase_skipped",
        "task_id": 7,
        "data": {"plan_files": ["a.py"]},
    }
    set_to_plan = {
        "event_type": "set_to_plan",
        "task_id": 7,
        "data": {"files": ["orchestrator"]},
    }

    lock_last = load_plan_files_from_events(
        str(_make_runs_db(tmp_path, [phase_skipped, set_to_plan], name="a.db"))
    )
    file_last = load_plan_files_from_events(
        str(_make_runs_db(tmp_path, [set_to_plan, phase_skipped], name="b.db"))
    )

    for records in (lock_last, file_last):
        assert records["7"].fidelity == FIDELITY_FILE_LEVEL
        assert records["7"].files == ("a.py",)


def test_load_plan_files_from_events_latest_wins_within_one_fidelity_tier(tmp_path):
    """(d) within one tier the later row (higher id) wins."""
    db_path = _make_runs_db(
        tmp_path,
        [
            {"event_type": "phase_skipped", "task_id": 7, "data": {"plan_files": ["old.py"]}},
            {"event_type": "phase_skipped", "task_id": 7, "data": {"plan_files": ["new.py"]}},
        ],
    )
    assert load_plan_files_from_events(str(db_path))["7"].files == ("new.py",)

    lock_db = _make_runs_db(
        tmp_path,
        [
            {"event_type": "set_to_plan", "task_id": 8, "data": {"files": ["old_mod"]}},
            {"event_type": "set_to_plan", "task_id": 8, "data": {"files": ["new_mod"]}},
        ],
        name="lock.db",
    )
    assert load_plan_files_from_events(str(lock_db))["8"].files == ("new_mod",)


def test_load_plan_files_from_events_skips_every_unusable_row(tmp_path):
    """(e)(f)(g): missing key, empty list, malformed/NULL data, NULL task_id,
    and unrelated event types are all skipped without raising."""
    db_path = _make_runs_db(
        tmp_path,
        [
            # (e) phase_skipped with no plan_files key at all.
            {"event_type": "phase_skipped", "task_id": 1, "data": {"reason": "x"}},
            # (e) plan_files present but empty.
            {"event_type": "phase_skipped", "task_id": 2, "data": {"plan_files": []}},
            # (e) malformed JSON payload.
            {"event_type": "phase_skipped", "task_id": 3, "data": "{not json"},
            # (e) NULL payload.
            {"event_type": "phase_skipped", "task_id": 4, "data": None},
            # (e) payload decodes to a non-dict.
            {"event_type": "set_to_plan", "task_id": 5, "data": "[1,2,3]"},
            # (e) files is a wrong-typed bare string.
            {"event_type": "set_to_plan", "task_id": 6, "data": {"files": "orchestrator"}},
            # (f) NULL task_id.
            {"event_type": "phase_skipped", "task_id": None, "data": {"plan_files": ["z.py"]}},
            # (g) an unrelated event type carrying a files key.
            {"event_type": "merge_finalized", "task_id": 9, "data": {"files": ["q.py"]}},
            {"event_type": "lock_acquired", "task_id": 10, "data": {"plan_files": ["q.py"]}},
        ],
    )
    assert load_plan_files_from_events(str(db_path)) == {}


def test_load_plan_files_from_events_does_not_let_an_empty_later_row_erase_a_hit(tmp_path):
    """A later row whose list is empty is skipped, not treated as an update
    that blanks the earlier real signal."""
    db_path = _make_runs_db(
        tmp_path,
        [
            {"event_type": "phase_skipped", "task_id": 7, "data": {"plan_files": ["real.py"]}},
            {"event_type": "phase_skipped", "task_id": 7, "data": {"plan_files": []}},
        ],
    )
    assert load_plan_files_from_events(str(db_path))["7"].files == ("real.py",)


def test_load_plan_files_from_events_on_empty_db_returns_empty_mapping(tmp_path):
    assert load_plan_files_from_events(str(_make_runs_db(tmp_path, []))) == {}


# ---------------------------------------------------------------------------
# load_plan_files_from_disk — recover plan scope from the two on-disk plan
# locations, new-then-old, mirroring git_ops.py:6833-6835 and
# harness.py:2721-2735.
#
#   canonical: <base>/.task-meta/<name>/plan.json   (config.py:1343
#              TASK_META_DIRNAME, artifacts.py:287 meta_root_for)
#   legacy:    <base>/<name>/.task/plan.json        (in the real layout this
#              is only an ABSOLUTE SYMLINK to the canonical artifact,
#              artifacts.py:354-386)
# ---------------------------------------------------------------------------


def _write_meta_root_plan(base: Path, name: str, plan: object) -> Path:
    path = base / ".task-meta" / name / "plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan if isinstance(plan, str) else json.dumps(plan))
    return path


def _write_legacy_plan(base: Path, name: str, plan: object) -> Path:
    path = base / name / ".task" / "plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan if isinstance(plan, str) else json.dumps(plan))
    return path


def test_load_plan_files_from_disk_reads_canonical_meta_root(tmp_path):
    """(a) the canonical meta-root plan, keyed by its in-file task_id."""
    _write_meta_root_plan(tmp_path, "2085", {"task_id": 2085, "files": ["x.py"]})

    records = load_plan_files_from_disk(str(tmp_path))

    assert list(records) == ["2085"]
    record = records["2085"]
    assert isinstance(record, PlanFilesRecord)
    assert record.files == ("x.py",)
    assert record.source == "meta_root_plan_json"
    assert record.fidelity == FIDELITY_FILE_LEVEL


def test_load_plan_files_from_disk_reads_legacy_worktree_plan(tmp_path):
    """(b) the legacy in-worktree location is read too."""
    _write_legacy_plan(tmp_path, "1074", {"task_id": 1074, "files": ["legacy.py"]})

    record = load_plan_files_from_disk(str(tmp_path))["1074"]

    assert record.files == ("legacy.py",)
    assert record.source == "legacy_worktree_plan_json"
    assert record.fidelity == FIDELITY_FILE_LEVEL


def test_load_plan_files_from_disk_prefers_meta_root_over_legacy(tmp_path):
    """(c) when both exist for one task the meta-root wins."""
    _write_meta_root_plan(tmp_path, "500", {"task_id": 500, "files": ["new.py"]})
    _write_legacy_plan(tmp_path, "500", {"task_id": 500, "files": ["stale.py"]})

    record = load_plan_files_from_disk(str(tmp_path))["500"]

    assert record.files == ("new.py",)
    assert record.source == "meta_root_plan_json"


def test_load_plan_files_from_disk_prefers_in_file_task_id_over_dir_name(tmp_path):
    """(d) the plan self-identifies — a pooled lane directory name is not the
    task id (the same self-identification _find_lane_by_plan_task_id relies
    on, git_ops.py:6793-6845) — falling back to the dir name when absent."""
    _write_meta_root_plan(tmp_path, "lane-3", {"task_id": 2222, "files": ["a.py"]})
    _write_meta_root_plan(tmp_path, "3131", {"files": ["b.py"]})
    # A numeric task_id must key as its string form, not as an int.
    _write_legacy_plan(tmp_path, "lane-9", {"task_id": 4444, "files": ["c.py"]})

    records = load_plan_files_from_disk(str(tmp_path))

    assert set(records) == {"2222", "3131", "4444"}
    assert records["2222"].files == ("a.py",)
    assert records["3131"].files == ("b.py",)
    assert records["4444"].files == ("c.py",)


def test_load_plan_files_from_disk_resolves_legacy_symlink_without_duplicating(tmp_path):
    """(e) the REAL layout: the legacy path is an absolute symlink into the
    meta-root. Reading it must not crash and must not produce a second,
    lower-precedence record for the same task."""
    canonical = _write_meta_root_plan(
        tmp_path, "2464", {"task_id": 2464, "files": ["real.py"]}
    )
    legacy_task_dir = tmp_path / "2464" / ".task"
    legacy_task_dir.mkdir(parents=True)
    (legacy_task_dir / "plan.json").symlink_to(canonical.resolve())

    records = load_plan_files_from_disk(str(tmp_path))

    assert list(records) == ["2464"]
    assert records["2464"].source == "meta_root_plan_json"
    assert records["2464"].files == ("real.py",)


def test_load_plan_files_from_disk_skips_every_unusable_entry(tmp_path):
    """(f) malformed JSON, missing/empty/non-list files, a dangling symlink,
    a non-directory entry, and the .task-meta dir itself appearing in the
    base listing are all skipped without raising."""
    _write_meta_root_plan(tmp_path, "1", "{not json at all")
    _write_meta_root_plan(tmp_path, "2", {"task_id": 2})
    _write_meta_root_plan(tmp_path, "3", {"task_id": 3, "files": []})
    _write_meta_root_plan(tmp_path, "4", {"task_id": 4, "files": "not-a-list"})
    _write_meta_root_plan(tmp_path, "5", '["a list, not an object"]')
    _write_legacy_plan(tmp_path, "6", "{also not json")

    # A dangling symlink at the legacy path (its meta-root target was removed).
    dangling_dir = tmp_path / "7" / ".task"
    dangling_dir.mkdir(parents=True)
    (dangling_dir / "plan.json").symlink_to(tmp_path / ".task-meta" / "7" / "plan.json")

    # A plain FILE sitting directly in the worktree base (not a directory).
    (tmp_path / "stray-file.txt").write_text("not a worktree")

    # A worktree dir with no .task/ at all.
    (tmp_path / "8").mkdir()

    # plan.json is a DIRECTORY rather than a file.
    (tmp_path / ".task-meta" / "9" / "plan.json").mkdir(parents=True)

    assert load_plan_files_from_disk(str(tmp_path)) == {}


def test_load_plan_files_from_disk_does_not_treat_task_meta_as_a_worktree(tmp_path):
    """(f) .task-meta must be skipped when iterating the base dir, or a
    <base>/.task-meta/.task/plan.json would be mis-scanned as a legacy lane."""
    _write_meta_root_plan(tmp_path, "10", {"task_id": 10, "files": ["ok.py"]})
    trap = tmp_path / ".task-meta" / ".task"
    trap.mkdir(parents=True)
    (trap / "plan.json").write_text(json.dumps({"task_id": 999, "files": ["trap.py"]}))

    records = load_plan_files_from_disk(str(tmp_path))

    assert "999" not in records
    assert records["10"].files == ("ok.py",)


def test_load_plan_files_from_disk_on_absent_base_returns_empty(tmp_path):
    """(g) an absent worktree_base returns an empty dict rather than raising."""
    assert load_plan_files_from_disk(str(tmp_path / "no-such-base")) == {}
    # An existing but empty base is also fine.
    assert load_plan_files_from_disk(str(tmp_path)) == {}
