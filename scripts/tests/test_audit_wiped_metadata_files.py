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
    _SOURCE_PRECEDENCE,
    CONFIRMED_NULL_SHA_DONE_PATH,
    CONTRADICTED_REAL_MERGE_SHA,
    FIDELITY_FILE_LEVEL,
    FIDELITY_LOCK_LEVEL,
    NO_MERGE_EVENT,
    NO_SUCCESSFUL_MERGE_SHA,
    AuditCoverage,
    PlanFilesRecord,
    ProjectAudit,
    TaskRecord,
    WipeCandidate,
    audit_project,
    classify_wipe_signature,
    format_json,
    format_report,
    load_merge_signatures,
    load_plan_files_from_disk,
    load_plan_files_from_events,
    load_task_records,
    merge_plan_file_sources,
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
    """(f) .task-meta is a SIBLING of the worktrees, not one of them, and must
    be skipped when iterating the base dir — otherwise
    ``<base>/.task-meta/.task/plan.json`` is ingested as a legacy LANE record.

    That trap path is reachable by BOTH scans (the meta-root scan sees a
    ``.task`` entry; a non-skipping base scan sees a ``.task-meta`` lane), so
    mere absence of the trap task id cannot discriminate between them. The
    real observable is the SOURCE LABEL: nothing found under the meta-root may
    ever be attributed to ``legacy_worktree_plan_json``, since a mislabelled
    provenance is exactly what would mislead a downstream repair.
    """
    _write_meta_root_plan(tmp_path, "10", {"task_id": 10, "files": ["ok.py"]})
    trap = tmp_path / ".task-meta" / ".task"
    trap.mkdir(parents=True)
    (trap / "plan.json").write_text(json.dumps({"task_id": 999, "files": ["trap.py"]}))

    records = load_plan_files_from_disk(str(tmp_path))

    assert records["10"].files == ("ok.py",)
    assert [r.source for r in records.values()] == ["meta_root_plan_json"] * len(records)


def test_load_plan_files_from_disk_on_absent_base_returns_empty(tmp_path):
    """(g) an absent worktree_base returns an empty dict rather than raising."""
    assert load_plan_files_from_disk(str(tmp_path / "no-such-base")) == {}
    # An existing but empty base is also fine.
    assert load_plan_files_from_disk(str(tmp_path)) == {}


# ---------------------------------------------------------------------------
# merge_plan_file_sources — four-way precedence across disk and event sources:
#   meta_root_plan_json > legacy_worktree_plan_json
#     > phase_skipped_event > set_to_plan_event
# ---------------------------------------------------------------------------


def _rec(files, source, fidelity=FIDELITY_FILE_LEVEL):
    return PlanFilesRecord(files=tuple(files), source=source, fidelity=fidelity)


def test_merge_plan_file_sources_prefers_disk_over_event_within_file_level():
    """(a) both are FILE_LEVEL, but the PERSISTED plan artifact is
    authoritative over an event snapshot of it."""
    disk = {"7": _rec(["from_plan.py"], "meta_root_plan_json")}
    events = {"7": _rec(["from_event.py"], "phase_skipped_event")}

    merged = merge_plan_file_sources(disk, events)

    assert merged["7"].source == "meta_root_plan_json"
    assert merged["7"].files == ("from_plan.py",)


def test_merge_plan_file_sources_prefers_legacy_disk_over_event_snapshot():
    """(a) the legacy disk artifact still outranks an event snapshot."""
    disk = {"7": _rec(["legacy.py"], "legacy_worktree_plan_json")}
    events = {"7": _rec(["evented.py"], "phase_skipped_event")}

    assert merge_plan_file_sources(disk, events)["7"].source == "legacy_worktree_plan_json"


def test_merge_plan_file_sources_prefers_any_file_level_over_lock_level():
    """(b) every FILE_LEVEL source beats the LOCK_LEVEL set_to_plan record."""
    lock = _rec(["orchestrator"], "set_to_plan_event", FIDELITY_LOCK_LEVEL)
    for source in ("meta_root_plan_json", "legacy_worktree_plan_json"):
        merged = merge_plan_file_sources({"7": _rec(["f.py"], source)}, {"7": lock})
        assert merged["7"].source == source
        assert merged["7"].fidelity == FIDELITY_FILE_LEVEL

    merged = merge_plan_file_sources(
        {}, {"7": _rec(["f.py"], "phase_skipped_event")}
    )
    assert merged["7"].source == "phase_skipped_event"


def test_merge_plan_file_sources_carries_through_single_source_tasks():
    """(c) a task present in only one map keeps its source and fidelity."""
    disk = {"1": _rec(["only_disk.py"], "meta_root_plan_json")}
    events = {"2": _rec(["orchestrator"], "set_to_plan_event", FIDELITY_LOCK_LEVEL)}

    merged = merge_plan_file_sources(disk, events)

    assert merged["1"] == _rec(["only_disk.py"], "meta_root_plan_json")
    assert merged["2"].source == "set_to_plan_event"
    assert merged["2"].fidelity == FIDELITY_LOCK_LEVEL
    assert merged["2"].files == ("orchestrator",)


def test_merge_plan_file_sources_does_not_mutate_either_input():
    """(d) neither input dict is mutated."""
    disk = {"7": _rec(["d.py"], "meta_root_plan_json")}
    events = {"7": _rec(["e.py"], "phase_skipped_event"), "8": _rec(["x.py"], "phase_skipped_event")}
    disk_before = dict(disk)
    events_before = dict(events)

    merged = merge_plan_file_sources(disk, events)

    assert disk == disk_before
    assert events == events_before
    assert merged is not disk and merged is not events


def test_merge_plan_file_sources_covers_the_union_of_task_ids():
    """(e) the result covers the union of both inputs' task ids."""
    disk = {"1": _rec(["a.py"], "meta_root_plan_json"), "2": _rec(["b.py"], "legacy_worktree_plan_json")}
    events = {"2": _rec(["c.py"], "phase_skipped_event"), "3": _rec(["d.py"], "phase_skipped_event")}

    assert set(merge_plan_file_sources(disk, events)) == {"1", "2", "3"}


def test_merge_plan_file_sources_on_empty_inputs_returns_empty():
    assert merge_plan_file_sources({}, {}) == {}


def test_merge_plan_file_sources_precedence_is_stated_once_and_is_total():
    """The ordering lives in a single module-level tuple, and every source
    label the loaders can emit appears in it — an unranked source would
    otherwise be silently unorderable."""
    assert _SOURCE_PRECEDENCE == (
        "meta_root_plan_json",
        "legacy_worktree_plan_json",
        "phase_skipped_event",
        "set_to_plan_event",
    )


# ---------------------------------------------------------------------------
# classify_wipe_signature / load_merge_signatures — the REFINED, state-aware,
# per-TASK discriminator.
#
# The naive per-EVENT signature ("a merge_finalized with merge_sha=null")
# over-reports ~20x: measured over all 1622 merge_finalized events in the live
# runs.db, 446 carry a null sha but 365 of those are state='blocked' and 14
# are state='conflict' — FAILED merge attempts that were later retried and
# landed with a real sha. Those never reached DONE and so wiped nothing.
# Payload shape below mirrors the sole emit site, merge_queue.py:3830-3844.
# ---------------------------------------------------------------------------


def _finalized(state, merge_sha=None, **extra):
    payload = {
        "request_id": "req-1",
        "branch": "task/7",
        "state": state,
        "snapshot_tip": "aaa111",
        "merge_sha": merge_sha,
        "superseded_by": None,
        "generation": 0,
        "reason": None,
    }
    payload.update(extra)
    return payload


def test_classify_already_merged_with_null_sha_is_confirmed():
    """(a) the real DONE-with-no-sha shortcut (21 live events, all null-sha)."""
    assert classify_wipe_signature([_finalized("already_merged")]) == (
        CONFIRMED_NULL_SHA_DONE_PATH
    )


def test_classify_events_that_never_obtained_a_sha_are_no_successful_merge_sha():
    """(b) events exist but NEVER include a non-null merge_sha."""
    assert classify_wipe_signature(
        [_finalized("conflict"), _finalized("abandoned")]
    ) == NO_SUCCESSFUL_MERGE_SHA


def test_classify_null_sha_failure_followed_by_a_real_merge_is_contradicted():
    """(c) THE OVER-REPORT GUARD. A null-sha 'blocked' row followed by a
    non-null-sha 'done' row is a FAILED ATTEMPT THAT WAS RETRIED AND LANDED —
    it never took the workflow to DONE with _merge_sha=None, so it wiped
    nothing and must NOT be reported as a confirmed wipe."""
    assert classify_wipe_signature(
        [_finalized("blocked"), _finalized("done", "abc123")]
    ) == CONTRADICTED_REAL_MERGE_SHA


def test_classify_no_events_is_unknown_not_clean():
    """(d) other DONE paths (found_on_main recovery, eval mode) wipe without
    ever emitting merge_finalized, so silence is UNKNOWN, not exoneration."""
    assert classify_wipe_signature([]) == NO_MERGE_EVENT


def test_classify_already_merged_wins_over_a_sibling_successful_row():
    """(e) the already_merged shortcut IS the wipe, so it outranks a sibling
    row that did carry a real sha — in either order."""
    assert classify_wipe_signature(
        [_finalized("already_merged"), _finalized("blocked", "xyz789")]
    ) == CONFIRMED_NULL_SHA_DONE_PATH
    assert classify_wipe_signature(
        [_finalized("done", "xyz789"), _finalized("already_merged")]
    ) == CONFIRMED_NULL_SHA_DONE_PATH


def test_classify_treats_a_done_row_with_a_null_sha_as_confirmed():
    """Defensive: state='done' carries a non-null sha in 1159/1159 live rows,
    so this shape was not observed — but if it ever occurs it IS the wipe
    (DONE reached with _merge_sha=None), not a retried failure."""
    assert classify_wipe_signature([_finalized("done")]) == (
        CONFIRMED_NULL_SHA_DONE_PATH
    )


def test_classify_tolerates_junk_payload_entries():
    """A non-dict payload entry is ignored rather than crashing the sweep.

    Junk-ONLY input degrades to NO_MERGE_EVENT (unknown), not to
    NO_SUCCESSFUL_MERGE_SHA — the latter is a positive claim ("this task
    attempted a merge and never obtained a sha") that undecodable rows do not
    support. Manufacturing a finding out of junk is the failure mode this
    audit exists to avoid.
    """
    assert classify_wipe_signature(["not-a-dict", None]) == NO_MERGE_EVENT
    assert classify_wipe_signature(
        ["junk", _finalized("done", "abc123")]
    ) == CONTRADICTED_REAL_MERGE_SHA


def test_classify_treats_an_empty_string_sha_as_no_sha():
    """An empty-string merge_sha is not a real merge sha."""
    assert classify_wipe_signature([_finalized("blocked", "")]) == (
        NO_SUCCESSFUL_MERGE_SHA
    )


def test_load_merge_signatures_groups_by_task_in_id_order(tmp_path):
    db_path = _make_runs_db(
        tmp_path,
        [
            {"event_type": "merge_finalized", "task_id": 7, "data": _finalized("blocked")},
            {"event_type": "merge_finalized", "task_id": 8, "data": _finalized("already_merged")},
            {"event_type": "merge_finalized", "task_id": 7, "data": _finalized("done", "abc123")},
        ],
    )
    signatures = load_merge_signatures(str(db_path))

    assert set(signatures) == {"7", "8"}
    assert [p["state"] for p in signatures["7"]] == ["blocked", "done"]
    assert signatures["7"][1]["merge_sha"] == "abc123"
    assert [p["state"] for p in signatures["8"]] == ["already_merged"]


def test_load_merge_signatures_skips_malformed_and_ignores_other_event_types(tmp_path):
    db_path = _make_runs_db(
        tmp_path,
        [
            {"event_type": "merge_finalized", "task_id": 7, "data": "{not json"},
            {"event_type": "merge_finalized", "task_id": 7, "data": None},
            {"event_type": "merge_finalized", "task_id": 7, "data": "[1,2,3]"},
            {"event_type": "merge_finalized", "task_id": None, "data": _finalized("done", "a")},
            {"event_type": "merge_attempt", "task_id": 9, "data": _finalized("done", "a")},
            {"event_type": "merge_finalized", "task_id": 7, "data": _finalized("conflict")},
        ],
    )
    signatures = load_merge_signatures(str(db_path))

    assert set(signatures) == {"7"}
    assert [p["state"] for p in signatures["7"]] == ["conflict"]


def test_load_merge_signatures_on_empty_db_returns_empty(tmp_path):
    assert load_merge_signatures(str(_make_runs_db(tmp_path, []))) == {}


# ---------------------------------------------------------------------------
# audit_project — end-to-end over one project root:
#   <root>/.taskmaster/tasks/tasks.db      (harness.py:1877 convention)
#   <root>/data/orchestrator/runs.db
#   <root>/.worktrees                      (config.py:1410 worktree_dir)
# ---------------------------------------------------------------------------


def _make_project(tmp_path, tasks=(), events=(), plans=(), name="proj"):
    """Build a whole project root with the three inputs audit_project reads.

    *plans* is a list of ``(worktree_name, plan_dict)`` written to the
    canonical meta-root location.
    """
    root = tmp_path / name
    tasks_dir = root / ".taskmaster" / "tasks"
    tasks_dir.mkdir(parents=True)
    _make_tasks_db(tasks_dir, list(tasks))

    runs_dir = root / "data" / "orchestrator"
    runs_dir.mkdir(parents=True)
    _make_runs_db(runs_dir, list(events))

    worktrees = root / ".worktrees"
    worktrees.mkdir(parents=True)
    for wt_name, plan in plans:
        _write_meta_root_plan(worktrees, wt_name, plan)
    return root


def test_audit_project_reports_a_wiped_task_with_full_provenance(tmp_path):
    """(a) non-empty plan scope + empty metadata.files IS reported, carrying
    every field a downstream repair would need."""
    root = _make_project(
        tmp_path,
        tasks=[{"id": 2464, "status": "done", "metadata": {"files": []}}],
        events=[
            {
                "event_type": "merge_finalized",
                "task_id": 2464,
                "data": _finalized("already_merged"),
            }
        ],
        plans=[("2464", {"task_id": 2464, "files": ["a.py", "b.py"]})],
    )

    audit = audit_project(str(root))

    assert len(audit.candidates) == 1
    candidate = audit.candidates[0]
    assert candidate.task_id == 2464
    assert candidate.tag == "master"
    assert candidate.status == "done"
    assert candidate.plan_files == ("a.py", "b.py")
    assert candidate.plan_files_source == "meta_root_plan_json"
    assert candidate.plan_files_fidelity == FIDELITY_FILE_LEVEL
    assert candidate.wipe_signature == CONFIRMED_NULL_SHA_DONE_PATH


def test_audit_project_does_not_report_a_task_whose_files_survived(tmp_path):
    """(b) non-empty metadata.files means nothing was wiped."""
    root = _make_project(
        tmp_path,
        tasks=[{"id": 1, "status": "done", "metadata": {"files": ["a.py"]}}],
        plans=[("1", {"task_id": 1, "files": ["a.py"]})],
    )
    audit = audit_project(str(root))

    assert audit.candidates == []
    assert audit.coverage.total_tasks == 1
    assert audit.coverage.tasks_with_file_level_signal == 1


def test_audit_project_does_not_report_a_task_whose_plan_declared_no_scope(tmp_path):
    """(c) an empty plan file list is not a declared scope, so an empty
    metadata.files is not evidence of a wipe."""
    root = _make_project(
        tmp_path,
        tasks=[{"id": 1, "status": "done", "metadata": {"files": []}}],
        plans=[("1", {"task_id": 1, "files": []})],
    )
    audit = audit_project(str(root))

    assert audit.candidates == []
    # No usable plan signal was recovered, so the task is UNKNOWN, not clean.
    assert audit.coverage.tasks_without_plan_signal == 1
    assert audit.coverage.tasks_with_file_level_signal == 0


def test_audit_project_counts_plan_records_with_no_matching_task(tmp_path):
    """(d) a plan/event signal for a task absent from tasks.db is counted
    separately rather than crashing or being silently dropped."""
    root = _make_project(
        tmp_path,
        tasks=[{"id": 1, "status": "done", "metadata": {"files": []}}],
        plans=[
            ("1", {"task_id": 1, "files": ["a.py"]}),
            ("9999", {"task_id": 9999, "files": ["ghost.py"]}),
        ],
    )
    audit = audit_project(str(root))

    assert [c.task_id for c in audit.candidates] == [1]
    assert audit.coverage.plan_records_without_task == 1


def test_audit_project_orders_candidates_by_tag_then_numeric_id(tmp_path):
    """(e) deterministic ordering — numeric, so 100 sorts after 20."""
    ids = [100, 20, 3]
    root = _make_project(
        tmp_path,
        tasks=(
            [{"id": i, "tag": "master", "metadata": {"files": []}} for i in ids]
            + [{"id": 50, "tag": "alpha", "metadata": {"files": []}}]
        ),
        plans=[(str(i), {"task_id": i, "files": [f"{i}.py"]}) for i in ids + [50]],
    )
    audit = audit_project(str(root))

    assert [(c.tag, c.task_id) for c in audit.candidates] == [
        ("alpha", 50),
        ("master", 3),
        ("master", 20),
        ("master", 100),
    ]


def test_audit_project_coverage_counts_every_tier(tmp_path):
    """(f) total, file-level, lock-level-only, and no-signal counts, where
    no-signal includes tasks with no plan record at all."""
    root = _make_project(
        tmp_path,
        tasks=[
            {"id": 1, "metadata": {"files": []}},   # file-level signal
            {"id": 2, "metadata": {"files": []}},   # lock-level signal only
            {"id": 3, "metadata": {"files": []}},   # no signal at all
            {"id": 4, "metadata": {"files": ["kept.py"]}},  # file-level, intact
        ],
        events=[
            {"event_type": "set_to_plan", "task_id": 2, "data": {"files": ["orchestrator"]}},
            {"event_type": "phase_skipped", "task_id": 4, "data": {"plan_files": ["kept.py"]}},
        ],
        plans=[("1", {"task_id": 1, "files": ["one.py"]})],
    )
    audit = audit_project(str(root))

    assert audit.coverage.total_tasks == 4
    assert audit.coverage.tasks_with_file_level_signal == 2   # tasks 1 and 4
    assert audit.coverage.tasks_with_lock_level_signal_only == 1  # task 2
    assert audit.coverage.tasks_without_plan_signal == 1      # task 3
    assert audit.coverage.project_root == str(root)
    # Task 2's candidate carries the lock-level label so a repair cannot
    # mistake a module path for a plan.files entry.
    lock_candidates = [c for c in audit.candidates if c.task_id == 2]
    assert lock_candidates[0].plan_files_fidelity == FIDELITY_LOCK_LEVEL
    assert lock_candidates[0].plan_files == ("orchestrator",)


def test_audit_project_degrades_to_no_merge_event_without_a_runs_db(tmp_path):
    """A project with no runs.db still audits; every signature is UNKNOWN."""
    root = _make_project(
        tmp_path,
        tasks=[{"id": 1, "metadata": {"files": []}}],
        plans=[("1", {"task_id": 1, "files": ["a.py"]})],
    )
    (root / "data" / "orchestrator" / "runs.db").unlink()

    audit = audit_project(str(root))

    assert [c.wipe_signature for c in audit.candidates] == [NO_MERGE_EVENT]


def test_audit_project_on_an_empty_project_reports_zero_and_does_not_crash(tmp_path):
    root = _make_project(tmp_path)
    audit = audit_project(str(root))

    assert audit.candidates == []
    assert audit.coverage.total_tasks == 0
    assert audit.coverage.tasks_without_plan_signal == 0


# ---------------------------------------------------------------------------
# format_report / format_json.
#
# The COVERAGE block is LOAD-BEARING, not decoration: without it a reader sees
# "N damaged tasks" and concludes that is the whole blast radius, when in fact
# most tasks have no recoverable plan.files and are genuinely UNKNOWN.
# ---------------------------------------------------------------------------


def _candidate(task_id=1, **overrides):
    fields = {
        "tag": "master",
        "task_id": task_id,
        "status": "done",
        "plan_files": ("a.py", "b.py"),
        "plan_files_source": "meta_root_plan_json",
        "plan_files_fidelity": FIDELITY_FILE_LEVEL,
        "wipe_signature": CONFIRMED_NULL_SHA_DONE_PATH,
    }
    fields.update(overrides)
    return WipeCandidate(**fields)


def _coverage(root="/tmp/proj", **overrides):
    fields = {
        "project_root": root,
        "total_tasks": 3264,
        "tasks_with_file_level_signal": 815,
        "tasks_with_lock_level_signal_only": 400,
        "tasks_without_plan_signal": 2049,
        "plan_records_without_task": 2,
    }
    fields.update(overrides)
    return AuditCoverage(**fields)


def _audit(candidates=(), root="/tmp/proj", **coverage_overrides):
    return ProjectAudit(
        project_root=root,
        candidates=list(candidates),
        coverage=_coverage(root, **coverage_overrides),
    )


def test_format_report_renders_every_field_a_repair_would_need(tmp_path):
    """(a) each line carries task_id, tag, status, signature, source,
    fidelity and the plan file count, grouped under the project root."""
    report = format_report([_audit([_candidate(2464)])])

    assert "/tmp/proj" in report
    assert "2464" in report
    assert "master" in report
    assert "done" in report
    assert CONFIRMED_NULL_SHA_DONE_PATH in report
    assert "meta_root_plan_json" in report
    assert FIDELITY_FILE_LEVEL in report
    assert "2" in report  # the plan file count


def test_format_report_always_prints_coverage_even_with_zero_candidates():
    """(b) THE LOAD-BEARING ASSERTION. A zero-candidate audit must still state
    how much of the population it could not see, instead of printing a bare
    'no candidates' line that reads as 'nothing is damaged'."""
    report = format_report([_audit([])])

    assert "COVERAGE" in report
    assert "3264" in report      # total tasks scanned
    assert "2049" in report      # tasks with NO recoverable plan signal
    lowered = report.lower()
    assert "no plan signal" in lowered or "without plan signal" in lowered


def test_format_report_marks_lock_level_candidates_with_an_explicit_caveat():
    """(c) a LOCK_LEVEL candidate's paths are MODULE paths, and the report
    must say so — a repair that backfilled them verbatim would corrupt
    metadata.files."""
    report = format_report(
        [
            _audit(
                [
                    _candidate(
                        7,
                        plan_files=("orchestrator",),
                        plan_files_source="set_to_plan_event",
                        plan_files_fidelity=FIDELITY_LOCK_LEVEL,
                    )
                ]
            )
        ]
    )

    lowered = report.lower()
    assert "lock-level" in lowered or "lock_level" in lowered
    assert "module" in lowered
    assert "not" in lowered  # "not verbatim plan.files"


def test_format_report_separates_contradicted_from_confirmed_candidates():
    """(d) CONTRADICTED_REAL_MERGE_SHA candidates go in their own section so a
    repair job cannot blindly consume them alongside confirmed ones."""
    report = format_report(
        [
            _audit(
                [
                    _candidate(1, wipe_signature=CONFIRMED_NULL_SHA_DONE_PATH),
                    _candidate(2, wipe_signature=CONTRADICTED_REAL_MERGE_SHA),
                ]
            )
        ]
    )

    assert CONTRADICTED_REAL_MERGE_SHA in report
    confirmed_at = report.index(CONFIRMED_NULL_SHA_DONE_PATH)
    contradicted_at = report.index(CONTRADICTED_REAL_MERGE_SHA)
    # Distinct sections, confirmed first.
    assert confirmed_at < contradicted_at
    lowered = report.lower()
    assert "contradicted" in lowered


def test_format_json_emits_an_object_with_untruncated_files_and_coverage():
    """(e)(f) a JSON OBJECT (not a bare array), carrying full file lists plus
    coverage, round-tripping through json.loads so a follow-up repair task can
    consume it directly."""
    long_files = tuple(f"pkg/module_{i}.py" for i in range(40))
    payload = json.loads(
        format_json([_audit([_candidate(2464, plan_files=long_files)])])
    )

    assert isinstance(payload, dict)
    assert list(payload) == ["projects"]
    project = payload["projects"][0]
    assert project["project_root"] == "/tmp/proj"
    assert project["coverage"]["total_tasks"] == 3264
    assert project["coverage"]["tasks_without_plan_signal"] == 2049
    candidate = project["candidates"][0]
    assert candidate["task_id"] == 2464
    assert candidate["plan_files"] == list(long_files)  # UNTRUNCATED
    assert candidate["plan_files_fidelity"] == FIDELITY_FILE_LEVEL
    assert candidate["wipe_signature"] == CONFIRMED_NULL_SHA_DONE_PATH


def test_format_json_includes_zero_candidate_projects_with_their_coverage():
    """A clean project still appears, with its coverage — the JSON consumer
    needs the same honesty about invisibility the human report gives."""
    payload = json.loads(format_json([_audit([])]))

    project = payload["projects"][0]
    assert project["candidates"] == []
    assert project["coverage"]["total_tasks"] == 3264


def test_format_report_and_json_handle_multiple_projects():
    audits = [_audit([_candidate(1)], root="/tmp/a"), _audit([], root="/tmp/b")]

    report = format_report(audits)
    assert "/tmp/a" in report and "/tmp/b" in report

    payload = json.loads(format_json(audits))
    assert [p["project_root"] for p in payload["projects"]] == ["/tmp/a", "/tmp/b"]
