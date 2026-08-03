"""Tests for scripts/audit_combine_gate_marker_loss.py — the READ-ONLY
detector for curator-combine metadata loss.

Task 3591: ``TaskInterceptor``'s combine path
(fused-memory/src/fused_memory/server/task_interceptor.py:2100) writes
``{'curator_action': 'combine', 'curator_justification', 'combined_at'}``
with ``metadata_mode='replace'``, so every OTHER metadata key the task
carried is dropped. This module tests the detector that enumerates the
observable blast radius. Neither the detector nor these tests ever mutate a
task, ticket, or manifest record.

Mirrors test_audit_wiped_metadata_files.py: pure functions get direct pytest
coverage; ``main()`` gets subprocess coverage.

NO TEST HERE ASSERTS A COUNT OR TASK ID DERIVED FROM THE LIVE DATABASES.
tasks.db and tickets.db are mutated continuously by the running orchestrator,
so a test pinning "the live DB yields N findings" would be a guessed threshold
that goes red the moment another task is combined. Every assertion runs
against synthetic temp databases built by the helpers below, whose contents
the test controls exactly.
"""
from __future__ import annotations

import itertools
import json
import sqlite3
from pathlib import Path

from audit_combine_gate_marker_loss import (
    CombineTarget,
    load_combine_targets,
    load_ticket_expectations,
    tickets_db_path,
)

# ---------------------------------------------------------------------------
# load_combine_targets — the combine-target population.
#
# The tasks-table schema and the tasks.db builder live in scripts/tests/
# conftest.py behind the `make_tasks_db` fixture (task 3336). It passes
# `metadata` through VERBATIM when it is a str or None, which is exactly what
# the malformed-metadata degradation matrix below needs.
# ---------------------------------------------------------------------------

# The exact three keys the combine path leaves behind (task_interceptor.py:2100).
_WIPE_SIGNATURE = {
    "curator_action": "combine",
    "curator_justification": "duplicate of 3100",
    "combined_at": "2026-08-01T00:00:00+00:00",
}


def test_load_combine_targets_selects_only_combine_rows(make_tasks_db):
    """Only tasks whose metadata.curator_action == 'combine' are returned."""
    db = make_tasks_db([
        {"id": 10, "status": "pending", "metadata": _WIPE_SIGNATURE},
        {"id": 11, "status": "done", "metadata": {"source": "agent-followup"}},
        {"id": 12, "status": "done", "metadata": {"curator_action": "create"}},
    ])

    targets = load_combine_targets(str(db))

    assert set(targets) == {("master", 10)}


def test_load_combine_targets_record_shape(make_tasks_db):
    """Each record is a CombineTarget carrying tag/task_id/status/metadata_keys."""
    db = make_tasks_db([
        {"id": 3157, "tag": "master", "status": "in-progress", "metadata": _WIPE_SIGNATURE},
    ])

    target = load_combine_targets(str(db))[("master", 3157)]

    assert isinstance(target, CombineTarget)
    assert target.tag == "master"
    assert target.task_id == 3157
    assert target.status == "in-progress"
    # metadata_keys is the LIVE key set — the three-key wipe signature here.
    assert set(target.metadata_keys) == set(_WIPE_SIGNATURE)


def test_load_combine_targets_keys_by_tag_and_id(make_tasks_db):
    """Keyed by the full (tag, id) primary key, never collapsed to a bare id."""
    db = make_tasks_db([
        {"id": 5, "tag": "master", "status": "done", "metadata": _WIPE_SIGNATURE},
        {"id": 5, "tag": "other", "status": "pending", "metadata": _WIPE_SIGNATURE},
    ])

    targets = load_combine_targets(str(db))

    assert set(targets) == {("master", 5), ("other", 5)}
    assert targets[("other", 5)].status == "pending"


def test_load_combine_targets_metadata_keys_are_a_tuple(make_tasks_db):
    """metadata_keys is a tuple (NamedTuple field parity with the precedent)."""
    db = make_tasks_db([
        {"id": 1, "metadata": {**_WIPE_SIGNATURE, "task_kind": "deterministic"}},
    ])

    target = load_combine_targets(str(db))[("master", 1)]

    assert isinstance(target.metadata_keys, tuple)
    assert "task_kind" in target.metadata_keys


# ---------------------------------------------------------------------------
# The degradation matrix.
#
# One corrupt row must never abort a sweep over thousands of tasks: each
# malformed shape is skipped, and the well-formed combine row alongside it
# still comes back. Asserted row-by-row rather than as one loop so a
# regression names the exact shape that broke.
# ---------------------------------------------------------------------------

# `make_tasks_db` defaults to the name 'tasks.db' inside a single tmp_path, so
# two calls in ONE test would collide on "table tasks already exists". A
# counter gives each call its own file, which is what lets the matrix below be
# asserted row-by-row rather than as one opaque loop.
_DB_SEQ = itertools.count()


def _survives_alongside(make_tasks_db, bad_metadata):
    """Seed a bad row next to a good combine row; return the surviving keys."""
    db = make_tasks_db(
        [
            {"id": 1, "status": "done", "metadata": bad_metadata},
            {"id": 2, "status": "done", "metadata": _WIPE_SIGNATURE},
        ],
        name=f"tasks-{next(_DB_SEQ)}.db",
    )
    return set(load_combine_targets(str(db)))


def test_load_combine_targets_skips_null_metadata(make_tasks_db):
    assert _survives_alongside(make_tasks_db, None) == {("master", 2)}


def test_load_combine_targets_skips_empty_string_metadata(make_tasks_db):
    assert _survives_alongside(make_tasks_db, "") == {("master", 2)}


def test_load_combine_targets_skips_invalid_json_metadata(make_tasks_db):
    assert _survives_alongside(make_tasks_db, "{not json at all") == {("master", 2)}


def test_load_combine_targets_skips_json_list_metadata(make_tasks_db):
    assert _survives_alongside(make_tasks_db, '["curator_action"]') == {("master", 2)}


def test_load_combine_targets_skips_json_scalar_metadata(make_tasks_db):
    """A bare JSON scalar decodes fine but is not a dict — skipped, not raised."""
    assert _survives_alongside(make_tasks_db, '"combine"') == {("master", 2)}
    assert _survives_alongside(make_tasks_db, "17") == {("master", 2)}
    assert _survives_alongside(make_tasks_db, "null") == {("master", 2)}


def test_load_combine_targets_skips_dict_without_curator_action(make_tasks_db):
    assert _survives_alongside(make_tasks_db, '{"source": "prd"}') == {("master", 2)}


def test_load_combine_targets_skips_curator_action_create(make_tasks_db):
    """curator_action='create' is the curator's OTHER verdict — not a wipe."""
    assert _survives_alongside(make_tasks_db, '{"curator_action": "create"}') == {("master", 2)}


def test_load_combine_targets_skips_non_string_curator_action(make_tasks_db):
    """A wrong-typed curator_action is corrupt data, not a combine."""
    assert _survives_alongside(make_tasks_db, '{"curator_action": 1}') == {("master", 2)}
    assert _survives_alongside(make_tasks_db, '{"curator_action": null}') == {("master", 2)}


def test_load_combine_targets_empty_db_returns_empty_dict(make_tasks_db):
    """No rows at all is an empty result, never a raise."""
    assert load_combine_targets(str(make_tasks_db([]))) == {}


# ---------------------------------------------------------------------------
# load_ticket_expectations — comparison source (1), the creating ticket.
#
# The schema MIRRORS THE LIVE ONE (data/reconciliation/tickets.db, verified
# read-only) so tests exercise real column shapes and NOT NULL constraints
# rather than invented ones. It stays LOCAL to this file rather than moving
# into scripts/tests/conftest.py, exactly as test_audit_wiped_metadata_files.py
# keeps its audit-specific `_make_runs_db` local: the tickets schema is
# specific to this detector, so promoting it would widen the blast radius of a
# change to every scripts/ test for no reuse benefit.
# ---------------------------------------------------------------------------

_TICKETS_SCHEMA = """
CREATE TABLE tickets (
    ticket_id      TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL,
    candidate_json TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'pending',
    task_id        TEXT,
    reason         TEXT,
    result_json    TEXT,
    created_at     TEXT NOT NULL,
    resolved_at    TEXT,
    expires_at     TEXT NOT NULL,
    escalated_at   TEXT
);
"""


def _make_tickets_db(tmp_path: Path, rows: list[dict], name: str = "tickets.db") -> Path:
    """Build a temp tickets.db mirroring the live schema and insert *rows*.

    Each row dict may carry ``project_id`` (required), ``task_id``, ``status``,
    and either ``candidate_json`` (passed through VERBATIM when a str, so a
    test can insert malformed JSON) or ``metadata`` (wrapped into the real
    ``{'kwargs', 'metadata', 'project_root'}`` candidate shape).
    """
    db_path = tmp_path / name
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_TICKETS_SCHEMA)
        for i, row in enumerate(rows):
            if "candidate_json" in row:
                candidate = row["candidate_json"]
                if candidate is not None and not isinstance(candidate, str):
                    candidate = json.dumps(candidate)
            else:
                candidate = json.dumps({
                    "kwargs": {"title": f"task {i}"},
                    "metadata": row.get("metadata", {}),
                    "project_root": "/home/leo/src/dark-factory",
                })
            task_id = row.get("task_id")
            conn.execute(
                "INSERT INTO tickets (ticket_id, project_id, candidate_json, "
                "status, task_id, reason, result_json, created_at, resolved_at, "
                "expires_at, escalated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row.get("ticket_id", f"tkt_{i}"),
                    row["project_id"],
                    candidate,
                    row.get("status", "created"),
                    None if task_id is None else str(task_id),
                    None,
                    None,
                    "2026-08-01T00:00:00+00:00",
                    None,
                    "2026-09-01T00:00:00+00:00",
                    None,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def test_tickets_db_path_convention(tmp_path):
    """<root>/data/reconciliation/tickets.db."""
    assert tickets_db_path(str(tmp_path)) == tmp_path / "data" / "reconciliation" / "tickets.db"


def test_load_ticket_expectations_returns_submit_metadata_keyed_by_task_id(tmp_path):
    """The submit payload is candidate_json['metadata'], keyed by task id."""
    db = _make_tickets_db(tmp_path, [
        {"project_id": "dark_factory", "task_id": 3157,
         "metadata": {"source": "prd", "task_kind": "deterministic"}},
    ])

    expectations = load_ticket_expectations(str(db), "dark_factory")

    assert expectations == {"3157": {"source": "prd", "task_kind": "deterministic"}}


def test_load_ticket_expectations_keys_are_strings(tmp_path):
    """task_id is a TEXT column: keys are str so an int task id joins correctly."""
    db = _make_tickets_db(tmp_path, [
        {"project_id": "dark_factory", "task_id": 42, "metadata": {"source": "prd"}},
    ])

    expectations = load_ticket_expectations(str(db), "dark_factory")

    assert list(expectations) == ["42"]
    assert 42 not in expectations


def test_load_ticket_expectations_never_imports_another_projects_payload(tmp_path):
    """TRAP (a), MEASURED LIVE: a `reify` created-row carries task_id='3157'
    sitting right next to a `dark_factory` row for the same id. A task_id-only
    query would silently import another project's submit payload and then
    report every dark_factory key it lacks as LOST. The project_id predicate
    is what stops that, so it is pinned here rather than left to review."""
    db = _make_tickets_db(tmp_path, [
        {"project_id": "reify", "task_id": 3157,
         "metadata": {"reify_only_marker": "NEVER-CROSS-PROJECTS"}},
        {"project_id": "dark_factory", "task_id": 3157,
         "metadata": {"source": "prd"}},
    ])

    expectations = load_ticket_expectations(str(db), "dark_factory")

    assert expectations == {"3157": {"source": "prd"}}
    assert "reify_only_marker" not in json.dumps(expectations)


def test_load_ticket_expectations_isolates_a_foreign_only_task_id(tmp_path):
    """The same collision with NO dark_factory row at all: the reify payload
    must not appear under 3157 either — it must simply be absent."""
    db = _make_tickets_db(tmp_path, [
        {"project_id": "reify", "task_id": 3157,
         "metadata": {"reify_only_marker": "NEVER-CROSS-PROJECTS"}},
    ])

    assert load_ticket_expectations(str(db), "dark_factory") == {}


def test_load_ticket_expectations_ignores_non_created_status(tmp_path):
    """Only status='created' rows evidence a task that was actually filed."""
    db = _make_tickets_db(tmp_path, [
        {"project_id": "dark_factory", "task_id": 1, "status": "pending",
         "metadata": {"source": "prd"}},
        {"project_id": "dark_factory", "task_id": 2, "status": "combined",
         "metadata": {"source": "prd"}},
        {"project_id": "dark_factory", "task_id": 3, "status": "dropped",
         "metadata": {"source": "prd"}},
        {"project_id": "dark_factory", "task_id": 4, "status": "created",
         "metadata": {"source": "prd"}},
    ])

    assert list(load_ticket_expectations(str(db), "dark_factory")) == ["4"]


def test_load_ticket_expectations_skips_null_task_id(tmp_path):
    """A created-row with no task_id cannot be joined — skipped, not raised."""
    db = _make_tickets_db(tmp_path, [
        {"project_id": "dark_factory", "task_id": None, "metadata": {"source": "prd"}},
        {"project_id": "dark_factory", "task_id": 9, "metadata": {"source": "prd"}},
    ])

    assert list(load_ticket_expectations(str(db), "dark_factory")) == ["9"]


def test_load_ticket_expectations_skips_malformed_candidate_json(tmp_path):
    """One corrupt candidate_json must not abort the sweep."""
    db = _make_tickets_db(tmp_path, [
        {"project_id": "dark_factory", "task_id": 1, "candidate_json": "{not json"},
        {"project_id": "dark_factory", "task_id": 2, "candidate_json": "[1, 2, 3]"},
        {"project_id": "dark_factory", "task_id": 3, "candidate_json": '"scalar"'},
        {"project_id": "dark_factory", "task_id": 4, "metadata": {"source": "prd"}},
    ])

    assert list(load_ticket_expectations(str(db), "dark_factory")) == ["4"]


def test_load_ticket_expectations_skips_absent_or_wrong_typed_metadata(tmp_path):
    """A candidate whose 'metadata' is missing or not a dict yields no expectation."""
    db = _make_tickets_db(tmp_path, [
        {"project_id": "dark_factory", "task_id": 1,
         "candidate_json": {"kwargs": {}, "project_root": "/x"}},
        {"project_id": "dark_factory", "task_id": 2,
         "candidate_json": {"kwargs": {}, "metadata": ["source"], "project_root": "/x"}},
        {"project_id": "dark_factory", "task_id": 3,
         "candidate_json": {"kwargs": {}, "metadata": None, "project_root": "/x"}},
        {"project_id": "dark_factory", "task_id": 4, "metadata": {"source": "prd"}},
    ])

    assert list(load_ticket_expectations(str(db), "dark_factory")) == ["4"]


def test_load_ticket_expectations_absent_db_returns_empty(tmp_path):
    """A project with no tickets.db is honest 'no comparison source', not a crash."""
    assert load_ticket_expectations(str(tmp_path / "nope.db"), "dark_factory") == {}
