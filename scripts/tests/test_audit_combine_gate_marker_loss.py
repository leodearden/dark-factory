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

from audit_combine_gate_marker_loss import (
    CombineTarget,
    load_combine_targets,
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
