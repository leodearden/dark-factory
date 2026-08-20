"""Tests for scripts/census_tagger_debris.py — the READ-ONLY tagger-debris census.

PRD task epsilon of plans/module-tagger-retirement-prd.md (decision 3). The
census enumerates every task record carrying ``metadata.files_tagged_at`` — the
stamp the retired module tagger left behind — across all six project corpora,
and classifies each on three axes so DF 3113 P4a and DF 3427 can consume a
machine-readable candidate set instead of a prose claim.

NO TEST HERE ASSERTS A COUNT OR TASK ID DERIVED FROM THE LIVE DATABASES.
This norm is inherited verbatim from both sibling suites —
scripts/tests/test_audit_wiped_metadata_files.py and
tests/scripts/test_repair_wiped_metadata_files.py:11-21, the latter citing a
candidate count that moved 40 -> 43 -> 45 across a single task's planning
sessions with one id changing signature class in between. The six corpora are
mutated continuously by six running orchestrators, so a test pinning "the live
DB yields N stamped records" would be a guessed threshold that goes red the
moment any task merges. Every assertion below runs against synthetic tuples or
synthetic tmp_path databases whose contents the test controls exactly.

The one place the required POSITIVE CONTROLS (reify 6068/5602/5632,
dark_factory 3113) are asserted is against the COMMITTED ARTIFACT — a static
repo file, not a live database — which is stable under corpus drift and is
precisely the task's user-observable signal.

Mirrors the sibling split: pure functions get direct pytest coverage;
``main()`` gets subprocess coverage.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from census_tagger_debris import (
    NEVER_RECONCILED,
    NO_PRIOR_SCOPE,
    POST_WIPE_OVERWRITE,
    RECONCILED,
    STATUS_NON_TERMINAL,
    STATUS_TERMINAL,
    ScopeEvent,
    _connect_readonly,
    classify_record,
    load_stamped_records,
)

# ---------------------------------------------------------------------------
# The classification vocabulary and the pure three-axis classifier.
#
# Every input below is a hand-built tuple. Timestamps are ISO-8601 strings with
# a timezone, the shape measured in BOTH live columns the census compares:
# events.timestamp and metadata.files_tagged_at. The comparison is a plain
# string compare, which is total and correct for same-offset ISO-8601 — the
# stamp and the events are written by the same process family.
# ---------------------------------------------------------------------------

_STAMP = "2026-08-08T01:04:58+00:00"
_BEFORE = "2026-08-01T00:00:00+00:00"
_AFTER = "2026-08-15T00:00:00+00:00"


def _event(timestamp: str, event_type: str = "set_to_plan", event_id: int = 1) -> ScopeEvent:
    return ScopeEvent(
        timestamp=timestamp,
        event_type=event_type,
        event_id=event_id,
        fidelity="lock_level",
        file_count=2,
    )


def test_classification_vocabulary_constants_have_exact_string_values():
    """(a) The six labels are the artifact's public vocabulary.

    DF 3113 P4a and DF 3427 will read these strings out of the committed JSON,
    so a rename is a breaking change to a consumer that cannot see this repo's
    constants. Pinning the literals here makes that breakage a failing test
    rather than a silently-unjoinable artifact.
    """
    assert STATUS_TERMINAL == "terminal"
    assert STATUS_NON_TERMINAL == "non_terminal"
    assert RECONCILED == "plan_reconciled"
    assert NEVER_RECONCILED == "never_reconciled"
    assert POST_WIPE_OVERWRITE == "post_wipe_overwrite"
    assert NO_PRIOR_SCOPE == "no_prior_scope"


@pytest.mark.parametrize("status", ["done", "cancelled"])
def test_terminal_statuses_classify_terminal(status):
    """(b) The terminal axis is the repair's own allowlist, not a re-spelling."""
    result = classify_record(_STAMP, status, [])
    assert result.status_class == STATUS_TERMINAL


@pytest.mark.parametrize(
    "status", ["pending", "in-progress", "blocked", "deferred", "merge-deferred"]
)
def test_every_other_status_classifies_non_terminal(status):
    """(b) An ALLOWLIST, so a status the system grows later falls on the
    non_terminal side — reported as a live victim rather than silently
    excluded from the population the census exists to find."""
    result = classify_record(_STAMP, status, [])
    assert result.status_class == STATUS_NON_TERMINAL


def test_scope_event_after_the_stamp_is_plan_reconciled():
    """(c) A scope event postdating the stamp means the tagger's guess was
    superseded by a real derivation — the record is no longer a live victim."""
    result = classify_record(_STAMP, "pending", [_event(_AFTER)])
    assert result.reconciliation == RECONCILED
    assert result.wipe_signature == NO_PRIOR_SCOPE


def test_scope_event_before_the_stamp_is_post_wipe_overwrite():
    """(c) A scope event predating the stamp means an authoritative scope
    EXISTED and the tagger stamped over it — the damaging case."""
    result = classify_record(_STAMP, "pending", [_event(_BEFORE)])
    assert result.wipe_signature == POST_WIPE_OVERWRITE
    assert result.reconciliation == NEVER_RECONCILED


def test_events_on_both_sides_of_the_stamp_yield_both_classifications():
    """(c) The two axes are INDEPENDENT: a record can have been stamped over a
    prior scope AND later reconciled. Collapsing them to one label would lose
    exactly the distinction the repair needs."""
    result = classify_record(
        _STAMP, "pending", [_event(_BEFORE, event_id=1), _event(_AFTER, event_id=2)]
    )
    assert result.reconciliation == RECONCILED
    assert result.wipe_signature == POST_WIPE_OVERWRITE


def test_no_scope_events_at_all_is_never_reconciled_and_no_prior_scope():
    """(c) The live-victim cell: the tagger's guess is still the only scope
    this record has ever had."""
    result = classify_record(_STAMP, "pending", [])
    assert result.reconciliation == NEVER_RECONCILED
    assert result.wipe_signature == NO_PRIOR_SCOPE


def test_event_exactly_at_the_stamp_decides_neither_axis():
    """(d) THE BOUNDARY, pinned explicitly rather than left to inference.

    Comparison is strict (``>`` / ``<``), so an event bearing the same instant
    as the stamp is evidence of neither reconciliation nor overwrite. The two
    writes are not ordered with respect to each other at equal timestamps, and
    inventing an order would be a guess presented as a measurement.
    """
    result = classify_record(_STAMP, "pending", [_event(_STAMP)])
    assert result.reconciliation == NEVER_RECONCILED
    assert result.wipe_signature == NO_PRIOR_SCOPE


def test_reconciliation_evidence_names_the_deciding_event():
    """(e) INV-2: no classification is a prose-only claim.

    The EARLIEST post-stamp event is the deciding one — the first thing that
    superseded the tagger's guess.
    """
    events = [
        _event("2026-08-20T00:00:00+00:00", event_type="phase_skipped", event_id=9),
        _event("2026-08-10T00:00:00+00:00", event_type="set_to_plan", event_id=4),
    ]
    result = classify_record(_STAMP, "pending", events)

    assert result.reconciliation == RECONCILED
    assert result.reconciled_by.event_type == "set_to_plan"
    assert result.reconciled_by.event_id == 4
    assert result.reconciled_by.timestamp == "2026-08-10T00:00:00+00:00"


def test_overwrite_evidence_names_the_latest_prior_scope_event():
    """(e) The LATEST pre-stamp event is the deciding one — the most recent
    authoritative scope that the tagger's stamp wrote over."""
    events = [
        _event("2026-07-01T00:00:00+00:00", event_type="set_to_plan", event_id=2),
        _event("2026-08-07T00:00:00+00:00", event_type="phase_skipped", event_id=7),
    ]
    result = classify_record(_STAMP, "done", events)

    assert result.wipe_signature == POST_WIPE_OVERWRITE
    assert result.preceded_by.event_type == "phase_skipped"
    assert result.preceded_by.event_id == 7
    assert result.preceded_by.timestamp == "2026-08-07T00:00:00+00:00"


def test_absent_evidence_is_explicitly_null_not_a_missing_key():
    """(e) An unclassified axis still carries its evidence keys, all None. A
    MISSING key in the artifact would be indistinguishable from a serializer
    bug; a present null says "looked, found nothing"."""
    result = classify_record(_STAMP, "pending", [])

    assert result.reconciled_by._asdict() == {
        "event_type": None,
        "event_id": None,
        "timestamp": None,
    }
    assert result.preceded_by._asdict() == {
        "event_type": None,
        "event_id": None,
        "timestamp": None,
    }


# ---------------------------------------------------------------------------
# Synthetic corpus fixtures.
#
# Copied in SHAPE from tests/scripts/test_repair_wiped_metadata_files.py:946-1051
# rather than imported: scripts/tests/conftest.py's ``make_tasks_db`` fixture
# does not reach this directory (tests/scripts/conftest.py is sys.path wiring
# only), and the two test directories cannot share imports for the same reason
# recorded at orchestrator/tests/test_deterministic_runner.py:31-32. The schemas
# mirror the live ones so the census exercises real column shapes.
#
# Two deliberate departures from that sibling:
#   * ``_make_runs_db`` honours an explicit per-event ``timestamp``, falling
#     back to the sibling's ascending default. Every census classification is a
#     timestamp comparison, so a test that cannot place an event on a chosen
#     side of the stamp cannot test this module at all.
#   * ``_make_project`` takes no *plans* argument: unlike the audit, the census
#     never reads on-disk plan artifacts. It gains ``with_runs_db`` instead, so
#     the missing-event-log degradation path has a fixture.
#
# Every fixture below is a SYNTHETIC temp project root. Not one test points at
# a live corpus: those databases mutate continuously, so an assertion derived
# from them would be a guessed threshold.
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


def _make_tasks_db(dir_path: Path, rows: list[dict]) -> Path:
    db_path = dir_path / "tasks.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_TASKS_SCHEMA)
        for row in rows:
            metadata = row.get("metadata")
            # A raw string passes through UNCONVERTED so a test can inject
            # malformed JSON; anything else is encoded.
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
                    None,
                    None,
                    None,
                    row.get("status", "done"),
                    "medium",
                    metadata,
                    "2026-08-01T00:00:00+00:00",
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _make_runs_db(dir_path: Path, events: list[dict]) -> Path:
    db_path = dir_path / "runs.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_EVENTS_SCHEMA)
        for i, event in enumerate(events):
            data = event.get("data")
            if data is not None and not isinstance(data, str):
                data = json.dumps(data)
            conn.execute(
                "INSERT INTO events (timestamp, run_id, task_id, event_type, "
                "phase, role, data, cost_usd, duration_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.get("timestamp", f"2026-08-01T00:00:{i:02d}+00:00"),
                    "run-1",
                    None if event.get("task_id") is None else str(event["task_id"]),
                    event["event_type"],
                    None,
                    None,
                    data,
                    None,
                    None,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _make_project(tmp_path, tasks=(), events=(), name="proj", with_runs_db=True) -> Path:
    """Build a synthetic project root with the two inputs census_project reads."""
    root = tmp_path / name
    tasks_dir = root / ".taskmaster" / "tasks"
    tasks_dir.mkdir(parents=True)
    _make_tasks_db(tasks_dir, list(tasks))

    if with_runs_db:
        runs_dir = root / "data" / "orchestrator"
        runs_dir.mkdir(parents=True)
        _make_runs_db(runs_dir, list(events))
    return root


# --- fixture-builder self-checks -------------------------------------------
#
# A broken builder must surface AS ITSELF rather than as a bogus failure in the
# code under test. Same reason the sibling suite carries these.


def test_make_tasks_db_roundtrips_the_rows_it_was_given(tmp_path):
    db_path = _make_tasks_db(
        tmp_path,
        [
            {"id": 7, "status": "pending", "metadata": {"files_tagged_at": _STAMP}},
            {"id": 8, "tag": "other", "status": "done", "metadata": None},
        ],
    )
    conn = sqlite3.connect(db_path)
    try:
        rows = sorted(conn.execute("SELECT tag, id, status, metadata FROM tasks"))
    finally:
        conn.close()

    assert rows == [
        ("master", 7, "pending", json.dumps({"files_tagged_at": _STAMP})),
        ("other", 8, "done", None),
    ]


def test_make_runs_db_assigns_ascending_ids_in_list_order(tmp_path):
    """Event ID order IS emission order throughout this module — the loaders
    read ``ORDER BY id``. A builder that reordered rows would silently invert
    every ordering assertion downstream."""
    db_path = _make_runs_db(
        tmp_path,
        [
            {"event_type": "set_to_plan", "task_id": 1, "data": {"files": ["a.py"]}},
            {"event_type": "phase_skipped", "task_id": 1, "data": {"plan_files": ["b.py"]}},
            {"event_type": "lock_acquired", "task_id": 2, "data": {"modules": ["c/"]}},
        ],
    )
    conn = sqlite3.connect(db_path)
    try:
        rows = list(conn.execute("SELECT id, event_type FROM events ORDER BY id"))
    finally:
        conn.close()

    assert rows == [(1, "set_to_plan"), (2, "phase_skipped"), (3, "lock_acquired")]


def test_make_runs_db_honours_an_explicit_timestamp(tmp_path):
    db_path = _make_runs_db(
        tmp_path,
        [{"event_type": "set_to_plan", "task_id": 1, "timestamp": _AFTER, "data": {}}],
    )
    conn = sqlite3.connect(db_path)
    try:
        assert list(conn.execute("SELECT timestamp FROM events")) == [(_AFTER,)]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# load_stamped_records — the population the census exists to enumerate.
# ---------------------------------------------------------------------------


def test_load_stamped_records_returns_only_rows_carrying_the_stamp(tmp_path):
    db_path = _make_tasks_db(
        tmp_path,
        [
            {"id": 1, "metadata": {"files_tagged_at": _STAMP, "files": ["a.py"]}},
            {"id": 2, "metadata": {"files": ["b.py"]}},
        ],
    )
    records = load_stamped_records(str(db_path))

    assert set(records) == {("master", 1)}


@pytest.mark.parametrize(
    ("label", "metadata"),
    [
        ("null metadata", None),
        ("malformed json", "{not json"),
        ("non-dict payload", "[1, 2, 3]"),
        ("absent stamp", {"files": ["a.py"]}),
        ("empty-string stamp", {"files_tagged_at": ""}),
        ("null stamp", {"files_tagged_at": None}),
    ],
)
def test_load_stamped_records_skips_unusable_rows_without_raising(tmp_path, label, metadata):
    """One corrupt row must never abort a 12,000-row sweep. Each of these is
    reported as "not stamped" rather than as a crash — the same defensive
    posture audit_wiped_metadata_files._decode_files takes."""
    db_path = _make_tasks_db(tmp_path, [{"id": 1, "metadata": metadata}])

    assert load_stamped_records(str(db_path)) == {}, label


def test_load_stamped_records_carries_the_records_current_scope(tmp_path):
    db_path = _make_tasks_db(
        tmp_path,
        [
            {
                "id": 42,
                "tag": "master",
                "status": "pending",
                "metadata": {"files_tagged_at": _STAMP, "files": ["x/y.py", "z.py"]},
            }
        ],
    )
    record = load_stamped_records(str(db_path))[("master", 42)]

    assert record.tag == "master"
    assert record.task_id == 42
    assert record.status == "pending"
    assert record.files_tagged_at == _STAMP
    assert record.metadata_files == ("x/y.py", "z.py")


@pytest.mark.parametrize(
    ("label", "files"),
    [
        ("empty list, the reify-5632 shape", []),
        ("wrong-typed files", "not-a-list"),
        ("absent files", None),
    ],
)
def test_stamped_record_scope_degrades_to_empty_never_raises(tmp_path, label, files):
    """A stamped record whose ``files`` is empty or wrong-typed still appears —
    it is exactly the population under census. Coercion goes through the
    audit's own ``_coerce_file_list`` so all three scripts agree byte-for-byte
    about what "this record's current scope" means."""
    metadata = {"files_tagged_at": _STAMP}
    if files is not None:
        metadata["files"] = files
    db_path = _make_tasks_db(tmp_path, [{"id": 1, "metadata": metadata}])

    record = load_stamped_records(str(db_path))[("master", 1)]
    assert record.metadata_files == (), label


def test_load_stamped_records_keys_on_the_full_tag_and_id_primary_key(tmp_path):
    """The live corpora use a single ``master`` tag, but the schema PERMITS the
    same numeric id under two tags and collapsing them would silently merge two
    distinct tasks — the same reason load_task_records keys this way."""
    db_path = _make_tasks_db(
        tmp_path,
        [
            {"id": 5, "tag": "master", "metadata": {"files_tagged_at": _STAMP}},
            {"id": 5, "tag": "feature", "metadata": {"files_tagged_at": _AFTER}},
        ],
    )
    records = load_stamped_records(str(db_path))

    assert set(records) == {("master", 5), ("feature", 5)}
    assert records[("feature", 5)].files_tagged_at == _AFTER


def test_the_censuss_own_opener_cannot_write(tmp_path):
    """THE SAFETY PROPERTY. Every corpus connection this module opens is a
    mode=ro URI, so the sweep is structurally incapable of mutating a live task
    record even while fused-memory holds the same file open in WAL mode. Proven
    against the opener itself, not asserted in prose."""
    db_path = _make_tasks_db(tmp_path, [{"id": 1, "metadata": {"files_tagged_at": _STAMP}}])

    conn = _connect_readonly(str(db_path))
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("UPDATE tasks SET status = 'cancelled'")
    finally:
        conn.close()
