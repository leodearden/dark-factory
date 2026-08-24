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
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from _task_db_scan import (
    AUDIT_EXIT_NO_ROOT,
    AUDIT_EXIT_NOTHING_AUDITED,
    AUDIT_EXIT_OK,
)
from audit_wiped_metadata_files import (
    _EVENT_PLAN_SOURCES,
    CONFIRMED_NULL_SHA_DONE_PATH,
    FIDELITY_FILE_LEVEL,
    FIDELITY_LOCK_LEVEL,
    NO_MERGE_EVENT,
)
from census_tagger_debris import (
    _RECONCILIATIONS,
    DEFAULT_JSON_OUT,
    DEFAULT_MD_OUT,
    EXIT_NO_ROOT,
    EXIT_NOTHING_SCANNED,
    EXIT_OK,
    EXIT_STALE,
    LOCK_RECONCILED,
    NEVER_RECONCILED,
    NO_PRIOR_SCOPE,
    POST_WIPE_OVERWRITE,
    RECONCILED,
    SCHEMA_VERSION,
    SCOPE_EVENT_SOURCES,
    STATUS_NON_TERMINAL,
    STATUS_TERMINAL,
    ScopeEvent,
    _connect_readonly,
    _module_is_explained_by,
    _record_to_dict,
    build_report,
    census_project,
    classify_record,
    load_scope_events,
    load_stamped_records,
    render_markdown,
    write_artifacts,
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


# The record's OWN metadata.files — the tagger's surviving guess. Every
# axis-2 echo test below is a question about whether some event's paths are
# merely a re-derivation of THIS list.
_GUESS = ("scripts/census_tagger_debris.py", "tests/scripts/test_census_tagger_debris.py")


def _event(
    timestamp: str,
    event_type: str = "set_to_plan",
    event_id: int = 1,
    files: tuple[str, ...] = ("mod/a.py", "mod/b.py"),
    fidelity: str = FIDELITY_LOCK_LEVEL,
) -> ScopeEvent:
    """A PLAN-level scope event by default (set_to_plan), never a lock.

    ``file_count`` is derived from *files* here for the same reason the loader
    derives it there: two fields describing one list must not be able to drift.
    """
    return ScopeEvent(
        timestamp=timestamp,
        event_type=event_type,
        event_id=event_id,
        fidelity=fidelity,
        file_count=len(files),
        files=files,
    )


def _lock(timestamp: str, modules: tuple[str, ...], event_id: int = 1) -> ScopeEvent:
    """A lock_acquired scope event carrying *modules* — the echo-test subject."""
    return _event(
        timestamp,
        event_type="lock_acquired",
        event_id=event_id,
        files=modules,
        fidelity=FIDELITY_LOCK_LEVEL,
    )


def test_classification_vocabulary_constants_have_exact_string_values():
    """(a) The seven labels are the artifact's public vocabulary.

    DF 3113 P4a and DF 3427 will read these strings out of the committed JSON,
    so a rename is a breaking change to a consumer that cannot see this repo's
    constants. Pinning the literals here makes that breakage a failing test
    rather than a silently-unjoinable artifact.
    """
    assert STATUS_TERMINAL == "terminal"
    assert STATUS_NON_TERMINAL == "non_terminal"
    assert RECONCILED == "plan_reconciled"
    assert LOCK_RECONCILED == "lock_reconciled"
    assert NEVER_RECONCILED == "never_reconciled"
    assert POST_WIPE_OVERWRITE == "post_wipe_overwrite"
    assert NO_PRIOR_SCOPE == "no_prior_scope"


@pytest.mark.parametrize("status", ["done", "cancelled"])
def test_terminal_statuses_classify_terminal(status):
    """(b) The terminal axis is the repair's own allowlist, not a re-spelling."""
    result = classify_record(_STAMP, status, [], metadata_files=())
    assert result.status_class == STATUS_TERMINAL


@pytest.mark.parametrize(
    "status", ["pending", "in-progress", "blocked", "deferred", "merge-deferred"]
)
def test_every_other_status_classifies_non_terminal(status):
    """(b) An ALLOWLIST, so a status the system grows later falls on the
    non_terminal side — reported as a live victim rather than silently
    excluded from the population the census exists to find."""
    result = classify_record(_STAMP, status, [], metadata_files=())
    assert result.status_class == STATUS_NON_TERMINAL


def test_scope_event_after_the_stamp_is_plan_reconciled():
    """(c) A scope event postdating the stamp means the tagger's guess was
    superseded by a real derivation — the record is no longer a live victim."""
    result = classify_record(_STAMP, "pending", [_event(_AFTER)], metadata_files=())
    assert result.reconciliation == RECONCILED
    assert result.wipe_signature == NO_PRIOR_SCOPE


def test_scope_event_before_the_stamp_is_post_wipe_overwrite():
    """(c) A scope event predating the stamp means an authoritative scope
    EXISTED and the tagger stamped over it — the damaging case."""
    result = classify_record(_STAMP, "pending", [_event(_BEFORE)], metadata_files=())
    assert result.wipe_signature == POST_WIPE_OVERWRITE
    assert result.reconciliation == NEVER_RECONCILED


def test_events_on_both_sides_of_the_stamp_yield_both_classifications():
    """(c) The two axes are INDEPENDENT: a record can have been stamped over a
    prior scope AND later reconciled. Collapsing them to one label would lose
    exactly the distinction the repair needs."""
    result = classify_record(
        _STAMP,
        "pending",
        [_event(_BEFORE, event_id=1), _event(_AFTER, event_id=2)],
        metadata_files=(),
    )
    assert result.reconciliation == RECONCILED
    assert result.wipe_signature == POST_WIPE_OVERWRITE


def test_no_scope_events_at_all_is_never_reconciled_and_no_prior_scope():
    """(c) The live-victim cell: the tagger's guess is still the only scope
    this record has ever had."""
    result = classify_record(_STAMP, "pending", [], metadata_files=())
    assert result.reconciliation == NEVER_RECONCILED
    assert result.wipe_signature == NO_PRIOR_SCOPE


def test_event_exactly_at_the_stamp_decides_neither_axis():
    """(d) THE BOUNDARY, pinned explicitly rather than left to inference.

    Comparison is strict (``>`` / ``<``), so an event bearing the same instant
    as the stamp is evidence of neither reconciliation nor overwrite. The two
    writes are not ordered with respect to each other at equal timestamps, and
    inventing an order would be a guess presented as a measurement.
    """
    result = classify_record(_STAMP, "pending", [_event(_STAMP)], metadata_files=())
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
    result = classify_record(_STAMP, "pending", events, metadata_files=())

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
    result = classify_record(_STAMP, "done", events, metadata_files=())

    assert result.wipe_signature == POST_WIPE_OVERWRITE
    assert result.preceded_by.event_type == "phase_skipped"
    assert result.preceded_by.event_id == 7
    assert result.preceded_by.timestamp == "2026-08-07T00:00:00+00:00"


def test_absent_evidence_is_explicitly_null_not_a_missing_key():
    """(e) An unclassified axis still carries its evidence keys, all None. A
    MISSING key in the artifact would be indistinguishable from a serializer
    bug; a present null says "looked, found nothing"."""
    result = classify_record(_STAMP, "pending", [], metadata_files=())

    assert result.reconciled_by._asdict() == {
        "event_type": None,
        "event_id": None,
        "timestamp": None,
        "fidelity": None,
    }
    assert result.preceded_by._asdict() == {
        "event_type": None,
        "event_id": None,
        "timestamp": None,
        "fidelity": None,
    }


# ---------------------------------------------------------------------------
# Axis 2, CORRECTED (review fix, 2026-08-24).
#
# THE DEFECT THESE TESTS PIN SHUT. v1 treated ANY post-stamp lock_acquired as
# proof that a real derivation superseded the tagger's guess. It is not:
# Scheduler._get_modules (orchestrator/src/orchestrator/scheduler.py:8797-8801)
# computes the locked module set as derive_modules(metadata["files"], depth) —
# straight from metadata.files, which for a never-reconciled tagger-stamped
# record IS the guess. A post-stamp lock is therefore an ECHO of the guess, not
# evidence against it, and v1 was reporting the majority of live victims to
# DF 3113 P4a / DF 3427 as already repaired.
#
# Every input below is synthetic. The echo test is depth-INVARIANT by
# construction: files_to_modules truncates each path to `depth` components, so
# every module derive_modules can emit is a component-wise PREFIX of some entry
# in files. Equality at one depth would have been wrong — lock_depth is
# 12/10/4/4/3/unset(2) across the six corpora and CHANGED mid-tagger-era
# (dark-factory 4->12, reify 4->10).
# ---------------------------------------------------------------------------


def test_lock_reconciled_is_a_third_axis_2_value_beside_the_two_v1_labels():
    """(a) The two v1 labels KEEP their spellings — DF 3113 P4a and DF 3427
    join on these strings out of the committed JSON, so renaming either would
    break a consumer that cannot see this module's constants. The new class is
    additive."""
    assert LOCK_RECONCILED == "lock_reconciled"
    assert RECONCILED == "plan_reconciled"
    assert NEVER_RECONCILED == "never_reconciled"


def test_the_report_vocabulary_enumerates_all_three_reconciliation_values():
    """(a) build_report iterates THIS tuple to emit its cells, so a value
    missing here becomes a silently absent count rather than an explicit zero."""
    assert set(_RECONCILIATIONS) == {RECONCILED, LOCK_RECONCILED, NEVER_RECONCILED}
    assert len(_RECONCILIATIONS) == 3


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        ("a", True),
        ("a/b", True),
        ("a/b/c.py", True),
        ("a/b/", True),
        ("b", False),
        ("a/c", False),
        ("a/b/c.py/d", False),
        ("", False),
    ],
)
def test_module_prefix_matching_is_component_wise(module, expected):
    """The helper is a PATH-prefix test on '/'-split components."""
    assert _module_is_explained_by(module, ("a/b/c.py",)) is expected


def test_a_raw_string_prefix_is_not_a_path_prefix():
    """'a/bcd/e.py'.startswith('a/b') is True and MEANINGLESS. A str.startswith
    implementation would call an unrelated module an echo and silently downgrade
    a genuine reconciliation to never_reconciled."""
    assert _module_is_explained_by("a/b", ("a/bcd/e.py",)) is False


def test_a_post_stamp_lock_that_merely_echoes_the_guess_is_not_reconciliation():
    """(b) THE MANDATED PIN — modules exactly equal to the record's own files.

    The scheduler derived them FROM metadata.files, so they assert nothing the
    tagger did not already assert. The record is still carrying the guess.
    """
    result = classify_record(_STAMP, "pending", [_lock(_AFTER, _GUESS)], _GUESS)

    assert result.reconciliation == NEVER_RECONCILED
    assert result.reconciled_by._asdict() == {
        "event_type": None,
        "event_id": None,
        "timestamp": None,
        "fidelity": None,
    }


def test_a_post_stamp_lock_truncated_to_a_shallower_depth_is_still_an_echo():
    """(b) The second mandated shape: derive_modules TRUNCATES, so the modules
    are shorter than the files they came from and never equal to them."""
    files = ("a/b/c/d.py",)
    result = classify_record(_STAMP, "pending", [_lock(_AFTER, ("a/b",))], files)

    assert result.reconciliation == NEVER_RECONCILED


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_the_echo_test_holds_at_every_derive_depth(depth):
    """(c) DEPTH INDEPENDENCE — what keeps no lock_depth value baked in.

    lock_depth is 12/10/4/4/3/unset(2) across the six corpora and CHANGED
    mid-tagger-era, so a test pinning equality at one depth would regress
    silently the next time an operator retunes it.
    """
    files = ("orchestrator/src/orchestrator/scheduler.py",)
    module = "/".join(files[0].split("/")[:depth])
    result = classify_record(_STAMP, "pending", [_lock(_AFTER, (module,))], files)

    assert result.reconciliation == NEVER_RECONCILED
    assert result.reconciled_by.event_type is None


def test_a_post_stamp_lock_with_an_unexplained_module_is_lock_reconciled():
    """(d) A lock naming something the record's own files CANNOT explain did
    not come from re-deriving them, so it is real evidence — but it gets its
    OWN label, never plan_reconciled, because a consumer must be able to filter
    the weaker class out."""
    lock = _lock(_AFTER, ("shared/", *_GUESS), event_id=12)
    result = classify_record(_STAMP, "pending", [lock], _GUESS)

    assert result.reconciliation == LOCK_RECONCILED
    assert result.reconciled_by.event_type == "lock_acquired"
    assert result.reconciled_by.event_id == 12
    assert result.reconciled_by.timestamp == _AFTER


@pytest.mark.parametrize("event_type", ["set_to_plan", "phase_skipped"])
def test_a_plan_event_is_never_echo_filtered(event_type):
    """(e) A plan event is a genuine plan-derived ASSERTION, not a
    re-derivation of metadata.files, so it counts even when its file list is
    identical to the guess. Echo-filtering it would erase the one signal the
    audit's own lens already trusts."""
    events = [_event(_AFTER, event_type=event_type, event_id=3, files=_GUESS)]
    result = classify_record(_STAMP, "pending", events, _GUESS)

    assert result.reconciliation == RECONCILED
    assert result.reconciled_by.event_type == event_type


def test_a_plan_event_outranks_a_genuine_lock_even_when_the_lock_came_first():
    """(e) PRECEDENCE. Axis 2 is not "the earliest post-stamp event" any more:
    the plan-level assertion is the stronger signal and must never be masked by
    a weaker lock that merely happens to be older."""
    lock = _lock("2026-08-09T00:00:00+00:00", ("shared/",), event_id=5)
    plan = _event(
        "2026-08-20T00:00:00+00:00", event_type="phase_skipped", event_id=6, files=("x.py",)
    )
    result = classify_record(_STAMP, "pending", [lock, plan], _GUESS)

    assert result.reconciliation == RECONCILED
    assert result.reconciled_by.event_type == "phase_skipped"
    assert result.reconciled_by.event_id == 6


def test_with_no_guess_on_the_record_a_lock_cannot_be_an_echo():
    """(f) THE reify-5632 SHAPE (metadata.files == []). Nothing can explain the
    lock's modules, so it is real evidence. Pinned explicitly so an empty guess
    can never divide-by-zero into a false echo."""
    result = classify_record(_STAMP, "done", [_lock(_AFTER, ("scripts/",))], ())

    assert result.reconciliation == LOCK_RECONCILED
    assert result.reconciled_by.event_type == "lock_acquired"


def test_a_pre_stamp_lock_echoing_the_guess_is_still_a_post_wipe_overwrite():
    """(g) THE ASYMMETRY IS DELIBERATE, and this test is what stops a later
    contributor from "consistently" applying the echo filter to axis 3 too.

    A PRE-stamp lock proves a file-derived scope existed BEFORE the tagger
    stamped — it cannot be an echo of a guess that did not yet exist. Filtering
    it would erase exactly the wipe signal the census exists to surface.
    """
    result = classify_record(_STAMP, "pending", [_lock(_BEFORE, _GUESS)], _GUESS)

    assert result.wipe_signature == POST_WIPE_OVERWRITE
    assert result.preceded_by.event_type == "lock_acquired"
    assert result.preceded_by.timestamp == _BEFORE
    assert result.reconciliation == NEVER_RECONCILED


def test_evidence_carries_the_deciding_events_fidelity_on_both_axes():
    """(h) A consumer must be able to tell a file_level assertion from a
    lock_level one WITHOUT re-deriving it — the two are not interchangeable
    (a module path must never be written back as a plan.files entry)."""
    plan = _event(
        _AFTER,
        event_type="phase_skipped",
        event_id=2,
        files=("a.py",),
        fidelity=FIDELITY_FILE_LEVEL,
    )
    prior = _lock(_BEFORE, ("shared/",), event_id=1)
    result = classify_record(_STAMP, "pending", [plan, prior], _GUESS)

    assert result.reconciled_by.fidelity == FIDELITY_FILE_LEVEL
    assert result.preceded_by.fidelity == FIDELITY_LOCK_LEVEL


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


# ---------------------------------------------------------------------------
# load_scope_events — the timestamped event lens.
#
# This is the one place the census does NOT call the audit's own reader.
# load_plan_files_from_events collapses each task to a single highest-fidelity
# record and never selects ``timestamp``, and every census classification is a
# comparison against that timestamp. The LENS DEFINITION — which event types
# carry a scope, under which payload key, at what fidelity — nonetheless stays
# single-sourced in the audit; only the query differs. The first test below is
# what holds that line.
# ---------------------------------------------------------------------------


def test_scope_event_sources_are_derived_from_the_audits_table_not_re_spelled():
    """INV-5: the census must expose no SECOND COPY of the event lens.

    Checked by OBJECT IDENTITY on each shared entry, not by equality: a
    re-spelled ``("plan_files", "phase_skipped_event", FIDELITY_FILE_LEVEL)``
    literal in this module would build a distinct tuple at import time and fail
    here, while a table derived from the imported one shares the very objects.
    So a future edit to the audit's lens is inherited automatically, and a
    contributor who copies the two entries instead fails CI.
    """
    for event_type, entry in _EVENT_PLAN_SOURCES.items():
        assert SCOPE_EVENT_SOURCES[event_type] is entry, event_type

    assert set(SCOPE_EVENT_SOURCES) == set(_EVENT_PLAN_SOURCES) | {"lock_acquired"}


def test_set_to_plan_and_phase_skipped_carry_their_audit_fidelities(tmp_path):
    """(a) A set_to_plan payload is DELIBERATELY lock-level (it carries the
    module set, per event_store.py:77-82), while phase_skipped.plan_files is
    true file-level. Fidelity stays load-bearing here for the same reason it is
    in the audit: a module path must never be mistaken for a plan.files entry."""
    db_path = _make_runs_db(
        tmp_path,
        [
            {"event_type": "set_to_plan", "task_id": 1, "data": {"files": ["mod/"]}},
            {
                "event_type": "phase_skipped",
                "task_id": 1,
                "data": {"plan_files": ["a.py", "b.py"]},
            },
        ],
    )
    events = load_scope_events(str(db_path), {"1"})["1"]

    assert [(e.event_type, e.fidelity, e.file_count) for e in events] == [
        ("set_to_plan", FIDELITY_LOCK_LEVEL, 1),
        ("phase_skipped", FIDELITY_FILE_LEVEL, 2),
    ]


def test_lock_acquired_with_real_module_paths_is_a_scope_event(tmp_path):
    """(b) The task names "a set_to_plan/phase_skipped{plan_files} event or
    real lock set" as the reconciliation signal, so lock_acquired is in the
    census's lens even though the audit does not read it."""
    db_path = _make_runs_db(
        tmp_path,
        [{"event_type": "lock_acquired", "task_id": 9, "data": {"modules": ["orchestrator/", "shared/"]}}],
    )
    events = load_scope_events(str(db_path), {"9"})["9"]

    assert [(e.event_type, e.fidelity, e.file_count) for e in events] == [
        ("lock_acquired", FIDELITY_LOCK_LEVEL, 2)
    ]


def test_the_synthetic_fallback_lock_is_not_evidence_of_reconciliation(tmp_path):
    """(b) THE INVERSION THIS PREVENTS. The conflict-with-nothing fallback lock
    renders as modules == ["task-<id>"] — a sentinel, not a derived scope.
    Counting it as evidence would classify the tagger-era dispatches that
    derived NOTHING as plan-reconciled, inverting the classification for
    exactly the population the census exists to find."""
    db_path = _make_runs_db(
        tmp_path,
        [{"event_type": "lock_acquired", "task_id": 4514, "data": {"modules": ["task-4514"]}}],
    )

    assert load_scope_events(str(db_path), {"4514"}) == {}


def test_a_mixed_lock_drops_only_the_sentinel(tmp_path):
    """(b) A lock carrying the sentinel ALONGSIDE real paths is still a real
    lock. Only the sentinel is dropped, and the file count reflects that."""
    db_path = _make_runs_db(
        tmp_path,
        [
            {
                "event_type": "lock_acquired",
                "task_id": 77,
                "data": {"modules": ["task-77", "scripts/"]},
            }
        ],
    )
    events = load_scope_events(str(db_path), {"77"})["77"]

    assert len(events) == 1
    assert events[0].file_count == 1


def test_every_scope_event_carries_its_sentinel_stripped_paths(tmp_path):
    """schema v2: the PATHS are load-bearing, not just the count — the axis-2
    echo test asks whether a lock's modules are a re-derivation of the record's
    own metadata.files, which is a question about the paths themselves."""
    db_path = _make_runs_db(
        tmp_path,
        [
            {
                "event_type": "lock_acquired",
                "task_id": 4514,
                "data": {"modules": ["task-4514", "a/b.py"]},
            },
            {"event_type": "set_to_plan", "task_id": 4514, "data": {"files": ["x.py", "y.py"]}},
            {"event_type": "phase_skipped", "task_id": 4514, "data": {"plan_files": ["p.py"]}},
        ],
    )
    events = load_scope_events(str(db_path), {"4514"})["4514"]

    # The synthetic sentinel is stripped from what the lock RECORDS, not just
    # from the count — otherwise the echo test would compare against a path
    # that never existed.
    assert [event.files for event in events] == [("a/b.py",), ("x.py", "y.py"), ("p.py",)]


def test_file_count_can_never_drift_from_the_paths_it_counts(tmp_path):
    """Two fields describing one list. Derived at the single construction
    site, and asserted here so a later edit cannot set them independently."""
    db_path = _make_runs_db(
        tmp_path,
        [
            {
                "event_type": "lock_acquired",
                "task_id": 7,
                "data": {"modules": ["task-7", "a/", "b/"]},
            },
            {"event_type": "phase_skipped", "task_id": 7, "data": {"plan_files": ["p.py"]}},
        ],
    )
    events = load_scope_events(str(db_path), {"7"})["7"]

    assert events
    for event in events:
        assert event.file_count == len(event.files)


@pytest.mark.parametrize(
    ("label", "event"),
    [
        ("null task_id", {"event_type": "set_to_plan", "task_id": None, "data": {"files": ["a"]}}),
        ("null data", {"event_type": "set_to_plan", "task_id": 1, "data": None}),
        ("malformed json", {"event_type": "set_to_plan", "task_id": 1, "data": "{nope"}),
        ("non-dict payload", {"event_type": "set_to_plan", "task_id": 1, "data": "[1,2]"}),
        ("missing key", {"event_type": "set_to_plan", "task_id": 1, "data": {"other": ["a"]}}),
        ("empty list", {"event_type": "phase_skipped", "task_id": 1, "data": {"plan_files": []}}),
        ("wrong-typed list", {"event_type": "phase_skipped", "task_id": 1, "data": {"plan_files": "a"}}),
        ("empty lock", {"event_type": "lock_acquired", "task_id": 1, "data": {"modules": []}}),
    ],
)
def test_load_scope_events_skips_unusable_rows_without_raising(tmp_path, label, event):
    """(c) An unusable row is NOT a scope assertion, so it is skipped. Never
    raising matters at live scale: dark_factory's event log carries 21,345
    lock_acquired rows alone, and one corrupt payload cannot be allowed to
    abort the sweep."""
    db_path = _make_runs_db(tmp_path, [event])

    assert load_scope_events(str(db_path), {"1", "None"}) == {}, label


def test_every_matching_event_is_retained_with_its_timestamp(tmp_path):
    """(d) THE REASON THIS FUNCTION EXISTS AT ALL.

    The audit's load_plan_files_from_events would collapse these three rows to
    ONE highest-fidelity record and discard the timestamps. The census needs
    all three, in emission order, because a record's classification turns on
    which side of the stamp each event fell.
    """
    db_path = _make_runs_db(
        tmp_path,
        [
            {"event_type": "set_to_plan", "task_id": 3, "timestamp": _BEFORE, "data": {"files": ["a"]}},
            {"event_type": "phase_skipped", "task_id": 3, "timestamp": _STAMP, "data": {"plan_files": ["b"]}},
            {"event_type": "lock_acquired", "task_id": 3, "timestamp": _AFTER, "data": {"modules": ["c/"]}},
        ],
    )
    events = load_scope_events(str(db_path), {"3"})["3"]

    assert [e.timestamp for e in events] == [_BEFORE, _STAMP, _AFTER]
    assert [e.event_id for e in events] == [1, 2, 3]


def test_only_the_requested_task_ids_are_returned(tmp_path):
    """(e) The sweep asks for the stamped ids only. Decoding all 21k+
    lock_acquired payloads to then discard almost all of them would make a
    six-corpus sweep needlessly expensive."""
    db_path = _make_runs_db(
        tmp_path,
        [
            {"event_type": "set_to_plan", "task_id": 1, "data": {"files": ["a"]}},
            {"event_type": "set_to_plan", "task_id": 2, "data": {"files": ["b"]}},
        ],
    )

    assert set(load_scope_events(str(db_path), {"2"})) == {"2"}


def test_an_empty_task_id_set_returns_nothing(tmp_path):
    """A project with zero stamped records asks for zero ids. That must be an
    empty result, never "all events"."""
    db_path = _make_runs_db(
        tmp_path, [{"event_type": "set_to_plan", "task_id": 1, "data": {"files": ["a"]}}]
    )

    assert load_scope_events(str(db_path), set()) == {}


# ---------------------------------------------------------------------------
# census_project — one project root, end to end.
# ---------------------------------------------------------------------------


def test_census_project_reports_only_stamped_records_sorted_numerically(tmp_path):
    """(a) 100 must FOLLOW 20, not precede it. Same reason the audit sorts on
    int(task_id) at _candidate_sort_key:573-578: a lexical id order makes an
    operator scanning the artifact lose their place."""
    root = _make_project(
        tmp_path,
        tasks=[
            {"id": 100, "metadata": {"files_tagged_at": _STAMP}},
            {"id": 20, "metadata": {"files_tagged_at": _STAMP}},
            {"id": 3, "metadata": {"files": ["a.py"]}},
        ],
    )
    census = census_project(str(root))

    assert [record.task_id for record in census.records] == [20, 100]


def test_census_record_carries_every_field_a_consumer_joins_on(tmp_path):
    """(b) INV-2: each row states its classification AND the evidence for it.

    The merge signature is the audit's own verdict over this task's
    merge_finalized history — a second, independently derived piece of
    evidence, in the vocabulary DF 3113 P4a and DF 3427 already consume.
    """
    root = _make_project(
        tmp_path,
        name="dark-factory",
        tasks=[
            {
                "id": 3113,
                "status": "pending",
                "metadata": {"files_tagged_at": _STAMP, "files": ["orchestrator/x.py"]},
            }
        ],
        events=[
            {
                "event_type": "set_to_plan",
                "task_id": 3113,
                "timestamp": _BEFORE,
                "data": {"files": ["orchestrator/"]},
            },
            {
                "event_type": "merge_finalized",
                "task_id": 3113,
                "data": {"state": "already_merged", "merge_sha": None},
            },
        ],
    )
    (record,) = census_project(str(root)).records

    assert record.project_id == "dark_factory"
    assert record.tag == "master"
    assert record.task_id == 3113
    assert record.status == "pending"
    assert record.files_tagged_at == _STAMP
    assert record.status_class == STATUS_NON_TERMINAL
    assert record.reconciliation == NEVER_RECONCILED
    assert record.wipe_signature == POST_WIPE_OVERWRITE
    assert record.preceded_by.event_type == "set_to_plan"
    assert record.preceded_by.timestamp == _BEFORE
    assert record.reconciled_by.event_type is None
    assert record.metadata_files == ("orchestrator/x.py",)
    assert record.merge_signature == CONFIRMED_NULL_SHA_DONE_PATH


def test_merge_signature_defaults_to_no_merge_event_not_to_a_clean_verdict(tmp_path):
    """(b) The correlation must be REAL REUSE, not a hardcoded default. With no
    merge_finalized row the audit's classifier returns NO_MERGE_EVENT — UNKNOWN,
    not clean, because found_on_main recovery and eval mode both reach DONE
    without emitting one."""
    root = _make_project(
        tmp_path, tasks=[{"id": 1, "metadata": {"files_tagged_at": _STAMP}}]
    )
    (record,) = census_project(str(root)).records

    assert record.merge_signature == NO_MERGE_EVENT


def test_a_missing_event_log_degrades_loudly_not_into_a_clean_project(tmp_path):
    """(c) NO-SILENT-FAIL-SOFT. Without runs.db, "no scope event postdates the
    stamp" is UNKNOWN, not measured — but every stamped record must still be
    reported, and the coverage block must say the event log was unreadable.
    Reporting the same rows with a silent clean verdict would tell DF 3427 the
    tagger's guesses were never superseded, which is a claim this run did not
    make."""
    root = _make_project(
        tmp_path,
        with_runs_db=False,
        tasks=[
            {"id": 1, "status": "pending", "metadata": {"files_tagged_at": _STAMP}},
            {"id": 2, "status": "done", "metadata": {"files_tagged_at": _STAMP}},
        ],
    )
    census = census_project(str(root))

    assert [record.task_id for record in census.records] == [1, 2]
    for record in census.records:
        assert record.reconciliation == NEVER_RECONCILED
        assert record.wipe_signature == NO_PRIOR_SCOPE
        assert record.merge_signature == NO_MERGE_EVENT
    assert census.coverage.event_log_read is False
    assert census.coverage.stamped_records == 2


def test_an_unreadable_tasks_db_raises_rather_than_reporting_a_partial_result(tmp_path):
    """(d) THE ONE-AUDIT-PER-ROOT-OR-RAISE CONTRACT.

    sweep_project_roots documents that its callback returns exactly one result
    per root or raises sqlite3.Error, because that equality is what makes "no
    results but some unreadable" mean precisely "every root failed" — the gate
    exit 3 rests on. Returning a partial or empty census here would re-open the
    false green that exit code exists to close.
    """
    root = tmp_path / "corrupt"
    tasks_dir = root / ".taskmaster" / "tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "tasks.db").write_bytes(b"this is not a database")

    with pytest.raises(sqlite3.Error):
        census_project(str(root))


def test_coverage_is_always_reported_even_for_a_project_with_no_stamps(tmp_path):
    """(e) A zero-record project must still report what was LOOKED AT. "Found
    nothing" and "looked at nothing" are different claims, and only the
    coverage block can tell them apart."""
    root = _make_project(
        tmp_path,
        name="know-live",
        tasks=[{"id": 1, "metadata": {"files": ["a.py"]}}, {"id": 2, "metadata": None}],
    )
    census = census_project(str(root))

    assert census.records == []
    assert census.coverage.project_id == "know_live"
    assert census.coverage.total_tasks == 2
    assert census.coverage.stamped_records == 0
    assert census.coverage.event_log_read is True


def test_a_post_stamp_scope_event_marks_the_record_reconciled(tmp_path):
    """The axis-2 path through the real loader: a post-stamp lock naming a
    module the record's own (here absent) files cannot explain is real
    evidence — and lands in the WEAKER lock_reconciled class, never
    plan_reconciled. See the axis-2 section above for why."""
    root = _make_project(
        tmp_path,
        tasks=[{"id": 5, "status": "done", "metadata": {"files_tagged_at": _STAMP}}],
        events=[
            {
                "event_type": "lock_acquired",
                "task_id": 5,
                "timestamp": _AFTER,
                "data": {"modules": ["scripts/"]},
            }
        ],
    )
    (record,) = census_project(str(root)).records

    assert record.status_class == STATUS_TERMINAL
    assert record.reconciliation == LOCK_RECONCILED
    assert record.reconciled_by.timestamp == _AFTER


def _echo_project(tmp_path, modules, files=("orchestrator/src/orchestrator/scheduler.py",), name="proj"):
    """A stamped record whose ONLY post-stamp scope event is a lock."""
    return _make_project(
        tmp_path,
        name=name,
        tasks=[
            {
                "id": 5,
                "status": "pending",
                "metadata": {"files_tagged_at": _STAMP, "files": list(files)},
            }
        ],
        events=[
            {
                "event_type": "lock_acquired",
                "task_id": 5,
                "timestamp": _AFTER,
                "data": {"modules": list(modules)},
            }
        ],
    )


def test_a_lock_echoing_the_records_own_files_is_not_reconciliation_end_to_end(tmp_path):
    """(c) THE STEP-18 UNIT PIN, RE-ASSERTED THROUGH THE REAL LOADERS. A
    correct classify_record wired up wrongly — metadata_files not threaded from
    the record — would still pass the unit tests and fail here."""
    root = _echo_project(tmp_path, modules=["orchestrator/src"])
    (record,) = census_project(str(root)).records

    assert record.reconciliation == NEVER_RECONCILED
    assert record.reconciled_by.event_type is None
    assert record.reconciled_by.fidelity is None


def test_a_lock_with_an_unexplained_module_is_lock_reconciled_end_to_end(tmp_path):
    """(c) The other side of the same wiring: a module the record's own files
    cannot explain is real evidence, in the weaker class."""
    root = _echo_project(tmp_path, modules=["orchestrator/src", "shared/"])
    (record,) = census_project(str(root)).records

    assert record.reconciliation == LOCK_RECONCILED
    assert record.reconciled_by.event_type == "lock_acquired"
    assert record.reconciled_by.fidelity == FIDELITY_LOCK_LEVEL


def test_the_emitted_row_states_the_fidelity_behind_each_axis(tmp_path):
    """(b) A consumer joining this artifact must be able to tell a file-level
    assertion from a lock-level one without re-deriving it from event_type."""
    root = _make_project(
        tmp_path,
        tasks=[
            {
                "id": 5,
                "status": "pending",
                "metadata": {"files_tagged_at": _STAMP, "files": ["scripts/a.py"]},
            }
        ],
        events=[
            {
                "event_type": "lock_acquired",
                "task_id": 5,
                "timestamp": _BEFORE,
                "data": {"modules": ["scripts"]},
            },
            {
                "event_type": "phase_skipped",
                "task_id": 5,
                "timestamp": _AFTER,
                "data": {"plan_files": ["scripts/a.py"]},
            },
        ],
    )
    (record,) = census_project(str(root)).records
    row = _record_to_dict(record)

    assert row["reconciled_by"]["fidelity"] == FIDELITY_FILE_LEVEL
    assert row["preceded_by"]["fidelity"] == FIDELITY_LOCK_LEVEL


def test_an_undecided_axis_emits_fidelity_present_and_null(tmp_path):
    """(b) THE ALL-KEYS-ALWAYS RULE (INV-2). A missing key must never be
    readable as "not looked" — the null is the measurement."""
    root = _make_project(
        tmp_path,
        tasks=[{"id": 5, "status": "pending", "metadata": {"files_tagged_at": _STAMP}}],
    )
    (record,) = census_project(str(root)).records
    row = _record_to_dict(record)

    for key in ("reconciled_by", "preceded_by"):
        assert set(row[key]) == {"event_type", "event_id", "timestamp", "fidelity"}
        assert row[key]["fidelity"] is None


def test_project_id_is_the_root_basename_with_underscores(tmp_path):
    """The six corpora spell their ids with underscores where the directory
    uses hyphens (solar-challenge-platform -> solar_challenge_platform). The
    artifact must use the id spelling its consumers already key on."""
    root = _make_project(tmp_path, name="solar-challenge-platform", tasks=[])

    assert census_project(str(root)).coverage.project_id == "solar_challenge_platform"


# ---------------------------------------------------------------------------
# build_report — the committed artifact's shape, and its determinism.
#
# Modelled on fused-memory/tests/test_census_memory_metadata.py's
# TestBuildReportDeterminism. The artifact is committed to the repo, so
# "re-running reproduces the counts" is checkable by `git diff --exit-code` —
# but ONLY if nothing in the report varies between two runs over the same
# corpus. These tests are what protect that property.
# ---------------------------------------------------------------------------


def _census(tmp_path, name="proj", tasks=(), events=(), with_runs_db=True):
    return census_project(
        str(_make_project(tmp_path, tasks=tasks, events=events, name=name, with_runs_db=with_runs_db))
    )


def _stamped(task_id, status="pending", files=("a.py",)):
    return {
        "id": task_id,
        "status": status,
        "metadata": {"files_tagged_at": _STAMP, "files": list(files)},
    }


def test_schema_version_is_the_first_key_and_is_two(tmp_path):
    """(a) First key, so a reader opening the raw JSON sees the version before
    anything it would have to interpret under that version.

    v2, not v1: axis 2 gained a value and every evidence object gained a key,
    so a consumer written against v1 must be able to DETECT the change. That is
    exactly what schema_version is for.
    """
    report = build_report([_census(tmp_path, tasks=[_stamped(1)])])

    assert next(iter(report)) == "schema_version"
    assert report["schema_version"] == SCHEMA_VERSION == 2


def test_params_says_how_the_artifact_was_produced(tmp_path):
    """(b) The artifact carries no clock read, so the params block IS its
    provenance: which roots were swept, what the labels mean, and the exact
    command that regenerates it."""
    report = build_report([_census(tmp_path, name="know-live", tasks=[])])
    params = report["params"]

    assert params["project_roots"] == [str(tmp_path / "know-live")]
    assert params["stamp_key"] == "metadata.files_tagged_at"
    assert params["classification"] == {
        "status_class": [STATUS_TERMINAL, STATUS_NON_TERMINAL],
        "reconciliation": [RECONCILED, LOCK_RECONCILED, NEVER_RECONCILED],
        "wipe_signature": [POST_WIPE_OVERWRITE, NO_PRIOR_SCOPE],
    }
    assert "census_tagger_debris.py" in params["regen_command"]


def test_zero_valued_classification_cells_are_present_not_omitted(tmp_path):
    """(c) A MISSING key must not be readable as a zero. DF 3427 will read the
    live-victim cell straight out of this artifact; if the cell vanished when
    it hit zero, "no victims" and "the schema changed" would look identical."""
    report = build_report([_census(tmp_path, tasks=[_stamped(1, status="pending")])])
    block = report["projects"]["proj"]

    assert block["total_tasks"] == 1
    assert block["stamped_records"] == 1
    assert block["status_class"] == {STATUS_TERMINAL: 0, STATUS_NON_TERMINAL: 1}
    assert block["reconciliation"] == {RECONCILED: 0, LOCK_RECONCILED: 0, NEVER_RECONCILED: 1}
    assert block["wipe_signature"] == {POST_WIPE_OVERWRITE: 0, NO_PRIOR_SCOPE: 1}

    # All twelve three-axis intersections, every one present.
    assert len(block["cells"]) == 12
    assert block["cells"]["non_terminal|never_reconciled|no_prior_scope"] == 1
    assert block["cells"]["terminal|plan_reconciled|post_wipe_overwrite"] == 0
    assert sum(block["cells"].values()) == block["stamped_records"]


def test_every_project_block_carries_a_lock_reconciled_count_even_at_zero(tmp_path):
    """(d) schema v2 added an axis-2 value. A consumer that reads
    `reconciliation[lock_reconciled]` must find a 0, not a KeyError, for a
    project that happens to have none."""
    report = build_report([_census(tmp_path, tasks=[_stamped(1)])])
    block = report["projects"]["proj"]

    assert block["reconciliation"][LOCK_RECONCILED] == 0
    # 2 status classes x 2 wipe signatures = 4 cells mention the new value.
    assert sum(1 for cell in block["cells"] if LOCK_RECONCILED in cell) == 4


def test_a_project_with_no_stamped_records_still_gets_a_full_block(tmp_path):
    """(c) Same reason: an absent project block and a project with nothing to
    report must not be the same artifact."""
    block = build_report([_census(tmp_path, tasks=[{"id": 1, "metadata": None}])])["projects"]["proj"]

    assert block["total_tasks"] == 1
    assert block["stamped_records"] == 0
    assert sum(block["cells"].values()) == 0
    assert len(block["cells"]) == 12


def test_records_are_totally_ordered_and_never_truncated(tmp_path):
    """(d) Total order by (project_id asc, NUMERIC task id asc). The JSON is
    the COMPLETE record — only the markdown is capped."""
    censuses = [
        _census(tmp_path, name="reify", tasks=[_stamped(100), _stamped(20)]),
        _census(tmp_path, name="autopilot-video", tasks=[_stamped(7)]),
    ]
    report = build_report(censuses)

    assert [(r["project_id"], r["task_id"]) for r in report["records"]] == [
        ("autopilot_video", 7),
        ("reify", 20),
        ("reify", 100),
    ]


def test_building_twice_is_byte_identical_and_input_order_does_not_matter(tmp_path):
    """(e) THE REPRODUCIBILITY PROPERTY, which `git diff --exit-code` on the
    committed artifact ultimately rests on."""
    a = _census(tmp_path, name="reify", tasks=[_stamped(2), _stamped(1)])
    b = _census(tmp_path, name="know-live", tasks=[_stamped(9)])

    first = json.dumps(build_report([a, b]), indent=2, sort_keys=False)
    second = json.dumps(build_report([a, b]), indent=2, sort_keys=False)
    shuffled = json.dumps(build_report([b, a]), indent=2, sort_keys=False)

    assert first == second
    assert first == shuffled


def _walk_keys(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield path, key
            yield from _walk_keys(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for item in node:
            yield from _walk_keys(item, path)


def test_the_report_carries_no_clock_read_anywhere(tmp_path):
    """(f) WHY THIS TEST EXISTS AND MUST NOT BE "FIXED".

    All three committed plans/*.json artifacts in this repo deliberately carry
    no generated_at. That absence is what makes a re-run diff PURE SIGNAL: with
    a clock read in the file, every regeneration would diff dirty and the
    task's "re-running reproduces the counts" signal would be destroyed. A
    later contributor adding one back fails here rather than silently.

    ``timestamp`` is permitted ONLY inside the two evidence objects, where it
    names WHEN A RECORDED EVENT HAPPENED — corpus-derived and stable across
    runs — never when this run happened.
    """
    report = build_report([_census(tmp_path, tasks=[_stamped(1)])])

    for parent, key in _walk_keys(report):
        assert key not in {"generated_at", "created_at", "run_at", "sha", "commit", "git_sha"}
        if key == "timestamp":
            assert parent.endswith(("reconciled_by", "preceded_by")), parent


def test_coverage_is_always_present_and_names_an_unreadable_event_log(tmp_path):
    """(g) An incomplete sweep must SAY SO in the artifact itself, not only on
    a stderr line nobody kept."""
    censuses = [
        _census(tmp_path, name="reify", tasks=[_stamped(1)], with_runs_db=False),
        _census(tmp_path, name="know-live", tasks=[_stamped(2)]),
    ]
    coverage = build_report(censuses)["coverage"]

    assert coverage["projects_swept"] == 2
    assert coverage["projects_without_event_log"] == ["reify"]
    assert coverage["total_tasks"] == 2
    assert coverage["stamped_records"] == 2


def test_coverage_reports_an_empty_shortfall_list_rather_than_omitting_it(tmp_path):
    """A clean sweep still states the shortfall list, empty. Omission would
    read as "not checked"."""
    coverage = build_report([_census(tmp_path, tasks=[_stamped(1)])])["coverage"]

    assert coverage["projects_without_event_log"] == []


# ---------------------------------------------------------------------------
# render_markdown / write_artifacts — the committed pair.
# ---------------------------------------------------------------------------


def test_the_markdown_covers_every_swept_project_including_empty_ones(tmp_path):
    """(a) An operator reads the markdown; a project silently missing from the
    table is indistinguishable from a project that was never swept."""
    report = build_report(
        [
            _census(tmp_path, name="reify", tasks=[_stamped(1)]),
            _census(tmp_path, name="know-live", tasks=[{"id": 2, "metadata": None}]),
        ]
    )
    markdown = render_markdown(report)

    assert "reify" in markdown
    assert "know_live" in markdown
    assert NEVER_RECONCILED in markdown
    assert "census_tagger_debris.py" in markdown


def test_the_markdown_gives_each_axis_2_value_its_own_column(tmp_path):
    """(f) An operator reads the markdown. Three classes collapsed into one
    "reconciled" column would hide exactly the distinction schema v2 adds."""
    markdown = render_markdown(build_report([_census(tmp_path, tasks=[_stamped(1)])]))
    header = next(line for line in markdown.splitlines() if line.startswith("| project |"))

    for label in ("plan reconciled", "lock reconciled", "never reconciled"):
        assert label in header, header


def test_the_markdown_states_that_a_lock_is_derived_from_metadata_files(tmp_path):
    """(f) THE CAVEAT, in the operator's own terms. Without it a reader takes
    `lock_reconciled` at face value — which is precisely the v1 defect, moved
    from the code into the reader's head."""
    markdown = render_markdown(
        build_report([_census(tmp_path, tasks=[_stamped(1)])])
    ).lower()

    # a lock's modules come FROM metadata.files, so it is not independent
    assert "derived from `metadata.files`" in markdown
    assert "not an independent scope derivation" in markdown
    # ...therefore it is the weaker class, and the consumer must choose
    assert "weaker signal than `plan_reconciled`" in markdown
    assert "decide for itself" in markdown
    # ...while plan_reconciled is the genuine article
    assert "genuine plan-derived assertion" in markdown
    # ...and a lock_reconciled record may still be carrying the guess
    assert "may still be carrying the tagger's guess" in markdown


def test_rendering_is_deterministic_and_ends_with_one_trailing_newline(tmp_path):
    """(b) Same property, same reason, as the JSON: a re-run must diff clean."""
    report = build_report([_census(tmp_path, tasks=[_stamped(1)])])

    assert render_markdown(report) == render_markdown(report)
    assert render_markdown(report).endswith("\n")
    assert not render_markdown(report).endswith("\n\n")


def test_the_markdown_says_which_file_is_authoritative(tmp_path):
    """(c) The markdown is CAPPED and the JSON is not, so the markdown must
    say so — otherwise a consumer could read a truncated table as the whole
    population."""
    markdown = render_markdown(build_report([_census(tmp_path, tasks=[_stamped(1)])]))

    assert "module-tagger-debris-census.json" in markdown
    assert "3113" in markdown and "3427" in markdown


def test_the_markdown_names_a_coverage_shortfall_rather_than_omitting_it(tmp_path):
    """(c) An incomplete sweep must be legible in the readable twin too."""
    report = build_report(
        [_census(tmp_path, name="reify", tasks=[_stamped(1)], with_runs_db=False)]
    )

    assert "reify" in render_markdown(report)
    assert "event log" in render_markdown(report).lower()


def test_write_artifacts_writes_both_files_and_the_json_round_trips(tmp_path):
    """(d) The exact serializer the convention pins: indent=2,
    sort_keys=False (key ORDER is meaning here — schema_version leads), and a
    trailing newline so the file is POSIX-clean and diffs by line."""
    report = build_report([_census(tmp_path, tasks=[_stamped(1)])])
    json_out = tmp_path / "out" / "census.json"
    md_out = tmp_path / "out" / "census.md"

    written_json, written_md = write_artifacts(report, json_out, md_out)

    assert written_json == json_out and written_md == md_out
    raw = json_out.read_text()
    assert raw == json.dumps(report, indent=2, sort_keys=False) + "\n"
    assert json.loads(raw) == report
    assert md_out.read_text() == render_markdown(report)


def test_neither_file_is_touched_when_rendering_fails(tmp_path):
    """(e) THE ATOMICITY PROPERTY (bake_off_storage_shape.write_artifacts).

    The markdown is rendered BEFORE either destination is replaced, so a stale
    .md can never accompany a fresh .json. Proven by handing write_artifacts a
    report that cannot render and asserting BOTH existing files survive
    byte-for-byte — not merely that no new file appeared.
    """
    json_out = tmp_path / "census.json"
    md_out = tmp_path / "census.md"
    json_out.write_text("PREVIOUS JSON")
    md_out.write_text("PREVIOUS MD")

    with pytest.raises(KeyError):
        write_artifacts({"schema_version": 1}, json_out, md_out)

    assert json_out.read_text() == "PREVIOUS JSON"
    assert md_out.read_text() == "PREVIOUS MD"


def test_default_output_paths_are_the_tasks_user_observable_signal():
    """(f) These two paths ARE task 4525's deliverable, and the .md path
    satisfies its second delivered_check (grep module-tagger-debris-census
    under plans/). Resolved __file__-relatively, never hardcoded absolute, so
    a worktree run writes into its own tree rather than the main checkout."""
    assert DEFAULT_JSON_OUT.as_posix().endswith("plans/module-tagger-debris-census.json")
    assert DEFAULT_MD_OUT.as_posix().endswith("plans/module-tagger-debris-census.md")
    assert DEFAULT_JSON_OUT.is_absolute()


# ---------------------------------------------------------------------------
# CLI coverage.
#
# Driven by SUBPROCESS, never `python -m`: a directly-executed script places
# its own directory at sys.path[0], which is the only reason the child resolves
# `import audit_wiped_metadata_files` and `import _task_db_scan`. That is the
# flat-sibling import contract at _task_db_scan.py:93-103, and `python -m`
# would break every test below at once.
#
# Not one test here points at a live corpus.
# ---------------------------------------------------------------------------

_SCRIPT = str(Path(__file__).parent.parent.parent / "scripts" / "census_tagger_debris.py")


def _run_cli(*args):
    return subprocess.run([sys.executable, _SCRIPT, *args], capture_output=True, text=True)


def _corrupt_root(tmp_path, name="corrupt"):
    """A root whose tasks.db EXISTS (so discovery keeps it) but is unreadable."""
    root = tmp_path / name
    tasks_dir = root / ".taskmaster" / "tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "tasks.db").write_bytes(b"not a database")
    return root


def test_a_successful_sweep_exits_zero_and_writes_both_artifacts(tmp_path):
    """(a) THE USER-OBSERVABLE SIGNAL: the sweep runs, exits 0, and produces
    the artifact pair."""
    root = _make_project(tmp_path, tasks=[_stamped(1)])
    json_out, md_out = tmp_path / "c.json", tmp_path / "c.md"

    result = _run_cli(
        "--project-root", str(root), "--json-out", str(json_out), "--md-out", str(md_out)
    )

    assert result.returncode == EXIT_OK, result.stderr
    assert json.loads(json_out.read_text())["schema_version"] == SCHEMA_VERSION
    assert md_out.exists()


def test_no_resolvable_root_exits_two_and_writes_nothing(tmp_path):
    """(b) Nothing was swept, so there is nothing to publish. Overwriting a
    good artifact on this path would replace real findings with an empty file."""
    json_out = tmp_path / "c.json"

    result = _run_cli(
        "--project-root", str(tmp_path / "does-not-exist"),
        "--json-out", str(json_out), "--md-out", str(tmp_path / "c.md"),
    )

    assert result.returncode == EXIT_NO_ROOT
    assert result.stderr.strip()
    assert not json_out.exists()


def test_every_root_unreadable_exits_three_and_writes_nothing(tmp_path):
    """(c) EXIT 3 IS NOT A CLEAN RUN. Roots resolved but every one failed, so
    the census measured nothing. Writing an empty artifact here would record
    "no debris found" for a sweep that looked at nothing —
    docs/legibility/design-invariants.md, no-silent-fail-soft."""
    json_out = tmp_path / "c.json"

    result = _run_cli(
        "--project-root", str(_corrupt_root(tmp_path)),
        "--json-out", str(json_out), "--md-out", str(tmp_path / "c.md"),
    )

    assert result.returncode == EXIT_NOTHING_SCANNED
    assert "clean" in result.stderr.lower()
    assert not json_out.exists()


def test_one_unreadable_root_among_several_warns_and_continues(tmp_path):
    """(d) A single corrupt project must not abort the sweep — but the
    incompleteness must reach BOTH the stderr warning and the artifact's own
    coverage block, so it survives a run whose stderr nobody kept."""
    good = _make_project(tmp_path, name="reify", tasks=[_stamped(1)])
    bad = _corrupt_root(tmp_path)
    json_out = tmp_path / "c.json"

    result = _run_cli(
        "--project-root", str(good), "--project-root", str(bad),
        "--json-out", str(json_out), "--md-out", str(tmp_path / "c.md"),
    )

    assert result.returncode == EXIT_OK, result.stderr
    assert "incomplete" in result.stderr.lower()
    coverage = json.loads(json_out.read_text())["coverage"]
    assert coverage["projects_skipped_unreadable"] == [str(bad)]


def test_check_mode_agrees_with_a_freshly_written_artifact(tmp_path):
    """(e) The reproducibility claim, machine-checkable without writing."""
    root = _make_project(tmp_path, tasks=[_stamped(1)])
    json_out, md_out = tmp_path / "c.json", tmp_path / "c.md"
    args = ("--project-root", str(root), "--json-out", str(json_out), "--md-out", str(md_out))

    assert _run_cli(*args).returncode == EXIT_OK
    before = json_out.read_text()

    result = _run_cli(*args, "--check")

    assert result.returncode == EXIT_OK, result.stderr
    assert json_out.read_text() == before, "--check must never write"


def test_check_mode_reports_drift_without_writing(tmp_path):
    """(e) Drift exits 1 and NAMES the divergence, and still does not write —
    --check reports, it never repairs."""
    root = _make_project(tmp_path, tasks=[_stamped(1)])
    json_out, md_out = tmp_path / "c.json", tmp_path / "c.md"
    args = ("--project-root", str(root), "--json-out", str(json_out), "--md-out", str(md_out))
    _run_cli(*args)
    before = json_out.read_text()

    # Drift the corpus the way a live one drifts: a new stamped record lands.
    conn = sqlite3.connect(root / ".taskmaster" / "tasks" / "tasks.db")
    conn.execute(
        "INSERT INTO tasks (tag, id, title, status, metadata, updated_at) "
        "VALUES ('master', 999, 't', 'pending', ?, '2026-08-01T00:00:00+00:00')",
        (json.dumps({"files_tagged_at": _STAMP}),),
    )
    conn.commit()
    conn.close()

    result = _run_cli(*args, "--check")

    assert result.returncode == EXIT_STALE
    assert result.stderr.strip()
    assert json_out.read_text() == before


def test_check_mode_treats_a_missing_artifact_as_stale(tmp_path):
    """(e) An absent artifact cannot be reproducing anything."""
    root = _make_project(tmp_path, tasks=[_stamped(1)])

    result = _run_cli(
        "--project-root", str(root),
        "--json-out", str(tmp_path / "absent.json"), "--md-out", str(tmp_path / "absent.md"),
        "--check",
    )

    assert result.returncode == EXIT_STALE
    assert not (tmp_path / "absent.json").exists()


def test_json_flag_prints_the_report_instead_of_writing(tmp_path):
    root = _make_project(tmp_path, tasks=[_stamped(1)])
    json_out = tmp_path / "c.json"

    result = _run_cli(
        "--project-root", str(root), "--json-out", str(json_out),
        "--md-out", str(tmp_path / "c.md"), "--json",
    )

    assert result.returncode == EXIT_OK, result.stderr
    assert json.loads(result.stdout)["schema_version"] == SCHEMA_VERSION
    assert not json_out.exists()


# ---------------------------------------------------------------------------
# Lockstep guard for the decision that this script keeps its own EXIT_* ladder
# instead of adopting _task_db_scan.run_audit_cli. The shared constants are
# IMPORTED, never re-spelled as 0/2/3 literals, or the guard would drift
# exactly as the thing it guards against. Same shape as
# test_repair_wiped_metadata_files.py:1685.
# ---------------------------------------------------------------------------


def test_exit_codes_stay_in_lockstep_with_the_shared_audit_ladder():
    """The three shared codes must keep the SAME integers as Tier 3's.

    The census adopts sweep_project_roots and this numbering but deliberately
    NOT run_audit_cli: that function's exit 1 means "the sweep found something
    dirty", and the census ALWAYS finds records, which would make its mandated
    exit-0 reproducibility signal unreachable. Exit 1 is therefore reused here
    for a DIFFERENT meaning — the artifact is stale — so the other three codes
    agreeing is exactly what has to be frozen. Renumbering deliberately is
    allowed; it must edit this test rather than drift past it unnoticed.
    """
    assert EXIT_OK == AUDIT_EXIT_OK
    assert EXIT_NO_ROOT == AUDIT_EXIT_NO_ROOT
    assert EXIT_NOTHING_SCANNED == AUDIT_EXIT_NOTHING_AUDITED
    assert EXIT_STALE == 1


# ---------------------------------------------------------------------------
# The committed-artifact contract — the machine-checkable form of task 4525's
# user-observable signal.
#
# These read the STATIC repo files, never a live database, so they are stable
# under corpus drift. That is what makes it safe to assert the four positive
# controls here and nowhere else: the artifact is a committed snapshot, and
# regenerating it is a deliberate, reviewed act.
#
# DELIBERATELY ABSENT: any assertion on a total count, on the PRD's "33 live
# victims (11 dark_factory, 22 reify)" figure, or on any other live-DB-derived
# value. The PRD's number was measured with a differently-scoped lens and does
# not reproduce under the strict three-axis intersection this task specifies;
# asserting it would be a doomed test no implementation could satisfy.
# ---------------------------------------------------------------------------

_ARTIFACT_JSON = Path(__file__).parent.parent.parent / "plans" / "module-tagger-debris-census.json"
_ARTIFACT_MD = Path(__file__).parent.parent.parent / "plans" / "module-tagger-debris-census.md"

_REQUIRED_PROJECT_IDS = (
    "dark_factory",
    "reify",
    "autopilot_video",
    "know_live",
    "pump_web_ui",
    "solar_challenge_platform",
)

# The four records task 4525 names as positive controls. Asserted against the
# COMMITTED artifact, not a live query, so corpus drift cannot turn them red.
_POSITIVE_CONTROLS = (("reify", 6068), ("reify", 5602), ("reify", 5632), ("dark_factory", 3113))


@pytest.fixture(scope="module")
def artifact():
    assert _ARTIFACT_JSON.exists(), f"the committed artifact is missing: {_ARTIFACT_JSON}"
    return json.loads(_ARTIFACT_JSON.read_text(encoding="utf-8"))


def test_the_committed_pair_exists_and_the_json_parses(artifact):
    """(a) Both halves are committed. A JSON without its readable twin is a
    half-published artifact."""
    assert _ARTIFACT_MD.exists()
    assert artifact["records"]


def test_the_artifact_cannot_silently_lag_a_schema_bump(artifact):
    """(b) Bumping SCHEMA_VERSION without regenerating leaves consumers reading
    an old shape under a new version number. This is what forces the pair to
    move together."""
    assert artifact["schema_version"] == SCHEMA_VERSION


def test_all_six_corpora_are_present_in_both_halves(artifact):
    """(c) A missing project is indistinguishable from a project with nothing
    to report unless the block is present with its counts."""
    markdown = _ARTIFACT_MD.read_text(encoding="utf-8")
    for project_id in _REQUIRED_PROJECT_IDS:
        block = artifact["projects"][project_id]
        assert isinstance(block["total_tasks"], int)
        assert isinstance(block["stamped_records"], int)
        assert project_id in markdown


@pytest.mark.parametrize(("project_id", "task_id"), _POSITIVE_CONTROLS)
def test_the_required_positive_controls_are_present_and_classified(artifact, project_id, task_id):
    """(d) The four records task 4525 names. Each must carry a non-empty stamp
    and a full three-axis classification — a control present but unclassified
    would prove nothing."""
    matches = [
        record for record in artifact["records"]
        if record["project_id"] == project_id and record["task_id"] == task_id
    ]
    assert len(matches) == 1, f"{project_id} {task_id} missing from the artifact"
    (record,) = matches

    assert record["files_tagged_at"]
    assert record["status_class"] in {STATUS_TERMINAL, STATUS_NON_TERMINAL}
    assert record["reconciliation"] in {RECONCILED, LOCK_RECONCILED, NEVER_RECONCILED}
    assert record["wipe_signature"] in {POST_WIPE_OVERWRITE, NO_PRIOR_SCOPE}


def test_every_record_carries_its_complete_evidence_key_set(artifact):
    """(e) INV-2: no row is a prose-only claim. Every record states the
    deciding event for each axis, with an explicit null where there was none."""
    for record in artifact["records"]:
        for key in ("reconciled_by", "preceded_by"):
            assert set(record[key]) == {
                "event_type",
                "event_id",
                "timestamp",
                "fidelity",
            }, record["task_id"]
        assert record["merge_signature"]


def test_per_project_counts_agree_with_the_records_array(artifact):
    """(f) An INTERNAL-CONSISTENCY check that pins no absolute number, so it
    cannot go red on corpus drift: whatever the counts are, the summary and the
    detail must be telling the same story."""
    counted = Counter(record["project_id"] for record in artifact["records"])

    for project_id, block in artifact["projects"].items():
        assert block["stamped_records"] == counted.get(project_id, 0), project_id
        assert sum(block["cells"].values()) == block["stamped_records"], project_id

    assert artifact["coverage"]["stamped_records"] == len(artifact["records"])
