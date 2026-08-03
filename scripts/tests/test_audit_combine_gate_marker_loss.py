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
import subprocess
import sys
from pathlib import Path

from audit_combine_gate_marker_loss import (
    SOURCE_MANIFEST,
    SOURCE_NONE,
    SOURCE_TICKET,
    SEVERITY_BENIGN,
    SEVERITY_DISPATCH,
    SEVERITY_GATE_REMOVING,
    SEVERITY_INFORMATIONAL,
    SEVERITY_PROVENANCE,
    TERMINAL_STATUSES,
    AuditCoverage,
    CombineTarget,
    Finding,
    ManifestExpectation,
    ProjectAudit,
    _COVERAGE_CAVEAT,
    _FIDELITY_CAVEAT,
    _severity_rank,
    audit_project,
    build_manifest_index,
    classify_gap,
    format_json,
    format_report,
    load_combine_targets,
    load_ticket_expectations,
    resolve_project_id,
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


# ---------------------------------------------------------------------------
# build_manifest_index — comparison source (2), the capability manifests.
#
# Keyed by task_id and built by GLOBBING every manifest, deliberately WITHOUT
# consulting live metadata.prd_path: prd_path is itself one of the wiped keys,
# so following the live pointer would blind the detector precisely on the
# victims it exists to find.
# ---------------------------------------------------------------------------

_GREP_CHECK = {"kind": "grep", "pattern": "def foo", "paths": ["a.py"], "expect": "present"}
_SCRIPT_CHECK = {"kind": "script", "script": "scripts/x.sh", "timeout_secs": 30}
_MANUAL_CHECK = {"kind": "manual", "reason": "needs a human eye"}


def _capability(name: str, check: dict | None) -> dict:
    cap = {"name": name, "binding": "capability->producer (wired)", "verdict": "PASS"}
    if check is not None:
        cap["delivered_check"] = check
    return cap


def _write_manifest(root: Path, relpath: str, doc) -> Path:
    """Write *doc* to ``<root>/<relpath>``; a str is written VERBATIM.

    Verbatim passthrough is what lets a test seed unparseable YAML.
    """
    import yaml

    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc if isinstance(doc, str) else yaml.safe_dump(doc), encoding="utf-8")
    return path


def test_build_manifest_index_maps_task_id_to_expectation(tmp_path):
    """task_id -> (manifest_path, prd_path, label, delivered_check_names)."""
    manifest = _write_manifest(tmp_path, "plans/a-prd.capability-manifest.yaml", {
        "prd": "plans/a-prd.md",
        "schema_version": 1,
        "tasks": [{"label": "δ", "task_id": 3157,
                   "capabilities": [_capability("cap-one", _GREP_CHECK)]}],
    })

    index = build_manifest_index(str(tmp_path))

    assert set(index) == {"3157"}
    expectation = index["3157"]
    assert isinstance(expectation, ManifestExpectation)
    assert expectation.manifest_path == str(manifest)
    assert expectation.prd_path == "plans/a-prd.md"
    assert expectation.label == "δ"
    assert expectation.delivered_check_names == ("cap-one",)


def test_build_manifest_index_keeps_grep_AND_script_drops_only_manual(tmp_path):
    """THE MECHANICAL-KINDS RULE, corrected against the real stamping site.

    fused-memory/src/fused_memory/server/manifest_stamping.py:311 reads
    `if check is None or check.kind not in ('grep', 'script'): continue` — so
    BOTH grep and script are copied into metadata.delivered_checks and only
    'manual' is dropped. A grep-only filter would silently under-count the
    expected entries and yield FALSE NEGATIVES on exactly the highest-severity
    class, so the rule is pinned here with all three kinds on one task.
    """
    _write_manifest(tmp_path, "plans/b-prd.capability-manifest.yaml", {
        "prd": "plans/b-prd.md",
        "schema_version": 1,
        "tasks": [{"label": "ε", "task_id": 3319, "capabilities": [
            _capability("cap-grep", _GREP_CHECK),
            _capability("cap-script", _SCRIPT_CHECK),
            _capability("cap-manual", _MANUAL_CHECK),
            _capability("cap-no-check", None),
        ]}],
    })

    names = build_manifest_index(str(tmp_path))["3319"].delivered_check_names

    assert names == ("cap-grep", "cap-script")


def test_build_manifest_index_globs_both_plans_and_docs_prds(tmp_path):
    """Both manifest homes are swept: <root>/plans and <root>/docs/prds."""
    _write_manifest(tmp_path, "plans/c-prd.capability-manifest.yaml", {
        "prd": "plans/c-prd.md", "schema_version": 1,
        "tasks": [{"label": "α", "task_id": 1,
                   "capabilities": [_capability("c", _GREP_CHECK)]}],
    })
    _write_manifest(tmp_path, "docs/prds/d-prd.capability-manifest.yaml", {
        "prd": "docs/prds/d-prd.md", "schema_version": 1,
        "tasks": [{"label": "β", "task_id": 2,
                   "capabilities": [_capability("d", _GREP_CHECK)]}],
    })

    assert set(build_manifest_index(str(tmp_path))) == {"1", "2"}


def test_build_manifest_index_resolves_a_task_with_no_live_prd_path(tmp_path):
    """THE POINT OF THE REVERSE INDEX. The index is built by GLOBBING, never by
    following a task's live metadata.prd_path — which is itself one of the
    wiped keys. A task whose live metadata carries no prd_path at all (the
    3157/3319 shape) must still resolve, because nothing in this function ever
    consults tasks.db."""
    _write_manifest(tmp_path, "plans/e-prd.capability-manifest.yaml", {
        "prd": "plans/e-prd.md", "schema_version": 1,
        "tasks": [{"label": "δ", "task_id": 3157,
                   "capabilities": [_capability("gate", _GREP_CHECK)]}],
    })

    index = build_manifest_index(str(tmp_path))

    # No tasks.db exists under tmp_path at all, and the lookup still succeeds.
    assert not (tmp_path / ".taskmaster").exists()
    assert index["3157"].delivered_check_names == ("gate",)


def test_build_manifest_index_skips_manifest_tasks_with_no_task_id(tmp_path):
    """task_id is None at authoring time, stamped by commit_planning. An
    unstamped block binds nothing and cannot be keyed."""
    _write_manifest(tmp_path, "plans/f-prd.capability-manifest.yaml", {
        "prd": "plans/f-prd.md", "schema_version": 1,
        "tasks": [
            {"label": "α", "capabilities": [_capability("x", _GREP_CHECK)]},
            {"label": "β", "task_id": 7, "capabilities": [_capability("y", _GREP_CHECK)]},
        ],
    })

    assert set(build_manifest_index(str(tmp_path))) == {"7"}


def test_build_manifest_index_flags_a_task_bound_by_two_manifests(tmp_path):
    """Ambiguity is REPORTED, never resolved by silently taking the first.

    Measured: 31 live manifests bind 250 task_ids with ZERO bound twice, so
    this is a defensive path — but a silent first-wins would attribute the
    wrong delivered_checks to a task if it ever occurred.
    """
    first = _write_manifest(tmp_path, "plans/g-prd.capability-manifest.yaml", {
        "prd": "plans/g-prd.md", "schema_version": 1,
        "tasks": [{"label": "α", "task_id": 55,
                   "capabilities": [_capability("from-g", _GREP_CHECK)]}],
    })
    second = _write_manifest(tmp_path, "plans/h-prd.capability-manifest.yaml", {
        "prd": "plans/h-prd.md", "schema_version": 1,
        "tasks": [{"label": "β", "task_id": 55,
                   "capabilities": [_capability("from-h", _GREP_CHECK)]}],
    })

    expectation = build_manifest_index(str(tmp_path))["55"]

    assert expectation.ambiguous is True
    assert set(expectation.bound_by) == {str(first), str(second)}


def test_build_manifest_index_unambiguous_binding_is_not_flagged(tmp_path):
    """The ordinary case carries ambiguous=False and a single binding."""
    manifest = _write_manifest(tmp_path, "plans/i-prd.capability-manifest.yaml", {
        "prd": "plans/i-prd.md", "schema_version": 1,
        "tasks": [{"label": "α", "task_id": 56,
                   "capabilities": [_capability("only", _GREP_CHECK)]}],
    })

    expectation = build_manifest_index(str(tmp_path))["56"]

    assert expectation.ambiguous is False
    assert expectation.bound_by == (str(manifest),)


def test_build_manifest_index_records_parse_failures_without_raising(tmp_path):
    """An unparseable or schema-invalid manifest is skipped and RECORDED.

    Recorded rather than swallowed: the count reaches the coverage block, so a
    sweep that could not read half the manifests never reads as complete.
    """
    bad_yaml = _write_manifest(tmp_path, "plans/j-prd.capability-manifest.yaml",
                               "prd: [unclosed\n  - nope")
    bad_schema = _write_manifest(tmp_path, "plans/k-prd.capability-manifest.yaml", {
        "prd": "plans/k-prd.md", "schema_version": 99, "tasks": [],
    })
    _write_manifest(tmp_path, "plans/l-prd.capability-manifest.yaml", {
        "prd": "plans/l-prd.md", "schema_version": 1,
        "tasks": [{"label": "α", "task_id": 8,
                   "capabilities": [_capability("ok", _GREP_CHECK)]}],
    })

    failures: list[str] = []
    index = build_manifest_index(str(tmp_path), parse_failures=failures)

    assert set(index) == {"8"}
    assert len(failures) == 2
    assert any(str(bad_yaml) in f for f in failures)
    assert any(str(bad_schema) in f for f in failures)


def test_build_manifest_index_no_manifest_dirs_returns_empty(tmp_path):
    """A project with no plans/ or docs/prds/ at all yields {} , never a raise."""
    failures: list[str] = []
    assert build_manifest_index(str(tmp_path), parse_failures=failures) == {}
    assert failures == []


# ---------------------------------------------------------------------------
# classify_gap — RANK BY CONSUMER, NOT BY GAP SIZE.
#
# task_kind is the single most COMMON gap (all 24 live victims lost it) but is
# load-bearing only in the rare deterministic case. Ranking by frequency would
# bury the delivered_checks losses — the only ones that silently remove a
# mark-done gate — under a wall of benign rows. These tests are what make the
# consumer ordering mechanical rather than aspirational.
# ---------------------------------------------------------------------------

def test_classify_gap_delivered_checks_is_gate_removing(tmp_path):
    """Rank 0: wiping delivered_checks removes the mark-done gate outright."""
    severity, reason = classify_gap("delivered_checks", [{"name": "x", "kind": "grep"}])

    assert severity == SEVERITY_GATE_REMOVING
    assert reason


def test_classify_gap_prd_keys_are_provenance(tmp_path):
    """Rank 1: prd_path / prd_task_label are provenance, not a gate."""
    for key in ("prd_path", "prd_task_label"):
        severity, reason = classify_gap(key, "plans/some-prd.md")
        assert severity == SEVERITY_PROVENANCE, key
        assert reason, key


def test_classify_gap_task_kind_deterministic_is_dispatch(tmp_path):
    """Rank 2: only a 'deterministic' expected value changes dispatch."""
    severity, reason = classify_gap("task_kind", "deterministic")

    assert severity == SEVERITY_DISPATCH
    assert reason


def test_classify_gap_task_kind_normal_is_benign_with_a_stated_reason(tmp_path):
    """LOWEST rank. Scheduler.is_deterministic() (scheduler.py:2046) tests
    `metadata.get('task_kind') == 'deterministic'`, so an ABSENT task_kind is
    behaviourally byte-identical to task_kind='normal'. Losing it therefore
    changes nothing, and the demotion prints its own reason rather than asking
    a reader to take it on trust."""
    for value in ("normal", "", "something-else", None, 17):
        severity, reason = classify_gap("task_kind", value)
        assert severity == SEVERITY_BENIGN, value
        assert "deterministic" in reason, value


def test_classify_gap_task_kind_branch_inspects_the_expected_value(tmp_path):
    """The task_kind branch keys on the EXPECTED VALUE, not the key name — the
    same key yields two different severities."""
    assert classify_gap("task_kind", "deterministic")[0] != classify_gap("task_kind", "normal")[0]


def test_classify_gap_other_known_keys_are_informational(tmp_path):
    """Every other submit-payload key is informational: real provenance loss,
    but no consumer that gates or dispatches."""
    for key in ("source", "spawn_context", "spawned_from", "escalation_id",
                "suggestion_hash", "execution_class", "complexity", "modules", "files"):
        severity, reason = classify_gap(key, "whatever")
        assert severity == SEVERITY_INFORMATIONAL, key
        assert reason, key


def test_classify_gap_unknown_key_falls_through_informational(tmp_path):
    """An unforeseen key must not raise — the metadata vocabulary grows."""
    severity, reason = classify_gap("some_future_key_nobody_has_written_yet", {"a": 1})

    assert severity == SEVERITY_INFORMATIONAL
    assert reason


def test_severity_rank_orders_gate_removing_first(tmp_path):
    """Strict rank order: gate-removing < provenance < dispatch <
    informational < benign."""
    ordered = [
        SEVERITY_GATE_REMOVING,
        SEVERITY_PROVENANCE,
        SEVERITY_DISPATCH,
        SEVERITY_INFORMATIONAL,
        SEVERITY_BENIGN,
    ]
    ranks = [_severity_rank(s) for s in ordered]

    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ordered)
    assert _severity_rank(SEVERITY_GATE_REMOVING) == 0


def test_severity_rank_fails_soft_on_an_unknown_severity(tmp_path):
    """An unknown severity sorts LAST rather than raising, mirroring
    _source_rank in audit_wiped_metadata_files.py."""
    assert _severity_rank("not-a-severity") > _severity_rank(SEVERITY_BENIGN)


def test_severity_constants_are_distinct_strings(tmp_path):
    """Module-level str constants, all distinct — no accidental aliasing."""
    constants = [SEVERITY_GATE_REMOVING, SEVERITY_PROVENANCE, SEVERITY_DISPATCH,
                 SEVERITY_INFORMATIONAL, SEVERITY_BENIGN]
    assert all(isinstance(c, str) for c in constants)
    assert len(set(constants)) == len(constants)


# ---------------------------------------------------------------------------
# audit_project — the join.
#
# Unions both comparison sources with PER-KEY provenance and emits one Finding
# per key that a source expected but live metadata no longer carries.
# ---------------------------------------------------------------------------

def _make_project(tmp_path, make_tasks_db, *, name="proj", tasks=(), tickets=None,
                  manifests=()):
    """Build a synthetic project root with tasks.db, tickets.db and manifests."""
    root = tmp_path / name
    (root / ".taskmaster" / "tasks").mkdir(parents=True, exist_ok=True)
    make_tasks_db(list(tasks), directory=root / ".taskmaster" / "tasks")
    if tickets is not None:
        (root / "data" / "reconciliation").mkdir(parents=True, exist_ok=True)
        _make_tickets_db(root / "data" / "reconciliation", list(tickets))
    for relpath, doc in manifests:
        _write_manifest(root, relpath, doc)
    return root


def _manifest_doc(task_id, label="δ", prd="plans/x-prd.md", checks=(("gate", _GREP_CHECK),)):
    return {
        "prd": prd,
        "schema_version": 1,
        "tasks": [{"label": label, "task_id": task_id,
                   "capabilities": [_capability(n, c) for n, c in checks]}],
    }


def _by_key(audit):
    return {f.key: f for f in audit.findings}


def test_audit_project_emits_one_finding_per_ticket_evidenced_lost_key(
        tmp_path, make_tasks_db):
    """(1) The ticket carried keys the live wipe signature no longer has."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[{"id": 100, "status": "pending", "metadata": _WIPE_SIGNATURE}],
        tickets=[{"project_id": "dark_factory", "task_id": 100, "metadata": {
            "task_kind": "deterministic", "source": "prd", "spawn_context": "prd_decompose",
        }}],
    )

    audit = audit_project(str(root), "dark_factory")

    assert isinstance(audit, ProjectAudit)
    assert set(_by_key(audit)) == {"task_kind", "source", "spawn_context"}
    assert all(f.expected_source == SOURCE_TICKET for f in audit.findings)
    assert all(isinstance(f, Finding) for f in audit.findings)
    assert _by_key(audit)["task_kind"].severity == SEVERITY_DISPATCH


def test_audit_project_finding_carries_the_full_record(tmp_path, make_tasks_db):
    """Every Finding names tag/task_id/status/key/severity/source/terminal."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[{"id": 100, "tag": "master", "status": "blocked",
                "metadata": _WIPE_SIGNATURE}],
        tickets=[{"project_id": "dark_factory", "task_id": 100,
                  "metadata": {"source": "prd"}}],
    )

    finding = audit_project(str(root), "dark_factory").findings[0]

    assert finding.tag == "master"
    assert finding.task_id == 100
    assert finding.status == "blocked"
    assert finding.key == "source"
    assert finding.severity == SEVERITY_INFORMATIONAL
    assert finding.expected_source == SOURCE_TICKET
    assert finding.terminal is False
    assert finding.reason


def test_audit_project_manifest_sourced_findings_without_a_ticket(tmp_path, make_tasks_db):
    """(2) THE 3157/3319 SHAPE: no ticket at all, but a manifest binds the task,
    so delivered_checks / prd_path / prd_task_label are still recoverable."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[{"id": 3157, "status": "pending", "metadata": _WIPE_SIGNATURE}],
        tickets=[],
        manifests=[("plans/m-prd.capability-manifest.yaml",
                    _manifest_doc(3157, label="δ", prd="plans/m-prd.md"))],
    )

    audit = audit_project(str(root), "dark_factory")
    by_key = _by_key(audit)

    assert set(by_key) == {"delivered_checks", "prd_path", "prd_task_label"}
    assert all(f.expected_source == SOURCE_MANIFEST for f in audit.findings)
    assert by_key["delivered_checks"].severity == SEVERITY_GATE_REMOVING
    assert by_key["prd_path"].severity == SEVERITY_PROVENANCE
    assert audit.coverage.targets_with_manifest == 1
    assert audit.coverage.targets_with_ticket == 0


def test_audit_project_manifest_with_no_mechanical_checks_expects_no_delivered_checks(
        tmp_path, make_tasks_db):
    """commit_planning stamps metadata.delivered_checks only when at least one
    MECHANICAL check exists (manifest_stamping.py: `if not mechanical: skip`).
    A manual-only manifest must therefore never yield a gate-removing finding."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[{"id": 42, "status": "pending", "metadata": _WIPE_SIGNATURE}],
        tickets=[],
        manifests=[("plans/n-prd.capability-manifest.yaml",
                    _manifest_doc(42, checks=(("manual-only", _MANUAL_CHECK),)))],
    )

    by_key = _by_key(audit_project(str(root), "dark_factory"))

    assert "delivered_checks" not in by_key
    assert set(by_key) == {"prd_path", "prd_task_label"}


def test_audit_project_unions_both_sources_with_per_key_provenance(
        tmp_path, make_tasks_db):
    """Both sources speak: the ticket wins shared keys, but the manifest's
    delivered_checks is KEPT (a submit payload rarely carries it)."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[{"id": 7, "status": "pending", "metadata": _WIPE_SIGNATURE}],
        tickets=[{"project_id": "dark_factory", "task_id": 7,
                  "metadata": {"source": "prd", "prd_path": "plans/from-ticket.md"}}],
        manifests=[("plans/o-prd.capability-manifest.yaml",
                    _manifest_doc(7, prd="plans/from-manifest.md"))],
    )

    by_key = _by_key(audit_project(str(root), "dark_factory"))

    assert set(by_key) == {"source", "prd_path", "prd_task_label", "delivered_checks"}
    assert by_key["delivered_checks"].expected_source == SOURCE_MANIFEST
    assert by_key["prd_path"].expected_source == SOURCE_TICKET
    assert by_key["prd_task_label"].expected_source == SOURCE_MANIFEST
    assert by_key["source"].expected_source == SOURCE_TICKET


def test_audit_project_emits_a_none_row_when_no_source_covers_a_target(
        tmp_path, make_tasks_db):
    """(3) NEVER SILENTLY SKIPPED. A combine target with neither a ticket nor a
    manifest is emitted with expected_source='none' and counted in coverage, so
    a partial sweep can never read as complete."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[{"id": 900, "status": "pending", "metadata": _WIPE_SIGNATURE}],
        tickets=[],
    )

    audit = audit_project(str(root), "dark_factory")

    assert len(audit.findings) == 1
    assert audit.findings[0].expected_source == SOURCE_NONE
    assert audit.findings[0].task_id == 900
    assert audit.coverage.targets_without_comparison_source == 1
    assert audit.coverage.total_combine_targets == 1


def test_audit_project_reports_key_loss_never_value_drift(tmp_path, make_tasks_db):
    """(4) A key PRESENT in live metadata is not reported even when its VALUE
    differs. The ticket is a submit-time snapshot, so comparing values would
    surface benign post-submit edits as findings."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[{"id": 8, "status": "pending", "metadata": {
            **_WIPE_SIGNATURE, "task_kind": "normal", "source": "agent-followup"}}],
        tickets=[{"project_id": "dark_factory", "task_id": 8, "metadata": {
            "task_kind": "deterministic", "source": "prd"}}],
    )

    assert audit_project(str(root), "dark_factory").findings == []


def test_audit_project_ignores_non_combine_tasks(tmp_path, make_tasks_db):
    """(5) A task the curator never combined is not a target, ticket or no."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[{"id": 9, "status": "pending", "metadata": {"source": "prd"}}],
        tickets=[{"project_id": "dark_factory", "task_id": 9,
                  "metadata": {"source": "prd", "task_kind": "deterministic"}}],
    )

    audit = audit_project(str(root), "dark_factory")

    assert audit.findings == []
    assert audit.coverage.total_combine_targets == 0


def test_audit_project_sorts_gate_removing_first_then_numeric_task_id(
        tmp_path, make_tasks_db):
    """Gate-removing rows come FIRST regardless of task id, and within a
    severity 100 follows 20 rather than preceding it (numeric, not lexical)."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[
            {"id": 20, "status": "pending", "metadata": _WIPE_SIGNATURE},
            {"id": 100, "status": "pending", "metadata": _WIPE_SIGNATURE},
            {"id": 300, "status": "pending", "metadata": _WIPE_SIGNATURE},
        ],
        tickets=[
            {"project_id": "dark_factory", "task_id": 20, "metadata": {"source": "prd"}},
            {"project_id": "dark_factory", "task_id": 100, "metadata": {"source": "prd"}},
        ],
        manifests=[("plans/p-prd.capability-manifest.yaml", _manifest_doc(300))],
    )

    findings = audit_project(str(root), "dark_factory").findings

    assert findings[0].severity == SEVERITY_GATE_REMOVING
    assert findings[0].task_id == 300
    informational = [f.task_id for f in findings if f.key == "source"]
    assert informational == [20, 100]


def test_audit_project_terminal_flag_tracks_status(tmp_path, make_tasks_db):
    """done/cancelled are terminal; pending/in-progress/blocked are live."""
    assert set(TERMINAL_STATUSES) == {"done", "cancelled"}
    statuses = {1: "done", 2: "cancelled", 3: "pending", 4: "in-progress", 5: "blocked"}
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[{"id": i, "status": s, "metadata": _WIPE_SIGNATURE}
               for i, s in statuses.items()],
        tickets=[{"project_id": "dark_factory", "task_id": i, "metadata": {"source": "prd"}}
                 for i in statuses],
    )

    terminal = {f.task_id: f.terminal for f in audit_project(str(root), "dark_factory").findings}

    assert terminal == {1: True, 2: True, 3: False, 4: False, 5: False}


def test_audit_project_coverage_counts_and_parse_failures(tmp_path, make_tasks_db):
    """AuditCoverage names every population the report leans on."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[
            {"id": 1, "status": "done", "metadata": _WIPE_SIGNATURE},
            {"id": 2, "status": "done", "metadata": _WIPE_SIGNATURE},
            {"id": 3, "status": "done", "metadata": _WIPE_SIGNATURE},
            {"id": 4, "status": "done", "metadata": {"source": "prd"}},
        ],
        tickets=[{"project_id": "dark_factory", "task_id": 1, "metadata": {"source": "prd"}}],
        manifests=[
            ("plans/q-prd.capability-manifest.yaml", _manifest_doc(2)),
            ("plans/broken-prd.capability-manifest.yaml", "prd: [unclosed\n  - nope"),
        ],
    )

    coverage = audit_project(str(root), "dark_factory").coverage

    assert isinstance(coverage, AuditCoverage)
    assert coverage.total_combine_targets == 3
    assert coverage.targets_with_ticket == 1
    assert coverage.targets_with_manifest == 1
    assert coverage.targets_without_comparison_source == 1
    assert coverage.manifest_parse_failures == 1


def test_audit_project_tolerates_a_missing_tickets_db(tmp_path, make_tasks_db):
    """No tickets.db at all degrades to manifest-only, never a raise."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[{"id": 11, "status": "pending", "metadata": _WIPE_SIGNATURE}],
        manifests=[("plans/r-prd.capability-manifest.yaml", _manifest_doc(11))],
    )

    assert not (root / "data" / "reconciliation" / "tickets.db").exists()
    assert set(_by_key(audit_project(str(root), "dark_factory"))) == {
        "delivered_checks", "prd_path", "prd_task_label"}


# ---------------------------------------------------------------------------
# format_report / format_json — the MANDATORY honesty blocks.
#
# Both blocks print even for a zero-finding audit. A clean-looking report that
# omitted them would let a sweep which could see almost nothing read as a
# clean bill of health — the no-silent-fail-soft violation named in
# docs/legibility/design-invariants.md.
#
# The tests pin the module-level CONSTANTS rather than re-stating their prose,
# so an editor improving the wording updates one place and the tests follow.
# ---------------------------------------------------------------------------

def _audit(findings=(), coverage=None, root="/tmp/proj", project_id="dark_factory"):
    return ProjectAudit(
        project_root=root,
        project_id=project_id,
        findings=list(findings),
        coverage=coverage or AuditCoverage(0, 0, 0, 0, 0),
    )


def _finding(**kw):
    base = dict(tag="master", task_id=1, status="pending", key="source",
                severity=SEVERITY_INFORMATIONAL, expected_source=SOURCE_TICKET,
                terminal=False, reason="because")
    base.update(kw)
    return Finding(**base)


def test_format_report_always_emits_both_caveats_even_with_zero_findings():
    """The whole point: an empty finding list is NOT a clean bill of health."""
    report = format_report([_audit()])

    assert _FIDELITY_CAVEAT in report
    assert _COVERAGE_CAVEAT in report


def test_fidelity_caveat_states_the_submit_time_bound_and_measured_agreement():
    """The constant carries the evidence, so no caller can paraphrase it away."""
    assert "submit" in _FIDELITY_CAVEAT.lower()
    assert "367/368" in _FIDELITY_CAVEAT      # task_kind agreement
    assert "368/368" in _FIDELITY_CAVEAT      # source agreement
    assert "3295" in _FIDELITY_CAVEAT         # the mismatch proving mutability
    # Case-insensitive: the constant SHOUTS the label on purpose.
    assert "ticket-evidenced" in _FIDELITY_CAVEAT.lower()


def test_coverage_caveat_names_the_uncoverable_population():
    assert "comparison source" in _COVERAGE_CAVEAT.lower()


def test_format_report_names_the_no_comparison_source_count():
    """COVERAGE prints targets_without_comparison_source as a number."""
    report = format_report([_audit(coverage=AuditCoverage(91, 66, 40, 25, 2))])

    assert "25" in report
    assert "91" in report
    assert "2" in report


def test_format_report_does_not_call_ticket_evidenced_on_manifest_findings():
    """A manifest-sourced finding rests on the sidecar, not on a ticket. The
    per-finding line must say so rather than inheriting the ticket label."""
    report = format_report([_audit(findings=[
        _finding(task_id=3157, key="delivered_checks", severity=SEVERITY_GATE_REMOVING,
                 expected_source=SOURCE_MANIFEST),
    ])])

    line = next(ln for ln in report.splitlines() if "3157" in ln)
    assert f"source={SOURCE_MANIFEST}" in line
    assert "ticket-evidenced" not in line


def test_format_report_renders_live_before_terminal():
    """The live section comes FIRST: those are the rows an operator can act on."""
    report = format_report([_audit(findings=[
        _finding(task_id=11, terminal=False),
        _finding(task_id=22, terminal=True, status="done"),
    ])])

    assert report.index("11") < report.index("22")
    lowered = report.lower()
    assert lowered.index("live") < lowered.index("terminal")


def test_format_report_renders_terminal_findings_in_full():
    """Terminal findings are suppressed from the EXIT CODE, never the report."""
    report = format_report([_audit(findings=[
        _finding(task_id=22, terminal=True, status="done", key="task_kind",
                 severity=SEVERITY_BENIGN),
    ])])

    assert "22" in report
    assert "task_kind" in report


def test_format_report_prints_the_benign_reason():
    """A benign demotion prints WHY, so a reader never takes it on trust."""
    report = format_report([_audit(findings=[
        _finding(key="task_kind", severity=SEVERITY_BENIGN,
                 reason="absent is identical to 'normal'"),
    ])])

    assert "absent is identical to 'normal'" in report


def test_format_report_groups_by_project_root():
    report = format_report([_audit(root="/tmp/a"), _audit(root="/tmp/b")])

    assert "/tmp/a" in report and "/tmp/b" in report
    assert report.index("/tmp/a") < report.index("/tmp/b")


def test_format_json_returns_an_object_carrying_coverage():
    """An OBJECT, never a bare array, so coverage travels with the findings."""
    payload = json.loads(format_json([_audit(
        findings=[_finding(task_id=5)],
        coverage=AuditCoverage(91, 66, 40, 25, 2),
    )]))

    assert isinstance(payload, dict)
    project = payload["projects"][0]
    assert project["project_root"] == "/tmp/proj"
    assert project["coverage"]["targets_without_comparison_source"] == 25
    assert project["findings"][0]["task_id"] == 5
    assert project["findings"][0]["expected_source"] == SOURCE_TICKET


def test_format_json_carries_the_fidelity_caveat():
    """The honesty block travels to machine consumers too."""
    payload = json.loads(format_json([_audit()]))

    assert _FIDELITY_CAVEAT in json.dumps(payload)


def test_formatters_are_pure_and_print_nothing(capsys):
    """Both return str and neither prints — main() does the single print."""
    assert isinstance(format_report([_audit()]), str)
    assert isinstance(format_json([_audit()]), str)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# resolve_project_id + the project-id mismatch guard.
#
# THIS IS TRAP (a) INVERTED, and the most dangerous failure mode this detector
# has. project_id is derived heuristically from the directory name. A WRONG
# derivation matches zero ticket rows, which makes every combine target look
# totally wiped and produces a catastrophic wall of confident false positives
# that an operator would read as a real incident. Failing loudly instead is
# the no-silent-fail-soft rule in docs/legibility/design-invariants.md.
# ---------------------------------------------------------------------------

def test_resolve_project_id_derives_from_the_directory_name(tmp_path):
    """dark-factory -> dark_factory: '-' maps to '_'."""
    assert resolve_project_id("/home/leo/src/dark-factory") == "dark_factory"
    assert resolve_project_id("/x/y/reify") == "reify"
    assert resolve_project_id("/x/y/a-b-c") == "a_b_c"


def test_resolve_project_id_override_wins(tmp_path):
    assert resolve_project_id("/home/leo/src/dark-factory", override="reify") == "reify"


def test_audit_project_flags_a_project_id_matching_zero_of_many(tmp_path, make_tasks_db):
    """The guard fires: the resolved id matches NOTHING in a tickets.db that
    plainly holds created-rows for other project_ids."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[{"id": 1, "status": "pending", "metadata": _WIPE_SIGNATURE}],
        tickets=[
            {"project_id": "reify", "task_id": 1, "metadata": {"source": "prd"}},
            {"project_id": "other_project", "task_id": 2, "metadata": {"source": "prd"}},
        ],
    )

    audit = audit_project(str(root), "dark_factory")

    assert audit.project_id_mismatch is not None
    # Names the resolved id AND the ids actually present, so an operator can
    # see the misconfiguration rather than guessing at a wall of findings.
    assert "dark_factory" in audit.project_id_mismatch
    assert "reify" in audit.project_id_mismatch
    assert "other_project" in audit.project_id_mismatch


def test_audit_project_guard_is_inert_when_the_id_matches(tmp_path, make_tasks_db):
    """A correct id, even alongside other projects' rows, never fires."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[{"id": 1, "status": "pending", "metadata": _WIPE_SIGNATURE}],
        tickets=[
            {"project_id": "reify", "task_id": 1, "metadata": {"source": "prd"}},
            {"project_id": "dark_factory", "task_id": 1, "metadata": {"source": "prd"}},
        ],
    )

    assert audit_project(str(root), "dark_factory").project_id_mismatch is None


def test_audit_project_guard_is_inert_when_tickets_db_is_absent(tmp_path, make_tasks_db):
    """An absent tickets.db is honest 'no comparison source', NOT a
    misconfiguration — firing here would cry wolf on every fresh project."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[{"id": 1, "status": "pending", "metadata": _WIPE_SIGNATURE}],
    )

    audit = audit_project(str(root), "dark_factory")

    assert audit.project_id_mismatch is None
    assert audit.coverage.targets_without_comparison_source == 1


def test_audit_project_guard_is_inert_on_a_genuinely_empty_tickets_db(
        tmp_path, make_tasks_db):
    """A tickets.db with no created-rows at all is likewise honest emptiness."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[{"id": 1, "status": "pending", "metadata": _WIPE_SIGNATURE}],
        tickets=[{"project_id": "reify", "task_id": 1, "status": "pending",
                  "metadata": {"source": "prd"}}],
    )

    assert audit_project(str(root), "dark_factory").project_id_mismatch is None


def test_audit_project_mismatch_reaches_the_report_and_the_json(tmp_path, make_tasks_db):
    """The signal must reach the CALLER so main() can exit non-zero rather
    than printing a clean-looking wall of findings."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[{"id": 1, "status": "pending", "metadata": _WIPE_SIGNATURE}],
        tickets=[{"project_id": "reify", "task_id": 1, "metadata": {"source": "prd"}}],
    )

    audit = audit_project(str(root), "dark_factory")

    assert "dark_factory" in format_report([audit])
    assert audit.project_id_mismatch in format_report([audit])
    payload = json.loads(format_json([audit]))
    assert payload["projects"][0]["project_id_mismatch"] == audit.project_id_mismatch


# ---------------------------------------------------------------------------
# main() — CLI contract, exercised by SUBPROCESS.
#
# Pure functions get direct pytest coverage above; main() gets subprocess
# coverage, following the precedent's stated split. Shelling out to the script
# path is also what proves the flat-sibling `_task_db_scan` import resolves for
# a DIRECTLY-EXECUTED script (never `python -m`).
#   exit 0 = no live findings; 1 = a live finding or a project_id mismatch;
#   2 = no root resolved; 3 = roots resolved but NOTHING was scanned.
# ---------------------------------------------------------------------------

_SCRIPT = str(Path(__file__).parent.parent / "audit_combine_gate_marker_loss.py")


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, _SCRIPT, *args], capture_output=True, text=True
    )


def test_main_exit_0_when_every_finding_is_terminal(tmp_path, make_tasks_db):
    """THE DOCUMENTED STEADY STATE. The ~67 terminal combine targets were
    deliberately left un-remediated; if they drove the exit code this detector
    would be permanently red and therefore ignored."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[{"id": 50, "status": "done", "metadata": _WIPE_SIGNATURE}],
        tickets=[{"project_id": "dark_factory", "task_id": 50,
                  "metadata": {"source": "prd", "task_kind": "normal"}}],
    )

    result = _run_cli("--project-root", str(root), "--project-id", "dark_factory")

    assert result.returncode == 0
    assert "50" in result.stdout          # still REPORTED in full
    assert "COVERAGE" in result.stdout
    assert "FIDELITY" in result.stdout


def test_main_exit_1_and_ranks_a_live_delivered_checks_loss_first(
        tmp_path, make_tasks_db):
    """The user-observable signal from the task description: a NON-TERMINAL
    delivered_checks loss exits 1 and is ranked FIRST in stdout."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[
            {"id": 60, "status": "pending", "metadata": _WIPE_SIGNATURE},
            {"id": 61, "status": "pending", "metadata": _WIPE_SIGNATURE},
        ],
        tickets=[{"project_id": "dark_factory", "task_id": 61,
                  "metadata": {"source": "prd"}}],
        manifests=[("plans/s-prd.capability-manifest.yaml", _manifest_doc(60))],
    )

    result = _run_cli("--project-root", str(root), "--project-id", "dark_factory")

    assert result.returncode == 1
    first = next(ln for ln in result.stdout.splitlines() if "task_id=" in ln)
    assert "key=delivered_checks" in first
    assert f"severity={SEVERITY_GATE_REMOVING}" in first


def test_main_exit_2_when_no_project_root_resolves(tmp_path):
    """Nothing resolvable -> exit 2, empty stdout, reason on stderr."""
    result = _run_cli("--project-root", str(tmp_path / "no-such-project"))

    assert result.returncode == 2
    assert result.stdout.strip() == ""
    assert "no project root" in result.stderr.lower()


def test_main_exit_3_when_roots_resolve_but_nothing_is_scanned(tmp_path):
    """A resolvable-but-unreadable tasks.db is NOT a clean run. Exit 3 keeps it
    distinct from 0, per docs/legibility/design-invariants.md."""
    root = tmp_path / "corrupt"
    (root / ".taskmaster" / "tasks").mkdir(parents=True)
    (root / ".taskmaster" / "tasks" / "tasks.db").write_text("this is not a database")

    result = _run_cli("--project-root", str(root))

    assert result.returncode == 3
    assert "not a clean result" in result.stderr.lower()


def test_main_exit_1_on_a_project_id_mismatch(tmp_path, make_tasks_db):
    """A mis-derived project_id must never print a clean-looking wall."""
    root = _make_project(
        tmp_path, make_tasks_db, name="wrongly-named",
        tasks=[{"id": 70, "status": "done", "metadata": _WIPE_SIGNATURE}],
        tickets=[{"project_id": "reify", "task_id": 70, "metadata": {"source": "prd"}}],
    )

    result = _run_cli("--project-root", str(root))

    assert result.returncode == 1
    assert "PROJECT_ID MISMATCH" in result.stdout


def test_main_json_flag_carries_coverage(tmp_path, make_tasks_db):
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[{"id": 80, "status": "pending", "metadata": _WIPE_SIGNATURE}],
        tickets=[{"project_id": "dark_factory", "task_id": 80,
                  "metadata": {"source": "prd"}}],
    )

    result = _run_cli("--project-root", str(root), "--project-id", "dark_factory", "--json")

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    project = payload["projects"][0]
    assert project["coverage"]["total_combine_targets"] == 1
    assert project["findings"][0]["task_id"] == 80


def test_main_project_root_is_repeatable(tmp_path, make_tasks_db):
    a = _make_project(tmp_path, make_tasks_db, name="a",
                      tasks=[{"id": 1, "status": "done", "metadata": _WIPE_SIGNATURE}])
    b = _make_project(tmp_path, make_tasks_db, name="b",
                      tasks=[{"id": 2, "status": "done", "metadata": _WIPE_SIGNATURE}])

    result = _run_cli("--project-root", str(a), "--project-root", str(b), "--json")

    payload = json.loads(result.stdout)
    assert [p["project_root"] for p in payload["projects"]] == [str(a), str(b)]


def test_main_project_id_overrides_derivation(tmp_path, make_tasks_db):
    """--project-id wins over the directory-name derivation."""
    root = _make_project(
        tmp_path, make_tasks_db, name="wrongly-named",
        tasks=[{"id": 90, "status": "pending", "metadata": _WIPE_SIGNATURE}],
        tickets=[{"project_id": "reify", "task_id": 90, "metadata": {"source": "prd"}}],
    )

    result = _run_cli("--project-root", str(root), "--project-id", "reify", "--json")

    payload = json.loads(result.stdout)
    assert payload["projects"][0]["project_id"] == "reify"
    assert payload["projects"][0]["project_id_mismatch"] is None
    assert payload["projects"][0]["findings"][0]["key"] == "source"


def test_main_report_on_stdout_warnings_on_stderr(tmp_path, make_tasks_db):
    """The report is stdout; every warning is stderr, so a pipe stays clean."""
    good = _make_project(tmp_path, make_tasks_db, name="good",
                         tasks=[{"id": 1, "status": "done", "metadata": _WIPE_SIGNATURE}])
    bad = tmp_path / "bad"
    (bad / ".taskmaster" / "tasks").mkdir(parents=True)
    (bad / ".taskmaster" / "tasks" / "tasks.db").write_text("not a database")

    result = _run_cli("--project-root", str(good), "--project-root", str(bad), "--json")

    assert json.loads(result.stdout)          # stdout is pure payload
    assert "warning" in result.stderr.lower()


def test_main_run_is_strictly_read_only(tmp_path, make_tasks_db):
    """THE READ-ONLY CLAIM, CHECKED. Every input database's mtime AND sha256
    are captured before and after a full run and must be unchanged."""
    import hashlib

    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[{"id": 95, "status": "pending", "metadata": _WIPE_SIGNATURE}],
        tickets=[{"project_id": "dark_factory", "task_id": 95,
                  "metadata": {"source": "prd"}}],
        manifests=[("plans/t-prd.capability-manifest.yaml", _manifest_doc(95))],
    )
    inputs = [
        root / ".taskmaster" / "tasks" / "tasks.db",
        root / "data" / "reconciliation" / "tickets.db",
        root / "plans" / "t-prd.capability-manifest.yaml",
    ]

    def fingerprint():
        return {
            str(p): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in inputs
        }

    before = fingerprint()
    _run_cli("--project-root", str(root), "--project-id", "dark_factory")

    assert fingerprint() == before
