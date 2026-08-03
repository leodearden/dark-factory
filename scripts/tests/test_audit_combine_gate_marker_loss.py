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
    SEVERITY_BENIGN,
    SEVERITY_DISPATCH,
    SEVERITY_GATE_REMOVING,
    SEVERITY_INFORMATIONAL,
    SEVERITY_PROVENANCE,
    CombineTarget,
    ManifestExpectation,
    _severity_rank,
    build_manifest_index,
    classify_gap,
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
