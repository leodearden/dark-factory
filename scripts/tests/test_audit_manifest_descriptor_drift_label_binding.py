"""Tests for load_task_store_scan, the one tasks.db read both directions consume,
and for the label-binding direction of scripts/audit_manifest_descriptor_drift.py.

Task 4907 added the second direction, task -> sidecar. A manifest-bearing task
whose ``prd_task_label`` its tracked sidecar does not declare is an UNBOUND
LABEL: the stamper binds nothing for it and copies it no delivered_checks. Those
rows are reported as their own list, never mixed into the drift findings.
"""
from __future__ import annotations

import pytest
from audit_manifest_descriptor_drift import (
    ManifestBinding,
    TaskStoreScan,
    UnboundLabel,
    audit_project,
    format_report,
    load_task_store_scan,
)
from manifest_drift_fixtures import (
    GREP_CHECK,
    SCRIPT_CHECK,
    check_entry,
    declaring_sidecar,
    labelled_row,
    make_project,
    manifest_doc,
    one_project,
    task_row,
    write_manifest,
)
from shared.task_statuses import TERMINAL, TaskStatus

# ---------------------------------------------------------------------------
# load_task_store_scan — ONE read of tasks.db feeds BOTH directions.
#
# The drift direction needs every row id and each task's delivered_checks; the
# label-binding direction needs each manifest-bearing task's status, label and
# derived sidecar. All three come back from one scan in one TaskStoreScan
# record, so the two halves of a report can never straddle a write the
# orchestrator makes to the live store in between.
# ---------------------------------------------------------------------------

def test_scan_binds_a_manifest_bearing_task_to_its_derived_sidecar(make_tasks_db):
    """The binding carries what the label-binding direction reports: the task,
    its status, its label verbatim, and the sidecar the stamper would open."""
    db = make_tasks_db([labelled_row(7, status="pending", label="γ1")])

    scan = load_task_store_scan(str(db))

    assert isinstance(scan, TaskStoreScan)
    assert scan.manifest_bindings == (
        ManifestBinding(task_id=7, status="pending", label="γ1",
                        manifest="plans/x-prd.capability-manifest.yaml"),
    )


@pytest.mark.parametrize("prd_path,derived", [
    ("plans/x-prd.md", "plans/x-prd.capability-manifest.yaml"),
    # No `.md` suffix: the stamper's re.sub strips nothing and just appends.
    ("plans/x-prd", "plans/x-prd.capability-manifest.yaml"),
], ids=["md-suffix", "no-md-suffix"])
def test_scan_derives_the_sidecar_by_the_stampers_rule(make_tasks_db, prd_path, derived):
    """``re.sub(r'\\.md$', '', prd_path) + '.capability-manifest.yaml'`` —
    manifest_stamping.py::_stamp_capability_manifests_impl step 1, exactly. The
    scan stops there: resolving that path against the project root is step 2,
    which the sweep applies (see
    test_the_derived_sidecar_is_resolved_against_the_root_as_the_stamper_does)."""
    db = make_tasks_db([labelled_row(7, prd_path=prd_path)])

    (binding,) = load_task_store_scan(str(db)).manifest_bindings

    assert binding.manifest == derived


@pytest.mark.parametrize("metadata", [
    {"prd_task_label": "α"},
    {"prd_path": "plans/x-prd.md"},
    {"prd_path": "", "prd_task_label": "α"},
    {"prd_path": "plans/x-prd.md", "prd_task_label": ""},
    {"prd_path": None, "prd_task_label": "α"},
    {"prd_path": "plans/x-prd.md", "prd_task_label": None},
], ids=["no-path", "no-label", "empty-path", "empty-label", "null-path", "null-label"])
def test_scan_mirrors_the_stampers_falsy_admission_gate(make_tasks_db, metadata):
    """``if not prd_path or not prd_task_label: continue`` — the stamper's step 1.

    A task the stamper never admits is promised nothing, so it can never be an
    unbound label. Mirroring the gate exactly is what makes the sweep's
    population the stamper's population.
    """
    db = make_tasks_db([{"id": 7, "status": "pending", "metadata": metadata}])

    assert load_task_store_scan(str(db)).manifest_bindings == ()


@pytest.mark.parametrize("metadata", [
    {"prd_path": 5, "prd_task_label": "α"},
    {"prd_path": ["plans/x-prd.md"], "prd_task_label": "α"},
    {"prd_path": "plans/x-prd.md", "prd_task_label": 1},
    {"prd_path": "plans/x-prd.md", "prd_task_label": ["α"]},
    {"prd_path": "plans/x-prd.md", "prd_task_label": True},
], ids=["int-path", "list-path", "int-label", "list-label", "bool-label"])
def test_scan_admits_no_truthy_non_string_key(make_tasks_db, metadata):
    """THE ONE DELIBERATE DIVERGENCE from the stamper, which would admit these.

    It would then fail to derive a sidecar path from a non-string prd_path, or
    never match a non-string label against the sidecar's string labels — so
    such a row can neither be bound nor usefully reported. No live row has this
    shape.
    """
    db = make_tasks_db([{"id": 7, "status": "pending", "metadata": metadata}])

    assert load_task_store_scan(str(db)).manifest_bindings == ()


@pytest.mark.parametrize("raw", [
    "{not json",
    None,
    '["plans/x-prd.md", "α"]',
    '"plans/x-prd.md"',
], ids=["malformed-json", "null", "json-list", "json-string"])
def test_undecodable_metadata_yields_no_binding_and_the_scan_continues(
        make_tasks_db, raw):
    """A corrupt metadata blob is a row to skip, never a reason to abort a
    sweep over thousands of tasks — and the rows after it are still bound."""
    db = make_tasks_db([
        {"id": 1, "status": "pending", "metadata": raw},
        labelled_row(2),
    ])

    scan = load_task_store_scan(str(db))

    assert [b.task_id for b in scan.manifest_bindings] == [2]
    assert scan.row_ids == {1, 2}


def test_bindings_come_back_in_numeric_task_id_order(make_tasks_db):
    """Numeric, not insertion or lexicographic order ("200" < "30" < "4"): the
    report built from these rows must diff cleanly between runs."""
    db = make_tasks_db([labelled_row(30), labelled_row(200), labelled_row(4)])

    bindings = load_task_store_scan(str(db)).manifest_bindings

    assert [b.task_id for b in bindings] == [4, 30, 200]


def test_widening_the_loader_changed_neither_existing_output(make_tasks_db):
    """The row-id set and the delivered_checks mapping the drift direction reads
    are exactly what they were before the loader also collected bindings.

    A task with no metadata at all is still a ROW (so a sidecar binding it is
    not "without a db row"), and still has no delivered_checks. A task that
    carries both a binding and delivered_checks feeds both outputs from its one
    decoded metadata. A row under another tag reaches none of the three: the
    ``tag = 'master'`` pin covers the bindings too.
    """
    gate = check_entry("gate", GREP_CHECK)
    cap = check_entry("cap", SCRIPT_CHECK)
    db = make_tasks_db([
        {"id": 1, "status": "pending", "metadata": None},
        {"id": 2, "status": "done", "metadata": {"delivered_checks": [gate]}},
        labelled_row(3, status="done", delivered_checks=[cap, "not-a-dict", {"name": 5}]),
        {**labelled_row(4, delivered_checks=[gate]), "tag": "other-tag"},
    ])

    scan = load_task_store_scan(str(db))

    assert scan.row_ids == {1, 2, 3}
    assert scan.delivered_checks == {2: {"gate": gate}, 3: {"cap": cap}}
    assert [b.task_id for b in scan.manifest_bindings] == [3]


# ---------------------------------------------------------------------------
# THE LABEL-BINDING DIRECTION — task -> sidecar (task 4907).
#
# The drift walk is keyed on each sidecar's STAMPED task_id, so it cannot see a
# task whose prd_task_label matches no sidecar entry at all. This direction
# walks the other way, from every task the stamper would admit to the sidecar
# it would open, and lists each label that sidecar does not declare in
# `unbound_labels`, never in `findings`.
# ---------------------------------------------------------------------------

def _unbound_pairs(audit) -> list[tuple[int, str]]:
    return [(row.task_id, row.label) for row in audit.unbound_labels]


def test_a_label_its_tracked_sidecar_does_not_declare_is_one_unbound_row(
        tmp_path, make_tasks_db):
    """THE ROW SHAPE. It carries the labels the sidecar DOES declare, in the
    sidecar's own order, so a reader can act without opening the file."""
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[labelled_row(7, status="pending", label="ω")],
        manifests=[("plans/x-prd.capability-manifest.yaml", declaring_sidecar("η0", "α"))],
    )

    audit = audit_project(str(root))

    assert audit.unbound_labels == [UnboundLabel(
        task_id=7, label="ω", status="pending",
        manifest="plans/x-prd.capability-manifest.yaml",
        declared_labels=("η0", "α"),
    )]
    assert audit.findings == []


def test_a_declared_label_is_bound_whatever_its_entry_task_id_says(
        tmp_path, make_tasks_db):
    """The stamper matches on LABEL, not task_id (manifest_stamping step 4).

    So a declared label is bound while its entry is still unstamped
    (``task_id: null``), and even while the entry carries another task's id.
    """
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[labelled_row(7, label="α"), labelled_row(8, label="β")],
        manifests=[("plans/x-prd.capability-manifest.yaml", {
            "prd": "plans/x-prd.md", "schema_version": 1,
            "tasks": [{"label": "α", "task_id": None, "capabilities": []},
                      {"label": "β", "task_id": 5, "capabilities": []}],
        })],
    )

    assert audit_project(str(root)).unbound_labels == []


def test_a_task_whose_derived_sidecar_is_not_tracked_is_no_row(
        tmp_path, make_tasks_db):
    """No sidecar, no promise. The stamper's step 2 opens only a sidecar that
    exists, so a task whose derived sidecar is absent is a complete no-op. An
    UNTRACKED file is not part of the corpus this sweep reads (see
    _tracked_manifest_paths). Neither case is an unbound label."""
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[labelled_row(7, prd_path="plans/absent-prd.md", label="α"),
               labelled_row(8, prd_path="plans/untracked-prd.md", label="α")],
    )
    write_manifest(root, "plans/untracked-prd.capability-manifest.yaml",
                    declaring_sidecar("β", prd="plans/untracked-prd.md"))

    assert audit_project(str(root)).unbound_labels == []


@pytest.mark.parametrize("spell_prd_path", [
    pytest.param(lambda root: "./plans/x-prd.md", id="dot-slash"),
    pytest.param(lambda root: str(root / "plans" / "x-prd.md"), id="absolute-in-root"),
    pytest.param(lambda root: f"../{root.name}/plans/x-prd.md", id="re-entering-dotdot"),
])
def test_the_derived_sidecar_is_resolved_against_the_root_as_the_stamper_does(
        tmp_path, make_tasks_db, spell_prd_path):
    """The stamper's step 2 resolves the derived path against the project root
    before it opens anything, so every in-root spelling reaches the one tracked
    sidecar. Matching the raw string instead would count these tasks as having
    no tracked sidecar: a miss that nothing reports."""
    root = tmp_path / "proj"
    prd_path = spell_prd_path(root)
    make_project(
        tmp_path, make_tasks_db,
        tasks=[labelled_row(7, prd_path=prd_path, label="ω"),
               labelled_row(8, prd_path=prd_path, label="α")],
        manifests=[("plans/x-prd.capability-manifest.yaml", declaring_sidecar("α"))],
    )

    audit = audit_project(str(root))

    assert [(row.manifest, row.task_id) for row in audit.unbound_labels] == [
        ("plans/x-prd.capability-manifest.yaml", 7)]
    assert audit.coverage.tasks_bound_to_a_declared_label == 1
    assert audit.coverage.tasks_without_a_tracked_sidecar == 0


def test_a_derived_sidecar_outside_the_root_is_no_row(tmp_path, make_tasks_db):
    """The stamper refuses to read a sidecar that resolves outside its project
    root, so such a task is promised nothing: the same no-op as an absent
    sidecar, even when a sidecar really exists out there."""
    elsewhere = tmp_path / "elsewhere"
    write_manifest(elsewhere, "plans/x-prd.capability-manifest.yaml", declaring_sidecar("α"))
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[labelled_row(7, prd_path=str(elsewhere / "plans" / "x-prd.md"), label="ω"),
               labelled_row(8, prd_path="../elsewhere/plans/x-prd.md", label="ω")],
        manifests=[("plans/x-prd.capability-manifest.yaml", declaring_sidecar("α"))],
    )

    audit = audit_project(str(root))

    assert audit.unbound_labels == []
    assert audit.coverage.tasks_without_a_tracked_sidecar == 2


def test_an_unresolvable_sidecar_path_is_no_row_and_the_sweep_continues(
        tmp_path, make_tasks_db):
    """An embedded NUL makes the derived path unresolvable, so the stamper can
    open nothing for it. One such row must never abort a sweep over thousands."""
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[labelled_row(7, prd_path="plans/x\x00-prd.md", label="ω"),
               labelled_row(8, label="ω")],
        manifests=[("plans/x-prd.capability-manifest.yaml", declaring_sidecar("α"))],
    )

    audit = audit_project(str(root))

    assert _unbound_pairs(audit) == [(8, "ω")]
    assert audit.coverage.tasks_without_a_tracked_sidecar == 1


def test_label_matching_is_exact_so_a_transliteration_is_a_row(
        tmp_path, make_tasks_db):
    """'gamma-1' is not 'γ1'. This is task 4590's defect shape, and the reason
    this direction exists: the stamper compares the two strings exactly, so a
    transliterated label binds nothing, and after its one planning-time
    ``missing_labels`` entry nothing says so again."""
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[labelled_row(7, label="gamma-1")],
        manifests=[("plans/x-prd.capability-manifest.yaml", declaring_sidecar("γ1"))],
    )

    assert _unbound_pairs(audit_project(str(root))) == [(7, "gamma-1")]


def test_two_undeclared_labels_on_one_sidecar_are_two_rows(tmp_path, make_tasks_db):
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[labelled_row(7, label="ω"), labelled_row(8, label="ψ")],
        manifests=[("plans/x-prd.capability-manifest.yaml", declaring_sidecar("α"))],
    )

    assert _unbound_pairs(audit_project(str(root))) == [(7, "ω"), (8, "ψ")]


def test_unbound_rows_sort_by_manifest_then_numeric_task_id(tmp_path, make_tasks_db):
    """Manifest relpath first, then NUMERIC task id ("30" < "4" as strings), so
    a report built from these rows diffs cleanly between runs."""
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[labelled_row(30, prd_path="plans/b-prd.md", label="ω"),
               labelled_row(4, prd_path="plans/b-prd.md", label="ψ"),
               labelled_row(200, prd_path="plans/a-prd.md", label="ω")],
        manifests=[
            ("plans/b-prd.capability-manifest.yaml", declaring_sidecar("α", prd="plans/b-prd.md")),
            ("plans/a-prd.capability-manifest.yaml", declaring_sidecar("α", prd="plans/a-prd.md")),
        ],
    )

    rows = audit_project(str(root)).unbound_labels

    assert [(row.manifest, row.task_id) for row in rows] == [
        ("plans/a-prd.capability-manifest.yaml", 200),
        ("plans/b-prd.capability-manifest.yaml", 4),
        ("plans/b-prd.capability-manifest.yaml", 30),
    ]


# The drift dimension's coverage fields: the whole of AuditCoverage as it stood
# before the label-binding direction was added.
_DRIFT_COVERAGE_FIELDS = (
    "manifests_swept",
    "mechanical_capabilities_seen",
    "mechanical_capabilities_compared",
    "capabilities_without_task_entry",
    "task_entries_with_no_sidecar_capability",
    "manifest_tasks_without_db_row",
    "malformed_task_entries",
    "unconvertible_sidecar_descriptors",
    "manifest_parse_failures",
    "manifest_parse_failure_details",
    "uncomparable_details",
    "git_discovery_failed",
)


def test_unbound_rows_leave_the_drift_findings_and_coverage_untouched(
        tmp_path, make_tasks_db):
    """Two dimensions, reported separately. Adding tasks whose labels bind
    nothing changes neither the drift findings nor any drift counter."""
    drifted = [check_entry("gate", {**GREP_CHECK, "pattern": "def bar"})]
    manifests = [("plans/a-prd.capability-manifest.yaml",
                  manifest_doc(100, prd="plans/a-prd.md"))]
    baseline = audit_project(str(make_project(
        tmp_path, make_tasks_db, name="baseline",
        tasks=[task_row(100, drifted)], manifests=manifests)))
    with_unbound = audit_project(str(make_project(
        tmp_path, make_tasks_db, name="with-unbound",
        tasks=[task_row(100, drifted),
               labelled_row(200, prd_path="plans/a-prd.md", label="ω"),
               labelled_row(201, prd_path="plans/a-prd.md", label="ψ")],
        manifests=manifests)))

    assert _unbound_pairs(with_unbound) == [(200, "ω"), (201, "ψ")]
    assert baseline.findings != []
    assert with_unbound.findings == baseline.findings
    for field in _DRIFT_COVERAGE_FIELDS:
        assert getattr(with_unbound.coverage, field) == getattr(baseline.coverage, field), field


# The label-binding direction's coverage counters. With len(unbound_labels) they
# partition every task the stamper would admit.
_LABEL_BINDING_COUNTERS = (
    "manifest_bearing_tasks",
    "tasks_bound_to_a_declared_label",
    "tasks_without_a_tracked_sidecar",
    "tasks_on_an_unparseable_sidecar",
)


def _label_binding_counters(coverage) -> dict[str, int]:
    return {name: getattr(coverage, name) for name in _LABEL_BINDING_COUNTERS}


def test_manifest_bearing_tasks_equal_bound_plus_unbound_plus_every_skip_class(
        tmp_path, make_tasks_db):
    """THE ARITHMETIC CLOSES, as it does for the drift direction (see
    scripts/tests/test_audit_manifest_descriptor_drift.py::test_seen_equals_compared_plus_every_skip_class),
    so the unbound list can never be mistaken for the whole population it came from.

    One project exercising every outcome. The absent and the untracked sidecar
    share one term, because neither is part of the tracked corpus. The
    unparseable sidecar gets its own term: it IS tracked, so calling it
    "without a tracked sidecar" would be false.
    """
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[
            labelled_row(1, label="α"),
            labelled_row(2, label="ω"),
            labelled_row(3, prd_path="plans/absent-prd.md"),
            labelled_row(4, prd_path="plans/untracked-prd.md"),
            labelled_row(5, prd_path="plans/broken-prd.md"),
            labelled_row(6, prd_path="plans/broken-prd.md", label="β"),
            # Excluded by the stamper's falsy gate, so in NO term at all.
            {"id": 7, "status": "pending", "metadata": {"prd_path": "plans/x-prd.md"}},
        ],
        manifests=[
            ("plans/x-prd.capability-manifest.yaml", declaring_sidecar("α")),
            ("plans/broken-prd.capability-manifest.yaml", "prd: [unclosed\n  nope: {"),
        ],
    )
    write_manifest(root, "plans/untracked-prd.capability-manifest.yaml",
                    declaring_sidecar("α", prd="plans/untracked-prd.md"))

    audit = audit_project(str(root))
    c = audit.coverage

    assert _label_binding_counters(c) == {
        "manifest_bearing_tasks": 6,
        "tasks_bound_to_a_declared_label": 1,
        "tasks_without_a_tracked_sidecar": 2,
        "tasks_on_an_unparseable_sidecar": 2,
    }
    assert _unbound_pairs(audit) == [(2, "ω")]
    assert c.manifest_bearing_tasks == (
        c.tasks_bound_to_a_declared_label
        + len(audit.unbound_labels)
        + c.tasks_without_a_tracked_sidecar
        + c.tasks_on_an_unparseable_sidecar
    )
    # Two failure modes, never conflated: the existing count stays one per
    # SIDECAR, and still names it, however many tasks point at it.
    assert c.manifest_parse_failures == 1
    assert c.manifest_parse_failure_details[0].startswith(
        "plans/broken-prd.capability-manifest.yaml: ")


def test_label_binding_counters_are_zero_without_manifest_bearing_tasks(
        tmp_path, make_tasks_db):
    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=GREP_CHECK,
                        task_entry=check_entry("gate", GREP_CHECK))

    counters = _label_binding_counters(audit_project(str(root)).coverage)

    assert counters == dict.fromkeys(_LABEL_BINDING_COUNTERS, 0)


def test_label_binding_counters_are_zero_when_git_discovery_failed(
        tmp_path, make_tasks_db):
    """Nothing was enumerated, so nothing was classified. That matches the
    drift counters, which are zero there too; the discovery failure itself is
    what the report's notice and exit 1 carry."""
    root = make_project(tmp_path, make_tasks_db, tasks=[labelled_row(7)])
    not_a_checkout = tmp_path / "bare"
    not_a_checkout.mkdir()

    audit = audit_project(str(root), str(not_a_checkout))

    assert audit.coverage.git_discovery_failed is True
    assert audit.unbound_labels == []
    assert _label_binding_counters(audit.coverage) == dict.fromkeys(
        _LABEL_BINDING_COUNTERS, 0)


def test_coverage_block_counts_the_label_binding_classes_as_aligned_rows(
        tmp_path, make_tasks_db):
    """Four labelled rows, asserted as WHOLE LINES with their alignment (see
    scripts/tests/test_audit_manifest_descriptor_drift_report_cli.py::test_report_coverage_rows_render_with_their_column_alignment).

    The no-tracked-sidecar class is COUNTED and never LISTED. Live, it is 407
    tasks for which the stamper does nothing at all, and listing them would bury
    the few rows the report exists to show. capabilities_without_task_entry is
    treated the same way.
    """
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[labelled_row(1, label="α"), labelled_row(2, label="ω"),
               labelled_row(918273, prd_path="plans/absent-prd.md")],
        manifests=[("plans/x-prd.capability-manifest.yaml", declaring_sidecar("α"))],
    )

    report = format_report([audit_project(str(root))])
    lines = report.splitlines()

    assert "    manifest-bearing tasks:             3" in lines
    assert "    tasks bound to a declared label:    1" in lines
    assert "    tasks with no tracked sidecar:      1" in lines
    assert "    tasks on an unparseable sidecar:    0" in lines
    assert "918273" not in report


# LIVE or HISTORICAL — the mechanical form of task 4907's adjudication. The
# hazard an unbound label carries is that a future commit_planning touching its
# task stamps and copies nothing, and that can only happen to a task that is not
# yet terminal.

def _row_with_status(status: str) -> UnboundLabel:
    return UnboundLabel(task_id=7, label="ω", status=status,
                        manifest="plans/x-prd.capability-manifest.yaml",
                        declared_labels=("α",))


@pytest.mark.parametrize("status", ["done", "cancelled"])
def test_a_done_or_cancelled_row_is_historical(status):
    """No future commit_planning can touch a terminal task, so its label can no
    longer cost anything. It is still REPORTED; it just is not live."""
    assert _row_with_status(status).is_live is False


@pytest.mark.parametrize(
    "status", ["pending", "deferred", "blocked", "in-progress", "merge-deferred"])
def test_a_non_terminal_row_is_live(status):
    """A deferred or blocked task can still be pulled back into a planning
    batch, and while it is live its label is still cheap to fix."""
    assert _row_with_status(status).is_live is True


@pytest.mark.parametrize("status", ["", "archived", "DONE"],
                         ids=["empty", "unknown", "case-variant"])
def test_an_unrecognised_status_is_live(status):
    """Fails toward REPORTING. A status the shared vocabulary does not know is
    never read as a confident 'nothing to fix'."""
    assert _row_with_status(status).is_live is True


def test_liveness_is_the_shared_terminal_set_not_a_local_copy():
    """Across the whole vocabulary, a row is live exactly when its status is
    outside shared.task_statuses.TERMINAL. Statuses are passed as the raw
    strings tasks.db holds, so a local re-spelling of the terminal set that
    drifts from the shared one fails here."""
    for status in TaskStatus:
        assert _row_with_status(status.value).is_live is (status not in TERMINAL), status


def test_live_and_historical_rows_share_one_list(tmp_path, make_tasks_db):
    """ONE list, both kinds, every row named. The live subset is derived from
    it rather than reported in place of it."""
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[labelled_row(7, status="done", label="ω"),
               labelled_row(8, status="pending", label="ψ"),
               labelled_row(9, status="cancelled", label="χ")],
        manifests=[("plans/x-prd.capability-manifest.yaml", declaring_sidecar("α"))],
    )

    rows = audit_project(str(root)).unbound_labels

    assert [(row.task_id, row.label) for row in rows] == [(7, "ω"), (8, "ψ"), (9, "χ")]
    assert [row.task_id for row in rows if row.is_live] == [8]
