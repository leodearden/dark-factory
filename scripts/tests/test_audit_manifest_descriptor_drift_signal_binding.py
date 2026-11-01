"""Tests for scripts/audit_manifest_descriptor_drift.py's signal-binding check
(task 6582), end to end: the scan, the audit rows, both formatters and the exit
code.
"""
from __future__ import annotations

import json

import pytest
from audit_manifest_descriptor_drift import (
    _SIGNAL_BINDING_CAVEAT,
    EXIT_DRIFT,
    StaleSignalBinding,
    _is_dirty,
    audit_project,
    format_json,
    format_report,
    load_task_store_scan,
)
from manifest_drift_fixtures import labelled_row, make_project, run_cli

# ---------------------------------------------------------------------------
# THE SIGNAL-BINDING CHECK (task 6582).
#
# A sidecar's `signal_leaves` names the block(s) whose task status the
# dashboard reads as the PRD's signal (plans/dashboard-prd-lanes-prd.md §4.7,
# §5.2). A signal leaf bound to a task this store lacks, or to a task whose own
# prd_path is another PRD, is reported as a StaleSignalBinding row, never mixed
# into the other lists.
# ---------------------------------------------------------------------------

_SIGNAL_SIDECAR = "plans/x-prd.capability-manifest.yaml"


def _signalling(prd, blocks, signal_leaves, *, external=None):
    """A sidecar binding *blocks* — ``[(label, task_id), ...]`` — with no
    capabilities, naming *signal_leaves* (the key is omitted when None).
    *external* maps a label to the external_task_id its block binds instead."""
    external = external or {}
    tasks = []
    for label, task_id in blocks:
        block: dict[str, object] = {"label": label, "task_id": task_id, "capabilities": []}
        if label in external:
            block["external_task_id"] = external[label]
        tasks.append(block)
    doc: dict[str, object] = {"prd": prd, "schema_version": 1, "tasks": tasks}
    if signal_leaves is not None:
        doc["signal_leaves"] = signal_leaves
    return doc


def _prd_row(task_id, prd_path):
    """A task row carrying only a prd_path, so it binds no label of its own."""
    return {"id": task_id, "status": "pending", "metadata": {"prd_path": prd_path}}


def _row6_project(tmp_path, make_tasks_db, *, name="row6", signal_label="ζ",
                  signal_leaves=True):
    """Boundary row 6: the signal leaf is bound to another PRD's task (8)."""
    return make_project(
        tmp_path, make_tasks_db, name=name,
        tasks=[_prd_row(7, "plans/x-prd.md"), _prd_row(8, "plans/other-prd.md")],
        manifests=[(_SIGNAL_SIDECAR, _signalling(
            "plans/x-prd.md", [("α", 7), (signal_label, 8)],
            [signal_label] if signal_leaves else None))],
    )


_ROW6 = StaleSignalBinding(
    manifest=_SIGNAL_SIDECAR, label="ζ", task_id=8, reason="prd_mismatch",
    sidecar_prd="plans/x-prd.md", task_prd="plans/other-prd.md")


def test_scan_collects_each_rows_prd_path(make_tasks_db):
    """Taken in the SAME one scan, so the signal check reads the snapshot the
    other two directions read. Only a non-empty string prd_path is kept, and
    the tag='master' pin covers it too."""
    db = make_tasks_db([
        labelled_row(7, prd_path="plans/x-prd.md"),
        _prd_row(8, "plans/y-prd.md"),
        {"id": 9, "status": "pending", "metadata": {"prd_path": 5}},
        {"id": 10, "status": "pending", "metadata": {}},
        {"id": 11, "status": "pending", "metadata": {"prd_path": ""}},
        {**_prd_row(12, "plans/z-prd.md"), "tag": "other-tag"},
    ])

    scan = load_task_store_scan(str(db))

    assert scan.prd_paths == {7: "plans/x-prd.md", 8: "plans/y-prd.md"}
    assert scan.row_ids == {7, 8, 9, 10, 11}
    assert scan.delivered_checks == {}
    assert [b.task_id for b in scan.manifest_bindings] == [7]


def test_a_signal_label_bound_to_another_prds_task_is_reported(tmp_path, make_tasks_db):
    """THE ROW-6 SCENARIO, and nothing else moves: the drift findings and every
    coverage counter but the signal check's own two equal those of the same
    project with no signal_leaves."""
    audit = audit_project(str(_row6_project(tmp_path, make_tasks_db)))
    unsignalled = audit_project(str(_row6_project(
        tmp_path, make_tasks_db, name="unsignalled", signal_leaves=False)))

    assert audit.stale_signal_bindings == [_ROW6]
    assert audit.findings == []
    assert (audit.coverage.signal_leaves_named, audit.coverage.signal_leaves_checked) == (1, 1)
    assert audit.coverage == unsignalled.coverage._replace(
        signal_leaves_named=1, signal_leaves_checked=1)


def test_a_signal_label_bound_to_a_task_missing_from_the_store_is_reported(
        tmp_path, make_tasks_db):
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[_prd_row(7, "plans/x-prd.md")],
        manifests=[(_SIGNAL_SIDECAR, _signalling(
            "plans/x-prd.md", [("α", 7), ("ζ", 9)], ["ζ"]))],
    )

    assert audit_project(str(root)).stale_signal_bindings == [StaleSignalBinding(
        manifest=_SIGNAL_SIDECAR, label="ζ", task_id=9, reason="task_missing",
        sidecar_prd="plans/x-prd.md", task_prd=None)]


def test_a_signal_leaf_task_with_no_prd_path_is_a_mismatch(tmp_path, make_tasks_db):
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[_prd_row(7, "plans/x-prd.md"),
               {"id": 8, "status": "pending", "metadata": {}}],
        manifests=[(_SIGNAL_SIDECAR, _signalling(
            "plans/x-prd.md", [("α", 7), ("ζ", 8)], ["ζ"]))],
    )

    assert audit_project(str(root)).stale_signal_bindings == [StaleSignalBinding(
        manifest=_SIGNAL_SIDECAR, label="ζ", task_id=8, reason="prd_mismatch",
        sidecar_prd="plans/x-prd.md", task_prd=None)]


def test_an_agreeing_signal_binding_is_no_row(tmp_path, make_tasks_db):
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[_prd_row(7, "plans/x-prd.md"), _prd_row(8, "plans/x-prd.md")],
        manifests=[(_SIGNAL_SIDECAR, _signalling(
            "plans/x-prd.md", [("α", 7), ("ζ", 8)], ["ζ"]))],
    )

    assert audit_project(str(root)).stale_signal_bindings == []


@pytest.mark.parametrize("external", [None, {"ζ": "reify:5613"}],
                         ids=["unbound", "external"])
def test_unbound_and_external_signal_leaves_are_no_row(tmp_path, make_tasks_db, external):
    """Nothing LOCAL to check: an unbound leaf binds no task yet, and an external
    one names another registry, which plans/dashboard-prd-lanes-prd.md §10
    leaves unresolved. Such a leaf is NAMED but not CHECKED."""
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[_prd_row(7, "plans/x-prd.md")],
        manifests=[(_SIGNAL_SIDECAR, _signalling(
            "plans/x-prd.md", [("α", 7), ("ζ", None)], ["ζ"], external=external))],
    )

    audit = audit_project(str(root))

    assert audit.stale_signal_bindings == []
    assert (audit.coverage.signal_leaves_named, audit.coverage.signal_leaves_checked) == (1, 0)


def test_a_sidecar_without_signal_leaves_yields_no_row(tmp_path, make_tasks_db):
    root = _row6_project(tmp_path, make_tasks_db, signal_leaves=False)

    audit = audit_project(str(root))

    assert audit.stale_signal_bindings == []
    assert (audit.coverage.signal_leaves_named, audit.coverage.signal_leaves_checked) == (0, 0)


def test_signal_leaf_counters_sum_across_sidecars(tmp_path, make_tasks_db):
    """Every named leaf is counted, and every locally bound one is checked,
    whether or not it turns out stale."""
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[_prd_row(7, "plans/x-prd.md"), _prd_row(8, "plans/other-prd.md"),
               _prd_row(9, "plans/y-prd.md")],
        manifests=[
            (_SIGNAL_SIDECAR, _signalling(
                "plans/x-prd.md", [("α", 7), ("ζ", 8), ("η", None)], ["α", "ζ", "η"])),
            ("plans/y-prd.capability-manifest.yaml", _signalling(
                "plans/y-prd.md", [("β", 9)], ["β"])),
        ],
    )

    audit = audit_project(str(root))

    assert [row.label for row in audit.stale_signal_bindings] == ["ζ"]
    assert (audit.coverage.signal_leaves_named, audit.coverage.signal_leaves_checked) == (4, 3)


def _row6_lines(text, label="ζ"):
    """The lines of *text* naming every value boundary row 6 requires the
    report to name (label, task and both PRD values), spelled as literals so
    they are checked independently of the formatter."""
    fields = (f"manifest={_SIGNAL_SIDECAR}", f"label={label}", "task_id=8",
              "reason=prd_mismatch", "sidecar_prd=plans/x-prd.md",
              "task_prd=plans/other-prd.md")
    return [line for line in text.splitlines() if all(field in line for field in fields)]


def test_report_names_the_stale_signal_binding_under_its_own_section(
        tmp_path, make_tasks_db):
    report = format_report([audit_project(str(_row6_project(tmp_path, make_tasks_db)))])

    assert "  -- stale signal bindings (1) --" in report.splitlines()
    assert _SIGNAL_BINDING_CAVEAT in report
    assert len(_row6_lines(report)) == 1


def test_the_signal_binding_section_prints_even_when_empty(tmp_path, make_tasks_db):
    """Looked and found none, as opposed to never looked — and the coverage
    block says how many leaves there were to look at."""
    root = _row6_project(tmp_path, make_tasks_db, signal_leaves=False)

    report = format_report([audit_project(str(root))])
    lines = report.splitlines()

    assert "  -- stale signal bindings (0) --" in lines
    assert _SIGNAL_BINDING_CAVEAT in report
    assert "    signal leaves named:                0" in lines
    assert "    signal leaves checked:              0" in lines


def test_coverage_block_counts_the_signal_leaves_as_aligned_rows(
        tmp_path, make_tasks_db):
    report = format_report([audit_project(str(_row6_project(tmp_path, make_tasks_db)))])
    lines = report.splitlines()

    assert "    signal leaves named:                1" in lines
    assert "    signal leaves checked:              1" in lines


def test_format_json_carries_stale_signal_bindings_and_their_caveat(
        tmp_path, make_tasks_db):
    payload = json.loads(format_json(
        [audit_project(str(_row6_project(tmp_path, make_tasks_db)))]))

    (project,) = payload["projects"]
    assert project["stale_signal_bindings"] == [
        {"manifest": _SIGNAL_SIDECAR, "label": "ζ", "task_id": 8,
         "reason": "prd_mismatch", "sidecar_prd": "plans/x-prd.md",
         "task_prd": "plans/other-prd.md"},
    ]
    assert project["coverage"]["signal_leaves_named"] == 1
    assert project["coverage"]["signal_leaves_checked"] == 1
    assert payload["stale_signal_bindings_caveat"] == _SIGNAL_BINDING_CAVEAT


def test_a_stale_signal_binding_alone_makes_the_run_dirty(tmp_path, make_tasks_db):
    """No historical exemption: the dashboard reads a signal leaf whatever its
    task's status."""
    assert _is_dirty([audit_project(str(_row6_project(tmp_path, make_tasks_db)))])


def test_main_exit_1_when_a_stale_signal_binding_is_the_only_issue(
        tmp_path, make_tasks_db):
    """An ASCII label, so no assertion depends on the child's stdout encoding."""
    root = _row6_project(tmp_path, make_tasks_db, signal_label="zeta")

    result = run_cli("--project-root", str(root))

    assert result.returncode == EXIT_DRIFT, result.stdout
    assert len(_row6_lines(result.stdout, label="zeta")) == 1
