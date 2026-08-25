"""Tests for scripts/audit_manifest_descriptor_drift.py — the READ-ONLY sweep
that reports capability-manifest ``delivered_check`` descriptors which have
drifted from the ``metadata.delivered_checks`` entry on their producer task.

Task 4545: ``metadata.delivered_checks`` is copied ONE WAY, sidecar -> task, at
``commit_planning`` (fused-memory/src/fused_memory/server/manifest_stamping.py
step 5), and nothing ever syncs back. A hand-repaired task record therefore sits
beside a stale sidecar, and any re-decompose silently re-stamps the stale
spelling over the repair. This module tests the detector for that regeneration
hazard. Neither the detector nor these tests ever mutate a task record or a
manifest file.

Mirrors test_audit_combine_gate_marker_loss.py: pure functions get direct pytest
coverage; ``main()`` gets subprocess coverage.

NO TEST HERE ASSERTS A COUNT OR TASK ID DERIVED FROM THE LIVE tasks.db.
tasks.db is gitignored, mutated continuously by the running orchestrator, and
absent from a clean clone, so a test pinning "the live DB yields N drift rows"
would be a guessed threshold going red on unrelated branches. Every
tasks.db-dependent assertion below runs against synthetic temp databases built
by the helpers here, whose contents the test controls exactly.

The ONE live-corpus test in this file
(:func:`test_live_sidecars_carry_the_resynced_descriptors`) reads only TRACKED
GIT FILES and opens no database at all — the same legitimacy as
shared/tests/test_capability_manifest.py::TestCheckedInManifestCorpus and
scripts/tests/test_lms_marker_contract.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from audit_manifest_descriptor_drift import (
    MECHANICAL_CHECK_KINDS,
    DescriptorDrift,
    ProjectAudit,
    audit_project,
)

# ---------------------------------------------------------------------------
# Fixtures. The tasks-table schema and the tasks.db builder live in
# scripts/tests/conftest.py behind the `make_tasks_db` fixture (task 3336).
#
# A synthetic project here needs one thing the combine-audit fixtures do not:
# a real `git init` + `git add`, because this sweep discovers manifests via
# `git ls-files` (TRACKED files only) rather than by globbing. No commit is
# needed — ls-files reads the INDEX.
# ---------------------------------------------------------------------------

_GREP_CHECK = {"kind": "grep", "pattern": "def foo", "paths": ["a.py"], "expect": "present"}
_SCRIPT_CHECK = {"kind": "script", "script": "scripts/x.sh", "args": ["--v"], "timeout_secs": 30}
_MANUAL_CHECK = {"kind": "manual", "reason": "needs a human eye"}


def _capability(name: str, check: dict | None) -> dict:
    cap: dict[str, object] = {
        "name": name, "binding": "capability->producer (wired)", "verdict": "PASS"}
    if check is not None:
        cap["delivered_check"] = check
    return cap


def _write_manifest(root: Path, relpath: str, doc) -> Path:
    """Write *doc* to ``<root>/<relpath>``; a str is written VERBATIM.

    Verbatim passthrough is what lets a test seed unparseable YAML.
    """
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc if isinstance(doc, str) else yaml.safe_dump(doc), encoding="utf-8")
    return path


def _manifest_doc(task_id, label="δ", prd="plans/x-prd.md", checks=(("gate", _GREP_CHECK),)):
    return {
        "prd": prd,
        "schema_version": 1,
        "tasks": [{"label": label, "task_id": task_id,
                   "capabilities": [_capability(n, c) for n, c in checks]}],
    }


def _git_init(root: Path) -> None:
    """``git init`` + ``git add -A`` so `git ls-files` finds the manifests.

    Local user config only — never --global — so the suite cannot mutate the
    developer's git configuration. No commit: ls-files reads the index.
    """
    import subprocess

    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    for key, value in (("user.email", "t@example.invalid"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(root), "config", key, value],
                       check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True, capture_output=True)


def _make_project(tmp_path, make_tasks_db, *, name="proj", tasks=(), manifests=()):
    """Build a synthetic project root: tasks.db, manifest sidecars, a git index."""
    root = tmp_path / name
    (root / ".taskmaster" / "tasks").mkdir(parents=True, exist_ok=True)
    make_tasks_db(list(tasks), directory=root / ".taskmaster" / "tasks")
    for relpath, doc in manifests:
        _write_manifest(root, relpath, doc)
    _git_init(root)
    return root


def _entry(name: str, check: dict, **overrides) -> dict:
    """One ``metadata.delivered_checks`` entry: a check dict plus its name."""
    return {"name": name, **check, **overrides}


def _task(task_id: int, entries) -> dict:
    return {"id": task_id, "status": "done", "metadata": {"delivered_checks": list(entries)}}


def _one_project(tmp_path, make_tasks_db, *, sidecar_check, task_entry,
                 cap="gate", task_id=100):
    """The minimal one-manifest/one-task shape most comparison tests need."""
    return _make_project(
        tmp_path, make_tasks_db,
        tasks=[_task(task_id, [task_entry])],
        manifests=[("plans/a-prd.capability-manifest.yaml",
                    _manifest_doc(task_id, checks=((cap, sidecar_check),)))],
    )


def _triples(audit) -> set[tuple[str, int, str]]:
    return {(d.manifest, d.task_id, d.capability) for d in audit.findings}


# ---------------------------------------------------------------------------
# audit_project — the comparison core.
# ---------------------------------------------------------------------------

def test_identical_descriptors_yield_no_finding(tmp_path, make_tasks_db):
    """The steady state: sidecar and task record agree, so nothing is reported."""
    root = _one_project(tmp_path, make_tasks_db,
                        sidecar_check=_GREP_CHECK,
                        task_entry=_entry("gate", _GREP_CHECK))

    audit = audit_project(str(root))

    assert isinstance(audit, ProjectAudit)
    assert audit.findings == []


def test_differing_pattern_is_one_finding_with_the_identity_triple(
        tmp_path, make_tasks_db):
    """The headline row shape: (manifest relpath, task_id, capability name)."""
    root = _one_project(tmp_path, make_tasks_db,
                        sidecar_check=_GREP_CHECK,
                        task_entry=_entry("gate", {**_GREP_CHECK, "pattern": "def bar"}))

    audit = audit_project(str(root))

    assert len(audit.findings) == 1
    drift = audit.findings[0]
    assert isinstance(drift, DescriptorDrift)
    assert (drift.manifest, drift.task_id, drift.capability) == (
        "plans/a-prd.capability-manifest.yaml", 100, "gate")
    assert drift.differing_fields == ("pattern",)
    # The manifest path is REPO-RELATIVE, never absolute: the report must be
    # readable against a checkout, and an absolute tmp_path is meaningless.
    assert not Path(drift.manifest).is_absolute()
    # Both spellings are carried, so a reader never has to open two files to
    # see what drifted.
    assert drift.sidecar_check["pattern"] == "def foo"
    assert drift.task_check["pattern"] == "def bar"
    assert drift.label == "δ"


def test_differing_paths_is_a_finding(tmp_path, make_tasks_db):
    root = _one_project(tmp_path, make_tasks_db,
                        sidecar_check=_GREP_CHECK,
                        task_entry=_entry("gate", {**_GREP_CHECK, "paths": ["b.py"]}))

    audit = audit_project(str(root))

    assert [d.differing_fields for d in audit.findings] == [("paths",)]


def test_differing_expect_present_vs_absent_is_a_finding(tmp_path, make_tasks_db):
    """expect flips the whole MEANING of a grep check, so it must be compared."""
    root = _one_project(tmp_path, make_tasks_db,
                        sidecar_check=_GREP_CHECK,
                        task_entry=_entry("gate", {**_GREP_CHECK, "expect": "absent"}))

    audit = audit_project(str(root))

    assert [d.differing_fields for d in audit.findings] == [("expect",)]


def test_differing_kind_grep_vs_script_is_a_finding(tmp_path, make_tasks_db):
    """kind drift is caught too — a strictly STRONGER rule than the
    (pattern, paths, expect) tuple the task text names."""
    root = _one_project(tmp_path, make_tasks_db,
                        sidecar_check=_GREP_CHECK,
                        task_entry=_entry("gate", _SCRIPT_CHECK))

    audit = audit_project(str(root))

    assert len(audit.findings) == 1
    assert "kind" in audit.findings[0].differing_fields


@pytest.mark.parametrize("field,value", [
    ("script", "scripts/y.sh"),
    ("args", ["--other"]),
    ("timeout_secs", 60),
])
def test_script_kind_descriptor_fields_are_compared(
        tmp_path, make_tasks_db, field, value):
    """kind=script carries three fields of its own; each is compared."""
    root = _one_project(tmp_path, make_tasks_db,
                        sidecar_check=_SCRIPT_CHECK,
                        task_entry=_entry("gate", {**_SCRIPT_CHECK, field: value}))

    audit = audit_project(str(root))

    assert [d.differing_fields for d in audit.findings] == [(field,)]


def test_abbreviated_task_entry_omitting_defaults_is_NOT_a_finding(
        tmp_path, make_tasks_db):
    """THE NORMALIZATION PROPERTY — what keeps the live count at 8, not 22.

    Many real task records carry ABBREVIATED delivered_checks entries that omit
    the defaulted keys entirely (no ``script``, no ``args``, no
    ``timeout_secs``). A raw dict comparison reads those absences as drift
    (``entry.get('args')`` is None against an expected ``[]``) and reports 14
    extra live rows — tasks 4651, 2862, 2863, 2855-2858 and 2860 — that are
    pure absent-vs-default artifacts, not descriptor drift.

    Normalizing BOTH sides through DeliveredCheckMeta fills those defaults. It
    is also the semantically correct rule: the evaluator itself re-validates
    through ``DeliveredCheckMeta(**check)``
    (orchestrator/src/orchestrator/delivered_checks.py::run_delivered_check), so
    two descriptors that normalize identically EVALUATE identically.
    """
    abbreviated = {"name": "gate", "kind": "grep", "pattern": "def foo",
                   "paths": ["a.py"], "expect": "present"}
    assert "args" not in abbreviated and "script" not in abbreviated
    assert "timeout_secs" not in abbreviated

    root = _one_project(tmp_path, make_tasks_db,
                        sidecar_check=_GREP_CHECK, task_entry=abbreviated)

    assert audit_project(str(root)).findings == []


def test_manual_kind_capability_is_skipped_entirely(tmp_path, make_tasks_db):
    """A manual check is never copied to metadata, so it can never drift.

    manifest_stamping.py step 5 filters ``check.kind not in ('grep', 'script')``,
    so comparing a manual capability would report a permanent false positive on
    every manual-checked capability in the corpus.
    """
    assert MECHANICAL_CHECK_KINDS == ("grep", "script")
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_task(100, [])],
        manifests=[("plans/a-prd.capability-manifest.yaml",
                    _manifest_doc(100, checks=(("manual-cap", _MANUAL_CHECK),)))],
    )

    audit = audit_project(str(root))

    assert audit.findings == []
    assert audit.coverage.mechanical_capabilities_compared == 0


def test_capability_without_a_delivered_check_is_skipped(tmp_path, make_tasks_db):
    """delivered_check: None binds no gate, so there is nothing to compare."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_task(100, [])],
        manifests=[("plans/a-prd.capability-manifest.yaml",
                    _manifest_doc(100, checks=(("no-check", None),)))],
    )

    audit = audit_project(str(root))

    assert audit.findings == []
    assert audit.coverage.mechanical_capabilities_compared == 0


def test_unstamped_task_block_binds_nothing_and_is_skipped(tmp_path, make_tasks_db):
    """task_id: None is authoring time, before commit_planning stamps it."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_task(100, [_entry("gate", {**_GREP_CHECK, "pattern": "different"})])],
        manifests=[("plans/a-prd.capability-manifest.yaml", _manifest_doc(None))],
    )

    audit = audit_project(str(root))

    assert audit.findings == []
    assert audit.coverage.mechanical_capabilities_compared == 0


def test_findings_sort_deterministically(tmp_path, make_tasks_db):
    """Sorted by manifest path, then task_id, then capability name.

    A report whose row ORDER depends on filesystem or sqlite iteration order
    cannot be diffed between runs, so the sort is part of the contract.
    """
    drifted = {**_GREP_CHECK, "pattern": "drifted"}
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[
            _task(200, [_entry("z-cap", drifted), _entry("a-cap", drifted)]),
            _task(100, [_entry("gate", drifted)]),
        ],
        manifests=[
            ("plans/z-prd.capability-manifest.yaml",
             _manifest_doc(100, prd="plans/z-prd.md")),
            ("plans/a-prd.capability-manifest.yaml",
             _manifest_doc(200, prd="plans/a-prd.md",
                           checks=(("z-cap", _GREP_CHECK), ("a-cap", _GREP_CHECK)))),
        ],
    )

    audit = audit_project(str(root))

    assert [(d.manifest, d.task_id, d.capability) for d in audit.findings] == [
        ("plans/a-prd.capability-manifest.yaml", 200, "a-cap"),
        ("plans/a-prd.capability-manifest.yaml", 200, "z-cap"),
        ("plans/z-prd.capability-manifest.yaml", 100, "gate"),
    ]


def test_audit_records_both_project_root_and_manifest_root(
        tmp_path, make_tasks_db):
    """A --manifest-root run must be unambiguous in the report: BOTH roots are
    recorded, so a reader never has to guess which tree was swept against which
    task store."""
    root = _one_project(tmp_path, make_tasks_db,
                        sidecar_check=_GREP_CHECK,
                        task_entry=_entry("gate", _GREP_CHECK))

    default = audit_project(str(root))
    assert default.project_root == str(root)
    assert default.manifest_root == str(root)

    other = _make_project(tmp_path, make_tasks_db, name="other")
    decoupled = audit_project(str(root), str(other))
    assert decoupled.project_root == str(root)
    assert decoupled.manifest_root == str(other)
