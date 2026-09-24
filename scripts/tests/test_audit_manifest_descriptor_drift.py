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

Task 4907 added the second direction, task -> sidecar. A manifest-bearing task
whose ``prd_task_label`` its tracked sidecar does not declare is an UNBOUND
LABEL: the stamper binds nothing for it and copies it no delivered_checks. Those
rows are reported as their own list, never mixed into the drift findings.

Mirrors test_audit_combine_gate_marker_loss.py: pure functions get direct pytest
coverage; ``main()`` gets subprocess coverage.

NO TEST HERE ASSERTS A COUNT OR TASK ID DERIVED FROM THE LIVE tasks.db.
tasks.db is gitignored, mutated continuously by the running orchestrator, and
absent from a clean clone, so a test pinning "the live DB yields N drift rows"
would be a guessed threshold going red on unrelated branches. Every
tasks.db-dependent assertion below runs against synthetic temp databases built
by the helpers here, whose contents the test controls exactly.

The live-corpus tests in this file
(:func:`test_live_sidecars_carry_the_resynced_descriptors` and
:func:`test_live_sidecars_still_declare_none_of_the_adjudicated_labels`) read
only TRACKED GIT FILES and open no database at all — the same legitimacy as
shared/tests/test_capability_manifest.py::TestCheckedInManifestCorpus and
scripts/tests/test_lms_marker_contract.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from _task_db_scan import (
    AUDIT_EXIT_FINDINGS,
    AUDIT_EXIT_NO_ROOT,
    AUDIT_EXIT_NOTHING_AUDITED,
    AUDIT_EXIT_OK,
    format_kv_line,
)
from audit_manifest_descriptor_drift import (
    _COVERAGE_CAVEAT,
    _DISCOVERY_FAILED_NOTICE,
    _UNBOUND_CAVEAT,
    EXIT_DRIFT,
    EXIT_NO_ROOT,
    EXIT_NOTHING_AUDITED,
    EXIT_OK,
    MECHANICAL_CHECK_KINDS,
    DescriptorDrift,
    ManifestBinding,
    ProjectAudit,
    TaskStoreScan,
    UnboundLabel,
    _is_dirty,
    audit_project,
    format_json,
    format_report,
    load_task_store_scan,
)
from shared.capability_manifest import load_capability_manifest
from shared.task_statuses import TERMINAL, TaskStatus

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


# ---------------------------------------------------------------------------
# COVERAGE — the three classes that never reach a comparison at all.
#
# Each is COUNTED and NOT reported as a finding. The finding list is a
# comparison of MATCHED PAIRS only, so presenting it as the whole corpus would
# be a no-silent-fail-soft violation (docs/legibility/design-invariants.md).
# ---------------------------------------------------------------------------

def test_capability_with_no_same_named_task_entry_is_coverage_not_a_finding(
        tmp_path, make_tasks_db):
    """An ABSENCE is not a DISAGREEMENT.

    Measured live on this corpus: 32 mechanical capabilities whose producer
    task carries no same-named metadata.delivered_checks entry. That class is a
    different defect — its dominant cause is the curator-combine `metadata`
    wipe — and it is already OWNED by scripts/audit_combine_gate_marker_loss.py
    (tasks 3146/3329). Double-filing it here would both duplicate that audit
    and break this sweep's "exactly 8" result.
    """
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_task(100, [_entry("other-cap", _GREP_CHECK)])],
        manifests=[("plans/a-prd.capability-manifest.yaml", _manifest_doc(100))],
    )

    audit = audit_project(str(root))

    assert audit.findings == []
    assert audit.coverage.capabilities_without_task_entry == 1
    # ELIGIBLE, but NOT compared. It was a mechanical capability the sweep
    # reached and tried to pair, and the pairing failed — so it counts toward
    # `seen` and must NOT count toward `compared`, whose whole job is to state
    # how many descriptor comparisons actually happened.
    assert audit.coverage.mechanical_capabilities_seen == 1
    assert audit.coverage.mechanical_capabilities_compared == 0
    # This fixture is ALSO a rename (sidecar 'gate' vs record 'other-cap'), so
    # the reverse row fires on the same task — that pairing is the rename
    # signature. See the dedicated test below.
    assert audit.coverage.task_entries_with_no_sidecar_capability == 1


def test_a_renamed_capability_shows_up_in_BOTH_directions(tmp_path, make_tasks_db):
    """A rename is drift, and the sidecar->task walk alone cannot see it.

    When a hand-repair RENAMES a capability on the task record, the sidecar's
    old name lands in capabilities_without_task_entry — a bucket this report
    attributes to audit_combine_gate_marker_loss.py and says is never
    remediated from here. Without the reverse walk, a genuine drift would be
    silently misfiled as somebody else's problem. Both rows firing on the SAME
    task is the signature, and the detail NAMES the manifest so a reader can
    act on it.
    """
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_task(100, [_entry("gate-renamed", _GREP_CHECK)])],
        manifests=[("plans/a-prd.capability-manifest.yaml", _manifest_doc(100))],
    )

    audit = audit_project(str(root))

    assert audit.coverage.capabilities_without_task_entry == 1
    assert audit.coverage.task_entries_with_no_sidecar_capability == 1
    named = " ".join(audit.coverage.uncomparable_details)
    assert "gate-renamed" in named
    assert "plans/a-prd.capability-manifest.yaml" in named


def test_stale_mechanical_entry_under_a_now_manual_capability_is_seen(
        tmp_path, make_tasks_db):
    """THE SECOND REVERSE SHAPE: grep -> manual on the sidecar.

    A manual capability is skipped by the forward walk (manifest_stamping step
    5 never copies one), and a re-decompose would LEAVE the stale mechanical
    entry in place rather than clearing it, because that step does
    `if not mechanical: continue`. So the stale record entry would otherwise be
    invisible from both ends.
    """
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_task(100, [_entry("gate", _GREP_CHECK)])],
        manifests=[("plans/a-prd.capability-manifest.yaml",
                    _manifest_doc(100, checks=(("gate", _MANUAL_CHECK),)))],
    )

    audit = audit_project(str(root))

    assert audit.findings == []
    # Not seen by the forward walk at all — a manual capability is never
    # mechanical, so it is neither seen nor missing-an-entry.
    assert audit.coverage.mechanical_capabilities_seen == 0
    assert audit.coverage.capabilities_without_task_entry == 0
    # But the stale record entry IS surfaced, by the reverse walk.
    assert audit.coverage.task_entries_with_no_sidecar_capability == 1


def test_seen_equals_compared_plus_every_skip_class(tmp_path, make_tasks_db):
    """THE ARITHMETIC CLOSES, so no reader has to derive a count by subtraction.

    One corpus exercising all four terms at once: a paired-and-compared
    capability, one with no task entry, and one whose task entry will not
    validate. (The fourth term, an unconvertible SIDECAR descriptor, cannot be
    provoked by data — see its own test.)
    """
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_task(100, [
            _entry("paired", {**_GREP_CHECK, "pattern": "drifted"}),
            {"name": "unvalidatable", "kind": "grep", "pattern": "p"},  # no expect
        ])],
        manifests=[("plans/a-prd.capability-manifest.yaml", _manifest_doc(100, checks=(
            ("paired", _GREP_CHECK),
            ("unvalidatable", _GREP_CHECK),
            ("orphan-sidecar", _GREP_CHECK),
        )))],
    )

    c = audit_project(str(root)).coverage

    assert c.mechanical_capabilities_seen == 3
    assert c.mechanical_capabilities_compared == 1
    assert c.capabilities_without_task_entry == 1
    assert c.malformed_task_entries == 1
    assert c.unconvertible_sidecar_descriptors == 0
    assert c.mechanical_capabilities_seen == (
        c.mechanical_capabilities_compared
        + c.capabilities_without_task_entry
        + c.malformed_task_entries
        + c.unconvertible_sidecar_descriptors
    )


def test_an_unconvertible_sidecar_descriptor_degrades_to_coverage(
        tmp_path, make_tasks_db, monkeypatch):
    """One bad sidecar descriptor must NOT abort the sweep.

    FAULT-INJECTED on purpose: today the conversion cannot fail, because a
    validated sidecar grep/script DeliveredCheck shares
    _check_kind_conditional_fields with DeliveredCheckMeta. The guard exists
    because that coupling is IMPLICIT — if the two models ever diverge, an
    unguarded raise escapes audit_project into _task_db_scan.sweep_project_roots,
    which catches only sqlite3.Error, aborting every remaining project root.
    Injecting the failure is the only way to pin the degradation, and pinning it
    is the point: the alternative is discovering the coupling broke by losing a
    whole multi-root sweep to a traceback.
    """
    import audit_manifest_descriptor_drift as mod

    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_task(100, [_entry("boom", _GREP_CHECK),
                           _entry("fine", {**_GREP_CHECK, "pattern": "drifted"})])],
        manifests=[("plans/a-prd.capability-manifest.yaml", _manifest_doc(100, checks=(
            ("boom", _GREP_CHECK), ("fine", _GREP_CHECK),
        )))],
    )

    real = mod._expected_meta

    def flaky(capability_name, check):
        if capability_name == "boom":
            raise ValueError("models diverged")
        return real(capability_name, check)

    monkeypatch.setattr(mod, "_expected_meta", flaky)

    audit = mod.audit_project(str(root))

    # The OTHER capability was still compared and still produced its finding.
    assert _triples(audit) == {("plans/a-prd.capability-manifest.yaml", 100, "fine")}
    assert audit.coverage.unconvertible_sidecar_descriptors == 1
    # NAMED with manifest and capability, never merely counted.
    named = " ".join(audit.coverage.uncomparable_details)
    assert "boom" in named and "plans/a-prd.capability-manifest.yaml" in named
    # Not conflated with the TASK-side channel.
    assert audit.coverage.malformed_task_entries == 0


def test_manifest_task_with_no_db_row_is_coverage_not_a_finding(
        tmp_path, make_tasks_db):
    """A stamped task_id with no tasks.db row binds nothing comparable.

    Measured live on this corpus: 6 manifest task blocks whose stamped task_id
    has no row. Counted separately from the missing-entry class above because
    the two have different causes and different owners; collapsing them would
    misattribute 6 rows into a population of 32.
    """
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_task(999, [_entry("gate", _GREP_CHECK)])],
        manifests=[("plans/a-prd.capability-manifest.yaml", _manifest_doc(100))],
    )

    audit = audit_project(str(root))

    assert audit.findings == []
    assert audit.coverage.manifest_tasks_without_db_row == 1
    # The whole task block is skipped, so none of its capabilities are even
    # SEEN — the skip happens above the capability loop, not inside it.
    assert audit.coverage.mechanical_capabilities_seen == 0
    assert audit.coverage.mechanical_capabilities_compared == 0


@pytest.mark.parametrize("bad_entry,why", [
    ({"name": "gate", "kind": "grep", "pattern": "p", "expect": "present",
      "typo_key": "x"}, "unknown key (extra='forbid')"),
    ({"name": "gate", "kind": "manual", "reason": "r"}, "kind='manual' is not a meta kind"),
    ({"name": "gate", "kind": "grep", "pattern": "p"}, "grep entry missing expect"),
])
def test_unvalidatable_task_entry_is_coverage_and_is_NAMED(
        tmp_path, make_tasks_db, bad_entry, why):
    """A task entry that will not validate cannot be compared — and is NAMED.

    Counted in malformed_task_entries AND named in the coverage details with
    its (task_id, capability). A count alone tells an operator that coverage is
    incomplete but not where to look, which swallows the failure at exactly the
    reporting boundary no-silent-fail-soft is about.
    """
    root = _one_project(tmp_path, make_tasks_db,
                        sidecar_check=_GREP_CHECK, task_entry=bad_entry)

    audit = audit_project(str(root))

    assert audit.findings == [], why
    assert audit.coverage.malformed_task_entries == 1
    named = " ".join(audit.coverage.uncomparable_details)
    assert "100" in named and "gate" in named
    # Not conflated with the sidecar-parse channel.
    assert audit.coverage.manifest_parse_failures == 0
    # Seen but never compared: the pair existed, the normalization failed.
    assert audit.coverage.mechanical_capabilities_seen == 1
    assert audit.coverage.mechanical_capabilities_compared == 0


@pytest.mark.parametrize("bad_doc", [
    "prd: [unclosed\n  nope: {",                       # invalid YAML syntax
    {"prd": "plans/a-prd.md", "schema_version": 1,     # schema-invalid document
     "tasks": [{"label": "δ", "task_id": 100,
                "capabilities": [{"name": "gate", "delivered_check": {"kind": "nonsense"}}]}]},
])
def test_unloadable_sidecar_is_recorded_and_the_sweep_continues(
        tmp_path, make_tasks_db, bad_doc):
    """One bad sidecar never aborts the sweep, and is NAMED with its error."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_task(100, [_entry("gate", {**_GREP_CHECK, "pattern": "drifted"})])],
        manifests=[
            ("plans/bad-prd.capability-manifest.yaml", bad_doc),
            ("plans/good-prd.capability-manifest.yaml", _manifest_doc(100)),
        ],
    )

    audit = audit_project(str(root))

    # The REMAINING manifest was still swept and still produced its finding.
    assert _triples(audit) == {("plans/good-prd.capability-manifest.yaml", 100, "gate")}
    assert audit.coverage.manifests_swept == 1
    assert audit.coverage.manifest_parse_failures == 1
    details = audit.coverage.manifest_parse_failure_details
    assert len(details) == 1
    assert details[0].startswith("plans/bad-prd.capability-manifest.yaml: ")


def test_manifests_swept_and_compared_are_counted(tmp_path, make_tasks_db):
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_task(100, []), _task(200, [])],
        manifests=[
            ("plans/a-prd.capability-manifest.yaml",
             _manifest_doc(100, checks=(("c1", _GREP_CHECK), ("c2", _SCRIPT_CHECK),
                                        ("c3", _MANUAL_CHECK)))),
            ("docs/prds/b-prd.capability-manifest.yaml",
             _manifest_doc(200, prd="docs/prds/b-prd.md")),
        ],
    )

    coverage = audit_project(str(root)).coverage

    assert coverage.manifests_swept == 2
    # c1 + c2 + the b-prd gate; c3 is manual and never mechanical.
    assert coverage.mechanical_capabilities_seen == 3
    # Neither task carries any delivered_checks entry, so nothing PAIRED —
    # every one of the three is eligible-but-unpaired. `compared` must state
    # comparison volume, not eligibility.
    assert coverage.mechanical_capabilities_compared == 0
    assert coverage.capabilities_without_task_entry == 3


# ---------------------------------------------------------------------------
# load_task_store_scan — ONE read of tasks.db feeds BOTH directions.
#
# The drift direction needs every row id and each task's delivered_checks; the
# label-binding direction needs each manifest-bearing task's status, label and
# derived sidecar. All three come back from one scan in one TaskStoreScan
# record, so the two halves of a report can never straddle a write the
# orchestrator makes to the live store in between.
# ---------------------------------------------------------------------------

def _labelled(task_id, *, status="pending", prd_path="plans/x-prd.md", label="α", **extra):
    """A task row whose metadata carries prd_path + prd_task_label (plus *extra*)."""
    return {"id": task_id, "status": status,
            "metadata": {"prd_path": prd_path, "prd_task_label": label, **extra}}


def test_scan_binds_a_manifest_bearing_task_to_its_derived_sidecar(make_tasks_db):
    """The binding carries what the label-binding direction reports: the task,
    its status, its label verbatim, and the sidecar the stamper would open."""
    db = make_tasks_db([_labelled(7, status="pending", label="γ1")])

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
    manifest_stamping.py::_stamp_capability_manifests_impl step 1, exactly, so
    the sweep opens the very sidecar a re-stamp would."""
    db = make_tasks_db([_labelled(7, prd_path=prd_path)])

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
        _labelled(2),
    ])

    scan = load_task_store_scan(str(db))

    assert [b.task_id for b in scan.manifest_bindings] == [2]
    assert scan.row_ids == {1, 2}


def test_bindings_come_back_in_numeric_task_id_order(make_tasks_db):
    """Numeric, not insertion or lexicographic order ("200" < "30" < "4"): the
    report built from these rows must diff cleanly between runs."""
    db = make_tasks_db([_labelled(30), _labelled(200), _labelled(4)])

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
    gate = _entry("gate", _GREP_CHECK)
    cap = _entry("cap", _SCRIPT_CHECK)
    db = make_tasks_db([
        {"id": 1, "status": "pending", "metadata": None},
        {"id": 2, "status": "done", "metadata": {"delivered_checks": [gate]}},
        _labelled(3, status="done", delivered_checks=[cap, "not-a-dict", {"name": 5}]),
        {**_labelled(4, delivered_checks=[gate]), "tag": "other-tag"},
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

def _declaring(*labels, prd="plans/x-prd.md"):
    """A sidecar declaring exactly *labels*, in that order, every entry unstamped.

    No block carries a capability: the label-binding direction reads labels
    only.
    """
    return {"prd": prd, "schema_version": 1,
            "tasks": [{"label": label, "task_id": None, "capabilities": []}
                      for label in labels]}


def _unbound_pairs(audit) -> list[tuple[int, str]]:
    return [(row.task_id, row.label) for row in audit.unbound_labels]


def test_a_label_its_tracked_sidecar_does_not_declare_is_one_unbound_row(
        tmp_path, make_tasks_db):
    """THE ROW SHAPE. It carries the labels the sidecar DOES declare, in the
    sidecar's own order, so a reader can act without opening the file."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_labelled(7, status="pending", label="ω")],
        manifests=[("plans/x-prd.capability-manifest.yaml", _declaring("η0", "α"))],
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
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_labelled(7, label="α"), _labelled(8, label="β")],
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
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_labelled(7, prd_path="plans/absent-prd.md", label="α"),
               _labelled(8, prd_path="plans/untracked-prd.md", label="α")],
    )
    _write_manifest(root, "plans/untracked-prd.capability-manifest.yaml",
                    _declaring("β", prd="plans/untracked-prd.md"))

    assert audit_project(str(root)).unbound_labels == []


def test_label_matching_is_exact_so_a_transliteration_is_a_row(
        tmp_path, make_tasks_db):
    """'gamma-1' is not 'γ1'. This is task 4590's defect shape, and the reason
    this direction exists: the stamper compares the two strings exactly, so a
    transliterated label silently binds nothing."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_labelled(7, label="gamma-1")],
        manifests=[("plans/x-prd.capability-manifest.yaml", _declaring("γ1"))],
    )

    assert _unbound_pairs(audit_project(str(root))) == [(7, "gamma-1")]


def test_two_undeclared_labels_on_one_sidecar_are_two_rows(tmp_path, make_tasks_db):
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_labelled(7, label="ω"), _labelled(8, label="ψ")],
        manifests=[("plans/x-prd.capability-manifest.yaml", _declaring("α"))],
    )

    assert _unbound_pairs(audit_project(str(root))) == [(7, "ω"), (8, "ψ")]


def test_unbound_rows_sort_by_manifest_then_numeric_task_id(tmp_path, make_tasks_db):
    """Manifest relpath first, then NUMERIC task id ("30" < "4" as strings), so
    a report built from these rows diffs cleanly between runs."""
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_labelled(30, prd_path="plans/b-prd.md", label="ω"),
               _labelled(4, prd_path="plans/b-prd.md", label="ψ"),
               _labelled(200, prd_path="plans/a-prd.md", label="ω")],
        manifests=[
            ("plans/b-prd.capability-manifest.yaml", _declaring("α", prd="plans/b-prd.md")),
            ("plans/a-prd.capability-manifest.yaml", _declaring("α", prd="plans/a-prd.md")),
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
    drifted = [_entry("gate", {**_GREP_CHECK, "pattern": "def bar"})]
    manifests = [("plans/a-prd.capability-manifest.yaml",
                  _manifest_doc(100, prd="plans/a-prd.md"))]
    baseline = audit_project(str(_make_project(
        tmp_path, make_tasks_db, name="baseline",
        tasks=[_task(100, drifted)], manifests=manifests)))
    with_unbound = audit_project(str(_make_project(
        tmp_path, make_tasks_db, name="with-unbound",
        tasks=[_task(100, drifted),
               _labelled(200, prd_path="plans/a-prd.md", label="ω"),
               _labelled(201, prd_path="plans/a-prd.md", label="ψ")],
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
    test_seen_equals_compared_plus_every_skip_class), so the unbound list can
    never be mistaken for the whole population it came from.

    One project exercising every outcome. The absent and the untracked sidecar
    share one term, because neither is part of the tracked corpus. The
    unparseable sidecar gets its own term: it IS tracked, so calling it
    "without a tracked sidecar" would be false.
    """
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[
            _labelled(1, label="α"),
            _labelled(2, label="ω"),
            _labelled(3, prd_path="plans/absent-prd.md"),
            _labelled(4, prd_path="plans/untracked-prd.md"),
            _labelled(5, prd_path="plans/broken-prd.md"),
            _labelled(6, prd_path="plans/broken-prd.md", label="β"),
            # Excluded by the stamper's falsy gate, so in NO term at all.
            {"id": 7, "status": "pending", "metadata": {"prd_path": "plans/x-prd.md"}},
        ],
        manifests=[
            ("plans/x-prd.capability-manifest.yaml", _declaring("α")),
            ("plans/broken-prd.capability-manifest.yaml", "prd: [unclosed\n  nope: {"),
        ],
    )
    _write_manifest(root, "plans/untracked-prd.capability-manifest.yaml",
                    _declaring("α", prd="plans/untracked-prd.md"))

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
    root = _one_project(tmp_path, make_tasks_db,
                        sidecar_check=_GREP_CHECK,
                        task_entry=_entry("gate", _GREP_CHECK))

    counters = _label_binding_counters(audit_project(str(root)).coverage)

    assert counters == dict.fromkeys(_LABEL_BINDING_COUNTERS, 0)


def test_label_binding_counters_are_zero_when_git_discovery_failed(
        tmp_path, make_tasks_db):
    """Nothing was enumerated, so nothing was classified. That matches the
    drift counters, which are zero there too; the discovery failure itself is
    what the report's notice and exit 1 carry."""
    root = _make_project(tmp_path, make_tasks_db, tasks=[_labelled(7)])
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
    test_report_coverage_rows_render_with_their_column_alignment).

    The no-tracked-sidecar class is COUNTED and never LISTED. Live, it is 407
    tasks for which the stamper does nothing at all, and listing them would bury
    the few rows the report exists to show. capabilities_without_task_entry is
    treated the same way.
    """
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_labelled(1, label="α"), _labelled(2, label="ω"),
               _labelled(918273, prd_path="plans/absent-prd.md")],
        manifests=[("plans/x-prd.capability-manifest.yaml", _declaring("α"))],
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
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_labelled(7, status="done", label="ω"),
               _labelled(8, status="pending", label="ψ"),
               _labelled(9, status="cancelled", label="χ")],
        manifests=[("plans/x-prd.capability-manifest.yaml", _declaring("α"))],
    )

    rows = audit_project(str(root)).unbound_labels

    assert [(row.task_id, row.label) for row in rows] == [(7, "ω"), (8, "ψ"), (9, "χ")]
    assert [row.task_id for row in rows if row.is_live] == [8]


# ---------------------------------------------------------------------------
# Non-vacuity / loudness. A silently-empty corpus must NEVER render as a clean
# zero: an empty corpus and a clean corpus are indistinguishable in the finding
# count, and only one of them is good news.
# ---------------------------------------------------------------------------

def test_manifest_root_that_is_not_a_git_checkout_is_DIRTY_not_clean(
        tmp_path, make_tasks_db):
    """git discovery failure is recorded, not degraded to an empty sweep."""
    root = _one_project(tmp_path, make_tasks_db,
                        sidecar_check=_GREP_CHECK,
                        task_entry=_entry("gate", _GREP_CHECK))
    not_a_checkout = tmp_path / "bare"
    not_a_checkout.mkdir()

    audit = audit_project(str(root), str(not_a_checkout))

    assert audit.coverage.git_discovery_failed is True
    assert audit.findings == []
    assert audit.coverage.manifests_swept == 0
    # NAMED, not merely flagged.
    assert any("ls-files" in d for d in audit.coverage.uncomparable_details)
    assert _is_dirty([audit]) is True


def test_a_root_with_zero_tracked_manifests_is_clean_not_dirty(
        tmp_path, make_tasks_db):
    """THE OTHER HALF, pinned so the distinction cannot collapse.

    A foreign project may legitimately have no capability manifests at all.
    That is a real, complete, clean sweep of an empty corpus — unlike the
    discovery FAILURE above, where the corpus size is unknown.
    """
    root = _make_project(tmp_path, make_tasks_db, tasks=[_task(100, [])])

    audit = audit_project(str(root))

    assert audit.coverage.manifests_swept == 0
    assert audit.coverage.git_discovery_failed is False
    assert _is_dirty([audit]) is False


# ---------------------------------------------------------------------------
# Formatting. Both formatters are PURE — they return str and print nothing;
# main() does the single print.
# ---------------------------------------------------------------------------

def _drifted_audit(tmp_path, make_tasks_db):
    root = _one_project(tmp_path, make_tasks_db,
                        sidecar_check=_GREP_CHECK,
                        task_entry=_entry("gate", {**_GREP_CHECK, "pattern": "def bar"}))
    return audit_project(str(root))


def _unbound_audit(tmp_path, make_tasks_db):
    """Nothing drifted; one historical and one LIVE unbound label on one sidecar."""
    root = _make_project(
        tmp_path, make_tasks_db, name="unbound",
        tasks=[_labelled(7, status="done", label="ω"),
               _labelled(8, status="pending", label="ψ")],
        manifests=[("plans/x-prd.capability-manifest.yaml", _declaring("η0", "α"))],
    )
    return audit_project(str(root))


def test_formatters_are_pure_and_print_nothing(tmp_path, make_tasks_db, capsys):
    """Both return str and neither prints — main() does the single print."""
    audits = [_drifted_audit(tmp_path, make_tasks_db),
              _unbound_audit(tmp_path, make_tasks_db)]
    capsys.readouterr()

    assert isinstance(format_report(audits), str)
    assert isinstance(format_json(audits), str)
    assert capsys.readouterr().out == ""


def test_coverage_block_is_printed_even_on_a_zero_finding_sweep(
        tmp_path, make_tasks_db):
    """ALWAYS printed. The whole point is that the finding list is a comparison
    of matched pairs, and a reader must be told the size of the remainder."""
    root = _one_project(tmp_path, make_tasks_db,
                        sidecar_check=_GREP_CHECK,
                        task_entry=_entry("gate", _GREP_CHECK))

    report = format_report([audit_project(str(root))])

    assert "COVERAGE" in report
    assert _COVERAGE_CAVEAT in report


def test_report_coverage_rows_render_with_their_column_alignment(
        tmp_path, make_tasks_db):
    """WHOLE LINES with their alignment, never a bare digit substring.

    A bare `assert "32" in report` passes off any number anywhere in the text,
    including a task id, so it cannot tell a correct count from a coincidence.
    """
    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_task(100, [_entry("other", _GREP_CHECK)])],
        manifests=[("plans/a-prd.capability-manifest.yaml", _manifest_doc(100))],
    )

    lines = format_report([audit_project(str(root))]).splitlines()

    assert "    manifests swept:                    1" in lines
    # SEEN and COMPARED render as two ADJACENT rows: the eligible population
    # and the matched-pair count are different numbers, and a report whose
    # thesis is "the findings are not the whole corpus" must not overstate the
    # one figure that says how much was actually compared.
    assert "    mechanical capabilities seen:       1" in lines
    assert "    mechanical capabilities compared:   0" in lines
    assert "    capabilities with no task entry:    1" in lines
    assert "    task entries with no capability:    1" in lines
    assert "    manifest tasks with no db row:      0" in lines
    assert "    unvalidatable task entries:         0" in lines
    assert "    unconvertible sidecar descriptors:  0" in lines
    assert "    manifests that failed to parse:     0" in lines


def test_finding_line_names_manifest_task_capability_and_fields(
        tmp_path, make_tasks_db):
    """Every finding line carries the identity triple AND what differs."""
    report = format_report([_drifted_audit(tmp_path, make_tasks_db)])

    assert format_kv_line([
        ("manifest", "plans/a-prd.capability-manifest.yaml"),
        ("task_id", 100),
        ("capability", "gate"),
        ("fields", "pattern"),
    ]) in report.splitlines()


def test_report_names_both_roots_so_a_decoupled_run_is_unambiguous(
        tmp_path, make_tasks_db):
    root = _make_project(tmp_path, make_tasks_db, tasks=[_task(100, [])])
    other = _make_project(tmp_path, make_tasks_db, name="other")

    report = format_report([audit_project(str(root), str(other))])

    assert str(root) in report
    assert str(other) in report


def test_report_says_so_prominently_when_git_discovery_failed(
        tmp_path, make_tasks_db):
    """A silently-empty corpus must never RENDER as a clean zero either."""
    root = _make_project(tmp_path, make_tasks_db, tasks=[_task(100, [])])
    not_a_checkout = tmp_path / "bare"
    not_a_checkout.mkdir()

    report = format_report([audit_project(str(root), str(not_a_checkout))])

    # On the CONSTANT, matching the _COVERAGE_CAVEAT test's shape. A hardcoded
    # substring pins WORDING rather than behaviour: it fails a reword that is
    # just as loud, and survives a rewrite that guts the warning.
    assert _DISCOVERY_FAILED_NOTICE in report
    # ABOVE the rows, so a reader sees the zero is UNKNOWN before reading it.
    lines = report.splitlines()
    assert lines.index(_DISCOVERY_FAILED_NOTICE) < next(
        i for i, ln in enumerate(lines) if "drifted descriptors" in ln)


def test_format_json_emits_an_object_with_projects_coverage_and_findings(
        tmp_path, make_tasks_db):
    """An OBJECT, not an array: coverage travels WITH the findings, so a
    machine consumer cannot read the finding list without the caveat."""
    payload = json.loads(format_json([_drifted_audit(tmp_path, make_tasks_db)]))

    assert isinstance(payload, dict)
    assert set(payload) >= {"projects"}
    (project,) = payload["projects"]
    assert project["project_root"].endswith("proj")
    assert project["manifest_root"] == project["project_root"]
    assert project["coverage"]["mechanical_capabilities_compared"] == 1
    assert project["coverage"]["git_discovery_failed"] is False
    (finding,) = project["findings"]
    assert finding["manifest"] == "plans/a-prd.capability-manifest.yaml"
    assert finding["task_id"] == 100
    assert finding["capability"] == "gate"
    assert finding["differing_fields"] == ["pattern"]
    # BOTH normalized descriptors travel, so a consumer never has to re-read
    # the two sources to learn what the two spellings actually were.
    assert finding["sidecar_check"]["pattern"] == "def foo"
    assert finding["task_check"]["pattern"] == "def bar"
    # The caveat travels in the payload too, for the same reason.
    assert _COVERAGE_CAVEAT in json.dumps(payload)


# ---------------------------------------------------------------------------
# main() — driven by SUBPROCESS, never in-process.
#
# Shelling out to the script PATH (never `python -m`) is also what PROVES the
# flat-sibling `import _task_db_scan` resolves: a directly-executed script puts
# its own directory at sys.path[0], and scripts/tests/conftest.py's sys.path
# insertion does not reach a child process.
#
# Every test passes an explicit tmp_path --project-root, so the LIVE default
# root (/home/leo/src/dark-factory) is never reached from this suite.
#   exit 0 = no drift; 1 = drift or a failed discovery; 2 = no root resolved;
#   3 = roots resolved but NOTHING was audited.
# ---------------------------------------------------------------------------

_SCRIPT = str(Path(__file__).parent.parent / "audit_manifest_descriptor_drift.py")


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, _SCRIPT, *args], capture_output=True, text=True
    )


def test_main_exit_0_on_a_clean_project_and_still_prints_coverage(
        tmp_path, make_tasks_db):
    """A clean sweep still shows its coverage — see _format_coverage."""
    root = _one_project(tmp_path, make_tasks_db,
                        sidecar_check=_GREP_CHECK,
                        task_entry=_entry("gate", _GREP_CHECK))

    result = _run_cli("--project-root", str(root))

    assert result.returncode == 0, result.stderr
    assert "COVERAGE" in result.stdout


def test_main_exit_1_and_names_the_drifted_row(tmp_path, make_tasks_db):
    root = _one_project(tmp_path, make_tasks_db,
                        sidecar_check=_GREP_CHECK,
                        task_entry=_entry("gate", {**_GREP_CHECK, "pattern": "def bar"}))

    result = _run_cli("--project-root", str(root))

    assert result.returncode == 1
    assert "plans/a-prd.capability-manifest.yaml" in result.stdout
    assert "task_id=100" in result.stdout
    assert "capability=gate" in result.stdout
    assert "fields=pattern" in result.stdout


def test_main_exit_2_when_no_project_root_resolves(tmp_path):
    """The literal 2, NOT the imported constant.

    Deliberate, and the sibling suite records why: importing the constant lets
    a renumber stay green while the epilog keeps promising 2 to operators and
    CI. The number in the contract is what a consumer branches on.
    """
    result = _run_cli("--project-root", str(tmp_path / "no-such-project"))

    assert result.returncode == 2
    assert "no project root" in result.stderr.lower()


def test_main_exit_3_when_the_only_tasks_db_is_unreadable(tmp_path):
    """Roots resolved but every one failed, so NOTHING was audited.

    3, hardcoded, for the same reason as 2 above. Never treat it as clean:
    stdout on this path is a well-formed EMPTY payload.
    """
    root = tmp_path / "corrupt"
    (root / ".taskmaster" / "tasks").mkdir(parents=True)
    (root / ".taskmaster" / "tasks" / "tasks.db").write_text("this is not a database")

    result = _run_cli("--project-root", str(root))

    assert result.returncode == 3
    assert "nothing was" in result.stderr.lower()


def test_main_json_payload_shape(tmp_path, make_tasks_db):
    root = _one_project(tmp_path, make_tasks_db,
                        sidecar_check=_GREP_CHECK,
                        task_entry=_entry("gate", {**_GREP_CHECK, "pattern": "def bar"}))

    result = _run_cli("--project-root", str(root), "--json")

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    (project,) = payload["projects"]
    assert project["coverage"]["mechanical_capabilities_compared"] == 1
    assert [f["capability"] for f in project["findings"]] == ["gate"]


def test_main_project_root_is_repeatable(tmp_path, make_tasks_db):
    """Each root produces its own audit block in ONE render."""
    a = _one_project(tmp_path, make_tasks_db,
                     sidecar_check=_GREP_CHECK,
                     task_entry=_entry("gate", _GREP_CHECK))
    b = _make_project(
        tmp_path, make_tasks_db, name="b",
        tasks=[_task(200, [_entry("gate", {**_GREP_CHECK, "pattern": "drifted"})])],
        manifests=[("plans/b-prd.capability-manifest.yaml",
                    _manifest_doc(200, prd="plans/b-prd.md"))],
    )

    result = _run_cli("--project-root", str(a), "--project-root", str(b), "--json")

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert [p["project_root"] for p in payload["projects"]] == [str(a), str(b)]
    assert [len(p["findings"]) for p in payload["projects"]] == [0, 1]


def test_manifest_root_decouples_the_manifest_tree_from_the_task_store(
        tmp_path, make_tasks_db):
    """THE FLAG'S REASON FOR EXISTING, with both halves asserted.

    `.taskmaster/` is gitignored and exists only in the primary checkout, so a
    task WORKTREE has no tasks.db at all — and discover_project_roots DROPS a
    root without one. Sweeping a worktree's sidecars therefore requires reading
    the manifests from tree B while reading the task store from root A.

    The second half is what makes this non-vacuous: WITHOUT the flag the same
    invocation reports A's own (empty) manifest tree, so the flag is shown to
    genuinely change discovery rather than merely being accepted.
    """
    store = _make_project(tmp_path, make_tasks_db, name="store",
                          tasks=[_task(300, [_entry("gate", {**_GREP_CHECK,
                                                             "pattern": "drifted"})])])
    sidecars = tmp_path / "sidecars"
    _write_manifest(sidecars, "plans/w-prd.capability-manifest.yaml",
                    _manifest_doc(300, prd="plans/w-prd.md"))
    _git_init(sidecars)

    decoupled = _run_cli("--project-root", str(store),
                         "--manifest-root", str(sidecars), "--json")
    assert decoupled.returncode == 1
    (project,) = json.loads(decoupled.stdout)["projects"]
    assert project["project_root"] == str(store)
    assert project["manifest_root"] == str(sidecars)
    assert [f["capability"] for f in project["findings"]] == ["gate"]

    # The report NAMES both roots, so a decoupled run is unambiguous on stdout.
    named = _run_cli("--project-root", str(store), "--manifest-root", str(sidecars))
    assert str(store) in named.stdout and str(sidecars) in named.stdout

    # WITHOUT the flag: A's own manifest tree, which is empty. Same task store,
    # same drifted record, zero findings — so the flag changed discovery.
    default = _run_cli("--project-root", str(store), "--json")
    assert default.returncode == 0
    (only,) = json.loads(default.stdout)["projects"]
    assert only["manifest_root"] == str(store)
    assert only["findings"] == []
    assert only["coverage"]["manifests_swept"] == 0


def test_manifest_root_with_two_project_roots_warns_and_still_runs(
        tmp_path, make_tasks_db):
    """The on_roots warn-not-fail shape: ONE manifest tree over MANY task
    stores is a coherent thing to ask for, so it is warned about, not
    rejected."""
    a = _make_project(tmp_path, make_tasks_db, name="a", tasks=[_task(1, [])])
    b = _make_project(tmp_path, make_tasks_db, name="b", tasks=[_task(2, [])])
    sidecars = _make_project(tmp_path, make_tasks_db, name="side")

    result = _run_cli("--project-root", str(a), "--project-root", str(b),
                      "--manifest-root", str(sidecars), "--json")

    assert result.returncode == 0
    assert "warning" in result.stderr.lower()
    assert len(json.loads(result.stdout)["projects"]) == 2


def test_main_manifest_root_that_is_not_a_checkout_is_loudly_non_zero(
        tmp_path, make_tasks_db):
    """It must NOT report a clean zero — the corpus size is UNKNOWN, not zero."""
    root = _make_project(tmp_path, make_tasks_db, tasks=[_task(100, [])])
    not_a_checkout = tmp_path / "bare"
    not_a_checkout.mkdir()

    result = _run_cli("--project-root", str(root), "--manifest-root", str(not_a_checkout))

    assert result.returncode != 0
    # On the CONSTANT, not a hardcoded substring — see the format_report
    # counterpart above for why.
    assert _DISCOVERY_FAILED_NOTICE in result.stdout


def test_main_run_is_strictly_read_only(tmp_path, make_tasks_db):
    """THE READ-ONLY CLAIM, CHECKED. The tasks.db AND every manifest YAML have
    their (mtime, sha256) captured before and after a full run, and the whole
    mapping must be unchanged."""
    import hashlib

    root = _make_project(
        tmp_path, make_tasks_db,
        tasks=[_task(100, [_entry("gate", {**_GREP_CHECK, "pattern": "drifted"})])],
        manifests=[
            ("plans/a-prd.capability-manifest.yaml", _manifest_doc(100)),
            ("docs/prds/b-prd.capability-manifest.yaml",
             _manifest_doc(100, label="γ", prd="docs/prds/b-prd.md")),
        ],
    )
    inputs = [
        root / ".taskmaster" / "tasks" / "tasks.db",
        root / "plans" / "a-prd.capability-manifest.yaml",
        root / "docs" / "prds" / "b-prd.capability-manifest.yaml",
    ]

    def fingerprint():
        return {
            str(p): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in inputs
        }

    before = fingerprint()
    assert _run_cli("--project-root", str(root)).returncode == 1

    assert fingerprint() == before


def test_exit_constants_alias_the_shared_tier_3_codes():
    """The per-script NAMES survive; the VALUES have ONE home.

    Keeps the aliases honest, exactly as the sibling audit's counterpart does:
    a local re-spelling would drift from what run_audit_cli actually returns.
    """
    assert (EXIT_OK, EXIT_DRIFT, EXIT_NO_ROOT, EXIT_NOTHING_AUDITED) == (
        AUDIT_EXIT_OK, AUDIT_EXIT_FINDINGS, AUDIT_EXIT_NO_ROOT, AUDIT_EXIT_NOTHING_AUDITED)


# ---------------------------------------------------------------------------
# REPORTING THE LABEL-BINDING DIRECTION (task 4907).
#
# Every unbound row is NAMED, historical and live alike, and the section prints
# even when it is empty: a reader must be able to tell "looked and found none"
# from "never looked".
# ---------------------------------------------------------------------------

def _unbound_line(task_id, label, status, hazard):
    return format_kv_line([
        ("manifest", "plans/x-prd.capability-manifest.yaml"),
        ("task_id", task_id),
        ("label", label),
        ("status", status),
        ("hazard", hazard),
    ])


def test_report_names_every_unbound_row_and_marks_the_live_one(
        tmp_path, make_tasks_db):
    """Each row line carries (manifest, task_id, label, status) and a hazard that
    tells LIVE from historical at a glance. The sidecar's declared labels ride
    on the continuation line, in the sidecar's own order."""
    lines = format_report([_unbound_audit(tmp_path, make_tasks_db)]).splitlines()

    for row_line in (_unbound_line(7, "ω", "done", "historical"),
                     _unbound_line(8, "ψ", "pending", "LIVE")):
        assert row_line in lines
        assert lines[lines.index(row_line) + 1] == "      declared labels: η0, α"


def test_the_unbound_section_and_its_caveat_print_even_when_empty(
        tmp_path, make_tasks_db):
    """Never silent, for the reason the coverage block always prints. The caveat
    is asserted by its CONSTANT, as the coverage caveat's test does."""
    root = _one_project(tmp_path, make_tasks_db,
                        sidecar_check=_GREP_CHECK,
                        task_entry=_entry("gate", _GREP_CHECK))

    report = format_report([audit_project(str(root))])

    assert "  -- unbound task labels (0, 0 live) --" in report.splitlines()
    assert _UNBOUND_CAVEAT in report


def test_the_total_line_counts_both_dimensions_and_the_live_rows(
        tmp_path, make_tasks_db):
    """A report carrying two kinds of row must not close on a drift-only total."""
    audits = [_drifted_audit(tmp_path, make_tasks_db),
              _unbound_audit(tmp_path, make_tasks_db)]

    last = format_report(audits).splitlines()[-1]

    assert last == (
        "1 drifted descriptor(s), 2 unbound task label(s) (1 live) across 2 project(s)")


def test_discovery_failed_notice_sits_above_the_unbound_section_too(
        tmp_path, make_tasks_db):
    """An empty unbound list from an unenumerable corpus is as meaningless as an
    empty finding list, so the notice precedes both."""
    root = _make_project(tmp_path, make_tasks_db, tasks=[_labelled(7)])
    not_a_checkout = tmp_path / "bare"
    not_a_checkout.mkdir()

    lines = format_report([audit_project(str(root), str(not_a_checkout))]).splitlines()

    assert lines.index(_DISCOVERY_FAILED_NOTICE) < next(
        i for i, ln in enumerate(lines) if "unbound task labels" in ln)


def test_format_json_carries_unbound_rows_with_an_explicit_liveness_flag(
        tmp_path, make_tasks_db):
    """``is_live`` is a property, so ``_asdict()`` alone would drop it; the JSON
    writer carries it explicitly. The new caveat travels under its own key,
    leaving the drift caveat as it was."""
    payload = json.loads(format_json([_unbound_audit(tmp_path, make_tasks_db)]))

    (project,) = payload["projects"]
    assert project["unbound_labels"] == [
        {"task_id": 7, "label": "ω", "status": "done",
         "manifest": "plans/x-prd.capability-manifest.yaml",
         "declared_labels": ["η0", "α"], "is_live": False},
        {"task_id": 8, "label": "ψ", "status": "pending",
         "manifest": "plans/x-prd.capability-manifest.yaml",
         "declared_labels": ["η0", "α"], "is_live": True},
    ]
    assert payload["unbound_labels_caveat"] == _UNBOUND_CAVEAT
    assert payload["caveat"] == _COVERAGE_CAVEAT


# ---------------------------------------------------------------------------
# THE 21 MEASURED UNBOUND LABELS (task 4907).
#
# Measured 2026-09-23 at main fbf0b5c683, read-only against the live tasks.db:
# every manifest-bearing task whose tracked sidecar does not declare its
# prd_task_label. A DATED RECORD, never re-measured. Each row is
# (sidecar relpath, task_id, label, status, class). Each class is adjudicated in
# its sidecar's .md twin, under "Unbound task labels (task 4907 adjudication)".
# ---------------------------------------------------------------------------

_EVAL_REVIVAL = "plans/eval-framework-revival-prd.capability-manifest.yaml"
_FOUND_ON_MAIN = "plans/found-on-main-provenance-integrity-prd.capability-manifest.yaml"
_DASHBOARD = "plans/dashboard-availability-prd.capability-manifest.yaml"
_STATE_GRAPH = "plans/task-escalation-state-graph-prd.capability-manifest.yaml"

_MEASURED_UNBOUND_ROWS = (
    # A1 — the wave before the sidecar. Twin section in
    # plans/eval-framework-revival-prd.capability-manifest.md.
    (_EVAL_REVIVAL, 2464, "α", "done", "A1"),
    (_EVAL_REVIVAL, 2466, "β", "done", "A1"),
    (_EVAL_REVIVAL, 2469, "γ", "done", "A1"),
    (_EVAL_REVIVAL, 2470, "δ", "done", "A1"),
    (_EVAL_REVIVAL, 2471, "ε", "done", "A1"),
    (_EVAL_REVIVAL, 2472, "ι", "done", "A1"),
    (_EVAL_REVIVAL, 2473, "ζ", "done", "A1"),
    (_EVAL_REVIVAL, 2474, "η", "done", "A1"),
    (_EVAL_REVIVAL, 2475, "θ", "done", "A1"),
    (_EVAL_REVIVAL, 2476, "κ", "done", "A1"),
    (_EVAL_REVIVAL, 2477, "λ", "done", "A1"),
    (_EVAL_REVIVAL, 2478, "μ", "done", "A1"),
    (_EVAL_REVIVAL, 2479, "ν", "done", "A1"),
    (_EVAL_REVIVAL, 2480, "ξ", "done", "A1"),
    (_EVAL_REVIVAL, 2825, "ο", "done", "A1"),
    # A2 — the PRD's intermediates, left out of a leaf-only sidecar. Twin
    # section in plans/found-on-main-provenance-integrity-prd.capability-manifest.md.
    (_FOUND_ON_MAIN, 2674, "α", "done", "A2"),
    (_FOUND_ON_MAIN, 2675, "δ", "done", "A2"),
    (_FOUND_ON_MAIN, 2676, "ι", "done", "A2"),
    (_FOUND_ON_MAIN, 2677, "β", "done", "A2"),
    # B — an out-of-plan task with a hand-written label. Twin section in
    # plans/dashboard-availability-prd.capability-manifest.md.
    (_DASHBOARD, 3289, "restore-gate", "done", "B"),
    # C — an agent follow-up with a synthesized label; the only live row. Twin
    # section in plans/task-escalation-state-graph-prd.capability-manifest.md.
    (_STATE_GRAPH, 4172, "kappa-followup", "pending", "C"),
)

# What each of those sidecars declared at the same measurement.
_MEASURED_DECLARED_LABELS = {
    _EVAL_REVIVAL: ("π", "ρ", "σ", "υ", "φ"),
    _FOUND_ON_MAIN: ("γ", "ε", "ζ", "η", "κ", "θ"),
    _DASHBOARD: ("α", "β", "γ", "δ", "ε", "ζ", "η", "θ"),
    _STATE_GRAPH: ("α", "η0", "β", "γ1", "γ2", "γ3", "δ", "ζ", "η", "θ", "ι", "κ", "λ", "μ"),
}


def _prd_of(relpath: str) -> str:
    return relpath.replace(".capability-manifest.yaml", ".md")


def _unbound_rows_as_project(tmp_path, make_tasks_db):
    """Rebuild the 21 measured rows as ONE synthetic project, with decoys.

    Each sidecar declares its measured label set, and each task carries its
    measured label and status. The decoys make the assertion a genuine SET
    equality over a mixed corpus rather than "every task present is unbound":
    a task whose label its sidecar declares, a manifest-bearing task on an
    untracked sidecar, a task the stamper's falsy gate excludes, and one
    descriptor-drift row.
    """
    tasks = [
        _labelled(task_id, status=status, prd_path=_prd_of(relpath), label=label)
        for relpath, task_id, label, status, _class in _MEASURED_UNBOUND_ROWS
    ]
    manifests = [
        (relpath, _declaring(*labels, prd=_prd_of(relpath)))
        for relpath, labels in _MEASURED_DECLARED_LABELS.items()
    ]

    tasks += [
        _labelled(9001, status="pending", prd_path=_prd_of(_STATE_GRAPH), label="κ"),
        _labelled(9002, status="pending", prd_path="plans/untracked-prd.md", label="α"),
        {"id": 9003, "status": "pending",
         "metadata": {"prd_path": _prd_of(_DASHBOARD), "prd_task_label": ""}},
        _task(9004, [_entry("gate", {**_GREP_CHECK, "pattern": "drifted"})]),
    ]
    manifests.append(("plans/decoy-prd.capability-manifest.yaml",
                      _manifest_doc(9004, prd="plans/decoy-prd.md")))

    root = _make_project(tmp_path, make_tasks_db, name="unbound-corpus",
                         tasks=tasks, manifests=manifests)
    _write_manifest(root, "plans/untracked-prd.capability-manifest.yaml",
                    _declaring("β", prd="plans/untracked-prd.md"))
    return root


def test_the_21_measured_unbound_rows_are_reported_exactly(tmp_path, make_tasks_db):
    """SET EQUALITY on the 21 NAMED rows, exactly one of them live.

    The durable stand-in for the live measurement, and synthetic for the reason
    this module's docstring gives: tasks.db is gitignored, so an assertion on it
    would go red on unrelated branches. A bare count could stay green while the
    sweep reported the WRONG rows; set equality cannot.
    """
    assert [row[4] for row in _MEASURED_UNBOUND_ROWS].count("A1") == 15
    assert [row[4] for row in _MEASURED_UNBOUND_ROWS].count("A2") == 4

    audit = audit_project(str(_unbound_rows_as_project(tmp_path, make_tasks_db)))

    assert len(audit.unbound_labels) == 21
    assert {(row.manifest, row.task_id, row.label) for row in audit.unbound_labels} == {
        (relpath, task_id, label)
        for relpath, task_id, label, _status, _class in _MEASURED_UNBOUND_ROWS
    }
    assert [(row.task_id, row.label) for row in audit.unbound_labels if row.is_live] == [
        (4172, "kappa-followup")]
    # The decoys land where they belong, and the drift direction still reports
    # exactly its own row.
    assert audit.coverage.tasks_bound_to_a_declared_label == 1
    assert audit.coverage.tasks_without_a_tracked_sidecar == 1
    assert audit.coverage.manifest_bearing_tasks == 23
    assert _triples(audit) == {("plans/decoy-prd.capability-manifest.yaml", 9004, "gate")}


_ADJUDICATED_LABELS_BY_SIDECAR = {
    sidecar: {label for relpath, _task_id, label, _status, _class in _MEASURED_UNBOUND_ROWS
              if relpath == sidecar}
    for sidecar in _MEASURED_DECLARED_LABELS
}


@pytest.mark.parametrize("relpath", sorted(_ADJUDICATED_LABELS_BY_SIDECAR))
def test_live_sidecars_still_declare_none_of_the_adjudicated_labels(relpath):
    """The adjudication's premise, checked against the TRACKED sidecars.

    Parametrized so a failure names its own sidecar. This is the only part of
    the 21-row record CI can run: tasks.db is gitignored, so the sweep itself
    never runs there. It pins only the premise, that none of the adjudicated
    labels is declared, and not each sidecar's full label set, which a
    legitimate PRD amendment may grow.

    MAINTENANCE CONTRACT — READ THIS BEFORE "FIXING" A FAILURE HERE. If a
    sidecar legitimately gains one of these labels (the fix for a class-A row
    would be to complete the sidecar), that row is no longer unbound. Update
    _MEASURED_UNBOUND_ROWS and the sidecar's "Unbound task labels (task 4907
    adjudication)" twin section in the SAME commit. Do not relax this pin.
    """
    root = _repo_root()
    if root is None:
        pytest.skip("not a git checkout")

    # NON-VACUITY FLOOR: a renamed or deleted sidecar must not pass by finding
    # nothing to check, and neither must one emptied of every label.
    tracked = subprocess.run(
        ["git", "-C", root, "ls-files", "--", relpath],
        capture_output=True, text=True, timeout=30,
    )
    assert tracked.stdout.strip(), f"{relpath} is not tracked in {root}"
    declared = {task.label for task in load_capability_manifest(Path(root) / relpath).tasks}
    assert declared, f"{relpath} declares no labels at all"

    adjudicated = _ADJUDICATED_LABELS_BY_SIDECAR[relpath]
    assert declared.isdisjoint(adjudicated), (
        f"{relpath} now declares {sorted(declared & adjudicated)}, which task 4907 "
        f"adjudicated as unbound. See this test's maintenance contract."
    )


# ---------------------------------------------------------------------------
# THE EIGHT MEASURED DRIFT ROWS (task 4545).
#
# Each row is (manifest relpath, task_id, label, capability, stale_pattern_
# fields, resynced_pattern_fields) — the two spellings as MEASURED on base
# 4ab3b731f3, transcribed verbatim. `stale` is what the sidecar carried before
# the resync; `resynced` is what the (already-repaired) task record carries and
# what the sidecar must carry after step 8.
#
# AUTHORING HAZARD, row toolcall-markup: its stale pattern IS the raw envelope
# closer sentinel. A literal here would force this file's own author to emit
# that sentinel inside a tool call, reproducing exactly the leak the
# toolcall-markup PRD exists to CONTAIN. It is therefore built from chr(60) —
# never typed. The resynced value 'parameter name=' is CANONICAL_OPENER_PREFIX
# written without its leading angle bracket: deliberate, and verified
# equivalent (the bracket-free and full-sentinel anchors return an identical
# 3-line match set on the pinned worktree-inventory.json, with a negative
# control against README.md returning rc=1).
# ---------------------------------------------------------------------------

_INVOKE_CLOSER = chr(60) + "/invoke" + chr(62)


def _grep(pattern, paths, expect="present"):
    return {"kind": "grep", "pattern": pattern, "paths": list(paths), "expect": expect}


_MEASURED_DRIFT_ROWS = (
    (
        "plans/agent-transcript-archival-prd.capability-manifest.yaml", 2730, "γ",
        "archive-root-shipped-on-in-legibility-yaml",
        # The single-line regex cannot match block-style YAML; the capability
        # IS delivered, at docs/legibility/legibility.yaml:20-21.
        _grep(r"agent_transcript_roots:\s*\[?\s*data/orchestrator/agent-transcripts",
              ["docs/legibility/legibility.yaml"]),
        _grep("data/orchestrator/agent-transcripts",
              ["docs/legibility/legibility.yaml"]),
    ),
    (
        "plans/flake-ledger-prd.capability-manifest.yaml", 3793, "ι",
        "report-surfaces-the-hold-with-its-owner-and-age",
        # cli.py's flake_ledger_cmd is a 25-line click wrapper; the owner+age
        # render lives in flake_report.py, where the BARE token matches 3 lines
        # and the braced form matches exactly 1.
        _grep("owner_task_id", ["orchestrator/src/orchestrator/cli.py"]),
        _grep(r"owner=\{d\.owner_task_id\}",
              ["orchestrator/src/orchestrator/flake_report.py"]),
    ),
    (
        "plans/os-sandbox-worktree-containment-prd.capability-manifest.yaml", 2906, "α4",
        "enforcement-matrix-suite-exists",
        # BENIGN — both spellings deliver. Resynced anyway so the sweep can
        # assert zero drift rather than carrying an allowlist.
        _grep("test_sandbox_enforcement_matrix", ["orchestrator/tests/"]),
        _grep("Landlock enforcement-matrix suite", ["orchestrator/tests/"]),
    ),
    (
        "plans/task-escalation-state-graph-prd.capability-manifest.yaml", 3534, "η0",
        "dispatch-gate-any-level-veto",
        # The ONE row whose `expect` also flips. Five has_open_l1 sites
        # legitimately remain — three unrelated functions keep L1-only dedup
        # and were never in 3534's scope — so the absent-grep could not hold.
        _grep(r"has_open_l1\(task_id\)", ["orchestrator/src/orchestrator/harness.py"],
              expect="absent"),
        _grep("vetoes_done_flip", ["orchestrator/src/orchestrator/harness.py"]),
    ),
    (
        "plans/task-escalation-state-graph-prd.capability-manifest.yaml", 3536, "γ1",
        "no-steward-less-escalated-exit-at-merge-entry",
        # THE FILENAME-VS-CONTENT MODE. orchestrator/tests/
        # test_workflow_merge_gating_strand.py DOES exist, but kind=grep
        # searches CONTENT and a test module does not mention its own name.
        _grep("test_workflow_merge_gating_strand", ["orchestrator/tests/"]),
        _grep("TestNoStrandExitProperty", ["orchestrator/tests/"]),
    ),
    (
        "plans/toolcall-markup-containment-prd.capability-manifest.yaml", 3691, "δ",
        "committed-evidence-file-survives-the-sweep",
        _grep(_INVOKE_CLOSER,
              ["docs/task-recovery-2026-05-13/worktree-inventory.json"]),
        _grep("parameter name=",
              ["docs/task-recovery-2026-05-13/worktree-inventory.json"]),
    ),
    (
        "plans/transcript-preservation-seam-prd.capability-manifest.yaml", 3618, "α",
        "complete-gz-consumer-set-including-memory-eval-corpus",
        # DELIBERATELY SUPERSEDED, and it stays FAILING by design: task 3578
        # RESTORED gzip reading after 3618 removed it, so BOTH spellings fail
        # on main. Only the SPELLING is resynced here — this row is not to be
        # "repaired" to pass, and its sibling capability writer-emits-plain-
        # jsonl (pattern ^import gzip) is not drifted and is not touched.
        _grep(r"jsonl\.gz", [
            "fused-memory/scripts/memory_eval_transcript_corpus.py",
            "scripts/gc_agent_transcripts.py",
            "scripts/legibility/",
            "shared/src/shared/transcript_archive.py",
        ], expect="absent"),
        _grep(r"gzip\.open", [
            "fused-memory/scripts/memory_eval_transcript_corpus.py",
            "scripts/gc_agent_transcripts.py",
            "scripts/legibility/",
            "shared/src/shared/transcript_archive.py",
        ], expect="absent"),
    ),
    (
        "plans/warm-lane-infra-repatriation-prd.capability-manifest.yaml", 3075, "γ",
        "per-lane-recheck-not-snapshot",
        # A comment naming the REJECTED --assigned-lanes alternative trips the
        # whole-file absent-grep; the anchored form excludes comment lines and
        # returns 0 hits.
        _grep("--assigned-lanes", ["orchestrator/scripts/warm-lane/",
                                   "orchestrator/src/orchestrator/"], expect="absent"),
        _grep(r"^[^#]*--assigned-lanes", ["orchestrator/scripts/warm-lane/",
                                          "orchestrator/src/orchestrator/"],
              expect="absent"),
    ),
)


def _rows_as_project(tmp_path, make_tasks_db, *, sidecar_index):
    """Rebuild all 8 measured rows as ONE synthetic project.

    *sidecar_index* picks which spelling the SIDECAR side carries: 4 for the
    stale one (reproducing the pre-resync corpus) or 5 for the resynced one
    (the post-resync corpus). The tasks.db side always carries the resynced
    spelling, because the task records were already repaired by hand — that
    asymmetry IS the defect being detected.

    Decoys are included so the assertion is a genuine SET equality over a
    mixed corpus rather than "everything present drifted": a non-drifted grep
    row, an ABBREVIATED task entry omitting script/args/timeout_secs, and a
    manual-kind capability.
    """
    by_manifest: dict[str, list] = {}
    tasks: dict[int, list] = {}
    for relpath, task_id, label, cap, stale, resynced in _MEASURED_DRIFT_ROWS:
        by_manifest.setdefault(relpath, []).append(
            (label, task_id, cap, [stale, resynced][sidecar_index - 4]))
        tasks.setdefault(task_id, []).append(_entry(cap, resynced))

    # Decoys, all on one extra task/manifest.
    tasks[9001] = [
        _entry("decoy-agrees", _GREP_CHECK),
        # Abbreviated: no script, no args, no timeout_secs. Must NOT drift.
        {"name": "decoy-abbreviated", "kind": "grep", "pattern": "def foo",
         "paths": ["a.py"], "expect": "present"},
    ]

    manifests = []
    for relpath, rows in by_manifest.items():
        blocks: dict[tuple[str, int], list] = {}
        for label, task_id, cap, check in rows:
            blocks.setdefault((label, task_id), []).append(_capability(cap, check))
        manifests.append((relpath, {
            "prd": relpath.replace(".capability-manifest.yaml", ".md"),
            "schema_version": 1,
            "tasks": [{"label": label, "task_id": task_id, "capabilities": caps}
                      for (label, task_id), caps in blocks.items()],
        }))
    manifests.append(("plans/decoy-prd.capability-manifest.yaml", {
        "prd": "plans/decoy-prd.md", "schema_version": 1,
        "tasks": [{"label": "ζ", "task_id": 9001, "capabilities": [
            _capability("decoy-agrees", _GREP_CHECK),
            _capability("decoy-abbreviated", _GREP_CHECK),
            _capability("decoy-manual", _MANUAL_CHECK),
        ]}],
    }))

    return _make_project(
        tmp_path, make_tasks_db, name=f"corpus{sidecar_index}",
        tasks=[_task(tid, entries) for tid, entries in sorted(tasks.items())],
        manifests=manifests,
    )


def test_the_eight_measured_drift_rows_are_reported_exactly(tmp_path, make_tasks_db):
    """SET EQUALITY on the eight NAMED triples, then the paired flip to zero.

    This is the DURABLE stand-in for the live before/after measurement, and it
    is synthetic for the reason this module's docstring gives: tasks.db is
    gitignored, orchestrator-mutated and absent from a clean clone, so a live
    assertion on it would go red on unrelated branches. The synthetic corpus
    reproduces all 8 real rows verbatim — stale sidecar spelling against
    resynced task-record spelling — so the test cannot stay green while the
    sweep reports the WRONG rows, which a bare count assertion would allow.
    """
    stale_root = _rows_as_project(tmp_path, make_tasks_db, sidecar_index=4)

    audit = audit_project(str(stale_root))

    assert _triples(audit) == {
        (relpath, task_id, cap)
        for relpath, task_id, _label, cap, _stale, _resynced in _MEASURED_DRIFT_ROWS
    }

    # THE PAIRED FLIP: the same corpus with the RESYNCED spellings on the
    # sidecar side reports nothing. Without this half, a sweep that reported
    # every capability as drifted would still satisfy the set equality above
    # for these eight.
    resynced_root = _rows_as_project(tmp_path, make_tasks_db, sidecar_index=5)

    assert audit_project(str(resynced_root)).findings == []


def test_the_measured_rows_carry_the_two_differing_field_sets(
        tmp_path, make_tasks_db):
    """Two of the eight differ in MORE than `pattern`, and that is pinned.

    3793 also moved `paths` (cli.py -> flake_report.py) and 3534 also flips
    `expect` (absent -> present). A sweep that compared only `pattern` would
    report all 8 triples and still be wrong about these two.
    """
    audit = audit_project(str(_rows_as_project(tmp_path, make_tasks_db, sidecar_index=4)))
    fields = {(d.task_id, d.capability): d.differing_fields for d in audit.findings}

    assert fields[(3793, "report-surfaces-the-hold-with-its-owner-and-age")] == (
        "paths", "pattern")
    assert fields[(3534, "dispatch-gate-any-level-veto")] == ("expect", "pattern")
    assert all(
        f == ("pattern",) for (tid, _cap), f in fields.items() if tid not in (3793, 3534)
    )


# ---------------------------------------------------------------------------
# The LIVE corpus pin — PURE GIT, no tasks.db, no databases at all.
#
# Same legitimacy as shared/tests/test_capability_manifest.py::
# TestCheckedInManifestCorpus and scripts/tests/test_lms_marker_contract.py:
# it reads only TRACKED files in this checkout. That is what makes it a
# legitimate live assertion where a tasks.db one would not be.
# ---------------------------------------------------------------------------

def _repo_root():
    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


@pytest.mark.parametrize(
    "relpath,task_id,label,capability,resynced",
    [(r[0], r[1], r[2], r[3], r[5]) for r in _MEASURED_DRIFT_ROWS],
    ids=[f"{r[1]}-{r[3]}" for r in _MEASURED_DRIFT_ROWS],
)
def test_live_sidecars_carry_the_resynced_descriptors(
        relpath, task_id, label, capability, resynced):
    """Each of the 8 tracked sidecars carries its task record's spelling.

    Parametrized so a failure NAMES its own manifest/label/capability rather
    than reporting "one of eight".

    MAINTENANCE CONTRACT — READ THIS BEFORE "FIXING" A FAILURE HERE. This is an
    EXACT pin, not a not-the-stale-spelling pin, and it is deliberately the
    stricter of the two: because tasks.db is gitignored the sweep itself can
    never run in CI, so this is the ONLY automated guard that the resync
    survives. The cost of that strictness is that it also fires on a LEGITIMATE
    change — a later task that re-repairs one of these eight checks on BOTH
    sides has zero drift and is entirely correct, and will still turn this red.
    THAT IS NOT A STALENESS BUG. If you changed one of these checks on purpose,
    update its row in _MEASURED_DRIFT_ROWS in the SAME commit (the `resynced`
    element, index 5) and re-run
    `scripts/audit_manifest_descriptor_drift.py --project-root <primary>
    --manifest-root <this checkout>` to confirm it still reports zero — which
    is the assertion this pin is standing in for. Only a sidecar that disagrees
    with its task record is the defect this test was written to catch.
    """
    root = _repo_root()
    if root is None:
        pytest.skip("not a git checkout")

    # NON-VACUITY FLOOR: assert the sidecar is TRACKED before reading it, so a
    # renamed or deleted manifest cannot make this test pass by finding
    # nothing to check.
    tracked = subprocess.run(
        ["git", "-C", root, "ls-files", "--", relpath],
        capture_output=True, text=True, timeout=30,
    )
    assert tracked.stdout.strip(), f"{relpath} is not tracked in {root}"

    doc = load_capability_manifest(Path(root) / relpath)
    matches = [
        cap for task in doc.tasks if task.label == label
        for cap in task.capabilities if cap.name == capability
    ]
    assert len(matches) == 1, (
        f"expected exactly one {capability!r} under label {label!r} "
        f"in {relpath}, found {len(matches)}"
    )

    check = matches[0].delivered_check
    assert check is not None
    assert check.model_dump() == {
        "kind": resynced["kind"],
        "pattern": resynced["pattern"],
        "expect": resynced["expect"],
        "paths": resynced["paths"],
        "script": None,
        "args": [],
        "timeout_secs": None,
        "reason": None,
    }, (
        f"{relpath} label {label} capability {capability} (task {task_id}) does "
        f"not carry the descriptor recorded in _MEASURED_DRIFT_ROWS. Either the "
        f"task-4545 resync was reverted (the sidecar is STALE and a re-decompose "
        f"would re-stamp the stale spelling over the repair), OR this check was "
        f"legitimately re-repaired on BOTH sides since — in which case update "
        f"this row's `resynced` element and see this test's maintenance contract."
    )
