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
"""
from __future__ import annotations

from pathlib import Path

import pytest
from audit_manifest_descriptor_drift import (
    DescriptorDrift,
    ManifestBinding,
    ProjectAudit,
    TaskStoreScan,
    UnboundLabel,
    audit_project,
    format_report,
    load_task_store_scan,
)
from manifest_drift_fixtures import (
    GREP_CHECK,
    MANUAL_CHECK,
    PATH_CHECK,
    SCRIPT_CHECK,
    check_entry,
    declaring_sidecar,
    finding_triples,
    labelled_row,
    make_project,
    manifest_doc,
    one_project,
    task_row,
    write_manifest,
)
from shared.task_statuses import TERMINAL, TaskStatus

# ---------------------------------------------------------------------------
# audit_project — the comparison core.
# ---------------------------------------------------------------------------

def test_identical_descriptors_yield_no_finding(tmp_path, make_tasks_db):
    """The steady state: sidecar and task record agree, so nothing is reported."""
    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=GREP_CHECK,
                        task_entry=check_entry("gate", GREP_CHECK))

    audit = audit_project(str(root))

    assert isinstance(audit, ProjectAudit)
    assert audit.findings == []


def test_differing_pattern_is_one_finding_with_the_identity_triple(
        tmp_path, make_tasks_db):
    """The headline row shape: (manifest relpath, task_id, capability name)."""
    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=GREP_CHECK,
                        task_entry=check_entry("gate", {**GREP_CHECK, "pattern": "def bar"}))

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
    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=GREP_CHECK,
                        task_entry=check_entry("gate", {**GREP_CHECK, "paths": ["b.py"]}))

    audit = audit_project(str(root))

    assert [d.differing_fields for d in audit.findings] == [("paths",)]


def test_differing_expect_present_vs_absent_is_a_finding(tmp_path, make_tasks_db):
    """expect flips the whole MEANING of a grep check, so it must be compared."""
    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=GREP_CHECK,
                        task_entry=check_entry("gate", {**GREP_CHECK, "expect": "absent"}))

    audit = audit_project(str(root))

    assert [d.differing_fields for d in audit.findings] == [("expect",)]


def test_differing_kind_grep_vs_script_is_a_finding(tmp_path, make_tasks_db):
    """kind drift is caught too — a strictly STRONGER rule than the
    (pattern, paths, expect) tuple the task text names."""
    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=GREP_CHECK,
                        task_entry=check_entry("gate", SCRIPT_CHECK))

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
    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=SCRIPT_CHECK,
                        task_entry=check_entry("gate", {**SCRIPT_CHECK, field: value}))

    audit = audit_project(str(root))

    assert [d.differing_fields for d in audit.findings] == [(field,)]


def test_identical_path_descriptors_are_compared_and_agree(tmp_path, make_tasks_db):
    """kind=path is mechanical: the stamper copies it, so the sweep compares it."""
    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=PATH_CHECK,
                        task_entry=check_entry("gate", PATH_CHECK))

    audit = audit_project(str(root))

    assert audit.findings == []
    assert audit.coverage.mechanical_capabilities_compared == 1
    assert audit.coverage.task_entries_with_no_sidecar_capability == 0


def test_a_path_sidecar_against_a_grep_task_entry_is_kind_drift(
        tmp_path, make_tasks_db):
    """The half-landed shape of a MODE-3 repair: sidecar rewritten, record not."""
    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=PATH_CHECK,
                        task_entry=check_entry("gate", GREP_CHECK))

    audit = audit_project(str(root))

    assert len(audit.findings) == 1
    assert "kind" in audit.findings[0].differing_fields


def test_differing_paths_on_a_path_check_is_a_finding(tmp_path, make_tasks_db):
    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=PATH_CHECK,
                        task_entry=check_entry("gate", {**PATH_CHECK, "paths": ["scripts/y.sh"]}))

    audit = audit_project(str(root))

    assert [d.differing_fields for d in audit.findings] == [("paths",)]


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

    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=GREP_CHECK, task_entry=abbreviated)

    assert audit_project(str(root)).findings == []


def test_manual_kind_capability_is_skipped_entirely(tmp_path, make_tasks_db):
    """A manual check is never copied to metadata, so it can never drift.

    manifest_stamping.py step 5 copies only
    ``shared.capability_manifest.MECHANICAL_CHECK_KINDS``, of which manual is
    never one, so comparing a manual capability would report a permanent false
    positive on every manual-checked capability in the corpus.
    """
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(100, [])],
        manifests=[("plans/a-prd.capability-manifest.yaml",
                    manifest_doc(100, checks=(("manual-cap", MANUAL_CHECK),)))],
    )

    audit = audit_project(str(root))

    assert audit.findings == []
    assert audit.coverage.mechanical_capabilities_compared == 0


def test_capability_without_a_delivered_check_is_skipped(tmp_path, make_tasks_db):
    """delivered_check: None binds no gate, so there is nothing to compare."""
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(100, [])],
        manifests=[("plans/a-prd.capability-manifest.yaml",
                    manifest_doc(100, checks=(("no-check", None),)))],
    )

    audit = audit_project(str(root))

    assert audit.findings == []
    assert audit.coverage.mechanical_capabilities_compared == 0


def test_unstamped_task_block_binds_nothing_and_is_skipped(tmp_path, make_tasks_db):
    """task_id: None is authoring time, before commit_planning stamps it."""
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(100, [check_entry("gate", {**GREP_CHECK, "pattern": "different"})])],
        manifests=[("plans/a-prd.capability-manifest.yaml", manifest_doc(None))],
    )

    audit = audit_project(str(root))

    assert audit.findings == []
    assert audit.coverage.mechanical_capabilities_compared == 0


def test_findings_sort_deterministically(tmp_path, make_tasks_db):
    """Sorted by manifest path, then task_id, then capability name.

    A report whose row ORDER depends on filesystem or sqlite iteration order
    cannot be diffed between runs, so the sort is part of the contract.
    """
    drifted = {**GREP_CHECK, "pattern": "drifted"}
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[
            task_row(200, [check_entry("z-cap", drifted), check_entry("a-cap", drifted)]),
            task_row(100, [check_entry("gate", drifted)]),
        ],
        manifests=[
            ("plans/z-prd.capability-manifest.yaml",
             manifest_doc(100, prd="plans/z-prd.md")),
            ("plans/a-prd.capability-manifest.yaml",
             manifest_doc(200, prd="plans/a-prd.md",
                           checks=(("z-cap", GREP_CHECK), ("a-cap", GREP_CHECK)))),
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
    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=GREP_CHECK,
                        task_entry=check_entry("gate", GREP_CHECK))

    default = audit_project(str(root))
    assert default.project_root == str(root)
    assert default.manifest_root == str(root)

    other = make_project(tmp_path, make_tasks_db, name="other")
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
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(100, [check_entry("other-cap", GREP_CHECK)])],
        manifests=[("plans/a-prd.capability-manifest.yaml", manifest_doc(100))],
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
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(100, [check_entry("gate-renamed", GREP_CHECK)])],
        manifests=[("plans/a-prd.capability-manifest.yaml", manifest_doc(100))],
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
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(100, [check_entry("gate", GREP_CHECK)])],
        manifests=[("plans/a-prd.capability-manifest.yaml",
                    manifest_doc(100, checks=(("gate", MANUAL_CHECK),)))],
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
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(100, [
            check_entry("paired", {**GREP_CHECK, "pattern": "drifted"}),
            {"name": "unvalidatable", "kind": "grep", "pattern": "p"},  # no expect
        ])],
        manifests=[("plans/a-prd.capability-manifest.yaml", manifest_doc(100, checks=(
            ("paired", GREP_CHECK),
            ("unvalidatable", GREP_CHECK),
            ("orphan-sidecar", GREP_CHECK),
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


def test_seen_arithmetic_is_untouched_by_an_external_registry_block(
        tmp_path, make_tasks_db):
    """THE SAME ARITHMETIC, with a foreign-registry block in the corpus.

    The new class is skipped ABOVE the capability loop, exactly where the
    task_id-is-None and no-db-row skips already happen, so its capabilities
    never enter the eligible population. The identity therefore closes over the
    same four terms and does NOT grow a fifth — a reader who learned the
    arithmetic from the test above does not have to relearn it.
    """
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(100, [
            check_entry("paired", {**GREP_CHECK, "pattern": "drifted"}),
            {"name": "unvalidatable", "kind": "grep", "pattern": "p"},  # no expect
        ])],
        manifests=[
            ("plans/a-prd.capability-manifest.yaml", manifest_doc(100, checks=(
                ("paired", GREP_CHECK),
                ("unvalidatable", GREP_CHECK),
                ("orphan-sidecar", GREP_CHECK),
            ))),
            ("plans/b-prd.capability-manifest.yaml", manifest_doc(
                None, label="η", external_task_id="reify:5613",
                checks=(("foreign-gate", GREP_CHECK),))),
        ],
    )

    c = audit_project(str(root)).coverage

    assert c.external_registry_task_blocks == 1
    # 3, not 4: the external block's mechanical capability is never seen.
    assert c.mechanical_capabilities_seen == 3
    assert c.mechanical_capabilities_compared == 1
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
    which catches only UNREADABLE_STORE_ERRORS, aborting every remaining
    project root.
    Injecting the failure is the only way to pin the degradation, and pinning it
    is the point: the alternative is discovering the coupling broke by losing a
    whole multi-root sweep to a traceback.
    """
    import audit_manifest_descriptor_drift as mod

    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(100, [check_entry("boom", GREP_CHECK),
                           check_entry("fine", {**GREP_CHECK, "pattern": "drifted"})])],
        manifests=[("plans/a-prd.capability-manifest.yaml", manifest_doc(100, checks=(
            ("boom", GREP_CHECK), ("fine", GREP_CHECK),
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
    assert finding_triples(audit) == {("plans/a-prd.capability-manifest.yaml", 100, "fine")}
    assert audit.coverage.unconvertible_sidecar_descriptors == 1
    # NAMED with manifest and capability, never merely counted.
    named = " ".join(audit.coverage.uncomparable_details)
    assert "boom" in named and "plans/a-prd.capability-manifest.yaml" in named
    # Not conflated with the TASK-side channel.
    assert audit.coverage.malformed_task_entries == 0


def test_manifest_task_with_no_db_row_is_coverage_not_a_finding(
        tmp_path, make_tasks_db):
    """A stamped task_id with no tasks.db row binds nothing comparable.

    Counted separately from the missing-entry class above because the two have
    different causes and different owners; collapsing them would misattribute
    the rows into a population a different audit owns.

    This bucket means a STALE OR UNSTAMPED binding — an integer that was
    supposed to name a row in THIS project's store and does not. It is
    remediable by re-stamping or retiring the block, and the assertion below
    that such a block does NOT increment external_registry_task_blocks is the
    other half of that distinction: a foreign-registry binding is an EXPLAINED
    absence, not a stale one, and is remediable by neither action.
    """
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(999, [check_entry("gate", GREP_CHECK)])],
        manifests=[("plans/a-prd.capability-manifest.yaml", manifest_doc(100))],
    )

    audit = audit_project(str(root))

    assert audit.findings == []
    assert audit.coverage.manifest_tasks_without_db_row == 1
    # THE TWO CLASSES MUST NOT COLLAPSE. A stale integer binding is not a
    # foreign-registry binding, and telling an operator otherwise is how this
    # bucket came to be read as "six blocks to re-stamp or retire".
    assert audit.coverage.external_registry_task_blocks == 0
    # The whole task block is skipped, so none of its capabilities are even
    # SEEN — the skip happens above the capability loop, not inside it.
    assert audit.coverage.mechanical_capabilities_seen == 0
    assert audit.coverage.mechanical_capabilities_compared == 0


def test_external_registry_task_block_is_its_own_coverage_class(
        tmp_path, make_tasks_db):
    """A block bound by external_task_id is an EXPLAINED absence, not a stale one.

    Its producer is live — in ANOTHER project's registry — so it can never
    appear in this project's tasks.db, and neither remediation that
    manifest_tasks_without_db_row implies (re-stamp, or retire the block)
    applies to it. It gets its own counter so the report says which of the two
    an operator is looking at.
    """
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(999, [check_entry("gate", GREP_CHECK)])],
        manifests=[("plans/a-prd.capability-manifest.yaml",
                    manifest_doc(None, label="η", external_task_id="reify:5613"))],
    )

    audit = audit_project(str(root))

    assert audit.findings == []
    assert audit.coverage.external_registry_task_blocks == 1
    assert audit.coverage.manifest_tasks_without_db_row == 0
    # Same skip position as the no-db-row class: above the capability loop, so
    # the block's capabilities are not even SEEN.
    assert audit.coverage.mechanical_capabilities_seen == 0
    assert audit.coverage.mechanical_capabilities_compared == 0


def _declare_project_id(root: Path, project_id: str) -> None:
    """Give a synthetic root the manifest that names its project_id.

    ``resolve_project_id_for_root`` reads the declared id from
    ``dark-factory-orchestrator.yaml`` first and falls back to the
    basename derivation. Declaring it makes the tests below say WHICH id
    they mean instead of depending on tmp_path's directory naming.
    """
    (root / "dark-factory-orchestrator.yaml").write_text(
        f"project_id: {project_id}\n", encoding="utf-8")


def test_external_task_id_naming_the_audited_project_is_not_excused(
        tmp_path, make_tasks_db):
    """A block cannot excuse ITSELF from comparison by claiming to be foreign.

    ``external_task_id`` is only an explanation while the registry it names
    belongs to somebody else. The model can validate the SHAPE of the value
    (``ExternalDep.parse`` wants two non-empty parts) but has no idea which
    project a sidecar is being swept for — so ``home_proj:5613`` in
    home_proj's OWN corpus validates fine, and folding it into
    external_registry_task_blocks would permanently and silently drop a
    block's capabilities out of the eligible population under a row that
    reads as an explained absence. That is the same misattribution the field
    exists to prevent, pointed at the side this audit is responsible for.
    """
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(5613, [check_entry("gate", GREP_CHECK)])],
        manifests=[("plans/a-prd.capability-manifest.yaml",
                    manifest_doc(None, label="η",
                                  external_task_id="home_proj:5613"))],
    )
    _declare_project_id(root, "home_proj")

    coverage = audit_project(str(root)).coverage

    assert coverage.self_bound_external_task_blocks == 1
    # THE TWO MUST NOT COLLAPSE, in this direction either: a mis-authored
    # binding is not an explained absence, and its remediation is neither
    # of the other two rows' (fix the binding — a local producer is a task_id).
    assert coverage.external_registry_task_blocks == 0
    assert coverage.manifest_tasks_without_db_row == 0


def test_a_self_bound_external_block_is_NAMED_not_merely_counted(
        tmp_path, make_tasks_db):
    """A count alone cannot be acted on — the details locate the block.

    Same rule as every other uncomparable class here: an operator told only
    that "1 block is self-bound" cannot find out WHICH, which swallows the
    failure at exactly the reporting boundary no-silent-fail-soft is about.
    """
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(5613, [check_entry("gate", GREP_CHECK)])],
        manifests=[("plans/a-prd.capability-manifest.yaml",
                    manifest_doc(None, label="η",
                                  external_task_id="home_proj:5613"))],
    )
    _declare_project_id(root, "home_proj")

    details = audit_project(str(root)).coverage.uncomparable_details

    (detail,) = [d for d in details if "external_task_id" in d]
    assert "plans/a-prd.capability-manifest.yaml" in detail
    assert "η" in detail
    assert "home_proj:5613" in detail


def test_a_foreign_block_whose_number_collides_with_a_local_row_stays_foreign(
        tmp_path, make_tasks_db):
    """The check keys on the PROJECT half, never on the number.

    Task ids are per-registry, so a collision between reify's 5613 and this
    project's 5613 is ordinary and says nothing: dark-factory's own ids were
    at 5173 when this field was introduced against reify's 5332-5616, and
    they will pass them. A guard that fired on "the numeric part resolves to
    a local row" would therefore start misreporting the legitimately
    foreign blocks in this repo's corpus as mis-authored, on a schedule set
    by task numbering. This test pins that it does not.
    """
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(5613, [check_entry("gate", GREP_CHECK)])],
        manifests=[("plans/a-prd.capability-manifest.yaml",
                    manifest_doc(None, label="η",
                                  external_task_id="reify:5613"))],
    )
    _declare_project_id(root, "home_proj")

    coverage = audit_project(str(root)).coverage

    assert coverage.external_registry_task_blocks == 1
    assert coverage.self_bound_external_task_blocks == 0
    assert coverage.uncomparable_details == ()


def test_self_binding_check_uses_the_declared_id_not_the_directory_name(
        tmp_path, make_tasks_db):
    """Resolution goes through the repo's own rename-stable resolver.

    ``resolve_project_id_for_root`` prefers the id declared in
    ``dark-factory-orchestrator.yaml`` over the directory basename, so a
    checkout whose directory was renamed still self-identifies. Re-deriving
    that rule locally would give this audit a second, drifting answer to
    "which project is this" — the same single-authority argument that keeps
    the wire-form check in ``ExternalDep.parse``.
    """
    root = make_project(
        tmp_path, make_tasks_db, name="renamed-checkout",
        tasks=[task_row(5613, [check_entry("gate", GREP_CHECK)])],
        manifests=[("plans/a-prd.capability-manifest.yaml",
                    manifest_doc(None, label="η",
                                  external_task_id="home_proj:5613"))],
    )
    _declare_project_id(root, "home_proj")

    coverage = audit_project(str(root)).coverage

    # Caught on the DECLARED id, though the directory is named otherwise.
    assert coverage.self_bound_external_task_blocks == 1
    assert coverage.external_registry_task_blocks == 0


def test_seen_arithmetic_is_untouched_by_a_self_bound_block(
        tmp_path, make_tasks_db):
    """The mis-authored class skips in the SAME place as the explained one.

    Both branches classify above the capability loop, so neither grows a
    fifth term in the seen == compared + skips identity. A reader who learned
    the arithmetic from the two tests above does not have to relearn it.
    """
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(100, [check_entry("paired", GREP_CHECK)])],
        manifests=[
            ("plans/a-prd.capability-manifest.yaml",
             manifest_doc(100, checks=(("paired", GREP_CHECK),))),
            ("plans/b-prd.capability-manifest.yaml",
             manifest_doc(None, label="η", external_task_id="home_proj:5613",
                           checks=(("self-bound-gate", GREP_CHECK),))),
        ],
    )
    _declare_project_id(root, "home_proj")

    c = audit_project(str(root)).coverage

    assert c.self_bound_external_task_blocks == 1
    # 1, not 2: the self-bound block's capability is never seen either.
    assert c.mechanical_capabilities_seen == 1
    assert c.mechanical_capabilities_compared == (
        c.mechanical_capabilities_seen
        - c.capabilities_without_task_entry
        - c.malformed_task_entries
        - c.unconvertible_sidecar_descriptors
    )


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
    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=GREP_CHECK, task_entry=bad_entry)

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
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(100, [check_entry("gate", {**GREP_CHECK, "pattern": "drifted"})])],
        manifests=[
            ("plans/bad-prd.capability-manifest.yaml", bad_doc),
            ("plans/good-prd.capability-manifest.yaml", manifest_doc(100)),
        ],
    )

    audit = audit_project(str(root))

    # The REMAINING manifest was still swept and still produced its finding.
    assert finding_triples(audit) == {("plans/good-prd.capability-manifest.yaml", 100, "gate")}
    assert audit.coverage.manifests_swept == 1
    assert audit.coverage.manifest_parse_failures == 1
    details = audit.coverage.manifest_parse_failure_details
    assert len(details) == 1
    assert details[0].startswith("plans/bad-prd.capability-manifest.yaml: ")


def test_manifests_swept_and_compared_are_counted(tmp_path, make_tasks_db):
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(100, []), task_row(200, [])],
        manifests=[
            ("plans/a-prd.capability-manifest.yaml",
             manifest_doc(100, checks=(("c1", GREP_CHECK), ("c2", SCRIPT_CHECK),
                                        ("c3", MANUAL_CHECK)))),
            ("docs/prds/b-prd.capability-manifest.yaml",
             manifest_doc(200, prd="docs/prds/b-prd.md")),
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
    test_seen_equals_compared_plus_every_skip_class), so the unbound list can
    never be mistaken for the whole population it came from.

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
    test_report_coverage_rows_render_with_their_column_alignment).

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
