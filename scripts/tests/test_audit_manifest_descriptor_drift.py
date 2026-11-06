"""Tests for the DRIFT direction of scripts/audit_manifest_descriptor_drift.py,
the READ-ONLY sweep: audit_project's comparison of each sidecar
``delivered_check`` against its producer task's ``metadata.delivered_checks``
entry, and the coverage classes that never reach a comparison.

Task 4545: ``metadata.delivered_checks`` is copied ONE WAY, sidecar -> task, at
``commit_planning`` (fused-memory/src/fused_memory/server/manifest_stamping.py
step 5), and nothing ever syncs back. A hand-repaired task record therefore sits
beside a stale sidecar, and any re-decompose silently re-stamps the stale
spelling over the repair. This module tests the detector for that regeneration
hazard. Neither the detector nor these tests ever mutate a task record or a
manifest file.

The label-binding direction, the signal-binding check, reporting/CLI and the
measured corpus are tested in the sibling
test_audit_manifest_descriptor_drift_{label_binding,signal_binding,report_cli,measured_corpus}.py
modules, over the builders in manifest_drift_fixtures.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from audit_manifest_descriptor_drift import (
    DescriptorDrift,
    ProjectAudit,
    audit_project,
)
from manifest_drift_fixtures import (
    GREP_CHECK,
    MANUAL_CHECK,
    PATH_CHECK,
    SCRIPT_CHECK,
    check_entry,
    finding_triples,
    make_project,
    manifest_doc,
    one_project,
    task_row,
)

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
