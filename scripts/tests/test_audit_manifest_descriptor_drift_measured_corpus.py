"""Tests that the dated measured-corpus records (task 4545's eight drift rows,
task 4907's 21 unbound labels) are reported exactly by
scripts/audit_manifest_descriptor_drift.py::audit_project, plus the live
tracked-file pins that stand in for the sweep in CI.

NO TEST HERE ASSERTS A COUNT OR TASK ID DERIVED FROM THE LIVE tasks.db.
tasks.db is gitignored, mutated continuously by the running orchestrator, and
absent from a clean clone, so a test pinning "the live DB yields N drift rows"
would be a guessed threshold going red on unrelated branches. Every
tasks.db-dependent assertion below runs against synthetic temp databases built
by manifest_drift_fixtures, whose contents the test controls exactly.

The live-corpus tests in this file
(:func:`test_live_sidecars_carry_the_resynced_descriptors`,
:func:`test_live_sidecars_still_declare_none_of_the_adjudicated_labels` and
:func:`test_identifier_anchor_names_a_definition_inside_its_certified_suite`)
read only TRACKED GIT FILES and open no database at all — the same legitimacy
as shared/tests/test_capability_manifest.py::TestCheckedInManifestCorpus and
scripts/tests/test_lms_marker_contract.py.
"""
from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from audit_manifest_descriptor_drift import audit_project
from git_checkout_root import checkout_root_or_skip
from manifest_drift_fixtures import (
    GREP_CHECK,
    MANUAL_CHECK,
    capability_doc,
    check_entry,
    declaring_sidecar,
    finding_triples,
    git_init,
    labelled_row,
    make_project,
    manifest_doc,
    task_row,
    write_manifest,
)
from shared.capability_manifest import DeliveredCheck, load_capability_manifest
from shared.delivered_check_polarity import (
    CheckOutcome,
    build_grep_argv,
    interpret_grep_rc,
)

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
        labelled_row(task_id, status=status, prd_path=_prd_of(relpath), label=label)
        for relpath, task_id, label, status, _class in _MEASURED_UNBOUND_ROWS
    ]
    manifests = [
        (relpath, declaring_sidecar(*labels, prd=_prd_of(relpath)))
        for relpath, labels in _MEASURED_DECLARED_LABELS.items()
    ]

    tasks += [
        labelled_row(9001, status="pending", prd_path=_prd_of(_STATE_GRAPH), label="κ"),
        labelled_row(9002, status="pending", prd_path="plans/untracked-prd.md", label="α"),
        {"id": 9003, "status": "pending",
         "metadata": {"prd_path": _prd_of(_DASHBOARD), "prd_task_label": ""}},
        task_row(9004, [check_entry("gate", {**GREP_CHECK, "pattern": "drifted"})]),
    ]
    manifests.append(("plans/decoy-prd.capability-manifest.yaml",
                      manifest_doc(9004, prd="plans/decoy-prd.md")))

    root = make_project(tmp_path, make_tasks_db, name="unbound-corpus",
                         tasks=tasks, manifests=manifests)
    write_manifest(root, "plans/untracked-prd.capability-manifest.yaml",
                    declaring_sidecar("β", prd="plans/untracked-prd.md"))
    return root


def test_the_21_measured_unbound_rows_are_reported_exactly(tmp_path, make_tasks_db):
    """SET EQUALITY on the 21 NAMED rows, exactly one of them live.

    The durable stand-in for the live measurement, and synthetic for the reason
    this module's docstring gives: tasks.db is gitignored, so an assertion on it
    would go red on unrelated branches. A bare count could stay green while the
    sweep reported the WRONG rows; set equality cannot.
    """
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
    assert finding_triples(audit) == {("plans/decoy-prd.capability-manifest.yaml", 9004, "gate")}


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
    root = checkout_root_or_skip()

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
        # RE-REPAIRED ON BOTH SIDES (task 4783): the docstring-prose anchor
        # broke silently if anyone reworded the docstring, so this row now
        # anchors on the suite's own class identifier instead — same rule as
        # the γ1/3536 row below. See
        # test_identifier_anchor_names_a_definition_inside_its_certified_suite.
        _grep("test_sandbox_enforcement_matrix", ["orchestrator/tests/"]),
        _grep("TestSandboxEnforcementMatrix", ["orchestrator/tests/"]),
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
        # The resynced anchor is identifier-anchored, checked by
        # test_identifier_anchor_names_a_definition_inside_its_certified_suite.
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

#: The rows whose resynced pattern is an IDENTIFIER ANCHOR: a class/def name
#: certifying that a test suite exists. (task_id, capability) -> the suite file
#: every hit must stay inside.
_CERTIFIED_SUITE_BY_ANCHORED_ROW = {
    (2906, "enforcement-matrix-suite-exists"):
        "orchestrator/tests/test_sandbox_enforcement_matrix.py",
    (3536, "no-steward-less-escalated-exit-at-merge-entry"):
        "orchestrator/tests/test_workflow_merge_gating_strand.py",
}


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
        tasks.setdefault(task_id, []).append(check_entry(cap, resynced))

    # Decoys, all on one extra task/manifest.
    tasks[9001] = [
        check_entry("decoy-agrees", GREP_CHECK),
        # Abbreviated: no script, no args, no timeout_secs. Must NOT drift.
        {"name": "decoy-abbreviated", "kind": "grep", "pattern": "def foo",
         "paths": ["a.py"], "expect": "present"},
    ]

    manifests = []
    for relpath, rows in by_manifest.items():
        blocks: dict[tuple[str, int], list] = {}
        for label, task_id, cap, check in rows:
            blocks.setdefault((label, task_id), []).append(capability_doc(cap, check))
        manifests.append((relpath, {
            "prd": relpath.replace(".capability-manifest.yaml", ".md"),
            "schema_version": 1,
            "tasks": [{"label": label, "task_id": task_id, "capabilities": caps}
                      for (label, task_id), caps in blocks.items()],
        }))
    manifests.append(("plans/decoy-prd.capability-manifest.yaml", {
        "prd": "plans/decoy-prd.md", "schema_version": 1,
        "tasks": [{"label": "ζ", "task_id": 9001, "capabilities": [
            capability_doc("decoy-agrees", GREP_CHECK),
            capability_doc("decoy-abbreviated", GREP_CHECK),
            capability_doc("decoy-manual", MANUAL_CHECK),
        ]}],
    }))

    return make_project(
        tmp_path, make_tasks_db, name=f"corpus{sidecar_index}",
        tasks=[task_row(tid, entries) for tid, entries in sorted(tasks.items())],
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

    assert finding_triples(audit) == {
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

def _live_delivered_check(root: str, relpath: str, label: str,
                          capability: str) -> DeliveredCheck:
    """The delivered_check of the one *capability* under *label* in a TRACKED sidecar."""
    # NON-VACUITY FLOOR: assert the sidecar is TRACKED before reading it, so a
    # renamed or deleted manifest cannot make a caller pass by finding nothing
    # to check.
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
    assert check is not None, (
        f"{relpath} label {label} capability {capability} carries no delivered_check"
    )
    return check


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
    root = checkout_root_or_skip()

    check = _live_delivered_check(root, relpath, label, capability)
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


@dataclass(frozen=True)
class _AnchorReading:
    """One grep check's working-tree hits, read against the suite it certifies."""

    outcome: CheckOutcome
    paths_hit: frozenset[str]
    defines_in_suite: bool
    stderr: str


#: Fills build_grep_argv's mandatory ref slot only to be removed again: with no
#: tree argument git grep reads the working tree, which is the tree
#: _live_delivered_check reads the sidecar from.
_REF_SLOT_PLACEHOLDER = "<working-tree>"

_DEFINED_NAME = re.compile(r"\s*(?:async\s+)?(?:class|def)\s+(\w+)")


def _working_tree_grep_argv(root: str, pattern: str,
                            paths: Sequence[str]) -> list[str]:
    """build_grep_argv's argv minus its ref, plus the output-only --null and
    --column, which change neither the match set nor the exit code."""
    argv = build_grep_argv(pattern, paths, project_root=root,
                           ref=_REF_SLOT_PLACEHOLDER)
    removed = argv.pop(argv.index("-e") + 2)
    assert removed == _REF_SLOT_PLACEHOLDER, (
        f"build_grep_argv no longer puts its ref right after the pattern: {argv}"
    )
    flags_at = argv.index("-E")
    argv[flags_at:flags_at] = ["--null", "--column"]
    return argv


def _match_starts_in_defined_name(text: str, column: int) -> bool:
    """Whether git's 1-based byte *column* of a line's first match falls in the
    name that line's class/def defines, not in a comment or body after it."""
    defined = _DEFINED_NAME.match(text)
    if defined is None:
        return False
    start, end = (len(text[:offset].encode()) for offset in defined.span(1))
    return start < column <= end


def _read_identifier_anchor(root: str, *, pattern: str, paths: Sequence[str],
                            expect: str | None, suite: str) -> _AnchorReading:
    """Run the runtime's grep over the working tree instead of a ref, and judge
    its exit code with the runtime's own interpret_grep_rc."""
    completed = subprocess.run(
        _working_tree_grep_argv(root, pattern, paths),
        capture_output=True, text=True, timeout=30,
    )
    hits = [line.split("\0", 2) for line in completed.stdout.split("\n") if line]
    return _AnchorReading(
        outcome=interpret_grep_rc(completed.returncode, expect),
        paths_hit=frozenset(path for path, _column, _text in hits),
        defines_in_suite=any(
            path == suite and _match_starts_in_defined_name(text, int(column))
            for path, column, text in hits
        ),
        stderr=completed.stderr,
    )


@pytest.mark.parametrize("files,outcome,paths_hit,defines_in_suite", [
    pytest.param({"suite.py": "class TestAnchor:\n    pass\n"},
                 CheckOutcome.PASS, {"tests/suite.py"}, True,
                 id="definition-in-suite"),
    pytest.param({"suite.py": "# TestAnchor is only mentioned here\n"},
                 CheckOutcome.PASS, {"tests/suite.py"}, False,
                 id="prose-only-in-suite"),
    pytest.param({"suite.py": "def helper():  # TestAnchor\n"},
                 CheckOutcome.PASS, {"tests/suite.py"}, False,
                 id="prose-on-a-def-line"),
    pytest.param({"suite.py": "class TestAnchor:\n    pass\n",
                  "other.py": "# see TestAnchor\n"},
                 CheckOutcome.PASS, {"tests/suite.py", "tests/other.py"}, True,
                 id="leaks-outside-suite"),
    pytest.param({"suite.py": "x = 1\n", "other.py": "class TestAnchor:\n"},
                 CheckOutcome.PASS, {"tests/other.py"}, False,
                 id="definition-only-outside-suite"),
    pytest.param({"suite.py": "x = 1\n"},
                 CheckOutcome.FAIL, frozenset(), False,
                 id="matches-nothing"),
])
def test_anchor_reading_tells_a_definition_from_prose_a_leak_and_a_miss(
        tmp_path, files, outcome, paths_hit, defines_in_suite):
    """Each live identifier-anchor assertion can fail, shown on a synthetic repo."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    for name, text in files.items():
        (tests_dir / name).write_text(text, encoding="utf-8")
    git_init(tmp_path)

    reading = _read_identifier_anchor(
        str(tmp_path), pattern="TestAnchor", paths=["tests/"], expect="present",
        suite="tests/suite.py")

    assert (reading.outcome, reading.paths_hit, reading.defines_in_suite) == (
        outcome, paths_hit, defines_in_suite), reading.stderr


@pytest.mark.parametrize(
    "key,suite",
    list(_CERTIFIED_SUITE_BY_ANCHORED_ROW.items()),
    ids=[f"{task_id}-{capability}"
         for task_id, capability in _CERTIFIED_SUITE_BY_ANCHORED_ROW],
)
def test_identifier_anchor_names_a_definition_inside_its_certified_suite(key, suite):
    """A suite-exists row's live anchor matches on main, only inside its suite,
    and there names a class/def rather than prose the suite merely contains."""
    root = checkout_root_or_skip()

    rows = [r for r in _MEASURED_DRIFT_ROWS if (r[1], r[3]) == key]
    assert len(rows) == 1, (
        f"_CERTIFIED_SUITE_BY_ANCHORED_ROW key {key!r} resolves to {len(rows)} "
        f"_MEASURED_DRIFT_ROWS rows, expected exactly one"
    )
    relpath, task_id, label, capability, _stale, resynced = rows[0]
    assert resynced["expect"] == "present", (
        f"{key!r} is expect={resynced['expect']!r}; only a present-polarity row "
        f"can be an identifier anchor"
    )

    check = _live_delivered_check(root, relpath, label, capability)
    assert check.pattern is not None, (
        f"expected a grep check with a pattern, got kind={check.kind!r}"
    )
    reading = _read_identifier_anchor(
        root, pattern=check.pattern, paths=check.paths, expect=check.expect,
        suite=suite)

    where = (f"{relpath} label {label} (task {task_id}) anchor "
             f"{check.pattern!r} under {check.paths!r}")
    assert reading.outcome is CheckOutcome.PASS, (
        f"{where}: verdict {reading.outcome.value} (stderr: {reading.stderr!r}) — "
        f"_run_grep_check would not report DELIVERED"
    )
    assert reading.paths_hit == {suite}, (
        f"{where}: leaked outside the certified suite {suite}: "
        f"{sorted(reading.paths_hit)}"
    )
    assert reading.defines_in_suite, (
        f"{where}: no hit inside {suite} starts in the name a class/def defines"
    )
