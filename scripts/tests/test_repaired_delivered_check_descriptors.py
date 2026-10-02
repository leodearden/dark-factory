"""Live-corpus pins for repaired capability-manifest delivered_check descriptors.

Two populations are pinned. Task 6036 re-anchored descriptors on BOTH sides: the
producer task's ``metadata.delivered_checks`` entry in tasks.db AND the sidecar
the stamper copies it from. Task 5256 resynced sidecars, sidecar-only, to task
records that had already been repaired. CI can never see the tasks.db half, so
either way these pins guard the SIDECAR half: a stale sidecar would be
re-stamped over the repaired record by the next re-decompose.
They read only tracked files in this checkout and open no database.

MAINTENANCE CONTRACT: an exact pin also fires on a legitimate later both-sides
re-repair. In that case, update the row here in the same change and re-run
``scripts/audit_manifest_descriptor_drift.py --project-root <primary>
--manifest-root <this checkout>`` to confirm zero drift for that task.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from audit_delivered_checks import load_manifest_checks, structural_findings
from git_checkout_root import checkout_root_or_skip
from shared.capability_manifest import load_capability_manifest

# Task 6036: re-anchored on both sides.
_REANCHORED = [
    (
        "docs/prds/recurring-deterministic-tasks.capability-manifest.yaml",
        4681, "r6", "predicate-variant-script-exists",
        {"kind": "path", "pattern": None, "expect": "present",
         "paths": ["scripts/reclaim-orphaned-worktrees-predicate.sh"]},
    ),
    (
        "plans/uuid-prefix-resolution-prd.capability-manifest.yaml",
        5324, "δ", "recon-harness-passes-service-resolver",
        {"kind": "grep", "pattern": "resolver=", "expect": "present",
         "paths": ["fused-memory/src/fused_memory/reconciliation/harness.py"]},
    ),
    (
        "plans/agent-transcript-archival-prd.capability-manifest.yaml",
        2792, "β", "backstop-at-cleanup-worktree-chokepoint",
        {"kind": "grep", "pattern": "archive_before_delete", "expect": "present",
         "paths": ["orchestrator/src/orchestrator/git_ops.py"]},
    ),
    (
        "plans/fable-architect-eval-admission-prd.capability-manifest.yaml",
        2862, "τ1", "eval-bootstrap-smoke-gate",
        {"kind": "path", "pattern": None, "expect": "present",
         "paths": ["scripts/eval_bootstrap_smoke.sh"]},
    ),
]

# Task 5256: sidecar resynced to the already-repaired task record.
_RESYNCED_TO_TASK_RECORD = [
    (
        "plans/flake-ledger-prd.capability-manifest.yaml",
        3789, "ε", "remote-path-drops-all-three-side-effects-today",
        {"kind": "grep", "pattern": "record_merge_flake_suppression", "expect": "present",
         "paths": ["orchestrator/src/orchestrator/merge_queue.py"]},
    ),
    (
        "plans/merge-lane-throughput-prd.capability-manifest.yaml",
        5051, "B", "setup-md-remote-verify-host-section",
        {"kind": "grep", "pattern": "remote merge-verify host", "expect": "present",
         "paths": ["SETUP.md"]},
    ),
    (
        "plans/session-resume-eligibility-seam-prd.capability-manifest.yaml",
        3733, "ε", "storm-prose-no-longer-misdirects-to-ntp",
        {"kind": "grep", "pattern": r"clock skew \(NTP\)", "expect": "absent",
         "paths": ["orchestrator/src/orchestrator/harness.py"]},
    ),
    (
        "plans/memory-referent-fidelity-prd.capability-manifest.yaml",
        3669, "δ", "entities-param-on-add-memory",
        {"kind": "grep", "pattern": r"entities_gate\(", "expect": "present",
         "paths": ["fused-memory/src/fused_memory/server/tools.py"]},
    ),
    (
        "plans/dashboard-one-datum-one-path-prd.capability-manifest.yaml",
        5588, "γ1", "registry-receipt-metadata",
        {"kind": "grep", "pattern": "_received_at", "expect": "present",
         "paths": ["dashboard/src/dashboard/static/redux/data.js",
                   "dashboard/src/dashboard/static/redux/datum.js"]},
    ),
]

_REPAIRED_DESCRIPTORS = [*_REANCHORED, *_RESYNCED_TO_TASK_RECORD]

_RETIRED_TO_MANUAL = [
    (
        "plans/plan-deviation-recording-prd.capability-manifest.yaml",
        5754, "δ", "watcher-per-task-limit-kept",
    ),
]


def _the_one_capability(root: str, relpath: str, label: str, capability: str):
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
    return matches[0]


@pytest.mark.parametrize(
    "relpath,task_id,label,capability,expected",
    _REPAIRED_DESCRIPTORS,
    ids=[f"{r[1]}-{r[3]}" for r in _REPAIRED_DESCRIPTORS],
)
def test_repaired_sidecar_carries_the_repaired_descriptor(
        relpath, task_id, label, capability, expected):
    root = checkout_root_or_skip()
    check = _the_one_capability(root, relpath, label, capability).delivered_check

    assert check is not None
    dump = check.model_dump()
    assert {key: dump[key] for key in expected} == expected, (
        f"{relpath} label {label} capability {capability} (task {task_id}) does not "
        f"carry the repaired descriptor pinned in this module. If it was re-repaired "
        f"on BOTH sides on purpose, follow this module's MAINTENANCE CONTRACT."
    )


@pytest.mark.parametrize(
    "relpath,task_id,label,capability",
    _RETIRED_TO_MANUAL,
    ids=[f"{r[1]}-{r[3]}" for r in _RETIRED_TO_MANUAL],
)
def test_retired_capability_is_manual_in_its_sidecar(relpath, task_id, label, capability):
    root = checkout_root_or_skip()
    check = _the_one_capability(root, relpath, label, capability).delivered_check

    assert check is not None
    assert check.kind == "manual", (
        f"{relpath} label {label} capability {capability} (task {task_id}) is "
        f"kind={check.kind!r}; task 6036 retired it to manual. If it was "
        f"re-repaired on BOTH sides on purpose, follow this module's MAINTENANCE CONTRACT."
    )
    assert isinstance(check.reason, str) and check.reason.strip()


def test_audit_structural_section_names_no_reanchored_grep_capability():
    root = checkout_root_or_skip()
    reanchored_greps = {(r[1], r[3]) for r in _REANCHORED if r[4]["kind"] == "grep"}
    rows = [row for row in load_manifest_checks(root, {})[0]
            if (row.task_id, row.name) in reanchored_greps]

    assert {(row.task_id, row.name) for row in rows} == reanchored_greps

    flagged = {(s.row.task_id, s.row.name, s.code)
               for s in structural_findings(rows, repo_root=root, ref="HEAD")}
    assert flagged == set()
