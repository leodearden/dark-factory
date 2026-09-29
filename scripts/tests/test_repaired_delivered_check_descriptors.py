"""Live-corpus pins for delivered_check descriptors that task 6036 repaired on BOTH sides.

Each repair changed a producer task's ``metadata.delivered_checks`` entry in
tasks.db AND the capability-manifest sidecar the stamper copies it from. CI can
never see the tasks.db half, so these pins guard the SIDECAR half: a reverted
sidecar would be re-stamped over the repaired record by the next re-decompose.
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
from shared.capability_manifest import load_capability_manifest

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

_RETIRED_TO_MANUAL = [
    (
        "plans/plan-deviation-recording-prd.capability-manifest.yaml",
        5754, "δ", "watcher-per-task-limit-kept",
    ),
]


def _repo_root():
    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _checkout_root() -> str:
    root = _repo_root()
    if root is None:
        pytest.skip("not a git checkout")
    return root


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
    _REANCHORED,
    ids=[f"{r[1]}-{r[3]}" for r in _REANCHORED],
)
def test_repaired_sidecar_carries_the_repaired_descriptor(
        relpath, task_id, label, capability, expected):
    root = _checkout_root()
    check = _the_one_capability(root, relpath, label, capability).delivered_check

    assert check is not None
    assert check.model_dump() == {
        **expected, "script": None, "args": [], "timeout_secs": None, "reason": None,
    }, (
        f"{relpath} label {label} capability {capability} (task {task_id}) does not "
        f"carry the task-6036 repaired descriptor. If it was re-repaired on BOTH "
        f"sides on purpose, follow this module's MAINTENANCE CONTRACT."
    )


@pytest.mark.parametrize(
    "relpath,task_id,label,capability",
    _RETIRED_TO_MANUAL,
    ids=[f"{r[1]}-{r[3]}" for r in _RETIRED_TO_MANUAL],
)
def test_retired_capability_is_manual_in_its_sidecar(relpath, task_id, label, capability):
    root = _checkout_root()
    check = _the_one_capability(root, relpath, label, capability).delivered_check

    assert check is not None
    assert check.kind == "manual", (
        f"{relpath} label {label} capability {capability} (task {task_id}) is "
        f"kind={check.kind!r}; task 6036 retired it to manual. If it was "
        f"re-repaired on BOTH sides on purpose, follow this module's MAINTENANCE CONTRACT."
    )
    assert isinstance(check.reason, str) and check.reason.strip()


def test_audit_structural_section_names_no_repaired_capability():
    root = _checkout_root()
    repaired = {(r[1], r[3]) for r in _REANCHORED} | {(r[1], r[3]) for r in _RETIRED_TO_MANUAL}
    rows = [row for row in load_manifest_checks(root, {})[0]
            if (row.task_id, row.name) in repaired]

    loaded = {(row.task_id, row.name) for row in rows}
    for _relpath, task_id, _label, capability, expected in _REANCHORED:
        if expected["kind"] == "grep":
            assert (task_id, capability) in loaded, (
                f"task {task_id} {capability!r} is not among the audit's grep rows"
            )

    flagged = {(s.row.task_id, s.row.name, s.code)
               for s in structural_findings(rows, repo_root=root, ref="HEAD")}
    assert flagged == set()
