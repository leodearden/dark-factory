"""Tripwire: no tracked sidecar's delivered_check may newly name a stale path.

Every tracked ``*.capability-manifest.yaml`` grep/path descriptor is swept
through the audit's stale-path pass at HEAD, so a file move or a
``sys.modules`` shim conversion that strands one fails in the MOVER's own
verify. The remedy is to repath BOTH halves (the sidecar here, and the task
record via ``update_task``), or to drop the descriptor. See
``docs/task-authoring.md`` §3.3.

MAINTENANCE CONTRACT: ``_KNOWN_STALE`` only shrinks. Delete a row when it is
repaired; never add one to make a new strand pass.
"""
from __future__ import annotations

import pytest
from audit_delivered_checks import StalePathSweep, load_manifest_checks, stale_path_findings
from git_checkout_root import checkout_root_or_skip

# Every row below was stranded by task 5036's move of the merge lane into
# orchestrator/merge_lane/ (ce8a05ab3c "Merge task/6174 into main"), which
# left these modules as sys.modules shims over the moved code.
_MERGE_QUEUE = "orchestrator/src/orchestrator/merge_queue.py"
_LANDING_EVIDENCE = "orchestrator/src/orchestrator/landing_evidence.py"
_MERGE_DRIFT = "orchestrator/src/orchestrator/merge_drift.py"
_MERGE_DISPOSITION = "orchestrator/src/orchestrator/merge_disposition.py"
_MERGE_GATES = "orchestrator/src/orchestrator/merge_gates.py"

_LANDED_NOT_DONE = "docs/prds/landed-not-done-recovery.capability-manifest.yaml"
_LIVE_SHADOW = "plans/live-shadow-eval-prd.capability-manifest.yaml"
_DURABLE_NON_LANDED = "plans/merge-status-durable-non-landed-prd.capability-manifest.yaml"
_VERDICT_INTEGRITY = "plans/merge-verdict-integrity-prd.capability-manifest.yaml"
_AMENDMENT_DELIVERY = "plans/task-amendment-delivery-prd.capability-manifest.yaml"

_KNOWN_STALE: frozenset[tuple[str, int, str, str]] = frozenset({
    (_LANDED_NOT_DONE, 4646, "parked-disposition-label-distinct-from-pruned", _MERGE_QUEUE),
    (_LANDED_NOT_DONE, 4646, "tally-key-pre-registered-or-the-label-is-swallowed", _MERGE_QUEUE),
    (_LANDED_NOT_DONE, 4646, "terminal-status-set-exists-and-is-importable", _MERGE_QUEUE),
    (_LANDED_NOT_DONE, 4647, "degenerate-branch-check-already-implemented", _LANDING_EVIDENCE),
    (_LANDED_NOT_DONE, 4647, "landing-evidence-module-already-exists-delta-extends-it",
     _LANDING_EVIDENCE),
    (_LANDED_NOT_DONE, 4647, "landing-verdict-dataclass-does-not-collide", _LANDING_EVIDENCE),
    (_LANDED_NOT_DONE, 4647, "no-op-rejection-reason-in-the-closed-vocabulary",
     _LANDING_EVIDENCE),
    (_LANDED_NOT_DONE, 4647, "patch-id-containment-helper-exists-and-is-reusable",
     _LANDING_EVIDENCE),
    (_LANDED_NOT_DONE, 4647, "reason-codes-must-be-registered-in-reason-explanations",
     _LANDING_EVIDENCE),
    (_LANDED_NOT_DONE, 4648, "branch-work-landed-contract-from-delta", _LANDING_EVIDENCE),
    (_LANDED_NOT_DONE, 4651, "landed-outbox-row-survives-for-parked-statuses-upstream",
     _MERGE_QUEUE),
    (_LANDED_NOT_DONE, 4651, "landing-contract-available-upstream", _LANDING_EVIDENCE),
    (_LANDED_NOT_DONE, 4652, "b9-needs-gammas-parked-vs-pruned-label", _MERGE_QUEUE),
    (_LIVE_SHADOW, 5388, "merge-landing-hook-wired", _MERGE_QUEUE),
    (_DURABLE_NON_LANDED, 4830, "finalize-payload-carries-superseded-by", _MERGE_QUEUE),
    (_VERDICT_INTEGRITY, 2886, "drift-counter-persisted", _MERGE_DRIFT),
    (_VERDICT_INTEGRITY, 2887, "foreign-drift-disposition", _MERGE_DISPOSITION),
    (_AMENDMENT_DELIVERY, 4033, "commit-ownership-gate-at-the-sibling-gate-site", _MERGE_GATES),
    (_AMENDMENT_DELIVERY, 4033, "gate-reads-the-descope-record", _MERGE_GATES),
})

_REMEDY = (
    "repath BOTH halves (this sidecar, and the task record via update_task) to "
    "where the code lives now, or drop the descriptor"
)


@pytest.fixture(scope="module")
def sweep() -> StalePathSweep:
    root = checkout_root_or_skip()
    rows, _ = load_manifest_checks(root, {})
    return stale_path_findings(rows, repo_root=root, ref="HEAD")


def _found(sweep: StalePathSweep) -> dict[tuple[str, int, str, str], str]:
    return {
        (f.row.manifest or "", f.row.task_id, f.row.name, f.scope.path):
            f.code + (f", removed in {f.scope.removed_in}" if f.scope.removed_in else "")
        for f in sweep.findings
    }


def test_no_sidecar_descriptor_newly_names_a_shim_or_removed_path(sweep):
    found = _found(sweep)
    new = sorted(set(found) - _KNOWN_STALE)

    assert new == [], "\n".join(
        f"{manifest}: task {task_id} capability {name!r} names {path} "
        f"[{found[(manifest, task_id, name, path)]}]; {_REMEDY}"
        for manifest, task_id, name, path in new
    )


def test_known_stale_rows_are_still_stale(sweep):
    repaired = sorted(_KNOWN_STALE - set(_found(sweep)))

    assert repaired == [], (
        "repaired: delete these rows from _KNOWN_STALE:\n"
        + "\n".join(str(row) for row in repaired)
    )


def test_every_swept_path_was_classified(sweep):
    assert sweep.paths_unclassified == 0, (
        "the stale-path classifier could not answer for some sidecar paths; "
        "the two tests above would otherwise read that as nothing stale"
    )
