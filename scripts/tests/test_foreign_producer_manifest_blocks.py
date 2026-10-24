"""Live-corpus pin that a capability-manifest block whose producer lives in
another project's registry binds ``external_task_id`` and never a local
``task_id``, and so contributes no descriptor row to this project's audits.
Reads only TRACKED GIT FILES and opens no database — the same legitimacy as
shared/tests/test_capability_manifest.py::TestCheckedInManifestCorpus.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from audit_delivered_checks import load_manifest_checks
from git_checkout_root import checkout_root_or_skip
from shared.capability_manifest import load_capability_manifest

# ---------------------------------------------------------------------------
# The tracked manifest blocks whose PRODUCER lives in another project's
# registry, not dark-factory's: (manifest relpath, label, expected
# external_task_id).
#
# PROVENANCE (measured by tasks 4731 and 6047). Each id is a live
# `tag='master'` row in that registry's .taskmaster/tasks/tasks.db whose
# metadata.prd_task_label equals the block's label and whose prd_path names
# this sidecar's PRD. The registry spelling — the half before the colon — is
# that project root's dark-factory-orchestrator.yaml `fused_memory.project_id`.
# ---------------------------------------------------------------------------
_EXTERNAL_REGISTRY_BLOCKS = (
    ("plans/os-sandbox-worktree-containment-prd.capability-manifest.yaml",
     "γ6a", "reify:5332"),
    ("plans/os-sandbox-worktree-containment-prd.capability-manifest.yaml",
     "γ6b", "reify:5333"),
    ("plans/warm-lane-infra-repatriation-prd.capability-manifest.yaml",
     "η", "reify:5613"),
    ("plans/warm-lane-infra-repatriation-prd.capability-manifest.yaml",
     "θ", "reify:5614"),
    ("plans/warm-lane-infra-repatriation-prd.capability-manifest.yaml",
     "ι", "reify:5615"),
    ("plans/warm-lane-infra-repatriation-prd.capability-manifest.yaml",
     "κ", "reify:5616"),
    ("plans/load-throttle-harmonisation-prd.capability-manifest.yaml",
     "ρ1", "reify:7334"),
    ("plans/load-throttle-harmonisation-prd.capability-manifest.yaml",
     "ρ2", "reify:7335"),
    ("plans/load-throttle-harmonisation-prd.capability-manifest.yaml",
     "ρ3", "reify:7336"),
    ("plans/os-sandbox-worktree-containment-prd.capability-manifest.yaml",
     "γ7a", "autopilot_video:649"),
    ("plans/os-sandbox-worktree-containment-prd.capability-manifest.yaml",
     "γ7b", "know_live:592"),
    ("plans/os-sandbox-worktree-containment-prd.capability-manifest.yaml",
     "γ7c", "pump_web_ui:16"),
    ("plans/os-sandbox-worktree-containment-prd.capability-manifest.yaml",
     "γ7d", "solar_challenge:95"),
    ("plans/os-sandbox-worktree-containment-prd.capability-manifest.yaml",
     "γ7e", "solar_challenge_platform:164"),
)

_BLOCK_IDS = [label for _, label, _ in _EXTERNAL_REGISTRY_BLOCKS]


@pytest.mark.parametrize(
    "relpath,label,external_task_id",
    _EXTERNAL_REGISTRY_BLOCKS,
    ids=_BLOCK_IDS,
)
def test_live_foreign_producer_blocks_declare_their_external_registry(
        relpath, label, external_task_id):
    """The foreign-producer blocks bind external_task_id, not task_id.

    Parametrized so a failure NAMES its own manifest and label rather than
    reporting "one of N".

    MAINTENANCE CONTRACT — READ THIS BEFORE "FIXING" A FAILURE HERE. These ids
    are not guesses: each was verified as a live `tag='master'` row in its
    registry's task store with a label matching its block (see the provenance
    note on _EXTERNAL_REGISTRY_BLOCKS above). A failure here means a SIDECAR
    WAS EDITED, not that the test needs relaxing. Two shapes to expect:

      - `task_id` came back on one of these blocks. That is the exact hazard
        the model's mutual-exclusion validator guards, arriving through the
        one path the validator cannot see coming: a dark-factory decompose
        re-using one of these Greek labels, which manifest_stamping step 4
        stamps without consulting the block's existing contents. The producer
        is still foreign; do NOT let the local stamp stand.
      - the label or the id changed. Confirm against its registry FIRST,
        then update the row here in the SAME commit.
    """
    root = checkout_root_or_skip()

    # NON-VACUITY FLOOR: assert the sidecar is TRACKED before reading it, so a
    # renamed or deleted manifest cannot make this test pass by finding
    # nothing to check.
    tracked = subprocess.run(
        ["git", "-C", root, "ls-files", "--", relpath],
        capture_output=True, text=True, timeout=30,
    )
    assert tracked.stdout.strip(), f"{relpath} is not tracked in {root}"

    doc = load_capability_manifest(Path(root) / relpath)
    matches = [task for task in doc.tasks if task.label == label]
    assert len(matches) == 1, (
        f"expected exactly one task block labelled {label!r} in {relpath}, "
        f"found {len(matches)}"
    )

    block = matches[0]
    assert block.external_task_id == external_task_id, (
        f"{relpath} label {label} binds external_task_id "
        f"{block.external_task_id!r}, expected {external_task_id!r} — see this "
        f"test's maintenance contract before changing the expectation."
    )
    # THE OTHER HALF, and the one that actually matters: a local task_id here
    # would misattribute a foreign task's capability claim to a dark-factory
    # one.
    assert block.task_id is None, (
        f"{relpath} label {label} carries a LOCAL task_id ({block.task_id}) "
        f"alongside its foreign producer {external_task_id!r}."
    )


@pytest.fixture(scope="module")
def live_manifest_rows():
    return load_manifest_checks(checkout_root_or_skip(), {})[0]


@pytest.mark.parametrize(
    "relpath,label,external_task_id",
    _EXTERNAL_REGISTRY_BLOCKS,
    ids=_BLOCK_IDS,
)
def test_live_foreign_producer_block_contributes_no_audit_row(
        live_manifest_rows, relpath, label, external_task_id):
    """A foreign-bound block yields no row for this project's audits.

    The audits join a sidecar block to this project's task store by its
    NUMBER. Task ids are per-registry, so a foreign id left in `task_id`
    lands on whatever unrelated local task holds that number once this
    project's counter passes it. That already happened: dark-factory's ids
    passed reify's, and ρ1–ρ3 misjoined to unrelated DF 7334–7336, where
    their grep and path checks read as false broken / vacuous / unwired
    live gates.
    """
    leaked = [
        (row.task_id, row.name)
        for row in live_manifest_rows
        if row.manifest == relpath and row.task_label == label
    ]
    assert not leaked, (
        f"{relpath} label {label} (producer {external_task_id}) leaked "
        f"{len(leaked)} audit row(s) {leaked}: its foreign producer is being "
        f"joined by NUMBER to a local task."
    )
