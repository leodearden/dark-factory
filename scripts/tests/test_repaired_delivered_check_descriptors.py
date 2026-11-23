"""Live-corpus pins for repaired capability-manifest delivered_check descriptors.

Three populations are pinned. Task 6036 re-anchored descriptors on BOTH sides: the
producer task's ``metadata.delivered_checks`` entry in tasks.db AND the sidecar
the stamper copies it from; task 9538 did the same for task 8461's file-existence
gate, which a docstring had been satisfying as a grep. Tasks 5256 and 6209
resynced sidecars, sidecar-only, to task records that had already been repaired. CI can never see the tasks.db half, so
either way these pins guard the SIDECAR half: a stale sidecar would be
re-stamped over the repaired record by the next re-decompose. Every resynced
producer is done, as is 8461, so those rows are also evaluated at HEAD through
the runtime primitive for their kind. They read only tracked files in this
checkout and open no database. Task 6047 re-pointed a FOREIGN descriptor, warm-lane κ, whose
task-record half lives in reify's store; reify's tree is not this checkout, so
its non-vacuity is checked behaviourally against committed fixture trees rather
than at HEAD.

MAINTENANCE CONTRACT: an exact pin also fires on a legitimate later both-sides
re-repair. In that case, update the row here in the same change and re-run
``scripts/audit_manifest_descriptor_drift.py --project-root <primary>
--manifest-root <this checkout>`` to confirm zero drift for that task.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from audit_delivered_checks import evaluate_row, load_manifest_checks, structural_findings
from committed_git_repo import committed_repo
from git_checkout_root import checkout_root_or_skip
from shared.capability_manifest import load_capability_manifest
from shared.delivered_check_polarity import CheckOutcome, evaluate_path_at_tree
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

# Task 9538: task 8461's path gate, pinned below and behind the fixture-tree tests.
_QUALITY_BOUNDARY = (
    "plans/quality-dashboard-prd.capability-manifest.yaml",
    8461, "ε", "boundary-test-for-quality-payload",
    {"kind": "path", "pattern": None, "expect": "present",
     "paths": ["dashboard/tests/js/boundary_quality.test.mjs"]},
)

# Tasks 6036 and 9538: re-anchored on both sides.
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
    _QUALITY_BOUNDARY,
]

# Tasks 5256 and 6209: sidecar resynced to the already-repaired task record.
_RESYNCED_TO_TASK_RECORD = [
    (
        "plans/flake-ledger-prd.capability-manifest.yaml",
        3787, "γ", "merge-boundary-call-sites-exist",
        {"kind": "grep", "pattern": r"effective_merge_module_configs\(", "expect": "present",
         "paths": ["orchestrator/src/orchestrator/merge_lane/post_merge_verify.py"]},
    ),
    (
        "plans/flake-ledger-prd.capability-manifest.yaml",
        3789, "ε", "remote-path-drops-all-three-side-effects-today",
        {"kind": "grep", "pattern": "record_merge_flake_suppression", "expect": "present",
         "paths": ["orchestrator/src/orchestrator/merge_lane/post_merge_verify.py"]},
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
    (
        "plans/uuid-prefix-resolution-prd.capability-manifest.yaml",
        5323, "β", "resolver-unavailable-raises",
        {"kind": "grep", "pattern": r"raise ResolverUnavailable\(", "expect": "present",
         "paths": ["fused-memory/src/fused_memory/services/id_resolver.py"]},
    ),
]

_REPAIRED_DESCRIPTORS = [*_REANCHORED, *_RESYNCED_TO_TASK_RECORD]

_RETIRED_TO_MANUAL = [
    (
        "plans/plan-deviation-recording-prd.capability-manifest.yaml",
        5754, "δ", "watcher-per-task-limit-kept",
    ),
]

# Task 6047: re-pointed foreign descriptor (producer reify:5616).
_KAPPA = (
    "plans/warm-lane-infra-repatriation-prd.capability-manifest.yaml",
    "κ", "reify-scripts-reduced-to-contract-primitives",
)
# plans/warm-lane-infra-repatriation-prd.md §κ: "Remove the seven relocated scripts".
_RELOCATED_WARM_LANE_SCRIPTS = (
    "scripts/warm-lane-gc.sh",
    "scripts/warm-lane-gc-sweep.sh",
    "scripts/thin-warm-lane.sh",
    "scripts/warm-lane-disk-guard.sh",
    "scripts/warm-lane-audit.sh",
    "scripts/warm-lane-degenerate-ref-check.sh",
    "scripts/provision-warm-lane-fs.sh",
)
# PRD §5: κ keeps the project's one primitive and its base-advance counterpart.
_CONTRACT_PRIMITIVES = ("scripts/seed-warm-lane.sh", "scripts/refresh-warm-base.sh")


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


def _kappa_check():
    relpath, label, capability = _KAPPA
    check = _the_one_capability(
        checkout_root_or_skip(), relpath, label, capability).delivered_check

    assert check is not None
    assert check.kind == "path" and check.expect == "absent", (
        f"{relpath} label {label} capability {capability} is kind={check.kind!r} "
        f"expect={check.expect!r}, not path/absent: either the vacuous "
        f"scripts/verify-pipeline-paths.txt grep is back or the check changed kind."
    )
    return check


@pytest.fixture(scope="module")
def sidecar_rows():
    """Every tracked sidecar's grep and path descriptors, as the drift audit reads them."""
    return load_manifest_checks(checkout_root_or_skip(), {})[0]


def _the_one_row(rows, relpath: str, label: str, capability: str):
    matches = [row for row in rows
               if (row.manifest, row.task_label, row.name) == (relpath, label, capability)]
    assert len(matches) == 1, (
        f"expected exactly one grep or path {capability!r} under label {label!r} "
        f"in {relpath}, found {len(matches)}"
    )
    return matches[0]


def _reify_tree_holding(tmp_path: Path, scripts) -> str:
    return committed_repo(tmp_path / "reify", dict.fromkeys(scripts, "#!/bin/sh\n"))


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


# Rows whose producer is done. The rest of _REANCHORED is left out: it holds a
# pending producer (5324) whose forward-looking gate legitimately fails at HEAD.
_DONE_PRODUCER_HEADS = [r[:4] for r in [*_RESYNCED_TO_TASK_RECORD, _QUALITY_BOUNDARY]]


@pytest.mark.parametrize(
    "relpath,task_id,label,capability",
    _DONE_PRODUCER_HEADS,
    ids=[f"{r[1]}-{r[3]}" for r in _DONE_PRODUCER_HEADS],
)
def test_done_producer_sidecar_descriptor_delivers_at_head(
        sidecar_rows, relpath, task_id, label, capability):
    row = _the_one_row(sidecar_rows, relpath, label, capability)

    outcome = evaluate_row(row, repo_root=checkout_root_or_skip(), ref="HEAD")
    assert outcome is CheckOutcome.PASS, (
        f"{relpath} label {label} capability {capability} (task {task_id}) evaluates "
        f"{outcome.value} at HEAD, but its producer is done, so its sidecar "
        f"delivered_check must deliver. If code moved, repath BOTH halves (the task "
        f"record via update_task, the sidecar here) and update this module's row per "
        f"the MAINTENANCE CONTRACT."
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


def test_audit_structural_section_names_no_reanchored_grep_capability(sidecar_rows):
    root = checkout_root_or_skip()
    reanchored_greps = {(r[1], r[3]) for r in _REANCHORED if r[4]["kind"] == "grep"}
    rows = [row for row in sidecar_rows if (row.task_id, row.name) in reanchored_greps]

    assert {(row.task_id, row.name) for row in rows} == reanchored_greps

    flagged = {(s.row.task_id, s.row.name, s.code)
               for s in structural_findings(rows, repo_root=root, ref="HEAD")}
    assert flagged == set()


@pytest.mark.parametrize(
    "survivor", _RELOCATED_WARM_LANE_SCRIPTS, ids=lambda path: Path(path).name,
)
def test_kappa_gate_holds_while_any_relocated_script_survives(tmp_path, survivor):
    """Every one of κ's seven deletions is gated, so the gate cannot pass early."""
    check = _kappa_check()
    repo = _reify_tree_holding(tmp_path, [*_CONTRACT_PRIMITIVES, survivor])

    outcome = evaluate_path_at_tree(
        list(check.paths), expect=check.expect, repo_root=repo, ref="HEAD",
    )
    assert outcome is CheckOutcome.FAIL, (
        f"κ's delivered_check evaluates {outcome.value} on a tree that still holds "
        f"{survivor}; it must gate every relocated script's deletion."
    )


def test_kappa_gate_opens_once_only_the_contract_primitives_remain(tmp_path):
    """No retained contract primitive is in the gate, so κ can actually finish."""
    check = _kappa_check()
    repo = _reify_tree_holding(tmp_path, _CONTRACT_PRIMITIVES)

    outcome = evaluate_path_at_tree(
        list(check.paths), expect=check.expect, repo_root=repo, ref="HEAD",
    )
    assert outcome is CheckOutcome.PASS, (
        f"κ's delivered_check evaluates {outcome.value} on a tree holding only "
        f"{_CONTRACT_PRIMITIVES}; a retained primitive must not be in the gate."
    )


@pytest.mark.parametrize(
    "files,expected",
    [
        pytest.param(
            {"dashboard/tests/test_boundary_js.py":
                '"""``js/boundary_quality.test.mjs`` holds their client half."""\n'},
            CheckOutcome.FAIL, id="prose-only",
        ),
        pytest.param(
            {"dashboard/tests/js/boundary_quality.test.mjs":
                "import test from 'node:test';\n"},
            CheckOutcome.PASS, id="file-only",
        ),
    ],
)
def test_quality_boundary_gate_follows_the_test_file_not_the_prose(
        tmp_path, sidecar_rows, files, expected):
    """esc-8461-1's two failure modes, inverted: prose naming the file must not
    deliver, and the file must deliver whatever it says."""
    relpath, _task_id, label, capability, _expected = _QUALITY_BOUNDARY
    row = _the_one_row(sidecar_rows, relpath, label, capability)
    repo = committed_repo(tmp_path / "dashboard", files)

    outcome = evaluate_row(row, repo_root=repo, ref="HEAD")
    assert outcome is expected, (
        f"task 8461's delivered_check (kind={row.kind!r}) evaluates {outcome.value} "
        f"on a tree holding only {sorted(files)}; it must follow the test file's "
        f"existence, not any mention of its name."
    )
