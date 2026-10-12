"""Tripwire: no tracked sidecar's delivered_check may name a stale path.

Every tracked ``*.capability-manifest.yaml`` grep/path descriptor is swept
through the audit's stale-path pass at HEAD, so a file move or a
``sys.modules`` shim conversion that strands one fails in the MOVER's own
verify. The remedy is to repath BOTH halves (the sidecar here, and the task
record via ``update_task``), or to drop the descriptor. See
``docs/task-authoring.md`` §3.3.
"""
from __future__ import annotations

import pytest
from audit_delivered_checks import StalePathSweep, load_manifest_checks, stale_path_findings
from git_checkout_root import checkout_root_or_skip

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


def test_no_sidecar_descriptor_names_a_shim_or_removed_path(sweep):
    found = _found(sweep)

    assert found == {}, "\n".join(
        f"{manifest}: task {task_id} capability {name!r} names {path} "
        f"[{code}]; {_REMEDY}"
        for (manifest, task_id, name, path), code in sorted(found.items())
    )


def test_every_swept_path_was_classified(sweep):
    assert sweep.paths_unclassified == 0, (
        "the stale-path classifier could not answer for some sidecar paths; "
        "the test above would otherwise read that as nothing stale"
    )
