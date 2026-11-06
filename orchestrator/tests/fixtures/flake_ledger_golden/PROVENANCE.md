# Provenance — `flake_ledger_golden` fixtures

Captured from base `0928b02294`, before task 6864 (offline-lane red-path ε1)
touched any flake source file. They record the merge gate's texts as they
were, so the claim that ε1 leaves them unchanged (apart from the cycle in the
de-flake title) is pinned against a record and not against the code itself.

| File | Producing code site (at capture) | Loader |
|---|---|---|
| `merge_gate_deflake_description.txt` | `orchestrator/src/orchestrator/flake_ledger.py::build_deflake_task_arguments`, the `description` key | `orchestrator/tests/_flake_golden.py::merge_gate_deflake_description` |
| `merge_gate_regression_detail.txt` | `orchestrator/src/orchestrator/flake_debt_drain.py::_regression_detail` | `orchestrator/tests/_flake_golden.py::merge_gate_regression_detail` |

Both are `string.Template` texts with no trailing newline. Every variable value
(test id, project, stamps, counts, commit, head, task id) is a `$placeholder`.
So are the three instruction constants (`RESPONSIBILITY_FIX_ROOT_DEFECT`,
`RESPONSIBILITY_REMOVE_FROM_LEDGER`, `NEVER_WIDEN_A_TIMEOUT`): the loader fills
them from the live `orchestrator.flake_ledger` values, so task 5850 can reword
those constants without touching these files.

Never hand-edit these bytes to make a red test green. A red pinning test means
the merge gate's text changed, and that change must be deliberate.
