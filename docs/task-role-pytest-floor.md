# The task-role pytest floor: narrowed to touched tests plus the guard family

Decided by Leo on gate 5702, 2026-10-20. Implemented by task 7253.

## The question

At `role='task'`, a diff that touches production code under a module fires the
task-3294 floor (`orchestrator/src/orchestrator/verify_plan.py::_derive_module_runs`,
arm 3). Before this ruling the floor ran the owning module's full pytest suite.
The gate's evidence was all from `orchestrator`, and it had three parts:

- the floor fired on 170 of 375 orchestrator task-role plans;
- the orchestrator merge leg takes 482 s;
- the floor made one out-of-scope catch, and a whole-tree guard test made it.

## The decision

At task role the floor runs the diff's touched collectable tests plus the owning
module's **whole-tree guard family**, as one FILE_SCOPED run. That family is
every test module whose module-level `pytestmark` is
`pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)`, and
`orchestrator/src/orchestrator/pytest_markers.py::is_whole_tree_guard_module`
decides membership.

The full-suite floor survives as the fallback in two cases:

- the module has no guard family (no lister was passed, or none was found);
- the module's own `-m` deselects the entire narrowed selection.

Its reason then ends ` — not narrowed: <why>`.

Merge role is unchanged. Under `merge_verify_breadth: full` the merge gate still
runs the full suite.

## Why it is safe

- **The merge gate re-runs the full suite.** The narrowing moves where a
  cross-consumer red is caught. It does not change whether the red is caught.
- **The only measured task-role catch was a guard test**, and the guard family
  still runs.
- **Scanners ⊆ family is enforced.**
  `orchestrator/tests/test_whole_tree_scan_timeout_guard.py` turns red on any
  detected `rglob`/`glob('*.py')` scanner that `is_whole_tree_guard_module`
  does not admit. Without that check a new scanner would silently drop out of
  task-role verify.
- **Coverage stays monotone in the diff.** A co-committed test only adds a
  target. That is the property task 3294 restored.

## When to revisit

Revisit whether to re-widen once the verify-result cache is live at task role
(tasks 5653 and 5657). Restoring the full suite means changing one function:
`orchestrator/src/orchestrator/verify_plan.py::_task_role_production_pytest_run`.
