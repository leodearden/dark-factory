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
`pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)`. The family's single home
is `shared/src/shared/whole_tree_guard_family.py`:
`::is_whole_tree_guard_module` decides membership, and
`::WHOLE_TREE_SCAN_TEST_TIMEOUT` is the ceiling the mark spells. Every leg
that runs a member must budget at least `::min_leg_budget_for_family(W)`, W
being its recorded worst green run, so the ceiling fires before verify kills
the leg; `tests/scripts/test_module_verify_budgets.py::test_every_leg_running_the_family_outlasts_its_ceiling`
gates it.

The full-suite floor survives as the fallback in two cases:

- the module has no guard family (no lister was passed, or none was found);
- the module's own `-m` deselects the entire narrowed selection.

Its reason then ends ` — not narrowed: <why>`.

Known gap: a module's family is enumerated from `<prefix>/tests` only, so the
scripts leg's family is scripts/tests alone. Until task 8402 gave scripts/tests
a family, a scripts/ diff fell back to the full scripts suite, tests/scripts
included; now the marked tests/scripts guards do not run at task role for it,
so for that leg this is a coverage regression, caught only at the merge gate.
Task 8503 owns the fix;
`tests/scripts/test_task_role_floor_narrows_every_leg.py::test_the_scripts_floor_runs_the_tests_scripts_guards`
is a strict xfail that turns red when it lands.

Known gap: task 5654 moved the whole-tree gates out of shared/tests, scripts/tests
and tests/scripts into tests/repo_guards. While they lived in shared/tests they
were shared family members, so a shared-only production diff ran them at task
role. Now they belong to no member's family, and a task-role leg runs only the
touched tests plus its own family, so no production diff runs them at task role.
They still run at every merge gate: tests/repo_guards declares
`verify_cache: never`, and `merge_verify_breadth: full` runs every module. This
is distinct from task 8503's gap, which concerns enumerating a family across
directories. The follow-up to plan the tests/repo_guards leg on task-role
production diffs was filed by task 5654.

Merge role is unchanged. Under `merge_verify_breadth: full` the merge gate still
runs the full suite.

## Why it is safe

- **The merge gate re-runs the full suite.** The narrowing moves where a
  cross-consumer red is caught. It does not change whether the red is caught.
- **The only measured task-role catch was a guard test**, and the guard family
  still runs.
- **Scanners ⊆ family is enforced**, by two mechanisms. For orchestrator,
  `orchestrator/tests/test_whole_tree_scan_timeout_guard.py` turns red on any
  detected `rglob`/`glob('*.py')` scanner that `is_whole_tree_guard_module`
  does not admit; the same check covers every module that consumes the shared
  tree pass (`orchestrator/tests/_tree_scan.py`), so a guard that stops globbing
  privately and joins the pass cannot drop out of the family either. For
  fused-memory, shared, scripts/tests, tests/scripts and tests/repo_guards,
  each tree's `test_repo_wide_guard_family.py` instance of
  `shared/src/shared/testing_repo_wide_tests.py::judge_repo_wide_family` turns
  red on any repo-wide test module outside the family. Without these checks a
  new scanner would silently drop out of task-role verify.
- **Each leg narrows on the real tree.**
  `tests/scripts/test_task_role_floor_narrows_every_leg.py` derives the
  task-role plan of the orchestrator, shared, fused-memory and scripts legs
  from the real module configs, and pins that none of them falls back to the
  full suite.
- **Coverage stays monotone in the diff.** A co-committed test only adds a
  target. That is the property task 3294 restored.

## When to revisit

Revisit whether to re-widen once the verify-result cache is live at task role
(tasks 5653 and 5657). Restoring the full suite means changing one function:
`orchestrator/src/orchestrator/verify_plan.py::_task_role_production_pytest_run`.
