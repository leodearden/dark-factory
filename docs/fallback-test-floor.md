# The fallback test floor: no fallback diff runs zero tests by accident

Implemented by task 8714; task 7041 adds the no_tests_ran outcome.

## The question

When a verify's resolved module set is empty (no task module maps to a
registered module config), verify takes the fallback branch
(`orchestrator/src/orchestrator/verify.py::_build_fallback_config`).
Before this change a `.py` diff with no derivable test target ran no tests at
all, at task role and at merge role alike. That is a source-only diff with no
touched test, conftest or test-data file. Lint and type check still ran.

The evidence:

- reify task 9761 touched `scripts/workspace-feature-union.py` together with
  `crates/workspace-feature-union/src/lib.rs`, `Cargo.toml`, `Cargo.lock` and
  `scripts/crate-strata.json`. Its merges, reify `ca362eada0` and `33578e66f4`,
  ran only `__fallback__:lint` and `__fallback__:type`. They landed
  deterministically red.
- reify task 9839 (`scripts/assert-crate-strata.py` alone) ran the same way.
- Task 7041 found the same hole from `solar_challenge_platform`.

The cargo scoper (`orchestrator/src/orchestrator/verify.py::_apply_cargo_scope`)
cannot restore a dropped command. Whenever the fallback fires there is a `.py`
file, and `.py` is outside its polyglot safe-list.

## The rule

`orchestrator/src/orchestrator/verify_fallback_floor.py::fallback_test_floor_reason`
decides whether the fallback's test slot becomes `config.test_command`, run
verbatim. It checks these arms in order, and the first that applies wins:

1. **No test command configured** (`None` or empty): no floor. The project
   has explicitly opted out of tests.
2. **The diff carries non-Python source** (`.rs`): floor, at both roles. This
   applies even when a Python target was derived, because the non-Python part
   has no Python-derived target.
3. **A test target was derived**: no floor. The existing narrowing stands
   (tasks 1852, 2344, 2368).
4. **A non-default test command is configured**: floor, at both roles. A
   project that declares a command has said what its gate is.
5. **Only the bare `pytest` default, at merge role**: floor. Pytest's rc=5
   ("no tests ran") stays red under task 1852, so a tree with no tests fails
   the merge gate instead of reading green.
6. **Only the bare `pytest` default, at task role**: no floor. The task-1852
   skip still runs no pytest, and its plan record still reads SKIPPED with a
   reason, but the verdict is now `no_tests_ran` (see below).

The rule is applied at two points, and each record states the floor only for
its own layer:

- The plan decision
  (`orchestrator/src/orchestrator/verify_plan.py::_derive_fallback_runs`,
  reached through `derive_verify_plan`) floors on its idealized flat file
  list.
- The executed config
  (`orchestrator/src/orchestrator/verify.py::_run_scoped_verification_body`)
  floors on `_build_fallback_config`'s rescoped output. Its reason reaches
  `VerifyResult.plan` only through
  `orchestrator/src/orchestrator/verify.py::_executed_fallback_plan`'s
  `pytest_reason`, because
  `orchestrator/src/orchestrator/verify.py::_safe_derive_executed_fallback_plan`
  reconciles the decision derived without its floor.

The two layers can disagree. The known case is the task-2368 mixed
root+single-subproject source-only diff: the decision floors, while execution
runs the mixed branch's own derived command. `VerifyResult.plan` then does not
claim the floor.

## When the fallback still runs no tests

`orchestrator/src/orchestrator/verify_fallback_floor.py::fallback_ran_no_tests`
is applied by
`orchestrator/src/orchestrator/verify.py::_no_tests_ran_verdict`. A
passing fallback verify whose executed test slot is empty, while the project
configures a `test_command`, fails with category `no_tests_ran`, at either
role.

- Today only arm 6 reaches it. At merge role the floor already fills the slot,
  so the check is the second enforcement point behind it.
- A lint or type red from the same run is reported as itself.
- Its `cause_hint` addresses the agent: add or modify a test and declare it in
  the task's files. The fallback derives test targets from the task's declared
  files, not from the diff, so an undeclared new test repeats the failure. The
  operator remedies below are deliberately left out of the hint.
- `orchestrator/src/orchestrator/verify.py::verify_failure_is_preexisting_on_main`
  never probes a `no_tests_ran` failure: main would reproduce it whenever it
  runs the same fallback over the same files.
- `verify_allow_no_tests_ran: true` lets it pass, with a WARNING. The knob is
  read from the local config only until task 7067 carries it on the verify
  specs.
- The category is a string sentinel until S4 α (task 9128) makes it a
  `FailureCategory` member.

## Who it affects

As of 2026-11-10, every project except autopilot-video declares a
non-default `test_command`, so arms 5 and 6 matter only to autopilot-video.

In dark_factory every subproject registers a module config, and so do
`scripts/` and `tests/scripts/`. But the fallback is chosen by an empty
resolved module set, not by file path.
`orchestrator/src/orchestrator/workflow.py::TaskWorkflow._resolve_module_configs`
returns `[]` when none of the task's modules maps to a registered module
config. That covers a task whose declared files sit only under `skills/**`,
`plans/**` or the repo-root `.py` files. It also covers a task that declared
no files at all: the scheduler then uses the synthetic `task-<id>` lock,
whatever files the diff actually touches.

This holds at task role and at merge role. `merge_verify_breadth: full` does
not widen an empty set
(`orchestrator/src/orchestrator/merge_lane/worker.py::_merge_boundary_module_configs`),
and dark_factory does not set `merge_verify_workspace`. So for such a task, a
source-only `.py` diff with no derivable test target now runs the fleet
`test_command` chain at both roles, segmented at task role and whole at merge
role. Before, it ran no tests.

Registering a module config for `skills/**` or `plans/**` (task 9068) narrows
only tasks whose declared files sit there. It does not change a task that
declared no files.

## Opting out, or narrowing

- To run no tests at all, set `test_command: "true"`, as autotrade does. An
  empty `test_command` also opts out of tests entirely.
- To accept a fallback verify whose test slot runs nothing, while keeping
  tests wherever one is derivable, set `verify_allow_no_tests_ran: true`.
- For narrower coverage of a path, register a module config for it.
  `tests/scripts/orchestrator.yaml` did exactly that for task 3350. This
  narrows only tasks whose declared files sit under that path; a task that
  declared no files still resolves an empty module set and takes the
  fallback.

## What it does not change

- `_build_fallback_config`'s narrowing when a test target is derivable. Where
  it derives none, the floor overrides it. In particular, a source-only diff
  under a single real subproject (task 2344's pure-subproject branch) now runs
  the whole configured `test_command`, not that subproject's suite alone.
- The module-config branch.
- `force_workspace` and `merge_verify_workspace`.
