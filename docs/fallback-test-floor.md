# The fallback test floor: no fallback diff runs zero tests by accident

Implemented by task 8714, which also lands the core of task 7041.

## The question

When a project has no module config for a path, verify takes the fallback
branch (`orchestrator/src/orchestrator/verify.py::_build_fallback_config`).
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
   skip is kept, and its plan record still reads SKIPPED with a reason.

The same rule is applied twice. The plan decision
(`orchestrator/src/orchestrator/verify_plan.py::_derive_fallback_runs`)
records a floored test run as FULL_SUITE with the floor's reason. The
executed config (`orchestrator/src/orchestrator/verify.py::_run_scoped_verification_body`)
runs it, and the reason reaches `VerifyResult.plan` through
`orchestrator/src/orchestrator/verify.py::_executed_fallback_plan`'s
`pytest_reason`.

## Who it affects

As of 2026-11-10, every project except autopilot-video declares a
non-default `test_command`, so arms 5 and 6 matter only to autopilot-video.

In dark_factory every subproject registers a module config, and so do
`scripts/` and `tests/scripts/`. Only `skills/**`, `plans/**` and the
repo-root `.py` files reach the fallback, and only at task role:
`merge_verify_breadth: full` routes the merge gate through the full registry.
A task-role diff confined to those paths now runs the fleet `test_command`
chain, segmented, where it previously ran no tests.

## Opting out, or narrowing

- To run no tests at all, set `test_command: "true"`, as autotrade does.
- For narrower coverage of a path, register a module config for it.
  `tests/scripts/orchestrator.yaml` did exactly that for task 3350.

## What it does not change

- `_build_fallback_config`'s narrowing when a test target is derivable.
- The module-config branch.
- `force_workspace` and `merge_verify_workspace`.
