# Pre-merge gate for retuning an operational config knob

Decided by task 3886, which was spun out of task 3866.

## The question

`orchestrator/tests/conftest.py::_isolate_orch_config` is an autouse fixture.
It binds every orchestrator unit test to this repo's live
`dark-factory-orchestrator.yaml`. So when a test pins a literal derived from a
knob, retuning that knob turns the test red. Commit `094d634465` did exactly
this: it raised `lock_depth` from 4 to 12 and broke three tests on main.

The task offered three options:

1. Add a pre-merge gate for config retunes.
2. Narrow the binding, so that unit tests read the package defaults.
3. Accept the exposure.

## The decision

- **Option 1 is declined as framed.** A retune that enters the merge lane is
  already gated.
- **Option 2 is rejected.**
- **Option 3 is adopted for the one ungated path:** a human commit made
  directly to main.
- **The standing remedy is a fourth option:** write tests in the
  knob-agnostic form.

## Why

### The merge lane already gates a config retune

`094d634465` never entered the merge lane. It has a single parent: it was
committed directly to main. Two gates apply to a diff that does go through
the lane:

- `merge_verify_breadth: "full"` makes merge-role verify run every registered
  module's full suite, whatever the diff touches. See
  `orchestrator/src/orchestrator/verify_plan.py::derive_verify_plan`.
- `git.merge_config_only_full_gate_globs` names `dark-factory-orchestrator.yaml`.
  It forces the full gate for a config-only diff on the merge-role path where
  `is_merge_verify` is false. INV-1's nothing-to-run escalation does not reach
  that path. See
  `orchestrator/src/orchestrator/verify.py::_merge_config_only_diff_forces_full_gate`.
  This entry was the task's only config change. It is red tier: it is not in
  `RELOADABLE_FIELDS`, so it takes effect at the next orchestrator restart.

### Option 2 would turn a loud failure into a silent one

The binding is deliberate: it makes tests see the values the factory actually
runs at. The heal commit `c54d7cf81c` records why. The tests that
`094d634465` broke failed loudly, because their self-validating preconditions
caught fixture data that had gone stale. Under option 2 they would have
passed, at a depth the factory does not run at.

A test that genuinely asserts a package default can opt in to
`conftest.py::code_default_config`. This task used that fixture itself.
Wiring the glob above turned
`orchestrator/tests/test_verify.py::TestMergeConfigOnlyDiffForcesFullGate::test_empty_globs_default_config_returns_false`
red. That test had been passing only because this repo happened not to set
the key.

### Option 3: the remaining exposure is not specific to config files

The ungated path is a human commit made directly to main. `hooks/project-checks`
runs no pytest for any kind of file. A pytest gate for yaml alone would cover
the least common kind of commit, leave `.py` commits open, and put a
multi-minute suite in every human commit. `run_main_tip_sweep` is on by
default and runs every 30 minutes, which bounds how long a red tip goes
unnoticed.

### The standing remedy: knob-agnostic tests

A test that derives its fixture from the live value cannot be broken by a
retune. `orchestrator/tests/_workflow_helpers.py::same_module_siblings` is
the example, from `c54d7cf81c`. When a retune breaks a test:

- If the test merely depends on the value, fix it to read the value live.
- If the test truly asserts a package default, add `code_default_config`.
- Never pin the new value.

## Guards

- `tests/scripts/test_config_retune_gate_decision.py` asserts three premises:
  - merge breadth is full;
  - the glob covers this repo's config;
  - a config-file layer wins over the package defaults.

  It also checks that every module config is reachable at the live
  `lock_depth`, so a downward retune that would half-apply one is caught. The
  file sits in `tests/scripts/` so that the directory's own module config
  runs it under full verify. A repo-root file cannot have a module config of
  its own.
- `orchestrator/tests/test_operational_config_binding.py` asserts the binding
  itself.

Neither guard catches a stale fixture literal in some other test. That case
is caught later: by the merge gate, or by the main-tip sweep.
