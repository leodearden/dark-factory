# fused-memory RCA: a CWD-relative config path manufactured a phantom main-red — 2026-09-13

**Context.** Task 5444 asked which gate let a "born-red" test through: `fused-memory/tests/test_referent_repair.py::TestTheStormGateProjectRoot::test_the_taskmaster_project_root_is_never_used_as_a_fallback`,
reported as failing deterministically on main since the commit that introduced it, 18 days
earlier. Every measurement below was taken first-hand in worktree `.worktrees/5444`; the SHA
each was taken at is named, because several were taken while this branch was being built and
the numbers move.

## Verdict: there is no gate hole of the shape the task assumed

The test was **green under both registered verify commands from birth to today**. No gate ever
saw red, because there was never a red to see. The 18 days is fully explained by that.

Measured at base `bd4574c8d1` (main tip at the time), same commit, same test, two CWDs:

```
uv run --directory fused-memory pytest tests/test_referent_repair.py \
    -k taskmaster_project_root_is_never_used                        -> 1 passed
uv run --project   fused-memory pytest fused-memory/tests/test_referent_repair.py \
    -k taskmaster_project_root_is_never_used                        -> 1 failed
```

The failure is `AttributeError: 'NoneType' object has no attribute 'project_root'`, byte-identical
to the report that generated the task. Both registered commands `cd` into `fused-memory/` — the
root full-breadth `test_command` in `dark-factory-orchestrator.yaml`, whose fused-memory leg is
`cd ../fused-memory && uv run pytest tests/ --timeout=300`, and the module command in
`fused-memory/orchestrator.yaml`, `uv run --directory fused-memory pytest tests/ --tb=short -q
--timeout=300` — so both take the first line, and a human at the repo root takes
the second.

Corroboration from `data/orchestrator/runs.db` (read-only queries against the live DB):

- Task 3672, which introduced the test: **6 × `workflow_verify` `passed:true`**, then
  **`merge_verify` `passed:true`** on merge SHA `6dba49cf00a13a8a67d65268126641e2cf86b4c9`,
  `duration_ms: 1945285` — a 1,945 s run, i.e. the full suite, not a scoped subset.
- **270 `merge_verify` runs recorded `passed:true` between 2026-08-25 and 2026-09-13.**
  `dark-factory-orchestrator.yaml` sets `merge_verify_breadth: "full"`, so a merge verify runs
  every module's suite; the 1,945 s above is that breadth showing up in the clock.
- Across the entire `events` table, **not one record — passing or failing — mentions this test's
  name**, and `flake_occurrence` holds **zero rows** for it. The gates ran, and they reported
  correctly.

## Mechanism: `BaseSettings` + a relative path + a config section older than the test

`fused_memory.config.schema::FusedMemoryConfig` is a pydantic-settings `BaseSettings`, not a plain
`BaseModel`. Its `settings_customise_sources` reads `CONFIG_PATH` with a default of the **relative**
`config/config.yaml`, and `fused_memory.config.schema::YamlSettingsSource.__call__` returns `{}` —
silently — when that path does not exist. So every field a caller does not pass explicitly is
filled from a file resolved against the **process CWD**.

With CWD = `fused-memory/`, `config/config.yaml` resolves to the tracked
`fused-memory/config/config.yaml`, whose `taskmaster:` section makes `config.taskmaster` a real
`TaskmasterConfig`, so the test's leaf assignment succeeds. From anywhere else the YAML layer
vanishes, `taskmaster` falls to its declared default of `None`, and the assignment raises.

The inputs are all older than the test and none changed in the window:

| input | commit | date |
|---|---|---|
| the `taskmaster:` section in `fused-memory/config/config.yaml` | `2e0e8b49f8` (sole commit ever to touch it) | 2026-03-14 |
| the test | `57f2db345b` | 2026-08-25 |

Five months apart. The outcome is therefore constant across the whole window: a function of the
pytest CWD, and of nothing else that moved.

**The environment is the other half of the same leak, and the file path does not close it.**
`model_config` sets `env_prefix=''` with `env_nested_delimiter='__'` and `case_sensitive=False`, so
a *bare* variable named after any of the model's nineteen top-level fields is an unprefixed
override that outranks the YAML. Measured at `3e2676b2a2`, with the variable inherited from the
launching shell: both `TASKMASTER='{"project_root": "/pwned-by-env"}'` and
`TASKMASTER__PROJECT_ROOT=/pwned-by-env` resolve `config.taskmaster.project_root` to
`/pwned-by-env`, and both reproduce with `CONFIG_PATH` pointing at a file that does not exist.

## Cost

A suite whose result depends on the ambient environment of whoever runs it is not merely untidy;
it produced a *false* signal that survived expert scrutiny twice.

- A phantom main-red: the suite was green for the gate and red for a human on the same commit.
- Commit `a484d16588` (2026-09-12, branch `task/4448`, unmerged) — `fix(main-red): plant the
  taskmaster trap as a section, not a leaf` — a fix for a test that was never main-red.
- Task 5444 itself, filed on a false premise, asking which gate had failed when none had.
- **Two independent expert analyses reached the same wrong conclusion**, one of them a read-only
  agent re-deriving it here from the same evidence before pytest settled it. Both read
  `Field(default=None)` and stopped, without reading `settings_customise_sources` two hundred lines
  below it. The trap is not that the answer is hard; it is that the wrong answer is what a careful
  reader gets.

## The generalisation: the fix already existed, one directory away

This exact class was RCA'd for the orchestrator subproject in
`plans/eval-metric-collector-orch-config-leak-rca-2026-07-22.md` (task 2957) and hardened under
task 2719 with two fixtures in `orchestrator/tests/conftest.py`: an autouse
`_isolate_orch_config` pinning `ORCH_CONFIG_PATH` to the canonical **absolute** path — its
docstring states outright that "the absolute path is also CWD-independent, so the config no longer
depends on running from `orchestrator/`" — and an opt-in `code_default_config` re-pointing at a
guaranteed-absent file.

fused-memory had only `preserve_config_path`, which saved and restored `CONFIG_PATH` around each
test but never *pinned* it. That guards intra-suite pollution and is blind to what the suite
inherits. The hardening was applied subproject-locally and nothing propagated it.

**That is the recurring hole, and it is not about config at all:** a test-isolation hardening
lands in one subproject's `conftest.py`, and no mechanism carries it to a sibling with the same
defect. The sibling keeps a weaker fixture that *looks* like it covers the same ground. Two
subprojects, one `BaseSettings`-plus-relative-path pattern, one RCA already written, and the
second occurrence still took an 18-day investigation to reach.

## The fix, and what sized it

Mirror of the orchestrator's shape into `fused-memory/tests/conftest.py`: autouse
`_isolate_fm_config` pins `CONFIG_PATH` to the canonical config by absolute path (derived from
`__file__`, never from `Path.cwd()`, which would reintroduce the bug) and scrubs the inherited
env surface; opt-in `code_default_config` supplies pure schema defaults. `preserve_config_path`
is absorbed rather than kept — `monkeypatch.setenv` already restores on teardown, so keeping both
would leave two autouse fixtures owning one variable with no defined ordering. The scrubbed names
are derived from `FusedMemoryConfig.model_fields` at import, so a field added later is covered
without an edit; `PATH`, `OPENAI_API_KEY` and `MEM0_API_KEY` do not match and are untouched.

Pinning the **canonical** file rather than an absent one was chosen so collateral is zero by
construction: it reproduces the CWD=`fused-memory/` semantics every currently-green test was
written against.

**Blast radius, re-measured at `eb04f1d1c8`** (the whole suite with the YAML layer removed, less
the two assertions in this branch that pin the config file's *presence* and so cannot survive its
removal): `1 failed, 19783 passed, 3 skipped` in 265.69s. The single failure is the taskmaster
test. **Exactly one test in ~19.8k reads a value only the ambient file supplies** — which is
precisely why a CWD-dependent config could sit under this suite unnoticed, and why the absent-file
variant was rejected: it would have red-walled main on that one test, the one this task was
forbidden to re-fix and which `a484d16588` already owns on an unmerged branch.

Once `a484d16588` lands, that last dependency is gone and pinning to a guaranteed-absent file
becomes free. **That is the stronger end state** — it would isolate the suite from the tracked
config as well as from the CWD — and it should be revisited then, not now.

Full suite at `c7df77ee67` under the registered module command: `19788 passed, 3 skipped` in
416.48s, exit 0. With `TASKMASTER`, `TASKMASTER__PROJECT_ROOT` and `SERVER__PORT` exported into
pytest's environment, the config suites still pass — 346 of them — where before the fix the
inherited value won.

## A second hole, confirmed and deliberately not fixed here

Task 5444's third line conjectured that a correctly-classified pre-existing red main could sit
unfixed forever. **That conjecture is true as a general property.** It is also *not what happened
here* — the gates all passed, so no pre-existing-main classification ever ran and no escalation
was ever filed for this test. Fixing it would not have changed the 18 days by a day.

The dead end, as read in the code: a red main classified `test_failure` terminates at
`suggested_action='await_preexisting_main_hotfix'` (`orchestrator/src/orchestrator/workflow.py`
raises it at three sites; `orchestrator/src/orchestrator/merge_queue.py::_file_main_health_escalation`
carries it as the default), and no code path produces that hotfix.
`merge_queue.py::AUTO_HEAL_MECHANICAL_CATEGORIES` is `frozenset({'compile_error'})`, so
`test_failure` is excluded by design and `_spawn_main_health_fix_task` is unreachable for it. The
escalation dedupes with an unbounded window, so every later task folds into one parent. And
`harness.py::Harness._run_main_tip_sweep` records the swept SHA *before* it branches on the
verdict, then returns early on a non-advancing SHA — which is why `critical_gate.py` says in so
many words that "a streak can never accumulate on a stuck red main".

It is left for a follow-up task rather than bolted on here: it lands in the merge lane — the
auto-heal eligibility gate and the `_mark_blocked` call sites — which is the highest-blast-radius
subsystem in the repo and deserves its own plan and review, whereas this change's whole virtue is
that it alters no product behaviour. Bundling them would also make one diff span two unrelated
subprojects.
