# Per-test pytest timeout: the measurement, and the value derived from it

Task 5442, measured 2026-09-17 on branch `task/5442` at base `832d6faf16`.

This file is the PROVENANCE every `[tool.pytest.ini_options].timeout` in this
repo cites. It exists so the next agent never has to re-derive or re-guess the
number. The canonical rationale for *why* the cap exists at all — what it
catches, and why `timeout_method` and `--max-worker-restart=0` are deliberately
left alone — is `shared/pyproject.toml`'s `timeout` block and is not restated
here.

## What this replaces

Until this measurement, every config carrying the setting said the same thing:
"300 IS INTERIM AND JUDGEMENT-PICKED, NOT MEASURED. Nobody has yet measured the
real test-duration distribution under load; a follow-up task owns that
measurement and owns replacing this with a data-driven value." This is that
measurement. Three configs — the repo root, `cockpit` and `sampler` — carried no
`timeout` at all, so their runs had no per-test wall-clock cap whatsoever.

## The derivation rule

    derived = max(_ABSOLUTE_FLOOR_SECONDS, round_up_to_multiple_of_60(worst_unmarked_wall_clock * 8))

Adopted wholesale from what this repo already justifies, rather than invented:

- The **8x headroom factor** is
  `orchestrator/tests/_orch_helpers.py::UNDER_LOAD_HEADROOM_FACTOR`.
  8x rather than 2x because the xdist worker deaths the cap exists to avoid were
  observed at loadavg 250-423 — one inflation step PAST the load at which
  measurements can be taken — so the value has to clear a figure nobody has
  managed to measure directly.
- The **floor** is
  `orchestrator/tests/test_whole_tree_scan_timeout_guard.py::_ABSOLUTE_FLOOR_SECONDS`,
  which is itself anchored to an independent measurement of the whole-tree-scan
  family. It was 300 when this sweep ran (`_MEASURED_UNDER_LOAD_WORST_CASE =
  30.75` at loadavg 120-176, x8 = 246, rounded up); task 5572 re-anchored it to
  **420** (51.87 x 8 = 414.96, rounded up — see the addendum at the end).
  Re-evaluating the rule with the 420 floor leaves the ini value at 540:
  max(420, 540) = 540. It is a never-narrow term: taking the max of two measured
  anchors can only validate or raise the number, never discard the older
  evidence and re-open the CPU-starvation false-red that the 60 -> 300 raise
  closed.
- **Rounding to a multiple of 60** keeps the value readable as whole minutes,
  matching how the existing 60 and 300 read.

**Marked tests are excluded from the governing set by design.** A
`@pytest.mark.timeout(N)` OVERRIDES the ini default in both directions rather
than raising a floor under it (`orchestrator/tests/_orch_helpers.py`'s
`VERIFY_CLI_PER_TEST_TIMEOUT` block is the single home of that reasoning), so
the ini default only ever has to cover UNMARKED tests. Every table below reports
the worst marked and the worst unmarked separately, and only the unmarked column
feeds the arithmetic.

## Host shape and load

32 cores. Real fleet load throughout — this is not a synthetic soak and not a
quiet host, which matters because the defect the cap exists to survive is CPU
STARVATION, not a hung test. Loadavg is recorded per run in the table below;
across the sweep the 1-minute figure ranged from 67 to 365, and the orchestrator
run in particular landed inside the **loadavg 250-423 band in which the xdist
worker deaths that justify the 8x factor were originally observed**.

## Method

Two inifile-resolution modes, because pytest reads exactly ONE inifile — the
rootdir's — and never merges across `pyproject.toml` files.

Per-member (each member's own `pyproject.toml` is the inifile, so its own
`addopts`/xdist apply):

```
cd <member> && env -u VIRTUAL_ENV uv run --project . pytest tests -q --durations=0 --timeout=0 -p no:cacheprovider
```

Root-bound (the ROOT `pyproject.toml` is the inifile — the mode with the
coverage gap), run from the repo root:

```
env -u VIRTUAL_ENV uv run --project shared pytest <targets> -q --durations=0 --timeout=0 -p no:cacheprovider
```

`--timeout=0` disables the cap so true durations are observed rather than
truncated at 300. `-p no:cacheprovider` keeps the runs from writing a
`.pytest_cache` into the tree. `env -u VIRTUAL_ENV` matters because an agent Bash
session inherits the MAIN checkout's `VIRTUAL_ENV` and would otherwise resolve
main's code rather than this worktree's. Runs were **sequential**, never
concurrent: parallel runs would be self-inflicted load rather than the real fleet
load this measurement is about.

`-n auto` resolved to **32 workers** in these runs, not the fleet's 8: the worker
count normally comes from `PYTEST_XDIST_AUTO_NUM_WORKERS` in
`dark-factory-orchestrator.yaml`, which a bare agent shell does not set. That is
the correct shape to measure — `dashboard/pyproject.toml` already records that the
ini default "governs bare local/agent invocations", which is exactly the
invocation class that resolves `auto` to the core count.

### Reading the distribution tables

pytest's `--durations=0` HIDES every entry below 0.005s and reports only their
count. The percentiles below fold those hidden entries back in at the bottom of
the distribution rather than computing over the visible tail alone, which would
bias every quantile upward — in the `shared` run, for instance, 14873 of the
16509 phase entries were hidden and only 1636 printed. The practical consequence is that p50 and most p90
figures are uninformative (the majority of entries really are sub-5ms), so the
tables report p99 upward and the derivation uses the max.

## Per-suite distributions

All figures in seconds. `worst unmarked` is the largest single phase duration of
a test carrying no `@pytest.mark.timeout`; `worst marked` is the largest carried
by one that does, shown only to make the exclusion auditable. `loadavg` is the
1-minute figure immediately before and after each run.

### Per-member mode

| suite | execution | result | loadavg | call p99 / p99.9 / max | setup max | teardown max | worst unmarked | worst marked |
|---|---|---|---|---|---|---|---|---|
| `sampler` | serial | 52 passed, 3.13s | 73.8 → 74.5 | 0.40 / 0.66 / **0.66** | 0.01 | – | 0.66 call | none |
| `cockpit` | `-n auto` (32) | 447 passed, 34.52s | 74.5 → 83.3 | 2.40 / 3.45 / **4.85** | 0.15 | 0.33 | 4.85 call | 3.45 call |
| `shared` | serial | 5495 passed, 236.94s | 83.3 → 92.2 | 0.31 / 1.98 / **7.12** | **14.01** | 0.20 | 14.01 setup | 4.13 call |
| `dashboard` | `-n auto` (32) | 2471 passed, 112.11s | 92.2 → 85.9 | 0.88 / 2.76 / **6.03** | 0.55 | 4.63 | 6.03 call | none in top 80 |
| `escalation` | `-n auto` (32) | 1841 passed, 31.73s | 85.9 → 75.3 | 0.46 / 1.24 / **3.12** | 0.88 | 0.06 | 1.59 call | 3.12 call |
| `fused-memory` | `-n auto` (32) | 21038 passed, 427.12s | 75.3 → 89.5 | 0.57 / 11.62 / **42.63** | 6.89 | 0.29 | **42.63 call** | 18.97 call |
| `orchestrator` | `-n auto` (32) | 21428 passed / 4 failed, 2668.71s | 89.5 → 67.7 (peak 365 mid-run) | 2.61 / 12.08 / **45.36** | **51.87** | 10.36 | 27.13 call | 51.87 setup |

### Root-bound mode

| targets | execution | result | loadavg | call p99 / p99.9 / max | setup max | teardown max | worst unmarked |
|---|---|---|---|---|---|---|---|
| `tests/scripts/` + `scripts/tests/` | serial | 5498 passed / 1 failed, 795.90s | 67.7 → 75.8 | 1.81 / 7.97 / **20.17** | **25.27** | 0.46 | 25.27 setup |
| `fused-memory/tests/test_check_bare_magicmock_config.py` + `orchestrator/tests/test_hard_v2_fixture_pool.py` | serial | 212 passed, 204.42s | 80.9 → 100.6 | – | – | – | **61.52 call** |

The second row is a targeted re-measurement of the two worst unmarked tests the
per-member sweep found, run under the ROOT inifile. It is the row that decides
the value — see below.

## The slowest unmarked test per suite

| suite | test | worst phase |
|---|---|---|
| `fused-memory` | `tests/test_check_bare_magicmock_config.py::TestAllScannedTestDirsClean::test_every_scanned_tests_directory_exits_zero` | **61.52 call** (root-bound); 41.76 (per-member) |
| `fused-memory` | `tests/test_check_bare_magicmock_config.py::TestWallClockDeadlineBaselineIntegrity::test_no_scanned_file_outside_the_baseline_carries_a_violation` | 54.52 call (root-bound); 42.63 (per-member) |
| `orchestrator` | `tests/test_hard_v2_fixture_pool.py::TestMintedPool::test_reference_unavailable_only_when_no_landing_merge_exists` | 45.48 call (root-bound); 27.13 (per-member) |
| root dirs | `tests/scripts/test_pyright_version_pin.py::test_npx_in_a_fresh_worktree_does_not_resolve_the_pin_on_its_own` | 25.27 setup |
| `shared` | `tests/test_config_dir_archival_gate.py::TestEverySiteIsAudited::test_every_construction_site_is_audited` | 14.01 setup |
| `dashboard` | `tests/test_escalation_analytics.py::TestBuildEscalationAnalyticsPerf::test_cold_10k_archive_cpu_budget` | 6.03 call |
| `cockpit` | `tests/test_priority.py::TestDroppedBelowOpen::test_dropped_scores_below_open` | 4.85 call |
| `escalation` | `tests/test_watcher_rearm.py::test_explicit_level_still_filters` | 1.59 call |
| `sampler` | `tests/test_load_store.py::TestTrailingWindow::test_caps_at_window_minus_one_prior_rows` | 0.66 call |

The two governing tests are both whole-tree scans that shell out to
`fused-memory/scripts/check_bare_magicmock_config.py` over every scanned
`tests/` directory in the repo. They are structurally the same family that task
4215 marked with `WHOLE_TREE_SCAN_TEST_TIMEOUT` inside `orchestrator/tests/`, but
in `fused-memory/tests/` they carry no marker at all (verified: that file
contains zero `pytest.mark.timeout` occurrences), so they sit squarely on the ini
default. That is what makes them governing rather than excludable.

### Why the scan-family carve-out was not applied to these two

This is the sharpest objection to the value below, so it is answered here rather
than left for a reader to find. `orchestrator/tests/test_whole_tree_scan_timeout_guard.py`'s
module docstring carves the family out of exactly this arithmetic: widening the
global `timeout` is "NOT ... the remedy for THIS hazard ... doing it for that
reason would blunt the hang-catching ceiling for the other ~16000 tests to buy
headroom only ~13 modules need." Two members of that same family are what drive
the 540 derived below. The tension is real and is not dissolved by observing
that the global raise answers a different hazard.

It is nonetheless derived from the tree AS MEASURED, for three reasons.

**The ini default's job is defined over the tests that do not opt out.** What the
default must be is a question about the repository as it stands, not about a
repository in which an edit that was never made had been made. A number derived
from a hypothetically-marked tree would be justified by nothing a later reader
could reproduce from the code.

**Marking them is not this task's to do.** `pytestmark = pytest.mark.timeout(...)`
in `fused-memory/tests/test_check_bare_magicmock_config.py` is an edit to a test
module, outside the scope that governs this work ("every module
`pyproject.toml`"). This task could not both mark them and derive from the
marked tree in one commit, and deriving from a tree it had not actually produced
would be worse than deriving from the one in front of it.

**What changes if they are marked is arithmetic, not another measurement.** With
those two excluded, the worst unmarked figure in the corpus becomes 45.48s
(`orchestrator/tests/test_hard_v2_fixture_pool.py`, root-bound) — which is NOT a
tree scan: it performs no `glob`/`rglob` at all, so the next-worst figure is a
genuinely different kind of test rather than the same family one rung down.
45.48 x 8 = 363.84, rounded up to a whole minute and floored at 300, gives
**420** rather than 540. That is the better end state — ~16000 tests under a
tighter hang-catching ceiling, and the ~13+2 modules that need the headroom
carrying it explicitly — and reaching it needs only the marker edit plus a
re-derivation, no re-measurement. Filed as a follow-up, which owns both halves;
until it lands the value stands at 540 and the corpus above is what justifies it.

### What was excluded, and why

Every test named in the `worst marked` column above carries
`@pytest.mark.timeout`, almost all of them
`pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)` at module level.
Verified by AST rather than by grep — module-level `pytestmark`, class
decorators and function decorators each counted. The two largest figures in the
entire corpus are both marked and both excluded:
`orchestrator/tests/test_merge_lane_ratchet.py` (51.87s setup) and
`orchestrator/tests/test_steward_scaffolding_guards.py` (45.36s call). A marker
OVERRIDES the ini default rather than raising a floor under it, so the default
never governs them.

## THE FINDING THAT DECIDES THE VALUE: root-bound is SLOWER, not faster

The intuitive expectation — that a serial run contends with itself less than a
32-worker one and so yields smaller per-test durations — is FALSE here, measured
rather than assumed. The same three unmarked tests, re-run under the ROOT inifile
serially at loadavg 80-100, came in **1.3x to 1.7x SLOWER** than under their own
member's `-n auto`:

| test | per-member (`-n auto`, 32) | root-bound (serial) |
|---|---|---|
| `test_every_scanned_tests_directory_exits_zero` | 41.76 | **61.52** |
| `test_no_scanned_file_outside_the_baseline_carries_a_violation` | 42.63 | **54.52** |
| `test_reference_unavailable_only_when_no_landing_merge_exists` | 27.13 | **45.48** |

This matters twice over. It means the mode that until this task had NO cap at all
is also the mode that produces the LARGEST per-test wall clock in the corpus — so
measuring only per-member mode would have under-derived the value. And it means
the root config could not simply have copied a member's number.

### Run-to-run spread

Two further samples of the governing file, taken minutes apart on the same tree
at loadavg 100-114, resolved their own member inifile serially and gave
39.00 / 43.13 for `test_every_scanned_tests_directory_exits_zero` and
37.38 / 45.38 for `test_no_scanned_file_outside_the_baseline_carries_a_violation`
— a ~1.6x band on byte-identical input, matching the 2.91x band
`tests/scripts/orchestrator.yaml` records for its own suite. The derivation
therefore takes **the worst run, never the mean and never the freshest**, which is
this repo's existing convention for exactly this reason.

## The arithmetic

    worst unmarked wall clock, any suite, any phase, any mode  =  61.52s
    x UNDER_LOAD_HEADROOM_FACTOR (8)                           = 492.16s
    round up to a multiple of 60                               = 540s
    max(300, 540)                                              = 540s
      (300 was the floor on 2026-09-17; with task 5572's 420, max(420, 540) = 540s)

## THE RESULT: 540 seconds. The measurement RAISES the value; it does not validate it.

300 was too small. It is only 4.9x the worst unmarked wall clock this sweep
measured, against the 8x the repo's own worker-death evidence demands, and a
single honest whole-tree scan run under the root inifile already consumes a fifth
of it. **540** is the first value for this setting that is derived rather than
picked, and it is the same value in all eight configs so that the cap a test runs
under no longer depends on which directory pytest happened to resolve its rootdir
from.

The cost of the raise is unchanged in kind from the 60 -> 300 raise and is stated
plainly: a genuinely hung test now takes 9 minutes to surface rather than 5.
That is the price of not false-redding a healthy test starved of CPU, and the
`@pytest.mark.timeout(N)` escape hatch remains available for any test that wants
to surface faster.

## Residue, recorded rather than left silent

1. **RESOLVED by task 5572** — the anchor is now 51.87s and the floor 420; see
   "Re-anchoring addendum (task 5572, 2026-10-08)" below. The original record
   follows.
   **The whole-tree-scan family's measurement anchor is stale, though its value
   is no longer binding.** `_ABSOLUTE_FLOOR_SECONDS = 300` is justified by
   `_MEASURED_UNDER_LOAD_WORST_CASE = 30.75` (task 4215, loadavg 120-176) times
   the same 8x factor. This sweep measured a MARKED member of that family at
   **51.87s** setup (`orchestrator/tests/test_merge_lane_ratchet.py`) and another
   at 45.36s call — 1.7x the figure the floor rests on, which would derive 420
   rather than 300. The family ceiling itself is NOT left short: the never-narrow
   rule forbids `WHOLE_TREE_SCAN_TEST_TIMEOUT` sitting below the ini default, so
   it moves to 540 and clears 414.96 comfortably. What remains stale is the
   ANCHOR under the floor, and re-anchoring it is deliberately not done here —
   `_ABSOLUTE_FLOOR_SECONDS` exists precisely to be independent of the ini
   default, so a task that measured the ini default is the wrong place to move
   it. Filed as a follow-up.

2. **The CLI `--timeout=300` on every verify `test_command` is now TIGHTER than
   the ini default.** A CLI `--timeout` overrides the ini, and `--timeout=300`
   appears on every per-module `*/orchestrator.yaml`, on
   `tests/scripts/orchestrator.yaml`, on `scripts/orchestrator.yaml` and in
   `dark-factory-orchestrator.yaml`'s fan-out, pinned by
   `tests/scripts/test_fallback_verify_config.py::test_per_module_merge_verify_raises_per_test_timeout`.
   So the merge gate now enforces a cap this measurement says is too small, while
   bare local/agent invocations get 540. That inversion is real and is named here
   rather than discovered later; the yaml knob is outside this task's scope ("every
   module pyproject.toml") and is filed as a follow-up.

3. **The two governing tests are an unmarked branch of the whole-tree-scan
   family.** Marking them the way `orchestrator/tests/` marks its own members
   would re-derive this value as 420 rather than 540, putting ~16000 tests under
   a tighter hang-catching ceiling. The full argument for why this task derived
   from the tree as measured instead is above, under "Why the scan-family
   carve-out was not applied to these two", and is not restated here. Filed as a
   follow-up owning both halves: the marker edit in
   `fused-memory/tests/test_check_bare_magicmock_config.py`, and the
   re-derivation that follows from it.

4. **Root-bound mode was measured over the repo-root-owned test directories in
   full, and over the member trees only by targeted re-measurement of the worst
   unmarked tests the per-member sweep found.** A full serial root-bound run of
   every member tree was not attempted: `orchestrator` alone took 44 minutes at
   32 workers. The targeted form is what the value rests on, and it is the
   conservative direction — the measured member-to-root inflation factor is
   1.3-1.7x, so applying the largest of it to the next-worst unmarked figure in
   the corpus (25.27s) yields ~43s, still comfortably below the 61.52s the value
   is derived from.

## Re-anchoring addendum (task 5572, 2026-10-08)

Task 5572 re-derived two marked-test ceilings in `orchestrator/tests/` by the
rule above: the whole-tree-scan family's floor
(`test_whole_tree_scan_timeout_guard.py::_ABSOLUTE_FLOOR_SECONDS`) and
`test_merge_queue_concurrent_verify.py::HEAVY_BARRIER_TEST_TIMEOUT`.

### Method

Base `057aeed7d0`, cwd `orchestrator/`, `-n auto` = 32 workers, started
2026-10-08T04:03:50Z and finished 04:06:04Z:

```
env -u VIRTUAL_ENV uv run --project . pytest \
  tests/test_archive_sole_locator_gate.py tests/test_cited_test_class_drift.py \
  tests/test_eval_boundary_suite.py tests/test_event_loop_antipattern_guard.py \
  tests/test_info_l0_mechanical_roles.py tests/test_killpg_frozen_pgid_guard.py \
  tests/test_lane_lifecycle_gitops.py tests/test_local_repo_seeder_guard.py \
  tests/test_lock_release_single_writer_guard.py tests/test_marker_registration_drift.py \
  tests/test_mcp_post_transport.py tests/test_merge_lane_alias_names.py \
  tests/test_merge_lane_ratchet.py tests/test_merge_worker_retired.py \
  tests/test_prune_chokepoint_guard.py tests/test_raw_semaphore_access_guard.py \
  tests/test_scheduler_hold_history.py tests/test_steward_scaffolding_guards.py \
  tests/test_timeout_marker_inversion_guard.py tests/test_whole_tree_scan_timeout_guard.py \
  tests/test_workflow_factory.py \
  tests/test_merge_queue_concurrent_verify.py tests/test_merge_speculation.py \
  tests/test_concurrent_verify_boundary.py tests/test_multihost_verify_integration.py \
  -q --durations=0 --timeout=0 -p no:cacheprovider
```

The first 21 modules are the whole-tree-scan family; the last four carry the
`HEAVY_BARRIER_TEST_TIMEOUT` and capstone marks.

Result: **980 passed in 127.82s**.

Load, 1-minute: 455.06 a few minutes before launch, 152.34 at start, 101.37 at
end. 5-minute: 277.59 at start, 214.06 at end. This was a TARGETED run of two
families, not the full suite, so its contention is lower than task 5442's
full-suite run above.

### Per-family worst

| Family | Test | Phase / per-test total |
|---|---|---|
| whole-tree scan | `test_merge_lane_ratchet.py::TestLanePatchTargets::test_real_tree_union_anchor` | 37.61 setup |
| whole-tree scan | `test_merge_lane_alias_names.py::test_no_tracked_file_reaches_a_missing_name_through_an_alias` | 34.62 call |
| whole-tree scan | `test_steward_scaffolding_guards.py::TestAbsoluteTmpProjectRootLiteralsAreCensused::test_every_absolute_tmp_project_root_literal_is_adjudicated` | 34.07 call |
| heavy barrier | `test_merge_speculation.py::TestSpecLaneAbortReleasesLane::test_operator_halt_releases_spec_lane` | 14.59 total / 14.15 call |
| heavy barrier | `test_merge_speculation.py::TestSpecLaneAbortReleasesLane::test_waiter_walking_away_releases_spec_lane` | 13.05 total |
| heavy barrier (capstone) | `test_multihost_verify_integration.py::TestUnreachableHostCapstone::test_a_cancel_against_a_down_host_parks_the_slot_and_reprobe_unparks_it` | 12.28 total |
| heavy barrier | `test_merge_queue_concurrent_verify.py::TestRunnerUnavailableHeadCascade::test_ru_head_cascade_reruns_speculative_downstream` | 5.45 total |

The heavy-barrier anchor is a per-test TOTAL (setup + call + teardown), because
that is what a pytest-timeout mark bounds. The whole-tree anchor stays a phase
figure, because a phase is all task 5442 recorded; it is a lower bound on that
test's total.

### Whole-tree-scan floor

    anchor = max(51.87 from task 5442's full suite, 37.61 today) = 51.87s
             (the worst run, never the freshest)
    x UNDER_LOAD_HEADROOM_FACTOR (8)                          = 414.96s
    round up to a multiple of 60                              = 420s  -> _ABSOLUTE_FLOOR_SECONDS

`WHOLE_TREE_SCAN_TEST_TIMEOUT` stays **540**: it clears the 420 floor, and the
540 ini default binds through the never-narrow rule.

### HEAVY_BARRIER_TEST_TIMEOUT

    measured worst, per-test total                      = 14.59s
    x UNDER_LOAD_HEADROOM_FACTOR (8)                    = 116.72s -> rounds up to 120s
    computed wait budget, heaviest marked class         = 255s (TestCascadeErrorContainment)
    computed wait budget, late-arrival classes          = 245s
    never-narrow ini default                            = 540s
    result                                              = 540s

The ini default is the binding term. The wait budgets are those
`TestTimeoutMarkCoverage` recomputes from source, and that guard still checks
every class against the mark. The constant was `5 * MERGE_RESULT_TIMEOUT + 75`
(300) and is now a literal pinned by
`test_merge_queue_concurrent_verify.py::TestHeavyBarrierTimeoutConstant`.

The conclusion does not depend on today's lighter load. Task 5442's full suite
put every MARKED orchestrator test at or below 51.87s on 2026-09-17. Used as
the anchor, that looser bound derives only 420, still below 540.

`HOST_CAPSTONE_TEST_TIMEOUT` (`test_multihost_verify_integration.py`) has a
computed budget of 360s. It stays at **600** as a literal, rather than
following its old `2 x HEAVY_BARRIER_TEST_TIMEOUT` to 1080.

Task 5570 may raise verify's CLI `--timeout` from 300 to 540. Either way, 540
is outside the inversion band
`DELIBERATE_TIGHT_BOUND_CEILING < N < VERIFY_CLI_PER_TEST_TIMEOUT`: it is above
300 today, and at 540 it would sit on the band's open upper edge.
