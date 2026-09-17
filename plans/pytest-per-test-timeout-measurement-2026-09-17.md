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

    derived = max(300, round_up_to_multiple_of_60(worst_unmarked_wall_clock * 8))

Adopted wholesale from what this repo already justifies, rather than invented:

- The **8x headroom factor** is
  `orchestrator/tests/test_whole_tree_scan_timeout_guard.py::_REQUIRED_HEADROOM_FACTOR`.
  8x rather than 2x because the xdist worker deaths the cap exists to avoid were
  observed at loadavg 250-423 — one inflation step PAST the load at which
  measurements can be taken — so the value has to clear a figure nobody has
  managed to measure directly.
- The **300 floor** is that same file's `_ABSOLUTE_FLOOR_SECONDS`, which is
  itself anchored to an independent measurement
  (`_MEASURED_UNDER_LOAD_WORST_CASE = 30.75` at loadavg 120-176, x8 = 246,
  rounded up). It is a never-narrow term: taking the max of two measured anchors
  can only validate or raise the number, never discard the older evidence and
  re-open the CPU-starvation false-red that the 60 -> 300 raise closed.
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
    x _REQUIRED_HEADROOM_FACTOR (8)                            = 492.16s
    round up to a multiple of 60                               = 540s
    max(300, 540)                                              = 540s

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

1. **The whole-tree-scan family's own ceiling is now under-derived by its own
   rule.** `WHOLE_TREE_SCAN_TEST_TIMEOUT` is anchored to
   `_MEASURED_UNDER_LOAD_WORST_CASE = 30.75` (task 4215, loadavg 120-176). This
   sweep measured a MARKED member of that family at **51.87s** setup
   (`orchestrator/tests/test_merge_lane_ratchet.py`) and another at 45.36s call.
   Under the same 8x rule that family would now derive 420, not 300. This task
   raises the constant only as far as the never-narrow rule requires (it must not
   fall below the ini default) and deliberately does NOT re-anchor
   `_MEASURED_UNDER_LOAD_WORST_CASE` or `_ABSOLUTE_FLOOR_SECONDS`, which are
   independent of the ini default by design. Filed as a follow-up.

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

3. **Root-bound mode was measured over the repo-root-owned test directories in
   full, and over the member trees only by targeted re-measurement of the worst
   unmarked tests the per-member sweep found.** A full serial root-bound run of
   every member tree was not attempted: `orchestrator` alone took 44 minutes at
   32 workers. The targeted form is what the value rests on, and it is the
   conservative direction — the measured member-to-root inflation factor is
   1.3-1.7x, so applying the largest of it to the next-worst unmarked figure in
   the corpus (25.27s) yields ~43s, still comfortably below the 61.52s the value
   is derived from.
