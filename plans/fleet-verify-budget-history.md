# Fallback fleet-chain verify budget — history

This file holds the history of one lever: `verify_command_timeout_secs` /
`verify_cold_command_timeout_secs` in `dark-factory-orchestrator.yaml`, the
budget of the FALLBACK fleet chain (`test_command` in the same file). The yaml
comment above the two keys states the current disposition only; the dated
history and findings live here.

The measured figures are not copied into this file either. They live once, in
`tests/scripts/test_fallback_verify_config.py::MEASURED_FLEET_SEGMENT_SECS`
(with dated provenance in `MEASURED_FLEET_SEGMENT_PROVENANCE`). Since task 3496
the table's `orchestrator` row is derived from
`tests/scripts/test_module_verify_budgets.py::ORCHESTRATOR_BUDGET_CENSUS`. The
figures below are dated snapshots of those records.

Value history (warm / cold):

- 1800 / 3600 until 2026-07-31.
- 3600 / 5400 from 2026-07-31 (task 3350, commit `416313e39e`).
- 7200 / 10800 from 2026-09-12 (Leo, commit `36c4c71eb4`).

---

## Until 2026-07-31 — 1800s on a "~2 min" premise

The budget was sized under the comment "Full warm verify here is ~2 min". By
then that was wrong by an order of magnitude: the premise predated the monorepo
growing to nine chained suites. The same comment carried the intent this
history keeps returning to: "Tight timeouts surface hangs (e.g. asyncio-sleep
loops in a misconfigured mock) on the first verify that touches the module, not
8 days later at a post-completion full review."

Task 3062 attempt-2 (run started 2026-07-31T02:00:48Z under `nice -n 15 ionice
-c2 -n7`, surfaced as esc-3062-3) timed out at 1800.66s before dashboard's
segment started, after passing 25,536 tests with zero failures. That was a
structural overrun, not a hang. A ceiling below the measured floor cannot be
cleared by a healthy run, so it surfaces no hangs. It manufactures
`infra_timeout` on the honest green path. That run's per-segment figures seeded
the measured table, and dashboard, sampler and cockpit have had no figure in it
ever since.

## 2026-07-31 — task 3350: 3600 warm / 5400 cold

The floor was five measured segments summing to 1838.60s. With task 3384's
`scripts/tests` (~113s) and the estimates for the three unreached suites
(dashboard ~190, sampler ~60, cockpit ~44), the honest green path was ~2246s,
~1.6x under 3600. Cold got 5400 because it additionally pays the ~180s
`uv sync --all-packages` preprovision plus uv sync across six subprojects.

The load variance budgeted for was the two same-day 2026-07-31 orchestrator
measurements (1157.62s vs 1366.23s, ~18% apart). Task 4902 showed that figure
to be false, not merely soft (below).

The "tight timeouts surface hangs" intent moved to the per-module budgets. Each
subproject's own `orchestrator.yaml` budget is sized from that module's measured
worst run. The intent cannot hold at the fleet level, for the reason the 1800s
era demonstrated.

## 2026-08-28 — task 4902: the orchestrator segment re-measured by hand

Commit `685f558728` (2026-08-20) set `verify_admission_pytest_n: "8"`, capping
orchestrator's xdist fanout. Its median green full-suite cost stepped from
691.40s to 1765.95s (~2.6x). The table stayed at 1366.23 and the floor guard
stayed green for eight days, until 4902 re-measured the segment by hand. The
guard's own docstring had predicted exactly this. The corpus, selection rules
and percentile conventions are recorded beside
`test_fallback_verify_config.py::POST_CAP_ORCHESTRATOR_GREEN_SECS`.

Post-cap green full-suite runs: n=28, 864.83s to 3310.50s (a 3.8x spread), 14
of the 28 above 1800s. This replaced the ~18% variance claim.

**Finding.** The path was green at the median (~2645s, 1.36x under 3600), thin
at p90 (~3431s, 1.05x), and over the ceiling at the observed maximum (~4190s).
That last row was not hypothetical. One run had already consumed the full
ceiling and been recorded as a false `infra_timeout`: timed_out true, rc 1,
3600.649s, started 2026-08-28T17:25:05Z, observed at
`.worktrees/4023/.task/verify/attempt-1.orchestrator.summary.json`.

**The inlined figures are the evidence; the path is not.** A
`.task/verify/*.summary.json` is a transient, per-attempt artifact. It is
overwritten by the next attempt in the same worktree and pruned with the
worktree. Re-checked 2026-08-30, that exact path had already been rewritten by
a later, unrelated attempt (rc 1, timed_out false, 2884.15s, started
2026-08-30T08:49:51Z). On the same day a different worktree showed the same
ceiling hit (timed_out true, 3605.06s). To re-establish the phenomenon, re-mine
the corpus glob (`.worktrees/*/.task/verify/*.orchestrator.summary.json`), never
one path.

The remedies, recorded as the pinned operator decision on task 3353's L1:

- exempt the `orchestrator` prefix from the -n cap;
- give `orchestrator/orchestrator.yaml` its own budget;
- run task 3589's fixed-CPUQuota A/B.

Task 4902 took none of them. It changed no budget, no -n cap and no per-module
yaml.

## 2026-09-12 — Leo: 7200 warm / 10800 cold

The raise overrode the yaml's then-standing guidance, "not raising this number
again". That guidance named the right answers: scope verify to the task's
declared modules, or shard the suite (task 3353). Neither was implemented, and
the cost of waiting was being paid in false reds. Five worktrees were lost to
false verify timeouts in six days (esc-4211-6), each burning ~1h of 8 workers.
Task 3202, complete and review-clean, could not pass verify at any priority.

At the time the raise loosened only `orchestrator`. That module was the single
entry in `MODULE_BUDGET_EXCLUSIONS` and hard-fell-through to this value. 7200
was the figure already ruled as task 3353 scope B. The cold value was scaled x2
so the ruled 1.5x cold/warm relationship was preserved, because warm <= cold is
asserted (tasks 3350 and 3397).

The cost: a genuinely wedged run holds a verify slot for 2h warm / 3h cold
instead of 1h / 1.5h. `verify_admission_task_slots` is 1, so that slot is the
whole task-verify lane. This was accepted against ~1 stranded green worktree per
day.

The revert condition, as written: "when task 3353 lands (split or shard the
orchestrator suite) or the module declares its own budget, put these back to
3600/5400 ... the next raise should be refused and 3353 done instead."

## After the raise — tasks 5422, 3353 (D17) and 5408

- **Task 5422.** `orchestrator/orchestrator.yaml` now declares its own 7200
  warm (ruled, esc-4211-6) and 10800 cold, and `MODULE_BUDGET_EXCLUSIONS` is
  empty. From then on the root budget bounds only the fallback fleet chain.
- **Task 3353 census**, measured 2026-09-14 over the regime since
  2026-09-12T08: n=14, p50 3274.92 / p90 3684.59 / max 4626.17, 0 timed out.
- **Task 5408.** Dashboard, escalation and cockpit moved to pytest-xdist.
  Measured on that tree:
  - dashboard: 2299 passed / 1 xfailed in 95.07s
  - escalation: 1721 passed / 2 xfailed in 49.63s
  - cockpit: 346 passed in 16.78s

## 2026-10-02 — task 3496: reconciliation, and the revert finding

The table now measures six of the nine suites. `scripts/tests` became a measured
row (113.0, task 3384's standalone run), and the `orchestrator` row became the
census p50. The floor is 3860.29s. The honest green path is ~4154s / ~4564s /
~5506s at census p50 / p90 / max, which is 1.73x / 1.58x / 1.31x under 7200.

**Finding, recorded and not acted on.** Both revert triggers have fired: task
3353 is done, and the module declared its own budget. The revert still cannot be
executed as written. 3600 is below the measured floor and ~0.87x the honest
green path at p50, so
`test_fallback_verify_config.py::test_fallback_verify_budget_clears_the_measured_fleet_chain_floor`
refuses it. Lowering the fleet budget now needs an operator decision about the
fallback chain itself.

The trend that decision faces is in the orchestrator segment alone:

| Date | Orchestrator segment | Source |
|---|---|---|
| 2026-07-31 | 1366s | one run |
| 2026-08-28 | 1765.95s median | task 4902 |
| 2026-09-14 | 3274.92s median | task 3353 census |

`scripts/tests` adds load sensitivity of its own. It runs ~113s idle; combined
with `tests/scripts/` in one clause it measured 288.65s / 411.87s / 494.20s under
contention (task 3460).
