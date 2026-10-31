# Cancel-on-red of a plan's still-waiting legs — decision δ (task 5630)

Task 5630 is task δ of `plans/verify-admission-fifo-prd.md`.

**Decision (2026-10-31): not worth a task.** On DF, legs granted after a sibling
module had already ended red took **0.95 %** of slot-busy time, with a 95 % interval of
0.14–2.35 %. That is below the 3 % line (≈ 0.7 h/day) at both ends of the interval. No
cancel-on-red task is filed.

A larger lever turned up during the measurement and is filed separately as
`tkt_0RWDTQFWFYP1VM33YFRB7EZHF0`. fused-memory test attempts that timed out and were
retried held 5.31 % of slot-busy time (see *Timeout retries*).

## Numbers

Source: `verify_admission_plan` rows in each project's `data/orchestrator/runs.db` with
`order='arrival'`, `outcome='returned'` and at least one leg. The store was opened read-only
at 2026-10-31T00:17Z. Slot-busy time is Σ `held_secs` over every admitted test leg of every
role. The headline numerator is task-role legs only.

| | dark_factory | reify |
|---|---|---|
| Window (first → last kept row) | 2026-10-17T06:07:41Z → 2026-10-31T00:04:31Z, **13.75 d** | 2026-10-14T04:00:32Z → 2026-10-30T23:43:09Z, 16.82 d |
| Plans (total / task / background) | 1,284 / 1,240 / 44 | 601 / 556 / 45 |
| n = task plans with ≥ 2 modules and a module red on its final attempt | **78** (120 counting all roles) | 0 |
| Slot-busy | 995,917 s = 276.6 h | 1,344,065 s = 373.4 h |
| **Avoidable, module-level (decision basis)** | **9,445 s = 2.62 h = 0.948 %**, interval 0.143–2.346 %, 0.19 h/day | 0 s = 0 % |
| Avoidable, literal leg-level (transparency only) | 52,757 s = 14.65 h = 5.30 % | 2,195 s = 0.16 % |
| Imposed wait (avoidable hold time while ≥ 1 other plan's leg waited) | 9,445 s, coverage 1.000 | 0 s |
| Background (sweep / checkpoint) post-red time, outside scope | 129,385 s = 35.94 h | 0 s |
| Cost side: avoidable legs / of them queued at the red / avoidable modules that were themselves red / plans that would lose a red | 26 / 20 / 7 / 5 | 0 / 0 / 0 / 0 |
| Retried attempts (`retried_attempts`): attempts / of them timed out / held / share of slot-busy / modules retried / ended green | 44 / 44 / 52,889 s = 14.69 h / 5.31 % / 34 / 30 | 1 / 1 / 1,552 s / 0.12 % / 1 / 1 |
| Verdict (script) | `decline` | `thin_data` |

reify's `thin_data` is structural, not under-sampling. Measured: 600 of its 601 plans have one
leg, and that leg's module is `null` (518 legs) or `'__fallback__'` (84 legs). A plan with one
unit has no sibling to cancel, so the lever is zero there.

## Definitions

- **Module level (the decision basis).** A plan's legs are grouped by module, and a module is
  judged by its **final** attempt (by `granted_at`). A module is red when that attempt has
  `rc != 0` or `timed_out`. Its red end is that attempt's `granted_at + held_secs`. A leg is
  avoidable when it was granted after the earliest red end among the **other** modules of its
  plan. A leg already running at that moment is out of scope.
- **Literal leg level.** This is the same rule with every leg as its own unit, which is the
  PRD's wording read literally. It is reported for transparency and does not decide.
  `orchestrator/src/orchestrator/verify.py::run_verification` re-runs a pure-timeout test
  attempt, and each attempt is recorded as its own leg. So the literal reading counts a
  timed-out attempt as "red", and every retry after it as "avoidable". Of the 34 retried
  modules, 30 then passed. A cancel keyed on a timed-out attempt would have turned those
  green plans red. The 5.30 % describes that harmful mechanism, not the proposed one.
- **Short window.** The PRD asks for ≥ 14 days of arrival rows. DF has 13.75 d, and its
  14-day mark is 2026-10-31T06:07:41Z. Under 14 d, the census decides only when the whole
  95 % interval lies on one side of 3 % (a bootstrap over plans, 2,000 resamples, seed 5630).
  Otherwise it returns `thin_data`. DF's upper bound of 2.35 % is below 3 %, so it decides.
  n must also be ≥ 40, and DF has 78. n counts task plans only because only they enter the
  numerator.

## Not measured: lint/type early red

The PRD also asks about legs granted after the plan's lint/type had failed. That is **not
recorded per plan**:

- `verify_admission_plan` carries only the slot-holding test legs
  (`orchestrator/src/orchestrator/verify_admission_record.py::AdmissionLeg`). Lint and type
  run beside the test leg, outside the slot.
- The only other per-verify record, `workflow_verify`, carries no `plan_id` and no per-unit
  times. It is emitted only for passing verifies: 1,119 of the 1,119 rows since
  2026-10-17T06:07:41Z have `passed=true`.

Any figure here would be a join on task id and timestamp, which is a guess.

## Timeout retries (filed: `tkt_0RWDTQFWFYP1VM33YFRB7EZHF0`)

Every one of the window's 59 timed-out legs (47 task, 12 background) is the fused-memory test
leg, held 1,200–1,307 s each. All 44 retried attempts timed out (`retried_attempts.timed_out`,
which counts timeouts apart from env-recovery and flake reruns). The ticket asks why that
suite overruns `fused-memory/orchestrator.yaml`'s 1,200 s budget under the slot. It does not
ask to change the retry.

## Context the decision leaves alone

- Background plans (main-tip sweep, review checkpoint) spent 35.9 h after a red module. A
  sweep is a drift detector that needs every module's result, so cancelling its legs would
  defeat it. That time is in the slot-busy denominator only.
- PRD open question 3 (grant a plan's cheapest leg first) was conditional on δ filing
  cancel-on-red. It lapses with this declination.

## Reproduce

```
python3 scripts/verify_admission_cancel_on_red_census.py \
  --db dark_factory=/home/leo/src/dark-factory/data/orchestrator/runs.db \
  --db reify=/home/leo/src/reify/data/orchestrator/runs.db
```

The script is stdlib-only and read-only, and prints one JSON object per `--db`. Later rows
widen the window, so a re-run will not reproduce these figures digit for digit.
