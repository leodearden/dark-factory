# fable-trial-v2 γ1 — stage-1 calibration report

**Gate:** esc-3633-1 (task 3633) · **PRD:** `plans/fable-architect-trial-v2-prd.md` §γ1
**Run:** 2026-08-24T14:52Z → 2026-08-25T00:5xZ (operator session, live OAuth pool)
**Cells:** 42 main pass + 3 cap-tainted re-runs = **45**; every fixture has one admissible cell
**Driver:** `scripts/run_fable_trial_v2_campaign.py` (β2, task 3632), consumed unmodified
**Consumer:** γ2 — banding ratification + regime ruling + stage-2 authorization

> **This report decides whether stage 2 runs at all.** It is provisional in one
> respect by design: `Q_ceiling` is empirically derived here and **ratified or
> adjusted by Leo at γ2** (PRD D6).

## Recipe as executed

| parameter | value | source |
|---|---|---|
| candidate | `architect-opus-max` (incumbent only) | recipe |
| trials | 1 per fixture | recipe |
| cell type | plan-only (`run_architect_eval`) | recipe |
| `max_budget_usd` | 15 | `--budget architect-opus-max=15` |
| `max_architect_turns` | 120 | fixture-pinned (`tasks_hard_v2/*.json`) |
| `timeout_minutes` | 180 | fixture-pinned |
| fixture pool | `orchestrator/src/orchestrator/evals/tasks_hard_v2/` | recipe |
| `--max-parallel` | 3 | paced for host load (~146 at launch) |
| account roster | `data/eval-campaign/eval-accounts-maxa-first.yaml` (7) | eval convention |

## Headline result — the live hypothesis is FALSIFIED

The gate's live hypothesis was that **today's incumbent plans all candidates**
(zero turn-exhaustions in 5,744 architect invocations since 2026-06-01), in
which case the historical seam would be exhausted and a null result would be the
success. That is **not** what stage 1 found.

| quantity | value |
|---|---|
| cells run | **45** (42 admissible + 3 cap-excluded, all three re-run to an admissible cell) |
| **planRate** (`plan_steps > 0`, judge-free, admitted pool) | **0.7381** (31/42) |
| fixtures the incumbent could **not** plan at all | **11 of 42 (26.2%)** |
| turn exhaustions (`turns_used >= 120`) | **0** — max observed 118 |
| budget exhaustions (`cost_usd >= 15`) | **0** — max observed $11.38 on a no-plan cell |

Both halves matter, and they point the same way:

* **Zero exhaustions.** The 5,744-invocation premise is *confirmed*. At a
  120-turn / $15 ceiling the incumbent never ran out of either resource.
* **And yet it failed to plan on a quarter of the pool.** The eleven no-plan cells
  terminated at 5, 19, 23, 26, 27, 27, 30, 36, 49, 55 and 72 turns — nowhere near
  the ceiling — and none exhausted budget (max $11.38 of $15).

So the honest reading is **not** "the historical seam is exhausted". It is that
**the seam moved**: the incumbent's failure mode at production ceilings is a
silent early no-plan, not resource exhaustion. The census that built this pool
selected on `error_max_turns` / `error_max_budget_usd`, which is precisely the
signature that has disappeared — while the underlying difficulty has not. A
pool selected on today's failure mode would look different again.

This band is where the hypothesis under test lives ("fable can plan tasks the
incumbent cannot"), and it is **not empty**. Stage 2 has a real target
population.

## Q_ceiling derivation (provisional — γ2 ratifies)

**Rule (PRD D6):** anchor on v1 **incumbent** cells on **validly-referenced**
fixtures, from the `data/eval-campaign/` dumps.

### Step 1 — reconstruct the v1 dataset (it is three dumps, not one)

`data/eval-campaign/fable-architect-only-results.json` is the 2026-07-28 phase,
in which ~40% of cells were killed by 429 caps and **both reify fixtures were
fully starved** — `data/eval-campaign/run_resume_phase.sh` records exactly this.
Two resume phases supersede it for the fixtures they re-ran:

| dump | supersedes |
|---|---|
| `fable-architect-only-results.json` (07-28) | base — all 6 fixtures |
| `resume-20260729/reify/…` | `reify_task_12`, `reify_task_27` |
| `resume-20260729/t2430/…` | `df_task_2430_adv_plan` |

Merging "resume supersedes base" yields 72 cells (6 fixtures × 4 candidates × 3
trials). **The merge is verified, not assumed** — it reproduces five
independently-published figures digit-for-digit:

| figure | published | reconstructed |
|---|---|---|
| fable lead on the five 3/3 fixtures | PRD P1: +0.0073 | **+0.0073** |
| fable lead on `reify_task_12` (1/3) | PRD P1: +0.3233 | **+0.3233** |
| headline mean lead | PRD P1: +0.0600 | **+0.0600** |
| fable $/usable plan | decision record: $4.429 | **$4.429** |
| incumbent $/usable plan | decision record: $3.693 | **$3.693** |

Using the un-merged base dump alone would have scored `reify_task_12` and
`reify_task_27` at plan_quality 0.0 across all incumbent trials — cap starvation
misread as incumbent failure — and produced a materially different Q_ceiling.

### Step 2 — which v1 fixtures were validly referenced AT RUN TIME

Not today's fixture state. The `reference` blocks for `reify_task_12`,
`reify_task_27` and `df_task_18` were back-filled by eval-revival σ in commit
`6d56acee58` (2026-08-06), **eight days after** the v1 campaign. Verified by
reading each fixture at `6d56acee58^`:

| v1 fixture | `reference` at run time | in anchor? |
|---|---|---|
| `df_task_18` | absent | no |
| `reify_task_12` | absent | no |
| `reify_task_27` | absent | no |
| `df_task_2284_adv_regression` | present | **yes** |
| `df_task_2339_adv_verify` | present | **yes** |
| `df_task_2430_adv_plan` | present | **yes** |

This is PRD P2 defect (2) — those three were judged on plausibility, not
fidelity — so they are correctly excluded from an anchor that must mean "quality
measured against a real reference diff".

### Step 3 — the anchor population (n=9 incumbent cells)

| fixture | trials |
|---|---|
| `df_task_2284_adv_regression` | 0.85, 0.87, 0.84 |
| `df_task_2339_adv_verify` | 0.87, 0.87, 0.88 |
| `df_task_2430_adv_plan` | 0.94, 0.95, 0.94 |

| statistic | value |
|---|---|
| min | 0.8400 |
| p25 / median | 0.8700 |
| **mean** | **0.8900** |
| p75 | 0.9400 |
| max | 0.9500 |
| stdev | 0.0418 |

**Provisional `Q_ceiling` = 0.8900** — the anchor **mean**: what the v1 incumbent
reaches on average when planning competently against a real reference. Chosen
over the median (0.8700) deliberately, because D6 resolves ambiguity to RETAIN, a
higher threshold discards fewer fixtures, and misbanding-to-discard is
permanently lossy while misbanding-to-retain costs stage-2 spend.

**Caveat γ2 must weigh:** all three anchor fixtures are **adversarial** fixtures,
because those are the only validly-referenced v1 fixtures that exist. The PRD
excludes adversarial fixtures from the v2 pool as adding nothing to a plan-only
screen — so the anchor is drawn from a fixture class the screen itself does not
use. That is a consequence of the v1 empty-reference defect, not a choice, and it
is the strongest single argument for treating 0.8900 as provisional.

## Band partition (PRD D6, computed by the driver)

Computed by `scripts/run_fable_trial_v2_campaign.py::partition_bands` at
`--q-ceiling 0.89`. Discard **only** unambiguous ceiling (planned AND valid
reference AND `plan_quality >= Q_ceiling`); retain everything else; ambiguity
resolves to RETAIN.

| band | fixtures | disposition |
|---|---:|---|
| ceiling | 7 | **DISCARD** |
| intermittent | 24 | retain |
| no_plan | 11 | retain |
| unmeasured | 0 | — (all three re-run to admissible) |
| **retained** | **35** | → stage 2 |

**Discarded (7):** `df_task_1229`, `df_task_2169`, `kl_task_543`,
`reify_task_2656`, `reify_task_2958`, `reify_task_3228`, `reify_task_3443`.

### Sensitivity — what the Q_ceiling ruling actually buys

| Q_ceiling | ceiling (discard) | intermittent | no_plan | unmeasured | **retained** |
|---:|---:|---:|---:|---:|---:|
| 0.84 (anchor min) | 11 | 20 | 11 | 0 | **31** |
| 0.87 (anchor median) | 10 | 21 | 11 | 0 | **32** |
| **0.89 (anchor mean — provisional)** | **7** | **24** | **11** | **0** | **35** |
| 0.94 (anchor p75) | 0 | 31 | 11 | 0 | **42** |

The ruling moves the retained set by only 31→35 across the plausible range
(0.84–0.89), so **Q_ceiling is not the lever that controls stage-2 cost.** The
reference gap below is.

## plan_quality validity — the binding constraint

`plan_quality` is interpretable **only where a valid `reference` block exists**.
Of 31 planned cells:

| population | n | min | median | mean | max |
|---|---:|---:|---:|---:|---:|
| **fidelity-scored** (valid reference) | 16 | 0.70 | 0.88 | **0.8556** | 0.93 |
| **plausibility-scored** (`judged_without_reference`) | 15 | 0.92 | 0.93 | **0.9367** | 0.95 |

**Plausibility scoring inflates plan_quality by +0.0810 on average, and
compresses the distribution to the top:** every one of the 15 plausibility-scored
cells lands in [0.92, 0.95], while the fidelity-scored cells spread across
[0.70, 0.93]. A judge with no reference diff cannot tell a good plan from a
plausible one, so it rates nearly everything highly.

This is PRD P2 defect (2) measured directly, and it corroborates independently:
`df_task_18` scored **0.9333** in v1 (plausibility) and **0.72** here
(fidelity) — a 0.21 drop on the same fixture, from the same incumbent, once σ's
back-filled reference made fidelity scoring possible.

**Consequence for the partition.** All 15 plausibility-scored cells band
`intermittent` at rung 3 (reference validity not known-good) and are RETAINED —
not because the incumbent struggled on them, but because *we cannot measure
whether it struggled*. Their raw scores (0.92–0.95) sit above `Q_ceiling`, so
they would band `ceiling` and be discarded **if** those scores were trustworthy —
and the +0.0810 inflation above is direct evidence that they are not. D6's
ambiguity→retain rule is doing exactly the right thing here; the cost is that
**15 of the 35 retained fixtures (43%) are retained on unmeasurability alone.**

That is the single largest lever on stage-2 cost, and it is addressable: those 15
were minted `planrate_only` by β1 because they lacked a clean single
`Merge task/<id>` SHA. Minting references for them would move them to a real
measurement rather than a precautionary retain.

## cap_excluded and judge-cost actuals

| quantity | value |
|---|---|
| `cap_excluded` | **3** of 45 (6.7%) — all three re-run to an admissible cell |
| spend on cap-excluded cells (bought no measurement) | **$26.64** |
| judge cost recorded | **$19.31** (5.4% of total spend) |
| judge cost in v1 | **$0.00 across all 235 cells** (defect ι3, now fixed) |

Cap-excluded fixtures: `reify_task_2324`, `reify_task_2325`, `reify_task_3779` —
all three re-run per recipe step 2 (see "Re-run" below).

**`judge_cost_usd` is now real.** v1's dollar figures understated absolute cost
by omitting it entirely; v2's include it. v1-vs-v2 absolute costs are therefore
**not** directly comparable — ratios are. ζ must repeat this caveat.

## Realized stage-2 cost projection per regime option

### Inputs (all measured here, none assumed from v1)

| input | value | basis |
|---|---|---|
| incumbent $/cell at production ceilings | **$8.57** | γ1 realized: $359.86 / 42 **admissible** cells, judge cost included ($8.00 per cell run) |
| v1 incumbent $/cell | $3.282 | v1 merged dump, judge cost **excluded** |
| fable/incumbent cell-cost ratio | **1.27×** | v1 `architect-fable-high` $4.183 / `architect-opus-max` $3.282 |
| retained set at Q=0.89 | **35 fixtures** | band partition above |
| stage-2 shape | 3 trials × 2 arms | PRD δ |

### Projection

Cells per arm = 35 × 3 = **105**; both arms = **210 cells**.

| regime (PRD D5) | recipe | projected stage-2 cost |
|---|---|---|
| **(a) equal-cost** | both arms `max_budget_usd=15` | **~$2,040** |
| **(b) equal-turns** | both at 120 turns, fable budget lifted to $25 | **~$2,440** |
| **(c) both arms** | (a) + fable-at-$25 arm; incumbent arm shared | **~$3,590** |

Across the plausible Q_ceiling range the spread is $1,810 (Q=0.84, R=31) to
$2,450 (Q=0.94, R=42) for equal-cost, and up to $4,300 for both-arms at R=42.

### ⚠️ This is 5–8× the PRD's estimate, and the two-stage rationale does not survive it

The PRD estimates δ at "~48 cells (retained-set dependent) ≈ $250–400+". Both
halves of that estimate are falsified by measurement:

* **Cell count:** 48 assumed ≈ 8 retained fixtures. Realized retained set is
  **35**, giving **210 cells** — 4.4× more.
* **Cell cost:** the estimate implicitly used v1's ~$3.3/cell. At production
  ceilings the realized cost is **$8.57/cell**, 2.6× higher, because v1 ran at
  50 turns / $20-default while γ1 runs at the production 120 / $15.

The PRD further justifies the whole two-stage design as *"stage 1 exists to
discover exactly that for ~$150 instead of ~$400+"*. Realized stage 1 cost
**$359.86** — roughly what stage 2 was estimated at — so the economic argument
for staging **as costed** does not hold. It does still hold *directionally*:
stage 1 cost $345 and has already discarded 7 fixtures and, more importantly,
told us the retained set is 35 rather than 8. Discovering that inside stage 2
would have cost far more.

**Two assumptions in the fable arm, stated rather than buried:**

1. The 1.27× ratio comes from `architect-fable-high`, but δ runs
   `architect-fable-max`. The v1 opus high→max step was 1.385×, so a
   fable-max cell is plausibly ~$11–14, not $10.43 — the projections above are
   more likely **under**-estimates than over.
2. Fable cell cost at a 120-turn ceiling has never been measured. Every fable
   number here is extrapolated from v1's 50-turn cells.

### Levers available to γ2 if ~$2,000 is not authorizable

Presented as options, not recommendations — the regime ruling is Leo's (D5), and
so is any rescope:

* **Mint references for the 15 `planrate_only` fixtures.** They are 43% of the
  retained set and are retained purely because their quality is unmeasurable. A
  real measurement would either discard them (large saving) or retain them on
  evidence.
* **Reduce trials from 3 to 2**, or run the no-plan band at 3 trials and the
  intermittent band at 1 — the no-plan band is where the hypothesis lives and is
  judge-free, so it is the cheapest signal per dollar.
* **Run the no-plan band only** (11 fixtures → 66 cells → ~$640 equal-cost).
  This directly tests "fable can plan what the incumbent cannot" and is scored on
  planRate, which needs no reference diff at all.

## Operational findings

### 1. The reserved eval account is auth-dead — φ's "rare" premise does not hold

The γ1 gate asserts "the UsageGate (revival φ) should make [cap-taint] rare".
Realized cap-taint was 6.7% overall, but peaked at **20% (2 of 10)** during a
degraded window, wasting 21% of spend at that point.

φ itself worked exactly as designed — it failed over on every attempt and marked
each refused cell `cap_tainted` with `plan_quality=None` rather than a false 0.0
(`orchestrator/src/orchestrator/evals/runner.py::run_architect_eval`). What it
cannot do is route around a pool with no healthy account left:

| account | state during γ1 |
|---|---|
| **max-a** | **AUTH-DEAD** — HTTP 401 "OAuth access token is invalid", every cell (13+ failovers) |
| max-f | CAPPED (weekly) — resets Aug 26 2pm Europe/London |
| max-g | CAPPED (session) — reset 8pm, re-capped, reset 1am |
| max-e, max-c, max-b, max-d | usable, shared with the live fleet |

**max-a is the account reserved for eval runs.** The documented convention
(`config/usage-accounts.yaml` header; `data/eval-campaign/run_resume_phase.sh`)
is "orchestrators see 6 accounts; eval runners see 7", giving a campaign a
reserve the fleet cannot exhaust. With max-a's `CLAUDE_OAUTH_TOKEN_A` stale in
`/home/leo/src/dark-factory/.env`, **that convention is not in force** — every
eval campaign until it is refreshed contends directly with production. The
dark-factory orchestrator journal showed cap events in the same window, so the
campaign is not merely a victim of the squeeze; it contributes to it.

**Operator action:** refresh `CLAUDE_OAUTH_TOKEN_A`. This is the highest-value
fix for any future campaign, and it is independent of the fable question.

### 2. A cap-tainted architect cell is not free

Each of the three cap-excluded cells banked the spend of its attempts before the
refusal (~$8.88 average) and burned ~33 min of bounded-patience wait, then cost
that again on the prescribed re-run. Cap-taint costs real money and wall-clock,
twice.

### 3. No-plan cells are genuine, not instrument artifacts

Each cell logs exactly 3 `MCP initialize transient error: ConnectError` lines.
These are the **ambient** `.mcp.json` HTTP servers (`fused-memory`,
`escalation`, `reify-debug`), unreachable from an eval worktree. Plan-tools is
injected separately as a stdio server
(`orchestrator/src/orchestrator/evals/runner.py::run_architect_eval` via
`_inject_plan_tools_mcp`) and is not in that set. The error count is **uniform
across cells that produced 29-step plans and cells that produced none**, so it
cannot be the cause of a no-plan.

This was checked deliberately: had plan-tools been down, every no-plan cell would
have been a fabricated 0.0 and the partition would have erred in D6's
permanently-lossy direction. It corroborates independently — `reify_task_12`
no-plans here, and v1's incumbent planned only 1 of 3 trials on the same fixture.

### 4. Realized pool size exceeded the recipe's estimate

The gate estimated "~34±4 cells". β1 minted **39 included + 3 continuity = 42**,
above the stated band. Combined with the 2.5× per-cell cost this is the whole of
the spend overrun.

## Per-fixture results (45 cells over 42 fixtures)

One row per fixture, showing its **admissible** cell. `planRate` is per-fixture at
1 trial, so it is 1 or 0 — the pool-level rate is 0.7381 over the 42 admissible
cells. `ref valid` is the `judged_without_reference` marker (eval-revival σ)
inverted: "no" means plan_quality is plausibility-scored and **not** interpretable
as fidelity. `cost` sums every cell for that fixture, including a discarded
cap-tainted attempt where one occurred (`re-run` = Y).

| fixture | outcome | turns | plan_steps | planRate | plan_quality | ref valid | band | cost | judge | re-run |
|---|---|---:|---:|:-:|---:|:-:|---|---:|---:|:-:|
| `df_task_1229` | done | 83 | 29 | 1 | 0.90 | yes | ceiling | $8.13 | $0.86 |  |
| `df_task_18` | done | 88 | 28 | 1 | 0.72 | yes | intermittent | $8.60 | $1.48 |  |
| `df_task_2169` | done | 96 | 16 | 1 | 0.89 | yes | ceiling | $10.46 | $1.35 |  |
| `df_task_2260` | done | 55 | 0 | 0 | 0.00 | yes | no_plan | $8.18 | $0.00 |  |
| `df_task_882` | done | 57 | 3 | 1 | 0.95 | no | intermittent | $3.82 | $0.31 |  |
| `kl_task_543` | done | 90 | 24 | 1 | 0.90 | yes | ceiling | $14.85 | $0.83 |  |
| `reify_task_12` | blocked | 5 | 0 | 0 | 0.00 | yes | no_plan | $0.30 | $0.00 |  |
| `reify_task_2320` | done | 78 | 16 | 1 | 0.85 | yes | intermittent | $9.83 | $0.64 |  |
| `reify_task_2324` | done | 27 | 0 | 0 | 0.00 | yes | no_plan | $9.76 | $0.00 | Y |
| `reify_task_2325` | done | 61 | 10 | 1 | 0.92 | no | intermittent | $13.67 | $0.33 | Y |
| `reify_task_2330` | done | 47 | 17 | 1 | 0.95 | no | intermittent | $10.72 | $0.50 |  |
| `reify_task_2336` | done | 59 | 10 | 1 | 0.93 | no | intermittent | $8.61 | $0.32 |  |
| `reify_task_2379` | done | 46 | 16 | 1 | 0.93 | no | intermittent | $6.26 | $0.64 |  |
| `reify_task_2384` | done | 86 | 16 | 1 | 0.93 | no | intermittent | $12.30 | $0.34 |  |
| `reify_task_2531` | done | 23 | 0 | 0 | 0.00 | yes | no_plan | $1.53 | $0.00 |  |
| `reify_task_2573` | done | 30 | 0 | 0 | 0.00 | yes | no_plan | $1.43 | $0.00 |  |
| `reify_task_2654` | done | 76 | 16 | 1 | 0.70 | yes | intermittent | $6.18 | $0.58 |  |
| `reify_task_2655` | done | 85 | 20 | 1 | 0.92 | no | intermittent | $12.66 | $0.72 |  |
| `reify_task_2656` | done | 85 | 13 | 1 | 0.92 | yes | ceiling | $6.93 | $0.60 |  |
| `reify_task_2696` | done | 94 | 14 | 1 | 0.95 | no | intermittent | $9.77 | $0.42 |  |
| `reify_task_2699` | done | 19 | 0 | 0 | 0.00 | yes | no_plan | $1.33 | $0.00 |  |
| `reify_task_27` | done | 79 | 24 | 1 | 0.80 | yes | intermittent | $5.53 | $0.72 |  |
| `reify_task_2778` | done | 27 | 0 | 0 | 0.00 | yes | no_plan | $1.53 | $0.00 |  |
| `reify_task_2908` | done | 118 | 18 | 1 | 0.92 | no | intermittent | $15.58 | $0.84 |  |
| `reify_task_2911` | done | 89 | 18 | 1 | 0.87 | yes | intermittent | $11.52 | $0.51 |  |
| `reify_task_2958` | done | 77 | 17 | 1 | 0.93 | yes | ceiling | $5.77 | $0.67 |  |
| `reify_task_3004` | done | 80 | 14 | 1 | 0.93 | no | intermittent | $8.52 | $0.38 |  |
| `reify_task_3024` | done | 104 | 18 | 1 | 0.88 | yes | intermittent | $9.77 | $0.61 |  |
| `reify_task_3092` | done | 81 | 16 | 1 | 0.95 | no | intermittent | $10.19 | $0.44 |  |
| `reify_task_3095` | done | 84 | 20 | 1 | 0.80 | yes | intermittent | $10.93 | $1.00 |  |
| `reify_task_3228` | done | 100 | 13 | 1 | 0.91 | yes | ceiling | $11.43 | $0.44 |  |
| `reify_task_3443` | done | 70 | 7 | 1 | 0.92 | yes | ceiling | $7.06 | $0.75 |  |
| `reify_task_3586` | done | 82 | 13 | 1 | 0.95 | no | intermittent | $14.26 | $0.45 |  |
| `reify_task_3779` | done | 75 | 10 | 1 | 0.82 | yes | intermittent | $18.11 | $0.49 | Y |
| `reify_task_3822` | done | 65 | 20 | 1 | 0.95 | no | intermittent | $11.27 | $0.60 |  |
| `reify_task_3834` | done | 84 | 16 | 1 | 0.88 | yes | intermittent | $11.52 | $0.73 |  |
| `reify_task_3845` | done | 115 | 16 | 1 | 0.95 | no | intermittent | $11.37 | $0.41 |  |
| `reify_task_3883` | done | 49 | 0 | 0 | 0.00 | yes | no_plan | $3.29 | $0.00 |  |
| `reify_task_4026` | done | 36 | 0 | 0 | 0.00 | yes | no_plan | $4.06 | $0.00 |  |
| `reify_task_4086` | done | 73 | 8 | 1 | 0.92 | no | intermittent | $7.37 | $0.36 |  |
| `reify_task_4370` | done | 26 | 0 | 0 | 0.00 | yes | no_plan | $4.10 | $0.00 |  |
| `reify_task_4832` | done | 72 | 0 | 0 | 0.00 | yes | no_plan | $11.38 | $0.00 |  |

## Re-run of cap-tainted cells (recipe step 2)

Three fixtures were cap-excluded on the main pass and were re-run so that
**every fixture has one admissible cell**, as the recipe requires. The re-run
used the identical driver and parameters over a fixture subset derived at call
time from cells lacking an admissible result — never a hand-typed list.

| fixture | main pass | re-run | final band |
|---|---|---|---|
| `reify_task_2324` | cap-tainted, $8.08 | 27 turns, 0 steps | **no_plan** (retain) |
| `reify_task_2325` | cap-tainted, $8.78 | 61 turns, 10 steps, pq 0.92, no valid ref | **intermittent** (retain) |
| `reify_task_3779` | cap-tainted, $9.77 | 75 turns, 10 steps, pq 0.82, valid ref | **intermittent** (retain) |

All three retain, so the retained set is unchanged at 35 — but they are now
retained **on measurement** rather than on D6's ambiguity rule, and the
`unmeasured` band is empty. `reify_task_2324` adds an eleventh member to the
no-plan band.

## What γ2 must rule

1. **Ratify or adjust `Q_ceiling`** (provisional 0.8900, anchor mean). Note the
   sensitivity table: the retained set moves only 31→35 across 0.84–0.89, so this
   ruling is not the cost lever it might appear to be. The live caveat is that
   the anchor is drawn entirely from adversarial fixtures, the only
   validly-referenced v1 cells that exist.
2. **Rule the comparison regime** — equal-cost (~$2,040), equal-turns (~$2,440),
   or both arms (~$3,590), all at the retained set of 35. Deliberately
   un-defaulted per D5.
3. **Authorize (or decline, or rescope) stage-2 spend** at 5–8× the PRD's
   estimate. The levers listed above — minting references for the 15
   `planrate_only` fixtures, reducing trials, or running the no-plan band alone
   (~$640) — are options for Leo, not recommendations from this report.

**What this report does NOT conclude.** It does not recommend admitting or
declining fable; that is η's ruling on ζ's evidence, and stage 2 has not run. It
does not conclude the historical seam is exhausted — the opposite, on the
measurement. And it does not treat the 15 plausibility-scored fixtures as
ceiling-saturated despite their 0.92–0.95 scores, because the +0.0810 inflation
measured here is direct evidence those scores cannot bear that weight.

## Reproduction

```bash
# main pass (42 cells)
python3 scripts/run_fable_trial_v2_campaign.py --run \
  --candidate architect-opus-max --budget architect-opus-max=15 \
  --trials 1 --max-parallel 3 --stage1 --q-ceiling 0.89 \
  --config dark-factory-orchestrator.yaml

# re-analysis / re-banding at another threshold (zero spend)
python3 scripts/run_fable_trial_v2_campaign.py --results-dir <dir> \
  --candidate architect-opus-max --budget architect-opus-max=15 \
  --trials 1 --stage1 --q-ceiling <Q>
```

**Do not point `--results-dir` at the packaged `evals/results/` directory.** It
holds 19 pre-existing v1 `architect-opus-max` cells on the three continuity
fixture stems (`reify_task_12`, `reify_task_27`, `df_task_18`), which pass both
axes of `scripts/run_fable_trial_v2_campaign.py::filter_campaign_results` and
would silently contaminate the calibration. This campaign's cells were
snapshotted to an isolated directory by modification time before re-analysis.
