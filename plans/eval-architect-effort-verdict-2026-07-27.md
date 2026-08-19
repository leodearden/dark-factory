# Architect-effort eval campaign — C4 verdict (2026-07-27)

**Gate:** esc-2848-1 (task 2848) · **Consumer:** task 2539
**Run:** `orchestrator eval-ofat --config dark-factory-orchestrator.yaml --trials 1 --max-parallel 2`
over the 22 stock fixtures. Started 2026-07-27 11:29 BST, finished 2026-07-28 01:31 BST
(14h02m wall clock), driver exit rc=0.
**Spend:** $578.85 over 118 saved cells (architect $262.17 · implementer $200.72 · judge $115.96).

## Verdict: MARGINAL — no automatic fleet-routing flip

`architect-opus-high` is directionally better *and* cheaper than the production
incumbent `architect-opus-max`, but the margin does not clear a clear-pass bar at
`trials=1`. Per the esc-2848-1 runbook this means task 2539 should **escalate**
rather than perform the config flip.

## Go/no-go precondition

`scripts/eval_bootstrap_smoke.sh` was re-run first and reached a REAL pass
(2026-07-27 08:41–11:27 BST, ~$21):

> SMOKE PASS: 4 architect result(s) with plan_steps>0 & done (BUG 1 fixed);
> 5 implementer worktree venv(s) on Python 3.13.\* with aiosqlite importable,
> and 3 verified implementer cell(s) (outcome=done & tests_pass=true) (BUG 2 fixed).

This is the first smoke pass since the LAYER-7 fix (task 2957) landed; the previous
failing run (07-22 18:15) pre-dated it. Architect cells carried non-zero
`plan_quality` (0.93–0.96) and no cell false-failed `tests_pass`.

## Headline numbers

19 fixtures (3 excluded — see *Data hygiene*). `planRate` = fraction of fixtures where
the architect actually emitted a plan (`plan_steps > 0`). `meanPQ_all` scores a
no-plan cell as 0; `meanPQ_planned` averages only cells that produced a plan.

| architect | planRate | meanPQ_all | meanPQ_planned | $/plan | vs incumbent |
|---|---|---|---|---|---|
| `architect-opus-max` *(incumbent)* | 95% | 0.8005 | 0.8450 | $4.363 | — |
| **`architect-opus-high`** | **95%** | **0.8221** | 0.8678 | **$3.655** | **+0.0216 PQ, 0.84× cost** |
| `architect-sonnet-high` | 84% | 0.7495 | 0.8900 | $1.599 | −0.0511 PQ, 0.37× cost |
| `architect-fable-high` | 74% | 0.6574 | 0.8921 | $3.456 | −0.1432 PQ, 0.79× cost |

### The discriminator is reliability, not quality

Quality *when a plan is produced* is effectively tied across all four candidates
(0.845–0.892) — and the two cheap candidates are nominally **highest**. On the 13
fixtures where all four emitted a plan, means cluster at 0.875–0.895 with wins split
5/4/5/7. At `trials=1` that spread is noise.

What actually separates the candidates is how often they emit a plan at all:

| architect | no-plan fixtures (genuine, outside cap windows) |
|---|---|
| `architect-opus-max` | 1 — `df_task_2778` |
| `architect-opus-high` | 1 — `reify_task_5021` |
| `architect-sonnet-high` | 3 — `df_task_18`, `reify_task_12`, `reify_task_5221` |
| `architect-fable-high` | 5 — `df_task_18`, `reify_task_12`, `reify_task_3981`, `reify_task_5021`, `reify_task_5221` |

A no-plan cell burns real budget ($0.5–$3) and returns nothing, so the cheap
candidates' apparent cost advantage is partly illusory: `architect-sonnet-high` is
$1.599/fixture but $2.037 per *usable* plan; `architect-fable-high` is $3.456/fixture
but $4.731 per usable plan — i.e. **fable is no cheaper per usable plan than the
opus-max incumbent** while failing to plan 5× as often.

## Why MARGINAL and not a clear pass

- `trials=1`, n=19 → no confidence interval. +0.0216 mean PQ on a 0–1 scale is well
  inside run-to-run noise.
- The advantage is consistent in *direction* (higher quality, equal reliability, 16%
  cheaper) but small in magnitude.
- The run suffered a cap-contamination event (below) that cost 3 fixtures.

A confirm batch (`eval-confirm --arch architect-opus-high --impl claude-sonnet-max
--trials 3 --max-parallel 2`) was **not** run — Step 2's conditional gate specifies it
only when the OFAT top architect differs from production, and the top result here is
inside noise of the incumbent rather than a distinct winner. Running a 3-trial confirm
on a difference this small would not resolve it either; what would is re-running the
OFAT screen at `--trials 3`.

## Data hygiene

**Excluded fixtures (3 of 22):**
- `df_task_2430`, `df_task_2430_adv_plan` — **total cap loss**. All four architects
  returned `$0.00 / 0 steps` within seconds. Root cause is unambiguous from the run
  log: the Claude CLI returned `"api_error_status": 429, "You've hit your session
  limit · resets 8pm"`. Zero signal, all candidates affected identically.
- `df_task_2370` — **asymmetric cap taint**. Two of four architects (fable, sonnet)
  hit the 19:14 cap window; the other two completed normally.

**Cap timeline:** `max-g` (session, 17:17 & 19:14), `max-f` (weekly → Jul 29 2pm),
`max-e` (session), `max-c` (weekly → Jul 29 11am). 40 invocations returned 429,
concentrated at 17:xx and 19:xx. All other no-plan cells sit outside these windows and
are genuine candidate failures.

**Ride-along baseline cells:** 80 of 198 cells failed with
`ValueError: Task <id> has no embedded plan. Run --plan-only to generate one first.`
16 of the 22 fixtures carry `plan: null`, and the 5 non-architect cells per fixture
(3 implementer incumbents + 2 judges) run against a *frozen* plan, so they cannot
execute without one — 16 × 5 = exactly 80. **This does not touch the architect
question**: architect cells generate their plan live and are immune, so all 88
architect cells ran. The implementer/judge rows in the emitted composite table are
therefore based on only 6 fixtures and should not be read as a fleet-wide baseline.

## Reporting defects found (recommend filing)

1. **The C4 composite table cannot rank architects.** Every architect row renders
   `quality=0.0000, composite=0.0000` even at `plan_quality=0.95`.
   `run_architect_eval` (`evals/runner.py`) builds `EvalMetrics` without setting
   `tests_pass`, which defaults `False`; both `compute_composite` and
   `blend_composite` (`evals/metrics.py`) hard-gate on `tests_pass` and return `0.0`.
   Every number in this verdict had to be recomputed from the per-cell result JSONs.
2. **`report.select_survivors` would pick the architect survivor alphabetically.** It
   ranks by descending `composite` with an ascending-config-name tiebreak; since all
   architect composites tie at 0.0, the tiebreak decides, yielding
   `architect-fable-high` on name order alone. Currently dormant — the code comment
   states the auto-driver is deliberately not yet wired and the operator picks
   `--arch` manually — but this is a live trap the moment that follow-up lands.
3. **A 429 cap is silently scored as `plan_quality=0.0`**, indistinguishable from a
   genuinely terrible plan. When the architect *and* the plan judge both 429, the cell
   records `plan_quality=0.0` via the structural floor with no cap marker on the
   result JSON. Cap-tainted cells had to be identified by correlating result mtimes
   against log timestamps. A `cap_tainted` / `invocation_error` field on `EvalMetrics`
   would make this recoverable automatically.
4. Non-fatal but noisy: every eval cell logs `Cold-verify shared-venv pre-provision
   failed (rc=2) — No pyproject.toml found in current directory or any parent
   directory`, then proceeds and passes.

## Bearing on the fable-architect gate (esc-2862-1)

This run is not the fable hard-subset campaign, but it is a strong preliminary signal:
`architect-fable-high` tied on quality-when-planned (0.8921, nominally best) yet had
the **worst reliability by a wide margin** (74% plan rate; 5 genuine no-plan fixtures,
including 3 of the 6 hard fixtures that campaign targets — `reify_task_12`,
`reify_task_5221`, `df_task_18`) and was **not cheaper per usable plan** than the
incumbent. The fable subset campaign should treat plan-production reliability, not
plan quality, as the primary axis.

## Recommendation

1. **No fleet-routing flip on this evidence.** Task 2539 escalates (marginal).
2. If the question is worth settling, the cheapest decisive next step is re-running the
   OFAT screen at `--trials 3` for the two opus candidates only, in a window with full
   account capacity — not a confirm batch.
3. Fix reporting defect 1 before the next campaign; it silently makes the primary
   output useless for the architect role.
