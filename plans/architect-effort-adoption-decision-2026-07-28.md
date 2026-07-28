# Architect effort/model adoption decision — ι (2026-07-28)

**Task:** 2539 · **PRD:** `plans/adaptive-model-routing-prd.md` §ι ("Architect
effort/model decide-and-act (adoption of an eval-revival verdict)")
**Evidence:** `plans/eval-architect-effort-verdict-2026-07-27.md`, produced under
gate esc-2848-1 (task 2848) explicitly "for consumption by task 2539".
**Run:** `orchestrator eval-ofat --config dark-factory-orchestrator.yaml --trials 1
--max-parallel 2` over the 22 stock fixtures (19 scored after 3 cap-tainted
exclusions). Started 2026-07-27 11:29 BST, finished 2026-07-28 01:31 BST (14h02m
wall clock), driver exit rc=0, $578.85 spend. Preceded by a REAL
`eval_bootstrap_smoke.sh` pass (2026-07-27 08:41–11:27 BST) — the first since task
2957's layer-7 `ORCH_*` scrub.

## Headline numbers (from the report)

| architect | planRate | meanPQ_all | $/plan | vs incumbent |
|---|---|---|---|---|
| `architect-opus-max` *(incumbent)* | 95% | 0.8005 | $4.363 | — |
| `architect-opus-high` | 95% | 0.8221 | $3.655 | +0.0216 PQ, 0.84× cost |
| `architect-sonnet-high` | 84% | 0.7495 | $1.599 | −0.0511 PQ, 0.37× cost |
| `architect-fable-high` | 74% | 0.6574 | $3.456 | −0.1432 PQ, 0.79× cost |

## Verdict: MARGINAL — no clear pass

`architect-opus-high` is directionally better *and* ~16% cheaper than the
incumbent `architect-opus-max` at equal reliability (95% plan rate on both), but
+0.0216 mean `plan_quality` on a 0–1 scale at `trials=1`, n=19, with no confidence
interval sits inside run-to-run noise. This does not clear the PRD §ι "clear pass"
bar, so per the runbook this task takes the **escalate** branch, not the config-flip
branch.

The sonnet-architect-for-small-tasks candidate is **also rejected**, on
reliability rather than quality: `architect-sonnet-high` ties (nominally leads) on
quality-when-planned (0.8900) but only produces a plan 84% of the time (3 no-plan
fixtures: `df_task_18`, `reify_task_12`, `reify_task_5221`), which makes its real
cost $2.037 per *usable* plan rather than the headline $1.599/fixture.

## Methodological finding for future campaigns

The discriminator across all four candidates is plan-production **reliability**,
not plan **quality**. Quality-when-planned is effectively tied (0.845–0.892 across
all four; on the 13 fixtures where all four produced a plan, means cluster at
0.875–0.895 with wins split 5/4/5/7 — noise at `trials=1`). What actually separates
the candidates is how often they emit a plan at all: plan rates split 95/95/84/74.
A no-plan cell burns real budget ($0.5–$3) and returns nothing, so a cheap
candidate's headline cost advantage can be partly illusory once quoted per
*usable* plan instead of per fixture. Any future architect (or other-role)
eval campaign should report and rank on reliability first, quality-when-produced
second — not a single blended quality number that a low plan rate can quietly
inflate.

## Decision and action taken

**No fleet-routing flip.** `models.architect` remains `"opus"` and
`effort.architect` remains `"max"` in
`orchestrator/src/orchestrator/defaults.yaml` (unchanged; already pinned by
`orchestrator/tests/test_config.py:995-996`). A provenance comment naming this
decision record has been added at the point of use so the hold is legible to
future operators (see that file's `models`/`effort` blocks). An `escalate_info`
record carrying the report and the table above has been filed (non-blocking —
the marginal outcome is a spec'd, successful result of this task, not an
obstruction).

## Recommended decisive next step (from the report)

Re-run the OFAT screen at `--trials 3` for the two opus candidates only
(`architect-opus-max` vs `architect-opus-high`), in a window with full account
capacity. A 3-trial *confirm* batch was deliberately not run and would not
resolve a difference this small either — Step 2's conditional confirm gate
applies only when the OFAT top architect differs from production, and here the
top result is inside noise of the incumbent rather than a distinct winner. What
would resolve it is more trials on the OFAT screen itself, for the opus pair
only (sonnet/fable are already ruled out on reliability, independent of any
quality refinement).

## Bearing on the fable-architect gate (esc-2862-1)

This run is not the fable hard-subset campaign, but it is a strong preliminary
signal: `architect-fable-high` tied on quality-when-planned (0.8921, nominally
best) yet had the **worst reliability by a wide margin** (74% plan rate; 5
genuine no-plan fixtures, including 3 of the 6 hard fixtures that campaign
targets — `reify_task_12`, `reify_task_5221`, `df_task_18`) and was **not
cheaper per usable plan** than the incumbent ($4.731/usable-plan vs $4.363). The
fable subset campaign should treat plan-production reliability, not plan
quality, as its primary axis.

## Out of scope (filed as follow-ups, not fixed here)

Per decision 11 (measurement/adoption split), `evals/runner.py`, the benchmark
suite, and the judge/Elo machinery are owned by eval-framework-revival, not this
task. The verdict report's "Reporting defects found" section documents three
defects worth fixing before the next campaign (chiefly: `run_architect_eval`
never sets `tests_pass`, so `compute_composite`/`blend_composite` hard-gate every
architect row to `quality=0.0000`, which would in turn make
`report.select_survivors` pick a survivor alphabetically). These are filed as
low-priority follow-up task candidates rather than edited in place here.
