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

| architect | planRate | meanPQ_all | meanPQ_planned | $/fixture | $/usable-plan | vs incumbent |
|---|---|---|---|---|---|---|
| `architect-opus-max` *(incumbent)* | 95% | 0.8005 | 0.8450 | $4.363 | ~$4.60 | — |
| `architect-opus-high` | 95% | 0.8221 | 0.8678 | $3.655 | ~$3.85 | +0.0216 PQ, 0.84× cost |
| `architect-sonnet-high` | 84% | 0.7495 | 0.8900 | $1.599 | $2.037 | −0.0511 PQ, 0.37× cost |
| `architect-fable-high` | 74% | 0.6574 | 0.8921 | $3.456 | $4.731 | −0.1432 PQ, 0.79× cost |

`$/fixture` divides total candidate spend by all 19 scored fixtures, whether or
not a plan resulted; `$/usable-plan` divides by only the fixtures where a plan
was actually produced (see *planRate*) — the two are not interchangeable (see
*Methodological finding*, below).

## Verdict: MARGINAL — no clear pass

`architect-opus-high` is directionally better *and* ~16% cheaper than the
incumbent `architect-opus-max` at equal reliability (95% plan rate on both), but
+0.0216 mean `plan_quality` on a 0–1 scale at `trials=1`, n=19, with no confidence
interval sits inside run-to-run noise. This does not clear the PRD §ι "clear pass"
bar, so per the runbook this task takes the **escalate** branch, not the config-flip
branch.

The sonnet-architect-for-small-tasks candidate is **also rejected**, on
reliability rather than quality: `architect-sonnet-high` ties on
quality-when-planned (0.8900, second-highest nominally — `architect-fable-high`
nominally leads at 0.8921) but only produces a plan 84% of the time (3 no-plan
fixtures: `df_task_18`, `reify_task_12`, `reify_task_5221`), which makes its real
cost $2.037 per *usable* plan rather than the headline $1.599/fixture.

## Methodological finding for future campaigns

The discriminator across all four candidates is plan-production **reliability**
(plan rates 95/95/84/74), not plan **quality** (quality-when-planned tied at
0.845–0.892 across all four). A no-plan cell burns real budget and returns
nothing, so a cheap candidate's headline `$/fixture` advantage can be partly
illusory once quoted per usable plan instead — see
`plans/eval-architect-effort-verdict-2026-07-27.md` §"The discriminator is
reliability, not quality" for the full fixture-level breakdown. Any future
architect (or other-role) eval campaign should report and rank on reliability
first, quality-when-produced second — not a single blended quality number that
a low plan rate can quietly inflate.

## Decision and action taken

**No fleet-routing flip.** `models.architect` remains `"opus"` and
`effort.architect` remains `"max"` in
`orchestrator/src/orchestrator/defaults.yaml` (unchanged; already pinned by the
defaults-preservation assertions in `orchestrator/tests/test_config.py`'s
`test_project_config_overrides_defaults` — `config.models.architect == 'opus'`
/ `config.effort.architect == 'max'`). A provenance comment naming this
decision record has been added at the point of use so the hold is legible to
future operators (see that file's `models`/`effort` blocks). An `escalate_info`
record (`esc-2539-3`) carrying the report and the table above has been filed
(non-blocking — the marginal outcome is a spec'd, successful result of this
task, not an obstruction).

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
targets) and was **not cheaper per usable plan** than the incumbent
($4.731/usable-plan vs the incumbent's ~$4.60/usable-plan) — see
`plans/eval-architect-effort-verdict-2026-07-27.md` §"Bearing on the
fable-architect gate" for the fixture-level detail. The fable subset campaign
should treat plan-production reliability, not plan quality, as its primary
axis.

## Out of scope (filed as follow-ups, not fixed here)

Per decision 11 (measurement/adoption split), `evals/runner.py`, the benchmark
suite, and the judge/Elo machinery are owned by eval-framework-revival, not this
task. The verdict report's "Reporting defects found" section documents three
defects worth fixing before the next campaign (chiefly: `run_architect_eval`
never sets `tests_pass`, so `compute_composite`/`blend_composite` hard-gate every
architect row to `quality=0.0000`, which would in turn make
`report.select_survivors` pick a survivor alphabetically). These are filed as
follow-up tasks rather than edited in place here: **task 3099** (medium
priority — composite-zeroing + the resulting alphabetical `select_survivors`
tie-break, defects 1+2 combined by curation) and **task 3118** (high priority —
missing `cap_tainted`/`invocation_error` marker on a 429-scored cell, defect
3).
