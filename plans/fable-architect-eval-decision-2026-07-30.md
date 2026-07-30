# Fable-architect eval decision record — hard-subset composite (2026-07-30)

**Task:** 2863 (τ2) · **PRD:** `plans/fable-architect-eval-admission-prd.md` §τ2 ·
**Gate:** esc-2862-1 (task 2862, `status=resolved`, resolved 2026-07-30T15:30:58Z by
`eval-campaign-2026-07-27`) · **Consumers:** τ3 (admission ratification gate) and task
2544 (fable admission flip, `plans/adaptive-model-routing-prd.md` ξ, amended)

**Run:** `orchestrator eval-ofat --tasks-dir <hard-subset> --trials 3`, hard-subset =
the six PRD fixtures (`df_task_2284_adv_regression`, `df_task_2339_adv_verify`,
`df_task_2430_adv_plan`, `reify_task_12`, `reify_task_27`, `df_task_18`) × four
architect candidates (`architect-opus-max` *(incumbent)*, `architect-opus-high`,
`architect-sonnet-high`, `architect-fable-high`). Architect cells are plan-only: one
live architect call + one judge call per cell, downstream frozen — that is what makes
a 3-trial, 6-fixture, 4-candidate screen (72 cells) cheap enough to run as a screen.

**No config change applied by this record.** Per PRD decision 4, admitting a new
top-tier architect model is fleet-autonomy expansion, which the owner ratifies —
never auto-applied on a clear pass. That ratification happens downstream, at τ3
(admission ratification gate) and, if ratified, is executed by task 2544. This record
produces evidence and a recommendation only.

## Artifact provenance

The campaign artifacts this record is built from **exist but are gitignored**, and
live only in the main checkout (`/home/leo/src/dark-factory/...`) — not in this
task's worktree, and not committed on any ref:

- Per-cell result JSONs: `orchestrator/src/orchestrator/evals/results/*.json` — that
  directory's `.gitignore` is `*` plus `!.gitignore`, so only the `.gitignore` file
  itself is tracked.
- Three raw campaign dumps under `data/eval-campaign/` — ignored by root
  `.gitignore:9:/data/`:
  - `data/eval-campaign/fable-architect-only-results.json` (07-28 partial, 72 cells)
  - `data/eval-campaign/resume-20260729/reify/fable-architect-only-results.json`
    (phase 1, 24 cells)
  - `data/eval-campaign/resume-20260729/t2430/fable-architect-only-results.json`
    (phase 2, 12 cells)

Because none of this can be committed, this record **transcribes** the numbers below
with explicit provenance for each, rather than linking to committed files, and ships
a reproduction recipe (§8) that regenerates every table from the live artifacts at
their absolute paths.

## How to read this record (INV-2)

Sections 2 through 6 are **RAW OBSERVATION** — recomputed independently from the
per-cell result JSONs (§2, §3, §4), or transcribed with explicit attribution from the
campaign operator's own statistics (§5), or established as a documented absence with
its negative evidence (§6). Section 7 is the **sole INTERPRETATION** section — the
recommendation. Section 8 is a reproduction appendix. This separation is deliberate
and load-bearing: a downstream reader (τ3, task 2544) should be able to trust §2-§6
without trusting this record's judgment, and should be able to find that judgment
confined to one clearly marked place.

1. Provenance (this section)
2. Cell inventory and data hygiene — RAW OBSERVATION
3. Hard-subset OFAT screen — per-architect plan quality and $/plan — RAW OBSERVATION
4. Per-fixture breakdown and the metric-definition reconciliation — RAW OBSERVATION
5. Statistical significance — RAW OBSERVATION
6. End-to-end confirm stage: NOT RUN — RAW OBSERVATION
7. Recommendation — INTERPRETATION
8. Appendix: reproduction recipe

---

## 2. Cell inventory and data hygiene

_RAW OBSERVATION._

The campaign ran in phases, spread over five distinct waves (grouped by result-file
modification date):

| wave | cells (hard-fixture × architect) | description |
|---|---|---|
| 2026-07-20 | 9 | pre-fable, killed campaign (cap contamination; superseded) |
| 2026-07-27 | 24 | 1-trial pilot |
| 2026-07-28 | 72 | full 3-trial screen, all 6 hard fixtures |
| 2026-07-29 | 24 | phase-1 re-run of `reify_task_12` + `reify_task_27` |
| 2026-07-30 | 12 | phase-2 re-run of `df_task_2430_adv_plan` |

**Selection rule: supersede-by-rerun.** Per fixture, take the newest wave carrying a
complete 3-trial cell set for all four architect candidates:

| fixture | scored from wave |
|---|---|
| `df_task_2284_adv_regression` | 2026-07-28 |
| `df_task_2339_adv_verify` | 2026-07-28 |
| `df_task_18` | 2026-07-28 |
| `reify_task_12` | 2026-07-29 |
| `reify_task_27` | 2026-07-29 |
| `df_task_2430_adv_plan` | 2026-07-30 |

This yields **72 unique cells, 18 per candidate, no gaps** — equivalent to the
campaign operator's own de-dup on `(task_id, config_name, trial)` preferring the
latest phase (esc-2862-1 triage note, "PHASE 2 COMPLETE" section). Independently
re-merging the three raw dumps on that same key — later-wave entries overriding
earlier ones — collapses 108 raw entries (72 + 24 + 12) to exactly 72 unique
`(task_id, config_name, trial)` keys, 18 per candidate, confirming the rule reduces to
precisely the intended de-dup with no leftover collisions or gaps (§8 carries the
merge recipe).

**Why the re-runs exist.** The 2026-07-28 wave was ~40% cap-starved: the operator's
triage note reports 37 `blocked` cells in that wave, of which 29 carry `cost_usd ==
0`. Phases 1 and 2 (07-29, 07-30) re-ran exactly the starved fixtures
(`reify_task_12`, `reify_task_27`, `df_task_2430_adv_plan`).

**The one fixture never re-run: `df_task_18`.** All 12 of its 07-28 cells (4
candidates × 3 trials) were inspected individually. Every one carries real cost
($2.42-$7.08), real `plan_steps` (10-28) and a real, non-zero `plan_quality`
(0.65-0.97) — none read as cap-kills. Its 6 `blocked` outcomes of 12
(`architect-opus-max` 0/3 blocked, `architect-fable-high` 1/3, `architect-opus-high`
2/3, `architect-sonnet-high` 3/3) are therefore genuine downstream blocks, not cap
kills, and the fixture did not need re-running.

**`cap_tainted` / `invocation_error` fields (task 3118).** These per-cell fields
exist only on the 2026-07-30 wave, where inspected cells read `cap_tainted: false` /
`invocation_error: null` — the task-3118 fix landed between the 07-28/07-29 waves and
the 07-30 wave. They are absent entirely from cells minted on or before 07-29. Cap
identification for the older waves therefore relies on the cost /  `plan_steps` /
timestamp correlation above, not a structural marker.

---

## 3. Hard-subset OFAT screen — per-architect plan quality and $/plan

_RAW OBSERVATION._

Headline table over the 72 scored cells (§2), recomputed independently from the
per-cell result JSONs:

| architect | planRate | doneRate | meanPQ_all | meanPQ_planned | $total (18 cells) | $/usable-plan |
|---|---|---|---|---|---|---|
| `architect-opus-max` *(incumbent)* | 88.9% | 88.9% | 0.8078 | 0.9087 | $59.08 | $3.693 |
| `architect-opus-high` | 83.3% | 88.9% | 0.7761 | 0.9107 | $42.66 | $2.844 |
| `architect-sonnet-high` | 83.3% | 77.8% | 0.7072 | 0.8487 | $26.26 | $1.750 |
| `architect-fable-high` | 94.4% | 94.4% | 0.8678 | 0.9188 | $75.30 | $4.429 |

This recomputation is independent corroboration, not a transcription: it reproduces
esc-2862-1's resolution table exactly (same wave selection, same 72-cell set), which
is the intended INV-2 cross-check — an independently-arrived-at number that matches
the gate record it consumes rather than an unverified copy of it.

**Metric definitions:**

- `planRate` = fraction of the 18 cells with `plan_steps > 0`.
- `doneRate` = fraction of the 18 cells with `outcome == 'done'`.
- `meanPQ_all` = mean `plan_quality` over all 18 cells; a no-plan cell (`plan_steps ==
  0`) scores 0 in this average.
- `meanPQ_planned` = mean `plan_quality` over only the cells that produced a plan
  (the `planRate` numerator).
- `$total` = sum of `cost_usd` over all 18 cells for that candidate.
- `$/usable-plan` = `$total` divided by the count of *planned* cells only (the
  `planRate` numerator), not by 18. This carries forward the ι precedent's finding
  that `$/fixture` (i.e. dividing by all scored cells) and `$/usable-plan` are **not
  interchangeable** — a candidate that fails to plan on some cells still spends money
  on them, so per-fixture cost understates the real price of a plan you can actually
  use (`plans/architect-effort-adoption-decision-2026-07-28.md` §"Methodological
  finding for future campaigns").

**Why `plan_quality`, not the artifacts' `composite_score` field, is the quality
axis.** Every architect cell inspected — across all three raw dumps and the per-cell
result JSONs — carries `composite_score: 0.0`, regardless of its `plan_quality`. This
is a known reporting defect, not a real 0.0: `run_architect_eval` never sets
`tests_pass` on an `EvalMetrics`, and both `compute_composite` and `blend_composite`
(`orchestrator/src/orchestrator/evals/metrics.py`) hard-gate their return to `0.0`
whenever `tests_pass` is unset. This was first documented in
`plans/eval-architect-effort-verdict-2026-07-27.md` §"Reporting defects found" #1 and
filed as **task 3099**. Every number in the table above was therefore recomputed from
the per-cell JSONs' `plan_quality` / `plan_steps` / `cost_usd` / `outcome` fields, not
read off a composite column.

---

## 4. Per-fixture breakdown and the metric-definition reconciliation

_RAW OBSERVATION._

**Planned/3 per fixture × candidate** (`plan_steps > 0`, the `planRate` numerator):

| fixture | opus-max | opus-high | sonnet-high | fable-high |
|---|---|---|---|---|
| `df_task_2284_adv_regression` | 3/3 | 3/3 | 2/3 | 3/3 |
| `df_task_2339_adv_verify` | 3/3 | 3/3 | 3/3 | 3/3 |
| `df_task_18` | 3/3 | 3/3 | 3/3 | 3/3 |
| `reify_task_12` | 1/3 | 0/3 | 1/3 | 2/3 |
| `reify_task_27` | 3/3 | 3/3 | 3/3 | 3/3 |
| `df_task_2430_adv_plan` | 3/3 | 3/3 | 3/3 | 3/3 |

**Done/3 per fixture × candidate** (`outcome == 'done'`, the `doneRate` numerator):

| fixture | opus-max | opus-high | sonnet-high | fable-high |
|---|---|---|---|---|
| `df_task_2284_adv_regression` | 3/3 | 3/3 | 3/3 | 3/3 |
| `df_task_2339_adv_verify` | 3/3 | 3/3 | 3/3 | 3/3 |
| `df_task_18` | 3/3 | 1/3 | 0/3 | 2/3 |
| `reify_task_12` | 3/3 | 3/3 | 2/3 | 3/3 |
| `reify_task_27` | 1/3 | 3/3 | 3/3 | 3/3 |
| `df_task_2430_adv_plan` | 3/3 | 3/3 | 3/3 | 3/3 |

**The metric-definition reconciliation — load-bearing.** esc-2862-1's resolution and
its own triage note name *different* discriminating fixtures, which reads as a
self-contradiction unless the metric each is using is made explicit. It is not a
contradiction: both are correct, under different metrics.

- Under **planRate**, the discriminator is `reify_task_12` — fable 2/3, opus-max 1/3,
  opus-high 0/3, sonnet 1/3 — and every other fixture is 3/3 for both fable and
  opus-max. This is the metric behind the resolution's leave-one-fixture-out
  argument (§5).
- Under **doneRate**, the discriminators are `df_task_18` (fable 2/3, opus-max 3/3,
  opus-high 1/3, sonnet 0/3) and `reify_task_27` (opus-max 1/3, everyone else 3/3).
  This is the metric behind the triage note's per-candidate `done` table.

Neither account is wrong. This record reports both tables above for exactly that
reason, so a downstream reader does not have to reconcile the resolution against its
own triage note unassisted.

---

## 5. Statistical significance

_RAW OBSERVATION — transcribed with attribution._ The figures in this section are the
campaign operator's own statistics, as recorded in esc-2862-1's resolution
(`resolved_by: eval-campaign-2026-07-27`). They are not re-derived here: the per-cell
JSONs give the raw `plan_quality` / `outcome` values behind them (§3, §4), but the
resampling itself (the paired bootstrap, the permutation draws, the Wilson intervals)
is not reproducible from those files without redoing the operator's procedure, so
this record transcribes rather than silently re-presents them as its own computation.

- **Paired per-fixture CI95** on the mean `plan_quality` difference (fable −
  opus-max): **[-0.0433, +0.1633] — crosses zero.**
- **Permutation test** (label swap within fixture-trial, n=18), two-sided: **p =
  0.127.**
- **Plan-rate Wilson CIs overlap almost completely**: fable [74.2%, 99.0%] vs
  opus-max [67.2%, 96.9%]. The nominal plan-rate "win" is literally one cell (17/18
  vs 16/18).
- **Quality-when-planned is a tie**: 0.9188 vs 0.9087, with overlapping CIs.
- **Leave-one-fixture-out**: dropping `reify_task_12` collapses the paired diff from
  +0.060 to **+0.0073**; dropping any other single fixture leaves it at
  +0.069..+0.072. The entire nominal effect rides on one high-variance fixture.
- **Cross-campaign observation**: pooled over both this campaign and the earlier
  22-fixture run (`plans/eval-architect-effort-verdict-2026-07-27.md`),
  `reify_task_12` is 2/4 for fable and 2/4 for opus-max — it is high-variance for
  *every* candidate, not a fable-specific weakness, and whichever candidate wins its
  coin flip in a given run wins the aggregate comparison.

---

## 6. End-to-end confirm stage: NOT RUN

_RAW OBSERVATION._ The PRD's τ1 recipe conditionally calls for an `eval-confirm`
stage (end-to-end, both architect and implementer live, ≥3 trials) whenever an
architect other than production tops the screen. **This did not run.** That absence
is recorded here as an absence, with its negative evidence, rather than by
fabricating an end-to-end table.

**The evidence.** `run_end_to_end` stamps every end-to-end cell's config name as the
architect+implementer pair, joined with `+`:
`config_name = f'{arch_config.name}+{impl_config.name}'`
(`orchestrator/src/orchestrator/evals/runner.py:840`). `save_result` then writes each
cell to `{result.task_id}__{result.config_name}__{result.run_id}.json`
(`runner.py:1310`). A confirm cell would therefore necessarily appear as a
`+`-containing filename under `evals/results/` — and **no such file exists**: a scan
of the results directory finds zero filenames containing `+`.

The only non-architect config names written to that directory since 2026-07-26 are
`claude-sonnet-max`, `claude-opus-max`, `claude-opus-high`, `judge-sonnet` and
`judge-haiku` (7 cells each) — the OFAT screen's own ride-along implementer/judge
baseline cells (PRD decision 2), not a confirm batch.

**The operator's stated reason (esc-2862-1).** The confirm stage reuses the same 6
fixtures × 3 trials as the screen, so it would inherit the identical
high-variance-fixture problem documented in §5 rather than resolving it; and it
measures end-to-end $/done, not architect plan quality, so even a clean run would not
directly settle the architect question this record answers. On that basis the
operator judged it would not be a good use of spend and did not launch it.

**The honest wrinkle.** The recipe's step-4 trigger condition *did* fire —
`architect-fable-high` topped the screen on both `planRate` and `doneRate` (§3) — so
the confirm was skipped by operator judgement, not because the gate's own trigger
went unmet. The triage note explicitly surfaced this as a spend decision and asked
Leo to authorise the confirm run; the resolution then closed the gate without
launching it (see esc-2862-1's triage note, "WHY THE GATE STILL DOES NOT CLOSE" and
its resolution's disposition).

**Consequence for downstream consumers.** End-to-end composite, $/done, and their
CI95 for the fable-vs-incumbent confirm pair are **unavailable**. The recommendation
in §7 rests on the screen (§3-§5) alone — it does not, and cannot, cite an end-to-end
number.
