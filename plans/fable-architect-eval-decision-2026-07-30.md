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
