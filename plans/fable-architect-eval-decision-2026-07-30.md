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
