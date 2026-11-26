# Write-triage judge field cap: 1,200 → 4,000 chars (task 6076, 2026-10-03)

This record is written by hand. Nothing regenerates it. Its call-cost numbers are copied from
`calibration/write_triage_judge_call_cost.json` (`generated_at` 2026-10-03T10:41:28+00:00), produced by
`scripts/measure_write_triage_judge_call.py`. The recall and placement rows are quoted from two studies whose
bundles live under the main checkout's gitignored `data/` directory. They are not re-measured here.

## Decision

The judge's per-field character cap moves from a hard-coded 1,200 to the hot-reloadable leaf
`write_triage.judge_field_chars`, default 4,000. The cap applies to the new entry and to every candidate, and a
cut field is still marked `…[elided]`.

## Recall evidence (relation task, the deciding evidence)

Source: the Jev trial, `data/write-triage-jev-trial-2026-09-29/results/third_run.md` (pre-registered secondary)
and `followup.md`. It used the gpt-4o-mini production judge on a slate of one, with the target fixed.

| population | `distinct` on rater-confirmed belonging links at 1,200 | at 4,000 | discordant | p |
|---|---|---|---|---|
| 219 fresh dark_factory hand links (third run) | 50/211 (24%) | 23/211 (11%) | 27 vs 0 | < 10⁻⁷ |
| 140 hand links, round 2 (followup) | 27/132 (20%) | 11/132 (8%) | — | — |

At 4,000 chars the judge handles contradictions no differently (4 vs 4). The cause is the length of
dark_factory notes: 19 of 20 dark_factory corrections in round 2 involve a note over 1,200 chars, so the
relevant text was cut off.

## Placement (slate task): exploratory, NOT improved

Source: `data/write-triage-stronger-models-2026-09-30/results/report.md`, "Slate task" table. It covers the 84
frozen reify fixture cases with the gpt-4o-mini production wording and single-rater truth.

| cap | width | misfiled | duplicates placed (of 75) |
|---|---|---|---|
| 1,200 (frozen) | 5 | 6/77 | 65 |
| 1,200 (rerun) | 5 | 5/77 | 66 |
| 4,000 | 5 | 8/77 | 62 |
| 1,200 (frozen) | 20 | 4/80 | 68 |
| 1,200 (rerun) | 20 | 4/81 | 69 |
| 4,000 | 20 | 9/80 | 62 |

At 4,000 chars the judge places slightly worse on these slates (62 vs 65–66 duplicates, paired p 0.07–0.25).
These slate tests had no pre-registered alpha and are descriptive. The confirmed gain is on the fixed-target
relation task above, not on placement.

## Call cost at 4,000 chars (measured)

Measured with gpt-4o-mini (`judge_reasoning_effort` null, so temperature 0) through the shipped `judge_write`, 20
sequential calls per width. The slates were worst case: every field longer than the cap, built from the
calibration fixture's prose, with 36-char uuid ids. Input tokens are the provider's own count and include the
system instructions. Seconds are wall time around the whole `judge_write` call. No call errored.

| width | prompt chars | input tokens (min / median / max) | seconds p50 / p95 / max | errors |
|---|---|---|---|---|
| 5 | 27,142 | 7,071 / 7,071 / 7,071 | 1.45 / 2.97 / 2.98 | 0 |
| 10 | 47,447 | 12,349 / 12,349 / 12,349 | 1.39 / 2.50 / 2.93 | 0 |
| 20 | 88,057 | 23,024 / 23,024 / 23,024 | 1.72 / 4.67 / 5.27 | 0 |

At gpt-4o-mini's list price ($0.15 per 1M input tokens), the worst-case input costs about $0.0011, $0.0019 and
$0.0035 per call at widths 5, 10 and 20. Real slates mostly carry fields shorter than the cap, so they cost less.

## Timeout verdict

The measured width-20 p95 is 4.67 s.

| bound | verdict |
|---|---|
| shipped `judge_timeout_seconds`, 15 s (raised from 10 s by task 6148) | PASS |
| 10 s, the figure this task originally named | PASS |
| half the shipped timeout, 7.5 s | PASS |

The timeout default is unchanged here.
