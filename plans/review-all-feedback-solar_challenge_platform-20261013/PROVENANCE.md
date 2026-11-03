# Provenance of the backing data

Package for `plans/review-all-feedback-solar_challenge_platform-20261013.md`. Assembled 2026-10-16 by the resumed lead session (`13d7dc93-4e8e-49fe-af2d-6ee94d9207ea`, cwd `/home/leo/src/solar-challenge-platform`). Only ephemeral items were copied; durable records are pointed at.

## Durable records (pointers, not copied)

| Record | Where |
|---|---|
| The review-all run report, findings record and metrics snapshot | solar-challenge-platform `plans/review-all-solar_challenge_platform-20261013.{md,json}`, `plans/quality-metrics/review-all-solar_challenge_platform-20261013.json` (commit e4f2a2b) |
| The program doc (26 streams, rulings, author-pass amendments, FILED table and the dated ledger trailer) | solar-challenge-platform `plans/review-all-program-solar_challenge_platform-2026-10-14.md` — generated; every commit from 207d800 to ee22615 is in `metrics/program-commits.txt` |
| The 26 stream PRDs and stamped capability-manifest sidecars | solar-challenge-platform `docs/prds/review-all-*.md` / `*.capability-manifest.yaml` |
| The dead lead's transcript and its 48 subagent transcripts | `~/.claude/projects/-home-leo-src-solar-challenge-platform/b1f15ebe-9d89-4f08-b6a2-b193192cc210.jsonl` and `…/b1f15ebe-…/subagents/agent-*.jsonl` |
| The resumed lead's transcript and its 15 subagent transcripts | `~/.claude/projects/-home-leo-src-solar-challenge-platform/13d7dc93-4e8e-49fe-af2d-6ee94d9207ea.jsonl` and `…/13d7dc93-…/subagents/agent-*.jsonl` |
| Author-seat spawn briefs (2026-10-14) and ruling-session briefs (2026-10-15) | `~/.claude/spawn-briefs/review-all-scp-2026-10-14/`, `~/.claude/spawn-briefs/review-all-scp-2026-10-15/` |
| Project memory notes written by the lead sessions (`review-all-20261013-phase6-resume-state.md`, `s4-/s6-/s13-/s15-*-prd-authored.md`, `decompose-filing-gotchas.md`, `briefing-yaml-tracking-gotcha.md`, `deferred-actions-as-deterministic-tasks.md`) | `~/.claude/projects/-home-leo-src-solar-challenge-platform/memory/` |
| fused-memory records: the run's `observations_and_summaries` entries (10-14 and 10-15) and the `decisions_and_rationale` entries for every ruling | fused-memory, `project_id = solar_challenge_platform`, agent ids `review-all-review-all-solar_challenge_platform-20261013*` |
| Task store state (tasks 1065–1305, cancellations with `x_acceptance_reason`, edges, notes) | fused-memory task store for `/home/leo/src/solar-challenge-platform` |
| Phase 3–4 workflow inputs and result (`args.json` 1.0 MB, `args-slim.json`, `workflow.js`, `workflow-result.json` 958 KB, `snapshot.json`, `prior_open.json`) | the dead session's scratch dir (below); NOT copied — not the subject of this feedback and large; copy before `/tmp` is cleared if phases 3–4 are to be studied |

## Ephemeral items (copied)

Source A — the dead lead's scratch dir, `/tmp/claude-1000/-home-leo-src-solar-challenge-platform/b1f15ebe-9d89-4f08-b6a2-b193192cc210/scratchpad/review-all/review-all-solar_challenge_platform-20261013/` (lost on reboot or `/tmp` clean):

- `program-state/gen_program_doc.py` — the hand-rolled program-doc generator (SPOT for the program doc; "the lead edits nothing by hand after"). Edited by both leads; the trailer paragraph is a hard-coded string in it.
- `program-state/gen_prd_briefs.py`, `stream_items.json` — the author-seat brief generator and its stream list.
- `program-state/rulings.json` — the 20 deliberation rulings (R15 wording amended 2026-10-16).
- `program-state/filings.json` — Phase 5 mechanical filings.
- `program-state/filed/S*.json` — one FILED record per decomposed stream (26 files; S2/S7/S11/S12/S15/S16 and the S1/S4/S10/S13 records were written by the resumed lead from the seat returns).
- `program-state/prd-results/S*.json` — one author-pass amendment record per stream (lead rulings on each author's open questions).
- `lead-actions/lead-actions-batch-3..19.md` — the clerk action files (batches 1–2 were not written as files; 3–9 by the dead lead, 10–19 by the resumed lead).
- `briefs/prd-seat-briefs/S*.md` — the per-stream seat briefs the author/decompose seats read.

Source B — the resumed lead's scratch dir, `/tmp/claude-1000/-home-leo-src-solar-challenge-platform/13d7dc93-4e8e-49fe-af2d-6ee94d9207ea/scratchpad/`:

- `seat-returns/decompose/seat-DECOMPOSE-*.txt` — the final JSON return of every author and decompose seat of the dead session, extracted from its subagent transcripts (`seat-DECOMPOSE-S27-a6a4f99d` etc. are AUTHOR seats despite the prefix; the extractor keyed on the `"stream"` field). `S4-S13-S10-S1-returns-pretty.txt` is the pretty-printed form the resumed lead read.
- `seat-returns/clerks/seat-LEADACTIONS-*.txt` — the dead session's clerk returns.
- `briefs/resume-session/brief-*.md` — the dead lead's decompose-seat and clerk prompts (verbatim, extracted from its transcript) that the resumed lead reused as templates; `spawn-S2-cmd.sh` the verbatim author-spawn command; `filed-ids.md` the label→id reference handed to the six decompose seats.

Source C — the resumed session's persisted tool results, `~/.claude/projects/-home-leo-src-solar-challenge-platform/13d7dc93-4e8e-49fe-af2d-6ee94d9207ea/tool-results/`:

- `seat-returns/author-handbacks/S2-S7-S11-S12-S15-S16-handbacks-as-read-2026-10-15.txt` — the six spawn-mode author hand-backs (`result.md`) exactly as the resumed lead read them on 2026-10-15 15:00. **Their originals under `~/.claude/fleet/sessions/prd-solar_challenge_platform-*/result.md` had been reaped by 2026-10-16**; two ruling sessions reported them missing. The 10-14 hand-backs for S1/S3/S4/S5/S6/S10/S11/S13 survive only inside the dead lead's transcript.

Source D — fleet session dirs `~/.claude/fleet/sessions/rule-solar_challenge_platform-*/result.md` (reaped on a schedule):

- `seat-returns/ruling-sessions/*.result.md` — the four 2026-10-16 ruling sessions' hand-backs (S15 Q3, S16 Q1, 582-vs-617, 382).

## Derived data

- `metrics/seats.csv` — one row per subagent of both lead sessions (63 rows): kind, stream, model, effort, turns, input/output/cache tokens, first/last timestamp, `degenerate` flag. `approx_subagent_tokens = input + output + cache_create`; cache reads are listed separately because they dominate the raw totals. Derived from the subagent transcripts' `usage` blocks by `metrics` extraction in the resumed session (script inline in its transcript, 2026-10-16).
- `metrics/seats-summary.md` — the same, aggregated by session × kind × model.
- `metrics/lead-context-dead-lead-b1f15ebe.csv`, `metrics/lead-context-resume-lead-13d7dc93.csv` — per assistant turn, the lead's context size (input + cache read + cache create) and output tokens: the curve that ends in the 2026-10-14 context-limit death.
- `metrics/program-commits.txt` — the git log of every program-doc / PRD / sidecar commit in solar-challenge-platform from e4f2a2b to ee22615 with timestamps (the Phase 5–6 timeline).
