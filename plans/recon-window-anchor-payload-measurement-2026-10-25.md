# Stage 1 "new" payload growth after the run-start watermark anchor: the measurement

Task 6077 architect, 2026-10-25 ~23:00Z, read-only (main-checkout data and the
live services). Recorded as measured, not re-run: the corpus moves every cycle.

## What was measured

Task 4465 (`acb35a765b`) anchors `Watermark.last_episode_timestamp` and
`last_memory_timestamp` at `run.started_at` instead of cycle completion
(`fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness.run_full_cycle`).
The next cycle's "new" lists are cut at that watermark by
`fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py::MemoryConsolidator._filter_records_newer_than_watermark`.
The question: how much do the "New Mem0 Memories" and "New Episodes" lists
grow, and does Stage 1 re-consolidate its predecessor's output?

## Context and premise

`fused-memory.service` was active since 2026-10-25T16:22:15Z, long after
`acb35a765b` merged on 2026-09-30. Every archived Stage 1 payload renders its
"Mem0 memory freshness cutoff" equal to the PREVIOUS run's `started_at` (run
`e08a247d`'s cutoff is 15:18:59.448254, run `22b24b34`'s start). The live
watermarks row agrees: dark_factory's `last_memory_timestamp` equals run
`bb053cf1`'s `started_at`, 21:20:57.813705. The anchor is live.

## Sources

- `data/orchestrator/agent-transcripts/recon-<run_id>/<enc>/<session>.jsonl`:
  the first `user` message is the full Stage 1 payload. The archive starts
  2026-10-22, so no pre-merge payload exists and the baseline is re-rendered.
- `data/reconciliation/reconciliation.db`: `runs` (`started_at`,
  `completed_at`, `stage_reports`) and `watermarks`.
- Qdrant `fused_<project>`: payload `created_at`, `source`, `agent_id`, `kind`
  (scrolled with `data` and `memory` excluded).

## Results

### (1) Post-merge counts, as rendered

183 distinct full runs across 9 projects, 2026-10-22 to 2026-10-25, retried
sessions deduped. New episodes = 0 in all 183 (see "What could not be
measured").

| project | new Mem0 per payload {count: runs} |
|---|---|
| dark_factory (27 runs) | {0: 24, 1: 2, 2: 1} |
| autopilot_video | {0: 16, 1: 2} |
| know_live | {0: 16, 1: 2} |
| reify | {0: 16, 1: 2, 2: 1} |
| pump_web_ui | {3: 17, 4: 1} |
| solar_challenge | {0: 3, 1: 8, 2: 6, 3: 1, 4: 1} |
| solar_challenge_platform | {0: 5, 1: 9, 2: 3, 3: 2} |
| scoping_intake | 3 to 19, median about 8 |
| stream_touch | 1 to 65 |

### (2) Baseline, re-rendered from the live dark_factory corpus

For consecutive completed full runs (P, R): old window (P.completed_at,
R.started_at], new window (P.started_at, R.started_at]. 220 post-merge pairs, P
running 15 to 23 minutes. Surviving records per cycle: old window mean 16.4,
median 15; the anchor adds mean 1.6, median 1, i.e. +9.9%.

The 445 surviving anchor-added records over all 251 pairs since 2026-09-25, by
class (5 were not attributed to a class in the architect's record):

| class | records |
|---|---|
| external agent writes (the skip hole 4465 closes) | 278 |
| write-triage amendments | 103 |
| `orchestrator_completion` | 41 |
| sightings | 7 |
| reconciliation-sourced: 6 `recon-stage-memory_consolidator`, 2 `recon-stage-task_knowledge_sync`, 2 `cycle_summary`, 1 `stage2_suppress_guard` | 11 (2.5%) |

Survivorship caveat: every cycle writes 3 self-output records mid-cycle (the
Stage 1 and Stage 2 `cycle_summary` mirrors, Stage 2's `task_count_snapshot`),
which the next cycle's pool-cap trim and snapshot self-prune hard-delete. An
UNCAPPED view at fetch time would see ~4.6 added per cycle, ~3 of them
self-output. Verified on pump_web_ui run `b5cca0a4` (15:42:28 to 15:52:52): its
mirrors were created at 15:44:27 and 15:49:23, which the old anchor hid.

### (3) Re-consolidation

In corpora below the 250-record fetch cap, every cycle's "new" list carries its
predecessor's self-output. pump_web_ui (146 points) showed 63 self-output lines
of 64 across 18 runs: typically "Stage memory_consolidator cycle summary for
run_id: <prev>", "Stage task_knowledge_sync cycle summary ...", and one
empty-body line (the `task_count_snapshot` after `strip_snapshot_lines`).

- All projects together displayed 167 self-output lines. Stage 1 tool calls
  naming any of those 167 ids, read or write: **0**.
- Only 3 write calls named any displayed new record at all, all legitimate
  consolidation of external near-duplicate preferences in stream_touch.
- Payload cost: self-output lines are **0.445%** of Stage 1 user-payload
  characters across all 183 runs. Worst case pump_web_ui, about 550 chars
  (about 140 tokens) of a 5.3k-char payload, plus the large system prompt.
  dark_factory: 0.
- Benefit case: in dark_factory exactly 1 displayed record across 27 runs fell
  in the anchor-added window: `013bd19d`, a `claude-task-7480-implementer`
  gotcha created at 15:33:18 inside run `22b24b34` (15:18:59 to 15:34:32), an
  external write the old anchor would have silently skipped.

## What could not be measured

Episode growth. New episodes were 0 in all 183 payloads:
`stats.stage1_fetch_degraded == ['episodes']` on 215 of 238 completed full runs
since 2026-10-22 (every project except pump_web_ui), and a live
`get_episodes(dark_factory, last_n=5)` returned "Graphiti retrieve_episodes
timed out after 30.0s". Stage 1 receives no episodes at all. Pre-existing:
ticket `tkt_0RW7MMT8MBH2F97QHZ448NKJGR` and escalation `esc-6077-1` cover the
cause, task 6084 the cursor consequence.

dark_factory's Mem0 counts are a sample, not the window: Stage 1 fetches
`mem0.get_all(limit=memory_limit=250)` (sonnet tier) in point-id order; with
33,767 points the "new" list is a 0.74% sample of the window (task 7492).

## Decision

Immaterial. No source-metadata filter is added to the "new" lists, and the
anchor stays at `run.started_at`. Moving it back to completion would reopen the
external-write skip hole: 278 of the 445 anchor-added records were external.

## Re-measure when

- Task 7492 lands. An uncapped, newest-first fetch would show every project its
  predecessor's ~3 self-output records per cycle: still about 550 chars.
- Stage 1 starts issuing writes against `cycle_summary` or
  `task_count_snapshot` records.

If either makes the churn material, the remedy stays the one named in
`fused-memory/tests/reconciliation/test_recon_window_anchor.py`'s module
docstring: drop reconciliation-sourced records from the "new" list by source
metadata, in `memory_consolidator.py`.
