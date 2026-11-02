# Reconciliation evidence caps: 2026-10 measurement against the 1M window (task 5499)

The baseline the next re-evaluation diffs against. Every reconciliation evidence cap
in scope is a named constant whose clips are disclosed through
`fused_memory/utils/clip_disclosure.py` (task 5365). This record says what each
value is based on, including the caps left unchanged.

All measurements are read-only and were taken on 2026-10-25 at base `bfb65b4c39`.

---

## 1. Data sources

| Source | What it gave | How it was read |
|---|---|---|
| `.taskmaster/tasks/tasks.db` (main checkout) | task description lengths (M1) | `SELECT length(description) FROM tasks` over all 7,453 rows, and again over the latest 2,000 by id |
| Qdrant `fused_<project>` collections | Mem0 body lengths (M2) | scroll, `len(payload['data'])` per point |
| `data/reconciliation/reconciliation.db` `runs` | backlog chunk sizes (M3) and Stage-1 wall clock (M8) | `trigger_reason` is `backlog_chunk:<N>:<events>`. A chunk followed by chunk N+1 for the same project stopped on `token_budget`, because the 1,000-event peek was not exhausted. A chunk of exactly 1,000 events stopped on the peek. |
| `data/orchestrator/agent-transcripts/recon-*/` | context window use (M4) and cap firing (M5) | the first user message is the stage payload. Per-turn `usage` gives the context size. `…[+N chars elided]` markers and `_NOTE:` lines show which disclosed cap fired. |
| `runs.stage_reports` | raw-text fallback frequency (M6) | `stats.raw_text_fallback` per stage report |
| `event_buffer` | buffered event mix (M7) | `event_type` counts |

The direct record of budget consumption is the `Context assembly complete … N estimated
tokens` log line. It exists only in the 29.3G user journal, and `journalctl` grep timed
out twice. That is why consumption had to be inferred from successor chunks in M3, and
why this task adds the durable stat in section 5.

## 2. Measurements

**M1. Task descriptions (chars).**

| set | n | p50 | p90 | p95 | max | > 300 |
|---|---|---|---|---|---|---|
| all tasks | 7,453 | 1,832 | 3,796 | 4,821 | 26,294 | 95.3% |

Latest 2,000 tasks: mean 2,198. At a 4,000 cap, 7.0% clip and the mean rendered
length is 2,052, against 300 today.

**M2. Mem0 bodies (chars of payload `data`).**

| project | n | p50 | p90 | > 500 | > 2,000 | mean rendered at 500 | mean rendered at 2,000 |
|---|---|---|---|---|---|---|---|
| dark_factory | 33,724 | 681 | 1,977 | 60.4% | 9.7% | 360 | 763 |
| reify | 38,876 | 606 | 1,494 | 55.6% | 4.7% | 340 | 654 |
| solar_challenge | 4,378 | 811 | — | 78.1% | 14.7% | 453 | 939 |

**M3. Budget consumption on the backlog path.** 318 `backlog_chunk` runs,
2026-04-08 to 2026-10-18. 80 were followed by a successor chunk, so the budget stopped
them. All 80 were under 1,000 events, with a median of **268**. On solar_challenge, 37 of
67 chunks were exactly 1,000 events, so the peek bound them and the budget did not.

**M4. Context window.**

| model | sample | observation |
|---|---|---|
| claude-sonnet-5-5 | all 486 archived recon stage transcripts (10-22..25) | Stage-1 first turn p50 84.6k, max 100.8k tokens, of which about 74k is fixed CLI, system and tool overhead. Stage-2 peak max 149k. |
| claude-sonnet-5-5 | all 896 archived sonnet transcripts | none above 200k |
| claude-opus-5-5 | 400 most recent CLI sessions | max 571,612 tokens; 46 above 200k |

**M5. Disclosed-cap firing** in 198 archived post-5365 Stage-1 payloads and 167 Stage-2
payloads.

| cap | fired |
|---|---|
| `_RECORD_BODY_CHARS` (elision markers) | 392 markers in 75 S1 payloads |
| `episode_limit` fetch note | 0 of 198 |
| `memory_limit` fetch note | 171 of 198 |
| `MAX_ACTIVE_TASKS_RENDERED` / 50k clamp | 0 |
| `MAX_DONE_TASKS_RETAINED` window | 147 (S2) |
| `MAX_CANCELLED_TASKS_RETAINED` window | 106 (S1) + 95 (S2) |

At `_RECORD_BODY_CHARS` = 2,000, 269 of the 392 clipped items render whole. The largest
observed S1 payload grows from 75,736 to 101,339 chars (about +6.4k tokens), so the first
turn stays near 107k tokens.

**M6. `_extract_report` raw-text fallback.** 0 of 2,836 post-5365 stage reports carry
`raw_text_fallback`, because `stages/base.py::BaseStage.run` replaces the report with the
`recon_report_state`-assembled one whenever RRS is configured.

**M7. Cost model.** The buffered mix (433 events) is 56% task events, 36% `memory_added`
and 7% `memory_deleted`. Each event brings 0.15 unique-task context items. The per-event
cost today is 65,000 / 268 ≈ 243 estimated tokens. At the raised caps it is ≈ 468
(×1.93). A mean-preserving synthetic chunk of 268 events costs 114,265 estimated tokens
at the raised caps, against a modelled ≈ 116.5k. The analytic estimate is ≈ 127k. This is
the fixture in
`fused-memory/tests/test_context_assembler.py::test_shipped_budget_holds_a_measured_backlog_chunk`.

**M8. Backlog chunk wall clock.** Stage-1 p50 is 240s at 150-300 events and 889s at
500-1,000 events. The max is 4,082s, against `stage_timeout_seconds` 3600. Wall clock
bounds breadth; the window does not.

## 3. Decisions per cap

| cap | old | new | reason |
|---|---|---|---|
| `context_assembler.py::_TASK_DESC_CHARS` | 300 | **4,000** | 95% of descriptions clip at 300. At 4,000, 7.0% clip (just above p90 3,796). |
| `context_assembler.py::_MEMORY_BODY_CHARS` | 500 | **2,000** | 55-78% of bodies clip at 500. At 2,000, 9.7% / 4.7% / 14.7% clip. |
| `stages/memory_consolidator.py::_RECORD_BODY_CHARS` | 500 | **2,000** | 269 of 392 observed clips render whole. The S1 first turn stays near 107k, under the 200k this sonnet-capable path respects. |
| `ReconciliationConfig.token_budget` (schema and `config.yaml`) | 65,000 | **130,000** | Per-event cost ×1.93 at the raised caps. ×2 holds about 268 events per chunk, so breadth and drain throughput are unchanged and the window buys depth. The opus first turn is about 74k + 130k estimated tokens. |
| `cli_stage_runner.py::_FALLBACK_SUMMARY_CHARS` | 2,000 | unchanged | The path never fires (M6), so there is nothing to size against. |
| `sonnet_episode_limit` / `sonnet_memory_limit` | 125 / 250 | unchanged | The episode fetch note fired 0 times. The memory note fires because mem0 `get_all` scrolls in point-id order, so the section is an unordered sample and no count makes it complete. A follow-up is filed for that. |
| `opus_episode_limit` / `opus_memory_limit` | 500 / 1,000 | unchanged | Same as the sonnet limits. |
| `task_filter.py::MAX_DONE_TASKS_RETAINED` | 30 | unchanged | A recency window by design (`window_note`). Any finite value fires against 5,126 done tasks. |
| `task_filter.py::MAX_CANCELLED_TASKS_RETAINED` | 15 | unchanged | Same as `MAX_DONE_TASKS_RETAINED`. |
| `task_filter.py::MAX_ACTIVE_TASKS_RENDERED` | 50 | unchanged | Never fired in 365 S1/S2 payloads. |
| `task_filter.py` `max_chars` clamp | 50,000 | unchanged | Never fired in 365 S1/S2 payloads. |

`_MEMORY_BODY_CHARS` and `_RECORD_BODY_CHARS` share a value but stay two constants. The
first spends `token_budget` on the opus-only backlog path. The second has no competing
budget and is bounded by the sonnet-tier 200k window. They can diverge.

## 4. The two window facts

- **Backlog path (`token_budget`, `_TASK_DESC_CHARS`, `_MEMORY_BODY_CHARS`).**
  `ContextAssembler` is used only by `harness.py::BacklogIterator.run`, which always runs
  on the opus tier. opus-5-5 CLI sessions were observed at 571,612 tokens, so this path
  has the 1M window.
- **Time-windowed path (`_RECORD_BODY_CHARS`, episode and memory limits).** It can run
  on sonnet. No sonnet-5-5 session was seen above 149k, so nothing shows that the CLI
  gives sonnet more than 200k. This path stays designed under 200k.

## 5. Re-evaluation recipe

After the next backlog drain, read the budget stat this task adds:

```sql
SELECT project_id, trigger_reason,
       json_extract(stage_reports, '$.memory_consolidator.stats.stage1_assembled_budget')
FROM runs
WHERE trigger_reason LIKE 'backlog_chunk:%'
ORDER BY started_at DESC;
```

The value is `{estimated_tokens, token_budget, events, context_items, events_deferred}`.
Off the backlog path it is `null`. `events_deferred > 0` marks a chunk the budget
stopped. Confirm that events per budget-bound chunk stay near 268 at `token_budget`
130000. If they fall, the per-item caps are spending the breadth the budget was sized to
hold.

`token_budget` is not in `config/reload.py::RELOADABLE_FIELDS`, so changing it needs a
fused-memory restart.
