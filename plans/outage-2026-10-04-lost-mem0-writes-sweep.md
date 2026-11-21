# 2026-10-04 OpenAI credit outage: lost mem0 writes sweep (task 7145)

Source: `data/reconciliation/write_journal.db`, table `write_ops`, opened read-only.
`prune_write_ops` had NOT pruned the window (rows span 2026-04-07 onward).

## Window and match

- First 429 row: 2026-10-03T23:23:19Z. Last 429 row: 2026-10-04T09:22:36Z. (The window opens
  before the 2026-10-04T00:00Z start the task suggested: 19 rows are on 2026-10-03.)
- Matched: **244** rows (206 `add_memory`, 38 `add_system_record`), every one `success=0`,
  `result_summary` stores `[]`, error `429 ... credit_balance_exhausted`.
- By project: solar_challenge 74, dark_factory 71, reify 41, solar_challenge_platform 40,
  pump_web_ui 6, autopilot_video 6, know_live 6.

## Recoverability

The journal keeps `params.content` capped at 200 characters. A row with content under 200
characters is complete; a row at exactly 200 is a digest.

| Outcome | Rows |
|---|---|
| Matched | 244 |
| Full content in journal | 69 (67 distinct texts; two snapshot texts were journaled twice) |
| Already present in Mem0 (exact and loose-needle check of every `fused_<project>` Qdrant collection, read-only) | 0 |
| Re-added from journal | 67 |
| Re-added from escalation esc-375-2 (digest-only row of `claude-task-375-implementer`) | 1 |
| Unrecoverable (digest only, 200-char prefix) | 174 |

The 67 + 1 re-adds were submitted with `add_memory` under their original project and category,
with metadata `x_readded_from_write_op=<first 8 chars of the write_ops id>` (`esc-375-2` for
the escalation one) and `x_outage=2026-10-04`. The esc-375-2 text was re-checked against
solar-challenge main (`COLOUR_PALETTE`, `_with_alpha`, `_FIGURES`, `monkeypatch.setitem` all
still present) and no existing memory covers it; task 829's convention refines where the palette
is defined but does not replace it.

## Caveat: the re-adds are queued, not yet landed

On 2026-11-21 OpenAI credits were exhausted again, so every re-add returned
`stores_pending_retry=["mem0"]` and sits in `mem0_intents` with status `retrying` (68 rows
whose metadata carries `x_readded_from_write_op`). They land when credits return. Do not
resubmit them. To confirm:

```sql
select status, count(*) from mem0_intents
where metadata like '%x_readded_from_write_op%' group by 1;
```

Done means `completed` = 68. A `dead` row needs its text re-added by hand from this journal.

## Unrecoverable rows (digest only), by project, category and agent_id

The first 200 characters of each are still in `write_ops.params`. Other possible sources
(not searched): the task agents' transcripts under `data/orchestrator/agent-transcripts/`.


This list has 175 rows. One of them, the `claude-task-375-implementer` row, was recovered from esc-375-2 and re-added, which is why the table above says 174.
