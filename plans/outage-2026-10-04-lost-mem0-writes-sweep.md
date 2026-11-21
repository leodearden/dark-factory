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

| project | category | agent_id | rows | write_ops id prefixes |
|---|---|---|---|---|
| dark_factory | observations_and_summaries | (null) | 25 | b0b3a6ff 8d0325b4 4207be7c bd2372d5 30c29d86 7b6b98c0 7125e801 fc91055d 02aec74a 810593bc e3d37968 b0fc916c 56d1ddd9 b8d248fb 20042999 4b015114 409a9b89 a4781408 3fc238a7 e116a642 9ee2c486 5ee471ec 60519aad 7b0b7219 a63e775e |
| dark_factory | observations_and_summaries | claude-task-3015-architect | 1 | 9f535bc2 |
| dark_factory | observations_and_summaries | claude-task-3235-architect | 1 | e90a5fd1 |
| dark_factory | observations_and_summaries | claude-task-5321-architect | 1 | 03ce087f |
| dark_factory | observations_and_summaries | claude-task-6246-architect | 1 | c5a9e8f3 |
| dark_factory | observations_and_summaries | claude-task-6247-architect | 1 | be08cd07 |
| dark_factory | observations_and_summaries | claude-task-6286-architect | 1 | 877f0f58 |
| dark_factory | observations_and_summaries | claude-task-6293-architect | 1 | a0eaadaa |
| dark_factory | observations_and_summaries | orchestrator-task-3096 | 1 | 89d6b8c8 |
| dark_factory | observations_and_summaries | orchestrator-task-4045 | 1 | 2e6b74e7 |
| dark_factory | observations_and_summaries | orchestrator-task-4094 | 1 | 542d8461 |
| dark_factory | observations_and_summaries | orchestrator-task-4155 | 1 | 43a41050 |
| dark_factory | observations_and_summaries | orchestrator-task-4529 | 1 | 816b8009 |
| dark_factory | observations_and_summaries | orchestrator-task-5074 | 1 | a6e29d6d |
| dark_factory | observations_and_summaries | orchestrator-task-5152 | 1 | ce0fb917 |
| dark_factory | observations_and_summaries | orchestrator-task-5163 | 1 | 38fe68da |
| dark_factory | observations_and_summaries | orchestrator-task-5269 | 1 | c2b593f9 |
| dark_factory | observations_and_summaries | orchestrator-task-5304 | 1 | e2b3208a |
| dark_factory | observations_and_summaries | orchestrator-task-5328 | 1 | 4e673d87 |
| dark_factory | observations_and_summaries | orchestrator-task-5338 | 1 | ca02f977 |
| dark_factory | observations_and_summaries | orchestrator-task-5339 | 1 | 5ef2456b |
| dark_factory | observations_and_summaries | orchestrator-task-5346 | 1 | df143ceb |
| dark_factory | observations_and_summaries | orchestrator-task-5492 | 1 | 7e8e39cf |
| dark_factory | observations_and_summaries | orchestrator-task-6076 | 1 | 167cd6a1 |
| dark_factory | observations_and_summaries | orchestrator-task-6150 | 1 | 42c1ed14 |
| dark_factory | procedural_knowledge | claude-task-3296-architect | 1 | 033c5117 |
| dark_factory | procedural_knowledge | claude-task-4452-debugger | 1 | 42982444 |
| dark_factory | procedural_knowledge | claude-task-5335-implementer | 1 | 5876e160 |
| dark_factory | procedural_knowledge | claude-task-6181-architect | 1 | b992a822 |
| dark_factory | procedural_knowledge | claude-task-6193-architect | 1 | 5e6e6d67 |
| dark_factory | procedural_knowledge | claude-task-6291-architect | 1 | f1ccab1f |
| reify | observations_and_summaries | (null) | 6 | 41bb37aa 79bfde70 f96e0da9 a230ca77 db72fe30 65810353 |
| reify | observations_and_summaries | claude-task-6347-implementer | 1 | 305f0161 |
| reify | observations_and_summaries | claude-task-6504-architect | 1 | 9c01f113 |
| reify | observations_and_summaries | claude-task-6504-implementer | 1 | e355c39b |
| reify | observations_and_summaries | claude-task-7327-architect | 1 | 58ac1744 |
| reify | observations_and_summaries | orchestrator-task-5783 | 1 | 8028eb76 |
| reify | observations_and_summaries | orchestrator-task-6288 | 1 | 67b36def |
| reify | observations_and_summaries | orchestrator-task-6468 | 1 | 9e393fea |
| reify | observations_and_summaries | orchestrator-task-6932 | 1 | 918c9ad0 |
| reify | observations_and_summaries | orchestrator-task-6990 | 1 | 4da67dbb |
| reify | observations_and_summaries | orchestrator-task-7205 | 1 | 76df97cd |
| reify | observations_and_summaries | orchestrator-task-7216 | 1 | 89999295 |
| reify | observations_and_summaries | orchestrator-task-7253 | 1 | 1f43ca25 |
| reify | observations_and_summaries | orchestrator-task-7306 | 1 | 592d71d0 |
| reify | observations_and_summaries | orchestrator-task-7315 | 1 | 732ac5c1 |
| reify | procedural_knowledge | claude-task-6347-architect | 1 | fa354ca3 |
| reify | procedural_knowledge | claude-task-6582-implementer | 1 | d65086af |
| reify | procedural_knowledge | claude-task-7189-implementer | 2 | d3eeb6bd c0725e64 |
| reify | procedural_knowledge | claude-task-7327-implementer | 1 | 4e9f6aac |
| solar_challenge | observations_and_summaries | (null) | 21 | 5274a3e4 8e39b929 626a9f45 fc613614 53dca707 5e33b57e 819f47db b84f922c 22614128 226a81c6 175282d0 0e2bef50 21f1cb69 9b5f8774 9f58fd9c 40b229c0 4050dd09 04c726e4 042022b5 19d0f8d7 9020fc33 |
| solar_challenge | observations_and_summaries | claude-task-340-implementer | 1 | da80af10 |
| solar_challenge | observations_and_summaries | claude-task-354-architect | 1 | 4bf2668e |
| solar_challenge | observations_and_summaries | claude-task-379-architect | 1 | d877187d |
| solar_challenge | observations_and_summaries | claude-task-379-implementer | 1 | 1323d672 |
| solar_challenge | observations_and_summaries | claude-task-413-implementer | 1 | 85fef08f |
| solar_challenge | observations_and_summaries | claude-task-434-architect | 2 | a1483bef 6a239b84 |
| solar_challenge | observations_and_summaries | claude-task-434-implementer | 1 | ef4565ea |
| solar_challenge | observations_and_summaries | claude-task-445-architect | 1 | 28727700 |
| solar_challenge | observations_and_summaries | orchestrator-task-327 | 1 | 47a5e5b0 |
| solar_challenge | observations_and_summaries | orchestrator-task-348 | 1 | 169c2f2b |
| solar_challenge | observations_and_summaries | orchestrator-task-375 | 1 | 1a0a499e |
| solar_challenge | observations_and_summaries | orchestrator-task-379 | 1 | 53c0084d |
| solar_challenge | observations_and_summaries | orchestrator-task-393 | 1 | 177aa521 |
| solar_challenge | observations_and_summaries | orchestrator-task-407 | 1 | 1a78dea1 |
| solar_challenge | observations_and_summaries | orchestrator-task-413 | 1 | d021b0dc |
| solar_challenge | observations_and_summaries | orchestrator-task-432 | 1 | d476f700 |
| solar_challenge | observations_and_summaries | orchestrator-task-433 | 1 | 21939ee7 |
| solar_challenge | observations_and_summaries | orchestrator-task-437 | 1 | 6c37c0c1 |
| solar_challenge | observations_and_summaries | orchestrator-task-445 | 1 | 5c60d8f8 |
| solar_challenge | observations_and_summaries | orchestrator-task-457 | 1 | 83cb11f4 |
| solar_challenge | preferences_and_norms | claude-task-340-architect | 1 | fa534a74 |
| solar_challenge | preferences_and_norms | claude-task-375-implementer | 1 | dfb6e378 (recovered and re-added from esc-375-2) |
| solar_challenge | preferences_and_norms | claude-task-375-steward | 1 | 03fd0efd |
| solar_challenge | procedural_knowledge | claude-task-195-architect | 1 | 1f717903 |
| solar_challenge | procedural_knowledge | claude-task-195-implementer | 1 | d38f4f82 |
| solar_challenge | procedural_knowledge | claude-task-354-implementer | 1 | 123670e0 |
| solar_challenge | procedural_knowledge | claude-task-379-architect | 1 | e34b5dd6 |
| solar_challenge | procedural_knowledge | claude-task-379-implementer | 1 | 050f787a |
| solar_challenge | procedural_knowledge | claude-task-393-architect | 1 | 9a2f26ef |
| solar_challenge | procedural_knowledge | claude-task-393-implementer | 1 | 3d274382 |
| solar_challenge | procedural_knowledge | claude-task-413-implementer | 1 | 63269190 |
| solar_challenge | procedural_knowledge | claude-task-432-architect | 1 | 1a720448 |
| solar_challenge | procedural_knowledge | claude-task-445-implementer | 1 | 344914c4 |
| solar_challenge | procedural_knowledge | claude-task-448-architect | 1 | a5387eec |
| solar_challenge | procedural_knowledge | claude-task-448-implementer | 1 | 291fb335 |
| solar_challenge | procedural_knowledge | claude-task-449-architect | 1 | d4d6be47 |
| solar_challenge | procedural_knowledge | claude-task-453-implementer | 1 | 395dcd1b |
| solar_challenge | procedural_knowledge | claude-task-454-architect | 1 | 558a32a3 |
| solar_challenge | procedural_knowledge | claude-task-457-architect | 1 | e82dcf41 |
| solar_challenge_platform | observations_and_summaries | (null) | 10 | e73f53b8 aa70a538 0fb68e74 40374f0a f77c18ce 17ee61bf 59c20a59 72cfc2bb acdc2a3f 1f0a1e8c |
| solar_challenge_platform | observations_and_summaries | claude-task-352-implementer | 1 | 5bb6547e |
| solar_challenge_platform | observations_and_summaries | claude-task-528-debugger | 1 | 1616f541 |
| solar_challenge_platform | observations_and_summaries | claude-task-528-implementer | 1 | f7513c76 |
| solar_challenge_platform | observations_and_summaries | claude-task-770-implementer | 2 | 858be856 0785a9fc |
| solar_challenge_platform | observations_and_summaries | orchestrator-task-425 | 1 | 924b01eb |
| solar_challenge_platform | observations_and_summaries | orchestrator-task-431 | 1 | bc4caab5 |
| solar_challenge_platform | observations_and_summaries | orchestrator-task-572 | 1 | def5fb7e |
| solar_challenge_platform | observations_and_summaries | orchestrator-task-756 | 1 | 4d7fda88 |
| solar_challenge_platform | observations_and_summaries | orchestrator-task-763 | 1 | 2222817a |
| solar_challenge_platform | observations_and_summaries | orchestrator-task-776 | 1 | db99f1cf |
| solar_challenge_platform | observations_and_summaries | orchestrator-task-782 | 1 | 16569d98 |
| solar_challenge_platform | observations_and_summaries | recon-stage-task_knowledge_sync | 1 | 4fd5f7af |
| solar_challenge_platform | preferences_and_norms | claude-task-506-architect | 1 | 82e25f58 |
| solar_challenge_platform | preferences_and_norms | claude-task-511-implementer | 1 | c97f7646 |
| solar_challenge_platform | procedural_knowledge | claude-task-506-architect | 1 | 062ec4c1 |
| solar_challenge_platform | procedural_knowledge | claude-task-506-implementer | 1 | 46e3a829 |
| solar_challenge_platform | procedural_knowledge | claude-task-518-implementer | 2 | 6d31de7c 6bd6ac5b |
| solar_challenge_platform | procedural_knowledge | claude-task-528-implementer | 1 | f0982d5d |
| solar_challenge_platform | procedural_knowledge | claude-task-599-implementer | 1 | 63be87ad |
| solar_challenge_platform | procedural_knowledge | claude-task-742-implementer | 1 | 757025d5 |
| solar_challenge_platform | procedural_knowledge | claude-task-770-implementer | 1 | 766a578f |
| solar_challenge_platform | procedural_knowledge | claude-task-782-implementer | 1 | 98daee7d |

This list has 175 rows. One of them, the `claude-task-375-implementer` row, was recovered from esc-375-2 and re-added, which is why the table above says 174.
