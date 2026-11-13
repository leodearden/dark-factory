# Edge BM25 leg ranking (task 9542): measurements and gate

Task 9542 changes the edge BM25 leg
(`fused-memory/src/fused_memory/backends/falkor_edge_search.py::FalkorEdgeSearch.edge_fulltext_search`):

1. its RediSearch query carries no `(@group_id:...)` prefix; the Cypher still scopes
   by `e.group_id IN $group_ids`;
2. an identifier tier: the edges matching EVERY digit-bearing searchable term are
   ranked first, when all of them fit in the leg's limit.

The query policy lives in `fused-memory/src/fused_memory/backends/falkor_fulltext.py::build_edge_queries`.

## Verdict: two of three gates FAIL

| gate | rule | result |
|---|---|---|
| M1 write path | self-recall@20 B >= A and C >= B | **FAIL**: A 0.675, B 0.515, C 0.615 |
| M2 877 check | under C the live 877 edges f0465663, dbf74612, 38241d9e, bba24b78 are ranks 1-4 | **FAIL**: ranks 5, 6, 8, 9 (bba24b78, 38241d9e, dbf74612, f0465663) |
| M3 E1 | no recall measure falls, contamination does not rise | pass: all six measures identical |

Per the plan, a failed gate is escalated (design_concern), not tuned around.

## Run context

| run | host / graph | HEAD | UTC window |
|---|---|---|---|
| E1 before | FalkorDB localhost:6379, project dark_factory+reify | `5cf03dd94d` (base main) | 2026-11-13T00:17:25Z .. 00:20:36Z |
| M1 + M2 | FalkorDB localhost:6379, graph `dark_factory`, GRAPH.RO_QUERY only | `51d7bab3f5` | 2026-11-13T01:04:06Z .. 2026-11-13T01:05:25Z |
| E1 after | FalkorDB localhost:6379, project dark_factory+reify | `51d7bab3f5` | 2026-11-13T00:58:17Z .. 01:01:18Z |

`measure_leg.py` is the instrument and `m1-m2-results.json` its output. Every
production query is a GRAPH.RO_QUERY. The shipped leg (C) is
`search_utils.edge_fulltext_search` on an unconnected `_MultiTenantFalkorDriver`
whose `execute_query` replays each issued query read-only. A and B re-run C's
own union Cypher with other params:

* A, pre-change: query `build_query(sanitize(q), ['dark_factory'], 128)`, `promoted_uuids=[]`;
* B, union only: the shipped union query, `promoted_uuids=[]`;
* C, the shipped leg.

Overlap is |X ∩ Y| / max(|X|, |Y|), and 1.0 when both are empty.

## M1: write path

graphiti's `resolve_extracted_edges` invalidation search: the edge's own fact
as the query, `SearchFilters()`, group `[dark_factory]`, 20 rows. Sample:
every 325-th live edge in uuid order, starting at the first, first 200 (65129 live edges: `invalid_at IS NULL`, `:Entity`
endpoints, group `dark_factory`).

| variant | self-recall@20 | mean self-rank (found only) |
|---|---:|---:|
| A pre-change | 0.675 | 2.46 |
| B union only | 0.515 | 5.47 |
| C shipped | 0.615 | 4.03 |

* mean overlap@20: A/B 0.103, B/C 0.963;
* tier query issued for 32% of facts, and the tier fired (promoted) for 22%;
* on the 44 facts whose tier fired, the self edge is found by A 36, B 24, C 44;
* slowest shipped leg: 486 ms.

The loss is item 1, not the tier. Spot checks of facts that A finds and B does
not, with the RediSearch score of the self edge under each query:

| self edge | A self score, rank | B self score | B top score |
|---|---|---:|---:|
| `0f741ed4` "The wedged writer's release() reaps the thread..." | 45.0, rank 1 | 14.92 | 30.0 |
| `136e3251` "The polite service-restart path's agents_idle..." | 30.33, rank 2 | 1.625 | 25.5 |
| `15dfaa10` "Task A(5050) precedes task G(5058)..." | 63.0, rank 1 | 12.6 | 21.5 |

For a long query that is the edge's own fact, the prefixed query scores the
self edge far above the rest. The unprefixed union does not, so graphiti's
write-path duplicate and invalidation candidates lose their best match.

Hypothesis: under the `(@group_id:...)` intersection, FalkorDB 41800 scores an
edge by how many union terms it matches. The plain union sums per-term scores
that favour short facts. Scratch fixture R1 (a 3-term query) showed the
opposite trade, which is what the plan measured.

## M2: read path

`fused-memory/tests/fixtures/production_query_sample.jsonl` (44 rows), with
`memory_service`'s `num_results = int(5 * 1.5) + 1 = 8`, so 16 rows per leg, and
`SearchFilters()`.

* observed_count-weighted share of traffic whose top 16 changed against A: B 100%, C 100%;
* the tier query was issued for 21 of 44 queries and promoted edges for 5.

| query_id | text | observed_count | overlap@16 A/B | overlap@16 B/C | tier rows | promoted facts |
|---|---|---:|---:|---:|---:|---|
| prod-brief-b187a6e6cb36 | project overview architecture goals | 74702 | 0.12 | 1.00 |  |  |
| prod-brief-3ca4080f6509 | coding conventions and project norms | 74944 | 0.12 | 1.00 |  |  |
| prod-brief-33aee7837524 | recent decisions and rationale | 74884 | 0.06 | 1.00 |  |  |
| prod-brief-e43e40cb1ad0 | task reify_task_12 context and related decisions | 54097 | 0.06 | 0.88 | 4 | The plan involves smoke-testing minimax-m25-fp8-new with reify_task_12 to determ; The effect of fable is carried by the reify_task_12 fixture.; Reify_task_12 carries no 'reference' block, leading to a grading based on plausi; The nominal advantage of fable over incumbent rides on one fixture, reify_task_1 |
| prod-tail-bd7e0a851eee | 2026-04-21 mass task reset git branch state recover in-flight work | 33 | 0.00 | 1.00 | 17 |  |
| prod-tail-b1627623a71a | 5295 | 1 | 1.00 | 1.00 | 2 | The follow-up agent-followup-5295-remote-task-files-derivation can reuse the app; Follow-up agent-followup-5295-remote-task-files-derivation can reuse the keyword |
| prod-tail-6eccc8bfe560 | Graphiti search degraded fused-memory query execution path | 1 | 0.12 | 1.00 |  |  |
| prod-tail-18b2cac32997 | Human curator gate: consolidate 4 overlapping reify-audit PTODO-detect | 2 | 0.00 | 1.00 | 0 |  |
| prod-tail-01f8e2bec32c | TargetedReconciler TaskInterceptor write_lock bypass | 29 | 0.12 | 1.00 |  |  |
| prod-tail-bbd2f5810314 | Task 1675 phantom done commits e161348f 60b85659 14bb892c db17363e leg | 1 | 0.06 | 1.00 | 0 |  |
| prod-tail-b05761e79538 | TestDryRunZeroSideEffects consolidation | 2 | 0.25 | 1.00 |  |  |
| prod-tail-673c1801f840 | blockers for: Strengthen LSP bridge test coverage: helpers, assertions | 36 | 0.19 | 1.00 | 17 |  |
| prod-tail-8caf3e2b65fb | blockers for: ρ1b fused-memory: transition-table log-mode at intercept | 5 | 0.00 | 1.00 | 0 |  |
| prod-tail-e36c67c39105 | cargo skim output wrapper condensation timeout bypass | 41 | 0.00 | 1.00 |  |  |
| prod-tail-5ab98c3509e5 | cycle summary run_id stage memory_consolidator | 29 | 0.12 | 1.00 |  |  |
| prod-tail-095648382582 | de-flake flaky timing wall-clock test merge gate ambush warm-lane pref | 1 | 0.00 | 1.00 |  |  |
| prod-tail-a1c3fad46fe9 | engine_purposes.rs merged_purposes merged commit task 4016 ζ boundary  | 1 | 0.00 | 1.00 | 0 |  |
| prod-tail-a1c775469bbe | flag marker f7998d1d 68093151 28bc288c stage 1 b4242d75 | 1 | 0.00 | 1.00 | 0 |  |
| prod-tail-7ae969361dc5 | gate BLOCKED do NOT flip task_metadata.enforce esc-2184-1 | 1 | 0.12 | 1.00 | 0 |  |
| prod-tail-9ee325c0ade4 | kernel_registry multi-kernel dispatch lex-min selection capability des | 1 | 0.31 | 1.00 |  |  |
| prod-tail-9f67572c3d47 | merge_request coalesce gate dead request_id resubmit | 1 | 0.00 | 1.00 |  |  |
| prod-tail-fa37aa9d637d | project overview and current status | 340 | 0.00 | 1.00 |  |  |
| prod-tail-b0d0230837fb | project overview current status architecture | 33 | 0.12 | 1.00 |  |  |
| prod-tail-b22fc044e988 | project overview current status autopilot video | 29 | 0.00 | 1.00 |  |  |
| prod-tail-be9f0536fc05 | project overview current status recent decisions | 40 | 0.00 | 1.00 |  |  |
| prod-tail-cc2a6ade44cb | project overview current status solar challenge | 39 | 0.12 | 1.00 |  |  |
| prod-tail-a524e77a7cf4 | recent decisions and known issues | 84 | 0.19 | 1.00 |  |  |
| prod-tail-70876520d417 | recent decisions and project status | 45 | 0.00 | 1.00 |  |  |
| prod-tail-f13a68e6b0d7 | recent decisions and session summary | 36 | 0.00 | 1.00 |  |  |
| prod-tail-db52c23ec966 | recent task completions and status changes | 30 | 0.00 | 1.00 |  |  |
| prod-tail-5049c1e2d9d1 | stage1_cycle_summary run_id pool cap maintenance | 1 | 0.06 | 1.00 | 0 |  |
| prod-tail-0a3e2549ef85 | stage1_flag_marker flag_for_stage2 | 43 | 0.75 | 1.00 | 0 |  |
| prod-tail-79b71adab3e2 | stage1_flag_suppression | 30 | 1.00 | 1.00 | 10 | The stage1_flag_suppression system implements end-to-end deduplication.; Task 1185 added the producer that writes stage1_flag_suppression records with ro; Task 1185 was completed through commit fbf82da6, which wired up the producer for; Task 1186 was completed through commit 90eeee33, which enforced stage1_flag_supp; The stage1_flag_suppression matches the census exactly with 16 records.; stage1_flag_suppression carries kind plus a sorted-unique plural flag_types, or ; The stage-facing stage1_flag_suppression remedy is retired, not built out.; Task 4863 rejects kind=stage1_flag_suppression add_memory writes from recon-stag; write_suppression_record's operator mirror goes through memory_service.add_memor; The identifiers include the stage1_flag_suppression call in the test_recon_marke |
| prod-tail-fb558b0a13e3 | stage1_flag_suppression task_id | 29 | 0.38 | 1.00 | 10 | The stage1_flag_suppression system implements end-to-end deduplication.; Task 1185 added the producer that writes stage1_flag_suppression records with ro; Task 1185 was completed through commit fbf82da6, which wired up the producer for; Task 1186 was completed through commit 90eeee33, which enforced stage1_flag_supp; The stage1_flag_suppression matches the census exactly with 16 records.; stage1_flag_suppression carries kind plus a sorted-unique plural flag_types, or ; The stage-facing stage1_flag_suppression remedy is retired, not built out.; Task 4863 rejects kind=stage1_flag_suppression add_memory writes from recon-stag; write_suppression_record's operator mirror goes through memory_service.add_memor; The identifiers include the stage1_flag_suppression call in the test_recon_marke |
| prod-tail-27486ece8dd5 | stage1_flag_suppression task_id=165 | 32 | 0.38 | 1.00 | 0 |  |
| prod-tail-e4918ee31ec6 | stage1_flag_suppression task_id=452 | 105 | 0.38 | 1.00 | 0 |  |
| prod-tail-66f71c02b8dd | sync intrinsics calibration | 1 | 0.25 | 1.00 |  |  |
| prod-tail-65bdf7bd04e5 | task 19 P2 beta canonical telemetry schema types completed done | 1 | 0.00 | 1.00 | 0 |  |
| prod-tail-d590b99b502e | task 289 285 276 204 287 completion observation done commit | 1 | 0.00 | 1.00 | 0 |  |
| prod-tail-a38e864205f2 | task 312 review TestHandleDryRun test structure cleanup | 1 | 0.00 | 1.00 | 1 | Manual /compact did not re-mint session_id on CLI 2.1.312. |
| prod-tail-761661aad4da | task 552 T-ROMA RoMa dense warp real backend done completion | 1 | 0.00 | 1.00 | 17 |  |
| prod-tail-bf55d0478466 | task count snapshot | 32 | 0.19 | 1.00 |  |  |
| prod-tail-279a7f47ee34 | task memory observation mapping e2496a40 tasks 89 100 101 102 | 1 | 0.19 | 1.00 | 0 |  |
| prod-tail-14c93cb10f7b | verify cold timeout cascade merge_verify_cold_command_timeout_secs def | 1 | 0.38 | 1.00 |  |  |

Promotions to note for the reviewer: `task 312 review ...` promotes
"Manual /compact did not re-mint session_id on CLI 2.1.312.", because `312`
matches a version number. `stage1_flag_suppression` is itself an identifier
term, because its `1` is a digit.

### The 877 query

`task 877 context and related decisions`, C, 16 rows. The tier holds 10 edges, so all of them
are promoted. Five have `invalid_at` set: `5bf20c42`, `4acfcebd`, `1330ec7b`,
`5d8b1c36` and `8813cc31`. Four of them are "Task 877 ..." facts, and they rank
above the live ones on union score. `SearchFilters()` does not exclude
invalidated edges. `memory_service` drops them after fusion
(`fused-memory/src/fused_memory/services/memory_service.py::_edge_is_retired`,
which reads `invalid_at` only), but inside the leg they take ranks 1-4. The
other five have `invalid_at` null. `bba24b78` carries an `expired_at`, which
`_edge_is_retired` deliberately ignores. Since planning, a fifth live 877 edge
has appeared (`4effe5d2`, a fact about this task's plan). Under A and B, none of
the four named edges is in the top 16.

| C rank | edge | fact |
|---:|---|---|
| 1 | `5bf20c42` | Task 877 adds catch sites for AllAccountsCappedException in fused-memory. |
| 2 | `4acfcebd` | Task 877 (Add catch sites for AllAccountsCappedException in orchestrator and fused-memory) was  |
| 3 | `1330ec7b` | Task 877 was previously blocked but is now actionable since prerequisite Task 881 was already c |
| 4 | `5d8b1c36` | Only Task 877 addresses the missing catch sites for AllAccountsCappedException. |
| 5 | `bba24b78` | AllAccountsCappedException catch sites added in fused-memory via task 877 (commit 371d476388d7c |
| 6 | `38241d9e` | AllAccountsCappedException catch sites added in production orchestrator via task 877 (commit 37 |
| 7 | `4effe5d2` | production_query_sample.jsonl is part of the M2 blast-radius census plus the 877 check. |
| 8 | `dbf74612` | The FIRST id is chosen from drifted id 877 based on the replacement rule. |
| 9 | `f0465663` | The 877 ranking observation was filed by the architect as tkt_0RVJFNZTMNA6N1HE3CR9VPZ2M9. |
| 10 | `8813cc31` | In the decision to unblock workflow, the user chose option 2 of 3 regarding task 877. |
| 11 | `408c786a` | Per-context instance labels reused across contexts are considered homonyms, not one referent. |
| 12 | `27f6edcd` | The clean-worktree Phase 1 is an open decision related to uv run. |
| 13 | `45d58b1f` | The vllm-eval subsystem decisions include parking devstral. |
| 14 | `e06fa4e7` | Leo participated in a discussion session regarding WarmLanePool hardening decisions on 2026-07- |
| 15 | `e74b0221` | CGL-ι Phase-2 human-review data decisions were made regarding consolidate_namespace_families.py |
| 16 | `831b1352` | The phrase 'recent decisions and changes' in memory.md is identified as a rewording that needs  |

## M3: E1 before and after

Artifacts: `e1-before/` and `e1-after/`, byte copies.

| measure | before | after |
|---|---:|---:|
| canonical-in-top-5 | 0.2255 (n=102) | 0.2255 (n=102) |
| canonical-in-top-10 | 0.2353 (n=102) | 0.2353 (n=102) |
| canonical-in-top-5-held-out | 0.1143 (n=35) | 0.1143 (n=35) |
| claim-recall | 0.4118 (n=34) | 0.4118 (n=34) |
| contamination-share | 0.0039 (n=510) | 0.0039 (n=510) |
| topic-canonical-present (tripwire) | 31 of 34 failing | 31 of 34 failing |

The two reports differ only in corpus counts (live writes between the runs)
and in one line: `'why does the build script rerun every time'` is now
`served by graphiti, mem0` instead of `mem0`. No phrasing's canonical moved
across top 5 or top 10.

The four digit-bearing phrasings: `'why are graphiti edges 3ef4c09f and
912e58e8 orphaned'` is not in the top 10 in both runs. `inv-geo-1-mesh-contract-mock`
passes in both. The `C1` and `G6` phrasings are not listed as failing in
either report. E1 is a weak instrument here: only 4 of its 102 phrasings carry
digits, and its canonicals are Mem0 records.

## Other notes

* Arm-harness known-item ranks change meaning after this task, as they did
  after task 6523: the edge leg's cutoff is now 65 terms (64 survive), while the
  node and episode legs keep their prefix and die at 64.
* The default fused-memory suite on HEAD `51d7bab3f5`, in the dispatched agent
  sandbox: 35470 passed, 3 failed (`test_predicate_contradiction.py`: no
  pytest on `/usr/bin/python`; `test_roll_up_journal_reads_script.py` x2:
  "unable to open database file"), and `test_lane_dependency_probe.py` (3)
  refused because `/home/leo/.mem0` is not writable. None of these touches the
  edge leg. Not reproduced outside the sandbox.
