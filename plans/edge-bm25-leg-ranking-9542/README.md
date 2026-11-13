# Edge BM25 leg ranking (task 9542): measurements and gate

Task 9542 changes the edge BM25 leg
(`fused-memory/src/fused_memory/backends/falkor_edge_search.py::FalkorEdgeSearch.edge_fulltext_search`).
The shipped design comes from the steward ruling on esc-9542-1:

* **Identifier tier, live edges only.** The tier is the set of edges that match
  EVERY digit-bearing searchable term of the query (a RediSearch intersection),
  read under the leg's own filters plus `e.invalid_at IS NULL`. The tier is ranked
  first only when all of it fits in the leg's limit; the tier query reads
  `limit + 1` rows to detect overflow. Inside the tier, edges are ordered by the
  union's score.
* **The union keeps its `(@group_id:...)` prefix.** Its filters are unchanged, so
  the ranked rows keep graphiti's `SearchFilters()` semantics, invalidated edges
  included.

The query policy lives in `fused-memory/src/fused_memory/backends/falkor_fulltext.py::build_edge_queries`.
The tier's live-edge predicate is `falkor_edge_search.py::_ACTIVE_EDGE_PREDICATE`.
It is a noted copy of `graphiti_client.py::_ACTIVE_EDGE_PREDICATE`, because
`graphiti_client` imports `falkor_edge_search`.

Item 1 of the task, dropping the prefix from the edge leg's union, is closed:
**measured and rejected: the prefix's non-additive scoring helps write-path
self-recall** (see "Item 1: measured and rejected" below, and the steward's A+tier
run).

## Verdict: all three gates pass

The gates are from the esc-9542-1 re-gate.

| gate | rule | result |
|---|---|---|
| M1 write path | self-recall@20, shipped C >= pre-change A, on the same sample in the same run | pass: C 0.68, A 0.615 |
| 877 check | every live 877 edge is in C's top 5, and no invalidated edge is promoted | pass: the five live 877 edges are ranks 1-5; all five promoted edges are live |
| M3 E1 | unchanged | pass: all six measures identical; no canonical moved across top 5 or top 10 |

## Run context

| run | host / graph | HEAD | UTC window |
|---|---|---|---|
| E1 before | FalkorDB localhost:6379, project dark_factory+reify | `5cf03dd94d` (base main) | 2026-11-13T00:17:25Z .. 00:20:36Z |
| M1 + M2 + 877, final | FalkorDB localhost:6379, graph `dark_factory`, GRAPH.RO_QUERY only | `7f33813f09` | 2026-11-13T01:57:01Z .. 01:57:48Z |
| E1 after, final | FalkorDB localhost:6379, project dark_factory+reify | `7f33813f09` | 2026-11-13T01:55:02Z .. 01:57:41Z |
| superseded: M1 + M2 with the prefix dropped | as above, GRAPH.RO_QUERY only | `51d7bab3f5` | 2026-11-13T01:04:06Z .. 01:05:25Z |
| superseded: steward A+tier (esc-9542-1) | as above, GRAPH.RO_QUERY only | `bfb62cbf44` | 2026-11-13T01:27:06Z |

`measure_leg.py` is the instrument, and `m1-m2-results.json` is its final output.
Every production query is a GRAPH.RO_QUERY. The shipped leg (C) is
`search_utils.edge_fulltext_search` on an unconnected `_MultiTenantFalkorDriver`,
whose `execute_query` replays each issued query read-only. A re-runs C's own
union Cypher with `promoted_uuids=[]`:

* A, pre-change: the query `build_query(sanitize(q), ['dark_factory'], 128)`, with no promotion.
  For every query, the instrument checks that this is exactly the union query C
  issued.
* C, the shipped leg.

Overlap is |X ∩ Y| / max(|X|, |Y|), and 1.0 when both are empty.

## M1: write path

This is graphiti's `resolve_extracted_edges` invalidation search: the edge's own
fact is the query, with `SearchFilters()`, group `[dark_factory]` and 20 rows.
The sample is every 325th live edge in uuid order, starting at the first, first
200. There were 65165 live edges (`invalid_at IS NULL`, `:Entity` endpoints, group
`dark_factory`). The graph moves between runs, so this sample is not the same 200
as in the superseded runs.

| variant | self-recall@20 | mean self-rank (found only) |
|---|---:|---:|
| A pre-change | 0.615 | 2.36 |
| C shipped | 0.68 | 2.04 |

* mean overlap@20 A/C: 0.965;
* the tier query was issued for 37.5% of facts, and the tier fired (promoted) for 25%;
* on the 50 facts whose tier fired, the self edge is found by A 37 times and by C 50 times;
* slowest shipped leg: 276 ms.

## M2: read path

`fused-memory/tests/fixtures/production_query_sample.jsonl` (44 rows), with
`memory_service`'s `num_results = int(5 * 1.5) + 1 = 8`, so 16 rows per leg, and
`SearchFilters()`.

* The observed_count-weighted share of traffic whose top 16 changed against A is
  19.4%. Almost all of it is the briefing query `task reify_task_12 context and
  related decisions` (54097 observations).
* The tier query was issued for 21 of 44 queries, and promoted edges for 7.

Only the queries whose tier query was issued are listed. For every other row,
C equals A (overlap 1.00, unchanged).

| query_id | text | observed_count | overlap@16 A/C | changed | tier rows | promoted facts |
|---|---|---:|---:|---|---:|---|
| prod-brief-e43e40cb1ad0 | task reify_task_12 context and related decisions | 54097 | 0.81 | yes | 4 | four reify_task_12 facts ("The effect of fable is carried by the reify_task_12 fixture.", ...) |
| prod-tail-bd7e0a851eee | 2026-04-21 mass task reset git branch state recover in-flight work | 33 | 0.44 | yes | 9 | nine facts carrying 2026, 04 and 21, two of them the AllAccountsCappedException 877 edges |
| prod-tail-b1627623a71a | 5295 | 1 | 1.00 |  | 2 | two agent-followup-5295 facts |
| prod-tail-18b2cac32997 | Human curator gate: consolidate 4 overlapping reify-audit PTODO-detect | 2 | 1.00 |  | 0 |  |
| prod-tail-bbd2f5810314 | Task 1675 phantom done commits e161348f 60b85659 14bb892c db17363e leg | 1 | 1.00 |  | 0 |  |
| prod-tail-673c1801f840 | blockers for: Strengthen LSP bridge test coverage: helpers, assertions | 36 | 1.00 |  | 17 | (overflow) |
| prod-tail-8caf3e2b65fb | blockers for: ρ1b fused-memory: transition-table log-mode at intercept | 5 | 1.00 |  | 0 |  |
| prod-tail-a1c3fad46fe9 | engine_purposes.rs merged_purposes merged commit task 4016 ζ boundary | 1 | 1.00 |  | 0 |  |
| prod-tail-a1c775469bbe | flag marker f7998d1d 68093151 28bc288c stage 1 b4242d75 | 1 | 1.00 |  | 0 |  |
| prod-tail-7ae969361dc5 | gate BLOCKED do NOT flip task_metadata.enforce esc-2184-1 | 1 | 1.00 |  | 0 |  |
| prod-tail-5049c1e2d9d1 | stage1_cycle_summary run_id pool cap maintenance | 1 | 1.00 |  | 0 |  |
| prod-tail-0a3e2549ef85 | stage1_flag_marker flag_for_stage2 | 43 | 1.00 |  | 0 |  |
| prod-tail-79b71adab3e2 | stage1_flag_suppression | 30 | 1.00 |  | 10 | ten stage1_flag_suppression facts, already A's top 10 |
| prod-tail-fb558b0a13e3 | stage1_flag_suppression task_id | 29 | 0.62 | yes | 10 | the same ten stage1_flag_suppression facts |
| prod-tail-27486ece8dd5 | stage1_flag_suppression task_id=165 | 32 | 1.00 |  | 0 |  |
| prod-tail-e4918ee31ec6 | stage1_flag_suppression task_id=452 | 105 | 1.00 |  | 0 |  |
| prod-tail-65bdf7bd04e5 | task 19 P2 beta canonical telemetry schema types completed done | 1 | 1.00 |  | 0 |  |
| prod-tail-d590b99b502e | task 289 285 276 204 287 completion observation done commit | 1 | 1.00 |  | 0 |  |
| prod-tail-a38e864205f2 | task 312 review TestHandleDryRun test structure cleanup | 1 | 0.94 | yes | 1 | "Manual /compact did not re-mint session_id on CLI 2.1.312." |
| prod-tail-761661aad4da | task 552 T-ROMA RoMa dense warp real backend done completion | 1 | 0.44 | yes | 9 | nine "Task 552 ..." facts |
| prod-tail-279a7f47ee34 | task memory observation mapping e2496a40 tasks 89 100 101 102 | 1 | 1.00 |  | 0 |  |

The full per-query rows, including the promoted facts verbatim, are in
`m1-m2-results.json` under `m2.per_query`.

False promotions, listed for the reviewer and accepted for now per esc-9542-1:

* `task 312 review ...` promotes "Manual /compact did not re-mint session_id on
  CLI 2.1.312.", because `312` matches a version number.
* `2026-04-21 mass task reset ...` intersects `2026`, `04` and `21`, and promotes
  nine facts that carry the date. Two of them, the 877 AllAccountsCappedException
  edges, mention it only in passing ("Previously had no catch sites as of
  2026-04-21"). Whether the other seven are about the mass reset is not judged
  here.
* `stage1_flag_suppression` is itself an identifier term, because its `1` is a
  digit. The facts it promotes are on topic.

## The 877 query

`task 877 context and related decisions` under C, 16 rows. The tier holds
5 live edges, all promoted, and all five live 877 edges are ranks 1-5. Here
"live 877 edges" means every live `:Entity` edge in `dark_factory` whose name or
fact holds the token `877` after `sanitize()`, found by scanning the M1 live-edge
read without RediSearch. The five invalidated 877 edges (`5bf20c42`, `4acfcebd`,
`1330ec7b`, `5d8b1c36`, `8813cc31`) are not promoted. `5bf20c42` still ranks 7th on
union score, which is what `SearchFilters()` allows, and `memory_service` drops it
after fusion (`fused-memory/src/fused_memory/services/memory_service.py::_edge_is_retired`).
Under A, none of the four edges named during planning is in the top 16, and the
retired `5bf20c42` is rank 2.

| C rank | edge | retired | fact |
|---:|---|---|---|
| 1 | `f0465663` |  | The 877 ranking observation was filed by the architect as tkt_0RVJFNZTMNA6N1HE3CR9VPZ2M9. |
| 2 | `dbf74612` |  | The FIRST id is chosen from drifted id 877 based on the replacement rule. |
| 3 | `4effe5d2` |  | production_query_sample.jsonl is part of the M2 blast-radius census plus the 877 check. |
| 4 | `bba24b78` |  | AllAccountsCappedException catch sites added in fused-memory via task 877 (commit 371d476388d7c... |
| 5 | `38241d9e` |  | AllAccountsCappedException catch sites added in production orchestrator via task 877 (commit 37... |
| 6 | `b0ead89e` |  | The decision regarding dark-factory's checkout was verified by rolling back one commit and re-p... |
| 7 | `5bf20c42` | yes | Task 877 adds catch sites for AllAccountsCappedException in fused-memory. |
| 8 | `63a916ee` |  | Task 3254 had five records related to L2 escalations that were collapsed into one survivor reco... |
| 9-16 | ... |  | other "Task NNNN ..." prose facts |

## M3: E1 before and after

The artifacts are byte copies: `e1-before/` and `e1-after/`. `e1-after-51d7bab3f5/`
is the superseded after-run with the prefix dropped.

| measure | before | after |
|---|---:|---:|
| canonical-in-top-5 | 0.2255 (n=102) | 0.2255 (n=102) |
| canonical-in-top-10 | 0.2353 (n=102) | 0.2353 (n=102) |
| canonical-in-top-5-held-out | 0.1143 (n=35) | 0.1143 (n=35) |
| claim-recall | 0.4118 (n=34) | 0.4118 (n=34) |
| contamination-share | 0.0039 (n=510) | 0.0039 (n=510) |
| topic-canonical-present (tripwire) | 31 of 34 failing | 31 of 34 failing |

The two reports differ only in two places:

* corpus counts, from live writes between the runs;
* `'the planner started over because its earlier plan was not on disk'` and the
  `memory-hygiene` topic are now `served by graphiti, mem0` instead of `mem0`.

No phrasing's canonical moved across top 5 or top 10.

The four digit-bearing phrasings behave the same in both runs:

* `'why are graphiti edges 3ef4c09f and 912e58e8 orphaned'` is not in the top 10;
* `inv-geo-1-mesh-contract-mock` passes;
* the `C1` and `G6` phrasings are not listed as failing in either report.

E1 is a weak instrument here: only 4 of its 102 phrasings carry digits, and its
canonicals are Mem0 records.

## Item 1: measured and rejected

The plan's item 1 dropped the `(@group_id:...)` prefix from the edge leg's union and
kept group scoping in Cypher. It was planned on scratch fixture R1, a 3-term query.
Without the prefix, a long fact matching `task` plus a rare term ranked 1st. With
the prefix, it was outside the top 5.

On production the trade went the other way for graphiti's write path, which
queries with whole facts. At `51d7bab3f5`, with `m1-m2-results-51d7bab3f5.json`
and the instrument in this file's history at `dceb4cb329`:

| variant | self-recall@20 | mean self-rank (found only) |
|---|---:|---:|
| A pre-change (prefix) | 0.675 | 2.46 |
| B union, prefix dropped | 0.515 | 5.47 |
| C B + tier (invalidated edges included) | 0.615 | 4.03 |

Spot checks of facts that A finds and B does not, with the RediSearch score of the
self edge under each query:

| self edge | A self score, rank | B self score | B top score |
|---|---|---:|---:|
| `0f741ed4` "The wedged writer's release() reaps the thread..." | 45.0, rank 1 | 14.92 | 30.0 |
| `136e3251` "The polite service-restart path's agents_idle..." | 30.33, rank 2 | 1.625 | 25.5 |
| `15dfaa10` "Task A(5050) precedes task G(5058)..." | 63.0, rank 1 | 12.6 | 21.5 |

Hypothesis: under the `(@group_id:...)` intersection, FalkorDB 41800 scores an
edge by how many union terms it matches. The plain union instead sums per-term
scores, and those favour short facts.

The steward's run for esc-9542-1 (`steward_aplus_tier.py`,
`steward-aplus-tier-results.txt`, HEAD `bfb62cbf44`, 65148 live edges) put the tier
on the prefixed union. Self-recall@20 was A 0.61, A+tier 0.665, A+livetier 0.665,
and C 0.615. On the 877 query, A+tier (invalidated edges promoted) put five
invalidated edges at ranks 1-5. A+livetier, the shipped design, put the five live
877 edges at ranks 1-5.

That run also found the invalidated-edge defect, and it is why the tier reads live
edges only. At `51d7bab3f5` the `(877)` tier held 10 edges, five with `invalid_at`
set. Four of those were "Task 877 ..." facts, and they took C ranks 1-4. Promoting
a retired edge spends a leg slot on a row the read path discards after fusion.

## Other notes

* After this task, the meaning of arm-harness known-item ranks changes for
  digit-bearing known-item queries, as it did after task 6523: the identifier
  tier reorders them. The edge leg's over-length cutoff is unchanged: one group
  id, 64 searchable terms die and 63 survive, the same as the node and episode
  legs.
* Live tests on scratch FalkorDB localhost:6380 at `7f33813f09`:
  `test_falkor_edge_search_integration.py -m integration`, 25 passed. These include
  `test_retired_edges_are_not_promoted` and
  `test_the_tier_fits_the_leg_on_its_live_edges`.
* Default fused-memory suite (`-n 8`) at `7f33813f09`, in the dispatched agent
  sandbox: 35481 passed, 9 failed, 12 skipped, 14 xfailed. All nine failures are
  sandbox environment refusals in files that do not touch the edge leg:
  * `test_lane_dependency_probe.py` x6: `/home/leo/.mem0` is not writable;
  * `test_predicate_contradiction.py` x1: `/usr/bin/python` has no pytest;
  * `test_roll_up_journal_reads_script.py` x2: "unable to open database file".

  The same three files failed at `51d7bab3f5`. Not reproduced outside the
  sandbox.
