# Graphiti edge ranking: why task 877's live edges miss its own top 5 (task 6523)

## 1. Question and verdict

Question: for the query `task 877 context and related decisions` (graph
`dark_factory`, `stores=['graphiti']`, limit 5), why is no live edge that
mentions 877 ranked in the top 5?

Verdict: it is a ranking defect, caused by three independent mechanisms. RRF
fusion itself is sound: it fuses whatever the two legs hand it.

1. Retired edges compete for leg slots and fused slots. **Fixed by this task.**
2. The `(@group_id:"g")` prefix makes FalkorDB's union scoring non-additive.
   Filed as `tkt_0RWXMJ0N3Z9DYWT0N485FP9YF6`.
3. Rare id tokens lose to prose tokens, even without the prefix. Filed under
   the same ticket.

## 2. Measurement context

The architect took every measurement below on 2026-11-12. It is the context
these claims hold in:

- Read-only, on production FalkorDB `docker-falkordb-1` (graph module 41800,
  `RESULTSET_SIZE` 10000), graph `dark_factory` (69,513 `RELATES_TO` edges),
  main `511ed2c1f2`. Probes ran from the task 6523 worktree's `.venv`.
- The scratch repro in section 6 ran on `docker-falkordb-scratch-1` (port
  6380), in a throwaway graph that was deleted afterwards.
- **The cosine leg was not measured.** An agent shell carries no
  `OPENAI_API_KEY`, so the query vector could not be produced. Section 7 gives
  only a bound taken from the public search.

## 3. Search path and slot arithmetic

- The `search` MCP tool calls `MemoryService._search_graphiti`. That calls
  `GraphitiBackend.search` with `num_results = int(1.5 * limit) + 1`, so limit
  5 becomes 8.
- `GraphitiBackend.search` calls graphiti `Graphiti.search` with
  `EDGE_HYBRID_SEARCH_RRF`. Before this task the `SearchFilters()` was empty.
- `edge_search` runs the BM25 and cosine legs through
  `fused_memory/backends/falkor_edge_search.py::FalkorEdgeSearch`, each with
  `LIMIT 2 * num_results`, so 16 rows per leg. It fuses them with `rrf`
  (score 1/rank) and truncates the fused list to 8.
- Only after that does `_search_graphiti` drop retired edges
  (`services/memory_service.py::_edge_is_retired`, that is, `invalid_at` set).
  So a retired edge that fills a leg or fused slot has already displaced a
  live candidate. The arm harness calls `GraphitiBackend.search` directly and
  had no post-hoc drop at all.

## 4. The 877 edge set now

- 9 edges carry the fulltext token `877`.
- 4 are live under `graphiti_client.py::_ACTIVE_EDGE_PREDICATE`
  (`e.invalid_at IS NULL`): `f0465663`, `dbf74612`, `38241d9e` (the original
  "AllAccountsCappedException catch sites added in production orchestrator
  via task 877" edge) and `bba24b78`.
- 5 are retired: `5bf20c42`, `4acfcebd`, `1330ec7b`, `5d8b1c36`, `8813cc31`.
- The 2026-10-08 measurement of "1 live" counted an edge as live only when
  both `invalid_at` and `expired_at` were unset. The repo's predicate uses
  `invalid_at` alone.

## 5. BM25 leg ranks

Full query `(@group_id:"dark_factory") (task | 877 | context | related | decisions)`:

- Rank 1: `b0ead89e`, score 8.5, live.
- Rank 2: `5bf20c42`, score 7.0, **retired**.
- Ranks 3-19: short "Task N is related to ..." facts at 6.0.
- `38241d9e` and `bba24b78` fall below the 10000-row cap.

For the prefixed `877` alone, `38241d9e` is rank 7 of 9 at 0.5455. Retired
hits hold the top three scores.

Without the prefix, for the full query, `38241d9e` is rank 278 (rank 227 among
live edges) at 7.0. The top rows score 14-20: they are short facts matching
`context`, `decisions` or `related`.

## 6. Scorer behaviour

| query | 38241d9e | 5bf20c42 |
|---|---|---|
| unprefixed `877` | 6.0 | 12.0 |
| unprefixed `task` | 1.0 | 2.0 |
| unprefixed `task \| 877` | 7.0 | 14.0 |
| prefixed `(task \| 877)` | 0.1 | 7.0 |
| prefixed `(877 \| task)` | 0.5455 | 7.0 |

Without the prefix, the union scores the sum of its terms in either order
(7.0 for `38241d9e` with both orders). With the prefix, `38241d9e`'s score
depends on operand order, and both orders score far below that 7.0 sum.
Meanwhile retired `5bf20c42` keeps 7.0.

Scratch repro: a long edge matching `task` and `877` scores 0.1875 prefixed,
below a short edge matching `task` only at 0.2857. Without the prefix the two
score 2.5 and 1.0.

## 7. Fused results

- Limit 5 returns `b0ead89e`, `6b5c0138`, `7c34a516`, `fdc87d44`, `dd0a9fc3`.
  None of them mentions 877.
- Limit 40 (61 requested, 122 rows per leg) returns no edge that mentions
  877. All 40 rows are short "Task N ... related/decision" facts. This is the
  only bound on the cosine leg (see section 2).

The query template itself is no longer used by the briefing:
`shared/src/shared/briefing_queries.py::TASK_SEMANTIC` is `'{title} {area}'`.
The mechanisms still apply to every edge search.

## 8. What this task changed

- `GraphitiBackend.search` passes graphiti
  `search_filter=graphiti_client.py::_active_edge_filter()`. That filter is
  now the active-edge invariant's single home: `_ACTIVE_EDGE_PREDICATE` is
  compiled from it with graphiti's `edge_search_filter_query_constructor`.
  Both legs' WHERE now carries `invalid_at IS NULL`, so a retired edge takes
  no leg or fused slot.
- `MemoryService`'s post-hoc `_edge_is_retired` checks stay in place as
  redundant enforcement.
- Graphiti's internal search during `add_episode` does not go through
  `GraphitiBackend.search`, so the write path is untouched.
- Arm-harness ranks change meaning. `arm_harness/retrieval.py::probe_retrieval_utility`
  and `arm_harness/embedding_graph_phase.py::_known_item_rank` both call
  `GraphitiBackend.search`, so their known-item and retrieval-utility ranks
  now exclude retired edges. A known item whose only citing edge was
  invalidated during replay used to count as a hit and now counts as a miss.
  Arm results recorded before task 6523 are therefore not like-for-like with
  later ones.
- Tests: `fused-memory/tests/test_graphiti_search_active_edges.py`, a
  default-suite Cypher pin plus a live-FalkorDB pin marked `integration`.
- Expected effect on this query: retired `5bf20c42` leaves BM25 rank 2. This
  change does **not** bring `38241d9e` into the top 5, because mechanisms 2
  and 3 remain.

## 9. What was filed

`tkt_0RWXMJ0N3Z9DYWT0N485FP9YF6` covers mechanisms 2 and 3:

- **Prefix inversion.** Removing or reshaping the `@group_id` prefix emitted by
  `fused_memory/backends/falkor_fulltext.py::build_query` changes the edge BM25
  leg on the **write path** too. Graphiti ingestion finds edge-invalidation
  candidates through that leg
  (`graphiti_core/utils/maintenance/edge_operations.py::resolve_extracted_edges`
  searches with `EDGE_HYBRID_SEARCH_RRF` and `SearchFilters()`). It needs its
  own before/after measurement.
- **Id-token priority.** Whether a rare id token should outrank prose tokens
  is a retrieval-policy choice for every exact-token query.

## Probe Cypher

These are read-only probes. Run them with `GRAPH.RO_QUERY dark_factory`, where
`$q` is the output of `falkor_fulltext.py::build_query`. Use the bare
`(task | 877 | ...)` form for the unprefixed rows.

BM25 leg, every row with its score and liveness:

```cypher
CALL db.idx.fulltext.queryRelationships('RELATES_TO', $q)
YIELD relationship AS e, score
WITH e, score, startNode(e) AS n, endNode(e) AS m
WHERE n:Entity AND m:Entity AND e.group_id IN ['dark_factory']
RETURN e.uuid, score, e.invalid_at IS NULL AS live, e.fact
ORDER BY score DESC
```

The 877 edge set (section 4) is the same probe with `$q` set to
`(@group_id:"dark_factory") (877)`.
