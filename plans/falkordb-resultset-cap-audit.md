# FalkorDB result-set cap: audit of every read in `graphiti_client.py`

This is the one place where the measured row counts, the paging cost, the per-read classification, the open residual and the ticket cross-references are written down. Source code points here and does not restate them, so a re-measurement is a one-file edit. The rationale for each mechanism stays next to its definition in `fused-memory/src/fused_memory/backends/graphiti_client.py` (`_paged_ro_query`, `_apply_incompleteness_policy`, `IncompleteEnumerationError`, `_read_all_group_episodes`).

Every figure below carries its date. The figures rot: Entity nodes on `dark_factory` read 16038, then 16083, then 16262 over roughly 24 hours of task 4340. Each claim is stated with its reason so that it can be falsified rather than trusted.

History: first written as an in-source comment block by task 4340 (re-checked 2026-08-17). Task 4869 moved it here (2026-09-24) after paginating the four reads that audit had left at risk.

## The cap

FalkorDB returns at most `RESULTSET_SIZE` rows per query and truncates silently: no error, no marker. Nothing in this repo sets or overrides it.

- Measured 2026-08-17 against localhost:6379: `GRAPH.CONFIG GET RESULTSET_SIZE -> 10000`.
- Re-measured 2026-09-24 (task 4869 architect), same command, same answer: `10000`.
- Corroborated end to end on 2026-09-24 by the task-4869 live test (`TestStaleEmbeddingsLiveFalkorDB` in `fused-memory/tests/test_graph_read_pagination.py`). On a throwaway graph of 12000 embedded Entity nodes, the old unpaginated `query_stale_node_embeddings` returned 10000; the paginated one returns 12000. The same day the test was extended to a throwaway graph of 12000 embedded RELATES_TO edges. An unpaginated read of it returned 10000, and `query_stale_edge_embeddings` returned all 12000.

`_RESULTSET_SIZE` in `graphiti_client.py` encodes this value as an ASSUMPTION. That is why `_paged_ro_query` cross-checks every read against a server-side census instead of trusting it.

## Live row counts

### 2026-08-17 (task 4340), localhost:6379, RESULTSET_SIZE=10000

    graph          Entity nodes   valid-edge rows   an unpaginated read saw
    dark_factory       16083            25040                10000
    reify              23616            31659                10000

The same audit put the embedded populations at about 16083 Entity nodes on `dark_factory` (`query_stale_node_embeddings`) and about 15242 / 22392 edges carrying `fact_embedding` (`query_stale_edge_embeddings`, dark_factory / reify).

### 2026-09-24 (task 4869 architect), read-only `count(*)` probes against localhost:6379

    graph          Entity   directed RELATES_TO   with fact_embedding   Episodic in-group (on the label)
    dark_factory    19287         18380                  15216                3498 (3576)
    reify           25598         24174                  22345                4737 (4851)

Every Entity node on `dark_factory` carried `name_embedding`. The gap between the Episodic in-group count and the label count is rows whose `group_id` names another project. On `dark_factory` the gap is 78 rows, attributed to the task-2115 cross-graph leak. On `reify` it is 114 rows, which is the arithmetic difference and has not been separately attributed.

On these figures both stale-embedding reads were truncated on both graphs. `retrieve_episodes` stood at 35% (`dark_factory`) and 47% (`reify`) of the cap, and growing.

## MEASURED COST

The following text is moved verbatim from the task-4340 comment block:

> MEASURED COST of paging, and the keyset rewrite it rules out.  Measured
> 2026-08-18 against localhost:6379, warm, 3 repeats, median reported; the
> whole enumeration (census + every page), page_size 5000:
>
>     graph          read          UNPAGINATED     PAGED       rows
>     dark_factory   entity nodes     860 ms      862 ms      16262
>     dark_factory   valid edges      771 ms     3301 ms      25382
>     reify          entity nodes    3326 ms     1498 ms      23671
>     reify          valid edges     3126 ms     3685 ms      31783
>
> The UNPAGINATED column is the OLD behaviour and returned 10000 truncated
> rows for its money, so it is a cost floor, not a comparable answer.  One
> full detect_stale_with_edges (both reads) costs ~4.2 s on dark_factory —
> about +2.6 s per reconciliation cycle — and ~5.2 s on reify, which is
> FASTER than the ~6.5 s the two truncated reads used to cost there.  Paging
> a large result set in 5000-row chunks beats transferring one 10000-row set.
>
> KEYSET/SEEK PAGINATION WAS TRIED AND DECLINED, on measurement rather than
> on taste.  The concern it answers is real in principle: ORDER BY ... SKIP k
> LIMIT n re-scans and re-sorts the whole matched population per page, so an
> enumeration is O(P * N log N) where a seek would be O(N log N).  For the
> node read the seek form is available — `WHERE n.uuid > $last ORDER BY
> n.uuid LIMIT k` over the RANGE index ensure_indices creates on
> Entity(uuid).  Measured head to head, running SEEK FIRST each round so any
> warm-cache advantage favoured it:
>
>     graph          node SEEK (keyset)        node SKIP (offset)
>     dark_factory   [781, 904, 862] ms        [796, 862, 1610] ms
>     reify          [1414, 1310, 1577] ms     [1814, 1498, 1214] ms
>
> Indistinguishable — identical medians on dark_factory, ~6% on reify, well
> inside the run-to-run spread.  The asymptotic argument does not bite at
> this N: 4-5 pages over ~16-24k rows, where the sort is not the bottleneck.
> So a second paging mode in _paged_ro_query would buy no measured latency
> and cost a second code path to keep correct.  Re-open only WITH a
> measurement showing the sort dominating — and note the edge read, which is
> the expensive one, cannot use it anyway: its ORDER BY is the composite
> (e.uuid, n.uuid) needed for a total order over ROWS.
>
> DOWNSTREAM FAN-OUT, the other cost this fix moved.  detect_stale_dry_run
> issues one get_valid_edges_for_node per non-empty-summary entity, and now
> runs over the COMPLETE node set instead of a truncated 10000.  Measured
> 0.70 ms/entity amortised at max_concurrency=10 on dark_factory (0.39 ms on
> reify), so the full fan-out is ~9.0 s against ~7.0 s before (dark_factory)
> and ~8.8 s against ~3.9 s (reify).  Seconds, bounded, and it is the price
> of the answer being right; it is not a liveness risk at these sizes.

### Keyset for `retrieve_episodes` does not reopen that decision

Task 4869 pages `retrieve_episodes` with a keyset cursor. That does NOT reverse the "keyset declined" verdict above. The verdict was a latency argument against adding a second mode INSIDE `_paged_ro_query`. The episode read goes through graphiti-core rather than `ro_query`, and `EpisodicNode.get_by_group_ids(limit, uuid_cursor)` is the only paging that API exposes. It was also chosen for its termination proof, not for latency: the reader stops only on an EMPTY page, so a server cap below the page size cannot truncate it, and it needs neither a refusal guard nor a census.

### Task 4869's reads: paging cost UNMEASURED

`query_stale_node_embeddings`, `query_stale_edge_embeddings`, `query_edges_by_time_range` and `retrieve_episodes` have no measured paging cost. All four are maintenance or cold paths (reindex, stale-edge cleanup, episode listing). The architect tried a single-page timing probe against production `dark_factory` on 2026-09-24 (SKIP/LIMIT over `name_embedding`). It produced no output within 300 s and was killed (exit 124), and FalkorDB answered a trivial count in 4 ms right afterwards. The cause is unknown. Hypothesis: contention around a fused-memory service restart at 09:22:44 that day. Treat the cost as unknown until someone measures it. Do not estimate it from the 4340 tables.

The two vector reads cut each page in `WITH ... ORDER BY ... SKIP ... LIMIT` BEFORE `RETURN` projects the embedding. The intent is that each page's sort carries node or edge refs and materialises only `page_size` vectors, not every matched vector. This is REASONED, NOT MEASURED. The throwaway-graph live test confirms only that FalkorDB accepts both templates and pages correctly through them.

## Per-read classification (post-4869, re-audited 2026-09-24)

Method: `grep -n "ro_query(\|get_by_group_ids\|\.execute_query(" fused-memory/src/fused_memory/backends/graphiti_client.py`, then classify every call site. No read fits none of the categories below, so none is AT RISK.

### PAGINATED via `_paged_ro_query` (SKIP/LIMIT pages plus a `count(*)` census)

- `get_all_valid_edges` -> `enumerate_all_valid_edges` (task 4340)
- `list_entity_nodes` -> `enumerate_entity_nodes` (task 4340)
- `query_stale_node_embeddings` (task 4869). A truncated read made an embedding-dimension migration look COMPLETE when it was not. The operator's evidence of success was the very read being truncated.
- `query_stale_edge_embeddings` (task 4869). It has no `invalid_at` filter, so it includes superseded edges, which still need re-embedding.
- `query_edges_by_time_range` (task 4869). It is bounded only by the caller's window, and its consumer (`CleanupManager.find_stale_edges`) feeds `bulk_remove_edges`.

All five apply the shared `_apply_incompleteness_policy`: STRUCTURAL kinds raise `IncompleteEnumerationError`, EMPIRICAL kinds warn and return. The three 4869 reads and `enumerate_entity_nodes` dedup re-emitted boundary rows with `_first_row_per_uuid`.

### PAGINATED via keyset through graphiti-core

- `retrieve_episodes` -> `_read_all_group_episodes` (task 4869). The truncation was the worst shape of this bug. graphiti-core orders by uuid DESC, so a capped read dropped the lowest uuids. The Python `created_at` sort then picked the most recent of the survivors: the WRONG episodes, not merely fewer. Page-budget exhaustion raises rather than returning a uuid-ordered prefix.
  - What it costs (reasoned from the query text, not measured). Every keyset page runs graphiti-core's `MATCH (e:Episodic) WHERE e.group_id IN $group_ids AND e.uuid < $uuid RETURN DISTINCT ... ORDER BY uuid DESC LIMIT $limit`. That matches and sorts every episode below the cursor, `content` included. The read is therefore O(P · N log N) in the database and O(N) in transfer, and it grows with the group's episode count. The caller wants only `last_n` episodes, which `fused-memory/src/fused_memory/server/tools.py::get_episodes` caps at 1000.
  - DECLINED ALTERNATIVE: a bounded read ordered by `created_at`. Before task 4869, a comment in `retrieve_episodes` named the cheaper fix: our own Cypher with `ORDER BY e.created_at DESC, e.uuid DESC LIMIT $last_n`. With `last_n` at most 1000, below the cap, it would need no paging at all. It would make `_read_all_group_episodes` unnecessary and turn the O(N) transfer into an O(`last_n`) one. The task-4869 plan declined it for three reasons:
    1. It hard-codes graphiti-core's episode RETURN projection and record decoder (`get_episodic_node_from_record`) in this repo. The keyset path calls the public `EpisodicNode.get_by_group_ids` and inherits both.
    2. The database would sort `created_at` values as stored. `_as_sortable_utc` reads a naive datetime as UTC and sorts a missing `created_at` last. The ordering tests for tasks 2055 and 2079 (`fused-memory/tests/test_get_episodes_ordering.py`, `fused-memory/tests/test_get_episodes_ordering_residual.py`) pin that behaviour, and a server-side sort would drop it.
    3. Its correctness rests on the 1000 bound, which lives in `tools.py` and not in `retrieve_episodes`. Another caller that passed a `last_n` above the cap would be silently truncated again.

    Re-open it only with a measurement showing that this cold-path read's cost matters, and with an answer to reasons 2 and 3.

### UNPAGINATED BY DESIGN, warns at the cap

- `find_entity_nodes_by_name_substring` (task 5264). Its survivor-first `ORDER BY` cannot double as the total `ORDER BY n.uuid` that SKIP/LIMIT paging needs, and a selective substring returns a handful of rows. It logs a WARNING when a result reaches `_RESULTSET_SIZE`.

### ASSESSED SAFE, with the reason (a bare list would not be checkable)

- Every uuid-keyed lookup: the key is unique, so the result is 0 or 1 rows. Examples: `redact_episode_content`, both probes in `reassign_edge`, the `delete_entity_node` pre-check, `get_node_text`, `get_edge_text`, `get_edge_invalid_at`, and the per-link existence probe in `redirect_node_mentions`, which also carries `LIMIT 1`.
- Every exact-name lookup: bounded by the duplicate-name count, which `find_duplicate_entity_nodes` reports in single digits. Examples: `resolve_entity_by_name`, `get_nodes_by_exact_name`, `find_duplicate_entity_nodes`.
- Every single-row aggregate: one row by construction. Examples: `count_foreign_relationships` (added 2026-08-25), `node_count`, and the `_census_count` probe itself.
- Server-side grouped or filtered aggregates that SCAN the whole graph but return a small RESULT set. The cap applies to rows RETURNED, not rows scanned, so a `count`/`collect` that folds 20k rows into a handful is safe. Example: `_scan_duplicate_entity_names`, one row per duplicated name.
- `CALL db.indexes()` (`list_indices`): one row per index, single digits.
- Every per-node neighbourhood read. `get_valid_edges_for_node` is named because task 4340 asked about it specifically. Its row count is ONE node's valid degree, and a single node would have to hold more than 10000 of the graph's ~12506 valid edges (2026-08-17) to reach the cap. It is left unpaginated deliberately. Others: `get_connected_entity_uuids`, `dedup_valid_edges_for_node`, and the MENTIONS enumeration in `redirect_node_mentions` (added 2026-09-21).
  - `redirect_node_mentions` is the one per-node read whose ceiling grows with the graph. Its row count is one entity's incoming MENTIONS, which cannot exceed the graph's Episodic population: 3576 on `dark_factory` and 4851 on `reify` (label counts, 2026-09-24). A truncated enumeration would leave links behind for the caller's DETACH DELETE to destroy. Re-check this read when an Episodic label count approaches the cap.

## History: the task-4340 compounding hazard

Moved from the task-4340 comment block:

> FIXED HERE — both were measurably truncated, and they COMPOUND:
> ``detect_stale_with_edges`` calls them on consecutive lines and the two
> truncations were INDEPENDENT, so an entity surviving the node cut could
> still lose every edge to the edge cut, yielding a bogus "stale, zero valid
> facts" verdict that ``rebuild_entity_from_edges`` then WROTE BACK into
> ``n.summary``.  Corrupting, not merely under-reporting.
>   - get_all_valid_edges  -> enumerate_all_valid_edges
>   - list_entity_nodes    -> enumerate_entity_nodes
> The old names survive as thin shims with UNCHANGED signatures applying a
> SPLIT incompleteness policy: they RAISE IncompleteEnumerationError on a
> STRUCTURAL incompleteness (a read that was never validly performed — its
> emptiness or prefix is fabricated, and returning it is what let '' be
> written back over real summaries) and WARN-and-return on an EMPIRICAL one
> (a census disagreement, transient on a continuously-written graph).  The
> completeness signal itself is a first-class return value on the enumerate_*
> methods, which never raise.  No consumer ACTS on it yet — the two
> reconciliation sweeps and cleanup_count_snapshots still call the shims, so
> they cannot yet distinguish "swept a complete corpus" from "swept what we
> could fetch".

Wiring that signal through to the consumers is the first open ticket under TICKETS.

### A third consumer, outside `graphiti_client.py`

`fused-memory/scripts/measure_plural_enum_guard_recall.py` (task 4576) pages through `_paged_ro_query` and composes both of its Cypher strings from `_ALL_VALID_EDGES_MATCH`. The read-only recall probe therefore measures the same population this module enumerates. It supplies its OWN projection and its own `count(DISTINCT e.uuid)` census, so it decides completeness in distinct EDGES rather than in rows, and it derives its own verdict instead of reading `paged.complete`. Nothing in `graphiti_client.py` changes on its account.

## OPEN RESIDUAL, left open deliberately

A materially-short `INCOMPLETE_SHORT_READ` still returns a partial collection. The `force=True` path of `fused-memory/src/fused_memory/services/memory_service.py::MemoryService.rebuild_entity_summaries` will write it back, blanking the summary of any entity whose edges fell in the missing remainder. That path never consults staleness, so it writes to every entity the node read returned.

Do NOT close it by tightening the shims. Guard 4 fires on any shortfall at all, including a single concurrently-invalidated edge, so raising there would take down the live rebuild for exactly the transient that the warn-not-raise decision rejected. The fix belongs at the consumer: a policy on how short is too short, applied where the destructive write is decided. That is the same code the first ticket below must touch.

## TICKETS

Every id below is a TICKET id, not a task id. The curator resolves tickets to tasks asynchronously.

- `tkt_0RSJP8CH1M9GAAJTABV8FZB4AH`: wire the completeness signal (`PagedRead`) through to consumers, moving the incompleteness policy out of `_apply_incompleteness_policy` to where the write is decided. Status: OPEN.
- `tkt_0RSJP92VQNATQB0FSR20YMXGW8`: re-measure the task-2613 stale-status miss rate against the now-complete corpus. It was computed against a truncated denominator. Status: OPEN.
- `tkt_0RSJP82N82SNKT2BHRT3HWK3DA`: paginate the four reads the 2026-08-17 audit left AT RISK. Status: RESOLVED by task 4869.
- `tkt_0RSKFG5RX196H9CJ0RXGJCZF4F`: move this audit out of source into a reference doc. Status: RESOLVED by task 4869 (this file).
