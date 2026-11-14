# FalkorDB stopword IN-list crash: measurements and code-site audit

This is the one place where the measurements and the per-site verdicts for the FalkorDB#3348 crash are written down: where the stopword indices sit, which query shapes crash and which survive, and every Cypher site that fused-memory runs or reaches. The rule that follows from them is in `OPERATIONS.md` §"FalkorDB settings and the recreate gate" → "Experiments never touch production FalkorDB", which points here and does not restate them. That way a re-measurement is a one-file edit.

Every figure below is dated. Each verdict gives its reason, so that it can be checked and proved wrong rather than taken on trust. Written by task 7794 (2026-11-14). The architect took the measurements. The implementer added the `GRAPH.RO_QUERY` row of the matrix.

**Bottom line (2026-11-14): no site is reachable today, so there is nothing to fix in-tree.** Five fused-memory symbols, two committed scripts and three graphiti_core sites have the crashing shape. Each one is safe only because of where its values come from. The group_id sites stop being safe if a project whose id is a stopword is ever registered (see "The latent condition").

## The hazard

FalkorDB#3348 is in the pinned image: FalkorDB 4.18.0, `falkordb/falkordb@sha256:58a95a33d591b25648bbb13268a23329aa39e70b4cbeff0752c65f8b4beba6db`. An IN-list over an indexed property goes to the index as a union of term nodes. For a value on the label's stopword list, `RediSearch_CreateTokenNode` returns NULL, and `QueryNode_AddChildren` dereferences that NULL: the server segfaults. No 4.x release fixes it. The 6.0 engine fixes this crash, and the move to it is gated by task 7795.

It has already crashed production. On 2026-10-04, an ad-hoc Entity read `MATCH (n:Entity) WHERE n.uuid IN ['a','b'] RETURN count(n)` segfaulted `docker-falkordb-1` (esc-infra-1). It is a pure read, so read-only access is no protection.

## Where it bites (measured 2026-11-14)

Read-only `docker exec docker-falkordb-1 redis-cli GRAPH.RO_QUERY dark_factory 'CALL db.indexes() ...'` against production showed the stopword list on three labels:

| label | indexed properties |
|---|---|
| Entity | uuid RANGE, created_at RANGE, group_id RANGE+FULLTEXT, name RANGE+FULLTEXT, summary FULLTEXT |
| Episodic | created_at RANGE, group_id RANGE+FULLTEXT, uuid RANGE, valid_at RANGE, content FULLTEXT, source FULLTEXT, source_description FULLTEXT |
| Community | uuid RANGE, name FULLTEXT, group_id FULLTEXT |

RELATES_TO (even with FULLTEXT on name, fact and group_id), MENTIONS, HAS_EPISODE, NEXT_EPISODE, HAS_MEMBER and Saga report stopwords `[]`.

The list is `graphiti_core.driver.falkordb.STOPWORDS`, which graphiti-core 0.28.2 passes to `createNodeIndex`. It has 33 words: `a is the an and are as at be but by for if in into it no not of on or such that their then there these they this to was will with`. The match ignores case: `'The'` crashes like `'the'`.

The list is set per label, not per property. Entity's `uuid` has only a RANGE index and is in no fulltext index, yet an IN-list on it crashed. Whether an IN-list on a property with no index at all (for example Entity `summary` on its own) crashes was not measured.

`GRAPH.LIST` on production (2026-11-14): no graph is named after a stopword. The graphs were `dark_factory`, `reify`, `know_live`, `knowlive`, `solar_challenge`, `solar_challenge_platform`, `market_predictor`, `pump_web_ui`, `scoping_intake`, `stream_touch`, `autopilot_video`, `default_db`, `evalmem_*`, `_test_*`, `_probe*`, `probe_e1_*` and `sweep_e4_*`.

## Crash-shape matrix (measured 2026-11-14)

Fixture: Entity with graphiti's RANGE index on uuid, group_id, name and created_at, plus the stopword FULLTEXT index on name, summary and group_id. RELATES_TO has a RANGE index on uuid, group_id and name, plus a FULLTEXT index on name, fact and group_id. The graph holds one edge `(:Entity {uuid:'u1', name:'a', group_id:'g'})-[:RELATES_TO {uuid:'e1', name:'by', group_id:'g', fact:'f'}]->(:Entity {uuid:'u2', name:'by', group_id:'g'})`. Each query ran on a fresh container. A query is CRASHED when `PING` failed afterwards. In the queries below `'a'`, `'the'`, `'by'`, `'in'`, `'The'` and `'A'` are stopwords, while `'b'`, `'u1'`, `'x'` and `'g'` are not.

| CRASHED | ALIVE |
|---|---|
| `MATCH (n:Entity) WHERE n.uuid IN ['a','b']` | control: `n.uuid IN ['u1','x']` (no stopword) |
| `n.uuid IN ['a']` (a single element) | `n.uuid = 'a'` (a single equality) |
| `n.uuid IN ['u1','the']` (one stopword among real values) | `n.name = 'a'` |
| `n.uuid IN ['The','A']` (the match ignores case) | `MATCH (n:Entity {uuid: 'a'})` (a property map) |
| `CYPHER uuids=['a','b'] ... WHERE n.uuid IN $uuids` (the parameter form graphiti's driver sends) | `UNWIND ['a','b'] AS u MATCH (n:Entity {uuid: u})` |
| `n.uuid = 'a' OR n.uuid = 'b'` | `UNWIND ['a','b'] AS u MATCH (n:Entity) WHERE n.uuid = u` |
| `n.name IN ['a','by']` | `MERGE (n:Entity {uuid: 'a'})` |
| `n.group_id IN ['in','g']` | `toString(n.uuid) IN ['a','b','u1']` (returns the correct count, 1) |
| `n.group_id = 'g' AND n.uuid IN ['a','b']` | `CYPHER uuids=['a','u1'] ... WHERE toString(n.uuid) IN $uuids` |
| `MATCH (x:Entity {uuid:'u1'}) OPTIONAL MATCH (n:Entity) WHERE n.uuid IN ['a','b']` | `MATCH ()-[e:RELATES_TO]->() WHERE e.uuid IN ['a','b']` |
| `MATCH (n:Entity) WHERE n.name IN ['a','by'] RETURN n.uuid` (a pure read sent through `GRAPH.QUERY`) | `MATCH ()-[e:RELATES_TO]->() WHERE e.name IN ['by','a']` |
| `GRAPH.RO_QUERY ... WHERE n.uuid IN ['a','b']` (container exited 139, SIGSEGV; implementer's run) | |

The query that runs a fulltext search and then filters its YIELDed nodes with an IN-list, `CALL db.idx.fulltext.queryNodes(...) YIELD node AS n ... WHERE n.group_id IN $group_ids`, is **UNMEASURED**. That is the shape of graphiti's `node_fulltext_search`.

**Method.** The probes ran in a private throwaway container of the pinned image with no published port: `docker run -d --name <unique> <image>`, queried through `docker exec <name> redis-cli GRAPH.QUERY`, removed with `docker rm -f`. Never run a crash probe on production, and never on the shared scratch server (`:6380`), which every test session and the offline lane depend on.

## Safe forms for new code

All four were measured ALIVE above. They differ in cost:

- **A single `=`**, or **a property map** `(n:Label {prop: $v})`. Both keep the index.
- **`UNWIND $list AS u MATCH (n:Label {prop: u})`**. This keeps the index, so it is the choice for a hot lookup. A duplicate value in the list gives duplicate rows, so dedupe the list first or add `WITH DISTINCT`.
- **`toString(n.prop) IN $list`**. This is #3348's workaround. It hides the property from the planner, so the lookup becomes a label scan. Entity on `dark_factory` holds 16k+ nodes (`plans/falkordb-resultset-cap-audit.md`), and an unbounded scan can hit the 10000-row `RESULTSET_SIZE` cap.

## Site audit (2026-11-14)

The grep was `IN \$|IN \[|IN \{`, plus OR-of-equality chains, over `fused-memory/src` and `fused-memory/scripts`. The only OR-equality hit is SQL in `reconciliation/journal.py`, not Cypher. graphiti_core 0.28.2 was traced from every entry point fused-memory calls.

### fused-memory's own Cypher

| site | filter | label | value source | verdict | reason |
|---|---|---|---|---|---|
| `fused-memory/src/fused_memory/backends/falkor_edge_search.py::_bfs_reached_edges`, `::_bfs_reached_nodes` | `origin.uuid IN $bfs_origin_node_uuids` | Entity, Episodic | BFS origin uuids | affected shape, NOT REACHABLE | These BFS legs never run. fused-memory uses only `EDGE_HYBRID_SEARCH_RRF` (`Graphiti.search`) and `NODE_HYBRID_SEARCH_RRF` (`Graphiti.search_`), and neither includes BFS. fused-memory never passes `bfs_origin_node_uuids` or `center_node_uuid` (`backends/graphiti_client.py::GraphitiBackend.search` accepts `center_node_uuid`, but no caller passes it). Even if BFS ran, the origins would be uuid4 strings read from the graph. |
| `falkor_edge_search.py::FalkorEdgeSearch.node_bfs_search` | `n.group_id IN $group_ids`, `origin.group_id IN $group_ids` | Entity, Episodic | `[project_id]` | affected shape, NOT REACHABLE | Not executed, for the BFS reason above. If it ran, the list would hold only the queried graph's own name (see the group_id note below). |
| `backends/exact_episode_window.py::ExactEpisodeWindowGraphiti.retrieve_episodes` | `e.group_id IN $group_ids` | Episodic | `[project_id]` | affected shape, NOT REACHABLE today | See the group_id note. Latent (see "The latent condition"). |
| `backends/recent_episodes.py::_RECENT_GROUP_EPISODES_CYPHER` | `e.group_id IN $group_ids` | Episodic | `[project_id]` | affected shape, NOT REACHABLE today | See the group_id note. Latent. |
| `falkor_edge_search.py::FalkorEdgeSearch.edge_fulltext_search`, `::edge_similarity_search`, `::edge_bfs_search` | `e.group_id IN $group_ids`, and the ORDER BY `e.uuid IN $promoted_uuids` | RELATES_TO | `[project_id]`, promoted edge uuids | NOT AFFECTED | RELATES_TO carries no stopwords and an edge IN-list was measured ALIVE. The ORDER BY one is an expression, not a filter. |
| `backends/falkor_graph_operations.py::FalkorGraphOperations.edge_get_by_uuids` | `e.uuid IN $uuids` | RELATES_TO | edge uuids | NOT AFFECTED | edge label |
| `backends/graphiti_client.py::GraphitiBackend.bulk_remove_edges` (count and delete), `::get_edges_by_uuids` | `e.uuid IN $uuids` | RELATES_TO | edge uuids | NOT AFFECTED | edge label |
| `services/memory_service.py::rebuild_entity_summaries` | none in Cypher | Entity | caller-supplied `entity_uuids`, not validated as UUIDs | NOT REACHABLE | It filters the caller's uuids in Python against a full read. No Cypher IN-list is built from them. |

**The group_id note.** The `group_ids` value is `[project_id]`. `graphiti_core.decorators.handle_multiple_group_ids` fans a multi-group call out as `[gid]` on graph `gid`, so the list only ever holds the name of the graph being queried. Stopword indices exist only on graphs that fused-memory provisions. `GraphitiBackend._ensure_indices` provisions only `_registered_graph_ids`, derived from `configured_known_projects_map`. `_MultiTenantFalkorDriver` suppresses graphiti's own auto-indexing. No registered graph, and no existing graph, is named after a stopword.

### graphiti_core 0.28.2 sites reached from fused-memory

All three are NOT REACHABLE by a stopword:

- **`graphiti_core/search/search_utils.py::node_similarity_search`**, `n:Entity` with `n.group_id IN $group_ids`. Reached from MCP `get_entity` → `MemoryService.get_entity` → `GraphitiBackend.search_nodes` → `Graphiti.search_` → `FalkorEdgeSearch.node_similarity_search` (which uses the built-in search), and from `add_episode` → `graphiti_core/utils/maintenance/node_operations.py::resolve_extracted_nodes` → `_collect_candidate_nodes`. The value is `[project_id]`, the graph's own name.
- **`search_utils.py::node_fulltext_search`**, the same group_id filter applied after the YIELD. Same paths, same values. Its post-YIELD shape is the UNMEASURED one above, and its value is safe either way.
- **`graphiti_core/nodes.py::EntityNode.get_by_uuids`**, `n:Entity` with `n.uuid IN $uuids`. Reached from `graphiti_core/utils/maintenance/edge_operations.py::resolve_extracted_edges` on the `add_episode` path. `FalkorGraphOperations` does not override `node_get_by_uuids`. The values are edge endpoint uuids, either minted by uuid4 or read from the graph.

### graphiti_core sites not reached by any fused-memory path

- `nodes.py::EpisodicNode.get_by_uuids`: `add_episode` is never passed `previous_episode_uuids`.
- `utils/maintenance/graph_data_operations.py::retrieve_episodes`: overridden by `ExactEpisodeWindowGraphiti.retrieve_episodes`.
- `nodes.py::EntityNode.get_by_group_ids` and `utils/maintenance/community_operations.py::get_community_clusters`: reached only through `Graphiti.build_communities`. `GraphitiBackend.build_communities` has no caller, so this is latent.
- `nodes.py::Node.delete_by_uuids`: only `Graphiti.remove_episode` calls it. `GraphitiBackend.remove_episode` uses `get_by_uuid` plus `Node.delete`, and both match with a property map.
- `nodes.py::EpisodicNode.get_by_group_ids`, `::CommunityNode.get_by_group_ids`, `::CommunityNode.get_by_uuids`: no fused-memory caller. `SagaNode`'s lookups are not affected, since Saga reports no stopwords.
- The episode and community fulltext and similarity searches in `search_utils.py`: no recipe fused-memory uses includes them.
- `search_utils.py::get_embeddings_for_nodes`, `::get_embeddings_for_communities`: only the MMR rerankers call them, and fused-memory uses none of those.
- `search_utils.py::get_mentioned_nodes`, `::get_communities_by_nodes`, `::get_edge_invalidation_candidates` (`n.uuid IN [...] OR m.uuid IN [...]`), `::get_relevant_nodes`, `::get_relevant_edges`: nothing in graphiti-core 0.28.2 calls them.
- `graphiti_core/driver/falkordb/operations/*.py`: reached only through the `Graphiti.nodes` and `Graphiti.edges` namespaces, which fused-memory never touches.
- `graph_data_operations.py::clear_data`: no fused-memory caller.
- `graphiti_core/search/search_filters.py`, the IN and OR filters: these filter edge properties only.

### Caller-supplied values

The MCP tools that take a uuid (`delete_episode`, `redact_episode_content`, `get_entity_by_uuid`, `update_edge`, and `add_episode`'s `uuid=` branch) reach graphiti only through property-map or single-value lookups. Entity names and free text reach FalkorDB only inside the RediSearch fulltext query string, never in a Cypher IN-list. No Falkor-backed site runs an IN-list without a label.

### Committed scripts that read production

These are outside the task's stated scope (`fused-memory/src`) but carry the same hazard:

| site | filter | label | value source | verdict | reason |
|---|---|---|---|---|---|
| `fused-memory/scripts/score_overreach_composite.py::EPISODE_CONTENT_CYPHER` | `e.uuid IN $uuids` | Episodic | episode uuids from flag records | affected shape, NOT REACHABLE | The values are uuid4 episode ids. |
| `fused-memory/scripts/replay_overreach_extraction.py::PREVIOUS_EPISODES_CYPHER` | `e.group_id IN $group_ids` | Episodic | the episode's own `group_id` | affected shape, NOT REACHABLE | The value is a graph name, as in the group_id note. |

The scripts' edge-only sites are not affected: `score_overreach_composite.py`'s RELATES_TO read, and `audit_unverified_completion_claims.py`'s `any(u IN $uuids WHERE u IN r.episodes)` on RELATES_TO.

## The latent condition

Every group_id site above becomes reachable if a project whose canonical id is a stopword is registered in `fused-memory/src/fused_memory/models/scope.py::configured_known_projects_map`. `GraphitiBackend._ensure_indices` would then provision stopword indices on a graph of that name, and every `[project_id]` list would hold a stopword. `fused-memory/src/fused_memory/utils/validation.py::validate_project_id` and `::canonicalize_project_id` accept `'a'`, `'it'` and `'no'`. The read tools have no registry gate, but an unregistered graph gets no indices, so reading one cannot crash. The architect filed follow-up ticket `tkt_0RWZYRKW8TXKT6CD2113N3QZER` to refuse such ids when a project is registered.

## How to re-measure

Re-run all of this after the FalkorDB upgrade gate (task 7795), after any graphiti-core bump, and after any change that adds a graphiti search recipe (BFS, MMR, episode or community search) or wires up `build_communities`.

1. **Sites**: `git grep -nE 'IN \$|IN \[|IN \{' -- fused-memory/src fused-memory/scripts`, plus a search for OR-chained equalities on one property. Then trace graphiti_core's IN sites (`grep -nE 'IN \$|IN \[' ` under the installed `graphiti_core/`) from fused-memory's entry points.
2. **Index state**: read-only, on production: `docker exec docker-falkordb-1 redis-cli GRAPH.RO_QUERY <graph> 'CALL db.indexes()'` and `docker exec docker-falkordb-1 redis-cli GRAPH.LIST`. Read only. No IN-list, no OR-equality.
3. **Shape matrix**: a private container of the pinned image, as in "Method" above. Build the fixture with `CREATE INDEX FOR (n:Entity) ON (...)` and `CALL db.idx.fulltext.createNodeIndex({label: 'Entity', stopwords: <STOPWORDS>}, ...)`. Run each query on a fresh container and `PING` afterwards, then `docker rm -f`.
