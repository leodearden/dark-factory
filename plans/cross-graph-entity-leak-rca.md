# Cross-graph entity leak — root-cause analysis

**Status:** RCA complete (verified 2026-07-06 by a 5-agent research pass; supersedes and preserves the forensics from task 2115, now cancelled in favour of this doc + filed remediation).
**Remediation:** Phase 0a filed as a standalone task (see §8); Phases 0b/1/2 via `plans/cross-graph-entity-leak-prd.md` (this doc is the "design/RCA" input the PRD builds on; it replaces the aspirational `cross-graph-entity-leak-design.md` name from 2115's metadata).

## 1. Summary

Entity/edge/episode writes for one project physically land in **another project's FalkorDB graph key**, while keeping the correct `group_id` *property*. Two independent, confirmed root causes:

1. **Shared-driver race** — `graphiti_core` mutates the one shared client's driver inside `add_episode`; concurrent cross-project writes misroute (and can even split one episode across two graphs).
2. **Un-normalized `project_id` on the memory path** — divergent spellings (`dark_factory` / `dark-factory` / `-home-leo-src-dark-factory`; `know_live` / `know-live` / `knowlive`) become distinct graph keys and Qdrant collections.

The leak is **active**: the newest misroute (reify episode → `dark_factory` graph, 2026-07-06T11:02Z) was produced by the fused-memory instance running at the time of this RCA. Cadence ≈ one misrouted episode/day, always whole-episode batches.

## 2. Blast radius (empirical, read-only GRAPH.RO_QUERY, re-scanned 2026-07-06)

- **2,352 foreign Entity nodes** (group_id ≠ containing graph key) across 9 populated graphs, bidirectional. Of the original scan's ~2,361: **1,550 exist ONLY in the wrong graph** (destroyed by any naive "delete foreign group_ids" purge); 811 duplicate a home node, and the wrong-graph copy may hold **unique edges** (e.g. node `f02a32ea` "orchestrator": degree 143 in reify, 5 different edges in dark_factory, 0 in know_live).
- **~1,489+ foreign RELATES_TO edges** (4 worst graphs: reify 693, dark_factory 458, know_live 233, autopilot_video 105); each carries its own `fact_embedding` vecf32.
- Foreign Episodic nodes: reify 113, dark_factory 76, know_live 39, autopilot_video 18.
- Worst graphs: reify holds **952 foreign** entities (dark_factory 626, know_live 144, knowlive 60, my_solar_challenge 44, autopilot_video 36, solar_challenge_platform 34, know-live 8); dark_factory holds 793 (reify 707); reify's own nodes displaced elsewhere ≈ 1,124 (≥709 exist ONLY in a foreign key).
- `pump_web_ui` / `pump-web-ui` are 0% foreign — only projects that ran **concurrently** are contaminated (corroborates root cause 1).
- **`knowlive`'s data exists ONLY as misroutes** — its own graph key is empty; 66 `knowlive`-tagged entities live inside know_live/reify.
- Graph-key inventory shows the normalization families directly: `dark_factory`(9,329) / `dark-factory`(0) / `-home-leo-src-dark-factory`(0); `know_live`(2,454) / `know-live`(104) / `knowlive`(0); `pump_web_ui`(44) / `pump-web-ui`(6); plus empty junk keys `my-project`, `default`, `test-project`, `_probe`, `_probe_check`, `1098`.

## 3. Root cause 1 — shared-driver mutation race (THE leak)

Installed wheel: `graphiti-core 0.28.2` at repo-root `.venv/lib/python3.13/site-packages/graphiti_core/` (uv **workspace** venv — not `fused-memory/.venv`). The `graphiti` git submodule is **inert**: `[tool.uv.sources]` does not redirect `graphiti-core` (fused-memory/pyproject.toml:26-28; uv.lock pins the PyPI registry) — patching the submodule does nothing.

`graphiti_core/graphiti.py:881-890`, inside `add_episode`:

```python
if group_id != self.driver._database:
    self.driver = self.driver.clone(database=group_id)   # :889
    self.clients.driver = self.driver                     # :890
```

- Mutates **both** `self.driver` and the shared `GraphitiClients.driver`. Same pattern in `add_episode_bulk` (:1112-1116; fused-memory never calls it).
- fused-memory holds **one shared Graphiti client** (`graphiti_client.py:227`, built once at `:343-348`).
- The durable queue runs up to **3 different project groups concurrently** (`durable_queue.py:322`; `semaphore_limit=3`, `workers_per_group=1` — config.yaml:53-54). Because same-group writes are already serial, **cross-group interleaving is the only concurrency the queue has** — every concurrent pair is a potential misroute.
- **Worse than last-writer-wins:** every pipeline helper re-reads `clients.driver` at call time (~49 reads in graphiti.py), so a concurrent `add_episode(group_B)` redirects group A's pipeline **mid-flight** — a single episode's writes can split across two graphs.
- **Persistent post-return:** the driver stays pointed at the last-written group's graph. Any later shared-client op without an explicit `driver=` targets the wrong graph. Latent in-tree victim: `GraphitiBackend.build_communities` (`graphiti_client.py:617-623`) omits `driver=` (currently uncalled in production code).
- `GraphitiBackend.add_episode` (`:366-395`) has **no way to override the driver**: upstream `Graphiti.add_episode` accepts **no `driver` parameter in any released version or current main** (reads — `search`/`search_`/`retrieve_episodes`/`build_communities` — do accept `driver=`; fused-memory's search paths already pass it and are safe).
- Data signature matches: misroutes arrive as whole episodes (Episodic + entities + edges, microsecond-clustered) under the wrong graph key with the **correct `group_id` property**.

### Upstream status (as of 2026-07-06)

- Bug present **verbatim** in 0.28.2, 0.29.0/0.29.1/**0.29.2** (latest), and current `main` — upgrading buys nothing.
- Filed upstream: **issue #1331** ("add_episode() mutates shared self.driver causing cross-group data contamination", open, no maintainer response). Community fix PRs **#1294** and **#1305** (local-driver clone approaches) open and unreviewed ~4 months. Sibling read-side routing bugs: #1161, #1325/PR #1326, PR #1572.
- Upstream docs recommend group_id-per-tenant on one client — exactly the pattern that races on FalkorDB — with no concurrency guidance.
- `Graphiti.__init__` accepts `graph_driver=` (supported API) and does **no I/O** (no index build; telemetry is gated/swallowed) → the per-group-client fix (§8) needs no upstream change and survives any future upgrade. 0.28.2→0.29.2 breaking changes, if ever upgrading: 0-based episode indices, first-class-field attribute-merge semantics, 250-char attribute cap (0.29.1).

## 4. Root cause 2 — memory path never normalizes `project_id`

- Task path normalizes via `resolve_project_id` (`models/scope.py:117-135`: lower + dash→underscore; ~20 call sites in the interceptor). Memory tools pass `project_id` **verbatim**: `server/tools.py` add_episode/add_memory/search/get_entity → `memory_service.py` (zero `resolve_project_id` references) → `Scope.graphiti_group_id` returns raw (`scope.py:209-212`) → `_driver_for`/`_graph_for` use it verbatim as the FalkorDB graph name (`graphiti_client.py:236-252`).
- `validate_project_id` (`utils/validation.py:98-105`) **allows hyphens**.
- Mem0/Qdrant mirror: `Scope.mem0_collection_name` = `f'{prefix}_{project_id}'` (`scope.py:214-216`) → `fused_dark-factory` vs `fused_dark_factory`; `mem0_user_id` (raw) co-diverges.
- **Partial mitigation already landed (task 1549):** the known-project gate (`server/tools.py:558`, `utils/validation.py:173+`) **rejects** write-tool calls whose project_id isn't a normalized registry key — on the five gated tools (add_episode, add_memory, delete_memory, delete_episode, update_edge), and only when `DASHBOARD_KNOWN_PROJECT_ROOTS` is set. **Ungated:** all read tools (silently read the wrong/empty graph) and the graph-mutating tools refresh/set_entity_summary, rename_entity, merge_entities, delete_entity, rebuild_entity_summaries, replay_to_graphiti. Live stray graphs predate the gate or came via ungated paths.
- **Double-prefix Qdrant collections (`reify_reify`, `reify_`, `autopilot_video_autopilot_video`, `fused_fused_memory`) are NOT a code bug** — legacy per-project deployments set `collection_prefix` to the project name (e.g. `/home/leo/src/reify/fused-memory-config.yaml:43` `collection_prefix: "reify"`); the newer reify config already uses `"fused"`. Needs data cleanup, not a code fix. Note `_list_project_collections` strips only `fused_`, so `reify_*` collections are invisible to `get_status` discovery.
- One normalization fix already shipped in-tree as the pattern to copy: `get_external_statuses` inlines `project_id.lower().replace('-','_')` (`tools.py:2543-2546`).

### Key-derivation chokepoints (for the Phase-0b fix)

No single chokepoint exists. Three seams together cover everything:
1. **MCP boundary** — normalize in every memory-tool prologue (~17 sites, mechanical; extend the existing `validate_project_id` prologue). The known-project gate then checks the normalized form, so `'dark-factory'` starts **working** instead of being rejected (deliberate behavior flip).
2. **`Scope` field_validator** (`models/scope.py:202-224`) — covers `graphiti_group_id`, `mem0_collection_name`, `mem0_user_id` for all Scope-mediated traffic (all of Mem0). Insufficient alone: `memory_service.py` bypasses Scope with raw `group_id=project_id` on ~10 paths (:1872-2135).
3. **GraphitiBackend argument normalization** — normalize `group_id`/`group_ids` at backend method entry (not just `_driver_for`), otherwise graph *name* and node `group_id` *property*/search filters diverge. This is also the only seam that covers **durable-queue replay**, which re-executes persisted raw `payload['group_id']` (`durable_queue.py:52`; `memory_service.py:940-967`).

**Hazards:** path-shaped ids (`-home-leo-src-dark-factory`) must be **LOUD-rejected, not normalized** — normalization yields `_home_leo_src_dark_factory`, a *new* wrong key (reuse `is_path_shaped_name`, `scripts/investigate_cross_graph_duplication.py:65-79`). `knowlive` (no separator) cannot be fixed by normalization — purge/consolidation path only. On flip day, populated hyphen-form keys (`know-live` graph: 104 entities; `pump-web-ui`: 6; Qdrant `fused_dark-factory` etc.) become **unreachable via API** — they must be merged/purged in the same change window. Canonical form is unambiguously the underscore form (registry keys, config.yaml:77 comment, factory-init's hyphen-free rule) — no correctly-stored project breaks.

## 5. Harm

- Search **results** are safe (all search paths filter `WHERE n.group_id IN [pid]` — `search_utils.py:217/601/689`): foreign nodes in a graph don't surface.
- **Recall loss:** ≥709 reify-only nodes displaced into other keys are invisible to reify search.
- **Graph-wide helpers with NO group_id filter are corrupted:** `rebuild_entity_summaries`, `list_entity_nodes`, `get_all_valid_edges`, `find_duplicate_entity_nodes` (name-keyed → cross-project false "duplicates" — the likely esc-4995 trigger), `get_status` node_count (reify inflated by ~950).
- **W6 interaction (defused 2026-07-06):** the W6 write-time identity gate (tasks 2198/2202) would have **destructively auto-merged** misrouted foreign nodes as exact-name duplicates — permanent cross-project edge contamination + loss of migration evidence. Defused before dispatch: 2198 amended (mandatory `n.group_id = $group_id` predicate on `get_nodes_by_exact_name` + `find_duplicate_entity_nodes`, plus a seeded-foreign-node test); 2210 got the matching advisory (filtered startup scan; cross-group collisions route to this remediation, never auto-merge). **Verify the landed 2198 diff carries the predicate.**

## 6. Remediation phases (strict order)

- **Phase 0a — stop the race** (filed as a standalone task; see §8). Blocking for everything: migrating first re-contaminates.
- **Phase 0b — normalization** (PRD): three-seam normalization per §4, LOUD-reject path-shaped ids, tighten/keep `validate_project_id`, mirror on Mem0 collection derivation, handle flip-day data (see hazards).
- **Phase 1 — migration** (PRD): build the **missing cross-graph move primitive** + dry-run-first script modeled on `fused-memory/scripts/purge_knowlive_namespace.py` (dry-run default; JSON manifest = the recovery record; destructive steps flagged; its docstring states outright there is no clean re-key primitive). **Validated primitive** (proven byte-identical on throwaway graphs): read embeddings via `GRAPH.RO_QUERY --compact` (exact float32 strings — the textual/decoded form truncates to 6 decimals, LOSSY); recreate in target graph via `CREATE … SET name_embedding=vecf32([exact…])` / `fact_embedding=vecf32([exact…])`; reattach edges + episodes; then `DETACH DELETE` from source. Per misrouted node: **MOVE** if displaced-only (1,550 nodes — never purge); **MERGE unique edges into the home copy then delete** the wrong-graph copy if duplicate-uuid. Existing tools cannot do this — `merge_entities` is single-graph (`graphiti_client.py:1013`); delete/rename address graphs by request param. Run with the write queue quiesced; paged row queries only (`collect()` truncates); emit the manifest before `--apply`; re-query after-counts.
- **Phase 2 — consolidation** (PRD): canonical = underscore form. Consolidate know-live/know_live/knowlive and pump-web-ui/pump_web_ui; strand-handle orphan group_ids with no home graph; `GRAPH.DELETE` empty stale keys (`dark-factory`, `-home-leo-src-dark-factory`, `my-project`, `test-project`, `default`, `1098`); mirror on Qdrant collections (incl. the legacy double-prefix `reify_*` / `autopilot_video_*` / `fused_fused_memory` cleanup, `scripts/cleanup_test_collections.py` pattern).

Deploy note: the running service only picks up Phase-0a after an out-of-cgroup `systemctl --user restart fused-memory.service` (never `restart-fused-memory.sh --drain` — hangs, task 2090); the same restart brings the already-merged 2111 sibling-restore guardrail live.

## 7. Fix option space for Phase 0a (evaluated)

1. **Per-group Graphiti client cache — CHOSEN.** Cache `dict[group_id, Graphiti]` beside `_cloned_drivers`; each built with `Graphiti(graph_driver=self._driver_for(group_id), llm_client=<shared>, embedder=<shared>, cross_encoder=<shared>)`. `__init__` does no I/O; `FalkorDriver.clone` is a cheap shallow copy sharing the one connection (`falkordb_driver.py:307-320`; `_MultiTenantFalkorDriver.clone` preserved, `graphiti_client.py:210-214`). **Stability property:** the per-group client's driver already has `_database == group_id`, so the upstream mutation branch is never taken — race structurally unreachable, zero patching. Caveat: pass a single shared `cross_encoder` explicitly or each instance builds its own default `OpenAIRerankerClient`. ~15 lines + tests.
2. Whole-call `asyncio.Lock` around `add_episode` — correct but serializes ALL projects' episode writes (5-15 LLM calls each; ~3× throughput loss). A narrower entry-lock is **insufficient** (mid-flight `clients.driver` reads). Stopgap only.
3. Per-call `driver=` override — **does not exist** for the write path in any version. Dead end.
4. Property/contextvar shim on `Graphiti.driver` — deep surgery on two attributes; dominated by option 1.
5. Upgrade/vendor upstream — no fixed release exists; vendoring PR #1294/#1305 means carrying a fork. Not needed given option 1.

## 8. References

- **Tasks:** 2115 (origin, cancelled → this doc), 2116 (diagnostic, done — `scripts/investigate_cross_graph_duplication.py`), 1549 (known-project gate), 2073/2118 (dedup lineage), 1937 (knowlive purge), 937 (dash/underscore precedent), 2086/1684 (uuid tools), 2217 (upstream watch, separate dedupe_edges bug), W6 = 2198/2202/2207/2210/2213 (identity gate; PRD `plans/fm-memory-identity-prd.md` scopes this leak out).
- **Escalations/obs:** esc-4995-1, reify obs `d5ae1a53`; example leaked node `f02a32ea`.
- **Key files:** `fused-memory/src/fused_memory/backends/graphiti_client.py`, `models/scope.py`, `utils/validation.py`, `services/durable_queue.py`, `services/memory_service.py`, `server/tools.py`, `fused-memory/scripts/purge_knowlive_namespace.py`.
- **Upstream:** github.com/getzep/graphiti issues #1331, #1161, #1325; PRs #1294, #1305, #1326, #1572.
- Full forensic scripts: reify session scratchpad 2026-07-06; live-state re-scan: this session (dark_factory, 2026-07-06).
