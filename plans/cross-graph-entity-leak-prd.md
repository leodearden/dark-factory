# cross-graph-entity-leak — normalization, migration, consolidation (Phases 0b/1/2)

**Status:** active — authored 2026-07-06. Approach **B + H** (high-stakes data integrity: memory-store routing + a destructive live-data migration).
**RCA (authoritative background, read first):** `plans/cross-graph-entity-leak-rca.md` (commit `5e13217a7a`). Origin: task 2115 (cancelled/superseded 2026-07-06).
**Out-of-batch prerequisite:** **task 2266** — Phase 0a (per-group Graphiti client cache; stops the shared-driver race). Already filed; this PRD's deploy capstone (δ) deploys it.

## Goal

Finish the cross-graph entity-leak remediation: (0b) make divergent `project_id` spellings impossible on the memory path, (1) move the ~2,352 misrouted entities (+ edges + episodes) home without losing a byte, (2) consolidate the divergent graph-key / Qdrant-collection families.

What an operator/agent observes when this lands:
- A memory write with `project_id='dark-factory'` **succeeds** and lands in the `dark_factory` graph (today the gated tools reject it; ungated tools misroute it); reads with either spelling return the same data. A path-shaped id (`-home-leo-src-dark-factory`) is **loudly rejected** with a specific error.
- reify search recalls the ≥709 reify nodes currently displaced into foreign graph keys; `get_status` node counts stop being inflated (~950 foreign in reify); graph-wide helpers (`rebuild_entity_summaries`, `find_duplicate_entity_nodes`) stop seeing cross-project false duplicates; W6-ε's startup dup-name alarms stop firing on 2115 artifacts.
- `GRAPH.LIST` shows one graph per project (no `know-live`/`knowlive`/`pump-web-ui` siblings, no empty junk keys); Qdrant shows one `fused_<project>` collection per project (no `fused_dark-factory`, no legacy `reify_reify`/`reify_`/`autopilot_video_autopilot_video`/`fused_fused_memory`).

## Background

See the RCA §§2–5 for blast radius, root causes, chokepoint inventory, and hazards. Load-bearing facts for this PRD:
- Root cause 1 (shared-driver race) is owned by **task 2266**, not this PRD. Migration before 2266 is deployed re-contaminates — hence the δ→η ordering.
- Root cause 2: the memory path passes `project_id` verbatim into graph keys and Qdrant collection names; there is **no single chokepoint** — three seams together cover everything (MCP tool prologues; `Scope`; `GraphitiBackend` group-arg entry, which alone covers durable-queue replay of persisted raw `group_id`s).
- The cross-graph move primitive **does not exist** (`merge_entities` is single-graph; the purge script's docstring says so outright). The vecf32 read/recreate approach is **validated byte-identical** on throwaway graphs via `GRAPH.RO_QUERY --compact` exact float32 strings (RCA §6 Phase 1) — the *decoded/textual* float form truncates to 6 decimals and is lossy.
- W6 (`plans/fm-memory-identity-prd.md`, tasks 2198/2202/2207/2210/2213) was amended 2026-07-06 with the `n.group_id = $group_id` property filter — this PRD must **not** duplicate that work, and W6's exact-name collapse no longer threatens the migration evidence.

### Substrate reality (G3 — verified against main 2026-07-06)

- `resolve_project_id` (scope.py:117-135, lower+dash→underscore) and the known-project gate (`_known_project_gate`, tools.py:558; registry keys always normalized) **exist**. In-tree normalization precedent: `get_external_statuses` (tools.py:2543-2546).
- `is_path_shaped_name` **exists** (fused-memory/scripts/investigate_cross_graph_duplication.py:65-79) — lift/share, don't reinvent.
- `scripts/restart-fused-memory.sh` **exists, executable** (repo root); its no-arg path is `systemctl --user restart` + health wait (task-2212 deploy precedent; never `--drain`).
- Dry-run-first manifest precedent **exists**: `fused-memory/scripts/purge_knowlive_namespace.py` (+ its mock-based test).
- **The fused-memory test suite is mock-based** (MagicMock graphs; no live-FalkorDB fixture — verified in tests/test_merge_entities.py, test_purge_knowlive_namespace.py). Byte-fidelity therefore CANNOT be proven by unit tests alone; per the ops-scripts lesson (mock-only tests shipped 2 broken scripts), the live gates (η, ι) mandate a **live throwaway-graph rehearsal** before any `--apply`.
- Deterministic task machinery (pure-gate preset `always_escalates=true` without `before_done`; auto-deploy preset with `before_done`) **exists** (DeterministicRunner; CLAUDE.md). Note: `before_done.script` must exist at `submit_task` time — which is why η/ι are **pure gates** (their scripts are authored by ζ/θ *after* filing) while δ can carry `before_done` (its script exists today).

## Resolved design decisions (do not relitigate)

1. **Normalize-and-accept, reject only path-shaped.** `canonicalize_project_id` = lower + `-`→`_`, raising a specific error on path-shaped input (leading `-`/`/`); hyphen spellings are silently canonicalized (matches the task path and `get_external_statuses`), and the known-project gate then checks the **normalized** form — so `'dark-factory'` flips from rejected to accepted-into-`dark_factory`. Deliberate behavior flip; unknown projects are still rejected by the gate. Path-shaped ids are **never** normalized (they'd map to a new wrong key, e.g. `_home_leo_src_dark_factory`).
2. **Three-seam defense-in-depth, all idempotent:** (i) MCP tool prologues (the boundary), (ii) a `Scope` `field_validator` (covers all Mem0 derivation: collection name + `user_id`), (iii) `GraphitiBackend` group-arg normalization at public-method entry — normalizing the `group_id`/`group_ids` **arguments** (not just `_driver_for`) so graph key, node `group_id` property, and search filters always agree; seam (iii) alone covers durable-queue **replay** of persisted raw group_ids. Triple application is safe (idempotent), and no single seam covers all callers (RCA §4).
3. **Migration = MOVE or MERGE, never purge.** Displaced-only nodes (≈1,550 — recomputed live, not baked into tests) are MOVED; duplicate-uuid pairs (≈811) have the wrong-graph copy's **unique edges merged** into the home copy, then the copy deleted. Embedding fidelity via raw `--compact` float32 strings → `vecf32([...])` recreation; edges (with `fact_embedding`) and Episodic `MENTIONS` links move with their nodes.
4. **Orphan group_ids route via an explicit, human-reviewable alias map** shipped with the script (e.g. `knowlive→know_live`, `know-live→know_live`, `pump-web-ui→pump_web_ui`); unmapped orphans are listed in the manifest as `UNRESOLVED` and block `--apply` for that node only (no silent scope drops, no silent new-graph creation).
5. **Live-data application is operator-gated, not TDD.** ζ/θ deliver *code* (script + mock-based unit tests: Cypher shapes, classification, manifest schema, create-before-delete ordering, `--compact`→`vecf32` passthrough as pure functions over recorded fixtures). The *runs* against live data are deterministic **pure-gate** tasks (η, ι) whose runbooks mandate: live throwaway-graph rehearsal (seed a synthetic contaminated pair under `_probe`-prefixed keys, run end-to-end, verify byte-identical via `--compact` comparison, delete probes) → live dry-run → human manifest review → `--apply` → post-verify. This is the task-2085 routing lesson applied.
6. **Family consolidation (Phase 2) rewrites identity, not just location:** moving `know-live`-keyed nodes into `know_live` also rewrites the `group_id` **property** to the canonical form (unlike Phase-1 moves, where the property is already correct and only the key is wrong); the Qdrant collection merge likewise rewrites point `user_id` payloads. Same primitive, `rewrite_group_id`/`rewrite_user_id` parameter.
7. **Deploy before migrate.** δ (deterministic restart, task-2212 pattern) gates η: no live `--apply` until 2266 + 0b are the *running* code. Post-δ there are no new misroutes, so a hard write-queue quiesce is not required for correctness; the runbook still schedules η at queue-idle to avoid read races during `DETACH DELETE`.

## Contract (B + H)

**S1 — `canonicalize_project_id(raw: str) -> str`** (utils/validation.py, α). lower + `-`→`_`; raises `PathShapedProjectIdError` (or ValueError subclass) on path-shaped input; idempotent (`f(f(x)) == f(x)`).
**S2 — `Scope` field_validator** (models/scope.py, α). Applies S1 to `project_id` at construction; `graphiti_group_id` / `mem0_collection_name` / `mem0_user_id` therefore always canonical. Mem0Backend's instance cache stays coherent automatically (keyed downstream of Scope).
**S3 — MCP prologue normalization** (server/tools.py, β). Every memory tool (write, read, and graph-mutating — the ungated set in RCA §4 included) applies S1 **before** `validate_project_id` and the known-project gate. Error contract: path-shaped → the S1 error surfaced as a tool error naming the offending value.
**S4 — `GraphitiBackend` group-arg normalization** (graphiti_client.py, γ). Public methods taking `group_id`/`group_ids` canonicalize arguments at entry (decorator or helper — tactical); `_driver_for`/`_graph_for` and every Cypher `$group_id` param therefore agree. Covers durable-queue replay (`payload['group_id']` re-executed verbatim today).
**S5 — `move_entity_across_graphs(uuid, source_graph, target_graph, *, rewrite_group_id=None)`** (ε). Post: node exists in target with **byte-identical** `name_embedding`; all its RELATES_TO edges (with byte-identical `fact_embedding`s) and Episodic `MENTIONS` links reattached; node absent from source; create-before-delete ordering; idempotent (re-run detects already-moved and no-ops).
**S6 — `merge_foreign_duplicate(uuid, wrong_graph, home_graph)`** (ε). Post: every edge unique to the wrong-graph copy exists on the home copy; the copy is deleted; edge count home' = home + unique(wrong); no edge lost.
**S7 — migration/consolidation script CLI** (ζ, θ). Dry-run is the default and writes a JSON manifest (per node: uuid, name, source, target, disposition MOVE|MERGE|UNRESOLVED, edge/episode counts); `--apply` requires an existing manifest and re-verifies counts after; exit non-zero on any UNRESOLVED at `--apply`. Modeled on purge_knowlive_namespace.py.

### Boundary-test sketch (facing both sides)

| # | Scenario | Preconditions | Postconditions (asserted) |
|---|---|---|---|
| B1 | Hyphen write canonicalized (producer=MCP / consumer=read path) | registry contains dark_factory; write with `project_id='dark-factory'` | write ACCEPTED; lands in `dark_factory` graph key with `group_id='dark_factory'` property; `search`/`get_entity` with either spelling returns it |
| B2 | Path-shaped reject | write/read with `-home-leo-src-x` | specific loud error at the tool boundary AND at `Scope()` construction; nothing written |
| B3 | Replay canonicalized | durable-queue row persisted with raw `group_id='know-live'` | replay writes into `know_live` key with canonical property (asserted via backend-entry normalization; mock Cypher-param assertion) |
| B4 | Move fidelity (ε; unit=mock Cypher/param shapes; live=η rehearsal) | seeded node w/ embedding + 2 edges + 1 episode link in wrong graph | after move: present in target, absent in source; `--compact` embedding strings identical; edges+episode link reattached; re-run no-op |
| B5 | Duplicate merge (ε) | same uuid in two graphs, wrong copy has 1 unique edge | unique edge present on home copy; wrong copy deleted; no edge lost |
| B6 | Script end-to-end (ζ; live rehearsal at η) | synthetic contaminated pair under `_probe*` keys | dry-run manifest classifies MOVE/MERGE/UNRESOLVED correctly; `--apply` → zero foreign nodes on the pair; post-verify counts match manifest; re-run no-op |
| B7 | Family consolidation (θ; live rehearsal at ι) | `_probe`-family: sibling key + double-prefix collection | nodes moved with `group_id` property rewritten; Qdrant points merged with `user_id` rewritten; emptied sibling key GRAPH.DELETEd only at count 0 |

## Cross-PRD relationship (G4)

| Other stream / PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| **task 2266** (Phase 0a, filed) | consumes | per-group Graphiti client cache in graphiti_client.py | 2266 | δ deps on it; γ file-coexists (disjoint regions: arg entry vs client cache) |
| **W6** `fm-memory-identity` (2198…2213) | adjacent | identity collapse in graphiti_client.py; group_id property filter ALREADY amended into 2198/2210 | W6 owns identity; **this PRD owns routing/normalization/migration** | no duplication; γ coexists under narrow-file-lock; W6-ε alarms consume η's cleanliness |
| **M4** `recon-project-scope` (plans/recon-project-scope-prd.md) | adjacent vocabulary | ProjectScope/NewType inside reconciliation only; explicitly excludes 2115 territory | M4 | no file overlap; α's canonicalizer is the memory-path analogue (adopt M4 vocabulary later if desired) |
| **W8** `fm-task-dedup` | none | task-path routing (backend add_task) — different subsystem | W8 | no seam |

## Decomposition plan

DAG (Greek labels; IDs at decompose). Priorities: α–δ high (active-integrity code), ε–η high (recovery), θ–ι medium.

- **α — canonicalize_project_id + Scope validator + path-shape reject** *(utils/validation.py, models/scope.py)*. Intermediate — unlocks β, γ. Signal: roped into β/γ (foundation); unit: S1 idempotency, S2 derivation set (B2's Scope half).
- **β — MCP-boundary normalization across all memory tools** *(server/tools.py)*. **Leaf.** S3. Signal: **B1 + B2** (tool half). Prereqs: α.
- **γ — GraphitiBackend group-arg normalization (covers replay)** *(backends/graphiti_client.py)*. **Leaf.** S4. Signal: **B3** + B1's property/filter-agreement half. Prereqs: α. File-coexists with 2266/W6 (disjoint methods).
- **δ — Phase-0 deploy capstone** *(deterministic, auto-deploy preset)*. `before_done.script=scripts/restart-fused-memory.sh` (no args), `target_unit=fused-memory.service`, `timeout_secs=180` (task-2212 pattern). Signal: `get_status` fresh uptime; running code includes 2266+β+γ (also activates the merged 2111 guardrail). Prereqs: **2266**, β, γ.
- **ε — cross-graph move + duplicate-merge primitives** *(new `fused_memory/maintenance/cross_graph_move.py`)*. Intermediate — unlocks ζ, θ. S5+S6. Signal: **B4 + B5** (mock-unit level: Cypher/param shapes, create-before-delete ordering, `--compact`→`vecf32` passthrough over recorded fixtures). Prereqs: —.
- **ζ — migration script: census → classify → manifest → apply → verify** *(fused-memory/scripts/migrate_cross_graph_leak.py)*. **Leaf.** S7 + alias map (decision 4). Signal: **B6** (mock-unit level; classification incl. UNRESOLVED blocking). Prereqs: ε.
- **η — Phase-1 live migration gate** *(deterministic PURE GATE: `always_escalates=true`, no before_done)*. Runbook in-task: rehearse on `_probe*` throwaway pair (B6 live) → live dry-run → human reviews manifest (incl. alias map + any UNRESOLVED) → `--apply` at queue-idle → post-verify zero foreign + reify recall spot-check → resolve. Signal: manifest artifact + escalation + recorded post-verify counts. Prereqs: ζ, **δ**.
- **θ — consolidation script: family merges + collection merges + junk-key deletion** *(fused-memory/scripts/consolidate_namespace_families.py)*. **Leaf.** Decision 6; reuses ε primitives with rewrite params; Qdrant scroll→upsert→delete; GRAPH.DELETE only at count 0. Signal: **B7** (mock-unit level). Prereqs: ε.
- **ι — Phase-2 live consolidation gate** *(deterministic PURE GATE)*. Runbook: `_probe` rehearsal (B7 live) → dry-run → human review (solar-family question, `reify_` disposition) → apply → post-verify `GRAPH.LIST`/collection inventory. Signal: manifest + escalation + recorded inventory. Prereqs: θ, η.

Ordering rationale: β/γ land behind α; δ makes the fix *running* code; η's destructive apply is gated on δ (no re-contamination) and its own live rehearsal; θ/ι run last so Phase-1 moves don't double-handle family strays.

## Out of scope

- **The shared-driver race fix itself** — task 2266 (filed; δ deploys it).
- **W6 identity work** incl. the group_id property filter (already amended into 2198/2210).
- **graphiti-core upgrade / upstream engagement** (#1331, PRs #1294/#1305) — watch-task territory (cf. 2217).
- **Legacy per-project `collection_prefix` config cleanup at the *source*** (reify/autopilot old configs already corrected; ι only merges the residual collections).
- **Preventing future junk keys from non-MCP writers** (raw scripts bypassing the backend) — the three seams cover all product paths; ad-hoc scripts remain out of contract.

## Open questions (tactical — surfaced, not blocking)

1. **`my_solar_challenge` vs `solar_challenge_platform`** — same project? Both populated (771 / 1,528). **Suggested resolution:** treat as SEPARATE (no merge) unless the human states otherwise at ι's manifest review; the alias map makes it a one-line change.
2. **`reify_` (empty-project-id) Qdrant collection disposition** — inspect contents at ι review; merge into `fused_reify` if attributable, else archive-dump + delete.
3. **γ mechanism** — decorator vs explicit per-method helper calls. Implementer's choice during γ; must cover every public group-arg method (grep-verifiable).
4. **Embedding read transport in ε** — redis-cli `--compact` subprocess vs raw RESP socket. Either is fine if the float32 strings are passed through untouched (the pure-function unit tests pin the passthrough); decide during ε.
5. **η scheduling** — queue-idle window selection (decision 7 makes it safety-margin, not correctness). Decide at η.

## Metadata surfaced for the future tracking-infra session

The orchestrator does not currently read `user_observable_signal` / `consumer_ref` / substrate-confirmed metadata; filed as substrate for a future tracking session.
