# fm-memory-identity — write-time entity identity + edge-uuid uniqueness

**Stream:** W6 (bug-hotspot remediation program 2026-07-06, Wave 1).
**Status:** active — authored 2026-07-06. Approach **B + H** (high-stakes data integrity).
**Program doc (authoritative G4 seam map):** `plans/bug-hotspot-remediation-program-2026-07-06.md`.
**Survey findings:** `plans/bug-hotspot-survey-2026-07-06-full-findings.json` → cluster `fm-memory`, findings 0 (no write-time identity guarantee → 4-function reactive sweep chain) and 1 (`redirect_node_edges` breaks edge-uuid uniqueness).

## Goal

Make entity identity and edge-uuid uniqueness **write-time guarantees** in fused-memory's Graphiti backend, replacing the current post-hoc reactive sweep chain and the fragile hand-written edge-dedup idiom.

What an operator/agent observes when this lands:
- Two concurrent `add_episode` writes for the same entity name (or two unrelated episodes minting the same name) resolve to **exactly one** entity node when read back through `get_entity`/`search` — not two that a human must manually merge in FalkorDB.
- After any `merge_entities`, a graph-wide scan finds **no two edges sharing a uuid**; the `WITH DISTINCT` graph-element-identity idiom in the edge-read queries (wrong twice in production: `cfed95c706`, task-2084 pair `be71e7850a`/`c3193549a7`) is gone.
- The recurring "duplicate Graphiti node → manual FalkorDB merge" operator runbook (operator-memory incident class: tasks 2073/2081/2110/2118, the `/unblock` Graphiti-dedup protocol) becomes obsolete for **newly-created** duplicates.

## Background

`MemoryService._execute_graphiti_write` (memory_service.py:807-873) calls `graphiti.add_episode(...)` then chains **four independent best-effort post-hoc passes**: `_dedup_episode_edges` (:484), `_restore_superseded_dependency_edges` (:612), `_dedup_episode_nodes` (:531), `_normalize_task_node_names` (:687). Each extracts touched names/edges from the just-returned result, re-scans via `find_duplicate_entity_nodes`, and `except Exception: log-and-continue`. Root cause (stated in `_dedup_episode_nodes`' own docstring): **graphiti_core's ingestion-time entity resolution only resolves each extracted node against fuzzy hybrid embedding+BM25 candidates — there is no exact-name Cypher gate at write time.** No FalkorDB uniqueness constraint exists: `build_indices_and_constraints` is stubbed to a no-op on the multi-tenant driver (graphiti_client.py:195), and even the upstream FalkorDriver builds only range+fulltext indices, never a uniqueness constraint. Because each sweep only re-scans names/uuids touched by the episode that just completed, a duplicate created by a concurrent sibling worker (if `workers_per_group` is ever raised above the config-pinned 1) survives until a human merges it.

Separately, `redirect_node_edges` (graphiti_client.py:849-953) deliberately copies `SET new.uuid = old.uuid` (lines 902, 930) when redirecting an edge onto the surviving node during a merge, minting a second relationship element carrying an already-used uuid. `Edge.uuid` is therefore no longer graph-wide unique, so **every** edge-reading query must defend with `WITH DISTINCT e` / `WITH DISTINCT n, e` (graph-element identity) instead of a plain uuid-keyed dedup — a fragile idiom already gotten wrong twice.

### Substrate reality (G3 — verified against main 2026-07-06)

- `get_nodes_by_exact_name(name, group_id)` exact-name Cypher lookup **exists** (graphiti_client.py:1175). `find_duplicate_entity_nodes` (:1209) already returns matches **canonical/survivor-first** (most valid edges → oldest → uuid). `merge_entities`/`redirect_node_edges` (:955/:849) exist. `list_graphs()` (:1864) enumerates every FalkorDB graph. `properties(e)` is valid FalkorDB Cypher (used in graphiti_core edge queries). — the chokepoint, the merge primitive, the startup enumerator, and the property-copy substrate all exist.
- `graphiti_core` is an **installed PyPI package (0.28.2), not editable** — reinforcing the standing doctrine **wrap, don't patch**. The gate wraps `add_episode`; it never modifies `resolve_extracted_nodes`.
- `DurableWriteQueue`'s per-group `_group_locks` (durable_queue.py:136) serialize only `_claim_next` — **not** `_process_item`. With `workers_per_group>1`, two workers run `add_episode` for the same group **concurrently**. So the identity guarantee needs its **own** per-group lock around `add_episode`+reconcile; the queue's claim-lock does not provide it.
- FalkorDB has **no per-row uuid generator** (`randomUUID()` not available; unused anywhere in the codebase). Fresh per-edge uuids must be generated Python-side (`uuid.uuid4`).

## Resolved design decisions (do not relitigate)

1. **The write-time gate is a per-`group_id` `asyncio.Lock` + an in-critical-section exact-name reconcile — NOT a FalkorDB uniqueness constraint.** Rationale: graphiti_core mints nodes internally and writes uuid-keyed, so a `(group_id, Entity.name)` uniqueness constraint would make graphiti_core's internal write **error mid-ingestion** on a collision rather than resolve-to-existing — a data-integrity hazard on the hottest write path — and the multi-tenant driver builds no constraints anyway. fused-memory is single-process, so an in-process per-group `asyncio.Lock` fully serializes the check-then-create window. **The lock is the load-bearing guarantee; the startup duplicate-scan alarm (ε) is the safety net.** The DB constraint is **deferred** (see Open questions) pending a multi-process-writer future.
2. **Wrap, don't patch graphiti_core.** No edits to `resolve_extracted_nodes` / the installed package.
3. **The identity lock is distinct from the durable-queue claim-lock** and is held **only** across the Graphiti `add_episode`+reconcile (never a Mem0 write) so it doesn't over-serialize dual-store writes.
4. **`redirect_node_edges` assigns a fresh Python `uuid4` per redirected edge** (enumerate the redirect set, one fresh uuid per edge, keyed on the stable internal element `ID()` to disambiguate any pre-existing shared uuids), and carries `superseded_edge_uuid = <old uuid>` as an audit property. No reliance on a FalkorDB uuid function.
5. **The read-query simplification (ζ) is HARD-gated on the migration (ε).** Switching `get_valid_edges_for_node`/`get_all_valid_edges` from element-identity `WITH DISTINCT` to uuid-keyed dedup is correct only once **no** two edges share a uuid — which needs both δ (stop minting new dup-uuid edges) **and** ε (repair legacy dup-uuid edges). ζ depends on both.
6. **The one-shot dup-uuid-edge repair runs as an idempotent startup scan (ε), not a live-data operator task** — avoiding the recurring "operational live-data work mis-routed into the TDD/architect pipeline" hazard (task-2085 class). ε is code + a startup hook, TDD-verifiable against seeded graphs; the production migration happens automatically on the next fused-memory restart (out-of-cgroup `systemctl --user restart fused-memory.service` per program-doc decision 6 — **not** `restart-fused-memory.sh --drain`, which hangs). It is combined with the dup-**node** alarm into one startup identity-integrity pass.
7. **Fold, don't delete, the four sweeps.** `_reconcile_episode_identity(result, group_id)` subsumes all four as legacy-cleanup running **inside the identity lock**; they still catch pre-gate legacy dups and intra-batch dups but are no longer the sole defense.

## Contract (B + H)

### Seam signatures + invariants

**S1 — `GraphitiBackend._resolve_or_create_entity(name, group_id) -> str` (new, α).**
Post-condition: on return, **exactly one** `Entity` node with that exact `name` exists in `group_id`'s graph; returns its uuid. Implementation: `get_nodes_by_exact_name`; on ≥2 matches, `find_duplicate_entity_nodes` (survivor-first) + `merge_entities(dep, survivor)` folding all but the canonical; on 0 matches, no-op (creation stays graphiti_core's job — this primitive resolves/collapses, it does not mint). **Must be called only while holding S2's lock.** Idempotent.

**S2 — `GraphitiBackend._identity_lock_for(group_id) -> asyncio.Lock` (new, α).**
A per-`group_id` lock registry. Invariant: held across `add_episode` + `_reconcile_episode_identity` (S3) as one critical section; never held across a Mem0 write. Ordering: acquire → `add_episode` → reconcile → release.

**S3 — `MemoryService._reconcile_episode_identity(result, group_id) -> ReconcileStats` (new, β; folds the 4 sweeps).**
Runs **inside** S2's lock, immediately after `add_episode`. Subsumes `_dedup_episode_edges`, `_restore_superseded_dependency_edges`, `_dedup_episode_nodes` (via S1), `_normalize_task_node_names`. Idempotent, best-effort per sub-pass (a failing sub-pass logs and continues — it must not fail the write), returns a stats struct for the journal. Preserves the intra-batch edge-dedup behaviour added by **task 2118**.

**S4 — `redirect_node_edges` invariant (δ).** Every `RELATES_TO` element carries a graph-wide-unique `uuid`; a redirected edge additionally carries `superseded_edge_uuid = <original uuid>`. No two elements share a uuid.

**S5 — post-ζ read invariant.** `get_valid_edges_for_node`/`get_all_valid_edges` dedup by `(endpoint uuid, edge uuid)` — valid because S4 makes `e.uuid` unique. Double-attribution (each directed edge under both endpoints) and self-loop collapse are preserved.

**S6 — startup identity-integrity pass (ε).** On backend init, for each graph from `list_graphs()`: (a) exact-name duplicate `Entity` nodes → **loud WARN/alarm** naming `group_id`+name (safety net for S1/S2); (b) duplicate-uuid `RELATES_TO` edges → **one-shot idempotent repair** (re-mint fresh uuid + `superseded_edge_uuid`, preserving the edge set). No-op on a clean graph.

### Boundary-test sketch (facing both sides of the seam)

| # | Scenario | Preconditions | Postconditions (asserted) |
|---|---|---|---|
| B1 | Concurrent same-name write (producer=write path / consumer=read path) | two `add_episode` for the same single group, same entity name, `workers_per_group=2` | `get_entity(name)`/`search` returns **1** node; journal shows a merge; no unhandled error |
| B2 | Cross-episode duplicate | episode A mints "Foo"; later episode B mints "Foo" | after B, `get_entity("Foo")` = 1 node |
| B3 | Lock scope | a Graphiti `add_episode` in flight for group G | a Mem0 write for G is **not** blocked (lock is Graphiti-only) |
| B4 | Merge edge uuids (δ) | `merge_entities` on two nodes with several edges each | every redirected edge has a fresh unique uuid + `superseded_edge_uuid` set; graph-wide `count(*)` per uuid ≤ 1 |
| B5 | Startup dup-node alarm (ε-a) | graph pre-seeded with two "Foo" `Entity` nodes | init emits a WARN line naming group+"Foo" |
| B6 | Repair migration (ε-b) | graph pre-seeded with two edges sharing a uuid | repair → per-uuid `count(*)` ≤ 1, **edge set preserved** (same count of distinct elements, none lost/merged); re-run = no-op |
| B7 | Read simplification (ζ) | a graph that HAD dup-uuid edges, now repaired | uuid-keyed `get_all_valid_edges`/`get_valid_edges_for_node` return **identical** edge sets/counts to the `WITH DISTINCT` version; neither method contains `WITH DISTINCT` |

## Cross-PRD relationship (G4)

| Other stream / PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| **M5** `fm-cancellederror-convention` (PRD/tasks not yet filed) | shares files | `memory_service.py` + `graphiti_client.py` — M5 rewrites the gather/`CancelledError` sites; W6 edits identity logic (disjoint code regions) | M5 owns gather idiom; **W6 owns identity** | coexist via narrow-file-lock + rebase; W6 tasks **must not touch** the gather sites |
| **W5** `recon-reliability` | consumes | fm-recon edge reads (`get_valid_edges_for_node`/`get_all_valid_edges`) + Mem0 dedup-exempt path | W5 (recon side) / W6 (edge invariant) | W6's restored uuid invariant benefits recon reads; no shared code |
| **task 2115** (deferred, DESIGN-FIRST) | adjacent hazard | cross-graph entity leak via graphiti_core's shared-driver `clone()` race in `add_episode` | 2115 (separate bug) | **out of scope**; but W6's B1 concurrency test (`workers_per_group=2`) could trip 2115's leak — B1 must isolate to one group and assert group-scoped identity (see Open questions) |

Per the program G4 seam table: **"Entity write-time identity + `redirect_node_edges` uuid semantics" is owned by W6; no other stream touches `graphiti_client.py`.**

## Decomposition plan

DAG (Greek labels; task IDs assigned at decompose):

- **α — `_resolve_or_create_entity` chokepoint + per-group identity lock** *(graphiti_client.py)*. Intermediate — unlocks β. Adds S1 + S2 using existing exact-name/merge primitives. Signal: roped into β (C-as-integration-gate: α is a foundation with no standalone user signal). Prereqs: —.
- **β — wire the write-time gate + fold the 4 sweeps** *(memory_service.py)*. **Leaf / integration-gate for Stream A.** Holds α's lock across `add_episode`+`_reconcile_episode_identity` (S3), folding all four sweeps. Signal: **B1** (two concurrent same-name `add_episode` → `get_entity`/`search` = 1 node) + B2, B3. Prereqs: **α**, **task 2118** (in-progress — fold the post-2118 intra-batch edge-dedup).
- **δ — fresh per-edge uuids in `redirect_node_edges` + `superseded_edge_uuid`** *(graphiti_client.py)*. **Leaf.** S4. Signal: **B4**. Prereqs: — (file-lock-coexists with α).
- **ε — startup identity-integrity pass: dup-node alarm + dup-uuid-edge repair** *(graphiti_client.py + backend init)*. **Leaf** (migration + safety net). S6. Signal: **B5** + **B6**. Prereqs: **δ** (repair uses the fresh-uuid convention).
- **ζ — simplify edge-read queries to uuid-keyed dedup** *(graphiti_client.py)*. **Leaf** (cleanup). S5; deletes the `WITH DISTINCT` idiom + its justification comments. Signal: **B7**. Prereqs: **δ, ε** (HARD — no dup uuids may exist before the read path trusts uuid uniqueness).

Ordering rationale: the four `graphiti_client.py` tasks (α, δ, ε, ζ) touch **disjoint methods** of one file — the narrow-file-lock serialises their dispatch and each rebases cleanly; the deps enforce δ→ε→ζ and α→β correctness ordering.

## Out of scope

- **CancelledError / `asyncio.gather` idioms** (fm-memory findings 2, 3) → **M5**. W6 keeps every edit to identity logic; it does not touch the gather/`CancelledError` sites in `memory_service.py`/`graphiti_client.py`/`tools.py`.
- **`_CYCLE_SUMMARY_STAGE_TO_RECON_POOL` duplication** (fm-memory finding 4) → recon/W5 (or M5) territory; not identity.
- **Cross-graph entity leak / shared-driver `clone()` race** (task 2115, deferred DESIGN-FIRST) — a distinct data-integrity bug W6 does not fix (noted as an adjacent hazard for B1).
- **Mem0-side dedup exemption** → W5.
- **A FalkorDB `(group_id, Entity.name)` uniqueness constraint** — deferred (see Open questions).

## Open questions (tactical — surfaced, not blocking)

1. **FalkorDB uniqueness constraint as defense-in-depth.** Deferred by design decision 1 (the lock is the guarantee). If fused-memory ever moves to multi-process writers against one graph, revisit adding a `GRAPH.CONSTRAINT`-backed `(group_id, Entity.name)` MANDATORY/UNIQUE constraint — **but** its interaction with graphiti_core's internal uuid-keyed `CREATE` (does it error or resolve?) must be probed first on the deployed FalkorDB version before relying on it. **Suggested resolution:** leave unbuilt; the ε startup alarm covers detection. Decide when multi-writer lands.
2. **Per-edge fresh-uuid application mechanism in δ.** Default: enumerate the redirect set and issue a **per-edge** redirect keyed on the stable internal element `ID()` (small, rare operation → O(N) queries acceptable). Optional optimization: a single `UNWIND $pairs` bulk redirect. **Suggested resolution:** ship the per-edge loop (substrate-minimal, no `ID()`-stability assumptions across queries); optimise only if a merge with many edges shows up hot. Decide during δ.
3. **B1 vs task 2115.** Running B1 at `workers_per_group=2` exercises graphiti_core's shared-driver `clone()` race (2115). **Suggested resolution:** B1 uses a single group_id and asserts group-scoped node identity only; it does not assert cross-graph isolation (2115's concern). Keep B1 tightly scoped so a 2115 leak can't mask/immask the identity assertion. Decide during β.
4. **M5 dep wiring.** M5 (`fm-cancellederror-convention`) tasks were **not filed** at W6 decompose time, so no integer dep could be wired. W6's edits are confined to identity logic (disjoint from M5's gather sites), so the narrow-file-lock + rebase handles the whole-file coexistence. **Suggested resolution:** if M5 is filed before W6 dispatches, wire β/δ/ε/ζ → M5's `memory_service.py`/`graphiti_client.py` tasks as a follow-up. Recorded here per the program-doc AFK-autonomy clause; taken as the safe default.

## Metadata surfaced for the future tracking-infra session

The orchestrator does **not** currently read `user_observable_signal` / `consumer_ref` / substrate-confirmed metadata; these are filed as substrate for a future tracking session.
