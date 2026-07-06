# Capability manifest — fm-memory-identity (W6) PRD

Per-leaf capability→evidence bindings (mechanizing G3 + G6) for
`plans/fm-memory-identity-prd.md`. All bindings verified against main on 2026-07-06
in the authoring session. No numeric bounds and no grammar fixtures in this batch.
The one "loud alarm" signal (ε B5) is a **positive detection** observable, not an
input-rejection mechanism — bound to the emitted WARN log line, not a G6-branch-4
rejection check.

**Tasks:** α (intermediate — foundation, roped into β as its integration-gate; no
standalone user signal), β/δ/ε/ζ (leaves). All producer dependencies resolve
**upstream** (DAG-direction verified).

## α — `_resolve_or_create_entity` chokepoint + per-group identity lock *(intermediate)*

Foundation for β; no standalone leaf signal (C-as-integration-gate → β). Deliverables
bound as β's upstream producers below. Substrate it builds on:

| Capability | Evidence | Verdict |
|---|---|---|
| Exact-name Cypher lookup exists | grep:fused-memory/src/fused_memory/backends/graphiti_client.py:1175 (`get_nodes_by_exact_name`) | PASS (wired — `MemoryService.get_entity` already calls it) |
| Survivor-first duplicate enumerator | grep:graphiti_client.py:1209 (`find_duplicate_entity_nodes`, ordered `edge_count DESC, created_at ASC, uuid ASC`) | PASS |
| Merge primitive to fold duplicates | grep:graphiti_client.py:955 (`merge_entities`) → :849 (`redirect_node_edges`) | PASS (wired) |
| No competing per-group identity lock already (must add) | grep:durable_queue.py:136 (`_group_locks`) guards only `_claim_next` (:289-314), NOT `_process_item`/`add_episode` (:316-335) | PASS (gap confirmed — new lock justified) |

## β — write-time gate + fold the 4 sweeps *(leaf / integration-gate, Stream A)*

**Signal (B1):** two concurrent `add_episode` for the same entity name (`workers_per_group=2`) → `get_entity`/`search` returns **exactly one** node.

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| Per-group identity `asyncio.Lock` (S2) | producer:task-α (upstream) | PASS (producer upstream) |
| `_resolve_or_create_entity` chokepoint (S1) | producer:task-α (upstream) | PASS (producer upstream) |
| The 4 sweeps to fold exist | grep:memory_service.py:484 (`_dedup_episode_edges`), :531 (`_dedup_episode_nodes`), :612 (`_restore_superseded_dependency_edges`), :687 (`_normalize_task_node_names`) | PASS |
| Fold site (where the sweeps chain today) | grep:memory_service.py:807 (`_execute_graphiti_write`), :850-858 (the four sweep calls) | PASS (wired) |
| `add_episode` is the wrapped write entry | grep:memory_service.py:838 (`self.graphiti.add_episode(...)`) | PASS (wired) |
| Read path that observes node identity | grep:graphiti_client.py:1175 (`get_entity` resolves via exact-name) | PASS (wired) |
| Intra-batch edge-dedup to preserve in the fold | producer:task-2118 (in-progress, upstream dep) | PASS (producer upstream) |
| DAG-direction | α, 2118 both upstream of β | PASS |

## δ — fresh per-edge uuids in `redirect_node_edges` + `superseded_edge_uuid` *(leaf)*

**Signal (B4):** after `merge_entities`, graph-wide per-uuid `count(*) ≤ 1`; each redirected edge carries `superseded_edge_uuid`.

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| The `SET new.uuid = old.uuid` copy to replace (defect site) | grep:graphiti_client.py:902, :930 | PASS (defect confirmed present) |
| Per-property `CREATE ... SET` substrate (preserves vecf32 embedding) | grep:graphiti_client.py:901-912 (existing per-property SET) | PASS (wired) |
| `properties(e)` FalkorDB substrate (for property copy) | grep:graphiti/graphiti_core/models/edges/edge_db_queries.py:203 (`properties(e) AS attributes`) | PASS |
| Fresh-uuid generation | Python stdlib `uuid.uuid4` (no FalkorDB `randomUUID` — confirmed unused in-repo) | PASS |
| `superseded_edge_uuid` field-population (result-field twin) | producer:task-δ writes a **non-sentinel** uuid string on the production redirect path (this task) | PASS (populated on production path) |
| Observable via merge caller | grep:graphiti_client.py:955 (`merge_entities` → `redirect_node_edges`) | PASS (wired) |
| DAG-direction | no intra-batch producer needed (file-lock coexists with α) | PASS |

## ε — startup identity-integrity scan: dup-node alarm + dup-uuid-edge repair *(leaf)*

**Signals (B5, B6):** seeded dup-name graph → WARN alarm naming group+name; seeded dup-uuid-edge graph → repaired to per-uuid `count(*) ≤ 1`, edge set preserved; clean graph = no-op.

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| Enumerate every FalkorDB graph | grep:graphiti_client.py:1864 (`list_graphs`) → client `list_graphs()` (:340 usage) | PASS (wired) |
| Exact-name duplicate detection | grep:graphiti_client.py:1209 (`find_duplicate_entity_nodes`), :1175 (`get_nodes_by_exact_name`) | PASS |
| WARN/alarm log emission (B5 observable) | grep:graphiti_client.py:944 (`logger.info` pattern in redirect) → module `logger` available for `logger.warning` | PASS |
| Dup-uuid-edge repair convention (fresh uuid + `superseded_edge_uuid`) | producer:task-δ (upstream) | PASS (producer upstream) |
| Backend-init hook to run the pass | grep:graphiti_client.py:261 (`initialize`), :252 (`_ensure_indices` per-graph hook precedent) | PASS (wired) |
| Read-back to assert edge-set preserved (B6) | grep:graphiti_client.py:760 (`get_all_valid_edges`) | PASS (wired) |
| DAG-direction | δ upstream of ε | PASS |

## ζ — simplify edge-read queries to uuid-keyed dedup *(leaf)*

**Signal (B7):** the two read methods no longer contain `WITH DISTINCT`; on a (repaired) graph that had dup-uuid edges, uuid-keyed dedup returns **identical** edge sets/counts to the `WITH DISTINCT` version.

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| `WITH DISTINCT` idiom to remove (defect sites) | grep:graphiti_client.py:723 (`WITH DISTINCT e`), :803 (`WITH DISTINCT n, e`) + justification comments :702-710, :770-782 | PASS (idiom confirmed present) |
| **uuid-uniqueness premise that makes uuid-keyed dedup correct** | producer:task-δ (stops new dup uuids) **and** producer:task-ε (repairs legacy dup uuids) — **both upstream** | PASS (producers upstream — without both, the "identical edge sets" premise is FALSE) |
| Read substrate | grep:graphiti_client.py:726, :806 (`ro_query`) | PASS (wired) |
| DAG-direction | δ **and** ε both upstream of ζ | PASS |

## Result

**No FAIL bindings.** Every capability resolves to `grep:<file>:<line> wired`,
`producer:task-<label> upstream`, or stdlib. The load-bearing G6 binding is ζ's
uuid-uniqueness premise, which is satisfied only because both δ and ε are hard
upstream dependencies of ζ (enforced as real `add_dependency` edges at filing). β's
"exactly one node" premise is satisfied because α (lock + chokepoint) is upstream.
