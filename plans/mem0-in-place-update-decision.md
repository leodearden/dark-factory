# DECISION: In-place Mem0 update tool — tool shape & authorization model

## §0 — Status & provenance

**Status:** DECIDED
**Task:** dark_factory task 3055
**Filed by:** reify reconciliation Stage 2 (run `5ddb8671-7cf4-4092-ba89-1fd66af05286`), from Stage 1
finding `c073e7f9-877f-48a7-8eaf-3c2bf9aae640` (second limb — original scope was content-amend
in-place update).

This document is a **decision, not an implementation**. It decides *whether* and *how*
fused-memory should expose an in-place Mem0 update tool, and specifies the contract precisely
enough that no further design decisions are needed downstream. All code lands on the already-filed
follow-up, dark_factory task **3088** (`depends on 3055`) — see §7 for the section-by-section
hand-off and the proposed `metadata.delivered_checks` gate.

## §1 — Problem

Task 3055 names two failure modes. They are **one root cause wearing two faces**: the fused-memory
Mem0 MCP write surface has no in-place update path — it is add-only (`add_memory`,
`add_system_record`, `add_episode`) plus `delete_memory`. Both failure modes are direct
consequences of that single gap, and both are resolved by the same decision:

1. **Curator-gate amendment deadlock / correction-entry accretion.** Lacking any update path,
   correcting an existing Mem0 record forces writing a brand-new corrective memory instead of
   amending the original — causing accretion (duplicate/near-duplicate entries competing for
   authority) and repeated curator-gate deadlocks over which entry is canonical.
2. **Consolidation-closure metadata stamping.** The emerging consolidation-closure norm (no code
   or docs footprint yet in this repo — see §2's scoping caveat) wants to stamp a shared
   `metadata.topic` on every member of a superseded cluster, so a deterministic lookup
   (`get_memories_by_metadata`) can later enumerate that cluster with certainty instead of gambling
   on semantic-search recall. With no update path, this metadata-only patch — no content change at
   all — cannot be done in place either, forcing the identical add/delete-and-rewrite workaround as
   failure mode 1.

**Root cause: no in-place update on the Mem0 MCP write surface.** Closing that one gap resolves
both failure modes. Treating them as two separate problems would produce two overlapping tools
that each have to independently solve the same point-id-preservation and payload-overwrite problem
(§2, §5) — which is itself an INV-5 (`no-lockstep-duplication`) risk, not just redundant effort.

## §2 — Ground-truth capability audit

**GROUND-TRUTH CORRECTION.** Task 3055's filing states: *"There is therefore no way to change an
existing Mem0 entry's content OR metadata in place."* That claim is **true of the MCP surface** and
**false of the system**. Re-verified against this worktree at HEAD `3e9b1b0f01` (pre-1):

| Layer | Capability | Status (verified) |
|---|---|---|
| Qdrant | Payload-only `set_payload` (no vector required) | Exists — `fused-memory/tests/test_mem0_qdrant_integration.py:69-90` (`test_set_payload_without_vector`) |
| mem0 library | `AsyncMemory.update` / `_update_memory` | Exists, but payload-**overwriting** (see §5) |
| `Mem0Backend.update(memory_id, data, scope, metadata=None)` | Content + metadata update in place, preserves the Qdrant point id | **EXISTS** — `fused-memory/src/fused_memory/backends/mem0_client.py:238-262`, docstring included, warning explicitly about the payload-overwrite trap |
| `MemoryService` wrapper (journaled + event-emitting) | Service-layer call to the above | **MISSING** — no call site anywhere in `services/memory_service.py` |
| MCP tool | Any `update_memory`-shaped tool | **MISSING** — the complete set of Mem0-write tools on the current surface is `add_memory` (`server/tools.py:1051`), `add_system_record` (`:1282`), `add_episode` (`:905`), and `delete_memory` (`:1941`). `update_edge` (`:2060`) exists but is **Graphiti-only** — it edits an edge's fact text / `invalid_at` and never touches Mem0. |
| `EventType.memory_updated` | Event type for an in-place update | **EXISTS** — `models/reconciliation.py:16` — but is currently emitted from exactly two call sites, both Graphiti: `MemoryService.update_edge` (`services/memory_service.py:3685`) and `MemoryService.reassign_edge` (`:3778`) |

So **the actual gap is service-layer + MCP exposure of an already-existing backend primitive** —
not a missing primitive. This materially shrinks task 3088's scope: it does not need to invent an
in-place Mem0 write, it needs to wrap and expose one that already ships and is already relied on in
production (next point).

**This gap is not theoretical — it is already causing measurable harm:**

1. **The near-duplicate guard instructs agents to do something that does not exist.**
   `fused-memory/src/fused_memory/server/near_duplicate_guard.py` soft-blocks a near-duplicate
   `procedural_knowledge` write with the hint *"Search first and **update** or skip instead of
   writing a duplicate"* (`:40`), and for a known-contradictory topic cluster, *"search first and
   **update/consolidate** the existing entries"* (`:52`). Both hints instruct the agent to do the
   one thing the MCP surface cannot do. This is failure mode 1, reproduced verbatim in the guard's
   own remediation text — the guard has been telling agents to do the impossible since it shipped.

2. **Three maintenance scripts were forced to delete records that only needed a patch —
   permanently losing `created_at` and provenance.**
   `fused-memory/scripts/prune_recon_cycle_summaries.py:25-30` states this outright: *"Mem0/Qdrant
   exposes `delete_memory` but no in-place payload-update primitive on this path — the same
   constraint documented in `scripts/sweep_orphan_flag_markers.py` (task-1659) ... Re-tagging via
   delete+re-add would also change `created_at` and lose provenance. So the effective operation
   here is PRUNE-to-N."* `scripts/sweep_orphan_flag_markers.py` (task-1659) hit the identical wall
   and also deleted rather than retagged.

3. **A working in-place precedent already exists in this repo, and it already solved the hard
   part.** `scripts/tag_cgl_eta_rehome_scope.py` DOES edit Mem0 entries in place, by calling
   `memory.mem0.update` directly (`:44-47`) — because it solved the payload-overwrite trap
   described in §5: mem0's `_update_memory` rebuilds the *entire* Qdrant payload from
   `deepcopy(metadata) if metadata else {}`, so calling `update` with `metadata=None` (the default)
   silently **wipes every custom payload key** the point already carried. The script's fix — read
   the existing payload back, strip mem0-owned keys via `_MEM0_MANAGED_METADATA_KEYS` (`:235`), and
   forward only the remaining custom-provenance subset as `metadata=` — is the one piece of
   hard-won knowledge any service-layer wrapper MUST reuse, not re-derive (§5, §6).

4. **The consumers that make metadata stamping worth enabling already exist and already ship.**
   `MemoryService.get_memories_by_metadata` (`services/memory_service.py:3305`) and
   `Mem0Backend.count_by_metadata` (`backends/mem0_client.py:296`) already provide exact,
   non-semantic Qdrant payload-filter lookup — precisely the "deterministic lookup can identify all
   members of that cluster" that failure mode 2 needs. Tagging survivors is only valuable *because*
   this read side already exists; this decision is only about closing the write side that would
   feed it.

**Scoping caveat.** "Consolidation-closure" (the norm of stamping `metadata.topic` across a
superseded cluster) has **no code or docs footprint in this repo** as of this writing — a
repo-wide search for the term returns nothing outside task 3055's own filing text. This decision
treats it as a real, named consumer (it motivates the metadata-only arm's existence and its
lighter authorization bar, §4), but does not assume it has, or will ever have, a codified spec
beyond what task 3055 states. A future task that formalizes "consolidation-closure" should
reconcile against §4/§5 here rather than re-deciding the write-path question this doc already
settles.
