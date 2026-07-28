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

## §3 — Decision 1: tool shape

**DECIDED: one unified MCP tool, `update_memory`, Mem0-only, with two independently-optional
arms** — `content` (semantic amend) and the metadata-patch family (`metadata_patch` /
`metadata_delete_keys` / `metadata_mode`) — rather than two separate tools. At least one arm must
be supplied; supplying none is a `ValidationError`.

### Alternatives considered and rejected

- **Two narrow tools** (`amend_memory_content` + `update_memory_metadata`). Rejected: the
  point-id/UUID-preservation guarantee and the mem0 payload-overwrite mitigation (§2 point 3, §5)
  would then have to live in two places that must agree byte-for-byte — an INV-5
  (`no-lockstep-duplication`) violation waiting to happen the first time one of the two tools is
  touched and the other isn't. It would also force a caller that legitimately wants to amend
  content **and** re-tag metadata in the same logical edit into two separate writes with two
  journal entries and a torn intermediate state (the record briefly exists with new content but
  stale metadata, or vice versa).
- **Extending `add_memory` with an optional `memory_id`.** Rejected: it overloads the
  near-duplicate guard's write path (which is specifically about *new* content colliding with
  existing content) and makes an in-place mutation indistinguishable from a create in the write
  journal — a journal reader could no longer tell "this row is a new memory" from "this row
  overwrote memory X" without inspecting a side channel.

### The contract

**Signature** (mirrors `delete_memory`'s parameter shape — `server/tools.py:1941-1949` — since
callers already hold `store`/`project_id`/`memory_id` from a prior `search` result and should not
have to remember a different shape for update vs. delete):

```python
async def update_memory(
    memory_id: str,
    store: str,
    project_id: str,
    content: str | None = None,
    metadata_patch: dict | None = None,
    metadata_delete_keys: list[str] | None = None,
    metadata_mode: str = 'merge',       # 'merge' | 'replace' — governs metadata_patch only
    reason: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
    metadata: dict | None = None,        # envelope only — may carry _causation_id; NOT the patch
    ctx: Context | None = None,
) -> dict[str, Any]:
```

**Naming collision warning (implementer must preserve this distinction):** the pre-existing
`metadata: dict | None` parameter name is the causation/envelope kwarg every other tool already
uses (consumed by `_extract_causation(metadata, agent_id)`, `server/tools.py:479-497`) — it is
never stored on the record. The new record-payload argument is deliberately named
**`metadata_patch`**, not `metadata`, so the two can never be confused at the call site or in the
implementation. Do not rename either to make them match.

- **`store`**: validated against `_VALID_STORES` exactly like `delete_memory` (`:1998-2002`). A
  valid-but-wrong value of `'graphiti'` is rejected with a `ValidationError` naming `update_edge`
  as the correct tool — **do not silently fan out**. `store` is accepted (rather than omitted,
  since the tool is Mem0-only in practice) purely for this fail-fast ergonomics: every `search`
  result already carries a `store` field, so a caller updating a record it just found does not have
  to branch client-side before calling; it calls the same way it would call `delete_memory`, and a
  wrong `store` value produces an actionable error instead of `update_memory` quietly assuming
  Mem0 was meant.
- **`content`** (the content-amend arm): when supplied, must be non-empty (mirrors `update_edge`'s
  `fact` validation, `:2116-2122`). Routes through the existing `Mem0Backend.update` (§5) —
  re-embeds unconditionally; this is inherent to mem0's own `update()`, not a design choice (§5).
- **`metadata_patch`** (the metadata-patch arm) is a **shallow merge** over the record's existing
  custom payload when `metadata_mode='merge'` (the default) — it never replaces the whole payload.
  Reserved/mem0-owned keys (`_MEM0_MANAGED_METADATA_KEYS` — `data`, `hash`, `created_at`,
  `updated_at`, `user_id`, `agent_id`, `run_id`, `actor_id`, `role`; see §2 point 3 and §6) present
  in `metadata_patch` are **rejected at the boundary** with a `ValidationError` naming the offending
  key(s) — never silently dropped, so a caller can never believe it wrote `created_at`.
- **`metadata_mode='replace'`** replaces the entire **custom-provenance subset** of the payload
  with exactly what `metadata_patch` supplies (mem0-owned keys are still preserved verbatim
  underneath — "replace" never means "replace the whole Qdrant payload"). Named and valued to
  match `update_task`'s existing `metadata_mode` vocabulary (`'merge'` / `'replace'`) deliberately —
  same operator-facing naming, not the same code (tasks and Mem0 records are different backends;
  see §6 for why the underlying logic justifiably differs, not just the label).
- **`metadata_delete_keys: list[str] | None`** is the explicit **key-deletion mechanism** — this is
  the "sentinel" the deletion question (task 3055 planning note) asks to be decided. **Decision: no
  magic sentinel value.** A sentinel string or `None`-means-delete convention was rejected because
  it either collides with legitimate data (a real metadata value equal to the sentinel string) or
  reintroduces exactly the ambiguity task-2180's metadata-wipe incident was about (`append=False`
  silently meaning "replace with nothing" — `backends/sqlite_task_backend.py:3214-3226`). A
  dedicated list parameter has no such ambiguity: a key is deleted iff its name appears in
  `metadata_delete_keys`, full stop; an *omitted* key — the common case — can never mean delete.
  A key present in **both** `metadata_patch` and `metadata_delete_keys` is a `ValidationError`
  (contradictory intent, same fail-loud posture as `update_edge`'s `invalid_at` /
  `clear_invalid_at` mutual-exclusivity check, `:2134-2139`). Reserved keys are rejected in
  `metadata_delete_keys` exactly as in `metadata_patch` — a caller cannot delete `created_at`
  either.
- **`reason`**: required (non-empty) when `content` is supplied; optional when only the
  metadata-patch arm is used (§4 — this is one concrete piece of the differential bar). Not stored
  in the patched payload itself — flows to the write journal only (§5).
- **At least one of `content`, `metadata_patch`, `metadata_delete_keys` must be non-empty.**
  Supplying none is a `ValidationError` (mirrors `update_edge`'s equivalent
  neither-argument-supplied check, `:2140-2144`).
- **The point id / UUID is preserved and echoed in the result envelope** — e.g.
  `{'status': 'updated', 'store': 'mem0', 'id': memory_id, 'content_amended': bool,
  'metadata_patched': bool, ...}`, mirroring `update_edge`'s `{'status': 'reassigned', 'store':
  'graphiti', **result}` convention (`services/memory_service.py:3785`) — so a caller can assert
  identity stability directly from the response, not by re-fetching.

Per **INV-5**, the follow-up must **extract** `_MEM0_MANAGED_METADATA_KEYS` and the
read-existing-payload-then-forward-custom-subset logic currently private to
`scripts/tag_cgl_eta_rehome_scope.py` into a shared module that both the script and the new
service path import — never copy the constant or restate mem0's payload-overwrite semantics in a
second place. §6 makes the specific module-location call.
