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

2. **Two maintenance scripts deleted records that arguably only needed a patch —
   permanently losing `created_at` and provenance.** *(The "forced" causal claim here is
   corrected below; see the 2026-08-02 note.)*
   `fused-memory/scripts/prune_recon_cycle_summaries.py:25-30` states this outright: *"Mem0/Qdrant
   exposes `delete_memory` but no in-place payload-update primitive on this path — the same
   constraint documented in `scripts/sweep_orphan_flag_markers.py` (task-1659) ... Re-tagging via
   delete+re-add would also change `created_at` and lose provenance. So the effective operation
   here is PRUNE-to-N."* `scripts/sweep_orphan_flag_markers.py` (task-1659) hit the identical wall
   and also deleted rather than retagged.

   > **Correction (2026-08-02, task 3175 / escalation `esc-3175-1`).** Re-verified against this
   > repo, this "forced" claim does not hold for either script — the missing in-place-update
   > primitive was not the actual blocker in either case. For `sweep_orphan_flag_markers.py`:
   > backfilling the missing `kind` key would restore no capability, because nothing reads these
   > Mem0 marker records — task 2406 retired the Mem0 marker write path,
   > `fused-memory/src/fused_memory/reconciliation/flag_dedup.py:127-131` confirms reads in that
   > module never consult Mem0 and the recon_ledger SQLite table is the sole read source, and the
   > only live runtime reader of the Mem0 marker population is a reaper that enumerates markers
   > only to delete them (`_sweep_stale_mem0_flag_markers`,
   > `reconciliation/stages/task_knowledge_sync.py:1272-1340`). For
   > `prune_recon_cycle_summaries.py`: backfilling `recon_pool` would make pre-existing piles
   > visible to standing enforcement's filter (`reconciliation/summary_pool.py:314`), but the
   > eviction sort key (`summary_pool.py:365-388`) is `(is_ledger_stamp, has_parseable_created_at,
   > created_at)` ascending with the head deleted, against a cap of 2 hardcoded at
   > `reconciliation/stages/memory_consolidator.py:83` — a backfilled pre-existing *narrative*
   > summary (i.e. not a `record_type=ledger_stamp` record, which sorts to the tail and survives)
   > sorts to the head — `is_ledger_stamp=False` first, then `0` for a parseable `created_at` ahead
   > of a sibling whose `created_at` is still missing/unparseable, then oldest-`created_at` first —
   > and is deleted on the next cycle in which the pool exceeds the cap of 2. Preserving
   > `created_at` makes that outcome *more* certain, not less; retagging there would have been
   > deferred deletion, not a fix. See task 3175 and `esc-3175-1` for the full evidence trail. This
   > correction does not
   > reopen the DECISION recorded in §3-§7 of this document (building `update_memory` stands on its
   > own merits) — it corrects only this specific misattributed justification for it. Amending the
   > adjudicated decision text itself is task 3055/3088 owner territory; this note is deliberately
   > additive, not a rewrite.

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
- **`metadata_mode='replace'` with an absent or empty `metadata_patch` is a `ValidationError`**,
  naming both arguments. Decided explicitly here rather than left to fall out of the other gates,
  because the combination is otherwise *reachable and silently destructive*: the at-least-one-arm
  check below is satisfied by `content` alone, and the replace+`metadata_delete_keys` rejection one
  bullet down does not apply, so
  `update_memory(memory_id=..., content='new text', metadata_mode='replace')` would pass every
  other gate and — per §5(b)'s combined-call rule — wipe every custom provenance key on the record
  (`kind`, `src_project`, `topic`, …). That is precisely the task-2180 metadata-wipe failure mode
  (`append=False` silently meaning "replace with nothing") this section cites two bullets down as
  its own reason for rejecting a magic deletion sentinel, and it would defeat the
  provenance-preservation guarantee §6 calls the entire point of the tool. **"Replace with nothing"
  is therefore never an implicit wipe**: a caller that genuinely wants to clear custom keys must
  name them in `metadata_delete_keys`, where the intent is explicit and auditable in the journal
  row. Note the asymmetry is deliberate — `metadata_mode='merge'` (the default) with no
  `metadata_patch` is *not* an error, because an empty merge is a well-defined no-op on the
  metadata arm; only `replace` turns emptiness into destruction.
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
- **`metadata_mode='replace'` together with a non-empty `metadata_delete_keys` is also a
  `ValidationError`**, naming both arguments in the message. Decided here rather than left for the
  implementer to guess: replace mode (§5(b)) already replaces the entire custom-provenance subset
  with exactly what `metadata_patch` supplies, so every custom key not named there is dropped
  regardless — a delete list under replace mode is therefore simultaneously **redundant** (the
  named keys are dropped anyway) and **contradictory** (it implies some keys survive replace that
  in fact do not), and §5(b)'s replace path maps to a single `overwrite_payload` call with no place
  for a delete list to apply against. Rejecting the combination — rather than silently no-op'ing
  it, silently applying the delete after the overwrite, or guessing at an ordering — matches the
  same fail-loud posture already applied one bullet up (the both-lists key conflict) and to
  `update_edge`'s `invalid_at`/`clear_invalid_at` mutual-exclusivity check (`:2134-2139`): silently
  discarding a caller-supplied argument is exactly the class of "caller believes it wrote something
  it did not" failure the reserved-key rejection above exists to prevent.
- **`reason`**: required (non-empty) when `content` is supplied; optional when only the
  metadata-patch arm is used (§4 — this is one concrete piece of the differential bar). Not stored
  in the patched payload itself — flows to the write journal only (§5).
- **At least one of `content`, `metadata_patch`, `metadata_delete_keys` must be non-empty.**
  Supplying none is a `ValidationError` (mirrors `update_edge`'s equivalent
  neither-argument-supplied check, `:2140-2144`).
- **A `memory_id` that does not exist is a structured `{'error_type': 'MemoryNotFound'}` rejection,
  in every arm** — never a success envelope. Decided here because the §5(b) metadata-only fast paths
  make the alternative a *silent-success hole*, not merely an unspecified case: the
  `metadata_patch`-alone and `metadata_delete_keys`-alone routes call
  `AsyncQdrantClient.set_payload`/`delete_payload` directly with no read-before-write, and Qdrant
  treats both as **no-ops for an unknown point id**, returning `acknowledged`/`completed` rather
  than an error. Without an explicit existence check, `update_memory` would emit the success
  envelope below (`{'status': 'updated', 'id': memory_id, 'metadata_patched': True}`) **and** a
  journal row claiming a write that never touched anything — the exact "caller believes it wrote
  something it did not" failure class the reserved-key rejection above exists to prevent, and a
  violation of this repo's no-silent-fail-soft invariant that `Mem0Backend.get_point_by_id`'s
  docstring (`backends/mem0_client.py:426-446`) goes out of its way to honour by distinguishing
  *absent* from *timed-out*. **Every arm therefore verifies existence first via the §5(c) read leg**
  (`MemoryService.get_memory_by_id` → `Mem0Backend.get_point_by_id`) before any write. A read
  *timeout* (as opposed to a confirmed absence) must **not** be reported as `MemoryNotFound` — it is
  a distinct transient error, and `get_point_by_id` already separates the two so the implementer
  does not have to infer it. This makes the metadata-only fast paths one read + one write; the
  one-backend-**write** invariant stated at the end of §5(b) is unaffected, since the existence
  check is a read.
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

## §4 — Decision 2: authorization model, and the metadata-only vs content-amend bar

### The honest baseline

**There is no authorization layer on this write surface today.** Verified: `delete_memory`
(`server/tools.py:1941-2012`) and `update_edge` (`:2058-2156`) — the two closest existing
in-place/destructive Mem0-and-Graphiti mutation tools — have no allowlist, no confirmation flag,
no dry-run, and no role check. Their guard prologues do only identity resolution
(`_resolve_identity`), project-id canonicalization/validation
(`_canonicalize_project_id_arg`/`validate_project_id`), and the known-project gate
(`_known_project_gate`) — none of which is *authorization*, all of which would pass for any
caller. `_install_safe_tool_wrapper` (`server/main.py:1639-1682`) is a `BaseException` catcher
around FastMCP's dispatch chokepoint — defence-in-depth against a tool handler crashing the
server, not an authorization boundary.

Two real precedents exist, and this decision draws on both without pretending either is stronger
than it is:

- **`add_system_record`** (`server/tools.py:1282-1341`): rejects any `agent_id` not starting with
  `'recon-stage-'` with `error_type='DedupExemptNotPermitted'`, checked *before* project/backlog
  validation (`:1327-1332`). Its own docstring (`:1302-1307`) concedes the honest limit: *"this
  checks the agent_id **string** the caller provides — it is a naming convention enforced
  server-side, not cryptographic authorization. A caller that deliberately passes
  agent_id='recon-stage-\*' can still reach this path."* This decision adopts that same honesty
  rather than claiming a security property the system cannot deliver: `agent_id` is
  self-reported by the caller, and every guard built on it (including the one this decision adds)
  is a **misuse deterrent for cooperating callers**, not a defense against an adversarial one.
- **`middleware/recon_write_policy.py:1-30`** — a **structural** precedent, not a code dependency.
  This module gates a completely different write surface (`update_task` / `set_task_status` on
  *tasks*, consulted from `TaskInterceptor`) — it has no relationship to Mem0 records and the new
  tool must not import it. What it offers is the **shape** worth reusing: an `agent_id`-prefix gate
  (Gate 1-3 in `check()`) WITH a carve-out for a "non-load-bearing, metadata-only, merge-mode"
  write (`is_terminal_annotation_clear`/`is_terminal_annotation_add`, `:441-602`) plus a mandatory
  `_causation_id` tracing co-key riding alongside it (`_CAUSATION_TRACING_KEYS`, `:151`). That is
  *exactly* the shape "does a metadata-only patch get a lighter bar" needs — a prefix gate plus an
  explicit metadata-only carve-out plus mandatory attribution — so this decision mirrors that
  shape for the Mem0 write surface instead of inventing a new authorization-tier vocabulary
  (INV-5). It is cited for its *design*, never called into.

### The authorization model (DECIDED)

**A new config section, `Mem0UpdateConfig`, added as its own top-level field on
`FusedMemoryConfig`** (`config/schema.py`) — deliberately **not** nested under
`ReconciliationConfig` alongside the `procedural_knowledge_near_dup_*` knobs, even though those are
the closest existing precedent for a write-time server-enforced guard. `schema.py`'s own ownership
note on those knobs (`:1071-1081`) already flags the trap: they live on `ReconciliationConfig`
only because the guard they drive is "the write-time counterpart to Stage-1's reactive
procedural_knowledge consolidation," and explicitly warns *"if this guard ever grows independent
of reconciliation, move these two fields to a dedicated server-owned config section instead of
assuming colocation implies subsystem ownership."* The Mem0-update authorization gate is general —
recon Stage 1 is merely its first sanctioned caller, not its owner — so it starts in the place that
note says a growing guard should move *to*, rather than repeating the trap and needing a later
migration. `CuratorConfig` / `TicketJanitorConfig` / `SummaryRebuildConfig` are the existing
precedent for a named capability getting its own top-level config section instead of being
shoehorned into `ReconciliationConfig`.

```python
class Mem0UpdateConfig(BaseModel):
    """Authorization + kill-switch config for the update_memory MCP tool (task 3088).

    Two independently-configurable allowlists implement the differential bar decided in
    plans/mem0-in-place-update-decision.md §4. Fail-safe default is a KILL SWITCH plus
    narrow allowlists, not open allowlists — an unconfigured or partially-corrupt config
    permits as little as possible (fail CLOSED), since this is a mutation-authorization
    gate, not a soft-block guard where fail-open-on-error would be the safe direction.
    """
    enabled: bool = Field(default=True)  # named kill switch; False rejects every caller
    content_amend_allowed_agent_prefixes: list[str] = Field(default_factory=lambda: ['recon-stage-'])
    metadata_patch_allowed_agent_prefixes: list[str] = Field(default_factory=lambda: ['recon-stage-'])
    storm_threshold: int = Field(default=20)          # content-amend calls per agent_id per window
    storm_window_seconds: float = Field(default=3600.0)
```

- **Fail-safe default**: both allowlists ship seeded with exactly `['recon-stage-']` — the same
  literal prefix `add_system_record` already uses (INV-5: reuse the string, don't mint a second
  one) — which satisfies the task's minimum bar ("must at minimum admit recon Stage 1
  `memory_consolidator`," named in task 3088's scope) out of the box, with nothing else able to
  call either arm until an operator deliberately widens a list.
- **Hot-reload tier: green**, alongside the existing `procedural_knowledge_near_dup_*` leaves in
  `config/reload.py`'s `RELOADABLE_FIELDS` (`:43-55`) — but the follow-up **must** satisfy
  `reload.py`'s own reload-safety rule (module docstring, `:8-16`) first: a leaf is only
  reload-safe if every consumer re-reads it *live* from the shared config object on each call. The
  implementer must write a `resolve_mem0_update_authorization(...)`-shaped live-read helper (same
  pattern as `resolve_near_dup_guard_enabled` / `resolve_near_dup_threshold` in
  `server/near_duplicate_guard.py`) before adding `mem0_update.*` leaves to `RELOADABLE_FIELDS` —
  registering the leaf without the live-read helper would silently reintroduce a restart
  requirement disguised as a hot-reload.
- **Named kill switch**: `mem0_update.enabled`. `False` rejects every caller regardless of
  `agent_id`, with a structured `{'error_type': 'Mem0UpdateToolDisabled'}` — same shape convention
  as every other rejection on this surface.
- **Model**: this section is the direct analogue of `plans/stage1-entity-standing-decision-prd.md`'s
  `### Authorization gate (server-side, in the tool handler)` (line 110) — a server-side gate in
  the tool handler, checked first (mirroring `add_system_record`'s ordering, `:1323-1326`, "before
  any project/backlog/category validation work... happens on its behalf"), returning a structured
  rejection (INV-1) rather than raising.
- **Attributable writes**: both arms require a resolved `agent_id` matching their respective
  allowlist. The content-amend arm additionally requires a non-empty `reason` (§3) — this is the
  concrete "agent_id plus a reason" requirement task 3088 names, applied to the arm where a silent
  rewrite of semantic content is possible. Every accepted call is journaled regardless of arm or
  reason (§5) — attribution is never optional at the journal layer, only at the argument-validation
  layer.

### The differential bar (DECIDED: yes, metadata-only gets a lighter bar — but never an unguarded one)

Task 3055's third scope question is answered on three concrete axes, not intuition:

1. **Reconciliation semantics.** `scripts/tag_cgl_eta_rehome_scope.py:44-47` deliberately bypasses
   `MemoryService` today specifically because *"a cosmetic provenance tag must not trigger a recon
   cycle over these very facts."* Decision: the content-amend arm **always** emits
   `EventType.memory_updated`; the metadata-patch arm does **not** emit it by default (an
   `emit_event: bool = False` internal service-layer parameter exists for a future caller that
   wants one, but no MCP-level argument surfaces it in the initial ship — task 3088 may add one if
   a concrete consumer needs it).
2. **Write path / embedding cost.** The content-amend arm re-embeds unconditionally (inherent to
   `mem0.update()`, not a choice — §5). The metadata-patch arm **never** re-embeds — it routes
   around mem0's `Memory.update` entirely (§5) precisely so tagging survivors stays cheap and
   never perturbs semantic ranking.
3. **Blast radius.** Metadata is not inert: `get_memories_by_metadata` / `count_by_metadata` make
   it the input to *deterministic* retrieval, so a bad patch corrupts lookup silently, not loudly.
   **"Lighter bar" therefore means a lower tier, never an unguarded one:** reserved-key rejection
   (§3) and write-journal logging (§5) apply to **both** arms, unconditionally, with no
   metadata-only exemption.

Two further, concrete tiering mechanisms (beyond the three axes above) implement "lighter" at the
authorization layer specifically:

- **Independently configurable allowlists** (not independently *valued* by default — see above):
  an operator can widen `metadata_patch_allowed_agent_prefixes` (e.g. to admit an interactive
  curator session prefix for consolidation-closure tagging) without touching
  `content_amend_allowed_agent_prefixes` at all. The bar is structurally decoupled even though it
  ships identical on day one.
- **Mandatory vs. optional `reason`** (§3): content-amend requires proof of "why"; metadata-only
  does not.
- **Storm-counter exemption** (below): only content-amend calls count toward the storm threshold.
  A mistagged metadata patch is cheap to notice (`get_memory_by_id`) and cheap to correct (another
  metadata-only patch); a runaway silent content rewrite is not.

### INV-4 `storm-escape-required`

An in-place content amend is a silent-rewrite primitive — the archetypal fail-soft-adjacent path
INV-4 exists for. This decision specifies the escape rather than leaving it to the implementer:

- **What is counted**: content-amend (`content` arm) calls only, keyed per `agent_id`. Metadata-only
  patches are excluded (see differential-bar rationale above).
- **Mechanism**: a rolling-window counter mirroring the **shape** of
  `ReconciliationHarness`'s existing storm counters (`reconciliation/harness.py` — e.g. the
  `_dead_owner_suppressions` deque + `dead_owner_suppression_storm_threshold`/`window_seconds` +
  single-fire-per-window escalation via a stable finding fingerprint, `:159-168`, `:480-494`) —
  implemented as **new code**, owned by `MemoryService` alongside its other instance state
  (`services/memory_service.py:758-772`), not by importing the harness's counter object.

  **Correction — this replaces a factually wrong justification in an earlier draft of this
  section.** Reuse was previously rejected here on the claim that the harness's counters "live
  inside the per-process `ReconciliationHarness` instance, a different lifetime and process from
  the MCP server's tool-dispatch path that will host this counter." That claim is **false**:
  `ReconciliationHarness` is constructed at `server/main.py:891-897`, inside the `if
  config.reconciliation and config.reconciliation.enabled:` block opened at `:702`, in the *same*
  asyncio process that hosts FastMCP tool dispatch — there is no process boundary to cross. The two
  real reasons not to bind this counter to the harness are wiring/reachability facts, not a process
  boundary:
  1. **Conditional existence.** Harness construction is gated on `config.reconciliation.enabled`
     (`:702`). A storm alarm bound to the harness would silently cease to exist whenever
     reconciliation is disabled — vanishing in exactly the degraded configuration where an
     unattended rewrite loop is *least* likely to be noticed any other way. INV-4 requires the
     escape to exist unconditionally, so it cannot depend on a component that is itself
     conditionally constructed.
  2. **Construction order.** `MemoryService(config)` is constructed at `:537` — before
     `curator_escalator` (`:607`) and before `ReconciliationHarness` (`:891`), both built later in
     the same startup function. No harness or curator-escalator reference is in scope at
     `MemoryService.__init__` time. Reusing either would require both a post-construction setter
     **and** acceptance of their conditional lifetimes; the design below needs a setter (for the
     project-root map only — see "Resolving `project_id` → `project_root`" below) but crucially
     **not** a dependency on any conditionally-constructed component, which is what reason 1 is
     about.

  Same *shape* as the harness counters (deque of timestamps per `agent_id`, threshold + window
  config) — new code, deliberately not a new vocabulary. The single-fire-per-window *folding* of
  repeated breaches is **not** hand-rolled, though — see the escalation channel below.
- **Threshold**: `mem0_update.storm_threshold` (default 20) content-amend calls per `agent_id`
  within `mem0_update.storm_window_seconds` (default 3600s / 1h) — both green-tier hot-reloadable
  alongside the allowlists above, for the same reason.
- **Emission channel (DECIDED): a new, zero-arg `Mem0UpdateStormEscalator`**, in
  `fused-memory/src/fused_memory/middleware/`, modelled directly on
  `middleware/scope_violation_escalator.py` — the house pattern for exactly this job (its own
  module docstring already records that its design mirrors `CuratorEscalator`). This is the
  load-bearing half of this finding: `MemoryService` has **no** escalation API today — a repo-wide
  grep for `escalat` in `services/memory_service.py` returns exactly one hit, a docstring mention
  at `:3282` — so without naming a concrete channel here, an implementer would have had to invent
  the delivery path, risking a counter that counts correctly but reaches nobody. Mirroring
  `ScopeViolationEscalator` means reusing, not reimplementing (INV-5):
  - **Defensive import** of the optional `escalation` workspace package
    (`scope_violation_escalator.py:50-61`) — when the package is absent, the escalate call is a
    logged no-op and the triggering `update_memory` call still succeeds, consistent with the
    "monitoring alarm, not a rate limiter" rule below.
  - **A per-project `EscalationQueue` cache keyed on `project_root`** (`_queue_for`,
    `scope_violation_escalator.py:126-138`), landing in `{project_root}/data/escalations` — the
    same queue an operator already watches for scope-violation and curator escalations.
  - **Burst folding via `escalation.dedupe.submit_or_dedupe`**
    (`scope_violation_escalator.py:247`), with a `dedupe_fingerprint` computed over `(category,
    agent_id)` (i.e. `compute_content_fingerprint('mem0_in_place_update_storm', agent_id)`) and
    `infra_dedupe_categories=('mem0_in_place_update_storm',)`. This **replaces** the hand-rolled
    "single-fire-per-window" dedup an implementer would otherwise have had to build by hand: a
    sustained storm from one `agent_id` folds into one pending escalation (incrementing
    `dedupe_count`) instead of paging once per call past threshold — satisfying the fingerprint
    requirement below by **reuse**, not reimplementation.
  - **Ownership/wiring**: constructed zero-arg inside `MemoryService.__init__`, exactly like
    `ScopeViolationEscalator()` at `server/main.py:633`/`:650` — no reference to the harness or to
    `curator_escalator` is needed, no startup reordering, and it behaves identically whether
    reconciliation is enabled or disabled. This is precisely what makes reason 1 above hold: the
    alarm's existence never depends on `config.reconciliation.enabled`.

  **Resolving `project_id` → `project_root` (DECIDED — this is the gap that would otherwise force
  the implementer to invent a delivery path).** `_queue_for` needs a filesystem `project_root`, but
  `update_memory`'s signature (§3) carries only `project_id`, and **`MemoryService` has no
  `project_root` anywhere** — verified: a grep for `project_root` in
  `services/memory_service.py` returns zero hits, and `ScopeViolationEscalator`'s own callers get
  the root passed in from `server/main.py`'s `_known_projects_map`. Nothing analogous is in scope
  inside `MemoryService`. The decision:

  1. **Inject the existing registry, do not derive a new one.** `server/main.py:627` already builds
     `_known_projects_map = build_known_projects_map(_primary_root, _extra_roots)` — a
     `{project_id: project_root}` map that is explicitly "the single source of truth for the project
     registry" and is already handed to `ReconciliationHarness` (`:894`) and `TicketJanitor`
     (`:985`) as `known_projects=`. `update_memory` resolves its `project_id` against that same
     snapshot. Note `:627` sits **before** the `config.reconciliation.enabled` block opened at
     `:702`, so the map is built unconditionally — reason 1 above still holds.
  2. **Delivered via a `set_known_projects(...)` setter**, following the established
     `set_event_buffer`/`set_write_journal`/`set_recon_ledger`/`set_planned_registry` pattern
     (`services/memory_service.py:774-788`) and called at server startup after `:627`. A setter is
     required because `MemoryService` is constructed at `:537`, before the map exists at `:627`;
     this is a pure-data injection with no lifetime coupling to any conditional component, which is
     why it does not reintroduce reason 1's problem.
  3. **Fallback when the root cannot be resolved — structured WARN log + no-op, never a guess.**
     If the setter was never called, or `project_id` is absent from the map, the escalator logs a
     structured WARN (naming the `project_id`, the storm count, and the window, so the signal is
     still recoverable from logs) and skips submission. The triggering `update_memory` call still
     succeeds — identical to the `HAS_ESCALATION=False` defensive-import path above, so there is
     one no-op posture for the alarm, not two. **Explicitly forbidden: falling back to
     `config.taskmaster.project_root`**, which defaults to `'.'` and would silently write
     escalations into the server process's cwd — an alarm that appears to fire but reaches nobody
     is strictly worse than one that says in the log that it could not fire.
- **Who hears about it, and how**: `MemoryService.update_memory`'s post-write observe step (§5(a)
  step 7) calls the owned `Mem0UpdateStormEscalator`'s report method once the per-`agent_id` count
  reaches `mem0_update.storm_threshold` within the configured window. It submits a
  `severity='blocking'` escalation, category `mem0_in_place_update_storm`, fingerprinted on
  `(category, agent_id)` as above — not on count/window, so repeated storms from the same agent
  fold into a single pending escalation instead of paging once per breach (mirroring
  `_DEAD_OWNER_STORM_FINDING`'s stable-fingerprint / variable-detail split, `harness.py:164-168`).
  `blocking`, not `info`, because a runaway silent-rewrite loop is a "someone must look at this
  now" condition, matching this repo's `recon_watchdog_kill_storm` precedent — not a routine
  triage item.
- **Behaviour when reconciliation is disabled: unchanged.** The counter and the escalator are both
  owned by `MemoryService` and constructed unconditionally in its `__init__`; neither has any
  dependency on `config.reconciliation.enabled`, `ReconciliationHarness`, or `curator_escalator`, so
  this alarm fires identically regardless of whether reconciliation is on.
- **This is a monitoring alarm, not a rate limiter.** Crossing the threshold does **not** reject
  the triggering call — the write still proceeds and is still journaled. INV-4's house pattern
  (the harness storm counters cited above) is uniformly "observe the rate, escalate loudly," never
  "block at N+1." A hard block here would risk a legitimate large consolidation cycle failing
  mid-run over its own success count. The implementer must not invent a blocking behavior that was
  never decided.

## §5 — Implementation hand-off

### (a) Layering

Mirrors `update_edge` exactly: MCP tool in `server/tools.py` → `MemoryService.update_memory` →
`_journaled_backend_call(backend='mem0', operation=...)` → `_write_journal.log_write_op` →
conditional `_emit_event`. The tool-level guard sequence, in order:

1. `_resolve_identity(agent_id, session_id, ctx)` — must run first; nothing downstream can check
   `agent_id` against an allowlist before it is resolved.
2. **The §4 authorization gate**, checked immediately next — before project canonicalization,
   before `store`/arm validation, before anything else — mirroring `add_system_record`'s explicit
   rationale for the same ordering (`:1323-1326`): the gate "is the whole point of this tool," so
   an unauthorized caller is rejected before any other validation work happens on its behalf. Which
   allowlist(s) apply depends on which arm(s) the call requests (content → the content-amend
   allowlist; metadata_patch/metadata_delete_keys → the metadata-patch allowlist; both arms in one
   call → both allowlists must pass).
3. `_canonicalize_project_id_arg` → `validate_project_id` → `_known_project_gate` — reused verbatim
   from `delete_memory`'s prologue (`:1976-1983`). **No `_backlog_gate`** — that gate exists for
   tools that create new backlog pressure (`add_memory`, `add_system_record`); `update_memory`
   mutates an existing record and creates none, matching `delete_memory`'s (not `add_memory`'s)
   omission of it.
4. `store` validated against `_VALID_STORES`; `store='graphiti'` is rejected with a
   `ValidationError` naming `update_edge` (§3).
5. Arm validation: at least one of `content`/`metadata_patch`/`metadata_delete_keys` non-empty;
   `content` non-empty when supplied; `reason` non-empty when `content` is supplied; reserved
   (`_MEM0_MANAGED_METADATA_KEYS`) keys rejected from both `metadata_patch` and
   `metadata_delete_keys`; a key present in both is rejected as contradictory (§3).
6. `_extract_causation(metadata, agent_id)` (`:479-497`) — unchanged from every sibling tool.
7. Dispatch to `memory_service.update_memory(...)`, which performs the storm-counter
   observe-and-maybe-escalate step (content-amend calls only, post-write) alongside the journal and
   event steps below. "Maybe-escalate" means a call into `MemoryService`'s own, zero-arg-constructed
   `Mem0UpdateStormEscalator` (§4) once the per-`agent_id` count reaches
   `mem0_update.storm_threshold` within the window — never a call into `ReconciliationHarness`,
   which is conditionally constructed and may not exist at all (§4).

### (b) The write-path fork — the load-bearing mechanical finding

Verified directly against this environment's pinned mem0 (`.venv/lib/python3.13/site-packages/mem0`,
same dependency the fused-memory venv resolves):

- **`AsyncMemory.update`'s own docstring is wrong.** `mem0/memory/main.py:2253-2254` reads:
  *"metadata (dict, optional): Additional metadata to update. Existing metadata fields not
  specified here will be preserved."* This is false. `_update_memory` (`:2449-2509`) builds
  `new_metadata = deepcopy(metadata) if metadata is not None else {}` (`:2463`) — a **fresh** dict,
  not a merge — and re-attaches only nine keys from the existing payload: `data` (`:2465`,
  overwritten to the new content), `hash` (`:2466`, recomputed), `created_at` (`:2467`, preserved
  from the existing point), `updated_at` (`:2468`, always regenerated to now), `user_id` /
  `agent_id` / `run_id` (`:2471-2476`, only if not already present in the caller's `metadata`),
  and `actor_id` / `role` (`:2478-2481`). Every one of these nine keys is exactly
  `_MEM0_MANAGED_METADATA_KEYS`'s membership (`scripts/tag_cgl_eta_rehome_scope.py:235-238`) — the
  script's constant is not a guess, it is this exact list, independently confirmed by reading
  `_update_memory`'s source. Do not trust the docstring; trust `_update_memory`.
- **`Memory.update` always supplies a vector, so Qdrant's payload-only path is unreachable through
  it.** `_update_memory` unconditionally computes `embeddings` (`:2483-2488`) and calls
  `self.vector_store.update(vector_id=memory_id, vector=embeddings, payload=new_metadata)`
  (`:2490-2495`) — `vector` is never `None`. Qdrant's own `update` wrapper
  (`mem0/vector_stores/qdrant.py:347-370`) branches on `if vector is not None and payload is not
  None:` (`:356`) → full-point `upsert` with a `PointStruct` (`:357-358`); only the `else` arm
  (`:359-370`) contains the payload-only `set_payload` call (`:361-365`). Since `_update_memory`
  always supplies both, that `if` branch always wins — the `set_payload` arm is **structurally
  unreachable** through `Memory.update`/`Mem0Backend.update`, confirming the plan's premise with
  the exact mechanism, not just the symptom.

**Consequence (DECIDED):**

- The **content-amend arm** routes through the existing `Mem0Backend.update` (`mem0_client.py:238-262`)
  unchanged. Re-embedding, the `updated_at` rewrite, and the mem0-internal `db.add_history` row
  (`main.py:2498-2508`) are all *correct* here — a real content change should perturb ranking and
  should leave its own trace in mem0's history table, in addition to the fused-memory-level journal
  and `memory_updated` event (§4).
- **Combined call** (`content` *and* `metadata_patch`/`metadata_delete_keys` supplied together —
  §3 deliberately allows this, to avoid the torn-intermediate-state problem a two-tool design
  would have had): when `content` is present, the metadata-arm changes are folded into the *same*
  `Mem0Backend.update` call, not routed separately. Concretely: read the existing payload (c),
  strip mem0-owned keys, apply the `metadata_patch` merge/replace and `metadata_delete_keys`
  removals to that custom subset exactly as the metadata-only path would, then forward the
  resulting dict as `Mem0Backend.update`'s `metadata=` argument alongside the new `data=content` —
  the same read-modify-forward dance `tag_cgl_eta_rehome_scope.py`'s `apply_tags` already performs
  (§2 point 3), just with the metadata delta coming from the caller instead of a fixed scope tag.
  There is exactly one write, one journal row, and one `memory_updated` event (content-amend's
  event rule wins whenever `content` is present) — never two writes for one call.
- The **metadata-only arm** (no `content` in the call) must **not** route through
  `Mem0Backend.update` / mem0's `Memory.update` at all — doing so would needlessly re-embed,
  rewrite `updated_at`, and append a spurious mem0 history row for what may be a purely cosmetic
  tag. Instead, a **new**, payload-only `Mem0Backend` method(s) using the backend's existing
  `_get_async_qdrant()` (`mem0_client.py:272-281`) and Qdrant's own partial-payload primitives —
  verified present on `AsyncQdrantClient`: `set_payload`, `delete_payload`, `overwrite_payload`.
  Routing among these three is decided by *which arguments the call supplies*, read as one coherent
  decision table rather than three unrelated primitives:
  - **`metadata_patch` alone** (`metadata_mode='merge'`, the default, no `metadata_delete_keys`):
    `AsyncQdrantClient.set_payload(payload=<custom subset>, points=[memory_id])`. Qdrant's
    `set_payload` is *already* a genuine partial-payload merge at the storage layer (unlike mem0's
    `_update_memory`) — no read-modify-write cycle is needed to compute the new payload.
    `tests/test_mem0_qdrant_integration.py:69-90` (`test_set_payload_without_vector`) already covers
    this primitive.
  - **`metadata_delete_keys` alone** (no `metadata_patch`): `AsyncQdrantClient.delete_payload(
    keys=[...], points=[memory_id])` — also native, also no read-modify-write cycle.

  **Both fast paths still perform the §3 existence check first.** "No read-modify-write cycle"
  means the *new payload* is computed without reading the old one — it does **not** mean the call
  skips the §5(c) read leg. Qdrant returns `acknowledged`/`completed` for `set_payload` and
  `delete_payload` against an unknown point id, so omitting the existence check here is exactly what
  would turn these two fast paths into silent-success holes (§3). Each is therefore one existence
  read + one write; the one-backend-**write** invariant below counts writes, not reads, and is
  unaffected.
  - **Both `metadata_patch` and `metadata_delete_keys` supplied together** (the
    `metadata_mode='replace'` + `metadata_delete_keys` combination is instead rejected at the
    boundary, §3): a single read-modify-`overwrite_payload` write — read the existing payload via
    (c) below, apply the merge and the deletions to the custom subset **in memory**, then issue
    exactly one `overwrite_payload` call with the mem0-owned key subset re-attached underneath.
    **Decided this way rather than issuing `set_payload` followed by `delete_payload` as two
    independent native calls**: two round-trips have no ordering guarantee, no atomicity, and no
    rollback — if the second call fails, the record is left half-patched while the journal row
    (§5(a)) claims the whole edit landed, reintroducing *inside the unified tool* precisely the
    "torn intermediate state" §3 cites as its own reason for rejecting the two-tool design. Routing
    through read-modify-`overwrite_payload` instead adds no new machinery — it is the exact shape
    plain `replace` mode already needs (next bullet), and it reuses the (c) read leg: one
    round-trip, one atomic payload write, no ordering question to answer.
  - **`metadata_mode='replace'` alone** (no `metadata_delete_keys`): the entire custom-provenance
    subset is replaced with exactly what `metadata_patch` supplies. This is the same
    read-modify-`overwrite_payload` shape as the combined merge+delete case immediately above —
    `overwrite_payload` replaces the **entire** point payload, so the mem0-owned key subset must be
    read back and re-attached before calling it, or the point loses its own `data`/`hash`/
    `created_at` and becomes unreadable by mem0's own `get`/`search`. This read reuses (c) below —
    no new read primitive.
  - Reserved-key rejection at the tool boundary (§3) means none of these operations ever needs to
    defend against a caller supplying a mem0-owned key directly.

  **Invariant (checkable, not aspirational): every `update_memory` call performs exactly one
  backend write, in every arm combination.** One `Mem0Backend.update` call when `content` is
  present (the "Combined call" case above folds any metadata delta into that same call); otherwise
  exactly one of `set_payload` / `delete_payload` / `overwrite_payload` on the metadata-only path
  per the table above. This is what makes the "one journal row, one event" guarantee (§5(a)) true
  uniformly across every arm combination, not only on the content arm.

### (c) Read-before-write

`MemoryService.get_memory_by_id` (`:3339-3375`, shipped by task 2765 — commit `3a4de2a71e`)
already returns the full raw Qdrant payload via `Mem0Backend.get_point_by_id`. This is the existing
read leg for **both** the content-amend arm's metadata-reforwarding (mirroring
`tag_cgl_eta_rehome_scope.py`'s `apply_tags`) **and** the metadata-patch arm's replace-mode
read-before-`overwrite_payload`. No new read primitive is needed anywhere in this design.

### (d) The TDD plan the implementer inherits

Harness: `create_mcp_server(mock_service)` + `await mcp_server._tool_manager.call_tool(...)` + a
local `_parse_tool_result` JSON decoder — exactly `tests/test_update_edge_tool.py:14-75`'s pattern
(its own header comments itself as "Step N: RED tests (fail before step-N+1 implementation)," the
established convention for this repo's tool-level TDD). Extend the existing
`TestMem0BackendUpdate` class (`tests/test_mem0_client.py:379`, whose docstring currently records
that `Mem0Backend.update` "had zero callers") rather than starting a new one. Service-level tests
alongside `TestUpdateEdge`/`TestDeleteMemory` in `tests/test_memory_service.py`. Authz-gate tests
alongside `tests/server/test_add_system_record_gate.py`.

Required RED tests (enumerated so the follow-up inherits a concrete plan rather than re-deriving
one):

- Point-id/UUID stability across both arms (response echoes the same id the call was made with).
- `created_at` preservation across both arms.
- **Content amend with NO `metadata_patch`/`metadata_delete_keys` supplied preserves every
  pre-existing custom payload key verbatim** (e.g. `kind`, `src_project`, `topic`) — defends §2
  point 3 / §5(b)'s read-modify-forward requirement, and **is the load-bearing regression test in
  this list**: `created_at` (the bullet above) is one of the keys mem0's own `_update_memory`
  unconditionally re-preserves (§5(b)), so an implementation containing the exact payload-overwrite
  bug §2 point 3 documents — forgetting the read-existing-payload-then-reforward-custom-subset
  dance `scripts/tag_cgl_eta_rehome_scope.py`'s `apply_tags` already had to solve — would still pass
  the `created_at` test. This test asserts on keys mem0 does *not* restore on its own, so it is the
  only listed test that actually fails against that bug; it must not be dropped as redundant with
  the `created_at` test.
- Metadata-patch shallow-merge preserves unlisted existing custom keys (merge mode).
- `metadata_mode='replace'` replaces the custom subset but still preserves mem0-owned keys
  underneath.
- `metadata_delete_keys` removes exactly the named keys and nothing else.
- Reserved-key rejection: a mem0-owned key in `metadata_patch` or `metadata_delete_keys` is a
  `ValidationError`.
- A key present in both `metadata_patch` and `metadata_delete_keys` is a `ValidationError`.
- **`metadata_mode='replace'` supplied together with a non-empty `metadata_delete_keys` is a
  `ValidationError`** naming both arguments — defends §3's replace+delete combination rule.
- **`metadata_mode='replace'` with an absent/empty `metadata_patch` is a `ValidationError`** naming
  both arguments — including the specific reachable shape
  `update_memory(memory_id=..., content='new text', metadata_mode='replace')`, which passes every
  other gate. Assert additionally that the record's custom provenance keys (`kind`, `src_project`,
  `topic`) are **unchanged** after the rejected call, so the test fails against an implementation
  that validates but has already wiped. Defends §3's no-implicit-wipe rule (task-2180 regression).
  Companion positive case: `metadata_mode='merge'` with no `metadata_patch` is **not** an error.
- **A `memory_id` that does not exist is rejected with `{'error_type': 'MemoryNotFound'}` in every
  arm** — parameterised over the content-amend arm, the `metadata_patch`-alone arm, and the
  `metadata_delete_keys`-alone arm, since the latter two are the §5(b) fast paths where Qdrant's
  unknown-point-id no-op would otherwise manufacture a false success. Assert both that no backend
  write is issued **and** that no journal row is written. Separately assert that a read *timeout*
  from `get_point_by_id` surfaces as its own transient error, **not** as `MemoryNotFound`.
- **A metadata-only call supplying both `metadata_patch` (merge) and `metadata_delete_keys` in the
  same call routes through exactly one `overwrite_payload` backend call** — never a `set_payload`
  followed by a `delete_payload` — and the resulting payload shows both the merged keys and the
  removed keys. Defends §5(b)'s read-modify-`overwrite_payload` decision for the combined
  metadata-only case, and doubles as one concrete instance of the one-backend-write invariant
  §5(b) now states explicitly.
- `store='graphiti'` is rejected, error message names `update_edge`.
- Neither arm supplied → `ValidationError`.
- `content` supplied with empty/missing `reason` → `ValidationError`; metadata-only arm with no
  `reason` succeeds.
- Authz-gate denial: an `agent_id` not matching the relevant allowlist is rejected before any
  write, for each arm independently.
- `mem0_update.enabled=False` rejects every caller regardless of `agent_id`.
- `EventType.memory_updated` is emitted for the content-amend arm and **not** emitted by default
  for a metadata-only patch.
- **The metadata-only arm does not re-embed and leaves `updated_at` untouched.** Defends §4/§5(b)'s
  write-path/embedding-cost axis of the differential bar — the bullet above pins only the "no
  event" half of that asymmetry. Assertion shape: on the metadata-only path, the embedder (or, at
  the `Mem0Backend` boundary, `Mem0Backend.update`/mem0's `Memory.update`) is asserted **never
  called** (mock/spy), and the record's `updated_at` value is byte-identical before and after the
  call.
- Both arms produce a `WriteJournal.log_write_op`/`log_backend_op` row regardless of event
  emission.
- **A combined `content` + (`metadata_patch` and/or `metadata_delete_keys`) call produces exactly
  one backend write (one `Mem0Backend.update` call), one journal row, one `memory_updated` event,
  and a payload reflecting both the new content and the merged/deleted metadata.** Defends §5(b)'s
  "Combined call" decision — introduced by this document and, before this step, entirely untested
  despite §5(b) asserting the one-write/one-row/one-event guarantee explicitly; this test is what
  makes that assertion checkable rather than aspirational.
- Storm counter fires the `mem0_in_place_update_storm` escalation once `storm_threshold`
  content-amend calls from the same `agent_id` land within `storm_window_seconds`, and does
  **not** count metadata-only calls toward it; the triggering call still succeeds (§4).

Respect `fused-memory/pyproject.toml`'s pytest conventions: `asyncio_mode = "strict"` (`:33`),
default deselection of live tests (`addopts = "...-m 'not integration'"`, `:34`), and
`qdrant_skipif()` (`tests/test_mem0_qdrant_integration.py:16,25`) for any test that needs a live
Qdrant.

## §6 — Risks & rejected alternatives

- **Uncontrolled mutation of historical records.** The mitigation is journal + event + config gate
  (§4/§5), not immutability — this repo already treats Mem0/Graphiti records as correctable
  (`delete_memory`, `update_edge`) rather than append-only, so an in-place Mem0 update is
  consistent with the existing posture, not a new risk category. What was missing was the
  *guardrail*, not the precedent for mutability; §4 supplies the guardrail this decision adds.
- **The `created_at` preservation guarantee is the entire point.** It is the specific property
  whose *absence* forced `prune_recon_cycle_summaries.py` and `sweep_orphan_flag_markers.py` to
  delete rather than retag (§2). Both arms must preserve it; the RED test list in §5(d) makes this
  non-negotiable rather than aspirational.

  > **Correction (2026-08-02, task 3175 / escalation `esc-3175-1`).** The "forced ... to delete
  > rather than retag" causal claim above does not survive verification for either script — see
  > the correction note in §2 for the full evidence. The `created_at`-preservation requirement
  > stands on §3-§5 merits; the causal justification for *why it was "the entire point"* does not
  > — see the §2 correction, not this bullet, for the mechanism.
- **INV-5 `no-lockstep-duplication` — the module-location call for `_MEM0_MANAGED_METADATA_KEYS`.**
  §3 deferred *where* the shared extraction lands; deciding it here: **`fused_memory/backends/mem0_client.py`**
  (module-level, alongside `Mem0Backend`), **not** `fused_memory/maintenance/rehome_scope_tag.py`.
  Reasoning: `rehome_scope_tag.py`'s docstring (`:11-14`) scopes it explicitly to CGL-eta
  rehome-specific content-tagging (`apply_scope_tag`/`scope_tag_for`/`CGL_ETA_REHOME_KIND`) — a
  different concern from the generic "what does mem0 do to a payload on update" knowledge
  `_MEM0_MANAGED_METADATA_KEYS` represents. `mem0_client.py` is where that knowledge already lives
  today (`Mem0Backend.update`'s own docstring, `:245-256`, states the exact constraint this
  constant encodes) and where the new metadata-only-arm backend method(s) (§5(b)) will live too —
  so the constant, the docstring-documented constraint it encodes, and the two backend methods
  that depend on it are colocated in one file, with a pinning test
  (`tests/test_mem0_client.py`) asserting the frozenset's membership matches `_update_memory`'s
  actual preserved-key set. `scripts/tag_cgl_eta_rehome_scope.py` imports the constant from there
  instead of defining it; `rehome_scope_tag.py` is untouched, since its own concern (content-tag
  prefixing) is orthogonal.
- **INV-5, second obligation — do not duplicate `update_task`'s metadata-merge footgun one layer
  up.** Task 1827's whole-blob `metadata_mode='replace'` destroyed sibling keys on tasks; task 395
  resolved the general question of key-deletion/replace semantics for `update_task`
  (`backends/sqlite_task_backend.py:_resolve_metadata_mode`, `:3204-3254`) by requiring an
  *explicit* `metadata_mode` co-signal rather than an implicit/ambiguous one (the `append=False`
  rejection, `:3243-3254`, is exactly this principle). `update_memory`'s `metadata_mode` reuses
  that **naming** and that **explicit-intent philosophy** deliberately — same operator-facing
  vocabulary (`'merge'`/`'replace'`), same "reject ambiguity, don't guess" posture — but does
  **not** share a code path with `_resolve_metadata_mode`, and justifiably so: tasks store metadata
  as a flat JSON column resolved in Python; Mem0 records store it as a Qdrant point payload
  resolved via `set_payload`/`overwrite_payload`. A shared helper across two different backends
  with different native partial-update primitives would itself be a false abstraction — INV-5
  targets duplicated *logic that must agree byte-for-byte*, not duplicated *vocabulary*, and the
  vocabulary is what's shared here.
- **The near-duplicate guard's existing hints become actionable, not yet verified correct.** Once
  `update_memory` lands, `server/near_duplicate_guard.py`'s "search first and update or skip" (`:40`)
  and "update/consolidate" (`:52`) hints stop instructing agents to do the impossible (§2 point 1).
  The follow-up must re-read both hint strings against the shipped tool's actual name/arguments and
  confirm they still read correctly (e.g. that "update" unambiguously means `update_memory` and not
  some other tool) rather than leaving them newly-actionable but subtly wrong.

## §7 — Hand-off

**No new follow-up task is filed by this decision.** Task **3088** ("Implement the in-place Mem0
update tool...", `status=pending`, `priority=high`) already exists and already declares
`dependencies: [3055]` — confirmed via `get_task(3088)` and `get_task(3055)` against the canonical
store (`/home/leo/src/dark-factory/.taskmaster/tasks/tasks.db`) at hand-off time. Filing a second
implementation task would create a duplicate for the curator to reconcile against an
already-correct dependency edge.

### Scope-resolution mapping

Every scope bullet in task 3088's description is answered by a specific section of this document,
so its implementer needs no further design decisions:

| 3088 scope bullet | Resolved by |
|---|---|
| "An MCP tool that patches metadata and/or amends content of an existing Mem0 record IN PLACE, preserving the Qdrant point id" | §3 (full tool contract: signature, arms, `store` handling, point-id preservation in the result envelope) |
| "Decide and implement re-embedding policy: a content amend must re-embed; a metadata-only patch must NOT" | §5(b) (the write-path fork, verified against mem0's actual `_update_memory`/Qdrant source) |
| "Merge semantics for metadata must be non-destructive shallow-merge by default... Provide an explicit replace mode and key-deletion" | §3 (`metadata_mode='merge'`\|`'replace'`, `metadata_delete_keys`) |
| "Authorization model per 3055: at minimum usable by recon Stage 1 (memory_consolidator)" | §4 (`Mem0UpdateConfig`, both allowlists seeded with `'recon-stage-'`) |
| "Writes should be journalled/attributable (agent_id + a reason)" | §4 (mandatory `reason` for content-amend, optional for metadata-only) + §5(a) (unconditional journal row for both arms) |

**No 3088 scope bullet is left unresolved by this document.** (Two design details that emerged
only while writing this doc and are *not* literally present in 3088's text — the combined
content+metadata-patch call in one write, §5(b), and the INV-4 storm counter, §4 — are additions
this decision makes to satisfy the design invariants gating `/review` phase 2, not gaps in 3088's
original ask.)

### Proposed `metadata.delivered_checks` (INV-1 — for the implementer to apply to task 3088)

These checks belong on task **3088** itself (the delivering task), per `DeliveredCheckMeta`'s
gating rule: checks live on the task that claims to deliver the capability, and gate *its*
dependents — so a future task that depends on 3088 is protected from trusting a `done` status
that doesn't correspond to a real capability on `main`, exactly the failure category INV-1
exists to close off. Shape follows `docs/task-authoring.md:288-308` (`grep` preferred over
`script`); this is proposed text, not applied from here — the implementer or orchestrator adds it
to 3088's `metadata.delivered_checks` when 3088 is picked up:

```json
[
  {
    "name": "update_memory_mcp_tool_exists",
    "kind": "grep",
    "pattern": "async def update_memory\\(",
    "expect": "present",
    "paths": ["fused-memory/src/fused_memory/server/tools.py"]
  },
  {
    "name": "update_memory_service_method_exists",
    "kind": "grep",
    "pattern": "async def update_memory\\(",
    "expect": "present",
    "paths": ["fused-memory/src/fused_memory/services/memory_service.py"]
  },
  {
    "name": "mem0_update_authz_config_exists",
    "kind": "grep",
    "pattern": "class Mem0UpdateConfig",
    "expect": "present",
    "paths": ["fused-memory/src/fused_memory/config/schema.py"]
  }
]
```

All three are cheap, exact structural greps against `main` (no working-checkout dependency), and
together assert the headline contract this decision makes machine-checkable: a real MCP tool, a
real service method, and a real config-enforced authorization gate — not merely a task record
marked `done`.

### What would falsify this decision

- **The storm-counter scoping (content-amend only, §4) is wrong** if recon Stage 1's legitimate
  per-cycle tagging volume routinely trips a counter that *does* include metadata-only calls, or if
  a legitimate bulk content-correction cycle routinely trips the content-amend counter at its
  default threshold — either would mean the threshold (or the arm split itself) needs
  recalibration, not that the mechanism is wrong.
- **The authorization model is under-scoped** if a second, non-`recon-stage-*` caller needs
  content-amend or metadata-patch authority before an operator has had a chance to widen the
  relevant allowlist — e.g. an interactive curator-gate correction flow. §4 anticipated this
  (independently configurable allowlists) but shipped both identical; a real second caller
  arriving quickly would validate widening one list, not redesigning the gate.
- **The write-path fork (§5(b)) is version-pinned to the mem0/qdrant-client releases verified in
  this worktree.** If either library's internals change (e.g. `_update_memory` starts merging
  metadata instead of replacing it, or `vector_store.update` stops always supplying a vector), the
  "metadata-only must route around `Memory.update`" mechanical justification must be re-verified
  against the new source before the follow-up ships against it.
- **"Consolidation-closure" turns out to need more than a single-key `metadata.topic` stamp** (e.g.
  a structured/versioned tagging scheme) once it is ever formally specified (§2's scoping caveat) —
  the shallow-merge contract in §3 already supports multi-key patches without change, but a
  schema-validated metadata vocabulary would need its own reserved-key list layered on top of
  `_MEM0_MANAGED_METADATA_KEYS`, not a replacement of it.

### Addendum 2026-08-12 — the under-scoping clause fired, and resolved differently than predicted

The second falsification bullet above ("the authorization model is under-scoped if a second,
non-`recon-stage-*` caller needs content-amend or metadata-patch authority… a real second caller
arriving quickly would validate widening one list, not redesigning the gate") **fired** within two
weeks of shipping: the interactive memory-consolidation sitting (task 3524 / esc-3524-1, the exact
"interactive curator-gate correction flow" the bullet names). Two of its predictions held and one
did not:

- **Held:** the gate mechanism needed no redesign — the resolver, arms, and kill switch are
  untouched.
- **Held:** the new caller arrived under a dedicated narrow prefix (`curator-`), not a broad one.
- **Did not hold:** the resolution widened **both** lists, not one. Ruling (b) on esc-3524-1
  (2026-08-11) granted `curator-` content-amend AND metadata-patch, because gate 3200's
  retain-and-tag write shape stamps retained peers via metadata-only patches — content-amend alone
  would be the destructive half without the preserving half. "Widen the metadata bar alone"
  remains the supported path for tagging-only flows; consolidation is not one.

A second ruling (2026-08-12) then promoted the grant from a per-machine `config.yaml` override to
the **schema default** (`['recon-stage-', 'curator-']` on both arms), after the override tripped
the `test_recon_amend_tool_advertisement.py` premise tests: the skill that performs the sitting
(`skills/curate-fused-memories`) does not work without the grant, so it belongs to every
deployment, and `config.yaml`'s `mem0_update:` block returns to fully-commented. §4's code block
above shows the original single-entry defaults; read it as the design as first shipped, not the
current values.
