# PRD: Memory metadata vocabulary + Mem0 corpus shape

**Project:** dark-factory (fused-memory + orchestrator). **Status:** active, 2026-07-29. **Approach:** B+H (contracts + two-way boundary tests).
**Origin:** spawn brief `df-memory-vocabulary-prd` (Leo, attended session 2026-07-29). This PRD owns the two decisions the 2026-07-29 deferral wave (3111/3112/3129/3133/3136) is blocked on: the **metadata vocabulary** (`topic` / `canonical` / `kind` / `parent_id` / `supersedes`) and the **Mem0 corpus-shape decision** (Option B vs C). All substrate below re-verified against main `d42f510669` on 2026-07-29 by a three-agent verification pass; deltas from the brief are recorded inline.
**Siblings:** `docs/prds/memory-write-path-convergence.md` (17 leaves, in flight — this PRD amends its §1/D4/§8 in a companion commit); `plans/memory-subsystem-eval-design.md` (E2 is implemented here as leaf ζ; the rest of the eval program is out of scope); `plans/mem0-in-place-update-decision.md` (3055, DECIDED — its reserved-key list is the layer *below* this vocabulary).

## 1. Goal (G1 consumer + user-observable surface)

Memory metadata stops being a pile of per-writer inventions and becomes a **validated, retrieval-load-bearing vocabulary**, and the Mem0 corpus converges to a shape in which **consolidation can never again make a topic's authoritative entry its least retrievable member**. Observable surfaces:

- **Writers** (every `add_memory`/`add_system_record` caller — MCP tools, recon stages, direct service callers): a malformed vocabulary key (`kind` outside the registry, scalar/short-hex `supersedes` member, dead `parent_id`, second `canonical` on a topic) is **rejected with a structured error naming the rule**; an unknown non-`x_` key emits a census warning line. Today `add_memory` persists any dict verbatim (`services/memory_service.py:2192-2287` — no allowlist, no schema).
- **Readers**: a topic query surfaces the topic's canonical (leaf-3111 anchoring, un-deferred by this PRD's gate); a child-only match resolves `parent_id` **upward** to its grouped document instead of returning nothing; `get_memories_by_metadata({'topic': T})` deterministically enumerates a consolidated topic — the closure predicate 3112 needs.
- **Operators**: the E2 storage-shape bake-off report (leaf ζ) — a per-arm decision table (claim-recall@k, discoverability, tokens/query, guard adequacy) that ratifies or refutes the provisional shape choice at a born-at-L2 gate (leaf η), converting the B-vs-C argument into a measurement.
- **The five deferred tasks** un-defer with their `x_deferred_reason` contracts discharged: each names this PRD as the decision it was waiting for.

Ultimate consumer: a fleet agent at its moment of need — concretely the briefing assembler's four hardcoded per-prompt queries (`orchestrator/src/orchestrator/agents/briefing.py:978-1001`, `limit=5`, injected into every role prompt from 14 call sites), the highest-volume retrieval surface in the system.

## 2. Scope (first-class decision)

**All of Mem0; Graphiti excluded by decision.** (Ratified by Leo 2026-07-29.)

- The vocabulary + write-time validation cover **all three Mem0 categories** (`procedural_knowledge`, `preferences_and_norms`, `observations_and_summaries`) at the **service seam** — `MemoryService.add_memory` + `add_system_record` — not `server/tools.py`. Reason: `add_system_record` is a second unguarded metadata write path, and `summary_pool.py:362` / `standing_decision_writer.py:425` call the service directly, bypassing the MCP layer; a tools-layer validator would leak. This makes `observations_and_summaries` (16k of 24k entries, previously zero tooling, where every session recap lands) covered by construction.
- Existing detectors keep their own scopes but are amended toward cross-category (κ hardening; β/3127 is already cross-category by contract C1). The eval-doc E6 embedding-ANN upgrade that would make the lexical audit script feasible at 24k-entry scale is **out of scope** (pointer: eval-program PRD).
- **Graphiti is excluded by decision, not omission.** Corrected rationale (brief's "no dedup machinery whatsoever" is refuted): Graphiti has substantial *post-hoc* dedup (`graphiti_client.py:1342` `dedup_valid_edges_for_node`, `:1549` `_repair_duplicate_edge_uuids`, `memory_service.py:1014` `_dedup_episode_edges`, three reconciliation sweeps). What it lacks is a *write-time* guard — a real but unmeasured gap with no incident behind it, in a store whose edge/episode model doesn't carry the Qdrant payload namespace these five keys live in. A Graphiti write-time guard is a future PRD if pain is ever measured.

## 3. The corpus-shape decision (G5 — provisional C, ratified by measurement)

**Pathology:** consolidating N near-duplicates into one long canonical makes the canonical the least retrievable member of its own cluster (centroid embedding). Measured twice: 168c3a6b (rank 10/10 at 0.7155, then deleted) and re-verified 2026-07-29 on its replacement bbc063a7 (~9k chars, absent from limit=10 while ten short siblings ranked 0.66–0.76, including one it marks superseded). Post-consolidation write rate measurably doubled. Property of entry length, not of specific entries.

**Decision: provisional Option C**, ratified or refuted by leaf ζ (the eval-design E2 bake-off) at gate η:

- **Default output of consolidation** = N **short single-claim peer entries** sharing `metadata.topic`, exactly one carrying `canonical: true`, each independently retrievable (3112 defect 1); the canonical is **short** (an index/summary claim, not a concatenation).
- **Topic-anchored recall** (3111) pins the topic's canonical into results when any same-topic entry matches, at the `MemoryService.search` seam (`memory_service.py:2720`) so all 12 call sites — including the near-dup guard's internal search (`server/tools.py:1233`) and β/3127's candidate retrieval — get it for free.
- **Child records** (3129) remain, under both options, as the representation for **triage attach outcomes only** (`kind ∈ {amendment, sighting}`, `parent_id` = canonical) — not as the output of consolidation.
- Why provisional-C rather than argument-final: the recorded evidence (inversion measured twice; the 3111-metadata counterexample where B would suppress the four short siblings that carried the *correct* rule; B addressing 0% of the existing corpus since `parent_id` is stamped only by β's write-time triage behind a default-off flag; θ-under-B regenerating the pathology every round uninstrumented; subtractive-suppression composing unsafely with `mem0_dedup`'s post-filter vs additive-pin composing safely) all favor C — but the peer shape's editorial cost and its claim-recall/token profile are unmeasured, and E2 was designed precisely to measure them. The gate defaults to ratifying C; its escalation text carries the exact re-amendment list if the decision table says otherwise.
- **Sequencing consequence:** 3129, once amended (D5–D7 below), is **shape-neutral** and un-defers immediately — it no longer commits a consolidation default, and it unblocks the write-path chain β/3127→γ/3128. Only 3111, 3112, 3133, 3136 wait on gate η.

## 4. Contracts (H)

### V1 — Vocabulary contract

> **Every Mem0-persisted metadata dict is validated at the service seam against one registry; vocabulary keys are shape-checked strictly, unknown non-`x_` keys warn to a census line, and validation failure is a structured rejection — never a silent drop or a silent accept.**

Reserved vocabulary (layered **on top of** 3055's `_MEM0_MANAGED_METADATA_KEYS` — the 9 mem0-owned keys stay the bottom layer, single home per 3055 §6; and alongside the server-stamped keys `category`, `recon_pool`, `run_id`, `planned`):

| Key | Shape | Semantics | Primary readers |
|---|---|---|---|
| `topic` | kebab-case slug (shared regex constant with `ProceduralTopicCluster.topic_id` — **one namespace**, see D4) | topic membership; the retrieval-anchoring + deterministic-closure key | 3111 anchoring; 3112 closure scroll; guard clusters; E1 registry |
| `canonical` | bool | ≤1 `true` per (project, topic) — enforced at write via live `count_memories_by_metadata` re-check (INV-3) | 3111 anchoring; `pick_survivor` (`audit_duplicate_memories.py:197`) |
| `kind` | member of the closed **kind registry** (code constant; census-grandfathered live set + `amendment`, `sighting`) | record-type axis. Distinct from `source` (writer-provenance axis) — both defined here; the `source`-set-but-`kind`-missing drift (`tools.py:1595-1597`) gets a lint | equality filters at `task_knowledge_sync.py:1740,1832,1871`, `tools.py:1164`, `flag_dedup`, `scope_freshness`, standing-decision writers; grouped read; prompt-embedded filter examples (E10/E11 — rendered/pinned per INV-5) |
| `parent_id` | full UUID; **must resolve to a live same-project Mem0 entry at write** | child→canonical attachment (triage attach outcomes only) | grouped read (3129); triage attach (3127); orphan lint |
| `supersedes` | **list of full UUIDs** (canonical shape); legacy scalar tolerated on read via one shared `normalize_supersedes()` helper | supersession pointers | `targeted.py:1464` (today: bare truthiness on a scalar); 3112's closure predicate (today: would iterate a 36-char string — fixed by the helper); staleness sweeps |

- Unknown non-`x_` top-level keys: **warn, don't reject** (task-metadata census precedent) — census line `memory_metadata.schema_warning project_id=<p> agent_id=<a> code=<class> key=<k>` (grep-anchored, never renamed). `x_`-prefixed keys pass silently. Warn **storms** carry a rate-threshold escalation (INV-4): a drifting writer flooding unknown keys is heard, not logged into oblivion.
- Rejection error type is contract-fixed: `MemoryMetadataValidationError`, with a `hint` naming the violated rule and the registry location. Canonical-uniqueness rejection is contract-fixed: `CanonicalUniquenessViolation`, naming the existing canonical's id.
- Boundary tests both sides: writer side (each malformed shape rejects with the named error; valid writes round-trip unchanged; unknown key warns + storms escalate) and reader side (registry importable; prompt-pinning drift test fails when `_MEMORY_INSTRUCTIONS`/recon prompt examples drift from the registry).

### V2 — Record-ontology + grouped-read contract

> **Peers are the shape of knowledge; children are the shape of activity. The grouping key is strictly `metadata.parent_id`; entries sharing only `metadata.topic` are never grouped.**

- **Peer** = ordinary entry with `topic` set (single claim, independently retrievable). **Canonical** = the one peer with `canonical: true`. **Child** = `parent_id` + `kind ∈ {amendment, sighting}` — produced by triage attach (C1), never by consolidation under C.
- Grouped read (3129): children collapse into the canonical's grouped document (amendment digests + sighting count). **Upward resolution is mandatory**: a child-only match fetches its parent and returns the grouped document — a child's content is never unreachable. (Under suppress-only, a child-only match returns nothing — the retrieval-regression branch of Option B; closed here by decision.)
- **`contested` children are never suppressed** until adjudicated — a correction must not be demoted to a truncated digest under the entry it contests (the esc-5712 five-week-wrong-appendix shape).
- The suppression filter lives **only** in the grouped-read path (`server/grouped_read.py`) — it must not leak into `MemoryService.search` generally, or it breaks `mem0_dedup.find_prior_memories`' per-record metadata post-filter (`mem0_dedup.py:85-91`), hides duplicates from the detector, and hides candidates from the write guard.

### V3 — Lifecycle contract

> **No operation may silently orphan a child or dangle a pointer it could have seen.**

- `parent_id` liveness validated at write (V1). `delete_memory` on an entry with children (live `count_memories_by_metadata({'parent_id': id})` re-check, INV-3) **refuses** with contract-fixed `ParentHasChildrenError` listing child ids, unless `cascade=true` (children deleted too, journalled). Re-parenting (`reparent_to`) is delivered by 3133's amended consolidate op (it requires 3088's `update_memory`), not by leaf δ.
- `consolidate_memories` (3133, amended): folded entries' children are re-parented to the new canonical atomically; under C its default arm **retains** peers (topic-stamps them via `update_memory`, metadata-only patch, no re-embed) instead of deleting them; `supersedes` on the canonical lists only genuinely absorbed (deleted) entries.
- `supersedes` gets **no** write-time liveness check (targets legitimately die later); dangling-pointer census belongs to the κ report / future E4 sweep.

## 5. Resolved design decisions

- **D1 — canonical marker is `canonical: true`**, not a kind value. It has the only live entries (6) and the only existing reader; role-in-topic and record-type are orthogonal axes (a canonical is also of some kind). 3111's `topic_anchored_canonical_kinds` config knob is dropped in its amendment; anchoring selects `topic=T AND canonical=true`.
- **D2 — `supersedes` is a list.** The single writer (`harness.py:1167`) migrates; readers go through `normalize_supersedes()` (accepts scalar/list/None); no corpus rewrite required for correctness (retro normalization rides θ's stamping sweep where it touches entries anyway).
- **D3 — `kind` is a closed registry; unknown other keys warn.** Strict-reject on unknown `kind` is safe because kind writers are in-repo code + prompts (enumerated by the census leaf, grandfathered); strict-reject on arbitrary keys would break unenumerated writers — census first, tiers later (task-metadata precedent).
  - *Amendment (leaf β, against leaf α's census).* **The decision stands, but its stated safety rationale was MEASURED FALSE.** Kind writers are not confined to in-repo code + prompts: 242 of the 329 live `kind` values occur exactly once, i.e. the population is agent-invented free text and open in practice. A day-one strict-reject would therefore turn every newly invented kind into a hard memory-write failure on the live fleet. Leaf β consequently ships the full strict-reject machinery (typed error, rule-naming hint, tests) behind **`memory_metadata.enforce_kind_registry`, default `false`** — held off pending §10 open question (1) — while the general shape checks sit behind `memory_metadata.enforce`, also default `false`. This mirrors the very precedent D3 cites: `TaskMetadataConfig.enforce` (`config/schema.py:331`) shipped warn-mode-first and was flipped after soak. The config leaf is named here so doc and code cannot drift.
- **D4 — one topic namespace.** `ProceduralTopicCluster.topic_id` values and `metadata.topic` values are the same namespace, same slug regex (one shared constant; the 5 seeded cluster ids already conform). ζ/3135's auto-seed must set `cluster.topic_id == canonical.metadata.topic` (amendment). The E1 topic registry derives from this namespace.
- **D5 — grouping key pinned.** Strictly `parent_id`; never topic (V2). The write-path PRD §1 sentence "one grouped document per topic" is corrected in the companion commit.
- **D6 — upward resolution mandatory; contested never suppressed** (V2).
- **D7 — orphan rules** (V3): refuse-by-default delete; cascade explicit; reparent via the consolidate op.
- **D8 — enforcement lives at the service seam**, not `server/tools.py` (three write paths converge there; see §2).
- **D9 — provisional C + measurement gate** (§3): ζ implements eval-design E2 exactly (arms: status-quo / C-peers / B-grouped / each ± 3111-pin; the "hybrid" of eval-doc open question 5 *is* C); η is a pure deterministic gate (3169 pattern) defaulting to ratify-C.
- **D10 — ζ also delivers the audit-recall measurement** (3136's deferral item 3): run `audit_duplicate_memories.py` against α/3130's labeled fixture and report recall on the paraphrase class — the number that decides how much to trust the κ report.
- **D11 — retro stamping is in scope** (leaf θ), bounded to known consolidated clusters (the 6 `canonical: true` entries, curator-gate enumerated clusters incl. DF gates 2969/2973/3011/3016/3036/3063/3092 and the reify gate census on 3112, α's fixture topics). This closes the "anchoring ships against 0% of the existing corpus" gap. The write-path PRD's §7 no-retro-sweep exclusion stands *for that PRD*; this PRD owns this bounded sweep.
- **D12 — `_MEM0_MANAGED_METADATA_KEYS` extraction** (script → `backends/mem0_client.py`, per 3055's decided home) is owned by whichever of β / 3088 lands first; both texts carry the defensive wording (extract if absent, else import). Never two copies (INV-5).
  - *β amendment (2026-08-01), measured cost of that home:* `backends/mem0_client.py` imports `mem0.AsyncMemory` and `config.schema.FusedMemoryConfig` at module scope, so importing `fused_memory/memory_metadata.py` — which binds the set from there rather than copying it — transitively pulls the mem0 SDK and the config model. The registry is therefore **pure** (no mutable state, no queue writer, no optional `escalation` dependency) but **not import-cheap**, and an out-of-process consumer that cannot install `mem0` cannot import it. β deliberately did **not** re-home the set into a leaf module (e.g. `backends/mem0_metadata_keys.py`, re-exported from `mem0_client` to keep every import site and object identity intact): that would overturn 3055 §6's decided home while 3088 is still pending against it. Whoever of 3088 / a follow-up revisits D12 should decide it explicitly; the practical impact today is nil, since `mem0` is a hard dependency of the `fused-memory` package.

## 6. Pre-conditions / substrate (G3 — verified 2026-07-29 against main d42f510669)

| Assumed capability | Evidence |
|---|---|
| Service-seam write chokepoint + existing (single-kind) metadata munging | `services/memory_service.py:2166` (`add_memory`), `:2192-2207` (verbatim dict copy; `_normalize_task_id_metadata`; cycle-summary tagging), `add_system_record` shares the path |
| Arbitrary-key exact-match metadata reads (downward join, Mem0-only) | `get_memories_by_metadata`/`count_memories_by_metadata`/`get_memory_by_id` — `server/tools.py:1568,1466,1648`; Qdrant `FieldCondition` build `mem0_client.py:326-330,389-393`; empty-filter rejected |
| Retrieval seam all 12 call sites share | `MemoryService.search` `memory_service.py:2720`; sort `:2800-2804`; truncation `:2830`; Mem0 raw-cosine `:2975`; Graphiti synthetic decay `:2917`; category-only push-down `mem0_client.py:184-188` |
| Live `kind` population (grandfather set start) | **REFUTED — superseded by leaf α's measurement** (`plans/memory-metadata-census-report.json` @ `b5af3e4b03`, `coverage.complete = true`). Measured: **329 distinct `kind` values** across 2,478 kinded records; 47,150 of 49,628 records (**95.0%**) carry no `kind` at all. Top five by count: `cycle_summary` (1,323), `cgl_eta_cross_target_rehome` (253), `task_completion_note` (101), `task_completion` (68), `completion_note` (62). **242 of the 329 distinct values are singletons.** This row's original ten-name list was wrong in both directions: five of its names (`stage1_flag_marker`, `project_status_correction`, `consolidated_scope_correction`, `entity_standing_decision`, `count_snapshot_cleanup_audit`) measure **zero** live records, and it omitted ~319 kinds that do exist. Normative home for the grandfather set: **`fused_memory.memory_metadata.KIND_REGISTRY`** — consumers import it; per INV-1/INV-5 this row deliberately does not restate the value list. See §10 open questions (1) and (2). |
| `supersedes` today: scalar writer + truthiness reader | `reconciliation/harness.py:1167`; `reconciliation/targeted.py:1464` (sibling discriminator `stage2_suppress` at `:1466` shows the refined pattern) |
| `canonical` today: one reader, no writer | `scripts/audit_duplicate_memories.py:197` |
| `metadata.topic` / memory-land `parent_id`: zero code footprint | targeted greps: 0 hits in `fused-memory/src` + tests |
| Topic-cluster config + guard consumption | `config/schema.py:376-406` (4 fields, `extra='forbid'`), `:1117` config leaf (green-tier reload `config/reload.py:55`); `near_duplicate_guard.py:88-119` first-match phrase matcher |
| Seeded-ephemeral-store probe pattern for ζ | `tests/test_recon_dedup_premise.py` (per-worker scoped project_id, `real_embedder=True`); embedder handle `middleware/task_curator.py:659`; cleanup script prefix caveat |
| α calibration fixture + recall@k machinery (ζ input) | task 3130 in-progress, live claimant; 104-record fixture + `calibrate_write_triage.py` in its worktree |
| Metadata-only in-place patch (θ, 3133-C-arm) | `Mem0Backend.update` exists `mem0_client.py:238-262` (zero callers); service/MCP layer owed by 3088 (contract: `plans/mem0-in-place-update-decision.md`) — **θ deps 3088** |
| Pure-gate task shape for η | task-authoring: `task_kind=deterministic` + `always_escalates` + no `before_done`; live exemplar task 3169 |
| Writer-instruction site | `orchestrator/src/orchestrator/agents/roles.py:218` `_MEMORY_INSTRUCTIONS`, injected into 5 roles; the word "metadata" absent today |
| Prompt-embedded metadata-filter examples (ι must pin) | `recon_self_model.py:288,467,488`; `prompts/stage1.py:414-424`, `stage2.py:359,374`, `stage3.py:42,174-182` |

No novel substrate is assumed beyond what the leaves themselves produce.

## 7. Out of scope

- The wider eval program (E1 retrieval-health monitor, E3 golden set, E4 sweep-as-metric, E6 ANN upgrade, E7 telemetry/shadow-replay, E8/E9) — future eval-program PRD over `plans/memory-subsystem-eval-design.md`. ζ implements E2 only (+ D10's audit-recall measurement).
- Graphiti write-time guard / vocabulary alignment (§2 decision).
- Full-corpus retro re-categorization or unbounded stamping sweeps (θ is bounded to known clusters).
- The `memory_hints` dead channel and briefing-query redesign (eval-doc §4/§9.6 finding — measurement first).
- `update_memory` itself (3088), citation repointing (3108), XML-leak cure (3083), `reexamine_when` (3139/ν) — existing owners.

## 8. Cross-PRD / seam ownership (G4)

| Seam | Owner | This PRD's edge |
|---|---|---|
| Vocabulary registry, validation, census line, kind/source axes, topic namespace, lifecycle rules | **this PRD** (leaves α–ε) | single normative registry module; consumers import, never restate (INV-5) |
| Corpus-shape default + ratification | **this PRD** (ζ, η) | gate resolution un-blocks 3111/3112/3133/3136; re-amendment list lives in η's escalation text |
| Reserved-key bottom layer (`_MEM0_MANAGED_METADATA_KEYS`) + `update_memory` tool | 3055 (decided) / 3088 (pending) | D12 extraction ownership; θ and 3133's C-arm consume the tool |
| Retrieval anchoring | 3111 (deferred→un-deferred at η) | amendment: marker = `canonical: true`, land at service seam; deps η+θ |
| Consolidation-gate end-state + mechanical closure | 3112 (deferred→un-deferred at η) | amendment applies its 5 recorded fixes; closure predicate uses V1 topic scroll + `normalize_supersedes()`; enforcement at `set_task_status` interceptor keyed on `metadata.operational_mode=='gate'` — which 3084's auto-close then respects by construction |
| Child records + grouped read | 3129 (deferred→un-deferred **now**) | amendment: D5/D6/D7 pinned in text; deps β+δ |
| Transactional consolidate | 3133 (deferred→un-deferred at η) | amendment: C retain-and-tag arm, orphan re-parenting, deterministic-scroll acceptance (no semantic-probe premise); deps η+δ+γ+3088 |
| Dedup report scheduling | 3136 (deferred→un-deferred at η) | amendment: detector/provenance framing (enumeration inert), report hardening, topic carve-out; deps η |
| Write-path PRD §1/D4/§8 text | `docs/prds/memory-write-path-convergence.md` | companion commit in this batch's turn: grouped-read sentence corrected per D5, D4 annotated with the gate, §8 gains rows for 3111/3112/this-PRD |
| Writer briefings | orchestrator `roles.py` (ι here; 3131/ε-writepath later) | ι adds the vocabulary section now; 3131's inversion (behind 3169) edits a different sentence — sequence note in both |
| Guard topic clusters / auto-seed | 3135 (pending) | amendment: `cluster.topic_id == canonical.metadata.topic` (D4) |
| Triage candidate retrieval | 3127 (pending) | amendment note: retrieval goes through `MemoryService.search` (3111 covers it; no second topic implementation — INV-5) |

## 9. Decomposition plan (signals are the G2 gate; Greek labels this-PRD-local)

Deps: β←α; γ,δ,ε,ι←β; ζ←β (+3130 ext); η←ζ; θ←β,ε (+3088 ext). Existing-task re-wiring listed in §8 and filed as amendments, not new tasks.

- **α — corpus metadata census:** `fused-memory/scripts/census_memory_metadata.py` scrolls all three Mem0 categories in both live projects (paginated), enumerating top-level key population, `kind` values, `supersedes` shape census (scalar vs list), `topic`/`canonical`/`parent_id` occurrences; emits a JSON+markdown report artifact. *Signal:* running the script produces the report; β's registry grandfather list cites it.
- **β — vocabulary module + service-seam validation:** registry module (vocabulary keys, kind registry, slug constant, `normalize_supersedes()`, shape validators) wired into `MemoryService.add_memory` + `add_system_record`; strict-reject with `MemoryMetadataValidationError`+hint; unknown-key census line + storm-threshold escalation (INV-4); D12 defensive handling of `_MEM0_MANAGED_METADATA_KEYS`. *Signal:* malformed `supersedes` member / unknown `kind` / dead `parent_id` reject with the named error and hint; valid write round-trips; unknown non-`x_` key emits the census line; storm test escalates.
- **γ — supersedes normalization:** `harness.py:1167` writes a list; `targeted.py:1464` (and exported helper for 3112) read via `normalize_supersedes()`. *Signal:* new `project_status_correction` entries carry list-shaped `supersedes`; helper handles None/scalar/list (tests); classifier behavior unchanged on legacy scalar fixtures.
- **δ — parent lifecycle:** write-time `parent_id` liveness; `delete_memory` child-refusal (`ParentHasChildrenError`, live re-count, INV-3) with explicit `cascade` opt-in (journalled). *Signal:* dead-parent write rejects; parent-delete with 2 children refuses listing both ids; `cascade=true` deletes children with journal rows (tests).
- **ε — canonical uniqueness + topic namespace:** `canonical` bool validation; ≤1 per (project, topic) via live count re-check (`CanonicalUniquenessViolation` naming the incumbent); shared slug regex applied to both `metadata.topic` and `ProceduralTopicCluster.topic_id` (one constant). *Signal:* second same-topic `canonical: true` write rejects naming the incumbent; malformed slug rejects on both the memory and config sides (tests).
- **ζ — E2 storage-shape bake-off + audit-recall measurement:** seeded ephemeral collections (isolation pattern per §6), arms status-quo / C-peers / B-grouped / each ± topic-anchored pin, realistic distractor slab, arms authored blind to metrics; metrics per arm: claim-recall@k, canonical/topic discoverability, tokens/query, near-dup-guard candidate adequacy (pure-guard replay); plus D10's `audit_duplicate_memories.py` recall vs α/3130's labeled fixture. Committed decision-table report. *Signal:* `plans/e2-storage-shape-bakeoff-report.md` (+JSON) exists with all four metrics per arm and the audit-recall number; rerunnable script committed; ephemeral collections cleaned up (cleanup-script prefix extended).
- **η — shape ratification gate** (`task_kind=deterministic`, `always_escalates`, no `before_done`, deps ζ): born-at-L2 escalation directing the operator to ζ's decision table; default resolution ratifies C (resume → un-blocks 3111/3112/3133/3136 via dep edges); the if-B re-amendment list is spelled out in the escalation text (3111 cancel-or-rescope; 3112 defect-1 inverted, defect-2 stands; 3133 reverts to delete-arm default with mandatory upward resolution; 3136 carve-out dropped; 3129 unchanged either way). *Signal:* on ζ landing, a `milestone_gate`-class escalation exists citing the report path; resolution recorded either way.
- **θ — retro stamping sweep** (deps β, ε, 3088): bounded per D11; stamps `topic` (+`canonical` where established; normalizes `supersedes` in passing) via `update_memory` metadata-only patch (no re-embed); idempotent; per-run report. *Signal:* for each stamped topic, `get_memories_by_metadata({'topic': T})` returns the cluster including its canonical; report lists per-topic stamped counts; second run is a no-op.
- **ι — writer-instruction + prompt pinning:** `_MEMORY_INSTRUCTIONS` gains a metadata-vocabulary section (5 roles); recon prompt-embedded filter examples (E10/E11 sites) rendered from or pinned to the registry; drift test. *Signal:* rendered architect briefing names the vocabulary keys; the pinning test fails when registry and prompt text drift (demonstrated in test).

**G7 note:** every leaf walked against `docs/legibility/design-invariants.md`; no waivers required. The census-warn path carries its storm escalation (INV-4); registries/error-types are code, not prose (INV-1); uniqueness/child checks re-read live state (INV-3); rejection errors are structured (INV-2); registry, slug constant, and normalize helper are single-home with pinning tests (INV-5).

## 10. Open questions (tactical, implementation-time)

**Raised by leaf α's census (esc-3194-1); both must be decided before `memory_metadata.enforce_kind_registry` is flipped on:**

1. **Does `kind` stay a closed registry?** D3 assumes kind writers are in-repo code + prompts, but the census measures **242 of 329** live values as singletons — the population is agent-invented and open in practice. Decide one of: (a) `kind` remains a closed registry with a scheduled re-grandfathering sweep that re-seeds `KIND_REGISTRY` from a fresh census; (b) `kind` is demoted to a warn-only census axis and never rejects; (c) `kind` gains a tiered/prefixed escape hatch (e.g. an `x_`-style prefix for agent-invented kinds that is exempt from membership). Until this is decided, `enforce_kind_registry` stays `false` and the rejection path ships dormant but tested.
2. **What happens to the un-kinded majority?** **95.0%** of live records (47,150 of 49,628) carry no `kind` at all, so whatever is decided about un-kinded records dominates any migration far more than the kind vocabulary does. Decide: is `kind` ever to become *required*, and if so for which writers (all service-seam writes? only reconciliation writers? only new kinds?) — and does the existing un-kinded corpus get backfilled, grandfathered permanently, or swept by leaf θ?

**Remaining tactical questions:**

- Census pagination mechanics (scroll batch size vs raised limit) and report artifact location convention.
- Exact slug regex (charset/length cap) — must accept the 5 seeded cluster ids.
- Storm-threshold values for the unknown-key warn escalation (config, not hardcoded; calibrate from census volume).
- κ carve-out mechanics detail (cluster classified "topic cluster — expected under C" vs delete-candidates) — constrained by 3136's amended text, decided in-task.
- Blind-authoring protocol for ζ's arms (two-agent cross-check vs single-author-blind-to-metrics).
- Whether θ also seeds `ProceduralTopicCluster` entries for stamped topics or leaves that to ζ/3135's auto-seed path (lean: leave to 3135; θ stamps memory-side only).
