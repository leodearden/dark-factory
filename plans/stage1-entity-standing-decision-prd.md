# PRD: Entity-scoped standing decisions for recon flag suppression

**Status:** active — authored 2026-07-22 (session prd-df-2867). Resolves esc-2867-1 / task 2867.
**Research basis:** `plans/stage1-entity-standing-decision-research-2026-07-22.md` (all file:line
refs verified against main 2026-07-22; six open questions resolved with Leo this session).
**Approach:** B+H (contract + boundary-test sketch) per G5 — load-bearing suppression seam,
~8 mechanisms, and the 1185-1190 precedent (the *simpler* task-scoped contract needed six
hardening tasks).

## Goal

Stop the recon system re-paying a full Stage-2 investigation (and risking duplicate curator
tasks) every cycle for entity-scoped complaints already adjudicated by a standing decision —
"entity X is too big / topic-conflated" — while preserving the task-1966 visibility guarantee
(a standing decision must never hide a genuinely new finding).

Operator-observable when landed:
- A flag re-deriving an adjudicated size/conflation complaint is dropped in Stage 1's
  post-processor chain; the cycle summary stat `entity_standing_decision_suppressed` counts it;
  recurrence history is preserved (excluded from marker acknowledgment, not erased).
- A recon_report finding citing a standing-decided entity is **annotated, never dropped**:
  the persisted finding carries `standing_decision_id`, visible verbatim in Stage 2's payload
  rendering; Stage 2 skips re-investigation absent a new concrete fact.
- Reify's 'orchestrator' entity (the motivating incident, finding a43f43c9 → task 2867) is
  covered from day one via backfill of mem0 record b0057f3d.

## Background

The existing suppression system keys on `(task_id, flag_type)`: advisory prompt layer
(`prompts/stage1.py:485-546`) + authoritative `filter_suppressed()` gate
(`reconciliation/flag_dedup.py:344`, first step of `dedup_flags()`), reading SQLite
`recon_ledger` rows (`record_kind='stage1_flag_suppression'`). Three structural gaps
(research §Gap analysis):

1. A flag with no task anchor always passes the gate (`flag_dedup.py:521-524`).
2. The recon_report findings channel **deliberately bypasses** `filter_suppressed`
   (`task_knowledge_sync.py:2715-2718`, task-1966 visibility guarantee) — the motivating
   incident travelled this channel, so the literal esc-2867-1 ask (extend the gate) would
   not have prevented it.
3. The standing-decision records observed in the wild (`recurring_flag_standing_decision`,
   `stage1_finding_correction`) are unratified LLM conventions: zero code reads them; no
   expiry, no snapshot, no authorization.

Bounding lessons: task 1966 (blanket suppression hid a legitimate new flag_type — never
blanket-suppress an entity), task 2503 (exact/family flag_type matching evaded by rewording),
tasks 1185-1190 (producer→reader contracts across serialization boundaries need explicit
hardening).

## Resolved design decisions (this session, with Leo)

| # | Decision | Choice |
|---|---|---|
| 1 | Authorization | **Evidence-gated Stage-2 autonomy**: the writer tool accepts a Stage-2 write only when the cited evidence satisfies ≥1 *locally-resolvable* human-touched artifact (arm 1) OR ≥3 independent `investigation_outcome` records on the same entity (arm 2). Enforced **server-side** in the tool handler, not prompts. Foreign ids (e.g. orchestrator escalation ids) are recorded verbatim as context but never counted. Operator/interactive writes via the helper bypass the gate with an explicit `authorized_by` override. |
| 2 | Grounds vocabulary | **Closed enum**, initially one value: `structural_size_conflation`. Single shared constant consumed by writer, filter, and prompt renderer. Complaints citing a specific edge uuid are by definition outside it (the escape hatch b0057f3d's prose already specifies). |
| 3 | Channel B semantics | **Annotate-only, never drop.** Preserves 1966 outright. Cost-per-recurrence collapses to a skip via the Stage-2 prompt rule. |
| 4 | Staleness defaults | TTL **90d** (`expires_at` never None for this kind — unlike immortal task-scoped rows); growth tolerance **+25% or +15 edges**, whichever trips first. Both operator-overridable per record. |
| 5 | Backfill | **b0057f3d only** (reify 'orchestrator' entity). The two `stage1_finding_correction` records are finding-corrections, not standing decisions — they stay evidence-only. |
| 6 | Ad-hoc mem0 kinds | **Demoted to evidence-only.** The ledger kind is the sole machine-consulted form; the new prompt section states this explicitly. |
| 7 | Writer surface | New stage-callable `@mcp.tool()` on the recon-report server (`server/recon_report.py:2099` factory, seam at :2107-2280) — the only write path Stage 2 has. Added to `STAGE1_DISALLOWED` and `STAGE3_DISALLOWED` (`cli_stage_runner.py:65-72`): verified that recon-report tools are otherwise visible to all three stages; only Stage 2 may call it. |
| 8 | Arm-2 substrate (G3 catch) | "Not-actionable investigation record" has **no machine-filterable convention today** (live examples carry only `category: observations_and_summaries`). This PRD adds the structured **`investigation_outcome`** mem0 kind (metadata: `kind`, `entity_uuid`, `actionable: false`, run id), written by Stage 2 on not-actionable conclusions going forward. Arm 2 counts these via `get_memories_by_metadata` (exact-match AND — works today, `mem0_client.py:341-424`). Day-one emptiness is acceptable: arm 1 + backfill cover early cases. |
| 9 | Sweep placement | **Stage-2 tail**, alongside the existing unconditional 14-day marker sweeps (`TaskKnowledgeSync.run()` tail, `task_knowledge_sync.py:2252-2279` area). Stage 3 is read-only by design (`STAGE3_DISALLOWED`, `cli_stage_runner.py:67`) and cannot flip ledger state. Consequence: freshness lags **one cycle** behind Stage-1's filter — accepted (growth is organic/slow; +25% tolerance; TTL backstop) and documented in the contract. No new Graphiti dependency in Stage 1 or `flag_dedup`. |
| 10 | Hook A match bias | **Under-suppression over over-suppression** (1966's direction). Strong path: LLM-stamped `entity_uuid` + `grounds` flag fields matching an active row. Fallback (stamps omitted): UUID-regex scan finds the row's entity uuid in the flag text AND **no other UUID is present** (any second UUID ≈ edge/new-fact citation → never suppress) AND the flag_type matches a small token-family list bound to the grounds value. A fallback miss costs one cycle of noise, never a hidden finding. |

## Sketch of approach

One authoritative record — `entity_standing_decision` in the SQLite recon_ledger — consulted
at two hook points and bounded by three staleness mechanisms:

```
                      write (Stage 2 via evidence-gated tool | operator via helper)
                                        │
                 recon_ledger row: entity_uuid, grounds, expires_at(90d),
                 edge_count_at_decision, evidence refs, state
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
   Hook A (drop)                   Hook B (annotate)               Staleness
   Stage-1 post-processor      cite_entity/add_finding:        TTL gc (existing)
   chain sibling of            tool response carries the       + Stage-2 tail growth
   filter_suppressed;          decision; persisted finding     sweep (+25%/+15 →
   suppression semantics,      gains standing_decision_id;     expired) + merge_
   stat + storm escape         Stage-2 prompt rule: no         entities hook (either
                               re-investigation absent a       uuid → expired)
                               NEW concrete fact; NEVER drops
```

Prompt layer (advisory, both stages): schema rendered from `recon_self_model` (single source,
like `render_suppression_schema_section` at `recon_self_model.py:308` — today stage1-only;
extended to stage2), plus a pre-emission advisory check via `get_memories_by_metadata` against
the mem0 mirror.

## Contract (G5-H)

### Record schema (`record_kind='entity_standing_decision'`)

| Field | Semantics |
|---|---|
| `project_id` | Standard ledger scoping (rows are per-project; backfill writes a reify row). |
| `entity_uuid` | Required. New nullable ledger column (migration; `ALTER TABLE ADD COLUMN`). |
| `grounds` | Closed enum, single shared constant: `{structural_size_conflation}`. |
| `decided_at` | Write time. |
| `expires_at` | Required (never None): default `decided_at + 90d`. Existing `gc()` machinery (`recon_ledger.py:295-363`) enforces. |
| `edge_count_at_decision` | Sampled at write time via `get_valid_edges_for_node(entity_uuid)` (`graphiti_client.py:945`) — `len()` of the result. |
| `evidence` | List of refs `{type: mem0|escalation|task, id, locally_resolved: bool}`. Foreign ids verbatim, `locally_resolved=false`. |
| `state` | `active | expired | revoked`; non-active rows carry `expiry_reason ∈ {ttl, growth, merge, operator}` (INV-2: structured fact at the transition, never re-derived from logs). |

Mem0 mirror: best-effort, same rule as suppressions — **reads never consult Mem0**
(`flag_dedup.py:1158` precedent); the mirror exists for the prompt layer's advisory
pre-emission check only.

### Authorization gate (server-side, in the tool handler)

- **Arm 1**: ≥1 cited mem0 record that is locally resolvable AND human-touched
  (interactive-agent authorship predicate — exact predicate is task-β tactical).
- **Arm 2**: ≥3 `investigation_outcome` records with matching `entity_uuid`,
  `actionable: false`, and independence (distinct run ids — exact predicate task-β tactical),
  counted via `get_memories_by_metadata`.
- Failure → structured rejection naming the unmet arm and what evidence would satisfy it
  (ValidationError+hint house pattern; INV-1). No row is written.
- The helper (`write_entity_standing_decision()`, mirroring `write_suppression_record` at
  `flag_dedup.py:1086-1278`) accepts `authorized_by=<operator-id>` to bypass the gate for
  operator/interactive/backfill writes; the stage tool never passes it.

### Hook A match semantics (Stage-1 post-processor chain)

Sibling filter next to `filter_suppressed` in the `memory_consolidator.py:232-328` chain
(precedents: `filter_terminal_metadata_flags`, `filter_already_tracked_systemic_patterns`).
Pure-SQLite/sync — no Graphiti calls (decision 9).

Suppress a flag iff an active row exists for the entity AND either:
1. **Strong**: flag carries `entity_uuid` == row's AND `grounds` == row's (both LLM-stamped
   per the rendered schema); OR
2. **Fallback**: UUID-regex scan of the flag text finds the row's `entity_uuid`, finds **no
   other UUID**, and the flag_type matches the token-family list bound to the row's grounds
   (family list lives beside the enum — single source, INV-5).

Suppression semantics identical to `filter_suppressed`: excluded from marker acknowledgment,
recurrence history preserved, per-cycle stat `entity_standing_decision_suppressed`.
Ledger read failure → **fail-open** (no suppression this cycle).

### Hook B semantics (recon_report channel)

- `cite_entity` / `add_finding` (state methods `recon_report.py:922,:1553`): when a cited
  entity has an active row, the **tool response** carries
  `{standing_decision_id, grounds, decided_at, summary}` back to the Stage-1 LLM immediately,
  and the persisted finding dict gains `standing_decision_id`.
- The annotation flows to Stage 2 with **zero renderer change** — `_format_flagged`
  (`task_knowledge_sync.py:3238`) JSON-dumps flag dicts verbatim into the payload (verified).
- One Stage-2 prompt rule: an annotated finding needs no re-investigation and must not spawn
  a curator task **absent a NEW concrete fact** (a cited edge uuid is a new concrete fact —
  the annotation is informational; the rule, not the mechanism, handles the nuance).
- **Never drops.** Finding counts are identical with and without an active decision.

### Staleness + ordering

- **TTL**: existing gc, reason=`ttl`.
- **Growth sweep** (Stage-2 tail, with the marker sweeps): per active row, live edge count
  (`get_valid_edges_for_node`; few rows, per-row calls fine) vs `edge_count_at_decision`;
  `live > decision × 1.25 OR live ≥ decision + 15` → `expired`, reason=`growth`. Sweep
  failure → row stays active this cycle (fail-safe, TTL-bounded) + consecutive-failure
  streak counter escalating after 3 cycles (INV-4).
- **Merge invalidation**: `merge_entities` (`memory_service.py:4204-4269`) expires decisions
  on **either** uuid, reason=`merge` — the post-merge entity is a new subject; re-deriving
  the complaint once is correct. Also the first fix for the dangling-uuid hazard (research
  fact 9). `rename_entity` is uuid-stable — no action.
- **Ordering**: the sweep runs after Stage-1's filter in the same cycle (decision 9);
  freshness lags ≤1 cycle. Accepted and documented; INV-3 corroboration is the sweep itself.

### Storm escapes (INV-4)

- A single decision suppressing more than N flags in one cycle (or across a streak of
  cycles) → recon escalation for review (thresholds tactical, task γ; suggested N=5/cycle).
- Sweep-failure streak ≥3 consecutive cycles → recon escalation.
- Hook B is storm-immune by construction (never drops).

## Pre-conditions for activating

None external — all substrate verified on main 2026-07-22 (research §Verified facts + this
session's sweep: edge-count query `graphiti_client.py:945`; sweep seam
`task_knowledge_sync.py:2252-2279`; tool seam `recon_report.py:2099`; schema-render seam
`recon_self_model.py:308`; payload verbatim-render `task_knowledge_sync.py:3238`). The one
absent capability (arm-2 record convention) is queued **in-batch** as part of task ε (G3
resolution b).

## Cross-PRD relationship

No cross-PRD seams. All mechanisms and consumers are within fused-memory's reconciliation
subsystem, wired by tasks in this batch. The reify backfill is **data** (one ledger row in
the shared fused-memory store), not a code seam into the reify project.

## Decomposition plan

Greek labels; ids assigned at decompose. All `task_kind='normal'`. Leaf = no other batch task
depends on it.

| # | Task (modules) | Deps | Type | Observable signal |
|---|---|---|---|---|
| α | **Ledger substrate + lifecycle**: migration (nullable `entity_uuid` column), `entity_standing_decision` record kind, grounds enum + bound token-family list + `investigation_outcome` kind constant (all single-source shared constants, consumed by β/γ/ε), state machine incl. `expiry_reason`, 90d-TTL default wired into `gc()`. (`recon_ledger.py`, shared constants) | — | intermediate | Unlocks β/γ/δ/ε/ζ. Write→list→gc-expire round-trip via ledger API; gc on a past-`expires_at` row flips `state='expired', reason='ttl'`. |
| β | **Writer + authorization**: `write_entity_standing_decision()` helper (ledger row + mem0 mirror + decision-time fingerprint sample), recon-report `@mcp.tool()`, `STAGE1_DISALLOWED`/`STAGE3_DISALLOWED` entries, server-side evidence gate (arms 1+2), structured rejection. (`flag_dedup.py` or sibling, `recon_report.py`, `cli_stage_runner.py`) | α | intermediate | Unlocks ε/η. Tool call with insufficient evidence → structured rejection naming the unmet arm, no row; sufficient arm-2 evidence → active row with sampled `edge_count_at_decision`, listable. |
| γ | **Hook A filter + storm escape**: sibling filter in the consolidator chain, strong+fallback match per contract, suppression semantics, `entity_standing_decision_suppressed` stat, per-decision storm counter → recon escalation. (`flag_dedup.py`, `memory_consolidator.py`) | α | leaf | Cycle with active row + matching flag: stat increments, flag absent from Stage-2 payload, recurrence history preserved; flag text carrying a second UUID passes untouched; >N suppressions by one decision in a cycle files a recon escalation. |
| δ | **Hook B annotation**: active-decision lookup on `cite_entity`/`add_finding`, tool response carries decision, persisted finding gains `standing_decision_id`, Stage-2 prompt rule. Shared active-decision-by-uuid lookup helper with γ (INV-5). (`recon_report.py`, `prompts/stage2.py`) | α | leaf | `cite_entity` on a decided entity returns the decision in the tool response; the finding appears in Stage 2's rendered payload **with** `standing_decision_id`; finding count unchanged vs no-decision baseline (never-drops). |
| ε | **Prompt/self-model layer + `investigation_outcome` convention + demotion**: `render_entity_standing_decision_schema_section` in `recon_self_model` → stage1 AND stage2 prompts (stage2 newly consumes the renderer); pre-emission advisory via `get_memories_by_metadata`; structured `investigation_outcome` kind (Stage-2 prompt instruction + schema section) feeding β's arm 2; prompt text stating the ledger kind is the sole machine-consulted form (ad-hoc kinds = evidence-only). (`recon_self_model.py`, `prompts/stage1.py`, `prompts/stage2.py`) | α, β | leaf | Drift/pinning test (house 2559): rendered stage1+stage2 prompts contain the schema section generated from the live enum/schema — byte-identical to renderer output; stage2 prompt contains the annotated-finding rule and the `investigation_outcome` instruction. |
| ζ | **Freshness sweep + merge invalidation**: Stage-2 tail sweep (growth thresholds per contract, reason=`growth`), sweep-failure streak escalation, `merge_entities` hook (reason=`merge`). (`task_knowledge_sync.py`, `memory_service.py`) | α, β | leaf | Seeded row + entity grown past tolerance → row `expired/growth` at Stage-2 tail and next cycle's Hook A passes the flag; `merge_entities` on either uuid → `expired/merge`; 3 consecutive sweep failures → recon escalation. |
| η | **Backfill + E2E integration gate**: migrate b0057f3d → reify-project ledger row (fresh fingerprint sampled at migration, 90d TTL, evidence refs verbatim incl. mem0 originals + escalation id; originals stamped evidence-only); end-to-end test: synthetic flag naming the entity uuid → suppressed at Hook A; synthetic finding citing it → annotated at Hook B; corrections **not** migrated. (migration script, integration tests) | β, γ, δ, ζ | leaf (integration gate) | Reify 'orchestrator' entity has an active listable row; one E2E run demonstrates suppression stat + annotation + never-drop in the same cycle. |

### Boundary-test sketch (G5-H; η names this as its signal frame)

| # | Scenario | Pre | Post |
|---|---|---|---|
| 1 | Stage-2 write, insufficient evidence | no qualifying records | structured rejection naming unmet arm; no row |
| 2 | Stage-2 write, 3 `investigation_outcome` records | records with matching `entity_uuid`, distinct runs | active row; `edge_count_at_decision` sampled |
| 3 | Hook A strong match | active row; flag stamped uuid+grounds | flag suppressed; stat +1; recurrence preserved |
| 4 | Hook A second-UUID escape | active row; flag text has entity uuid + edge uuid | flag passes untouched |
| 5 | Hook A expired row | row `expired` | flag passes |
| 6 | Hook B annotate | active row; `cite_entity` on uuid | response carries decision; finding persisted with `standing_decision_id`; renders in Stage-2 payload |
| 7 | Hook B never-drops | same cycle ± active row | finding count identical |
| 8 | Growth expiry | live edges > tolerance | row `expired/growth` at Stage-2 tail; next-cycle Hook A passes |
| 9 | Sweep fail-safe | Graphiti error ×3 cycles | row stays active; escalation on 3rd |
| 10 | TTL expiry | `expires_at` past | gc flips `expired/ttl` |
| 11 | Merge invalidation | decision on uuid A; `merge_entities(A,B)` | row `expired/merge` |
| 12 | Storm escape | one decision suppresses >N flags in a cycle | recon escalation filed |
| 13 | Backfill | b0057f3d migrated | active reify row; corrections absent from ledger |
| 14 | Tool visibility | Stage-1/Stage-3 runner config | writer tool in both disallow lists (config assertion) |

## Capability bindings draft (for the decompose-time manifest)

Verified producers on main (grep-anchored evidence for the sidecar): ledger + gc
`recon_ledger.py:227-260,:295-363`; consolidator chain seam `memory_consolidator.py:232-328`;
`filter_suppressed` `flag_dedup.py:344`; helper template `flag_dedup.py:1086-1278`;
`cite_entity`/`add_finding` `recon_report.py:922,:1553`, tool factory `:2099`; disallow lists
`cli_stage_runner.py:65-72`; renderer `recon_self_model.py:308` (stage1.py:496 consumption);
verbatim payload render `task_knowledge_sync.py:3238`; sweep seam
`task_knowledge_sync.py:2252-2279`; edge fetch `graphiti_client.py:945`; merge site
`memory_service.py:4204-4269`; metadata query `mem0_client.py:341-424`. In-batch producers:
grounds enum/family list + record kind (α), writer tool (β), `investigation_outcome`
convention (ε — **absent on main today**, decision 8). No numeric-floor or grammar-fixture
class capabilities in this batch; rejection-class signals (rows 1, 4, 7, 14) bind to their
in-batch producer tasks with the test observing the rejection fire.

## G7 walk (advisory, author-time)

- `contracts-machine-checked` — authorization + grounds validated at the tool boundary
  (structured rejection); tool visibility via disallow lists; arm-2 evidence is a structured
  kind, not prose. ✓
- `structured-facts-at-failure` — `expiry_reason` on every state flip; suppressions carry
  decision id + stat; rejections name the unmet arm. ✓
- `corroborate-before-acting` — the growth sweep *is* the corroboration of decision-time
  snapshot vs live graph; ≤1-cycle lag documented, TTL-bounded. ✓
- `storm-escape-required` — Hook A suppression storm counter; sweep-failure streak; Hook B
  storm-immune (never drops). ✓
- `no-lockstep-duplication` — grounds enum + family list single-source; schema rendered from
  `recon_self_model` into both prompts; shared active-decision lookup helper across hooks
  A/B. ✓

## Out of scope

- Additional grounds values beyond `structural_size_conflation` (add via the enum when a
  second complaint class earns a standing decision).
- Any drop behaviour on the recon_report channel (decision 3: annotate-only, permanently —
  revisiting requires a new PRD against the 1966 analysis).
- Retroactive coercion of `stage1_finding_correction` records (decision 5).
- Payload indexes for `get_memories_by_metadata` (full scan fine at current sizes —
  research fact 8).
- A generic per-entity suppression framework beyond grounds-scoped rows (1966 forbids
  blanket entity suppression).

## Open questions (tactical — decide at the named task)

1. **Arm-1 human-authorship predicate** — exact `agent_id`/metadata test for
   "interactively-curated". Suggested: agent_id allowlist pattern (`claude-interactive*`,
   operator ids). Decide in β.
2. **Arm-2 independence predicate** — distinct `run_id`s vs distinct calendar days.
   Suggested: distinct run ids. Decide in β.
3. **Ledger column-vs-payload placement** for `grounds` / fingerprint / evidence (queried
   fields → columns; blob fields → payload JSON). Suggested: `entity_uuid` + `grounds` as
   columns, rest in payload. Decide in α.
4. **Storm thresholds** — N per cycle and streak length. Suggested: 5/cycle, streak 3.
   Decide in γ (suppression) / ζ (sweep).
5. **Grounds token-family seed list** for the fallback match. Decide in γ.
6. **Evidence-only stamping shape** on the migrated mem0 originals (`x_`-namespace metadata
   per the Tier-C convention). Decide in η.
