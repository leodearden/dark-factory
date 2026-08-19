# Research: entity-scoped standing decisions for Stage-1 flag suppression (esc-2867-1 / task 2867)

**Status: research findings for discussion — nothing implemented, escalation left pending.**
Session: research-df-2867 (2026-07-22). Sources: code map by two Explore agents + live
mem0/ledger/escalation records; file:line refs verified against current main.

## TL;DR

The (task_id, flag_type) suppression system structurally cannot suppress complaints
derived from raw entity state — they have no stable task anchor. Worse, **the motivating
incident travelled the recon_report findings channel, which deliberately bypasses
`filter_suppressed` entirely** — so the literal ask in esc-2867-1 ("extend the Stage-1
flag-suppression check") would not have stopped the incident that motivated it. The fix
needs a first-class *entity standing decision* record (SQLite recon_ledger, like
existing suppressions) consulted at **two** hook points, with staleness bounded by
TTL + entity-growth fingerprint + merge invalidation. **Recommendation: small /prd
(~6 tasks), not a single task.**

## Verified facts

1. **Suppression architecture today** — two layers: advisory prompt
   (`prompts/stage1.py:485-546`) + authoritative code gate `filter_suppressed()`
   (`flag_dedup.py:344-540`), run as the first step of `dedup_flags()`
   (`flag_dedup.py:722-726`) from `MemoryConsolidator.run()`
   (`stages/memory_consolidator.py:288-297`).
2. **The gate reads SQLite, not mem0.** `memory_service.recon_ledger.list_suppressions()`
   (`recon_ledger.py:227-260`, `WHERE record_kind='stage1_flag_suppression' AND
   state='active'`). Mem0 records are a best-effort *mirror* only ("reads never consult
   Mem0", `flag_dedup.py:1158`). Ledger row identity: `(project_id, record_kind,
   task_id, flag_type, run_id)`; an `expires_at` column already exists but suppression
   rows pin it `None` (immortal, operator-managed — `flag_dedup.py:1142-1143`).
3. **A flag with no task_id always passes** (`flag_dedup.py:521-524`). No entity scoping
   exists anywhere in the gate, the ledger schema, or the record vocabulary.
4. **The god-node/topic-conflation heuristic is not code.** No threshold, no derivation
   site — it is emergent Stage-1 LLM behavior under the general mandate
   (`prompts/stage1.py:53-59`) when it sees a large entity. It can only be gated at
   emission/persistence, not derivation.
5. **The motivating instance bypassed the gate's channel entirely.** Finding a43f43c9
   went `add_finding`+`cite_entity` → recon_report → Stage 2 poll → cross-project
   routing → task 2867. That channel deliberately does **not** pass through
   `filter_suppressed` (`task_knowledge_sync.py:2715-2718`: a suppression "can never
   hide a systemic_pattern finding") — a *deliberate* visibility guarantee from the
   task-1966 bug (blanket suppression hid a legitimate new flag_type 6+ cycles).
6. **`recurring_flag_standing_decision` / `stage1_finding_correction` are unratified LLM
   conventions.** Zero code/prompt/skill reads or writes them; they are hand-authored
   mem0 records by Stage 2's LLM via `add_memory` (Stage 1/2 are only blocked from
   task-writes + Bash/Edit/Write — `cli_stage_runner.py:36-65`). Live instances: reify
   `b0057f3d` (kind=recurring_flag_standing_decision; metadata: `entity_uuid`,
   `task_ids`; **no flag_type/grounds, no expiry, no snapshot** — the "size/conflation
   grounds alone" scoping lives only in prose) and 2× stage1_finding_correction
   (`baf8ca57`, `12c3a5ce`, same shape + `related_finding_id`).
7. **Structured entity linkage already exists on the finding side**: `cited_entities`
   with per-entry `entity_uuid`, resolved server-side by `cite_entity`
   (`recon_report.py:132,152,1561-1583`); `harness._derive_affected_ids` already folds
   cited entity uuids into finding dedup identity. Flagged items are free-form dicts
   (`f.get('flag_type')`) — adding an `entity_uuid` key is schema-trivial but relies on
   the LLM stamping it.
8. **Metadata-query surface**: `get_memories_by_metadata(filters={'kind':...,
   'entity_uuid':...})` works today — exact top-level equality, AND across keys
   (`mem0_client.py:341-424`); no payload index on those keys (full scan, fine at
   current collection sizes).
9. **merge_entities orphans uuid-scoped records**: it deletes the deprecated node and
   rewrites nothing outside Graphiti (`memory_service.py:4204-4269`,
   `graphiti_client.py:1435-1500`). `rename_entity` is uuid-stable (safe).
10. **Staleness precedents in-tree**: ledger `expires_at` TTL machinery
    (`recon_ledger.py:295-363`); 14-day mem0 marker age sweeps
    (`task_knowledge_sync.py:658,674`); and — closest fit — the
    `scope_freshness.py` fingerprint pattern (store `subject_*` snapshot at decision
    time, compare against live state, mismatch → reinvestigate, `:259-261,:302-323`).
11. **Prior lessons that bound this design**: task 1966 (task-scoped blanket suppression
    hid a legitimate new finding → never blanket-suppress an entity), task 2503 (exact
    flag_type string match evaded by rewording → `canonical_flag_type_family()`
    token-multiset matching, `flag_dedup.py:271-311`), tasks 1185-1190 (the *simpler*
    task-scoped producer→reader contract needed ~6 hardening tasks).
12. `write_suppression_record()` (`flag_dedup.py:1086-1278`) has **zero production call
    sites** — it is an operator API. The standing-decision writes observed in the wild
    were Stage-2-LLM-authored, autonomously.

## Gap analysis

The complaint class "entity X is too big / conflated" is re-derived from raw graph
state each cycle. Every existing defense keys on task_id (± flag_type family), so:
no task_id → gate passes (fact 3); reworded flag_type → family match may still miss;
recon_report channel → not gated at all (fact 5). Each recurrence costs a Stage-2
investigation (LLM calls) and risks a duplicate curator task (reify 5191/5243 both
filed then cancelled; 5 not-actionable investigation records on one entity).

## Design sketch (recommended)

**New first-class record: `entity_standing_decision`** in the recon_ledger (migration:
add nullable `entity_uuid` column; SQLite `ALTER TABLE ADD COLUMN` is cheap).

Fields: `entity_uuid`, `grounds` (small **closed vocabulary**, initially just
`structural_size_conflation`), `decided_at`, `expires_at` (**default TTL ~90d — not
immortal**, unlike task-scoped rows), evidence refs (task_ids, mem0 ids, escalation
id), and a freshness fingerprint: `edge_count_at_decision` (+ optionally newest-edge
timestamp). Writer helper `write_entity_standing_decision()` mirroring
`write_suppression_record` (ledger row + mem0 mirror), schema rendered into prompts
via `recon_self_model` like `render_suppression_schema_section()`.

**Why a closed `grounds` enum, not flag_type matching**: task-2503 showed free-text
flag_type drifts; task-1966 showed blanket scoping over-suppresses. A one-value enum
the LLM classifies into ("is this complaint about aggregate size/conflation?") is
robust to wording and precisely scoped. Complaints citing a **specific edge uuid**
(a NEW concrete fact) are defined as outside `structural_size_conflation` — exactly
the escape hatch b0057f3d's prose already specifies.

**Hook point A — flagged_items chain** (the literal ask): a sibling filter next to
`filter_suppressed` in the `memory_consolidator.py:232-328` post-processor chain
(precedents: `filter_terminal_metadata_flags`, `filter_already_tracked_systemic_patterns`).
Match: flag's new optional `entity_uuid` field **plus a UUID-regex scan of the flag
text** (catches LLM omission; UUIDs are high-entropy so no false-positive risk) against
active rows with matching grounds. Drop = suppression semantics: excluded from marker
acknowledgment (recurrence history preserved), stat
`entity_standing_decision_suppressed`.

**Hook point B — recon_report findings channel** (what would have actually stopped the
incident): **annotate, never drop** — preserving 1966's visibility guarantee. When
`cite_entity`/`add_finding` links a finding to an entity with an active standing
decision, the tool **response** carries the decision (id + content) back to the Stage-1
LLM immediately, and the persisted finding gets a `standing_decision_id` annotation
that Stage 2's payload rendering surfaces with a one-line prompt rule: "an annotated
finding needs no re-investigation and must not spawn a curator task absent a NEW
concrete fact." Cost per recurrence collapses from a full Stage-2 investigation to a
skip.

**Prompt layer** (advisory, both stages): pre-emission check via
`get_memories_by_metadata(filters={'kind': 'entity_standing_decision', 'entity_uuid':
...})` — works today, fact 8.

### Staleness bounding (anti-over-suppression)

1. **TTL**: `expires_at` default ~90d; the claim "current size is organic growth"
   naturally decays. Existing `gc()` machinery already enforces it.
2. **Growth fingerprint sweep**: per recon cycle, a small sweep (Stage-1 post-processing
   or Stage-3 integrity; few rows) compares live edge count vs `edge_count_at_decision`;
   growth past tolerance (e.g. +25% or +15 edges) → row `state='expired'`. Keeps the
   gate itself pure-SQLite/sync (no new async Graphiti dependency in `flag_dedup`);
   sweep failure = row stays active this cycle, bounded by TTL anyway.
3. **merge_entities hook**: merging either uuid → expire the decision (post-merge
   entity is a new subject; re-deriving the complaint once is *correct* then). Also
   the first fix for the dangling-uuid hazard (fact 9). `rename` needs nothing.
4. **Scope guards**: grounds-scoped only (never blanket per entity); channel B never
   drops; edge-specific complaints never match.

## Scope: /prd (recommended) vs single task

**Recommend a small /prd, ~6 leaves**: (1) ledger migration + record kind + lifecycle
(TTL/GC/merge-expiry); (2) writer helper + authorization policy + mem0 mirror;
(3) hook A filter + stats + tests; (4) hook B annotation (cite_entity/add_finding
response + finding annotation + Stage-2 rendering); (5) prompt updates both stages +
self-model schema section; (6) freshness sweep + round-trip/staleness/merge tests.
Producer→reader contract across a serialization boundary is exactly the shape that
needed 1185-1190's six hardening tasks last time; and the seam count (ledger schema,
flag_dedup, recon_report server, two prompts, self-model) exceeds one coherent edit.

**Minimal alternative** (single task): hook A only + operator-written records. Honest
caveat: it would *not* have prevented the motivating incident (channel B), only the
cheaper flagged_items variant — fails the "would this have stopped the thing that
prompted it" test.

## Open questions for Leo

1. **Authorization** — may Stage 2 autonomously write standing decisions (recon
   self-silencing risk), or human-gated only? Middle ground: Stage 2 may write one only
   when citing ≥1 human-touched artifact (resolved escalation / interactively-curated
   session) or ≥N independent not-actionable investigation records.
2. **Grounds vocabulary** — closed enum (recommended) vs flag_types-family list?
3. **Channel B semantics** — annotate-only (recommended) vs also drop after N annotated
   recurrences?
4. **Defaults** — TTL 90d? growth tolerance +25%/+15 edges?
5. **Backfill** — migrate reify's b0057f3d (and the two corrections) into ledger rows so
   the reify 'orchestrator' entity is covered from day one, keeping mem0 originals as
   evidence?
6. Whether the existing ad-hoc kinds (`recurring_flag_standing_decision`,
   `stage1_finding_correction`) stay as free-form evidence records (and prompts stop
   presenting them as suppression-effective), with the ledger kind as the sole
   machine-consulted form.
