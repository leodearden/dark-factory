# Capability manifest — stage1-entity-standing-decision-prd.md

Decompose-time capability→evidence bindings (G3+G6 mechanization), verified against
main `01a5319a39` on 2026-07-22 (PRD committed at `06beddc7e5`). Machine-readable
twin: `stage1-entity-standing-decision-prd.capability-manifest.yaml` (same stem;
task_id stamped by `commit_planning`).

**Path corrections vs the PRD text** (files moved under package subdirs; every seam
verified present): `memory_consolidator.py` and `task_knowledge_sync.py` live under
`reconciliation/stages/`; `graphiti_client.py` and `mem0_client.py` under `backends/`;
`memory_service.py` under `services/`. `get_memories_by_metadata` is implemented at
`services/memory_service.py:3298` (MCP tool `server/tools.py:1567`; exact-match AND
primitive `backends/mem0_client.py:296 count_by_metadata`). Task descriptions carry
the corrected anchors.

## α — Ledger substrate + lifecycle (intermediate; unlocks β/γ/δ/ε/ζ)

| Capability | Evidence | Verdict |
|---|---|---|
| SQLite recon_ledger store + SCHEMA const + upsert/list/gc machinery | grep:`fused-memory/src/fused_memory/reconciliation/recon_ledger.py:295` `async def gc(` — wired (gc callsite `stages/task_knowledge_sync.py:827`) | PASS |
| Migration path (ALTER TABLE ADD COLUMN nullable `entity_uuid`) | recon_ledger owns its SCHEMA; α itself performs the migration (house pattern per task 2219) | PASS |
| TTL-expiry flip (state='expired', reason='ttl' — not DELETE) | **α's own work**: existing gc() DELETEs marker kinds; α adds the flip arm for this record kind. In-batch producer = α | PASS |

## β — Writer + authorization (intermediate; unlocks ε/ζ/η)

| Capability | Evidence | Verdict |
|---|---|---|
| Helper template (`write_suppression_record`) | grep:`reconciliation/flag_dedup.py:1086` | PASS |
| Recon-report tool factory seam | grep:`server/recon_report.py:2099` `create_recon_report_server` | PASS |
| Stage disallow lists | grep:`reconciliation/cli_stage_runner.py:65-67` `STAGE1_DISALLOWED`/`STAGE3_DISALLOWED` — **NOTE at :68-72 forbids adding recon-report tools (in-process-writes rationale); β's tool is the first with durable ledger writes, so β adds entries AND amends the comment** | PASS |
| Arm-2 metadata counting (`get_memories_by_metadata`, exact-match AND) | grep:`services/memory_service.py:3298`; count primitive `backends/mem0_client.py:296` | PASS |
| Edge-count sampling (`get_valid_edges_for_node`) | grep:`backends/graphiti_client.py:945` | PASS |
| Structured rejection naming unmet arm | rejection-mechanism: built + bound by β itself; β's tests observe the rejection fire (PRD boundary row 1) | PASS |
| `investigation_outcome` kind constant | producer:task-α upstream (day-one record emptiness accepted by design — PRD decision 8; arm 1 + backfill cover early cases) | PASS |

## γ — Hook A filter + storm escape (unlocks η)

| Capability | Evidence | Verdict |
|---|---|---|
| Consolidator post-processor chain seam | grep:`reconciliation/stages/memory_consolidator.py:260,:280` (`filter_terminal_metadata_flags`, `filter_already_tracked_systemic_patterns` precedents) | PASS |
| Sibling suppression-gate precedent + semantics | grep:`reconciliation/flag_dedup.py:344` `filter_suppressed` (fail-open WARNING precedent verified at its except-arm) | PASS |
| No-task-anchor gap being closed | grep:`reconciliation/flag_dedup.py:521-524` (anchorless flags pass) | PASS |
| Grounds enum + token-family list | producer:task-α upstream (single source, INV-5) | PASS |
| Per-cycle stats + recon escalation substrate | house substrate (cycle summary stats; 1755 storm-counter precedent) | PASS |

## δ — Hook B annotation (unlocks η)

| Capability | Evidence | Verdict |
|---|---|---|
| `add_finding` / `cite_entity` state methods | grep:`server/recon_report.py:922,:1553` (PRD's line order swapped; both present) | PASS |
| Zero-renderer-change verbatim payload flow | grep:`reconciliation/stages/task_knowledge_sync.py:3238` `_format_flagged`, `:3268` `json.dumps(item)` | PASS |
| Active-decision-by-uuid lookup | producer:task-α upstream (shared helper with γ, INV-5) | PASS |

## ε — Prompt/self-model layer + investigation_outcome convention (leaf)

| Capability | Evidence | Verdict |
|---|---|---|
| Schema-section renderer precedent | grep:`reconciliation/recon_self_model.py:310` `render_suppression_schema_section` (consumed `prompts/stage1.py:498-560`) | PASS |
| Stage-2 renderer consumption | **new wiring by ε itself** (stage2 newly consumes the renderer — in-batch producer = ε; drift/pinning test per house 2559 is the binding) | PASS |
| Advisory pre-emission metadata query | grep:`services/memory_service.py:3298` | PASS |
| `investigation_outcome` kind constant | producer:task-α upstream | PASS |

## ζ — Freshness sweep + merge invalidation (unlocks η)

| Capability | Evidence | Verdict |
|---|---|---|
| Stage-2 tail sweep seam (marker-sweep siblings) | grep:`reconciliation/stages/task_knowledge_sync.py:659-1092` sweep helpers + `ledger.gc()` callsite `:827` | PASS |
| Live edge count (`get_valid_edges_for_node`) | grep:`backends/graphiti_client.py:945` | PASS |
| `merge_entities` hook site | grep:`services/memory_service.py:4204` | PASS |
| `edge_count_at_decision` snapshot field | producer:task-α upstream | PASS |
| Recon escalation on sweep-failure streak | house substrate (1755 precedent; INV-4) | PASS |

## η — Backfill + E2E integration gate (leaf; G5-H gate)

| Capability | Evidence | Verdict |
|---|---|---|
| Backfill source record exists | **verified live 2026-07-22**: mem0 `b0057f3d-dc53-4cf8-9d1f-9959bd0897bd` (reify, `kind=recurring_flag_standing_decision`, `entity_uuid=f02a32ea-0efd-4865-94b4-97a412d8ffda`) — sole record of its kind (`get_memories_by_metadata` total=1) | PASS |
| Writer + gate (migration uses β's helper with `authorized_by`) | producer:task-β upstream | PASS |
| Hook A / Hook B / sweep legs of the E2E | producer:tasks γ/δ/ζ upstream (DAG-direction ✓ — all legs upstream of η) | PASS |
| Never-drop count-invariance (boundary row 7) | rejection-class binding to in-batch producer δ; η's E2E observes it | PASS |
| Tool-visibility config assertion (boundary row 14) | rejection-class binding to in-batch producer β; η asserts both disallow-list entries | PASS |

## G6 note

No numeric-floor or grammar-fixture class capabilities in this batch. Rejection-class
signals (boundary rows 1, 4, 7, 14) bind to their in-batch producer tasks with tests
observing the rejection/invariance fire, per PRD §Capability bindings draft.

## G7 walk (decompose-time, all 7 tasks)

No unresolved hits; no waivers. Notable judgment: γ's ledger-read fail-open carries no
streak escalation — this mirrors `filter_suppressed`'s house behavior byte-for-byte
(WARNING at the failure point, pass-through), the failure direction is the safe one
(under-suppression: noise surfaces, findings never hidden — 1966's direction), and the
two genuinely-new fail-soft paths (per-decision suppression storm in γ, sweep-failure
streak in ζ) both carry recon-escalation storm escapes per PRD §Storm escapes.
