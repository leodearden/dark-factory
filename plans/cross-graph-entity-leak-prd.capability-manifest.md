# Capability manifest — cross-graph-entity-leak-prd

Per-leaf capability→evidence bindings (mechanized G3+G6). PRD: `plans/cross-graph-entity-leak-prd.md` (commit `1cdcdeacba`); RCA: `plans/cross-graph-entity-leak-rca.md` (`5e13217a7a`). All grep cites verified against main 2026-07-06. Intermediates (α, ε) are covered through their consuming leaves.

## β — MCP-boundary normalization (signal B1 + B2)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `canonicalize_project_id` (S1) exists | `producer:task-α` — **upstream** dep, deliverable is exactly this fn + error type | PASS (producer-upstream) |
| Every memory tool has a prologue to hook | `grep:fused-memory/src/fused_memory/server/tools.py` — `validate_project_id` called in each memory-tool prologue (add_episode :642 region, add_memory :746, search :834, get_entity :1049, ungated mutators :1456-1895) | PASS (wired) |
| Known-project gate compares against **normalized** registry keys | `grep:server/tools.py:558` `_known_project_gate`; registry built via `resolve_project_id` at `models/scope.py:186` | PASS (wired) |
| B1 premise: `'dark-factory'` → accepted into `dark_factory` | gate passes iff normalized key ∈ registry; `dark_factory` is a registry key (config-registered project) | PASS |
| B2 rejection: path-shaped id rejected with specific error | rejection mechanism is α's deliverable (upstream); β's test authors the path-shaped input and observes the diagnostic. Path-shape detector precedent: `grep:fused-memory/scripts/investigate_cross_graph_duplication.py:65-79` `is_path_shaped_name` | PASS (producer-upstream + rejection-check authored in-task) |

## γ — GraphitiBackend group-arg normalization (signal B3)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `canonicalize_project_id` | `producer:task-α` upstream | PASS |
| Single graph-access funnel to normalize behind | `grep:backends/graphiti_client.py:236-252` `_driver_for`/`_graph_for` | PASS (wired) |
| Replay path reaches backend with persisted raw group_id | `grep:services/memory_service.py:940-967` replay executes `payload['group_id']`; `services/durable_queue.py:52` persisted column | PASS (wired) |
| B3 assertion is param-level (mock Cypher/param equality), no live-DB premise | test-suite convention is mock-graph (tests/test_merge_entities.py, test_purge_knowlive_namespace.py) | PASS |

## δ — Phase-0 deploy capstone (signal: fresh uptime, deterministic auto-deploy)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `scripts/restart-fused-memory.sh` exists, executable, no-arg = restart + health wait, exit 0/1 | `ls -la` `-rwxrwxr-x`; script lines 46-65 (`systemctl --user restart` + `/health` poll); `--drain` NOT used (task-2090 hang) | PASS (wired) |
| Deterministic auto-deploy preset (before_done + always_escalates=false), cross-unit blocking + fresh-PID verify | CLAUDE.md deterministic-task contract; worked precedent `producer:task-2212` (same script, same target_unit, filed and validated at submit) | PASS |
| `get_status` exposes fresh-uptime observability | precedent task-2212 signal (`uptime_seconds`); get_status MCP tool live | PASS |
| Race-fix code to deploy | `producer:task-2266` — **upstream**, filed 2026-07-06 | PASS (producer-upstream) |

## ζ — migration script (signal B6, mock-unit level)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Move/merge primitives (S5/S6) | `producer:task-ε` upstream | PASS |
| Byte-identical embedding transport (`--compact` float32 strings → `vecf32([...])`) | validated experiment recorded in RCA §6 Phase 1 (proven byte-identical on throwaway graphs, 2026-07-06); ζ's unit tests pin the passthrough as pure functions over recorded fixture strings | PASS (validated-experiment) |
| Census enumeration (GRAPH.LIST + paged foreign-node queries) | precedent `fused-memory/scripts/investigate_cross_graph_duplication.py` (merged, task 2116) + this session's live re-scan | PASS (wired) |
| Dry-run-first manifest pattern | precedent `fused-memory/scripts/purge_knowlive_namespace.py` (+ mock test tests/test_purge_knowlive_namespace.py) | PASS (wired) |
| No baked blast-radius numbers in tests (counts recomputed live) | PRD decision 3 — avoids numeric-premise fragility | PASS (no numeric bound asserted) |

## η — Phase-1 live migration gate (deterministic PURE GATE)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Pure-gate preset (`always_escalates=true`, no `before_done`) is a valid deterministic combo | CLAUDE.md field-combo table row 3 ("pure gate") | PASS |
| Script to run | `producer:task-ζ` upstream | PASS |
| Race fix + normalization RUNNING before apply | `producer:task-δ` upstream (deploy gate) | PASS (anti-recontamination ordering) |
| Live rehearsal substrate (`_probe*` throwaway keys) | `_probe`/`_probe_check` keys already exist in GRAPH.LIST (live scan 2026-07-06); GRAPH.DELETE cleans up | PASS |
| Post-verify observability (zero-foreign counts, recall spot-check) | read-only RO_QUERY census (2116 script) + `search`/`get_entity` | PASS |

## θ — consolidation script (signal B7, mock-unit level)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Move primitive with `rewrite_group_id` | `producer:task-ε` upstream (S5 param) | PASS |
| Qdrant collection enumeration/scroll/upsert/delete | `grep:backends/mem0_client.py:386-398` `_list_project_collections` (qdrant client wired); qdrant-client scroll/upsert are standard API | PASS (wired) |
| Guarded junk-key deletion (GRAPH.DELETE only at count 0) | purge_knowlive_namespace.py precedent (destructive-step flagging) | PASS |
| Alias map is reviewable config, unmapped → UNRESOLVED blocks apply | PRD decision 4 (no silent scope drops) | PASS |

## ι — Phase-2 live consolidation gate (deterministic PURE GATE)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Pure-gate preset | CLAUDE.md (as η) | PASS |
| Script | `producer:task-θ` upstream | PASS |
| Ordering after main migration | `producer:task-η` upstream (no double-handling of family strays) | PASS |
| Open data questions routed to this gate's human review (solar family, `reify_`) | PRD §Open questions 1-2 — review inputs, not premises of any RED test | PASS |

**No FAIL bindings.** Notable resolutions made during authoring: (a) B4-B6 byte-fidelity assertions were **re-homed from unit tests to the η/ι live rehearsals** after verifying the test suite is mock-based (mock-only-test hazard, ops-scripts lesson); (b) η/ι were made **pure gates** rather than act-then-ask because `before_done.script` is validated at submit time and their scripts are authored by ζ/θ after filing.
