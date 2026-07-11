# Capability Manifest — stage3-ledger-presence

Mechanizes G3+G6 for the PRD `plans/stage3-ledger-presence-prd.md`. One block per
task; each capability the task's signal asserts is bound to evidence. Any FAIL
value blocks the batch. All bindings below resolve **PASS** — the change is a read
of existing substrate (2219 `ReconLedgerStore`, 2229 `write_cycle_summary`) plus
one new read-only tool produced upstream of its single consumer.

Substrate confirmed on `main` = `6525ad7152`. Greek labels match the PRD
decomposition; task IDs filled in after filing.

## τ1 — `get_cycle_summary_presence` read-only MCP tool  *(task 2436; LEAF, tool-level signal)*
- `ReconLedgerStore.get_by_identity(project_id, record_kind, task_id='', flag_type='', run_id='')` → `grep:fused-memory/src/fused_memory/reconciliation/recon_ledger.py:203` on main — **PASS**
- `memory_service.recon_ledger: ReconLedgerStore | None` populated + `getattr(...,None)` guard precedent → `services/memory_service.py:516` (decl) / `:535` (set); precedent in `write_cycle_summary` — **PASS**
- Read-only tool registration shape (`@mcp.tool()` + `@mcp_tool_errors()`, `_canonicalize_project_id_arg`) → `server/tools.py:1128` `count_memories_by_metadata`, backed by `services/memory_service.py:2684` — **PASS**
- Stage-3 auto-allow (allow-by-default DISALLOW-list) → `cli_stage_runner.py:65` `STAGE3_DISALLOWED = writes+builtins` only; keep the tool OUT of every `DISALLOW_*` list — **PASS**
- Cycle_summary identity == `(project_id, 'cycle_summary', '', <stage>, <run_id>)` → `write_cycle_summary` record construction in `summary_pool.py` (`record_kind='cycle_summary'`, `task_id=''`, `flag_type=stage`, `run_id`) — **PASS** (verified live)
- Stage disambiguation (task_knowledge_sync vs memory_consolidator under shared run_id) → tool takes explicit `stage` → `flag_type`; prevents the task-1653 collision — **PASS**
- field-population (anti-sentinel): tool returns a real bool from a real DB read, not a stubbed constant → unit test drives a real `ReconLedgerStore.upsert` → read (no mock) — **PASS**
- G6 numeric/exactness branches N/A → presence is a boolean, no numeric bound / closed-form claim asserted — **PASS**

## τ2 — Stage 3 consult wiring + write→read boundary test  *(task 2437; LEAF, integration signal)*
- `get_cycle_summary_presence` capability → `producer:τ1` (task 2436) upstream — τ2 depends-on τ1; no producer downstream — **PASS**
- write→read seam: `write_cycle_summary` → `ledger.upsert` exists → `summary_pool.py` write path on main — **PASS**
- boundary-test home exists → `fused-memory/tests/test_summary_pool.py` already imports `write_cycle_summary` **and** `ReconLedgerStore` — **PASS**  *(corrects the brief's `test_recon_reliability_integration.py`, confirmed never in git history)*
- anti-inversion: ledger present → present, absent → missing (not wired backwards) → boundary test asserts BOTH directions — **PASS**
- fail-safe monotonicity: the existing Mem0 two-path fallback is RETAINED for every inconclusive read (ledger unwired / tool error) → design invariant, asserted by keeping Path 1/Path 2 and the inconclusive branch in the prompt — **PASS**
- Known-gap closure is observable → grep-absent assertion (the `stage3.py:9-22` comment removed) + tool/rule present in the prompt-content test — **PASS**
- G6 numeric/exactness branches N/A → boolean presence + prompt-content assertions, no numeric claim — **PASS**

## META
*Decompose-and-queue without further oversight → complete, coherent, cohesive, good?* **Yes.** Single consumer named and in-PRD (no orphan producer); both leaves carry honest user-observable signals (real-upsert unit test + write→read seam + prompt-content), not synthetic passes; all substrate verified on current main with the one brief inaccuracy corrected to a real test home; single owner, no reciprocal seam; one proportionate boundary test right-sized for a LOW-risk fail-safe change.
