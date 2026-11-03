# Capability manifest — S11 Curator pipeline and candidate types

PRD: `plans/review-all-s11-curator-pipeline-prd.md`. YAML twin:
`plans/review-all-s11-curator-pipeline-prd.capability-manifest.yaml`, which validates against
`shared.capability_manifest.CapabilityManifestDoc`. Every mechanical `delivered_check` was
evaluated with `shared.delivered_check_polarity` against `main` (`ae3047ec3a`, 2026-10-10). All
22 report FAIL at the authoring tree, which is the healthy polarity: each must flip to PASS when
its producer lands.

**Code anchors** verified against main `ae3047ec3a` (2026-10-10), cited by symbol.

**No FAIL bindings.** External producers sit upstream in the DAG:
- task 4513 is upstream of α, and transitively of everything;
- task 6385 is upstream of ζ.

## α: Extract the curator vocabulary into `middleware/curator_types.py`

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| `curator_types.py` exists | path present | built by α | PASS |
| driver no longer defines the five classes | grep absent `task_curator.py` | `CandidateTask`, `CuratorDecision`, `CuratorFailureError`, `PreparedCandidate`, `RewrittenTask` are defined there on main | PASS |
| former SCC members do not import the driver | grep absent over the six modules | 4 guards import under `TYPE_CHECKING`, `curator_escalator` imports `CuratorFailureError`, the ZOT sweep imports `embedding_text` (main) | PASS |
| registry-guard imports hoisted | grep absent (indented `from …guard import`) | `_maybe_blocklist_drop`, `_maybe_premise_refuted_drop` and `_maybe_route_deterministic` import function-locally on main | PASS |
| `normalize_title` mirror deleted | grep absent `def _normalize_title(` in `candidate_key.py` | mirror on main; α makes the public function the single home (PRD decision 2) | PASS |
| leaf loads nothing heavy | path present `test_curator_types_leaf.py` | α's fixture: `python -I` import + `sys.modules` assertion + AST import scan | PASS |

## β: Delete the dead single-ticket worker path and `_intra_batch_key`

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| four dead defs absent | grep absent `fused-memory/src/` | the defs exist on main; `git grep` finds no caller in `fused-memory/src`, `orchestrator/src` or `shared/src` | PASS |
| mirror-site prose corrected | grep absent `_intra_batch_key` in `agents/triage.py`, `candidate_key.py` | both name it on main | PASS |
| behaviour retargeted to public seams | manual | real `TicketStore` fixtures in `fused-memory/tests/test_ticket_worker.py`; `submit_task`/`resolve_ticket` public on `TaskInterceptor` | PASS |

## γ: `resolve_ticket` reports an open ticket as open

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| `TicketStatus` enum in the row owner | grep present `^class TicketStatus\(` in `ticket_store.py` | built by γ; the `awaiting_curation` member comes from 4513 (upstream) | PASS |
| no synthetic `failed/timeout` | grep absent `'reason': 'timeout'` in `task_interceptor.py` | present on main in `_resolve_ticket_raw` | PASS |
| open reason emitted | grep present `wait_timeout` | built by γ | PASS |
| dead orchestrator quote deleted | grep absent `def submit_resolve_instructions(` in `roles.py` | defined on main; no production renderer since `bfd19a296c` (`git log -S`) | PASS |
| canonical caller doc states the open rule | grep present `wait_timeout` in `skills/_shared/ticket-failure-handling.md` | absent on main | PASS |
| vocabulary parity fixture | path present `test_resolve_ticket_vocabulary.py` | built by γ (B5) | PASS |
| open hold owned and bounded (INV-7) | manual | `TicketJanitor` worker-dead reap and the `TicketStore` startup `server_restart` sweep (main); 4513's `awaiting_curation` backlog escalation (upstream) | PASS |
| rejection-mechanism check (G6 branch 4) | n/a | γ asserts no rejection; the `failed`-only-when-terminal invariant is pinned behaviourally by B1–B4 | PASS |

## δ: One renderer, one invoker, one failure mapper (intermediate → ε)

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| batch invoke twin deleted | grep absent `async def _call_llm_batch(` | defined on main | PASS |
| batch section twin deleted | grep absent `def _build_batch_section(` | defined on main | PASS |
| one curator wildcard site | grep present `TaskCurator\._invoke_curator` in `shared/tests/test_wildcard_mcp_scoping_gate.py` | `_KNOWN_SITES` names `_call_llm` and `_call_llm_batch` on main | PASS |
| invoker constructor seam | manual | `invoke_with_cap_retry` (shared CLI invoker) is already the call both twins make | PASS |

## ε: One curation pipeline for every batch size

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| `curate_prepared` is the pipeline | grep present `async def curate_prepared(` | built by ε | PASS |
| batch pipeline twin gone | grep absent `curate_batch_prepared` in `task_curator.py`, `task_interceptor.py` (narrowed at commit_planning: prose mentions in `config/schema.py` and `recon_claim_verification_guard.py` are outside ε's files) | defined and called on main | PASS |
| exact-match runs in batches (B6) | manual | `TaskCurator._pre_llm_exact_match` and `note_created` (called on the create path of `_dispatch_ticket_decision`) exist on main; only `curate` calls the former today | PASS |
| outage paths | manual, `producer:task-4513` | upstream; it provides `awaiting_curation`, internal `defer` and `curator_action='uncurated'` | PASS |

## ζ: Bound the curator's corpus and R4 reads; surface failed corpus streams

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| corpus read filtered | grep absent `await self._taskmaster.get_tasks(project_root)` in `task_curator.py` | unfiltered call in `_build_corpus` on main. Substrate: `TaskBackendProtocol.get_tasks(..., statuses=)` and `shared.task_statuses.ACTIVE` exist on main | PASS |
| R4 via metadata lookup | grep present `key=['"]suggestion_hash['"]` in `task_interceptor.py` | `producer:task-6385` (upstream) builds `TaskBackendProtocol.find_tasks_by_metadata`; it raises on a store error and never returns an empty result on failure | PASS |
| unavailable-streams census | grep present `unavailable_streams` in `curator_types.py` | built by ζ on α's `PoolWithheld` | PASS |
| dependency-stream substrate | manual | `TaskBackendProtocol.get_dependency_edges` and `get_statuses_raw(ids=)` exist on main | PASS |
| numeric premise (G6 branch 1) | basis stated | ~24 ms per candidate (1,693 open rows) against ~140 ms (6,665 rows): a stdlib `json.loads` proxy on the live store, 2026-10-10. The signal asserts the filter and the census, not a latency threshold | PASS |
