# Capability Manifest — Tier-1 Prompt-Optimization Pilot

Mechanizes G3 (substrate exists) + G6 (premise valid) per **leaf** task of
`plans/tier1-prompt-optimization-prd.md`. Each capability a task's
`user_observable_signal` asserts is bound to evidence. **Any FAIL binding blocks
the batch.** Substrate verified live on `main` `ba8151bcc6`, 2026-07-12.

Binding vocabulary: `grep:file:line wired` (substrate present) · `producer:T-N`
(built by an upstream batch task) · `producer:2484` (built by a cross-PRD
upstream, wired as a dep) · `floor:bound>X` (numeric premise, pool ≫ floor) ·
`fallback-mechanism` (a real rejection/fallback path built here).
FAIL sentinels: declared-only · test-only · producer-downstream · producer-absent · fixture-ERROR · bound≤floor · rejection-absent.

| Leaf | Asserted capability | Binding | Verdict |
|---|---|---|---|
| **T1** loader | `shared/` importable by orchestrator **and** fused_memory (P-1) | `grep:shared/src/shared/__init__.py exists`; orchestrator imports `shared` (`run_store.py`, `usage_gate.py`, …); fused_memory imports it (`task_curator.py:41 from shared.cli_invoke import …`) | **PASS** |
| **T1** loader | loader composes `CONTRACT + artifact-heuristics`, records provenance | `producer:T1` (the deliverable itself) | **PASS** (self) |
| **T1** loader | unpin restores in-code constant (rollback lever) | `fallback-mechanism` — a real fallback path built + boundary-tested here | **PASS** |
| **T2** reviewer split | `_reviewer_role` exists to refactor | `grep:orchestrator/src/orchestrator/agents/roles.py:379` | **PASS** |
| **T2** reviewer split | reviewer's **live** output contract = post-2484 (`submit_review_verdict` tool) | `producer:2484` (verdict-servers δ; wired as dep — P-3). Freezing pre-2484 "output pure JSON" text would be **declared-only/stale** → the 2484 dep is what makes this PASS | **PASS (via dep 2484)** |
| **T2** reviewer split | reviewer_trial rollout produces a valid verdict via the live transport | `grep:orchestrator/src/orchestrator/evals/reviewer_trial/runner.py` (rollout harness) + `producer:2484` (transport) | **PASS** |
| **T2** reviewer split | CONTRACT-token boundary test (artifact can't alter contract) | `producer:T2` (G5 two-way boundary test = the signal) | **PASS** |
| **T3** curator split | `_SYSTEM_PROMPT` / `_BATCH_SYSTEM_PROMPT` exist to refactor | `grep:fused-memory/src/fused_memory/middleware/task_curator.py:350,:428` | **PASS** |
| **T3** curator split | curator still emits via `--json-schema` (NOT migrated by verdict-servers) | `grep:task_curator.py:2002 output_schema=CURATOR_OUTPUT_SCHEMA`, `:2093` batch — verdict-servers 2487 explicitly preserves this path. So the JSON-schema CONTRACT section is real | **PASS** |
| **T3** curator split | parity: replay yields same action pre/post refactor | `producer:T3` (parity regression = the signal) | **PASS** |
| **T4** reviewer corpus | `runs.db task_results` carries review_cycles/verify_attempts for FN mining | `grep:data/orchestrator/runs.db task_results(review_cycles,verify_attempts,cost_usd)` (2,086 rows) | **PASS** |
| **T4** reviewer corpus | `data/escalations/` FN-label source present | `grep:data/escalations/ (~2,352 files)` | **PASS** |
| **T4** reviewer corpus | achievable annotated corpus ≥~50 (target ~100) for a 2:1:7 split | `floor:candidate_pool=2086 task_results ≫ 50` — pool vastly exceeds floor; **FN-yield per mining heuristic refined in T4** (Open-Q §9). Not a hard number claimed as done-signal | **PASS** (pool-backed; not a bare number) |
| **T4** reviewer corpus | existing scorer runs green over expanded corpus | `grep:orchestrator/src/orchestrator/evals/reviewer_trial/scorer.py` | **PASS** |
| **T5** curator corpus | `tickets.db` carries curator decisions to replay | `grep:data/reconciliation/tickets.db` (4,019 decisions) | **PASS** |
| **T5** curator corpus | scorer conforms to the loop's scorer Protocol | `producer:T6` (Protocol; wired dep T5→T6) | **PASS** |
| **T5** curator corpus | labels are honest (decisions ≠ ground truth) | `fallback-mechanism` — frontier-adjudication + human spot-check protocol (D-6) before use as labels | **PASS** |
| **T6** loop engine | `invoke_agent` accepts an optimizer model ≠ executor (P-2) | `grep:orchestrator/src/orchestrator/agents/invoke.py:52 invoke_agent(… model: str = 'opus' …)` | **PASS** |
| **T6** loop engine | bounded edits / rejected-edit buffer / paired+repeat gate | `producer:T6` (the engine deliverable) | **PASS** (self) |
| **T6** loop engine | never mutates CONTRACT | `producer:T6` — optimizer handed only the heuristics text (D-3 by-construction) | **PASS** |
| **T7** smoke | full stack loader→reviewer HEURISTICS→loop→scorer→report | `producer:T2` + `producer:T6` (wired deps) | **PASS** |
| **T7** smoke | held-out verdict + provenance + CONTRACT-untouched proof on a fixture | `producer:T7` (the hermetic fixture run = the signal) | **PASS** |
| **T8** canary | `runs.db` logs cost-per-done-task / requeue / review_cycles / verify_attempts | `grep:runs.db task_results(cost_usd,verify_attempts,review_cycles,steward_cost_usd)`; requeue-rate + cost-per-done derivable from status + cost columns | **PASS** |
| **T8** canary | rollback = unpin via the loader | `producer:T1` (wired dep T8→T1) | **PASS** |

**Result: all bindings PASS — batch is clear to queue.** The only premise
needing care is T4's corpus size, bound as a pool floor (2,086 candidate rows ≫
the ≥50 needed), with the per-heuristic FN-yield deliberately left as a T4
tactical refinement (§9), not asserted as a fixed number in the done-signal (G6).

**Cross-PRD note:** T2's reviewer-contract binding PASSes *only via the wired dep
on 2484* — without it, T2 would freeze a soon-deleted contract (a stale/declared
premise). This is why 2484 is a hard bare-integer dep (P-3), not advisory.
