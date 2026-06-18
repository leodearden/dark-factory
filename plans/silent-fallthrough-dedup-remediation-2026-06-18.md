# Silent-fall-through-on-error: sweep findings & dedup-centric remediation

**Date:** 2026-06-18
**Origin:** task 1799 (`get_external_statuses` silent strand of reify 4635). User asked to sweep the
whole codebase for the same anti-pattern, categorise, and produce a fix plan.
**Status:** APPROVED (scope = all 48 sites, 6+ phases; add wholesale prevention; file as fused-memory
tasks via the /prd decompose flow). This doc is the durable analysis + strategy record; the formal PRD
+ task graph are decomposed from it.

## The anti-pattern

Code hits an abnormal / failure / unexpected-shape condition and returns a **benign default** (empty
dict/list, `None`, `False`, `0`, unchanged input) **without** signalling the failure in any observable
way — no raise, no error slot set, no `WARNING+` log, no escalation. The failure result is
**indistinguishable from a legitimate empty/negative success**, so the caller silently misreads it and
the system degrades *fail-closed AND invisible*. This is the exact inverse of the project directive
**"prefer loud escalation over silent degradation."**

Reference (task 1799): `Scheduler.get_external_statuses` falls through to `return {}, None` when the
MCP result does not parse to a dict — empty cache *with the error slot left None* — so the cross-project
dep gate fails closed with zero events/logs/escalations and stranded reify 4635 for hours.

## Root cause (user's reframe — drives the whole plan)

> "This class of errors is the cost of untreated code duplication. The fixes should include meticulous
> refactoring to extract and unify the duplicated logic" — and **file follow-up tasks for any other
> duplication discovered in the process.**

The evidence: **almost every one of the 48 sites has a twin — in the same file or a sibling file —
that already does it correctly (loud).** The defect is overwhelmingly "one branch of a copy-paste pair
forgot to mirror its loud sibling." Therefore each fix is *extract the one correct implementation into a
shared/local primitive and route every divergent copy through it* — not a per-site WARNING sprinkle.

Documented loud↔silent twins (extraction targets):
- `harness._reap_orphan_worktrees:1570` (loud) ↔ `harness._reconcile_stranded_in_progress:1745` (silent)
- `merge_queue_store._save_raw`/`load` (loud) ↔ `merge_queue_store._load_raw:153` (silent)
- `sqlite_task_backend._row_to_task:252` (loud, deduped WARNING) ↔ `_merge_metadata:1283`/`remove_dependency:1105` (silent)
- `workflow._run_reviewer:4476` (loud, ERROR verdict) ↔ `verify.py:283`/`agent_loop.py:387`/`review_checkpoint.py:244` (silent)
- `app.py:953` capped_now ("so the silent degrade is always visible in logs") ↔ `app.py:962` accounts_summary (DEBUG)
- recon `harness:1677` journal-read WARNING ↔ `harness:1648` persistence-count (silent)
- dashboard `fetch_tasks`/`_shape_one_project:175` offline marker (loud) ↔ `discover_orchestrators:291`/`fetch_external_statuses:217` (discard the marker)
- scheduler external-dep path `_external_resolver_failed` flag + grace counter (loud) ↔ `acquire_next:2486` local backfill (discards `_backfill_err`)

## Substrate decision (verified)

`dark-factory-shared` (`shared/src/shared/`) is a workspace dep of **orchestrator, fused-memory,
dashboard, sampler** (orchestrator already imports it widely). So the canonical primitives live in
`shared/`:
- **`shared/mcp_envelope.py`** — `parse_tool_result(result, key, expected_type) -> (value, error)`:
  logs a *distinct* WARNING for each abnormal shape (no text block / key absent / inner-not-dict /
  wrong-type) and never collapses a non-raising malformed payload into a bare `None`/`{}`/`[]`.
- **`shared/safe_io.py`** — `load_json_or_warn(path, *, default, on_corrupt)`: splits the benign
  `FileNotFoundError`/first-run-absent branch (silent) from `JSONDecodeError`/`ValueError` corruption
  (WARNING + optional fail-closed / quarantine). Reuses a deduped-warning style like `_row_to_task`.
- **resolver-guard** helper for the `(value, error)` discipline: `if err or not value: warn + fail-safe`.
- **agent-result guard** for structured-output: return a distinguishable ERROR/parse-failed sentinel
  instead of laundering a failed agent run into a neutral verdict / "0 findings".

**escalation** does NOT depend on `shared` → its one site (`dedupe.py:292`) is fixed in-package by
mirroring its own twins (`queue.py:~636`, `watcher.py:100`, which fall back to `datetime.min`).

**Decomposition constraint:** no task may span packages (cross-package scopes blow the architect
budget — see feedback memory). The shared primitives are ONE task (package: shared); every consuming
package gets its OWN migration task that depends on it.

## Inventory — 48 verified instances (4 critical, 15 high, 12 medium, 17 low)

Each survived adversarial verification (a skeptic opened the code and tried to refute it).

| # | File:line | Symbol | Category | Sev | Family |
|---|---|---|---|---|---|
| 1 | orchestrator/.../scheduler.py:1508 | get_statuses | parse_guard | 🔴 | A |
| 2 | orchestrator/.../scheduler.py:2486 | acquire_next (local backfill) | empty_on_error | 🔴 | B |
| 3 | orchestrator/.../scheduler.py:1219 | get_tasks | parse_guard | 🔴 | A |
| 4 | orchestrator/.../scheduler.py:1122 | _parse_tool_text_result (ROOT) | parse_guard | 🟠 | A |
| 5 | orchestrator/.../merge_queue_store.py:153 | _load_raw | bare_except | 🟠 | C |
| 6 | orchestrator/.../scheduler.py:1430 | get_status | parse_guard | 🟢 | A |
| 7 | orchestrator/.../merge_queue.py:627 | _main_health_fingerprint | bare_except | 🟢 | H |
| 8 | orchestrator/.../harness.py:1745 | _reconcile_stranded_in_progress | empty_on_error | 🟠 | B |
| 9 | orchestrator/.../substrate_gate.py:93 | extract_probe_set | parse_guard | 🟠 | A |
| 10 | orchestrator/.../b3_gate.py:78 | _load_state | bare_except | 🟠 | C |
| 11 | orchestrator/.../b3_gate.py:401 | _resolve_cap | bare_except | 🟡 | C |
| 12 | orchestrator/.../harness.py:4508 | _scan_for_terminal_active_tasks | empty_on_error | 🟢 | B |
| 13 | orchestrator/.../harness.py:2141 | _reconcile_one_stranded (plan.lock) | bare_except | 🟡 | C |
| 14 | orchestrator/.../review_checkpoint.py:244 | _run_review | empty_on_error | 🟡 | E |
| 15 | orchestrator/.../workflow.py:7441 | _build_train_state | empty_on_error | 🟢 | B |
| 16 | orchestrator/.../git_ops.py:2020 | get_merge_diff_files | empty_on_error | 🟢 | H |
| 17 | orchestrator/.../evals/metrics.py:245 | _git_diff_stats | bare_except | 🟢 | H |
| 18 | fused-memory/.../reconciliation/agent_loop.py:387 | _call_claude_cli | parse_guard | 🟠 | E |
| 19 | fused-memory/.../reconciliation/stages/memory_consolidator.py:263 | episodes fetch | empty_on_error | 🟡 | B |
| 20 | fused-memory/.../reconciliation/stages/memory_consolidator.py:279 | mem0 get_all fetch | empty_on_error | 🟡 | B |
| 21 | fused-memory/.../reconciliation/stages/memory_consolidator.py:278 | results key default | missing_key | 🟢 | A |
| 22 | fused-memory/.../reconciliation/verify.py:283 | CodebaseVerifier.verify | missing_key | 🟠 | E |
| 23 | fused-memory/.../reconciliation/harness.py:1648 | _finding_persistence_count | degrade_suppresses | 🟡 | G |
| 24 | fused-memory/.../reconciliation/harness.py:713 | _fetch_task_count_census | parse_guard | 🟡 | A |
| 25 | fused-memory/.../reconciliation/targeted.py:565 | _sweep_cancelled_descendants | parse_guard | 🟠 | A |
| 26 | fused-memory/.../reconciliation/stages/task_knowledge_sync.py:1688 | _apply_post_flight_guards | empty_on_error | 🟠 | A/B |
| 27 | fused-memory/.../reconciliation/stages/base.py:357 | _find_fused_memory_server | bare_except | 🟢 | C |
| 28 | fused-memory/.../reconciliation/queue_health.py:43 | summarize_graphiti_queue_health | missing_key | 🟢 | A |
| 29 | fused-memory/.../middleware/task_interceptor.py:3704 | interceptor_write_succeeded | missing_key | 🟡 | A |
| 30 | fused-memory/.../middleware/task_interceptor.py:1384 | _check_escalation_idempotency | empty_on_error | 🟠 | B |
| 31 | fused-memory/.../middleware/task_interceptor.py:3306 | _extract_metadata_files | missing_key | 🟢 | A |
| 32 | fused-memory/.../services/memory_service.py:918 | MemoryService.search | empty_on_error | 🟡 | F |
| 33 | fused-memory/.../backends/sqlite_task_backend.py:1283 | _merge_metadata | bare_except | 🟠 | D |
| 34 | fused-memory/.../backends/sqlite_task_backend.py:1105 | remove_dependency | bare_except | 🟢 | D |
| 35 | dashboard/.../data/db.py:156 | with_db | empty_on_error | 🔴 | F |
| 36 | dashboard/.../data/db.py:117 | DbPool.get | none_false | 🟠 | F |
| 37 | dashboard/.../data/orchestrator.py:291 | discover_orchestrators | empty_on_error | 🟠 | F |
| 38 | dashboard/.../data/tasks.py:217 | fetch_external_statuses | parse_guard | 🟠 | A/F |
| 39 | dashboard/.../data/redux_api.py:245 | _shape_wal_status | parse_guard | 🟡 | F |
| 40 | dashboard/.../data/orchestrator.py:198 | read_task_artifacts | bare_except | 🟡 | C |
| 41 | dashboard/.../app.py:962 | api_curator (accounts_summary) | degrade_suppresses | 🟡 | G |
| 42 | dashboard/.../data/metrics.py:282 | _split_queue_stats | parse_guard | 🟢 | A |
| 43 | dashboard/.../static/redux/tab_overview.jsx:41 | HostLoadCard.fetchLoad | empty_on_error | 🟢 | H |
| 44 | escalation/.../dedupe.py:292 | find_dedupe_parent | bare_except | 🟢 | H (local twin) |
| 45 | shared/.../pytest_jobserver.py:99 | pytest_configure | bare_except | 🟢 | H |
| 46 | sampler/.../metrics.py:210 | collect_process_metrics | empty_on_error | 🟠 | H |
| 47 | sampler/.../metrics.py:62 | parse_pressure_file | empty_on_error | 🟢 | H |
| 48 | scripts/reviewer_redundancy_diagnostic.py:55 | load_review | none_false | 🟢 | H |

Severity legend: 🔴 critical · 🟠 high · 🟡 medium · 🟢 low.

## Duplication families → unification target

- **A — MCP/tool-result envelope parsing** (parse_guard core): every resolver re-implements
  `isinstance(x, expected) else benign-default`. → migrate onto `shared.mcp_envelope.parse_tool_result`
  returning `(value, error)`. Sites: 1,3,4,6,9,21,24,25,26,28,29,31,38,42.
- **B — resolver error-slot discipline**: callers unpack `(value, err)` into `_`. → bind `err`, branch
  with a fail-safe guard mirroring `_external_resolver_failed` + grace counter. Sites: 2,8,12,15,19,20,26,30.
- **C — load-or-warn JSON state files**: → `shared.safe_io.load_json_or_warn`, split benign-absent from
  corrupt, fail-closed where safety-bearing. Sites: 5,10,11,13,27,40.
- **D — corrupt-metadata-blob (sqlite backend)**: → extract `_row_to_task`'s deduped-WARNING handler;
  write paths refuse/quarantine, never clobber `external_deps`/`memory_hints`. Sites: 33,34. (fused-memory only)
- **E — agent structured-output verdict extraction**: → shared agent-result guard returning a
  distinguishable ERROR sentinel (mirror `workflow._run_reviewer:4476`). Sites: 14,18,22.
- **F — dashboard offline/degraded marker propagation + visible logging**: → unify on the existing
  `{'offline':True,'error':...}` contract; bump DEBUG→WARNING; surface a red-badge sentinel. Sites:
  32,35,36,37,38,39,41,42. (dashboard; 32 is fused-memory memory_service — separate task)
- **G — degrade-suppresses-escalation**: mirror loud sibling; fail-toward-escalate. Sites: 23,41.
- **H — observability/offline tail**: mostly add-WARNING + mirror in-package twin; some reuse C/A.
  Sites: 7,16,17,43,44,45,46,47,48.

## Phased, package-scoped task graph

`P0` is the shared substrate; cross-cutting migrations depend on it. Family-D and the escalation/
sampler/scripts tail are independent of P0. Each task is single-package and RED-test-first
(reproduce the silent strand end-to-end BEFORE the fix, mirroring 1799).

- **P0 — shared loud-and-safe primitives** [shared] — `mcp_envelope.parse_tool_result`,
  `safe_io.load_json_or_warn`, resolver-guard, agent-result guard, with full unit tests. Blocks most below.
- **P1 — scheduler dispatch core** [orchestrator, dep P0] — sites 4,3,1,6,2 (root + get_tasks/statuses/
  status + backfill). Companion to 1799 (which fixes get_external_statuses); thread the error slot e2e.
- **P2 — orchestrator recovery/governor & misc** [orchestrator, dep P0] — sites 8,13,5,10,11,15,9,14,
  12,16,17,7. Split if needed (C-loaders vs B-resolvers vs E vs tail) but keep single-package.
- **P3 — fused-memory reconciliation** [fused-memory, dep P0] — sites 18,22,25,26,24,23,19,20,21,28,27.
  Split: recon-parse(A), recon-verify(E), recon-consolidator(B/G).
- **P4 — fused-memory task-backend metadata** [fused-memory, indep] — sites 33,34. Sharp irreversibility;
  focused review. The clobber-guard is the priority.
- **P5 — fused-memory middleware + services** [fused-memory, dep P0] — sites 30,29,31,32.
- **P6 — dashboard signal layer** [dashboard, dep P0 partial] — sites 35,36,37,38,39,40,41,42,43.
- **P7 — tail** [per-package single tasks] — escalation 44 (local twin), shared 45, sampler 46,47, scripts 48.
- **P8 — prevention: lint/CI rule** [shared / repo CI, dep ALL migrations] — AST/ruff-plugin rule for
  (a) `*, _ = await <resolver>(...)` (error-slot to `_`) and (b) `except (...): return <empty-literal>`
  with no `logger.warn/error/exception` in the handler body. Land last so it passes clean; encodes the
  "loud over silent" directive as an enforced invariant.

## Standing directive on every task

1. **Unify, don't patch.** Where a loud twin exists, extract the shared implementation and route the
   divergent copy through it; do not duplicate a WARNING into both branches.
2. **RED-test-first.** Reproduce the silent strand (non-raising malformed payload / corrupt file /
   discarded error slot) through the real code path and confirm it FAILS before fixing.
3. **Fail-closed where safety-bearing** (caps, dep gates, terminal-state guards, metadata writes);
   fail-safe-wait + visible WARNING elsewhere; never suppress an escalation to degrade.
4. **File follow-up tasks for other duplication discovered** while in the touched files — do not fix
   opportunistically beyond scope; capture it as a new task so the dedup effort compounds.
