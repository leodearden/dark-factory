# PRD: fm-cancellederror-convention — structural CancelledError convention in fused-memory

**Status**: active — stream M5 of the 2026-07-06 bug-hotspot remediation program
(`plans/bug-hotspot-remediation-program-2026-07-06.md`). Date: 2026-07-06.
Approach: bare B (mechanical; per program brief, G5 pre-answered).

## Goal

Make fused-memory's asyncio-cancellation conventions **structural** instead of
hand-copied, closing the single most recurrent silent-failure class in the
subsystem (8+ historical fix tasks: 484, 512, 516, 647, 191/192, 519, 1151):

1. Extract the full gather-then-classify idiom (gather with
   `return_exceptions=True`, re-raise captured cancellation, then apply the
   site's Pass-2 policy) as two named helpers, and migrate every fused-memory
   call site — including the recon-stage sites that **currently swallow captured
   CancelledError values** (verified live bug, see Background).
2. Enforce the tools.py MCP-handler error-handling convention with one decorator
   applied uniformly, asserted by a mechanical walker test — so the convention
   cannot silently regress per-function again.
3. Dedup the canonical stage→recon_pool constant map into one importable leaf
   module (removes a documented drift-prone cross-module contract).

Operator-observable outcome: a cancellation delivered during shutdown aborts
reads/sweeps cleanly (propagates) instead of being degraded into warning lines,
error-dict responses, or "delete failed; not counted" accounting; MCP error
responses keep the exact `{'error', 'error_type'}` shape everywhere; CI carries
two mechanical drift-guards (gather-site guard, tool-handler walker) that fail
loudly on any new regression of either convention.

## Background — verified state of the substrate (2026-07-06)

All file:line references re-verified against main on 2026-07-06.

- `fused_memory/utils/async_utils.py:37` — `propagate_cancellations(results)`
  centralizes **Pass 1** only (re-raise the first bare BaseException in gather
  results). Pass 2 (classify ordinary Exceptions) is hand-copied at 5 sites with
  two distinct semantics:
  - raise-first-after-logging: `services/memory_service.py:1755-1777` and
    `:1795-1832` (get_entity exact + fuzzy branches),
    `backends/graphiti_client.py:1616-1632` (stale-summary edge fetch).
  - collect-per-item-and-continue: `services/memory_service.py:2639-2668`
    (rebuild_entity_summaries), `reconciliation/context_assembler.py:131-150`
    (degrade failed fetch to empty context).
- **Live swallow class (γ's premise, verified)**: many other
  `gather(return_exceptions=True)` sites never got Pass 1 at all and use
  `isinstance(result, BaseException)` log-and-continue — a captured
  CancelledError value is silently degraded to a WARNING and the run keeps
  going. Verified instances: `reconciliation/stages/task_knowledge_sync.py:1234`
  (and its sibling sweeps at ~1408/1608/1733/2198/2230/2273/2328/2803),
  `reconciliation/flag_dedup.py:2258`, `reconciliation/summary_pool.py:121`,
  `reconciliation/stage1_stall_detector.py:168+199`. This is the same class the
  8 historical fixes addressed — these sites were never covered.
- **tools.py convention state**: 49 `@mcp.tool()` handlers in
  `server/tools.py`; 42 carry the hand-written
  `except (asyncio.CancelledError, KeyboardInterrupt, SystemExit): raise`
  guard before their broad `except Exception → {'error', 'error_type'}` tail;
  4 task/ticket-routing handlers lack it (`submit_task` tail at :2826,
  `resolve_ticket` :2888, `list_tickets` :2940, `cancel_ticket` :3025);
  3 handlers have no broad tail at all (`get_wal_status`,
  `unhalt_reconciliation`, `get_external_statuses`).
- **G6 severity correction (important — read before writing any RED test)**:
  the survey's claim that the 4 unguarded handlers "silently convert
  cancellation into an error-dict response" is **false at runtime** on current
  main. Verified two ways: (a) on CPython ≥3.8 (repo runs 3.13),
  `except Exception` does **not** catch `asyncio.CancelledError` (it is a bare
  BaseException) — empirically confirmed; (b) the central defence-in-depth
  wrapper `_install_safe_tool_wrapper` (`server/main.py:1475-1521`, tested by
  `tests/test_tool_safe_wrapper.py`) wraps FastMCP's ToolManager.call_tool for
  **all** tools and re-raises bare CancelledError while containing every other
  BaseException. So the tools.py work is **convention enforcement + dedup of 42
  hand-copies + drift-proofing** (the dangerous future edit is a tail widened to
  `except BaseException`, which *would* swallow cancellation before the central
  wrapper sees it) — NOT a live-bug fix. **Do not author a RED test asserting
  the 4 handlers swallow cancellation today; it cannot go red.**
- Task 1151 (SqliteTaskBackend._txn) is DONE — its shape (asyncio.shield around
  commit/rollback, suppress widened to BaseException, commit 8d343f1976) is a
  *transaction-cleanup* convention, deliberately different from the read-path
  re-raise convention here. Do not re-fix; do not "align" it to the decorator.
- Stage→recon_pool map: `services/memory_service.py:99-102`
  (`_CYCLE_SUMMARY_STAGE_TO_RECON_POOL`) is a documented intentional duplicate
  of `_STAGE1_CYCLE_SUMMARY_RECON_POOL`
  (`reconciliation/stages/memory_consolidator.py:62`) and
  `_STAGE2_CYCLE_SUMMARY_RECON_POOL`
  (`reconciliation/stages/task_knowledge_sync.py:1127`), held in sync only by a
  drift-guard test (`tests/test_memory_service.py:4383-4397`). The blocking
  circular import that forced the duplication (documented at
  memory_service.py:85-98) runs through `reconciliation/stages/*` →
  `services/live_workflow_detector` → `services/__init__` → memory_service. A
  **leaf** module under `fused_memory/reconciliation/` breaks the cycle:
  `reconciliation/__init__.py` has no imports (verified — one constant only),
  so `from fused_memory.reconciliation.recon_pool_map import ...` is safe from
  both directions.

## Consumers (G1)

- `gather_or_raise` / `gather_collect` helpers → consumed by the 5 migrated
  idiom sites (β) and the ~15 recon-stage swallow sites (γ); the gather-site
  guard test is the enforcement consumer.
- `mcp_tool_errors` decorator → consumed by all 49 `@mcp.tool()` handlers in
  `server/tools.py`; the walker test is the enforcement consumer. (The
  recon_report server's tools are a possible later adopter — out of scope here,
  see Out of scope.)
- `recon_pool_map` module → consumed by `memory_service._infer_recon_pool` /
  `_looks_like_cycle_summary` and by both recon stages' pool-cap/trim paths.

No orphan mechanisms: every helper lands in the same batch as its call-site
migrations.

## Resolved design decisions

1. **Helper home = `fused_memory.utils.async_utils`, not `shared/`.** Checked
   per brief: the orchestrator's `gather(return_exceptions=True)` sites are
   different idioms — teardown-drain (`merge_queue.py:6379+`, `harness.py:1647`,
   deliberately awaiting cancelled tasks) and retry-substitution
   (`workflow.py:5033`) — and the dashboard already has its own
   `safe_gather_result` (`dashboard/data/utils.py:13`), which is M3's seam.
   With no same-idiom consumer outside fused-memory, a `shared/` home would
   have no named consumer (G1). Decision #4 (duplication doctrine) sanctions a
   later promotion to `shared/` if a real second consumer appears.
2. **Two helpers, not one**: `gather_or_raise(coros, *, label, logger)` —
   Pass 1, then log each Exception at WARNING and raise the first (get_entity
   semantics); `gather_collect(coros)` — Pass 1, then return the per-item
   value-or-Exception list for caller accumulation (rebuild / context_assembler
   / best-effort-sweep semantics). Pass-2 policy is thereby chosen by *name* at
   the call site instead of re-derived by hand.
3. **Best-effort sweeps keep per-item Exception tolerance but stop tolerating
   cancellation.** Migrating a `isinstance(r, BaseException)` log-and-continue
   sweep to `gather_collect` narrows the per-item guard to `Exception` and lets
   captured CancelledError propagate — the intended behaviour change, matching
   the convention set by tasks 484/512/516.
4. **Decorator owns the generic tail.** `mcp_tool_errors(operation=None)`
   (new small module `fused_memory/server/tool_errors.py`, importable without
   the 4k-line tools.py) wraps a handler: re-raises
   `(asyncio.CancelledError, KeyboardInterrupt, SystemExit)`, converts any other
   `Exception` to `{'error': str(e), 'error_type': type(e).__name__}` with
   `logger.exception`, and stamps a marker attribute (e.g.
   `__mcp_tool_errors__ = True`) on the wrapper for the walker test. Applied
   **between** `@mcp.tool()` and the function.
5. **Uniform application, specialized excepts stay.** All 49 handlers get the
   decorator; the 42 hand-copied guard+generic tails are removed; handler-local
   *specialized* except branches (e.g. `NodeNotFoundError`, ValueError→
   ValidationError shapes) remain inside the body. The 3 currently-propagating
   handlers gain the decorator too — behaviour delta is an improvement in error
   fidelity only (today FastMCP wraps their escapes as `ToolError` and the
   central wrapper returns `error_type='ToolError'`; with the decorator the
   true exception class is reported; shape unchanged).
6. **Central wrapper stays.** `_install_safe_tool_wrapper` remains as
   defence-in-depth (it also covers the recon_report server and BaseException
   groups); the decorator is the per-handler convention layer beneath it.
7. **Walker test asserts the property, not a count** (G6): construct
   `create_mcp_server(AsyncMock())` (existing pattern:
   `tests/test_tool_safe_wrapper.py:33`), iterate
   `server._tool_manager` tools, assert every registered tool's fn carries the
   decorator marker. No hardcoded "49"/"4" anywhere in signals.
8. **Gather-site guard = source-walk with explicit allowlist.** A test walks
   `fused_memory/` sources (AST) for `asyncio.gather(..., return_exceptions=True)`
   calls and fails unless each site is (a) inside `utils/async_utils.py`, or
   (b) in an explicit allowlist of documented deliberate-drain sites (teardown
   paths in `services/durable_queue.py:166`, `middleware/task_interceptor.py:525`,
   `reconciliation/harness.py:1575`, `server/main.py:1034`, and any similar
   drain-only awaits), or (c) converted to the helpers. New raw sites must
   either use the helpers or be allowlisted with justification — same
   grep-guard pattern as M1.
9. **recon_pool map home = `fused_memory/reconciliation/recon_pool_map.py`**
   (leaf module, no fused_memory imports): defines
   `STAGE1_CYCLE_SUMMARY_RECON_POOL`, `STAGE2_CYCLE_SUMMARY_RECON_POOL`, and
   `CYCLE_SUMMARY_STAGE_TO_RECON_POOL`. memory_service and both stages import
   it; the existing drift-guard test is retained but tightened to assert the
   three references are the *same objects* (regression check during migration).
   Prompt-template strings in `reconciliation/prompts/stage{1,2}.py` are NOT
   touched (W5 owns prompt rendering).

## Pre-conditions

None upstream. All assumed substrate verified on main 2026-07-06 (see
Background and the capability manifest beside this PRD).

## Cross-PRD relationship (G4)

| Other stream | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| W6 fm-memory-identity | shares files | `memory_service.py`, `graphiti_client.py` (β edits idiom only, never identity/merge logic) | W6 owns identity; M5 owns the gather idiom | file locks serialize |
| W5 recon-reliability | shares files | `task_knowledge_sync.py`, `flag_dedup.py`, `summary_pool.py`, `stage1_stall_detector.py` (γ edits gather idiom only, never recon logic/prompts) | W5 owns recon logic + prompts; M5 owns the gather idiom | file locks serialize; W5 unfiled at authoring time |
| M3 dashboard-alignment | none (checked) | dashboard keeps its own `safe_gather_result`; M5 does not touch dashboard/ | M3 | no seam |
| W9 workflow-state-machine | none (flagged) | `workflow.py:5033` reviewer-gather has a latent captured-CancelledError drop (result neither retried nor re-raised) — noted for W9, M5 does not touch orchestrator/ | W9 | observation handed off in this PRD only |

M5 introduces no seam that any other stream must consume; nothing in the
program's G4 table is redefined here.

## Decomposition plan

Labels α–ζ; deps in parentheses. All tasks project `dark_factory`.

- **α — Add `gather_or_raise` / `gather_collect` to `utils/async_utils.py`**
  (intermediate; unlocks β, γ). Built on `propagate_cancellations`. Signal:
  `tests/test_async_utils.py` — a results batch containing a captured
  CancelledError raises CancelledError through both helpers before any Pass-2
  handling; `gather_or_raise` logs every Exception then raises the first;
  `gather_collect` returns per-item value-or-Exception with cancellation
  already propagated.
- **β (α) — Migrate the 5 hand-copied idiom sites onto the helpers**
  (intermediate; γ's guard consumes the conversion): memory_service.py
  get_entity×2 + rebuild_entity_summaries, graphiti_client.py stale-summary
  fetch, context_assembler.py assemble. Behaviour-preserving for Exceptions
  (existing tests in test_memory_service.py / test_rebuild_entity_summaries.py /
  test_context_assembler.py stay green); hand-rolled Pass-2 loops deleted.
  Signal: `get_entity` on a backend that fails one concurrent call still
  returns the standard error path (first exception raised, all logged), and a
  cancellation mid-gather propagates — same observable contract, now via the
  named helpers; no `propagate_cancellations(` call remains outside
  async_utils.py except via the helpers.
- **γ (α, β) — Convert the unguarded recon gather sites + add the gather-site
  guard test** (leaf). Convert the verified swallow sites
  (task_knowledge_sync sweeps, flag_dedup ack paths, summary_pool cap-trim,
  stage1_stall_detector count/write phases) to `gather_collect` with
  `isinstance(r, Exception)` per-item guards; add the AST guard test
  (decision 8) with the documented drain-site allowlist. Signal: guard test
  fails on any raw `gather(return_exceptions=True)` site outside
  helpers/allowlist (RED before conversion completes, GREEN after); behavioural
  test: a captured CancelledError in a sweep's results aborts the sweep
  (propagates) instead of logging "delete failed …; not counted" — operator
  observes recon runs abort cleanly on shutdown.
- **δ — `mcp_tool_errors` decorator + close the 4 convention gaps**
  (intermediate; unlocks ε). New `server/tool_errors.py` + tests; apply to
  submit_task / resolve_ticket / list_tickets / cancel_ticket, removing their
  bare generic tails. Signal: unit tests — a wrapped handler raising
  RuntimeError returns `{'error','error_type'}` and logs; raising
  CancelledError propagates; the 4 handlers' MCP error responses keep the
  exact same shape as before (observable via any MCP client).
  **Framing guard (G6): do not write a RED test claiming these handlers swallow
  cancellation today — they do not (see Background); the RED signal is the
  absence of the decorator/marker.**
- **ε (δ) — Uniform decorator application + walker test** (leaf). Apply
  `mcp_tool_errors` to all remaining `@mcp.tool()` handlers in tools.py
  (including the 3 no-tail handlers), delete the 42 hand-copied
  guard+generic-tail blocks (keep specialized except branches), add the walker
  test: build `create_mcp_server(AsyncMock())`, iterate registered tools,
  assert every fn carries the marker (property, no count). Signal: walker test
  green over the real registered server; grep shows zero
  `except (asyncio.CancelledError, KeyboardInterrupt, SystemExit)` hand-copies
  left in tools.py; live MCP error responses unchanged in shape.
- **ζ — Extract `reconciliation/recon_pool_map.py` single-source constants**
  (leaf, independent, complexity=simple). New leaf module per decision 9;
  memory_service + both stages import it; drift-guard test tightened to
  identity; import-order smoke test reproducing the documented circular-import
  entry path (`import fused_memory.reconciliation.stages.memory_consolidator`
  before touching `fused_memory.services`) stays green. Signal: the recon_pool
  tag values observable on cycle_summary writes are byte-identical
  (`stage1_cycle_summary` / `stage2_cycle_summary`; the ops prune script's
  filters still match), with the map defined exactly once.

Dependency DAG: α → β → γ; δ → ε; ζ standalone.

## Out of scope

- Recon pipeline logic, prompts, ledger/write-policy (W5) — γ touches W5's
  files for the gather idiom only.
- Memory-store identity semantics, `_resolve_or_create_entity`,
  `redirect_node_edges` (W6).
- Orchestrator and dashboard gather sites (W9's `workflow.py:5033` latent
  CancelledError drop is flagged in the G4 table above; M3 owns dashboard
  fan-out helpers).
- The recon_report MCP server's handlers (`server/recon_report.py`) — covered
  by the central safe wrapper today; adopting `mcp_tool_errors` there is a
  natural follow-up once this PRD lands, not filed here.
- Task 1151's `_txn` shield convention — done, different convention, untouched.
- `_install_safe_tool_wrapper` — stays as-is (defence-in-depth layer).

## Open questions (tactical)

1. **Exact allowlist membership for the γ guard test.** The drain-site list in
   decision 8 was enumerated by grep on 2026-07-06; the implementer should
   re-enumerate at implementation time and classify any site added since —
   the guard's allowlist carries a one-line justification per entry. Decide
   during γ.
2. **`gather_or_raise` logging signature.** Whether the label/logger are
   required kwargs or derived from the caller module is a local API choice;
   the invariant is only that every Exception is logged before the first is
   raised. Decide during α.
3. **Whether ε's walker also asserts marker presence via `mcp.list_tools()`
   vs `_tool_manager` internals** — either is fine; prefer whichever existing
   test precedent (`test_tool_safe_wrapper.py`, `test_get_entity_by_uuid.py`)
   proves stable. Decide during ε.
