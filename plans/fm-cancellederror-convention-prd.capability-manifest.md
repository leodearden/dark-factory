# Capability manifest — fm-cancellederror-convention-prd.md

Per-leaf capability→evidence bindings (mechanizing G3+G6). All evidence
re-verified against main on 2026-07-06 by the authoring session
(agent claude-prd-fm-cancellederror-convention). Line numbers are as-of that
verification; re-grep at dispatch if main has moved.

## α — gather_or_raise / gather_collect helpers (intermediate)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `propagate_cancellations` exists and is the wired Pass-1 primitive | grep: `fused-memory/src/fused_memory/utils/async_utils.py:37` (def); wired at memory_service.py:1766,1821 · graphiti_client.py:1621 · context_assembler.py:140 | PASS wired |
| Existing test file to extend | `fused-memory/tests/test_async_utils.py` exists | PASS |
| A gather results list can contain captured CancelledError values | Python semantics of `gather(return_exceptions=True)` (BaseExceptions captured as values); documented in async_utils.py module docstring lines 7-30 | PASS |

## β — migrate 5 hand-copied idiom sites (intermediate)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| The 5 sites exist with the hand-rolled Pass-2 | grep: memory_service.py:1755-1777 (get_entity exact), :1795-1832 (get_entity fuzzy), :2639-2668 (rebuild_entity_summaries), graphiti_client.py:1616-1632, context_assembler.py:131-150 | PASS |
| Behaviour tests exist to hold semantics | `fused-memory/tests/test_rebuild_entity_summaries.py`, `tests/test_context_assembler.py`, `tests/test_memory_service.py` all exist | PASS |
| Producer of helpers is upstream | task α is a wired `add_dependency` prerequisite of β | PASS producer:α upstream |

## γ — convert unguarded recon gather sites + guard test (LEAF)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Live swallow premise: sites degrade captured CancelledError to WARNING-and-continue | grep: `isinstance(result, BaseException)` log-and-continue at task_knowledge_sync.py:1234-1242, flag_dedup.py:2258-2266, summary_pool.py:121-129, stage1_stall_detector.py:168-177 and :199-206 — none call propagate_cancellations (grep confirms only memory_service/graphiti_client/context_assembler import it) | PASS (premise TRUE — verified 2026-07-06) |
| Helpers producible | producer: tasks α (helpers) and β (conversion exemplars) upstream in dep closure | PASS producer upstream |
| Deliberate-drain sites exist for the allowlist (guard must not over-fire) | grep: durable_queue.py:166, task_interceptor.py:525, harness.py:1575, main.py:1032-1034 (comment documents drain intent) | PASS |
| Rejection mechanism (guard test fails on raw site) | The guard is the new rejection mechanism delivered BY this leaf; RED state = any unconverted site outside the allowlist, demonstrable by running the new test before conversion completes | PASS (self-delivered, anti-inversion OK: nothing downstream) |

## δ — mcp_tool_errors decorator + 4 gap closures (intermediate)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Exactly-4-gaps premise (re-verified, property not count in signals) | AST walk 2026-07-06: 49 `@mcp.tool()` handlers in server/tools.py; broad-tail-without-guard = submit_task (tail :2826), resolve_ticket (:2888), list_tickets (:2940), cancel_ticket (:3025) | PASS |
| **Negative-assertion check (G6 branch 4)**: "handlers swallow cancellation today" | REJECTED as a signal — authored the check: `except Exception` does NOT catch CancelledError on CPython 3.13 (empirically run), and `_install_safe_tool_wrapper` (server/main.py:1475-1521) re-raises bare CancelledError for all tools. Signals were rewritten to structural presence (decorator/marker), never behavioral swallow | PASS (false premise excluded from signals) |
| Standard error-dict shape to preserve | grep: `{'error': str(e), 'error_type': type(e).__name__}` — the uniform tail shape across tools.py (e.g. :2828, :2890, :2942, :3027, set_task_status tail) | PASS |

## ε — uniform application + walker test (LEAF)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Server constructible in tests without backends | grep: `tests/test_tool_safe_wrapper.py:33-38` — `create_mcp_server(AsyncMock())` + `server._tool_manager.add_tool` precedent | PASS wired |
| Registered tools walkable | grep: `tests/test_get_entity_by_uuid.py:93` — `[t.name for t in await mcp_server.list_tools()]`; `_tool_manager.call_tool` used across tool tests | PASS wired |
| Decorator producible | producer: task δ upstream (wired dep) | PASS producer:δ upstream |
| Central wrapper unaffected | `_install_safe_tool_wrapper` idempotent wrap at ToolManager.call_tool (server/main.py:1499-1503); decorator sits below it — no interaction beyond re-raise ordering, covered by existing test_tool_safe_wrapper.py | PASS |
| Rejection mechanism (walker fails on unwrapped handler) | Delivered BY this leaf: walker asserts marker on every registered tool; RED demonstrable by leaving any handler unwrapped | PASS (self-delivered) |

## ζ — recon_pool_map leaf module (LEAF, complexity=simple)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Duplicate map + constants exist as described | grep: memory_service.py:99-102 (`_CYCLE_SUMMARY_STAGE_TO_RECON_POOL`), memory_consolidator.py:62, task_knowledge_sync.py:1127 | PASS |
| Circular-import constraint is real but bypassable via a leaf module | memory_service.py:85-98 documents the cycle (stages → live_workflow_detector → services/__init__ → memory_service); `fused_memory/reconciliation/__init__.py` verified import-free (single float constant) → `reconciliation.recon_pool_map` import executes no service code | PASS |
| Drift-guard test exists to retain/tighten | grep: tests/test_memory_service.py:4383-4397 (equality assertions), tests/test_stages.py:13940-13943 | PASS |
| Downstream string consumers keep matching | summary_pool.py filters on `recon_pool` values; scripts prune path documented at memory_service.py:77-80 — values are unchanged by extraction (byte-identical strings) | PASS |

No FAIL bindings. Batch clear to file.
