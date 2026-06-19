# Capability manifest — external-dep-gate-flat-statuses-fix-prd

Mechanizes G3 (substrate exists + wired) and G6 (premise valid) per leaf. Evidence verified
against `main` on 2026-06-19. All bindings PASS — no FAIL → batch is clear to queue.

## Task A — Consumer reads the flat `{dep: status}` shape + two-way seam test

| Capability asserted by the signal | Evidence (on main) | Verdict |
|---|---|---|
| The tool `get_external_statuses` returns a **flat** `{dep: status}` dict (the thing to parse) | `grep:fused-memory/src/fused_memory/server/tools.py:2231` — `return result` where `result: dict[str,str]`; live direct call returned `{"dark_factory:1846":"done"}` (flat, no wrapper) | PASS (producer present + wired on main) |
| A real `done` external dep is resolvable to `'done'` (the premise the seam test asserts — G6 branch 3, end-to-end) | `get_external_statuses(["dark_factory:1846"])` → `{"dark_factory:1846":"done"}`; `dark_factory:1846` status = `done` (`done_provenance.commit 1cfdcb70`) | PASS (capability in this task's own reach; no downstream dep) |
| Envelope parser to build on (`parse_tool_result` / `EnvelopeShape`) | `grep:shared/src/shared/mcp_envelope.py:117` (`def parse_tool_result`), `:180-181` (`KEY_ABSENT`) | PASS (declared + used by scheduler today) |
| The `ExternalResolverError` error-slot + `missing`-dep guard to preserve | `grep:orchestrator/src/orchestrator/scheduler.py:1545,1551-1560` (the 1799 semantics) | PASS (wired on main) |
| The drifted mocks the task must correct (G6 branch 4 — RED proves the real shape currently fails) | `grep:orchestrator/tests/test_cross_project_dispatch_integration.py:195` (`json.dumps({'statuses': statuses})`); `grep:orchestrator/tests/test_scheduler.py:4062,4082,4154` | PASS (targets exist; wrapped shape confirmed) |
| Dispatch actually fires when the gate is satisfied (end-to-end observable) | `_deps_satisfied` external gate `grep:scheduler.py:1973-1996`; gate consumed in `acquire_next` per-tick | PASS (consuming entry path on main) |

## Task B — Escalate persistent `resolver_degraded` holds (depends on A, upstream)

| Capability asserted by the signal | Evidence (on main) | Verdict |
|---|---|---|
| An escalation pathway that files to a human exists and fires from the external-dep policy | `grep:orchestrator/src/orchestrator/scheduler.py:1705-1746` — `_on_external_dep_block(...)` → `_mark_blocked(escalate_to_human=True)` (already used by the sentinel cause) | PASS (wired into the policy on main) |
| The per-task hold-streak counter to parallel for the new cause | `grep:orchestrator/src/orchestrator/scheduler.py:1574-1613` (`_note_external_hold`; `:1589` explicitly does NOT touch `_external_unresolved_counts`) | PASS (declared on main) |
| A threshold config knob to gate escalation | `grep:orchestrator/src/orchestrator/scheduler.py:1651,2645,2708` (`max_external_dep_unresolved_cycles`) | PASS (config field exists) |
| Escalation-on-N-ticks is achievable (G6 branch 1 — threshold is a defensible config, not a guessed numeric floor) | Reuses the existing sentinel grace-counter mechanism; N = existing config value (or sibling) — no novel numeric-accuracy claim | PASS (no guessed bound) |

**DAG-direction note (anti-inversion):** Task A is **upstream** of Task B (B `depends_on` A).
Every capability B relies on is delivered by A or already on main — none is owned by a task that
depends on B. PASS.
