# S15 — fused-memory MCP server boundary (PRD)

Program `review-all-dark_factory-20261008` (`plans/review-all-program-dark_factory-2026-10-10.md`), stream S15,
ruling **R14** (2026-10-10). Size L, risk medium, wave 3. Status: AUTHORED (agent seat, 2026-11-10); critique
folded 2026-11-11; the rulings of the sitting of 2026-11-12 folded 2026-11-12 (§Rulings) and fold-verified the
same day (main `0f21f5e457`, no plan path changed since `7848173a7d`); aligned with the sibling drafts and the
later lead rulings 2026-11-13; not decomposed. No design question remains
open.

**Code anchors** verified against main `7848173a7d` (2026-11-12). Since the previous anchor `ba17934861`
(2026-11-11), `server/tools.py`, `server/main.py`, `config/schema.py`, `middleware/task_interceptor.py`,
`services/memory_service.py`, the unit template, the unit-parity script and `shared/tests/loop_blocking_allowlist.py`
changed; every symbol below was re-read and every count re-measured at `7848173a7d`. Main moves fast —
cite-by-symbol; re-locate lines at implementation time. Every count below is `measured at 7848173a7d` unless it
names another SHA.

## Goal

The next change to fused-memory's MCP surface — a new tool, a new write guard, a new override flag, a new
shutdown step, a new collaborator — is made by an agent. Today that agent has to read sibling tools to learn
which prologue applies, which guards a sibling skips, which flags to strip, which sum to update and which setter
arms a feature (`docs/code-quality.md`: quality is the cost and risk of the next change). After this PRD:

- a tool **declares** its project scope; the dispatch stack parses it once; a tool that takes `project_id` or
  `project_root` without a declaration fails the server build (R14);
- every tool family is a module that imports a `ToolContext` downward, and `server/tools.py::create_mcp_server`
  is a composition of family registrars, not a 10,067-line closure;
- `add_memory`'s and `submit_task`'s guards are ordered gate lists, and each sibling write entry point declares
  which gates it runs;
- write-time control flags are one registry, stripped at one point;
- the shutdown is declared phases of concurrent closes; the force-exit budget is the sum of each phase's largest
  timeout, and the budget validator, the schema check and the tests compute it rather than restate it; the room
  that frees inside systemd's unchanged 90 s stop timeout pays for an HTTP drain sized from measured tool-call
  durations, so a tool call running across a deploy returns its answer (with task 9493);
- `add_memory`'s and `add_system_record`'s reply carries a typed per-store error map;
- `run_server` builds collaborators and passes them to constructors through one interceptor builder;
- fused-memory publishes the read model the dashboard's recon, write-journal and curator panels need (R19; S22
  consumes it).

User-observable: a hyphen-spelled `project_id` behaves identically on every tool; five tools that accept any
`project_id` today — `get_status`, `get_queue_stats`, `get_dead_letters`, `replay_dead_letters`,
`replay_event_dead_letters` — start refusing an empty or unsafe id with `validate_project_id`'s refusal, and six
graph-changing tools — `delete_entity`, `merge_entities`, `rename_entity`, `set_entity_summary`,
`refresh_entity_summary`, `rebuild_entity_summaries` — start refusing an unregistered project (both behaviour
changes ruled 2026-11-12); `tools/list` is unchanged by the partition until the description pass; a fused-memory
without Taskmaster answers every task tool with one error code; `get_status(panels=…)` serves the dashboard
panels; an `add_memory` whose store leg failed names that store and its error in `store_errors`; with task 9493
landed, a `systemctl --user restart fused-memory` during a slow `submit_task` returns that call's answer, not a
reset, and the stop completes inside the unit's 90 s stop timeout with every declared phase logged.

## Background

- `server/tools.py` is 11,463 lines; `create_mcp_server` spans 10,067 of them and holds 67 `@mcp.tool` closures,
  31 helper closures, 28 closure-local bindings and 7 function-local imports. Largest closures:
  `consolidate_memories` 951 lines, `add_memory` 787, `submit_task` 447, `_citation_repoint_gate` 402. Tool
  docstrings — served to every MCP client as tool descriptions — total 2,815 lines / 153,146 characters
  (152,242 at `ba17934861`, 138,169 at `6551b75aba`).
- Scope prologues are hand-copied: 36 tools take `project_id`, 26 take `project_root`, 5 take neither. The 36
  `project_id` tools follow four patterns: 28 canonicalize and validate; 3 validate only (`delete_dead_letters`,
  `trigger_reconciliation`, `unhalt_reconciliation`); 1 canonicalizes only (`get_queue_stats`); 4 do neither
  (`get_status`, `get_dead_letters`, `replay_dead_letters`, `replay_event_dead_letters`). So seven never
  canonicalize and five never validate. 12 tools apply `_known_project_gate`; of the 11 graph-mutating tools, 6
  do not (`delete_entity`, `merge_entities`, `rename_entity`, `set_entity_summary`, `refresh_entity_summary`,
  `rebuild_entity_summaries`). Line counts: `_canonicalize_project_id_arg(` 32, `validate_project_id(` 32,
  `_normalize_project_root(` 24, `isinstance(_normalized, dict)` 21. `_halt_payload` re-canonicalizes to
  compensate. 24 test files (32 sites) patch `fused_memory.server.tools.resolve_main_checkout` (task 4867 added
  `tests/test_unknown_argument_guard_fused_memory.py` since `ba17934861`).
- Project-root normalization blocks the event loop on a cold cache miss: the nested sync
  `_normalize_project_root` calls `models/scope.py::resolve_main_checkout`, which runs `subprocess.run(git)`.
  `shared/tests/loop_blocking_allowlist.py` records this as one INV-8 defect (`loop-thread-occupancy-bounded`)
  in 23 `to_file` rows keyed `('fused-memory/src/fused_memory/server/tools.py', 'create_mcp_server.<tool>', …)`;
  the 23 inline call lines are `_normalize_project_root(project_root)`. Three tools already offload it
  correctly — `declare_defect`, `record_defect_outcome`, `lookup_defect_ref` run
  `await asyncio.to_thread(_normalize_project_root, project_root)`. Task **5071** (deferred) owns "offload
  fused-memory's remaining inline blocking calls", this cluster included.
- The dispatch chokepoint is `ToolManager.call_tool` of the bundled FastMCP (`mcp` 1.27.0); it runs before
  `Tool.run`'s pydantic argument validation (`fn_metadata.call_fn_with_arg_validation`).
  `server/main.py::_install_tool_dispatch_guards` installs `_install_safe_tool_wrapper` (innermost), task 4867's
  `server/unknown_argument_guard.py::install_unknown_argument_guard` (done, merged 2026-11-11 `daff96ef3c`) and
  `server/markup_guard.py::install_markup_guard` (outermost); its docstring says "Order is the whole mechanism".
  The two guards install through `server/tool_middleware_bridge.py::install_tool_middleware`, which 4867
  extracted; `server/main.py::_install_recon_report_guards` gives the recon-report server the safe wrapper and the
  unknown-argument guard. Task 4643 (pending) generalizes the guards into one chain,
  `server/boundary_guards.py::install_boundary_guards` — "ONE shim, ONE chain, not two separate wrappings of
  ToolManager.call_tool (SPOT)". All 67 tools return a dict and all carry `@mcp.tool()` + `@mcp_tool_errors()`.
  906 test lines call `mcp._tool_manager.call_tool`; 108 test files build the server through
  `create_mcp_server`, and none of them gets main.py's guard stack.
- Taskmaster-configured is encoded twice: `_taskmaster_configured = task_interceptor is not None` and a
  mid-function rebinding `task_interceptor = TaskInterceptor(None, None, _fallback_buffer)`. 8
  `type: ignore[union-attr]`; 3 error codes for "not configured" (`ConfigurationError` ×7 in tools.py,
  `RuntimeError` from `TaskInterceptor._ensure_taskmaster`, `ConfigError` ×4 in `task_interceptor.py`). No
  consumer outside fused-memory routes on any of the three (git grep over orchestrator, dashboard, shared,
  scripts, skills).
- `server/main.py` is 2,675 lines; `run_server` is 811. `_graceful_shutdown` runs 11 `_run_shielded` steps
  strictly in sequence: 9 at `_CLEANUP_STEP_TIMEOUT` (5 s), the harness at 25 s and `memory_service.close` at
  20 s — a 90 s worst case against `_FORCE_EXIT_BUDGET` = 75 s, whose comment says 70 s and whose test computes a
  literal `5 * _CLEANUP_STEP_TIMEOUT`; `_run_shielded` also waits up to a literal 1.0 s for a timed-out step's
  cancellation. The docstrings of `_graceful_shutdown` and `_shutdown_with_watchdog` still call the force-exit
  timer "45s". The 90 s `TimeoutStopSec` is restated four times: the unit template, `config/schema.py`'s
  `_MIRROR_*` literals (with 75), the constant `server/main.py::_SYSTEMD_TIMEOUT_STOP_SECS` (beside
  `_FORCE_EXIT_BUDGET = 75.0`, which `_arm_force_exit` takes as its default), and
  `scripts/check_fused_memory_unit_parity.py::REQUIRED_SERVICE_DIRECTIVES` (`"TimeoutStopSec=90"`, exercised by
  `tests/scripts/test_check_fused_memory_unit_parity.py`); the operator example
  `fused-memory/fused-memory.service.example-systemd-config` carries a fifth (task 9493 edits that file; ζ leaves
  it). `ServerConfig.graceful_shutdown_timeout`'s description restates 75 s and 90 s again, and
  `tests/test_config_schema.py::TestServerConfigGracefulShutdownTimeout` and
  `::TestServerConfigGracefulShutdownTimeoutBudgetValidation` pin the 10 s default and the 75 s arithmetic
  (30 and 15 refused, 14 accepted). Before the force-exit timer is armed, `run_server`'s `finally` stops the
  recon-report state (`stop_reaper`, `stop_persistence`), closes `topic_cluster_store`, shuts down the two
  uvicorn servers (bounded by `timeout_graceful_shutdown`), stops the watchdog thread (2 s join) and cancels
  seven loop tasks (janitor, checkpoint, rebuild_summaries, write_ops_prune, mem0_retry, remint_sweep, loop_lag).
  `MemoryService._bounded_closes`/`close_budget_secs` is the in-tree precedent for a declared (label, closer,
  timeout) tuple summed by its own budget, and
  `tests/test_server_shutdown.py::TestShutdownBudgetArithmetic::test_memory_close_step_budget_dominates_bounded_close`
  pins that the `memory_service.close` step dominates it.
- A restart today cuts the tool calls running at that moment. Measured 2026-11-12 and recorded in task **9493**
  (filed 2026-11-12, high; done, merged `1d0ae21e44`): the server receives two SIGTERMs within 0.2 s on every restart sampled
  (40 of 40), and `server/main.py::_make_operator_stop_callback` treats the second as force-exit, so uvicorn
  skips its connection wait and the 10 s `ServerConfig.graceful_shutdown_timeout` drain never takes effect.
  9493 delivers one stop signal and keeps the drain at 10 s, because today's sum leaves no room
  (10 s drain + 2 s join + 75 s budget = 87 s of 90 s). `fused-memory.socket` holds the ports across a restart,
  so new requests queue and are answered by the new process.
- `run_server` wires 13 distinct collaborators after construction: `MemoryService` five setters
  (`set_write_journal`, `set_known_projects`, `set_recon_journal`, `set_recon_ledger`, `set_event_buffer`) and
  `.taskmaster =`; `recon_journal.set_write_journal`; `targeted.planned_episode_registry =` and
  `targeted.task_interceptor =`; the interceptor's two setters and two `_wire_*` helpers. Two `in locals()`
  probes guard optional builds.
- 14 distinct `server.main` private names are patched by dotted path or `setattr` in tests;
  `_operator_stop_received` alone on 18 lines of 2 test files (8 in `tests/server/test_asgi_exception_shield.py`),
  and the watchdog and drain globals on 48 lines of `tests/test_watchdog_thread.py`, 13 of
  `tests/test_drain_signal_handler.py` and 2 of `tests/test_event_loop_lag.py` (those three files unchanged
  since `ba17934861`).
- `models/memory.py::AddMemoryResponse` has `memory_ids`, `stores_written`, `stores_pending_retry`, `category`,
  `message` and `classification_fallback`; S26 β (task 9147, pending) adds `stores_queued`. No error field exists.
  Two producers append each failed leg's error text to `message`: `services/memory_service.py::MemoryService.add_memory`
  (`[graphiti_error: …]`, `[mem0_error: …]`, the Mem0 empty-result text) and `MemoryService.add_system_record`
  (`[mem0_error: …]` or the empty-result text); the other two `AddMemoryResponse(` calls, in
  `reconciliation/flag_dedup.py`, return an empty reply. The write journal keeps only the first failing store's
  error (`error=_graphiti_error or _mem0_error or _mem0_empty_result`). The MCP tool passes the model through as
  `result.model_dump()`.

## Findings disposition

Contract §8 step 1 (`find_tasks_by_metadata(key="x_finding_key", value=<key>)`) re-run 2026-11-12 for all 12
keys: no task carries any of them. Step 2 (`search_tasks` ≥ 0.6, confirmed by `get_task`) re-run 2026-11-12 for
each leaf's remedy, σ and the changed ζ included: no live duplicate; task 9493 is the separate double-signal fix
the sitting created (ζ depends on it); adjacent live tasks are in §Cross-PRD relationship (task 5071 among them).

| Key | Sev | Disposition | Note (re-read at 7848173a7d) |
|---|---|---|---|
| fk-a9f21093207e | medium | leaf β (α0, β0 prerequisites) | still present: 7 of 36 `project_id` tools skip canonicalization and 5 skip validation; 6 of 11 graph mutators ungated |
| fk-d085b903f01c | medium | leaf ζ | still present: 11 sequential steps, 90 s worst case vs 75 s budget; comment says 70 s, two docstrings say 45 s; 90 s restated four times |
| fk-db357b64fcae | medium | leaf η (with S26 γ 9158, S10 μ2 9280, S12 κ1) | 13 wirings measured: S26 γ 9158 removes four `MemoryService` setters and `.taskmaster =`; S10 μ2 9280 the four interceptor wirings; S12 κ1 the two `targeted.` back-wires (S12 C6); η the journal setter, the two interceptor arms and the 2 `in locals()` probes. `memory_service.set_event_buffer` is the one wiring that stays: S26 C3 keeps it as the one pre-start setter because `EventBuffer.queue_stats_fn` needs the service's queue (S26 D4) |
| fk-22ab8d1f3a42 | low | leaf α (tools.py) + leaf λ (interceptor raise) | still present: rebinding, 8 `type: ignore[union-attr]`, 3 error codes |
| fk-1cadfcc36cfa | low (weakened, re-verified G6) | leaf δ | the "fifth flag" is no longer hypothetical: task 4643 adds `allow_uuid_prefix`, stripped "exactly as allow_mcp_markup is", at the same per-tool sites (6 `strip_markup_override(` calls) |
| fk-1509bcd45e2a | low (weakened, re-verified G6) | leaf α | harm half refuted (search rows are the write-after-miss metric's sole source); duplication remains: 4 memory-read try/journal blocks plus 6 task-read journal calls; R14 names the decorator |
| fk-45cf67bff540 | medium | leaf ε5 (ε1–ε4 contribute) | still present: one closure, 67 tools |
| fk-330a60eef271 | high | leaf γ (δ contributes) | still present: about 20 inline gates in `add_memory` (task 6956 added a completion-record triage input on 2026-11-12); siblings apply different subsets |
| fk-9214f842ca92 | medium | leaf ε5 | still present: 11,463 lines; 88% one function |
| fk-009fbbf855e7 | medium | leaf ι | grown: 153,146 description characters; `get_statuses` still carries "(Deliberately no line numbers above …)" |
| fk-b3d95cdaa0d3 | medium | leaf θ1 (θ2 contributes) | still present: process lifecycle as module globals; 14 patched private names |
| fk-05a87fd1777d | medium | leaf σ (closes); stamped onto 9147 | ruled 2026-11-12 (OQ-5 option B): 7650 (done) shipped `Mem0WriteState`; S26 β 9147 adds `stores_queued`; the per-leg error text is still only prose in `message` — σ makes it the typed `AddMemoryResponse.store_errors`. No acceptance record |

## Pre-conditions for activating

Upstream tasks (edge X → Y: X depends on Y, Y lands first):

- Wave-2 per-file chains end at S10 μ2 **9280** on both `server/tools.py` (S26 λ 9194, μ 9203, ρ 9243, S10 α 9180,
  γ 9195, μ2 9280) and `server/main.py` (S26 γ 9158, S10 κ1 9248, μ2 9280). Every S15 leaf declaring either
  file comes after 9280 through κ (R54's per-file rule extended to wave 3). 9280's transitive closure (store
  `dependencies`, read 2026-11-12) holds 6718, 6719, 6721 and 9158 and 50 open prerequisites, none of them 8528
  or 5040; 4867 is done. S15 adds no explicit edge onto them; the lead may add them for legibility.
- λ → S10 ν **9282** (last wave-2 declarer of `middleware/task_interceptor.py`).
- β0 → task **4643** (`install_boundary_guards`, one chain); β → task **6815** (`_to_underscore_canonical`
  alias); θ1 → task **8804** (extract `server/background_loops.py`); η → task **5462** (one `resolve_data_dir`).
  All four are pending with no open prerequisite (read 2026-11-12).
- ζ → task **9493** (one stop signal; filed 2026-11-12 by the lead outside this plan, high; done, merged
  `1d0ae21e44`, read 2026-11-13, so the edge is already satisfied). β0 and θ1, the later `server/main.py`
  leaves, follow it through ζ. κ also edits `server/main.py` and takes no edge onto 9493: the lead's ruling set
  9493's place on that file by the file locks.
- σ → S26 φ **9252** (last S26 declarer of `services/memory_service.py`; its closure holds 9147, measured) and
  σ → S26 β **9147** (`stores_queued`; named for legibility).
- Wave-3 siblings (§Cross-PRD relationship): κ → S19 α; η → S12 κ1 (C6); ε4 → S1 θ1a; ε5 → S19 δ; ε1 → S12 δ
  and δ → S12 δ (OQ-11 ruled (b), as S12/OQ-11 (a1)). None of S19 α, S19 δ, S1 θ1a, S12 δ or S12 κ1 waits on
  8528 or 5040 (store paths from each one's numeric deps, measured 2026-11-12: 9181, 4809, 5270, 9254, 8144,
  9282 and 7876 reach neither). S16 β2 does, so ε1 takes no edge onto it; S16's draft takes S16 β2 → ε1 instead. S12 owns S12 δ → α (its
  identity task edits the helpers at their new home). Residual risk the lead named: S12 α waits on task 7876
  (pending, low, no prerequisites); if 7876 stalls, ε1–ω stall with S12's chain to its δ.
- Merge-lane parking: no S15 leaf declares `orchestrator/src/orchestrator/merge_lane/worker.py` or
  `reconciliation/targeted.py`. η → S12 κ1, which is not behind 8528 or 5040; the S12/S13 cycle that once blocked
  wiring that edge is cut by the lead's ruling S13/OQ-6 (a). λ → 9282 is not behind 8528.

Substrate verified (G3): `ToolManager.call_tool(name, arguments, context, convert_result)`,
`ToolManager.list_tools()`, `Tool.fn_metadata.convert_result` and `Tool.parameters` exist in the installed
`mcp` 1.27.0; `uvicorn/server.py::Server._wait_tasks_to_complete` (called by `Server.shutdown`) waits for open
connections only `while … not self.force_exit`;
`fused_memory.utils.validation.canonicalize_project_id`, `validate_known_project_id` (permissive on `{}`),
`models/scope.py::resolve_main_checkout`, `MemoryService.close_budget_secs`, `TriageFailOpenCounter`,
`server/tool_middleware_bridge.py::install_tool_middleware` exist on main;
`scripts/check_fused_memory_unit_parity.py` and `tests/test_systemd_unit_config.py` exist.

## Sketch of approach

Four tracks, each serialized on its own file by real edges:

1. **tools.py track** (after 9280): κ publishes the dashboard read model (first, so S22 is not held behind the
   rest) → α0 gives project-root normalization an injectable resolver, offloads it from the event loop and
   retires 32 private patch sites → α builds `ToolContext` once (interceptor bound once, one not-configured
   code, journaled-read decorator) → β0 makes `create_mcp_server` install the whole `call_tool` guard stack →
   β declares scope per tool and parses it as the stack member inside the outer guards and ahead of the
   uuid-prefix guard → ε1…ε5 move the families verbatim into `server/tool_families/` (ε1 after S12 δ, so the
   moves carry its 17 call-site edits) → δ lands the gate protocol, the control-key registry and `submit_task`'s
   gate list → γ lands `add_memory`'s gate list and the siblings' declared gate sets → ι rewrites tool
   descriptions to the agent-facing contract.
2. **main.py track** (after 9280, κ): ζ declares the shutdown phases and lengthens the drain (after task 9493)
   → β0 (shared with track 1) → θ1 turns process lifecycle into an object → θ2 moves the HTTP/ASGI boundary → η
   builds collaborators in `server/composition.py` with one interceptor builder.
3. **interceptor tail**: λ makes `TaskInterceptor` raise the declared not-configured error once S10's
   `task_interceptor.py` chain has landed.
4. **memory-service tail**: σ adds the typed per-store error map after S26's `services/memory_service.py` chain
   (9252).

ω is the integration gate and signal leaf.

## Resolved design decisions

- **D1 — Scope is declared and parsed at dispatch (R14 option 1).** `ctx.tool(scope=…)` replaces
  `@mcp.tool() @mcp_tool_errors()`; `scope` is a required keyword (no default), so a registration without a
  declaration is a `TypeError` at server build (registration runs inside `create_mcp_server`). At build,
  `ScopeGate.verify` enumerates `ToolManager.list_tools()` and raises `ToolScopeUndeclared` for any registered
  tool whose input schema has `project_id` or `project_root` and no declaration (a tool registered with a bare
  `@mcp.tool()` included), and `ToolScopeMismatch` for a declaration whose parameter is absent from the
  schema. Two checks, construction and use (heuristic 10, redundantly).
- **D2 — `create_mcp_server` installs the whole `call_tool` stack, through one installer (ruled 2026-11-12).** β0
  moves main.py's `_install_tool_dispatch_guards`, `_install_safe_tool_wrapper` and `_install_recon_report_guards`
  into `server/boundary_guards.py` (task 4643's module) and makes `create_mcp_server` the primary server's one
  installer caller; β adds the scope parse. Outermost → innermost: 4643's markup guard, 4867's unknown-argument
  guard, the scope parse, 4643's uuid-prefix guard, the safe wrapper, the tool. Every server that has the tools
  has the stack — the 108 test files that build through `create_mcp_server` included — so tests and production
  canonicalize and guard identically. The scope parse sees post-repair, post-strip arguments, and the uuid-prefix
  guard sees the canonical project (its `project_for` reads the declared scope's value), so the stack holds one
  project parse; the scope parse contains its own defects because it sits outside the safe wrapper (C2). The
  uuid-prefix member is installed only where a prefix resolver is wired (`create_mcp_server`'s own
  `prefix_resolver` port; production passes one, C2). A server built without a project registry files nothing
  from the markup guard's residue and storm sinks (`server/markup_guard.py::_resolve_project_root` returns `None`
  without one), so the 108 test builds stay hermetic. One installer is 4643's "ONE shim, ONE chain" and R14's
  "dispatch wrapper as the one scope/markup gate". Cost: tests that pass markup or undeclared arguments on
  purpose now meet the guards; 5 test files that build through `create_mcp_server` carry markup (measured at
  `ba17934861`; β0 re-measures at its base, and adds the tests that pass undeclared arguments, now that 4867's
  guard is on main); test servers built without `prefix_resolver` are unaffected by the uuid-prefix member.
- **D3 — A scope refusal is the body's old refusal.** The gate returns the same `{'error', 'error_type'}` dict
  today's prologue returns for the same input, converted with the invoked tool's own
  `fn_metadata.convert_result` when `convert_result=True` — byte-identical for every tool whose prologue already
  runs the declared checks. The five tools that gain validation (ruled 2026-11-12) answer an empty or unsafe id
  with `validate_project_id`'s refusal (a behaviour change, listed under User-observable). Never a `ToolError`:
  all 67 tools return a dict and consumers route on `error_type` inside it. A scope argument that is absent or not
  a `str` is passed through untouched, so pydantic's own `ToolError` answers it exactly as today (the gate runs
  before `Tool.run`'s validation). Check order = today's prologue order: canonicalize → validate →
  known-project registry → backlog.
- **D4 — Dead-letter tools are declared raw, with a reason.** `get_dead_letters`, `delete_dead_letters`,
  `replay_dead_letters`, `replay_event_dead_letters` keep their raw spelling as a declaration
  (`ToolScope.raw(reason=…)`): rows written under a pre-canonicalization spelling stay addressable. Raw means
  "not canonicalized", never "not validated": a raw declaration validates (ruled 2026-11-12), so
  `delete_dead_letters` keeps its "prevents accidental cross-project deletes" validation
  (`tests/test_dead_letters_tool.py` pins it) and the three others gain it — dead-letter rows were written by
  validating writers, so no stored id fails it. `get_status`, `trigger_reconciliation` and
  `unhalt_reconciliation` become canonical, and `_halt_payload`'s re-canonicalization and
  `trigger_reconciliation`'s remedy-from-halt workaround go.
- **D5 — Every graph-mutating tool is declared `write` and so gains the known-project gate (ruled 2026-11-12).**
  The six ungated mutators refuse an unregistered project like the five gated ones; the behaviour change is
  accepted.
- **D6 — Identity stays a per-body call (lead ruling, recorded on task 6721).** `_resolve_identity` (S14 μ makes
  it return `AgentIdentity`) is not moved into the dispatch stack: a typed identity cannot cross the
  pydantic-validated arguments dict without rewriting every tool signature, and the alternative — a `ContextVar`
  the stack sets and bodies read — is implicit per-call state against heuristic 7 (*prefer stateless interactions
  between modules*). α moves it verbatim, with today's signature, into `tool_context.py` beside
  `extract_causation`; S12 δ adds the header rule to both at this home after α (S12's edge) and before ε1 (OQ-11).
- **D7 — Families are `register(mcp, ctx)` modules under `server/tool_families/`, moved verbatim (lead ruling).**
  A family imports `ToolContext`, `tool_scope`, `tool_args`, `task_status_facts`, `write_gates` and lower
  packages; never another family, never `server/tools.py`. `create_mcp_server` builds `ToolContext`, calls each
  registrar in `FAMILY_REGISTRARS` order and installs the dispatch stack. The moves change no behaviour and no
  tool description; `tools/list` is snapshotted by ε1 and compared by every later ε. "Verbatim" includes every
  per-tool identity call S12 δ left: the writes and the three reads whose `agent_id` names the reader
  (`get_entity`, `get_entity_by_uuid`, `get_episodes`) keep S12 δ's refusal of a differing value, and `search`
  keeps its exclusion (its `agent_id` is an authoring-agent filter; the header neither refuses nor replaces it;
  lead ruling on S12-2's three readings).
- **D8 — Gates are values in an ordered tuple.** A gate takes a frozen request and returns the request
  (possibly transformed with `dataclasses.replace`) or a `Refusal`; advisories (today's warnings folded into the
  ack) ride on the request. `run_gates` stops at the first refusal. Sibling entry points declare their tuple,
  with any skipped gate named in a `skips` table pinned by test (heuristic 3: entry point × policy is one table,
  not N inline if-chains). The service-level checks inside `MemoryService.add_memory` stay S26's.
- **D9 — One control-key registry.** `WRITE_TIME_CONTROL_KEYS` is built from the owning constants
  (`shared.toolcall_markup.MARKUP_OVERRIDE_KEY`, 4643's prefix-override key, the near-duplicate and
  dangling-citation keys, `_causation_id`); `split_write_controls` returns typed `WriteControls` and a payload
  stripped of every key; tools read the typed flags, never `metadata.get(...) is True`. The causation id comes
  from S12 δ's resolver (OQ-11 ruled (b)): S12 δ lands first and reads S12 C3's `X-Fused-Memory-Causation-Id`
  header; under Leo's S12-2 ruling it REFUSES a write whose `metadata['_causation_id']` differs from the header
  (structured error naming both values, one log event) and attributes a matching or missing value to the header.
  δ passes the resolver's value to `split_write_controls(metadata, header_causation_id=…)`, so the registry never
  sees a mismatch: it takes the header value when given, else `metadata['_causation_id']`, else a fresh id, and
  strips the key either way. The source comes from `AgentIdentity`, not an `agent_id` prefix.
- **D10 — The shutdown is declared phases of concurrent closes; the plan is static data in a config leaf; closers
  bind at runtime (Leo, OQ-1 option D; module home by lead ruling).** `config/shutdown_plan.py` declares the
  ordered phases; each phase is a tuple of `(name, timeout, after)` steps whose closers run concurrently under the
  phase, and each step names the steps it must follow. `FORCE_EXIT_BUDGET_SECS` is the sum of each phase's
  largest timeout plus headroom, computed by the module. A test asserts that every `after` member sits in an
  earlier phase (the dependency order pinned by test, heuristic 10). `config/schema.py` imports the module
  downward (main.py imports schema, so the plan cannot live in main.py without the cycle the `_MIRROR_*` literals
  were invented to dodge). main.py owns the sequence: it binds a closer to every declared name and runs the
  phases. This places R14's "shutdown sequence tuple (main.py owns)" seam in `config/shutdown_plan.py` (lead
  ruling; program seam row updated) and refines R14's "(name, closer, timeout) tuple summed by the budget test"
  into phases summed per phase (Leo, OQ-1). The seven loop cancels, the recon-report stop and the topic-store
  close join the phases. `TimeoutStopSec` stays 90 s in the unit template. The computed figures, from today's
  timeouts and the orderings `_graceful_shutdown`'s docstring states (C3's table): three phases, 25 + 5 + 20 =
  50 s, plus 5 s headroom = a 55 s budget (today 75 s), so the stop's worst case is drain + 2 s join + 55 s, and
  the drain may grow to anything strictly under 33 s.
- **D11 — The drain is sized from measured tool-call durations, inside the room D10 frees (lead decision,
  residual R-S15-a = B).** Task 9493 (outside this plan; landed first, done and merged `1d0ae21e44`) makes exactly one stop signal reach the
  server, so the drain takes effect, keeps it at 10 s and records the tool-call durations (p50/p90/max per tool,
  from the server log's request timestamps) in its completion note and in `tests/test_server_shutdown.py`'s
  docstring. ζ then sets `ServerConfig.graceful_shutdown_timeout` from those durations (re-measured if stale; the
  LLM-backed `submit_task` and `add_memory` decide it; for example 30 s) and the budget check covers it:
  `graceful_shutdown_timeout + PRE_ARM_BOUND_SECS + FORCE_EXIT_BUDGET_SECS < SYSTEMD_TIMEOUT_STOP_SECS`
  (30 + 2 + 55 = 87 < 90 for the example).
- **D12 — Collaborators via constructors; the reconciler through S12's provider member (ruled 2026-11-12,
  OQ-9 (a)).** `ReconciliationJournal` takes `write_journal`, and one `build_task_interceptor(…,
  reconciliation=None)` replaces the two arms. η's builder passes S12 C6's `InterceptorCollaborators(reconciler=…)`
  provider; the interceptor resolves it with itself as the write gate, and that provider callable is what
  constructs `TargetedReconciler(…, task_writes=port, planned_episode_registry=…)`. S12 κ1 deletes both
  `targeted.` back-wires (`targeted.task_interceptor =`, `targeted.planned_episode_registry =`).
  `memory_service.set_event_buffer` stays: S26 C3 keeps it as the one pre-start setter (S26 D4).
- **D13 — One not-configured code.** `McpErrorType.NOT_CONFIGURED = 'ConfigurationError'` (the spelling seven
  tool sites already emit) is added to S14's vocabulary by α; λ makes the interceptor raise it.
- **D14 — The dashboard read model is typed reply models served by `get_status(panels=…)` (lead ruling, flagged
  as a reading of R19).** This is R19's option 1 (extend the producer's status tools). Each panel's SQL moves
  into the store that owns the table's DDL; the reply models live in `shared/` so the dashboard (which has no
  `fused_memory` dependency) parses the reply into the same types fused-memory produced.
  `shared/dashboard_panels.py` holds wire reply models only — no SQL, no table or index names — so it is not
  R19's rejected option 2 (a shared schema module both packages read the database through). A panel whose store
  is absent or failing is reported `unavailable` with its reason — never as empty rows (R19, "degrade visibly").
  Derived state is computed by the owner: the burst panel serves `effective_state`, so no dashboard module
  restates fused-memory's cooldown.
- **D15 — A failed store leg is a typed field, not prose (Leo, OQ-5 option B).** `AddMemoryResponse.store_errors:
  dict[SourceStore, str]` (C7), empty by default, filled by both producers for a leg that failed: a Graphiti
  enqueue failure, a Mem0 exception and the Mem0 empty result. A leg queued for retry is not an error — it is in
  `stores_pending_retry`. A store in `store_errors` is in none of the three store lists. 9147's message renderer
  renders the errors from the field; until σ lands, 9147 is amended to keep each failed store's error text (lead
  amendment), so the reply never loses it in between.

## Decision partition and eval hermeticity (Leo's standing rules of 2026-11-12)

- **One decision partition per decision area** (invariant `one-decision-partition-per-area`, INV-23, minted by
  the scheduler plan S17; cited by its slug, not restated). The plan
  has one table with an LLM row: `ADD_MEMORY_GATES` (γ), whose write-triage gate hands only the middle
  similarity band that code cannot settle to `server/write_triage_judge.py::judge_write`; code decides every
  other band and every other gate; a judge failure fails open into a counted burst (`TriageFailOpenCounter`)
  that escalates to a human. γ moves the gates in today's order and adds no LLM row. The scope gate (C2), the
  control-key registry (C5), the sibling gate tables and the shutdown phases (C3) are code only.
- **Evals are hermetic.** Not engaged: the plan adds or moves no store, sink, filer, queue or archive an eval cell
  can reach (κ's read model is read-only; σ adds a reply field). β0 widens the guard stack to every
  `create_mcp_server` build; a build without a project registry files nothing (D2, boundary row 21b).

## Contracts

S15 owns C1–C7; a dependent stream holds a leaf on `S15/C<n>`.

```python
# C1 — fused-memory/src/fused_memory/server/tool_context.py (α)        every tool family consumes
@dataclass(frozen=True)
class ToolContext:
    memory_service: MemoryService
    interceptor: TaskInterceptor                  # real, or the null interceptor (S10 μ2's null bundle); never None
    taskmaster_configured: bool                   # derived once from the binding above
    write_journal: WriteJournal | None
    reconciliation_harness: ReconciliationHarness | None
    backlog_policy: BacklogPolicy | None
    event_queue: EventQueue | None
    curator_usage_gate: UsageGate | None
    known_projects: Mapping[str, str]             # {} when unwired; never None
    topic_cluster_store: TopicClusterStore | None
    id_resolver: IdResolver | None
    checkout_resolver: Callable[[str], str]       # α0; production default models.scope.resolve_main_checkout
    panels: DashboardPanelReader | None           # κ
    triage_fail_open_counter: TriageFailOpenCounter   # one per server (INV-4)
    scopes: Mapping[str, ToolScope]               # filled by ctx.tool, read by ScopeGate (β)
    @classmethod
    def build(cls, memory_service, task_interceptor=None, write_journal=None, **optional) -> 'ToolContext'
    def tool(self, *, scope: ToolScope) -> Callable[[ToolFn], ToolFn]   # β: mcp.tool + mcp_tool_errors + record
    def not_configured(self, backend: str) -> dict          # error_envelope with McpErrorType.NOT_CONFIGURED
    def halt_payload(self, project_id: str | None) -> dict | None
    def journaled_read(self, operation: str, *, summarize: Callable[[Any], dict]) -> Callable[[ToolFn], ToolFn]
    async def normalize_project_root(self, project_root: str) -> str | dict   # β: get_external_statuses' per-reference call; asyncio.to_thread (α0's offload)
def resolve_identity(agent_id, session_id, ctx) -> tuple[...]   # moved verbatim from tools.py::_resolve_identity (today's signature; S14 μ's return type)
def extract_causation(metadata, agent_id) -> tuple[str, str, dict | None]   # moved verbatim from tools.py::_extract_causation
# S12 δ (after α, before ε1; OQ-11 ruled (b)) adds the `ctx` parameter and the header rule to both at this home,
# refusing a differing agent_id or _causation_id (S12-2 ruling) on writes and on get_entity, get_entity_by_uuid and
# get_episodes, never on search's agent_id filter (lead ruling; the ε moves carry it verbatim); δ then replaces
# extract_causation with C5's parse
```
Invariants: one `ToolContext` per `create_mcp_server` call; no tool body reads a collaborator that is not a
`ToolContext` member; `journaled_read` writes one `kind='read'` row on success and one `success=False` row then
re-raises on failure (audit-then-raise; the `search` row shape is unchanged because task 3213's metric reads it);
`taskmaster_configured` is False exactly when the null interceptor is bound; α moves the identity and causation
helpers verbatim, with today's signatures and behaviour. Errors: `build` raises `TypeError` for a missing
required collaborator; nothing else raises at build.

```python
# C2 — fused-memory/src/fused_memory/server/tool_scope.py (β)          every tool family declares; boundary_guards installs (β0, β)
class ScopeKind(StrEnum): PROJECT_ID = 'project_id'; PROJECT_ROOT = 'project_root'; NONE = 'none'
@dataclass(frozen=True)
class ToolScope:
    kind: ScopeKind
    param: str | None
    optional: bool = False                 # None passes through (global scope, e.g. get_queue_stats)
    canonical: bool = True                 # False only through ToolScope.raw(reason=...)
    raw_reason: str | None = None
    validate: bool = True                  # validate_project_id; no factory sets it False, raw included (ruled 2026-11-12)
    require_known: bool = False            # the known_projects registry gate
    backlog: bool = False                  # the per-project backlog gate (memory-content writes)
    @classmethod
    def read(cls, *, optional: bool = False) -> 'ToolScope'
    @classmethod
    def write(cls, *, backlog: bool = False) -> 'ToolScope'          # require_known=True
    @classmethod
    def raw(cls, *, reason: str, optional: bool = True) -> 'ToolScope'   # canonical=False, validate=True
    @classmethod
    def task_root(cls) -> 'ToolScope'                                # project_root → main checkout
    UNSCOPED: ClassVar['ToolScope']
class ToolScopeUndeclared(Exception): ...   # message names the tool and the scope-shaped parameter
class ToolScopeMismatch(Exception): ...     # message names the tool, the declared param and the schema's params
class ScopeGate:
    def __init__(self, *, known_projects: Mapping[str, str], backlog_policy: BacklogPolicy | None,
                 checkout_resolver: Callable[[str], str]) -> None
    def verify(self, mcp: FastMCP, scopes: Mapping[str, ToolScope]) -> None   # at server build (D1)
    async def parse(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any] | ScopeRefusal
@dataclass(frozen=True)
class ScopeRefusal: payload: dict          # today's prologue dict for the same input
# fused-memory/src/fused_memory/server/boundary_guards.py (task 4643's module; β0 moves the installers here, β adds the member)
def install_boundary_guards(mcp: FastMCP, *, scope_gate: ScopeGate | None, known_projects: Mapping[str, str],
                            prefix_resolver: PrefixResolver | None) -> None
    # outermost → innermost: markup guard, unknown-argument guard, scope parse, uuid-prefix guard, safe wrapper, tool; idempotent
    # PrefixResolver is shared/src/shared/uuid_prefix_guard.py::PrefixResolver
def install_recon_report_guards(mcp: FastMCP) -> None   # β0: today's main.py::_install_recon_report_guards, moved; safe wrapper inside, unknown-argument guard outside
```
Invariants: every registered tool has exactly one declaration; a tool body receives the canonical spelling of its
declared parameter (or `None` when declared optional); the arguments dict handed to the tool is a copy;
`install_boundary_guards` is idempotent (sentinel attribute, as the existing wrappers) and has one production
caller, `create_mcp_server`; the uuid-prefix member is installed exactly when `prefix_resolver` is given —
`create_mcp_server(…, prefix_resolver: PrefixResolver | None = None)` is the guard's own port, never
`ToolContext.id_resolver`; `run_server` passes a per-use resolver, `lambda project_id, prefix:
IdResolver.for_memory_service(memory_service).resolve_prefix(project_id, prefix)`, so a hot reload of the
resolver's config still applies (as `create_mcp_server._resolver` does for the tool bodies); tests opt in by
passing one; with `None` the member is absent and the installer logs a WARNING once (INV-11); the uuid-prefix
guard sits inside the scope parse, so its `project_for` reads the declared scope's canonical value; every declared
scope validates; a scope argument that is absent or not a `str` is passed through untouched, so pydantic's own
error answers it; `parse` runs the checkout resolution as `await asyncio.to_thread(self._checkout_resolver, root)`
— the gate never calls a blocking resolver on the loop thread (INV-8). Error semantics: a refusal is returned,
never raised (D3); `ScopeGate.parse` contains its own defects (re-raises `CancelledError`; any other exception
becomes the safe wrapper's `{'error', 'error_type'}` shape), so nothing escapes to the session task group.

```python
# C3 — fused-memory/src/fused_memory/config/shutdown_plan.py (ζ)       main.py binds and runs; config/schema.py derives
@dataclass(frozen=True)
class ShutdownStep:
    name: str                       # 'background_loops.cancel_all', 'recon_report_state.stop', 'memory_service.close', ...
    timeout_secs: float
    after: frozenset[str]           # steps that must have finished first; each a member of an earlier phase
@dataclass(frozen=True)
class ShutdownPhase:
    name: str                       # 'stop_producers', 'close_consumers', 'close_stores'
    steps: tuple[ShutdownStep, ...]   # run concurrently, each under its own timeout
SHUTDOWN_PHASES: tuple[ShutdownPhase, ...]   # run in order
STEP_CANCEL_GRACE_SECS: float               # _run_shielded's wait for a timed-out step's cancellation (today a literal 1.0)
FORCE_EXIT_HEADROOM_SECS: float             # > len(SHUTDOWN_PHASES) * STEP_CANCEL_GRACE_SECS
FORCE_EXIT_BUDGET_SECS: float               # sum(max(step.timeout_secs) per phase) + FORCE_EXIT_HEADROOM_SECS, computed
WATCHDOG_JOIN_TIMEOUT_SECS: float           # the join's single home; _stop_watchdog_thread's default reads it (today a literal 2.0 there)
PRE_ARM_BOUND_SECS: float                   # the bounded waits before the force-exit arm, starting with WATCHDOG_JOIN_TIMEOUT_SECS
SYSTEMD_TIMEOUT_STOP_SECS: float            # equals scripts/fused-memory.service.template TimeoutStopSec (parsed by test)
# fused-memory/src/fused_memory/server/main.py (ζ; θ1 moves both into server/process_lifecycle.py)
class ShutdownPlanMismatch(Exception): ...
def bind_shutdown_plan(closers: Mapping[str, Callable[[], Awaitable[None] | None] | None]) -> tuple['BoundPhase', ...]
async def run_shutdown_plan(plan: tuple['BoundPhase', ...]) -> None
```
The draft phase table ζ starts from (timeouts are today's; `after` is the order `_graceful_shutdown`'s docstring
states, plus the loops' users):

| Phase | Steps (timeout s; must follow) | Phase timeout |
|---|---|---|
| `stop_producers` | `background_loops.cancel_all` (5), `recon_report_state.stop` (5), `topic_cluster_store.close` (5), `task_interceptor.drain` (5), `sqlite_watchdog.close` (5), `harness_loop_task` (25) | 25 s |
| `close_consumers` | `task_interceptor.close` (5; drain, loops — the remint sweep calls the interceptor), `curator_usage_gate.shutdown` (5; drain), `event_queue.close` (5; sqlite_watchdog), `taskmaster.close` (5; harness, drain, loops — the checkpoint loop touches its database), `recon_journal.close` (5; harness, drain, loops) | 5 s |
| `close_stores` | `curator_cost_store.close` (5; usage gate), `memory_service.close` (20; event_queue, taskmaster, harness, loops), `recon_ledger.close` (5; recon_journal) | 20 s |

Computed: 50 s of phases + 5 s headroom = `FORCE_EXIT_BUDGET_SECS` 55 s (the headroom exceeds 3 phases × 1 s
cancel grace). Today the harness is still running when the interceptor closes; the table cancels the harness
first, which is stricter than today. ζ may add an `after` edge with a recorded reason; an added edge can add a
phase, and the module recomputes the budget — no figure is restated.

Invariants: the closers mapping's keys equal the declared names exactly (a `None` closer means the component was
not built; the step is skipped and logged); phases run in declared order; within a phase every step starts
together, each under `_run_shielded` with its declared timeout, and the phase ends when all have finished or
timed out; every `after` member is a step of an earlier phase (pinned by test); `background_loops.cancel_all`
cancels the seven loop tasks and awaits them under one `gather` and one timeout; a sync closer stays on the loop
thread (no SQLite close moves to a worker thread); the force-exit timer is armed before the first phase; the two
uvicorn `server.shutdown()` calls and the watchdog-thread stop stay before the arm (bounded by
`timeout_graceful_shutdown` and the 2 s join); the `memory_service.close` step's timeout exceeds
`MemoryService.close_budget_secs()` (the dominance test is kept); `ServerConfig`'s validator and every test read
`FORCE_EXIT_BUDGET_SECS`, `PRE_ARM_BOUND_SECS` and `SYSTEMD_TIMEOUT_STOP_SECS` from this module and assert
`graceful_shutdown_timeout + PRE_ARM_BOUND_SECS + FORCE_EXIT_BUDGET_SECS < SYSTEMD_TIMEOUT_STOP_SECS`, the drain
included; the join timeout lives here, not in main.py, because this module cannot import main.py (D10's cycle);
`scripts/check_fused_memory_unit_parity.py` (bare `python3`, cannot import `fused_memory`) reads `TimeoutStopSec`
from the unit template rather than restating it. Error: `ShutdownPlanMismatch` names the undeclared closers and
the unbound steps.

```python
# C4 — shared/src/shared/dashboard_panels.py (κ)                        S22's dashboard readers consume
class PanelName(StrEnum): RECON = 'recon'; MEMORY_OPS = 'memory_ops'; CURATOR = 'curator'
class ReconPanelRequest(BaseModel): recent_runs_limit: int = 50
class MemoryOpsPanelRequest(BaseModel): start: str; end: str          # ISO-8601, [start, end)
class CuratorPanelRequest(BaseModel): since: str                       # resolved_at >= since
class ReconRun(BaseModel): id; project_id; run_type; trigger_reason; started_at; completed_at; events_processed; status
class Watermark(BaseModel): project_id; last_full_run_completed; last_episode_timestamp; last_memory_timestamp; last_task_change_timestamp
class LastAttemptedRun(BaseModel): id; status; started_at; completed_at
class BufferStats(BaseModel): buffered_count: int; oldest_buffered_at: str | None
class BurstStateRow(BaseModel): agent_id; state; effective_state; last_write_at; burst_started_at   # effective_state computed by the owner
class LatestVerdict(BaseModel): run_id; severity; action_taken; reviewed_at; is_phantom: bool   # shared.phantom_verdict classifies
class ReconPanel(BaseModel): recent_runs: list[ReconRun]; watermarks: list[Watermark]; last_attempted: dict[str, LastAttemptedRun]; buffer: BufferStats; bursts: list[BurstStateRow]; latest_verdict: LatestVerdict | None
class MemoryOpsBucket(BaseModel): hour: str; kind: Literal['read', 'write']; operation: str; count: int
class MemoryOpsPanel(BaseModel): start: str; end: str; buckets: list[MemoryOpsBucket]; read_rollup_present: bool
class TicketResolution(BaseModel): created_at: str; resolved_at: str; status: str
class CuratorPanel(BaseModel): since: str; resolutions: list[TicketResolution]; refused_count: int
class PanelUnavailable(BaseModel): panel: PanelName; reason: str
class PanelsReply(BaseModel): recon: ReconPanel | PanelUnavailable | None; memory_ops: MemoryOpsPanel | PanelUnavailable | None; curator: CuratorPanel | PanelUnavailable | None
# fused-memory: get_status(project_id=None, panels: dict[str, dict] | None = None) adds result['panels'] = PanelsReply.model_dump()
# fused-memory/src/fused_memory/server/dashboard_panels.py: DashboardPanelReader(recon_journal, event_buffer, write_journal, ticket_store).read(requests) -> PanelsReply
# the event-buffer owner: EventBuffer.burst_state_rows(*, now) -> list[BurstStateRow]   # applies the instance's burst_cooldown_seconds
```
Invariants: every column the dashboard's `data/reconciliation.py`, `data/write_journal.py` and `data/metrics.py`
readers select from fused-memory tables at κ's base is a field of a panel model (11 SQL sites over 8 tables at
`7848173a7d` — `reconciliation.py` runs 3, watermarks 1, event_buffer 2, burst_state 1, judge_verdicts 1;
`write_journal.py` write_ops 1, read_op_counts 1; `metrics.py` tickets 1 — `journal_entries` went with task 5890;
κ re-measures at its base and pins the list by test); each panel's SQL lives in the module that owns the table's
DDL — after S19 α the journal and event-buffer owners are `recon_state/journal.py` (runs, watermarks,
judge_verdicts) and `recon_state/event_buffer.py` (event_buffer, burst_state), then `services/write_journal.py`
(write_ops), `services/read_rollup.py` (read_op_counts) and `middleware/ticket_store.py` (tickets);
`INDEXED BY idx_wo_created` moves with it. The `event_buffer` and `burst_state` DDL also exists as a copy in the
journal module; κ reads through the event-buffer owner and leaves the copy to S12 or S19. `effective_state` is
`'idle'` once `last_write_at` is past the instance's `burst_cooldown_seconds`, and no dashboard module restates
the cooldown. Omitting `panels` leaves `get_status`'s reply byte-identical. Errors: a malformed panel request is a
`ValidationError` dict; an unreadable store is a `PanelUnavailable` naming the store and the exception class.

```python
# C5 — fused-memory/src/fused_memory/server/write_gates.py and write_controls.py (δ)   γ and the families consume
@dataclass(frozen=True)
class Refusal: gate: str; payload: dict                  # the wire payload today's gate returns, unchanged
@dataclass(frozen=True)
class Advisory: gate: str; key: str; value: Any          # folded into the success ack under `key`
class WriteGate(Protocol[R]):
    name: str
    async def __call__(self, request: R) -> R | Refusal
async def run_gates(gates: Sequence[WriteGate[R]], request: R) -> R | Refusal
WRITE_TIME_CONTROL_KEYS: frozenset[str]
@dataclass(frozen=True)
class WriteControls:
    allow_mcp_markup: bool; allow_near_duplicate: bool; allow_dangling_citations: bool
    allow_uuid_prefix: bool; causation_id: str | None
def split_write_controls(metadata: Mapping[str, Any] | None, *, header_causation_id: str | None = None) -> tuple[WriteControls, dict | None]
    # header_causation_id is S12 δ's resolved header value (OQ-11 (b)); a differing metadata _causation_id was refused upstream by that resolver
SUBMIT_TASK_GATES: tuple[WriteGate[SubmitTaskRequest], ...]       # δ, in tool_families/task_write.py
ADD_MEMORY_GATES: tuple[WriteGate[AddMemoryRequest], ...]         # γ, in tool_families/memory_write.py
WRITE_ENTRY_GATES: Mapping[str, tuple[str, ...]]                  # γ: entry point → gate names; skips explicit
```
Invariants: literal-`True` semantics for every flag (as today); the payload returned by `split_write_controls`
holds no key of the registry; with a header present the causation id is the header's value, and a differing
metadata `_causation_id` never reaches `split_write_controls` (S12 δ's resolver refuses it, Leo's S12-2 ruling);
a gate never mutates its input; the first refusal wins and names its gate in the server log; the order of
`ADD_MEMORY_GATES` equals today's inline order (pinned).

```python
# C6 — fused-memory/src/fused_memory/server/tool_families/<family>.py (ε1–ε5)   server/tools.py composes
def register(mcp: FastMCP, ctx: ToolContext) -> None
# server/tools.py after ε5
FAMILY_REGISTRARS: tuple[Callable[[FastMCP, ToolContext], None], ...]
def create_mcp_server(memory_service, task_interceptor=None, write_journal=None, *, ...) -> FastMCP   # signature unchanged by the ε moves (κ adds the panel reader, α0 checkout_resolver, β0 prefix_resolver)
```
Invariants: no module under `tool_families/` imports another family or `server/tools.py` (AST test); `tools/list`
(names, input schemas, descriptions) is identical before and after each ε; tool → family is the table in
§Partition measurement.

```python
# C7 — fused-memory/src/fused_memory/models/memory.py (σ), extending S26 C2's AddMemoryResponse   every add_memory / add_system_record client reads
class AddMemoryResponse(BaseModel):
    ...                                                      # S26 C2's fields, stores_queued (task 9147) included
    store_errors: dict[SourceStore, str] = Field(default_factory=dict)   # a failed leg's error text, keyed by store
```
Invariants: a store in `store_errors` is in none of `stores_written`, `stores_queued` and `stores_pending_retry`
(a model validator raises on a violation); a leg queued for retry is never an error; both producers
(`MemoryService.add_memory`, `MemoryService.add_system_record`) fill it for a Graphiti enqueue failure, a Mem0
exception and the Mem0 empty result; 9147's message renderer renders the errors from the field; the MCP tool
passes it through unchanged (`result.model_dump()`). Journalling the whole map in `result_summary` is σ's option,
not part of the contract.

## Boundary-test sketch

| # | Seam | Scenario | Preconditions | Postconditions |
|---|---|---|---|---|
| 1 | C1 | a server built with `task_interceptor=None` answers each task tool | no Taskmaster | `error_type == 'ConfigurationError'` on every task tool; no `RuntimeError`/`ConfigError` (after λ) |
| 2 | C1 | each journaled read succeeds, then fails | real `WriteJournal` over `tmp_path` | one `kind='read'` row per call; failure row `success=False`, then the exception reaches the caller |
| 3 | C2 | a tool with `project_id` registered by bare `@mcp.tool()` | test-local family | `create_mcp_server` raises `ToolScopeUndeclared` naming it |
| 4 | C2 | `'know-live'` sent to every canonical-declared tool | parametrized over `ctx.scopes` | the body receives `'know_live'`; raw-declared tools receive `'know-live'` |
| 5 | C2 | a path-shaped id through `call_tool`, `convert_result` False and True | — | the payload equals today's prologue dict (the validation refusal for the five tools that gain it); no `ToolError` |
| 5b | C2 | `project_id=123`, and a required `project_id` omitted | — | the same pydantic `ToolError` text as today; the gate passed the arguments through |
| 6 | C2 | an unregistered project on each `write`-declared tool | `known_projects` fixture | the known-project refusal, the six formerly ungated mutators included |
| 7 | C2 | `get_status('know-live')` with `know_live` halted | fake judge | `halted: True` and `halted_projects == ['know_live']` agree |
| 8 | C2 | a worktree `project_root` | injected `checkout_resolver` | the body receives the main checkout; no test patches `server.tools` |
| 8b | C2 | a `project_root` tool call with a resolver that records `threading.current_thread()` | injected recording resolver | the recorded thread is not the event-loop thread (INV-8) |
| 9 | C3 | a closer not declared, and a declared step with no closer key | — | `ShutdownPlanMismatch` names both |
| 10 | C3 | ζ's default `graceful_shutdown_timeout` + `PRE_ARM_BOUND_SECS` + `FORCE_EXIT_BUDGET_SECS` | template parsed | strictly under `SYSTEMD_TIMEOUT_STOP_SECS`, which equals the template's `TimeoutStopSec` (90) |
| 11 | C3 | every closer hangs | short declared timeouts in a test plan of three phases | each phase's steps start together; phases start in declared order; total ≤ Σ phase max timeout + phases × `STEP_CANCEL_GRACE_SECS` + ε |
| 11b | C3 | the declared `SHUTDOWN_PHASES` | — | every step's `after` members sit in an earlier phase; no step is declared twice; `FORCE_EXIT_BUDGET_SECS` equals the computed sum |
| 12 | C4 | `get_status(panels={'recon': {}})` on fixture `reconciliation.db` | known rows | fields equal the fixture; a missing table → `PanelUnavailable`, never `[]` |
| 12b | C4 | a `burst_state` row whose `last_write_at` is past the cooldown | fixture `EventBuffer(burst_cooldown_seconds=…)` | `effective_state == 'idle'` while `state` keeps the stored value |
| 13 | C4 | memory ops without `read_op_counts` | write_ops only | `read_rollup_present: False`, write buckets only |
| 14 | C5 | one refusal fixture per `ADD_MEMORY_GATES` member | gate order table | refused by exactly that gate; the order equals the table |
| 15 | C5 | each sibling entry point | `WRITE_ENTRY_GATES` | the gates run equal the declared tuple; a skip carries its reason |
| 16 | C5 | every `WRITE_TIME_CONTROL_KEYS` key to every metadata-forwarding tool, and `split_write_controls` with and without `header_causation_id` | real stores or fakes at the backend seam | the persisted payload holds none of them; a given `header_causation_id` is the causation id; with no header, `metadata['_causation_id']` is |
| 17 | C6 | `tools/list` vs the ε1 snapshot | — | identical names, schemas, descriptions (until ι updates the snapshot) |
| 18 | C6 | import graph of `server/tool_families/` | AST | no family → family or family → `server.tools` edge |
| 19 | θ1 | operator stop → exit path | `ProcessLifecycle(exit_process=fake)` | exit code 0 recorded; no test assigns a `server.main` global; 9493's two-SIGTERM drain test still passes |
| 20 | η | reconciliation enabled and disabled builds | config fixtures | one `build_task_interceptor` call each; the enabled build passes the `InterceptorCollaborators(reconciler=…)` provider; disabled arm's reconciliation members `None` by construction |
| 21 | β0 | a server from `create_mcp_server` with no main.py involvement | test build | markup in an argument is refused by the markup guard; a second install wraps nothing; the stack order, outermost to innermost, is markup, unknown-argument, scope parse, uuid-prefix (when wired), safe wrapper |
| 21b | β0 | a prefix-shaped `get_memory_by_id` argument, and an unrepairable markup payload, on a server built without `prefix_resolver` or a project registry | test build, no resolver, `known_projects={}` | the prefix argument reaches the tool unchanged; the installer logged a WARNING once that the uuid-prefix member is absent; the markup refusal carries a null `escalation_id` and nothing is filed |
| 21d | β0 | a prefix citation on a server built with `prefix_resolver` | fake `PrefixResolver` | the citation is expanded (the uuid-prefix member is installed) |
| 21c | β | a hyphen-spelled `project_id`, and a worktree `project_root`, each with a prefix citation | fake `PrefixResolver` keyed by canonical project | expanded exactly as the canonical spelling is; the resolver saw the canonical project |
| 22 | C7 | an `add_memory` whose Mem0 leg raises, one whose Graphiti enqueue fails, and an `add_system_record` whose Mem0 add returns no ids | fake backends | `store_errors` names the failed store with its error text; that store is in no store list; the rendered message still carries the text; a leg queued for retry is not in `store_errors` |

Facing the producer side: rows 3, 5, 5b, 9, 11, 11b, 14, 18, 21, 21b, 21d, 22; facing the consumer side: rows 4,
7, 8, 8b, 12, 12b, 16, 17, 21c. ω runs rows 1, 4, 5, 12, 14, 17 and 22 through `mcp._tool_manager.call_tool` on a
server composed by η's builders.

## Partition measurement (heuristic 14, protocol of `docs/code-quality.md`)

`server/tools.py`: 52% prose; the tool descriptions are in their right home except their maintainer-facing part
(ι). Named partition of `create_mcp_server` by tool family, tool-body lines (prose included; measured at
`ba17934861` — the deltas to `7848173a7d` are `submit_task` +14 and `add_memory` +3 lines), shared closures, and
the heuristic-13 symptom each candidate would hit:

| Family module (leaf) | Tools | Body lines | Shared closures it needs | Symptom |
|---|---|---|---|---|
| `ops.py` (ε1) | get_status, replay_to_graphiti, get_queue_stats, get_curator_state, get_wal_status, replay_dead_letters, replay_event_dead_letters, get_dead_letters, delete_dead_letters, trigger_reconciliation, unhalt_reconciliation, reload_config, `/health`, `/alive` | 784 | `halt_payload` (C1) | none |
| `scheduler.py` (ε1) | set_task_priority_override, clear_task_priority_override, reorder_pin_queue, get_pin_queue, get_scheduler_state, get_scheduler_events, request_park_eviction | 490 | override and park-eviction store (S14 θ 6718) | none once 6718 lands; today `main.py` imports `server.tools._checkpoint_overrides_db_if_exists` (a private reach 6718 retires) |
| `defects.py` (ε1) | declare_defect, record_defect_outcome, lookup_defect_ref | 106 | `reject_ticket_id` (`server/tool_args.py`) | none |
| `task_read.py` (ε2) | get_tasks, get_statuses, get_external_statuses, get_task, search_tasks, find_tasks_by_metadata, list_tickets | 654 | paging helpers move with it | none |
| `task_write.py` (ε2) | set_task_status, set_task_claimant, submit_task, resolve_ticket, cancel_ticket, commit_planning, update_task, remove_task, add_dependency, remove_dependency, rebuild_candidate_key_index | 1,373 | `reject_ticket_id` | none |
| `memory_read.py` (ε3) | search, count_memories_by_metadata, get_cycle_summary_presence, scan_memory_content, get_memories_by_metadata, get_memory_by_id, get_entity, get_entity_by_uuid, get_episodes | 997 | prefix helpers move with it | none |
| `graph_admin.py` (ε3) | delete_episode, redact_episode_content, update_edge, refresh_entity_summary, set_entity_summary, rename_entity, merge_entities, reassign_edge, delete_entity, ensure_entity_node, rebuild_entity_summaries | 781 | `batched_task_statuses` (`server/task_status_facts.py`) | none |
| `memory_write.py` + `memory_write_checks.py` (ε4) | add_episode, add_memory, add_system_record | 1,166 + 362 helpers | `batched_task_statuses` | none |
| `memory_curation.py` + `citation_repoint.py` (ε5) | propose_consolidation, delete_memory, update_memory, consolidate_memories | 1,531 + 825 helpers | none shared | none |

Every shared closure is either a `ToolContext` member (C1), the scope gate (C2) or a leaf module the families
import downward (`tool_args.py`, `task_status_facts.py`); no family needs another. Measured residuals the moves
do not fix and this PRD does not take: `consolidate_memories` (951 lines) and `add_memory` (787) are heuristic-4
problems inside one function — γ shrinks `add_memory`; `consolidate_memories` stays whole (recorded in ε5's
commit). `server/main.py` follows task 8804's partition measurement (composition root, background loops,
builders, HTTP boundary, process lifecycle; the HTTP/lifecycle cycle is fixed by putting `_run_shielded` and
`_is_shutdown_initiated` with lifecycle): 8804 extracts the loops, β0 the dispatch-stack installation, θ1 the
lifecycle, θ2 the HTTP boundary, η the builders.

## Cross-PRD relationship

Edge notation: "X → Y" means X depends on Y (Y lands first).

| Other side | Direction | Mechanism | Owner | Order / action |
|---|---|---|---|---|
| S26 β **9147** | S26 owns | `AddMemoryResponse` (S26 C2) | S26 | fk-05a87fd1777d stamped onto 9147 and closed by σ (OQ-5 option B); lead amendment: 9147's renderer keeps each failed store's error text until σ moves its source to `store_errors` |
| S26 φ **9252** | same files | `services/memory_service.py` (last S26 declarer; its closure holds 9147) | S26 | σ → 9252 (the per-file rule, as the ruling places it) |
| S26 γ **9158** | consumes | `MemoryService` built with its collaborators; `set_event_buffer` the one pre-start setter (S26 C3, D4) | S26 | in 9280's closure, so η is after it through κ |
| S26 λ 9194, μ **9203**, ρ 9243 | same file | `server/tools.py`; μ adds `store_inventory` (S26 C7) | S26 | through the chain to 9280; ε1 moves `/health`, which reads C7 |
| S10 α 9180, γ 9195, κ1 9248, μ2 **9280** | same files | `tools.py`, `main.py`; `InterceptorCollaborators` | S10 | κ → 9280; every other tools.py/main.py leaf is after κ; η builds on μ2's bundle |
| S10 ν **9282** | same file | `middleware/task_interceptor.py` (last wave-2 declarer) | S10 | λ → 9282 |
| S10 ο **9285** | none | `reconciliation/targeted.py` (behind 8528 through 9283) | S10 | no S15 leaf edits `targeted.py` (OQ-9 ruled (a)) |
| Task **9493** (filed 2026-11-12, high; done, merged `1d0ae21e44`) | prerequisite, same files | one stop signal (unit `KillMode` or the stop callback's repeat-signal rule); drain kept at 10 s; tool-call durations recorded | lead (outside every plan) | ζ → 9493; ζ lengthens the drain inside the phases' room (D11); θ1 carries 9493's stop-signal rule into `ProcessLifecycle` unchanged |
| S12 δ (C3), wave 3 | S12 consumes this plan's homes; S15 consumes its resolver | header identity in `resolve_identity` / `extract_causation` at `server/tool_context.py` | S12 | OQ-11 ruled (b), as S12/OQ-11 (a1): S12 δ → α (α moves both helpers verbatim); ε1 → S12 δ and δ → S12 δ. S12 δ adds `ctx` and the header rule at `server/tool_context.py` — refusing a differing `agent_id` or `_causation_id` (Leo's S12-2 ruling) on writes and on `get_entity`, `get_entity_by_uuid` and `get_episodes`, never on `search`'s `agent_id` filter (lead ruling; ε1–ε5 carry the exclusion verbatim, D7) — and passes `ctx` at the 17 `_extract_causation(` call sites in `server/tools.py` before the verbatim moves carry them; δ reads the causation header through S12 δ's resolver |
| S12 κ1 (C6), wave 3 | consumes | `InterceptorCollaborators.reconciler` provider; `TargetedReconciler(…, task_writes=, planned_episode_registry=)` | S12 | η → S12 κ1 (OQ-9 ruled (a), one ruling with S12/OQ-2); η's builder passes S12 C6's provider; S12 κ1 deletes both `targeted.` back-wires. The S12/S13 cycle is cut by the lead's ruling S13/OQ-6 (a) |
| S12 δ, κ1, κ2, ζ, wave 3 | same files | `server/tools.py` and `config/schema.py` (δ), `middleware/task_interceptor.py`, `server/main.py` and the event-buffer module (κ1), `server/main.py` (κ2), the journal module (ζ) | lead | S15 takes the edges above (ε1, δ → S12 δ; η → S12 κ1); S12 owns S12 δ → α. The other same-file pairs (S12 δ against β0 and β on `tools.py`; λ and S12 κ1 on `task_interceptor.py`; S15's main.py leaves and S12 κ1, κ2; ζ and S12 δ on `schema.py`) are left to the lead's R54 per-file order; xref to S12 |
| S19 α, wave 3 | same files | `reconciliation/journal.py`, `reconciliation/event_buffer.py` move to `recon_state/` | S19 | κ → S19 α (lead ruling OQ-8); C4's owner paths are the modules owning the DDL after S19 α; η (after κ) edits `recon_state/journal.py`. S19 β1 and β2 share `server/tools.py` and `server/main.py` with import-line repoints only: rebase-only under file locks (lead ruling S19/Q-C6), no edge |
| S19 δ, wave 3 | subsumes 5983 | pure consolidation helpers out of `server/` | S19 | ε5 → S19 δ (replaces the edge onto 5983, which S19 cancels into its δ) |
| S16 β2, wave 3 | same files | `reload_config`, `RELOADABLE_FIELDS`; deletes `config/reload.py`; edits `config/schema.py`, `server/tools.py`, `server/main.py` | S16 | S16 β2 → ε1 (S16's draft takes it: β2 edits `reload_config` in `tool_families/ops.py` and `tests/test_config_reload.py` after ε1 moves them). Never ε1 → S16 β2: S16 β2 waits on 8528 and 5040 (S16 α → 9229 → 9202 → 9134 → 8528; S16 β1 → 9210 → 9166 → 8528; both reach 5040, re-measured 2026-11-12), so that edge would park ε1–ω. ζ and β2 on `schema.py`: already ordered, ζ first, through S16 β2 → ε1 → β → β0 → ζ (no extra edge) |
| S1 θ1a, wave 3 | same file | `server/tools.py::_report_unverified_claims` becomes async; markup guard's sinks | S1 | ε4 → S1 θ1a (ε4 moves `_report_unverified_claims` with `add_memory`/`add_episode`) |
| S13, wave 3 | same files | `server/tools.py`, `config/schema.py` (its λ) | S13 | file lock; xref |
| S17, wave 3 | none | `reconciliation/targeted.py` | S17 | no S15 leaf edits it (OQ-9 ruled (a)) |
| S24, wave 3 | restates | its fused-memory lease arithmetic uses `TimeoutStopSec` 90 s | S24 | no action: `TimeoutStopSec` stays 90 s (OQ-1 option D), so S24's arithmetic needs no change (the earlier "≥ 123 s" row is withdrawn) |
| S14 ξ **6719** | consumes | `McpErrorType`, `error_envelope` | S14 | in 9280's closure; α adds `NOT_CONFIGURED` (S10's precedent: a member is added by the leaf that needs it) |
| S14 μ **6721** | consumes | `AgentIdentity`; `_resolve_identity` returns it | S14 | in 9280's closure; δ and η consume it; the lead records OQ-6's answer (identity stays per tool) on 6721 |
| S14 θ **6718** | consumes | shared override and park-eviction store | S14 | in 9280's closure; ε1 consumes it |
| Task **4643** | same chokepoint | `install_boundary_guards` (one chain) | 4643 | β0 → 4643; lead note on 4643 |
| Task **4867** | same chokepoint | unknown-argument guard | done (merged 2026-11-11 `daff96ef3c`) | its guard sits between the safe wrapper and the markup guard on main; β0 moves the installation with the rest |
| Task **5071** | overlapping | INV-8 offload of `resolve_main_checkout` cold misses | 5071 | α0 does the 23-row MCP-handler cluster; lead `amend` narrows 5071 to its other sites |
| Task **8804** | same file | `server/background_loops.py` | 8804 | θ1 → 8804; lead `amend` keeps 8804 to its step (a) |
| Task **5462** | deletable owner | the 4 data-dir derivations (fk-74517af4aeed) | 5462 | η → 5462 |
| Task **6815** | same file | `_to_underscore_canonical` alias | 6815 | β → 6815 |
| Tasks **5717**, **6295** | same file, wider scope | docstring corrections plus a caller audit (5717) and recon prompts (6295) | theirs | no edge; lead `amend` hands their `tools.py` docstring halves to ι |
| Task **6235** | subsumed | `set_task_status` docstring | ι | cancel naming ι |
| Task **4469** | subsumed | `allow_mcp_markup` strip in add_system_record/update_memory; re-export shim | δ | cancel naming δ; δ carries fk-358e8dfe8d02 |
| Task **5511** | subsumed | `get_statuses` description size | ι | cancel naming ι |
| Task **4044** | subsumed | audit-then-raise on read tools untested | α | cancel naming α (boundary row 2) |
| S22 (wave 3, concurrent) | S22 consumes | C4 read model | S15 κ | S22's two consumer leaves take the edges S22 ε → κ and S22 ζ → κ (S22's decompose wires them); S22 deletes the 11 SQL sites, 8 DDL fixtures and `_DEFAULT_BURST_COOLDOWN`, and closes fk-1c1d247caaa7 |
| Gate **6698** | downstream | trigger chain | lead | 6698 → γ (the one high leaf) |

## Live tasks on `server/tools.py` and `server/main.py`

161 non-terminal tasks declare `server/tools.py` and 34 declare `server/main.py` in `metadata.files` (read
2026-11-12). They are not ordered against S15 (R54 leaves non-chain tasks to the file locks). After each ε lands,
the clerk amends the `metadata.files` of every non-terminal task that declares `server/tools.py` and names a tool
of the moved family (title or description), adding the family module — the decompose return's `moves` list
carries the tool → module table above. β0, θ1, θ2 and η do the same for the main.py symbols they move. No edge is
added.

## Decomposition plan

Every leaf: `metadata.x_finding_key` (list), `x_finding_run`, `source = "review-all"`, file-level
`metadata.files`; priority = the highest severity of the keys it closes; reviewed against
`docs/code-quality.md`, heuristics cited by name: every move or extraction (α, β0, β, ε1–ε5, θ1, θ2, η) is bound
by heuristic 13, *files make internal sense in isolation*, and heuristic 14, *no file too large* with its
no-cheating clause — a new module imports downward and never back into the file it left. Same-file leaves are
serialized by the edges below; the shared governance files (`shared/tests/loop_blocking_allowlist.py`) are
edited by unordered leaves on disjoint rows and left to the file locks (R54). Every leaf that moves a symbol
keyed in `shared/tests/loop_blocking_allowlist.py` re-keys the moved rows' `relpath`/`qualname` and never
re-blesses a site whose call changed.

Sizing exceptions, stated: ε2–ε5 move 1,700–2,400 lines each, above the overlay's 300–1,500 band; a verbatim
move counts twice in a diff and is pinned by the `tools/list` snapshot (row 17) and the import-graph test
(row 18), and each commit records its after-measure. α0's 26 files cross the >15 review trigger: it is a
mechanical codemod, hand-carried if lock-starved for more than 24 h (R57's precedent).

**κ — Dashboard panel read model served by fused-memory `get_status(panels=…)`** (priority medium; ~850 LOC)
- Files: `shared/src/shared/dashboard_panels.py` (new), `fused-memory/src/fused_memory/server/dashboard_panels.py` (new), `fused-memory/src/fused_memory/recon_state/journal.py`, `fused-memory/src/fused_memory/recon_state/event_buffer.py`, `fused-memory/src/fused_memory/services/write_journal.py`, `fused-memory/src/fused_memory/services/read_rollup.py`, `fused-memory/src/fused_memory/middleware/ticket_store.py`, `fused-memory/src/fused_memory/server/tools.py`, `fused-memory/src/fused_memory/server/main.py`, `fused-memory/tests/test_dashboard_panels_read_model.py` (new), `shared/tests/test_dashboard_panels.py` (new)
- Deps: 9280, S19 α
- The two `recon_state/` paths are S19 α's; today they are `reconciliation/journal.py` and
  `reconciliation/event_buffer.py` (the clerk repaths if S19's move is ruled otherwise).
- Keys: none of S15's; contributes to S22's fk-1c1d247caaa7 (S22's consumer leaf closes it).
- Produces C4. Moves each panel query into the owning store as a public read; `EventBuffer.burst_state_rows`
  computes `effective_state` with the instance's cooldown; `run_server` builds the `DashboardPanelReader` and
  passes it to `create_mcp_server`; `get_status` gains `panels`.
- Signal: on the deployed server, `get_status(panels={"recon": {}, "memory_ops": {"start": …, "end": …},
  "curator": {"since": …}})` returns the three panels; rows 12, 12b and 13 green in CI.
- Deletes: nothing in S15 (the dashboard's SQL, DDL copies and cooldown constant go in S22's leaves). G7: INV-5,
  INV-9, INV-11, INV-13.

**α0 — fused-memory project-root normalization takes an injected checkout resolver and runs off the event loop** (priority medium; ~350 LOC, codemod)
- Files: `fused-memory/src/fused_memory/server/tools.py`, `shared/tests/loop_blocking_allowlist.py`, `fused-memory/tests/_hermetic_task_stack.py`, `fused-memory/tests/_submit_stack.py`, `fused-memory/tests/server/test_markup_tripwire_gate.py`, `fused-memory/tests/test_declared_scope_stamp.py`, `fused-memory/tests/test_get_external_statuses.py`, `fused-memory/tests/test_get_statuses_pagination.py`, `fused-memory/tests/test_markup_guard_fused_memory.py`, `fused-memory/tests/test_mcp_boundary_canonicalization.py`, `fused-memory/tests/test_operational_routing_boundary_matrix.py`, `fused-memory/tests/test_park_eviction_tool.py`, `fused-memory/tests/test_recon_reliability_integration.py`, `fused-memory/tests/test_recurring_gate_boundary.py`, `fused-memory/tests/test_scheduler_overrides_tools.py`, `fused-memory/tests/test_scheduler_state_tools.py`, `fused-memory/tests/test_status_envelope_contract.py`, `fused-memory/tests/test_submit_resolve_tools.py`, `fused-memory/tests/test_submit_task_enumeration_gate.py`, `fused-memory/tests/test_task_interceptor.py`, `fused-memory/tests/test_task_tools.py`, `fused-memory/tests/test_task_write_agent_id.py`, `fused-memory/tests/test_tool_errors.py`, `fused-memory/tests/test_tools_claimant.py`, `fused-memory/tests/test_tools_validation.py`, `fused-memory/tests/test_unknown_argument_guard_fused_memory.py`
- Deps: κ
- Keys: contributes to fk-a9f21093207e.
- `create_mcp_server(…, checkout_resolver=resolve_main_checkout)`; `_normalize_project_root` calls it. In the same
  change every inline `_normalize_project_root(project_root)` (23 lines) becomes
  `await asyncio.to_thread(_normalize_project_root, project_root)` — the idiom the three defects tools already
  use — because a resolver reached through a parameter is invisible to the loop-blocking scanner
  (`shared/tests/loop_blocking_scan.py`: "An unresolvable name is silence"), and leaving the call inline would
  hide a live INV-8 violation behind stale rows. The 23 `to_file` rows of the `resolve_main_checkout` cluster
  are then deleted as **fixed**. The 32 dotted-path patches of `server.tools.resolve_main_checkout` (24 test
  files) become the keyword (tests stance: the seam the patches reached through was the defect).
- Signal: `git grep -n "fused_memory.server.tools.resolve_main_checkout" -- fused-memory/tests` is empty;
  row 8b green; the loop-blocking gate (`shared/tests/test_loop_blocking_gate.py`) is green with the 23 rows
  gone; the fused-memory suite is green.
- Deletes: 32 private patch sites; 23 inline blocking calls and their 23 allowlist rows. G7: INV-8, INV-10.

**α — fused-memory `ToolContext`: interceptor bound once, one not-configured code, journaled reads** (priority low; ~800 LOC)
- Files: `fused-memory/src/fused_memory/server/tool_context.py` (new), `fused-memory/src/fused_memory/server/tools.py`, `shared/src/shared/mcp_envelope.py`, `fused-memory/tests/test_tool_context.py` (new), `fused-memory/tests/test_identity_resolution.py`, `fused-memory/tests/test_memory_service.py`
- Deps: α0
- Keys: fk-22ab8d1f3a42, fk-1509bcd45e2a. Absorbs task 4044.
- Produces C1. `ToolContext.build` binds the interceptor once, real or null (S10 μ2's null bundle), at the top;
  closures read `ctx.…`; the 7 `ConfigurationError` literal sites become `ctx.not_configured(…)`;
  `ctx.journaled_read` decorates the 4 memory reads and 6 task reads that journal today (the 3 that do not stay
  unjournaled: their rows have no consumer, measured in fk-1509bcd45e2a); `_backlog_gate`, `_known_project_gate`,
  `_halt_payload`, `_log_read`, the triage counter and `_resolve_identity`/`_extract_causation` (verbatim, with
  today's signatures; S12 δ edits them at this home after α, S12's edge) move to `tool_context.py`; the two test
  files importing those functions from `server.tools` retarget.
- Signal: rows 1 (tools half) and 2 green; a fused-memory started without Taskmaster answers `get_tasks` with
  `error_type: ConfigurationError`.
- Deletes: the mid-function rebinding, 8 `type: ignore[union-attr]`, the `_taskmaster_configured` closure local,
  4 hand-copied journal blocks and 6 journal calls. G7: INV-2, INV-4, INV-5.

**β0 — fused-memory's tool-call guard stack is installed once, by `create_mcp_server`** (priority medium; ~450 LOC)
- Files: `fused-memory/src/fused_memory/server/boundary_guards.py`, `fused-memory/src/fused_memory/server/tools.py`, `fused-memory/src/fused_memory/server/main.py`, `fused-memory/tests/test_tool_safe_wrapper.py`, `fused-memory/tests/test_markup_guard_fused_memory.py`, `fused-memory/tests/test_review_suggestion_curator_intake_e2e.py`, `fused-memory/tests/server/test_markup_tripwire_gate.py`, `fused-memory/tests/server/test_add_memory_write_triage_gate.py`, `fused-memory/tests/test_consolidation_proposal_tool.py`, `fused-memory/tests/test_delete_memory_citation_guard.py`
- Deps: α, ζ, 4643
- Keys: contributes to fk-a9f21093207e.
- `_install_safe_tool_wrapper`, `_install_recon_report_guards` and the stack order documented by
  `_install_tool_dispatch_guards` move from main.py into `server/boundary_guards.py` (4643's one chain);
  `create_mcp_server` is the primary server's one installer caller; main.py's installer goes.
  `_build_recon_report_components` calls `boundary_guards.install_recon_report_guards`; the recon-report server
  keeps its safe wrapper and unknown-argument guard. The stack, outermost to innermost: markup guard,
  unknown-argument guard (4867), (β's scope parse), uuid-prefix guard, safe wrapper, tool (C2); the uuid-prefix
  member is installed exactly when `prefix_resolver` is given — `create_mcp_server` gains its own
  `prefix_resolver: PrefixResolver | None = None` port (not `ToolContext.id_resolver`, which production never
  passes), `run_server` passes the per-use `lambda project_id, prefix:
  IdResolver.for_memory_service(memory_service).resolve_prefix(project_id, prefix)` so hot reload is kept, and
  with `None` the member is absent and the installer logs a WARNING once (INV-11); tests opt in by passing one,
  so the 24 test files carrying prefix-shaped tokens, measured by the critic, are unaffected unless they do. The
  order test in `test_markup_guard_fused_memory.py` retargets to a server built by `create_mcp_server`. The five
  test files that pass markup through `create_mcp_server` (measured at `ba17934861`) are adjusted; β0 measures the
  tests that pass undeclared arguments at its base (4867's guard is on main) and adds them to its declared files
  at decompose. A build with no project registry files nothing from the markup guard's sinks (row 21b).
- Signal: rows 21, 21b and 21d green; on the deployed unit, a tool call carrying MCP markup is refused by the
  markup guard exactly as before (the server log names the guard), and an `add_memory` citing an 8-hex prefix
  returns `meta.uuid_prefix_repair` as before β0.
- Deletes: `server/main.py::_install_tool_dispatch_guards`, `_install_safe_tool_wrapper` and
  `_install_recon_report_guards` from main.py, and the second wrapping path of `ToolManager.call_tool`. G7: INV-9,
  INV-10.

**β — fused-memory tools declare their project scope; one dispatch scope gate parses it** (priority medium; ~1,100 LOC)
- Files: `fused-memory/src/fused_memory/server/tool_scope.py` (new), `fused-memory/src/fused_memory/server/tool_context.py`, `fused-memory/src/fused_memory/server/tools.py`, `fused-memory/src/fused_memory/server/boundary_guards.py`, `shared/tests/loop_blocking_allowlist.py`, `fused-memory/tests/test_tool_scope.py` (new), `fused-memory/tests/test_mcp_boundary_canonicalization.py`, `fused-memory/tests/test_graphiti_group_arg_canonicalization.py`, `fused-memory/tests/test_project_id_agreement.py`, `fused-memory/tests/test_dead_letters_tool.py`
- Deps: β0, 6815
- Keys: fk-a9f21093207e.
- Produces C2 (D1–D5). All 67 tools register through `ctx.tool(scope=…)`; 5 are `UNSCOPED`
  (`get_external_statuses` keeps its per-reference normalization); 4 dead-letter tools are `raw` and validate;
  the graph mutators and memory writes are `write`; `add_episode`/`add_memory`/`add_system_record` add
  `backlog=True`; the 26 `project_root` tools are `task_root`, and `ScopeGate.parse` keeps α0's offload
  (`await asyncio.to_thread(self._checkout_resolver, root)`). The three defects tools already normalize (through
  `asyncio.to_thread`, B1's precedent); only `rebuild_candidate_key_index` passes the raw root today (to
  `taskmaster.reaudit_candidate_key_index`), so it is the one tool whose behaviour changes under `task_root`
  unless the leaf finds a reason it must see the caller's path, recorded as `raw(reason=…)`. The scope gate
  joins `install_boundary_guards` as the project-argument check ahead of the uuid-prefix guard (inside the markup
  and unknown-argument guards, outside the safe wrapper) and contains its own defects (C2).
  `get_external_statuses`' per-reference normalization moves to `ToolContext.normalize_project_root` (C1), which
  keeps α0's `asyncio.to_thread` offload.
- Behaviour changes (both ruled 2026-11-12): `get_status`, `get_queue_stats`, `get_dead_letters`,
  `replay_dead_letters` and `replay_event_dead_letters` start refusing an empty or unsafe `project_id`; the six
  formerly ungated graph mutators start refusing an unregistered project.
- Signal: rows 3–8b and 21c green; live: `get_status(project_id="know-live")` answers `halted` consistent with
  `halted_projects`, `trigger_reconciliation(project_id="dark-factory")` is accepted under `dark_factory`, and
  `get_dead_letters(project_id="")` returns the validation refusal.
- Deletes: 30 `_canonicalize_project_id_arg` and 30 `validate_project_id` calls, 22 `_normalize_project_root`
  calls (the 23rd, `get_external_statuses`' per-reference call, becomes `ctx.normalize_project_root`) with 21
  `isinstance(_normalized, dict)` narrowings (32/32/24/21 line hits including definitions and imports), the
  `_canonicalize_project_id_arg` adapter itself (the gate calls `utils/validation.py::canonicalize_project_id`),
  `_halt_payload`'s compensation, `trigger_reconciliation`'s remedy workaround, the 4 copied "pass the canonical
  spelling" docstring notes. G7: INV-1, INV-5, INV-8, INV-9, INV-10.

**ε1 — Move fused-memory's ops, scheduler and defects tool families out of `create_mcp_server`** (priority medium; ~1,500 LOC moved)
- Files: `fused-memory/src/fused_memory/server/tools.py`, `fused-memory/src/fused_memory/server/tool_families/__init__.py` (new), `fused-memory/src/fused_memory/server/tool_families/ops.py` (new), `fused-memory/src/fused_memory/server/tool_families/scheduler.py` (new), `fused-memory/src/fused_memory/server/tool_families/defects.py` (new), `fused-memory/src/fused_memory/server/tool_args.py` (new), `shared/tests/loop_blocking_allowlist.py`, `fused-memory/tests/test_mcp_tool_surface.py` (new), `fused-memory/tests/data/mcp_tool_surface.json` (new), `fused-memory/tests/test_config_reload.py`, `fused-memory/tests/test_park_eviction_tool.py`
- Deps: β, S12 δ (OQ-11 ruled (b): the moves carry S12 δ's `ctx` edits at the 17 `_extract_causation` call sites,
  and every per-tool identity call verbatim, `search`'s exclusion included, D7)
- Keys: contributes to fk-45cf67bff540, fk-9214f842ca92.
- S16 β2 waits on ε1 for `reload_config` (S16's edge); never ε1 → S16 β2 (§Cross-PRD relationship: that edge
  would park ε1–ω behind 8528 and 5040).
- Produces C6's registrar shape and the `tools/list` snapshot (row 17) and import-graph test (row 18). Re-keys
  the `create_mcp_server.request_park_eviction` allowlist row (`filed`) to `tool_families/scheduler.py`.
- Signal: after deploy, the live `tools/list` equals `tests/data/mcp_tool_surface.json`; the unit is active.
- Deletes: 21 tool closures from `create_mcp_server`. G7: INV-5, INV-8 (rows re-keyed, none re-blessed), INV-12
  (no new suppression).

**ε2 — Move fused-memory's task-read and task-write tool families** (priority medium; ~2,000 LOC moved)
- Files: `fused-memory/src/fused_memory/server/tools.py`, `fused-memory/src/fused_memory/server/tool_families/task_read.py` (new), `fused-memory/src/fused_memory/server/tool_families/task_write.py` (new), `fused-memory/src/fused_memory/server/tool_args.py`, `shared/tests/loop_blocking_allowlist.py`, `fused-memory/tests/_fm_helpers.py`, `fused-memory/tests/test_get_statuses_pagination.py`, `fused-memory/tests/test_find_tasks_by_metadata_tool.py`
- Deps: ε1
- Keys: contributes to fk-45cf67bff540, fk-9214f842ca92.
- Re-keys the `create_mcp_server.submit_task` allowlist row (`accepted`) to `tool_families/task_write.py`.
- Signal: row 17 green against the unchanged snapshot; live `tools/list` unchanged.
- Deletes: 18 tool closures. G7: INV-5, INV-8 (rows re-keyed, none re-blessed).

**ε3 — Move fused-memory's memory-read and graph-admin tool families** (priority medium; ~1,900 LOC moved)
- Files: `fused-memory/src/fused_memory/server/tools.py`, `fused-memory/src/fused_memory/server/tool_families/memory_read.py` (new), `fused-memory/src/fused_memory/server/tool_families/graph_admin.py` (new), `fused-memory/src/fused_memory/server/task_status_facts.py` (new), `fused-memory/tests/server/test_search_tool.py`
- Deps: ε2
- Keys: contributes to fk-45cf67bff540, fk-9214f842ca92.
- Moves `search`, `get_entity`, `get_entity_by_uuid` and `get_episodes` with their identity calls as S12 δ left
  them: the three reads keep the refusal of a differing `agent_id`, and `search` keeps its exclusion (D7; S12's
  boundary row 10d stays green).
- Signal: row 17 green; live `tools/list` unchanged.
- Deletes: 20 tool closures, `_verify_mint_referent` and `_batched_task_statuses` as closures. G7: INV-5.

**ε4 — Move fused-memory's memory-write tool family** (priority medium; ~1,700 LOC moved)
- Files: `fused-memory/src/fused_memory/server/tools.py`, `fused-memory/src/fused_memory/server/tool_families/memory_write.py` (new), `fused-memory/src/fused_memory/server/tool_families/memory_write_checks.py` (new), `fused-memory/tests/server/test_add_memory_write_triage_gate.py`, `fused-memory/tests/server/test_write_triage_supersession.py`
- Deps: ε3, S1 θ1a
- Keys: contributes to fk-45cf67bff540, fk-9214f842ca92.
- Moves `_report_unverified_claims` with `add_memory`/`add_episode` as S1 θ1a left it.
- Signal: row 17 green; live `tools/list` unchanged.
- Deletes: 3 tool closures and 10 helper closures. G7: INV-5.

**ε5 — Move fused-memory's memory-curation family; `create_mcp_server` becomes a composition of family registrars** (priority medium; ~2,400 LOC moved)
- Files: `fused-memory/src/fused_memory/server/tools.py`, `fused-memory/src/fused_memory/server/tool_families/memory_curation.py` (new), `fused-memory/src/fused_memory/server/tool_families/citation_repoint.py` (new), `fused-memory/tests/test_mcp_tool_surface.py`
- Deps: ε4, S19 δ
- Keys: fk-45cf67bff540, fk-9214f842ca92.
- `create_mcp_server` = build `ToolContext`, run `FAMILY_REGISTRARS`, install the dispatch stack. The commit
  records the after-measurement (tools.py line count, each family's) and that `consolidate_memories` stays
  whole.
- Signal: row 17 and 18 green; live `tools/list` unchanged; `server/tools.py` holds no tool body.
- Deletes: the last 4 tool closures and 8 citation helper closures; `create_mcp_server`'s 28 closure locals.
  G7: INV-5, INV-9.

**δ — fused-memory write-gate protocol, one write-time control-key registry, and `submit_task` as an ordered gate list** (priority low; ~900 LOC)
- Files: `fused-memory/src/fused_memory/server/write_gates.py` (new), `fused-memory/src/fused_memory/server/write_controls.py` (new), `fused-memory/src/fused_memory/server/tool_context.py`, `fused-memory/src/fused_memory/server/tool_families/task_write.py`, `fused-memory/src/fused_memory/server/tool_families/memory_write.py`, `fused-memory/src/fused_memory/server/tool_families/memory_curation.py`, `fused-memory/src/fused_memory/server/markup_tripwire.py`, `fused-memory/tests/test_write_controls.py` (new), `fused-memory/tests/test_submit_task_gate_list.py` (new), `fused-memory/tests/test_markup_guard_fused_memory.py`, `shared/tests/test_toolcall_markup.py`
- Deps: ε5, S12 δ (OQ-11 ruled (b); also reached through ε1)
- Keys: fk-1cadfcc36cfa; carries fk-358e8dfe8d02 (task 4469's key). Absorbs task 4469.
- Produces C5's protocol half and `SUBMIT_TASK_GATES`. `extract_causation` becomes C5's
  `split_write_controls(metadata, header_causation_id=…)` at every call site; δ reads the causation header through
  S12 δ's resolver, which has already refused a differing `_causation_id` (Leo's S12-2 ruling), and passes its
  value (D9); the source comes from `AgentIdentity`, not an `agent_id` prefix. The `markup_tripwire.py`
  re-export shim goes (4469).
- Signal: row 16 green; live: `submit_task(metadata={"allow_near_duplicate": true, …})` files a task whose
  `get_task` metadata carries no control key.
- Deletes: the per-tool `strip_markup_override` calls (6) and `allow_near_duplicate` pops, the re-export shim.
  Priority inversion recorded: δ (low) is a prerequisite of γ (high); no override (contract §4). G7: INV-5,
  INV-9.

**γ — fused-memory `add_memory` as an ordered gate list; sibling write entry points declare their gate sets** (priority **high**; ~1,300 LOC)
- Files: `fused-memory/src/fused_memory/server/tool_families/memory_write.py`, `fused-memory/src/fused_memory/server/tool_families/memory_write_checks.py`, `fused-memory/src/fused_memory/server/tool_families/memory_curation.py`, `fused-memory/src/fused_memory/server/write_gates.py`, `fused-memory/tests/test_add_memory_gate_list.py` (new), `fused-memory/tests/test_write_entry_gates.py` (new)
- Deps: δ
- Keys: fk-330a60eef271.
- Produces `ADD_MEMORY_GATES` and `WRITE_ENTRY_GATES` (C5): category, entities, the three recon-stage content
  guards (on `AgentIdentity.is_recon_stage`), flag-kind, live-status, premature-completion, write triage (with
  task 6956's completion-record input), topic-cluster, near-duplicate, completion-claim — today's order, measured
  and pinned; `add_system_record`, `add_episode`, `update_memory` and `consolidate_memories` declare theirs with
  explicit skips. The write-triage gate is the table's one LLM row and keeps its partition
  (`one-decision-partition-per-area`, INV-23; §Decision partition).
- Signal: rows 14–15 green; the server log names the refusing gate on a live refused `add_memory`.
- Deletes: about 20 inline if-chains in `add_memory`; each sibling's ad-hoc subset. G7: INV-1, INV-5, INV-10.
- High: gate 6698 → γ at decompose.

**ι — fused-memory tool descriptions carry the agent-facing contract only** (priority medium; ~1,500 LOC changed, mostly removed)
- Files: `fused-memory/src/fused_memory/server/tool_families/ops.py`, `fused-memory/src/fused_memory/server/tool_families/scheduler.py`, `fused-memory/src/fused_memory/server/tool_families/defects.py`, `fused-memory/src/fused_memory/server/tool_families/task_read.py`, `fused-memory/src/fused_memory/server/tool_families/task_write.py`, `fused-memory/src/fused_memory/server/tool_families/memory_read.py`, `fused-memory/src/fused_memory/server/tool_families/graph_admin.py`, `fused-memory/src/fused_memory/server/tool_families/memory_write.py`, `fused-memory/src/fused_memory/server/tool_families/memory_curation.py`, `fused-memory/tests/data/mcp_tool_surface.json`, `fused-memory/tests/test_tool_descriptions.py` (new)
- Deps: γ
- Keys: fk-009fbbf855e7. Absorbs tasks 5511 and 6235, and the `tools.py` docstring halves of 5717 and 6295
  (lead `amend`s; their caller audit and prompt work stay theirs).
- Each description states purpose, arguments, return, refusals and remedies; maintainer and reviewer prose
  (provenance, measurements, rejected alternatives) moves to the commit message or, where the next agent needs
  it, a test or `docs/` with a pointer (Comments stance). The rubric test's markers are recorded in the test.
  The `get_statuses` pagination contract names `returned` as the served count and `page_size` as the requested
  size (5717's correction); `set_task_status` names every caller-selectable `done_provenance` kind (6235's).
  No size target (`docs/code-quality.md` §Do not steer by); the commit records before/after characters.
- Signal: live `tools/list` matches the updated snapshot; rubric test green.
- Deletes: the reviewer-facing prose, e.g. `get_statuses`' "(Deliberately no line numbers above …)". G7: INV-9.

**λ — fused-memory's `TaskInterceptor` raises the declared not-configured error** (priority low; ~120 LOC; `complexity='simple'`)
- Files: `fused-memory/src/fused_memory/middleware/task_interceptor.py`, `fused-memory/src/fused_memory/backends/task_backend_errors.py`, `fused-memory/tests/test_not_configured_envelope.py` (new)
- Deps: α, 9282
- Keys: contributes to fk-22ab8d1f3a42 (α closes the tools half; λ the interceptor half).
- `_ensure_taskmaster` raises `TaskmasterNotConfigured` (`error_type = McpErrorType.NOT_CONFIGURED`); the 4
  `'ConfigError'` literals render the same code through `error_envelope`.
- Signal: row 1 green with no `RuntimeError`/`ConfigError` code.
- Deletes: 2 of the 3 not-configured spellings. G7: INV-2.

**ζ — fused-memory's shutdown runs declared phases of concurrent closes; the force-exit budget is computed from them and the drain is sized from measured tool-call durations** (priority medium; ~700 LOC)
- Files: `fused-memory/src/fused_memory/config/shutdown_plan.py` (new), `fused-memory/src/fused_memory/server/main.py`, `fused-memory/src/fused_memory/config/schema.py`, `fused-memory/tests/test_server_shutdown.py`, `fused-memory/tests/test_config_schema.py`, `fused-memory/tests/test_systemd_unit_config.py`, `scripts/fused-memory.service.template`, `scripts/check_fused_memory_unit_parity.py`, `tests/scripts/test_check_fused_memory_unit_parity.py`
- Deps: κ, 9493
- Keys: fk-d085b903f01c.
- Produces C3 (D10, D11). `_graceful_shutdown`'s eleven sequential steps and the three closes that run outside
  the budget today become `SHUTDOWN_PHASES`, starting from C3's draft table: the seven loop cancels as one step,
  `background_loops.cancel_all` (one timeout, one `gather`; bound to `server/background_loops.py` once 8804 lands,
  else to a `run_server`-local closer), the recon-report stop as one step and `topic_cluster_store.close` as one
  step, all moved from `run_server`'s `finally` into the phases after the force-exit arm; the uvicorn shutdowns
  and the watchdog stop stay before it. Each step keeps today's timeout unless the leaf records a measured shorter
  bound. The phase order is pinned by row 11b; `FORCE_EXIT_BUDGET_SECS` is computed (55 s at the draft table),
  `STEP_CANCEL_GRACE_SECS` replaces `_run_shielded`'s literal 1.0, and `PRE_ARM_BOUND_SECS` sums the bounded
  pre-arm waits, starting with the watchdog join, whose single home becomes `WATCHDOG_JOIN_TIMEOUT_SECS`
  (`_stop_watchdog_thread`'s default reads it instead of a literal 2.0). ζ confirms on a live stop that the
  operator-stop path drains once (the two `serve()` tasks drain concurrently inside the `gather`, so the
  `finally`'s second `shutdown()` calls return at once); if not, `PRE_ARM_BOUND_SECS` counts the second wait.
  The drain: ζ reads the tool-call durations task 9493 recorded (re-measuring from the server log's request
  timestamps if they are older than a week), sets `ServerConfig.graceful_shutdown_timeout`'s default from the
  LLM-backed tools' p90 and maximum (for example 30 s), and keeps it strictly under `SYSTEMD_TIMEOUT_STOP_SECS −
  PRE_ARM_BOUND_SECS − FORCE_EXIT_BUDGET_SECS` (33 s at the draft table). `ServerConfig`'s validator and row 10
  assert drain + `PRE_ARM_BOUND_SECS` + budget < `TimeoutStopSec`; the two `tests/test_config_schema.py` classes
  that pin the 10 s default and the 75 s arithmetic are rewritten to compute their refused and accepted values
  from `config/shutdown_plan.py` (no figure restated), and main.py's `_arm_force_exit` defaults to
  `FORCE_EXIT_BUDGET_SECS`. `TimeoutStopSec` stays 90 in the unit
  template; the parity script reads it from the template; the template's comment names
  `config/shutdown_plan.py::FORCE_EXIT_BUDGET_SECS` instead of `_FORCE_EXIT_BUDGET`. No unit re-install is needed
  for ζ (task 9493 carries the one re-install).
- Signal: rows 9–11b green; with 9493 and ζ deployed, a live `systemctl --user restart fused-memory` during a
  slow `submit_task` returns that call's answer, not a reset; the journal logs every phase and the stop ends
  inside 90 s; `scripts/check_fused_memory_unit_parity.py` passes.
- Deletes: the 70 s comment, the two "45s" timer figures in `_graceful_shutdown`'s and `_shutdown_with_watchdog`'s
  docstrings, `_MIRROR_FORCE_EXIT_BUDGET`, `_MIRROR_SYSTEMD_TIMEOUT_STOP_SECS`,
  `server/main.py::_FORCE_EXIT_BUDGET` and `::_SYSTEMD_TIMEOUT_STOP_SECS` (main.py and the tests read
  `config/shutdown_plan.py`'s; no alias stays), `test_schema_mirror_constants_match_main`, the literal `5 * _CLEANUP_STEP_TIMEOUT`, the `"TimeoutStopSec=90"`
  literal in `REQUIRED_SERVICE_DIRECTIVES` (the fourth copy), the 75 s / 90 s figures restated in
  `ServerConfig.graceful_shutdown_timeout`'s description. G7: INV-5, INV-7, INV-9.

**θ1 — fused-memory process lifecycle is a `ProcessLifecycle` object, not module globals** (priority medium; ~950 LOC)
- Files: `fused-memory/src/fused_memory/server/process_lifecycle.py` (new), `fused-memory/src/fused_memory/server/main.py`, `fused-memory/tests/server/test_asgi_exception_shield.py`, `fused-memory/tests/test_operator_signal_handler.py`, `fused-memory/tests/test_server_shutdown.py`, `fused-memory/tests/test_main_curator_usage_gate.py`, `fused-memory/tests/test_durable_queue.py`, `fused-memory/tests/test_event_queue.py`, `fused-memory/tests/test_config_schema.py`, `fused-memory/tests/test_watchdog_thread.py`, `fused-memory/tests/test_drain_signal_handler.py`, `fused-memory/tests/test_event_loop_lag.py`
- Deps: ζ, β0, 8804
- Keys: fk-b3d95cdaa0d3.
- `ProcessLifecycle(exit_process=os._exit, install_signal=signal.signal, sd_notify=…)` owns the operator-stop
  flag, force-exit timer, watchdog thread, drain signal handler and singleton socket; `_run_shielded`,
  `_is_shutdown_initiated`, the shutdown plan runner and the operator-stop handler (with task 9493's stop-signal
  rule, unchanged) live with it (8804's redrawn partition); `main()` constructs one and passes it to `run_server`.
- Signal: row 19 green; live `systemctl --user stop fused-memory` still exits 0.
- Deletes: 5 module globals and the private patch sites on them (63 test lines in the watchdog, drain and
  loop-lag test files plus the 18 in the shield and signal-handler tests). G7: INV-7, INV-10.

**θ2 — fused-memory's HTTP/ASGI boundary leaves `server/main.py`** (priority medium; ~500 LOC)
- Files: `fused-memory/src/fused_memory/server/http_boundary.py` (new), `fused-memory/src/fused_memory/server/main.py`, `fused-memory/tests/server/test_recon_report.py`, `fused-memory/tests/server/test_asgi_exception_shield.py`
- Deps: θ1
- Keys: contributes to fk-b3d95cdaa0d3.
- Moves `_ASGIExceptionShield` (built with the lifecycle's `is_shutdown_initiated`), the JSON error handler,
  `_build_uvicorn_config` and `_claim_listener`. (The dispatch-stack installers left main.py in β0.)
- Signal: the fused-memory suite green; live unit active and serving both ports.
- Deletes: the HTTP partition from main.py. G7: INV-5.

**η — fused-memory collaborators via constructors; one `TaskInterceptor` builder in `server/composition.py`** (priority medium; ~800 LOC)
- Files: `fused-memory/src/fused_memory/server/composition.py` (new), `fused-memory/src/fused_memory/server/main.py`, `fused-memory/src/fused_memory/recon_state/journal.py`, `shared/tests/loop_blocking_allowlist.py`, `shared/tests/test_usage_gate_shutdown_pairing.py`, `fused-memory/tests/test_gather_convention_guard.py`, `fused-memory/tests/test_composition.py` (new), `fused-memory/tests/test_journal.py`
- Deps: θ2, S12 κ1, 5462
- Keys: fk-db357b64fcae.
- `build_reconciliation(...) -> ReconciliationComponents | None`, `build_task_interceptor(..., reconciliation=None)`
  — η's builder passes S12 C6's `InterceptorCollaborators(reconciler=…)` provider; the interceptor resolves it
  with itself as the write gate; S12 κ1 deletes both `targeted.` back-wires — and the store builders (8804's
  partition 3) in `composition.py`; `ReconciliationJournal(…, write_journal=…)`; `run_server` composes and serves.
  `test_journal.py` (2 `set_write_journal` calls) moves to the constructor keyword; the other test files that
  construct the journal keep working because the keyword defaults to `None`. Governance files that key on the
  moved code: the five `run_server` rows of `shared/tests/loop_blocking_allowlist.py` (two `accepted` for the
  `configured_known_projects_map` and `build_topic_cluster_store` reaches — task 6158 re-keyed the first on
  2026-11-02 — three `filed`) are re-keyed to the builders' qualnames, never re-blessed; the module that
  constructs the curator `UsageGate` also references its `.shutdown` (the shutdown-plan closer binding lives with
  the builder), or `test_usage_gate_shutdown_pairing.py::TestGateSelfIntegrity::test_the_teardown_detector_fires_on_real_code`'s
  positive control is repointed to that module; `test_gather_convention_guard.py`'s
  `('fused_memory/server/main.py', 'run_server')` blessing follows the two-server gather if it moves.
- Parking: η → S12 κ1, which is not behind 8528 or 5040 (re-measured 2026-11-12 through 9280, 9282 and 8144).
- Signal: row 20 green; the startup log shows one interceptor construction on the live unit.
- Deletes: `recon_journal.set_write_journal`, the two interceptor arms, the 2 `in locals()` probes.
  `targeted.task_interceptor =` and `targeted.planned_episode_registry =` go in S12 κ1 (C6);
  `memory_service.set_event_buffer` is the one wiring that stays (S26 C3, D4). G7: INV-8 (rows re-keyed),
  INV-10, INV-11.

**σ — fused-memory add_memory and add_system_record replies carry a typed per-store error map** (priority medium; ~200 LOC)
- Files: `fused-memory/src/fused_memory/models/memory.py`, `fused-memory/src/fused_memory/services/memory_service.py`, `fused-memory/tests/test_add_memory_response.py`, `fused-memory/tests/test_memory_service.py`
- Deps: 9252 (S26's last `services/memory_service.py` task; its closure holds 9147, measured), 9147 (named for
  legibility)
- Keys: fk-05a87fd1777d (closes; stamped onto 9147 too).
- Produces C7 (D15). `AddMemoryResponse.store_errors: dict[SourceStore, str]`, empty by default, with a model
  validator for the disjointness invariant; `MemoryService.add_memory` fills it for a Graphiti enqueue failure, a
  Mem0 exception and the Mem0 empty result; `MemoryService.add_system_record` for its Mem0 exception or empty
  result; a leg queued for retry stays out of it; 9147's renderer reads the field. Journalling the whole map in
  `result_summary` (today `write_ops.error` keeps only the first failing store's error) is the leaf's option.
- Signal: row 22 green; an `add_memory` against a dev server whose Mem0 backend is unreachable answers
  `store_errors: {"mem0": "…"}` with `mem0` in no store list.
- Deletes: the prose-only error channel (the error text becomes a rendering of the field). G7: INV-2 (a failure is
  a structured field, not prose), INV-9.

**ω — fused-memory MCP server boundary integration gate** (priority medium; ~400 LOC, tests only)
- Files: `fused-memory/tests/test_mcp_server_boundary.py` (new)
- Deps: ι, γ, η, θ2, λ, κ, σ
- Keys: none (integration gate; the signal leaf).
- Runs boundary rows 1, 4, 5, 12, 14, 17 and 22 through `mcp._tool_manager.call_tool` on a server composed by η's
  builders.
- Signal: green in CI; after the final deploy, a live hyphen-spelled `project_id` on a write tool, a
  `get_status(panels=…)` call and the unchanged-count `tools/list` all behave as the rows state.

**Deletable accounting (synthesis `deletable`; here "→" names the deleting leaf, not a dependency).** Prologue calls and narrowings → β (α0 offloads the 23 inline
resolver calls first). 8 `type: ignore[union-attr]` and the not-configured spellings → α, λ. 4 read-journal
copies → α. Post-construction wirings (13 measured) and 2 `locals()` probes → S26 γ 9158 (four `MemoryService`
setters and `.taskmaster =`; `set_event_buffer` stays under S26 C3), S10 μ2 9280 (four interceptor wirings and the
two `_wire_*` helpers), S12 κ1 (the two `targeted.` back-wires), η (the journal setter, the two interceptor arms,
the probes). The false 70 s comment, the two 45 s figures and the hand-counted test → ζ. The 4 data-dir
derivations → task 5462 (fk-74517af4aeed), η depends on it.

## Mechanical tasks (contract §8)

The report's `## Mechanical filings` table holds no S15 key. Subsumed mechanical or follow-up work (no
dependency edges): 4469 (fk-358e8dfe8d02) is absorbed by δ; 5511 and 6235 by ι; 4044 by α (lead `cancel` actions,
reasons naming the leaf). Narrowed,
not cancelled: 5071 (keeps its non-MCP-handler sites), 5717 (keeps its caller audit), 6295 (keeps the recon
prompts), 8804 (keeps step (a)) — lead `amend` actions. Amended: 9147 (its renderer keeps each failed store's
error text; lead amendment from the S15-4 ruling). Outside this plan, filed by the lead on 2026-11-12: task 9493
(one stop signal), which ζ depends on.

## Out of scope

- The recon-report server's three tools (`server/recon_report.py`, port 8003): separate FastMCP, no project scope
  (β0 moves only its guard installer).
- `consolidate_memories`' body (951 lines, heuristic 4) beyond the move; `MemoryService.add_memory`'s
  service-level checks (S26).
- Tool-level UUID-shape validation of id arguments (task 3144) — a candidate declaration kind later, not here.
- The `runs.db` events read model (R36; S22).
- The duplicate `event_buffer`/`burst_state` DDL in the journal module (S12 or S19).
- The double stop signal and the unit's kill mode (task 9493, outside this plan).
- fused-memory's startup time, now the bulk of a deploy's wait (median 38 s, measured 2026-11-12); no task owns it.

## Rulings (sitting of 2026-11-12)

- **OQ-1** (S15-1): option D — independent closes run concurrently in declared phases, budget = the sum of each
  phase's largest timeout (computed: 50 s + 5 s headroom = 55 s); `TimeoutStopSec` stays 90; the three
  run_server closes join the phases; ζ lands after task 9493 and sizes the drain from measured tool-call
  durations inside the freed room (D10, D11, ζ). Ruled by Leo (the recommendation was raising `TimeoutStopSec` to
  ≥ 123 s); the split of the double-signal fix into task 9493 is ruled by the lead (residual R-S15-a = B).
  Refines R14's "(name, closer, timeout) tuple summed by the budget test" into phases summed per phase.
- **OQ-2** (S15-2): as recommended — `create_mcp_server` installs the whole guard stack through one installer
  (D2, β0). Ruled by Leo.
- **OQ-3** (S15-3): as recommended — the six ungated graph mutators gain the known-project refusal; the behaviour
  change is accepted (D5). Ruled by Leo.
- **OQ-4**: as proposed — module names and placement; the program doc's R14 seam row for the shutdown tuple
  becomes `config/shutdown_plan.py` (D7, D10). Ruled by the lead; amends R14's seam row "shutdown sequence tuple
  (main.py owns)".
- **OQ-5** (S15-4): option B — task σ gives the replies a typed per-store error map after 9252; the acceptance
  record is dropped; 9147 is amended to keep the error text meanwhile (D15, σ). Ruled by Leo (the recommendation
  was an acceptance record); completes R14's "AddMemoryResponse carries per-store outcomes".
- **OQ-6**: identity stays a per-tool call to the resolver; the answer is recorded on task 6721 (D6). Ruled by
  the lead; answers S14's seam row expecting the dispatch wrapper to parse identity.
- **OQ-7**: `get_status(panels=…)` with wire models in `shared/dashboard_panels.py`, no SQL, table or index names
  (D14). Ruled by the lead, flagged as a reading of R19.
- **OQ-8**: κ goes first on both tracks, right after 9280 and S19 α. Ruled by the lead.
- **OQ-9** (S15-5): as recommended (a) — the reconciler through S12 C6's provider member; η → S12 κ1 (D12). Ruled
  by Leo; one ruling with S12/OQ-2.
- **OQ-10** (S15-6): as recommended — every declared scope validates, raw included; the five tools are a listed
  behaviour change (D3, D4). Ruled by Leo.
- **OQ-11**: (b), as S12/OQ-11 (a1) — ε1 and δ wait on S12 δ; S12 δ adds the header rule and `ctx` at the
  helpers' new home before the moves; δ reads the causation header through S12 δ's resolver (D9). Ruled by the
  lead, changing this plan's recommendation (a); the S12–S13 cycle that made (b) unwirable is cut by the lead's
  ruling S13/OQ-6 (a).

Cross-stream consequences applied: Leo's S12-2 ruling (a differing `agent_id` or `_causation_id` is refused by
S12 δ's resolver) shapes D9 and C5, and the lead's reading of it (the refusal also covers `get_entity`,
`get_entity_by_uuid` and `get_episodes`; `search`'s `agent_id` filter is excluded) is carried verbatim by the ε
moves (D7); the S24 row asking for a longer stop timeout is withdrawn; Leo's two standing rules are checked in
§Decision partition and eval hermeticity (`one-decision-partition-per-area`, INV-23).

## Open questions (tactical)

1. **Snapshot format for `tools/list` (ε1).** JSON of `[{name, inputSchema, description}]` sorted by name.
   Decide in ε1.
2. **`PanelUnavailable.reason` vocabulary (κ).** Free text naming the store and exception class, or an enum.
   Suggested: store name + exception class; decide in κ.
3. **Which of the 26 `project_root` tools need `resolve_project_id` in the body after β.** Measured in β; the
   body keeps what it uses.
4. **Tests that pass undeclared arguments through `create_mcp_server` (β0).** 4867's guard is on main; β0
   measures them at its base and adds them to its declared files at decompose.
5. **ζ's final phase table.** C3's draft is computed from today's timeouts and stated orderings; ζ records any
   added `after` edge with its reason, and the module recomputes the budget and the drain room.
