# Capability manifest — S15 fused-memory MCP server boundary

PRD: `plans/review-all-s15-fused-memory-mcp-server-boundary-prd.md`. YAML twin:
`plans/review-all-s15-fused-memory-mcp-server-boundary-prd.capability-manifest.yaml` (every `task_id: null`
until decompose; signal leaf ω). Rulings of the sitting of 2026-11-12 folded: OQ-1 option D (ζ), OQ-5 option B
(new leaf σ; the acceptance record is dropped), OQ-11 (b) (ε1 and δ wait on S12 δ); the rest as recommended or as
the lead ruled.

**Code anchors** verified against main `7848173a7d` (2026-11-12). Bindings cite symbols, not lines. Every
mechanical check below was measured at `7848173a7d`: producer checks fail today, `*-substrate` checks pass today
(66 mechanical checks, 0 polarity violations; 2 manual OPEN rows; re-run by the fold-verifier). The five `*-substrate` rows are manifest rows
only: decompose does not copy them into a task's `metadata.delivered_checks`, where
`shared/delivered_check_polarity.py::lint_delivered_checks` rejects a check already green as `vacuous_present`.

## κ — Dashboard panel read model served by `get_status(panels=…)` (C4)

| Capability | Binding | Verdict |
|---|---|---|
| Dashboard already calls fused-memory `get_status` over MCP | `dashboard/src/dashboard/data/metrics.py::_split_status` reads its reply | PASS |
| Panel tables have one owning module each | DDL in the journal module (runs, watermarks, judge_verdicts), the event-buffer module (event_buffer, burst_state; a DDL copy also sits in the journal module), `services/write_journal.py` (write_ops), `services/read_rollup.py` (read_op_counts), `middleware/ticket_store.py` (tickets); the journal and event-buffer modules are `recon_state/…` after S19 α (κ depends on it) | PASS |
| Phantom-verdict rule importable by both packages | substrate: `shared/src/shared/phantom_verdict.py::is_phantom_verdict_row` (check `phantom-verdict-substrate`) | PASS |
| Reply models S22 parses | producer: `shared/src/shared/dashboard_panels.py::PanelsReply` (check `panels-reply-models`) | PASS (κ delivers) |
| Reader composing the owners' reads | producer: `server/dashboard_panels.py::DashboardPanelReader` (check `panel-reader`) | PASS (κ delivers) |
| Burst cooldown applied by its owner | producer: `EventBuffer.burst_state_rows` computes `effective_state` from `burst_cooldown_seconds` (check `burst-effective-state-owner`); today the dashboard restates it as `_DEFAULT_BURST_COOLDOWN = 150` | PASS (κ delivers) |

## α0 — Project-root normalization takes an injected checkout resolver and runs off the event loop

| Capability | Binding | Verdict |
|---|---|---|
| Production resolver | substrate: `models/scope.py::resolve_main_checkout` (check `resolve-main-checkout-substrate`) | PASS |
| Offload idiom in the tree | `await asyncio.to_thread(_normalize_project_root, project_root)` in `declare_defect`, `record_defect_outcome`, `lookup_defect_ref` (3 sites) | PASS |
| Injection seam | producer: `checkout_resolver` keyword of `create_mcp_server` (check `checkout-resolver-seam`) | PASS (α0 delivers) |
| No inline resolution on the loop (INV-8) | anti-inversion: 23 inline `_normalize_project_root(project_root)` today; their 23 `to_file` allowlist rows are deleted as fixed (check `no-inline-checkout-resolution`) | PASS (α0 delivers) |
| No private patch of the resolver | anti-inversion: 32 sites in 24 test files today (check `no-private-resolver-patch`) | PASS (α0 delivers) |

## α — `ToolContext` (C1)

| Capability | Binding | Verdict |
|---|---|---|
| `McpErrorType` / `error_envelope` to extend | `shared/src/shared/mcp_envelope.py` exists; `McpErrorType` arrives with S14 ξ 6719 (in 9280's closure) | PASS |
| Identity helpers moved verbatim | `server/tools.py::_resolve_identity`, `::_extract_causation` keep today's signatures; S12 δ adds its header rule and `ctx` at the new home afterwards (S12's edge S12 δ → α), before ε1 (OQ-11 ruled (b)); the refusal covers writes and `get_entity`, `get_entity_by_uuid`, `get_episodes`, never `search`'s `agent_id` filter, and ε1–ε5 carry both verbatim (lead ruling on S12-2) | PASS |
| Per-server triage counter | substrate: `server/write_triage.py::TriageFailOpenCounter` (check `triage-counter-substrate`) | PASS |
| `ToolContext`, `journaled_read`, `NOT_CONFIGURED` | producer checks `tool-context`, `journaled-read-decorator`, `not-configured-code` | PASS (α delivers) |
| One interceptor binding | anti-inversion check `no-interceptor-rebinding` | PASS (α delivers) |

## β0 — One tool-call guard stack, installed by `create_mcp_server`

| Capability | Binding | Verdict |
|---|---|---|
| One-chain module | task 4643 creates `server/boundary_guards.py::install_boundary_guards` (β0 depends on 4643) | PASS |
| Unknown-argument guard on main | task 4867 (done, `daff96ef3c`): `server/unknown_argument_guard.py::install_unknown_argument_guard`, through `server/tool_middleware_bridge.py::install_tool_middleware` | PASS |
| Stack installed by the server builder | producer check `stack-installed-by-create-mcp-server` | PASS (β0 delivers) |
| No second installer | anti-inversion check `no-second-installer` (`server/main.py::_install_tool_dispatch_guards` today) | PASS (β0 delivers) |

## β — Scope declared per tool, parsed by the dispatch scope gate (C2)

| Capability | Binding | Verdict |
|---|---|---|
| Dispatch chokepoint and its result conversion | `mcp` 1.27.0: `ToolManager.call_tool(name, arguments, context, convert_result)`, `ToolManager.list_tools()`, `Tool.parameters`, `Tool.fn_metadata.convert_result`; `call_tool` runs before pydantic argument validation | PASS |
| Every tool returns a dict (refusal by return is safe) | AST: 66 tools `-> dict[str, Any]`, 1 `-> dict[str, str]` | PASS |
| One canonicalizer | substrate: `utils/validation.py::canonicalize_project_id` (check `canonicalize-substrate`) | PASS |
| Registry gate | `utils/validation.py::validate_known_project_id` | PASS |
| Gate, registration error, validation flag, offloaded resolver | producer checks `scope-gate`, `undeclared-is-registration-error`, `scope-validates`, `scope-gate-offloads-resolver` | PASS (β delivers) |
| Prologues gone | anti-inversion checks `no-per-body-canonicalize`, `no-root-narrowing` | PASS (β delivers) |

## ε1–ε5 — Tool families moved out of `create_mcp_server` (C6)

| Capability | Binding | Verdict |
|---|---|---|
| Families separable without family-to-family imports | AST closure graph: every shared closure is a C1 member, the C2 gate, or `reject_ticket_id` / `batched_task_statuses` (leaf modules) | PASS |
| Override and park-eviction store outside tools.py | S14 θ 6718 (in 9280's closure) | PASS |
| Allowlist rows that move with tools | `create_mcp_server.request_park_eviction` (ε1) and `create_mcp_server.submit_task` (ε2) in `shared/tests/loop_blocking_allowlist.py`, re-keyed by the moving leaf | PASS |
| Family modules, snapshot, registrars | producer path/grep checks per leaf (`ops-family-module` … `memory-curation-family-module`, `tool-surface-snapshot`, `family-registrars`) | PASS (ε delivers) |
| tools.py holds no tool body | anti-inversion check `curation-moved` | PASS (ε5 delivers) |

## δ — Gate protocol, control-key registry, `submit_task` gate list (C5)

| Capability | Binding | Verdict |
|---|---|---|
| Owning constant for the markup override key | `shared.toolcall_markup.MARKUP_OVERRIDE_KEY` | PASS |
| Typed caller identity for the recon-stage source | S14 μ 6721 `AgentIdentity` (in 9280's closure) | PASS |
| Header causation id | a `header_causation_id` parameter fed by S12 δ's resolver (OQ-11 ruled (b): δ → S12 δ), which reads S12 C3's `CAUSATION_ID_HEADER` and refuses a differing `_causation_id` (Leo's S12-2 ruling) | PASS |
| Registry, protocol, gate list | producer checks `control-key-registry`, `write-gate-protocol`, `submit-task-gate-list` | PASS (δ delivers) |
| No per-tool strip | anti-inversion check `no-per-tool-markup-strip` (over `server/tools.py` and `server/tool_families/`, so `write_controls.py` may call the owning stripper) | PASS (δ delivers) |

## γ — `add_memory` gate list; declared sibling gate sets

| Capability | Binding | Verdict |
|---|---|---|
| Gate protocol | δ's `WriteGate` / `run_gates` (γ depends on δ) | PASS |
| Gate list and entry table | producer checks `add-memory-gate-list`, `declared-entry-gate-sets` | PASS (γ delivers) |

## ι — Tool descriptions carry the agent-facing contract

| Capability | Binding | Verdict |
|---|---|---|
| Descriptions are served as `fn.__doc__` | bundled FastMCP `Tool.from_function` uses the docstring when no `description=` is passed; no tool passes one | PASS |
| Reviewer prose gone; rubric enforced | checks `no-reviewer-prose-in-descriptions`, `description-rubric-test` | PASS (ι delivers) |

## λ — `TaskInterceptor` raises the declared not-configured error

| Capability | Binding | Verdict |
|---|---|---|
| Last wave-2 editor of `task_interceptor.py` | S10 ν 9282 (λ depends on it) | PASS |
| Typed exception and no bare `RuntimeError` | checks `typed-not-configured-raise`, `no-runtime-error-spelling` | PASS (λ delivers) |

## ζ — Declared shutdown phases (C3); drain sized from measured tool-call durations

| Capability | Binding | Verdict |
|---|---|---|
| In-tree precedent for a summed (label, closer, timeout) tuple | substrate: `services/memory_service.py::MemoryService.close_budget_secs` (check `bounded-closes-precedent-substrate`) | PASS |
| Guards to keep | `tests/test_server_shutdown.py::TestShutdownBudgetArithmetic::test_systemd_timeout_stop_secs_matches_unit_file` and `::test_memory_close_step_budget_dominates_bounded_close` | PASS |
| One stop signal, so a drain takes effect; tool-call durations recorded | task 9493 (filed 2026-11-12 outside this plan, high; done, merged `1d0ae21e44`, read 2026-11-13; ζ → 9493, satisfied) | PASS |
| Phases, pinned order, coalesced loop step, derivation | producer checks `declared-shutdown-phases`, `phase-order-pinned-by-test`, `loop-cancels-one-step`; anti-inversion checks `no-schema-mirror-literals`, `no-main-py-budget-copies` (`server/main.py::_FORCE_EXIT_BUDGET` and `::_SYSTEMD_TIMEOUT_STOP_SECS` today), `no-hand-counted-step-sum` | PASS (ζ delivers) |
| Budget check covers the drain | producer check `drain-in-budget-check` (`ServerConfig`'s validator reads `PRE_ARM_BOUND_SECS` from `config/shutdown_plan.py`) | PASS (ζ delivers) |
| Fourth copy of the stop timeout retired | anti-inversion check `parity-script-reads-template` (`scripts/check_fused_memory_unit_parity.py::REQUIRED_SERVICE_DIRECTIVES` holds `"TimeoutStopSec=90"` today; `TimeoutStopSec` stays 90 in the template) | PASS (ζ delivers) |
| A running tool call survives a deploy | manual: with 9493 and ζ deployed, a live restart during a slow `submit_task` returns its answer; every phase logged; stop inside 90 s (`live-restart-returns-inflight-answer`) | OPEN |

## θ1, θ2 — Process lifecycle object; HTTP boundary module

| Capability | Binding | Verdict |
|---|---|---|
| Partition measurement to follow | task 8804's measured partition of `server/main.py` (θ1 depends on 8804) | PASS |
| Lifecycle object; no test global rebinding | checks `process-lifecycle-object`, `no-test-global-rebinding`, `no-watchdog-global-patching` | PASS (θ1 delivers) |
| HTTP boundary moved | checks `http-boundary-module`, `shield-moved` | PASS (θ2 delivers) |

## η — Constructor wiring and one interceptor builder

| Capability | Binding | Verdict |
|---|---|---|
| `MemoryService` constructed with collaborators; `set_event_buffer` the one pre-start setter | S26 γ 9158 (S26 C3, D4) | PASS |
| `TaskInterceptor` collaborators bundle | S10 μ2 9280 (`InterceptorCollaborators`) | PASS |
| Reconciler built by the interceptor | S12 κ1 (C6): η's builder passes S12 C6's `InterceptorCollaborators(reconciler=…)` provider; the interceptor resolves it with itself as the write gate; S12 κ1 deletes both `targeted.` back-wires (η depends on S12 κ1; OQ-9 ruled (a)) | PASS |
| One builder; factory passed; no `locals()` probes; no journal setter | checks `one-interceptor-builder`, `builder-supplies-reconciler-provider`, `no-locals-probes`, `no-journal-setter-wiring` | PASS (η delivers) |

## σ — Typed per-store error map on the add_memory / add_system_record reply (C7)

| Capability | Binding | Verdict |
|---|---|---|
| The model and its producers | `models/memory.py::AddMemoryResponse` (S26 C2; 9147 adds `stores_queued`); `services/memory_service.py::MemoryService.add_memory` and `::add_system_record` build the error text today | PASS |
| Last editor of `services/memory_service.py` on S26's chain | S26 φ 9252, whose closure holds 9147 (σ depends on both) | PASS |
| Field and producers | producer checks `store-errors-field`, `store-errors-producers` | PASS (σ delivers) |

## ω — fused-memory MCP server boundary integration gate (signal leaf)

| Capability | Binding | Verdict |
|---|---|---|
| Boundary test module | check `boundary-test-module` (rows 1, 4, 5, 12, 14, 17, 22; ω depends on σ too) | PASS (ω delivers) |
| Live behaviour after the final deploy | manual: three live calls against the deployed unit (`live-boundary-behaviour`) | OPEN |
