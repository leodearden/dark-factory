# Capability manifest — S3 escalation server and queue boundary

PRD: `plans/review-all-s3-escalation-server-and-queue-boundary-prd.md`. YAML twin:
`plans/review-all-s3-escalation-server-and-queue-boundary-prd.capability-manifest.yaml`.

**Code anchors** verified against main `e4b333a601` (2026-11-10); the 62 automated checks were
re-run at `77693cb399` the same day, and the 61 that remain after the fold-verify were re-run at
`38a200d213`. Bindings cite symbols, not lines. Every grep polarity was
measured with `shared/src/shared/delivered_check_polarity.py::lint_delivered_checks` against each
leaf's declared files (no blocking finding; three `absent_overbroad` warnings, each cleared by an
upstream leaf: ε2's reverse-import and ζ's `harness: Any` patterns match `git_authority.py` and
`merge_lane_resolution.py`, which ε1 moves out first; λ's `please do not` matches `server.py`'s
`declare_pin`, which ζ moves into λ's files first): a producer check's `present` pattern is absent today
and its `absent` pattern is present today; a substrate row passes today and says so. The sitting's
rulings of 2026-11-10 are applied: there is no interim adapter (OQ-1), so the gate conversion and its
checks moved from α to ε2; the port record carries a `MergeState` and the one enum-to-enum table is
S6 μ's (OQ-8). The author's leaf η is gone: its work is task 5409's (PRD §Findings disposition).
Fold-verified 2026-11-10 at `38a200d213` (no plan path changed since `77693cb399`; every check re-run
there). The lead's sequencing rulings of that day are applied: task 4833 lands first and carries
`fk-66746efddd4d`, so ε2's `_terminal_retention` check is gone (a sibling-satisfied check) and the
probe's deletion is a substrate row produced by 4833.

## α — `MergeLaneQueries` port and records; `dedupe_merge_request` rename

| Capability | Binding | Verdict |
|---|---|---|
| `shared.merge_state.MergeState` carries the live/terminal partitions the port's `state` uses | `shared/src/shared/merge_state.py::MergeState`, `TERMINAL_STATES`, `EPISTEMIC_STATES`; substrate | PASS |
| The lane's internal state is already a closed enum S6 μ can map totally | `orchestrator/src/orchestrator/merge_lane/types.py::ItemLifecycleState` (10 members, measured at e4b333a601); substrate | PASS |
| The port exists at the path S6 μ edits | produced by α — `class MergeLaneQueries` in `orchestrator/src/orchestrator/merge_lane/queries.py` | PASS (α delivers) |
| The record and the error S6 μ fills and raises | produced by α — `class LiveRequestState`, `class QueryUnavailable` in `queries.py` | PASS (α delivers) |
| The rename set is closed | produced by α — `def dedupe_merge_request` present; `coalesce_or_enqueue_merge_request\(` absent from `orchestrator/src/` and `escalation/src/`; the five dedupe-sense identifiers absent from `orchestrator/src/`; `merge_deduplicated = ` present in `event_store.py` | PASS (α delivers) |

## β — `_mutate_pending`/`_mutate_terminal` under every mutator; callback delivery contract

| Capability | Binding | Verdict |
|---|---|---|
| Per-id advisory lock to build the primitive on | `escalation/src/escalation/queue.py::escalation_id_lock`; substrate | PASS |
| A shared root-only pending load to fold into the primitive | `escalation/src/escalation/queue.py::EscalationQueue._load_root_pending` (`0a0d1fbb24`); substrate | PASS |
| `asyncio.AbstractEventLoop.call_soon_threadsafe` is the hop | stdlib; the harness already uses `run_coroutine_threadsafe` in `harness.py::Harness._schedule_coro_threadsafe`; substrate | PASS |
| One primitive per record state under every mutator (D12) | produced by β — `def _mutate_pending` and `def _mutate_terminal` present; boundary rows 4–5 | PASS (β delivers) |
| Rejection: a mutation on a non-pending record writes nothing | produced by β — `not_pending` outcome, no root file resurrected (row 5) | PASS (β delivers) |
| Callback runs where the registrant asked; a closed loop is a counted drop | produced by β — `deliver_on` in `queue.py`, `deliver_on=` in `harness.py`; rows 6–7 | PASS (β delivers) |

## γ — `scan.py`, late-resolution policy into `amendments.py`, importers repointed, queue prose

| Capability | Binding | Verdict |
|---|---|---|
| The scan helpers reference nothing in `EscalationQueue` | `iter_all_escalation_paths`, `read_escalation_for_scan` import `escalation.models`, `escalation.classify`, `escalation.archive` only; substrate | PASS |
| Every external importer is known | `git grep` at `e4b333a601`: 11 production and 4 test `from escalation.queue import` lines naming a scan helper (PRD §Background); substrate | PASS |
| `amendments.py` exists to receive the late-resolution policy | `escalation/src/escalation/amendments.py` (task 6319, done `0935e63d88`); substrate | PASS |
| No re-export shim left in `queue.py` | produced by γ — `def read_escalation_for_scan` absent from `queue.py`, present in `scan.py`; `def _build_late_resolution` absent from `queue.py` | PASS (γ delivers) |
| The stale inventory is gone | produced by γ — `ALREADY GUARDED elsewhere` absent from `escalation/src` | PASS (γ delivers) |

## δ — operator tools registered from `orchestrator/mcp/ops_tools.py`; `HarnessOps`, `WorkflowSlots`

| Capability | Binding | Verdict |
|---|---|---|
| fastmcp registers tools after construction (`FastMCP.tool(fn)`, `add_tool`) | installed fastmcp 3.2.2 `fastmcp/server/server.py::FastMCP.tool` / `add_tool` (re-checked 2026-11-10); substrate | PASS |
| The public harness methods the tools call exist | `harness.py::Harness.is_workflow_active`, `cancel_workflow`, `task_runtime_snapshot`, `halt_merge_queue`, `force_unhalt_merge_queue`, `get_merge_halt_status`, `force_halt_scheduler`, `force_resume_scheduler`, `reload_config`, `clear_verify_host_quarantine` (all at `e4b333a601`); substrate | PASS |
| A git runner replaces the private `_run` | `shared/src/shared/git_async.py::run_git` (S5 owns; exists; S5 C1 assigns this repoint to S3 δ); substrate | PASS |
| Tools registered onto the escalation instance from the orchestrator | produced by δ — `def register_ops_tools`, `class HarnessOps`, `class WorkflowSlots` present in `ops_tools.py` | PASS (δ delivers) |
| `release_workflow` waits through the slot port (OQ-7, ruled (a)) | produced by δ — `await [A-Za-z_.]*wait_cleared\(` present in `ops_tools.py` (the Protocol's own `async def wait_cleared(` does not match) | PASS (δ delivers) |
| The operator tools left escalation (not satisfiable by S5) | produced by δ — `async def release_warm_worktree` absent from `escalation/src/` | PASS (δ delivers) |
| No private git runner import survives | produced by δ (its own repoint; S5 β depends on δ) — `from orchestrator.git_ops import _run` absent | PASS (δ delivers) |
| The recon stage classification stops naming tools the shared server no longer registers | produced by δ — `mcp__escalation__release_workflow` absent from `fused-memory/tests/test_stages.py` (its stale-bucket assertion fails otherwise) | PASS (δ delivers) |

## θ — per-task landing index for the archive tier; honest history envelopes

| Capability | Binding | Verdict |
|---|---|---|
| Records carry `task_id` independent of the id key | `escalation/src/escalation/models.py::Escalation.task_id`; `queue.submit`'s divergence note; substrate | PASS |
| A startup pass exists to backfill the index | `escalation/src/escalation/sweep.py::run_startup_sweep` (runs at every `create_server`); substrate | PASS |
| `get()` retries a mid-read relocation | `escalation/src/escalation/queue.py::EscalationQueue.get` (task 5118's retry); substrate | PASS |
| Measured baseline to beat | live `data/escalations` at `e4b333a601`: default `get_by_task` 316–411 ms, pending 8.4–9.7 ms; substrate (measurement) | PASS |
| The index is written and read for the archive tier | produced by θ — `by-task` present in `queue.py`; `_iter_archive_paths\('esc-\*\.json'\)` absent (present today and on `task/4567`) | PASS (θ delivers) |
| "Pruned" is a positive fact | produced by θ — `tombstone` present in `archive.py`; tombstones carry the parsed `filed_at`/`level`/`category`/`severity` | PASS (θ delivers) |
| Rejection: a pruned record with no index line is not lost to a `level=` filter | produced by θ — boundary row 14 (manual) | PASS (θ delivers) |
| "Ever filed" holds after pruning | produced by θ — `pruned_count` present under `escalation/src/escalation/` (in `server.py` until ζ moves it to `read_tools.py`); boundary row 14 | PASS (θ delivers) |
| Rejection: an index-unaware writer cannot hide a pending record | produced by θ — boundary row 12b (manual); OQ-6 ruled (a) | PASS (θ delivers) |

## ι — `RESOLVE_GATES` and `PROMOTE_GATES` as ordered gate tables

| Capability | Binding | Verdict |
|---|---|---|
| The two tools exist where ι edits them | `escalation/src/escalation/server.py::create_server.<locals>.resolve_issue` and `promote_to_l2` (ι runs before the ladder split; ζ moves them); substrate | PASS |
| The ceiling's carve-out rule to fold into one gate exists | `escalation/src/escalation/authority.py::l2_auto_close_class`; substrate | PASS |
| The two gates added since the draft exist to tabulate | `escalation/src/escalation/policy_gate.py::identity_resolve_refusal` (task 8422) and `escalation/src/escalation/models.py::recut_action_conflict` (task 5516); substrate | PASS |
| Gate tables and the refusal type exist | produced by ι — `RESOLVE_GATES`, `class Refusal` present in `escalation/src/escalation/ladder_gates.py` | PASS (ι delivers) |
| Rejection: a refusal names its gate | produced by ι — boundary row 17 (INV-2) | PASS (ι delivers) |

## κ — header names, the one header reader and the watcher identity in `escalation/models.py`

| Capability | Binding | Verdict |
|---|---|---|
| The ceiling to derive the levels value from exists | `escalation/src/escalation/authority.py::ROLE_LEVEL_ALLOWLIST`; substrate | PASS |
| The models module already holds the sentinel prefixes the constants join | `escalation/src/escalation/models.py::HARNESS_SENTINEL_ROLE_PREFIXES` (task 5224); substrate | PASS |
| The supervisor's header injection point exists | `orchestrator/src/orchestrator/mcp_lifecycle.py::mcp_config_json(escalation_headers=)`; substrate | PASS |
| fastmcp lower-cases inbound header names | fastmcp 3.2.2 `fastmcp/server/dependencies.py::get_http_headers` (`name.lower()`); substrate | PASS |
| One home, one reader | produced by κ — `IDENTITY_HEADER` and `def read_caller_headers` present in `models.py`; `_IDENTITY_HEADER` absent from `escalation/src/`; `_WATCHER_ESCALATION_HEADERS` absent from `orchestrator/src`; `_WATCHER_AUTO_IDENTITY = ` absent from `authority.py`; `IDENTITY_HEADER = ` absent from `scripts/sitting/` | PASS (κ delivers) |

## ε1 — merge read/cancel/relane tools from `orchestrator/mcp/merge_tools.py`; three modules move; filing port

| Capability | Binding | Verdict |
|---|---|---|
| `MergeLaneQueries` exists to consume | produced by α (upstream in batch) — `class MergeLaneQueries` in `merge_lane/queries.py` | PASS (α delivers) |
| A native implementation and the total state table exist | produced by S6 μ (cross-plan; ε1 depends on it — OQ-1, OQ-8) — S6's `LaneQueries` and `LIVE_STATE_OF` in `merge_lane/queries.py`, with its exhaustiveness test | PASS (S6 μ delivers) |
| The server's ring probe and `retention=` keywords are already gone | produced by task 4833 (lands first, lead ruling 2026-11-10) — `_get_terminal_retention` absent from `server.py` (4833's own check) | PASS (4833 delivers) |
| Task 4651's `escalation.git_authority` import sits where ε1 can repoint it | produced by task 4651 in S8's `orchestrator/src/orchestrator/stranded_reconcile.py` (4651 depends on S8 ε2a; ε1 depends on 4651 and on S8 ε2b) | PASS (4651 delivers) |
| The journal is an explicit collaborator the harness can hand over | `orchestrator/src/orchestrator/merge_lane/queue_store.py::MergeQueueStore`, constructed by the harness; substrate | PASS |
| The ladder's dedupe path exists to wrap | `escalation/src/escalation/dedupe.py::submit_or_dedupe_off_loop`; substrate | PASS |
| The moved modules have no escalation-only consumer | measured at `e4b333a601`: `git_authority.py` imported by `server.py`, `merge_lane_resolution.py` and tests (incl. `orchestrator/tests/test_citation_gate_positive_arm.py`); `merge_lane_resolution.py` by `server.py`; `merge_status_answer.py` by `server.py`; prose pointers in `harness.py`, `merge_lane/landing_evidence.py`, `agents/landed_sha_searches.py`, `shared/`; substrate | PASS |
| Registration entry point and filing port exist | produced by ε1 — `def register_merge_tools` in `merge_tools.py`; `async def submit_or_dedupe\(` in `escalation/filing_tools.py` | PASS (ε1 delivers) |
| The journal probe and the live-state mapping leave escalation, with no copy in the orchestrator | produced by ε1 — `_get_merge_journal`, `_map_live_state` absent from `escalation/src/` and `orchestrator/src/` | PASS (ε1 delivers) |
| The tools map through the one table, from a typed verdict (OQ-8) | produced by ε1 — `LIVE_STATE_OF` present in `merge_tools.py`; `state: ItemLifecycleState \| None` present in `merge_lane/types.py`; `test_unrecognised_raw_still_passes_through` absent from `escalation/tests/` and `orchestrator/tests/` | PASS (ε1 delivers) |
| The moved filer mints its record id through the port (N3) | produced by ε1 — `mint_escalation_id` present in `merge_tools.py` | PASS (ε1 delivers) |
| `git_authority.py` sits beside its consumer (OQ-4) | produced by ε1 — path `orchestrator/src/orchestrator/mcp/git_authority.py` present, `escalation/src/escalation/git_authority.py` absent | PASS (ε1 delivers) |
| The recon deny list stops naming a tool :8103 no longer serves | produced by ε1 — `mcp__escalation__merge_relane` absent from `fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py` | PASS (ε1 delivers) |
| The terminal tier is read from its producer (INV-13) | produced by ε1 over S6 μ — `merge_status` on a landed request answers from the native `terminal_state` (manual, boundary rows 3/9) | PASS (ε1 delivers) |

## ε2 — `merge_request` from `orchestrator/mcp/merge_tools.py`; the dedupe gate on the native port; the private probes go

| Capability | Binding | Verdict |
|---|---|---|
| `merge_tools.py` exists to receive `merge_request` | produced by ε1 (upstream in batch) | PASS (ε1 delivers) |
| The native port answers the gate | produced by S6 μ (cross-plan; ε2 depends on it) | PASS (S6 μ delivers) |
| `merge_request` carries the shared `workflow_verify` constructor | produced by task 5409 (lands first, in today's `server.py`) | PASS (5409 delivers) |
| `merge_request` has no `retry_failed_only` to carry | produced by S7's deletion task (S7-3 ruled (b); lands first) | PASS (S7 δ delivers) |
| `merge_request` registered from the orchestrator | produced by ε2 — `async def merge_request\(` present in `merge_tools.py` | PASS (ε2 delivers) |
| The gate consumes the port, not a dict | produced by ε2 — `_snapshot_verify_state` and `live_snapshot` absent from `intake.py` | PASS (ε2 delivers) |
| Rejection: an unanswerable query is visible to the caller, never a silent dedupe | produced by ε2 — `lane_unanswerable` present in `merge_lane/types.py`; event + response field asserted by boundary row 2 (manual) | PASS (ε2 delivers) |
| No private harness probe survives | produced by ε2 — `_get_merge_worker` absent from `escalation/src/` and `orchestrator/src/` | PASS (ε2 delivers) |
| Escalation imports nothing from orchestrator | produced by ε2 (pinned by ζ's guard) — `^[[:space:]]*from orchestrator[. ]` absent from `escalation/src` | PASS (ε2 delivers) |
| The briefing known gap closes with its producer (contract §7) | produced by ε2 — `duck-typed harness._merge_worker` absent from `review/briefing.yaml` | PASS (ε2 delivers) |

## ζ — ladder tools from three escalation modules; `LadderLookups`; `create_server` composition; layering guard

| Capability | Binding | Verdict |
|---|---|---|
| The fused-memory harness's call shape survives | `fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._start_escalation_server` calls `create_server(queue, store_identity=..., prefix_resolver=...)` — positional queue + keywords C7b keeps; substrate | PASS |
| The four harness touches of ladder tools are enumerable as lookups | measured: `pre_increment_routing_tier`, `granted_files_delivery_blocker`, `scheduler.get_statuses` + `is_workflow_active`, `git_ops` (PRD §Background); `claim_policy_class` adds none; substrate | PASS |
| `server.py` registers no tool itself | produced by ζ — `^[[:space:]]*@mcp\.tool\(` absent from `server.py` (29 decorators today; the middleware comment does not match) | PASS (ζ delivers) |
| The lookups bundle and the filing registrar exist | produced by ζ — `class LadderLookups` in `lookups.py`; `def register_filing_tools` in `filing_tools.py` | PASS (ζ delivers) |
| Rejection: a reverse import is caught | produced by ζ — `escalation/tests/test_layering.py` exists and fails on a seeded fixture module (INV-10) | PASS (ζ delivers) |
| `harness: Any` leaves the package | produced by ζ — `harness: Any` absent from `escalation/src/` | PASS (ζ delivers) |

## λ — tool descriptions and module prose under the Comments stance

| Capability | Binding | Verdict |
|---|---|---|
| Reviewer-addressed prose leaves the tool descriptions | produced by λ — `please do not` absent from `escalation/src` (present today in `declare_pin`) | PASS (λ delivers) |
| Agent-directed imperatives survive | before/after imperative lists diffed in the commit; manual | PASS (λ delivers) |
| Prose quality | judged by review against `docs/code-quality.md` Comments stance; manual | PASS (λ delivers) |

## μ — integration gate and companion docs

| Capability | Binding | Verdict |
|---|---|---|
| Real-HTTP server fixture to lift | `escalation/tests/conftest.py::_serve_escalation_mcp_impl` (scoped to `escalation/tests/` today); substrate | PASS |
| A fused-memory restart path outside the fleet sweep | `scripts/restart-fused-memory.sh` and the fused-memory staleness backstop in `scripts/orchestrator-watchdog.py` (`FM_WATCHED_PATHS` includes `escalation/src/`); substrate | PASS |
| Both suites can reach the fixture | produced by μ — path `escalation/src/escalation/testing_http.py` present | PASS (μ delivers) |
| The suite exists | produced by μ — `escalation/tests/test_s3_boundary.py` and `orchestrator/tests/test_s3_boundary.py` present | PASS (μ delivers) |
| Docs name the new seams | produced by μ — `coalesce_or_enqueue_merge_request` absent from `ARCHITECTURE.md` and `skills/merge-queue/references/two-layer-model.md` | PASS (μ delivers) |
| Live served shapes | :8102 lists 29 tools, :8103 lists 13, after the orchestrator restarts on the new code and fused-memory restarts through `scripts/restart-fused-memory.sh` or its staleness backstop; manual | PASS (μ delivers) |
