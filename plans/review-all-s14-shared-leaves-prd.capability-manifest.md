# Capability Manifest — review-all-s14-shared-leaves-prd

Mechanizes G3 (substrate exists and is wired) and G6 (premise valid) per leaf of
`plans/review-all-s14-shared-leaves-prd.md`. One block per leaf; each capability its signal asserts is
bound to on-main evidence or to an upstream producer leaf. Any **FAIL** blocks queueing; **OPEN**
bindings are cross-stream producer ids resolved at decompose (see the end of this file).

**Evidence** verified against main `ae3047ec3a` (2026-10-10), cited by symbol. **Domain flags:**
tooling/infra — grammar-fixture and numeric-floor checks N/A. Live checks: capability→producer
(wired), DAG-direction, field-population, rejection-mechanism. Machine-readable twin:
`plans/review-all-s14-shared-leaves-prd.capability-manifest.yaml`.

---

## α — Shared layout, project-id and merge-subject leaves *(intermediate → β γ δ ε η θ κ ρ; S12 S13 S19 S22 S24 S25)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `shared` is a package every consumer already imports | capability→producer (wired) | `shared.task_claimant` imported by dashboard `data/tasks.py`, fused-memory `reconciliation/targeted.py`, orchestrator `harness.py` | PASS |
| Current store spellings to centralise (the values the functions return) | field-population | `harness.py` runs.db join; `sqlite_task_backend` tasks.db join; `EscalationConfig.queue_dir` default `'data/escalations'`; `scheduler.py` scheduler_state.json join; `config.py` override/park path methods | PASS |
| The canonical project-id rule and the fold to move | capability→producer | `fused_memory.utils.validation::_to_underscore_canonical`; `orchestrator.session_registry::normalize_project_token` + `PROJECT_TOKEN_ALIASES` | PASS |
| The writer the merge-subject pair must reproduce | capability→producer | `orchestrator.git_ops::_merge_subject` (`f'Merge {branch} into {main_branch}'`); `_merge_marker_pattern` anchored derivation is the parse precedent | PASS |
| Worktree-aware main-checkout derivation is stdlib-reachable | rejection-mechanism | a linked worktree's `.git` is a `gitdir:` file pointing under `<main>/.git/worktrees/`; α's test builds both layouts and asserts the main root, and that a non-checkout location raises | PASS (built+bound by α) |

## β — Orchestrator adopts layout and project id *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `shared.project_layout` / `shared.project_id` | DAG-direction | α upstream | PASS |
| A config validator can refuse a field value at load | capability→producer (wired) | `orchestrator.config` already imports pydantic `field_validator`/`model_validator` | PASS |
| Every live yaml's `queue_dir` equals the layout value (refusal changes nothing live) | G6 premise | 10 yamls under `/home/leo/src/*/dark-factory-orchestrator.yaml` set `queue_dir: "data/escalations"`; none differs | PASS |
| Every live yaml declares `fused_memory.project_id` (derived default changes nothing live) | G6 premise | 18/18 yamls declare it; the `reify-*` clones declare `reify` | PASS |

## γ — fused-memory adopts layout and project id *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `shared.project_layout` / `shared.project_id` | DAG-direction | α upstream | PASS |
| fused-memory escalators file through a queue dir it builds itself | capability→producer (wired) | `_QUEUE_DIRNAME` in `middleware/curator_escalator.py`, `_folded_escalation.py` and five siblings; `EscalationQueue(Path(project_root) / _QUEUE_DIRNAME)` | PASS |
| `resolve_project_id` / `known_project_roots_from_env` keep their call contracts | capability→producer | `fused_memory.models.scope` defines both; γ delegates bodies, signatures unchanged | PASS |

## δ — Merge-subject readers derive from the shared pair *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `shared.merge_subject` | DAG-direction | α upstream | PASS |
| All reader grammars named in the key exist to replace | capability→producer | `git_ops::DEFAULT_COMMIT_CITATION_PATTERN`, `merge_lane/gates::ALREADY_LANDED_CITATION_PATTERN`, `services/landed_on_main::_merge_subject`, `evals/task_sampler::_MERGE_SUBJECT_RE`, `evals/reviewer_trial/mining` `--grep`, `warm-lane/lib_task_citation.sh::task_citation_message_cites` | PASS |
| A merge written with a non-default prefix is found by `find_merge_marker` | capability→producer (wired) | `git_ops::GitOps.find_merge_marker` uses `_merge_marker_pattern(main_branch)` derived from `_merge_subject` | PASS |
| The bash library already takes a prefix argument | capability→producer | `task_citation_message_cites msg id prefix_re` | PASS |

## ε — Scripts adopt the shared store locators *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `shared.project_layout` | DAG-direction | α upstream | PASS |
| Scripts bind this checkout's `shared` before import | capability→producer | **S25 `scripts/_checkout.py` bootstrap leaf** (fk-638eb09e52b1) — 13 of the 14 scripts sampled import no `shared` today | OPEN (cross-stream id at decompose) |
| `AuditCoverage` can carry an absent-store row | field-population | `scripts/audit_wiped_metadata_files.py::AuditCoverage` (NamedTuple, 6 fields); `audit_project` has the `runs_db.exists()` branch that today degrades silently | PASS (ε adds the field and populates it on that branch) |

## ζ — One MCP session client and write-success predicate *(intermediate → ι κ ξ; S21 S22 S25)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| The client to move exists with retry, idempotency, handshake | capability→producer (wired) | `orchestrator.mcp_lifecycle::McpSession` (`_raw_call`, `_retry_backoffs`, `maybe_inject_client_op_id`), used by `mcp_call` | PASS |
| Its dependencies are already in `shared` or injectable | capability→producer | `shared.mcp_post` (`MCP_POST_HEADERS`, `mcp_endpoint_url`, `decode_mcp_response_body`), `shared.mcp_idempotency::maybe_inject_client_op_id`; `orchestrator.fm_retry::fm_retry_backoffs` and `ORCHESTRATOR_MCP_IDENTITY` are passed in | PASS |
| The success predicate to unify | capability→producer | `task_interceptor::interceptor_write_succeeded`; `migrate_metadata_modules_to_files::write_failure_reason` | PASS |
| A synchronous client for scripts (lead ruling 2026-10-10, S21 extent): initialize handshake capturing the server-minted `mcp-session-id` for the stateful escalation servers and none for stateless fused-memory, POST `/mcp` with no trailing slash, SSE decode, session release, `isError` surfaced, a neutral error type | capability→producer | built by ζ as `SyncMcpSession` beside `McpSession`; S21's alarm poster consumes it | PASS (built+bound by ζ) |
| An `isError` reply is rejected rather than returned as a value | rejection-mechanism | built by ζ (`EnvelopeShape.IS_ERROR`); B9's stub-server fixture observes it fire | PASS (built+bound by ζ) |

## η — Typed scheduler state snapshot *(leaf; seam for S17)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `shared.project_layout.scheduler_state_json` | DAG-direction | α upstream | PASS |
| Producer emits the 13 keys the model declares | field-population | `Scheduler.get_state_snapshot` return dict (skip_counts … requeue_cooldowns, snapshot_at) | PASS |
| The MCP read path to observe through | capability→producer (wired) | `server/tools.py` `get_scheduler_state` → `mcp_tools/scheduler_state.py::read_scheduler_state` | PASS |
| `pydantic` available in `shared` | capability→producer | `shared/pyproject.toml` `pydantic>=2.7` | PASS |

## θ — One scheduler-override store in shared *(leaf; seam for S17)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `shared.project_layout` override/park paths | DAG-direction | α upstream; β and η upstream for the shared files | PASS |
| The store to move, with pin-order and collision logic | capability→producer | `orchestrator.overrides::OverrideStore` (`set_override`, `clear_override`, `reorder_pin_queue`, `get_pin_queue`), `PinOrderCollision`; `orchestrator.park_eviction_requests::ParkEvictionRequestStore` | PASS |
| Durability pragmas already shared for sync sqlite3 | capability→producer | `shared.sqlite_sync_base::apply_full_durability_pragmas_sync`; `shared.async_sqlite_base::CheckpointResult` | PASS |
| The MCP tools to observe B5 through | capability→producer (wired) | `server/tools.py` `set_task_priority_override`, `get_pin_queue` | PASS |
| No package but the shared store issues SQL on these files | rejection-mechanism | enforced by ρ's `CREATE TABLE … overrides` family plus θ's own test that the fused-memory tool module holds no override SQL | PASS (built by θ, guarded by ρ) |

## ξ — Declared MCP error codes *(intermediate → κ ι; leaf signal B8)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `shared.mcp_envelope` edits after ζ | DAG-direction | ζ upstream; θ upstream for `tools.py` | PASS |
| The envelope renderer and the backend exceptions | capability→producer | `server/tool_errors.py::mcp_tool_errors`; `backends/task_backend_errors.py::TaskNotFoundError`, `TaskmasterError` | PASS |
| A missing task through `get_task` yields `TaskNotFoundError` on the wire today | G6 premise | the repair script's recorded reply `{"error": "No tasks found for ID(s): 77", "error_type": "TaskNotFoundError"}` (`repair_wiped_metadata_files.py`) | PASS |

## ν — The memory child rule in one home *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `memory_metadata` is importable from `services/` without a cycle | capability→producer | `memory_metadata.py` imports only `backends.mem0_client`, `topic_slug`, `utils.validation`; `services/memory_service.py` already imports it at module scope | PASS |
| The two implementations to unify | capability→producer | `server/grouped_read.py::_parent_id_in_meta`; `services/topic_anchor.py::_is_child_shaped` | PASS |

## μ — Parsed caller identity *(leaf; seam for S12 S15)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `tools.py` and `targeted.py` edits after ν and γ | DAG-direction | ν, γ upstream | PASS |
| The classification table to preserve | capability→producer | `shared.task_transitions::derive_actor_class` and `ActorClass` | PASS |
| A recon-stage gate fires on `recon-stage-*` today (the rejection B7 extends to the alias) | rejection-mechanism | `server/tools.py` `add_memory` returns `ReconSnapshotWriteRejected` for a recon-stage `temporal_facts` count snapshot | PASS |
| The identity entry point | capability→producer (wired) | `server/tools.py::_resolve_identity` called by every tool body | PASS |

## κ — Dashboard adopts S14's leaves *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Layout, session client, error codes, task priority | DAG-direction | α, ζ, ξ, θ upstream | PASS |
| The private protocol code to delete | capability→producer | `dashboard/data/memory.py` `McpSession`, `_parse_sse_response`, `_parse_mcp_response`, `_extract_tool_result`; consumers import `mcp_tool_call` | PASS |
| An outage is rendered distinctly (not `{}`) | rejection-mechanism | built by ζ's typed shapes; κ's test drives `mcp_tool_call` against a stopped server | PASS (built+bound by κ) |

## ι — Scripts adopt the shared MCP client *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Session client + predicate + error codes | DAG-direction | ζ, ξ upstream | PASS |
| These scripts already bind `shared/src` | capability→producer | migrate already imports `shared.mcp_post.mcp_endpoint_url`; `repair_wiped_metadata_files` imports through migrate; fused-memory scripts run in the fused-memory environment | PASS |

## λ — Shared consecutive-streak primitive *(intermediate → ο π; S12 S24)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| The semantics to generalise | capability→producer | `orchestrator.streaks::StreakCounter` (bump/clear/gc/mark_escalated), `StreakRegistry.gc` | PASS |
| Latch-once fires exactly once per run | rejection-mechanism | built by λ; B10 observes a second fire refused until `success` | PASS (built+bound by λ) |

## ο — Orchestrator streaks on the primitive *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Primitive and serialized hub files | DAG-direction | λ, θ, δ upstream | PASS |
| The streak sites exist | capability→producer | `scheduler.py` `_fm_read_failure_streak`, `_delivered_check_error_streak`; `harness.py` `_session_resume_fallback_streak`, `_recovery_streak_filed_at`; `recovery_emission::_Streak`; `zero_progress_requeue::_Streak`; `git_ops` `_consecutive_exhausted`; `service_restart` `_consecutive_executor_failures`; `memory_recall` `_outage_streak`; `merge_lane/worker` `_resource_audit_violation_streak` | PASS |

## π — fused-memory and dashboard streaks on the primitive *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Primitive | DAG-direction | λ upstream | PASS |
| The streak sites exist | capability→producer | `path_scope_adjudicator` / `task_curator` `_consecutive_zero_output_timeouts`; `task_curator::_DegradedStreak`; `judge` `_consecutive_infra_failures`; `memory_service` `_referent_repair_streaks`; `link_heal_executor` `streak`; `link_adjudicator::_ShardFailureStreak`; `arm_harness/replay::ConsecutiveFailureBreaker`; dashboard `mcp_fanout` `_bypass_streaks` | PASS |

## ρ — Integration gate: the spelling guard *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Every family's migration landed before the guard runs clean | DAG-direction | β γ δ ε ζ η θ ι κ μ ξ upstream | PASS |
| census_trigger's handshake is owned by a live task | INV-12 `Debt(owner)` | **S21's census_trigger transport leaf** | OPEN (cross-stream id at decompose) |
| The matcher is proved, not assumed | rejection-mechanism | B11 seeded fixtures observed red | PASS (built+bound by ρ) |

---

## OPEN bindings resolved at decompose

- ε ← S25's `scripts/_checkout.py` bootstrap leaf: `add_dependency(ε, <S25 id>)` from the FILED table.
  If S25 has not filed when S14 decomposes, ε is filed with the binding OPEN and the lead wires it.
- ρ's census_trigger allow-list entry: owner = S21's transport leaf id from the FILED table.
