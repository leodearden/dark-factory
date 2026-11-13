# Capability Manifest — instruments-honesty-prd

Mechanizes G3 (substrate exists and is **wired**) + G6 (premise valid) per leaf of
`plans/instruments-honesty-prd.md`. One block per task; each capability bound to on-main evidence. Any
**FAIL** binding blocks queueing. Machine-readable twin: `plans/instruments-honesty-prd.capability-manifest.yaml`
(stamped by `commit_planning`).

**Domain flags:** tooling/infra — no grammar/DSL, so grammar-fixture checks **N/A**; no numeric accuracy bound
is asserted by any signal (the follow-up's figures are evidence, not thresholds), so numeric-floor checks
**N/A**. Live checks: **capability→producer (wired)**, **DAG-direction**, **field-population**,
**rejection-mechanism**, **readers-prove-their-producer** (INV-13).

**Code anchors** verified against main `a42f82cb52` (2026-11-05) by symbol; re-locate at implementation time.
Substrate evidence is the seat reports `plans/portfolio-review-2026-11-01/followups/instruments-seats/V1-created-at.md`,
`V2-orchestrator-writers.md`, `V3-escalation-velocity.md` and the critic `C2-prd-critic.md`. Every delivered-check
pattern in the sidecar had 0 matches on main at authoring (polarity: fails now, passes when the leaf lands).

---

## α — `tasks.created_at` + `created_at_source`  *(intermediate → η, θ; leaf signal on `get_task`)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Version ladder and additive-column idiom to extend | capability→producer (wired) | `fused-memory/src/fused_memory/backends/sqlite_task_backend.py::_SCHEMA_VERSION` (= 5), `::_migrate_v1_to_v2` template, `add_column_if_missing`; V1 §1 | PASS |
| One row-creating site to stamp | capability→producer (wired) | `SqliteTaskBackend.add_task` is the only INSERT into `tasks`; six callers (ticket dispatch, batch tickets, planning_mode, `recurrence_mint`, `consolidation_auto_executor`, `task_knowledge_sync`); it already reads one `now` for `pending_since` and `updated_at`; V1 §2 | PASS |
| Readers tolerate a new column; projection lists keys explicitly | capability→producer (wired) | `fused-memory/src/fused_memory/backends/task_projection.py::TASK_ROW_KEYS`; `_row_to_task` guards on `row_keys` (the `candidate_key` pattern); every direct SQL reader uses explicit column lists; V1 §3 | PASS |
| Backfill sources exist | field-population | runs.db events cover 5,747 of 8,515 df tasks; `pending_since_backfilled` marks the migration rows; blank `updated_at` rows exist and are skipped; V1 §4 | PASS |
| Migration is lazy per store, never at server start; a raw read-only connection never migrates, but a fused-memory read does: `SqliteTaskBackend._get_read_access` awaits `_get_write_access` (DDL + `_migrate`) on a cache miss, so `get_tasks` on a v5 store migrates it and serves both keys as None until backfilled (corrected by task 8558 review, 2026-11-12) | premise (G6 branch 3) | V1 §5; hence the separate backfill script, and the `'created_at' in row_keys` guard for a connection whose v6 ALTER never landed | PASS |
| Schema-mirror fixtures do not break | rejection-mechanism | `shared/src/shared/task_store_schema.py` subset guards accept an added reference column; only `fused-memory/tests/test_sqlite_task_backend_schema.py` must agree; nine `user_version == 5` asserts in `test_sqlite_task_backend.py` are updated by α; V1 §1 | PASS |

## β1 — Scheduler snapshot truth  *(intermediate → η, θ)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Force-write exists and is async | capability→producer | `orchestrator/src/orchestrator/scheduler.py::Scheduler.flush_state_snapshot` (bypasses throttle and dedup, swallows errors); zero callers on main; V2 §A | PASS |
| Four state-change sites | capability→producer (wired) | `harness.py::Harness.pause_scheduler`, `::resume_scheduler`, `::_load_persisted_scheduler_pause` (calls `scheduler.pause` directly), `::run` shutdown `finally:`; V2 §A | PASS |
| Authoritative pause row readable read-only | capability→producer (wired) | `orchestrator/src/orchestrator/run_store.py::RunStore.save_scheduler_pause/load_scheduler_pause/clear_scheduler_pause`; row deleted on resume; `pause_class` compared as a raw string; V2 §A' | PASS |
| Shared view type to extend; dashboard imports only `shared` | capability→producer (wired) | `shared/src/shared/scheduler_state.py::SchedulerStateSnapshot`; `dashboard/src/dashboard/data/scheduler.py` imports `shared.scheduler_state` and has no `fused_memory` dependency (C2 D1) | PASS |
| Test double needs an awaitable flush | rejection-mechanism | `orchestrator/tests/test_harness_pause_visibility.py` uses a `MagicMock` scheduler; β1 gives it an `AsyncMock` flush; V2 §A | PASS |

## β2 — Run-store counters  *(intermediate → η, θ)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Lifetime counter exists, persists, is reachable at the report site | capability→producer (wired) | `orchestrator/src/orchestrator/artifacts.py::TaskArtifacts.get_review_cycles_total` (per-task `.task-meta` store, no caller of the reset hook); reachable after `run()`; V2 §E, C1 | PASS |
| Current writer and its per-dispatch semantics | premise | `workflow.py` increments `metrics.review_cycles` only after a replan, skipped on `max_review_cycles` exhaustion, new object per dispatch; `run_store.py` task_results INSERT copies it; V2 §E | PASS |
| Readers that shift and are named | capability→producer (wired) | `dashboard/src/dashboard/data/performance.py` (`> 0` multi-pass, `== 0` one-pass); `orchestrator/evals/prompt_opt/canary.py::compute_window_metrics`; `orchestrator/evals/reviewer_trial/mining.py::mine_fn_candidates` (`WHERE review_cycles >= 1`); 8391 owns the one-pass predicate; V2 §E | PASS |
| Cap-pause ordering defect | premise | `harness.py` calls `_run_store.finish_run` before setting `report.paused_for_cap` / `cap_pause_duration_secs` from `usage_gate.total_pause_secs`; `run_store.py::RunStore.finish_run` writes both columns; no test pins the order; V2 §D | PASS |
| Single-row table pattern for the digest window | capability→producer (wired) | `run_store.py` `main_sweep_cold_control`, `add_column_if_missing_sync`, `_migrate_scheduler_state_ewa_value`; `harness.py::_init_digest_state` is the one init site (the test helper delegates to it); V2 §B, C2 #7 | PASS |
| EWA coupling premise | premise | `_maybe_write_digest` computes `done_in_step` over the same `[window_start, window_end]` lookback that is 24 h after any restart; V2 §B | PASS |

## γ — Merge-halt events and halt-hours datum  *(intermediate → η, θ)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `EventType` is extensible; no sweep enumerates the full set | capability→producer (wired) | `orchestrator/src/orchestrator/event_store.py::EventType`; `orchestrator/tests/test_event_store.py::test_all_event_types_valid` requires name == value; V2 §C | PASS |
| Transition sites and an existing callback shape | capability→producer (wired) | `merge_lane/halt.py::HaltState` (`halt_lane`, `unhalt_lane`, `unhalt_lanes_owned_by`, `unhalt_all_lanes`, `halt_for_wip`, `unhalt_wip`, `operator_halt`, `release_lanes_for_shutdown`; `on_resume` fires unconditionally from `unhalt_all_lanes`, hence edge detection); constructed in `merge_lane/worker.py`; V2 §C | PASS |
| Restart re-halt path to flag | capability→producer (wired) | `harness.py::_rehydrate_merge_halt` → `halt_for_wip`; V2 §C | PASS |
| Dashboard template for a per-project event datum | capability→producer (wired) | `dashboard/src/dashboard/data/merge_queue.py::merge_attempts` (`with_db`, `Datum`); `data/merge_halt.py` reads live status only today; V2 §C | PASS |
| "No halt event recorded yet" distinct from 0 h | readers-prove-their-producer (INV-13) | γ builds the state; θ row 15 verifies it on a store with no halt rows | PASS (built + bound by γ) |

## δ — Resolution provenance  *(intermediate → η, θ; depends on 8422)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Class parameter and validation exist | capability→producer (wired) | `escalation/src/escalation/server.py::resolve_issue` validates against `models.RESOLUTION_CLASSES` (`invalid_resolution_class`); `queue.py::EscalationQueue.resolve` + `_stamped_resolution_class` chokepoint; cascade members inherit; V3 §1 | PASS |
| New record field persists without migration | capability→producer (wired) | `models.py::Escalation.to_dict/from_dict` filter unknown keys; per-field round-trip tests in `escalation/tests/test_models.py`; V3 §1, §4 | PASS |
| Negative identity gate is enforceable; positive is not | premise (G6 branch 4) | identity is server-attributed only from the `X-Escalation-Identity` header (`authority.py::ROLE_LEVEL_ALLOWLIST`); header-less L2 callers keep a self-declared `resolved_by`; hence `ruled_by_human_forbidden` on executor / auto-watcher identities only; V3 §2, C2 #11 | PASS |
| Tier readers key on one function | capability→producer (wired) | `classify.py::classify_resolver_tier(resolved_by)` (string signature, three bare-string callers, kept); `shadow_ruling.py` and `dashboard/src/dashboard/data/escalation_analytics.py` `_lifespan_block` / `_workflow_block` route through the new `resolver_tier(record)`; V3 §3, C2 #10 | PASS |
| Callers to change are real modules | capability→producer (wired) | `skills/escalation-watcher/SKILL.md` (0 `resolution_class` mentions vs 19 in the L1 skill); `scripts/sitting/payloads.py::resolve_issue_payload`; `scripts/sitting/prepare_sitting.py::_cmd_resolve_answers`; C2 #9 | PASS |
| Executor identities and executor-side requirement are upstream | DAG-direction | decision-stream α (8422, in progress) owns `_HUMAN_RESOLVERS` membership and the executor requirement in `server.py`; δ depends on 8422 | PASS |

## ε — Brief roots, activity, landings  *(intermediate → η, θ; depends on 8457)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Registry-derived roots helper exists | capability→producer (wired) | `scripts/sitting/inventory.py::queue_project_roots`, used by `prepare_sitting.read_sources`; not by `return_brief.build` today; V3 §5 | PASS |
| Existing measurement envelope to keep | capability→producer (wired) | `scripts/sitting/fleet_state.py::landed(runs_db, window, *, now) -> Measurement[int]`, `::measure_projects -> Mapping[str, ProjectState]`; C2 #3 | PASS |
| Landing datum | DAG-direction | `shared/src/shared/quality_series/landings.py::landings(runs_db, repo, since_cursor)` is 8457 (pending), not on main; ε depends on it; 8457 amended to promise `since_cursor=None` ⇒ full history; C2 #4 | PASS |
| Activity signals exist | field-population | systemd `ActiveState`; runs.db `scheduler_state` row (written by `harness.py::force_halt_scheduler` for operator halts too); escalation queue timestamps; V3 §8, C2 #2 | PASS |
| Digest call is off the loop thread | loop-thread-occupancy-bounded (INV-8) | `landings()` runs a git subprocess; `harness.py::_maybe_write_digest` is a coroutine; ε wraps the call in `asyncio.to_thread`; C2 #5 | PASS |
| Drop-in fallback premise for sequencing | premise | `scripts/_task_db_scan.py::discover_project_roots` falls back to the dark-factory root alone without the drop-in, so η follows ε; C2 #1 | PASS |

## ζ — Counting rules and the data map  *(intermediate → η, θ)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Homes for each rule exist | capability→producer (wired) | `shared/src/shared/cost_store.py::_SCHEMA` account_events DDL; `event_store.py` `workflow_verify` comment block (verify-shaped rows carry duration in `data`); `dashboard/src/dashboard/data/burndown.py::BURNDOWN_SCHEMA` and the `state` discriminator; OPERATIONS.md §"Reading the verify artefact archive"; `skills/review-all/SKILL.md` Phase 0 and `references/report-format.md`; `skills/hotspot-survey/SKILL.md` refresh-by-key precedent | PASS |
| Pointed-at vocabularies exist | capability→producer (wired) | `event_store.py::EventType`; `merge_lane/disposition.py::MAIN_RED`; `metadata.spawn_context` values `main_health_auto_heal`, `main_health_fix_filed`, `offline_lane_red` (S5 of the follow-up); `classify.py` tiers | PASS |
| View predicate matches the reader's | premise | `state IS NULL OR state = 'value'` is the predicate `burndown.py` applies narrowing to `_Measurement` (legacy rows carry NULL); C2 #13 | PASS |

## η — Deploy gate  *(intermediate → θ; OPERATOR GATE, deterministic)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Host files to patch exist | capability→producer | `~/.config/systemd/user/return-brief.service.d/known-project-roots.conf` (two roots); `fused-memory.service` and `dark-factory-dashboard.service` root lists carry `/home/leo/src/autotrade` (follow-up S5 §14b) | PASS |
| Restart mechanism exists | capability→producer (wired) | `scripts/restart-all-orchestrators.sh --drain` (fleet sweep; OPERATIONS.md §"Fleet redeploy & watchdog") | PASS |
| Owner amendments are to live tasks | DAG-direction | 8253, 8391, 5353, 7858 pending; 8457 pending; amended at decompose by `update_task` append, with edges 8253→ε and 8391→β2 | PASS |

## θ — Integration gate  *(leaf; the PRD's signal leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Rows 1–21 executable | DAG-direction | all producers upstream through η; rows 3–5 run on a fixture orchestrator by construction (a live pause is not to be induced), the rest on live df and reify stores | PASS |
| Record is appended to the PRD | field-population | the gate writes `## Integration gate record` with observed postconditions; a row whose precondition did not occur in the window is recorded "not yet observed", never seeded (INV-13) | PASS |

## ι — Seven-day observation  *(leaf; `complexity='simple'`, dated milestone)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Each producer has a countable production trace | readers-prove-their-producer (INV-13) | `created_at_source='stamped'` rows; `scheduler_state.json` mtime at a real pause; `merge_halted` rows; L2 closes since deploy with a class; nightly `data/return-brief.md` §2–§5 coverage | PASS |
